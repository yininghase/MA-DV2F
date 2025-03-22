import os
import copy
import glob
import torch
import numpy as np
from tqdm import tqdm
from data_process import load_yaml
from argparse import ArgumentParser


def get_angle_diff(angle_diff):
    
    angle_diff_1 = angle_diff%(2*np.pi)
    angle_diff_2 = (-angle_diff)%(2*np.pi)
    mask = torch.argmin(torch.cat((angle_diff_1[...,None],angle_diff_2[...,None]), axis=-1), axis=-1).float()
    angle_diff = angle_diff_1*(1-mask) - angle_diff_2*mask
    
    return angle_diff

def adapt_trajectory(trajectories, starts, flip_yaw=1, map_scale_factor=1):
    
    trajs = []
    
    for trajs_i in trajectories.values():
        trajs.append([[state['x'], state['y'], flip_yaw*state['yaw'], 0] for state in trajs_i])
        
    adapted_traj = [[[-i*10, -i*10, 0, 0]] for i in range(len(starts))]
    starts_tensor = torch.tensor(starts)
    
    used_veh_idx = []
    
    for i in range(len(trajs)):
        starts_i = torch.tensor(trajs[i][0])
        dist_i = torch.norm(starts_i[:3]-starts_tensor[:,:3], dim=-1, p=2)
        veh_idx = torch.argmin(dist_i).item()
        
        assert veh_idx not in used_veh_idx, "Error in matching trajectory: Multiple Matching!"
        assert dist_i[veh_idx]<0.1/map_scale_factor, "Error in matching trajectory: Large Matching Error!"
        adapted_traj[veh_idx] = trajs[i]
        used_veh_idx.append(veh_idx)
       
    return adapted_traj

def transfer_data_yaml_dict_to_tensor(trajectories, time_step, default_state):
    
    current_states = copy.deepcopy(default_state)
            
    for i in range(len(trajectories)):
        
        j = min(time_step, len(trajectories[i])-1)
        
        if j>0:
            current_states[i] = trajectories[i][j]
        
    return torch.tensor(current_states)
    

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--test_data_folder', type=str, help="yaml benchmark folder path")
    parser.add_argument('--yaml_result_folder', type=str, help="yaml results folder path")
    parser.add_argument('--torch_result_folder', type=str, help="torch results folder path")
    parser.add_argument('--map_rescale_factor', type=float, default=1, help="only for GCBF or GCBF+, the map_rescale_factor should be 1.5/0.05=30")
    parser.add_argument('--default_obstacles_radius', type=int, default=2, help="obstacle radius in CL-MAPF or CSDO")
    parser.add_argument('--flip_yaw', action='store_true', help='CL-MAPF uses left-hand system, the yaw needs to be flipped!')
    args = parser.parse_args()
    
    test_data_folder = args.test_data_folder
    yaml_result_folder = args.yaml_result_folder
    torch_result_folder = args.torch_result_folder
    map_rescale_factor = args.map_rescale_factor
    default_obstacles_radius = args.default_obstacles_radius
    flip_yaw = -1 if args.flip_yaw else 1
       
    os.makedirs(torch_result_folder, exist_ok=True)
    
    yaml_test_files = glob.glob(os.path.join(test_data_folder, '*','*','*','*.yaml'))
    
    yaml_dict = {}
    
    for yaml_file in yaml_test_files:
        
        tmp = yaml_file.split('_')
        num_vehicles = int(tmp[-2][6:])
        num_obstacles = int(tmp[-3][4:])
        key = (num_vehicles, num_obstacles)
        
        if key in yaml_dict.keys():
            yaml_dict[key].append(yaml_file)
        else:
            yaml_dict[key] = [yaml_file]
        

    for key in yaml_dict.keys():
        
        yaml_dict[key].sort(key=lambda x: int(((x.split('_')[-1]).split('.')[0])[2:]))
        
        num_vehicles, num_obstacles = key
        print(f'Currently working on {num_vehicles} vehicles and {num_obstacles} obstacles!')
        
        X_data_path = os.path.join(torch_result_folder, f"X_data_vehicle={num_vehicles}_obstacle={num_obstacles}.pt")
        y_data_path = os.path.join(torch_result_folder, f"y_model_data_vehicle={num_vehicles}_obstacle={num_obstacles}.pt")
        batches_data_path = os.path.join(torch_result_folder, f"batches_data_vehicle={num_vehicles}_obstacle={num_obstacles}.pt")
        trajectory_data_path = os.path.join(torch_result_folder, f"trajectory_data_vehicle={num_vehicles}_obstacle={num_obstacles}.pt")
        
        if os.path.exists(X_data_path) and os.path.exists(y_data_path) and os.path.exists(batches_data_path) and os.path.exists(trajectory_data_path):
            continue
            # pass
        
        X_data = []
        trajectory_data = [0]
        
        for yaml_test_file in tqdm(yaml_dict[key]):
            
            yaml_test_data = load_yaml(yaml_test_file)
            
            starts = [[] for i in range(len(yaml_test_data['agents']))]
            targets = [[] for i in range(len(yaml_test_data['agents']))]
            for i in range(len(yaml_test_data['agents'])):
                agent_i = yaml_test_data['agents'][i]
                agent_idx = int(agent_i['name'][5:])
                starts[agent_idx] = agent_i['start']+[0]
                targets[agent_idx] = agent_i['goal']+[0]
                
            
            starts = torch.tensor(starts)
            starts[...,2] = (flip_yaw*starts[...,2]+np.pi)%(2*np.pi)-np.pi
            starts = starts.tolist()
            
            targets = torch.tensor(targets)
            targets[...,2] = (flip_yaw*targets[...,2]+np.pi)%(2*np.pi)-np.pi
            targets = targets.tolist()
            
            yaml_result_file = yaml_test_file.replace(test_data_folder, yaml_result_folder)
            
            yaml_result_data = None if not os.path.exists(yaml_result_file) else load_yaml(yaml_result_file)
            
            if yaml_result_data is None or yaml_result_data['schedule'] is None: 
                
                X_data_i = torch.tensor(starts).repeat(5, 1, 1)
            
            else:
                
                trajectories = yaml_result_data['schedule']
                
                # if trajectories is not None and len(trajectories) != num_vehicles:
                #     print(f'Incomplete trajectories in file: {yaml_file}')
                    
                trajectories = adapt_trajectory(trajectories, starts, flip_yaw, map_rescale_factor)
                trajectory_length = max([len(trajectories[i]) for i in range(len(trajectories))])
                
                # default_state = starts # keep the vehicle with unknown trajectory at the start point.
                
                default_state = [trajectories[i][0] for i in range(len(trajectories))] # remove the vehicle with unknown trajectory
                
                X_data_i = [torch.tensor(default_state)[None,:,:]]
                last_state = torch.tensor(default_state)
                # _ = transfer_data_yaml_dict_to_tensor(trajectories, 0, starts)
                
                for i in range(1,trajectory_length):
                    current_state = transfer_data_yaml_dict_to_tensor(trajectories, i, default_state)
                    current_state[...,2] = (current_state[...,2]+np.pi)%(2*np.pi)-np.pi
                    
                    diff_state = current_state-last_state
                    diff_state[...,2] = get_angle_diff(diff_state[...,2])
                    weights = torch.linspace(0,1,6)
                    new_states = last_state[None,:,:] + torch.lerp(torch.zeros_like(diff_state[None,:,:]), 
                                                                   diff_state[None,:,:], weights[:,None,None])
                    new_states[...,2] = (new_states[...,2]+np.pi)%(2*np.pi)-np.pi
                    new_states = new_states[1:]
                    
                    # new_states = current_state[None,...]
                    
                    X_data_i.append(new_states)
                    last_state = current_state
                
                X_data_i= torch.cat(X_data_i, dim=0)
             
            targets = torch.tensor(targets).repeat(len(X_data_i), 1, 1)
            X_data_i= torch.cat((X_data_i, targets), dim=-1)
            
            X_data_i[...,2] = (X_data_i[...,2]+np.pi)%(2*np.pi)-np.pi
            X_data_i[...,6] = (X_data_i[...,6]+np.pi)%(2*np.pi)-np.pi
            
            if num_obstacles>0:
                obstacles = torch.tensor(yaml_test_data['map']['obstacles'])
                prepend = torch.zeros((len(obstacles), 4))
                append = torch.ones((len(obstacles), 2))
                append[:,0] *= default_obstacles_radius
                obstacles = torch.cat((prepend, obstacles, append), dim=-1)
                obstacles = obstacles.repeat(len(X_data_i), 1, 1)
                X_data_i = torch.cat((X_data_i, obstacles), dim=1)
            
            X_data_i[:,:num_vehicles,:2] *= map_rescale_factor
            X_data_i[:,:,4:6] *= map_rescale_factor
            X_data_i[:,num_vehicles:,6] *= map_rescale_factor
            
            X_data.append(X_data_i)
            trajectory_data.append(trajectory_data[-1]+len(X_data_i))
        
        
        X_data = torch.cat(X_data, dim=0).reshape(-1,8)
        y_data = torch.zeros((trajectory_data[-1]*num_vehicles,2))
        batches_data = torch.tensor([num_vehicles+num_obstacles, num_vehicles]).repeat(trajectory_data[-1],1)
        trajectory_data = torch.tensor(trajectory_data)*(num_vehicles+num_obstacles)
        
        torch.save(X_data, X_data_path)
        torch.save(y_data, y_data_path)
        torch.save(batches_data, batches_data_path)
        torch.save(trajectory_data, trajectory_data_path)
        
            