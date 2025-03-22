import os
import torch
import numpy as np
from argparse import ArgumentParser
from itertools import combinations

from data_process import load_data, load_yaml

def calculate_metrics(config): 
    
    assert os.path.exists(config["data folder"]), \
        f"The given folder of '{config['data folder']}' does not exist!"        
    
    txt_path = os.path.join(config["data folder"], 'evaluation.txt')
    with open(txt_path, 'w') as f:
        pass
    
    for num_vehicles in range(1, config["num of vehicles"]+1):
        for num_obstacles in range(config["num of obstacles"]+1):
            
            data = load_data(num_vehicles = num_vehicles, 
                             num_obstacles = num_obstacles,
                             folders = config["data folder"],
                             load_all_simpler = False,
                             horizon = 1,
                             load_trajectory=True,
                             load_model_prediction=True)

            if len(data) == 0: continue
            
            metrics= {}

            len_batch = num_vehicles+num_obstacles
            key = (len_batch, num_vehicles)
            value = data[key]
            
            states, _, _, trajectory_idx = value
            # metrics["number of trajectories"].update({key:len(trajectory_idx)-1})
            metrics["number of trajectories"] = len(trajectory_idx)-1
            
            states = states.reshape(-1, len_batch, 8)
            trajectory_idx = torch.round(trajectory_idx/len_batch).long()
            
            trajectory = states[:,:num_vehicles,:2]
            
            travel_distance = torch.norm(torch.diff(trajectory, dim=0, prepend=trajectory[:1,:,:]), p=2, dim=-1)
            travel_distance = torch.cumsum(travel_distance, dim=0)
            travel_distance = travel_distance[trajectory_idx[1:]-1]-travel_distance[trajectory_idx[:-1]]
            metrics["travel distance"] = travel_distance
            
            collisions = calculate_collision_times(states, config["car size"], num_vehicles, num_obstacles)
            
            metrics["collisions"] = collisions
            
            if key == (1,1):
                
                metrics["collision rate"] = torch.tensor(0)
            
            else:
            
                collision_rate = len(collisions)/torch.sum(travel_distance)
                metrics["collision rate"] = collision_rate
            
            success_to_goal, reach_goal, safe_vehicle = calculate_reach_goal(states[trajectory_idx[1:]-1,:num_vehicles,:], 
                                                                            collisions, trajectory_idx,
                                                                            config["position tolerance"], 
                                                                            config["angle tolerance"],
                                                                            num_vehicles, num_obstacles,
                                                                            return_states_detail=True)
            
            metrics["success to goal"] = success_to_goal
            
            success_to_goal_rate = torch.sum(success_to_goal)/(num_vehicles*len(success_to_goal))
            metrics["success to goal rate"] = success_to_goal_rate
            
            metrics["reach goal"] = reach_goal
            
            reach_goal_rate = torch.sum(reach_goal)/(num_vehicles*len(reach_goal))
            metrics["reach goal rate"] = reach_goal_rate
            
            metrics["safe vehicle"] = safe_vehicle
            
            safe_vehicle_rate = torch.sum(safe_vehicle)/(num_vehicles*len(safe_vehicle))
            metrics["safe vehicle rate"] = safe_vehicle_rate
            
            inter_state_keys = []
            
            if config['intermediate success rate'] is not None:
                if f'({num_vehicles},{num_obstacles})' in config['intermediate success rate'].keys():
                    success_rate_n_seconds = config['intermediate success rate'][f'({num_vehicles},{num_obstacles})']
                else:
                    success_rate_n_seconds = config['intermediate success rate']['default']
            else:
                success_rate_n_seconds = []
                    
            for n_seconds in success_rate_n_seconds:
                
                i = np.ceil(n_seconds/0.2).astype(int)
            
                trajectory_idx_inter = torch.min(trajectory_idx[:-1]+i, 
                                                trajectory_idx[1:]-1)
                if len(collisions)>0:
                    mask_inter = (collisions[:,0][:,None]>=trajectory_idx[:-1][None,:]) & \
                                (collisions[:,0][:,None]<=trajectory_idx_inter[None,:])
                    mask_inter = torch.any(mask_inter, dim=-1)
                    collisions_inter = collisions[mask_inter]
                else:
                    collisions_inter = torch.empty((0,3))
                
                success_inter_state = calculate_reach_goal(states[trajectory_idx_inter,:num_vehicles,:], 
                                                        collisions_inter, trajectory_idx,
                                                        config["position tolerance"], 
                                                        config["angle tolerance"],
                                                        num_vehicles, num_obstacles)
                
                success_inter_state_rate = torch.sum(success_inter_state)/(num_vehicles*len(success_inter_state))
                metrics[f"success rate {n_seconds} second(s)"] = success_inter_state_rate 
                inter_state_keys.append(f"success rate {n_seconds} second(s)")
            
            trajectory_efficiency = calculate_trajectory_efficiency(travel_distance, 
                                        states[trajectory_idx[:-1],:num_vehicles,:2], 
                                        states[trajectory_idx[:-1],:num_vehicles,4:6], 
                                        success_to_goal)
            
            metrics["trajectory efficiency"] = trajectory_efficiency
            
            print(f"number of vehicle: {num_vehicles}, number of obstacle: {num_obstacles}")
            print(f"number of trajectories: {metrics['number of trajectories']}")
            print(f"success to goal rate: {metrics['success to goal rate']}")
            print(f"collision rate: {metrics['collision rate']}")
            print(f"reach goal rate: {metrics['reach goal rate']}")
            print(f"safe vehicle rate: {metrics['safe vehicle rate']}")
            print(f"trajectory efficiency: {metrics['trajectory efficiency']}")
            for key in inter_state_keys:
                print(f"{key}: {metrics[key]}")
            print('-'*10)
            
            with open(txt_path, 'a+') as f:
                
                print(f"number of vehicle: {num_vehicles}, number of obstacle: {num_obstacles}",file=f)
                print(f"number of trajectories: {metrics['number of trajectories']}",file=f)
                print(f"success to goal rate: {metrics['success to goal rate']}",file=f)
                print(f"collision rate: {metrics['collision rate']}",file=f)
                print(f"reach goal rate: {metrics['reach goal rate']}",file=f)
                print(f"safe vehicle rate: {metrics['safe vehicle rate']}", file=f)
                print(f"trajectory efficiency: {metrics['trajectory efficiency']}",file=f)
                for key in inter_state_keys:
                    print(f"{key}: {metrics[key]}", file=f)
                print('-'*10,file=f)
    
            if config['save metrics']:
                torch.save(metrics, os.path.join(config["data folder"], f'metrics_vehicle={num_vehicles}_obstacle={num_obstacles}.pt'))
            
            del data, metrics
               

def calculate_collision_times(states, vehicle_size, num_vehicles, num_obstacles, memory_limit=1e7):
    
    collisions = []
    
    if num_vehicles > 1:
        all_comb = torch.tensor(list(combinations(range(num_vehicles), 2)))
        
        batch_size = np.ceil(memory_limit/len(all_comb)).astype(int)
        batch_num = np.ceil(len(states)/batch_size).astype(int)
        prepend = torch.zeros((1, len(all_comb))).long()
        
        for i in range(batch_num):
            idx_1 = i*batch_size
            idx_2 = min((i+1)*batch_size, len(states))
        
            vehicle_states_comb = (states[idx_1:idx_2,:num_vehicles,:].permute(1,0,2))[all_comb]
                    
            collisions_t = check_collision_rectangular_rectangular_batch(vehicle_states_comb, vehicle_size) #[T,C,2,8]
            del vehicle_states_comb
            
            T,C = torch.where(torch.diff(collisions_t, dim=0, prepend=prepend)==1)
            prepend = collisions_t[-1:]
            
            if len(T)>0:
                collisions.append(torch.cat((T[:,None]+idx_1,all_comb[C]), dim=-1))
            
            del collisions_t
                
    
    if num_obstacles > 0:
        
        batch_size = np.ceil(memory_limit/(num_obstacles*num_vehicles)).astype(int)
        batch_num = np.ceil(len(states)/batch_size).astype(int)
        prepend = torch.zeros((1, num_vehicles, num_obstacles)).long()
        
        for i in range(batch_num):
            idx_1 = i*batch_size
            idx_2 = min((i+1)*batch_size, len(states))
                
            collisions_t = check_collision_rectangular_circle_batch(states[idx_1:idx_2,:num_vehicles,:], 
                                                                    states[idx_1:idx_2,num_vehicles:,:], 
                                                                    vehicle_size)
            T,V,O = torch.where(torch.diff(collisions_t, dim=0, prepend=prepend)==1)
            prepend = collisions_t[-1:]
            
            if len(T)>0:
                collisions.append(torch.stack((T+idx_1,V,O+num_vehicles), dim=-1))
                
            del collisions_t
    
    if len(collisions) > 0:      
        collisions = torch.cat(collisions, dim=0)
    else:
        collisions = torch.empty((0,3))
        
    return collisions

def check_collision_rectangular_circle_batch(vehicle_states, obstacle_states, vehicle_size):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vehicle_states = vehicle_states.to(device)
    obstacle_states = obstacle_states.to(device)
    
    vehicle_corners_left_top = torch.tensor([vehicle_size[1]/2, vehicle_size[0]/2],
                                            device=device).float()
    
    inv_rotation = torch.empty((*vehicle_states.shape[:-1],2,2), device=device).float() #[T,Nv,2,2]
    
    inv_rotation[...,0,0] = torch.cos(vehicle_states[...,2])
    inv_rotation[...,0,1] = torch.sin(vehicle_states[...,2])
    inv_rotation[...,1,0] = -torch.sin(vehicle_states[...,2])
    inv_rotation[...,1,1] = torch.cos(vehicle_states[...,2])
    
    vect_rect_to_cir = (inv_rotation[:,:,None,:,:]@(obstacle_states[:,None,:,4:6,None]-vehicle_states[:,:,None,:2,None])).squeeze(-1) #[T,Nv,No,2]
    
    min_dist = torch.norm(torch.clip((torch.abs(vect_rect_to_cir)-vehicle_corners_left_top), min=0, max=None),p=2,dim=-1) #[T,Nv,No]
    
    collision = (min_dist-obstacle_states[:,None,:,6]) <= 0    
    
    return collision.cpu().long()


def check_collision_rectangular_rectangular_batch(vehicle_states_comb, vehicle_size):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vehicle_states_comb = vehicle_states_comb.to(device).permute(2,0,1,3)
    
    not_collision = torch.zeros(vehicle_states_comb.shape[:2], dtype=bool, device=device) #[T,Nc,2,8]
    
    proj_vect = []
    
    vehicle_corners = torch.tensor([[vehicle_size[1]/2, vehicle_size[0]/2],
                                    [-vehicle_size[1]/2, vehicle_size[0]/2],
                                    [-vehicle_size[1]/2, -vehicle_size[0]/2],
                                    [vehicle_size[1]/2, -vehicle_size[0]/2],],
                                    device=device).float()
    
    rotation = torch.empty((*vehicle_states_comb.shape[:-1],2,2), device=device).float() #[T,Nc,2,2,2]
    
    rotation[...,0,0] = torch.cos(vehicle_states_comb[...,2])
    rotation[...,0,1] = -torch.sin(vehicle_states_comb[...,2])
    rotation[...,1,0] = torch.sin(vehicle_states_comb[...,2])
    rotation[...,1,1] = torch.cos(vehicle_states_comb[...,2])
    
    vehicle_corners = (rotation[:,:,:,None,:,:] @ vehicle_corners[None,None,:,:,None]).squeeze(-1)
    vehicle_corners = vehicle_corners + vehicle_states_comb[:,:,:,None,:2] #[T,Nc,2,4,2]

    proj_vect = rotation.transpose(-1,-2).flatten(-3,-2) #[T,Nc,4,2]
    
    for i in range(4):
        
        if torch.all(not_collision):
            break
            
        proj_vehicle_0 = (vehicle_corners[:,:,0] @ proj_vect[:,:,i,:,None]).squeeze(-1)
        proj_vehicle_1 = (vehicle_corners[:,:,1] @ proj_vect[:,:,i,:,None]).squeeze(-1)
        
        not_collision |= (torch.amax(proj_vehicle_0, dim=-1)<torch.amin(proj_vehicle_1, dim=-1)) | \
                         (torch.amax(proj_vehicle_1, dim=-1)<torch.amin(proj_vehicle_0, dim=-1))
    
    return ~not_collision.cpu().long()

def check_collision_rectangular_circle(state_rect, state_cir, vehicle_size):
    
    assert len(state_rect) == len(state_cir) or len(state_cir) == 1, \
        "Mismatch of data length in collision check of one vehicle and one obstacle!"
    
    vehicle_corners_left_top = torch.tensor([vehicle_size[1]/2, vehicle_size[0]/2])
    
    inv_rotation = torch.empty((len(state_rect),2,2))
    
    inv_rotation[:,0,0] = torch.cos(state_rect[:,2])
    inv_rotation[:,0,1] = torch.sin(state_rect[:,2])
    inv_rotation[:,1,0] = -torch.sin(state_rect[:,2])
    inv_rotation[:,1,1] = torch.cos(state_rect[:,2])
    
    vect_rect_to_cir = (inv_rotation @ (state_cir[:,4:6,None]-state_rect[:,:2,None])).squeeze(-1)
    
    min_dist = torch.norm(torch.clip((torch.abs(vect_rect_to_cir)-vehicle_corners_left_top),min=0,max=None),p=2,dim=-1)
    
    collision = (min_dist - state_cir[:,6]) <= 0    
    
    return collision

def check_collision_rectangular_rectangular(state_1, state_2, vehicle_size):
    
    assert len(state_1) == len(state_2), \
        "Mismatch of data length in collision check of two vehicles!"
    
    not_collision = torch.zeros(len(state_1), dtype=bool)
    
    proj_vect = []
    
    vehicle_corners = torch.tensor([[vehicle_size[1]/2, vehicle_size[0]/2],
                                    [-vehicle_size[1]/2, vehicle_size[0]/2],
                                    [-vehicle_size[1]/2, -vehicle_size[0]/2],
                                    [vehicle_size[1]/2, -vehicle_size[0]/2],
                                    ])
    
    rotation_1 = torch.empty((len(state_1),2,2))
    
    rotation_1[:,0,0] = torch.cos(state_1[:,2])
    rotation_1[:,0,1] = -torch.sin(state_1[:,2])
    rotation_1[:,1,0] = torch.sin(state_1[:,2])
    rotation_1[:,1,1] = torch.cos(state_1[:,2])
    
    vehicle_corners_1 = (rotation_1[:,None,:,:] @ vehicle_corners[None,:,:,None]).squeeze(-1)
    vehicle_corners_1 = vehicle_corners_1 + state_1[:,None,:2]
    
    proj_vect.append(rotation_1[:,:,0])
    proj_vect.append(rotation_1[:,:,1])
    
    rotation_2 = torch.empty((len(state_2),2,2))
    
    rotation_2[:,0,0] = torch.cos(state_2[:,2])
    rotation_2[:,0,1] = -torch.sin(state_2[:,2])
    rotation_2[:,1,0] = torch.sin(state_2[:,2])
    rotation_2[:,1,1] = torch.cos(state_2[:,2])
    
    vehicle_corners_2 = (rotation_2[:,None,:,:] @ vehicle_corners[None,:,:,None]).squeeze(-1)
    vehicle_corners_2 = vehicle_corners_2+ state_2[:,None,:2]
    
    proj_vect.append(rotation_2[:,:,0])
    proj_vect.append(rotation_2[:,:,1])
    
    for i in range(4):
        
        if torch.all(not_collision):
            break
            
        proj_vehicle_1 = (vehicle_corners_1[(not_collision==False),:,:] @ proj_vect[i][(not_collision==False),:,None]).squeeze(-1)
        proj_vehicle_2 = (vehicle_corners_2[(not_collision==False),:,:] @ proj_vect[i][(not_collision==False),:,None]).squeeze(-1)
        
        not_collision[(not_collision==False)] |= (torch.amax(proj_vehicle_1, dim=-1)<torch.amin(proj_vehicle_2, dim=-1)) | \
                                                 (torch.amax(proj_vehicle_2, dim=-1)<torch.amin(proj_vehicle_1, dim=-1))
    
    return ~not_collision


def calculate_reach_goal(final_states, collisions, trajectory_idx, position_tolerance, 
                         angle_tolerance, num_vehicles, num_obstacles, consider_angle=True,
                         return_states_detail=False):
    
    vehicle_collisions = torch.zeros((len(trajectory_idx)-1,num_vehicles+num_obstacles), dtype=bool)
    if len(collisions)>0:
        col_traj_idx = torch.bucketize(collisions[:,0].contiguous(), 
                                        boundaries=trajectory_idx.contiguous(), 
                                        right=True)-1
        vehicle_collisions[col_traj_idx, collisions[:,1]] = True
        vehicle_collisions[col_traj_idx, collisions[:,2]] = True
    vehicle_collisions = vehicle_collisions[:,:num_vehicles]
    
    pos_diff = torch.norm(final_states[:,:,:2]-final_states[:,:,4:6],p=2,dim=-1)
    reach_goal = (pos_diff<=position_tolerance) 
    
    if consider_angle:
        
        angle_diff_1 = (final_states[:,:,2] - final_states[:,:,6])[...,None]%(2*np.pi)
        angle_diff_2 = 2*np.pi - angle_diff_1
        angle_diff = torch.amin(torch.concat((angle_diff_1,angle_diff_2), dim=-1), dim=-1)

        reach_goal = reach_goal & (angle_diff<=angle_tolerance)   
    
    success_to_goal = (~vehicle_collisions) & reach_goal
    
    if return_states_detail:
        return success_to_goal, reach_goal, (~vehicle_collisions)
    else:
        return success_to_goal


def calculate_trajectory_efficiency(trajectory_distance, start, goal, success_index):
        
    start_goal_distance = torch.norm(start-goal, p=2, dim=-1)[success_index]
    trajectory_distance = trajectory_distance[success_index]
    trajectory_efficiency = torch.sum(start_goal_distance)/torch.sum(trajectory_distance)
    
    return trajectory_efficiency
            

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default="./configs/calculate_metrics.yaml", help='specify configuration path')
    args = parser.parse_args()
    
    config_path= args.config_path
    config = load_yaml(config_path)
    
    if isinstance(config["data folder"], list):
        data_folders = config["data folder"].copy()
        
        for data_folder in data_folders:
            print(f"Currently evaluating {data_folder}: ")
            config["data folder"] = data_folder
            calculate_metrics(config)
            print("*"*10)
        
    else:
        print(f"Currently evaluating {config['data folder']}: ")
        calculate_metrics(config)
        print("*"*10)