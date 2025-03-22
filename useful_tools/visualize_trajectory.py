import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from argparse import ArgumentParser

WORK_SPACE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(WORK_SPACE)

from visualization import Visualize_Trajectory
from data_process import load_data, load_yaml

def clip_trajectory(states, n_steps=50):
    
    states = torch.cat((states, states[-1:].repeat(n_steps, 1, 1)), dim=0)
    
    state_diff = torch.norm(torch.diff(states[...,:3], dim=0), p=2, dim=-1)
    T,V = state_diff.shape
    
    state_diff_sum_n = torch.zeros_like(state_diff)
    
    for i in range(n_steps):
        state_diff_sum_n[:T-i,:] += state_diff[i:,:] 
    
    pos_diff_mask = torch.all(state_diff_sum_n[...,:2]<0.1, dim=-1)
    angle_diff = state_diff_sum_n[...,2]%(2*np.pi)
    angle_diff = torch.min(angle_diff, 2*np.pi-angle_diff)
    angle_diff_mask = angle_diff<np.pi/18
    stop_mask = (pos_diff_mask & angle_diff_mask).long()
    stop_mask = torch.flip(torch.cumprod(torch.flip(stop_mask, dims=(0,)), dim=0), dims=(0,))
    traj_idx = torch.clip(torch.sum(1-stop_mask, dim=-1)+n_steps, min=n_steps, max=T)
    
    states = states[:traj_idx]
    
    return states

def visualize_trajectory(config):
    
    assert os.path.exists(config["data folder"]), \
        f"The given folder of '{config['data folder']}' does not exist!"        
    
    os.makedirs(config['plot folder'], exist_ok=True)
    
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
            
            print(f'Currently working on {num_vehicles} vehicles and {num_obstacles} obstacles!')
            
            len_batch = num_vehicles+num_obstacles
            key = (len_batch, num_vehicles)
            value = data[key]
            
            states, _, _, trajectory_idx = value
            
            states = states.reshape(-1, len_batch, 8)
            trajectory_idx = (trajectory_idx/len_batch).long()
            if f'{num_vehicles},{num_obstacles}' in config["selected indices"] and \
                config["selected indices"][f'{num_vehicles},{num_obstacles}']:
                indices = config["selected indices"][(num_vehicles, num_obstacles)]
            else:
                indices = np.random.choice(np.arange(len(trajectory_idx)-1), size=min(len(trajectory_idx)-1, config["num random selection"]),
                                           replace=False)
            
            for selected_idx in tqdm(indices):
                selected_states = states[trajectory_idx[selected_idx]:trajectory_idx[selected_idx+1]]
                starts = selected_states[0,:num_vehicles,:4].numpy()
                targets = selected_states[0,:num_vehicles,4:7].numpy()
                obstacles = selected_states[0, num_vehicles:, 4:7].numpy()
                
                selected_states = clip_trajectory(selected_states[:,:num_vehicles,:4]).numpy()
                
                config["starts"] = starts
                config["targets"] = targets
                config["obstacles"] = obstacles
                config["name"] = f"vehicle={num_vehicles}_obstacle={num_obstacles}_run={selected_idx}"
                config["num of vehicles"] = num_vehicles
                config["num of obstacles"] = num_obstacles
                
                visualization = Visualize_Trajectory(config)
                visualization.create_video(selected_states, None, None)
                visualization.plot_trajectory(selected_states)


if __name__ == "__main__":     
    
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default=f"{WORK_SPACE}/configs/visualize_trajectory.yaml", 
                        help='specify configuration path')
    args = parser.parse_args()
    
    config_path= args.config_path
    config = load_yaml(config_path)  
    config['plot folder'] = os.path.join(WORK_SPACE, config['plot folder'], 
                                         os.path.basename(config['data folder']))
    visualize_trajectory(config)
