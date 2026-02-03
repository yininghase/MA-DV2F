import os
import time
import numpy as np
from numpy import pi
import torch
from tqdm import tqdm
from argparse import ArgumentParser

from gnn import IterativeGNNModel
from data_process import get_problem, load_yaml, load_test_data
from visualization import Visualize_Trajectory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_early_stopping(states, position_tolerance=1.0, angle_tolerance=0.2, 
                         stop_tolerance=0.1, n_steps=10):
    
    pos_diff = torch.norm(torch.diff(states[...,:2], dim=1), p=2, dim=-1)
    L,T,V = pos_diff.shape
    
    pos_diff_sum_n = torch.zeros_like(pos_diff)
    
    for i in range(n_steps):
        pos_diff_sum_n[:,:T-i,:] += pos_diff[:,i:,:] 
    
    pos_diff_mask = torch.all(pos_diff_sum_n<stop_tolerance, dim=-1).long()
    pos_diff_mask = torch.flip(torch.cumprod(torch.flip(pos_diff_mask, dims=(-1,)), dim=-1), dims=(-1,))
    
    pos_error = torch.norm(states[:,1:,:,:2]-states[:,-1:,:,:2], p=2, dim=-1)
    angle_error = torch.abs(states[:,1:,:,2]-states[:,-1:,:,2])
    end_error_mask = (pos_error<position_tolerance) & (angle_error<angle_tolerance)
    end_error_mask = torch.all(end_error_mask, dim=-1).long()
    end_error_mask = torch.flip(torch.cumprod(torch.flip(end_error_mask, dims=(-1,)), dim=-1), dims=(-1,))
    
    stop_mask = end_error_mask | pos_diff_mask
    
    traj_idx = torch.clip(torch.sum(1-stop_mask, dim=-1)+n_steps, min=n_steps, max=T)
    traj_mask = torch.arange(T)[None,:].repeat(L,1)<traj_idx[:,None]
    
    return traj_mask, traj_idx
    

def run_vehicle_dynamics(prev_state, control, dt=0.2):
        
        x_t = prev_state[...,0]
        y_t = prev_state[...,1]
        psi_t = prev_state[...,2]
        v_t = prev_state[...,3]
        
        pedal = control[...,0]
        steering = control[...,1]

        # Vehicle Dynamic Equation
        x_t = x_t+v_t*torch.cos(psi_t)*dt
        y_t = y_t+v_t*torch.sin(psi_t)*dt
        psi_t = psi_t+v_t*dt*torch.tan(steering)/2.0
        psi_t = (psi_t+pi)%(2*pi)-pi
        v_t = 0.99*v_t+pedal*dt
        
        next_state = torch.cat((x_t[...,None], y_t[...,None], 
                                psi_t[...,None], v_t[...,None]), 
                                dim=-1)

        return next_state


@torch.no_grad()
def inference_gnn_model(starts, targets, obstacles, model, forward_steps=1, filter_edges=False, 
                        stop_tolerance=0.1, n_steps=10, steering_angle_noise=False):
    
    N1,V,_ = starts.shape
    N2,O,_ = obstacles.shape
    
    assert N1==N2, "Error: mismatched data length!"
    
    batches = torch.tensor([[V+O, V]]).repeat((N1,1))
    batches = batches.long().to(model.device)
    
    current_states = starts.float().clone()
    targets = targets.float().clone()
    
    states = [current_states]
    controls = []
    time_list = [] 
    
    stop_cnt = 0
    
    for i in range(forward_steps):
    
        vehicle_inputs = torch.zeros((N1,V,8)).float()
        vehicle_inputs[:,:,:4] = current_states
        vehicle_inputs[:,:,4:7] = targets
        
        obstacle_inputs = torch.zeros((N2,O,8)).float()
        obstacle_inputs[:,:,:2] = obstacles[:,:,:2]
        obstacle_inputs[:,:,4:6] = obstacles[:,:,:2]
        obstacle_inputs[:,:,7] = obstacles[:,:,2]
        
        model_inputs = torch.cat((vehicle_inputs, obstacle_inputs), dim=1).reshape(-1,8).to(model.device)
        
        start_time = time.time()
        model_control = model(model_inputs, batches, filter_edges)
        end_time = time.time()
        model_control = model_control.detach().cpu().reshape(N1,V,2)
        
        if steering_angle_noise:
            model_control = introduce_steering_angle_noise(model_control)
        
        current_states = run_vehicle_dynamics(current_states.clone(), model_control.clone(), dt=0.2)
        
        pos_diff = torch.norm(states[-1][...,:2]-current_states[...,:2], p=2, dim=-1)
        
        states.append(current_states)
        controls.append(model_control)
        time_list.append(end_time-start_time)
        
        if torch.all(pos_diff<stop_tolerance):
            stop_cnt += 1
        else:
            stop_cnt = 0
        
        if stop_cnt == n_steps:
            break
    
    states = torch.stack(states[:-1]).permute(1,0,2,3)
    controls = torch.stack(controls).permute(1,0,2,3)
    batches = batches.detach().cpu()
    runtime = np.sum(time_list)
    
    return states, controls, targets, batches, runtime


def inference_multiple_cases_parallel(config):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device available now:', device)
    
    problem = np.array(config['problem'])
    assert problem.shape == (2,) and problem[0] >= 1 and problem[1] >= 0, \
        "Invalid input of problem_collection!"

    num_vehicles, num_obstacles = problem
    batch_size = config["batch size"]
    algorithm_type = config["algorithm type"]
    simulation_time = config["simulation time"]
    position_tolerance = config["position tolerance"]
    angle_tolerance = config["angle tolerance"]
    stop_tolerance = config["stop tolerance"]
    filter_edges = config["filter edges"]
    steering_angle_noise = config["steering angle noise"]
    
    print(f'Currently working on {num_vehicles} vehicles and {num_obstacles} obstacles.')
    
    if config["collect data"]:
                
        if not os.path.exists(config["data folder"]):
            os.makedirs(config["data folder"])

        data = {
                "X_data": torch.tensor([]), 
                "y_model_data": torch.tensor([]), "batches_data": torch.tensor([]),
                "X_data_path": os.path.join(config["data folder"], 
                                            f"X_data_vehicle={num_vehicles}_obstacle={num_obstacles}.pt"),
                "y_model_data_path": os.path.join(config["data folder"], 
                                                f"y_model_data_vehicle={num_vehicles}_obstacle={num_obstacles}.pt"),
                "batches_data_path": os.path.join(config["data folder"], 
                                                f"batches_data_vehicle={num_vehicles}_obstacle={num_obstacles}.pt"), 
                }
        
        if config["collect trajectory"]:
            data.update({"trajectory_data": torch.tensor([0]),
                         "trajectory_data_path": os.path.join(config["data folder"], 
                         f"trajectory_data_vehicle={num_vehicles}_obstacle={num_obstacles}.pt"),
            })
    
    if algorithm_type == 'gnn':
        
        assert os.path.exists(config["model path"]), "Given model path does not exist!"
        
        model = IterativeGNNModel(horizon = 1,  
                                max_num_vehicles = num_vehicles, 
                                max_num_obstacles = num_obstacles,
                                mode = "inference",
                                load_all_simpler = False, 
                                device = device,
                                )
        model.load_state_dict(torch.load(config["model path"]))
        
    
    elif algorithm_type == 'dvf':
        
        model = IterativeGNNModel(horizon = 1,  
                                max_num_vehicles = num_vehicles, 
                                max_num_obstacles = num_obstacles,
                                mode = "dynamic velocity field",
                                load_all_simpler = False, 
                                device = device,
                                )
    
    else:
        raise NotImplementedError("Specified algorithm has not been implemented!")
    
    model.to(device)
    model.eval()
    
    if config["test data souce"] == "fixed test data":
        
        assert os.path.exists(config["test data folder"]), \
            "The test data folder does not exist!"
        
        test_data = load_test_data(num_vehicles = num_vehicles,
                                    num_obstacles = num_obstacles,
                                    load_all_simpler = False, 
                                    folders = config["test data folder"],
                                    lim_length = config["test data each case"],
                                    )

        config["simulation runs"] = len(test_data)
    
    elif config["test data souce"] == "on the fly":
        pass
    
    else:
        raise NotImplementedError("Unknown test data source!")
    
    num_finished = 0
    
    pbar = tqdm(total=config["simulation runs"])

    while num_finished<config["simulation runs"]:
        
        B = min(batch_size, config["simulation runs"]-num_finished)
        
        if config["test data souce"] == "fixed test data":
            
            starts, targets, obstacles, _ = zip(*test_data[num_finished:num_finished+B])
            
            starts = torch.stack(starts, dim=0)
            targets = torch.stack(targets, dim=0)
            obstacles = torch.stack(obstacles, dim=0)
        
        elif config["test data souce"] == "on the fly":
                  
            vehicles, obstacles = get_problem(num_vehicles, num_obstacles, data_length=B)
            starts = vehicles[:,:,:4]
            targets = vehicles[:,:,4:]
        
        states, controls, targets, batches, runtime = inference_gnn_model(starts, targets, obstacles, model, 
                                                                          simulation_time, filter_edges, stop_tolerance,
                                                                          steering_angle_noise=steering_angle_noise)
        
        traj_mask, traj_idx = check_early_stopping(torch.cat((starts[:,None,:,:], states), dim=1), 
                                                   position_tolerance, angle_tolerance, stop_tolerance)
        
        T = traj_mask.shape[1]
        
        vehicles_data = torch.cat((states, targets[:,None,:,:].repeat((1,T,1,1)), 
                                   torch.zeros((B, T, num_vehicles, 1))), dim=-1)
        obstacle_data = torch.cat((torch.zeros((B, T, num_obstacles, 4)), 
                                   obstacles[:,None,:,:].repeat((1,T,1,1)), 
                                   torch.zeros((B, T, num_obstacles, 1))), dim=-1)
    
        X_data = torch.cat((vehicles_data, obstacle_data), dim=-2)[traj_mask].reshape(-1,8)
        y_model_data = controls[traj_mask].reshape(-1,2)
        batches_data = (batches[:,None,:].repeat((1,T,1)))[traj_mask].reshape(-1,2)
        trajectory_data = torch.cumsum(traj_idx, dim=0)*(num_obstacles+num_vehicles)
        
        del traj_mask, traj_idx, states, controls, targets, batches
    
        data["X_data"] = torch.cat((data["X_data"], X_data))
        data["y_model_data"] = torch.cat((data["y_model_data"], y_model_data))
        data["batches_data"] = torch.cat((data["batches_data"], batches_data))
        if config["collect trajectory"]:
            data["trajectory_data"] = torch.cat((data["trajectory_data"], trajectory_data+data["trajectory_data"][-1]))
    
        torch.save(data["X_data"], data["X_data_path"])
        torch.save(data["y_model_data"], data["y_model_data_path"])
        torch.save(data["batches_data"], data["batches_data_path"])
        if config["collect trajectory"]:
            torch.save(data["trajectory_data"], data["trajectory_data_path"])
            
        num_finished += B

        pbar.update(B)

def introduce_steering_angle_noise(model_control, sigma=3.0):
    theta = torch.abs(model_control[:,:,1])
    offset = torch.normal(torch.zeros_like(theta), sigma*theta+np.pi/90)
    model_control[:,:,1] = torch.clip(model_control[:,:,1] + offset, min=-0.8, max=0.8)
    return model_control

if __name__ == "__main__":     
    
    
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default="./configs/inference.yaml", 
                        help='specify configuration path')
    args = parser.parse_args()
    
    config_path= args.config_path
    config = load_yaml(config_path)
    
    if config["algorithm type"] == 'gnn':
        name = os.path.basename(config["model path"]).split(".")[0]
    elif config["algorithm type"] == 'dvf':
        name = 'dynamic_velocity_field'

    config["data folder"] = os.path.join(config["data folder"], name)
    
    for problem in config['problem collection']:
        config['problem'] = problem
        inference_multiple_cases_parallel(config)