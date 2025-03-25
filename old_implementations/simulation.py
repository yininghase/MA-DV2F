import os
import sys
import random
import numpy as np
from numpy import pi
import torch
from scipy.optimize import minimize

WORK_SPACE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(WORK_SPACE)

from mpc import ModelPredictiveControl
from visualization import Visualize_Trajectory
from data_process import change_to_relative_frame, get_angle_diff
from dvf_cpu import dynamic_velocity_field


def sim_run(config, model = None, device = 'cpu'):
    
    if config["run other algorithm"] == 'mpc':
        mpc = ModelPredictiveControl(config)

        if mpc.control_init is None:
            u = np.zeros((mpc.horizon, mpc.num_vehicle, 2))
        else:
            u = mpc.control_init[:mpc.horizon]
    
        # we have limit on steering angle and pedal   
        bounds = np.array([[-1, 1], [-0.8, 0.8]])
        bounds = np.tile(bounds, (mpc.horizon*mpc.num_vehicle, 1))

        ref = mpc.targets
    
    elif config["run other algorithm"] == 'dvf':
        config["horizon"] = 1
        
    elif not config['enable dvf'] and not os.path.exists(config["model path"]):
        raise NotImplementedError("Unknown optimization method or Inexisting model!")

    state_i = np.array([config["starts"]])
    targets = np.array(config["targets"])

    sim_total = config["simulation time"]
    horizon = config["horizon"]
    num_vehicle = len(config["starts"])
    num_obstacle = len(config["obstacles"])
    
    predict_info_opt = np.empty((0, horizon+1, num_vehicle, 4)) # to store predicted states from MPC
    predict_info_model = np.empty((0, horizon+1, num_vehicle, 4)) # to store predicted states from Model

    label_data_opt =  np.empty((0, horizon, num_vehicle, 2)) # to store predicted controls from MPC
    label_data_model = np.empty((0, horizon, num_vehicle, 2)) # to store predicted controls from Model
    
    # range for random offset 
    offsets = [1.0, 1.001]
    offset = offsets[random.randint(0, 1)]

    for i in range(1, sim_total+1):
        ### Optimization Prediction from MPC ###
        if config["run other algorithm"] == 'mpc':
            u = u[1:,...]
            u = np.concatenate((u, u[-1][None,...]), axis=0)      
            u_solution = minimize(mpc.cost_function, 
                                  u.flatten(), 
                                  (state_i[-1], ref),
                                  method='SLSQP',
                                  bounds=bounds,
                                  tol=1e-5)
            
            u_opt = u_solution.x.reshape(mpc.horizon, mpc.num_vehicle, 2)
            y_opt = mpc.plant_model(state_i[-1], mpc.dt, u_opt[0])
            label_data_opt = np.concatenate((label_data_opt, u_opt[None,...]))
            predict_info_opt = np.concatenate((predict_info_opt, get_predictions(horizon, state_i[-1], u_opt)[None,...]))
            u = u_opt
        
        elif config["run other algorithm"] == 'dvf':
            starts_torch = torch.from_numpy(state_i[-1])[None,...]
            targets_torch = torch.from_numpy(config["targets"])[None,...]
            obstacles_torch = torch.from_numpy(config["obstacles"])[None,...]
            u_opt = dynamic_velocity_field(starts_torch, targets_torch, obstacles_torch)[1].numpy()
            u = u_opt
            y_opt = vehicle_kinematic(state_i[-1], u_opt[0])

        ### Model Prediction ###
        if model is not None:
            filter_edges = config['filter edges']
            model.eval()
            
            if config["sensor noise"]:
                veh_observation = introduce_sensor_noise(state_i[-1], num_vehicle)
            else:
                veh_observation = state_i[-1]
            
            vehicles = torch.tensor(np.concatenate((veh_observation, np.array(targets)), axis=1)).float()
            vehicles = torch.cat([vehicles, torch.zeros(num_vehicle, 1)], dim=1)
            
            if num_obstacle > 0:
                obstacles = torch.tensor(config["obstacles"]).float()
                obstacles = torch.cat((torch.zeros(num_obstacle,4), obstacles, torch.ones(num_obstacle, 1)), dim=1)
                model_input = torch.cat((vehicles, obstacles), dim=0)
            else:
                model_input = vehicles
            
            obstacles = slice(num_vehicle, num_vehicle+num_obstacle)
            model_input[obstacles,:2] = model_input[obstacles,4:6]
            model_input[obstacles,-1] = model_input[obstacles,-2]
            model_input[obstacles,-2] = 0
            
            model_input = model_input.to(device)
            batches = torch.tensor([[num_vehicle+num_obstacle, num_vehicle]]).long().to(device)
            u_model = model(model_input, batches, filter_edges).detach().cpu().numpy()
                
            u_model = u_model.reshape(num_vehicle, horizon, 2)
            u_model = np.transpose(u_model, (1,0,2))
            
            if config["steering angle noise"]:
                u_model = introduce_steering_angle_noise(u_model, num_vehicle)
            
            if config["pedal noise"]:
                u_model = introduce_pedal_noise(u_model, state_i[-1], num_vehicle)
            
            y_model = vehicle_kinematic(state_i[-1], u_model[0])
            label_data_model = np.concatenate((label_data_model, u_model[None,...]))
            predict_info_model = np.concatenate((predict_info_model, get_predictions(horizon, state_i[-1], u_model)[None,...]))
        
        ### Simulation ###
        y = None           
        if model is not None:
            y = y_model[None,...]
        else:
            y = y_opt[None,...]

        if config["random offset"]:
            state_quotient = (i/sim_total) if i >= 10 else 1
            y = introduce_random_offset(y, num_vehicle, state_quotient, base_offset=offset)
            

        state_i = np.concatenate((state_i, y), axis=0)
        
        # check if the vehicle has not moved the last n steps and if this is not the case end the simulation
        n_steps = 10 # if the vehicle has not moved more then "stop tolerance" for n steps we end the simulation
        if len(state_i) > n_steps and \
            np.all(np.sum(np.linalg.norm(np.diff(state_i[-n_steps:,:,:2], axis=0), axis=-1), axis=0) < config["stop tolerance"]):
            break
    
    num_step = len(state_i)-1
    
    # check if all the vehicle reach their goals at the end of the simulation with in tolerance,
    # if not, the simulation failed, the data should not be collected
    if  model is None and config["collect data"] and \
        not (np.all(np.linalg.norm(state_i[-1:,:,:2] - targets[:,:2], axis=-1) < config["position tolerance"]) and \
            np.all(get_angle_diff(state_i[-1:,:, 2], targets[:,2]) < config["angle tolerance"])) :
        
        print(f"max position error: {np.max(np.linalg.norm(state_i[-1:,:,:2] - targets[:,:2], axis=-1)):.6f}")
        print(f"max angle error: {np.max(np.abs(state_i[-1:,:, 2] - targets[:,2])):.6f}")
        
        success = False
    
    else:
        success = True
    
    
    if (config["save plot"] or config["show plot"]):
        
        if "control init trajectory" in config.keys() and config["control init trajectory"] is not None:
            visualization = Visualize_Trajectory(config)
            predict_info_init = config["control init trajectory"][None,...]
            visualization.plot_initialization_optimization(state_i, predict_info_opt, predict_info_init)
            
        else: 
            config["is model"] = (model is not None)
            visualization = Visualize_Trajectory(config)
            visualization.create_video(state_i, predict_info_opt, predict_info_model)
            visualization.plot_trajectory(state_i)
        
    ###################
    # COLLECTING TRAINING DATA
    if config["collect data"]:

        # For each data point of vehicle we have a vehicle position (x,y), angle and velocity and the desired position and angle, 
        # For each data point of obstacle we have a obstacle position (x,y), radius 
        # To distinguish the vehicle and obstacle, we add a sign addtionally, for vehicle it is 0, for obstacle it is 1
        # so for vehicle: [x, y, angle, v, x_d, y_d, angle_d, 0], for obstacle: [0, 0, 0, 0, x, y, r, 1]
        # for the problem of m vehicles and n obstacles, we stack the first m vehicles and n obstacles together like
        # [[x, y, angle, v, x_d, y_d, angle_d, 0], # vehicle 1 
        #             ......
        #  [x, y, angle, v, x_d, y_d, angle_d, 0], # vehicle m
        #  [0, 0, 0, 0, x, y, r, 1],   # obstacle 1
        #             ......
        #  [0, 0, 0, 0, x, y, r, 1],   # obstacle n
        # ] 
        # 
        vehicles = np.concatenate((state_i[:-1], np.tile(targets, (len(state_i)-1, 1, 1)), np.zeros((len(state_i)-1,num_vehicle,1))), axis=-1)
        if num_obstacle > 0:
            obstacles = config["obstacles"]
            obstacles = np.concatenate((np.zeros((num_obstacle, 4)), obstacles, np.ones((num_obstacle,1))), axis=-1)
            X_tensor = np.concatenate((vehicles, np.tile(obstacles, (len(state_i)-1, 1, 1))), axis=1)
        else:
            X_tensor = vehicles
        
        X_tensor = torch.tensor(X_tensor.reshape(-1,8))
        
        # Each ground truth label of vehicle data points has to contain the predicted steps of 
        # pedal and steering angle at each step of the horizon, like
        # [[pedal_0, steering_angle_0, pedal_1, steering_angle_1, ..., pedal_t, steering_angle_t]  # vehicle 1 
        #  ...
        #  [pedal_0, steering_angle_0, pedal_1, steering_angle_1, ..., pedal_t, steering_angle_t]  # vehicle m
        # ]
        # for obstacle data points there is no ground truth label
        y_tensor_model = torch.tensor(label_data_model)
        if len(y_tensor_model) > 0:
            y_tensor_model = torch.transpose(y_tensor_model, 1, 2).reshape(-1, horizon*2)
        
        y_tensor_GT = torch.tensor(label_data_opt)
        y_tensor_GT = torch.transpose(y_tensor_GT, 1, 2).reshape(-1, horizon*2)
        
        # we need additional tensor to record the size of the problem, 
        # i.e, how many vehicles and obstacles are included in the case
        # for a simulation time of T, we can get datas of T problems
        # so the batch tensor is like
        # [[num_vehicles + num_obstacles, num_vehicles], # problem 1
        #           ......
        #  [num_vehicles + num_obstacles, num_vehicles], # problem T
        # ]
        batches_tensor = torch.tensor([[num_vehicle+num_obstacle, num_vehicle]]).repeat(len(state_i)-1, 1)
        
    else:
        X_tensor = None
        y_tensor_GT = None
        y_tensor_model = None
        batches_tensor = None
    
    return X_tensor, batches_tensor, y_tensor_GT, y_tensor_model, success, num_step # simulation_sucessfull


def vehicle_kinematic(prev_state, control, dt=0.2):
    x_t = prev_state[...,0]
    y_t = prev_state[...,1]
    psi_t = prev_state[...,2]
    v_t = prev_state[...,3]
    
    pedal = control[...,0]
    steering = control[...,1]

    # Vehicle Kinematic Equation
    x_t = x_t+v_t*np.cos(psi_t)*dt
    y_t = y_t+v_t*np.sin(psi_t)*dt
    psi_t = psi_t+v_t*dt*np.tan(steering)/2.0
    psi_t = (psi_t+pi)%(2*pi)-pi
    v_t = 0.99*v_t+pedal*dt
    
    next_state = np.concatenate((x_t[...,None], 
                                    y_t[...,None], 
                                    psi_t[...,None], 
                                    v_t[...,None]), 
                                axis=-1)

    return next_state
  

def get_predictions(horizon, initial_state, u, dt=0.2):
    
    predicted_state = np.array([initial_state])
    for i in range(horizon):
        predicted = vehicle_kinematic(predicted_state[-1], u[i])
        predicted_state = np.concatenate((predicted_state, predicted[None,...]))
    
    return predicted_state

def introduce_random_offset(y, num_vehicle, state_quotient, base_offset=1):

    sigma = np.array([0.25,0.25,np.pi/18,0.25])
    offset = np.random.normal(0, sigma*(base_offset-state_quotient), (num_vehicle,4))
    
    return y + offset

def introduce_sensor_noise(x, num_vehicle, sigma=[0.05,0.05,np.pi/36,0.05]):

    v = np.abs(x[:,3:4])
    offset = np.random.normal(np.zeros((num_vehicle,4)), np.array(sigma)*v)
    
    return x + offset

def introduce_steering_angle_noise(u, num_vehicle, sigma=0.25):
    
    theta = np.abs(u[0,:,1])
    offset = np.random.normal(np.zeros(num_vehicle), sigma*theta+np.pi/90)
    u[0,:,1] = np.clip(u[0,:,1] + offset, a_min=-0.8, a_max=0.8)
    
    u[0,1:,1] = np.random.uniform(low=-0.8, high=0.8,size=u[0,1:,1].shape)
    
    return u

def introduce_pedal_noise(u, x, num_vehicle, sigma=0.25):
    
    vel = np.abs(x[:,3])
    offset = np.random.normal(np.zeros(num_vehicle), sigma*vel)
    u[0,:,0] = np.clip(u[0,:,0] + offset, a_min=-1.0, a_max=1.0)
    
    u[0,1:,0] = np.random.uniform(low=-1, high=1,size=u[0,1:,0].shape)
    
    return u
