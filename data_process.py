import os
import torch
import numpy as np
import random
import yaml

from numpy import pi

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

START_DIST_THRESH = 0.01
TARGET_DIST_THRESH = 7
OBSTACLE_DIST_THRESH = 7
MIN_OBSTACLE_RADIUS = 1 # 2
MAX_OBSTACLE_RADIUS = 3 # 2

def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config


def get_obstacles_normal(num_obstacles, data_length, position_range=25):
    
    obstacle_range = (MAX_OBSTACLE_RADIUS+MIN_OBSTACLE_RADIUS+OBSTACLE_DIST_THRESH)*np.sqrt(num_obstacles/2)
    obstacle_range = max(obstacle_range, position_range)
    
    if num_obstacles == 0:
        return np.empty((data_length, 0, 3))
    
    pos = np.random.uniform(low=-obstacle_range, 
                            high=obstacle_range,
                            size=(data_length, num_obstacles, 2))
    
    radius = np.random.uniform(low=MIN_OBSTACLE_RADIUS, 
                               high=MAX_OBSTACLE_RADIUS, 
                               size=(data_length, num_obstacles, 1))
    
    obstacles = np.concatenate((pos, radius), axis=-1)
    
    return obstacles


# def get_obstacles_normal(num_obstacles, data_length, position_range=25):
    
#     obstacle_range = (2*MAX_OBSTACLE_RADIUS+OBSTACLE_DIST_THRESH)*np.sqrt(num_obstacles/2)
    
#     low_range_shift = min(-position_range+obstacle_range,0)
#     high_range_shift = max(position_range-obstacle_range,0)
    
#     if num_obstacles == 0:
#         return np.empty((data_length, 0, 3))
    
#     obstacles_index = np.ones((data_length, num_obstacles), dtype=bool)
#     obstacles = np.empty((data_length, num_obstacles, 3))
    
#     failed_time = 0
#     last_failed = np.sum(obstacles_index)
    
#     while last_failed>0:
        
#         dyn_pos_range = obstacle_range+(MAX_OBSTACLE_RADIUS+MIN_OBSTACLE_RADIUS)*(failed_time//50)
#         obstacles[obstacles_index] = np.random.uniform(low=[-dyn_pos_range, -dyn_pos_range, MIN_OBSTACLE_RADIUS], 
#                                                        high=[dyn_pos_range, dyn_pos_range, MAX_OBSTACLE_RADIUS], 
#                                                        size=(np.sum(obstacles_index), 3))
        
#         selected_index = np.ones_like(obstacles_index, dtype=bool)
        
#         for i in range(num_obstacles):
#             for j in range(i+1,num_obstacles):
#                 dist = np.linalg.norm(obstacles[:,i,:2]-obstacles[:,j,:2],ord=2,axis=-1) \
#                      - obstacles[:,i,2]-obstacles[:,j,2]
#                 selected_index[:,i] = selected_index[:,i] & (dist>OBSTACLE_DIST_THRESH*1.5)
#                 selected_index[:,j] = selected_index[:,j] & (dist>OBSTACLE_DIST_THRESH*1.5)
    
#         obstacles_index[obstacles_index & selected_index] = False
        
#         if last_failed <= np.sum(obstacles_index):
#             failed_time += 1
        
#         last_failed = np.sum(obstacles_index)
    
#     shift_center = np.random.uniform(low=low_range_shift, 
#                                      high=high_range_shift,
#                                      size=(data_length, 2))
    
#     obstacles[...,:2] += shift_center[:,None,:]
    
#     return obstacles


def get_obstacles_collision_mode(num_obstacles, data_length, position_range=25):
    
    obstacle_range = (MAX_OBSTACLE_RADIUS+MIN_OBSTACLE_RADIUS+OBSTACLE_DIST_THRESH)*np.sqrt(num_obstacles/2)
    
    low_range_shift = min(-position_range+obstacle_range,0)
    high_range_shift = max(position_range-obstacle_range,0)
    
    if num_obstacles == 0:
        return np.empty((data_length, 0, 3))
    
    obstacles_index = np.ones((data_length, num_obstacles), dtype=bool)
    obstacles = np.empty((data_length, num_obstacles, 3))
    
    failed_time = 0
    last_failed = np.sum(obstacles_index)
    
    while last_failed>0:
        
        dyn_pos_range = obstacle_range+(MAX_OBSTACLE_RADIUS+MIN_OBSTACLE_RADIUS)*(failed_time//50)
        obstacles[obstacles_index] = np.random.uniform(low=[-dyn_pos_range, -dyn_pos_range, MIN_OBSTACLE_RADIUS], 
                                                       high=[dyn_pos_range, dyn_pos_range, MAX_OBSTACLE_RADIUS], 
                                                       size=(np.sum(obstacles_index), 3))
        
        selected_index = np.ones_like(obstacles_index, dtype=bool)
        
        for i in range(num_obstacles):
            for j in range(i+1,num_obstacles):
                dist = np.linalg.norm(obstacles[:,i,:2]-obstacles[:,j,:2],ord=2,axis=-1) \
                     - obstacles[:,i,2]-obstacles[:,j,2]
                selected_index[:,i] = selected_index[:,i] & (dist>OBSTACLE_DIST_THRESH)
                selected_index[:,j] = selected_index[:,j] & (dist>OBSTACLE_DIST_THRESH)
    
        obstacles_index[obstacles_index & selected_index] = False
        
        if last_failed <= np.sum(obstacles_index):
            failed_time += 1
        
        last_failed = np.sum(obstacles_index)
    
    shift_center = np.random.uniform(low=low_range_shift, 
                                     high=high_range_shift,
                                     size=(data_length, 2))
    
    obstacles[...,:2] += shift_center[:,None,:]
    
    return obstacles

def get_vehicles_collision_mode(num_vehicles, obstacles, zero_velocity=True, position_range=25, 
                                velocity_range=3.5, vehicle_radius=1.5, num_collision_center=1):
    
    collision_range = min((2+np.ceil(num_vehicles/10))*(4*vehicle_radius), 50)
    data_length, num_obstacles, _ = obstacles.shape
    
    low_range_pos = np.array([-collision_range, -collision_range])
    high_range_pos = np.array([collision_range, collision_range])
    low_range_shift = min(-position_range+collision_range,0)
    high_range_shift = max(position_range-collision_range,0)
    
    if num_vehicles == 1 and num_obstacles == 0:
        
        pos = np.random.uniform(low=-position_range, high=position_range, 
                                size=(data_length, num_vehicles, 2))
        angle = np.random.uniform(low=-np.pi, high=np.pi, size=(data_length, num_vehicles, 1))
        
        if zero_velocity:
            vel = np.zeros((data_length, num_vehicles, 1))
        elif np.random.rand()>0.5:
            vel = np.random.uniform(low=-velocity_range, high=velocity_range, size=(data_length, num_vehicles, 1))
        else:
            vel = np.random.normal(loc=0, scale=(velocity_range+1)/3, size=(data_length, num_vehicles, 1))
            
        starts = np.concatenate((pos,angle,vel), axis=-1)
        
        pos = np.random.uniform(low=-position_range, high=position_range, 
                                size=(data_length, num_vehicles, 2))
        angle = np.random.uniform(low=-np.pi, high=np.pi, size=(data_length, num_vehicles, 1))
        targets = np.concatenate((pos, angle), axis=-1)
        
        return np.append(starts, targets, axis=2)
    
    if num_obstacles == 0:
        shift_center = np.random.uniform(low=low_range_shift, high=high_range_shift, size=(data_length,2))
        collision_center = np.random.uniform(low=-collision_range, high=collision_range, 
                                         size=(data_length, num_collision_center, 2))
        collision_center += shift_center[:,None,:]
        
    else:
        shift_center = np.mean(obstacles[...,:2], axis=1)   
        
        vec_to_obs = obstacles[...,:2] - shift_center[:,None,:]
        vec_percent = np.random.rand(data_length, num_collision_center, num_obstacles)
        vec_percent = vec_percent/np.sum(vec_percent, axis=-1, keepdims=True)
        collision_center = shift_center[:,None,:] + np.sum(vec_to_obs[:,None,:,:]*vec_percent[:,:,:,None], axis=-2)/2
          
    
    idx_x = np.repeat(np.arange(data_length)[:,None], num_vehicles, -1)
    idx_y = np.random.choice(num_collision_center, size=(data_length, num_vehicles), replace=True)
    collision_center = collision_center[idx_x, idx_y]
    
    vehicle_index = np.ones((data_length, num_vehicles), dtype=bool)
    pos = np.empty((data_length, num_vehicles, 2))
    
    failed_time = 0
    last_failed = np.sum(vehicle_index)
    
    while last_failed>0:
        
        pos[vehicle_index] = np.random.uniform(low=low_range_pos-2*vehicle_radius*(failed_time//50), 
                                               high=high_range_pos+2*vehicle_radius*(failed_time//50), 
                                               size=(np.sum(vehicle_index), 2))
        pos[vehicle_index] += collision_center[vehicle_index]
        
        selected_index = np.ones_like(vehicle_index, dtype=bool)
        
        for i in range(num_vehicles):
            for j in range(i+1,num_vehicles):
                dist = np.linalg.norm(pos[:,i,:2]-pos[:,j,:2],ord=2,axis=-1)-2*vehicle_radius
                selected_index[:,i] = selected_index[:,i] & (dist>TARGET_DIST_THRESH)
                selected_index[:,j] = selected_index[:,j] & (dist>TARGET_DIST_THRESH)
        
        if num_obstacles>0:
            for i in range(num_vehicles):
                for j in range(num_obstacles):
                    dist = np.linalg.norm(pos[:,i,:2]-obstacles[:,j,:2],ord=2,axis=-1)-obstacles[:,j,2]-vehicle_radius
                    selected_index[:,i] = selected_index[:,i] & (dist>TARGET_DIST_THRESH)  
        
        vehicle_index[vehicle_index & selected_index] = False
        
        if last_failed <= np.sum(vehicle_index):
            failed_time += 1
        
        last_failed = np.sum(vehicle_index)
    
    angle = np.random.uniform(low=-np.pi, high=np.pi, size=(data_length, num_vehicles, 1))
    targets = np.concatenate((pos,angle), axis=-1)
    
    if num_vehicles>1:
        v = collision_center-targets[:,:,:2]
        v = v/np.linalg.norm(v,axis=-1,keepdims=True)
        h = v[:,:,[1,0]]
        h[:,:,0] *= -1 
        
    vehicle_index = np.ones((data_length, num_vehicles), dtype=bool)
    pos = np.empty((data_length, num_vehicles, 2))
    
    failed_time = 0
    last_failed = np.sum(vehicle_index)
    
    while last_failed>0:

        if num_vehicles>1:
            high_bound = collision_range+2*vehicle_radius*(failed_time//50)
            shifts_v = np.random.uniform(low=0, high=high_bound, size=(np.sum(vehicle_index), 1))
            shifts_h = np.random.normal(loc=np.zeros_like(shifts_v), scale=shifts_v*np.tan(np.pi/12))
            pos[vehicle_index] = shifts_v*v[vehicle_index]+shifts_h*h[vehicle_index]
        else:
            pos[vehicle_index] = np.random.uniform(low=low_range_shift-5*(failed_time//50), 
                                                       high=high_range_shift+5*(failed_time//50), 
                                                       size=(np.sum(vehicle_index),2))
        
        pos[vehicle_index] += collision_center[vehicle_index]
        
        selected_index = np.ones_like(vehicle_index, dtype=bool)
        
        for i in range(num_vehicles-1):
            for j in range(i+1,num_vehicles):
                dist = np.linalg.norm(pos[:,i,:2]-pos[:,j,:2],ord=2,axis=-1)-2*vehicle_radius
                selected_index[:,i] = selected_index[:,i] & (dist>START_DIST_THRESH)
                selected_index[:,j] = selected_index[:,j] & (dist>START_DIST_THRESH)
                
        if num_obstacles>0:
            for i in range(num_vehicles):
                for j in range(num_obstacles):
                    dist = np.linalg.norm(pos[:,i,:2]-obstacles[:,j,:2],ord=2,axis=-1)-obstacles[:,j,2]-vehicle_radius
                    selected_index[:,i] = selected_index[:,i] & (dist>START_DIST_THRESH)    
        
        vehicle_index[vehicle_index & selected_index] = False
        
        if last_failed <= np.sum(vehicle_index):
            failed_time += 1
        
        last_failed = np.sum(vehicle_index)
    
    if zero_velocity:
        vel = np.zeros((data_length, num_vehicles, 1))
    elif np.random.rand()>0.5:
        vel = np.random.uniform(low=-velocity_range, high=velocity_range, size=(data_length, num_vehicles, 1))
    else:
        vel = np.random.normal(loc=0, scale=(velocity_range+1)/3, size=(data_length, num_vehicles, 1))
            
    
    angle = np.random.uniform(low=-np.pi, high=np.pi, size=(data_length, num_vehicles, 1))
    starts = np.concatenate((pos, angle, vel), axis=-1)
    
    return np.append(starts, targets, axis=2)


def get_vehicles_normal(num_vehicles, obstacles, zero_velocity=True, position_range=25, 
                        velocity_range=3.5, vehicle_radius=1.5):
    
    data_length, num_obstacles, _ = obstacles.shape
    
    if num_vehicles == 1 and num_obstacles == 0:
        
        pos = np.random.uniform(low=-position_range, high=position_range, 
                                size=(data_length, num_vehicles, 2))
        angle = np.random.uniform(low=-np.pi, high=np.pi, size=(data_length, num_vehicles, 1))
        
        if zero_velocity:
            vel = np.zeros((data_length, num_vehicles, 1))
        elif np.random.rand()>0.5:
            vel = np.random.uniform(low=-velocity_range, high=velocity_range, size=(data_length, num_vehicles, 1))
        else:
            vel = np.random.normal(loc=0, scale=(velocity_range+1)/3, size=(data_length, num_vehicles, 1))
            
        starts = np.concatenate((pos,angle,vel), axis=-1)
        
        pos = np.random.uniform(low=-position_range, high=position_range, 
                                size=(data_length, num_vehicles, 2))
        angle = np.random.uniform(low=-np.pi, high=np.pi, size=(data_length, num_vehicles, 1))
        targets = np.concatenate((pos, angle), axis=-1)
        
        return np.append(starts, targets, axis=2)
    
    if num_obstacles == 0:
        shift_center = np.random.uniform(low=-position_range, 
                                         high=position_range, 
                                         size=(data_length,2))
        
    else:
        shift_center = np.mean(obstacles[...,:2], axis=1)   
        
        vec_to_obs = obstacles[...,:2] - shift_center[:,None,:]
        vec_percent = np.random.rand(data_length, num_obstacles)
        vec_percent = vec_percent/np.sum(vec_percent, axis=-1, keepdims=True)
        shift_center += np.sum(vec_to_obs*vec_percent[:,:,None], axis=1)/2
        
        perturbed_coord = np.random.uniform(low=-5, high=5, size=(data_length, 2))
        shift_center += perturbed_coord
    
    obstacles_unbiased = obstacles.copy()
    obstacles_unbiased[...,:2] = obstacles_unbiased[...,:2]-shift_center[:,None,:]
    
    vehicle_index = np.ones((data_length, num_vehicles), dtype=bool)
    pos = np.empty((data_length, num_vehicles, 2))
    
    failed_time = 0
    last_failed = np.sum(vehicle_index)
    
    while last_failed>0:
        
        pos[vehicle_index] = np.random.uniform(low=-position_range-2*vehicle_radius*(failed_time//50), 
                                               high=position_range+2*vehicle_radius*(failed_time//50), 
                                               size=(np.sum(vehicle_index), 2))
        
        selected_index = np.ones_like(vehicle_index, dtype=bool)
        
        for i in range(num_vehicles):
            for j in range(i+1,num_vehicles):
                dist = np.linalg.norm(pos[:,i,:2]-pos[:,j,:2],ord=2,axis=-1)-2*vehicle_radius
                selected_index[:,i] = selected_index[:,i] & (dist>TARGET_DIST_THRESH)
                selected_index[:,j] = selected_index[:,j] & (dist>TARGET_DIST_THRESH)
        
        if num_obstacles>0:
            for i in range(num_vehicles):
                for j in range(num_obstacles):
                    dist = np.linalg.norm(pos[:,i,:2]-obstacles_unbiased[:,j,:2],ord=2,axis=-1) \
                         - obstacles_unbiased[:,j,2]-vehicle_radius
                    selected_index[:,i] = selected_index[:,i] & (dist>TARGET_DIST_THRESH)    
        
        vehicle_index[vehicle_index & selected_index] = False
        
        if last_failed <= np.sum(vehicle_index):
            failed_time += 1
        
        last_failed = np.sum(vehicle_index)
    
    angle = np.random.uniform(low=-np.pi, high=np.pi, size=(data_length, num_vehicles, 1))
    targets = np.concatenate((pos,angle), axis=-1)
        
    vehicle_index = np.ones((data_length, num_vehicles), dtype=bool)
    pos = np.empty((data_length, num_vehicles, 2))
    
    failed_time = 0
    last_failed = np.sum(vehicle_index)
    
    while last_failed>0:
        
        pos[vehicle_index] = np.random.uniform(low=-position_range-2*vehicle_radius*(failed_time//50), 
                                                   high=position_range+2*vehicle_radius*(failed_time//50), 
                                                   size=(np.sum(vehicle_index), 2))
        
        selected_index = np.ones_like(vehicle_index, dtype=bool)
        
        for i in range(num_vehicles):
            for j in range(i+1,num_vehicles):
                dist = np.linalg.norm(pos[:,i,:2]-pos[:,j,:2],ord=2,axis=-1)-2*vehicle_radius
                selected_index[:,i] = selected_index[:,i] & (dist>START_DIST_THRESH)
                selected_index[:,j] = selected_index[:,j] & (dist>START_DIST_THRESH)
        
        if num_obstacles>0:
            for i in range(num_vehicles):
                for j in range(num_obstacles):
                    dist = np.linalg.norm(pos[:,i,:2]-obstacles_unbiased[:,j,:2],ord=2,axis=-1) \
                         - obstacles_unbiased[:,j,2]-vehicle_radius
                    selected_index[:,i] = selected_index[:,i] & (dist>START_DIST_THRESH)   
        
        vehicle_index[vehicle_index & selected_index] = False
        
        if last_failed <= np.sum(vehicle_index):
            failed_time += 1
        
        last_failed = np.sum(vehicle_index)
        
    if zero_velocity:
        vel = np.zeros((data_length, num_vehicles, 1))
    elif np.random.rand()>0.5:
        vel = np.random.uniform(low=-velocity_range, high=velocity_range, size=(data_length, num_vehicles, 1))
    else:
        vel = np.random.normal(loc=0, scale=(velocity_range+1)/3, size=(data_length, num_vehicles, 1))
        
    angle = np.random.uniform(low=-np.pi, high=np.pi, size=(data_length, num_vehicles, 1))
    starts = np.concatenate((pos, angle, vel), axis=-1)
    
    starts[...,:2] += shift_center[:,None,:]
    targets[...,:2] += shift_center[:,None,:]
    
    return np.append(starts, targets, axis=2)


def get_vehicles_parking_mode(num_vehicles, obstacles, position_range=25, parking_range=10, 
                              velocity_range=3.5, vehicle_radius=1.5):
    
    zero_velocity = False
    
    data_length, num_obstacles, _ = obstacles.shape
    
    if num_vehicles == 1 and num_obstacles == 0:
        
        pos = np.random.uniform(low=-position_range, high=position_range, 
                                size=(data_length, num_vehicles, 2))
        angle = np.random.uniform(low=-np.pi, high=np.pi, size=(data_length, num_vehicles, 1))
        
        if zero_velocity:
            vel = np.zeros((data_length, num_vehicles, 1))
        elif np.random.rand()>0.5:
            vel = np.random.uniform(low=-velocity_range, high=velocity_range, size=(data_length, num_vehicles, 1))
        else:
            vel = np.random.normal(loc=0, scale=(velocity_range+1)/3, size=(data_length, num_vehicles, 1))
            
        starts = np.concatenate((pos,angle,vel), axis=-1)
        
        pos = np.random.uniform(low=-parking_range, high=parking_range, 
                                    size=(data_length, num_vehicles, 2)) + pos
        angle = np.random.uniform(low=-np.pi, high=np.pi, size=(data_length, num_vehicles, 1))
        targets = np.concatenate((pos, angle), axis=-1)
        
        return np.append(starts, targets, axis=2)
    
    if num_obstacles == 0:
        shift_center = np.random.uniform(low=-position_range, 
                                         high=position_range, 
                                         size=(data_length,2))
        
    else:
        shift_center = np.mean(obstacles[...,:2], axis=1)   
        
        vec_to_obs = obstacles[...,:2] - shift_center[:,None,:]
        vec_percent = np.random.rand(data_length, num_obstacles)
        vec_percent = vec_percent/np.sum(vec_percent, axis=-1, keepdims=True)
        shift_center += np.sum(vec_to_obs*vec_percent[:,:,None], axis=1)/2
        
        perturbed_coord = np.random.uniform(low=-5, high=5, size=(data_length, 2))
        shift_center += perturbed_coord
    
    obstacles_unbiased = obstacles.copy()
    obstacles_unbiased[...,:2] = obstacles_unbiased[...,:2]-shift_center[:,None,:]
    
    vehicle_index = np.ones((data_length, num_vehicles), dtype=bool)
    pos = np.empty((data_length, num_vehicles, 2))
    
    failed_time = 0
    last_failed = np.sum(vehicle_index)
    
    while last_failed>0:
        
        pos[vehicle_index] = np.random.uniform(low=-position_range-2*vehicle_radius*(failed_time//50), 
                                               high=position_range+2*vehicle_radius*(failed_time//50), 
                                               size=(np.sum(vehicle_index), 2))
        
        selected_index = np.ones_like(vehicle_index, dtype=bool)
        
        for i in range(num_vehicles):
            for j in range(i+1,num_vehicles):
                dist = np.linalg.norm(pos[:,i,:2]-pos[:,j,:2],ord=2,axis=-1)-2*vehicle_radius
                selected_index[:,i] = selected_index[:,i] & (dist>TARGET_DIST_THRESH)
                selected_index[:,j] = selected_index[:,j] & (dist>TARGET_DIST_THRESH)
        
        if num_obstacles>0:
            for i in range(num_vehicles):
                for j in range(num_obstacles):
                    dist = np.linalg.norm(pos[:,i,:2]-obstacles_unbiased[:,j,:2],ord=2,axis=-1) \
                         - obstacles_unbiased[:,j,2]-vehicle_radius
                    selected_index[:,i] = selected_index[:,i] & (dist>TARGET_DIST_THRESH)
        
        vehicle_index[vehicle_index & selected_index] = False
        
        if last_failed <= np.sum(vehicle_index):
            failed_time += 1
        
        last_failed = np.sum(vehicle_index)
    
    angle = np.random.uniform(low=-np.pi, high=np.pi, size=(data_length, num_vehicles, 1))
    targets = np.concatenate((pos,angle), axis=-1)
    
    
    vehicle_index = np.ones((data_length, num_vehicles), dtype=bool)
    pos = np.empty((data_length, num_vehicles, 2))
    
    failed_time = 0
    last_failed = np.sum(vehicle_index)
    
    while last_failed>0:
        
        pos[vehicle_index] = np.random.uniform(low=-parking_range-(failed_time//50), 
                                               high=parking_range+(failed_time//50), 
                                               size=(np.sum(vehicle_index), 2)) + targets[vehicle_index,:2]
        
        selected_index = np.ones_like(vehicle_index, dtype=bool)
        
        for i in range(num_vehicles):
            for j in range(i+1,num_vehicles):
                dist = np.linalg.norm(pos[:,i,:2]-pos[:,j,:2],ord=2,axis=-1)-2*vehicle_radius
                selected_index[:,i] = selected_index[:,i] & (dist>START_DIST_THRESH)
                selected_index[:,j] = selected_index[:,j] & (dist>START_DIST_THRESH)
        
        if num_obstacles>0:
            for i in range(num_vehicles):
                for j in range(num_obstacles):
                    dist = np.linalg.norm(pos[:,i,:2]-obstacles_unbiased[:,j,:2],ord=2,axis=-1) \
                         - obstacles_unbiased[:,j,2]-vehicle_radius
                    selected_index[:,i] = selected_index[:,i] & (dist>START_DIST_THRESH)  
        
        vehicle_index[vehicle_index & selected_index] = False
        
        if last_failed <= np.sum(vehicle_index):
            failed_time += 1
        
        last_failed = np.sum(vehicle_index)
        
    angle = np.random.uniform(low=-np.pi, high=np.pi, size=(data_length, num_vehicles, 1))
    
    if zero_velocity:
        vel = np.zeros((data_length, num_vehicles, 1))
    elif np.random.rand()>0.5:
        vel = np.random.uniform(low=-velocity_range, high=velocity_range, size=(data_length, num_vehicles, 1))
    else:
        vel = np.random.normal(loc=0, scale=(velocity_range+1)/3, size=(data_length, num_vehicles, 1))
        
    starts = np.concatenate((pos, angle, vel), axis=-1)
    
    starts[...,:2] += shift_center[:,None,:]
    targets[...,:2] += shift_center[:,None,:]

    return np.append(starts, targets, axis=2)


def get_problem(num_vehicles, num_obstacles, data_length = 1, position_range=25, 
                 zero_velocity=True, collision_mode=True, parking_mode=False):
    ''' The function to get the problem with num_vehicles and num_obstacles'''
    
    
    if collision_mode:
        obstacles = get_obstacles_collision_mode(num_obstacles, data_length, position_range)
        vehicles = get_vehicles_collision_mode(num_vehicles, obstacles, zero_velocity, position_range)
    
    elif parking_mode:
        obstacles = get_obstacles_normal(num_obstacles, data_length, position_range)
        vehicles = get_vehicles_parking_mode(num_vehicles, obstacles, position_range)
    
    else:
        obstacles = get_obstacles_normal(num_obstacles, data_length, position_range)
        vehicles = get_vehicles_normal(num_vehicles, obstacles, zero_velocity, position_range)

    return vehicles, obstacles
        
  

def load_data(num_vehicles, num_obstacles, load_all_simpler=True, folders="./data/data_generation", 
              horizon=0, load_trajectory=False, load_model_prediction=False):
    '''load data from folder'''
    data = {}
    
    if isinstance(folders, str):
        folders = [folders]
    else:
        assert isinstance(folders, list), \
            "invalid input of data folders, should be string of list or string"
    
    if load_all_simpler:
        # load all simpler case with less vehicles and obstacles
        for i in range(1, num_vehicles+1):
            for j in range(num_obstacles+1):
                for folder in folders:
                    X_data_path = os.path.join(folder, f"X_data_vehicle={i}_obstacle={j}.pt")
                    
                    if load_model_prediction:
                        y_data_path = os.path.join(folder, f"y_model_data_vehicle={i}_obstacle={j}.pt")  
                    else:
                        y_data_path = os.path.join(folder, f"y_GT_data_vehicle={i}_obstacle={j}.pt")
                                            
                    batches_data_path = os.path.join(folder, f"batches_data_vehicle={i}_obstacle={j}.pt")
                    
                    if load_trajectory:
                        trajectory_data_path = os.path.join(folder, f"trajectory_data_vehicle={i}_obstacle={j}.pt")
                        if not os.path.exists(trajectory_data_path):
                            continue
    
                    if os.path.exists(X_data_path) and os.path.exists(y_data_path) and os.path.exists(batches_data_path):
                        
                        new_X_data = torch.load(X_data_path).float()
                        new_y_data = torch.load(y_data_path).float()
                        new_batches_data = torch.load(batches_data_path).long()
                        
                        if isinstance(horizon, int) and horizon>0 and 2*horizon<new_y_data.shape[-1]:
                            new_y_data = new_y_data[:,:2*horizon]
                        
                        if load_trajectory:
                            new_trajectory_data = torch.load(trajectory_data_path).long()
                        
                        if (i+j,i) in data.keys():
                            
                            if load_trajectory:
                                X_data, y_data, batches_data, trajectory_data = data[(i+j,i)]
                            else:
                                X_data, y_data, batches_data = data[(i+j,i)]
                            
                            X_data = torch.cat((X_data, new_X_data))
                            y_data = torch.cat((y_data, new_y_data))
                            batches_data = torch.cat((batches_data, new_batches_data))
                            
                            if load_trajectory:
                                trajectory_data = torch.cat((trajectory_data, new_trajectory_data[1:]+trajectory_data[-1]))
                                data[(i+j,i)] = [X_data, y_data, batches_data, trajectory_data]
                            
                            else:
                                data[(i+j,i)] = [X_data, y_data, batches_data]
                            
                        else:
                            
                            if load_trajectory:
                                data[(i+j,i)] = [new_X_data, new_y_data, new_batches_data, new_trajectory_data]
                            else:
                                data[(i+j,i)] = [new_X_data, new_y_data, new_batches_data]
                        
    else:
        # load only the case with of given num of vehicles and obstacles
        
        i = num_vehicles
        j = num_obstacles
        
        for folder in folders:
            X_data_path = os.path.join(folder, f"X_data_vehicle={num_vehicles}_obstacle={num_obstacles}.pt")
            
            if load_model_prediction:
                y_data_path = os.path.join(folder, f"y_model_data_vehicle={i}_obstacle={j}.pt")  
            else:
                y_data_path = os.path.join(folder, f"y_GT_data_vehicle={i}_obstacle={j}.pt")
                
            batches_data_path = os.path.join(folder, f"batches_data_vehicle={num_vehicles}_obstacle={num_obstacles}.pt")
            
            if load_trajectory:
                trajectory_data_path = os.path.join(folder, f"trajectory_data_vehicle={i}_obstacle={j}.pt")
                if not os.path.exists(trajectory_data_path):
                    continue
            
            if not (os.path.exists(X_data_path) and os.path.exists(y_data_path) and os.path.exists(batches_data_path)):
                continue
            
            new_X_data = torch.load(X_data_path).float()
            new_y_data = torch.load(y_data_path).float()
            new_batches_data = torch.load(batches_data_path).long()
            
            if isinstance(horizon, int) and horizon>0 and 2*horizon<new_y_data.shape[-1]:
                new_y_data = new_y_data[:,:2*horizon]
            
            if load_trajectory:
                new_trajectory_data = torch.load(trajectory_data_path).long()
            
            
            if os.path.exists(X_data_path) and os.path.exists(y_data_path) and os.path.exists(batches_data_path):
                if (i+j,i) in data.keys():
                    
                    if load_trajectory:
                        X_data, y_data, batches_data, trajectory_data = data[(i+j,i)]
                    else:
                        X_data, y_data, batches_data = data[(i+j,i)]
                    
                    X_data = torch.cat((X_data, new_X_data))
                    y_data = torch.cat((y_data, new_y_data))
                    batches_data = torch.cat((batches_data, new_batches_data))
                    
                    if load_trajectory:
                        trajectory_data = torch.cat((trajectory_data, new_trajectory_data[1:]+trajectory_data[-1]))
                        data[(i+j,i)] = [X_data, y_data, batches_data, trajectory_data]
                    
                    else:
                        data[(i+j,i)] = [X_data, y_data, batches_data]
                    
                else:
                    
                    if load_trajectory:
                        data[(i+j,i)] = [new_X_data, new_y_data, new_batches_data, new_trajectory_data]
                    else:
                        data[(i+j,i)] = [new_X_data, new_y_data, new_batches_data]

        # if len(data) == 0:
        #     print(f"The data with {num_vehicles} vehicles and {num_obstacles} obstacles does not exist!")
         
    return data

def load_train_data(num_vehicles, num_obstacles, load_all_simpler=True, folders="./data/test_dataset", 
                    lim_length=None, random_permutation=False):
    
    if isinstance(folders, str):
        folders = [folders]
    else:
        assert isinstance(folders, list), \
            "invalid input of data folders, should be string of list or string"
    
    data = {}
    
    if load_all_simpler:
        # load all simpler case with less vehicles and obstacles
        for i in range(1, num_vehicles+1):
            for j in range(num_obstacles+1):
                
                X_data = torch.empty((0,i+j,8))
                
                for folder in folders:
                    
                    X_data_path = os.path.join(folder, f"train_data_vehicle={i}_obstacle={j}.pt")
    
                    if os.path.exists(X_data_path):
                        X_data = torch.cat((X_data, torch.load(X_data_path).float()))
                
                if isinstance(lim_length,int) and lim_length<len(X_data):
                    
                    if random_permutation:
                        rand_perm = np.random.permutation(len(X_data))
                        X_data = X_data[rand_perm]
                    
                    X_data = X_data[:lim_length]    
                
                if len(X_data)>0:
                    batches_data = torch.tensor([[i+j,i]]).repeat((len(X_data), 1))
                    data[(i+j,i)] = [X_data.flatten(0,1), batches_data]          
                        
    else:
        # load only the case with of given num of vehicles and obstacles
        
        i = num_vehicles
        j = num_obstacles
        
        X_data = torch.empty((0,i+j,8))
        
        for folder in folders:
            
            X_data_path = os.path.join(folder, f"train_data_vehicle={i}_obstacle={j}.pt")

            if os.path.exists(X_data_path):
                X_data = torch.cat((X_data, torch.load(X_data_path).float()))
                
        if isinstance(lim_length,int) and lim_length<len(X_data):
            
            if random_permutation:
                rand_perm = np.random.permutation(len(X_data))
                X_data = X_data[rand_perm]
            
            X_data = X_data[:lim_length]
        
        batches_data = torch.tensor([[i+j,i]]).repeat((len(X_data), 1))
        data[(i+j,i)] = [X_data.flatten(0,1), batches_data]    

        assert len(X_data) > 0,\
                f"The data with {num_vehicles} vehicles and {num_obstacles} obstacles does not exist!"
        
        
    return data

def load_test_data(num_vehicles, num_obstacles, load_all_simpler=True, folders="./data/test_dataset", 
                   lim_length=None, random_permutation=False):
    
    if isinstance(folders, str):
        folders = [folders]
    else:
        assert isinstance(folders, list), \
            "invalid input of data folders, should be string of list or string"
    
    data = []
    
    if load_all_simpler:
        # load all simpler case with less vehicles and obstacles
        for i in range(1, num_vehicles+1):
            for j in range(num_obstacles+1):
                
                test_data = torch.empty((0,i+j,8))
                
                for folder in folders:
                    
                    test_data_path = os.path.join(folder, f"test_data_vehicle={i}_obstacle={j}.pt")
    
                    if os.path.exists(test_data_path):
                        test_data = torch.cat((test_data, torch.load(test_data_path).float()))
                
                if isinstance(lim_length,int) and lim_length<len(test_data):
                    
                    if random_permutation:
                        rand_perm = np.random.permutation(len(test_data))
                        test_data = test_data[rand_perm]
                
                    test_data = test_data[:lim_length]    
                
                starts = test_data[:,:i,:4]
                targets = test_data[:,:i, 4:7]
                obstacles = test_data[:,i:i+j,4:7]
                problem_mark = torch.tensor([i, j]).long().repeat((len(test_data),1))
                
                data.extend(list(zip(starts,targets,obstacles,problem_mark)))            
                        
    else:
        # load only the case with of given num of vehicles and obstacles
        
        i = num_vehicles
        j = num_obstacles
        
        test_data = torch.empty((0,i+j,8))
        
        for folder in folders:
            
            test_data_path = os.path.join(folder, f"test_data_vehicle={i}_obstacle={j}.pt")

            if os.path.exists(test_data_path):
                test_data = torch.cat((test_data, torch.load(test_data_path).float()))
        
               
        if isinstance(lim_length,int) and lim_length<len(test_data):
            
            if random_permutation:
                rand_perm = np.random.permutation(len(test_data))
                test_data = test_data[rand_perm]
                        
            test_data = test_data[:lim_length]
        
        starts = test_data[:,:i,:4]
        targets = test_data[:,:i, 4:7]
        obstacles = test_data[:,i:i+j,4:7]
        
        problem_mark = torch.tensor([i, j]).long().repeat((len(test_data),1))
                
        data.extend(list(zip(starts,targets,obstacles,problem_mark)))


        assert len(data) > 0,\
                f"The data with {num_vehicles} vehicles and {num_obstacles} obstacles does not exist!"
        
        
    return data


# need to be changed
def change_to_relative_frame(data, num_vehicles, num_obstacles):
    ''' The function to change the goal of the vehicle and the obstacle position to the local frame of vehicle'''
    assert num_vehicles == 1, "Only one vehicle mode can use relative frame mode!"
    
    data_reshape = data.reshape(-1, num_vehicles+num_obstacles, 8).float()
    vehicles = data_reshape[:,0,:]
    obstacles = data_reshape[:,1:,:] 
    
    rotation_matrices = torch.zeros((len(data_reshape), 2, 2))
    rotation_matrices[:, 0, 0] = torch.cos(vehicles[:, 2])
    rotation_matrices[:, 0, 1] = torch.sin(vehicles[:, 2])
    rotation_matrices[:, 1, 0] = -torch.sin(vehicles[:, 2])
    rotation_matrices[:, 1, 1] = torch.cos(vehicles[:, 2])
    
    relative_points = (vehicles[:, 4:7]-vehicles[:, :3])
    relative_points[:,:2] = torch.matmul(rotation_matrices, relative_points[:,:2,None]).squeeze(-1)
    vehicles[:, 4:7] = relative_points
    
    if num_obstacles>0:
    
        relative_points = (obstacles[:, :, 4:6]-vehicles[:, None, :2])
        relative_points = torch.matmul(rotation_matrices[:, None, :,:], relative_points[:,:,:,None]).squeeze(-1)
        obstacles[:, :, 4:6] = relative_points
        
    data_reshape[:,0,:] = vehicles
    data_reshape[:,1:,:] = obstacles
    data = data_reshape.reshape(-1,8)

    return data


def split_train_valid(data):
    '''This function is to split the training data and validation data'''
    train_data = {}
    valid_data = {}
    
    for key, values in data.items():
        
        X, batches =  values
        X = X.reshape(len(batches), key[0], -1)
        
        train_index, valid_index = train_test_split(np.arange(len(batches)), test_size=0.2)
        
        train_X = X[train_index].reshape(len(train_index)*key[0], -1)
        train_batches = batches[train_index]
        
        valid_X = X[valid_index].reshape(len(valid_index)*key[0], -1)
        valid_batches = batches[valid_index]
        
        assert train_X.shape[1] == 8 and valid_X.shape[1] == 8, \
                "Error in train and valid data split!"
        
        train_data[key] = [train_X, train_batches]
        valid_data[key] = [valid_X, valid_batches]
        
    return train_data, valid_data
        

def get_angle_diff(angle_1, angle_2, mode="directed_vector"):
    
    if mode == "directed_vector":
        angle_diff_1 = (angle_1-angle_2)[...,None]%(2*np.pi)
        angle_diff_2 = 2*np.pi - angle_diff_1
        angle_diff = np.amin(np.concatenate((angle_diff_1,angle_diff_2), axis=-1), axis=-1)
        
    elif mode =="undirected_vector":
        angle_diff_1 = (angle_1-angle_2)[...,None]%(2*np.pi)
        angle_diff_2 = 2*np.pi - angle_diff_1
        angle_diff_3 = (angle_1+angle_2)[...,None]%(2*np.pi)
        angle_diff_4 = 2*np.pi - angle_diff_3
        angle_diff = np.amin(np.concatenate((angle_diff_1,angle_diff_2,angle_diff_3,angle_diff_4), axis=-1), axis=-1)
    
    return angle_diff


class GNN_Dataset(Dataset):
    def __init__(self, data, augmentation=True, sample_each_case=None, horizon=1):
        
        self.augmentation = augmentation
        self.dt = 0.2
        self.data = []
        
        for key, value in data.items():
            
            len_batch, len_vehicle = key
            len_obstacle = len_batch-len_vehicle
            X, batch = value
            
            assert len(X) == sum(batch[:,0])
            
            N = len(batch)
            
            if N==0 : continue
            
            X = X.reshape(N, len_batch, 8)
            
            batch = batch.long()
    
            obstacle_random_angle = torch.from_numpy(np.random.uniform(low=-np.pi, high=np.pi, size=(N, len_batch-len_vehicle)))
            
            X[:,len_vehicle:len_batch,:2] = X[:,len_vehicle:len_batch,4:6]
            X[:,len_vehicle:len_batch,2] = obstacle_random_angle
                  
            X[:,len_vehicle:len_batch,-1] = X[:,len_vehicle:len_batch,-2]
            X[:,len_vehicle:len_batch,-2] = obstacle_random_angle
            
            X[:,:,2] = (X[:,:,2]+pi)%(2*pi)-pi
            X[:,:,-2] = (X[:,:,-2]+pi)%(2*pi)-pi
            
            if isinstance(sample_each_case, int) and sample_each_case>0:
                samples = list(zip(X,batch))
                random.shuffle(samples)
                self.data.extend(samples[:min(sample_each_case, len(samples))])
            else:
                self.data.extend(list(zip(X,batch)))
        
        random.shuffle(self.data)
                  
    def transform_coordinate_one(self, X):
        
        coord_min = torch.amin(torch.cat((X[:,:2], X[:,4:6])), dim=0)
        coord_max = torch.amax(torch.cat((X[:,:2], X[:,4:6])), dim=0)
        coord_center = (coord_max+coord_min)/2
        
        theta = np.random.uniform(low=-np.pi, high=np.pi)
        R = torch.tensor([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]]).float()
        
        X[:,:2] = (R@((X[:,:2]-coord_center)[:,:,None])).squeeze(-1)
        X[:,4:6] = (R@((X[:,4:6]-coord_center)[:,:,None])).squeeze(-1)
        
        X[:,2] = (X[:,2] + theta + np.pi)%(2*np.pi) - np.pi
        X[:,6] = (X[:,6] + theta + np.pi)%(2*np.pi) - np.pi
        
        coord_min = torch.amin(torch.cat((X[:,:2], X[:,4:6])), dim=0)
        coord_max = torch.amax(torch.cat((X[:,:2], X[:,4:6])), dim=0)
        
        t_x = np.random.uniform(low=min(-10, -25-coord_min[0]), high=max(10, 25-coord_max[0]))
        t_y = np.random.uniform(low=min(-10, -25-coord_min[1]), high=max(10, 25-coord_max[1]))
        t = torch.tensor([t_x, t_y]).float()
        
        X[:,:2] = X[:,:2] + t
        X[:,4:6] = X[:,4:6] + t
            
        return X
    
    def transform_coordinate_batch(self, X):
        
        coord_min = torch.amin(torch.cat((X[:,:,:2], X[:,:,4:6]),dim=1), dim=1)
        coord_max = torch.amax(torch.cat((X[:,:,:2], X[:,:,4:6]),dim=1), dim=1)
        coord_center = (coord_max+coord_min)/2
        
        theta = torch.from_numpy(np.random.uniform(low=-np.pi, high=np.pi, size=(len(X)))).float()
        r11 = torch.cos(theta)
        r12 = -torch.sin(theta)
        r21 = torch.sin(theta)
        r22 = torch.cos(theta)
        
        R = torch.empty((len(X),1,2,2)).float()
        
        R[:,0,0,0] = r11
        R[:,0,0,1] = r12
        R[:,0,1,0] = r21
        R[:,0,1,1] = r22
        
        X[:,:,:2] = (R@((X[:,:,:2]-coord_center[:,None,:2])[:,:,:,None])).squeeze(-1)
        X[:,:,4:6] = (R@((X[:,:,4:6]-coord_center[:,None,:2])[:,:,:,None])).squeeze(-1)
        
        X[:,:,2] = (X[:,:,2] + theta[:,None] + np.pi)%(2*np.pi) - np.pi
        X[:,:,6] = (X[:,:,6] + theta[:,None] + np.pi)%(2*np.pi) - np.pi
        
        coord_min = torch.amin(torch.cat((X[:,:,:2], X[:,:,4:6]),dim=1), dim=1)
        coord_max = torch.amax(torch.cat((X[:,:,:2], X[:,:,4:6]),dim=1), dim=1)
        
        t_x = torch.from_numpy(np.random.uniform(low=torch.clip(-25-coord_min[:,0], max=-10), 
                                                    high=torch.clip(25-coord_max[:,0], min=10), 
                                                    )).float()
        t_y = torch.from_numpy(np.random.uniform(low=torch.clip(-25-coord_min[:,1], max=-10), 
                                                    high=torch.clip(25-coord_max[:,1], min=10), 
                                                    )).float()
        t = torch.empty((len(X),1,2)).float()
        t[:,0,0] = t_x
        t[:,0,1] = t_y
        
        X[:,:,:2] = X[:,:,:2] + t
        X[:,:,4:6] = X[:,:,4:6] + t
            
        return X
           
    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, index):
            return self.data[index]


def collect_fn(batch):
    X,n = zip(*batch)
    X = torch.cat(X)
    n = torch.stack(n)
    return (X,n)
    
    
class GNN_DataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False, num_workers = 0):
        super().__init__(dataset = dataset, 
                         batch_size = batch_size, 
                         shuffle = shuffle,
                         drop_last = drop_last,
                         collate_fn = collect_fn,
                         num_workers = num_workers,
                         )
