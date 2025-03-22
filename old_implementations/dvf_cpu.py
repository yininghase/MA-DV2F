import numpy as np
from numpy import pi
import torch

def dynamic_velocity_field(starts, targets, obstacles, dt=0.2, 
                           pedal_lim=1, steering_angle_lim=0.8, 
                           safe_distance=1.5, parking_distance=5, 
                           default_v=2.5, vehicle_radius=1.5,
                           pos_tolerance=0.1, ang_tolerance=0.2):
    
    N_veh = starts.shape[1]
    N_obst = obstacles.shape[1]
    
    uni_orient_starts = torch.stack((torch.cos(starts[...,2]), 
                                        torch.sin(starts[...,2])), dim=-1)
    
    uni_orient_targets = torch.stack((torch.cos(targets[...,2]), 
                                        torch.sin(targets[...,2])), dim=-1)
    
    next_xy = uni_orient_starts*starts[...,3:4]*dt + starts[...,:2]
    
    vec_to_targets = targets[:,:,:2]-next_xy[:,:,:2]
    dist_to_targets = torch.norm(vec_to_targets, dim=-1, p=2, keepdim=True)
    uni_vec_targets = vec_to_targets/(dist_to_targets+1e-8)
    
    vec_avoid_collision = torch.zeros((len(next_xy), N_veh, 2))
    
    for i in range(N_obst):
        vec_to_obstacles = obstacles[:,i:i+1,:2]-next_xy[:,:,:2]
        dist_to_obstacles_center = torch.norm(vec_to_obstacles, dim=-1, p=2, keepdim=True)
        dist_to_obstacles = dist_to_obstacles_center-obstacles[:,i:i+1,2:3]-vehicle_radius
        uni_vec_obstacles = vec_to_obstacles/(dist_to_obstacles_center+1e-8)
        
        dynamic_safe_distance = safe_distance + starts[...,3:4].abs()
        
        mask_collision = torch.sum(vec_to_obstacles*vec_to_targets, dim=-1)>0
        mask_collision = mask_collision & (dist_to_obstacles<=dynamic_safe_distance)[...,0]
        
        vec_avoid_collision += torch.clip(dist_to_obstacles-dynamic_safe_distance, 
                                          min=-dynamic_safe_distance, 
                                          max=torch.zeros_like(dynamic_safe_distance))*uni_vec_obstacles
                    
        vec_to_obstacles_3d = torch.cat((uni_vec_obstacles, torch.zeros((len(next_xy), N_veh, 1))), dim=-1)
        vec_direct = torch.zeros_like(vec_to_obstacles_3d)
        vec_direct[...,-1] = 1
        
        vec_around_obstacles = torch.cross(vec_direct, vec_to_obstacles_3d, dim=-1)[...,:2]
        vec_around_obstacles = vec_around_obstacles/(torch.norm(vec_around_obstacles, dim=-1, p=2, keepdim=True)+1e-8)
        zero_vec_mask = torch.all(vec_around_obstacles==0, dim=-1)
        vec_around_obstacles[zero_vec_mask] = torch.tensor([1.,0.])
        vec_avoid_collision[mask_collision] += (torch.clip(dist_to_obstacles, 
                                                            min=torch.zeros_like(dynamic_safe_distance), 
                                                            max=dynamic_safe_distance)*vec_around_obstacles)[mask_collision]
    
    for i in range(N_veh):
        for j in range(i+1, N_veh):
            vec_to_vehicles = next_xy[:,j,:2]-next_xy[:,i,:2]
            dist_to_vehicles_center = torch.norm(vec_to_vehicles, dim=-1, p=2, keepdim=True)
            dist_to_vehicles = dist_to_vehicles_center-2*vehicle_radius
            uni_vec_vehicles = vec_to_vehicles/(dist_to_vehicles_center+1e-8)
            
            dynamic_safe_distance = safe_distance + starts[:,i,3:4].abs() + starts[:,j,3:4].abs()
            
            mask_collision_1 = torch.sum(vec_to_vehicles*vec_to_targets[:,i,:], dim=-1)>0
            mask_collision_1 = mask_collision_1 & (dist_to_vehicles <= dynamic_safe_distance)[...,0]
            
            mask_collision_2 = torch.sum(-vec_to_vehicles*vec_to_targets[:,j,:], dim=-1)>0
            mask_collision_2 = mask_collision_2 & (dist_to_vehicles <= dynamic_safe_distance)[...,0]
            
            vec_avoid_collision[:,i,:] += torch.clip(dist_to_vehicles-dynamic_safe_distance, 
                                                     min=-dynamic_safe_distance, 
                                                     max=torch.zeros_like(dynamic_safe_distance))*uni_vec_vehicles
            vec_avoid_collision[:,j,:] -= torch.clip(dist_to_vehicles-(dynamic_safe_distance), 
                                                     min=-dynamic_safe_distance, 
                                                     max=torch.zeros_like(dynamic_safe_distance))*uni_vec_vehicles
            
            vec_to_vehicles_3d = torch.cat((uni_vec_vehicles, torch.zeros((len(next_xy), 1))), dim=-1)
            vec_direct = torch.zeros_like(vec_to_vehicles_3d)
            vec_direct[...,-1] = 1

            vec_around_vehicles = torch.cross(vec_direct, vec_to_vehicles_3d, dim=-1)[...,:2]
            vec_around_vehicles = vec_around_vehicles/(torch.norm(vec_around_vehicles, dim=-1, p=2, keepdim=True)+1e-8)
            vec_avoid_collision[:,i,:][mask_collision_1] += (torch.clip(dist_to_vehicles, 
                                                                        min=torch.zeros_like(dynamic_safe_distance), 
                                                                        max=dynamic_safe_distance)*\
                                                                        vec_around_vehicles)[mask_collision_1]
            
            vec_around_vehicles = torch.cross(vec_direct, -vec_to_vehicles_3d, dim=-1)[...,:2]
            vec_around_vehicles = vec_around_vehicles/(torch.norm(vec_around_vehicles, dim=-1, p=2, keepdim=True)+1e-8)
            vec_avoid_collision[:,j,:][mask_collision_2] += (torch.clip(dist_to_vehicles, 
                                                                        min=torch.zeros_like(dynamic_safe_distance), 
                                                                        max=dynamic_safe_distance)*\
                                                                        vec_around_vehicles)[mask_collision_2]
            
    vec_to_targets = uni_vec_targets.clone()
    
    mask_stop = (dist_to_targets<pos_tolerance)
    factor1 = torch.sum(vec_to_targets*uni_orient_targets, dim=-1, keepdim=True) 
    factor2 = torch.clip(dist_to_targets, min=0, max=parking_distance)/parking_distance
    factor2[~mask_stop] += 1
    factor = 2*((factor1>=0).float()-0.5)*factor2
    
    mask_parking = dist_to_targets[...,0]<=parking_distance
    vec_to_targets[mask_parking] = (uni_orient_targets+factor*vec_to_targets)[mask_parking]
    
    factor = torch.sum(uni_orient_starts*uni_vec_targets, dim=-1, keepdim=True)
    factor = ((factor>=0).float()-0.5)*2
    mask_parking_margin = (dist_to_targets[...,0]>parking_distance) & \
                  (dist_to_targets[...,0]<=parking_distance+0.5*default_v**2)
    vec_to_targets[mask_parking_margin] = (vec_to_targets*factor)[mask_parking_margin]
    
    vec_to_targets = vec_to_targets/(torch.norm(vec_to_targets, dim=-1, p=2, keepdim=True)+1e-8)
    
    vec_ref_orient = vec_to_targets+vec_avoid_collision
    vec_ref_orient = vec_ref_orient/(torch.norm(vec_ref_orient, dim=-1, p=2, keepdim=True)+1e-8)
    
    ref_orent = torch.arctan2(vec_ref_orient[...,1], vec_ref_orient[...,0])
    
    theta_range = 0.99*starts[...,3].abs()*np.tan(steering_angle_lim)*dt/2
    
    diff_orient_1 = (ref_orent-starts[...,2])%(2*pi)
    diff_orient_2 = (starts[...,2]-ref_orent)%(2*pi)
    
    mask_diff_orient = diff_orient_1<=diff_orient_2
    
    diff_orient_1 = torch.clip(diff_orient_1, min=torch.zeros_like(theta_range), max=theta_range)
    diff_orient_2 = torch.clip(diff_orient_2, min=torch.zeros_like(theta_range), max=theta_range)
    diff_orient = torch.amin(torch.stack((diff_orient_1, diff_orient_2), dim=-1), dim=-1)
    
    ref_orent = torch.empty_like(starts[...,2])
    ref_orent[mask_diff_orient] = starts[...,2][mask_diff_orient] + diff_orient_1[mask_diff_orient]
    ref_orent[~mask_diff_orient] = starts[...,2][~mask_diff_orient] - diff_orient_2[~mask_diff_orient]
    ref_steer = torch.arctan(2*(ref_orent-starts[...,2])/(dt*starts[...,3]+1e-8))
    
    ref_orent = (ref_orent+pi)%(2*pi)-pi
    
    mask_stop = mask_stop[...,0] & (diff_orient<ang_tolerance)
    
    ref_vel = torch.ones((len(next_xy), N_veh))*default_v
    factor = torch.clip(torch.clip(dist_to_targets[...,0]/parking_distance, min=0, max=1) + 
                         torch.clip(diff_orient/default_v, min=0, max=1), min=0, max=1)
    factor[~mask_stop] = factor[~mask_stop].sqrt()
    ref_vel *= factor
    
    mask_parking = mask_parking | mask_parking_margin
    mask_backwards = torch.sum(uni_vec_targets*uni_orient_starts, dim=-1) < -0.25
    ref_vel[mask_backwards & mask_parking] = -ref_vel[mask_backwards & mask_parking].abs()
    
    mask_forwards = torch.sum(uni_vec_targets*uni_orient_starts, dim=-1) > 0.25
    ref_vel[mask_forwards & mask_parking] = ref_vel[mask_forwards & mask_parking].abs()
    
    mask_remain = (~mask_forwards) & (~mask_backwards)
    vel_sign = ((starts[...,3]>=0).float()-0.5)*2
    ref_vel[mask_remain & mask_parking] = (ref_vel.abs()*vel_sign)[mask_remain & mask_parking]
    
    mask_backwards = torch.sum(vec_ref_orient*uni_orient_starts, dim=-1) < 0
    
    ref_vel[mask_backwards & (~mask_parking)] *= -1
    
    uni_ref_orent = torch.stack((torch.cos(ref_orent), 
                                    torch.sin(ref_orent)), dim=-1)
    
    ##### double check ####
    
    mask_forwards = torch.zeros((len(starts), N_veh), dtype=bool)
    mask_backwards = torch.zeros((len(starts), N_veh), dtype=bool)
    
    for i in range(N_obst):
        vec_to_obstacles = obstacles[:,i:i+1,:2]-next_xy[:,:,:2]
        dist_to_obstacles_center = torch.norm(vec_to_obstacles, dim=-1, p=2)
        dist_to_obstacles = dist_to_obstacles_center-obstacles[:,i:i+1,2]-vehicle_radius
        uni_vec_obstacles = vec_to_obstacles/(dist_to_obstacles_center[...,None]+1e-8)
        
        dynamic_safe_distance = safe_distance + starts[...,3:4].abs()
        mask_unsafe = dist_to_obstacles<(dynamic_safe_distance[...,0]-1)
        
        ref_cos_theta = torch.sum(uni_ref_orent*uni_vec_obstacles, dim=-1)
        
        mask_forwards = mask_forwards | (mask_unsafe & (ref_cos_theta<0))
        mask_backwards = mask_backwards | (mask_unsafe & (ref_cos_theta>0))
        

    for i in range(N_veh):
        for j in range(i+1, N_veh):
            vec_to_vehicles = next_xy[:,j,:2]-next_xy[:,i,:2]
            dist_to_vehicles_center = torch.norm(vec_to_vehicles, dim=-1, p=2)
            dist_to_vehicles = dist_to_vehicles_center-2*vehicle_radius
            uni_vec_vehicles = vec_to_vehicles/(dist_to_vehicles_center[...,None]+1e-8)
            
            dynamic_safe_distance = safe_distance + starts[:,i,3:4].abs() + starts[:,j,3:4].abs()           
            mask_unsafe = dist_to_vehicles<(dynamic_safe_distance[...,0]-1)
            
            ref_cos_theta_1 = torch.sum(uni_ref_orent[:,i]*uni_vec_vehicles, dim=-1)
            mask_forwards[:,i] = mask_forwards[:,i] | (mask_unsafe & (ref_cos_theta_1<0))
            mask_backwards[:,i] = mask_backwards[:,i] | (mask_unsafe & (ref_cos_theta_1>0))
            
            ref_cos_theta_2 = torch.sum(-uni_ref_orent[:,j]*uni_vec_vehicles, dim=-1)
            mask_forwards[:,j] = mask_forwards[:,j] | (mask_unsafe & (ref_cos_theta_2<0))
            mask_backwards[:,j] = mask_backwards[:,j] | (mask_unsafe & (ref_cos_theta_2>0))
            
    mask_zero = mask_forwards & mask_backwards
    
    ref_vel[mask_forwards] = ref_vel[mask_forwards].abs()
    ref_vel[mask_backwards] = -(ref_vel[mask_backwards].abs())
    ref_vel[mask_zero] = 0
    
    ref_vel = torch.clip(ref_vel, min=0.99*starts[...,3]-pedal_lim*dt, max=0.99*starts[...,3]+pedal_lim*dt)
    ref_pedal = (ref_vel-0.99*starts[...,3])/dt
    
    ref_states = torch.cat((next_xy, ref_orent[...,None], ref_vel[...,None]), dim=-1)
    ref_controls = torch.stack((ref_pedal, ref_steer), dim=-1)
    
    return ref_states, ref_controls 