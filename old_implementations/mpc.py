import numpy as np
from numpy import pi
from itertools import permutations
from data_process import get_angle_diff

class ModelPredictiveControl:
    def __init__(self, config):
        self.horizon = config["horizon"]
        self.dt = 0.2
        self.starts = config["starts"]
        self.targets= config["targets"]
        self.obstacles = config["obstacles"] 
        self.control_init = config["control init"] 
        self.dis_cost = config["distance cost"] 
        self.ang_cost = config["angle cost"] 
        self.col_cost = config["collision cost"] 
        self.col_radius = config["collision radius"] 
        self.obs_cost = config["obstacle cost"] 
        self.obs_radius = config["obstacle radius"]
        self.vel_cost = config["velocity cost"] 
        self.vel_limit = config["velocity limit"] 
        
        self.smooth_cost = config["smoothness cost"]
        self.travel_dist_cost = config["travel dist cost"]

        
        assert len(self.starts) == len(self.targets)
        self.num_vehicle = len(self.starts)
        self.num_obstacle = len(self.obstacles)
        
        self.vehicle_pair = np.array(list(permutations(range(self.num_vehicle), 2)))
        # print()

    def plant_model(self, prev_state, dt, control):
        
        x_t = prev_state[...,0]
        y_t = prev_state[...,1]
        psi_t = prev_state[...,2]
        v_t = prev_state[...,3]
        
        pedal = control[...,0]
        steering = control[...,1]

        # Vehicle Dynamic Equation
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

    def cost_function(self, u, *args):
        u = u.reshape(self.horizon, self.num_vehicle, 2)
        
        state = np.array(args[0])
        ref = np.array(args[1])
        
        state_history =np.array([state])
        
        for i in range(self.horizon):
            control = u[i]
            state = self.plant_model(state, self.dt, control)
            state_history= np.concatenate((state_history, state[None,...]))


        # target cost 
        pos_diff = np.linalg.norm(ref[:,:2]-state_history[:,:,:2], axis=-1, ord=2)        
        cost = np.sum(pos_diff[1:])*self.dis_cost 
        
        angle_diff = get_angle_diff(state_history[1:,:,2], ref[:,2]) 
        cost += np.sum(angle_diff)*self.ang_cost
        
        # obstacle cost 
        if self.obs_cost > 0 and self.num_obstacle > 0:
            dist = np.linalg.norm(self.obstacles[:,:2]-state_history[1:,:,None,:2], axis=-1, ord=2)-self.obstacles[:,2]
            dist = np.clip(dist, a_min=0, a_max=None) + 1e-8
            cost += np.sum((1/dist - 1/self.obs_radius) * (dist < self.obs_radius)) * self.obs_cost
        
        # smoothness cost (steering angle and pedal input)
        if self.smooth_cost > 0:
            cost += np.sum(np.diff(u, axis=0)) * self.smooth_cost
        
        # travel distance cost  
        if self.travel_dist_cost > 0:
            cost += np.sum(np.linalg.norm(np.diff(state_history[...:2], axis=0), ord=2, axis=-1)) * self.travel_dist_cost
            
        if self.vel_cost > 0:
            cost += np.sum(np.clip(np.abs(state_history[1:,:,3])-self.vel_limit, a_min=0, a_max=None))*self.vel_cost
        
        # collision cost  
        if self.col_cost > 0 and self.num_vehicle > 1:
            dist = np.linalg.norm(state_history[1:, self.vehicle_pair[:,1], :2] - state_history[1:, self.vehicle_pair[:,0], :2], ord=2, axis=-1)            
            cost += np.sum((1/dist-1/self.col_radius) * (dist < self.col_radius))*self.col_cost            
            
        return cost
    