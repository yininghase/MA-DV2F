import os
import torch
import copy

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

from calculate_metrics import (check_collision_rectangular_circle, 
                               check_collision_rectangular_rectangular)

def get_random_obstacle(center, radius, color, circle_obstacle=True):
    
    if circle_obstacle or np.random.rand()>0.9:
        obstacle = mpatches.Circle(center, radius, color=color, fill=True)
    
    elif np.random.rand()>0.5:
        n = np.random.randint(low=3, high=10)
        a = np.random.uniform(low=0.75, high=1, size=n-2)
        a = np.concatenate([a, np.array([1,1])], axis=0)
        b = np.random.uniform(low=0, high=np.pi*2, size=n-2)
        c = np.random.uniform(low=0, high=np.pi)
        b = np.concatenate([b, np.array([c,-c])], axis=0)%(2*np.pi)
        
        rank = np.argsort(b, axis=0)
        a = a[rank]
        b = b[rank]
        a = np.clip(a*radius, a_min=1, a_max=None)
        vertice = np.stack([a*np.cos(b), a*np.sin(b)], axis=1) + center
        
        obstacle = mpatches.Polygon(vertice, closed=True, color=color, fill=True)
        
    else:
        
        n = np.random.randint(low=3, high=10)
        a = np.random.uniform(low=0.75, high=1, size=n-3)
        a = np.concatenate([a, np.array([1,1,1])], axis=0)
        b = np.random.uniform(low=0, high=np.pi*2, size=n-2)
        c1 = np.random.uniform(low=0, high=np.pi)
        c2 = np.random.uniform(low=np.pi, high=c1+np.pi)
        c1 = c1 + b[-1]
        c2 = c2 + b[-1]
        b = np.concatenate([b, np.array([c1,c1])], axis=0)
        
        rank = np.argsort(b, axis=0)
        a = a[rank]
        b = b[rank]
        a = np.clip(a*radius, a_min=1, a_max=None)
        vertice = np.stack([a*np.cos(b), a*np.sin(b)], axis=1) + center
        
        obstacle = mpatches.Polygon(vertice, closed=True, color=color, fill=True)
        
    return obstacle   


class Visualize_Trajectory:
    def __init__(self, config, show_attention=False):
        
        self.config = config
        
        num_colors = max(self.config["num of vehicles"],
                         self.config["num of obstacles"])
        cmap = plt.get_cmap('brg')
        self.cmap = cmap(np.linspace(0,1,num_colors))[...,:3]
        # self.cmap = np.concatenate((np.zeros((1,3)), self.cmap))
        
        self.show_attention = show_attention
        
        self.patch_obs = []
        for i, obs in enumerate(self.config["obstacles"]):
            self.patch_obs.append(get_random_obstacle(obs[:2], obs[2], color=self.cmap[i]))

    # starts: [num_vehicle, [x, y, psi]]
    # targets: [num_vehicle, [x, y, psi]]
    
    def base_plot(self, is_trajectory):
        
        if self.show_attention:
            self.fig = plt.figure(figsize=(2*self.config["figure size"], 
                                        self.config["figure size"]))
            self.ax = self.fig.add_subplot(1,2,1)
            self.ax_ = self.fig.add_subplot(1,2,2)
            
        else:
            self.fig = plt.figure(figsize=(self.config["figure size"], 
                                        self.config["figure size"]))
            self.ax = self.fig.add_subplot()
        
        starts = self.config["starts"]
        targets = self.config["targets"]
        obstacles = self.config["obstacles"]
        
        x_coord_min = np.concatenate((obstacles[:,0]-obstacles[:,2], 
                                      starts[:,0], targets[:,0]), axis=0)
        x_coord_max = np.concatenate((obstacles[:,0]+obstacles[:,2], 
                                      starts[:,0], targets[:,0]), axis=0)
        y_coord_min = np.concatenate((obstacles[:,1]-obstacles[:,2],  
                                      starts[:,1], targets[:,1]), axis=0)
        y_coord_max = np.concatenate((obstacles[:,1]+obstacles[:,2],  
                                      starts[:,1], targets[:,1]), axis=0)
        x_min = np.floor(np.amin(x_coord_min))
        x_max = np.ceil(np.amax(x_coord_max))
        y_min = np.floor(np.amin(y_coord_min))
        y_max = np.ceil(np.amax(y_coord_max))
        
        center = [(x_max+x_min)/2, (y_max+y_min)/2]
        
        fig_limit = max(x_max-x_min, y_max-y_min)
        fig_limit = ((np.ceil(fig_limit/5)+4)*5)/2
            
        self.ax.set_xlim([-fig_limit+center[0], fig_limit+center[0]])
        self.ax.set_ylim([-fig_limit+center[1], fig_limit+center[1]])
        self.ax.set_xticks(np.linspace(-fig_limit+center[0], fig_limit+center[0], 
                                        num=11, endpoint=True))
        self.ax.set_yticks(np.linspace(-fig_limit+center[1], fig_limit+center[1], 
                                        num=11, endpoint=True))
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        
        self.patch_vehicles = []
        self.patch_vehicles_arrow = []
        self.predicts_opt = []
        self.predicts_model = []
        self.predicts_init = []
        
        starts_new = self.car_patch_pos(starts)
        targets_new = self.car_patch_pos(targets)

        for i in range(self.config["num of vehicles"]):
            # cars
            
            patch_car = mpatches.Rectangle([0,0], 
                                            self.config["car size"][0], 
                                            self.config["car size"][1], 
                                            color=self.cmap[i])
            patch_car.set_xy(starts_new[i,:2])
            patch_car.angle = np.rad2deg(starts_new[i,2])-90
            self.patch_vehicles.append(patch_car)
            
            patch_car_arrow = mpatches.FancyArrow(starts[i,0]-0.9*np.cos(starts[i,2]), 
                                                  starts[i,1]-0.9*np.sin(starts[i,2]), 
                                                  1.5*np.cos(starts[i,2]), 
                                                  1.5*np.sin(starts[i,2]), 
                                                  width=0.1, color='w')
            self.patch_vehicles_arrow.append(patch_car_arrow)

            patch_goal = mpatches.Rectangle([0,0], 
                                            self.config["car size"][0], 
                                            self.config["car size"][1], 
                                            color=self.cmap[i], 
                                            ls='dashdot', fill=False)
            
            patch_goal.set_xy(targets_new[i,:2])
            patch_goal.angle = np.rad2deg(targets_new[i,2])-90
            
            patch_goal_arrow = mpatches.FancyArrow(targets[i,0]-0.9*np.cos(targets[i,2]), 
                                                   targets[i,1]-0.9*np.sin(targets[i,2]), 
                                                   1.5*np.cos(targets[i,2]), 
                                                   1.5*np.sin(targets[i,2]), 
                                                   width=0.1, 
                                                   color=self.cmap[i])

            self.ax.add_patch(patch_car)
            self.ax.add_patch(patch_goal)
            self.ax.add_patch(patch_car_arrow)
            self.ax.add_patch(patch_goal_arrow)

            self.frame = plt.text(12, 12, "", fontsize=15)

            # trajectories
            if self.config["run other algorithm"] == 'mpc':
                if is_trajectory:
                    predict_opt, = self.ax.plot([], [], 'r--', linewidth=1)
                elif i == 0:
                    predict_opt, = self.ax.plot([], [], 'r--', linewidth=1, label="Optimization")
                else:
                    predict_opt, = self.ax.plot([], [], 'r--', linewidth=1, label="_Optimization")
                self.predicts_opt.append(predict_opt)
            
                if self.config["control init"] is not None:
                    if is_trajectory:
                        predict_init, = self.ax.plot([], [], 'b--', linewidth=1)
                    elif i == 0:
                        predict_init, = self.ax.plot([], [], 'b--', linewidth=1, label="Initialization")
                    else:
                        predict_init, = self.ax.plot([], [], 'b--', linewidth=1, label="_Initialization")
                    self.predicts_init.append(predict_init)
    
            vehicle_mark, = self.ax.plot([], [], color=self.cmap[i], marker='.', linewidth=1, label=f"vehicle {i+1}")
        
        for i, obs in enumerate(obstacles):
            self.ax.add_patch(copy.copy(self.patch_obs[i]))
            obstacle_mark, = self.ax.plot([], [], color=self.cmap[i], marker='.', linewidth=1, label=f"obstacle {i+1}")
        
        self.fig.tight_layout()
    
    def create_video(self, data, predict_opt, predict_model, attention=None):
        self.base_plot(is_trajectory=False)
        self.data = data
        
        if self.config["is model"]:
            self.predict_model = predict_model
            
        if self.config["run other algorithm"] == 'mpc':
            self.predict_opt = predict_opt 
            
        self.attention = attention
            
        car_animation = animation.FuncAnimation(self.fig, self.update_plot, frames=range(len(data)-1), interval=100, repeat=True, blit=False)
        
        if self.config["save plot"]:
            if not os.path.exists(self.config["plot folder"]):
                os.makedirs(self.config["plot folder"])
                
            car_animation.save(os.path.join(self.config["plot folder"], 
                                            self.config["name"] + ".gif"))
            
        if self.config["show plot"]:
            plt.show()
        
        plt.close()


    def update_plot(self, num):
        
        data = self.data[num,...]               

        for i in range(self.config["num of vehicles"]):
            # vehicle
            data_ = self.car_patch_pos(data[i][None,...])
            self.patch_vehicles[i].set_xy(data_[0,:2])
            self.patch_vehicles[i].angle = np.rad2deg(data_[0,2])-90
            self.patch_vehicles_arrow[i].set_data(x=data[i,0]-0.9*np.cos(data[i,2]), 
                                                    y=data[i,1]-0.9*np.sin(data[i,2]), 
                                                    dx=1.5*np.cos(data[i,2]), 
                                                    dy=1.5*np.sin(data[i,2]))
            
            
            if self.config["run other algorithm"] == 'mpc':
                self.predicts_opt[i].set_data(self.predict_opt[num, :, i, 0], self.predict_opt[num, :, i, 1])
        
        if self.show_attention and self.attention is not None:
            self.ax_.imshow(self.attention[num], vmin=-2.5, vmax=2.5, cmap="gray")
            self.ax_.set_xticks(ticks=[i for i in range(self.config["num of vehicles"]+self.config["num of obstacles"])],
                                labels = [f"vehicle {i+1}" for i in range(self.config["num of vehicles"])] + \
                                         [f"obstacle {i+1}" for i in range(self.config["num of obstacles"])])
            self.ax_.set_yticks(ticks=[i for i in range(self.config["num of vehicles"])],
                                labels = [f"vehicle {i+1}" for i in range(self.config["num of vehicles"])])
            
    def plot_trajectory(self, points):
        self.base_plot(is_trajectory=True)
        max_time = points.shape[0]
        
        for i in range(self.config["num of vehicles"]):
            veh_points = points[:, i, :2][:,None,:]
            segments = np.concatenate([veh_points[:-1], veh_points[1:]], axis=1)
            norm = plt.Normalize(0, max_time)
            lc = LineCollection(segments, cmap="viridis", norm=norm)
            lc.set_array(range(points.shape[0]))
            lc.set_linewidth(1.0)
            line = self.ax.add_collection(lc)
        
        collision = np.zeros((points.shape[0], points.shape[1]), dtype=bool)
        
        for i in range(self.config["num of vehicles"]-1):
            for j in range(i+1, self.config["num of vehicles"]):
                collisions_ij = check_collision_rectangular_rectangular(torch.from_numpy(points[:,i,:]).float(), 
                                                                        torch.from_numpy(points[:,j,:]).float(), 
                                                                        vehicle_size=self.config["car size"]).numpy()
                collision[collisions_ij,i]=True
                collision[collisions_ij,j]=True
        
        for i in range(self.config["num of vehicles"]):
            for j in range(self.config["num of obstacles"]):
                obstacle_j = self.config["obstacles"][j]
                obstacle_j = np.concatenate((np.zeros(4),obstacle_j,np.ones(1)))[None,:]
                
                collisions_ij = check_collision_rectangular_circle(torch.from_numpy(points[:,i,:]).float(), 
                                                                   torch.from_numpy(obstacle_j).float(), 
                                                                   vehicle_size=self.config["car size"]).numpy()
                
                collision[collisions_ij,i]=True
               
        cbar = self.fig.colorbar(line, ax=self.ax, fraction=0.05, pad=0.01)
        cbar.ax.set_ylabel("Timestep", fontsize=15)
        
        if np.sum(collision)>0:
            self.ax.scatter(points[collision][:,0], points[collision][:,1], s=15, c="r", marker="o")
        
        if self.config["save plot"]:
            if not os.path.exists(self.config["plot folder"]):
                os.makedirs(self.config["plot folder"])
            plt.savefig(os.path.join(self.config["plot folder"], 
                                     self.config["name"]+".png"), bbox_inches='tight')
        
        if self.config["show plot"]:
            plt.show()

        plt.close()
        
    def car_patch_pos(self, posture):
        
        posture_new = posture.copy()
        
        posture_new[...,0] = posture[...,0] - np.sin(posture[...,2])*(self.config["car size"][0]/2) \
                                        - np.cos(posture[...,2])*(self.config["car size"][1]/2)
        posture_new[...,1] = posture[...,1] + np.cos(posture[...,2])*(self.config["car size"][0]/2) \
                                        - np.sin(posture[...,2])*(self.config["car size"][1]/2)
        
        return posture_new
    
    def calculate_cost(self, coordinates, targets):
        
        dist_cost = self.config["distance_cost"]
        obst_cost = self.config["obstacle_cost"]
        obs_radius = self.config["obstacle_radius"]
        obstacles = self.config["obstacles"]
        num_obstacles = self.config["num of obstacles"]
        
        loss = np.linalg.norm(coordinates - targets[None, None, :2], ord=2, axis=-1)*dist_cost
        
        if  obst_cost > 0 and num_obstacles > 0:
            dist = np.linalg.norm(coordinates[:,:,None,:]-obstacles[None,None,:,:2], ord=2, axis=-1)-obstacles[None,None,:,2]-obs_radius
            dist = (np.clip(-dist, a_min=0, a_max=None))**2
            loss += np.sum(dist, axis=-1) * obst_cost
        
        return loss


class Visualize_Velocity_Field_Single_Frame:
    def __init__(self, config):
        
        self.config = config 
        self.dt = 0.2
        self.ego_vehicle = np.array(config['ego vehicle'])
        self.other_vehicles = np.array(config['other vehicles'])
        self.obstacles = np.array(config['obstacles'])
        self.target= np.array(config['target'])
        self.car_size = config['car size'] #car_size=[1.0,2.5]
        self.fig_size = config['figure size']
        self.safe_distance = config.get('safe distance', 1.5)
        self.parking_distance = config.get('parking distance', 5)
        self.default_ref_vel = config.get('default reference velocity', 2.5)
        self.vehicle_radius = config.get('vehicle radius', 1.5)
        
        assert 2*self.vehicle_radius> np.linalg.norm(np.array(self.car_size), ord=2, axis=-1), \
               "Vehicle radius can not cover car size!"
        
        self.num_obstacles = len(self.obstacles)
        self.num_vehicles = len(self.other_vehicles)
        
        self.dyn_safe_dist_obs = self.safe_distance + np.abs(self.ego_vehicle[3])
        self.dyn_safe_dist_veh = self.safe_distance + np.abs(self.ego_vehicle[3])+np.abs(self.other_vehicles[:,3])
        self.dyn_parking_dist = self.parking_distance + 0.5*self.default_ref_vel**2
        
        x_coord_min = np.concatenate((self.obstacles[:,0]-self.obstacles[:,2]-self.dyn_safe_dist_obs-self.vehicle_radius, 
                                      self.other_vehicles[:,0]-self.dyn_safe_dist_veh-2*self.vehicle_radius, 
                                      self.target[0:1]-self.dyn_parking_dist, 
                                      self.ego_vehicle[0:1]), axis=0)
        x_coord_max = np.concatenate((self.obstacles[:,0]+self.obstacles[:,2]+self.dyn_safe_dist_obs+self.vehicle_radius, 
                                      self.other_vehicles[:,0]+self.dyn_safe_dist_veh+2*self.vehicle_radius, 
                                      self.target[0:1]+self.dyn_parking_dist, 
                                      self.ego_vehicle[0:1]), axis=0)
        y_coord_min = np.concatenate((self.obstacles[:,1]-self.obstacles[:,2]-self.dyn_safe_dist_obs-self.vehicle_radius,  
                                      self.other_vehicles[:,1]-self.dyn_safe_dist_veh-2*self.vehicle_radius, 
                                      self.target[1:2]-self.dyn_parking_dist,  
                                      self.ego_vehicle[1:2]), axis=0)
        y_coord_max = np.concatenate((self.obstacles[:,1]+self.obstacles[:,2]+self.dyn_safe_dist_obs+self.vehicle_radius,  
                                      self.other_vehicles[:,1]+self.dyn_safe_dist_veh+2*self.vehicle_radius, 
                                      self.target[1:2]+self.dyn_parking_dist,  
                                      self.ego_vehicle[1:2]), axis=0)
        x_min = np.floor(np.amin(x_coord_min)/2.5)*2.5
        x_max = np.ceil(np.amax(x_coord_max)/2.5)*2.5
        y_min = np.floor(np.amin(y_coord_min)/2.5)*2.5
        y_max = np.ceil(np.amax(y_coord_max)/2.5)*2.5
        
        center = np.array([(x_max+x_min)/2, (y_max+y_min)/2])
        self.ego_vehicle[:2] -= center
        self.target[:2] -= center
        self.other_vehicles[:,:2] -= center[None,:]
        self.obstacles[:,:2] -= center[None,:]
        self.fig_range = max((x_max-x_min)/2, (y_max-y_min)/2)
        self.ego_next = [self.ego_vehicle[0]+self.ego_vehicle[3]*np.cos(self.ego_vehicle[2])*self.dt,
                         self.ego_vehicle[1]+self.ego_vehicle[3]*np.sin(self.ego_vehicle[2])*self.dt,
                         self.ego_vehicle[2]+self.ego_vehicle[3]*self.dt*np.tan(-0.8)/2.0,
                         self.ego_vehicle[2]+self.ego_vehicle[3]*self.dt*np.tan(0.8)/2.0]
        
        self.num_points_x_y = 17
        self.num_point_around = 16
        self.vector_scale = 0.4*self.fig_range/(self.num_points_x_y-1)
        
        num_colors = len(self.other_vehicles)+len(self.obstacles)
        self.cmap = np.array([[0,0,0]])
        
        if num_colors>0:
            cmap = plt.get_cmap('brg')
            self.cmap = np.concatenate((self.cmap, cmap(np.linspace(0,1,num_colors))[...,:3]), axis=0)   


    def get_velocity_field(self, query_points, pos_tolerance=0.25):
            
            query_points = torch.from_numpy(query_points) 
            target = torch.from_numpy(self.target)
            vehicles = torch.from_numpy(self.other_vehicles) 
            obstacles = torch.from_numpy(self.obstacles)
            
            uni_orient_targets = torch.stack((torch.cos(target[2]), 
                                            torch.sin(target[2])), dim=-1)
            
            vec_to_targets = target[None, :2]-query_points[...,:2]
            dist_to_targets = torch.norm(vec_to_targets, dim=-1, p=2, keepdim=True)
            uni_vec_targets = vec_to_targets/(dist_to_targets+1e-8)
            
            vec_avoid_collision = torch.zeros((len(query_points), 2))
            
            ##### obstacle avoiding #####
            dynamic_safe_distance = torch.tensor(self.dyn_safe_dist_obs).item()
            vec_to_obstacles = obstacles[None,:,:2]-query_points[:,None,:2]
            dist_to_obstacles_center = torch.norm(vec_to_obstacles, dim=-1, p=2, keepdim=True)
            dist_to_obstacles = dist_to_obstacles_center-obstacles[None,:,2:3]-self.vehicle_radius
            
            mask_collision = torch.sum(vec_to_obstacles*vec_to_targets[:,None,:], dim=-1)>0
            mask_collision = mask_collision & (dist_to_obstacles<=dynamic_safe_distance)[...,0]
            vec_to_obstacles = vec_to_obstacles/(dist_to_obstacles_center+1e-8)
            
            vec_back_obstacles = torch.clip(dist_to_obstacles-dynamic_safe_distance, 
                                            min=-dynamic_safe_distance, 
                                            max=0)*vec_to_obstacles
            vec_avoid_obstacles = vec_back_obstacles.clone()
                        
            vec_to_obstacles_3d = torch.cat((vec_to_obstacles, torch.zeros_like(dist_to_obstacles)), dim=-1)
            
            vec_direct = torch.zeros_like(vec_to_obstacles_3d)
            vec_direct[...,-1] = 1
            
            vec_around_obstacles = torch.cross(vec_direct, vec_to_obstacles_3d, dim=-1)[...,:2]
            vec_around_obstacles = vec_around_obstacles/(torch.norm(vec_around_obstacles, dim=-1, p=2, keepdim=True)+1e-8)
            vec_around_obstacles = torch.clip(dist_to_obstacles, 
                                              min=0, 
                                              max=dynamic_safe_distance)*vec_around_obstacles
            vec_around_obstacles[~mask_collision] = 0
            vec_avoid_obstacles += vec_around_obstacles
            
            vec_avoid_collision += torch.sum(vec_avoid_obstacles, dim=1)
            
            ##### vehicle avoiding #####
            dynamic_safe_distance = torch.tensor(self.dyn_safe_dist_veh)[None,:,None]
            vec_to_vehicles = vehicles[None,:,:2]-query_points[:,None,:2]
            dist_to_vehicles_center = torch.norm(vec_to_vehicles, dim=-1, p=2, keepdim=True)
            dist_to_vehicles = dist_to_vehicles_center-2*self.vehicle_radius
            
            mask_collision = torch.sum(vec_to_vehicles*vec_to_targets[:,None,:], dim=-1)>0
            mask_collision = mask_collision & (dist_to_vehicles<=dynamic_safe_distance)[...,0]
            vec_to_vehicles = vec_to_vehicles/(dist_to_vehicles_center+1e-8)
            
            vec_back_vehicles = torch.clip(dist_to_vehicles-dynamic_safe_distance, 
                                            min=-dynamic_safe_distance, 
                                            max=torch.zeros_like(dynamic_safe_distance))*vec_to_vehicles
            vec_avoid_vehicles = vec_back_vehicles.clone()
            
            vec_to_vehicles_3d = torch.cat((vec_to_vehicles, torch.zeros_like(dist_to_vehicles)), dim=-1)
            
            vec_direct = torch.zeros_like(vec_to_vehicles_3d)
            vec_direct[...,-1] = 1
            
            vec_around_vehicles = torch.cross(vec_direct, vec_to_vehicles_3d, dim=-1)[...,:2]
            vec_around_vehicles = vec_around_vehicles/(torch.norm(vec_around_vehicles, dim=-1, p=2, keepdim=True)+1e-8)
            vec_around_vehicles = torch.clip(dist_to_vehicles, 
                                             min=torch.zeros_like(dynamic_safe_distance), 
                                             max=dynamic_safe_distance)*vec_around_vehicles
            vec_around_vehicles[~mask_collision] = 0
            vec_avoid_vehicles += vec_around_vehicles
            
            vec_avoid_collision += torch.sum(vec_avoid_vehicles, dim=1)
                    
            vec_to_targets = uni_vec_targets.clone()
            
            mask_stop = (dist_to_targets<pos_tolerance)
            
            factor1 = torch.sum(vec_to_targets*uni_orient_targets[None,:], dim=-1, keepdim=True) 
            factor2 = torch.clip(dist_to_targets, min=0, max=self.parking_distance)/self.parking_distance
            factor2[~mask_stop] += 1
            factor = 2*((factor1>=0).float()-0.5)*factor2
            
            mask_parking = dist_to_targets[...,0]<=self.parking_distance
            vec_to_targets[mask_parking] = (uni_orient_targets+factor*vec_to_targets)[mask_parking]
            vec_to_targets = vec_to_targets/(torch.norm(vec_to_targets, dim=-1, p=2, keepdim=True)+1e-8)
            
            vec_ref_orient = vec_to_targets+vec_avoid_collision
            
            return vec_ref_orient.numpy(), vec_to_targets.numpy(), vec_avoid_collision.numpy(), \
                   vec_avoid_vehicles.numpy(), vec_avoid_obstacles.numpy(), \
                   vec_back_obstacles.numpy(), vec_around_obstacles.numpy(),\
                   vec_back_vehicles.numpy(), vec_around_vehicles.numpy()


    def car_patch_pos(self, posture):
        posture_new = posture.copy()
        posture_new[...,0] = posture[...,0]-np.sin(posture[...,2])*(self.car_size[0]/2) \
                                           -np.cos(posture[...,2])*(self.car_size[1]/2)
        posture_new[...,1] = posture[...,1]+np.cos(posture[...,2])*(self.car_size[0]/2) \
                                           -np.sin(posture[...,2])*(self.car_size[1]/2)
        return posture_new
    
    def get_query_points(self):
        
        query_points = np.stack(np.meshgrid(np.linspace(-self.fig_range,self.fig_range,
                                                        self.num_points_x_y), 
                                            np.linspace(-self.fig_range,self.fig_range,
                                                        self.num_points_x_y)), axis=-1).reshape(-1,2)
        
        vec_to_obstacles = self.obstacles[None,:,:2]-query_points[:,None,:]
        dist_to_obstacles = np.linalg.norm(vec_to_obstacles, axis=-1, ord=2)-self.obstacles[None,:,2]
        mask_obstacles = dist_to_obstacles>self.dyn_safe_dist_obs+1
        
        vec_to_vehicles = self.other_vehicles[None,:,:2]-query_points[:,None,:]
        dist_to_vehicles = np.linalg.norm(vec_to_vehicles, axis=-1, ord=2)-self.vehicle_radius
        mask_vehicles = dist_to_vehicles>self.dyn_safe_dist_veh[None,:]+1
        
        vec_to_targets = self.target[None,:2]-query_points
        dist_to_targets = np.linalg.norm(vec_to_targets, axis=-1, ord=2)
        mask_targets = dist_to_targets>self.parking_distance+1
        
        vec_to_egos = self.ego_vehicle[None,:2]-query_points
        dist_to_egos = np.linalg.norm(vec_to_egos, axis=-1, ord=2)
        mask_egos = dist_to_egos>2
        
        mask = np.all(np.concatenate((mask_obstacles, mask_vehicles, 
                                      mask_targets[...,None], mask_egos[...,None]), axis=-1), axis=-1)
        query_points = query_points[mask]
        
        angle = np.linspace(-np.pi, np.pi, self.num_point_around, endpoint=False)
        radius = np.linspace(1, self.vehicle_radius+self.dyn_safe_dist_obs-1e-8, 3, endpoint=True)
        query_points_1 = np.stack(np.meshgrid(angle, radius), axis=-1).reshape(-1,2)
        angle = query_points_1[...,0]
        radius = query_points_1[...,1]
        angle_bias = np.arctan2(self.obstacles[:,1]-self.target[1], self.obstacles[:,0]-self.target[0])
        query_points_1 = np.stack((self.obstacles[:,0][:,None]+np.cos(angle[None,:]+angle_bias[:,None])*\
                                  (self.obstacles[:,2][:,None]+radius[None,:]),
                                   self.obstacles[:,1][:,None]+np.sin(angle[None,:]+angle_bias[:,None])*\
                                  (self.obstacles[:,2][:,None]+radius[None,:]),),
                                   axis=-1).reshape(-1,2)
        
        query_points_2 = []
        for i in range(len(self.dyn_safe_dist_veh)):
            angle = np.linspace(-np.pi, np.pi, self.num_point_around, endpoint=False)
            radius = np.linspace(1, self.vehicle_radius+self.dyn_safe_dist_veh[i]-1e-8, 3, endpoint=True)
            query_points_2_i = np.stack(np.meshgrid(angle, radius), axis=-1).reshape(-1,2)
            angle = query_points_2_i[...,0]
            radius = query_points_2_i[...,1]
            angle_bias = np.arctan2(self.other_vehicles[:,1]-self.target[1], 
                                    self.other_vehicles[:,0]-self.target[0])
            query_points_2.append(np.stack((self.other_vehicles[:,0][:,None]+np.cos(angle[None,:]+angle_bias[:,None])*\
                                           (radius[None,:]+self.vehicle_radius),
                                            self.other_vehicles[:,1][:,None]+np.sin(angle[None,:]+angle_bias[:,None])*\
                                           (radius[None,:]+self.vehicle_radius),),
                                            axis=-1).reshape(-1,2))
        query_points_2 = np.concatenate(query_points_2, axis=0)
        
        
        angle = np.linspace(-np.pi, np.pi, self.num_point_around, endpoint=False)
        radius = np.linspace(2, self.parking_distance-1e-8, 3, endpoint=True)
        query_points_3 = np.stack(np.meshgrid(angle, radius), axis=-1).reshape(-1,2)
        angle = query_points_3[...,0]
        radius = query_points_3[...,1]
        query_points_3 = np.stack((self.target[0]+np.cos(angle+self.target[2])*radius,
                                   self.target[1]+np.sin(angle+self.target[2])*radius,),
                                   axis=-1).reshape(-1,2)
        
        query_points = np.concatenate((query_points, 
                                       query_points_1, 
                                       query_points_2, 
                                       query_points_3), axis=0)
        
        return query_points


    def plot_velocity_field(self, only_final_vector=False):
        
        fig = plt.figure(figsize=(self.fig_size,self.fig_size))
        ax = fig.add_subplot()
        
        ego_vehicle_new = self.car_patch_pos(self.ego_vehicle)
        target_new = self.car_patch_pos(self.target)
        other_vehicles_new = self.car_patch_pos(self.other_vehicles)

        for i in range(len(other_vehicles_new)):
            
            dynamic_safe_distance = self.dyn_safe_dist_veh[i]
            danger_range = mpatches.Circle(self.other_vehicles[i,:2], 
                                            dynamic_safe_distance+2*self.vehicle_radius, 
                                            color=self.cmap[i+1], ls='dotted', 
                                            fill=False, lw=2, alpha=0.5)
            
            ax.add_patch(danger_range)
            
            # cars
            patch_car = mpatches.Rectangle([0,0], self.car_size[0], self.car_size[1], color=self.cmap[i+1])
            patch_car.set_xy(other_vehicles_new[i,:2])
            patch_car.angle = np.rad2deg(other_vehicles_new[i,2])-90
            patch_car_arrow = mpatches.FancyArrow(self.other_vehicles[i,0]-0.9*np.cos(self.other_vehicles[i,2]), 
                                                    self.other_vehicles[i,1]-0.9*np.sin(self.other_vehicles[i,2]), 
                                                    1.5*np.cos(self.other_vehicles[i,2]), 
                                                    1.5*np.sin(self.other_vehicles[i,2]), 
                                                    width=0.1, color='w')

            ax.add_patch(patch_car)
            ax.add_patch(patch_car_arrow)

            vehicle_mark, =  ax.plot([], [], color=self.cmap[i+1], marker='.', 
                                        linewidth=1, label=f"other vehicle {i+1}")
            
        patch_car = mpatches.Rectangle([0,0], self.car_size[0], self.car_size[1], color=self.cmap[0])
        patch_car.set_xy(ego_vehicle_new[:2])
        patch_car.angle = np.rad2deg(ego_vehicle_new[2])-90
        patch_car_arrow = mpatches.FancyArrow(self.ego_vehicle[0]-0.9*np.cos(self.ego_vehicle[2]), 
                                                self.ego_vehicle[1]-0.9*np.sin(self.ego_vehicle[2]), 
                                                1.5*np.cos(self.ego_vehicle[2]), 
                                                1.5*np.sin(self.ego_vehicle[2]), 
                                                width=0.1, color='w')
        
        ax.add_patch(patch_car)
        ax.add_patch(patch_car_arrow)
        vehicle_mark, =  ax.plot([], [], color=self.cmap[0], marker='.', 
                                    linewidth=1, label=f"ego vehicle")
        
        patch_target = mpatches.Rectangle([0,0], self.car_size[0], self.car_size[1], color=self.cmap[0], 
                                            ls='dashdot', fill=False)
        patch_target.set_xy(target_new[:2])
        patch_target.angle = np.rad2deg(target_new[2])-90        
        patch_target_arrow = mpatches.FancyArrow(self.target[0]-0.9*np.cos(self.target[2]), 
                                                    self.target[1]-0.9*np.sin(self.target[2]), 
                                                    1.5*np.cos(self.target[2]), 
                                                    1.5*np.sin(self.target[2]), 
                                                    width=0.1, 
                                                    color=self.cmap[0])
        
        parking_range = mpatches.Circle(self.target[:2], self.parking_distance, color=self.cmap[0], 
                                        ls='dotted', fill=False, lw=2, alpha=0.5)
        
        uncertain_direction_range = mpatches.Wedge(self.target[:2], 
                                                    self.parking_distance, 
                                                    -180, 180,
                                                    width=-0.5*self.default_ref_vel**2,
                                                    alpha=0.25)
        ax.add_patch(patch_target)
        ax.add_patch(patch_target_arrow)
        ax.add_patch(parking_range)
        ax.add_patch(uncertain_direction_range)
        
        for i in range(len(self.obstacles)):
            
            dynamic_safe_distance = self.dyn_safe_dist_obs
            danger_range = mpatches.Circle(self.obstacles[i,:2], 
                                            dynamic_safe_distance+self.obstacles[i,2]+self.vehicle_radius, 
                                            color=self.cmap[i+1+len(self.other_vehicles)], 
                                            ls='dotted', fill=False, lw=2, alpha=0.5)
            ax.add_patch(danger_range)
            
            obstacle = mpatches.Circle(self.obstacles[i,:2], self.obstacles[i,2], 
                                        color=self.cmap[i+1+len(self.other_vehicles)], fill=True)
            
            ax.add_patch(obstacle)
            
            obstacle_mark, = ax.plot([], [], color=self.cmap[i+1+len(self.other_vehicles)], marker='.', 
                                        linewidth=1, label=f"obstacle {i+1}")
        
        
        ##### get query points #####
        
        query_points = self.get_query_points()
        query_points = np.concatenate((query_points, np.tile(self.ego_vehicle[2:][None,:], (len(query_points),1))), axis=-1)
        
        query_next_point = np.array([[self.ego_next[0], self.ego_next[1], self.ego_vehicle[2], self.ego_vehicle[3]]])
        
        vec_ref_orient, vec_to_targets, vec_avoid_collision, vec_avoid_vehicles, vec_avoid_obstacles, \
        vec_back_obstacles, vec_around_obstacles, vec_back_vehicles, vec_around_vehicles = self.get_velocity_field(query_points)
        
        if only_final_vector: 
            vec_ref_orient = vec_ref_orient/(np.linalg.norm(vec_ref_orient, ord=2, axis=-1, keepdims=True)+1e-8)
            vec_ref_orient *= self.vector_scale
        
        else:
            vec_ref_orient *= self.vector_scale
            vec_to_targets *= self.vector_scale
            vec_avoid_collision *= self.vector_scale
            vec_avoid_vehicles *= self.vector_scale
            vec_avoid_obstacles *= self.vector_scale
            vec_back_obstacles *= self.vector_scale
            vec_around_obstacles *= self.vector_scale
            vec_back_vehicles *= self.vector_scale
            vec_around_vehicles *= self.vector_scale
        
        for i in range(len(query_points)):
            
            ax.arrow(query_points[i,0], query_points[i,1], vec_ref_orient[i,0], vec_ref_orient[i,1], width=0.07)
            
            if only_final_vector: continue
            
            for j in range(len(self.other_vehicles)):
                if vec_avoid_vehicles[i,j,0] == 0 and vec_avoid_vehicles[i,j,1] == 0:
                    continue
                ax.arrow(query_points[i,0], query_points[i,1], vec_to_targets[i,0], vec_to_targets[i,1], width=0.07, 
                    color='gray', alpha=0.5)
                ax.arrow(query_points[i,0], query_points[i,1], vec_avoid_vehicles[i,j,0], vec_avoid_vehicles[i,j,1],
                            width=0.07, color=self.cmap[j+1], alpha=0.5)
            
            for j in range(len(self.obstacles)):
                if vec_avoid_obstacles[i,j,0] == 0 and vec_avoid_obstacles[i,j,1] == 0:
                    continue
                ax.arrow(query_points[i,0], query_points[i,1], vec_to_targets[i,0], vec_to_targets[i,1], width=0.07, 
                    color='gray', alpha=0.5)
                ax.arrow(query_points[i,0], query_points[i,1], vec_avoid_obstacles[i,j,0], vec_avoid_obstacles[i,j,1],
                            width=0.07, color=self.cmap[j+len(self.other_vehicles)+1], alpha=0.5)
                
        vec_ref_orient_next = self.get_velocity_field(query_next_point)[0]
        patch_next_arrow = mpatches.FancyArrow(query_next_point[0,0], 
                                                query_next_point[0,1], 
                                                2*vec_ref_orient_next[0,0], 
                                                2*vec_ref_orient_next[0,1], 
                                                width=0.1, 
                                                color=self.cmap[0])
        ax.add_patch(patch_next_arrow)
        
        orient_next_range = mpatches.Wedge((query_next_point[0,0], query_next_point[0,1]), 2.5, 
                                            np.rad2deg(self.ego_next[2]), 
                                            np.rad2deg(self.ego_next[3]), 
                                            alpha=0.5)
        ax.add_patch(orient_next_range)
        
        if self.config["save plot"]:
            if not os.path.exists(self.config["plot folder"]):
                os.makedirs(self.config["plot folder"])
            plt.savefig(os.path.join(self.config["plot folder"], 
                                     self.config["name"]+".png"), 
                        bbox_inches='tight', dpi=800)
    
        if self.config["show plot"]:
            plt.show()
            
        plt.close()


class Visualize_Velocity_Field:
    def __init__(self, config):
        
        self.config = config 
        self.dt = 0.2
        self.vehicles = np.array(config['starts'])
        self.obstacles = np.array(config['obstacles'])
        self.targets = np.array(config["targets"])
        self.car_size = config['car size'] #car_size=[1.0,2.5]
        self.fig_size = config['figure size']
        self.safe_distance = config.get('safe distance', 1.5)
        self.parking_distance = config.get('parking distance', 5)
        self.default_ref_vel = config.get('default reference velocity', 2.5)
        self.vehicle_radius = config.get('vehicle radius', 1.5)
        
        assert 2*self.vehicle_radius > np.linalg.norm(np.array(self.car_size), ord=2, axis=-1), \
               "Vehicle radius can not cover car size!"
        
        self.num_obstacles = len(self.obstacles)
        self.num_vehicles = len(self.vehicles)
        
        self.dyn_parking_dist = self.parking_distance + 0.5*self.default_ref_vel**2
        
        x_coord_min = np.concatenate((self.obstacles[:,0]-self.obstacles[:,2]-self.safe_distance-self.vehicle_radius, 
                                      self.vehicles[:,0]-self.safe_distance-2*self.vehicle_radius, 
                                      self.targets[:,0]-self.dyn_parking_dist, 
                                      ), axis=0)
        x_coord_max = np.concatenate((self.obstacles[:,0]+self.obstacles[:,2]+self.safe_distance+self.vehicle_radius, 
                                      self.vehicles[:,0]+self.safe_distance+2*self.vehicle_radius, 
                                      self.targets[:,0]+self.safe_distance, 
                                      ), axis=0)
        y_coord_min = np.concatenate((self.obstacles[:,1]-self.obstacles[:,2]-self.safe_distance-self.vehicle_radius,  
                                      self.vehicles[:,1]-self.safe_distance-2*self.vehicle_radius, 
                                      self.targets[:,1]-self.safe_distance,  
                                      ), axis=0)
        y_coord_max = np.concatenate((self.obstacles[:,1]+self.obstacles[:,2]+self.safe_distance+self.vehicle_radius,  
                                      self.vehicles[:,1]+self.safe_distance+2*self.vehicle_radius, 
                                      self.targets[:,1]+self.safe_distance,  
                                      ), axis=0)
        x_min = np.floor(np.amin(x_coord_min)/2.5)*2.5
        x_max = np.ceil(np.amax(x_coord_max)/2.5)*2.5
        y_min = np.floor(np.amin(y_coord_min)/2.5)*2.5
        y_max = np.ceil(np.amax(y_coord_max)/2.5)*2.5
        
        self.center = np.array([(x_max+x_min)/2, (y_max+y_min)/2])
        self.fig_range = max((x_max-x_min)/2, (y_max-y_min)/2)
        
        self.num_points_x_y = int((2*self.fig_range)//2.5)+1
        self.num_point_around = 16
        self.vector_scale = 0.4*self.fig_range/(self.num_points_x_y-1)
        
        num_colors = self.num_obstacles + self.num_vehicles
        cmap = plt.get_cmap('brg')
        self.cmap = cmap(np.linspace(0,1,num_colors))[...,:3]
        
    def update_parameters(self, states):
        
        self.dyn_safe_dist_obs = self.safe_distance + np.abs(states[:,3])
        self.dyn_safe_dist_veh = self.safe_distance + np.abs(states[:,None,3])+np.abs(states[None,:,3])
        
        self.state_next = [states[:,0]+states[:,3]*np.cos(states[:,2])*self.dt,
                           states[:,1]+states[:,3]*np.sin(states[:,2])*self.dt,
                           states[:,2]-np.abs(states[:,3]*self.dt*np.tan(0.8)/2.0),
                           states[:,2]+np.abs(states[:,3]*self.dt*np.tan(0.8)/2.0)]
        
        self.state_next = np.stack(self.state_next, axis=-1)
        

    def get_velocity_field(self, query_points, states, pos_tolerance=0.25):
        
        veh_idx = self.veh_idx
        
        query_points = torch.from_numpy(query_points) 
        targets = torch.from_numpy(self.targets[veh_idx])
        vehicles = torch.from_numpy(np.delete(states, veh_idx, 0)) 
        obstacles = torch.from_numpy(self.obstacles)
        
        uni_orient_targets = torch.stack((torch.cos(targets[2]), 
                                            torch.sin(targets[2])), dim=-1)
        
        vec_to_targets = targets[None, :2]-query_points[...,:2]
        dist_to_targets = torch.norm(vec_to_targets, dim=-1, p=2, keepdim=True)
        uni_vec_targets = vec_to_targets/(dist_to_targets+1e-8)
        
        vec_avoid_collision = torch.zeros((len(query_points), 2))
        
        ##### obstacle avoiding #####
        dynamic_safe_distance = torch.tensor(self.dyn_safe_dist_obs[veh_idx]).item()
        vec_to_obstacles = obstacles[None,:,:2]-query_points[:,None,:2]
        dist_to_obstacles_center = torch.norm(vec_to_obstacles, dim=-1, p=2, keepdim=True)
        dist_to_obstacles = dist_to_obstacles_center-obstacles[None,:,2:3]-self.vehicle_radius
        
        mask_collision = torch.sum(vec_to_obstacles*vec_to_targets[:,None,:], dim=-1)>0
        mask_collision = mask_collision & (dist_to_obstacles<=dynamic_safe_distance)[...,0]
        vec_to_obstacles = vec_to_obstacles/(dist_to_obstacles_center+1e-8)
        
        vec_back_obstacles = torch.clip(dist_to_obstacles-dynamic_safe_distance, 
                                        min=-dynamic_safe_distance, 
                                        max=0)*vec_to_obstacles
        vec_avoid_obstacles = vec_back_obstacles.clone()
                    
        vec_to_obstacles_3d = torch.cat((vec_to_obstacles, torch.zeros_like(dist_to_obstacles)), dim=-1)
        
        vec_direct = torch.zeros_like(vec_to_obstacles_3d)
        vec_direct[...,-1] = 1
        
        vec_around_obstacles = torch.cross(vec_direct, vec_to_obstacles_3d, dim=-1)[...,:2]
        vec_around_obstacles = vec_around_obstacles/(torch.norm(vec_around_obstacles, dim=-1, p=2, keepdim=True)+1e-8)
        vec_around_obstacles = torch.clip(dist_to_obstacles, 
                                            min=0, 
                                            max=dynamic_safe_distance)*vec_around_obstacles
        vec_around_obstacles[~mask_collision] = 0
        vec_avoid_obstacles += vec_around_obstacles
        
        vec_avoid_collision += torch.sum(vec_avoid_obstacles, dim=1)
        
        ##### vehicle avoiding #####
        dynamic_safe_distance = torch.tensor(np.delete(self.dyn_safe_dist_veh[veh_idx], veh_idx, 0))[None,:,None]
        vec_to_vehicles = vehicles[None,:,:2]-query_points[:,None,:2]
        dist_to_vehicles_center = torch.norm(vec_to_vehicles, dim=-1, p=2, keepdim=True)
        dist_to_vehicles = dist_to_vehicles_center-2*self.vehicle_radius
        
        mask_collision = torch.sum(vec_to_vehicles*vec_to_targets[:,None,:], dim=-1)>0
        mask_collision = mask_collision & (dist_to_vehicles<=dynamic_safe_distance)[...,0]
        vec_to_vehicles = vec_to_vehicles/(dist_to_vehicles_center+1e-8)
        
        vec_back_vehicles = torch.clip(dist_to_vehicles-dynamic_safe_distance, 
                                        min=-dynamic_safe_distance, 
                                        max=torch.zeros_like(dynamic_safe_distance))*vec_to_vehicles
        vec_avoid_vehicles = vec_back_vehicles.clone()
        
        vec_to_vehicles_3d = torch.cat((vec_to_vehicles, torch.zeros_like(dist_to_vehicles)), dim=-1)
        
        vec_direct = torch.zeros_like(vec_to_vehicles_3d)
        vec_direct[...,-1] = 1
        
        vec_around_vehicles = torch.cross(vec_direct, vec_to_vehicles_3d, dim=-1)[...,:2]
        vec_around_vehicles = vec_around_vehicles/(torch.norm(vec_around_vehicles, dim=-1, p=2, keepdim=True)+1e-8)
        vec_around_vehicles = torch.clip(dist_to_vehicles, 
                                            min=torch.zeros_like(dynamic_safe_distance), 
                                            max=dynamic_safe_distance)*vec_around_vehicles
        vec_around_vehicles[~mask_collision] = 0
        vec_avoid_vehicles += vec_around_vehicles
        
        vec_avoid_collision += torch.sum(vec_avoid_vehicles, dim=1)
                
        vec_to_targets = uni_vec_targets.clone()
        
        mask_stop = (dist_to_targets<pos_tolerance)
        
        factor1 = torch.sum(vec_to_targets*uni_orient_targets[None,:], dim=-1, keepdim=True) 
        factor2 = torch.clip(dist_to_targets, min=0, max=self.parking_distance)/self.parking_distance
        factor2[~mask_stop] += 1
        factor = 2*((factor1>=0).float()-0.5)*factor2
        
        mask_parking = dist_to_targets[...,0]<=self.parking_distance
        vec_to_targets[mask_parking] = (uni_orient_targets+factor*vec_to_targets)[mask_parking]
        
        if (dist_to_targets[-1,0].item() > self.parking_distance) & \
            (dist_to_targets[-1,0].item() <= self.dyn_parking_dist):
                
            uni_orient_starts = torch.stack((torch.cos(query_points[...,2]), 
                                                torch.sin(query_points[...,2])), dim=-1)
        
            factor = torch.sum(uni_orient_starts*uni_vec_targets, dim=-1, keepdim=True)
            factor = ((factor>=0).float()-0.5)*2
            mask_parking_margin = (dist_to_targets[...,0] > self.parking_distance) & \
                            (dist_to_targets[...,0] <= self.dyn_parking_dist)
            vec_to_targets[mask_parking_margin] = (vec_to_targets*factor)[mask_parking_margin]
        
        vec_to_targets = vec_to_targets/(torch.norm(vec_to_targets, dim=-1, p=2, keepdim=True)+1e-8)
        
        vec_ref_orient = vec_to_targets+vec_avoid_collision
        vec_ref_orient = vec_ref_orient/(torch.norm(vec_ref_orient, p=2, dim=-1, keepdim=True)+1e-8)
        
        return vec_ref_orient.numpy()


    def car_patch_pos(self, posture):
        posture_new = posture.copy()
        posture_new[...,0] = posture[...,0]-np.sin(posture[...,2])*(self.car_size[0]/2) \
                                           -np.cos(posture[...,2])*(self.car_size[1]/2)
        posture_new[...,1] = posture[...,1]+np.cos(posture[...,2])*(self.car_size[0]/2) \
                                           -np.sin(posture[...,2])*(self.car_size[1]/2)
        return posture_new
    
    def get_query_points(self, states):
        
        veh_idx = self.veh_idx
        
        targets = self.targets
        vehicles = states
        obstacles = self.obstacles
        
        query_points = np.stack(np.meshgrid(np.linspace(-self.fig_range+self.center[0],self.fig_range+self.center[0],
                                                        self.num_points_x_y), 
                                            np.linspace(-self.fig_range+self.center[1],self.fig_range+self.center[1],
                                                        self.num_points_x_y)), axis=-1).reshape(-1,2)
        
        vec_to_obstacles = obstacles[None,:,:2]-query_points[:,None,:]
        dist_to_obstacles = np.linalg.norm(vec_to_obstacles, axis=-1, ord=2)-self.obstacles[None,:,2]
        mask_obstacles = dist_to_obstacles>1
        
        vec_to_vehicles = vehicles[None,:,:2]-query_points[:,None,:]
        dist_to_vehicles = np.linalg.norm(vec_to_vehicles, axis=-1, ord=2)-self.vehicle_radius
        mask_vehicles = dist_to_vehicles>1
        
        vec_to_targets = targets[None,:,:2]-query_points[:,None,:]
        dist_to_targets = np.linalg.norm(vec_to_targets, axis=-1, ord=2)
        mask_targets = dist_to_targets>2
        
        mask = np.all(np.concatenate((mask_obstacles, mask_vehicles, mask_targets), axis=-1), axis=-1)
        query_points = query_points[mask]
        
        query_point_next = self.state_next[veh_idx:veh_idx+1, :2]
        query_points = np.concatenate((query_points, query_point_next), axis=0)
        
        return query_points


    def base_plot(self):
        
        veh_idx = self.veh_idx

        self.update_parameters(self.vehicles)
        
        self.fig = plt.figure(figsize=(self.fig_size,self.fig_size))
        self.ax = self.fig.add_subplot()
        self.ax.set_xlim([-self.fig_range+self.center[0]-self.vector_scale-1, 
                            self.fig_range+self.center[0]+self.vector_scale+1])
        self.ax.set_ylim([-self.fig_range+self.center[1]-self.vector_scale-1, 
                            self.fig_range+self.center[1]+self.vector_scale+1])
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        vehicles_new = self.car_patch_pos(self.vehicles)
        targets_new = self.car_patch_pos(self.targets)
        
        self.patch_vehicles = []
        self.patch_vehicles_arrow = []
        self.patch_vehicles_danger_range = []   
        self.patch_obstacles_danger_range = []
    
        for i in range(len(self.vehicles)):
            
            dynamic_safe_distance = self.dyn_safe_dist_veh[veh_idx, i]
            alpha = 0 if i==veh_idx else 0.5
            danger_range = mpatches.Circle(self.vehicles[i,:2], dynamic_safe_distance+2*self.vehicle_radius, 
                                            color=self.cmap[i], ls='dotted', fill=False, lw=2, alpha=alpha)
            
            self.patch_vehicles_danger_range.append(danger_range)
            self.ax.add_patch(danger_range)
            
            # cars
            patch_car = mpatches.Rectangle([0,0], self.car_size[0], self.car_size[1], color=self.cmap[i])
            patch_car.set_xy(vehicles_new[i,:2])
            patch_car.angle = np.rad2deg(vehicles_new[i,2])-90
            patch_car_arrow = mpatches.FancyArrow(self.vehicles[i,0]-0.9*np.cos(self.vehicles[i,2]), 
                                                    self.vehicles[i,1]-0.9*np.sin(self.vehicles[i,2]), 
                                                    1.5*np.cos(self.vehicles[i,2]), 
                                                    1.5*np.sin(self.vehicles[i,2]), 
                                                    width=0.1, color='w')

            self.patch_vehicles.append(patch_car)
            self.patch_vehicles_arrow.append(patch_car_arrow)
            
            self.ax.add_patch(patch_car)
            self.ax.add_patch(patch_car_arrow)

            vehicle_mark, =  self.ax.plot([], [], color=self.cmap[i], marker='.', 
                                        linewidth=1, label=f"vehicle {i+1}")
        
            patch_target = mpatches.Rectangle([0,0], self.car_size[0], self.car_size[1], color=self.cmap[i], 
                                            ls='dashdot', fill=False)
            patch_target.set_xy(targets_new[i, :2])
            patch_target.angle = np.rad2deg(targets_new[i, 2])-90        
            patch_target_arrow = mpatches.FancyArrow(self.targets[i,0]-0.9*np.cos(self.targets[i,2]), 
                                                        self.targets[i,1]-0.9*np.sin(self.targets[i,2]), 
                                                        1.5*np.cos(self.targets[i,2]), 
                                                        1.5*np.sin(self.targets[i,2]), 
                                                        width=0.1, 
                                                        color=self.cmap[i])
            
            self.ax.add_patch(patch_target)
            self.ax.add_patch(patch_target_arrow)
        
        
        parking_range = mpatches.Circle(self.targets[veh_idx,:2], self.parking_distance, color=self.cmap[0], 
                                        ls='dotted', fill=False, lw=2, alpha=0.5)
        
        uncertain_direction_range = mpatches.Wedge(self.targets[veh_idx,:2], 
                                                    self.parking_distance, 
                                                    -180, 180,
                                                    width=-0.5*self.default_ref_vel**2,
                                                    color=self.cmap[0],
                                                    alpha=0.20)
        self.ax.add_patch(parking_range)
        self.ax.add_patch(uncertain_direction_range)
        
        for i in range(len(self.obstacles)):
            
            dynamic_safe_distance = self.dyn_safe_dist_obs[veh_idx]
            danger_range = mpatches.Circle(self.obstacles[i,:2], 
                                            dynamic_safe_distance+self.obstacles[i,2]+self.vehicle_radius, 
                                            color=self.cmap[i+self.num_vehicles], ls='dotted', fill=False, lw=2, alpha=0.5)
            self.patch_obstacles_danger_range.append(danger_range)
            self.ax.add_patch(danger_range)
            
            obstacle = mpatches.Circle(self.obstacles[i,:2], self.obstacles[i,2], 
                                        color=self.cmap[i+self.num_vehicles], fill=True)
            
            self.ax.add_patch(obstacle)
            
            obstacle_mark, = self.ax.plot([], [], color=self.cmap[i], marker='.', 
                                        linewidth=1, label=f"obstacle {i+1}")
        
        
        ##### get query points #####
        query_points = self.get_query_points(self.vehicles)
        query_points = np.concatenate((query_points, np.tile(self.vehicles[veh_idx:veh_idx+1, 2:], 
                                                                (len(query_points),1))), axis=-1)
        query_next_point = query_points[-1]
        
        vec_ref_orient = self.get_velocity_field(query_points, self.vehicles)
        vec_ref_orient_next = vec_ref_orient[-1]
        
        query_points = query_points[:-1]
        vec_ref_orient = vec_ref_orient[:-1]
        vec_ref_orient *= self.vector_scale
        
        self.query_points = []
        
        for i in range(len(query_points)):
            
            patch_arrow = mpatches.FancyArrow(query_points[i,0], 
                                        query_points[i,1], 
                                        vec_ref_orient[i,0], 
                                        vec_ref_orient[i,1], 
                                        width=0.07,
                                        color='gray')
            self.query_points.append(patch_arrow)
            self.ax.add_patch(patch_arrow)
        
        self.patch_next_arrow = mpatches.FancyArrow(query_next_point[0], 
                                                    query_next_point[1], 
                                                    2*vec_ref_orient_next[0], 
                                                    2*vec_ref_orient_next[1], 
                                                    width=0.1)
        self.ax.add_patch(self.patch_next_arrow)
        
        self.orient_next_range = mpatches.Wedge((query_next_point[0], query_next_point[1]), 2.5, 
                                                    np.rad2deg(self.state_next[veh_idx, 2]), 
                                                    np.rad2deg(self.state_next[veh_idx, 3]), 
                                                    alpha=0.5)
        self.ax.add_patch(self.orient_next_range)
        
        self.fig.tight_layout()
            
        return

    def create_video(self, trajectorys, veh_idx):
        self.trajectorys = trajectorys
        self.veh_idx = veh_idx
        self.base_plot()
            
        car_animation = animation.FuncAnimation(self.fig, self.update_plot, frames=range(len(trajectorys)-1), interval=100, repeat=True, blit=False)
        
        if self.config["save plot"]:
            if not os.path.exists(self.config["plot folder"]):
                os.makedirs(self.config["plot folder"])
                
            car_animation.save(os.path.join(self.config["plot folder"], 
                                            self.config["name"] + f"_veh={veh_idx}_dvf.gif"))
            
        if self.config["show plot"]:
            plt.show()
            
        plt.close()

    def update_plot(self, num):
        
        states_t = self.trajectorys[num,...]   
        veh_idx = self.veh_idx
        self.update_parameters(states_t)      
        states_t_ = self.car_patch_pos(states_t)      
        
        for i in range(self.num_vehicles):
            # vehicle
            self.patch_vehicles[i].set_xy(states_t_[i,:2])
            self.patch_vehicles[i].angle = np.rad2deg(states_t_[i,2])-90
            self.patch_vehicles_arrow[i].set_data(x=states_t[i,0]-0.9*np.cos(states_t[i,2]), 
                                                  y=states_t[i,1]-0.9*np.sin(states_t[i,2]), 
                                                  dx=1.5*np.cos(states_t[i,2]), 
                                                  dy=1.5*np.sin(states_t[i,2]))
            
            self.patch_vehicles_danger_range[i].set_center(states_t[i,:2])
            self.patch_vehicles_danger_range[i].set_radius(self.dyn_safe_dist_veh[veh_idx, i]+
                                                           2*self.vehicle_radius)
            
        for i in range(self.num_obstacles):
            self.patch_obstacles_danger_range[i].set_radius(self.dyn_safe_dist_obs[veh_idx]+
                                                            self.obstacles[i,2]+self.vehicle_radius)
            
        for i in range(len(self.query_points)):
            patch_arrow = self.query_points.pop()
            patch_arrow.remove()
        
        query_points = self.get_query_points(states_t)
        query_points = np.concatenate((query_points, np.tile(states_t[veh_idx:veh_idx+1, 2:], 
                                                             (len(query_points),1))), axis=-1)
        query_next_point = query_points[-1]
        
        vec_ref_orient = self.get_velocity_field(query_points, states_t)
        vec_ref_orient_next = vec_ref_orient[-1]
        
        query_points = query_points[:-1]
        vec_ref_orient = vec_ref_orient[:-1]
        vec_ref_orient *= self.vector_scale
        
        for i in range(len(query_points)):
            patch_arrow = mpatches.FancyArrow(query_points[i,0], 
                                              query_points[i,1], 
                                              vec_ref_orient[i,0], 
                                              vec_ref_orient[i,1], 
                                              width=0.07, color='gray')
            self.query_points.append(patch_arrow)
            self.ax.add_patch(patch_arrow)
        
        self.patch_next_arrow.set_data(x=query_next_point[0], 
                                       y=query_next_point[1], 
                                       dx=2*vec_ref_orient_next[0], 
                                       dy=2*vec_ref_orient_next[1])
        
        self.orient_next_range.set_center((query_next_point[0], query_next_point[1]))
        self.orient_next_range.set_theta1(np.rad2deg(self.state_next[veh_idx, 2]))
        self.orient_next_range.set_theta2(np.rad2deg(self.state_next[veh_idx, 3]))
