import torch
import numpy as np
from torch.nn import ReLU, Tanh, BatchNorm1d
from numpy import pi
from torch_geometric.nn import Linear
from torch_scatter import scatter_add
from itertools import permutations

from u_attention_conv import MyTransformerConv


class ConvResidualBlock(torch.nn.Module):
    def __init__(self, io_node_num, hidden_node_num, key_query_len=512):
        super().__init__()

        self.conv1 = MyTransformerConv(io_node_num, hidden_node_num, key_query_len=key_query_len)
        self.bn1 = BatchNorm1d(hidden_node_num)
        self.a1 = ReLU()
        self.conv2 = MyTransformerConv(hidden_node_num, io_node_num, key_query_len=key_query_len)
        self.bn2 = BatchNorm1d(io_node_num)
        self.a2 = ReLU()
         
    def forward(self, x0, edges):
        
        x = self.conv1(x0, edges)
        x = self.bn1(x)
        x = self.a1(x)
        
        x = self.conv2(x, edges)
        x = self.bn2(x)
        x = x+x0
        x = self.a2(x)
                
        return x


class LinearBlock(torch.nn.Module):
    def __init__(self, in_node_num, out_node_num, activation="relu"):
        super().__init__()
        
        self.linear = Linear(in_node_num, out_node_num, weight_initializer='kaiming_uniform')
        self.bn = BatchNorm1d(out_node_num)
        
        if activation == "relu":
            self.a = ReLU()
        elif activation == "tanh":
            self.a = Tanh()
        else:
            raise NotImplementedError("Not implement this type of activation function!")
        
    def forward(self, x, x0=None):
        
        x = self.linear(x)
        x = self.bn(x)
        
        if x0 is not None:
            x=x+x0
        
        x = self.a(x)
        
        return x


class LinearResidualBlock(torch.nn.Module):
    def __init__(self, io_node_num, hidden_node_num):
        super().__init__()
        
        self.linear1 = LinearBlock(io_node_num, hidden_node_num)
        self.linear2 = LinearBlock(hidden_node_num, io_node_num)
    
        
    def forward(self, x0):
        
        x = self.linear1(x0)
        x = self.linear2(x,x0)
        
        return x
       

class IterativeGNNModel(torch.nn.Module):
    def __init__(self, horizon, max_num_vehicles, max_num_obstacles, num_blocks=4,  
                 device='cpu', mode="inference", load_all_simpler=True, 
                 safe_distance=1.5, parking_distance=5, default_v=2.5, vehicle_radius=1.5): 
        super().__init__()
        self.device = device
        self.horizon = horizon
        self.num_blocks = num_blocks
        self.dt = 0.2
        self.input_length = 8
        self.output_length = 2
        self.bound = torch.tensor([1, 0.8]).to(self.device)
        self.mode = mode
        self.load_all_simpler = load_all_simpler
        self.safe_distance = safe_distance
        self.parking_distance = parking_distance
        self.default_v = default_v
        self.veh_radius = vehicle_radius
            
        self.max_num_vehicles = max_num_vehicles
        self.max_num_obstacles = max_num_obstacles
        
        self.edge_template = self.generate_edge_template()
        
        self.block0 = LinearBlock(self.input_length,80)
        self.block1 = ConvResidualBlock(80,160)
        self.block2 = ConvResidualBlock(80,160)
        self.block3 = LinearBlock(80, self.output_length, activation="tanh")
        
    def generate_edge_template(self):
        
        assert self.max_num_vehicles >= 1, \
               'Must have at least one vehicle!'
        
        assert self.max_num_obstacles >= 0, \
               'Number of obstacle should be positive integer!'
               
        edge_template = {}
        
        if self.load_all_simpler:
        
            for num_vehicles in range(1, self.max_num_vehicles + 1):
                for num_obstacles in range(self.max_num_obstacles + 1):
                    
                    edges_vehicles = torch.tensor([[],[]]).long().to(self.device)
                    edges_obstacles = torch.tensor([[],[]]).long().to(self.device)
                    
                    if num_vehicles > 1:
                        all_perm = list(permutations(range(num_vehicles), 2))
                        vehicle_1, vehicle_2 = zip(*all_perm)
                        vehicle_to_vehicle = torch.tensor([vehicle_1, vehicle_2]).to(self.device)
                        edges_vehicles = torch.cat((edges_vehicles, vehicle_to_vehicle),dim=-1)
                    
                    if num_obstacles > 0:
                        obstacles = torch.arange(num_vehicles, num_vehicles+num_obstacles).tile(num_vehicles).to(self.device)
                        vehicles = torch.arange(num_vehicles).repeat_interleave(num_obstacles).to(self.device)
                        obstacle_to_vehicle = torch.cat((obstacles[None,:], vehicles[None,:]),dim=0)
                        edges_obstacles = torch.cat((edges_obstacles, obstacle_to_vehicle),dim=-1)
                    
                    edge_template[(num_vehicles+num_obstacles, num_vehicles)] = [edges_vehicles, edges_obstacles]
        else:
            
            num_vehicles = self.max_num_vehicles
            num_obstacles = self.max_num_obstacles
            
            edges_vehicles = torch.tensor([[],[]]).long().to(self.device)
            edges_obstacles = torch.tensor([[],[]]).long().to(self.device)
            
            if num_vehicles > 1:
                all_perm = list(permutations(range(num_vehicles), 2))
                vehicle_1, vehicle_2 = zip(*all_perm)
                vehicle_to_vehicle = torch.tensor([vehicle_1, vehicle_2]).to(self.device)
                edges_vehicles = torch.cat((edges_vehicles, vehicle_to_vehicle),dim=-1)
            
            if num_obstacles > 0:
                obstacles = torch.arange(num_vehicles, num_vehicles+num_obstacles).tile(num_vehicles).to(self.device)
                vehicles = torch.arange(num_vehicles).repeat_interleave(num_obstacles).to(self.device)
                obstacle_to_vehicle = torch.cat((obstacles[None,:], vehicles[None,:]),dim=0)
                edges_obstacles = torch.cat((edges_obstacles, obstacle_to_vehicle),dim=-1)
            
            edge_template[(num_vehicles+num_obstacles, num_vehicles)] = [edges_vehicles, edges_obstacles]
        
        return edge_template

    
    def get_edges(self, batches):
        
        edges_vehicles = torch.tensor([[],[]]).long().to(self.device)
        edges_obstacles = torch.tensor([[],[]]).long().to(self.device)
        
        batches_offset = torch.cumsum(batches[:,0],dim=0)[:-1]
        batches_offset = torch.cat((torch.tensor([0], device=self.device), batches_offset))
        
        for batch in torch.unique(batches, dim=0):
                
            index = torch.all(batches == batch, dim=-1)
            
            if torch.sum(index) == 0:
                continue
            
            offset = batches_offset[index]
            edges_batch_vehicles, edges_batch_obstacles = self.edge_template[tuple(batch.tolist())]
            
            edges_vehicles = torch.cat([edges_vehicles, (edges_batch_vehicles[:,None,:]+offset[None,:,None]).reshape(2,-1)], dim=-1)
            edges_obstacles = torch.cat([edges_obstacles, (edges_batch_obstacles[:,None,:]+offset[None,:,None]).reshape(2,-1)], dim=-1)
        
        return edges_vehicles, edges_obstacles
    
    def forward_nn(self, x, edges):
        
        x = self.block0(x)
        x = self.block1(x, edges)
        x = self.block2(x, edges)
        x = self.block3(x)
        x = x*self.bound
        
        return x
    
    @torch.no_grad()
    def filter_edges(self, x, edges):
        
        position_ego = x[edges[1],:2]
        position_neighbor = x[edges[0],:2]
        dist = torch.norm(position_ego-position_neighbor, p=2, dim=-1)
        veh_idx_edg = (x[edges[0],7]==0).float()
        margin_threshold = x[edges[1],3].abs() + veh_idx_edg*x[edges[0],3].abs()
        threshold = margin_threshold + 2*self.safe_distance + x[edges[0], 7] \
                  + (veh_idx_edg+1) * self.veh_radius
        
        mask = dist<=threshold
        
        edges_filtered = ((edges.T)[mask]).T
        
        return edges_filtered

    def forward(self, x0, batches, filter_edges=True):
        
        veh_idx = (x0[:,-1] == 0)
        obst_idx = (x0[:,-1] != 0)
        
        edges_vehicles, edges_obstacles = self.get_edges(batches)
        edges_org = torch.cat((edges_vehicles, edges_obstacles), dim=-1)
        
        if filter_edges:
            edges = self.filter_edges(x0, edges_org)
        else:
            edges = edges_org
        
        if self.mode == "supervised training":
            
            assert self.horizon == 1, \
                "In training mode of iterative GNN, the horizon need to be 1!"
            
            x = self.forward_nn(x0, edges)
            x_vehicles = x[veh_idx]
            x_obstacles = x[obst_idx]
            
            controls = [x_vehicles, x_obstacles]
            
            return controls
        
        elif self.mode == "self supervised training": 
            
            targets = self.get_reference(x0, edges)
                                
            current = x0[veh_idx, :4].clone()
            
            x = self.forward_nn(x0.clone(), edges)
            
            controls = x[veh_idx]
            statics = x[obst_idx]
            
            next_pred = self.vehicle_dynamic(current, controls)
            
            x_next_pred = torch.empty(x0.shape, device=self.device)
            x_next_pred[obst_idx] = x0[obst_idx]
            x_next_pred[veh_idx,4:] = x0[veh_idx,4:]
            x_next_pred[veh_idx,:4] = next_pred
            
            return controls, targets, x_next_pred, edges, statics

        elif self.mode == "dynamic velocity field":
            
            targets = self.get_reference(x0, edges)
            
            return targets
        
        else:
            
            x = self.forward_nn(x0, edges)
            controls = x[veh_idx]
        
            return controls
    
    
    @torch.no_grad()
    def get_reference(self, x, edges, pos_tolerance=0.25, ang_tolerance=0.2):
        
        veh_idx = (x[:,-1] == 0)
        obst_idx = (x[:,-1] != 0)
        
        starts = x[:,:4].clone()
        targets = x[:,4:].clone()
        
        uni_orient_starts = torch.stack((torch.cos(starts[...,2]), 
                                         torch.sin(starts[...,2])), dim=-1)
        
        uni_orient_targets = torch.stack((torch.cos(targets[...,2]), 
                                          torch.sin(targets[...,2])), dim=-1)
        
        next_xy = veh_idx[:,None].float()*uni_orient_starts*starts[...,3:4]*self.dt + starts[...,:2]
        
        vec_to_targets = targets[...,:2]-next_xy[...,:2]
        dist_to_targets = torch.norm(vec_to_targets, dim=-1, p=2, keepdim=True)
        
        vec_avoid_collision = torch.zeros((len(next_xy), 2), device=x.device)
        
        vec_to_objs = next_xy[edges[0]]-next_xy[edges[1]]
        vec_to_tars = vec_to_targets[edges[1]].clone()
        
        obj_radius = x[edges[0],7:8]
        veh_idx_edg = veh_idx[edges[0]].float()
        dist_to_objs_center = torch.norm(vec_to_objs, dim=-1, p=2, keepdim=True)
        dist_to_objs = dist_to_objs_center-obj_radius-self.veh_radius*(veh_idx_edg+1)[:,None]
        uni_vec_objs = vec_to_objs/(dist_to_objs_center+1e-8)
        
        margin_safe_dist_edg = x[edges[1],3].abs()+veh_idx_edg*x[edges[0],3].abs()
        safe_dist_edg = (self.safe_distance+margin_safe_dist_edg)[:,None]
        
        mask_collision = torch.sum(vec_to_objs*vec_to_tars, dim=-1)>0
        mask_collision = mask_collision & (dist_to_objs<=safe_dist_edg)[...,0]
        
        vec_back_obj = torch.clip(dist_to_objs-safe_dist_edg, min=-safe_dist_edg, 
                                   max=torch.zeros_like(safe_dist_edg))*uni_vec_objs
        vec_avoid_collision += scatter_add(vec_back_obj, edges[1], dim=0, dim_size=len(x))
                
        vec_to_objs_3d = torch.cat((uni_vec_objs, torch.zeros((len(uni_vec_objs), 1), device=x.device)), dim=-1)
        
        ##### roundabout mode #####
        vec_direct = torch.zeros_like(vec_to_objs_3d)
        vec_direct[...,-1] = 1
        ##### roundabout mode #####
        
        vec_around_objs = torch.cross(vec_direct, vec_to_objs_3d, dim=-1)[...,:2]
        vec_around_objs = vec_around_objs/(torch.norm(vec_around_objs, dim=-1, p=2, keepdim=True)+1e-8)
        vec_around_objs = torch.clip(dist_to_objs, min=torch.zeros_like(safe_dist_edg), 
                                     max=safe_dist_edg)*vec_around_objs
        vec_around_objs[~mask_collision] = 0
        vec_avoid_collision += scatter_add(vec_around_objs, edges[1], dim=0, dim_size=len(x))
        
        uni_vec_targets = vec_to_targets/(dist_to_targets+1e-8)
        vec_to_targets = uni_vec_targets.clone()
        
        mask_stop = (dist_to_targets<pos_tolerance)
        
        factor1 = torch.sum(vec_to_targets*uni_orient_targets, dim=-1, keepdim=True) 
        factor2 = torch.clip(dist_to_targets, min=0, max=self.parking_distance)/self.parking_distance
        factor2[~mask_stop] += 1
        factor = 2*((factor1>=0).float()-0.5)*factor2
        
        mask_parking = dist_to_targets[...,0]<=self.parking_distance
        vec_to_targets[mask_parking] = (uni_orient_targets+factor*vec_to_targets)[mask_parking]
        
        factor = torch.sum(uni_orient_starts*uni_vec_targets, dim=-1, keepdim=True)
        factor = ((factor>=0).float()-0.5)*2
        mask_parking_margin = (dist_to_targets[...,0]>self.parking_distance) & \
                              (dist_to_targets[...,0]<=self.parking_distance + 0.5*self.default_v**2)
        vec_to_targets[mask_parking_margin] = (vec_to_targets*factor)[mask_parking_margin]
    
        vec_to_targets = vec_to_targets/(torch.norm(vec_to_targets, dim=-1, p=2, keepdim=True)+1e-8)
        
        vec_ref_orient = vec_to_targets+vec_avoid_collision
        vec_ref_orient = vec_ref_orient/(torch.norm(vec_ref_orient, dim=-1, p=2, keepdim=True)+1e-8)
        
        ref_orent = torch.arctan2(vec_ref_orient[...,1], vec_ref_orient[...,0])
        
        theta_range = 0.99*starts[...,3].abs()*torch.tan(self.bound[1])*self.dt/2
        
        diff_orient_1 = (ref_orent-starts[...,2])%(2*pi)
        diff_orient_2 = (starts[...,2]-ref_orent)%(2*pi)
        diff_orient = torch.amin(torch.stack((diff_orient_1, diff_orient_2), dim=-1), dim=-1)
        
        mask_diff_orient = diff_orient_1<=diff_orient_2
        
        diff_orient_1 = torch.clip(diff_orient_1, min=torch.zeros_like(theta_range), max=theta_range)
        diff_orient_2 = torch.clip(diff_orient_2, min=torch.zeros_like(theta_range), max=theta_range)
        
        ref_orent = torch.empty_like(starts[...,2])
        ref_orent[mask_diff_orient] = starts[...,2][mask_diff_orient] + diff_orient_1[mask_diff_orient]
        ref_orent[~mask_diff_orient] = starts[...,2][~mask_diff_orient] - diff_orient_2[~mask_diff_orient]
        ref_steer = torch.arctan(2*(ref_orent-starts[...,2])/(self.dt*starts[...,3]+1e-8))
        
        ref_orent = (ref_orent+pi)%(2*pi)-pi
        mask_stop = mask_stop[...,0] & (diff_orient<ang_tolerance)
        
        ref_vel = torch.ones((len(x),), device=x.device)*self.default_v
        factor = torch.clip(torch.clip(dist_to_targets[...,0]/self.parking_distance, min=0, max=1) + 
                            torch.clip(diff_orient/self.default_v, min=0, max=1), min=0, max=1)
        factor[~mask_stop] = factor[~mask_stop].sqrt()
        # factor[mask_stop] = factor[mask_stop]**2
        ref_vel *= factor
        
        mask_parking = mask_parking | mask_parking_margin
        mask_backwards = torch.sum(uni_vec_targets*uni_orient_starts, dim=-1) < -0.25
        ref_vel[mask_backwards & mask_parking] = -ref_vel[mask_backwards & mask_parking].abs()
        
        mask_forwards = torch.sum(uni_vec_targets*uni_orient_starts, dim=-1) > 0.25
        ref_vel[mask_forwards & mask_parking] = ref_vel[mask_forwards & mask_parking].abs()
        
        mask_remain = (~mask_forwards) & (~mask_backwards)
        vel_sign = ((starts[...,3]>=0).float()-0.5)*2
        ref_vel[mask_remain & mask_parking] = (ref_vel.abs()*vel_sign)[mask_remain & mask_parking]
        
        # mask_backwards = torch.sum(uni_vec_targets*uni_orient_starts, dim=-1) < 0
        mask_backwards = torch.sum(vec_ref_orient*uni_orient_starts, dim=-1) < 0
        ref_vel[mask_backwards & (~mask_parking)] *= -1
        
        uni_ref_orent = torch.stack((torch.cos(ref_orent), 
                                     torch.sin(ref_orent)), dim=-1)
        
        ##### double check #####
        mask_unsafe = dist_to_objs[...,0]<(safe_dist_edg[...,0]-1)
        ref_cos_theta = torch.sum(uni_ref_orent[edges[1]]*uni_vec_objs, dim=-1)
        
        mask_forwards = (mask_unsafe & (ref_cos_theta<0)).long()
        mask_forwards = scatter_add(mask_forwards, edges[1], dim=0, dim_size=len(x))>0
        mask_backwards = (mask_unsafe & (ref_cos_theta>0)).long()
        mask_backwards = scatter_add(mask_backwards, edges[1], dim=0, dim_size=len(x))>0
        mask_zero = mask_forwards & mask_backwards
        
        ref_vel[mask_forwards] = ref_vel[mask_forwards].abs()
        ref_vel[mask_backwards] = -(ref_vel[mask_backwards].abs())
        ref_vel[mask_zero] = 0
        
        ref_vel = torch.clip(ref_vel, min=0.99*starts[...,3]-self.bound[0]*self.dt, 
                                      max=0.99*starts[...,3]+self.bound[0]*self.dt)
        ref_pedal = (ref_vel-0.99*starts[...,3])/self.dt
        
        ref_controls = torch.stack((ref_pedal, ref_steer), dim=-1)
        
        return ref_controls[veh_idx]
    
    @staticmethod
    def introduce_random_offset(x, state_quotient, sigma=[1,1,pi/18,0.5]):

        with torch.no_grad():
            veh_idx = (x[:,-1] == 0)
            num_vehicles = torch.sum(veh_idx).item()
            sigma = np.array(sigma)
            
            if isinstance(state_quotient, int):
                offset = np.random.normal(0, sigma*state_quotient, (num_vehicles,4))
            else:
                offset = np.random.normal(np.zeros(((num_vehicles,4))), 
                                          sigma[None,:]*state_quotient[:,None], 
                                          (num_vehicles,4))
                
            offset = torch.from_numpy(offset).float().to(x.device)
            x[veh_idx,:4] += offset
    
        return x
    
    def vehicle_dynamic(self, state, control):
        x_t = state[:,0]+state[:,3]*torch.cos(state[:,2])*self.dt
        y_t = state[:,1]+state[:,3]*torch.sin(state[:,2])*self.dt
        psi_t = state[:,2]+state[...,3]*self.dt*torch.tan(control[:,1])/2.0
        psi_t = (psi_t + pi)%(2*pi) - pi
        v_t = 0.99*state[:,3]+control[:,0]*self.dt
        
        return torch.cat((x_t[...,None], y_t[...,None], psi_t[...,None], v_t[...,None]), dim=-1)

   
