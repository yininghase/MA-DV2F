import numpy as np
from numpy import pi
import torch.nn as nn
import torch
from torch_scatter import scatter_add

def inverse_huber(x, delta=0.2):
    mask = (torch.abs(x)<delta).float()
    return torch.abs(x)*mask + (x**2/(2*delta)+delta/2)*(1-mask)

class WeightedMeanSquaredLoss(nn.Module):
    def __init__(self, weight=[1.0,0.8], device = 'cpu'):
        super().__init__()
        
        self.weights = torch.tensor(weight).to(device)
        self.device = device    
    
    def forward(self, preds, targets, be_static=None):
        loss_1 = torch.sum(((preds - targets)/self.weights)**2, dim=-1)
        loss = torch.mean(loss_1)
        
        if be_static is not None and len(be_static)>0:
            loss_2 = torch.sum((be_static/self.weights)**2, dim=-1)
            loss += torch.mean(loss_2)*0.1
            
        return loss

    
class SelfSuperVisedLearningLoss(nn.Module):
    def __init__(self, safe_distance=1.5, park_distance=5, vehicle_radius=1.5, default_v=2.5,
                 bound=[1.0,0.8], device = 'cpu'):
        super().__init__()
        
        self.dt = 0.2
        self.bound = torch.tensor(bound).to(device)
        self.safe_distance = safe_distance
        self.park_distance = park_distance
        self.vehicle_radius = vehicle_radius
        self.default_v = default_v
        self.device = device
    
    def forward(self, y_pred, y_ref, X_cur, edges, be_static=None):
        
        X_ref = self.vehicle_dynamic(X_cur, y_ref)
        
        veh_idx = X_cur[:,-1] == 0
        mask_1 = torch.norm(X_ref[veh_idx,:2]-X_cur[veh_idx,4:6], dim=-1)-(self.park_distance \
                            +X_cur[veh_idx,3].abs()) > 0
        
        mask_2 = torch.abs(X_ref[veh_idx,3]) >= self.default_v-1e-5
        
        mask_3 = torch.sign(X_ref[veh_idx,3])*(y_ref[...,0]-y_pred[...,0])<=1e-5
                        
        mask = (mask_1 & mask_2 & mask_3).float()

        loss_1 = ((y_ref[...,1]-y_pred[...,1])/self.bound[1])**2
        loss_1 = torch.mean(loss_1)
        loss_2_1 = ((y_ref[...,0]-y_pred[...,0])/self.bound[0])**2
        loss_2_2 = self.get_loss_for_speed(X_ref, y_ref[...,1], y_ref[...,0], y_pred[...,0], edges)
        loss_2 = torch.mean(loss_2_1*(1-mask)+loss_2_2*mask)
        
        loss = loss_1+loss_2
        
        if be_static is not None:
            loss_3 = torch.sum((be_static/self.bound)**2, dim=-1)
            loss += torch.mean(loss_3)*0.1
        
        return loss
    
    def get_loss_for_speed(self, X_ref, theta_ref, pedal_ref, pedal_pred, edges):
        
        veh_idx = X_ref[:,-1] == 0
        veh_idx_edg = veh_idx[edges[0]].float()
        
        speed_pred = X_ref[:,3].clone()
        speed_pred[veh_idx] = speed_pred[veh_idx] + (pedal_pred-pedal_ref) * self.dt
        
        margin_safe_dist_edg = speed_pred[edges[1]].abs()+veh_idx_edg*speed_pred[edges[0]].abs()
        
        safe_dist_edg = margin_safe_dist_edg + self.safe_distance
        safe_dist_edg_pred = safe_dist_edg
        safe_dist_edg_ref = safe_dist_edg
        
        vec_orient = torch.stack((torch.cos(theta_ref), torch.sin(theta_ref)), dim=-1)
        vel_velocity = vec_orient*speed_pred[veh_idx,None]
        
        X_pred = X_ref.clone()
        X_pred[veh_idx,3] = speed_pred[veh_idx].clone()
        X_pred[veh_idx,:2] = X_ref[veh_idx,:2] + vel_velocity*self.dt
        
        loss_tar = torch.norm(X_pred[veh_idx,:2], p=2, dim=-1)
        loss_col = torch.relu(safe_dist_edg_pred + X_pred[edges[0],7] + (veh_idx_edg+1)*self.vehicle_radius - \
                              torch.norm(X_pred[edges[0],:2]-X_pred[edges[1],:2], p=2, dim=-1))
        loss_col = loss_col**2 + loss_col
        loss_col = scatter_add(loss_col, edges[1], dim=0, dim_size=len(X_pred))[veh_idx]
        loss_pred = loss_tar+loss_col
        
        loss_tar = torch.norm(X_ref[veh_idx,:2], p=2, dim=-1)
        loss_col = torch.relu(safe_dist_edg_ref + X_ref[edges[0],7] + (veh_idx_edg+1)*self.vehicle_radius - \
                              torch.norm(X_ref[edges[0],:2]-X_ref[edges[1],:2], p=2, dim=-1))
        loss_col = loss_col**2 + loss_col
        loss_col = scatter_add(loss_col, edges[1], dim=0, dim_size=len(X_ref))[veh_idx]
        loss_ref = loss_tar+loss_col
        
        loss_diff = loss_pred-loss_ref
        mask = (loss_diff>=0).float()
        loss_diff = (loss_diff**2+loss_diff)*mask + loss_diff*(1-mask)
        
        return loss_diff
    
    def vehicle_dynamic(self, X, y):
        veh_idx = X[:,-1] == 0
        obst_idx = ~veh_idx
        current = X[veh_idx,:4]
        control = y
        
        x_t = current[:,0]+current[:,3]*torch.cos(current[:,2])*self.dt
        y_t = current[:,1]+current[:,3]*torch.sin(current[:,2])*self.dt
        psi_t = current[:,2]+current[...,3]*self.dt*torch.tan(control[:,1])/2.0
        psi_t = (psi_t + pi)%(2*pi) - pi
        v_t = 0.99*current[:,3]+control[:,0]*self.dt
        
        next = torch.cat((x_t[...,None], y_t[...,None], 
                          psi_t[...,None], v_t[...,None]), dim=-1)
        
        X_next = torch.empty(X.shape, device=X.device)
        X_next[obst_idx] = X[obst_idx]
        X_next[veh_idx,4:] = X[veh_idx,4:]
        X_next[veh_idx,:4] = next
        
        return X_next 
    
