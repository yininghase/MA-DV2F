import numpy as np
from numpy import pi
import torch.nn as nn
import torch
from torch_scatter import scatter_add

def inverse_huber(x, delta=0.2):
    """Inverse Huber loss (smooth L1 variant).

    Computes |x| for |x| < delta and (x^2/(2*delta) + delta/2) otherwise.

    Args:
        x (Tensor): Input tensor.
        delta (float): Threshold between linear and quadratic regions.

    Returns:
        Tensor: Loss values with the same shape as x.
    """
    mask = (torch.abs(x)<delta).float()
    return torch.abs(x)*mask + (x**2/(2*delta)+delta/2)*(1-mask)

class WeightedMeanSquaredLoss(nn.Module):
    """Weighted mean squared error loss for supervised training.

    Applies per-output-dimension weights to the MSE, with an optional
    regularisation term for static (obstacle) nodes.
    """
    def __init__(self, weight=[1.0,0.8], device = 'cpu'):
        """Initialise the weighted MSE loss.

        Args:
            weight (list): Per-dimension loss weights.
            device (str): Torch device string.
        
        Returns:
            None.
        """
        super().__init__()
        
        self.weights = torch.tensor(weight).to(device)
        self.device = device    
    
    def forward(self, preds, targets, be_static=None):
        """Compute the weighted MSE loss.

        Args:
            preds (Tensor): Predicted control outputs.
            targets (Tensor): Ground-truth control targets.
            be_static (Tensor, optional): Regularisation penalty for static nodes.

        Returns:
            Tensor: Scalar loss value.
        """
        loss_1 = torch.sum(((preds - targets)/self.weights)**2, dim=-1)
        loss = torch.mean(loss_1)
        
        if be_static is not None and len(be_static)>0:
            loss_2 = torch.sum((be_static/self.weights)**2, dim=-1)
            loss += torch.mean(loss_2)*0.1
            
        return loss

    
class SelfSuperVisedLearningLoss(nn.Module):
    """Self-supervised loss for training the GNN against the DVF reference.

    Compares the GNN-predicted controls to the analytically computed DVF reference
    controls. The loss combines steering-angle MSE and a speed-matching term that
    additionally penalises predicted trajectories that lead to collisions or miss
    the target relative to the DVF reference trajectory.
    """
    def __init__(self, safe_distance=1.5, park_distance=5, vehicle_radius=1.5, default_v=2.5,
                 bound=[1.0,0.8], device = 'cpu'):
        """Initialise the self-supervised loss.

        Args:
            safe_distance (float): Minimum safe distance between agents.
            park_distance (float): Distance threshold for parking mode.
            vehicle_radius (float): Vehicle bounding radius.
            default_v (float): Default reference speed.
            bound (list): Control bounds [max_pedal, max_steering].
            device (str): Torch device string.
        
        Returns:
            None.
        """
        super().__init__()
        
        self.dt = 0.2
        self.bound = torch.tensor(bound).to(device)
        self.safe_distance = safe_distance
        self.park_distance = park_distance
        self.vehicle_radius = vehicle_radius
        self.default_v = default_v
        self.device = device
    
    def forward(self, y_pred, y_ref, X_cur, edges, be_static=None):
        """Compute the self-supervised loss.

        Args:
            y_pred (Tensor): GNN-predicted controls [N, 2].
            y_ref (Tensor): DVF reference controls [N, 2].
            X_cur (Tensor): Current state tensor [N_total, 8].
            edges (list[Tensor, Tensor]): Vehicle and obstacle edge indices.
            be_static (Tensor, optional): Static-node control predictions.

        Returns:
            Tensor: Scalar combined loss (steering + speed + optional static).
        """
        
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
        """Compute the speed sub-loss by comparing predicted and reference trajectories.

        Rolls out both the DVF reference and GNN-predicted controls, then penalises
        cases where the GNN trajectory is worse (higher collision risk or larger
        target error) than the reference.

        Args:
            X_ref (Tensor): Reference state after applying DVF dynamics [N_total, 8].
            theta_ref (Tensor): Reference steering angle [N_veh].
            pedal_ref (Tensor): Reference pedal command [N_veh].
            pedal_pred (Tensor): GNN-predicted pedal command [N_veh].
            edges (list[Tensor, Tensor]): Edge indices.

        Returns:
            Tensor: Per-vehicle speed loss.
        """
        
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
        """Apply the bicycle kinematics model to all vehicles in the batch.

        Args:
            X (Tensor): Current states [N_total, 8] (x, y, psi, v, tx, ty, tpsi, type).
            y (Tensor): Control commands [N_veh, 2] (pedal, steering).

        Returns:
            Tensor: Updated states after one time step [N_total, 8].
        """
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
    
