#!/usr/bin/env python
# coding: utf-8
"""
Unsupervised Loss Functions for DeepOPF Training

This module implements the unsupervised loss function based on DeepOPF-NGT paper:
L = k_obj * L_obj + L_cons + k_d * L_d

Where:
- L_obj: Generation cost minimization
- L_cons: Constraint violation penalties (generator limits, branch flow, angle difference)
- L_d: Load deviation penalty

All calculations are differentiable for gradient-based training.

Author: Auto-generated
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ==================== Differentiable Power System Calculations ====================

def compute_power_injection(Vm, Va, G, B):
    """
    Compute active and reactive power injection at each bus.
    
    Power flow equations:
        P_i = V_i * sum_j(V_j * (G_ij * cos(θ_i - θ_j) + B_ij * sin(θ_i - θ_j)))
        Q_i = V_i * sum_j(V_j * (G_ij * sin(θ_i - θ_j) - B_ij * cos(θ_i - θ_j)))
    
    Args:
        Vm: Voltage magnitudes [batch, n_bus] (p.u.)
        Va: Voltage angles [batch, n_bus] (rad)
        G: Conductance matrix [n_bus, n_bus]
        B: Susceptance matrix [n_bus, n_bus]
        
    Returns:
        P: Active power injection [batch, n_bus] (p.u.)
        Q: Reactive power injection [batch, n_bus] (p.u.)
    """
    batch_size = Vm.shape[0]
    n_bus = Vm.shape[1]
    
    # Angle difference matrix: θ_i - θ_j for all i, j
    # Va is [batch, n_bus], we need [batch, n_bus, n_bus]
    Va_i = Va.unsqueeze(2)  # [batch, n_bus, 1]
    Va_j = Va.unsqueeze(1)  # [batch, 1, n_bus]
    theta_diff = Va_i - Va_j  # [batch, n_bus, n_bus]
    
    # Voltage product matrix: V_i * V_j for all i, j
    Vm_i = Vm.unsqueeze(2)  # [batch, n_bus, 1]
    Vm_j = Vm.unsqueeze(1)  # [batch, 1, n_bus]
    V_product = Vm_i * Vm_j  # [batch, n_bus, n_bus]
    
    # Trigonometric terms
    cos_theta = torch.cos(theta_diff)  # [batch, n_bus, n_bus]
    sin_theta = torch.sin(theta_diff)  # [batch, n_bus, n_bus]
    
    # G and B are [n_bus, n_bus], broadcast to [batch, n_bus, n_bus]
    G_expanded = G.unsqueeze(0)  # [1, n_bus, n_bus]
    B_expanded = B.unsqueeze(0)  # [1, n_bus, n_bus]
    
    # Power injection calculation
    # P_i = sum_j(V_i * V_j * (G_ij * cos(θ_ij) + B_ij * sin(θ_ij)))
    P = torch.sum(V_product * (G_expanded * cos_theta + B_expanded * sin_theta), dim=2)
    
    # Q_i = sum_j(V_i * V_j * (G_ij * sin(θ_ij) - B_ij * cos(θ_ij)))
    Q = torch.sum(V_product * (G_expanded * sin_theta - B_expanded * cos_theta), dim=2)
    
    return P, Q


def compute_branch_power(Vm, Va, Gf, Bf, Gt, Bt, Cf, Ct):
    """
    Compute branch power flow (complex power at from and to ends).
    
    Args:
        Vm: Voltage magnitudes [batch, n_bus] (p.u.)
        Va: Voltage angles [batch, n_bus] (rad)
        Gf, Bf: From-end admittance matrices [n_branch, n_bus]
        Gt, Bt: To-end admittance matrices [n_branch, n_bus]
        Cf: Connection matrix from bus [n_branch, n_bus]
        Ct: Connection matrix to bus [n_branch, n_bus]
        
    Returns:
        Sf: Apparent power at from end [batch, n_branch] (p.u.)
        St: Apparent power at to end [batch, n_branch] (p.u.)
    """
    # Complex voltage: V = Vm * exp(j * Va)
    # In real form: V_real = Vm * cos(Va), V_imag = Vm * sin(Va)
    V_real = Vm * torch.cos(Va)  # [batch, n_bus]
    V_imag = Vm * torch.sin(Va)  # [batch, n_bus]
    
    # Voltage at from bus: V_f = Cf @ V
    Vf_real = torch.matmul(V_real, Cf.T)  # [batch, n_branch]
    Vf_imag = torch.matmul(V_imag, Cf.T)  # [batch, n_branch]
    Vf_mag = torch.sqrt(Vf_real**2 + Vf_imag**2 + 1e-8)
    
    # Voltage at to bus: V_t = Ct @ V
    Vt_real = torch.matmul(V_real, Ct.T)  # [batch, n_branch]
    Vt_imag = torch.matmul(V_imag, Ct.T)  # [batch, n_branch]
    Vt_mag = torch.sqrt(Vt_real**2 + Vt_imag**2 + 1e-8)
    
    # Branch current at from end: I_f = Yf @ V
    # I_f = (Gf + jBf) @ (V_real + jV_imag)
    #     = Gf @ V_real - Bf @ V_imag + j(Gf @ V_imag + Bf @ V_real)
    If_real = torch.matmul(V_real, Gf.T) - torch.matmul(V_imag, Bf.T)
    If_imag = torch.matmul(V_imag, Gf.T) + torch.matmul(V_real, Bf.T)
    
    # Branch current at to end: I_t = Yt @ V
    It_real = torch.matmul(V_real, Gt.T) - torch.matmul(V_imag, Bt.T)
    It_imag = torch.matmul(V_imag, Gt.T) + torch.matmul(V_real, Bt.T)
    
    # Apparent power: S = V * conj(I)
    # S_f = V_f * conj(I_f) = (Vf_real + jVf_imag) * (If_real - jIf_imag)
    Sf_real = Vf_real * If_real + Vf_imag * If_imag  # P_f
    Sf_imag = Vf_imag * If_real - Vf_real * If_imag  # Q_f
    Sf = torch.sqrt(Sf_real**2 + Sf_imag**2 + 1e-8)  # |S_f|
    
    # S_t = V_t * conj(I_t)
    St_real = Vt_real * It_real + Vt_imag * It_imag  # P_t
    St_imag = Vt_imag * It_real - Vt_real * It_imag  # Q_t
    St = torch.sqrt(St_real**2 + St_imag**2 + 1e-8)  # |S_t|
    
    return Sf, St


def compute_generation(P, Q, Pd, Qd, gen_bus_idx):
    """
    Compute generator power output from power balance.
    
    At generator buses: Pg = P + Pd (power injection + load at that bus)
    
    Args:
        P: Active power injection [batch, n_bus] (p.u.)
        Q: Reactive power injection [batch, n_bus] (p.u.)
        Pd: Active load demand [batch, n_bus] (p.u.)
        Qd: Reactive load demand [batch, n_bus] (p.u.)
        gen_bus_idx: Generator bus indices [n_gen]
        
    Returns:
        Pg: Active generation [batch, n_gen] (p.u.)
        Qg: Reactive generation [batch, n_gen] (p.u.)
    """
    # Pg = P_injection + Pd at generator buses
    # This is because P_injection = Pg - Pd, so Pg = P_injection + Pd
    Pg = P[:, gen_bus_idx] + Pd[:, gen_bus_idx]
    Qg = Q[:, gen_bus_idx] + Qd[:, gen_bus_idx]
    
    return Pg, Qg


def compute_generation_separate(P, Q, Pd, Qd, bus_Pg, bus_Qg):
    """
    Compute generator power output from power balance with separate P and Q bus indices.
    
    This handles the case where active and reactive power generators may be at different buses.
    
    Args:
        P: Active power injection [batch, n_bus] (p.u.)
        Q: Reactive power injection [batch, n_bus] (p.u.)
        Pd: Active load demand [batch, n_bus] (p.u.)
        Qd: Reactive load demand [batch, n_bus] (p.u.)
        bus_Pg: Active power generator bus indices [n_gen_P]
        bus_Qg: Reactive power generator bus indices [n_gen_Q]
        
    Returns:
        Pg: Active generation [batch, n_gen_P] (p.u.)
        Qg: Reactive generation [batch, n_gen_Q] (p.u.)
    """
    # Pg = P_injection + Pd at P generator buses
    Pg = P[:, bus_Pg] + Pd[:, bus_Pg]
    # Qg = Q_injection + Qd at Q generator buses
    Qg = Q[:, bus_Qg] + Qd[:, bus_Qg]
    
    return Pg, Qg


# ==================== Loss Function Components ====================

def cost_loss(Pg, gencost, baseMVA, gen_idx):
    """
    Compute generation cost (objective function L_obj).
    
    Cost function: C = sum_i(c2_i * Pg_i^2 + c1_i * Pg_i + c0_i)
    
    Args:
        Pg: Active generation [batch, n_gen] (p.u.)
        gencost: Cost coefficients [n_gen_total, 2 or 3] (columns: c2, c1, [c0])
        baseMVA: Base MVA for conversion
        gen_idx: Generator indices to use
        
    Returns:
        cost: Generation cost [batch] ($/h), always positive
    """
    # Clamp Pg to non-negative values (negative generation is non-physical)
    # This prevents cost from becoming negative
    Pg_clamped = F.relu(Pg)
    
    # Convert to MW
    Pg_MW = Pg_clamped * baseMVA
    
    # Get cost coefficients for active generators
    # gencost format: [c2, c1] or [c2, c1, c0] for quadratic cost
    c2 = torch.from_numpy(gencost[gen_idx, 0]).float().to(Pg.device)
    c1 = torch.from_numpy(gencost[gen_idx, 1]).float().to(Pg.device)
    
    # Quadratic cost: c2 * Pg^2 + c1 * Pg (+ c0 if available)
    cost_per_gen = c2.unsqueeze(0) * Pg_MW**2 + c1.unsqueeze(0) * Pg_MW
    
    # Add c0 if available (gencost has 3 columns)
    if gencost.shape[1] >= 3:
        c0 = torch.from_numpy(gencost[gen_idx, 2]).float().to(Pg.device)
        cost_per_gen = cost_per_gen + c0.unsqueeze(0)
    
    cost = torch.sum(cost_per_gen, dim=1)  # [batch]
    
    # Ensure cost is always positive (should be by design, but clamp for safety)
    cost = F.relu(cost) + 1e-6  # Add small value to avoid zero
    
    return cost


def generator_violation_loss(Pg, Qg, Pg_min, Pg_max, Qg_min, Qg_max):
    """
    Compute generator power limit violation penalty (L_g).
    
    Uses soft penalty: max(0, Pg - Pg_max)^2 + max(0, Pg_min - Pg)^2
    
    Args:
        Pg: Active generation [batch, n_gen] (p.u.)
        Qg: Reactive generation [batch, n_gen] (p.u.)
        Pg_min, Pg_max: Active power limits [n_gen] (p.u.)
        Qg_min, Qg_max: Reactive power limits [n_gen] (p.u.)
        
    Returns:
        loss: Generator violation penalty [batch]
    """
    # Ensure tensors are on same device
    device = Pg.device
    Pg_min = Pg_min.to(device)
    Pg_max = Pg_max.to(device)
    Qg_min = Qg_min.to(device)
    Qg_max = Qg_max.to(device)
    
    # Active power violations (squared penalty)
    Pg_over = F.relu(Pg - Pg_max.unsqueeze(0))  # Over upper limit
    Pg_under = F.relu(Pg_min.unsqueeze(0) - Pg)  # Under lower limit
    Pg_vio = torch.sum(Pg_over**2 + Pg_under**2, dim=1)
    
    # Reactive power violations
    Qg_over = F.relu(Qg - Qg_max.unsqueeze(0))
    Qg_under = F.relu(Qg_min.unsqueeze(0) - Qg)
    Qg_vio = torch.sum(Qg_over**2 + Qg_under**2, dim=1)
    
    return Pg_vio + Qg_vio


def branch_power_violation_loss(Sf, St, S_max):
    """
    Compute branch power flow violation penalty (L_Sl).
    
    Args:
        Sf: Apparent power at from end [batch, n_branch] (p.u.)
        St: Apparent power at to end [batch, n_branch] (p.u.)
        S_max: Branch power limits [n_branch] (p.u.)
        
    Returns:
        loss: Branch power violation penalty [batch]
    """
    device = Sf.device
    S_max = S_max.to(device)
    
    # Violation when |S| > S_max
    Sf_over = F.relu(Sf - S_max.unsqueeze(0))
    St_over = F.relu(St - S_max.unsqueeze(0))
    
    loss = torch.sum(Sf_over**2 + St_over**2, dim=1)
    
    return loss


def branch_angle_violation_loss(Va, branch_ft, ang_min, ang_max):
    """
    Compute branch voltage angle difference violation penalty (L_θl).
    
    Args:
        Va: Voltage angles [batch, n_bus] (rad)
        branch_ft: Branch from-to indices [n_branch, 2]
        ang_min, ang_max: Angle difference limits [n_branch] (rad)
        
    Returns:
        loss: Angle violation penalty [batch]
    """
    device = Va.device
    
    # Get from and to bus indices
    f_bus = branch_ft[:, 0].long()
    t_bus = branch_ft[:, 1].long()
    
    # Angle difference: θ_f - θ_t
    theta_diff = Va[:, f_bus] - Va[:, t_bus]  # [batch, n_branch]
    
    # Violations
    ang_min = ang_min.to(device)
    ang_max = ang_max.to(device)
    
    ang_over = F.relu(theta_diff - ang_max.unsqueeze(0))
    ang_under = F.relu(ang_min.unsqueeze(0) - theta_diff)
    
    loss = torch.sum(ang_over**2 + ang_under**2, dim=1)
    
    return loss


def load_deviation_loss(P, Q, Pd, Qd, load_bus_idx):
    """
    Compute load deviation penalty (L_d).
    
    This ensures predicted load matches actual demand.
    At load buses: P_load_pred = -P_injection (when no generation)
    
    Based on DeepOPF-NGT paper (excluding zero injection buses):
    L_d = sum_{i in N_L/N_Z} [(P_hat_di - P_di)^2 + (Q_hat_di - Q_di)^2]
    
    Args:
        P: Active power injection [batch, n_bus] (p.u.)
        Q: Reactive power injection [batch, n_bus] (p.u.)
        Pd: Active load demand [batch, n_bus] (p.u.)
        Qd: Reactive load demand [batch, n_bus] (p.u.)
        load_bus_idx: Load bus indices to check (non-generator, non-ZIB buses)
        
    Returns:
        loss: Load deviation penalty [batch]
    """
    # At pure load buses (no generation):
    # P_injection = Pg - Pd = 0 - Pd = -Pd
    # So: P_predicted_load = -P_injection should equal Pd
    # Deviation: (P_injection + Pd) should be 0 for load buses
    
    P_dev = (P[:, load_bus_idx] + Pd[:, load_bus_idx])**2
    Q_dev = (Q[:, load_bus_idx] + Qd[:, load_bus_idx])**2
    
    loss = torch.sum(P_dev + Q_dev, dim=1)
    
    return loss


def voltage_magnitude_violation_loss(Vm, Vm_min, Vm_max, bus_idx=None):
    """
    Compute voltage magnitude violation penalty (L_z for ZIBs or general voltage constraint).
    
    Based on DeepOPF-NGT paper:
    L_z = sum_{i in N_Z} [max(V_i - V_max_i, 0)^2 + max(V_min_i - V_i, 0)^2]
    
    This can be applied to:
    - Zero Injection Buses (ZIBs) only: L_z
    - All buses: General voltage constraint violation
    
    Args:
        Vm: Voltage magnitudes [batch, n_bus] (p.u.)
        Vm_min: Minimum voltage magnitude [n_bus] or scalar (p.u.)
        Vm_max: Maximum voltage magnitude [n_bus] or scalar (p.u.)
        bus_idx: Optional bus indices to check. If None, check all buses.
        
    Returns:
        loss: Voltage magnitude violation penalty [batch]
    """
    device = Vm.device
    
    # Convert bounds to tensors if needed
    if isinstance(Vm_min, (int, float)):
        Vm_min = torch.tensor(Vm_min, device=device)
    else:
        Vm_min = Vm_min.to(device)
    
    if isinstance(Vm_max, (int, float)):
        Vm_max = torch.tensor(Vm_max, device=device)
    else:
        Vm_max = Vm_max.to(device)
    
    # Select buses if specified
    if bus_idx is not None:
        Vm_check = Vm[:, bus_idx]
        if Vm_min.dim() > 0 and Vm_min.shape[0] > 1:
            Vm_min = Vm_min[bus_idx]
        if Vm_max.dim() > 0 and Vm_max.shape[0] > 1:
            Vm_max = Vm_max[bus_idx]
    else:
        Vm_check = Vm
    
    # Violations (squared penalty)
    Vm_over = F.relu(Vm_check - Vm_max)   # Over upper limit
    Vm_under = F.relu(Vm_min - Vm_check)  # Under lower limit
    
    loss = torch.sum(Vm_over**2 + Vm_under**2, dim=1)
    
    return loss


# ==================== Adaptive Weight Scheduler ====================

class AdaptiveWeightScheduler:
    """
    Adaptive weight scheduler for balancing multi-term loss functions.
    
    Based on DeepOPF-NGT paper:
    k_i^t = min(k_obj * L_obj / L_i, k_i_max)
    
    This dynamically adjusts weights to balance gradient contributions
    from different loss terms.
    """
    
    def __init__(self, k_obj=0.01, k_g_max=500.0, k_Sl_max=500.0, 
                 k_theta_max=500.0, k_z_max=500.0, k_d_max=1000.0):
        """
        Initialize scheduler with maximum weight limits.
        
        Based on DeepOPF-NGT paper for IEEE 300-bus system:
        - Initial weights: k_0g = k_0Sl = k_0theta = k_0d = 1
        - Update formula: k_ti = min(k_obj * L_obj / L_i, k_i_max)
        - Upper bounds (k_i_max) are tuned via trial-and-error
        
        Args:
            k_obj: Weight for objective (cost) loss (fixed)
            k_g_max: Maximum weight for generator constraint loss
            k_Sl_max: Maximum weight for branch power loss
            k_theta_max: Maximum weight for angle difference loss
            k_z_max: Maximum weight for voltage magnitude constraint loss (ZIBs)
            k_d_max: Maximum weight for load deviation loss
        """
        self.k_obj = k_obj
        self.k_g_max = k_g_max
        self.k_Sl_max = k_Sl_max
        self.k_theta_max = k_theta_max
        self.k_z_max = k_z_max
        self.k_d_max = k_d_max
        
        # Current weights - initialize to 1.0 as per DeepOPF-NGT paper
        # (k_0g = k_0Sl = k_0theta = k_0d = 1)
        self.k_g = 1.0
        self.k_Sl = 1.0
        self.k_theta = 1.0
        self.k_z = 1.0
        self.k_d = 1.0
        
        # Loss history for smoothing (exponential moving average)
        self.L_obj_avg = None
        self.L_g_avg = None
        self.L_Sl_avg = None
        self.L_theta_avg = None
        self.L_z_avg = None
        self.L_d_avg = None
        self.ema_alpha = 0.9  # Smoothing factor (90% old, 10% new for stability)
    
    def update(self, L_obj, L_g, L_Sl, L_theta, L_z, L_d):
        """
        Update weights based on current loss values.
        
        Args:
            L_obj: Current objective (cost) loss (scalar)
            L_g: Current generator constraint loss (scalar)
            L_Sl: Current branch power loss (scalar)
            L_theta: Current angle difference loss (scalar)
            L_z: Current voltage magnitude constraint loss (scalar)
            L_d: Current load deviation loss (scalar)
        """
        # Convert to Python floats and use absolute values to avoid negative weights
        # Use .detach().item() to properly extract scalar from tensor without gradient tracking
        def safe_float(x):
            if hasattr(x, 'detach'):
                return abs(x.detach().item())
            elif hasattr(x, 'item'):
                return abs(x.item())
            else:
                return abs(float(x))
        
        L_obj = safe_float(L_obj)
        L_g = safe_float(L_g)
        L_Sl = safe_float(L_Sl)
        L_theta = safe_float(L_theta)
        L_z = safe_float(L_z)
        L_d = safe_float(L_d)
        
        # Update exponential moving averages
        if self.L_obj_avg is None:
            self.L_obj_avg = L_obj
            self.L_g_avg = L_g
            self.L_Sl_avg = L_Sl
            self.L_theta_avg = L_theta
            self.L_z_avg = L_z
            self.L_d_avg = L_d
        else:
            self.L_obj_avg = self.ema_alpha * self.L_obj_avg + (1 - self.ema_alpha) * L_obj
            self.L_g_avg = self.ema_alpha * self.L_g_avg + (1 - self.ema_alpha) * L_g
            self.L_Sl_avg = self.ema_alpha * self.L_Sl_avg + (1 - self.ema_alpha) * L_Sl
            self.L_theta_avg = self.ema_alpha * self.L_theta_avg + (1 - self.ema_alpha) * L_theta
            self.L_z_avg = self.ema_alpha * self.L_z_avg + (1 - self.ema_alpha) * L_z
            self.L_d_avg = self.ema_alpha * self.L_d_avg + (1 - self.ema_alpha) * L_d
        
        # Compute adaptive weights: k_i = min(k_obj * |L_obj| / L_i, k_i_max)
        # Use absolute value of L_obj and clamp weights to positive range
        eps = 1e-8  # Avoid division by zero
        k_min = 0.1  # Minimum weight to ensure constraints are always penalized
        
        if self.L_g_avg > eps:
            self.k_g = max(k_min, min(self.k_obj * self.L_obj_avg / self.L_g_avg, self.k_g_max))
        
        if self.L_Sl_avg > eps:
            self.k_Sl = max(k_min, min(self.k_obj * self.L_obj_avg / self.L_Sl_avg, self.k_Sl_max))
        
        if self.L_theta_avg > eps:
            self.k_theta = max(k_min, min(self.k_obj * self.L_obj_avg / self.L_theta_avg, self.k_theta_max))
        
        if self.L_z_avg > eps:
            self.k_z = max(k_min, min(self.k_obj * self.L_obj_avg / self.L_z_avg, self.k_z_max))
        
        if self.L_d_avg > eps:
            self.k_d = max(k_min, min(self.k_obj * self.L_obj_avg / self.L_d_avg, self.k_d_max))
    
    def get_weights(self):
        """Return current weights as a dictionary."""
        return {
            'k_obj': self.k_obj,
            'k_g': self.k_g,
            'k_Sl': self.k_Sl,
            'k_theta': self.k_theta,
            'k_z': self.k_z,
            'k_d': self.k_d,
        }
    
    def __repr__(self):
        return (f"AdaptiveWeightScheduler(k_obj={self.k_obj:.2f}, k_g={self.k_g:.2f}, "
                f"k_Sl={self.k_Sl:.2f}, k_theta={self.k_theta:.2f}, k_z={self.k_z:.2f}, k_d={self.k_d:.2f})")


# ==================== Combined Unsupervised Loss ====================

class UnsupervisedOPFLoss(nn.Module):
    """
    Combined unsupervised loss for OPF training.
    
    Based on DeepOPF-NGT paper:
    L = k_obj * L_obj + L_cons + k_d * L_d
    
    Where L_cons = k_g * L_g + k_Sl * L_Sl + k_theta * L_theta + k_z * L_z
    
    Components:
    - L_obj: Generation cost minimization
    - L_g: Generator power limit violations
    - L_Sl: Branch power flow violations
    - L_theta: Branch angle difference violations
    - L_z: Voltage magnitude violations (especially for ZIBs)
    - L_d: Load deviation penalty
    
    This module manages all loss computations and adaptive weight scheduling.
    """
    
    def __init__(self, sys_data, config, use_adaptive_weights=True):
        """
        Initialize unsupervised loss module.
        
        Args:
            sys_data: PowerSystemData object containing system parameters
            config: Configuration object
            use_adaptive_weights: Whether to use adaptive weight scheduling
        """
        super().__init__()
        
        self.config = config
        self.use_adaptive_weights = use_adaptive_weights
        
        # Store system data as buffers (non-trainable)
        self.register_buffer('G', sys_data.G)
        self.register_buffer('B', sys_data.B)
        self.register_buffer('Gf', sys_data.Gf)
        self.register_buffer('Bf', sys_data.Bf)
        self.register_buffer('Gt', sys_data.Gt)
        self.register_buffer('Bt', sys_data.Bt)
        self.register_buffer('Cf', sys_data.Cf)
        self.register_buffer('Ct', sys_data.Ct)
        self.register_buffer('S_max', sys_data.S_max)
        self.register_buffer('Pg_min', sys_data.Pg_min)
        self.register_buffer('Pg_max', sys_data.Pg_max)
        self.register_buffer('Qg_min', sys_data.Qg_min)
        self.register_buffer('Qg_max', sys_data.Qg_max)
        
        # Store indices as numpy arrays
        self.bus_Pg = sys_data.bus_Pg  # Active power generator bus indices
        self.bus_Qg = sys_data.bus_Qg  # Reactive power generator bus indices
        self.gen_bus_idx = sys_data.bus_Pg  # Generator bus indices (for backward compatibility)
        self.gen_idx = sys_data.idxPg  # Generator indices in gen array (for cost calculation)
        self.gen_idx_Qg = sys_data.idxQg  # Reactive generator indices
        self.bus_slack = sys_data.bus_slack  # Slack bus index
        
        # Branch angle limits (from branch data)
        branch = sys_data.branch if isinstance(sys_data.branch, np.ndarray) else sys_data.branch.numpy()
        ang_min = branch[:, 3] * math.pi / 180  # Column 3: ANGMIN
        ang_max = branch[:, 4] * math.pi / 180  # Column 4: ANGMAX
        branch_ft = (branch[:, 0:2] - 1).astype(int)  # From-to indices (0-indexed)
        
        self.register_buffer('ang_min', torch.from_numpy(ang_min).float())
        self.register_buffer('ang_max', torch.from_numpy(ang_max).float())
        self.register_buffer('branch_ft', torch.from_numpy(branch_ft))
        
        # Generator cost coefficients
        self.gencost = sys_data.gencost if isinstance(sys_data.gencost, np.ndarray) else sys_data.gencost.numpy()
        self.baseMVA = sys_data.baseMVA.item() if hasattr(sys_data.baseMVA, 'item') else float(sys_data.baseMVA)
        
        # Voltage magnitude limits
        # Default to [0.9, 1.1] p.u. if not specified
        Vm_min = getattr(sys_data, 'Vm_min', None)
        Vm_max = getattr(sys_data, 'Vm_max', None)
        if Vm_min is None:
            Vm_min = getattr(config, 'VmLb', 0.9)
        if Vm_max is None:
            Vm_max = getattr(config, 'VmUb', 1.1)
        
        # Convert to tensors
        if isinstance(Vm_min, (int, float)):
            self.register_buffer('Vm_min', torch.tensor(Vm_min).float())
        else:
            self.register_buffer('Vm_min', torch.from_numpy(np.array(Vm_min)).float())
        
        if isinstance(Vm_max, (int, float)):
            self.register_buffer('Vm_max', torch.tensor(Vm_max).float())
        else:
            self.register_buffer('Vm_max', torch.from_numpy(np.array(Vm_max)).float())
        
        # Zero Injection Buses (ZIBs): buses with no generation and no load
        # These are identified as buses not in gen_bus_idx and with zero Pd/Qd
        # For now, we'll compute ZIBs dynamically or use all non-generator buses for voltage check
        all_buses = np.arange(config.Nbus)
        
        # Identify ZIBs from bus data if available
        if hasattr(sys_data, 'bus_zib') and sys_data.bus_zib is not None:
            self.zib_idx = sys_data.bus_zib
        else:
            # Default: no ZIBs explicitly identified, use all buses for voltage constraint
            self.zib_idx = None
        
        # Load bus indices (buses that are not generator buses, excluding ZIBs for L_d)
        if self.zib_idx is not None:
            # Exclude both generator buses and ZIBs from load deviation calculation
            non_gen_buses = np.setdiff1d(all_buses, self.gen_bus_idx)
            self.load_bus_idx = np.setdiff1d(non_gen_buses, self.zib_idx)
        else:
            self.load_bus_idx = np.setdiff1d(all_buses, self.gen_bus_idx)
        
        # Initialize weight scheduler
        k_obj = getattr(config, 'k_obj', 1.0)
        k_g_max = getattr(config, 'k_g', 100.0)
        k_Sl_max = getattr(config, 'k_Sl', 100.0)
        k_theta_max = getattr(config, 'k_theta', 100.0)
        k_z_max = getattr(config, 'k_z', 100.0)
        k_d_max = getattr(config, 'k_d', 100.0)
        
        self.weight_scheduler = AdaptiveWeightScheduler(
            k_obj=k_obj,
            k_g_max=k_g_max,
            k_Sl_max=k_Sl_max,
            k_theta_max=k_theta_max,
            k_z_max=k_z_max,
            k_d_max=k_d_max
        )
        
        # Fixed weights (used when adaptive is disabled)
        self.k_obj = k_obj
        self.k_g = getattr(config, 'k_g', 100.0)
        self.k_Sl = getattr(config, 'k_Sl', 100.0)
        self.k_theta = getattr(config, 'k_theta', 100.0)
        self.k_z = getattr(config, 'k_z', 100.0)
        self.k_d = getattr(config, 'k_d', 100.0)
    
    def forward(self, Vm_pred, Va_pred_no_slack, Pd, Qd, update_weights=True):
        """
        Compute unsupervised loss.
        
        Based on DeepOPF-NGT paper:
        L = k_obj * L_obj + k_g * L_g + k_Sl * L_Sl + k_theta * L_theta + k_z * L_z + k_d * L_d
        
        Args:
            Vm_pred: Predicted voltage magnitude [batch, n_bus] (scaled)
            Va_pred_no_slack: Predicted voltage angle without slack bus [batch, n_bus-1] (scaled)
            Pd: Active load demand [batch, n_bus] (p.u.)
            Qd: Reactive load demand [batch, n_bus] (p.u.)
            update_weights: Whether to update adaptive weights
            
        Returns:
            loss: Total loss (scalar)
            loss_dict: Dictionary of individual loss components
        """
        device = Vm_pred.device
        batch_size = Vm_pred.shape[0]
        
        # De-normalize predictions
        Vm = Vm_pred / self.config.scale_vm.item()
        Va_no_slack = Va_pred_no_slack / self.config.scale_va.item()
        
        # Insert slack bus angle (0)
        Va = torch.zeros(batch_size, self.config.Nbus, device=device)
        Va[:, :self.bus_slack] = Va_no_slack[:, :self.bus_slack]
        Va[:, self.bus_slack+1:] = Va_no_slack[:, self.bus_slack:]
        # Va[:, self.bus_slack] = 0 (already zeros)
        
        # Compute power injection
        P, Q = compute_power_injection(Vm, Va, self.G, self.B)
        
        # Compute generation with separate P and Q bus indices
        # This handles the case where idxPg and idxQg have different sizes
        Pg, Qg = compute_generation_separate(P, Q, Pd, Qd, self.bus_Pg, self.bus_Qg)
        
        # Compute branch power
        Sf, St = compute_branch_power(Vm, Va, self.Gf, self.Bf, self.Gt, self.Bt, self.Cf, self.Ct)
        
        # ==================== Compute Loss Components ====================
        
        # 1. Objective: Generation cost (L_obj)
        L_obj = cost_loss(Pg, self.gencost, self.baseMVA, self.gen_idx)
        L_obj_mean = torch.mean(L_obj)
        
        # 2. Generator constraint violation (L_g)
        L_g = generator_violation_loss(Pg, Qg, self.Pg_min, self.Pg_max, self.Qg_min, self.Qg_max)
        L_g_mean = torch.mean(L_g)
        
        # 3. Branch power violation (L_Sl)
        L_Sl = branch_power_violation_loss(Sf, St, self.S_max)
        L_Sl_mean = torch.mean(L_Sl)
        
        # 4. Branch angle violation (L_theta)
        L_theta = branch_angle_violation_loss(Va, self.branch_ft, self.ang_min, self.ang_max)
        L_theta_mean = torch.mean(L_theta)
        
        # 5. Voltage magnitude violation (L_z) - for ZIBs or all buses
        # Apply to ZIBs if explicitly identified, otherwise apply to all buses
        if self.zib_idx is not None and len(self.zib_idx) > 0:
            L_z = voltage_magnitude_violation_loss(Vm, self.Vm_min, self.Vm_max, self.zib_idx)
        else:
            # Apply voltage constraint to all buses (general voltage constraint)
            L_z = voltage_magnitude_violation_loss(Vm, self.Vm_min, self.Vm_max, None)
        L_z_mean = torch.mean(L_z)
        
        # 6. Load deviation (L_d)
        L_d = load_deviation_loss(P, Q, Pd, Qd, self.load_bus_idx)
        L_d_mean = torch.mean(L_d)
        
        # ==================== Get Weights ====================
        if self.use_adaptive_weights:
            if update_weights:
                self.weight_scheduler.update(L_obj_mean, L_g_mean, L_Sl_mean, L_theta_mean, L_z_mean, L_d_mean)
            weights = self.weight_scheduler.get_weights()
            k_obj = weights['k_obj']
            k_g = weights['k_g']
            k_Sl = weights['k_Sl']
            k_theta = weights['k_theta']
            k_z = weights['k_z']
            k_d = weights['k_d']
        else:
            k_obj = self.k_obj
            k_g = self.k_g
            k_Sl = self.k_Sl
            k_theta = self.k_theta
            k_z = self.k_z
            k_d = self.k_d
        
        # ==================== Combined Loss ====================
        # L = k_obj * L_obj + L_cons + k_d * L_d
        # where L_cons = k_g * L_g + k_Sl * L_Sl + k_theta * L_theta + k_z * L_z
        loss = (k_obj * L_obj_mean + 
                k_g * L_g_mean + 
                k_Sl * L_Sl_mean + 
                k_theta * L_theta_mean + 
                k_z * L_z_mean + 
                k_d * L_d_mean)
        
        # Prepare loss dictionary for logging
        loss_dict = {
            'total': loss.item(),
            'cost': L_obj_mean.item(),
            'gen_vio': L_g_mean.item(),
            'branch_pf_vio': L_Sl_mean.item(),
            'branch_ang_vio': L_theta_mean.item(),
            'voltage_vio': L_z_mean.item(),
            'load_dev': L_d_mean.item(),
            'weights': {
                'k_obj': k_obj,
                'k_g': k_g,
                'k_Sl': k_Sl,
                'k_theta': k_theta,
                'k_z': k_z,
                'k_d': k_d,
            }
        }
        
        return loss, loss_dict


# ==================== Utility Functions ====================

def denormalize_voltage(Vm_scaled, Va_scaled, config, VmLb, VmUb):
    """
    Denormalize scaled voltage predictions.
    
    Args:
        Vm_scaled: Scaled voltage magnitude [batch, n_bus]
        Va_scaled: Scaled voltage angle [batch, n_bus-1]
        config: Configuration object
        VmLb, VmUb: Voltage magnitude bounds
        
    Returns:
        Vm: Voltage magnitude in p.u.
        Va: Voltage angle in rad (with slack bus inserted)
    """
    scale_vm = config.scale_vm.item() if hasattr(config.scale_vm, 'item') else float(config.scale_vm)
    scale_va = config.scale_va.item() if hasattr(config.scale_va, 'item') else float(config.scale_va)
    
    # Denormalize Vm: Vm = Vm_scaled / scale_vm * (VmUb - VmLb) + VmLb
    if isinstance(VmLb, torch.Tensor):
        Vm = Vm_scaled / scale_vm * (VmUb - VmLb) + VmLb
    else:
        Vm = Vm_scaled / scale_vm  # Simple denormalization if no bounds
    
    # Denormalize Va
    Va = Va_scaled / scale_va
    
    return Vm, Va


if __name__ == "__main__":
    print("Unsupervised Loss Module for DeepOPF Training")
    print("=" * 60)
    print("Based on DeepOPF-NGT paper:")
    print("  L = k_obj * L_obj + k_g * L_g + k_Sl * L_Sl + k_theta * L_theta + k_z * L_z + k_d * L_d")
    print()
    print("Components:")
    print("  - compute_power_injection: Differentiable power flow calculation")
    print("  - compute_branch_power: Branch power flow calculation")
    print("  - compute_generation: Generator power extraction")
    print("  - cost_loss: Generation cost (L_obj)")
    print("  - generator_violation_loss: Generator limit penalty (L_g)")
    print("  - branch_power_violation_loss: Branch power penalty (L_Sl)")
    print("  - branch_angle_violation_loss: Angle difference penalty (L_theta)")
    print("  - voltage_magnitude_violation_loss: Voltage limit penalty (L_z) - NEW")
    print("  - load_deviation_loss: Load balance penalty (L_d)")
    print("  - AdaptiveWeightScheduler: Dynamic weight balancing")
    print("  - UnsupervisedOPFLoss: Combined loss module")

