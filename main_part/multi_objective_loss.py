#!/usr/bin/env python
# coding: utf-8
"""
Multi-Objective Loss Functions for Pareto-Adaptive OPF Training

This module extends the unsupervised loss function to support multi-objective 
optimization with preference-weighted cost and carbon emission objectives.

Loss Function:
L = λ_cost * L_cost + λ_carbon * L_carbon + k_g * L_gen + k_Sl * L_branch + k_d * L_load

Where:
- L_cost: Economic generation cost
- L_carbon: Carbon emission (new objective)
- L_gen, L_branch, L_load: Constraint violation penalties

Author: Auto-generated
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Import base unsupervised loss components
from unsupervised_loss import (
    UnsupervisedOPFLoss,
    compute_power_injection,
    compute_branch_power,
    compute_generation_separate,
    cost_loss,
    generator_violation_loss,
    branch_power_violation_loss,
    branch_angle_violation_loss,
    load_deviation_loss,
    AdaptiveWeightScheduler
)


# ==================== Carbon Emission Loss ====================

def carbon_emission_loss(Pg, gci_values, baseMVA):
    """
    Compute carbon emission loss (differentiable).
    
    Carbon Emission = sum_i(GCI_i * Pg_i * baseMVA)
    
    Where GCI_i is the Generator Carbon Intensity (tCO2/MWh) for generator i.
    
    Args:
        Pg: Active generation [batch, n_gen] (p.u.)
        gci_values: Generator Carbon Intensity values [n_gen] (tCO2/MWh)
        baseMVA: Base MVA for conversion
        
    Returns:
        emission: Carbon emission [batch] (tCO2/h)
    """
    # Clamp Pg to non-negative values (negative generation is non-physical)
    Pg_clamped = F.relu(Pg)
    
    # Convert to MW
    Pg_MW = Pg_clamped * baseMVA
    
    # Ensure gci_values is on the same device
    if not isinstance(gci_values, torch.Tensor):
        gci_values = torch.tensor(gci_values, dtype=torch.float32)
    gci_values = gci_values.to(Pg.device)
    
    # Carbon emission: sum_i(GCI_i * Pg_i_MW)
    emission_per_gen = gci_values.unsqueeze(0) * Pg_MW  # [batch, n_gen]
    emission = torch.sum(emission_per_gen, dim=1)  # [batch]
    
    return emission


def normalized_carbon_emission_loss(Pg, gci_values, baseMVA, carbon_scale=1000.0):
    """
    Compute normalized carbon emission loss for better gradient balance.
    
    Normalizes the carbon emission by a scale factor to be comparable with cost.
    
    Args:
        Pg: Active generation [batch, n_gen] (p.u.)
        gci_values: Generator Carbon Intensity values [n_gen] (tCO2/MWh)
        baseMVA: Base MVA for conversion
        carbon_scale: Scale factor for normalization (default: 1000.0)
        
    Returns:
        normalized_emission: Scaled carbon emission [batch]
    """
    emission = carbon_emission_loss(Pg, gci_values, baseMVA)
    return emission * carbon_scale


# ==================== Multi-Objective Loss Class ====================

class MultiObjectiveOPFLoss(nn.Module):
    """
    Multi-Objective Loss for Pareto-Adaptive OPF Training.
    
    Extends UnsupervisedOPFLoss to support preference-weighted multi-objective 
    optimization with both economic cost and carbon emission objectives.
    
    Loss = λ_cost * L_cost + λ_carbon * L_carbon + k_g * L_g + k_Sl * L_Sl + k_θ * L_θ + k_d * L_d
    """
    
    def __init__(self, sys_data, config, gci_values, use_adaptive_weights=False):
        """
        Initialize multi-objective loss module.
        
        Args:
            sys_data: PowerSystemData object containing system parameters
            config: Configuration object
            gci_values: Generator Carbon Intensity values [n_gen] (tCO2/MWh)
            use_adaptive_weights: Whether to use adaptive weight scheduling for constraints
        """
        super().__init__()
        
        self.config = config
        self.use_adaptive_weights = use_adaptive_weights
        
        # Store GCI values
        gci_tensor = torch.from_numpy(gci_values).float() if isinstance(gci_values, np.ndarray) else gci_values.float()
        self.register_buffer('gci_values', gci_tensor)
        
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
        
        # Store Vm bounds for denormalization
        self.register_buffer('VmLb', sys_data.VmLb)
        self.register_buffer('VmUb', sys_data.VmUb)
        
        # Store indices as numpy arrays
        self.bus_Pg = sys_data.bus_Pg  # Active power generator bus indices
        self.bus_Qg = sys_data.bus_Qg  # Reactive power generator bus indices
        self.gen_idx = sys_data.idxPg  # Generator indices in gen array (for cost calculation)
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
        
        # Load bus indices (buses that are not generator buses)
        all_buses = np.arange(config.Nbus)
        self.load_bus_idx = np.setdiff1d(all_buses, self.bus_Pg)
        
        # Constraint weights (from config, based on DeepOPF-NGT paper)
        # Note: k_obj = 0.01 recommended for 300-bus (cost ~500k, constraints ~0-50)
        # Initial constraint weights = 1.0, dynamically adjusted via adaptive scheduler
        self.k_obj = getattr(config, 'k_obj', 0.01)
        self.k_g = getattr(config, 'k_g', 1.0)
        self.k_Sl = getattr(config, 'k_Sl', 1.0)
        self.k_theta = getattr(config, 'k_theta', 1.0)
        self.k_d = getattr(config, 'k_d', 1.0)
        
        # Weight upper bounds (higher limits to give adaptive scheduler more room)
        self.k_g_max = getattr(config, 'k_g_max', 500.0)
        self.k_Sl_max = getattr(config, 'k_Sl_max', 500.0)
        self.k_theta_max = getattr(config, 'k_theta_max', 500.0)
        self.k_d_max = getattr(config, 'k_d_max', 1000.0)
        
        # Carbon emission scale factor (to balance with cost)
        self.carbon_scale = getattr(config, 'carbon_scale', 30.0)
        
        # Initialize weight scheduler for constraints
        # Based on DeepOPF-NGT: k_ti = min(k_obj * L_obj / L_i, k_i_max)
        if use_adaptive_weights:
            self.weight_scheduler = AdaptiveWeightScheduler(
                k_obj=self.k_obj,
                k_g_max=self.k_g_max,
                k_Sl_max=self.k_Sl_max,
                k_theta_max=self.k_theta_max,
                k_d_max=self.k_d_max
            )
            print(f"  Adaptive weights enabled (DeepOPF-NGT)")
            print(f"    k_obj={self.k_obj}, k_g_max={self.k_g_max}, k_Sl_max={self.k_Sl_max}, "
                  f"k_theta_max={self.k_theta_max}, k_d_max={self.k_d_max}")
    
    def forward(self, Vm_pred, Va_pred_no_slack, Pd, Qd, 
                lambda_cost=0.9, lambda_carbon=0.1, 
                update_weights=False, return_details=False):
        """
        Compute multi-objective loss with preference weighting.
        
        Args:
            Vm_pred: Predicted voltage magnitude [batch, n_bus] (scaled)
            Va_pred_no_slack: Predicted voltage angle without slack bus [batch, n_bus-1] (scaled)
            Pd: Active load demand [batch, n_bus] (p.u.)
            Qd: Reactive load demand [batch, n_bus] (p.u.)
            lambda_cost: Weight for economic cost objective (default: 0.9)
            lambda_carbon: Weight for carbon emission objective (default: 0.1)
            update_weights: Whether to update adaptive weights (if enabled)
            return_details: Whether to return detailed loss components
            
        Returns:
            loss: Total loss (scalar)
            loss_dict: Dictionary of individual loss components (if return_details=True)
        """
        device = Vm_pred.device
        batch_size = Vm_pred.shape[0]
        
        # De-normalize predictions
        Vm = Vm_pred / self.config.scale_vm.item() * (self.VmUb - self.VmLb) + self.VmLb
        Va_no_slack = Va_pred_no_slack / self.config.scale_va.item()
        
        # Insert slack bus angle (0)
        Va = torch.zeros(batch_size, self.config.Nbus, device=device)
        Va[:, :self.bus_slack] = Va_no_slack[:, :self.bus_slack]
        Va[:, self.bus_slack+1:] = Va_no_slack[:, self.bus_slack:]
        # Va[:, self.bus_slack] = 0 (already zeros)
        
        # Compute power injection
        P, Q = compute_power_injection(Vm, Va, self.G, self.B)
        
        # Compute generation with separate P and Q bus indices
        Pg, Qg = compute_generation_separate(P, Q, Pd, Qd, self.bus_Pg, self.bus_Qg)
        
        # Compute branch power
        Sf, St = compute_branch_power(Vm, Va, self.Gf, self.Bf, self.Gt, self.Bt, self.Cf, self.Ct)
        
        # ==================== Compute Loss Components ====================
        
        # 1. Economic Cost Objective
        L_cost = cost_loss(Pg, self.gencost, self.baseMVA, self.gen_idx)
        L_cost_mean = torch.mean(L_cost)
        
        # 2. Carbon Emission Objective
        # Use GCI values for the active generators
        gci_for_active_gens = self.gci_values[self.gen_idx]
        L_carbon = carbon_emission_loss(Pg, gci_for_active_gens, self.baseMVA)
        L_carbon_mean = torch.mean(L_carbon) * self.carbon_scale  # Scale for balance
        
        # 3. Generator constraint violation
        L_g = generator_violation_loss(Pg, Qg, self.Pg_min, self.Pg_max, self.Qg_min, self.Qg_max)
        L_g_mean = torch.mean(L_g)
        
        # 4. Branch power violation
        L_Sl = branch_power_violation_loss(Sf, St, self.S_max)
        L_Sl_mean = torch.mean(L_Sl)
        
        # 5. Branch angle violation
        L_theta = branch_angle_violation_loss(Va, self.branch_ft, self.ang_min, self.ang_max)
        L_theta_mean = torch.mean(L_theta)
        
        # 6. Load deviation
        L_d = load_deviation_loss(P, Q, Pd, Qd, self.load_bus_idx)
        L_d_mean = torch.mean(L_d)
        
        # 7. Load satisfaction rate (percentage) - for TensorBoard logging
        # Similar to evaluate_multi_objective.py calculation
        with torch.no_grad():
            # Active power deviation at load buses (absolute)
            P_dev_abs = torch.abs(P[:, self.load_bus_idx] + Pd[:, self.load_bus_idx])  # [batch, n_load]
            Pd_total_error_per_sample = torch.sum(P_dev_abs, dim=1)  # [batch]
            
            # Total load per sample
            Pd_total_per_sample = torch.sum(torch.abs(Pd[:, self.load_bus_idx]), dim=1)  # [batch]
            
            # Relative error
            eps = 1e-6
            Pd_rel_error = Pd_total_error_per_sample / torch.clamp(Pd_total_per_sample, min=eps)
            
            # Mean relative error (percentage)
            Pd_mean_rel_error_pct = torch.mean(Pd_rel_error) * 100.0
            
            # Load satisfaction rate = 100% - mean_relative_error%
            load_satisfy_pct = max(0.0, 100.0 - Pd_mean_rel_error_pct.item())
        
        # 8. Compute preference-weighted satisfaction value
        # Simple weighted sum of cost and carbon objectives (same as L_objective)
        with torch.no_grad():
            # Weighted objective: lambda_cost * cost + lambda_carbon * carbon_scaled
            # This is identical to L_objective calculation below
            satisfaction_value = lambda_cost * L_cost_mean.item() + lambda_carbon * L_carbon_mean.item()
        
        # ==================== Get Constraint Weights ====================
        if self.use_adaptive_weights:
            if update_weights:
                # Use combined objective for adaptive weight computation
                L_obj_combined = lambda_cost * L_cost_mean + lambda_carbon * L_carbon_mean
                # Note: L_z (ZIBs) is 0.0 as we don't have ZIBs constraints in this simplified loss
                self.weight_scheduler.update(L_obj_combined, L_g_mean, L_Sl_mean, L_theta_mean, 0.0, L_d_mean)
            weights = self.weight_scheduler.get_weights()
            k_g = weights['k_g']
            k_Sl = weights['k_Sl']
            k_theta = weights['k_theta']
            k_d = weights['k_d']
        else:
            k_g = self.k_g
            k_Sl = self.k_Sl
            k_theta = self.k_theta
            k_d = self.k_d
        
        # ==================== Combined Multi-Objective Loss ====================
        # Objective: weighted combination of cost and carbon
        L_objective = lambda_cost * L_cost_mean + lambda_carbon * L_carbon_mean
        
        # Constraint penalties
        L_constraints = k_g * L_g_mean + k_Sl * L_Sl_mean + k_theta * L_theta_mean + k_d * L_d_mean
        
        # Total loss
        loss = L_objective + L_constraints
        
        if return_details:
            loss_dict = {
                'total': loss.item(),
                'objective': L_objective.item(),
                'cost': L_cost_mean.item(),
                'carbon': L_carbon_mean.item() / self.carbon_scale,  # Un-scaled for logging
                'cost_weighted': (lambda_cost * L_cost_mean).item(),
                'carbon_weighted': (lambda_carbon * L_carbon_mean).item(),
                'satisfaction': satisfaction_value,  # Preference-weighted satisfaction value
                'gen_vio': L_g_mean.item(),
                'branch_pf_vio': L_Sl_mean.item(),
                'branch_ang_vio': L_theta_mean.item(),
                'load_dev': L_d_mean.item(),
                'load_satisfy_pct': load_satisfy_pct,  # Load satisfaction rate (%)
                'constraints': L_constraints.item(),
                'lambda_cost': lambda_cost,
                'lambda_carbon': lambda_carbon,
                'weights': {
                    'k_g': k_g,
                    'k_Sl': k_Sl,
                    'k_theta': k_theta,
                    'k_d': k_d,
                }
            }
            return loss, loss_dict
        
        return loss


# ==================== GCI Configuration ====================

# GCI Lookup Tables (tCO2/MWh) - from evaluate_multi_objective.py
FUEL_LOOKUP_CO2 = {
    "ANT": 0.9095,  # Anthracite Coal
    "BIT": 0.8204,  # Bituminous Coal
    "Oil": 0.7001,  # Heavy Oil
    "GAS": 0.5173,  # Natural Gas
    "CCGT": 0.3621,  # Gas Combined Cycle
    "ICE": 0.6030,  # Internal Combustion Engine
    "Thermal": 0.6874,  # Thermal Power (General)
    "NUC": 0.0,     # Nuclear Power
    "RE": 0.0,      # Renewable Energy
    "HYD": 0.0,     # Hydropower
    "N/A": 0.0      # Default case
}


def get_gci_for_generators(config, sys_data):
    """
    Assign GCI (Carbon Emission Intensity) values based on marginal generation cost.
    
    Low-cost generators → High carbon (coal)
    High-cost generators → Low carbon (CCGT)
    
    Args:
        config: Configuration object
        sys_data: PowerSystemData object
        
    Returns:
        gci_values: Array of GCI values for each generator [n_gen]
    """
    n_gen = sys_data.gen.shape[0] if isinstance(sys_data.gen, np.ndarray) else sys_data.gen.numpy().shape[0]
    gencost = sys_data.gencost if isinstance(sys_data.gencost, np.ndarray) else sys_data.gencost.numpy()
    
    gci_values = np.zeros(n_gen)
    
    # Get marginal cost coefficient c1 (column 1 in gencost format [c2, c1, ...])
    c1_values = gencost[:n_gen, 1]
    
    # Compute percentiles for classification
    p25 = np.percentile(c1_values, 25)
    p50 = np.percentile(c1_values, 50)
    p75 = np.percentile(c1_values, 75)
    
    for i in range(n_gen):
        c1 = c1_values[i]
        
        if c1 <= p25:
            # Lowest cost quartile → Coal (highest carbon)
            fuel_type = "BIT" if i % 2 == 0 else "ANT"
        elif c1 <= p50:
            # Second quartile → Heavy Oil
            fuel_type = "Oil"
        elif c1 <= p75:
            # Third quartile → Natural Gas
            fuel_type = "GAS"
        else:
            # Highest cost quartile → CCGT (lowest carbon)
            fuel_type = "CCGT"
        
        gci_values[i] = FUEL_LOOKUP_CO2[fuel_type]
    
    return gci_values


# ==================== Module Test ====================

if __name__ == "__main__":
    print("Multi-Objective Loss Module for Pareto-Adaptive OPF Training")
    print("=" * 60)
    print("Components:")
    print("  - carbon_emission_loss: Differentiable carbon emission calculation")
    print("  - normalized_carbon_emission_loss: Scaled carbon emission for gradient balance")
    print("  - MultiObjectiveOPFLoss: Combined multi-objective loss with preference weighting")
    print("  - get_gci_for_generators: Assign GCI values based on generation cost")
    print("\nUsage:")
    print("  loss_fn = MultiObjectiveOPFLoss(sys_data, config, gci_values)")
    print("  loss = loss_fn(Vm_pred, Va_pred, Pd, Qd, lambda_cost=0.9, lambda_carbon=0.1)")

