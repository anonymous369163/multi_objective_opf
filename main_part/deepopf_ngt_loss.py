#!/usr/bin/env python
# coding: utf-8
"""
DeepOPF-NGT Unsupervised Loss Module

This module implements the unsupervised loss function from DeepOPF-NGT paper,
with custom backward function using analytical Jacobian computation.

Key features:
1. Kron Reduction: Only predict non-ZIB nodes, recover ZIB voltages algebraically
2. Custom torch.autograd.Function with analytical gradient computation
3. Adaptive penalty weight scheduling

Reference: DeepOPF-NGT paper (IEEE 300-bus system implementation)

Author: Ported from main_DeepOPFNGT_M3.ipynb
Date: 2024
"""

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
from scipy import sparse 


def matrix_diag_torch(x):
    """
    Create batch diagonal matrices from 2D tensor - PURE PYTORCH VERSION.
    x is [batch, dim], output is [batch, dim, dim] with x on diagonal.
    
    This avoids CPU-GPU transfers and is much faster than the NumPy version.
    """
    # x: [batch, dim]
    batch_size, dim = x.shape
    # Create identity-like mask and multiply
    result = torch.zeros(batch_size, dim, dim, dtype=x.dtype, device=x.device)
    # Use diagonal_scatter or advanced indexing
    idx = torch.arange(dim, device=x.device)
    result[:, idx, idx] = x
    return result
 
class DeepOPFNGTParams:
    """
    Container for DeepOPF-NGT system parameters including Kron Reduction matrices
    """
    def __init__(self):
        # Basic system info
        self.Nbus = None
        self.batch_size = None
        self.device = None
        
        # Node indices
        self.bus_slack = None
        self.bus_Pg = None          # Active power generator buses
        self.bus_Qg = None          # Reactive power generator buses
        self.bus_Pd = None          # Buses with active load
        self.bus_Qd = None          # Buses with reactive load
        self.bus_Pnet_nonPg = None  # Load buses (no generator)
        self.bus_Pnet_nonQg = None  # Load buses (no generator)
        self.bus_Pnet_all = None    # All non-ZIB buses
        self.bus_ZIB_all = None     # Zero injection buses
        self.bus_Pnet_noslack_all = None  # Non-ZIB buses excluding slack
        
        # Prediction dimensions
        self.NPred_Vm = None        # Number of Vm to predict (non-ZIB buses)
        self.NPred_Va = None        # Number of Va to predict (non-ZIB buses - 1)
        self.NZIB = None            # Number of ZIB buses
        
        # Kron Reduction parameters
        self.param_ZIMV = None      # ZIB recovery matrix (complex voltage)
        self.param_ZIM = None       # ZIB gradient parameter matrix
        
        # Index arrays for Jacobian
        self.idx_Pnet = None        # Indices for non-ZIB nodes
        self.idx_ZIB = None         # Indices for ZIB nodes
        self.idx_bus_Pnet_slack = None  # Slack bus position in non-ZIB array
        
        # Jacobian component matrices
        self.Me = None              # Jacobian component (Ybus-based)
        self.Mf = None              # Jacobian component (Ybus-based)
        self.MGB = None             # G-B matrix for power
        self.MBG = None             # B-G matrix for power
        
        # Batch-expanded tensors (for batch training)
        self.Me_re_tensor = None
        self.Mf_re_tensor = None
        self.MGB_re_tensor = None
        self.MBG_re_tensor = None
        self.param_ZIM_tensor_re = None
        
        # Generator limits
        self.MAXMIN_Pg_tensor = None
        self.MAXMIN_Qg_tensor = None
        self.gencost_tensor = None
        
        # Voltage limits
        self.VmLb = None
        self.VmUb = None
        
        # Sparse Ybus for power calculation
        self.Ybus = None
        
        # Penalty coefficients (will be updated adaptively)
        self.kcost = None
        self.kpd = None
        self.kqd = None
        self.kgenp = None
        self.kgenq = None
        self.kv = None
        
        # Coefficient limits
        self.kpd_max = None
        self.kqd_max = None
        self.kgenp_max = None
        self.kgenq_max = None
        self.kv_max = None
        
        # Adaptive flag
        self.flag_k = 2  # 1=fixed, 2=adaptive
        
        # Multi-objective parameters (backward compatible: all have defaults)
        self.use_multi_objective = False  # Default: single-objective (cost only)
        self.lambda_cost = 0.9            # Economic cost weight
        self.lambda_carbon = 0.1          # Carbon emission weight
        self.carbon_scale = 30.0          # Carbon scale factor
        self.gci_tensor = None            # GCI values for carbon emission calculation
        # Multi-objective aggregation options
        self.mo_objective_mode = "weighted_sum"  # "weighted_sum", "normalized_sum", "soft_tchebycheff"
        self.mo_use_running_scale = True
        self.mo_ema_beta = 0.99
        self.mo_tau = 0.2
        self.mo_eps = 1e-8

        # Running scales for normalization
        self._ema_cost = None
        self._ema_carbon_scaled = None


def compute_ngt_params(sys_data, config):
    """
    Compute all parameters needed for DeepOPF-NGT unsupervised training.
    
    This includes:
    - ZIB node identification
    - Kron Reduction matrices
    - Jacobian component matrices
    - Penalty coefficient initialization
    
    Args:
        sys_data: PowerSystemData object from data_loader
        config: Configuration object
        
    Returns:
        DeepOPFNGTParams object with all parameters
    """
    params = DeepOPFNGTParams()
    params.Nbus = config.Nbus
    params.batch_size = config.batch_size_training
    params.device = config.device
    
    # Get Ybus as dense matrix for Kron Reduction
    if sparse.issparse(sys_data.Ybus):
        Ybus_full = sys_data.Ybus.toarray()
    else:
        Ybus_full = np.array(sys_data.Ybus)
    
    # Store sparse Ybus for power calculations
    if sparse.issparse(sys_data.Ybus):
        params.Ybus = sys_data.Ybus
    else:
        params.Ybus = sparse.csr_matrix(sys_data.Ybus)
    
    # ============================================================
    # Step 1: Identify node types
    # ============================================================
    params.bus_slack = int(sys_data.bus_slack)
    params.bus_Pg = sys_data.bus_Pg.astype(int) if isinstance(sys_data.bus_Pg, np.ndarray) else np.array(sys_data.bus_Pg).astype(int)
    params.bus_Qg = sys_data.bus_Qg.astype(int) if isinstance(sys_data.bus_Qg, np.ndarray) else np.array(sys_data.bus_Qg).astype(int)
    
    # Get load bus indices from RPd/RQd first row
    RPd_sample = sys_data.RPd[0, :] if len(sys_data.RPd.shape) > 1 else sys_data.RPd
    RQd_sample = sys_data.RQd[0, :] if len(sys_data.RQd.shape) > 1 else sys_data.RQd
    
    params.bus_Pd = np.squeeze(np.where(np.abs(RPd_sample) > 0), axis=0)
    params.bus_Qd = np.squeeze(np.where(np.abs(RQd_sample) > 0), axis=0)
    
    # Ensure 1D arrays
    if params.bus_Pd.ndim == 0:
        params.bus_Pd = np.array([params.bus_Pd.item()])
    if params.bus_Qd.ndim == 0:
        params.bus_Qd = np.array([params.bus_Qd.item()])
    
    # Find load buses that are not generator buses (for load deviation constraint)
    Pnet_nonPg = RPd_sample.copy()
    Pnet_nonQg = RQd_sample.copy()
    Pnet_nonPg[params.bus_Pg] = 0
    Pnet_nonQg[params.bus_Qg] = 0
    params.bus_Pnet_nonPg = np.squeeze(np.where(np.abs(Pnet_nonPg) > 0), axis=0)
    params.bus_Pnet_nonQg = np.squeeze(np.where(np.abs(Pnet_nonQg) > 0), axis=0)
    
    # Ensure 1D arrays
    if params.bus_Pnet_nonPg.ndim == 0:
        params.bus_Pnet_nonPg = np.array([params.bus_Pnet_nonPg.item()])
    if params.bus_Pnet_nonQg.ndim == 0:
        params.bus_Pnet_nonQg = np.array([params.bus_Pnet_nonQg.item()])
    
    # Generator bus indices (all generators including those with zero output)
    bus_gen = sys_data.gen[:, 0].astype(int) - 1
    
    # Find non-ZIB buses (have either load or generation)
    Pnet = RPd_sample.copy()
    Pnet[bus_gen] = Pnet[bus_gen] + 10  # Mark generator buses
    
    params.bus_Pnet_all = np.squeeze(np.where(np.abs(Pnet) > 0), axis=0)
    params.bus_ZIB_all = np.squeeze(np.where(np.abs(Pnet) == 0), axis=0)
    
    # Ensure 1D arrays
    if params.bus_Pnet_all.ndim == 0:
        params.bus_Pnet_all = np.array([params.bus_Pnet_all.item()])
    if params.bus_ZIB_all.ndim == 0:
        params.bus_ZIB_all = np.array([params.bus_ZIB_all.item()])
    
    params.NZIB = len(params.bus_ZIB_all)
    
    # Find slack bus position in non-ZIB array
    idx_bus_Pnet_slack = np.where(params.bus_Pnet_all == params.bus_slack)[0]
    params.idx_bus_Pnet_slack = idx_bus_Pnet_slack
    
    # Non-ZIB buses excluding slack (for Va prediction)
    params.bus_Pnet_noslack_all = np.delete(params.bus_Pnet_all, idx_bus_Pnet_slack, axis=0)
    
    # Prediction dimensions
    params.NPred_Vm = len(params.bus_Pnet_all)
    params.NPred_Va = len(params.bus_Pnet_noslack_all)
    
    # Index arrays for Jacobian (combining real and imaginary parts)
    params.idx_Pnet = np.concatenate((params.bus_Pnet_all, params.bus_Pnet_all + config.Nbus), axis=0)
    params.idx_ZIB = np.concatenate((params.bus_ZIB_all, params.bus_ZIB_all + config.Nbus), axis=0)
    
    
    # ============================================================
    # Step 2: Kron Reduction matrices
    # ============================================================
    if params.NZIB > 0:
        # Complex voltage recovery matrix: V_ZIB = param_ZIMV @ V_nonZIB
        Yyy = Ybus_full[np.ix_(params.bus_ZIB_all, params.bus_ZIB_all)]
        Yyx = Ybus_full[np.ix_(params.bus_ZIB_all, params.bus_Pnet_all)]
        params.param_ZIMV = -np.linalg.inv(Yyy) @ Yyx
        
        # Gradient parameter matrix for ZIB (in Cartesian coordinates)
        Gba = np.real(Yyx)
        Bba = np.imag(Yyx)
        Gbb = np.real(Yyy)
        Bbb = np.imag(Yyy)
        
        Ax_r1 = np.concatenate((Gba, -Bba), axis=1)
        Ax_r2 = np.concatenate((Bba, Gba), axis=1)
        Ax = np.concatenate((Ax_r1, Ax_r2), axis=0)
        
        Ay_r1 = np.concatenate((Gbb, -Bbb), axis=1)
        Ay_r2 = np.concatenate((Bbb, Gbb), axis=1)
        Ay = np.concatenate((Ay_r1, Ay_r2), axis=0)
        
        params.param_ZIM = -np.linalg.inv(Ay) @ Ax
    else:
        params.param_ZIMV = None
        params.param_ZIM = None
    
    # ============================================================
    # Step 3: Jacobian component matrices
    # ============================================================
    # Me, Mf matrices for Jacobian in Cartesian coordinates
    Me1 = np.concatenate((Ybus_full.real, -Ybus_full.imag), axis=1)
    Me2 = np.concatenate((-Ybus_full.imag, -Ybus_full.real), axis=1)
    params.Me = np.concatenate((Me1, Me2), axis=0)
    
    Mf1 = np.concatenate((Ybus_full.imag, Ybus_full.real), axis=1)
    Mf2 = np.concatenate((Ybus_full.real, -Ybus_full.imag), axis=1)
    params.Mf = np.concatenate((Mf1, Mf2), axis=0)
    
    # G-B and B-G matrices for power calculation
    params.MGB = np.concatenate((Ybus_full.real, -Ybus_full.imag), axis=1)
    params.MBG = np.concatenate((Ybus_full.imag, Ybus_full.real), axis=1)
    
    # ============================================================
    # Step 4: Prepare batch-expanded tensors
    # ============================================================
    batch_size = config.batch_size_training
    
    # Expand and convert to tensors
    Me_expd = np.expand_dims(params.Me, axis=0)
    Mf_expd = np.expand_dims(params.Mf, axis=0)
    Me_re = np.repeat(Me_expd, batch_size, axis=0)
    Mf_re = np.repeat(Mf_expd, batch_size, axis=0)
    
    MGB_expd = np.expand_dims(params.MGB, axis=0)
    MBG_expd = np.expand_dims(params.MBG, axis=0)
    MGB_re = np.repeat(MGB_expd, batch_size, axis=0)
    MBG_re = np.repeat(MBG_expd, batch_size, axis=0)
    
    params.Me_re_tensor = torch.from_numpy(Me_re).float()
    params.Mf_re_tensor = torch.from_numpy(Mf_re).float()
    params.MGB_re_tensor = torch.from_numpy(MGB_re).float()
    params.MBG_re_tensor = torch.from_numpy(MBG_re).float()
    
    if params.param_ZIM is not None:
        param_ZIM_tensor = torch.from_numpy(params.param_ZIM).float()
        param_ZIM_tensor_expd = torch.unsqueeze(param_ZIM_tensor, dim=0)
        params.param_ZIM_tensor_re = param_ZIM_tensor_expd.repeat_interleave(batch_size, dim=0)
    else:
        params.param_ZIM_tensor_re = None
    
    # ============================================================
    # Step 5: Generator limits and cost coefficients
    # ============================================================
    baseMVA = sys_data.baseMVA.item() if hasattr(sys_data.baseMVA, 'item') else float(sys_data.baseMVA)
    
    # MAXMIN_Pg: column 0 is max, column 1 is min
    MAXMIN_Pg = sys_data.gen[sys_data.idxPg, 3:5] / baseMVA
    MAXMIN_Qg = sys_data.gen[sys_data.idxQg, 1:3] / baseMVA
    params.MAXMIN_Pg_tensor = torch.from_numpy(MAXMIN_Pg).float()
    params.MAXMIN_Qg_tensor = torch.from_numpy(MAXMIN_Qg).float()
    
    # Generation cost coefficients
    # Reference code uses gencost_Pg = gencost[idxPg, :] with shape (Npg, 2)
    # where columns are [c2, c1] for cost = c2*Pg^2 + c1*Pg
    gencost = sys_data.gencost
    if gencost.shape[1] > 4:
        # MATPOWER format with header columns: MODEL, STARTUP, SHUTDOWN, NCOST, c2, c1, c0
        # Extract only [c2, c1] (columns 4 and 5)
        gencost_Pg = gencost[sys_data.idxPg, 4:6]
    elif gencost.shape[1] >= 2:
        # Simplified format with just [c2, c1] or [c2, c1, ...]
        gencost_Pg = gencost[sys_data.idxPg, :2]
    else:
        raise ValueError(f"gencost must have at least 2 columns, got {gencost.shape[1]}")
    
    # Ensure shape is (Npg, 2)
    assert gencost_Pg.shape[1] == 2, f"gencost_Pg should have 2 columns, got {gencost_Pg.shape}"
    params.gencost_tensor = torch.from_numpy(gencost_Pg).float()
    
    # Voltage limits
    if hasattr(sys_data, 'VmLb') and sys_data.VmLb is not None:
        if isinstance(sys_data.VmLb, torch.Tensor):
            params.VmLb = sys_data.VmLb.clone()
            params.VmUb = sys_data.VmUb.clone()
        else:
            params.VmLb = torch.tensor([sys_data.VmLb.item() if hasattr(sys_data.VmLb, 'item') else float(sys_data.VmLb)])
            params.VmUb = torch.tensor([sys_data.VmUb.item() if hasattr(sys_data.VmUb, 'item') else float(sys_data.VmUb)])
    else:
        # Default voltage limits
        params.VmLb = torch.tensor([0.94])
        params.VmUb = torch.tensor([1.06])
    
    # ============================================================
    # Step 6: Initialize penalty coefficients
    # ============================================================
    params.kcost = getattr(config, 'ngt_kcost', 0.0002)
    params.flag_k = getattr(config, 'ngt_flag_k', 2)  # 2 = adaptive
    # Objective weight multiplier (to balance objective vs constraints)
    params.obj_weight_multiplier = getattr(config, 'ngt_obj_weight_multiplier', 1.0)
    
    # Coefficient limits
    params.kpd_max = torch.tensor([getattr(config, 'ngt_kpd_max', 100.0)])
    params.kqd_max = torch.tensor([getattr(config, 'ngt_kqd_max', 100.0)])
    params.kgenp_max = torch.tensor([getattr(config, 'ngt_kgenp_max', 2000.0)])
    params.kgenq_max = torch.tensor([getattr(config, 'ngt_kgenq_max', 2000.0)])
    params.kv_max = torch.tensor([getattr(config, 'ngt_kv_max', 500.0)])
    
    # Initial coefficients
    Npg = len(params.bus_Pg)
    Nqg = len(params.bus_Qg)
    Npd = len(params.bus_Pnet_nonPg)
    Nqd = len(params.bus_Pnet_nonQg)
    
    if params.flag_k == 2:
        # Adaptive: start with 1.0
        params.kpd = torch.ones(Npd)
        params.kqd = torch.ones(Nqd)
        params.kgenp = torch.ones(Npg)
        params.kgenq = torch.ones(Nqg)
        params.kv = torch.ones(params.NZIB) if params.NZIB > 0 else torch.zeros(1)
    else:
        # Fixed
        params.kpd = torch.ones(Npd) * 100.0
        params.kqd = torch.ones(Nqd) * 100.0
        params.kgenp = torch.ones(Npg) * 2000.0
        params.kgenq = torch.ones(Nqg) * 2000.0
        params.kv = torch.ones(params.NZIB) * 100.0 if params.NZIB > 0 else torch.zeros(1)
    
    # ============================================================
    # Step 7: Multi-objective setup (only if enabled, backward compatible)
    # ============================================================
    params.use_multi_objective = getattr(config, 'ngt_use_multi_objective', False)
    
    if params.use_multi_objective:
        params.lambda_cost = getattr(config, 'ngt_lambda_cost', 0.9)
        params.lambda_carbon = getattr(config, 'ngt_lambda_carbon', 0.1)
        params.carbon_scale = getattr(config, 'ngt_carbon_scale', 30.0)
        from utils import get_gci_for_generators
        gci_values = get_gci_for_generators(sys_data)
        gci_for_Pg = gci_values[sys_data.idxPg]  # Only for active generators
        params.gci_tensor = torch.from_numpy(gci_for_Pg).float()
        # Multi-objective aggregation parameters
        params.mo_objective_mode = getattr(config, 'ngt_mo_objective_mode', 'weighted_sum')
        params.mo_use_running_scale = getattr(config, 'ngt_mo_use_running_scale', True)
        params.mo_ema_beta = getattr(config, 'ngt_mo_ema_beta', 0.99)
        params.mo_tau = getattr(config, 'ngt_mo_tau', 0.2)
        params.mo_eps = getattr(config, 'ngt_mo_eps', 1e-8)
    
    return params


def create_penalty_v_class(params):
    """
    Create the Penalty_V class as a closure over params.
    
    This is necessary because torch.autograd.Function cannot have instance attributes,
    so we create the class dynamically with params in the closure.
    
    Args:
        params: DeepOPFNGTParams object
        
    Returns:
        Penalty_V class
    """
    
    class Penalty_V(Function):
        """
        Custom autograd function for DeepOPF-NGT unsupervised loss.
        
        Forward:
        - Recover ZIB node voltages using Kron Reduction
        - Compute power injection and generation
        - Calculate constraint violation losses
        - Adaptively update penalty coefficients
        
        Backward:
        - Use analytical Jacobian for gradient computation
        - Handle Kron Reduction chain rule
        """
        
        @staticmethod
        def forward(ctx, V, PQd, preference):
            """
            Forward pass: compute loss and save tensors for backward.
            
            Args:
                V: Predicted voltages [batch, NPred_Va + NPred_Vm]
                   First NPred_Va elements are Va (without slack)
                   Next NPred_Vm elements are Vm (non-ZIB buses)
                PQd: Load data [batch, num_Pd + num_Qd] in p.u.
                preference: Preference tensor for multi-objective optimization
                
            Returns:
                loss: Total unsupervised loss (scalar)
            """
            # Get only_obj flag from params (set by DeepOPFNGTLoss.forward)
            only_obj = getattr(params, '_only_obj', False)
            device = V.device
            Nsam = V.shape[0]
            Nbus = params.Nbus
            
            # Violation threshold
            kdelta = 1e-4
            
            # Insert slack bus Va (=0) to get full non-ZIB voltage
            V_np = V.detach().cpu().numpy()
            xam_P = np.insert(V_np, params.idx_bus_Pnet_slack[0], 0, axis=1)
            
            # Extract Va and Vm (for non-ZIB buses)
            Va_nonZIB = xam_P[:, :params.NPred_Vm]  # All non-ZIB Va including slack (now 0)
            Vm_nonZIB = xam_P[:, params.NPred_Vm:]  # All non-ZIB Vm
            
            # Convert to complex voltage: V = Vm * exp(j*Va)
            Vx = Vm_nonZIB * np.exp(1j * Va_nonZIB)
            
            # Recover ZIB node voltages using Kron Reduction
            if params.NZIB > 0 and params.param_ZIMV is not None:
                Vy = np.dot(params.param_ZIMV, Vx.T).T
            else:
                Vy = None
            
            # Build full voltage vector
            Ve = np.zeros((Nsam, Nbus))  # Real part
            Vf = np.zeros((Nsam, Nbus))  # Imaginary part
            Ve[:, params.bus_Pnet_all] = Vx.real
            Vf[:, params.bus_Pnet_all] = Vx.imag
            if Vy is not None:
                Ve[:, params.bus_ZIB_all] = Vy.real
                Vf[:, params.bus_ZIB_all] = Vy.imag
            
            Pred_V = Ve + 1j * Vf
            
            # Parse load data
            PQd_np = PQd.detach().cpu().numpy()
            num_Pd = len(params.bus_Pd)
            
            Pdtest = np.zeros((Nsam, Nbus))
            Qdtest = np.zeros((Nsam, Nbus))
            Pdtest[:, params.bus_Pd] = PQd_np[:, :num_Pd]
            Qdtest[:, params.bus_Qd] = PQd_np[:, num_Pd:]
            
            # Convert to tensors for loss calculation
            Pdtest_tensor = torch.from_numpy(Pdtest).float().to(device)
            Qdtest_tensor = torch.from_numpy(Qdtest).float().to(device)
            
            # Calculate power injection: S = V * conj(I) = V * conj(Y @ V)
            Pred_S = np.zeros((Nsam, Nbus), dtype=np.complex128)
            for i in range(Nsam):
                I = params.Ybus.dot(Pred_V[i]).conj()
                Pred_S[i] = np.multiply(Pred_V[i], I)
            
            Pred_P = torch.from_numpy(np.real(Pred_S)).float().to(device)
            Pred_Q = torch.from_numpy(np.imag(Pred_S)).float().to(device)
            
            # Generator output: Pg = P + Pd at generator buses
            Pg = Pred_P + Pdtest_tensor
            Qg = Pred_Q + Qdtest_tensor
            
            # Move tensors to device
            MAXMIN_Pg_tensor = params.MAXMIN_Pg_tensor.to(device)
            MAXMIN_Qg_tensor = params.MAXMIN_Qg_tensor.to(device)
            gencost_tensor = params.gencost_tensor.to(device)
            VmLb = params.VmLb.to(device)
            VmUb = params.VmUb.to(device)
            
            # ==================== Loss Components ====================
            
            # 1. Generator P limits: max(0, Pg - Pgmax)^2 + max(0, Pgmin - Pg)^2
            loss_Pgi = torch.sum(
                torch.clamp(Pg[:, params.bus_Pg] - MAXMIN_Pg_tensor[:, 0], min=0).pow(2) +
                torch.clamp(MAXMIN_Pg_tensor[:, 1] - Pg[:, params.bus_Pg], min=0).pow(2),
                dim=0
            )
            
            # 2. Generator Q limits
            loss_Qgi = torch.sum(
                torch.clamp(Qg[:, params.bus_Qg] - MAXMIN_Qg_tensor[:, 0], min=0).pow(2) +
                torch.clamp(MAXMIN_Qg_tensor[:, 1] - Qg[:, params.bus_Qg], min=0).pow(2),
                dim=0
            )
            
            # 3. Load deviation (non-generator buses should have Pg=0, Qg=0)
            loss_Pdi = torch.sum(Pg[:, params.bus_Pnet_nonPg].pow(2), dim=0)
            loss_Qdi = torch.sum(Qg[:, params.bus_Pnet_nonQg].pow(2), dim=0)
            
            # 4. Generation cost: c2*Pg^2 + c1*|Pg| (with extra penalty for negative Pg)
            absPg = torch.where(
                Pg[:, params.bus_Pg] > 0,
                Pg[:, params.bus_Pg],
                -Pg[:, params.bus_Pg] * 2.0
            )
            loss_Pgcost = (
                gencost_tensor[:, 0] * torch.pow(Pg[:, params.bus_Pg], 2) +
                gencost_tensor[:, 1] * absPg
            ).sum()
            
            # 4.5 Multi-objective: Carbon emission loss + objective aggregation
            if params.use_multi_objective:
                # Parse preference tensor
                if preference is None or preference.numel() == 0:
                    lam_cost = torch.full((Nsam,), float(params.lambda_cost), device=device, dtype=Pg.dtype)
                    lam_carbon = torch.full((Nsam,), float(params.lambda_carbon), device=device, dtype=Pg.dtype)
                else: 
                    lam_cost = preference[:, 0]
                    lam_carbon = preference[:, 1]

                # Carbon emission calculation
                gci_tensor = params.gci_tensor.to(device)
                Pg_clamped = torch.clamp(Pg[:, params.bus_Pg], min=0)
                carbon_per = torch.sum(Pg_clamped * gci_tensor.unsqueeze(0), dim=1)
                carbon_scaled_per = carbon_per * params.carbon_scale
                loss_carbon_scaled = torch.sum(carbon_scaled_per)

                # Cost calculation (per-sample)
                cost_per = torch.sum(gencost_tensor[:, 0].unsqueeze(0) * (Pg[:, params.bus_Pg] ** 2)
                                  + gencost_tensor[:, 1].unsqueeze(0) * absPg, dim=1)

                # Update EMA scales for normalization
                cur_cost = float(cost_per.mean().detach().cpu().item())
                cur_carbon = float(carbon_scaled_per.mean().detach().cpu().item())
                if params.mo_use_running_scale:
                    if params._ema_cost is None:
                        params._ema_cost = cur_cost
                    else:
                        params._ema_cost = params.mo_ema_beta * params._ema_cost + (1 - params.mo_ema_beta) * cur_cost
                    if params._ema_carbon_scaled is None:
                        params._ema_carbon_scaled = cur_carbon
                    else:
                        params._ema_carbon_scaled = params.mo_ema_beta * params._ema_carbon_scaled + (1 - params.mo_ema_beta) * cur_carbon
                    scale_cost = max(params._ema_cost, params.mo_eps)
                    scale_carbon = max(params._ema_carbon_scaled, params.mo_eps)
                else:
                    scale_cost = max(cur_cost, params.mo_eps)
                    scale_carbon = max(cur_carbon, params.mo_eps)

                # Objective aggregation modes
                if params.mo_objective_mode == 'normalized_sum':
                    cost_norm = cost_per / scale_cost
                    carbon_norm = carbon_scaled_per / scale_carbon
                    loss_obj_per = scale_cost * (lam_cost * cost_norm + lam_carbon * carbon_norm)
                    w_cost_eff = lam_cost
                    w_carbon_total = lam_carbon * params.carbon_scale * (scale_cost / scale_carbon)
                elif params.mo_objective_mode == 'soft_tchebycheff':
                    tau = max(float(params.mo_tau), params.mo_eps)
                    a = lam_cost * (cost_per / scale_cost)
                    b = lam_carbon * (carbon_scaled_per / scale_carbon)
                    logits = torch.stack([a, b], dim=1) / tau
                    w = torch.softmax(logits, dim=1)
                    loss_obj_per = scale_cost * (tau * torch.logsumexp(logits, dim=1)) 
                    w_cost_eff = w[:, 0] * lam_cost
                    w_carbon_total = w[:, 1] * lam_carbon * params.carbon_scale * (scale_cost / scale_carbon)
                else:
                    # 'weighted_sum' (default)
                    loss_obj_per = lam_cost * cost_per + lam_carbon * carbon_scaled_per
                    w_cost_eff = lam_cost
                    w_carbon_total = lam_carbon * params.carbon_scale

                loss_obj = torch.sum(loss_obj_per)
                loss_carbon = loss_carbon_scaled
            else:
                # Single-objective: only cost
                loss_obj = loss_Pgcost
                loss_carbon = torch.tensor(0.0).to(device)
                w_cost_eff = torch.ones((Nsam,), device=device, dtype=Pg.dtype)
                w_carbon_total = torch.zeros((Nsam,), device=device, dtype=Pg.dtype)
            
            # Store for logging
            params._loss_cost = loss_Pgcost.detach().item()
            params._loss_carbon = loss_carbon.detach().item()
            
            # Store per-sample objective values for multi-objective analysis
            if params.use_multi_objective:
                params._cost_per_mean = cost_per.mean().detach().item()
                params._carbon_per_mean = carbon_scaled_per.mean().detach().item()
            else:
                params._cost_per_mean = loss_Pgcost.detach().item() / Nsam
                params._carbon_per_mean = 0.0
            
            # 5. ZIB voltage violation
            if params.NZIB > 0:
                Vm_ZIB = torch.from_numpy(
                    np.sqrt(Ve[:, params.bus_ZIB_all]**2 + Vf[:, params.bus_ZIB_all]**2)
                ).float().to(device)
                loss_Vi = torch.sum(
                    torch.clamp(VmLb[0] - Vm_ZIB, min=0).pow(2) +
                    torch.clamp(Vm_ZIB - VmUb[0], min=0).pow(2),
                    dim=0
                )
            else:
                loss_Vi = torch.zeros(1).to(device)
            
            # Adaptive penalty weight update
            kcost = torch.tensor([params.kcost]).to(device)
            
            if params.flag_k == 2:
                # Adaptive: k_i = min(kcost * L_obj / L_i, k_max)
                kgenp = torch.min(kcost * loss_obj / (loss_Pgi + 1e-4), params.kgenp_max.to(device))
                kgenq = torch.min(kcost * loss_obj / (loss_Qgi + 1e-4), params.kgenq_max.to(device))
                kpd = torch.min(kcost * loss_obj / (loss_Pdi + 1e-4), params.kpd_max.to(device))
                kqd = torch.min(kcost * loss_obj / (loss_Qdi + 1e-4), params.kqd_max.to(device))
                kv = torch.min(kcost * loss_obj / (loss_Vi + 1e-4), params.kv_max.to(device)) if params.NZIB > 0 else torch.zeros(1).to(device)
                
                # Update params for logging
                params.kgenp = kgenp.detach().cpu()
                params.kgenq = kgenq.detach().cpu()
                params.kpd = kpd.detach().cpu()
                params.kqd = kqd.detach().cpu()
                params.kv = kv.detach().cpu()
            else:
                kgenp = params.kgenp.to(device)
                kgenq = params.kgenq.to(device)
                kpd = params.kpd.to(device)
                kqd = params.kqd.to(device)
                kv = params.kv.to(device)
            
            # Combined loss
            obj_weight = params.obj_weight_multiplier
            ls_cost = (kcost * loss_obj * obj_weight) / Nsam
            ls_Pg = (kgenp * loss_Pgi).sum() / Nsam
            ls_Qg = (kgenq * loss_Qgi).sum() / Nsam
            ls_Pd = (kpd * loss_Pdi).sum() / Nsam
            ls_Qd = (kqd * loss_Qdi).sum() / Nsam
            ls_V = (kv * loss_Vi).sum() / Nsam if params.NZIB > 0 else torch.tensor(0.0).to(device)
            
            # [IMPROVEMENT] Support only_obj parameter: only compute objective loss if enabled
            if only_obj:
                loss_out = ls_cost
            else:
                loss_out = ls_cost + ls_Pg + ls_Qg + ls_Pd + ls_Qd + ls_V
            
            # Store detailed loss components for validation analysis
            params._loss_obj = loss_obj.detach().item()
            params._loss_Pgi_sum = loss_Pgi.sum().detach().item()
            params._loss_Qgi_sum = loss_Qgi.sum().detach().item()
            params._loss_Pdi_sum = loss_Pdi.sum().detach().item()
            params._loss_Qdi_sum = loss_Qdi.sum().detach().item()
            params._loss_Vi_sum = loss_Vi.sum().detach().item()
            params._ls_cost = ls_cost.detach().item()
            params._ls_Pg = ls_Pg.detach().item()
            params._ls_Qg = ls_Qg.detach().item()
            params._ls_Pd = ls_Pd.detach().item()
            params._ls_Qd = ls_Qd.detach().item()
            params._ls_V = ls_V.detach().item()
            
            # Save for backward
            ctx.save_for_backward(
                Pg, Qg,
                torch.as_tensor(Ve), torch.as_tensor(Vf),
                kgenp, kgenq, kpd, kqd,
                torch.as_tensor(xam_P),
                w_cost_eff, w_carbon_total
            )
            ctx.device = device
            ctx.kdelta = kdelta
            ctx.only_obj = only_obj  # Save only_obj flag for backward
            
            return loss_out
        
        @staticmethod
        def backward(ctx, grad_output):
            """
            Backward pass: compute gradient using analytical Jacobian.
            """
            Pg, Qg, Ve, Vf, kgenp, kgenq, kpd, kqd, xam_P, w_cost_eff, w_carbon_total = ctx.saved_tensors
            device = ctx.device
            kdelta = ctx.kdelta
            
            Nsam, Nbus = Ve.shape
            
            # Move tensors to device
            Ve_tensor = Ve.float().to(device)
            Vf_tensor = Vf.float().to(device)
            Vef_tensor = torch.cat((Ve_tensor, Vf_tensor), dim=1)
            xam_P_tensor = xam_P.float().to(device)
            
            MAXMIN_Pg_tensor = params.MAXMIN_Pg_tensor.to(device)
            MAXMIN_Qg_tensor = params.MAXMIN_Qg_tensor.to(device)
            gencost_tensor = params.gencost_tensor.to(device)
            VmLb = params.VmLb.to(device)
            VmUb = params.VmUb.to(device)
            kcost = torch.tensor([params.kcost]).to(device)
            obj_weight = torch.tensor([params.obj_weight_multiplier]).to(device)
            
            # [IMPROVEMENT] Support only_obj parameter: skip voltage violation gradient if only_obj=True
            only_obj = getattr(ctx, 'only_obj', False)
            
            # ZIB voltage violation gradient
            if params.NZIB > 0 and not only_obj:
                Vm_ZIB = torch.sqrt(
                    Ve_tensor[:, params.bus_ZIB_all]**2 + 
                    Vf_tensor[:, params.bus_ZIB_all]**2
                )
                kv = params.kv.to(device)
                
                mat_Vmax = torch.where(
                    Vm_ZIB - VmUb[0] > kdelta,
                    (Vm_ZIB - VmUb[0]) / Vm_ZIB,
                    torch.tensor([0.0]).to(device)
                )
                mat_Vmin = torch.where(
                    Vm_ZIB - VmLb[0] < -kdelta,
                    (Vm_ZIB - VmLb[0]) / Vm_ZIB,
                    torch.tensor([0.0]).to(device)
                )
                mat_V = (mat_Vmin + mat_Vmax) * kv
                mat_V2 = torch.cat((mat_V, mat_V), dim=1)
            else:
                mat_V2 = None
            
            # Power gradient computation
            mat_P = torch.zeros(Nsam, Nbus).to(device)
            mat_Q = torch.zeros(Nsam, Nbus).to(device)
            
            # Pg constraint gradient
            mat_Pgmin = torch.where(
                Pg[:, params.bus_Pg] - MAXMIN_Pg_tensor[:, 1] < -kdelta,
                2 * (Pg[:, params.bus_Pg] - MAXMIN_Pg_tensor[:, 1]),
                torch.tensor([0.0]).to(device)
            )
            mat_Pgmax = torch.where(
                Pg[:, params.bus_Pg] - MAXMIN_Pg_tensor[:, 0] > kdelta,
                2 * (Pg[:, params.bus_Pg] - MAXMIN_Pg_tensor[:, 0]),
                torch.tensor([0.0]).to(device)
            )
            
            # Pg cost gradient (original economic cost)
            mat_Pgneg = torch.where(
                Pg[:, params.bus_Pg] > 0,
                torch.tensor([1.0]).to(device),
                torch.tensor([-2.0]).to(device)
            )
            mat_Pgcost_raw = (
                (2 * gencost_tensor[:, 0]).repeat(Nsam, 1) * Pg[:, params.bus_Pg] +
                gencost_tensor[:, 1].repeat(Nsam, 1) * mat_Pgneg
            )
            
            # Multi-objective: add carbon emission gradient
            if params.use_multi_objective:
                gci_tensor = params.gci_tensor.to(device)
                carbon_mask = (Pg[:, params.bus_Pg] > 0).float()
                mat_carbon_grad = gci_tensor.repeat(Nsam, 1) * carbon_mask
                mat_Pgcost = (w_cost_eff.view(Nsam, 1) * mat_Pgcost_raw) + (w_carbon_total.view(Nsam, 1) * mat_carbon_grad)
            else:
                mat_Pgcost = mat_Pgcost_raw
            
            # [IMPROVEMENT] Support only_obj parameter: only compute objective gradient if enabled
            only_obj = getattr(ctx, 'only_obj', False)
            
            if only_obj:
                # Only compute objective gradient (cost + carbon), skip constraint violations
                mat_P[:, params.bus_Pg] = mat_Pgcost * kcost * obj_weight
                # Set constraint gradients to zero
                mat_Q[:, params.bus_Qg] = torch.zeros_like(mat_Q[:, params.bus_Qg])
                mat_P[:, params.bus_Pnet_nonPg] = torch.zeros_like(mat_P[:, params.bus_Pnet_nonPg])
                mat_Q[:, params.bus_Pnet_nonQg] = torch.zeros_like(mat_Q[:, params.bus_Pnet_nonQg])
            else:
                # Compute full gradients (objective + constraints)
                mat_P[:, params.bus_Pg] = (mat_Pgmin + mat_Pgmax) * kgenp.reshape(1, -1) + mat_Pgcost * kcost * obj_weight
                
                # Qg constraint gradient
                mat_Qgmin = torch.where(
                    Qg[:, params.bus_Qg] - MAXMIN_Qg_tensor[:, 1] < -kdelta,
                    2 * (Qg[:, params.bus_Qg] - MAXMIN_Qg_tensor[:, 1]),
                    torch.tensor([0.0]).to(device)
                )
                mat_Qgmax = torch.where(
                    Qg[:, params.bus_Qg] - MAXMIN_Qg_tensor[:, 0] > kdelta,
                    2 * (Qg[:, params.bus_Qg] - MAXMIN_Qg_tensor[:, 0]),
                    torch.tensor([0.0]).to(device)
                )
                mat_Q[:, params.bus_Qg] = (mat_Qgmin + mat_Qgmax) * kgenq.reshape(1, -1)
                
                # Load deviation gradient
                mat_P[:, params.bus_Pnet_nonPg] = torch.where(
                    torch.abs(Pg[:, params.bus_Pnet_nonPg]) > kdelta,
                    2 * Pg[:, params.bus_Pnet_nonPg],
                    torch.tensor([0.0]).to(device)
                ) * kpd.reshape(1, -1)
                
                mat_Q[:, params.bus_Pnet_nonQg] = torch.where(
                    torch.abs(Qg[:, params.bus_Pnet_nonQg]) > kdelta,
                    2 * Qg[:, params.bus_Pnet_nonQg],
                    torch.tensor([0.0]).to(device)
                ) * kqd.reshape(1, -1)
            
            # Prepare for Jacobian multiplication
            mat_P3 = torch.unsqueeze(mat_P[:, params.bus_Pnet_all], 2)
            mat_Q3 = torch.unsqueeze(mat_Q[:, params.bus_Pnet_all], 2)
            mat_PQ3 = torch.cat((mat_P3, mat_Q3), dim=1)
            
            # Jacobian matrix computation
            # Ensure batch tensors match current batch size
            if Nsam != params.batch_size:
                # Recreate batch tensors for this batch size
                Me_re = np.repeat(np.expand_dims(params.Me, 0), Nsam, axis=0)
                Mf_re = np.repeat(np.expand_dims(params.Mf, 0), Nsam, axis=0)
                MGB_re = np.repeat(np.expand_dims(params.MGB, 0), Nsam, axis=0)
                MBG_re = np.repeat(np.expand_dims(params.MBG, 0), Nsam, axis=0)
                
                Me_re_tensor = torch.from_numpy(Me_re).float().to(device)
                Mf_re_tensor = torch.from_numpy(Mf_re).float().to(device)
                MGB_re_tensor = torch.from_numpy(MGB_re).float().to(device)
                MBG_re_tensor = torch.from_numpy(MBG_re).float().to(device)
                
                if params.param_ZIM is not None:
                    param_ZIM_tensor = torch.from_numpy(params.param_ZIM).float()
                    param_ZIM_tensor_expd = torch.unsqueeze(param_ZIM_tensor, dim=0)
                    param_ZIM_tensor_re = param_ZIM_tensor_expd.repeat_interleave(Nsam, dim=0).float().to(device)
                else:
                    param_ZIM_tensor_re = None
            else:
                Me_re_tensor = params.Me_re_tensor.float().to(device)
                Mf_re_tensor = params.Mf_re_tensor.float().to(device)
                MGB_re_tensor = params.MGB_re_tensor.float().to(device)
                MBG_re_tensor = params.MBG_re_tensor.float().to(device)
                param_ZIM_tensor_re = params.param_ZIM_tensor_re.float().to(device) if params.param_ZIM_tensor_re is not None else None
            
            # Build Jacobian
            diage_expd = torch.unsqueeze(torch.cat((Ve_tensor, Ve_tensor), dim=1), dim=2)
            diagf_expd = torch.unsqueeze(torch.cat((Vf_tensor, Vf_tensor), dim=1), dim=2)
            
            diage_re = diage_expd.repeat_interleave(2 * Nbus, dim=2)
            diagf_re = diagf_expd.repeat_interleave(2 * Nbus, dim=2)
            
            Vef_expd = torch.unsqueeze(Vef_tensor, dim=1)
            Vef_re = Vef_expd.repeat_interleave(Nbus, dim=1)
            
            a_tensor = torch.sum(MGB_re_tensor * Vef_re, dim=2)
            b_tensor = torch.sum(MBG_re_tensor * Vef_re, dim=2)
            
            # Create diagonal matrices - PURE PYTORCH (no CPU-GPU transfer!)
            a_diag = matrix_diag_torch(a_tensor)
            b_diag = matrix_diag_torch(b_tensor)
            
            Mab_diag1 = torch.cat((a_diag, b_diag), dim=2)
            Mab_diag2 = torch.cat((-b_diag, a_diag), dim=2)
            Mab_diag = torch.cat((Mab_diag1, Mab_diag2), dim=1)
            
            # Full Jacobian: J = diag(Ve) @ Me + diag(Vf) @ Mf + Mab
            J_tensor = (diage_re * Me_re_tensor + diagf_re * Mf_re_tensor + Mab_diag).float()
            
            # Extract Jacobian for non-ZIB nodes
            Jx1 = J_tensor[:, params.idx_Pnet, :]
            Jx = Jx1[:, :, params.idx_Pnet].float()
            
            # Handle ZIB contribution to Jacobian
            if params.NZIB > 0 and param_ZIM_tensor_re is not None:
                Jy1 = J_tensor[:, params.idx_Pnet, :]
                Jy = Jy1[:, :, params.idx_ZIB].float()
                Jyx = torch.bmm(Jy, param_ZIM_tensor_re.float())
                Jcom = Jx + Jyx
            else:
                Jcom = Jx
            
            # Convert to polar coordinates
            Vax = xam_P_tensor[:, :params.NPred_Vm]
            Vmx = xam_P_tensor[:, params.NPred_Vm:]
            
            dPQdVe = Jcom[:, :, :params.NPred_Vm]
            dPQdVf = Jcom[:, :, params.NPred_Vm:]
            
            dPQdVe2 = torch.cat((dPQdVe, dPQdVe), dim=2)
            dPQdVf2 = torch.cat((dPQdVf, dPQdVf), dim=2)
            
            # dVe/dVa, dVe/dVm, dVf/dVa, dVf/dVm
            dVedVa = -Vmx * torch.sin(Vax)
            dVedVm = torch.cos(Vax)
            dVfdVa = Vmx * torch.cos(Vax)
            dVfdVm = torch.sin(Vax)
            
            dVedVa_expd = torch.unsqueeze(dVedVa, dim=1)
            dVedVm_expd = torch.unsqueeze(dVedVm, dim=1)
            dVfdVa_expd = torch.unsqueeze(dVfdVa, dim=1)
            dVfdVm_expd = torch.unsqueeze(dVfdVm, dim=1)
            
            dVedVa_rep = dVedVa_expd.repeat_interleave(params.NPred_Vm, dim=1)
            dVedVm_rep = dVedVm_expd.repeat_interleave(params.NPred_Vm, dim=1)
            dVfdVa_rep = dVfdVa_expd.repeat_interleave(params.NPred_Vm, dim=1)
            dVfdVm_rep = dVfdVm_expd.repeat_interleave(params.NPred_Vm, dim=1)
            
            dVedVam = torch.cat((dVedVa_rep, dVedVm_rep), dim=2)
            dVfdVam = torch.cat((dVfdVa_rep, dVfdVm_rep), dim=2)
            dVe2dVam = torch.cat((dVedVam, dVedVam), dim=1)
            dVf2dVam = torch.cat((dVfdVam, dVfdVam), dim=1)
            
            # Chain rule: dPQ/dVam = dPQ/dVe * dVe/dVam + dPQ/dVf * dVf/dVam
            Jcom_pole = dPQdVe2 * dVe2dVam + dPQdVf2 * dVf2dVam
            
            # Remove slack bus column
            J_slack = torch.cat((
                Jcom_pole[:, :, :params.idx_bus_Pnet_slack[0]],
                Jcom_pole[:, :, params.idx_bus_Pnet_slack[0] + 1:]
            ), dim=2)
            
            # ZIB voltage gradient
            if params.NZIB > 0 and mat_V2 is not None and param_ZIM_tensor_re is not None:
                Vefy = Vef_tensor[:, params.idx_ZIB]
                dLdVefy = Vefy * mat_V2
                dLdVefy_expd = torch.unsqueeze(dLdVefy, dim=2)
                dLdVefy_re = dLdVefy_expd.repeat_interleave(2 * params.NPred_Vm, dim=2)
                dLdVefx = dLdVefy_re * param_ZIM_tensor_re
                
                dLdVex2 = torch.cat((dLdVefx[:, :, :params.NPred_Vm], dLdVefx[:, :, :params.NPred_Vm]), dim=2)
                dLdVfx2 = torch.cat((dLdVefx[:, :, params.NPred_Vm:], dLdVefx[:, :, params.NPred_Vm:]), dim=2)
                dLdVefx2 = torch.cat((dLdVex2, dLdVfx2), dim=1)
                
                dVedVam_expd = torch.cat((dVedVa_expd, dVedVm_expd), dim=2)
                dVfdVam_expd = torch.cat((dVfdVa_expd, dVfdVm_expd), dim=2)
                dVedVamy_re = dVedVam_expd.repeat_interleave(2 * params.NZIB, dim=1)
                dVfdVamy_re = dVfdVam_expd.repeat_interleave(2 * params.NZIB, dim=1)
                dVefdVamy = torch.cat((dVedVamy_re, dVfdVamy_re), dim=1)
                
                dLdVamx = dLdVefx2 * dVefdVamy
                dLdVamx_slack = torch.cat((
                    dLdVamx[:, :, :params.idx_bus_Pnet_slack[0]],
                    dLdVamx[:, :, params.idx_bus_Pnet_slack[0] + 1:]
                ), dim=2)
            else:
                dLdVamx_slack = torch.zeros(Nsam, 1, params.NPred_Va + params.NPred_Vm).to(device)
            
            # Final gradient
            matJ = mat_PQ3 * J_slack
            grad_Vloss = torch.sum(matJ, dim=1) + torch.sum(dLdVamx_slack, dim=1)
            
            return grad_output.to(device) * grad_Vloss.to(device), None, None
    
    return Penalty_V


class DeepOPFNGTLoss(nn.Module):
    """
    Wrapper module for DeepOPF-NGT unsupervised loss.
    
    This class manages the Penalty_V autograd function and provides
    a convenient interface for training.
    """
    
    def __init__(self, sys_data, config):
        """
        Initialize the DeepOPF-NGT loss module.
        
        Args:
            sys_data: PowerSystemData object from data_loader
            config: Configuration object
        """
        super().__init__()
        
        # Compute system parameters
        self.params = compute_ngt_params(sys_data, config)
        
        # Create Penalty_V class
        self.Penalty_V = create_penalty_v_class(self.params)
        
        # Store reference to data
        self.config = config
        self.sys_data = sys_data
        
        # GPU caching flag
        self._gpu_cached = False
    
    def cache_to_gpu(self, device):
        """
        Pre-cache all parameters to GPU to avoid repeated .to(device) calls.
        Call this once before training starts.
        
        Args:
            device: Target device (e.g., 'cuda:0')
        """
        if self._gpu_cached:
            return
        
        params = self.params
        
        # Cache constant tensors
        params.MAXMIN_Pg_tensor = params.MAXMIN_Pg_tensor.to(device)
        params.MAXMIN_Qg_tensor = params.MAXMIN_Qg_tensor.to(device)
        params.gencost_tensor = params.gencost_tensor.to(device)
        params.VmLb = params.VmLb.to(device)
        params.VmUb = params.VmUb.to(device)
        params.kpd_max = params.kpd_max.to(device)
        params.kqd_max = params.kqd_max.to(device)
        params.kgenp_max = params.kgenp_max.to(device)
        params.kgenq_max = params.kgenq_max.to(device)
        params.kv_max = params.kv_max.to(device)
        
        # Cache Jacobian matrices
        params.Me_re_tensor = params.Me_re_tensor.to(device)
        params.Mf_re_tensor = params.Mf_re_tensor.to(device)
        params.MGB_re_tensor = params.MGB_re_tensor.to(device)
        params.MBG_re_tensor = params.MBG_re_tensor.to(device)
        if params.param_ZIM_tensor_re is not None:
            params.param_ZIM_tensor_re = params.param_ZIM_tensor_re.to(device)
        
        self._gpu_cached = True
        
    def forward(self, V_pred, PQd, preference=None, only_obj=False):
        """
        Compute unsupervised loss.
        
        Args:
            V_pred: Predicted voltages [batch, NPred_Va + NPred_Vm]
                   Va (without slack) followed by Vm (non-ZIB buses)
            PQd: Load data [batch, num_Pd + num_Qd]
            preference: Preference tensor for multi-objective optimization
            only_obj: If True, only compute objective loss (cost + carbon), 
                     skip constraint violation losses (Pg, Qg, Pd, Qd, V limits)
            
        Returns:
            loss: Scalar loss value
            loss_dict: Dictionary with loss components for logging
        """
        # Convert preference to tensor if provided
        if preference is None:
            preference_t = V_pred.new_empty((0,))
        else:
            if torch.is_tensor(preference):
                preference_t = preference.detach().to(device=V_pred.device, dtype=V_pred.dtype)
            else:
                preference_t = torch.tensor(preference, device=V_pred.device, dtype=V_pred.dtype)

        # Store only_obj flag in params for use in forward
        self.params._only_obj = only_obj
        loss = self.Penalty_V.apply(V_pred, PQd, preference_t)

        # Build loss dict for logging
        loss_dict = {
            'total': loss.item() if isinstance(loss, torch.Tensor) else loss,
            'kgenp_mean': self.params.kgenp.mean().item() if hasattr(self.params.kgenp, 'mean') else 0,
            'kgenq_mean': self.params.kgenq.mean().item() if hasattr(self.params.kgenq, 'mean') else 0,
            'kpd_mean': self.params.kpd.mean().item() if hasattr(self.params.kpd, 'mean') else 0,
            'kqd_mean': self.params.kqd.mean().item() if hasattr(self.params.kqd, 'mean') else 0,
            'kv_mean': self.params.kv.mean().item() if hasattr(self.params.kv, 'mean') else 0,
            'loss_cost': getattr(self.params, '_loss_cost', 0.0),
            'loss_carbon': getattr(self.params, '_loss_carbon', 0.0),
            # Validation analysis metrics
            'loss_obj': getattr(self.params, '_loss_obj', 0.0),
            'cost_per_mean': getattr(self.params, '_cost_per_mean', 0.0),
            'carbon_per_mean': getattr(self.params, '_carbon_per_mean', 0.0),
            'loss_Pgi_sum': getattr(self.params, '_loss_Pgi_sum', 0.0),
            'loss_Qgi_sum': getattr(self.params, '_loss_Qgi_sum', 0.0),
            'loss_Pdi_sum': getattr(self.params, '_loss_Pdi_sum', 0.0),
            'loss_Qdi_sum': getattr(self.params, '_loss_Qdi_sum', 0.0),
            'loss_Vi_sum': getattr(self.params, '_loss_Vi_sum', 0.0),
            'ls_cost': getattr(self.params, '_ls_cost', 0.0),
            'ls_Pg': getattr(self.params, '_ls_Pg', 0.0),
            'ls_Qg': getattr(self.params, '_ls_Qg', 0.0),
            'ls_Pd': getattr(self.params, '_ls_Pd', 0.0),
            'ls_Qd': getattr(self.params, '_ls_Qd', 0.0),
            'ls_V': getattr(self.params, '_ls_V', 0.0),
        }
        
        return loss, loss_dict
    
    def get_output_dims(self):
        """Get the expected output dimensions for the models."""
        return {
            'Va': self.params.NPred_Va,  # Non-ZIB buses excluding slack
            'Vm': self.params.NPred_Vm,  # Non-ZIB buses
            'total': self.params.NPred_Va + self.params.NPred_Vm
        }
    
    def get_bus_indices(self):
        """Get bus indices for data preprocessing."""
        return {
            'bus_Pnet_all': self.params.bus_Pnet_all,
            'bus_Pnet_noslack_all': self.params.bus_Pnet_noslack_all,
            'bus_ZIB_all': self.params.bus_ZIB_all,
            'bus_Pd': self.params.bus_Pd,
            'bus_Qd': self.params.bus_Qd,
        } 

