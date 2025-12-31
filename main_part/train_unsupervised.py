#!/usr/bin/env python
# coding: utf-8
"""
Unsupervised Training (DeepOPF-NGT) for DeepOPF-V

This module implements DeepOPF-NGT unsupervised training using a single NetV model
to predict voltage solutions for optimal power flow problems.

Author: Peng Yue
Date: December 15th, 2025

Usage:
    python train_unsupervised.py
    NGT_LAMBDA_COST=0.9 python train_unsupervised.py
"""

import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np 
import time
import os
import sys 
import math

# 添加项目根目录和 main_part 目录到 sys.path，确保模块无论从哪里导入都能正常工作
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from config import BaseConfig, _SCRIPT_DIR
from models import NetV
from data_loader import load_all_data, load_ngt_training_data, create_ngt_training_loader
from utils import (TensorBoardLogger, plot_unsupervised_training_curves,
                   get_genload, get_vioPQg, get_viobran2)
from deepopf_ngt_loss import DeepOPFNGTLoss 
from unified_eval import (build_ctx_from_ngt, NGTPredictor, evaluate_unified, 
                          _ensure_1d_int, _as_numpy)


# ==================== Unsupervised Training Configuration ====================

class UnsupervisedConfig(BaseConfig):
    """Configuration for DeepOPF-NGT unsupervised training."""
    
    def __init__(self):
        super().__init__()
        
        # ==================== NGT Cost & Objective ====================
        self.ngt_kcost = 0.0002  # Cost coefficient
        self.ngt_obj_weight_multiplier = float(os.environ.get('NGT_OBJ_WEIGHT_MULT', '10.0'))
        
        # Adaptive weight flag: 1 = fixed, 2 = adaptive
        self.ngt_flag_k = 2
        
        # Maximum penalty weights
        self.ngt_kpd_max = 100.0
        self.ngt_kqd_max = 100.0
        self.ngt_kgenp_max = 2000.0
        self.ngt_kgenq_max = 2000.0
        self.ngt_kv_max = 500.0
        
        # Initial penalty weights
        self.ngt_kpd_init = 100.0
        self.ngt_kqd_init = 100.0
        self.ngt_kgenp_init = 2000.0
        self.ngt_kgenq_init = 2000.0
        self.ngt_kv_init = 100.0
        
        # Post-processing coefficient
        self.ngt_k_dV = 0.1
        
        # ==================== NGT Dataset ====================
        self.ngt_Ntrain = 600
        self.ngt_Ntest = 2500
        self.ngt_Nhis = 3
        self.ngt_Nsample = 50000
        
        # ==================== NGT Training ====================
        self.ngt_Epoch = int(os.environ.get('NGT_EPOCH', '4500'))
        self.ngt_batch_size = int(os.environ.get('NGT_BATCH_SIZE', '50'))
        self.ngt_Lr = float(os.environ.get('NGT_LR', '1e-4'))
        self.ngt_s_epoch = int(os.environ.get('NGT_S_EPOCH', '3000'))
        self.ngt_p_epoch = int(os.environ.get('NGT_P_EPOCH', '10'))
        
        # Network architecture
        self.ngt_khidden = np.array([64, 224], dtype=int)
        self.ngt_hidden_units = 1
        
        # ==================== Voltage Bounds ====================
        if self.Nbus == 300:
            self.ngt_VmLb, self.ngt_VmUb = 0.94, 1.06
            self.ngt_VaLb = -math.pi * 21 / 180
            self.ngt_VaUb = math.pi * 40 / 180
        elif self.Nbus == 118:
            self.ngt_VmLb, self.ngt_VmUb = 1.02, 1.06
            self.ngt_VaLb = -math.pi * 20 / 180
            self.ngt_VaUb = math.pi * 16 / 180
        else:
            self.ngt_VmLb, self.ngt_VmUb = 0.98, 1.06
            self.ngt_VaLb = -math.pi * 17 / 180
            self.ngt_VaUb = -math.pi * 4 / 180
        
        self.ngt_random_seed = 12343
        
        # ==================== Multi-Objective ====================
        self.ngt_use_multi_objective = os.environ.get('NGT_MULTI_OBJ', 'True').lower() == 'true'
        self.ngt_lambda_cost = float(os.environ.get('NGT_LAMBDA_COST', '0.1'))
        self.ngt_lambda_carbon = 1.0 - self.ngt_lambda_cost
        self.ngt_carbon_scale = float(os.environ.get('NGT_CARBON_SCALE', '10.0'))
        
        # Preference conditioning
        self.ngt_use_preference_conditioning = os.environ.get('NGT_PREF_CONDITIONING', 'False').lower() == 'true'
        self.ngt_mo_objective_mode = "soft_tchebycheff"
        self.ngt_mo_use_running_scale = True
        self.ngt_mo_ema_beta = 0.99
        self.ngt_mo_eps = 1e-8
        self.ngt_mo_tau = 0.1
        
        # ==================== Pretrain VAE Paths ====================
        self.pretrain_model_path_vm = os.path.join(self.model_save_dir,
            f'modelvm{self.Nbus}r{self.sys_R}N{self.model_version}Lm8642_vae_E1000F1.pth')
        self.pretrain_model_path_va = os.path.join(self.model_save_dir,
            f'modelva{self.Nbus}r{self.sys_R}N{self.model_version}La8642_vae_E1000F1.pth')
    
    def print_config(self):
        """Print configuration summary."""
        super().print_config()
        print(f"\n[Unsupervised NGT Training Config]")
        print(f"  Epochs: {self.ngt_Epoch}")
        print(f"  Learning rate: {self.ngt_Lr}")
        print(f"  Batch size: {self.ngt_batch_size}")
        print(f"  Multi-objective: {self.ngt_use_multi_objective}")
        if self.ngt_use_multi_objective:
            print(f"  Lambda cost/carbon: {self.ngt_lambda_cost}/{self.ngt_lambda_carbon}")


def get_unsupervised_config():
    """Get unsupervised training configuration."""
    return UnsupervisedConfig()


# For backward compatibility
def get_config():
    """Backward compatible alias for get_unsupervised_config."""
    return get_unsupervised_config()


def _init_loss_history():
    return {
        'total': [],
        'kgenp_mean': [],
        'kgenq_mean': [],
        'kpd_mean': [],
        'kqd_mean': [],
        'kv_mean': [],
        'cost': [],
        'carbon': [],
    }

def _compute_constraint_satisfaction(
    V_pred,  # [batch, NPred_Va + NPred_Vm] in NGT format
    x_train,  # [batch, input_dim] load data
    ngt_data,
    sys_data,
    config,
    device,
):
    """
    Compute constraint satisfaction rates from predicted voltages.
    
    Returns:
        dict with satisfaction rates: Pg, Qg, Vm, branch_ang, branch_pf
    """
    batch_size = V_pred.shape[0]
    
    # Reconstruct full voltage from NGT format
    V_pred_np = V_pred.detach().cpu().numpy()
    xam_P = np.insert(V_pred_np, ngt_data['idx_bus_Pnet_slack'][0], 0, axis=1)
    Va_len_with_slack = ngt_data['NPred_Va'] + 1
    Va_nonZIB = xam_P[:, :Va_len_with_slack]
    Vm_nonZIB = xam_P[:, Va_len_with_slack:Va_len_with_slack + ngt_data['NPred_Vm']]
    Vx = Vm_nonZIB * np.exp(1j * Va_nonZIB)
    
    # Recover ZIB voltages if needed
    if ngt_data['NZIB'] > 0 and ngt_data.get('param_ZIMV') is not None:
        Vy = np.dot(ngt_data['param_ZIMV'], Vx.T).T
    else:
        Vy = None
    
    Ve = np.zeros((batch_size, config.Nbus))
    Vf = np.zeros((batch_size, config.Nbus))
    Ve[:, ngt_data['bus_Pnet_all']] = Vx.real
    Vf[:, ngt_data['bus_Pnet_all']] = Vx.imag
    if Vy is not None:
        Ve[:, ngt_data['bus_ZIB_all']] = Vy.real
        Vf[:, ngt_data['bus_ZIB_all']] = Vy.imag
    
    Pred_Vm_full = np.sqrt(Ve**2 + Vf**2)
    Pred_Va_full = np.arctan2(Vf, Ve)
    Pred_V = Pred_Vm_full * np.exp(1j * Pred_Va_full)
    
    # Get load data
    x_train_np = x_train.detach().cpu().numpy()
    num_Pd = len(ngt_data['bus_Pd'])
    Pd_sample = np.zeros((batch_size, config.Nbus))
    Qd_sample = np.zeros((batch_size, config.Nbus))
    Pd_sample[:, ngt_data['bus_Pd']] = x_train_np[:, :num_Pd]
    Qd_sample[:, ngt_data['bus_Qd']] = x_train_np[:, num_Pd:]
    
    # Compute power flow
    baseMVA = float(sys_data.baseMVA.item() if hasattr(sys_data.baseMVA, 'item') else sys_data.baseMVA)
    Pred_Pg, Pred_Qg, _, _ = get_genload(
        Pred_V, Pd_sample, Qd_sample,
        sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
    )
    
    # Calculate constraint satisfaction
    MAXMIN_Pg = ngt_data['MAXMIN_Pg']
    MAXMIN_Qg = ngt_data['MAXMIN_Qg']
    _, _, _, _, _, vio_PQg, _, _, _, _ = get_vioPQg(
        Pred_Pg, sys_data.bus_Pg, MAXMIN_Pg,
        Pred_Qg, sys_data.bus_Qg, MAXMIN_Qg,
        config.DELTA
    )
    if torch.is_tensor(vio_PQg):
        vio_PQg = vio_PQg.numpy()
    
    Pg_satisfy = float(np.mean(vio_PQg[:, 0]))
    Qg_satisfy = float(np.mean(vio_PQg[:, 1]))
    
    # Voltage satisfaction
    VmLb = config.ngt_VmLb
    VmUb = config.ngt_VmUb
    Vm_vio_upper = np.mean(Pred_Vm_full > VmUb) * 100
    Vm_vio_lower = np.mean(Pred_Vm_full < VmLb) * 100
    Vm_satisfy = 100 - Vm_vio_upper - Vm_vio_lower
    
    # Branch constraints (optional, may be slower)
    branch_ang_satisfy = 100.0
    branch_pf_satisfy = 100.0
    try:
        # Compute BRANFT from branch data if available
        if hasattr(sys_data, 'branch') and sys_data.branch is not None:
            branch_np = sys_data.branch if isinstance(sys_data.branch, np.ndarray) else sys_data.branch.numpy()
            BRANFT_data = branch_np[:, 0:2] - 1  # Convert to 0-indexed
            
            if hasattr(sys_data, 'Yf') and hasattr(sys_data, 'Yt'):
                vio_branang, vio_branpf, _, _, _, _, _, _ = get_viobran2(
                    Pred_V, Pred_Va_full, branch_np, sys_data.Yf, sys_data.Yt,
                    BRANFT_data, baseMVA, config.DELTA
                )
                if torch.is_tensor(vio_branang):
                    vio_branang = vio_branang.numpy()
                if torch.is_tensor(vio_branpf):
                    vio_branpf = vio_branpf.numpy()
                branch_ang_satisfy = float(np.mean(vio_branang))
                branch_pf_satisfy = float(np.mean(vio_branpf))
    except Exception as e:
        # If branch constraint calculation fails, skip it
        pass
    
    return {
        'Pg_satisfy': Pg_satisfy,
        'Qg_satisfy': Qg_satisfy,
        'Vm_satisfy': Vm_satisfy,
        'branch_ang_satisfy': branch_ang_satisfy,
        'branch_pf_satisfy': branch_pf_satisfy,
    }

def train_unsupervised_ngt(config, lambda_cost, lambda_carbon, sys_data=None, device=None, tb_logger=None):
    """
    DeepOPF-NGT Unsupervised Training - EXACTLY matching reference implementation.
    
    This function uses a SINGLE NetV model to predict [Va, Vm] for non-ZIB nodes,
    exactly as in main_DeepOPFNGT_M3.ipynb. It does NOT use the dual-model architecture
    (NetVm + NetVa) that supervised training uses.
    
    Key differences from supervised training:
    1. Single NetV model (not separate Vm/Va models)
    2. Output includes sigmoid with scale/bias for physical constraints
    3. Uses Kron Reduction: only predicts non-ZIB nodes
    4. Unsupervised loss with analytical Jacobian backward
    5. Adaptive penalty weight scheduling
    
    Loss function:
    L = k_cost * L_cost + k_genp * L_Pg + k_genq * L_Qg + k_pd * L_Pd + k_qd * L_Qd + k_v * L_V
    
    Args:
        config: Configuration object with ngt_* parameters
        sys_data: PowerSystemData object (optional, will load if None)
        device: Device (optional, uses config.device if None)
        tb_logger: TensorBoardLogger instance for logging (optional)
        
    Returns:
        model: Trained NetV model
        loss_history: Dictionary of training losses
        time_train: Training time in seconds
        ngt_data: NGT training data dictionary
        sys_data: Updated system data
    """ 
    
    # Device setup
    if device is None:
        device = config.device
    
    # ============================================================
    # Step 1: Load NGT-specific training data
    # ============================================================
    ngt_data, sys_data = load_ngt_training_data(config, sys_data)
    
    input_dim = ngt_data['input_dim']
    output_dim = ngt_data['output_dim']
    Vscale = ngt_data['Vscale'].to(device)
    Vbias = ngt_data['Vbias'].to(device)
    
    # ============================================================
    # Step 1.5: Check if preference conditioning is enabled
    # ============================================================
    use_multi_objective = getattr(config, 'ngt_use_multi_objective', False)
    use_pref_conditioning = _use_multi_objective(config) and _use_preference_conditioning(config)
    
    # If using preference conditioning, input_channels should include preference dimension (2)
    model_input_dim = input_dim + 2 if use_pref_conditioning else input_dim
    
    # Create preference base tensor if needed
    pref_base = _make_pref_base(device, lambda_cost, lambda_carbon) if use_pref_conditioning else None
    
    # ============================================================
    # Step 2: Create NetV model (single unified model)
    # ============================================================
    model = NetV(
        input_channels=model_input_dim,
        output_channels=output_dim,
        hidden_units=config.ngt_hidden_units,
        khidden=config.ngt_khidden,
        Vscale=Vscale,
        Vbias=Vbias
    )
    model.to(device)
    
    # ============================================================
    # Step 3: Create optimizer (Adam with NGT learning rate)
    # ============================================================
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ngt_Lr)
    
    # ============================================================
    # Step 4: Create loss function (Penalty_V with custom backward)
    # ============================================================
    config.ngt_lambda_cost = lambda_cost
    config.ngt_lambda_carbon = lambda_carbon
    loss_fn = DeepOPFNGTLoss(sys_data, config)
    
    # Pre-cache parameters to GPU for faster training
    loss_fn.cache_to_gpu(device)
    
    # ============================================================
    # Step 5: Create training DataLoader
    # ============================================================
    training_loader = create_ngt_training_loader(ngt_data, config) 
    
    # ============================================================
    # Step 6: Training loop (exactly matching reference)
    # ============================================================
    loss_history = {
        'total': [],
        'kgenp_mean': [],
        'kgenq_mean': [],
        'kpd_mean': [],
        'kqd_mean': [],
        'kv_mean': [],
        # Multi-objective components (always present for compatibility)
        'cost': [],
        'carbon': [],
    }
    
    n_epochs = config.ngt_Epoch
    batch_size = config.ngt_batch_size
    p_epoch = config.ngt_p_epoch
    s_epoch = config.ngt_s_epoch
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        running_loss = 0.0
        running_kgenp = 0.0
        running_kgenq = 0.0
        running_kpd = 0.0
        running_kqd = 0.0
        running_kv = 0.0
        running_cost = 0.0
        running_carbon = 0.0
        # Initialize validation analysis accumulators
        running_loss_obj = 0.0
        running_cost_per_mean = 0.0
        running_carbon_per_mean = 0.0
        running_loss_Pgi_sum = 0.0
        running_loss_Qgi_sum = 0.0
        running_loss_Pdi_sum = 0.0
        running_loss_Qdi_sum = 0.0
        running_loss_Vi_sum = 0.0
        running_ls_cost = 0.0
        running_ls_Pg = 0.0
        running_ls_Qg = 0.0
        running_ls_Pd = 0.0
        running_ls_Qd = 0.0
        running_ls_V = 0.0
        n_batches = 0
        
        model.train()
        
        for step, (train_x, train_y) in enumerate(training_loader):
            train_x = train_x.to(device)             
            # train_x is the input: [Pd_nonzero, Qd_nonzero] / baseMVA
            # This IS the PQd data, just need to pass it correctly
            PQd_batch = train_x  # Input IS the load data in p.u.
            
            # Forward pass - concatenate preference if using preference conditioning
            if use_pref_conditioning:
                # Expand preference base to match batch size and concatenate
                B = train_x.shape[0]
                pref_batch = pref_base.expand(B, -1).to(device=train_x.device, dtype=train_x.dtype)
                model_input = torch.cat([train_x, pref_batch], dim=1)
            else:
                model_input = train_x
                pref_batch = None
            
            # Model outputs [Va_nonZIB_noslack, Vm_nonZIB] with sigmoid
            yvtrain_hat = model(model_input)
            
            # Compute loss (Penalty_V.apply)
            loss, loss_dict = loss_fn(yvtrain_hat, PQd_batch, preference=pref_batch)
            
            # Backward pass (uses custom analytical Jacobian)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            running_loss += loss.item()
            running_kgenp += loss_dict['kgenp_mean']
            running_kgenq += loss_dict['kgenq_mean']
            running_kpd += loss_dict['kpd_mean']
            running_kqd += loss_dict['kqd_mean']
            running_kv += loss_dict['kv_mean']
            running_cost += loss_dict.get('loss_cost', 0.0)
            running_carbon += loss_dict.get('loss_carbon', 0.0)
            # Validation analysis metrics
            # Objective function metrics
            running_loss_obj += loss_dict.get('loss_obj', 0.0)  # Total objective value (weighted sum of cost and carbon)
            running_cost_per_mean += loss_dict.get('cost_per_mean', 0.0)  # Average generation cost per sample
            running_carbon_per_mean += loss_dict.get('carbon_per_mean', 0.0)  # Average carbon emission per sample (scaled)
            # Constraint violation metrics (unweighted sums)
            running_loss_Pgi_sum += loss_dict.get('loss_Pgi_sum', 0.0)  # Sum of generator active power constraint violations
            running_loss_Qgi_sum += loss_dict.get('loss_Qgi_sum', 0.0)  # Sum of generator reactive power constraint violations
            running_loss_Pdi_sum += loss_dict.get('loss_Pdi_sum', 0.0)  # Sum of load active power constraint violations
            running_loss_Qdi_sum += loss_dict.get('loss_Qdi_sum', 0.0)  # Sum of load reactive power constraint violations
            running_loss_Vi_sum += loss_dict.get('loss_Vi_sum', 0.0)  # Sum of zero-injection bus (ZIB) voltage constraint violations
            # Weighted loss components (used in total loss calculation)
            running_ls_cost += loss_dict.get('ls_cost', 0.0)  # Weighted cost objective loss term
            running_ls_Pg += loss_dict.get('ls_Pg', 0.0)  # Weighted generator active power constraint loss term
            running_ls_Qg += loss_dict.get('ls_Qg', 0.0)  # Weighted generator reactive power constraint loss term
            running_ls_Pd += loss_dict.get('ls_Pd', 0.0)  # Weighted load active power constraint loss term
            running_ls_Qd += loss_dict.get('ls_Qd', 0.0)  # Weighted load reactive power constraint loss term
            running_ls_V += loss_dict.get('ls_V', 0.0)  # Weighted voltage constraint loss term
            n_batches += 1
        
        # Average losses for this epoch
        if n_batches > 0:
            avg_loss = running_loss / n_batches
            avg_kgenp = running_kgenp / n_batches
            avg_kgenq = running_kgenq / n_batches
            avg_kpd = running_kpd / n_batches
            avg_kqd = running_kqd / n_batches
            avg_kv = running_kv / n_batches
            avg_cost = running_cost / n_batches
            avg_carbon = running_carbon / n_batches
            # Validation analysis averages
            avg_loss_obj = running_loss_obj / n_batches
            avg_cost_per_mean = running_cost_per_mean / n_batches
            avg_carbon_per_mean = running_carbon_per_mean / n_batches
            avg_loss_Pgi_sum = running_loss_Pgi_sum / n_batches
            avg_loss_Qgi_sum = running_loss_Qgi_sum / n_batches
            avg_loss_Pdi_sum = running_loss_Pdi_sum / n_batches
            avg_loss_Qdi_sum = running_loss_Qdi_sum / n_batches
            avg_loss_Vi_sum = running_loss_Vi_sum / n_batches
            avg_ls_cost = running_ls_cost / n_batches
            avg_ls_Pg = running_ls_Pg / n_batches
            avg_ls_Qg = running_ls_Qg / n_batches
            avg_ls_Pd = running_ls_Pd / n_batches
            avg_ls_Qd = running_ls_Qd / n_batches
            avg_ls_V = running_ls_V / n_batches
            
            loss_history['total'].append(avg_loss)
            loss_history['kgenp_mean'].append(avg_kgenp)
            loss_history['kgenq_mean'].append(avg_kgenq)
            loss_history['kpd_mean'].append(avg_kpd)
            loss_history['kqd_mean'].append(avg_kqd)
            loss_history['kv_mean'].append(avg_kv)
            loss_history['cost'].append(avg_cost)
            loss_history['carbon'].append(avg_carbon)
        
        # Print progress (matching reference format)
        # Print first epoch immediately, then every p_epoch epochs
        should_print = (epoch == 0) or ((epoch + 1) % p_epoch == 0)
        if should_print:
            # Estimate time remaining using wall-clock time
            elapsed = time.time() - start_time
            if epoch > 0:
                time_per_epoch = elapsed / (epoch + 1)
                remaining = time_per_epoch * (n_epochs - epoch - 1)
                time_info = f" | Elapsed: {elapsed/60:.1f}min, ETA: {remaining/60:.1f}min"
            else:
                time_info = ""
            
            # Get sample predictions for range info
            with torch.no_grad():
                sample_x = ngt_data['x_train'][:batch_size].to(device)
                if use_pref_conditioning:
                    sample_pref = pref_base.expand(sample_x.shape[0], -1).to(device=sample_x.device, dtype=sample_x.dtype)
                    sample_input = torch.cat([sample_x, sample_pref], dim=1)
                else:
                    sample_input = sample_x
                sample_pred = model(sample_input)
                Va_pred = sample_pred[:, :ngt_data['NPred_Va']]
                Vm_pred = sample_pred[:, ngt_data['NPred_Va']:]
            
            print(f"epoch {epoch+1}/{n_epochs} loss={avg_loss:.4f} "
                  f"Va[{Va_pred.min().item():.4f},{Va_pred.max().item():.4f}] "
                  f"Vm[{Vm_pred.min().item():.4f},{Vm_pred.max().item():.4f}]{time_info}")
            print(f"  kcost={config.ngt_kcost} kgenp={avg_kgenp:.2f} kgenq={avg_kgenq:.2f} "
                  f"kpd={avg_kpd:.2f} kqd={avg_kqd:.2f} kv={avg_kv:.2f}")
            # Print multi-objective info if enabled
            if use_multi_objective:
                lambda_cost = getattr(config, 'ngt_lambda_cost', 0.9)
                lambda_carbon = getattr(config, 'ngt_lambda_carbon', 0.1)
                print(f"  [Multi-Obj] cost={avg_cost:.2f} carbon={avg_carbon:.4f} "
                      f"λ_cost={lambda_cost} λ_carbon={lambda_carbon}")
        
        # TensorBoard logging
        if tb_logger:
            tb_logger.log_scalar('loss/total', avg_loss, epoch)
            tb_logger.log_scalar('loss/cost', avg_cost, epoch)
            tb_logger.log_scalar('loss/carbon', avg_carbon, epoch)
            tb_logger.log_scalar('weights/kgenp', avg_kgenp, epoch)
            tb_logger.log_scalar('weights/kgenq', avg_kgenq, epoch)
            tb_logger.log_scalar('weights/kpd', avg_kpd, epoch)
            tb_logger.log_scalar('weights/kqd', avg_kqd, epoch)
            tb_logger.log_scalar('weights/kv', avg_kv, epoch)
            # Validation analysis metrics
            tb_logger.log_scalar('validation/loss_obj', avg_loss_obj, epoch)
            tb_logger.log_scalar('validation/cost_per_mean', avg_cost_per_mean, epoch)
            tb_logger.log_scalar('validation/carbon_per_mean', avg_carbon_per_mean, epoch)
            tb_logger.log_scalar('validation/loss_Pgi_sum', avg_loss_Pgi_sum, epoch)
            tb_logger.log_scalar('validation/loss_Qgi_sum', avg_loss_Qgi_sum, epoch)
            tb_logger.log_scalar('validation/loss_Pdi_sum', avg_loss_Pdi_sum, epoch)
            tb_logger.log_scalar('validation/loss_Qdi_sum', avg_loss_Qdi_sum, epoch)
            tb_logger.log_scalar('validation/loss_Vi_sum', avg_loss_Vi_sum, epoch)
            tb_logger.log_scalar('validation/ls_cost', avg_ls_cost, epoch)
            tb_logger.log_scalar('validation/ls_Pg', avg_ls_Pg, epoch)
            tb_logger.log_scalar('validation/ls_Qg', avg_ls_Qg, epoch)
            tb_logger.log_scalar('validation/ls_Pd', avg_ls_Pd, epoch)
            tb_logger.log_scalar('validation/ls_Qd', avg_ls_Qd, epoch)
            tb_logger.log_scalar('validation/ls_V', avg_ls_V, epoch)
            
            # Log constraint satisfaction rates (true constraint satisfaction metrics)
            try:
                with torch.no_grad():
                    sample_x = ngt_data['x_train'][:batch_size].to(device)
                    if use_pref_conditioning:
                        sample_pref = pref_base.expand(sample_x.shape[0], -1).to(device=sample_x.device, dtype=sample_x.dtype)
                        sample_input = torch.cat([sample_x, sample_pref], dim=1)
                    else:
                        sample_input = sample_x
                    sample_pred = model(sample_input)
                    constraint_stats = _compute_constraint_satisfaction(
                        sample_pred, sample_x, ngt_data, sys_data, config, device
                    )
                    tb_logger.log_scalar('constraint_satisfaction/Pg', constraint_stats['Pg_satisfy'], epoch)
                    tb_logger.log_scalar('constraint_satisfaction/Qg', constraint_stats['Qg_satisfy'], epoch)
                    tb_logger.log_scalar('constraint_satisfaction/Vm', constraint_stats['Vm_satisfy'], epoch)
                    if constraint_stats['branch_ang_satisfy'] < 100.0:
                        tb_logger.log_scalar('constraint_satisfaction/branch_ang', constraint_stats['branch_ang_satisfy'], epoch)
                        tb_logger.log_scalar('constraint_satisfaction/branch_pf', constraint_stats['branch_pf_satisfy'], epoch)
            except Exception as e:
                # If constraint calculation fails, skip it (don't break training)
                pass
        
        # Save models periodically (include lambda_cost in filename for multi-preference training)
        if (epoch + 1) % 100 == 0 and (epoch + 1) >= s_epoch:
            lc_str = f"_lc{config.ngt_lambda_cost:.1f}" if use_multi_objective else ""
            save_path = f'{config.model_save_dir}/NetV_ngt_{config.Nbus}bus{lc_str}_E{epoch+1}.pth'
            torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)
            print(f"  Model saved: {save_path}")
    
    time_train = time.time() - start_time
    print(f"\nTraining completed in {time_train:.2f} seconds ({time_train/60:.2f} minutes)")
    
    # Save final model (include lambda_cost in filename for multi-preference training)
    lc_str = f"_lc{config.ngt_lambda_cost:.1f}" if use_multi_objective else "_single"
    final_path = f'{config.model_save_dir}/NetV_ngt_{config.Nbus}bus{lc_str}_E{n_epochs}_final.pth'
    torch.save(model.state_dict(), final_path, _use_new_zipfile_serialization=False)
    print(f"Final model saved: {final_path}")
    
    return model, loss_history, time_train, ngt_data, sys_data


# ============================================================================
# Helper Functions for Multi-Objective Training
# ============================================================================

def _use_multi_objective(config) -> bool:
    return bool(getattr(config, "ngt_use_multi_objective", False))


def _use_preference_conditioning(config) -> bool:
    """
    是否把偏好向量喂给模型（条件化学习）。
    默认 False，保证单目标训练不引入偏好信息。
    """
    return bool(getattr(config, "ngt_use_preference_conditioning", False))


def _make_pref_base(device, lambda_cost: float, lambda_carbon: float):
    """
    构造 [1,2] 的 preference base（不在 batch 内重复创建张量）。
    """
    return torch.tensor([[lambda_cost, lambda_carbon]], dtype=torch.float32, device=device)


def _expand_pref(pref_base, batch_size: int):
    """
    [1,2] -> [B,2]
    """
    return pref_base.expand(batch_size, -1)

# ===================== [MO-PREF] Loss / Eval helpers =====================

def _loss_supports_preference(loss_fn) -> bool:
    """Check if DeepOPFNGTLoss.forward accepts a 'preference' kwarg."""
    import inspect
    try:
        sig = inspect.signature(loss_fn.forward)
        return "preference" in sig.parameters
    except Exception:
        return False


def _call_ngt_loss(loss_fn, config, V_pred, x_in, pref_batch=None, only_obj=False):
    """
    Call DeepOPFNGTLoss with correct preference alignment.
    - If loss supports preference kwarg: pass pref_batch directly (batch or sample level).
    - Else: only allow pref_level='batch' and sync lambda to loss/config using pref_batch[0].
    
    Args:
        loss_fn: DeepOPFNGTLoss instance
        config: Configuration object
        V_pred: Predicted voltages
        x_in: Load data
        pref_batch: Preference tensor (optional)
        only_obj: If True, only compute objective loss, skip constraint violations
    """
    if pref_batch is None:
        return loss_fn(V_pred, x_in, only_obj=only_obj)

    if _loss_supports_preference(loss_fn):
        return loss_fn(V_pred, x_in, preference=pref_batch, only_obj=only_obj) 


def main(debug=False):
    """
    Main function with support for training
    """
    # Load configuration
    config = get_config()
     
    config.print_config()
    
    # Create output directories if they don't exist
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    
    # Load data
    sys_data, _, BRANFT = load_all_data(config)
     
    # ==================== Unsupervised Training (DeepOPF-NGT) ==================== 
    
    # Create TensorBoard logger if enabled
    tb_logger = None
    tb_enabled = os.environ.get('TB_ENABLED', 'False').lower() == 'true'
    
    # Get lambda_cost for lc_str (used in save path)
    lambda_cost = getattr(config, 'ngt_lambda_cost', 0.9)
    lc_str = f"{lambda_cost:.1f}".replace('.', '')
    lambda_carbon = 1 - lambda_cost

    if tb_enabled and not debug:
        log_comment = f"ngt_mlp_lc{lc_str}_{config.Nbus}bus"
        runs_dir = os.path.join(os.path.dirname(__file__), 'runs')
        os.makedirs(runs_dir, exist_ok=True)
        tb_logger = TensorBoardLogger(log_dir=runs_dir, comment=log_comment)
    
    # Train NGT model
    if not debug: 
        model_ngt, loss_history, time_train, ngt_data, sys_data = train_unsupervised_ngt(
            config, lambda_cost, lambda_carbon, sys_data, config.device,
            tb_logger=tb_logger
        )
    else:
        model_path = "saved_models/NetV_ngt_300bus_E4500_final.pth"
        ngt_data, sys_data = load_ngt_training_data(config, sys_data)
        model_ngt = NetV(
                    input_channels=ngt_data['input_dim'],
                    output_channels=ngt_data['output_dim'],
                    hidden_units=config.ngt_hidden_units,
                    khidden=config.ngt_khidden,
                    Vscale=ngt_data['Vscale'],
                    Vbias=ngt_data['Vbias']).to(config.device)
        state_dict = torch.load(model_path, map_location=config.device, weights_only=True)
        model_ngt.load_state_dict(state_dict)
        loss_history = None
        time_train = None
    
    # Close TensorBoard logger
    if tb_logger:
        tb_logger.close()
    
    # Convert loss history to separate lists for compatibility
    if loss_history is not None:
        lossvm = loss_history['total']
        lossva = loss_history.get('kgenp_mean', [])
        # Plot unsupervised training curves
        plot_unsupervised_training_curves(loss_history)
    else:
        lossvm = None
        lossva = None
    
    # ==================== Evaluate NGT Model ==================== 
    # Build context for unified evaluation
    ctx = build_ctx_from_ngt(config, sys_data, ngt_data, BRANFT, config.device)
    
    # Create NGT predictor
    # [MO-PREF] if NetV expects [x,pref], wrap it so predictor can still call model(x)
    use_pref = _use_multi_objective(config) and _use_preference_conditioning(config)
    if use_pref:
        # Wrap model if it expects preference concatenation
        pref_base = _make_pref_base(config.device, lambda_cost, lambda_carbon)
        class _ConcatPrefWrapper(nn.Module):
            def __init__(self, base_model, pref_base):
                super().__init__()
                self.base_model = base_model
                self.register_buffer("_pref_base", pref_base)
            def forward(self, x):
                B = x.shape[0]
                pref = self._pref_base.expand(B, -1).to(device=x.device, dtype=x.dtype)
                return self.base_model(torch.cat([x, pref], dim=1))
        model_ngt_eval = _ConcatPrefWrapper(model_ngt, pref_base)
    else:
        model_ngt_eval = model_ngt
    
    predictor = NGTPredictor(model_ngt_eval)
    
    eval_results_unified = evaluate_unified(
        ctx, predictor,
        apply_post_processing=True,
        verbose=True
    ) 
    eval_results = eval_results_unified
    
    # Combine training and evaluation results
    results = {
        'loss_history': loss_history,
        'time_train': time_train,
        'ngt_data': ngt_data,
        **eval_results  # Include all evaluation metrics
    }
    
    # Save evaluation results
    eval_save_path = f'{config.model_save_dir}/ngt_mlp_eval_{config.Nbus}bus{lc_str}.npz'
    eval_to_save = {k: v for k, v in eval_results.items() 
                    if isinstance(v, (int, float, np.ndarray))}
    np.savez(eval_save_path, **eval_to_save)
    print(f"\nEvaluation results saved to: {eval_save_path}") 


if __name__ == "__main__": 
    debug = bool(int(os.environ.get('DEBUG', '0')))
    main(debug=debug)