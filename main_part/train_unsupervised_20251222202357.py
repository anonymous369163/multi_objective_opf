#!/usr/bin/env python
# coding: utf-8
"""
Unsupervised Training (DeepOPF-NGT) for DeepOPF-V
Extended to support multiple model types: VAE, Flow, Diffusion, GAN, etc.

Author: Peng Yue
Date: December 15th, 2025

Usage:
    python train_unsupervised_20251222202357.py
    NGT_LAMBDA_COST=0.9 python train_unsupervised_20251222202357.py
"""

import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np 
import time
import os
import sys 
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BaseConfig, _SCRIPT_DIR
from models import NetV
from data_loader import load_all_data, load_ngt_training_data, create_ngt_training_loader
from utils import (TensorBoardLogger, initialize_flow_model_near_zero, plot_unsupervised_training_curves,
                   get_genload, get_vioPQg, get_viobran2, get_clamp, dPQbus_dV, dSlbus_dV, get_hisdV, get_dV)
from deepopf_ngt_loss import DeepOPFNGTLoss 
from unified_eval import (build_ctx_from_ngt, NGTPredictor, NGTFlowPredictor, evaluate_unified, 
                          post_process_like_evaluate_model, EvalContext, _ensure_1d_int, _as_numpy, _as_torch,
                          _build_finc, _kron_reconstruct_zib)


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
        
        # ==================== NGT Flow Model ====================
        self.ngt_use_flow_model = os.environ.get('NGT_USE_FLOW', 'True').lower() == 'true'
        self.ngt_flow_inf_steps = int(os.environ.get('NGT_FLOW_STEPS', '10'))
        self.ngt_use_projection = os.environ.get('NGT_USE_PROJ', 'False').lower() == 'true'
        self.ngt_flow_hidden_dim = int(os.environ.get('NGT_FLOW_HIDDEN_DIM', '144'))
        self.ngt_flow_num_layers = int(os.environ.get('NGT_FLOW_NUM_LAYERS', '2'))
        
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
        print(f"  Use Flow model: {self.ngt_use_flow_model}")


def get_unsupervised_config():
    """Get unsupervised training configuration."""
    return UnsupervisedConfig()


# For backward compatibility
def get_config():
    """Backward compatible alias for get_unsupervised_config."""
    return get_unsupervised_config()


# ============================================================================
# NGT Flow Model Helper Functions
# ============================================================================

def flow_forward_ngt_unified(
    flow_model,
    x,
    z_anchor,
    preference,
    num_steps: int = 10,
    training: bool = True,
    P_tan_t=None,                 # None => no projection; Tensor => projection in normalized V-space
    eps: float = 1e-8,
    max_inv: float = 1e3,
):
    """
    Unified flow integration for NGT unsupervised training.

    If P_tan_t is provided, project velocity in normalized V-space:
        V_norm = sigmoid(z)
        v_norm = v_z * dV_norm/dz
        v_norm_proj = P_tan_t(v_norm)
        v_z_proj = v_norm_proj * (dV_norm/dz)^{-1}

    Args:
        flow_model: PreferenceConditionedNetV model (must have predict_velocity, Vscale, Vbias)
        x: [batch, input_dim]
        z_anchor: [batch, dim] latent anchor
        preference: [batch, 2] or None
        num_steps: Euler steps
        training: if True, z requires grad
        P_tan_t: [dim, dim] projection matrix defined on normalized V-space, or None
        eps: numerical stability for inverse Jacobian
        max_inv: clamp for inverse Jacobian to avoid explosion

    Returns:
        V_pred: [batch, dim] in physical space: sigmoid(z)*Vscale + Vbias
    """
    batch_size = x.shape[0]
    device = x.device
    dt = 1.0 / num_steps

    # Start from anchor (latent space)
    z = z_anchor.detach().clone()
    if training:
        z = z.requires_grad_(True)

    # Ensure projection matrix device/dtype if provided
    if P_tan_t is not None:
        P_tan_t = P_tan_t.to(device=device, dtype=z.dtype)

    for step in range(num_steps):
        t = torch.full((batch_size, 1), step * dt, device=device, dtype=z.dtype)
        v_latent = flow_model.predict_velocity(x, z, t, preference)

        if P_tan_t is None:
            z = z + v_latent * dt
            continue

        # ---- Projection branch (normalized V-space) ----
        sig_z = torch.sigmoid(z)                      # V_norm
        J_z_to_Vnorm = sig_z * (1.0 - sig_z)          # dV_norm/dz, shape [B, D]
        J_inv = torch.clamp(1.0 / (J_z_to_Vnorm + eps), max=max_inv)

        v_norm = v_latent * J_z_to_Vnorm              # z-space -> normalized V-space
        v_norm_proj = v_norm @ P_tan_t.T              # project in normalized V-space
        v_proj = v_norm_proj * J_inv                  # back to z-space

        z = z + v_proj * dt

    # Physical space output (matches your original behavior)
    V_pred = torch.sigmoid(z) * flow_model.Vscale + flow_model.Vbias
    return V_pred


# --- Backward-compatible wrappers (optional, but recommended) ---
def flow_forward_ngt(flow_model, x, z_anchor, preference, num_steps=10, training=True):
    return flow_forward_ngt_unified(flow_model=flow_model, x=x, z_anchor=z_anchor, preference=preference, num_steps=num_steps,
        training=training, P_tan_t=None)
def flow_forward_ngt_projected(flow_model, x, z_anchor, P_tan_t, preference, num_steps=10, training=True):
    return flow_forward_ngt_unified(flow_model=flow_model, x=x, z_anchor=z_anchor, preference=preference, num_steps=num_steps,
        training=training, P_tan_t=P_tan_t)


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

def train_ngt_core(
    *,
    config,
    model,
    loss_fn,
    optimizer,
    training_loader,
    device,
    ngt_data,
    sys_data,  # Added for constraint satisfaction calculation
    # ---- strategy callbacks ----
    forward_and_loss_fn,     # (batch)->(loss, loss_dict, V_pred_for_stats)
    sample_pred_fn,          # ()-> V_pred_sample (for printing ranges)
    # ---- io/logging ----
    tb_logger=None,
    print_prefix="",
    save_prefix="NetV_ngt",
    save_tag="",             # e.g. "_lc0.8" or "_flow_lc08"
    extra_tb_logging_fn=None, # optional hook (flow velocity diagnostics, baseline lines, etc.)
    grad_clip_norm=None,
    scheduler=None  # Learning rate scheduler (optional)
):
    n_epochs = config.ngt_Epoch
    batch_size = config.ngt_batch_size
    p_epoch = config.ngt_p_epoch
    s_epoch = config.ngt_s_epoch

    loss_history = _init_loss_history()
    start_time = time.time()

    for epoch in range(n_epochs):
        model.train()

        running = dict(
            loss=0.0, kgenp=0.0, kgenq=0.0, kpd=0.0, kqd=0.0, kv=0.0, cost=0.0, carbon=0.0, n=0
        )
        
        # Track gradient norm for diagnostics (before optimizer.step() clears it)
        epoch_grad_norms = []

        for batch in training_loader:
            loss, loss_dict, _ = forward_and_loss_fn(batch)

            optimizer.zero_grad()
            loss.backward() 
            
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            running['loss'] += float(loss.item())
            running['kgenp'] += float(loss_dict['kgenp_mean'])
            running['kgenq'] += float(loss_dict['kgenq_mean'])
            running['kpd'] += float(loss_dict['kpd_mean'])
            running['kqd'] += float(loss_dict['kqd_mean'])
            running['kv'] += float(loss_dict['kv_mean'])
            running['cost'] += float(loss_dict.get('loss_cost', 0.0))
            running['carbon'] += float(loss_dict.get('loss_carbon', 0.0))
            # Validation analysis metrics
            running['loss_obj'] = running.get('loss_obj', 0.0) + float(loss_dict.get('loss_obj', 0.0))
            running['cost_per_mean'] = running.get('cost_per_mean', 0.0) + float(loss_dict.get('cost_per_mean', 0.0))
            running['carbon_per_mean'] = running.get('carbon_per_mean', 0.0) + float(loss_dict.get('carbon_per_mean', 0.0))
            running['loss_Pgi_sum'] = running.get('loss_Pgi_sum', 0.0) + float(loss_dict.get('loss_Pgi_sum', 0.0))
            running['loss_Qgi_sum'] = running.get('loss_Qgi_sum', 0.0) + float(loss_dict.get('loss_Qgi_sum', 0.0))
            running['loss_Pdi_sum'] = running.get('loss_Pdi_sum', 0.0) + float(loss_dict.get('loss_Pdi_sum', 0.0))
            running['loss_Qdi_sum'] = running.get('loss_Qdi_sum', 0.0) + float(loss_dict.get('loss_Qdi_sum', 0.0))
            running['loss_Vi_sum'] = running.get('loss_Vi_sum', 0.0) + float(loss_dict.get('loss_Vi_sum', 0.0))
            running['ls_cost'] = running.get('ls_cost', 0.0) + float(loss_dict.get('ls_cost', 0.0))
            running['ls_Pg'] = running.get('ls_Pg', 0.0) + float(loss_dict.get('ls_Pg', 0.0))
            running['ls_Qg'] = running.get('ls_Qg', 0.0) + float(loss_dict.get('ls_Qg', 0.0))
            running['ls_Pd'] = running.get('ls_Pd', 0.0) + float(loss_dict.get('ls_Pd', 0.0))
            running['ls_Qd'] = running.get('ls_Qd', 0.0) + float(loss_dict.get('ls_Qd', 0.0))
            running['ls_V'] = running.get('ls_V', 0.0) + float(loss_dict.get('ls_V', 0.0))
            running['n'] += 1

        # Learning rate scheduler step (per-epoch, not per-batch)
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            if tb_logger is not None:
                tb_logger.log_scalar('training/learning_rate', current_lr, epoch)

        # epoch averages
        n = max(running['n'], 1)
        avg_loss = running['loss'] / n
        avg_kgenp = running['kgenp'] / n
        avg_kgenq = running['kgenq'] / n
        avg_kpd = running['kpd'] / n
        avg_kqd = running['kqd'] / n
        avg_kv = running['kv'] / n
        avg_cost = running['cost'] / n
        avg_carbon = running['carbon'] / n
        # Validation analysis averages
        avg_loss_obj = running.get('loss_obj', 0.0) / n
        avg_cost_per_mean = running.get('cost_per_mean', 0.0) / n
        avg_carbon_per_mean = running.get('carbon_per_mean', 0.0) / n
        avg_loss_Pgi_sum = running.get('loss_Pgi_sum', 0.0) / n
        avg_loss_Qgi_sum = running.get('loss_Qgi_sum', 0.0) / n
        avg_loss_Pdi_sum = running.get('loss_Pdi_sum', 0.0) / n
        avg_loss_Qdi_sum = running.get('loss_Qdi_sum', 0.0) / n
        avg_loss_Vi_sum = running.get('loss_Vi_sum', 0.0) / n
        avg_ls_cost = running.get('ls_cost', 0.0) / n
        avg_ls_Pg = running.get('ls_Pg', 0.0) / n
        avg_ls_Qg = running.get('ls_Qg', 0.0) / n
        avg_ls_Pd = running.get('ls_Pd', 0.0) / n
        avg_ls_Qd = running.get('ls_Qd', 0.0) / n
        avg_ls_V = running.get('ls_V', 0.0) / n
        
        # Calculate average gradient norm for this epoch
        avg_grad_norm = None
        if epoch_grad_norms:
            avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms)
            # Store in model for access by extra_tb_logging_fn
            if hasattr(model, 'net'):
                model._last_grad_norm = avg_grad_norm

        loss_history['total'].append(avg_loss)
        loss_history['kgenp_mean'].append(avg_kgenp)
        loss_history['kgenq_mean'].append(avg_kgenq)
        loss_history['kpd_mean'].append(avg_kpd)
        loss_history['kqd_mean'].append(avg_kqd)
        loss_history['kv_mean'].append(avg_kv)
        loss_history['cost'].append(avg_cost)
        loss_history['carbon'].append(avg_carbon)

        # print
        should_print = (epoch == 0) or ((epoch + 1) % p_epoch == 0)
        if should_print:
            elapsed = time.time() - start_time
            if epoch > 0:
                time_per_epoch = elapsed / (epoch + 1)
                remaining = time_per_epoch * (n_epochs - epoch - 1)
                time_info = f" | Elapsed: {elapsed/60:.1f}min, ETA: {remaining/60:.1f}min"
            else:
                time_info = ""

            with torch.no_grad():
                sample_pred = sample_pred_fn()
                Va_pred = sample_pred[:, :ngt_data['NPred_Va']]
                Vm_pred = sample_pred[:, ngt_data['NPred_Va']:]
            print(f"{print_prefix}epoch {epoch+1}/{n_epochs} loss={avg_loss:.4f} "
                  f"Va[{Va_pred.min().item():.4f},{Va_pred.max().item():.4f}] "
                  f"Vm[{Vm_pred.min().item():.4f},{Vm_pred.max().item():.4f}]{time_info}")
            print(f"{print_prefix}  kgenp={avg_kgenp:.2f} kgenq={avg_kgenq:.2f} "
                  f"kpd={avg_kpd:.2f} kqd={avg_kqd:.2f} kv={avg_kv:.2f}")
            # Print cost and carbon if multi-objective
            if avg_cost > 0 or avg_carbon > 0:
                print(f"{print_prefix}  cost={avg_cost:.2f} carbon={avg_carbon:.2f}")
            
            # Compute and print constraint satisfaction (on sample predictions)
            try:
                sample_pred = sample_pred_fn()
                sample_x = ngt_data['x_train'][:config.ngt_batch_size].to(device)
                constraint_stats = _compute_constraint_satisfaction(
                    sample_pred, sample_x, ngt_data, sys_data, config, device
                )
                print(f"{print_prefix}  Constraint satisfaction: Pg={constraint_stats['Pg_satisfy']:.2f}% "
                      f"Qg={constraint_stats['Qg_satisfy']:.2f}% Vm={constraint_stats['Vm_satisfy']:.2f}%")
                if constraint_stats['branch_ang_satisfy'] < 100.0:
                    print(f"{print_prefix}    Branch: ang={constraint_stats['branch_ang_satisfy']:.2f}% "
                          f"pf={constraint_stats['branch_pf_satisfy']:.2f}%")
            except Exception as e:
                # If constraint calculation fails, skip it
                pass

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
                    sample_pred = sample_pred_fn()
                    sample_x = ngt_data['x_train'][:config.ngt_batch_size].to(device)
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

            if extra_tb_logging_fn is not None:
                extra_tb_logging_fn(epoch, avg_cost, avg_carbon)

        # save checkpoint
        if (epoch + 1) % 100 == 0 and (epoch + 1) >= s_epoch:
            save_path = f"{config.model_save_dir}/{save_prefix}_{config.Nbus}bus{save_tag}_E{epoch+1}.pth"
            torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)
            print(f"{print_prefix}  Model saved: {save_path}")

    time_train = time.time() - start_time
    final_path = f"{config.model_save_dir}/{save_prefix}_{config.Nbus}bus{save_tag}_E{n_epochs}_final.pth"
    torch.save(model.state_dict(), final_path, _use_new_zipfile_serialization=False)
    print(f"{print_prefix}Training completed in {time_train:.2f}s, final saved: {final_path}")

    return loss_history, time_train


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
# Helper functions for gradient descent training
# ============================================================================

def _compute_constraint_violation(loss_dict):
    """Compute total constraint violation from loss dictionary."""
    return (
        loss_dict.get('loss_Pgi_sum', 0.0) +
        loss_dict.get('loss_Qgi_sum', 0.0) +
        loss_dict.get('loss_Pdi_sum', 0.0) +
        loss_dict.get('loss_Qdi_sum', 0.0) +
        loss_dict.get('loss_Vi_sum', 0.0)
    )


def _compute_weighted_objective(loss_dict, lambda_cost, lambda_carbon):
    """Compute weighted objective function: lambda_cost * cost + lambda_carbon * carbon."""
    return (
        lambda_cost * loss_dict.get('loss_cost', 0.0) +
        lambda_carbon * loss_dict.get('loss_carbon', 0.0)
    )


def _clip_gradient(grad_V, grad_clip_norm):
    """Clip gradient per sample to specified norm."""
    if grad_clip_norm is None or grad_clip_norm <= 0:
        return grad_V
    grad_norm_per_sample = torch.norm(grad_V, dim=1, keepdim=True)
    clip_coef = grad_clip_norm / (grad_norm_per_sample + 1e-8)
    clip_coef = torch.clamp(clip_coef, max=1.0)
    return grad_V * clip_coef


def _update_V_anchor_batch(
    V_anchor_batch, grad_V, learning_rate, Vscale, Vbias,
    train_x, ngt_data, sys_data, config, device, BRANFT,
    use_post_processing, epoch, batch_idx
):
    """Update V_anchor batch using gradient descent with optional post-processing."""
    with torch.no_grad():
        # Gradient descent update
        V_anchor_batch_new = V_anchor_batch - learning_rate * grad_V
        
        # Clamp to valid voltage range
        min_v = Vbias - 2 * Vscale
        max_v = Vbias + 2 * Vscale
        V_anchor_batch_new = torch.clamp(V_anchor_batch_new, min=min_v, max=max_v)
        
        # Apply post-processing if enabled
        if use_post_processing:
            try:
                V_anchor_batch_new = _apply_post_processing_to_batch(
                    V_anchor_batch_new.detach().cpu().numpy(),
                    train_x,
                    ngt_data,
                    sys_data,
                    config,
                    device,
                    BRANFT,
                    verbose=(epoch == 0 and batch_idx == 0)
                )
                # Ensure voltage remains in valid range after post-processing
                V_anchor_batch_new = torch.clamp(V_anchor_batch_new, min=min_v, max=max_v)
            except Exception as e:
                if epoch == 0 and batch_idx == 0:
                    print(f"[Warning] Post-processing failed: {e}, using gradient-updated voltage")
    
    return V_anchor_batch_new.detach()


def _compute_improvement_metrics(initial_stats, final_stats, lambda_cost, lambda_carbon):
    """Compute improvement metrics between initial and final solutions."""
    def _compute_pct_improvement(initial_val, final_val):
        """Helper to compute absolute and percentage improvement."""
        improvement = initial_val - final_val
        improvement_pct = 100 * improvement / (initial_val + 1e-8)
        return improvement, improvement_pct
    
    # Compute weighted objectives
    initial_weighted_obj = (
        lambda_cost * initial_stats['avg_cost'] +
        lambda_carbon * initial_stats['avg_carbon']
    )
    final_weighted_obj = (
        lambda_cost * final_stats['avg_cost'] +
        lambda_carbon * final_stats['avg_carbon']
    )
    weighted_obj_improvement, weighted_obj_improvement_pct = _compute_pct_improvement(
        initial_weighted_obj, final_weighted_obj
    )
    
    # Compute all improvement metrics
    loss_improvement, loss_improvement_pct = _compute_pct_improvement(
        initial_stats['avg_loss'], final_stats['avg_loss']
    )
    cost_improvement, cost_improvement_pct = _compute_pct_improvement(
        initial_stats['avg_cost'], final_stats['avg_cost']
    )
    carbon_improvement, carbon_improvement_pct = _compute_pct_improvement(
        initial_stats['avg_carbon'], final_stats['avg_carbon']
    )
    constraint_violation_improvement, constraint_violation_improvement_pct = _compute_pct_improvement(
        initial_stats['avg_constraint_violation'], final_stats['avg_constraint_violation']
    )
    
    improvement = {
        'loss_improvement': loss_improvement,
        'loss_improvement_pct': loss_improvement_pct,
        'cost_improvement': cost_improvement,
        'cost_improvement_pct': cost_improvement_pct,
        'carbon_improvement': carbon_improvement,
        'carbon_improvement_pct': carbon_improvement_pct,
        'weighted_obj_improvement': weighted_obj_improvement,
        'weighted_obj_improvement_pct': weighted_obj_improvement_pct,
        'constraint_violation_improvement': constraint_violation_improvement,
        'constraint_violation_improvement_pct': constraint_violation_improvement_pct,
    }
    
    return improvement, initial_weighted_obj, final_weighted_obj


def _print_results_summary(initial_stats, final_stats, improvement, lambda_cost, lambda_carbon, initial_weighted_obj, final_weighted_obj):
    """Print formatted results summary."""
    print("\n" + "="*60)
    print("Gradient Descent Validation Results")
    print("="*60)
    
    # Initial solution
    print(f"\nInitial Solution:")
    print(f"  Loss: {initial_stats['avg_loss']:.4f}")
    print(f"  Cost: {initial_stats['avg_cost']:.2f}")
    print(f"  Carbon: {initial_stats['avg_carbon']:.4f}")
    print(f"  Weighted Objective: {initial_weighted_obj:.2f} (λ_cost={lambda_cost}*cost + λ_carbon={lambda_carbon}*carbon)")
    print(f"  Constraint Violation: {initial_stats['avg_constraint_violation']:.4f}")
    cs_init = initial_stats['constraint_satisfaction']
    print(f"  Constraint Satisfaction: Pg={cs_init['Pg_satisfy']:.2f}%, "
          f"Qg={cs_init['Qg_satisfy']:.2f}%, Vm={cs_init['Vm_satisfy']:.2f}%")
    
    # Final solution
    print(f"\nFinal Solution:")
    print(f"  Loss: {final_stats['avg_loss']:.4f}")
    print(f"  Cost: {final_stats['avg_cost']:.2f}")
    print(f"  Carbon: {final_stats['avg_carbon']:.4f}")
    print(f"  Weighted Objective: {final_weighted_obj:.2f} (λ_cost={lambda_cost}*cost + λ_carbon={lambda_carbon}*carbon)")
    print(f"  Constraint Violation: {final_stats['avg_constraint_violation']:.4f}")
    cs_final = final_stats['constraint_satisfaction']
    print(f"  Constraint Satisfaction: Pg={cs_final['Pg_satisfy']:.2f}%, "
          f"Qg={cs_final['Qg_satisfy']:.2f}%, Vm={cs_final['Vm_satisfy']:.2f}%")
    
    # Improvement
    print(f"\nImprovement:")
    print(f"  Loss: {improvement['loss_improvement']:.4f} ({improvement['loss_improvement_pct']:.2f}%)")
    print(f"  Cost: {improvement['cost_improvement']:.2f} ({improvement['cost_improvement_pct']:.2f}%)")
    print(f"  Carbon: {improvement['carbon_improvement']:.4f} ({improvement['carbon_improvement_pct']:.2f}%)")
    print(f"  Weighted Objective: {improvement['weighted_obj_improvement']:.2f} "
          f"({improvement['weighted_obj_improvement_pct']:.2f}%) [KEY METRIC]")
    print(f"  Constraint Violation: {improvement['constraint_violation_improvement']:.4f} "
          f"({improvement['constraint_violation_improvement_pct']:.2f}%)")
    
    # Trade-off analysis
    print(f"\n[Trade-off Analysis]")
    if improvement['weighted_obj_improvement'] > 0:
        print(f"  [OK] Weighted objective improved by {improvement['weighted_obj_improvement']:.2f} "
              f"({improvement['weighted_obj_improvement_pct']:.2f}%)")
        if improvement['constraint_violation_improvement'] < 0:
            print(f"  [WARN] Constraint violation increased by {abs(improvement['constraint_violation_improvement']):.4f}")
            print(f"  -> Acceptable trade-off: objective improvement outweighs constraint degradation")
        else:
            print(f"  [OK] Constraint violation also improved by {improvement['constraint_violation_improvement']:.4f}")
    else:
        print(f"  [FAIL] Weighted objective degraded by {abs(improvement['weighted_obj_improvement']):.2f}")
        if improvement['constraint_violation_improvement'] < 0:
            print(f"  [FAIL] Constraint violation also increased by {abs(improvement['constraint_violation_improvement']):.4f}")
            print(f"  -> Overall degradation: both objective and constraints worsened")
    
    print("="*60)


def train_unsupervised_ngt_gradient_descent(
    config, sys_data=None, device=None,
    lambda_cost=0.9, lambda_carbon=0.1,
    use_projection=False,
    num_iterations=50,
    learning_rate=1e-5,  # Default to much smaller learning rate
    tb_logger=None,
    use_drift_correction=True,  # Enable drift correction by default
    lambda_cor=5.0,  # Drift correction gain
    grad_clip_norm=1.0,  # Gradient clipping norm
    use_post_processing=True,  # Enable post-processing after each gradient update
):
    """
    Validation function: Direct gradient descent on V_anchor.
    
    This function tests whether we can improve V_anchor by:
    1. Starting from VAE-generated V_anchor (physical space)
    2. Computing loss gradient w.r.t. V_anchor (ONLY OBJECTIVE, NO CONSTRAINTS - only_obj=True) 
    4. Updating V_anchor directly using projected gradient
    5. Iterating to find better solutions (optimizing objective while maintaining constraints via post-processing)
    
    NOTE: This function uses only_obj=True, meaning it only optimizes the objective function
    (cost + carbon) and ignores constraint violations in the loss. Constraints are maintained
    through post-processing after each gradient update.
    
    Args:
        config: Configuration object
        sys_data: PowerSystemData object (optional, will load if None)
        device: Device (optional, uses config.device if None)
        lambda_cost: Weight for cost objective
        lambda_carbon: Weight for carbon objective
        use_projection: Whether to use P_tan_t for gradient projection
        num_iterations: Number of gradient descent iterations
        learning_rate: Learning rate for gradient updates (DEFAULT: 1e-5, recommended range: 1e-5 to 1e-6)
        tb_logger: TensorBoardLogger instance for logging (optional)
        use_drift_correction: Whether to use drift correction (default: True)
        lambda_cor: Drift correction gain (default: 5.0)
        grad_clip_norm: Gradient clipping norm (default: 1.0)
        
    Returns:
        results: Dictionary containing:
            - V_anchor_initial: Initial V_anchor from VAE
            - V_anchor_final: Final V_anchor after gradient descent
            - loss_history: Loss values during iterations
            - constraint_stats_initial: Constraint satisfaction for initial V_anchor
            - constraint_stats_final: Constraint satisfaction for final V_anchor
            - improvement: Improvement metrics 
    """
    
    if device is None:
        device = config.device
    
    # ========================================================================
    # Initialization
    # ========================================================================
    
    # Load system data and compute BRANFT (needed for post-processing)
    if sys_data is None:
        from data_loader import load_all_data
        sys_data, _, BRANFT = load_all_data(config)
    else:
        branch_np = sys_data.branch if isinstance(sys_data.branch, np.ndarray) else sys_data.branch.numpy()
        BRANFT = branch_np[:, 0:2] - 1  # Convert to 0-indexed
    
    # Load NGT training data
    ngt_data, sys_data = load_ngt_training_data(config, sys_data)
    Vscale = ngt_data['Vscale'].to(device)
    Vbias = ngt_data['Vbias'].to(device)
    
    from models import create_model
    
    bus_slack = int(sys_data.bus_slack)
    
    # Load and freeze VAE models for anchor generation
    def _load_frozen_vae_or_raise():
        """Load and freeze VAE models for generating voltage anchors."""
        vae_vm_path = config.pretrain_model_path_vm
        vae_va_path = config.pretrain_model_path_va
        input_dim = ngt_data['input_dim']
        
        vae_vm = create_model('vae', input_dim, config.Nbus, config, is_vm=True).to(device)
        vae_va = create_model('vae', input_dim, config.Nbus - 1, config, is_vm=False).to(device)
        vae_vm.load_state_dict(torch.load(vae_vm_path, map_location=device, weights_only=True), strict=True)
        vae_va.load_state_dict(torch.load(vae_va_path, map_location=device, weights_only=True), strict=True)
        
        vae_vm.eval()
        vae_va.eval()
        for p in vae_vm.parameters():
            p.requires_grad = False
        for p in vae_va.parameters():
            p.requires_grad = False
        return vae_vm, vae_va
    
    vae_vm, vae_va = _load_frozen_vae_or_raise()
    
    # Setup loss function
    config.ngt_lambda_cost = lambda_cost
    config.ngt_lambda_carbon = lambda_carbon
    loss_fn = DeepOPFNGTLoss(sys_data, config)
    loss_fn.cache_to_gpu(device)
    
    # Setup preference for multi-objective optimization
    use_multi_objective = _use_multi_objective(config)
    use_pref = use_multi_objective and _use_preference_conditioning(config)
    pref_base = _make_pref_base(device, lambda_cost, lambda_carbon) if use_pref else None
    
    # Create training DataLoader with indices
    class IndexedTensorDataset(Data.Dataset):
        """Dataset that returns data with indices for batch tracking."""
        def __init__(self, *tensors):
            assert all(t.size(0) == tensors[0].size(0) for t in tensors)
            self.tensors = tensors
        
        def __getitem__(self, index):
            return tuple(t[index] for t in self.tensors) + (index,)
        
        def __len__(self):
            return self.tensors[0].size(0)
    
    indexed_dataset = IndexedTensorDataset(ngt_data['x_train'], ngt_data['y_train'])
    training_loader = Data.DataLoader(indexed_dataset, batch_size=config.ngt_batch_size, shuffle=False)
    
    # Precompute initial V_anchor (physical space) for all training samples
    print("\n[Gradient Descent] Computing V_anchor from VAE...")
    V_anchor_all = _precompute_V_anchor_physical_all(
        config=config,
        sys_data=sys_data,
        ngt_data=ngt_data,
        device=device,
        vae_vm=vae_vm,
        vae_va=vae_va,
        Vscale=Vscale,
        Vbias=Vbias,
        bus_slack=bus_slack,
    )
    
    # Initialize loss history tracking
    loss_history = {
        'total': [],
        'cost': [],
        'carbon': [],
        'constraint_violation': [],
    }
    
    # ========================================================================
    # Gradient Descent Training Loop
    # ========================================================================
    
    start_time = time.time()
    
    for epoch in range(num_iterations):
        epoch_losses = []
        epoch_costs = []
        epoch_carbons = []
        epoch_constraint_violations = []
        epoch_weighted_objectives = []  # Track weighted objective function
        
        for batch_idx, (train_x, train_y, batch_indices) in enumerate(training_loader):
            train_x = train_x.to(device)
            batch_size = train_x.shape[0]
            
            # Get current V_anchor for this batch and enable gradients
            V_anchor_batch = V_anchor_all[batch_indices].clone().requires_grad_(True)
            
            # Setup preference for loss computation
            pref_batch = _expand_pref(pref_base, batch_size) if use_pref else None
            
            # Compute loss and gradient
            loss, loss_dict = _call_ngt_loss(loss_fn, config, V_anchor_batch, train_x, pref_batch, only_obj=False)
            
            grad_V = torch.autograd.grad(
                outputs=loss,
                inputs=V_anchor_batch,
                create_graph=False,
                retain_graph=False,
            )[0]
            
            # Clip gradient to prevent large updates
            grad_V = _clip_gradient(grad_V, grad_clip_norm)
            
            # Update V_anchor using gradient descent with optional post-processing
            V_anchor_batch_new = _update_V_anchor_batch(
                V_anchor_batch=V_anchor_batch,
                grad_V=grad_V,
                learning_rate=learning_rate,
                Vscale=Vscale,
                Vbias=Vbias,
                train_x=train_x,
                ngt_data=ngt_data,
                sys_data=sys_data,
                config=config,
                device=device,
                BRANFT=BRANFT,
                use_post_processing=use_post_processing,
                epoch=epoch,
                batch_idx=batch_idx
            )
            
            # Update stored V_anchor
            V_anchor_all[batch_indices] = V_anchor_batch_new
            
            # Record metrics
            epoch_losses.append(loss.item())
            epoch_costs.append(loss_dict.get('loss_cost', 0.0))
            epoch_carbons.append(loss_dict.get('loss_carbon', 0.0))
            epoch_weighted_objectives.append(_compute_weighted_objective(loss_dict, lambda_cost, lambda_carbon))
            epoch_constraint_violations.append(_compute_constraint_violation(loss_dict))
        
        # Compute average metrics for this epoch
        n_batches = len(epoch_losses)
        if n_batches > 0:
            avg_loss = sum(epoch_losses) / n_batches
            avg_cost = sum(epoch_costs) / n_batches
            avg_carbon = sum(epoch_carbons) / n_batches
            avg_constraint_violation = sum(epoch_constraint_violations) / n_batches
            avg_weighted_obj = sum(epoch_weighted_objectives) / n_batches
        else:
            avg_loss = avg_cost = avg_carbon = avg_constraint_violation = avg_weighted_obj = 0.0
        
        # Update loss history
        loss_history['total'].append(avg_loss)
        loss_history['cost'].append(avg_cost)
        loss_history['carbon'].append(avg_carbon)
        loss_history['constraint_violation'].append(avg_constraint_violation)
        
        # Print progress periodically
        print_interval = max(1, num_iterations // 500)
        if (epoch + 1) % print_interval == 0 or epoch == 0:
            print(f"  Iteration {epoch+1}/{num_iterations}: "
                  f"loss={avg_loss:.4f}, cost={avg_cost:.2f}, carbon={avg_carbon:.4f}, "
                  f"weighted_obj={avg_weighted_obj:.2f}, constraint_vio={avg_constraint_violation:.4f}")
        
        # TensorBoard logging
        if tb_logger:
            tb_logger.log_scalar('gradient_descent/loss', avg_loss, epoch)
            tb_logger.log_scalar('gradient_descent/cost', avg_cost, epoch)
            tb_logger.log_scalar('gradient_descent/carbon', avg_carbon, epoch)
            tb_logger.log_scalar('gradient_descent/constraint_violation', avg_constraint_violation, epoch)
    
    elapsed_time = time.time() - start_time
    print(f"\n[Gradient Descent] Completed in {elapsed_time:.2f}s")
    
    # ========================================================================
    # Evaluate Results
    # ========================================================================
    
    print("\n[Gradient Descent] Computing final statistics...")
    
    # Recompute initial V_anchor (since V_anchor_all was modified during training)
    initial_V_anchor_all = _precompute_V_anchor_physical_all(
        config=config,
        sys_data=sys_data,
        ngt_data=ngt_data,
        device=device,
        vae_vm=vae_vm,
        vae_va=vae_va,
        Vscale=Vscale,
        Vbias=Vbias,
        bus_slack=bus_slack,
    )
    
    # Evaluate initial and final solutions
    initial_stats = _evaluate_solution_batch(
        V_pred=initial_V_anchor_all,
        training_loader=training_loader,
        loss_fn=loss_fn,
        config=config,
        ngt_data=ngt_data,
        sys_data=sys_data,
        device=device,
        pref_base=pref_base,
        use_pref=use_pref,
    )
    
    final_stats = _evaluate_solution_batch(
        V_pred=V_anchor_all,
        training_loader=training_loader,
        loss_fn=loss_fn,
        config=config,
        ngt_data=ngt_data,
        sys_data=sys_data,
        device=device,
        pref_base=pref_base,
        use_pref=use_pref,
    )
    
    # Compute improvement metrics
    improvement, initial_weighted_obj, final_weighted_obj = _compute_improvement_metrics(
        initial_stats, final_stats, lambda_cost, lambda_carbon
    )
    
    # Print results summary
    _print_results_summary(
        initial_stats, final_stats, improvement, lambda_cost, lambda_carbon,
        initial_weighted_obj, final_weighted_obj
    )
    
    # Prepare results
    results = {
        'V_anchor_initial': initial_V_anchor_all.detach().cpu().numpy(),
        'V_anchor_final': V_anchor_all.detach().cpu().numpy(),
        'loss_history': loss_history,
        'initial_stats': initial_stats,
        'final_stats': final_stats,
        'improvement': improvement,
        'config': {
            'num_iterations': num_iterations,
            'learning_rate': learning_rate,
            'use_projection': use_projection,
            'lambda_cost': lambda_cost,
            'lambda_carbon': lambda_carbon,
        },
    }
    
    return results


def _precompute_V_anchor_physical_all(
    *,
    config, sys_data, ngt_data, device,
    vae_vm, vae_va,
    Vscale, Vbias,
    bus_slack,
):
    """
    Precompute V_anchor in physical space (NGT format) for all training samples.
    
    Returns:
        V_anchor_physical: [N, output_dim] in physical space
    """
    x_train = ngt_data['x_train'].to(device)
    bus_Pnet_all = ngt_data['bus_Pnet_all']
    bus_Pnet_noslack_all = bus_Pnet_all[bus_Pnet_all != bus_slack]
    
    # VAE forward pass
    with torch.no_grad():
        Vm_scaled = vae_vm(x_train, use_mean=True)           # [N, Nbus] (scaled)
        Va_scaled_noslack = vae_va(x_train, use_mean=True)   # [N, Nbus-1] (scaled)
    
    scale_vm = config.scale_vm.item() if hasattr(config.scale_vm, 'item') else float(config.scale_vm)
    scale_va = config.scale_va.item() if hasattr(config.scale_va, 'item') else float(config.scale_va)
    
    VmLb = sys_data.VmLb
    VmUb = sys_data.VmUb
    if isinstance(VmLb, np.ndarray):
        VmLb = torch.from_numpy(VmLb).float().to(device)
        VmUb = torch.from_numpy(VmUb).float().to(device)
    elif isinstance(VmLb, torch.Tensor):
        VmLb = VmLb.to(device)
        VmUb = VmUb.to(device)
    
    Vm_anchor_full = Vm_scaled / scale_vm * (VmUb - VmLb) + VmLb
    Va_anchor_full_noslack = Va_scaled_noslack / scale_va
    
    N_samples = x_train.shape[0]
    Va_anchor_full = torch.zeros(N_samples, config.Nbus, device=device)
    Va_anchor_full[:, :bus_slack] = Va_anchor_full_noslack[:, :bus_slack]
    Va_anchor_full[:, bus_slack + 1:] = Va_anchor_full_noslack[:, bus_slack:]
    
    Vm_nonZIB = Vm_anchor_full[:, bus_Pnet_all]
    Va_nonZIB_noslack = Va_anchor_full[:, bus_Pnet_noslack_all]
    V_anchor_physical = torch.cat([Va_nonZIB_noslack, Vm_nonZIB], dim=1)
    
    return V_anchor_physical.detach()


def _evaluate_solution_batch(
    V_pred,
    training_loader,
    loss_fn,
    config,
    ngt_data,
    sys_data,
    device,
    pref_base,
    use_pref,
):
    """
    Evaluate a batch of solutions and return statistics.
    
    Returns:
        dict with avg_loss, avg_cost, avg_carbon, avg_constraint_violation, constraint_satisfaction
    """
    total_loss = 0.0
    total_cost = 0.0
    total_carbon = 0.0
    total_constraint_violation = 0.0
    n_samples = 0
    
    all_constraint_stats = []
    
    with torch.no_grad():
        for train_x, train_y, batch_indices in training_loader:
            train_x = train_x.to(device)
            batch_size = train_x.shape[0]
            V_batch = V_pred[batch_indices].to(device)
            
            # Setup preference
            if use_pref:
                pref_batch = _expand_pref(pref_base, batch_size)
            else:
                pref_batch = None
            
            # Compute loss (only objective, no constraints)
            loss, loss_dict = _call_ngt_loss(loss_fn, config, V_batch, train_x, pref_batch, only_obj=True)
            
            # Accumulate statistics
            total_loss += loss.item() * batch_size
            total_cost += loss_dict.get('loss_cost', 0.0) * batch_size
            total_carbon += loss_dict.get('loss_carbon', 0.0) * batch_size
            total_constraint_violation += _compute_constraint_violation(loss_dict) * batch_size
            
            # Compute constraint satisfaction
            constraint_stats = _compute_constraint_satisfaction(
                V_batch, train_x, ngt_data, sys_data, config, device
            )
            all_constraint_stats.append(constraint_stats)
            
            n_samples += batch_size
    
    # Average statistics
    avg_loss = total_loss / max(n_samples, 1)
    avg_cost = total_cost / max(n_samples, 1)
    avg_carbon = total_carbon / max(n_samples, 1)
    avg_constraint_violation = total_constraint_violation / max(n_samples, 1)
    
    # Average constraint satisfaction
    avg_constraint_satisfaction = {
        'Pg_satisfy': sum(s['Pg_satisfy'] for s in all_constraint_stats) / len(all_constraint_stats) if all_constraint_stats else 100.0,
        'Qg_satisfy': sum(s['Qg_satisfy'] for s in all_constraint_stats) / len(all_constraint_stats) if all_constraint_stats else 100.0,
        'Vm_satisfy': sum(s['Vm_satisfy'] for s in all_constraint_stats) / len(all_constraint_stats) if all_constraint_stats else 100.0,
        'branch_ang_satisfy': sum(s.get('branch_ang_satisfy', 100.0) for s in all_constraint_stats) / len(all_constraint_stats) if all_constraint_stats else 100.0,
        'branch_pf_satisfy': sum(s.get('branch_pf_satisfy', 100.0) for s in all_constraint_stats) / len(all_constraint_stats) if all_constraint_stats else 100.0,
    }
    
    return {
        'avg_loss': avg_loss,
        'avg_cost': avg_cost,
        'avg_carbon': avg_carbon,
        'avg_constraint_violation': avg_constraint_violation,
        'constraint_satisfaction': avg_constraint_satisfaction,
    }


# ------------------------ helpers ------------------------

def _ngt_to_full_voltage(V_ngt, ngt_data, sys_data, config, device):
    """
    Convert NGT format voltage to full voltage (Vm_full, Va_full).
    
    Args:
        V_ngt: [batch, NPred_Va + NPred_Vm] in NGT format (physical space)
        ngt_data: NGT data dictionary
        sys_data: System data
        config: Configuration
        device: Device
        
    Returns:
        Vm_full: [batch, Nbus] full voltage magnitude
        Va_full: [batch, Nbus] full voltage angle (with slack inserted)
    """
    batch_size = V_ngt.shape[0]
    bus_slack = int(sys_data.bus_slack)
    
    # Convert to numpy
    V_ngt_np = V_ngt.detach().cpu().numpy() if torch.is_tensor(V_ngt) else V_ngt
    
    # Insert slack bus Va (=0) to get full non-ZIB voltage
    xam_P = np.insert(V_ngt_np, ngt_data['idx_bus_Pnet_slack'][0], 0, axis=1)
    Va_len_with_slack = ngt_data['NPred_Va'] + 1
    Va_nonZIB = xam_P[:, :Va_len_with_slack]
    Vm_nonZIB = xam_P[:, Va_len_with_slack:Va_len_with_slack + ngt_data['NPred_Vm']]
    
    # Convert to complex and reconstruct full voltage
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
    
    Vm_full = np.sqrt(Ve**2 + Vf**2)
    Va_full = np.arctan2(Vf, Ve)
    
    return Vm_full, Va_full


def _full_to_ngt_voltage(Vm_full, Va_full, ngt_data, sys_data, config):
    """
    Convert full voltage back to NGT format.
    
    Args:
        Vm_full: [batch, Nbus] full voltage magnitude
        Va_full: [batch, Nbus] full voltage angle (with slack)
        ngt_data: NGT data dictionary
        sys_data: System data
        config: Configuration
        
    Returns:
        V_ngt: [batch, NPred_Va + NPred_Vm] in NGT format (physical space)
    """
    batch_size = Vm_full.shape[0]
    bus_slack = int(sys_data.bus_slack)
    bus_Pnet_all = _ensure_1d_int(ngt_data['bus_Pnet_all'])
    bus_Pnet_noslack_all = bus_Pnet_all[bus_Pnet_all != bus_slack]
    
    # Extract non-ZIB voltages
    Vm_nonZIB = Vm_full[:, bus_Pnet_all]
    Va_nonZIB_noslack = Va_full[:, bus_Pnet_noslack_all]
    
    # Concatenate to NGT format: [Va_nonZIB_noslack, Vm_nonZIB]
    V_ngt = np.concatenate([Va_nonZIB_noslack, Vm_nonZIB], axis=1)
    
    return V_ngt


def _apply_post_processing_to_batch(
    V_ngt_batch, train_x_batch, ngt_data, sys_data, config, device, BRANFT, verbose=False
):
    """
    Apply post-processing to a batch of NGT format voltages.
    
    Args:
        V_ngt_batch: [batch, NPred_Va + NPred_Vm] in NGT format (physical space)
        train_x_batch: [batch, input_dim] load data
        ngt_data: NGT data dictionary
        sys_data: System data
        config: Configuration
        device: Device
        BRANFT: Branch from-to indices
        verbose: Whether to print warnings
        
    Returns:
        V_ngt_corrected: [batch, NPred_Va + NPred_Vm] corrected NGT format voltage
    """
    batch_size = V_ngt_batch.shape[0]
    
    # Convert NGT to full voltage
    Vm_full, Va_full = _ngt_to_full_voltage(V_ngt_batch, ngt_data, sys_data, config, device)
    
    # Build load arrays
    num_Pd = len(ngt_data['bus_Pd'])
    Pd_full = np.zeros((batch_size, config.Nbus))
    Qd_full = np.zeros((batch_size, config.Nbus))
    train_x_np = train_x_batch.detach().cpu().numpy() if torch.is_tensor(train_x_batch) else train_x_batch
    Pd_full[:, ngt_data['bus_Pd']] = train_x_np[:, :num_Pd]
    Qd_full[:, ngt_data['bus_Qd']] = train_x_np[:, num_Pd:]
    
    # Create a minimal EvalContext for post-processing
    try:
        # Create a minimal context-like structure
        class TempContext:
            def __init__(self):
                self.config = config
                self.sys_data = sys_data
                self.BRANFT = BRANFT
                self.device = device
                self.Nbus = config.Nbus
                self.Ntest = batch_size
                self.bus_slack = int(sys_data.bus_slack)
                self.baseMVA = float(sys_data.baseMVA.item() if hasattr(sys_data.baseMVA, 'item') else sys_data.baseMVA)
                self.branch = _as_numpy(sys_data.branch)
                self.Ybus = sys_data.Ybus
                self.Yf = sys_data.Yf
                self.Yt = sys_data.Yt
                self.bus_Pg = _ensure_1d_int(sys_data.bus_Pg)
                self.bus_Qg = _ensure_1d_int(sys_data.bus_Qg)
                self.MAXMIN_Pg = _as_numpy(ngt_data['MAXMIN_Pg'])
                self.MAXMIN_Qg = _as_numpy(ngt_data['MAXMIN_Qg'])
                self.idxPg = _ensure_1d_int(sys_data.idxPg)
                self.gencost = _as_numpy(sys_data.gencost)
                self.gencost_Pg = _as_numpy(ngt_data.get('gencost_Pg', None))
                self.his_V = _as_numpy(sys_data.his_V)
                self.hisVm_min = _as_numpy(sys_data.hisVm_min)
                self.hisVm_max = _as_numpy(sys_data.hisVm_max)
                self.bus_Pnet_all = _ensure_1d_int(ngt_data['bus_Pnet_all'])
                self.bus_Pnet_noslack_all = self.bus_Pnet_all[self.bus_Pnet_all != self.bus_slack]
                self.bus_ZIB_all = _ensure_1d_int(ngt_data['bus_ZIB_all']) if 'bus_ZIB_all' in ngt_data else None
                self.param_ZIMV = ngt_data.get('param_ZIMV', None)
                self.VmLb = getattr(config, 'ngt_VmLb', None)
                self.VmUb = getattr(config, 'ngt_VmUb', None)
                self.DELTA = float(getattr(config, 'DELTA', 1e-4))
                self.k_dV = float(getattr(config, 'k_dV', 1.0))
                # [IMPROVEMENT] Use current voltage for Jacobian calculation instead of historical
                # This ensures more accurate linearization when voltage deviates from historical values
                self.flag_hisv = False  # Use current voltage, not historical
                self.Pdtest = Pd_full
                self.Qdtest = Qd_full
                # Store current voltage for Jacobian calculation
                self.current_V = Vm_full * np.exp(1j * Va_full)
        
        temp_ctx = TempContext()
        
        # Apply post-processing
        Vm_corrected, Va_corrected, _, _ = post_process_like_evaluate_model(
            temp_ctx, Vm_full, Va_full
        )
        
        # [IMPROVEMENT] Re-apply Kron reconstruction to ensure ZIB nodes satisfy Kron relationship
        # This is critical: post-processing may have corrected ZIB nodes, but we need to ensure
        # they still satisfy the Kron reduction relationship for NGT format consistency
        # Strategy: Keep non-ZIB corrections, then reconstruct ZIB from corrected non-ZIB using Kron
        if ngt_data.get('param_ZIMV') is not None and ngt_data.get('bus_ZIB_all') is not None:
            # Extract corrected non-ZIB voltages (these are the ones we want to keep)
            bus_Pnet_all = _ensure_1d_int(ngt_data['bus_Pnet_all'])
            bus_slack = int(sys_data.bus_slack)
            bus_Pnet_noslack_all = bus_Pnet_all[bus_Pnet_all != bus_slack]
            
            # Get corrected non-ZIB voltages
            Vm_nonZIB_corrected = Vm_corrected[:, bus_Pnet_all].copy()
            
            # Reconstruct Va_nonZIB with slack insertion (need full Va for all non-ZIB buses)
            # bus_Pnet_all includes slack, so we need to extract Va for all non-ZIB buses
            # Va_corrected shape: [batch, Nbus] - contains all buses including slack and ZIB
            # Simply extract Va for bus_Pnet_all (slack angle is already 0 in Va_corrected)
            Va_nonZIB_with_slack = Va_corrected[:, bus_Pnet_all].copy()
            # Ensure slack bus angle is 0 (should already be, but enforce it)
            idx_slack_in_Pnet = np.where(bus_Pnet_all == bus_slack)[0]
            if len(idx_slack_in_Pnet) > 0:
                Va_nonZIB_with_slack[:, idx_slack_in_Pnet[0]] = 0.0
            
            # Convert to complex and reconstruct ZIB using Kron
            Vx_corrected = Vm_nonZIB_corrected * np.exp(1j * Va_nonZIB_with_slack)
            Vy_reconstructed = np.dot(ngt_data['param_ZIMV'], Vx_corrected.T).T
            
            # Update corrected voltages: keep non-ZIB corrections, use Kron-reconstructed ZIB
            Vm_corrected_final = Vm_corrected.copy()
            Va_corrected_final = Va_corrected.copy()
            Vm_corrected_final[:, ngt_data['bus_ZIB_all']] = np.abs(Vy_reconstructed)
            Va_corrected_final[:, ngt_data['bus_ZIB_all']] = np.angle(Vy_reconstructed)
            
            Vm_corrected = Vm_corrected_final
            Va_corrected = Va_corrected_final
        
        # Convert back to NGT format
        V_ngt_corrected = _full_to_ngt_voltage(Vm_corrected, Va_corrected, ngt_data, sys_data, config)
        
        return torch.tensor(V_ngt_corrected, dtype=torch.float32, device=device)
        
    except Exception as e:
        # If post-processing fails, return original voltage
        if verbose:
            print(f"[Warning] Post-processing failed: {e}, returning original voltage")
        return V_ngt_batch if torch.is_tensor(V_ngt_batch) else torch.tensor(V_ngt_batch, dtype=torch.float32, device=device)


def _analyze_projection_matrix(P_tan_t, ngt_data, sys_data, config, device, learning_rate=0.01):
    """
    Analyze the projection matrix to understand why constraints might not be preserved.
    
    This function helps diagnose:
    1. What constraints the projection matrix actually preserves
    2. The quality of the projection (how much of the space is preserved)
    3. Potential issues with the projection approach
    """
    print("\n" + "="*60)
    print("Projection Matrix Analysis")
    print("="*60)
    
    # Basic properties
    dim = P_tan_t.shape[0]
    trace_P = torch.trace(P_tan_t).item()
    rank_P = torch.linalg.matrix_rank(P_tan_t).item()
    
    print(f"\n1. Basic Properties:")
    print(f"   Dimension: {dim}")
    print(f"   Trace: {trace_P:.4f} (should be < {dim} for proper projection)")
    print(f"   Rank: {rank_P} (should be < {dim})")
    print(f"   Preserved space dimension: {rank_P}")
    print(f"   Constrained space dimension: {dim - rank_P}")
    
    # Eigenvalue analysis
    eigenvals, eigenvecs = torch.linalg.eig(P_tan_t)
    eigenvals_real = eigenvals.real
    eigenvals_imag = eigenvals.imag
    
    print(f"\n2. Eigenvalue Analysis:")
    print(f"   Real eigenvalues range: [{eigenvals_real.min().item():.4f}, {eigenvals_real.max().item():.4f}]")
    print(f"   Expected: eigenvalues should be 0 or 1 (for projection matrix)")
    print(f"   Eigenvalues close to 1: {(torch.abs(eigenvals_real - 1.0) < 0.1).sum().item()}")
    print(f"   Eigenvalues close to 0: {(torch.abs(eigenvals_real) < 0.1).sum().item()}")
    
    # Check what constraints are preserved
    print(f"\n3. Constraint Coverage:")
    print(f"   [WARNING] P_tan only preserves LINEARIZED generator constraints (Pg, Qg)")
    print(f"   It does NOT preserve:")
    print(f"     - Load balance constraints (Pd, Qd) - these are in the loss function")
    print(f"     - Voltage bounds (Vm_min, Vm_max) - these are in the loss function")
    print(f"     - Branch constraints - these are in the loss function")
    print(f"     - ZIB constraints - these are in the loss function")
    
    # Nonlinearity analysis
    print(f"\n4. Nonlinearity Issue:")
    print(f"   [CRITICAL] Power flow constraints are HIGHLY NONLINEAR")
    print(f"   - Pg(V) = f(Vm, Va) is quadratic in V")
    print(f"   - Linearization dPg/dV is only valid near the current point")
    print(f"   - When V changes by dV, the actual constraint change is:")
    print(f"     dPg ~ dPg/dV @ dV + O(dV^2)  (second-order terms matter!)")
    print(f"   - Projection only preserves the linear term, not the quadratic term")
    
    # Step size analysis
    print(f"\n5. Step Size Requirements:")
    print(f"   For linearization to be valid, we need: ||ΔV|| << 1")
    print(f"   With learning_rate={learning_rate:.6f}, typical step size:")
    print(f"     ||ΔV|| ≈ lr * ||grad||")
    print(f"   If ||grad|| is large, linearization breaks down")
    
    # Recommendations
    print(f"\n6. Recommendations:")
    print(f"   a) Use MUCH smaller learning rate (e.g., 1e-4 to 1e-5)")
    print(f"   b) Add drift correction: correction = -λ * F^+ @ f(V)")
    print(f"      This pulls the state back to feasible region")
    print(f"   c) Recompute projection matrix periodically (not just once)")
    print(f"   d) Consider using only for small steps, then re-linearize")
    print(f"   e) Projection alone is NOT sufficient - need correction term")
    
    print("="*60 + "\n")


# ------------------------ helpers ------------------------

def _setup_projection_matrix_for_flow(sys_data, config, ngt_data, device):
    try:
        from flow_model.post_processing import ConstraintProjectionV2
        projector = ConstraintProjectionV2(sys_data, config, use_historical_jacobian=True)
        P_tan_full, _, _ = projector.compute_projection_matrix()

        bus_Pnet_all = ngt_data['bus_Pnet_all']
        bus_slack = int(sys_data.bus_slack)
        Nbus = config.Nbus

        bus_Pnet_noslack = bus_Pnet_all[bus_Pnet_all != bus_slack]
        all_buses_noslack = np.concatenate([np.arange(bus_slack), np.arange(bus_slack + 1, Nbus)])

        idx_Vm_in_Ptan = bus_Pnet_all
        idx_Va_in_Ptan = []
        for bus in bus_Pnet_noslack:
            pos = np.where(all_buses_noslack == bus)[0][0]
            idx_Va_in_Ptan.append(Nbus + pos)
        idx_Va_in_Ptan = np.array(idx_Va_in_Ptan)

        idx_flow_in_Ptan = np.concatenate([idx_Va_in_Ptan, idx_Vm_in_Ptan])
        P_tan_flow = P_tan_full[np.ix_(idx_flow_in_Ptan, idx_flow_in_Ptan)]
        return torch.tensor(P_tan_flow, dtype=torch.float32, device=device)
    except Exception as e:
        return None


def _precompute_z_anchor_all(
    *,
    config, sys_data, ngt_data, device,
    vae_vm, vae_va,
    Vscale, Vbias,
    bus_slack,
    use_flow_anchor, 
    flow_inf_steps,
):
    idx_train = ngt_data['idx_train']
    x_train = ngt_data['x_train'].to(device)
    bus_Pnet_all = ngt_data['bus_Pnet_all']
    bus_Pnet_noslack_all = bus_Pnet_all[bus_Pnet_all != bus_slack]

    eps = 1e-6

    # ---- Step A: VAE -> physical anchor in NGT format ----
    with torch.no_grad():
        Vm_scaled = vae_vm(x_train, use_mean=True)           # [N, Nbus] (scaled)
        Va_scaled_noslack = vae_va(x_train, use_mean=True)   # [N, Nbus-1] (scaled)

    scale_vm = config.scale_vm.item() if hasattr(config.scale_vm, 'item') else float(config.scale_vm)
    scale_va = config.scale_va.item() if hasattr(config.scale_va, 'item') else float(config.scale_va)

    VmLb = sys_data.VmLb
    VmUb = sys_data.VmUb
    if isinstance(VmLb, np.ndarray):
        VmLb = torch.from_numpy(VmLb).float().to(device)
        VmUb = torch.from_numpy(VmUb).float().to(device)
    elif isinstance(VmLb, torch.Tensor):
        VmLb = VmLb.to(device); VmUb = VmUb.to(device)

    Vm_anchor_full = Vm_scaled / scale_vm * (VmUb - VmLb) + VmLb
    Va_anchor_full_noslack = Va_scaled_noslack / scale_va

    Va_anchor_full = torch.zeros(len(idx_train), config.Nbus, device=device)
    Va_anchor_full[:, :bus_slack] = Va_anchor_full_noslack[:, :bus_slack]
    Va_anchor_full[:, bus_slack + 1:] = Va_anchor_full_noslack[:, bus_slack:]

    Vm_nonZIB = Vm_anchor_full[:, bus_Pnet_all]
    Va_nonZIB_noslack = Va_anchor_full[:, bus_Pnet_noslack_all]
    V_anchor_physical = torch.cat([Va_nonZIB_noslack, Vm_nonZIB], dim=1)

    # ---- Step B: physical -> latent (pre-sigmoid) via logit ----
    u = (V_anchor_physical - Vbias) / (Vscale + 1e-12)
    u = torch.clamp(u, eps, 1 - eps)
    z_vae_anchor = torch.log(u / (1 - u)) 
    
    return z_vae_anchor.detach()  


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

def _sample_pref_dirichlet(batch_size: int, device, alpha: float = 1.0, eps: float = 1e-6):
    """
    Dirichlet(α, α) 采样，输出 [B,2]，和为 1
    α<1 偏向角点；α=1 均匀；α>1 偏向中间
    """
    dist = torch.distributions.Dirichlet(torch.tensor([alpha, alpha], device=device))
    pref = dist.sample((batch_size,))
    return torch.clamp(pref, eps, 1.0 - eps)

def _sample_pref_beta(batch_size: int, device, a: float = 1.0, b: float = 1.0, eps: float = 1e-6):
    """
    λ_cost ~ Beta(a,b)，λ_carbon = 1-λ_cost
    """
    dist = torch.distributions.Beta(concentration1=a, concentration0=b)
    lam_cost = dist.sample((batch_size,)).to(device)
    lam_cost = torch.clamp(lam_cost, eps, 1.0 - eps)
    pref = torch.stack([lam_cost, 1.0 - lam_cost], dim=1)
    return pref

def _sample_pref_uniform(batch_size: int, device, eps: float = 1e-6):
    """
    λ_cost ~ Uniform(0,1)
    """
    lam_cost = torch.rand(batch_size, device=device)
    lam_cost = torch.clamp(lam_cost, eps, 1.0 - eps)
    pref = torch.stack([lam_cost, 1.0 - lam_cost], dim=1)
    return pref

def build_preference_sampler(config, device, lambda_cost: float, lambda_carbon: float):
    """
    返回一个函数: pref_sampler(batch_size)->pref_batch 或 None
    - 单目标：永远返回 None（完全不引入偏好）
    - 多目标且 conditioning=True：
        - config.ngt_pref_sampling = "fixed" 或 "random"
        - config.ngt_pref_level = "batch" 或 "sample"
        - config.ngt_pref_method = "dirichlet"|"beta"|"uniform"
        - 可选角点混合：config.ngt_pref_corner_prob
    """
    use_mo = _use_multi_objective(config)
    use_cond = _use_preference_conditioning(config)

    if (not use_mo) or (not use_cond):
        # 单目标或不做条件化：不喂 preference
        def _none_sampler(batch_size: int):
            return None
        return _none_sampler

    # 多目标 + 条件化：才进入这里
    sampling = getattr(config, "ngt_pref_sampling", "fixed")     # "fixed" or "random"
    level = getattr(config, "ngt_pref_level", "batch")          # "batch" or "sample"
    method = getattr(config, "ngt_pref_method", "dirichlet")    # "dirichlet"|"beta"|"uniform"
    corner_prob = float(getattr(config, "ngt_pref_corner_prob", 0.0))  # e.g. 0.1
    eps = float(getattr(config, "ngt_pref_eps", 1e-6))

    pref_base = _make_pref_base(device, lambda_cost, lambda_carbon)

    def _draw_random(B: int):
        if method == "dirichlet":
            alpha = float(getattr(config, "ngt_pref_dirichlet_alpha", 1.0))
            pref = _sample_pref_dirichlet(B, device, alpha=alpha, eps=eps)
        elif method == "beta":
            a = float(getattr(config, "ngt_pref_beta_a", 1.0))
            b = float(getattr(config, "ngt_pref_beta_b", 1.0))
            pref = _sample_pref_beta(B, device, a=a, b=b, eps=eps)
        elif method == "uniform":
            pref = _sample_pref_uniform(B, device, eps=eps)
        else:
            raise ValueError(f"Unknown ngt_pref_method: {method}")
        return pref

    def pref_sampler(batch_size: int):
        if sampling == "fixed":
            # 固定偏好：每个 batch 都是同一个 λ（但仍然喂给模型）
            return pref_base.expand(batch_size, -1)

        # sampling == "random"
        if level == "batch":
            # 每个 batch 采一个 λ，再扩展到全 batch
            pref1 = _draw_random(1).expand(batch_size, -1)
            pref = pref1
        elif level == "sample":
            # 每个样本采一个 λ
            pref = _draw_random(batch_size)
        else:
            raise ValueError(f"Unknown ngt_pref_level: {level}")

        # 角点混合（可选）：以一定概率强制取 [1,0]/[0,1]
        if corner_prob > 0:
            mask = (torch.rand(batch_size, device=device) < corner_prob)
            if mask.any():
                # 随机分配到两个角点
                choose = torch.rand(batch_size, device=device) < 0.5
                pref_corner = pref.clone()
                pref_corner[choose] = torch.tensor([1.0 - eps, eps], device=device)
                pref_corner[~choose] = torch.tensor([eps, 1.0 - eps], device=device)
                pref = torch.where(mask.unsqueeze(1), pref_corner, pref)

        return pref

    return pref_sampler


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


def _make_mo_save_tag(config, lambda_cost: float) -> str:
    """
    Stable tag for checkpoints / npz naming.
    - single objective => '_single'
    - multi objective + random preference conditioning => '_prefRand'
    - otherwise => '_lc{lambda_cost}'
    """
    if not _use_multi_objective(config):
        return "_single"
    if _use_preference_conditioning(config) and getattr(config, "ngt_pref_sampling", "fixed") == "random":
        return "_prefRand"
    return f"_lc{lambda_cost:.1f}".replace(".", "")


class _ConcatPrefWrapper(nn.Module):
    """
    Wrap a NetV model that expects [x, pref] input, expose forward(x) only.
    Used to keep NGTPredictor/evaluate_unified unchanged.
    """
    def __init__(self, base_model: nn.Module, pref_base: torch.Tensor):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("_pref_base", pref_base)  # [1,2]

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        pref = self._pref_base.expand(B, -1).to(device=x.device, dtype=x.dtype)
        return self.base_model(torch.cat([x, pref], dim=1))


def _wrap_ngt_model_for_eval_if_needed(model: nn.Module, config, device, lambda_cost: float, lambda_carbon: float):
    """
    If NetV was trained with preference concatenation, wrap it so that
    NGTPredictor can still call model(x) (without needing code changes elsewhere).
    """
    use_pref = _use_multi_objective(config) and _use_preference_conditioning(config)
    if not use_pref:
        return model
    pref_base = _make_pref_base(device, lambda_cost, lambda_carbon)  # [1,2]
    return _ConcatPrefWrapper(model, pref_base)



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
    use_flow_model = getattr(config, 'ngt_use_flow_model', False)
    
    # Get lambda_cost for lc_str (used in save path)
    lambda_cost = getattr(config, 'ngt_lambda_cost', 0.9)
    lc_str = f"{lambda_cost:.1f}".replace('.', '')
    lambda_carbon = 1 - lambda_cost

    if tb_enabled and not debug:
        model_type_name = "flow" if use_flow_model else "mlp"
        log_comment = f"ngt_{model_type_name}_lc{lc_str}_{config.Nbus}bus"
        runs_dir = os.path.join(os.path.dirname(__file__), 'runs')
        os.makedirs(runs_dir, exist_ok=True)
        tb_logger = TensorBoardLogger(log_dir=runs_dir, comment=log_comment)
    
    if use_flow_model: 
        flow_inf_steps = getattr(config, 'ngt_flow_inf_steps', 10)
        use_projection = getattr(config, 'ngt_use_projection', False) 
        
        zero_init = os.environ.get('NGT_FLOW_ZERO_INIT', 'True').lower() == 'true' 
        results = train_unsupervised_ngt_gradient_descent(
            config=config, sys_data=sys_data, device=config.device,
            lambda_cost=lambda_cost,
            lambda_carbon=lambda_carbon,
            use_projection=False,  # 使用投影矩阵
            num_iterations=500,   # 迭代次数
            learning_rate=1e-5,   # 更小的学习率 (1e-5 to 1e-6 recommended)
            tb_logger=None,       # 可选：TensorBoard logger
            use_drift_correction=False,  # 启用drift correction
            lambda_cor=5.0,  # Drift correction gain
            grad_clip_norm=1.0,  # 梯度裁剪
            use_post_processing=True,  # 启用后处理
        )   
        loss_history = None
        # model_ngt, loss_history, time_train, ngt_data, sys_data, use_projection_train, P_tan_t_train = train_unsupervised_ngt_flow(
        #     config, sys_data, config.device,
        #     lambda_cost=lambda_cost,
        #     lambda_carbon=lambda_carbon,
        #     flow_inf_steps=flow_inf_steps,
        #     use_projection=use_projection, 
        #     tb_logger=tb_logger,
        #     zero_init=zero_init,
        #     debug=debug
        # ) 
    else:
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
    # if use_flow_model:
    #     # For Flow model, use Flow-specific evaluation with projection support
    #     from utils import get_genload
    #     # Prepare test data
    #     x_test = ngt_data['x_test'].to(config.device)
    #     Real_Vm = ngt_data['yvm_test'].numpy()
    #     Real_Va_full = ngt_data['yva_test'].numpy()
        
    #     baseMVA = float(sys_data.baseMVA)
    #     Pdtest = np.zeros((len(ngt_data['idx_test']), config.Nbus))
    #     Qdtest = np.zeros((len(ngt_data['idx_test']), config.Nbus))
    #     bus_Pd = ngt_data['bus_Pd']
    #     bus_Qd = ngt_data['bus_Qd']
    #     idx_test = ngt_data['idx_test']
    #     Pdtest[:, bus_Pd] = sys_data.RPd[idx_test][:, bus_Pd] / baseMVA
    #     Qdtest[:, bus_Qd] = sys_data.RQd[idx_test][:, bus_Qd] / baseMVA
        
    #     MAXMIN_Pg = ngt_data['MAXMIN_Pg']
    #     MAXMIN_Qg = ngt_data['MAXMIN_Qg']
    #     gencost = ngt_data['gencost_Pg']
        
    #     # Real cost
    #     Real_V = Real_Vm * np.exp(1j * Real_Va_full)
    #     Real_Pg, Real_Qg, _, _ = get_genload(
    #         Real_V, Pdtest, Qdtest,
    #         sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
    #     )
    #     Real_cost = gencost[:, 0] * (Real_Pg * baseMVA)**2 + gencost[:, 1] * np.abs(Real_Pg * baseMVA)
        
    #     # Preference tensor
    #     preference = torch.tensor([[lambda_cost, lambda_carbon]], dtype=torch.float32, device=config.device)
        
    #     # Load VAE models for anchor generation
    #     vae_vm_path = config.pretrain_model_path_vm
    #     vae_va_path = config.pretrain_model_path_va
    #     from models import create_model
    #     vae_vm = create_model('vae', ngt_data['input_dim'], config.Nbus, config, is_vm=True)
    #     vae_va = create_model('vae', ngt_data['input_dim'], config.Nbus - 1, config, is_vm=False)
    #     vae_vm.to(config.device)
    #     vae_va.to(config.device)
    #     vae_vm.load_state_dict(torch.load(vae_vm_path, map_location=config.device, weights_only=True), strict=False)
    #     vae_va.load_state_dict(torch.load(vae_va_path, map_location=config.device, weights_only=True), strict=False)
    #     vae_vm.eval()
    #     vae_va.eval() 
        
    #     # Build context for unified evaluation
    #     ctx = build_ctx_from_ngt(config, sys_data, ngt_data, BRANFT, config.device)
        
    #     # Create Flow predictor
    #     predictor = NGTFlowPredictor(
    #         model_flow=model_ngt,
    #         vae_vm=vae_vm,
    #         vae_va=vae_va,
    #         ngt_data=ngt_data,
    #         preference=preference,
    #         flow_forward_ngt=flow_forward_ngt,
    #         flow_forward_ngt_projected=flow_forward_ngt_projected if use_projection_train else None,
    #         use_projection=use_projection_train,
    #         P_tan_t=P_tan_t_train,
    #         flow_inf_steps=flow_inf_steps
    #     )
        
    #     eval_results_unified = evaluate_unified(
    #         ctx, predictor,
    #         apply_post_processing=True,
    #         verbose=True
    #     )  
    #     eval_results = eval_results_unified 
        
    # else: 
    #     # Build context for unified evaluation
    #     ctx = build_ctx_from_ngt(config, sys_data, ngt_data, BRANFT, config.device)
        
    #     # Create NGT predictor
    #     # [MO-PREF] if NetV expects [x,pref], wrap it so predictor can still call model(x)
    #     model_ngt_eval = _wrap_ngt_model_for_eval_if_needed(
    #         model_ngt, config, config.device, lambda_cost, lambda_carbon
    #     )
    #     predictor = NGTPredictor(model_ngt_eval)

        
    #     eval_results_unified = evaluate_unified(
    #         ctx, predictor,
    #         apply_post_processing=True,
    #         verbose=True
    #     ) 
    #     # Use unified results as primary (more consistent with supervised evaluation)
    #     eval_results = eval_results_unified
    
    # # Combine training and evaluation results
    # results = {
    #     'loss_history': loss_history,
    #     'time_train': time_train,
    #     'ngt_data': ngt_data,
    #     **eval_results  # Include all evaluation metrics
    # }
    
    # # Save training history (include lambda_cost and model type for identification)
    # model_type_str = "flow" if use_flow_model else "mlp" 
    
    # # Save evaluation results
    # eval_save_path = f'{config.model_save_dir}/ngt_{model_type_str}_eval_{config.Nbus}bus{lc_str}.npz'
    # eval_to_save = {k: v for k, v in eval_results.items() 
    #                 if isinstance(v, (int, float, np.ndarray))}
    # np.savez(eval_save_path, **eval_to_save) 


if __name__ == "__main__": 
    debug = bool(int(os.environ.get('DEBUG', '0')))
    main(debug=debug)