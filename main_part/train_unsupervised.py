#!/usr/bin/env python
# coding: utf-8
# Training Script for DeepOPF-V
# Extended to support multiple model types: VAE, Flow, Diffusion, GAN, etc.
# Author: Peng Yue
# Date: December 15th, 2025

import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np 
import time
import os
import sys 
import math

# Add parent directory to path for flow_model imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from models import NetV
from data_loader import load_all_data, load_ngt_training_data, create_ngt_training_loader
from utils import (TensorBoardLogger, initialize_flow_model_near_zero, plot_unsupervised_training_curves,
                   get_genload, get_vioPQg, get_viobran2)
from deepopf_ngt_loss import DeepOPFNGTLoss 
from unified_eval import build_ctx_from_ngt, NGTPredictor, NGTFlowPredictor, evaluate_unified


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
            running_loss_obj += loss_dict.get('loss_obj', 0.0)
            running_cost_per_mean += loss_dict.get('cost_per_mean', 0.0)
            running_carbon_per_mean += loss_dict.get('carbon_per_mean', 0.0)
            running_loss_Pgi_sum += loss_dict.get('loss_Pgi_sum', 0.0)
            running_loss_Qgi_sum += loss_dict.get('loss_Qgi_sum', 0.0)
            running_loss_Pdi_sum += loss_dict.get('loss_Pdi_sum', 0.0)
            running_loss_Qdi_sum += loss_dict.get('loss_Qdi_sum', 0.0)
            running_loss_Vi_sum += loss_dict.get('loss_Vi_sum', 0.0)
            running_ls_cost += loss_dict.get('ls_cost', 0.0)
            running_ls_Pg += loss_dict.get('ls_Pg', 0.0)
            running_ls_Qg += loss_dict.get('ls_Qg', 0.0)
            running_ls_Pd += loss_dict.get('ls_Pd', 0.0)
            running_ls_Qd += loss_dict.get('ls_Qd', 0.0)
            running_ls_V += loss_dict.get('ls_V', 0.0)
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


def train_unsupervised_ngt_flow(
    config, sys_data=None, device=None,
    lambda_cost=0.9, lambda_carbon=0.1,
    flow_inf_steps=10, use_projection=False,
    anchor_model_path=None, anchor_preference=None,
    tb_logger=None, zero_init=True, debug=False,
    debug_model_path=None,
):

    if device is None:
        device = config.device

    # Step 1: Load NGT data
    ngt_data, sys_data = load_ngt_training_data(config, sys_data)
    input_dim = ngt_data['input_dim']
    output_dim = ngt_data['output_dim']
    Vscale = ngt_data['Vscale'].to(device)
    Vbias = ngt_data['Vbias'].to(device)


    from models import create_model, PreferenceConditionedNetV

    bus_slack = int(sys_data.bus_slack)
    use_flow_anchor = anchor_model_path is not None

    # Helper: load & freeze VAE
    def _load_frozen_vae_or_raise():
        vae_vm_path = config.pretrain_model_path_vm
        vae_va_path = config.pretrain_model_path_va
        if not os.path.exists(vae_vm_path) or not os.path.exists(vae_va_path):
            raise FileNotFoundError(
                f"Pretrained VAE not found:\n"
                f"  Vm: {vae_vm_path}\n"
                f"  Va: {vae_va_path}\n"
                f"Please train VAE models first."
            )
        vae_vm = create_model('vae', input_dim, config.Nbus, config, is_vm=True).to(device)
        vae_va = create_model('vae', input_dim, config.Nbus - 1, config, is_vm=False).to(device)
        vae_vm.load_state_dict(torch.load(vae_vm_path, map_location=device, weights_only=True), strict=True)
        vae_va.load_state_dict(torch.load(vae_va_path, map_location=device, weights_only=True), strict=True)
        vae_vm.eval(); vae_va.eval()
        for p in vae_vm.parameters(): p.requires_grad = False
        for p in vae_va.parameters(): p.requires_grad = False
        return vae_vm, vae_va

    # Step 2: Anchor models
    anchor_flow_model = None
    anchor_pref_tensor = None
    vae_vm = None
    vae_va = None 

    vae_vm, vae_va = _load_frozen_vae_or_raise()

    # Step 3: Create Flow model
    hidden_dim = getattr(config, 'ngt_flow_hidden_dim', 144)
    num_layers = getattr(config, 'ngt_flow_num_layers', 2)

    model = PreferenceConditionedNetV(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        Vscale=Vscale,
        Vbias=Vbias,
        preference_dim=2,
        preference_hidden=64
    ).to(device)

    print(f"\nFlow model created: hidden_dim={hidden_dim}, num_layers={num_layers}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    if zero_init:
        initialize_flow_model_near_zero(model, scale=0.5)

    # Debug mode
    if debug:
        if debug_model_path is None:
            debug_model_path = f"{config.model_save_dir}/NetV_ngt_flow_{config.Nbus}bus_lc05_E1000_final.pth"
        if not os.path.exists(debug_model_path):
            raise FileNotFoundError(f"Debug model not found: {debug_model_path}")
        model.load_state_dict(torch.load(debug_model_path, map_location=device, weights_only=True))
        model.eval()

        P_tan_t = None
        if use_projection:
            P_tan_t = _setup_projection_matrix_for_flow(sys_data, config, ngt_data, device)
            if P_tan_t is None:
                use_projection = False

        loss_history = _init_loss_history()
        return model, loss_history, 0.0, ngt_data, sys_data, use_projection, P_tan_t

    # Step 4: optimizer with learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ngt_Lr)
    
    # Add learning rate scheduler to help escape local minima
    # Use CosineAnnealingLR with warm restarts for better convergence
    # T_0: initial restart period, T_mult: period multiplier
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=500, T_mult=2, eta_min=config.ngt_Lr * 0.01
    )

    # Step 5: loss (base λ for logs/baseline; may be overridden per-batch if pref random & loss doesn't accept preference)
    config.ngt_lambda_cost = lambda_cost
    config.ngt_lambda_carbon = lambda_carbon

    loss_fn = DeepOPFNGTLoss(sys_data, config)
    loss_fn.cache_to_gpu(device)

    # Step 6: indexed DataLoader
    class IndexedTensorDataset(Data.Dataset):
        def __init__(self, *tensors):
            assert all(t.size(0) == tensors[0].size(0) for t in tensors)
            self.tensors = tensors
        def __getitem__(self, index):
            return tuple(t[index] for t in self.tensors) + (index,)
        def __len__(self):
            return self.tensors[0].size(0)

    indexed_dataset = IndexedTensorDataset(ngt_data['x_train'], ngt_data['y_train'])
    training_loader = Data.DataLoader(indexed_dataset, batch_size=config.ngt_batch_size, shuffle=True)

    # Step 7: projection matrix (optional)
    P_tan_t = None
    if use_projection:
        P_tan_t = _setup_projection_matrix_for_flow(sys_data, config, ngt_data, device)
        if P_tan_t is None:
            use_projection = False

    # ===================== [MO-PREF] preference sampling =====================
    pref_sampler = build_preference_sampler(config, device, lambda_cost, lambda_carbon)
    use_pref = _use_multi_objective(config) and _use_preference_conditioning(config)
    pref_base = _make_pref_base(device, lambda_cost, lambda_carbon) if use_pref else None

    # Step 8: precompute anchors z_anchor_all
    z_anchor_all = _precompute_z_anchor_all(
        config=config,
        sys_data=sys_data,
        ngt_data=ngt_data,
        device=device,
        vae_vm=vae_vm,
        vae_va=vae_va,
        Vscale=Vscale,
        Vbias=Vbias,
        bus_slack=bus_slack,
        use_flow_anchor=use_flow_anchor,
        anchor_flow_model=anchor_flow_model,
        anchor_pref_tensor=anchor_pref_tensor,
        flow_inf_steps=flow_inf_steps,
    )

    # Step 9: optional VAE baseline (fixed preference base)
    vae_baseline_cost = 0.0
    vae_baseline_carbon = 0.0
    vae_baseline_weighted = 0.0
    carbon_scale = getattr(config, 'ngt_carbon_scale', 30.0)

    if tb_logger:
        with torch.no_grad():
            total_cost = 0.0
            total_carbon = 0.0
            n_samples = 0
            for train_x, train_y, batch_indices in training_loader:
                train_x = train_x.to(device)
                z_anchor_batch = z_anchor_all[batch_indices]
                V_anchor = torch.sigmoid(z_anchor_batch) * model.Vscale + model.Vbias

                # baseline uses fixed pref_base if conditioning enabled
                pref_batch = _expand_pref(pref_base, train_x.shape[0]) if use_pref else None
                _, ld = _call_ngt_loss(loss_fn, config, V_anchor, train_x, pref_batch)

                bs = len(batch_indices)
                total_cost += float(ld.get('loss_cost', 0.0)) * bs
                total_carbon += float(ld.get('loss_carbon', 0.0)) * bs
                n_samples += bs

            vae_baseline_cost = total_cost / max(n_samples, 1)
            vae_baseline_carbon = total_carbon / max(n_samples, 1)
            vae_baseline_weighted = lambda_cost * vae_baseline_cost + lambda_carbon * vae_baseline_carbon * carbon_scale

    # Step 10: strategy callbacks
    def forward_and_loss_fn(batch):
        train_x, train_y, batch_indices = batch
        train_x = train_x.to(device)
        z_anchor_batch = z_anchor_all[batch_indices]

        # [MO-PREF] random/fixed sampling (or None)
        pref_batch = pref_sampler(train_x.shape[0])

        V_pred = flow_forward_ngt_unified(
            flow_model=model,
            x=train_x,
            z_anchor=z_anchor_batch,
            preference=pref_batch,
            num_steps=flow_inf_steps,
            training=True,
            P_tan_t=P_tan_t if (use_projection and P_tan_t is not None) else None,
        )

        # [MO-PREF] aligned loss
        loss, loss_dict = _call_ngt_loss(loss_fn, config, V_pred, train_x, pref_batch)
        return loss, loss_dict, V_pred

    def sample_pred_fn():
        bs = config.ngt_batch_size
        sample_x = ngt_data['x_train'][:bs].to(device)
        sample_anchor = z_anchor_all[:bs]

        # for printing: use fixed preference base (stable)
        sample_pref = _expand_pref(pref_base, bs) if use_pref else None

        return flow_forward_ngt_unified(
            flow_model=model,
            x=sample_x,
            z_anchor=sample_anchor,
            preference=sample_pref,
            num_steps=flow_inf_steps,
            training=False,
            P_tan_t=P_tan_t if (use_projection and P_tan_t is not None) else None,
        )

    def extra_tb_logging(epoch, avg_cost, avg_carbon):
        weighted_obj = lambda_cost * avg_cost + lambda_carbon * avg_carbon * carbon_scale
        
        if tb_logger is not None:
            tb_logger.log_scalar('baseline/vae_cost', vae_baseline_cost, epoch)
            tb_logger.log_scalar('baseline/vae_carbon', vae_baseline_carbon, epoch)
            tb_logger.log_scalar('baseline/vae_weighted', vae_baseline_weighted, epoch)
            tb_logger.log_scalar('objective/weighted', weighted_obj, epoch)
            tb_logger.log_scalar('objective/cost', avg_cost, epoch)
            tb_logger.log_scalar('objective/carbon', avg_carbon, epoch)

    save_tag = _make_mo_save_tag(config, lambda_cost)

    core_kwargs = dict(
        config=config, model=model, loss_fn=loss_fn, optimizer=optimizer,
        training_loader=training_loader, device=device, ngt_data=ngt_data,
        sys_data=sys_data,
        forward_and_loss_fn=forward_and_loss_fn,
        sample_pred_fn=sample_pred_fn,
        tb_logger=tb_logger,
        print_prefix="[Flow] ",
        save_prefix="NetV_ngt_flow",
        save_tag=save_tag,
        extra_tb_logging_fn=extra_tb_logging,
        grad_clip_norm=5.0,  # Increased from 1.0 to allow larger gradient updates
        scheduler=scheduler,
    )

    loss_history, time_train = train_ngt_core(**core_kwargs)

    return model, loss_history, time_train, ngt_data, sys_data, use_projection, P_tan_t

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
    anchor_flow_model,
    anchor_pref_tensor,
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

    if not use_flow_anchor:
        return z_vae_anchor.detach()

    # ---- Step C: progressive anchor: push z_vae_anchor through anchor flow ----
    with torch.no_grad():
        anchor_pref_batch = anchor_pref_tensor.expand(len(idx_train), -1)
        # 这里沿用你原逻辑：用 anchor flow 产生更好的 latent anchor
        z_anchor_all = anchor_flow_model.flow_backward(
            x_train, z_vae_anchor, anchor_pref_batch,
            num_steps=flow_inf_steps, apply_sigmoid=False, training=False
        ).detach()

    return z_anchor_all


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


def _call_ngt_loss(loss_fn, config, V_pred, x_in, pref_batch=None):
    """
    Call DeepOPFNGTLoss with correct preference alignment.
    - If loss supports preference kwarg: pass pref_batch directly (batch or sample level).
    - Else: only allow pref_level='batch' and sync lambda to loss/config using pref_batch[0].
    """
    if pref_batch is None:
        return loss_fn(V_pred, x_in)

    if _loss_supports_preference(loss_fn):
        return loss_fn(V_pred, x_in, preference=pref_batch) 


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

    if tb_enabled and not debug:
        model_type_name = "flow" if use_flow_model else "mlp"
        log_comment = f"ngt_{model_type_name}_lc{lc_str}_{config.Nbus}bus"
        runs_dir = os.path.join(os.path.dirname(__file__), 'runs')
        os.makedirs(runs_dir, exist_ok=True)
        tb_logger = TensorBoardLogger(log_dir=runs_dir, comment=log_comment)
    
    if use_flow_model:
        lambda_carbon = getattr(config, 'ngt_lambda_carbon', 0.1)
        flow_inf_steps = getattr(config, 'ngt_flow_inf_steps', 10)
        use_projection = getattr(config, 'ngt_use_projection', False)
        
        # Check for progressive training (anchor from previous Flow model)
        anchor_model_path = os.environ.get('NGT_ANCHOR_MODEL_PATH', None)
        anchor_lambda_cost = os.environ.get('NGT_ANCHOR_LAMBDA_COST', None)
        
        if anchor_model_path and anchor_lambda_cost:
            anchor_preference = [float(anchor_lambda_cost), 1.0 - float(anchor_lambda_cost)]
        else: 
            anchor_preference = None
        
        zero_init = os.environ.get('NGT_FLOW_ZERO_INIT', 'True').lower() == 'true' 
        model_ngt, loss_history, time_train, ngt_data, sys_data, use_projection_train, P_tan_t_train = train_unsupervised_ngt_flow(
            config, sys_data, config.device,
            lambda_cost=lambda_cost,
            lambda_carbon=lambda_carbon,
            flow_inf_steps=flow_inf_steps,
            use_projection=use_projection,
            anchor_model_path=anchor_model_path,
            anchor_preference=anchor_preference,
            tb_logger=tb_logger,
            zero_init=zero_init,
            debug=debug
        ) 
    else:
        if not debug:
            lambda_carbon = getattr(config, 'ngt_lambda_carbon', 0.1)
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
    
    if use_flow_model:
        # For Flow model, use Flow-specific evaluation with projection support
        from utils import get_genload
        # Prepare test data
        x_test = ngt_data['x_test'].to(config.device)
        Real_Vm = ngt_data['yvm_test'].numpy()
        Real_Va_full = ngt_data['yva_test'].numpy()
        
        baseMVA = float(sys_data.baseMVA)
        Pdtest = np.zeros((len(ngt_data['idx_test']), config.Nbus))
        Qdtest = np.zeros((len(ngt_data['idx_test']), config.Nbus))
        bus_Pd = ngt_data['bus_Pd']
        bus_Qd = ngt_data['bus_Qd']
        idx_test = ngt_data['idx_test']
        Pdtest[:, bus_Pd] = sys_data.RPd[idx_test][:, bus_Pd] / baseMVA
        Qdtest[:, bus_Qd] = sys_data.RQd[idx_test][:, bus_Qd] / baseMVA
        
        MAXMIN_Pg = ngt_data['MAXMIN_Pg']
        MAXMIN_Qg = ngt_data['MAXMIN_Qg']
        gencost = ngt_data['gencost_Pg']
        
        # Real cost
        Real_V = Real_Vm * np.exp(1j * Real_Va_full)
        Real_Pg, Real_Qg, _, _ = get_genload(
            Real_V, Pdtest, Qdtest,
            sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
        )
        Real_cost = gencost[:, 0] * (Real_Pg * baseMVA)**2 + gencost[:, 1] * np.abs(Real_Pg * baseMVA)
        
        # Preference tensor
        preference = torch.tensor([[lambda_cost, lambda_carbon]], dtype=torch.float32, device=config.device)
        
        # Load VAE models for anchor generation
        vae_vm_path = config.pretrain_model_path_vm
        vae_va_path = config.pretrain_model_path_va
        from models import create_model
        vae_vm = create_model('vae', ngt_data['input_dim'], config.Nbus, config, is_vm=True)
        vae_va = create_model('vae', ngt_data['input_dim'], config.Nbus - 1, config, is_vm=False)
        vae_vm.to(config.device)
        vae_va.to(config.device)
        vae_vm.load_state_dict(torch.load(vae_vm_path, map_location=config.device, weights_only=True), strict=False)
        vae_va.load_state_dict(torch.load(vae_va_path, map_location=config.device, weights_only=True), strict=False)
        vae_vm.eval()
        vae_va.eval() 
        
        # Build context for unified evaluation
        ctx = build_ctx_from_ngt(config, sys_data, ngt_data, BRANFT, config.device)
        
        # Create Flow predictor
        predictor = NGTFlowPredictor(
            model_flow=model_ngt,
            vae_vm=vae_vm,
            vae_va=vae_va,
            ngt_data=ngt_data,
            preference=preference,
            flow_forward_ngt=flow_forward_ngt,
            flow_forward_ngt_projected=flow_forward_ngt_projected if use_projection_train else None,
            use_projection=use_projection_train,
            P_tan_t=P_tan_t_train,
            flow_inf_steps=flow_inf_steps
        )
        
        eval_results_unified = evaluate_unified(
            ctx, predictor,
            apply_post_processing=True,
            verbose=True
        )  
        eval_results = eval_results_unified 
        
    else: 
        # Build context for unified evaluation
        ctx = build_ctx_from_ngt(config, sys_data, ngt_data, BRANFT, config.device)
        
        # Create NGT predictor
        # [MO-PREF] if NetV expects [x,pref], wrap it so predictor can still call model(x)
        model_ngt_eval = _wrap_ngt_model_for_eval_if_needed(
            model_ngt, config, config.device, lambda_cost, lambda_carbon
        )
        predictor = NGTPredictor(model_ngt_eval)

        
        eval_results_unified = evaluate_unified(
            ctx, predictor,
            apply_post_processing=True,
            verbose=True
        ) 
        # Use unified results as primary (more consistent with supervised evaluation)
        eval_results = eval_results_unified
    
    # Combine training and evaluation results
    results = {
        'loss_history': loss_history,
        'time_train': time_train,
        'ngt_data': ngt_data,
        **eval_results  # Include all evaluation metrics
    }
    
    # Save training history (include lambda_cost and model type for identification)
    model_type_str = "flow" if use_flow_model else "mlp" 
    
    # Save evaluation results
    eval_save_path = f'{config.model_save_dir}/ngt_{model_type_str}_eval_{config.Nbus}bus{lc_str}.npz'
    eval_to_save = {k: v for k, v in eval_results.items() 
                    if isinstance(v, (int, float, np.ndarray))}
    np.savez(eval_save_path, **eval_to_save)
    
    return model_ngt, results 


if __name__ == "__main__": 
    debug = bool(int(os.environ.get('DEBUG', '0')))
    model_ngt, results = main(debug=debug)