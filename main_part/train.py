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
from models import NetV, get_available_model_types
from data_loader import load_all_data, load_ngt_training_data, create_ngt_training_loader
from utils import (TensorBoardLogger, initialize_flow_model_near_zero,
                   save_results, plot_training_curves, plot_unsupervised_training_curves,
                   get_genload, get_vioPQg, get_viobran2)
from deepopf_ngt_loss import DeepOPFNGTLoss 
from unified_eval import (
    build_ctx_from_supervised, build_ctx_from_ngt,
    SupervisedPredictor, NGTPredictor, NGTFlowPredictor,
    evaluate_unified
) 


def train_voltage_magnitude(config, model_vm, optimizer_vm, training_loader_vm, sys_data, criterion, device,
                            model_type='simple', pretrain_model=None, scheduler=None):
    """
    Train voltage magnitude prediction model with support for multiple model types
    
    Args:
        config: Configuration object
        model_vm: Voltage magnitude model
        optimizer_vm: Optimizer
        training_loader_vm: Training data loader
        sys_data: System data
        criterion: Loss function
        device: Device (CPU/GPU)
        model_type: Type of model ('simple', 'vae', 'rectified', 'diffusion', 'gan', 'wgan', etc.)
        pretrain_model: Pretrained VAE model for flow models (required for 'rectified')
        scheduler: Learning rate scheduler (optional)
        
    Returns:
        model_vm: Trained model
        lossvm: Training losses
        time_train: Training time
    """
    print('=' * 60)
    print(f'Training Voltage Magnitude (Vm) Model - Type: {model_type}')
    print('=' * 60)
    
    lossvm = []
    start_time = time.process_time()
    
    # Get VAE beta from config
    vae_beta = getattr(config, 'vae_beta', 1.0)
    
    for epoch in range(config.EpochVm):
        running_loss = 0.0
        model_vm.train()
        
        for step, (train_x, train_y) in enumerate(training_loader_vm):
            train_x, train_y = train_x.to(device), train_y.to(device)
            batch_dim = train_x.shape[0]
            
            optimizer_vm.zero_grad()
            
            # ==================== Model-specific training logic ====================
            if model_type == 'simple':
                # Original MLP supervised training
                yvmtrain_hat = model_vm(train_x)
                loss = criterion(train_y, yvmtrain_hat)
                
            elif model_type == 'vae':
                # VAE training: reconstruction loss + KL divergence
                # 传入 train_y 让 Encoder 同时看到条件 x 和目标 y
                y_pred, mean, logvar = model_vm.encoder_decode(train_x, train_y)
                loss = model_vm.loss(y_pred, train_y, mean, logvar, beta=vae_beta)
                yvmtrain_hat = y_pred
                
            elif model_type in ['rectified', 'gaussian', 'conditional', 'interpolation']:
                # Flow Matching training
                t_batch = torch.rand([batch_dim, 1]).to(device)
                
                if model_type == 'rectified' and pretrain_model is not None:
                    # Use VAE to generate anchor points
                    with torch.no_grad():
                        z_batch = pretrain_model(train_x, use_mean=True)
                else:
                    # Use random noise as starting point
                    z_batch = torch.randn_like(train_y).to(device)
                
                # Flow forward: get interpolation point and target velocity
                yt, vec_target = model_vm.flow_forward(train_y, t_batch, z_batch, model_type)
                
                # Predict velocity
                vec_pred = model_vm.predict_vec(train_x, yt, t_batch)
                
                # Calculate loss
                loss = model_vm.loss(train_y, z_batch, vec_pred, vec_target, model_type)
                yvmtrain_hat = vec_pred + z_batch  # Approximate prediction
                
            elif model_type == 'diffusion':
                # Diffusion model training with optional VAE anchor
                t_batch = torch.rand([batch_dim, 1]).to(device)
                noise = torch.randn_like(train_y).to(device)
                
                # Check if using VAE anchor for diffusion
                use_vae_anchor = getattr(config, 'use_vae_anchor', False)
                
                if use_vae_anchor and pretrain_model is not None:
                    # Use VAE to generate anchor points as starting distribution
                    # This modifies the diffusion process to start from VAE prediction + noise
                    with torch.no_grad():
                        vae_anchor = pretrain_model(train_x, use_mean=True)
                    # The diffusion starts from VAE prediction, target is train_y
                    # Modified forward: y_t = sqrt(alpha_t) * train_y + sqrt(1-alpha_t) * (noise + vae_residual)
                    # where vae_residual = train_y - vae_anchor represents what VAE missed
                    noise_pred = model_vm.predict_noise_with_anchor(train_x, train_y, t_batch, noise, vae_anchor)
                else:
                    # Standard diffusion: pure Gaussian noise
                    noise_pred = model_vm.predict_noise(train_x, train_y, t_batch, noise)
                
                loss = model_vm.loss(noise_pred, noise)
                yvmtrain_hat = train_y  # For display purposes
                
            elif model_type in ['gan', 'wgan']:
                # GAN/WGAN training
                z_batch = torch.randn([batch_dim, config.latent_dim]).to(device)
                y_pred = model_vm(train_x, z_batch)
                
                # Discriminator loss
                loss_d = model_vm.loss_d(train_x, train_y, y_pred)
                
                # Generator loss (update less frequently)
                if step % 5 == 0:
                    loss_g = model_vm.loss_g(train_x, y_pred)
                    loss = loss_d + loss_g
                else:
                    loss = loss_d
                    
                yvmtrain_hat = y_pred
                
            elif model_type == 'consistency_training':
                # Consistency model training
                z_batch = torch.randn_like(train_y).to(device)
                N = math.ceil(math.sqrt((epoch * (1000**2 - 4) / config.EpochVm) + 4) - 1) + 1
                boundaries = model_vm.kerras_boundaries(1.0, 0.002, N, 1).to(device)
                t_idx = torch.randint(0, N - 1, (batch_dim, 1), device=device)
                t_1 = boundaries[t_idx + 1]
                t_2 = boundaries[t_idx]
                
                # Need a vector model for consistency training (use self if not provided)
                if not hasattr(model_vm, 'target_model'):
                    model_vm.target_model = model_vm
                    
                loss = model_vm.loss(train_x, train_y, z_batch, t_1, t_2, sys_data, model_vm)
                yvmtrain_hat = train_y  # For display
                
            elif model_type == 'consistency_distillation':
                # Consistency distillation training
                z_batch = torch.randn_like(train_y).to(device)
                forward_step = 10
                N = math.ceil(1000 * (epoch / config.EpochVm) + 4) + forward_step
                boundaries = torch.linspace(0, 1 - 1e-3, N).to(device)
                t_idx = torch.randint(0, N - forward_step, (batch_dim, 1), device=device)
                t_1 = boundaries[t_idx]
                
                # Need a pretrained flow model
                if pretrain_model is None:
                    raise ValueError("Consistency distillation requires a pretrained flow model")
                    
                loss = model_vm.loss(train_x, train_y, z_batch, t_1, 1/N, forward_step, sys_data, pretrain_model)
                yvmtrain_hat = train_y  # For display
                
            else:
                raise NotImplementedError(f"Model type '{model_type}' not implemented")
            
            # Backward pass
            running_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_vm.parameters(), max_norm=1.0)
            optimizer_vm.step()

        lossvm.append(running_loss)
        
        # Learning rate scheduler step (per-epoch, not per-batch)
        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % config.p_epoch == 0:
            if hasattr(yvmtrain_hat, 'detach'):
                print(f'Epoch {epoch+1}: Loss = {running_loss:.6f}, '
                      f'Output range: [{torch.min(yvmtrain_hat).detach():.6f}, '
                      f'{torch.max(yvmtrain_hat).detach():.6f}]')
            else:
                print(f'Epoch {epoch+1}: Loss = {running_loss:.6f}')
         
        # Save trained model periodically
        if (epoch + 1) % 100 == 0 and (epoch + 1) >= config.s_epoch:
            save_path = f'{config.PATHVms}_{model_type}_E{epoch+1}F{config.flagVm}.pth'
            torch.save(model_vm.state_dict(), save_path, _use_new_zipfile_serialization=False)
            print(f'  Model saved: {save_path}')
            
    time_train = time.process_time() - start_time  
    print(f'\nVm training completed in {time_train:.2f} seconds ({time_train/60:.2f} minutes)')
    
    # Save final model
    final_path = f'{config.PATHVm[:-4]}_{model_type}.pth'
    torch.save(model_vm.state_dict(), final_path, _use_new_zipfile_serialization=False)
    print(f'Final model saved: {final_path}')
    
    return model_vm, lossvm, time_train


def train_voltage_angle(config, model_va, optimizer_va, training_loader_va, criterion, device,
                        model_type='simple', pretrain_model=None, scheduler=None):
    """
    Train voltage angle prediction model with support for multiple model types
    
    Args:
        config: Configuration object
        model_va: Voltage angle model
        optimizer_va: Optimizer
        training_loader_va: Training data loader
        criterion: Loss function
        device: Device (CPU/GPU)
        model_type: Type of model ('simple', 'vae', 'rectified', 'diffusion', 'gan', 'wgan', etc.)
        pretrain_model: Pretrained VAE model for flow models (required for 'rectified')
        scheduler: Learning rate scheduler (optional)
        
    Returns:
        model_va: Trained model
        lossva: Training losses
        time_train: Training time
    """
    print('\n' + '=' * 60)
    print(f'Training Voltage Angle (Va) Model - Type: {model_type}')
    print('=' * 60)
    
    lossva = []
    start_time = time.process_time()
    
    # Get VAE beta from config
    vae_beta = getattr(config, 'vae_beta', 1.0)
    
    for epoch in range(config.EpochVa):
        running_loss = 0.0
        model_va.train()
        
        for step, (train_x, train_y) in enumerate(training_loader_va):
            train_x, train_y = train_x.to(device), train_y.to(device)
            batch_dim = train_x.shape[0]
            
            optimizer_va.zero_grad()
            
            # ==================== Model-specific training logic ====================
            if model_type == 'simple':
                # Original MLP supervised training
                yvatrain_hat = model_va(train_x)
                loss = criterion(train_y, yvatrain_hat)
                
            elif model_type == 'vae':
                # VAE training: reconstruction loss + KL divergence
                # 传入 train_y 让 Encoder 同时看到条件 x 和目标 y
                y_pred, mean, logvar = model_va.encoder_decode(train_x, train_y)
                loss = model_va.loss(y_pred, train_y, mean, logvar, beta=vae_beta)
                yvatrain_hat = y_pred
                
            elif model_type in ['rectified', 'gaussian', 'conditional', 'interpolation']:
                # Flow Matching training
                t_batch = torch.rand([batch_dim, 1]).to(device)
                
                if model_type == 'rectified' and pretrain_model is not None:
                    # Use VAE to generate anchor points
                    with torch.no_grad():
                        z_batch = pretrain_model(train_x, use_mean=True)
                else:
                    # Use random noise as starting point
                    z_batch = torch.randn_like(train_y).to(device)
                
                # Flow forward: get interpolation point and target velocity
                yt, vec_target = model_va.flow_forward(train_y, t_batch, z_batch, model_type)
                
                # Predict velocity
                vec_pred = model_va.predict_vec(train_x, yt, t_batch)
                
                # Calculate loss
                loss = model_va.loss(train_y, z_batch, vec_pred, vec_target, model_type)
                yvatrain_hat = vec_pred + z_batch  # Approximate prediction
                
            elif model_type == 'diffusion':
                # Diffusion model training with optional VAE anchor
                t_batch = torch.rand([batch_dim, 1]).to(device)
                noise = torch.randn_like(train_y).to(device)
                
                # Check if using VAE anchor for diffusion
                use_vae_anchor = getattr(config, 'use_vae_anchor', False)
                
                if use_vae_anchor and pretrain_model is not None:
                    # Use VAE to generate anchor points as starting distribution
                    with torch.no_grad():
                        vae_anchor = pretrain_model(train_x, use_mean=True)
                    # Modified diffusion with VAE anchor
                    noise_pred = model_va.predict_noise_with_anchor(train_x, train_y, t_batch, noise, vae_anchor)
                else:
                    # Standard diffusion: pure Gaussian noise
                    noise_pred = model_va.predict_noise(train_x, train_y, t_batch, noise)
                
                loss = model_va.loss(noise_pred, noise)
                yvatrain_hat = train_y  # For display purposes
                
            elif model_type in ['gan', 'wgan']:
                # GAN/WGAN training
                z_batch = torch.randn([batch_dim, config.latent_dim]).to(device)
                y_pred = model_va(train_x, z_batch)
                
                # Discriminator loss
                loss_d = model_va.loss_d(train_x, train_y, y_pred)
                
                # Generator loss (update less frequently)
                if step % 5 == 0:
                    loss_g = model_va.loss_g(train_x, y_pred)
                    loss = loss_d + loss_g
                else:
                    loss = loss_d
                    
                yvatrain_hat = y_pred
                
            elif model_type == 'consistency_training':
                # Consistency model training
                z_batch = torch.randn_like(train_y).to(device)
                N = math.ceil(math.sqrt((epoch * (1000**2 - 4) / config.EpochVa) + 4) - 1) + 1
                boundaries = model_va.kerras_boundaries(1.0, 0.002, N, 1).to(device)
                t_idx = torch.randint(0, N - 1, (batch_dim, 1), device=device)
                t_1 = boundaries[t_idx + 1]
                t_2 = boundaries[t_idx]
                
                # Need a vector model for consistency training (use self if not provided)
                if not hasattr(model_va, 'target_model'):
                    model_va.target_model = model_va
                    
                loss = model_va.loss(train_x, train_y, z_batch, t_1, t_2, None, model_va)
                yvatrain_hat = train_y  # For display
                
            elif model_type == 'consistency_distillation':
                # Consistency distillation training
                z_batch = torch.randn_like(train_y).to(device)
                forward_step = 10
                N = math.ceil(1000 * (epoch / config.EpochVa) + 4) + forward_step
                boundaries = torch.linspace(0, 1 - 1e-3, N).to(device)
                t_idx = torch.randint(0, N - forward_step, (batch_dim, 1), device=device)
                t_1 = boundaries[t_idx]
                
                # Need a pretrained flow model
                if pretrain_model is None:
                    raise ValueError("Consistency distillation requires a pretrained flow model")
                    
                loss = model_va.loss(train_x, train_y, z_batch, t_1, 1/N, forward_step, None, pretrain_model)
                yvatrain_hat = train_y  # For display
                
            else:
                raise NotImplementedError(f"Model type '{model_type}' not implemented")
            
            # Backward pass
            running_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_va.parameters(), max_norm=1.0)
            optimizer_va.step()

        lossva.append(running_loss)
        
        # Learning rate scheduler step (per-epoch, not per-batch)
        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % config.p_epoch == 0:
            if hasattr(yvatrain_hat, 'detach'):
                print(f'Epoch {epoch+1}: Loss = {running_loss:.6f}, '
                      f'Output range: [{torch.min(yvatrain_hat).detach():.6f}, '
                      f'{torch.max(yvatrain_hat).detach():.6f}]')
            else:
                print(f'Epoch {epoch+1}: Loss = {running_loss:.6f}')

        # Save trained model periodically
        if (epoch + 1) % 100 == 0 and (epoch + 1) >= config.s_epoch:
            save_path = f'{config.PATHVas}_{model_type}_E{epoch+1}F{config.flagVa}.pth'
            torch.save(model_va.state_dict(), save_path, _use_new_zipfile_serialization=False)
            print(f'  Model saved: {save_path}')

    time_train = time.process_time() - start_time  
    
    print(f'\nVa training completed in {time_train:.2f} seconds ({time_train/60:.2f} minutes)')
    
    # Save final model
    final_path = f'{config.PATHVa[:-4]}_{model_type}.pth'
    torch.save(model_va.state_dict(), final_path, _use_new_zipfile_serialization=False)
    print(f'Final model saved: {final_path}')
    
    return model_va, lossva, time_train

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
    return flow_forward_ngt_unified(
        flow_model=flow_model,
        x=x,
        z_anchor=z_anchor,
        preference=preference,
        num_steps=num_steps,
        training=training,
        P_tan_t=None,
    )


def flow_forward_ngt_projected(flow_model, x, z_anchor, P_tan_t, preference, num_steps=10, training=True):
    return flow_forward_ngt_unified(
        flow_model=flow_model,
        x=x,
        z_anchor=z_anchor,
        preference=preference,
        num_steps=num_steps,
        training=training,
        P_tan_t=P_tan_t,
    )


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
            
            # Calculate gradient norm BEFORE clipping and step (for diagnostics)
            if extra_tb_logging_fn is not None:
                total_grad_norm = 0.0
                param_count = 0
                for param in model.parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.norm().item() ** 2
                        param_count += 1
                if param_count > 0:
                    grad_norm = (total_grad_norm ** 0.5) / param_count
                    epoch_grad_norms.append(grad_norm)
            
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

        # tensorboard (通用)
        if tb_logger:
            tb_logger.log_scalar('loss/total', avg_loss, epoch)
            tb_logger.log_scalar('loss/cost', avg_cost, epoch)
            tb_logger.log_scalar('loss/carbon', avg_carbon, epoch)
            tb_logger.log_scalar('weights/kgenp', avg_kgenp, epoch)
            tb_logger.log_scalar('weights/kgenq', avg_kgenq, epoch)
            tb_logger.log_scalar('weights/kpd', avg_kpd, epoch)
            tb_logger.log_scalar('weights/kqd', avg_kqd, epoch)
            tb_logger.log_scalar('weights/kv', avg_kv, epoch)
            
            # Log EMA scales if available (for multi-objective normalization)
            if hasattr(loss_fn, 'params') and hasattr(loss_fn.params, '_ema_cost'):
                if loss_fn.params._ema_cost is not None:
                    tb_logger.log_scalar('normalization/ema_cost_scale', loss_fn.params._ema_cost, epoch)
                if hasattr(loss_fn.params, '_ema_carbon_scaled') and loss_fn.params._ema_carbon_scaled is not None:
                    tb_logger.log_scalar('normalization/ema_carbon_scale', loss_fn.params._ema_carbon_scaled, epoch)
            
            # Log constraint satisfaction rates
            try:
                with torch.no_grad():
                    sample_pred = sample_pred_fn()
                    sample_x = ngt_data['x_train'][:config.ngt_batch_size].to(device)
                    constraint_stats = _compute_constraint_satisfaction(
                        sample_pred, sample_x, ngt_data, sys_data, config, device
                    )
                    tb_logger.log_scalar('constraint/Pg_satisfy', constraint_stats['Pg_satisfy'], epoch)
                    tb_logger.log_scalar('constraint/Qg_satisfy', constraint_stats['Qg_satisfy'], epoch)
                    tb_logger.log_scalar('constraint/Vm_satisfy', constraint_stats['Vm_satisfy'], epoch)
                    if constraint_stats['branch_ang_satisfy'] < 100.0:
                        tb_logger.log_scalar('constraint/branch_ang_satisfy', constraint_stats['branch_ang_satisfy'], epoch)
                        tb_logger.log_scalar('constraint/branch_pf_satisfy', constraint_stats['branch_pf_satisfy'], epoch)
            except Exception as e:
                # If constraint calculation fails, skip it
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
    print('=' * 60)
    print('DeepOPF-NGT Unsupervised Training (Reference Implementation)')
    print('=' * 60)

    if device is None:
        device = config.device

    # Step 1: Load NGT-specific training data
    ngt_data, sys_data = load_ngt_training_data(config, sys_data)

    input_dim = ngt_data['input_dim']
    output_dim = ngt_data['output_dim']
    Vscale = ngt_data['Vscale'].to(device)
    Vbias = ngt_data['Vbias'].to(device)

    print(f"\nModel configuration:")
    print(f"  Input dim: {input_dim}")
    print(f"  Output dim: {output_dim} (Va: {ngt_data['NPred_Va']}, Vm: {ngt_data['NPred_Vm']})")
    print(f"  Hidden layers: {config.ngt_khidden}")
    print(f"  Epochs: {config.ngt_Epoch}")
    print(f"  Batch size: {config.ngt_batch_size}")
    print(f"  Learning rate: {config.ngt_Lr}")

    # [MO-PREF] preference sampler (returns None if not (MO & conditioning))
    pref_sampler = build_preference_sampler(config, device, lambda_cost, lambda_carbon)
    use_pref = _use_multi_objective(config) and _use_preference_conditioning(config)
    pref_dim = 2
    netv_input_dim = input_dim + (pref_dim if use_pref else 0)

    # Step 2: Create NetV model
    model = NetV(
        input_channels=netv_input_dim,
        output_channels=output_dim,
        hidden_units=config.ngt_hidden_units,
        khidden=config.ngt_khidden,
        Vscale=Vscale,
        Vbias=Vbias
    ).to(device)

    pref_base = _make_pref_base(device, lambda_cost, lambda_carbon) if use_pref else None

    print(f"\nNetV model created:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Preference conditioning: {use_pref} (multi_objective={_use_multi_objective(config)})")
    if use_pref:
        print(f"  Pref sampling: {getattr(config,'ngt_pref_sampling','fixed')}, "
              f"level={getattr(config,'ngt_pref_level','batch')}, "
              f"method={getattr(config,'ngt_pref_method','dirichlet')}")

    # Step 3: optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ngt_Lr)

    # Step 4: loss
    config.ngt_lambda_cost = lambda_cost
    config.ngt_lambda_carbon = lambda_carbon
    loss_fn = DeepOPFNGTLoss(sys_data, config)
    loss_fn.cache_to_gpu(device)

    # Step 5: DataLoader
    training_loader = create_ngt_training_loader(ngt_data, config)

    def forward_and_loss_fn(batch):
        train_x, train_y = batch
        train_x = train_x.to(device)

        pref_batch = pref_sampler(train_x.shape[0])  # None or [B,2]

        if pref_batch is not None:
            model_in = torch.cat([train_x, pref_batch], dim=1)
        else:
            model_in = train_x

        V_pred = model(model_in)

        # [MO-PREF] loss aligned with preference
        loss, loss_dict = _call_ngt_loss(loss_fn, config, V_pred, train_x, pref_batch)
        return loss, loss_dict, V_pred

    def sample_pred_fn():
        sample_x = ngt_data['x_train'][:config.ngt_batch_size].to(device)
        if use_pref:
            pref_batch = _expand_pref(pref_base, sample_x.shape[0])
            sample_in = torch.cat([sample_x, pref_batch], dim=1)
        else:
            sample_in = sample_x
        return model(sample_in)

    save_tag = _make_mo_save_tag(config, lambda_cost)

    loss_history, time_train = train_ngt_core(
        config=config, model=model, loss_fn=loss_fn, optimizer=optimizer,
        training_loader=training_loader, device=device, ngt_data=ngt_data,
        sys_data=sys_data,
        forward_and_loss_fn=forward_and_loss_fn,
        sample_pred_fn=sample_pred_fn,
        tb_logger=tb_logger,
        print_prefix="",
        save_prefix="NetV_ngt",
        save_tag=save_tag,
    )

    return model, loss_history, time_train, ngt_data, sys_data


def train_unsupervised_ngt_flow(
    config, sys_data=None, device=None,
    lambda_cost=0.9, lambda_carbon=0.1,
    flow_inf_steps=10, use_projection=False,
    anchor_model_path=None, anchor_preference=None,
    tb_logger=None, zero_init=True, debug=False,
    debug_model_path=None,
):
    print('=' * 70)
    print('DeepOPF-NGT Unsupervised Training with Rectified Flow Model')
    print('=' * 70)
    print(f'Preference base: lambda_cost={lambda_cost}, lambda_carbon={lambda_carbon}')
    print(f'Flow steps: {flow_inf_steps}, Projection: {use_projection}')

    if device is None:
        device = config.device

    # Step 1: Load NGT data
    ngt_data, sys_data = load_ngt_training_data(config, sys_data)
    input_dim = ngt_data['input_dim']
    output_dim = ngt_data['output_dim']
    Vscale = ngt_data['Vscale'].to(device)
    Vbias = ngt_data['Vbias'].to(device)

    print(f"\nModel configuration:")
    print(f"  Input dim: {input_dim}")
    print(f"  Output dim: {output_dim} (Va: {ngt_data['NPred_Va']}, Vm: {ngt_data['NPred_Vm']})")
    print(f"  Epochs: {config.ngt_Epoch}")
    print(f"  Batch size: {config.ngt_batch_size}")
    print(f"  Learning rate: {config.ngt_Lr}")

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
        print(f"  VAE models loaded and frozen.")
        return vae_vm, vae_va

    # Step 2: Anchor models
    anchor_flow_model = None
    anchor_pref_tensor = None
    vae_vm = None
    vae_va = None

    if use_flow_anchor:
        print("\n--- Loading Flow Anchor Model (Progressive Training) ---")
        if anchor_preference is None:
            raise ValueError("anchor_preference must be provided when using anchor_model_path")
        if not os.path.exists(anchor_model_path):
            raise FileNotFoundError(f"Anchor Flow model not found: {anchor_model_path}")

        hidden_dim = getattr(config, 'ngt_flow_hidden_dim', 144)
        num_layers = getattr(config, 'ngt_flow_num_layers', 2)

        anchor_flow_model = PreferenceConditionedNetV(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            Vscale=Vscale,
            Vbias=Vbias,
            preference_dim=2,
            preference_hidden=64
        ).to(device)

        anchor_flow_model.load_state_dict(torch.load(anchor_model_path, map_location=device, weights_only=True))
        anchor_flow_model.eval()
        for p in anchor_flow_model.parameters():
            p.requires_grad = False

        anchor_pref_tensor = torch.tensor([anchor_preference], dtype=torch.float32, device=device)
        print(f"  Anchor Flow model loaded and frozen: {anchor_model_path}")

        vae_vm, vae_va = _load_frozen_vae_or_raise()
    else:
        print("\n--- Loading VAE Anchor Models ---")
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
        print("  Flow model output layer initialized near zero (scale=0.5)")

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
    print(f"  Learning rate scheduler: CosineAnnealingWarmRestarts (T_0=500, T_mult=2, eta_min={config.ngt_Lr * 0.01:.2e})")

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
            print("[Warning] Projection setup failed, disabling projection")
            use_projection = False

    # ===================== [MO-PREF] preference sampling =====================
    pref_sampler = build_preference_sampler(config, device, lambda_cost, lambda_carbon)
    use_pref = _use_multi_objective(config) and _use_preference_conditioning(config)
    pref_base = _make_pref_base(device, lambda_cost, lambda_carbon) if use_pref else None

    print(f"  Preference conditioning: {use_pref} (multi_objective={_use_multi_objective(config)})")
    if use_pref:
        print(f"  Pref sampling: {getattr(config,'ngt_pref_sampling','fixed')}, "
              f"level={getattr(config,'ngt_pref_level','batch')}, "
              f"method={getattr(config,'ngt_pref_method','dirichlet')}")

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
        
        # Log to TensorBoard if available
        if tb_logger is not None:
            tb_logger.log_scalar('baseline/vae_cost', vae_baseline_cost, epoch)
            tb_logger.log_scalar('baseline/vae_carbon', vae_baseline_carbon, epoch)
            tb_logger.log_scalar('baseline/vae_weighted', vae_baseline_weighted, epoch)
            tb_logger.log_scalar('objective/weighted', weighted_obj, epoch)
            tb_logger.log_scalar('objective/cost', avg_cost, epoch)
            tb_logger.log_scalar('objective/carbon', avg_carbon, epoch)
        
        # ========== Flow Velocity Diagnostics ==========
        # This helps diagnose why flow model might not be moving
        # Always compute diagnostics (even without TensorBoard) for logging
        with torch.no_grad():
            bs = min(config.ngt_batch_size, len(ngt_data['x_train']))
            sample_x = ngt_data['x_train'][:bs].to(device)
            sample_anchor = z_anchor_all[:bs]
            sample_pref = _expand_pref(pref_base, bs) if use_pref else None
            
            # Get flow prediction
            sample_pred = flow_forward_ngt_unified(
                flow_model=model,
                x=sample_x,
                z_anchor=sample_anchor,
                preference=sample_pref,
                num_steps=flow_inf_steps,
                training=False,
                P_tan_t=P_tan_t if (use_projection and P_tan_t is not None) else None,
            )
            
            # Convert anchor to physical space for comparison
            sample_anchor_physical = torch.sigmoid(sample_anchor) * model.Vscale + model.Vbias
            
            # Compute delta: how much did flow move from anchor?
            delta = (sample_pred - sample_anchor_physical).abs()
            delta_mean = delta.mean().item()
            delta_max = delta.max().item()
            delta_norm = delta.norm(dim=1).mean().item()
            
            # Velocity at anchor (t=0): what velocity does model predict at starting point?
            t_zero = torch.zeros(bs, 1, device=device, dtype=sample_anchor.dtype)
            velocity_at_anchor = model.predict_velocity(sample_x, sample_anchor, t_zero, sample_pref)
            velocity_norm = velocity_at_anchor.norm(dim=1).mean().item()
            velocity_mean_abs = velocity_at_anchor.abs().mean().item()
            velocity_max_abs = velocity_at_anchor.abs().max().item()
            
            # Velocity at midpoint (t=0.5): check if velocity changes during integration
            z_mid = sample_anchor + 0.5 * velocity_at_anchor * (1.0 / flow_inf_steps)
            t_mid = torch.full((bs, 1), 0.5, device=device, dtype=sample_anchor.dtype)
            velocity_at_mid = model.predict_velocity(sample_x, z_mid, t_mid, sample_pref)
            velocity_mid_norm = velocity_at_mid.norm(dim=1).mean().item()
            
            # Log to TensorBoard if available
            if tb_logger is not None:
                tb_logger.log_scalar('flow/delta_from_anchor_mean', delta_mean, epoch)
                tb_logger.log_scalar('flow/delta_from_anchor_max', delta_max, epoch)
                tb_logger.log_scalar('flow/delta_from_anchor_norm', delta_norm, epoch)
                tb_logger.log_scalar('flow/velocity_at_anchor_norm', velocity_norm, epoch)
                tb_logger.log_scalar('flow/velocity_at_anchor_mean_abs', velocity_mean_abs, epoch)
                tb_logger.log_scalar('flow/velocity_at_anchor_max_abs', velocity_max_abs, epoch)
                tb_logger.log_scalar('flow/velocity_at_mid_norm', velocity_mid_norm, epoch)
            
            # Gradient diagnostics: use pre-computed gradient norms from training loop
            grad_norm_avg = None
            if hasattr(model, '_last_grad_norm'):
                grad_norm_avg = model._last_grad_norm
            
            if tb_logger is not None and grad_norm_avg is not None:
                tb_logger.log_scalar('flow/grad_norm_avg', grad_norm_avg, epoch)
            
            # Print to console/log file for analysis (always print, even without TensorBoard)
            print(f"[Flow] Velocity diagnostics (epoch {epoch+1}):")
            print(f"  Delta from anchor: mean={delta_mean:.6f}, max={delta_max:.6f}, norm={delta_norm:.6f}")
            print(f"  Velocity at anchor: norm={velocity_norm:.6f}, mean_abs={velocity_mean_abs:.6f}, max_abs={velocity_max_abs:.6f}")
            print(f"  Velocity at mid (t=0.5): norm={velocity_mid_norm:.6f}")
            if grad_norm_avg is not None:
                print(f"  Gradient norm (avg): {grad_norm_avg:.6e}")
            else:
                print(f"  Gradient norm (avg): N/A (not computed)")
            print(f"  [Flow] Cost: {avg_cost:.2f}, Carbon: {avg_carbon:.2f}, Weighted: {weighted_obj:.2f}")
            print(f"  [Flow] VAE baseline - Cost: {vae_baseline_cost:.2f}, Carbon: {vae_baseline_carbon:.2f}, Weighted: {vae_baseline_weighted:.2f}")
            
            # Compute and print constraint satisfaction for Flow model
            try:
                constraint_stats = _compute_constraint_satisfaction(
                    sample_pred, sample_x, ngt_data, sys_data, config, device
                )
                print(f"  [Flow] Constraint satisfaction: Pg={constraint_stats['Pg_satisfy']:.2f}% "
                      f"Qg={constraint_stats['Qg_satisfy']:.2f}% Vm={constraint_stats['Vm_satisfy']:.2f}%")
                if constraint_stats['branch_ang_satisfy'] < 100.0:
                    print(f"    Branch: ang={constraint_stats['branch_ang_satisfy']:.2f}% "
                          f"pf={constraint_stats['branch_pf_satisfy']:.2f}%")
                
                # Log to TensorBoard if available
                if tb_logger is not None:
                    tb_logger.log_scalar('flow/constraint/Pg_satisfy', constraint_stats['Pg_satisfy'], epoch)
                    tb_logger.log_scalar('flow/constraint/Qg_satisfy', constraint_stats['Qg_satisfy'], epoch)
                    tb_logger.log_scalar('flow/constraint/Vm_satisfy', constraint_stats['Vm_satisfy'], epoch)
                    if constraint_stats['branch_ang_satisfy'] < 100.0:
                        tb_logger.log_scalar('flow/constraint/branch_ang_satisfy', constraint_stats['branch_ang_satisfy'], epoch)
                        tb_logger.log_scalar('flow/constraint/branch_pf_satisfy', constraint_stats['branch_pf_satisfy'], epoch)
            except Exception as e:
                # If constraint calculation fails, skip it
                pass

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
        print(f"[Warning] Projection setup failed: {e}")
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


def _sync_lambdas_to_loss_and_config(loss_fn, config, lam_cost: float, lam_carbon: float):
    """
    Best-effort sync so that even if loss_fn doesn't take preference arg,
    the scalarization weights used inside loss match the sampled preference.
    """
    # config (most common path: loss_fn keeps a reference)
    config.ngt_lambda_cost = float(lam_cost)
    config.ngt_lambda_carbon = float(lam_carbon)

    # loss_fn may keep its own config reference
    if hasattr(loss_fn, "config"):
        try:
            loss_fn.config.ngt_lambda_cost = float(lam_cost)
            loss_fn.config.ngt_lambda_carbon = float(lam_carbon)
        except Exception:
            pass

    # some implementations cache scalars directly
    for k in ["ngt_lambda_cost", "lambda_cost", "lam_cost"]:
        if hasattr(loss_fn, k):
            try:
                setattr(loss_fn, k, float(lam_cost))
            except Exception:
                pass
    for k in ["ngt_lambda_carbon", "lambda_carbon", "lam_carbon"]:
        if hasattr(loss_fn, k):
            try:
                setattr(loss_fn, k, float(lam_carbon))
            except Exception:
                pass

    # optional setter
    if hasattr(loss_fn, "set_lambdas") and callable(getattr(loss_fn, "set_lambdas")):
        try:
            loss_fn.set_lambdas(float(lam_cost), float(lam_carbon))
        except Exception:
            pass
    if hasattr(loss_fn, "set_preference") and callable(getattr(loss_fn, "set_preference")):
        try:
            loss_fn.set_preference(float(lam_cost), float(lam_carbon))
        except Exception:
            pass


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
     
    print("=" * 60)
    print(f"DeepOPF-V (Extended Version)")
    print("=" * 60)
    
    config.print_config()
    
    # Get model type
    model_type = config.model_type
    print(f"\nSelected model type: {model_type}")
    print(f"Available model types: {get_available_model_types()}") 
    
    # Create output directories if they don't exist
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    print(f"\nModel save directory: {config.model_save_dir}")
    print(f"Results directory: {config.results_dir}")
    
    # Load data
    sys_data, dataloaders, BRANFT = load_all_data(config)
    
    # Initialize models based on model_type
    input_channels = sys_data.x_train.shape[1]
    output_channels_vm = sys_data.yvm_train.shape[1]
    output_channels_va = sys_data.yva_train.shape[1]
    
    print(f"\nInput dimension: {input_channels}")
    print(f"Vm output dimension: {output_channels_vm}")
    print(f"Va output dimension: {output_channels_va}")
    
    # Check training mode first (unsupervised uses single NetV model, not dual models)
    training_mode = getattr(config, 'training_mode', 'unsupervised')
    
    # Initialize variables
    model_vm = None
    model_va = None
    pretrain_model_vm = None
    pretrain_model_va = None
    weight_decay = getattr(config, 'weight_decay', 0)
    criterion = nn.MSELoss()
    
    if training_mode == 'unsupervised':
        # ==================== Unsupervised Training (DeepOPF-NGT) ====================
        print("\n" + "=" * 60)
        print("Unsupervised Training Mode (DeepOPF-NGT)")
        print("=" * 60) 
        
        # Create TensorBoard logger if enabled
        tb_logger = None
        tb_enabled = os.environ.get('TB_ENABLED', 'False').lower() == 'true' # Check if TensorBoard is enabled
        use_flow_model = getattr(config, 'ngt_use_flow_model', False) #  Check if Flow model is enabled

        if tb_enabled and not debug:
            lambda_cost = getattr(config, 'ngt_lambda_cost', 0.9) # Get lambda cost
            model_type_name = "flow" if use_flow_model else "mlp" # Get model type name
            lc_str = f"{lambda_cost:.1f}".replace('.', '') # Get lambda cost string
            log_comment = f"ngt_{model_type_name}_lc{lc_str}_{config.Nbus}bus" # Get log comment
            runs_dir = os.path.join(os.path.dirname(__file__), 'runs') # Get runs directory
            os.makedirs(runs_dir, exist_ok=True) # Create runs directory
            tb_logger = TensorBoardLogger(log_dir=runs_dir, comment=log_comment) # Create TensorBoard logger
            print(f"[TensorBoard] Logging to: {runs_dir} (comment: {log_comment})") # Print log comment
        
        if use_flow_model:
            # Use Rectified Flow model
            print("Model Type: Rectified Flow (PreferenceConditionedNetV)")
            lambda_cost = getattr(config, 'ngt_lambda_cost', 0.9)
            lambda_carbon = getattr(config, 'ngt_lambda_carbon', 0.1)
            flow_inf_steps = getattr(config, 'ngt_flow_inf_steps', 10)
            use_projection = getattr(config, 'ngt_use_projection', False)
            
            # Check for progressive training (anchor from previous Flow model)
            anchor_model_path = os.environ.get('NGT_ANCHOR_MODEL_PATH', None)
            anchor_lambda_cost = os.environ.get('NGT_ANCHOR_LAMBDA_COST', None)
            
            if anchor_model_path and anchor_lambda_cost:
                anchor_preference = [float(anchor_lambda_cost), 1.0 - float(anchor_lambda_cost)]
                print(f"Progressive Training Mode:")
                print(f"  Anchor model: {anchor_model_path}")
                print(f"  Anchor preference: {anchor_preference}")
            else:
                anchor_model_path = None
                anchor_preference = None
                print("Independent Training Mode (VAE anchor)")
            
            # Check if zero_init is enabled via env var
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
            # Use MLP model  
            print("Model Type: MLP (NetV - Reference Implementation)")
            if not debug:
                lambda_cost = getattr(config, 'ngt_lambda_cost', 0.9)
                lambda_carbon = getattr(config, 'ngt_lambda_carbon', 0.1)
                model_ngt, loss_history, time_train, ngt_data, sys_data = train_unsupervised_ngt(
                    config, lambda_cost, lambda_carbon, sys_data, config.device,
                    tb_logger=tb_logger
                )
            else:
                # load a trained model to validate its performance debug mode
                model_path = "saved_models/NetV_ngt_300bus_E4500_final.pth"
                print(f"[Debug Mode] Loading trained NGT model from {model_path}")
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
                print("  NGT model loaded (weights assigned).")
                loss_history = None
                time_train = None
        
        # Close TensorBoard logger
        if tb_logger:
            tb_logger.close()
            print("[TensorBoard] Logger closed")
        
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
        # Use both original and unified evaluation methods for comparison
        print("\n" + "=" * 80) 
        
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
            Real_cost_total = np.sum(Real_cost, axis=1)
            
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
            
            # ===== Method 2: Unified Evaluation =====
            print("\n" + "-" * 80)
            print("Method 2: Unified Evaluation (evaluate_unified)")
            print("-" * 80)
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
            print("\n" + "-" * 80)  
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
        lc_str = f"_lc{config.ngt_lambda_cost:.1f}" if config.ngt_use_multi_objective else "_single"
        save_path = f'{config.model_save_dir}/ngt_{model_type_str}_results_{config.Nbus}bus{lc_str}.npz'
        if loss_history is not None:
            np.savez(save_path, **{k: v for k, v in loss_history.items()})
            print(f"\nTraining history saved to: {save_path}")
        
        # Save evaluation results
        eval_save_path = f'{config.model_save_dir}/ngt_{model_type_str}_eval_{config.Nbus}bus{lc_str}.npz'
        eval_to_save = {k: v for k, v in eval_results.items() 
                        if isinstance(v, (int, float, np.ndarray))}
        np.savez(eval_save_path, **eval_to_save)
        print(f"Evaluation results saved to: {eval_save_path}")
        
        print("\n" + "=" * 60)
        print(f"DeepOPF-NGT ({model_type_str.upper()}) Training and Evaluation completed!")
        if time_train is not None:
            print(f"Training time: {time_train:.2f} seconds ({time_train/60:.2f} minutes)")
        else:
            print("Training time: N/A (model loaded from file)")
        print("=" * 60)
        
        return model_ngt, None, results 

    elif training_mode == 'supervised':
        # ==================== Supervised Training ==================== 
        print("\n" + "=" * 60)
        print("Supervised Training Mode (Label-based Loss)")
        print("=" * 60)
        # Create models using factory function
        from models import create_model  # Import create_model function
        model_vm = create_model(model_type, input_channels, output_channels_vm, config, is_vm=True)
        model_va = create_model(model_type, input_channels, output_channels_va, config, is_vm=False)
        
        # Check if we need VAE anchor (for rectified flow or diffusion with use_vae_anchor=True)
        use_vae_anchor = getattr(config, 'use_vae_anchor', False)
        need_vae_anchor = model_type == 'rectified' or (model_type == 'diffusion' and use_vae_anchor)
        
        if need_vae_anchor:
            anchor_type = "rectified flow" if model_type == 'rectified' else "diffusion (use_vae_anchor=True)"
            print(f"\n[Info] Loading VAE anchor models for {anchor_type}...")
            
            # Load pretrained VAE models (needed as anchor generators)
            if config.pretrain_model_path_vm and os.path.exists(config.pretrain_model_path_vm):
                print(f"\nLoading pretrained Vm VAE from: {config.pretrain_model_path_vm}")
                pretrain_model_vm = create_model('vae', input_channels, output_channels_vm, config, is_vm=True)
                pretrain_model_vm.to(config.device)
                state_dict = torch.load(config.pretrain_model_path_vm, map_location=config.device, weights_only=True)
                pretrain_model_vm.load_state_dict(state_dict)
                pretrain_model_vm.eval()
                print(f"  Successfully loaded Vm VAE model!")  
            else:
                print(f"\n[Warning] Vm VAE not found: {config.pretrain_model_path_vm}")
                print("  Will use zero initialization for anchors in test mode.")
                
            if config.pretrain_model_path_va and os.path.exists(config.pretrain_model_path_va):
                print(f"\nLoading pretrained Va VAE from: {config.pretrain_model_path_va}")
                pretrain_model_va = create_model('vae', input_channels, output_channels_va, config, is_vm=False)
                pretrain_model_va.to(config.device)
                state_dict = torch.load(config.pretrain_model_path_va, map_location=config.device, weights_only=True)
                pretrain_model_va.load_state_dict(state_dict)
                pretrain_model_va.eval()
                print(f"  Successfully loaded Va VAE model!")
            else:
                print(f"\n[Warning] Va VAE not found: {config.pretrain_model_path_va}")
                print("  Will use zero initialization for anchors in test mode.")
            
            # Attach pretrain_model to FM/DM models
            model_vm.pretrain_model = pretrain_model_vm
            model_va.pretrain_model = pretrain_model_va
        elif model_type == 'diffusion':
            print(f"\n[Info] Diffusion model with use_vae_anchor=False, using Gaussian noise as starting point.")
     
        # ==================== Training Mode ====================   
        model_vm.to(config.device)
        model_va.to(config.device)
        print(f'\nModels moved to: {config.device}')
        
        # Initialize optimizers
        optimizer_vm = torch.optim.Adam(model_vm.parameters(), lr=config.Lrm, weight_decay=weight_decay)
        optimizer_va = torch.optim.Adam(model_va.parameters(), lr=config.Lra, weight_decay=weight_decay) 
    
        # Initialize schedulers (optional, only for supervised mode)
        scheduler_vm = None
        scheduler_va = None
        if hasattr(config, 'learning_rate_decay') and config.learning_rate_decay:
            step_size, gamma = config.learning_rate_decay
            scheduler_vm = torch.optim.lr_scheduler.StepLR(optimizer_vm, step_size=step_size, gamma=gamma)
            scheduler_va = torch.optim.lr_scheduler.StepLR(optimizer_va, step_size=step_size, gamma=gamma)
            print(f"Learning rate scheduler enabled: step_size={step_size}, gamma={gamma}") 
        
        if not debug:
            # Train Vm model
            model_vm, lossvm, time_train_vm = train_voltage_magnitude(
                config, model_vm, optimizer_vm, dataloaders['train_vm'],
                sys_data, criterion, config.device, model_type=model_type,
                pretrain_model=pretrain_model_vm, scheduler=scheduler_vm
            )
            
            # Train Va model
            model_va, lossva, time_train_va = train_voltage_angle(
                config, model_va, optimizer_va, dataloaders['train_va'],
                criterion, config.device, model_type=model_type,
                pretrain_model=pretrain_model_va, scheduler=scheduler_va
            )
        else: 
            vm_ckpt_path = "/home/yuepeng/code/multi_objective_opf/main_part/saved_models/modelvm300r2N1Lm8642E1000_simple.pth"
            va_ckpt_path = "/home/yuepeng/code/multi_objective_opf/main_part/saved_models/modelva300r2N1La8642_simple_E1000F1.pth"
            print(f"\n[Debug Mode] Loading trained Vm model from {vm_ckpt_path}")
            model_vm.load_state_dict(torch.load(vm_ckpt_path, map_location=config.device, weights_only=True))
            print("  Vm model loaded (weights assigned).")
            print(f"[Debug Mode] Loading trained Va model from {va_ckpt_path}")
            model_va.load_state_dict(torch.load(va_ckpt_path, map_location=config.device, weights_only=True))
            print("  Va model loaded (weights assigned).")
            lossvm = None
            lossva = None  
        
        # Also run unified evaluation for comparison
        print("\n" + "=" * 80)
        print("Running UNIFIED evaluate_unified(...)")
        print("=" * 80) 

        ctx = build_ctx_from_supervised(config, sys_data, dataloaders, BRANFT, config.device)
        predictor = SupervisedPredictor(
            model_vm, model_va, dataloaders,
            model_type=model_type,
            pretrain_model_vm=pretrain_model_vm,
            pretrain_model_va=pretrain_model_va,
        )
        results_unified = evaluate_unified(ctx, predictor, apply_post_processing=True, verbose=True)
        
        results["results_unified"] = results_unified 
        
        if not debug:
            # Save results (only if training mode or explicitly requested) 
            save_results(config, results, lossvm, lossva)
            plot_training_curves(lossvm, lossva)
        return model_vm, model_va, results 


if __name__ == "__main__": 
    debug = bool(int(os.environ.get('DEBUG', '0')))
    main(debug=debug)

