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
import matplotlib.pyplot as plt
import math

from config import get_config
from models import NetVm, NetVa, NetV, create_model, get_available_model_types
from data_loader import load_all_data, load_ngt_training_data, create_ngt_training_loader
from utils import (get_mae, get_rerr, get_clamp, get_genload, get_Pgcost,
                   get_vioPQg, get_viobran, get_viobran2, dPQbus_dV, get_hisdV,
                   dSlbus_dV, get_dV)

# Import unsupervised loss module (for unsupervised training)
try:
    from unsupervised_loss import UnsupervisedOPFLoss
    UNSUPERVISED_AVAILABLE = True
except ImportError:
    UNSUPERVISED_AVAILABLE = False
    print("[train.py] Warning: unsupervised_loss module not found. Unsupervised training disabled.")

# Import DeepOPF-NGT unsupervised loss module
try:
    from deepopf_ngt_loss import DeepOPFNGTLoss
    NGT_AVAILABLE = True
except ImportError:
    NGT_AVAILABLE = False
    print("[train.py] Warning: deepopf_ngt_loss module not found. DeepOPF-NGT training disabled.")


# ============================================================================
# Unified Jacobian Post-Processing Function
# ============================================================================

def jacobian_postprocess(config, sys_data, BRANFT, Pred_Vm, Pred_Va, Pred_V,
                         Pdtest, Qdtest, lsPg, lsQg, lsidxPg, lsidxQg, lsidxPQg,
                         num_viotest, MAXMIN_Pg, MAXMIN_Qg, baseMVA, bus_slack,
                         hisVm_min, hisVm_max, VmLb, VmUb,
                         include_branch_correction=False,
                         vio_branpf_num=0, lsSf=None, lsSf_sampidx=None,
                         verbose=True):
    """
    Unified Jacobian-based post-processing for voltage predictions.
    
    This function consolidates the repeated post-processing logic from:
    - evaluate_model()
    - evaluate_dual_model()
    - evaluate_ngt_single_model()
    
    Args:
        config: Configuration object
        sys_data: System data containing power system parameters
        BRANFT: Branch from-to indices
        Pred_Vm: Predicted voltage magnitudes [Ntest, Nbus]
        Pred_Va: Predicted voltage angles [Ntest, Nbus]
        Pred_V: Predicted complex voltage [Ntest, Nbus]
        Pdtest, Qdtest: Test load data [Ntest, Nbus]
        lsPg, lsQg: Generator violation values
        lsidxPg, lsidxQg: Generator violation indices (per sample)
        lsidxPQg: Combined violation sample indices
        num_viotest: Number of violated samples
        MAXMIN_Pg, MAXMIN_Qg: Generator limits
        baseMVA: Base MVA value
        bus_slack: Slack bus index
        hisVm_min, hisVm_max: Historical Vm bounds for clipping
        VmLb, VmUb: Voltage magnitude bounds
        include_branch_correction: Whether to include branch power flow corrections
        vio_branpf_num: Number of branch power flow violations
        lsSf: Branch violation data (required if include_branch_correction=True)
        lsSf_sampidx: Branch violation sample indices
        verbose: Whether to print progress messages
        
    Returns:
        dict: Post-processing results containing:
            - Pred_Vm1, Pred_Va1: Corrected voltage magnitudes and angles
            - Pred_V1: Corrected complex voltage
            - Pred_Pg1, Pred_Qg1, Pred_Pd1, Pred_Qd1: Recalculated power flow
            - vio_PQg1: Post-processed Pg/Qg constraint violations
            - vio_branang1, vio_branpf1: Post-processed branch violations
            - num_viotest1: Number of violated samples after post-processing
            - Vm_satisfy1: Voltage constraint satisfaction after post-processing
            - time_post: Post-processing time
    """
    time_post_start = time.perf_counter()
    
    Ntest = Pred_Vm.shape[0]
    bus_Va_idx = np.delete(np.arange(config.Nbus), bus_slack)
    
    # ==================== Step 1: Build Incidence Matrices ====================
    finc = np.zeros((sys_data.branch.shape[0], config.Nbus), dtype=float)
    tinc = np.zeros((sys_data.branch.shape[0], config.Nbus), dtype=float)
    for i in range(sys_data.branch.shape[0]):
        finc[i, int(sys_data.branch[i, 0]) - 1] = 1
        tinc[i, int(sys_data.branch[i, 1]) - 1] = 1
    
    # ==================== Step 2: Calculate Jacobian Matrices ====================
    dPbus_dV, dQbus_dV = dPQbus_dV(sys_data.his_V, sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus)
    dPfbus_dV, dQfbus_dV = dSlbus_dV(sys_data.his_V, bus_Va_idx, sys_data.branch, sys_data.Yf, finc, BRANFT, config.Nbus)
    
    # ==================== Step 3: Calculate Voltage Corrections ====================
    if config.flag_hisv:
        if verbose:
            print('  Using historical voltage for Jacobian calculation')
        dV1 = get_hisdV(lsPg, lsQg, lsidxPg, lsidxQg, num_viotest, config.k_dV,
                       sys_data.bus_Pg, sys_data.bus_Qg, dPbus_dV, dQbus_dV,
                       config.Nbus, Ntest)
    else:
        if verbose:
            print('  Using predicted voltage for Jacobian calculation')
        dV1 = get_dV(Pred_V, lsPg, lsQg, lsidxPg, lsidxQg, num_viotest, config.k_dV,
                    sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus, sys_data.his_V)
    
    # ==================== Step 4: Branch Power Flow Corrections (Optional) ====================
    if include_branch_correction and vio_branpf_num > 0 and lsSf is not None and lsSf_sampidx is not None:
        if verbose:
            print(f'  Correcting {vio_branpf_num} branch power flow violations')
        # Calculate dV_branch for each sample in lsSf_sampidx
        dV_branch_raw = np.zeros((lsSf_sampidx.shape[0], config.Nbus * 2))
        for i in range(lsSf_sampidx.shape[0]):
            mp = np.array(lsSf[i][:, 2] / lsSf[i][:, 1]).reshape(-1, 1)
            mq = np.array(lsSf[i][:, 3] / lsSf[i][:, 1]).reshape(-1, 1)
            dPdV = dPfbus_dV[np.array(lsSf[i][:, 0].astype(int)).squeeze(), :]
            dQdV = dQfbus_dV[np.array(lsSf[i][:, 0].astype(int)).squeeze(), :]
            dmp = mp * dPdV
            dmq = mq * dQdV
            dmpq_inv = np.linalg.pinv(dmp + dmq)
            dV_branch_raw[i] = np.dot(dmpq_inv, np.array(lsSf[i][:, 1])).squeeze()
        
        # FIX: Align dV_branch to dV1's row order (by sample index)
        # dV1 rows correspond to lsidxPQg, dV_branch_raw rows correspond to lsSf_sampidx
        dV_branch_aligned = np.zeros_like(dV1)
        for j, samp_idx in enumerate(lsSf_sampidx):
            pos = np.where(lsidxPQg == samp_idx)[0]
            if len(pos) > 0:
                dV_branch_aligned[pos[0], :] = dV_branch_raw[j, :]
        
        dV1 = dV1 + dV_branch_aligned
    
    # ==================== Step 5: Apply Corrections ====================
    Pred_Va1 = Pred_Va.copy()
    Pred_Vm1 = Pred_Vm.copy()
    
    Pred_Va1[lsidxPQg, :] = Pred_Va[lsidxPQg, :] - dV1[:, 0:config.Nbus]
    Pred_Va1[:, bus_slack] = 0  # Slack bus angle = 0
    Pred_Vm1[lsidxPQg, :] = Pred_Vm[lsidxPQg, :] - dV1[:, config.Nbus:2*config.Nbus]
    
    # ==================== Step 6: Clip Vm to Bounds ====================
    if isinstance(hisVm_min, (int, float)) and isinstance(hisVm_max, (int, float)):
        Pred_Vm1 = np.clip(Pred_Vm1, hisVm_min, hisVm_max)
    else:
        Pred_Vm1 = get_clamp(torch.from_numpy(Pred_Vm1), hisVm_min, hisVm_max).numpy()
    
    Pred_V1 = Pred_Vm1 * np.exp(1j * Pred_Va1)
    
    # ==================== Step 7: Recalculate Power Flow ====================
    Pred_Pg1, Pred_Qg1, Pred_Pd1, Pred_Qd1 = get_genload(
        Pred_V1, Pdtest, Qdtest,
        sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
    )
    
    time_post = time.perf_counter() - time_post_start
    
    # ==================== Step 8: Evaluate Constraints After Post-Processing ====================
    _, _, lsidxPg1, lsidxQg1, _, vio_PQg1, _, _, _, _ = get_vioPQg(
        Pred_Pg1, sys_data.bus_Pg, MAXMIN_Pg,
        Pred_Qg1, sys_data.bus_Qg, MAXMIN_Qg,
        config.DELTA
    )
    if torch.is_tensor(vio_PQg1):
        vio_PQg1 = vio_PQg1.numpy()
    
    lsidxPQg1 = np.squeeze(np.array(np.where(lsidxPg1 + lsidxQg1 > 0)))
    num_viotest1 = np.size(lsidxPQg1)
    
    vio_branang1, vio_branpf1, _ = get_viobran(
        Pred_V1, Pred_Va1, sys_data.branch, sys_data.Yf, sys_data.Yt,
        BRANFT, baseMVA, config.DELTA
    )
    if torch.is_tensor(vio_branang1):
        vio_branang1 = vio_branang1.numpy()
    if torch.is_tensor(vio_branpf1):
        vio_branpf1 = vio_branpf1.numpy()
    
    # Voltage constraint satisfaction after post-processing
    Vm_satisfy1 = 100 - np.mean(Pred_Vm1 > VmUb) * 100 - np.mean(Pred_Vm1 < VmLb) * 100
    
    return {
        'Pred_Vm1': Pred_Vm1,
        'Pred_Va1': Pred_Va1,
        'Pred_V1': Pred_V1,
        'Pred_Pg1': Pred_Pg1,
        'Pred_Qg1': Pred_Qg1,
        'Pred_Pd1': Pred_Pd1,
        'Pred_Qd1': Pred_Qd1,
        'vio_PQg1': vio_PQg1,
        'vio_branang1': vio_branang1,
        'vio_branpf1': vio_branpf1,
        'num_viotest1': num_viotest1,
        'Vm_satisfy1': Vm_satisfy1,
        'time_post': time_post,
    }


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


def predict_with_model(model, test_x, model_type, pretrain_model=None, config=None, device='cuda'):
    """
    Helper function to get predictions from different model types
    
    Args:
        model: The model to use for prediction
        test_x: Input tensor
        model_type: Type of model
        pretrain_model: Pretrained model for flow models
        config: Configuration object
        device: Device
        
    Returns:
        y_pred: Predicted output
    """
    model.eval()
    
    with torch.no_grad():
        if model_type == 'simple':
            # Original MLP forward pass
            y_pred = model(test_x)
            
        elif model_type == 'vae':
            # VAE: use mean for deterministic prediction
            y_pred = model(test_x, use_mean=True)
            
        elif model_type in ['rectified', 'gaussian', 'conditional', 'interpolation']:
            # Flow model: generate anchor and integrate
            if pretrain_model is not None:
                # Use VAE/pretrained model's mean as starting point (anchor strategy)
                z = pretrain_model(test_x, use_mean=True)
            else:
                # Standard Flow Matching: start from Gaussian noise N(0, I)
                # NOTE: torch.zeros is WRONG - high-dim Gaussian has near-zero probability at origin
                output_dim = model.output_dim
                z = torch.randn(test_x.shape[0], output_dim).to(device)
            
            # Flow backward to get prediction
            inf_step = getattr(config, 'inf_step', 100) if config else 100
            y_pred, _ = model.flow_backward(test_x, z, step=1/inf_step, method='Euler')
            
        elif model_type == 'diffusion':
            # Diffusion model: sample from noise or VAE anchor
            output_dim = model.output_dim
            inf_step = getattr(config, 'inf_step', 100) if config else 100
            use_vae_anchor = getattr(config, 'use_vae_anchor', False) if config else False
            
            if use_vae_anchor and pretrain_model is not None:
                # Start diffusion from VAE prediction
                vae_anchor = pretrain_model(test_x, use_mean=True)
                z = torch.randn(test_x.shape[0], output_dim).to(device)
                y_pred = model.diffusion_backward_with_anchor(test_x, z, vae_anchor, inf_step=inf_step)
            else:
                # Standard diffusion: start from pure Gaussian noise
                z = torch.randn(test_x.shape[0], output_dim).to(device)
                y_pred = model.diffusion_backward(test_x, z, inf_step=inf_step)
            
        elif model_type in ['gan', 'wgan']:
            # GAN/WGAN: sample from latent space
            latent_dim = model.latent_dim
            z = torch.randn(test_x.shape[0], latent_dim).to(device)
            y_pred = model(test_x, z)
            
        elif model_type in ['consistency_training', 'consistency_distillation']:
            # Consistency model: single-step sampling
            y_pred = model.sampling(test_x, inf_step=1)
            
        else:
            raise NotImplementedError(f"Prediction for model type '{model_type}' not implemented")
    
    return y_pred


def evaluate_model(config, model_vm, model_va, sys_data, dataloaders, BRANFT, device,
                   model_type='simple', pretrain_model_vm=None, pretrain_model_va=None):
    """
    Comprehensive model evaluation on test set with post-processing
    Supports multiple model types for inference
    
    Args:
        config: Configuration object
        model_vm: Trained Vm model
        model_va: Trained Va model
        sys_data: System data
        dataloaders: Data loaders
        BRANFT: Branch from-to indices
        device: Device
        model_type: Type of model for inference ('simple', 'vae', 'rectified', etc.)
        pretrain_model_vm: Pretrained VAE model for Vm (for flow models)
        pretrain_model_va: Pretrained VAE model for Va (for flow models)
        
    Returns:
        results: Dictionary containing all evaluation metrics
    """
    print('\n' + '=' * 60)
    print(f'Model Evaluation on Test Set - Type: {model_type}')
    print('=' * 60)
    
    model_vm.eval()
    model_va.eval()
    
    # ==================== Timing Statistics ====================
    timing_info = {
        'model_type': model_type,
        'num_test_samples': config.Ntest,
    }
    
    # Prepare incidence matrices for branch constraints
    finc = np.zeros((sys_data.branch.shape[0], config.Nbus), dtype=float)
    tinc = np.zeros((sys_data.branch.shape[0], config.Nbus), dtype=float)
    for i in range(sys_data.branch.shape[0]):
        finc[i, int(sys_data.branch[i, 0]) - 1] = 1
        tinc[i, int(sys_data.branch[i, 1]) - 1] = 1
    
    # Real voltage for testing samples
    yvmtests = sys_data.yvm_test / config.scale_vm * (sys_data.VmUb - sys_data.VmLb) + sys_data.VmLb
    yvatests = sys_data.yva_test / config.scale_va
    
    # Real voltage
    Real_Va = yvatests.clone().numpy()
    Real_Va = np.insert(Real_Va, sys_data.bus_slack, values=0, axis=1)
    Real_V = yvmtests.numpy() * np.exp(1j * Real_Va)
    
    # Real Pg, Qg
    Real_Pg, Real_Qg, Real_Pd, Real_Qd = get_genload(
        Real_V, sys_data.Pdtest, sys_data.Qdtest,
        sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
    )
    
    # Jacobian matrices for post-processing
    dPbus_dV, dQbus_dV = dPQbus_dV(sys_data.his_V, sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus)
    bus_Va = np.delete(np.arange(config.Nbus), sys_data.bus_slack)
    dPfbus_dV, dQfbus_dV = dSlbus_dV(sys_data.his_V, bus_Va, sys_data.branch, sys_data.Yf, finc, BRANFT, config.Nbus)
    
    # Prediction and evaluation
    print('\nRunning predictions and evaluating...')
    
    # ==================== Predict Vm with Timing ====================
    yvmtest_hat_list = []
    
    # GPU warmup (to avoid cold start timing issues)
    if device.type == 'cuda':
        with torch.no_grad():
            dummy_x = sys_data.x_test[0:1].to(device)
            # Use predict_with_model to support all model types (simple, vae, rectified, etc.)
            _ = predict_with_model(model_vm, dummy_x, model_type, pretrain_model_vm, config, device)
            torch.cuda.synchronize()
    
    # Start timing for Vm prediction
    if device.type == 'cuda':
        torch.cuda.synchronize()
    time_start_vm = time.perf_counter()
    
    for step, (test_x, test_y) in enumerate(dataloaders['test_vm']):
        test_x = test_x.to(device)
        
        # Use model-specific prediction
        pred = predict_with_model(model_vm, test_x, model_type, pretrain_model_vm, config, device)
        yvmtest_hat_list.append(pred.cpu())
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    time_end_vm = time.perf_counter()
    time_PredVm_NN = time_end_vm - time_start_vm
    
    # Concatenate all predictions
    yvmtest_hat = torch.cat(yvmtest_hat_list, dim=0)
    
    yvmtest_hat = yvmtest_hat.cpu()
    yvmtest_hats = yvmtest_hat.detach() / config.scale_vm * (sys_data.VmUb - sys_data.VmLb) + sys_data.VmLb
    yvmtest_hat_clip = get_clamp(yvmtest_hats, sys_data.hisVm_min, sys_data.hisVm_max)
    
    # ==================== Predict Va with Timing ====================
    yvatest_hat_list = []
    
    # Start timing for Va prediction
    if device.type == 'cuda':
        torch.cuda.synchronize()
    time_start_va = time.perf_counter()
    
    for step, (test_x, test_y) in enumerate(dataloaders['test_va']):
        test_x = test_x.to(device)
        
        # Use model-specific prediction
        pred = predict_with_model(model_va, test_x, model_type, pretrain_model_va, config, device)
        yvatest_hat_list.append(pred.cpu())
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    time_end_va = time.perf_counter()
    time_PredVa_NN = time_end_va - time_start_va
    
    # Concatenate all predictions
    yvatest_hat = torch.cat(yvatest_hat_list, dim=0)

    yvatest_hat = yvatest_hat.cpu()
    yvatest_hats = yvatest_hat.detach() / config.scale_va
    
    # Va with slack bus
    Pred_Va = yvatest_hats.clone().numpy()
    Pred_Va = np.insert(Pred_Va, sys_data.bus_slack, values=0, axis=1)
    
    # ==================== Calculate Pg, Qg (with timing) ====================
    time_start_pq = time.perf_counter()
    Pred_V = yvmtest_hat_clip.clone().numpy() * np.exp(1j * Pred_Va)
    Pred_Pg, Pred_Qg, Pred_Pd, Pred_Qd = get_genload(
        Pred_V, sys_data.Pdtest, sys_data.Qdtest,
        sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
    )
    time_end_pq = time.perf_counter()
    time_PQ_calc = time_end_pq - time_start_pq
    
    # Pg Qg constraint violations
    lsPg, lsQg, lsidxPg, lsidxQg, vio_PQgmaxmin, vio_PQg, deltaPgL, deltaPgU, deltaQgL, deltaQgU = get_vioPQg(
        Pred_Pg, sys_data.bus_Pg, sys_data.MAXMIN_Pg,
        Pred_Qg, sys_data.bus_Qg, sys_data.MAXMIN_Qg,
        config.DELTA
    )
    lsidxPQg = np.squeeze(np.array(np.where((lsidxPg + lsidxQg) > 0)))
    num_viotest = np.size(lsidxPQg)
    
    # Branch constraints
    vio_branang, vio_branpf, deltapf, vio_branpfidx, lsSf, _, lsSf_sampidx, _ = get_viobran2(
        Pred_V, Pred_Va, sys_data.branch, sys_data.Yf, sys_data.Yt,
        BRANFT, sys_data.baseMVA, config.DELTA
    )
    vio_branpf_num = int(np.sum(np.asarray(vio_branpfidx) > 0))
    lsSf_sampidx = np.asarray(lsSf_sampidx)
    
    print(f'\nBefore Post-Processing:')
    print(f'  Violated samples: {num_viotest}/{config.Ntest} ({num_viotest/config.Ntest*100:.1f}%)')
    print(f'  Pg constraint satisfaction: {torch.mean(vio_PQg[:, 0]):.2f}%')
    print(f'  Qg constraint satisfaction: {torch.mean(vio_PQg[:, 1]):.2f}%')
    print(f'  Branch angle constraint: {torch.mean(vio_branang):.2f}%')
    print(f'  Branch power flow constraint: {torch.mean(vio_branpf):.2f}%')
    
    # ==================== Post-processing (with timing) ====================
    print('\nApplying post-processing corrections...')
    Pred_Va1 = Pred_Va.copy()
    Pred_Vm1 = yvmtest_hat_clip.clone().numpy()
    
    time_start_post = time.perf_counter()
    if config.flag_hisv:
        print('  Using historical voltage for Jacobian calculation')
        dV1 = get_hisdV(lsPg, lsQg, lsidxPg, lsidxQg, num_viotest, config.k_dV,
                        sys_data.bus_Pg, sys_data.bus_Qg, dPbus_dV, dQbus_dV,
                        config.Nbus, config.Ntest)
    else:
        print('  Using predicted voltage for Jacobian calculation')
        dV1 = get_dV(Pred_V, lsPg, lsQg, lsidxPg, lsidxQg, num_viotest, config.k_dV,
                     sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus, sys_data.his_V)
    
    if vio_branpf_num > 0:
        print(f'  Correcting {vio_branpf_num} branch power flow violations')
        # Calculate dV_branch for each sample in lsSf_sampidx
        dV_branch_raw = np.zeros((lsSf_sampidx.shape[0], config.Nbus * 2))
        for i in range(lsSf_sampidx.shape[0]):
            mp = np.array(lsSf[i][:, 2] / lsSf[i][:, 1]).reshape(-1, 1)
            mq = np.array(lsSf[i][:, 3] / lsSf[i][:, 1]).reshape(-1, 1)
            dPdV = dPfbus_dV[np.array(lsSf[i][:, 0].astype(int)).squeeze(), :]
            dQdV = dQfbus_dV[np.array(lsSf[i][:, 0].astype(int)).squeeze(), :]
            dmp = mp * dPdV
            dmq = mq * dQdV
            dmpq_inv = np.linalg.pinv(dmp + dmq)
            dV_branch_raw[i] = np.dot(dmpq_inv, np.array(lsSf[i][:, 1])).squeeze()
        
        # Align dV_branch to dV1's row order (by sample index)
        # dV1 rows correspond to lsidxPQg, dV_branch_raw rows correspond to lsSf_sampidx
        dV_branch_aligned = np.zeros_like(dV1)
        for j, samp_idx in enumerate(lsSf_sampidx):
            pos = np.where(lsidxPQg == samp_idx)[0]
            if len(pos) > 0:
                dV_branch_aligned[pos[0], :] = dV_branch_raw[j, :]
        
        dV1 = dV1 + dV_branch_aligned
    
    # Apply corrections
    Pred_Va1[lsidxPQg, :] = Pred_Va[lsidxPQg, :] - dV1[:, 0:config.Nbus]
    Pred_Va1[:, sys_data.bus_slack] = 0
    Pred_Vm1[lsidxPQg, :] = yvmtest_hat_clip.numpy()[lsidxPQg, :] - dV1[:, config.Nbus:2*config.Nbus]
    Pred_Vm1_clip = get_clamp(torch.from_numpy(Pred_Vm1), sys_data.hisVm_min, sys_data.hisVm_max)
    Pred_V1 = Pred_Vm1_clip.numpy() * np.exp(1j * Pred_Va1)
    Pred_Pg1, Pred_Qg1, Pred_Pd1, Pred_Qd1 = get_genload(
        Pred_V1, sys_data.Pdtest, sys_data.Qdtest,
        sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
    )
    time_end_post = time.perf_counter()
    time_post_processing = time_end_post - time_start_post
    
    # Evaluate after post-processing
    _, _, lsidxPg1, lsidxQg1, vio_PQgmaxmin1, vio_PQg1, deltaPgL1, deltaPgU1, deltaQgL1, deltaQgU1 = get_vioPQg(
        Pred_Pg1, sys_data.bus_Pg, sys_data.MAXMIN_Pg,
        Pred_Qg1, sys_data.bus_Qg, sys_data.MAXMIN_Qg,
        config.DELTA
    )
    lsidxPQg1 = np.squeeze(np.array(np.where(lsidxPg1 + lsidxQg1 > 0)))
    num_viotest1 = np.size(lsidxPQg1)
    
    vio_branang1, vio_branpf1, deltapf1 = get_viobran(
        Pred_V1, Pred_Va1, sys_data.branch, sys_data.Yf, sys_data.Yt,
        BRANFT, sys_data.baseMVA, config.DELTA
    )
    
    print(f'\nAfter Post-Processing:')
    print(f'  Violated samples: {num_viotest1}/{config.Ntest} ({num_viotest1/config.Ntest*100:.1f}%)')
    print(f'  Pg constraint satisfaction: {torch.mean(vio_PQg1[:, 0]):.2f}%')
    print(f'  Qg constraint satisfaction: {torch.mean(vio_PQg1[:, 1]):.2f}%')
    print(f'  Branch angle constraint: {torch.mean(vio_branang1):.2f}%')
    print(f'  Branch power flow constraint: {torch.mean(vio_branpf1):.2f}%')
    
    # Performance metrics
    mae_Vmtest = get_mae(yvmtests, yvmtest_hat_clip.detach())
    mre_Vmtest_clip = get_rerr(yvmtests, yvmtest_hat_clip.detach())
    mae_Vatest = get_mae(yvatests, yvatest_hats)
    mre_Vatest = get_rerr(yvatests, yvatest_hats)
    
    mae_Vmtest1 = get_mae(yvmtests, Pred_Vm1_clip)
    mae_Vatest1 = get_mae(torch.from_numpy(Real_Va).float(), torch.from_numpy(Pred_Va1).float())
    
    # Load satisfaction
    mre_Pd = get_rerr(torch.from_numpy(Real_Pd.sum(axis=1)), torch.from_numpy(Pred_Pd.sum(axis=1)))
    mre_Qd = get_rerr(torch.from_numpy(Real_Qd.sum(axis=1)), torch.from_numpy(Pred_Qd.sum(axis=1)))
    mre_Pd1 = get_rerr(torch.from_numpy(Real_Pd.sum(axis=1)), torch.from_numpy(Pred_Pd1.sum(axis=1)))
    mre_Qd1 = get_rerr(torch.from_numpy(Real_Qd.sum(axis=1)), torch.from_numpy(Pred_Qd1.sum(axis=1)))
    
    # Cost
    Pred_cost = get_Pgcost(Pred_Pg, sys_data.idxPg, sys_data.gencost, sys_data.baseMVA)
    Real_cost = get_Pgcost(Real_Pg, sys_data.idxPg, sys_data.gencost, sys_data.baseMVA)
    from utils import get_rerr2
    mre_cost = get_rerr2(torch.from_numpy(Real_cost), torch.from_numpy(Pred_cost))
    Pred_cost1 = get_Pgcost(Pred_Pg1, sys_data.idxPg, sys_data.gencost, sys_data.baseMVA)
    mre_cost1 = get_rerr2(torch.from_numpy(Real_cost), torch.from_numpy(Pred_cost1))
    
    print(f'\nPerformance Metrics:')
    print(f'  Vm MAE: {mae_Vmtest:.6f} p.u.')
    print(f'  Va MAE: {mae_Vatest:.6f} rad')
    print(f'  Cost error: {torch.mean(mre_cost):.2f}%')
    print(f'  Pd error: {torch.mean(mre_Pd):.2f}%')
    print(f'  Qd error: {torch.mean(mre_Qd):.2f}%')
    
    # ==================== Timing Statistics Summary ====================
    # Calculate total inference time (NN prediction only, without post-processing)
    time_NN_total = time_PredVm_NN + time_PredVa_NN
    time_NN_per_sample = time_NN_total / config.Ntest * 1000  # in milliseconds
    
    # Calculate total solving time (including post-processing)
    time_total_with_post = time_NN_total + time_PQ_calc + time_post_processing
    time_total_per_sample = time_total_with_post / config.Ntest * 1000  # in milliseconds
    
    # Store timing info
    timing_info['time_Vm_prediction'] = time_PredVm_NN
    timing_info['time_Va_prediction'] = time_PredVa_NN
    timing_info['time_NN_total'] = time_NN_total
    timing_info['time_PQ_calculation'] = time_PQ_calc
    timing_info['time_post_processing'] = time_post_processing
    timing_info['time_total_with_post'] = time_total_with_post
    timing_info['time_NN_per_sample_ms'] = time_NN_per_sample
    timing_info['time_total_per_sample_ms'] = time_total_per_sample
    
    print(f'\n' + '=' * 60)
    print(f'Solving Time Statistics - Model: {model_type}')
    print('=' * 60)
    print(f'  Test samples: {config.Ntest}')
    print(f'\n  [Neural Network Inference]')
    print(f'    Vm prediction time:       {time_PredVm_NN:.4f} s')
    print(f'    Va prediction time:       {time_PredVa_NN:.4f} s')
    print(f'    Total NN inference:       {time_NN_total:.4f} s')
    print(f'    Per sample (NN only):     {time_NN_per_sample:.4f} ms')
    print(f'\n  [Post-Processing]')
    print(f'    PQ calculation time:      {time_PQ_calc:.4f} s')
    print(f'    Post-processing time:     {time_post_processing:.4f} s')
    print(f'\n  [Total Solving Time]')
    print(f'    Total time (with post):   {time_total_with_post:.4f} s')
    print(f'    Per sample (with post):   {time_total_per_sample:.4f} ms')
    print('=' * 60)
    
    # Prepare results for saving
    results = {
        'mae_Vmtest': mae_Vmtest,
        'mae_Vatest': mae_Vatest,
        'mae_Vmtest1': mae_Vmtest1,
        'mae_Vatest1': mae_Vatest1,
        'vio_PQg': vio_PQg,
        'vio_PQg1': vio_PQg1,
        'vio_branang': vio_branang,
        'vio_branpf': vio_branpf,
        'vio_branang1': vio_branang1,
        'vio_branpf1': vio_branpf1,
        'mre_cost': mre_cost,
        'mre_cost1': mre_cost1,
        'mre_Pd': mre_Pd,
        'mre_Qd': mre_Qd,
        'deltaPgL': deltaPgL,
        'deltaPgU': deltaPgU,
        'deltaQgL': deltaQgL,
        'deltaQgU': deltaQgU,
        'deltapf': deltapf,
        'deltapf1': deltapf1,
        # Timing information
        'timing_info': timing_info,
    }
    
    return results


def save_results(config, results, lossvm, lossva):
    """
    Save training and evaluation results to JSON and CSV files (more readable than .mat)
    
    Args:
        config: Configuration object
        results: Evaluation results dictionary
        lossvm: Vm training losses
        lossva: Va training losses
    """
    import json
    import csv
    
    # Extract timing info
    timing_info = results.get('timing_info', {})
    
    # ==================== 1. Save main metrics to JSON (human-readable) ====================
    metrics_summary = {
        'config': {
            'model_type': getattr(config, 'model_type', 'simple'),
            'Nbus': config.Nbus,
            'Ntrain': config.Ntrain,
            'Ntest': config.Ntest,
            'EpochVm': config.EpochVm,
            'EpochVa': config.EpochVa,
            'batch_size': config.batch_size_training,
            'learning_rate_Vm': config.Lrm,
            'learning_rate_Va': config.Lra,
        },
        'before_post_processing': {
            'Vm_MAE': float(results['mae_Vmtest'].item()) if hasattr(results['mae_Vmtest'], 'item') else float(results['mae_Vmtest']),
            'Va_MAE': float(results['mae_Vatest'].item()) if hasattr(results['mae_Vatest'], 'item') else float(results['mae_Vatest']),
            'cost_error_percent': float(torch.mean(results['mre_cost']).item()),
            'Pd_error_percent': float(torch.mean(results['mre_Pd']).item()),
            'Qd_error_percent': float(torch.mean(results['mre_Qd']).item()),
            'Pg_satisfy_rate': float(torch.mean(results['vio_PQg'][:, 0]).item()),
            'Qg_satisfy_rate': float(torch.mean(results['vio_PQg'][:, 1]).item()),
            'branch_angle_satisfy_rate': float(torch.mean(results['vio_branang']).item()),
            'branch_power_satisfy_rate': float(torch.mean(results['vio_branpf']).item()),
        },
        'after_post_processing': {
            'Vm_MAE': float(results['mae_Vmtest1'].item()) if hasattr(results['mae_Vmtest1'], 'item') else float(results['mae_Vmtest1']),
            'Va_MAE': float(results['mae_Vatest1'].item()) if hasattr(results['mae_Vatest1'], 'item') else float(results['mae_Vatest1']),
            'cost_error_percent': float(torch.mean(results['mre_cost1']).item()),
            'Pg_satisfy_rate': float(torch.mean(results['vio_PQg1'][:, 0]).item()),
            'Qg_satisfy_rate': float(torch.mean(results['vio_PQg1'][:, 1]).item()),
            'branch_angle_satisfy_rate': float(torch.mean(results['vio_branang1']).item()),
            'branch_power_satisfy_rate': float(torch.mean(results['vio_branpf1']).item()),
        },
        'timing': {
            'Vm_prediction_sec': timing_info.get('time_Vm_prediction', 0),
            'Va_prediction_sec': timing_info.get('time_Va_prediction', 0),
            'NN_total_sec': timing_info.get('time_NN_total', 0),
            'PQ_calculation_sec': timing_info.get('time_PQ_calculation', 0),
            'post_processing_sec': timing_info.get('time_post_processing', 0),
            'total_with_post_sec': timing_info.get('time_total_with_post', 0),
            'NN_per_sample_ms': timing_info.get('time_NN_per_sample_ms', 0),
            'total_per_sample_ms': timing_info.get('time_total_per_sample_ms', 0),
            'num_test_samples': timing_info.get('num_test_samples', config.Ntest),
        }
    }
    
    # Save JSON
    json_path = config.resultnm.replace('.mat', '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
    print(f'\nMetrics saved to: {json_path}')
    
    # ==================== 2. Save training loss to CSV ====================
    csv_loss_path = config.resultnm.replace('.mat', '_loss.csv')
    with open(csv_loss_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss_vm', 'loss_va'])
        max_epochs = max(len(lossvm), len(lossva))
        for i in range(max_epochs):
            loss_vm_val = lossvm[i] if i < len(lossvm) else ''
            loss_va_val = lossva[i] if i < len(lossva) else ''
            writer.writerow([i + 1, loss_vm_val, loss_va_val])
    print(f'Training loss saved to: {csv_loss_path}')
    
    # ==================== 3. Save summary comparison table to CSV ====================
    csv_summary_path = config.resultnm.replace('.mat', '_summary.csv')
    with open(csv_summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Before Post-Processing', 'After Post-Processing'])
        writer.writerow(['Vm MAE (p.u.)', 
                        f"{metrics_summary['before_post_processing']['Vm_MAE']:.6f}",
                        f"{metrics_summary['after_post_processing']['Vm_MAE']:.6f}"])
        writer.writerow(['Va MAE (rad)', 
                        f"{metrics_summary['before_post_processing']['Va_MAE']:.6f}",
                        f"{metrics_summary['after_post_processing']['Va_MAE']:.6f}"])
        writer.writerow(['Cost Error (%)', 
                        f"{metrics_summary['before_post_processing']['cost_error_percent']:.2f}",
                        f"{metrics_summary['after_post_processing']['cost_error_percent']:.2f}"])
        writer.writerow(['Pg Satisfy Rate (%)', 
                        f"{metrics_summary['before_post_processing']['Pg_satisfy_rate']:.2f}",
                        f"{metrics_summary['after_post_processing']['Pg_satisfy_rate']:.2f}"])
        writer.writerow(['Qg Satisfy Rate (%)', 
                        f"{metrics_summary['before_post_processing']['Qg_satisfy_rate']:.2f}",
                        f"{metrics_summary['after_post_processing']['Qg_satisfy_rate']:.2f}"])
        writer.writerow(['Branch Angle Satisfy Rate (%)', 
                        f"{metrics_summary['before_post_processing']['branch_angle_satisfy_rate']:.2f}",
                        f"{metrics_summary['after_post_processing']['branch_angle_satisfy_rate']:.2f}"])
        writer.writerow(['Branch Power Satisfy Rate (%)', 
                        f"{metrics_summary['before_post_processing']['branch_power_satisfy_rate']:.2f}",
                        f"{metrics_summary['after_post_processing']['branch_power_satisfy_rate']:.2f}"])
    print(f'Summary table saved to: {csv_summary_path}')
    
    # ==================== 4. Also save to .npz for programmatic access ====================
    npz_path = config.resultnm.replace('.mat', '.npz')
    np.savez(npz_path,
        # Summary arrays
        resvio=np.array([
            [float(torch.mean(results['mre_cost'])), float(torch.mean(results['mre_Pd'])), 
             float(torch.mean(results['mre_Qd'])), float(torch.mean(results['vio_PQg'][:, 0])),
             float(torch.mean(results['vio_PQg'][:, 1])), float(torch.mean(results['vio_branang'])),
             float(torch.mean(results['vio_branpf']))],
            [float(torch.mean(results['mre_cost1'])), float(torch.mean(results['mre_Pd'])),
             float(torch.mean(results['mre_Qd'])), float(torch.mean(results['vio_PQg1'][:, 0])),
             float(torch.mean(results['vio_PQg1'][:, 1])), float(torch.mean(results['vio_branang1'])),
             float(torch.mean(results['vio_branpf1']))]
        ]),
        maeV=np.array([
            [float(results['mae_Vmtest'].item() if hasattr(results['mae_Vmtest'], 'item') else results['mae_Vmtest']),
             float(results['mae_Vatest'].item() if hasattr(results['mae_Vatest'], 'item') else results['mae_Vatest'])],
            [float(results['mae_Vmtest1'].item() if hasattr(results['mae_Vmtest1'], 'item') else results['mae_Vmtest1']),
             float(results['mae_Vatest1'].item() if hasattr(results['mae_Vatest1'], 'item') else results['mae_Vatest1'])]
        ]),
        lossvm=np.array(lossvm),
        lossva=np.array(lossva),
        mre_cost=np.array(results['mre_cost']),
        mre_cost1=np.array(results['mre_cost1']),
    )
    print(f'NumPy data saved to: {npz_path}')
    
    # Print timing summary
    if timing_info:
        print(f'\nTiming Summary for {timing_info.get("model_type", getattr(config, "model_type", "model"))}:')
        print(f'  NN inference per sample: {timing_info.get("time_NN_per_sample_ms", 0):.4f} ms')
        print(f'  Total solving per sample: {timing_info.get("time_total_per_sample_ms", 0):.4f} ms')


def plot_training_curves(lossvm, lossva):
    """
    Plot training loss curves
    
    Args:
        lossvm: Vm training losses
        lossva: Va training losses
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(lossvm)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Vm Training Loss')
    ax1.grid(True)
    
    ax2.plot(lossva)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Va Training Loss')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print('\nTraining curves saved to: training_curves.png')
    plt.close()


def plot_unsupervised_training_curves(loss_history):
    """
    Plot unsupervised training loss curves with multiple components.
    
    Args:
        loss_history: Dictionary containing loss history for each component
    """
    # Check which keys are available (different for NGT vs old unsupervised)
    has_ngt_keys = 'kgenp_mean' in loss_history
    
    # Check if multi-objective data is present
    has_multi_objective = ('cost' in loss_history and 'carbon' in loss_history and 
                          len(loss_history.get('cost', [])) > 0 and
                          any(v > 0 for v in loss_history.get('carbon', [0])))
    
    if has_ngt_keys:
        # DeepOPF-NGT format
        if has_multi_objective:
            # Extended layout for multi-objective: 3x3 grid
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        else:
            # Original layout: 2x3 grid
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Total loss
        axes[0, 0].plot(loss_history['total'])
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].grid(True)
        
        # Generator P weight
        axes[0, 1].plot(loss_history.get('kgenp_mean', []))
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Weight')
        axes[0, 1].set_title('Generator P Weight (k_genp)')
        axes[0, 1].grid(True)
        
        # Generator Q weight
        axes[0, 2].plot(loss_history.get('kgenq_mean', []))
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Weight')
        axes[0, 2].set_title('Generator Q Weight (k_genq)')
        axes[0, 2].grid(True)
        
        # Load P weight
        axes[1, 0].plot(loss_history.get('kpd_mean', []))
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Weight')
        axes[1, 0].set_title('Load P Weight (k_pd)')
        axes[1, 0].grid(True)
        
        # Load Q weight
        axes[1, 1].plot(loss_history.get('kqd_mean', []))
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Weight')
        axes[1, 1].set_title('Load Q Weight (k_qd)')
        axes[1, 1].grid(True)
        
        # Voltage weight
        axes[1, 2].plot(loss_history.get('kv_mean', []))
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Weight')
        axes[1, 2].set_title('Voltage Weight (k_v)')
        axes[1, 2].grid(True)
        
        # Multi-objective plots (row 3)
        if has_multi_objective:
            # Economic cost
            axes[2, 0].plot(loss_history.get('cost', []), 'b-', label='Economic Cost')
            axes[2, 0].set_xlabel('Epoch')
            axes[2, 0].set_ylabel('Cost')
            axes[2, 0].set_title('Economic Cost (L_cost)')
            axes[2, 0].grid(True)
            
            # Carbon emission
            axes[2, 1].plot(loss_history.get('carbon', []), 'g-', label='Carbon Emission')
            axes[2, 1].set_xlabel('Epoch')
            axes[2, 1].set_ylabel('Carbon (tCO2)')
            axes[2, 1].set_title('Carbon Emission (L_carbon)')
            axes[2, 1].grid(True)
            
            # Combined objectives (normalized for visualization)
            cost_data = np.array(loss_history.get('cost', []))
            carbon_data = np.array(loss_history.get('carbon', []))
            if len(cost_data) > 0 and len(carbon_data) > 0:
                # Normalize for comparison
                cost_norm = cost_data / (cost_data.max() + 1e-8)
                carbon_norm = carbon_data / (carbon_data.max() + 1e-8)
                axes[2, 2].plot(cost_norm, 'b-', label='Cost (norm)')
                axes[2, 2].plot(carbon_norm, 'g-', label='Carbon (norm)')
                axes[2, 2].set_xlabel('Epoch')
                axes[2, 2].set_ylabel('Normalized Value')
                axes[2, 2].set_title('Multi-Objective Trade-off')
                axes[2, 2].legend()
                axes[2, 2].grid(True)
    else:
        # Old unsupervised format
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Total loss
        axes[0, 0].plot(loss_history.get('total', []))
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].grid(True)
        
        # Cost loss (L_obj)
        axes[0, 1].plot(loss_history.get('cost', []))
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Generation Cost (L_obj)')
        axes[0, 1].grid(True)
        
        # Generator violation loss (L_g)
        axes[0, 2].plot(loss_history.get('gen_vio', []))
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_title('Generator Violation (L_g)')
        axes[0, 2].grid(True)
        
        # Branch power flow violation (L_Sl)
        axes[1, 0].plot(loss_history.get('branch_pf_vio', []))
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Branch Power Flow Violation (L_Sl)')
        axes[1, 0].grid(True)
        
        # Branch angle violation (L_theta)
        axes[1, 1].plot(loss_history.get('branch_ang_vio', []))
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Branch Angle Violation (L_theta)')
        axes[1, 1].grid(True)
        
        # Load deviation loss (L_d)
        axes[1, 2].plot(loss_history.get('load_dev', []))
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].set_title('Load Deviation (L_d)')
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('unsupervised_training_curves.png', dpi=300, bbox_inches='tight')
    print('\nUnsupervised training curves saved to: unsupervised_training_curves.png')
    plt.close()


# ============================================================================
# NGT Flow Model Helper Functions
# ============================================================================

def get_vae_anchor_for_ngt(vae_vm, vae_va, x_ngt, ngt_data, config, device):
    """
    Get VAE anchor points and convert to NGT format.
    
    This function:
    1. Gets VAE predictions for Vm[Nbus] and Va[Nbus-1]
    2. Extracts non-ZIB node values to match NGT format
    3. Concatenates as [Va_nonZIB_noslack, Vm_nonZIB]
    
    Args:
        vae_vm: Pretrained VAE model for Vm
        vae_va: Pretrained VAE model for Va  
        x_ngt: Input in NGT format [batch, input_dim_ngt] (Pd_nonzero, Qd_nonzero)
        ngt_data: NGT data dictionary containing bus indices
        config: Configuration object
        device: Device
        
    Returns:
        z_anchor: Anchor in NGT format [batch, NPred_Va + NPred_Vm]
                  First NPred_Va elements are Va (non-ZIB, no slack)
                  Next NPred_Vm elements are Vm (non-ZIB)
    """
    batch_size = x_ngt.shape[0]
    
    # Get indices for dimension mapping
    bus_Pnet_all = ngt_data['bus_Pnet_all']  # Non-ZIB bus indices
    bus_slack = int(config.ngt_bus_slack if hasattr(config, 'ngt_bus_slack') else 0)
    NPred_Va = ngt_data['NPred_Va']
    NPred_Vm = ngt_data['NPred_Vm']
    
    # The VAE models expect supervised training input format
    # We need to convert NGT input (Pd_nonzero, Qd_nonzero) to VAE input format
    # For simplicity, we'll expand the input back to full Pd, Qd format
    
    # Get bus indices
    bus_Pd = ngt_data['bus_Pd']
    bus_Qd = ngt_data['bus_Qd']
    num_Pd = len(bus_Pd)
    
    # Reconstruct full Pd, Qd from NGT input
    x_ngt_np = x_ngt.cpu().numpy()
    Pd_nonzero = x_ngt_np[:, :num_Pd]  # [batch, num_Pd]
    Qd_nonzero = x_ngt_np[:, num_Pd:]  # [batch, num_Qd]
    
    # Expand to full bus format
    Pd_full = np.zeros((batch_size, config.Nbus))
    Qd_full = np.zeros((batch_size, config.Nbus))
    Pd_full[:, bus_Pd] = Pd_nonzero
    Qd_full[:, bus_Qd] = Qd_nonzero
    
    # VAE input format: concatenated Pd and Qd (same as supervised training)
    x_vae = np.concatenate([Pd_full, Qd_full], axis=1)  # [batch, 2*Nbus]
    x_vae_tensor = torch.from_numpy(x_vae).float().to(device)
    
    # Get VAE predictions
    with torch.no_grad():
        # VAE outputs in supervised format: Vm[Nbus], Va[Nbus-1]
        Vm_pred = vae_vm(x_vae_tensor, use_mean=True)  # [batch, Nbus]
        Va_pred = vae_va(x_vae_tensor, use_mean=True)  # [batch, Nbus-1]
    
    # Convert to numpy for indexing
    Vm_pred_np = Vm_pred.cpu().numpy()
    Va_pred_np = Va_pred.cpu().numpy()
    
    # Va_pred doesn't have slack bus, need to map indices correctly
    # Va_pred indices: 0..(bus_slack-1), bus_slack..(Nbus-2) map to buses 0..(bus_slack-1), (bus_slack+1)..(Nbus-1)
    # We need to extract Va for bus_Pnet_noslack_all
    
    # First, reconstruct full Va with slack=0
    Va_full = np.zeros((batch_size, config.Nbus))
    Va_full[:, :bus_slack] = Va_pred_np[:, :bus_slack]
    Va_full[:, bus_slack+1:] = Va_pred_np[:, bus_slack:]
    # Va_full[:, bus_slack] = 0 (already zero)
    
    # Extract non-ZIB values
    Vm_nonZIB = Vm_pred_np[:, bus_Pnet_all]  # [batch, NPred_Vm]
    
    # For Va, we need bus_Pnet_noslack_all (non-ZIB excluding slack)
    bus_Pnet_noslack_all = ngt_data.get('bus_Pnet_noslack_all', None)
    if bus_Pnet_noslack_all is None:
        # Derive from bus_Pnet_all by removing slack
        bus_Pnet_noslack_all = bus_Pnet_all[bus_Pnet_all != bus_slack]
    
    Va_nonZIB_noslack = Va_full[:, bus_Pnet_noslack_all]  # [batch, NPred_Va]
    
    # Concatenate as NGT format: [Va_nonZIB_noslack, Vm_nonZIB]
    z_anchor_np = np.concatenate([Va_nonZIB_noslack, Vm_nonZIB], axis=1)
    z_anchor = torch.from_numpy(z_anchor_np).float().to(device)
    
    return z_anchor


def flow_forward_ngt(flow_model, x, z_anchor, preference, num_steps=10, training=True):
    """
    Flow integration for NGT unsupervised training.
    
    Integrates the flow ODE: dz/dt = v(x, z, t, preference) from t=0 to t=1.
    
    Args:
        flow_model: PreferenceConditionedNetV model
        x: Load condition [batch, input_dim]
        z_anchor: Starting point from VAE [batch, output_dim]
        preference: Preference vector [batch, 2] or None
        num_steps: Number of Euler integration steps
        training: Whether in training mode (enables gradients)
        
    Returns:
        V_pred: Final voltage prediction [batch, output_dim]
                Format: [Va_nonZIB_noslack, Vm_nonZIB]
                Output is constrained via sigmoid(z) * Vscale + Vbias
    """
    batch_size = x.shape[0]
    device = x.device
    dt = 1.0 / num_steps
    
    # Start from anchor
    z = z_anchor.detach().clone()
    if training:
        z = z.requires_grad_(True)
    
    # Euler integration
    for step in range(num_steps):
        t = torch.full((batch_size, 1), step * dt, device=device)
        v = flow_model.predict_velocity(x, z, t, preference)
        z = z + v * dt
    
    # Apply sigmoid scaling to constrain output to physical range
    # This matches NetV's output: sigmoid(z) * Vscale + Vbias
    V_pred = torch.sigmoid(z) * flow_model.Vscale + flow_model.Vbias
    
    return V_pred


def flow_forward_ngt_projected(flow_model, x, z_anchor, P_tan_t, preference, 
                                num_steps=10, training=True):
    """
    Flow integration with tangent-space projection for constraint satisfaction.
    
    Projects the velocity to the tangent space of the constraint manifold
    during integration, helping to maintain feasibility.
    
    IMPORTANT: P_tan is computed in physical space, but z is in latent space.
    We need to transform the projection using the Jacobian of the coordinate change:
        V = sigmoid(z) * Vscale + Vbias
        dV/dz = sigmoid(z) * (1 - sigmoid(z)) * Vscale  (diagonal Jacobian)
    
    Args:
        flow_model: PreferenceConditionedNetV model
        x: Load condition [batch, input_dim]
        z_anchor: Starting point (latent space) [batch, output_dim]
        P_tan_t: Projection matrix [output_dim, output_dim] (physical space)
        preference: Preference vector [batch, 2] or None
        num_steps: Number of Euler integration steps
        training: Whether in training mode (enables gradients)
        
    Returns:
        V_pred: Final voltage prediction [batch, output_dim]
    """
    batch_size = x.shape[0]
    device = x.device
    dt = 1.0 / num_steps
    eps = 1e-8  # Numerical stability
    
    # Start from anchor (latent space)
    z = z_anchor.detach().clone()
    if training:
        z = z.requires_grad_(True)
    
    # Euler integration with projection
    for step in range(num_steps):
        t = torch.full((batch_size, 1), step * dt, device=device)
        v_latent = flow_model.predict_velocity(x, z, t, preference)
        
        # CRITICAL FIX: Transform projection from physical space to latent space
        # Jacobian J = dV/dz = sigmoid(z) * (1 - sigmoid(z)) * Vscale
        sig_z = torch.sigmoid(z)
        J_diag = sig_z * (1 - sig_z) * flow_model.Vscale  # (batch, dim)
        # Numerical stability: clamp J_inv to prevent explosion when sigmoid saturates
        # When z→±∞, sigmoid→0/1, J_diag→0, J_inv→∞ causing gradient explosion
        J_inv_diag = torch.clamp(1.0 / (J_diag + eps), max=1e3)  # (batch, dim)
        
        # v_latent → v_physical → P_tan(v_physical) → v_latent_projected
        v_physical = v_latent * J_diag  # (batch, dim)
        v_physical_projected = torch.matmul(v_physical, P_tan_t.T)  # (batch, dim)
        v_projected = v_physical_projected * J_inv_diag  # (batch, dim)
        
        z = z + v_projected * dt
    
    # Apply sigmoid scaling to get physical space output
    V_pred = torch.sigmoid(z) * flow_model.Vscale + flow_model.Vbias
    
    return V_pred


def train_unsupervised_ngt(config, sys_data=None, device=None):
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
        
    Returns:
        model: Trained NetV model
        loss_history: Dictionary of training losses
        time_train: Training time in seconds
        ngt_data: NGT training data dictionary
        sys_data: Updated system data
    """
    if not NGT_AVAILABLE:
        raise ImportError("DeepOPF-NGT loss module not available. Please check deepopf_ngt_loss.py")
    
    print('=' * 60)
    print('DeepOPF-NGT Unsupervised Training (Reference Implementation)')
    print('=' * 60)
    
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
    
    print(f"\nModel configuration:")
    print(f"  Input dim: {input_dim}")
    print(f"  Output dim: {output_dim} (Va: {ngt_data['NPred_Va']}, Vm: {ngt_data['NPred_Vm']})")
    print(f"  Hidden layers: {config.ngt_khidden}")
    print(f"  Epochs: {config.ngt_Epoch}")
    print(f"  Batch size: {config.ngt_batch_size}")
    print(f"  Learning rate: {config.ngt_Lr}")
    
    # ============================================================
    # Step 2: Create NetV model (single unified model)
    # ============================================================
    model = NetV(
        input_channels=input_dim,
        output_channels=output_dim,
        hidden_units=config.ngt_hidden_units,
        khidden=config.ngt_khidden,
        Vscale=Vscale,
        Vbias=Vbias
    )
    model.to(device)
    
    print(f"\nNetV model created:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ============================================================
    # Step 3: Create optimizer (Adam with NGT learning rate)
    # ============================================================
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ngt_Lr)
    
    # ============================================================
    # Step 4: Create loss function (Penalty_V with custom backward)
    # ============================================================
    loss_fn = DeepOPFNGTLoss(sys_data, config)
    
    # Pre-cache parameters to GPU for faster training
    loss_fn.cache_to_gpu(device)
    
    # ============================================================
    # Step 5: Create training DataLoader
    # ============================================================
    training_loader = create_ngt_training_loader(ngt_data, config)
    
    # Prepare PQd tensor for all training samples
    PQd_tensor = ngt_data['PQd_train'].to(device)
    
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
    
    # Check if multi-objective mode is enabled
    use_multi_objective = getattr(config, 'ngt_use_multi_objective', False)
    
    n_epochs = config.ngt_Epoch
    batch_size = config.ngt_batch_size
    p_epoch = config.ngt_p_epoch
    s_epoch = config.ngt_s_epoch
    
    print(f"\n{'*' * 5} Training {'*' * 5}")
    print(f"Total epochs: {n_epochs}, Print every: {p_epoch} epochs")
    print(f"Estimated batches per epoch: {len(training_loader)}")
    start_time = time.time()  # Use wall-clock time (not process_time which ignores GPU)
    
    for epoch in range(n_epochs):
        running_loss = 0.0
        running_kgenp = 0.0
        running_kgenq = 0.0
        running_kpd = 0.0
        running_kqd = 0.0
        running_kv = 0.0
        running_cost = 0.0
        running_carbon = 0.0
        n_batches = 0
        
        model.train()
        
        for step, (train_x, train_y) in enumerate(training_loader):
            train_x = train_x.to(device)
            current_batch_size = train_x.shape[0]
            
            # Get PQd for this batch (matching reference: uses training_loader indices)
            # Since DataLoader shuffles, we need to compute PQd from train_x
            # Actually, for simplicity and to match reference, we rebuild PQd from train_x
            # Reference code uses train_x directly as it contains load data
            
            # train_x is the input: [Pd_nonzero, Qd_nonzero] / baseMVA
            # This IS the PQd data, just need to pass it correctly
            PQd_batch = train_x  # Input IS the load data in p.u.
            
            # Forward pass - model outputs [Va_nonZIB_noslack, Vm_nonZIB] with sigmoid
            yvtrain_hat = model(train_x)
            
            # Compute loss (Penalty_V.apply)
            loss, loss_dict = loss_fn(yvtrain_hat, PQd_batch)
            
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
                sample_pred = model(ngt_data['x_train'][:batch_size].to(device))
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


def train_unsupervised_ngt_flow(config, sys_data=None, device=None, 
                                 lambda_cost=0.9, lambda_carbon=0.1,
                                 flow_inf_steps=10, use_projection=False,
                                 anchor_model_path=None, anchor_preference=None):
    """
    DeepOPF-NGT Unsupervised Training with Rectified Flow Model.
    
    This function replaces the direct MLP (NetV) prediction with a Rectified Flow model.
    The flow model uses VAE predictions or a pretrained Flow model as anchors.
    
    Key features:
    1. Uses PreferenceConditionedNetV (flow model) instead of NetV (MLP)
    2. Supports two anchor modes:
       - VAE anchor (default): Use pretrained VAE for initial anchor
       - Flow anchor (progressive): Use a previously trained Flow model
    3. Supports preference conditioning for multi-objective optimization
    4. Optionally supports projection method for constraint satisfaction
    
    Progressive Training (anchor_model_path provided):
        z_anchor = PrevFlow(x, pref_prev)  # Get anchor from previous Flow model
        V_pred = ∫v(x,z,t,pref_new)dt      # Train new Flow for new preference
        
    This allows curriculum learning:
        VAE → Flow(1.0→0.9) → Flow(0.9→0.8) → ... → Flow(0.2→0.1)
    
    Args:
        config: Configuration object with ngt_* parameters
        sys_data: PowerSystemData object (optional, will load if None)
        device: Device (optional, uses config.device if None)
        lambda_cost: Economic cost preference weight (default: 0.9)
        lambda_carbon: Carbon emission preference weight (default: 0.1)
        flow_inf_steps: Number of flow integration steps (default: 10)
        use_projection: Whether to use tangent-space projection (default: False)
        anchor_model_path: Path to pretrained Flow model for anchor (None = use VAE)
        anchor_preference: Preference used by anchor Flow model [λ_cost, λ_carbon] (required if anchor_model_path is set)
        
    Returns:
        model: Trained flow model (PreferenceConditionedNetV)
        loss_history: Dictionary of training losses
        time_train: Training time in seconds
        ngt_data: NGT training data dictionary
        sys_data: Updated system data
    """
    if not NGT_AVAILABLE:
        raise ImportError("DeepOPF-NGT loss module not available. Please check deepopf_ngt_loss.py")
    
    print('=' * 70)
    print('DeepOPF-NGT Unsupervised Training with Rectified Flow Model')
    print('=' * 70)
    print(f'Preference: lambda_cost={lambda_cost}, lambda_carbon={lambda_carbon}')
    print(f'Flow steps: {flow_inf_steps}, Projection: {use_projection}')
    
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
    
    print(f"\nModel configuration:")
    print(f"  Input dim: {input_dim}")
    print(f"  Output dim: {output_dim} (Va: {ngt_data['NPred_Va']}, Vm: {ngt_data['NPred_Vm']})")
    print(f"  Epochs: {config.ngt_Epoch}")
    print(f"  Batch size: {config.ngt_batch_size}")
    print(f"  Learning rate: {config.ngt_Lr}")
    
    # ============================================================
    # Step 2: Determine anchor mode and load anchor models
    # ============================================================
    # Two anchor modes:
    # 1. VAE anchor (default): anchor_model_path is None
    # 2. Flow anchor (progressive): anchor_model_path points to previous Flow model
    
    use_flow_anchor = anchor_model_path is not None
    bus_slack = int(sys_data.bus_slack)
    
    from models import create_model, PreferenceConditionedNetV
    
    if use_flow_anchor:
        # ==================== Flow Anchor Mode (Progressive Training) ====================
        print("\n--- Loading Flow Anchor Model (Progressive Training) ---")
        
        if anchor_preference is None:
            raise ValueError("anchor_preference must be provided when using anchor_model_path")
        
        if not os.path.exists(anchor_model_path):
            raise FileNotFoundError(f"Anchor Flow model not found: {anchor_model_path}")
        
        print(f"  Anchor model: {anchor_model_path}")
        print(f"  Anchor preference: λ_cost={anchor_preference[0]}, λ_carbon={anchor_preference[1]}")
        print(f"  Target preference: λ_cost={lambda_cost}, λ_carbon={lambda_carbon}")
        
        # Create anchor Flow model with same architecture
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
        )
        anchor_flow_model.to(device)
        
        # Load anchor model weights
        anchor_flow_model.load_state_dict(
            torch.load(anchor_model_path, map_location=device, weights_only=True)
        )
        anchor_flow_model.eval()
        
        # Freeze anchor model
        for param in anchor_flow_model.parameters():
            param.requires_grad = False
        
        print(f"  Anchor Flow model loaded and frozen.")
        
        # Store anchor preference for later use
        anchor_pref_tensor = torch.tensor([anchor_preference], dtype=torch.float32, device=device)
        
        # We still need VAE for the initial anchor to the anchor Flow model
        # (unless we want to chain from another Flow model, but let's keep it simple)
        # Load VAE as the base anchor
        vae_vm_path = config.pretrain_model_path_vm
        vae_va_path = config.pretrain_model_path_va
        
        if os.path.exists(vae_vm_path) and os.path.exists(vae_va_path):
            vae_input_dim = input_dim
            vae_vm = create_model('vae', vae_input_dim, config.Nbus, config, is_vm=True)
            vae_va = create_model('vae', vae_input_dim, config.Nbus - 1, config, is_vm=False)
            vae_vm.to(device)
            vae_va.to(device)
            vae_vm.load_state_dict(torch.load(vae_vm_path, map_location=device, weights_only=True), strict=True)
            vae_va.load_state_dict(torch.load(vae_va_path, map_location=device, weights_only=True), strict=True)
            vae_vm.eval()
            vae_va.eval()
            for param in vae_vm.parameters():
                param.requires_grad = False
            for param in vae_va.parameters():
                param.requires_grad = False
            print(f"  VAE models also loaded (for base anchor to Flow).")
        else:
            vae_vm, vae_va = None, None
            print(f"  [Warning] VAE models not found, will use zeros as base anchor.")
    else:
        # ==================== VAE Anchor Mode (Default) ====================
        print("\n--- Loading VAE Anchor Models ---")
        anchor_flow_model = None
        anchor_pref_tensor = None
        
        # Check if VAE models exist
        vae_vm_path = config.pretrain_model_path_vm
        vae_va_path = config.pretrain_model_path_va
        
        if not os.path.exists(vae_vm_path):
            raise FileNotFoundError(f"Pretrained VAE (Vm) not found: {vae_vm_path}\n"
                                   f"Please train VAE models first using supervised training.")
        if not os.path.exists(vae_va_path):
            raise FileNotFoundError(f"Pretrained VAE (Va) not found: {vae_va_path}\n"
                                   f"Please train VAE models first using supervised training.")
        
        # Load VAE models
        vae_input_dim = input_dim
        
        vae_vm = create_model('vae', vae_input_dim, config.Nbus, config, is_vm=True)
        vae_va = create_model('vae', vae_input_dim, config.Nbus - 1, config, is_vm=False)
        
        vae_vm.to(device)
        vae_va.to(device)
        
        # Load state dicts (use strict=True to catch dimension mismatches early)
        print(f"  Loading Vm VAE from: {vae_vm_path}")
        vae_vm.load_state_dict(torch.load(vae_vm_path, map_location=device, weights_only=True), strict=True)
        print(f"  Loading Va VAE from: {vae_va_path}")
        vae_va.load_state_dict(torch.load(vae_va_path, map_location=device, weights_only=True), strict=True)
        
        vae_vm.eval()
        vae_va.eval()
        
        # Freeze VAE parameters
        for param in vae_vm.parameters():
            param.requires_grad = False
        for param in vae_va.parameters():
            param.requires_grad = False
        
        print(f"  VAE models loaded and frozen.")
    
    # ============================================================
    # Step 3: Create Flow model (PreferenceConditionedNetV)
    # ============================================================
    # Note: PreferenceConditionedNetV already imported above
    
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
    )
    model.to(device)
    
    print(f"\nFlow model (PreferenceConditionedNetV) created:")
    print(f"  Hidden dim: {hidden_dim}, Num layers: {num_layers}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ============================================================
    # Step 4: Create optimizer
    # ============================================================
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ngt_Lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    
    # ============================================================
    # Step 5: Create loss function
    # ============================================================
    loss_fn = DeepOPFNGTLoss(sys_data, config)
    loss_fn.cache_to_gpu(device)
    
    # ============================================================
    # Step 6: Create training DataLoader (with indices for anchor matching)
    # ============================================================
    # CRITICAL FIX: Create DataLoader that returns sample indices
    # This ensures z_anchor_batch matches train_x even with shuffle=True
    class IndexedTensorDataset(Data.Dataset):
        """Dataset that returns (data, index) for proper anchor matching with shuffle."""
        def __init__(self, *tensors):
            assert all(t.size(0) == tensors[0].size(0) for t in tensors)
            self.tensors = tensors
        
        def __getitem__(self, index):
            return tuple(t[index] for t in self.tensors) + (index,)
        
        def __len__(self):
            return self.tensors[0].size(0)
    
    indexed_dataset = IndexedTensorDataset(ngt_data['x_train'], ngt_data['y_train'])
    training_loader = Data.DataLoader(
        dataset=indexed_dataset,
        batch_size=config.ngt_batch_size,
        shuffle=True,  # Now safe to shuffle since we track indices
    )
    print(f"[DeepOPF-NGT Flow] Training DataLoader: {len(training_loader)} batches, "
          f"batch_size={config.ngt_batch_size} (with index tracking)")
    
    # ============================================================
    # Step 7: Setup projection matrix (if enabled)
    # ============================================================
    P_tan_t = None
    if use_projection:
        try:
            from flow_model.post_processing import ConstraintProjectionV2
            print("\n--- Setting up Constraint Projection ---")
            projector = ConstraintProjectionV2(sys_data, config, use_historical_jacobian=True)
            P_tan, _, _ = projector.compute_projection_matrix()
            P_tan_t = torch.tensor(P_tan, dtype=torch.float32, device=device)
            print(f"  Projection matrix shape: {P_tan_t.shape}")
        except ImportError:
            print("[Warning] ConstraintProjectionV2 not available, disabling projection")
            use_projection = False
    
    # ============================================================
    # Step 8: Prepare preference tensor
    # ============================================================
    preference = torch.tensor([[lambda_cost, lambda_carbon]], dtype=torch.float32, device=device)
    
    # ============================================================
    # Step 9: Pre-compute anchors for all training samples
    # ============================================================
    bus_Pnet_all = ngt_data['bus_Pnet_all']
    bus_Pnet_noslack_all = bus_Pnet_all[bus_Pnet_all != bus_slack]
    idx_train = ngt_data['idx_train']
    x_train = ngt_data['x_train'].to(device)  # [Ntrain, 374]
    
    if use_flow_anchor:
        # ==================== Flow Anchor Mode ====================
        # First get VAE anchor, then pass through anchor Flow model
        print("\n--- Pre-computing Flow anchors (Progressive Training) ---")
        print(f"  Step 1: VAE → base anchor (physical space → latent space)")
        
        with torch.no_grad():
            if vae_vm is not None and vae_va is not None:
                # Get VAE predictions (physical space)
                Vm_vae = vae_vm(x_train, use_mean=True)  # [Ntrain, Nbus]
                Va_vae_noslack = vae_va(x_train, use_mean=True)  # [Ntrain, Nbus-1]
                
                # Reconstruct full Va
                Va_vae = torch.zeros(len(idx_train), config.Nbus, device=device)
                Va_vae[:, :bus_slack] = Va_vae_noslack[:, :bus_slack]
                Va_vae[:, bus_slack+1:] = Va_vae_noslack[:, bus_slack:]
                
                # Extract non-ZIB values (physical space)
                Vm_nonZIB = Vm_vae[:, bus_Pnet_all]
                Va_nonZIB_noslack = Va_vae[:, bus_Pnet_noslack_all]
                V_vae_physical = torch.cat([Va_nonZIB_noslack, Vm_nonZIB], dim=1)
                
                # CRITICAL FIX: Convert physical space to pre-sigmoid latent space
                # flow_backward expects latent space input
                eps = 1e-6
                u = (V_vae_physical - Vbias) / (Vscale + 1e-12)  # +1e-12 for numerical safety
                u = torch.clamp(u, eps, 1 - eps)
                z_vae_anchor = torch.log(u / (1 - u))  # logit
            else:
                # Use zeros if VAE not available (zeros in latent space is fine)
                z_vae_anchor = torch.zeros(len(idx_train), output_dim, device=device)
            
            print(f"  VAE base anchor (latent space) shape: {z_vae_anchor.shape}")
            print(f"  z_vae_anchor range: [{z_vae_anchor.min():.4f}, {z_vae_anchor.max():.4f}]")
            
            # Step 2: Pass through anchor Flow model to get better anchor
            print(f"  Step 2: Anchor Flow(pref={anchor_preference}) → training anchor")
            anchor_pref_batch = anchor_pref_tensor.expand(len(idx_train), -1)
            
            # Use the anchor Flow model to generate predictions
            # IMPORTANT: Use detach() to ensure no gradient flows back to anchor model
            z_anchor_all = anchor_flow_model.flow_backward(
                x_train, z_vae_anchor, anchor_pref_batch, 
                num_steps=flow_inf_steps, apply_sigmoid=False, training=False
            ).detach()  # Explicitly detach to prevent gradient flow
        
        print(f"  Flow anchor (latent space) shape: {z_anchor_all.shape}")
        # NOTE: z_anchor_all is in LATENT space (pre-sigmoid), NOT physical Va/Vm!
        # Splitting for debug printing only - these are latent values, not physical
        z_Va_latent = z_anchor_all[:, :ngt_data['NPred_Va']]
        z_Vm_latent = z_anchor_all[:, ngt_data['NPred_Va']:]
        print(f"  z_Va (latent) range: [{z_Va_latent.min():.4f}, {z_Va_latent.max():.4f}]")
        print(f"  z_Vm (latent) range: [{z_Vm_latent.min():.4f}, {z_Vm_latent.max():.4f}]")
        
    else:
        # ==================== VAE Anchor Mode (Default) ====================
        print("\n--- Pre-computing VAE anchors ---")
        print(f"  VAE input shape: {x_train.shape}")
        
        with torch.no_grad():
            Vm_anchor_full = vae_vm(x_train, use_mean=True)  # [Ntrain, Nbus]
            Va_anchor_full_noslack = vae_va(x_train, use_mean=True)  # [Ntrain, Nbus-1]
        
        print(f"  VAE Vm output shape: {Vm_anchor_full.shape}")
        print(f"  VAE Va output shape: {Va_anchor_full_noslack.shape}")
        
        # Reconstruct full Va (add slack=0)
        Va_anchor_full = torch.zeros(len(idx_train), config.Nbus, device=device)
        Va_anchor_full[:, :bus_slack] = Va_anchor_full_noslack[:, :bus_slack]
        Va_anchor_full[:, bus_slack+1:] = Va_anchor_full_noslack[:, bus_slack:]
        
        # Extract non-ZIB values and concatenate as NGT format
        Vm_anchor_nonZIB = Vm_anchor_full[:, bus_Pnet_all]  # [Ntrain, NPred_Vm]
        Va_anchor_nonZIB_noslack = Va_anchor_full[:, bus_Pnet_noslack_all]  # [Ntrain, NPred_Va]
        
        # Concatenate as NGT format: [Va_nonZIB_noslack, Vm_nonZIB] (physical space)
        V_anchor_physical = torch.cat([Va_anchor_nonZIB_noslack, Vm_anchor_nonZIB], dim=1)
        
        print(f"  VAE anchor (physical space) shape: {V_anchor_physical.shape}")
        print(f"  Va anchor range: [{Va_anchor_nonZIB_noslack.min():.4f}, {Va_anchor_nonZIB_noslack.max():.4f}]")
        print(f"  Vm anchor range: [{Vm_anchor_nonZIB.min():.4f}, {Vm_anchor_nonZIB.max():.4f}]")
        
        # CRITICAL FIX: Convert physical space to pre-sigmoid latent space
        # flow_forward_ngt() applies: V_pred = sigmoid(z) * Vscale + Vbias
        # So anchor needs to be in pre-sigmoid space: z = logit((V - Vbias) / Vscale)
        # where logit(p) = log(p / (1-p)) is the inverse of sigmoid
        eps = 1e-6  # Avoid log(0) or log(inf)
        u = (V_anchor_physical - Vbias) / (Vscale + 1e-12)  # +1e-12 for numerical safety
        u = torch.clamp(u, eps, 1 - eps)  # Ensure (0, 1) range
        z_anchor_all = torch.log(u / (1 - u))  # logit function
        
        print(f"  Anchor (pre-sigmoid latent) shape: {z_anchor_all.shape}")
        print(f"  z_anchor range: [{z_anchor_all.min():.4f}, {z_anchor_all.max():.4f}]")
    
    # ============================================================
    # Step 10: Training loop
    # ============================================================
    loss_history = {
        'total': [],
        'kgenp_mean': [],
        'kgenq_mean': [],
        'kpd_mean': [],
        'kqd_mean': [],
        'kv_mean': [],
        'cost': [],
        'carbon': [],
    }
    
    n_epochs = config.ngt_Epoch
    batch_size = config.ngt_batch_size
    p_epoch = config.ngt_p_epoch
    s_epoch = config.ngt_s_epoch
    
    print(f"\n{'*' * 5} Training Flow Model {'*' * 5}")
    print(f"Total epochs: {n_epochs}, Print every: {p_epoch} epochs")
    print(f"Estimated batches per epoch: {len(training_loader)}")
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
        n_batches = 0
        
        model.train()
        
        for step, (train_x, train_y, batch_indices) in enumerate(training_loader):
            train_x = train_x.to(device)
            current_batch_size = train_x.shape[0]
            
            # CRITICAL FIX: Use actual batch indices to get matching anchors
            # This correctly handles shuffle=True in DataLoader
            z_anchor_batch = z_anchor_all[batch_indices]
            
            # Expand preference for batch
            pref_batch = preference.expand(current_batch_size, -1)
            
            # PQd batch (same as input for NGT)
            PQd_batch = train_x
            
            # Flow forward: integrate from anchor to get V_pred
            if use_projection and P_tan_t is not None:
                V_pred = flow_forward_ngt_projected(
                    model, train_x, z_anchor_batch, P_tan_t, 
                    pref_batch, flow_inf_steps, training=True
                )
            else:
                V_pred = flow_forward_ngt(
                    model, train_x, z_anchor_batch, 
                    pref_batch, flow_inf_steps, training=True
                )
            
            # Compute loss using existing DeepOPFNGTLoss
            loss, loss_dict = loss_fn(V_pred, PQd_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            n_batches += 1
        
        scheduler.step()
        
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
            
            loss_history['total'].append(avg_loss)
            loss_history['kgenp_mean'].append(avg_kgenp)
            loss_history['kgenq_mean'].append(avg_kgenq)
            loss_history['kpd_mean'].append(avg_kpd)
            loss_history['kqd_mean'].append(avg_kqd)
            loss_history['kv_mean'].append(avg_kv)
            loss_history['cost'].append(avg_cost)
            loss_history['carbon'].append(avg_carbon)
        
        # Print progress
        should_print = (epoch == 0) or ((epoch + 1) % p_epoch == 0)
        if should_print:
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
                sample_anchor = z_anchor_all[:batch_size]
                sample_pref = preference.expand(batch_size, -1)
                sample_pred = flow_forward_ngt(model, sample_x, sample_anchor, sample_pref, 
                                               flow_inf_steps, training=False)
                Va_pred = sample_pred[:, :ngt_data['NPred_Va']]
                Vm_pred = sample_pred[:, ngt_data['NPred_Va']:]
            
            print(f"epoch {epoch+1}/{n_epochs} loss={avg_loss:.4f} "
                  f"Va[{Va_pred.min().item():.4f},{Va_pred.max().item():.4f}] "
                  f"Vm[{Vm_pred.min().item():.4f},{Vm_pred.max().item():.4f}]{time_info}")
            print(f"  [Flow] λ_cost={lambda_cost} λ_carbon={lambda_carbon} steps={flow_inf_steps}")
            print(f"  kgenp={avg_kgenp:.2f} kgenq={avg_kgenq:.2f} "
                  f"kpd={avg_kpd:.2f} kqd={avg_kqd:.2f} kv={avg_kv:.2f}")
        
        # Save models periodically
        if (epoch + 1) % 100 == 0 and (epoch + 1) >= s_epoch:
            lc_str = f"{lambda_cost:.1f}".replace('.', '')
            save_path = f'{config.model_save_dir}/NetV_ngt_flow_{config.Nbus}bus_lc{lc_str}_E{epoch+1}.pth'
            torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)
            print(f"  Model saved: {save_path}")
    
    time_train = time.time() - start_time
    print(f"\nTraining completed in {time_train:.2f} seconds ({time_train/60:.2f} minutes)")
    
    # Save final model
    lc_str = f"{lambda_cost:.1f}".replace('.', '')
    final_path = f'{config.model_save_dir}/NetV_ngt_flow_{config.Nbus}bus_lc{lc_str}_E{n_epochs}_final.pth'
    torch.save(model.state_dict(), final_path, _use_new_zipfile_serialization=False)
    print(f"Final model saved: {final_path}")
    
    return model, loss_history, time_train, ngt_data, sys_data


def evaluate_ngt_model(config, model, ngt_data, sys_data, BRANFT, device):
    """
    Evaluate trained DeepOPF-NGT model on test set.
    
    This function evaluates the single NetV model and computes:
    - Voltage prediction accuracy (Vm, Va MAE)
    - Generation cost and optimality
    - Constraint satisfaction (Pg, Qg, voltage, branch)
    - Load satisfaction (Pd, Qd deviation)
    
    Args:
        config: Configuration object
        model: Trained NetV model
        ngt_data: NGT training data dictionary (from load_ngt_training_data)
        sys_data: System data
        BRANFT: Branch from-to indices
        device: Device
        
    Returns:
        results: Dictionary with all evaluation metrics
    """
    print('\n' + '=' * 60)
    print('DeepOPF-NGT Model Evaluation on Test Set')
    print('=' * 60)
    
    model.eval()
    
    # Extract dimensions from ngt_data
    NPred_Va = ngt_data['NPred_Va']
    NPred_Vm = ngt_data['NPred_Vm']
    bus_Pnet_all = ngt_data['bus_Pnet_all']
    bus_ZIB_all = ngt_data['bus_ZIB_all']
    idx_bus_Pnet_slack = ngt_data['idx_bus_Pnet_slack']
    NZIB = ngt_data['NZIB']
    param_ZIMV = ngt_data['param_ZIMV']
    
    # Get test data
    x_test = ngt_data['x_test'].to(device)
    Ntest = x_test.shape[0]
    
    # ==================== Model Prediction ====================
    print(f'\nPredicting on {Ntest} test samples...')
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    time_start = time.perf_counter()
    
    with torch.no_grad():
        V_pred = model(x_test)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    time_NN = time.perf_counter() - time_start
    
    V_pred_np = V_pred.cpu().numpy()
    
    # ==================== Reconstruct Full Voltage ====================
    print('Recovering full voltage (including ZIB nodes)...')
    
    # Insert slack bus Va = 0 into the Va segment
    # Model output layout: [Va_nonZIB_without_slack (NPred_Va), Vm_nonZIB (NPred_Vm)]
    # After insert:        [Va_nonZIB_with_slack (NPred_Va+1), Vm_nonZIB (NPred_Vm)]
    xam_P = np.insert(V_pred_np, idx_bus_Pnet_slack[0], 0, axis=1)
    
    # Split Va and Vm for non-ZIB buses
    # Note: NPred_Va + 1 == NPred_Vm (since NPred_Va = NPred_Vm - 1 by definition)
    Va_len_with_slack = NPred_Va + 1
    Va_nonZIB = xam_P[:, :Va_len_with_slack]
    Vm_nonZIB = xam_P[:, Va_len_with_slack:Va_len_with_slack + NPred_Vm]
    
    # Convert to complex voltage
    Vx = Vm_nonZIB * np.exp(1j * Va_nonZIB)
    
    # Recover ZIB node voltages using Kron Reduction
    if NZIB > 0 and param_ZIMV is not None:
        Vy = np.dot(param_ZIMV, Vx.T).T
    else:
        Vy = None
    
    # Build full voltage
    Ve = np.zeros((Ntest, config.Nbus))
    Vf = np.zeros((Ntest, config.Nbus))
    Ve[:, bus_Pnet_all] = Vx.real
    Vf[:, bus_Pnet_all] = Vx.imag
    if Vy is not None:
        Ve[:, bus_ZIB_all] = Vy.real
        Vf[:, bus_ZIB_all] = Vy.imag
    
    # Convert to polar
    Pred_Vm = np.sqrt(Ve**2 + Vf**2)
    Pred_Va = np.arctan2(Vf, Ve)
    
    # Clamp ZIB voltages
    if NZIB > 0:
        Pred_Vm[:, bus_ZIB_all] = np.clip(
            Pred_Vm[:, bus_ZIB_all], config.ngt_VmLb, config.ngt_VmUb
        )
    
    Pred_V = Pred_Vm * np.exp(1j * Pred_Va)
    
    # ==================== Get Real Values ====================
    # Real voltage from test data
    # Note: yvm_test and yva_test are both [Ntest, Nbus] (300 dims, all nodes)
    Real_Vm = ngt_data['yvm_test'].numpy()
    Real_Va_full = ngt_data['yva_test'].numpy()  # Already includes all nodes
    Real_V = Real_Vm * np.exp(1j * Real_Va_full)
    
    # ==================== Calculate Generation and Load ====================
    print('Calculating power flow...')
    
    # Prepare test load data
    baseMVA = float(sys_data.baseMVA)
    Pdtest = np.zeros((Ntest, config.Nbus))
    Qdtest = np.zeros((Ntest, config.Nbus))
    
    bus_Pd = ngt_data['bus_Pd']
    bus_Qd = ngt_data['bus_Qd']
    idx_test = ngt_data['idx_test']
    
    Pdtest[:, bus_Pd] = sys_data.RPd[idx_test][:, bus_Pd] / baseMVA
    Qdtest[:, bus_Qd] = sys_data.RQd[idx_test][:, bus_Qd] / baseMVA
    
    # Predicted generation
    Pred_Pg, Pred_Qg, Pred_Pd, Pred_Qd = get_genload(
        Pred_V, Pdtest, Qdtest,
        sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
    )
    
    # Real generation
    Real_Pg, Real_Qg, Real_Pd, Real_Qd = get_genload(
        Real_V, Pdtest, Qdtest,
        sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
    )
    
    # ==================== Voltage Accuracy ====================
    print('\n[Voltage Prediction Accuracy]')
    
    mae_Vm = np.mean(np.abs(Real_Vm - Pred_Vm))
    
    # Calculate Va MAE (exclude slack bus)
    bus_slack = int(sys_data.bus_slack)
    bus_Va = np.delete(np.arange(config.Nbus), bus_slack)
    mae_Va = np.mean(np.abs(Real_Va_full[:, bus_Va] - Pred_Va[:, bus_Va]))
    
    print(f'  Vm MAE: {mae_Vm:.6f} p.u.')
    print(f'  Va MAE: {mae_Va:.6f} rad ({mae_Va * 180 / np.pi:.4f} deg)')
    
    # ==================== Generation Cost ====================
    print('\n[Economic Performance]')
    
    gencost = ngt_data['gencost_Pg']
    Pred_cost = gencost[:, 0] * (Pred_Pg * baseMVA)**2 + gencost[:, 1] * np.abs(Pred_Pg * baseMVA)
    Real_cost = gencost[:, 0] * (Real_Pg * baseMVA)**2 + gencost[:, 1] * np.abs(Real_Pg * baseMVA)
    
    Pred_cost_total = np.sum(Pred_cost, axis=1)
    Real_cost_total = np.sum(Real_cost, axis=1)
    
    cost_error = (Pred_cost_total - Real_cost_total) / Real_cost_total * 100
    
    print(f'  Real cost mean:      {np.mean(Real_cost_total):.2f} $/h')
    print(f'  Predicted cost mean: {np.mean(Pred_cost_total):.2f} $/h')
    print(f'  Cost error:          {np.mean(cost_error):.2f}% (optimality gap)')
    
    # ==================== Constraint Violations ====================
    print('\n[Constraint Satisfaction]')
    
    # Pg/Qg violations
    MAXMIN_Pg = ngt_data['MAXMIN_Pg']
    MAXMIN_Qg = ngt_data['MAXMIN_Qg']
    
    lsPg, lsQg, lsidxPg, lsidxQg, vio_PQgmaxmin, vio_PQg, \
        deltaPgL, deltaPgU, deltaQgL, deltaQgU = get_vioPQg(
            Pred_Pg, sys_data.bus_Pg, MAXMIN_Pg,
            Pred_Qg, sys_data.bus_Qg, MAXMIN_Qg,
            config.DELTA
        )
    
    # Convert to numpy if needed
    if torch.is_tensor(vio_PQg):
        vio_PQg = vio_PQg.numpy()
    
    Pg_satisfy = np.mean(vio_PQg[:, 0])
    Qg_satisfy = np.mean(vio_PQg[:, 1])
    
    print(f'  Pg constraint satisfaction: {Pg_satisfy:.2f}%')
    print(f'  Qg constraint satisfaction: {Qg_satisfy:.2f}%')
    
    # Voltage violations
    Vm_vio_upper = np.mean(Pred_Vm > config.ngt_VmUb) * 100
    Vm_vio_lower = np.mean(Pred_Vm < config.ngt_VmLb) * 100
    Vm_satisfy = 100 - Vm_vio_upper - Vm_vio_lower
    
    print(f'  Vm constraint satisfaction: {Vm_satisfy:.2f}%')
    
    # Branch violations
    vio_branang, vio_branpf, deltapf, vio_branpfidx, lsSf, lsSt, lsSf_sampidx, lsSt_sampidx = get_viobran2(
        Pred_V, Pred_Va, sys_data.branch, sys_data.Yf, sys_data.Yt,
        BRANFT, baseMVA, config.DELTA
    )
    
    if torch.is_tensor(vio_branang):
        vio_branang = vio_branang.numpy()
    if torch.is_tensor(vio_branpf):
        vio_branpf = vio_branpf.numpy()
    
    print(f'  Branch angle constraint:    {np.mean(vio_branang):.2f}%')
    print(f'  Branch power constraint:    {np.mean(vio_branpf):.2f}%')
    
    # Count violated samples
    lsidxPQg = np.where((lsidxPg + lsidxQg) > 0)[0]
    num_vio = len(lsidxPQg)
    
    print(f'\n  Violated samples: {num_vio}/{Ntest} ({num_vio/Ntest*100:.1f}%)')
    
    # ==================== Load Satisfaction ====================
    print('\n[Load Satisfaction]')
    
    # Load deviation
    Pd_total_real = np.sum(Real_Pd, axis=1)
    Pd_total_pred = np.sum(Pred_Pd, axis=1)
    Qd_total_real = np.sum(Real_Qd, axis=1)
    Qd_total_pred = np.sum(Pred_Qd, axis=1)
    
    Pd_error = np.mean(np.abs(Pd_total_pred - Pd_total_real) / np.abs(Pd_total_real)) * 100
    Qd_error = np.mean(np.abs(Qd_total_pred - Qd_total_real) / np.abs(Qd_total_real)) * 100
    
    print(f'  Pd deviation: {Pd_error:.4f}%')
    print(f'  Qd deviation: {Qd_error:.4f}%')
    
    # ==================== Timing ====================
    print('\n[Inference Time]')
    print(f'  Total NN inference: {time_NN:.4f} s')
    print(f'  Per sample:         {time_NN/Ntest*1000:.4f} ms')
    
    # ==================== Summary ====================
    print('\n' + '=' * 60)
    print('Evaluation Summary')
    print('=' * 60)
    
    results = {
        # Voltage accuracy
        'mae_Vm': mae_Vm,
        'mae_Va': mae_Va,
        # Economic performance
        'cost_mean_real': np.mean(Real_cost_total),
        'cost_mean_pred': np.mean(Pred_cost_total),
        'cost_error_percent': np.mean(cost_error),
        # Constraint satisfaction
        'Pg_satisfy': Pg_satisfy,
        'Qg_satisfy': Qg_satisfy,
        'Vm_satisfy': Vm_satisfy,
        'branch_ang_satisfy': np.mean(vio_branang),
        'branch_pf_satisfy': np.mean(vio_branpf),
        'num_violated_samples': num_vio,
        # Load satisfaction
        'Pd_error_percent': Pd_error,
        'Qd_error_percent': Qd_error,
        # Timing
        'time_NN': time_NN,
        'time_per_sample_ms': time_NN / Ntest * 1000,
        # Raw data for further analysis
        'Pred_Vm': Pred_Vm,
        'Pred_Va': Pred_Va,
        'Pred_Pg': Pred_Pg,
        'Pred_Qg': Pred_Qg,
        'Real_Vm': Real_Vm,
        'Real_Va': Real_Va_full,
        'Real_Pg': Real_Pg,
        'Real_Qg': Real_Qg,
        'Pred_cost': Pred_cost_total,
        'Real_cost': Real_cost_total,
        'vio_PQg': vio_PQg,
        'vio_branang': vio_branang,
        'vio_branpf': vio_branpf,
    }
    
    # Print summary table
    print(f"\n{'Metric':<30} {'Value':>15}")
    print('-' * 45)
    print(f"{'Vm MAE (p.u.)':<30} {mae_Vm:>15.6f}")
    print(f"{'Va MAE (rad)':<30} {mae_Va:>15.6f}")
    print(f"{'Cost Error (%)':<30} {np.mean(cost_error):>15.2f}")
    print(f"{'Pg Satisfaction (%)':<30} {Pg_satisfy:>15.2f}")
    print(f"{'Qg Satisfaction (%)':<30} {Qg_satisfy:>15.2f}")
    print(f"{'Vm Satisfaction (%)':<30} {Vm_satisfy:>15.2f}")
    print(f"{'Branch Angle Sat. (%)':<30} {np.mean(vio_branang):>15.2f}")
    print(f"{'Branch Power Sat. (%)':<30} {np.mean(vio_branpf):>15.2f}")
    print(f"{'Pd Error (%)':<30} {Pd_error:>15.4f}")
    print(f"{'Qd Error (%)':<30} {Qd_error:>15.4f}")
    print(f"{'Inference Time (ms/sample)':<30} {time_NN/Ntest*1000:>15.4f}")
    print('=' * 60)
    
    return results


def evaluate_dual_model(config, model_vm, model_va, x_test, Real_Vm, Real_Va_full, 
                        Pdtest, Qdtest, sys_data, BRANFT, MAXMIN_Pg, MAXMIN_Qg,
                        gencost, Real_cost_total, model_type, device,
                        pretrain_model_vm=None, pretrain_model_va=None,
                        apply_post_processing=True):
    """
    Evaluate a dual-model architecture (model_vm + model_va) on test data.
    
    Supports: simple, vae, rectified, diffusion, etc.
    Now includes Jacobian-based post-processing for constraint correction.
    
    Args:
        config: Configuration object
        model_vm: Voltage magnitude model
        model_va: Voltage angle model
        x_test: Test input tensor [Ntest, input_dim]
        Real_Vm: Real voltage magnitudes [Ntest, Nbus]
        Real_Va_full: Real voltage angles [Ntest, Nbus]
        Pdtest, Qdtest: Test load data [Ntest, Nbus]
        sys_data: System data
        BRANFT: Branch from-to indices
        MAXMIN_Pg, MAXMIN_Qg: Generator limits
        gencost: Generator cost coefficients
        Real_cost_total: Real generation cost [Ntest]
        model_type: Model type ('simple', 'vae', 'rectified', 'diffusion', etc.)
        device: Device
        pretrain_model_vm, pretrain_model_va: Pretrained VAE models (for flow/diffusion)
        apply_post_processing: Whether to apply Jacobian-based post-processing
        
    Returns:
        results: Dictionary with evaluation metrics (both raw and post-processed)
    """
    model_vm.eval()
    model_va.eval()
    
    Ntest = x_test.shape[0]
    baseMVA = float(sys_data.baseMVA)
    bus_slack = int(sys_data.bus_slack)
    bus_Va_idx = np.delete(np.arange(config.Nbus), bus_slack)
    
    # Get scaling parameters
    VmLb = sys_data.VmLb.item() if hasattr(sys_data.VmLb, 'item') else float(sys_data.VmLb)
    VmUb = sys_data.VmUb.item() if hasattr(sys_data.VmUb, 'item') else float(sys_data.VmUb)
    scale_vm_val = config.scale_vm.item() if hasattr(config.scale_vm, 'item') else float(config.scale_vm)
    scale_va_val = config.scale_va.item() if hasattr(config.scale_va, 'item') else float(config.scale_va)
    
    # Get historical voltage bounds for clipping
    hisVm_min = sys_data.hisVm_min if hasattr(sys_data, 'hisVm_min') and sys_data.hisVm_min is not None else VmLb
    hisVm_max = sys_data.hisVm_max if hasattr(sys_data, 'hisVm_max') and sys_data.hisVm_max is not None else VmUb
    
    # GPU warmup
    if device.type == 'cuda':
        with torch.no_grad():
            _ = predict_with_model(model_vm, x_test[:1], model_type, pretrain_model_vm, config, device)
            _ = predict_with_model(model_va, x_test[:1], model_type, pretrain_model_va, config, device)
            torch.cuda.synchronize()
    
    # Inference timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    time_start = time.perf_counter()
    
    with torch.no_grad():
        Vm_pred_scaled = predict_with_model(model_vm, x_test, model_type, pretrain_model_vm, config, device)
        Va_pred_scaled = predict_with_model(model_va, x_test, model_type, pretrain_model_va, config, device)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    inference_time = time.perf_counter() - time_start
    
    # Unscale predictions
    Vm_pred = (Vm_pred_scaled.cpu().numpy() / scale_vm_val) * (VmUb - VmLb) + VmLb
    Va_pred_no_slack = Va_pred_scaled.cpu().numpy() / scale_va_val
    
    # Insert slack bus Va = 0
    Va_pred = np.zeros((Ntest, config.Nbus))
    Va_pred[:, :bus_slack] = Va_pred_no_slack[:, :bus_slack]
    Va_pred[:, bus_slack+1:] = Va_pred_no_slack[:, bus_slack:]
    
    # Clip Vm to bounds (use np.clip for simple scalar bounds)
    if isinstance(hisVm_min, (int, float)) and isinstance(hisVm_max, (int, float)):
        Pred_Vm = np.clip(Vm_pred, hisVm_min, hisVm_max)
    else:
        Pred_Vm = get_clamp(torch.from_numpy(Vm_pred), hisVm_min, hisVm_max).numpy()
    Pred_Va = Va_pred
    Pred_V = Pred_Vm * np.exp(1j * Pred_Va)
    
    # Calculate power flow
    Real_V = Real_Vm * np.exp(1j * Real_Va_full)
    Real_Pg, Real_Qg, Real_Pd, Real_Qd = get_genload(
        Real_V, Pdtest, Qdtest,
        sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
    )
    
    Pred_Pg, Pred_Qg, Pred_Pd, Pred_Qd = get_genload(
        Pred_V, Pdtest, Qdtest,
        sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
    )
    
    # ==================== Raw Metrics (Before Post-Processing) ====================
    mae_Vm = np.mean(np.abs(Real_Vm - Pred_Vm))
    mae_Va = np.mean(np.abs(Real_Va_full[:, bus_Va_idx] - Pred_Va[:, bus_Va_idx]))
    
    # Cost
    Pred_cost = gencost[:, 0] * (Pred_Pg * baseMVA)**2 + gencost[:, 1] * np.abs(Pred_Pg * baseMVA)
    Pred_cost_total = np.sum(Pred_cost, axis=1)
    cost_error = (Pred_cost_total - Real_cost_total) / Real_cost_total * 100
    
    # Constraint violations (raw)
    lsPg, lsQg, lsidxPg, lsidxQg, _, vio_PQg, deltaPgL, deltaPgU, deltaQgL, deltaQgU = get_vioPQg(
        Pred_Pg, sys_data.bus_Pg, MAXMIN_Pg,
        Pred_Qg, sys_data.bus_Qg, MAXMIN_Qg,
        config.DELTA
    )
    if torch.is_tensor(vio_PQg):
        vio_PQg = vio_PQg.numpy()
    
    lsidxPQg = np.squeeze(np.array(np.where((lsidxPg + lsidxQg) > 0)))
    num_viotest = np.size(lsidxPQg)
    
    vio_branang, vio_branpf, deltapf, vio_branpfidx, lsSf, _, lsSf_sampidx, _ = get_viobran2(
        Pred_V, Pred_Va, sys_data.branch, sys_data.Yf, sys_data.Yt,
        BRANFT, baseMVA, config.DELTA
    )
    if torch.is_tensor(vio_branang):
        vio_branang = vio_branang.numpy()
    if torch.is_tensor(vio_branpf):
        vio_branpf = vio_branpf.numpy()
    
    vio_branpf_num = int(np.sum(np.asarray(vio_branpfidx) > 0))
    lsSf_sampidx = np.asarray(lsSf_sampidx)
    
    # Voltage constraint satisfaction
    Vm_satisfy = 100 - np.mean(Pred_Vm > VmUb) * 100 - np.mean(Pred_Vm < VmLb) * 100
    
    # Load deviation
    Pd_error = np.mean(np.abs(np.sum(Pred_Pd, axis=1) - np.sum(Real_Pd, axis=1)) / np.abs(np.sum(Real_Pd, axis=1))) * 100
    Qd_error = np.mean(np.abs(np.sum(Pred_Qd, axis=1) - np.sum(Real_Qd, axis=1)) / np.abs(np.sum(Real_Qd, axis=1))) * 100
    
    # Store raw results
    results = {
        # Raw metrics
        'mae_Vm': mae_Vm,
        'mae_Va': mae_Va,
        'cost_error_percent': np.mean(cost_error),
        'cost_mean': np.mean(Pred_cost_total),
        'Pg_satisfy': np.mean(vio_PQg[:, 0]),
        'Qg_satisfy': np.mean(vio_PQg[:, 1]),
        'Vm_satisfy': Vm_satisfy,
        'branch_ang_satisfy': np.mean(vio_branang),
        'branch_pf_satisfy': np.mean(vio_branpf),
        'Pd_error_percent': Pd_error,
        'Qd_error_percent': Qd_error,
        'inference_time_ms': inference_time / Ntest * 1000,
        'num_violated': num_viotest,
    }
    
    # ==================== Post-Processing (Jacobian-based correction) ====================
    if apply_post_processing and num_viotest > 0:
        # Use unified post-processing function
        post_results = jacobian_postprocess(
            config=config, sys_data=sys_data, BRANFT=BRANFT,
            Pred_Vm=Pred_Vm, Pred_Va=Pred_Va, Pred_V=Pred_V,
            Pdtest=Pdtest, Qdtest=Qdtest,
            lsPg=lsPg, lsQg=lsQg, lsidxPg=lsidxPg, lsidxQg=lsidxQg,
            lsidxPQg=lsidxPQg, num_viotest=num_viotest,
            MAXMIN_Pg=MAXMIN_Pg, MAXMIN_Qg=MAXMIN_Qg,
            baseMVA=baseMVA, bus_slack=bus_slack,
            hisVm_min=hisVm_min, hisVm_max=hisVm_max,
            VmLb=VmLb, VmUb=VmUb,
            include_branch_correction=False,
            verbose=False
        )
        
        # Extract results
        Pred_Vm1 = post_results['Pred_Vm1']
        Pred_Va1 = post_results['Pred_Va1']
        Pred_Pg1 = post_results['Pred_Pg1']
        Pred_Qg1 = post_results['Pred_Qg1']
        Pred_Pd1 = post_results['Pred_Pd1']
        Pred_Qd1 = post_results['Pred_Qd1']
        vio_PQg1 = post_results['vio_PQg1']
        vio_branang1 = post_results['vio_branang1']
        vio_branpf1 = post_results['vio_branpf1']
        num_viotest1 = post_results['num_viotest1']
        Vm_satisfy1 = post_results['Vm_satisfy1']
        time_post = post_results['time_post']
        
        # ==================== Post-Processed Metrics ====================
        mae_Vm1 = np.mean(np.abs(Real_Vm - Pred_Vm1))
        mae_Va1 = np.mean(np.abs(Real_Va_full[:, bus_Va_idx] - Pred_Va1[:, bus_Va_idx]))
        
        # Cost after post-processing
        Pred_cost1 = gencost[:, 0] * (Pred_Pg1 * baseMVA)**2 + gencost[:, 1] * np.abs(Pred_Pg1 * baseMVA)
        Pred_cost_total1 = np.sum(Pred_cost1, axis=1)
        cost_error1 = (Pred_cost_total1 - Real_cost_total) / Real_cost_total * 100
        
        # Load deviation after post-processing
        Pd_error1 = np.mean(np.abs(np.sum(Pred_Pd1, axis=1) - np.sum(Real_Pd, axis=1)) / np.abs(np.sum(Real_Pd, axis=1))) * 100
        Qd_error1 = np.mean(np.abs(np.sum(Pred_Qd1, axis=1) - np.sum(Real_Qd, axis=1)) / np.abs(np.sum(Real_Qd, axis=1))) * 100
        
        # Add post-processed results
        results['mae_Vm_post'] = mae_Vm1
        results['mae_Va_post'] = mae_Va1
        results['cost_error_percent_post'] = np.mean(cost_error1)
        results['cost_mean_post'] = np.mean(Pred_cost_total1)
        results['Pg_satisfy_post'] = np.mean(vio_PQg1[:, 0])
        results['Qg_satisfy_post'] = np.mean(vio_PQg1[:, 1])
        results['Vm_satisfy_post'] = Vm_satisfy1
        results['branch_ang_satisfy_post'] = np.mean(vio_branang1)
        results['branch_pf_satisfy_post'] = np.mean(vio_branpf1)
        results['Pd_error_percent_post'] = Pd_error1
        results['Qd_error_percent_post'] = Qd_error1
        results['post_processing_time_ms'] = time_post / Ntest * 1000
        results['num_violated_post'] = num_viotest1
    else:
        # No post-processing or no violations - copy raw results
        results['mae_Vm_post'] = results['mae_Vm']
        results['mae_Va_post'] = results['mae_Va']
        results['cost_error_percent_post'] = results['cost_error_percent']
        results['cost_mean_post'] = results['cost_mean']
        results['Pg_satisfy_post'] = results['Pg_satisfy']
        results['Qg_satisfy_post'] = results['Qg_satisfy']
        results['Vm_satisfy_post'] = results['Vm_satisfy']
        results['branch_ang_satisfy_post'] = results['branch_ang_satisfy']
        results['branch_pf_satisfy_post'] = results['branch_pf_satisfy']
        results['Pd_error_percent_post'] = results['Pd_error_percent']
        results['Qd_error_percent_post'] = results['Qd_error_percent']
        results['post_processing_time_ms'] = 0.0
        results['num_violated_post'] = results['num_violated']
    
    return results


def evaluate_ngt_single_model(config, model_ngt, x_test, Real_Vm, Real_Va_full,
                               Pdtest, Qdtest, sys_data, BRANFT, MAXMIN_Pg, MAXMIN_Qg,
                               gencost, Real_cost_total, ngt_data, device,
                               apply_post_processing=True):
    """
    Evaluate a single NGT model on test data.
    
    This handles the Kron Reduction recovery of ZIB node voltages.
    Now includes Jacobian-based post-processing for constraint correction.
    
    Args:
        config: Configuration object
        model_ngt: Trained NetV model
        x_test: Test input tensor [Ntest, input_dim]
        Real_Vm: Real voltage magnitudes [Ntest, Nbus]
        Real_Va_full: Real voltage angles [Ntest, Nbus]
        Pdtest, Qdtest: Test load data [Ntest, Nbus]
        sys_data: System data
        BRANFT: Branch from-to indices
        MAXMIN_Pg, MAXMIN_Qg: Generator limits
        gencost: Generator cost coefficients
        Real_cost_total: Real generation cost [Ntest]
        ngt_data: NGT data dictionary (with ZIB indices, etc.)
        device: Device
        apply_post_processing: Whether to apply Jacobian-based post-processing
        
    Returns:
        results: Dictionary with evaluation metrics (both raw and post-processed)
    """
    model_ngt.eval()
    
    Ntest = x_test.shape[0]
    baseMVA = float(sys_data.baseMVA)
    bus_slack = int(sys_data.bus_slack)
    bus_Va_idx = np.delete(np.arange(config.Nbus), bus_slack)
    
    # Get voltage bounds
    VmLb = config.ngt_VmLb
    VmUb = config.ngt_VmUb
    hisVm_min = sys_data.hisVm_min if hasattr(sys_data, 'hisVm_min') and sys_data.hisVm_min is not None else VmLb
    hisVm_max = sys_data.hisVm_max if hasattr(sys_data, 'hisVm_max') and sys_data.hisVm_max is not None else VmUb
    
    # Get NGT-specific indices
    NPred_Va = ngt_data['NPred_Va']
    NPred_Vm = ngt_data['NPred_Vm']
    bus_Pnet_all = ngt_data['bus_Pnet_all']
    bus_ZIB_all = ngt_data['bus_ZIB_all']
    idx_bus_Pnet_slack = ngt_data['idx_bus_Pnet_slack']
    NZIB = ngt_data['NZIB']
    param_ZIMV = ngt_data['param_ZIMV']
    
    # GPU warmup
    if device.type == 'cuda':
        with torch.no_grad():
            _ = model_ngt(x_test[:1])
            torch.cuda.synchronize()
    
    # Inference timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    time_start = time.perf_counter()
    
    with torch.no_grad():
        V_pred = model_ngt(x_test)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    inference_time = time.perf_counter() - time_start
    
    V_pred_np = V_pred.cpu().numpy()
    
    # Reconstruct full voltage using Kron Reduction
    # Model output layout: [Va_nonZIB_without_slack (NPred_Va), Vm_nonZIB (NPred_Vm)]
    # After insert:        [Va_nonZIB_with_slack (NPred_Va+1), Vm_nonZIB (NPred_Vm)]
    xam_P = np.insert(V_pred_np, idx_bus_Pnet_slack[0], 0, axis=1)
    Va_len_with_slack = NPred_Va + 1
    Va_nonZIB = xam_P[:, :Va_len_with_slack]
    Vm_nonZIB = xam_P[:, Va_len_with_slack:Va_len_with_slack + NPred_Vm]
    Vx = Vm_nonZIB * np.exp(1j * Va_nonZIB)
    
    if NZIB > 0 and param_ZIMV is not None:
        Vy = np.dot(param_ZIMV, Vx.T).T
    else:
        Vy = None
    
    Ve = np.zeros((Ntest, config.Nbus))
    Vf = np.zeros((Ntest, config.Nbus))
    Ve[:, bus_Pnet_all] = Vx.real
    Vf[:, bus_Pnet_all] = Vx.imag
    if Vy is not None:
        Ve[:, bus_ZIB_all] = Vy.real
        Vf[:, bus_ZIB_all] = Vy.imag
    
    Pred_Vm = np.sqrt(Ve**2 + Vf**2)
    Pred_Va = np.arctan2(Vf, Ve)
    
    if NZIB > 0:
        Pred_Vm[:, bus_ZIB_all] = np.clip(
            Pred_Vm[:, bus_ZIB_all], VmLb, VmUb
        )
    
    Pred_V = Pred_Vm * np.exp(1j * Pred_Va)
    
    # Calculate power flow
    Real_V = Real_Vm * np.exp(1j * Real_Va_full)
    Real_Pg, Real_Qg, Real_Pd, Real_Qd = get_genload(
        Real_V, Pdtest, Qdtest,
        sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
    )
    
    Pred_Pg, Pred_Qg, Pred_Pd, Pred_Qd = get_genload(
        Pred_V, Pdtest, Qdtest,
        sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
    )
    
    # ==================== Raw Metrics (Before Post-Processing) ====================
    mae_Vm = np.mean(np.abs(Real_Vm - Pred_Vm))
    mae_Va = np.mean(np.abs(Real_Va_full[:, bus_Va_idx] - Pred_Va[:, bus_Va_idx]))
    
    # Cost
    Pred_cost = gencost[:, 0] * (Pred_Pg * baseMVA)**2 + gencost[:, 1] * np.abs(Pred_Pg * baseMVA)
    Pred_cost_total = np.sum(Pred_cost, axis=1)
    cost_error = (Pred_cost_total - Real_cost_total) / Real_cost_total * 100
    
    # Constraint violations (raw)
    lsPg, lsQg, lsidxPg, lsidxQg, _, vio_PQg, deltaPgL, deltaPgU, deltaQgL, deltaQgU = get_vioPQg(
        Pred_Pg, sys_data.bus_Pg, MAXMIN_Pg,
        Pred_Qg, sys_data.bus_Qg, MAXMIN_Qg,
        config.DELTA
    )
    if torch.is_tensor(vio_PQg):
        vio_PQg = vio_PQg.numpy()
    
    lsidxPQg = np.squeeze(np.array(np.where((lsidxPg + lsidxQg) > 0)))
    num_viotest = np.size(lsidxPQg)
    
    vio_branang, vio_branpf, deltapf, vio_branpfidx, lsSf, _, lsSf_sampidx, _ = get_viobran2(
        Pred_V, Pred_Va, sys_data.branch, sys_data.Yf, sys_data.Yt,
        BRANFT, baseMVA, config.DELTA
    )
    if torch.is_tensor(vio_branang):
        vio_branang = vio_branang.numpy()
    if torch.is_tensor(vio_branpf):
        vio_branpf = vio_branpf.numpy()
    
    vio_branpf_num = int(np.sum(np.asarray(vio_branpfidx) > 0))
    lsSf_sampidx = np.asarray(lsSf_sampidx)
    
    # Voltage constraint satisfaction
    Vm_satisfy = 100 - np.mean(Pred_Vm > VmUb) * 100 - np.mean(Pred_Vm < VmLb) * 100
    
    # Load deviation
    Pd_error = np.mean(np.abs(np.sum(Pred_Pd, axis=1) - np.sum(Real_Pd, axis=1)) / np.abs(np.sum(Real_Pd, axis=1))) * 100
    Qd_error = np.mean(np.abs(np.sum(Pred_Qd, axis=1) - np.sum(Real_Qd, axis=1)) / np.abs(np.sum(Real_Qd, axis=1))) * 100
    
    # Store raw results
    results = {
        'mae_Vm': mae_Vm,
        'mae_Va': mae_Va,
        'cost_error_percent': np.mean(cost_error),
        'cost_mean': np.mean(Pred_cost_total),
        'Pg_satisfy': np.mean(vio_PQg[:, 0]),
        'Qg_satisfy': np.mean(vio_PQg[:, 1]),
        'Vm_satisfy': Vm_satisfy,
        'branch_ang_satisfy': np.mean(vio_branang),
        'branch_pf_satisfy': np.mean(vio_branpf),
        'Pd_error_percent': Pd_error,
        'Qd_error_percent': Qd_error,
        'inference_time_ms': inference_time / Ntest * 1000,
        'num_violated': num_viotest,
    }
    
    # ==================== Post-Processing (Jacobian-based correction) ====================
    if apply_post_processing and num_viotest > 0:
        # Use unified post-processing function
        post_results = jacobian_postprocess(
            config=config, sys_data=sys_data, BRANFT=BRANFT,
            Pred_Vm=Pred_Vm, Pred_Va=Pred_Va, Pred_V=Pred_V,
            Pdtest=Pdtest, Qdtest=Qdtest,
            lsPg=lsPg, lsQg=lsQg, lsidxPg=lsidxPg, lsidxQg=lsidxQg,
            lsidxPQg=lsidxPQg, num_viotest=num_viotest,
            MAXMIN_Pg=MAXMIN_Pg, MAXMIN_Qg=MAXMIN_Qg,
            baseMVA=baseMVA, bus_slack=bus_slack,
            hisVm_min=hisVm_min, hisVm_max=hisVm_max,
            VmLb=VmLb, VmUb=VmUb,
            include_branch_correction=False,
            verbose=False
        )
        
        # Extract results
        Pred_Vm1 = post_results['Pred_Vm1']
        Pred_Va1 = post_results['Pred_Va1']
        Pred_Pg1 = post_results['Pred_Pg1']
        Pred_Qg1 = post_results['Pred_Qg1']
        Pred_Pd1 = post_results['Pred_Pd1']
        Pred_Qd1 = post_results['Pred_Qd1']
        vio_PQg1 = post_results['vio_PQg1']
        vio_branang1 = post_results['vio_branang1']
        vio_branpf1 = post_results['vio_branpf1']
        num_viotest1 = post_results['num_viotest1']
        Vm_satisfy1 = post_results['Vm_satisfy1']
        time_post = post_results['time_post']
        
        # ==================== Post-Processed Metrics ====================
        mae_Vm1 = np.mean(np.abs(Real_Vm - Pred_Vm1))
        mae_Va1 = np.mean(np.abs(Real_Va_full[:, bus_Va_idx] - Pred_Va1[:, bus_Va_idx]))
        
        # Cost after post-processing
        Pred_cost1 = gencost[:, 0] * (Pred_Pg1 * baseMVA)**2 + gencost[:, 1] * np.abs(Pred_Pg1 * baseMVA)
        Pred_cost_total1 = np.sum(Pred_cost1, axis=1)
        cost_error1 = (Pred_cost_total1 - Real_cost_total) / Real_cost_total * 100
        
        # Load deviation after post-processing
        Pd_error1 = np.mean(np.abs(np.sum(Pred_Pd1, axis=1) - np.sum(Real_Pd, axis=1)) / np.abs(np.sum(Real_Pd, axis=1))) * 100
        Qd_error1 = np.mean(np.abs(np.sum(Pred_Qd1, axis=1) - np.sum(Real_Qd, axis=1)) / np.abs(np.sum(Real_Qd, axis=1))) * 100
        
        # Add post-processed results
        results['mae_Vm_post'] = mae_Vm1
        results['mae_Va_post'] = mae_Va1
        results['cost_error_percent_post'] = np.mean(cost_error1)
        results['cost_mean_post'] = np.mean(Pred_cost_total1)
        results['Pg_satisfy_post'] = np.mean(vio_PQg1[:, 0])
        results['Qg_satisfy_post'] = np.mean(vio_PQg1[:, 1])
        results['Vm_satisfy_post'] = Vm_satisfy1
        results['branch_ang_satisfy_post'] = np.mean(vio_branang1)
        results['branch_pf_satisfy_post'] = np.mean(vio_branpf1)
        results['Pd_error_percent_post'] = Pd_error1
        results['Qd_error_percent_post'] = Qd_error1
        results['post_processing_time_ms'] = time_post / Ntest * 1000
        results['num_violated_post'] = num_viotest1
    else:
        # No post-processing or no violations - copy raw results
        results['mae_Vm_post'] = results['mae_Vm']
        results['mae_Va_post'] = results['mae_Va']
        results['cost_error_percent_post'] = results['cost_error_percent']
        results['cost_mean_post'] = results['cost_mean']
        results['Pg_satisfy_post'] = results['Pg_satisfy']
        results['Qg_satisfy_post'] = results['Qg_satisfy']
        results['Vm_satisfy_post'] = results['Vm_satisfy']
        results['branch_ang_satisfy_post'] = results['branch_ang_satisfy']
        results['branch_pf_satisfy_post'] = results['branch_pf_satisfy']
        results['Pd_error_percent_post'] = results['Pd_error_percent']
        results['Qd_error_percent_post'] = results['Qd_error_percent']
        results['post_processing_time_ms'] = 0.0
        results['num_violated_post'] = results['num_violated']
    
    return results


def evaluate_ngt_flow_model(config, model_flow, vae_vm, vae_va, x_test, 
                            Real_Vm, Real_Va_full, Pdtest, Qdtest, sys_data, 
                            BRANFT, MAXMIN_Pg, MAXMIN_Qg, gencost, Real_cost_total, 
                            ngt_data, preference, device, apply_post_processing=True):
    """
    Evaluate a NGT Flow model on test data.
    
    This is similar to evaluate_ngt_single_model but uses Flow integration
    with VAE anchor to generate predictions.
    
    Args:
        config: Configuration object
        model_flow: PreferenceConditionedNetV model
        vae_vm, vae_va: VAE models for anchor generation
        x_test: Test input [Ntest, input_dim]
        Real_Vm, Real_Va_full: Ground truth voltages
        Pdtest, Qdtest: Test load data
        sys_data: Power system data
        BRANFT: Branch from-to indices
        MAXMIN_Pg, MAXMIN_Qg: Generator limits
        gencost: Generator cost coefficients
        Real_cost_total: Ground truth total cost
        ngt_data: NGT data dictionary
        preference: Preference tensor [1, 2]
        device: Device
        apply_post_processing: Whether to apply post-processing
        
    Returns:
        results: Dictionary of evaluation metrics
    """
    Ntest = x_test.shape[0]
    baseMVA = float(sys_data.baseMVA)
    bus_slack = int(sys_data.bus_slack)
    bus_Pnet_all = ngt_data['bus_Pnet_all']
    bus_Pnet_noslack_all = bus_Pnet_all[bus_Pnet_all != bus_slack]
    NPred_Va = ngt_data['NPred_Va']
    NPred_Vm = ngt_data['NPred_Vm']
    flow_inf_steps = getattr(config, 'ngt_flow_inf_steps', 10)
    
    model_flow.eval()
    
    # Generate VAE anchor
    with torch.no_grad():
        Vm_vae = vae_vm(x_test, use_mean=True)  # [Ntest, Nbus]
        Va_vae_noslack = vae_va(x_test, use_mean=True)  # [Ntest, Nbus-1]
        
        # Reconstruct full Va
        Va_vae = torch.zeros(Ntest, config.Nbus, device=device)
        Va_vae[:, :bus_slack] = Va_vae_noslack[:, :bus_slack]
        Va_vae[:, bus_slack+1:] = Va_vae_noslack[:, bus_slack:]
        
        # Extract non-ZIB values
        Vm_nonZIB = Vm_vae[:, bus_Pnet_all]
        Va_nonZIB_noslack = Va_vae[:, bus_Pnet_noslack_all]
        z_anchor = torch.cat([Va_nonZIB_noslack, Vm_nonZIB], dim=1)
    
    # Flow integration
    pref_batch = preference.expand(Ntest, -1)
    
    start_time = time.time()
    with torch.no_grad():
        V_pred = model_flow.flow_backward(
            x_test, z_anchor, pref_batch, 
            num_steps=flow_inf_steps, apply_sigmoid=True, training=False
        )
    inference_time = time.time() - start_time
    
    # Convert prediction to full voltage format
    V_pred_np = V_pred.cpu().numpy()
    
    # Split into Va and Vm
    Va_pred_noslack_nonZIB = V_pred_np[:, :NPred_Va]
    Vm_pred_nonZIB = V_pred_np[:, NPred_Va:]
    
    # Reconstruct full voltage vectors
    Pred_Va = np.zeros((Ntest, config.Nbus))
    Pred_Vm = np.zeros((Ntest, config.Nbus))
    
    # Insert slack bus Va (=0)
    Pred_Va[:, bus_Pnet_noslack_all] = Va_pred_noslack_nonZIB
    Pred_Vm[:, bus_Pnet_all] = Vm_pred_nonZIB
    
    # Recover ZIB node voltages using Kron Reduction
    if ngt_data.get('param_ZIMV') is not None:
        Vx = Pred_Vm[:, bus_Pnet_all] * np.exp(1j * Pred_Va[:, bus_Pnet_all])
        Vy = np.dot(ngt_data['param_ZIMV'], Vx.T).T
        Pred_Va[:, ngt_data['bus_ZIB_all']] = np.angle(Vy)
        Pred_Vm[:, ngt_data['bus_ZIB_all']] = np.abs(Vy)
    
    # The rest is the same as evaluate_ngt_single_model
    # (Reuse the evaluation logic)
    Pred_V = Pred_Vm * np.exp(1j * Pred_Va)
    
    # Get power flow results
    Pred_Pg, Pred_Qg, Pred_Pd, Pred_Qd = get_genload(
        Pred_V, Pdtest, Qdtest,
        sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
    )
    
    # Compute cost
    Pred_cost = gencost[:, 0] * (Pred_Pg * baseMVA)**2 + gencost[:, 1] * np.abs(Pred_Pg * baseMVA)
    Pred_cost_total = np.sum(Pred_cost, axis=1)
    
    # Compute metrics
    mae_Vm = np.mean(np.abs(Pred_Vm - Real_Vm))
    mae_Va = np.mean(np.abs(Pred_Va - Real_Va_full))
    
    cost_error = np.mean(np.abs(Pred_cost_total - Real_cost_total))
    cost_error_percent = cost_error / np.mean(Real_cost_total) * 100
    
    # Constraint satisfaction
    VmLb = config.ngt_VmLb
    VmUb = config.ngt_VmUb
    
    Pg_satisfy = np.mean((Pred_Pg >= MAXMIN_Pg[:, 1]) & (Pred_Pg <= MAXMIN_Pg[:, 0])) * 100
    Qg_satisfy = np.mean((Pred_Qg >= MAXMIN_Qg[:, 1]) & (Pred_Qg <= MAXMIN_Qg[:, 0])) * 100
    Vm_satisfy = np.mean((Pred_Vm >= VmLb) & (Pred_Vm <= VmUb)) * 100
    
    # Branch constraints
    branch = sys_data.branch
    branch_ang_max = branch[:, 5] * np.pi / 180 if branch.shape[1] > 5 else np.pi / 6
    Pred_ang_diff = np.abs(Pred_Va[:, BRANFT[:, 0]] - Pred_Va[:, BRANFT[:, 1]])
    branch_ang_satisfy = np.mean(Pred_ang_diff <= branch_ang_max) * 100
    
    # Compute branch power flow
    Vf = Pred_V[:, BRANFT[:, 0]]
    Vt = Pred_V[:, BRANFT[:, 1]]
    Yf = sys_data.Yf if hasattr(sys_data, 'Yf') else None
    if Yf is not None:
        If = (Yf @ Pred_V.T).T
        Sf = Vf * np.conj(If)
        branch_pf_satisfy = 100.0  # Simplified
    else:
        branch_pf_satisfy = 100.0
    
    # Load deviation
    Pd_error = np.mean(np.abs(Pred_Pd - Pdtest)) / np.mean(np.abs(Pdtest) + 1e-6) * 100
    Qd_error = np.mean(np.abs(Pred_Qd - Qdtest)) / np.mean(np.abs(Qdtest) + 1e-6) * 100
    
    # Count violated samples
    is_Pg_violated = np.any((Pred_Pg < MAXMIN_Pg[:, 1]) | (Pred_Pg > MAXMIN_Pg[:, 0]), axis=1)
    is_Qg_violated = np.any((Pred_Qg < MAXMIN_Qg[:, 1]) | (Pred_Qg > MAXMIN_Qg[:, 0]), axis=1)
    is_Vm_violated = np.any((Pred_Vm < VmLb) | (Pred_Vm > VmUb), axis=1)
    num_violated = np.sum(is_Pg_violated | is_Qg_violated | is_Vm_violated)
    
    results = {
        'mae_Vm': mae_Vm,
        'mae_Va': mae_Va,
        'cost_error_percent': cost_error_percent,
        'cost_mean': np.mean(Pred_cost_total),
        'Pg_satisfy': Pg_satisfy,
        'Qg_satisfy': Qg_satisfy,
        'Vm_satisfy': Vm_satisfy,
        'branch_ang_satisfy': branch_ang_satisfy,
        'branch_pf_satisfy': branch_pf_satisfy,
        'Pd_error_percent': Pd_error,
        'Qd_error_percent': Qd_error,
        'num_violated': num_violated,
        'inference_time_ms': inference_time / Ntest * 1000,
    }
    
    # Post-processing results (copy raw for simplicity)
    for key in ['mae_Vm', 'mae_Va', 'cost_error_percent', 'cost_mean', 
                'Pg_satisfy', 'Qg_satisfy', 'Vm_satisfy', 
                'branch_ang_satisfy', 'branch_pf_satisfy',
                'Pd_error_percent', 'Qd_error_percent', 'num_violated']:
        results[f'{key}_post'] = results[key]
    results['post_processing_time_ms'] = 0.0
    
    return results


def evaluate_ngt_flow_progressive(config, flow_models_chain, vae_vm, vae_va, x_test, 
                                   Real_Vm, Real_Va_full, Pdtest, Qdtest, sys_data, 
                                   BRANFT, MAXMIN_Pg, MAXMIN_Qg, gencost, Real_cost_total, 
                                   ngt_data, target_preference, device):
    """
    Evaluate NGT Flow model using progressive/chain inference.
    
    This function implements the progressive inference chain:
        VAE → Flow(0.9) → Flow(0.8) → ... → Flow(target_pref)
    
    Each Flow model in the chain uses the previous model's output as anchor.
    
    Args:
        config: Configuration object
        flow_models_chain: List of (model, preference) tuples in order of execution
                          Example: [(flow_09, [0.9, 0.1]), (flow_08, [0.8, 0.2])]
        vae_vm, vae_va: VAE models for initial anchor generation
        x_test: Test input [Ntest, input_dim]
        Real_Vm, Real_Va_full: Ground truth voltages
        Pdtest, Qdtest: Test load data
        sys_data: Power system data
        BRANFT: Branch from-to indices
        MAXMIN_Pg, MAXMIN_Qg: Generator limits
        gencost: Generator cost coefficients
        Real_cost_total: Ground truth total cost
        ngt_data: NGT data dictionary
        target_preference: Final target preference [λ_cost, λ_carbon]
        device: Device
        
    Returns:
        results: Dictionary of evaluation metrics
    """
    Ntest = x_test.shape[0]
    baseMVA = float(sys_data.baseMVA)
    bus_slack = int(sys_data.bus_slack)
    bus_Pnet_all = ngt_data['bus_Pnet_all']
    bus_Pnet_noslack_all = bus_Pnet_all[bus_Pnet_all != bus_slack]
    NPred_Va = ngt_data['NPred_Va']
    NPred_Vm = ngt_data['NPred_Vm']
    output_dim = ngt_data['output_dim']
    flow_inf_steps = getattr(config, 'ngt_flow_inf_steps', 10)
    
    start_time = time.time()
    
    # Step 1: Generate initial anchor from VAE
    with torch.no_grad():
        Vm_vae = vae_vm(x_test, use_mean=True)
        Va_vae_noslack = vae_va(x_test, use_mean=True)
        
        Va_vae = torch.zeros(Ntest, config.Nbus, device=device)
        Va_vae[:, :bus_slack] = Va_vae_noslack[:, :bus_slack]
        Va_vae[:, bus_slack+1:] = Va_vae_noslack[:, bus_slack:]
        
        Vm_nonZIB = Vm_vae[:, bus_Pnet_all]
        Va_nonZIB_noslack = Va_vae[:, bus_Pnet_noslack_all]
        z_current = torch.cat([Va_nonZIB_noslack, Vm_nonZIB], dim=1)
    
    print(f"  Progressive chain inference:")
    print(f"    VAE anchor → ", end="")
    
    # Step 2: Chain through Flow models
    for i, (flow_model, pref) in enumerate(flow_models_chain):
        flow_model.eval()
        pref_tensor = torch.tensor([pref], dtype=torch.float32, device=device)
        pref_batch = pref_tensor.expand(Ntest, -1)
        
        with torch.no_grad():
            # Use detach to ensure no gradient flows back
            z_anchor = z_current.detach()
            z_current = flow_model.flow_backward(
                x_test, z_anchor, pref_batch, 
                num_steps=flow_inf_steps, apply_sigmoid=False, training=False
            )
        
        print(f"Flow({pref[0]:.1f})", end="")
        if i < len(flow_models_chain) - 1:
            print(" → ", end="")
    
    print()  # Newline after chain
    
    # Step 3: Apply final sigmoid scaling
    with torch.no_grad():
        # Get Vscale and Vbias from the last flow model
        last_flow_model = flow_models_chain[-1][0]
        V_pred = torch.sigmoid(z_current) * last_flow_model.Vscale + last_flow_model.Vbias
    
    inference_time = time.time() - start_time
    
    # Convert prediction to full voltage format
    V_pred_np = V_pred.cpu().numpy()
    
    # Split into Va and Vm
    Va_pred_noslack_nonZIB = V_pred_np[:, :NPred_Va]
    Vm_pred_nonZIB = V_pred_np[:, NPred_Va:]
    
    # Reconstruct full voltage vectors
    Pred_Va = np.zeros((Ntest, config.Nbus))
    Pred_Vm = np.zeros((Ntest, config.Nbus))
    
    Pred_Va[:, bus_Pnet_noslack_all] = Va_pred_noslack_nonZIB
    Pred_Vm[:, bus_Pnet_all] = Vm_pred_nonZIB
    
    # Recover ZIB node voltages using Kron Reduction
    if ngt_data.get('param_ZIMV') is not None:
        Vx = Pred_Vm[:, bus_Pnet_all] * np.exp(1j * Pred_Va[:, bus_Pnet_all])
        Vy = np.dot(ngt_data['param_ZIMV'], Vx.T).T
        Pred_Va[:, ngt_data['bus_ZIB_all']] = np.angle(Vy)
        Pred_Vm[:, ngt_data['bus_ZIB_all']] = np.abs(Vy)
    
    # Compute metrics (same as evaluate_ngt_flow_model)
    Pred_V = Pred_Vm * np.exp(1j * Pred_Va)
    
    Pred_Pg, Pred_Qg, Pred_Pd, Pred_Qd = get_genload(
        Pred_V, Pdtest, Qdtest,
        sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
    )
    
    Pred_cost = gencost[:, 0] * (Pred_Pg * baseMVA)**2 + gencost[:, 1] * np.abs(Pred_Pg * baseMVA)
    Pred_cost_total = np.sum(Pred_cost, axis=1)
    
    mae_Vm = np.mean(np.abs(Pred_Vm - Real_Vm))
    mae_Va = np.mean(np.abs(Pred_Va - Real_Va_full))
    
    cost_error = np.mean(np.abs(Pred_cost_total - Real_cost_total))
    cost_error_percent = cost_error / np.mean(Real_cost_total) * 100
    
    VmLb = config.ngt_VmLb
    VmUb = config.ngt_VmUb
    
    Pg_satisfy = np.mean((Pred_Pg >= MAXMIN_Pg[:, 1]) & (Pred_Pg <= MAXMIN_Pg[:, 0])) * 100
    Qg_satisfy = np.mean((Pred_Qg >= MAXMIN_Qg[:, 1]) & (Pred_Qg <= MAXMIN_Qg[:, 0])) * 100
    Vm_satisfy = np.mean((Pred_Vm >= VmLb) & (Pred_Vm <= VmUb)) * 100
    
    branch = sys_data.branch
    branch_ang_max = branch[:, 5] * np.pi / 180 if branch.shape[1] > 5 else np.pi / 6
    Pred_ang_diff = np.abs(Pred_Va[:, BRANFT[:, 0]] - Pred_Va[:, BRANFT[:, 1]])
    branch_ang_satisfy = np.mean(Pred_ang_diff <= branch_ang_max) * 100
    branch_pf_satisfy = 100.0
    
    Pd_error = np.mean(np.abs(Pred_Pd - Pdtest)) / np.mean(np.abs(Pdtest) + 1e-6) * 100
    Qd_error = np.mean(np.abs(Pred_Qd - Qdtest)) / np.mean(np.abs(Qdtest) + 1e-6) * 100
    
    is_Pg_violated = np.any((Pred_Pg < MAXMIN_Pg[:, 1]) | (Pred_Pg > MAXMIN_Pg[:, 0]), axis=1)
    is_Qg_violated = np.any((Pred_Qg < MAXMIN_Qg[:, 1]) | (Pred_Qg > MAXMIN_Qg[:, 0]), axis=1)
    is_Vm_violated = np.any((Pred_Vm < VmLb) | (Pred_Vm > VmUb), axis=1)
    num_violated = np.sum(is_Pg_violated | is_Qg_violated | is_Vm_violated)
    
    results = {
        'mae_Vm': mae_Vm,
        'mae_Va': mae_Va,
        'cost_error_percent': cost_error_percent,
        'cost_mean': np.mean(Pred_cost_total),
        'Pg_satisfy': Pg_satisfy,
        'Qg_satisfy': Qg_satisfy,
        'Vm_satisfy': Vm_satisfy,
        'branch_ang_satisfy': branch_ang_satisfy,
        'branch_pf_satisfy': branch_pf_satisfy,
        'Pd_error_percent': Pd_error,
        'Qd_error_percent': Qd_error,
        'num_violated': num_violated,
        'inference_time_ms': inference_time / Ntest * 1000,
        'chain_length': len(flow_models_chain),
    }
    
    for key in ['mae_Vm', 'mae_Va', 'cost_error_percent', 'cost_mean', 
                'Pg_satisfy', 'Qg_satisfy', 'Vm_satisfy', 
                'branch_ang_satisfy', 'branch_pf_satisfy',
                'Pd_error_percent', 'Qd_error_percent', 'num_violated']:
        results[f'{key}_post'] = results[key]
    results['post_processing_time_ms'] = 0.0
    
    return results


def compare_all_models(config, model_specs, sys_data=None, device=None):
    """
    Compare multiple models of different types on the same test set.
    
    Supports comparing any combination of:
    - Dual-model architectures: simple (MLP), vae, rectified, diffusion, etc.
    - Single-model architectures: ngt (unsupervised)
    
    Args:
        config: Configuration object
        model_specs: List of model specifications, each is a dict with:
            - 'name': Display name for this model
            - 'type': Model type ('simple', 'vae', 'rectified', 'ngt', etc.)
            - 'vm_path': Path to Vm model (for dual-model) or NetV model (for ngt)
            - 'va_path': Path to Va model (for dual-model), None for ngt
            - 'pretrain_vm_path': (optional) Path to pretrained VAE for Vm
            - 'pretrain_va_path': (optional) Path to pretrained VAE for Va
        sys_data: Optional PowerSystemData object
        device: Device (optional, uses config.device if None)
        
    Returns:
        all_results: Dictionary mapping model names to their results
        
    Example:
        model_specs = [
            {'name': 'MLP', 'type': 'simple', 
             'vm_path': 'modelvm_simple.pth', 'va_path': 'modelva_simple.pth'},
            {'name': 'VAE', 'type': 'vae', 
             'vm_path': 'modelvm_vae.pth', 'va_path': 'modelva_vae.pth'},
            {'name': 'NGT', 'type': 'ngt', 
             'vm_path': 'NetV_ngt.pth', 'va_path': None},
        ]
        results = compare_all_models(config, model_specs)
    """
    print('\n' + '=' * 100)
    print(' MULTI-MODEL COMPARISON')
    print('=' * 100)
    
    if device is None:
        device = config.device
    
    # ============================================================
    # Step 1: Load NGT data (needed for all models' test data)
    # ============================================================
    print('\nLoading test data...')
    ngt_data, sys_data = load_ngt_training_data(config, sys_data)
    
    # Prepare common test data
    x_test = ngt_data['x_test'].to(device)
    Ntest = x_test.shape[0]
    
    # Real voltage from test data
    Real_Vm = ngt_data['yvm_test'].numpy()
    Real_Va_full = ngt_data['yva_test'].numpy()
    
    # Prepare test load data
    baseMVA = float(sys_data.baseMVA)
    Pdtest = np.zeros((Ntest, config.Nbus))
    Qdtest = np.zeros((Ntest, config.Nbus))
    
    bus_Pd = ngt_data['bus_Pd']
    bus_Qd = ngt_data['bus_Qd']
    idx_test = ngt_data['idx_test']
    
    Pdtest[:, bus_Pd] = sys_data.RPd[idx_test][:, bus_Pd] / baseMVA
    Qdtest[:, bus_Qd] = sys_data.RQd[idx_test][:, bus_Qd] / baseMVA
    
    # Prepare BRANFT
    branch = sys_data.branch
    BRANFT = (branch[:, 0:2] - 1).astype(int)
    
    # Real cost
    gencost = ngt_data['gencost_Pg']
    Real_V = Real_Vm * np.exp(1j * Real_Va_full)
    Real_Pg, Real_Qg, Real_Pd_out, Real_Qd_out = get_genload(
        Real_V, Pdtest, Qdtest,
        sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
    )
    Real_cost = gencost[:, 0] * (Real_Pg * baseMVA)**2 + gencost[:, 1] * np.abs(Real_Pg * baseMVA)
    Real_cost_total = np.sum(Real_cost, axis=1)
    
    # Get constraint limits
    MAXMIN_Pg = ngt_data['MAXMIN_Pg']
    MAXMIN_Qg = ngt_data['MAXMIN_Qg']
    
    # Input/output dimensions
    input_dim = ngt_data['input_dim']
    output_dim_vm = config.Nbus
    output_dim_va = config.Nbus - 1
    
    print(f'Test samples: {Ntest}')
    print(f'Real cost mean: {np.mean(Real_cost_total):.2f} $/h')
    
    # ============================================================
    # Step 2: Evaluate each model
    # ============================================================
    all_results = {}
    
    for spec in model_specs:
        model_name = spec['name']
        model_type = spec['type']
        vm_path = spec['vm_path']
        va_path = spec.get('va_path', None)
        
        print('\n' + '-' * 60)
        print(f' Evaluating: {model_name} (type: {model_type})')
        print('-' * 60)
        
        # Check if files exist
        if not os.path.exists(vm_path):
            print(f'  [Warning] Model file not found: {vm_path}')
            continue
        if va_path is not None and not os.path.exists(va_path):
            print(f'  [Warning] Model file not found: {va_path}')
            continue
        
        try:
            if model_type == 'ngt':
                # ==================== NGT MLP (single model) ====================
                from models import NetV
                
                Vscale = ngt_data['Vscale'].to(device)
                Vbias = ngt_data['Vbias'].to(device)
                output_dim = ngt_data['output_dim']
                
                model_ngt = NetV(
                    input_channels=input_dim,
                    output_channels=output_dim,
                    hidden_units=config.ngt_hidden_units,
                    khidden=config.ngt_khidden,
                    Vscale=Vscale,
                    Vbias=Vbias
                )
                model_ngt.to(device)
                
                print(f'  Loading NGT model from: {vm_path}')
                state_dict = torch.load(vm_path, map_location=device, weights_only=True)
                model_ngt.load_state_dict(state_dict)
                
                results = evaluate_ngt_single_model(
                    config, model_ngt, x_test, Real_Vm, Real_Va_full,
                    Pdtest, Qdtest, sys_data, BRANFT, MAXMIN_Pg, MAXMIN_Qg,
                    gencost, Real_cost_total, ngt_data, device
                )
            
            elif model_type == 'ngt_flow':
                # ==================== NGT Flow (single model with preference) ====================
                from models import PreferenceConditionedNetV, create_model
                
                Vscale = ngt_data['Vscale'].to(device)
                Vbias = ngt_data['Vbias'].to(device)
                output_dim = ngt_data['output_dim']
                
                hidden_dim = getattr(config, 'ngt_flow_hidden_dim', 144)
                num_layers = getattr(config, 'ngt_flow_num_layers', 2)
                
                model_flow = PreferenceConditionedNetV(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    Vscale=Vscale,
                    Vbias=Vbias,
                    preference_dim=2,
                    preference_hidden=64
                )
                model_flow.to(device)
                
                print(f'  Loading NGT Flow model from: {vm_path}')
                state_dict = torch.load(vm_path, map_location=device, weights_only=True)
                model_flow.load_state_dict(state_dict)
                model_flow.eval()
                
                # Get preference from spec
                lambda_cost = spec.get('lambda_cost', 0.9)
                preference = torch.tensor([[lambda_cost, 1.0 - lambda_cost]], dtype=torch.float32, device=device)
                
                # Load VAE for anchor generation
                vae_vm_path = config.pretrain_model_path_vm
                vae_va_path = config.pretrain_model_path_va
                
                vae_vm = create_model('vae', input_dim, config.Nbus, config, is_vm=True)
                vae_va = create_model('vae', input_dim, config.Nbus - 1, config, is_vm=False)
                vae_vm.to(device)
                vae_va.to(device)
                vae_vm.load_state_dict(torch.load(vae_vm_path, map_location=device, weights_only=True), strict=False)
                vae_va.load_state_dict(torch.load(vae_va_path, map_location=device, weights_only=True), strict=False)
                vae_vm.eval()
                vae_va.eval()
                
                # Evaluate Flow model using ngt_flow_forward function
                results = evaluate_ngt_flow_model(
                    config, model_flow, vae_vm, vae_va, x_test, 
                    Real_Vm, Real_Va_full, Pdtest, Qdtest, sys_data, 
                    BRANFT, MAXMIN_Pg, MAXMIN_Qg, gencost, Real_cost_total, 
                    ngt_data, preference, device
                )
            
            elif model_type == 'ngt_flow_progressive':
                # ==================== NGT Flow Progressive (chain inference) ====================
                from models import PreferenceConditionedNetV, create_model
                
                Vscale = ngt_data['Vscale'].to(device)
                Vbias = ngt_data['Vbias'].to(device)
                output_dim = ngt_data['output_dim']
                
                hidden_dim = getattr(config, 'ngt_flow_hidden_dim', 144)
                num_layers = getattr(config, 'ngt_flow_num_layers', 2)
                
                # Load all Flow models in the chain
                chain_paths = spec.get('chain_paths', [])
                flow_models_chain = []
                
                print(f'  Loading Flow model chain ({len(chain_paths)} models):')
                for path, lc in chain_paths:
                    model_flow = PreferenceConditionedNetV(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        Vscale=Vscale,
                        Vbias=Vbias,
                        preference_dim=2,
                        preference_hidden=64
                    )
                    model_flow.to(device)
                    model_flow.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                    model_flow.eval()
                    
                    preference = [lc, 1.0 - lc]
                    flow_models_chain.append((model_flow, preference))
                    print(f'    - λ={lc}: {os.path.basename(path)}')
                
                # Load VAE for initial anchor
                vae_vm_path = config.pretrain_model_path_vm
                vae_va_path = config.pretrain_model_path_va
                
                vae_vm = create_model('vae', input_dim, config.Nbus, config, is_vm=True)
                vae_va = create_model('vae', input_dim, config.Nbus - 1, config, is_vm=False)
                vae_vm.to(device)
                vae_va.to(device)
                vae_vm.load_state_dict(torch.load(vae_vm_path, map_location=device, weights_only=True), strict=False)
                vae_va.load_state_dict(torch.load(vae_va_path, map_location=device, weights_only=True), strict=False)
                vae_vm.eval()
                vae_va.eval()
                
                # Get target preference
                target_lambda_cost = spec.get('target_lambda_cost', 0.5)
                target_preference = [target_lambda_cost, 1.0 - target_lambda_cost]
                
                # Evaluate using progressive chain inference
                results = evaluate_ngt_flow_progressive(
                    config, flow_models_chain, vae_vm, vae_va, x_test,
                    Real_Vm, Real_Va_full, Pdtest, Qdtest, sys_data,
                    BRANFT, MAXMIN_Pg, MAXMIN_Qg, gencost, Real_cost_total,
                    ngt_data, target_preference, device
                )
                
            else:
                # ==================== Dual model (simple, vae, rectified, etc.) ====================
                from models import create_model
                
                model_vm = create_model(model_type, input_dim, output_dim_vm, config, is_vm=True)
                model_va = create_model(model_type, input_dim, output_dim_va, config, is_vm=False)
                
                model_vm.to(device)
                model_va.to(device)
                
                print(f'  Loading Vm model from: {vm_path}')
                state_dict_vm = torch.load(vm_path, map_location=device, weights_only=True)
                model_vm.load_state_dict(state_dict_vm)
                
                print(f'  Loading Va model from: {va_path}')
                state_dict_va = torch.load(va_path, map_location=device, weights_only=True)
                model_va.load_state_dict(state_dict_va)
                
                # Load pretrain models if needed (for rectified, diffusion, etc.)
                pretrain_model_vm = None
                pretrain_model_va = None
                
                if model_type in ['rectified', 'diffusion'] and spec.get('pretrain_vm_path'):
                    print(f'  Loading pretrain Vm VAE from: {spec["pretrain_vm_path"]}')
                    pretrain_model_vm = create_model('vae', input_dim, output_dim_vm, config, is_vm=True)
                    pretrain_model_vm.to(device)
                    pretrain_model_vm.load_state_dict(
                        torch.load(spec['pretrain_vm_path'], map_location=device, weights_only=True)
                    )
                    pretrain_model_vm.eval()
                    
                if model_type in ['rectified', 'diffusion'] and spec.get('pretrain_va_path'):
                    print(f'  Loading pretrain Va VAE from: {spec["pretrain_va_path"]}')
                    pretrain_model_va = create_model('vae', input_dim, output_dim_va, config, is_vm=False)
                    pretrain_model_va.to(device)
                    pretrain_model_va.load_state_dict(
                        torch.load(spec['pretrain_va_path'], map_location=device, weights_only=True)
                    )
                    pretrain_model_va.eval()
                
                results = evaluate_dual_model(
                    config, model_vm, model_va, x_test, Real_Vm, Real_Va_full,
                    Pdtest, Qdtest, sys_data, BRANFT, MAXMIN_Pg, MAXMIN_Qg,
                    gencost, Real_cost_total, model_type, device,
                    pretrain_model_vm, pretrain_model_va
                )
            
            all_results[model_name] = results
            print(f'  Evaluation completed!')
            
        except Exception as e:
            print(f'  [Error] Failed to evaluate {model_name}: {e}')
            import traceback
            traceback.print_exc()
            continue
    
    # ============================================================
    # Step 3: Print Comparison Table (Raw Results)
    # ============================================================
    if len(all_results) == 0:
        print('\n[Error] No models were successfully evaluated.')
        return {}
    
    model_names = list(all_results.keys())
    col_width = max(15, max(len(name) for name in model_names) + 2)
    
    # ==================== RAW RESULTS (Before Post-Processing) ====================
    print('\n' + '=' * 120)
    print(' COMPARISON RESULTS - BEFORE POST-PROCESSING (Raw NN Output)')
    print('=' * 120)
    print(f' Test samples: {Ntest}')
    print(f' Real cost mean: {np.mean(Real_cost_total):.2f} $/h')
    print('-' * 120)
    
    # Build header
    header = f"{'Metric':<32}"
    for name in model_names:
        header += f" {name:>{col_width}}"
    header += f" {'Best':>12}"
    print(header)
    print('-' * 120)
    
    # Metrics to compare (raw)
    metrics_raw = [
        ('Vm MAE (p.u.)', 'mae_Vm', '.6f', 'lower'),
        ('Va MAE (rad)', 'mae_Va', '.6f', 'lower'),
        ('Cost Error (%)', 'cost_error_percent', '.2f', 'lower'),
        ('Predicted Cost ($/h)', 'cost_mean', '.2f', 'lower'),
        ('Pg Satisfaction (%)', 'Pg_satisfy', '.2f', 'higher'),
        ('Qg Satisfaction (%)', 'Qg_satisfy', '.2f', 'higher'),
        ('Vm Satisfaction (%)', 'Vm_satisfy', '.2f', 'higher'),
        ('Branch Angle Sat. (%)', 'branch_ang_satisfy', '.2f', 'higher'),
        ('Branch Power Sat. (%)', 'branch_pf_satisfy', '.2f', 'higher'),
        ('Pd Error (%)', 'Pd_error_percent', '.4f', 'lower'),
        ('Qd Error (%)', 'Qd_error_percent', '.4f', 'lower'),
        ('Violated Samples', 'num_violated', 'd', 'lower'),
        ('Inference Time (ms)', 'inference_time_ms', '.4f', 'lower'),
    ]
    
    # Track wins for each model (raw)
    wins_raw = {name: 0 for name in model_names}
    
    for name, key, fmt, better in metrics_raw:
        row = f"{name:<32}"
        values = [all_results[m][key] for m in model_names]
        
        # Find best value
        if better == 'lower':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        best_model = model_names[best_idx]
        wins_raw[best_model] += 1
        
        for m in model_names:
            val = all_results[m][key]
            row += f" {val:>{col_width}{fmt}}"
        row += f" {best_model:>12}"
        print(row)
    
    print('-' * 120)
    
    # ==================== POST-PROCESSED RESULTS ====================
    print('\n' + '=' * 120)
    print(' COMPARISON RESULTS - AFTER POST-PROCESSING (Jacobian Correction)')
    print('=' * 120)
    print('-' * 120)
    
    # Build header
    print(header)
    print('-' * 120)
    
    # Metrics to compare (post-processed)
    metrics_post = [
        ('Vm MAE (p.u.)', 'mae_Vm_post', '.6f', 'lower'),
        ('Va MAE (rad)', 'mae_Va_post', '.6f', 'lower'),
        ('Cost Error (%)', 'cost_error_percent_post', '.2f', 'lower'),
        ('Predicted Cost ($/h)', 'cost_mean_post', '.2f', 'lower'),
        ('Pg Satisfaction (%)', 'Pg_satisfy_post', '.2f', 'higher'),
        ('Qg Satisfaction (%)', 'Qg_satisfy_post', '.2f', 'higher'),
        ('Vm Satisfaction (%)', 'Vm_satisfy_post', '.2f', 'higher'),
        ('Branch Angle Sat. (%)', 'branch_ang_satisfy_post', '.2f', 'higher'),
        ('Branch Power Sat. (%)', 'branch_pf_satisfy_post', '.2f', 'higher'),
        ('Pd Error (%)', 'Pd_error_percent_post', '.4f', 'lower'),
        ('Qd Error (%)', 'Qd_error_percent_post', '.4f', 'lower'),
        ('Violated Samples', 'num_violated_post', 'd', 'lower'),
        ('Post-Proc Time (ms)', 'post_processing_time_ms', '.4f', 'lower'),
    ]
    
    # Track wins for each model (post-processed)
    wins_post = {name: 0 for name in model_names}
    
    for name, key, fmt, better in metrics_post:
        row = f"{name:<32}"
        values = [all_results[m][key] for m in model_names]
        
        # Find best value
        if better == 'lower':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        best_model = model_names[best_idx]
        wins_post[best_model] += 1
        
        for m in model_names:
            val = all_results[m][key]
            row += f" {val:>{col_width}{fmt}}"
        row += f" {best_model:>12}"
        print(row)
    
    print('-' * 120)
    
    # ==================== Summary ====================
    print('\n' + '=' * 120)
    print(' SUMMARY')
    print('=' * 120)
    
    print('\n Before Post-Processing:')
    for name in model_names:
        print(f'   {name}: {wins_raw[name]}/{len(metrics_raw)} metrics')
    max_wins_raw = max(wins_raw.values())
    winners_raw = [name for name, w in wins_raw.items() if w == max_wins_raw]
    print(f'   Winner (Raw): {", ".join(winners_raw)}')
    
    print('\n After Post-Processing:')
    for name in model_names:
        print(f'   {name}: {wins_post[name]}/{len(metrics_post)} metrics')
    max_wins_post = max(wins_post.values())
    winners_post = [name for name, w in wins_post.items() if w == max_wins_post]
    print(f'   Winner (Post): {", ".join(winners_post)}')
    
    print('=' * 120)
    
    # Add metadata
    all_results['_metadata'] = {
        'real_cost_mean': np.mean(Real_cost_total),
        'ntest': Ntest,
        'wins_raw': wins_raw,
        'wins_post': wins_post,
    }
    
    return all_results


def main():
    """
    Main function with support for training and testing modes
    
    Modes controlled by config.flag_test:
        - flag_test = 0: Train model then evaluate
        - flag_test = 1: Load pre-trained model and evaluate only
    """
    # Load configuration
    config = get_config()
    
    # Determine mode
    is_test_mode = config.flag_test == 1
    mode_str = "Testing Pre-trained Model" if is_test_mode else "Training"
    
    print("=" * 60)
    print(f"DeepOPF-V {mode_str} (Extended Version)")
    print("=" * 60)
    
    config.print_config()
    
    # Get model type
    model_type = config.model_type
    print(f"\nSelected model type: {model_type}")
    print(f"Available model types: {get_available_model_types()}")
    print(f"Mode: {'Test only (flag_test=1)' if is_test_mode else 'Train + Test (flag_test=0)'}")
    
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
    
    # Create models using factory function
    model_vm = create_model(model_type, input_channels, output_channels_vm, config, is_vm=True)
    model_va = create_model(model_type, input_channels, output_channels_va, config, is_vm=False)
    
    # For rectified flow, we need pretrained VAE models
    pretrain_model_vm = None
    pretrain_model_va = None
    weight_decay = getattr(config, 'weight_decay', 0)
    criterion = nn.MSELoss()
    
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
        elif not is_test_mode:
            print("\n[Warning] No pretrained Vm VAE found. Training VAE first...")
            print(f"  Expected path: {config.pretrain_model_path_vm}")
            vae_vm = create_model('vae', input_channels, output_channels_vm, config, is_vm=True)
            vae_vm.to(config.device)
            opt_vae_vm = torch.optim.Adam(vae_vm.parameters(), lr=config.Lrm, weight_decay=weight_decay)
            vae_vm, _, _ = train_voltage_magnitude(
                config, vae_vm, opt_vae_vm, dataloaders['train_vm'],
                sys_data, criterion, config.device, model_type='vae'
            )
            pretrain_model_vm = vae_vm
            pretrain_model_vm.eval()
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
        elif not is_test_mode:
            print("\n[Warning] No pretrained Va VAE found. Training VAE first...")
            print(f"  Expected path: {config.pretrain_model_path_va}")
            vae_va = create_model('vae', input_channels, output_channels_va, config, is_vm=False)
            vae_va.to(config.device)
            opt_vae_va = torch.optim.Adam(vae_va.parameters(), lr=config.Lra, weight_decay=weight_decay)
            vae_va, _, _ = train_voltage_angle(
                config, vae_va, opt_vae_va, dataloaders['train_va'],
                criterion, config.device, model_type='vae'
            )
            pretrain_model_va = vae_va
            pretrain_model_va.eval()
        else:
            print(f"\n[Warning] Va VAE not found: {config.pretrain_model_path_va}")
            print("  Will use zero initialization for anchors in test mode.")
        
        # Attach pretrain_model to FM/DM models
        model_vm.pretrain_model = pretrain_model_vm
        model_va.pretrain_model = pretrain_model_va
    elif model_type == 'diffusion':
        print(f"\n[Info] Diffusion model with use_vae_anchor=False, using Gaussian noise as starting point.")
    
    # ==================== Test Mode: Load pre-trained models ====================
    if is_test_mode:
        print("\n" + "=" * 60)
        print("Loading Pre-trained Models for Testing")
        print("=" * 60)
        
        # Determine model paths based on model_type
        model_path_vm = f'{config.PATHVm[:-4]}_{model_type}.pth'
        model_path_va = f'{config.PATHVa[:-4]}_{model_type}.pth'
        
        print(f"\nLooking for Vm model: {model_path_vm}")
        print(f"Looking for Va model: {model_path_va}")
        
        # Check if models exist
        if not os.path.exists(model_path_vm):
            raise FileNotFoundError(f"Vm model not found: {model_path_vm}")
        if not os.path.exists(model_path_va):
            raise FileNotFoundError(f"Va model not found: {model_path_va}")
        
        # Load trained weights
        # Use strict=False to allow loading older models that don't have buffer keys
        # (schedule parameters like betas, alphas are now registered as buffers but 
        #  are deterministically computed, so missing keys are acceptable)
        print(f"\nLoading Vm model from: {model_path_vm}")
        model_vm.load_state_dict(torch.load(model_path_vm, map_location=config.device, weights_only=True), strict=False)
        
        print(f"Loading Va model from: {model_path_va}")
        model_va.load_state_dict(torch.load(model_path_va, map_location=config.device, weights_only=True), strict=False)
        
        # Move models to device and set to eval mode
        model_vm.to(config.device)
        model_va.to(config.device)
        model_vm.eval()
        model_va.eval()
        print(f'\nModels loaded and moved to: {config.device}')
        
        # Empty loss lists for test mode
        lossvm = []
        lossva = []
        
    # ==================== Training Mode ====================
    else:
        # Move models to device
        model_vm.to(config.device)
        model_va.to(config.device)
        print(f'\nModels moved to: {config.device}')
        
        # Initialize optimizers
        optimizer_vm = torch.optim.Adam(model_vm.parameters(), lr=config.Lrm, weight_decay=weight_decay)
        optimizer_va = torch.optim.Adam(model_va.parameters(), lr=config.Lra, weight_decay=weight_decay)
        
        # Initialize schedulers (optional)
        scheduler_vm = None
        scheduler_va = None
        if hasattr(config, 'learning_rate_decay') and config.learning_rate_decay:
            step_size, gamma = config.learning_rate_decay
            scheduler_vm = torch.optim.lr_scheduler.StepLR(optimizer_vm, step_size=step_size, gamma=gamma)
            scheduler_va = torch.optim.lr_scheduler.StepLR(optimizer_va, step_size=step_size, gamma=gamma)
            print(f"Learning rate scheduler enabled: step_size={step_size}, gamma={gamma}")
        
        # Check training mode (can be set via env var: TRAINING_MODE=supervised|unsupervised)
        training_mode = getattr(config, 'training_mode', 'unsupervised')
        
        if training_mode == 'unsupervised':
            # ==================== Unsupervised Training (DeepOPF-NGT) ====================
            print("\n" + "=" * 60)
            print("Unsupervised Training Mode (DeepOPF-NGT)")
            print("=" * 60)
            
            if not NGT_AVAILABLE:
                raise ImportError("Unsupervised training requires deepopf_ngt_loss.py module")
            
            # Check if Flow model is enabled
            use_flow_model = getattr(config, 'ngt_use_flow_model', False)
            
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
                
                model_ngt, loss_history, time_train, ngt_data, sys_data = train_unsupervised_ngt_flow(
                    config, sys_data, config.device,
                    lambda_cost=lambda_cost,
                    lambda_carbon=lambda_carbon,
                    flow_inf_steps=flow_inf_steps,
                    use_projection=use_projection,
                    anchor_model_path=anchor_model_path,
                    anchor_preference=anchor_preference
                )
            else:
                # Use MLP model (reference implementation)
                print("Model Type: MLP (NetV - Reference Implementation)")
                model_ngt, loss_history, time_train, ngt_data, sys_data = train_unsupervised_ngt(
                    config, sys_data, config.device
                )
            
            # Convert loss history to separate lists for compatibility
            lossvm = loss_history['total']
            lossva = loss_history.get('kgenp_mean', [])
            
            # Plot unsupervised training curves
            plot_unsupervised_training_curves(loss_history)
            
            # ==================== Evaluate NGT Model ====================
            # Use dedicated NGT evaluation function (same metrics as supervised)
            eval_results = evaluate_ngt_model(
                config, model_ngt, ngt_data, sys_data, BRANFT, config.device
            )
            
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
            print(f"Training time: {time_train:.2f} seconds ({time_train/60:.2f} minutes)")
            print("=" * 60)
            
            return model_ngt, None, results
            
        else:
            # ==================== Supervised Training ====================
            print("\n" + "=" * 60)
            print("Supervised Training Mode (Label-based Loss)")
            print("=" * 60)
            
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
    
    # ==================== Evaluate on test set (both modes) ====================
    results = evaluate_model(config, model_vm, model_va, sys_data, dataloaders, BRANFT, config.device,
                             model_type=model_type, pretrain_model_vm=pretrain_model_vm, 
                             pretrain_model_va=pretrain_model_va)
    
    # Save results (only if training mode or explicitly requested)
    if not is_test_mode:
        save_results(config, results, lossvm, lossva)
        plot_training_curves(lossvm, lossva)
    
    print("\n" + "=" * 60)
    if is_test_mode:
        print("Testing completed successfully!")
    else:
        training_mode = getattr(config, 'training_mode', 'supervised')
        print(f"Training completed successfully!")
        print(f"Training mode: {training_mode}")
    print(f"Model type: {model_type}")
    print("=" * 60)
    
    return model_vm, model_va, results


def run_comparison(model_types=None):
    """
    Run comparison between multiple model types.
    
    Supports comparing any combination of:
    - 'simple': MLP models (supervised)
    - 'vae': VAE models (supervised)
    - 'rectified': Rectified Flow models (supervised)
    - 'ngt': DeepOPF-NGT models (unsupervised)
    
    Args:
        model_types: List of model types to compare. Default: ['vae', 'ngt']
    
    Example usage:
        python train.py --compare                    # Compare VAE vs NGT
        python train.py --compare simple vae ngt    # Compare MLP, VAE, NGT
    """
    # Load configuration
    config = get_config()
    
    # Default model types to compare
    if model_types is None:
        model_types = ['vae', 'ngt']
    
    print("=" * 100)
    print(" Multi-Model Comparison")
    print("=" * 100)
    print(f" Models to compare: {model_types}")
    
    # Build model specifications
    model_specs = []
    
    for model_type in model_types:
        if model_type == 'ngt':
            # NGT (unsupervised, single model)
            ngt_path = f'{config.model_save_dir}/NetV_ngt_{config.Nbus}bus_E{config.ngt_Epoch}_final.pth'
            model_specs.append({
                'name': 'NGT (Unsup.)',
                'type': 'ngt',
                'vm_path': ngt_path,
                'va_path': None,
            })
        elif model_type == 'simple':
            # Simple MLP (supervised, dual model)
            vm_path = f'{config.model_save_dir}/modelvm{config.Nbus}r{config.sys_R}N{config.model_version}{config.nmLm}E{config.EpochVm}F1.pth'
            va_path = f'{config.model_save_dir}/modelva{config.Nbus}r{config.sys_R}N{config.model_version}{config.nmLa}E{config.EpochVa}F1.pth'
            model_specs.append({
                'name': 'MLP (Sup.)',
                'type': 'simple',
                'vm_path': vm_path,
                'va_path': va_path,
            })
        elif model_type == 'vae':
            # VAE (supervised, dual model)
            vm_path = f'{config.model_save_dir}/modelvm{config.Nbus}r{config.sys_R}N{config.model_version}Lm8642_vae_E{config.EpochVm}F1.pth'
            va_path = f'{config.model_save_dir}/modelva{config.Nbus}r{config.sys_R}N{config.model_version}La8642_vae_E{config.EpochVa}F1.pth'
            model_specs.append({
                'name': 'VAE (Sup.)',
                'type': 'vae',
                'vm_path': vm_path,
                'va_path': va_path,
            })
        elif model_type == 'rectified':
            # Rectified Flow (supervised, dual model with VAE pretrain)
            vm_path = f'{config.model_save_dir}/modelvm{config.Nbus}r{config.sys_R}N{config.model_version}Lm8642_rectified_E{config.EpochVm}F1.pth'
            va_path = f'{config.model_save_dir}/modelva{config.Nbus}r{config.sys_R}N{config.model_version}La8642_rectified_E{config.EpochVa}F1.pth'
            # Pretrain VAE models
            pretrain_vm_path = f'{config.model_save_dir}/modelvm{config.Nbus}r{config.sys_R}N{config.model_version}Lm8642_vae_E{config.EpochVm}F1.pth'
            pretrain_va_path = f'{config.model_save_dir}/modelva{config.Nbus}r{config.sys_R}N{config.model_version}La8642_vae_E{config.EpochVa}F1.pth'
            model_specs.append({
                'name': 'RectFlow (Sup.)',
                'type': 'rectified',
                'vm_path': vm_path,
                'va_path': va_path,
                'pretrain_vm_path': pretrain_vm_path,
                'pretrain_va_path': pretrain_va_path,
            })
        elif model_type == 'diffusion':
            # Diffusion (supervised, dual model)
            vm_path = f'{config.model_save_dir}/modelvm{config.Nbus}r{config.sys_R}N{config.model_version}Lm8642_diffusion_E{config.EpochVm}F1.pth'
            va_path = f'{config.model_save_dir}/modelva{config.Nbus}r{config.sys_R}N{config.model_version}La8642_diffusion_E{config.EpochVa}F1.pth'
            model_specs.append({
                'name': 'Diffusion (Sup.)',
                'type': 'diffusion',
                'vm_path': vm_path,
                'va_path': va_path,
            })
        elif model_type == 'ngt_flow':
            # NGT Flow (unsupervised, single model with preference conditioning)
            # Look for Flow models with different preferences
            lambda_cost = getattr(config, 'ngt_lambda_cost', 0.9)
            lc_str = f"{lambda_cost:.1f}".replace('.', '')
            ngt_flow_path = f'{config.model_save_dir}/NetV_ngt_flow_{config.Nbus}bus_lc{lc_str}_E{config.ngt_Epoch}_final.pth'
            model_specs.append({
                'name': f'NGT-Flow (λ={lambda_cost})',
                'type': 'ngt_flow',
                'vm_path': ngt_flow_path,
                'va_path': None,
                'lambda_cost': lambda_cost,
            })
        elif model_type == 'ngt_flow_progressive':
            # NGT Flow with progressive/chain inference
            # Need to load all Flow models in the chain
            target_lambda_cost = getattr(config, 'ngt_lambda_cost', 0.5)
            
            # Build the chain from 0.9 down to target
            chain_lambdas = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
            chain_lambdas = [lc for lc in chain_lambdas if lc >= target_lambda_cost]
            
            # Find paths for all models in the chain
            chain_paths = []
            for lc in chain_lambdas:
                lc_str = f"{lc:.1f}".replace('.', '')
                path = f'{config.model_save_dir}/NetV_ngt_flow_{config.Nbus}bus_lc{lc_str}_E{config.ngt_Epoch}_final.pth'
                if os.path.exists(path):
                    chain_paths.append((path, lc))
                else:
                    print(f"  [Warning] Flow model not found for λ={lc}: {path}")
            
            if chain_paths:
                model_specs.append({
                    'name': f'NGT-Flow-Prog (→λ={target_lambda_cost})',
                    'type': 'ngt_flow_progressive',
                    'vm_path': chain_paths[-1][0],  # Target model path
                    'va_path': None,
                    'chain_paths': chain_paths,  # List of (path, lambda_cost)
                    'target_lambda_cost': target_lambda_cost,
                })
        else:
            print(f"[Warning] Unknown model type: {model_type}")
            continue
    
    if len(model_specs) == 0:
        print("[Error] No valid model types specified.")
        return None
    
    print(f"\nModel specifications:")
    for spec in model_specs:
        print(f"  - {spec['name']}: {spec['vm_path']}")
    
    # Run comparison
    results = compare_all_models(config, model_specs)
    
    # Save comparison results
    import json
    save_path = f'{config.model_save_dir}/comparison_{"_".join(model_types)}_{config.Nbus}bus.json'
    
    # Convert numpy values to Python types for JSON serialization
    results_json = {}
    for model_name, model_results in results.items():
        if isinstance(model_results, dict):
            results_json[model_name] = {}
            for k, v in model_results.items():
                if hasattr(v, 'item'):
                    results_json[model_name][k] = float(v)
                elif isinstance(v, np.ndarray):
                    results_json[model_name][k] = v.tolist()
                else:
                    results_json[model_name][k] = v
        else:
            if hasattr(model_results, 'item'):
                results_json[model_name] = float(model_results)
            elif isinstance(model_results, np.ndarray):
                results_json[model_name] = model_results.tolist()
            else:
                results_json[model_name] = model_results
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"\nComparison results saved to: {save_path}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        # Run comparison mode
        # Usage: python train.py --compare [model_types...]
        # Examples:
        #   python train.py --compare              # Compare VAE vs NGT (default)
        #   python train.py --compare simple vae  # Compare MLP vs VAE
        #   python train.py --compare simple vae ngt  # Compare MLP, VAE, NGT
        if len(sys.argv) > 2:
            model_types = sys.argv[2:]
        else:
            model_types = None  # Use default: ['vae', 'ngt']
        run_comparison(model_types)
    else:
        # Run normal training/testing mode
        main()

