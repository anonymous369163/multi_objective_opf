#!/usr/bin/env python
# coding: utf-8
# Training Script for DeepOPF-V
# Extended to support multiple model types: VAE, Flow, Diffusion, GAN, etc.
# Author: Wanjun HUANG
# Date: July 4th, 2021

import torch
import torch.nn as nn
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
        dV_branch = np.zeros((lsSf_sampidx.shape[0], config.Nbus * 2))
        for i in range(lsSf_sampidx.shape[0]):
            mp = np.array(lsSf[i][:, 2] / lsSf[i][:, 1]).reshape(-1, 1)
            mq = np.array(lsSf[i][:, 3] / lsSf[i][:, 1]).reshape(-1, 1)
            dPdV = dPfbus_dV[np.array(lsSf[i][:, 0].astype(int)).squeeze(), :]
            dQdV = dQfbus_dV[np.array(lsSf[i][:, 0].astype(int)).squeeze(), :]
            dmp = mp * dPdV
            dmq = mq * dQdV
            dmpq_inv = np.linalg.pinv(dmp + dmq)
            dV_branch[i] = np.dot(dmpq_inv, np.array(lsSf[i][:, 1])).squeeze()
        dV1 = dV1 + dV_branch
    
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


def test_on_training_set(config, model_vm, model_va, sys_data, device, model_type='simple'):
    """
    Test models on training set with timing statistics
    
    Args:
        config: Configuration object
        model_vm: Trained Vm model
        model_va: Trained Va model
        sys_data: System data
        device: Device
        model_type: Type of model
        
    Returns:
        mae_Vmtrain, mae_Vatrain: Mean absolute errors
        mre_Vmtrain_clip, mre_Vatrain: Mean relative errors
        timing_info: Dictionary with timing statistics
    """
    print('\n' + '=' * 60)
    print(f'Testing on Training Set - Model: {model_type}')
    print('=' * 60)
    
    model_vm.eval()
    model_va.eval()
    
    num_samples = sys_data.x_train.shape[0]
    
    # GPU warmup
    if device.type == 'cuda':
        with torch.no_grad():
            dummy_x = sys_data.x_train[0:1].to(device)
            _ = model_vm(dummy_x)
            torch.cuda.synchronize()
    
    # ==================== Test Vm with Timing ====================
    xtrain = sys_data.x_train.to(device)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    time_start_vm = time.perf_counter()
    
    with torch.no_grad():
        yvmtrain_hat = model_vm(xtrain)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    time_end_vm = time.perf_counter()
    time_vm = time_end_vm - time_start_vm
    
    yvmtrain_hat = yvmtrain_hat.cpu()
    
    yvmtrains = sys_data.yvm_train / config.scale_vm * (sys_data.VmUb - sys_data.VmLb) + sys_data.VmLb
    yvmtrain_hats = yvmtrain_hat.detach() / config.scale_vm * (sys_data.VmUb - sys_data.VmLb) + sys_data.VmLb
    yvmtrain_hat_clip = get_clamp(yvmtrain_hats, sys_data.hisVm_min, sys_data.hisVm_max)

    mae_Vmtrain = get_mae(yvmtrains, yvmtrain_hat_clip.detach())
    mre_Vmtrain_clip = get_rerr(yvmtrains, yvmtrain_hat_clip.detach())
    
    print(f'Vm Training Set Results:')
    print(f'  MAE: {mae_Vmtrain:.6f} p.u.')
    print(f'  MRE: {torch.mean(mre_Vmtrain_clip):.4f}% (max: {torch.max(mre_Vmtrain_clip):.4f}%)')
    
    # ==================== Test Va with Timing ====================
    if device.type == 'cuda':
        torch.cuda.synchronize()
    time_start_va = time.perf_counter()
    
    with torch.no_grad():
        yvatrain_hat = model_va(xtrain)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    time_end_va = time.perf_counter()
    time_va = time_end_va - time_start_va
    
    yvatrain_hat = yvatrain_hat.cpu()
    yvatrains = sys_data.yva_train / config.scale_va
    yvatrain_hats = yvatrain_hat.detach() / config.scale_va

    mae_Vatrain = get_mae(yvatrains, yvatrain_hats)
    mre_Vatrain = get_rerr(yvatrains, yvatrain_hats)
    
    print(f'Va Training Set Results:')
    print(f'  MAE: {mae_Vatrain:.6f} rad')
    print(f'  MRE: {torch.mean(mre_Vatrain):.4f}% (max: {torch.max(mre_Vatrain):.4f}%)')
    
    # ==================== Timing Summary ====================
    time_total = time_vm + time_va
    time_per_sample = time_total / num_samples * 1000  # in ms
    
    print(f'\nBatch Inference Timing (Training Set):')
    print(f'  Samples: {num_samples}')
    print(f'  Vm prediction: {time_vm:.4f} s')
    print(f'  Va prediction: {time_va:.4f} s')
    print(f'  Total: {time_total:.4f} s')
    print(f'  Per sample: {time_per_sample:.4f} ms')
    
    timing_info = {
        'num_samples': num_samples,
        'time_vm': time_vm,
        'time_va': time_va,
        'time_total': time_total,
        'time_per_sample_ms': time_per_sample,
    }
    
    return mae_Vmtrain, mre_Vmtrain_clip, mae_Vatrain, mre_Vatrain, timing_info


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
                z = pretrain_model(test_x, use_mean=True)
            else:
                # If no pretrain model, use zero as starting point
                output_dim = model.output_dim
                z = torch.zeros(test_x.shape[0], output_dim).to(device)
            
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
    vio_branpf_num = np.size(np.where(vio_branpfidx > 0))
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
        dV_branch = np.zeros((lsSf_sampidx.shape[0], config.Nbus * 2))
        for i in range(lsSf_sampidx.shape[0]):
            mp = np.array(lsSf[i][:, 2] / lsSf[i][:, 1]).reshape(-1, 1)
            mq = np.array(lsSf[i][:, 3] / lsSf[i][:, 1]).reshape(-1, 1)
            dPdV = dPfbus_dV[np.array(lsSf[i][:, 0].astype(int)).squeeze(), :]
            dQdV = dQfbus_dV[np.array(lsSf[i][:, 0].astype(int)).squeeze(), :]
            dmp = mp * dPdV
            dmq = mq * dQdV
            dmpq_inv = np.linalg.pinv(dmp + dmq)
            dV_branch[i] = np.dot(dmpq_inv, np.array(lsSf[i][:, 1])).squeeze()
        dV1 = dV1 + dV_branch
    
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
    
    if has_ngt_keys:
        # DeepOPF-NGT format
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
    }
    
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
            n_batches += 1
        
        # Average losses for this epoch
        if n_batches > 0:
            avg_loss = running_loss / n_batches
            avg_kgenp = running_kgenp / n_batches
            avg_kgenq = running_kgenq / n_batches
            avg_kpd = running_kpd / n_batches
            avg_kqd = running_kqd / n_batches
            avg_kv = running_kv / n_batches
            
            loss_history['total'].append(avg_loss)
            loss_history['kgenp_mean'].append(avg_kgenp)
            loss_history['kgenq_mean'].append(avg_kgenq)
            loss_history['kpd_mean'].append(avg_kpd)
            loss_history['kqd_mean'].append(avg_kqd)
            loss_history['kv_mean'].append(avg_kv)
        
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
        
        # Save models periodically
        if (epoch + 1) % 100 == 0 and (epoch + 1) >= s_epoch:
            save_path = f'{config.model_save_dir}/NetV_ngt_{config.Nbus}bus_E{epoch+1}.pth'
            torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)
            print(f"  Model saved: {save_path}")
    
    time_train = time.time() - start_time
    print(f"\nTraining completed in {time_train:.2f} seconds ({time_train/60:.2f} minutes)")
    
    # Save final model
    final_path = f'{config.model_save_dir}/NetV_ngt_{config.Nbus}bus_E{n_epochs}_final.pth'
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
    
    # Insert slack bus Va = 0
    xam_P = np.insert(V_pred_np, idx_bus_Pnet_slack[0], 0, axis=1)
    
    # Split Va and Vm for non-ZIB buses
    Va_nonZIB = xam_P[:, :NPred_Vm]
    Vm_nonZIB = xam_P[:, NPred_Vm:]
    
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
    
    vio_branpf_num = np.size(np.where(vio_branpfidx > 0))
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
    xam_P = np.insert(V_pred_np, idx_bus_Pnet_slack[0], 0, axis=1)
    Va_nonZIB = xam_P[:, :NPred_Vm]
    Vm_nonZIB = xam_P[:, NPred_Vm:]
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
    
    vio_branpf_num = np.size(np.where(vio_branpfidx > 0))
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
                # ==================== NGT (single model) ====================
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
        
        # Check training mode
        training_mode = getattr(config, 'training_mode', 'supervised')
        
        if training_mode == 'unsupervised':
            # ==================== Unsupervised Training (DeepOPF-NGT) ====================
            # Use the reference implementation with SINGLE NetV model
            # This is exactly matching main_DeepOPFNGT_M3.ipynb
            print("\n" + "=" * 60)
            print("Unsupervised Training Mode (DeepOPF-NGT Reference Implementation)")
            print("=" * 60)
            
            if not NGT_AVAILABLE:
                raise ImportError("Unsupervised training requires deepopf_ngt_loss.py module")
            
            # Use train_unsupervised_ngt which exactly matches the reference code
            # This uses a SINGLE NetV model (not dual Vm/Va models)
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
            
            # Save training history
            save_path = f'{config.model_save_dir}/ngt_results_{config.Nbus}bus.npz'
            np.savez(save_path, **{k: v for k, v in loss_history.items()})
            print(f"\nTraining history saved to: {save_path}")
            
            # Save evaluation results
            eval_save_path = f'{config.model_save_dir}/ngt_eval_{config.Nbus}bus.npz'
            eval_to_save = {k: v for k, v in eval_results.items() 
                          if isinstance(v, (int, float, np.ndarray))}
            np.savez(eval_save_path, **eval_to_save)
            print(f"Evaluation results saved to: {eval_save_path}")
            
            print("\n" + "=" * 60)
            print(f"DeepOPF-NGT Training and Evaluation completed!")
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
            
            # Test on training set (only for simple model type currently)
            if model_type == 'simple':
                _, _, _, _, train_timing = test_on_training_set(
                    config, model_vm, model_va, sys_data, config.device, model_type=model_type
                )
            else:
                print(f"\n[Note] Skipping training set test for model type '{model_type}'")
                print("       Use evaluate_model_generative() for generative models")
    
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


def verify_evaluation_consistency(config):
    """
    Verify that the new evaluate_dual_model produces consistent results
    with the original evaluate_model function.
    
    Uses VAE model as a test case.
    """
    print('\n' + '=' * 100)
    print(' VERIFICATION: Comparing evaluate_model vs evaluate_dual_model')
    print('=' * 100)
    
    device = config.device
    
    # Load data using the original method
    sys_data, dataloaders, BRANFT_orig = load_all_data(config)
    
    # Load VAE models
    input_dim = sys_data.x_train.shape[1]
    output_dim_vm = config.Nbus
    output_dim_va = config.Nbus - 1
    
    from models import create_model
    model_vm = create_model('vae', input_dim, output_dim_vm, config, is_vm=True)
    model_va = create_model('vae', input_dim, output_dim_va, config, is_vm=False)
    
    model_vm.to(device)
    model_va.to(device)
    
    # Load weights
    vm_path = f'{config.model_save_dir}/modelvm{config.Nbus}r{config.sys_R}N{config.model_version}Lm8642_vae_E{config.EpochVm}F1.pth'
    va_path = f'{config.model_save_dir}/modelva{config.Nbus}r{config.sys_R}N{config.model_version}La8642_vae_E{config.EpochVa}F1.pth'
    
    print(f'\nLoading VAE models:')
    print(f'  Vm: {vm_path}')
    print(f'  Va: {va_path}')
    
    model_vm.load_state_dict(torch.load(vm_path, map_location=device, weights_only=True))
    model_va.load_state_dict(torch.load(va_path, map_location=device, weights_only=True))
    
    # ==================== Test with original evaluate_model ====================
    print('\n--- Running original evaluate_model ---')
    results_orig = evaluate_model(
        config, model_vm, model_va, sys_data, dataloaders, BRANFT_orig, device,
        model_type='vae'
    )
    
    # ==================== Test with new evaluate_dual_model ====================
    print('\n--- Running new evaluate_dual_model (via compare_all_models) ---')
    
    # Load NGT data for compare_all_models
    ngt_data, sys_data_ngt = load_ngt_training_data(config, sys_data)
    
    # Prepare test data
    x_test = ngt_data['x_test'].to(device)
    Ntest = x_test.shape[0]
    
    Real_Vm = ngt_data['yvm_test'].numpy()
    Real_Va_full = ngt_data['yva_test'].numpy()
    
    baseMVA = float(sys_data.baseMVA)
    Pdtest = np.zeros((Ntest, config.Nbus))
    Qdtest = np.zeros((Ntest, config.Nbus))
    
    bus_Pd = ngt_data['bus_Pd']
    bus_Qd = ngt_data['bus_Qd']
    idx_test = ngt_data['idx_test']
    
    Pdtest[:, bus_Pd] = sys_data.RPd[idx_test][:, bus_Pd] / baseMVA
    Qdtest[:, bus_Qd] = sys_data.RQd[idx_test][:, bus_Qd] / baseMVA
    
    branch = sys_data.branch
    BRANFT = (branch[:, 0:2] - 1).astype(int)
    
    gencost = ngt_data['gencost_Pg']
    Real_V = Real_Vm * np.exp(1j * Real_Va_full)
    Real_Pg, _, _, _ = get_genload(
        Real_V, Pdtest, Qdtest,
        sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
    )
    Real_cost = gencost[:, 0] * (Real_Pg * baseMVA)**2 + gencost[:, 1] * np.abs(Real_Pg * baseMVA)
    Real_cost_total = np.sum(Real_cost, axis=1)
    
    MAXMIN_Pg = ngt_data['MAXMIN_Pg']
    MAXMIN_Qg = ngt_data['MAXMIN_Qg']
    
    # Reload models (they might have been modified)
    model_vm.load_state_dict(torch.load(vm_path, map_location=device, weights_only=True))
    model_va.load_state_dict(torch.load(va_path, map_location=device, weights_only=True))
    
    results_new = evaluate_dual_model(
        config, model_vm, model_va, x_test, Real_Vm, Real_Va_full,
        Pdtest, Qdtest, sys_data, BRANFT, MAXMIN_Pg, MAXMIN_Qg,
        gencost, Real_cost_total, 'vae', device,
        apply_post_processing=True
    )
    
    # ==================== Compare Results ====================
    print('\n' + '=' * 100)
    print(' COMPARISON: Original vs New Implementation')
    print('=' * 100)
    print(f"{'Metric':<35} {'Original':>18} {'New (Raw)':>18} {'New (Post)':>18}")
    print('-' * 100)
    
    # Note: Original evaluate_model uses different naming conventions
    # and uses torch tensors, so we need to extract values carefully
    
    comparisons = [
        ('Vm MAE (before post)', 
         float(results_orig['mae_Vmtest'].item()) if hasattr(results_orig['mae_Vmtest'], 'item') else results_orig['mae_Vmtest'],
         results_new['mae_Vm'],
         results_new['mae_Vm_post']),
        ('Pg Satisfy (before post)', 
         float(torch.mean(results_orig['vio_PQg'][:, 0]).item()),
         results_new['Pg_satisfy'],
         results_new['Pg_satisfy_post']),
        ('Qg Satisfy (before post)', 
         float(torch.mean(results_orig['vio_PQg'][:, 1]).item()),
         results_new['Qg_satisfy'],
         results_new['Qg_satisfy_post']),
        ('Branch Ang (before post)', 
         float(torch.mean(results_orig['vio_branang']).item()),
         results_new['branch_ang_satisfy'],
         results_new['branch_ang_satisfy_post']),
        ('Branch PF (before post)', 
         float(torch.mean(results_orig['vio_branpf']).item()),
         results_new['branch_pf_satisfy'],
         results_new['branch_pf_satisfy_post']),
    ]
    
    all_close = True
    for name, orig, new_raw, new_post in comparisons:
        diff = abs(orig - new_raw)
        status = "OK" if diff < 0.01 else "DIFF"
        if diff >= 0.01:
            all_close = False
        print(f"{name:<35} {orig:>18.4f} {new_raw:>18.4f} {new_post:>18.4f} [{status}]")
    
    print('-' * 100)
    
    if all_close:
        print('\n[SUCCESS] Results are consistent between original and new implementation!')
    else:
        print('\n[WARNING] Some differences detected. This may be due to:')
        print('  - Different test data sampling (NGT uses different random sampling)')
        print('  - Different scaling/clipping methods')
        print('  - Different data preprocessing')
    
    print('=' * 100)
    
    return results_orig, results_new


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
    elif len(sys.argv) > 1 and sys.argv[1] == '--verify':
        # Run verification mode to compare original vs new evaluation
        config = get_config()
        verify_evaluation_consistency(config)
    else:
        # Run normal training/testing mode
        main()

