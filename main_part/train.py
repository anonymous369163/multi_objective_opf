#!/usr/bin/env python
# coding: utf-8
# Training Script for DeepOPF-V
# Extended to support multiple model types: VAE, Flow, Diffusion, GAN, etc.
# Author: Wanjun HUANG
# Date: July 4th, 2021

import torch
import torch.nn as nn
import numpy as np
import scipy.io
import time
import os
import matplotlib.pyplot as plt
import math

from config import get_config
from models import NetVm, NetVa, create_model, get_available_model_types
from data_loader import load_all_data
from utils import (get_mae, get_rerr, get_clamp, get_genload, get_Pgcost,
                   get_vioPQg, get_viobran, get_viobran2, dPQbus_dV, get_hisdV,
                   dSlbus_dV)

# Import unsupervised loss module (for unsupervised training)
try:
    from unsupervised_loss import UnsupervisedOPFLoss
    UNSUPERVISED_AVAILABLE = True
except ImportError:
    UNSUPERVISED_AVAILABLE = False
    print("[train.py] Warning: unsupervised_loss module not found. Unsupervised training disabled.")


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
            
            if scheduler is not None:
                scheduler.step()

        lossvm.append(running_loss)

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
            
            if scheduler is not None:
                scheduler.step()

        lossva.append(running_loss)

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
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Total loss
    axes[0, 0].plot(loss_history['total'])
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True)
    
    # Cost loss (L_obj)
    axes[0, 1].plot(loss_history['cost'])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Generation Cost (L_obj)')
    axes[0, 1].grid(True)
    
    # Generator violation loss (L_g)
    axes[0, 2].plot(loss_history['gen_vio'])
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].set_title('Generator Violation (L_g)')
    axes[0, 2].grid(True)
    
    # Branch power flow violation (L_Sl)
    axes[1, 0].plot(loss_history['branch_pf_vio'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Branch Power Flow Violation (L_Sl)')
    axes[1, 0].grid(True)
    
    # Branch angle violation (L_theta)
    axes[1, 1].plot(loss_history['branch_ang_vio'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Branch Angle Violation (L_theta)')
    axes[1, 1].grid(True)
    
    # Load deviation loss (L_d)
    axes[1, 2].plot(loss_history['load_dev'])
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].set_title('Load Deviation (L_d)')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('unsupervised_training_curves.png', dpi=300, bbox_inches='tight')
    print('\nUnsupervised training curves saved to: unsupervised_training_curves.png')
    plt.close()


def train_unsupervised(config, model_vm, model_va, optimizer_vm, optimizer_va, 
                       training_loader, sys_data, device, model_type='simple',
                       pretrain_model_vm=None, pretrain_model_va=None,
                       scheduler_vm=None, scheduler_va=None):
    """
    Train Vm and Va models jointly using unsupervised loss (no labels).
    
    The loss function is based on DeepOPF-NGT:
    L = k_obj * L_obj + k_g * L_g + k_Sl * L_Sl + k_theta * L_theta + k_d * L_d
    
    Where:
    - L_obj: Generation cost minimization
    - L_g: Generator power limit violations
    - L_Sl: Branch power flow violations
    - L_theta: Branch angle difference violations
    - L_d: Load deviation penalty
    
    Args:
        config: Configuration object
        model_vm: Voltage magnitude model
        model_va: Voltage angle model
        optimizer_vm: Optimizer for Vm model
        optimizer_va: Optimizer for Va model
        training_loader: Training data loader (x only, no y labels used)
        sys_data: System data containing power system parameters
        device: Device (CPU/GPU)
        model_type: Type of model ('simple', 'vae', 'rectified', etc.)
        pretrain_model_vm: Pretrained VAE for Vm (for rectified flow)
        pretrain_model_va: Pretrained VAE for Va (for rectified flow)
        scheduler_vm: Learning rate scheduler for Vm
        scheduler_va: Learning rate scheduler for Va
        
    Returns:
        model_vm, model_va: Trained models
        loss_history: Dictionary of training losses
        time_train: Training time
    """
    if not UNSUPERVISED_AVAILABLE:
        raise ImportError("Unsupervised loss module not available. Please check unsupervised_loss.py")
    
    print('=' * 60)
    print(f'Unsupervised Training (Joint Vm & Va) - Type: {model_type}')
    print('=' * 60)
    
    # Initialize unsupervised loss module
    use_adaptive = getattr(config, 'use_adaptive_weights', True)
    loss_fn = UnsupervisedOPFLoss(sys_data, config, use_adaptive_weights=use_adaptive)
    loss_fn = loss_fn.to(device)
    
    # Store VmLb and VmUb in loss_fn for denormalization
    loss_fn.register_buffer('VmLb', sys_data.VmLb.to(device))
    loss_fn.register_buffer('VmUb', sys_data.VmUb.to(device))
    
    # Get VAE beta for VAE training
    vae_beta = getattr(config, 'vae_beta', 1.0)
    
    # Initialize loss history
    loss_history = {
        'total': [],
        'cost': [],
        'gen_vio': [],
        'branch_pf_vio': [],
        'branch_ang_vio': [],
        'load_dev': [],
    }
    
    # Number of epochs (use max of Vm and Va epochs)
    n_epochs = max(config.EpochVm, config.EpochVa)
    
    start_time = time.process_time()
    
    # Prepare load data (Pd, Qd) for all training samples
    # We need to create Pd, Qd tensors that match each batch
    baseMVA = sys_data.baseMVA.item() if hasattr(sys_data.baseMVA, 'item') else float(sys_data.baseMVA)
    
    # Full Pd/Qd arrays (converted to torch tensors on device)
    Pd_full = torch.from_numpy(sys_data.RPd[:config.Ntrain] / baseMVA).float().to(device)
    Qd_full = torch.from_numpy(sys_data.RQd[:config.Ntrain] / baseMVA).float().to(device)
    
    for epoch in range(n_epochs):
        running_loss = 0.0
        running_cost = 0.0
        running_gen_vio = 0.0
        running_branch_pf = 0.0
        running_branch_ang = 0.0
        running_load_dev = 0.0
        n_batches = 0
        
        model_vm.train()
        model_va.train()
        
        for step, (train_x, _) in enumerate(training_loader):
            # Note: train_y (labels) is ignored in unsupervised training
            train_x = train_x.to(device)
            batch_size = train_x.shape[0]
            
            # Get Pd, Qd for this batch
            start_idx = step * config.batch_size_training
            end_idx = start_idx + batch_size
            Pd_batch = Pd_full[start_idx:end_idx]
            Qd_batch = Qd_full[start_idx:end_idx]
            
            # Zero gradients
            optimizer_vm.zero_grad()
            optimizer_va.zero_grad()
            
            # ==================== Model-specific forward pass ====================
            if model_type == 'simple':
                # Simple MLP forward pass
                Vm_pred = model_vm(train_x)
                Va_pred = model_va(train_x)
                
            elif model_type == 'vae':
                # VAE forward pass - use sampling during training
                # For unsupervised, we don't have target y, so we use x only
                Vm_pred = model_vm(train_x, use_mean=False)  # Sample from latent
                Va_pred = model_va(train_x, use_mean=False)
                
            elif model_type in ['rectified', 'gaussian', 'conditional', 'interpolation']:
                # Flow Matching - use VAE anchor if available
                t_batch = torch.rand([batch_size, 1]).to(device)
                
                if pretrain_model_vm is not None:
                    with torch.no_grad():
                        z_vm = pretrain_model_vm(train_x, use_mean=True)
                else:
                    z_vm = torch.zeros(batch_size, model_vm.output_dim).to(device)
                
                if pretrain_model_va is not None:
                    with torch.no_grad():
                        z_va = pretrain_model_va(train_x, use_mean=True)
                else:
                    z_va = torch.zeros(batch_size, model_va.output_dim).to(device)
                
                # For unsupervised, we integrate from anchor to get prediction
                inf_step = getattr(config, 'inf_step', 20)
                step_size = 1.0 / inf_step
                
                Vm_pred, _ = model_vm.flow_backward(train_x, z_vm, step=step_size, method='Euler')
                Va_pred, _ = model_va.flow_backward(train_x, z_va, step=step_size, method='Euler')
                
            elif model_type == 'diffusion':
                # Diffusion model - sample from noise
                inf_step = getattr(config, 'inf_step', 20)
                use_vae_anchor = getattr(config, 'use_vae_anchor', False)
                
                if use_vae_anchor and pretrain_model_vm is not None:
                    with torch.no_grad():
                        vae_anchor_vm = pretrain_model_vm(train_x, use_mean=True)
                        vae_anchor_va = pretrain_model_va(train_x, use_mean=True)
                    z_vm = torch.randn(batch_size, model_vm.output_dim).to(device)
                    z_va = torch.randn(batch_size, model_va.output_dim).to(device)
                    Vm_pred = model_vm.diffusion_backward_with_anchor(train_x, z_vm, vae_anchor_vm, inf_step=inf_step)
                    Va_pred = model_va.diffusion_backward_with_anchor(train_x, z_va, vae_anchor_va, inf_step=inf_step)
                else:
                    z_vm = torch.randn(batch_size, model_vm.output_dim).to(device)
                    z_va = torch.randn(batch_size, model_va.output_dim).to(device)
                    Vm_pred = model_vm.diffusion_backward(train_x, z_vm, inf_step=inf_step)
                    Va_pred = model_va.diffusion_backward(train_x, z_va, inf_step=inf_step)
                    
            else:
                raise NotImplementedError(f"Unsupervised training for '{model_type}' not implemented")
            
            # ==================== Compute Unsupervised Loss ====================
            loss, loss_dict = loss_fn(Vm_pred, Va_pred, Pd_batch, Qd_batch, update_weights=True)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model_vm.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model_va.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer_vm.step()
            optimizer_va.step()
            
            # Accumulate losses
            running_loss += loss_dict['total']
            running_cost += loss_dict['cost']
            running_gen_vio += loss_dict['gen_vio']
            running_branch_pf += loss_dict['branch_pf_vio']
            running_branch_ang += loss_dict['branch_ang_vio']
            running_load_dev += loss_dict['load_dev']
            n_batches += 1
        
        # Learning rate scheduler step
        if scheduler_vm is not None:
            scheduler_vm.step()
        if scheduler_va is not None:
            scheduler_va.step()
        
        # Average losses for this epoch
        avg_loss = running_loss / n_batches
        avg_cost = running_cost / n_batches
        avg_gen_vio = running_gen_vio / n_batches
        avg_branch_pf = running_branch_pf / n_batches
        avg_branch_ang = running_branch_ang / n_batches
        avg_load_dev = running_load_dev / n_batches
        
        # Store in history
        loss_history['total'].append(avg_loss)
        loss_history['cost'].append(avg_cost)
        loss_history['gen_vio'].append(avg_gen_vio)
        loss_history['branch_pf_vio'].append(avg_branch_pf)
        loss_history['branch_ang_vio'].append(avg_branch_ang)
        loss_history['load_dev'].append(avg_load_dev)
        
        # Print progress
        if (epoch + 1) % config.p_epoch == 0:
            weights = loss_fn.weight_scheduler.get_weights() if use_adaptive else {}
            print(f'Epoch {epoch+1}/{n_epochs}: Total={avg_loss:.4f}, '
                  f'Cost={avg_cost:.2f}, GenVio={avg_gen_vio:.6f}, '
                  f'BranchPF={avg_branch_pf:.6f}, LoadDev={avg_load_dev:.6f}')
            if use_adaptive and (epoch + 1) % (config.p_epoch * 5) == 0:
                print(f'  Weights: k_g={weights.get("k_g", 0):.2f}, '
                      f'k_Sl={weights.get("k_Sl", 0):.2f}, k_d={weights.get("k_d", 0):.2f}')
        
        # Save models periodically
        if (epoch + 1) % 100 == 0 and (epoch + 1) >= config.s_epoch:
            save_path_vm = f'{config.PATHVms}_{model_type}_unsup_E{epoch+1}F{config.flagVm}.pth'
            save_path_va = f'{config.PATHVas}_{model_type}_unsup_E{epoch+1}F{config.flagVa}.pth'
            torch.save(model_vm.state_dict(), save_path_vm, _use_new_zipfile_serialization=False)
            torch.save(model_va.state_dict(), save_path_va, _use_new_zipfile_serialization=False)
            print(f'  Models saved: {save_path_vm}, {save_path_va}')
    
    time_train = time.process_time() - start_time
    print(f'\nUnsupervised training completed in {time_train:.2f} seconds ({time_train/60:.2f} minutes)')
    
    # Save final models
    final_path_vm = f'{config.PATHVm[:-4]}_{model_type}_unsup.pth'
    final_path_va = f'{config.PATHVa[:-4]}_{model_type}_unsup.pth'
    torch.save(model_vm.state_dict(), final_path_vm, _use_new_zipfile_serialization=False)
    torch.save(model_va.state_dict(), final_path_va, _use_new_zipfile_serialization=False)
    print(f'Final models saved: {final_path_vm}, {final_path_va}')
    
    return model_vm, model_va, loss_history, time_train


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
            # ==================== Unsupervised Training ====================
            print("\n" + "=" * 60)
            print("Unsupervised Training Mode (Physics-based Loss)")
            print("=" * 60)
            
            if not UNSUPERVISED_AVAILABLE:
                raise ImportError("Unsupervised training requires unsupervised_loss.py module")
            
            # Joint training of Vm and Va using unsupervised loss
            model_vm, model_va, loss_history, time_train = train_unsupervised(
                config, model_vm, model_va, optimizer_vm, optimizer_va,
                dataloaders['train_vm'],  # Use Vm loader (x is the same for both)
                sys_data, config.device, model_type=model_type,
                pretrain_model_vm=pretrain_model_vm,
                pretrain_model_va=pretrain_model_va,
                scheduler_vm=scheduler_vm, scheduler_va=scheduler_va
            )
            
            # Convert loss history to separate lists for compatibility
            lossvm = loss_history['total']
            lossva = loss_history['cost']  # Use cost as Va placeholder
            
            # Plot unsupervised training curves
            plot_unsupervised_training_curves(loss_history)
            
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


if __name__ == "__main__":
    main()

