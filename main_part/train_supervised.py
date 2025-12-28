#!/usr/bin/env python
# coding: utf-8
# Training Script for DeepOPF-V 
# supervised training mode
# Extended to support multiple model types: VAE, Flow, Diffusion, GAN, etc.
# Extended to support multi-preference supervised training with Flow models
# Author: Peng Yue
# Date: December 15th, 2025

import torch
import torch.nn as nn 
import time
import os
import sys 
import math
import random
import numpy as np

# Add parent directory to path for flow_model imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from models import get_available_model_types
from data_loader import load_all_data, load_multi_preference_dataset, create_multi_preference_dataloader
from utils import save_results, plot_training_curves 
from unified_eval import build_ctx_from_supervised, SupervisedPredictor, evaluate_unified


def wrap_angle_difference(dx, NPred_Va):
    """
    Wrap angle difference to [-pi, pi] range for Va dimensions.
    
    This is critical for training preference trajectory models, as it ensures
    the velocity field learns the shortest-arc path between angles, avoiding
    artificial large gradients from 2π jumps.
    
    Optimized version: Uses pure PyTorch to avoid CPU/GPU transfers.
    
    Args:
        dx: Difference tensor [..., output_dim] (can be torch.Tensor or numpy array)
        NPred_Va: Number of Va dimensions (first NPred_Va elements)
        
    Returns:
        dx_wrapped: Wrapped difference with same type as input
    """
    is_torch = torch.is_tensor(dx)
    
    if is_torch:
        # Pure PyTorch implementation (fast, no CPU/GPU transfer)
        dx_wrapped = dx.clone()
        if NPred_Va > 0:
            # Use PyTorch's atan2 for angle wrapping (vectorized, GPU-friendly)
            dx_angle = dx[..., :NPred_Va]
            dx_wrapped[..., :NPred_Va] = torch.atan2(torch.sin(dx_angle), torch.cos(dx_angle))
        return dx_wrapped
    else:
        # NumPy fallback for non-tensor inputs
        dx_np = np.asarray(dx)
        dx_wrapped = dx_np.copy()
        
        if NPred_Va > 0:
            # Wrap angle difference to [-pi, pi] range for Va dimensions
            for dim_idx in range(min(NPred_Va, dx_np.shape[-1])):
                dx_angle = dx_np[..., dim_idx]
                dx_wrapped[..., dim_idx] = np.arctan2(np.sin(dx_angle), np.cos(dx_angle))
        
        return dx_wrapped


def interpolate_bridge(x_a, x_b, t, NPred_Va):
    """
    Interpolate between two states x_a and x_b at interpolation parameter t ∈ [0, 1].
    
    Optimized version: Assumes inputs are 1D tensors and t is a scalar float.
    This is faster for the common case in Flow Matching training.
    
    This function implements Flow Matching bridge interpolation:
    - Va dimensions: shortest-arc interpolation (wrapped)
    - Vm dimensions: linear interpolation
    
    Args:
        x_a: Starting state [output_dim] (1D tensor)
        x_b: Ending state [output_dim] (1D tensor)
        t: Interpolation parameter (scalar float in [0, 1])
        NPred_Va: Number of Va dimensions (first NPred_Va elements)
        
    Returns:
        x_t: Interpolated state at t [output_dim] (1D tensor)
    """
    # Optimized: assume x_a, x_b are 1D tensors and t is scalar
    # This avoids unnecessary dimension checks and operations
    output_dim = x_a.shape[0]
    x_t = x_a.clone()
    
    # Va dimensions: shortest-arc interpolation
    if NPred_Va > 0:
        va_a = x_a[:NPred_Va]
        va_b = x_b[:NPred_Va]
        
        # Compute wrapped angle difference (pure PyTorch, no CPU/GPU transfer)
        dva = va_b - va_a
        dva_wrapped = wrap_angle_difference(dva, NPred_Va)
        
        # Interpolate: θ_t = θ_a + t * Δθ_wrapped
        x_t[:NPred_Va] = va_a + t * dva_wrapped
    
    # Vm dimensions: linear interpolation
    if NPred_Va < output_dim:
        vm_a = x_a[NPred_Va:]
        vm_b = x_b[NPred_Va:]
        x_t[NPred_Va:] = (1 - t) * vm_a + t * vm_b
    
    return x_t


def sample_single_flow_matching_pair(solutions, lambda_normalized_tensor,
                                     strategy='mixed', min_dlambda=0.0, device='cpu'):
    """
    Sample a single pair of (λ, x) for Flow Matching training (optimized version).
    
    This is faster than sample_flow_matching_pairs because it only samples one pair
    instead of creating all possible pairs and then choosing one.
    Uses pre-computed lambda_normalized_tensor for faster access.
    
    Args:
        solutions: List of solution tensors [x_0, x_1, ..., x_{K-1}] for one scene
        lambda_normalized_tensor: Pre-computed normalized lambda tensor [K] on device
        strategy: Sampling strategy ('adjacent', 'random', 'mixed')
        min_dlambda: Minimum normalized Δλ to filter out (default: 0.0 = no filtering)
        device: Device for tensors
        
    Returns:
        pair: Tuple (λ_a_norm, x_a, λ_b_norm, x_b, t, dlambda_norm) or None if no valid pair
    """
    K = len(solutions)
    if K < 2:
        return None
    
    max_attempts = 10  # Maximum attempts to find a valid pair
    for _ in range(max_attempts):
        if strategy == 'adjacent':
            # Sample a random adjacent pair
            k = random.randint(0, K - 2)
            idx_a, idx_b = k, k + 1
            
        elif strategy == 'random':
            # Sample two random distinct indices
            indices = torch.randperm(K, device=device)[:2]
            idx_a, idx_b = indices[0].item(), indices[1].item()
            if idx_b < idx_a:
                idx_a, idx_b = idx_b, idx_a
            if idx_a == idx_b:
                continue
                
        elif strategy == 'mixed':
            # 50% chance adjacent, 50% chance random
            if random.random() < 0.5:
                # Adjacent
                k = random.randint(0, K - 2)
                idx_a, idx_b = k, k + 1
            else:
                # Random
                indices = torch.randperm(K, device=device)[:2]
                idx_a, idx_b = indices[0].item(), indices[1].item()
                if idx_b < idx_a:
                    idx_a, idx_b = idx_b, idx_a
                if idx_a == idx_b or idx_b >= K:
                    continue
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        
        # Get normalized lambda values from pre-computed tensor (faster than dict lookup)
        lambda_a_norm_val = lambda_normalized_tensor[idx_a].item()
        lambda_b_norm_val = lambda_normalized_tensor[idx_b].item()
        dlambda_norm_val = lambda_b_norm_val - lambda_a_norm_val
        
        # Check minimum dlambda
        if dlambda_norm_val < min_dlambda:
            continue
        
        # Create tensors (only for the selected pair) - use unsqueeze for faster creation
        lambda_a_norm = lambda_normalized_tensor[idx_a].unsqueeze(0).unsqueeze(0)  # [1, 1]
        lambda_b_norm = lambda_normalized_tensor[idx_b].unsqueeze(0).unsqueeze(0)  # [1, 1]
        dlambda_norm = torch.tensor([[dlambda_norm_val]], device=device, dtype=torch.float32)
        t = torch.rand(1, device=device)
        
        return (lambda_a_norm, solutions[idx_a], lambda_b_norm, solutions[idx_b], t, dlambda_norm)
    
    # Failed to find valid pair after max_attempts
    return None


def sample_flow_matching_pairs(solutions, lambda_values, lambda_normalized_dict, 
                                strategy='mixed', min_dlambda=0.0, device='cpu'):
    """
    Sample pairs of (λ, x) for Flow Matching training.
    
    Args:
        solutions: List of solution tensors [x_0, x_1, ..., x_{K-1}] for one scene
        lambda_values: List of lambda_carbon values [λ_0, λ_1, ..., λ_{K-1}]
        lambda_normalized_dict: Dictionary mapping lambda_carbon to normalized lambda
        strategy: Sampling strategy ('adjacent', 'random', 'mixed')
        min_dlambda: Minimum normalized Δλ to filter out (default: 0.0 = no filtering)
        device: Device for tensors
        
    Returns:
        pairs: List of tuples [(λ_a_norm, x_a, λ_b_norm, x_b, t, dlambda_norm), ...]
               where t is random interpolation parameter and dlambda_norm is normalized Δλ
    """
    K = len(solutions)
    if K < 2:
        return []
    
    pairs = []
    
    if strategy == 'adjacent':
        # Only sample adjacent pairs (current method)
        for k in range(K - 1):
            lambda_a = lambda_values[k]
            lambda_b = lambda_values[k + 1]
            lambda_a_norm = torch.tensor([[lambda_normalized_dict[lambda_a]]], device=device, dtype=torch.float32)
            lambda_b_norm = torch.tensor([[lambda_normalized_dict[lambda_b]]], device=device, dtype=torch.float32)
            dlambda_norm = lambda_b_norm - lambda_a_norm
            
            if dlambda_norm.item() >= min_dlambda:
                t = torch.rand(1, device=device)  # Random interpolation point
                pairs.append((lambda_a_norm, solutions[k], lambda_b_norm, solutions[k + 1], t, dlambda_norm))
    
    elif strategy == 'random':
        # Randomly sample any two points
        for _ in range(K - 1):  # Sample same number as adjacent
            # Sample two distinct indices
            indices = torch.randperm(K, device=device)[:2]
            idx_a, idx_b = indices[0].item(), indices[1].item()
            
            # Ensure idx_b > idx_a (or swap)
            if idx_b < idx_a:
                idx_a, idx_b = idx_b, idx_a
            
            lambda_a = lambda_values[idx_a]
            lambda_b = lambda_values[idx_b]
            lambda_a_norm = torch.tensor([[lambda_normalized_dict[lambda_a]]], device=device, dtype=torch.float32)
            lambda_b_norm = torch.tensor([[lambda_normalized_dict[lambda_b]]], device=device, dtype=torch.float32)
            dlambda_norm = lambda_b_norm - lambda_a_norm
            
            if dlambda_norm.item() >= min_dlambda:
                t = torch.rand(1, device=device)  # Random interpolation point
                pairs.append((lambda_a_norm, solutions[idx_a], lambda_b_norm, solutions[idx_b], t, dlambda_norm))
    
    elif strategy == 'mixed':
        # 50% adjacent + 50% random
        num_adjacent = (K - 1) // 2
        num_random = (K - 1) - num_adjacent
        
        # Adjacent pairs
        for k in range(num_adjacent):
            if k >= K - 1:
                break
            lambda_a = lambda_values[k]
            lambda_b = lambda_values[k + 1]
            lambda_a_norm = torch.tensor([[lambda_normalized_dict[lambda_a]]], device=device, dtype=torch.float32)
            lambda_b_norm = torch.tensor([[lambda_normalized_dict[lambda_b]]], device=device, dtype=torch.float32)
            dlambda_norm = lambda_b_norm - lambda_a_norm
            
            if dlambda_norm.item() >= min_dlambda:
                t = torch.rand(1, device=device)
                pairs.append((lambda_a_norm, solutions[k], lambda_b_norm, solutions[k + 1], t, dlambda_norm))
        
        # Random pairs
        for _ in range(num_random):
            indices = torch.randperm(K, device=device)[:2]
            idx_a, idx_b = indices[0].item(), indices[1].item()
            
            if idx_b < idx_a:
                idx_a, idx_b = idx_b, idx_a
            
            if idx_a == idx_b or idx_b >= K:
                continue
                
            lambda_a = lambda_values[idx_a]
            lambda_b = lambda_values[idx_b]
            lambda_a_norm = torch.tensor([[lambda_normalized_dict[lambda_a]]], device=device, dtype=torch.float32)
            lambda_b_norm = torch.tensor([[lambda_normalized_dict[lambda_b]]], device=device, dtype=torch.float32)
            dlambda_norm = lambda_b_norm - lambda_a_norm
            
            if dlambda_norm.item() >= min_dlambda:
                t = torch.rand(1, device=device)
                pairs.append((lambda_a_norm, solutions[idx_a], lambda_b_norm, solutions[idx_b], t, dlambda_norm))
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    return pairs


def rk2_step(model, scene, x_current, lambda_current, lambda_next, NPred_Va):
    """
    Perform one RK2 (Heun) integration step for preference trajectory.
    
    This function is used in training to align with inference-time RK2 method.
    
    Args:
        model: Flow model with predict_vec method
        scene: Scene features [B, input_dim]
        x_current: Current state [B, output_dim]
        lambda_current: Current lambda (normalized) [B, 1]
        lambda_next: Next lambda (normalized) [B, 1]
        NPred_Va: Number of Va dimensions (for angle wrapping)
        
    Returns:
        x_next: Next state after RK2 step [B, output_dim]
    """
    dlambda = lambda_next - lambda_current
    
    # Stage 1: Euler step (predictor)
    v0 = model.predict_vec(scene, x_current, lambda_current, lambda_current)
    x_euler = x_current + dlambda * v0
    
    # Stage 2: Use velocity at next point (corrector)
    v1 = model.predict_vec(scene, x_euler, lambda_next, lambda_next)
    
    # Final step: average of v0 and v1
    x_next = x_current + dlambda * 0.5 * (v0 + v1)
    
    return x_next 

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


def train_multi_preference(config, model, multi_pref_data, sys_data, device,
                           model_type='simple', pretrain_model=None, scheduler=None):
    """
    Train a preference-conditioned model for multi-objective OPF.
    
    This function trains a single model that can predict optimal power flow solutions
    for different preference settings (lambda_carbon values).
    
    Supports multiple model types:
    - 'simple': MLP with preference concatenated to input
    - 'vae': VAE with preference concatenated to input
    - 'rectified'/'flow': Flow model with preference-aware MLP (FiLM conditioning)
    
    Training modes (controlled by config.multi_pref_training_mode):
    - 'standard': Standard Flow Matching from anchor (VAE/noise) to target solution
    - 'preference_trajectory': Learn velocity field dx/dλ on preference trajectory
      (from MVP-1 experiments). Model learns to flow along Pareto frontier.
      Initial point can be provided by VAE at λ=0.
    
    Args:
        config: Configuration object
        model: Model to train (type depends on model_type)
        multi_pref_data: Multi-preference data dictionary from load_multi_preference_dataset
        sys_data: Power system data
        device: Device (CPU/GPU)
        model_type: Type of model ('simple', 'vae', 'rectified', 'flow', etc.)
        pretrain_model: Optional pretrained VAE model for anchor generation (flow models)
        scheduler: Learning rate scheduler (optional)
    
    Returns:
        model: Trained model
        losses: Training losses per epoch
        time_train: Total training time
    """
    print('=' * 60)
    print(f'Training Multi-Preference Model - Type: {model_type}')
    print('=' * 60)
    
    # Extract training data (only use training set, not validation set)
    x_train = multi_pref_data['x_train'].to(device)
    y_train_by_pref = multi_pref_data['y_train_by_pref']
    lambda_carbon_values = multi_pref_data['lambda_carbon_values']
    n_train = multi_pref_data['n_train']
    n_val = multi_pref_data['n_val']
    Vscale = multi_pref_data['Vscale'].to(device)
    Vbias = multi_pref_data['Vbias'].to(device)
    
    # Move y_train tensors to device
    y_train_by_pref_device = {lc: y.to(device) for lc, y in y_train_by_pref.items()}
    
    print(f"\nTraining data:")
    print(f"  Training samples: {n_train}")
    print(f"  Validation samples: {n_val} (reserved for evaluation)")
    print(f"  Preferences: {len(lambda_carbon_values)}")
    print(f"  Lambda carbon range: [{lambda_carbon_values[0]:.2f}, {lambda_carbon_values[-1]:.2f}]")
    
    # Get training hyperparameters
    num_epochs = getattr(config, 'multi_pref_epochs', config.EpochVm)
    batch_size = getattr(config, 'ngt_batch_size', config.batch_size_training)
    learning_rate = getattr(config, 'multi_pref_lr', config.Lrm)
    flow_type = getattr(config, 'multi_pref_flow_type', 'rectified')
    
    print(f"\nTraining config:")
    print(f"  Model type: {model_type}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    if model_type in ['rectified', 'flow', 'gaussian', 'conditional', 'interpolation']:
        print(f"  Flow type: {flow_type}")
    
    # Create optimizer
    weight_decay = getattr(config, 'weight_decay', 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Create scheduler if not provided
    if scheduler is None and hasattr(config, 'learning_rate_decay') and config.learning_rate_decay:
        step_size, gamma = config.learning_rate_decay
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # Create dataloader
    dataloader = create_multi_preference_dataloader(multi_pref_data, config, shuffle=True)
    
    # Training loop
    losses = []
    start_time = time.process_time()
    
    # Normalize lambda_carbon values for model input
    lc_max = max(lambda_carbon_values) if max(lambda_carbon_values) > 0 else 1.0
    
    # Get VAE beta from config
    vae_beta = getattr(config, 'vae_beta', 1.0)
    
    # Criterion for simple models
    criterion = nn.MSELoss()
    
    # Training mode: 'standard' (Flow Matching from anchor to target) or 'preference_trajectory' (learn dx/dλ on trajectory)
    training_mode = getattr(config, 'multi_pref_training_mode', 'standard')  # Default: standard Flow Matching
    print(f"  Training mode: {training_mode}")
    
    # Loss weights for preference trajectory training
    if training_mode == 'preference_trajectory':
        alpha = getattr(config, 'multi_pref_loss_alpha', 1.0)
        beta = getattr(config, 'multi_pref_loss_beta', 0.5)
        gamma = getattr(config, 'multi_pref_loss_gamma', 0.0)
        print(f"  Loss weights: alpha (Lv) = {alpha}, beta (L1) = {beta}, gamma (Lroll) = {gamma}")
        if gamma > 0:
            print(f"    -> Combined loss: L = {alpha} * Lv + {beta} * L1 + {gamma} * Lroll")
        else:
            print(f"    -> Combined loss: L = {alpha} * Lv + {beta} * L1")
        
        # Scheduled Sampling configuration
        use_scheduled_sampling = getattr(config, 'multi_pref_scheduled_sampling', True)
        scheduled_sampling_p_min = getattr(config, 'multi_pref_scheduled_sampling_p_min', 0.2)
        if use_scheduled_sampling:
            print(f"  Scheduled Sampling: enabled (p: 1.0 -> {scheduled_sampling_p_min})")
        else:
            print(f"  Scheduled Sampling: disabled")
        
        # Multi-step rollout configuration
        rollout_horizon = getattr(config, 'multi_pref_rollout_horizon', 4)
        rollout_use_rk2 = getattr(config, 'multi_pref_rollout_use_rk2', True)
        if gamma > 0:
            print(f"  Multi-step rollout: H = {rollout_horizon}, method = {'RK2' if rollout_use_rk2 else 'Euler'}")
    
    # Preference sampling strategy: 'batch' (same for all samples) or 'sample' (each sample different)
    pref_sampling_strategy = getattr(config, 'multi_pref_sampling_strategy', 'sample')  # Default: per-sample
    
    print(f"  Preference sampling strategy: {pref_sampling_strategy}")
    if pref_sampling_strategy == 'sample':
        print(f"    -> Each sample in batch will have independently sampled preference")
    else:
        print(f"    -> All samples in batch will share the same preference")
    
    # Normalize lambda_carbon values for model input (used in preference_trajectory mode)
    lambda_carbon_sorted = sorted(lambda_carbon_values)
    lambda_min = lambda_carbon_sorted[0]
    lambda_max = lambda_carbon_sorted[-1]
    lambda_normalized_dict = {
        lc: (lc - lambda_min) / (lambda_max - lambda_min) if lambda_max > lambda_min else 0.0
        for lc in lambda_carbon_sorted
    }
    
    # Get NPred_Va for Va wrap handling in preference_trajectory mode
    NPred_Va = multi_pref_data.get('NPred_Va', None)
    if NPred_Va is None:
        # Fallback: infer from bus_Pnet_noslack_all
        bus_Pnet_noslack_all = multi_pref_data.get('bus_Pnet_noslack_all')
        if bus_Pnet_noslack_all is not None:
            NPred_Va = len(bus_Pnet_noslack_all)
        else:
            # Last resort: assume half of output_dim
            NPred_Va = output_dim // 2
    print(f"  NPred_Va (for Va wrap): {NPred_Va}")
    
    # ==================== Preference Trajectory Training Mode Setup ====================
    if training_mode == 'preference_trajectory' and model_type in ['rectified', 'flow', 'gaussian', 'conditional', 'interpolation']:
        # Check if Flow Matching is enabled (only check once, before training loop)
        use_flow_matching = getattr(config, 'multi_pref_flow_matching', False)
        fm_strategy = getattr(config, 'multi_pref_fm_strategy', 'mixed')
        fm_min_dlambda = getattr(config, 'multi_pref_fm_min_dlambda', 0.0)
        fm_weight_by_dlambda = getattr(config, 'multi_pref_fm_weight_by_dlambda', False)
        
        if use_flow_matching:
            print(f"  Flow Matching: enabled (strategy={fm_strategy}, min_dlambda={fm_min_dlambda})")
        else:
            print(f"  Flow Matching: disabled (using adjacent-point sampling)")
        
        # Pre-compute normalized lambda tensor for faster access (optimization)
        lambda_normalized_tensor = torch.tensor(
            [lambda_normalized_dict[lc] for lc in lambda_carbon_sorted],
            device=device, dtype=torch.float32
        )  # [K] shape, pre-computed once
    else:
        use_flow_matching = False
        fm_strategy = None
        fm_min_dlambda = 0.0
        fm_weight_by_dlambda = False
        lambda_normalized_tensor = None
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # ==================== Preference Trajectory Training Mode ====================
        if training_mode == 'preference_trajectory' and model_type in ['rectified', 'flow', 'gaussian', 'conditional', 'interpolation']:
            
            # Sample training pairs from trajectory segments
            for batch_x, batch_indices in dataloader:
                batch_x = batch_x.to(device)
                batch_indices = batch_indices.to(device)
                batch_size_actual = batch_x.shape[0]
                
                optimizer.zero_grad()
                
                if use_flow_matching:
                    # ==================== Flow Matching Sampling ====================
                    # Sample pairs using Flow Matching strategy (adjacent, random, or mixed)
                    batch_x_t = []  # Interpolated state x_t
                    batch_lambda_t = []  # Interpolated lambda λ_t
                    batch_v_target = []  # Target velocity v*
                    batch_dlambda_norm = []  # Normalized Δλ (for weighting)
                    batch_scene = []  # Scene features s
                    batch_solutions = []  # Store all solutions for each sample (for Lroll)
                    batch_lambda_lists = []  # Store aligned lambda lists for each sample (for Lroll)
                    batch_x_b = []  # Endpoint x_b (for L1 endpoint consistency)
                    batch_lambda_b = []  # Endpoint lambda_b (for L1 endpoint consistency)
                    
                    for i in range(batch_size_actual):
                        sample_idx = batch_indices[i].item()
                        scene_features = batch_x[i]
                        
                        # Get solutions for this sample across all preferences
                        # CRITICAL: Build solutions, lambda_list, and lambda_norm_list together to ensure index alignment
                        # If some preferences are missing, we skip them, so solutions may be shorter
                        # than the full lambda_carbon_sorted list. We must use aligned indices.
                        solutions = []
                        lambda_list = []  # Aligned lambda_carbon list (for Lroll)
                        lambda_norm_list = []  # Aligned normalized lambda list (for FM sampling)
                        for lc in lambda_carbon_sorted:
                            if lc in y_train_by_pref_device:
                                solutions.append(y_train_by_pref_device[lc][sample_idx])
                                lambda_list.append(lc)  # Store aligned lambda_carbon (for Lroll)
                                lambda_norm_list.append(lambda_normalized_dict[lc])  # Store aligned normalized lambda (for FM)
                        
                        if len(solutions) < 2:
                            # Skip if not enough preferences
                            continue
                        
                        # Convert lambda_norm_list to tensor for faster access
                        lambda_norm_tensor = torch.tensor(lambda_norm_list, device=device, dtype=torch.float32)
                        
                        # Sample a single Flow Matching pair (optimized: don't create all pairs)
                        # This is much faster than creating all pairs and then choosing one
                        # Use aligned lambda_norm_tensor to ensure index matching with solutions
                        pair = sample_single_flow_matching_pair(
                            solutions, lambda_norm_tensor,
                            strategy=fm_strategy, min_dlambda=fm_min_dlambda, device=device
                        )
                        
                        if pair is None:
                            continue
                        
                        # Store solutions and aligned lambda_list for Lroll (after pair check to keep sync with batch_scene)
                        batch_solutions.append(solutions)
                        batch_lambda_lists.append(lambda_list)  # Store aligned lambda_list for Lroll
                        
                        # Unpack the single pair
                        lambda_a_norm, x_a, lambda_b_norm, x_b, t, dlambda_norm = pair
                        
                        # Flatten dimensions for proper broadcasting
                        # lambda_a_norm, lambda_b_norm, dlambda_norm are [1, 1], squeeze to scalar
                        lambda_a_val = lambda_a_norm.squeeze()  # scalar
                        lambda_b_val = lambda_b_norm.squeeze()  # scalar
                        dlambda_val = dlambda_norm.squeeze()    # scalar
                        t_val = t.squeeze()  # scalar
                        
                        # Interpolate to get x_t (optimized: assumes 1D tensors and scalar t)
                        x_t = interpolate_bridge(x_a, x_b, t_val.item(), NPred_Va)
                        # x_t is already 1D [output_dim] from optimized interpolate_bridge
                        
                        # Compute target velocity: v* = (x_b - x_a) / (λ_b - λ_a)
                        dx = x_b - x_a
                        dx_wrapped = wrap_angle_difference(dx, NPred_Va)
                        v_target = dx_wrapped / (dlambda_val + 1e-8)  # [output_dim]
                        
                        # Interpolate lambda: λ_t = (1-t)λ_a + t*λ_b
                        lambda_t_val = (1 - t_val) * lambda_a_val + t_val * lambda_b_val
                        lambda_t = lambda_t_val.unsqueeze(0)  # [1] for stacking
                        
                        batch_x_t.append(x_t)
                        batch_lambda_t.append(lambda_t)
                        batch_v_target.append(v_target)
                        batch_dlambda_norm.append(dlambda_val.unsqueeze(0))  # [1] for stacking
                        batch_scene.append(scene_features)
                        batch_x_b.append(x_b)  # Store endpoint for L1 loss
                        batch_lambda_b.append(lambda_b_norm)  # Store endpoint lambda for L1 loss
                    
                    if len(batch_x_t) == 0:
                        continue
                    
                    # Stack batches
                    batch_size_fm = len(batch_x_t)  # Actual batch size after filtering
                    x_t = torch.stack(batch_x_t)  # [batch_fm, output_dim] - Interpolated state
                    lambda_t = torch.stack(batch_lambda_t)  # [batch_fm, 1] - Interpolated lambda
                    v_target = torch.stack(batch_v_target)  # [batch_fm, output_dim] - Target velocity
                    dlambda_norm = torch.stack(batch_dlambda_norm)  # [batch_fm, 1] - Normalized Δλ
                    scene_batch = torch.stack(batch_scene)  # [batch_fm, input_dim]
                    x_b_gt = torch.stack(batch_x_b)  # [batch_fm, output_dim] - Ground truth endpoint
                    lambda_b_norm = torch.stack(batch_lambda_b)  # [batch_fm, 1] - Endpoint lambda
                    
                    # Predict velocity at interpolated point
                    v_pred = model.predict_vec(scene_batch, x_t, lambda_t, lambda_t)
                    
                    # Compute velocity loss with optional Δλ weighting
                    if fm_weight_by_dlambda:
                        # Weight by Δλ: w = clip(Δλ, w_min, w_max)
                        w_min, w_max = 0.05, 1.0
                        weights = torch.clamp(dlambda_norm, w_min, w_max)
                        loss_v = torch.mean(weights * torch.sum((v_pred - v_target) ** 2, dim=1, keepdim=True))
                    else:
                        loss_v = criterion(v_pred, v_target)
                    
                    # (B) Endpoint consistency loss (L1): ensures that integrating from x_t along v_pred reaches x_b
                    # This is critical for Flow Matching: the model must learn not just the velocity field,
                    # but also ensure that integrating along the predicted velocity actually reaches the correct endpoint.
                    # Without this constraint, the model might predict velocities that look correct locally,
                    # but lead to trajectory drift and constraint violations when integrated.
                    dlambda_to_b = lambda_b_norm - lambda_t  # [batch_fm, 1] - Lambda increment from x_t to x_b
                    dlambda_to_b = dlambda_to_b + 1e-8  # Avoid division by zero
                    x_pred_to_b = x_t + dlambda_to_b * v_pred  # [batch_fm, output_dim] - Predicted endpoint
                    dx_pred = x_pred_to_b - x_b_gt  # [batch_fm, output_dim] - Prediction error
                    dx_pred_wrapped = wrap_angle_difference(dx_pred, NPred_Va)  # Wrap Va dimensions to avoid 2π jumps
                    loss_l1 = torch.mean(dx_pred_wrapped ** 2)  # MSE on wrapped difference
                    
                    # Note: batch_solutions is already populated for Lroll calculation
                    
                else:
                    # ==================== Adjacent-Point Sampling (Original Method) ====================
                    # For each sample in batch, sample a trajectory segment (k, k+1)
                    batch_x_current = []  # Current state x_k
                    batch_x_next = []     # Next state x_{k+1}
                    batch_lambda_current = []  # Current lambda λ_k
                    batch_lambda_next = []      # Next lambda λ_{k+1}
                    batch_scene = []      # Scene features s
                    batch_solutions = []  # Store all solutions for each sample (for Lroll)
                    batch_lambda_lists = []  # Store aligned lambda lists for each sample (for Lroll)
                    
                    for i in range(batch_size_actual):
                        sample_idx = batch_indices[i].item()
                        scene_features = batch_x[i]
                        
                        # Get solutions for this sample across all preferences
                        # CRITICAL: Build solutions and lambda_list together to ensure index alignment
                        # If some preferences are missing, we skip them, so solutions may be shorter
                        # than the full lambda_carbon_sorted list. We must use aligned indices.
                        solutions = []
                        lambda_list = []  # Aligned lambda_carbon list
                        for lc in lambda_carbon_sorted:
                            if lc in y_train_by_pref_device:
                                solutions.append(y_train_by_pref_device[lc][sample_idx])
                                lambda_list.append(lc)  # Store aligned lambda_carbon
                        
                        if len(solutions) < 2:
                            # Skip if not enough preferences
                            continue
                        
                        # Store solutions and aligned lambda_list for Lroll
                        batch_solutions.append(solutions)
                        batch_lambda_lists.append(lambda_list)  # Store aligned lambda_list for Lroll
                        
                        # Randomly sample a segment (k, k+1) from trajectory
                        # Use aligned indices: k is index in solutions/lambda_list, not lambda_carbon_sorted
                        k = random.randint(0, len(solutions) - 2)
                        x_k = solutions[k]
                        x_k_next = solutions[k+1]
                        lambda_k = lambda_list[k]  # Use aligned lambda_list, not lambda_carbon_sorted
                        lambda_k_next = lambda_list[k+1]  # Use aligned lambda_list, not lambda_carbon_sorted
                        
                        batch_x_current.append(x_k)
                        batch_x_next.append(x_k_next)
                        batch_lambda_current.append(lambda_k)
                        batch_lambda_next.append(lambda_k_next)
                        batch_scene.append(scene_features)
                    
                    if len(batch_x_current) == 0:
                        continue
                    
                    # Stack batches
                    x_current_gt = torch.stack(batch_x_current)  # [batch, output_dim] - Ground truth current state
                    x_next_gt = torch.stack(batch_x_next)  # [batch, output_dim] - Ground truth next state
                    scene_batch = torch.stack(batch_scene)  # [batch, input_dim]
                    
                    # Normalize lambda values
                    lambda_current_norm = torch.tensor(
                        [[lambda_normalized_dict[lc]] for lc in batch_lambda_current],
                        device=device, dtype=torch.float32
                    )  # [batch, 1]
                    lambda_next_norm = torch.tensor(
                        [[lambda_normalized_dict[lc]] for lc in batch_lambda_next],
                        device=device, dtype=torch.float32
                    )  # [batch, 1]
                    
                    # ==================== Scheduled Sampling (adjacent-point mode only) ====================
                    # With probability p, use ground truth x_k; with (1-p), use model prediction x̃_k
                    # This helps model learn to use its own predictions as input (critical for multi-step)
                    # 
                    # Key difference:
                    # - Without SS: Always use GT → model only sees perfect inputs → exposure bias at inference
                    # - With SS: Gradually use predictions → model learns to handle its own errors → better multi-step
                    use_scheduled_sampling = getattr(config, 'multi_pref_scheduled_sampling', True)
                    if use_scheduled_sampling and epoch > 0:  # Skip scheduled sampling in first epoch
                        # Linear decay: p = 1.0 -> p_min over training
                        scheduled_sampling_p_min = getattr(config, 'multi_pref_scheduled_sampling_p_min', 0.2)
                        p = max(scheduled_sampling_p_min, 1.0 - (epoch / num_epochs) * (1.0 - scheduled_sampling_p_min))
                        
                        # For each sample, decide whether to use GT or predicted state
                        use_gt = torch.rand(batch_size_actual, device=device) < p
                        
                        # If using predicted state, predict it from the previous trajectory point
                        # Strategy: For each sample, if we need prediction, roll back one step and predict forward
                        x_current_pred = x_current_gt.clone()  # Initialize with GT
                        
                        # For samples that need prediction (use_gt == False), predict from previous point
                        need_pred = ~use_gt
                        if need_pred.any():
                            # For each sample needing prediction, find its previous point in trajectory
                            # Since we randomly sampled segment (k, k+1), previous point is k-1
                            # We need to reconstruct this from batch_solutions
                            for i in range(batch_size_actual):
                                if need_pred[i] and i < len(batch_solutions):
                                    solutions = batch_solutions[i]
                                    # Find the k index used for this sample
                                    # We stored k in batch_lambda_current, need to find corresponding solution index
                                    lambda_k = batch_lambda_current[i]
                                    try:
                                        k_idx = lambda_carbon_sorted.index(lambda_k)
                                        if k_idx > 0:
                                            # Get previous state x_{k-1}
                                            x_prev = solutions[k_idx - 1]
                                            lambda_prev = lambda_carbon_sorted[k_idx - 1]
                                            lambda_prev_norm = torch.tensor([[lambda_normalized_dict[lambda_prev]]], device=device, dtype=torch.float32)
                                            lambda_k_norm = lambda_current_norm[i:i+1]
                                            
                                            # Predict velocity from x_{k-1} to x_k
                                            scene_i = scene_batch[i:i+1]
                                            v_prev = model.predict_vec(scene_i, x_prev.unsqueeze(0), lambda_prev_norm, lambda_prev_norm)
                                            dlambda_prev = lambda_k_norm - lambda_prev_norm
                                            x_current_pred[i] = x_prev + dlambda_prev.squeeze(0) * v_prev.squeeze(0)
                                    except (ValueError, IndexError):
                                        # Fallback to GT if we can't find previous point
                                        x_current_pred[i] = x_current_gt[i]
                        
                        # Use GT or predicted state based on use_gt mask
                        x_current = torch.where(use_gt.unsqueeze(-1), x_current_gt, x_current_pred)
                    else:
                        x_current = x_current_gt  # Always use ground truth
                    
                    # Compute ground truth velocity: v = (x_{k+1} - x_k) / (λ_{k+1} - λ_k)
                    # CRITICAL: Wrap angle difference for Va dimensions to avoid 2π jumps
                    # This ensures the model learns the shortest-arc path between angles
                    dx = x_next_gt - x_current_gt  # Use GT for velocity target
                    dx_wrapped = wrap_angle_difference(dx, NPred_Va)  # Wrap Va dimensions
                    
                    dlambda = lambda_next_norm - lambda_current_norm
                    dlambda = dlambda + 1e-8  # Avoid division by zero
                    v_target = dx_wrapped / dlambda  # [batch, output_dim]
                    
                    # Predict velocity: model takes [scene, state, time, pref] and outputs v
                    # For preference trajectory training:
                    #   - scene: scene features (load data)
                    #   - state: current state x_k on trajectory (may be GT or predicted)
                    #   - time: use normalized lambda position as "time" (0 to 1 along trajectory)
                    #   - pref: normalized lambda value (preference parameter)
                    # Note: In trajectory mode, we use lambda position as both time and preference
                    # This allows the model to learn the velocity field along the Pareto frontier
                    t_trajectory = lambda_current_norm  # Use normalized lambda position as "time" parameter
                    pref_trajectory = lambda_current_norm  # Use normalized lambda as preference parameter
                    
                    # Predict velocity at current state with current lambda
                    v_pred = model.predict_vec(scene_batch, x_current, t_trajectory, pref_trajectory)
                    
                    # ==================== Loss Components (adjacent-point mode) ====================
                    # (A) Teacher-forcing velocity loss (Lv)
                    loss_v = criterion(v_pred, v_target)
                    
                    # (B) One-step state loss (L1): directly constrains one-step prediction error
                    # This helps reduce local error accumulation and improves multi-step stability
                    x_pred_one_step = x_current + dlambda * v_pred  # One-step prediction
                    dx_pred = x_pred_one_step - x_next_gt  # Prediction error (use GT for target)
                    dx_pred_wrapped = wrap_angle_difference(dx_pred, NPred_Va)  # Wrap Va dimensions to avoid 2π jumps
                    # Compute L1 loss using wrapped difference (ensures shortest-arc path for angles)
                    loss_l1 = torch.mean(dx_pred_wrapped ** 2)  # MSE on wrapped difference
                
                # ==================== Common: Multi-step unroll loss (Lroll) ====================
                # (C) Multi-step unroll loss: rollout H steps and compute error at each step
                # This directly addresses error accumulation (ratio≈4.22 problem)
                loss_roll = torch.tensor(0.0, device=device)
                gamma = getattr(config, 'multi_pref_loss_gamma', 0.0)
                rollout_horizon = getattr(config, 'multi_pref_rollout_horizon', 4)
                rollout_use_rk2 = getattr(config, 'multi_pref_rollout_use_rk2', True)
                
                if gamma > 0 and len(batch_solutions) > 0:
                    # Check if we have enough samples with sufficient trajectory length
                    valid_samples = [i for i, sols in enumerate(batch_solutions) if len(sols) >= rollout_horizon + 1]
                    
                    if len(valid_samples) > 0:
                        # Sample a random starting point k for each valid sample
                        k_starts = []
                        x_rollout_gt_list = []  # [sample_idx, h, output_dim]
                        lambda_rollout_gt_list = []  # [sample_idx, h, 1]
                        
                        for i in valid_samples:
                            solutions = batch_solutions[i]
                            lambda_list = batch_lambda_lists[i]  # Use aligned lambda_list, not lambda_carbon_sorted
                            max_start_k = len(solutions) - rollout_horizon - 1
                            k_start = random.randint(0, max_start_k)
                            k_starts.append(k_start)
                            
                            # Get ground truth states for this sample's rollout
                            # CRITICAL: Use aligned indices - k_h is index in solutions/lambda_list, not lambda_carbon_sorted
                            x_gt_sample = []
                            lambda_gt_sample = []
                            for h in range(rollout_horizon + 1):
                                k_h = k_start + h
                                x_gt_sample.append(solutions[k_h])
                                lambda_gt_sample.append(lambda_list[k_h])  # Use aligned lambda_list
                            x_rollout_gt_list.append(x_gt_sample)
                            lambda_rollout_gt_list.append(lambda_gt_sample)
                        
                        if len(x_rollout_gt_list) > 0:
                            # Stack ground truth states: [num_valid, H+1, output_dim]
                            x_rollout_gt = torch.stack([torch.stack(x_gt) for x_gt in x_rollout_gt_list])  # [num_valid, H+1, output_dim]
                            lambda_rollout_gt = torch.stack([
                                torch.tensor([[lambda_normalized_dict[lc]] for lc in lambda_gt], device=device, dtype=torch.float32)
                                for lambda_gt in lambda_rollout_gt_list
                            ])  # [num_valid, H+1, 1]
                            
                            # Get corresponding scene features
                            # Note: scene_batch size matches batch_solutions size (after filtering)
                            scene_rollout = scene_batch[valid_samples]  # [num_valid, input_dim]
                            
                            # Start from ground truth at k
                            x_rollout_pred = [x_rollout_gt[:, 0, :]]  # [num_valid, output_dim]
                            
                            # Rollout H steps
                            for h in range(rollout_horizon):
                                lambda_h = lambda_rollout_gt[:, h, :]  # [num_valid, 1]
                                lambda_h_next = lambda_rollout_gt[:, h + 1, :]  # [num_valid, 1]
                                
                                if rollout_use_rk2:
                                    # Use RK2 for rollout (aligned with inference)
                                    x_next_pred = rk2_step(
                                        model, scene_rollout, x_rollout_pred[-1],
                                        lambda_h, lambda_h_next, NPred_Va
                                    )
                                else:
                                    # Use Euler for rollout
                                    v_h = model.predict_vec(scene_rollout, x_rollout_pred[-1], lambda_h, lambda_h)
                                    dlambda_h = lambda_h_next - lambda_h
                                    x_next_pred = x_rollout_pred[-1] + dlambda_h * v_h
                                
                                x_rollout_pred.append(x_next_pred)
                            
                            # Compute loss at each step (uniform weights)
                            rollout_losses = []
                            for h in range(1, rollout_horizon + 1):  # Skip h=0 (initial state)
                                dx_h = x_rollout_pred[h] - x_rollout_gt[:, h, :]  # [num_valid, output_dim]
                                dx_h_wrapped = wrap_angle_difference(dx_h, NPred_Va)
                                loss_h = torch.mean(dx_h_wrapped ** 2)
                                rollout_losses.append(loss_h)
                            
                            if len(rollout_losses) > 0:
                                loss_roll = torch.mean(torch.stack(rollout_losses))
                
                # Combined loss: L = α * Lv + β * L1 + γ * Lroll
                alpha = getattr(config, 'multi_pref_loss_alpha', 1.0)  # Weight for velocity loss
                beta = getattr(config, 'multi_pref_loss_beta', 0.5)    # Weight for one-step loss
                loss = alpha * loss_v + beta * loss_l1 + gamma * loss_roll
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Store individual losses for logging (only for first batch of first epoch to avoid spam)
                if epoch == 0 and num_batches == 1:
                    if gamma > 0:
                        print(f"  [First batch] Lv = {loss_v.item():.6f}, L1 = {loss_l1.item():.6f}, Lroll = {loss_roll.item():.6f}, L_total = {loss.item():.6f}")
                    else:
                        print(f"  [First batch] Lv = {loss_v.item():.6f}, L1 = {loss_l1.item():.6f}, L_total = {loss.item():.6f}")
        
        # ==================== Standard Training Mode ====================
        else:
            for batch_x, batch_indices in dataloader:
                batch_x = batch_x.to(device)
                batch_indices = batch_indices.to(device)
                batch_size_actual = batch_x.shape[0]
                
                if pref_sampling_strategy == 'sample':
                    # Randomly sample a preference for each sample in the batch
                    # Each sample can have a different preference
                    lc_batch = [random.choice(lambda_carbon_values) for _ in range(batch_size_actual)]
                    
                    # Get corresponding y values for each sample based on its preference
                    # Since each sample may have different preference, we need to gather them individually
                    # More efficient: use list comprehension with indexing
                    batch_y = torch.stack([
                        y_train_by_pref_device[lc][batch_indices[i]] 
                        for i, lc in enumerate(lc_batch)
                    ], dim=0)
                    
                    # Create preference tensor (normalized) - each sample has its own preference
                    pref = torch.tensor([[lc / lc_max] for lc in lc_batch], device=device, dtype=torch.float32)
                else:
                    # Original batch-level sampling: same preference for all samples in batch
                    lc = random.choice(lambda_carbon_values)
                    batch_y = y_train_by_pref_device[lc][batch_indices]
                    pref = torch.full((batch_size_actual, 1), lc / lc_max, device=device)
                
                optimizer.zero_grad()
                
                # ==================== Model-specific training logic ====================
                if model_type == 'simple':
                    # Simple MLP: concatenate preference to input
                    x_with_pref = torch.cat([batch_x, pref], dim=1)
                    y_pred = model(x_with_pref)
                    loss = criterion(y_pred, batch_y)
                    
                elif model_type == 'vae':
                    # VAE: use preference_aware_mlp if available, otherwise concatenate
                    use_pref_aware = getattr(config, 'vae_use_preference_aware', True)
                    if use_pref_aware and hasattr(model, 'pref_dim') and model.pref_dim > 0:
                        # Use preference_aware_mlp with FiLM conditioning
                        y_pred, mean, logvar = model.encoder_decode(batch_x, batch_y, pref=pref)
                    else:
                        # Fallback: concatenate preference to input
                        x_with_pref = torch.cat([batch_x, pref], dim=1)
                        y_pred, mean, logvar = model.encoder_decode(x_with_pref, batch_y)
                    loss = model.loss(y_pred, batch_y, mean, logvar, beta=vae_beta)
                    
                elif model_type in ['rectified', 'flow', 'gaussian', 'conditional', 'interpolation']:
                    # Flow Matching with preference conditioning
                    t_batch = torch.rand([batch_size_actual, 1], device=device)
                    
                    # Get anchor points
                    if pretrain_model is not None:
                        with torch.no_grad():
                            # For VAE anchor with preference, use preference_aware_mlp if available
                            use_pref_aware = getattr(config, 'vae_use_preference_aware', True)
                            if use_pref_aware and hasattr(pretrain_model, 'pref_dim') and pretrain_model.pref_dim > 0:
                                # Use preference_aware_mlp with FiLM conditioning
                                # For initial anchor, use lambda=0 (or minimum lambda)
                                lambda_min_val = min(lambda_carbon_values)
                                pref_anchor = torch.full((batch_size_actual, 1), lambda_min_val / lc_max, device=device)
                                z_batch = pretrain_model(batch_x, use_mean=True, pref=pref_anchor)
                            else:
                                # Fallback: concatenate preference to input
                                # Use lambda=0 for anchor
                                lambda_min_val = min(lambda_carbon_values)
                                x_with_pref_anchor = torch.cat([batch_x, torch.full((batch_size_actual, 1), lambda_min_val / lc_max, device=device)], dim=1)
                                z_batch = pretrain_model(x_with_pref_anchor, use_mean=True)
                    else:
                        z_batch = torch.randn_like(batch_y, device=device)
                    
                    # Use 'rectified' as default flow type for 'flow' model_type
                    actual_flow_type = flow_type if model_type != 'flow' else 'rectified'
                    
                    # Flow forward: get interpolation point and target velocity
                    yt, vec_target = model.flow_forward(batch_y, t_batch, z_batch, actual_flow_type)
                    
                    # Predict velocity with preference conditioning
                    vec_pred = model.predict_vec(batch_x, yt, t_batch, pref)
                    
                    # Calculate loss
                    loss = model.loss(batch_y, z_batch, vec_pred, vec_target, actual_flow_type)
                    
                elif model_type == 'diffusion':
                    # Diffusion with preference concatenated to input
                    t_batch = torch.rand([batch_size_actual, 1], device=device)
                    noise = torch.randn_like(batch_y, device=device)
                    
                    x_with_pref = torch.cat([batch_x, pref], dim=1)
                    
                    if pretrain_model is not None:
                        with torch.no_grad():
                            vae_anchor = pretrain_model(x_with_pref, use_mean=True)
                        noise_pred = model.predict_noise_with_anchor(x_with_pref, batch_y, t_batch, noise, vae_anchor)
                    else:
                        noise_pred = model.predict_noise(x_with_pref, batch_y, t_batch, noise)
                    
                    loss = model.loss(noise_pred, noise)
                    
                else:
                    raise ValueError(f"Unsupported model type for multi-preference training: {model_type}")
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step()
        
        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)
        
        # Print progress
        if (epoch + 1) % config.p_epoch == 0:
            if training_mode == 'preference_trajectory' and model_type in ['rectified', 'flow', 'gaussian', 'conditional', 'interpolation']:
                # For preference trajectory mode, we can't easily track individual losses here
                # (they're computed per batch). The first batch log will show the breakdown.
                print(f'Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}')
            else:
                print(f'Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}')
        
        # Save checkpoint periodically
        s_epoch = getattr(config, 'multi_pref_s_epoch', getattr(config, 's_epoch', 0))
        if (epoch + 1) % 100 == 0 and (epoch + 1) >= s_epoch:
            # Ensure save directory exists
            os.makedirs(config.model_save_dir, exist_ok=True)
            save_path = f'{config.model_save_dir}/model_multi_pref_{model_type}_E{epoch+1}.pth'
            torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)
            print(f'  Checkpoint saved: {save_path}')
    
    time_train = time.process_time() - start_time
    print(f'\nMulti-preference {model_type} training completed in {time_train:.2f} seconds ({time_train/60:.2f} minutes)')
    
    # ==================== Save Models ====================
    print('\n' + '=' * 60)
    print('Saving Trained Models')
    print('=' * 60)
    
    # Ensure save directory exists
    os.makedirs(config.model_save_dir, exist_ok=True)
    print(f'  Save directory: {config.model_save_dir}')
    
    # Save final model (always save at the end)
    final_path = f'{config.model_save_dir}/model_multi_pref_{model_type}_final.pth'
    try:
        torch.save(model.state_dict(), final_path, _use_new_zipfile_serialization=False)
        # Verify file was saved
        if os.path.exists(final_path):
            file_size = os.path.getsize(final_path) / (1024 * 1024)  # MB
            print(f'  [SUCCESS] Final model saved: {final_path} ({file_size:.2f} MB)')
        else:
            print(f'  [WARNING] Model file not found after saving: {final_path}')
    except Exception as e:
        print(f'  [ERROR] Failed to save final model: {e}')
    
    # Also save with epoch number for reference
    final_path_epoch = f'{config.model_save_dir}/model_multi_pref_{model_type}_E{num_epochs}.pth'
    try:
        torch.save(model.state_dict(), final_path_epoch, _use_new_zipfile_serialization=False)
        if os.path.exists(final_path_epoch):
            file_size = os.path.getsize(final_path_epoch) / (1024 * 1024)  # MB
            print(f'  [SUCCESS] Final model (with epoch) saved: {final_path_epoch} ({file_size:.2f} MB)')
        else:
            print(f'  [WARNING] Model file not found after saving: {final_path_epoch}')
    except Exception as e:
        print(f'  [ERROR] Failed to save epoch model: {e}')
    
    # For rectified/flow models: if VAE anchor was used, note that it should be saved separately
    if model_type in ['rectified', 'flow', 'gaussian', 'conditional', 'interpolation']:
        if pretrain_model is not None:
            print(f'  [INFO] VAE anchor model was used but not saved here (load from pre-trained)')
            print(f'         If you trained a new VAE anchor, save it separately.')
    
    print('=' * 60)
    
    return model, losses, time_train


def main(debug=False):
    """
    Main function with support for training.
    
    Supports two modes:
    1. Standard supervised training (separate Vm/Va models)
    2. Multi-preference supervised training (single preference-conditioned Flow model)
    
    Set config.use_multi_objective_supervised=True or MULTI_PREF_SUPERVISED=True
    to enable multi-preference mode.
    """
    # Load configuration
    config = get_config()
     
    print("=" * 60)
    print(f"DeepOPF-V")
    print("=" * 60)
    
    config.print_config()
    
    # Create output directories if they don't exist
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    print(f"\nModel save directory: {config.model_save_dir}")
    print(f"Results directory: {config.results_dir}")
    
    # Check if multi-preference supervised training is enabled
    use_multi_objective = getattr(config, 'use_multi_objective_supervised', False)
    
    if use_multi_objective:
        # ==================== Multi-Preference Supervised Training ====================
        return main_multi_preference(config, debug)
    else:
        # ==================== Standard Supervised Training ====================
        return main_standard(config, debug)


def main_multi_preference(config, debug=False):
    """
    Main function for multi-preference supervised training.
    
    Trains a single preference-conditioned model on multi-preference dataset.
    Supports multiple model types: simple, vae, rectified/flow, diffusion.
    Uses NGT-style data format and post-processing.
    """
    from models import create_model, NetV, get_available_model_types
    from unified_eval import (
        MultiPreferencePredictor, 
        build_ctx_from_multi_preference,
        evaluate_unified
    )
    
    print("\n" + "=" * 60)
    print("Multi-Preference Supervised Training Mode")
    print("=" * 60)
    
    # Get model type
    model_type = config.model_type
    print(f"\nSelected model type: {model_type}")
    print(f"Available model types: {get_available_model_types()}")
    
    # Load multi-preference dataset
    multi_pref_data, sys_data = load_multi_preference_dataset(config)
    
    # Get dimensions
    input_dim = multi_pref_data['input_dim']
    output_dim = multi_pref_data['output_dim']
    pref_dim = getattr(config, 'pref_dim', 1)
    Vscale = multi_pref_data['Vscale']
    Vbias = multi_pref_data['Vbias']
    
    print(f"\nModel dimensions:")
    print(f"  Input dim: {input_dim}")
    print(f"  Output dim: {output_dim}")
    print(f"  Preference dim: {pref_dim}")
    
    # Import from flow_model
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'flow_model'))
    from net_utiles import FM, VAE, DM
    
    # Create model based on model_type
    pretrain_model = None
    
    if model_type == 'simple':
        # Simple MLP with preference concatenated to input
        # Use NetV with input_dim + pref_dim
        model = NetV(
            input_channels=input_dim + pref_dim,
            output_channels=output_dim,
            hidden_units=config.ngt_hidden_units,
            khidden=config.ngt_khidden,
            Vscale=Vscale,
            Vbias=Vbias
        )
        print(f"\n[Multi-Pref] Created Simple MLP model")
        print(f"  Input dim (with pref): {input_dim + pref_dim}")
        print(f"  Hidden layers: {config.ngt_khidden}")
        
    elif model_type == 'vae':
        # VAE with preference conditioning via FiLM (preference_aware_mlp)
        # Use preference_aware_mlp instead of concatenating preference to input
        use_pref_aware = getattr(config, 'vae_use_preference_aware', True)  # Default: use preference_aware_mlp
        
        if use_pref_aware:
            model = VAE(
                network='preference_aware_mlp',
                input_dim=input_dim,  # Don't concatenate preference
                output_dim=output_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                latent_dim=config.latent_dim,
                output_act=None,
                pred_type='node',
                use_cvae=getattr(config, 'use_cvae', True),
                pref_dim=pref_dim
            )
            print(f"\n[Multi-Pref] Created VAE model with preference_aware_mlp")
            print(f"  Input dim: {input_dim} (preference handled via FiLM)")
            print(f"  Preference dim: {pref_dim}")
        else:
            # Fallback: concatenate preference to input (old method)
            model = VAE(
                network='mlp',
                input_dim=input_dim + pref_dim,
                output_dim=output_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                latent_dim=config.latent_dim,
                output_act=None,
                pred_type='node',
                use_cvae=getattr(config, 'use_cvae', True)
            )
            print(f"\n[Multi-Pref] Created VAE model (preference concatenated)")
            print(f"  Input dim (with pref): {input_dim + pref_dim}")
        
        print(f"  Hidden dim: {config.hidden_dim}")
        print(f"  Latent dim: {config.latent_dim}")
        
    elif model_type in ['rectified', 'flow', 'gaussian', 'conditional', 'interpolation']:
        # Flow model with preference_aware_mlp (FiLM conditioning)
        model = FM(
            network='preference_aware_mlp',
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=getattr(config, 'ngt_flow_hidden_dim', 144),
            num_layers=getattr(config, 'ngt_flow_num_layers', 2),
            time_step=config.time_step,
            output_norm=False,
            pred_type='velocity',
            pref_dim=pref_dim
        )
        print(f"\n[Multi-Pref] Created Flow model with preference_aware_mlp")
        print(f"  Hidden dim: {getattr(config, 'ngt_flow_hidden_dim', 144)}")
        print(f"  Num layers: {getattr(config, 'ngt_flow_num_layers', 2)}")
        
        # Optional: Load VAE anchor model for flow
        if getattr(config, 'multi_pref_use_vae_anchor', False):
            print("\n[Info] Loading VAE anchor model for flow training...")
            # Create VAE with preference_aware_mlp (matching main model)
            use_pref_aware = getattr(config, 'vae_use_preference_aware', True)
            if use_pref_aware:
                pretrain_model = VAE(
                    network='preference_aware_mlp',
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    latent_dim=config.latent_dim,
                    output_act=None,
                    pred_type='node',
                    use_cvae=True,
                    pref_dim=pref_dim
                )
            else:
                # Fallback: concatenate preference to input
                pretrain_model = VAE(
                    network='mlp',
                    input_dim=input_dim + pref_dim,
                    output_dim=output_dim,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    latent_dim=config.latent_dim,
                    output_act=None,
                    pred_type='node',
                    use_cvae=True
                )
            # Try to load pretrained weights
            vae_path = f'{config.model_save_dir}/model_multi_pref_vae_final.pth'
            if os.path.exists(vae_path):
                pretrain_model.load_state_dict(torch.load(vae_path, map_location=config.device, weights_only=True))
                pretrain_model.to(config.device)
                pretrain_model.eval()
                print(f"  Loaded VAE anchor from: {vae_path}")
            else:
                print(f"  VAE anchor not found at {vae_path}, using random noise")
                pretrain_model = None
                
    elif model_type == 'diffusion':
        # Diffusion model with preference concatenated to input
        model = DM(
            network='mlp',
            input_dim=input_dim + pref_dim,
            output_dim=output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            time_step=config.time_step,
            output_norm=False,
            pred_type='node'
        )
        print(f"\n[Multi-Pref] Created Diffusion model")
        print(f"  Input dim (with pref): {input_dim + pref_dim}")
        print(f"  Hidden dim: {config.hidden_dim}")
        
    else:
        raise ValueError(f"Unsupported model type for multi-preference training: {model_type}")
    
    model.to(config.device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    if not debug:
        # Load BRANFT for post-processing
        from data_loader import load_all_data
        _, _, BRANFT = load_all_data(config)
        
        model, losses, train_time = train_multi_preference(
            config, model, multi_pref_data, sys_data, config.device,
            model_type=model_type,
            pretrain_model=pretrain_model
        )
        
        # Verify model was saved
        final_model_path = f'{config.model_save_dir}/model_multi_pref_{model_type}_final.pth'
        if os.path.exists(final_model_path):
            print(f'\n[VERIFIED] Model successfully saved and verified at: {final_model_path}')
        else:
            print(f'\n[WARNING] Model file not found at expected path: {final_model_path}')
            print('  Attempting to save model again...')
            os.makedirs(config.model_save_dir, exist_ok=True)
            torch.save(model.state_dict(), final_model_path, _use_new_zipfile_serialization=False)
            if os.path.exists(final_model_path):
                print(f'  [SUCCESS] Model re-saved: {final_model_path}')
            else:
                print(f'  [ERROR] Failed to save model. Please check permissions and disk space.')
    else:
        print("\n[Debug Mode] Skipping training...")
        # Load pre-trained model if available
        model_path = f'{config.model_save_dir}/model_multi_pref_{model_type}_final.pth'
        if os.path.exists(model_path):
            print(f"  Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=config.device, weights_only=True))
        else:
            print(f"  Warning: Model not found at {model_path}")
        
        from data_loader import load_all_data
        _, _, BRANFT = load_all_data(config)
    
    # Evaluation
    print("\n" + "=" * 80)
    print(f"Running Multi-Preference Evaluation (Model: {model_type})")
    print("=" * 80)
    
    # Evaluate on a few representative preferences
    test_lambda_carbons = [0.0, 25.0, 50.0]
    results_all = {}
    
    for lc in test_lambda_carbons:
        print(f"\n--- Evaluating lambda_carbon = {lc:.2f} ---")
        
        # Build evaluation context with specific preference for ground truth
        ctx = build_ctx_from_multi_preference(
            config, sys_data, multi_pref_data, BRANFT, config.device,
            lambda_carbon=lc  # Use corresponding ground truth labels
        )
        
        # Get training mode (for preference_trajectory mode inference)
        training_mode = getattr(config, 'multi_pref_training_mode', 'standard')
        
        # Create predictor with specific preference
        predictor = MultiPreferencePredictor(
            model=model,
            multi_pref_data=multi_pref_data,
            lambda_carbon=lc,
            model_type=model_type,
            pretrain_model=pretrain_model,
            num_flow_steps=getattr(config, 'multi_pref_flow_steps', 10),
            flow_method='euler',
            training_mode=training_mode
        )
        
        # Run evaluation
        results = evaluate_unified(ctx, predictor, apply_post_processing=True, verbose=True)
        results_all[lc] = results
    
    print("\n" + "=" * 80)
    print("Multi-Preference Evaluation Complete")
    print("=" * 80)
    
    return results_all


def main_standard(config, debug=False):
    """
    Main function for standard supervised training (original logic).
    
    Trains separate Vm and Va models.
    """
    from models import create_model
    
    # Get model type
    model_type = config.model_type
    print(f"\nSelected model type: {model_type}")
    print(f"Available model types: {get_available_model_types()}")
    
    # Load data
    sys_data, dataloaders, BRANFT = load_all_data(config)
    
    # Initialize models based on model_type
    input_channels = sys_data.x_train.shape[1]
    output_channels_vm = sys_data.yvm_train.shape[1]
    output_channels_va = sys_data.yva_train.shape[1]
    
    print(f"\nInput dimension: {input_channels}")
    print(f"Vm output dimension: {output_channels_vm}")
    print(f"Va output dimension: {output_channels_va}") 
    
    # Initialize variables
    model_vm = None
    model_va = None
    pretrain_model_vm = None
    pretrain_model_va = None
    weight_decay = getattr(config, 'weight_decay', 0)
    criterion = nn.MSELoss()  
    
    # ==================== Supervised Training ==================== 
    print("\n" + "=" * 60)
    print("Supervised Training Mode (Label-based Loss)")
    print("=" * 60)
    
    # Create models using factory function
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
        model_vm, _, _ = train_voltage_magnitude(
            config, model_vm, optimizer_vm, dataloaders['train_vm'],
            sys_data, criterion, config.device, model_type=model_type,
            pretrain_model=pretrain_model_vm, scheduler=scheduler_vm
        )
        
        # Train Va model
        model_va, _, _ = train_voltage_angle(
            config, model_va, optimizer_va, dataloaders['train_va'],
            criterion, config.device, model_type=model_type,
            pretrain_model=pretrain_model_va, scheduler=scheduler_va
        )
    else: 
        vm_ckpt_path = "main_part/saved_models/modelvm300r2N1Lm8642E1000_simple.pth"
        va_ckpt_path = "main_part/saved_models/modelva300r2N1La8642E1000_simple.pth"
        print(f"\n[Debug Mode] Loading trained Vm model from {vm_ckpt_path}")
        model_vm.load_state_dict(torch.load(vm_ckpt_path, map_location=config.device, weights_only=True))
        print("  Vm model loaded (weights assigned).")
        print(f"[Debug Mode] Loading trained Va model from {va_ckpt_path}")
        model_va.load_state_dict(torch.load(va_ckpt_path, map_location=config.device, weights_only=True))
        print("  Va model loaded (weights assigned).") 
    
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
    return results_unified   # 返回评估结果 

if __name__ == "__main__": 
    debug = bool(int(os.environ.get('DEBUG', '0')))
    results = main(debug=debug) 

