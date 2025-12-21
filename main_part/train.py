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
from utils import (get_genload,
                   get_vioPQg, get_viobran2, TensorBoardLogger, initialize_flow_model_near_zero,
                   save_results, plot_training_curves, plot_unsupervised_training_curves)
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
 
def compare_ngt_evaluation_results(res_original, res_unified, verbose=True):
    """
    对比NGT模型的原始评估方法和统一评估方法的结果。
    
    Args:
        res_original: 原始评估方法的结果（evaluate_ngt_model 或 evaluate_ngt_flow_model）
        res_unified: 统一评估方法的结果（evaluate_unified）
        verbose: 是否打印详细对比信息
        
    Returns:
        dict: 对比摘要，包含差异信息
    """
    def to_float(x):
        if isinstance(x, (float, int)):
            return float(x)
        if torch.is_tensor(x):
            if x.numel() == 1:
                return float(x.detach().cpu().item())
            else:
                return float(x.detach().cpu().mean().item())
        if isinstance(x, np.ndarray):
            if x.size == 1:
                return float(x.item())
            else:
                return float(np.mean(x))
        return float(x)
    
    if verbose:
        print("\n" + "#" * 80)
        print("Comparing NGT Evaluation Results: ORIGINAL vs UNIFIED")
        print("#" * 80)
    
    # 定义要对比的指标
    keys_mapping = {
        # 原始方法 -> 统一方法
        'mae_Vm': 'mae_Vmtest',
        'mae_Va': 'mae_Vatest',
        'cost_error_percent': ('mre_cost', 'mean'),  # mre_cost可能是tensor，需要取mean
        'Pg_satisfy': ('vio_PQg', 0),
        'Qg_satisfy': ('vio_PQg', 1),
        'Vm_satisfy': None,  # 需要特殊处理
        'branch_ang_satisfy': ('vio_branang', 'mean'),
        'branch_pf_satisfy': ('vio_branpf', 'mean'),
        'Pd_error_percent': ('mre_Pd', 'mean'),
        'Qd_error_percent': ('mre_Qd', 'mean'),
    }
    
    comparison = {}
    max_diff = 0.0
    max_diff_key = None
    
    for orig_key, unified_key in keys_mapping.items():
        if orig_key not in res_original:
            continue
            
        orig_val = to_float(res_original[orig_key])
        
        # 处理不同的键格式
        if unified_key is None:
            # Vm_satisfy 需要特殊处理
            continue
        elif isinstance(unified_key, tuple):
            key, idx_or_op = unified_key
            if key not in res_unified:
                continue
            val = res_unified[key]
            if idx_or_op == 'mean':
                # 直接取mean（如mre_cost, mre_Pd等）
                unified_val = to_float(val)
            else:
                # 处理数组索引，如 vio_PQg[:, 0]
                if isinstance(val, np.ndarray):
                    unified_val = to_float(np.mean(val[:, idx_or_op]))
                elif torch.is_tensor(val):
                    unified_val = to_float(torch.mean(val[:, idx_or_op]))
                else:
                    unified_val = to_float(val[:, idx_or_op])
        else:
            if unified_key not in res_unified:
                continue
            unified_val = to_float(res_unified[unified_key])
        
        diff = abs(orig_val - unified_val)
        comparison[orig_key] = {
            "original": orig_val,
            "unified": unified_val,
            "diff": diff,
            "diff_pct": (diff / (abs(orig_val) + 1e-12)) * 100.0 if abs(orig_val) > 1e-12 else 0.0
        }
        
        if diff > max_diff:
            max_diff = diff
            max_diff_key = orig_key
        
        if verbose:
            print(f"{orig_key:20s}  original={orig_val:12.6f}   unified={unified_val:12.6f}   "
                  f"diff={diff:.6e}   ({comparison[orig_key]['diff_pct']:.4f}%)")
    
    # 对比后处理结果（如果有）
    if 'mae_Vm_post' in res_original or 'mae_Vmtest1' in res_unified:
        post_keys = {
            'mae_Vm_post': 'mae_Vmtest1',
            'mae_Va_post': 'mae_Vatest1',
            'cost_error_percent_post': ('mre_cost1', 'mean'),
            'Pg_satisfy_post': ('vio_PQg1', 0),
            'Qg_satisfy_post': ('vio_PQg1', 1),
            'branch_ang_satisfy_post': ('vio_branang1', 'mean'),
            'branch_pf_satisfy_post': ('vio_branpf1', 'mean'),
        }
        
        if verbose:
            print("\n" + "-" * 80)
            print("Post-Processing Results Comparison:")
            print("-" * 80)
        
        for orig_key, unified_key in post_keys.items():
            if orig_key not in res_original:
                continue
                
            orig_val = to_float(res_original[orig_key])
            
            if isinstance(unified_key, tuple):
                key, idx_or_op = unified_key
                if key not in res_unified:
                    continue
                val = res_unified[key]
                if idx_or_op == 'mean':
                    unified_val = to_float(val)
                else:
                    if isinstance(val, np.ndarray):
                        unified_val = to_float(np.mean(val[:, idx_or_op]))
                    elif torch.is_tensor(val):
                        unified_val = to_float(torch.mean(val[:, idx_or_op]))
                    else:
                        unified_val = to_float(val[:, idx_or_op])
            else:
                if unified_key not in res_unified:
                    continue
                unified_val = to_float(res_unified[unified_key])
            
            diff = abs(orig_val - unified_val)
            comparison[f"{orig_key}_post"] = {
                "original": orig_val,
                "unified": unified_val,
                "diff": diff,
                "diff_pct": (diff / (abs(orig_val) + 1e-12)) * 100.0 if abs(orig_val) > 1e-12 else 0.0
            }
            
            if verbose:
                print(f"{orig_key:20s}  original={orig_val:12.6f}   unified={unified_val:12.6f}   "
                      f"diff={diff:.6e}   ({comparison[f'{orig_key}_post']['diff_pct']:.4f}%)")
    
    # 对比时间信息
    if verbose:
        print("\n" + "-" * 80)
        print("Timing Comparison:")
        print("-" * 80)
        if 'inference_time_ms' in res_original:
            print(f"ORIGINAL inference time: {res_original['inference_time_ms']:.4f} ms/sample")
        if 'timing_info' in res_unified:
            timing = res_unified['timing_info']
            print(f"UNIFIED inference time: {timing.get('time_NN_per_sample_ms', 0):.4f} ms/sample")
            print(f"UNIFIED post-processing time: {timing.get('time_post_processing', 0):.4f} s")
    
    comparison["max_diff"] = max_diff
    comparison["max_diff_key"] = max_diff_key
    comparison["is_identical"] = max_diff < 1e-4  # 允许一定的数值误差
    
    if verbose:
        print("\n" + "#" * 80)
        if comparison["is_identical"]:
            print("RESULT: Results are CONSISTENT (within numerical precision)")
        else:
            print(f"RESULT: Results differ. Max difference: {max_diff:.6e} in '{max_diff_key}'")
        print("#" * 80 + "\n")
    
    return comparison


# ============================================================================
# NGT Flow Model Helper Functions
# ============================================================================
 
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
    # CRITICAL: Projection matrix P_tan is computed in NORMALIZED V-space, not physical V-space!
    # Flow: V_physical = sigmoid(z) * Vscale + Vbias
    # Normalized: V_normalized = sigmoid(z) = (V_physical - Vbias) / Vscale
    # Projection: P_tan operates on normalized V-space (F was scaled by scale_vec)
    # 
    # Correct transformation chain:
    # v_latent (z-space) → v_normalized (normalized V-space) → P_tan(v_normalized) → v_latent_projected
    for step in range(num_steps):
        t = torch.full((batch_size, 1), step * dt, device=device)
        v_latent = flow_model.predict_velocity(x, z, t, preference)
        
        # Transform from z-space to normalized V-space
        # dV_normalized/dz = sigmoid(z) * (1 - sigmoid(z))
        sig_z = torch.sigmoid(z)
        J_z_to_Vnorm = sig_z * (1 - sig_z)  # (batch, dim) - Jacobian from z to normalized V
        # Numerical stability: clamp to prevent explosion when sigmoid saturates
        J_inv_Vnorm_to_z = torch.clamp(1.0 / (J_z_to_Vnorm + eps), max=1e3)  # (batch, dim)
        
        # v_latent (z-space) → v_normalized (normalized V-space)
        v_normalized = v_latent * J_z_to_Vnorm  # (batch, dim)
        
        # Project in normalized V-space (where P_tan is defined)
        v_normalized_projected = torch.matmul(v_normalized, P_tan_t.T)  # (batch, dim)
        
        # Transform back from normalized V-space to z-space
        v_projected = v_normalized_projected * J_inv_Vnorm_to_z  # (batch, dim)
        
        z = z + v_projected * dt
    
    # Apply sigmoid scaling to get physical space output
    V_pred = torch.sigmoid(z) * flow_model.Vscale + flow_model.Vbias
    
    return V_pred


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
    config.ngt_lambda_cost = lambda_cost
    config.ngt_lambda_carbon = lambda_carbon
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
        
        # TensorBoard logging (every epoch, not just print epochs)
        if tb_logger:
            # Loss components
            tb_logger.log_scalar('loss/total', avg_loss, epoch)
            tb_logger.log_scalar('loss/cost', avg_cost, epoch)
            tb_logger.log_scalar('loss/carbon', avg_carbon, epoch)
            
            # Weighted objective (for comparing optimization progress)
            lambda_cost_val = getattr(config, 'ngt_lambda_cost', 0.9)
            lambda_carbon_val = getattr(config, 'ngt_lambda_carbon', 0.1)
            carbon_scale = getattr(config, 'ngt_carbon_scale', 30.0)
            weighted_obj = lambda_cost_val * avg_cost + lambda_carbon_val * avg_carbon * carbon_scale
            tb_logger.log_scalar('objective/weighted', weighted_obj, epoch)
            tb_logger.log_scalar('objective/cost', avg_cost, epoch)
            tb_logger.log_scalar('objective/carbon', avg_carbon, epoch)
            
            # Adaptive penalty weights (higher = better constraint satisfaction)
            # Formula: k = min(kcost * L_obj / (L_constraint + eps), k_max)
            # When constraint violation (L_constraint) is small, k is large (capped at k_max)
            tb_logger.log_scalar('weights/kgenp', avg_kgenp, epoch)
            tb_logger.log_scalar('weights/kgenq', avg_kgenq, epoch)
            tb_logger.log_scalar('weights/kpd', avg_kpd, epoch)
            tb_logger.log_scalar('weights/kqd', avg_kqd, epoch)
            tb_logger.log_scalar('weights/kv', avg_kv, epoch)
            
            # Constraint SATISFACTION - WEIGHT NORMALIZATION (indirect indicator)
            # WARNING: k/k_max is NOT the real constraint satisfaction rate!
            # It only indicates that weights are high (constraint violation loss is small on training data)
            tb_logger.log_scalar('satisfaction_weight_norm/Pg', avg_kgenp / 2000.0, epoch)  # kgenp_max=2000
            tb_logger.log_scalar('satisfaction_weight_norm/Qg', avg_kgenq / 2000.0, epoch)  # kgenq_max=2000
            tb_logger.log_scalar('satisfaction_weight_norm/Pd', avg_kpd / 100.0, epoch)     # kpd_max=100
            tb_logger.log_scalar('satisfaction_weight_norm/Qd', avg_kqd / 100.0, epoch)     # kqd_max=100
            tb_logger.log_scalar('satisfaction_weight_norm/V', avg_kv / 500.0, epoch)       # kv_max=500
            
            # REAL Constraint Satisfaction Rate (actual violation check)
            # Compute on a sample batch to get true satisfaction rates
            with torch.no_grad():
                sample_pred = model(ngt_data['x_train'][:batch_size].to(device))
                Va_pred_tb = sample_pred[:, :ngt_data['NPred_Va']]
                Vm_pred_tb = sample_pred[:, ngt_data['NPred_Va']:]
                
                # Compute real constraint satisfaction using get_vioPQg
                # Reconstruct full voltage and compute power flow
                # (numpy already imported at top of file)
                
                # Reconstruct full voltage from NGT format
                xam_P = np.insert(sample_pred.cpu().numpy(), ngt_data['idx_bus_Pnet_slack'][0], 0, axis=1)
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
                
                Pred_Vm_sample = np.sqrt(Ve**2 + Vf**2)
                Pred_Va_sample = np.arctan2(Vf, Ve)
                Pred_V_sample = Pred_Vm_sample * np.exp(1j * Pred_Va_sample)
                
                # Get sample load data
                sample_x_np = ngt_data['x_train'][:batch_size].cpu().numpy()
                num_Pd = len(ngt_data['bus_Pd'])
                Pd_sample = np.zeros((batch_size, config.Nbus))
                Qd_sample = np.zeros((batch_size, config.Nbus))
                Pd_sample[:, ngt_data['bus_Pd']] = sample_x_np[:, :num_Pd]
                Qd_sample[:, ngt_data['bus_Qd']] = sample_x_np[:, num_Pd:]
                
                # Compute power flow
                Pred_Pg_sample, Pred_Qg_sample, _, _ = get_genload(
                    Pred_V_sample, Pd_sample, Qd_sample,
                    sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
                )
                
                # Calculate real constraint satisfaction
                MAXMIN_Pg = ngt_data['MAXMIN_Pg']
                MAXMIN_Qg = ngt_data['MAXMIN_Qg']
                _, _, _, _, _, vio_PQg_sample, _, _, _, _ = get_vioPQg(
                    Pred_Pg_sample, sys_data.bus_Pg, MAXMIN_Pg,
                    Pred_Qg_sample, sys_data.bus_Qg, MAXMIN_Qg,
                    config.DELTA
                )
                if torch.is_tensor(vio_PQg_sample):
                    vio_PQg_sample = vio_PQg_sample.numpy()
                
                real_Pg_satisfy = np.mean(vio_PQg_sample[:, 0])
                real_Qg_satisfy = np.mean(vio_PQg_sample[:, 1])
                
                # Voltage satisfaction
                VmLb = config.ngt_VmLb
                VmUb = config.ngt_VmUb
                Vm_vio_upper = np.mean(Pred_Vm_sample > VmUb) * 100
                Vm_vio_lower = np.mean(Pred_Vm_sample < VmLb) * 100
                real_Vm_satisfy = 100 - Vm_vio_upper - Vm_vio_lower
                
                # Log REAL satisfaction rates
                tb_logger.log_scalar('satisfaction_real/Pg', real_Pg_satisfy, epoch)
                tb_logger.log_scalar('satisfaction_real/Qg', real_Qg_satisfy, epoch)
                tb_logger.log_scalar('satisfaction_real/Vm', real_Vm_satisfy, epoch)
            tb_logger.log_scalar('pred/Va_min', Va_pred_tb.min().item(), epoch)
            tb_logger.log_scalar('pred/Va_max', Va_pred_tb.max().item(), epoch)
            tb_logger.log_scalar('pred/Vm_min', Vm_pred_tb.min().item(), epoch)
            tb_logger.log_scalar('pred/Vm_max', Vm_pred_tb.max().item(), epoch)
        
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
                                 anchor_model_path=None, anchor_preference=None,
                                 tb_logger=None, zero_init=True, debug=False):
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
        tb_logger: TensorBoardLogger instance for logging (optional)
        zero_init: Whether to initialize flow model output near zero (default: True)
        debug: If True, skip training and load pre-trained model from fixed path (default: False)
        
    Returns:
        model: Trained flow model (PreferenceConditionedNetV)
        loss_history: Dictionary of training losses
        time_train: Training time in seconds
        ngt_data: NGT training data dictionary
        sys_data: Updated system data
    """ 
    
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
    
    # Initialize flow model output near zero (so initial velocity is small)
    if zero_init:
        initialize_flow_model_near_zero(model, scale=0.01)
        print("  Flow model output layer initialized near zero (scale=0.01)")
    
    # ============================================================
    # Debug Mode: Load pre-trained model and skip training
    # ============================================================
    if debug:
        debug_model_path = '/home/yuepeng/code/multi_objective_opf/main_part/saved_models/NetV_ngt_flow_300bus_lc05_E1000_final.pth'
        print(f"\n{'=' * 70}")
        print(f"[Debug Mode] Loading pre-trained model (skipping training)")
        print(f"{'=' * 70}")
        print(f"Model path: {debug_model_path}")
        
        if not os.path.exists(debug_model_path):
            raise FileNotFoundError(f"Debug model not found: {debug_model_path}")
        
        # Load model weights
        model.load_state_dict(torch.load(debug_model_path, map_location=device, weights_only=True))
        model.eval()
        print(f"  Model loaded successfully.")
        
        # Create empty loss history for compatibility
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
        
        # Set time_train to 0 (no training time)
        time_train = 0.0
        
        # Initialize P_tan_t to None (will be set if use_projection is True)
        P_tan_t = None
        if use_projection:
            # Setup projection matrix for debug mode
            try:
                from flow_model.post_processing import ConstraintProjectionV2
                print("\n--- Setting up Constraint Projection (Debug Mode) ---")
                projector = ConstraintProjectionV2(sys_data, config, use_historical_jacobian=True)
                P_tan_full, _, _ = projector.compute_projection_matrix()
                
                bus_Pnet_all = ngt_data['bus_Pnet_all']
                bus_slack = int(sys_data.bus_slack)
                Nbus = config.Nbus
                
                bus_Pnet_noslack = bus_Pnet_all[bus_Pnet_all != bus_slack]
                all_buses_noslack = np.concatenate([np.arange(bus_slack), np.arange(bus_slack+1, Nbus)])
                
                idx_Vm_in_Ptan = bus_Pnet_all
                idx_Va_in_Ptan = []
                for bus in bus_Pnet_noslack:
                    pos = np.where(all_buses_noslack == bus)[0][0]
                    idx_Va_in_Ptan.append(Nbus + pos)
                idx_Va_in_Ptan = np.array(idx_Va_in_Ptan)
                
                # Combine indices: [Vm indices, Va indices]
                idx_flow_in_Ptan = np.concatenate([idx_Va_in_Ptan, idx_Vm_in_Ptan])
                # Extract submatrix for Flow output dimensions (both rows and columns)
                P_tan_flow = P_tan_full[np.ix_(idx_flow_in_Ptan, idx_flow_in_Ptan)]
                P_tan_t = torch.from_numpy(P_tan_flow).float().to(device)
                print(f"  Full projection matrix shape: {P_tan_full.shape}")
                print(f"  Flow projection matrix shape: {P_tan_t.shape}")
                print(f"  Flow output dim: {len(idx_flow_in_Ptan)}")
            except Exception as e:
                print(f"  Warning: Could not setup projection matrix in debug mode: {e}")
                P_tan_t = None
        
        print(f"\n[Debug Mode] Model loaded, returning without training.")
        return model, loss_history, time_train, ngt_data, sys_data, use_projection, P_tan_t
    
    # ============================================================
    # Step 4: Create optimizer (matching MLP training - no weight_decay, no scheduler)
    # ============================================================
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ngt_Lr)
    
    # ============================================================
    # Step 5: Create loss function
    # ============================================================
    # CRITICAL FIX: Sync lambda values from function parameters to config
    # This ensures loss_fn uses the same lambda values as preference conditioning
    # DeepOPFNGTLoss reads lambda from config.ngt_lambda_cost/ngt_lambda_carbon
    config.ngt_lambda_cost = lambda_cost
    config.ngt_lambda_carbon = lambda_carbon
    # Also ensure multi-objective is enabled if lambda values are provided
    if not hasattr(config, 'ngt_use_multi_objective') or not config.ngt_use_multi_objective:
        config.ngt_use_multi_objective = True
    print(f"[Flow Training] Synced lambda values to config: λ_cost={lambda_cost}, λ_carbon={lambda_carbon}")
    
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
            P_tan_full, _, _ = projector.compute_projection_matrix()
            # P_tan_full is (599, 599) for 300-bus: [Vm(300), Va_noslack(299)]
            # Flow output is (465,): [Va_nonZIB_noslack(232), Vm_nonZIB(233)]
            # Need to extract and reorder to match Flow output format
            
            bus_Pnet_all = ngt_data['bus_Pnet_all']  # Non-ZIB buses
            bus_slack = int(sys_data.bus_slack)
            Nbus = config.Nbus
            
            # Build index mapping from Flow output to P_tan_full
            # P_tan_full columns: [Vm(0:300), Va_noslack(300:599)]
            # Flow output: [Va_nonZIB_noslack, Vm_nonZIB]
            
            # Find non-ZIB non-slack buses in original index
            bus_Pnet_noslack = bus_Pnet_all[bus_Pnet_all != bus_slack]
            
            # Indices in P_tan_full format [Vm, Va_noslack]:
            # - Vm part: bus indices directly (0:300)
            # - Va_noslack part: map bus index to Va_noslack position (300+)
            all_buses_noslack = np.concatenate([np.arange(bus_slack), np.arange(bus_slack+1, Nbus)])
            
            # Build mapping: for each bus in bus_Pnet_all/bus_Pnet_noslack, find its column in P_tan_full
            idx_Vm_in_Ptan = bus_Pnet_all  # Vm columns directly use bus index
            idx_Va_in_Ptan = []
            for bus in bus_Pnet_noslack:
                # Find position of bus in all_buses_noslack (Va column index after Vm)
                pos = np.where(all_buses_noslack == bus)[0][0]
                idx_Va_in_Ptan.append(Nbus + pos)
            idx_Va_in_Ptan = np.array(idx_Va_in_Ptan)
            
            # Flow output order: [Va_nonZIB_noslack, Vm_nonZIB]
            # Map to P_tan_full: [idx_Va_in_Ptan, idx_Vm_in_Ptan]
            flow_to_ptan_idx = np.concatenate([idx_Va_in_Ptan, idx_Vm_in_Ptan])
            
            # Extract submatrix for Flow output dimensions
            P_tan_flow = P_tan_full[np.ix_(flow_to_ptan_idx, flow_to_ptan_idx)]
            
            P_tan_t = torch.tensor(P_tan_flow, dtype=torch.float32, device=device)
            print(f"  Full projection matrix shape: {P_tan_full.shape}")
            print(f"  Flow projection matrix shape: {P_tan_t.shape}")
            print(f"  Flow output dim: {len(flow_to_ptan_idx)}")
        except ImportError as e:
            print(f"[Warning] ConstraintProjectionV2 not available ({e}), disabling projection")
            use_projection = False
        except Exception as e:
            print(f"[Warning] Projection setup failed ({e}), disabling projection")
            import traceback
            traceback.print_exc()
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
            # VAE outputs are SCALED/NORMALIZED values, NOT physical values!
            Vm_scaled = vae_vm(x_train, use_mean=True)  # [Ntrain, Nbus] - scaled
            Va_scaled_noslack = vae_va(x_train, use_mean=True)  # [Ntrain, Nbus-1] - scaled
        
        print(f"  VAE Vm output (scaled) shape: {Vm_scaled.shape}")
        print(f"  VAE Va output (scaled) shape: {Va_scaled_noslack.shape}")
        print(f"  VAE Vm scaled range: [{Vm_scaled.min():.4f}, {Vm_scaled.max():.4f}]")
        print(f"  VAE Va scaled range: [{Va_scaled_noslack.min():.4f}, {Va_scaled_noslack.max():.4f}]")
        
        # CRITICAL FIX: Denormalize VAE outputs to physical values!
        # Vm: Vm_physical = Vm_scaled / scale_vm * (VmUb - VmLb) + VmLb
        # Va: Va_physical = Va_scaled / scale_va
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
        
        # Denormalize Vm
        Vm_anchor_full = Vm_scaled / scale_vm * (VmUb - VmLb) + VmLb
        # Denormalize Va
        Va_anchor_full_noslack = Va_scaled_noslack / scale_va
        
        print(f"  After denormalization:")
        print(f"  Vm physical range: [{Vm_anchor_full.min():.4f}, {Vm_anchor_full.max():.4f}]")
        print(f"  Va physical range: [{Va_anchor_full_noslack.min():.4f}, {Va_anchor_full_noslack.max():.4f}]")
        
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
    # Step 10: Evaluate VAE baseline for TensorBoard reference
    # ============================================================
    vae_baseline_cost = 0.0
    vae_baseline_carbon = 0.0
    if tb_logger:
        print("\n--- Evaluating VAE baseline for TensorBoard reference ---")
        with torch.no_grad():
            # Evaluate VAE anchor on training data
            total_cost = 0.0
            total_carbon = 0.0
            n_samples = 0
            for step, (train_x, train_y, batch_indices) in enumerate(training_loader):
                train_x = train_x.to(device)
                z_anchor_batch = z_anchor_all[batch_indices]
                
                # Convert z_anchor back to physical space (same as flow output)
                V_anchor = torch.sigmoid(z_anchor_batch) * model.Vscale + model.Vbias
                
                _, loss_dict = loss_fn(V_anchor, train_x)
                total_cost += loss_dict.get('loss_cost', 0.0) * len(batch_indices)
                total_carbon += loss_dict.get('loss_carbon', 0.0) * len(batch_indices)
                n_samples += len(batch_indices)
            
            vae_baseline_cost = total_cost / max(n_samples, 1)
            vae_baseline_carbon = total_carbon / max(n_samples, 1)
            vae_baseline_weighted = lambda_cost * vae_baseline_cost + lambda_carbon * vae_baseline_carbon * config.ngt_carbon_scale
            
            print(f"  VAE Baseline - Cost: {vae_baseline_cost:.2f}, Carbon: {vae_baseline_carbon:.4f}")
            print(f"  VAE Baseline - Weighted Objective: {vae_baseline_weighted:.4f}")
    
    # ============================================================
    # Step 11: Training loop
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
        
        # NOTE: Removed scheduler.step() to match MLP training exactly
        
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
            # Use same forward function as training (projected vs non-projected)
            with torch.no_grad():
                sample_x = ngt_data['x_train'][:batch_size].to(device)
                sample_anchor = z_anchor_all[:batch_size]
                sample_pref = preference.expand(batch_size, -1)
                if use_projection and P_tan_t is not None:
                    sample_pred = flow_forward_ngt_projected(
                        model, sample_x, sample_anchor, P_tan_t, 
                        sample_pref, flow_inf_steps, training=False
                    )
                else:
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
        
        # TensorBoard logging (every epoch)
        if tb_logger:
            # Loss components
            tb_logger.log_scalar('loss/total', avg_loss, epoch)
            tb_logger.log_scalar('loss/cost', avg_cost, epoch)
            tb_logger.log_scalar('loss/carbon', avg_carbon, epoch)
            
            # Weighted objective (for comparing optimization progress)
            carbon_scale = getattr(config, 'ngt_carbon_scale', 30.0)
            weighted_obj = lambda_cost * avg_cost + lambda_carbon * avg_carbon * carbon_scale
            tb_logger.log_scalar('objective/weighted', weighted_obj, epoch)
            tb_logger.log_scalar('objective/cost', avg_cost, epoch)
            tb_logger.log_scalar('objective/carbon', avg_carbon, epoch)
            
            # VAE baseline reference (horizontal lines for comparison)
            tb_logger.log_scalar('baseline/vae_cost', vae_baseline_cost, epoch)
            tb_logger.log_scalar('baseline/vae_carbon', vae_baseline_carbon, epoch)
            tb_logger.log_scalar('baseline/vae_weighted', vae_baseline_weighted, epoch)
            
            # Adaptive penalty weights (higher = better constraint satisfaction)
            # Formula: k = min(kcost * L_obj / (L_constraint + eps), k_max)
            # When constraint violation (L_constraint) is small, k is large (capped at k_max)
            tb_logger.log_scalar('weights/kgenp', avg_kgenp, epoch)
            tb_logger.log_scalar('weights/kgenq', avg_kgenq, epoch)
            tb_logger.log_scalar('weights/kpd', avg_kpd, epoch)
            tb_logger.log_scalar('weights/kqd', avg_kqd, epoch)
            tb_logger.log_scalar('weights/kv', avg_kv, epoch)
            
            # Constraint SATISFACTION - WEIGHT NORMALIZATION (indirect indicator)
            # WARNING: k/k_max is NOT the real constraint satisfaction rate!
            # It only indicates that weights are high (constraint violation loss is small on training data)
            tb_logger.log_scalar('satisfaction_weight_norm/Pg', avg_kgenp / 2000.0, epoch)  # kgenp_max=2000
            tb_logger.log_scalar('satisfaction_weight_norm/Qg', avg_kgenq / 2000.0, epoch)  # kgenq_max=2000
            tb_logger.log_scalar('satisfaction_weight_norm/Pd', avg_kpd / 100.0, epoch)     # kpd_max=100
            tb_logger.log_scalar('satisfaction_weight_norm/Qd', avg_kqd / 100.0, epoch)     # kqd_max=100
            tb_logger.log_scalar('satisfaction_weight_norm/V', avg_kv / 500.0, epoch)       # kv_max=500
            
            # Prediction ranges and velocity diagnostics
            with torch.no_grad():
                sample_x = ngt_data['x_train'][:batch_size].to(device)
                sample_anchor = z_anchor_all[:batch_size]
                sample_pref = preference.expand(batch_size, -1)
                
                # Get flow prediction (use same forward function as training)
                if use_projection and P_tan_t is not None:
                    sample_pred_tb = flow_forward_ngt_projected(
                        model, sample_x, sample_anchor, P_tan_t, 
                        sample_pref, flow_inf_steps, training=False
                    )
                else:
                    sample_pred_tb = flow_forward_ngt(model, sample_x, sample_anchor, sample_pref, 
                                                      flow_inf_steps, training=False)
                Va_pred_tb = sample_pred_tb[:, :ngt_data['NPred_Va']]
                Vm_pred_tb = sample_pred_tb[:, ngt_data['NPred_Va']:]
                
                # Convert anchor back to physical space for comparison
                sample_anchor_physical = torch.sigmoid(sample_anchor) * model.Vscale + model.Vbias
                Va_anchor = sample_anchor_physical[:, :ngt_data['NPred_Va']]
                Vm_anchor = sample_anchor_physical[:, ngt_data['NPred_Va']:]
                
                # Velocity diagnostics: how much did Flow change from anchor?
                delta_Va = (sample_pred_tb[:, :ngt_data['NPred_Va']] - Va_anchor).abs()
                delta_Vm = (sample_pred_tb[:, ngt_data['NPred_Va']:] - Vm_anchor).abs()
                
                # Direct velocity at t=0 (what the model outputs at anchor)
                t_zero = torch.zeros(sample_x.shape[0], 1, device=device)
                velocity_at_anchor = model.predict_velocity(sample_x, sample_anchor, t_zero, sample_pref)
                velocity_norm = velocity_at_anchor.norm(dim=1).mean()
                
                # Compute REAL constraint satisfaction rate
                # (numpy already imported at top of file)
                
                # Reconstruct full voltage from Flow prediction
                xam_P = np.insert(sample_pred_tb.cpu().numpy(), ngt_data['idx_bus_Pnet_slack'][0], 0, axis=1)
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
                
                Pred_Vm_sample = np.sqrt(Ve**2 + Vf**2)
                Pred_Va_sample = np.arctan2(Vf, Ve)
                Pred_V_sample = Pred_Vm_sample * np.exp(1j * Pred_Va_sample)
                
                # Get sample load data
                sample_x_np = sample_x.cpu().numpy()
                num_Pd = len(ngt_data['bus_Pd'])
                Pd_sample = np.zeros((batch_size, config.Nbus))
                Qd_sample = np.zeros((batch_size, config.Nbus))
                Pd_sample[:, ngt_data['bus_Pd']] = sample_x_np[:, :num_Pd]
                Qd_sample[:, ngt_data['bus_Qd']] = sample_x_np[:, num_Pd:]
                
                # Compute power flow
                Pred_Pg_sample, Pred_Qg_sample, _, _ = get_genload(
                    Pred_V_sample, Pd_sample, Qd_sample,
                    sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
                )
                
                # Calculate real constraint satisfaction
                MAXMIN_Pg = ngt_data['MAXMIN_Pg']
                MAXMIN_Qg = ngt_data['MAXMIN_Qg']
                _, _, _, _, _, vio_PQg_sample, _, _, _, _ = get_vioPQg(
                    Pred_Pg_sample, sys_data.bus_Pg, MAXMIN_Pg,
                    Pred_Qg_sample, sys_data.bus_Qg, MAXMIN_Qg,
                    config.DELTA
                )
                if torch.is_tensor(vio_PQg_sample):
                    vio_PQg_sample = vio_PQg_sample.numpy()
                
                real_Pg_satisfy = np.mean(vio_PQg_sample[:, 0])
                real_Qg_satisfy = np.mean(vio_PQg_sample[:, 1])
                
                # Voltage satisfaction
                VmLb = config.ngt_VmLb
                VmUb = config.ngt_VmUb
                Vm_vio_upper = np.mean(Pred_Vm_sample > VmUb) * 100
                Vm_vio_lower = np.mean(Pred_Vm_sample < VmLb) * 100
                real_Vm_satisfy = 100 - Vm_vio_upper - Vm_vio_lower
                
                # Log REAL satisfaction rates
                tb_logger.log_scalar('satisfaction_real/Pg', real_Pg_satisfy, epoch)
                tb_logger.log_scalar('satisfaction_real/Qg', real_Qg_satisfy, epoch)
                tb_logger.log_scalar('satisfaction_real/Vm', real_Vm_satisfy, epoch)
            
            # Prediction ranges
            tb_logger.log_scalar('pred/Va_min', Va_pred_tb.min().item(), epoch)
            tb_logger.log_scalar('pred/Va_max', Va_pred_tb.max().item(), epoch)
            tb_logger.log_scalar('pred/Vm_min', Vm_pred_tb.min().item(), epoch)
            tb_logger.log_scalar('pred/Vm_max', Vm_pred_tb.max().item(), epoch)
            
            # Velocity/movement diagnostics
            tb_logger.log_scalar('velocity/norm_at_t0', velocity_norm.item(), epoch)
            tb_logger.log_scalar('velocity/delta_Va_mean', delta_Va.mean().item(), epoch)
            tb_logger.log_scalar('velocity/delta_Vm_mean', delta_Vm.mean().item(), epoch)
            tb_logger.log_scalar('velocity/delta_Va_max', delta_Va.max().item(), epoch)
            tb_logger.log_scalar('velocity/delta_Vm_max', delta_Vm.max().item(), epoch)
            
            # Anchor (VAE) baseline ranges for reference
            tb_logger.log_scalar('anchor/Va_min', Va_anchor.min().item(), epoch)
            tb_logger.log_scalar('anchor/Va_max', Va_anchor.max().item(), epoch)
            tb_logger.log_scalar('anchor/Vm_min', Vm_anchor.min().item(), epoch)
            tb_logger.log_scalar('anchor/Vm_max', Vm_anchor.max().item(), epoch)
        
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
    
    # Return projection info for evaluation
    return model, loss_history, time_train, ngt_data, sys_data, use_projection, P_tan_t


def evaluate_ngt_flow_model(config, model_flow, vae_vm, vae_va, x_test, 
                            Real_Vm, Real_Va_full, Pdtest, Qdtest, sys_data, 
                            BRANFT, MAXMIN_Pg, MAXMIN_Qg, gencost, Real_cost_total, 
                            ngt_data, preference, device, apply_post_processing=True,
                            use_projection=None, P_tan_t=None):
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
    
    # Check if projection should be used (match training setting)
    if use_projection is None:
        use_projection = getattr(config, 'ngt_use_projection', False)
    
    model_flow.eval()
    
    # Generate VAE anchor (convert to latent space for flow_backward)
    with torch.no_grad():
        Vm_vae = vae_vm(x_test, use_mean=True)  # [Ntest, Nbus] - scaled
        Va_vae_noslack = vae_va(x_test, use_mean=True)  # [Ntest, Nbus-1] - scaled
        
        # Denormalize VAE outputs to physical values
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
        
        # Denormalize
        Vm_vae_physical = Vm_vae / scale_vm * (VmUb - VmLb) + VmLb
        Va_vae_physical_noslack = Va_vae_noslack / scale_va
        
        # Reconstruct full Va
        Va_vae_physical = torch.zeros(Ntest, config.Nbus, device=device)
        Va_vae_physical[:, :bus_slack] = Va_vae_physical_noslack[:, :bus_slack]
        Va_vae_physical[:, bus_slack+1:] = Va_vae_physical_noslack[:, bus_slack:]
        
        # Extract non-ZIB values (physical space)
        Vm_nonZIB = Vm_vae_physical[:, bus_Pnet_all]
        Va_nonZIB_noslack = Va_vae_physical[:, bus_Pnet_noslack_all]
        V_anchor_physical = torch.cat([Va_nonZIB_noslack, Vm_nonZIB], dim=1)
        
        # Convert physical space to pre-sigmoid latent space (for flow_backward)
        eps = 1e-6
        Vscale = ngt_data['Vscale'].to(device)
        Vbias = ngt_data['Vbias'].to(device)
        u = (V_anchor_physical - Vbias) / (Vscale + 1e-12)
        u = torch.clamp(u, eps, 1 - eps)
        z_anchor = torch.log(u / (1 - u))  # logit
    
    # Flow integration (use projection if enabled and available)
    pref_batch = preference.expand(Ntest, -1)
    
    start_time = time.time()
    with torch.no_grad():
        if use_projection and P_tan_t is not None:
            # Use projection during flow integration (same as training)
            try:
                V_pred = flow_forward_ngt_projected(
                    model_flow, x_test, z_anchor, P_tan_t,
                    pref_batch, flow_inf_steps, training=False
                )
                print(f"[Evaluation] Using projection (P_tan_t shape: {P_tan_t.shape})")
            except Exception as e:
                print(f"[Warning] Projection failed during evaluation ({e}), falling back to standard flow")
                V_pred = flow_forward_ngt(
                    model_flow, x_test, z_anchor,
                    pref_batch, flow_inf_steps, training=False
                )
        else:
            # Standard flow integration (no projection)
            if use_projection:
                print(f"[Evaluation] Projection enabled but P_tan_t is None, using standard flow")
            V_pred = flow_forward_ngt(
                model_flow, x_test, z_anchor,
                pref_batch, flow_inf_steps, training=False
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
    
    # Detailed cost comparison for debugging
    print('\n' + '=' * 80)
    print('Detailed Cost Analysis')
    print('=' * 80)
    print(f'Real Cost Statistics:')
    print(f'  Mean:   {np.mean(Real_cost_total):.6f}')
    print(f'  Median: {np.median(Real_cost_total):.6f}')
    print(f'  Min:    {np.min(Real_cost_total):.6f}')
    print(f'  Max:    {np.max(Real_cost_total):.6f}')
    print(f'  Std:    {np.std(Real_cost_total):.6f}')
    print(f'\nPredicted Cost Statistics:')
    print(f'  Mean:   {np.mean(Pred_cost_total):.6f}')
    print(f'  Median: {np.median(Pred_cost_total):.6f}')
    print(f'  Min:    {np.min(Pred_cost_total):.6f}')
    print(f'  Max:    {np.max(Pred_cost_total):.6f}')
    print(f'  Std:    {np.std(Pred_cost_total):.6f}')
    print(f'\nCost Comparison:')
    print(f'  Mean(Pred) - Mean(Real): {np.mean(Pred_cost_total) - np.mean(Real_cost_total):.6f}')
    print(f'  Relative difference:     {(np.mean(Pred_cost_total) - np.mean(Real_cost_total)) / np.mean(Real_cost_total) * 100:.4f}%')
    print(f'  Samples where Pred < Real: {np.sum(Pred_cost_total < Real_cost_total)} / {len(Pred_cost_total)} ({np.sum(Pred_cost_total < Real_cost_total) / len(Pred_cost_total) * 100:.2f}%)')
    print(f'  Samples where Pred > Real: {np.sum(Pred_cost_total > Real_cost_total)} / {len(Pred_cost_total)} ({np.sum(Pred_cost_total > Real_cost_total) / len(Pred_cost_total) * 100:.2f}%)')
    print(f'  Samples where Pred = Real: {np.sum(np.abs(Pred_cost_total - Real_cost_total) < 1e-6)} / {len(Pred_cost_total)}')
    print(f'\nCost Error Metrics:')
    print(f'  Absolute error (mean):     {cost_error:.6f}')
    print(f'  Relative error (mean):     {cost_error_percent:.4f}%')
    relative_error_signed = np.mean((Pred_cost_total - Real_cost_total) / Real_cost_total * 100)
    print(f'  Signed relative error:     {relative_error_signed:.4f}%')
    print('=' * 80)
    
    # Constraint satisfaction (use same method as evaluate_ngt_model)
    print('\n[Constraint Satisfaction]')
    VmLb = config.ngt_VmLb
    VmUb = config.ngt_VmUb
    
    # Use get_vioPQg for consistent constraint checking (same as evaluate_ngt_model)
    lsPg, lsQg, lsidxPg, lsidxQg, vio_PQgmaxmin, vio_PQg, \
        deltaPgL, deltaQgU, deltaQgL, deltaQgU = get_vioPQg(
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
    Vm_vio_upper = np.mean(Pred_Vm > VmUb) * 100
    Vm_vio_lower = np.mean(Pred_Vm < VmLb) * 100
    Vm_satisfy = 100 - Vm_vio_upper - Vm_vio_lower
    
    print(f'  Vm constraint satisfaction: {Vm_satisfy:.2f}%')
    
    # Branch constraints (use get_viobran2 for consistency)
    vio_branang, vio_branpf, deltapf, vio_branpfidx, lsSf, lsSt, lsSf_sampidx, lsSt_sampidx = get_viobran2(
        Pred_V, Pred_Va, sys_data.branch, sys_data.Yf, sys_data.Yt,
        BRANFT, baseMVA, config.DELTA
    )
    
    if torch.is_tensor(vio_branang):
        vio_branang = vio_branang.numpy()
    if torch.is_tensor(vio_branpf):
        vio_branpf = vio_branpf.numpy()
    
    branch_ang_satisfy = np.mean(vio_branang)
    branch_pf_satisfy = np.mean(vio_branpf)
    
    print(f'  Branch angle constraint:    {branch_ang_satisfy:.2f}%')
    print(f'  Branch power constraint:    {branch_pf_satisfy:.2f}%')
    
    # Count violated samples
    lsidxPQg = np.where((lsidxPg + lsidxQg) > 0)[0]
    num_violated = len(lsidxPQg)
    
    print(f'\n  Violated samples: {num_violated}/{Ntest} ({num_violated/Ntest*100:.1f}%)')
    
    # Load deviation
    print('\n[Load Satisfaction]')
    Pd_total_real = np.sum(Pdtest, axis=1)
    Pd_total_pred = np.sum(Pred_Pd, axis=1)
    Qd_total_real = np.sum(Qdtest, axis=1)
    Qd_total_pred = np.sum(Pred_Qd, axis=1)
    
    Pd_error = np.mean(np.abs(Pd_total_pred - Pd_total_real) / np.abs(Pd_total_real)) * 100
    Qd_error = np.mean(np.abs(Qd_total_pred - Qd_total_real) / np.abs(Qd_total_real)) * 100
    
    print(f'  Pd deviation: {Pd_error:.4f}%')
    print(f'  Qd deviation: {Qd_error:.4f}%')
    
    # Timing
    print('\n[Inference Time]')
    print(f'  Total NN inference: {inference_time:.4f} s')
    print(f'  Per sample:         {inference_time/Ntest*1000:.4f} ms')
    
    # Evaluation Summary
    print('\n' + '=' * 60)
    print('Evaluation Summary (Flow Model)')
    print('=' * 60)
    
    results = {
        'mae_Vm': mae_Vm,
        'mae_Va': mae_Va,
        'cost_error_percent': cost_error_percent,
        'cost_mean': np.mean(Pred_cost_total),
        'cost_mean_real': np.mean(Real_cost_total),
        'Pg_satisfy': Pg_satisfy,
        'Qg_satisfy': Qg_satisfy,
        'Vm_satisfy': Vm_satisfy,
        'branch_ang_satisfy': branch_ang_satisfy,
        'branch_pf_satisfy': branch_pf_satisfy,
        'Pd_error_percent': Pd_error,
        'Qd_error_percent': Qd_error,
        'num_violated': num_violated,
        'inference_time_ms': inference_time / Ntest * 1000,
        'time_NN': inference_time,
    }
    
    # Print summary table (same format as evaluate_ngt_model)
    print(f"\n{'Metric':<30} {'Value':>15}")
    print('-' * 45)
    print(f"{'Vm MAE (p.u.)':<30} {mae_Vm:>15.6f}")
    print(f"{'Va MAE (rad)':<30} {mae_Va:>15.6f}")
    print(f"{'Cost Error (%)':<30} {cost_error_percent:>15.2f}")
    print(f"{'Pg Satisfaction (%)':<30} {Pg_satisfy:>15.2f}")
    print(f"{'Qg Satisfaction (%)':<30} {Qg_satisfy:>15.2f}")
    print(f"{'Vm Satisfaction (%)':<30} {Vm_satisfy:>15.2f}")
    print(f"{'Branch Angle Sat. (%)':<30} {branch_ang_satisfy:>15.2f}")
    print(f"{'Branch Power Sat. (%)':<30} {branch_pf_satisfy:>15.2f}")
    print(f"{'Pd Error (%)':<30} {Pd_error:>15.4f}")
    print(f"{'Qd Error (%)':<30} {Qd_error:>15.4f}")
    print('=' * 60)
    
    # Post-processing results (copy raw for simplicity)
    for key in ['mae_Vm', 'mae_Va', 'cost_error_percent', 'cost_mean', 
                'Pg_satisfy', 'Qg_satisfy', 'Vm_satisfy', 
                'branch_ang_satisfy', 'branch_pf_satisfy',
                'Pd_error_percent', 'Qd_error_percent', 'num_violated']:
        results[f'{key}_post'] = results[key]
    results['post_processing_time_ms'] = 0.0
    
    return results


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
            predictor = NGTPredictor(model_ngt)
            
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

