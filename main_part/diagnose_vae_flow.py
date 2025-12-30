#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VAE+Flow 误差诊断分析

分析 VAE+Flow 模型效果不好的原因：
1. VAE 重建误差：x -> z -> x' 的误差
2. 潜空间 Flow 误差：z_start -> Flow积分 -> z_pred vs z_true
3. 潜空间线性插值 baseline：z_start -> 线性插值 -> z_interp vs z_true
4. 误差分解：总误差 = VAE编码 + Flow/插值 + VAE解码

结论：
- 如果 VAE 重建误差大 -> VAE 训练不够
- 如果 Flow 误差 > 线性插值误差 -> Flow 没学好
- 如果 Flow 误差 ≈ 线性插值误差 -> 潜空间已经线性化，Flow 没有额外价值
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_part.config import get_config
from main_part.data_loader import load_multi_preference_dataset
from flow_model.linearized_vae import LinearizedVAE
from flow_model.latent_flow_matching import LatentFlowModel, LatentFlowWithVAE

# Set matplotlib to use non-Chinese fonts
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_models(config, multi_pref_data, device, vae_path, flow_path=None):
    """Load VAE and optionally Flow model."""
    input_dim = multi_pref_data['input_dim']
    output_dim = multi_pref_data['output_dim']
    NPred_Va = multi_pref_data.get('NPred_Va', output_dim // 2)
    
    # Create and load VAE
    vae = LinearizedVAE(
        input_dim=input_dim,
        output_dim=output_dim,
        latent_dim=getattr(config, 'linearized_vae_latent_dim', 32),
        hidden_dim=getattr(config, 'linearized_vae_hidden_dim', 256),
        num_layers=getattr(config, 'linearized_vae_num_layers', 3),
        pref_dim=1,
        NPred_Va=NPred_Va
    ).to(device)
    
    vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
    vae.eval()
    print(f"[OK] VAE loaded from: {vae_path}")
    
    # Load Flow model if path provided
    flow_model = None
    if flow_path and os.path.exists(flow_path):
        flow_model = LatentFlowModel(
            scene_dim=input_dim,
            latent_dim=vae.latent_dim,
            hidden_dim=getattr(config, 'latent_flow_hidden_dim', 256),
            num_layers=getattr(config, 'latent_flow_num_layers', 4)
        ).to(device)
        
        checkpoint = torch.load(flow_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            flow_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            flow_model.load_state_dict(checkpoint)
        flow_model.eval()
        print(f"[OK] Flow model loaded from: {flow_path}")
    
    return vae, flow_model


def analyze_vae_latent_linearity(vae, x_data, y_by_pref, lambda_values, device, n_samples=5):
    """
    分析 VAE 潜空间的线性化质量：使用 PCA 检查是否近似一维。
    
    返回：PC1 解释方差比例、PC1 与 lambda 的相关性
    """
    print("\n" + "=" * 70)
    print("0. VAE Latent Space Linearity Analysis")
    print("=" * 70)
    
    from sklearn.decomposition import PCA
    from scipy.stats import pearsonr
    
    vae.eval()
    n_samples = min(n_samples, x_data.shape[0])
    lambda_max = max(lambda_values)
    
    pc1_ratios = []
    pc12_ratios = []
    correlations = []
    
    with torch.no_grad():
        for sample_idx in range(n_samples):
            z_list = []
            for lc in lambda_values:
                if lc not in y_by_pref:
                    continue
                sol = y_by_pref[lc][sample_idx:sample_idx+1].to(device)
                scene = x_data[sample_idx:sample_idx+1].to(device)
                pref = torch.tensor([[lc / lambda_max]], device=device, dtype=torch.float32)
                z = vae.encode(scene, sol, pref, use_mean=True)
                z_list.append(z.cpu().numpy().flatten())
            
            if len(z_list) < 3:
                continue
                
            z_array = np.array(z_list)  # [K, latent_dim]
            
            # PCA
            pca = PCA(n_components=min(5, z_array.shape[1]))
            z_pca = pca.fit_transform(z_array)
            explained_var = pca.explained_variance_ratio_
            
            # Correlation between PC1 and lambda
            lambdas_norm = np.array([lc / lambda_max for lc in lambda_values if lc in y_by_pref])
            if len(lambdas_norm) == len(z_pca):
                corr, p_val = pearsonr(z_pca[:, 0], lambdas_norm)
                pc1_ratios.append(explained_var[0])
                pc12_ratios.append(explained_var[0] + explained_var[1])
                correlations.append(abs(corr))
    
    if len(pc1_ratios) == 0:
        print("  [Error] Could not compute linearity metrics")
        return None
    
    results = {
        'pc1_ratio_mean': np.mean(pc1_ratios),
        'pc1_ratio_std': np.std(pc1_ratios),
        'pc12_ratio_mean': np.mean(pc12_ratios),
        'pc12_ratio_std': np.std(pc12_ratios),
        'correlation_mean': np.mean(correlations),
        'correlation_std': np.std(correlations),
    }
    
    print(f"\n  PCA Analysis (averaged over {len(pc1_ratios)} samples):")
    print(f"    PC1 explains:     {results['pc1_ratio_mean']*100:.1f}% +/- {results['pc1_ratio_std']*100:.1f}%")
    print(f"    PC1+PC2 explains: {results['pc12_ratio_mean']*100:.1f}% +/- {results['pc12_ratio_std']*100:.1f}%")
    print(f"    |PC1 vs lambda correlation|: {results['correlation_mean']:.4f} +/- {results['correlation_std']:.4f}")
    
    # Assessment
    print(f"\n  Assessment:")
    if results['pc1_ratio_mean'] > 0.9:
        print(f"    [PC1] ✅ Excellent! Latent space is approximately 1D (PC1 > 90%)")
    elif results['pc1_ratio_mean'] > 0.7:
        print(f"    [PC1] ⚠️  Good. Latent space is mostly 1D (PC1 > 70%)")
    elif results['pc1_ratio_mean'] > 0.5:
        print(f"    [PC1] ⚠️  Fair. Latent space has some 1D structure (PC1 > 50%)")
    else:
        print(f"    [PC1] ❌ Poor. Latent space is not well linearized (PC1 < 50%)")
    
    if results['correlation_mean'] > 0.9:
        print(f"    [Corr] ✅ Excellent! PC1 is highly correlated with lambda (|r| > 0.9)")
    elif results['correlation_mean'] > 0.7:
        print(f"    [Corr] ⚠️  Good. PC1 is moderately correlated with lambda (|r| > 0.7)")
    else:
        print(f"    [Corr] ❌ Poor. PC1 is not well correlated with lambda (|r| < 0.7)")
    
    return results


def analyze_vae_reconstruction(vae, x_data, y_by_pref, lambda_values, device, n_samples=50):
    """
    分析 VAE 重建质量：x -> encode -> z -> decode -> x'
    
    返回：每个偏好下的重建 MSE 和 MAE
    """
    print("\n" + "=" * 70)
    print("1. VAE Reconstruction Analysis (x -> z -> x')")
    print("=" * 70)
    
    vae.eval()
    results = {'mse_by_pref': {}, 'mae_by_pref': {}, 'overall_mse': 0, 'overall_mae': 0}
    
    n_samples = min(n_samples, x_data.shape[0])
    total_mse = 0
    total_mae = 0
    n_total = 0
    
    with torch.no_grad():
        for lc in lambda_values:
            if lc not in y_by_pref:
                continue
            
            y_true = y_by_pref[lc][:n_samples].to(device)
            x = x_data[:n_samples].to(device)
            pref = torch.full((n_samples, 1), lc / max(lambda_values), device=device)
            
            # Encode and decode
            # vae.encode returns z directly (using mean by default)
            z_mean = vae.encode(x, y_true, pref, use_mean=True)
            y_recon = vae.decode(x, z_mean)
            
            # Compute errors
            mse = F.mse_loss(y_recon, y_true).item()
            mae = F.l1_loss(y_recon, y_true).item()
            
            results['mse_by_pref'][lc] = mse
            results['mae_by_pref'][lc] = mae
            total_mse += mse
            total_mae += mae
            n_total += 1
    
    results['overall_mse'] = total_mse / n_total if n_total > 0 else 0
    results['overall_mae'] = total_mae / n_total if n_total > 0 else 0
    
    print(f"\n  Overall Reconstruction Error:")
    print(f"    MSE: {results['overall_mse']:.6f}")
    print(f"    MAE: {results['overall_mae']:.6f}")
    
    # Show a few representative preferences
    sample_prefs = [lambda_values[0], lambda_values[len(lambda_values)//2], lambda_values[-1]]
    print(f"\n  Per-Preference Reconstruction Error:")
    for lc in sample_prefs:
        if lc in results['mse_by_pref']:
            print(f"    lambda={lc:.1f}: MSE={results['mse_by_pref'][lc]:.6f}, MAE={results['mae_by_pref'][lc]:.6f}")
    
    return results


def analyze_latent_space_flow(vae, flow_model, x_data, y_by_pref, lambda_values, device, n_samples=50):
    """
    分析潜空间中 Flow 模型的预测质量。
    
    比较：
    1. Flow 积分：z_start -> Flow -> z_pred
    2. 线性插值：z_start -> Linear Interp -> z_interp
    3. Ground Truth：z_true (VAE encode)
    
    这可以隔离 VAE 编解码误差，单独评估 Flow 模型。
    """
    print("\n" + "=" * 70)
    print("2. Latent Space Flow Analysis (isolating Flow model error)")
    print("=" * 70)
    
    if flow_model is None:
        print("  [Skip] Flow model not loaded")
        return None
    
    vae.eval()
    flow_model.eval()
    
    n_samples = min(n_samples, x_data.shape[0])
    lambda_min = min(lambda_values)
    lambda_max = max(lambda_values)
    
    # Normalize lambda values to [0, 1]
    def normalize_lambda(lc):
        return (lc - lambda_min) / (lambda_max - lambda_min) if lambda_max > lambda_min else 0.0
    
    results = {
        'flow_mse_by_pref': {},
        'flow_mae_by_pref': {},
        'interp_mse_by_pref': {},
        'interp_mae_by_pref': {},
        'z_mse_start_end': 0,  # Distance between z_start and z_end
    }
    
    with torch.no_grad():
        # Get starting point (lambda=0)
        lc_start = lambda_values[0]
        lc_end = lambda_values[-1]
        
        x = x_data[:n_samples].to(device)
        y_start = y_by_pref[lc_start][:n_samples].to(device)
        y_end = y_by_pref[lc_end][:n_samples].to(device)
        
        pref_start = torch.full((n_samples, 1), normalize_lambda(lc_start), device=device)
        pref_end = torch.full((n_samples, 1), normalize_lambda(lc_end), device=device)
        
        # Encode start and end points
        z_start = vae.encode(x, y_start, pref_start, use_mean=True)
        z_end = vae.encode(x, y_end, pref_end, use_mean=True)
        
        # Measure z distance (how far apart are the endpoints in latent space)
        z_dist = torch.norm(z_end - z_start, dim=1).mean().item()
        results['z_mse_start_end'] = F.mse_loss(z_end, z_start).item()
        print(f"\n  Latent space distance (z_start to z_end):")
        print(f"    Mean L2 norm: {z_dist:.4f}")
        print(f"    MSE: {results['z_mse_start_end']:.6f}")
        
        # For each target preference, compare Flow vs Linear Interpolation
        flow_errors = []
        interp_errors = []
        
        for lc_target in lambda_values:
            if lc_target not in y_by_pref:
                continue
            
            y_target = y_by_pref[lc_target][:n_samples].to(device)
            pref_target = torch.full((n_samples, 1), normalize_lambda(lc_target), device=device)
            
            # Ground truth: encode target
            z_true = vae.encode(x, y_target, pref_target, use_mean=True)
            
            # Method 1: Flow integration from z_start
            r_start = torch.full((n_samples, 1), normalize_lambda(lc_start), device=device)
            r_target = torch.full((n_samples, 1), normalize_lambda(lc_target), device=device)
            
            # flow_model.integrate expects (scene, z_start, r_start, r_end, num_steps, method)
            # r_start and r_end are floats, not tensors
            r_start_val = normalize_lambda(lc_start)
            r_target_val = normalize_lambda(lc_target)
            z_flow = flow_model.integrate(
                x, z_start, r_start_val, r_target_val,
                num_steps=getattr(config, 'latent_flow_inf_steps', 20),
                method=getattr(config, 'latent_flow_inf_method', 'heun')
            )
            
            # Method 2: Linear interpolation in latent space
            t = normalize_lambda(lc_target)  # t in [0, 1]
            z_interp = (1 - t) * z_start + t * z_end
            
            # Compute errors in latent space
            flow_mse = F.mse_loss(z_flow, z_true).item()
            flow_mae = F.l1_loss(z_flow, z_true).item()
            interp_mse = F.mse_loss(z_interp, z_true).item()
            interp_mae = F.l1_loss(z_interp, z_true).item()
            
            results['flow_mse_by_pref'][lc_target] = flow_mse
            results['flow_mae_by_pref'][lc_target] = flow_mae
            results['interp_mse_by_pref'][lc_target] = interp_mse
            results['interp_mae_by_pref'][lc_target] = interp_mae
            
            flow_errors.append(flow_mse)
            interp_errors.append(interp_mse)
        
        # Aggregate
        results['flow_mse_avg'] = np.mean(flow_errors)
        results['interp_mse_avg'] = np.mean(interp_errors)
        results['flow_mae_avg'] = np.mean([results['flow_mae_by_pref'][lc] for lc in results['flow_mae_by_pref']])
        results['interp_mae_avg'] = np.mean([results['interp_mae_by_pref'][lc] for lc in results['interp_mae_by_pref']])
    
    print(f"\n  Latent Space Prediction Error (Flow vs Linear Interpolation):")
    print(f"    Flow Integration:    MSE={results['flow_mse_avg']:.6f}, MAE={results['flow_mae_avg']:.6f}")
    print(f"    Linear Interpolation: MSE={results['interp_mse_avg']:.6f}, MAE={results['interp_mae_avg']:.6f}")
    
    if results['flow_mse_avg'] < results['interp_mse_avg']:
        improvement = (1 - results['flow_mse_avg'] / results['interp_mse_avg']) * 100
        print(f"    -> Flow is {improvement:.1f}% BETTER than linear interpolation in latent space")
    else:
        degradation = (results['flow_mse_avg'] / results['interp_mse_avg'] - 1) * 100
        print(f"    -> Flow is {degradation:.1f}% WORSE than linear interpolation in latent space")
        print(f"    -> DIAGNOSIS: Flow model has not learned useful dynamics!")
    
    return results


def analyze_end_to_end_error(vae, flow_model, x_data, y_by_pref, lambda_values, device, n_samples=50):
    """
    端到端误差分析：从原始空间出发，经过整个 VAE+Flow 流程。
    
    比较：
    1. VAE+Flow：x -> encode(y_start) -> z_start -> Flow -> z_pred -> decode -> y_pred
    2. VAE+Interp：x -> encode(y_start, y_end) -> z_start, z_end -> interp -> decode -> y_pred
    3. 原始空间线性插值：y_start -> interp -> y_pred
    """
    print("\n" + "=" * 70)
    print("3. End-to-End Error Analysis (full pipeline)")
    print("=" * 70)
    
    vae.eval()
    
    n_samples = min(n_samples, x_data.shape[0])
    lambda_min = min(lambda_values)
    lambda_max = max(lambda_values)
    
    def normalize_lambda(lc):
        return (lc - lambda_min) / (lambda_max - lambda_min) if lambda_max > lambda_min else 0.0
    
    results = {
        'vae_flow': {'mse': [], 'mae': []},
        'vae_interp': {'mse': [], 'mae': []},
        'original_interp': {'mse': [], 'mae': []},
    }
    
    with torch.no_grad():
        x = x_data[:n_samples].to(device)
        
        # Get endpoints
        lc_start = lambda_values[0]
        lc_end = lambda_values[-1]
        y_start = y_by_pref[lc_start][:n_samples].to(device)
        y_end = y_by_pref[lc_end][:n_samples].to(device)
        
        pref_start = torch.full((n_samples, 1), normalize_lambda(lc_start), device=device)
        pref_end = torch.full((n_samples, 1), normalize_lambda(lc_end), device=device)
        
        # Encode endpoints
        z_start = vae.encode(x, y_start, pref_start, use_mean=True)
        z_end = vae.encode(x, y_end, pref_end, use_mean=True)
        
        for lc_target in lambda_values:
            if lc_target not in y_by_pref:
                continue
            
            y_true = y_by_pref[lc_target][:n_samples].to(device)
            t = normalize_lambda(lc_target)
            
            # Method 1: VAE+Flow (if available)
            if flow_model is not None:
                r_start_val = normalize_lambda(lc_start)
                r_target_val = t
                
                z_flow = flow_model.integrate(
                    x, z_start, r_start_val, r_target_val,
                    num_steps=getattr(config, 'latent_flow_inf_steps', 20),
                    method=getattr(config, 'latent_flow_inf_method', 'heun')
                )
                y_flow = vae.decode(x, z_flow)
                
                results['vae_flow']['mse'].append(F.mse_loss(y_flow, y_true).item())
                results['vae_flow']['mae'].append(F.l1_loss(y_flow, y_true).item())
            
            # Method 2: VAE+Interpolation (latent space linear interp + decode)
            z_interp = (1 - t) * z_start + t * z_end
            y_vae_interp = vae.decode(x, z_interp)
            
            results['vae_interp']['mse'].append(F.mse_loss(y_vae_interp, y_true).item())
            results['vae_interp']['mae'].append(F.l1_loss(y_vae_interp, y_true).item())
            
            # Method 3: Original space linear interpolation
            y_orig_interp = (1 - t) * y_start + t * y_end
            
            results['original_interp']['mse'].append(F.mse_loss(y_orig_interp, y_true).item())
            results['original_interp']['mae'].append(F.l1_loss(y_orig_interp, y_true).item())
    
    # Aggregate results
    for method in results:
        if results[method]['mse']:
            results[method]['mse_avg'] = np.mean(results[method]['mse'])
            results[method]['mae_avg'] = np.mean(results[method]['mae'])
        else:
            results[method]['mse_avg'] = float('inf')
            results[method]['mae_avg'] = float('inf')
    
    print(f"\n  End-to-End Prediction Error:")
    print(f"    Original Space Linear Interp: MSE={results['original_interp']['mse_avg']:.6f}, MAE={results['original_interp']['mae_avg']:.6f}")
    print(f"    VAE + Latent Interp:          MSE={results['vae_interp']['mse_avg']:.6f}, MAE={results['vae_interp']['mae_avg']:.6f}")
    if flow_model is not None:
        print(f"    VAE + Flow:                   MSE={results['vae_flow']['mse_avg']:.6f}, MAE={results['vae_flow']['mae_avg']:.6f}")
    
    # Diagnosis
    print(f"\n  [DIAGNOSIS]")
    
    # Compare VAE+Interp vs Original Interp
    if results['vae_interp']['mse_avg'] > results['original_interp']['mse_avg']:
        ratio = results['vae_interp']['mse_avg'] / results['original_interp']['mse_avg']
        print(f"    - VAE latent interp is {ratio:.1f}x WORSE than original space interp")
        print(f"      -> VAE adds error (encoding/decoding loss)")
    else:
        improvement = (1 - results['vae_interp']['mse_avg'] / results['original_interp']['mse_avg']) * 100
        print(f"    - VAE latent interp is {improvement:.1f}% BETTER than original space interp")
        print(f"      -> VAE has learned useful latent structure")
    
    # Compare VAE+Flow vs VAE+Interp
    if flow_model is not None:
        if results['vae_flow']['mse_avg'] > results['vae_interp']['mse_avg']:
            ratio = results['vae_flow']['mse_avg'] / results['vae_interp']['mse_avg']
            print(f"    - VAE+Flow is {ratio:.1f}x WORSE than VAE+Latent Interp")
            print(f"      -> Flow model is adding error, not helping!")
            print(f"      -> CONCLUSION: Flow model needs more training or architecture changes")
        else:
            improvement = (1 - results['vae_flow']['mse_avg'] / results['vae_interp']['mse_avg']) * 100
            print(f"    - VAE+Flow is {improvement:.1f}% BETTER than VAE+Latent Interp")
            print(f"      -> Flow model has learned useful dynamics")
    
    return results


def analyze_error_breakdown(vae, flow_model, x_data, y_by_pref, lambda_values, device, n_samples=50):
    """
    误差分解：将总误差分解为各个组件的贡献。
    
    Total Error = Encoding Error + Latent Prediction Error + Decoding Error
    """
    print("\n" + "=" * 70)
    print("4. Error Breakdown Analysis")
    print("=" * 70)
    
    vae.eval()
    
    n_samples = min(n_samples, x_data.shape[0])
    lambda_min = min(lambda_values)
    lambda_max = max(lambda_values)
    
    def normalize_lambda(lc):
        return (lc - lambda_min) / (lambda_max - lambda_min) if lambda_max > lambda_min else 0.0
    
    # Pick a representative target preference (middle)
    lc_target = lambda_values[len(lambda_values) // 2]
    lc_start = lambda_values[0]
    
    with torch.no_grad():
        x = x_data[:n_samples].to(device)
        y_start = y_by_pref[lc_start][:n_samples].to(device)
        y_target = y_by_pref[lc_target][:n_samples].to(device)
        
        pref_start = torch.full((n_samples, 1), normalize_lambda(lc_start), device=device)
        pref_target = torch.full((n_samples, 1), normalize_lambda(lc_target), device=device)
        
        # Step 1: Encoding
        z_start = vae.encode(x, y_start, pref_start, use_mean=True)
        z_target_true = vae.encode(x, y_target, pref_target, use_mean=True)
        
        # Step 2: Latent prediction (Flow or Interp)
        t = normalize_lambda(lc_target)
        z_end = vae.encode(x, y_by_pref[lambda_values[-1]][:n_samples].to(device), 
                           torch.full((n_samples, 1), 1.0, device=device), use_mean=True)
        
        z_interp = (1 - t) * z_start + t * z_end  # Linear interp
        
        if flow_model is not None:
            r_start_val = normalize_lambda(lc_start)
            r_target_val = t
            z_flow = flow_model.integrate(x, z_start, r_start_val, r_target_val, num_steps=20, method='heun')
        else:
            z_flow = z_interp
        
        # Step 3: Decoding
        y_recon_from_true_z = vae.decode(x, z_target_true)  # Decode from true z
        y_recon_from_flow_z = vae.decode(x, z_flow)  # Decode from flow z
        y_recon_from_interp_z = vae.decode(x, z_interp)  # Decode from interp z
        
        # Compute errors
        # Pure decoding error: y_target vs decode(encode(y_target))
        decode_error = F.mse_loss(y_recon_from_true_z, y_target).item()
        
        # Latent prediction error (Flow)
        latent_flow_error = F.mse_loss(z_flow, z_target_true).item()
        
        # Latent prediction error (Interp)
        latent_interp_error = F.mse_loss(z_interp, z_target_true).item()
        
        # Total end-to-end error
        total_flow_error = F.mse_loss(y_recon_from_flow_z, y_target).item()
        total_interp_error = F.mse_loss(y_recon_from_interp_z, y_target).item()
    
    print(f"\n  Target preference: lambda={lc_target}")
    print(f"\n  Error Components:")
    print(f"    [A] VAE Encode-Decode Error (x->z->x'):       {decode_error:.6f}")
    print(f"    [B] Latent Flow Prediction Error (z_flow):    {latent_flow_error:.6f}")
    print(f"    [C] Latent Interp Prediction Error (z_interp): {latent_interp_error:.6f}")
    print(f"\n  End-to-End Errors:")
    print(f"    [D] VAE+Flow Total Error:   {total_flow_error:.6f}")
    print(f"    [E] VAE+Interp Total Error: {total_interp_error:.6f}")
    
    print(f"\n  [INTERPRETATION]")
    print(f"    - If A is large: VAE reconstruction is the bottleneck")
    print(f"    - If B > C: Flow is worse than linear interpolation in latent space")
    print(f"    - If D > E: Flow adds error instead of helping")
    
    # Diagnosis
    if decode_error > 0.01:
        print(f"\n  [!] VAE reconstruction error ({decode_error:.4f}) is significant")
        print(f"      -> Consider training VAE longer or increasing latent_dim")
    
    if latent_flow_error > latent_interp_error:
        ratio = latent_flow_error / latent_interp_error
        print(f"\n  [!] Flow latent error is {ratio:.1f}x worse than linear interp")
        print(f"      -> Flow model has NOT learned useful dynamics")
        print(f"      -> Consider: more training, different architecture, or just use linear interp")
    
    return {
        'decode_error': decode_error,
        'latent_flow_error': latent_flow_error,
        'latent_interp_error': latent_interp_error,
        'total_flow_error': total_flow_error,
        'total_interp_error': total_interp_error,
    }


def plot_diagnosis(results, save_path='main_part/results/vae_flow_diagnosis.png'):
    """Plot diagnostic results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: VAE Reconstruction Error by Preference
    if 'vae_recon' in results and results['vae_recon']:
        ax = axes[0, 0]
        prefs = sorted(results['vae_recon']['mse_by_pref'].keys())
        mse_vals = [results['vae_recon']['mse_by_pref'][p] for p in prefs]
        ax.plot(prefs, mse_vals, 'b-o', linewidth=2, markersize=4)
        ax.set_xlabel('Lambda Carbon', fontsize=11)
        ax.set_ylabel('Reconstruction MSE', fontsize=11)
        ax.set_title('VAE Reconstruction Error by Preference', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=results['vae_recon']['overall_mse'], color='r', linestyle='--', 
                   label=f"Avg: {results['vae_recon']['overall_mse']:.4f}")
        ax.legend()
    
    # Plot 2: Latent Space Flow vs Interp
    if 'latent_flow' in results and results['latent_flow']:
        ax = axes[0, 1]
        prefs = sorted(results['latent_flow']['flow_mse_by_pref'].keys())
        flow_mse = [results['latent_flow']['flow_mse_by_pref'][p] for p in prefs]
        interp_mse = [results['latent_flow']['interp_mse_by_pref'][p] for p in prefs]
        ax.plot(prefs, flow_mse, 'b-o', linewidth=2, markersize=4, label='Flow')
        ax.plot(prefs, interp_mse, 'r--s', linewidth=2, markersize=4, label='Linear Interp')
        ax.set_xlabel('Lambda Carbon', fontsize=11)
        ax.set_ylabel('Latent Space MSE', fontsize=11)
        ax.set_title('Latent Space: Flow vs Linear Interpolation', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 3: End-to-End Comparison
    if 'end_to_end' in results and results['end_to_end']:
        ax = axes[1, 0]
        methods = ['Original\nInterp', 'VAE +\nLatent Interp']
        mse_vals = [
            results['end_to_end']['original_interp']['mse_avg'],
            results['end_to_end']['vae_interp']['mse_avg'],
        ]
        colors = ['green', 'orange']
        
        # Add VAE+Flow if available
        if results['end_to_end']['vae_flow']['mse']:
            methods.append('VAE +\nFlow')
            mse_vals.append(results['end_to_end']['vae_flow']['mse_avg'])
            colors.append('blue')
        
        bars = ax.bar(methods, mse_vals, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('End-to-End MSE', fontsize=11)
        ax.set_title('End-to-End Prediction Error Comparison', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, mse_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Error Breakdown
    if 'breakdown' in results and results['breakdown']:
        ax = axes[1, 1]
        components = ['VAE\nReconstruction', 'Latent\nFlow Error', 'Latent\nInterp Error']
        values = [
            results['breakdown']['decode_error'],
            results['breakdown']['latent_flow_error'],
            results['breakdown']['latent_interp_error']
        ]
        colors = ['purple', 'blue', 'red']
        bars = ax.bar(components, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('MSE', fontsize=11)
        ax.set_title('Error Component Breakdown', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005, 
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Diagnosis plot saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='VAE+Flow Error Diagnosis')
    parser.add_argument('--vae-ckpt', type=str, default='main_part/saved_models/linearized_vae_epoch2900.pth',
                        help='Path to VAE checkpoint')
    parser.add_argument('--flow-ckpt', type=str, default='main_part/saved_models/latent_flow_epoch1600.pth',
                        help='Path to Flow model checkpoint')
    parser.add_argument('--n-samples', type=int, default=50, help='Number of samples to analyze')
    args = parser.parse_args()
    
    print("=" * 70)
    print("VAE+Flow Error Diagnosis")
    print("=" * 70)
    
    # Load config and data
    global config
    config = get_config()
    device = config.device
    
    print(f"\nLoading data...")
    multi_pref_data, sys_data = load_multi_preference_dataset(config)
    
    # Load models
    print(f"\nLoading models...")
    vae, flow_model = load_models(config, multi_pref_data, device, args.vae_ckpt, args.flow_ckpt)
    
    # Prepare data
    x_data = multi_pref_data['x_val'] if 'x_val' in multi_pref_data else multi_pref_data['x_train']
    y_by_pref = multi_pref_data['y_val_by_pref'] if 'y_val_by_pref' in multi_pref_data else multi_pref_data['y_train_by_pref']
    lambda_values = sorted(multi_pref_data['lambda_carbon_values'])
    
    print(f"\nData: {x_data.shape[0]} samples, {len(lambda_values)} preferences")
    
    # Run analyses
    results = {}
    
    # 0. VAE Latent Linearity
    results['vae_linearity'] = analyze_vae_latent_linearity(
        vae, x_data, y_by_pref, lambda_values, device, n_samples=min(10, args.n_samples)
    )
    
    # 1. VAE Reconstruction
    results['vae_recon'] = analyze_vae_reconstruction(
        vae, x_data, y_by_pref, lambda_values, device, args.n_samples
    )
    
    # 2. Latent Space Flow Analysis (skip if no Flow model)
    if flow_model is not None:
        results['latent_flow'] = analyze_latent_space_flow(
            vae, flow_model, x_data, y_by_pref, lambda_values, device, args.n_samples
        )
    else:
        results['latent_flow'] = None
        print("\n[Skip] Latent Space Flow Analysis (Flow model not loaded)")
    
    # 3. End-to-End Error Analysis
    results['end_to_end'] = analyze_end_to_end_error(
        vae, flow_model, x_data, y_by_pref, lambda_values, device, args.n_samples
    )
    
    # 4. Error Breakdown (skip if no Flow model)
    if flow_model is not None:
        results['breakdown'] = analyze_error_breakdown(
            vae, flow_model, x_data, y_by_pref, lambda_values, device, args.n_samples
        )
    else:
        results['breakdown'] = None
        print("\n[Skip] Error Breakdown Analysis (Flow model not loaded)")
    
    # Plot results
    plot_diagnosis(results)
    
    # Final Summary
    print("\n" + "=" * 70)
    print("FINAL DIAGNOSIS SUMMARY")
    print("=" * 70)
    
    if results['vae_recon']['overall_mse'] > 0.01:
        print("\n[!] VAE RECONSTRUCTION IS A BOTTLENECK")
        print(f"    MSE: {results['vae_recon']['overall_mse']:.6f}")
        print("    Recommendation: Train VAE longer or increase latent_dim")
    else:
        print(f"\n[OK] VAE reconstruction is reasonable (MSE: {results['vae_recon']['overall_mse']:.6f})")
    
    if results['latent_flow'] and results['latent_flow']['flow_mse_avg'] > results['latent_flow']['interp_mse_avg']:
        ratio = results['latent_flow']['flow_mse_avg'] / results['latent_flow']['interp_mse_avg']
        print(f"\n[!] FLOW MODEL IS WORSE THAN LINEAR INTERPOLATION ({ratio:.1f}x)")
        print("    The Flow model has NOT learned useful dynamics in latent space")
        print("    Recommendations:")
        print("    1. Train Flow model longer (more epochs)")
        print("    2. Try different architecture (more layers, larger hidden dim)")
        print("    3. Check if latent space is too non-linear for Flow to learn")
        print("    4. Consider just using linear interpolation in latent space")
    elif results['latent_flow']:
        print(f"\n[OK] Flow model performs similar to or better than linear interpolation")
    
    if results['end_to_end']['vae_flow']['mse_avg'] > results['end_to_end']['original_interp']['mse_avg']:
        print(f"\n[!] VAE+FLOW IS WORSE THAN ORIGINAL SPACE INTERPOLATION")
        print("    The VAE+Flow pipeline adds more error than it removes")
        print("    Consider using simpler approaches (e.g., direct MLP or original space interp)")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
