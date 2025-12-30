#!/usr/bin/env python
# coding: utf-8
"""
Validation and Visualization Script for VAE+Flow Model

This script provides tools to:
1. Validate the trained Linearized VAE (check latent space linearity)
2. Evaluate the complete VAE+Flow model on multi-preference data
3. Compare VAE+Flow with other models (e.g., original space Flow Matching)
4. Visualize latent trajectories and prediction errors

Usage:
    python validate_vae_flow.py [--mode validate|compare|visualize]

Author: Auto-generated from VAE+Flow plan
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from data_loader import load_multi_preference_dataset

# Import VAE+Flow components
from flow_model.linearized_vae import LinearizedVAE, visualize_latent_linearity
from flow_model.latent_flow_matching import LatentFlowModel, LatentFlowWithVAE


def load_vae_flow_model(config, multi_pref_data, device, vae_ckpt=None, flow_ckpt=None):
    """
    Load trained VAE+Flow model from checkpoint.
    
    Supports multiple loading modes:
    1. Combined checkpoint (model_multi_pref_vae_flow_combined.pth)
    2. Individual checkpoints (linearized_vae_*.pth + latent_flow_*.pth)
    3. Custom paths via vae_ckpt and flow_ckpt arguments
    4. Config paths via config.pretrained_vae_path and config.pretrained_flow_path
    
    Args:
        config: Configuration object
        multi_pref_data: Multi-preference data dictionary
        device: Device (CPU/GPU)
        vae_ckpt: Optional custom path to VAE checkpoint
        flow_ckpt: Optional custom path to Flow checkpoint
    
    Returns:
        vae: Loaded LinearizedVAE
        flow_model: Loaded LatentFlowModel
        combined_model: LatentFlowWithVAE for inference
    """
    input_dim = multi_pref_data['input_dim']
    output_dim = multi_pref_data['output_dim']
    NPred_Va = multi_pref_data.get('NPred_Va', output_dim // 2)
    
    # Determine model paths (priority: arguments > config > default)
    combined_path = f'{config.model_save_dir}/model_multi_pref_vae_flow_combined.pth'
    
    # VAE path priority: argument > config.pretrained_vae_path > default
    if vae_ckpt:
        vae_path = vae_ckpt
    elif hasattr(config, 'pretrained_vae_path') and config.pretrained_vae_path:
        vae_path = config.pretrained_vae_path
    else:
        vae_path = f'{config.model_save_dir}/linearized_vae_final.pth'
    
    # Flow path priority: argument > config.pretrained_flow_path > default
    if flow_ckpt:
        flow_path = flow_ckpt
    elif hasattr(config, 'pretrained_flow_path') and config.pretrained_flow_path:
        flow_path = config.pretrained_flow_path
    else:
        flow_path = f'{config.model_save_dir}/latent_flow_final.pth'
    
    # Create VAE
    latent_dim = getattr(config, 'linearized_vae_latent_dim', 32)
    vae = LinearizedVAE(
        input_dim=input_dim,
        output_dim=output_dim,
        latent_dim=latent_dim,
        hidden_dim=getattr(config, 'linearized_vae_hidden_dim', 256),
        num_layers=getattr(config, 'linearized_vae_num_layers', 3),
        pref_dim=1,
        NPred_Va=NPred_Va
    ).to(device)
    
    # Create Flow model
    flow_model = LatentFlowModel(
        scene_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=getattr(config, 'latent_flow_hidden_dim', 256),
        num_layers=getattr(config, 'latent_flow_num_layers', 4)
    ).to(device)
    
    # Load weights - try combined first, then individual
    if os.path.exists(combined_path) and not vae_ckpt and not flow_ckpt:
        print(f"Loading combined model from: {combined_path}")
        checkpoint = torch.load(combined_path, map_location=device, weights_only=True)
        vae.load_state_dict(checkpoint['vae_state_dict'])
        flow_model.load_state_dict(checkpoint['flow_model_state_dict'])
    else:
        # Load individual models
        if os.path.exists(vae_path):
            print(f"Loading VAE from: {vae_path}")
            vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
        else:
            raise FileNotFoundError(f"VAE model not found at {vae_path}")
        
        if os.path.exists(flow_path):
            print(f"Loading Flow model from: {flow_path}")
            flow_model.load_state_dict(torch.load(flow_path, map_location=device, weights_only=True))
        else:
            raise FileNotFoundError(f"Flow model not found at {flow_path}")
    
    vae.eval()
    flow_model.eval()
    
    # Create combined model
    combined_model = LatentFlowWithVAE(vae, flow_model).to(device)
    
    print(f"VAE+Flow model loaded successfully")
    print(f"  VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")
    print(f"  Flow parameters: {sum(p.numel() for p in flow_model.parameters()):,}")
    
    return vae, flow_model, combined_model


def validate_latent_linearity(vae, multi_pref_data, device, num_samples=5, save_dir=None):
    """
    Validate that the latent space is approximately one-dimensional.
    
    Args:
        vae: Trained LinearizedVAE
        multi_pref_data: Multi-preference data dictionary
        device: Device
        num_samples: Number of samples to visualize
        save_dir: Directory to save figures
    
    Returns:
        results: Dictionary with linearity metrics
    """
    print("\n" + "=" * 60)
    print("Validating Latent Space Linearity")
    print("=" * 60)
    
    x_train = multi_pref_data['x_train'].to(device)
    y_train_by_pref = multi_pref_data['y_train_by_pref']
    lambda_carbon_values = sorted(multi_pref_data['lambda_carbon_values'])
    n_train = multi_pref_data['n_train']
    
    # Move y_train to device
    y_train_by_pref_device = {lc: y.to(device) for lc, y in y_train_by_pref.items()}
    
    # Normalize lambda values
    lambda_max = max(lambda_carbon_values)
    
    results = {
        'explained_variance_ratio': [],
        'pc1_vs_lambda_correlation': []
    }
    
    from sklearn.decomposition import PCA
    from scipy.stats import pearsonr
    
    vae.eval()
    
    # Analyze multiple samples
    sample_indices = np.linspace(0, n_train - 1, num_samples, dtype=int)
    
    for sample_idx in sample_indices:
        print(f"\nSample {sample_idx}:")
        
        # Collect latent codes for this sample
        z_list = []
        with torch.no_grad():
            for lc in lambda_carbon_values:
                sol = y_train_by_pref_device[lc][sample_idx:sample_idx+1]
                scene = x_train[sample_idx:sample_idx+1]
                pref = torch.tensor([[lc / lambda_max]], device=device)
                
                z = vae.encode(scene, sol, pref, use_mean=True)
                z_list.append(z.cpu().numpy().flatten())
        
        z_array = np.array(z_list)  # [K, latent_dim]
        
        # PCA analysis
        pca = PCA(n_components=min(5, z_array.shape[1]))
        z_pca = pca.fit_transform(z_array)
        
        explained_var = pca.explained_variance_ratio_
        results['explained_variance_ratio'].append(explained_var)
        
        print(f"  PC1 explains {explained_var[0]:.1%} variance")
        print(f"  PC2 explains {explained_var[1]:.1%} variance")
        print(f"  PC1+PC2 explains {explained_var[0] + explained_var[1]:.1%} variance")
        
        # Check correlation between PC1 and lambda
        lambdas_normalized = np.array(lambda_carbon_values) / lambda_max
        corr, p_value = pearsonr(z_pca[:, 0], lambdas_normalized)
        results['pc1_vs_lambda_correlation'].append(abs(corr))
        
        print(f"  PC1 vs lambda correlation: {abs(corr):.4f} (p={p_value:.2e})")
        
        # Save visualization
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            vis_path = f'{save_dir}/latent_linearity_sample_{sample_idx}.png'
            visualize_latent_linearity(vae, x_train, y_train_by_pref_device, 
                                       sample_idx=sample_idx, save_path=vis_path)
    
    # Summary statistics
    print("\n" + "-" * 40)
    print("Summary Statistics:")
    pc1_mean = np.mean([ev[0] for ev in results['explained_variance_ratio']])
    pc1_std = np.std([ev[0] for ev in results['explained_variance_ratio']])
    corr_mean = np.mean(results['pc1_vs_lambda_correlation'])
    corr_std = np.std(results['pc1_vs_lambda_correlation'])
    
    print(f"  PC1 explained variance: {pc1_mean:.1%} +/- {pc1_std:.1%}")
    print(f"  PC1 vs lambda correlation: {corr_mean:.4f} +/- {corr_std:.4f}")
    
    results['pc1_mean'] = pc1_mean
    results['pc1_std'] = pc1_std
    results['corr_mean'] = corr_mean
    results['corr_std'] = corr_std
    
    return results


def wrap_angle_difference(diff, NPred_Va):
    """Wrap angle difference to [-pi, pi] for Va dimensions."""
    if NPred_Va <= 0:
        return diff
    wrapped = diff.clone()
    va_diff = wrapped[..., :NPred_Va]
    wrapped[..., :NPred_Va] = torch.atan2(torch.sin(va_diff), torch.cos(va_diff))
    return wrapped


def evaluate_vae_flow_model(combined_model, multi_pref_data, device, 
                            use_val=True, num_steps=20, method='heun'):
    """
    Evaluate VAE+Flow model on multi-preference data.
    
    Args:
        combined_model: LatentFlowWithVAE model
        multi_pref_data: Multi-preference data dictionary
        device: Device
        use_val: Use validation set (True) or training set (False)
        num_steps: Number of ODE integration steps
        method: ODE solver method
    
    Returns:
        results: Dictionary with evaluation metrics
    """
    print("\n" + "=" * 60)
    print("Evaluating VAE+Flow Model")
    print("=" * 60)
    
    # Select data
    if use_val and 'x_val' in multi_pref_data:
        x_data = multi_pref_data['x_val'].to(device)
        y_by_pref = multi_pref_data['y_val_by_pref']
        n_samples = multi_pref_data['n_val']
        data_split = 'validation'
    else:
        x_data = multi_pref_data['x_train'].to(device)
        y_by_pref = multi_pref_data['y_train_by_pref']
        n_samples = multi_pref_data['n_train']
        data_split = 'training'
    
    print(f"Using {data_split} set: {n_samples} samples")
    
    # Move y to device
    y_by_pref_device = {lc: y.to(device) for lc, y in y_by_pref.items()}
    
    lambda_carbon_values = sorted(multi_pref_data['lambda_carbon_values'])
    lambda_min = min(lambda_carbon_values)
    lambda_max = max(lambda_carbon_values)
    output_dim = multi_pref_data['output_dim']
    NPred_Va = multi_pref_data.get('NPred_Va', output_dim // 2)
    
    # Normalize lambda to [0, 1]
    r_values = [(lc - lambda_min) / (lambda_max - lambda_min) 
                for lc in lambda_carbon_values]
    
    combined_model.eval()
    
    results = {
        'per_pref_mse': {},
        'per_pref_mae': {},
        'overall_mse': 0.0,
        'overall_mae': 0.0
    }
    
    total_mse = 0.0
    total_mae = 0.0
    n_total = 0
    
    with torch.no_grad():
        for lc_idx, (lc, r) in enumerate(zip(lambda_carbon_values, r_values)):
            # Get ground truth
            y_gt = y_by_pref_device[lc]  # [n_samples, output_dim]
            
            # Get starting point (lambda=0 solution)
            y_start = y_by_pref_device[lambda_carbon_values[0]]
            pref_start = torch.zeros((n_samples, 1), device=device)
            z_start = combined_model.encode(x_data, y_start, pref_start)
            
            # Predict by integrating in latent space
            z_target = combined_model.flow_model.integrate(
                x_data, z_start, r_start=0.0, r_end=r, 
                num_steps=num_steps, method=method
            )
            y_pred = combined_model.decode(x_data, z_target)
            
            # Compute errors (with angle wrapping)
            diff = y_pred - y_gt
            diff_wrapped = wrap_angle_difference(diff, NPred_Va)
            
            mse = (diff_wrapped ** 2).mean().item()
            mae = diff_wrapped.abs().mean().item()
            
            results['per_pref_mse'][lc] = mse
            results['per_pref_mae'][lc] = mae
            
            total_mse += mse * n_samples
            total_mae += mae * n_samples
            n_total += n_samples
            
            if lc_idx % 10 == 0:
                print(f"  Lambda={lc:.2f}: MSE={mse:.6f}, MAE={mae:.6f}")
    
    results['overall_mse'] = total_mse / n_total
    results['overall_mae'] = total_mae / n_total
    
    print("\n" + "-" * 40)
    print("Overall Results:")
    print(f"  MSE: {results['overall_mse']:.6f}")
    print(f"  MAE: {results['overall_mae']:.6f}")
    
    return results


def compare_with_baseline(combined_model, multi_pref_data, device, baseline_model=None):
    """
    Compare VAE+Flow model with baseline models.
    
    Args:
        combined_model: LatentFlowWithVAE model
        multi_pref_data: Multi-preference data dictionary
        device: Device
        baseline_model: Optional baseline model for comparison
    
    Returns:
        comparison: Dictionary with comparison results
    """
    print("\n" + "=" * 60)
    print("Comparing VAE+Flow with Baselines")
    print("=" * 60)
    
    # Evaluate VAE+Flow
    vae_flow_results = evaluate_vae_flow_model(
        combined_model, multi_pref_data, device,
        use_val=True, num_steps=20, method='heun'
    )
    
    # Linear interpolation baseline
    print("\nBaseline: Linear Interpolation")
    linear_interp_results = evaluate_linear_interpolation(multi_pref_data, device)
    
    comparison = {
        'vae_flow': vae_flow_results,
        'linear_interpolation': linear_interp_results
    }
    
    # Print comparison
    print("\n" + "-" * 40)
    print("Comparison Summary:")
    print(f"{'Method':<25} {'MSE':>12} {'MAE':>12}")
    print("-" * 50)
    print(f"{'VAE+Flow':<25} {vae_flow_results['overall_mse']:>12.6f} {vae_flow_results['overall_mae']:>12.6f}")
    print(f"{'Linear Interpolation':<25} {linear_interp_results['overall_mse']:>12.6f} {linear_interp_results['overall_mae']:>12.6f}")
    
    improvement = (linear_interp_results['overall_mse'] - vae_flow_results['overall_mse']) / linear_interp_results['overall_mse'] * 100
    print(f"\nVAE+Flow improves MSE by {improvement:.1f}%")
    
    return comparison


def evaluate_linear_interpolation(multi_pref_data, device):
    """Evaluate linear interpolation baseline."""
    # Use validation set
    if 'x_val' in multi_pref_data:
        y_by_pref = {lc: y.to(device) for lc, y in multi_pref_data['y_val_by_pref'].items()}
        n_samples = multi_pref_data['n_val']
    else:
        y_by_pref = {lc: y.to(device) for lc, y in multi_pref_data['y_train_by_pref'].items()}
        n_samples = multi_pref_data['n_train']
    
    lambda_values = sorted(multi_pref_data['lambda_carbon_values'])
    output_dim = multi_pref_data['output_dim']
    NPred_Va = multi_pref_data.get('NPred_Va', output_dim // 2)
    
    y_start = y_by_pref[lambda_values[0]]
    y_end = y_by_pref[lambda_values[-1]]
    
    total_mse = 0.0
    total_mae = 0.0
    n_total = 0
    
    for lc in lambda_values:
        y_gt = y_by_pref[lc]
        
        # Linear interpolation
        t = (lc - lambda_values[0]) / (lambda_values[-1] - lambda_values[0])
        y_pred = (1 - t) * y_start + t * y_end
        
        diff = y_pred - y_gt
        diff_wrapped = wrap_angle_difference(diff, NPred_Va)
        
        mse = (diff_wrapped ** 2).mean().item()
        mae = diff_wrapped.abs().mean().item()
        
        total_mse += mse * n_samples
        total_mae += mae * n_samples
        n_total += n_samples
    
    return {
        'overall_mse': total_mse / n_total,
        'overall_mae': total_mae / n_total
    }


def visualize_trajectory_comparison(combined_model, multi_pref_data, device, 
                                    sample_idx=0, save_path=None):
    """
    Visualize predicted vs ground truth trajectories in latent and output space.
    
    Args:
        combined_model: LatentFlowWithVAE model
        multi_pref_data: Multi-preference data dictionary
        device: Device
        sample_idx: Which sample to visualize
        save_path: Path to save figure
    """
    from sklearn.decomposition import PCA
    
    x_data = multi_pref_data['x_train'].to(device)
    y_by_pref = {lc: y.to(device) for lc, y in multi_pref_data['y_train_by_pref'].items()}
    lambda_values = sorted(multi_pref_data['lambda_carbon_values'])
    
    lambda_min = min(lambda_values)
    lambda_max = max(lambda_values)
    r_values = [(lc - lambda_min) / (lambda_max - lambda_min) for lc in lambda_values]
    
    output_dim = multi_pref_data['output_dim']
    NPred_Va = multi_pref_data.get('NPred_Va', output_dim // 2)
    
    combined_model.eval()
    
    # Get ground truth trajectory
    z_gt_list = []
    y_gt_list = []
    
    scene = x_data[sample_idx:sample_idx+1]
    
    with torch.no_grad():
        for lc, r in zip(lambda_values, r_values):
            y_gt = y_by_pref[lc][sample_idx:sample_idx+1]
            y_gt_list.append(y_gt.cpu().numpy().flatten())
            
            pref = torch.tensor([[r]], device=device)
            z_gt = combined_model.encode(scene, y_gt, pref)
            z_gt_list.append(z_gt.cpu().numpy().flatten())
    
    z_gt_array = np.array(z_gt_list)
    y_gt_array = np.array(y_gt_list)
    
    # Get predicted trajectory (from z_0)
    y_start = y_by_pref[lambda_values[0]][sample_idx:sample_idx+1]
    pref_start = torch.zeros((1, 1), device=device)
    z_start = combined_model.encode(scene, y_start, pref_start)
    
    z_pred_trajectory = combined_model.flow_model.integrate_trajectory(
        scene, z_start, r_values, method='heun'
    )
    z_pred_array = z_pred_trajectory.squeeze(0).cpu().numpy()  # [K, latent_dim]
    
    # Decode predicted trajectory
    y_pred_list = []
    for i in range(len(r_values)):
        z_i = z_pred_trajectory[:, i, :]
        y_pred = combined_model.decode(scene, z_i)
        y_pred_list.append(y_pred.cpu().numpy().flatten())
    y_pred_array = np.array(y_pred_list)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Latent space PCA
    pca_z = PCA(n_components=2)
    z_all = np.vstack([z_gt_array, z_pred_array])
    z_pca_all = pca_z.fit_transform(z_all)
    z_gt_pca = z_pca_all[:len(z_gt_array)]
    z_pred_pca = z_pca_all[len(z_gt_array):]
    
    ax = axes[0, 0]
    ax.plot(z_gt_pca[:, 0], z_gt_pca[:, 1], 'b-o', label='Ground Truth', markersize=3, alpha=0.7)
    ax.plot(z_pred_pca[:, 0], z_pred_pca[:, 1], 'r--s', label='Predicted', markersize=3, alpha=0.7)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Latent Space Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Latent PC1 vs lambda
    ax = axes[0, 1]
    ax.plot(lambda_values, z_gt_pca[:, 0], 'b-o', label='GT PC1', markersize=4)
    ax.plot(lambda_values, z_pred_pca[:, 0], 'r--s', label='Pred PC1', markersize=4)
    ax.set_xlabel('Lambda')
    ax.set_ylabel('PC1')
    ax.set_title('Latent PC1 vs Lambda')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Output space error vs lambda
    errors = []
    for i, lc in enumerate(lambda_values):
        diff = y_pred_array[i] - y_gt_array[i]
        # Wrap Va dimensions
        diff[:NPred_Va] = np.arctan2(np.sin(diff[:NPred_Va]), np.cos(diff[:NPred_Va]))
        errors.append(np.sqrt(np.mean(diff ** 2)))
    
    ax = axes[1, 0]
    ax.plot(lambda_values, errors, 'g-o', markersize=4)
    ax.set_xlabel('Lambda')
    ax.set_ylabel('RMSE')
    ax.set_title('Prediction Error vs Lambda')
    ax.grid(True, alpha=0.3)
    
    # 4. Selected output dimensions
    ax = axes[1, 1]
    dims_to_show = [0, NPred_Va // 2, NPred_Va, NPred_Va + output_dim // 4]
    dims_to_show = [d for d in dims_to_show if d < output_dim]
    
    for d in dims_to_show[:4]:
        label = f'Va[{d}]' if d < NPred_Va else f'Vm[{d-NPred_Va}]'
        ax.plot(lambda_values, y_gt_array[:, d], '-', label=f'{label} GT', alpha=0.7)
        ax.plot(lambda_values, y_pred_array[:, d], '--', label=f'{label} Pred', alpha=0.7)
    
    ax.set_xlabel('Lambda')
    ax.set_ylabel('Value')
    ax.set_title('Selected Output Dimensions')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'VAE+Flow Trajectory Comparison (Sample {sample_idx})')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory comparison to {save_path}")
    
    plt.close()


def main():
    """Main function for validation script."""
    parser = argparse.ArgumentParser(description='Validate VAE+Flow Model')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['validate', 'evaluate', 'compare', 'visualize', 'all'],
                       help='Validation mode')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of samples for visualization')
    parser.add_argument('--save-dir', type=str, default='main_part/results/vae_flow_validation',
                       help='Directory to save results')
    parser.add_argument('--vae-ckpt', type=str, default=None,
                       help='Path to VAE checkpoint (e.g., saved_models/linearized_vae_epoch2900.pth)')
    parser.add_argument('--flow-ckpt', type=str, default=None,
                       help='Path to Flow checkpoint (e.g., saved_models/latent_flow_epoch500.pth)')
    args = parser.parse_args()
    
    # Load config and data
    config = get_config()
    device = config.device
    
    print("=" * 60)
    print("VAE+Flow Model Validation")
    print("=" * 60)
    
    # Load multi-preference dataset
    multi_pref_data, sys_data = load_multi_preference_dataset(config)
    
    # Load model with custom checkpoint paths if provided
    try:
        vae, flow_model, combined_model = load_vae_flow_model(
            config, multi_pref_data, device,
            vae_ckpt=args.vae_ckpt,
            flow_ckpt=args.flow_ckpt
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease train the VAE+Flow model first using:")
        print("  MODEL_TYPE=vae_flow python train_supervised.py")
        print("\nOr specify custom checkpoint paths:")
        print("  python validate_vae_flow.py --vae-ckpt path/to/vae.pth --flow-ckpt path/to/flow.pth")
        return
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.mode in ['validate', 'all']:
        # Validate latent linearity
        linearity_results = validate_latent_linearity(
            vae, multi_pref_data, device, 
            num_samples=args.num_samples,
            save_dir=args.save_dir
        )
    
    if args.mode in ['evaluate', 'all']:
        # Evaluate model
        eval_results = evaluate_vae_flow_model(
            combined_model, multi_pref_data, device,
            use_val=True, num_steps=20, method='heun'
        )
    
    if args.mode in ['compare', 'all']:
        # Compare with baselines
        comparison = compare_with_baseline(
            combined_model, multi_pref_data, device
        )
    
    if args.mode in ['visualize', 'all']:
        # Visualize trajectories
        for sample_idx in [0, 10, 20]:
            if sample_idx < multi_pref_data['n_train']:
                save_path = f'{args.save_dir}/trajectory_comparison_sample_{sample_idx}.png'
                visualize_trajectory_comparison(
                    combined_model, multi_pref_data, device,
                    sample_idx=sample_idx, save_path=save_path
                )
    
    print("\n" + "=" * 60)
    print("Validation Complete")
    print(f"Results saved to: {args.save_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
