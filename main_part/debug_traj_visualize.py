#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""debug_traj_visualize.py

Debug visualization module for trajectory student distillation.
Plots Pareto front showing:
- Anchor (Y0) starting points
- Target (Y_star) from GT + teacher pseudo labels
- Student predicted endpoints

This module is designed to be imported and called from distill_traj_student.py
with minimal changes to the main training script.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Tuple


def _ensure_numpy(x):
    """Convert tensor to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_objectives_batch(
    Y: torch.Tensor,
    ctx,
    n_va: int,
    reconstruct_fn,
    get_genload_fn,
    compute_cost_fn,
    compute_carbon_fn,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cost and carbon objectives for a batch of solutions.
    
    Args:
        Y: [B, K, D] or [B, D] tensor of solutions in NGT format
        ctx: EvalContext with power system data
        n_va: Number of Va dimensions
        reconstruct_fn: Function to reconstruct full Vm/Va from partial
        get_genload_fn: Function to compute Pg, Qg from V
        compute_cost_fn: Function to compute generation cost
        compute_carbon_fn: Function to compute carbon emission
        
    Returns:
        costs: [B, K] or [B] array of cost values
        carbons: [B, K] or [B] array of carbon values
    """
    Y_np = _ensure_numpy(Y)
    
    if Y_np.ndim == 2:
        # [B, D] -> [B, 1, D]
        Y_np = Y_np[:, np.newaxis, :]
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, K, D = Y_np.shape
    costs = np.zeros((B, K))
    carbons = np.zeros((B, K))
    
    for k in range(K):
        Y_k = Y_np[:, k, :]  # [B, D]
        
        # Reconstruct full Vm/Va
        Pred_Vm_full, Pred_Va_full = reconstruct_fn(ctx, Y_k)
        
        # Compute power flow
        Pred_V = Pred_Vm_full * np.exp(1j * Pred_Va_full)
        Pred_Pg, Pred_Qg, _, _ = get_genload_fn(
            Pred_V, ctx.Pdtest, ctx.Qdtest, 
            ctx.bus_Pg, ctx.bus_Qg, ctx.Ybus
        )
        
        # Compute objectives
        costs[:, k] = compute_cost_fn(Pred_Pg, ctx)
        carbons[:, k] = compute_carbon_fn(Pred_Pg, ctx)
    
    if squeeze_output:
        costs = costs.squeeze(1)
        carbons = carbons.squeeze(1)
    
    return costs, carbons


def plot_trajectory_debug(
    fine_norm: torch.Tensor,
    Y0: torch.Tensor,           # [B, Kf, D] anchor trajectory
    Y_star: torch.Tensor,       # [B, Kf, D] target (GT + teacher pseudo)
    Y_pred: torch.Tensor,       # [B, Kf, D] student predicted endpoints
    ctx,
    n_va: int,
    sample_idx: int = 0,        # Which sample in batch to visualize
    save_path: Optional[str] = None,
    title_suffix: str = "",
) -> plt.Figure:
    """
    Plot Pareto front comparison for a single sample.
    
    Shows:
    - Anchor (Y0): Starting points (grey circles)
    - Target (Y_star): GT + teacher pseudo labels (blue squares)  
    - Predicted (Y_pred): Student endpoints (red triangles)
    - Arrows from anchor to target and anchor to predicted
    
    Args:
        fine_norm: [Kf] normalized preference grid
        Y0: [B, Kf, D] anchor trajectory
        Y_star: [B, Kf, D] target trajectory
        Y_pred: [B, Kf, D] student predicted trajectory
        ctx: EvalContext
        n_va: Number of Va dimensions
        sample_idx: Which sample to visualize
        save_path: Optional path to save figure
        title_suffix: Additional text for title
    """
    # Import here to avoid circular imports
    from unified_eval import (
        reconstruct_full_from_partial,
        _compute_cost,
        _compute_carbon,
    )
    from utils import get_genload
    
    # Extract single sample
    Y0_s = Y0[sample_idx:sample_idx+1]      # [1, Kf, D]
    Y_star_s = Y_star[sample_idx:sample_idx+1]
    Y_pred_s = Y_pred[sample_idx:sample_idx+1]
    
    # Compute objectives
    cost_Y0, carbon_Y0 = compute_objectives_batch(
        Y0_s, ctx, n_va, reconstruct_full_from_partial, 
        get_genload, _compute_cost, _compute_carbon
    )
    cost_star, carbon_star = compute_objectives_batch(
        Y_star_s, ctx, n_va, reconstruct_full_from_partial,
        get_genload, _compute_cost, _compute_carbon
    )
    cost_pred, carbon_pred = compute_objectives_batch(
        Y_pred_s, ctx, n_va, reconstruct_full_from_partial,
        get_genload, _compute_cost, _compute_carbon
    )
    
    # Squeeze batch dimension
    cost_Y0 = cost_Y0.squeeze(0)        # [Kf]
    carbon_Y0 = carbon_Y0.squeeze(0)
    cost_star = cost_star.squeeze(0)
    carbon_star = carbon_star.squeeze(0)
    cost_pred = cost_pred.squeeze(0)
    carbon_pred = carbon_pred.squeeze(0)
    
    fine_norm_np = _ensure_numpy(fine_norm)
    Kf = len(fine_norm_np)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color map for lambda values
    colors = plt.cm.viridis(fine_norm_np)
    
    # Plot arrows from anchor to target and anchor to prediction
    for k in range(Kf):
        # Anchor -> Target (blue arrow)
        ax.annotate('', xy=(cost_star[k], carbon_star[k]), 
                   xytext=(cost_Y0[k], carbon_Y0[k]),
                   arrowprops=dict(arrowstyle='->', color='blue', alpha=0.3, lw=0.8))
        
        # Anchor -> Prediction (red arrow)
        ax.annotate('', xy=(cost_pred[k], carbon_pred[k]),
                   xytext=(cost_Y0[k], carbon_Y0[k]),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.3, lw=0.8))
    
    # Plot points
    scatter_Y0 = ax.scatter(cost_Y0, carbon_Y0, c=fine_norm_np, cmap='viridis',
                            marker='o', s=80, alpha=0.7, edgecolors='grey', 
                            linewidths=1.5, label='Anchor (Y0)', zorder=3)
    
    ax.scatter(cost_star, carbon_star, c=fine_norm_np, cmap='viridis',
               marker='s', s=100, alpha=0.9, edgecolors='blue',
               linewidths=2, label='Target (Y*)', zorder=4)
    
    ax.scatter(cost_pred, carbon_pred, c=fine_norm_np, cmap='viridis',
               marker='^', s=100, alpha=0.9, edgecolors='red',
               linewidths=2, label='Predicted', zorder=5)
    
    # Colorbar
    cbar = plt.colorbar(scatter_Y0, ax=ax)
    cbar.set_label('Normalized Lambda (0=cost, 1=carbon)', fontsize=11)
    
    # Labels and title
    ax.set_xlabel('Generation Cost', fontsize=12)
    ax.set_ylabel('Carbon Emission', fontsize=12)
    ax.set_title(f'Trajectory Debug: Anchor -> Target vs Prediction\n'
                 f'Sample {sample_idx}, Kf={Kf} points{title_suffix}', fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved debug plot: {save_path}')
    
    return fig


def plot_trajectory_errors(
    fine_norm: torch.Tensor,
    Y_star: torch.Tensor,       # [B, Kf, D] target
    Y_pred: torch.Tensor,       # [B, Kf, D] student predicted
    n_va: int,
    sample_idx: int = 0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot per-preference-point errors in solution space.
    
    Shows:
    - MSE between Y_star and Y_pred for each lambda point
    - Separate plots for Va and Vm dimensions
    """
    Y_star_s = _ensure_numpy(Y_star[sample_idx])   # [Kf, D]
    Y_pred_s = _ensure_numpy(Y_pred[sample_idx])
    fine_norm_np = _ensure_numpy(fine_norm)
    
    Kf, D = Y_star_s.shape
    
    # Compute per-point MSE
    mse_per_k = np.mean((Y_star_s - Y_pred_s) ** 2, axis=1)  # [Kf]
    
    # Separate Va and Vm
    mse_va = np.mean((Y_star_s[:, :n_va] - Y_pred_s[:, :n_va]) ** 2, axis=1)
    mse_vm = np.mean((Y_star_s[:, n_va:] - Y_pred_s[:, n_va:]) ** 2, axis=1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Total MSE
    axes[0].plot(fine_norm_np, mse_per_k, 'o-', color='purple', markersize=5)
    axes[0].set_xlabel('Normalized Lambda')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Total MSE per Lambda')
    axes[0].grid(True, alpha=0.3)
    
    # Va MSE
    axes[1].plot(fine_norm_np, mse_va, 's-', color='blue', markersize=5)
    axes[1].set_xlabel('Normalized Lambda')
    axes[1].set_ylabel('MSE (Va)')
    axes[1].set_title(f'Va MSE (dims 0:{n_va})')
    axes[1].grid(True, alpha=0.3)
    
    # Vm MSE
    axes[2].plot(fine_norm_np, mse_vm, '^-', color='green', markersize=5)
    axes[2].set_xlabel('Normalized Lambda')
    axes[2].set_ylabel('MSE (Vm)')
    axes[2].set_title(f'Vm MSE (dims {n_va}:{D})')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Prediction Errors by Lambda (Sample {sample_idx})', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved error plot: {save_path}')
    
    return fig


def debug_single_batch(
    Y0: torch.Tensor,
    Y_star: torch.Tensor,
    Y_pred: torch.Tensor,
    fine_norm: torch.Tensor,
    ctx,
    n_va: int,
    sample_idx: int = 0,
    save_dir: str = 'debug_plots',
    epoch: int = 0,
) -> Dict[str, Any]:
    """
    Complete debug visualization for a single batch.
    
    Call this during training to visualize what's happening.
    
    Args:
        Y0: [B, Kf, D] anchor trajectory
        Y_star: [B, Kf, D] target trajectory (GT + teacher pseudo)
        Y_pred: [B, Kf, D] student predicted trajectory
        fine_norm: [Kf] normalized preference grid
        ctx: EvalContext
        n_va: Number of Va dimensions
        sample_idx: Which sample to visualize
        save_dir: Directory to save plots
        epoch: Current epoch (for filename)
        
    Returns:
        Dict with computed metrics
    """
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot Pareto front
    fig1 = plot_trajectory_debug(
        fine_norm=fine_norm,
        Y0=Y0,
        Y_star=Y_star,
        Y_pred=Y_pred,
        ctx=ctx,
        n_va=n_va,
        sample_idx=sample_idx,
        save_path=os.path.join(save_dir, f'pareto_E{epoch:04d}.png'),
        title_suffix=f' (Epoch {epoch})',
    )
    plt.close(fig1)
    
    # Plot errors
    fig2 = plot_trajectory_errors(
        fine_norm=fine_norm,
        Y_star=Y_star,
        Y_pred=Y_pred,
        n_va=n_va,
        sample_idx=sample_idx,
        save_path=os.path.join(save_dir, f'errors_E{epoch:04d}.png'),
    )
    plt.close(fig2)
    
    # Compute summary metrics
    Y_star_np = _ensure_numpy(Y_star[sample_idx])
    Y_pred_np = _ensure_numpy(Y_pred[sample_idx])
    Y0_np = _ensure_numpy(Y0[sample_idx])
    
    mse_pred_to_target = np.mean((Y_pred_np - Y_star_np) ** 2)
    mse_anchor_to_target = np.mean((Y0_np - Y_star_np) ** 2)
    improvement_ratio = mse_anchor_to_target / (mse_pred_to_target + 1e-12)
    
    metrics = {
        'mse_pred_to_target': mse_pred_to_target,
        'mse_anchor_to_target': mse_anchor_to_target,
        'improvement_ratio': improvement_ratio,
    }
    
    print(f'\n[Debug Sample {sample_idx}] Epoch {epoch}:')
    print(f'  MSE(Y0 -> Y*) = {mse_anchor_to_target:.6f}')
    print(f'  MSE(pred -> Y*) = {mse_pred_to_target:.6f}')
    print(f'  Improvement ratio = {improvement_ratio:.2f}x')
    
    return metrics


def create_eval_context_for_debug(multi_pref_data, sys_data, device):
    """
    Create EvalContext for debug visualization.
    
    Args:
        multi_pref_data: Multi-preference dataset dict
        sys_data: System data dict
        device: torch device
        
    Returns:
        ctx: EvalContext object
    """
    from unified_eval import EvalContext
    
    # Get validation data
    x_val = multi_pref_data.get('x_val')
    if x_val is None:
        x_val = multi_pref_data.get('x_train')
    
    if isinstance(x_val, torch.Tensor):
        x_val = x_val.cpu().numpy()
    
    # Get one sample of y for shape info
    lambda_values = list(multi_pref_data['y_train_by_pref'].keys())
    y_sample = multi_pref_data['y_train_by_pref'][lambda_values[0]]
    if isinstance(y_sample, torch.Tensor):
        y_sample = y_sample.cpu().numpy()
    
    # Create context
    ctx = EvalContext(
        x_test=x_val,
        y_test=y_sample,  # Just for shape
        sys_data=sys_data,
        use_post_processing=False,
    )
    
    return ctx
