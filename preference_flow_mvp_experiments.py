#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preference Flow MVP Experiments
根据 ideas/约束流模型学习.md 中的 MVP 实验计划，进行可行性验证

MVP-0: 分析数据中的 kink 情况
MVP-1: 简单的差分速度场拟合实验
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_part.config import get_config
from main_part.data_loader import load_multi_preference_dataset

# Set matplotlib to use non-Chinese fonts
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

def mvp0_analyze_kinks(multi_pref_data, sample_idx=0, output_dir='results'):
    """
    MVP-0: 分析数据中的 kink 情况
    
    对每个场景 s，选择几个关键输出维度，画出它们随 λ 的曲线，
    以及 Δx_k = |x_{k+1}-x_k| 随 k 的曲线。
    
    Args:
        multi_pref_data: Multi-preference data dictionary
        sample_idx: Index of sample to analyze
        output_dir: Directory to save plots
    """
    print("\n" + "=" * 80)
    print("MVP-0: Analyzing Kinks in Preference Trajectories")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    y_train_by_pref = multi_pref_data['y_train_by_pref']
    lambda_carbon_values = sorted(multi_pref_data['lambda_carbon_values'])
    output_dim = multi_pref_data['output_dim']
    
    # Get solutions for this sample across all preferences
    solutions = {}
    for lc in lambda_carbon_values:
        if lc in y_train_by_pref:
            # Convert to numpy for analysis
            if isinstance(y_train_by_pref[lc], torch.Tensor):
                solutions[lc] = y_train_by_pref[lc][sample_idx].detach().cpu().numpy()
            else:
                solutions[lc] = y_train_by_pref[lc][sample_idx]
    
    # Sort by lambda_carbon
    sorted_lambdas = sorted(solutions.keys())
    sorted_solutions = [solutions[lc] for lc in sorted_lambdas]
    sorted_solutions = np.array(sorted_solutions)  # Shape: [n_prefs, output_dim]
    
    print(f"\nAnalyzing sample {sample_idx}")
    print(f"  Number of preferences: {len(sorted_lambdas)}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Lambda range: [{sorted_lambdas[0]:.2f}, {sorted_lambdas[-1]:.2f}]")
    
    # Calculate Δx_k = |x_{k+1} - x_k| for each k
    deltas = []
    for k in range(len(sorted_solutions) - 1):
        delta = np.linalg.norm(sorted_solutions[k+1] - sorted_solutions[k])
        deltas.append(delta)
    deltas = np.array(deltas)
    
    # Find kink regions (where delta is large)
    delta_threshold = np.percentile(deltas, 90)  # Top 10% as kinks
    kink_indices = np.where(deltas > delta_threshold)[0]
    
    print(f"\nKink Analysis:")
    print(f"  Mean delta: {np.mean(deltas):.6f}")
    print(f"  Max delta: {np.max(deltas):.6f}")
    print(f"  Delta threshold (90th percentile): {delta_threshold:.6f}")
    print(f"  Number of kink regions: {len(kink_indices)}")
    if len(kink_indices) > 0:
        print(f"  Kink locations (lambda indices): {kink_indices.tolist()}")
        print(f"  Kink lambda values: {[sorted_lambdas[i] for i in kink_indices]}")
    
    # Select a few key output dimensions to visualize
    # Choose dimensions that show variation
    var_by_dim = np.var(sorted_solutions, axis=0)
    top_var_indices = np.argsort(var_by_dim)[-5:]  # Top 5 most varying dimensions
    
    print(f"\nSelected key dimensions (top 5 by variance): {top_var_indices.tolist()}")
    
    # Plot 1: Delta x_k vs k
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot delta curve
    axes[0].plot(range(len(deltas)), deltas, 'b-', linewidth=1.5, label='|x_{k+1} - x_k|')
    axes[0].axhline(y=delta_threshold, color='r', linestyle='--', linewidth=1, label=f'Threshold (90th percentile)')
    if len(kink_indices) > 0:
        axes[0].scatter(kink_indices, deltas[kink_indices], color='red', s=50, zorder=5, label='Kink regions')
    axes[0].set_xlabel('Preference Index k', fontsize=12)
    axes[0].set_ylabel('|x_{k+1} - x_k|', fontsize=12)
    axes[0].set_title(f'Sample {sample_idx}: Trajectory Discontinuity Analysis', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Key variables vs lambda
    for i, dim_idx in enumerate(top_var_indices):
        values = sorted_solutions[:, dim_idx]
        axes[1].plot(sorted_lambdas, values, 'o-', linewidth=1.5, markersize=3, 
                    label=f'Dim {dim_idx}')
    
    # Mark kink regions
    if len(kink_indices) > 0:
        for kink_idx in kink_indices:
            axes[1].axvline(x=sorted_lambdas[kink_idx], color='red', linestyle='--', 
                          alpha=0.5, linewidth=1)
    
    axes[1].set_xlabel('Lambda Carbon (λ)', fontsize=12)
    axes[1].set_ylabel('Variable Value', fontsize=12)
    axes[1].set_title(f'Sample {sample_idx}: Key Variables vs Preference', fontsize=14)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'mvp0_kink_analysis_sample_{sample_idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {save_path}")
    plt.close()
    
    # Plot 3: Heatmap of all dimensions (optional, for overview)
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(sorted_solutions.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Preference Index (Lambda Carbon)', fontsize=12)
    ax.set_ylabel('Output Dimension', fontsize=12)
    ax.set_title(f'Sample {sample_idx}: Full Trajectory Heatmap', fontsize=14)
    plt.colorbar(im, ax=ax, label='Variable Value')
    
    # Mark kink regions
    if len(kink_indices) > 0:
        for kink_idx in kink_indices:
            ax.axvline(x=kink_idx, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'mvp0_trajectory_heatmap_sample_{sample_idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved heatmap to: {save_path}")
    plt.close()
    
    return {
        'deltas': deltas,
        'kink_indices': kink_indices,
        'kink_lambdas': [sorted_lambdas[i] for i in kink_indices] if len(kink_indices) > 0 else [],
        'delta_threshold': delta_threshold,
        'mean_delta': np.mean(deltas),
        'max_delta': np.max(deltas)
    }


def mvp1_velocity_field_fitting(multi_pref_data, sample_idx=0, output_dir='results', 
                                hidden_dim=128, num_layers=3, num_epochs=200, lr=1e-3):
    """
    MVP-1: 简单的差分速度场拟合实验
    
    用简单的 MLP 学习 dx/dλ，从 x(λ_1) 用 Euler 积分到 x(λ_K)，
    看能否接近真实 x(λ_K)。
    
    Args:
        multi_pref_data: Multi-preference data dictionary
        sample_idx: Index of sample to use
        output_dir: Directory to save results
        hidden_dim: Hidden dimension for MLP
        num_layers: Number of layers for MLP
        num_epochs: Training epochs
        lr: Learning rate
    """
    print("\n" + "=" * 80)
    print("MVP-1: Velocity Field Fitting and Euler Integration")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    y_train_by_pref = multi_pref_data['y_train_by_pref']
    lambda_carbon_values = sorted(multi_pref_data['lambda_carbon_values'])
    output_dim = multi_pref_data['output_dim']
    
    # Get solutions for this sample
    solutions = {}
    for lc in lambda_carbon_values:
        if lc in y_train_by_pref:
            if isinstance(y_train_by_pref[lc], torch.Tensor):
                solutions[lc] = y_train_by_pref[lc][sample_idx].to(device)
            else:
                solutions[lc] = torch.tensor(y_train_by_pref[lc][sample_idx], device=device)
    
    sorted_lambdas = sorted(solutions.keys())
    sorted_solutions = torch.stack([solutions[lc] for lc in sorted_lambdas])  # [n_prefs, output_dim]
    
    print(f"\nSample {sample_idx}:")
    print(f"  Number of preferences: {len(sorted_lambdas)}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Lambda range: [{sorted_lambdas[0]:.2f}, {sorted_lambdas[-1]:.2f}]")
    
    # Normalize lambda to [0, 1] for better training
    lambda_min = sorted_lambdas[0]
    lambda_max = sorted_lambdas[-1]
    lambda_normalized = [(lc - lambda_min) / (lambda_max - lambda_min) if lambda_max > lambda_min else 0.0 
                         for lc in sorted_lambdas]
    lambda_tensor = torch.tensor(lambda_normalized, device=device, dtype=torch.float32).unsqueeze(1)
    
    # Compute finite difference velocities (ground truth)
    velocities_gt = []
    for k in range(len(sorted_solutions) - 1):
        dx = sorted_solutions[k+1] - sorted_solutions[k]
        dlambda = lambda_tensor[k+1] - lambda_tensor[k]
        v = dx / (dlambda + 1e-8)  # Avoid division by zero
        velocities_gt.append(v)
    velocities_gt = torch.stack(velocities_gt)  # [n_prefs-1, output_dim]
    
    # Prepare training data: (x_k, lambda_k) -> v_k
    x_train = sorted_solutions[:-1]  # [n_prefs-1, output_dim]
    lambda_train = lambda_tensor[:-1]  # [n_prefs-1, 1]
    v_train = velocities_gt  # [n_prefs-1, output_dim]
    
    print(f"\nTraining data:")
    print(f"  x_train shape: {x_train.shape}")
    print(f"  lambda_train shape: {lambda_train.shape}")
    print(f"  v_train shape: {v_train.shape}")
    
    # Create simple MLP: input = [x, lambda], output = velocity
    class VelocityMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
            super().__init__()
            layers = []
            # Input: [x, lambda] = [output_dim + 1]
            layers.append(nn.Linear(output_dim + 1, hidden_dim))
            layers.append(nn.ReLU())
            
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.net = nn.Sequential(*layers)
        
        def forward(self, x, lambda_val):
            # x: [batch, output_dim], lambda_val: [batch, 1]
            x_concat = torch.cat([x, lambda_val], dim=1)
            return self.net(x_concat)
    
    model = VelocityMLP(output_dim, output_dim, hidden_dim, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"\nModel architecture:")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num layers: {num_layers}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    losses = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        v_pred = model(x_train, lambda_train)
        loss = criterion(v_pred, v_train)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: Loss = {loss.item():.6f}")
    
    print(f"  Final loss: {losses[-1]:.6f}")
    
    # Test: Euler integration from lambda_0 to lambda_K
    print(f"\nTesting Euler integration...")
    model.eval()
    
    with torch.no_grad():
        # Start from first point
        x_integrated = [sorted_solutions[0].clone()]
        lambda_current = lambda_tensor[0]
        x_current = sorted_solutions[0].clone()
        
        # Euler integration with small steps
        n_steps = len(sorted_lambdas) - 1
        step_size = (lambda_tensor[-1] - lambda_tensor[0]) / n_steps
        
        for step in range(n_steps):
            # Predict velocity at current point
            v_pred = model(x_current.unsqueeze(0), lambda_current.unsqueeze(0))
            v_pred = v_pred.squeeze(0)
            
            # Euler step
            x_current = x_current + step_size * v_pred
            lambda_current = lambda_tensor[0] + (step + 1) * step_size
            
            x_integrated.append(x_current.clone())
        
        x_integrated = torch.stack(x_integrated)  # [n_prefs, output_dim]
    
    # Compute errors
    errors = torch.norm(x_integrated - sorted_solutions, dim=1)  # [n_prefs]
    final_error = errors[-1].item()
    mean_error = errors.mean().item()
    
    print(f"\nIntegration Results:")
    print(f"  Final point error: {final_error:.6f}")
    print(f"  Mean error along trajectory: {mean_error:.6f}")
    print(f"  Max error: {errors.max().item():.6f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training loss
    axes[0, 0].plot(losses)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss', fontsize=14)
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Error along trajectory
    axes[0, 1].plot(sorted_lambdas, errors.cpu().numpy(), 'b-', linewidth=1.5)
    axes[0, 1].set_xlabel('Lambda Carbon (λ)', fontsize=12)
    axes[0, 1].set_ylabel('Integration Error ||x_pred - x_true||', fontsize=12)
    axes[0, 1].set_title('Error Along Trajectory', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Compare a few key dimensions
    var_by_dim = torch.var(sorted_solutions, dim=0)
    top_var_indices = torch.argsort(var_by_dim)[-3:].cpu().numpy()  # Top 3
    
    for dim_idx in top_var_indices:
        true_vals = sorted_solutions[:, dim_idx].cpu().numpy()
        pred_vals = x_integrated[:, dim_idx].cpu().numpy()
        axes[1, 0].plot(sorted_lambdas, true_vals, 'o-', linewidth=1.5, markersize=3, 
                       label=f'True Dim {dim_idx}')
        axes[1, 0].plot(sorted_lambdas, pred_vals, '--', linewidth=1.5, 
                       label=f'Pred Dim {dim_idx}')
    
    axes[1, 0].set_xlabel('Lambda Carbon (λ)', fontsize=12)
    axes[1, 0].set_ylabel('Variable Value', fontsize=12)
    axes[1, 0].set_title('True vs Integrated Trajectory (Key Dimensions)', fontsize=14)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Velocity prediction error
    with torch.no_grad():
        v_pred_all = model(x_train, lambda_train)
        v_error = torch.norm(v_pred_all - v_train, dim=1).cpu().numpy()
    
    axes[1, 1].plot(sorted_lambdas[:-1], v_error, 'r-', linewidth=1.5)
    axes[1, 1].set_xlabel('Lambda Carbon (λ)', fontsize=12)
    axes[1, 1].set_ylabel('Velocity Prediction Error', fontsize=12)
    axes[1, 1].set_title('Velocity Field Prediction Error', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'mvp1_velocity_fitting_sample_{sample_idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {save_path}")
    plt.close()
    
    return {
        'model': model,
        'final_error': final_error,
        'mean_error': mean_error,
        'max_error': errors.max().item(),
        'training_losses': losses,
        'integration_errors': errors.cpu().numpy()
    }


def main():
    """Main function to run MVP experiments"""
    print("=" * 80)
    print("Preference Flow MVP Experiments")
    print("=" * 80)
    
    # Load configuration
    config = get_config()
    
    # Load multi-preference dataset
    print("\nLoading multi-preference dataset...")
    multi_pref_data, sys_data = load_multi_preference_dataset(config)
    
    n_samples = multi_pref_data['n_train']
    print(f"\nDataset loaded:")
    print(f"  Training samples: {n_samples}")
    print(f"  Preferences: {len(multi_pref_data['lambda_carbon_values'])}")
    
    # Create output directory
    output_dir = 'results/mvp_experiments'
    os.makedirs(output_dir, exist_ok=True)
    
    # Run MVP-0: Analyze kinks for a few samples
    print("\n" + "=" * 80)
    print("Running MVP-0: Kink Analysis")
    print("=" * 80)
    
    sample_indices_to_analyze = [0, min(10, n_samples-1), min(20, n_samples-1)]
    kink_results = {}
    
    for sample_idx in sample_indices_to_analyze:
        print(f"\n--- Analyzing sample {sample_idx} ---")
        result = mvp0_analyze_kinks(multi_pref_data, sample_idx=sample_idx, output_dir=output_dir)
        kink_results[sample_idx] = result
    
    # Summary of kink analysis
    print("\n" + "=" * 80)
    print("MVP-0 Summary:")
    print("=" * 80)
    for sample_idx, result in kink_results.items():
        print(f"\nSample {sample_idx}:")
        print(f"  Mean delta: {result['mean_delta']:.6f}")
        print(f"  Max delta: {result['max_delta']:.6f}")
        print(f"  Number of kinks: {len(result['kink_indices'])}")
        if len(result['kink_lambdas']) > 0:
            print(f"  Kink lambda values: {result['kink_lambdas']}")
    
    # Run MVP-1: Velocity field fitting for a sample
    print("\n" + "=" * 80)
    print("Running MVP-1: Velocity Field Fitting")
    print("=" * 80)
    
    sample_idx = 0  # Use first sample for MVP-1
    mvp1_result = mvp1_velocity_field_fitting(
        multi_pref_data, 
        sample_idx=sample_idx, 
        output_dir=output_dir,
        hidden_dim=128,
        num_layers=3,
        num_epochs=200,
        lr=1e-3
    )
    
    print("\n" + "=" * 80)
    print("MVP-1 Summary:")
    print("=" * 80)
    print(f"  Final integration error: {mvp1_result['final_error']:.6f}")
    print(f"  Mean error along trajectory: {mvp1_result['mean_error']:.6f}")
    print(f"  Max error: {mvp1_result['max_error']:.6f}")
    
    # Determine feasibility
    print("\n" + "=" * 80)
    print("Feasibility Assessment:")
    print("=" * 80)
    
    # MVP-0 assessment
    avg_kinks = np.mean([len(r['kink_indices']) for r in kink_results.values()])
    avg_max_delta = np.mean([r['max_delta'] for r in kink_results.values()])
    
    print(f"\nMVP-0 (Kink Analysis):")
    print(f"  Average number of kinks per sample: {avg_kinks:.1f}")
    print(f"  Average max delta: {avg_max_delta:.6f}")
    if avg_kinks < 5:
        print("  -> Kinks are sparse, flow model should be feasible")
    elif avg_kinks < 15:
        print("  -> Moderate kinks, may need MoE or stronger corrector")
    else:
        print("  -> Many kinks, will need MoE/latent/strong corrector")
    
    # MVP-1 assessment
    print(f"\nMVP-1 (Velocity Field Fitting):")
    print(f"  Final error: {mvp1_result['final_error']:.6f}")
    if mvp1_result['final_error'] < 0.1:
        print("  -> Excellent: Error is low, flow model is very feasible")
    elif mvp1_result['final_error'] < 1.0:
        print("  -> Good: Error is moderate, flow model is feasible with anchor alignment")
    else:
        print("  -> Warning: Error is high, need anchor alignment + corrector")
    
    print("\n" + "=" * 80)
    print("Experiments Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

