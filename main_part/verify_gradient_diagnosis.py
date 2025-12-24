#!/usr/bin/env python
# coding: utf-8
"""
Gradient Diagnosis Verification Script

This script verifies the analysis about why different preferences lead to similar results:
1. Check gradient collinearity: cos(grad_V_cost, grad_V_carbon)
2. Check gradient directions under different preferences
3. Check projection impact: ||g_tan|| / ||g_obj||
4. Check restoration gate alpha distribution
5. Check cost-carbon correlation

Usage:
    conda activate pdp && python verify_gradient_diagnosis.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.stats import pearsonr
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from data_loader import load_all_data
from deepopf_ngt_loss import DeepOPFNGTLoss
from models import NetV


def compute_gradient_statistics(loss_fn, model, dataloader, device, preference):
    """
    Compute gradient statistics for a given preference.
    
    Returns:
        stats_dict: Dictionary containing all statistics
    """
    # Enable statistics collection
    loss_fn.params._collect_grad_stats = True
    loss_fn.params._grad_stats_list = []
    
    # Also collect objective values
    cost_values = []
    carbon_values = []
    
    model.train()  # Set to train mode to enable gradients
    for batch_idx, (PQd_batch, _) in enumerate(dataloader):
        if batch_idx >= 10:  # Use first 10 batches for diagnosis
            break
            
        PQd_batch = PQd_batch.to(device)
        batch_size = PQd_batch.shape[0]
        
        # Create preference tensor
        if preference is not None:
            lambda_cost, lambda_carbon = preference
            pref_tensor = torch.zeros((batch_size, 2), device=device)
            pref_tensor[:, 0] = lambda_cost
            pref_tensor[:, 1] = lambda_carbon
        else:
            pref_tensor = None
        
        # Forward pass to get predictions
        # Check if model supports preference conditioning
        if hasattr(model, 'use_pref_conditioning') and model.use_pref_conditioning:
            V_pred = model(PQd_batch, pref_tensor)
        else:
            V_pred = model(PQd_batch)
        
        # Compute loss (this will trigger backward and collect stats)
        loss, loss_dict = loss_fn(V_pred, PQd_batch, preference=pref_tensor, only_obj=False)
        
        # Collect objective values (detach to avoid gradient tracking)
        cost_values.append(loss_dict.get('cost_per_mean', 0.0))
        carbon_values.append(loss_dict.get('carbon_per_mean', 0.0))
        
        # Backward pass to compute gradients (but don't update)
        loss.backward()
        
        # Clear gradients for next iteration
        model.zero_grad()
    
    # Aggregate statistics
    all_stats = defaultdict(list)
    for batch_stats in loss_fn.params._grad_stats_list:
        for key, value in batch_stats.items():
            if value is not None:
                if isinstance(value, np.ndarray):
                    all_stats[key].extend(value.flatten())
                else:
                    all_stats[key].append(value)
    
    # Compute summary statistics
    summary = {}
    for key, values in all_stats.items():
        if len(values) > 0:
            arr = np.array(values)
            summary[f'{key}_mean'] = float(np.mean(arr))
            summary[f'{key}_std'] = float(np.std(arr))
            summary[f'{key}_min'] = float(np.min(arr))
            summary[f'{key}_max'] = float(np.max(arr))
            summary[f'{key}_median'] = float(np.median(arr))
            summary[f'{key}_p25'] = float(np.percentile(arr, 25))
            summary[f'{key}_p75'] = float(np.percentile(arr, 75))
    
    summary['cost_mean'] = float(np.mean(cost_values)) if cost_values else 0.0
    summary['carbon_mean'] = float(np.mean(carbon_values)) if carbon_values else 0.0
    
    # Store raw gradient directions for comparison
    if 'grad_final_normalized' in all_stats:
        # Take mean across samples for each preference
        grad_dirs = np.array(all_stats['grad_final_normalized'])
        if grad_dirs.size > 0:
            summary['grad_final_mean'] = np.mean(grad_dirs, axis=0)
    
    return summary, all_stats


def compare_preferences(loss_fn, model, dataloader, device, preferences):
    """
    Compare gradient statistics across different preferences.
    
    Args:
        preferences: List of (lambda_cost, lambda_carbon) tuples
    """
    results = {}
    grad_directions = {}
    
    print("=" * 80)
    print("Gradient Diagnosis: Comparing Different Preferences")
    print("=" * 80)
    
    for pref in preferences:
        lambda_cost, lambda_carbon = pref
        print(f"\n[Preference] lambda_cost={lambda_cost:.2f}, lambda_carbon={lambda_carbon:.2f}")
        print("-" * 80)
        
        # Reset statistics
        loss_fn.params._collect_grad_stats = False
        loss_fn.params._grad_stats_list = []
        
        summary, raw_stats = compute_gradient_statistics(
            loss_fn, model, dataloader, device, pref
        )
        results[pref] = summary
        if 'grad_final_mean' in summary:
            grad_directions[pref] = summary['grad_final_mean']
        
        # Print key statistics
        print(f"  Cosine similarity (cost vs carbon): {summary.get('cos_cost_carbon_mean', 0):.4f} ± {summary.get('cos_cost_carbon_std', 0):.4f}")
        print(f"  Projection ratio (||g_tan||/||g_obj||): {summary.get('proj_ratio_mean', 0):.4f} ± {summary.get('proj_ratio_std', 0):.4f}")
        print(f"  Restoration alpha: {summary.get('alpha_restore_mean', 0):.4f} ± {summary.get('alpha_restore_std', 0):.4f}")
        print(f"  Cost norm: {summary.get('norm_cost_mean', 0):.4f}")
        print(f"  Carbon norm: {summary.get('norm_carbon_mean', 0):.4f}")
        print(f"  Final cost: {summary.get('cost_mean', 0):.4f}")
        print(f"  Final carbon: {summary.get('carbon_mean', 0):.4f}")
    
    # Compare gradient directions between preferences
    print("\n" + "=" * 80)
    print("Gradient Direction Comparison")
    print("=" * 80)
    
    pref_list = list(preferences)
    for i in range(len(pref_list)):
        for j in range(i + 1, len(pref_list)):
            pref_i = pref_list[i]
            pref_j = pref_list[j]
            
            if pref_i in grad_directions and pref_j in grad_directions:
                g_i = grad_directions[pref_i]
                g_j = grad_directions[pref_j]
                
                # Compute cosine similarity
                dot = np.dot(g_i, g_j)
                norm_i = np.linalg.norm(g_i)
                norm_j = np.linalg.norm(g_j)
                cos_sim = dot / (norm_i * norm_j + 1e-12)
                
                print(f"  cos(g[{pref_i}], g[{pref_j}]) = {cos_sim:.4f}")
    
    return results, grad_directions


def analyze_cost_carbon_correlation(loss_fn, model, dataloader, device, num_samples=100):
    """
    Analyze correlation between cost and carbon across random samples.
    """
    print("\n" + "=" * 80)
    print("Cost-Carbon Correlation Analysis")
    print("=" * 80)
    
    cost_values = []
    carbon_values = []
    
    model.eval()
    with torch.no_grad():
        count = 0
        for PQd_batch, _ in dataloader:
            if count >= num_samples:
                break
                
            PQd_batch = PQd_batch.to(device)
            batch_size = PQd_batch.shape[0]
            
            # Use default preference (0.9, 0.1)
            pref_tensor = torch.tensor([[0.9, 0.1]], device=device).expand(batch_size, -1)
            
            # Check if model supports preference conditioning
            if hasattr(model, 'use_pref_conditioning') and model.use_pref_conditioning:
                V_pred = model(PQd_batch, pref_tensor)
            else:
                V_pred = model(PQd_batch)
            loss, loss_dict = loss_fn(V_pred, PQd_batch, preference=pref_tensor, only_obj=False)
            
            # Get per-sample values (approximate from batch mean)
            cost_values.append(loss_dict.get('cost_per_mean', 0.0))
            carbon_values.append(loss_dict.get('carbon_per_mean', 0.0))
            
            count += batch_size
    
    if len(cost_values) > 1:
        correlation, p_value = pearsonr(cost_values, carbon_values)
        print(f"  Pearson correlation: {correlation:.4f} (p-value: {p_value:.4e})")
        
        if correlation > 0.95:
            print("  [WARNING] Very high correlation (>0.95) suggests Pareto front is narrow!")
        elif correlation > 0.8:
            print("  [WARNING] High correlation (>0.8) may limit trade-off diversity")
        else:
            print("  [OK] Correlation is moderate, trade-off should be possible")
    else:
        print("  Not enough samples for correlation analysis")
    
    return cost_values, carbon_values


def plot_diagnosis_results(results, cost_values, carbon_values, output_dir='results'):
    """
    Plot diagnosis results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot cosine similarity distribution
    if any('cos_cost_carbon_mean' in results[pref] for pref in results):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Cosine similarity by preference
        ax = axes[0, 0]
        prefs = list(results.keys())
        cos_means = [results[pref].get('cos_cost_carbon_mean', 0) for pref in prefs]
        cos_stds = [results[pref].get('cos_cost_carbon_std', 0) for pref in prefs]
        pref_labels = [f"({p[0]:.1f},{p[1]:.1f})" for p in prefs]
        
        ax.bar(range(len(prefs)), cos_means, yerr=cos_stds, capsize=5)
        ax.set_xticks(range(len(prefs)))
        ax.set_xticklabels(pref_labels, rotation=45)
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Cost-Carbon Gradient Cosine Similarity')
        ax.axhline(y=0.95, color='r', linestyle='--', label='High correlation threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Projection ratio
        ax = axes[0, 1]
        proj_means = [results[pref].get('proj_ratio_mean', 0) for pref in prefs]
        proj_stds = [results[pref].get('proj_ratio_std', 0) for pref in prefs]
        
        ax.bar(range(len(prefs)), proj_means, yerr=proj_stds, capsize=5, color='orange')
        ax.set_xticks(range(len(prefs)))
        ax.set_xticklabels(pref_labels, rotation=45)
        ax.set_ylabel('Projection Ratio')
        ax.set_title('||g_tan|| / ||g_obj|| (Projection Impact)')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Restoration alpha
        ax = axes[1, 0]
        alpha_means = [results[pref].get('alpha_restore_mean', 0) for pref in prefs]
        alpha_stds = [results[pref].get('alpha_restore_std', 0) for pref in prefs]
        
        ax.bar(range(len(prefs)), alpha_means, yerr=alpha_stds, capsize=5, color='green')
        ax.set_xticks(range(len(prefs)))
        ax.set_xticklabels(pref_labels, rotation=45)
        ax.set_ylabel('Restoration Alpha')
        ax.set_title('Restoration Gate Value')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Cost-Carbon scatter
        ax = axes[1, 1]
        if len(cost_values) > 0 and len(carbon_values) > 0:
            ax.scatter(cost_values, carbon_values, alpha=0.6, s=20)
            ax.set_xlabel('Cost')
            ax.set_ylabel('Carbon (scaled)')
            ax.set_title('Cost vs Carbon (Pareto Front Shape)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_diagnosis.png'), dpi=150, bbox_inches='tight')
        print(f"\n[OK] Diagnosis plots saved to {output_dir}/gradient_diagnosis.png")
        plt.close()


def main():
    """Main function."""
    print("=" * 80)
    print("Gradient Diagnosis Verification Script")
    print("=" * 80)
    
    # Load configuration
    config = get_config()
    device = config.device
    
    # Load system data
    print("\n[1/4] Loading system data...")
    result = load_all_data(config)
    if isinstance(result, tuple):
        sys_data = result[0]  # First element is sys_data
    else:
        sys_data = result
    print(f"  [OK] Loaded system with {config.Nbus} buses")
    
    # Create loss function
    print("\n[2/4] Creating loss function...")
    loss_fn = DeepOPFNGTLoss(sys_data, config)
    loss_fn.cache_to_gpu(device)
    print("  [OK] Loss function created")
    
    # Create a simple model (or load existing)
    print("\n[3/4] Creating model...")
    output_dims = loss_fn.get_output_dims()
    bus_indices = loss_fn.get_bus_indices()
    input_dim = len(bus_indices['bus_Pd']) + len(bus_indices['bus_Qd'])
    
    model = NetV(
        input_channels=input_dim,
        output_channels=output_dims['total'],
        hidden_units=config.ngt_hidden_units,
        khidden=config.ngt_khidden,
        Vscale=torch.ones(output_dims['total']),
        Vbias=torch.zeros(output_dims['total'])
    )
    model.to(device)
    # Initialize model with small random weights for testing
    for param in model.parameters():
        if len(param.shape) >= 2:
            nn.init.xavier_uniform_(param, gain=0.1)
        else:
            nn.init.zeros_(param)
    print("  [OK] Model created (randomly initialized for testing)")
    
    # Create data loader
    print("\n[4/4] Creating data loader...")
    from data_loader import create_ngt_training_loader, load_ngt_training_data
    result = load_ngt_training_data(config, sys_data)
    if isinstance(result, tuple):
        ngt_data = result[0]
        sys_data = result[1] if len(result) > 1 else sys_data
    else:
        ngt_data = result
    dataloader = create_ngt_training_loader(ngt_data, config)
    print("  [OK] Data loader created")
    
    # Define preferences to test
    preferences = [
        (0.9, 0.1),  # Cost-focused
        (0.1, 0.9),  # Carbon-focused
        (0.5, 0.5),  # Balanced
    ]
    
    # Run diagnosis
    print("\n" + "=" * 80)
    print("Running Diagnosis...")
    print("=" * 80)
    
    results, grad_directions = compare_preferences(
        loss_fn, model, dataloader, device, preferences
    )
    
    # Analyze cost-carbon correlation
    cost_values, carbon_values = analyze_cost_carbon_correlation(
        loss_fn, model, dataloader, device, num_samples=100
    )
    
    # Plot results
    plot_diagnosis_results(results, cost_values, carbon_values)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Diagnosis Summary")
    print("=" * 80)
    
    # Check each hypothesis
    cos_sim_mean = np.mean([results[pref].get('cos_cost_carbon_mean', 0) for pref in preferences])
    print(f"\n[A] Gradient Collinearity:")
    print(f"    Mean cosine similarity: {cos_sim_mean:.4f}")
    if cos_sim_mean > 0.95:
        print("    [WARNING] HYPOTHESIS A CONFIRMED: Gradients are highly collinear!")
        print("       -> Cost and carbon gradients point in similar directions")
        print("       -> This explains why different preferences yield similar results")
    else:
        print("    [OK] Gradients are not highly collinear")
    
    proj_ratio_mean = np.mean([results[pref].get('proj_ratio_mean', 0) for pref in preferences])
    print(f"\n[C] Projection Impact:")
    print(f"    Mean projection ratio: {proj_ratio_mean:.4f}")
    if proj_ratio_mean < 0.3:
        print("    [WARNING] HYPOTHESIS C CONFIRMED: Projection removes most objective gradient!")
        print("       -> Most of the objective gradient is projected away")
        print("       -> This could explain why preferences don't differentiate")
    else:
        print("    [OK] Projection preserves reasonable amount of objective gradient")
    
    alpha_mean = np.mean([results[pref].get('alpha_restore_mean', 0) for pref in preferences])
    print(f"\n[C] Restoration Dominance:")
    print(f"    Mean restoration alpha: {alpha_mean:.4f}")
    if alpha_mean > 0.5:
        print("    [WARNING] HYPOTHESIS C CONFIRMED: Restoration often dominates!")
        print("       -> Restoration forces are frequently active")
        print("       -> This could mask preference differences")
    else:
        print("    [OK] Restoration is not dominating")
    
    if len(cost_values) > 1:
        correlation, _ = pearsonr(cost_values, carbon_values)
        print(f"\n[D] Pareto Front Shape:")
        print(f"    Cost-Carbon correlation: {correlation:.4f}")
        if correlation > 0.95:
            print("    [WARNING] HYPOTHESIS D CONFIRMED: Pareto front is very narrow!")
            print("       -> Cost and carbon are highly correlated")
            print("       -> Limited trade-off space available")
        else:
            print("    [OK] Pareto front has reasonable width")
    
    print("\n" + "=" * 80)
    print("Diagnosis Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

