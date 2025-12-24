#!/usr/bin/env python
# coding: utf-8
# Training Script for DeepOPF-V
# Extended to support multiple model types: VAE, Flow, Diffusion, GAN, etc.
# Author: Peng Yue
# Date: December 15th, 2025

import torch 
import torch.utils.data as Data
import numpy as np 
import time
import os
import sys 
import math
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add parent directory to path for flow_model imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from models import NetV
from data_loader import load_all_data, load_ngt_training_data 
from deepopf_ngt_loss import DeepOPFNGTLoss 
from unified_eval import post_process_like_evaluate_model, EvalContext, _ensure_1d_int, _as_numpy

 
# ============================================================================
# Helper functions for gradient descent training
# ============================================================================

def _compute_constraint_violation(loss_dict):
    """Compute total constraint violation from loss dictionary."""
    return (
        loss_dict.get('loss_Pgi_sum', 0.0) +
        loss_dict.get('loss_Qgi_sum', 0.0) +
        loss_dict.get('loss_Pdi_sum', 0.0) +
        loss_dict.get('loss_Qdi_sum', 0.0) +
        loss_dict.get('loss_Vi_sum', 0.0)
    )


def _compute_weighted_objective(loss_dict, lambda_cost, lambda_carbon):
    """Compute weighted objective function: lambda_cost * cost + lambda_carbon * carbon."""
    return (
        lambda_cost * loss_dict.get('loss_cost', 0.0) +
        lambda_carbon * loss_dict.get('loss_carbon', 0.0)
    )


def _clip_gradient(grad_V, grad_clip_norm):
    """Clip gradient per sample to specified norm."""
    if grad_clip_norm is None or grad_clip_norm <= 0:
        return grad_V
    grad_norm_per_sample = torch.norm(grad_V, dim=1, keepdim=True)
    clip_coef = grad_clip_norm / (grad_norm_per_sample + 1e-8)
    clip_coef = torch.clamp(clip_coef, max=1.0)
    return grad_V * clip_coef


def _analyze_objective_correlation(costs, carbons, results_dir):
    """
    Analyze correlation between cost and carbon objectives.
    
    Args:
        costs: List or array of cost values
        carbons: List or array of carbon values (unscaled)
        results_dir: Directory to save the plot
        
    Returns:
        dict with correlation statistics and plot path
    """
    costs = np.array(costs)
    carbons = np.array(carbons)
    
    # Remove any NaN or inf values
    valid_mask = np.isfinite(costs) & np.isfinite(carbons)
    costs = costs[valid_mask]
    carbons = carbons[valid_mask]
    
    if len(costs) < 2:
        return {
            'pearson_r': None,
            'pearson_p': None,
            'spearman_r': None,
            'spearman_p': None,
            'plot_path': None,
            'message': 'Insufficient valid data for correlation analysis'
        }
    
    # Compute correlations
    pearson_r, pearson_p = pearsonr(costs, carbons)
    spearman_r, spearman_p = spearmanr(costs, carbons)
    
    # Create visualization
    plot_path = os.path.join(results_dir, 'cost_carbon_correlation.png')
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(costs, carbons, alpha=0.5, s=20)
    plt.xlabel('Cost ($/h)', fontsize=12)
    plt.ylabel('Carbon Emission (tCO2/h)', fontsize=12)
    plt.title(f'Cost vs Carbon Correlation\nPearson r={pearson_r:.4f} (p={pearson_p:.2e}), Spearman r={spearman_r:.4f} (p={spearman_p:.2e})', 
              fontsize=13)
    plt.grid(True, alpha=0.3)
    
    # Add correlation line
    z = np.polyfit(costs, carbons, 1)
    p = np.poly1d(z)
    plt.plot(costs, p(costs), "r--", alpha=0.8, linewidth=2, label=f'Linear fit: y={z[0]:.4f}x+{z[1]:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 60)
    print("Objective Correlation Analysis")
    print("=" * 60)
    print(f"Number of samples: {len(costs)}")
    print(f"\nPearson correlation: r = {pearson_r:.4f}, p-value = {pearson_p:.2e}")
    print(f"Spearman correlation: r = {spearman_r:.4f}, p-value = {spearman_p:.2e}")
    
    # Interpretation
    if abs(pearson_r) > 0.9:
        corr_level = "very high"
        interpretation = "Cost and carbon are highly correlated, which may limit Pareto frontier coverage"
    elif abs(pearson_r) > 0.7:
        corr_level = "high"
        interpretation = "Cost and carbon are moderately correlated, Pareto frontier may be limited"
    elif abs(pearson_r) > 0.5:
        corr_level = "moderate"
        interpretation = "Cost and carbon have moderate correlation, some Pareto trade-offs exist"
    elif abs(pearson_r) > 0.3:
        corr_level = "low"
        interpretation = "Cost and carbon have low correlation, good potential for Pareto frontier"
    else:
        corr_level = "very low"
        interpretation = "Cost and carbon are nearly independent, excellent Pareto frontier potential"
    
    print(f"\nCorrelation level: {corr_level}")
    print(f"Interpretation: {interpretation}")
    print(f"\nPlot saved to: {plot_path}")
    print("=" * 60)
    
    return {
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'n_samples': len(costs),
        'plot_path': plot_path,
        'correlation_level': corr_level,
        'interpretation': interpretation,
        'cost_mean': float(np.mean(costs)),
        'cost_std': float(np.std(costs)),
        'carbon_mean': float(np.mean(carbons)),
        'carbon_std': float(np.std(carbons)),
    }


def analyze_fixed_load_from_training_data(config, sys_data, ngt_data):
    """
    Analyze correlation between cost and carbon from REAL training data (ground truth solutions).
    
    This function uses the actual training data (ground truth voltage solutions) to compute
    cost and carbon, then groups samples by similar total load and analyzes correlation
    within each group.
    
    Args:
        config: Configuration object
        sys_data: PowerSystemData object
        ngt_data: NGT data dictionary
        
    Returns:
        dict with analysis results
    """
    print("\n" + "=" * 60)
    print("Fixed Load Correlation Analysis from Training Data")
    print("=" * 60)
    
    from utils import get_genload, get_Pgcost, get_gci_for_generators
    
    # Get training data
    x_train = ngt_data['x_train']  # Load data
    num_Pd = len(ngt_data['bus_Pd'])
    
    # Get ground truth voltage from training data
    # yvm_train and yva_train are in NGT format (non-ZIB only)
    yvm_train_ngt = ngt_data['y_train'][:, num_Pd + len(ngt_data['bus_Qd']):]  # Vm part
    yva_train_ngt = ngt_data['y_train'][:, num_Pd + len(ngt_data['bus_Qd']):-len(ngt_data['bus_Pnet_all'])]  # Va part (need to check exact indices)
    
    # Actually, let's use the full voltage from sys_data if available
    # Or reconstruct from NGT format
    bus_slack = int(sys_data.bus_slack)
    bus_Pnet_all = _ensure_1d_int(ngt_data['bus_Pnet_all'])
    bus_Pnet_noslack_all = bus_Pnet_all[bus_Pnet_all != bus_slack]
    
    # Reconstruct full voltage from NGT format
    # NGT format: [Va_nonZIB_noslack, Vm_nonZIB] (both in physical space)
    y_train_ngt = ngt_data['y_train'].numpy()
    NPred_Va = ngt_data['NPred_Va']
    NPred_Vm = ngt_data['NPred_Vm']
    
    # Split NGT format: [Va_nonZIB_noslack, Vm_nonZIB]
    Va_train_ngt = y_train_ngt[:, :NPred_Va]  # [N, NPred_Va] - Va for non-ZIB, no slack
    Vm_train_ngt = y_train_ngt[:, NPred_Va:NPred_Va + NPred_Vm]  # [N, NPred_Vm] - Vm for non-ZIB
    
    # Reconstruct full voltage
    N_samples = y_train_ngt.shape[0]
    Vm_full = np.zeros((N_samples, config.Nbus))
    Va_full = np.zeros((N_samples, config.Nbus))
    
    # Insert Va for non-slack, non-ZIB buses
    # bus_Pnet_noslack_all excludes slack but includes all non-ZIB buses
    Va_full[:, bus_Pnet_noslack_all] = Va_train_ngt
    # Va for slack is 0 (already initialized)
    
    # Insert Vm for non-ZIB buses (bus_Pnet_all includes slack)
    Vm_full[:, bus_Pnet_all] = Vm_train_ngt
    
    # Recover ZIB voltages if needed
    if ngt_data.get('NZIB', 0) > 0 and ngt_data.get('param_ZIMV') is not None:
        # Convert non-ZIB to complex (need to include slack in Va)
        # For non-ZIB buses including slack: Va[slack]=0, Va[others]=Va_train_ngt
        Va_nonZIB_with_slack = Va_full[:, bus_Pnet_all].copy()
        idx_slack_in_Pnet = np.where(bus_Pnet_all == bus_slack)[0]
        if len(idx_slack_in_Pnet) > 0:
            Va_nonZIB_with_slack[:, idx_slack_in_Pnet[0]] = 0.0
        
        Vx = Vm_full[:, bus_Pnet_all] * np.exp(1j * Va_nonZIB_with_slack)
        # Recover ZIB
        Vy = np.dot(ngt_data['param_ZIMV'], Vx.T).T
        Vm_full[:, ngt_data['bus_ZIB_all']] = np.abs(Vy)
        Va_full[:, ngt_data['bus_ZIB_all']] = np.angle(Vy)
    
    # Calculate total load for each sample
    train_x_np = x_train.numpy()
    total_loads = np.sum(train_x_np[:, :num_Pd], axis=1)
    
    # Build full load arrays
    Pd_full = np.zeros((N_samples, config.Nbus))
    Qd_full = np.zeros((N_samples, config.Nbus))
    Pd_full[:, ngt_data['bus_Pd']] = train_x_np[:, :num_Pd]
    Qd_full[:, ngt_data['bus_Qd']] = train_x_np[:, num_Pd:]
    
    # Compute generation from voltage
    V_complex = Vm_full * np.exp(1j * Va_full)
    bus_Pg = _ensure_1d_int(sys_data.bus_Pg)
    bus_Qg = _ensure_1d_int(sys_data.bus_Qg)
    
    # get_genload returns Pg and Qg already for generator buses only
    Pg, Qg, Pd_computed, Qd_computed = get_genload(
        V_complex, Pd_full, Qd_full, bus_Pg, bus_Qg, sys_data.Ybus
    )
    
    # Compute cost and carbon
    baseMVA = float(sys_data.baseMVA.item() if hasattr(sys_data.baseMVA, 'item') else sys_data.baseMVA)
    costs = get_Pgcost(Pg, sys_data.idxPg, sys_data.gencost, baseMVA)
    
    # Compute carbon
    gci_values = get_gci_for_generators(sys_data)
    gci_for_Pg = gci_values[sys_data.idxPg]
    Pg_MVA = Pg * baseMVA
    carbons = np.sum(Pg_MVA * gci_for_Pg.reshape(1, -1), axis=1)
    
    # Compute total generation and losses
    total_gen = np.sum(Pg_MVA, axis=1)
    total_load = np.sum(total_loads * baseMVA)
    # Note: losses = total_gen - total_load (approximate, ignoring reactive power)
    
    # Group samples by similar total load (use percentiles or bins for better grouping)
    # Calculate load range
    load_min = np.min(total_loads)
    load_max = np.max(total_loads)
    load_range = load_max - load_min
    
    # Use adaptive tolerance: larger tolerance to group more samples
    # Try different tolerance levels
    tolerance_levels = [0.005, 0.01, 0.02, 0.05]  # 0.5%, 1%, 2%, 5%
    
    best_tolerance = None
    best_groups = []
    for tolerance_ratio in tolerance_levels:
        tolerance = tolerance_ratio * load_range  # Absolute tolerance
        # Group samples using clustering approach
        sorted_loads_idx = np.argsort(total_loads)
        sorted_loads = total_loads[sorted_loads_idx]
        
        load_groups = []
        current_group = [sorted_loads_idx[0]]
        current_load = sorted_loads[0]
        
        for i in range(1, len(sorted_loads)):
            if abs(sorted_loads[i] - current_load) <= tolerance:
                # Add to current group
                current_group.append(sorted_loads_idx[i])
            else:
                # Start new group
                if len(current_group) >= 3:
                    load_groups.append((np.mean(total_loads[current_group]), current_group))
                current_group = [sorted_loads_idx[i]]
                current_load = sorted_loads[i]
        
        # Don't forget last group
        if len(current_group) >= 3:
            load_groups.append((np.mean(total_loads[current_group]), current_group))
        
        if len(load_groups) > len(best_groups):
            best_groups = load_groups
            best_tolerance = tolerance_ratio
    
    load_groups = best_groups
    print(f"Found {len(load_groups)} load groups with >=3 samples (tolerance: {best_tolerance*100:.1f}% of range)")
    print(f"Total samples: {N_samples}")
    print(f"Load range: [{load_min:.3f}, {load_max:.3f}] p.u.")
    
    if len(load_groups) == 0:
        print("Warning: No load groups found with >=3 samples. Trying with >=2 samples...")
        # Fallback: use >=2 samples
        for tolerance_ratio in tolerance_levels:
            tolerance = tolerance_ratio * load_range
            sorted_loads_idx = np.argsort(total_loads)
            sorted_loads = total_loads[sorted_loads_idx]
            
            load_groups = []
            current_group = [sorted_loads_idx[0]]
            current_load = sorted_loads[0]
            
            for i in range(1, len(sorted_loads)):
                if abs(sorted_loads[i] - current_load) <= tolerance:
                    current_group.append(sorted_loads_idx[i])
                else:
                    if len(current_group) >= 2:
                        load_groups.append((np.mean(total_loads[current_group]), current_group))
                    current_group = [sorted_loads_idx[i]]
                    current_load = sorted_loads[i]
            
            if len(current_group) >= 2:
                load_groups.append((np.mean(total_loads[current_group]), current_group))
            
            if len(load_groups) > 0:
                break
    
    # Analyze correlation within each group
    group_correlations = []
    group_costs = []
    group_carbons = []
    
    # Additional analysis: Check if total generation power is similar within groups
    group_total_gen = []
    
    for load_val, indices in load_groups:
        group_costs_array = costs[indices]
        group_carbons_array = carbons[indices]
        
        # Compute total generation for this group
        group_Pg = Pg[indices]
        group_total_gen_array = np.sum(group_Pg, axis=1) * baseMVA
        
        if len(group_costs_array) >= 3:
            # Compute correlation
            pearson_r, pearson_p = pearsonr(group_costs_array, group_carbons_array)
            spearman_r, spearman_p = spearmanr(group_costs_array, group_carbons_array)
            
            # Check correlation between total generation and cost/carbon
            corr_gen_cost = pearsonr(group_total_gen_array, group_costs_array)[0]
            corr_gen_carbon = pearsonr(group_total_gen_array, group_carbons_array)[0]
            
            # Compute losses for this group
            # Total load should be sum of Pd for all buses
            group_loads = np.sum(Pd_full[indices], axis=1) * baseMVA
            group_losses = group_total_gen_array - group_loads
            corr_loss_cost = pearsonr(group_losses, group_costs_array)[0] if len(group_losses) > 1 else 0.0
            corr_loss_carbon = pearsonr(group_losses, group_carbons_array)[0] if len(group_losses) > 1 else 0.0
            
            # Check if slack generator explains the variation
            bus_slack = int(sys_data.bus_slack)
            # Find if slack bus has a generator
            slack_is_gen = bus_slack in bus_Pg
            if slack_is_gen:
                slack_gen_idx = np.where(bus_Pg == bus_slack)[0][0]
                slack_gen_power = Pg[indices, slack_gen_idx] * baseMVA
                corr_slack_cost = pearsonr(slack_gen_power, group_costs_array)[0] if len(slack_gen_power) > 1 else 0.0
                corr_slack_carbon = pearsonr(slack_gen_power, group_carbons_array)[0] if len(slack_gen_power) > 1 else 0.0
                slack_power_std = np.std(slack_gen_power)
            else:
                corr_slack_cost = corr_slack_carbon = 0.0
                slack_power_std = 0.0
            
            # Check variation in individual generator powers
            group_Pg_variation = np.std(Pg[indices], axis=0) * baseMVA
            max_var_gen_idx = np.argmax(group_Pg_variation)
            max_var_gen_power = Pg[indices, max_var_gen_idx] * baseMVA
            corr_max_gen_cost = pearsonr(max_var_gen_power, group_costs_array)[0] if len(max_var_gen_power) > 1 else 0.0
            corr_max_gen_carbon = pearsonr(max_var_gen_power, group_carbons_array)[0] if len(max_var_gen_power) > 1 else 0.0
            
            group_correlations.append({
                'load': float(load_val),
                'n_samples': len(indices),
                'pearson_r': float(pearson_r),
                'pearson_p': float(pearson_p),
                'spearman_r': float(spearman_r),
                'spearman_p': float(spearman_p),
                'cost_mean': float(np.mean(group_costs_array)),
                'cost_std': float(np.std(group_costs_array)),
                'carbon_mean': float(np.mean(group_carbons_array)),
                'carbon_std': float(np.std(group_carbons_array)),
                'total_gen_mean': float(np.mean(group_total_gen_array)),
                'total_gen_std': float(np.std(group_total_gen_array)),
                'corr_gen_cost': float(corr_gen_cost),
                'corr_gen_carbon': float(corr_gen_carbon),
                'loss_mean': float(np.mean(group_losses)),
                'loss_std': float(np.std(group_losses)),
                'corr_loss_cost': float(corr_loss_cost),
                'corr_loss_carbon': float(corr_loss_carbon),
                'slack_is_gen': slack_is_gen,
                'corr_slack_cost': float(corr_slack_cost) if slack_is_gen else None,
                'corr_slack_carbon': float(corr_slack_carbon) if slack_is_gen else None,
                'slack_power_std': float(slack_power_std) if slack_is_gen else None,
                'max_var_gen_std': float(group_Pg_variation[max_var_gen_idx]),
                'corr_max_gen_cost': float(corr_max_gen_cost),
                'corr_max_gen_carbon': float(corr_max_gen_carbon),
            })
            
            group_costs.append(group_costs_array)
            group_carbons.append(group_carbons_array)
            group_total_gen.append(group_total_gen_array)
    
    # Analyze results
    if len(group_correlations) > 0:
        pearson_rs = [g['pearson_r'] for g in group_correlations]
        avg_correlation = np.mean(pearson_rs)
        std_correlation = np.std(pearson_rs)
        min_correlation = np.min(pearson_rs)
        max_correlation = np.max(pearson_rs)
        
        print(f"\nAverage Pearson correlation within load groups: {avg_correlation:.4f} ± {std_correlation:.4f}")
        print(f"Correlation range: [{min_correlation:.4f}, {max_correlation:.4f}]")
        print(f"Number of groups analyzed: {len(group_correlations)}")
        
        # Check if total generation varies within groups
        avg_gen_std = np.mean([g['total_gen_std'] for g in group_correlations])
        avg_corr_gen_cost = np.mean([g['corr_gen_cost'] for g in group_correlations])
        avg_corr_gen_carbon = np.mean([g['corr_gen_carbon'] for g in group_correlations])
        
        # Analyze losses
        avg_loss_std = np.mean([g['loss_std'] for g in group_correlations])
        avg_corr_loss_cost = np.mean([g['corr_loss_cost'] for g in group_correlations])
        avg_corr_loss_carbon = np.mean([g['corr_loss_carbon'] for g in group_correlations])
        
        print(f"\nWithin-group analysis:")
        print(f"  Average std of total generation: {avg_gen_std:.2f} MW")
        print(f"  Average std of losses: {avg_loss_std:.2f} MW")
        print(f"  Average correlation (total_gen vs cost): {avg_corr_gen_cost:.4f}")
        print(f"  Average correlation (total_gen vs carbon): {avg_corr_gen_carbon:.4f}")
        print(f"  Average correlation (loss vs cost): {avg_corr_loss_cost:.4f}")
        print(f"  Average correlation (loss vs carbon): {avg_corr_loss_carbon:.4f}")
        print(f"  Average correlation (cost vs carbon): {avg_correlation:.4f}")
        
        # Analyze slack generator impact
        groups_with_slack = [g for g in group_correlations if g.get('slack_is_gen', False)]
        if len(groups_with_slack) > 0:
            avg_slack_std = np.mean([g['slack_power_std'] for g in groups_with_slack])
            avg_corr_slack_cost = np.mean([g['corr_slack_cost'] for g in groups_with_slack])
            avg_corr_slack_carbon = np.mean([g['corr_slack_carbon'] for g in groups_with_slack])
            print(f"\nSlack generator analysis ({len(groups_with_slack)} groups with slack gen):")
            print(f"  Average std of slack generator power: {avg_slack_std:.2f} MW")
            print(f"  Average correlation (slack_gen vs cost): {avg_corr_slack_cost:.4f}")
            print(f"  Average correlation (slack_gen vs carbon): {avg_corr_slack_carbon:.4f}")
        
        avg_max_var_std = np.mean([g['max_var_gen_std'] for g in group_correlations])
        avg_corr_max_gen_cost = np.mean([g['corr_max_gen_cost'] for g in group_correlations])
        avg_corr_max_gen_carbon = np.mean([g['corr_max_gen_carbon'] for g in group_correlations])
        print(f"\nMost variable generator analysis:")
        print(f"  Average std of max-variable generator: {avg_max_var_std:.2f} MW")
        print(f"  Average correlation (max_var_gen vs cost): {avg_corr_max_gen_cost:.4f}")
        print(f"  Average correlation (max_var_gen vs carbon): {avg_corr_max_gen_carbon:.4f}")
        
        print(f"\n[KEY INSIGHT]")
        if avg_loss_std < 1.0 and avg_gen_std > 5.0:
            print(f"  Losses vary little ({avg_loss_std:.2f} MW), but total generation varies much ({avg_gen_std:.2f} MW).")
            print(f"  This suggests that even with fixed load, different load distributions lead to")
            print(f"  different optimal power allocations, where slack or dominant generators compensate.")
            print(f"  When one generator increases power, it increases both cost and carbon,")
            print(f"  creating positive correlation despite GCI-c1 reverse design.")
        else:
            print(f"  Even with fixed load, total generation varies due to different network losses.")
            print(f"  Total Gen = Load + Losses. When losses increase, all generators must produce more,")
            print(f"  leading to higher cost AND higher carbon, creating positive correlation.")
        
        # Overall correlation (all samples)
        overall_pearson, overall_p = pearsonr(costs, carbons)
        total_gen_all = np.sum(Pg, axis=1) * baseMVA
        overall_gen_cost_r = pearsonr(total_gen_all, costs)[0]
        overall_gen_carbon_r = pearsonr(total_gen_all, carbons)[0]
        
        print(f"\nOverall correlation (all samples):")
        print(f"  cost vs carbon: {overall_pearson:.4f} (p={overall_p:.2e})")
        print(f"  total_gen vs cost: {overall_gen_cost_r:.4f}")
        print(f"  total_gen vs carbon: {overall_gen_carbon_r:.4f}")
        
        # Visualize
        plot_path = os.path.join(config.results_dir, 'fixed_load_correlation_training_data.png')
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: Correlation by load group
        loads = [g['load'] for g in group_correlations]
        corrs = [g['pearson_r'] for g in group_correlations]
        axes[0, 0].bar(range(len(loads)), corrs, alpha=0.7)
        axes[0, 0].axhline(y=avg_correlation, color='r', linestyle='--', label=f'Average: {avg_correlation:.4f}')
        axes[0, 0].axhline(y=overall_pearson, color='g', linestyle='--', label=f'Overall: {overall_pearson:.4f}')
        axes[0, 0].set_xlabel('Load Group Index', fontsize=12)
        axes[0, 0].set_ylabel('Pearson Correlation', fontsize=12)
        axes[0, 0].set_title('Cost-Carbon Correlation by Load Group (Training Data)', fontsize=13)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Overall scatter plot with color by total generation
        total_gen_all = np.sum(Pg, axis=1) * baseMVA
        scatter = axes[0, 1].scatter(costs, carbons, c=total_gen_all, alpha=0.5, s=15, cmap='viridis')
        axes[0, 1].set_xlabel('Cost ($/h)', fontsize=12)
        axes[0, 1].set_ylabel('Carbon Emission (tCO2/h)', fontsize=12)
        axes[0, 1].set_title(f'All Training Samples (r={overall_pearson:.4f})', fontsize=13)
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 1], label='Total Gen (MW)')
        z = np.polyfit(costs, carbons, 1)
        p = np.poly1d(z)
        axes[0, 1].plot(costs, p(costs), "r--", alpha=0.8, linewidth=2)
        
        # Plot 3: Scatter plot for a few representative groups
        n_show = min(2, len(group_costs))  # Only show 2 groups in 2x2 layout
        for i in range(n_show):
            costs_rep = group_costs[i]
            carbons_rep = group_carbons[i]
            load_rep = group_correlations[i]['load']
            corr_rep = group_correlations[i]['pearson_r']
            
            axes[1, i].scatter(costs_rep, carbons_rep, alpha=0.6, s=30)
            axes[1, i].set_xlabel('Cost ($/h)', fontsize=10)
            axes[1, i].set_ylabel('Carbon Emission (tCO2/h)', fontsize=10)
            axes[1, i].set_title(f'Load={load_rep:.2f}, r={corr_rep:.4f}', fontsize=11)
            axes[1, i].grid(True, alpha=0.3)
            z = np.polyfit(costs_rep, carbons_rep, 1)
            p = np.poly1d(z)
            axes[1, i].plot(costs_rep, p(costs_rep), "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nPlot saved to: {plot_path}")
        print("=" * 60)
        
        return {
            'group_correlations': group_correlations,
            'average_pearson': float(avg_correlation),
            'std_pearson': float(std_correlation),
            'min_pearson': float(min_correlation),
            'max_pearson': float(max_correlation),
            'overall_pearson': float(overall_pearson),
            'overall_p': float(overall_p),
            'plot_path': plot_path,
        }
    else:
        print("No valid load groups found for analysis")
        return None


def analyze_fixed_load_correlation(config, sys_data, ngt_data, loss_fn, device, num_samples_per_load=50):
    """
    Analyze correlation between cost and carbon for samples with similar total load.
    
    This function tests the hypothesis that when total load is fixed, different power
    allocation strategies may create trade-off space between cost and carbon.
    
    Args:
        config: Configuration object
        sys_data: PowerSystemData object
        ngt_data: NGT data dictionary
        loss_fn: DeepOPFNGTLoss instance
        device: Device
        num_samples_per_load: Number of samples to analyze per load group
        
    Returns:
        dict with analysis results
    """
    print("\n" + "=" * 60)
    print("Fixed Load Correlation Analysis")
    print("=" * 60)
    
    # Get training data
    x_train = ngt_data['x_train'].to(device)
    num_Pd = len(ngt_data['bus_Pd'])
    
    # Calculate total load for each sample
    train_x_np = x_train.detach().cpu().numpy()
    total_loads = np.sum(train_x_np[:, :num_Pd], axis=1)
    
    # Group samples by similar total load (within 1% tolerance)
    unique_loads = np.unique(total_loads)
    tolerance = 0.01  # 1% tolerance
    
    # Find load groups
    load_groups = []
    for load in unique_loads:
        group_indices = np.where(np.abs(total_loads - load) < tolerance * abs(load))[0]
        if len(group_indices) >= 3:  # Need at least 3 samples for correlation
            load_groups.append((load, group_indices))
    
    print(f"Found {len(load_groups)} load groups with >=3 samples")
    
    # Analyze correlation within each group
    group_correlations = []
    group_costs = []
    group_carbons = []
    
    # Load VAE models for generating voltage anchors
    from models import create_model
    vae_vm = create_model('vae', ngt_data['input_dim'], config.Nbus, config, is_vm=True).to(device)
    vae_va = create_model('vae', ngt_data['input_dim'], config.Nbus - 1, config, is_vm=False).to(device)
    vae_vm.load_state_dict(torch.load(config.pretrain_model_path_vm, map_location=device, weights_only=True), strict=True)
    vae_va.load_state_dict(torch.load(config.pretrain_model_path_va, map_location=device, weights_only=True), strict=True)
    vae_vm.eval()
    vae_va.eval()
    
    for load_val, indices in load_groups[:10]:  # Analyze first 10 groups
        if len(indices) > num_samples_per_load:
            indices = indices[:num_samples_per_load]
        
        # Get load data for this group
        group_x = x_train[indices]
        
        # Get initial V_anchor for this group using VAE
        group_costs_list = []
        group_carbons_list = []
        
        with torch.no_grad():
            # Generate V_anchor
            Vscale = ngt_data['Vscale'].to(device)
            Vbias = ngt_data['Vbias'].to(device)
            bus_slack = int(sys_data.bus_slack)
            
            # Use helper function to compute V_anchor
            # Create temporary ngt_data for this group
            temp_ngt_data = {**ngt_data, 'x_train': group_x}
            V_anchor_group = _precompute_V_anchor_physical_all(
                config=config,
                sys_data=sys_data,
                ngt_data=temp_ngt_data,
                device=device,
                vae_vm=vae_vm,
                vae_va=vae_va,
                Vscale=Vscale,
                Vbias=Vbias,
                bus_slack=bus_slack,
            )
            
            # Compute loss to get cost and carbon for each sample
            for i in range(len(group_x)):
                V_single = V_anchor_group[i:i+1].requires_grad_(False)
                x_single = group_x[i:i+1]
                _, loss_dict = loss_fn(V_single, x_single, only_obj=False)
                
                # Get per-sample values
                cost_per = loss_dict.get('cost_per_sample')
                carbon_per = loss_dict.get('carbon_per_sample')
                if cost_per is not None and carbon_per is not None:
                    if isinstance(cost_per, np.ndarray) and len(cost_per) > 0:
                        group_costs_list.append(float(cost_per[0]))
                    else:
                        group_costs_list.append(loss_dict.get('cost_per_mean', 0.0))
                    if isinstance(carbon_per, np.ndarray) and len(carbon_per) > 0:
                        group_carbons_list.append(float(carbon_per[0]))
                    else:
                        group_carbons_list.append(loss_dict.get('carbon_per_mean', 0.0))
        
        if len(group_costs_list) >= 3:
            costs_array = np.array(group_costs_list)
            carbons_array = np.array(group_carbons_list)
            
            # Compute correlation
            pearson_r, pearson_p = pearsonr(costs_array, carbons_array)
            spearman_r, spearman_p = spearmanr(costs_array, carbons_array)
            
            group_correlations.append({
                'load': float(load_val),
                'n_samples': len(group_costs_list),
                'pearson_r': float(pearson_r),
                'pearson_p': float(pearson_p),
                'spearman_r': float(spearman_r),
                'spearman_p': float(spearman_p),
                'cost_mean': float(np.mean(costs_array)),
                'cost_std': float(np.std(costs_array)),
                'carbon_mean': float(np.mean(carbons_array)),
                'carbon_std': float(np.std(carbons_array)),
            })
            
            group_costs.append(costs_array)
            group_carbons.append(carbons_array)
    
    # Analyze results
    if len(group_correlations) > 0:
        avg_correlation = np.mean([g['pearson_r'] for g in group_correlations])
        print(f"\nAverage Pearson correlation within load groups: {avg_correlation:.4f}")
        print(f"Number of groups analyzed: {len(group_correlations)}")
        
        # Compare with overall correlation
        if len(group_costs) > 0 and len(group_carbons) > 0:
            all_costs_fixed = np.concatenate(group_costs)
            all_carbons_fixed = np.concatenate(group_carbons)
            overall_pearson, _ = pearsonr(all_costs_fixed, all_carbons_fixed)
            print(f"Overall correlation (all fixed-load samples): {overall_pearson:.4f}")
        
        # Visualize
        plot_path = os.path.join(config.results_dir, 'fixed_load_correlation.png')
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Correlation by load group
        loads = [g['load'] for g in group_correlations]
        corrs = [g['pearson_r'] for g in group_correlations]
        axes[0].bar(range(len(loads)), corrs, alpha=0.7)
        axes[0].axhline(y=avg_correlation, color='r', linestyle='--', label=f'Average: {avg_correlation:.4f}')
        axes[0].set_xlabel('Load Group Index', fontsize=12)
        axes[0].set_ylabel('Pearson Correlation', fontsize=12)
        axes[0].set_title('Cost-Carbon Correlation by Load Group', fontsize=13)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot for one representative group
        if len(group_costs) > 0:
            # Use the group with most samples
            largest_group_idx = np.argmax([len(c) for c in group_costs])
            costs_rep = group_costs[largest_group_idx]
            carbons_rep = group_carbons[largest_group_idx]
            load_rep = group_correlations[largest_group_idx]['load']
            corr_rep = group_correlations[largest_group_idx]['pearson_r']
            
            axes[1].scatter(costs_rep, carbons_rep, alpha=0.6, s=30)
            axes[1].set_xlabel('Cost ($/h)', fontsize=12)
            axes[1].set_ylabel('Carbon Emission (tCO2/h)', fontsize=12)
            axes[1].set_title(f'Fixed Load Group (Load={load_rep:.3f}, r={corr_rep:.4f})', fontsize=13)
            axes[1].grid(True, alpha=0.3)
            
            # Add linear fit
            z = np.polyfit(costs_rep, carbons_rep, 1)
            p = np.poly1d(z)
            axes[1].plot(costs_rep, p(costs_rep), "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nPlot saved to: {plot_path}")
        print("=" * 60)
        
        return {
            'group_correlations': group_correlations,
            'average_correlation': float(avg_correlation),
            'plot_path': plot_path,
        }
    else:
        print("No valid load groups found for analysis")
        return None

from models import create_model

def train_unsupervised_ngt_gradient_descent(
    config, sys_data, device=None, 
    num_iterations=50,
    learning_rate=1e-5,  # Default to much smaller learning rate
    tb_logger=None, 
    grad_clip_norm=1.0,  # Gradient clipping norm
    use_post_processing=True,  # Enable post-processing after each gradient update
):
    """
    Validation function: Direct gradient descent on V_anchor.
    
    This function tests whether we can improve V_anchor by:
    1. Starting from VAE-generated V_anchor (physical space)
    2. Computing loss gradient w.r.t. V_anchor (ONLY OBJECTIVE, NO CONSTRAINTS - only_obj=True) 
    4. Updating V_anchor directly using projected gradient
    5. Iterating to find better solutions (optimizing objective while maintaining constraints via post-processing)
    
    NOTE: This function uses only_obj=True, meaning it only optimizes the objective function
    (cost + carbon) and ignores constraint violations in the loss. Constraints are maintained
    through post-processing after each gradient update.
    
    Args:
        config: Configuration object
        sys_data: PowerSystemData object (optional, will load if None)
        device: Device (optional, uses config.device if None) 
        num_iterations: Number of gradient descent iterations
        learning_rate: Learning rate for gradient updates (DEFAULT: 1e-5, recommended range: 1e-5 to 1e-6)
        tb_logger: TensorBoardLogger instance for logging (optional) 
        lambda_cor: Drift correction gain (default: 5.0)
        grad_clip_norm: Gradient clipping norm (default: 1.0) 
    """
    
    if device is None:
        device = config.device
    
    # ========================================================================
    # Initialization
    # ========================================================================
    # Load system data and compute BRANFT (needed for post-processing) 
    branch_np = sys_data.branch if isinstance(sys_data.branch, np.ndarray) else sys_data.branch.numpy()
    BRANFT = branch_np[:, 0:2] - 1  # Convert to 0-indexed
    # Load NGT training data
    ngt_data, sys_data = load_ngt_training_data(config, sys_data)
    Vscale = ngt_data['Vscale'].to(device)
    Vbias = ngt_data['Vbias'].to(device)
    
    bus_slack = int(sys_data.bus_slack)
    
    # Load and freeze VAE models for anchor generation
    def _load_frozen_vae_or_raise():
        """Load and freeze VAE models for generating voltage anchors."""
        vae_vm_path = config.pretrain_model_path_vm
        vae_va_path = config.pretrain_model_path_va
        input_dim = ngt_data['input_dim']
        
        vae_vm = create_model('vae', input_dim, config.Nbus, config, is_vm=True).to(device)
        vae_va = create_model('vae', input_dim, config.Nbus - 1, config, is_vm=False).to(device)
        vae_vm.load_state_dict(torch.load(vae_vm_path, map_location=device, weights_only=True), strict=True)
        vae_va.load_state_dict(torch.load(vae_va_path, map_location=device, weights_only=True), strict=True)
        
        vae_vm.eval()
        vae_va.eval()
        for p in vae_vm.parameters():
            p.requires_grad = False
        for p in vae_va.parameters():
            p.requires_grad = False
        return vae_vm, vae_va
    
    vae_vm, vae_va = _load_frozen_vae_or_raise()
    
    # Setup loss function 
    loss_fn = DeepOPFNGTLoss(sys_data, config)
    loss_fn.cache_to_gpu(device) 
    
    # Create training DataLoader with indices
    class IndexedTensorDataset(Data.Dataset):
        """Dataset that returns data with indices for batch tracking."""
        def __init__(self, *tensors):
            assert all(t.size(0) == tensors[0].size(0) for t in tensors)
            self.tensors = tensors
        
        def __getitem__(self, index):
            return tuple(t[index] for t in self.tensors) + (index,)
        
        def __len__(self):
            return self.tensors[0].size(0)
    
    indexed_dataset = IndexedTensorDataset(ngt_data['x_train'], ngt_data['y_train'])
    training_loader = Data.DataLoader(indexed_dataset, batch_size=config.ngt_batch_size, shuffle=False)
    
    # Precompute initial V_anchor (physical space) for all training samples
    print("\n[Gradient Descent] Computing V_anchor from VAE...")
    V_anchor_all = _precompute_V_anchor_physical_all(
        config=config,
        sys_data=sys_data,
        ngt_data=ngt_data,
        device=device,
        vae_vm=vae_vm,
        vae_va=vae_va,
        Vscale=Vscale,
        Vbias=Vbias,
        bus_slack=bus_slack,
    )
    
    # Initialize loss history tracking
    loss_history = {
        'total': [],
        'cost': [],
        'carbon': [],
        'constraint_violation': [],
    }
    
    # Collect per-sample objective values for correlation analysis
    all_costs = []
    all_carbons = []
    
    # ========================================================================
    # Gradient Descent Training Loop
    # ========================================================================
    
    start_time = time.time()
    
    for epoch in range(num_iterations):
        epoch_losses = []
        epoch_costs = []
        epoch_carbons = []
        epoch_constraint_violations = []
        epoch_weighted_objectives = []  # Track weighted objective function
        
        for batch_idx, (train_x, train_y, batch_indices) in enumerate(training_loader):
            train_x = train_x.to(device) 
            
            # Get current V_anchor for this batch and enable gradients
            V_anchor_batch = V_anchor_all[batch_indices].clone().requires_grad_(True) 
            
            # Compute loss and gradient 
            loss, loss_dict = loss_fn(V_anchor_batch, train_x, only_obj=False) 
            
            grad_V = torch.autograd.grad(
                outputs=loss,
                inputs=V_anchor_batch,
                create_graph=False,
                retain_graph=False,
            )[0]
            
            # Clip gradient to prevent large updates
            grad_V = _clip_gradient(grad_V, grad_clip_norm)
            
            # Update V_anchor using gradient descent with optional post-processing 
            with torch.no_grad():
                # Gradient descent update
                V_anchor_batch_new = V_anchor_batch - learning_rate * grad_V
                
                # Clamp to valid voltage range
                min_v = Vbias - 2 * Vscale
                max_v = Vbias + 2 * Vscale
                V_anchor_batch_new = torch.clamp(V_anchor_batch_new, min=min_v, max=max_v)
                
                # Apply post-processing if enabled
                if use_post_processing:
                    try:
                        V_anchor_batch_new = _apply_post_processing_to_batch(
                            V_anchor_batch_new.detach().cpu().numpy(),
                            train_x,
                            ngt_data,
                            sys_data,
                            config,
                            device,
                            BRANFT,
                            verbose=(epoch == 0 and batch_idx == 0)
                        )
                        # Ensure voltage remains in valid range after post-processing
                        V_anchor_batch_new = torch.clamp(V_anchor_batch_new, min=min_v, max=max_v)
                    except Exception as e:
                        if epoch == 0 and batch_idx == 0:
                            print(f"[Warning] Post-processing failed: {e}, using gradient-updated voltage") 
            
            # Update stored V_anchor
            V_anchor_all[batch_indices] = V_anchor_batch_new
            
            # Record metrics
            epoch_losses.append(loss.item())
            epoch_costs.append(loss_dict.get('loss_cost', 0.0))
            epoch_carbons.append(loss_dict.get('loss_carbon', 0.0))
            lambda_cost = config.ngt_lambda_cost
            lambda_carbon = 1 - lambda_cost
            epoch_weighted_objectives.append(_compute_weighted_objective(loss_dict, lambda_cost, lambda_carbon))
            epoch_constraint_violations.append(_compute_constraint_violation(loss_dict))
            
            # Collect per-sample values for correlation analysis (only at final epoch)
            if epoch == num_iterations - 1:
                cost_samples = loss_dict.get('cost_per_sample')
                carbon_samples = loss_dict.get('carbon_per_sample')
                if cost_samples is not None and carbon_samples is not None:
                    all_costs.extend(cost_samples)
                    all_carbons.extend(carbon_samples)
        
        # Compute average metrics for this epoch
        n_batches = len(epoch_losses)
        if n_batches > 0:
            avg_loss = sum(epoch_losses) / n_batches
            avg_cost = sum(epoch_costs) / n_batches
            avg_carbon = sum(epoch_carbons) / n_batches
            avg_constraint_violation = sum(epoch_constraint_violations) / n_batches
            avg_weighted_obj = sum(epoch_weighted_objectives) / n_batches
        else:
            avg_loss = avg_cost = avg_carbon = avg_constraint_violation = avg_weighted_obj = 0.0
        
        # Update loss history
        loss_history['total'].append(avg_loss)
        loss_history['cost'].append(avg_cost)
        loss_history['carbon'].append(avg_carbon)
        loss_history['constraint_violation'].append(avg_constraint_violation)
        
        # Print progress periodically
        print_interval = max(1, num_iterations // 500)
        if (epoch + 1) % print_interval == 0 or epoch == 0:
            print(f"  Iteration {epoch+1}/{num_iterations}: "
                  f"loss={avg_loss:.4f}, cost={avg_cost:.2f}, carbon={avg_carbon:.4f}, "
                  f"weighted_obj={avg_weighted_obj:.2f}, constraint_vio={avg_constraint_violation:.4f}, "
                  f"lambda_cost={lambda_cost:.2f}, lambda_carbon={lambda_carbon:.2f}")
        
        # TensorBoard logging
        if tb_logger:
            tb_logger.log_scalar('gradient_descent/loss', avg_loss, epoch)
            tb_logger.log_scalar('gradient_descent/cost', avg_cost, epoch)
            tb_logger.log_scalar('gradient_descent/carbon', avg_carbon, epoch)
            tb_logger.log_scalar('gradient_descent/constraint_violation', avg_constraint_violation, epoch)
    
    elapsed_time = time.time() - start_time
    print(f"\n[Gradient Descent] Completed in {elapsed_time:.2f}s")
    
    # Analyze correlation between cost and carbon
    if len(all_costs) > 0 and len(all_carbons) > 0:
        correlation_result = _analyze_objective_correlation(all_costs, all_carbons, config.results_dir)
        return {
            'loss_history': loss_history,
            'correlation_analysis': correlation_result,
        }
    else:
        return {'loss_history': loss_history}


def _precompute_V_anchor_physical_all(
    *,
    config, sys_data, ngt_data, device,
    vae_vm, vae_va,
    Vscale, Vbias,
    bus_slack,
):
    """
    Precompute V_anchor in physical space (NGT format) for all training samples.
    
    Returns:
        V_anchor_physical: [N, output_dim] in physical space
    """
    x_train = ngt_data['x_train'].to(device)
    bus_Pnet_all = ngt_data['bus_Pnet_all']
    bus_Pnet_noslack_all = bus_Pnet_all[bus_Pnet_all != bus_slack]
    
    # VAE forward pass
    with torch.no_grad():
        Vm_scaled = vae_vm(x_train, use_mean=True)           # [N, Nbus] (scaled)
        Va_scaled_noslack = vae_va(x_train, use_mean=True)   # [N, Nbus-1] (scaled)
    
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
    
    Vm_anchor_full = Vm_scaled / scale_vm * (VmUb - VmLb) + VmLb
    Va_anchor_full_noslack = Va_scaled_noslack / scale_va
    
    N_samples = x_train.shape[0]
    Va_anchor_full = torch.zeros(N_samples, config.Nbus, device=device)
    Va_anchor_full[:, :bus_slack] = Va_anchor_full_noslack[:, :bus_slack]
    Va_anchor_full[:, bus_slack + 1:] = Va_anchor_full_noslack[:, bus_slack:]
    
    Vm_nonZIB = Vm_anchor_full[:, bus_Pnet_all]
    Va_nonZIB_noslack = Va_anchor_full[:, bus_Pnet_noslack_all]
    V_anchor_physical = torch.cat([Va_nonZIB_noslack, Vm_nonZIB], dim=1)
    
    return V_anchor_physical.detach()

 
# ------------------------ helpers ------------------------

def _ngt_to_full_voltage(V_ngt, ngt_data, sys_data, config, device):
    """
    Convert NGT format voltage to full voltage (Vm_full, Va_full).
    
    Args:
        V_ngt: [batch, NPred_Va + NPred_Vm] in NGT format (physical space)
        ngt_data: NGT data dictionary
        sys_data: System data
        config: Configuration
        device: Device
        
    Returns:
        Vm_full: [batch, Nbus] full voltage magnitude
        Va_full: [batch, Nbus] full voltage angle (with slack inserted)
    """
    batch_size = V_ngt.shape[0]
    bus_slack = int(sys_data.bus_slack)
    
    # Convert to numpy
    V_ngt_np = V_ngt.detach().cpu().numpy() if torch.is_tensor(V_ngt) else V_ngt
    
    # Insert slack bus Va (=0) to get full non-ZIB voltage
    xam_P = np.insert(V_ngt_np, ngt_data['idx_bus_Pnet_slack'][0], 0, axis=1)
    Va_len_with_slack = ngt_data['NPred_Va'] + 1
    Va_nonZIB = xam_P[:, :Va_len_with_slack]
    Vm_nonZIB = xam_P[:, Va_len_with_slack:Va_len_with_slack + ngt_data['NPred_Vm']]
    
    # Convert to complex and reconstruct full voltage
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
    
    Vm_full = np.sqrt(Ve**2 + Vf**2)
    Va_full = np.arctan2(Vf, Ve)
    
    return Vm_full, Va_full


def _full_to_ngt_voltage(Vm_full, Va_full, ngt_data, sys_data, config):
    """
    Convert full voltage back to NGT format.
    
    Args:
        Vm_full: [batch, Nbus] full voltage magnitude
        Va_full: [batch, Nbus] full voltage angle (with slack)
        ngt_data: NGT data dictionary
        sys_data: System data
        config: Configuration
        
    Returns:
        V_ngt: [batch, NPred_Va + NPred_Vm] in NGT format (physical space)
    """
    batch_size = Vm_full.shape[0]
    bus_slack = int(sys_data.bus_slack)
    bus_Pnet_all = _ensure_1d_int(ngt_data['bus_Pnet_all'])
    bus_Pnet_noslack_all = bus_Pnet_all[bus_Pnet_all != bus_slack]
    
    # Extract non-ZIB voltages
    Vm_nonZIB = Vm_full[:, bus_Pnet_all]
    Va_nonZIB_noslack = Va_full[:, bus_Pnet_noslack_all]
    
    # Concatenate to NGT format: [Va_nonZIB_noslack, Vm_nonZIB]
    V_ngt = np.concatenate([Va_nonZIB_noslack, Vm_nonZIB], axis=1)
    
    return V_ngt


def _apply_post_processing_to_batch(
    V_ngt_batch, train_x_batch, ngt_data, sys_data, config, device, BRANFT, verbose=False
):
    """
    Apply post-processing to a batch of NGT format voltages.
    
    Args:
        V_ngt_batch: [batch, NPred_Va + NPred_Vm] in NGT format (physical space)
        train_x_batch: [batch, input_dim] load data
        ngt_data: NGT data dictionary
        sys_data: System data
        config: Configuration
        device: Device
        BRANFT: Branch from-to indices
        verbose: Whether to print warnings
        
    Returns:
        V_ngt_corrected: [batch, NPred_Va + NPred_Vm] corrected NGT format voltage
    """
    batch_size = V_ngt_batch.shape[0]
    
    # Convert NGT to full voltage
    Vm_full, Va_full = _ngt_to_full_voltage(V_ngt_batch, ngt_data, sys_data, config, device)
    
    # Build load arrays
    num_Pd = len(ngt_data['bus_Pd'])
    Pd_full = np.zeros((batch_size, config.Nbus))
    Qd_full = np.zeros((batch_size, config.Nbus))
    train_x_np = train_x_batch.detach().cpu().numpy() if torch.is_tensor(train_x_batch) else train_x_batch
    Pd_full[:, ngt_data['bus_Pd']] = train_x_np[:, :num_Pd]
    Qd_full[:, ngt_data['bus_Qd']] = train_x_np[:, num_Pd:]
    
    # Create a minimal EvalContext for post-processing
    try:
        # Create a minimal context-like structure
        class TempContext:
            def __init__(self):
                self.config = config
                self.sys_data = sys_data
                self.BRANFT = BRANFT
                self.device = device
                self.Nbus = config.Nbus
                self.Ntest = batch_size
                self.bus_slack = int(sys_data.bus_slack)
                self.baseMVA = float(sys_data.baseMVA.item() if hasattr(sys_data.baseMVA, 'item') else sys_data.baseMVA)
                self.branch = _as_numpy(sys_data.branch)
                self.Ybus = sys_data.Ybus
                self.Yf = sys_data.Yf
                self.Yt = sys_data.Yt
                self.bus_Pg = _ensure_1d_int(sys_data.bus_Pg)
                self.bus_Qg = _ensure_1d_int(sys_data.bus_Qg)
                self.MAXMIN_Pg = _as_numpy(ngt_data['MAXMIN_Pg'])
                self.MAXMIN_Qg = _as_numpy(ngt_data['MAXMIN_Qg'])
                self.idxPg = _ensure_1d_int(sys_data.idxPg)
                self.gencost = _as_numpy(sys_data.gencost)
                self.gencost_Pg = _as_numpy(ngt_data.get('gencost_Pg', None))
                self.his_V = _as_numpy(sys_data.his_V)
                self.hisVm_min = _as_numpy(sys_data.hisVm_min)
                self.hisVm_max = _as_numpy(sys_data.hisVm_max)
                self.bus_Pnet_all = _ensure_1d_int(ngt_data['bus_Pnet_all'])
                self.bus_Pnet_noslack_all = self.bus_Pnet_all[self.bus_Pnet_all != self.bus_slack]
                self.bus_ZIB_all = _ensure_1d_int(ngt_data['bus_ZIB_all']) if 'bus_ZIB_all' in ngt_data else None
                self.param_ZIMV = ngt_data.get('param_ZIMV', None)
                self.VmLb = getattr(config, 'ngt_VmLb', None)
                self.VmUb = getattr(config, 'ngt_VmUb', None)
                self.DELTA = float(getattr(config, 'DELTA', 1e-4))
                self.k_dV = float(getattr(config, 'k_dV', 1.0))
                # [IMPROVEMENT] Use current voltage for Jacobian calculation instead of historical
                # This ensures more accurate linearization when voltage deviates from historical values
                self.flag_hisv = False  # Use current voltage, not historical
                self.Pdtest = Pd_full
                self.Qdtest = Qd_full
                # Store current voltage for Jacobian calculation
                self.current_V = Vm_full * np.exp(1j * Va_full)
        
        temp_ctx = TempContext()
        
        # Apply post-processing
        Vm_corrected, Va_corrected, _, dbg_info = post_process_like_evaluate_model(
            temp_ctx, Vm_full, Va_full
        )
        
        # [FIX] Only re-apply Kron reconstruction if post-processing used strict subspace mode
        # If relax_ngt_post=True (default), post_process_like_evaluate_model already corrected
        # all nodes (including ZIB) in full space, and applying Kron reconstruction here would
        # UNDO those corrections and snap back to the (potentially infeasible) manifold.
        # Strategy: Only apply Kron reconstruction when using strict subspace mode.
        use_strict_subspace = dbg_info.get('mode') == 'strict_subspace'
        if use_strict_subspace and ngt_data.get('param_ZIMV') is not None and ngt_data.get('bus_ZIB_all') is not None:
            # Extract corrected non-ZIB voltages (these are the ones we want to keep)
            bus_Pnet_all = _ensure_1d_int(ngt_data['bus_Pnet_all'])
            bus_slack = int(sys_data.bus_slack)
            bus_Pnet_noslack_all = bus_Pnet_all[bus_Pnet_all != bus_slack]
            
            # Get corrected non-ZIB voltages
            Vm_nonZIB_corrected = Vm_corrected[:, bus_Pnet_all].copy()
            
            # Reconstruct Va_nonZIB with slack insertion (need full Va for all non-ZIB buses)
            # bus_Pnet_all includes slack, so we need to extract Va for all non-ZIB buses
            # Va_corrected shape: [batch, Nbus] - contains all buses including slack and ZIB
            # Simply extract Va for bus_Pnet_all (slack angle is already 0 in Va_corrected)
            Va_nonZIB_with_slack = Va_corrected[:, bus_Pnet_all].copy()
            # Ensure slack bus angle is 0 (should already be, but enforce it)
            idx_slack_in_Pnet = np.where(bus_Pnet_all == bus_slack)[0]
            if len(idx_slack_in_Pnet) > 0:
                Va_nonZIB_with_slack[:, idx_slack_in_Pnet[0]] = 0.0
            
            # Convert to complex and reconstruct ZIB using Kron
            Vx_corrected = Vm_nonZIB_corrected * np.exp(1j * Va_nonZIB_with_slack)
            Vy_reconstructed = np.dot(ngt_data['param_ZIMV'], Vx_corrected.T).T
            
            # Update corrected voltages: keep non-ZIB corrections, use Kron-reconstructed ZIB
            Vm_corrected_final = Vm_corrected.copy()
            Va_corrected_final = Va_corrected.copy()
            Vm_corrected_final[:, ngt_data['bus_ZIB_all']] = np.abs(Vy_reconstructed)
            Va_corrected_final[:, ngt_data['bus_ZIB_all']] = np.angle(Vy_reconstructed)
            
            Vm_corrected = Vm_corrected_final
            Va_corrected = Va_corrected_final
        
        # Convert back to NGT format
        V_ngt_corrected = _full_to_ngt_voltage(Vm_corrected, Va_corrected, ngt_data, sys_data, config)
        
        return torch.tensor(V_ngt_corrected, dtype=torch.float32, device=device)
        
    except Exception as e:
        # If post-processing fails, return original voltage
        if verbose:
            print(f"[Warning] Post-processing failed: {e}, returning original voltage")
        return V_ngt_batch if torch.is_tensor(V_ngt_batch) else torch.tensor(V_ngt_batch, dtype=torch.float32, device=device)

 
# ===================== [MO-PREF] Loss / Eval helpers =====================
def main():
    """
    Main function with support for training
    """
    # Load configuration
    config = get_config()
    config.print_config()
    
    # Create output directories if they don't exist
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True) 
    # Load data
    sys_data, _, _ = load_all_data(config) 
    # results = train_unsupervised_ngt_gradient_descent(
    #             config=config, sys_data=sys_data, device=config.device,  
    #             num_iterations=500,   # 迭代次数
    #             learning_rate=1e-5,   # 更小的学习率 (1e-5 to 1e-6 recommended)
    #             tb_logger=None,       # 可选：TensorBoard logger 
    #             grad_clip_norm=1.0,  # 梯度裁剪
    #             use_post_processing=False,  # 启用后处理
    #             )
    
    # # Print correlation analysis results if available
    # if results and 'correlation_analysis' in results:
    #     corr_analysis = results['correlation_analysis']
    #     if corr_analysis.get('pearson_r') is not None:
    #         print("\n" + "=" * 60)
    #         print("Final Correlation Analysis Summary")
    #         print("=" * 60)
    #         print(f"Cost range: [{np.min(corr_analysis.get('cost_mean', 0) - corr_analysis.get('cost_std', 0)):.2f}, "
    #               f"{np.max(corr_analysis.get('cost_mean', 0) + corr_analysis.get('cost_std', 0)):.2f}] $/h")
    #         print(f"Carbon range: [{np.min(corr_analysis.get('carbon_mean', 0) - corr_analysis.get('carbon_std', 0)):.2f}, "
    #               f"{np.max(corr_analysis.get('carbon_mean', 0) + corr_analysis.get('carbon_std', 0)):.2f}] tCO2/h")
    #         print(f"\n{corr_analysis.get('interpretation', '')}")
    #         print("=" * 60)
    
    # Additional analysis: Fixed load correlation from REAL training data
    print("\n[Additional Analysis] Analyzing fixed load correlation from training data...")
    ngt_data, sys_data = load_ngt_training_data(config, sys_data)
    
    fixed_load_results = analyze_fixed_load_from_training_data(
        config=config,
        sys_data=sys_data,
        ngt_data=ngt_data
    )
    
    if fixed_load_results:
        print("\n" + "=" * 60)
        print("Key Findings from Training Data Analysis")
        print("=" * 60)
        print(f"Overall correlation (all samples): {fixed_load_results['overall_pearson']:.4f}")
        print(f"Average correlation within load groups: {fixed_load_results['average_pearson']:.4f} ± {fixed_load_results['std_pearson']:.4f}")
        print(f"Correlation range across groups: [{fixed_load_results['min_pearson']:.4f}, {fixed_load_results['max_pearson']:.4f}]")
        
        if fixed_load_results['average_pearson'] < fixed_load_results['overall_pearson']:
            print("\n[CONFIRMED] Hypothesis CONFIRMED: Fixed load groups show LOWER correlation than overall!")
            print("  This suggests that load variation is a major factor in the high correlation.")
        else:
            print("\n[NOT CONFIRMED] Hypothesis NOT confirmed: Fixed load groups show similar or HIGHER correlation.")
            print("  This suggests the correlation is due to power allocation patterns, not just load variation.")
        print("=" * 60)

if __name__ == "__main__":
    main()