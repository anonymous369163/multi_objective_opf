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
from main_part.data_loader import load_multi_preference_dataset, load_ngt_training_data
from main_part.utils import get_genload, get_Pgcost, get_carbon_emission_vectorized
from main_part.opf_by_pypower import PyPowerOPFSolver


def ngt_state_error(x_pred, x_true, NPred_Va):
    """
    Compute error between NGT-format states with Va wrap handling.
    
    Args:
        x_pred: [..., output_dim] predicted state (can be numpy or torch)
        x_true: [..., output_dim] true state (same type as x_pred)
        NPred_Va: Number of Va dimensions (first NPred_Va elements)
        
    Returns:
        error: scalar error (float if numpy, tensor if torch)
    """
    is_torch = torch.is_tensor(x_pred)
    if is_torch:
        device = x_pred.device
        x_pred_np = x_pred.detach().cpu().numpy()
        x_true_np = x_true.detach().cpu().numpy()
    else:
        x_pred_np = np.asarray(x_pred)
        x_true_np = np.asarray(x_true)
    
    # Handle Va periodicity for angle dimensions (first NPred_Va dimensions)
    dx = x_pred_np - x_true_np
    dx_wrapped = dx.copy()
    
    if NPred_Va > 0:
        # Wrap angle difference to [-pi, pi] range
        for dim_idx in range(min(NPred_Va, dx.shape[-1])):
            dx_angle = dx[..., dim_idx]
            dx_wrapped[..., dim_idx] = np.arctan2(np.sin(dx_angle), np.cos(dx_angle))
    
    # Compute norm along last dimension
    error = np.linalg.norm(dx_wrapped, axis=-1)
    
    if is_torch:
        # [FIX] Use torch.from_numpy().to() to ensure dtype and device consistency
        return torch.from_numpy(error).to(device=device, dtype=x_pred.dtype)
    else:
        return error


def reconstruct_full_from_partial_simple(V_partial, bus_Pnet_all, bus_Pnet_noslack_all, 
                                         bus_slack, Nbus, param_ZIMV=None, bus_ZIB_all=None,
                                         VmLb=None, VmUb=None):
    """
    Reconstruct full voltage from NGT partial format.
    
    V_partial layout: [Va_noslack_nonZIB, Vm_nonZIB]
    
    This function matches the standard implementation in unified_eval.py::reconstruct_full_from_partial
    and _kron_reconstruct_zib to ensure consistency.
    """
    # Ensure inputs are 1D integer arrays
    bus_Pnet_all = np.asarray(bus_Pnet_all, dtype=int).reshape(-1)
    bus_Pnet_noslack_all = np.asarray(bus_Pnet_noslack_all, dtype=int).reshape(-1)
    bus_slack = int(bus_slack)
    
    NPred_Va = len(bus_Pnet_noslack_all)
    NPred_Vm = len(bus_Pnet_all)
    assert V_partial.shape[1] == NPred_Va + NPred_Vm, \
        f"V_partial dim mismatch: got {V_partial.shape[1]}, expect {NPred_Va + NPred_Vm}"
    
    Va_noslack_nonZIB = V_partial[:, :NPred_Va]
    Vm_nonZIB = V_partial[:, NPred_Va:]
    
    Pred_Va_full = np.zeros((V_partial.shape[0], Nbus), dtype=float)
    Pred_Vm_full = np.zeros((V_partial.shape[0], Nbus), dtype=float)
    
    # Insert Va for non-slack, non-ZIB buses
    Pred_Va_full[:, bus_Pnet_noslack_all] = Va_noslack_nonZIB
    # Slack bus angle is always 0 (reference bus)
    Pred_Va_full[:, bus_slack] = 0.0
    
    # Insert Vm for non-ZIB buses (including slack)
    Pred_Vm_full[:, bus_Pnet_all] = Vm_nonZIB
    
    # Kron reconstruct ZIB if available
    # This matches the standard _kron_reconstruct_zib implementation
    if param_ZIMV is not None and bus_ZIB_all is not None:
        bus_ZIB_all = np.asarray(bus_ZIB_all, dtype=int).reshape(-1)
        if len(bus_ZIB_all) > 0:
            # Ensure slack bus angle is 0 before Kron reconstruction
            # (should already be set above, but enforce it for safety)
            Pred_Va_full[:, bus_slack] = 0.0
            
            # Compute complex voltage for non-ZIB buses (including slack)
            # bus_Pnet_all includes slack, and Pred_Va_full[:, bus_slack] is already 0
            Vx = Pred_Vm_full[:, bus_Pnet_all] * np.exp(1j * Pred_Va_full[:, bus_Pnet_all])
            
            # Kron reduction: Vy = param_ZIMV @ Vx
            Vy = (np.asarray(param_ZIMV) @ Vx.T).T  # [n_samples, NZIB]
            
            # Update ZIB voltages
            Pred_Va_full[:, bus_ZIB_all] = np.angle(Vy)
            Pred_Vm_full[:, bus_ZIB_all] = np.abs(Vy)
    
    # Optional Vm clamp bounds (NGT-style)
    if VmLb is not None and VmUb is not None:
        Pred_Vm_full = np.clip(Pred_Vm_full, VmLb, VmUb)
    
    return Pred_Vm_full, Pred_Va_full

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
    
    # [FIX] Get NPred_Va and NPred_Vm for Va wrap handling
    NPred_Va = multi_pref_data.get('NPred_Va', output_dim // 2)  # Fallback to half if not available
    NPred_Vm = multi_pref_data.get('NPred_Vm', output_dim - NPred_Va)
    
    # [FIX] Calculate normalized Δx_k with Va wrap and per-dimension standardization
    # Compute per-dimension standard deviations for normalization
    std_by_dim = np.std(sorted_solutions, axis=0, ddof=1) + 1e-8  # Add epsilon to avoid division by zero
    
    deltas = []
    deltas_raw = []  # Keep raw deltas for comparison
    for k in range(len(sorted_solutions) - 1):
        dx = sorted_solutions[k+1] - sorted_solutions[k]
        deltas_raw.append(np.linalg.norm(dx))
        
        # [FIX] Handle Va periodicity for angle dimensions (first NPred_Va dimensions)
        dx_normalized = dx.copy()
        for dim_idx in range(NPred_Va):
            # Wrap angle difference to [-pi, pi] range using atan2
            dx_angle = dx[dim_idx]
            dx_normalized[dim_idx] = np.arctan2(np.sin(dx_angle), np.cos(dx_angle))
        
        # [FIX] Normalize by standard deviation (element-wise division)
        dx_normalized = dx_normalized / std_by_dim
        delta = np.linalg.norm(dx_normalized)
        deltas.append(delta)
    
    deltas = np.array(deltas)
    deltas_raw = np.array(deltas_raw)
    
    # [FIX] Use median-based kink score for more robust detection
    median_delta = np.median(deltas)
    kink_scores = deltas / (median_delta + 1e-8)
    # Consider kink if score > 3 (more robust than percentile)
    kink_threshold_score = 3.0
    kink_indices = np.where(kink_scores > kink_threshold_score)[0]
    
    # Also report percentile-based threshold for comparison
    delta_threshold_percentile = np.percentile(deltas, 90)  # Top 10% as kinks
    
    print(f"\nKink Analysis:")
    print(f"  Mean delta (normalized): {np.mean(deltas):.6f}")
    print(f"  Median delta (normalized): {median_delta:.6f}")
    print(f"  Max delta (normalized): {np.max(deltas):.6f}")
    print(f"  Mean delta (raw): {np.mean(deltas_raw):.6f}")
    print(f"  Max delta (raw): {np.max(deltas_raw):.6f}")
    print(f"  Kink threshold (score > {kink_threshold_score}): {kink_threshold_score * median_delta:.6f}")
    print(f"  Delta threshold (90th percentile): {delta_threshold_percentile:.6f}")
    print(f"  Number of kink regions (score-based): {len(kink_indices)}")
    if len(kink_indices) > 0:
        print(f"  Kink locations (lambda indices): {kink_indices.tolist()}")
        print(f"  Kink lambda values: {[sorted_lambdas[i] for i in kink_indices]}")
        print(f"  Kink scores: {kink_scores[kink_indices]}")
    
    # Select a few key output dimensions to visualize
    # Choose dimensions that show variation
    var_by_dim = np.var(sorted_solutions, axis=0)
    top_var_indices = np.argsort(var_by_dim)[-5:]  # Top 5 most varying dimensions
    
    print(f"\nSelected key dimensions (top 5 by variance): {top_var_indices.tolist()}")
    
    # Plot 1: Delta x_k vs k
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot delta curve (normalized)
    axes[0].plot(range(len(deltas)), deltas, 'b-', linewidth=1.5, label='|x_{k+1} - x_k| (normalized)')
    axes[0].axhline(y=kink_threshold_score * median_delta, color='r', linestyle='--', linewidth=1, 
                   label=f'Kink threshold (score > {kink_threshold_score})')
    axes[0].axhline(y=delta_threshold_percentile, color='orange', linestyle='--', linewidth=1, 
                   label='90th percentile')
    if len(kink_indices) > 0:
        axes[0].scatter(kink_indices, deltas[kink_indices], color='red', s=50, zorder=5, label='Kink regions')
    axes[0].set_xlabel('Preference Index k', fontsize=12)
    axes[0].set_ylabel('|x_{k+1} - x_k| (normalized)', fontsize=12)
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
        'deltas_raw': deltas_raw,
        'kink_indices': kink_indices,
        'kink_lambdas': [sorted_lambdas[i] for i in kink_indices] if len(kink_indices) > 0 else [],
        'kink_scores': kink_scores,
        'delta_threshold_percentile': delta_threshold_percentile,
        'kink_threshold_score': kink_threshold_score,
        'median_delta': median_delta,
        'mean_delta': np.mean(deltas),
        'max_delta': np.max(deltas),
        'mean_delta_raw': np.mean(deltas_raw),
        'max_delta_raw': np.max(deltas_raw)
    }


def mvp1_velocity_field_fitting(multi_pref_data, sys_data=None, config=None,
                                num_train_samples=20, num_test_samples=10, 
                                output_dir='results', hidden_dim=256, num_layers=3, 
                                num_epochs=300, lr=1e-3, batch_size=32):
    """
    MVP-1: 差分速度场拟合实验（多样本版本）
    
    用简单的 MLP 学习 dx/dλ，从 x(λ_1) 用 Euler 积分到 x(λ_K)，
    看能否接近真实 x(λ_K)。
    
    改进：
    1. 使用真实的 Δλ 逐段积分（而不是均匀 step）
    2. 模型输入包含场景 s (x_train) 作为条件
    3. 用多个样本训练和测试
    
    Args:
        multi_pref_data: Multi-preference data dictionary
        num_train_samples: Number of samples to use for training
        num_test_samples: Number of samples to use for testing
        output_dir: Directory to save results
        hidden_dim: Hidden dimension for MLP
        num_layers: Number of layers for MLP
        num_epochs: Training epochs
        lr: Learning rate
        batch_size: Batch size for training
    """
    print("\n" + "=" * 80)
    print("MVP-1: Velocity Field Fitting and Euler Integration (Multi-Sample)")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    y_train_by_pref = multi_pref_data['y_train_by_pref']
    x_train_data = multi_pref_data['x_train']  # [n_train, input_dim]
    lambda_carbon_values = sorted(multi_pref_data['lambda_carbon_values'])
    output_dim = multi_pref_data['output_dim']
    input_dim = multi_pref_data['input_dim']
    n_total_samples = multi_pref_data['n_train']
    NPred_Va = multi_pref_data.get('NPred_Va', output_dim // 2)  # [FIX] Get NPred_Va for Va wrap
    
    # [FIX] Check if we can compute objective space errors (need sys_data and config)
    # Check all required attributes (more robust than checking bus_Pg which may not exist)
    compute_objective_errors = (
        sys_data is not None and config is not None and
        hasattr(sys_data, 'Ybus') and hasattr(sys_data, 'baseMVA') and
        hasattr(sys_data, 'idxPg') and hasattr(sys_data, 'gencost')
    )
    
    # [NEW] Setup OPF solver for re-running OPF to get original Pg (for validation)
    # Only enable this for first test sample to save time (re-running OPF is slow)
    enable_opf_validation = compute_objective_errors  # Can be set to False to skip validation
    opf_solver = None
    if enable_opf_validation:
        try:
            # [FIX] Ensure main_part is in sys.path for opf_by_pypower.py's imports
            main_part_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main_part')
            if main_part_dir not in sys.path:
                sys.path.insert(0, main_part_dir)
            
            # Get case file path (assume standard location)
            case_m_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       'main_part', 'data', f'case{config.Nbus}_ieee_modified.m')
            if not os.path.exists(case_m_path):
                # Try alternative path
                case_m_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          'main_part', 'data', f'case{config.Nbus}_ieee_modified.m')
            
            if os.path.exists(case_m_path):
                # Load ngt_data for OPF solver
                from main_part.data_loader import load_ngt_training_data
                ngt_data_for_opf, _ = load_ngt_training_data(config, sys_data=sys_data)
                
                # Initialize OPF solver (will be reused for each preference)
                opf_solver = PyPowerOPFSolver(
                    case_m_path=case_m_path,
                    ngt_data=ngt_data_for_opf,
                    verbose=False,
                    use_multi_objective=True,
                    lambda_cost=1.0,  # Will be overridden per preference
                    lambda_carbon=0.0,  # Will be overridden per preference
                    carbon_scale=30.0,  # Default carbon scale
                    sys_data=sys_data
                )
                print("\n[Info] OPF solver initialized for original Pg validation (will re-run OPF for first test sample)")
            else:
                print(f"\n[Warning] Case file not found at {case_m_path}, skipping OPF validation")
                enable_opf_validation = False
        except Exception as e:
            print(f"\n[Warning] Failed to initialize OPF solver for validation: {e}")
            import traceback
            traceback.print_exc()
            enable_opf_validation = False
    
    if compute_objective_errors:
        print("\n[Info] System data available - will compute objective space errors (cost/carbon)")
        # Get necessary system parameters
        bus_Pnet_all = multi_pref_data['bus_Pnet_all']
        bus_Pnet_noslack_all = multi_pref_data['bus_Pnet_noslack_all']
        bus_slack = int(sys_data.bus_slack)
        baseMVA = float(sys_data.baseMVA)
        Ybus = sys_data.Ybus
        
        # [FIX] Get bus_Pg and bus_Qg (matching standard implementation in data_loader.py)
        try:
            # Standard way: bus_Pg = gen[idxPg, 0] - 1, bus_Qg = gen[idxQg, 0] - 1
            if hasattr(sys_data, 'bus_Pg') and sys_data.bus_Pg is not None:
                bus_Pg = np.asarray(sys_data.bus_Pg, dtype=int).reshape(-1)
            elif hasattr(sys_data, 'idxPg') and hasattr(sys_data, 'gen'):
                # Derive from gen matrix using idxPg (matching data_loader.py line 185)
                bus_Pg = sys_data.gen[sys_data.idxPg, 0].astype(int) - 1
            else:
                raise AttributeError("Cannot determine bus_Pg: need either sys_data.bus_Pg or sys_data.idxPg + sys_data.gen")
            
            if hasattr(sys_data, 'bus_Qg') and sys_data.bus_Qg is not None:
                bus_Qg = np.asarray(sys_data.bus_Qg, dtype=int).reshape(-1)
            elif hasattr(sys_data, 'idxQg') and hasattr(sys_data, 'gen'):
                # Derive from gen matrix using idxQg (matching data_loader.py line 186)
                bus_Qg = sys_data.gen[sys_data.idxQg, 0].astype(int) - 1
            else:
                # Fallback: assume same buses for Pg and Qg (not ideal, but better than failing)
                bus_Qg = bus_Pg
                print(f"  [Warning] bus_Qg not found, using bus_Pg as fallback")
        except Exception as e:
            compute_objective_errors = False
            print(f"  [Warning] Cannot determine bus_Pg/bus_Qg: {e}")
            print("  Skipping objective space error calculation")
        
        if compute_objective_errors:
            # Get cost and carbon calculation parameters
            idxPg = sys_data.idxPg
            gencost = sys_data.gencost
            
            # Try to get GCI values for carbon calculation
            try:
                from main_part.utils import get_gci_for_generators
                gci_values = get_gci_for_generators(sys_data)
                print(f"  GCI values loaded: {len(gci_values)} generators")
            except Exception as e:
                gci_values = None
                print(f"  [Warning] GCI values not available: {e}")
                print("  Carbon calculation will be skipped")
            
            # Get load data for power flow calculation
            # x_train format: [Pd_nonzero, Qd_nonzero] / baseMVA
            bus_Pd = multi_pref_data['bus_Pd']
            bus_Qd = multi_pref_data['bus_Qd']
        else:
            gci_values = None
    else:
        print("\n[Info] System data not available - skipping objective space error calculation")
        gci_values = None
    
    # Select train and test samples
    num_train_samples = min(num_train_samples, n_total_samples)
    num_test_samples = min(num_test_samples, n_total_samples - num_train_samples)
    
    # [FIX] Random split with fixed seed for reproducibility (avoid distribution bias)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_total_samples)
    train_sample_indices = perm[:num_train_samples].tolist()
    test_sample_indices = perm[num_train_samples:num_train_samples + num_test_samples].tolist()
    
    print(f"\nDataset info:")
    print(f"  Total samples: {n_total_samples}")
    print(f"  Training samples: {num_train_samples}")
    print(f"  Test samples: {num_test_samples}")
    print(f"  Preferences: {len(lambda_carbon_values)}")
    print(f"  Input dim: {input_dim}, Output dim: {output_dim}")
    print(f"  Lambda range: [{lambda_carbon_values[0]:.2f}, {lambda_carbon_values[-1]:.2f}]")
    
    # Normalize lambda to [0, 1] for better training
    lambda_min = lambda_carbon_values[0]
    lambda_max = lambda_carbon_values[-1]
    lambda_normalized = [(lc - lambda_min) / (lambda_max - lambda_min) if lambda_max > lambda_min else 0.0 
                         for lc in lambda_carbon_values]
    lambda_tensor_all = torch.tensor(lambda_normalized, device=device, dtype=torch.float32)
    
    # Prepare training data from multiple samples
    print(f"\nPreparing training data from {num_train_samples} samples...")
    all_x = []  # Current state x
    all_s = []  # Scene s (input features)
    all_lambda = []  # Current lambda
    all_v = []  # Ground truth velocity
    
    for sample_idx in train_sample_indices:
        # Get solutions for this sample
        solutions = []
        for lc in lambda_carbon_values:
            if lc in y_train_by_pref:
                if isinstance(y_train_by_pref[lc], torch.Tensor):
                    solutions.append(y_train_by_pref[lc][sample_idx].to(device))
                else:
                    solutions.append(torch.tensor(y_train_by_pref[lc][sample_idx], device=device))
        
        sorted_solutions = torch.stack(solutions)  # [n_prefs, output_dim]
        scene_features = x_train_data[sample_idx].to(device)  # [input_dim]
        
        # Compute velocities using real Δλ
        for k in range(len(sorted_solutions) - 1):
            dx = sorted_solutions[k+1] - sorted_solutions[k]
            dlambda = lambda_tensor_all[k+1] - lambda_tensor_all[k]
            v = dx / (dlambda + 1e-8)  # Avoid division by zero
            
            all_x.append(sorted_solutions[k])
            all_s.append(scene_features)
            all_lambda.append(lambda_tensor_all[k].unsqueeze(0))
            all_v.append(v)
    
    # Stack all training data
    x_train = torch.stack(all_x)  # [n_train_pairs, output_dim]
    s_train = torch.stack(all_s)  # [n_train_pairs, input_dim]
    lambda_train = torch.stack(all_lambda)  # [n_train_pairs, 1]
    v_train = torch.stack(all_v)  # [n_train_pairs, output_dim]
    
    n_train_pairs = x_train.shape[0]
    print(f"  Total training pairs: {n_train_pairs}")
    print(f"  x_train shape: {x_train.shape}")
    print(f"  s_train shape: {s_train.shape}")
    print(f"  lambda_train shape: {lambda_train.shape}")
    print(f"  v_train shape: {v_train.shape}")
    
    # Create MLP: input = [s, x, lambda], output = velocity
    class VelocityMLP(nn.Module):
        def __init__(self, scene_dim, state_dim, output_dim, hidden_dim, num_layers):
            super().__init__()
            layers = []
            # Input: [s, x, lambda] = [scene_dim + state_dim + 1]
            input_total_dim = scene_dim + state_dim + 1
            layers.append(nn.Linear(input_total_dim, hidden_dim))
            layers.append(nn.ReLU())
            
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.net = nn.Sequential(*layers)
        
        def forward(self, s, x, lambda_val):
            # s: [batch, scene_dim], x: [batch, state_dim], lambda_val: [batch, 1]
            x_concat = torch.cat([s, x, lambda_val], dim=1)
            return self.net(x_concat)
    
    model = VelocityMLP(input_dim, output_dim, output_dim, hidden_dim, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"\nModel architecture:")
    print(f"  Input: [scene ({input_dim}), state ({output_dim}), lambda (1)]")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num layers: {num_layers}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop with batching
    print(f"\nTraining for {num_epochs} epochs...")
    losses = []
    n_batches = (n_train_pairs + batch_size - 1) // batch_size
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Shuffle data
        indices = torch.randperm(n_train_pairs, device=device)
        x_train_shuffled = x_train[indices]
        s_train_shuffled = s_train[indices]
        lambda_train_shuffled = lambda_train[indices]
        v_train_shuffled = v_train[indices]
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_train_pairs)
            
            batch_x = x_train_shuffled[start_idx:end_idx]
            batch_s = s_train_shuffled[start_idx:end_idx]
            batch_lambda = lambda_train_shuffled[start_idx:end_idx]
            batch_v = v_train_shuffled[start_idx:end_idx]
            
            optimizer.zero_grad()
            
            # Forward pass
            v_pred = model(batch_s, batch_x, batch_lambda)
            loss = criterion(v_pred, batch_v)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")
    
    print(f"  Final loss: {losses[-1]:.6f}")
    
    # [NEW] Stage 1: Teacher-forcing error (velocity field fitting quality)
    print(f"\n[Stage 1] Evaluating Teacher-forcing Error (Velocity Field Fitting)...")
    model.eval()
    with torch.no_grad():
        v_pred_all = model(s_train, x_train, lambda_train)
        v_error = torch.norm(v_pred_all - v_train, dim=1)  # [n_train_pairs]
        teacher_forcing_mean = float(v_error.mean())
        teacher_forcing_std = float(v_error.std())
        
    print(f"  Teacher-forcing error (train): Mean = {teacher_forcing_mean:.6f}, Std = {teacher_forcing_std:.6f}")
    
    # [NEW] Compute baseline velocities (for Baseline-B: constant velocity field)
    baseline_v_mean = v_train.mean(dim=0, keepdim=True)  # [1, output_dim] - global average velocity
    
    # Test: Euler integration on test samples using REAL Δλ
    print(f"\n[Stage 2-3] Testing Euler integration on {num_test_samples} test samples...")
    print("  Using REAL Δλ for integration (not uniform steps)")
    model.eval()
    
    test_results = []
    
    # [NEW] Stage 1 (test): Teacher-forcing on test samples
    teacher_forcing_test_errors = []
    
    with torch.no_grad():
        for test_idx, sample_idx in enumerate(test_sample_indices):
            # Get solutions for this test sample
            solutions = []
            for lc in lambda_carbon_values:
                if lc in y_train_by_pref:
                    if isinstance(y_train_by_pref[lc], torch.Tensor):
                        solutions.append(y_train_by_pref[lc][sample_idx].to(device))
                    else:
                        solutions.append(torch.tensor(y_train_by_pref[lc][sample_idx], device=device))
            
            sorted_solutions = torch.stack(solutions)  # [n_prefs, output_dim]
            scene_features = x_train_data[sample_idx].to(device)  # [input_dim]
            
            # [NEW] Stage 1 (test): Teacher-forcing error on test sample
            # Compute velocity predictions at true states
            v_pred_test = []
            v_true_test = []
            for k in range(len(sorted_solutions) - 1):
                lambda_current = lambda_tensor_all[k].view(1, 1)
                # Predict velocity at true x_k
                v_pred_k = model(scene_features.view(1, -1), sorted_solutions[k:k+1], lambda_current)
                v_pred_test.append(v_pred_k.squeeze(0))
                # True velocity (finite difference)
                dlambda_real = lambda_tensor_all[k+1] - lambda_tensor_all[k]
                v_true_k = (sorted_solutions[k+1] - sorted_solutions[k]) / (dlambda_real + 1e-8)
                v_true_test.append(v_true_k)
            
            v_pred_test = torch.stack(v_pred_test)  # [n_prefs-1, output_dim]
            v_true_test = torch.stack(v_true_test)  # [n_prefs-1, output_dim]
            teacher_forcing_error_test = float(torch.norm(v_pred_test - v_true_test, dim=1).mean())
            teacher_forcing_test_errors.append(teacher_forcing_error_test)
            
            # Stage 2: One-step rollout errors (local dynamics quality)
            one_step_errors = []
            for k in range(len(sorted_solutions) - 1):
                lambda_current = lambda_tensor_all[k].view(1, 1)
                dlambda_real = lambda_tensor_all[k+1] - lambda_tensor_all[k]
                
                # Predict velocity at true x_k
                v_pred = model(scene_features.view(1, -1), sorted_solutions[k:k+1], lambda_current)
                v_pred = v_pred.squeeze(0)
                
                # One-step state error (state space, with Va wrap)
                x_one_step = sorted_solutions[k] + dlambda_real * v_pred
                one_step_error = ngt_state_error(x_one_step.unsqueeze(0), sorted_solutions[k+1:k+2], NPred_Va).item()
                
                one_step_errors.append(one_step_error)
            
            one_step_errors = np.array(one_step_errors)
            one_step_mean_error = np.mean(one_step_errors)
            
            # [FIX] Baseline-A: Linear interpolation (true baseline, not oracle)
            # Linear interpolation between x(λ0) and x(λK) with Va shortest-arc handling
            x_linear_interp = []
            x_start = sorted_solutions[0]
            x_end = sorted_solutions[-1]
            lambda_start = lambda_tensor_all[0]
            lambda_end = lambda_tensor_all[-1]
            for k in range(len(sorted_solutions)):
                lambda_k = lambda_tensor_all[k]
                if lambda_end > lambda_start:
                    alpha = (lambda_k - lambda_start) / (lambda_end - lambda_start)
                    # [FIX] For Va dimensions, use shortest-arc interpolation
                    x_interp = x_start.clone()
                    # Handle Va dimensions (first NPred_Va) with wrap
                    if NPred_Va > 0:
                        for dim_idx in range(min(NPred_Va, x_start.shape[0])):
                            va_start = x_start[dim_idx].item()
                            va_end = x_end[dim_idx].item()
                            # Compute wrapped difference (shortest arc)
                            d_va = va_end - va_start
                            d_va_wrapped = np.arctan2(np.sin(d_va), np.cos(d_va))  # Wrap to [-pi, pi]
                            # Interpolate along shortest arc
                            x_interp[dim_idx] = va_start + alpha * d_va_wrapped
                        # For Vm dimensions (after NPred_Va), use regular linear interpolation
                        for dim_idx in range(NPred_Va, x_start.shape[0]):
                            x_interp[dim_idx] = x_start[dim_idx] + alpha * (x_end[dim_idx] - x_start[dim_idx])
                    else:
                        # No Va dimensions, regular linear interpolation for all dimensions
                        x_interp = x_start + alpha * (x_end - x_start)
                else:
                    x_interp = x_start.clone()
                x_linear_interp.append(x_interp)
            x_linear_interp = torch.stack(x_linear_interp)
            # [FIX] Use Va-wrap error calculation
            errors_baseline_a = ngt_state_error(x_linear_interp, sorted_solutions, NPred_Va).cpu().numpy()
            
            # [NEW] Baseline-B: Constant velocity field (global average)
            x_baseline_b = [sorted_solutions[0].clone()]
            x_current_b = sorted_solutions[0].clone()
            for k in range(len(sorted_solutions) - 1):
                dlambda_real = lambda_tensor_all[k+1] - lambda_tensor_all[k]
                # Use global average velocity
                v_const = baseline_v_mean.squeeze(0)  # [output_dim]
                x_current_b = x_current_b + dlambda_real * v_const
                x_baseline_b.append(x_current_b.clone())
            x_baseline_b = torch.stack(x_baseline_b)
            # [FIX] Use Va-wrap error calculation
            errors_baseline_b = ngt_state_error(x_baseline_b, sorted_solutions, NPred_Va).cpu().numpy()
            
            # Stage 3: Multi-step rollout (learned model)
            x_integrated = [sorted_solutions[0].clone()]
            x_current = sorted_solutions[0].clone()
            
            for k in range(len(sorted_solutions) - 1):
                lambda_current = lambda_tensor_all[k].view(1, 1)
                dlambda_real = lambda_tensor_all[k+1] - lambda_tensor_all[k]
                
                # Predict velocity at current point
                v_pred = model(scene_features.view(1, -1), x_current.view(1, -1), lambda_current)
                v_pred = v_pred.squeeze(0)
                
                # Euler step with REAL Δλ
                x_current = x_current + dlambda_real * v_pred
                x_integrated.append(x_current.clone())
            
            x_integrated = torch.stack(x_integrated)  # [n_prefs, output_dim]
            
            # [FIX] Compute errors (multi-step) with Va wrap
            errors = ngt_state_error(x_integrated, sorted_solutions, NPred_Va).cpu().numpy()  # [n_prefs]
            final_error = float(errors[-1])  # [FIX] Use float() instead of .item()
            mean_error = float(errors.mean())
            max_error = float(errors.max())
            
            baseline_b_mean_error = np.mean(errors_baseline_b)
            
            # [NEW] Stage 3: Objective space errors (cost/carbon) - if system data available
            objective_errors = {}
            if compute_objective_errors:
                try:
                    # Get load for this sample
                    x_sample = x_train_data[sample_idx].cpu().numpy()
                    num_Pd = len(bus_Pd)
                    
                    # Prepare load data (p.u. format)
                    Pd_sample_pu = np.zeros(config.Nbus)
                    Qd_sample_pu = np.zeros(config.Nbus)
                    Pd_sample_pu[bus_Pd] = x_sample[:num_Pd]  # Keep in p.u.
                    Qd_sample_pu[bus_Qd] = x_sample[num_Pd:]  # Keep in p.u.
                    
                    # Reconstruct full voltage for true solutions
                    # NOTE: sorted_solutions is from y_train_by_pref, which stores voltage in PHYSICAL UNITS
                    # (Va in radians, Vm in p.u.) - no normalization/Vscale/Vbias needed
                    sorted_solutions_np = sorted_solutions.cpu().numpy()  # [n_prefs, output_dim]
                    x_integrated_np = x_integrated.cpu().numpy()  # [n_prefs, output_dim]
                    x_baseline_b_np = x_baseline_b.cpu().numpy()  # [n_prefs, output_dim]
                    
                    
                    # Convert to full voltage
                    # reconstruct_full_from_partial_simple expects physical units (p.u. for Vm, radians for Va)
                    Vm_true_full, Va_true_full = reconstruct_full_from_partial_simple(
                        sorted_solutions_np, bus_Pnet_all, bus_Pnet_noslack_all, 
                        bus_slack, config.Nbus, param_ZIMV=multi_pref_data.get('param_ZIMV'),
                        bus_ZIB_all=multi_pref_data.get('bus_ZIB_all'),
                        VmLb=getattr(config, 'ngt_VmLb', None),
                        VmUb=getattr(config, 'ngt_VmUb', None))
                    
                    Vm_pred_full, Va_pred_full = reconstruct_full_from_partial_simple(
                        x_integrated_np, bus_Pnet_all, bus_Pnet_noslack_all,
                        bus_slack, config.Nbus, param_ZIMV=multi_pref_data.get('param_ZIMV'),
                        bus_ZIB_all=multi_pref_data.get('bus_ZIB_all'),
                        VmLb=getattr(config, 'ngt_VmLb', None),
                        VmUb=getattr(config, 'ngt_VmUb', None))
                    Vm_base_b_full, Va_base_b_full = reconstruct_full_from_partial_simple(
                        x_baseline_b_np, bus_Pnet_all, bus_Pnet_noslack_all,
                        bus_slack, config.Nbus, param_ZIMV=multi_pref_data.get('param_ZIMV'),
                        bus_ZIB_all=multi_pref_data.get('bus_ZIB_all'),
                        VmLb=getattr(config, 'ngt_VmLb', None),
                        VmUb=getattr(config, 'ngt_VmUb', None))
                    
                    # Compute Pg from voltage using power flow
                    # NOTE: V (complex voltage) should be in p.u. format
                    # Ybus (admittance matrix) is also in p.u. format
                    # So power injection P = real(V * conj(Ybus @ V)) is in p.u.
                    V_true = Vm_true_full * np.exp(1j * Va_true_full)
                    V_pred = Vm_pred_full * np.exp(1j * Va_pred_full)
                    V_base_b = Vm_base_b_full * np.exp(1j * Va_base_b_full)
                    
                    # Expand Pd/Qd to full bus dimension for get_genload (p.u. format)
                    Pd_full_pu = np.tile(Pd_sample_pu, (len(lambda_carbon_values), 1))  # [n_prefs, Nbus] in p.u.
                    Qd_full_pu = np.tile(Qd_sample_pu, (len(lambda_carbon_values), 1))  # [n_prefs, Nbus] in p.u.
                    
                    # Compute Pg from voltage using power flow (get_genload expects p.u. format)
                    Pg_true_pu, _, _, _ = get_genload(V_true, Pd_full_pu, Qd_full_pu, bus_Pg, bus_Qg, Ybus)
                    
                    # [NEW] VALIDATION: Re-run OPF to get original Pg and compare with get_genload Pg
                    # This validates if get_genload's Pg allocation matches OPF's original Pg allocation
                    Pg_true_original = None  # Will be filled if validation is enabled
                    cost_true_original = None
                    carbon_true_original = None
                    validation_passed = False
                    
                    if enable_opf_validation and opf_solver is not None and test_idx == 0:
                        print(f"\n  [VALIDATION] Re-running OPF for sample {sample_idx} to get original Pg...")
                        try:
                            # Prepare load input for OPF solver (x_sample is already in p.u. format)
                            x_load_for_opf = x_sample  # [input_dim] in p.u. format
                            
                            # Re-run OPF for each preference to get original Pg
                            Pg_original_all = []  # Will store Pg for each preference
                            cost_original_all = []
                            carbon_original_all = []
                            
                            for pref_idx, lc in enumerate(lambda_carbon_values):
                                # Use same preference weights as dataset generation:
                                # lambda_cost = 1.0 (fixed), lambda_carbon = lc (varies from 0 to 100)
                                # Reference: expand_training_data_multi_preference.py line 429-431
                                lambda_cost_actual = 1.0
                                lambda_carbon_actual = lc
                                
                                # Re-run OPF with same preference to get original Pg
                                result_opf = opf_solver.forward(x_load_for_opf, preference=[lambda_cost_actual, lambda_carbon_actual])
                                
                                if result_opf.get('success', False):
                                    # Extract original Pg from OPF result (in MW, need to convert to p.u.)
                                    Pg_gen_MW = result_opf['gen']['Pg_MW']  # [ngen] in MW
                                    # Extract only active generators (idxPg)
                                    Pg_active_MW = Pg_gen_MW[idxPg]  # [len(idxPg)] in MW
                                    Pg_active_pu = Pg_active_MW / baseMVA  # Convert to p.u.
                                    Pg_original_all.append(Pg_active_pu)
                                    
                                    # Get cost and carbon from OPF result summary
                                    cost_original = result_opf['summary'].get('economic_cost', 0.0)
                                    carbon_original = result_opf['summary'].get('carbon_emission', 0.0)
                                    cost_original_all.append(cost_original)
                                    carbon_original_all.append(carbon_original)
                                else:
                                    print(f"    [Warning] OPF failed for lambda_carbon={lc:.2f}, skipping...")
                                    # Use get_genload Pg as fallback
                                    Pg_original_all.append(Pg_true_pu[pref_idx])
                                    cost_original_all.append(0.0)
                                    carbon_original_all.append(0.0)
                            
                            if len(Pg_original_all) == len(lambda_carbon_values):
                                Pg_true_original = np.array(Pg_original_all)  # [n_prefs, len(idxPg)] in p.u.
                                cost_true_original = np.array(cost_original_all)  # [n_prefs]
                                carbon_true_original = np.array(carbon_original_all)  # [n_prefs]
                                
                                # Compare original Pg vs get_genload Pg
                                Pg_diff = np.abs(Pg_true_original - Pg_true_pu)  # [n_prefs, len(idxPg)]
                                Pg_diff_mean = np.mean(Pg_diff)
                                Pg_diff_max = np.max(Pg_diff)
                                
                                print(f"  [VALIDATION] Original Pg vs get_genload Pg comparison:")
                                print(f"    Mean absolute difference: {Pg_diff_mean:.6f} p.u.")
                                print(f"    Max absolute difference: {Pg_diff_max:.6f} p.u.")
                                if Pg_diff_mean < 0.01:
                                    print(f"    -> Good agreement (mean diff < 0.01 p.u.)")
                                    validation_passed = True
                                elif Pg_diff_mean < 0.1:
                                    print(f"    -> Moderate difference (mean diff < 0.1 p.u.)")
                                else:
                                    print(f"    -> WARNING: Large difference (mean diff >= 0.1 p.u.)")
                                    print(f"       This suggests get_genload Pg allocation differs from OPF original Pg")
                                
                                # Cost and carbon comparison will be done after cost_true and carbon_true are computed
                        except Exception as e:
                            print(f"  [Warning] Failed to re-run OPF for validation: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # Compute Pg for predicted and baseline (use p.u. format - correct way)
                    Pg_pred, _, _, _ = get_genload(V_pred, Pd_full_pu, Qd_full_pu, bus_Pg, bus_Qg, Ybus)
                    Pg_base_b, _, _, _ = get_genload(V_base_b, Pd_full_pu, Qd_full_pu, bus_Pg, bus_Qg, Ybus)
                    
                    # Compute cost and carbon (get_genload returns Pg in p.u. format)
                    cost_true = get_Pgcost(Pg_true_pu, idxPg, gencost, baseMVA)  # [n_prefs]
                    cost_pred = get_Pgcost(Pg_pred, idxPg, gencost, baseMVA)  # [n_prefs]
                    cost_base_b = get_Pgcost(Pg_base_b, idxPg, gencost, baseMVA)  # [n_prefs]
                    
                    # [NEW] Store validation results (original Pg vs get_genload Pg)
                    if Pg_true_original is not None:
                        objective_errors['Pg_true_original'] = Pg_true_original
                        objective_errors['Pg_true_pu'] = Pg_true_pu  # Store for comparison
                        objective_errors['cost_true_original'] = cost_true_original
                        if carbon_true_original is not None:
                            objective_errors['carbon_true_original'] = carbon_true_original
                        objective_errors['validation_passed'] = validation_passed
                        
                        # Store Pg difference statistics
                        Pg_diff = np.abs(Pg_true_original - Pg_true_pu)
                        objective_errors['Pg_diff_mean'] = np.mean(Pg_diff)
                        objective_errors['Pg_diff_max'] = np.max(Pg_diff)
                    
                    cost_error = np.abs(cost_pred - cost_true)  # [n_prefs]
                    cost_error_base_b = np.abs(cost_base_b - cost_true)  # [n_prefs]
                    
                    objective_errors['cost_true'] = cost_true
                    objective_errors['cost_pred'] = cost_pred
                    objective_errors['cost_base_b'] = cost_base_b
                    objective_errors['cost_error'] = cost_error
                    objective_errors['cost_error_base_b'] = cost_error_base_b
                    objective_errors['cost_error_mean'] = np.mean(cost_error)
                    objective_errors['cost_error_final'] = cost_error[-1]
                    
                    if gci_values is not None:
                        # Extract GCI values for active generators only
                        gci_active = gci_values[idxPg]  # [len(idxPg)]
                        
                        # Compute carbon (get_genload returns Pg in p.u. format)
                        carbon_true = get_carbon_emission_vectorized(Pg_true_pu, gci_active, baseMVA)  # [n_prefs]
                        carbon_pred = get_carbon_emission_vectorized(Pg_pred, gci_active, baseMVA)  # [n_prefs]
                        carbon_base_b = get_carbon_emission_vectorized(Pg_base_b, gci_active, baseMVA)  # [n_prefs]
                        
                        carbon_error = np.abs(carbon_pred - carbon_true)  # [n_prefs]
                        carbon_error_base_b = np.abs(carbon_base_b - carbon_true)  # [n_prefs]
                        
                        objective_errors['carbon_true'] = carbon_true
                        objective_errors['carbon_pred'] = carbon_pred
                        objective_errors['carbon_base_b'] = carbon_base_b
                        objective_errors['carbon_error'] = carbon_error
                        objective_errors['carbon_error_base_b'] = carbon_error_base_b
                        objective_errors['carbon_error_mean'] = np.mean(carbon_error)
                        objective_errors['carbon_error_final'] = carbon_error[-1]
                        
                        # [VALIDATION] Compare cost and carbon with original OPF results
                        if Pg_true_original is not None and len(cost_true_original) == len(cost_true):
                            cost_diff_orig_vs_genload = np.abs(cost_true_original - cost_true)
                            objective_errors['cost_diff_original_vs_genload'] = cost_diff_orig_vs_genload
                            cost_diff_orig_mean = np.mean(cost_diff_orig_vs_genload)
                            cost_diff_orig_max = np.max(cost_diff_orig_vs_genload)
                            if test_idx == 0:  # Only print for first test sample
                                print(f"\n  [VALIDATION] Original cost vs get_genload cost comparison:")
                                print(f"    Mean absolute difference: {cost_diff_orig_mean:.2f} $/h")
                                print(f"    Max absolute difference: {cost_diff_orig_max:.2f} $/h")
                            
                            if carbon_true_original is not None and len(carbon_true_original) == len(carbon_true):
                                carbon_diff_orig_vs_genload = np.abs(carbon_true_original - carbon_true)
                                objective_errors['carbon_diff_original_vs_genload'] = carbon_diff_orig_vs_genload
                                carbon_diff_orig_mean = np.mean(carbon_diff_orig_vs_genload)
                                carbon_diff_orig_max = np.max(carbon_diff_orig_vs_genload)
                                if test_idx == 0:  # Only print for first test sample
                                    print(f"  [VALIDATION] Original carbon vs get_genload carbon comparison:")
                                    print(f"    Mean absolute difference: {carbon_diff_orig_mean:.2f} tCO2/h")
                                    print(f"    Max absolute difference: {carbon_diff_orig_max:.2f} tCO2/h")
                except Exception as e:
                    print(f"  [Warning] Failed to compute objective errors for sample {sample_idx}: {e}")
                    objective_errors = {}
            
            test_results.append({
                'sample_idx': sample_idx,
                # Stage 1 (test): Teacher-forcing
                'teacher_forcing_error': teacher_forcing_error_test,
                # Stage 2: One-step errors
                'one_step_errors': one_step_errors,
                'one_step_mean_error': one_step_mean_error,
                # Stage 3: Multi-step errors (learned model)
                'final_error': final_error,
                'mean_error': mean_error,
                'max_error': max_error,
                'errors': errors,
                'x_integrated': x_integrated.cpu(),
                'x_true': sorted_solutions.cpu(),
                # Baseline-A: Linear interpolation
                'baseline_a_errors': errors_baseline_a,
                'baseline_a_mean_error': float(np.mean(errors_baseline_a)),
                'baseline_a_final_error': float(errors_baseline_a[-1]),
                # Baseline-B: Constant velocity field
                'baseline_b_errors': errors_baseline_b,
                'baseline_b_mean_error': baseline_b_mean_error,
                'baseline_b_final_error': float(errors_baseline_b[-1]),
                'x_baseline_b': x_baseline_b.cpu(),
                # [NEW] Objective space errors
                'objective_errors': objective_errors
            })
            
            if (test_idx + 1) % 5 == 0:
                print(f"  Test sample {test_idx+1}/{num_test_samples}: "
                      f"One-step={one_step_mean_error:.6f}, "
                      f"Multi-step={mean_error:.6f}, "
                      f"Baseline-B={baseline_b_mean_error:.6f}")
    
    # Aggregate statistics
    # Stage 1: Teacher-forcing (train - already computed above)
    # Stage 1 (test): Teacher-forcing on test samples
    avg_teacher_forcing_test = float(np.mean(teacher_forcing_test_errors)) if teacher_forcing_test_errors else None
    
    # Stage 2: One-step rollout
    one_step_mean_errors = [r['one_step_mean_error'] for r in test_results]
    avg_one_step_error = np.mean(one_step_mean_errors)
    
    # Stage 3: Multi-step rollout (learned model)
    final_errors = [r['final_error'] for r in test_results]
    mean_errors = [r['mean_error'] for r in test_results]
    max_errors = [r['max_error'] for r in test_results]
    avg_final_error = np.mean(final_errors)
    avg_mean_error = np.mean(mean_errors)
    avg_max_error = np.mean(max_errors)
    
    # Baseline-A: Linear interpolation
    baseline_a_mean_errors = [r['baseline_a_mean_error'] for r in test_results]
    baseline_a_final_errors = [r['baseline_a_final_error'] for r in test_results]
    avg_baseline_a_mean = np.mean(baseline_a_mean_errors)
    avg_baseline_a_final = np.mean(baseline_a_final_errors)
    
    # Baseline-B: Constant velocity field
    baseline_b_mean_errors = [r['baseline_b_mean_error'] for r in test_results]
    baseline_b_final_errors = [r['baseline_b_final_error'] for r in test_results]
    avg_baseline_b_mean = np.mean(baseline_b_mean_errors)
    avg_baseline_b_final = np.mean(baseline_b_final_errors)
    
    # [NEW] Aggregate objective space errors
    if compute_objective_errors and len([r for r in test_results if r['objective_errors']]) > 0:
        cost_errors_all = [r['objective_errors']['cost_error_mean'] for r in test_results if r['objective_errors']]
        cost_errors_final_all = [r['objective_errors']['cost_error_final'] for r in test_results if r['objective_errors']]
        avg_cost_error = np.mean(cost_errors_all) if cost_errors_all else None
        avg_cost_error_final = np.mean(cost_errors_final_all) if cost_errors_final_all else None
        
        if gci_values is not None:
            carbon_errors_all = [r['objective_errors']['carbon_error_mean'] for r in test_results if r['objective_errors'] and 'carbon_error_mean' in r['objective_errors']]
            carbon_errors_final_all = [r['objective_errors']['carbon_error_final'] for r in test_results if r['objective_errors'] and 'carbon_error_final' in r['objective_errors']]
            avg_carbon_error = np.mean(carbon_errors_all) if carbon_errors_all else None
            avg_carbon_error_final = np.mean(carbon_errors_final_all) if carbon_errors_final_all else None
        else:
            avg_carbon_error = None
            avg_carbon_error_final = None
    else:
        avg_cost_error = None
        avg_cost_error_final = None
        avg_carbon_error = None
        avg_carbon_error_final = None
    
    print(f"\nAggregate Test Results:")
    print(f"\n  [Stage 1] Teacher-forcing (Velocity Field Fitting):")
    print(f"    Train error: Mean = {teacher_forcing_mean:.6f} ± {teacher_forcing_std:.6f}")
    if avg_teacher_forcing_test is not None:
        print(f"    Test error: Mean = {avg_teacher_forcing_test:.6f} ± {np.std(teacher_forcing_test_errors):.6f}")
    print(f"\n  [Stage 2] One-step Rollout (Local Dynamics):")
    print(f"    Average one-step error: {avg_one_step_error:.6f} ± {np.std(one_step_mean_errors):.6f}")
    print(f"\n  [Stage 3] Multi-step Rollout (Learned Model):")
    print(f"    Average final error: {avg_final_error:.6f} ± {np.std(final_errors):.6f}")
    print(f"    Average mean error: {avg_mean_error:.6f} ± {np.std(mean_errors):.6f}")
    print(f"    Average max error: {avg_max_error:.6f}")
    print(f"\n  [Baseline-A] Linear Interpolation (True Endpoints):")
    print(f"    Average mean error: {avg_baseline_a_mean:.6f} ± {np.std(baseline_a_mean_errors):.6f}")
    print(f"    Average final error: {avg_baseline_a_final:.6f} ± {np.std(baseline_a_final_errors):.6f}")
    
    print(f"\n  [Baseline-B] Constant Velocity Field:")
    print(f"    Average mean error: {avg_baseline_b_mean:.6f} ± {np.std(baseline_b_mean_errors):.6f}")
    print(f"    Average final error: {avg_baseline_b_final:.6f} ± {np.std(baseline_b_final_errors):.6f}")
    
    if avg_cost_error is not None:
        print(f"\n  [Objective Space] Cost/Carbon Errors:")
        print(f"    Average cost error: {avg_cost_error:.2f} $/h ± {np.std(cost_errors_all):.2f}")
        print(f"    Average cost error (final): {avg_cost_error_final:.2f} $/h")
        if avg_carbon_error is not None:
            print(f"    Average carbon error: {avg_carbon_error:.2f} tCO2/h ± {np.std(carbon_errors_all):.2f}")
            print(f"    Average carbon error (final): {avg_carbon_error_final:.2f} tCO2/h")
        
        # [NEW] Print validation summary (OPF original Pg vs get_genload Pg)
        validation_results = [r.get('objective_errors', {}).get('Pg_true_original') for r in test_results 
                             if r.get('objective_errors', {}).get('Pg_true_original') is not None]
        if len(validation_results) > 0:
            # Get validation info from first test sample (the one we validated)
            first_validated = None
            for r in test_results:
                obj_err = r.get('objective_errors', {})
                if obj_err.get('Pg_true_original') is not None:
                    first_validated = r
                    break
            
            if first_validated is not None:
                obj_err = first_validated['objective_errors']
                print(f"\n  [VALIDATION SUMMARY] OPF Original Pg vs get_genload Pg (Sample {first_validated['sample_idx']}):")
                
                # Compare Pg directly
                if 'Pg_diff_mean' in obj_err and 'Pg_diff_max' in obj_err:
                    print(f"    Pg difference: Mean={obj_err['Pg_diff_mean']:.6f} p.u., Max={obj_err['Pg_diff_max']:.6f} p.u.")
                
                # Compare cost
                if 'cost_true_original' in obj_err and 'cost_true' in obj_err:
                    cost_diff = np.abs(obj_err['cost_true_original'] - obj_err['cost_true'])
                    print(f"    Cost difference (original vs get_genload): Mean={np.mean(cost_diff):.2f} $/h, Max={np.max(cost_diff):.2f} $/h")
                
                # Compare carbon
                if 'carbon_true_original' in obj_err and 'carbon_true' in obj_err:
                    carbon_diff = np.abs(obj_err['carbon_true_original'] - obj_err['carbon_true'])
                    print(f"    Carbon difference (original vs get_genload): Mean={np.mean(carbon_diff):.2f} tCO2/h, Max={np.max(carbon_diff):.2f} tCO2/h")
                
                # Final assessment
                if 'Pg_diff_mean' in obj_err:
                    if obj_err['Pg_diff_mean'] < 0.01:
                        print(f"    -> Good agreement: get_genload Pg allocation matches OPF original (mean diff < 0.01 p.u.)")
                    elif obj_err['Pg_diff_mean'] < 0.1:
                        print(f"    -> Moderate difference: get_genload Pg allocation differs moderately (mean diff < 0.1 p.u.)")
                    else:
                        print(f"    -> WARNING: Large difference (mean diff >= 0.1 p.u.)")
                        print(f"       -> get_genload Pg allocation differs significantly from OPF original")
                        print(f"       -> Objective errors may not be reliable due to Pg allocation mismatch")
                
                validation_passed = obj_err.get('validation_passed', False)
                print(f"    Validation status: {'PASSED' if validation_passed else 'MODERATE/FAILED'}")
    
    # [NEW] Plot results with three-stage comparison and baselines
    n_plots = min(3, num_test_samples)
    fig, axes = plt.subplots(n_plots, 3, figsize=(18, 5*n_plots))
    if n_plots == 1:
        axes = axes.reshape(1, -1)
    
    for plot_idx in range(n_plots):
        result = test_results[plot_idx]
        errors_plot = result['errors']  # Multi-step errors
        one_step_errors = result['one_step_errors']
        baseline_b_errors = result['baseline_b_errors']
        
        # Plot 1: Three-stage error comparison along trajectory
        axes[plot_idx, 0].plot(lambda_carbon_values[:-1], one_step_errors, 'g-', linewidth=1.5, 
                               label='One-step', alpha=0.7)
        axes[plot_idx, 0].plot(lambda_carbon_values, errors_plot, 'b-', linewidth=1.5, 
                               label='Multi-step (Learned)', alpha=0.7)
        axes[plot_idx, 0].plot(lambda_carbon_values, baseline_b_errors, 'r--', linewidth=1.5, 
                               label='Baseline-B (Const Vel)', alpha=0.7)
        axes[plot_idx, 0].set_xlabel('Lambda Carbon (λ)', fontsize=11)
        axes[plot_idx, 0].set_ylabel('Integration Error', fontsize=11)
        axes[plot_idx, 0].set_title(f'Sample {result["sample_idx"]}: Error Comparison\n'
                                   f'1-step={result["one_step_mean_error"]:.4f}, '
                                   f'Multi={result["mean_error"]:.4f}, '
                                   f'Base-B={result["baseline_b_mean_error"]:.4f}', fontsize=11)
        axes[plot_idx, 0].legend(fontsize=9)
        axes[plot_idx, 0].grid(True, alpha=0.3)
        
        # Plot 2: Compare key dimensions (Learned vs True)
        x_true = result['x_true']
        x_integrated = result['x_integrated']
        x_baseline_b = result['x_baseline_b']
        var_by_dim = torch.var(x_true, dim=0)
        top_var_indices = torch.argsort(var_by_dim)[-3:].cpu().numpy()
        
        for dim_idx in top_var_indices:
            true_vals = x_true[:, dim_idx].numpy()
            pred_vals = x_integrated[:, dim_idx].numpy()
            base_b_vals = x_baseline_b[:, dim_idx].numpy()
            axes[plot_idx, 1].plot(lambda_carbon_values, true_vals, 'k-o', linewidth=1.5, markersize=3, 
                                  label=f'True Dim {dim_idx}', alpha=0.8)
            axes[plot_idx, 1].plot(lambda_carbon_values, pred_vals, 'b--', linewidth=1.5, 
                                  label=f'Learned Dim {dim_idx}', alpha=0.7)
            axes[plot_idx, 1].plot(lambda_carbon_values, base_b_vals, 'r:', linewidth=1.5, 
                                  label=f'Base-B Dim {dim_idx}', alpha=0.6)
        
        axes[plot_idx, 1].set_xlabel('Lambda Carbon (λ)', fontsize=11)
        axes[plot_idx, 1].set_ylabel('Variable Value', fontsize=11)
        axes[plot_idx, 1].set_title(f'Sample {result["sample_idx"]}: Trajectory Comparison', fontsize=11)
        axes[plot_idx, 1].legend(fontsize=7, ncol=3)
        axes[plot_idx, 1].grid(True, alpha=0.3)
        
        # Plot 3: Error distribution comparison
        axes[plot_idx, 2].hist(one_step_errors, bins=15, alpha=0.5, edgecolor='green', 
                               label='One-step', color='green')
        axes[plot_idx, 2].hist(errors_plot, bins=15, alpha=0.5, edgecolor='blue', 
                               label='Multi-step', color='blue')
        axes[plot_idx, 2].hist(baseline_b_errors, bins=15, alpha=0.5, edgecolor='red', 
                               label='Baseline-B', color='red')
        axes[plot_idx, 2].axvline(x=result['one_step_mean_error'], color='green', linestyle='--', linewidth=2)
        axes[plot_idx, 2].axvline(x=result['mean_error'], color='blue', linestyle='--', linewidth=2)
        axes[plot_idx, 2].axvline(x=result['baseline_b_mean_error'], color='red', linestyle='--', linewidth=2)
        axes[plot_idx, 2].set_xlabel('Integration Error', fontsize=11)
        axes[plot_idx, 2].set_ylabel('Frequency', fontsize=11)
        axes[plot_idx, 2].set_title(f'Sample {result["sample_idx"]}: Error Distribution Comparison', fontsize=11)
        axes[plot_idx, 2].legend(fontsize=9)
        axes[plot_idx, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'mvp1_multi_sample_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {save_path}")
    plt.close()
    
    # [NEW] Plot objective space errors (cost/carbon) if available
    if compute_objective_errors and len([r for r in test_results if r.get('objective_errors')]) > 0:
        n_obj_plots = min(3, len([r for r in test_results if r.get('objective_errors')]))
        if n_obj_plots > 0:
            fig, axes = plt.subplots(n_obj_plots, 2, figsize=(14, 5*n_obj_plots))
            if n_obj_plots == 1:
                axes = axes.reshape(1, -1)
            
            plot_count = 0
            for plot_idx, result in enumerate(test_results):
                if plot_count >= n_obj_plots:
                    break
                obj_errors = result.get('objective_errors', {})
                if not obj_errors:
                    continue
                
                # Plot 1: Cost vs lambda
                if 'cost_true' in obj_errors:
                    axes[plot_count, 0].plot(lambda_carbon_values, obj_errors['cost_true'], 'k-o', 
                                           linewidth=1.5, markersize=3, label='True Cost', alpha=0.8)
                    axes[plot_count, 0].plot(lambda_carbon_values, obj_errors['cost_pred'], 'b--', 
                                           linewidth=1.5, label='Predicted Cost', alpha=0.7)
                    axes[plot_count, 0].plot(lambda_carbon_values, obj_errors['cost_base_b'], 'r:', 
                                           linewidth=1.5, label='Baseline-B Cost', alpha=0.6)
                    axes[plot_count, 0].set_xlabel('Lambda Carbon (λ)', fontsize=11)
                    axes[plot_count, 0].set_ylabel('Cost ($/h)', fontsize=11)
                    axes[plot_count, 0].set_title(f'Sample {result["sample_idx"]}: Cost vs Preference\n'
                                                 f'Mean error: {obj_errors["cost_error_mean"]:.1f} $/h', fontsize=11)
                    axes[plot_count, 0].legend(fontsize=9)
                    axes[plot_count, 0].grid(True, alpha=0.3)
                
                # Plot 2: Carbon vs lambda (if available)
                if 'carbon_true' in obj_errors:
                    axes[plot_count, 1].plot(lambda_carbon_values, obj_errors['carbon_true'], 'k-o', 
                                            linewidth=1.5, markersize=3, label='True Carbon', alpha=0.8)
                    axes[plot_count, 1].plot(lambda_carbon_values, obj_errors['carbon_pred'], 'b--', 
                                            linewidth=1.5, label='Predicted Carbon', alpha=0.7)
                    axes[plot_count, 1].plot(lambda_carbon_values, obj_errors['carbon_base_b'], 'r:', 
                                            linewidth=1.5, label='Baseline-B Carbon', alpha=0.6)
                    axes[plot_count, 1].set_xlabel('Lambda Carbon (λ)', fontsize=11)
                    axes[plot_count, 1].set_ylabel('Carbon Emission (tCO2/h)', fontsize=11)
                    axes[plot_count, 1].set_title(f'Sample {result["sample_idx"]}: Carbon vs Preference\n'
                                                 f'Mean error: {obj_errors["carbon_error_mean"]:.2f} tCO2/h', fontsize=11)
                    axes[plot_count, 1].legend(fontsize=9)
                    axes[plot_count, 1].grid(True, alpha=0.3)
                else:
                    # If no carbon, show cost error instead
                    if 'cost_error' in obj_errors:
                        axes[plot_count, 1].plot(lambda_carbon_values, obj_errors['cost_error'], 'b-', 
                                                linewidth=1.5, label='Cost Error', alpha=0.7)
                        axes[plot_count, 1].plot(lambda_carbon_values, obj_errors['cost_error_base_b'], 'r--', 
                                                linewidth=1.5, label='Baseline-B Cost Error', alpha=0.7)
                        axes[plot_count, 1].set_xlabel('Lambda Carbon (λ)', fontsize=11)
                        axes[plot_count, 1].set_ylabel('Cost Error ($/h)', fontsize=11)
                        axes[plot_count, 1].set_title(f'Sample {result["sample_idx"]}: Cost Error vs Preference', fontsize=11)
                        axes[plot_count, 1].legend(fontsize=9)
                        axes[plot_count, 1].grid(True, alpha=0.3)
                
                plot_count += 1
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, f'mvp1_objective_space_results.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved objective space plot to: {save_path}")
            plt.close()
    
    # [NEW] Plot training loss and three-stage error comparison across samples
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Training loss
    axes[0, 0].plot(losses)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss (Teacher-forcing)', fontsize=14)
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Three-stage error comparison across test samples
    x_samples = range(len(test_results))
    axes[0, 1].plot(x_samples, one_step_mean_errors, 'go-', label='One-step (Stage 2)', linewidth=2, markersize=6)
    axes[0, 1].plot(x_samples, mean_errors, 'bo-', label='Multi-step (Stage 3)', linewidth=2, markersize=6)
    axes[0, 1].plot(x_samples, baseline_b_mean_errors, 'rs--', label='Baseline-B (Const Vel)', linewidth=2, markersize=6)
    axes[0, 1].axhline(y=avg_one_step_error, color='green', linestyle=':', alpha=0.5, 
                       label=f'Avg 1-step: {avg_one_step_error:.4f}')
    axes[0, 1].axhline(y=avg_mean_error, color='blue', linestyle=':', alpha=0.5, 
                       label=f'Avg Multi: {avg_mean_error:.4f}')
    axes[0, 1].axhline(y=avg_baseline_b_mean, color='red', linestyle=':', alpha=0.5, 
                       label=f'Avg Base-B: {avg_baseline_b_mean:.4f}')
    axes[0, 1].set_xlabel('Test Sample Index', fontsize=12)
    axes[0, 1].set_ylabel('Mean Integration Error', fontsize=12)
    axes[0, 1].set_title('Three-Stage Error Comparison Across Samples', fontsize=14)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Final error comparison
    axes[1, 0].bar([i - 0.25 for i in x_samples], final_errors, width=0.25, alpha=0.7, 
                   label='Multi-step (Learned)', color='blue')
    axes[1, 0].bar([i + 0.0 for i in x_samples], baseline_b_final_errors, width=0.25, alpha=0.7, 
                   label='Baseline-B', color='red')
    axes[1, 0].axhline(y=avg_final_error, color='blue', linestyle='--', linewidth=2, 
                       label=f'Avg Multi: {avg_final_error:.4f}')
    axes[1, 0].axhline(y=avg_baseline_b_final, color='red', linestyle='--', linewidth=2, 
                       label=f'Avg Base-B: {avg_baseline_b_final:.4f}')
    axes[1, 0].set_xlabel('Test Sample Index', fontsize=12)
    axes[1, 0].set_ylabel('Final Integration Error', fontsize=12)
    axes[1, 0].set_title('Final Error: Learned vs Baseline-B', fontsize=14)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # [NEW] Plot 4: Error ratio (Multi-step / One-step) - shows error accumulation
    error_ratios = [m / (o + 1e-8) for m, o in zip(mean_errors, one_step_mean_errors)]
    axes[1, 1].plot(x_samples, error_ratios, 'mo-', linewidth=2, markersize=6)
    axes[1, 1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Ratio = 1.0')
    axes[1, 1].axhline(y=np.mean(error_ratios), color='magenta', linestyle=':', 
                       label=f'Avg ratio: {np.mean(error_ratios):.2f}')
    axes[1, 1].set_xlabel('Test Sample Index', fontsize=12)
    axes[1, 1].set_ylabel('Error Ratio (Multi-step / One-step)', fontsize=12)
    axes[1, 1].set_title('Error Accumulation Factor', fontsize=14)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'mvp1_training_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved summary plot to: {save_path}")
    plt.close()
    
    return {
        'model': model,
        # Stage 1: Teacher-forcing
        'teacher_forcing_mean': teacher_forcing_mean,
        'teacher_forcing_std': teacher_forcing_std,
        # Stage 2: One-step rollout
        'avg_one_step_error': avg_one_step_error,
        'one_step_errors_std': np.std(one_step_mean_errors),
        # Stage 3: Multi-step rollout
        'avg_final_error': avg_final_error,
        'avg_mean_error': avg_mean_error,
        'avg_max_error': avg_max_error,
        'std_final_error': np.std(final_errors),
        'std_mean_error': np.std(mean_errors),
        # Baseline-A: Linear interpolation
        'avg_baseline_a_mean': avg_baseline_a_mean,
        'avg_baseline_a_final': avg_baseline_a_final,
        'std_baseline_a_mean': np.std(baseline_a_mean_errors),
        # Baseline-B: Constant velocity
        'avg_baseline_b_mean': avg_baseline_b_mean,
        'avg_baseline_b_final': avg_baseline_b_final,
        'std_baseline_b_mean': np.std(baseline_b_mean_errors),
        # Error accumulation
        'avg_error_ratio': np.mean(error_ratios),
        # [NEW] Objective space errors
        'avg_cost_error': avg_cost_error,
        'avg_cost_error_final': avg_cost_error_final,
        'avg_carbon_error': avg_carbon_error,
        'avg_carbon_error_final': avg_carbon_error_final,
        # Full results
        'test_results': test_results,
        'training_losses': losses
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
        print(f"  Mean delta (normalized): {result['mean_delta']:.6f}")
        print(f"  Max delta (normalized): {result['max_delta']:.6f}")
        print(f"  Median delta (normalized): {result['median_delta']:.6f}")
        print(f"  Number of kinks (score-based): {len(result['kink_indices'])}")
        if len(result['kink_lambdas']) > 0:
            print(f"  Kink lambda values: {result['kink_lambdas']}")
            print(f"  Kink scores: {result['kink_scores'][result['kink_indices']]}")
    
    # Run MVP-1: Velocity field fitting for multiple samples
    print("\n" + "=" * 80)
    print("Running MVP-1: Velocity Field Fitting (Multi-Sample)")
    print("=" * 80)
    
    mvp1_result = mvp1_velocity_field_fitting(
        multi_pref_data,
        sys_data=sys_data,  # [NEW] Pass sys_data for objective space calculation
        config=config,      # [NEW] Pass config for objective space calculation
        num_train_samples=20,
        num_test_samples=10,
        output_dir=output_dir,
        hidden_dim=256,
        num_layers=3,
        num_epochs=300,
        lr=1e-3,
        batch_size=32
    )
    
    print("\n" + "=" * 80)
    print("MVP-1 Summary:")
    print("=" * 80)
    print(f"  Average final error: {mvp1_result['avg_final_error']:.6f}")
    print(f"  Average mean error: {mvp1_result['avg_mean_error']:.6f}")
    print(f"  Average max error: {mvp1_result['avg_max_error']:.6f}")
    print(f"  Std final error: {mvp1_result['std_final_error']:.6f}")
    
    # Determine feasibility
    print("\n" + "=" * 80)
    print("Feasibility Assessment:")
    print("=" * 80)
    
    # MVP-0 assessment
    avg_kinks = np.mean([len(r['kink_indices']) for r in kink_results.values()])
    avg_max_delta = np.mean([r['max_delta'] for r in kink_results.values()])  # Normalized delta
    avg_median_delta = np.mean([r['median_delta'] for r in kink_results.values()])
    
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
    print(f"\nMVP-1 (Velocity Field Fitting - Multi-Sample):")
    print(f"\n  [Stage 1] Teacher-forcing: {mvp1_result['teacher_forcing_mean']:.6f}")
    if mvp1_result['teacher_forcing_mean'] < 0.01:
        print("    -> Excellent: Velocity field is well learned")
    elif mvp1_result['teacher_forcing_mean'] < 0.1:
        print("    -> Good: Velocity field learning is acceptable")
    else:
        print("    -> Warning: Velocity field learning may need improvement")
    
    print(f"\n  [Stage 2] One-step: {mvp1_result['avg_one_step_error']:.6f}")
    print(f"  [Stage 3] Multi-step: {mvp1_result['avg_final_error']:.6f}")
    
    # Compare learned model vs baselines
    print(f"\n  [Comparison] Learned Model vs Baselines:")
    
    # vs Baseline-A (linear interpolation)
    if mvp1_result['avg_mean_error'] < mvp1_result['avg_baseline_a_mean']:
        improvement_a = (1 - mvp1_result['avg_mean_error'] / mvp1_result['avg_baseline_a_mean']) * 100
        print(f"    vs Baseline-A (linear interp): {improvement_a:.1f}% BETTER")
    else:
        degradation_a = (mvp1_result['avg_mean_error'] / mvp1_result['avg_baseline_a_mean'] - 1) * 100
        print(f"    vs Baseline-A (linear interp): {degradation_a:.1f}% WORSE")
    
    # vs Baseline-B (constant velocity)
    if mvp1_result['avg_mean_error'] < mvp1_result['avg_baseline_b_mean']:
        improvement_b = (1 - mvp1_result['avg_mean_error'] / mvp1_result['avg_baseline_b_mean']) * 100
        print(f"    vs Baseline-B (const velocity): {improvement_b:.1f}% BETTER")
    else:
        degradation_b = (mvp1_result['avg_mean_error'] / mvp1_result['avg_baseline_b_mean'] - 1) * 100
        print(f"    vs Baseline-B (const velocity): {degradation_b:.1f}% WORSE")
    
    # Error accumulation check
    if mvp1_result['avg_error_ratio'] < 2.0:
        print(f"    -> Error accumulation: Good (ratio={mvp1_result['avg_error_ratio']:.2f})")
    elif mvp1_result['avg_error_ratio'] < 5.0:
        print(f"    -> Error accumulation: Moderate (ratio={mvp1_result['avg_error_ratio']:.2f}), may need anchor alignment")
    else:
        print(f"    -> Error accumulation: High (ratio={mvp1_result['avg_error_ratio']:.2f}), need anchor alignment + corrector")
    
    # Overall feasibility
    if mvp1_result['avg_final_error'] < 0.1:
        print("\n  -> Overall: Excellent - flow model is very feasible")
    elif mvp1_result['avg_final_error'] < 1.0:
        print("\n  -> Overall: Good - flow model is feasible with anchor alignment")
    else:
        print("\n  -> Overall: Warning - need anchor alignment + corrector")
    
    print("\n" + "=" * 80)
    print("Experiments Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

