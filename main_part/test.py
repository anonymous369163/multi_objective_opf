#!/usr/bin/env python
# coding: utf-8
"""
Multi-Preference Model Evaluation and Pareto Front Analysis

This script evaluates models trained with train_multi_preference.py:
- Simple (MLP): standard mode, NGT format output
- VAE: standard mode, NGT format output  
- Rectified Flow: preference_trajectory mode

Also evaluates Ground Truth solutions for Pareto front comparison.

Outputs:
- Pareto front visualization with model predictions and ground truth
- Feasibility markers for each solution
- Complete metrics table (MAE, constraint satisfaction, etc.)
- Hypervolume calculation

Usage:
    python test.py                    # Evaluate all models
    python test.py --simple --vae     # Evaluate specific models
    python test.py --gt-only          # Evaluate ground truth only
    
Author: Peng Yue
Date: 2025-01
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt 

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'flow_model'))

# Import configuration from train_multi_preference
from train_multi_preference import MultiPreferenceConfig as BaseMultiPrefConfig


class TestMultiPrefConfig(BaseMultiPrefConfig):
    """Extended config for testing that supports both standard and preference_trajectory modes."""
    
    def __init__(self):
        super().__init__()
        # Training mode: 'standard' or 'preference_trajectory'
        self.multi_pref_training_mode = os.environ.get('MULTI_PREF_TRAINING_MODE', 'standard')
    
    def print_config(self):
        """Print configuration summary."""
        super().print_config()
        print(f"  Training Mode: {self.multi_pref_training_mode}")


def get_multi_preference_config():
    """Get test configuration with multi_pref_training_mode support."""
    return TestMultiPrefConfig()


# Alias for backward compatibility
MultiPreferenceConfig = TestMultiPrefConfig
from models import NetV
from data_loader import load_multi_preference_dataset

# Import unified evaluation framework
from unified_eval import (
    build_ctx_from_multi_preference,
    MultiPreferencePredictor,
    evaluate_unified, extract_summary_metrics,
    compute_pareto_hypervolumes,
    print_metrics_table,
    save_evaluation_results,
    reconstruct_full_from_partial,
    get_genload, get_vioPQg, get_viobran2,
    _as_numpy, _as_torch,
)

from utils import get_carbon_emission_vectorized


# ==================== Ground Truth Predictor ====================

class GroundTruthPredictor:
    """
    Predictor that returns ground truth solutions for evaluation.
    Used to evaluate constraint satisfaction and cost/carbon of optimal solutions.
    """
    
    def __init__(self, y_gt_ngt: torch.Tensor, multi_pref_data: dict):
        """
        Args:
            y_gt_ngt: Ground truth solutions in NGT format [N, output_dim]
            multi_pref_data: Multi-preference data dict (for reconstruction params)
        """
        self.y_gt_ngt = y_gt_ngt
        self.multi_pref_data = multi_pref_data
    
    def predict(self, ctx):
        """Return ground truth as prediction."""
        from unified_eval import PredPack
        
        y_np = _as_numpy(self.y_gt_ngt)
        Pred_Vm_full, Pred_Va_full = reconstruct_full_from_partial(ctx, y_np)
        
        return PredPack(
            Pred_Vm_full=Pred_Vm_full,
            Pred_Va_full=Pred_Va_full,
            time_vm=0.0,
            time_va=0.0,
            time_nn_total=0.0,
        )


# ==================== Model Loading Functions ====================

def load_multi_pref_model(config, model_type, multi_pref_data, device, training_mode=None):
    """
    Load a model trained with train_multi_preference.py.
    
    Args:
        config: MultiPreferenceConfig
        model_type: 'simple', 'vae', or 'rectified'
        multi_pref_data: Multi-preference data dict
        device: torch device
        training_mode: 'standard' or 'preference_trajectory' (overrides config if provided)
    
    Returns:
        model: Loaded model
        pretrain_model: Pretrained VAE model (for rectified flow in preference_trajectory mode)
        actual_training_mode: The actual training mode used
    """
    from net_utiles import FM, VAE
    
    input_dim = multi_pref_data['input_dim']
    output_dim = multi_pref_data['output_dim']
    pref_dim = config.pref_dim
    
    Vscale = multi_pref_data['Vscale']
    Vbias = multi_pref_data['Vbias']
    
    pretrain_model = None
    
    # Use provided training_mode or get from config
    actual_training_mode = training_mode or getattr(config, 'multi_pref_training_mode', 'standard')
    
    if model_type == 'simple':
        # Simple MLP: NetV with preference concatenated to input
        # Output is NGT format: [Va_nonZIB_noslack, Vm_nonZIB]
        model = NetV(
            input_dim + pref_dim, output_dim,
            config.ngt_hidden_units, config.ngt_khidden,
            Vscale, Vbias
        )
        print(f"    Model: NetV (MLP with sigmoid scaling)")
        print(f"    Input: {input_dim} + {pref_dim} (pref) = {input_dim + pref_dim}")
        print(f"    Output: {output_dim} (NGT format)")
        
    elif model_type == 'vae':
        # VAE model with preference conditioning
        vae_args = dict(
            output_dim=output_dim, hidden_dim=config.hidden_dim,
            num_layers=config.num_layers, latent_dim=config.latent_dim,
            output_act=None, pred_type='node', use_cvae=True
        )
        if config.vae_use_preference_aware:
            model = VAE(network='preference_aware_mlp', input_dim=input_dim, pref_dim=pref_dim, **vae_args)
            print(f"    Model: VAE (preference_aware_mlp with FiLM conditioning)")
        else:
            model = VAE(network='mlp', input_dim=input_dim + pref_dim, **vae_args)
            print(f"    Model: VAE (MLP with concatenated preference)")
        print(f"    Latent dim: {config.latent_dim}")
        
    elif model_type in ['rectified', 'gaussian', 'conditional', 'interpolation']:
        # Flow model with preference-aware MLP
        model = FM(
            network='preference_aware_mlp', input_dim=input_dim, output_dim=output_dim,
            hidden_dim=config.hidden_dim, num_layers=config.num_layers,
            time_step=config.time_step, output_norm=False, pred_type='velocity', pref_dim=pref_dim
        )
        print(f"    Model: Flow Matching ({model_type})")
        print(f"    Training mode: {actual_training_mode}")
        
        # Load pretrained VAE for preference_trajectory mode (used as anchor)
        if actual_training_mode == 'preference_trajectory':
            pretrain_model_path = os.path.join(config.model_save_dir, "model_multi_pref_vae_final.pth")
            if os.path.exists(pretrain_model_path):
                vae_args = dict(
                    output_dim=output_dim, hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers, latent_dim=config.latent_dim,
                    output_act=None, pred_type='node', use_cvae=True
                )
                pretrain_model = VAE(network='preference_aware_mlp', input_dim=input_dim, pref_dim=pref_dim, **vae_args)
                pretrain_model.load_state_dict(torch.load(pretrain_model_path, map_location=device, weights_only=True))
                pretrain_model.to(device)
                pretrain_model.eval()
                print(f"    Loaded pretrained VAE anchor: {pretrain_model_path}")
            else:
                print(f"    [WARNING] Pretrained VAE not found: {pretrain_model_path}")
                print(f"              Flow model will use random initialization as anchor")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load model weights
    # For rectified models, use new naming: model_multi_pref_rectified_traj_{cbf_tag}_final.pth
    if model_type == 'rectified':
        # Try to get CBF tag from config
        use_cbf = getattr(config, 'multi_pref_use_cbf_qp_train', None)
        if use_cbf is not None:
            # Config has CBF settings, generate tag
            if use_cbf:
                beta = getattr(config, 'multi_pref_cbf_beta', 0.5)
                cbf_tag = f"cbf{beta:.1f}".replace('.', '')
            else:
                cbf_tag = "nocbf"
            model_path = os.path.join(config.model_save_dir, f"model_multi_pref_rectified_traj_{cbf_tag}_final.pth")
        else:
            # Config doesn't have CBF settings, try both possible paths
            # First try cbf05 (most common)
            model_path = os.path.join(config.model_save_dir, "model_multi_pref_rectified_traj_cbf05_final.pth")
            if not os.path.exists(model_path):
                # Fallback to nocbf
                model_path = os.path.join(config.model_save_dir, "model_multi_pref_rectified_traj_nocbf_final.pth")
    else:
        # For other model types (simple, vae), use old naming
        model_path = os.path.join(config.model_save_dir, f"model_multi_pref_{model_type}_final.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    print(f"    Loaded model: {model_path}")
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, pretrain_model, actual_training_mode


# ==================== Evaluation Functions ====================

def evaluate_model_on_lambdas(config, model, multi_pref_data, sys_data, BRANFT, device,
                               model_type, lambdas, pretrain_model=None, training_mode='standard', verbose=False):
    """
    Evaluate a model across multiple lambda values.
    
    Args:
        config: Configuration object
        model: Trained model
        multi_pref_data: Multi-preference data dict
        sys_data: Power system data
        BRANFT: Branch from-to indices
        device: torch device
        model_type: 'simple', 'vae', or 'rectified'
        lambdas: List of lambda_carbon values to evaluate
        pretrain_model: Pretrained VAE model (for rectified flow)
        training_mode: 'standard' or 'preference_trajectory'
        verbose: Print detailed evaluation info
    
    Returns:
        List of result dicts for each lambda
    """
    results = []
    
    for lc in lambdas:
        print(f"\n    lambda_carbon = {lc:.2f}")
        
        ctx = build_ctx_from_multi_preference(
            config, sys_data, multi_pref_data, BRANFT, device, lambda_carbon=lc
        )
        
        predictor = MultiPreferencePredictor(
            model=model,
            multi_pref_data=multi_pref_data,
            lambda_carbon=lc,
            model_type=model_type,
            num_flow_steps=config.multi_pref_flow_steps,
            training_mode=training_mode,
            pretrain_model=pretrain_model,
            vae_use_mean=True,  # Use mean for evaluation (deterministic)
        )
        
        eval_result = evaluate_unified(ctx, predictor, apply_post_processing=True, verbose=verbose)
        
        # Compute lambda_cost from lambda_carbon
        lc_max = max(multi_pref_data['lambda_carbon_values'])
        lambda_cost = 1.0 - (lc / lc_max) if lc_max > 0 else 1.0
        
        summary = extract_summary_metrics(
            eval_result, f"{model_type.upper()}_lc{lc:.0f}",
            category=model_type,
            lambda_cost=lambda_cost,
            use_post_processed=True
        )
        summary['lambda_carbon'] = lc
        summary['training_mode'] = training_mode
        results.append(summary)
        
        print(f"      Cost: {summary['cost_mean']:.2f}, Carbon: {summary['carbon_mean']:.4f}")
        print(f"      Pg: {summary['Pg_satisfy']:.1f}%, Qg: {summary['Qg_satisfy']:.1f}%, Vm: {summary['Vm_satisfy']:.1f}%")
        # Print power balance satisfaction rate
        mre_Pd = summary.get('mre_Pd', 100.0)
        mre_Qd = summary.get('mre_Qd', 100.0)
        min_satisfaction = 99.5  # 100 - 0.5% error threshold
        if mre_Pd < min_satisfaction or mre_Qd < min_satisfaction:
            print(f"      [WARNING] Power Balance Satisfaction: Pd={mre_Pd:.2f}%, Qd={mre_Qd:.2f}% (need >= {min_satisfaction}%)")
    
    return results


def evaluate_ground_truth(config, multi_pref_data, sys_data, BRANFT, device, lambdas, verbose=False):
    """
    Evaluate ground truth solutions to get cost/carbon and feasibility.
    
    Returns:
        List of result dicts for each lambda
    """
    results = []
    y_val_by_pref = multi_pref_data['y_val_by_pref']
    
    for lc in lambdas:
        if lc not in y_val_by_pref:
            print(f"    [SKIP] lambda_carbon={lc:.2f} not in validation set")
            continue
        
        print(f"\n    lambda_carbon = {lc:.2f}")
        
        ctx = build_ctx_from_multi_preference(
            config, sys_data, multi_pref_data, BRANFT, device, lambda_carbon=lc
        )
        
        y_gt = y_val_by_pref[lc].to(device)
        predictor = GroundTruthPredictor(y_gt, multi_pref_data)
        
        eval_result = evaluate_unified(ctx, predictor, apply_post_processing=False, verbose=verbose)
        
        # Compute lambda_cost
        lc_max = max(multi_pref_data['lambda_carbon_values'])
        lambda_cost = 1.0 - (lc / lc_max) if lc_max > 0 else 1.0
        
        summary = extract_summary_metrics(
            eval_result, f"GT_lc{lc:.0f}",
            category='ground_truth',
            lambda_cost=lambda_cost,
            use_post_processed=False  # GT doesn't need post-processing
        )
        summary['lambda_carbon'] = lc
        results.append(summary)
        
        print(f"      Cost: {summary['cost_mean']:.2f}, Carbon: {summary['carbon_mean']:.4f}")
        print(f"      Pg: {summary['Pg_satisfy']:.1f}%, Qg: {summary['Qg_satisfy']:.1f}%, Vm: {summary['Vm_satisfy']:.1f}%")
        # Print power balance satisfaction rate (100 = perfect match)
        mre_Pd = summary.get('mre_Pd', 100.0)
        mre_Qd = summary.get('mre_Qd', 100.0)
        print(f"      Power Balance Satisfaction: Pd={mre_Pd:.2f}%, Qd={mre_Qd:.2f}%")
    
    return results


def compute_feasibility(result, thresholds=None):
    """
    Compute overall feasibility score for a result.
    
    IMPORTANT: Also checks power balance satisfaction rate (mre_Pd).
    mre_Pd is calculated as: 100 - |(Pred_Pd - Real_Pd) / Real_Pd| * 100
    This is a satisfaction rate (100 = perfect match, can be negative if error > 100%).
    If the satisfaction rate is too low, the solution is physically infeasible.
    
    Note: Qd satisfaction is not used for feasibility check because reactive power 
    calculation inherently has more error due to its strong dependence on voltage.
    
    Args:
        result: Result dict with constraint satisfaction metrics
        thresholds: Dict of thresholds for each constraint (default: 99.0%)
                   'power_balance_Pd' is the max allowed relative error (0.5% = 99.5% satisfaction)
    
    Returns:
        is_feasible: Boolean
        feasibility_score: Float (average constraint satisfaction)
    """
    if thresholds is None:
        thresholds = {
            'Pg': 99.0, 'Qg': 99.0, 'Vm': 95.0, 'branch': 99.0,
            'power_balance_Pd': 1.0  # Max 1.0% relative error = min 99.0% satisfaction
        }
    
    pg_sat = result.get('Pg_satisfy', 100.0)
    qg_sat = result.get('Qg_satisfy', 100.0)
    vm_sat = result.get('Vm_satisfy', 100.0)
    branch_sat = min(
        result.get('branch_ang_satisfy', 100.0),
        result.get('branch_pf_satisfy', 100.0)
    )
    
    # Check active power balance satisfaction rate
    # mre_Pd is now a satisfaction rate (100 - relative_error), higher is better
    # Only use Pd for feasibility check - Qd error is inherently larger
    mre_Pd = result.get('mre_Pd', 100.0)  # Satisfaction rate (100 = perfect, can be negative)
    # Convert threshold from error (0.5%) to satisfaction rate (99.5%)
    min_satisfaction = 100.0 - thresholds['power_balance_Pd']
    power_balance_ok = mre_Pd >= min_satisfaction
    
    is_feasible = (
        pg_sat >= thresholds['Pg'] and
        qg_sat >= thresholds['Qg'] and
        vm_sat >= thresholds['Vm'] and
        branch_sat >= thresholds['branch'] and
        power_balance_ok
    )
    
    feasibility_score = (pg_sat + qg_sat + vm_sat + branch_sat) / 4.0
    
    return is_feasible, feasibility_score


# ==================== Visualization Functions ====================

def plot_pareto_front_with_gt(all_results, ref_point, hypervolumes, save_path, title_suffix=""):
    """
    Plot Pareto front with ground truth, feasibility markers, and model comparison.
    
    Args:
        all_results: List of result dicts
        ref_point: Reference point for hypervolume
        hypervolumes: Dict of hypervolume values
        save_path: Path to save figure
        title_suffix: Additional text for the title
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Compute axis limits based on GT or feasible solutions for better visibility
    gt_results = [r for r in all_results if r.get('category') == 'ground_truth']
    feasible_results = [r for r in all_results if compute_feasibility(r)[0]]
    
    # Determine which results to use for axis limits
    if gt_results:
        limit_results = gt_results
    elif feasible_results:
        limit_results = feasible_results
    else:
        limit_results = all_results
    
    limit_costs = np.array([r['cost_mean'] for r in limit_results])
    limit_carbons = np.array([r['carbon_mean'] for r in limit_results])
    
    # Calculate axis limits with 10% margin
    cost_range = limit_costs.max() - limit_costs.min()
    carbon_range = limit_carbons.max() - limit_carbons.min()
    
    # Ensure minimum range to avoid zero division
    cost_range = max(cost_range, limit_costs.mean() * 0.1)
    carbon_range = max(carbon_range, limit_carbons.mean() * 0.1)
    
    x_min = limit_costs.min() - cost_range * 0.1
    x_max = limit_costs.max() + cost_range * 0.15
    y_min = limit_carbons.min() - carbon_range * 0.1
    y_max = limit_carbons.max() + carbon_range * 0.15
    
    # Define styles for each category
    category_styles = {
        'ground_truth': {'color': '#FFD700', 'marker': '*', 'size': 400, 'label': 'Ground Truth (OPF)'},
        'simple': {'color': '#E74C3C', 'marker': 's', 'size': 200, 'label': 'Simple MLP (standard)'},
        'vae': {'color': '#3498DB', 'marker': 'o', 'size': 200, 'label': 'VAE (standard)'},
        'rectified': {'color': '#27AE60', 'marker': '^', 'size': 200, 'label': 'Rectified Flow'},
    }
    
    # Group results by category
    categories = {}
    for r in all_results:
        cat = r.get('category', 'unknown')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)
    
    # Plot each category
    legend_handles = []
    
    for cat in ['ground_truth', 'simple', 'vae', 'rectified']:
        if cat not in categories:
            continue
        
        style = category_styles.get(cat, {'color': 'gray', 'marker': 'x', 'size': 150, 'label': cat}).copy()
        cat_results = categories[cat]
        
        # For rectified, add training mode to label
        if cat == 'rectified' and cat_results:
            mode = cat_results[0].get('training_mode', 'unknown')
            if mode != 'unknown':
                style['label'] = f"Rectified Flow ({mode})"
        
        costs = np.array([r['cost_mean'] for r in cat_results])
        carbons = np.array([r['carbon_mean'] for r in cat_results])
        
        # Compute feasibility for each point
        feasibility = [compute_feasibility(r) for r in cat_results]
        feasible_mask = np.array([f[0] for f in feasibility])
        
        # Plot feasible points (filled)
        if np.any(feasible_mask):
            scatter_f = ax.scatter(
                costs[feasible_mask], carbons[feasible_mask],
                c=style['color'], marker=style['marker'], s=style['size'],
                label=f"{style['label']} (feasible)", zorder=4 if cat == 'ground_truth' else 3,
                edgecolors='black', linewidths=1.5, alpha=0.9
            )
            legend_handles.append(scatter_f)
        
        # Plot infeasible points (hollow)
        if np.any(~feasible_mask):
            scatter_inf = ax.scatter(
                costs[~feasible_mask], carbons[~feasible_mask],
                c='white', marker=style['marker'], s=style['size'],
                label=f"{style['label']} (infeasible)", zorder=3,
                edgecolors=style['color'], linewidths=2.5, alpha=0.7
            )
            legend_handles.append(scatter_inf)
        
        # Connect points to show Pareto front (sorted by cost)
        if len(costs) > 1:
            sorted_idx = np.argsort(costs)
            ax.plot(
                costs[sorted_idx], carbons[sorted_idx],
                color=style['color'], linestyle='--', alpha=0.4, linewidth=2
            )
        
        # Add annotations for ground truth
        if cat == 'ground_truth':
            for r, cost, carbon in zip(cat_results, costs, carbons):
                lc = r.get('lambda_carbon', 0)
                ax.annotate(
                    f"lc={lc:.0f}", (cost, carbon),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=8, alpha=0.7, fontweight='medium'
                )
    
    # Plot reference point (only if within axis limits, otherwise add annotation)
    if ref_point[0] <= x_max and ref_point[1] <= y_max:
        ax.scatter(
            ref_point[0], ref_point[1], c='gray', marker='X', s=250,
            label='Reference Point', zorder=2, edgecolors='black', linewidths=1.5
        )
    else:
        # Reference point is outside the plot - add text annotation
        ax.annotate(
            f'Ref Point\n({ref_point[0]:.0f}, {ref_point[1]:.2f})',
            xy=(x_max * 0.95, y_max * 0.95),
            fontsize=9, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7)
        )
    
    # Labels and formatting
    ax.set_xlabel('Economic Cost ($/h)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Carbon Emission (tCO2/h)', fontsize=14, fontweight='bold')
    
    # Build title with training mode info from results
    training_modes = set()
    for r in all_results:
        mode = r.get('training_mode', 'unknown')
        if mode != 'unknown' and r.get('category') not in ['ground_truth']:
            training_modes.add(mode)
    mode_str = ", ".join(training_modes) if training_modes else ""
    
    title = f'Pareto Front: Multi-Preference Models vs Ground Truth\n(Filled = Feasible, Hollow = Infeasible){title_suffix}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set axis limits based on GT/feasible solutions for better visibility
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Add hypervolume text box
    hv_text = "Hypervolumes:\n"
    for cat in ['ground_truth', 'simple', 'vae', 'rectified', 'all']:
        if cat in hypervolumes:
            cat_name = category_styles.get(cat, {}).get('label', cat)
            hv_text += f"  {cat_name}: {hypervolumes[cat]:.2f}\n"
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.02, 0.98, hv_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, fontfamily='monospace')
    
    # Add feasibility legend
    feas_text = "Feasibility Thresholds:\n"
    feas_text += "  Pg, Qg, Branch >= 99%\n"
    feas_text += "  Vm >= 95%\n"
    feas_text += "  Pd Balance Sat >= 99.0% (100 - relative_error)\n"
    feas_text += "\nNote: Solutions with low cost\n"
    feas_text += "but low power balance sat or\n"
    feas_text += "Vm violations are infeasible."
    ax.text(0.02, 0.72, feas_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, fontfamily='monospace')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPareto front saved to: {save_path}")
    plt.close()


def print_comparison_table(all_results):
    """Print a comparison table of all results."""
    print("\n" + "=" * 140)
    print(" Comparison Table: Cost vs Carbon vs Feasibility (including Power Balance)")
    print("=" * 140)
    
    header = f"{'Model':<25} {'Category':<12} {'lc':<6} {'Cost ($/h)':<12} {'Carbon':<10} {'Pg%':<7} {'Qg%':<7} {'Vm%':<7} {'Pd_sat%':<9} {'Qd_sat%':<9} {'Feasible':<10}"
    print(header)
    print("-" * 140)
    
    for r in sorted(all_results, key=lambda x: (x.get('category', 'z'), x.get('lambda_carbon', 0))):
        is_feas, _ = compute_feasibility(r)
        feas_str = "Yes" if is_feas else "No"
        lc = r.get('lambda_carbon', 0)
        mre_Pd = r.get('mre_Pd', 100.0)  # Satisfaction rate (100 = perfect)
        mre_Qd = r.get('mre_Qd', 100.0)  # Satisfaction rate (100 = perfect)
        
        print(f"{r['name']:<25} {r.get('category', 'unknown'):<12} {lc:<6.0f} "
              f"{r['cost_mean']:<12.2f} {r['carbon_mean']:<10.4f} "
              f"{r['Pg_satisfy']:<7.1f} {r['Qg_satisfy']:<7.1f} {r['Vm_satisfy']:<7.1f} "
              f"{mre_Pd:<9.2f} {mre_Qd:<9.2f} {feas_str:<10}")
    
    print("-" * 140)
    print("\nNote: Pd_sat% and Qd_sat% show power balance satisfaction rate (100 = perfect match).")
    print("      Low satisfaction (<99.0%) means the solution doesn't satisfy power flow equations.")
    print("      A solution with low cost but low power balance satisfaction is NOT a valid OPF solution.")


# ==================== Main Function ====================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Multi-Preference Model Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test.py                           # Evaluate all models (simple, vae, rectified, gt)
    python test.py --simple --vae            # Evaluate only Simple and VAE models
    python test.py --rectified --training-mode preference_trajectory  # Rectified with specific mode
    python test.py --gt-only                 # Evaluate ground truth only
    python test.py --lambdas 0,50,100        # Evaluate on specific lambda values
        """
    )
    
    parser.add_argument('--simple', action='store_true', help='Evaluate Simple (MLP) model (standard mode)')
    parser.add_argument('--vae', action='store_true', help='Evaluate VAE model (standard mode)')
    parser.add_argument('--rectified', action='store_true', help='Evaluate Rectified Flow model')
    parser.add_argument('--gt', '--ground-truth', action='store_true', dest='gt', help='Evaluate Ground Truth')
    parser.add_argument('--gt-only', action='store_true', help='Evaluate Ground Truth only')
    parser.add_argument('--all', '-a', action='store_true', help='Evaluate all models')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--lambdas', type=str, default='0,10,25,50,70,90,100',
                        help='Comma-separated lambda_carbon values to evaluate')
    parser.add_argument('--training-mode', type=str, default='preference_trajectory',
                        choices=['standard', 'preference_trajectory'],
                        help='Training mode for rectified flow model (default: preference_trajectory)')
    
    args = parser.parse_args()
    
    # If no model specified, evaluate all
    if not (args.simple or args.vae or args.rectified or args.gt or args.gt_only):
        args.all = True
    
    if args.all:
        args.simple = True
        args.vae = True
        args.rectified = True
        args.gt = True
    
    if args.gt_only:
        args.simple = False
        args.vae = False
        args.rectified = False
        args.gt = True
    
    # Parse lambda values
    args.lambda_values = [float(x) for x in args.lambdas.split(',')]
    
    return args


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("=" * 100)
    print(" Multi-Preference Model Evaluation & Pareto Front Analysis")
    print("=" * 100)
    
    # Load configuration
    config = get_multi_preference_config()
    device = config.device
    
    print(f"\nConfiguration:")
    print(f"  Nbus: {config.Nbus}")
    print(f"  Device: {device}")
    print(f"  Model directory: {config.model_save_dir}")
    print(f"  Lambda values: {args.lambda_values}")
    
    # Load data
    print("\nLoading multi-preference dataset...")
    multi_pref_data, sys_data = load_multi_preference_dataset(config)
    
    # Compute BRANFT
    BRANFT = torch.from_numpy(sys_data.branch[:, 0:2] - 1).long()
    
    # Filter lambda values to those available in the dataset
    available_lambdas = multi_pref_data['lambda_carbon_values']
    lambdas = [lc for lc in args.lambda_values if lc in available_lambdas]
    
    if not lambdas:
        print(f"[WARNING] None of the requested lambda values {args.lambda_values} are available.")
        print(f"          Available values: {available_lambdas[:10]}...")
        lambdas = available_lambdas[:7]  # Use first 7
    
    print(f"  Evaluating on lambda_carbon: {lambdas}")
    
    all_results = []
    
    # ============================================================
    # 1. Evaluate Ground Truth
    # ============================================================
    if args.gt:
        print("\n" + "=" * 70)
        print(" 1. Evaluating Ground Truth (OPF Solutions)")
        print("=" * 70)
        
        gt_results = evaluate_ground_truth(
            config, multi_pref_data, sys_data, BRANFT, device, lambdas, verbose=args.verbose
        )
        all_results.extend(gt_results)
        print(f"\n  Evaluated {len(gt_results)} ground truth solutions")
    
    # ============================================================
    # 2. Evaluate Simple (MLP) Model (standard mode)
    # ============================================================
    if args.simple:
        print("\n" + "=" * 70)
        print(" 2. Evaluating Simple (MLP) Model (standard mode)")
        print("=" * 70)
        
        try:
            model, _, training_mode = load_multi_pref_model(
                config, 'simple', multi_pref_data, device, training_mode='standard'
            )
            simple_results = evaluate_model_on_lambdas(
                config, model, multi_pref_data, sys_data, BRANFT, device,
                'simple', lambdas, training_mode='standard', verbose=args.verbose
            )
            all_results.extend(simple_results)
            print(f"\n  Evaluated {len(simple_results)} Simple model predictions")
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
    
    # ============================================================
    # 3. Evaluate VAE Model (standard mode)
    # ============================================================
    if args.vae:
        print("\n" + "=" * 70)
        print(" 3. Evaluating VAE Model (standard mode)")
        print("=" * 70)
        
        try:
            model, _, training_mode = load_multi_pref_model(
                config, 'vae', multi_pref_data, device, training_mode='standard'
            )
            vae_results = evaluate_model_on_lambdas(
                config, model, multi_pref_data, sys_data, BRANFT, device,
                'vae', lambdas, training_mode='standard', verbose=args.verbose
            )
            all_results.extend(vae_results)
            print(f"\n  Evaluated {len(vae_results)} VAE model predictions")
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
    
    # ============================================================
    # 4. Evaluate Rectified Flow Model (preference_trajectory mode)
    # ============================================================
    if args.rectified:
        # Use training_mode from args (defaults to preference_trajectory)
        rectified_training_mode = args.training_mode
        
        print("\n" + "=" * 70)
        print(f" 4. Evaluating Rectified Flow Model ({rectified_training_mode} mode)")
        print("=" * 70)
        
        try:
            model, pretrain_model, actual_mode = load_multi_pref_model(
                config, 'rectified', multi_pref_data, device, 
                training_mode=rectified_training_mode
            )
            flow_results = evaluate_model_on_lambdas(
                config, model, multi_pref_data, sys_data, BRANFT, device,
                'rectified', lambdas, pretrain_model=pretrain_model, 
                training_mode=actual_mode, verbose=args.verbose
            )
            all_results.extend(flow_results)
            print(f"\n  Evaluated {len(flow_results)} Rectified Flow predictions ({actual_mode})")
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
    
    # ============================================================
    # 5. Results Analysis
    # ============================================================
    if len(all_results) == 0:
        print("\n[ERROR] No models were successfully evaluated!")
        print("Please check model paths and run training first.")
        return
    
    # Print comparison table
    print_comparison_table(all_results)
    
    # Print complete metrics table
    print_metrics_table(all_results, "Complete Evaluation Metrics")
    
    # ============================================================
    # 6. Compute Hypervolumes
    # ============================================================
    print("\n" + "=" * 70)
    print(" Pareto Front Analysis & Hypervolume")
    print("=" * 70)
    
    costs = np.array([r['cost_mean'] for r in all_results])
    carbons = np.array([r['carbon_mean'] for r in all_results])
    
    # Compute reference point based on FEASIBLE solutions only (or GT if available)
    # This prevents outliers from distorting the plot
    gt_results = [r for r in all_results if r.get('category') == 'ground_truth']
    feasible_results = [r for r in all_results if compute_feasibility(r)[0]]
    
    if gt_results:
        # Use Ground Truth range as reference (most reliable)
        ref_costs = np.array([r['cost_mean'] for r in gt_results])
        ref_carbons = np.array([r['carbon_mean'] for r in gt_results])
        print(f"\n  Using Ground Truth range for reference point")
    elif feasible_results:
        # Fall back to feasible solutions
        ref_costs = np.array([r['cost_mean'] for r in feasible_results])
        ref_carbons = np.array([r['carbon_mean'] for r in feasible_results])
        print(f"\n  Using feasible solutions for reference point")
    else:
        # Last resort: use all results
        ref_costs = costs
        ref_carbons = carbons
        print(f"\n  [WARNING] No feasible solutions, using all results for reference point")
    
    # Add small margin (5%) instead of 10% to keep points visible
    ref_point = np.array([
        np.max(ref_costs) * 1.05,
        np.max(ref_carbons) * 1.05
    ])
    print(f"  Reference point: cost={ref_point[0]:.2f}, carbon={ref_point[1]:.4f}")
    
    # Compute hypervolumes for each category
    hypervolumes = {}
    for cat in ['ground_truth', 'simple', 'vae', 'rectified']:
        cat_results = [r for r in all_results if r.get('category') == cat]
        if cat_results:
            cat_costs = np.array([r['cost_mean'] for r in cat_results])
            cat_carbons = np.array([r['carbon_mean'] for r in cat_results])
            
            # Simple hypervolume approximation
            points = np.column_stack([cat_costs, cat_carbons])
            sorted_idx = np.argsort(points[:, 0])
            points = points[sorted_idx]
            
            hv = 0.0
            prev_carbon = ref_point[1]
            for cost, carbon in points:
                if carbon < prev_carbon:
                    hv += (ref_point[0] - cost) * (prev_carbon - carbon)
                    prev_carbon = carbon
            
            hypervolumes[cat] = hv
            print(f"  Hypervolume ({cat}): {hv:.2f}")
    
    # Total hypervolume
    all_points = np.column_stack([costs, carbons])
    sorted_idx = np.argsort(all_points[:, 0])
    all_points = all_points[sorted_idx]
    
    hv_all = 0.0
    prev_carbon = ref_point[1]
    for cost, carbon in all_points:
        if carbon < prev_carbon:
            hv_all += (ref_point[0] - cost) * (prev_carbon - carbon)
            prev_carbon = carbon
    hypervolumes['all'] = hv_all
    print(f"  Hypervolume (all): {hv_all:.2f}")
    
    # ============================================================
    # 7. Plot Pareto Front
    # ============================================================
    plot_pareto_front_with_gt(
        all_results, ref_point, hypervolumes,
        save_path=f'{config.results_dir}/pareto_front_multi_preference.png'
    )
    
    # ============================================================
    # 8. Save Results
    # ============================================================
    save_evaluation_results(
        all_results, hypervolumes, ref_point,
        f'{config.results_dir}/multi_preference_evaluation_results.json',
        config=config
    )
    
    print("\n" + "=" * 100)
    print(" Evaluation completed!")
    print("=" * 100)
    
    return all_results


if __name__ == "__main__":
    main()
