#!/usr/bin/env python
# coding: utf-8
"""
GT Pareto Front Test (Ablation Study)

This script tests Flow models' ability to track the Pareto front
when starting from Ground Truth at λ=0:
- Start from GT at λ=0
- Integrate to all λ values using Flow model
- Compare predicted vs GT to measure tracking accuracy

This is useful for ablation studies to evaluate the Flow model's
trajectory integration quality independent of the anchor model.

Usage:
    python test_virtual_segment.py                       # Test with basic TFM model
    python test_virtual_segment.py --model-variant refiner_v2  # Test Refiner V2 Flow
    python test_virtual_segment.py --model-path /path/to/model.pth  # Custom model
    python test_virtual_segment.py --flow-steps 200      # More ODE integration steps
    
Author: Peng Yue
Date: January 2026
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'flow_model'))

from train_multi_preference_tfm import MultiPreferenceConfig
from data_loader import load_multi_preference_dataset
from unified_eval import (
    build_ctx_from_multi_preference,
    evaluate_unified,
    extract_summary_metrics,
    reconstruct_full_from_partial,
    _as_numpy,
    PredPack,
)


# ==================== Configuration ====================

def get_config():
    """Get configuration."""
    return MultiPreferenceConfig()


# ==================== Helper Classes ====================

class DirectPredictor:
    """Simple predictor that returns pre-computed Vm/Va."""
    def __init__(self, Vm_full, Va_full):
        self.Vm_full = Vm_full
        self.Va_full = Va_full
    
    def predict(self, ctx):
        return PredPack(
            Pred_Vm_full=self.Vm_full,
            Pred_Va_full=self.Va_full,
            time_vm=0.0, time_va=0.0, time_nn_total=0.0
        )


# ==================== Model Variant Configurations ====================

MODEL_VARIANTS = {
    'basic': {
        'filename': 'model_multi_pref_rectified_traj_tfm_final.pth',
        'display_name': 'Basic TFM',
        'train_script': 'train_multi_preference_tfm.py'
    },
    'refiner_v2': {
        'filename': 'model_multi_pref_refiner_v2_flow_final.pth',
        'display_name': 'Refiner V2 (3-Stage)',
        'train_script': 'train_multi_preference_tfm_refiner_v2.py'
    },
}


# ==================== Test GT Pareto Front ====================

def test_gt_pareto_front(config, multi_pref_data, sys_data, BRANFT, device, 
                         num_steps=100, verbose=True, model_variant='basic', model_path=None):
    """
    Test Pareto front generation from GT at λ=0 (Ablation Study).
    
    This tests different Flow models' ability to track the Pareto front:
    - Start from Ground Truth at λ=0
    - Integrate to all λ values to generate Pareto front
    - Compare predicted vs GT to measure tracking accuracy
    
    Args:
        config: Configuration object
        multi_pref_data: Multi-preference data dict
        sys_data: Power system data
        BRANFT: Branch from-to indices
        device: torch device
        num_steps: Number of ODE integration steps
        verbose: Print detailed output
        model_variant: 'basic' | 'refiner_v2'
        model_path: Custom model path (overrides model_variant)
    
    Returns:
        dict: Results including MAE, constraint satisfaction, Pareto front metrics
    """
    from net_utiles import FM
    
    # Determine model path and display name
    if model_path:
        flow_path = model_path
        display_name = f"Custom ({os.path.basename(model_path)})"
    else:
        variant_info = MODEL_VARIANTS.get(model_variant, MODEL_VARIANTS['basic'])
        flow_path = os.path.join(config.model_save_dir, variant_info['filename'])
        display_name = variant_info['display_name']
    
    print("\n" + "=" * 70)
    print(f" Test Pareto Front from GT at λ=0 [{display_name}]")
    print("=" * 70)
    
    input_dim = multi_pref_data['input_dim']
    output_dim = multi_pref_data['output_dim']
    pref_dim = config.pref_dim
    
    # ===== Load Flow Model =====
    print(f"\n[1] Loading {display_name} model...")
    
    flow_model = FM(
        network='preference_aware_mlp', input_dim=input_dim, output_dim=output_dim,
        hidden_dim=config.hidden_dim, num_layers=config.num_layers,
        time_step=config.time_step, output_norm=False, pred_type='velocity', pref_dim=pref_dim
    )
    
    if not os.path.exists(flow_path):
        if model_path:
            raise FileNotFoundError(f"Custom model not found: {flow_path}")
        else:
            variant_info = MODEL_VARIANTS.get(model_variant, MODEL_VARIANTS['basic'])
            raise FileNotFoundError(f"{display_name} model not found: {flow_path}\n"
                                   f"Please train with: python main_part/{variant_info['train_script']}")
    
    flow_model.load_state_dict(torch.load(flow_path, map_location=device, weights_only=True))
    flow_model.to(device).eval()
    print(f"  Model path: {flow_path}")
    print(f"  Parameters: {sum(p.numel() for p in flow_model.parameters()):,}")
    
    # ===== Get Data =====
    print("\n[2] Preparing data...")
    
    x_val = multi_pref_data['x_val'].to(device)
    y_val_by_pref = multi_pref_data['y_val_by_pref']
    
    lambda_values = sorted(multi_pref_data['lambda_carbon_values'])
    lambda_min = lambda_values[0]
    lambda_max = lambda_values[-1]
    
    if lambda_min not in y_val_by_pref:
        raise ValueError(f"GT at λ={lambda_min} not available in validation set")
    
    y_gt_start = y_val_by_pref[lambda_min].to(device)  # GT at λ=0
    batch_size = x_val.shape[0]
    
    print(f"  Validation samples: {batch_size}")
    print(f"  Lambda range: [{lambda_min}, {lambda_max}]")
    print(f"  Starting from: GT at λ={lambda_min}")
    
    # ===== Build Pareto Front =====
    print(f"\n[3] Building Pareto front ({num_steps} ODE steps per segment)...")
    
    pareto_lambdas = sorted([lc for lc in lambda_values if lc in y_val_by_pref])
    print(f"  Testing λ values: {pareto_lambdas}")
    
    ctx = build_ctx_from_multi_preference(
        config, sys_data, multi_pref_data, BRANFT, device, lambda_carbon=lambda_min
    )
    
    pareto_results = []
    t_start = time.perf_counter()
    
    for target_lc in pareto_lambdas:
        # Normalize target lambda
        lambda_target_norm = (target_lc - lambda_min) / (lambda_max - lambda_min) \
            if lambda_max > lambda_min else 0.0
        
        # Integrate from λ=0 to target λ
        with torch.no_grad():
            if target_lc == lambda_min:
                # No integration needed
                y_pred = y_gt_start
            else:
                x_current = y_gt_start.clone()
                lambda_curr = torch.zeros((batch_size, 1), device=device)
                lambda_target_tensor = torch.full((batch_size, 1), lambda_target_norm, device=device)
                
                step_dlambda = (lambda_target_tensor - lambda_curr) / num_steps
                
                for step in range(num_steps):
                    v = flow_model.predict_vec(x_val, x_current, lambda_curr, lambda_curr)
                    x_current = x_current + step_dlambda * v
                    lambda_curr = lambda_curr + step_dlambda
                
                y_pred = x_current
        
        # Get GT for this lambda
        y_gt_target = y_val_by_pref[target_lc].to(device)
        
        # Compute MAE
        mae_to_gt = torch.mean(torch.abs(y_pred - y_gt_target)).item()
        
        # Reconstruct and evaluate
        V_pred_np = _as_numpy(y_pred)
        Pred_Vm, Pred_Va = reconstruct_full_from_partial(ctx, V_pred_np)
        
        pred_predictor = DirectPredictor(Pred_Vm, Pred_Va)
        eval_pred = evaluate_unified(ctx, pred_predictor, apply_post_processing=False, verbose=False)
        metrics_pred = extract_summary_metrics(eval_pred, f"Flow_lc{target_lc}", use_post_processed=False)
        
        # Get GT metrics
        V_gt_np = _as_numpy(y_gt_target)
        GT_Vm, GT_Va = reconstruct_full_from_partial(ctx, V_gt_np)
        gt_predictor = DirectPredictor(GT_Vm, GT_Va)
        eval_gt = evaluate_unified(ctx, gt_predictor, apply_post_processing=False, verbose=False)
        metrics_gt = extract_summary_metrics(eval_gt, f"GT_lc{target_lc}", use_post_processed=False)
        
        pareto_results.append({
            'lambda_carbon': target_lc,
            'mae_to_gt': mae_to_gt,
            'cost_pred': metrics_pred['cost_mean'],
            'carbon_pred': metrics_pred['carbon_mean'],
            'cost_gt': metrics_gt['cost_mean'],
            'carbon_gt': metrics_gt['carbon_mean'],
            'Pg_satisfy_pred': metrics_pred['Pg_satisfy'],
            'Qg_satisfy_pred': metrics_pred['Qg_satisfy'],
            'Pg_satisfy_gt': metrics_gt['Pg_satisfy'],
            'Qg_satisfy_gt': metrics_gt['Qg_satisfy'],
            'feasible_pred': metrics_pred['Pg_satisfy'] >= 99.0 and metrics_pred['Qg_satisfy'] >= 99.0,
            'feasible_gt': metrics_gt['Pg_satisfy'] >= 99.0 and metrics_gt['Qg_satisfy'] >= 99.0,
        })
        
        print(f"  λ={target_lc:5.1f}: MAE={mae_to_gt:.6f}, "
              f"Cost(pred/gt)={metrics_pred['cost_mean']:.1f}/{metrics_gt['cost_mean']:.1f}, "
              f"Carbon(pred/gt)={metrics_pred['carbon_mean']:.4f}/{metrics_gt['carbon_mean']:.4f}, "
              f"Feasible={pareto_results[-1]['feasible_pred']}")
    
    inference_time = time.perf_counter() - t_start
    
    # ===== Plot Pareto Front =====
    print("\n[4] Plotting Pareto front comparison...")
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    costs_pred = np.array([r['cost_pred'] for r in pareto_results])
    carbons_pred = np.array([r['carbon_pred'] for r in pareto_results])
    costs_gt = np.array([r['cost_gt'] for r in pareto_results])
    carbons_gt = np.array([r['carbon_gt'] for r in pareto_results])
    feasible_pred = np.array([r['feasible_pred'] for r in pareto_results])
    
    # Plot GT Pareto front
    ax.plot(costs_gt, carbons_gt, 'k--', alpha=0.5, linewidth=2, zorder=2)
    ax.scatter(costs_gt, carbons_gt, c='#FFD700', marker='*', s=300, 
               label='Ground Truth', edgecolors='black', linewidths=1.5, zorder=4)
    
    # Plot Flow predictions
    ax.plot(costs_pred, carbons_pred, color='#9B59B6', linestyle='-', alpha=0.6, linewidth=2, zorder=2)
    
    if np.any(feasible_pred):
        ax.scatter(costs_pred[feasible_pred], carbons_pred[feasible_pred], 
                   c='#9B59B6', marker='D', s=200, label=f'{display_name} (Feasible)', 
                   edgecolors='black', linewidths=1.5, zorder=3)
    
    if np.any(~feasible_pred):
        ax.scatter(costs_pred[~feasible_pred], carbons_pred[~feasible_pred], 
                   c='white', marker='D', s=200, label=f'{display_name} (Infeasible)', 
                   edgecolors='#E74C3C', linewidths=2.5, zorder=3)
    
    # Annotate lambda values
    for i, r in enumerate(pareto_results):
        if i % 2 == 0 or i == len(pareto_results) - 1:
            ax.annotate(f"λ={r['lambda_carbon']:.0f}", 
                       (r['cost_gt'], r['carbon_gt']),
                       textcoords="offset points", xytext=(8, 8), fontsize=8, alpha=0.7)
    
    # Compute statistics
    mae_avg = np.mean([r['mae_to_gt'] for r in pareto_results])
    cost_error_avg = np.mean(np.abs(costs_pred - costs_gt))
    carbon_error_avg = np.mean(np.abs(carbons_pred - carbons_gt))
    feasibility_rate = np.mean(feasible_pred) * 100
    
    ax.set_xlabel('Economic Cost ($/h)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Carbon Emission (tCO2/h)', fontsize=12, fontweight='bold')
    ax.set_title(f'Pareto Front: {display_name} (from GT at λ=0)\n'
                 f'Avg MAE={mae_avg:.6f}, Feasibility={feasibility_rate:.1f}%, ODE Steps={num_steps}', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Metrics text
    metrics_text = (
        f"{display_name} Metrics:\n"
        f"  Model: {os.path.basename(flow_path)}\n"
        f"  Avg MAE to GT:    {mae_avg:.6f}\n"
        f"  Avg Cost Error:   {cost_error_avg:.2f} $/h\n"
        f"  Avg Carbon Error: {carbon_error_avg:.6f} tCO2/h\n"
        f"  Feasibility Rate: {feasibility_rate:.1f}%\n"
        f"  Inference Time:   {inference_time:.2f}s"
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9)
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, fontfamily='monospace')
    
    plt.tight_layout()
    
    # Generate save filename based on model variant
    save_filename = f'pareto_from_gt_{model_variant}.png'
    save_path = os.path.join(config.results_dir, save_filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n  Plot saved to: {save_path}")
    plt.close()
    
    # Print summary
    print("\n" + "=" * 70)
    print(f" Summary: {display_name} from GT λ=0")
    print("=" * 70)
    print(f"  Average MAE to GT:     {mae_avg:.6f}")
    print(f"  Average Cost Error:    {cost_error_avg:.2f} $/h")
    print(f"  Average Carbon Error:  {carbon_error_avg:.6f} tCO2/h")
    print(f"  Feasibility Rate:      {feasibility_rate:.1f}% ({int(np.sum(feasible_pred))}/{len(feasible_pred)})")
    print(f"  Total Inference Time:  {inference_time:.2f}s")
    
    return {
        'model_variant': model_variant,
        'model_path': flow_path,
        'display_name': display_name,
        'pareto_results': pareto_results,
        'mae_avg': mae_avg,
        'cost_error_avg': cost_error_avg,
        'carbon_error_avg': carbon_error_avg,
        'feasibility_rate': feasibility_rate,
        'inference_time': inference_time,
        'num_steps': num_steps,
    }


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description='GT Pareto Front Test (Ablation Study)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_virtual_segment.py                              # Test basic TFM model
    python test_virtual_segment.py --model-variant refiner_v2   # Test Refiner V2 Flow
    python test_virtual_segment.py --model-path /path/to/model.pth  # Custom model
    python test_virtual_segment.py --flow-steps 200             # More ODE integration steps
        """
    )
    parser.add_argument('--model-variant', type=str, default='basic', choices=['basic', 'refiner_v2'],
                        help='Model variant: basic, refiner_v2')
    parser.add_argument('--model-path', type=str, default=None, help='Custom model path (overrides --model-variant)')
    parser.add_argument('--flow-steps', type=int, default=100, help='Number of ODE integration steps')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    print("=" * 70)
    print(" GT Pareto Front Test (Ablation Study)")
    print("=" * 70)
    
    config = get_config()
    device = config.device
    
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Flow ODE steps: {args.flow_steps}")
    
    # Load data
    print("\nLoading dataset...")
    multi_pref_data, sys_data = load_multi_preference_dataset(config)
    BRANFT = torch.from_numpy(sys_data.branch[:, 0:2] - 1).long()
    
    print(f"  Validation samples: {multi_pref_data['n_val']}")
    print(f"  Lambda values: {multi_pref_data['lambda_carbon_values']}")
    
    # Run test
    try:
        results = test_gt_pareto_front(
            config, multi_pref_data, sys_data, BRANFT, device,
            num_steps=args.flow_steps, verbose=args.verbose,
            model_variant=args.model_variant, model_path=args.model_path
        )
        print("\n" + "=" * 70)
        print(" GT Pareto Front Test Completed!")
        print("=" * 70)
        return results
    except FileNotFoundError as e:
        print(f"\n[ERROR] Cannot run test: {e}")
        return None


if __name__ == "__main__":
    main()
