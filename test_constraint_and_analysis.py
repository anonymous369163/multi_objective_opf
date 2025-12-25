#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test constraint calculation correctness and analyze prediction errors.

This script:
1. Verifies constraint calculation by replacing predictions with ground truth optimal values
2. Analyzes prediction errors (VAE anchor, Flow improved, vs ground truth)
3. Provides detailed comparison and visualization
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'main_part'))

from config import get_config
from data_loader import load_multi_preference_dataset, load_all_data
from unified_eval import (
    build_ctx_from_multi_preference,
    MultiPreferencePredictor,
    evaluate_unified,
    _as_numpy,
    _as_torch,
    _kron_reconstruct_zib,
    reconstruct_full_from_partial
)
from utils import (
    get_genload, get_vioPQg, get_viobran2, get_mae, get_rerr2,
    get_Pgcost, get_carbon_emission_vectorized
)

# Set matplotlib to use non-Chinese fonts
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def verify_constraint_calculation(ctx, Real_Vm_full, Real_Va_full, verbose=True):
    """
    Verify constraint calculation by using ground truth optimal values.
    
    If ground truth values satisfy constraints, then constraint calculation is correct.
    """
    print("\n" + "=" * 80)
    print("Constraint Calculation Verification")
    print("=" * 80)
    print("Testing with GROUND TRUTH optimal values...")
    
    # Use ground truth values
    Test_Vm = Real_Vm_full.copy()
    Test_Va = Real_Va_full.copy()
    Test_V = Test_Vm * np.exp(1j * Test_Va)
    
    # Compute power flow
    Test_Pg, Test_Qg, Test_Pd, Test_Qd = get_genload(
        Test_V, ctx.Pdtest, ctx.Qdtest, ctx.bus_Pg, ctx.bus_Qg, ctx.Ybus
    )
    
    # Check constraints
    lsPg, lsQg, lsidxPg, lsidxQg, vio_PQgmaxmin, vio_PQg, deltaPgL, deltaQgL, deltaPgU, deltaQgU = get_vioPQg(
        Test_Pg, ctx.bus_Pg, ctx.MAXMIN_Pg,
        Test_Qg, ctx.bus_Qg, ctx.MAXMIN_Qg,
        ctx.DELTA
    )
    
    vio_branang, vio_branpf, deltapf, vio_branpfidx, lsSf, _, lsSf_sampidx, _ = get_viobran2(
        Test_V, Test_Va, ctx.branch, ctx.Yf, ctx.Yt, ctx.BRANFT, ctx.baseMVA, ctx.DELTA
    )
    
    lsidxPQg = np.asarray(np.where((lsidxPg + lsidxQg) > 0)[0]).ravel()
    num_vio = int(lsidxPQg.size)
    
    # Results
    pg_satisfy = float(np.mean(_as_numpy(vio_PQg[:, 0])))
    qg_satisfy = float(np.mean(_as_numpy(vio_PQg[:, 1])))
    branch_angle_satisfy = float(np.mean(_as_numpy(vio_branang)))
    branch_pf_satisfy = float(np.mean(_as_numpy(vio_branpf)))
    
    print(f"\n[Results with Ground Truth Values]")
    print(f"  Violated samples: {num_vio}/{ctx.Ntest} ({num_vio/ctx.Ntest*100:.1f}%)")
    print(f"  Pg constraint satisfaction: {pg_satisfy:.2f}%")
    print(f"  Qg constraint satisfaction: {qg_satisfy:.2f}%")
    print(f"  Branch angle constraint: {branch_angle_satisfy:.2f}%")
    print(f"  Branch power flow constraint: {branch_pf_satisfy:.2f}%")
    
    # Verification
    if pg_satisfy > 99.0 and qg_satisfy > 99.0 and branch_angle_satisfy > 99.0 and branch_pf_satisfy > 99.0:
        print(f"\n[VERIFIED] Constraint calculation is CORRECT!")
        print(f"  Ground truth values satisfy constraints (>99%),")
        print(f"  so constraint calculation logic is working properly.")
        return True
    else:
        print(f"\n[WARNING] Ground truth values have constraint violations!")
        print(f"  This may indicate:")
        print(f"  1. Constraint calculation has bugs, OR")
        print(f"  2. Ground truth values are not truly optimal (OPF solver issues)")
        return False
    
    return True


def get_vae_anchor_predictions(ctx, pretrain_model, multi_pref_data, lambda_carbon, device):
    """Get VAE anchor predictions (before Flow improvement)."""
    pretrain_model.eval()
    
    x = ctx.x_test.to(device)
    Ntest = x.shape[0]
    
    # Get preference
    lambda_carbon_values = multi_pref_data.get('lambda_carbon_values', [55.0])
    lc_max = max(lambda_carbon_values) if max(lambda_carbon_values) > 0 else 1.0
    pref = torch.full((Ntest, 1), lambda_carbon / lc_max, device=device)
    
    # Get VAE prediction (in NGT format: partial voltage)
    with torch.no_grad():
        if hasattr(pretrain_model, 'pref_dim') and pretrain_model.pref_dim > 0:
            V_partial_vae = pretrain_model(x, use_mean=True, pref=pref)
        else:
            x_with_pref = torch.cat([x, pref], dim=1)
            V_partial_vae = pretrain_model(x_with_pref, use_mean=True)
    
    # Convert to numpy and reconstruct full voltage
    V_partial_vae = _as_numpy(V_partial_vae)
    Vm_vae_full, Va_vae_full = reconstruct_full_from_partial(ctx, V_partial_vae)
    
    return Vm_vae_full, Va_vae_full


# Use the existing reconstruct_full_from_partial from unified_eval


def analyze_prediction_errors(ctx, Vm_pred, Va_pred, Vm_real, Va_real, label="Prediction"):
    """Analyze prediction errors in detail."""
    print(f"\n[{label} Error Analysis]")
    
    # Voltage errors
    Vm_error = Vm_pred - Vm_real
    Va_error = Va_pred - Va_real
    
    Vm_mae = np.mean(np.abs(Vm_error))
    Va_mae = np.mean(np.abs(Va_error))
    Vm_rmse = np.sqrt(np.mean(Vm_error ** 2))
    Va_rmse = np.sqrt(np.mean(Va_error ** 2))
    
    print(f"  Vm MAE: {Vm_mae:.6f} p.u.")
    print(f"  Vm RMSE: {Vm_rmse:.6f} p.u.")
    print(f"  Vm Max Error: {np.max(np.abs(Vm_error)):.6f} p.u.")
    print(f"  Va MAE: {Va_mae:.6f} rad ({Va_mae*180/np.pi:.4f} deg)")
    print(f"  Va RMSE: {Va_rmse:.6f} rad ({Va_rmse*180/np.pi:.4f} deg)")
    print(f"  Va Max Error: {np.max(np.abs(Va_error)):.6f} rad ({np.max(np.abs(Va_error))*180/np.pi:.4f} deg)")
    
    # Power flow errors
    V_pred = Vm_pred * np.exp(1j * Va_pred)
    V_real = Vm_real * np.exp(1j * Va_real)
    
    Pred_Pg, Pred_Qg, Pred_Pd, Pred_Qd = get_genload(
        V_pred, ctx.Pdtest, ctx.Qdtest, ctx.bus_Pg, ctx.bus_Qg, ctx.Ybus
    )
    Real_Pg, Real_Qg, Real_Pd, Real_Qd = get_genload(
        V_real, ctx.Pdtest, ctx.Qdtest, ctx.bus_Pg, ctx.bus_Qg, ctx.Ybus
    )
    
    # Cost errors
    # Use _compute_cost from unified_eval which handles gencost_Pg correctly
    from unified_eval import _compute_cost
    Pred_cost = _compute_cost(Pred_Pg, ctx)
    Real_cost = _compute_cost(Real_Pg, ctx)
    cost_mre = np.mean(np.abs((Pred_cost - Real_cost) / (Real_cost + 1e-8))) * 100
    
    # Load satisfaction errors
    Pd_error = np.abs(Pred_Pd.sum(axis=1) - Real_Pd.sum(axis=1))
    Qd_error = np.abs(Pred_Qd.sum(axis=1) - Real_Qd.sum(axis=1))
    Pd_mre = np.mean(Pd_error / (np.abs(Real_Pd.sum(axis=1)) + 1e-8)) * 100
    Qd_mre = np.mean(Qd_error / (np.abs(Real_Qd.sum(axis=1)) + 1e-8)) * 100
    
    print(f"  Cost MRE: {cost_mre:.4f}%")
    print(f"  Pd MRE: {Pd_mre:.4f}%")
    print(f"  Qd MRE: {Qd_mre:.4f}%")
    
    return {
        'Vm_mae': Vm_mae,
        'Va_mae': Va_mae,
        'Vm_rmse': Vm_rmse,
        'Va_rmse': Va_rmse,
        'cost_mre': cost_mre,
        'Pd_mre': Pd_mre,
        'Qd_mre': Qd_mre,
        'Vm_error': Vm_error,
        'Va_error': Va_error,
        'Pred_Pg': Pred_Pg,
        'Real_Pg': Real_Pg
    }


def compare_vae_flow_groundtruth(ctx, Vm_vae, Va_vae, Vm_flow, Va_flow, Vm_real, Va_real, save_dir='results'):
    """Compare VAE anchor, Flow improved, and ground truth values."""
    print("\n" + "=" * 80)
    print("VAE vs Flow vs Ground Truth Comparison")
    print("=" * 80)
    
    # Analyze each
    stats_vae = analyze_prediction_errors(ctx, Vm_vae, Va_vae, Vm_real, Va_real, "VAE Anchor")
    stats_flow = analyze_prediction_errors(ctx, Vm_flow, Va_flow, Vm_real, Va_real, "Flow Improved")
    
    # Improvement from VAE to Flow
    print(f"\n[Flow Improvement over VAE]")
    vae_to_flow_vm = (stats_vae['Vm_mae'] - stats_flow['Vm_mae']) / stats_vae['Vm_mae'] * 100
    vae_to_flow_va = (stats_vae['Va_mae'] - stats_flow['Va_mae']) / stats_vae['Va_mae'] * 100
    vae_to_flow_cost = (stats_vae['cost_mre'] - stats_flow['cost_mre']) / (stats_vae['cost_mre'] + 1e-8) * 100
    
    print(f"  Vm MAE improvement: {vae_to_flow_vm:.2f}%")
    print(f"  Va MAE improvement: {vae_to_flow_va:.2f}%")
    print(f"  Cost MRE improvement: {vae_to_flow_cost:.2f}%")
    
    # Create visualizations
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Error distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Vm error distribution
    axes[0, 0].hist(stats_vae['Vm_error'].flatten(), bins=50, alpha=0.5, label='VAE', density=True)
    axes[0, 0].hist(stats_flow['Vm_error'].flatten(), bins=50, alpha=0.5, label='Flow', density=True)
    axes[0, 0].set_xlabel('Vm Error (p.u.)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Vm Error Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Va error distribution
    axes[0, 1].hist(stats_vae['Va_error'].flatten(), bins=50, alpha=0.5, label='VAE', density=True)
    axes[0, 1].hist(stats_flow['Va_error'].flatten(), bins=50, alpha=0.5, label='Flow', density=True)
    axes[0, 1].set_xlabel('Va Error (rad)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Va Error Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Vm scatter: VAE vs Flow
    sample_idx = np.random.choice(Vm_real.shape[0], min(100, Vm_real.shape[0]), replace=False)
    vae_vm_sample = Vm_vae[sample_idx].flatten()
    flow_vm_sample = Vm_flow[sample_idx].flatten()
    real_vm_sample = Vm_real[sample_idx].flatten()
    
    axes[1, 0].scatter(real_vm_sample, vae_vm_sample, alpha=0.5, s=10, label='VAE', marker='o')
    axes[1, 0].scatter(real_vm_sample, flow_vm_sample, alpha=0.5, s=10, label='Flow', marker='^')
    axes[1, 0].plot([real_vm_sample.min(), real_vm_sample.max()], 
                    [real_vm_sample.min(), real_vm_sample.max()], 'r--', lw=2, label='Perfect')
    axes[1, 0].set_xlabel('Ground Truth Vm (p.u.)')
    axes[1, 0].set_ylabel('Predicted Vm (p.u.)')
    axes[1, 0].set_title('Vm: VAE vs Flow vs Ground Truth')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Va scatter: VAE vs Flow
    vae_va_sample = Va_vae[sample_idx].flatten()
    flow_va_sample = Va_flow[sample_idx].flatten()
    real_va_sample = Va_real[sample_idx].flatten()
    
    axes[1, 1].scatter(real_va_sample, vae_va_sample, alpha=0.5, s=10, label='VAE', marker='o')
    axes[1, 1].scatter(real_va_sample, flow_va_sample, alpha=0.5, s=10, label='Flow', marker='^')
    axes[1, 1].plot([real_va_sample.min(), real_va_sample.max()], 
                    [real_va_sample.min(), real_va_sample.max()], 'r--', lw=2, label='Perfect')
    axes[1, 1].set_xlabel('Ground Truth Va (rad)')
    axes[1, 1].set_ylabel('Predicted Va (rad)')
    axes[1, 1].set_title('Va: VAE vs Flow vs Ground Truth')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'vae_flow_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n[Saved] Comparison plot: {fig_path}")
    plt.close()
    
    # 2. Error statistics by bus
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Vm MAE by bus
    vae_vm_mae_by_bus = np.mean(np.abs(stats_vae['Vm_error']), axis=0)
    flow_vm_mae_by_bus = np.mean(np.abs(stats_flow['Vm_error']), axis=0)
    
    bus_indices = np.arange(ctx.Nbus)
    axes[0].plot(bus_indices, vae_vm_mae_by_bus, 'o-', label='VAE', markersize=3, alpha=0.7)
    axes[0].plot(bus_indices, flow_vm_mae_by_bus, '^-', label='Flow', markersize=3, alpha=0.7)
    axes[0].set_xlabel('Bus Index')
    axes[0].set_ylabel('Vm MAE (p.u.)')
    axes[0].set_title('Vm MAE by Bus')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Va MAE by bus
    vae_va_mae_by_bus = np.mean(np.abs(stats_vae['Va_error']), axis=0)
    flow_va_mae_by_bus = np.mean(np.abs(stats_flow['Va_error']), axis=0)
    
    axes[1].plot(bus_indices, vae_va_mae_by_bus, 'o-', label='VAE', markersize=3, alpha=0.7)
    axes[1].plot(bus_indices, flow_va_mae_by_bus, '^-', label='Flow', markersize=3, alpha=0.7)
    axes[1].set_xlabel('Bus Index')
    axes[1].set_ylabel('Va MAE (rad)')
    axes[1].set_title('Va MAE by Bus')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'error_by_bus.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"[Saved] Error by bus plot: {fig_path}")
    plt.close()
    
    return stats_vae, stats_flow


def main():
    """Main test function."""
    print("=" * 80)
    print("Constraint Verification and Prediction Error Analysis")
    print("=" * 80)
    
    # Load configuration
    config = get_config()
    device = config.device
    
    # Load multi-preference dataset
    print("\n[1/5] Loading multi-preference dataset...")
    multi_pref_data, sys_data = load_multi_preference_dataset(config)
    
    # Load BRANFT for post-processing
    _, _, BRANFT = load_all_data(config)
    
    # Test with a specific preference
    test_lambda_carbon = 50.0  # Can be changed
    print(f"\n[2/5] Testing with lambda_carbon = {test_lambda_carbon:.2f}")
    
    # Build evaluation context
    ctx = build_ctx_from_multi_preference(
        config, sys_data, multi_pref_data, BRANFT, device,
        lambda_carbon=test_lambda_carbon
    )
    
    # ==================== Step 1: Verify Constraint Calculation ====================
    print("\n[3/5] Verifying constraint calculation...")
    constraint_correct = verify_constraint_calculation(
        ctx, ctx.Real_Vm_full, ctx.Real_Va_full, verbose=True
    )
    
    if not constraint_correct:
        print("\n[WARNING] Constraint calculation may have issues!")
        print("  Proceeding with analysis, but results should be interpreted carefully.")
    
    # ==================== Step 2: Load Models ====================
    print("\n[4/5] Loading trained models...")
    
    model_type = getattr(config, 'model_type', 'rectified')  # Default to rectified
    model_path = f'{config.model_save_dir}/model_multi_pref_{model_type}_final.pth'
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print("  Please train the model first.")
        print(f"  Available models in {config.model_save_dir}:")
        if os.path.exists(config.model_save_dir):
            for f in os.listdir(config.model_save_dir):
                if f.startswith('model_multi_pref_') and f.endswith('.pth'):
                    print(f"    - {f}")
        return
    
    # Import model classes
    flow_model_path = os.path.join(script_dir, 'flow_model')
    if flow_model_path not in sys.path:
        sys.path.insert(0, flow_model_path)
    from net_utiles import FM, VAE
    
    # Load main model
    input_dim = multi_pref_data['input_dim']
    output_dim = multi_pref_data['output_dim']
    pref_dim = 1
    
    model = None
    pretrain_model = None
    
    if model_type in ['rectified', 'flow', 'gaussian', 'conditional', 'interpolation']:
        model = FM(
            network='preference_aware_mlp',
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=getattr(config, 'ngt_flow_hidden_dim', 144),
            num_layers=getattr(config, 'ngt_flow_num_layers', 2),
            time_step=config.time_step,
            output_norm=False,
            pred_type='velocity',
            pref_dim=pref_dim
        )
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        print(f"  Loaded Flow model: {model_path}")
        
        # Load VAE anchor model
        vae_path = f'{config.model_save_dir}/model_multi_pref_vae_final.pth'
        pretrain_model = None
        if os.path.exists(vae_path):
            use_pref_aware = getattr(config, 'vae_use_preference_aware', True)
            if use_pref_aware:
                pretrain_model = VAE(
                    network='preference_aware_mlp',
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    latent_dim=config.latent_dim,
                    output_act=None,
                    pred_type='node',
                    use_cvae=True,
                    pref_dim=pref_dim
                )
            else:
                pretrain_model = VAE(
                    network='mlp',
                    input_dim=input_dim + pref_dim,
                    output_dim=output_dim,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    latent_dim=config.latent_dim,
                    output_act=None,
                    pred_type='node',
                    use_cvae=True
                )
            pretrain_model.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
            pretrain_model.to(device)
            pretrain_model.eval()
            print(f"  Loaded VAE anchor model: {vae_path}")
        else:
            print(f"  [WARNING] VAE anchor model not found: {vae_path}")
    elif model_type == 'vae':
        # VAE model
        use_pref_aware = getattr(config, 'vae_use_preference_aware', True)
        if use_pref_aware:
            model = VAE(
                network='preference_aware_mlp',
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                latent_dim=config.latent_dim,
                output_act=None,
                pred_type='node',
                use_cvae=True,
                pref_dim=pref_dim
            )
        else:
            model = VAE(
                network='mlp',
                input_dim=input_dim + pref_dim,
                output_dim=output_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                latent_dim=config.latent_dim,
                output_act=None,
                pred_type='node',
                use_cvae=True
            )
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        print(f"  Loaded VAE model: {model_path}")
    else:
        print(f"  [INFO] Model type '{model_type}' not supported for this analysis.")
        print("  Skipping model prediction analysis.")
        model = None
        pretrain_model = None
    
    # ==================== Step 3: Get Predictions ====================
    print("\n[5/5] Getting predictions...")
    
    if model is None:
        print("  [SKIP] No model loaded, skipping prediction analysis.")
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Constraint calculation: {'CORRECT' if constraint_correct else 'NEEDS REVIEW'}")
        print("\n" + "=" * 80)
        print("Analysis complete!")
        print("=" * 80)
        return
    
    # Get Flow prediction (final)
    predictor = MultiPreferencePredictor(
        model=model,
        multi_pref_data=multi_pref_data,
        lambda_carbon=test_lambda_carbon,
        model_type=model_type,
        pretrain_model=pretrain_model,
        num_flow_steps=getattr(config, 'multi_pref_flow_steps', 10),
        flow_method='euler'
    )
    
    pred_pack = predictor.predict(ctx)
    Vm_flow = pred_pack.Pred_Vm_full
    Va_flow = pred_pack.Pred_Va_full
    
    # Get VAE anchor prediction (if available)
    Vm_vae = None
    Va_vae = None
    if pretrain_model is not None:
        Vm_vae, Va_vae = get_vae_anchor_predictions(
            ctx, pretrain_model, multi_pref_data, test_lambda_carbon, device
        )
        print("  Got VAE anchor predictions")
    
    # Ground truth
    Vm_real = ctx.Real_Vm_full
    Va_real = ctx.Real_Va_full
    
    print("  Got Flow predictions")
    print("  Got ground truth values")
    
    # ==================== Step 4: Analyze Errors ====================
    print("\n" + "=" * 80)
    print("Prediction Error Analysis")
    print("=" * 80)
    
    # Analyze Flow prediction
    stats_flow = analyze_prediction_errors(ctx, Vm_flow, Va_flow, Vm_real, Va_real, "Flow Model")
    
    # Analyze VAE if available
    if Vm_vae is not None:
        stats_vae = analyze_prediction_errors(ctx, Vm_vae, Va_vae, Vm_real, Va_real, "VAE Anchor")
        
        # Compare VAE vs Flow vs Ground Truth
        compare_vae_flow_groundtruth(
            ctx, Vm_vae, Va_vae, Vm_flow, Va_flow, Vm_real, Va_real,
            save_dir=config.results_dir
        )
    else:
        print("\n[INFO] VAE anchor not available for comparison.")
    
    # ==================== Summary ====================
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Constraint calculation: {'CORRECT' if constraint_correct else 'NEEDS REVIEW'}")
    print(f"Flow model Vm MAE: {stats_flow['Vm_mae']:.6f} p.u.")
    print(f"Flow model Va MAE: {stats_flow['Va_mae']:.6f} rad ({stats_flow['Va_mae']*180/np.pi:.4f} deg)")
    print(f"Flow model Cost MRE: {stats_flow['cost_mre']:.4f}%")
    
    if Vm_vae is not None:
        print(f"\nVAE to Flow improvement:")
        print(f"  Vm MAE: {((stats_vae['Vm_mae'] - stats_flow['Vm_mae']) / stats_vae['Vm_mae'] * 100):.2f}%")
        print(f"  Va MAE: {((stats_vae['Va_mae'] - stats_flow['Va_mae']) / stats_vae['Va_mae'] * 100):.2f}%")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

