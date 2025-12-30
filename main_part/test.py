#!/usr/bin/env python
# coding: utf-8
"""
Multi-Preference Model Evaluation and Pareto Front Analysis

This script evaluates and compares multiple model types on Pareto front:
- Supervised learning: MLP, VAE
- Unsupervised learning: NGT MLP
- Rectified Flow: Single-step and Progressive Chain inference

Uses unified_eval as the core evaluation engine.

Outputs:
- Pareto front visualization with different model categories
- Complete metrics table (MAE, constraint satisfaction, etc.)
- Hypervolume calculation

Usage:
    # Evaluate all models (default)
    python test.py
    
    # Evaluate specific model types
    python test.py --supervised --unsupervised
    python test.py --flow-single --flow-progressive
    
    # Short options
    python test.py -s -u -f -p

Author: Peng Yue
Date: 2025-12-20
"""

import os
import sys
import argparse
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_unsupervised_20251222202357 import get_unsupervised_config as get_config
from models import NetV, create_model
from data_loader import load_all_data
import torch.utils.data as Data

# Import unified evaluation framework
from unified_eval import (
    build_ctx_from_supervised,
    SupervisedPredictor, NGTPredictor, NGTFlowPredictor,
    evaluate_unified, extract_summary_metrics,
    compute_pareto_hypervolumes,
    plot_pareto_front_extended, print_metrics_table,
    save_evaluation_results, convert_to_serializable,
    _as_numpy,
)

# Import flow forward function
from train_unsupervised import flow_forward_ngt


def compute_ngt_vscale_vbias(config, sys_data):
    """
    Compute Vscale and Vbias for NGT/Flow models from config.
    These are used for sigmoid scaling in NetV/PreferenceConditionedNetV models.
    
    Format: [Va_nonZIB_noslack (NPred_Va), Vm_nonZIB (NPred_Vm)]
    """
    if not hasattr(sys_data, 'bus_Pnet_all') or sys_data.bus_Pnet_all is None:
        raise ValueError("sys_data must have bus_Pnet_all computed (from load_training_data)")
    
    bus_Pnet_all = sys_data.bus_Pnet_all
    bus_Pnet_noslack_all = sys_data.bus_Pnet_noslack_all 
    
    NPred_Va = len(bus_Pnet_noslack_all)
    NPred_Vm = len(bus_Pnet_all)
    
    # Get NGT voltage bounds from config
    VmLb = config.ngt_VmLb
    VmUb = config.ngt_VmUb
    VaLb = config.ngt_VaLb
    VaUb = config.ngt_VaUb
    
    # Va part: [VaUb - VaLb, ...]
    Vascale = torch.ones(NPred_Va) * (VaUb - VaLb)
    Vabias = torch.ones(NPred_Va) * VaLb
    
    # Vm part: [VmUb - VmLb, ...]
    Vmscale = torch.ones(NPred_Vm) * (VmUb - VmLb)
    Vmbias = torch.ones(NPred_Vm) * VmLb
    
    # Combined: [Va_nonZIB_noslack, Vm_nonZIB]
    Vscale = torch.cat((Vascale, Vmscale), dim=0)
    Vbias = torch.cat((Vabias, Vmbias), dim=0)
    
    return Vscale, Vbias


def get_ngt_dimensions(config, sys_data):
    """
    Get input and output dimensions for NGT/Flow models from sys_data.
    
    Returns:
        input_dim: Input dimension (same as supervised: [Pd_nonzero, Qd_nonzero])
        output_dim: Output dimension (NPred_Va + NPred_Vm)
    """
    if not hasattr(sys_data, 'bus_Pnet_all') or sys_data.bus_Pnet_all is None:
        raise ValueError("sys_data must have bus_Pnet_all computed (from load_training_data)")
    
    # Input dimension: same as supervised models (non-zero Pd and Qd)
    # This should match the input_dim from load_training_data
    if hasattr(sys_data, 'x_test'):
        input_dim = sys_data.x_test.shape[1]
    else:
        # Fallback: compute from bus_Pd and bus_Qd
        bus_Pd = sys_data.idx_Pd if hasattr(sys_data, 'idx_Pd') else None
        bus_Qd = sys_data.idx_Qd if hasattr(sys_data, 'idx_Qd') else None
        if bus_Pd is not None and bus_Qd is not None:
            input_dim = len(bus_Pd) + len(bus_Qd)
        else:
            raise ValueError("Cannot determine input_dim from sys_data")
    
    # Output dimension: NPred_Va + NPred_Vm
    output_dim = sys_data.NPred_Va + sys_data.NPred_Vm
    
    return input_dim, output_dim


def load_ngt_model(config, sys_data, model_path, device):
    """
    Load a trained NGT model from checkpoint.
    
    Args:
        config: Configuration object
        sys_data: PowerSystemData object (with bus_Pnet_all etc. computed)
        model_path: Path to model checkpoint
        device: Device to load model on
    """
    Vscale, Vbias = compute_ngt_vscale_vbias(config, sys_data)
    input_dim, output_dim = get_ngt_dimensions(config, sys_data)
    
    model = NetV(
        input_channels=input_dim,
        output_channels=output_dim,
        hidden_units=config.ngt_hidden_units,
        khidden=config.ngt_khidden,
        Vscale=Vscale.to(device),
        Vbias=Vbias.to(device)
    )
    model.to(device)
    
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def load_flow_model(config, sys_data, model_path, device):
    """
    Load a trained Flow model from checkpoint.
    
    Args:
        config: Configuration object
        sys_data: PowerSystemData object (with bus_Pnet_all etc. computed)
        model_path: Path to model checkpoint
        device: Device to load model on
    """
    Vscale, Vbias = compute_ngt_vscale_vbias(config, sys_data)
    input_dim, output_dim = get_ngt_dimensions(config, sys_data)
    
    hidden_dim = getattr(config, 'ngt_flow_hidden_dim', 144)
    num_layers = getattr(config, 'ngt_flow_num_layers', 2)
    
    model_flow = PreferenceConditionedNetV(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        Vscale=Vscale.to(device),
        Vbias=Vbias.to(device),
        preference_dim=2,
        preference_hidden=64
    )
    model_flow.to(device)
    model_flow.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model_flow.eval()
    
    return model_flow


def evaluate_ngt_mlp_model(config, ctx, sys_data, model_path, device, 
                            model_name, lambda_cost=1.0, verbose=False):
    """
    Evaluate a NGT MLP model using unified_eval.
    
    Returns:
        dict with summary metrics for Pareto analysis
    """
    print(f"\n  Evaluating {model_name} (NGT MLP)...")
    
    if not os.path.exists(model_path):
        print(f"    [SKIP] Model file not found: {model_path}")
        return None
    
    # Load model and create predictor
    model = load_ngt_model(config, sys_data, model_path, device)
    predictor = NGTPredictor(model)
    
    # Run unified evaluation
    eval_result = evaluate_unified(ctx, predictor, apply_post_processing=True, verbose=verbose)
    
    # Extract summary metrics
    summary = extract_summary_metrics(
        eval_result, model_name, 
        category='unsupervised', 
        lambda_cost=lambda_cost,
        use_post_processed=True
    )
    
    return summary


def evaluate_flow_single_model(config, ctx, sys_data, model_path, device,
                                model_name, lambda_cost=0.9, verbose=False):
    """
    Evaluate a Rectified Flow model with single-step inference (VAE -> Flow).
    
    Returns:
        dict with summary metrics for Pareto analysis
    """
    print(f"\n  Evaluating {model_name} (Flow Single)...")
    
    if not os.path.exists(model_path):
        print(f"    [SKIP] Model file not found: {model_path}")
        return None
    
    # Check VAE models exist
    vae_vm_path = config.pretrain_model_path_vm
    vae_va_path = config.pretrain_model_path_va
    
    if not os.path.exists(vae_vm_path) or not os.path.exists(vae_va_path):
        print(f"    [SKIP] VAE models not found (required for Flow anchor)")
        return None
    
    # Load Flow model
    model_flow = load_flow_model(config, sys_data, model_path, device)
    
    # Load VAE models for anchor
    input_dim, _ = get_ngt_dimensions(config, sys_data)
    output_dim_vm = config.Nbus
    output_dim_va = config.Nbus - 1
    
    vae_vm = create_model('vae', input_dim, output_dim_vm, config, is_vm=True)
    vae_va = create_model('vae', input_dim, output_dim_va, config, is_vm=False)
    vae_vm.to(device)
    vae_va.to(device)
    vae_vm.load_state_dict(torch.load(vae_vm_path, map_location=device, weights_only=True), strict=False)
    vae_va.load_state_dict(torch.load(vae_va_path, map_location=device, weights_only=True), strict=False)
    vae_vm.eval()
    vae_va.eval()
    
    # Create preference tensor
    preference = torch.tensor([[lambda_cost, 1.0 - lambda_cost]], dtype=torch.float32, device=device)
    
    # Create flow predictor
    # Note: NGTFlowPredictor expects ngt_data, but it only uses it for reference.
    # Since we're using ctx which has all necessary info, we can pass a minimal dict
    flow_inf_steps = getattr(config, 'ngt_flow_inf_steps', 10)
    predictor = NGTFlowPredictor(
        model_flow=model_flow,
        vae_vm=vae_vm,
        vae_va=vae_va,
        ngt_data={},  # Empty dict - not used when ctx has all info
        preference=preference,
        flow_forward_ngt=flow_forward_ngt,
        flow_inf_steps=flow_inf_steps,
    )
    
    # Run unified evaluation
    eval_result = evaluate_unified(ctx, predictor, apply_post_processing=True, verbose=verbose)
    
    # Extract summary metrics
    summary = extract_summary_metrics(
        eval_result, model_name,
        category='flow',
        lambda_cost=lambda_cost,
        use_post_processed=True
    )
    
    return summary


def evaluate_flow_progressive_model(config, ctx, sys_data, chain_configs, device,
                                     model_name, target_lambda_cost=0.5, verbose=False):
    """
    Evaluate Rectified Flow model with progressive chain inference.
    
    Chain inference: VAE -> Flow(0.9) -> Flow(0.8) -> ... -> Flow(target)
    
    Returns:
        dict with summary metrics for Pareto analysis
    """
    print(f"\n  Evaluating {model_name} (Flow Progressive)...")
    
    # Check all model files exist
    for path, lc in chain_configs:
        if not os.path.exists(path):
            print(f"    [SKIP] Model file not found: {path}")
            return None
    
    # Check VAE models exist
    vae_vm_path = config.pretrain_model_path_vm
    vae_va_path = config.pretrain_model_path_va
    
    if not os.path.exists(vae_vm_path) or not os.path.exists(vae_va_path):
        print(f"    [SKIP] VAE models not found (required for Flow anchor)")
        return None
    
    # Load all Flow models in chain
    input_dim, _ = get_ngt_dimensions(config, sys_data)
    output_dim_vm = config.Nbus
    output_dim_va = config.Nbus - 1
    
    # Load VAE models for initial anchor
    vae_vm = create_model('vae', input_dim, output_dim_vm, config, is_vm=True)
    vae_va = create_model('vae', input_dim, output_dim_va, config, is_vm=False)
    vae_vm.to(device)
    vae_va.to(device)
    vae_vm.load_state_dict(torch.load(vae_vm_path, map_location=device, weights_only=True), strict=False)
    vae_va.load_state_dict(torch.load(vae_va_path, map_location=device, weights_only=True), strict=False)
    vae_vm.eval()
    vae_va.eval()
    
    # Get test data dimensions
    x_test = ctx.x_test.to(device)
    Ntest = x_test.shape[0]
    bus_slack = int(ctx.bus_slack)
    bus_Pnet_all = ctx.bus_Pnet_all
    bus_Pnet_noslack_all = ctx.bus_Pnet_noslack_all
    flow_inf_steps = getattr(config, 'ngt_flow_inf_steps', 10)
    
    # Step 1: Generate initial anchor from VAE
    import time
    start_time = time.time()
    with torch.no_grad():
        Vm_vae = vae_vm(x_test, use_mean=True)
        Va_vae_noslack = vae_va(x_test, use_mean=True)
        
        # Unscale VAE predictions
        scale_vm = float(config.scale_vm.item() if hasattr(config.scale_vm, 'item') else config.scale_vm)
        scale_va = float(config.scale_va.item() if hasattr(config.scale_va, 'item') else config.scale_va)
        
        VmLb = ctx.sys_data.VmLb
        VmUb = ctx.sys_data.VmUb
        if isinstance(VmLb, np.ndarray):
            VmLb_t = torch.from_numpy(VmLb).float().to(device)
            VmUb_t = torch.from_numpy(VmUb).float().to(device)
        elif isinstance(VmLb, torch.Tensor):
            VmLb_t = VmLb.to(device).float()
            VmUb_t = VmUb.to(device).float()
        else:
            VmLb_t = torch.full((config.Nbus,), float(VmLb), device=device)
            VmUb_t = torch.full((config.Nbus,), float(VmUb), device=device)
        
        Vm_vae_phys = Vm_vae / scale_vm * (VmUb_t - VmLb_t) + VmLb_t
        Va_vae_phys_noslack = Va_vae_noslack / scale_va
        
        Va_full = torch.zeros(Ntest, config.Nbus, device=device)
        Va_full[:, :bus_slack] = Va_vae_phys_noslack[:, :bus_slack]
        Va_full[:, bus_slack + 1:] = Va_vae_phys_noslack[:, bus_slack:]
        
        Vm_nonZIB = Vm_vae_phys[:, bus_Pnet_all]
        Va_nonZIB_noslack = Va_full[:, bus_Pnet_noslack_all]
        V_anchor_phys = torch.cat([Va_nonZIB_noslack, Vm_nonZIB], dim=1)
    
    # Step 2: Chain through Flow models
    print(f"    Chain: VAE -> ", end="")
    z_current = None
    
    for i, (path, lc) in enumerate(chain_configs):
        model_flow = load_flow_model(config, sys_data, path, device)
        pref_tensor = torch.tensor([[lc, 1.0 - lc]], dtype=torch.float32, device=device)
        pref_batch = pref_tensor.expand(Ntest, -1)
        
        with torch.no_grad():
            if z_current is None:
                # First step: convert VAE anchor to logit space
                Vscale = model_flow.Vscale.to(device)
                Vbias = model_flow.Vbias.to(device)
                eps = 1e-6
                u = (V_anchor_phys - Vbias) / (Vscale + 1e-12)
                u = torch.clamp(u, eps, 1 - eps)
                z_anchor = torch.log(u / (1 - u))
            else:
                z_anchor = z_current
            
            z_current = flow_forward_ngt(
                model_flow, x_test, z_anchor,
                pref_batch, flow_inf_steps, training=False
            )
        
        print(f"Flow({lc:.1f})", end="")
        if i < len(chain_configs) - 1:
            print(" -> ", end="")
    print()
    
    # Step 3: Apply final sigmoid scaling
    with torch.no_grad():
        last_flow_model = load_flow_model(config, sys_data, chain_configs[-1][0], device)
        V_pred = torch.sigmoid(z_current) * last_flow_model.Vscale.to(device) + last_flow_model.Vbias.to(device)
    
    inference_time = time.time() - start_time
    
    # Convert prediction to full voltage format
    from unified_eval import reconstruct_full_from_partial, get_genload, get_vioPQg, get_viobran2
    
    V_pred_np = V_pred.cpu().numpy()
    Pred_Vm_full, Pred_Va_full = reconstruct_full_from_partial(ctx, V_pred_np)
    
    # Calculate power flow and metrics
    Pred_V = Pred_Vm_full * np.exp(1j * Pred_Va_full)
    Pred_Pg, Pred_Qg, Pred_Pd, Pred_Qd = get_genload(
        Pred_V, ctx.Pdtest, ctx.Qdtest,
        ctx.bus_Pg, ctx.bus_Qg, ctx.Ybus
    )
    
    # Compute cost and carbon
    from utils import get_carbon_emission_vectorized
    
    gencost = ctx.gencost_Pg
    baseMVA = ctx.baseMVA
    Pred_cost = gencost[:, 0] * (Pred_Pg * baseMVA)**2 + gencost[:, 1] * np.abs(Pred_Pg * baseMVA)
    Pred_cost_total = np.sum(Pred_cost, axis=1)
    carbon = get_carbon_emission_vectorized(Pred_Pg, ctx.gci_values, baseMVA)
    
    # Constraint satisfaction
    _, _, lsidxPg, lsidxQg, _, vio_PQg, _, _, _, _ = get_vioPQg(
        Pred_Pg, ctx.bus_Pg, ctx.MAXMIN_Pg,
        Pred_Qg, ctx.bus_Qg, ctx.MAXMIN_Qg,
        ctx.DELTA
    )
    if torch.is_tensor(vio_PQg):
        vio_PQg = vio_PQg.numpy()
    
    lsidxPQg = np.squeeze(np.array(np.where((lsidxPg + lsidxQg) > 0)))
    num_violated = np.size(lsidxPQg)
    
    # Voltage satisfaction
    VmLb_const = config.ngt_VmLb
    VmUb_const = config.ngt_VmUb
    Vm_satisfy = 100 - np.mean(Pred_Vm_full > VmUb_const) * 100 - np.mean(Pred_Vm_full < VmLb_const) * 100
    
    # Branch constraints
    vio_branang, vio_branpf, _, _, _, _, _, _ = get_viobran2(
        Pred_V, Pred_Va_full, ctx.branch, ctx.Yf, ctx.Yt,
        ctx.BRANFT, baseMVA, ctx.DELTA
    )
    if torch.is_tensor(vio_branang):
        vio_branang = vio_branang.numpy()
    if torch.is_tensor(vio_branpf):
        vio_branpf = vio_branpf.numpy()
    
    # MAE calculation
    Real_Vm = ctx.Real_Vm_full
    Real_Va_full = ctx.Real_Va_full
    mae_Vm = np.mean(np.abs(Real_Vm - Pred_Vm_full))
    bus_Va_idx = np.delete(np.arange(config.Nbus), bus_slack)
    mae_Va = np.mean(np.abs(Real_Va_full[:, bus_Va_idx] - Pred_Va_full[:, bus_Va_idx]))
    
    # Real cost for comparison
    Real_V = Real_Vm * np.exp(1j * Real_Va_full)
    Real_Pg, _, _, _ = get_genload(
        Real_V, ctx.Pdtest, ctx.Qdtest,
        ctx.bus_Pg, ctx.bus_Qg, ctx.Ybus
    )
    Real_cost = gencost[:, 0] * (Real_Pg * baseMVA)**2 + gencost[:, 1] * np.abs(Real_Pg * baseMVA)
    Real_cost_total = np.sum(Real_cost, axis=1)
    cost_error_percent = np.mean((Pred_cost_total - Real_cost_total) / Real_cost_total * 100)
    
    return {
        'name': model_name,
        'model_type': 'flow_progressive',
        'category': 'flow',
        'cost_mean': np.mean(Pred_cost_total),
        'carbon_mean': np.mean(carbon),
        'mae_Vm': mae_Vm,
        'mae_Va': mae_Va,
        'cost_error_percent': cost_error_percent,
        'Pg_satisfy': np.mean(vio_PQg[:, 0]),
        'Qg_satisfy': np.mean(vio_PQg[:, 1]),
        'Vm_satisfy': Vm_satisfy,
        'branch_ang_satisfy': np.mean(vio_branang),
        'branch_pf_satisfy': np.mean(vio_branpf),
        'num_violated': num_violated,
        'inference_time_ms': inference_time / Ntest * 1000,
        'lambda_cost': target_lambda_cost,
        'lambda_carbon': 1.0 - target_lambda_cost,
        'chain_length': len(chain_configs),
        'Pred_Pg': Pred_Pg,
    }


def create_dataloaders_from_ctx(ctx, config):
    """
    Create dataloaders from unified evaluation context.
    This ensures all models use the same test set.
    
    Note: The labels (yvm_test, yva_test) are converted back to normalized space
    for consistency, though SupervisedPredictor doesn't actually use the label values.
    """
    # Convert physical space labels back to normalized space for dataloader
    # (SupervisedPredictor doesn't use label values, but we keep consistency)
    scale_vm = config.scale_vm.item() if hasattr(config.scale_vm, 'item') else float(config.scale_vm)
    scale_va = config.scale_va.item() if hasattr(config.scale_va, 'item') else float(config.scale_va)
    
    # Handle VmLb and VmUb (can be scalar or array)
    if hasattr(ctx.sys_data.VmLb, 'item'):
        VmLb = ctx.sys_data.VmLb.item()
    elif isinstance(ctx.sys_data.VmLb, (int, float)):
        VmLb = float(ctx.sys_data.VmLb)
    else:
        VmLb = float(ctx.sys_data.VmLb[0])  # Take first element if array
    
    if hasattr(ctx.sys_data.VmUb, 'item'):
        VmUb = ctx.sys_data.VmUb.item()
    elif isinstance(ctx.sys_data.VmUb, (int, float)):
        VmUb = float(ctx.sys_data.VmUb)
    else:
        VmUb = float(ctx.sys_data.VmUb[0])  # Take first element if array
    
    # Ensure yvmtests and yvatests_noslack are torch tensors
    yvmtests = ctx.yvmtests if isinstance(ctx.yvmtests, torch.Tensor) else torch.from_numpy(ctx.yvmtests).float()
    yvatests_noslack = ctx.yvatests_noslack if isinstance(ctx.yvatests_noslack, torch.Tensor) else torch.from_numpy(ctx.yvatests_noslack).float()
    
    # Convert physical Vm to normalized: y_norm = (y_phys - VmLb) / (VmUb - VmLb) * scale_vm
    # Reverse of: y_phys = y_norm / scale_vm * (VmUb - VmLb) + VmLb
    yvm_test_norm = (yvmtests - VmLb) / (VmUb - VmLb + 1e-12) * scale_vm
    
    # Va conversion: Reverse of y_phys = y_norm / scale_va
    yva_test_norm = yvatests_noslack * scale_va
    
    # Create datasets using unified test data
    test_dataset_vm = Data.TensorDataset(ctx.x_test, yvm_test_norm)
    test_loader_vm = Data.DataLoader(
        dataset=test_dataset_vm,
        batch_size=config.batch_size_test,
        shuffle=False,
    )
    
    test_dataset_va = Data.TensorDataset(ctx.x_test, yva_test_norm)
    test_loader_va = Data.DataLoader(
        dataset=test_dataset_va,
        batch_size=config.batch_size_test,
        shuffle=False,
    )
    
    # Create dummy training loaders (not used by SupervisedPredictor.predict)
    # We use test data as placeholder to avoid errors
    dummy_train_dataset_vm = Data.TensorDataset(ctx.x_test[:1], yvm_test_norm[:1])
    dummy_train_loader_vm = Data.DataLoader(dummy_train_dataset_vm, batch_size=1, shuffle=False)
    
    dummy_train_dataset_va = Data.TensorDataset(ctx.x_test[:1], yva_test_norm[:1])
    dummy_train_loader_va = Data.DataLoader(dummy_train_dataset_va, batch_size=1, shuffle=False)
    
    dataloaders = {
        'train_vm': dummy_train_loader_vm,
        'train_va': dummy_train_loader_va,
        'test_vm': test_loader_vm,
        'test_va': test_loader_va,
    }
    
    return dataloaders


def evaluate_supervised_model(config, ctx, sys_data, model_type, model_paths, 
                               device, model_name, verbose=False):
    """
    Evaluate a supervised learning model (MLP or VAE) using unified_eval.
    Uses the same test set (ctx) as other models for fair comparison.
    
    Returns:
        dict with summary metrics for Pareto analysis
    """
    print(f"\n  Evaluating {model_name} ({model_type})...")
    
    vm_path = model_paths['vm']
    va_path = model_paths['va']
    
    if not os.path.exists(vm_path) or not os.path.exists(va_path):
        print(f"    [SKIP] Model files not found")
        return None
    
    # Load models
    # Input dimension: same as supervised models (non-zero Pd and Qd)
    input_dim = ctx.x_test.shape[1]
    output_dim_vm = config.Nbus
    output_dim_va = config.Nbus - 1
    
    model_vm = create_model(model_type, input_dim, output_dim_vm, config, is_vm=True)
    model_va = create_model(model_type, input_dim, output_dim_va, config, is_vm=False)
    
    model_vm.to(device)
    model_va.to(device)
    
    model_vm.load_state_dict(torch.load(vm_path, map_location=device, weights_only=True))
    model_va.load_state_dict(torch.load(va_path, map_location=device, weights_only=True))
    
    model_vm.eval()
    model_va.eval()
    
    # Create dataloaders from unified ctx (ensures same test set)
    dataloaders = create_dataloaders_from_ctx(ctx, config)
    
    # Create supervised predictor
    predictor = SupervisedPredictor(
        model_vm=model_vm,
        model_va=model_va,
        dataloaders=dataloaders,
        model_type=model_type,
    )
    
    # Run unified evaluation with post-processing
    eval_result = evaluate_unified(ctx, predictor, apply_post_processing=True, verbose=verbose)
    
    # Extract summary metrics
    summary = extract_summary_metrics(
        eval_result, model_name,
        category='supervised',
        lambda_cost=None,
        use_post_processed=True
    )
    
    return summary


def parse_args():
    """Parse command line arguments for model selection."""
    parser = argparse.ArgumentParser(
        description='Multi-Model Evaluation & Pareto Front Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all models (default)
  python test.py
  
  # Evaluate only supervised and unsupervised models
  python test.py --supervised --unsupervised
  
  # Evaluate only Flow models
  python test.py --flow-single --flow-progressive
  
  # Short options
  python test.py -s -u -f -p
  
  # Evaluate a custom Flow model (for debugging/diagnosis)
  python test.py --custom-flow saved_models/NetV_ngt_flow_300bus_lc08_E500_final.pth 0.8
        """
    )
    
    parser.add_argument('-s', '--supervised', action='store_true',
                        help='Evaluate supervised learning models (MLP, VAE)')
    parser.add_argument('-u', '--unsupervised', action='store_true',
                        help='Evaluate unsupervised NGT MLP models')
    parser.add_argument('-f', '--flow-single', action='store_true',
                        help='Evaluate Rectified Flow single-step models') 
    parser.add_argument('-a', '--all', action='store_true',
                        help='Evaluate all model types (default if no options specified)')
    parser.add_argument('--custom-flow', nargs=2, metavar=('PATH', 'LAMBDA'),
                        help='Evaluate a custom Flow model: --custom-flow <model_path> <lambda_cost>')
    parser.add_argument('--epochs', type=int, default=4500,
                        help='Number of training epochs for Flow model paths (default: 4500)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print detailed evaluation output')
    
    args = parser.parse_args()
    
    # If no specific model type is selected, evaluate all
    if not (args.supervised or args.unsupervised or args.flow_single):
        args.all = True
    
    if args.all:
        args.supervised = True
        args.unsupervised = True
        args.flow_single = True
    
    return args


def main():
    """Main evaluation function for multi-model Pareto comparison."""
    # Parse arguments
    args = parse_args()
    
    print("=" * 100)
    print(" Multi-Model Evaluation & Pareto Front Analysis (Using unified_eval)")
    print("=" * 100)
    
    # Print selected model types
    selected = []
    if args.supervised:
        selected.append("Supervised (MLP/VAE)")
    if args.unsupervised:
        selected.append("Unsupervised (NGT MLP)")
    if args.flow_single:
        selected.append("Flow (Single-step)") 
    print(f" Evaluating: {' | '.join(selected)}")
    print("=" * 100)
    
    # Load configuration
    config = get_config()
    device = config.device
    
    print(f"\nConfiguration:")
    print(f"  Nbus: {config.Nbus}")
    print(f"  Device: {device}")
    print(f"  Model directory: {config.model_save_dir}")
    
    # Load data (only once) - using supervised learning data for unified test set
    print("\nLoading test data (using supervised learning data for unified evaluation)...")
    sys_data, dataloaders, BRANFT = load_all_data(config)
    
    # Build unified evaluation context (now includes NGT/Flow required fields)
    ctx = build_ctx_from_supervised(config, sys_data, dataloaders, BRANFT, device)
    
    n_epochs = args.epochs
    all_results = []
    
    # ============================================================
    # 1. Evaluate Supervised Learning Models (MLP, VAE)
    # ============================================================
    if args.supervised:
        print("\n" + "=" * 70)
        print(" 1. Evaluating Supervised Learning Models (MLP, VAE)")
        print("=" * 70)
        
        supervised_configs = [
            {
                'name': 'MLP_sup',
                'type': 'simple',
                'vm': f'{config.model_save_dir}/modelvm300r2N1Lm8642E1000_simple.pth',   # main_part\saved_models\modelvm300r2N1Lm8642E1000_simple.pth
                'va': f'{config.model_save_dir}/modelva300r2N1La8642E1000_simple.pth',   # main_part\saved_models\modelva300r2N1La8642E1000_simple.pth
            },
            {
                'name': 'VAE_sup',
                'type': 'vae',
                'vm': config.pretrain_model_path_vm,
                'va': config.pretrain_model_path_va,
            },
        ]
        
        for sc in supervised_configs:
            result = evaluate_supervised_model(
                config, ctx, sys_data,
                sc['type'], {'vm': sc['vm'], 'va': sc['va']},
                device, sc['name'], verbose=args.verbose
            )
            if result is not None:
                all_results.append(result)
                print(f"    {sc['name']}: cost={result['cost_mean']:.2f}, carbon={result['carbon_mean']:.4f}")
    
    # ============================================================
    # 2. Evaluate Unsupervised NGT MLP Models
    # ============================================================
    if args.unsupervised:
        print("\n" + "=" * 70)
        print(" 2. Evaluating Unsupervised NGT MLP Models")
        print("=" * 70)
        
        ngt_mlp_configs = [
            {'name': 'NGT_lc0.1', 'path': f'NetV_ngt_{config.Nbus}bus_lc0.1_E{n_epochs}_final.pth', 'lambda_cost': 0.1}, # NetV_ngt_300bus_lc0.1_E4500_final 
        ]
        
        for nc in ngt_mlp_configs:
            model_path = os.path.join(config.model_save_dir, nc['path'])
            result = evaluate_ngt_mlp_model(
                config, ctx, sys_data, model_path, device,
                nc['name'], nc['lambda_cost'], verbose=args.verbose
            )
            if result is not None:
                all_results.append(result)
                print(f"    {nc['name']}: cost={result['cost_mean']:.2f}, carbon={result['carbon_mean']:.4f}")
    
    # ============================================================
    # 3. Evaluate Rectified Flow Models (Single-step)
    # ============================================================
    if args.flow_single:
        print("\n" + "=" * 70)
        print(" 3. Evaluating Rectified Flow Models (Single-step: VAE -> Flow)")
        print("=" * 70)
        
        flow_single_configs = [
            {'name': 'Flow_lc1.0', 'path': f'NetV_ngt_flow_{config.Nbus}bus_lc10_E{n_epochs}_final.pth', 'lambda_cost': 1.0},
            {'name': 'Flow_lc0.9', 'path': f'NetV_ngt_flow_{config.Nbus}bus_lc09_E{n_epochs}_final.pth', 'lambda_cost': 0.9},
            {'name': 'Flow_lc0.7', 'path': f'NetV_ngt_flow_{config.Nbus}bus_lc07_E{n_epochs}_final.pth', 'lambda_cost': 0.7},
            {'name': 'Flow_lc0.5', 'path': f'NetV_ngt_flow_{config.Nbus}bus_lc05_E{n_epochs}_final.pth', 'lambda_cost': 0.5},
            {'name': 'Flow_lc0.3', 'path': f'NetV_ngt_flow_{config.Nbus}bus_lc03_E{n_epochs}_final.pth', 'lambda_cost': 0.3},
            {'name': 'Flow_lc0.1', 'path': f'NetV_ngt_flow_{config.Nbus}bus_lc01_E{n_epochs}_final.pth', 'lambda_cost': 0.1},
        ]
        
        for fc in flow_single_configs:
            model_path = os.path.join(config.model_save_dir, fc['path'])
            result = evaluate_flow_single_model(
                config, ctx, sys_data, model_path, device,
                fc['name'], fc['lambda_cost'], verbose=args.verbose
            )
            if result is not None:
                all_results.append(result)
                print(f"    {fc['name']}: cost={result['cost_mean']:.2f}, carbon={result['carbon_mean']:.4f}")
    
    # ============================================================
    # 4. Evaluate Custom Flow Model (for diagnosis)
    # ============================================================
    if args.custom_flow:
        print("\n" + "=" * 70)
        print(" Custom Flow Model Evaluation (Diagnostic)")
        print("=" * 70)
        
        custom_path, custom_lambda = args.custom_flow
        custom_lambda = float(custom_lambda)
        
        if not os.path.isabs(custom_path):
            custom_path = os.path.join(config.model_save_dir, custom_path)
        
        if os.path.exists(custom_path):
            print(f"  Model path: {custom_path}")
            print(f"  Lambda cost: {custom_lambda}")
            
            custom_name = f"Flow_custom_lc{custom_lambda}"
            result = evaluate_flow_single_model(
                config, ctx, sys_data, custom_path, device,
                custom_name, custom_lambda, verbose=args.verbose
            )
            if result is not None:
                all_results.append(result)
                print(f"    {custom_name}: cost={result['cost_mean']:.2f}, carbon={result['carbon_mean']:.4f}")
        else:
            print(f"  [WARNING] Custom model not found: {custom_path}")
    
    # ============================================================
    # 5. Results Analysis
    # ============================================================
    if len(all_results) == 0:
        print("\n[ERROR] No models were successfully evaluated!")
        print("Please check model paths and run training first.")
        return
    
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
    
    ref_point = np.array([
        np.max(costs) * 1.1,
        np.max(carbons) * 1.1
    ])
    print(f"\n  Reference point: cost={ref_point[0]:.2f}, carbon={ref_point[1]:.4f}")
    
    hypervolumes = compute_pareto_hypervolumes(all_results, ref_point)
    
    for category in ['supervised', 'unsupervised', 'flow']:
        if category in hypervolumes:
            print(f"  Hypervolume ({category}): {hypervolumes[category]:.2f}")
    print(f"  Hypervolume (all): {hypervolumes['all']:.2f}")
    
    # ============================================================
    # 7. Plot Pareto Front
    # ============================================================
    plot_pareto_front_extended(
        all_results, ref_point, hypervolumes,
        save_path=f'{config.results_dir}/pareto_front_multi_model.png'
    )
    
    # ============================================================
    # 8. Print Summary Table (Cost vs Carbon)
    # ============================================================
    print("\n" + "=" * 100)
    print(" Summary: Cost vs Carbon by Model Category")
    print("=" * 100)
    print(f"\n{'Model':<25} {'Category':<15} {'lambda_cost':<12} {'Cost ($/h)':<15} {'Carbon (tCO2/h)':<18}")
    print("-" * 100)
    
    for r in sorted(all_results, key=lambda x: (x.get('category', 'z'), x['cost_mean'])):
        cat = r.get('category', 'unknown')
        lc = f"{r['lambda_cost']:.1f}" if r.get('lambda_cost') is not None else "N/A"
        print(f"{r['name']:<25} {cat:<15} {lc:<12} {r['cost_mean']:<15.2f} {r['carbon_mean']:<18.4f}")
    
    print("-" * 100)
    
    # ============================================================
    # 9. Save Results to JSON
    # ============================================================
    save_evaluation_results(
        all_results, hypervolumes, ref_point,
        f'{config.results_dir}/multi_model_comparison_results.json',
        config=config
    )
    
    print("\n" + "=" * 100)
    print(" Evaluation completed!")
    print("=" * 100)


if __name__ == "__main__":
    main()
