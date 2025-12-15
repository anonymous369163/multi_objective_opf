#!/usr/bin/env python
# coding: utf-8
"""
Multi-Preference Model Evaluation and Pareto Front Analysis

This script evaluates all trained NGT models with different preference weights,
compares them with the VAE baseline, plots the Pareto front, and computes hypervolume.

Usage:
    python evaluate_multi_preference.py

Author: Auto-generated
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from models import NetV, create_model
from data_loader import load_ngt_training_data
from train import evaluate_ngt_model, evaluate_dual_model
from utils import (
    get_genload, get_carbon_emission_vectorized, 
    compute_hypervolume, evaluate_pareto_front
)
from multi_objective_loss import get_gci_for_generators


def load_ngt_model(config, ngt_data, model_path, device):
    """Load a trained NGT model from checkpoint."""
    Vscale = ngt_data['Vscale'].to(device)
    Vbias = ngt_data['Vbias'].to(device)
    input_dim = ngt_data['input_dim']
    output_dim = ngt_data['output_dim']
    
    model = NetV(
        input_channels=input_dim,
        output_channels=output_dim,
        hidden_units=config.ngt_hidden_units,
        khidden=config.ngt_khidden,
        Vscale=Vscale,
        Vbias=Vbias
    )
    model.to(device)
    
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def compute_cost_and_carbon(Pred_Pg, gencost, gci_values_for_nodes, baseMVA):
    """
    Compute economic cost and carbon emission for predictions.
    
    Args:
        Pred_Pg: Predicted active power generation [Ntest, n_nodes] in p.u.
                 where n_nodes = len(bus_Pg) (generation nodes, not all generators)
        gencost: Generator cost coefficients [n_nodes, 2] (c2, c1)
        gci_values_for_nodes: GCI values for each generation node [n_nodes]
        baseMVA: Base MVA
        
    Returns:
        cost_mean: Mean economic cost ($/h)
        carbon_mean: Mean carbon emission (tCO2/h)
    """
    # Convert to MW
    Pg_MW = Pred_Pg * baseMVA
    
    # Economic cost: c2*Pg^2 + c1*|Pg|
    cost = gencost[:, 0] * (Pg_MW ** 2) + gencost[:, 1] * np.abs(Pg_MW)
    cost_per_sample = np.sum(cost, axis=1)  # Sum over generators
    cost_mean = np.mean(cost_per_sample)
    
    # Carbon emission: Σ GCI_i * Pg_i (using node-aligned GCI values)
    carbon = get_carbon_emission_vectorized(Pred_Pg, gci_values_for_nodes, baseMVA)
    carbon_mean = np.mean(carbon)
    
    return cost_mean, carbon_mean


def get_gci_for_generation_nodes(config, sys_data):
    """
    Get GCI values aligned with generation nodes (bus_Pg), not all generators.
    
    Since Pred_Pg has shape [Ntest, len(bus_Pg)], we need GCI values of same length.
    We use idxPg to map from generators to the correct indices.
    
    Args:
        config: Configuration object
        sys_data: PowerSystemData object
        
    Returns:
        gci_values: Array of GCI values aligned with bus_Pg [len(bus_Pg)]
    """
    # Get all generator GCI values
    gci_all = get_gci_for_generators(config, sys_data)
    
    # idxPg maps to which generators are at bus_Pg locations
    # gencost_Pg = gencost[idxPg, :2] gives us the right indexing
    idxPg = sys_data.idxPg if isinstance(sys_data.idxPg, np.ndarray) else np.array(sys_data.idxPg)
    
    # Select GCI values for the generators at bus_Pg nodes
    gci_for_nodes = gci_all[idxPg]
    
    return gci_for_nodes


def evaluate_single_model(config, model, ngt_data, sys_data, device, model_name):
    """
    Evaluate a single NGT model and return cost/carbon.
    
    Returns:
        dict with keys: cost_mean, carbon_mean, Pg_satisfy, Qg_satisfy, etc.
    """
    print(f"\n  Evaluating {model_name}...")
    
    # Get test data
    x_test = ngt_data['x_test'].to(device)
    Ntest = x_test.shape[0]
    
    # Model prediction
    model.eval()
    with torch.no_grad():
        V_pred = model(x_test)
    V_pred_np = V_pred.cpu().numpy()
    
    # Reconstruct full voltage (same logic as evaluate_ngt_model)
    NPred_Va = ngt_data['NPred_Va']
    NPred_Vm = ngt_data['NPred_Vm']
    bus_Pnet_all = ngt_data['bus_Pnet_all']
    bus_ZIB_all = ngt_data['bus_ZIB_all']
    idx_bus_Pnet_slack = ngt_data['idx_bus_Pnet_slack']
    NZIB = ngt_data['NZIB']
    param_ZIMV = ngt_data['param_ZIMV']
    
    xam_P = np.insert(V_pred_np, idx_bus_Pnet_slack[0], 0, axis=1)
    Va_len_with_slack = NPred_Va + 1
    Va_nonZIB = xam_P[:, :Va_len_with_slack]
    Vm_nonZIB = xam_P[:, Va_len_with_slack:Va_len_with_slack + NPred_Vm]
    Vx = Vm_nonZIB * np.exp(1j * Va_nonZIB)
    
    if NZIB > 0 and param_ZIMV is not None:
        Vy = np.dot(param_ZIMV, Vx.T).T
    else:
        Vy = None
    
    Ve = np.zeros((Ntest, config.Nbus))
    Vf = np.zeros((Ntest, config.Nbus))
    Ve[:, bus_Pnet_all] = Vx.real
    Vf[:, bus_Pnet_all] = Vx.imag
    if Vy is not None:
        Ve[:, bus_ZIB_all] = Vy.real
        Vf[:, bus_ZIB_all] = Vy.imag
    
    Pred_Vm = np.sqrt(Ve**2 + Vf**2)
    Pred_Va = np.arctan2(Vf, Ve)
    
    if NZIB > 0:
        Pred_Vm[:, bus_ZIB_all] = np.clip(
            Pred_Vm[:, bus_ZIB_all], config.ngt_VmLb, config.ngt_VmUb
        )
    
    Pred_V = Pred_Vm * np.exp(1j * Pred_Va)
    
    # Prepare test load data
    baseMVA = float(sys_data.baseMVA)
    Pdtest = np.zeros((Ntest, config.Nbus))
    Qdtest = np.zeros((Ntest, config.Nbus))
    
    bus_Pd = ngt_data['bus_Pd']
    bus_Qd = ngt_data['bus_Qd']
    idx_test = ngt_data['idx_test']
    
    Pdtest[:, bus_Pd] = sys_data.RPd[idx_test][:, bus_Pd] / baseMVA
    Qdtest[:, bus_Qd] = sys_data.RQd[idx_test][:, bus_Qd] / baseMVA
    
    # Calculate Pg
    Pred_Pg, Pred_Qg, _, _ = get_genload(
        Pred_V, Pdtest, Qdtest,
        sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
    )
    
    # Get GCI values (aligned with bus_Pg nodes) and cost coefficients
    gci_values = get_gci_for_generation_nodes(config, sys_data)
    gencost = ngt_data['gencost_Pg']
    
    # Compute cost and carbon
    cost_mean, carbon_mean = compute_cost_and_carbon(
        Pred_Pg, gencost, gci_values, baseMVA
    )
    
    return {
        'name': model_name,
        'cost_mean': cost_mean,
        'carbon_mean': carbon_mean,
        'Pred_Pg': Pred_Pg,
    }


def plot_pareto_front(results, ref_point, save_path='results/pareto_front_comparison.png'):
    """
    Plot Pareto front comparing all models.
    
    Args:
        results: List of dicts with 'name', 'cost_mean', 'carbon_mean'
        ref_point: Reference point [cost_ref, carbon_ref] for hypervolume
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 8))
    
    # Extract data
    names = [r['name'] for r in results]
    costs = np.array([r['cost_mean'] for r in results])
    carbons = np.array([r['carbon_mean'] for r in results])
    
    # Color scheme: NGT models in blue gradient, VAE in red
    colors = []
    markers = []
    for name in names:
        if 'VAE' in name:
            colors.append('red')
            markers.append('s')  # square
        elif 'single' in name.lower() or 'lc1.0' in name:
            colors.append('darkblue')
            markers.append('o')
        else:
            # Multi-objective: color gradient based on lambda_cost
            colors.append('steelblue')
            markers.append('o')
    
    # Plot each model
    for i, (name, cost, carbon, color, marker) in enumerate(zip(names, costs, carbons, colors, markers)):
        plt.scatter(cost, carbon, c=color, marker=marker, s=150, label=name, zorder=3, edgecolors='black', linewidths=1)
    
    # Connect NGT points to show Pareto front (sorted by cost)
    ngt_indices = [i for i, name in enumerate(names) if 'NGT' in name]
    if len(ngt_indices) > 1:
        ngt_costs = costs[ngt_indices]
        ngt_carbons = carbons[ngt_indices]
        sorted_idx = np.argsort(ngt_costs)
        plt.plot(ngt_costs[sorted_idx], ngt_carbons[sorted_idx], 
                 'b--', alpha=0.5, linewidth=2, label='_nolegend_')
    
    # Plot reference point
    plt.scatter(ref_point[0], ref_point[1], c='gray', marker='x', s=200, 
                label=f'Ref Point', zorder=2)
    
    # Labels and formatting
    plt.xlabel('Economic Cost ($/h)', fontsize=14)
    plt.ylabel('Carbon Emission (tCO2/h)', fontsize=14)
    plt.title('Pareto Front: Multi-Preference NGT vs VAE Baseline', fontsize=16)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    for i, (name, cost, carbon) in enumerate(zip(names, costs, carbons)):
        short_name = name.replace('NGT_', '').replace('_final', '')
        plt.annotate(short_name, (cost, carbon), textcoords="offset points", 
                     xytext=(5, 5), fontsize=8, alpha=0.8)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPareto front saved to: {save_path}")
    plt.close()


def main():
    """Main evaluation function."""
    print("=" * 70)
    print(" Multi-Preference Model Evaluation & Pareto Front Analysis")
    print("=" * 70)
    
    # Load configuration
    config = get_config()
    device = config.device
    
    print(f"\nConfiguration:")
    print(f"  Nbus: {config.Nbus}")
    print(f"  Device: {device}")
    print(f"  Model directory: {config.model_save_dir}")
    
    # Load data (only once)
    print("\nLoading test data...")
    ngt_data, sys_data = load_ngt_training_data(config)
    
    # Get baseMVA for reference point calculation
    baseMVA = float(sys_data.baseMVA)
    
    # Define model configurations to evaluate
    # Model naming: NetV_ngt_{Nbus}bus{_lc{lambda_cost}}_E{epochs}_final.pth
    n_epochs = config.ngt_Epoch
    model_configs = [
        {'name': 'NGT_single', 'path': f'NetV_ngt_{config.Nbus}bus_single_E{n_epochs}_final.pth', 'lambda_cost': 1.0},
        {'name': 'NGT_lc0.9', 'path': f'NetV_ngt_{config.Nbus}bus_lc0.9_E{n_epochs}_final.pth', 'lambda_cost': 0.9},
        {'name': 'NGT_lc0.7', 'path': f'NetV_ngt_{config.Nbus}bus_lc0.7_E{n_epochs}_final.pth', 'lambda_cost': 0.7},
        {'name': 'NGT_lc0.5', 'path': f'NetV_ngt_{config.Nbus}bus_lc0.5_E{n_epochs}_final.pth', 'lambda_cost': 0.5},
        {'name': 'NGT_lc0.3', 'path': f'NetV_ngt_{config.Nbus}bus_lc0.3_E{n_epochs}_final.pth', 'lambda_cost': 0.3},
        {'name': 'NGT_lc0.1', 'path': f'NetV_ngt_{config.Nbus}bus_lc0.1_E{n_epochs}_final.pth', 'lambda_cost': 0.1},
    ]
    
    # Evaluate NGT models
    print("\n" + "-" * 50)
    print(" Evaluating NGT Models")
    print("-" * 50)
    
    results = []
    for mc in model_configs:
        model_path = os.path.join(config.model_save_dir, mc['path'])
        if not os.path.exists(model_path):
            print(f"  [SKIP] Model not found: {mc['path']}")
            continue
        
        model = load_ngt_model(config, ngt_data, model_path, device)
        result = evaluate_single_model(config, model, ngt_data, sys_data, device, mc['name'])
        result['lambda_cost'] = mc['lambda_cost']
        result['lambda_carbon'] = 1.0 - mc['lambda_cost']
        results.append(result)
        
        print(f"    {mc['name']}: cost={result['cost_mean']:.2f} $/h, carbon={result['carbon_mean']:.4f} tCO2/h")
    
    if len(results) == 0:
        print("\n[ERROR] No trained models found! Please run training first:")
        print("  bash run_batch_training.sh")
        return
    
    # Try to load and evaluate VAE model for comparison
    print("\n" + "-" * 50)
    print(" Evaluating VAE Baseline (if available)")
    print("-" * 50)
    
    vae_vm_path = config.pretrain_model_path_vm
    vae_va_path = config.pretrain_model_path_va
    
    if os.path.exists(vae_vm_path) and os.path.exists(vae_va_path):
        print(f"  Loading VAE models from:")
        print(f"    Vm: {vae_vm_path}")
        print(f"    Va: {vae_va_path}")
        
        try:
            # Load VAE models
            input_dim = ngt_data['input_dim']
            output_dim_vm = config.Nbus
            output_dim_va = config.Nbus - 1
            
            model_vm = create_model('vae', input_dim, output_dim_vm, config, is_vm=True)
            model_va = create_model('vae', input_dim, output_dim_va, config, is_vm=False)
            
            model_vm.to(device)
            model_va.to(device)
            
            model_vm.load_state_dict(torch.load(vae_vm_path, map_location=device, weights_only=True))
            model_va.load_state_dict(torch.load(vae_va_path, map_location=device, weights_only=True))
            
            model_vm.eval()
            model_va.eval()
            
            # Evaluate VAE (simplified - compute Pg from predictions)
            x_test = ngt_data['x_test'].to(device)
            Ntest = x_test.shape[0]
            
            with torch.no_grad():
                Vm_pred = model_vm(x_test, use_mean=True).cpu().numpy()
                Va_pred = model_va(x_test, use_mean=True).cpu().numpy()
            
            # Unscale
            VmLb = sys_data.VmLb.item() if hasattr(sys_data.VmLb, 'item') else float(sys_data.VmLb)
            VmUb = sys_data.VmUb.item() if hasattr(sys_data.VmUb, 'item') else float(sys_data.VmUb)
            scale_vm = config.scale_vm.item() if hasattr(config.scale_vm, 'item') else float(config.scale_vm)
            scale_va = config.scale_va.item() if hasattr(config.scale_va, 'item') else float(config.scale_va)
            
            Vm_pred = Vm_pred / scale_vm * (VmUb - VmLb) + VmLb
            Va_pred = Va_pred / scale_va
            
            # Insert slack bus Va = 0
            bus_slack = int(sys_data.bus_slack)
            Va_full = np.zeros((Ntest, config.Nbus))
            Va_full[:, :bus_slack] = Va_pred[:, :bus_slack]
            Va_full[:, bus_slack+1:] = Va_pred[:, bus_slack:]
            
            Pred_V = Vm_pred * np.exp(1j * Va_full)
            
            # Prepare load data
            Pdtest = np.zeros((Ntest, config.Nbus))
            Qdtest = np.zeros((Ntest, config.Nbus))
            bus_Pd = ngt_data['bus_Pd']
            bus_Qd = ngt_data['bus_Qd']
            idx_test = ngt_data['idx_test']
            Pdtest[:, bus_Pd] = sys_data.RPd[idx_test][:, bus_Pd] / baseMVA
            Qdtest[:, bus_Qd] = sys_data.RQd[idx_test][:, bus_Qd] / baseMVA
            
            # Calculate Pg
            Pred_Pg, _, _, _ = get_genload(
                Pred_V, Pdtest, Qdtest,
                sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
            )
            
            # Compute cost and carbon (use node-aligned GCI values)
            gci_values = get_gci_for_generation_nodes(config, sys_data)
            gencost = ngt_data['gencost_Pg']
            vae_cost, vae_carbon = compute_cost_and_carbon(
                Pred_Pg, gencost, gci_values, baseMVA
            )
            
            results.append({
                'name': 'VAE_baseline',
                'cost_mean': vae_cost,
                'carbon_mean': vae_carbon,
                'lambda_cost': None,
                'lambda_carbon': None,
            })
            
            print(f"    VAE_baseline: cost={vae_cost:.2f} $/h, carbon={vae_carbon:.4f} tCO2/h")
            
        except Exception as e:
            print(f"  [WARNING] Failed to evaluate VAE: {e}")
    else:
        print(f"  [SKIP] VAE model not found at:")
        print(f"    {vae_vm_path}")
        print(f"    {vae_va_path}")
    
    # Compute Pareto front metrics
    print("\n" + "-" * 50)
    print(" Pareto Front Analysis")
    print("-" * 50)
    
    costs = np.array([r['cost_mean'] for r in results])
    carbons = np.array([r['carbon_mean'] for r in results])
    names = [r['name'] for r in results]
    
    # Set reference point (nadir point - worse than all solutions)
    ref_point = np.array([
        np.max(costs) * 1.1,
        np.max(carbons) * 1.1
    ])
    
    print(f"\n  Reference point: cost={ref_point[0]:.2f}, carbon={ref_point[1]:.4f}")
    
    # Compute hypervolume for NGT models only
    ngt_mask = np.array(['NGT' in name for name in names])
    ngt_points = np.column_stack([costs[ngt_mask], carbons[ngt_mask]])
    
    hypervolume = compute_hypervolume(ngt_points, ref_point)
    print(f"\n  Hypervolume (NGT models only): {hypervolume:.2f}")
    
    # Compute hypervolume including VAE
    all_points = np.column_stack([costs, carbons])
    hypervolume_all = compute_hypervolume(all_points, ref_point)
    print(f"  Hypervolume (all models): {hypervolume_all:.2f}")
    
    # Plot Pareto front
    plot_pareto_front(results, ref_point, 
                      save_path=f'{config.results_dir}/pareto_front_comparison.png')
    
    # Print summary table
    print("\n" + "=" * 70)
    print(" Summary Table")
    print("=" * 70)
    print(f"\n{'Model':<20} {'λ_cost':<10} {'λ_carbon':<10} {'Cost ($/h)':<15} {'Carbon (tCO2/h)':<15}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x['cost_mean']):
        lc = f"{r['lambda_cost']:.1f}" if r['lambda_cost'] is not None else "N/A"
        le = f"{r['lambda_carbon']:.1f}" if r['lambda_carbon'] is not None else "N/A"
        print(f"{r['name']:<20} {lc:<10} {le:<10} {r['cost_mean']:<15.2f} {r['carbon_mean']:<15.4f}")
    print("-" * 70)
    print(f"\nHypervolume (NGT): {hypervolume:.2f}")
    print(f"Hypervolume (All): {hypervolume_all:.2f}")
    
    # Save results to JSON
    save_results = {
        'models': [{k: v for k, v in r.items() if k != 'Pred_Pg'} for r in results],
        'hypervolume_ngt': float(hypervolume),
        'hypervolume_all': float(hypervolume_all),
        'ref_point': ref_point.tolist(),
        'config': {
            'Nbus': config.Nbus,
            'ngt_Epoch': config.ngt_Epoch,
        }
    }
    
    json_path = f'{config.results_dir}/multi_preference_results.json'
    os.makedirs(config.results_dir, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {json_path}")
    
    print("\n" + "=" * 70)
    print(" Evaluation completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

