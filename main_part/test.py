#!/usr/bin/env python
# coding: utf-8
# Testing/Inference Script for DeepOPF-V
# Author: Wanjun HUANG
# Date: July 4th, 2021

import torch
import numpy as np
import scipy.io
import time
import matplotlib.pyplot as plt
import argparse

from config import get_config
from models import NetVm, NetVa
from data_loader import load_all_data
from utils import (get_mae, get_rerr, get_rerr2, get_clamp, get_genload, get_Pgcost,
                   get_vioPQg, get_viobran, get_viobran2, dPQbus_dV, get_hisdV,
                   dSlbus_dV)


def load_trained_models(config, input_channels, output_channels_vm, output_channels_va, device, model_path_vm=None, model_path_va=None):
    """
    Load pre-trained models from .pth files
    
    Args:
        config: Configuration object
        input_channels: Number of input features
        output_channels_vm: Number of Vm outputs
        output_channels_va: Number of Va outputs
        device: Device to load models on
        model_path_vm: Custom path for Vm model (optional)
        model_path_va: Custom path for Va model (optional)
        
    Returns:
        model_vm: Loaded Vm model
        model_va: Loaded Va model
    """
    print("=" * 60)
    print("Loading Trained Models")
    print("=" * 60)
    
    # Initialize models
    model_vm = NetVm(input_channels, output_channels_vm, config.hidden_units, config.khidden_Vm)
    model_va = NetVa(input_channels, output_channels_va, config.hidden_units, config.khidden_Va)
    
    # Load model weights
    vm_path = model_path_vm if model_path_vm else config.PATHVm
    va_path = model_path_va if model_path_va else config.PATHVa
    
    try:
        model_vm.load_state_dict(torch.load(vm_path, map_location=device))
        print(f'Vm model loaded from: {vm_path}')
    except FileNotFoundError:
        print(f'Warning: Vm model file not found at {vm_path}')
        print('Please train the model first using train.py')
        return None, None
    
    try:
        model_va.load_state_dict(torch.load(va_path, map_location=device))
        print(f'Va model loaded from: {va_path}')
    except FileNotFoundError:
        print(f'Warning: Va model file not found at {va_path}')
        print('Please train the model first using train.py')
        return None, None
    
    # Set to evaluation mode
    model_vm.eval()
    model_va.eval()
    
    # Move to device
    if torch.cuda.is_available():
        model_vm.to(device)
        model_va.to(device)
        print(f'Models moved to: {device}')
    
    print("Models loaded successfully!")
    print("=" * 60)
    
    return model_vm, model_va


def inference_on_test_set(config, model_vm, model_va, sys_data, dataloaders, device):
    """
    Run inference on test set
    
    Args:
        config: Configuration object
        model_vm: Trained Vm model
        model_va: Trained Va model
        sys_data: System data
        dataloaders: Data loaders
        device: Device
        
    Returns:
        predictions: Dictionary of predictions
    """
    print("\n" + "=" * 60)
    print("Running Inference on Test Set")
    print("=" * 60)
    
    # Predict Vm
    print('Predicting voltage magnitudes...')
    time_start = time.process_time()
    yvmtest_hat = torch.zeros((config.Ntest, config.Nbus))
    
    with torch.no_grad():
        for step, (test_x, test_y) in enumerate(dataloaders['test_vm']):
            test_x = test_x.to(device)
            yvmtest_hat[step] = model_vm(test_x)
    
    time_vm = time.process_time() - time_start
    
    yvmtest_hat = yvmtest_hat.cpu()
    yvmtest_hats = yvmtest_hat.detach() / config.scale_vm * (sys_data.VmUb - sys_data.VmLb) + sys_data.VmLb
    yvmtest_hat_clip = get_clamp(yvmtest_hats, sys_data.hisVm_min, sys_data.hisVm_max)
    
    print(f'Vm prediction completed in {time_vm:.4f} seconds ({time_vm/config.Ntest*1000:.2f} ms/sample)')
    
    # Predict Va
    print('Predicting voltage angles...')
    time_start = time.process_time()
    yvatest_hat = torch.zeros((config.Ntest, config.Nbus - 1))
    
    with torch.no_grad():
        for step, (test_x, test_y) in enumerate(dataloaders['test_va']):
            test_x = test_x.to(device)
            yvatest_hat[step] = model_va(test_x)
    
    time_va = time.process_time() - time_start
    
    yvatest_hat = yvatest_hat.cpu()
    yvatest_hats = yvatest_hat.detach() / config.scale_va
    
    print(f'Va prediction completed in {time_va:.4f} seconds ({time_va/config.Ntest*1000:.2f} ms/sample)')
    
    # Va with slack bus
    Pred_Va = yvatest_hats.clone().numpy()
    Pred_Va = np.insert(Pred_Va, sys_data.bus_slack, values=0, axis=1)
    
    # Complex voltage
    Pred_V = yvmtest_hat_clip.clone().numpy() * np.exp(1j * Pred_Va)
    
    # Calculate power
    Pred_Pg, Pred_Qg, Pred_Pd, Pred_Qd = get_genload(
        Pred_V, sys_data.Pdtest, sys_data.Qdtest,
        sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
    )
    
    predictions = {
        'yvmtest_hat_clip': yvmtest_hat_clip,
        'yvatest_hats': yvatest_hats,
        'Pred_Va': Pred_Va,
        'Pred_V': Pred_V,
        'Pred_Pg': Pred_Pg,
        'Pred_Qg': Pred_Qg,
        'Pred_Pd': Pred_Pd,
        'Pred_Qd': Pred_Qd,
        'time_vm': time_vm,
        'time_va': time_va,
    }
    
    return predictions


def evaluate_predictions(config, predictions, sys_data, BRANFT):
    """
    Evaluate predictions and check constraint violations
    
    Args:
        config: Configuration object
        predictions: Predictions dictionary
        sys_data: System data
        BRANFT: Branch from-to indices
        
    Returns:
        metrics: Evaluation metrics
    """
    print("\n" + "=" * 60)
    print("Evaluating Predictions")
    print("=" * 60)
    
    # Ground truth
    yvmtests = sys_data.yvm_test / config.scale_vm * (sys_data.VmUb - sys_data.VmLb) + sys_data.VmLb
    yvatests = sys_data.yva_test / config.scale_va
    
    Real_Va = yvatests.clone().numpy()
    Real_Va = np.insert(Real_Va, sys_data.bus_slack, values=0, axis=1)
    Real_V = yvmtests.numpy() * np.exp(1j * Real_Va)
    
    Real_Pg, Real_Qg, Real_Pd, Real_Qd = get_genload(
        Real_V, sys_data.Pdtest, sys_data.Qdtest,
        sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
    )
    
    # Prediction accuracy
    mae_Vmtest = get_mae(yvmtests, predictions['yvmtest_hat_clip'].detach())
    mre_Vmtest = get_rerr(yvmtests, predictions['yvmtest_hat_clip'].detach())
    mae_Vatest = get_mae(yvatests, predictions['yvatest_hats'])
    mre_Vatest = get_rerr(yvatests, predictions['yvatest_hats'])
    
    print(f'\nPrediction Accuracy:')
    print(f'  Vm MAE: {mae_Vmtest:.6f} p.u.')
    print(f'  Vm MRE: {torch.mean(mre_Vmtest):.4f}% (max: {torch.max(mre_Vmtest):.4f}%)')
    print(f'  Va MAE: {mae_Vatest:.6f} rad')
    print(f'  Va MRE: {torch.mean(mre_Vatest):.4f}% (max: {torch.max(mre_Vatest):.4f}%)')
    
    # Constraint violations
    lsPg, lsQg, lsidxPg, lsidxQg, vio_PQgmaxmin, vio_PQg, deltaPgL, deltaPgU, deltaQgL, deltaQgU = get_vioPQg(
        predictions['Pred_Pg'], sys_data.bus_Pg, sys_data.MAXMIN_Pg,
        predictions['Pred_Qg'], sys_data.bus_Qg, sys_data.MAXMIN_Qg,
        config.DELTA
    )
    
    vio_branang, vio_branpf, deltapf = get_viobran(
        predictions['Pred_V'], predictions['Pred_Va'], sys_data.branch,
        sys_data.Yf, sys_data.Yt, BRANFT, sys_data.baseMVA, config.DELTA
    )
    
    lsidxPQg = np.squeeze(np.array(np.where((lsidxPg + lsidxQg) > 0)))
    num_violations = np.size(lsidxPQg)
    
    print(f'\nConstraint Satisfaction:')
    print(f'  Samples with violations: {num_violations}/{config.Ntest} ({num_violations/config.Ntest*100:.1f}%)')
    print(f'  Pg constraint: {torch.mean(vio_PQg[:, 0]):.2f}%')
    print(f'  Qg constraint: {torch.mean(vio_PQg[:, 1]):.2f}%')
    print(f'  Branch angle: {torch.mean(vio_branang):.2f}%')
    print(f'  Branch power flow: {torch.mean(vio_branpf):.2f}%')
    
    # Economic performance
    Pred_cost = get_Pgcost(predictions['Pred_Pg'], sys_data.idxPg, sys_data.gencost, sys_data.baseMVA)
    Real_cost = get_Pgcost(Real_Pg, sys_data.idxPg, sys_data.gencost, sys_data.baseMVA)
    mre_cost = get_rerr2(torch.from_numpy(Real_cost), torch.from_numpy(Pred_cost))
    
    mre_Pd = get_rerr(torch.from_numpy(Real_Pd.sum(axis=1)), torch.from_numpy(predictions['Pred_Pd'].sum(axis=1)))
    mre_Qd = get_rerr(torch.from_numpy(Real_Qd.sum(axis=1)), torch.from_numpy(predictions['Pred_Qd'].sum(axis=1)))
    
    print(f'\nEconomic and Load Performance:')
    print(f'  Cost error: {torch.mean(mre_cost):.2f}%')
    print(f'  Active load error: {torch.mean(mre_Pd):.2f}%')
    print(f'  Reactive load error: {torch.mean(mre_Qd):.2f}%')
    
    metrics = {
        'mae_Vmtest': mae_Vmtest,
        'mae_Vatest': mae_Vatest,
        'mre_Vmtest': mre_Vmtest,
        'mre_Vatest': mre_Vatest,
        'vio_PQg': vio_PQg,
        'vio_branang': vio_branang,
        'vio_branpf': vio_branpf,
        'num_violations': num_violations,
        'mre_cost': mre_cost,
        'mre_Pd': mre_Pd,
        'mre_Qd': mre_Qd,
    }
    
    return metrics


def visualize_results(metrics, predictions, sys_data, config):
    """
    Create visualization plots
    
    Args:
        metrics: Evaluation metrics
        predictions: Predictions
        sys_data: System data
        config: Configuration
    """
    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Vm prediction vs ground truth
    yvmtests = sys_data.yvm_test / config.scale_vm * (sys_data.VmUb - sys_data.VmLb) + sys_data.VmLb
    sample_idx = 0
    axes[0, 0].plot(yvmtests[sample_idx].numpy(), 'b-', label='Ground Truth')
    axes[0, 0].plot(predictions['yvmtest_hat_clip'][sample_idx].numpy(), 'r--', label='Prediction')
    axes[0, 0].set_xlabel('Bus Index')
    axes[0, 0].set_ylabel('Voltage Magnitude (p.u.)')
    axes[0, 0].set_title(f'Vm Prediction (Sample {sample_idx})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Va prediction vs ground truth
    yvatests = sys_data.yva_test / config.scale_va
    axes[0, 1].plot(yvatests[sample_idx].numpy(), 'b-', label='Ground Truth')
    axes[0, 1].plot(predictions['yvatest_hats'][sample_idx].numpy(), 'r--', label='Prediction')
    axes[0, 1].set_xlabel('Bus Index (excl. slack)')
    axes[0, 1].set_ylabel('Voltage Angle (rad)')
    axes[0, 1].set_title(f'Va Prediction (Sample {sample_idx})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Vm error distribution
    vm_errors = (predictions['yvmtest_hat_clip'] - yvmtests).numpy().flatten()
    axes[0, 2].hist(vm_errors, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 2].set_xlabel('Prediction Error (p.u.)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Vm Error Distribution')
    axes[0, 2].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Va error distribution
    va_errors = (predictions['yvatest_hats'] - yvatests).numpy().flatten()
    axes[1, 0].hist(va_errors, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Prediction Error (rad)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Va Error Distribution')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Constraint satisfaction
    constraint_types = ['Pg', 'Qg', 'Angle', 'Power Flow']
    satisfaction_rates = [
        torch.mean(metrics['vio_PQg'][:, 0]).item(),
        torch.mean(metrics['vio_PQg'][:, 1]).item(),
        torch.mean(metrics['vio_branang']).item(),
        torch.mean(metrics['vio_branpf']).item()
    ]
    bars = axes[1, 1].bar(constraint_types, satisfaction_rates, alpha=0.7, edgecolor='black')
    axes[1, 1].set_ylabel('Satisfaction Rate (%)')
    axes[1, 1].set_title('Constraint Satisfaction Rates')
    axes[1, 1].set_ylim([0, 105])
    axes[1, 1].axhline(y=100, color='g', linestyle='--', linewidth=2, label='100%')
    axes[1, 1].axhline(y=95, color='orange', linestyle='--', linewidth=1, label='95%')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Color bars based on satisfaction rate
    for bar, rate in zip(bars, satisfaction_rates):
        if rate >= 99:
            bar.set_color('green')
        elif rate >= 95:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Performance summary
    axes[1, 2].axis('off')
    summary_text = (
        f"Performance Summary\n"
        f"{'-' * 30}\n"
        f"Vm MAE: {metrics['mae_Vmtest']:.6f} p.u.\n"
        f"Va MAE: {metrics['mae_Vatest']:.6f} rad\n"
        f"\n"
        f"Violations: {metrics['num_violations']}/{config.Ntest}\n"
        f"({metrics['num_violations']/config.Ntest*100:.1f}%)\n"
        f"\n"
        f"Cost Error: {torch.mean(metrics['mre_cost']):.2f}%\n"
        f"Pd Error: {torch.mean(metrics['mre_Pd']):.2f}%\n"
        f"Qd Error: {torch.mean(metrics['mre_Qd']):.2f}%\n"
        f"\n"
        f"Inference Time:\n"
        f"  Vm: {predictions['time_vm']/config.Ntest*1000:.2f} ms/sample\n"
        f"  Va: {predictions['time_va']/config.Ntest*1000:.2f} ms/sample"
    )
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('test_results.png', dpi=300, bbox_inches='tight')
    print('Visualization saved to: test_results.png')
    plt.close()


def main():
    """
    Main testing/inference function
    """
    parser = argparse.ArgumentParser(description='DeepOPF-V Testing/Inference')
    parser.add_argument('--model_vm', type=str, default=None, help='Path to Vm model file')
    parser.add_argument('--model_va', type=str, default=None, help='Path to Va model file')
    parser.add_argument('--visualize', action='store_true', help='Create visualization plots')
    args = parser.parse_args()
    
    print("=" * 60)
    print("DeepOPF-V Testing/Inference")
    print("=" * 60)
    
    # Load configuration
    config = get_config()
    config.flag_test = 1  # Set to test mode
    config.print_config()
    
    # Load data
    sys_data, dataloaders, BRANFT = load_all_data(config)
    
    # Model dimensions
    input_channels = sys_data.x_test.shape[1]
    output_channels_vm = sys_data.yvm_test.shape[1]
    output_channels_va = sys_data.yva_test.shape[1]
    
    # Load trained models
    model_vm, model_va = load_trained_models(
        config, input_channels, output_channels_vm, output_channels_va,
        config.device, args.model_vm, args.model_va
    )
    
    if model_vm is None or model_va is None:
        print("\nExiting: Models could not be loaded.")
        return
    
    # Run inference
    predictions = inference_on_test_set(config, model_vm, model_va, sys_data, dataloaders, config.device)
    
    # Evaluate predictions
    metrics = evaluate_predictions(config, predictions, sys_data, BRANFT)
    
    # Visualization
    if args.visualize:
        visualize_results(metrics, predictions, sys_data, config)
    
    print("\n" + "=" * 60)
    print("Testing completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

