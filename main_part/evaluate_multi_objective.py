#!/usr/bin/env python
# coding: utf-8
"""
Multi-Objective Evaluation Script for DeepOPF-V

This script evaluates trained models (Simple MLP, VAE, Rectified Flow) 
on both economic cost and carbon emission objectives.

Author: Auto-generated
Date: 2024
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import time
import argparse

from config import get_config
from models import (NetVm, NetVa, create_model, get_available_model_types,
                    load_model_checkpoint, load_model_pair, infer_model_type_from_path)
from data_loader import load_all_data
from utils import (get_mae, get_clamp, get_genload, get_Pgcost, get_vioPQg, 
                   get_viobran, get_viobran2, get_carbon_emission, get_carbon_emission_vectorized,
                   dPQbus_dV, get_hisdV, get_dV, dSlbus_dV)

# GCI Lookup Tables (tCO2/MWh)
# Data source: PGLib-CO2, US EPA, IPCC
FUEL_LOOKUP_CO2 = {
    "ANT": 0.9095,  # Anthracite Coal
    "BIT": 0.8204,  # Bituminous Coal
    "Oil": 0.7001,  # Heavy Oil
    "GAS": 0.5173,  # Natural Gas
    "CCGT": 0.3621,  # Gas Combined Cycle
    "ICE": 0.6030,  # Internal Combustion Engine
    "Thermal": 0.6874,  # Thermal Power (General)
    "NUC": 0.0,     # Nuclear Power
    "RE": 0.0,      # Renewable Energy
    "HYD": 0.0,     # Hydropower
    "N/A": 0.0      # Default case
}


def get_gci_for_generators(config, sys_data):
    """
    Assign GCI (Carbon Emission Intensity) values based on marginal generation cost.
    
    Rationale (suitable for academic papers):
    =========================================
    In real power systems, there exists a fundamental trade-off between generation
    cost and carbon emissions:
    
    - Low marginal cost generators are typically conventional thermal units 
      (coal-fired, oil-fired) with high carbon emission intensities.
      
    - High marginal cost generators are typically cleaner technologies 
      (natural gas combined cycle) with lower carbon emission intensities.
    
    This creates a realistic cost-carbon trade-off for multi-objective optimization.
    
    Args:
        config: Configuration object
        sys_data: PowerSystemData object
        
    Returns:
        gci_values: Array of GCI values for each generator [n_gen]
    """
    n_gen = sys_data.gen.shape[0] if isinstance(sys_data.gen, np.ndarray) else sys_data.gen.numpy().shape[0]
    gencost = sys_data.gencost if isinstance(sys_data.gencost, np.ndarray) else sys_data.gencost.numpy()
    
    gci_values = np.zeros(n_gen)
    
    # Get marginal cost coefficient c1 (column 1 in gencost format [c2, c1, ...])
    c1_values = gencost[:n_gen, 1]
    
    # Compute percentiles for classification
    p25 = np.percentile(c1_values, 25)
    p50 = np.percentile(c1_values, 50)
    p75 = np.percentile(c1_values, 75)
    
    for i in range(n_gen):
        c1 = c1_values[i]
        
        if c1 <= p25:
            # Lowest cost quartile → Coal (highest carbon)
            fuel_type = "BIT" if i % 2 == 0 else "ANT"
        elif c1 <= p50:
            # Second quartile → Heavy Oil
            fuel_type = "Oil"
        elif c1 <= p75:
            # Third quartile → Natural Gas
            fuel_type = "GAS"
        else:
            # Highest cost quartile → CCGT (lowest carbon)
            fuel_type = "CCGT"
        
        gci_values[i] = FUEL_LOOKUP_CO2[fuel_type]
    
    return gci_values


def load_model(model_type, config, input_dim, output_dim_vm, output_dim_va, device, model_paths=None):
    """
    Load a trained model based on model type.
    
    使用统一的 load_model_checkpoint 函数简化模型加载流程。
    
    Args:
        model_type: Type of model ('simple', 'vae', 'rectified')
        config: Configuration object
        input_dim: Input dimension
        output_dim_vm: Vm output dimension
        output_dim_va: Va output dimension
        device: Device to load model on
        model_paths: Dictionary with 'vm' and 'va' paths (optional)
        
    Returns:
        model_vm, model_va: Loaded models (or None if failed)
    """
    print(f"\n--- Loading {model_type} model ---")
    
    # 构建默认模型路径
    if model_type == 'simple':
        # 原始 MLP 模型命名: modelvm300r2N1Lm8642E1000F1.pth
        default_vm_path = f'{config.model_save_dir}/modelvm{config.Nbus}r{config.sys_R}N{config.model_version}{config.nmLm}E{config.EpochVm}F1.pth'
        default_va_path = f'{config.model_save_dir}/modelva{config.Nbus}r{config.sys_R}N{config.model_version}{config.nmLa}E{config.EpochVa}F1.pth'
    else:
        # 生成模型命名: modelvm300r2N1Lm8642_vae_E1000F1.pth
        default_vm_path = os.path.join(config.model_save_dir, 
            f'modelvm{config.Nbus}r{config.sys_R}N{config.model_version}Lm8642_{model_type}_E{config.EpochVm}F1.pth')
        default_va_path = os.path.join(config.model_save_dir, 
            f'modelva{config.Nbus}r{config.sys_R}N{config.model_version}La8642_{model_type}_E{config.EpochVa}F1.pth')
    
    # 使用提供的路径或默认路径
    vm_path = model_paths.get('vm', default_vm_path) if model_paths else default_vm_path
    va_path = model_paths.get('va', default_va_path) if model_paths else default_va_path
    
    # 检查文件是否存在
    if not os.path.exists(vm_path) or not os.path.exists(va_path):
        print(f"  [Warning] Model files not found:")
        print(f"    Vm: {vm_path} (exists: {os.path.exists(vm_path)})")
        print(f"    Va: {va_path} (exists: {os.path.exists(va_path)})")
        return None, None
    
    try:
        # 使用统一加载函数加载 Vm 模型
        model_vm = load_model_checkpoint(
            vm_path,
            config=config,
            input_dim=input_dim,
            output_dim=output_dim_vm,
            is_vm=True,
            device=device,
            model_type=model_type,
            eval_mode=True
        )
        
        # 使用统一加载函数加载 Va 模型
        model_va = load_model_checkpoint(
            va_path,
            config=config,
            input_dim=input_dim,
            output_dim=output_dim_va,
            is_vm=False,
            device=device,
            model_type=model_type,
            eval_mode=True
        )
        
        return model_vm, model_va
        
    except FileNotFoundError as e:
        print(f"  [Warning] Model files not found: {e}")
        return None, None
    except Exception as e:
        print(f"  [Error] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def run_inference(model_vm, model_va, model_type, x_test, config, device):
    """
    Run inference with the model.
    
    Args:
        model_vm: Vm model
        model_va: Va model
        model_type: Type of model
        x_test: Test input tensor
        config: Configuration object
        device: Device
        
    Returns:
        Pred_Vm: Predicted voltage magnitudes [n_samples, n_bus]
        Pred_Va: Predicted voltage angles [n_samples, n_bus-1] (without slack)
        inference_time: Total inference time in seconds
    """
    n_samples = x_test.shape[0]
    n_bus = config.Nbus
    
    x_test = x_test.to(device)
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        if model_type == 'simple':
            # Simple MLP - separate Vm and Va models
            Pred_Vm_scaled = model_vm(x_test).cpu()
            Pred_Va_scaled = model_va(x_test).cpu()
            
        elif model_type == 'vae':
            # VAE model - separate Vm and Va models, use mean for inference
            Pred_Vm_scaled = model_vm(x_test, use_mean=True).cpu()
            Pred_Va_scaled = model_va(x_test, use_mean=True).cpu()
            
        elif model_type in ['rectified', 'flow']:
            # Rectified Flow model - separate Vm and Va models
            # Generate anchor points using pretrain_model (VAE) if available
            
            # For Vm
            if hasattr(model_vm, 'pretrain_model') and model_vm.pretrain_model is not None:
                z_vm = model_vm.pretrain_model(x_test, use_mean=True)
            else:
                z_vm = torch.randn(n_samples, n_bus).to(device)
            
            # For Va (n_bus - 1 because slack bus is excluded)
            if hasattr(model_va, 'pretrain_model') and model_va.pretrain_model is not None:
                z_va = model_va.pretrain_model(x_test, use_mean=True)
            else:
                z_va = torch.randn(n_samples, n_bus - 1).to(device)
            
            # Get inference steps from config
            inf_step = getattr(config, 'inf_step', 20)
            step_size = 1.0 / inf_step
            
            # Run backward flow for Vm
            Pred_Vm_scaled, _ = model_vm.flow_backward(x_test, z_vm, step=step_size, method='Euler')
            Pred_Vm_scaled = Pred_Vm_scaled.cpu()
            
            # Run backward flow for Va
            Pred_Va_scaled, _ = model_va.flow_backward(x_test, z_va, step=step_size, method='Euler')
            Pred_Va_scaled = Pred_Va_scaled.cpu()
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    
    return Pred_Vm_scaled, Pred_Va_scaled, inference_time


def evaluate_model(model_vm, model_va, model_type, sys_data, config, device, gci_values, BRANFT, apply_post_processing=True):
    """
    Evaluate a model on the test set with optional post-processing.
    
    Args:
        model_vm, model_va: Models to evaluate
        model_type: Type of model
        sys_data: PowerSystemData object
        config: Configuration object
        device: Device
        gci_values: GCI values for generators
        BRANFT: Branch from-to indices
        apply_post_processing: Whether to apply post-processing corrections
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Run inference
    Pred_Vm_scaled, Pred_Va_scaled, inference_time = run_inference(
        model_vm, model_va, model_type, sys_data.x_test, config, device
    )
    
    # Denormalize Vm
    VmLb = sys_data.VmLb.numpy() if isinstance(sys_data.VmLb, torch.Tensor) else sys_data.VmLb
    VmUb = sys_data.VmUb.numpy() if isinstance(sys_data.VmUb, torch.Tensor) else sys_data.VmUb
    
    Pred_Vm = Pred_Vm_scaled.numpy() / config.scale_vm.item() * (VmUb - VmLb) + VmLb
    
    # Clamp Vm to historical range
    hisVm_min = sys_data.hisVm_min.numpy() if isinstance(sys_data.hisVm_min, torch.Tensor) else sys_data.hisVm_min
    hisVm_max = sys_data.hisVm_max.numpy() if isinstance(sys_data.hisVm_max, torch.Tensor) else sys_data.hisVm_max
    yvmtest_hat_clip = torch.from_numpy(np.clip(Pred_Vm, hisVm_min, hisVm_max)).float()
    Pred_Vm = yvmtest_hat_clip.numpy()
    
    # Denormalize Va
    Pred_Va_no_slack = Pred_Va_scaled.numpy() / config.scale_va.item()
    
    # Insert slack bus (angle = 0)
    Pred_Va = np.insert(Pred_Va_no_slack, sys_data.bus_slack, values=0, axis=1)
    
    # Complex voltage
    Pred_V = Pred_Vm * np.exp(1j * Pred_Va)
    
    # Calculate power generation
    Pred_Pg, Pred_Qg, Pred_Pd, Pred_Qd = get_genload(
        Pred_V, sys_data.Pdtest, sys_data.Qdtest,
        sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
    )
    
    # Calculate economic cost
    Pred_cost = get_Pgcost(Pred_Pg, sys_data.idxPg, sys_data.gencost, sys_data.baseMVA)
    
    # Calculate carbon emission
    Pred_carbon = get_carbon_emission_vectorized(Pred_Pg, gci_values[sys_data.idxPg], sys_data.baseMVA)
    
    # ==================== Calculate Load Satisfaction Rate ====================
    # Load satisfaction rate measures how well the power balance is maintained
    # at load buses: P_injection + Pd = 0 (for load buses without generation)
    
    # Get load bus indices (all buses except generator buses)
    all_buses = np.arange(config.Nbus)
    load_bus_idx = np.setdiff1d(all_buses, sys_data.bus_Pg)
    
    # Calculate power injection directly from voltage
    # S = V * (Y * V)^*, P = Re(S)
    P_injection = np.zeros((Pred_V.shape[0], config.Nbus))
    Q_injection = np.zeros((Pred_V.shape[0], config.Nbus))
    for i in range(Pred_V.shape[0]):
        I = sys_data.Ybus.dot(Pred_V[i]).conj()
        S = np.multiply(Pred_V[i], I)
        P_injection[i] = np.real(S)
        Q_injection[i] = np.imag(S)
    
    # Load demand at load buses (Pdtest, Qdtest are positive for load consumption)
    Pd_demand = sys_data.Pdtest[:, load_bus_idx]  # (n_test, n_load_buses)
    Qd_demand = sys_data.Qdtest[:, load_bus_idx]
    
    # Power injection at load buses
    P_inj_load = P_injection[:, load_bus_idx]
    Q_inj_load = Q_injection[:, load_bus_idx]
    
    # Load balance error: |P_injection + Pd_demand| (should be 0 for perfect balance)
    # For load bus: P_injection = -Pd (power flows out of generator, into load)
    Pd_balance_error = np.abs(P_inj_load + Pd_demand)  # (n_test, n_load_buses)
    Qd_balance_error = np.abs(Q_inj_load + Qd_demand)
    
    # Total load per sample (sum of all load buses)
    Pd_total_per_sample = np.sum(np.abs(Pd_demand), axis=1, keepdims=True)  # (n_test, 1)
    Qd_total_per_sample = np.sum(np.abs(Qd_demand), axis=1, keepdims=True)
    
    # Total balance error per sample
    Pd_total_error_per_sample = np.sum(Pd_balance_error, axis=1)  # (n_test,)
    Qd_total_error_per_sample = np.sum(Qd_balance_error, axis=1)
    
    # Relative error normalized by total load (more meaningful metric)
    eps = 1e-6
    Pd_rel_error_normalized = Pd_total_error_per_sample / np.maximum(Pd_total_per_sample.squeeze(), eps)
    Qd_rel_error_normalized = Qd_total_error_per_sample / np.maximum(Qd_total_per_sample.squeeze(), eps)
    
    # Mean relative error (as percentage of total load)
    Pd_mean_rel_error = np.mean(Pd_rel_error_normalized) * 100
    Qd_mean_rel_error = np.mean(Qd_rel_error_normalized) * 100
    
    # Load satisfaction rate = 100% - mean_relative_error%
    Pd_satisfy_rate_before = max(0.0, 100.0 - Pd_mean_rel_error)
    Qd_satisfy_rate_before = max(0.0, 100.0 - Qd_mean_rel_error)
    
    # Per-sample satisfaction (total error < 5% of total load)
    threshold = 0.05  # 5% threshold
    sample_Pd_satisfied = Pd_rel_error_normalized < threshold
    sample_Qd_satisfied = Qd_rel_error_normalized < threshold
    Pd_sample_satisfy_rate_before = np.mean(sample_Pd_satisfied) * 100
    Qd_sample_satisfy_rate_before = np.mean(sample_Qd_satisfied) * 100
    
    # Mean absolute error in MW
    baseMVA = sys_data.baseMVA.item() if hasattr(sys_data.baseMVA, 'item') else float(sys_data.baseMVA)
    Pd_mae_MW = np.mean(Pd_balance_error) * baseMVA
    Qd_mae_MVAr = np.mean(Qd_balance_error) * baseMVA
    
    # Total load for reference (in MW)
    Pd_total_avg_MW = np.mean(Pd_total_per_sample) * baseMVA
    
    # Check constraint violations (before post-processing)
    lsPg, lsQg, lsidxPg, lsidxQg, vio_PQgmaxmin, vio_PQg, _, _, _, _ = get_vioPQg(
        Pred_Pg, sys_data.bus_Pg, sys_data.MAXMIN_Pg,
        Pred_Qg, sys_data.bus_Qg, sys_data.MAXMIN_Qg,
        config.DELTA
    )
    
    vio_branang, vio_branpf, deltapf, vio_branpfidx, lsSf, _, lsSf_sampidx, _ = get_viobran2(
        Pred_V, Pred_Va, sys_data.branch, sys_data.Yf, sys_data.Yt,
        BRANFT, sys_data.baseMVA, config.DELTA
    )
    
    # Calculate violation statistics (before post-processing)
    lsidxPQg = np.squeeze(np.array(np.where((lsidxPg + lsidxQg) > 0)))
    num_violations_before = np.size(lsidxPQg)
    vio_branpf_num = np.size(np.where(vio_branpfidx > 0))
    lsSf_sampidx = np.asarray(lsSf_sampidx)
    
    # Compute ground truth for comparison
    yvmtests = sys_data.yvm_test / config.scale_vm * (torch.tensor(VmUb) - torch.tensor(VmLb)) + torch.tensor(VmLb)
    yvatests = sys_data.yva_test / config.scale_va
    
    Real_Va = yvatests.numpy()
    Real_Va = np.insert(Real_Va, sys_data.bus_slack, values=0, axis=1)
    Real_V = yvmtests.numpy() * np.exp(1j * Real_Va)
    
    Real_Pg, Real_Qg, Real_Pd, Real_Qd = get_genload(
        Real_V, sys_data.Pdtest, sys_data.Qdtest,
        sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
    )
    
    Real_cost = get_Pgcost(Real_Pg, sys_data.idxPg, sys_data.gencost, sys_data.baseMVA)
    Real_carbon = get_carbon_emission_vectorized(Real_Pg, gci_values[sys_data.idxPg], sys_data.baseMVA)
    
    # ==================== Post-Processing ====================
    if apply_post_processing and num_violations_before > 0:
        print(f"    Applying post-processing to {num_violations_before} violated samples...")
        
        time_start_post = time.perf_counter()
        
        # Calculate Jacobian matrices using historical voltage
        dPbus_dV, dQbus_dV = dPQbus_dV(sys_data.his_V, sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus)
        
        # Build incidence matrix for branch constraints
        finc = np.zeros((sys_data.branch.shape[0], config.Nbus), dtype=float)
        for i in range(sys_data.branch.shape[0]):
            finc[i, int(sys_data.branch[i, 0]) - 1] = 1
        
        bus_Va = np.delete(np.arange(config.Nbus), sys_data.bus_slack)
        dPfbus_dV, dQfbus_dV = dSlbus_dV(sys_data.his_V, bus_Va, sys_data.branch, sys_data.Yf, finc, BRANFT, config.Nbus)
        
        # Calculate voltage corrections
        Pred_Va1 = Pred_Va.copy()
        Pred_Vm1 = yvmtest_hat_clip.clone().numpy()
        
        if config.flag_hisv:
            dV1 = get_hisdV(lsPg, lsQg, lsidxPg, lsidxQg, num_violations_before, config.k_dV,
                           sys_data.bus_Pg, sys_data.bus_Qg, dPbus_dV, dQbus_dV,
                           config.Nbus, config.Ntest)
        else:
            dV1 = get_dV(Pred_V, lsPg, lsQg, lsidxPg, lsidxQg, num_violations_before, config.k_dV,
                        sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus, sys_data.his_V)
        
        # Branch power flow corrections
        if vio_branpf_num > 0 and len(lsSf_sampidx) > 0:
            dV_branch = np.zeros((lsSf_sampidx.shape[0], config.Nbus * 2))
            for i in range(lsSf_sampidx.shape[0]):
                if lsSf[i].shape[0] > 0:
                    mp = np.array(lsSf[i][:, 2] / (lsSf[i][:, 1] + 1e-8)).reshape(-1, 1)
                    mq = np.array(lsSf[i][:, 3] / (lsSf[i][:, 1] + 1e-8)).reshape(-1, 1)
                    branch_idx = np.array(lsSf[i][:, 0].astype(int)).squeeze()
                    if branch_idx.ndim == 0:
                        branch_idx = np.array([branch_idx])
                    dPdV = dPfbus_dV[branch_idx, :]
                    dQdV = dQfbus_dV[branch_idx, :]
                    if dPdV.ndim == 1:
                        dPdV = dPdV.reshape(1, -1)
                        dQdV = dQdV.reshape(1, -1)
                    dmp = mp * dPdV
                    dmq = mq * dQdV
                    dmpq_inv = np.linalg.pinv(dmp + dmq)
                    dV_branch[i] = np.dot(dmpq_inv, np.array(lsSf[i][:, 1])).squeeze()
            # Match indices: dV1 has num_violations_before rows, dV_branch has len(lsSf_sampidx) rows
            # We need to add branch corrections to the corresponding samples
            for i, samp_idx in enumerate(lsSf_sampidx):
                # Find position in lsidxPQg
                pos = np.where(lsidxPQg == samp_idx)[0]
                if len(pos) > 0:
                    dV1[pos[0]] += dV_branch[i]
        
        # Apply corrections
        Pred_Va1[lsidxPQg, :] = Pred_Va[lsidxPQg, :] - dV1[:, 0:config.Nbus]
        Pred_Va1[:, sys_data.bus_slack] = 0
        Pred_Vm1[lsidxPQg, :] = yvmtest_hat_clip.numpy()[lsidxPQg, :] - dV1[:, config.Nbus:2*config.Nbus]
        Pred_Vm1_clip = get_clamp(torch.from_numpy(Pred_Vm1), sys_data.hisVm_min, sys_data.hisVm_max)
        
        # Recalculate complex voltage after post-processing
        Pred_V1 = Pred_Vm1_clip.numpy() * np.exp(1j * Pred_Va1)
        Pred_Pg1, Pred_Qg1, Pred_Pd1, Pred_Qd1 = get_genload(
            Pred_V1, sys_data.Pdtest, sys_data.Qdtest,
            sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
        )
        
        time_end_post = time.perf_counter()
        time_post_processing = time_end_post - time_start_post
        
        # Re-evaluate constraints after post-processing
        _, _, lsidxPg1, lsidxQg1, _, vio_PQg1, _, _, _, _ = get_vioPQg(
            Pred_Pg1, sys_data.bus_Pg, sys_data.MAXMIN_Pg,
            Pred_Qg1, sys_data.bus_Qg, sys_data.MAXMIN_Qg,
            config.DELTA
        )
        lsidxPQg1 = np.squeeze(np.array(np.where(lsidxPg1 + lsidxQg1 > 0)))
        num_violations_after = np.size(lsidxPQg1)
        
        vio_branang1, vio_branpf1, _ = get_viobran(
            Pred_V1, Pred_Va1, sys_data.branch, sys_data.Yf, sys_data.Yt,
            BRANFT, sys_data.baseMVA, config.DELTA
        )
        
        # Recalculate cost and carbon after post-processing
        Pred_cost1 = get_Pgcost(Pred_Pg1, sys_data.idxPg, sys_data.gencost, sys_data.baseMVA)
        Pred_carbon1 = get_carbon_emission_vectorized(Pred_Pg1, gci_values[sys_data.idxPg], sys_data.baseMVA)
        
        # Use post-processed results
        final_vio_PQg = vio_PQg1
        final_vio_branang = vio_branang1
        final_vio_branpf = vio_branpf1
        final_num_violations = num_violations_after
        final_cost = Pred_cost1
        final_carbon = Pred_carbon1
        
    else:
        time_post_processing = 0.0
        final_vio_PQg = vio_PQg
        final_vio_branang = vio_branang
        final_vio_branpf = vio_branpf
        final_num_violations = num_violations_before
        final_cost = Pred_cost
        final_carbon = Pred_carbon
        num_violations_after = num_violations_before
    
    metrics = {
        'model_type': model_type,
        # Before post-processing
        'cost_mean_before': np.mean(Pred_cost),
        'cost_std_before': np.std(Pred_cost),
        'carbon_mean_before': np.mean(Pred_carbon),
        'carbon_std_before': np.std(Pred_carbon),
        'pg_satisfy_before': torch.mean(vio_PQg[:, 0]).item(),
        'qg_satisfy_before': torch.mean(vio_PQg[:, 1]).item(),
        'num_violations_before': num_violations_before,
        # Load satisfaction rate (before post-processing)
        'pd_satisfy_before': Pd_satisfy_rate_before,
        'qd_satisfy_before': Qd_satisfy_rate_before,
        'pd_sample_satisfy_before': Pd_sample_satisfy_rate_before,
        'qd_sample_satisfy_before': Qd_sample_satisfy_rate_before,
        'pd_mean_rel_error': Pd_mean_rel_error,
        'qd_mean_rel_error': Qd_mean_rel_error,
        'pd_mae_MW': Pd_mae_MW,
        'qd_mae_MVAr': Qd_mae_MVAr,
        'pd_total_avg_MW': Pd_total_avg_MW,
        # After post-processing (or same as before if not applied)
        'cost_mean': np.mean(final_cost),
        'cost_std': np.std(final_cost),
        'carbon_mean': np.mean(final_carbon),
        'carbon_std': np.std(final_carbon),
        'real_cost_mean': np.mean(Real_cost),
        'real_carbon_mean': np.mean(Real_carbon),
        'pg_satisfy': torch.mean(final_vio_PQg[:, 0]).item(),
        'qg_satisfy': torch.mean(final_vio_PQg[:, 1]).item(),
        'branch_angle_satisfy': torch.mean(final_vio_branang).item(),
        'branch_pf_satisfy': torch.mean(final_vio_branpf).item(),
        # Load satisfaction (same as before, post-processing doesn't change load)
        'pd_satisfy': Pd_satisfy_rate_before,
        'qd_satisfy': Qd_satisfy_rate_before,
        'pd_sample_satisfy': Pd_sample_satisfy_rate_before,
        'qd_sample_satisfy': Qd_sample_satisfy_rate_before,
        'num_violations': final_num_violations,
        'total_samples': config.Ntest,
        'num_load_buses': len(load_bus_idx),
        'inference_time': inference_time,
        'inference_time_per_sample': inference_time / config.Ntest * 1000,  # ms
        'post_processing_time': time_post_processing,
    }
    
    return metrics


def print_results_table(all_metrics, config):
    """
    Print a formatted results table.
    
    Args:
        all_metrics: List of metric dictionaries
        config: Configuration object
    """
    print("\n" + "=" * 140)
    print("Multi-Objective Evaluation Results (After Post-Processing)")
    print("=" * 140)
    
    # Header
    print(f"\n{'Model':<15} | {'Cost ($/h)':<25} | {'Carbon (tCO2/h)':<20} | {'Pg Satisfy':<12} | {'Pd Satisfy':<12} | {'Pd Error':<10} | {'Violations':<15}")
    print("-" * 140)
    
    # Ground truth row
    if all_metrics:
        gt = all_metrics[0]  # Use first model's ground truth
        print(f"{'Ground Truth':<15} | {gt['real_cost_mean']:>22.2f}   | {gt['real_carbon_mean']:>17.4f}   | {'100.00%':>10} | {'100.00%':>10} | {'0.00%':>8} | {'0':>13}")
    
    print("-" * 140)
    
    # Model rows
    for m in all_metrics:
        model_name = m['model_type'].upper()
        cost_str = f"{m['cost_mean']:.2f} +/- {m['cost_std']:.2f}"
        carbon_str = f"{m['carbon_mean']:.4f} +/- {m['carbon_std']:.4f}"
        pg_str = f"{m['pg_satisfy']:.2f}%"
        pd_str = f"{m['pd_satisfy']:.2f}%"
        pd_err_str = f"{m['pd_mean_rel_error']:.2f}%"
        vio_str = f"{m['num_violations']}/{m['total_samples']} ({m['num_violations']/m['total_samples']*100:.1f}%)"
        
        print(f"{model_name:<15} | {cost_str:<25} | {carbon_str:<20} | {pg_str:>10} | {pd_str:>10} | {pd_err_str:>8} | {vio_str:>13}")
    
    # Additional load satisfaction details
    print("\n" + "-" * 140)
    print("Load Satisfaction Details (Active Power, normalized by total load):")
    print(f"{'Model':<15} | {'Rel Error (%)':<14} | {'Satisfy (%)':<12} | {'MAE (MW)':<12} | {'Total Load (MW)':<16} | {'Samples <5%':<12} | {'Load Buses':<10}")
    print("-" * 120)
    for m in all_metrics:
        model_name = m['model_type'].upper()
        print(f"{model_name:<15} | {m['pd_mean_rel_error']:>12.4f}   | {m['pd_satisfy']:>10.2f}   | {m.get('pd_mae_MW', 0):>10.4f} | {m.get('pd_total_avg_MW', 0):>14.2f} | {m['pd_sample_satisfy']:>10.2f}%   | {m.get('num_load_buses', 'N/A'):>8}")
    
    print("=" * 120)
    
    # Before vs After Post-Processing comparison
    print("\n" + "=" * 80)
    print("Before vs After Post-Processing Comparison")
    print("=" * 80)
    print(f"\n{'Model':<15} | {'Violations Before':<20} | {'Violations After':<20} | {'Improvement':<15}")
    print("-" * 80)
    
    for m in all_metrics:
        model_name = m['model_type'].upper()
        vio_before = m.get('num_violations_before', m['num_violations'])
        vio_after = m['num_violations']
        improvement = vio_before - vio_after
        improvement_pct = (improvement / vio_before * 100) if vio_before > 0 else 0
        
        before_str = f"{vio_before}/{m['total_samples']} ({vio_before/m['total_samples']*100:.1f}%)"
        after_str = f"{vio_after}/{m['total_samples']} ({vio_after/m['total_samples']*100:.1f}%)"
        impr_str = f"-{improvement} ({improvement_pct:.1f}%)"
        
        print(f"{model_name:<15} | {before_str:<20} | {after_str:<20} | {impr_str:<15}")
    
    print("=" * 80)
    
    # Additional statistics
    print("\nAdditional Statistics:")
    print("-" * 60)
    for m in all_metrics:
        print(f"\n{m['model_type'].upper()}:")
        print(f"  Branch Angle Satisfy: {m['branch_angle_satisfy']:.2f}%")
        print(f"  Branch Power Flow Satisfy: {m['branch_pf_satisfy']:.2f}%")
        
        # Cost comparison with ground truth
        cost_diff = (m['cost_mean'] - m['real_cost_mean']) / m['real_cost_mean'] * 100
        carbon_diff = (m['carbon_mean'] - m['real_carbon_mean']) / m['real_carbon_mean'] * 100
        print(f"  Cost vs GT: {cost_diff:+.2f}%")
        print(f"  Carbon vs GT: {carbon_diff:+.2f}%")
        
        # Post-processing time if available
        if 'post_processing_time' in m:
            print(f"  Post-processing time: {m['post_processing_time']:.4f}s")


# ==================== Pareto Model Evaluation Functions ====================

def load_pareto_model(config, input_dim, output_dim_vm, output_dim_va, device, 
                      lambda_cost=0.9, lambda_carbon=0.1, checkpoint='final'):
    """
    Load Pareto-adapted flow models.
    
    Args:
        config: Configuration object
        input_dim: Input dimension
        output_dim_vm: Vm output dimension
        output_dim_va: Va output dimension
        device: Device to load model on
        lambda_cost: Cost weight used in training
        lambda_carbon: Carbon weight used in training
        checkpoint: Checkpoint to load ('final' or epoch number)
        
    Returns:
        model_vm, model_va: Loaded Pareto flow models (or None if not found)
        vae_vm, vae_va: Loaded VAE anchor models (or None if not found)
    """
    print(f"\n--- Loading Pareto Flow Model [{lambda_cost}, {lambda_carbon}] ---")
    
    # Construct model paths
    pref_str = f"{lambda_cost}_{lambda_carbon}".replace(".", "")
    suffix = "final" if checkpoint == 'final' else f"E{checkpoint}"
    
    vm_path = os.path.join(config.model_save_dir, f'modelvm_pareto_{pref_str}_{suffix}.pth')
    va_path = os.path.join(config.model_save_dir, f'modelva_pareto_{pref_str}_{suffix}.pth')
    
    # Check if Pareto models exist
    if not os.path.exists(vm_path) or not os.path.exists(va_path):
        print(f"  [Warning] Pareto model files not found:")
        print(f"    Vm: {vm_path} (exists: {os.path.exists(vm_path)})")
        print(f"    Va: {va_path} (exists: {os.path.exists(va_path)})")
        return None, None, None, None
    
    try:
        # Load Pareto flow models
        model_vm = load_model_checkpoint(
            vm_path,
            config=config,
            input_dim=input_dim,
            output_dim=output_dim_vm,
            is_vm=True,
            device=device,
            model_type='rectified',
            eval_mode=True
        )
        
        model_va = load_model_checkpoint(
            va_path,
            config=config,
            input_dim=input_dim,
            output_dim=output_dim_va,
            is_vm=False,
            device=device,
            model_type='rectified',
            eval_mode=True
        )
        
        # Also load VAE anchor models
        vae_vm, vae_va = load_model('vae', config, input_dim, output_dim_vm, output_dim_va, device)
        
        # Attach VAE to flow models if needed
        if vae_vm is not None:
            model_vm.pretrain_model = vae_vm
            model_va.pretrain_model = vae_va
        
        print(f"  Pareto model loaded successfully!")
        return model_vm, model_va, vae_vm, vae_va
        
    except Exception as e:
        print(f"  [Error] Failed to load Pareto model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def evaluate_pareto_comparison(sys_data, config, device, gci_values, BRANFT,
                               lambda_cost=0.9, lambda_carbon=0.1):
    """
    Compare VAE [1,0] solutions with Pareto flow [lambda_cost, lambda_carbon] solutions.
    
    Args:
        sys_data: PowerSystemData object
        config: Configuration object
        device: Device
        gci_values: GCI values for generators
        BRANFT: Branch from-to indices
        lambda_cost: Cost weight for Pareto model
        lambda_carbon: Carbon weight for Pareto model
        
    Returns:
        comparison_results: Dictionary with comparison metrics
    """
    print("\n" + "=" * 80)
    print(f"Pareto Comparison: VAE [1,0] vs Flow [{lambda_cost}, {lambda_carbon}]")
    print("=" * 80)
    
    input_dim = sys_data.x_test.shape[1]
    output_dim_vm = sys_data.yvm_test.shape[1]
    output_dim_va = sys_data.yva_test.shape[1]
    
    # Load VAE model (baseline: [1,0] preference)
    vae_vm, vae_va = load_model('vae', config, input_dim, output_dim_vm, output_dim_va, device)
    
    if vae_vm is None:
        print("  [Error] Could not load VAE model for baseline comparison")
        return None
    
    # Evaluate VAE (baseline) - with post-processing
    print("\n--- Evaluating VAE Baseline [1, 0] ---")
    vae_metrics = evaluate_model(vae_vm, vae_va, 'vae', sys_data, config, device, gci_values, BRANFT, apply_post_processing=True)
    
    # Load Pareto model
    pareto_vm, pareto_va, _, _ = load_pareto_model(
        config, input_dim, output_dim_vm, output_dim_va, device,
        lambda_cost=lambda_cost, lambda_carbon=lambda_carbon
    )
    
    if pareto_vm is None:
        print(f"  [Error] Could not load Pareto model [{lambda_cost}, {lambda_carbon}]")
        return {'vae': vae_metrics, 'pareto': None}
    
    # Evaluate Pareto model (with post-processing)
    print(f"\n--- Evaluating Pareto Flow [{lambda_cost}, {lambda_carbon}] ---")
    pareto_metrics = evaluate_model(pareto_vm, pareto_va, 'rectified', sys_data, config, device, gci_values, BRANFT, apply_post_processing=True)
    pareto_metrics['model_type'] = f'pareto_{lambda_cost}_{lambda_carbon}'
    pareto_metrics['lambda_cost'] = lambda_cost
    pareto_metrics['lambda_carbon'] = lambda_carbon
    
    # Compute comparison statistics
    comparison = {
        'vae': vae_metrics,
        'pareto': pareto_metrics,
        'cost_reduction': vae_metrics['cost_mean'] - pareto_metrics['cost_mean'],
        'carbon_reduction': vae_metrics['carbon_mean'] - pareto_metrics['carbon_mean'],
        'cost_reduction_pct': (vae_metrics['cost_mean'] - pareto_metrics['cost_mean']) / vae_metrics['cost_mean'] * 100,
        'carbon_reduction_pct': (vae_metrics['carbon_mean'] - pareto_metrics['carbon_mean']) / vae_metrics['carbon_mean'] * 100,
    }
    
    # Print comparison summary
    print("\n" + "=" * 80)
    print("Pareto Comparison Summary")
    print("=" * 90)
    print(f"\n{'Metric':<30} | {'VAE [1,0]':<20} | {'Pareto Flow':<20} | {'Change':<15}")
    print("-" * 90)
    print(f"{'Cost ($/h)':<30} | {vae_metrics['cost_mean']:>18.2f} | {pareto_metrics['cost_mean']:>18.2f} | {comparison['cost_reduction']:>+13.2f}")
    print(f"{'Carbon (tCO2/h)':<30} | {vae_metrics['carbon_mean']:>18.4f} | {pareto_metrics['carbon_mean']:>18.4f} | {comparison['carbon_reduction']:>+13.4f}")
    print(f"{'Pg Satisfy (%)':<30} | {vae_metrics['pg_satisfy']:>18.2f} | {pareto_metrics['pg_satisfy']:>18.2f} | {pareto_metrics['pg_satisfy'] - vae_metrics['pg_satisfy']:>+13.2f}")
    print(f"{'Qg Satisfy (%)':<30} | {vae_metrics['qg_satisfy']:>18.2f} | {pareto_metrics['qg_satisfy']:>18.2f} | {pareto_metrics['qg_satisfy'] - vae_metrics['qg_satisfy']:>+13.2f}")
    print(f"{'Pd Satisfy (load, %)':<30} | {vae_metrics['pd_satisfy']:>18.2f} | {pareto_metrics['pd_satisfy']:>18.2f} | {pareto_metrics['pd_satisfy'] - vae_metrics['pd_satisfy']:>+13.2f}")
    print(f"{'Pd MAE (MW)':<30} | {vae_metrics.get('pd_mae_MW', 0):>18.4f} | {pareto_metrics.get('pd_mae_MW', 0):>18.4f} | {pareto_metrics.get('pd_mae_MW', 0) - vae_metrics.get('pd_mae_MW', 0):>+13.4f}")
    print(f"{'Violations':<30} | {vae_metrics['num_violations']:>18d} | {pareto_metrics['num_violations']:>18d} | {pareto_metrics['num_violations'] - vae_metrics['num_violations']:>+13d}")
    print("=" * 90)
    
    print(f"\nCost-Carbon Trade-off Analysis:")
    print(f"  Cost change: {comparison['cost_reduction_pct']:+.2f}%")
    print(f"  Carbon change: {comparison['carbon_reduction_pct']:+.2f}%")
    
    if comparison['carbon_reduction'] > 0:
        # Pareto model reduced carbon
        cost_per_carbon = -comparison['cost_reduction'] / comparison['carbon_reduction'] if comparison['carbon_reduction'] != 0 else float('inf')
        print(f"  Cost of reducing carbon: {cost_per_carbon:.2f} $/tCO2")
    
    return comparison


def curriculum_chain_inference(x_test, vae_vm, vae_va, flow_models, config, device, inf_steps=10):
    """
    Perform chain inference through curriculum flow models.
    
    Args:
        x_test: Test input data [n_samples, input_dim]
        vae_vm, vae_va: VAE anchor models
        flow_models: List of (flow_vm, flow_va) tuples in order
        config: Configuration object
        device: Device
        inf_steps: Number of inference steps per flow model
        
    Returns:
        Pred_Vm_scaled, Pred_Va_scaled: Final predictions
    """
    import time
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        # Start with VAE anchor
        z_vm = vae_vm(x_test, use_mean=True)
        z_va = vae_va(x_test, use_mean=True)
        
        # Apply each flow model in the chain
        for flow_vm, flow_va in flow_models:
            batch_size = x_test.shape[0]
            dt = 1.0 / inf_steps
            
            # Flow forward for Vm
            for step_idx in range(inf_steps):
                t = step_idx * dt
                t_tensor = torch.full((batch_size, 1), t, device=device)
                v_vm = flow_vm.model(x_test, z_vm, t_tensor)
                z_vm = z_vm + v_vm * dt
            
            # Flow forward for Va
            z_va_temp = z_va.clone()
            for step_idx in range(inf_steps):
                t = step_idx * dt
                t_tensor = torch.full((batch_size, 1), t, device=device)
                v_va = flow_va.model(x_test, z_va_temp, t_tensor)
                z_va_temp = z_va_temp + v_va * dt
            z_va = z_va_temp
    
    inference_time = time.perf_counter() - start_time
    
    return z_vm.cpu(), z_va.cpu(), inference_time


def evaluate_curriculum_models(sys_data, config, device, gci_values, BRANFT,
                               preference_schedule, inf_steps=10):
    """
    Evaluate curriculum learning models on test set.
    
    This function evaluates the chain of flow models for each preference level.
    For preference [0.9, 0.1], it uses: VAE -> flow_09_01
    For preference [0.8, 0.2], it uses: VAE -> flow_09_01 -> flow_08_02
    And so on...
    
    Args:
        sys_data: PowerSystemData object
        config: Configuration object
        device: Device
        gci_values: GCI values for generators
        BRANFT: Branch from-to indices
        preference_schedule: List of (lambda_cost, lambda_carbon) tuples
        inf_steps: Inference steps per flow model
        
    Returns:
        Dictionary with evaluation results for each preference
    """
    input_dim = sys_data.x_test.shape[1]
    output_dim_vm = sys_data.yvm_test.shape[1]
    output_dim_va = sys_data.yva_test.shape[1]
    
    # Load VAE models as anchor
    print("\n--- Loading VAE Anchor Models ---")
    vae_vm, vae_va = load_model('vae', config, input_dim, output_dim_vm, output_dim_va, device)
    
    if vae_vm is None:
        print("  [Error] Could not load VAE model")
        return None
    
    # First, evaluate VAE baseline
    print("\n--- Evaluating VAE Baseline [1, 0] ---")
    vae_metrics = evaluate_model(vae_vm, vae_va, 'vae', sys_data, config, device, 
                                  gci_values, BRANFT, apply_post_processing=False)
    
    # Load all flow models
    print("\n--- Loading Curriculum Flow Models ---")
    flow_models_dict = {}
    
    for lambda_cost, lambda_carbon in preference_schedule:
        pref_str = f"{lambda_cost}_{lambda_carbon}".replace(".", "")
        vm_path = os.path.join(config.model_save_dir, f'modelvm_pareto_{pref_str}_final.pth')
        va_path = os.path.join(config.model_save_dir, f'modelva_pareto_{pref_str}_final.pth')
        
        if not os.path.exists(vm_path) or not os.path.exists(va_path):
            print(f"  [Warning] Model for [{lambda_cost}, {lambda_carbon}] not found, skipping")
            continue
        
        try:
            flow_vm = load_model_checkpoint(
                vm_path, config=config, input_dim=input_dim, output_dim=output_dim_vm,
                is_vm=True, device=device, model_type='rectified', eval_mode=True
            )
            flow_va = load_model_checkpoint(
                va_path, config=config, input_dim=input_dim, output_dim=output_dim_va,
                is_vm=False, device=device, model_type='rectified', eval_mode=True
            )
            flow_models_dict[(lambda_cost, lambda_carbon)] = (flow_vm, flow_va)
            print(f"  Loaded [{lambda_cost}, {lambda_carbon}] flow model")
        except Exception as e:
            print(f"  [Error] Failed to load [{lambda_cost}, {lambda_carbon}]: {e}")
            continue
    
    # Evaluate each preference level with chain inference
    all_results = {'vae': vae_metrics}
    
    # Handle both tensor and numpy array
    if isinstance(sys_data.x_test, torch.Tensor):
        x_test = sys_data.x_test.float().to(device)
    else:
        x_test = torch.from_numpy(sys_data.x_test).float().to(device)
    
    # Build chain incrementally and evaluate
    current_chain = []
    
    for idx, (lambda_cost, lambda_carbon) in enumerate(preference_schedule):
        if (lambda_cost, lambda_carbon) not in flow_models_dict:
            continue
        
        # Add this stage to the chain
        current_chain.append(flow_models_dict[(lambda_cost, lambda_carbon)])
        
        print(f"\n--- Evaluating Curriculum Chain [{lambda_cost}, {lambda_carbon}] (chain length: {len(current_chain)}) ---")
        
        # Run chain inference
        Pred_Vm_scaled, Pred_Va_scaled, inference_time = curriculum_chain_inference(
            x_test, vae_vm, vae_va, current_chain, config, device, inf_steps
        )
        
        # Denormalize predictions (same as evaluate_model)
        VmLb = sys_data.VmLb.numpy() if isinstance(sys_data.VmLb, torch.Tensor) else sys_data.VmLb
        VmUb = sys_data.VmUb.numpy() if isinstance(sys_data.VmUb, torch.Tensor) else sys_data.VmUb
        Pred_Vm = Pred_Vm_scaled.numpy() / config.scale_vm.item() * (VmUb - VmLb) + VmLb
        
        # Clamp Vm to historical range
        hisVm_min = sys_data.hisVm_min.numpy() if isinstance(sys_data.hisVm_min, torch.Tensor) else sys_data.hisVm_min
        hisVm_max = sys_data.hisVm_max.numpy() if isinstance(sys_data.hisVm_max, torch.Tensor) else sys_data.hisVm_max
        Pred_Vm = np.clip(Pred_Vm, hisVm_min, hisVm_max)
        
        Pred_Va_no_slack = Pred_Va_scaled.numpy() / config.scale_va.item()
        Pred_Va = np.insert(Pred_Va_no_slack, sys_data.bus_slack, values=0, axis=1)
        
        # Compute complex voltage
        Pred_V = Pred_Vm * np.exp(1j * Pred_Va)
        
        # Calculate power generation
        Real_Pg, Real_Qg, Pred_Pd, Pred_Qd = get_genload(
            Pred_V, sys_data.Pdtest, sys_data.Qdtest,
            sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
        )
        
        # Check constraint violations
        _, _, _, _, _, vio_PQg, _, _, lsidxPQg, _ = get_vioPQg(
            Real_Pg, sys_data.bus_Pg, sys_data.MAXMIN_Pg,
            Real_Qg, sys_data.bus_Qg, sys_data.MAXMIN_Qg,
            config.DELTA
        )
        # lsidxPQg is a list of indices where violations occur
        num_violations = len(lsidxPQg) if isinstance(lsidxPQg, list) else np.size(lsidxPQg)
        
        # Calculate costs
        Pred_cost = get_Pgcost(Real_Pg, sys_data.idxPg, sys_data.gencost, sys_data.baseMVA)
        Pred_carbon = get_carbon_emission_vectorized(Real_Pg, gci_values[sys_data.idxPg], sys_data.baseMVA)
        
        # Calculate load satisfaction
        load_bus_idx = sys_data.load_bus_idx if hasattr(sys_data, 'load_bus_idx') else np.where(np.mean(np.abs(sys_data.Pdtest), axis=0) > 1e-6)[0]
        
        # Calculate power injection
        P_injection = np.zeros((Pred_V.shape[0], config.Nbus))
        for i in range(Pred_V.shape[0]):
            I = sys_data.Ybus.dot(Pred_V[i]).conj()
            S = np.multiply(Pred_V[i], I)
            P_injection[i] = np.real(S)
        
        # Load demand at load buses
        Pd_demand = sys_data.Pdtest[:, load_bus_idx]
        P_inj_load = P_injection[:, load_bus_idx]
        
        # Load balance error
        Pd_balance_error = np.abs(P_inj_load + Pd_demand)
        Pd_total_per_sample = np.sum(np.abs(Pd_demand), axis=1, keepdims=True)
        Pd_total_error_per_sample = np.sum(Pd_balance_error, axis=1)
        
        # Relative error
        eps = 1e-6
        Pd_rel_error = Pd_total_error_per_sample / np.maximum(Pd_total_per_sample.squeeze(), eps)
        Pd_mean_rel_error = np.mean(Pd_rel_error) * 100
        Pd_satisfy_rate = max(0.0, 100.0 - Pd_mean_rel_error)
        
        # MAE in MW
        baseMVA = sys_data.baseMVA.item() if hasattr(sys_data.baseMVA, 'item') else float(sys_data.baseMVA)
        Pd_mae_MW = np.mean(Pd_total_error_per_sample) * baseMVA
        
        metrics = {
            'model_type': f'curriculum_{lambda_cost}_{lambda_carbon}',
            'lambda_cost': lambda_cost,
            'lambda_carbon': lambda_carbon,
            'chain_length': len(current_chain),
            'cost_mean': float(np.mean(Pred_cost)),
            'cost_std': float(np.std(Pred_cost)),
            'carbon_mean': float(np.mean(Pred_carbon)),
            'carbon_std': float(np.std(Pred_carbon)),
            'pg_satisfy': float(torch.mean(vio_PQg[:, 0]).item()),
            'qg_satisfy': float(torch.mean(vio_PQg[:, 1]).item()),
            'num_violations': num_violations,
            'load_satisfy_pct': Pd_satisfy_rate,
            'load_mae_mw': Pd_mae_MW,
            'load_rel_error_pct': Pd_mean_rel_error,
            'inference_time': inference_time,
            'inference_time_per_sample': inference_time / config.Ntest * 1000,
        }
        
        all_results[f'{lambda_cost}_{lambda_carbon}'] = metrics
        
        # Print results
        print(f"  Cost: {metrics['cost_mean']:.2f} $/h")
        print(f"  Carbon: {metrics['carbon_mean']:.2f} tCO2/h")
        print(f"  Pg Satisfy: {metrics['pg_satisfy']:.2f}%")
        print(f"  Load Satisfy: {metrics['load_satisfy_pct']:.2f}%")
        print(f"  Load MAE: {metrics['load_mae_mw']:.4f} MW")
        print(f"  Violations: {metrics['num_violations']}")
    
    # Print summary table
    print("\n" + "=" * 100)
    print("CURRICULUM LEARNING EVALUATION SUMMARY")
    print("=" * 100)
    print(f"{'Model':<25} | {'Cost ($/h)':<15} | {'Carbon (tCO2/h)':<15} | {'Pg Sat (%)':<12} | {'Pd Sat (%)':<12} | {'Pd MAE (MW)':<12}")
    print("-" * 100)
    
    # Print VAE baseline - use pd_satisfy instead of load_satisfy_pct for VAE
    v = all_results['vae']
    vae_pd_satisfy = v.get('pd_satisfy', v.get('load_satisfy_pct', 0))
    vae_pd_mae = v.get('pd_mae_MW', v.get('load_mae_mw', 0))
    print(f"{'VAE [1.0, 0.0]':<25} | {v['cost_mean']:<15.2f} | {v['carbon_mean']:<15.2f} | {v['pg_satisfy']:<12.2f} | {vae_pd_satisfy:<12.2f} | {vae_pd_mae:<12.4f}")
    
    # Print curriculum stages
    for lambda_cost, lambda_carbon in preference_schedule:
        key = f'{lambda_cost}_{lambda_carbon}'
        if key in all_results:
            m = all_results[key]
            model_name = f"Flow [{lambda_cost}, {lambda_carbon}]"
            print(f"{model_name:<25} | {m['cost_mean']:<15.2f} | {m['carbon_mean']:<15.2f} | {m['pg_satisfy']:<12.2f} | {m['load_satisfy_pct']:<12.2f} | {m['load_mae_mw']:<12.4f}")
    
    print("=" * 100)
    
    # Compare with VAE
    print("\nChange from VAE baseline:")
    for lambda_cost, lambda_carbon in preference_schedule:
        key = f'{lambda_cost}_{lambda_carbon}'
        if key in all_results:
            m = all_results[key]
            cost_change = (m['cost_mean'] - vae_metrics['cost_mean']) / vae_metrics['cost_mean'] * 100
            carbon_change = (m['carbon_mean'] - vae_metrics['carbon_mean']) / vae_metrics['carbon_mean'] * 100
            print(f"  [{lambda_cost}, {lambda_carbon}]: Cost {cost_change:+.2f}%, Carbon {carbon_change:+.2f}%")
    
    return all_results


def plot_pareto_comparison(comparison_results, save_path=None):
    """
    Plot cost-carbon trade-off visualization.
    
    Args:
        comparison_results: Dictionary from evaluate_pareto_comparison
        save_path: Path to save the figure
    """
    import matplotlib.pyplot as plt
    
    if comparison_results is None or comparison_results.get('pareto') is None:
        print("  [Warning] Cannot plot: missing comparison data")
        return
    
    vae = comparison_results['vae']
    pareto = comparison_results['pareto']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Cost vs Carbon scatter
    ax1 = axes[0]
    ax1.scatter(vae['cost_mean'], vae['carbon_mean'], 
                s=200, c='blue', marker='o', label='VAE [1, 0]', zorder=3)
    ax1.scatter(pareto['cost_mean'], pareto['carbon_mean'], 
                s=200, c='red', marker='s', 
                label=f"Pareto [{pareto['lambda_cost']}, {pareto['lambda_carbon']}]", zorder=3)
    
    # Draw arrow showing trade-off direction
    ax1.annotate('', xy=(pareto['cost_mean'], pareto['carbon_mean']),
                 xytext=(vae['cost_mean'], vae['carbon_mean']),
                 arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    ax1.set_xlabel('Economic Cost ($/h)', fontsize=12)
    ax1.set_ylabel('Carbon Emission (tCO2/h)', fontsize=12)
    ax1.set_title('Cost-Carbon Trade-off', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Bar chart comparison
    ax2 = axes[1]
    metrics = ['Cost ($/h)', 'Carbon (×1000)', 'Pg Satisfy (%)', 'Qg Satisfy (%)']
    vae_vals = [vae['cost_mean'], vae['carbon_mean'] * 1000, vae['pg_satisfy'], vae['qg_satisfy']]
    pareto_vals = [pareto['cost_mean'], pareto['carbon_mean'] * 1000, pareto['pg_satisfy'], pareto['qg_satisfy']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, vae_vals, width, label='VAE [1, 0]', color='blue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, pareto_vals, width, 
                    label=f"Pareto [{pareto['lambda_cost']}, {pareto['lambda_carbon']}]", 
                    color='red', alpha=0.7)
    
    ax2.set_xlabel('Metric', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Model Performance Comparison', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, rotation=15, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Comparison plot saved to: {save_path}")
    
    plt.close()


def main():
    """Main function for multi-objective evaluation."""
    parser = argparse.ArgumentParser(description='Multi-Objective Evaluation for DeepOPF-V')
    parser.add_argument('--models', nargs='+', default=['simple', 'vae', 'rectified'], 
                        choices=['simple', 'vae', 'rectified'],
                        help='Model types to evaluate (default: all three)')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Directory containing model files')
    parser.add_argument('--inf_step', type=int, default=20,
                        help='Inference steps for flow model (default: 20)')
    parser.add_argument('--inf_steps', type=int, default=10,
                        help='Inference steps for curriculum flow model (default: 10)')
    # Pareto evaluation arguments
    parser.add_argument('--pareto', action='store_true',
                        help='Evaluate Pareto-adapted flow model and compare with VAE')
    parser.add_argument('--lambda_cost', type=float, default=0.9,
                        help='Cost weight for Pareto model (default: 0.9)')
    parser.add_argument('--lambda_carbon', type=float, default=0.1,
                        help='Carbon weight for Pareto model (default: 0.1)')
    parser.add_argument('--pareto_checkpoint', type=str, default='final',
                        help='Pareto model checkpoint to load (default: final)')
    # Curriculum evaluation arguments
    parser.add_argument('--curriculum', action='store_true',
                        help='Evaluate curriculum learning models (chain of flow models)')
    parser.add_argument('--start_cost', type=float, default=1.0,
                        help='Starting cost weight for curriculum (default: 1.0)')
    parser.add_argument('--end_cost', type=float, default=0.7,
                        help='Ending cost weight for curriculum (default: 0.7)')
    parser.add_argument('--pref_step', type=float, default=0.1,
                        help='Preference step size for curriculum (default: 0.1)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Multi-Objective Evaluation for DeepOPF-V")
    print("=" * 60)
    
    # Load configuration
    config = get_config()
    
    # Override inference steps if provided
    if args.inf_step:
        config.inf_step = args.inf_step
    
    config.print_config()
    
    device = config.device
    print(f"\nUsing device: {device}")
    print(f"Inference steps for flow model: {config.inf_step}")
    
    # Load data
    print("\nLoading data...")
    sys_data, dataloaders, BRANFT = load_all_data(config)
    
    # Get model dimensions
    input_dim = sys_data.x_test.shape[1]
    output_dim_vm = sys_data.yvm_test.shape[1]
    output_dim_va = sys_data.yva_test.shape[1]
    
    print(f"\nModel dimensions:")
    print(f"  Input: {input_dim}")
    print(f"  Vm output: {output_dim_vm}")
    print(f"  Va output: {output_dim_va}")
    
    # Get GCI values for generators
    print("\nConfiguring generator carbon intensities (GCI)...")
    gci_values = get_gci_for_generators(config, sys_data)
    print(f"  GCI range: [{gci_values.min():.4f}, {gci_values.max():.4f}] tCO2/MWh")
    print(f"  Generators with GCI > 0: {np.sum(gci_values > 0)}/{len(gci_values)}")
    
    # ==================== Curriculum Learning Evaluation Mode ====================
    if args.curriculum:
        print("\n" + "=" * 60)
        print("Curriculum Learning Evaluation Mode")
        print("=" * 60)
        
        # Generate preference schedule
        preference_schedule = []
        current_cost = args.start_cost - args.pref_step
        eps = 1e-9
        while current_cost >= args.end_cost - eps:
            lambda_cost = round(current_cost, 2)
            lambda_carbon = round(1.0 - lambda_cost, 2)
            preference_schedule.append((lambda_cost, lambda_carbon))
            current_cost -= args.pref_step
        
        print(f"Evaluating {len(preference_schedule)} curriculum stages:")
        for lc, le in preference_schedule:
            print(f"  [{lc:.2f}, {le:.2f}]")
        
        # Run curriculum evaluation
        curriculum_results = evaluate_curriculum_models(
            sys_data, config, device, gci_values, BRANFT,
            preference_schedule,
            inf_steps=args.inf_steps
        )
        
        print("\n" + "=" * 60)
        print("Curriculum Learning Evaluation completed!")
        print("=" * 60)
        return
    
    # ==================== Pareto Evaluation Mode ====================
    if args.pareto:
        print("\n" + "=" * 60)
        print("Pareto Evaluation Mode")
        print(f"Comparing VAE [1,0] vs Pareto Flow [{args.lambda_cost}, {args.lambda_carbon}]")
        print("=" * 60)
        
        # Run Pareto comparison
        comparison_results = evaluate_pareto_comparison(
            sys_data, config, device, gci_values, BRANFT,
            lambda_cost=args.lambda_cost,
            lambda_carbon=args.lambda_carbon
        )
        
        # Plot comparison
        if comparison_results is not None:
            pref_str = f"{args.lambda_cost}_{args.lambda_carbon}".replace(".", "")
            plot_path = os.path.join(config.results_dir, f'pareto_comparison_{pref_str}.png')
            plot_pareto_comparison(comparison_results, save_path=plot_path)
        
        print("\n" + "=" * 60)
        print("Pareto Evaluation completed!")
        print("=" * 60)
        return
    
    # ==================== Standard Evaluation Mode ====================
    # Evaluate each model type
    all_metrics = []
    
    for model_type in args.models:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_type.upper()}")
        print('='*60)
        
        # Load model
        model_vm, model_va = load_model(
            model_type, config, input_dim, output_dim_vm, output_dim_va, device
        )
        
        if model_vm is None:
            print(f"  [Skip] Could not load {model_type} model")
            continue
        
        # Evaluate
        try:
            metrics = evaluate_model(
                model_vm, model_va, model_type, 
                sys_data, config, device, gci_values, BRANFT
            )
            all_metrics.append(metrics)
            
            print(f"\n  Cost: {metrics['cost_mean']:.2f} +/- {metrics['cost_std']:.2f} $/h")
            print(f"  Carbon: {metrics['carbon_mean']:.4f} +/- {metrics['carbon_std']:.4f} tCO2/h")
            print(f"  Pg Satisfy: {metrics['pg_satisfy']:.2f}%")
            print(f"  Inference: {metrics['inference_time_per_sample']:.2f} ms/sample")
            
        except Exception as e:
            print(f"  [Error] Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary table
    if all_metrics:
        print_results_table(all_metrics, config)
    else:
        print("\n[Warning] No models were successfully evaluated.")
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
