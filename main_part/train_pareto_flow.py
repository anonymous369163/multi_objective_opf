#!/usr/bin/env python
# coding: utf-8
"""
Training Script for Pareto-Adaptive Rectified Flow Model

This script trains a rectified flow model to map from VAE-generated 
economic-only solutions (preference [1,0]) to Pareto-optimal solutions 
under a target preference (e.g., [0.9, 0.1]).

The training uses unsupervised multi-objective loss computed on the 
final output of the flow model.

Usage:
    python train_pareto_flow.py --lambda_cost 0.9 --lambda_carbon 0.1 --epochs 500

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
import matplotlib.pyplot as plt

from config import get_config
from models import create_model, load_model_checkpoint, PreferenceConditionedFM
from data_loader import load_all_data
from multi_objective_loss import MultiObjectiveOPFLoss, get_gci_for_generators
from utils import (
    TensorBoardLogger, 
    create_tensorboard_logger, 
    compute_trajectory_metrics,
    # GPU utilities
    gpu_memory_cleanup,
    check_gpu_temperature,
    add_training_delay,
    # Preference sampling
    sample_preferences_uniform,
    sample_preferences_curriculum,
    # Model initialization
    initialize_flow_model_near_zero,
    # Pareto evaluation
    compute_hypervolume,
    check_feasibility,
    evaluate_pareto_front,
    get_pareto_validation_metric,
)

# Import constraint projection for feasibility-aware training
from flow_model.post_processing import ConstraintProjectionV2


# ============================================================================
# Differentiable Flow Forward for Unified Model  
# ============================================================================

def differentiable_flow_forward_unified(flow_model, x, z_anchor, preference, num_steps=10):
    """
    Differentiable flow integration for unified preference-conditioned model.
    
    Maintains full gradient chain across all 10 integration steps for proper
    backpropagation. Gradients are automatically cleared between batches via
    optimizer.zero_grad(), so there's no gradient accumulation across different
    preference mappings.
    
    Args:
        flow_model: PreferenceConditionedFM instance (shared for both Vm and Va)
        x: Condition input [batch, input_dim]
        z_anchor: Starting point (VAE anchor) [batch, output_dim]
        preference: Preference vector [batch, 2]
        num_steps: Number of integration steps
        
    Returns:
        z_final: Final output after flow integration
    """
    batch_size = x.shape[0]
    device = x.device
    
    # Detach anchor to prevent gradient flowing back to VAE (VAE is frozen)
    z = z_anchor.detach().clone()
    z.requires_grad_(True)
    
    dt = 1.0 / num_steps
    
    for step_idx in range(num_steps):
        # Create new time tensor each step (avoid in-place modification issue)
        t_tensor = torch.full((batch_size, 1), step_idx * dt, device=device)
        
        # Predict velocity with preference conditioning - maintains gradient flow
        v = flow_model.model(x, z, t_tensor, preference)
        
        # Euler step - maintains full gradient chain through all steps
        z = z + v * dt
    
    return z


def differentiable_flow_forward_unified_projected(
    flow_model_vm, flow_model_va, x, z_vm, z_va, preference,
    P_tan_t, num_buses, num_steps=10
):
    """
    Differentiable flow integration for unified model with tangent-space projection.
    
    Args:
        flow_model_vm, flow_model_va: Unified flow models for Vm and Va
        x: Condition input [batch, input_dim]
        z_vm: Starting point for Vm [batch, num_buses]
        z_va: Starting point for Va [batch, num_buses-1]
        preference: Preference vector [batch, 2]
        P_tan_t: Tangent space projection matrix
        num_buses: Number of buses
        num_steps: Integration steps
        
    Returns:
        z_vm_final, z_va_final: Final outputs after projected flow integration
    """
    batch_size = x.shape[0]
    device = x.device
    dt = 1.0 / num_steps
    
    # Detach anchors and enable gradient
    z_vm = z_vm.detach().clone().requires_grad_(True)
    z_va = z_va.detach().clone().requires_grad_(True)
    
    for step_idx in range(num_steps):
        t_tensor = torch.full((batch_size, 1), step_idx * dt, device=device)
        
        # Predict velocities with preference conditioning
        v_vm = flow_model_vm.model(x, z_vm, t_tensor, preference)
        v_va = flow_model_va.model(x, z_va, t_tensor, preference)
        
        # Combine and project to tangent space
        v_combined = torch.cat([v_vm, v_va], dim=1)
        v_projected = torch.matmul(v_combined, P_tan_t.T)
        
        # Split back
        v_vm_proj = v_projected[:, :num_buses]
        v_va_proj = v_projected[:, num_buses:]
        
        # Euler step
        z_vm = z_vm + v_vm_proj * dt
        z_va = z_va + v_va_proj * dt
    
    return z_vm, z_va


def validation_flow_forward_unified(
    flow_model_vm, flow_model_va, test_x, z_vm, z_va, preferences,
    num_steps, use_projection, P_tan_t, num_buses
):
    """
    Flow forward for unified model validation, optionally with projection.
    
    Args:
        flow_model_vm, flow_model_va: Unified flow models
        test_x: Input condition
        z_vm, z_va: VAE anchors
        preferences: Preference tensor [batch, 2]
        num_steps: Integration steps
        use_projection: Use tangent-space projection
        P_tan_t: Precomputed projection matrix
        num_buses: Number of buses
        
    Returns:
        y_pred_vm, y_pred_va: Final predictions
    """
    if use_projection and P_tan_t is not None:
        y_pred_vm, y_pred_va = differentiable_flow_forward_unified_projected(
            flow_model_vm, flow_model_va, test_x, z_vm, z_va, preferences,
            P_tan_t, num_buses, num_steps=num_steps
        )
    else:
        y_pred_vm = differentiable_flow_forward_unified(
            flow_model_vm, test_x, z_vm, preferences, num_steps=num_steps
        )
        y_pred_va = differentiable_flow_forward_unified(
            flow_model_va, test_x, z_va, preferences, num_steps=num_steps
        )
    
    return y_pred_vm, y_pred_va


def load_pretrained_vae(config, input_dim, output_dim, is_vm, device):
    """
    Load pretrained VAE model as anchor generator.
    
    Args:
        config: Configuration object
        input_dim: Input dimension
        output_dim: Output dimension
        is_vm: True for Vm model, False for Va model
        device: Device to load model on
        
    Returns:
        vae_model: Loaded VAE model in eval mode
    """
    model_name = "Vm" if is_vm else "Va"
    
    # Construct VAE model path
    if is_vm:
        vae_path = config.pretrain_model_path_vm
    else:
        vae_path = config.pretrain_model_path_va
    
    if not os.path.exists(vae_path):
        raise FileNotFoundError(f"Pretrained VAE model not found: {vae_path}")
    
    print(f"[{model_name}] Loading pretrained VAE from: {vae_path}")
    
    # Create and load VAE model
    vae_model = create_model('vae', input_dim, output_dim, config, is_vm=is_vm)
    vae_model.to(device)
    state_dict = torch.load(vae_path, map_location=device, weights_only=True)
    vae_model.load_state_dict(state_dict)
    vae_model.eval()
    
    # Freeze VAE parameters (we don't want to update them)
    for param in vae_model.parameters():
        param.requires_grad = False
    
    print(f"[{model_name}] VAE model loaded and frozen successfully!")
    return vae_model


def create_flow_model(config, input_dim, output_dim, is_vm, device, 
                      pretrained_flow_path=None):
    """
    Create or load a flow model for Pareto adaptation.
    
    Args:
        config: Configuration object
        input_dim: Input dimension
        output_dim: Output dimension
        is_vm: True for Vm model, False for Va model
        device: Device
        pretrained_flow_path: Optional path to pretrained flow model
        
    Returns:
        flow_model: Flow model
    """
    model_name = "Vm" if is_vm else "Va"
    
    # Create flow model
    flow_model = create_model('rectified', input_dim, output_dim, config, is_vm=is_vm)
    flow_model.to(device)
    
    # Optionally load pretrained flow weights
    if pretrained_flow_path and os.path.exists(pretrained_flow_path):
        print(f"[{model_name}] Loading pretrained flow model from: {pretrained_flow_path}")
        state_dict = torch.load(pretrained_flow_path, map_location=device, weights_only=True)
        flow_model.load_state_dict(state_dict, strict=False)
        print(f"[{model_name}] Pretrained flow model loaded!")
    else:
        print(f"[{model_name}] Initializing fresh flow model")
    
    return flow_model


def differentiable_flow_forward(flow_model, x, z_anchor, num_steps=10):
    """
    Differentiable flow integration for training.
    
    Maintains full gradient chain across all integration steps for proper
    backpropagation through the entire flow trajectory.
    
    Args:
        flow_model: Flow model (FM class)
        x: Condition input [batch, input_dim]
        z_anchor: Starting point (VAE anchor) [batch, output_dim]
        num_steps: Number of integration steps
        
    Returns:
        z_final: Final output after flow integration [batch, output_dim]
    """
    batch_size = x.shape[0]
    device = x.device
    
    # Detach anchor to prevent gradient flowing back to VAE (VAE is frozen)
    # But enable gradient for flow model training
    z = z_anchor.detach().clone()
    z.requires_grad_(True)
    
    dt = 1.0 / num_steps
    
    for step_idx in range(num_steps):
        # Create new time tensor each step (avoid in-place modification issue)
        t_tensor = torch.full((batch_size, 1), step_idx * dt, device=device)
        
        # Predict velocity - maintains gradient flow
        v = flow_model.model(x, z, t_tensor)
        
        # Euler step - maintains full gradient chain
        z = z + v * dt
    
    return z


def differentiable_flow_forward_projected(
    flow_model_vm, flow_model_va, x, z_vm, z_va, 
    P_tan_t, num_buses, num_steps=10
):
    """
    Differentiable flow integration with constraint tangent-space projection.
    
    Maintains full gradient chain across all integration steps.
    
    Args:
        flow_model_vm: Flow model for Vm
        flow_model_va: Flow model for Va  
        x: Condition input [batch, input_dim]
        z_vm: Starting point for Vm [batch, num_buses]
        z_va: Starting point for Va [batch, num_buses-1] (no slack)
        P_tan_t: Tangent space projection matrix
        num_buses: Number of buses in the system
        num_steps: Number of integration steps
        
    Returns:
        z_vm_final, z_va_final: Final outputs after projected flow integration
    """
    batch_size = x.shape[0]
    device = x.device
    dt = 1.0 / num_steps
    
    # Detach anchors and enable gradient
    z_vm = z_vm.detach().clone().requires_grad_(True)
    z_va = z_va.detach().clone().requires_grad_(True)
    
    for step_idx in range(num_steps):
        # Create new time tensor each step
        t_tensor = torch.full((batch_size, 1), step_idx * dt, device=device)
        
        # Predict velocities - maintains gradient flow
        v_vm = flow_model_vm.model(x, z_vm, t_tensor)
        v_va = flow_model_va.model(x, z_va, t_tensor)
        
        # Combine and project
        v_combined = torch.cat([v_vm, v_va], dim=1)
        v_projected = torch.matmul(v_combined, P_tan_t.T)
        
        # Split back
        v_vm_proj = v_projected[:, :num_buses]
        v_va_proj = v_projected[:, num_buses:]
        
        # Euler step - maintains full gradient chain
        z_vm = z_vm + v_vm_proj * dt
        z_va = z_va + v_va_proj * dt
    
    return z_vm, z_va


def differentiable_flow_forward_drift_correction(
    flow_model_vm, flow_model_va, x, z_vm, z_va, 
    projector, num_buses, num_steps=10, lambda_cor=5.0,
    include_load_balance=False
):
    """
    Differentiable flow integration with Drift-Correction.
    
    Maintains full gradient chain across all integration steps.
    
    Formula: v_final = P_tan @ v_pred - λ * F⁺ @ g(z)
    
    Args:
        flow_model_vm, flow_model_va: Flow models
        x: Condition input [batch, input_dim]
        z_vm, z_va: Starting points
        projector: ConstraintProjectionV2 instance
        num_buses: Number of buses
        num_steps: Integration steps
        lambda_cor: Correction strength
        include_load_balance: Include load balance in projection
        
    Returns:
        z_vm_final, z_va_final: Final outputs
    """
    batch_size = x.shape[0]
    device = x.device
    dt = 1.0 / num_steps
    
    # Detach anchors and enable gradient
    z_vm = z_vm.detach().clone().requires_grad_(True)
    z_va = z_va.detach().clone().requires_grad_(True)
    
    # Precompute projection matrix
    P_tan, F, F_pinv = projector.compute_projection_matrix(
        include_slack=False, 
        include_load_balance=include_load_balance
    )
    P_tan_t = torch.tensor(P_tan, dtype=torch.float32, device=device)
    F_pinv_t = torch.tensor(F_pinv, dtype=torch.float32, device=device) if F_pinv is not None else None
    
    # Check for NaN in projection matrices
    if torch.isnan(P_tan_t).any():
        print("[WARNING] P_tan contains NaN! Falling back to identity projection.")
        P_tan_t = torch.eye(P_tan_t.shape[0], device=device)
    if F_pinv_t is not None and torch.isnan(F_pinv_t).any():
        print("[WARNING] F_pinv contains NaN! Disabling normal correction.")
        F_pinv_t = None
    
    for step_idx in range(num_steps):
        # Create new time tensor each step
        t_tensor = torch.full((batch_size, 1), step_idx * dt, device=device)
        
        # Predict velocities - maintains gradient flow
        v_vm = flow_model_vm.model(x, z_vm, t_tensor)
        v_va = flow_model_va.model(x, z_va, t_tensor)
        
        # Combine and project
        v_combined = torch.cat([v_vm, v_va], dim=1)
        v_tangent = torch.matmul(v_combined, P_tan_t.T)
        
        # Normal correction (computed without gradient to avoid circular dependency)
        if F_pinv_t is not None:
            with torch.no_grad():
                z_combined = torch.cat([z_vm.detach(), z_va.detach()], dim=1)
                residual = projector.compute_constraint_residual_batch(z_combined, x, num_buses)
                if residual is not None:
                    residual_t = torch.as_tensor(residual, dtype=torch.float32, device=device)
                    correction_raw = torch.matmul(residual_t, F_pinv_t.T)
                    # Clip correction to prevent numerical explosion
                    correction_norm = torch.norm(correction_raw, dim=1, keepdim=True)
                    max_correction = 1.0  # Maximum allowed correction magnitude
                    scale = torch.clamp(max_correction / (correction_norm + 1e-8), max=1.0)
                    correction = -lambda_cor * correction_raw * scale
                else:
                    correction = 0.0
        else:
            correction = 0.0
        
        # Final velocity and step - maintains full gradient chain
        v_corrected = v_tangent + correction
        z_vm = z_vm + v_corrected[:, :num_buses] * dt
        z_va = z_va + v_corrected[:, num_buses:] * dt
    
    return z_vm, z_va




def validation_flow_forward(
    flow_model_vm, flow_model_va, test_x, z_vm, z_va,
    num_steps, use_projection, use_drift_correction,
    P_tan_t, projector, num_buses, lambda_cor=5.0, include_load_balance=False
):
    """
    Flow forward for validation, optionally with projection.
    
    This function mirrors the training flow forward but operates in no_grad mode.
    Using the same projection method during validation ensures consistent evaluation.
    
    Args:
        flow_model_vm, flow_model_va: Flow models
        test_x: Input condition
        z_vm, z_va: VAE anchors
        num_steps: Integration steps
        use_projection: Use tangent-space projection
        use_drift_correction: Use drift-correction
        P_tan_t: Precomputed projection matrix (tensor)
        projector: ConstraintProjectionV2 instance
        num_buses: Number of buses
        lambda_cor: Correction strength
        include_load_balance: Include load balance in projection
        
    Returns:
        y_pred_vm, y_pred_va: Final predictions
    """
    if use_drift_correction and projector is not None:
        # Use drift-correction: tangent projection + normal correction
        y_pred_vm, y_pred_va = differentiable_flow_forward_drift_correction(
            flow_model_vm, flow_model_va, test_x, z_vm, z_va,
            projector, num_buses, num_steps=num_steps, lambda_cor=lambda_cor,
            include_load_balance=include_load_balance
        )
    elif use_projection and P_tan_t is not None:
        # Use tangent-space projection only
        y_pred_vm, y_pred_va = differentiable_flow_forward_projected(
            flow_model_vm, flow_model_va, test_x, z_vm, z_va,
            P_tan_t, num_buses, num_steps=num_steps
        )
    else:
        # Standard unprojected flow integration
        y_pred_vm = differentiable_flow_forward(
            flow_model_vm, test_x, z_vm, num_steps=num_steps
        )
        y_pred_va = differentiable_flow_forward(
            flow_model_va, test_x, z_va, num_steps=num_steps
        )
    
    return y_pred_vm, y_pred_va


def train_pareto_flow(config, flow_model_vm, flow_model_va, 
                      vae_model_vm, vae_model_va,
                      loss_fn, training_loader, sys_data, device,
                      lambda_cost=0.9, lambda_carbon=0.1,
                      epochs=500, lr=1e-4, inf_steps=10,
                      use_projection=False, use_drift_correction=False,
                      lambda_cor=5.0, zero_init=True,
                      include_load_balance=False,
                      use_adaptive_weights=False,
                      test_loader=None, early_stopping_patience=20,
                      val_freq=10,
                      tb_logger=None):
    """
    Train flow models to map from VAE anchor [1,0] to target preference.
    
    The training computes loss on the final output of the flow model.
    Uses differentiable flow integration to maintain gradient flow.
    
    Args:
        config: Configuration object
        flow_model_vm: Flow model for Vm
        flow_model_va: Flow model for Va
        vae_model_vm: Pretrained VAE for Vm (anchor generator)
        vae_model_va: Pretrained VAE for Va (anchor generator)
        loss_fn: MultiObjectiveOPFLoss instance
        training_loader: Training data loader
        sys_data: System data
        device: Device
        lambda_cost: Cost weight (default: 0.9)
        lambda_carbon: Carbon weight (default: 0.1)
        epochs: Number of training epochs
        lr: Learning rate
        inf_steps: Number of inference steps for flow integration
        use_projection: Whether to use constraint tangent-space projection
        use_drift_correction: Whether to use drift-correction (tangent + normal correction)
        lambda_cor: Correction strength for drift-correction (default: 5.0)
        zero_init: Whether to initialize flow model output near zero (default: True)
        test_loader: Test data loader for early stopping (optional)
        early_stopping_patience: Patience for early stopping (default: 20)
        val_freq: Validation frequency in epochs (default: 10)
        
    Returns:
        flow_model_vm, flow_model_va: Trained models
        loss_history: Training loss history
    """
    print("=" * 60)
    print(f"Training Pareto Flow: [1,0] → [{lambda_cost}, {lambda_carbon}]")
    print("=" * 60)
    print(f"Epochs: {epochs}, Learning rate: {lr}, Inference steps: {inf_steps}")
    
    # Early stopping configuration
    if test_loader is not None:
        print(f"Early stopping: ENABLED (patience={early_stopping_patience}, val_freq={val_freq})")
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_state_vm = None
        best_state_va = None
    else:
        print("Early stopping: DISABLED (no test loader provided)")
    
    if use_drift_correction:
        print(f"Constraint mode: DRIFT-CORRECTION (λ_cor={lambda_cor})")
    elif use_projection:
        print(f"Constraint mode: TANGENT PROJECTION")
    else:
        print(f"Constraint mode: DISABLED")
    
    print(f"Zero initialization: {'ENABLED' if zero_init else 'DISABLED'}")
    
    # Optimizers
    optimizer_vm = torch.optim.Adam(flow_model_vm.parameters(), lr=lr, weight_decay=1e-6)
    optimizer_va = torch.optim.Adam(flow_model_va.parameters(), lr=lr, weight_decay=1e-6)
    
    # Learning rate scheduler
    scheduler_vm = torch.optim.lr_scheduler.StepLR(optimizer_vm, step_size=200, gamma=0.5)
    scheduler_va = torch.optim.lr_scheduler.StepLR(optimizer_va, step_size=200, gamma=0.5)
    
    # Loss history
    loss_history = {
        'total': [],
        'objective': [],
        'cost': [],
        'carbon': [],
        'constraints': [],
        'gen_vio': [],
        'load_dev': [],
        'k_d': [],  # Track adaptive k_d weight
    }
    
    # Prepare load data (Pd, Qd) for all training samples
    baseMVA = sys_data.baseMVA.item() if hasattr(sys_data.baseMVA, 'item') else float(sys_data.baseMVA)
    Pd_full = torch.from_numpy(sys_data.RPd[:config.Ntrain] / baseMVA).float().to(device)
    Qd_full = torch.from_numpy(sys_data.RQd[:config.Ntrain] / baseMVA).float().to(device)
    
    # Setup constraint projection / drift correction if enabled
    P_tan_t = None
    projector = None
    
    if use_drift_correction or use_projection:
        print("Setting up constraint projection...")
        projector = ConstraintProjectionV2(sys_data, config, use_historical_jacobian=True)
        P_tan, _, _ = projector.compute_projection_matrix(
            include_slack=False, 
            include_load_balance=include_load_balance
        )
        P_tan_t = torch.tensor(P_tan, dtype=torch.float32, device=device)
        print(f"  Projection matrix shape: {P_tan_t.shape}")
        print(f"  Expected: ({2 * config.Nbus - 1}, {2 * config.Nbus - 1})")
        print(f"  Include load balance: {include_load_balance}")
        
        if use_drift_correction:
            print(f"  Drift-Correction λ_cor: {lambda_cor}")
    
    # Initialize flow models near zero if requested
    if zero_init:
        print("\nInitializing flow models near zero...")
        initialize_flow_model_near_zero(flow_model_vm, scale=0.01)
        initialize_flow_model_near_zero(flow_model_va, scale=0.01)
    
    # ==================== VAE Baseline Evaluation (Before Training) ====================
    # Evaluate VAE initial solution quality across entire training set
    print("\n" + "=" * 60)
    print("Evaluating VAE Baseline Quality (Before Training)")
    print("=" * 60)
    
    vae_model_vm.eval()
    vae_model_va.eval()
    
    vae_total_cost = 0.0
    vae_total_carbon = 0.0
    vae_total_gen_vio = 0.0
    vae_total_load_dev = 0.0
    vae_total_branch_pf_vio = 0.0
    vae_total_branch_ang_vio = 0.0
    vae_total_load_satisfy_pct = 0.0  # Load satisfaction rate
    vae_feasible_count = 0
    vae_n_samples = 0
    vae_n_batches = 0
    
    with torch.no_grad():
        for step, (train_x, _) in enumerate(training_loader):
            train_x = train_x.to(device)
            batch_size = train_x.shape[0]
            
            start_idx = step * config.batch_size_training
            end_idx = min(start_idx + batch_size, len(Pd_full))
            if end_idx - start_idx != batch_size:
                continue
            Pd_batch = Pd_full[start_idx:end_idx]
            Qd_batch = Qd_full[start_idx:end_idx]
            
            # Get VAE output
            z_vm = vae_model_vm(train_x, use_mean=True)
            z_va = vae_model_va(train_x, use_mean=True)
            
            # Evaluate VAE solution quality
            _, vae_loss_dict = loss_fn(
                z_vm, z_va, Pd_batch, Qd_batch,
                lambda_cost=lambda_cost, lambda_carbon=lambda_carbon,
                return_details=True
            )
            
            vae_total_cost += vae_loss_dict['cost'] * batch_size
            vae_total_carbon += vae_loss_dict['carbon'] * batch_size
            vae_total_gen_vio += vae_loss_dict['gen_vio'] * batch_size
            vae_total_load_dev += vae_loss_dict['load_dev'] * batch_size
            vae_total_branch_pf_vio += vae_loss_dict['branch_pf_vio'] * batch_size
            vae_total_branch_ang_vio += vae_loss_dict['branch_ang_vio'] * batch_size
            vae_total_load_satisfy_pct += vae_loss_dict['load_satisfy_pct']  # Accumulate for averaging
            
            # Check feasibility for each sample
            if vae_loss_dict['load_dev'] < 0.01 and vae_loss_dict['gen_vio'] < 0.001:
                vae_feasible_count += batch_size
            
            vae_n_samples += batch_size
            vae_n_batches += 1
    
    # Compute averages
    vae_baseline = {
        'cost': vae_total_cost / vae_n_samples,
        'carbon': vae_total_carbon / vae_n_samples,
        'gen_vio': vae_total_gen_vio / vae_n_samples,
        'load_dev': vae_total_load_dev / vae_n_samples,
        'branch_pf_vio': vae_total_branch_pf_vio / vae_n_samples,
        'branch_ang_vio': vae_total_branch_ang_vio / vae_n_samples,
        'load_satisfy_pct': vae_total_load_satisfy_pct / vae_n_batches,  # Average load satisfaction rate (%)
        'feasible_ratio': vae_feasible_count / vae_n_samples,
    }
    # Hard constraint violation (gen + branch, excluding load_dev)
    vae_baseline['hard_constraint_vio'] = (
        vae_baseline['gen_vio'] + 
        vae_baseline['branch_pf_vio'] + 
        vae_baseline['branch_ang_vio']
    )
    
    print(f"\n[VAE Baseline] Evaluated {vae_n_samples} samples:")
    print(f"  Cost:       {vae_baseline['cost']:.2f} $/h")
    print(f"  Carbon:     {vae_baseline['carbon']:.4f} tCO2/h")
    print(f"  Hard Constraint Vio: {vae_baseline['hard_constraint_vio']:.6f} (target: 0)")
    print(f"    - Gen Vio:    {vae_baseline['gen_vio']:.6f}")
    print(f"    - Branch PF:  {vae_baseline['branch_pf_vio']:.6f}")
    print(f"    - Branch Ang: {vae_baseline['branch_ang_vio']:.6f}")
    print(f"  Load Satisfy: {vae_baseline['load_satisfy_pct']:.2f}% (target: >= 99%)")
    print(f"  Feasible:   {vae_baseline['feasible_ratio']*100:.1f}%")
    
    # Log VAE baseline to TensorBoard (at step 0, will be shown as constant line)
    if tb_logger is not None:
        tb_logger.log_scalar('cost/vae', vae_baseline['cost'], 0)
        tb_logger.log_scalar('carbon/vae', vae_baseline['carbon'], 0)
        tb_logger.log_scalar('load_satisfy_pct/vae', vae_baseline['load_satisfy_pct'], 0)
        tb_logger.log_scalar('hard_constraint/vae', vae_baseline['hard_constraint_vio'], 0)
    
    print("=" * 60)
    
    # ==================== Constraint Violation Sanity Check ====================
    # This check verifies that constraint calculations are correct by comparing:
    # 1. VAE direct output (should have small violations if VAE is well-trained)
    # 2. VAE + noise (should have larger violations)
    # 3. Flow initial output (should be close to VAE due to zero initialization)
    print("\n" + "=" * 60)
    print("Constraint Violation Sanity Check")
    print("=" * 60)
    
    # Use one batch for quick sanity check
    with torch.no_grad():
        test_batch_x, _ = next(iter(training_loader))
        test_batch_x = test_batch_x.to(device)
        batch_size = test_batch_x.shape[0]
        
        Pd_batch = Pd_full[:batch_size]
        Qd_batch = Qd_full[:batch_size]
        
        # 1. VAE direct output
        z_vm_vae = vae_model_vm(test_batch_x, use_mean=True)
        z_va_vae = vae_model_va(test_batch_x, use_mean=True)
        
        _, vae_dict = loss_fn(z_vm_vae, z_va_vae, Pd_batch, Qd_batch,
                              lambda_cost=lambda_cost, lambda_carbon=lambda_carbon,
                              return_details=True)
        
        vae_constraint = (vae_dict['gen_vio'] + vae_dict['branch_pf_vio'] + 
                          vae_dict['branch_ang_vio'] + vae_dict['load_dev'])
        
        # 2. VAE + small noise (std=0.01)
        noise_small_vm = torch.randn_like(z_vm_vae) * 0.01
        noise_small_va = torch.randn_like(z_va_vae) * 0.01
        z_vm_noisy_small = z_vm_vae + noise_small_vm
        z_va_noisy_small = z_va_vae + noise_small_va
        
        _, noisy_small_dict = loss_fn(z_vm_noisy_small, z_va_noisy_small, 
                                       Pd_batch, Qd_batch,
                                       lambda_cost=lambda_cost, lambda_carbon=lambda_carbon,
                                       return_details=True)
        
        noisy_small_constraint = (noisy_small_dict['gen_vio'] + noisy_small_dict['branch_pf_vio'] + 
                                  noisy_small_dict['branch_ang_vio'] + noisy_small_dict['load_dev'])
        
        # 3. VAE + large noise (std=0.1)
        noise_large_vm = torch.randn_like(z_vm_vae) * 0.1
        noise_large_va = torch.randn_like(z_va_vae) * 0.1
        z_vm_noisy_large = z_vm_vae + noise_large_vm
        z_va_noisy_large = z_va_vae + noise_large_va
        
        _, noisy_large_dict = loss_fn(z_vm_noisy_large, z_va_noisy_large, 
                                       Pd_batch, Qd_batch,
                                       lambda_cost=lambda_cost, lambda_carbon=lambda_carbon,
                                       return_details=True)
        
        noisy_large_constraint = (noisy_large_dict['gen_vio'] + noisy_large_dict['branch_pf_vio'] + 
                                  noisy_large_dict['branch_ang_vio'] + noisy_large_dict['load_dev'])
        
        # 4. Flow initial output (with zero initialization, flow should output ≈ VAE anchor)
        y_pred_vm = differentiable_flow_forward(flow_model_vm, test_batch_x, z_vm_vae, num_steps=inf_steps)
        y_pred_va = differentiable_flow_forward(flow_model_va, test_batch_x, z_va_vae, num_steps=inf_steps)
        
        _, flow_init_dict = loss_fn(y_pred_vm, y_pred_va, Pd_batch, Qd_batch,
                                     lambda_cost=lambda_cost, lambda_carbon=lambda_carbon,
                                     return_details=True)
        
        flow_init_constraint = (flow_init_dict['gen_vio'] + flow_init_dict['branch_pf_vio'] + 
                                flow_init_dict['branch_ang_vio'] + flow_init_dict['load_dev'])
    
    print(f"\n[Sanity Check] Constraint Total Comparison:")
    print(f"  1. VAE direct:         {vae_constraint:.6f}")
    print(f"  2. VAE + noise(0.01):  {noisy_small_constraint:.6f} (delta={noisy_small_constraint-vae_constraint:+.6f})")
    print(f"  3. VAE + noise(0.1):   {noisy_large_constraint:.6f} (delta={noisy_large_constraint-vae_constraint:+.6f})")
    print(f"  4. Flow initial:       {flow_init_constraint:.6f} (delta={flow_init_constraint-vae_constraint:+.6f})")
    
    print(f"\n[Sanity Check] Individual Components:")
    print(f"  {'Component':<15} | {'VAE':>10} | {'Noise(0.01)':>12} | {'Noise(0.1)':>12} | {'Flow Init':>12}")
    print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    print(f"  {'gen_vio':<15} | {vae_dict['gen_vio']:>10.4f} | {noisy_small_dict['gen_vio']:>12.4f} | {noisy_large_dict['gen_vio']:>12.4f} | {flow_init_dict['gen_vio']:>12.4f}")
    print(f"  {'branch_pf_vio':<15} | {vae_dict['branch_pf_vio']:>10.4f} | {noisy_small_dict['branch_pf_vio']:>12.4f} | {noisy_large_dict['branch_pf_vio']:>12.4f} | {flow_init_dict['branch_pf_vio']:>12.4f}")
    print(f"  {'branch_ang_vio':<15} | {vae_dict['branch_ang_vio']:>10.4f} | {noisy_small_dict['branch_ang_vio']:>12.4f} | {noisy_large_dict['branch_ang_vio']:>12.4f} | {flow_init_dict['branch_ang_vio']:>12.4f}")
    print(f"  {'load_dev':<15} | {vae_dict['load_dev']:>10.4f} | {noisy_small_dict['load_dev']:>12.4f} | {noisy_large_dict['load_dev']:>12.4f} | {flow_init_dict['load_dev']:>12.4f}")
    
    # Sanity check warnings
    if noisy_large_constraint < vae_constraint * 1.5:
        print(f"\n  [WARNING] Adding noise(0.1) did NOT significantly increase constraints!")
        print(f"            This may indicate a problem with constraint calculation.")
    
    if flow_init_constraint < vae_constraint * 0.5:
        print(f"\n  [WARNING] Flow initial output has much LOWER constraints than VAE!")
        print(f"            This is unexpected for zero-initialized flow model.")
    
    print("=" * 60)
    
    start_time = time.process_time()
    
    for epoch in range(epochs):
        running_loss = 0.0
        running_obj = 0.0
        running_cost = 0.0
        running_carbon = 0.0
        running_constraints = 0.0
        running_gen_vio = 0.0
        running_branch_pf_vio = 0.0
        running_branch_ang_vio = 0.0
        running_load_dev = 0.0
        running_load_satisfy_pct = 0.0
        n_batches = 0
        
        flow_model_vm.train()
        flow_model_va.train()
        
        for step, (train_x, _) in enumerate(training_loader):
            train_x = train_x.to(device)
            batch_size = train_x.shape[0]
            
            # Get Pd, Qd for this batch
            start_idx = step * config.batch_size_training
            end_idx = min(start_idx + batch_size, len(Pd_full))
            if end_idx - start_idx != batch_size:
                # Handle last incomplete batch
                continue
            Pd_batch = Pd_full[start_idx:end_idx]
            Qd_batch = Qd_full[start_idx:end_idx]
            
            # Zero gradients
            optimizer_vm.zero_grad()
            optimizer_va.zero_grad()
            
            # ==================== Step 1: Get VAE anchor ====================
            with torch.no_grad():
                z_vm = vae_model_vm(train_x, use_mean=True)  # [batch, n_bus]
                z_va = vae_model_va(train_x, use_mean=True)  # [batch, n_bus-1]
            
            # Debug: Evaluate VAE anchor quality (first batch of first epoch)
            if epoch == 0 and step == 0:
                with torch.no_grad():
                    _, vae_loss_dict = loss_fn(
                        z_vm, z_va, Pd_batch, Qd_batch,
                        lambda_cost=lambda_cost, lambda_carbon=lambda_carbon,
                        return_details=True
                    )
                    print(f"\n[DEBUG] VAE Anchor Quality (before flow):")
                    print(f"  Cost: {vae_loss_dict['cost']:.2f} $/h")
                    print(f"  Carbon: {vae_loss_dict['carbon']:.2f} tCO2/h")
                    print(f"  Constraints (total): {vae_loss_dict['constraints']:.2f}")
                    print(f"    - Gen Violation (raw): {vae_loss_dict['gen_vio']:.6f}")
                    print(f"    - Branch PF Vio (raw): {vae_loss_dict['branch_pf_vio']:.6f}")
                    print(f"    - Branch Ang Vio (raw): {vae_loss_dict['branch_ang_vio']:.6f}")
                    print(f"    - Load Deviation (raw): {vae_loss_dict['load_dev']:.6f}")
                    weights = vae_loss_dict['weights']
                    print(f"  Weights: k_g={weights['k_g']}, k_Sl={weights['k_Sl']}, k_theta={weights['k_theta']}, k_d={weights['k_d']}")
                    print()
            
            # ==================== Step 2: Run differentiable flow integration ====================
            # Use our custom differentiable flow forward (maintains gradient flow)
            
            if use_drift_correction and projector is not None:
                # Use drift-correction: tangent projection + normal correction
                y_pred_vm, y_pred_va = differentiable_flow_forward_drift_correction(
                    flow_model_vm, flow_model_va, train_x, z_vm, z_va,
                    projector, config.Nbus, num_steps=inf_steps, lambda_cor=lambda_cor,
                    include_load_balance=include_load_balance
                )
            elif use_projection and P_tan_t is not None:
                # Use tangent-space projection only
                y_pred_vm, y_pred_va = differentiable_flow_forward_projected(
                    flow_model_vm, flow_model_va, train_x, z_vm, z_va,
                    P_tan_t, config.Nbus, num_steps=inf_steps
                )
            else:
                # Standard unprojected flow integration
                # For Vm - differentiable integration from VAE anchor
                y_pred_vm = differentiable_flow_forward(
                    flow_model_vm, train_x, z_vm, num_steps=inf_steps
                )
                
                # For Va - differentiable integration from VAE anchor
                y_pred_va = differentiable_flow_forward(
                    flow_model_va, train_x, z_va, num_steps=inf_steps
                )
            
            # ==================== Step 3: Compute multi-objective loss ====================
            loss, loss_dict = loss_fn(
                y_pred_vm, y_pred_va, Pd_batch, Qd_batch,
                lambda_cost=lambda_cost, lambda_carbon=lambda_carbon,
                update_weights=use_adaptive_weights,  # Enable dynamic weight update
                return_details=True
            )
            
            # ==================== Step 4: Backward and optimize ====================
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(flow_model_vm.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(flow_model_va.parameters(), max_norm=1.0)
            
            optimizer_vm.step()
            optimizer_va.step()
            
            # Accumulate losses
            running_loss += loss_dict['total']
            running_obj += loss_dict['objective']
            running_cost += loss_dict['cost']
            running_carbon += loss_dict['carbon']
            running_constraints += loss_dict['constraints']
            running_gen_vio += loss_dict['gen_vio']
            running_branch_pf_vio += loss_dict['branch_pf_vio']
            running_branch_ang_vio += loss_dict['branch_ang_vio']
            running_load_dev += loss_dict['load_dev']
            running_load_satisfy_pct += loss_dict['load_satisfy_pct']
            n_batches += 1
            
            # Clear intermediate tensors
            del loss, loss_dict, y_pred_vm, y_pred_va
        
        # Step schedulers
        scheduler_vm.step()
        scheduler_va.step()
        
        # GPU memory cleanup every epoch
        gpu_memory_cleanup()
        
        # GPU thermal protection every 10 epochs
        if (epoch + 1) % 10 == 0:
            check_gpu_temperature()
        
        # Average losses for this epoch
        if n_batches > 0:
            avg_loss = running_loss / n_batches
            avg_obj = running_obj / n_batches
            avg_cost = running_cost / n_batches
            avg_carbon = running_carbon / n_batches
            avg_constraints = running_constraints / n_batches
            avg_gen_vio = running_gen_vio / n_batches
            avg_branch_pf_vio = running_branch_pf_vio / n_batches
            avg_branch_ang_vio = running_branch_ang_vio / n_batches
            avg_load_dev = running_load_dev / n_batches
            avg_load_satisfy_pct = running_load_satisfy_pct / n_batches
            
            # Hard constraint violation (gen + branch, excluding load_dev)
            avg_hard_constraint_vio = avg_gen_vio + avg_branch_pf_vio + avg_branch_ang_vio
            
            # Store in history
            loss_history['total'].append(avg_loss)
            loss_history['objective'].append(avg_obj)
            loss_history['cost'].append(avg_cost)
            loss_history['carbon'].append(avg_carbon)
            loss_history['constraints'].append(avg_constraints)
            loss_history['gen_vio'].append(avg_gen_vio)
            loss_history['load_dev'].append(avg_load_dev)
            
            # Track adaptive weight k_d
            if use_adaptive_weights and hasattr(loss_fn, 'weight_scheduler'):
                loss_history['k_d'].append(loss_fn.weight_scheduler.k_d)
            else:
                loss_history['k_d'].append(loss_fn.k_d)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, "
                  f"Obj={avg_obj:.2f} (Cost={avg_cost:.2f}, Carbon={avg_carbon:.4f}), "
                  f"HardConstr={avg_hard_constraint_vio:.6f}, LoadSat={avg_load_satisfy_pct:.2f}%")
            # Print adaptive weights if enabled
            if use_adaptive_weights and hasattr(loss_fn, 'weight_scheduler'):
                weights = loss_fn.weight_scheduler.get_weights()
                print(f"  Adaptive weights: k_g={weights['k_g']:.1f}, k_Sl={weights['k_Sl']:.1f}, "
                      f"k_d={weights['k_d']:.1f} (load dev: {avg_load_dev:.4f})")
        
        # ==================== TensorBoard Logging (Simplified) ====================
        # Key metrics: hard constraints (should be 0) and load satisfaction (should be >= 99%)
        if tb_logger is not None and n_batches > 0:
            # VAE baseline (constant reference lines)
            tb_logger.log_scalar('cost/vae', vae_baseline['cost'], epoch)
            tb_logger.log_scalar('carbon/vae', vae_baseline['carbon'], epoch)
            tb_logger.log_scalar('load_satisfy_pct/vae', vae_baseline['load_satisfy_pct'], epoch)
            tb_logger.log_scalar('hard_constraint/vae', vae_baseline['hard_constraint_vio'], epoch)
            
            # Flow model output (training set)
            tb_logger.log_scalar('cost/flow_train', avg_cost, epoch)
            tb_logger.log_scalar('carbon/flow_train', avg_carbon, epoch)
            tb_logger.log_scalar('load_satisfy_pct/flow_train', avg_load_satisfy_pct, epoch)
            tb_logger.log_scalar('hard_constraint/flow_train', avg_hard_constraint_vio, epoch)
        
        # ==================== Validation and Early Stopping ====================
        if test_loader is not None and (epoch + 1) % val_freq == 0:
            flow_model_vm.eval()
            flow_model_va.eval()
            
            val_loss_total = 0.0
            val_load_dev = 0.0
            val_load_satisfy_pct = 0.0
            val_cost = 0.0
            val_n_batches = 0
            
            # Prepare test data
            Pd_test = torch.from_numpy(sys_data.RPd[config.Ntrain:] / baseMVA).float().to(device)
            Qd_test = torch.from_numpy(sys_data.RQd[config.Ntrain:] / baseMVA).float().to(device)
            
            # Also track VAE baseline on validation set
            vae_val_cost = 0.0
            vae_val_carbon = 0.0
            vae_val_load_satisfy_pct = 0.0
            val_carbon = 0.0
            val_gen_vio = 0.0
            val_branch_pf_vio = 0.0
            val_branch_ang_vio = 0.0
            
            with torch.no_grad():
                for step, (test_x, _) in enumerate(test_loader):
                    test_x = test_x.to(device)
                    batch_size = test_x.shape[0]
                    
                    start_idx = step * config.batch_size_test
                    end_idx = min(start_idx + batch_size, len(Pd_test))
                    if end_idx - start_idx != batch_size:
                        continue
                    
                    Pd_batch = Pd_test[start_idx:end_idx]
                    Qd_batch = Qd_test[start_idx:end_idx]
                    
                    # Get VAE anchor
                    z_vm = vae_model_vm(test_x, use_mean=True)
                    z_va = vae_model_va(test_x, use_mean=True)
                    
                    # Evaluate VAE quality on validation set
                    _, vae_val_dict = loss_fn(
                        z_vm, z_va, Pd_batch, Qd_batch,
                        lambda_cost=lambda_cost, lambda_carbon=lambda_carbon,
                        return_details=True
                    )
                    vae_val_cost += vae_val_dict['cost']
                    vae_val_carbon += vae_val_dict['carbon']
                    vae_val_load_satisfy_pct += vae_val_dict['load_satisfy_pct']
                    
                    # Flow forward with same projection method as training
                    y_pred_vm, y_pred_va = validation_flow_forward(
                        flow_model_vm, flow_model_va, test_x, z_vm, z_va,
                        num_steps=inf_steps,
                        use_projection=use_projection,
                        use_drift_correction=use_drift_correction,
                        P_tan_t=P_tan_t,
                        projector=projector,
                        num_buses=config.Nbus,
                        lambda_cor=lambda_cor,
                        include_load_balance=include_load_balance
                    )
                    
                    # Compute loss
                    val_loss, val_dict = loss_fn(
                        y_pred_vm, y_pred_va, Pd_batch, Qd_batch,
                        lambda_cost=lambda_cost, lambda_carbon=lambda_carbon,
                        update_weights=False, return_details=True
                    )
                    
                    val_loss_total += val_loss.item()
                    val_load_dev += val_dict['load_dev']
                    val_load_satisfy_pct += val_dict['load_satisfy_pct']
                    val_cost += val_dict['cost']
                    val_carbon += val_dict['carbon']
                    val_gen_vio += val_dict['gen_vio']
                    val_branch_pf_vio += val_dict['branch_pf_vio']
                    val_branch_ang_vio += val_dict['branch_ang_vio']
                    val_n_batches += 1
            
            if val_n_batches > 0:
                avg_val_loss = val_loss_total / val_n_batches
                avg_val_load_dev = val_load_dev / val_n_batches
                avg_val_load_satisfy_pct = val_load_satisfy_pct / val_n_batches
                avg_val_cost = val_cost / val_n_batches
                avg_val_carbon = val_carbon / val_n_batches
                avg_val_gen_vio = val_gen_vio / val_n_batches
                avg_val_branch_pf_vio = val_branch_pf_vio / val_n_batches
                avg_val_branch_ang_vio = val_branch_ang_vio / val_n_batches
                
                # Hard constraint violation (validation set) - excluding load_dev
                avg_val_hard_constraint_vio = avg_val_gen_vio + avg_val_branch_pf_vio + avg_val_branch_ang_vio
                
                # VAE baseline on validation set
                avg_vae_val_cost = vae_val_cost / val_n_batches
                avg_vae_val_carbon = vae_val_carbon / val_n_batches
                avg_vae_val_load_satisfy_pct = vae_val_load_satisfy_pct / val_n_batches
                
                # Track validation history
                loss_history.setdefault('val_loss', []).append(avg_val_loss)
                loss_history.setdefault('val_load_dev', []).append(avg_val_load_dev)
                
                print(f"  [Val] Cost={avg_val_cost:.2f}, Carbon={avg_val_carbon:.4f}, HardConstr={avg_val_hard_constraint_vio:.6f}, LoadSat={avg_val_load_satisfy_pct:.2f}%")
                print(f"  [VAE] Cost={avg_vae_val_cost:.2f}, Carbon={avg_vae_val_carbon:.4f}, LoadSat={avg_vae_val_load_satisfy_pct:.2f}%")
                
                # TensorBoard: Log validation metrics
                if tb_logger is not None:
                    # Flow model output (validation set)
                    tb_logger.log_scalar('cost/flow_val', avg_val_cost, epoch)
                    tb_logger.log_scalar('carbon/flow_val', avg_val_carbon, epoch)
                    tb_logger.log_scalar('load_satisfy_pct/flow_val', avg_val_load_satisfy_pct, epoch)
                    tb_logger.log_scalar('hard_constraint/flow_val', avg_val_hard_constraint_vio, epoch)
                    
                    # VAE on validation set
                    tb_logger.log_scalar('cost/vae_val', avg_vae_val_cost, epoch)
                    tb_logger.log_scalar('carbon/vae_val', avg_vae_val_carbon, epoch)
                    tb_logger.log_scalar('load_satisfy_pct/vae_val', avg_vae_val_load_satisfy_pct, epoch)
                
                # Early stopping check (use load deviation as primary metric for generalization)
                # Lower load deviation = better generalization
                val_metric = avg_val_load_dev  # Use load deviation for early stopping
                
                if val_metric < best_val_loss:
                    best_val_loss = val_metric
                    best_epoch = epoch + 1
                    patience_counter = 0
                    # Free old best_state memory before creating new one
                    if best_state_vm is not None:
                        del best_state_vm
                        del best_state_va
                    # Save best model state (use detach to avoid gradient tracking)
                    best_state_vm = {k: v.detach().cpu().clone() for k, v in flow_model_vm.state_dict().items()}
                    best_state_va = {k: v.detach().cpu().clone() for k, v in flow_model_va.state_dict().items()}
                    print(f"  [Val] New best! Load Dev={val_metric:.4f}")
                else:
                    patience_counter += 1
                    print(f"  [Val] No improvement. Patience: {patience_counter}/{early_stopping_patience}")
                    
                    if patience_counter >= early_stopping_patience:
                        print(f"\n[Early Stopping] No improvement for {early_stopping_patience} validations.")
                        print(f"  Best epoch: {best_epoch}, Best Load Dev: {best_val_loss:.4f}")
                        
                        # Restore best model
                        if best_state_vm is not None:
                            flow_model_vm.load_state_dict({k: v.to(device) for k, v in best_state_vm.items()})
                            flow_model_va.load_state_dict({k: v.to(device) for k, v in best_state_va.items()})
                            print("  Best model restored.")
                        break
            
            flow_model_vm.train()
            flow_model_va.train()
        
        # Save checkpoints periodically
        if (epoch + 1) % 100 == 0:
            save_checkpoint(config, flow_model_vm, flow_model_va, 
                           lambda_cost, lambda_carbon, epoch + 1)
    
    time_train = time.process_time() - start_time
    print(f"\nTraining completed in {time_train:.2f} seconds ({time_train/60:.2f} minutes)")
    
    # If early stopping was used and we have a best model, ensure it's loaded
    if test_loader is not None and best_state_vm is not None and patience_counter < early_stopping_patience:
        flow_model_vm.load_state_dict({k: v.to(device) for k, v in best_state_vm.items()})
        flow_model_va.load_state_dict({k: v.to(device) for k, v in best_state_va.items()})
        print(f"  Final: Best model from epoch {best_epoch} restored (Load Dev: {best_val_loss:.4f})")
    
    # Save final models
    save_checkpoint(config, flow_model_vm, flow_model_va, 
                   lambda_cost, lambda_carbon, epochs, final=True)
    
    return flow_model_vm, flow_model_va, loss_history


def save_checkpoint(config, model_vm, model_va, lambda_cost, lambda_carbon, epoch, final=False):
    """Save model checkpoint."""
    suffix = "final" if final else f"E{epoch}"
    pref_str = f"{lambda_cost}_{lambda_carbon}".replace(".", "")
    
    vm_path = os.path.join(config.model_save_dir, 
                          f'modelvm_pareto_{pref_str}_{suffix}.pth')
    va_path = os.path.join(config.model_save_dir, 
                          f'modelva_pareto_{pref_str}_{suffix}.pth')
    
    torch.save(model_vm.state_dict(), vm_path, _use_new_zipfile_serialization=False)
    torch.save(model_va.state_dict(), va_path, _use_new_zipfile_serialization=False)
    
    print(f"  Checkpoint saved: {vm_path}")


def generate_preference_schedule(start_cost=1.0, end_cost=0.5, step=0.05):
    """
    Generate a preference schedule for curriculum learning.
    
    Args:
        start_cost: Starting cost weight (default: 1.0)
        end_cost: Ending cost weight (default: 0.5)
        step: Step size for preference change (default: 0.05)
        
    Returns:
        List of (lambda_cost, lambda_carbon) tuples
    """
    schedule = []
    current_cost = start_cost - step  # First step from VAE
    # Use small epsilon for floating point comparison
    eps = 1e-9
    while current_cost >= end_cost - eps:
        lambda_cost = round(current_cost, 2)
        lambda_carbon = round(1.0 - lambda_cost, 2)
        schedule.append((lambda_cost, lambda_carbon))
        current_cost -= step
    return schedule


def train_curriculum_pareto_flow(
    config, sys_data, dataloaders, device,
    vae_model_vm, vae_model_va,
    gci_values,
    preference_schedule,
    epochs_per_stage=100,
    lr=1e-4, inf_steps=10,
    use_projection=False, use_drift_correction=False,
    lambda_cor=5.0, zero_init=True,
    include_load_balance=False,
    use_adaptive_weights=False,
    early_stopping_patience=10,
    val_freq=5,
    tb_logger=None
):
    """
    Train flow models using curriculum learning with progressive preference shifts.
    
    Each stage uses the previous stage's output as the anchor point, enabling
    small-step transitions that are easier to learn and generalize better.
    
    Args:
        config: Configuration object
        sys_data: System data
        dataloaders: Data loaders
        device: Device
        vae_model_vm, vae_model_va: Pretrained VAE models
        gci_values: Generator carbon intensities
        preference_schedule: List of (lambda_cost, lambda_carbon) tuples
        epochs_per_stage: Epochs per curriculum stage
        lr: Learning rate
        inf_steps: Inference steps
        use_projection: Enable tangent-space projection
        use_drift_correction: Enable drift-correction
        lambda_cor: Correction strength
        zero_init: Zero initialization for first stage
        include_load_balance: Include load balance in projection
        use_adaptive_weights: Enable adaptive weights
        early_stopping_patience: Early stopping patience
        val_freq: Validation frequency
        
    Returns:
        Trained flow models for all stages
    """
    print("=" * 70)
    print("CURRICULUM LEARNING FOR PARETO-ADAPTIVE FLOW")
    print("=" * 70)
    print(f"Preference schedule ({len(preference_schedule)} stages):")
    for i, (lc, le) in enumerate(preference_schedule):
        print(f"  Stage {i+1}: [{lc:.2f}, {le:.2f}]")
    print(f"Epochs per stage: {epochs_per_stage}")
    print("=" * 70)
    
    input_dim = sys_data.x_train.shape[1]
    output_dim_vm = sys_data.yvm_train.shape[1]
    output_dim_va = sys_data.yva_train.shape[1]
    
    training_loader = dataloaders['train_vm']
    test_loader = dataloaders.get('test_vm', None)
    
    # Prepare load data
    baseMVA = sys_data.baseMVA.item() if hasattr(sys_data.baseMVA, 'item') else float(sys_data.baseMVA)
    Pd_full = torch.from_numpy(sys_data.RPd[:config.Ntrain] / baseMVA).float().to(device)
    Qd_full = torch.from_numpy(sys_data.RQd[:config.Ntrain] / baseMVA).float().to(device)
    Pd_test = torch.from_numpy(sys_data.RPd[config.Ntrain:] / baseMVA).float().to(device)
    Qd_test = torch.from_numpy(sys_data.RQd[config.Ntrain:] / baseMVA).float().to(device)
    
    # Setup constraint projection
    P_tan_t = None
    projector = None
    if use_drift_correction or use_projection:
        print("\nSetting up constraint projection...")
        projector = ConstraintProjectionV2(sys_data, config, use_historical_jacobian=True)
        P_tan, _, _ = projector.compute_projection_matrix(
            include_slack=False, 
            include_load_balance=include_load_balance
        )
        P_tan_t = torch.tensor(P_tan, dtype=torch.float32, device=device)
        print(f"  Projection matrix shape: {P_tan_t.shape}")
    
    # Current anchor generator function
    # Starts with VAE, then updates to use previous stage's flow output
    current_anchor_vm = vae_model_vm
    current_anchor_va = vae_model_va
    use_flow_anchor = False  # Initially use VAE directly
    
    # Store all trained models
    all_models = {}
    all_histories = {}
    
    for stage_idx, (lambda_cost, lambda_carbon) in enumerate(preference_schedule):
        print("\n" + "=" * 70)
        print(f"STAGE {stage_idx + 1}/{len(preference_schedule)}: [{lambda_cost:.2f}, {lambda_carbon:.2f}]")
        print("=" * 70)
        
        # Memory cleanup before starting new stage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated() / (1024**3)
            print(f"  GPU memory at stage start: {mem_before:.2f} GB")
        
        # Create fresh flow models for this stage
        flow_model_vm = create_flow_model(config, input_dim, output_dim_vm, 
                                          is_vm=True, device=device)
        flow_model_va = create_flow_model(config, input_dim, output_dim_va, 
                                          is_vm=False, device=device)
        
        # Zero initialization only for first stage (subsequent stages start fresh)
        if zero_init and stage_idx == 0:
            print("Initializing flow models near zero (first stage)...")
            initialize_flow_model_near_zero(flow_model_vm, scale=0.01)
            initialize_flow_model_near_zero(flow_model_va, scale=0.01)
        elif stage_idx > 0:
            # For subsequent stages, also initialize near zero
            # since each stage learns a small delta
            print("Initializing flow models near zero (incremental stage)...")
            initialize_flow_model_near_zero(flow_model_vm, scale=0.01)
            initialize_flow_model_near_zero(flow_model_va, scale=0.01)
        
        # Attach VAE for potential internal use
        flow_model_vm.pretrain_model = vae_model_vm
        flow_model_va.pretrain_model = vae_model_va
        
        # Create loss function for this stage
        loss_fn = MultiObjectiveOPFLoss(sys_data, config, gci_values, 
                                        use_adaptive_weights=use_adaptive_weights)
        loss_fn = loss_fn.to(device)
        
        # Optimizers
        optimizer_vm = torch.optim.Adam(flow_model_vm.parameters(), lr=lr, weight_decay=1e-6)
        optimizer_va = torch.optim.Adam(flow_model_va.parameters(), lr=lr, weight_decay=1e-6)
        scheduler_vm = torch.optim.lr_scheduler.StepLR(optimizer_vm, step_size=50, gamma=0.7)
        scheduler_va = torch.optim.lr_scheduler.StepLR(optimizer_va, step_size=50, gamma=0.7)
        
        # Loss history for this stage
        loss_history = {
            'total': [], 'objective': [], 'cost': [], 'carbon': [],
            'constraints': [], 'gen_vio': [], 'load_dev': [], 'k_d': [],
            'val_loss': [], 'val_load_dev': []
        }
        
        # Early stopping
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_state_vm = None
        best_state_va = None
        
        start_time = time.process_time()
        
        for epoch in range(epochs_per_stage):
            running_loss = 0.0
            running_cost = 0.0
            running_carbon = 0.0
            running_load_dev = 0.0
            running_load_satisfy_pct = 0.0
            running_objective = 0.0
            running_constraints = 0.0
            running_gen_vio = 0.0
            running_branch_pf_vio = 0.0
            running_branch_ang_vio = 0.0
            n_batches = 0
            
            flow_model_vm.train()
            flow_model_va.train()
            
            for step, (train_x, _) in enumerate(training_loader):
                train_x = train_x.to(device)
                batch_size = train_x.shape[0]
                
                start_idx = step * config.batch_size_training
                end_idx = min(start_idx + batch_size, len(Pd_full))
                if end_idx - start_idx != batch_size:
                    continue
                Pd_batch = Pd_full[start_idx:end_idx]
                Qd_batch = Qd_full[start_idx:end_idx]
                
                optimizer_vm.zero_grad()
                optimizer_va.zero_grad()
                
                # ==================== Get anchor from previous stage ====================
                with torch.no_grad():
                    if use_flow_anchor:
                        # Use previous stage's flow output as anchor
                        # First get VAE output, then apply previous stage's flow
                        z_vae_vm = vae_model_vm(train_x, use_mean=True)
                        z_vae_va = vae_model_va(train_x, use_mean=True)
                        # Apply previous flow
                        z_vm = differentiable_flow_forward(
                            current_anchor_vm, train_x, z_vae_vm, num_steps=inf_steps
                        )
                        z_va = differentiable_flow_forward(
                            current_anchor_va, train_x, z_vae_va, num_steps=inf_steps
                        )
                    else:
                        # Use VAE directly
                        z_vm = vae_model_vm(train_x, use_mean=True)
                        z_va = vae_model_va(train_x, use_mean=True)
                
                # ==================== Flow integration with current stage model ====================
                if use_drift_correction and projector is not None:
                    y_pred_vm, y_pred_va = differentiable_flow_forward_drift_correction(
                        flow_model_vm, flow_model_va, train_x, z_vm, z_va,
                        projector, config.Nbus, num_steps=inf_steps, lambda_cor=lambda_cor,
                        include_load_balance=include_load_balance
                    )
                elif use_projection and P_tan_t is not None:
                    y_pred_vm, y_pred_va = differentiable_flow_forward_projected(
                        flow_model_vm, flow_model_va, train_x, z_vm, z_va,
                        P_tan_t, config.Nbus, num_steps=inf_steps
                    )
                else:
                    y_pred_vm = differentiable_flow_forward(
                        flow_model_vm, train_x, z_vm, num_steps=inf_steps
                    )
                    y_pred_va = differentiable_flow_forward(
                        flow_model_va, train_x, z_va, num_steps=inf_steps
                    )
                
                # ==================== Compute loss ====================
                loss, loss_dict = loss_fn(
                    y_pred_vm, y_pred_va, Pd_batch, Qd_batch,
                    lambda_cost=lambda_cost, lambda_carbon=lambda_carbon,
                    update_weights=use_adaptive_weights,
                    return_details=True
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(flow_model_vm.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(flow_model_va.parameters(), max_norm=1.0)
                optimizer_vm.step()
                optimizer_va.step()
                
                running_loss += loss_dict['total']
                running_cost += loss_dict['cost']
                running_carbon += loss_dict['carbon']
                running_load_dev += loss_dict['load_dev']
                running_load_satisfy_pct += loss_dict['load_satisfy_pct']
                running_objective += loss_dict['objective']
                running_constraints += loss_dict['constraints']
                running_gen_vio += loss_dict['gen_vio']
                running_branch_pf_vio += loss_dict['branch_pf_vio']
                running_branch_ang_vio += loss_dict['branch_ang_vio']
                n_batches += 1
                
                # Clear intermediate tensors
                del loss, loss_dict, y_pred_vm, y_pred_va
            
            scheduler_vm.step()
            scheduler_va.step()
            
            # GPU memory cleanup every epoch
            gpu_memory_cleanup()
            
            # GPU thermal protection every 10 epochs
            if (epoch + 1) % 10 == 0:
                check_gpu_temperature()
            
            # Record epoch metrics
            if n_batches > 0:
                avg_loss = running_loss / n_batches
                avg_cost = running_cost / n_batches
                avg_carbon = running_carbon / n_batches
                avg_load_dev = running_load_dev / n_batches
                avg_load_satisfy_pct = running_load_satisfy_pct / n_batches
                avg_objective = running_objective / n_batches
                avg_constraints = running_constraints / n_batches
                avg_gen_vio = running_gen_vio / n_batches
                avg_branch_pf_vio = running_branch_pf_vio / n_batches
                avg_branch_ang_vio = running_branch_ang_vio / n_batches
                
                # Hard constraint violation (excluding load_dev)
                avg_hard_constraint_vio = avg_gen_vio + avg_branch_pf_vio + avg_branch_ang_vio
                
                loss_history['total'].append(avg_loss)
                loss_history['cost'].append(avg_cost)
                loss_history['carbon'].append(avg_carbon)
                loss_history['load_dev'].append(avg_load_dev)
                loss_history['objective'].append(avg_objective)
                loss_history['constraints'].append(avg_constraints)
                loss_history['gen_vio'].append(avg_gen_vio)
                
                if use_adaptive_weights and hasattr(loss_fn, 'weight_scheduler'):
                    loss_history['k_d'].append(loss_fn.weight_scheduler.k_d)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs_per_stage}: Loss={avg_loss:.4f}, "
                      f"Cost={avg_cost:.2f}, Carbon={avg_carbon:.2f}, HardConstr={avg_hard_constraint_vio:.6f}, LoadSat={avg_load_satisfy_pct:.2f}%")
            
            # ==================== TensorBoard Logging (Simplified) ====================
            if tb_logger is not None and n_batches > 0:
                global_step = sum(epochs_per_stage for _ in range(stage_idx)) + epoch
                
                # Flow model output
                tb_logger.log_scalar('cost/flow_train', avg_cost, global_step)
                tb_logger.log_scalar('carbon/flow_train', avg_carbon, global_step)
                tb_logger.log_scalar('load_satisfy_pct/flow_train', avg_load_satisfy_pct, global_step)
                tb_logger.log_scalar('hard_constraint/flow_train', avg_hard_constraint_vio, global_step)
                
                # Current curriculum stage preference
                tb_logger.log_scalar('curriculum/lambda_cost', lambda_cost, global_step)
                tb_logger.log_scalar('curriculum/lambda_carbon', lambda_carbon, global_step)
            
            # ==================== Validation ====================
            if test_loader is not None and (epoch + 1) % val_freq == 0:
                flow_model_vm.eval()
                flow_model_va.eval()
                
                val_load_dev = 0.0
                val_load_satisfy_pct = 0.0
                val_gen_vio = 0.0
                val_branch_pf_vio = 0.0
                val_branch_ang_vio = 0.0
                val_n_batches = 0
                
                with torch.no_grad():
                    for step, (test_x, _) in enumerate(test_loader):
                        test_x = test_x.to(device)
                        batch_size = test_x.shape[0]
                        
                        start_idx = step * config.batch_size_test
                        end_idx = min(start_idx + batch_size, len(Pd_test))
                        if end_idx - start_idx != batch_size:
                            continue
                        
                        Pd_batch = Pd_test[start_idx:end_idx]
                        Qd_batch = Qd_test[start_idx:end_idx]
                        
                        # Get anchor (same logic as training)
                        if use_flow_anchor:
                            z_vae_vm = vae_model_vm(test_x, use_mean=True)
                            z_vae_va = vae_model_va(test_x, use_mean=True)
                            z_vm = differentiable_flow_forward(
                                current_anchor_vm, test_x, z_vae_vm, num_steps=inf_steps
                            )
                            z_va = differentiable_flow_forward(
                                current_anchor_va, test_x, z_vae_va, num_steps=inf_steps
                            )
                        else:
                            z_vm = vae_model_vm(test_x, use_mean=True)
                            z_va = vae_model_va(test_x, use_mean=True)
                        
                        # Flow forward with same projection method as training
                        y_pred_vm, y_pred_va = validation_flow_forward(
                            flow_model_vm, flow_model_va, test_x, z_vm, z_va,
                            num_steps=inf_steps,
                            use_projection=use_projection,
                            use_drift_correction=use_drift_correction,
                            P_tan_t=P_tan_t,
                            projector=projector,
                            num_buses=config.Nbus,
                            lambda_cor=lambda_cor,
                            include_load_balance=include_load_balance
                        )
                        
                        _, val_dict = loss_fn(
                            y_pred_vm, y_pred_va, Pd_batch, Qd_batch,
                            lambda_cost=lambda_cost, lambda_carbon=lambda_carbon,
                            update_weights=False, return_details=True
                        )
                        
                        val_load_dev += val_dict['load_dev']
                        val_load_satisfy_pct += val_dict['load_satisfy_pct']
                        val_gen_vio += val_dict['gen_vio']
                        val_branch_pf_vio += val_dict['branch_pf_vio']
                        val_branch_ang_vio += val_dict['branch_ang_vio']
                        val_n_batches += 1
                
                if val_n_batches > 0:
                    avg_val_load_dev = val_load_dev / val_n_batches
                    avg_val_load_satisfy_pct = val_load_satisfy_pct / val_n_batches
                    avg_val_hard_constraint_vio = (val_gen_vio + val_branch_pf_vio + val_branch_ang_vio) / val_n_batches
                    loss_history['val_load_dev'].append(avg_val_load_dev)
                    
                    print(f"    [Val] HardConstr={avg_val_hard_constraint_vio:.6f}, LoadSat={avg_val_load_satisfy_pct:.2f}%")
                    
                    # TensorBoard: Log validation metrics
                    if tb_logger is not None:
                        global_step = sum(epochs_per_stage for _ in range(stage_idx)) + epoch
                        tb_logger.log_scalar('load_satisfy_pct/flow_val', avg_val_load_satisfy_pct, global_step)
                        tb_logger.log_scalar('hard_constraint/flow_val', avg_val_hard_constraint_vio, global_step)
                    
                    if avg_val_load_dev < best_val_loss:
                        best_val_loss = avg_val_load_dev
                        best_epoch = epoch + 1
                        patience_counter = 0
                        # Free old best_state memory before creating new one
                        if best_state_vm is not None:
                            del best_state_vm
                            del best_state_va
                        best_state_vm = {k: v.detach().cpu().clone() for k, v in flow_model_vm.state_dict().items()}
                        best_state_va = {k: v.detach().cpu().clone() for k, v in flow_model_va.state_dict().items()}
                        print(f"    [Val] New best! Load Dev={avg_val_load_dev:.4f}")
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            print(f"    [Early Stop] at epoch {epoch+1}, best was epoch {best_epoch}")
                            break
                
                flow_model_vm.train()
                flow_model_va.train()
        
        stage_time = time.process_time() - start_time
        print(f"\nStage {stage_idx+1} completed in {stage_time:.1f}s")
        
        # Restore best model
        if best_state_vm is not None:
            flow_model_vm.load_state_dict({k: v.to(device) for k, v in best_state_vm.items()})
            flow_model_va.load_state_dict({k: v.to(device) for k, v in best_state_va.items()})
            print(f"  Restored best model from epoch {best_epoch} (Load Dev={best_val_loss:.4f})")
        
        # Save this stage's model
        save_checkpoint(config, flow_model_vm, flow_model_va, 
                       lambda_cost, lambda_carbon, epochs_per_stage, final=True)
        
        # Store metadata only (not full state_dict to save memory)
        pref_key = f"{lambda_cost}_{lambda_carbon}"
        all_models[pref_key] = {
            'preference': (lambda_cost, lambda_carbon),
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'model_saved': True  # Models are saved to disk via save_checkpoint
        }
        all_histories[pref_key] = loss_history
        
        # ==================== Update anchor for next stage ====================
        # Now the anchor becomes this stage's flow model
        # We need to keep the model in eval mode for anchor generation
        flow_model_vm.eval()
        flow_model_va.eval()
        
        # CRITICAL: Delete old anchor models to free GPU memory
        if use_flow_anchor and current_anchor_vm is not None:
            del current_anchor_vm
            del current_anchor_va
            torch.cuda.empty_cache()  # Force GPU memory cleanup
        
        current_anchor_vm = flow_model_vm
        current_anchor_va = flow_model_va
        use_flow_anchor = True
        
        # Clear GPU memory cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            mem_after = torch.cuda.memory_allocated() / (1024**3)
            print(f"  GPU memory after cleanup: {mem_after:.2f} GB")
        
        print(f"  Updated anchor for next stage to [{lambda_cost}, {lambda_carbon}] flow output")
    
    print("\n" + "=" * 70)
    print("CURRICULUM LEARNING COMPLETED")
    print("=" * 70)
    print(f"Trained {len(preference_schedule)} stages:")
    for pref_key, model_info in all_models.items():
        lc, le = model_info['preference']
        print(f"  [{lc:.2f}, {le:.2f}]: Best epoch {model_info['best_epoch']}, "
              f"Val Load Dev={model_info['best_val_loss']:.4f}")
    
    return all_models, all_histories


# ============================================================================
# Unified Preference-Conditioned Model Training
# ============================================================================

def create_unified_flow_model(config, input_dim, output_dim, is_vm, device):
    """
    Create unified preference-conditioned flow model.
    
    Args:
        config: Configuration object
        input_dim: Input dimension
        output_dim: Output dimension
        is_vm: True for Vm model, False for Va model
        device: Device
        
    Returns:
        model: PreferenceConditionedFM instance
    """
    model_name = "Vm" if is_vm else "Va"
    
    model = create_model('preference_flow', input_dim, output_dim, config, is_vm=is_vm)
    model.to(device)
    
    print(f"[{model_name}] Created unified preference-conditioned flow model")
    print(f"  Hidden dim: {config.hidden_dim}, Layers: {config.num_layers}")
    
    return model


def train_unified_pareto_flow(
    config, unified_model_vm, unified_model_va,
    vae_model_vm, vae_model_va,
    loss_fn, training_loader, sys_data, device,
    preference_schedule=None,  # For curriculum-style training
    epochs=500, lr=1e-4, inf_steps=10,
    zero_init=True,
    use_adaptive_weights=False,
    preference_sampling='curriculum',  # 'uniform', 'curriculum', 'fixed'
    fixed_preference=None,  # For 'fixed' mode
    test_loader=None,
    early_stopping_patience=20,
    val_freq=10,
    tb_logger=None,
    use_pareto_validation=False,  # Use hypervolume-based validation for early stopping
    pareto_val_prefs=None,  # Preference points for Pareto validation
    use_projection=False,  # Use tangent-space projection
    include_load_balance=False  # Include load balance in projection
):
    """
    Train unified preference-conditioned flow models.
    
    This function trains a SINGLE model that can generate solutions for ANY preference
    by conditioning on the preference vector [λ_cost, λ_carbon].
    
    Training Modes:
        - 'uniform': Sample preferences uniformly during training
        - 'curriculum': Progressive training with increasing carbon weight range
        - 'fixed': Train for a single fixed preference (for comparison)
    
    Args:
        config: Configuration object
        unified_model_vm, unified_model_va: PreferenceConditionedFM models
        vae_model_vm, vae_model_va: Pretrained VAE anchors
        loss_fn: MultiObjectiveOPFLoss instance
        training_loader: Training data loader
        sys_data: System data
        device: Device
        preference_schedule: List of (lambda_cost, lambda_carbon) for curriculum
        epochs: Total training epochs
        lr: Learning rate
        inf_steps: Flow integration steps
        zero_init: Initialize near zero
        use_adaptive_weights: Use adaptive constraint weights
        preference_sampling: 'uniform', 'curriculum', or 'fixed'
        fixed_preference: (lambda_cost, lambda_carbon) for fixed mode
        test_loader: Test loader for validation
        early_stopping_patience: Early stopping patience
        val_freq: Validation frequency
        
    Returns:
        unified_model_vm, unified_model_va: Trained models
        loss_history: Training history
    """
    print("=" * 70)
    print("UNIFIED PREFERENCE-CONDITIONED FLOW TRAINING")
    print("=" * 70)
    print(f"Training Mode: {preference_sampling}")
    print(f"Epochs: {epochs}, Learning rate: {lr}")
    
    if preference_sampling == 'curriculum' and preference_schedule is not None:
        print(f"Curriculum stages: {len(preference_schedule)}")
        epochs_per_stage = epochs // len(preference_schedule)
        print(f"Epochs per stage: {epochs_per_stage}")
    elif preference_sampling == 'fixed':
        if fixed_preference is None:
            fixed_preference = (0.9, 0.1)
        print(f"Fixed preference: [{fixed_preference[0]}, {fixed_preference[1]}]")
    
    # Initialize near zero if requested
    if zero_init:
        print("\nInitializing flow models near zero...")
        initialize_flow_model_near_zero(unified_model_vm, scale=0.01)
        initialize_flow_model_near_zero(unified_model_va, scale=0.01)
    
    # Attach VAE for anchor generation
    unified_model_vm.pretrain_model = vae_model_vm
    unified_model_va.pretrain_model = vae_model_va
    
    # Optimizers
    optimizer_vm = torch.optim.Adam(unified_model_vm.parameters(), lr=lr, weight_decay=1e-6)
    optimizer_va = torch.optim.Adam(unified_model_va.parameters(), lr=lr, weight_decay=1e-6)
    scheduler_vm = torch.optim.lr_scheduler.StepLR(optimizer_vm, step_size=200, gamma=0.5)
    scheduler_va = torch.optim.lr_scheduler.StepLR(optimizer_va, step_size=200, gamma=0.5)
    
    # Prepare load data
    baseMVA = sys_data.baseMVA.item() if hasattr(sys_data.baseMVA, 'item') else float(sys_data.baseMVA)
    Pd_full = torch.from_numpy(sys_data.RPd[:config.Ntrain] / baseMVA).float().to(device)
    Qd_full = torch.from_numpy(sys_data.RQd[:config.Ntrain] / baseMVA).float().to(device)
    
    # Setup constraint projection if enabled
    P_tan_t = None
    if use_projection:
        print("\nSetting up constraint projection for unified model...")
        projector = ConstraintProjectionV2(sys_data, config, use_historical_jacobian=True)
        P_tan, _, _ = projector.compute_projection_matrix(
            include_slack=False, 
            include_load_balance=include_load_balance
        )
        P_tan_t = torch.tensor(P_tan, dtype=torch.float32, device=device)
        print(f"  Projection matrix shape: {P_tan_t.shape}")
        print(f"  Include load balance: {include_load_balance}")
    
    # Loss history
    loss_history = {
        'total': [], 'objective': [], 'cost': [], 'carbon': [],
        'constraints': [], 'gen_vio': [], 'load_dev': [],
        'val_loss': [], 'val_load_dev': [],
        'current_max_carbon': []  # Track curriculum progress
    }
    
    # Early stopping
    # For Pareto validation: metric is "higher is better", init to -inf
    # For load_dev validation: metric is "lower is better", init to +inf
    if use_pareto_validation:
        best_val_loss = float('-inf')  # Higher validation metric is better
    else:
        best_val_loss = float('inf')   # Lower load deviation is better
    best_epoch = 0
    patience_counter = 0
    best_state_vm = None
    best_state_va = None
    
    # Curriculum state
    current_stage = 0
    current_max_carbon_weight = 0.1  # Start with small carbon weight
    
    start_time = time.process_time()
    
    for epoch in range(epochs):
        running_loss = 0.0
        running_cost = 0.0
        running_carbon = 0.0
        running_load_dev = 0.0
        running_load_satisfy_pct = 0.0
        running_gen_vio = 0.0
        running_branch_pf_vio = 0.0
        running_branch_ang_vio = 0.0
        running_objective = 0.0
        running_constraints = 0.0
        n_batches = 0
        
        unified_model_vm.train()
        unified_model_va.train()
        
        # Update curriculum stage
        if preference_sampling == 'curriculum' and preference_schedule is not None:
            stage_epochs = epochs // len(preference_schedule)
            new_stage = min(epoch // stage_epochs, len(preference_schedule) - 1)
            if new_stage != current_stage:
                current_stage = new_stage
                lambda_cost, lambda_carbon = preference_schedule[current_stage]
                current_max_carbon_weight = lambda_carbon
                print(f"\n[Curriculum] Stage {current_stage + 1}: max carbon weight = {current_max_carbon_weight:.2f}")
        
        for step, (train_x, _) in enumerate(training_loader):
            train_x = train_x.to(device)
            batch_size = train_x.shape[0]
            
            start_idx = step * config.batch_size_training
            end_idx = min(start_idx + batch_size, len(Pd_full))
            if end_idx - start_idx != batch_size:
                continue
            Pd_batch = Pd_full[start_idx:end_idx]
            Qd_batch = Qd_full[start_idx:end_idx]
            
            optimizer_vm.zero_grad()
            optimizer_va.zero_grad()
            
            # ==================== Sample preferences ====================
            if preference_sampling == 'uniform':
                preferences = sample_preferences_uniform(batch_size, device)
            elif preference_sampling == 'curriculum':
                preferences = sample_preferences_curriculum(
                    batch_size, device, 
                    current_max_carbon_weight=current_max_carbon_weight
                )
            else:  # 'fixed'
                lc, le = fixed_preference
                preferences = torch.tensor([[lc, le]], device=device).expand(batch_size, -1)
            
            # Get mean preference for loss computation
            mean_lambda_cost = preferences[:, 0].mean().item()
            mean_lambda_carbon = preferences[:, 1].mean().item()
            
            # ==================== Get VAE anchor ====================
            with torch.no_grad():
                z_vm = vae_model_vm(train_x, use_mean=True)
                z_va = vae_model_va(train_x, use_mean=True)
            
            # ==================== Flow forward with preference ====================
            if use_projection and P_tan_t is not None:
                y_pred_vm, y_pred_va = differentiable_flow_forward_unified_projected(
                    unified_model_vm, unified_model_va, train_x, z_vm, z_va, preferences,
                    P_tan_t, config.Nbus, num_steps=inf_steps
                )
            else:
                y_pred_vm = differentiable_flow_forward_unified(
                    unified_model_vm, train_x, z_vm, preferences, num_steps=inf_steps
                )
                y_pred_va = differentiable_flow_forward_unified(
                    unified_model_va, train_x, z_va, preferences, num_steps=inf_steps
                )
            
            # ==================== Compute loss ====================
            # Use mean preference for loss weighting
            loss, loss_dict = loss_fn(
                y_pred_vm, y_pred_va, Pd_batch, Qd_batch,
                lambda_cost=mean_lambda_cost, lambda_carbon=mean_lambda_carbon,
                update_weights=use_adaptive_weights,
                return_details=True
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unified_model_vm.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(unified_model_va.parameters(), max_norm=1.0)
            optimizer_vm.step()
            optimizer_va.step()
            
            running_loss += loss_dict['total']
            running_cost += loss_dict['cost']
            running_carbon += loss_dict['carbon']
            running_load_dev += loss_dict['load_dev']
            running_load_satisfy_pct += loss_dict['load_satisfy_pct']
            running_gen_vio += loss_dict['gen_vio']
            running_branch_pf_vio += loss_dict['branch_pf_vio']
            running_branch_ang_vio += loss_dict['branch_ang_vio']
            running_objective += loss_dict['objective']
            running_constraints += loss_dict['constraints']
            n_batches += 1
            
            # Clear intermediate tensors to prevent memory buildup
            del loss, loss_dict, y_pred_vm, y_pred_va, z_vm, z_va, preferences
        
        scheduler_vm.step()
        scheduler_va.step()
        
        # Periodic GPU cleanup (more frequent)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Record epoch metrics
        if n_batches > 0:
            avg_loss = running_loss / n_batches
            avg_cost = running_cost / n_batches
            avg_carbon = running_carbon / n_batches
            avg_load_dev = running_load_dev / n_batches
            avg_load_satisfy_pct = running_load_satisfy_pct / n_batches
            avg_gen_vio = running_gen_vio / n_batches
            avg_branch_pf_vio = running_branch_pf_vio / n_batches
            avg_branch_ang_vio = running_branch_ang_vio / n_batches
            avg_objective = running_objective / n_batches
            avg_constraints = running_constraints / n_batches
            
            # Hard constraint violation (excluding load_dev)
            avg_hard_constraint_vio = avg_gen_vio + avg_branch_pf_vio + avg_branch_ang_vio
            
            loss_history['total'].append(avg_loss)
            loss_history['cost'].append(avg_cost)
            loss_history['carbon'].append(avg_carbon)
            loss_history['load_dev'].append(avg_load_dev)
            loss_history['objective'].append(avg_objective)
            loss_history['constraints'].append(avg_constraints)
            loss_history['current_max_carbon'].append(current_max_carbon_weight)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, "
                  f"Cost={avg_cost:.2f}, Carbon={avg_carbon:.4f}, HardConstr={avg_hard_constraint_vio:.6f}, LoadSat={avg_load_satisfy_pct:.2f}%")
        
        # ==================== TensorBoard Logging (Simplified) ====================
        if tb_logger is not None and n_batches > 0:
            # Flow model output
            tb_logger.log_scalar('cost/flow_train', avg_cost, epoch)
            tb_logger.log_scalar('carbon/flow_train', avg_carbon, epoch)
            tb_logger.log_scalar('load_satisfy_pct/flow_train', avg_load_satisfy_pct, epoch)
            tb_logger.log_scalar('hard_constraint/flow_train', avg_hard_constraint_vio, epoch)
        
        # ==================== Validation ====================
        if test_loader is not None and (epoch + 1) % val_freq == 0:
            if use_pareto_validation:
                # Use Pareto front validation with hypervolume
                val_result = validate_pareto_front_unified(
                    unified_model_vm, unified_model_va,
                    vae_model_vm, vae_model_va,
                    loss_fn, test_loader, sys_data, config, device,
                    inf_steps=inf_steps,
                    preference_points=pareto_val_prefs,
                    use_projection=use_projection,
                    P_tan_t=P_tan_t
                )
                
                val_load_dev = val_result['mean_load_dev']
                val_metric = val_result['validation_metric']  # Higher is better
                
                loss_history['val_load_dev'].append(val_load_dev)
                loss_history.setdefault('val_hypervolume', []).append(val_result['hypervolume'])
                loss_history.setdefault('val_feasible_ratio', []).append(val_result['feasible_ratio'])
                
                print(f"  [Val] HV={val_result['hypervolume']:.2f}, "
                      f"Feasible={val_result['feasible_ratio']:.1%}, "
                      f"LoadDev={val_load_dev:.4f}, Metric={val_metric:.4f}")
                
                # TensorBoard logging (simplified)
                if tb_logger is not None:
                    # Flow model output
                    tb_logger.log_scalar('cost/flow_val', val_result['mean_cost'], epoch)
                    tb_logger.log_scalar('carbon/flow_val', val_result['mean_carbon'], epoch)
                    # Compute hard constraint violation from val_result if available
                    val_hard_constraint = (val_result.get('mean_gen_vio', 0.0) + 
                                          val_result.get('mean_branch_pf_vio', 0.0) + 
                                          val_result.get('mean_branch_ang_vio', 0.0))
                    # Compute load satisfaction from load_dev if not directly available
                    # Approximate: 100 - load_dev * scale (rough conversion)
                    val_load_satisfy_approx = max(0.0, 100.0 - val_load_dev * 5.0)  # Rough approximation
                    tb_logger.log_scalar('load_satisfy_pct/flow_val', val_load_satisfy_approx, epoch)
                    tb_logger.log_scalar('hard_constraint/flow_val', val_hard_constraint, epoch)
                    
                    # Pareto front visualization (useful for multi-objective)
                    if 'all_costs' in val_result and 'all_carbons' in val_result:
                        tb_logger.log_pareto_front_image(
                            epoch, 
                            val_result['all_costs'], 
                            val_result['all_carbons'],
                            feasible_mask=val_result.get('all_feasible'),
                            ref_point=val_result.get('ref_point'),
                            tag='pareto_front'
                        )
                
                # Early stopping: val_metric is "higher is better", so invert comparison
                if val_metric > best_val_loss:  # best_val_loss stores best metric (higher)
                    best_val_loss = val_metric
                    best_epoch = epoch + 1
                    patience_counter = 0
                    if best_state_vm is not None:
                        del best_state_vm
                        del best_state_va
                    best_state_vm = {k: v.detach().cpu().clone() for k, v in unified_model_vm.state_dict().items()}
                    best_state_va = {k: v.detach().cpu().clone() for k, v in unified_model_va.state_dict().items()}
                    print(f"  [Val] New best! Metric={val_metric:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"\n[Early Stop] at epoch {epoch+1}, best was epoch {best_epoch}")
                        break
            else:
                # Original validation: use load deviation only
                val_result = validate_unified_model(
                    unified_model_vm, unified_model_va,
                    vae_model_vm, vae_model_va,
                    loss_fn, test_loader, sys_data, config, device,
                    inf_steps, preference_sampling, fixed_preference, current_max_carbon_weight,
                    use_projection=use_projection, P_tan_t=P_tan_t
                )
                
                val_load_dev = val_result['load_dev']
                val_load_satisfy_pct = val_result['load_satisfy_pct']
                val_hard_constraint_vio = val_result['hard_constraint_vio']
                
                loss_history['val_load_dev'].append(val_load_dev)
                print(f"  [Val] HardConstr={val_hard_constraint_vio:.6f}, LoadSat={val_load_satisfy_pct:.2f}%")
                
                # TensorBoard: Log validation metrics
                if tb_logger is not None:
                    tb_logger.log_scalar('load_satisfy_pct/flow_val', val_load_satisfy_pct, epoch)
                    tb_logger.log_scalar('hard_constraint/flow_val', val_hard_constraint_vio, epoch)
                
                if val_load_dev < best_val_loss:
                    best_val_loss = val_load_dev
                    best_epoch = epoch + 1
                    patience_counter = 0
                    if best_state_vm is not None:
                        del best_state_vm
                        del best_state_va
                    best_state_vm = {k: v.detach().cpu().clone() for k, v in unified_model_vm.state_dict().items()}
                    best_state_va = {k: v.detach().cpu().clone() for k, v in unified_model_va.state_dict().items()}
                    print(f"  [Val] New best! LoadSat={val_load_satisfy_pct:.2f}%")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"\n[Early Stop] at epoch {epoch+1}, best was epoch {best_epoch}")
                        break
        
        # GPU thermal protection: check every 10 epochs
        if (epoch + 1) % 10 == 0:
            check_gpu_temperature(warning_temp=80, critical_temp=85, cooldown_time=30)
    
    train_time = time.process_time() - start_time
    print(f"\nTraining completed in {train_time:.1f}s ({train_time/60:.1f}min)")
    
    # Restore best model
    if best_state_vm is not None:
        unified_model_vm.load_state_dict({k: v.to(device) for k, v in best_state_vm.items()})
        unified_model_va.load_state_dict({k: v.to(device) for k, v in best_state_va.items()})
        if use_pareto_validation:
            print(f"Restored best model from epoch {best_epoch} (Pareto Metric={best_val_loss:.4f})")
        else:
            print(f"Restored best model from epoch {best_epoch} (Load Dev={best_val_loss:.4f})")
    
    # Save final model
    save_unified_checkpoint(config, unified_model_vm, unified_model_va, epochs)
    
    return unified_model_vm, unified_model_va, loss_history


def validate_unified_model(
    unified_model_vm, unified_model_va,
    vae_model_vm, vae_model_va,
    loss_fn, test_loader, sys_data, config, device,
    inf_steps, preference_sampling, fixed_preference, current_max_carbon_weight,
    use_projection=False, P_tan_t=None
):
    """Validate unified model on test set. Returns dict with metrics."""
    unified_model_vm.eval()
    unified_model_va.eval()
    
    baseMVA = sys_data.baseMVA.item() if hasattr(sys_data.baseMVA, 'item') else float(sys_data.baseMVA)
    Pd_test = torch.from_numpy(sys_data.RPd[config.Ntrain:] / baseMVA).float().to(device)
    Qd_test = torch.from_numpy(sys_data.RQd[config.Ntrain:] / baseMVA).float().to(device)
    
    val_load_dev = 0.0
    val_load_satisfy_pct = 0.0
    val_gen_vio = 0.0
    val_branch_pf_vio = 0.0
    val_branch_ang_vio = 0.0
    val_n_batches = 0
    
    with torch.no_grad():
        for step, (test_x, _) in enumerate(test_loader):
            test_x = test_x.to(device)
            batch_size = test_x.shape[0]
            
            start_idx = step * config.batch_size_test
            end_idx = min(start_idx + batch_size, len(Pd_test))
            if end_idx - start_idx != batch_size:
                continue
            
            Pd_batch = Pd_test[start_idx:end_idx]
            Qd_batch = Qd_test[start_idx:end_idx]
            
            # Use representative preference for validation
            if preference_sampling == 'fixed':
                lc, le = fixed_preference
            else:
                # Use middle of current range
                lc = 1.0 - current_max_carbon_weight / 2
                le = current_max_carbon_weight / 2
            
            preferences = torch.tensor([[lc, le]], device=device).expand(batch_size, -1)
            
            z_vm = vae_model_vm(test_x, use_mean=True)
            z_va = vae_model_va(test_x, use_mean=True)
            
            # Flow forward with optional projection
            y_pred_vm, y_pred_va = validation_flow_forward_unified(
                unified_model_vm, unified_model_va, test_x, z_vm, z_va, preferences,
                num_steps=inf_steps,
                use_projection=use_projection,
                P_tan_t=P_tan_t,
                num_buses=config.Nbus
            )
            
            _, val_dict = loss_fn(
                y_pred_vm, y_pred_va, Pd_batch, Qd_batch,
                lambda_cost=lc, lambda_carbon=le,
                update_weights=False, return_details=True
            )
            
            val_load_dev += val_dict['load_dev']
            val_load_satisfy_pct += val_dict['load_satisfy_pct']
            val_gen_vio += val_dict['gen_vio']
            val_branch_pf_vio += val_dict['branch_pf_vio']
            val_branch_ang_vio += val_dict['branch_ang_vio']
            val_n_batches += 1
    
    unified_model_vm.train()
    unified_model_va.train()
    
    if val_n_batches > 0:
        return {
            'load_dev': val_load_dev / val_n_batches,
            'load_satisfy_pct': val_load_satisfy_pct / val_n_batches,
            'hard_constraint_vio': (val_gen_vio + val_branch_pf_vio + val_branch_ang_vio) / val_n_batches,
        }
    else:
        return {'load_dev': 0.0, 'load_satisfy_pct': 100.0, 'hard_constraint_vio': 0.0}


def validate_pareto_front_unified(
    unified_model_vm, unified_model_va,
    vae_model_vm, vae_model_va,
    loss_fn, test_loader, sys_data, config, device,
    inf_steps=10,
    preference_points=None,
    feasibility_thresholds=None,
    ref_point=None,
    max_samples=500,  # Reduced to prevent GPU OOM during validation
    use_projection=False,  # Use tangent-space projection
    P_tan_t=None  # Precomputed projection matrix
):
    """
    Validate unified model by evaluating Pareto front quality across multiple preferences.
    
    This function generates solutions for multiple preference points and computes:
    1. Hypervolume of the Pareto front (larger is better)
    2. Feasibility ratio (percentage of solutions satisfying constraints)
    3. Average constraint violations
    
    Args:
        unified_model_vm, unified_model_va: Unified preference-conditioned models
        vae_model_vm, vae_model_va: VAE anchor models
        loss_fn: Multi-objective loss function
        test_loader: Test data loader
        sys_data: System data
        config: Configuration
        device: Device
        inf_steps: Number of flow integration steps
        preference_points: List of (lambda_cost, lambda_carbon) tuples to evaluate
                          If None, uses default [1.0, 0.9, 0.8, ..., 0.1]
        feasibility_thresholds: Dict with constraint thresholds
        ref_point: Reference point for hypervolume (if None, auto-computed)
        max_samples: Maximum number of test samples to use
        
    Returns:
        dict: {
            'hypervolume': float,
            'feasible_ratio': float,
            'mean_load_dev': float,
            'mean_cost': float,
            'mean_carbon': float,
            'pareto_points': np.ndarray,
            'validation_metric': float  # Combined metric for early stopping
        }
    """
    unified_model_vm.eval()
    unified_model_va.eval()
    
    # Default preference points: from economic-only to carbon-focused
    if preference_points is None:
        preference_points = [(1.0 - i * 0.1, i * 0.1) for i in range(10)]
    
    # Default feasibility thresholds
    if feasibility_thresholds is None:
        feasibility_thresholds = {
            'load_dev': 0.02,   # 2% load deviation
            'gen_vio': 0.005,   # 0.5% generator violation
        }
    
    baseMVA = sys_data.baseMVA.item() if hasattr(sys_data.baseMVA, 'item') else float(sys_data.baseMVA)
    Pd_test = torch.from_numpy(sys_data.RPd[config.Ntrain:] / baseMVA).float().to(device)
    Qd_test = torch.from_numpy(sys_data.RQd[config.Ntrain:] / baseMVA).float().to(device)
    
    # Collect results for each preference point
    all_costs = []
    all_carbons = []
    all_load_devs = []
    all_gen_vios = []
    all_feasible = []
    
    with torch.no_grad():
        for lc, le in preference_points:
            pref_costs = []
            pref_carbons = []
            pref_load_devs = []
            pref_gen_vios = []
            n_samples = 0
            
            for step, (test_x, _) in enumerate(test_loader):
                if n_samples >= max_samples:
                    break
                    
                test_x = test_x.to(device)
                batch_size = test_x.shape[0]
                
                start_idx = step * config.batch_size_test
                end_idx = min(start_idx + batch_size, len(Pd_test))
                if end_idx - start_idx != batch_size:
                    continue
                
                Pd_batch = Pd_test[start_idx:end_idx]
                Qd_batch = Qd_test[start_idx:end_idx]
                
                # Create preference tensor
                preferences = torch.tensor([[lc, le]], device=device).expand(batch_size, -1)
                
                # Get VAE anchor
                z_vm = vae_model_vm(test_x, use_mean=True)
                z_va = vae_model_va(test_x, use_mean=True)
                
                # Flow forward with optional projection
                y_pred_vm, y_pred_va = validation_flow_forward_unified(
                    unified_model_vm, unified_model_va, test_x, z_vm, z_va, preferences,
                    num_steps=inf_steps,
                    use_projection=use_projection,
                    P_tan_t=P_tan_t,
                    num_buses=config.Nbus
                )
                
                # Compute loss and get metrics
                _, loss_dict = loss_fn(
                    y_pred_vm, y_pred_va, Pd_batch, Qd_batch,
                    lambda_cost=lc, lambda_carbon=le,
                    update_weights=False, return_details=True
                )
                
                pref_costs.append(loss_dict['cost'])
                pref_carbons.append(loss_dict['carbon'])
                pref_load_devs.append(loss_dict['load_dev'])
                pref_gen_vios.append(loss_dict.get('gen_vio', 0.0))
                n_samples += batch_size
            
            # Average metrics for this preference
            if pref_costs:
                avg_cost = np.mean(pref_costs)
                avg_carbon = np.mean(pref_carbons)
                avg_load_dev = np.mean(pref_load_devs)
                avg_gen_vio = np.mean(pref_gen_vios)
                
                all_costs.append(avg_cost)
                all_carbons.append(avg_carbon)
                all_load_devs.append(avg_load_dev)
                all_gen_vios.append(avg_gen_vio)
                
                # Check feasibility
                is_feasible = check_feasibility(
                    {'load_dev': avg_load_dev, 'gen_vio': avg_gen_vio},
                    feasibility_thresholds
                )
                all_feasible.append(is_feasible)
    
    unified_model_vm.train()
    unified_model_va.train()
    
    # Convert to numpy arrays
    all_costs = np.array(all_costs)
    all_carbons = np.array(all_carbons)
    all_feasible = np.array(all_feasible)
    
    # Compute reference point if not provided
    if ref_point is None:
        ref_point = np.array([
            np.max(all_costs) * 1.1 if len(all_costs) > 0 else 1e6,
            np.max(all_carbons) * 1.1 if len(all_carbons) > 0 else 1e6
        ])
    
    # Evaluate Pareto front
    pareto_result = evaluate_pareto_front(all_costs, all_carbons, all_feasible, ref_point)
    
    # Compute combined validation metric
    validation_metric = get_pareto_validation_metric(
        pareto_result['hypervolume'],
        pareto_result['feasible_ratio'],
        hv_weight=0.6,
        feas_weight=0.4,
        hv_scale=ref_point[0] * ref_point[1]  # Normalize by max possible HV
    )
    
    # Build per-preference results for detailed logging
    preference_results = []
    for i, (lc, le) in enumerate(preference_points):
        if i < len(all_costs):
            preference_results.append({
                'lambda_cost': lc,
                'lambda_carbon': le,
                'cost': all_costs[i],
                'carbon': all_carbons[i],
                'load_dev': all_load_devs[i] if i < len(all_load_devs) else 0.0,
                'gen_vio': all_gen_vios[i] if i < len(all_gen_vios) else 0.0,
                'feasible': all_feasible[i] if i < len(all_feasible) else False
            })
    
    result = {
        'hypervolume': pareto_result['hypervolume'],
        'feasible_ratio': pareto_result['feasible_ratio'],
        'n_feasible': pareto_result['n_feasible'],
        'n_total': pareto_result['n_total'],
        'mean_load_dev': np.mean(all_load_devs) if all_load_devs else 0.0,
        'mean_cost': np.mean(all_costs) if len(all_costs) > 0 else 0.0,
        'mean_carbon': np.mean(all_carbons) if len(all_carbons) > 0 else 0.0,
        'pareto_points': pareto_result['pareto_points'],
        'ref_point': ref_point,
        'validation_metric': validation_metric,
        # Additional details for TensorBoard
        'all_costs': all_costs,
        'all_carbons': all_carbons,
        'all_feasible': all_feasible,
        'preference_results': preference_results
    }
    
    return result


def save_unified_checkpoint(config, model_vm, model_va, epoch):
    """Save unified model checkpoint."""
    vm_path = os.path.join(config.model_save_dir, f'unified_pareto_vm_E{epoch}.pth')
    va_path = os.path.join(config.model_save_dir, f'unified_pareto_va_E{epoch}.pth')
    
    torch.save(model_vm.state_dict(), vm_path, _use_new_zipfile_serialization=False)
    torch.save(model_va.state_dict(), va_path, _use_new_zipfile_serialization=False)
    
    print(f"  Unified model saved: {vm_path}")


def inference_unified_model(
    unified_model_vm, unified_model_va,
    vae_model_vm, vae_model_va,
    x, preference, device, num_steps=10
):
    """
    Run inference with unified model for any preference.
    
    Args:
        unified_model_vm, unified_model_va: Trained unified models
        vae_model_vm, vae_model_va: VAE anchors
        x: Input condition [batch, input_dim]
        preference: [lambda_cost, lambda_carbon] tuple or [batch, 2] tensor
        device: Device
        num_steps: Integration steps
        
    Returns:
        y_vm, y_va: Predicted Vm and Va
    """
    unified_model_vm.eval()
    unified_model_va.eval()
    
    with torch.no_grad():
        x = x.to(device)
        batch_size = x.shape[0]
        
        # Handle preference format
        if isinstance(preference, (tuple, list)):
            lc, le = preference
            preference = torch.tensor([[lc, le]], device=device).expand(batch_size, -1)
        else:
            preference = preference.to(device)
        
        # Get VAE anchor
        z_vm = vae_model_vm(x, use_mean=True)
        z_va = vae_model_va(x, use_mean=True)
        
        # Flow integration
        y_vm = unified_model_vm.flow_backward(x, z_vm, preference, num_steps=num_steps)
        y_va = unified_model_va.flow_backward(x, z_va, preference, num_steps=num_steps)
    
    return y_vm, y_va


def plot_training_curves(loss_history, lambda_cost, lambda_carbon, save_path=None):
    """Plot training loss curves."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Pareto Flow Training: λ_cost={lambda_cost}, λ_carbon={lambda_carbon}')
    
    # Total loss
    axes[0, 0].plot(loss_history['total'])
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True)
    
    # Objective
    axes[0, 1].plot(loss_history['objective'])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Objective (Weighted Cost + Carbon)')
    axes[0, 1].grid(True)
    
    # Cost
    axes[0, 2].plot(loss_history['cost'])
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('$/h')
    axes[0, 2].set_title('Economic Cost')
    axes[0, 2].grid(True)
    
    # Carbon
    axes[1, 0].plot(loss_history['carbon'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('tCO2/h')
    axes[1, 0].set_title('Carbon Emission')
    axes[1, 0].grid(True)
    
    # Constraints
    axes[1, 1].plot(loss_history['constraints'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Constraint Violations')
    axes[1, 1].grid(True)
    
    # Generator violation
    axes[1, 2].plot(loss_history['gen_vio'], label='Gen Vio')
    axes[1, 2].plot(loss_history['load_dev'], label='Load Dev')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].set_title('Detailed Violations')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Pareto-Adaptive Rectified Flow Model')
    
    # ==================== Model Architecture Selection ====================
    parser.add_argument('--model_mode', type=str, default='independent',
                        choices=['independent', 'unified'],
                        help='Model architecture mode:\n'
                             '  independent: Separate models per preference stage (original)\n'
                             '  unified: Single preference-conditioned model for all preferences')
    
    # Mode selection (for independent mode)
    parser.add_argument('--curriculum', action='store_true',
                        help='Enable curriculum learning mode (progressive preference shifts)')
    
    # Single-stage training parameters (for independent mode)
    parser.add_argument('--lambda_cost', type=float, default=0.9,
                        help='Weight for economic cost (default: 0.9)')
    parser.add_argument('--lambda_carbon', type=float, default=0.1,
                        help='Weight for carbon emission (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='Number of training epochs (default: 1500, matches 10 stages × 150 epochs)')
    
    # Curriculum learning parameters (tuned defaults from independent model experiments)
    parser.add_argument('--start_cost', type=float, default=1.0,
                        help='Starting cost weight for curriculum (default: 1.0)')
    parser.add_argument('--end_cost', type=float, default=0.1,
                        help='Ending cost weight for curriculum (default: 0.1)')
    parser.add_argument('--pref_step', type=float, default=0.1,
                        help='Preference step size for curriculum (default: 0.1)')
    parser.add_argument('--epochs_per_stage', type=int, default=150,
                        help='Epochs per curriculum stage (default: 150)')
    
    # ==================== Unified Model Parameters ====================
    parser.add_argument('--pref_sampling', type=str, default='curriculum',
                        choices=['uniform', 'curriculum', 'fixed'],
                        help='Preference sampling strategy for unified model:\n'
                             '  uniform: Sample preferences uniformly from [0,1]\n'
                             '  curriculum: Progressive sampling with increasing range\n'
                             '  fixed: Train for single fixed preference')
    
    # Common parameters
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--inf_steps', type=int, default=10,
                        help='Inference steps for flow backward (default: 10)')
    parser.add_argument('--use_pretrained_flow', action='store_true',
                        help='Initialize from pretrained rectified flow model')
    parser.add_argument('--use_projection', action='store_true', default=False,
                        help='Enable constraint tangent-space projection during training. '
                             'Note: Current projection tries to keep generator power CONSTANT, '
                             'not within bounds. May over-constrain the search. '
                             'Recommended: False (rely on loss function penalties instead).')
    parser.add_argument('--use_drift_correction', action='store_true', default=False,
                        help='Enable drift-correction (tangent + normal correction). '
                             'Only needed if VAE anchor violates constraints. '
                             'Default: False (VAE anchors are typically feasible).')
    parser.add_argument('--lambda_cor', type=float, default=5.0,
                        help='Correction strength for drift-correction (default: 5.0)')
    parser.add_argument('--no_zero_init', action='store_true',
                        help='Disable zero initialization of flow model output layer')
    parser.add_argument('--include_load_balance', action='store_true', default=True,
                        help='Include load balance constraints in projection matrix')
    parser.add_argument('--adaptive_weights', action='store_true', default=True,
                        help='Enable adaptive constraint weight scheduling (default: True)')
    parser.add_argument('--no_adaptive_weights', action='store_true',
                        help='Disable adaptive constraint weight scheduling')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (default: 20)')
    parser.add_argument('--val_freq', type=int, default=10,
                        help='Validation frequency in epochs (default: 10)')
    parser.add_argument('--pareto_validation', action='store_true', default=True,
                        help='Use Pareto front hypervolume for validation (unified model only)')
    
    # GPU thermal protection
    # parser.add_argument('--batch_delay', type=int, default=0,
    #                     help='Delay in ms between batches to reduce GPU load (default: 0)')
    # parser.add_argument('--temp_warning', type=int, default=80,
    #                     help='GPU temperature to show warning (default: 80°C)')
    # parser.add_argument('--temp_critical', type=int, default=85,
    #                     help='GPU temperature to pause training (default: 85°C)')
    
    args = parser.parse_args()
    
    # Handle adaptive_weights flag
    if args.no_adaptive_weights:
        args.adaptive_weights = False
    
    print("=" * 60)
    print("Pareto-Adaptive Rectified Flow Training")
    print("=" * 60)
    
    # Generate preference schedule (used by both modes)
    preference_schedule = generate_preference_schedule(
        args.start_cost, args.end_cost, args.pref_step
    )
    
    if args.model_mode == 'unified':
        print("=" * 60)
        print("Model Mode: UNIFIED (Single model for all preferences)")
        print("=" * 60)
        print(f"Preference sampling: {args.pref_sampling}")
        if args.pref_sampling == 'curriculum':
            print(f"Preference schedule: {len(preference_schedule)} stages")
        elif args.pref_sampling == 'fixed':
            print(f"Fixed preference: [{args.lambda_cost}, {args.lambda_carbon}]")
    elif args.curriculum:
        print("Model Mode: INDEPENDENT with CURRICULUM LEARNING")
        print(f"Preference schedule: {len(preference_schedule)} stages")
    else:
        print("Model Mode: INDEPENDENT with SINGLE PREFERENCE")
        print(f"Target preference: [{args.lambda_cost}, {args.lambda_carbon}]")
    
    # Load configuration
    config = get_config()
    config.print_config()
    
    device = config.device
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\nLoading data...")
    sys_data, dataloaders, BRANFT = load_all_data(config)
    
    # Get model dimensions
    input_dim = sys_data.x_train.shape[1]
    output_dim_vm = sys_data.yvm_train.shape[1]
    output_dim_va = sys_data.yva_train.shape[1]
    
    print(f"\nModel dimensions:")
    print(f"  Input: {input_dim}")
    print(f"  Vm output: {output_dim_vm}")
    print(f"  Va output: {output_dim_va}")
    
    # Load pretrained VAE models (anchor generators)
    print("\n--- Loading VAE Anchor Generators ---")
    vae_model_vm = load_pretrained_vae(config, input_dim, output_dim_vm, 
                                       is_vm=True, device=device)
    vae_model_va = load_pretrained_vae(config, input_dim, output_dim_va, 
                                       is_vm=False, device=device)
    
    # Get GCI values
    print("\n--- Configuring GCI Values ---")
    gci_values = get_gci_for_generators(config, sys_data)
    print(f"GCI range: [{gci_values.min():.4f}, {gci_values.max():.4f}] tCO2/MWh")
    
    # ============================================================================
    # Create TensorBoard Logger
    # ============================================================================
    print("\n--- Setting up TensorBoard Logger ---")
    tb_logger = create_tensorboard_logger(
        config, 
        model_mode=args.model_mode,
        pref_sampling=args.pref_sampling if args.model_mode == 'unified' else 'curriculum',
        lambda_cost=args.lambda_cost,
        lambda_carbon=args.lambda_carbon
    )
    
    # ============================================================================
    # UNIFIED MODEL MODE
    # ============================================================================
    if args.model_mode == 'unified':
        print("\n--- Creating Unified Preference-Conditioned Models ---")
        
        # Create unified models
        unified_model_vm = create_unified_flow_model(
            config, input_dim, output_dim_vm, is_vm=True, device=device
        )
        unified_model_va = create_unified_flow_model(
            config, input_dim, output_dim_va, is_vm=False, device=device
        )
        
        # Create loss function
        use_adaptive = args.adaptive_weights
        loss_fn = MultiObjectiveOPFLoss(sys_data, config, gci_values, use_adaptive_weights=use_adaptive)
        loss_fn = loss_fn.to(device)
        
        # Train unified model
        test_loader = dataloaders.get('test_vm', None)
        
        unified_model_vm, unified_model_va, loss_history = train_unified_pareto_flow(
            config, unified_model_vm, unified_model_va,
            vae_model_vm, vae_model_va,
            loss_fn, dataloaders['train_vm'], sys_data, device,
            preference_schedule=preference_schedule if args.pref_sampling == 'curriculum' else None,
            epochs=args.epochs,
            lr=args.lr,
            inf_steps=args.inf_steps,
            zero_init=not args.no_zero_init,
            use_adaptive_weights=use_adaptive,
            preference_sampling=args.pref_sampling,
            fixed_preference=(args.lambda_cost, args.lambda_carbon) if args.pref_sampling == 'fixed' else None,
            test_loader=test_loader,
            early_stopping_patience=args.patience,
            val_freq=args.val_freq,
            tb_logger=tb_logger,
            use_pareto_validation=args.pareto_validation,
            use_projection=args.use_projection,
            include_load_balance=args.include_load_balance
        )
        
        # Plot training curves
        plot_path = os.path.join(config.results_dir, f'unified_pareto_training_{args.pref_sampling}.png')
        plot_training_curves(loss_history, 0.5, 0.5, plot_path)  # Use middle preference for title
        
        # Close TensorBoard logger
        if tb_logger is not None:
            tb_logger.close()
        
        print("\n" + "=" * 60)
        print("Unified Model Training completed successfully!")
        print(f"Model can generate solutions for ANY preference [λ_cost, λ_carbon]")
        print("=" * 60)
        
    # ============================================================================
    # INDEPENDENT MODEL MODE
    # ============================================================================
    elif args.curriculum:
        # ==================== CURRICULUM LEARNING MODE ====================
        all_models, all_histories = train_curriculum_pareto_flow(
            config, sys_data, dataloaders, device,
            vae_model_vm, vae_model_va,
            gci_values,
            preference_schedule,
            epochs_per_stage=args.epochs_per_stage,
            lr=args.lr,
            inf_steps=args.inf_steps,
            use_projection=args.use_projection,
            use_drift_correction=args.use_drift_correction,
            lambda_cor=args.lambda_cor,
            zero_init=not args.no_zero_init,
            include_load_balance=args.include_load_balance,
            use_adaptive_weights=args.adaptive_weights,
            early_stopping_patience=args.patience,
            val_freq=args.val_freq,
            tb_logger=tb_logger
        )
        
        # Plot training curves for each stage
        for pref_key, history in all_histories.items():
            lc, le = pref_key.split('_')
            plot_path = os.path.join(config.results_dir, f'pareto_curriculum_{pref_key}.png')
            plot_training_curves(history, float(lc), float(le), plot_path)
        
        print("\n" + "=" * 60)
        print("Curriculum Learning completed successfully!")
        print(f"Trained {len(preference_schedule)} stages")
        print("=" * 60)
        
    else:
        # ==================== SINGLE PREFERENCE MODE ====================
        # Create flow models
        print("\n--- Creating Flow Models ---")
        pretrained_vm_path = None
        pretrained_va_path = None
        if args.use_pretrained_flow:
            pretrained_vm_path = os.path.join(config.model_save_dir,
                f'modelvm{config.Nbus}r{config.sys_R}N{config.model_version}Lm8642_rectified_E{config.EpochVm}F1.pth')
            pretrained_va_path = os.path.join(config.model_save_dir,
                f'modelva{config.Nbus}r{config.sys_R}N{config.model_version}La8642_rectified_E{config.EpochVa}F1.pth')
        
        flow_model_vm = create_flow_model(config, input_dim, output_dim_vm, 
                                          is_vm=True, device=device,
                                          pretrained_flow_path=pretrained_vm_path)
        flow_model_va = create_flow_model(config, input_dim, output_dim_va, 
                                          is_vm=False, device=device,
                                          pretrained_flow_path=pretrained_va_path)
        
        # Attach VAE models to flow models
        flow_model_vm.pretrain_model = vae_model_vm
        flow_model_va.pretrain_model = vae_model_va
        
        # Create multi-objective loss function
        print("\n--- Creating Multi-Objective Loss ---")
        use_adaptive = args.adaptive_weights
        loss_fn = MultiObjectiveOPFLoss(sys_data, config, gci_values, use_adaptive_weights=use_adaptive)
        loss_fn = loss_fn.to(device)
        if use_adaptive:
            print("  Adaptive constraint weight scheduling: ENABLED")
        
        # Train
        print("\n" + "=" * 60)
        test_loader = dataloaders.get('test_vm', None)
        
        flow_model_vm, flow_model_va, loss_history = train_pareto_flow(
            config, flow_model_vm, flow_model_va,
            vae_model_vm, vae_model_va,
            loss_fn, dataloaders['train_vm'], sys_data, device,
            lambda_cost=args.lambda_cost,
            lambda_carbon=args.lambda_carbon,
            epochs=args.epochs,
            lr=args.lr,
            inf_steps=args.inf_steps,
            use_projection=args.use_projection,
            use_drift_correction=args.use_drift_correction,
            lambda_cor=args.lambda_cor,
            zero_init=not args.no_zero_init,
            include_load_balance=args.include_load_balance,
            use_adaptive_weights=use_adaptive,
            test_loader=test_loader,
            early_stopping_patience=args.patience,
            val_freq=args.val_freq,
            tb_logger=tb_logger
        )
        
        # Plot training curves
        pref_str = f"{args.lambda_cost}_{args.lambda_carbon}".replace(".", "")
        plot_path = os.path.join(config.results_dir, f'pareto_training_{pref_str}.png')
        plot_training_curves(loss_history, args.lambda_cost, args.lambda_carbon, plot_path)
        
        # Close TensorBoard logger
        if tb_logger is not None:
            tb_logger.close()
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print(f"Model saved with preference [{args.lambda_cost}, {args.lambda_carbon}]")
        print("=" * 60)


if __name__ == "__main__":
    main()

