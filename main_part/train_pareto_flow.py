#!/usr/bin/env python
# coding: utf-8
"""
Training Script for Pareto-Adaptive Rectified Flow Model (Refactored)

This script trains a rectified flow model to map from VAE-generated 
economic-only solutions (preference [1,0]) to Pareto-optimal solutions 
under various target preferences.

Usage:
    python train_pareto_flow.py --model_mode unified --pref_sampling curriculum --epochs 1500
    python train_pareto_flow.py --model_mode independent --lambda_cost 0.9 --lambda_carbon 0.1

Author: Auto-generated
Date: 2024
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch 
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt

from config import get_config
from models import create_model
from data_loader import load_all_data
from multi_objective_loss import MultiObjectiveOPFLoss, get_gci_for_generators
from utils import ( 
    create_tensorboard_logger,  
    gpu_memory_cleanup,
    check_gpu_temperature, 
    sample_preferences_uniform,
    sample_preferences_curriculum,
    initialize_flow_model_near_zero,
    check_feasibility,
    evaluate_pareto_front,
    get_pareto_validation_metric,
)
from flow_model.post_processing import ConstraintProjectionV2


# ============================================================================
# Flow Forward Functions
# ============================================================================

def flow_forward_simple(flow_model, x, z_anchor, num_steps=10):
    """Basic differentiable flow integration (no preference conditioning)."""
    batch_size = x.shape[0]
    device = x.device
    
    z = z_anchor.detach().clone().requires_grad_(True)
    dt = 1.0 / num_steps
    
    for step_idx in range(num_steps):
        t = torch.full((batch_size, 1), step_idx * dt, device=device)
        v = flow_model.model(x, z, t)
        z = z + v * dt
    
    return z


def flow_forward_unified(flow_model, x, z_anchor, preference, num_steps=10):
    """Differentiable flow integration with preference conditioning."""
    batch_size = x.shape[0]
    device = x.device
    
    z = z_anchor.detach().clone().requires_grad_(True)
    dt = 1.0 / num_steps
    
    for step_idx in range(num_steps):
        t = torch.full((batch_size, 1), step_idx * dt, device=device)
        v = flow_model.model(x, z, t, preference)
        z = z + v * dt
    
    return z


def flow_forward_projected(flow_vm, flow_va, x, z_vm, z_va, P_tan_t, num_buses, 
                           num_steps=10, preference=None):
    """Flow integration with tangent-space projection."""
    batch_size = x.shape[0]
    device = x.device
    dt = 1.0 / num_steps
    
    z_vm = z_vm.detach().clone().requires_grad_(True)
    z_va = z_va.detach().clone().requires_grad_(True)
    
    for step_idx in range(num_steps):
        t = torch.full((batch_size, 1), step_idx * dt, device=device)
        
        # Predict velocities
        if preference is not None:
            v_vm = flow_vm.model(x, z_vm, t, preference)
            v_va = flow_va.model(x, z_va, t, preference)
        else:
            v_vm = flow_vm.model(x, z_vm, t)
            v_va = flow_va.model(x, z_va, t)
        
        # Project to tangent space
        v_combined = torch.cat([v_vm, v_va], dim=1)
        v_projected = torch.matmul(v_combined, P_tan_t.T)
        
        v_vm_proj = v_projected[:, :num_buses]
        v_va_proj = v_projected[:, num_buses:]
        
        z_vm = z_vm + v_vm_proj * dt
        z_va = z_va + v_va_proj * dt
    
    return z_vm, z_va


# ============================================================================
# Strategy Classes - Abstract Core Differences
# ============================================================================

class AnchorStrategy:
    """Strategy for obtaining initial anchor points (z_vm, z_va)."""
    
    def __init__(self, vae_vm, vae_va, prev_flow_vm=None, prev_flow_va=None, inf_steps=10):
        self.vae_vm = vae_vm
        self.vae_va = vae_va
        self.prev_flow_vm = prev_flow_vm
        self.prev_flow_va = prev_flow_va
        self.inf_steps = inf_steps

    def get_anchor(self, x):
        """Get anchor points from VAE (optionally transformed by previous flow)."""
        with torch.no_grad():
            z_vm = self.vae_vm(x, use_mean=True)
            z_va = self.vae_va(x, use_mean=True)
            
            # If previous flow model exists (curriculum chain), apply it
            if self.prev_flow_vm is not None:
                z_vm = flow_forward_simple(self.prev_flow_vm, x, z_vm, self.inf_steps)
                z_va = flow_forward_simple(self.prev_flow_va, x, z_va, self.inf_steps)
        
        return z_vm, z_va


class PreferenceStrategy:
    """Strategy for obtaining preference vectors."""
    
    def __init__(self, mode='fixed', fixed_value=None, max_carbon=0.1):
        """
        Args:
            mode: 'fixed', 'uniform', or 'curriculum'
            fixed_value: (lambda_cost, lambda_carbon) for fixed mode
            max_carbon: Maximum carbon weight for curriculum mode
        """
        self.mode = mode
        self.fixed_value = fixed_value or (0.9, 0.1)
        self.max_carbon = max_carbon

    def get_preference(self, batch_size, device):
        """Sample preferences based on strategy."""
        if self.mode == 'fixed':
            lc, le = self.fixed_value
            return torch.tensor([[lc, le]], device=device).expand(batch_size, -1)
        elif self.mode == 'uniform':
            return sample_preferences_uniform(batch_size, device)
        elif self.mode == 'curriculum':
            return sample_preferences_curriculum(batch_size, device, self.max_carbon)
        return None
    
    def get_mean_weights(self, preferences=None):
        """Get mean weights for loss computation."""
        if self.mode == 'fixed':
            return self.fixed_value
        elif preferences is not None:
            return preferences[:, 0].mean().item(), preferences[:, 1].mean().item()
        return 0.5, 0.5


# ============================================================================
# Generic Training Components
# ============================================================================

def train_one_epoch(
    flow_vm, flow_va, optimizers, anchor_strategy, pref_strategy,
    loss_fn, dataloader, Pd_full, Qd_full, config, device,
    is_unified=False, inf_steps=10, use_projection=False, P_tan_t=None,
    use_adaptive_weights=False
):
    """
    Generic training loop for one epoch.
    
    Handles both unified (preference-conditioned) and independent models.
    """
    flow_vm.train()
    flow_va.train()
    
    opt_vm, opt_va = optimizers
    
    # Metrics accumulator
    metrics = {
        'loss': 0.0, 'cost': 0.0, 'carbon': 0.0, 'objective': 0.0,
        'constraints': 0.0, 'load_dev': 0.0, 'load_satisfy_pct': 0.0,
        'satisfaction': 0.0, 'gen_vio': 0.0, 'branch_pf_vio': 0.0, 'branch_ang_vio': 0.0
    }
    n_batches = 0
    
    for step, (train_x, _) in enumerate(dataloader):
        train_x = train_x.to(device)
        batch_size = train_x.shape[0]
        
        # Get batch data
        start_idx = step * config.batch_size_training
        end_idx = min(start_idx + batch_size, len(Pd_full))
        if end_idx - start_idx != batch_size:
            continue
        Pd_batch = Pd_full[start_idx:end_idx]
        Qd_batch = Qd_full[start_idx:end_idx]
        
        opt_vm.zero_grad()
        opt_va.zero_grad()
        
        # 1. Get anchor points
        z_vm, z_va = anchor_strategy.get_anchor(train_x)
        
        # 2. Get preferences
        prefs = pref_strategy.get_preference(batch_size, device) if is_unified else None
        
        # 3. Flow forward
        if use_projection and P_tan_t is not None:
            y_vm, y_va = flow_forward_projected(
                flow_vm, flow_va, train_x, z_vm, z_va, 
                P_tan_t, config.Nbus, inf_steps, prefs
            )
        elif is_unified:
            y_vm = flow_forward_unified(flow_vm, train_x, z_vm, prefs, inf_steps)
            y_va = flow_forward_unified(flow_va, train_x, z_va, prefs, inf_steps)
        else:
            y_vm = flow_forward_simple(flow_vm, train_x, z_vm, inf_steps)
            y_va = flow_forward_simple(flow_va, train_x, z_va, inf_steps)
        
        # 4. Compute loss
        lc, le = pref_strategy.get_mean_weights(prefs)
        loss, loss_dict = loss_fn(
            y_vm, y_va, Pd_batch, Qd_batch,
            lambda_cost=lc, lambda_carbon=le,
            update_weights=use_adaptive_weights,
            return_details=True
        )
        
        # 5. Backward and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow_vm.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(flow_va.parameters(), 1.0)
        opt_vm.step()
        opt_va.step()
        
        # Accumulate metrics
        for key in metrics:
            if key in loss_dict:
                metrics[key] += loss_dict[key]
        metrics['loss'] += loss_dict['total']
        n_batches += 1
        
        del loss, loss_dict, y_vm, y_va
    
    # Average metrics
    return {k: v / max(n_batches, 1) for k, v in metrics.items()}, n_batches


def validate_model(
    flow_vm, flow_va, vae_vm, vae_va, loss_fn,
    test_loader, Pd_test, Qd_test, config, device,
    is_unified=False, inf_steps=10, pref_strategy=None,
    use_projection=False, P_tan_t=None
):
    """Generic validation function."""
    flow_vm.eval()
    flow_va.eval()
    
    metrics = {
        'loss': 0.0, 'cost': 0.0, 'carbon': 0.0,
        'load_dev': 0.0, 'load_satisfy_pct': 0.0, 'satisfaction': 0.0,
        'gen_vio': 0.0, 'branch_pf_vio': 0.0, 'branch_ang_vio': 0.0
    }
    n_batches = 0
    
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
            z_vm = vae_vm(test_x, use_mean=True)
            z_va = vae_va(test_x, use_mean=True)
            
            # Get preferences
            prefs = pref_strategy.get_preference(batch_size, device) if is_unified else None
            
            # Flow forward
            if use_projection and P_tan_t is not None:
                y_vm, y_va = flow_forward_projected(
                    flow_vm, flow_va, test_x, z_vm, z_va,
                    P_tan_t, config.Nbus, inf_steps, prefs
                )
            elif is_unified:
                y_vm = flow_forward_unified(flow_vm, test_x, z_vm, prefs, inf_steps)
                y_va = flow_forward_unified(flow_va, test_x, z_va, prefs, inf_steps)
            else:
                y_vm = flow_forward_simple(flow_vm, test_x, z_vm, inf_steps)
                y_va = flow_forward_simple(flow_va, test_x, z_va, inf_steps)
            
            lc, le = pref_strategy.get_mean_weights(prefs) if pref_strategy else (0.9, 0.1)
            _, loss_dict = loss_fn(
                y_vm, y_va, Pd_batch, Qd_batch,
                lambda_cost=lc, lambda_carbon=le,
                update_weights=False, return_details=True
            )
            
            for key in metrics:
                if key in loss_dict:
                    metrics[key] += loss_dict[key]
            metrics['loss'] += loss_dict['total']
            n_batches += 1
    
    flow_vm.train()
    flow_va.train()
    
    avg_metrics = {k: v / max(n_batches, 1) for k, v in metrics.items()}
    avg_metrics['hard_constraint_vio'] = (
        avg_metrics['gen_vio'] + avg_metrics['branch_pf_vio'] + avg_metrics['branch_ang_vio']
    )
    return avg_metrics


def evaluate_vae_baseline(vae_vm, vae_va, loss_fn, dataloader, Pd_full, Qd_full, 
                          config, device, lambda_cost=0.9, lambda_carbon=0.1):
    """Evaluate VAE baseline quality."""
    vae_vm.eval()
    vae_va.eval()
    
    # Metrics that need sample-weighted average
    sample_metrics = {'cost': 0.0, 'carbon': 0.0, 'gen_vio': 0.0, 'load_dev': 0.0,
                      'branch_pf_vio': 0.0, 'branch_ang_vio': 0.0, 'satisfaction': 0.0}
    # Metrics that need batch-averaged (already batch-level values)
    batch_metrics = {'load_satisfy_pct': 0.0}
    
    n_samples = 0
    n_batches = 0
    
    with torch.no_grad():
        for step, (train_x, _) in enumerate(dataloader):
            train_x = train_x.to(device)
            batch_size = train_x.shape[0]
            
            start_idx = step * config.batch_size_training
            end_idx = min(start_idx + batch_size, len(Pd_full))
            if end_idx - start_idx != batch_size:
                continue
            
            z_vm = vae_vm(train_x, use_mean=True)
            z_va = vae_va(train_x, use_mean=True)
            
            _, loss_dict = loss_fn(
                z_vm, z_va, Pd_full[start_idx:end_idx], Qd_full[start_idx:end_idx],
                lambda_cost=lambda_cost, lambda_carbon=lambda_carbon,
                return_details=True
            )
            
            # Accumulate sample-weighted metrics
            for key in sample_metrics:
                if key in loss_dict:
                    sample_metrics[key] += loss_dict[key] * batch_size
            # Accumulate batch-averaged metrics (no weighting)
            for key in batch_metrics:
                if key in loss_dict:
                    batch_metrics[key] += loss_dict[key]
            
            n_samples += batch_size
            n_batches += 1
    
    # Average: sample-weighted / n_samples, batch-averaged / n_batches
    result = {k: v / max(n_samples, 1) for k, v in sample_metrics.items()}
    result['load_satisfy_pct'] = batch_metrics['load_satisfy_pct'] / max(n_batches, 1)
    result['hard_constraint_vio'] = result['gen_vio'] + result['branch_pf_vio'] + result['branch_ang_vio']
    return result


# ============================================================================
# Model Loading/Creation Utilities
# ============================================================================

def load_pretrained_vae(config, input_dim, output_dim, is_vm, device):
    """Load pretrained VAE model as anchor generator."""
    model_name = "Vm" if is_vm else "Va"
    vae_path = config.pretrain_model_path_vm if is_vm else config.pretrain_model_path_va
    
    if not os.path.exists(vae_path):
        raise FileNotFoundError(f"Pretrained VAE not found: {vae_path}")
    
    print(f"[{model_name}] Loading VAE from: {vae_path}")
    
    vae_model = create_model('vae', input_dim, output_dim, config, is_vm=is_vm)
    vae_model.to(device)
    vae_model.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
    vae_model.eval()
    
    for param in vae_model.parameters():
        param.requires_grad = False
    
    return vae_model


def create_flow_model(config, input_dim, output_dim, is_vm, device, 
                      model_type='rectified', pretrained_path=None):
    """Create or load a flow model."""
    model_name = "Vm" if is_vm else "Va"
    
    flow_model = create_model(model_type, input_dim, output_dim, config, is_vm=is_vm)
    flow_model.to(device)
    
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"[{model_name}] Loading pretrained flow from: {pretrained_path}")
        flow_model.load_state_dict(
            torch.load(pretrained_path, map_location=device, weights_only=True), 
            strict=False
        )
    else:
        print(f"[{model_name}] Initializing fresh {model_type} model")
    
    return flow_model


def save_checkpoint(config, model_vm, model_va, suffix, prefix='pareto'):
    """Save model checkpoint."""
    vm_path = os.path.join(config.model_save_dir, f'model_vm_{prefix}_{suffix}.pth')
    va_path = os.path.join(config.model_save_dir, f'model_va_{prefix}_{suffix}.pth')
    
    torch.save(model_vm.state_dict(), vm_path, _use_new_zipfile_serialization=False)
    torch.save(model_va.state_dict(), va_path, _use_new_zipfile_serialization=False)
    
    print(f"  Saved: {vm_path}")


# ============================================================================
# Main Training Functions (Simplified)
# ============================================================================

def train_pareto_flow(
    config, flow_vm, flow_va, vae_vm, vae_va,
    loss_fn, train_loader, sys_data, device,
    lambda_cost=0.9, lambda_carbon=0.1,
    epochs=500, lr=1e-4, inf_steps=10,
    use_projection=False, zero_init=True,
    use_adaptive_weights=False,
    test_loader=None, early_stopping_patience=20, val_freq=10,
    tb_logger=None
):
    """
    Train flow models for a FIXED preference.
    
    Simplified version using strategy pattern and generic components.
    """
    print("=" * 60)
    print(f"Training Pareto Flow: [{lambda_cost}, {lambda_carbon}]")
    print(f"Epochs: {epochs}, LR: {lr}, Steps: {inf_steps}")
    print(f"Projection: {use_projection}, Zero-init: {zero_init}")
    print("=" * 60)
    
    # Initialize near zero
    if zero_init:
        initialize_flow_model_near_zero(flow_vm, scale=0.01)
        initialize_flow_model_near_zero(flow_va, scale=0.01)
    
    # Setup strategies
    anchor_strategy = AnchorStrategy(vae_vm, vae_va)
    pref_strategy = PreferenceStrategy(mode='fixed', fixed_value=(lambda_cost, lambda_carbon))
    
    # Setup optimizers
    opt_vm = torch.optim.Adam(flow_vm.parameters(), lr=lr, weight_decay=1e-6)
    opt_va = torch.optim.Adam(flow_va.parameters(), lr=lr, weight_decay=1e-6)
    sched_vm = torch.optim.lr_scheduler.StepLR(opt_vm, step_size=200, gamma=0.5)
    sched_va = torch.optim.lr_scheduler.StepLR(opt_va, step_size=200, gamma=0.5)
    
    # Prepare data tensors
    baseMVA = float(sys_data.baseMVA.item() if hasattr(sys_data.baseMVA, 'item') else sys_data.baseMVA)
    Pd_train = torch.from_numpy(sys_data.RPd[:config.Ntrain] / baseMVA).float().to(device)
    Qd_train = torch.from_numpy(sys_data.RQd[:config.Ntrain] / baseMVA).float().to(device)
    Pd_test = torch.from_numpy(sys_data.RPd[config.Ntrain:] / baseMVA).float().to(device)
    Qd_test = torch.from_numpy(sys_data.RQd[config.Ntrain:] / baseMVA).float().to(device)
    
    # Setup projection if enabled
    P_tan_t = None
    if use_projection:
        print("Setting up constraint projection...")
        projector = ConstraintProjectionV2(sys_data, config, use_historical_jacobian=True)
        P_tan, _, _ = projector.compute_projection_matrix()
        P_tan_t = torch.tensor(P_tan, dtype=torch.float32, device=device)
    
    # Evaluate VAE baseline
    print("\n--- VAE Baseline ---")
    vae_baseline = evaluate_vae_baseline(
        vae_vm, vae_va, loss_fn, train_loader, Pd_train, Qd_train,
        config, device, lambda_cost, lambda_carbon
    )
    satisfaction_str = f", Satisfaction: {vae_baseline['satisfaction']:.2f}" if 'satisfaction' in vae_baseline else ""
    print(f"Cost: {vae_baseline['cost']:.2f}, Carbon: {vae_baseline['carbon']:.4f}{satisfaction_str}")
    print(f"Hard Constraints: {vae_baseline['hard_constraint_vio']:.6f}, Load Satisfy: {vae_baseline['load_satisfy_pct']:.2f}%")
    
    # Training state
    loss_history = {'total': [], 'cost': [], 'carbon': [], 'load_dev': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_state = None
    
    start_time = time.process_time()
    
    for epoch in range(epochs):
        # Train one epoch
        metrics, n_batches = train_one_epoch(
            flow_vm, flow_va, (opt_vm, opt_va),
            anchor_strategy, pref_strategy,
            loss_fn, train_loader, Pd_train, Qd_train, config, device,
            is_unified=False, inf_steps=inf_steps,
            use_projection=use_projection, P_tan_t=P_tan_t,
            use_adaptive_weights=use_adaptive_weights
        )
        
        sched_vm.step()
        sched_va.step()
        
        # Record history
        loss_history['total'].append(metrics.get('loss', 0))
        loss_history['cost'].append(metrics.get('cost', 0))
        loss_history['carbon'].append(metrics.get('carbon', 0))
        loss_history['load_dev'].append(metrics.get('load_dev', 0))
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            hard_vio = metrics['gen_vio'] + metrics['branch_pf_vio'] + metrics['branch_ang_vio']
            satisfaction = metrics.get('satisfaction', 0.0)
            print(f"Epoch {epoch+1}/{epochs}: Loss={metrics['loss']:.4f}, "
                  f"Cost={metrics['cost']:.2f}, Carbon={metrics['carbon']:.4f}, "
                  f"Satisfaction={satisfaction:.2f}, "
                  f"HardConstr={hard_vio:.6f}, LoadSat={metrics['load_satisfy_pct']:.2f}%")
        
        # TensorBoard logging
        if tb_logger and n_batches > 0:
            hard_vio = metrics['gen_vio'] + metrics['branch_pf_vio'] + metrics['branch_ang_vio']
            tb_logger.log_scalar('cost/flow_train', metrics['cost'], epoch)
            tb_logger.log_scalar('carbon/flow_train', metrics['carbon'], epoch)
            tb_logger.log_scalar('satisfaction/flow_train', metrics.get('satisfaction', 0.0), epoch)
            tb_logger.log_scalar('load_satisfy_pct/flow_train', metrics['load_satisfy_pct'], epoch)
            tb_logger.log_scalar('hard_constraint/flow_train', hard_vio, epoch)
        
        # Validation
        if test_loader and (epoch + 1) % val_freq == 0:
            val_metrics = validate_model(
                flow_vm, flow_va, vae_vm, vae_va, loss_fn,
                test_loader, Pd_test, Qd_test, config, device,
                is_unified=False, inf_steps=inf_steps, pref_strategy=pref_strategy,
                use_projection=use_projection, P_tan_t=P_tan_t
            )
            
            print(f"  [Val] HardConstr={val_metrics['hard_constraint_vio']:.6f}, "
                  f"LoadSat={val_metrics['load_satisfy_pct']:.2f}%")
            
            if tb_logger:
                tb_logger.log_scalar('load_satisfy_pct/flow_val', val_metrics['load_satisfy_pct'], epoch)
                tb_logger.log_scalar('hard_constraint/flow_val', val_metrics['hard_constraint_vio'], epoch)
            
            # Early stopping
            val_metric = val_metrics['load_dev']
            if val_metric < best_val_loss:
                best_val_loss = val_metric
                best_epoch = epoch + 1
                patience_counter = 0
                best_state = {
                    'vm': {k: v.cpu().clone() for k, v in flow_vm.state_dict().items()},
                    'va': {k: v.cpu().clone() for k, v in flow_va.state_dict().items()}
                }
                print(f"  [Val] New best! LoadDev={val_metric:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\n[Early Stop] Best epoch: {best_epoch}")
                    break
        
        # GPU cleanup
        if (epoch + 1) % 10 == 0:
            gpu_memory_cleanup()
        
        # Periodic checkpoint
        if (epoch + 1) % 100 == 0:
            pref_str = f"{lambda_cost}_{lambda_carbon}".replace(".", "")
            save_checkpoint(config, flow_vm, flow_va, f"{pref_str}_E{epoch+1}")
    
    train_time = time.process_time() - start_time
    print(f"\nTraining completed in {train_time:.1f}s ({train_time/60:.1f}min)")
    
    # Restore best model
    if best_state:
        flow_vm.load_state_dict({k: v.to(device) for k, v in best_state['vm'].items()})
        flow_va.load_state_dict({k: v.to(device) for k, v in best_state['va'].items()})
        print(f"Restored best model from epoch {best_epoch}")
    
    # Save final
    pref_str = f"{lambda_cost}_{lambda_carbon}".replace(".", "")
    save_checkpoint(config, flow_vm, flow_va, f"{pref_str}_final")
    
    return flow_vm, flow_va, loss_history


def train_unified_pareto_flow(
    config, unified_vm, unified_va, vae_vm, vae_va,
    loss_fn, train_loader, sys_data, device,
    preference_schedule=None,
    epochs=500, lr=1e-4, inf_steps=10,
    zero_init=True, use_adaptive_weights=False,
    pref_sampling='curriculum', fixed_preference=None,
    test_loader=None, early_stopping_patience=20, val_freq=10,
    tb_logger=None, use_pareto_validation=False,
    use_projection=False
):
    """
    Train UNIFIED preference-conditioned flow models.
    
    One model handles ALL preferences via conditioning.
    """
    print("=" * 70)
    print("UNIFIED PREFERENCE-CONDITIONED FLOW TRAINING")
    print(f"Sampling: {pref_sampling}, Epochs: {epochs}, LR: {lr}")
    print("=" * 70)
    
    # Initialize
    if zero_init:
        initialize_flow_model_near_zero(unified_vm, scale=0.01)
        initialize_flow_model_near_zero(unified_va, scale=0.01)
    
    # Setup strategies
    anchor_strategy = AnchorStrategy(vae_vm, vae_va)
    current_max_carbon = 0.1
    pref_strategy = PreferenceStrategy(
        mode=pref_sampling,
        fixed_value=fixed_preference or (0.9, 0.1),
        max_carbon=current_max_carbon
    )
    
    # Optimizers
    opt_vm = torch.optim.Adam(unified_vm.parameters(), lr=lr, weight_decay=1e-6)
    opt_va = torch.optim.Adam(unified_va.parameters(), lr=lr, weight_decay=1e-6)
    sched_vm = torch.optim.lr_scheduler.StepLR(opt_vm, step_size=200, gamma=0.5)
    sched_va = torch.optim.lr_scheduler.StepLR(opt_va, step_size=200, gamma=0.5)
    
    # Data
    baseMVA = float(sys_data.baseMVA.item() if hasattr(sys_data.baseMVA, 'item') else sys_data.baseMVA)
    Pd_train = torch.from_numpy(sys_data.RPd[:config.Ntrain] / baseMVA).float().to(device)
    Qd_train = torch.from_numpy(sys_data.RQd[:config.Ntrain] / baseMVA).float().to(device)
    Pd_test = torch.from_numpy(sys_data.RPd[config.Ntrain:] / baseMVA).float().to(device)
    Qd_test = torch.from_numpy(sys_data.RQd[config.Ntrain:] / baseMVA).float().to(device)
    
    # Projection
    P_tan_t = None
    if use_projection:
        projector = ConstraintProjectionV2(sys_data, config, use_historical_jacobian=True)
        P_tan, _, _ = projector.compute_projection_matrix()
        P_tan_t = torch.tensor(P_tan, dtype=torch.float32, device=device)
    
    # Training state
    loss_history = {'total': [], 'cost': [], 'carbon': [], 'load_dev': [], 'val_loss': []}
    best_val_metric = float('-inf') if use_pareto_validation else float('inf')
    best_epoch = 0
    patience_counter = 0
    best_state = None
    current_stage = 0
    
    start_time = time.process_time()
    
    for epoch in range(epochs):
        # Update curriculum stage
        if pref_sampling == 'curriculum' and preference_schedule:
            stage_epochs = epochs // len(preference_schedule)
            new_stage = min(epoch // stage_epochs, len(preference_schedule) - 1)
            if new_stage != current_stage:
                current_stage = new_stage
                _, current_max_carbon = preference_schedule[current_stage]
                pref_strategy.max_carbon = current_max_carbon
                print(f"\n[Curriculum] Stage {current_stage+1}: max_carbon={current_max_carbon:.2f}")
        
        # Train
        metrics, n_batches = train_one_epoch(
            unified_vm, unified_va, (opt_vm, opt_va),
            anchor_strategy, pref_strategy,
            loss_fn, train_loader, Pd_train, Qd_train, config, device,
            is_unified=True, inf_steps=inf_steps,
            use_projection=use_projection, P_tan_t=P_tan_t,
            use_adaptive_weights=use_adaptive_weights
        )
        
        sched_vm.step()
        sched_va.step()
        
        # Record history
        loss_history['total'].append(metrics.get('loss', 0))
        loss_history['cost'].append(metrics.get('cost', 0))
        loss_history['carbon'].append(metrics.get('carbon', 0))
        loss_history['load_dev'].append(metrics.get('load_dev', 0))
        
        # Print
        if (epoch + 1) % 10 == 0:
            hard_vio = metrics['gen_vio'] + metrics['branch_pf_vio'] + metrics['branch_ang_vio']
            satisfaction = metrics.get('satisfaction', 0.0)
            print(f"Epoch {epoch+1}/{epochs}: Loss={metrics['loss']:.4f}, "
                  f"Cost={metrics['cost']:.2f}, Carbon={metrics['carbon']:.4f}, "
                  f"Satisfaction={satisfaction:.2f}, "
                  f"HardConstr={hard_vio:.6f}, LoadSat={metrics['load_satisfy_pct']:.2f}%")
        
        # TensorBoard
        if tb_logger and n_batches > 0:
            hard_vio = metrics['gen_vio'] + metrics['branch_pf_vio'] + metrics['branch_ang_vio']
            tb_logger.log_scalar('cost/flow_train', metrics['cost'], epoch)
            tb_logger.log_scalar('carbon/flow_train', metrics['carbon'], epoch)
            tb_logger.log_scalar('satisfaction/flow_train', metrics.get('satisfaction', 0.0), epoch)
            tb_logger.log_scalar('load_satisfy_pct/flow_train', metrics['load_satisfy_pct'], epoch)
            tb_logger.log_scalar('hard_constraint/flow_train', hard_vio, epoch)
        
        # Validation
        if test_loader and (epoch + 1) % val_freq == 0:
            if use_pareto_validation:
                val_result = validate_pareto_front_unified(
                    unified_vm, unified_va, vae_vm, vae_va,
                    loss_fn, test_loader, Pd_test, Qd_test, config, device,
                    inf_steps=inf_steps, use_projection=use_projection, P_tan_t=P_tan_t
                )
                val_metric = val_result['validation_metric']
                print(f"  [Val] HV={val_result['hypervolume']:.2f}, "
                      f"Feasible={val_result['feasible_ratio']:.1%}")
                
                # Early stopping (higher metric is better)
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_epoch = epoch + 1
                    patience_counter = 0
                    best_state = {
                        'vm': {k: v.cpu().clone() for k, v in unified_vm.state_dict().items()},
                        'va': {k: v.cpu().clone() for k, v in unified_va.state_dict().items()}
                    }
                    print(f"  [Val] New best! Metric={val_metric:.4f}")
                else:
                    patience_counter += 1
            else:
                val_metrics = validate_model(
                    unified_vm, unified_va, vae_vm, vae_va, loss_fn,
                    test_loader, Pd_test, Qd_test, config, device,
                    is_unified=True, inf_steps=inf_steps, pref_strategy=pref_strategy,
                    use_projection=use_projection, P_tan_t=P_tan_t
                )
                val_metric = val_metrics['load_dev']
                print(f"  [Val] HardConstr={val_metrics['hard_constraint_vio']:.6f}, "
                      f"LoadSat={val_metrics['load_satisfy_pct']:.2f}%")
                
                if val_metric < best_val_metric:
                    best_val_metric = val_metric
                    best_epoch = epoch + 1
                    patience_counter = 0
                    best_state = {
                        'vm': {k: v.cpu().clone() for k, v in unified_vm.state_dict().items()},
                        'va': {k: v.cpu().clone() for k, v in unified_va.state_dict().items()}
                    }
                    print(f"  [Val] New best! LoadDev={val_metric:.4f}")
                else:
                    patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"\n[Early Stop] Best epoch: {best_epoch}")
                break
        
        # GPU management
        if (epoch + 1) % 10 == 0:
            check_gpu_temperature(warning_temp=80, critical_temp=85, cooldown_time=30)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    train_time = time.process_time() - start_time
    print(f"\nTraining completed in {train_time:.1f}s ({train_time/60:.1f}min)")
    
    # Restore best
    if best_state:
        unified_vm.load_state_dict({k: v.to(device) for k, v in best_state['vm'].items()})
        unified_va.load_state_dict({k: v.to(device) for k, v in best_state['va'].items()})
        print(f"Restored best model from epoch {best_epoch}")
    
    save_checkpoint(config, unified_vm, unified_va, f"unified_{pref_sampling}_final", prefix='unified')
    
    return unified_vm, unified_va, loss_history


def validate_pareto_front_unified(
    unified_vm, unified_va, vae_vm, vae_va,
    loss_fn, test_loader, Pd_test, Qd_test, config, device,
    inf_steps=10, preference_points=None,
    feasibility_thresholds=None, ref_point=None,
    max_samples=500, use_projection=False, P_tan_t=None
):
    """Validate unified model by evaluating Pareto front quality."""
    unified_vm.eval()
    unified_va.eval()
    
    if preference_points is None:
        preference_points = [(1.0 - i * 0.1, i * 0.1) for i in range(10)]
    
    if feasibility_thresholds is None:
        feasibility_thresholds = {'load_dev': 0.02, 'gen_vio': 0.005}
    
    all_costs, all_carbons, all_load_devs, all_gen_vios, all_feasible = [], [], [], [], []
    
    with torch.no_grad():
        for lc, le in preference_points:
            pref_costs, pref_carbons, pref_load_devs, pref_gen_vios = [], [], [], []
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
                prefs = torch.tensor([[lc, le]], device=device).expand(batch_size, -1)
                
                z_vm = vae_vm(test_x, use_mean=True)
                z_va = vae_va(test_x, use_mean=True)
                
                if use_projection and P_tan_t is not None:
                    y_vm, y_va = flow_forward_projected(
                        unified_vm, unified_va, test_x, z_vm, z_va,
                        P_tan_t, config.Nbus, inf_steps, prefs
                    )
                else:
                    y_vm = flow_forward_unified(unified_vm, test_x, z_vm, prefs, inf_steps)
                    y_va = flow_forward_unified(unified_va, test_x, z_va, prefs, inf_steps)
                
                _, loss_dict = loss_fn(
                    y_vm, y_va, Pd_batch, Qd_batch,
                    lambda_cost=lc, lambda_carbon=le,
                    return_details=True
                )
                
                pref_costs.append(loss_dict['cost'])
                pref_carbons.append(loss_dict['carbon'])
                pref_load_devs.append(loss_dict['load_dev'])
                pref_gen_vios.append(loss_dict.get('gen_vio', 0.0))
                n_samples += batch_size
            
            if pref_costs:
                avg_cost = np.mean(pref_costs)
                avg_carbon = np.mean(pref_carbons)
                avg_load_dev = np.mean(pref_load_devs)
                avg_gen_vio = np.mean(pref_gen_vios)
                
                all_costs.append(avg_cost)
                all_carbons.append(avg_carbon)
                all_load_devs.append(avg_load_dev)
                all_gen_vios.append(avg_gen_vio)
                
                is_feasible = check_feasibility(
                    {'load_dev': avg_load_dev, 'gen_vio': avg_gen_vio},
                    feasibility_thresholds
                )
                all_feasible.append(is_feasible)
    
    unified_vm.train()
    unified_va.train()
    
    all_costs = np.array(all_costs)
    all_carbons = np.array(all_carbons)
    all_feasible = np.array(all_feasible)
    
    if ref_point is None:
        ref_point = np.array([
            np.max(all_costs) * 1.1 if len(all_costs) > 0 else 1e6,
            np.max(all_carbons) * 1.1 if len(all_carbons) > 0 else 1e6
        ])
    
    pareto_result = evaluate_pareto_front(all_costs, all_carbons, all_feasible, ref_point)
    validation_metric = get_pareto_validation_metric(
        pareto_result['hypervolume'],
        pareto_result['feasible_ratio'],
        hv_weight=0.6, feas_weight=0.4,
        hv_scale=ref_point[0] * ref_point[1]
    )
    
    return {
        'hypervolume': pareto_result['hypervolume'],
        'feasible_ratio': pareto_result['feasible_ratio'],
        'mean_load_dev': np.mean(all_load_devs) if all_load_devs else 0.0,
        'mean_cost': np.mean(all_costs) if len(all_costs) > 0 else 0.0,
        'mean_carbon': np.mean(all_carbons) if len(all_carbons) > 0 else 0.0,
        'validation_metric': validation_metric,
        'ref_point': ref_point
    }


# ============================================================================
# Visualization & Utilities
# ============================================================================

def generate_preference_schedule(start_cost=1.0, end_cost=0.1, step=0.1):
    """Generate preference schedule for curriculum learning."""
    schedule = []
    current = start_cost - step
    while current >= end_cost - 1e-9:
        lc = round(current, 2)
        le = round(1.0 - lc, 2)
        schedule.append((lc, le))
        current -= step
    return schedule


def plot_training_curves(loss_history, lambda_cost, lambda_carbon, save_path=None):
    """Plot training loss curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Training: lambda_cost={lambda_cost}, lambda_carbon={lambda_carbon}')
    
    axes[0, 0].plot(loss_history.get('total', []))
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(loss_history.get('cost', []))
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('$/h')
    axes[0, 1].set_title('Cost')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(loss_history.get('carbon', []))
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('tCO2/h')
    axes[1, 0].set_title('Carbon')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(loss_history.get('load_dev', []))
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Deviation')
    axes[1, 1].set_title('Load Deviation')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Pareto-Adaptive Rectified Flow')
    
    # GPU and hardware settings
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device id to use (e.g., 0, 1). None for auto-select or CPU')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Training batch size (overrides config)')
    parser.add_argument('--batch_size_test', type=int, default=None,
                        help='Test batch size (overrides config, defaults to batch_size if not set)')
    
    # Model mode
    parser.add_argument('--model_mode', type=str, default='unified',
                        choices=['independent', 'unified'],
                        help='independent: per-preference models, unified: single conditioned model')
    
    # Preference settings
    parser.add_argument('--lambda_cost', type=float, default=0.9)
    parser.add_argument('--lambda_carbon', type=float, default=0.1)
    parser.add_argument('--pref_sampling', type=str, default='curriculum',
                        choices=['uniform', 'curriculum', 'fixed'])
    
    # Curriculum settings
    parser.add_argument('--start_cost', type=float, default=1.0)
    parser.add_argument('--end_cost', type=float, default=0.1)
    parser.add_argument('--pref_step', type=float, default=0.1)
    
    # Training
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--inf_steps', type=int, default=10)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--val_freq', type=int, default=10)
    
    # Options
    parser.add_argument('--use_projection', action='store_true', default=False)
    parser.add_argument('--no_zero_init', action='store_true')
    parser.add_argument('--pareto_validation', action='store_true', default=True)
    
    # Loss weight settings (based on DeepOPF-NGT paper)
    # Initial weights: k_0g = k_0Sl = k_0theta = k_0d = 1
    # Note: k_obj needs careful tuning - too small causes constraint collapse
    #       Recommended: 0.01 for 300-bus system (cost ~500k, constraints ~0-50)
    parser.add_argument('--adaptive_weights', action='store_true', default=True,
                        help='Enable adaptive weight scheduling (DeepOPF-NGT)')
    parser.add_argument('--k_obj', type=float, default=1,
                        help='Weight for objective function (0.01 recommended for 300-bus)')
    parser.add_argument('--k_g', type=float, default=1.0,
                        help='Initial weight for generator constraint')
    parser.add_argument('--k_Sl', type=float, default=1.0,
                        help='Initial weight for branch power flow')
    parser.add_argument('--k_theta', type=float, default=1.0,
                        help='Initial weight for branch angle')
    parser.add_argument('--k_d', type=float, default=1.0,
                        help='Initial weight for load deviation')
    
    # Weight upper bounds (to prevent gradient explosion)
    # Set higher limits to give adaptive scheduler more room
    parser.add_argument('--k_g_max', type=float, default=10000.0,
                        help='Upper bound for generator constraint weight')
    parser.add_argument('--k_Sl_max', type=float, default=10000.0,
                        help='Upper bound for branch power flow weight')
    parser.add_argument('--k_theta_max', type=float, default=10000.0,
                        help='Upper bound for branch angle weight')
    parser.add_argument('--k_d_max', type=float, default=100000.0,
                        help='Upper bound for load deviation weight (higher for strict enforcement)')
    
    # Carbon scale factor
    parser.add_argument('--carbon_scale', type=float, default=30.0,
                        help='Scale factor for carbon emission (cost/carbon ratio)')
    
    args = parser.parse_args() 
    
    # Load config and data
    config = get_config()
    
    # Override config with args (args takes priority)
    config.k_obj = args.k_obj
    config.k_g = args.k_g
    config.k_Sl = args.k_Sl
    config.k_theta = args.k_theta
    config.k_d = args.k_d
    config.k_g_max = args.k_g_max
    config.k_Sl_max = args.k_Sl_max
    config.k_theta_max = args.k_theta_max
    config.k_d_max = args.k_d_max
    config.use_adaptive_weights = args.adaptive_weights
    config.carbon_scale = args.carbon_scale
    
    # Override batch size if provided
    if args.batch_size is not None:
        config.batch_size_training = args.batch_size
        print(f"[Args] Training batch size: {args.batch_size}")
    if args.batch_size_test is not None:
        config.batch_size_test = args.batch_size_test
        print(f"[Args] Test batch size: {args.batch_size_test}")
    elif args.batch_size is not None:
        # If only training batch size is set, use it for test as well
        config.batch_size_test = args.batch_size
    
    # Set GPU device
    if args.gpu is not None:
        if torch.cuda.is_available() and args.gpu < torch.cuda.device_count():
            device = torch.device(f'cuda:{args.gpu}')
            torch.cuda.set_device(args.gpu)
            print(f"[Args] Using GPU: {args.gpu} ({torch.cuda.get_device_name(args.gpu)})")
        else:
            print(f"[Warning] GPU {args.gpu} not available, falling back to config device")
            device = config.device
    else:
        device = config.device
    
    # Update config device for consistency
    config.device = device
    
    config.print_config()
    print(f"Batch size (train/test): {config.batch_size_training}/{config.batch_size_test}")
    
    print(f"\nDevice: {device}")
    print("\nLoading data...")
    sys_data, dataloaders, _ = load_all_data(config)
    
    input_dim = sys_data.x_train.shape[1]
    output_dim_vm = sys_data.yvm_train.shape[1]
    output_dim_va = sys_data.yva_train.shape[1]
    
    # Load VAE anchors
    print("\n--- Loading VAE Anchors ---")
    vae_vm = load_pretrained_vae(config, input_dim, output_dim_vm, True, device)
    vae_va = load_pretrained_vae(config, input_dim, output_dim_va, False, device)
    
    # GCI values
    gci_values = get_gci_for_generators(config, sys_data)
    
    # TensorBoard
    tb_logger = create_tensorboard_logger(
        config, model_mode=args.model_mode,
        pref_sampling=args.pref_sampling,
        lambda_cost=args.lambda_cost, lambda_carbon=args.lambda_carbon
    )
    
    # Loss function (uses config values which are overridden by args)
    loss_fn = MultiObjectiveOPFLoss(
        sys_data, config, gci_values, use_adaptive_weights=args.adaptive_weights
    ).to(device)
    
    # Preference schedule
    preference_schedule = generate_preference_schedule(
        args.start_cost, args.end_cost, args.pref_step
    )
    
    test_loader = dataloaders.get('test_vm', None)
    
    if args.model_mode == 'unified':
        print("\n--- Creating Unified Models ---")
        unified_vm = create_flow_model(config, input_dim, output_dim_vm, True, device, 'preference_flow')
        unified_va = create_flow_model(config, input_dim, output_dim_va, False, device, 'preference_flow')
        
        unified_vm, unified_va, history = train_unified_pareto_flow(
            config, unified_vm, unified_va, vae_vm, vae_va,
            loss_fn, dataloaders['train_vm'], sys_data, device,
            preference_schedule=preference_schedule if args.pref_sampling == 'curriculum' else None,
            epochs=args.epochs, lr=args.lr, inf_steps=args.inf_steps,
            zero_init=not args.no_zero_init,
            use_adaptive_weights=args.adaptive_weights,
            pref_sampling=args.pref_sampling,
            fixed_preference=(args.lambda_cost, args.lambda_carbon),
            test_loader=test_loader,
            early_stopping_patience=args.patience,
            val_freq=args.val_freq,
            tb_logger=tb_logger,
            use_pareto_validation=args.pareto_validation,
            use_projection=args.use_projection
        )
        
        plot_path = os.path.join(config.results_dir, f'unified_{args.pref_sampling}_training.png')
        plot_training_curves(history, 0.5, 0.5, plot_path)
        
    else:  # independent mode
        print("\n--- Creating Independent Models ---")
        flow_vm = create_flow_model(config, input_dim, output_dim_vm, True, device)
        flow_va = create_flow_model(config, input_dim, output_dim_va, False, device)
        
        flow_vm, flow_va, history = train_pareto_flow(
            config, flow_vm, flow_va, vae_vm, vae_va,
            loss_fn, dataloaders['train_vm'], sys_data, device,
            lambda_cost=args.lambda_cost, lambda_carbon=args.lambda_carbon,
            epochs=args.epochs, lr=args.lr, inf_steps=args.inf_steps,
            use_projection=args.use_projection,
            zero_init=not args.no_zero_init,
            use_adaptive_weights=args.adaptive_weights,
            test_loader=test_loader,
            early_stopping_patience=args.patience,
            val_freq=args.val_freq,
            tb_logger=tb_logger
        )
        
        pref_str = f"{args.lambda_cost}_{args.lambda_carbon}".replace(".", "")
        plot_path = os.path.join(config.results_dir, f'pareto_{pref_str}_training.png')
        plot_training_curves(history, args.lambda_cost, args.lambda_carbon, plot_path)
    
    if tb_logger:
        tb_logger.close()
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
