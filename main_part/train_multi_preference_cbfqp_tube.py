#!/usr/bin/env python
# coding: utf-8
"""
Multi-Preference Supervised Training for DeepOPF-V
Trains preference-conditioned models for multi-objective OPF.

Supports: simple, vae, rectified, diffusion

Author: Peng Yue
Date: December 2025

Usage:
    MODEL_TYPE=rectified python train_multi_preference.py
    MODEL_TYPE=vae LOAD_PRETRAINED_MODEL=1 python train_multi_preference.py
"""

import torch
import torch.nn as nn
import time
import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BaseConfig, _SCRIPT_DIR
from models import NetV
from data_loader import load_multi_preference_dataset, create_multi_preference_dataloader

# [CBF-QP TRAIN] Projection layer (training-time)
from cbf_qp_train_layer_tube import CBFQPTrainConfig, CBFQPProjectorNGT  # [TUBE]


# ==================== Multi-Preference Configuration ====================

class MultiPreferenceConfig(BaseConfig):
    """Configuration for multi-preference supervised training."""
    
    def __init__(self):
        super().__init__()
        
        # ==================== Multi-Preference Training ==================== 
        self.multi_pref_dataset_path = os.path.join(
            os.path.dirname(_SCRIPT_DIR), 'saved_data', 'multi_preference_solutions', 'fully_covered_dataset_2026-01-02.pt'
        )
        
        # Training parameters
        self.multi_pref_epochs = int(os.environ.get('MULTI_PREF_EPOCHS', '1000'))
        self.multi_pref_lr = float(os.environ.get('MULTI_PREF_LR', '1e-4'))
        self.multi_pref_flow_steps = int(os.environ.get('MULTI_PREF_FLOW_STEPS', '10'))
        self.multi_pref_batch_size = int(os.environ.get('MULTI_PREF_BATCH_SIZE', '50'))
        
        # Validation split
        self.multi_pref_val_ratio = float(os.environ.get('MULTI_PREF_VAL_RATIO', '0.2'))
        self.multi_pref_random_seed = int(os.environ.get('MULTI_PREF_RANDOM_SEED', '42'))
        
        # ==================== Unified Model Architecture ====================
        # These parameters are shared across VAE, Flow, Diffusion models
        # Chosen to keep model sizes roughly consistent (~100-200K params)
        # Simple: ~95K, Flow: ~183K, Diffusion: ~141K, VAE: ~390K (has Encoder+Decoder)
        self.hidden_dim = int(os.environ.get('HIDDEN_DIM', '128'))
        self.num_layers = int(os.environ.get('NUM_LAYERS', '2'))
        self.latent_dim = int(os.environ.get('LATENT_DIM', '64'))  # For VAE
        self.time_step = 1000  # For Flow/Diffusion ODE solver
        
        # Simple model (NetV) uses a special structure
        self.ngt_hidden_units = 1
        self.ngt_khidden = np.array([64, 224], dtype=int)
        
        # Training mode and flow type
        self.multi_pref_flow_type = self.model_type
        self.multi_pref_training_mode = os.environ.get('MULTI_PREF_TRAINING_MODE', 'preference_trajectory')  # 'standard', 'preference_trajectory'
        
        # Loss weights
        self.multi_pref_loss_alpha = float(os.environ.get('MULTI_PREF_LOSS_ALPHA', '1.0'))
        self.multi_pref_loss_beta = float(os.environ.get('MULTI_PREF_LOSS_BETA', '1000.0'))
        
        # Multi-step rollout
        self.multi_pref_rollout_use_rk2 = os.environ.get('MULTI_PREF_ROLLOUT_USE_RK2', 'True').lower() == 'true'

        # ==================== [CBF-QP TRAIN] Training-time safety projection ====================
        # Enable by setting env:
        #   MULTI_PREF_USE_CBF_QP_TRAIN=1
        # You can tune these without changing code.

        self.multi_pref_use_cbf_qp_train = os.environ.get('MULTI_PREF_USE_CBF_QP_TRAIN', '0').lower() in ['1', 'true', 'yes']
        self.multi_pref_cbf_beta = float(os.environ.get('MULTI_PREF_CBF_BETA', '0.5'))
        self.multi_pref_cbf_apply_prob = float(os.environ.get('MULTI_PREF_CBF_APPLY_PROB', '1.0'))

        # Trust region (per variable type)
        self.multi_pref_cbf_trust_va = float(os.environ.get('MULTI_PREF_CBF_TRUST_VA', '0.10'))   # radians
        self.multi_pref_cbf_trust_vm = float(os.environ.get('MULTI_PREF_CBF_TRUST_VM', '0.02'))   # p.u.

        # Constraint selection (near-bound / violated)
        self.multi_pref_cbf_eps_vm = float(os.environ.get('MULTI_PREF_CBF_EPS_VM', '0.02'))
        self.multi_pref_cbf_eps_pqg = float(os.environ.get('MULTI_PREF_CBF_EPS_PQG', '0.02'))
        self.multi_pref_cbf_eps_branch = float(os.environ.get('MULTI_PREF_CBF_EPS_BRANCH', '0.02'))
        self.multi_pref_cbf_k_vm = int(os.environ.get('MULTI_PREF_CBF_K_VM', '64'))
        self.multi_pref_cbf_k_pqg = int(os.environ.get('MULTI_PREF_CBF_K_PQG', '64'))
        self.multi_pref_cbf_k_branch = int(os.environ.get('MULTI_PREF_CBF_K_BRANCH', '32'))

        # Solver knobs
        self.multi_pref_cbf_max_iters = int(os.environ.get('MULTI_PREF_CBF_MAX_ITERS', '6'))
        self.multi_pref_cbf_detach_active_set = os.environ.get('MULTI_PREF_CBF_DETACH_ACTIVE_SET', '1').lower() in ['1', 'true', 'yes']
        self.multi_pref_cbf_penalty_rho = float(os.environ.get('MULTI_PREF_CBF_PENALTY_RHO', '1e7'))

        # Optional: distillation (encourage v_pred ≈ v_safe, reduce projection trigger at inference)
        self.multi_pref_cbf_distill_weight = float(os.environ.get('MULTI_PREF_CBF_DISTILL_WEIGHT', '0.0'))
        

        # ==================== [TUBE] Soft safety tube (bridge-friendly) ====================
        # Relax constraints during training: A*delta <= b + eps_tube
        # Separate eps for Vm / Pg,Qg / Branch (schedule start->end).
        self.multi_pref_tube_eps_vm_start = float(os.environ.get('MULTI_PREF_TUBE_EPS_VM_START', '0.00'))
        self.multi_pref_tube_eps_vm_end   = float(os.environ.get('MULTI_PREF_TUBE_EPS_VM_END',   '0.00'))
        self.multi_pref_tube_eps_pqg_start = float(os.environ.get('MULTI_PREF_TUBE_EPS_PQG_START', '0.00'))
        self.multi_pref_tube_eps_pqg_end   = float(os.environ.get('MULTI_PREF_TUBE_EPS_PQG_END',   '0.00'))
        self.multi_pref_tube_eps_branch_start = float(os.environ.get('MULTI_PREF_TUBE_EPS_BRANCH_START', '0.00'))
        self.multi_pref_tube_eps_branch_end   = float(os.environ.get('MULTI_PREF_TUBE_EPS_BRANCH_END',   '0.00'))
        self.multi_pref_tube_schedule = os.environ.get('MULTI_PREF_TUBE_SCHEDULE', 'linear')  # linear/cosine/exp
        self.multi_pref_tube_exp_k = float(os.environ.get('MULTI_PREF_TUBE_EXP_K', '5.0'))

        # ==================== [GATE] Skip QP solve if delta already safe ====================
        self.multi_pref_cbf_gate_before_solve = os.environ.get('MULTI_PREF_CBF_GATE', '1').lower() in ['1','true','yes']
        self.multi_pref_cbf_gate_eps = float(os.environ.get('MULTI_PREF_CBF_GATE_EPS', '1e-9'))

        # Optional: for RK2, rebuild A,b at x_euler for the 2nd stage (more accurate, slower)
        self.multi_pref_cbf_rk2_rebuild_ab = os.environ.get('MULTI_PREF_CBF_RK2_REBUILD_AB', '0').lower() in ['1','true','yes']

        # ==================== [BRIDGE] Penalize projection magnitude (shorter bridges) ====================
        # Encourage delta_exec close to delta_ref -> reduces reliance on projection over time.
        self.multi_pref_bridge_weight = float(os.environ.get('MULTI_PREF_BRIDGE_WEIGHT', '0.0'))
        # Preference conditioning
        self.pref_dim = 1
        
        # ==================== VAE Evaluation ====================
        self.vae_best_of_k = int(os.environ.get('VAE_BEST_OF_K', '32'))
        self.vae_use_mean = os.environ.get('VAE_USE_MEAN', '0').lower() in ('1', 'true', 'yes')
        self.vae_selection_mode = os.environ.get('VAE_SELECTION_MODE', 'constraint')
        self.vae_use_preference_aware = True
        self.vae_beta = 1.0
        
        # ==================== Flow Best-of-K Evaluation ====================
        self.flow_best_of_k = int(os.environ.get('FLOW_BEST_OF_K', '32'))  # K for Flow Best-of-K (1=disabled)
        self.flow_selection_mode = os.environ.get('FLOW_SELECTION_MODE', 'constraint')
        
        # ==================== Training Control ====================
        self.weight_decay = 1e-6
        self.p_epoch = 10   # Print every p_epoch epochs
        self.s_epoch = 800  # Start saving checkpoints after s_epoch
        
    def print_config(self):
        """Print configuration summary."""
        super().print_config()
        print(f"\n[Multi-Preference Training Config]")
        print(f"  Epochs: {self.multi_pref_epochs}")
        print(f"  Learning rate: {self.multi_pref_lr}")
        print(f"  Batch size: {self.multi_pref_batch_size}")
        print(f"  Training mode: {self.multi_pref_training_mode}")
        print(f"\n[Unified Model Architecture]")
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Num layers: {self.num_layers}")
        print(f"  Latent dim (VAE): {self.latent_dim}")
        print(f"  Simple khidden: {self.ngt_khidden}")
        print(f"\n[VAE Evaluation]")
        print(f"  Best-of-K: {self.vae_best_of_k} (use_mean={self.vae_use_mean})")
        print(f"\n[Flow Best-of-K Evaluation]")
        print(f"  Best-of-K: {self.flow_best_of_k} (mode={self.flow_selection_mode})")


def get_multi_preference_config():
    """Get multi-preference training configuration."""
    return MultiPreferenceConfig()


# ==================== Utility Functions ====================

def wrap_angle_difference(dx, NPred_Va):
    """Wrap angle difference to [-pi, pi] for Va dimensions."""
    if torch.is_tensor(dx):
        dx_wrapped = dx.clone()
        if NPred_Va > 0:
            dx_wrapped[..., :NPred_Va] = torch.atan2(
                torch.sin(dx[..., :NPred_Va]), 
                torch.cos(dx[..., :NPred_Va])
            )
        return dx_wrapped
    else:
        dx_np = np.asarray(dx).copy()
        if NPred_Va > 0:
            for i in range(min(NPred_Va, dx_np.shape[-1])):
                dx_np[..., i] = np.arctan2(np.sin(dx_np[..., i]), np.cos(dx_np[..., i]))
        return dx_np


def rk2_step(model, scene, x_current, lambda_current, lambda_next, NPred_Va):
    """RK2 (Heun) integration step for preference trajectory."""
    dlambda = lambda_next - lambda_current
    v0 = model.predict_vec(scene, x_current, lambda_current, lambda_current)
    x_euler = x_current + dlambda * v0
    v1 = model.predict_vec(scene, x_euler, lambda_next, lambda_next)
    return x_current + dlambda * 0.5 * (v0 + v1)


# ==================== Training Functions ====================

def train_multi_preference(config, model, multi_pref_data, sys_data, device,
                           model_type='simple', pretrain_model=None):
    """Train preference-conditioned model for multi-objective OPF."""
    
    print('=' * 60)
    print(f'Training Multi-Preference Model - Type: {model_type}')
    print('=' * 60)
    
    # Note: x_train is loaded through dataloader, which uses multi_pref_data['x_train'] 
    y_train_by_pref = {lc: y.to(device) for lc, y in multi_pref_data['y_train_by_pref'].items()}

    # [CBF-QP TRAIN] Build training-time projector (optional)
    cbf_cfg = CBFQPTrainConfig(
        enabled=bool(getattr(config, "multi_pref_use_cbf_qp_train", False)),
        beta=float(getattr(config, "multi_pref_cbf_beta", 0.5)),
        max_iters=int(getattr(config, "multi_pref_cbf_max_iters", 6)),
        detach_active_set=bool(getattr(config, "multi_pref_cbf_detach_active_set", True)),
        penalty_rho=float(getattr(config, "multi_pref_cbf_penalty_rho", 1e7)),
        trust_region_va=float(getattr(config, "multi_pref_cbf_trust_va", 0.10)),
        trust_region_vm=float(getattr(config, "multi_pref_cbf_trust_vm", 0.02)),
        slack_eps_vm=float(getattr(config, "multi_pref_cbf_eps_vm", 0.02)),
        slack_eps_pqg=float(getattr(config, "multi_pref_cbf_eps_pqg", 0.02)),
        slack_eps_branch=float(getattr(config, "multi_pref_cbf_eps_branch", 0.02)),
        k_vm=int(getattr(config, "multi_pref_cbf_k_vm", 64)),
        k_pqg=int(getattr(config, "multi_pref_cbf_k_pqg", 64)),
        k_branch=int(getattr(config, "multi_pref_cbf_k_branch", 32)),
        apply_prob=float(getattr(config, "multi_pref_cbf_apply_prob", 1.0)),
        distill_weight=float(getattr(config, "multi_pref_cbf_distill_weight", 0.0)),

        tube_eps_vm_start=float(getattr(config, "multi_pref_tube_eps_vm_start", 0.0)),
        tube_eps_vm_end=float(getattr(config, "multi_pref_tube_eps_vm_end", 0.0)),
        tube_eps_pqg_start=float(getattr(config, "multi_pref_tube_eps_pqg_start", 0.0)),
        tube_eps_pqg_end=float(getattr(config, "multi_pref_tube_eps_pqg_end", 0.0)),
        tube_eps_branch_start=float(getattr(config, "multi_pref_tube_eps_branch_start", 0.0)),
        tube_eps_branch_end=float(getattr(config, "multi_pref_tube_eps_branch_end", 0.0)),
        tube_schedule=str(getattr(config, "multi_pref_tube_schedule", "linear")),
        tube_exp_k=float(getattr(config, "multi_pref_tube_exp_k", 5.0)),
        gate_before_solve=bool(getattr(config, "multi_pref_cbf_gate_before_solve", True)),
        gate_eps=float(getattr(config, "multi_pref_cbf_gate_eps", 1e-9)),
    )
    projector = None
    if cbf_cfg.enabled:
        try:
            projector = CBFQPProjectorNGT(sys_data, multi_pref_data, device, cbf_cfg)
            print(f"[CBF-QP TRAIN] enabled: beta={cbf_cfg.beta}, apply_prob={cbf_cfg.apply_prob}, "
                  f"trust(Va)={cbf_cfg.trust_region_va}, trust(Vm)={cbf_cfg.trust_region_vm}")
        except Exception as e:
            print(f"[CBF-QP TRAIN] WARNING: failed to build projector, fallback to no projection. Error: {e}")
            projector = None
    lambda_values = multi_pref_data['lambda_carbon_values']
    n_train = multi_pref_data['n_train']
    
    print(f"\nData: {n_train} samples, {len(lambda_values)} preferences")
    print(f"Lambda range: [{lambda_values[0]:.2f}, {lambda_values[-1]:.2f}]")
    
    num_epochs = config.multi_pref_epochs
    lr = config.multi_pref_lr
    lc_max = max(lambda_values) if max(lambda_values) > 0 else 1.0
    vae_beta = config.vae_beta
    training_mode = config.multi_pref_training_mode
    
    print(f"\nConfig: epochs={num_epochs}, lr={lr}, mode={training_mode}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    dataloader = create_multi_preference_dataloader(multi_pref_data, config, shuffle=True)
    criterion = nn.MSELoss()
    
    lambda_sorted = sorted(lambda_values)
    lambda_min, lambda_max = lambda_sorted[0], lambda_sorted[-1]
    lambda_norm = {lc: (lc - lambda_min) / (lambda_max - lambda_min) if lambda_max > lambda_min else 0.0 
                   for lc in lambda_sorted}
    NPred_Va = multi_pref_data.get('NPred_Va', multi_pref_data.get('output_dim', 0) // 2)
    
    losses = []
    start_time = time.process_time()
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss, num_batches = 0.0, 0
        
        for batch_x, batch_idx in dataloader:
            batch_x, batch_idx = batch_x.to(device), batch_idx.to(device) 
            optimizer.zero_grad()
            
            if training_mode == 'preference_trajectory' and model_type == 'rectified':
                loss = _train_trajectory_step(
                    model, batch_x, batch_idx, y_train_by_pref, lambda_sorted, lambda_norm,
                    NPred_Va, device, config, epoch, num_epochs,
                    projector
                )
            else:
                loss = _train_standard_step(
                    model, batch_x, batch_idx, y_train_by_pref, lambda_values, lc_max,
                    model_type, pretrain_model, criterion, vae_beta, device, config
                )
            
            if loss is None: continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        losses.append(epoch_loss / max(num_batches, 1))
        
        if (epoch + 1) % config.p_epoch == 0:
            print(f'Epoch {epoch+1}: Loss = {losses[-1]:.6f}')
        
        if (epoch + 1) % 100 == 0 and (epoch + 1) >= config.s_epoch:
            os.makedirs(config.model_save_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{config.model_save_dir}/model_multi_pref_{model_type}_E{epoch+1}.pth')
    
    time_train = time.process_time() - start_time
    print(f'\nCompleted in {time_train:.2f}s ({time_train/60:.2f}min)')
    
    os.makedirs(config.model_save_dir, exist_ok=True)
    final_path = f'{config.model_save_dir}/model_multi_pref_{model_type}_final.pth'
    torch.save(model.state_dict(), final_path, _use_new_zipfile_serialization=False)
    print(f'Saved: {final_path}')
    
    return model, losses, time_train


def _train_trajectory_step(
                    model, batch_x, batch_idx, y_train_by_pref, lambda_sorted, lambda_norm,
                    NPred_Va, device, config, epoch, num_epochs,
                    projector
                ):
    """Training step for preference trajectory mode."""
    B = batch_x.shape[0]
    
    x_current_list, x_next_list, lambda_curr_list, lambda_next_list, scene_list = [], [], [], [], []
    
    for i in range(B):
        idx = batch_idx[i].item()
        solutions, lambdas = [], []
        for lc in lambda_sorted:
            if lc in y_train_by_pref:
                solutions.append(y_train_by_pref[lc][idx])
                lambdas.append(lc)
        
        if len(solutions) < 2: continue
        
        k = random.randint(0, len(solutions) - 2)
        x_current_list.append(solutions[k])
        x_next_list.append(solutions[k+1])
        lambda_curr_list.append(lambdas[k])
        lambda_next_list.append(lambdas[k+1])
        scene_list.append(batch_x[i])
    
    if not x_current_list: return None
    
    x_curr_gt = torch.stack(x_current_list)
    x_next_gt = torch.stack(x_next_list)
    scene = torch.stack(scene_list)
    
    lambda_curr_norm = torch.tensor([[lambda_norm[lc]] for lc in lambda_curr_list], device=device, dtype=torch.float32)
    lambda_next_norm = torch.tensor([[lambda_norm[lc]] for lc in lambda_next_list], device=device, dtype=torch.float32)
    
    dx = wrap_angle_difference(x_next_gt - x_curr_gt, NPred_Va)
    dlambda = lambda_next_norm - lambda_curr_norm + 1e-8
    v_target = dx / dlambda
    
    v_pred = model.predict_vec(scene, x_curr_gt, lambda_curr_norm, lambda_curr_norm)


    # [CBF-QP TRAIN] Optionally project the incremental update via CBF-QP (tube + gate)
    use_cbf = (projector is not None) and getattr(projector, "cfg", None) is not None and projector.cfg.enabled

    alpha = config.multi_pref_loss_alpha
    beta = config.multi_pref_loss_beta

    # [BRIDGE] projection magnitude penalty (encourage shorter bridges)
    loss_bridge = torch.tensor(0.0, device=device)

    # [TUBE] update tube eps schedule (call once per step; cheap)
    if use_cbf and hasattr(projector, "set_progress"):
        denom = max(int(num_epochs) - 1, 1)
        progress = float(epoch) / float(denom)
        projector.set_progress(progress)

    # Decide whether to apply CBF-QP this batch (honor apply_prob)
    use_cbf_batch = False
    A0 = b0 = None
    if use_cbf:
        ap = float(getattr(projector.cfg, "apply_prob", 1.0))
        if ap >= 1.0:
            use_cbf_batch = True
        else:
            use_cbf_batch = float(torch.rand(1, device=device)) <= ap

        if use_cbf_batch:
            # Build A,b once at x_curr (detached). Both Euler and RK2 can reuse this linearization.
            with torch.no_grad():
                A0, b0 = projector.build_Ab(x_curr_gt.detach(), scene.detach())

    # Use RK2 (Heun) method if enabled, otherwise use Euler method
    if config.multi_pref_rollout_use_rk2:
        # RK2: x_{n+1} = x_n + Δλ * 0.5*(v0 + v1)
        delta1_ref = dlambda * v_pred
        if use_cbf_batch:
            delta1_exec, _info1 = projector.maybe_project_delta_given_Ab(delta1_ref, A0, b0)
        else:
            delta1_exec = delta1_ref
        x_euler = x_curr_gt + delta1_exec

        # Step 2: predict v1 at (possibly safe) intermediate point
        v1 = model.predict_vec(scene, x_euler, lambda_next_norm, lambda_next_norm)
        delta2_ref = dlambda * 0.5 * (v_pred + v1)

        if use_cbf_batch:
            # Optional: rebuild A,b at x_euler (more accurate, slower)
            if bool(getattr(config, "multi_pref_cbf_rk2_rebuild_ab", False)):
                with torch.no_grad():
                    A1, b1 = projector.build_Ab(x_euler.detach(), scene.detach())
                delta2_exec, _info2 = projector.maybe_project_delta_given_Ab(delta2_ref, A1, b1)
            else:
                delta2_exec, _info2 = projector.maybe_project_delta_given_Ab(delta2_ref, A0, b0)

            # [BRIDGE] penalize the final-stage projection magnitude
            bridge_w = float(getattr(config, "multi_pref_bridge_weight", 0.0))
            if bridge_w > 0:
                loss_bridge = torch.mean((delta2_exec - delta2_ref) ** 2)
        else:
            delta2_exec = delta2_ref

        x_pred = x_curr_gt + delta2_exec
        v_used = delta2_exec / (dlambda + 1e-12)  # for loss
        distill = torch.mean((v_pred - v_used) ** 2) if (use_cbf_batch and projector.cfg.distill_weight > 0) else 0.0

    else:
        # Euler: x_{n+1} = x_n + Δλ * v
        delta_ref = dlambda * v_pred
        if use_cbf_batch:
            delta_exec, _info = projector.maybe_project_delta_given_Ab(delta_ref, A0, b0)
            bridge_w = float(getattr(config, "multi_pref_bridge_weight", 0.0))
            if bridge_w > 0:
                loss_bridge = torch.mean((delta_exec - delta_ref) ** 2)
        else:
            delta_exec = delta_ref

        x_pred = x_curr_gt + delta_exec
        v_used = delta_exec / (dlambda + 1e-12)
        distill = torch.mean((v_pred - v_used) ** 2) if (use_cbf_batch and projector.cfg.distill_weight > 0) else 0.0

    # [CBF-QP TRAIN] velocity loss uses the actually executed velocity (v_used)
    loss_v = torch.mean((v_used - v_target) ** 2)
    # Optional distillation regularizer (reduce projection trigger over time)
    if use_cbf_batch and projector is not None and projector.cfg.distill_weight > 0:
        loss_v = loss_v + projector.cfg.distill_weight * distill
    dx_pred = wrap_angle_difference(x_pred - x_next_gt, NPred_Va)
    
    loss_endpoint = torch.nn.functional.smooth_l1_loss(dx_pred, torch.zeros_like(dx_pred))
    # loss_endpoint = torch.mean(dx_pred ** 2)
    bridge_w = float(getattr(config, "multi_pref_bridge_weight", 0.0))
    return alpha * loss_v + beta * loss_endpoint + bridge_w * loss_bridge


def _train_standard_step(model, batch_x, batch_idx, y_train_by_pref, lambda_values, lc_max,
                         model_type, pretrain_model, criterion, vae_beta, device, config):
    """Training step for standard mode."""
    B = batch_x.shape[0]
    
    lc_batch = [random.choice(lambda_values) for _ in range(B)]
    batch_y = torch.stack([y_train_by_pref[lc][batch_idx[i]] for i, lc in enumerate(lc_batch)])
    pref = torch.tensor([[lc / lc_max] for lc in lc_batch], device=device, dtype=torch.float32)
    
    if model_type == 'simple':
        x_with_pref = torch.cat([batch_x, pref], dim=1)
        return criterion(model(x_with_pref), batch_y)
        
    elif model_type == 'vae':
        use_pref_aware = hasattr(model, 'pref_dim') and model.pref_dim > 0
        if use_pref_aware:
            y_pred, mean, logvar = model.encoder_decode(batch_x, batch_y, pref=pref)
        else:
            y_pred, mean, logvar = model.encoder_decode(torch.cat([batch_x, pref], dim=1), batch_y)
        return model.loss(y_pred, batch_y, mean, logvar, beta=vae_beta)
        
    elif model_type in ['rectified', 'gaussian', 'conditional', 'interpolation']:
        t_batch = torch.rand([B, 1], device=device)
        if pretrain_model:
            with torch.no_grad():
                z_batch = pretrain_model(batch_x, use_mean=True, pref=pref) if hasattr(pretrain_model, 'pref_dim') else pretrain_model(torch.cat([batch_x, pref], dim=1), use_mean=True)
        else:
            z_batch = torch.randn_like(batch_y)
        
        flow_type = config.multi_pref_flow_type
        yt, vec_target = model.flow_forward(batch_y, t_batch, z_batch, flow_type)
        vec_pred = model.predict_vec(batch_x, yt, t_batch, pref)
        return model.loss(batch_y, z_batch, vec_pred, vec_target, flow_type)
        
    elif model_type == 'diffusion':
        t_batch = torch.rand([B, 1], device=device)
        noise = torch.randn_like(batch_y)
        x_with_pref = torch.cat([batch_x, pref], dim=1)
        if pretrain_model:
            with torch.no_grad():
                vae_anchor = pretrain_model(x_with_pref, use_mean=True)
            noise_pred = model.predict_noise_with_anchor(x_with_pref, batch_y, t_batch, noise, vae_anchor)
        else:
            noise_pred = model.predict_noise(x_with_pref, batch_y, t_batch, noise)
        return model.loss(noise_pred, noise)
    
    return None


# ==================== Main Function ====================

def main(debug=False):
    """Main function for multi-preference supervised training."""
    from unified_eval import MultiPreferencePredictor, build_ctx_from_multi_preference, evaluate_unified

    
    config = get_multi_preference_config()
    
    print("=" * 60)
    print("DeepOPF-V: Multi-Preference Training")
    print("=" * 60)
    config.print_config()
    
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    
    model_type = config.model_type
    print(f"\nModel type: {model_type}")
    
    # Load data
    multi_pref_data, sys_data = load_multi_preference_dataset(config)
    
    # Compute BRANFT directly from sys_data.branch (branch from-to indices, 0-indexed)
    # BRANFT is used for branch constraint violation checking in evaluation
    BRANFT = torch.from_numpy(sys_data.branch[:, 0:2] - 1).long()
    
    input_dim = multi_pref_data['input_dim']
    output_dim = multi_pref_data['output_dim']  # Now in NGT format (non-ZIB)
    pref_dim = config.pref_dim
    
    # Data is now converted to NGT format in load_multi_preference_dataset
    # Format: [Va_noslack_nonZIB, Vm_nonZIB]
    # Vscale and Vbias from ngt_data should match output_dim
    NPred_Va = multi_pref_data['NPred_Va']
    NPred_Vm = multi_pref_data['NPred_Vm']
    
    # Verify dimensions match
    expected_output_dim = NPred_Va + NPred_Vm
    if output_dim != expected_output_dim:
        raise ValueError(f"output_dim mismatch: got {output_dim}, expected {expected_output_dim} "
                        f"(NPred_Va={NPred_Va} + NPred_Vm={NPred_Vm})")
    
    # Use Vscale and Vbias from ngt_data (dimensions already match NGT format)
    Vscale = multi_pref_data['Vscale']
    Vbias = multi_pref_data['Vbias']
    
    # Verify Vscale/Vbias dimensions
    if len(Vscale) != output_dim:
        raise ValueError(f"Vscale dimension mismatch: got {len(Vscale)}, expected {output_dim}")
    
    print(f"\nDimensions (NGT format): input={input_dim}, output={output_dim}, pref={pref_dim}")
    print(f"NPred_Va={NPred_Va}, NPred_Vm={NPred_Vm}, Vscale.shape={Vscale.shape}, Vbias.shape={Vbias.shape}")
    
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'flow_model'))
    from net_utiles import FM, VAE, DM
    
    model, pretrain_model = None, None
    
    if model_type == 'simple':
        model = NetV(input_dim + pref_dim, output_dim, config.ngt_hidden_units, config.ngt_khidden, Vscale, Vbias)
        
    elif model_type == 'vae':
        vae_args = dict(output_dim=output_dim, hidden_dim=config.hidden_dim,
                        num_layers=config.num_layers, latent_dim=config.latent_dim,
                        output_act=None, pred_type='node', use_cvae=True)
        if config.vae_use_preference_aware:
            model = VAE(network='preference_aware_mlp', input_dim=input_dim, pref_dim=pref_dim, **vae_args)
        else:
            model = VAE(network='mlp', input_dim=input_dim + pref_dim, **vae_args)
            
    elif model_type in ['rectified', 'gaussian', 'conditional', 'interpolation']:
        model = FM(network='preference_aware_mlp', input_dim=input_dim, output_dim=output_dim,
                   hidden_dim=config.hidden_dim, num_layers=config.num_layers,
                   time_step=config.time_step, output_norm=False, pred_type='velocity', pref_dim=pref_dim)
        if config.multi_pref_training_mode == 'preference_trajectory':
            pretrain_model_path = "main_part/saved_models/model_multi_pref_vae_final.pth"
            vae_args = dict(output_dim=output_dim, hidden_dim=config.hidden_dim,
                num_layers=config.num_layers, latent_dim=config.latent_dim,
                output_act=None, pred_type='node', use_cvae=True)
            pretrain_model = VAE(network='preference_aware_mlp', input_dim=input_dim, pref_dim=pref_dim, **vae_args)
            pretrain_model.load_state_dict(torch.load(pretrain_model_path, map_location=config.device))
            pretrain_model.to(config.device)  # Move model to device
            pretrain_model.eval()
            print(f"  Loaded pretrained VAE model: {pretrain_model_path}")
                   
    elif model_type == 'diffusion':
        model = DM(network='mlp', input_dim=input_dim + pref_dim, output_dim=output_dim,
                   hidden_dim=config.hidden_dim, num_layers=config.num_layers,
                   time_step=config.time_step, output_norm=False, pred_type='node')
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    

    if model: 
        model.to(config.device)
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}") 
            
    if not debug:
        model, _, _ = train_multi_preference(config, model, multi_pref_data, sys_data, config.device,
                                              model_type=model_type, pretrain_model=pretrain_model)
    else:
        print("\n[Debug Mode] Loading model...")
        path = f'{config.model_save_dir}/model_multi_pref_{model_type}_final.pth'
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=config.device, weights_only=True))
            model.eval()
            print(f"  Loaded: {path}")
    
    # Evaluation
    print("\n" + "=" * 80)
    print("Evaluation")
    print("=" * 80)
    
    test_lambdas = [0.0, 25.0, 50.0, 80.0, 90.0]
    results_all = {}
    
    vae_best_of_k = config.vae_best_of_k
    vae_use_mean = config.vae_use_mean
    vae_selection_mode = config.vae_selection_mode
    flow_best_of_k = config.flow_best_of_k
    flow_selection_mode = config.flow_selection_mode
    
    # Create NGT loss function if Best-of-K is enabled (for VAE or Flow)
    ngt_loss_fn = None
    need_ngt_loss = (model_type == 'vae' and vae_best_of_k > 1 and not vae_use_mean) or \
                   (model_type in ['rectified', 'gaussian', 'conditional', 'interpolation'] and flow_best_of_k > 1)
    
    if need_ngt_loss:
        from deepopf_ngt_loss import DeepOPFNGTLoss
        try:
            ngt_loss_fn = DeepOPFNGTLoss(sys_data, config)
            ngt_loss_fn.cache_to_gpu(config.device)
            if model_type in ['rectified', 'gaussian', 'conditional', 'interpolation']:
                print(f"[Eval] Flow Best-of-K enabled: K={flow_best_of_k}, mode={flow_selection_mode}")
            else:
                print(f"[Eval] VAE Best-of-K enabled: K={vae_best_of_k}, mode={vae_selection_mode}")
        except Exception as e:
            print(f"[Warning] Failed to create ngt_loss_fn: {e}")
            print(f"[Warning] Best-of-K will be disabled.")
            flow_best_of_k = 1
            vae_use_mean = True
    
    def eval_on_lambdas(lambdas):
        """Evaluate model on given lambda values."""
        res = {}
        for lc in lambdas:
            print(f"\n--- lambda_carbon = {lc:.2f} ---") 

            ctx = build_ctx_from_multi_preference(config, sys_data, multi_pref_data, BRANFT, config.device, lambda_carbon=lc)
            predictor = MultiPreferencePredictor(
                model=model, multi_pref_data=multi_pref_data, lambda_carbon=lc, model_type=model_type,
                num_flow_steps=config.multi_pref_flow_steps, training_mode=config.multi_pref_training_mode,
                ngt_loss_fn=ngt_loss_fn, vae_n_samples=vae_best_of_k,
                vae_use_mean=vae_use_mean, vae_selection_mode=vae_selection_mode,
                flow_n_samples=flow_best_of_k, flow_selection_mode=flow_selection_mode,
                pretrain_model=pretrain_model
            )
            res[lc] = evaluate_unified(ctx, predictor, apply_post_processing=True, verbose=True)
        return res
    
    # Evaluate on validation set
    print(f"\n{'=' * 40} VALIDATION SET {'=' * 40}")
    results_all['val'] = eval_on_lambdas(test_lambdas)
    
    # Evaluate on training set (to check overfitting)
    print(f"\n{'=' * 40} TRAINING SET {'=' * 40}")
    orig = {k: multi_pref_data.get(k) for k in ['x_val', 'n_val', 'y_val_by_pref']}
    multi_pref_data['x_val'] = multi_pref_data['x_train']
    multi_pref_data['n_val'] = multi_pref_data['n_train']
    multi_pref_data['y_val_by_pref'] = multi_pref_data['y_train_by_pref']
    results_all['train'] = eval_on_lambdas(test_lambdas)
    for k, v in orig.items():  # Restore
        if v is not None: multi_pref_data[k] = v
    
    print("\n" + "=" * 80)
    print("Evaluation Complete")
    print("=" * 80)
    
    return results_all


if __name__ == "__main__":
    debug = bool(int(os.environ.get('DEBUG', '1')))
    main(debug=debug)
