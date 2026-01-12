#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""distill_traj_student.py

Trajectory-Student Distillation (Scheme B)
========================================

Goal
----
Train a *trajectory-level* student generator that outputs the full Pareto-front
trajectory (ordered by lambda) in one shot:

    student(scene, Y_t, t_bridge, pref_grid) -> V_pred   (trajectory velocity field)

using a rectified / flow-matching style objective on a bridge from a coarse
trajectory Y0 to a target trajectory Y*.

In this script, Y* is built on a *fine* preference grid. For fine-grid points
not present in the original dataset lambdas, we synthesize pseudo labels by
integrating a pre-trained *teacher* trajectory flow (FM) along the preference
axis (lambda), starting from the nearest *lower* GT grid point.

This implements the "方案B":
  - Reuse the training skeleton of `train_multi_preference_trajfm.py` with
    traj_rectified (trajectory rectified flow matching)
  - Build Y_star_fine using teacher integration pseudo labels

Notes / Assumptions
-------------------
* Project structure matches your existing scripts:
    - config.py, data_loader.py, unified_eval.py, mlp_anchor.py
    - flow_model/net_utiles.py provides FM and TrajectoryFM
* Angle wrapping is critical for Va dims.
* This is a minimal runnable trainer (no heavy evaluation pipeline here).

Run
---
    python distill_traj_student.py

Environment knobs (optional)
----------------------------
  # training
  EPOCHS=600 LR=1e-4 BATCH=50
  TRAJ_END_W=1.0

  # fine grid
  FINE_K=41                     # number of points in [0,1]
  # or: FINE_STEP=2.5           # step in *original lambda units* (overrides FINE_K)

  # teacher
  TEACHER_CKPT=.../model_multi_pref_rectified_traj_tfm_final.pth
  TEACHER_HIDDEN_DIM=128 TEACHER_NUM_LAYERS=2 TEACHER_STEPS=50
  TEACHER_WRAP_EACH_STEP=1

  # student (reduced params)
  TRAJ_HIDDEN_DIM=128 TRAJ_CONV_LAYERS=2 TRAJ_KERNEL=3

  # coarse Y0
  USE_VAE_ANCHOR=1
"""

import os
import sys
import time 
import random 
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn


# Make project imports work (same pattern as your other scripts)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BaseConfig, _SCRIPT_DIR
from data_loader import load_multi_preference_dataset, create_multi_preference_dataloader


# ==================== Angle utils ====================

def wrap_angle_difference(dx: torch.Tensor, n_va: int) -> torch.Tensor:
    """Wrap *differences* for Va dims to [-pi, pi].
    
    Uses torch.cat to avoid inplace operations that break autograd.
    """
    if n_va <= 0:
        return dx
    # Wrap Va dims (first n_va dimensions along last axis)
    va_wrapped = torch.atan2(torch.sin(dx[..., :n_va]), torch.cos(dx[..., :n_va]))
    # Keep non-Va dims unchanged
    non_va = dx[..., n_va:]
    # Concatenate without inplace ops
    return torch.cat([va_wrapped, non_va], dim=-1)


def wrap_angles(x: torch.Tensor, n_va: int) -> torch.Tensor:
    """Wrap *values* for Va dims to [-pi, pi].
    
    Uses torch.cat to avoid inplace operations that break autograd.
    """
    if n_va <= 0:
        return x
    # Wrap Va dims (first n_va dimensions along last axis)
    va_wrapped = torch.atan2(torch.sin(x[..., :n_va]), torch.cos(x[..., :n_va]))
    # Keep non-Va dims unchanged
    non_va = x[..., n_va:]
    # Concatenate without inplace ops
    return torch.cat([va_wrapped, non_va], dim=-1)


# ==================== Config ====================

class DistillTrajStudentConfig(BaseConfig):
    """Minimal config for trajectory-student distillation."""

    def __init__(self):
        super().__init__()

        # Dataset
        self.multi_pref_dataset_path = os.path.join(
            os.path.dirname(_SCRIPT_DIR),
            'saved_data', 'multi_preference_solutions',
            'fully_covered_dataset_2026-01-02.pt'
        )

        # Training
        self.epochs = int(os.environ.get('EPOCHS', '600'))
        self.lr = float(os.environ.get('LR', '1e-4'))
        self.weight_decay = float(os.environ.get('WEIGHT_DECAY', '1e-6'))
        self.batch_size_training = int(os.environ.get('BATCH', '50'))
        # for create_multi_preference_dataloader compatibility
        self.multi_pref_batch_size = self.batch_size_training
        self.multi_pref_val_ratio = float(os.environ.get('MULTI_PREF_VAL_RATIO', '0.2'))
        self.multi_pref_random_seed = int(os.environ.get('MULTI_PREF_RANDOM_SEED', '42'))
        self.p_epoch = int(os.environ.get('P_EPOCH', '10'))

        # Loss weights
        self.traj_end_weight = float(os.environ.get('TRAJ_END_W', '1.0'))
        # Inference-consistency loss at t=0 (directly aligns with sample_trajectory(num_steps=1))
        self.alpha_t0 = float(os.environ.get('ALPHA_T0', '1.0'))

        # Fine preference grid
        # - FINE_STEP in original lambda units overrides FINE_K
        self.fine_k = int(os.environ.get('FINE_K', '41'))
        self.fine_step = os.environ.get('FINE_STEP', '').strip()
        self.fine_step = float(self.fine_step) if self.fine_step else None

        # Teacher
        self.time_step = 1000
        self.pref_dim = 1
        self.teacher_hidden_dim = int(os.environ.get('TEACHER_HIDDEN_DIM', '128'))
        self.teacher_num_layers = int(os.environ.get('TEACHER_NUM_LAYERS', '2'))
        self.teacher_steps = int(os.environ.get('TEACHER_STEPS', '50'))
        self.teacher_wrap_each_step = os.environ.get('TEACHER_WRAP_EACH_STEP', '1').lower() in ('1', 'true', 'yes')
        # Integration method: 'euler' (1st order) or 'heun' (2nd order, more stable but 2x teacher calls)
        self.teacher_method = os.environ.get('TEACHER_METHOD', 'heun').lower()
        # Adaptive steps: max Δλ per step. If set, steps = ceil(max|Δλ|/max_step).
        # Default 0.05 means at least 20 steps for full [0,1] range. Set to 0 to disable.
        self.teacher_max_step = float(os.environ.get('TEACHER_MAX_STEP', '0.05'))
        self.teacher_ckpt = os.environ.get(
            'TEACHER_CKPT',
            os.path.join(self.model_save_dir, 'model_multi_pref_rectified_traj_tfm_final.pth')
        )
        self.require_teacher = os.environ.get('REQUIRE_TEACHER', '1').lower() in ('1', 'true', 'yes')
        
        # Memory management: max batch size for teacher integration to avoid OOM
        # If B * Kf > teacher_max_batch, processing is split into chunks
        # Default 2048 is safe for most GPUs; reduce if OOM, increase for faster processing
        self.teacher_max_batch = int(os.environ.get('TEACHER_MAX_BATCH', '2048'))

        # ==================== Anchor Strategy ====================
        # 
        # Anchor choices (in priority order):
        #   1. USE_MLP_UNIFORM=1 (RECOMMENDED): Standard MLP at λ=0, same for all pref points
        #      - Available at inference time (no GT needed)
        #      - Almost identical to GT[λ=0] (MSE ≈ 1.6e-6)
        #      - 100% sign consistency
        #      - Preserves λ variation
        #
        #   2. USE_UNIFORM_ANCHOR=1: GT[λ=0] for all pref points
        #      - Only usable during training (GT not available at inference!)
        #      - Strongest V_target signal, but train-test mismatch risk
        #
        #   3. USE_VAE_ANCHOR=1: VAE with pref conditioning
        #      - Different anchor per pref point
        #      - May be too close to GT, small V_target
        #
        #   4. USE_VAE_ANCHOR=0, USE_MLP_UNIFORM=0: Standard MLP repeated
        #      - Same as option 1 but less efficient
        #
        self.use_mlp_uniform = os.environ.get('USE_MLP_UNIFORM', '1').lower() in ('1', 'true', 'yes')
        self.use_vae_anchor = os.environ.get('USE_VAE_ANCHOR', '0').lower() in ('1', 'true', 'yes')
        self.use_uniform_anchor = os.environ.get('USE_UNIFORM_ANCHOR', '0').lower() in ('1', 'true', 'yes')
        
        # VAE anchor config (only used if use_vae_anchor=True)
        # NOTE: These defaults MUST match the VAE training config in train_multi_preference.py
        self.vae_hidden_dim = int(os.environ.get('VAE_HIDDEN_DIM', '128'))
        self.vae_num_layers = int(os.environ.get('VAE_NUM_LAYERS', '2'))
        self.vae_latent_dim = int(os.environ.get('VAE_LATENT_DIM', '64'))  # Default 64 matches train_multi_preference.py
        self.vae_use_preference_aware = os.environ.get('VAE_USE_PREF_AWARE', '1').lower() in ('1', 'true', 'yes')
        self.vae_ckpt = os.environ.get(
            'VAE_CKPT',
            os.path.join(self.model_save_dir, 'model_multi_pref_vae_final.pth')
        )

        # Student TrajectoryFM (reduced params for efficiency)
        self.traj_hidden_dim = int(os.environ.get('TRAJ_HIDDEN_DIM', '128'))
        self.traj_conv_layers = int(os.environ.get('TRAJ_CONV_LAYERS', '2'))
        self.traj_kernel = int(os.environ.get('TRAJ_KERNEL', '3'))

        # Saving
        self.tag = os.environ.get('TAG', 'traj_student_distill')
        
        # Debug visualization
        self.debug_visualize = os.environ.get('DEBUG_VIS', '0').lower() in ('1', 'true', 'yes')
        self.debug_interval = int(os.environ.get('DEBUG_INTERVAL', '100'))  # Every N epochs
        self.debug_save_dir = os.environ.get('DEBUG_SAVE_DIR', 'debug_traj_plots')
        
        # ==================== Pareto Consistency Loss ====================
        # Enforces monotonicity in objective space: as lambda increases,
        # carbon should decrease and cost should increase (Pareto trade-off).
        # 
        # Options:
        #   PARETO_LOSS_MODE:
        #     'none': Disabled
        #     'direction': Lightweight - matches direction of change with Y_star
        #     'full': Full power flow computation to verify cost/carbon monotonicity (RECOMMENDED)
        #
        #   PARETO_ALPHA: Weight for Pareto loss (default 0.1 for "light" adjustment)
        #   PARETO_FREQ: Compute every N batches (1=every batch, 5~10 recommended for full mode)
        #
        self.pareto_loss_mode = os.environ.get('PARETO_LOSS_MODE', 'full').lower()
        self.pareto_alpha = float(os.environ.get('PARETO_ALPHA', '0.01'))
        self.pareto_freq = int(os.environ.get('PARETO_FREQ', '5'))
        # Additional options for 'full' mode
        self.pareto_margin = float(os.environ.get('PARETO_MARGIN', '0.0'))
        self.pareto_w_cost = float(os.environ.get('PARETO_W_COST', '1.0'))
        self.pareto_w_carbon = float(os.environ.get('PARETO_W_CARBON', '1.0'))
        self.pareto_norm_mode = os.environ.get('PARETO_NORM_MODE', 'batch')  # 'none', 'batch', 'running'

    def print_config(self):
        super().print_config()
        print('\n[Trajectory Student Distillation]')
        print(f'  epochs={self.epochs}, lr={self.lr:.0e}, batch={self.batch_size_training}')
        print(f'  fine grid: fine_k={self.fine_k}, fine_step={self.fine_step}')
        print(f'  teacher_ckpt={self.teacher_ckpt}')
        print(f'  teacher_steps={self.teacher_steps}, method={self.teacher_method}, wrap_each_step={self.teacher_wrap_each_step}')
        print(f'  teacher_max_step={self.teacher_max_step} (adaptive), max_batch={self.teacher_max_batch}')
        print(f'  [Anchor] use_mlp_uniform={self.use_mlp_uniform}, use_vae_anchor={self.use_vae_anchor}, use_uniform_anchor={self.use_uniform_anchor}')
        if self.use_mlp_uniform:
            print(f'    MLP Uniform (RECOMMENDED): Standard MLP at lambda=0 for all pref points')
            print(f'    - Available at inference, almost identical to GT[lambda=0]')
        elif self.use_uniform_anchor:
            print(f'    GT Uniform: Using GT[lambda=0] for all pref points (training only!)')
            print(f'    - WARNING: GT not available at inference, train-test mismatch risk')
        elif self.use_vae_anchor:
            print(f'    VAE: hidden={self.vae_hidden_dim}, layers={self.vae_num_layers}, latent={self.vae_latent_dim}')
            print(f'    pref_aware={self.vae_use_preference_aware}, ckpt={self.vae_ckpt}')
        else:
            print(f'    MLP: Standard MLP anchor (lambda=0 only)')
        print(f'  student: hidden={self.traj_hidden_dim}, conv_layers={self.traj_conv_layers}, kernel={self.traj_kernel}')
        print(f'  w_end={self.traj_end_weight}, alpha_t0={self.alpha_t0}')
        print(f'  tag={self.tag}')
        if self.debug_visualize:
            print(f'  [Debug] enabled, interval={self.debug_interval}, save_dir={self.debug_save_dir}')
        print(f'\n[Pareto Consistency Loss]')
        print(f'  mode={self.pareto_loss_mode}, alpha={self.pareto_alpha}, freq={self.pareto_freq}')
        if self.pareto_loss_mode == 'full':
            print(f'  margin={self.pareto_margin}, w_cost={self.pareto_w_cost}, w_carbon={self.pareto_w_carbon}')
            print(f'  norm_mode={self.pareto_norm_mode}')


# ==================== Fine grid helpers ====================

def build_fine_pref_grid(
    lambda_sorted: List[float],
    fine_k: int,
    fine_step: Optional[float],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """Return (pref_grid_fine_norm[Kf], lambda_fine_actual[Kf], lam_min, lam_max)."""
    lam_min = float(lambda_sorted[0])
    lam_max = float(lambda_sorted[-1])
    if lam_max <= lam_min:
        # degenerate
        pref = torch.zeros((1,), device=device, dtype=torch.float32)
        lam_act = torch.tensor([lam_min], device=device, dtype=torch.float32)
        return pref, lam_act, lam_min, lam_max

    if fine_step is not None and fine_step > 0:
        lam_vals = np.arange(lam_min, lam_max + 1e-9, fine_step, dtype=np.float32)
        if lam_vals[-1] < lam_max - 1e-6:
            lam_vals = np.append(lam_vals, np.float32(lam_max))
        lam_act = torch.tensor(lam_vals, device=device, dtype=torch.float32)
        pref = (lam_act - lam_min) / (lam_max - lam_min)
        pref = torch.clamp(pref, 0.0, 1.0)
        return pref, lam_act, lam_min, lam_max

    # default: uniform in normalized space
    pref = torch.linspace(0.0, 1.0, steps=int(max(2, fine_k)), device=device, dtype=torch.float32)
    lam_act = lam_min + pref * (lam_max - lam_min)
    return pref, lam_act, lam_min, lam_max


def compute_floor_start_indices(gt_norm: torch.Tensor, fine_norm: torch.Tensor) -> torch.Tensor:
    """For each fine lambda, find the index of the nearest GT grid point <= fine lambda.

    gt_norm: [K] sorted ascending
    fine_norm: [Kf] ascending
    returns: [Kf] long indices in [0, K-1]
    """
    # searchsorted expects 1D sorted
    idx = torch.searchsorted(gt_norm, fine_norm, right=True) - 1
    idx = torch.clamp(idx, 0, gt_norm.numel() - 1)
    return idx.long()


# ==================== Core: build Y_star_fine with pseudo labels ====================

# Default max batch size for teacher integration (to avoid OOM)
# Can be overridden via environment variable TEACHER_MAX_BATCH
_DEFAULT_TEACHER_MAX_BATCH = int(os.environ.get('TEACHER_MAX_BATCH', '2048'))


def _chunked_teacher_integrate(
    teacher,
    scene_flat: torch.Tensor,      # [N_total, C]
    x0_flat: torch.Tensor,         # [N_total, D]
    lam_start_flat: torch.Tensor,  # [N_total, 1] - starting λ (normalized)
    lam_target_flat: torch.Tensor, # [N_total, 1] - target λ (normalized)
    steps: int,
    n_va: int,
    wrap_each_step: bool,
    max_batch: int,
    method: str = 'heun',          # 'euler' or 'heun' (RK2)
    max_step: float = 0.0,         # Adaptive steps: max Δλ per step (0 = disabled)
) -> torch.Tensor:
    """Chunked teacher integration along LAMBDA axis (not flow-time!).
    
    CORRECT SEMANTIC (based on train_multi_preference_tfm.py):
    ==========================================================
    The FM teacher was trained with:
        v = teacher.predict_vec(scene, x, λ_current, λ_current)
    
    Both t and pref arguments receive the CURRENT λ value!
    The teacher learns to predict velocity along the λ axis.
    
    To integrate from λ_start to λ_target:
        1. Start from x0 at λ_start
        2. Integrate along λ axis: λ_start -> λ_target
        3. At each step, pref = current λ (changes during integration)
    
    Integration methods:
        - 'euler': 1st order, fast but may drift
        - 'heun': 2nd order (RK2), more stable, 2x teacher calls per step
    
    Adaptive steps:
        If max_step > 0, dynamically increase steps so that each step's Δλ
        does not exceed max_step. This prevents large jumps when Δλ is big.
    
    Splits large batches into smaller chunks and processes sequentially.
    """
    N_total = scene_flat.shape[0]
    device = scene_flat.device
    D = x0_flat.shape[1]
    use_heun = method.lower() == 'heun'
    
    # Compute total Δλ
    dlambda_total = lam_target_flat - lam_start_flat  # [N_total, 1]
    
    # Adaptive steps: ensure each step's Δλ <= max_step
    if max_step > 0:
        max_dlambda = float(dlambda_total.abs().max().item())
        if max_dlambda > 1e-8:
            adaptive_steps = int(np.ceil(max_dlambda / max_step))
            steps = max(steps, adaptive_steps)
    
    # Compute step size along λ axis
    dlambda_step = dlambda_total / float(max(1, steps))  # [N_total, 1]
    
    def _integrate_chunk(scene_c, x0_c, lam_start_c, dlambda_step_c, n_steps):
        """Integrate a single chunk using Euler or Heun."""
        x = x0_c.clone()
        lam_current = lam_start_c.clone()
        
        for i in range(n_steps):
            # Teacher prediction at current point: v = predict_vec(scene, x, λ, λ)
            v = teacher.predict_vec(scene_c, x, lam_current, lam_current)
            
            if use_heun and i < n_steps - 1:
                # Heun's method (2nd order RK2):
                # 1. Euler predictor: x_euler = x + Δλ * v
                # 2. Corrector: v_next at (x_euler, λ + Δλ)
                # 3. Final: x = x + Δλ * 0.5 * (v + v_next)
                x_euler = x + dlambda_step_c * v
                lam_next = lam_current + dlambda_step_c
                if wrap_each_step:
                    x_euler = wrap_angles(x_euler, n_va)
                v_next = teacher.predict_vec(scene_c, x_euler, lam_next, lam_next)
                x = x + dlambda_step_c * 0.5 * (v + v_next)
            else:
                # Euler method (1st order)
                x = x + dlambda_step_c * v
            
            lam_current = lam_current + dlambda_step_c
            if wrap_each_step:
                x = wrap_angles(x, n_va)
        
        return wrap_angles(x, n_va)
    
    if N_total <= max_batch:
        # Small enough, process all at once
        return _integrate_chunk(scene_flat, x0_flat, lam_start_flat, dlambda_step, steps)
    
    # Process in chunks
    xT_flat = torch.empty((N_total, D), device=device, dtype=x0_flat.dtype)
    
    for start_idx in range(0, N_total, max_batch):
        end_idx = min(start_idx + max_batch, N_total)
        
        scene_chunk = scene_flat[start_idx:end_idx]
        x0_chunk = x0_flat[start_idx:end_idx]
        lam_start_chunk = lam_start_flat[start_idx:end_idx]
        dlambda_step_chunk = dlambda_step[start_idx:end_idx]
        
        xT_flat[start_idx:end_idx] = _integrate_chunk(
            scene_chunk, x0_chunk, lam_start_chunk, dlambda_step_chunk, steps
        )
    
    return xT_flat


@torch.no_grad()
def build_y_star_fine(
    teacher,
    batch_x: torch.Tensor,             # [B,C]
    sample_idx: torch.Tensor,          # [B]
    y_stacked_gt: torch.Tensor,        # [K,N,D]
    gt_norm: torch.Tensor,             # [K]
    fine_norm: torch.Tensor,           # [Kf]
    start_idx_for_fine: torch.Tensor,  # [Kf]
    n_va: int,
    teacher_steps: int,
    wrap_each_step: bool,
    max_batch: int = _DEFAULT_TEACHER_MAX_BATCH,
    method: str = 'heun',              # 'euler' or 'heun' (RK2)
    max_step: float = 0.05,            # Adaptive steps threshold
) -> torch.Tensor:
    """Construct Y_star_fine: [B,Kf,D] using GT + teacher pseudo integration.
    
    CORRECT SEMANTIC (Trajectory Flow Matching):
    =============================================
    The FM teacher was trained with TFM (train_multi_preference_tfm.py):
        v = teacher.predict_vec(scene, x, λ_current, λ_current)
    
    Both time and pref arguments receive the CURRENT λ value!
    The teacher learns velocity along the λ-axis (NOT flow-time t).
    Loss: ||Δλ * v - (x_next - x_current)||²
    
    For fine grid points not in GT:
        - Start from the nearest lower GT solution: x_start = GT[floor(λ)]
        - Integrate along λ-axis: λ_start → λ_target
        - At each step: x = x + Δλ * v(scene, x, λ_current, λ_current)
    
    For fine grid points that exactly match GT:
        - Use the GT value directly (no integration needed)
    
    Vectorized implementation with chunked processing to avoid OOM.
    
    Args:
        max_batch: Maximum batch size for teacher integration.
                   If B * N_integrate > max_batch, splits into chunks.
                   Default: 2048 (can be set via TEACHER_MAX_BATCH env var)
    """
    device = batch_x.device
    B = int(batch_x.shape[0])
    Kf = int(fine_norm.numel())
    C = int(batch_x.shape[1])
    D = int(y_stacked_gt.shape[2])

    # === Step 1: Gather anchor points (GT[floor(λ)]) for fine grid points ===
    # For each fine grid point, use the nearest lower GT point as starting anchor
    # This is CORRECT because Teacher was trained to integrate between adjacent λ points
    k0_all = start_idx_for_fine.long()  # [Kf] - floor indices
    x0_all = y_stacked_gt[k0_all][:, sample_idx, :].permute(1, 0, 2)  # [B, Kf, D]

    # === Step 2: Identify which fine grid points need integration ===
    # Points that exactly match GT grid don't need integration
    lam_start_all = gt_norm[k0_all]  # [Kf] - starting λ for each fine grid point
    delta_lambda_all = torch.abs(fine_norm - lam_start_all)
    needs_integration = (delta_lambda_all >= 1e-8)  # [Kf] boolean mask
    
    # Initialize output: for GT points, use GT directly; for others, will be filled by teacher
    Y_star_fine = wrap_angles(x0_all.clone(), n_va)  # [B, Kf, D]

    if not needs_integration.any():
        return Y_star_fine

    # === Step 3: Flatten points that need integration ===
    j_integrate = needs_integration.nonzero(as_tuple=True)[0]  # [N_integrate]
    N_integrate = j_integrate.numel()

    scene_flat = batch_x.unsqueeze(1).expand(B, N_integrate, C).reshape(B * N_integrate, C)
    # Starting point: GT[floor(λ)] - anchor for integration
    x0_flat = x0_all[:, j_integrate, :].reshape(B * N_integrate, D)
    # Starting λ (normalized) - where we start integration
    lam_start_flat = lam_start_all[j_integrate].unsqueeze(0).expand(B, N_integrate).reshape(B * N_integrate, 1)
    # Target λ (normalized) - where we want to reach
    lam_target_flat = fine_norm[j_integrate].unsqueeze(0).expand(B, N_integrate).reshape(B * N_integrate, 1)

    # === Step 4: Integrate along λ axis from lam_start to lam_target ===
    # Teacher was trained: v = predict_vec(scene, x, λ_current, λ_current)
    # Integration goes from GT[floor(λ)] along λ axis to target λ
    # Uses Heun (RK2) for 2nd order accuracy + adaptive steps for stability
    xT_flat = _chunked_teacher_integrate(
        teacher=teacher,
        scene_flat=scene_flat,
        x0_flat=x0_flat,
        lam_start_flat=lam_start_flat,    # starting λ (floor)
        lam_target_flat=lam_target_flat,  # target λ
        steps=int(max(1, teacher_steps)),
        n_va=n_va,
        wrap_each_step=wrap_each_step,
        max_batch=max_batch,
        method=method,                     # 'euler' or 'heun'
        max_step=max_step,                 # adaptive step threshold
    )  # Returns [B * N_integrate, D], already wrapped

    # === Step 5: Scatter results back to Y_star_fine ===
    xT_reshaped = xT_flat.view(B, N_integrate, D)
    Y_star_fine[:, j_integrate, :] = xT_reshaped

    return Y_star_fine


# ==================== Core: build coarse trajectory Y0 ====================

@torch.no_grad()
def build_y0_coarse(
    anchor_model,
    batch_x: torch.Tensor,      # [B,C]
    fine_norm: torch.Tensor,    # [Kf]
    use_vae: bool = True,
    use_uniform_anchor: bool = False,  # Use GT[λ=0] for all pref points (training only!)
    use_mlp_uniform: bool = True,      # Use MLP(λ=0) for all pref points (RECOMMENDED)
    y_stacked_gt: torch.Tensor = None,  # [K, N, D] GT solutions
    sample_idx: torch.Tensor = None,    # [B] sample indices
    mlp_anchor_model = None,           # Standard MLP for mlp_uniform mode
) -> torch.Tensor:
    """Build coarse trajectory Y0: [B,Kf,D] using anchor model.
    
    Args:
        anchor_model: VAE (preference-aware) or Standard MLP anchor
        batch_x: Input scene features [B, C]
        fine_norm: Normalized preference grid [Kf] in [0, 1]
        use_vae: If True, use VAE with pref conditioning (different anchor per pref)
                 If False, use MLP which only predicts lambda=0 (same anchor for all pref)
        use_uniform_anchor: If True, use GT[λ=0] for all pref points
                           WARNING: GT not available at inference, train-test mismatch!
        use_mlp_uniform: If True, use MLP(λ=0) for all pref points (RECOMMENDED)
                        This is available at inference and almost identical to GT[λ=0].
        y_stacked_gt: GT solutions [K, N, D] for uniform anchor mode
        sample_idx: Sample indices [B] for uniform anchor mode
        mlp_anchor_model: Standard MLP for mlp_uniform mode
    
    Returns:
        Y0: Coarse trajectory [B, Kf, D]
        
    Anchor priority:
        1. use_mlp_uniform (RECOMMENDED): MLP(λ=0) for all pref points. Available at inference.
        2. use_uniform_anchor: GT[λ=0] for all pref points. Training only!
        3. use_vae: VAE with pref conditioning. May be too close to GT.
        4. None of above: Standard MLP repeated (same as option 1 but less efficient).
    """
    device = batch_x.device
    B = int(batch_x.shape[0])
    Kf = int(fine_norm.numel())
    C = int(batch_x.shape[1])
    
    # Priority 1: MLP uniform anchor (RECOMMENDED - available at inference)
    if use_mlp_uniform and mlp_anchor_model is not None:
        # Standard MLP at λ=0 for all pref points
        pref0 = torch.zeros((B, 1), device=device, dtype=batch_x.dtype)
        with torch.no_grad():
            y_mlp = mlp_anchor_model(batch_x, use_mean=True, pref=pref0)  # [B, D]
        # Expand to all pref points: [B, D] -> [B, Kf, D]
        return y_mlp[:, None, :].expand(B, Kf, -1).clone()
    
    # Priority 2: GT uniform anchor (training only, NOT available at inference!)
    if use_uniform_anchor and y_stacked_gt is not None and sample_idx is not None:
        # GT at λ=0 (first row) for selected samples
        y_gt0 = y_stacked_gt[0, sample_idx, :]  # [B, D]
        # Expand to all pref points: [B, D] -> [B, Kf, D]
        return y_gt0[:, None, :].expand(B, Kf, -1).clone()
    
    # Expand scene: [B, C] -> [B*Kf, C]
    x_rep = batch_x[:, None, :].expand(B, Kf, C).reshape(B * Kf, C)
    
    # Priority 3: VAE with pref conditioning
    if use_vae:
        pref_rep = fine_norm[None, :, None].expand(B, Kf, 1).reshape(B * Kf, 1)
        
        # VAE supports: model(x, use_mean=True, pref=pref)
        if hasattr(anchor_model, 'pref_dim') and anchor_model.pref_dim > 0:
            y0_rep = anchor_model(x_rep, use_mean=True, pref=pref_rep)
        else:
            # Fallback: concat pref to input (for non-preference-aware VAE)
            y0_rep = anchor_model(torch.cat([x_rep, pref_rep], dim=1), use_mean=True)
        return y0_rep.view(B, Kf, -1)
    
    # Fallback: Standard MLP repeated (less efficient than use_mlp_uniform)
    pref_zero = torch.zeros((B * Kf, 1), device=device, dtype=batch_x.dtype)
    if hasattr(anchor_model, 'pref_dim'):
        y0_rep = anchor_model(x_rep, use_mean=True, pref=pref_zero)
    else:
        y0_rep = anchor_model(torch.cat([x_rep, pref_zero], dim=1), use_mean=True)

    return y0_rep.view(B, Kf, -1)


# ==================== Training ====================

def save_student(config: DistillTrajStudentConfig, student: nn.Module, tag: str) -> str:
    os.makedirs(config.model_save_dir, exist_ok=True)
    path = os.path.join(config.model_save_dir, f'model_multi_pref_traj_student_{config.tag}_{tag}.pth')
    torch.save(student.state_dict(), path, _use_new_zipfile_serialization=False)
    print(f'  Saved: {os.path.basename(path)}')
    return path


def main():
    config = DistillTrajStudentConfig()
    print('=' * 80)
    print('Trajectory Student Distillation (Scheme B: teacher-integrated pseudo Y_star_fine)')
    print('=' * 80)
    config.print_config()

    # Repro (best-effort)
    seed = int(getattr(config, 'multi_pref_random_seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = config.device

    # Load data
    multi_pref_data, sys_data = load_multi_preference_dataset(config)
    input_dim = int(multi_pref_data['input_dim'])
    output_dim = int(multi_pref_data['output_dim'])
    n_va = int(multi_pref_data['NPred_Va'])
    n_vm = int(multi_pref_data['NPred_Vm'])

    # GT grid (sorted)
    lambda_values = list(multi_pref_data['lambda_carbon_values'])
    lambda_sorted = sorted([float(x) for x in lambda_values])
    lam_min = float(lambda_sorted[0])
    lam_max = float(lambda_sorted[-1])
    if lam_max <= lam_min:
        raise ValueError('lambda range is degenerate; cannot build preference trajectory.')

    # Put GT on GPU
    y_train_by_pref = {float(lc): y.to(device=device, dtype=torch.float32) for lc, y in multi_pref_data['y_train_by_pref'].items()}
    # Stack: [K,N,D] in *sorted* order
    y_stacked_gt = torch.stack([y_train_by_pref[lc] for lc in lambda_sorted], dim=0)
    K = int(y_stacked_gt.shape[0])
    N = int(y_stacked_gt.shape[1])
    print(f'\nData: N={N} samples, K={K} GT preferences')
    print(f'Lambda range: [{lam_min:.4f}, {lam_max:.4f}]')

    # Normalized GT grid tensor [K]
    gt_norm = torch.tensor([(lc - lam_min) / (lam_max - lam_min) for lc in lambda_sorted], device=device, dtype=torch.float32)

    # Fine grid
    fine_norm, fine_lambda_actual, _, _ = build_fine_pref_grid(lambda_sorted, config.fine_k, config.fine_step, device)
    Kf = int(fine_norm.numel())
    print(f'Fine grid: Kf={Kf}, first/last actual lambda: {float(fine_lambda_actual[0]):.4f} -> {float(fine_lambda_actual[-1]):.4f}')

    # Precompute start indices per fine lambda (floor on GT grid)
    start_idx_for_fine = compute_floor_start_indices(gt_norm, fine_norm)  # [Kf]

    # Dataloader
    dataloader = create_multi_preference_dataloader(multi_pref_data, config, shuffle=True)

    # Add flow_model to path for importing FM, VAE, TrajectoryFM
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'flow_model'))
    from net_utiles import FM, VAE, TrajectoryFM

    # Load anchor model(s) for coarse trajectory Y0
    anchor_model = None
    mlp_anchor_model = None
    
    # Load Standard MLP anchor (used for MLP uniform mode, RECOMMENDED)
    if config.use_mlp_uniform:
        from mlp_anchor import load_standard_mlp_anchor
        mlp_anchor_model = load_standard_mlp_anchor(config, sys_data, multi_pref_data, device)
        mlp_anchor_model.eval()
        for p in mlp_anchor_model.parameters():
            p.requires_grad_(False)
        print('\n[Anchor] MLP Uniform (RECOMMENDED): Standard MLP at lambda=0 for all pref points')
        print('  - Available at inference, almost identical to GT[lambda=0]')
    
    elif config.use_uniform_anchor:
        # GT[λ=0] uniform anchor (training only, NOT available at inference!)
        print('\n[Anchor] GT Uniform: Using GT[lambda=0] for all pref points')
        print('  WARNING: GT not available at inference, train-test mismatch risk!')
    
    elif config.use_vae_anchor:
        # Load preference-aware VAE as anchor
        vae_args = dict(
            output_dim=output_dim,
            hidden_dim=config.vae_hidden_dim,
            num_layers=config.vae_num_layers,
            latent_dim=config.vae_latent_dim,
            output_act=None,
            pred_type='node',
            use_cvae=True,
        )
        
        if config.vae_use_preference_aware:
            anchor_model = VAE(
                network='preference_aware_mlp',
                input_dim=input_dim,
                pref_dim=config.pref_dim,
                **vae_args
            )
            print('Creating preference-aware VAE anchor')
        else:
            anchor_model = VAE(
                network='mlp',
                input_dim=input_dim + config.pref_dim,
                **vae_args
            )
            print('Creating standard VAE anchor (pref concatenated)')
        
        anchor_model = anchor_model.to(device)
        
        # Load VAE checkpoint
        if os.path.exists(config.vae_ckpt):
            state = torch.load(config.vae_ckpt, map_location=device)
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            anchor_model.load_state_dict(state, strict=False)
            print(f'Loaded VAE anchor: {config.vae_ckpt}')
        else:
            raise FileNotFoundError(
                f'VAE checkpoint not found: {config.vae_ckpt}\n'
                f'Please train VAE first using train_multi_preference.py with model_type=vae, '
                f'or set USE_VAE_ANCHOR=0 to use random initialization.'
            )
        
        anchor_model.eval()
        for p in anchor_model.parameters():
            p.requires_grad_(False)
        print('Using coarse Y0 from VAE anchor (pref-conditioned, different anchor per lambda)')
    
    else:
        # Fallback: random Gaussian noise
        print('\n[Anchor] Random Gaussian: Using N(0, 0.1) noise as initial trajectory')

    # Load teacher FM
    teacher = FM(
        network='preference_aware_mlp',
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=config.teacher_hidden_dim,
        num_layers=config.teacher_num_layers,
        time_step=config.time_step,
        output_norm=False,
        pred_type='velocity',
        pref_dim=config.pref_dim,
    ).to(device)

    if os.path.exists(config.teacher_ckpt):
        state = torch.load(config.teacher_ckpt, map_location=device)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        teacher.load_state_dict(state, strict=False)
        print(f'Loaded teacher ckpt: {config.teacher_ckpt}')
    else:
        msg = f'[Teacher missing] teacher_ckpt not found: {config.teacher_ckpt}'
        if config.require_teacher:
            raise FileNotFoundError(msg)
        print(msg)
        print('WARNING: Without teacher, this script cannot build pseudo Y_star_fine.')

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Create trajectory student
    student = TrajectoryFM(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=config.traj_hidden_dim,
        num_conv_layers=config.traj_conv_layers,
        kernel_size=config.traj_kernel,
        output_norm=False,
    ).to(device)

    print(f'\nTeacher params: {sum(p.numel() for p in teacher.parameters()):,}')
    print(f'Student params: {sum(p.numel() for p in student.parameters()):,}')

    optim = torch.optim.Adam(student.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Debug visualization setup
    debug_ctx = None
    if config.debug_visualize:
        try:
            from debug_traj_visualize import create_eval_context_for_debug, debug_single_batch
            debug_ctx = create_eval_context_for_debug(multi_pref_data, sys_data, device)
            print(f'\n[Debug] Visualization enabled, will save to {config.debug_save_dir}/')
        except Exception as e:
            print(f'[Debug] Failed to create EvalContext: {e}')
            print('[Debug] Visualization disabled.')
            config.debug_visualize = False

    # ==================== Pareto Consistency Loss Setup ====================
    pareto_loss_fn = None
    pareto_direction_loss_fn = None
    
    if config.pareto_loss_mode == 'full':
        try:
            from pareto_loss import ParetoLossComputer, ParetoLossConfig, create_pareto_loss_from_multi_pref_data
            pareto_config = ParetoLossConfig(
                alpha=config.pareto_alpha,
                margin=config.pareto_margin,
                norm_mode=config.pareto_norm_mode,
                w_cost_mono=config.pareto_w_cost,
                w_carbon_mono=config.pareto_w_carbon,
                compute_freq=config.pareto_freq,
            )
            pareto_loss_fn = create_pareto_loss_from_multi_pref_data(
                multi_pref_data, sys_data, device, pareto_config
            )
            print(f'\n[Pareto] Full mode enabled: power flow computation for cost/carbon monotonicity')
        except Exception as e:
            print(f'[Pareto] Failed to create ParetoLossComputer: {e}')
            print('[Pareto] Falling back to direction mode.')
            config.pareto_loss_mode = 'direction'
    
    if config.pareto_loss_mode == 'direction':
        from pareto_loss import compute_pareto_direction_loss
        pareto_direction_loss_fn = compute_pareto_direction_loss
        print(f'\n[Pareto] Direction mode enabled: lightweight direction consistency loss')

    # Train loop
    student.train()
    start_time = time.process_time()

    for epoch in range(config.epochs):
        loss_sum = 0.0
        loss_fm_sum = 0.0
        loss_end_sum = 0.0
        loss_t0_sum = 0.0
        loss_pareto_sum = 0.0
        pareto_details_epoch = {}
        nb = 0

        for batch_x, batch_idx in dataloader:
            batch_x = batch_x.to(device, non_blocking=True)
            sample_idx = batch_idx.to(device, non_blocking=True).long()
            B = int(batch_x.shape[0])

            optim.zero_grad(set_to_none=True)

            # 1) Build coarse trajectory Y0
            # Priority: MLP uniform > GT uniform > VAE > random
            with torch.no_grad():
                Y0 = build_y0_coarse(
                    anchor_model=anchor_model, 
                    batch_x=batch_x, 
                    fine_norm=fine_norm, 
                    use_vae=config.use_vae_anchor,
                    use_uniform_anchor=config.use_uniform_anchor,
                    use_mlp_uniform=config.use_mlp_uniform,
                    y_stacked_gt=y_stacked_gt,
                    sample_idx=sample_idx,
                    mlp_anchor_model=mlp_anchor_model,
                )
                Y0 = wrap_angles(Y0, n_va)

            # 2) Build target trajectory Y_star_fine (GT + pseudo from teacher)
            with torch.no_grad():
                Y_star = build_y_star_fine(
                    teacher=teacher,
                    batch_x=batch_x,
                    sample_idx=sample_idx,
                    y_stacked_gt=y_stacked_gt,
                    gt_norm=gt_norm,
                    fine_norm=fine_norm,
                    start_idx_for_fine=start_idx_for_fine,
                    n_va=n_va,
                    teacher_steps=config.teacher_steps,
                    wrap_each_step=config.teacher_wrap_each_step,
                    max_batch=config.teacher_max_batch,
                    method=config.teacher_method,      # 'euler' or 'heun'
                    max_step=config.teacher_max_step,  # adaptive step threshold
                )

            # 3) Rectified trajectory bridge sample
            t = torch.rand((B, 1), device=device, dtype=torch.float32)
            t_expand = t.view(B, 1, 1)
            Yt = (1.0 - t_expand) * Y0 + t_expand * Y_star
            Yt = wrap_angles(Yt, n_va)

            # Constant-velocity target for rectified path
            V_target = wrap_angle_difference(Y_star - Y0, n_va)

            # 4) Student predicts velocity field
            V_pred = student(batch_x, Yt, t, fine_norm)  # [B,Kf,D]

            # FM loss (wrap error on Va dims)
            err_v = wrap_angle_difference(V_pred - V_target, n_va)
            loss_fm = (err_v ** 2).mean()

            # Endpoint consistency: Y_hat = Yt + (1-t)*V_pred
            Y_hat = Yt + (1.0 - t_expand) * V_pred
            Y_hat = wrap_angles(Y_hat, n_va)
            err_end = wrap_angle_difference(Y_hat - Y_star, n_va)
            loss_end = (err_end ** 2).mean()

            # Inference-consistency loss at t=0 (aligns with sample_trajectory(num_steps=1))
            # This directly enforces: Y0 + V(Y0, t=0) ≈ Y_star
            t0 = torch.zeros((B, 1), device=device)
            V0 = student(batch_x, Y0, t0, fine_norm)
            Y0_hat = wrap_angles(Y0 + V0, n_va)
            err_t0 = wrap_angle_difference(Y0_hat - Y_star, n_va)
            loss_t0 = (err_t0 ** 2).mean()

            # ==================== Pareto Consistency Loss ====================
            # Enforce monotonicity in objective space along lambda axis
            loss_pareto = torch.tensor(0.0, device=device)
            pareto_details = {}
            
            if config.pareto_loss_mode == 'full' and pareto_loss_fn is not None:
                # Full mode: compute cost/carbon from power flow
                loss_pareto, pareto_details = pareto_loss_fn(Y0_hat, fine_norm, batch_x)
            elif config.pareto_loss_mode == 'direction' and pareto_direction_loss_fn is not None:
                # Direction mode: match change direction with Y_star
                loss_pareto, pareto_details = pareto_direction_loss_fn(
                    Y0_hat, Y_star, n_va, alpha=config.pareto_alpha
                )

            loss = loss_fm + config.traj_end_weight * loss_end + config.alpha_t0 * loss_t0 + loss_pareto
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optim.step()

            loss_sum += float(loss.detach().cpu().item())
            loss_fm_sum += float(loss_fm.detach().cpu().item())
            loss_end_sum += float(loss_end.detach().cpu().item())
            loss_t0_sum += float(loss_t0.detach().cpu().item())
            loss_pareto_sum += float(loss_pareto.detach().cpu().item()) if isinstance(loss_pareto, torch.Tensor) else loss_pareto
            
            # Accumulate Pareto details for logging
            for k, v in pareto_details.items():
                if k not in pareto_details_epoch:
                    pareto_details_epoch[k] = 0.0
                pareto_details_epoch[k] += v if isinstance(v, (int, float)) else 0.0
            
            nb += 1

        denom = max(nb, 1)
        avg_fm = loss_fm_sum / denom
        avg_end = loss_end_sum / denom
        avg_t0 = loss_t0_sum / denom
        avg_pareto = loss_pareto_sum / denom
        avg_total = loss_sum / denom
        
        if (epoch + 1) % config.p_epoch == 0:
            log_str = (
                f'Epoch {epoch+1:04d} | '
                f'total={avg_total:.4e} | '
                f'fm={avg_fm:.4e} '
                f'end={avg_end:.4e} '
                f't0={avg_t0:.4e}'
            )
            if config.pareto_loss_mode != 'none':
                log_str += f' pareto={avg_pareto:.4e}'
                # Log Pareto violation ratios if available
                if 'sign_violation_ratio' in pareto_details_epoch:
                    log_str += f' (sign_vio={pareto_details_epoch["sign_violation_ratio"]/denom:.2%})'
                elif 'carbon_violation_ratio' in pareto_details_epoch:
                    log_str += f' (C_vio={pareto_details_epoch["carbon_violation_ratio"]/denom:.2%})'
            print(log_str)

        # Debug visualization
        if config.debug_visualize and debug_ctx is not None and (epoch + 1) % config.debug_interval == 0:
            try:
                # Use the last batch for visualization
                student.eval()
                with torch.no_grad():
                    t_zero = torch.zeros((B, 1), device=device)
                    V_pred_debug = student(batch_x, Y0, t_zero, fine_norm)
                    Y_pred_debug = Y0 + V_pred_debug
                    Y_pred_debug = wrap_angles(Y_pred_debug, n_va)
                
                debug_single_batch(
                    Y0=Y0,
                    Y_star=Y_star,
                    Y_pred=Y_pred_debug,
                    fine_norm=fine_norm,
                    ctx=debug_ctx,
                    n_va=n_va,
                    sample_idx=0,
                    save_dir=config.debug_save_dir,
                    epoch=epoch + 1,
                )
                student.train()
            except Exception as e:
                print(f'[Debug] Visualization failed: {e}')

        # optional periodic save
        if (epoch + 1) % 200 == 0:
            save_student(config, student, tag=f'E{epoch+1}')

    elapsed = time.process_time() - start_time
    print(f'\nTraining done in {elapsed:.2f}s ({elapsed/60:.2f} min)')
    save_student(config, student, tag='final')


if __name__ == '__main__':
    main()
