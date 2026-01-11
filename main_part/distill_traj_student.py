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
  TRAJ_END_W=0.1 TRAJ_SMOOTH_W=0.01

  # fine grid
  FINE_K=41                     # number of points in [0,1]
  # or: FINE_STEP=2.5           # step in *original lambda units* (overrides FINE_K)

  # teacher
  TEACHER_CKPT=.../model_multi_pref_rectified_traj_tfm_final.pth
  TEACHER_HIDDEN_DIM=128 TEACHER_NUM_LAYERS=2 TEACHER_STEPS=50
  TEACHER_WRAP_EACH_STEP=1

  # student
  TRAJ_HIDDEN_DIM=256 TRAJ_CONV_LAYERS=4 TRAJ_KERNEL=3

  # coarse Y0
  USE_MLP_ANCHOR=1
"""

import os
import sys
import time
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

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


# ==================== Teacher integration ====================

@torch.no_grad()
def teacher_integrate_to_lambda(
    teacher,
    scene: torch.Tensor,
    x0: torch.Tensor,
    lambda_start: torch.Tensor,   # [B,1] in [0,1]
    lambda_target: torch.Tensor,  # [B,1] in [0,1]
    steps: int,
    n_va: int,
    wrap_each_step: bool = True,
) -> torch.Tensor:
    """Euler integrate teacher vector field along lambda.

    x_{n+1} = x_n + dλ * teacher.predict_vec(scene, x_n, λ_n, λ_n)

    IMPORTANT: This routine assumes lambda_target >= lambda_start.
    """
    steps = int(max(1, steps))
    delta = lambda_target - lambda_start
    # If any negative due to numerical issues, clamp to 0 (no backward integration here).
    delta = torch.clamp(delta, min=0.0)

    if delta.max().item() < 1e-8:
        return wrap_angles(x0.clone(), n_va)

    x = x0.clone()
    lam = lambda_start.clone()
    dlam = delta / float(steps)

    for _ in range(steps):
        v = teacher.predict_vec(scene, x, lam, lam)
        x = x + dlam * v
        if wrap_each_step:
            x = wrap_angles(x, n_va)
        lam = lam + dlam

    return wrap_angles(x, n_va)


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
        self.traj_smooth_weight = float(os.environ.get('TRAJ_SMOOTH_W', '0.01'))

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
        self.teacher_ckpt = os.environ.get(
            'TEACHER_CKPT',
            os.path.join(self.model_save_dir, 'model_multi_pref_rectified_traj_tfm_final.pth')
        )
        self.require_teacher = os.environ.get('REQUIRE_TEACHER', '1').lower() in ('1', 'true', 'yes')
        
        # Memory management: max batch size for teacher integration to avoid OOM
        # If B * Kf > teacher_max_batch, processing is split into chunks
        # Default 2048 is safe for most GPUs; reduce if OOM, increase for faster processing
        self.teacher_max_batch = int(os.environ.get('TEACHER_MAX_BATCH', '2048'))

        # Coarse trajectory source (anchor model)
        # USE_VAE_ANCHOR=1: Use VAE with preference conditioning (recommended, supports different pref)
        # USE_VAE_ANCHOR=0: Use Standard MLP (only predicts lambda=0, same anchor for all pref)
        self.use_vae_anchor = os.environ.get('USE_VAE_ANCHOR', '1').lower() in ('1', 'true', 'yes')
        
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

        # Student TrajectoryFM
        self.traj_hidden_dim = int(os.environ.get('TRAJ_HIDDEN_DIM', '256'))
        self.traj_conv_layers = int(os.environ.get('TRAJ_CONV_LAYERS', '4'))
        self.traj_kernel = int(os.environ.get('TRAJ_KERNEL', '3'))

        # Saving
        self.tag = os.environ.get('TAG', 'traj_student_distill')

    def print_config(self):
        super().print_config()
        print('\n[Trajectory Student Distillation]')
        print(f'  epochs={self.epochs}, lr={self.lr:.0e}, batch={self.batch_size_training}')
        print(f'  fine grid: fine_k={self.fine_k}, fine_step={self.fine_step}')
        print(f'  teacher_ckpt={self.teacher_ckpt}')
        print(f'  teacher_steps={self.teacher_steps}, wrap_each_step={self.teacher_wrap_each_step}, max_batch={self.teacher_max_batch}')
        print(f'  [Anchor] use_vae_anchor={self.use_vae_anchor}')
        if self.use_vae_anchor:
            print(f'    VAE: hidden={self.vae_hidden_dim}, layers={self.vae_num_layers}, latent={self.vae_latent_dim}')
            print(f'    pref_aware={self.vae_use_preference_aware}, ckpt={self.vae_ckpt}')
        else:
            print(f'    MLP: Standard MLP anchor (lambda=0 only)')
        print(f'  student: hidden={self.traj_hidden_dim}, conv_layers={self.traj_conv_layers}, kernel={self.traj_kernel}')
        print(f'  w_end={self.traj_end_weight}, w_smooth={self.traj_smooth_weight}')
        print(f'  tag={self.tag}')


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
    lam_start_flat: torch.Tensor,  # [N_total, 1]
    lam_target_flat: torch.Tensor, # [N_total, 1]
    steps: int,
    n_va: int,
    wrap_each_step: bool,
    max_batch: int,
) -> torch.Tensor:
    """Chunked teacher integration to avoid OOM.
    
    Splits large batches into smaller chunks and processes sequentially.
    This trades speed for memory safety.
    """
    N_total = scene_flat.shape[0]
    device = scene_flat.device
    D = x0_flat.shape[1]
    
    if N_total <= max_batch:
        # Small enough, process all at once
        delta_flat = lam_target_flat - lam_start_flat
        dlam = delta_flat / float(steps)
        
        x = x0_flat.clone()
        lam = lam_start_flat.clone()
        
        for _ in range(steps):
            v = teacher.predict_vec(scene_flat, x, lam, lam)
            x = x + dlam * v
            if wrap_each_step:
                x = wrap_angles(x, n_va)
            lam = lam + dlam
        
        return wrap_angles(x, n_va)
    
    # Process in chunks
    xT_flat = torch.empty((N_total, D), device=device, dtype=x0_flat.dtype)
    
    for start_idx in range(0, N_total, max_batch):
        end_idx = min(start_idx + max_batch, N_total)
        
        scene_chunk = scene_flat[start_idx:end_idx]
        x0_chunk = x0_flat[start_idx:end_idx]
        lam_start_chunk = lam_start_flat[start_idx:end_idx]
        lam_target_chunk = lam_target_flat[start_idx:end_idx]
        
        delta_chunk = lam_target_chunk - lam_start_chunk
        dlam_chunk = delta_chunk / float(steps)
        
        x = x0_chunk.clone()
        lam = lam_start_chunk.clone()
        
        for _ in range(steps):
            v = teacher.predict_vec(scene_chunk, x, lam, lam)
            x = x + dlam_chunk * v
            if wrap_each_step:
                x = wrap_angles(x, n_va)
            lam = lam + dlam_chunk
        
        xT_flat[start_idx:end_idx] = wrap_angles(x, n_va)
    
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
) -> torch.Tensor:
    """Construct Y_star_fine: [B,Kf,D] using GT + teacher pseudo integration.
    
    Vectorized implementation with chunked processing to avoid OOM.
    
    Args:
        max_batch: Maximum batch size for teacher integration.
                   If B * N_integrate > max_batch, splits into chunks.
                   Default: 2048 (can be set via TEACHER_MAX_BATCH env var)
    
    Memory usage estimate:
        - B=50, Kf=41 → N_total ≈ 2050 samples (borderline)
        - B=100, Kf=41 → N_total ≈ 4100 samples (likely needs chunking)
    """
    device = batch_x.device
    B = int(batch_x.shape[0])
    Kf = int(fine_norm.numel())
    C = int(batch_x.shape[1])
    D = int(y_stacked_gt.shape[2])

    # === Step 1: Gather all starting points x0 for (B, Kf) combinations ===
    k0_all = start_idx_for_fine.long()  # [Kf]
    x0_all = y_stacked_gt[k0_all][:, sample_idx, :].permute(1, 0, 2)  # [B, Kf, D]

    # === Step 2: Compute lambda_start and delta_lambda for each (b, j) ===
    lam_start_all = gt_norm[k0_all]  # [Kf]
    delta_lambda_all = torch.clamp(fine_norm - lam_start_all, min=0.0)

    # === Step 3: Identify which fine grid points need integration ===
    needs_integration = (delta_lambda_all >= 1e-8)  # [Kf] boolean mask
    
    # Initialize output with wrapped x0
    Y_star_fine = wrap_angles(x0_all.clone(), n_va)  # [B, Kf, D]

    if not needs_integration.any():
        return Y_star_fine

    # === Step 4: Flatten points that need integration ===
    j_integrate = needs_integration.nonzero(as_tuple=True)[0]  # [N_integrate]
    N_integrate = j_integrate.numel()

    scene_flat = batch_x.unsqueeze(1).expand(B, N_integrate, C).reshape(B * N_integrate, C)
    x0_flat = x0_all[:, j_integrate, :].reshape(B * N_integrate, D)
    lam_start_flat = lam_start_all[j_integrate].unsqueeze(0).expand(B, N_integrate).reshape(B * N_integrate, 1)
    lam_target_flat = fine_norm[j_integrate].unsqueeze(0).expand(B, N_integrate).reshape(B * N_integrate, 1)

    # === Step 5: Chunked Euler integration (handles large batches) ===
    xT_flat = _chunked_teacher_integrate(
        teacher=teacher,
        scene_flat=scene_flat,
        x0_flat=x0_flat,
        lam_start_flat=lam_start_flat,
        lam_target_flat=lam_target_flat,
        steps=int(max(1, teacher_steps)),
        n_va=n_va,
        wrap_each_step=wrap_each_step,
        max_batch=max_batch,
    )  # Returns [B * N_integrate, D], already wrapped

    # === Step 6: Scatter results back to Y_star_fine ===
    # Reshape: [B * N_integrate, D] -> [B, N_integrate, D]
    xT_reshaped = xT_flat.view(B, N_integrate, D)
    # Write back to the corresponding fine grid indices
    Y_star_fine[:, j_integrate, :] = xT_reshaped

    return Y_star_fine


# ==================== Core: build coarse trajectory Y0 ====================

@torch.no_grad()
def build_y0_coarse(
    anchor_model,
    batch_x: torch.Tensor,      # [B,C]
    fine_norm: torch.Tensor,    # [Kf]
    use_vae: bool = True,
) -> torch.Tensor:
    """Build coarse trajectory Y0: [B,Kf,D] using anchor model.
    
    Args:
        anchor_model: VAE (preference-aware) or Standard MLP anchor
        batch_x: Input scene features [B, C]
        fine_norm: Normalized preference grid [Kf] in [0, 1]
        use_vae: If True, use VAE with pref conditioning (different anchor per pref)
                 If False, use MLP which only predicts lambda=0 (same anchor for all pref)
    
    Returns:
        Y0: Coarse trajectory [B, Kf, D]
        
    Key difference:
        - VAE (use_vae=True): anchor_model(x, pref=pref) generates different initial points
          for different preference values. This provides better starting points.
        - MLP (use_vae=False): anchor_model(x, pref=0) always predicts lambda=0 solution,
          so all preference points start from the same anchor. This is harder to learn from.
    """
    device = batch_x.device
    B = int(batch_x.shape[0])
    Kf = int(fine_norm.numel())
    C = int(batch_x.shape[1])
    
    # Expand scene: [B, C] -> [B*Kf, C]
    x_rep = batch_x[:, None, :].expand(B, Kf, C).reshape(B * Kf, C)
    
    if use_vae:
        # VAE with preference conditioning: each pref point gets different anchor
        pref_rep = fine_norm[None, :, None].expand(B, Kf, 1).reshape(B * Kf, 1)
        
        # VAE supports: model(x, use_mean=True, pref=pref)
        if hasattr(anchor_model, 'pref_dim') and anchor_model.pref_dim > 0:
            y0_rep = anchor_model(x_rep, use_mean=True, pref=pref_rep)
        else:
            # Fallback: concat pref to input (for non-preference-aware VAE)
            y0_rep = anchor_model(torch.cat([x_rep, pref_rep], dim=1), use_mean=True)
    else:
        # Standard MLP: always use pref=0 (lambda_min), same anchor for all pref points
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

    # Load anchor model for coarse trajectory Y0
    anchor_model = None
    use_vae_for_anchor = config.use_vae_anchor
    
    if config.use_vae_anchor:
        # Load preference-aware VAE as anchor (recommended)
        # VAE can generate different initial points for different preferences
        
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
            print(f'[WARNING] VAE checkpoint not found: {config.vae_ckpt}')
            print('  Falling back to Standard MLP anchor...')
            use_vae_for_anchor = False
            anchor_model = None
        
        if anchor_model is not None:
            anchor_model.eval()
            for p in anchor_model.parameters():
                p.requires_grad_(False)
            print('Using coarse Y0 from VAE anchor (pref-conditioned, different anchor per lambda)')
    
    if not use_vae_for_anchor:
        # Fallback: Load Standard MLP anchor (only predicts lambda=0)
        from mlp_anchor import load_standard_mlp_anchor
        try:
            anchor_model = load_standard_mlp_anchor(config, sys_data, multi_pref_data, device)
            anchor_model.eval()
            for p in anchor_model.parameters():
                p.requires_grad_(False)
            print('Using coarse Y0 from Standard MLP anchor (same anchor for all lambda)')
        except FileNotFoundError:
            anchor_model = None
            print('Using coarse Y0 from Gaussian noise (no anchor model available)')

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

    # Train loop
    student.train()
    start_time = time.process_time()

    for epoch in range(config.epochs):
        loss_sum = 0.0
        loss_fm_sum = 0.0
        loss_end_sum = 0.0
        loss_smooth_sum = 0.0
        nb = 0

        for batch_x, batch_idx in dataloader:
            batch_x = batch_x.to(device, non_blocking=True)
            sample_idx = batch_idx.to(device, non_blocking=True).long()
            B = int(batch_x.shape[0])

            optim.zero_grad(set_to_none=True)

            # 1) Build coarse trajectory Y0
            if anchor_model is not None:
                with torch.no_grad():
                    Y0 = build_y0_coarse(anchor_model, batch_x, fine_norm, use_vae=use_vae_for_anchor)
                    Y0 = wrap_angles(Y0, n_va)
            else:
                Y0 = torch.randn((B, Kf, output_dim), device=device, dtype=torch.float32)

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

            # Smoothness along preference axis (on Y_hat)
            diff = wrap_angle_difference(Y_hat[:, 1:, :] - Y_hat[:, :-1, :], n_va)
            loss_smooth = (diff ** 2).mean()

            loss = loss_fm + config.traj_end_weight * loss_end + config.traj_smooth_weight * loss_smooth
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optim.step()

            loss_sum += float(loss.detach().cpu().item())
            loss_fm_sum += float(loss_fm.detach().cpu().item())
            loss_end_sum += float(loss_end.detach().cpu().item())
            loss_smooth_sum += float(loss_smooth.detach().cpu().item())
            nb += 1

        denom = max(nb, 1)
        avg_fm = loss_fm_sum / denom
        avg_end = loss_end_sum / denom
        avg_smooth = loss_smooth_sum / denom
        avg_total = loss_sum / denom
        
        if (epoch + 1) % config.p_epoch == 0:
            # Compute weighted contributions
            weighted_fm = avg_fm  # weight = 1.0 (implicit)
            weighted_end = config.traj_end_weight * avg_end
            weighted_smooth = config.traj_smooth_weight * avg_smooth
            
            print(
                f'Epoch {epoch+1:04d} | '
                f'total={avg_total:.4e} | '
                f'fm={avg_fm:.4e} '
                f'end={avg_end:.4e} '
                f'smooth={avg_smooth:.4e}'
            )
            
            # Show weighted contributions and ratio analysis
            # This helps diagnose if weights are balanced
            if (epoch + 1) == config.p_epoch or (epoch + 1) % 100 == 0:
                # Calculate contribution percentages
                total_weighted = weighted_fm + weighted_end + weighted_smooth + 1e-12
                pct_fm = 100 * weighted_fm / total_weighted
                pct_end = 100 * weighted_end / total_weighted
                pct_smooth = 100 * weighted_smooth / total_weighted
                
                print(f'  [Weighted] fm*1.0={weighted_fm:.4e} ({pct_fm:.1f}%), '
                      f'end*{config.traj_end_weight}={weighted_end:.4e} ({pct_end:.1f}%), '
                      f'smooth*{config.traj_smooth_weight}={weighted_smooth:.4e} ({pct_smooth:.1f}%)')
                
                # Loss magnitude ratio analysis
                if avg_end > 1e-12:
                    ratio_fm_end = avg_fm / avg_end
                    print(f'  [Ratio] fm/end = {ratio_fm_end:.2f}x '
                          f'(if >> 1: consider increasing traj_end_weight)')
                
                # Early warning for potential weight imbalance
                if pct_fm > 90:
                    print(f'  [Warning] FM loss dominates ({pct_fm:.0f}%), '
                          f'endpoint learning may be suppressed. '
                          f'Consider increasing TRAJ_END_W (current: {config.traj_end_weight})')
                elif pct_end > 70:
                    print(f'  [Warning] Endpoint loss dominates ({pct_end:.0f}%), '
                          f'velocity field learning may be affected. '
                          f'Consider decreasing TRAJ_END_W (current: {config.traj_end_weight})')

        # optional periodic save
        if (epoch + 1) % 200 == 0:
            save_student(config, student, tag=f'E{epoch+1}')

    elapsed = time.process_time() - start_time
    print(f'\nTraining done in {elapsed:.2f}s ({elapsed/60:.2f} min)')
    save_student(config, student, tag='final')


if __name__ == '__main__':
    main()
