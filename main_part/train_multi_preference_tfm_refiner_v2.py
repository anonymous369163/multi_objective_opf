#!/usr/bin/env python
# coding: utf-8
"""
Multi-Preference Training with Simplified Refiner + 3-Stage Training Pipeline

This script implements:
  1) SimpleRefinerMLP: only predicts Δx (no L prediction), direct projection to λ=0
  2) 3-Stage Training Pipeline:
     - Stage 1: Independent training (Flow on real trajectory, Refiner learns anchor→GT(λ=0))
     - Stage 2: Flow robustness (freeze Refiner, perturb k=0 segment only)
     - Stage 3: Joint fine-tuning (end-to-end gradients, k=0 only)

Key Simplifications vs. refiner_v1:
  - Refiner only predicts Δx = x₀_GT - x_anchor (no L)
  - No virtual segment complexity (Flow always starts from λ=0)
  - 3-stage training for better convergence

Inference Pipeline:
  anchor → Refiner(Δx) → x̂₀ = anchor + Δx → Flow(λ=0 → λ_target) → final prediction

Author: Peng Yue
Date: January 2026
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BaseConfig, _SCRIPT_DIR
from data_loader import load_multi_preference_dataset, create_multi_preference_dataloader


# ==================== Configuration ====================

class RefinerV2Config(BaseConfig):
    """Configuration for Simplified Refiner + 3-Stage Training."""

    def __init__(self):
        super().__init__()

        # ==================== Dataset ====================
        self.multi_pref_dataset_path = os.path.join(
            os.path.dirname(_SCRIPT_DIR), 'saved_data', 'multi_preference_solutions', 'fully_covered_dataset_2026-01-02.pt'
        )

        # ==================== Model Architecture ====================
        self.hidden_dim = int(os.environ.get('HIDDEN_DIM', '128'))
        self.num_layers = int(os.environ.get('NUM_LAYERS', '2'))
        self.time_step = 1000
        self.pref_dim = 1

        # Refiner architecture
        self.refiner_hidden_dim = int(os.environ.get('REFINER_HIDDEN_DIM', '128'))
        self.refiner_num_layers = int(os.environ.get('REFINER_NUM_LAYERS', '2'))

        # ==================== Stage 1: Independent Training ====================
        self.stage1_flow_epochs = int(os.environ.get('STAGE1_FLOW_EPOCHS', '500'))
        self.stage1_flow_lr = float(os.environ.get('STAGE1_FLOW_LR', '1e-4'))
        self.stage1_refiner_epochs = int(os.environ.get('STAGE1_REFINER_EPOCHS', '200'))
        self.stage1_refiner_lr = float(os.environ.get('STAGE1_REFINER_LR', '1e-4'))

        # ==================== Stage 2: Flow Robustness ====================
        self.stage2_epochs = int(os.environ.get('STAGE2_EPOCHS', '300'))
        self.stage2_ramp_epochs = int(os.environ.get('STAGE2_RAMP_EPOCHS', '150'))
        self.stage2_flow_lr = float(os.environ.get('STAGE2_FLOW_LR', '1e-4'))

        # ==================== Stage 3: Joint Fine-tuning ====================
        self.stage3_epochs = int(os.environ.get('STAGE3_EPOCHS', '200'))
        self.stage3_flow_lr = float(os.environ.get('STAGE3_FLOW_LR', '1e-5'))
        self.stage3_refiner_lr = float(os.environ.get('STAGE3_REFINER_LR', '5e-5'))

        # ==================== TFM Options ====================
        self.multi_pref_tfm_sigma = float(os.environ.get('MULTI_PREF_TFM_SIGMA', '0.0'))
        self.multi_pref_tfm_alpha_eps = float(os.environ.get('MULTI_PREF_TFM_ALPHA_EPS', '1e-3'))
        self.multi_pref_loss_alpha = float(os.environ.get('MULTI_PREF_LOSS_ALPHA', '1.0'))
        self.multi_pref_loss_beta = float(os.environ.get('MULTI_PREF_LOSS_BETA', '0.0'))

        # ==================== Training Control ====================
        self.multi_pref_batch_size = int(os.environ.get('MULTI_PREF_BATCH_SIZE', '100'))
        self.batch_size_training = self.multi_pref_batch_size
        self.weight_decay = 1e-6
        self.p_epoch = 10
        self.s_epoch = 100  # Save checkpoint after this epoch
        
        # Skip stages (for resuming training)
        self.skip_stage1_flow = os.environ.get('SKIP_STAGE1_FLOW', '0').lower() in ['1', 'true', 'yes']
        self.skip_stage1_refiner = os.environ.get('SKIP_STAGE1_REFINER', '0').lower() in ['1', 'true', 'yes']
        self.skip_stage2 = os.environ.get('SKIP_STAGE2', '0').lower() in ['1', 'true', 'yes']
        self.skip_stage3 = os.environ.get('SKIP_STAGE3', '0').lower() in ['1', 'true', 'yes']

        # VAE config (for compatibility)
        self.vae_use_preference_aware = True

    def print_config(self):
        super().print_config()
        print(f"\n[3-Stage Training Config]")
        print(f"  Stage 1 Flow: {self.stage1_flow_epochs} epochs, LR={self.stage1_flow_lr:.0e}")
        print(f"  Stage 1 Refiner: {self.stage1_refiner_epochs} epochs, LR={self.stage1_refiner_lr:.0e}")
        print(f"  Stage 2: {self.stage2_epochs} epochs, ramp={self.stage2_ramp_epochs}, LR={self.stage2_flow_lr:.0e}")
        print(f"  Stage 3: {self.stage3_epochs} epochs, Flow LR={self.stage3_flow_lr:.0e}, Refiner LR={self.stage3_refiner_lr:.0e}")
        print(f"\n[Skip Stages]")
        print(f"  Skip Stage1 Flow: {self.skip_stage1_flow}")
        print(f"  Skip Stage1 Refiner: {self.skip_stage1_refiner}")
        print(f"  Skip Stage2: {self.skip_stage2}")
        print(f"  Skip Stage3: {self.skip_stage3}")


def get_config():
    return RefinerV2Config()


# ==================== Utility Functions ====================

def wrap_angle_difference(dx: torch.Tensor, NPred_Va: int) -> torch.Tensor:
    """Wrap angle differences for Va dims to [-pi, pi]."""
    if NPred_Va <= 0:
        return dx
    out = dx.clone()
    out[..., :NPred_Va] = torch.atan2(torch.sin(dx[..., :NPred_Va]), torch.cos(dx[..., :NPred_Va]))
    return out


def wrap_angles(x: torch.Tensor, NPred_Va: int) -> torch.Tensor:
    """Wrap angle values for Va dims to [-pi, pi]."""
    if NPred_Va <= 0:
        return x
    out = x.clone()
    out[..., :NPred_Va] = torch.atan2(torch.sin(x[..., :NPred_Va]), torch.cos(x[..., :NPred_Va]))
    return out


# ==================== SimpleRefinerMLP ====================

class SimpleRefinerMLP(nn.Module):
    """
    Simplified Refiner: only predicts Δx (no L).
    
    Input: (scene, anchor) where anchor is from Standard MLP
    Output: Δx such that x̂₀ = anchor + Δx ≈ x₀_GT
    """

    def __init__(self, scene_dim: int, anchor_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.scene_dim = int(scene_dim)
        self.anchor_dim = int(anchor_dim)
        in_dim = self.scene_dim + self.anchor_dim

        layers = []
        d = in_dim
        for _ in range(max(1, int(num_layers))):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.SiLU())
            d = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.head_dx = nn.Linear(d, self.anchor_dim)

    def forward(self, scene: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scene: [B, scene_dim] - input load scenario
            anchor: [B, anchor_dim] - anchor from pretrained model
        
        Returns:
            dx: [B, anchor_dim] - predicted correction
        """
        feat = self.encoder(torch.cat([scene, anchor], dim=1))
        dx = self.head_dx(feat)
        return dx


# ==================== Anchor Loader ====================

@torch.no_grad()
def _get_anchor(pretrain_model, scene: torch.Tensor) -> torch.Tensor:
    """Get deterministic anchor from pretrained model."""
    pref0 = torch.zeros((scene.shape[0], 1), device=scene.device, dtype=scene.dtype)
    return pretrain_model(scene, use_mean=True, pref=pref0)


# ==================== TFM Loss ====================

def _compute_tfm_displacement_loss(
    model, scene: torch.Tensor, x_curr: torch.Tensor, x_next: torch.Tensor,
    lam_curr: torch.Tensor, lam_next: torch.Tensor, config, NPred_Va: int
) -> torch.Tensor:
    """Compute TFM displacement loss: ||Δλ * v_pred - (x_next - x_t)||²"""
    B = x_curr.shape[0]
    alpha_eps = float(getattr(config, 'multi_pref_tfm_alpha_eps', 1e-3))
    alpha = torch.rand((B, 1), device=x_curr.device, dtype=x_curr.dtype)
    alpha = torch.clamp(alpha, min=alpha_eps, max=1.0 - alpha_eps)

    dl_seg = lam_next - lam_curr
    lam_t = lam_curr + alpha * dl_seg
    mu_t = (1.0 - alpha) * x_curr + alpha * x_next

    sigma = float(getattr(config, 'multi_pref_tfm_sigma', 0.0))
    if sigma > 0:
        x_t = mu_t + (sigma * torch.sqrt(torch.clamp(alpha * (1.0 - alpha), min=0.0))) * torch.randn_like(mu_t)
    else:
        x_t = mu_t

    dl_remain = torch.clamp(lam_next - lam_t, min=1e-8)
    dx_target = wrap_angle_difference(x_next - x_t, NPred_Va)

    v_pred = model.predict_vec(scene, x_t, lam_t, lam_t)
    delta = dl_remain * v_pred
    loss_displacement = torch.mean((delta - dx_target) ** 2)

    loss_alpha = float(getattr(config, 'multi_pref_loss_alpha', 1.0))
    loss_beta = float(getattr(config, 'multi_pref_loss_beta', 0.0))
    
    if loss_beta > 0:
        x_pred = x_t + delta
        dx_pred = wrap_angle_difference(x_pred - x_next, NPred_Va)
        loss_endpoint = F.smooth_l1_loss(dx_pred, torch.zeros_like(dx_pred))
        return loss_alpha * loss_displacement + loss_beta * loss_endpoint
    
    return loss_alpha * loss_displacement


# ==================== Stage 1: Independent Training ====================

def train_flow_stage1(config, model, multi_pref_data, device, y_stacked, lambda_norm_tensor, NPred_Va):
    """Stage 1: Train Flow model on real trajectory segments (standard TFM)."""
    print('\n' + '=' * 60)
    print('Stage 1: Flow Training (Real Trajectory)')
    print('=' * 60)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.stage1_flow_lr, weight_decay=config.weight_decay)
    dataloader = create_multi_preference_dataloader(multi_pref_data, config, shuffle=True)
    
    K = y_stacked.shape[0]
    model.train()
    start_time = time.process_time()

    for epoch in range(config.stage1_flow_epochs):
        epoch_loss, num_batches = 0.0, 0

        for batch_x, batch_idx in dataloader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_idx = batch_idx.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            B = batch_x.shape[0]
            sample_idx = batch_idx.long()
            
            # Sample random segment k in [0, K-2]
            k = torch.randint(0, K - 1, (B,), device=device)
            x_curr = y_stacked[k, sample_idx, :]
            x_next = y_stacked[k + 1, sample_idx, :]
            lam_curr = lambda_norm_tensor[k].view(-1, 1)
            lam_next = lambda_norm_tensor[k + 1].view(-1, 1)

            loss = _compute_tfm_displacement_loss(model, batch_x, x_curr, x_next, lam_curr, lam_next, config, NPred_Va)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % config.p_epoch == 0:
            print(f'  Epoch {epoch+1}/{config.stage1_flow_epochs}: Loss = {epoch_loss/num_batches:.4e}')

        if (epoch + 1) % 100 == 0 and (epoch + 1) >= config.s_epoch:
            _save_checkpoint(config, model, None, f'flow_s1_E{epoch+1}')

    elapsed = time.process_time() - start_time
    print(f'Stage 1 Flow completed in {elapsed/60:.2f} min')
    _save_checkpoint(config, model, None, 'flow_s1_final')
    
    return model


def train_refiner_stage1(config, refiner, pretrain_model, multi_pref_data, device, y_stacked, NPred_Va):
    """Stage 1: Train Refiner to predict direct projection anchor → GT(λ=0)."""
    print('\n' + '=' * 60)
    print('Stage 1: Refiner Training (Direct Projection)')
    print('=' * 60)

    optimizer = torch.optim.Adam(refiner.parameters(), lr=config.stage1_refiner_lr, weight_decay=config.weight_decay)
    dataloader = create_multi_preference_dataloader(multi_pref_data, config, shuffle=True)
    
    refiner.train()
    start_time = time.process_time()

    for epoch in range(config.stage1_refiner_epochs):
        epoch_loss, num_batches = 0.0, 0

        for batch_x, batch_idx in dataloader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_idx = batch_idx.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            sample_idx = batch_idx.long()
            
            # Get anchor and GT at λ=0
            x_anchor = _get_anchor(pretrain_model, batch_x)
            x0_gt = y_stacked[0, sample_idx, :]  # GT at λ=0

            # Target: Δx = x0_gt - x_anchor
            dx_gt = wrap_angle_difference(x0_gt - x_anchor, NPred_Va)

            # Predict
            dx_pred = refiner(batch_x, x_anchor)
            dx_err = wrap_angle_difference(dx_pred - dx_gt, NPred_Va)
            loss = torch.mean(dx_err ** 2)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % config.p_epoch == 0:
            print(f'  Epoch {epoch+1}/{config.stage1_refiner_epochs}: Loss = {epoch_loss/num_batches:.4e}')

    elapsed = time.process_time() - start_time
    print(f'Stage 1 Refiner completed in {elapsed/60:.2f} min')
    _save_checkpoint(config, None, refiner, 'refiner_s1_final')
    
    return refiner


# ==================== Stage 2: Flow Robustness ====================

def train_flow_stage2(config, model, refiner, pretrain_model, multi_pref_data, device, y_stacked, lambda_norm_tensor, NPred_Va):
    """Stage 2: Train Flow with mixed starting points (only perturb k=0 segment)."""
    print('\n' + '=' * 60)
    print('Stage 2: Flow Robustness Training (Perturb k=0 Only)')
    print('=' * 60)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.stage2_flow_lr, weight_decay=config.weight_decay)
    dataloader = create_multi_preference_dataloader(multi_pref_data, config, shuffle=True)
    
    K = y_stacked.shape[0]
    model.train()
    refiner.eval()  # Freeze Refiner
    start_time = time.process_time()

    for epoch in range(config.stage2_epochs):
        epoch_loss, num_batches = 0.0, 0
        # Mixing ratio: ramps from 0 to 1
        p = min(epoch / max(config.stage2_ramp_epochs, 1), 1.0)

        for batch_x, batch_idx in dataloader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_idx = batch_idx.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            B = batch_x.shape[0]
            sample_idx = batch_idx.long()
            
            # Sample random segment k in [0, K-2]
            k = torch.randint(0, K - 1, (B,), device=device)
            x_curr = y_stacked[k, sample_idx, :].clone()  # Clone to allow modification
            x_next = y_stacked[k + 1, sample_idx, :]
            lam_curr = lambda_norm_tensor[k].view(-1, 1)
            lam_next = lambda_norm_tensor[k + 1].view(-1, 1)

            # === ONLY perturb k=0 samples ===
            mask_k0 = (k == 0)
            if mask_k0.any():
                batch_x_k0 = batch_x[mask_k0]
                with torch.no_grad():
                    x_anchor = _get_anchor(pretrain_model, batch_x_k0)
                    dx_pred = refiner(batch_x_k0, x_anchor)
                    x_ref = wrap_angles(x_anchor + dx_pred, NPred_Va)
                
                x_gt_k0 = x_curr[mask_k0]  # GT at λ=0
                # Mixed starting point: (1-p)*GT + p*Refiner
                x_mixed = (1 - p) * x_gt_k0 + p * x_ref
                x_curr[mask_k0] = x_mixed

            # For k>0: x_curr remains unchanged (uses GT)
            loss = _compute_tfm_displacement_loss(model, batch_x, x_curr, x_next, lam_curr, lam_next, config, NPred_Va)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % config.p_epoch == 0:
            print(f'  Epoch {epoch+1}/{config.stage2_epochs}: Loss = {epoch_loss/num_batches:.4e}, p = {p:.2f}')

        if (epoch + 1) % 100 == 0 and (epoch + 1) >= config.s_epoch:
            _save_checkpoint(config, model, None, f'flow_s2_E{epoch+1}')

    elapsed = time.process_time() - start_time
    print(f'Stage 2 completed in {elapsed/60:.2f} min')
    _save_checkpoint(config, model, None, 'flow_s2_final')
    
    return model


# ==================== Stage 3: Joint Fine-tuning ====================

def train_joint_stage3(config, model, refiner, pretrain_model, multi_pref_data, device, y_stacked, lambda_norm_tensor, NPred_Va):
    """Stage 3: Joint training with end-to-end gradients (only for k=0)."""
    print('\n' + '=' * 60)
    print('Stage 3: Joint Fine-tuning (End-to-End)')
    print('=' * 60)

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': config.stage3_flow_lr},
        {'params': refiner.parameters(), 'lr': config.stage3_refiner_lr},
    ], weight_decay=config.weight_decay)
    
    dataloader = create_multi_preference_dataloader(multi_pref_data, config, shuffle=True)
    
    K = y_stacked.shape[0]
    model.train()
    refiner.train()
    start_time = time.process_time()

    for epoch in range(config.stage3_epochs):
        epoch_loss, num_batches = 0.0, 0

        for batch_x, batch_idx in dataloader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_idx = batch_idx.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            B = batch_x.shape[0]
            sample_idx = batch_idx.long()
            
            # Sample random segment k in [0, K-2]
            k = torch.randint(0, K - 1, (B,), device=device)
            x_curr = y_stacked[k, sample_idx, :].clone()
            x_next = y_stacked[k + 1, sample_idx, :]
            lam_curr = lambda_norm_tensor[k].view(-1, 1)
            lam_next = lambda_norm_tensor[k + 1].view(-1, 1)

            # === Joint training ONLY for k=0 samples ===
            mask_k0 = (k == 0)
            if mask_k0.any():
                batch_x_k0 = batch_x[mask_k0]
                x_anchor = _get_anchor(pretrain_model, batch_x_k0)
                
                # NO detach! Gradients flow back to Refiner
                dx_pred = refiner(batch_x_k0, x_anchor)
                x_ref = wrap_angles(x_anchor + dx_pred, NPred_Va)
                
                # Use 100% Refiner output (p=1.0 in Stage 3)
                x_curr[mask_k0] = x_ref

            # Compute TFM loss - gradients flow to both Flow and Refiner (for k=0)
            loss = _compute_tfm_displacement_loss(model, batch_x, x_curr, x_next, lam_curr, lam_next, config, NPred_Va)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % config.p_epoch == 0:
            print(f'  Epoch {epoch+1}/{config.stage3_epochs}: Loss = {epoch_loss/num_batches:.4e}')

        if (epoch + 1) % 100 == 0 and (epoch + 1) >= config.s_epoch:
            _save_checkpoint(config, model, refiner, f'joint_E{epoch+1}')

    elapsed = time.process_time() - start_time
    print(f'Stage 3 completed in {elapsed/60:.2f} min')
    _save_checkpoint(config, model, refiner, 'final')
    
    return model, refiner


# ==================== Checkpoint Management ====================

def _save_checkpoint(config, model, refiner, tag):
    """Save model checkpoints."""
    os.makedirs(config.model_save_dir, exist_ok=True)
    
    if model is not None:
        flow_path = os.path.join(config.model_save_dir, f'model_multi_pref_refiner_v2_flow_{tag}.pth')
        torch.save(model.state_dict(), flow_path, _use_new_zipfile_serialization=False)
        print(f'  Saved: {os.path.basename(flow_path)}')
    
    if refiner is not None:
        refiner_path = os.path.join(config.model_save_dir, f'model_multi_pref_refiner_v2_mlp_{tag}.pth')
        torch.save(refiner.state_dict(), refiner_path, _use_new_zipfile_serialization=False)
        print(f'  Saved: {os.path.basename(refiner_path)}')


def _load_checkpoint(config, model, refiner, tag, device):
    """Load model checkpoints if they exist."""
    flow_path = os.path.join(config.model_save_dir, f'model_multi_pref_refiner_v2_flow_{tag}.pth')
    refiner_path = os.path.join(config.model_save_dir, f'model_multi_pref_refiner_v2_mlp_{tag}.pth')
    
    loaded = False
    if model is not None and os.path.exists(flow_path):
        model.load_state_dict(torch.load(flow_path, map_location=device, weights_only=True))
        print(f'  Loaded: {os.path.basename(flow_path)}')
        loaded = True
    
    if refiner is not None and os.path.exists(refiner_path):
        refiner.load_state_dict(torch.load(refiner_path, map_location=device, weights_only=True))
        print(f'  Loaded: {os.path.basename(refiner_path)}')
        loaded = True
    
    return loaded


# ==================== Main ====================

def main():
    config = get_config()
    
    print("=" * 60)
    print("DeepOPF-V: Simplified Refiner + 3-Stage Training")
    print("=" * 60)
    config.print_config()

    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)

    device = config.device
    model_type = config.model_type
    
    if model_type not in ['rectified', 'gaussian', 'conditional', 'interpolation']:
        raise ValueError(f"Only Flow models supported, got: {model_type}")

    # Load data
    multi_pref_data, sys_data = load_multi_preference_dataset(config)
    input_dim = int(multi_pref_data['input_dim'])
    output_dim = int(multi_pref_data['output_dim'])
    NPred_Va = int(multi_pref_data['NPred_Va'])
    NPred_Vm = int(multi_pref_data['NPred_Vm'])

    if output_dim != (NPred_Va + NPred_Vm):
        raise ValueError(f"output_dim mismatch: got {output_dim}, expected {NPred_Va + NPred_Vm}")

    # Prepare training data
    y_train_by_pref = {lc: y.to(device) for lc, y in multi_pref_data['y_train_by_pref'].items()}
    lambda_values = multi_pref_data['lambda_carbon_values']
    lambda_sorted = sorted(lambda_values)
    lambda_min, lambda_max = lambda_sorted[0], lambda_sorted[-1]
    lambda_norm = {lc: (lc - lambda_min) / (lambda_max - lambda_min) if lambda_max > lambda_min else 0.0
                   for lc in lambda_sorted}
    
    y_stacked = torch.stack([y_train_by_pref[lc] for lc in lambda_sorted], dim=0)
    lambda_norm_tensor = torch.tensor([lambda_norm[lc] for lc in lambda_sorted], device=device, dtype=torch.float32)
    
    print(f"\nData: {multi_pref_data['n_train']} samples, {len(lambda_values)} preferences")
    print(f"y_stacked: {y_stacked.shape}, lambda_norm_tensor: {lambda_norm_tensor.shape}")

    # Create Flow model
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'flow_model'))
    from net_utiles import FM
    
    model = FM(
        network='preference_aware_mlp', input_dim=input_dim, output_dim=output_dim,
        hidden_dim=config.hidden_dim, num_layers=config.num_layers,
        time_step=config.time_step, output_norm=False, pred_type='velocity', pref_dim=config.pref_dim
    ).to(device)
    
    # Create SimpleRefinerMLP
    refiner = SimpleRefinerMLP(
        scene_dim=input_dim,
        anchor_dim=output_dim,
        hidden_dim=config.refiner_hidden_dim,
        num_layers=config.refiner_num_layers
    ).to(device)

    # Load Standard MLP anchor
    from mlp_anchor import load_standard_mlp_anchor
    pretrain_model = load_standard_mlp_anchor(config, sys_data, multi_pref_data, device)
    print(f"  Using Standard MLP as anchor")

    print(f"\nFlow params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Refiner params: {sum(p.numel() for p in refiner.parameters()):,}")

    # ==================== Stage 1: Independent Training ====================
    if not config.skip_stage1_flow:
        model = train_flow_stage1(config, model, multi_pref_data, device, y_stacked, lambda_norm_tensor, NPred_Va)
    else:
        print('\n[SKIP] Stage 1 Flow (loading checkpoint)')
        if not _load_checkpoint(config, model, None, 'flow_s1_final', device):
            # Try loading from standard TFM
            tfm_path = os.path.join(config.model_save_dir, 'model_multi_pref_rectified_traj_tfm_final.pth')
            if os.path.exists(tfm_path):
                model.load_state_dict(torch.load(tfm_path, map_location=device, weights_only=True))
                print(f'  Loaded existing TFM model: {os.path.basename(tfm_path)}')
            else:
                print('  [WARNING] No checkpoint found, training from scratch')
                model = train_flow_stage1(config, model, multi_pref_data, device, y_stacked, lambda_norm_tensor, NPred_Va)

    if not config.skip_stage1_refiner:
        refiner = train_refiner_stage1(config, refiner, pretrain_model, multi_pref_data, device, y_stacked, NPred_Va)
    else:
        print('\n[SKIP] Stage 1 Refiner (loading checkpoint)')
        if not _load_checkpoint(config, None, refiner, 'refiner_s1_final', device):
            print('  [WARNING] No checkpoint found, training from scratch')
            refiner = train_refiner_stage1(config, refiner, pretrain_model, multi_pref_data, device, y_stacked, NPred_Va)

    # ==================== Stage 2: Flow Robustness ====================
    if not config.skip_stage2:
        model = train_flow_stage2(config, model, refiner, pretrain_model, multi_pref_data, device, y_stacked, lambda_norm_tensor, NPred_Va)
    else:
        print('\n[SKIP] Stage 2 (loading checkpoint)')
        _load_checkpoint(config, model, None, 'flow_s2_final', device)

    # ==================== Stage 3: Joint Fine-tuning ====================
    if not config.skip_stage3:
        model, refiner = train_joint_stage3(config, model, refiner, pretrain_model, multi_pref_data, device, y_stacked, lambda_norm_tensor, NPred_Va)
    else:
        print('\n[SKIP] Stage 3')

    print('\n' + '=' * 60)
    print('Training Complete!')
    print('=' * 60)
    print(f'Final models saved to: {config.model_save_dir}')
    print(f'  Flow: model_multi_pref_refiner_v2_flow_final.pth')
    print(f'  Refiner: model_multi_pref_refiner_v2_mlp_final.pth')


if __name__ == "__main__":
    main()
