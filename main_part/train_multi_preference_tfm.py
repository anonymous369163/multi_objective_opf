#!/usr/bin/env python
# coding: utf-8
"""
Multi-Preference Training with Trajectory Flow Matching (TFM)

Trains preference-conditioned Flow models using preference trajectory mode.

Author: Peng Yue
Date: December 2025

Usage:
    MODEL_TYPE=rectified python train_multi_preference_tfm.py
    DEBUG=1 python train_multi_preference_tfm.py  # Evaluation only
"""

import torch
import torch.nn.functional as F
import time
import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BaseConfig, _SCRIPT_DIR
from data_loader import load_multi_preference_dataset, create_multi_preference_dataloader


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
        self.multi_pref_epochs = int(os.environ.get('MULTI_PREF_EPOCHS', '1000'))   # origin: 1000
        self.multi_pref_lr = float(os.environ.get('MULTI_PREF_LR', '1e-4'))
        self.multi_pref_flow_steps = int(os.environ.get('MULTI_PREF_FLOW_STEPS', '10'))
        self.multi_pref_batch_size = int(os.environ.get('MULTI_PREF_BATCH_SIZE', '100'))  # origin: 50
        
        # Validation split
        self.multi_pref_val_ratio = float(os.environ.get('MULTI_PREF_VAL_RATIO', '0.2'))
        self.multi_pref_random_seed = int(os.environ.get('MULTI_PREF_RANDOM_SEED', '42'))
        
        # Model Architecture
        self.hidden_dim = int(os.environ.get('HIDDEN_DIM', '128'))
        self.num_layers = int(os.environ.get('NUM_LAYERS', '2'))
        self.latent_dim = int(os.environ.get('LATENT_DIM', '64'))  # VAE anchor
        self.time_step = 1000

        # ==================== [TFM] Trajectory Flow Matching Options ==================== 
        # [TFM] Gaussian bridge noise std (in the same unit as x, i.e., radians for Va and p.u. for Vm).
        # Bridge sampling: x_t = (1-a)x_k + a x_{k+1} + sigma * sqrt(a(1-a)) * eps
        # Set to 0.0 to disable noise (still TFM target, but sampled exactly on the chord).
        self.multi_pref_tfm_sigma = float(os.environ.get('MULTI_PREF_TFM_SIGMA', '0.0'))

        # [TFM] Avoid alpha too close to 0/1 to prevent division-by-zero in (1-a)*dlambda.
        self.multi_pref_tfm_alpha_eps = float(os.environ.get('MULTI_PREF_TFM_ALPHA_EPS', '1e-3'))
        
        # Batch size (needed by DeepOPFNGTLoss for HV guidance)
        self.batch_size_training = self.multi_pref_batch_size
        
        # Loss weights for preference trajectory training
        # loss = alpha * loss_displacement (+ optional endpoint term)
        # 注意: 使用增量损失 ||Δλ·v - Δx||² 而非速度损失 ||v - Δx/Δλ||²，避免除以小 Δλ 导致的高方差
        self.multi_pref_loss_alpha = float(os.environ.get('MULTI_PREF_LOSS_ALPHA', '1.0'))    # 增量损失权重
        self.multi_pref_loss_beta = float(os.environ.get('MULTI_PREF_LOSS_BETA', '0.0'))      # 终点误差权重(标准FM不需要，设为0)
        
        # Multi-step rollout method
        self.multi_pref_rollout_use_rk2 = os.environ.get('MULTI_PREF_ROLLOUT_USE_RK2', 'False').lower() == 'true'
        # True: RK2(Heun)二阶精度, 每步2次模型调用, 更稳定
        # False: Euler一阶精度, 每步1次模型调用, 更快

        # Preference conditioning
        self.pref_dim = 1
        
        # VAE config (for loading pretrained anchor model)
        self.vae_use_preference_aware = True
        
        # ==================== Flow Best-of-K Evaluation ====================
        self.flow_best_of_k = int(os.environ.get('FLOW_BEST_OF_K', '32'))
        self.flow_selection_mode = os.environ.get('FLOW_SELECTION_MODE', 'constraint')
        
        # ==================== Training Control ====================
        self.weight_decay = 1e-6
        self.p_epoch = 10
        self.s_epoch = 800
        
    def print_config(self):
        """Print configuration summary."""
        super().print_config()
        print(f"\n[Training Config]")
        print(f"  Epochs: {self.multi_pref_epochs}, LR: {self.multi_pref_lr}, Batch: {self.multi_pref_batch_size}")
        print(f"  Training mode: trajectory")
        print(f"  Loss: displacement-based ||Δλ·v - Δx||² (alpha={self.multi_pref_loss_alpha}, beta={self.multi_pref_loss_beta})")
        print(f"\n[TFM Training]") 
        print(f"  Bridge sigma: {self.multi_pref_tfm_sigma}, alpha_eps: {self.multi_pref_tfm_alpha_eps}")


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


# ==================== Training Functions ====================

def _generate_model_filename(config, model_type, epoch=None, is_final=False):
    """
    生成模型文件名。
    
    格式: model_multi_pref_{type}_traj_tfm[_E{epoch}|_final].pth
    """ 
    base = f"model_multi_pref_{model_type}_traj_tfm"  
    
    if is_final:
        return f"{base}_final.pth"
    elif epoch is not None:
        return f"{base}_E{epoch}.pth"
    return f"{base}.pth"


def train_multi_preference(config, model, multi_pref_data, sys_data, device,
                           model_type='rectified', pretrain_model=None):
    """
    Train Flow model using preference trajectory mode (TFM).
    
    Note: Only supports Flow models (rectified, gaussian, etc.) with preference trajectory.
          For standard training of VAE/simple models, use train_multi_preference.py.
    """
    # ==================== CUDA Performance Optimization ====================
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster matmul
        torch.backends.cudnn.allow_tf32 = True
    
    print('=' * 60)
    print(f'TFM Training - Model: {model_type}')
    print('=' * 60)
    
    y_train_by_pref = {lc: y.to(device) for lc, y in multi_pref_data['y_train_by_pref'].items()}
    lambda_values = multi_pref_data['lambda_carbon_values']
    n_train = multi_pref_data['n_train']
    
    print(f"\nData: {n_train} samples, {len(lambda_values)} preferences")
    print(f"Lambda range: [{lambda_values[0]:.2f}, {lambda_values[-1]:.2f}]")
    
    num_epochs = config.multi_pref_epochs
    lr = config.multi_pref_lr
    
    print(f"\nConfig: epochs={num_epochs}, lr={lr}, mode=trajectory, use_rk2={config.multi_pref_rollout_use_rk2}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    
    # Learning rate scheduler: constant (no decay)
    scheduler = None
    print(f"[LR Scheduler] Constant LR = {lr:.2e}")
    
    dataloader = create_multi_preference_dataloader(multi_pref_data, config, shuffle=True)
    
    lambda_sorted = sorted(lambda_values)
    lambda_min, lambda_max = lambda_sorted[0], lambda_sorted[-1]
    lambda_norm = {lc: (lc - lambda_min) / (lambda_max - lambda_min) if lambda_max > lambda_min else 0.0 
                   for lc in lambda_sorted}
    NPred_Va = multi_pref_data.get('NPred_Va', multi_pref_data.get('output_dim', 0) // 2)
    
    # ==================== Pre-stack y_train_by_pref for Vectorized Training ====================
    # Stack all preference solutions into a single tensor [K, N, D] for fast indexing
    K = len(lambda_sorted)
    y_stacked = torch.stack([y_train_by_pref[lc] for lc in lambda_sorted], dim=0)  # [K, N, D]
    lambda_norm_tensor = torch.tensor([lambda_norm[lc] for lc in lambda_sorted], 
                                       device=device, dtype=torch.float32)  # [K]
    print(f"[Vectorized] Pre-stacked y_train: {y_stacked.shape}, lambda_norm_tensor: {lambda_norm_tensor.shape}")
    
    losses = []
    start_time = time.process_time()
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss, num_batches = 0.0, 0
        
        for batch_x, batch_idx in dataloader:
            batch_x, batch_idx = batch_x.to(device, non_blocking=True), batch_idx.to(device, non_blocking=True) 
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
            loss = _train_trajectory_step(
                model, batch_x, batch_idx, y_train_by_pref, lambda_sorted, lambda_norm,
                NPred_Va, device, config, y_stacked, lambda_norm_tensor
            ) 
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        losses.append(epoch_loss / max(num_batches, 1))
        
        # Update learning rate (if scheduler exists)
        if scheduler is not None:
            scheduler.step()
        
        if (epoch + 1) % config.p_epoch == 0:
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else lr
            # Get relative error and dx_target magnitude for better interpretability
            rel_err = getattr(_train_trajectory_step, 'last_relative_error', 0.0)
            dx_mag = getattr(_train_trajectory_step, 'last_dx_target_mean', 0.0)
            print(f'Epoch {epoch+1}: Loss = {losses[-1]:.4e}, RelErr = {rel_err:.2%}, |dx| = {dx_mag:.4e}, LR = {current_lr:.2e}')
        
        if (epoch + 1) % 100 == 0 and (epoch + 1) >= config.s_epoch:
            os.makedirs(config.model_save_dir, exist_ok=True)
            checkpoint_filename = _generate_model_filename(config, model_type, epoch=epoch+1, is_final=False)
            checkpoint_path = f'{config.model_save_dir}/{checkpoint_filename}'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'  Checkpoint saved: {checkpoint_filename}')
    
    time_train = time.process_time() - start_time
    print(f'\nCompleted in {time_train:.2f}s ({time_train/60:.2f}min)')
    
    os.makedirs(config.model_save_dir, exist_ok=True)
    final_filename = _generate_model_filename(config, model_type, epoch=None, is_final=True)
    final_path = f'{config.model_save_dir}/{final_filename}'
    torch.save(model.state_dict(), final_path, _use_new_zipfile_serialization=False)
    print(f'Saved: {final_filename}')
    
    return model, losses, time_train


def _train_trajectory_step(
    model, batch_x, batch_idx, y_train_by_pref, lambda_sorted, lambda_norm,
    NPred_Va, device, config, y_stacked=None, lambda_norm_tensor=None
):
    """Training step for preference trajectory mode using Trajectory Flow Matching (TFM).
    
    Uses vectorized operations for efficiency.
    """
    B = batch_x.shape[0]
    K = len(lambda_sorted)
    
    if K < 2:
        return None
    
    # ==================== Vectorized Sampling ====================
    if y_stacked is not None and lambda_norm_tensor is not None:
        # Vectorized: sample random k in [0, K-2] for each sample in batch
        k_indices = torch.randint(0, K - 1, (B,), device=device)
        sample_idx = batch_idx.long()
        
        # Gather x_curr and x_next using advanced indexing
        x_curr_gt = y_stacked[k_indices, sample_idx, :]
        x_next_gt = y_stacked[k_indices + 1, sample_idx, :]
        
        lambda_curr_norm = lambda_norm_tensor[k_indices].view(-1, 1)
        lambda_next_norm = lambda_norm_tensor[k_indices + 1].view(-1, 1)
        scene = batch_x
    else:
        # Fallback: loop-based sampling
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
    
    # ==================== TFM: Build intermediate bridge sample ==================== 
    dlambda_seg = lambda_next_norm - lambda_curr_norm
 
    # Sample alpha in (0,1), clamp for numerical stability
    alpha_eps = float(getattr(config, 'multi_pref_tfm_alpha_eps', 1e-3))
    alpha = torch.rand((B, 1), device=device, dtype=x_curr_gt.dtype)
    alpha = torch.clamp(alpha, min=alpha_eps, max=1.0 - alpha_eps)

    # Continuous preference location within the segment
    lambda_t_norm = lambda_curr_norm + alpha * dlambda_seg

    # Bridge mean and optional Gaussian noise
    mu_t = (1.0 - alpha) * x_curr_gt + alpha * x_next_gt
    sigma = float(getattr(config, 'multi_pref_tfm_sigma', 0.0))
    x_t = mu_t + (sigma * torch.sqrt(torch.clamp(alpha * (1.0 - alpha), min=0.0))) * torch.randn_like(mu_t) if sigma > 0 else mu_t

    # Remaining distance to endpoint
    dlambda_remain = torch.clamp(lambda_next_norm - lambda_t_norm, min=1e-8)

    # Target displacement (NOT divided by dlambda)
    dx_target = wrap_angle_difference(x_next_gt - x_t, NPred_Va)

    # Predict velocity at intermediate point
    v_pred = model.predict_vec(scene, x_t, lambda_t_norm, lambda_t_norm)
    dlambda_step = dlambda_remain
    x_base = x_t 

    # ==================== Euler Integration ====================
    delta = dlambda_step * v_pred
    x_pred = x_base + delta

    # ==================== Loss Computation ====================
    # Displacement loss: ||delta - dx_target||²
    loss_displacement = torch.mean((delta - dx_target) ** 2)
    
    # Compute relative error for better interpretability (for debugging)
    with torch.no_grad():
        dx_target_norm = torch.norm(dx_target, dim=-1, keepdim=True).clamp(min=1e-8)
        relative_error = torch.mean(torch.norm(delta - dx_target, dim=-1) / dx_target_norm.squeeze())
        # Store for logging (will be retrieved in training loop)
        _train_trajectory_step.last_relative_error = relative_error.item()
        _train_trajectory_step.last_dx_target_mean = torch.mean(torch.abs(dx_target)).item()
    
    # Optional endpoint loss (disabled by default, beta=0)
    loss_alpha = config.multi_pref_loss_alpha
    loss_beta = config.multi_pref_loss_beta
    
    if loss_beta > 0:
        dx_pred = wrap_angle_difference(x_pred - x_next_gt, NPred_Va)
        loss_endpoint = F.smooth_l1_loss(dx_pred, torch.zeros_like(dx_pred))
        return loss_alpha * loss_displacement + loss_beta * loss_endpoint
    
    return loss_alpha * loss_displacement


# ==================== Main Function ====================

def main(debug=False):
    """Main function for multi-preference supervised training."""
    from unified_eval import MultiPreferencePredictor, build_ctx_from_multi_preference, evaluate_unified

    
    config = get_multi_preference_config()
    
    print("=" * 60)
    print("DeepOPF-V: Multi-Preference Training (Trajectory Mode)")
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
    from net_utiles import FM
    
    # Import StandardMLPAnchor from train_multi_preference_tfm_lmlp.py
    from mlp_anchor import load_standard_mlp_anchor
    
    # Only support Flow models for trajectory training
    if model_type not in ['rectified', 'gaussian', 'conditional', 'interpolation']:
        raise ValueError(f"TFM training only supports Flow models (rectified, etc.), got: {model_type}")
    
    # Create Flow model
    model = FM(network='preference_aware_mlp', input_dim=input_dim, output_dim=output_dim,
               hidden_dim=config.hidden_dim, num_layers=config.num_layers,
               time_step=config.time_step, output_norm=False, pred_type='velocity', pref_dim=pref_dim)
    
    # Load Standard MLP as anchor generator
    # This is more appropriate than VAE because:
    # - Standard MLP is trained to predict the cost-optimal solution (lc=0)
    # - The trajectory starts from lc=0 and moves toward higher carbon weights
    pretrain_model = load_standard_mlp_anchor(config, sys_data, multi_pref_data, config.device)
    print(f"  Using Standard MLP as anchor (predicts lc=0 cost-optimal solution)")
    
    model.to(config.device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}") 
             
    model, _, _ = train_multi_preference(config, model, multi_pref_data, sys_data, config.device,
                                            model_type=model_type, pretrain_model=pretrain_model)  


# ==================== Model Path Configuration ====================
# Note: Standard MLP is now used as anchor generator instead of VAE.
# The Standard MLP model is loaded from main_part/saved_models/ automatically.
# See load_standard_mlp_anchor() in train_multi_preference_tfm_lmlp.py for details.


if __name__ == "__main__": 
    main()
