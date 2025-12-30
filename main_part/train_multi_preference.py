#!/usr/bin/env python
# coding: utf-8
"""
Multi-Preference Supervised Training for DeepOPF-V
Trains preference-conditioned models for multi-objective OPF.

Supports: simple, vae, flow, generative_vae, diffusion

Author: Peng Yue
Date: December 2025

Usage:
    MODEL_TYPE=flow python train_multi_preference.py
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
from models import NetV, NetVm, NetVa, get_available_model_types
from data_loader import load_all_data, load_multi_preference_dataset, create_multi_preference_dataloader


# ==================== Multi-Preference Configuration ====================

class MultiPreferenceConfig(BaseConfig):
    """Configuration for multi-preference supervised training."""
    
    def __init__(self):
        super().__init__()
        
        # ==================== Multi-Preference Training ====================
        self.use_multi_objective_supervised = os.environ.get('MULTI_PREF_SUPERVISED', 'True').lower() == 'true'
        self.multi_pref_dataset_path = os.path.join(
            os.path.dirname(_SCRIPT_DIR), 'saved_data', 'multi_preference_solutions', 'fully_covered_dataset.pt'
        )
        
        # Training parameters
        self.multi_pref_epochs = int(os.environ.get('MULTI_PREF_EPOCHS', '4500'))
        self.multi_pref_lr = float(os.environ.get('MULTI_PREF_LR', '1e-4'))
        self.multi_pref_flow_steps = int(os.environ.get('MULTI_PREF_FLOW_STEPS', '10'))
        self.multi_pref_batch_size = int(os.environ.get('MULTI_PREF_BATCH_SIZE', '32'))
        
        # Validation split
        self.multi_pref_val_ratio = float(os.environ.get('MULTI_PREF_VAL_RATIO', '0.2'))
        self.multi_pref_random_seed = int(os.environ.get('MULTI_PREF_RANDOM_SEED', '42'))
        
        # Model architecture
        self.multi_pref_hidden_dim = int(os.environ.get('MULTI_PREF_HIDDEN_DIM', '512'))
        self.multi_pref_num_layers = int(os.environ.get('MULTI_PREF_NUM_LAYERS', '5'))
        self.multi_pref_flow_type = self.model_type
        self.multi_pref_training_mode = os.environ.get('MULTI_PREF_TRAINING_MODE', 'preference_trajectory')
        
        # Loss weights
        self.multi_pref_loss_alpha = float(os.environ.get('MULTI_PREF_LOSS_ALPHA', '1.0'))
        self.multi_pref_loss_beta = float(os.environ.get('MULTI_PREF_LOSS_BETA', '1000.0'))
        self.multi_pref_loss_gamma = float(os.environ.get('MULTI_PREF_LOSS_GAMMA', '0.0'))
        
        # Scheduled sampling
        self.multi_pref_scheduled_sampling = os.environ.get('MULTI_PREF_SCHEDULED_SAMPLING', 'True').lower() == 'true'
        self.multi_pref_scheduled_sampling_p_min = float(os.environ.get('MULTI_PREF_SCHEDULED_SAMPLING_P_MIN', '0.2'))
        
        # Multi-step rollout
        self.multi_pref_rollout_horizon = int(os.environ.get('MULTI_PREF_ROLLOUT_HORIZON', '4'))
        self.multi_pref_rollout_use_rk2 = os.environ.get('MULTI_PREF_ROLLOUT_USE_RK2', 'True').lower() == 'true'
        
        # Preference conditioning
        self.pref_dim = 1
        self.multi_pref_use_vae_anchor = os.environ.get('MULTI_PREF_VAE_ANCHOR', 'True').lower() == 'true'
        
        # ==================== Linearized VAE (Stage 1) ====================
        self.skip_vae_training = os.environ.get('SKIP_VAE_TRAINING', 'False').lower() == 'true'
        self.pretrained_vae_path = os.environ.get('PRETRAINED_VAE_PATH', "main_part/saved_models/linearized_vae_epoch2900.pth")
        
        self.linearized_vae_epochs = int(os.environ.get('LINEARIZED_VAE_EPOCHS', '5000'))
        self.linearized_vae_batch_size = int(os.environ.get('LINEARIZED_VAE_BATCH_SIZE', '32'))
        self.linearized_vae_lr = float(os.environ.get('LINEARIZED_VAE_LR', '5e-4'))
        self.linearized_vae_latent_dim = int(os.environ.get('LINEARIZED_VAE_LATENT_DIM', '64'))
        self.linearized_vae_hidden_dim = int(os.environ.get('LINEARIZED_VAE_HIDDEN_DIM', '512'))
        self.linearized_vae_num_layers = int(os.environ.get('LINEARIZED_VAE_NUM_LAYERS', '4'))
        self.linearized_vae_weight_decay = float(os.environ.get('LINEARIZED_VAE_WEIGHT_DECAY', '1e-5'))
        
        # VAE loss weights
        self.linearized_vae_alpha_kl = float(os.environ.get('LINEARIZED_VAE_ALPHA_KL', '0.0001'))
        self.linearized_vae_beta_1d = float(os.environ.get('LINEARIZED_VAE_BETA_1D', '0.01'))
        self.linearized_vae_gamma_order = float(os.environ.get('LINEARIZED_VAE_GAMMA_ORDER', '0.005'))
        self.linearized_vae_delta_ngt = float(os.environ.get('LINEARIZED_VAE_DELTA_NGT', '0.0005'))
        self.linearized_vae_n_pref_samples = int(os.environ.get('LINEARIZED_VAE_N_PREF_SAMPLES', '12'))
        self.linearized_vae_lr_decay = [500, 0.8]
        self.linearized_vae_print_interval = int(os.environ.get('LINEARIZED_VAE_PRINT_INTERVAL', '100'))
        self.linearized_vae_save_interval = int(os.environ.get('LINEARIZED_VAE_SAVE_INTERVAL', '500'))
        
        # ==================== Latent Flow Model (Stage 2) ====================
        self.skip_flow_training = os.environ.get('SKIP_FLOW_TRAINING', 'False').lower() == 'true'
        self.pretrained_flow_path = os.environ.get('PRETRAINED_FLOW_PATH', "main_part/saved_models/latent_flow_epoch1600.pth")
        
        self.latent_flow_epochs = int(os.environ.get('LATENT_FLOW_EPOCHS', '2000'))
        self.latent_flow_batch_size = int(os.environ.get('LATENT_FLOW_BATCH_SIZE', '64'))
        self.latent_flow_lr = float(os.environ.get('LATENT_FLOW_LR', '5e-4'))
        self.latent_flow_hidden_dim = int(os.environ.get('LATENT_FLOW_HIDDEN_DIM', '512'))
        self.latent_flow_num_layers = int(os.environ.get('LATENT_FLOW_NUM_LAYERS', '4'))
        self.latent_flow_weight_decay = float(os.environ.get('LATENT_FLOW_WEIGHT_DECAY', '1e-5'))
        
        # Latent flow loss weights
        self.latent_flow_n_pair_samples = int(os.environ.get('LATENT_FLOW_N_PAIR_SAMPLES', '8'))
        self.latent_flow_use_rollout = os.environ.get('LATENT_FLOW_USE_ROLLOUT', 'True').lower() == 'true'
        self.latent_flow_gamma_rollout = float(os.environ.get('LATENT_FLOW_GAMMA_ROLLOUT', '0.1'))
        self.latent_flow_rollout_horizon = int(os.environ.get('LATENT_FLOW_ROLLOUT_HORIZON', '5'))
        self.latent_flow_rollout_use_heun = os.environ.get('LATENT_FLOW_ROLLOUT_USE_HEUN', 'True').lower() == 'true'
        self.latent_flow_beta_z1 = float(os.environ.get('LATENT_FLOW_BETA_Z1', '1.0'))
        self.latent_flow_alpha_fm = float(os.environ.get('LATENT_FLOW_ALPHA_FM', '1.0'))
        self.latent_flow_n_fm_samples = int(os.environ.get('LATENT_FLOW_N_FM_SAMPLES', '8'))
        self.latent_flow_fm_min_gap = int(os.environ.get('LATENT_FLOW_FM_MIN_GAP', '1'))
        self.latent_flow_fm_max_gap = int(os.environ.get('LATENT_FLOW_FM_MAX_GAP', '20'))
        self.latent_flow_lr_decay = [100, 0.9]
        self.latent_flow_print_interval = int(os.environ.get('LATENT_FLOW_PRINT_INTERVAL', '50'))
        self.latent_flow_save_interval = int(os.environ.get('LATENT_FLOW_SAVE_INTERVAL', '100'))
        self.latent_flow_inf_steps = int(os.environ.get('LATENT_FLOW_INF_STEPS', '20'))
        self.latent_flow_inf_method = os.environ.get('LATENT_FLOW_INF_METHOD', 'heun')
        
        # ==================== Generative VAE ====================
        self.generative_vae_n_samples = int(os.environ.get('GENERATIVE_VAE_N_SAMPLES', '5'))
        self.generative_vae_n_prefs = int(os.environ.get('GENERATIVE_VAE_N_PREFS', '3'))
        self.generative_vae_alpha_kl = float(os.environ.get('GENERATIVE_VAE_ALPHA_KL', '0.001'))
        self.generative_vae_delta_feas = float(os.environ.get('GENERATIVE_VAE_DELTA_FEAS', '0.01'))
        self.generative_vae_eta_obj = float(os.environ.get('GENERATIVE_VAE_ETA_OBJ', '0.001'))
        self.generative_vae_warmup_epochs = int(os.environ.get('GENERATIVE_VAE_WARMUP_EPOCHS', '100'))
        self.generative_vae_ramp_epochs = int(os.environ.get('GENERATIVE_VAE_RAMP_EPOCHS', '100'))
        self.generative_vae_free_bits = float(os.environ.get('GENERATIVE_VAE_FREE_BITS', '0.1'))
        self.generative_vae_feas_mode = os.environ.get('GENERATIVE_VAE_FEAS_MODE', 'softmin')
        self.generative_vae_tau_start = float(os.environ.get('GENERATIVE_VAE_TAU_START', '0.5'))
        self.generative_vae_tau_end = float(os.environ.get('GENERATIVE_VAE_TAU_END', '0.1'))
        self.generative_vae_lambda_sample = float(os.environ.get('GENERATIVE_VAE_LAMBDA_SAMPLE', '0.3'))
        self.generative_vae_ngt_chunk_size = int(os.environ.get('GENERATIVE_VAE_NGT_CHUNK_SIZE', '1024'))
        self.generative_vae_max_grad_norm = float(os.environ.get('GENERATIVE_VAE_MAX_GRAD_NORM', '1.0'))
        self.generative_vae_feas_threshold = float(os.environ.get('GENERATIVE_VAE_FEAS_THRESHOLD', '0.01'))
        self.generative_vae_epochs = int(os.environ.get('GENERATIVE_VAE_EPOCHS', '4000'))
        self.generative_vae_lr = float(os.environ.get('GENERATIVE_VAE_LR', '1e-4'))
        self.generative_vae_batch_size = int(os.environ.get('GENERATIVE_VAE_BATCH_SIZE', '32'))
        self.generative_vae_print_interval = int(os.environ.get('GENERATIVE_VAE_PRINT_INTERVAL', '10'))
        self.generative_vae_save_interval = int(os.environ.get('GENERATIVE_VAE_SAVE_INTERVAL', '200'))
        self.generative_vae_lambda_feas = float(os.environ.get('GENERATIVE_VAE_LAMBDA_FEAS', '0.1'))
        self.generative_vae_min_logvar = float(os.environ.get('GENERATIVE_VAE_MIN_LOGVAR', '-4.0'))
        self.generative_vae_n_feas_samples = int(os.environ.get('GENERATIVE_VAE_N_FEAS_SAMPLES', '4'))
        
        # ==================== VAE Evaluation ====================
        self.vae_best_of_k = int(os.environ.get('VAE_BEST_OF_K', '64'))
        self.vae_use_mean = os.environ.get('VAE_USE_MEAN', '0').lower() in ('1', 'true', 'yes')
        self.vae_model_path = os.environ.get('VAE_MODEL_PATH', "generative_vae_simple_final.pth")
        self.vae_selection_mode = os.environ.get('VAE_SELECTION_MODE', 'constraint')
        self.vae_feasibility_threshold = float(os.environ.get('VAE_FEASIBILITY_THRESHOLD', '0.01'))
        self.vae_best_of_k_chunk_size = int(os.environ.get('VAE_BEST_OF_K_CHUNK_SIZE', '1024'))
        self.vae_use_preference_aware = True
        self.vae_beta = 1.0
        
        # ==================== Shared with NGT (for model creation) ====================
        self.ngt_hidden_units = 1
        self.ngt_khidden = np.array([64, 224], dtype=int)
        self.ngt_flow_hidden_dim = int(os.environ.get('NGT_FLOW_HIDDEN_DIM', '144'))
        self.ngt_flow_num_layers = int(os.environ.get('NGT_FLOW_NUM_LAYERS', '2'))
        self.time_step = 1000
        self.hidden_dim = 512
        self.num_layers = 5
        self.weight_decay = 1e-6
        self.p_epoch = 10
        self.s_epoch = 800
        
    def print_config(self):
        """Print configuration summary."""
        super().print_config()
        print(f"\n[Multi-Preference Training Config]")
        print(f"  Epochs: {self.multi_pref_epochs}")
        print(f"  Learning rate: {self.multi_pref_lr}")
        print(f"  Batch size: {self.multi_pref_batch_size}")
        print(f"  Training mode: {self.multi_pref_training_mode}")
        print(f"  VAE latent dim: {self.linearized_vae_latent_dim}")
        print(f"  VAE Best-of-K: {self.vae_best_of_k} (use_mean={self.vae_use_mean})")


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

def train_generative_vae(config, multi_pref_data, sys_data, device, ngt_loss_fn):
    """Train Generative VAE with feasibility loss for Best-of-K sampling."""
    from flow_model.generative_vae_utils import lambda_to_key, compute_ngt_loss_chunked_differentiable
    from flow_model.net_utiles import VAE
    
    print('=' * 70)
    print('Training Generative VAE (with Feasibility Loss)')
    print('=' * 70)
    
    # Extract data
    x_train = multi_pref_data['x_train'].to(device)
    y_train_by_pref = multi_pref_data['y_train_by_pref']
    lambda_values = multi_pref_data['lambda_carbon_values']
    n_train, input_dim, output_dim = multi_pref_data['n_train'], multi_pref_data['input_dim'], multi_pref_data['output_dim']
    lc_max = max(lambda_values) if lambda_values else 1.0
    
    y_train_device = {lambda_to_key(lc): y.to(device) for lc, y in y_train_by_pref.items()}
    
    # Hyperparameters
    num_epochs = config.generative_vae_epochs
    batch_size = config.generative_vae_batch_size
    lr = config.generative_vae_lr
    latent_dim = config.linearized_vae_latent_dim
    hidden_dim = config.multi_pref_hidden_dim
    num_layers = config.multi_pref_num_layers
    vae_beta = getattr(config, 'multi_pref_vae_beta', 0.001)
    lambda_feas = config.generative_vae_lambda_feas
    min_logvar = config.generative_vae_min_logvar
    n_feas_samples = config.generative_vae_n_feas_samples
    warmup_epochs = config.generative_vae_warmup_epochs
    
    print(f"\nConfig: epochs={num_epochs}, batch={batch_size}, lr={lr}")
    print(f"VAE: hidden={hidden_dim}, latent={latent_dim}, layers={num_layers}")
    print(f"Loss: beta={vae_beta}, lambda_feas={lambda_feas}, warmup={warmup_epochs}")
    
    vae = VAE(network='preference_aware_mlp', input_dim=input_dim, output_dim=output_dim,
              hidden_dim=hidden_dim, num_layers=num_layers, latent_dim=latent_dim,
              output_act='none', pred_type='v', pref_dim=1).to(device)
    print(f"Parameters: {sum(p.numel() for p in vae.parameters()):,}")
    
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    from torch.utils.data import DataLoader, TensorDataset
    dataloader = DataLoader(TensorDataset(x_train, torch.arange(n_train)), batch_size=batch_size, shuffle=True)
    
    ngt_loss_fn.cache_to_gpu(device)
    carbon_scale = getattr(ngt_loss_fn.params, 'carbon_scale', 30.0)
    
    losses, skipped = [], 0
    start_time = time.process_time()
    print_interval = config.generative_vae_print_interval
    save_interval = config.generative_vae_save_interval
    
    for epoch in range(num_epochs):
        vae.train()
        epoch_loss, epoch_rec, epoch_kl, epoch_feas, n_batches = 0, 0, 0, 0, 0
        
        for batch_x, batch_idx in dataloader:
            batch_x, batch_idx = batch_x.to(device), batch_idx.to(device)
            B = batch_x.shape[0]
            
            lc = random.choice(lambda_values)
            batch_y = y_train_device[lambda_to_key(lc)][batch_idx]
            pref = torch.full((B, 1), lc / lc_max, device=device)
            
            optimizer.zero_grad()
            
            try:
                y_pred, mean, logvar = vae.encoder_decode(batch_x, batch_y, pref=pref)
                logvar = torch.clamp(logvar, min=min_logvar)
                
                L_rec = criterion(y_pred, batch_y)
                L_kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
                
                L_feas = torch.tensor(0.0, device=device)
                if epoch >= warmup_epochs and lambda_feas > 0:
                    K = n_feas_samples
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn(K, B, latent_dim, device=device)
                    z_samples = mean.unsqueeze(0) + std.unsqueeze(0) * eps
                    z_flat = z_samples.reshape(K * B, latent_dim)
                    
                    scene_exp = batch_x.unsqueeze(0).expand(K, -1, -1).reshape(K * B, -1)
                    pref_exp = pref.unsqueeze(0).expand(K, -1, -1).reshape(K * B, -1)
                    y_samples = vae.Decoder(scene_exp, z_flat, pref=pref_exp)
                    
                    lc_ratio = lc / lc_max
                    pref_raw = torch.tensor([[1.0 - lc_ratio, lc_ratio]], device=device).expand(K * B, -1)
                    
                    loss_dict = compute_ngt_loss_chunked_differentiable(
                        ngt_loss_fn.params, y_samples, scene_exp, pref_raw, 
                        chunk_size=1024, carbon_scale=carbon_scale
                    )
                    L_feas = loss_dict['constraint_scaled'].mean()
                
                loss = L_rec + vae_beta * L_kl + lambda_feas * L_feas
                
                if torch.isnan(loss) or torch.isinf(loss):
                    skipped += 1; continue
                
                loss.backward()
                
                if any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()) 
                       for p in vae.parameters()):
                    skipped += 1; optimizer.zero_grad(); continue
                
                torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_rec += L_rec.item()
                epoch_kl += L_kl.item()
                epoch_feas += L_feas.item()
                n_batches += 1
                
            except RuntimeError as e:
                print(f"[WARNING] Epoch {epoch}: {e}")
                skipped += 1; optimizer.zero_grad(); continue
        
        if n_batches > 0:
            losses.append(epoch_loss / n_batches)
            
            if epoch % print_interval == 0:
                feas_status = "ON" if epoch >= warmup_epochs else "OFF"
                print(f"Epoch {epoch:4d} | Loss: {losses[-1]:.4f} | "
                      f"rec: {epoch_rec/n_batches:.4f} | kl: {epoch_kl/n_batches:.4f} | "
                      f"feas: {epoch_feas/n_batches:.4f} ({feas_status})")
        
        if epoch > 0 and epoch % save_interval == 0:
            os.makedirs(config.model_save_dir, exist_ok=True)
            torch.save(vae.state_dict(), f'{config.model_save_dir}/generative_vae_simple_epoch{epoch}.pth')
    
    time_train = time.process_time() - start_time
    print(f"\nCompleted in {time_train:.2f}s, skipped={skipped}")
    
    os.makedirs(config.model_save_dir, exist_ok=True)
    final_path = f'{config.model_save_dir}/generative_vae_simple_final.pth'
    torch.save(vae.state_dict(), final_path)
    print(f"Saved: {final_path}")
    
    return vae, losses, {}


def train_multi_preference(config, model, multi_pref_data, sys_data, device,
                           model_type='simple', pretrain_model=None):
    """Train preference-conditioned model for multi-objective OPF."""
    
    print('=' * 60)
    print(f'Training Multi-Preference Model - Type: {model_type}')
    print('=' * 60)
    
    x_train = multi_pref_data['x_train'].to(device)
    y_train_by_pref = {lc: y.to(device) for lc, y in multi_pref_data['y_train_by_pref'].items()}
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
            B = batch_x.shape[0]
            optimizer.zero_grad()
            
            if training_mode == 'preference_trajectory' and model_type in ['rectified', 'flow']:
                loss = _train_trajectory_step(
                    model, batch_x, batch_idx, y_train_by_pref, lambda_sorted, lambda_norm,
                    NPred_Va, device, config, epoch, num_epochs
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


def _train_trajectory_step(model, batch_x, batch_idx, y_train_by_pref, lambda_sorted, lambda_norm,
                           NPred_Va, device, config, epoch, num_epochs):
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
    
    alpha = config.multi_pref_loss_alpha
    beta = config.multi_pref_loss_beta
    
    loss_v = torch.mean((v_pred - v_target) ** 2)
    x_pred = x_curr_gt + dlambda * v_pred
    dx_pred = wrap_angle_difference(x_pred - x_next_gt, NPred_Va)
    loss_l1 = torch.mean(dx_pred ** 2)
    
    return alpha * loss_v + beta * loss_l1


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
        
    elif model_type in ['rectified', 'flow', 'gaussian', 'conditional', 'interpolation']:
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
    _, _, BRANFT = load_all_data(config)
    
    input_dim = multi_pref_data['input_dim']
    output_dim = multi_pref_data['output_dim']
    pref_dim = config.pref_dim
    Vscale, Vbias = multi_pref_data['Vscale'], multi_pref_data['Vbias']
    
    print(f"\nDimensions: input={input_dim}, output={output_dim}, pref={pref_dim}")
    
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'flow_model'))
    from net_utiles import FM, VAE, DM
    
    model, pretrain_model = None, None
    
    if model_type == 'simple':
        model = NetV(input_dim + pref_dim, output_dim, config.ngt_hidden_units, config.ngt_khidden, Vscale, Vbias)
        
    elif model_type == 'vae':
        vae_args = dict(output_dim=output_dim, hidden_dim=config.multi_pref_hidden_dim,
                        num_layers=config.multi_pref_num_layers,
                        latent_dim=config.linearized_vae_latent_dim,
                        output_act=None, pred_type='node', use_cvae=True)
        if config.vae_use_preference_aware:
            model = VAE(network='preference_aware_mlp', input_dim=input_dim, pref_dim=pref_dim, **vae_args)
        else:
            model = VAE(network='mlp', input_dim=input_dim + pref_dim, **vae_args)
            
    elif model_type in ['rectified', 'flow', 'gaussian', 'conditional', 'interpolation']:
        model = FM(network='preference_aware_mlp', input_dim=input_dim, output_dim=output_dim,
                   hidden_dim=config.ngt_flow_hidden_dim, num_layers=config.ngt_flow_num_layers,
                   time_step=config.time_step, output_norm=False, pred_type='velocity', pref_dim=pref_dim)
                   
    elif model_type == 'diffusion':
        model = DM(network='mlp', input_dim=input_dim + pref_dim, output_dim=output_dim,
                   hidden_dim=config.hidden_dim, num_layers=config.num_layers,
                   time_step=config.time_step, output_norm=False, pred_type='node')
                   
    elif model_type == 'generative_vae':
        model = None
        print(f"\n[Generative VAE] Model created during training")
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    if model: 
        model.to(config.device)
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    load_pretrained = config.load_pretrained_model
    
    if load_pretrained:
        print("\n[Loading Pretrained Model]")
        if model_type == 'generative_vae':
            path = f'{config.model_save_dir}/generative_vae_simple_final.pth'
            model = VAE(network='preference_aware_mlp', input_dim=input_dim, output_dim=output_dim,
                        hidden_dim=config.multi_pref_hidden_dim, num_layers=config.multi_pref_num_layers,
                        latent_dim=config.linearized_vae_latent_dim,
                        output_act='none', pred_type='v', pref_dim=pref_dim).to(config.device)
        else:
            path = f'{config.model_save_dir}/model_multi_pref_{model_type}_final.pth'
        
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=config.device, weights_only=True))
            model.eval()
            print(f"  Loaded: {path}")
        else:
            raise FileNotFoundError(f"Model not found: {path}")
            
    elif not debug:
        if model_type == 'generative_vae':
            from deepopf_ngt_loss import DeepOPFNGTLoss
            ngt_loss_fn = DeepOPFNGTLoss(sys_data, config)
            model, _, _ = train_generative_vae(config, multi_pref_data, sys_data, config.device, ngt_loss_fn)
        else:
            model, _, _ = train_multi_preference(config, model, multi_pref_data, sys_data, config.device,
                                                  model_type=model_type, pretrain_model=pretrain_model)
    else:
        print("\n[Debug Mode] Loading model...")
        path = f'{config.model_save_dir}/model_multi_pref_{model_type}_final.pth'
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=config.device, weights_only=True))
    
    # Evaluation
    print("\n" + "=" * 80)
    print("Evaluation")
    print("=" * 80)
    
    test_lambdas = [0.0, 25.0, 50.0, 80.0, 90.0]
    results_all = {}
    
    vae_best_of_k = config.vae_best_of_k
    vae_use_mean = config.vae_use_mean
    vae_selection_mode = config.vae_selection_mode
    
    ngt_loss_fn = None
    if model_type == 'vae' and vae_best_of_k > 1 and not vae_use_mean:
        from deepopf_ngt_loss import DeepOPFNGTLoss
        ngt_loss_fn = DeepOPFNGTLoss(sys_data, config)
        ngt_loss_fn.cache_to_gpu(config.device)
    
    for lc in test_lambdas:
        print(f"\n--- lambda_carbon = {lc:.2f} ---")
        
        ctx = build_ctx_from_multi_preference(config, sys_data, multi_pref_data, BRANFT, config.device, lambda_carbon=lc)
        
        predictor = MultiPreferencePredictor(
            model=model, multi_pref_data=multi_pref_data, lambda_carbon=lc,
            model_type='vae' if model_type == 'generative_vae' else model_type,
            num_flow_steps=config.multi_pref_flow_steps,
            training_mode=config.multi_pref_training_mode,
            ngt_loss_fn=ngt_loss_fn, vae_n_samples=vae_best_of_k,
            vae_use_mean=vae_use_mean, vae_selection_mode=vae_selection_mode
        )
        
        results = evaluate_unified(ctx, predictor, apply_post_processing=True, verbose=True)
        results_all[lc] = results
    
    print("\n" + "=" * 80)
    print("Evaluation Complete")
    print("=" * 80)
    
    return results_all


if __name__ == "__main__":
    debug = bool(int(os.environ.get('DEBUG', '0')))
    main(debug=debug)
