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
    MODEL_TYPE=vae python train_multi_preference.py
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


# ==================== Configuration ====================

class MultiPreferenceConfig(BaseConfig):
    """Configuration for multi-preference supervised training."""
    
    def __init__(self):
        super().__init__()
        
        # Dataset path
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
        
        # Model architecture
        self.hidden_dim = int(os.environ.get('HIDDEN_DIM', '128'))
        self.num_layers = int(os.environ.get('NUM_LAYERS', '2'))
        self.latent_dim = int(os.environ.get('LATENT_DIM', '64'))
        self.time_step = 1000
        
        # Simple model (NetV) structure
        self.ngt_hidden_units = 1
        self.ngt_khidden = np.array([64, 224], dtype=int)
        
        # Flow type
        self.multi_pref_flow_type = self.model_type
        
        # Preference conditioning
        self.pref_dim = 1
        
        # VAE settings
        self.vae_best_of_k = int(os.environ.get('VAE_BEST_OF_K', '32'))
        self.vae_use_mean = os.environ.get('VAE_USE_MEAN', '0').lower() in ('1', 'true', 'yes')
        self.vae_selection_mode = os.environ.get('VAE_SELECTION_MODE', 'constraint')
        self.vae_use_preference_aware = True
        self.vae_beta = 1.0
        
        # Flow Best-of-K settings
        self.flow_best_of_k = int(os.environ.get('FLOW_BEST_OF_K', '32'))
        self.flow_selection_mode = os.environ.get('FLOW_SELECTION_MODE', 'constraint')
        
        # Training control
        self.weight_decay = 1e-6
        self.p_epoch = 10
        self.s_epoch = 800
        
        # CBF-QP Post-Processing (Inference)
        self.use_cbf_qp_post = os.environ.get('USE_CBF_QP_POST', '1') == '1'
        self.post_process_method = os.environ.get('POST_PROCESS_METHOD', '').strip().lower()
        if self.use_cbf_qp_post and not self.post_process_method:
            self.post_process_method = 'cbf_qp'
        self.cbf_beta = float(os.environ.get('CBF_BETA', '1.0'))
        
    def print_config(self):
        """Print configuration summary."""
        super().print_config()
        print(f"\n[Training Config]")
        print(f"  Epochs: {self.multi_pref_epochs}")
        print(f"  Learning rate: {self.multi_pref_lr}")
        print(f"  Batch size: {self.multi_pref_batch_size}")
        print(f"\n[Model Architecture]")
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Num layers: {self.num_layers}")
        print(f"  Latent dim (VAE): {self.latent_dim}")
        print(f"  Simple khidden: {self.ngt_khidden}")
        print(f"\n[VAE Evaluation]")
        print(f"  Best-of-K: {self.vae_best_of_k} (use_mean={self.vae_use_mean})")
        print(f"\n[Flow Evaluation]")
        print(f"  Best-of-K: {self.flow_best_of_k} (mode={self.flow_selection_mode})")
        print(f"\n[CBF-QP Configuration]")
        print(f"  Post-processing: use_cbf_qp_post={self.use_cbf_qp_post}, method='{self.post_process_method}'")
        print(f"  Inference beta: {self.cbf_beta}")


def get_multi_preference_config():
    """Get multi-preference training configuration."""
    return MultiPreferenceConfig()


# ==================== Training Functions ====================

def train_multi_preference(config, model, multi_pref_data, device, model_type='simple', pretrain_model=None):
    """Train preference-conditioned model for multi-objective OPF."""
    
    print('=' * 60)
    print(f'Training Multi-Preference Model - Type: {model_type}')
    print('=' * 60)
    
    y_train_by_pref = {lc: y.to(device) for lc, y in multi_pref_data['y_train_by_pref'].items()}
    lambda_values = multi_pref_data['lambda_carbon_values']
    n_train = multi_pref_data['n_train']
    
    print(f"\nData: {n_train} samples, {len(lambda_values)} preferences")
    print(f"Lambda range: [{lambda_values[0]:.2f}, {lambda_values[-1]:.2f}]")
    
    num_epochs = config.multi_pref_epochs
    lr = config.multi_pref_lr
    lc_max = max(lambda_values) if max(lambda_values) > 0 else 1.0
    vae_beta = config.vae_beta
    
    print(f"\nConfig: epochs={num_epochs}, lr={lr}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    dataloader = create_multi_preference_dataloader(multi_pref_data, config, shuffle=True)
    criterion = nn.MSELoss()
    
    losses = []
    start_time = time.process_time()
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss, num_batches = 0.0, 0
        
        for batch_x, batch_idx in dataloader:
            batch_x, batch_idx = batch_x.to(device), batch_idx.to(device)
            optimizer.zero_grad()
            
            loss = _train_step(
                model, batch_x, batch_idx, y_train_by_pref, lambda_values, lc_max,
                model_type, pretrain_model, criterion, vae_beta, device, config
            )
            
            if loss is None:
                continue
            
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


def _train_step(model, batch_x, batch_idx, y_train_by_pref, lambda_values, lc_max,
                model_type, pretrain_model, criterion, vae_beta, device, config):
    """Single training step."""
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
                if hasattr(pretrain_model, 'pref_dim'):
                    z_batch = pretrain_model(batch_x, use_mean=True, pref=pref)
                else:
                    z_batch = pretrain_model(torch.cat([batch_x, pref], dim=1), use_mean=True)
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


# ==================== Model Creation ====================

def create_model(config, input_dim, output_dim, pref_dim, Vscale, Vbias):
    """Create model based on model_type."""
    model_type = config.model_type
    
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'flow_model'))
    from net_utiles import FM, VAE, DM
    
    if model_type == 'simple':
        model = NetV(input_dim + pref_dim, output_dim, config.ngt_hidden_units, config.ngt_khidden, Vscale, Vbias)
        
    elif model_type == 'vae':
        vae_args = dict(
            output_dim=output_dim, hidden_dim=config.hidden_dim,
            num_layers=config.num_layers, latent_dim=config.latent_dim,
            output_act=None, pred_type='node', use_cvae=True
        )
        if config.vae_use_preference_aware:
            model = VAE(network='preference_aware_mlp', input_dim=input_dim, pref_dim=pref_dim, **vae_args)
        else:
            model = VAE(network='mlp', input_dim=input_dim + pref_dim, **vae_args)
            
    elif model_type in ['rectified', 'gaussian', 'conditional', 'interpolation']:
        model = FM(
            network='preference_aware_mlp', input_dim=input_dim, output_dim=output_dim,
            hidden_dim=config.hidden_dim, num_layers=config.num_layers,
            time_step=config.time_step, output_norm=False, pred_type='velocity', pref_dim=pref_dim
        )
                   
    elif model_type == 'diffusion':
        model = DM(
            network='mlp', input_dim=input_dim + pref_dim, output_dim=output_dim,
            hidden_dim=config.hidden_dim, num_layers=config.num_layers,
            time_step=config.time_step, output_norm=False, pred_type='node'
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model


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
    BRANFT = torch.from_numpy(sys_data.branch[:, 0:2] - 1).long()
    
    input_dim = multi_pref_data['input_dim']
    output_dim = multi_pref_data['output_dim']
    pref_dim = config.pref_dim
    
    NPred_Va = multi_pref_data['NPred_Va']
    NPred_Vm = multi_pref_data['NPred_Vm']
    
    # Verify dimensions
    expected_output_dim = NPred_Va + NPred_Vm
    if output_dim != expected_output_dim:
        raise ValueError(f"output_dim mismatch: got {output_dim}, expected {expected_output_dim}")
    
    Vscale = multi_pref_data['Vscale']
    Vbias = multi_pref_data['Vbias']
    
    if len(Vscale) != output_dim:
        raise ValueError(f"Vscale dimension mismatch: got {len(Vscale)}, expected {output_dim}")
    
    print(f"\nDimensions: input={input_dim}, output={output_dim}, pref={pref_dim}")
    print(f"NPred_Va={NPred_Va}, NPred_Vm={NPred_Vm}")
    
    # Create model
    model = create_model(config, input_dim, output_dim, pref_dim, Vscale, Vbias)
    model.to(config.device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train or load model
    if not debug:
        model, _, _ = train_multi_preference(config, model, multi_pref_data, config.device, model_type=model_type)
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
    
    # Create NGT loss function if Best-of-K is enabled
    ngt_loss_fn = None
    need_ngt_loss = (model_type == 'vae' and vae_best_of_k > 1 and not vae_use_mean) or \
                   (model_type in ['rectified', 'gaussian', 'conditional', 'interpolation'] and flow_best_of_k > 1)
    
    if need_ngt_loss:
        from deepopf_ngt_loss import DeepOPFNGTLoss
        try:
            ngt_loss_fn = DeepOPFNGTLoss(sys_data, config)
            ngt_loss_fn.cache_to_gpu(config.device)
            if model_type in ['rectified', 'gaussian', 'conditional', 'interpolation']:
                print(f"[Eval] Flow Best-of-K: K={flow_best_of_k}, mode={flow_selection_mode}")
            else:
                print(f"[Eval] VAE Best-of-K: K={vae_best_of_k}, mode={vae_selection_mode}")
        except Exception as e:
            print(f"[Warning] Failed to create ngt_loss_fn: {e}")
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
                num_flow_steps=config.multi_pref_flow_steps,
                ngt_loss_fn=ngt_loss_fn, vae_n_samples=vae_best_of_k,
                vae_use_mean=vae_use_mean, vae_selection_mode=vae_selection_mode,
                flow_n_samples=flow_best_of_k, flow_selection_mode=flow_selection_mode
            )
            res[lc] = evaluate_unified(ctx, predictor, apply_post_processing=True, verbose=True)
        return res
    
    # Evaluate on validation set
    print(f"\n{'=' * 40} VALIDATION SET {'=' * 40}")
    results_all['val'] = eval_on_lambdas(test_lambdas)
    
    # Evaluate on training set
    print(f"\n{'=' * 40} TRAINING SET {'=' * 40}")
    orig = {k: multi_pref_data.get(k) for k in ['x_val', 'n_val', 'y_val_by_pref']}
    multi_pref_data['x_val'] = multi_pref_data['x_train']
    multi_pref_data['n_val'] = multi_pref_data['n_train']
    multi_pref_data['y_val_by_pref'] = multi_pref_data['y_train_by_pref']
    results_all['train'] = eval_on_lambdas(test_lambdas)
    for k, v in orig.items():
        if v is not None:
            multi_pref_data[k] = v
    
    print("\n" + "=" * 80)
    print("Evaluation Complete")
    print("=" * 80)
    
    return results_all


if __name__ == "__main__":
    debug = bool(int(os.environ.get('DEBUG', '1')))
    main(debug=debug)
