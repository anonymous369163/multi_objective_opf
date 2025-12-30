#!/usr/bin/env python
# coding: utf-8
"""
Standard Supervised Training for DeepOPF-V
Trains separate Vm and Va models for single-objective OPF.

Author: Peng Yue
Date: December 2025

Usage:
    python train_standard.py
    DEBUG=1 python train_standard.py  # Skip training, load pretrained
"""

import torch
import torch.nn as nn
import time
import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BaseConfig, _SCRIPT_DIR
from models import create_model, get_available_model_types
from data_loader import load_all_data
from unified_eval import build_ctx_from_supervised, SupervisedPredictor, evaluate_unified


# ==================== Standard Training Configuration ====================

class StandardConfig(BaseConfig):
    """Configuration for standard supervised training (separate Vm/Va models)."""
    
    def __init__(self):
        super().__init__()
        
        # ==================== Training Parameters ====================
        self.EpochVm = 1000  # Max epoch for Vm
        self.EpochVa = 1000  # Max epoch for Va
        self.batch_size_training = 50
        self.batch_size_test = 50
        self.s_epoch = 800   # Min epoch for model saving
        self.p_epoch = 10    # Print interval
        
        # ==================== Hyperparameters ====================
        self.Lrm = 1e-3      # Learning rate for Vm
        self.Lra = 1e-3      # Learning rate for Va
        self.weight_decay = 1e-6
        self.learning_rate_decay = [1000, 0.9]  # [step_size, gamma]
        
        # ==================== Generative Model Parameters ====================
        self.latent_dim = 32
        self.time_step = 1000
        self.hidden_dim = 512
        self.num_layers = 5
        self.vae_beta = 1.0
        self.use_vae_anchor = True
        
        # ==================== Model Architecture ====================
        if self.Nbus == 300:
            self.khidden_Vm = np.array([8, 6, 4, 2], dtype=int)
            self.khidden_Va = np.array([8, 6, 4, 2], dtype=int)
        elif self.Nbus == 118:
            self.khidden_Vm = np.array([8, 4, 2], dtype=int)
            self.khidden_Va = np.array([8, 4, 2], dtype=int)
        else:
            self.khidden_Vm = np.array([8, 4, 2], dtype=int)
            self.khidden_Va = np.array([8, 4, 2], dtype=int)
        
        if self.Nbus >= 100:
            self.hidden_units = 128
        elif self.Nbus > 30:
            self.hidden_units = 64
        else:
            self.hidden_units = 16
        
        self.Lm = self.khidden_Vm.shape[0]
        self.La = self.khidden_Va.shape[0]
        
        # ==================== Model Save Paths ====================
        self.nmLm = 'Lm' + ''.join(str(k) for k in self.khidden_Vm)
        self.nmLa = 'La' + ''.join(str(k) for k in self.khidden_Va)
        
        self.PATHVm = f'{self.model_save_dir}/modelvm{self.Nbus}r{self.sys_R}N{self.model_version}{self.nmLm}E{self.EpochVm}.pth'
        self.PATHVa = f'{self.model_save_dir}/modelva{self.Nbus}r{self.sys_R}N{self.model_version}{self.nmLa}E{self.EpochVa}.pth'
        self.PATHVms = f'{self.model_save_dir}/modelvm{self.Nbus}r{self.sys_R}N{self.model_version}{self.nmLm}'
        self.PATHVas = f'{self.model_save_dir}/modelva{self.Nbus}r{self.sys_R}N{self.model_version}{self.nmLa}'
        
        # Pretrained VAE paths (for rectified flow)
        self.pretrain_model_path_vm = os.path.join(self.model_save_dir, 
            f'modelvm{self.Nbus}r{self.sys_R}N{self.model_version}Lm8642_vae_E{self.EpochVm}F1.pth')
        self.pretrain_model_path_va = os.path.join(self.model_save_dir, 
            f'modelva{self.Nbus}r{self.sys_R}N{self.model_version}La8642_vae_E{self.EpochVa}F1.pth')
        
        # Results path
        self.resultnm = (f'{self.results_dir}/res_{self.Nbus}r{self.sys_R}M{self.model_version}H{self.flag_hisv}'
                        f'NT{self.Ntrain}B{self.batch_size_training}'
                        f'Em{self.EpochVm}Ea{self.EpochVa}{self.nmLm}{self.nmLa}rp{self.REPEAT}.mat')
    
    def print_config(self):
        """Print configuration summary."""
        super().print_config()
        print(f"\n[Standard Training Config]")
        print(f"  Epochs (Vm/Va): {self.EpochVm}/{self.EpochVa}")
        print(f"  Learning rate (Vm/Va): {self.Lrm}/{self.Lra}")
        print(f"  Batch size: {self.batch_size_training}")
        print(f"  Hidden layers Vm: {self.khidden_Vm * self.hidden_units}")
        print(f"  Hidden layers Va: {self.khidden_Va * self.hidden_units}")
        if self.model_type in ['vae', 'rectified', 'diffusion']:
            print(f"  Latent dim: {self.latent_dim}")
            print(f"  VAE anchor: {self.use_vae_anchor}")


def get_standard_config():
    """Get standard training configuration."""
    return StandardConfig()


# ==================== Training Functions ====================

def train_model(config, model, optimizer, dataloader, criterion, device, 
                model_name='Vm', num_epochs=None, model_type='simple', 
                pretrain_model=None, scheduler=None, sys_data=None):
    """Unified training function for Vm/Va models."""
    num_epochs = num_epochs or (config.EpochVm if model_name == 'Vm' else config.EpochVa)
    vae_beta = getattr(config, 'vae_beta', 1.0)
    
    print('=' * 60)
    print(f'Training {model_name} Model - Type: {model_type}')
    print('=' * 60)
    
    losses = []
    start_time = time.process_time()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        
        for step, (train_x, train_y) in enumerate(dataloader):
            train_x, train_y = train_x.to(device), train_y.to(device)
            batch_dim = train_x.shape[0]
            optimizer.zero_grad()
            
            loss, y_hat = _compute_loss(
                model, train_x, train_y, batch_dim, device,
                model_type, pretrain_model, criterion, vae_beta,
                config, epoch, num_epochs, sys_data
            )
            
            running_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        losses.append(running_loss)
        if scheduler: scheduler.step()

        if (epoch + 1) % config.p_epoch == 0:
            print(f'Epoch {epoch+1}: Loss = {running_loss:.6f}')
         
        if (epoch + 1) % 100 == 0 and (epoch + 1) >= config.s_epoch:
            save_dir = config.PATHVms if model_name == 'Vm' else config.PATHVas
            save_path = f'{save_dir}_{model_type}_E{epoch+1}.pth'
            torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)
            print(f'  Saved: {save_path}')
            
    time_train = time.process_time() - start_time
    print(f'\n{model_name} training: {time_train:.2f}s ({time_train/60:.2f}min)')
    
    base_path = config.PATHVm if model_name == 'Vm' else config.PATHVa
    final_path = f'{base_path[:-4]}_{model_type}.pth'
    torch.save(model.state_dict(), final_path, _use_new_zipfile_serialization=False)
    print(f'Final saved: {final_path}')
    
    return model, losses, time_train


def _compute_loss(model, train_x, train_y, batch_dim, device, model_type, 
                  pretrain_model, criterion, vae_beta, config, epoch, num_epochs, sys_data):
    """Compute loss based on model type."""
    
    if model_type == 'simple':
        y_hat = model(train_x)
        return criterion(train_y, y_hat), y_hat
        
    elif model_type == 'vae':
        y_pred, mean, logvar = model.encoder_decode(train_x, train_y)
        return model.loss(y_pred, train_y, mean, logvar, beta=vae_beta), y_pred
        
    elif model_type in ['rectified', 'gaussian', 'conditional', 'interpolation']:
        t_batch = torch.rand([batch_dim, 1], device=device)
        if model_type == 'rectified' and pretrain_model:
            with torch.no_grad():
                z_batch = pretrain_model(train_x, use_mean=True)
        else:
            z_batch = torch.randn_like(train_y, device=device)
        yt, vec_target = model.flow_forward(train_y, t_batch, z_batch, model_type)
        vec_pred = model.predict_vec(train_x, yt, t_batch)
        return model.loss(train_y, z_batch, vec_pred, vec_target, model_type), vec_pred + z_batch
        
    elif model_type == 'diffusion':
        t_batch = torch.rand([batch_dim, 1], device=device)
        noise = torch.randn_like(train_y, device=device)
        use_vae_anchor = getattr(config, 'use_vae_anchor', False)
        if use_vae_anchor and pretrain_model:
            with torch.no_grad():
                vae_anchor = pretrain_model(train_x, use_mean=True)
            noise_pred = model.predict_noise_with_anchor(train_x, train_y, t_batch, noise, vae_anchor)
        else:
            noise_pred = model.predict_noise(train_x, train_y, t_batch, noise)
        return model.loss(noise_pred, noise), train_y
        
    elif model_type in ['gan', 'wgan']:
        z_batch = torch.randn([batch_dim, config.latent_dim], device=device)
        y_pred = model(train_x, z_batch)
        loss_d = model.loss_d(train_x, train_y, y_pred)
        loss = loss_d + model.loss_g(train_x, y_pred) if epoch % 5 == 0 else loss_d
        return loss, y_pred
        
    elif model_type == 'consistency_training':
        z_batch = torch.randn_like(train_y, device=device)
        N = math.ceil(math.sqrt((epoch * (1000**2 - 4) / num_epochs) + 4) - 1) + 1
        boundaries = model.kerras_boundaries(1.0, 0.002, N, 1).to(device)
        t_idx = torch.randint(0, N - 1, (batch_dim, 1), device=device)
        if not hasattr(model, 'target_model'):
            model.target_model = model
        return model.loss(train_x, train_y, z_batch, boundaries[t_idx+1], boundaries[t_idx], sys_data, model), train_y
        
    elif model_type == 'consistency_distillation':
        if not pretrain_model:
            raise ValueError("Consistency distillation requires pretrained flow model")
        z_batch = torch.randn_like(train_y, device=device)
        forward_step = 10
        N = math.ceil(1000 * (epoch / num_epochs) + 4) + forward_step
        boundaries = torch.linspace(0, 1 - 1e-3, N, device=device)
        t_idx = torch.randint(0, N - forward_step, (batch_dim, 1), device=device)
        return model.loss(train_x, train_y, z_batch, boundaries[t_idx], 1/N, forward_step, sys_data, pretrain_model), train_y
        
    else:
        raise NotImplementedError(f"Model type '{model_type}' not implemented")


# ==================== Main Function ====================

def main(debug=False):
    """Main function for standard supervised training."""
    config = get_standard_config()
    
    print("=" * 60)
    print("DeepOPF-V: Standard Supervised Training")
    print("=" * 60)
    config.print_config()
    
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    
    model_type = config.model_type
    print(f"\nModel type: {model_type}")
    print(f"Available: {get_available_model_types()}")
    
    # Load data
    sys_data, dataloaders, BRANFT = load_all_data(config)
    
    input_ch = sys_data.x_train.shape[1]
    output_vm = sys_data.yvm_train.shape[1]
    output_va = sys_data.yva_train.shape[1]
    
    print(f"\nDimensions: input={input_ch}, Vm={output_vm}, Va={output_va}")
    
    # Create models
    model_vm = create_model(model_type, input_ch, output_vm, config, is_vm=True)
    model_va = create_model(model_type, input_ch, output_va, config, is_vm=False)
    
    # Load VAE anchors if needed
    pretrain_vm, pretrain_va = None, None
    need_anchor = model_type == 'rectified' or (model_type == 'diffusion' and getattr(config, 'use_vae_anchor', False))
    
    if need_anchor:
        print(f"\n[Info] Loading VAE anchors...")
        for path, name, out_ch in [(config.pretrain_model_path_vm, 'Vm', output_vm),
                                    (config.pretrain_model_path_va, 'Va', output_va)]:
            if path and os.path.exists(path):
                m = create_model('vae', input_ch, out_ch, config, is_vm=(name=='Vm'))
                m.to(config.device)
                m.load_state_dict(torch.load(path, map_location=config.device, weights_only=True))
                m.eval()
                if name == 'Vm': pretrain_vm = m
                else: pretrain_va = m
                print(f"  Loaded {name} VAE: {path}")
        
        model_vm.pretrain_model = pretrain_vm
        model_va.pretrain_model = pretrain_va
    
    model_vm.to(config.device)
    model_va.to(config.device)
    print(f'\nDevice: {config.device}')
    
    # Setup optimizers & schedulers
    weight_decay = getattr(config, 'weight_decay', 0)
    opt_vm = torch.optim.Adam(model_vm.parameters(), lr=config.Lrm, weight_decay=weight_decay)
    opt_va = torch.optim.Adam(model_va.parameters(), lr=config.Lra, weight_decay=weight_decay)
    
    sched_vm, sched_va = None, None
    if hasattr(config, 'learning_rate_decay') and config.learning_rate_decay:
        step, gamma = config.learning_rate_decay
        sched_vm = torch.optim.lr_scheduler.StepLR(opt_vm, step_size=step, gamma=gamma)
        sched_va = torch.optim.lr_scheduler.StepLR(opt_va, step_size=step, gamma=gamma)
        print(f"LR scheduler: step={step}, gamma={gamma}")
    
    criterion = nn.MSELoss()
    
    # Train or load
    if not debug:
        print("\n" + "=" * 60)
        print("Training Mode")
        print("=" * 60)
        
        model_vm, _, _ = train_model(
            config, model_vm, opt_vm, dataloaders['train_vm'], criterion, config.device,
            model_name='Vm', model_type=model_type, pretrain_model=pretrain_vm, 
            scheduler=sched_vm, sys_data=sys_data
        )
        model_va, _, _ = train_model(
            config, model_va, opt_va, dataloaders['train_va'], criterion, config.device,
            model_name='Va', model_type=model_type, pretrain_model=pretrain_va,
            scheduler=sched_va, sys_data=sys_data
        )
    else:
        print("\n[Debug] Loading pretrained models...")
        vm_path = "main_part/saved_models/modelvm300r2N1Lm8642E1000_simple.pth"
        va_path = "main_part/saved_models/modelva300r2N1La8642E1000_simple.pth"
        model_vm.load_state_dict(torch.load(vm_path, map_location=config.device, weights_only=True))
        model_va.load_state_dict(torch.load(va_path, map_location=config.device, weights_only=True))
        print("  Models loaded.")
    
    # Evaluation
    print("\n" + "=" * 80)
    print("Evaluation")
    print("=" * 80)
    
    ctx = build_ctx_from_supervised(config, sys_data, dataloaders, BRANFT, config.device)
    predictor = SupervisedPredictor(
        model_vm, model_va, dataloaders,
        model_type=model_type,
        pretrain_model_vm=pretrain_vm,
        pretrain_model_va=pretrain_va,
    )
    return evaluate_unified(ctx, predictor, apply_post_processing=True, verbose=True)


if __name__ == "__main__":
    debug = bool(int(os.environ.get('DEBUG', '0')))
    main(debug=debug)
