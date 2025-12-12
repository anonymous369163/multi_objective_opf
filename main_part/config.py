#!/usr/bin/env python
# coding: utf-8
# Configuration file for DeepOPF-V
# Author: Wanjun HUANG
# Date: July 4th, 2021

import torch
import numpy as np
import os

# Get the directory where this config file is located
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    """Configuration class for DeepOPF-V model training and testing"""
    
    def __init__(self):
        # ==================== System Parameters ====================
        self.Nbus = 300  # Number of buses
        self.sys_R = 2  # Test case name (IEEE R2)
        
        # ==================== Mode Selection ====================
        self.flag_test = 0  # 0: train model; 1: test well-trained model
        self.flag_hisv = 1  # 1: use historical V to calculate dV; 0: use predicted V
        self.flagVm = 1
        self.flagVa = 1
        
        # ==================== Training Parameters ====================
        self.EpochVm = 1000  # Maximum epoch for Vm (as per DeepOPF-V paper)
        self.EpochVa = 1000  # Maximum epoch for Va (as per DeepOPF-V paper)
        self.batch_size_training = 512  # Mini-batch size for training (reduced to prevent OOM)
        self.batch_size_test = 256  # Mini-batch size for test (same as training to prevent OOM)
        self.s_epoch = 800  # Minimum epoch for .pth model saving
        self.p_epoch = 10  # Print loss for each "p_epoch" epoch
        
        # ==================== Hyperparameters ====================
        self.Lrm = 1e-3  # Learning rate for Vm
        self.Lra = 1e-3  # Learning rate for Va
        self.k_dV = 1  # Coefficient for dVa & dVm in post-processing
        self.DELTA = 1e-4  # Threshold of violation
        self.scale_vm = torch.tensor([10]).float()  # Scaling of output Vm
        self.scale_va = torch.tensor([10]).float()  # Scaling of output Va
        
        # ==================== Model Type Selection ====================
        # Available model types:
        #   'simple'    - Original MLP supervised learning (NetVm/NetVa)
        #   'vae'       - Variational Autoencoder
        #   'rectified' - Rectified Flow (requires VAE pretrain)
        #   'diffusion' - Diffusion Model
        #   'gan'       - Generative Adversarial Network
        #   'wgan'      - Wasserstein GAN
        #   'consistency_training'     - Consistency Model (training mode)
        #   'consistency_distillation' - Consistency Model (distillation mode)
        self.model_type = 'simple'  # Default: original MLP
        
        # ==================== Generative Model Parameters ====================
        self.latent_dim = 32          # Latent dimension for VAE/GAN
        self.time_step = 1000         # Time steps for Flow/Diffusion
        self.hidden_dim = 512         # Hidden dimension for generative models
        self.num_layers = 5           # Number of layers for generative models
        self.vae_beta = 1.0           # KL divergence weight for VAE
        self.weight_decay = 1e-6      # Weight decay for optimizer
        self.learning_rate_decay = [1000, 0.9]  # [step_size, gamma] for scheduler
        self.inf_step = 20            # Inference steps for Flow/Diffusion (100=high quality, 20=fast)
        
        # ==================== VAE Anchor Configuration ====================
        # For diffusion/rectified flow: choose starting point for generation
        #   True  - Use pretrained VAE to generate anchor (better quality, requires VAE)
        #   False - Use Gaussian noise as starting point (standard approach)
        self.use_vae_anchor = True    # Whether to use VAE as anchor for diffusion/flow models
        
        # ==================== Unsupervised Training Parameters ====================
        # Training mode: 'supervised' uses labels (y), 'unsupervised' uses physics-based loss
        self.training_mode = 'unsupervised'  # 'supervised' or 'unsupervised'
        
        # Unsupervised loss weights (based on DeepOPF-NGT paper)
        # L = k_obj * L_obj + k_g * L_g + k_Sl * L_Sl + k_theta * L_theta + k_d * L_d
        # 注意：约束权重需要足够大，否则模型会忽略约束
        self.k_obj = 0.001        # Weight for objective (generation cost) - 降低，先满足约束
        self.k_g = 1000.0         # Weight for generator power constraint violation
        self.k_Sl = 1000.0        # Weight for branch power flow violation
        self.k_theta = 100.0      # Weight for branch angle difference violation
        self.k_d = 1000.0         # Weight for load deviation penalty - 增大
        
        # Adaptive weight scheduling (dynamic balancing of gradient contributions)
        # When enabled: k_i^t = min(k_obj * L_obj / L_i, k_i_max)
        # 建议：训练初期禁用自适应权重，等模型稳定后再启用
        self.use_adaptive_weights = False  # 暂时禁用，使用固定权重稳定训练
        
        # ==================== Pareto-Adaptive Training Parameters ====================
        # Target preference for Pareto adaptation: [cost_weight, carbon_weight]
        # Training maps from VAE [1,0] solutions to target preference solutions
        self.target_preference = [0.9, 0.1]  # [λ_cost, λ_carbon]
        self.pareto_epochs = 500             # Number of epochs for Pareto adaptation training
        self.pareto_lr = 1e-4                # Learning rate for Pareto training
        self.pareto_inf_steps = 10           # Inference steps during training (fewer for efficiency)
        # Carbon scale: ensures cost and carbon are on same magnitude
        # Typical cost ~4000-5000 $/h, carbon ~100-200 tCO2/h
        # Scale = cost_typical / carbon_typical ≈ 4500 / 150 = 30
        self.carbon_scale = 30.0
        
        # ==================== Pretrain Model Path ====================
        # For rectified flow, need a pretrained VAE model as anchor generator
        # Paths will be set after model_version is defined (see below)
        
        # ==================== Model Architecture ====================
        # Hidden layers for voltage magnitude (Vm) prediction
        if self.Nbus == 300:
            self.khidden_Vm = np.array([8, 6, 4, 2], dtype=int)  # [1024, 768, 512, 256] units when hidden_units=128
            self.khidden_Va = np.array([8, 6, 4, 2], dtype=int)  # [1024, 768, 512, 256] units when hidden_units=128
            self.Neach = 12000  # 12000 test samples (as per DeepOPF-V paper for 300-bus system)
        elif self.Nbus == 118:
            self.khidden_Vm = np.array([8, 4, 2], dtype=int)
            self.khidden_Va = np.array([8, 4, 2], dtype=int)
            self.Neach = 2000  # 118-bus dataset has 10000 samples: 8000 train + 2000 test
        else:
            self.khidden_Vm = np.array([8, 4, 2], dtype=int)
            self.khidden_Va = np.array([8, 4, 2], dtype=int)
            self.Neach = 8000
        
        # Determine size of hidden layers
        if self.Nbus >= 100:
            self.hidden_units = 128
        elif self.Nbus > 30:
            self.hidden_units = 64
        else:
            self.hidden_units = 16
        
        self.Lm = self.khidden_Vm.shape[0]  # Number of hidden layers for Vm
        self.La = self.khidden_Va.shape[0]  # Number of hidden layers for Va
        
        # ==================== Dataset Parameters ====================
        self.Ntrain = int(4 * self.Neach)  # 48000 training samples (80% of 60000)
        self.Nsample = int(5 * self.Neach)  # 60000 total samples (as per DeepOPF-V paper)
        self.Ntest = int(self.Neach)  # 12000 test samples (20% of 60000)
        
        # ==================== Testing Parameters ====================
        self.REPEAT = 1  # Number of repeated computation for speedup test
        self.model_version = 1  # Version of model
        
        # ==================== File Paths ====================
        # Data paths - use absolute path relative to script location
        self.data_path = os.path.join(_SCRIPT_DIR, 'data') + os.sep
        self.training_data_file = 'XY_case300real.mat'
        self.system_param_file = f'pglib_opf_case{self.Nbus}_ieeer{self.sys_R}_para.mat'
        
        # Generate naming strings for hidden layers
        self.nmLm = 'Lm' + ''.join(str(k) for k in self.khidden_Vm)
        self.nmLa = 'La' + ''.join(str(k) for k in self.khidden_Va)
        
        # Model save paths (saved to saved_models folder) - use absolute path
        self.model_save_dir = os.path.join(_SCRIPT_DIR, 'saved_models')
        self.PATHVm = f'{self.model_save_dir}/modelvm{self.Nbus}r{self.sys_R}N{self.model_version}{self.nmLm}E{self.EpochVm}.pth'
        self.PATHVa = f'{self.model_save_dir}/modelva{self.Nbus}r{self.sys_R}N{self.model_version}{self.nmLa}E{self.EpochVa}.pth'
        self.PATHVms = f'{self.model_save_dir}/modelvm{self.Nbus}r{self.sys_R}N{self.model_version}{self.nmLm}'
        self.PATHVas = f'{self.model_save_dir}/modelva{self.Nbus}r{self.sys_R}N{self.model_version}{self.nmLa}'
        
        # Pretrained VAE model paths for rectified flow
        # Use the pre-trained VAE models in saved_models directory
        self.pretrain_model_path_vm = os.path.join(self.model_save_dir, 
            f'modelvm{self.Nbus}r{self.sys_R}N{self.model_version}Lm8642_vae_E{self.EpochVm}F1.pth')
        self.pretrain_model_path_va = os.path.join(self.model_save_dir, 
            f'modelva{self.Nbus}r{self.sys_R}N{self.model_version}La8642_vae_E{self.EpochVa}F1.pth')
        
        # Results path (saved to results folder) - use absolute path
        self.results_dir = os.path.join(_SCRIPT_DIR, 'results')
        self.resultnm = (f'{self.results_dir}/res_{self.Nbus}r{self.sys_R}M{self.model_version}H{self.flag_hisv}'
                        f'NT{self.Ntrain}B{self.batch_size_training}'
                        f'Em{self.EpochVm}Ea{self.EpochVa}{self.nmLm}{self.nmLa}rp{self.REPEAT}.mat')
        
        # ==================== Device Configuration ====================
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def print_config(self):
        """Print configuration summary"""
        print("=" * 60)
        print("DeepOPF-V Configuration")
        print("=" * 60)
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"\nSystem: {self.Nbus}-bus, R{self.sys_R}")
        print(f"Dataset: {self.Nsample} total samples ({self.Ntrain} train, {self.Ntest} test)")
        print(f"\nModel Type: {self.model_type}")
        print(f"Training Mode: {self.training_mode}")
        print(f"\nModel Architecture:")
        if self.model_type == 'simple':
            print(f"  Vm hidden layers: {self.khidden_Vm * self.hidden_units}")
            print(f"  Va hidden layers: {self.khidden_Va * self.hidden_units}")
        else:
            print(f"  Hidden dim: {self.hidden_dim}")
            print(f"  Num layers: {self.num_layers}")
            if self.model_type in ['vae', 'gan', 'wgan']:
                print(f"  Latent dim: {self.latent_dim}")
            if self.model_type in ['rectified', 'diffusion', 'consistency_training', 'consistency_distillation']:
                print(f"  Time steps: {self.time_step}")
                print(f"  Inference steps: {self.inf_step}")
            if self.model_type in ['rectified', 'diffusion']:
                print(f"  Use VAE anchor: {self.use_vae_anchor}")
        print(f"\nTraining Parameters:")
        print(f"  Epochs (Vm/Va): {self.EpochVm}/{self.EpochVa}")
        print(f"  Learning rate (Vm/Va): {self.Lrm}/{self.Lra}")
        print(f"  Batch size: {self.batch_size_training}")
        
        # Print unsupervised training parameters if in unsupervised mode
        if self.training_mode == 'unsupervised':
            print(f"\nUnsupervised Loss Weights:")
            print(f"  k_obj (cost): {self.k_obj}")
            print(f"  k_g (generator): {self.k_g}")
            print(f"  k_Sl (branch power): {self.k_Sl}")
            print(f"  k_theta (angle): {self.k_theta}")
            print(f"  k_d (load deviation): {self.k_d}")
            print(f"  Adaptive weights: {self.use_adaptive_weights}")
        
        # Print Pareto training parameters
        print(f"\nPareto-Adaptive Training:")
        print(f"  Target preference: {self.target_preference}")
        print(f"  Pareto epochs: {self.pareto_epochs}")
        print(f"  Pareto learning rate: {self.pareto_lr}")
        print(f"  Carbon scale factor: {self.carbon_scale}")
        
        print(f"\nConstraint threshold (DELTA): {self.DELTA}")
        print("=" * 60)


# Create global config instance
def get_config():
    """Get configuration instance"""
    return Config()


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    config.print_config()

