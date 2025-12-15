#!/usr/bin/env python
# coding: utf-8
# Configuration file for DeepOPF-V
# Author: Wanjun HUANG
# Date: July 4th, 2021

import torch
import numpy as np
import os
import math

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
        self.batch_size_training = 50  # Mini-batch size for training (reduced to prevent OOM)
        self.batch_size_test = 50  # Mini-batch size for test (same as training to prevent OOM)
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
        
        # ==================== VAE Anchor Configuration ====================
        # For diffusion/rectified flow: choose starting point for generation
        #   True  - Use pretrained VAE to generate anchor (better quality, requires VAE)
        #   False - Use Gaussian noise as starting point (standard approach)
        self.use_vae_anchor = True    # Whether to use VAE as anchor for diffusion/flow models
        
        # ==================== Unsupervised Training Parameters ====================
        # Training mode: 'supervised' uses labels (y), 'unsupervised' uses physics-based loss
        # Supports environment variable for bash configuration:
        #   export TRAINING_MODE=supervised  (or 'unsupervised')
        #   or: TRAINING_MODE=supervised python train.py
        self.training_mode = os.environ.get('TRAINING_MODE', 'unsupervised')  # 'supervised' or 'unsupervised'
        
        # Carbon scale: ensures cost and carbon are on same magnitude
        # Typical cost ~4000-5000 $/h, carbon ~100-200 tCO2/h
        # Scale = cost_typical / carbon_typical ≈ 4500 / 150 = 30
        self.carbon_scale = 30.0
        
        # ==================== DeepOPF-NGT Unsupervised Parameters ====================
        # Based on the DeepOPF-NGT paper implementation
        # Uses Kron Reduction to reduce prediction variables
        # Custom backward with analytical Jacobian for stable gradients
        
        # Cost coefficient (scales the generation cost objective)
        self.ngt_kcost = 0.0002
        
        # Adaptive weight flag: 1 = fixed weights, 2 = adaptive (recommended)
        self.ngt_flag_k = 2
        
        # Maximum penalty weights (used to cap adaptive weights)
        self.ngt_kpd_max = 100.0    # Load P deviation
        self.ngt_kqd_max = 100.0    # Load Q deviation
        self.ngt_kgenp_max = 2000.0 # Generator P constraint
        self.ngt_kgenq_max = 2000.0 # Generator Q constraint
        self.ngt_kv_max = 500.0     # Voltage magnitude (ZIB recovery)
        
        # Initial penalty weights (when flag_k = 1, fixed mode)
        self.ngt_kpd_init = 100.0
        self.ngt_kqd_init = 100.0
        self.ngt_kgenp_init = 2000.0
        self.ngt_kgenq_init = 2000.0
        self.ngt_kv_init = 100.0
        
        # Post-processing coefficient for voltage correction
        self.ngt_k_dV = 0.1
        
        # ==================== DeepOPF-NGT Training Configuration ====================
        # These parameters are ONLY used for unsupervised training mode
        # They match exactly the reference implementation (main_DeepOPFNGT_M3.ipynb)
        
        # Dataset sizes (exactly matching paper settings for 300-bus)
        self.ngt_Ntrain = 600       # Training samples (paper: 600)
        self.ngt_Ntest = 2500       # Test samples (paper: 2500)
        self.ngt_Nhis = 3           # Historical samples for post-processing
        self.ngt_Nsample = 50000    # Total samples pool to sample from
        
        # Training hyperparameters
        self.ngt_Epoch = 4500       # Training epochs (paper: 4500)
        self.ngt_batch_size = 50    # Batch size (paper: 50)
        self.ngt_Lr = 1e-4          # Learning rate (paper: 1e-4)
        self.ngt_s_epoch = 3000     # Start saving models after this epoch
        self.ngt_p_epoch = 10       # Print interval (reduced from 100 for faster feedback)
        
        # Network architecture (exactly matching paper)
        self.ngt_khidden = np.array([64, 224], dtype=int)  # Hidden layer sizes
        self.ngt_hidden_units = 1   # Hidden units multiplier
        
        # Voltage bounds for 300-bus system (matching paper exactly)
        # These are used to construct Vscale and Vbias for NetV model
        if self.Nbus == 300:
            self.ngt_VmLb = 0.94
            self.ngt_VmUb = 1.06
            self.ngt_VaLb = -math.pi * 21 / 180  # -21 degrees
            self.ngt_VaUb = math.pi * 40 / 180   # +40 degrees
        elif self.Nbus == 118:
            self.ngt_VmLb = 1.02
            self.ngt_VmUb = 1.06
            self.ngt_VaLb = -math.pi * 20 / 180  # -20 degrees
            self.ngt_VaUb = math.pi * 16 / 180   # +16 degrees
        else:  # 30-bus or other
            self.ngt_VmLb = 0.98
            self.ngt_VmUb = 1.06
            self.ngt_VaLb = -math.pi * 17 / 180  # -17 degrees
            self.ngt_VaUb = -math.pi * 4 / 180   # -4 degrees
        
        # Random seed for reproducibility
        self.ngt_random_seed = 12343
        
        # ==================== Multi-Objective Optimization (default: disabled) ====================
        # When enabled, the objective becomes: λ_cost * L_cost + λ_carbon * L_carbon
        # When disabled (default), only economic cost is optimized (backward compatible)
        # Supports environment variables for batch training with different preferences:
        #   NGT_MULTI_OBJ, NGT_LAMBDA_COST, NGT_LAMBDA_CARBON
        self.ngt_use_multi_objective = os.environ.get('NGT_MULTI_OBJ', 'True').lower() == 'true'
        self.ngt_lambda_cost = float(os.environ.get('NGT_LAMBDA_COST', '0.9'))
        self.ngt_lambda_carbon = float(os.environ.get('NGT_LAMBDA_CARBON', '0.1'))
        self.ngt_carbon_scale = 30.0          # Carbon emission scale factor (balance numerical range)
        
        # ==================== NGT Rectified Flow Model Parameters ====================
        # Enable Flow model for NGT unsupervised training (alternative to MLP)
        # The Flow model uses VAE predictions as anchors and integrates to get final predictions
        # Supports environment variables for flexible configuration:
        #   NGT_USE_FLOW, NGT_FLOW_STEPS, NGT_USE_PROJ, NGT_FLOW_HIDDEN_DIM, NGT_FLOW_NUM_LAYERS
        self.ngt_use_flow_model = os.environ.get('NGT_USE_FLOW', 'False').lower() == 'true'
        self.ngt_flow_inf_steps = int(os.environ.get('NGT_FLOW_STEPS', '10'))  # Number of Euler integration steps
        self.ngt_use_projection = os.environ.get('NGT_USE_PROJ', 'False').lower() == 'true'  # Use tangent-space projection
        # Flow model architecture (tuned to match NetV MLP parameter count ~360k for 300-bus)
        # hidden_dim=144, num_layers=2 gives 356,769 params vs NetV's 359,875 (ratio=0.99)
        self.ngt_flow_hidden_dim = int(os.environ.get('NGT_FLOW_HIDDEN_DIM', '144'))  # Hidden dimension for Flow model
        self.ngt_flow_num_layers = int(os.environ.get('NGT_FLOW_NUM_LAYERS', '2'))  # Number of hidden layers in Flow model
        
        # Progressive/curriculum training: use a previous Flow model to generate anchors
        # instead of using VAE anchors directly
        #   NGT_USE_FLOW_ANCHOR: Whether to use a Flow model as anchor source
        #   NGT_ANCHOR_MODEL_PATH: Path to the anchor Flow model (required if using Flow anchor)
        #   NGT_ANCHOR_LAMBDA_COST: lambda_cost of the anchor Flow model (for logging)
        self.ngt_use_flow_anchor = os.environ.get('NGT_USE_FLOW_ANCHOR', 'False').lower() == 'true'
        self.ngt_anchor_model_path = os.environ.get('NGT_ANCHOR_MODEL_PATH', '')
        self.ngt_anchor_lambda_cost = float(os.environ.get('NGT_ANCHOR_LAMBDA_COST', '1.0'))
        
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
            if self.model_type in ['rectified', 'diffusion']:
                print(f"  Use VAE anchor: {self.use_vae_anchor}")
        print(f"\nTraining Parameters:")
        print(f"  Epochs (Vm/Va): {self.EpochVm}/{self.EpochVa}")
        print(f"  Learning rate (Vm/Va): {self.Lrm}/{self.Lra}")
        print(f"  Batch size: {self.batch_size_training}")
        
        # Print unsupervised training parameters if in unsupervised mode
        if self.training_mode == 'unsupervised':
            print(f"\nUnsupervised Training Mode:")
            print(f"  Loss weights are controlled via command-line args (see train_pareto_flow.py)")
            print(f"  Based on DeepOPF-NGT: k_ti = min(k_obj * L_obj / L_i, k_i_max)")
            # Print args-overridden values if they exist
            if hasattr(self, 'k_obj'):
                print(f"  k_obj={self.k_obj}, k_g={getattr(self, 'k_g', 'N/A')}, "
                      f"k_Sl={getattr(self, 'k_Sl', 'N/A')}, k_theta={getattr(self, 'k_theta', 'N/A')}, "
                      f"k_d={getattr(self, 'k_d', 'N/A')}")
            if hasattr(self, 'use_adaptive_weights'):
                print(f"  Adaptive weights: {self.use_adaptive_weights}")
            
            # Print DeepOPF-NGT specific parameters
            print(f"\nDeepOPF-NGT Parameters:")
            print(f"  Training: Ntrain={self.ngt_Ntrain}, Ntest={self.ngt_Ntest}, Epochs={self.ngt_Epoch}")
            print(f"  Batch size: {self.ngt_batch_size}, Learning rate: {self.ngt_Lr}")
            print(f"  Network: hidden={list(self.ngt_khidden)}, hidden_units={self.ngt_hidden_units}")
            print(f"  Voltage bounds: Vm=[{self.ngt_VmLb}, {self.ngt_VmUb}], Va=[{self.ngt_VaLb:.4f}, {self.ngt_VaUb:.4f}] rad")
            print(f"  kcost (cost scale): {self.ngt_kcost}")
            print(f"  Adaptive mode: {'Adaptive' if self.ngt_flag_k == 2 else 'Fixed'}")
            print(f"  Max weights: kpd={self.ngt_kpd_max}, kgenp={self.ngt_kgenp_max}, kv={self.ngt_kv_max}")
            
            # Print multi-objective parameters
            if self.ngt_use_multi_objective:
                print(f"\nMulti-Objective Mode: ENABLED")
                print(f"  λ_cost={self.ngt_lambda_cost}, λ_carbon={self.ngt_lambda_carbon}")
                print(f"  Carbon scale: {self.ngt_carbon_scale}")
            else:
                print(f"\nMulti-Objective Mode: DISABLED (single-objective, cost only)")
            
            # Print Flow model parameters
            if self.ngt_use_flow_model:
                print(f"\nNGT Flow Model: ENABLED")
                print(f"  Hidden dim: {self.ngt_flow_hidden_dim}, Num layers: {self.ngt_flow_num_layers}")
                print(f"  Integration steps: {self.ngt_flow_inf_steps}")
                print(f"  Use projection: {self.ngt_use_projection}")
            else:
                print(f"\nNGT Flow Model: DISABLED (using MLP)")
        
        # Print carbon scale if available
        print(f"\nCarbon scale factor: {self.carbon_scale}")
        
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

