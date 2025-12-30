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
        self.flag_hisv = 1  # 1: use historical V to calculate dV; 0: use predicted V
        self.flagVm = 1
        self.flagVa = 1
        
        # ==================== Model Loading ====================
        # If True, skip training and directly load pretrained model for evaluation/comparison
        self.load_pretrained_model = bool(int(os.environ.get('LOAD_PRETRAINED_MODEL', '0')))   # 如果等于true，则跳过训练，直接加载预训练模型
        
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
        #   'vae_flow' - VAE+Flow model (two-stage training)
        self.model_type = os.environ.get('MODEL_TYPE', 'vae_flow')  # Default: original MLP  vae_flow
        self.multi_pref_flow_type = self.model_type 
        # Multi-preference training mode selection for preference-conditioned models
        # Options:
        #   'standard' - Flow Matching from anchor (VAE/noise) to target solution
        #   'preference_trajectory' - Learn velocity field dx/dλ on preference trajectory
        self.multi_pref_training_mode = 'preference_trajectory'  # Default: standard flow matching; change to 'preference_trajectory' for improved training
        
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
        
        # ==================== DeepOPF-NGT Unsupervised Parameters ====================
        # Based on the DeepOPF-NGT paper implementation
        # Uses Kron Reduction to reduce prediction variables
        # Custom backward with analytical Jacobian for stable gradients
        
        # Cost coefficient (scales the generation cost objective)
        self.ngt_kcost = 0.0002
        # Objective weight multiplier (increases objective function weight relative to constraints)
        # Higher value = more focus on optimizing objective, less on constraint satisfaction
        self.ngt_obj_weight_multiplier = float(os.environ.get('NGT_OBJ_WEIGHT_MULT', '10.0'))  # Default: 10x
        
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
        
        # Training hyperparameters (supports environment variables for batch training)
        self.ngt_Epoch = int(os.environ.get('NGT_EPOCH', '4500'))  # Training epochs (paper: 4500)
        self.ngt_batch_size = int(os.environ.get('NGT_BATCH_SIZE', '50'))  # Batch size (paper: 50)
        self.ngt_Lr = float(os.environ.get('NGT_LR', '1e-4'))  # Learning rate (paper: 1e-4)
        self.ngt_s_epoch = int(os.environ.get('NGT_S_EPOCH', '3000'))  # Start saving models after this epoch
        self.ngt_p_epoch = int(os.environ.get('NGT_P_EPOCH', '10'))  # Print interval
        
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
        self.ngt_lambda_cost = float(os.environ.get('NGT_LAMBDA_COST', '0.1'))
        self.ngt_lambda_carbon = 1.0 - self.ngt_lambda_cost 
        self.ngt_carbon_scale = float(os.environ.get('NGT_CARBON_SCALE', '10.0'))        # Carbon emission scale factor (balance numerical range)


        # [新增] 选择聚合方式
        self.ngt_use_preference_conditioning = os.environ.get('NGT_PREF_CONDITIONING', 'False').lower() == 'true'   # 多目标条件学习（条件化学习） # False 时：训练/推理都不喂 preference
        self.ngt_mo_objective_mode = "soft_tchebycheff"   # "weighted_sum" | "normalized_sum" | "soft_tchebycheff"

        # [新增] 归一化用的 EMA（normalized_sum / soft_tchebycheff 都会用到）
        self.ngt_mo_use_running_scale = True
        self.ngt_mo_ema_beta = 0.99
        self.ngt_mo_eps = 1e-8

        # [新增] soft_tchebycheff 的温度（越小越接近 hard max）
        self.ngt_mo_tau = 0.1
        
        # ==================== NGT Rectified Flow Model Parameters ====================
        # Enable Flow model for NGT unsupervised training (alternative to MLP)
        # The Flow model uses VAE predictions as anchors and integrates to get final predictions
        # Supports environment variables for flexible configuration:
        #   NGT_USE_FLOW, NGT_FLOW_STEPS, NGT_USE_PROJ, NGT_FLOW_HIDDEN_DIM, NGT_FLOW_NUM_LAYERS
        self.ngt_use_flow_model = os.environ.get('NGT_USE_FLOW', 'True').lower() == 'true'
        self.ngt_flow_inf_steps = int(os.environ.get('NGT_FLOW_STEPS', '10'))  # Number of Euler integration steps
        self.ngt_use_projection = os.environ.get('NGT_USE_PROJ', 'False').lower() == 'true'  # Use tangent-space projection
        # Flow model architecture (tuned to match NetV MLP parameter count ~360k for 300-bus)
        # hidden_dim=144, num_layers=2 gives 356,769 params vs NetV's 359,875 (ratio=0.99)
        self.ngt_flow_hidden_dim = int(os.environ.get('NGT_FLOW_HIDDEN_DIM', '144'))  # Hidden dimension for Flow model
        self.ngt_flow_num_layers = int(os.environ.get('NGT_FLOW_NUM_LAYERS', '2'))  # Number of hidden layers in Flow model
        
        # ==================== Multi-Preference Supervised Training ====================
        # Enable supervised training with multi-preference dataset
        # Uses a preference-conditioned Flow model trained on solutions for all preferences
        # Supports environment variables for configuration:
        #   MULTI_PREF_SUPERVISED, MULTI_PREF_EPOCHS, MULTI_PREF_LR, MULTI_PREF_FLOW_TYPE
        self.use_multi_objective_supervised = os.environ.get('MULTI_PREF_SUPERVISED', 'True').lower() == 'true'
        
        # Dataset path for multi-preference solutions
        self.multi_pref_dataset_path = os.path.join(
            os.path.dirname(_SCRIPT_DIR), 'saved_data', 'multi_preference_solutions', 'fully_covered_dataset.pt'
        )
        
        # Multi-preference model parameters
        self.multi_pref_epochs = int(os.environ.get('MULTI_PREF_EPOCHS', '4500'))  # Training epochs
        self.multi_pref_lr = float(os.environ.get('MULTI_PREF_LR', '1e-4'))  # Learning rate
        self.multi_pref_flow_steps = int(os.environ.get('MULTI_PREF_FLOW_STEPS', '10'))  # Sampling steps
        
        # Validation split ratio (for train/val split, default 0.2 = 20% for validation)
        self.multi_pref_val_ratio = float(os.environ.get('MULTI_PREF_VAL_RATIO', '0.2'))
        self.multi_pref_random_seed = int(os.environ.get('MULTI_PREF_RANDOM_SEED', '42'))  # Random seed for train/val split
        
        # Preference conditioning dimension (1 for lambda_carbon only)
        self.pref_dim = 1
        
        # Use VAE anchor for multi-preference Flow (if available)
        self.multi_pref_use_vae_anchor = os.environ.get('MULTI_PREF_VAE_ANCHOR', 'True').lower() == 'true'
        
        # Loss weights for preference trajectory training
        # L = alpha * Lv (velocity loss) + beta * L1 (one-step state loss) + gamma * Lroll (multi-step unroll loss)
        # Note: If loss_v ~ 0.2 and loss_l1 ~ 8e-5, beta should be ~1000-2500 to balance the contributions
        self.multi_pref_loss_alpha = float(os.environ.get('MULTI_PREF_LOSS_ALPHA', '1.0'))  # Weight for velocity loss
        self.multi_pref_loss_beta = float(os.environ.get('MULTI_PREF_LOSS_BETA', '1000.0'))    # Weight for one-step loss (increased to balance with loss_v when loss_v >> loss_l1)
        self.multi_pref_loss_gamma = float(os.environ.get('MULTI_PREF_LOSS_GAMMA', '0.0'))  # Weight for multi-step loss (0 = disabled)
        
        # Scheduled Sampling: probability of using ground truth vs model prediction
        # p starts at 1.0 and linearly decays to p_min over training
        self.multi_pref_scheduled_sampling = os.environ.get('MULTI_PREF_SCHEDULED_SAMPLING', 'True').lower() == 'true'
        self.multi_pref_scheduled_sampling_p_min = float(os.environ.get('MULTI_PREF_SCHEDULED_SAMPLING_P_MIN', '0.2'))  # Minimum probability
        
        # Multi-step unroll loss configuration
        self.multi_pref_rollout_horizon = int(os.environ.get('MULTI_PREF_ROLLOUT_HORIZON', '4'))  # Number of steps to unroll (H)
        self.multi_pref_rollout_use_rk2 = os.environ.get('MULTI_PREF_ROLLOUT_USE_RK2', 'True').lower() == 'true'  # Use RK2 for rollout
        
        # Flow Matching training configuration
        # Enable Flow Matching instead of adjacent-point sampling for preference trajectory training
        self.multi_pref_flow_matching = os.environ.get('MULTI_PREF_FLOW_MATCHING', 'False').lower() == 'true'
        # Sampling strategy: 'adjacent' (only adjacent points), 'random' (any two points), 'mixed' (50% adjacent + 50% random)
        self.multi_pref_fm_strategy = os.environ.get('MULTI_PREF_FM_STRATEGY', 'mixed')
        # Minimum normalized Δλ to filter out (avoid noise from very small segments)
        self.multi_pref_fm_min_dlambda = float(os.environ.get('MULTI_PREF_FM_MIN_DLAMBDA', '0.0'))
        # Weight loss by Δλ (suppress noise from small segments)
        self.multi_pref_fm_weight_by_dlambda = os.environ.get('MULTI_PREF_FM_WEIGHT_BY_DLAMBDA', 'False').lower() == 'true'
        # Per-dimension normalization for velocity loss (avoid certain dimensions dominating)
        self.multi_pref_fm_per_dim_norm = os.environ.get('MULTI_PREF_FM_PER_DIM_NORM', 'False').lower() == 'true' 
        
        # ==================== Linearized VAE + Latent Flow Configuration ====================
        # Two-stage training approach:
        # Stage 1: Train Linearized VAE with linearization constraints (L_1D, L_order)
        # Stage 2: Train Latent Flow Model on top of frozen VAE
        
        # Skip VAE training and load pre-trained VAE (for Stage 2 only training)
        # Set SKIP_VAE_TRAINING=True and PRETRAINED_VAE_PATH=path/to/checkpoint.pth
        self.skip_vae_training = os.environ.get('SKIP_VAE_TRAINING', 'False').lower() == 'true'
        self.pretrained_vae_path = os.environ.get('PRETRAINED_VAE_PATH', "main_part/saved_models/linearized_vae_epoch2900.pth")
        
        # --- Linearized VAE (Stage 1) ---
        # [UPDATED] Improved configuration for better reconstruction quality
        # Key changes:
        #   - latent_dim: 32 -> 64 (more capacity for reconstruction info)
        #   - hidden_dim: 256 -> 512 (stronger encoder/decoder)
        #   - num_layers: 3 -> 4 (deeper network)
        #   - beta_1d: 0.1 -> 0.01 (weaker linearization constraint, prioritize reconstruction)
        #   - gamma_order: 0.01 -> 0.005 (weaker ordering constraint)
        #   - epochs: 1000 -> 5000 (longer training)
        self.linearized_vae_epochs = int(os.environ.get('LINEARIZED_VAE_EPOCHS', '5000'))
        self.linearized_vae_batch_size = int(os.environ.get('LINEARIZED_VAE_BATCH_SIZE', '32'))
        self.linearized_vae_lr = float(os.environ.get('LINEARIZED_VAE_LR', '5e-4'))  # Slightly lower LR for stability
        self.linearized_vae_latent_dim = int(os.environ.get('LINEARIZED_VAE_LATENT_DIM', '64'))
        self.linearized_vae_hidden_dim = int(os.environ.get('LINEARIZED_VAE_HIDDEN_DIM', '512'))
        self.linearized_vae_num_layers = int(os.environ.get('LINEARIZED_VAE_NUM_LAYERS', '4'))
        self.linearized_vae_weight_decay = float(os.environ.get('LINEARIZED_VAE_WEIGHT_DECAY', '1e-5'))
        
        # Loss weights for Linearized VAE
        # L_total = L_rec + α·L_KL + β·L_1D + γ·L_order + δ·L_NGT
        # [UPDATED] Reduced linearization weights to prioritize reconstruction
        self.linearized_vae_alpha_kl = float(os.environ.get('LINEARIZED_VAE_ALPHA_KL', '0.0001'))  # KL divergence weight (reduced)
        self.linearized_vae_beta_1d = float(os.environ.get('LINEARIZED_VAE_BETA_1D', '0.01'))     # Low-rank constraint weight (reduced 10x)
        self.linearized_vae_gamma_order = float(os.environ.get('LINEARIZED_VAE_GAMMA_ORDER', '0.005'))  # Monotonic ordering weight (reduced 2x)
        self.linearized_vae_delta_ngt = float(os.environ.get('LINEARIZED_VAE_DELTA_NGT', '0.0005'))  # Physics constraint weight (reduced)
        
        # Number of preference samples per scene for linearization loss
        self.linearized_vae_n_pref_samples = int(os.environ.get('LINEARIZED_VAE_N_PREF_SAMPLES', '12'))  # More samples for better linearization
        
        # Learning rate decay for Linearized VAE (step_size, gamma)
        self.linearized_vae_lr_decay = [500, 0.8]  # Decay by 0.8 every 500 epochs (slower decay for longer training)
        
        # Logging intervals
        self.linearized_vae_print_interval = int(os.environ.get('LINEARIZED_VAE_PRINT_INTERVAL', '100'))
        self.linearized_vae_save_interval = int(os.environ.get('LINEARIZED_VAE_SAVE_INTERVAL', '500'))
        
        # --- Latent Flow Model (Stage 2) ---
        # Skip Flow training and load pre-trained Flow model
        # Set SKIP_FLOW_TRAINING=True and PRETRAINED_FLOW_PATH=path/to/checkpoint.pth
        self.skip_flow_training = os.environ.get('SKIP_FLOW_TRAINING', 'False').lower() == 'true'
        self.pretrained_flow_path = os.environ.get('PRETRAINED_FLOW_PATH', "main_part/saved_models/latent_flow_epoch1600.pth")
        
        # [UPDATED] Flow model config to match new VAE latent_dim=64
        self.latent_flow_epochs = int(os.environ.get('LATENT_FLOW_EPOCHS', '2000'))  # More epochs
        self.latent_flow_batch_size = int(os.environ.get('LATENT_FLOW_BATCH_SIZE', '64'))
        self.latent_flow_lr = float(os.environ.get('LATENT_FLOW_LR', '5e-4'))  # Higher LR
        self.latent_flow_hidden_dim = int(os.environ.get('LATENT_FLOW_HIDDEN_DIM', '512'))  # Match VAE hidden_dim
        self.latent_flow_num_layers = int(os.environ.get('LATENT_FLOW_NUM_LAYERS', '4'))
        self.latent_flow_weight_decay = float(os.environ.get('LATENT_FLOW_WEIGHT_DECAY', '1e-5'))
        
        # Number of adjacent pair samples per batch for velocity loss
        self.latent_flow_n_pair_samples = int(os.environ.get('LATENT_FLOW_N_PAIR_SAMPLES', '8'))
        
        # Rollout regularization
        self.latent_flow_use_rollout = os.environ.get('LATENT_FLOW_USE_ROLLOUT', 'True').lower() == 'true'
        self.latent_flow_gamma_rollout = float(os.environ.get('LATENT_FLOW_GAMMA_ROLLOUT', '0.1'))  # Weight for rollout loss
        self.latent_flow_rollout_horizon = int(os.environ.get('LATENT_FLOW_ROLLOUT_HORIZON', '5'))  # Steps to unroll
        self.latent_flow_rollout_use_heun = os.environ.get('LATENT_FLOW_ROLLOUT_USE_HEUN', 'True').lower() == 'true'  # Use Heun method (RK2) for rollout
        
        # One-step state consistency loss weight: L_z1 = ||(z_k + dr * v_pred) - z_{k+1}||^2
        self.latent_flow_beta_z1 = float(os.environ.get('LATENT_FLOW_BETA_Z1', '1.0'))
        
        # Flow-Matching loss (L_fm): trains velocity at interpolated bridge points
        # Sample (k, k+m), t~U(0,1), z_t = (1-t)*z_k + t*z_{k+m}
        # L_fm = ||v_pred(z_t, r_t) - (z_{k+m} - z_k) / (r_{k+m} - r_k)||^2
        self.latent_flow_alpha_fm = float(os.environ.get('LATENT_FLOW_ALPHA_FM', '1.0'))  # Weight for flow-matching loss
        self.latent_flow_n_fm_samples = int(os.environ.get('LATENT_FLOW_N_FM_SAMPLES', '8'))  # Number of FM samples per batch
        self.latent_flow_fm_min_gap = int(os.environ.get('LATENT_FLOW_FM_MIN_GAP', '1'))  # Min gap between (k, k+m)
        self.latent_flow_fm_max_gap = int(os.environ.get('LATENT_FLOW_FM_MAX_GAP', '20'))  # Max gap between (k, k+m)
        
        # Learning rate decay for Latent Flow (step_size, gamma)
        self.latent_flow_lr_decay = [100, 0.9]  # Decay by 0.9 every 100 epochs
        
        # Logging intervals
        self.latent_flow_print_interval = int(os.environ.get('LATENT_FLOW_PRINT_INTERVAL', '50'))
        self.latent_flow_save_interval = int(os.environ.get('LATENT_FLOW_SAVE_INTERVAL', '100'))
        
        # Inference settings
        self.latent_flow_inf_steps = int(os.environ.get('LATENT_FLOW_INF_STEPS', '20'))  # ODE integration steps
        self.latent_flow_inf_method = os.environ.get('LATENT_FLOW_INF_METHOD', 'heun')  # ODE solver: euler, heun, rk4
        
        # ==================== Generative VAE Configuration ====================
        # Generative VAE enables Best-of-K sampling for improved feasibility
        # Key idea: train VAE to produce a distribution where multiple samples are feasible
        
        # Sampling configuration
        self.generative_vae_n_samples = int(os.environ.get('GENERATIVE_VAE_N_SAMPLES', '5'))  # K for Best-of-K
        self.generative_vae_n_prefs = int(os.environ.get('GENERATIVE_VAE_N_PREFS', '3'))  # Preferences per batch
        
        # Loss weights (调整后：让监督损失 L_rec 主导训练，其他为辅助)
        # 原值: alpha_kl=0.01, delta_feas=0.1, eta_obj=0.01
        # 诊断发现 L_obj 原始值远大于 L_rec (~330倍)，导致 L_obj 主导训练
        # 修改: 降低权重约10倍，使 L_rec 占总损失的 60% 左右
        self.generative_vae_alpha_kl = float(os.environ.get('GENERATIVE_VAE_ALPHA_KL', '0.001'))  # KL divergence weight (原0.01)
        self.generative_vae_delta_feas = float(os.environ.get('GENERATIVE_VAE_DELTA_FEAS', '0.01'))  # Feasibility loss weight (原0.1)
        self.generative_vae_eta_obj = float(os.environ.get('GENERATIVE_VAE_ETA_OBJ', '0.001'))  # Objective loss weight (原0.01)
        
        # KL annealing configuration (延长 warmup 阶段，让模型先学好监督信号)
        self.generative_vae_warmup_epochs = int(os.environ.get('GENERATIVE_VAE_WARMUP_EPOCHS', '100'))  # Warmup: L_rec + L_kl only (原50)
        self.generative_vae_ramp_epochs = int(os.environ.get('GENERATIVE_VAE_RAMP_EPOCHS', '100'))  # Ramp: gradually add L_feas, L_obj (原50)
        self.generative_vae_free_bits = float(os.environ.get('GENERATIVE_VAE_FREE_BITS', '0.1'))  # Minimum KL per dimension
        
        # Feasibility loss aggregation
        self.generative_vae_feas_mode = os.environ.get('GENERATIVE_VAE_FEAS_MODE', 'softmin')  # softmin/softmean/cvar
        self.generative_vae_tau_start = float(os.environ.get('GENERATIVE_VAE_TAU_START', '0.5'))  # Initial temperature
        self.generative_vae_tau_end = float(os.environ.get('GENERATIVE_VAE_TAU_END', '0.1'))  # Final temperature
        
        # Dual reconstruction
        self.generative_vae_lambda_sample = float(os.environ.get('GENERATIVE_VAE_LAMBDA_SAMPLE', '0.3'))  # Weight for sample reconstruction
        
        # Memory management
        self.generative_vae_ngt_chunk_size = int(os.environ.get('GENERATIVE_VAE_NGT_CHUNK_SIZE', '1024'))  # Chunk size for NGT loss
        
        # Training stability
        self.generative_vae_max_grad_norm = float(os.environ.get('GENERATIVE_VAE_MAX_GRAD_NORM', '1.0'))  # Gradient clipping
        
        # Feasibility threshold (for Best-of-K selection)
        self.generative_vae_feas_threshold = float(os.environ.get('GENERATIVE_VAE_FEAS_THRESHOLD', '0.01'))  # Based on constraint_scaled
        
        # Training configuration
        self.generative_vae_epochs = int(os.environ.get('GENERATIVE_VAE_EPOCHS', '4000'))
        self.generative_vae_lr = float(os.environ.get('GENERATIVE_VAE_LR', '1e-4'))
        self.generative_vae_batch_size = int(os.environ.get('GENERATIVE_VAE_BATCH_SIZE', '32'))
        
        # Logging
        self.generative_vae_print_interval = int(os.environ.get('GENERATIVE_VAE_PRINT_INTERVAL', '10'))
        self.generative_vae_save_interval = int(os.environ.get('GENERATIVE_VAE_SAVE_INTERVAL', '200'))
        
        # ==================== Pretrain Model Path ====================
        # For rectified flow, need a pretrained VAE model as anchor generator
        # Paths will be set after model_version is defined (see below)
         
        # ==================== Model Architecture only for supervised learning====================
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
        # Support CUDA_DEVICE environment variable for multi-GPU training
        gpu_id = int(os.environ.get('CUDA_DEVICE', '0'))
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
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

