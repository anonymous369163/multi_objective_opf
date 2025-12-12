#!/usr/bin/env python
# coding: utf-8
# Neural Network Models for DeepOPF-V
# Author: Wanjun HUANG
# Date: July 4th, 2021
# Extended to support multiple model types: VAE, Flow, Diffusion, GAN, etc.

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add flow_model to path for importing generative models
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'flow_model'))

# Import generative models from net_utiles
try:
    from net_utiles import (
        Simple_NN,   # Simple MLP wrapper
        VAE,         # Variational Autoencoder
        GAN,         # Generative Adversarial Network
        WGAN,        # Wasserstein GAN
        DM,          # Diffusion Model
        FM,          # Flow Matching (Rectified Flow)
        CM,          # Consistency Model (training)
        CD,          # Consistency Model (distillation)
        MLP,         # Basic MLP network
    )
    GENERATIVE_MODELS_AVAILABLE = True
    print("[models.py] Successfully imported generative models from net_utiles")
except ImportError as e:
    GENERATIVE_MODELS_AVAILABLE = False
    print(f"[models.py] Warning: Could not import generative models: {e}")
    print("[models.py] Only 'simple' model type (NetVm/NetVa) will be available")


class NetVa(nn.Module):
    """
    Neural Network for Voltage Angle (Va) Prediction
    
    Fully-connected deep neural network with variable number of hidden layers (2-6).
    Uses ReLU activation function.
    
    Args:
        input_channels: Number of input features (load data dimension)
        output_channels: Number of output features (number of buses - 1, excluding slack bus)
        hidden_units: Base number of units for hidden layers
        khidden: Array defining hidden layer structure (e.g., [8,6,4,2] for [1024,768,512,256] units)
    """
    def __init__(self, input_channels, output_channels, hidden_units, khidden):
        super(NetVa, self).__init__()
        
        self.num_layer = khidden.shape[0]
        
        # First hidden layer
        self.fc1 = nn.Linear(input_channels, khidden[0]*hidden_units)  
        
        # Additional hidden layers (if specified)
        if self.num_layer >= 2: 
            self.fc2 = nn.Linear(khidden[0]*hidden_units, khidden[1]*hidden_units)
        
        if self.num_layer >= 3:
            self.fc3 = nn.Linear(khidden[1]*hidden_units, khidden[2]*hidden_units)
        
        if self.num_layer >= 4:
            self.fc4 = nn.Linear(khidden[2]*hidden_units, khidden[3]*hidden_units)
            
        if self.num_layer >= 5:
            self.fc5 = nn.Linear(khidden[3]*hidden_units, khidden[4]*hidden_units)
            
        if self.num_layer >= 6:
            self.fc6 = nn.Linear(khidden[4]*hidden_units, khidden[5]*hidden_units)
        
        # Final two layers (fixed structure)
        self.fcbfend = nn.Linear(khidden[khidden.shape[0]-1]*hidden_units, output_channels)   
        self.fcend = nn.Linear(output_channels, output_channels)  

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_channels)
            
        Returns:
            x_PredVa: Predicted voltage angles of shape (batch_size, output_channels)
        """
        x = F.relu(self.fc1(x))
        
        if self.num_layer >= 2:
            x = F.relu(self.fc2(x))
            
        if self.num_layer >= 3:
            x = F.relu(self.fc3(x))
            
        if self.num_layer >= 4:
            x = F.relu(self.fc4(x))
        
        if self.num_layer >= 5:
            x = F.relu(self.fc5(x))
            
        if self.num_layer >= 6:
            x = F.relu(self.fc6(x))
        
        # Fixed final two layers
        x = F.relu(self.fcbfend(x))
        x_PredVa = self.fcend(x)
                
        return x_PredVa


class NetVm(nn.Module):
    """
    Neural Network for Voltage Magnitude (Vm) Prediction
    
    Fully-connected deep neural network with variable number of hidden layers (2-6).
    Uses ReLU activation function.
    
    Args:
        input_channels: Number of input features (load data dimension)
        output_channels: Number of output features (number of buses)
        hidden_units: Base number of units for hidden layers
        khidden: Array defining hidden layer structure (e.g., [8,6,4,2] for [1024,768,512,256] units)
    """
    def __init__(self, input_channels, output_channels, hidden_units, khidden):
        super(NetVm, self).__init__()
        
        self.num_layer = khidden.shape[0]
        
        # First hidden layer
        self.fc1 = nn.Linear(input_channels, khidden[0]*hidden_units)
        
        # Additional hidden layers (if specified)
        if self.num_layer >= 2: 
            self.fc2 = nn.Linear(khidden[0]*hidden_units, khidden[1]*hidden_units)
        
        if self.num_layer >= 3:
            self.fc3 = nn.Linear(khidden[1]*hidden_units, khidden[2]*hidden_units)
        
        if self.num_layer >= 4:
            self.fc4 = nn.Linear(khidden[2]*hidden_units, khidden[3]*hidden_units)
            
        if self.num_layer >= 5:
            self.fc5 = nn.Linear(khidden[3]*hidden_units, khidden[4]*hidden_units)
            
        if self.num_layer >= 6:
            self.fc6 = nn.Linear(khidden[4]*hidden_units, khidden[5]*hidden_units)
        
        # Final two layers (fixed structure)
        self.fcbfend = nn.Linear(khidden[khidden.shape[0]-1]*hidden_units, output_channels)   
        self.fcend = nn.Linear(output_channels, output_channels) 
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_channels)
            
        Returns:
            x_PredVm: Predicted voltage magnitudes of shape (batch_size, output_channels)
        """
        x = F.relu(self.fc1(x))
        
        if self.num_layer >= 2:
            x = F.relu(self.fc2(x))
            
        if self.num_layer >= 3:
            x = F.relu(self.fc3(x))
            
        if self.num_layer >= 4:
            x = F.relu(self.fc4(x))
        
        if self.num_layer >= 5:
            x = F.relu(self.fc5(x))
            
        if self.num_layer >= 6:
            x = F.relu(self.fc6(x))
        
        # Fixed final two layers
        x = F.relu(self.fcbfend(x))
        x_PredVm = self.fcend(x)

        return x_PredVm


class PreferenceConditionedMLP(nn.Module):
    """
    Preference-Conditioned MLP for Pareto-Adaptive Flow Model.
    
    This network takes preference vector [λ_cost, λ_carbon] as additional condition,
    enabling a single model to generate solutions for any preference on the Pareto front.
    
    Architecture:
        - Input: (x, z, t, preference) where:
            - x: Load condition [batch, input_dim]
            - z: Current state (anchor or intermediate) [batch, output_dim]
            - t: Time step [batch, 1]
            - preference: [λ_cost, λ_carbon] normalized weights [batch, 2]
        - Output: Velocity vector [batch, output_dim]
    
    The preference is embedded and fused with other conditions via:
        1. Dedicated preference embedding layer
        2. FiLM-style modulation: scale and shift based on preference
    """
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, 
                 preference_dim=2, preference_hidden=64, act='relu'):
        super(PreferenceConditionedMLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.preference_dim = preference_dim
        
        act_list = {
            'relu': nn.ReLU(), 
            'silu': nn.SiLU(), 
            'softplus': nn.Softplus(),
            'gelu': nn.GELU()
        }
        act_fn = act_list.get(act, nn.ReLU())
        
        # Preference embedding network
        # Maps [λ_cost, λ_carbon] to a hidden representation
        self.pref_embed = nn.Sequential(
            nn.Linear(preference_dim, preference_hidden),
            nn.ReLU(),
            nn.Linear(preference_hidden, hidden_dim)
        )
        
        # FiLM modulation layers (preference -> scale, shift)
        self.pref_scale = nn.Linear(hidden_dim, hidden_dim)
        self.pref_shift = nn.Linear(hidden_dim, hidden_dim)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Condition embedding (for load x)
        self.cond_w = nn.Sequential(nn.Linear(input_dim, hidden_dim))
        self.cond_b = nn.Sequential(nn.Linear(input_dim, hidden_dim))
        
        # State embedding (for z)
        self.state_embed = nn.Sequential(nn.Linear(output_dim, hidden_dim), act_fn)
        
        # Main network
        net = []
        for _ in range(num_layers):
            net.extend([nn.Linear(hidden_dim, hidden_dim), act_fn])
        net.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*net)
        
    def forward(self, x, z, t, preference=None):
        """
        Forward pass with preference conditioning.
        
        Args:
            x: Load condition [batch, input_dim]
            z: Current state [batch, output_dim]  
            t: Time step [batch, 1]
            preference: Preference vector [batch, 2] where [λ_cost, λ_carbon]
                       If None, defaults to [1.0, 0.0] (pure cost minimization)
        
        Returns:
            v: Velocity vector [batch, output_dim]
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Default preference if not provided
        if preference is None:
            preference = torch.tensor([[1.0, 0.0]], device=device).expand(batch_size, -1)
        
        # Embed components
        state_emb = self.cond_w(x) * self.state_embed(z) + self.cond_b(x)
        time_emb = self.time_embed(t)
        pref_emb = self.pref_embed(preference)
        
        # Combine base features
        h = state_emb + time_emb
        
        # Apply FiLM modulation based on preference
        scale = self.pref_scale(pref_emb)
        shift = self.pref_shift(pref_emb)
        h = h * (1 + scale) + shift
        
        # Forward through main network
        v = self.net(h)
        
        return v


class PreferenceConditionedFM(nn.Module):
    """
    Preference-Conditioned Flow Matching Model.
    
    A unified model that can generate Pareto-optimal solutions for ANY preference
    by taking the preference vector as an additional condition input.
    
    Key Features:
        - Single model for entire Pareto front (no need for multiple stage-specific models)
        - Preference-aware velocity prediction
        - Compatible with curriculum learning (train progressively across preferences)
        - Memory efficient (one model instead of N stage models)
    
    Training Modes:
        1. Uniform sampling: Sample preferences uniformly from [0,1] during training
        2. Curriculum: Progressive training from [1,0] to target preferences
        3. Specific preference: Train for a single preference (degenerates to standard flow)
    """
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, 
                 preference_dim=2, time_step=1000, output_norm=False):
        super(PreferenceConditionedFM, self).__init__()
        
        self.model = PreferenceConditionedMLP(
            input_dim=input_dim,
            output_dim=output_dim, 
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            preference_dim=preference_dim
        )
        
        self.output_dim = output_dim
        self.con_dim = input_dim
        self.preference_dim = preference_dim
        self.time_step = time_step
        self.normalize = output_norm
        self.min_sd = 0.01
        
        # Will be set externally for anchor generation
        self.pretrain_model = None
        
    def flow_forward(self, y, t, z, preference=None, vec_type='rectified'):
        """
        Flow forward pass with preference conditioning.
        
        Args:
            y: Target solution [batch, output_dim]
            t: Time step [batch, 1]
            z: Starting point / anchor [batch, output_dim]
            preference: Preference vector [batch, 2]
            vec_type: Velocity type ('rectified' for linear interpolation)
            
        Returns:
            yt: Interpolated point
            vec: Target velocity
        """
        if self.normalize:
            y = 2 * y - 1
            
        if vec_type == 'rectified':
            yt = t * y + (1 - t) * z
            vec = y - z
        else:
            # Default to rectified flow
            yt = t * y + (1 - t) * z
            vec = y - z
            
        return yt, vec
    
    def predict_velocity(self, x, z, t, preference=None):
        """
        Predict velocity at given state with preference conditioning.
        
        Args:
            x: Condition [batch, input_dim]
            z: Current state [batch, output_dim]
            t: Time [batch, 1]
            preference: Preference vector [batch, 2]
            
        Returns:
            v: Predicted velocity [batch, output_dim]
        """
        return self.model(x, z, t, preference)
    
    def flow_backward(self, x, z, preference=None, num_steps=10, method='Euler'):
        """
        Flow backward/forward integration with preference conditioning.
        
        Integrates the ODE: dz/dt = v(x, z, t, preference)
        
        Args:
            x: Condition [batch, input_dim]
            z: Starting point (anchor) [batch, output_dim]
            preference: Preference vector [batch, 2]
            num_steps: Number of integration steps
            method: ODE solver ('Euler', 'RK4')
            
        Returns:
            z_final: Final state after integration [batch, output_dim]
        """
        batch_size = x.shape[0]
        device = x.device
        dt = 1.0 / num_steps
        
        with torch.no_grad():
            for step in range(num_steps):
                t = torch.full((batch_size, 1), step * dt, device=device)
                
                if method == 'Euler':
                    v = self.predict_velocity(x, z, t, preference)
                    z = z + v * dt
                elif method == 'RK4':
                    t_half = torch.full((batch_size, 1), (step + 0.5) * dt, device=device)
                    t_full = torch.full((batch_size, 1), (step + 1) * dt, device=device)
                    
                    k1 = self.predict_velocity(x, z, t, preference)
                    k2 = self.predict_velocity(x, z + 0.5 * dt * k1, t_half, preference)
                    k3 = self.predict_velocity(x, z + 0.5 * dt * k2, t_half, preference)
                    k4 = self.predict_velocity(x, z + dt * k3, t_full, preference)
                    z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                    
        return z
    
    def compute_loss(self, x, y, z, preference=None, vec_type='rectified'):
        """
        Compute velocity matching loss with preference conditioning.
        
        Args:
            x: Condition [batch, input_dim]
            y: Target solution [batch, output_dim]
            z: Anchor/starting point [batch, output_dim]
            preference: Preference vector [batch, 2]
            vec_type: Velocity type
            
        Returns:
            loss: MSE loss between predicted and target velocity
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Sample random time
        t = torch.rand(batch_size, 1, device=device)
        
        # Get interpolation point and target velocity
        yt, vec_target = self.flow_forward(y, t, z, preference, vec_type)
        
        # Predict velocity
        vec_pred = self.predict_velocity(x, yt, t, preference)
        
        # MSE loss
        loss = F.mse_loss(vec_pred, vec_target)
        
        return loss


def create_model(model_type, input_dim, output_dim, config, is_vm=True):
    """
    Factory function to create models based on model_type
    
    Args:
        model_type: Type of model ('simple', 'vae', 'rectified', 'diffusion', 'gan', 'wgan', 
                   'consistency_training', 'consistency_distillation')
        input_dim: Input dimension (number of features)
        output_dim: Output dimension (Nbus for Vm, Nbus-1 for Va)
        config: Configuration object with hyperparameters
        is_vm: Whether this is a Vm model (True) or Va model (False)
        
    Returns:
        model: The created model
    """
    model_name = "Vm" if is_vm else "Va"
    
    if model_type == 'simple':
        # Original MLP model (NetVm/NetVa)
        khidden = config.khidden_Vm if is_vm else config.khidden_Va
        model = NetVm(input_dim, output_dim, config.hidden_units, khidden) if is_vm else \
                NetVa(input_dim, output_dim, config.hidden_units, khidden)
        print(f"[{model_name}] Created Simple MLP model")
        
    elif not GENERATIVE_MODELS_AVAILABLE:
        raise ImportError(f"Generative models not available. Cannot create '{model_type}' model.")
        
    elif model_type == 'vae':
        # 使用 CVAE 模式（Encoder 同时看条件 x 和目标 y）
        use_cvae = getattr(config, 'use_cvae', True)
        model = VAE(
            network='mlp',
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            latent_dim=config.latent_dim,
            output_act=None,
            pred_type='node',
            use_cvae=use_cvae
        )
        print(f"[{model_name}] Created VAE model (latent_dim={config.latent_dim}, CVAE={use_cvae})")
        
    elif model_type in ['rectified', 'gaussian', 'conditional', 'interpolation']:
        model = FM(
            network='mlp',
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            time_step=config.time_step,
            output_norm=False,
            pred_type='node'
        )
        print(f"[{model_name}] Created Flow Matching model (type={model_type})")
        
    elif model_type == 'diffusion':
        model = DM(
            network='mlp',
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            time_step=config.time_step,
            output_norm=False,
            pred_type='node'
        )
        print(f"[{model_name}] Created Diffusion model")
        
    elif model_type == 'gan':
        model = GAN(
            network='mlp',
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            latent_dim=config.latent_dim,
            output_act=None,
            pred_type='node'
        )
        print(f"[{model_name}] Created GAN model")
        
    elif model_type == 'wgan':
        model = WGAN(
            network='mlp',
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            latent_dim=config.latent_dim,
            output_act=None,
            pred_type='node'
        )
        print(f"[{model_name}] Created WGAN model")
        
    elif model_type == 'consistency_training':
        model = CM(
            network='mlp',
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            time_step=config.time_step,
            output_norm=False,
            pred_type='node'
        )
        print(f"[{model_name}] Created Consistency Model (training)")
        
    elif model_type == 'consistency_distillation':
        model = CD(
            network='mlp',
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            time_step=config.time_step,
            output_norm=False,
            pred_type='node'
        )
        print(f"[{model_name}] Created Consistency Model (distillation)")
    
    elif model_type == 'preference_flow':
        # Preference-conditioned flow model for Pareto-adaptive generation
        model = PreferenceConditionedFM(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            preference_dim=2,  # [λ_cost, λ_carbon]
            time_step=config.time_step,
            output_norm=False
        )
        print(f"[{model_name}] Created Preference-Conditioned Flow Model (unified Pareto model)")
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def get_available_model_types():
    """Return list of available model types"""
    base_types = ['simple', 'preference_flow']  # preference_flow is always available
    if GENERATIVE_MODELS_AVAILABLE:
        base_types.extend([
            'vae', 'rectified', 'diffusion', 
            'gan', 'wgan',
            'consistency_training', 'consistency_distillation'
        ])
    return base_types


def infer_model_type_from_path(model_path):
    """
    从模型文件路径推断模型类型
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        model_type: 推断出的模型类型
    """
    path_lower = model_path.lower()
    
    if 'rectified' in path_lower:
        return 'rectified'
    elif 'vae' in path_lower:
        return 'vae'
    elif 'diffusion' in path_lower:
        return 'diffusion'
    elif 'wgan' in path_lower:
        return 'wgan'
    elif 'gan' in path_lower:
        return 'gan'
    elif 'consistency_distillation' in path_lower or 'cd_' in path_lower:
        return 'consistency_distillation'
    elif 'consistency_training' in path_lower or 'cm_' in path_lower:
        return 'consistency_training'
    else:
        return 'simple'


def load_model_checkpoint(
    model_path,
    config=None,
    input_dim=None,
    output_dim=None,
    is_vm=True,
    device=None,
    model_type=None,
    pretrain_model_path=None,
    eval_mode=True
):
    """
    统一的模型加载函数 - 一键加载训练好的模型
    
    Args:
        model_path: 模型文件路径 (.pth)
        config: 配置对象 (可选，如果不提供则使用默认配置)
        input_dim: 输入维度 (可选，如果不提供会尝试从checkpoint推断)
        output_dim: 输出维度 (可选，如果不提供会尝试从checkpoint推断)
        is_vm: 是否是Vm模型 (True) 或 Va模型 (False)
        device: 设备 (可选，默认使用config.device或cpu)
        model_type: 模型类型 (可选，如果不提供会从路径推断)
        pretrain_model_path: rectified flow需要的预训练VAE模型路径 (可选)
        eval_mode: 是否设置为评估模式 (默认True)
        
    Returns:
        model: 加载好的模型，已移至指定设备并设置好模式
        
    Example:
        >>> # 最简单的用法 - 只需提供模型路径
        >>> model_vm = load_model_checkpoint('saved_models/modelvm300r2N1Lm8642_vae_E1000F1.pth')
        
        >>> # 指定配置和维度
        >>> model_va = load_model_checkpoint(
        ...     'saved_models/modelva300r2N1La8642_rectified_E1000F1.pth',
        ...     config=config,
        ...     input_dim=374,
        ...     output_dim=299,
        ...     is_vm=False
        ... )
    """
    import os
    
    # 获取默认配置
    if config is None:
        from config import get_config
        config = get_config()
    
    # 设置设备
    if device is None:
        device = getattr(config, 'device', torch.device('cpu'))
    elif isinstance(device, str):
        device = torch.device(device)
    
    # 推断模型类型
    if model_type is None:
        model_type = infer_model_type_from_path(model_path)
    
    print(f"[load_model_checkpoint] Loading model from: {model_path}")
    print(f"[load_model_checkpoint] Inferred model type: {model_type}")
    
    # 首先尝试加载checkpoint以获取维度信息
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # 尝试从checkpoint推断维度
    if input_dim is None or output_dim is None:
        inferred_input_dim, inferred_output_dim = _infer_dims_from_state_dict(checkpoint, model_type, config)
        if input_dim is None:
            input_dim = inferred_input_dim
        if output_dim is None:
            output_dim = inferred_output_dim
    
    if input_dim is None or output_dim is None:
        raise ValueError(
            f"无法推断模型维度。请显式提供 input_dim 和 output_dim 参数。\n"
            f"当前值: input_dim={input_dim}, output_dim={output_dim}"
        )
    
    print(f"[load_model_checkpoint] Model dimensions: input={input_dim}, output={output_dim}")
    
    # 创建模型
    model = create_model(model_type, input_dim, output_dim, config, is_vm=is_vm)
    
    # 对于 rectified flow，需要加载预训练的 VAE 模型
    if model_type == 'rectified':
        if pretrain_model_path is None:
            # 尝试使用默认的预训练模型路径
            pretrain_model_path = config.pretrain_model_path_vm if is_vm else config.pretrain_model_path_va
        
        if os.path.exists(pretrain_model_path):
            print(f"[load_model_checkpoint] Loading pretrain VAE from: {pretrain_model_path}")
            pretrain_model = create_model('vae', input_dim, output_dim, config, is_vm=is_vm)
            pretrain_model.to(device)
            pretrain_state = torch.load(pretrain_model_path, map_location=device, weights_only=True)
            pretrain_model.load_state_dict(pretrain_state)
            pretrain_model.eval()
            model.pretrain_model = pretrain_model
        else:
            print(f"[load_model_checkpoint] Warning: Pretrain VAE not found at {pretrain_model_path}")
            print("[load_model_checkpoint] Rectified flow model may not work correctly without pretrain model")
    
    # 加载模型参数
    model.load_state_dict(checkpoint)
    
    # 移至设备
    model.to(device)
    
    # 设置模式
    if eval_mode:
        model.eval()
    
    print(f"[load_model_checkpoint] Model loaded successfully! (device={device}, eval_mode={eval_mode})")
    
    return model


def _infer_dims_from_state_dict(state_dict, model_type, config):
    """
    从 state_dict 推断模型的输入和输出维度
    
    Args:
        state_dict: 模型的参数字典
        model_type: 模型类型
        config: 配置对象
        
    Returns:
        input_dim, output_dim: 推断出的维度（如果无法推断则返回None）
    """
    input_dim = None
    output_dim = None
    
    try:
        if model_type == 'simple':
            # NetVm/NetVa: fc1.weight shape is [hidden_dim, input_dim]
            # fcend.weight shape is [output_dim, output_dim]
            if 'fc1.weight' in state_dict:
                input_dim = state_dict['fc1.weight'].shape[1]
            if 'fcend.weight' in state_dict:
                output_dim = state_dict['fcend.weight'].shape[0]
                
        elif model_type == 'vae':
            # VAE: encoder 第一层 weight shape depends on use_cvae
            # decoder 最后一层输出 output_dim
            for key in state_dict.keys():
                if 'encoder' in key and 'weight' in key:
                    # encoder.0.weight for CVAE has shape [hidden, input_dim + output_dim]
                    # For non-CVAE: [hidden, input_dim + latent_dim] or similar
                    break
                if 'decoder' in key and 'weight' in key and key.endswith('.weight'):
                    # 找到decoder最后一层
                    pass
            # 对于VAE，维度推断比较复杂，使用默认值
            if config is not None:
                if config.Nbus == 300:
                    input_dim = 374  # 300-bus系统的典型输入维度
                    # output_dim 需要根据 is_vm 判断，这里无法确定
                elif config.Nbus == 118:
                    input_dim = 158
                    
        elif model_type in ['rectified', 'diffusion', 'consistency_training', 'consistency_distillation']:
            # Flow/Diffusion models: network.0.weight shape is [hidden_dim, input_dim + output_dim + 1]
            # +1 是因为时间嵌入
            for key in state_dict.keys():
                if 'network.0.weight' in key or (key.startswith('network') and 'weight' in key):
                    first_layer_shape = state_dict[key].shape
                    # 无法直接区分 input_dim 和 output_dim，使用配置
                    break
            if config is not None:
                if config.Nbus == 300:
                    input_dim = 374
                elif config.Nbus == 118:
                    input_dim = 158
                    
        elif model_type in ['gan', 'wgan']:
            # GAN: generator.0.weight shape is [hidden_dim, input_dim + latent_dim]
            if config is not None:
                if config.Nbus == 300:
                    input_dim = 374
                elif config.Nbus == 118:
                    input_dim = 158
                    
    except Exception as e:
        print(f"[_infer_dims_from_state_dict] Warning: Could not infer dimensions: {e}")
    
    return input_dim, output_dim


def load_model_pair(
    vm_path,
    va_path,
    config=None,
    input_dim=None,
    output_dim_vm=None,
    output_dim_va=None,
    device=None,
    model_type=None,
    pretrain_vm_path=None,
    pretrain_va_path=None,
    eval_mode=True
):
    """
    同时加载 Vm 和 Va 模型对
    
    Args:
        vm_path: Vm模型文件路径
        va_path: Va模型文件路径
        config: 配置对象 (可选)
        input_dim: 输入维度 (可选)
        output_dim_vm: Vm输出维度 (可选，通常等于Nbus)
        output_dim_va: Va输出维度 (可选，通常等于Nbus-1)
        device: 设备 (可选)
        model_type: 模型类型 (可选，如果不提供会从路径推断)
        pretrain_vm_path: rectified flow的预训练Vm VAE路径 (可选)
        pretrain_va_path: rectified flow的预训练Va VAE路径 (可选)
        eval_mode: 是否设置为评估模式 (默认True)
        
    Returns:
        model_vm, model_va: 加载好的模型对
        
    Example:
        >>> model_vm, model_va = load_model_pair(
        ...     'saved_models/modelvm300r2N1Lm8642_vae_E1000F1.pth',
        ...     'saved_models/modelva300r2N1La8642_vae_E1000F1.pth'
        ... )
    """
    # 获取默认配置
    if config is None:
        from config import get_config
        config = get_config()
    
    # 设置默认输出维度
    if output_dim_vm is None:
        output_dim_vm = config.Nbus
    if output_dim_va is None:
        output_dim_va = config.Nbus - 1
    
    print("=" * 50)
    print("Loading Vm model...")
    model_vm = load_model_checkpoint(
        vm_path,
        config=config,
        input_dim=input_dim,
        output_dim=output_dim_vm,
        is_vm=True,
        device=device,
        model_type=model_type,
        pretrain_model_path=pretrain_vm_path,
        eval_mode=eval_mode
    )
    
    print("\n" + "=" * 50)
    print("Loading Va model...")
    model_va = load_model_checkpoint(
        va_path,
        config=config,
        input_dim=input_dim,
        output_dim=output_dim_va,
        is_vm=False,
        device=device,
        model_type=model_type,
        pretrain_model_path=pretrain_va_path,
        eval_mode=eval_mode
    )
    
    print("\n" + "=" * 50)
    print("Both models loaded successfully!")
    
    return model_vm, model_va


if __name__ == "__main__":
    # Test model creation
    import numpy as np
    
    input_channels = 374  # Example for 300-bus system
    output_channels_vm = 300
    output_channels_va = 299
    hidden_units = 128
    khidden = np.array([8, 6, 4, 2], dtype=int)
    
    model_vm = NetVm(input_channels, output_channels_vm, hidden_units, khidden)
    model_va = NetVa(input_channels, output_channels_va, hidden_units, khidden)
    
    print("NetVm Architecture:")
    print(model_vm)
    print(f"\nTotal parameters (Vm): {sum(p.numel() for p in model_vm.parameters()):,}")
    
    print("\nNetVa Architecture:")
    print(model_va)
    print(f"\nTotal parameters (Va): {sum(p.numel() for p in model_va.parameters()):,}")
    
    # Test forward pass
    batch_size = 10
    x_test = torch.randn(batch_size, input_channels)
    
    output_vm = model_vm(x_test)
    output_va = model_va(x_test)
    
    print(f"\nInput shape: {x_test.shape}")
    print(f"Output Vm shape: {output_vm.shape}")
    print(f"Output Va shape: {output_va.shape}")
    
    # Test available model types
    print(f"\nAvailable model types: {get_available_model_types()}")
    
    # ==================== 测试统一加载函数 ====================
    print("\n" + "=" * 60)
    print("Testing unified model loading functions")
    print("=" * 60)
    
    # 测试模型类型推断
    test_paths = [
        'modelvm300r2N1Lm8642E1000F1.pth',          # simple
        'modelvm300r2N1Lm8642_vae_E1000F1.pth',     # vae
        'modelvm300r2N1Lm8642_rectified_E1000F1.pth', # rectified
        'modelvm300r2_diffusion_E500.pth',          # diffusion
        'modelvm_wgan_E200.pth',                    # wgan
        'modelvm_gan_E200.pth',                     # gan
    ]
    
    print("\nModel type inference test:")
    for path in test_paths:
        inferred_type = infer_model_type_from_path(path)
        print(f"  {path} -> {inferred_type}")
    
    # 使用示例 (实际加载需要真实的模型文件)
    print("\n" + "-" * 60)
    print("Usage examples (requires actual model files):")
    print("-" * 60)
    print("""
# 方式1: 最简单 - 只需模型路径
model_vm = load_model_checkpoint('saved_models/modelvm300r2N1Lm8642_vae_E1000F1.pth')

# 方式2: 指定维度和设备
model_va = load_model_checkpoint(
    'saved_models/modelva300r2N1La8642_rectified_E1000F1.pth',
    input_dim=374,
    output_dim=299,
    is_vm=False,
    device='cuda:0'
)

# 方式3: 同时加载 Vm 和 Va 模型对
model_vm, model_va = load_model_pair(
    'saved_models/modelvm300r2N1Lm8642_vae_E1000F1.pth',
    'saved_models/modelva300r2N1La8642_vae_E1000F1.pth'
)

# 方式4: 加载 rectified flow (自动加载预训练VAE)
model_vm = load_model_checkpoint(
    'saved_models/modelvm300r2N1Lm8642_rectified_E1000F1.pth',
    pretrain_model_path='saved_models/modelvm300r2N1Lm8642_vae_E1000F1.pth'
)
""")

