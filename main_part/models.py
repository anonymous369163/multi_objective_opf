#!/usr/bin/env python
# coding: utf-8
# Neural Network Models for DeepOPF-V
# Author: Peng Yue
# Date: July 4th, 2021
# Extended to support multiple model types: VAE, Flow, Diffusion, GAN, etc.

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os 

# Add flow_model to path for importing generative models
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'flow_model'))
 
from net_utiles import ( 
                        VAE,         # Variational Autoencoder
                        GAN,         # Generative Adversarial Network
                        WGAN,        # Wasserstein GAN
                        DM,          # Diffusion Model
                        FM,          # Flow Matching (Rectified Flow)
                        CM,          # Consistency Model (training)
                        CD,          # Consistency Model (distillation) 
                        )  

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


class NetV(nn.Module):
    """
    DeepOPF-NGT Unified Model for Unsupervised Training
    
    This model is a direct port from the reference implementation (main_DeepOPFNGT_M3.ipynb).
    It jointly predicts Va and Vm for non-ZIB buses in a single network.
    
    Key features:
    - Single model predicts [Va_nonZIB_noslack, Vm_nonZIB] (e.g., 465 dims for 300-bus)
    - Uses sigmoid activation with scale and bias to constrain output to physical range
    - Network structure: [64, 224] hidden units (as per DeepOPF-NGT paper)
    
    Args:
        input_channels: Number of input features (non-zero Pd + non-zero Qd, e.g., 374)
        output_channels: Number of output features (NPred_Va + NPred_Vm, e.g., 465)
        hidden_units: Multiplier for hidden layer sizes (default: 1)
        khidden: Array defining hidden layer structure (e.g., [64, 224])
        Vscale: Scale tensor for sigmoid output [Va_scale..., Vm_scale...]
        Vbias: Bias tensor for sigmoid output [Va_bias..., Vm_bias...]
    
    Output range:
        Va: [VaLb, VaUb] (e.g., [-0.366, 0.698] rad for 300-bus)
        Vm: [VmLb, VmUb] (e.g., [0.94, 1.06] p.u. for 300-bus)
    """
    def __init__(self, input_channels, output_channels, hidden_units, khidden, Vscale, Vbias):
        super(NetV, self).__init__()
        
        self.num_layer = khidden.shape[0]
        
        # Register scale and bias as buffers (moved to device with model)
        self.register_buffer('scale', Vscale)
        self.register_buffer('bias', Vbias)
        
        # First hidden layer
        self.fc1 = nn.Linear(input_channels, khidden[0] * hidden_units)
        
        # Additional hidden layers
        if self.num_layer >= 2:
            self.fc2 = nn.Linear(khidden[0] * hidden_units, khidden[1] * hidden_units)
        
        if self.num_layer >= 3:
            self.fc3 = nn.Linear(khidden[1] * hidden_units, khidden[2] * hidden_units)
        
        if self.num_layer >= 4:
            self.fc4 = nn.Linear(khidden[2] * hidden_units, khidden[3] * hidden_units)
        
        if self.num_layer >= 5:
            self.fc5 = nn.Linear(khidden[3] * hidden_units, khidden[4] * hidden_units)
        
        if self.num_layer >= 6:
            self.fc6 = nn.Linear(khidden[4] * hidden_units, khidden[5] * hidden_units)
        
        # Final two layers (fixed structure as per reference)
        self.fcbfend = nn.Linear(khidden[self.num_layer - 1] * hidden_units, output_channels)
        self.fcend = nn.Linear(output_channels, output_channels)
    
    def forward(self, x):
        """
        Forward pass with sigmoid-scaled output.
        
        Args:
            x: Input tensor [batch_size, input_channels]
               Contains [Pd_nonzero, Qd_nonzero] / baseMVA
        
        Returns:
            x_PredV: Predicted voltages [batch_size, output_channels]
                     First NPred_Va elements are Va (non-ZIB, no slack)
                     Next NPred_Vm elements are Vm (non-ZIB)
                     Output is constrained to [Vbias, Vbias + Vscale]
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
        x = self.fcend(x)
        
        # Key: sigmoid with scale and bias to constrain output to physical range
        # output = sigmoid(x) * scale + bias
        # This ensures Va is in [VaLb, VaUb] and Vm is in [VmLb, VmUb]
        x_PredV = torch.sigmoid(x) * self.scale + self.bias
        
        return x_PredV


class FiLMGenerator(nn.Module):
    """
    Generate FiLM parameters (gamma, beta) for each hidden layer from preference.
    For stability, we use (1 + gamma) scaling.
    """
    def __init__(self, pref_dim: int, hidden_dims: list[int], film_hidden: int = 64):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.mlps = nn.ModuleList()
        for hdim in hidden_dims:
            self.mlps.append(nn.Sequential(
                nn.Linear(pref_dim, film_hidden),
                nn.SiLU(),
                nn.Linear(film_hidden, 2 * hdim)
            ))

        # small init makes FiLM start near identity (gamma,beta ~ 0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, pref: torch.Tensor):
        """
        Args:
            pref: [B, pref_dim]
        Returns:
            gammas: list of [B, hdim]
            betas : list of [B, hdim]
        """
        gammas, betas = [], []
        for mlp, hdim in zip(self.mlps, self.hidden_dims):
            gb = mlp(pref)                  # [B, 2*hdim]
            gamma, beta = gb[:, :hdim], gb[:, hdim:]
            gammas.append(gamma)
            betas.append(beta)
        return gammas, betas


def apply_film(h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
    # stable FiLM: (1 + gamma) * h + beta
    return h * (1.0 + gamma) + beta


class NetV_FiLM_Direct(nn.Module):
    """
    NetV backbone + FiLM(pref) in hidden layers, directly predicts V in physical range:
        V = sigmoid(raw) * scale + bias

    Args:
        input_channels: x dim (Pd/Qd nonzero)
        output_channels: V dim (Va_nonZIB_noslack + Vm_nonZIB)
        hidden_units: multiplier
        khidden: e.g. torch.tensor([64, 224])
        Vscale, Vbias: tensors [output_channels]
        pref_dim: preference dimension (e.g. 2 for [lambda_cost, lambda_carbon])
        act: activation function
    """
    def __init__(self, input_channels, output_channels, hidden_units, khidden,
                 Vscale, Vbias, pref_dim=2, film_hidden=64, act="relu"):
        super().__init__()
        self.num_layer = int(khidden.shape[0])

        self.register_buffer('scale', Vscale)
        self.register_buffer('bias', Vbias)

        act_fn = nn.ReLU() if act == "relu" else nn.SiLU()

        # build hidden dims
        hidden_dims = [int(khidden[i].item()) * hidden_units for i in range(self.num_layer)]

        # trunk layers
        self.fcs = nn.ModuleList()
        in_dim = input_channels
        for hdim in hidden_dims:
            self.fcs.append(nn.Linear(in_dim, hdim))
            in_dim = hdim

        # NetV fixed ending
        self.fcbfend = nn.Linear(hidden_dims[-1], output_channels)
        self.fcend = nn.Linear(output_channels, output_channels)

        self.act = act_fn

        # FiLM per hidden layer (only for trunk)
        self.film = FiLMGenerator(pref_dim=pref_dim, hidden_dims=hidden_dims, film_hidden=film_hidden)

    def forward(self, x: torch.Tensor, pref: torch.Tensor):
        """
        Args:
            x:    [B, input_channels]
            pref: [B, pref_dim]  (per-sample preference)
        Returns:
            V_pred: [B, output_channels] in physical range
        """
        gammas, betas = self.film(pref)

        h = x
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.act(h)
            h = apply_film(h, gammas[i], betas[i])

        h = self.act(self.fcbfend(h))
        h = self.fcend(h)

        V_pred = torch.sigmoid(h) * self.scale + self.bias
        return V_pred


def create_model(model_type, input_dim, output_dim, config, is_vm=True, 
                 Vscale=None, Vbias=None):
    """
    Factory function to create models based on model_type
    
    Args:
        model_type: Type of model ('simple', 'vae', 'rectified', 'diffusion', 'gan', 'wgan', 
                   'consistency_training', 'consistency_distillation', 'ngt')
        input_dim: Input dimension (number of features)
        output_dim: Output dimension (Nbus for Vm, Nbus-1 for Va, or combined for NGT)
        config: Configuration object with hyperparameters
        is_vm: Whether this is a Vm model (True) or Va model (False)
        Vscale: Scale tensor for NGT model (required for 'ngt' type)
        Vbias: Bias tensor for NGT model (required for 'ngt' type)
        
    Returns:
        model: The created model
    """
    model_name = "Vm" if is_vm else "Va"
    
    if model_type == 'ngt':
        # DeepOPF-NGT unified model for unsupervised training (MLP)
        if Vscale is None or Vbias is None:
            raise ValueError("Vscale and Vbias are required for 'ngt' model type")
        
        khidden = config.ngt_khidden  # [64, 224]
        hidden_units = config.ngt_hidden_units  # 1
        
        model = NetV(
            input_channels=input_dim,
            output_channels=output_dim,
            hidden_units=hidden_units,
            khidden=khidden,
            Vscale=Vscale,
            Vbias=Vbias
        )
        print(f"[NGT] Created DeepOPF-NGT unified model (MLP)")
        print(f"      Input dim: {input_dim}, Output dim: {output_dim}")
        print(f"      Hidden layers: {khidden}")
        return model
    
    elif model_type == 'simple':
        # Original MLP model (NetVm/NetVa)
        khidden = config.khidden_Vm if is_vm else config.khidden_Va
        model = NetVm(input_dim, output_dim, config.hidden_units, khidden) if is_vm else \
                NetVa(input_dim, output_dim, config.hidden_units, khidden)
        print(f"[{model_name}] Created Simple MLP model") 
        
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
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def get_available_model_types():
    """Return list of available model types"""
    base_types = ['simple', 'ngt']  # ngt always available for unsupervised training
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


