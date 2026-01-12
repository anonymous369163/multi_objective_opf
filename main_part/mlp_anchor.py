#!/usr/bin/env python
# coding: utf-8
"""
Standard MLP Anchor Generator

This module provides:
1. StandardMLPAnchor: Wrapper for Standard MLP (NetVm + NetVa) as anchor generator
2. load_standard_mlp_anchor: Function to load Standard MLP anchor from checkpoint

The Standard MLP anchor is used by:
- test.py: For Flow and Refiner models evaluation
- train_multi_preference_tfm_refiner_v2.py: For Refiner training

Author: Peng Yue
Date: January 2026
"""

import torch
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BaseConfig


# ==================== Standard MLP Anchor ====================

class StandardMLPAnchor:
    """
    Wrapper for Standard MLP (NetVm + NetVa) to serve as anchor generator.
    
    Provides the same interface as VAE for use as pretrain_model in Flow training.
    Standard MLP was trained on lc=0 data, so it provides the cost-optimal solution
    as the starting point for preference trajectory.
    """
    
    def __init__(self, model_vm, model_va, config, sys_data, fmt_converter, device):
        """
        Args:
            model_vm: Trained NetVm model (outputs [B, Nbus])
            model_va: Trained NetVa model (outputs [B, Nbus-1])
            config: Configuration object
            sys_data: Power system data
            fmt_converter: FormatConverter for FULL-BUS to NGT conversion
            device: torch device
        """
        self.model_vm = model_vm
        self.model_va = model_va
        self.config = config
        self.sys_data = sys_data
        self.fmt_converter = fmt_converter
        self.device = device
        
        # Set pref_dim = 1 for compatibility with unified_eval.py interface
        self.pref_dim = 1
        
        # Pre-compute conversion parameters
        self._setup_conversion_params()
    
    def _setup_conversion_params(self):
        """Setup parameters for scaled output to physical units conversion."""
        # Voltage bounds
        VmLb = self.sys_data.VmLb
        VmUb = self.sys_data.VmUb
        
        if isinstance(VmLb, np.ndarray):
            self.VmLb = torch.from_numpy(VmLb).float().to(self.device)
            self.VmUb = torch.from_numpy(VmUb).float().to(self.device)
        else:
            self.VmLb = VmLb.to(self.device) if isinstance(VmLb, torch.Tensor) else torch.tensor(VmLb, device=self.device)
            self.VmUb = VmUb.to(self.device) if isinstance(VmUb, torch.Tensor) else torch.tensor(VmUb, device=self.device)
        
        # Scale factors
        self.scale_vm = self.config.scale_vm
        self.scale_va = self.config.scale_va
        if isinstance(self.scale_vm, torch.Tensor):
            self.scale_vm = self.scale_vm.to(self.device)
            self.scale_va = self.scale_va.to(self.device)
        
        # Historical Vm range for clamping
        hisVm_min = self.sys_data.hisVm_min
        hisVm_max = self.sys_data.hisVm_max
        if isinstance(hisVm_min, np.ndarray):
            self.hisVm_min = torch.from_numpy(hisVm_min).float().to(self.device)
            self.hisVm_max = torch.from_numpy(hisVm_max).float().to(self.device)
        else:
            self.hisVm_min = hisVm_min.to(self.device) if isinstance(hisVm_min, torch.Tensor) else torch.tensor(hisVm_min, device=self.device)
            self.hisVm_max = hisVm_max.to(self.device) if isinstance(hisVm_max, torch.Tensor) else torch.tensor(hisVm_max, device=self.device)
        
        # Slack bus index
        self.bus_slack = self.sys_data.bus_slack if hasattr(self.sys_data, 'bus_slack') else 0
        self.Nbus = self.config.Nbus
    
    def __call__(self, scene, use_mean=True, pref=None):
        """
        Predict anchor in NGT format.
        
        Args:
            scene: Input [B, input_dim] (load scenario in sparse format)
            use_mean: Ignored (for VAE compatibility)
            pref: Ignored (Standard MLP always predicts lc=0 solution)
        
        Returns:
            anchor: [B, output_dim] in NGT format (normalized)
        """
        return self.forward(scene)
    
    @torch.no_grad()
    def forward(self, scene):
        """Forward pass: predict Vm/Va and convert to NGT format."""
        from utils import get_clamp
        
        device = scene.device
        B = scene.shape[0]
        
        # Predict Vm and Va in scaled format
        yvm_hat = self.model_vm(scene)  # [B, Nbus]
        yva_hat = self.model_va(scene)  # [B, Nbus-1]
        
        # Convert to physical units
        yvm_physical = yvm_hat / self.scale_vm * (self.VmUb - self.VmLb) + self.VmLb
        yva_physical = yva_hat / self.scale_va  # Va in radians
        
        # Clamp Vm to historical range
        Vm_full = get_clamp(yvm_physical, self.hisVm_min, self.hisVm_max)
        
        # Insert slack bus Va (value = 0) to get full Va
        Va_full = torch.zeros(B, self.Nbus, device=device, dtype=scene.dtype)
        if self.bus_slack == 0:
            Va_full[:, 1:] = yva_physical
        elif self.bus_slack == self.Nbus - 1:
            Va_full[:, :-1] = yva_physical
        else:
            Va_full[:, :self.bus_slack] = yva_physical[:, :self.bus_slack]
            Va_full[:, self.bus_slack+1:] = yva_physical[:, self.bus_slack:]
        
        # Convert to FULL-BUS format: [Va_noslack, Vm_all]
        if self.bus_slack == 0:
            Va_noslack = Va_full[:, 1:]
        elif self.bus_slack == self.Nbus - 1:
            Va_noslack = Va_full[:, :-1]
        else:
            Va_noslack = torch.cat([Va_full[:, :self.bus_slack], Va_full[:, self.bus_slack+1:]], dim=1)
        
        y_fullbus = torch.cat([Va_noslack, Vm_full], dim=1)  # [B, 2*Nbus-1]
        
        # Convert to NGT format using FormatConverter
        y_ngt = self.fmt_converter.y_fullbus_to_ngt(y_fullbus)
        
        if isinstance(y_ngt, np.ndarray):
            y_ngt = torch.from_numpy(y_ngt).float().to(device)
        
        return y_ngt
    
    def eval(self):
        """Set models to eval mode."""
        self.model_vm.eval()
        self.model_va.eval()
        return self
    
    def train(self, mode=True):
        """Set models to train mode."""
        self.model_vm.train(mode)
        self.model_va.train(mode)
        return self
    
    def parameters(self):
        """Yield all parameters from both models."""
        yield from self.model_vm.parameters()
        yield from self.model_va.parameters()
    
    def to(self, device):
        """Move models to device."""
        self.model_vm.to(device)
        self.model_va.to(device)
        self.device = device
        self._setup_conversion_params()
        return self


def load_standard_mlp_anchor(config, sys_data, multi_pref_data, device):
    """
    Load Standard MLP as anchor generator.
    
    Returns:
        StandardMLPAnchor: Anchor model with VAE-compatible interface
    """
    from models import NetVm, NetVa
    from format_converter import FormatConverter
    
    input_dim = multi_pref_data['input_dim']
    
    # Standard model uses: output_vm = Nbus, output_va = Nbus-1
    output_vm = config.Nbus
    output_va = config.Nbus - 1
    
    # Get hidden layer config
    if config.Nbus == 118:
        khidden_Vm = np.array([8, 4, 2], dtype=int)
        khidden_Va = np.array([8, 4, 2], dtype=int)
    elif config.Nbus == 300:
        khidden_Vm = np.array([8, 6, 4, 2], dtype=int)
        khidden_Va = np.array([8, 6, 4, 2], dtype=int)
    else:
        khidden_Vm = np.array([8, 4, 2], dtype=int)
        khidden_Va = np.array([8, 4, 2], dtype=int)
    
    hidden_units = 128 if config.Nbus >= 100 else (64 if config.Nbus > 30 else 16)
    
    # Create models
    model_vm = NetVm(input_dim, output_vm, hidden_units, khidden_Vm)
    model_va = NetVa(input_dim, output_va, hidden_units, khidden_Va)
    
    # Model paths
    nmLm = 'Lm' + ''.join(str(k) for k in khidden_Vm)
    nmLa = 'La' + ''.join(str(k) for k in khidden_Va)
    
    vm_path = os.path.join(config.model_save_dir, f"modelvm{config.Nbus}r{config.sys_R}N{config.model_version}{nmLm}E1000_simple.pth")
    va_path = os.path.join(config.model_save_dir, f"modelva{config.Nbus}r{config.sys_R}N{config.model_version}{nmLa}E1000_simple.pth")
    
    # Check files exist
    missing = []
    if not os.path.exists(vm_path):
        missing.append(f"Vm: {vm_path}")
    if not os.path.exists(va_path):
        missing.append(f"Va: {va_path}")
    
    if missing:
        raise FileNotFoundError(f"Standard MLP model files not found:\n  " + "\n  ".join(missing) +
                               f"\n\nPlease train with: DEBUG=0 MODEL_TYPE=simple python main_part/train_standard.py")
    
    # Load weights
    model_vm.load_state_dict(torch.load(vm_path, map_location=device, weights_only=True))
    model_va.load_state_dict(torch.load(va_path, map_location=device, weights_only=True))
    
    model_vm.to(device).eval()
    model_va.to(device).eval()
    
    # Create FormatConverter
    case_file = config.case_file if hasattr(config, 'case_file') else f"main_part/data/case{config.Nbus}_ieee_modified.m"
    fmt_converter = FormatConverter(case_file)
    
    # Create anchor
    anchor = StandardMLPAnchor(model_vm, model_va, config, sys_data, fmt_converter, device)
    
    print(f"  Loaded Standard MLP Anchor:")
    print(f"    Vm: {vm_path}")
    print(f"    Va: {va_path}")
    print(f"    Parameters: Vm={sum(p.numel() for p in model_vm.parameters()):,}, Va={sum(p.numel() for p in model_va.parameters()):,}")
    
    return anchor
