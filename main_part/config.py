#!/usr/bin/env python
# coding: utf-8
"""
Base Configuration for DeepOPF-V

This file contains ONLY shared/common parameters used across all training modes.
Mode-specific configurations are defined in their respective training files:
  - train_standard.py: StandardConfig
  - train_multi_preference.py: MultiPreferenceConfig
  - train_unsupervised_*.py: UnsupervisedConfig

Author: Peng Yue
Date: December 2025
"""

import torch
import numpy as np
import os
import math

# Get the directory where this config file is located
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class BaseConfig:
    """Base configuration with shared parameters across all training modes."""
    
    def __init__(self):
        # ==================== System Parameters ====================
        self.Nbus = 300  # Number of buses
        self.sys_R = 2   # Test case name (IEEE R2)
        
        # ==================== Mode Selection ====================
        self.flag_hisv = 1  # 1: use historical V to calculate dV; 0: use predicted V
        self.flagVm = 1
        self.flagVa = 1
        
        # ==================== Common Hyperparameters ====================
        self.DELTA = 1e-4  # Threshold of violation
        self.k_dV = 1      # Coefficient for dVa & dVm in post-processing
        self.scale_vm = torch.tensor([10]).float()  # Scaling of output Vm
        self.scale_va = torch.tensor([10]).float()  # Scaling of output Va
        
        # ==================== Batch Sizes (used by data_loader) ====================
        self.batch_size_training = 50
        self.batch_size_test = 50
        
        # ==================== Model Type Selection ====================
        # Available: 'simple', 'vae', 'rectified', 'diffusion', 'flow', etc.
        self.model_type = os.environ.get('MODEL_TYPE', 'vae')
        self.load_pretrained_model = bool(int(os.environ.get('LOAD_PRETRAINED_MODEL', '1')))
        
        # ==================== Dataset Parameters ====================
        if self.Nbus == 300:
            self.Neach = 12000
        elif self.Nbus == 118:
            self.Neach = 2000
        else:
            self.Neach = 8000
            
        self.Ntrain = int(4 * self.Neach)
        self.Nsample = int(5 * self.Neach)
        self.Ntest = int(self.Neach)
        
        # ==================== Testing Parameters ====================
        self.REPEAT = 1  # Number of repeated computation for speedup test
        self.model_version = 1
        
        # ==================== File Paths ====================
        self.data_path = os.path.join(_SCRIPT_DIR, 'data') + os.sep
        self.training_data_file = 'XY_case300real.mat'
        self.system_param_file = f'pglib_opf_case{self.Nbus}_ieeer{self.sys_R}_para.mat'
        self.model_save_dir = os.path.join(_SCRIPT_DIR, 'saved_models')
        self.results_dir = os.path.join(_SCRIPT_DIR, 'results')
        
        # ==================== Device Configuration ====================
        gpu_id = int(os.environ.get('CUDA_DEVICE', '0'))
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def print_config(self):
        """Print configuration summary."""
        print("=" * 60)
        print("DeepOPF-V Configuration")
        print("=" * 60)
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"\nSystem: {self.Nbus}-bus, R{self.sys_R}")
        print(f"Dataset: {self.Nsample} total ({self.Ntrain} train, {self.Ntest} test)")
        print(f"Model Type: {self.model_type}")
        print(f"Load Pretrained: {self.load_pretrained_model}")
        print("=" * 60)


def get_config():
    """Get base configuration instance."""
    return BaseConfig()


if __name__ == "__main__":
    config = get_config()
    config.print_config()
