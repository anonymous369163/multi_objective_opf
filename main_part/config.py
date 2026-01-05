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
        self.Nbus = 118  # Number of buses
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
        
        # ==================== Model Type Selection ====================
        # Available: 'simple', 'vae', 'rectified', 'diffusion', etc.
        self.model_type = os.environ.get('MODEL_TYPE', 'rectified')
        self.load_pretrained_model = bool(int(os.environ.get('LOAD_PRETRAINED_MODEL', '0')))
        
        # ==================== Dataset Parameters ====================
        if self.Nbus == 300:
            self.Neach = 12000
            self.case_m_path = os.path.join(_SCRIPT_DIR, "data/case300_ieee_modified.m")
        elif self.Nbus == 118:
            self.Neach = 2000
            self.case_m_path = os.path.join(_SCRIPT_DIR, "data/case118_ieee_modified.m")
        else:
            raise ValueError(f"Unsupported system size: {self.Nbus}")
            
        self.Ntrain = int(4 * self.Neach)
        self.Nsample = int(5 * self.Neach)
        self.Ntest = int(self.Neach)
        
        # ==================== Testing Parameters ==================== 
        self.model_version = 1 
        
        # ==================== File Paths ====================
        self.data_path = os.path.join(_SCRIPT_DIR, 'data') + os.sep 
        self.model_save_dir = os.path.join(_SCRIPT_DIR, 'saved_models')
        self.results_dir = os.path.join(_SCRIPT_DIR, 'results')
        
        # ==================== Device Configuration ====================
        gpu_id = int(os.environ.get('CUDA_DEVICE', '0'))
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        

        # ==================== NGT 相关参数，导入训练数据的时候需要 ====================
        # NOTE: These bounds should match the OPF solver's voltage constraints in the MATPOWER case file
        # For case118: Vmin=0.95, Vmax=1.05 (from case118_ieee_modified.m)
        # For case300: Vmin=0.94, Vmax=1.06 (from case300_ieee_modified.m)
        if self.Nbus == 300:
            self.ngt_VmLb, self.ngt_VmUb = 0.94, 1.06
            self.ngt_VaLb = -math.pi * 21 / 180
            self.ngt_VaUb = math.pi * 40 / 180
        elif self.Nbus == 118:
            # Match OPF voltage constraints: [0.95, 1.05] from case118_ieee_modified.m
            self.ngt_VmLb, self.ngt_VmUb = 0.95, 1.05
            self.ngt_VaLb = -math.pi * 20 / 180
            self.ngt_VaUb = math.pi * 16 / 180
        else:
            raise ValueError(f"Unsupported system size: {self.Nbus}")
        
        # ==================== NGT Dataset Parameters ====================
        self.ngt_Ntrain = 600
        self.ngt_Ntest = 2500
        self.ngt_Nhis = 3
        self.ngt_Nsample = 4000   # original: 50000
        self.ngt_random_seed = 12343
        self.training_data_file = "XY_case118real_from_npz_lc0.00.mat"  # "XY_case118real_from_npz_lc0.00.mat"  # "XY_case300real.mat"

    
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
