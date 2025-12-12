#!/usr/bin/env python
# coding: utf-8
# Data Loading and Preprocessing for DeepOPF-V
# Author: Wanjun HUANG
# Date: July 4th, 2021

import numpy as np
import scipy.io
import torch
import torch.utils.data as Data
from scipy import sparse
import math
import gc


class PowerSystemData:
    """
    Container for power system data and parameters
    """
    def __init__(self):
        # System parameters
        self.Ybus = None  # Node admittance matrix
        self.Yf = None    # Branch from admittance matrix
        self.Yt = None    # Branch to admittance matrix
        self.bus = None   # Bus data
        self.gen = None   # Generator data
        self.gencost = None  # Generation cost data
        self.branch = None   # Branch data
        self.baseMVA = None  # Base MVA
        self.bus_slack = None  # Slack bus index
        
        # Load and generation indices
        self.load_idx = None
        self.idxPg = None  # Active power generator indices
        self.idxQg = None  # Reactive power generator indices
        self.bus_Pg = None  # Buses with active power generation
        self.bus_Qg = None  # Buses with reactive power generation
        self.bus_PQg = None
        
        # Limits
        self.MAXMIN_Pg = None  # Active power limits
        self.MAXMIN_Qg = None  # Reactive power limits
        self.VmLb = None  # Voltage magnitude lower bound
        self.VmUb = None  # Voltage magnitude upper bound
        
        # Training data
        self.x_train = None
        self.yvm_train = None
        self.yva_train = None
        
        # Test data
        self.x_test = None
        self.yvm_test = None
        self.yva_test = None
        
        # Historical voltage data
        self.his_V = None
        self.his_Vm = None
        self.his_Va = None
        self.hisVm_max = None
        self.hisVm_min = None
        
        # Load and generation for test set
        self.Pdtest = None
        self.Qdtest = None
        self.Pgtest = None
        self.Qgtest = None
        
        # Full load data
        self.RPd = None
        self.RQd = None
        self.RPg = None
        self.RQg = None


def load_system_parameters(config):
    """
    Load power system parameters from .mat file
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary containing system parameters
    """
    # Load system parameter file
    matpara_path = config.data_path + config.system_param_file
    matpara = scipy.io.loadmat(matpara_path)
    
    sys_data = PowerSystemData()
    
    # Extract parameters
    sys_data.Ybus = sparse.csr_matrix(matpara['Ybus'])
    sys_data.Yf = sparse.csr_matrix(matpara['Yf'])
    sys_data.Yt = sparse.csr_matrix(matpara['Yt'])
    sys_data.bus = matpara['bus']
    sys_data.gen = matpara['gen']
    sys_data.gencost = matpara['gencost']
    sys_data.branch = matpara['branch']
    sys_data.baseMVA = matpara['baseMVA']
    
    # Find slack bus
    bus_slack = np.where(sys_data.bus[:, 1] == 3)
    sys_data.bus_slack = np.squeeze(bus_slack)
    
    print(f'Slack bus index: {sys_data.bus_slack}')
    
    return sys_data


def load_training_data(config, sys_data):
    """
    Load training samples from .mat file
    
    Args:
        config: Configuration object
        sys_data: PowerSystemData object
        
    Returns:
        Updated sys_data with training data loaded
    """
    # Load training data file
    data_path = config.data_path + config.training_data_file
    mat = scipy.io.loadmat(data_path)
    
    # Extract load indices
    sys_data.load_idx = np.squeeze(mat['load_idx']).astype(int) - 1
    
    # Load raw data
    RPd0 = mat['RPd']  # Active load
    RQd0 = mat['RQd']  # Reactive load
    sys_data.RPg = mat['RPg']  # Active generation
    sys_data.RQg = mat['RQg']  # Reactive generation
    
    # Voltage bounds
    VmLb = mat['VmLb']  # Lower bound
    VmUb = mat['VmUb']  # Upper bound
    
    # Voltage data (output labels)
    YVa = mat['RVa'] * math.pi / 180  # Convert to radians
    YVa = np.delete(YVa, sys_data.bus_slack, axis=1)  # Remove slack bus
    YVm = mat['RVm']
    
    # Scale voltage magnitude
    kvm = (YVm - VmLb) / (VmUb - VmLb)
    
    print(f'Output data shapes - Vm: {YVm.shape}, Va: {YVa.shape}')
    
    # Process input data (only non-zero loads)
    idx_Pd = np.squeeze(np.where(np.abs(RPd0[0, :]) > 0), axis=0)
    idx_Qd = np.squeeze(np.where(np.abs(RQd0[0, :]) > 0), axis=0)
    
    # Create input features
    x = np.concatenate((RPd0[:, idx_Pd], RQd0[:, idx_Qd]), axis=1) / sys_data.baseMVA
    
    print(f'Input data shape: {x.shape}')
    
    # Store full load data
    sys_data.RPd = np.zeros((config.Nsample, config.Nbus))
    sys_data.RQd = np.zeros((config.Nsample, config.Nbus))
    sys_data.RPd[:, sys_data.load_idx] = RPd0[0:config.Nsample]
    sys_data.RQd[:, sys_data.load_idx] = RQd0[0:config.Nsample]
    
    # Clean up
    del RPd0, RQd0
    gc.collect()
    
    # Process generator data (matching original notebook Cell 2-3)
    sys_data.idxPg = np.squeeze(np.where(sys_data.gen[:, 3] > 0), axis=0)  # Column 3 (Qmax) > 0
    sys_data.idxQg = np.squeeze(np.where(sys_data.gen[:, 1] > 0), axis=0)  # Column 1 (Pg) > 0
    
    sys_data.bus_Pg = sys_data.gen[sys_data.idxPg, 0].astype(int) - 1
    sys_data.bus_Qg = sys_data.gen[sys_data.idxQg, 0].astype(int) - 1
    sys_data.bus_PQg = np.concatenate((sys_data.bus_Pg, sys_data.bus_Qg + config.Nbus), axis=0)
    
    sys_data.MAXMIN_Pg = sys_data.gen[sys_data.idxPg, 3:5] / sys_data.baseMVA  # Columns 3:5
    sys_data.MAXMIN_Qg = sys_data.gen[sys_data.idxQg, 1:3] / sys_data.baseMVA  # Columns 1:3
    
    # Store bounds (keep as numpy - will be converted later if needed)
    sys_data.VmLb = VmLb
    sys_data.VmUb = VmUb
    
    # Note: MAXMIN_Pg and MAXMIN_Qg remain as numpy arrays (not tensors)
    # This matches the original notebook behavior
    
    # Store load indices count for env compatibility
    sys_data.num_pd = len(idx_Pd)
    sys_data.num_qd = len(idx_Qd)
    sys_data.idx_Pd = idx_Pd
    sys_data.idx_Qd = idx_Qd
    
    return x, kvm, YVa, sys_data


def prepare_datasets(config, x, yvm, yva, sys_data):
    """
    Prepare training and test datasets with scaling
    
    Args:
        config: Configuration object
        x: Input features
        yvm: Voltage magnitude outputs
        yva: Voltage angle outputs
        sys_data: PowerSystemData object
        
    Returns:
        sys_data with prepared datasets
    """
    # Convert to tensors
    yvm = torch.from_numpy(yvm).float()
    yva = torch.from_numpy(yva).float()
    
    # Scale outputs
    yvms = yvm * config.scale_vm
    yvas = yva * config.scale_va
    
    # Convert to tensors
    x_tensor = torch.from_numpy(x).float()
    yvm_tensor = yvms.float()
    yva_tensor = yvas.float()
    
    # Split into train and test
    sys_data.x_train = x_tensor[0:config.Ntrain]
    sys_data.yvm_train = yvm_tensor[0:config.Ntrain]
    sys_data.yva_train = yva_tensor[0:config.Ntrain]
    
    sys_data.x_test = x_tensor[config.Ntrain:config.Nsample]
    sys_data.yvm_test = yvm_tensor[config.Ntrain:config.Nsample]
    sys_data.yva_test = yva_tensor[config.Ntrain:config.Nsample]
    
    print(f'Training set - Vm range: [{torch.min(sys_data.yvm_train):.6f}, {torch.max(sys_data.yvm_train):.6f}]')
    print(f'Training set - Va range: [{torch.min(sys_data.yva_train):.6f}, {torch.max(sys_data.yva_train):.6f}]')
    
    # Prepare test load and generation data
    sys_data.Pdtest = sys_data.RPd[config.Ntrain:config.Nsample] / sys_data.baseMVA
    sys_data.Qdtest = sys_data.RQd[config.Ntrain:config.Nsample] / sys_data.baseMVA
    sys_data.Pgtest = sys_data.RPg[config.Ntrain:config.Nsample, sys_data.idxPg] / sys_data.baseMVA
    sys_data.Qgtest = sys_data.RQg[config.Ntrain:config.Nsample, sys_data.idxQg] / sys_data.baseMVA
    sys_data.Qgtest = sys_data.Qgtest.squeeze()
    
    # Calculate historical voltage statistics
    VmLb_tensor = torch.from_numpy(sys_data.VmLb).float()
    VmUb_tensor = torch.from_numpy(sys_data.VmUb).float()
    
    Real_Vmtrain = sys_data.yvm_train / config.scale_vm * (VmUb_tensor - VmLb_tensor) + VmLb_tensor
    Real_Vatrain = sys_data.yva_train / config.scale_va
    
    sys_data.hisVm_max, _ = torch.max(Real_Vmtrain, dim=0)
    sys_data.hisVm_min, _ = torch.min(Real_Vmtrain, dim=0)
    sys_data.his_Va = np.mean(np.insert(Real_Vatrain.numpy(), sys_data.bus_slack, values=0, axis=1), axis=0)
    sys_data.his_Vm = np.mean(Real_Vmtrain.numpy(), axis=0)
    sys_data.his_V = sys_data.his_Vm * np.exp(1j * sys_data.his_Va)
    
    # Convert system parameters to tensors (keep MAXMIN_Pg/Qg as numpy)
    # Note: MAXMIN_Pg and MAXMIN_Qg stay as numpy arrays to match original notebook
    sys_data.bus = torch.from_numpy(sys_data.bus).float()
    sys_data.bus_PQg = torch.from_numpy(sys_data.bus_PQg)
    sys_data.VmLb = torch.from_numpy(sys_data.VmLb).float()
    sys_data.VmUb = torch.from_numpy(sys_data.VmUb).float()
    
    # ============================================================
    # Add convenience attributes for env compatibility (post_processing.py, rfm_utils.py)
    # ============================================================
    
    # G and B tensors from Ybus (for power flow calculations)
    sys_data.G = torch.from_numpy(sys_data.Ybus.real.toarray()).float()
    sys_data.B = torch.from_numpy(sys_data.Ybus.imag.toarray()).float()
    
    # Gf, Bf, Gt, Bt from Yf and Yt (for branch power flow)
    sys_data.Gf = torch.from_numpy(sys_data.Yf.real.toarray()).float()
    sys_data.Bf = torch.from_numpy(sys_data.Yf.imag.toarray()).float()
    sys_data.Gt = torch.from_numpy(sys_data.Yt.real.toarray()).float()
    sys_data.Bt = torch.from_numpy(sys_data.Yt.imag.toarray()).float()
    
    # Generator power limits as torch tensors (p.u.)
    # MAXMIN_Pg[:, 0] is max, MAXMIN_Pg[:, 1] is min
    sys_data.Pg_max = torch.from_numpy(sys_data.MAXMIN_Pg[:, 0]).float()
    sys_data.Pg_min = torch.from_numpy(sys_data.MAXMIN_Pg[:, 1]).float()
    sys_data.Qg_max = torch.from_numpy(sys_data.MAXMIN_Qg[:, 0]).float()
    sys_data.Qg_min = torch.from_numpy(sys_data.MAXMIN_Qg[:, 1]).float()
    
    # Generator bus index (alias for env compatibility)
    sys_data.gen_bus_idx = sys_data.bus_Pg
    
    # Load bus indices (for env compatibility)
    sys_data.pd_bus_idx = sys_data.idx_Pd
    sys_data.qd_bus_idx = sys_data.idx_Qd
    
    # Branch limits (S_max) - if available in branch data (column 5 is RATE_A in MATPOWER format)
    # MATPOWER branch columns: [F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, ...]
    n_branch = sys_data.branch.shape[0]
    if sys_data.branch.shape[1] > 5:
        rate_a = sys_data.branch[:, 5]  # RATE_A column
        baseMVA = sys_data.baseMVA.item() if hasattr(sys_data.baseMVA, 'item') else float(sys_data.baseMVA)
        S_max = np.where(rate_a == 0, 1e10, rate_a / baseMVA)  # Convert to p.u.
    else:
        # No RATE_A data, set to no limit
        S_max = np.full(n_branch, 1e10)
    sys_data.S_max = torch.from_numpy(S_max).float()
    
    # Branch angle incidence matrix
    BRANFT = torch.from_numpy(sys_data.branch[:, 0:2] - 1).long()
    
    # Connection matrices Cf, Ct for branch calculations
    n_bus = config.Nbus
    n_branch = sys_data.branch.shape[0]
    f_bus = (sys_data.branch[:, 0] - 1).astype(int)  # From bus indices (0-indexed)
    t_bus = (sys_data.branch[:, 1] - 1).astype(int)  # To bus indices (0-indexed)
    
    Cf = np.zeros((n_branch, n_bus))
    Ct = np.zeros((n_branch, n_bus))
    for i in range(n_branch):
        Cf[i, f_bus[i]] = 1
        Ct[i, t_bus[i]] = 1
    sys_data.Cf = torch.from_numpy(Cf).float()
    sys_data.Ct = torch.from_numpy(Ct).float()
    
    return sys_data, BRANFT


def create_dataloaders(config, sys_data):
    """
    Create PyTorch DataLoader objects for training and testing
    
    Args:
        config: Configuration object
        sys_data: PowerSystemData object
        
    Returns:
        Dictionary containing dataloaders
    """
    # Training dataloaders
    training_dataset_vm = Data.TensorDataset(sys_data.x_train, sys_data.yvm_train)
    training_loader_vm = Data.DataLoader(
        dataset=training_dataset_vm,
        batch_size=config.batch_size_training,
        shuffle=False,
    )
    
    training_dataset_va = Data.TensorDataset(sys_data.x_train, sys_data.yva_train)
    training_loader_va = Data.DataLoader(
        dataset=training_dataset_va,
        batch_size=config.batch_size_training,
        shuffle=False,
    )
    
    # Test dataloaders
    test_dataset_vm = Data.TensorDataset(sys_data.x_test, sys_data.yvm_test)
    test_loader_vm = Data.DataLoader(
        dataset=test_dataset_vm,
        batch_size=config.batch_size_test,
        shuffle=False,
    )
    
    test_dataset_va = Data.TensorDataset(sys_data.x_test, sys_data.yva_test)
    test_loader_va = Data.DataLoader(
        dataset=test_dataset_va,
        batch_size=config.batch_size_test,
        shuffle=False,
    )
    
    dataloaders = {
        'train_vm': training_loader_vm,
        'train_va': training_loader_va,
        'test_vm': test_loader_vm,
        'test_va': test_loader_va,
    }
    
    return dataloaders


def load_all_data(config):
    """
    Main function to load all data and create dataloaders
    
    Args:
        config: Configuration object
        
    Returns:
        sys_data: PowerSystemData object with all data
        dataloaders: Dictionary of DataLoader objects
        BRANFT: Branch angle incidence matrix
    """
    print("=" * 60)
    print("Loading Power System Data")
    print("=" * 60)
    
    # Load system parameters
    sys_data = load_system_parameters(config)
    
    # Load training data
    x, yvm, yva, sys_data = load_training_data(config, sys_data)
    
    # Prepare datasets
    sys_data, BRANFT = prepare_datasets(config, x, yvm, yva, sys_data)
    
    # Create dataloaders
    dataloaders = create_dataloaders(config, sys_data)
    
    print("=" * 60)
    print("Data loading completed successfully")
    print("=" * 60)
    
    return sys_data, dataloaders, BRANFT


if __name__ == "__main__":
    # Test data loading
    from config import get_config
    
    config = get_config()
    sys_data, dataloaders, BRANFT = load_all_data(config)
    
    print(f"\nDataLoader info:")
    print(f"  Training batches (Vm): {len(dataloaders['train_vm'])}")
    print(f"  Training batches (Va): {len(dataloaders['train_va'])}")
    print(f"  Test batches (Vm): {len(dataloaders['test_vm'])}")
    print(f"  Test batches (Va): {len(dataloaders['test_va'])}")
    
    # Test one batch
    for batch_x, batch_y in dataloaders['train_vm']:
        print(f"\nSample batch shapes:")
        print(f"  Input (x): {batch_x.shape}")
        print(f"  Output (yvm): {batch_y.shape}")
        break

