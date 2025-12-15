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
import random


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
        
        # ============================================================
        # DeepOPF-NGT: ZIB (Zero Injection Bus) identification
        # ============================================================
        self.bus_Pnet_all = None         # All non-ZIB buses (have load or generation)
        self.bus_ZIB_all = None          # Zero injection buses (no load, no generation)
        self.bus_Pnet_nonPg = None       # Pure load buses (no P generator)
        self.bus_Pnet_nonQg = None       # Pure load buses (no Q generator)
        self.bus_Pnet_noslack_all = None # Non-ZIB buses excluding slack
        self.NZIB = None                 # Number of ZIB buses
        self.NPred_Vm = None             # Number of Vm to predict (non-ZIB)
        self.NPred_Va = None             # Number of Va to predict (non-ZIB - 1)


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
    
    # ============================================================
    # DeepOPF-NGT: ZIB (Zero Injection Bus) identification
    # ============================================================
    # Get a sample of load data for bus identification
    RPd_sample = sys_data.RPd[0, :]
    RQd_sample = sys_data.RQd[0, :]
    
    # Find load buses that are not generator buses (for load deviation constraint)
    Pnet_nonPg = RPd_sample.copy()
    Pnet_nonQg = RQd_sample.copy()
    Pnet_nonPg[sys_data.bus_Pg] = 0
    Pnet_nonQg[sys_data.bus_Qg] = 0
    
    sys_data.bus_Pnet_nonPg = np.squeeze(np.where(np.abs(Pnet_nonPg) > 0), axis=0)
    sys_data.bus_Pnet_nonQg = np.squeeze(np.where(np.abs(Pnet_nonQg) > 0), axis=0)
    
    # Ensure 1D arrays
    if sys_data.bus_Pnet_nonPg.ndim == 0:
        sys_data.bus_Pnet_nonPg = np.array([sys_data.bus_Pnet_nonPg.item()])
    if sys_data.bus_Pnet_nonQg.ndim == 0:
        sys_data.bus_Pnet_nonQg = np.array([sys_data.bus_Pnet_nonQg.item()])
    
    # All generator buses (from gen data)
    bus_gen = sys_data.gen[:, 0].astype(int) - 1
    
    # Find non-ZIB buses (have either load or generation)
    Pnet = RPd_sample.copy()
    Pnet[bus_gen] = Pnet[bus_gen] + 10  # Mark generator buses
    
    sys_data.bus_Pnet_all = np.squeeze(np.where(np.abs(Pnet) > 0), axis=0)
    sys_data.bus_ZIB_all = np.squeeze(np.where(np.abs(Pnet) == 0), axis=0)
    
    # Ensure 1D arrays
    if sys_data.bus_Pnet_all.ndim == 0:
        sys_data.bus_Pnet_all = np.array([sys_data.bus_Pnet_all.item()])
    if sys_data.bus_ZIB_all.ndim == 0:
        sys_data.bus_ZIB_all = np.array([sys_data.bus_ZIB_all.item()])
    
    sys_data.NZIB = len(sys_data.bus_ZIB_all)
    
    # Find slack bus position in non-ZIB array
    idx_bus_Pnet_slack = np.where(sys_data.bus_Pnet_all == sys_data.bus_slack)[0]
    
    # Non-ZIB buses excluding slack (for Va prediction)
    sys_data.bus_Pnet_noslack_all = np.delete(sys_data.bus_Pnet_all, idx_bus_Pnet_slack, axis=0)
    
    # Prediction dimensions
    sys_data.NPred_Vm = len(sys_data.bus_Pnet_all)
    sys_data.NPred_Va = len(sys_data.bus_Pnet_noslack_all)
    
    print(f'[DeepOPF-NGT] ZIB identification:')
    print(f'  Non-ZIB buses: {sys_data.NPred_Vm}')
    print(f'  ZIB buses: {sys_data.NZIB}')
    print(f'  Pure load P buses: {len(sys_data.bus_Pnet_nonPg)}')
    print(f'  Pure load Q buses: {len(sys_data.bus_Pnet_nonQg)}')
    
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


def prepare_ngt_data(sys_data, config):
    """
    Prepare data for DeepOPF-NGT unsupervised training.
    
    This creates training labels that only include non-ZIB nodes,
    matching the original notebook's yvtrain_Pnet format.
    
    Args:
        sys_data: PowerSystemData object with ZIB indices computed
        config: Configuration object
        
    Returns:
        yvm_train_ngt: Vm training data for non-ZIB nodes [Ntrain, NPred_Vm]
        yva_train_ngt: Va training data for non-ZIB nodes (no slack) [Ntrain, NPred_Va]
        yvm_test_ngt: Vm test data for non-ZIB nodes
        yva_test_ngt: Va test data for non-ZIB nodes (no slack)
    """
    if sys_data.bus_Pnet_all is None:
        raise ValueError("ZIB indices not computed. Run load_training_data first.")
    
    # Get unscaled Vm and Va
    VmLb = sys_data.VmLb
    VmUb = sys_data.VmUb
    
    # Unscale Vm back to [VmLb, VmUb]
    yvm_train_full = sys_data.yvm_train / config.scale_vm * (VmUb - VmLb) + VmLb
    yvm_test_full = sys_data.yvm_test / config.scale_vm * (VmUb - VmLb) + VmLb
    
    # Unscale Va back to radians
    yva_train_full = sys_data.yva_train / config.scale_va
    yva_test_full = sys_data.yva_test / config.scale_va
    
    # Insert slack bus Va (=0) to get full Va
    # yva_train/test has slack bus removed, so we need to add it back
    bus_slack = sys_data.bus_slack
    Nbus_full = config.Nbus
    
    yva_train_with_slack = torch.zeros((yva_train_full.shape[0], Nbus_full))
    yva_train_with_slack[:, :bus_slack] = yva_train_full[:, :bus_slack]
    yva_train_with_slack[:, bus_slack+1:] = yva_train_full[:, bus_slack:]
    
    yva_test_with_slack = torch.zeros((yva_test_full.shape[0], Nbus_full))
    yva_test_with_slack[:, :bus_slack] = yva_test_full[:, :bus_slack]
    yva_test_with_slack[:, bus_slack+1:] = yva_test_full[:, bus_slack:]
    
    # Extract non-ZIB nodes
    bus_Pnet_all = sys_data.bus_Pnet_all.tolist()
    bus_Pnet_noslack_all = sys_data.bus_Pnet_noslack_all.tolist()
    
    # Vm for non-ZIB buses
    yvm_train_ngt = yvm_train_full[:, bus_Pnet_all]
    yvm_test_ngt = yvm_test_full[:, bus_Pnet_all]
    
    # Va for non-ZIB buses (excluding slack)
    yva_train_ngt = yva_train_with_slack[:, bus_Pnet_noslack_all]
    yva_test_ngt = yva_test_with_slack[:, bus_Pnet_noslack_all]
    
    print(f'[DeepOPF-NGT] Prepared training data:')
    print(f'  Vm shape: {yvm_train_ngt.shape} (non-ZIB buses)')
    print(f'  Va shape: {yva_train_ngt.shape} (non-ZIB, no slack)')
    
    return yvm_train_ngt, yva_train_ngt, yvm_test_ngt, yva_test_ngt


def create_ngt_dataloaders(sys_data, config, yvm_train_ngt, yva_train_ngt, yvm_test_ngt, yva_test_ngt):
    """
    Create DataLoaders for DeepOPF-NGT unsupervised training.
    
    The output label is a combined [Va, Vm] vector for non-ZIB nodes.
    
    Args:
        sys_data: PowerSystemData object
        config: Configuration object
        yvm_train_ngt, yva_train_ngt: Training data for non-ZIB nodes
        yvm_test_ngt, yva_test_ngt: Test data for non-ZIB nodes
        
    Returns:
        Dictionary with 'train_ngt' and 'test_ngt' DataLoaders
    """
    # Combined output: [Va (non-ZIB, no slack), Vm (non-ZIB)]
    y_train_ngt = torch.cat([yva_train_ngt, yvm_train_ngt], dim=1)
    y_test_ngt = torch.cat([yva_test_ngt, yvm_test_ngt], dim=1)
    
    # Create datasets
    train_dataset_ngt = Data.TensorDataset(sys_data.x_train, y_train_ngt)
    test_dataset_ngt = Data.TensorDataset(sys_data.x_test, y_test_ngt)
    
    # Create dataloaders
    train_loader_ngt = Data.DataLoader(
        dataset=train_dataset_ngt,
        batch_size=config.batch_size_training,
        shuffle=True,  # Shuffle for training
    )
    
    test_loader_ngt = Data.DataLoader(
        dataset=test_dataset_ngt,
        batch_size=config.batch_size_test,
        shuffle=False,
    )
    
    print(f'[DeepOPF-NGT] Created DataLoaders:')
    print(f'  Train batches: {len(train_loader_ngt)}')
    print(f'  Test batches: {len(test_loader_ngt)}')
    print(f'  Output dim: {y_train_ngt.shape[1]} (Va: {yva_train_ngt.shape[1]}, Vm: {yvm_train_ngt.shape[1]})')
    
    return {
        'train_ngt': train_loader_ngt,
        'test_ngt': test_loader_ngt,
        'output_dim': y_train_ngt.shape[1],
        'va_dim': yva_train_ngt.shape[1],
        'vm_dim': yvm_train_ngt.shape[1],
    }


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


def load_ngt_training_data(config, sys_data=None):
    """
    Load training data specifically for DeepOPF-NGT unsupervised training.
    
    This function replicates EXACTLY the data preparation from main_DeepOPFNGT_M3.ipynb:
    - Random sampling of Ntrain (600) + Nhis (3) + Ntest (2500) samples from total pool
    - Input: [Pd_nonzero, Qd_nonzero] / baseMVA (374 dims for 300-bus)
    - Output: [Va_nonZIB_noslack, Vm_nonZIB] (465 dims for 300-bus)
    - Vscale and Vbias for sigmoid scaling in NetV model
    
    Args:
        config: Configuration object with ngt_* parameters
        sys_data: Optional PowerSystemData object. If None, loads system parameters.
        
    Returns:
        ngt_data: Dictionary containing:
            - 'x_train': Training input [Ntrain, input_dim]
            - 'x_test': Test input [Ntest, input_dim]
            - 'y_train': Training output [Ntrain, output_dim] (optional, for supervised ref)
            - 'y_test': Test output [Ntest, output_dim]
            - 'PQd_train': Load data for training [Ntrain, num_Pd + num_Qd]
            - 'PQd_test': Load data for test [Ntest, num_Pd + num_Qd]
            - 'Vscale': Scale tensor for NetV sigmoid [output_dim]
            - 'Vbias': Bias tensor for NetV sigmoid [output_dim]
            - 'his_V': Historical voltage for post-processing [Nbus] complex
            - 'idx_train': Training sample indices
            - 'idx_test': Test sample indices
            - 'idx_his': Historical sample indices
            - 'input_dim': Input dimension
            - 'output_dim': Output dimension (NPred_Va + NPred_Vm)
            - 'NPred_Va': Number of Va outputs
            - 'NPred_Vm': Number of Vm outputs
        sys_data: Updated PowerSystemData object
    """
    print("=" * 60)
    print("Loading Data for DeepOPF-NGT Unsupervised Training")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(config.ngt_random_seed)
    np.random.seed(config.ngt_random_seed)
    
    # Load system parameters if not provided
    if sys_data is None:
        sys_data = load_system_parameters(config)
    
    # Load training data file
    data_path = config.data_path + config.training_data_file
    mat = scipy.io.loadmat(data_path)
    
    # Extract raw data
    baseMVA = sys_data.baseMVA.item() if hasattr(sys_data.baseMVA, 'item') else float(sys_data.baseMVA)
    Nbus = config.Nbus
    
    # Load indices and raw data
    load_idx = np.squeeze(mat['load_idx']).astype(int) - 1
    RPd0 = mat['RPd']  # [Nsample, num_loads]
    RQd0 = mat['RQd']
    RPg = mat['RPg']
    RQg = mat['RQg']
    RVm = mat['RVm']
    RVa = mat['RVa'] * math.pi / 180  # Convert to radians
    
    print(f"Raw data shapes: RPd={RPd0.shape}, RQd={RQd0.shape}, RVm={RVm.shape}")
    
    # Expand load data to full Nbus (matching reference code)
    RPd = np.zeros((RPd0.shape[0], Nbus))
    RQd = np.zeros((RQd0.shape[0], Nbus))
    RPd[:, load_idx] = RPd0
    RQd[:, load_idx] = RQd0
    
    # Store for power flow calculations
    sys_data.RPd = RPd
    sys_data.RQd = RQd
    sys_data.RPg = RPg
    sys_data.RQg = RQg
    sys_data.load_idx = load_idx
    
    # Find slack bus
    bus_slack = int(sys_data.bus_slack)
    
    # ============================================================
    # Node type identification (exactly matching reference code)
    # ============================================================
    
    # Identify load buses
    bus_Pd = np.squeeze(np.where(np.abs(RPd[0, :]) > 0), axis=0)
    bus_Qd = np.squeeze(np.where(np.abs(RQd[0, :]) > 0), axis=0)
    
    print(f"Load buses: Pd={bus_Pd.shape}, Qd={bus_Qd.shape}")
    
    # Generator identification
    gen = sys_data.gen
    idxPg = np.squeeze(np.where(gen[:, 3] > 0), axis=0)  # Pmax > 0
    idxQg = np.squeeze(np.where(gen[:, 1] > 0), axis=0)  # Qmax > 0
    bus_Pg = gen[idxPg, 0].astype(int) - 1
    bus_Qg = gen[idxQg, 0].astype(int) - 1
    
    sys_data.idxPg = idxPg
    sys_data.idxQg = idxQg
    sys_data.bus_Pg = bus_Pg
    sys_data.bus_Qg = bus_Qg
    
    print(f"Generator buses: Pg={bus_Pg.shape}, Qg={bus_Qg.shape}")
    
    # Find pure load buses (no generator)
    Pnet_nonPg = RPd[0, :].copy()
    Pnet_nonQg = RQd[0, :].copy()
    Pnet_nonPg[bus_Pg] = 0
    Pnet_nonQg[bus_Qg] = 0
    bus_Pnet_nonPg = np.squeeze(np.where(np.abs(Pnet_nonPg) > 0), axis=0)
    bus_Pnet_nonQg = np.squeeze(np.where(np.abs(Pnet_nonQg) > 0), axis=0)
    
    sys_data.bus_Pnet_nonPg = bus_Pnet_nonPg
    sys_data.bus_Pnet_nonQg = bus_Pnet_nonQg
    
    # All generator buses
    bus_gen = gen[:, 0].astype(int) - 1
    
    # Find non-ZIB buses (have either load or generation)
    Pnet = RPd[0, :].copy()
    Pnet[bus_gen] = Pnet[bus_gen] + 10  # Mark generator buses
    
    bus_Pnet_all = np.squeeze(np.where(np.abs(Pnet) > 0), axis=0)
    bus_ZIB_all = np.squeeze(np.where(np.abs(Pnet) == 0), axis=0)
    
    NZIB = len(bus_ZIB_all) if bus_ZIB_all.ndim > 0 else (1 if bus_ZIB_all.size > 0 else 0)
    
    sys_data.bus_Pnet_all = bus_Pnet_all
    sys_data.bus_ZIB_all = bus_ZIB_all
    sys_data.NZIB = NZIB
    
    print(f"Non-ZIB buses: {len(bus_Pnet_all)}, ZIB buses: {NZIB}")
    
    # Find slack bus position in non-ZIB array
    idx_bus_Pnet_slack = np.where(bus_Pnet_all == bus_slack)[0]
    bus_Pnet_noslack_all = np.delete(bus_Pnet_all, idx_bus_Pnet_slack, axis=0)
    
    sys_data.bus_Pnet_noslack_all = bus_Pnet_noslack_all
    sys_data.idx_bus_Pnet_slack = idx_bus_Pnet_slack
    
    # Prediction dimensions
    NPred_Vm = len(bus_Pnet_all)
    NPred_Va = len(bus_Pnet_noslack_all)
    output_dim = NPred_Va + NPred_Vm
    
    sys_data.NPred_Vm = NPred_Vm
    sys_data.NPred_Va = NPred_Va
    
    print(f"Prediction dims: Va={NPred_Va}, Vm={NPred_Vm}, Total={output_dim}")
    
    # ============================================================
    # Random sampling (exactly matching reference code)
    # ============================================================
    Ntrain = config.ngt_Ntrain
    Ntest = config.ngt_Ntest
    Nhis = config.ngt_Nhis
    Nsample = config.ngt_Nsample
    
    # Sample indices
    idx_sample = random.sample(range(0, RPd.shape[0]), Nsample)
    idx_train = np.asarray(idx_sample[0:Ntrain])
    idx_train_label = np.asarray(idx_sample[Ntrain:Nhis + Ntrain])
    idx_test = np.asarray(idx_sample[-Ntest:])
    idx_his = idx_train_label
    
    print(f"Sample indices: train={idx_train.shape}, his={idx_his.shape}, test={idx_test.shape}")
    
    # ============================================================
    # Prepare input data (exactly matching reference code)
    # Input: [Pd_nonzero, Qd_nonzero] / baseMVA
    # ============================================================
    x = np.concatenate((RPd[:, bus_Pd], RQd[:, bus_Qd]), axis=1) / baseMVA
    input_dim = x.shape[1]
    
    print(f"Input data: shape={x.shape}")
    
    # ============================================================
    # Prepare output data (Vm, Va for non-ZIB nodes)
    # ============================================================
    yvm = RVm
    yva = RVa
    
    print(f"Voltage data: Vm=[{np.min(yvm):.4f}, {np.max(yvm):.4f}], "
          f"Va=[{np.min(yva):.4f}, {np.max(yva):.4f}] rad")
    
    # Convert to tensors
    x_tensor = torch.from_numpy(x).float()
    yva_tensor = torch.from_numpy(yva).float()
    yvm_tensor = torch.from_numpy(yvm).float()
    
    # Extract training and test data
    x_train = x_tensor[idx_train]
    x_test = x_tensor[idx_test]
    
    yva_train = yva_tensor[idx_train]
    yvm_train = yvm_tensor[idx_train]
    yva_test = yva_tensor[idx_test]
    yvm_test = yvm_tensor[idx_test]
    
    # Prepare combined output for non-ZIB nodes (matching reference yvtrain_Pnet)
    # [Va_nonZIB_noslack, Vm_nonZIB]
    y_train = torch.cat((
        yva_train[:, bus_Pnet_noslack_all.tolist()],
        yvm_train[:, bus_Pnet_all.tolist()]
    ), dim=1)
    
    y_test = torch.cat((
        yva_test[:, bus_Pnet_noslack_all.tolist()],
        yvm_test[:, bus_Pnet_all.tolist()]
    ), dim=1)
    
    print(f"Combined output: train={y_train.shape}, test={y_test.shape}")
    
    # ============================================================
    # Compute Vscale and Vbias (exactly matching reference code)
    # ============================================================
    VmLb = config.ngt_VmLb
    VmUb = config.ngt_VmUb
    VaLb = config.ngt_VaLb
    VaUb = config.ngt_VaUb
    
    # Va part
    Vascale = torch.ones(NPred_Va) * (VaUb - VaLb)
    Vabias = torch.ones(NPred_Va) * VaLb
    
    # Vm part
    Vmscale = torch.ones(NPred_Vm) * (VmUb - VmLb)
    Vmbias = torch.ones(NPred_Vm) * VmLb
    
    # Combined
    Vscale = torch.cat((Vascale, Vmscale), dim=0)
    Vbias = torch.cat((Vabias, Vmbias), dim=0)
    
    print(f"Vscale: shape={Vscale.shape}, range=[{Vscale.min():.4f}, {Vscale.max():.4f}]")
    print(f"Vbias: shape={Vbias.shape}, range=[{Vbias.min():.4f}, {Vbias.max():.4f}]")
    
    # Store voltage bounds
    sys_data.VmLb = torch.tensor([VmLb])
    sys_data.VmUb = torch.tensor([VmUb])
    
    # ============================================================
    # Prepare PQd data (load data for loss function)
    # ============================================================
    # PQd format: [Pd at bus_Pd, Qd at bus_Qd] in p.u.
    PQd_train = torch.from_numpy(np.concatenate([
        RPd[idx_train][:, bus_Pd] / baseMVA,
        RQd[idx_train][:, bus_Qd] / baseMVA
    ], axis=1)).float()
    
    PQd_test = torch.from_numpy(np.concatenate([
        RPd[idx_test][:, bus_Pd] / baseMVA,
        RQd[idx_test][:, bus_Qd] / baseMVA
    ], axis=1)).float()
    
    # ============================================================
    # Compute historical voltage for post-processing
    # ============================================================
    his_Vm = yvm_tensor[idx_his].numpy()
    his_Va = yva_tensor[idx_his].numpy()
    his_V = np.mean(his_Vm * np.exp(1j * his_Va), axis=0)
    
    sys_data.his_V = his_V
    sys_data.his_Vm = np.mean(his_Vm, axis=0)
    sys_data.his_Va = np.mean(his_Va, axis=0)
    
    # ============================================================
    # Store additional data in sys_data
    # ============================================================
    sys_data.bus_Pd = bus_Pd
    sys_data.bus_Qd = bus_Qd
    sys_data.gencost = sys_data.gencost
    
    # Generator limits
    sys_data.MAXMIN_Pg = gen[idxPg, 3:5] / baseMVA
    sys_data.MAXMIN_Qg = gen[idxQg, 1:3] / baseMVA
    
    # Test data for evaluation
    sys_data.Pdtest = RPd[idx_test] / baseMVA
    sys_data.Qdtest = RQd[idx_test] / baseMVA
    sys_data.Pgtest = RPg[idx_test][:, idxPg] / baseMVA
    sys_data.Qgtest = RQg[idx_test][:, idxQg] / baseMVA
    
    # Store test voltage for evaluation
    sys_data.yvm_test = yvm_test
    sys_data.yva_test = yva_test
    
    print("=" * 60)
    print("DeepOPF-NGT Data Loading Complete")
    print(f"  Training samples: {Ntrain}")
    print(f"  Test samples: {Ntest}")
    print(f"  Input dim: {input_dim}")
    print(f"  Output dim: {output_dim}")
    print("=" * 60)
    
    # ============================================================
    # Compute Kron Reduction parameter (param_ZIMV)
    # For recovering ZIB voltages: Vy = param_ZIMV @ Vx
    # ============================================================
    if NZIB > 0:
        Ybus = sys_data.Ybus
        Ya = Ybus[np.ix_(bus_ZIB_all, bus_ZIB_all)]
        Yb = Ybus[np.ix_(bus_ZIB_all, bus_Pnet_all)]
        
        # Only invert if Ya is square and non-singular
        if Ya.shape[0] == Ya.shape[1] and np.linalg.matrix_rank(Ya.toarray()) == Ya.shape[0]:
            param_ZIMV = -scipy.sparse.linalg.inv(Ya) @ Yb
        else:
            param_ZIMV = None
            print("[Warning] Cannot compute param_ZIMV - Ya is singular")
    else:
        param_ZIMV = None
    
    # ============================================================
    # Extract gencost for Pg
    # ============================================================
    gencost = sys_data.gencost
    gencost_Pg = gencost[idxPg, :2]  # [c2, c1] coefficients
    
    ngt_data = {
        # Training and test data
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test,
        'PQd_train': PQd_train,
        'PQd_test': PQd_test,
        'Vscale': Vscale,
        'Vbias': Vbias,
        'his_V': his_V,
        'idx_train': idx_train,
        'idx_test': idx_test,
        'idx_his': idx_his,
        
        # Dimensions
        'input_dim': input_dim,
        'output_dim': output_dim,
        'NPred_Va': NPred_Va,
        'NPred_Vm': NPred_Vm,
        
        # Bus indices
        'bus_Pd': bus_Pd,
        'bus_Qd': bus_Qd,
        'bus_Pnet_all': bus_Pnet_all,
        'bus_ZIB_all': bus_ZIB_all,
        'idx_bus_Pnet_slack': idx_bus_Pnet_slack,
        'NZIB': NZIB,
        
        # Kron Reduction parameter
        'param_ZIMV': param_ZIMV.toarray() if param_ZIMV is not None else None,
        
        # Generator limits and costs
        'gencost_Pg': gencost_Pg,
        'MAXMIN_Pg': sys_data.MAXMIN_Pg,
        'MAXMIN_Qg': sys_data.MAXMIN_Qg,
        
        # Test voltage for evaluation
        'yvm_test': yvm_test,
        'yva_test': yva_test,
    }
    
    return ngt_data, sys_data


def create_ngt_training_loader(ngt_data, config):
    """
    Create DataLoader for DeepOPF-NGT unsupervised training.
    
    Args:
        ngt_data: Dictionary from load_ngt_training_data()
        config: Configuration object
        
    Returns:
        training_loader: DataLoader for training
    """
    # Create dataset: (x_train, y_train) - y is for reference only
    training_dataset = Data.TensorDataset(ngt_data['x_train'], ngt_data['y_train'])
    
    training_loader = Data.DataLoader(
        dataset=training_dataset,
        batch_size=config.ngt_batch_size,
        shuffle=True,
    )
    
    print(f"[DeepOPF-NGT] Training DataLoader: {len(training_loader)} batches, "
          f"batch_size={config.ngt_batch_size}")
    
    return training_loader


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
    
    # Test NGT data loading
    print("\n" + "=" * 60)
    print("Testing DeepOPF-NGT Data Loading")
    print("=" * 60)
    
    ngt_data, sys_data_ngt = load_ngt_training_data(config)
    training_loader_ngt = create_ngt_training_loader(ngt_data, config)
    
    # Test one batch
    for batch_x, batch_y in training_loader_ngt:
        print(f"\nNGT Sample batch shapes:")
        print(f"  Input (x): {batch_x.shape}")
        print(f"  Output (y): {batch_y.shape}")
        break

