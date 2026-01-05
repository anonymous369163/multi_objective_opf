#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Format Converter for Multi-Preference OPF
==============================================

This module provides functions to convert between different data formats:
1. x_train: FULL-BUS -> SPARSE
2. y_train: FULL-BUS -> NGT  
3. y_pred:  NGT -> FULL-BUS

Dependencies: numpy only (no external power system libraries required)
Optional: torch (for PyTorch tensor support)

==============================================================================
DATA FORMAT DEFINITIONS
==============================================================================

1. x_train (Input Load Data):
   - FULL-BUS format: [Pd_all, Qd_all] = 2*Nbus columns (e.g., 236 for case118)
     - Pd_all: Active power demand for ALL buses (in p.u.)
     - Qd_all: Reactive power demand for ALL buses (in p.u.)
   
   - SPARSE format: [Pd_nonzero, Qd_nonzero] = len(bus_Pd) + len(bus_Qd) columns (e.g., 189 for case118)
     - Pd_nonzero: Active power only for buses with non-zero load
     - Qd_nonzero: Reactive power only for buses with non-zero load
     - This is the format expected by DeepOPF-NGT loss function!

2. y_train (Output Voltage Data):
   - FULL-BUS format: [Va_noslack, Vm_all] = 2*Nbus-1 columns (e.g., 235 for case118)
     - Va_noslack: Voltage angles for all buses except slack (in radians)
     - Vm_all: Voltage magnitudes for all buses (in p.u.)
   
   - NGT format: [Va_nonZIB_noslack, Vm_nonZIB] = NPred_Va + NPred_Vm columns (e.g., 215 for case118)
     - Va_nonZIB_noslack: Voltage angles for non-ZIB buses (excluding slack)
     - Vm_nonZIB: Voltage magnitudes for non-ZIB buses
     - ZIB = Zero Injection Bus (no load and no generator)
     - ZIB values can be reconstructed using Kron reduction

==============================================================================
USAGE EXAMPLES
==============================================================================

Example 1: Basic Usage
----------------------
```python
from format_converter import FormatConverter

# Initialize with case file
converter = FormatConverter('main_part/data/case118_ieee_modified.m')

# Convert x_train from FULL-BUS to SPARSE (for NGT loss function)
x_sparse = converter.x_fullbus_to_sparse(x_fullbus)
# Result: [N, 189] from [N, 236]

# Convert y_train from FULL-BUS to NGT (for model training)
y_ngt = converter.y_fullbus_to_ngt(y_fullbus)
# Result: [N, 215] from [N, 235]

# Reconstruct y from NGT to FULL-BUS (for evaluation)
y_fullbus = converter.y_ngt_to_fullbus(y_ngt)
# Result: [N, 235] from [N, 215]
```

Example 2: With PyTorch Tensors
-------------------------------
```python
import torch

# Works with both numpy arrays and PyTorch tensors
x_sparse_tensor = converter.x_fullbus_to_sparse(x_fullbus_tensor)
y_ngt_tensor = converter.y_fullbus_to_ngt(y_fullbus_tensor)
```

Example 3: Get Index Information
--------------------------------
```python
info = converter.get_info()
print(f"Load buses: Pd={len(info['bus_Pd'])}, Qd={len(info['bus_Qd'])}")
print(f"Non-ZIB buses: {info['NPred_Vm']}")
print(f"ZIB buses: {info['NZIB']}")
```

Author: Auto-generated
Date: 2026-01-03
"""

import re
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from typing import Tuple, Optional, Dict, Any, Union

# Optional PyTorch support
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class FormatConverter:
    """
    Data format converter for OPF problems.
    
    Handles conversion between:
    - FULL-BUS format: [all buses data]
    - SPARSE format: [non-zero load buses data]
    - NGT format: [non-ZIB buses data, excluding slack for Va]
    """
    
    def __init__(self, case_m_path: str, verbose: bool = True):
        """
        Initialize converter by parsing case file.
        
        Args:
            case_m_path: Path to MATPOWER .m case file
            verbose: Whether to print initialization info
        """
        self.case_m_path = case_m_path
        self.verbose = verbose
        
        # Parse case file
        self._parse_case_file()
        
        # Identify node types
        self._identify_node_types()
        
        # Compute Ybus and Kron matrix
        self._compute_ybus()
        self._compute_kron_matrix()
        
        if verbose:
            self._print_info()
    
    def _parse_case_file(self):
        """Parse MATPOWER .m file to extract bus, gen, and branch data."""
        with open(self.case_m_path, 'r', encoding='utf-8', errors='ignore') as f:
            txt = f.read()
        
        # Parse baseMVA
        m = re.search(r"mpc\.baseMVA\s*=\s*([0-9eE\.\+\-]+)", txt)
        if m:
            self.baseMVA = float(m.group(1))
        else:
            self.baseMVA = 100.0
        
        # Parse bus data
        self.bus = self._extract_matrix(txt, 'bus')
        self.Nbus = self.bus.shape[0]
        
        # Create bus_id to row_index mapping (for non-consecutive bus numbering)
        # bus[:, 0] contains the actual bus IDs from MATPOWER file
        self.bus_ids = self.bus[:, 0].astype(int)  # Original bus IDs (1-indexed in MATPOWER)
        self.bus_id_to_idx = {bus_id: idx for idx, bus_id in enumerate(self.bus_ids)}
        
        # Parse generator data
        self.gen = self._extract_matrix(txt, 'gen')
        self.Ngen = self.gen.shape[0]
        
        # Parse branch data
        self.branch = self._extract_matrix(txt, 'branch')
        self.Nbranch = self.branch.shape[0]
    
    def _extract_matrix(self, txt: str, name: str) -> np.ndarray:
        """Extract matrix from MATPOWER .m file."""
        m = re.search(rf"mpc\.{re.escape(name)}\s*=\s*\[", txt)
        if not m:
            raise KeyError(f"Cannot find 'mpc.{name}' in .m file")
        
        start = m.end()
        end = txt.find("];", start)
        block = txt[start:end]
        
        # Clean comments and continuation
        lines = []
        for line in block.splitlines():
            line = line.split("%", 1)[0].replace("...", " ")
            if line.strip():
                lines.append(line)
        
        # Parse values
        rows = [r.strip() for r in "\n".join(lines).split(";") if r.strip()]
        float_re = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
        data = [[float(x) for x in re.findall(float_re, r)] for r in rows]
        maxlen = max(len(row) for row in data)
        return np.array([row + [np.nan] * (maxlen - len(row)) for row in data])
    
    def _bus_id_to_row_idx(self, bus_id: int) -> int:
        """Convert MATPOWER bus ID (1-indexed) to row index (0-indexed).
        
        This handles non-consecutive bus numbering in MATPOWER files.
        """
        if bus_id in self.bus_id_to_idx:
            return self.bus_id_to_idx[bus_id]
        else:
            raise ValueError(f"Bus ID {bus_id} not found in bus data")
    
    def _bus_ids_to_row_indices(self, bus_ids: np.ndarray) -> np.ndarray:
        """Convert array of MATPOWER bus IDs to row indices."""
        return np.array([self.bus_id_to_idx[int(bid)] for bid in bus_ids])
    
    def _identify_node_types(self):
        """Identify different types of buses."""
        Nbus = self.Nbus
        
        # Bus data columns: bus_i, type, Pd, Qd, Gs, Bs, area, Vm, Va, baseKV, zone, Vmax, Vmin 1=PQ, 2=PV, 3=slack  
        bus_type = self.bus[:, 1]
        Pd_base = self.bus[:, 2]   # Active load (MW)
        Qd_base = self.bus[:, 3]   # Reactive load (MVAr)
        
        # Identify slack bus (type=3)
        slack_idx = np.where(bus_type == 3)[0]
        self.bus_slack = int(slack_idx[0]) if len(slack_idx) > 0 else 0
        
        # Identify load buses (Pd > 0 or Qd > 0)
        # Note: We use base case Pd/Qd to identify which buses CAN have loads
        self.bus_Pd = np.where(np.abs(Pd_base) > 0)[0]  # Buses with active load (row indices)
        self.bus_Qd = np.where(np.abs(Qd_base) > 0)[0]  # Buses with reactive load (row indices)
        
        # Generator data columns: bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin
        # gen[:, 0] contains MATPOWER bus IDs (not necessarily consecutive)
        gen_bus_ids = self.gen[:, 0].astype(int)  # Original bus IDs from MATPOWER
        # Convert to row indices using mapping
        gen_bus = self._bus_ids_to_row_indices(gen_bus_ids)  # Row indices (0-indexed)
        
        Pmax = self.gen[:, 8]  # Maximum active power
        Qmax = self.gen[:, 3]  # Maximum reactive power
        
        # Identify generators with Pmax > 0 (active power generators)
        idxPg = np.where(Pmax > 0)[0]
        self.bus_Pg = gen_bus[idxPg]
        
        # Identify generators with Qmax > 0 (reactive power generators)
        # For IEEE 118, all generators can provide reactive power
        idxQg = np.where(Qmax > 0)[0]
        self.bus_Qg = gen_bus[idxQg]
        
        # All generator buses (row indices)
        bus_gen_all = np.unique(gen_bus)
        
        # Identify Non-ZIB buses (have load OR generation)
        # ZIB = Zero Injection Bus (no load and no generation)
        has_load = np.zeros(Nbus, dtype=bool)
        has_load[self.bus_Pd] = True
        has_load[self.bus_Qd] = True
        
        has_gen = np.zeros(Nbus, dtype=bool)
        has_gen[bus_gen_all] = True
        
        non_zib_mask = has_load | has_gen
        self.bus_Pnet_all = np.where(non_zib_mask)[0]  # Non-ZIB buses (row indices)
        self.bus_ZIB_all = np.where(~non_zib_mask)[0]  # ZIB buses (row indices)
        
        # Non-ZIB buses excluding slack
        self.bus_Pnet_noslack_all = self.bus_Pnet_all[self.bus_Pnet_all != self.bus_slack]
        
        # Dimensions
        self.NPred_Vm = len(self.bus_Pnet_all)      # Number of Vm to predict (non-ZIB)
        self.NPred_Va = len(self.bus_Pnet_noslack_all)  # Number of Va to predict (non-ZIB, no slack)
        self.NZIB = len(self.bus_ZIB_all)           # Number of ZIB buses
        
        # Compute dimensions
        self.input_dim_fullbus = 2 * Nbus  # [Pd_all, Qd_all]
        self.input_dim_sparse = len(self.bus_Pd) + len(self.bus_Qd)
        self.output_dim_fullbus = 2 * Nbus - 1  # [Va_noslack, Vm_all]
        self.output_dim_ngt = self.NPred_Va + self.NPred_Vm
    
    def _compute_ybus(self):
        """
        Compute bus admittance matrix (Ybus) from branch data.
        
        This follows MATPOWER/PYPOWER makeYbus implementation.
        """
        baseMVA = self.baseMVA
        bus = self.bus
        branch = self.branch
        nb = self.Nbus
        nl = self.Nbranch
        
        # Bus data columns (0-indexed)
        GS = 4   # Gs (shunt conductance)
        BS = 5   # Bs (shunt susceptance)
        
        # Branch data columns (0-indexed)
        F_BUS = 0      # from bus
        T_BUS = 1      # to bus
        BR_R = 2       # resistance
        BR_X = 3       # reactance
        BR_B = 4       # total line charging susceptance
        TAP = 8        # transformer tap ratio
        SHIFT = 9      # transformer phase shift angle (degrees)
        BR_STATUS = 10 # branch status (1=in service, 0=out of service)
        
        # Get branch from/to buses (convert bus IDs to row indices)
        # Use mapping to handle non-consecutive bus numbering
        f = self._bus_ids_to_row_indices(branch[:, F_BUS])
        t = self._bus_ids_to_row_indices(branch[:, T_BUS])
        
        # Branch parameters
        status = branch[:, BR_STATUS]
        r = branch[:, BR_R]
        x = branch[:, BR_X]
        b = branch[:, BR_B]
        
        # Series impedance and admittance
        z = r + 1j * x
        y = status / z
        bsh = status * 1j * b / 2.0  # Half of line charging susceptance
        
        # Transformer tap ratio (complex with phase shift)
        tap = branch[:, TAP].copy()
        tap = np.where(tap == 0, 1.0, tap)  # Default tap = 1.0
        shift = branch[:, SHIFT] * np.pi / 180.0  # Convert to radians
        tap = tap * np.exp(1j * shift)
        
        # Branch admittance matrix elements
        Yff = (y + bsh) / (tap * np.conj(tap))
        Yft = -y / np.conj(tap)
        Ytf = -y / tap
        Ytt = y + bsh
        
        # Build Ybus from branch contributions
        data = np.hstack([Yff, Yft, Ytf, Ytt])
        row = np.hstack([f, f, t, t])
        col = np.hstack([f, t, f, t])
        Ybus = sparse.coo_matrix((data, (row, col)), shape=(nb, nb)).tocsr()
        
        # Add shunt admittance (from bus data)
        Ysh = (bus[:, GS] + 1j * bus[:, BS]) / baseMVA
        Ybus = Ybus + sparse.diags(Ysh, 0, shape=(nb, nb), format="csr")
        
        self.Ybus = Ybus
    
    def _compute_kron_matrix(self):
        """
        Compute Kron reduction matrix (param_ZIMV) for reconstructing ZIB voltages.
        
        For ZIB (Zero Injection Bus) nodes, the voltage can be reconstructed using:
            V_ZIB = param_ZIMV @ V_nonZIB
        
        where param_ZIMV = -inv(Ya) @ Yb
            Ya = Ybus[ZIB, ZIB]  - Admittance matrix for ZIB nodes
            Yb = Ybus[ZIB, nonZIB] - Admittance matrix from ZIB to non-ZIB
        """
        if self.NZIB == 0:
            self.param_ZIMV = None
            return
        
        Ybus = self.Ybus
        bus_ZIB = self.bus_ZIB_all
        bus_nonZIB = self.bus_Pnet_all
        
        # Extract submatrices
        Ya = Ybus[np.ix_(bus_ZIB, bus_ZIB)]  # [NZIB, NZIB]
        Yb = Ybus[np.ix_(bus_ZIB, bus_nonZIB)]  # [NZIB, NPred_Vm]
        
        # Check if Ya is invertible
        Ya_dense = Ya.toarray()
        if np.linalg.matrix_rank(Ya_dense) == Ya.shape[0]:
            # Compute Kron reduction matrix: param_ZIMV = -inv(Ya) @ Yb
            try:
                Ya_inv = sparse_linalg.inv(Ya.tocsc())
                param_ZIMV = -Ya_inv @ Yb
                self.param_ZIMV = param_ZIMV.toarray()
            except Exception as e:
                print(f"[Warning] Cannot compute param_ZIMV: {e}")
                self.param_ZIMV = None
        else:
            print(f"[Warning] Cannot compute param_ZIMV - Ya is singular (rank={np.linalg.matrix_rank(Ya_dense)}/{Ya.shape[0]})")
            self.param_ZIMV = None
    
    def _print_info(self):
        """Print converter initialization info."""
        print(f"\n{'='*60}")
        print(f"FormatConverter initialized")
        print(f"{'='*60}")
        print(f"Case file: {self.case_m_path}")
        print(f"Nbus: {self.Nbus}, Nbranch: {self.Nbranch}, baseMVA: {self.baseMVA}")
        print(f"Slack bus: {self.bus_slack}")
        print(f"Load buses: Pd={len(self.bus_Pd)}, Qd={len(self.bus_Qd)}")
        print(f"Generator buses: Pg={len(self.bus_Pg)}, Qg={len(self.bus_Qg)}")
        print(f"Non-ZIB buses: {self.NPred_Vm}, ZIB buses: {self.NZIB}")
        print(f"Kron matrix: {'Computed' if self.param_ZIMV is not None else 'Not available'}")
        if self.param_ZIMV is not None:
            print(f"  param_ZIMV shape: {self.param_ZIMV.shape}")
        print(f"\nDimensions:")
        print(f"  x FULL-BUS: {self.input_dim_fullbus} = [Pd_all({self.Nbus}), Qd_all({self.Nbus})]")
        print(f"  x SPARSE:   {self.input_dim_sparse} = [Pd({len(self.bus_Pd)}), Qd({len(self.bus_Qd)})]")
        print(f"  y FULL-BUS: {self.output_dim_fullbus} = [Va_noslack({self.Nbus-1}), Vm_all({self.Nbus})]")
        print(f"  y NGT:      {self.output_dim_ngt} = [Va({self.NPred_Va}), Vm({self.NPred_Vm})]")
        print(f"{'='*60}\n")
    
    # =========================================================================
    # Conversion 1: x_train FULL-BUS -> SPARSE
    # =========================================================================
    
    def x_fullbus_to_sparse(self, x_fullbus: Union[np.ndarray, 'torch.Tensor']) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Convert x from FULL-BUS format to SPARSE format.
        
        FULL-BUS format: [Pd_all (Nbus), Qd_all (Nbus)] = 2*Nbus columns
        SPARSE format:   [Pd_nonzero, Qd_nonzero] = len(bus_Pd) + len(bus_Qd) columns
        
        Args:
            x_fullbus: Input array of shape [N, 2*Nbus] in FULL-BUS format
                       Units: p.u. (per unit)
                       Can be numpy array or PyTorch tensor
        
        Returns:
            x_sparse: Array of shape [N, input_dim_sparse] in SPARSE format
                      Same type as input (numpy or tensor)
        
        Example:
            >>> x_sparse = converter.x_fullbus_to_sparse(x_fullbus)
            >>> print(f"Shape: {x_fullbus.shape} -> {x_sparse.shape}")
        """
        # Check if input is tensor
        is_tensor = HAS_TORCH and isinstance(x_fullbus, torch.Tensor)
        if is_tensor:
            x = x_fullbus.cpu().numpy()
            original_dtype = x_fullbus.dtype
            original_device = x_fullbus.device
        else:
            x = np.asarray(x_fullbus)
        
        was_1d = x.ndim == 1
        if was_1d:
            x = x.reshape(1, -1)
        
        assert x.shape[1] == 2 * self.Nbus, \
            f"Expected x.shape[1] = {2*self.Nbus}, got {x.shape[1]}"
        
        # Split into Pd_all and Qd_all
        Pd_all = x[:, :self.Nbus]
        Qd_all = x[:, self.Nbus:]
        
        # Extract non-zero load buses
        Pd_sparse = Pd_all[:, self.bus_Pd]
        Qd_sparse = Qd_all[:, self.bus_Qd]
        
        # Combine
        x_sparse = np.concatenate([Pd_sparse, Qd_sparse], axis=1)
        
        if was_1d:
            x_sparse = x_sparse.squeeze()
        
        # Convert back to tensor if needed
        if is_tensor:
            x_sparse = torch.from_numpy(x_sparse).to(dtype=original_dtype, device=original_device)
        
        return x_sparse
    
    def x_sparse_to_fullbus(self, x_sparse: Union[np.ndarray, 'torch.Tensor']) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Convert x from SPARSE format to FULL-BUS format (inverse).
        
        Args:
            x_sparse: Input array of shape [N, input_dim_sparse] in SPARSE format
                      Can be numpy array or PyTorch tensor
        
        Returns:
            x_fullbus: Array of shape [N, 2*Nbus] in FULL-BUS format
                       Same type as input
        """
        # Check if input is tensor
        is_tensor = HAS_TORCH and isinstance(x_sparse, torch.Tensor)
        if is_tensor:
            x = x_sparse.cpu().numpy()
            original_dtype = x_sparse.dtype
            original_device = x_sparse.device
        else:
            x = np.asarray(x_sparse)
        
        was_1d = x.ndim == 1
        if was_1d:
            x = x.reshape(1, -1)
        
        N = x.shape[0]
        num_Pd = len(self.bus_Pd)
        num_Qd = len(self.bus_Qd)
        
        # Initialize full arrays
        Pd_all = np.zeros((N, self.Nbus))
        Qd_all = np.zeros((N, self.Nbus))
        
        # Fill in non-zero values
        Pd_all[:, self.bus_Pd] = x[:, :num_Pd]
        Qd_all[:, self.bus_Qd] = x[:, num_Pd:num_Pd + num_Qd]
        
        x_fullbus = np.concatenate([Pd_all, Qd_all], axis=1)
        
        if was_1d:
            x_fullbus = x_fullbus.squeeze()
        
        # Convert back to tensor if needed
        if is_tensor:
            x_fullbus = torch.from_numpy(x_fullbus).to(dtype=original_dtype, device=original_device)
        
        return x_fullbus
    
    # =========================================================================
    # Conversion 2: y_train FULL-BUS -> NGT
    # =========================================================================
    
    def y_fullbus_to_ngt(self, y_fullbus: Union[np.ndarray, 'torch.Tensor']) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Convert y from FULL-BUS format to NGT format.
        
        FULL-BUS format: [Va_noslack (Nbus-1), Vm_all (Nbus)] = 2*Nbus-1 columns
        NGT format:      [Va_noslack_nonZIB, Vm_nonZIB] = NPred_Va + NPred_Vm columns
        
        This removes ZIB (Zero Injection Bus) nodes, which can be reconstructed
        using Kron reduction.
        
        Args:
            y_fullbus: Input array of shape [N, 2*Nbus-1] in FULL-BUS format
                       Va in radians, Vm in p.u.
                       Can be numpy array or PyTorch tensor
        
        Returns:
            y_ngt: Array of shape [N, output_dim_ngt] in NGT format
                   Same type as input
        
        Example:
            >>> y_ngt = converter.y_fullbus_to_ngt(y_fullbus)
            >>> print(f"Shape: {y_fullbus.shape} -> {y_ngt.shape}")
        """
        # Check if input is tensor
        is_tensor = HAS_TORCH and isinstance(y_fullbus, torch.Tensor)
        if is_tensor:
            y = y_fullbus.cpu().numpy()
            original_dtype = y_fullbus.dtype
            original_device = y_fullbus.device
        else:
            y = np.asarray(y_fullbus)
        
        was_1d = y.ndim == 1
        if was_1d:
            y = y.reshape(1, -1)
        
        Nbus = self.Nbus
        expected_dim = 2 * Nbus - 1
        assert y.shape[1] == expected_dim, \
            f"Expected y.shape[1] = {expected_dim}, got {y.shape[1]}"
        
        # Split into Va_noslack and Vm_all
        Va_noslack = y[:, :Nbus - 1]  # [N, Nbus-1]
        Vm_all = y[:, Nbus - 1:]      # [N, Nbus]
        
        # Build mapping from full bus index to Va_noslack index
        # Va_noslack[i] corresponds to:
        #   bus i     if i < slack
        #   bus i+1   if i >= slack
        def bus_to_va_idx(bus_idx):
            if bus_idx < self.bus_slack:
                return bus_idx
            else:
                return bus_idx - 1
        
        # Extract Va for non-ZIB, non-slack buses
        va_indices = np.array([bus_to_va_idx(b) for b in self.bus_Pnet_noslack_all])
        Va_ngt = Va_noslack[:, va_indices]
        
        # Extract Vm for non-ZIB buses
        Vm_ngt = Vm_all[:, self.bus_Pnet_all]
        
        # Combine
        y_ngt = np.concatenate([Va_ngt, Vm_ngt], axis=1)
        
        if was_1d:
            y_ngt = y_ngt.squeeze()
        
        # Convert back to tensor if needed
        if is_tensor:
            y_ngt = torch.from_numpy(y_ngt).to(dtype=original_dtype, device=original_device)
        
        return y_ngt
    
    # =========================================================================
    # Conversion 3: y NGT -> FULL-BUS
    # =========================================================================
    
    def y_ngt_to_fullbus(self, y_ngt: Union[np.ndarray, 'torch.Tensor'], 
                         param_ZIMV: Optional[np.ndarray] = None,
                         use_kron: bool = True) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Convert y from NGT format back to FULL-BUS format.
        
        NGT format:      [Va_noslack_nonZIB, Vm_nonZIB]
        FULL-BUS format: [Va_noslack (Nbus-1), Vm_all (Nbus)]
        
        This reconstructs ZIB node voltages using Kron reduction.
        
        Args:
            y_ngt: Input array of shape [N, output_dim_ngt] in NGT format
                   Can be numpy array or PyTorch tensor
            param_ZIMV: Optional Kron reconstruction matrix for ZIB nodes.
                        If None and use_kron=True, uses the pre-computed self.param_ZIMV.
                        Shape: [NZIB, NPred_Vm] (complex matrix)
            use_kron: Whether to use Kron reconstruction for ZIB nodes.
                      Default True. If False, ZIB nodes will be zeros.
        
        Returns:
            y_fullbus: Array of shape [N, 2*Nbus-1] in FULL-BUS format
                       Same type as input
        
        Example:
            >>> # Automatic Kron reconstruction (default)
            >>> y_fullbus = converter.y_ngt_to_fullbus(y_ngt)
            >>> # Without Kron reconstruction (ZIB = 0)
            >>> y_fullbus = converter.y_ngt_to_fullbus(y_ngt, use_kron=False)
            >>> # Custom Kron matrix
            >>> y_fullbus = converter.y_ngt_to_fullbus(y_ngt, param_ZIMV=custom_matrix)
        """
        # Use pre-computed param_ZIMV if not provided
        if param_ZIMV is None and use_kron:
            param_ZIMV = self.param_ZIMV
        
        # Check if input is tensor
        is_tensor = HAS_TORCH and isinstance(y_ngt, torch.Tensor)
        if is_tensor:
            y = y_ngt.cpu().numpy()
            original_dtype = y_ngt.dtype
            original_device = y_ngt.device
        else:
            y = np.asarray(y_ngt)
        
        was_1d = y.ndim == 1
        if was_1d:
            y = y.reshape(1, -1)
        
        N = y.shape[0]
        Nbus = self.Nbus
        
        # Split into Va and Vm (NGT format)
        Va_ngt = y[:, :self.NPred_Va]    # [N, NPred_Va]
        Vm_ngt = y[:, self.NPred_Va:]    # [N, NPred_Vm]
        
        # Initialize full arrays
        Va_noslack = np.zeros((N, Nbus - 1))  # [N, Nbus-1]
        Vm_all = np.zeros((N, Nbus))           # [N, Nbus]
        
        # Mapping from bus index to Va_noslack index
        def bus_to_va_idx(bus_idx):
            if bus_idx < self.bus_slack:
                return bus_idx
            else:
                return bus_idx - 1
        
        # Fill Va for non-ZIB, non-slack buses
        for i, bus_idx in enumerate(self.bus_Pnet_noslack_all):
            va_idx = bus_to_va_idx(bus_idx)
            Va_noslack[:, va_idx] = Va_ngt[:, i]
        
        # Fill Vm for non-ZIB buses
        Vm_all[:, self.bus_Pnet_all] = Vm_ngt
        
        # Kron reconstruction for ZIB buses (if available)
        if param_ZIMV is not None and self.NZIB > 0:
            Vm_all, Va_noslack = self._kron_reconstruct_zib(
                Vm_all, Va_noslack, param_ZIMV
            )
        
        # Combine [Va_noslack, Vm_all]
        y_fullbus = np.concatenate([Va_noslack, Vm_all], axis=1)
        
        if was_1d:
            y_fullbus = y_fullbus.squeeze()
        
        # Convert back to tensor if needed
        if is_tensor:
            y_fullbus = torch.from_numpy(y_fullbus).to(dtype=original_dtype, device=original_device)
        
        return y_fullbus
    
    def _kron_reconstruct_zib(self, Vm_all: np.ndarray, Va_noslack: np.ndarray,
                               param_ZIMV: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct ZIB node voltages using Kron reduction.
        
        V_ZIB = param_ZIMV @ V_nonZIB
        
        Args:
            Vm_all: [N, Nbus] voltage magnitudes (ZIB positions are zeros)
            Va_noslack: [N, Nbus-1] voltage angles (ZIB positions are zeros)
            param_ZIMV: [NZIB, NPred_Vm] complex Kron reconstruction matrix
        
        Returns:
            Vm_all, Va_noslack: Arrays with ZIB values filled in
        """
        N = Vm_all.shape[0]
        Nbus = self.Nbus
        
        # Build full Va (with slack = 0)
        Va_full = np.zeros((N, Nbus))
        Va_full[:, :self.bus_slack] = Va_noslack[:, :self.bus_slack]
        Va_full[:, self.bus_slack] = 0  # slack
        Va_full[:, self.bus_slack + 1:] = Va_noslack[:, self.bus_slack:]
        
        # Extract complex voltage for non-ZIB buses
        V_nonZIB = Vm_all[:, self.bus_Pnet_all] * np.exp(1j * Va_full[:, self.bus_Pnet_all])
        
        # Reconstruct ZIB voltages: V_ZIB = param_ZIMV @ V_nonZIB
        param_ZIMV = np.asarray(param_ZIMV)
        V_ZIB = np.dot(V_nonZIB, param_ZIMV.T)  # [N, NZIB]
        
        # Extract Vm and Va for ZIB buses
        Vm_ZIB = np.abs(V_ZIB)
        Va_ZIB = np.angle(V_ZIB)
        
        # Fill ZIB values
        Vm_all[:, self.bus_ZIB_all] = Vm_ZIB
        
        # Fill Va for ZIB buses in Va_noslack
        for i, zib_bus in enumerate(self.bus_ZIB_all):
            if zib_bus < self.bus_slack:
                Va_noslack[:, zib_bus] = Va_ZIB[:, i]
            elif zib_bus > self.bus_slack:
                Va_noslack[:, zib_bus - 1] = Va_ZIB[:, i]
            # Note: ZIB bus == slack is not possible by definition
        
        return Vm_all, Va_noslack
    
    # =========================================================================
    # Utility methods
    # =========================================================================
    
    def get_info(self) -> Dict[str, Any]:
        """Get converter configuration as dictionary."""
        return {
            'Nbus': self.Nbus,
            'Nbranch': self.Nbranch,
            'Ngen': self.Ngen,
            'baseMVA': self.baseMVA,
            'bus_slack': self.bus_slack,
            'bus_Pd': self.bus_Pd,
            'bus_Qd': self.bus_Qd,
            'bus_Pg': self.bus_Pg,
            'bus_Qg': self.bus_Qg,
            'bus_Pnet_all': self.bus_Pnet_all,
            'bus_ZIB_all': self.bus_ZIB_all,
            'bus_Pnet_noslack_all': self.bus_Pnet_noslack_all,
            'NPred_Va': self.NPred_Va,
            'NPred_Vm': self.NPred_Vm,
            'NZIB': self.NZIB,
            'input_dim_fullbus': self.input_dim_fullbus,
            'input_dim_sparse': self.input_dim_sparse,
            'output_dim_fullbus': self.output_dim_fullbus,
            'output_dim_ngt': self.output_dim_ngt,
            'param_ZIMV': self.param_ZIMV,
            'Ybus': self.Ybus,
        }
    
    def verify_kron_reconstruction(self, y_fullbus: np.ndarray, 
                                    verbose: bool = True) -> Dict[str, float]:
        """
        Verify Kron reconstruction accuracy using actual voltage data.
        
        This function:
        1. Converts y_fullbus -> y_ngt (removing ZIB nodes)
        2. Reconstructs y_fullbus using Kron reduction
        3. Compares original vs reconstructed ZIB voltages
        
        Args:
            y_fullbus: Voltage data in FULL-BUS format [N, 2*Nbus-1]
            verbose: Whether to print results
        
        Returns:
            Dictionary with error statistics:
            - 'Va_zib_max_error': Max Va error for ZIB nodes (radians)
            - 'Vm_zib_max_error': Max Vm error for ZIB nodes (p.u.)
            - 'Va_zib_mean_error': Mean Va error for ZIB nodes
            - 'Vm_zib_mean_error': Mean Vm error for ZIB nodes
        """
        if self.NZIB == 0:
            if verbose:
                print("No ZIB nodes - Kron verification skipped")
            return {'Va_zib_max_error': 0, 'Vm_zib_max_error': 0,
                    'Va_zib_mean_error': 0, 'Vm_zib_mean_error': 0}
        
        if self.param_ZIMV is None:
            raise ValueError("param_ZIMV not computed - cannot verify Kron reconstruction")
        
        y = np.asarray(y_fullbus)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        
        Nbus = self.Nbus
        
        # Extract original Va and Vm
        Va_noslack_orig = y[:, :Nbus - 1]
        Vm_orig = y[:, Nbus - 1:]
        
        # Convert to NGT and back with Kron
        y_ngt = self.y_fullbus_to_ngt(y)
        y_rec = self.y_ngt_to_fullbus(y_ngt, use_kron=True)
        
        Va_noslack_rec = y_rec[:, :Nbus - 1]
        Vm_rec = y_rec[:, Nbus - 1:]
        
        # Compare ZIB nodes
        def bus_to_va_idx(bus_idx):
            return bus_idx if bus_idx < self.bus_slack else bus_idx - 1
        
        zib_va_indices = np.array([bus_to_va_idx(b) for b in self.bus_ZIB_all 
                                   if b != self.bus_slack])
        zib_vm_indices = self.bus_ZIB_all
        
        Va_zib_orig = Va_noslack_orig[:, zib_va_indices]
        Va_zib_rec = Va_noslack_rec[:, zib_va_indices]
        Vm_zib_orig = Vm_orig[:, zib_vm_indices]
        Vm_zib_rec = Vm_rec[:, zib_vm_indices]
        
        Va_error = np.abs(Va_zib_orig - Va_zib_rec)
        Vm_error = np.abs(Vm_zib_orig - Vm_zib_rec)
        
        results = {
            'Va_zib_max_error': float(Va_error.max()),
            'Vm_zib_max_error': float(Vm_error.max()),
            'Va_zib_mean_error': float(Va_error.mean()),
            'Vm_zib_mean_error': float(Vm_error.mean()),
        }
        
        if verbose:
            print(f"\n=== Kron Reconstruction Verification ===")
            print(f"Samples tested: {y.shape[0]}")
            print(f"ZIB nodes: {self.NZIB}")
            print(f"  Va (angle) max error:  {results['Va_zib_max_error']:.6e} rad")
            print(f"  Va (angle) mean error: {results['Va_zib_mean_error']:.6e} rad")
            print(f"  Vm (mag) max error:    {results['Vm_zib_max_error']:.6e} p.u.")
            print(f"  Vm (mag) mean error:   {results['Vm_zib_mean_error']:.6e} p.u.")
        
        return results


# =============================================================================
# Standalone functions (for backward compatibility)
# =============================================================================

def create_converter(case_m_path: str, verbose: bool = True) -> FormatConverter:
    """Create a FormatConverter instance."""
    return FormatConverter(case_m_path, verbose)


# =============================================================================
# Demo and testing
# =============================================================================

def demo():
    """Demonstrate the format converter usage."""
    import torch
    
    print("=" * 70)
    print("FormatConverter Demo")
    print("=" * 70)
    
    # Initialize converter
    case_m = 'main_part/data/case118_ieee_modified.m'
    converter = FormatConverter(case_m, verbose=True)
    
    # Load test data
    print("\n[1] Loading test data...")
    dataset = torch.load('saved_data/multi_preference_solutions/fully_covered_dataset_2026-01-02.pt')
    x_fullbus = dataset['x_train'].numpy()[:10]  # First 10 samples
    y_fullbus = dataset['y_train_pref_lc_0.00'].numpy()[:10]
    
    print(f"  x_fullbus shape: {x_fullbus.shape}")
    print(f"  y_fullbus shape: {y_fullbus.shape}")
    
    # Test x conversion: FULL-BUS -> SPARSE -> FULL-BUS (numpy)
    print("\n[2] Testing x conversion: FULL-BUS -> SPARSE -> FULL-BUS (numpy)")
    x_sparse = converter.x_fullbus_to_sparse(x_fullbus)
    x_reconstructed = converter.x_sparse_to_fullbus(x_sparse)
    
    print(f"  x_fullbus:      {x_fullbus.shape}")
    print(f"  x_sparse:       {x_sparse.shape}")
    print(f"  x_reconstructed: {x_reconstructed.shape}")
    
    # Check reconstruction error
    diff_x = np.abs(x_fullbus - x_reconstructed).max()
    print(f"  Max reconstruction error: {diff_x:.10f}")
    assert diff_x < 1e-10, "x reconstruction failed!"
    print("  [OK] x conversion verified!")
    
    # Test y conversion: FULL-BUS -> NGT
    print("\n[3] Testing y conversion: FULL-BUS -> NGT")
    y_ngt = converter.y_fullbus_to_ngt(y_fullbus)
    
    print(f"  y_fullbus: {y_fullbus.shape}")
    print(f"  y_ngt:     {y_ngt.shape}")
    print("  [OK] y FULL-BUS -> NGT conversion done!")
    
    # Test y conversion: NGT -> FULL-BUS (with Kron reconstruction)
    print("\n[4] Testing y conversion: NGT -> FULL-BUS (with Kron reconstruction)")
    y_reconstructed = converter.y_ngt_to_fullbus(y_ngt, use_kron=True)
    
    print(f"  y_ngt:          {y_ngt.shape}")
    print(f"  y_reconstructed: {y_reconstructed.shape}")
    
    # Check non-ZIB nodes match
    Va_noslack_orig = y_fullbus[:, :converter.Nbus - 1]
    Vm_orig = y_fullbus[:, converter.Nbus - 1:]
    Va_noslack_rec = y_reconstructed[:, :converter.Nbus - 1]
    Vm_rec = y_reconstructed[:, converter.Nbus - 1:]
    
    # Helper to map bus idx to Va_noslack idx
    def bus_to_va_idx(bus_idx, slack):
        return bus_idx if bus_idx < slack else bus_idx - 1
    
    # Compare only non-ZIB nodes (should match exactly)
    va_indices = np.array([bus_to_va_idx(b, converter.bus_slack) for b in converter.bus_Pnet_noslack_all])
    Va_diff = np.abs(Va_noslack_orig[:, va_indices] - Va_noslack_rec[:, va_indices]).max()
    Vm_diff = np.abs(Vm_orig[:, converter.bus_Pnet_all] - Vm_rec[:, converter.bus_Pnet_all]).max()
    
    print(f"  Non-ZIB Va max diff: {Va_diff:.10f}")
    print(f"  Non-ZIB Vm max diff: {Vm_diff:.10f}")
    print("  [OK] y NGT -> FULL-BUS conversion done!")
    
    # Test Kron reconstruction verification with more samples
    print("\n[5] Verifying Kron reconstruction accuracy")
    y_fullbus_all = dataset['y_train_pref_lc_0.00'].numpy()[:100]  # More samples
    kron_results = converter.verify_kron_reconstruction(y_fullbus_all, verbose=True)
    
    # Check if Kron reconstruction is accurate
    if kron_results['Vm_zib_max_error'] < 1e-4 and kron_results['Va_zib_max_error'] < 1e-4:
        print("  [OK] Kron reconstruction is accurate!")
    else:
        print("  [WARNING] Kron reconstruction has significant errors")
    
    # Test PyTorch tensor support
    print("\n[6] Testing PyTorch tensor support")
    x_fullbus_tensor = torch.from_numpy(x_fullbus).float()
    y_fullbus_tensor = torch.from_numpy(y_fullbus).float()
    
    x_sparse_tensor = converter.x_fullbus_to_sparse(x_fullbus_tensor)
    y_ngt_tensor = converter.y_fullbus_to_ngt(y_fullbus_tensor)
    y_rec_tensor = converter.y_ngt_to_fullbus(y_ngt_tensor)
    
    print(f"  Input types:  x={type(x_fullbus_tensor).__name__}, y={type(y_fullbus_tensor).__name__}")
    print(f"  Output types: x_sparse={type(x_sparse_tensor).__name__}, y_ngt={type(y_ngt_tensor).__name__}")
    print(f"  x_sparse: {tuple(x_sparse_tensor.shape)}")
    print(f"  y_ngt:    {tuple(y_ngt_tensor.shape)}")
    print(f"  y_rec:    {tuple(y_rec_tensor.shape)}")
    
    # Verify tensor results match numpy results
    x_sparse_check = np.abs(x_sparse - x_sparse_tensor.numpy()).max()
    y_ngt_check = np.abs(y_ngt - y_ngt_tensor.numpy()).max()
    print(f"  Tensor vs numpy diff: x_sparse={x_sparse_check:.10f}, y_ngt={y_ngt_check:.10f}")
    assert x_sparse_check < 1e-6 and y_ngt_check < 1e-6, "Tensor conversion mismatch!"
    print("  [OK] PyTorch tensor support verified!")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    
    return converter


if __name__ == "__main__":
    demo()

