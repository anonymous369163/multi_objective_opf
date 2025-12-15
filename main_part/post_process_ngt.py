#!/usr/bin/env python
# coding: utf-8
"""
DeepOPF-NGT Post-Processing Functions

This module implements the NGT-specific post-processing functions,
including ZIB voltage recovery and the main post-processing pipeline.

Common functions (get_vioPQg, get_viobran2, dPQbus_dV, etc.) are 
imported from utils.py to avoid code duplication.

Reference: main_DeepOPFNGT_M3.ipynb

Author: Ported from main_DeepOPFNGT_M3.ipynb
Date: 2024
"""

import numpy as np

# Import common post-processing functions from utils.py
from utils import (
    get_vioPQg,
    get_viobran2,
    dPQbus_dV,
    dSlbus_dV,
    get_hisdV,
    get_dV,
    get_PQ,
    get_genload,
    get_Pgcost
)


# ============================================================
# NGT-Specific Utility Functions
# ============================================================

def cart2pol(x, y):
    """Convert Cartesian to polar coordinates."""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    """Convert polar to Cartesian coordinates."""
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


# ============================================================
# ZIB Voltage Recovery (NGT-Specific)
# ============================================================

def recover_zib_voltage(Vm_pred, Va_pred, param_ZIMV, bus_Pnet_all, bus_ZIB_all, Nbus):
    """
    Recover ZIB (Zero Injection Bus) voltages using Kron Reduction.
    
    This is specific to the DeepOPF-NGT method where only non-ZIB bus voltages
    are predicted, and ZIB voltages are recovered using a pre-computed 
    transformation matrix.
    
    Args:
        Vm_pred: Predicted Vm for non-ZIB buses [Nsam, NPred_Vm]
        Va_pred: Predicted Va for non-ZIB buses [Nsam, NPred_Vm]
        param_ZIMV: ZIB recovery matrix from Kron Reduction
        bus_Pnet_all: Non-ZIB bus indices
        bus_ZIB_all: ZIB bus indices
        Nbus: Total number of buses
        
    Returns:
        Vm_full, Va_full: Full voltage vectors [Nsam, Nbus]
    """
    Nsam = Vm_pred.shape[0]
    
    # Convert to complex voltage
    Vx = Vm_pred * np.exp(1j * Va_pred)
    
    # Recover ZIB voltages using Kron Reduction transformation
    Vy = np.dot(param_ZIMV, Vx.T).T
    
    # Build full voltage in rectangular coordinates
    Ve = np.zeros((Nsam, Nbus))
    Vf = np.zeros((Nsam, Nbus))
    
    Ve[:, bus_Pnet_all] = Vx.real
    Vf[:, bus_Pnet_all] = Vx.imag
    Ve[:, bus_ZIB_all] = Vy.real
    Vf[:, bus_ZIB_all] = Vy.imag
    
    # Convert back to polar coordinates
    Vm_full, Va_full = cart2pol(Ve, Vf)
    
    return Vm_full, Va_full


# ============================================================
# Main NGT Post-Processing Function
# ============================================================

def post_process_ngt(yvtest_hat, sys_data, config, k_dV=0.1, DELTA=1e-4):
    """
    Complete post-processing for DeepOPF-NGT predictions.
    
    This function:
    1. Splits the model output into Va and Vm
    2. Inserts slack bus voltage (Va=0)
    3. Recovers ZIB bus voltages using Kron Reduction
    4. Calculates generation, load, and constraint violations
    
    Args:
        yvtest_hat: Model output [Ntest, NPred_Va + NPred_Vm]
        sys_data: PowerSystemData object with grid data
        config: Configuration object with parameters
        k_dV: Voltage correction coefficient (default: 0.1)
        DELTA: Violation detection threshold (default: 1e-4)
        
    Returns:
        results: Dictionary with evaluation metrics including:
            - Vm_full, Va_full: Full voltage vectors
            - Pred_Pg, Pred_Qg: Generator outputs
            - vio_PQg: Constraint satisfaction percentages
            - lsPg, lsQg: Violation details
    """
    Ntest = yvtest_hat.shape[0]
    NPred_Va = sys_data.NPred_Va
    NPred_Vm = sys_data.NPred_Vm
    Nbus = config.Nbus
    
    # Split Va and Vm from combined output
    yvatest_hat = yvtest_hat[:, :NPred_Va]
    yvmtest_hat = yvtest_hat[:, NPred_Va:]
    
    # Insert slack bus Va = 0 (slack bus has Va reference of 0)
    idx_bus_Pnet_slack = sys_data.idx_bus_Pnet_slack
    xam_P = np.insert(yvtest_hat, idx_bus_Pnet_slack[0], 0, axis=1)
    
    Va_nonZIB = xam_P[:, :NPred_Vm]
    Vm_nonZIB = xam_P[:, NPred_Vm:]
    
    # Recover ZIB bus voltages using Kron Reduction
    if sys_data.NZIB > 0:
        Vm_full, Va_full = recover_zib_voltage(
            Vm_nonZIB, Va_nonZIB,
            sys_data.param_ZIMV,
            sys_data.bus_Pnet_all,
            sys_data.bus_ZIB_all,
            Nbus
        )
        
        # Clamp ZIB voltages to physical limits
        VmLb = config.ngt_VmLb
        VmUb = config.ngt_VmUb
        Vm_full[:, sys_data.bus_ZIB_all] = np.clip(
            Vm_full[:, sys_data.bus_ZIB_all], VmLb, VmUb
        )
    else:
        # No ZIB buses - direct assignment
        Vm_full = np.zeros((Ntest, Nbus))
        Va_full = np.zeros((Ntest, Nbus))
        Vm_full[:, sys_data.bus_Pnet_all] = Vm_nonZIB
        Va_full[:, sys_data.bus_Pnet_all] = Va_nonZIB
        Va_full[:, sys_data.bus_slack] = 0
    
    # Convert to complex voltage for power flow calculation
    Pred_V = Vm_full * np.exp(1j * Va_full)
    
    # Calculate generation and load
    Pred_Pg, Pred_Qg, Pred_Pd, Pred_Qd = get_genload(
        Pred_V, sys_data.Pdtest, sys_data.Qdtest,
        sys_data.bus_Pg, sys_data.bus_Qg, sys_data.Ybus
    )
    
    # Calculate constraint violations
    # Note: get_vioPQg returns torch tensors, convert if needed
    lsPg, lsQg, lsidxPg, lsidxQg, vio_PQgmaxmin, vio_PQg, \
        deltaPgL, deltaPgU, deltaQgL, deltaQgU = get_vioPQg(
            Pred_Pg, sys_data.bus_Pg, sys_data.MAXMIN_Pg,
            Pred_Qg, sys_data.bus_Qg, sys_data.MAXMIN_Qg,
            DELTA
        )
    
    results = {
        'Vm_full': Vm_full,
        'Va_full': Va_full,
        'Pred_V': Pred_V,
        'Pred_Pg': Pred_Pg,
        'Pred_Qg': Pred_Qg,
        'Pred_Pd': Pred_Pd,
        'Pred_Qd': Pred_Qd,
        'vio_PQg': vio_PQg,
        'vio_PQgmaxmin': vio_PQgmaxmin,
        'lsPg': lsPg,
        'lsQg': lsQg,
        'lsidxPg': lsidxPg,
        'lsidxQg': lsidxQg,
        'deltaPgL': deltaPgL,
        'deltaPgU': deltaPgU,
        'deltaQgL': deltaQgL,
        'deltaQgU': deltaQgU,
    }
    
    return results


if __name__ == "__main__":
    print("DeepOPF-NGT Post-Processing Module")
    print("=" * 60)
    print("\nNGT-Specific Functions:")
    print("  - recover_zib_voltage(): Recover ZIB voltages using Kron Reduction")
    print("  - post_process_ngt(): Complete NGT post-processing pipeline")
    print("  - cart2pol/pol2cart: Coordinate conversion utilities")
    print("\nImported from utils.py:")
    print("  - get_vioPQg(): Calculate Pg/Qg constraint violations")
    print("  - get_viobran2(): Calculate branch power flow violations")
    print("  - dPQbus_dV(): Compute power injection Jacobian")
    print("  - dSlbus_dV(): Compute branch power flow Jacobian")
    print("  - get_hisdV(): Calculate voltage correction (historical Jacobian)")
    print("  - get_dV(): Calculate voltage correction (predicted Jacobian)")
    print("  - get_PQ(): Calculate bus power injections")
    print("  - get_genload(): Calculate generator and load outputs")
    print("  - get_Pgcost(): Calculate generation cost")
