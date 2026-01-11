#!/usr/bin/env python
# coding: utf-8
"""
Utility Functions for DeepOPF-V
Author: Peng Yue
Date: December 2025

This module provides:
- Evaluation metrics (MAE, relative error, clamp)
- Power system calculations (power flow, generation cost, carbon emission)
- Constraint violation checking (generator limits, branch limits)
- Post-processing functions (Jacobian-based correction)
- Pareto front analysis (hypervolume computation)
- TensorBoard logging utilities
"""

import numpy as np
import torch
import math
import matplotlib.pyplot as plt


# ==================== Pareto Front Evaluation ====================

def compute_hypervolume(points, ref_point):
    """
    Compute hypervolume indicator for Pareto front evaluation.
    
    The hypervolume is the area dominated by the Pareto front
    and bounded by a reference point. Larger hypervolume = better Pareto front.
    
    Args:
        points: np.ndarray [N, 2] where each row is (cost, carbon) (minimization objectives)
        ref_point: np.ndarray [2] reference point (nadir point)
                   
    Returns:
        hv: float, hypervolume value (0 if no valid points)
    """
    if len(points) == 0:
        return 0.0
    
    points = np.asarray(points)
    ref_point = np.asarray(ref_point)
    
    # Filter points dominated by reference point
    valid_mask = np.all(points < ref_point, axis=1)
    valid_points = points[valid_mask]
    
    if len(valid_points) == 0:
        return 0.0
    
    try:
        from pymoo.indicators.hv import HV
        hv_indicator = HV(ref_point=ref_point)
        return float(hv_indicator(valid_points))
    except ImportError:
        # Fallback: 2D sweep line algorithm
        sorted_indices = np.argsort(valid_points[:, 0])
        sorted_points = valid_points[sorted_indices]
        
        hv = 0.0
        prev_y = ref_point[1]
        
        for point in sorted_points:
            x, y = point
            if y < prev_y:
                hv += (ref_point[0] - x) * (prev_y - y)
                prev_y = y
        
        return hv


# ==================== Evaluation Metrics ====================

def get_clamp(Pred, Predmin, Predmax):
    """Clamp predicted values within min/max bounds."""
    Pred_clip = Pred.clone()
    for i in range(Pred.shape[1]):
        Pred_clip[:, i] = Pred_clip[:, i].clamp(min=Predmin[i], max=Predmax[i])
    return Pred_clip


def get_mae(real, predict):
    """Calculate Mean Absolute Error."""
    if len(real) == len(predict):
        return torch.mean(torch.abs(real - predict))
    return None


def get_rerr(real, predict):
    """Calculate relative error (absolute value) in percentage."""
    if len(real) == len(predict):
        return torch.abs((predict - real) / real) * 100
    return None


def get_rerr2(real, predict):
    """Calculate relative error (signed) in percentage."""
    if len(real) == len(predict):
        return (predict - real) / real * 100
    return None


# ==================== Power System Calculations ====================

def get_PQ(V, Ybus):
    """Calculate active and reactive power at each bus."""
    S = np.zeros(V.shape, dtype=np.complex128)
    for i in range(V.shape[0]):
        I = Ybus.dot(V[i]).conj()
        S[i] = np.multiply(V[i], I)
    return np.real(S), np.imag(S)


def get_genload(V, Pdtest, Qdtest, bus_Pg, bus_Qg, Ybus):
    """Calculate generation and load at each bus."""
    S = np.zeros(V.shape, dtype=np.complex128)
    for i in range(V.shape[0]):
        I = Ybus.dot(V[i]).conj()
        S[i] = np.multiply(V[i], I)
    
    P, Q = np.real(S), np.imag(S)
    
    Pg = P[:, bus_Pg] + Pdtest[:, bus_Pg]
    Qg = Q[:, bus_Qg] + Qdtest[:, bus_Qg]
    Pd = -P.copy()
    Qd = -Q.copy()
    Pd[:, bus_Pg] = Pg - P[:, bus_Pg]
    Qd[:, bus_Qg] = Qg - Q[:, bus_Qg]
    
    return Pg, Qg, Pd, Qd


def get_Pgcost(Pg, idxPg, gencost, baseMVA):
    """
    Calculate generation cost.
    
    Supports both MATPOWER format (7 columns) and simplified format (2 columns).
    """
    # Determine column indices based on gencost format
    if gencost.shape[1] > 4:
        col_c2, col_c1 = 4, 5  # MATPOWER format
    else:
        col_c2, col_c1 = 0, 1  # Simplified format
    
    cost = np.zeros(Pg.shape[0])
    PgMVA = Pg * baseMVA
    
    for i in range(Pg.shape[0]):
        c2_term = gencost[idxPg, col_c2] * PgMVA[i, :] ** 2
        c1_term = gencost[idxPg, col_c1] * PgMVA[i, :]
        cost[i] = np.sum(c2_term + c1_term)
    
    return cost


# ==================== Carbon Emission Calculations ====================

def get_carbon_emission_vectorized(Pg, gci_values, baseMVA):
    """
    Vectorized carbon emission calculation.
    
    Carbon = Σ GCI_i × Pg_i (tCO2/h)
    """
    return np.dot(Pg * baseMVA, gci_values)


def get_gci_for_generators(sys_data):
    """
    Assign GCI (Carbon Emission Intensity) based on marginal generation cost.
    
    Low-cost generators → High carbon (coal)
    High-cost generators → Low carbon (CCGT)
    """
    FUEL_LOOKUP_CO2 = {
        "ANT": 0.9095, "BIT": 0.8204, "Oil": 0.7001, "GAS": 0.5173,
        "CCGT": 0.3621, "ICE": 0.6030, "Thermal": 0.6874,
        "NUC": 0.0, "RE": 0.0, "HYD": 0.0, "N/A": 0.0
    }
    
    n_gen = sys_data.gen.shape[0] if isinstance(sys_data.gen, np.ndarray) else sys_data.gen.numpy().shape[0]
    gencost = sys_data.gencost if isinstance(sys_data.gencost, np.ndarray) else sys_data.gencost.numpy()
    
    gci_values = np.zeros(n_gen)
    c1_values = gencost[:n_gen, 1]
    
    p25, p50, p75 = np.percentile(c1_values, [25, 50, 75])
    
    for i in range(n_gen):
        c1 = c1_values[i]
        if c1 <= p25:
            fuel_type = "BIT" if i % 2 == 0 else "ANT"
        elif c1 <= p50:
            fuel_type = "Oil"
        elif c1 <= p75:
            fuel_type = "GAS"
        else:
            fuel_type = "CCGT"
        gci_values[i] = FUEL_LOOKUP_CO2[fuel_type]
    
    return gci_values


# ==================== Constraint Violation Checking ====================

def get_vioPQg(Pred_Pg, bus_Pg, MAXMIN_Pg, Pred_Qg, bus_Qg, MAXMIN_Qg, DELTA):
    """Check Pg and Qg constraint violations."""
    n_samples = Pred_Pg.shape[0]
    vio_PQgmaxminnum = torch.zeros((n_samples, 4))
    vio_PQg = torch.zeros((n_samples, 2))
    lsPg, lsQg = [], []
    lsidxPg = np.zeros(n_samples, dtype=int)
    lsidxQg = np.zeros(n_samples, dtype=int)
    kP, kQ = 1, 1
    deltaPgL = np.array([[0, 0]])
    deltaPgU = np.array([[0, 0]])
    deltaQgL = np.array([[0, 0]])
    deltaQgU = np.array([[0, 0]])
    
    for i in range(n_samples):
        # Active power violations
        delta = Pred_Pg[i] - MAXMIN_Pg[:, 0]
        idxPgUB = np.array(np.where(delta > DELTA))
        if np.size(idxPgUB) > 0:
            PgUB = np.concatenate((idxPgUB, delta[idxPgUB]), axis=0).T
            deltaPgU = np.append(deltaPgU, PgUB, axis=0)

        delta = Pred_Pg[i] - MAXMIN_Pg[:, 1]
        idxPgLB = np.array(np.where(delta < -DELTA))
        if np.size(idxPgLB) > 0:
            PgLB = np.concatenate((idxPgLB, delta[idxPgLB]), axis=0).T
            deltaPgL = np.append(deltaPgL, PgLB, axis=0)
        
        if np.size(idxPgUB) > 0 and np.size(idxPgLB) > 0:
            PgLUB = np.concatenate((PgUB, PgLB), axis=0)
        elif np.size(idxPgUB) > 0:
            PgLUB = PgUB
        elif np.size(idxPgLB) > 0:
            PgLUB = PgLB
        
        if (np.size(idxPgUB) + np.size(idxPgLB)) > 0:
            PgLUB = PgLUB[PgLUB[:, 0].argsort()]
            lsPg.append(PgLUB)
            lsidxPg[i] = kP
            kP += 1

        # Reactive power violations
        delta = Pred_Qg[i] - MAXMIN_Qg[:, 0]
        idxQgUB = np.array(np.where(delta > DELTA))
        if np.size(idxQgUB) > 0:
            QgUB = np.concatenate((idxQgUB, delta[idxQgUB]), axis=0).T
            deltaQgU = np.append(deltaQgU, QgUB, axis=0)

        delta = Pred_Qg[i] - MAXMIN_Qg[:, 1]
        idxQgLB = np.array(np.where(delta < -DELTA))
        if np.size(idxQgLB) > 0:
            QgLB = np.concatenate((idxQgLB, delta[idxQgLB]), axis=0).T
            deltaQgL = np.append(deltaQgL, QgLB, axis=0)
            
        if np.size(idxQgUB) > 0 and np.size(idxQgLB) > 0:
            QgLUB = np.concatenate((QgUB, QgLB), axis=0)
        elif np.size(idxQgUB) > 0:
            QgLUB = QgUB
        elif np.size(idxQgLB) > 0:
            QgLUB = QgLB
         
        if (np.size(idxQgUB) + np.size(idxQgLB)) > 0:
            QgLUB = QgLUB[QgLUB[:, 0].argsort()]
            lsQg.append(QgLUB)
            lsidxQg[i] = kQ
            kQ += 1
                    
        vio_PQgmaxminnum[i, 0] = np.size(idxPgUB)
        vio_PQgmaxminnum[i, 1] = np.size(idxPgLB)
        vio_PQgmaxminnum[i, 2] = np.size(idxQgUB)
        vio_PQgmaxminnum[i, 3] = np.size(idxQgLB)
        
    # Calculate violation ratios
    vio_PQgmaxmin = torch.zeros((n_samples, 4))
    vio_PQgmaxmin[:, 0] = (1 - vio_PQgmaxminnum[:, 0] / bus_Pg.shape[0]) * 100
    vio_PQgmaxmin[:, 1] = (1 - vio_PQgmaxminnum[:, 1] / bus_Pg.shape[0]) * 100
    vio_PQgmaxmin[:, 2] = (1 - vio_PQgmaxminnum[:, 2] / bus_Qg.shape[0]) * 100
    vio_PQgmaxmin[:, 3] = (1 - vio_PQgmaxminnum[:, 3] / bus_Qg.shape[0]) * 100
    vio_PQg[:, 0] = (1 - (vio_PQgmaxminnum[:, 0] + vio_PQgmaxminnum[:, 1]) / bus_Pg.shape[0]) * 100
    vio_PQg[:, 1] = (1 - (vio_PQgmaxminnum[:, 2] + vio_PQgmaxminnum[:, 3]) / bus_Qg.shape[0]) * 100
     
    # Clean up initial dummy rows
    for arr in [deltaPgL, deltaPgU, deltaQgL, deltaQgU]:
        if arr.shape[0] > 1:
            arr = np.delete(arr, 0, axis=0)
    
    return lsPg, lsQg, lsidxPg, lsidxQg, vio_PQgmaxmin, vio_PQg, deltaPgL, deltaPgU, deltaQgL, deltaQgU


def get_viobran(Pred_V, Pred_Va, branch, Yf, Yt, BRANFT, baseMVA, DELTA):
    """Check branch constraint violations."""
    baseMVA_scalar = float(np.asarray(baseMVA).ravel()[0])
    branlp = np.asarray(branch[:, 2]).ravel() / baseMVA_scalar
    angminmax = branch[:, 3:5] * math.pi / 180
    Pred_branang = Pred_Va[:, BRANFT[:, 0]] - Pred_Va[:, BRANFT[:, 1]]
    
    n_samples = Pred_V.shape[0]
    vio_branangnum = torch.zeros(n_samples)
    vio_branpfnum = torch.zeros(n_samples)
    deltapf_list = []
    
    for i in range(n_samples):
        vio_branangnum[i] = np.size(np.where(Pred_branang[i, :] - angminmax[:, 0] < -DELTA)) \
                          + np.size(np.where(Pred_branang[i, :] - angminmax[:, 1] > DELTA))

        fV = Pred_V[i, BRANFT[:, 0]]
        tV = Pred_V[i, BRANFT[:, 1]]
        fI = Yf.dot(Pred_V[i]).conj()
        tI = Yt.dot(Pred_V[i]).conj()
        fS = np.multiply(fV, fI)
        tS = np.multiply(tV, tI)
        deltafS = np.array(np.abs(fS) - branlp).ravel()
        deltatS = np.array(np.abs(tS) - branlp).ravel()
        
        idxfs = np.array(np.where(deltafS > DELTA))
        idxts = np.array(np.where(deltatS > DELTA))
        vio_branpfnum[i] = np.size(idxfs) + np.size(idxts)
        
        if np.size(idxfs) >= 1:
            deltapf_list.append(np.concatenate((idxfs, deltafS[idxfs]), axis=0).T)
        if np.size(idxts) >= 1:
            deltapf_list.append(np.concatenate((idxts, deltatS[idxts]), axis=0).T)
    
    if deltapf_list:
        deltapf = np.vstack(deltapf_list)
        branch_indices = np.asarray(deltapf[:, 0]).ravel().astype(int)
        valid_mask = (branch_indices >= 0) & (branch_indices < len(branlp))
        if np.any(valid_mask):
            deltapfR = np.zeros(len(branch_indices))
            deltapfR[valid_mask] = deltapf[valid_mask, 1] / branlp[branch_indices[valid_mask]] * 100
            deltapf = np.insert(deltapf, 2, values=deltapfR, axis=1)
        else:
            deltapf = np.insert(deltapf, 2, values=np.zeros(len(branch_indices)), axis=1)
    else:
        deltapf = np.array([[0, 0, 0]])
 
    vio_branang = (1 - vio_branangnum / branch.shape[0]) * 100
    vio_branpf = (1 - vio_branpfnum / (branch.shape[0] * 2)) * 100
    return vio_branang, vio_branpf, deltapf


def get_viobran2(Pred_V, Pred_Va, branch, Yf, Yt, BRANFT, baseMVA, DELTA):
    """Check branch violations with detailed information for post-processing."""
    baseMVA_scalar = float(np.asarray(baseMVA).ravel()[0])
    branlp = np.asarray(branch[:, 2]).ravel() / baseMVA_scalar
    angminmax = branch[:, 3:5] * math.pi / 180
    Pred_branang = Pred_Va[:, BRANFT[:, 0]] - Pred_Va[:, BRANFT[:, 1]]
    
    n_samples = Pred_V.shape[0]
    vio_branangnum = torch.zeros(n_samples)
    vio_branpfnum = torch.zeros(n_samples)
    vio_branpfidx = torch.zeros(n_samples)
    lsSf, lsSt = [], []
    lsSf_sampidx, lsSt_sampidx = [], []
    deltapf_list = []
    
    for i in range(n_samples):
        vio_branangnum[i] = np.size(np.where(Pred_branang[i, :] - angminmax[:, 0] < -DELTA)) \
                          + np.size(np.where(Pred_branang[i, :] - angminmax[:, 1] > DELTA))

        fV = Pred_V[i, BRANFT[:, 0]]
        tV = Pred_V[i, BRANFT[:, 1]]
        fI = Yf.dot(Pred_V[i]).conj()
        tI = Yt.dot(Pred_V[i]).conj()
        fS = np.multiply(fV, fI)
        tS = np.multiply(tV, tI)
        deltafS = np.array(np.abs(fS) - branlp).ravel()
        deltatS = np.array(np.abs(tS) - branlp).ravel()
        
        idxfs = np.array(np.where(deltafS > DELTA)).reshape(-1, 1)
        idxts = np.array(np.where(deltatS > DELTA)).reshape(-1, 1)
        vio_branpfnum[i] = np.size(idxfs) + np.size(idxts)
        
        if np.size(idxfs) >= 1:
            ii = np.concatenate((idxfs, deltafS[idxfs]), axis=1)
            deltapf_list.append(ii)
            ii = np.concatenate((ii, np.real(fS[idxfs]), np.imag(fS[idxfs])), axis=1)
            lsSf.append(ii)
            lsSf_sampidx.append(i)
            
        if np.size(idxts) >= 1:
            ii = np.concatenate((idxts, deltatS[idxts]), axis=1)
            deltapf_list.append(ii)
            ii = np.concatenate((ii, np.real(tS[idxts]), np.imag(tS[idxts])), axis=1)
            lsSt.append(ii)
            lsSt_sampidx.append(i)

        if np.size(idxfs) + np.size(idxts) >= 1:
            vio_branpfidx[i] = i + 1
    
    if deltapf_list:
        deltapf = np.vstack(deltapf_list)
        branch_indices = np.asarray(deltapf[:, 0]).ravel().astype(int)
        valid_mask = (branch_indices >= 0) & (branch_indices < len(branlp))
        if np.any(valid_mask):
            deltapfR = np.zeros(len(branch_indices))
            deltapfR[valid_mask] = deltapf[valid_mask, 1] / branlp[branch_indices[valid_mask]] * 100
            deltapf = np.insert(deltapf, 2, values=deltapfR, axis=1)
        else:
            deltapf = np.insert(deltapf, 2, values=np.zeros(len(branch_indices)), axis=1)
    else:
        deltapf = np.array([[0, 0, 0]])
 
    vio_branang = (1 - vio_branangnum / branch.shape[0]) * 100
    vio_branpf = (1 - vio_branpfnum / (branch.shape[0] * 2)) * 100
    return vio_branang, vio_branpf, deltapf, vio_branpfidx, lsSf, lsSt, lsSf_sampidx, lsSt_sampidx


# ==================== Post-Processing Functions ====================

def dPQbus_dV(his_V, bus_Pg, bus_Qg, Ybus):
    """Calculate Jacobian matrix dP/dV and dQ/dV at buses."""
    V = his_V.copy()
    Ibus = Ybus.dot(his_V).conj()
    diagV = np.diag(V)
    diagIbus = np.diag(Ibus)
    diagVnorm = np.diag(V / np.abs(V))
    
    dSbus_dVm = np.dot(diagV, Ybus.dot(diagVnorm).conj()) + np.dot(diagIbus.conj(), diagVnorm)
    dSbus_dVa = 1j * np.dot(diagV, (diagIbus - Ybus.dot(diagV)).conj())
    dSbus_dV = np.concatenate((dSbus_dVa, dSbus_dVm), axis=1)
    
    return np.real(dSbus_dV), np.imag(dSbus_dV)


def get_hisdV(lsPg, lsQg, lsidxPg, lsidxQg, num_viotest, k_dV, bus_Pg, bus_Qg, dPbus_dV, dQbus_dV, Nbus, Ntest):
    """Calculate voltage correction using historical voltage Jacobian."""
    dV = np.zeros((num_viotest, Nbus * 2))
    j = 0
    for i in range(Ntest):
        if (lsidxPg[i] + lsidxQg[i]) > 0:
            if lsidxPg[i] > 0 and lsidxQg[i] > 0:
                idxPg = lsPg[lsidxPg[i] - 1][:, 0].astype(np.int32)
                idxQg = lsQg[lsidxQg[i] - 1][:, 0].astype(np.int32)
                dPQGbus_dV = np.concatenate((dPbus_dV[bus_Pg[idxPg], :], dQbus_dV[bus_Qg[idxQg], :]), axis=0)
                dPQg = np.concatenate((lsPg[lsidxPg[i] - 1][:, 1], lsQg[lsidxQg[i] - 1][:, 1]), axis=0)
            elif lsidxPg[i] > 0:
                idxPg = lsPg[lsidxPg[i] - 1][:, 0].astype(np.int32)
                dPQGbus_dV = dPbus_dV[bus_Pg[idxPg], :]
                dPQg = lsPg[lsidxPg[i] - 1][:, 1]
            elif lsidxQg[i] > 0:
                idxQg = lsQg[lsidxQg[i] - 1][:, 0].astype(np.int32)
                dPQGbus_dV = dQbus_dV[bus_Qg[idxQg], :]
                dPQg = lsQg[lsidxQg[i] - 1][:, 1]

            dV[j] = np.dot(np.linalg.pinv(dPQGbus_dV), dPQg * k_dV)
            j += 1
            
    return dV


def get_dV(Pred_V, lsPg, lsQg, lsidxPg, lsidxQg, num_viotest, k_dV, bus_Pg, bus_Qg, Ybus, his_V):
    """Calculate voltage correction using predicted voltage Jacobian."""
    dV = np.zeros((num_viotest, Pred_V.shape[1] * 2))
    j = 0
    for i in range(Pred_V.shape[0]):
        if (lsidxPg[i] + lsidxQg[i]) > 0:
            V = Pred_V[i].copy()
            Ibus = Ybus.dot(V).conj()
            diagV = np.diag(V)
            diagIbus = np.diag(Ibus)
            diagVnorm = np.diag(V / np.abs(V))

            dSbus_dVm = np.dot(diagV, Ybus.dot(diagVnorm).conj()) + np.dot(diagIbus.conj(), diagVnorm)
            dSbus_dVa = 1j * np.dot(diagV, (diagIbus - Ybus.dot(diagV)).conj())
            dSbus_dV = np.concatenate((dSbus_dVa, dSbus_dVm), axis=1)
            dPbus_dV, dQbus_dV = np.real(dSbus_dV), np.imag(dSbus_dV)
            
            if lsidxPg[i] > 0 and lsidxQg[i] > 0:
                idxPg = lsPg[lsidxPg[i] - 1][:, 0].astype(np.int32)
                idxQg = lsQg[lsidxQg[i] - 1][:, 0].astype(np.int32)
                dPQGbus_dV = np.concatenate((dPbus_dV[bus_Pg[idxPg], :], dQbus_dV[bus_Qg[idxQg], :]), axis=0)
                dPQg = np.concatenate((lsPg[lsidxPg[i] - 1][:, 1], lsQg[lsidxQg[i] - 1][:, 1]), axis=0)
            elif lsidxPg[i] > 0:
                idxPg = lsPg[lsidxPg[i] - 1][:, 0].astype(np.int32)
                dPQGbus_dV = dPbus_dV[bus_Pg[idxPg], :]
                dPQg = lsPg[lsidxPg[i] - 1][:, 1]
            elif lsidxQg[i] > 0:
                idxQg = lsQg[lsidxQg[i] - 1][:, 0].astype(np.int32)
                dPQGbus_dV = dQbus_dV[bus_Qg[idxQg], :]
                dPQg = lsQg[lsidxQg[i] - 1][:, 1]

            dV[j] = np.dot(np.linalg.pinv(dPQGbus_dV), dPQg * k_dV)
            j += 1
    return dV


def dSlbus_dV(his_V, bus_Va, branch, Yf, finc, BRANFT, Nbus):
    """Calculate derivative of branch power flow with respect to voltage."""
    V = his_V.copy()
    fV = V[BRANFT[:, 0]]
    fI = Yf.dot(V).conj()
    
    diagfI = np.diag(fI)
    diagfV = np.diag(fV)
    diagVnorm = np.diag(np.true_divide(V, np.abs(V)))
    
    dfS_dVm = np.dot(diagfV, Yf.dot(diagVnorm).conj()) + np.dot(diagfI.conj(), np.dot(finc, diagVnorm))
    dfP_dVm = np.real(dfS_dVm)
    dfQ_dVm = np.imag(dfS_dVm)
    
    diagV = np.diag(V)
    dfS_dVa = -1j * np.dot(diagfV, Yf.dot(diagV).conj()) + 1j * np.dot(diagfI.conj(), np.dot(finc, diagV))
    dfP_dVa = np.real(dfS_dVa)
    dfQ_dVa = np.imag(dfS_dVa)
    
    dPfbus_dV = np.concatenate((dfP_dVa, dfP_dVm), axis=1)
    dQfbus_dV = np.concatenate((dfQ_dVa, dfQ_dVm), axis=1)

    return dPfbus_dV, dQfbus_dV


# ==================== TensorBoard Logging ====================

class TensorBoardLogger:
    """Simplified TensorBoard logger for training monitoring."""
    
    def __init__(self, log_dir='runs/pareto_flow', comment=''):
        self.log_file = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            import datetime
            import os
            
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            run_name = f"{log_dir}/{timestamp}"
            if comment:
                run_name += f"_{comment}"
            
            self.writer = SummaryWriter(run_name)
            self.enabled = True
            print(f"[TensorBoard] Logging to: {run_name}")
        except ImportError:
            self.writer = None
            self.enabled = False
            print("[TensorBoard] tensorboard not available, logging disabled")
    
    def log_scalar(self, tag, value, step):
        if self.enabled and self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag, tag_scalar_dict, step):
        if self.enabled and self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_losses(self, step, loss_dict, prefix='train'):
        if not self.enabled:
            return
        for key, value in loss_dict.items():
            if isinstance(value, (int, float)):
                self.log_scalar(f'{prefix}/loss_{key}', value, step)
            elif torch.is_tensor(value):
                self.log_scalar(f'{prefix}/loss_{key}', value.item(), step)
    
    def flush(self):
        if self.enabled and self.writer:
            self.writer.flush()
    
    def close(self):
        if self.enabled and self.writer:
            self.writer.close()


def plot_unsupervised_training_curves(loss_history):
    """Plot unsupervised training loss curves."""
    has_ngt_keys = 'kgenp_mean' in loss_history
    
    if has_ngt_keys:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        axes[0, 0].plot(loss_history.get('total', []))
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(loss_history.get('kgenp_mean', []))
        axes[0, 1].set_title('Generator P Weight')
        axes[0, 1].grid(True)
        
        axes[0, 2].plot(loss_history.get('kgenq_mean', []))
        axes[0, 2].set_title('Generator Q Weight')
        axes[0, 2].grid(True)
        
        axes[1, 0].plot(loss_history.get('kpd_mean', []))
        axes[1, 0].set_title('Load P Weight')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(loss_history.get('kqd_mean', []))
        axes[1, 1].set_title('Load Q Weight')
        axes[1, 1].grid(True)
        
        axes[1, 2].plot(loss_history.get('kv_mean', []))
        axes[1, 2].set_title('Voltage Weight')
        axes[1, 2].grid(True)
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        axes[0, 0].plot(loss_history.get('total', []))
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(loss_history.get('cost', []))
        axes[0, 1].set_title('Generation Cost')
        axes[0, 1].grid(True)
        
        axes[0, 2].plot(loss_history.get('gen_vio', []))
        axes[0, 2].set_title('Generator Violation')
        axes[0, 2].grid(True)
        
        axes[1, 0].plot(loss_history.get('branch_pf_vio', []))
        axes[1, 0].set_title('Branch Power Violation')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(loss_history.get('branch_ang_vio', []))
        axes[1, 1].set_title('Branch Angle Violation')
        axes[1, 1].grid(True)
        
        axes[1, 2].plot(loss_history.get('load_dev', []))
        axes[1, 2].set_title('Load Deviation')
        axes[1, 2].grid(True)
    
    for ax in axes.flat:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig('unsupervised_training_curves.png', dpi=300, bbox_inches='tight')
    print('\nTraining curves saved to: unsupervised_training_curves.png')
    plt.close()
