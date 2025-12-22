#!/usr/bin/env python
# coding: utf-8
# Utility Functions for DeepOPF-V
# Author: Peng Yue
# Date: December 19th, 2025

import numpy as np
import torch
import torch.nn as nn
import math
import time

import matplotlib.pyplot as plt 
# ==================== GPU Memory and Thermal Protection ====================

def gpu_memory_cleanup():
    """Force GPU memory cleanup."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def check_gpu_temperature(warning_temp=80, critical_temp=85, cooldown_time=30):
    """
    Check GPU temperature and pause if too hot.
    
    Args:
        warning_temp: Temperature to show warning (default: 80°C)
        critical_temp: Temperature to pause training (default: 85°C)
        cooldown_time: Seconds to wait when critical (default: 30s)
        
    Returns:
        temp: Current GPU temperature, or None if unavailable
    """
    if not torch.cuda.is_available():
        return None
        
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            temp = int(result.stdout.strip().split('\n')[0])
            if temp >= critical_temp:
                print(f"\n[THERMAL] GPU {temp}°C >= {critical_temp}°C - Cooling down for {cooldown_time}s...")
                gpu_memory_cleanup()
                time.sleep(cooldown_time)
                return temp
            elif temp >= warning_temp:
                print(f" [GPU:{temp}°C]", end='', flush=True)
            return temp
    except Exception:
        pass
    return None


def add_training_delay(delay_ms=0):
    """
    Add small delay between batches to reduce GPU load.
    
    Args:
        delay_ms: Delay in milliseconds (0 = no delay)
    """
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)


# ==================== Preference Sampling Utilities ====================

def sample_preferences_uniform(batch_size, device, min_cost_weight=0.0, max_cost_weight=1.0):
    """
    Sample preference vectors uniformly from the valid range.
    
    Args:
        batch_size: Number of preferences to sample
        device: Device for tensor
        min_cost_weight: Minimum cost weight (default: 0.0)
        max_cost_weight: Maximum cost weight (default: 1.0)
        
    Returns:
        preferences: [batch_size, 2] tensor with [λ_cost, λ_carbon]
    """
    lambda_cost = torch.rand(batch_size, 1, device=device) * (max_cost_weight - min_cost_weight) + min_cost_weight
    lambda_carbon = 1.0 - lambda_cost
    preferences = torch.cat([lambda_cost, lambda_carbon], dim=1)
    return preferences


def sample_preferences_curriculum(batch_size, device, current_max_carbon_weight=0.1, 
                                   uniform_ratio=0.3):
    """
    Sample preferences with curriculum learning bias.
    
    Most samples come from easy region (high cost weight), with some exploration.
    Memory-optimized: uses pure PyTorch without numpy conversion.
    
    Args:
        batch_size: Number of preferences to sample
        device: Device
        current_max_carbon_weight: Maximum carbon weight for current curriculum stage
        uniform_ratio: Ratio of samples from uniform distribution (exploration)
        
    Returns:
        preferences: [batch_size, 2] tensor
    """
    n_uniform = int(batch_size * uniform_ratio)
    n_focused = batch_size - n_uniform
    
    # Pre-allocate output tensor
    preferences = torch.empty(batch_size, 2, device=device)
    
    # Focused sampling: higher probability for larger cost weights
    # Using rejection sampling with uniform distribution (simpler than Beta)
    if n_focused > 0:
        # Generate biased samples: use uniform and square to bias towards 1.0
        # This approximates Beta(2, 1) distribution
        u = torch.rand(n_focused, 1, device=device)
        # Square to bias towards higher values (cost weight)
        biased = u ** 0.5  # sqrt gives bias toward 1.0
        
        # Scale to valid range [1-current_max_carbon_weight, 1.0]
        min_cost = 1.0 - current_max_carbon_weight
        focused_cost = biased * current_max_carbon_weight + min_cost
        focused_carbon = 1.0 - focused_cost
        
        preferences[:n_focused, 0] = focused_cost.squeeze()
        preferences[:n_focused, 1] = focused_carbon.squeeze()
    
    # Uniform exploration
    if n_uniform > 0:
        uniform_cost = torch.rand(n_uniform, device=device) * current_max_carbon_weight + (1.0 - current_max_carbon_weight)
        uniform_carbon = 1.0 - uniform_cost
        
        preferences[n_focused:, 0] = uniform_cost
        preferences[n_focused:, 1] = uniform_carbon
    
    # Shuffle in-place
    perm = torch.randperm(batch_size, device=device)
    preferences = preferences[perm]
    
    return preferences


# ==================== Model Initialization Utilities ====================

def initialize_flow_model_near_zero(flow_model, scale=0.01):
    """
    Initialize flow model's output layer weights near zero.
    
    This ensures that initially the flow model produces very small velocities,
    so the output is close to the VAE anchor and doesn't drift away immediately.
    
    Args:
        flow_model: Flow model to initialize (supports various architectures)
        scale: Scale factor for initialization (default: 0.01)
    """
    # Find the last linear layer in the model
    # Support different model architectures:
    # - PreferenceConditionedNetV: has 'net' attribute (Sequential)
    # - Other flow models: may have 'model' attribute
    last_linear = None
    last_name = None
    
    with torch.no_grad():
        # Try different ways to access the model's modules
        if hasattr(flow_model, 'net'):
            # PreferenceConditionedNetV style
            target = flow_model.net
            prefix = 'net'
        elif hasattr(flow_model, 'model'):
            # Generic flow model style
            target = flow_model.model
            prefix = 'model'
        else:
            # Direct model
            target = flow_model
            prefix = ''
        
        for name, module in target.named_modules():
            if isinstance(module, nn.Linear):
                last_linear = module
                last_name = f"{prefix}.{name}" if prefix else name
        
        # Initialize the last layer with small weights
        if last_linear is not None:
            nn.init.normal_(last_linear.weight, mean=0.0, std=scale)
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)
            print(f"  Initialized output layer '{last_name}' with scale={scale}")
        else:
            print(f"  [Warning] No Linear layer found for zero-init")


# ==================== Pareto Front Evaluation ====================

def compute_hypervolume(points, ref_point):
    """
    Compute hypervolume indicator for Pareto front evaluation.
    
    The hypervolume is the area/volume dominated by the Pareto front
    and bounded by a reference point. Larger hypervolume = better Pareto front.
    
    Args:
        points: np.ndarray [N, 2] where each row is (cost, carbon)
                (minimization objectives)
        ref_point: np.ndarray [2] reference point (nadir point)
                   Should be worse than all Pareto points
                   
    Returns:
        hv: float, hypervolume value (0 if no valid points)
        
    Example:
        >>> points = np.array([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]])
        >>> ref_point = np.array([5.0, 5.0])
        >>> hv = compute_hypervolume(points, ref_point)
        >>> print(f"Hypervolume: {hv}")  # ~11.0
    """
    if len(points) == 0:
        return 0.0
    
    points = np.asarray(points)
    ref_point = np.asarray(ref_point)
    
    # Filter points that are dominated by reference point
    valid_mask = np.all(points < ref_point, axis=1)
    valid_points = points[valid_mask]
    
    if len(valid_points) == 0:
        return 0.0
    
    try:
        # Try to use pymoo for efficient computation
        from pymoo.indicators.hv import HV
        hv_indicator = HV(ref_point=ref_point)
        hv = hv_indicator(valid_points)
        return float(hv)
    except ImportError:
        # Fallback: simple 2D hypervolume calculation using sweep line algorithm
        # For 2D minimization problems, this computes the area dominated by Pareto front
        
        # Sort by first objective (cost) ascending
        sorted_indices = np.argsort(valid_points[:, 0])
        sorted_points = valid_points[sorted_indices]
        
        hv = 0.0
        prev_y = ref_point[1]  # Start from reference point's y
        
        for point in sorted_points:
            x, y = point
            if y < prev_y:  # This point contributes new area
                # Width: from current x to reference x
                width = ref_point[0] - x
                # Height: from current y up to previous y level
                height = prev_y - y
                hv += width * height
                prev_y = y  # Update the y level for next iteration
        
        return hv


def check_feasibility(loss_dict, thresholds=None):
    """
    Check if a solution satisfies constraint feasibility.
    
    Args:
        loss_dict: Dictionary with constraint violation values:
            - load_dev: Load deviation (p.u. or MW)
            - gen_vio: Generator violation
            - branch_pf_vio: Branch power flow violation (optional)
            - branch_ang_vio: Branch angle violation (optional)
        thresholds: Dictionary with threshold values:
            - load_dev: Maximum allowed load deviation (default: 0.01)
            - gen_vio: Maximum allowed generator violation (default: 0.001)
            
    Returns:
        feasible: bool, True if all constraints are satisfied
    """
    if thresholds is None:
        thresholds = {
            'load_dev': 0.01,    # 1% load deviation
            'gen_vio': 0.001,    # 0.1% generator violation
        }
    
    feasible = True
    
    if 'load_dev' in loss_dict and 'load_dev' in thresholds:
        if loss_dict['load_dev'] > thresholds['load_dev']:
            feasible = False
    
    if 'gen_vio' in loss_dict and 'gen_vio' in thresholds:
        if loss_dict['gen_vio'] > thresholds['gen_vio']:
            feasible = False
    
    return feasible


def evaluate_pareto_front(costs, carbons, feasible_mask, ref_point=None):
    """
    Evaluate Pareto front quality with hypervolume and feasibility metrics.
    
    Args:
        costs: np.ndarray [N] economic costs for each solution
        carbons: np.ndarray [N] carbon emissions for each solution
        feasible_mask: np.ndarray [N] bool, True if solution is feasible
        ref_point: np.ndarray [2] reference point for hypervolume
                   If None, uses 1.1 × max values
                   
    Returns:
        dict: {
            'hypervolume': float,      # Hypervolume of feasible solutions
            'feasible_ratio': float,   # Ratio of feasible solutions
            'n_feasible': int,         # Number of feasible solutions
            'n_total': int,            # Total number of solutions
            'pareto_points': np.ndarray # [n_feasible, 2] Pareto front points
        }
    """
    costs = np.asarray(costs)
    carbons = np.asarray(carbons)
    feasible_mask = np.asarray(feasible_mask, dtype=bool)
    
    n_total = len(costs)
    n_feasible = np.sum(feasible_mask)
    feasible_ratio = n_feasible / max(n_total, 1)
    
    result = {
        'hypervolume': 0.0,
        'feasible_ratio': feasible_ratio,
        'n_feasible': int(n_feasible),
        'n_total': n_total,
        'pareto_points': np.array([]).reshape(0, 2)
    }
    
    if n_feasible == 0:
        return result
    
    # Extract feasible solutions
    feasible_costs = costs[feasible_mask]
    feasible_carbons = carbons[feasible_mask]
    pareto_points = np.column_stack([feasible_costs, feasible_carbons])
    
    # Determine reference point
    if ref_point is None:
        ref_point = np.array([
            np.max(feasible_costs) * 1.1,
            np.max(feasible_carbons) * 1.1
        ])
    
    # Compute hypervolume
    hv = compute_hypervolume(pareto_points, ref_point)
    
    result['hypervolume'] = hv
    result['pareto_points'] = pareto_points
    result['ref_point'] = ref_point
    
    return result


def get_pareto_validation_metric(hypervolume, feasible_ratio, 
                                  hv_weight=0.7, feas_weight=0.3,
                                  normalize_hv=True, hv_scale=1e6):
    """
    Compute combined validation metric for early stopping.
    
    Higher value = better Pareto front.
    
    Args:
        hypervolume: Hypervolume value
        feasible_ratio: Ratio of feasible solutions [0, 1]
        hv_weight: Weight for hypervolume term
        feas_weight: Weight for feasibility term
        normalize_hv: Whether to normalize hypervolume
        hv_scale: Scale factor for hypervolume normalization
        
    Returns:
        metric: float, combined validation metric (higher is better)
    """
    if normalize_hv:
        norm_hv = hypervolume / max(hv_scale, 1.0)
    else:
        norm_hv = hypervolume
    
    metric = hv_weight * norm_hv + feas_weight * feasible_ratio
    return metric


# ==================== Evaluation Metrics ====================

def get_clamp(Pred, Predmin, Predmax):
    """
    Clamp predicted values within min/max bounds
    
    Args:
        Pred: Predicted values (batch_size, num_features)
        Predmin: Minimum values for each feature
        Predmax: Maximum values for each feature
        
    Returns:
        Pred_clip: Clamped predictions
    """
    Pred_clip = Pred.clone()
    for i in range(Pred.shape[1]):
        Pred_clip[:, i] = Pred_clip[:, i].clamp(min=Predmin[i])
        Pred_clip[:, i] = Pred_clip[:, i].clamp(max=Predmax[i])
    
    return Pred_clip


def get_mae(real, predict):
    """
    Calculate Mean Absolute Error
    
    Args:
        real: Real values
        predict: Predicted values
        
    Returns:
        err: Mean absolute error
    """
    if len(real) == len(predict):
        err = torch.mean(torch.abs(real - predict))  
        return err
    else:
        return None


def get_rerr(real, predict):
    """
    Calculate relative error (absolute value)
    
    Args:
        real: Real values
        predict: Predicted values
        
    Returns:
        err: Relative error in percentage
    """
    if len(real) == len(predict):
        err = torch.abs((predict - real) / real) * 100
        return err
    else:
        return None


def get_rerr2(real, predict):
    """
    Calculate relative error (signed)
    
    Args:
        real: Real values
        predict: Predicted values
        
    Returns:
        err: Signed relative error in percentage
    """
    if len(real) == len(predict):
        err = (predict - real) / real * 100
        return err
    else:
        return None


# ==================== Power System Calculations ====================

def get_PQ(V, Ybus):
    """
    Calculate active and reactive power at each bus
    
    Args:
        V: Complex voltage (num_samples, num_buses)
        Ybus: Bus admittance matrix (sparse)
        
    Returns:
        P: Active power
        Q: Reactive power
    """
    S = np.zeros(V.shape, dtype=np.complex128)
    for i in range(V.shape[0]):
        I = Ybus.dot(V[i]).conj()
        S[i] = np.multiply(V[i], I)
    
    P = np.real(S)
    Q = np.imag(S) 
    return P, Q


def get_genload(V, Pdtest, Qdtest, bus_Pg, bus_Qg, Ybus):
    """
    Calculate generation and load at each bus
    
    Args:
        V: Complex voltage
        Pdtest: Active load demand
        Qdtest: Reactive load demand
        bus_Pg: Buses with active generation
        bus_Qg: Buses with reactive generation
        Ybus: Bus admittance matrix
        
    Returns:
        Pg: Active generation
        Qg: Reactive generation
        Pd: Active load
        Qd: Reactive load
    """
    S = np.zeros(V.shape, dtype=np.complex128)
    for i in range(V.shape[0]):
        I = Ybus.dot(V[i]).conj()
        S[i] = np.multiply(V[i], I)
    
    P = np.real(S)
    Q = np.imag(S) 
    
    Pg = P[:, bus_Pg] + Pdtest[:, bus_Pg]
    Qg = Q[:, bus_Qg] + Qdtest[:, bus_Qg]   
    Pd = -P * 1.0
    Qd = -Q * 1.0
    Pd[:, bus_Pg] = Pg - P[:, bus_Pg]
    Qd[:, bus_Qg] = Qg - Q[:, bus_Qg]   
    return Pg, Qg, Pd, Qd


def get_Pgcost(Pg, idxPg, gencost, baseMVA):
    """
    Calculate generation cost
    
    Args:
        Pg: Active generation (p.u.)
        idxPg: Generator indices
        gencost: Cost coefficients (can be MATPOWER format or simplified format)
        baseMVA: Base MVA
        
    Returns:
        cost: Total generation cost for each sample
    """
    # Determine which columns to use based on gencost format
    # MATPOWER format: [MODEL, STARTUP, SHUTDOWN, NCOST, c2, c1, c0] (7 columns)
    # Simplified format: [c2, c1] or [c2, c1, ...] (2+ columns)
    if gencost.shape[1] > 4:
        # MATPOWER format: use columns 4 (c2) and 5 (c1)
        col_c2, col_c1 = 4, 5
    else:
        # Simplified format: use columns 0 (c2) and 1 (c1)
        col_c2, col_c1 = 0, 1
    
    cost = np.zeros(Pg.shape[0])
    PgMVA = Pg * baseMVA
    for i in range(Pg.shape[0]): 
        c1 = np.multiply(gencost[idxPg, col_c2], np.multiply(PgMVA[i, :], PgMVA[i, :]))
        c2 = np.multiply(gencost[idxPg, col_c1], PgMVA[i, :])
        cost[i] = np.sum(c1 + c2)
            
    return cost


# ==================== Constraint Violation Checking ====================

def get_vioPQg(Pred_Pg, bus_Pg, MAXMIN_Pg, Pred_Qg, bus_Qg, MAXMIN_Qg, DELTA):
    """
    Check Pg and Qg constraint violations
    
    Args:
        Pred_Pg: Predicted active generation
        bus_Pg: Buses with active generation
        MAXMIN_Pg: Active power limits [Pmin, Pmax]
        Pred_Qg: Predicted reactive generation
        bus_Qg: Buses with reactive generation
        MAXMIN_Qg: Reactive power limits [Qmin, Qmax]
        DELTA: Violation threshold
        
    Returns:
        Multiple outputs for violation analysis
    """
    vio_PQgmaxminnum = torch.zeros((Pred_Pg.shape[0], 4))
    vio_PQgmaxmin = torch.zeros((Pred_Pg.shape[0], 4))
    vio_PQg = torch.zeros((Pred_Pg.shape[0], 2))
    lsPg = list()
    lsQg = list()
    lsidxPg = np.zeros((Pred_Pg.shape[0]), dtype=int)
    lsidxQg = np.zeros((Pred_Pg.shape[0]), dtype=int)
    kP = 1
    kQ = 1
    deltaPgL = np.array([[0, 0]])
    deltaPgU = np.array([[0, 0]])
    deltaQgL = np.array([[0, 0]])
    deltaQgU = np.array([[0, 0]])
    
    for i in range(Pred_Pg.shape[0]):
        # Active power
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

        # Reactive power
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
    vio_PQgmaxmin[:, 0] = (1 - vio_PQgmaxminnum[:, 0] / bus_Pg.shape[0]) * 100
    vio_PQgmaxmin[:, 1] = (1 - vio_PQgmaxminnum[:, 1] / bus_Pg.shape[0]) * 100
    vio_PQgmaxmin[:, 2] = (1 - vio_PQgmaxminnum[:, 2] / bus_Qg.shape[0]) * 100
    vio_PQgmaxmin[:, 3] = (1 - vio_PQgmaxminnum[:, 3] / bus_Qg.shape[0]) * 100
    vio_PQg[:, 0] = (1 - (vio_PQgmaxminnum[:, 0] + vio_PQgmaxminnum[:, 1]) / bus_Pg.shape[0]) * 100
    vio_PQg[:, 1] = (1 - (vio_PQgmaxminnum[:, 2] + vio_PQgmaxminnum[:, 3]) / bus_Qg.shape[0]) * 100
     
    # Clean up initial dummy rows
    if deltaPgL.shape[0] > 1:
        deltaPgL = np.delete(deltaPgL, 0, axis=0)
        
    if deltaPgU.shape[0] > 1:    
        deltaPgU = np.delete(deltaPgU, 0, axis=0)
    
    if deltaQgL.shape[0] > 1:
        deltaQgL = np.delete(deltaQgL, 0, axis=0)
    
    if deltaQgU.shape[0] > 1:
        deltaQgU = np.delete(deltaQgU, 0, axis=0)
    
    return lsPg, lsQg, lsidxPg, lsidxQg, vio_PQgmaxmin, vio_PQg, deltaPgL, deltaPgU, deltaQgL, deltaQgU


def get_viobran(Pred_V, Pred_Va, branch, Yf, Yt, BRANFT, baseMVA, DELTA):
    """
    Check branch constraint violations
    
    Args:
        Pred_V: Predicted voltage magnitude
        Pred_Va: Predicted voltage angle
        branch: Branch data
        Yf: Branch from admittance
        Yt: Branch to admittance
        BRANFT: Branch from-to indices
        baseMVA: Base MVA
        DELTA: Violation threshold
        
    Returns:
        vio_branang: Branch angle violation ratio
        vio_branpf: Branch power flow violation ratio
        deltapf: Violation details
    """
    # Ensure branlp is 1D array and baseMVA is scalar (fixes broadcasting issues)
    baseMVA_scalar = float(np.asarray(baseMVA).ravel()[0])  # Convert to scalar
    branlp = np.asarray(branch[:, 2]).ravel() / baseMVA_scalar  # Column 2: branch power limit
    angminmax = branch[:, 3:5] * math.pi / 180  # Columns 3:5: angle limits
    Pred_branang = Pred_Va[:, BRANFT[:, 0]] - Pred_Va[:, BRANFT[:, 1]]
    vio_branangnum = torch.zeros(Pred_V.shape[0])
    vio_branpfnum = torch.zeros(Pred_V.shape[0])
    # Use list instead of np.append for O(1) append instead of O(n)
    deltapf_list = []
    
    for i in range(Pred_V.shape[0]):
        vio_branangnum[i] = np.size(np.where(Pred_branang[i, :] - angminmax[:, 0] < -DELTA)) \
                          + np.size(np.where(Pred_branang[i, :] - angminmax[:, 1] > DELTA))

        # Branch power flow
        fV = Pred_V[i, BRANFT[:, 0]]
        tV = Pred_V[i, BRANFT[:, 1]]
        fI = Yf.dot(Pred_V[i]).conj()
        tI = Yt.dot(Pred_V[i]).conj()       
        fS = np.multiply(fV, fI)
        tS = np.multiply(tV, tI)
        deltafS = np.abs(fS) - branlp
        deltatS = np.abs(tS) - branlp
        deltafS = np.array(deltafS).ravel()
        deltatS = np.array(deltatS).ravel()
        idxfs = np.array(np.where(deltafS > DELTA))
        idxts = np.array(np.where(deltatS > DELTA))
        vio_branpfnum[i] = np.size(idxfs) + np.size(idxts)
        
        # Save violated samples - use list for O(1) append
        if np.size(idxfs) >= 1:
            ii = np.concatenate((idxfs, deltafS[idxfs]), axis=0)
            deltapf_list.append(ii.T)
            
        if np.size(idxts) >= 1:
            ii = np.concatenate((idxts, deltatS[idxts]), axis=0)
            deltapf_list.append(ii.T)
    
    # Convert list to array once at the end
    if deltapf_list:
        deltapf = np.vstack(deltapf_list)
        # Calculate relative violation percentages
        branch_indices = np.asarray(deltapf[:, 0]).ravel().astype(int)
        valid_mask = (branch_indices >= 0) & (branch_indices < len(branlp))
        if np.any(valid_mask):
            deltapfR = np.zeros(len(branch_indices))
            delta_values = np.asarray(deltapf[valid_mask, 1]).ravel()
            branch_limits = np.asarray(branlp[branch_indices[valid_mask]]).ravel()
            deltapfR[valid_mask] = delta_values / branch_limits * 100
            deltapf = np.insert(deltapf, 2, values=deltapfR, axis=1)
        else:
            deltapf = np.insert(deltapf, 2, values=np.zeros(len(branch_indices)), axis=1)
    else:
        deltapf = np.array([[0, 0, 0]])  # Empty case with 3 columns
 
    vio_branang = (1 - vio_branangnum / branch.shape[0]) * 100 
    vio_branpf = (1 - vio_branpfnum / (branch.shape[0] * 2)) * 100
    return vio_branang, vio_branpf, deltapf


def get_viobran2(Pred_V, Pred_Va, branch, Yf, Yt, BRANFT, baseMVA, DELTA):
    """
    Check branch constraint violations with detailed information for post-processing
    
    Similar to get_viobran but returns additional details for correction
    """
    # Ensure branlp is 1D array and baseMVA is scalar (fixes broadcasting issues)
    baseMVA_scalar = float(np.asarray(baseMVA).ravel()[0])  # Convert to scalar
    branlp = np.asarray(branch[:, 2]).ravel() / baseMVA_scalar  # Column 2: branch power limit
    angminmax = branch[:, 3:5] * math.pi / 180  # Columns 3:5: angle limits
    Pred_branang = Pred_Va[:, BRANFT[:, 0]] - Pred_Va[:, BRANFT[:, 1]]
    vio_branangnum = torch.zeros(Pred_V.shape[0])
    vio_branpfnum = torch.zeros(Pred_V.shape[0])
    vio_branpfidx = torch.zeros(Pred_V.shape[0])
    lsSf = []
    lsSt = []
    lsSf_sampidx = []
    lsSt_sampidx = []
    # Use list instead of np.append for O(1) append instead of O(n)
    deltapf_list = []
    
    for i in range(Pred_V.shape[0]):
        vio_branangnum[i] = np.size(np.where(Pred_branang[i, :] - angminmax[:, 0] < -DELTA)) \
                          + np.size(np.where(Pred_branang[i, :] - angminmax[:, 1] > DELTA))

        # Branch power flow
        fV = Pred_V[i, BRANFT[:, 0]]
        tV = Pred_V[i, BRANFT[:, 1]]        
        fI = Yf.dot(Pred_V[i]).conj()
        tI = Yt.dot(Pred_V[i]).conj()
        fS = np.multiply(fV, fI)
        tS = np.multiply(tV, tI)
        deltafS = np.abs(fS) - branlp
        deltatS = np.abs(tS) - branlp
        deltafS = np.array(deltafS).ravel()
        deltatS = np.array(deltatS).ravel()
        idxfs = np.array(np.where(deltafS > DELTA)).reshape(-1, 1)
        idxts = np.array(np.where(deltatS > DELTA)).reshape(-1, 1)
        vio_branpfnum[i] = np.size(idxfs) + np.size(idxts)
        
        if np.size(idxfs) >= 1:           
            ii = np.concatenate((idxfs, deltafS[idxfs]), axis=1)  
            deltapf_list.append(ii)  # O(1) instead of O(n)
            ii = np.concatenate((ii, np.real(fS[idxfs]), np.imag(fS[idxfs])), axis=1)
            lsSf.append(ii)
            lsSf_sampidx.append(i)           
            
        if np.size(idxts) >= 1:
            ii = np.concatenate((idxts, deltatS[idxts]), axis=1)
            deltapf_list.append(ii)  # O(1) instead of O(n)
            ii = np.concatenate((ii, np.real(tS[idxts]), np.imag(tS[idxts])), axis=1)
            lsSt.append(ii)
            lsSt_sampidx.append(i)

        if np.size(idxfs) + np.size(idxts) >= 1:
            vio_branpfidx[i] = i + 1
    
    # Convert list to array once at the end (O(n) total instead of O(n²))
    if deltapf_list:
        deltapf = np.vstack(deltapf_list)
        # Calculate relative violation percentages
        branch_indices = np.asarray(deltapf[:, 0]).ravel().astype(int)
        valid_mask = (branch_indices >= 0) & (branch_indices < len(branlp))
        if np.any(valid_mask):
            deltapfR = np.zeros(len(branch_indices))
            delta_values = np.asarray(deltapf[valid_mask, 1]).ravel()
            branch_limits = np.asarray(branlp[branch_indices[valid_mask]]).ravel()
            deltapfR[valid_mask] = delta_values / branch_limits * 100
            deltapf = np.insert(deltapf, 2, values=deltapfR, axis=1)
        else:
            deltapf = np.insert(deltapf, 2, values=np.zeros(len(branch_indices)), axis=1)
    else:
        deltapf = np.array([[0, 0, 0]])  # Empty case with 3 columns
 
    vio_branang = (1 - vio_branangnum / branch.shape[0]) * 100 
    vio_branpf = (1 - vio_branpfnum / (branch.shape[0] * 2)) * 100
    return vio_branang, vio_branpf, deltapf, vio_branpfidx, lsSf, lsSt, lsSf_sampidx, lsSt_sampidx


# ==================== Post-Processing Functions ====================

def dPQbus_dV(his_V, bus_Pg, bus_Qg, Ybus):
    """
    Calculate Jacobian matrix dP/dV and dQ/dV at buses
    
    Args:
        his_V: Historical voltage
        bus_Pg: Buses with active generation
        bus_Qg: Buses with reactive generation
        Ybus: Bus admittance matrix
        
    Returns:
        dPbus_dV: dP/dV Jacobian
        dQbus_dV: dQ/dV Jacobian
    """
    V = his_V.copy()
    Ibus = Ybus.dot(his_V).conj()
    diagV = np.diag(V)
    diagIbus = np.diag(Ibus)
    diagVnorm = np.diag(V / np.abs(V))
    dSbus_dVm = np.dot(diagV, Ybus.dot(diagVnorm).conj()) + np.dot(diagIbus.conj(), diagVnorm)
    dSbus_dVa = 1j * np.dot(diagV, (diagIbus - Ybus.dot(diagV)).conj())
    dSbus_dV = np.concatenate((dSbus_dVa, dSbus_dVm), axis=1)
    dPbus_dV = np.real(dSbus_dV)
    dQbus_dV = np.imag(dSbus_dV)
    
    return dPbus_dV, dQbus_dV


def get_hisdV(lsPg, lsQg, lsidxPg, lsidxQg, num_viotest, k_dV, bus_Pg, bus_Qg, dPbus_dV, dQbus_dV, Nbus, Ntest):
    """
    Calculate voltage correction using historical voltage Jacobian
    
    Args:
        lsPg, lsQg: Lists of Pg/Qg violations
        lsidxPg, lsidxQg: Indices of violated samples
        num_viotest: Number of violated test samples
        k_dV: Correction coefficient
        bus_Pg, bus_Qg: Generator bus indices
        dPbus_dV, dQbus_dV: Jacobian matrices
        Nbus: Number of buses
        Ntest: Number of test samples
        
    Returns:
        dV: Voltage corrections
    """
    dV = np.zeros((num_viotest, Nbus * 2))
    j = 0
    for i in range(Ntest):
        if (lsidxPg[i] + lsidxQg[i]) > 0:
            if lsidxPg[i] > 0 and lsidxQg[i] > 0:
                idxPg = lsPg[lsidxPg[i] - 1][:, 0].astype(np.int32)
                idxQg = lsQg[lsidxQg[i] - 1][:, 0].astype(np.int32)
                busPg = bus_Pg[idxPg]
                busQg = bus_Qg[idxQg]
                dPQGbus_dV = np.concatenate((dPbus_dV[busPg, :], dQbus_dV[busQg, :]), axis=0)
                dPQg = np.concatenate((lsPg[lsidxPg[i] - 1][:, 1], lsQg[lsidxQg[i] - 1][:, 1]), axis=0)
            elif lsidxPg[i] > 0:  
                idxPg = lsPg[lsidxPg[i] - 1][:, 0].astype(np.int32)
                busPg = bus_Pg[idxPg]
                dPQGbus_dV = dPbus_dV[busPg, :]
                dPQg = lsPg[lsidxPg[i] - 1][:, 1]
            elif lsidxQg[i] > 0:     
                idxQg = lsQg[lsidxQg[i] - 1][:, 0].astype(np.int32)
                busQg = bus_Qg[idxQg]
                dPQGbus_dV = dQbus_dV[busQg, :]
                dPQg = lsQg[lsidxQg[i] - 1][:, 1]

            dV[j] = np.dot(np.linalg.pinv(dPQGbus_dV), dPQg * k_dV)           
            j += 1   
            
    return dV


def get_dV(Pred_V, lsPg, lsQg, lsidxPg, lsidxQg, num_viotest, k_dV, bus_Pg, bus_Qg, Ybus, his_V):
    """
    Calculate voltage correction using predicted voltage Jacobian
    
    Similar to get_hisdV but uses predicted voltage instead of historical
    """
    dV = np.zeros((num_viotest, Pred_V.shape[1] * 2))
    j = 0
    for i in range(Pred_V.shape[0]):
        if (lsidxPg[i] + lsidxQg[i]) > 0:
            # Calculate Jacobian for predicted voltage
            V = Pred_V[i].copy()
            Ibus = Ybus.dot(his_V).conj()
            diagV = np.diag(V)
            diagIbus = np.diag(Ibus)
            diagVnorm = np.diag(V / np.abs(V))

            dSbus_dVm = np.dot(diagV, Ybus.dot(diagVnorm).conj()) + np.dot(diagIbus.conj(), diagVnorm)
            dSbus_dVa = 1j * np.dot(diagV, (diagIbus - Ybus.dot(diagV)).conj())
            
            dSbus_dV = np.concatenate((dSbus_dVa, dSbus_dVm), axis=1)
            dPbus_dV = np.real(dSbus_dV)
            dQbus_dV = np.imag(dSbus_dV)
            
            if lsidxPg[i] > 0 and lsidxQg[i] > 0:
                idxPg = lsPg[lsidxPg[i] - 1][:, 0].astype(np.int32)
                idxQg = lsQg[lsidxQg[i] - 1][:, 0].astype(np.int32)
                busPg = bus_Pg[idxPg]
                busQg = bus_Qg[idxQg]
                dPQGbus_dV = np.concatenate((dPbus_dV[busPg, :], dQbus_dV[busQg, :]), axis=0)
                dPQg = np.concatenate((lsPg[lsidxPg[i] - 1][:, 1], lsQg[lsidxQg[i] - 1][:, 1]), axis=0)
            elif lsidxPg[i] > 0:  
                idxPg = lsPg[lsidxPg[i] - 1][:, 0].astype(np.int32)
                busPg = bus_Pg[idxPg]
                dPQGbus_dV = dPbus_dV[busPg, :]
                dPQg = lsPg[lsidxPg[i] - 1][:, 1]
            elif lsidxQg[i] > 0:     
                idxQg = lsQg[lsidxQg[i] - 1][:, 0].astype(np.int32)
                busQg = bus_Qg[idxQg]
                dPQGbus_dV = dQbus_dV[busQg, :]
                dPQg = lsQg[lsidxQg[i] - 1][:, 1]

            dV[j] = np.dot(np.linalg.pinv(dPQGbus_dV), dPQg * k_dV)           
            j += 1        
    return dV


def dSlbus_dV(his_V, bus_Va, branch, Yf, finc, BRANFT, Nbus):
    """
    Calculate derivative of branch power flow with respect to voltage
    
    Args:
        his_V: Historical voltage
        bus_Va: Bus voltage angle indices (excluding slack)
        branch: Branch data
        Yf: Branch from admittance
        finc: Branch from incidence matrix
        BRANFT: Branch from-to indices
        Nbus: Number of buses
        
    Returns:
        dPfbus_dV: dPf/dV Jacobian
        dQfbus_dV: dQf/dV Jacobian
    """
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
    # Use all buses (including slack) to match the 2*Nbus format expected by post-processing
    # Note: bus_Va parameter is kept for API compatibility but not used anymore
    dfP_dVa = np.real(dfS_dVa)  # Full Nbus columns
    dfQ_dVa = np.imag(dfS_dVa)  # Full Nbus columns
    
    dPfbus_dV = np.concatenate((dfP_dVa, dfP_dVm), axis=1)
    dQfbus_dV = np.concatenate((dfQ_dVa, dfQ_dVm), axis=1)

    return dPfbus_dV, dQfbus_dV


# ==================== Carbon Emission Calculations ====================

def get_carbon_emission(Pg, idxPg, gci_values, baseMVA):
    """
    Calculate carbon emission for each sample
    
    Carbon emission is calculated as: Carbon = Σ GCI_i × Pg_i
    where GCI is the carbon emission intensity (tCO2/MWh) for each generator.
    
    Args:
        Pg: Active generation (p.u.) [n_samples, n_gen]
        idxPg: Generator indices (which generators are active)
        gci_values: GCI (carbon emission intensity) for each generator (tCO2/MWh)
                   Should be an array of shape [n_gen]
        baseMVA: Base MVA for converting p.u. to MW
        
    Returns:
        carbon: Carbon emission (tCO2/h) for each sample [n_samples]
    
    Example:
        >>> # Assuming Pg is [100 samples, 54 generators] in p.u.
        >>> gci = np.array([0.82, 0.51, ...])  # GCI for each generator
        >>> carbon = get_carbon_emission(Pg, idxPg, gci, baseMVA=100)
    """
    carbon = np.zeros(Pg.shape[0])
    PgMVA = Pg * baseMVA  # Convert from p.u. to MW
    
    for i in range(Pg.shape[0]):
        # Carbon emission = Σ GCI_i × Pg_i (MW × tCO2/MWh = tCO2/h)
        carbon[i] = np.sum(gci_values[idxPg] * PgMVA[i, :])
    
    return carbon


def get_carbon_emission_vectorized(Pg, gci_values, baseMVA):
    """
    Vectorized version of carbon emission calculation (faster for large batches)
    
    Args:
        Pg: Active generation (p.u.) [n_samples, n_gen]
        gci_values: GCI for each generator (tCO2/MWh) [n_gen]
        baseMVA: Base MVA
        
    Returns:
        carbon: Carbon emission (tCO2/h) for each sample [n_samples]
    """
    PgMVA = Pg * baseMVA  # Convert from p.u. to MW
    # Matrix multiplication: [n_samples, n_gen] @ [n_gen] = [n_samples]
    carbon = np.dot(PgMVA, gci_values)
    return carbon


# ==================== TensorBoard Logging Utilities ====================

class TensorBoardLogger:
    """
    TensorBoard logger for Pareto Flow training.
    
    Records:
    - Loss components (total, objective, constraints)
    - Objective space (cost, carbon emission)  
    - Solution trajectory (movement from anchor to target)
    - Constraint violations (load deviation, gen limits, branch limits)
    
    Usage:
        logger = TensorBoardLogger(log_dir='runs/pareto_flow')
        logger.log_losses(epoch, loss_dict)
        logger.log_objectives(epoch, cost, carbon)
        logger.log_constraint_violations(epoch, vio_dict)
        logger.close()
    """
    
    def __init__(self, log_dir='runs/pareto_flow', comment=''):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory to save TensorBoard logs
            comment: Additional comment for run name
        """
        # Initialize log_file to None first
        self.log_file = None
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            import datetime
            import os
            
            # Create unique run name with timestamp
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            run_name = f"{log_dir}/{timestamp}"
            if comment:
                run_name += f"_{comment}"
            
            self.writer = SummaryWriter(run_name)
            self.enabled = True
            print(f"[TensorBoard] Logging to: {run_name}")
            
            # Create log file for TensorBoard metrics
            # Save log file in results directory (same as training logs)
            try:
                results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'main_part', 'results')
                os.makedirs(results_dir, exist_ok=True)
                
                # Create log file name based on comment
                log_filename = f"tb_logs_{comment}.txt" if comment else f"tb_logs_{timestamp}.txt"
                log_file_path = os.path.join(results_dir, log_filename)
                
                self.log_file = open(log_file_path, 'w', encoding='utf-8')
                self.log_file.write(f"TensorBoard Metrics Log\n")
                self.log_file.write(f"Run: {run_name}\n")
                self.log_file.write(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.log_file.write("=" * 80 + "\n\n")
                self.log_file.flush()
                print(f"[TensorBoard] Metrics log file: {log_file_path}")
            except Exception as e:
                print(f"[TensorBoard] Warning: Failed to create log file: {e}")
                self.log_file = None
        except ImportError:
            print("[TensorBoard] Warning: tensorboard not installed. Logging disabled.")
            print("  Install with: pip install tensorboard")
            self.writer = None
            self.enabled = False
        except Exception as e:
            print(f"[TensorBoard] Warning: Failed to initialize TensorBoard logger: {e}")
            self.writer = None
            self.enabled = False
    
    def log_scalar(self, tag, value, step):
        """Log a scalar value."""
        if self.enabled and self.writer is not None:
            self.writer.add_scalar(tag, value, step)
        
        # Also write to log file
        if self.log_file is not None:
            try:
                self.log_file.write(f"Epoch {step:6d} | {tag:40s} = {value:.6e}\n")
                self.log_file.flush()
            except Exception:
                pass  # Silently fail if log write fails
    
    def log_scalars(self, main_tag, tag_scalar_dict, step):
        """Log multiple scalars under the same main tag."""
        if self.enabled and self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        
        # Also write to log file
        if self.log_file is not None:
            try:
                self.log_file.write(f"Epoch {step:6d} | {main_tag}:\n")
                for tag, value in tag_scalar_dict.items():
                    self.log_file.write(f"  {tag:40s} = {value:.6e}\n")
                self.log_file.flush()
            except Exception:
                pass  # Silently fail if log write fails
    
    def log_losses(self, step, loss_dict, prefix='train'):
        """
        Log loss components.
        
        Args:
            step: Training step (epoch or batch)
            loss_dict: Dictionary with loss components
            prefix: 'train' or 'val'
        """
        if not self.enabled:
            return
        
        # Total loss
        if 'total' in loss_dict:
            self.log_scalar(f'{prefix}/loss_total', loss_dict['total'], step)
        
        # Objective components
        if 'objective' in loss_dict:
            self.log_scalar(f'{prefix}/loss_objective', loss_dict['objective'], step)
        if 'cost' in loss_dict:
            self.log_scalar(f'{prefix}/objective_cost', loss_dict['cost'], step)
        if 'carbon' in loss_dict:
            self.log_scalar(f'{prefix}/objective_carbon', loss_dict['carbon'], step)
        
        # Constraint components
        if 'constraints' in loss_dict:
            self.log_scalar(f'{prefix}/loss_constraints', loss_dict['constraints'], step)
    
    def log_objectives(self, step, cost, carbon, prefix='train'):
        """
        Log objective space values (cost vs carbon).
        
        Args:
            step: Training step
            cost: Economic cost ($/h)
            carbon: Carbon emission (tCO2/h)
            prefix: 'train' or 'val'
        """
        if not self.enabled:
            return
        
        self.log_scalars(f'{prefix}/objectives', {
            'cost': cost,
            'carbon': carbon
        }, step)
    
    def log_constraint_violations(self, step, vio_dict, prefix='train'):
        """
        Log constraint violation metrics.
        
        Args:
            step: Training step
            vio_dict: Dictionary with violation metrics:
                - load_dev: Load deviation
                - gen_vio: Generator limit violation
                - branch_pf_vio: Branch power flow violation
                - branch_ang_vio: Branch angle violation
            prefix: 'train' or 'val'
        """
        if not self.enabled:
            return
        
        # Load balance constraint
        if 'load_dev' in vio_dict:
            self.log_scalar(f'{prefix}/vio_load_deviation', vio_dict['load_dev'], step)
        
        # Generator constraints
        if 'gen_vio' in vio_dict:
            self.log_scalar(f'{prefix}/vio_generator', vio_dict['gen_vio'], step)
        
        # Branch constraints
        if 'branch_pf_vio' in vio_dict:
            self.log_scalar(f'{prefix}/vio_branch_power', vio_dict['branch_pf_vio'], step)
        if 'branch_ang_vio' in vio_dict:
            self.log_scalar(f'{prefix}/vio_branch_angle', vio_dict['branch_ang_vio'], step)
        
        # Combined view
        vio_scalars = {}
        for key in ['load_dev', 'gen_vio', 'branch_pf_vio', 'branch_ang_vio']:
            if key in vio_dict:
                vio_scalars[key] = vio_dict[key]
        if vio_scalars:
            self.log_scalars(f'{prefix}/violations', vio_scalars, step)
    
    def log_solution_trajectory(self, step, anchor_cost, anchor_carbon, 
                                 final_cost, final_carbon, 
                                 target_lambda_cost, target_lambda_carbon):
        """
        Log solution trajectory in objective space.
        
        Shows how the solution moves from anchor (VAE output) to final (Flow output).
        
        Args:
            step: Training step
            anchor_cost, anchor_carbon: Anchor point objectives (VAE output)
            final_cost, final_carbon: Final point objectives (Flow output)
            target_lambda_cost, target_lambda_carbon: Target preference weights
        """
        if not self.enabled:
            return
        
        # Anchor point (starting point, preference [1,0])
        self.log_scalar('trajectory/anchor_cost', anchor_cost, step)
        self.log_scalar('trajectory/anchor_carbon', anchor_carbon, step)
        
        # Final point (after flow, target preference)
        self.log_scalar('trajectory/final_cost', final_cost, step)
        self.log_scalar('trajectory/final_carbon', final_carbon, step)
        
        # Movement delta
        delta_cost = final_cost - anchor_cost
        delta_carbon = final_carbon - anchor_carbon
        self.log_scalar('trajectory/delta_cost', delta_cost, step)
        self.log_scalar('trajectory/delta_carbon', delta_carbon, step)
        
        # Target preference
        self.log_scalar('trajectory/target_lambda_cost', target_lambda_cost, step)
        self.log_scalar('trajectory/target_lambda_carbon', target_lambda_carbon, step)
    
    def log_pareto_scatter(self, step, costs, carbons, tag='pareto_front'):
        """
        Log Pareto front as scatter plot (for batch of solutions).
        
        Args:
            step: Training step
            costs: Array of cost values [batch_size]
            carbons: Array of carbon values [batch_size]
            tag: Tag for the figure
        """
        if not self.enabled:
            return
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import io
            from PIL import Image
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(costs, carbons, alpha=0.5, s=10)
            ax.set_xlabel('Cost ($/h)')
            ax.set_ylabel('Carbon Emission (tCO2/h)')
            ax.set_title(f'Solution Distribution (Step {step})')
            ax.grid(True, alpha=0.3)
            
            # Convert to image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            image = Image.open(buf)
            
            # Convert to tensor format for TensorBoard
            import numpy as np
            image_array = np.array(image)
            if image_array.ndim == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            elif image_array.shape[-1] == 4:
                image_array = image_array[:, :, :3]
            
            # HWC to CHW
            image_tensor = np.transpose(image_array, (2, 0, 1))
            self.writer.add_image(tag, image_tensor, step)
            
            plt.close(fig)
            buf.close()
        except Exception as e:
            pass  # Silently fail if plotting fails
    
    def log_adaptive_weights(self, step, weights_dict):
        """
        Log adaptive constraint weights.
        
        Args:
            step: Training step
            weights_dict: Dictionary with weight values (k_g, k_Sl, k_theta, k_d)
        """
        if not self.enabled:
            return
        
        for key, value in weights_dict.items():
            self.log_scalar(f'weights/{key}', value, step)
    
    def log_learning_rate(self, step, lr, name='lr'):
        """Log learning rate."""
        if self.enabled:
            self.log_scalar(f'optim/{name}', lr, step)
    
    def log_gradient_norm(self, step, grad_norm, name='grad_norm'):
        """Log gradient norm."""
        if self.enabled:
            self.log_scalar(f'optim/{name}', grad_norm, step)
    
    def log_curriculum_stage(self, step, stage, lambda_cost, lambda_carbon):
        """
        Log curriculum learning stage information.
        
        Args:
            step: Training step
            stage: Current curriculum stage
            lambda_cost, lambda_carbon: Current preference weights
        """
        if not self.enabled:
            return
        
        self.log_scalar('curriculum/stage', stage, step)
        self.log_scalar('curriculum/lambda_cost', lambda_cost, step)
        self.log_scalar('curriculum/lambda_carbon', lambda_carbon, step)
    
    def log_histogram(self, tag, values, step):
        """Log histogram of values."""
        if self.enabled and self.writer is not None:
            if isinstance(values, np.ndarray):
                self.writer.add_histogram(tag, values, step)
            elif torch.is_tensor(values):
                self.writer.add_histogram(tag, values.detach().cpu().numpy(), step)
    
    def log_model_weights(self, step, model, prefix='model'):
        """Log model weight histograms."""
        if not self.enabled:
            return
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.log_histogram(f'{prefix}/{name}_weight', param.data, step)
                self.log_histogram(f'{prefix}/{name}_grad', param.grad, step)
    
    def log_pareto_validation(self, step, val_result, prefix='val'):
        """
        Log Pareto front validation metrics.
        
        Args:
            step: Training step (epoch)
            val_result: Dictionary from validate_pareto_front_unified containing:
                - hypervolume: Hypervolume indicator (larger is better)
                - feasible_ratio: Ratio of feasible solutions
                - mean_load_dev: Mean load deviation
                - mean_cost: Mean economic cost
                - mean_carbon: Mean carbon emission
                - n_feasible: Number of feasible preference points
                - n_total: Total number of preference points
                - validation_metric: Combined metric for early stopping
            prefix: 'val' or 'train'
        """
        if not self.enabled:
            return
        
        # Main Pareto quality metrics
        self.log_scalar(f'{prefix}/pareto_hypervolume', val_result['hypervolume'], step)
        self.log_scalar(f'{prefix}/pareto_feasible_ratio', val_result['feasible_ratio'], step)
        self.log_scalar(f'{prefix}/pareto_validation_metric', val_result['validation_metric'], step)
        
        # Feasibility counts
        if 'n_feasible' in val_result and 'n_total' in val_result:
            self.log_scalar(f'{prefix}/pareto_n_feasible', val_result['n_feasible'], step)
            self.log_scalar(f'{prefix}/pareto_n_total', val_result['n_total'], step)
        
        # Objective statistics
        if 'mean_cost' in val_result:
            self.log_scalar(f'{prefix}/pareto_mean_cost', val_result['mean_cost'], step)
        if 'mean_carbon' in val_result:
            self.log_scalar(f'{prefix}/pareto_mean_carbon', val_result['mean_carbon'], step)
        
        # Constraint violation
        if 'mean_load_dev' in val_result:
            self.log_scalar(f'{prefix}/pareto_mean_load_dev', val_result['mean_load_dev'], step)
        
        # Combined view for comparison
        self.log_scalars(f'{prefix}/pareto_quality', {
            'hypervolume': val_result['hypervolume'],
            'feasible_ratio': val_result['feasible_ratio'],
            'metric': val_result['validation_metric']
        }, step)
    
    def log_pareto_front_image(self, step, costs, carbons, feasible_mask=None, 
                                ref_point=None, tag='pareto_front'):
        """
        Log Pareto front as an image with cost vs carbon scatter plot.
        
        Args:
            step: Training step
            costs: Array of cost values for each preference point
            carbons: Array of carbon values for each preference point
            feasible_mask: Boolean array indicating feasible points (optional)
            ref_point: Reference point [cost_ref, carbon_ref] for hypervolume (optional)
            tag: Tag for the image
        """
        if not self.enabled:
            return
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import io
            from PIL import Image
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            costs = np.array(costs)
            carbons = np.array(carbons)
            
            if feasible_mask is not None:
                feasible_mask = np.array(feasible_mask)
                # Plot feasible and infeasible points differently
                ax.scatter(costs[feasible_mask], carbons[feasible_mask], 
                          c='green', s=100, marker='o', label='Feasible', alpha=0.8)
                ax.scatter(costs[~feasible_mask], carbons[~feasible_mask], 
                          c='red', s=100, marker='x', label='Infeasible', alpha=0.8)
            else:
                ax.scatter(costs, carbons, c='blue', s=100, marker='o', alpha=0.8)
            
            # Plot reference point if provided
            if ref_point is not None:
                ax.scatter([ref_point[0]], [ref_point[1]], c='orange', s=200, 
                          marker='*', label='Reference Point', zorder=5)
                # Draw dashed lines from ref point
                ax.axhline(y=ref_point[1], color='orange', linestyle='--', alpha=0.3)
                ax.axvline(x=ref_point[0], color='orange', linestyle='--', alpha=0.3)
            
            # Add preference labels (approximate)
            n_points = len(costs)
            for i, (c, e) in enumerate(zip(costs, carbons)):
                lc = 1.0 - i * 0.1 if n_points == 10 else 1.0 - i / max(n_points-1, 1)
                ax.annotate(f'λ={lc:.1f}', (c, e), textcoords="offset points", 
                           xytext=(5, 5), fontsize=8, alpha=0.7)
            
            ax.set_xlabel('Economic Cost ($/h)', fontsize=12)
            ax.set_ylabel('Carbon Emission (tCO2/h)', fontsize=12)
            ax.set_title(f'Pareto Front Approximation (Epoch {step})', fontsize=14)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Convert to image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            image = Image.open(buf)
            
            # Convert to tensor format for TensorBoard
            image_array = np.array(image)
            if image_array.ndim == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            elif image_array.shape[-1] == 4:
                image_array = image_array[:, :, :3]
            
            # HWC to CHW
            image_tensor = np.transpose(image_array, (2, 0, 1))
            self.writer.add_image(tag, image_tensor, step)
            
            plt.close(fig)
            buf.close()
        except Exception as e:
            pass  # Silently fail if plotting fails
    
    def log_preference_performance(self, step, preference_results, prefix='val'):
        """
        Log performance metrics for each preference point.
        
        Args:
            step: Training step
            preference_results: List of dicts, each containing metrics for a preference point:
                [{'lambda_cost': 1.0, 'cost': ..., 'carbon': ..., 'load_dev': ..., 'feasible': ...}, ...]
            prefix: 'val' or 'train'
        """
        if not self.enabled or not preference_results:
            return
        
        for i, result in enumerate(preference_results):
            lc = result.get('lambda_cost', 1.0 - i * 0.1)
            pref_tag = f'{prefix}/pref_{lc:.1f}'
            
            if 'cost' in result:
                self.log_scalar(f'{pref_tag}_cost', result['cost'], step)
            if 'carbon' in result:
                self.log_scalar(f'{pref_tag}_carbon', result['carbon'], step)
            if 'load_dev' in result:
                self.log_scalar(f'{pref_tag}_load_dev', result['load_dev'], step)
    
    def flush(self):
        """Flush the writer."""
        if self.enabled and self.writer is not None:
            self.writer.flush()
    
    def close(self):
        """Close the writer."""
        if self.enabled and self.writer is not None:
            self.writer.close()
            print("[TensorBoard] Logger closed.")
        
        # Close log file
        if self.log_file is not None:
            try:
                import datetime
                self.log_file.write("\n" + "=" * 80 + "\n")
                self.log_file.write(f"Ended: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.log_file.close()
                print("[TensorBoard] Metrics log file closed.")
            except Exception:
                pass


def create_tensorboard_logger(config, model_mode='unified', pref_sampling='curriculum', 
                               lambda_cost=None, lambda_carbon=None):
    """
    Factory function to create TensorBoard logger with appropriate naming.
    
    Args:
        config: Configuration object
        model_mode: 'unified' or 'independent'
        pref_sampling: Preference sampling strategy
        lambda_cost, lambda_carbon: Target preference (for independent mode)
        
    Returns:
        TensorBoardLogger instance
    """
    import os
    
    # Create log directory
    log_base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'runs')
    os.makedirs(log_base, exist_ok=True)
    
    # Create descriptive comment
    if model_mode == 'unified':
        comment = f"unified_{pref_sampling}_Nbus{config.Nbus}"
    else:
        if lambda_cost is not None:
            pref_str = f"{lambda_cost}_{lambda_carbon}".replace(".", "")
            comment = f"independent_pref{pref_str}_Nbus{config.Nbus}"
        else:
            comment = f"independent_Nbus{config.Nbus}"
    
    return TensorBoardLogger(log_dir=log_base, comment=comment)


def compute_trajectory_metrics(loss_fn, anchor_vm, anchor_va, final_vm, final_va, 
                                Pd, Qd, lambda_cost, lambda_carbon):
    """
    Compute metrics for solution trajectory logging.
    
    Args:
        loss_fn: MultiObjectiveOPFLoss instance
        anchor_vm, anchor_va: VAE anchor outputs
        final_vm, final_va: Flow final outputs  
        Pd, Qd: Load demands
        lambda_cost, lambda_carbon: Target preference
        
    Returns:
        dict with anchor and final objective values
    """
    import torch
    
    with torch.no_grad():
        # Compute anchor objectives (VAE output, preference [1,0])
        _, anchor_dict = loss_fn(
            anchor_vm, anchor_va, Pd, Qd,
            lambda_cost=1.0, lambda_carbon=0.0,
            return_details=True
        )
        
        # Compute final objectives (Flow output, target preference)
        _, final_dict = loss_fn(
            final_vm, final_va, Pd, Qd,
            lambda_cost=lambda_cost, lambda_carbon=lambda_carbon,
            return_details=True
        )
    
    return {
        'anchor_cost': anchor_dict['cost'],
        'anchor_carbon': anchor_dict['carbon'],
        'final_cost': final_dict['cost'],
        'final_carbon': final_dict['carbon'],
        'anchor_load_dev': anchor_dict['load_dev'],
        'final_load_dev': final_dict['load_dev'],
        'anchor_gen_vio': anchor_dict['gen_vio'],
        'final_gen_vio': final_dict['gen_vio'],
    }


if __name__ == "__main__":
    print("=" * 60)
    print("DeepOPF-V Utility Functions Module")
    print("=" * 60)
    print("Contains evaluation metrics, power system calculations, and post-processing functions")
    print("\nTensorBoard logging utilities available:")
    print("  - TensorBoardLogger: Main logging class")
    print("  - create_tensorboard_logger: Factory function")
    print("  - compute_trajectory_metrics: Compute solution trajectory")
    
    # ==================== Toy Example: Hypervolume Calculation ====================
    print("\n" + "=" * 60)
    print("Testing Hypervolume Calculation")
    print("=" * 60)
    
    # Simple 2D example: 3 Pareto-optimal points
    # Minimization problem: lower cost and lower carbon is better
    #
    #  Carbon
    #    5 |------------+  ref_point
    #    4 | A          |
    #    3 |   \        |
    #    2 |    B       |
    #    1 |      \  C  |
    #    0 +------------+---> Cost
    #      0  1  2  3  4  5
    #
    # Point A: (1, 4) - low cost, high carbon
    # Point B: (2, 2) - balanced
    # Point C: (4, 1) - high cost, low carbon
    
    test_points = np.array([
        [1.0, 4.0],  # Point A
        [2.0, 2.0],  # Point B  
        [4.0, 1.0],  # Point C
    ])
    test_ref_point = np.array([5.0, 5.0])
    
    print(f"\nPareto points:")
    for i, (c, e) in enumerate(test_points):
        print(f"  Point {chr(65+i)}: cost={c}, carbon={e}")
    print(f"Reference point: cost={test_ref_point[0]}, carbon={test_ref_point[1]}")
    
    # Calculate hypervolume
    hv = compute_hypervolume(test_points, test_ref_point)
    
    # Manual calculation for verification:
    # The hypervolume is the area dominated by the Pareto front
    # 
    # Decompose into rectangles (from left to right):
    # 1. From A (1,4) to B (2,2): width=1, height=(5-4)=1, area=1
    # 2. From B (2,2) to C (4,1): width=2, height=(5-2)=3, but only (4-2)=2 counts
    #    Actually: width=(4-2)=2, from y=2 to ref_y=5, but B dominates until y=2
    #    Area below B: (4-2)*(5-2) = 6, but subtract overlap
    # 3. From C (4,1) to ref (5,5): width=1, height=(5-1)=4, area=4
    #
    # Using the standard HV formula:
    # Sort by x: A(1,4), B(2,2), C(4,1)
    # HV = (2-1)*(5-4) + (4-2)*(5-2) + (5-4)*(5-1)
    #    = 1*1 + 2*3 + 1*4 = 1 + 6 + 4 = 11
    
    expected_hv = 11.0
    print(f"\nComputed Hypervolume: {hv:.4f}")
    print(f"Expected Hypervolume: {expected_hv:.4f}")
    print(f"Match: {'PASS' if abs(hv - expected_hv) < 0.1 else 'FAIL'}")
    
    # Test with infeasible points
    print("\n--- Testing with feasibility filter ---")
    all_costs = np.array([1.0, 2.0, 4.0, 3.0, 2.5])    # 5 solutions
    all_carbons = np.array([4.0, 2.0, 1.0, 3.5, 2.8])
    feasible = np.array([True, True, True, False, False])  # Last 2 are infeasible
    
    result = evaluate_pareto_front(all_costs, all_carbons, feasible, test_ref_point)
    
    print(f"Total solutions: {result['n_total']}")
    print(f"Feasible solutions: {result['n_feasible']}")
    print(f"Feasible ratio: {result['feasible_ratio']:.2%}")
    print(f"Hypervolume (feasible only): {result['hypervolume']:.4f}")
    
    # Test combined metric
    combined = get_pareto_validation_metric(
        result['hypervolume'], 
        result['feasible_ratio'],
        hv_scale=20.0  # Scale HV to [0, ~1] range
    )
    print(f"Combined validation metric: {combined:.4f}")
    
    # Test GPU functions (non-destructive)
    print("\n--- Testing GPU Utilities ---")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_memory_cleanup()
        print("GPU memory cleanup: OK")
        temp = check_gpu_temperature(warning_temp=90, critical_temp=95)  # High thresholds for test
        if temp is not None:
            print(f"GPU temperature: {temp}°C")
    
    # Test preference sampling
    print("\n--- Testing Preference Sampling ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Uniform sampling
    prefs_uniform = sample_preferences_uniform(5, device)
    print(f"Uniform preferences (5 samples):")
    for i, p in enumerate(prefs_uniform):
        print(f"  [{p[0].item():.3f}, {p[1].item():.3f}]")
    
    # Curriculum sampling
    prefs_curriculum = sample_preferences_curriculum(5, device, current_max_carbon_weight=0.3)
    print(f"Curriculum preferences (max_carbon=0.3):")
    for i, p in enumerate(prefs_curriculum):
        print(f"  [{p[0].item():.3f}, {p[1].item():.3f}]")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

def plot_training_curves(lossvm, lossva):
    """
    Plot training loss curves
    
    Args:
        lossvm: Vm training losses
        lossva: Va training losses
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(lossvm)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Vm Training Loss')
    ax1.grid(True)
    
    ax2.plot(lossva)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Va Training Loss')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print('\nTraining curves saved to: training_curves.png')
    plt.close()


def plot_unsupervised_training_curves(loss_history):
    """
    Plot unsupervised training loss curves with multiple components.
    
    Args:
        loss_history: Dictionary containing loss history for each component
    """
    # Check which keys are available (different for NGT vs old unsupervised)
    has_ngt_keys = 'kgenp_mean' in loss_history
    
    # Check if multi-objective data is present
    has_multi_objective = ('cost' in loss_history and 'carbon' in loss_history and 
                          len(loss_history.get('cost', [])) > 0 and
                          any(v > 0 for v in loss_history.get('carbon', [0])))
    
    if has_ngt_keys:
        # DeepOPF-NGT format
        if has_multi_objective:
            # Extended layout for multi-objective: 3x3 grid
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        else:
            # Original layout: 2x3 grid
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Total loss
        axes[0, 0].plot(loss_history['total'])
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].grid(True)
        
        # Generator P weight
        axes[0, 1].plot(loss_history.get('kgenp_mean', []))
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Weight')
        axes[0, 1].set_title('Generator P Weight (k_genp)')
        axes[0, 1].grid(True)
        
        # Generator Q weight
        axes[0, 2].plot(loss_history.get('kgenq_mean', []))
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Weight')
        axes[0, 2].set_title('Generator Q Weight (k_genq)')
        axes[0, 2].grid(True)
        
        # Load P weight
        axes[1, 0].plot(loss_history.get('kpd_mean', []))
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Weight')
        axes[1, 0].set_title('Load P Weight (k_pd)')
        axes[1, 0].grid(True)
        
        # Load Q weight
        axes[1, 1].plot(loss_history.get('kqd_mean', []))
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Weight')
        axes[1, 1].set_title('Load Q Weight (k_qd)')
        axes[1, 1].grid(True)
        
        # Voltage weight
        axes[1, 2].plot(loss_history.get('kv_mean', []))
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Weight')
        axes[1, 2].set_title('Voltage Weight (k_v)')
        axes[1, 2].grid(True)
        
        # Multi-objective plots (row 3)
        if has_multi_objective:
            # Economic cost
            axes[2, 0].plot(loss_history.get('cost', []), 'b-', label='Economic Cost')
            axes[2, 0].set_xlabel('Epoch')
            axes[2, 0].set_ylabel('Cost')
            axes[2, 0].set_title('Economic Cost (L_cost)')
            axes[2, 0].grid(True)
            
            # Carbon emission
            axes[2, 1].plot(loss_history.get('carbon', []), 'g-', label='Carbon Emission')
            axes[2, 1].set_xlabel('Epoch')
            axes[2, 1].set_ylabel('Carbon (tCO2)')
            axes[2, 1].set_title('Carbon Emission (L_carbon)')
            axes[2, 1].grid(True)
            
            # Combined objectives (normalized for visualization)
            cost_data = np.array(loss_history.get('cost', []))
            carbon_data = np.array(loss_history.get('carbon', []))
            if len(cost_data) > 0 and len(carbon_data) > 0:
                # Normalize for comparison
                cost_norm = cost_data / (cost_data.max() + 1e-8)
                carbon_norm = carbon_data / (carbon_data.max() + 1e-8)
                axes[2, 2].plot(cost_norm, 'b-', label='Cost (norm)')
                axes[2, 2].plot(carbon_norm, 'g-', label='Carbon (norm)')
                axes[2, 2].set_xlabel('Epoch')
                axes[2, 2].set_ylabel('Normalized Value')
                axes[2, 2].set_title('Multi-Objective Trade-off')
                axes[2, 2].legend()
                axes[2, 2].grid(True)
    else:
        # Old unsupervised format
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Total loss
        axes[0, 0].plot(loss_history.get('total', []))
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].grid(True)
        
        # Cost loss (L_obj)
        axes[0, 1].plot(loss_history.get('cost', []))
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Generation Cost (L_obj)')
        axes[0, 1].grid(True)
        
        # Generator violation loss (L_g)
        axes[0, 2].plot(loss_history.get('gen_vio', []))
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_title('Generator Violation (L_g)')
        axes[0, 2].grid(True)
        
        # Branch power flow violation (L_Sl)
        axes[1, 0].plot(loss_history.get('branch_pf_vio', []))
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Branch Power Flow Violation (L_Sl)')
        axes[1, 0].grid(True)
        
        # Branch angle violation (L_theta)
        axes[1, 1].plot(loss_history.get('branch_ang_vio', []))
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Branch Angle Violation (L_theta)')
        axes[1, 1].grid(True)
        
        # Load deviation loss (L_d)
        axes[1, 2].plot(loss_history.get('load_dev', []))
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].set_title('Load Deviation (L_d)')
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('unsupervised_training_curves.png', dpi=300, bbox_inches='tight')
    print('\nUnsupervised training curves saved to: unsupervised_training_curves.png')
    plt.close()


def save_results(config, results, lossvm, lossva):
    """
    Save training and evaluation results to JSON and CSV files (more readable than .mat)
    
    Args:
        config: Configuration object
        results: Evaluation results dictionary
        lossvm: Vm training losses
        lossva: Va training losses
    """
    import json
    import csv
    
    # Extract timing info
    timing_info = results.get('timing_info', {})
    
    # ==================== 1. Save main metrics to JSON (human-readable) ====================
    metrics_summary = {
        'config': {
            'model_type': getattr(config, 'model_type', 'simple'),
            'Nbus': config.Nbus,
            'Ntrain': config.Ntrain,
            'Ntest': config.Ntest,
            'EpochVm': config.EpochVm,
            'EpochVa': config.EpochVa,
            'batch_size': config.batch_size_training,
            'learning_rate_Vm': config.Lrm,
            'learning_rate_Va': config.Lra,
        },
        'before_post_processing': {
            'Vm_MAE': float(results['mae_Vmtest'].item()) if hasattr(results['mae_Vmtest'], 'item') else float(results['mae_Vmtest']),
            'Va_MAE': float(results['mae_Vatest'].item()) if hasattr(results['mae_Vatest'], 'item') else float(results['mae_Vatest']),
            'cost_error_percent': float(torch.mean(results['mre_cost']).item()),
            'Pd_error_percent': float(torch.mean(results['mre_Pd']).item()),
            'Qd_error_percent': float(torch.mean(results['mre_Qd']).item()),
            'Pg_satisfy_rate': float(torch.mean(results['vio_PQg'][:, 0]).item()),
            'Qg_satisfy_rate': float(torch.mean(results['vio_PQg'][:, 1]).item()),
            'branch_angle_satisfy_rate': float(torch.mean(results['vio_branang']).item()),
            'branch_power_satisfy_rate': float(torch.mean(results['vio_branpf']).item()),
        },
        'after_post_processing': {
            'Vm_MAE': float(results['mae_Vmtest1'].item()) if hasattr(results['mae_Vmtest1'], 'item') else float(results['mae_Vmtest1']),
            'Va_MAE': float(results['mae_Vatest1'].item()) if hasattr(results['mae_Vatest1'], 'item') else float(results['mae_Vatest1']),
            'cost_error_percent': float(torch.mean(results['mre_cost1']).item()),
            'Pg_satisfy_rate': float(torch.mean(results['vio_PQg1'][:, 0]).item()),
            'Qg_satisfy_rate': float(torch.mean(results['vio_PQg1'][:, 1]).item()),
            'branch_angle_satisfy_rate': float(torch.mean(results['vio_branang1']).item()),
            'branch_power_satisfy_rate': float(torch.mean(results['vio_branpf1']).item()),
        },
        'timing': {
            'Vm_prediction_sec': timing_info.get('time_Vm_prediction', 0),
            'Va_prediction_sec': timing_info.get('time_Va_prediction', 0),
            'NN_total_sec': timing_info.get('time_NN_total', 0),
            'PQ_calculation_sec': timing_info.get('time_PQ_calculation', 0),
            'post_processing_sec': timing_info.get('time_post_processing', 0),
            'total_with_post_sec': timing_info.get('time_total_with_post', 0),
            'NN_per_sample_ms': timing_info.get('time_NN_per_sample_ms', 0),
            'total_per_sample_ms': timing_info.get('time_total_per_sample_ms', 0),
            'num_test_samples': timing_info.get('num_test_samples', config.Ntest),
        }
    }
    
    # Save JSON
    json_path = config.resultnm.replace('.mat', '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
    print(f'\nMetrics saved to: {json_path}')
    
    # ==================== 2. Save training loss to CSV ====================
    csv_loss_path = config.resultnm.replace('.mat', '_loss.csv')
    with open(csv_loss_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss_vm', 'loss_va'])
        max_epochs = max(len(lossvm), len(lossva))
        for i in range(max_epochs):
            loss_vm_val = lossvm[i] if i < len(lossvm) else ''
            loss_va_val = lossva[i] if i < len(lossva) else ''
            writer.writerow([i + 1, loss_vm_val, loss_va_val])
    print(f'Training loss saved to: {csv_loss_path}')
    
    # ==================== 3. Save summary comparison table to CSV ====================
    csv_summary_path = config.resultnm.replace('.mat', '_summary.csv')
    with open(csv_summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Before Post-Processing', 'After Post-Processing'])
        writer.writerow(['Vm MAE (p.u.)', 
                        f"{metrics_summary['before_post_processing']['Vm_MAE']:.6f}",
                        f"{metrics_summary['after_post_processing']['Vm_MAE']:.6f}"])
        writer.writerow(['Va MAE (rad)', 
                        f"{metrics_summary['before_post_processing']['Va_MAE']:.6f}",
                        f"{metrics_summary['after_post_processing']['Va_MAE']:.6f}"])
        writer.writerow(['Cost Error (%)', 
                        f"{metrics_summary['before_post_processing']['cost_error_percent']:.2f}",
                        f"{metrics_summary['after_post_processing']['cost_error_percent']:.2f}"])
        writer.writerow(['Pg Satisfy Rate (%)', 
                        f"{metrics_summary['before_post_processing']['Pg_satisfy_rate']:.2f}",
                        f"{metrics_summary['after_post_processing']['Pg_satisfy_rate']:.2f}"])
        writer.writerow(['Qg Satisfy Rate (%)', 
                        f"{metrics_summary['before_post_processing']['Qg_satisfy_rate']:.2f}",
                        f"{metrics_summary['after_post_processing']['Qg_satisfy_rate']:.2f}"])
        writer.writerow(['Branch Angle Satisfy Rate (%)', 
                        f"{metrics_summary['before_post_processing']['branch_angle_satisfy_rate']:.2f}",
                        f"{metrics_summary['after_post_processing']['branch_angle_satisfy_rate']:.2f}"])
        writer.writerow(['Branch Power Satisfy Rate (%)', 
                        f"{metrics_summary['before_post_processing']['branch_power_satisfy_rate']:.2f}",
                        f"{metrics_summary['after_post_processing']['branch_power_satisfy_rate']:.2f}"])
    print(f'Summary table saved to: {csv_summary_path}')
    
    # ==================== 4. Also save to .npz for programmatic access ====================
    npz_path = config.resultnm.replace('.mat', '.npz')
    np.savez(npz_path,
        # Summary arrays
        resvio=np.array([
            [float(torch.mean(results['mre_cost'])), float(torch.mean(results['mre_Pd'])), 
             float(torch.mean(results['mre_Qd'])), float(torch.mean(results['vio_PQg'][:, 0])),
             float(torch.mean(results['vio_PQg'][:, 1])), float(torch.mean(results['vio_branang'])),
             float(torch.mean(results['vio_branpf']))],
            [float(torch.mean(results['mre_cost1'])), float(torch.mean(results['mre_Pd'])),
             float(torch.mean(results['mre_Qd'])), float(torch.mean(results['vio_PQg1'][:, 0])),
             float(torch.mean(results['vio_PQg1'][:, 1])), float(torch.mean(results['vio_branang1'])),
             float(torch.mean(results['vio_branpf1']))]
        ]),
        maeV=np.array([
            [float(results['mae_Vmtest'].item() if hasattr(results['mae_Vmtest'], 'item') else results['mae_Vmtest']),
             float(results['mae_Vatest'].item() if hasattr(results['mae_Vatest'], 'item') else results['mae_Vatest'])],
            [float(results['mae_Vmtest1'].item() if hasattr(results['mae_Vmtest1'], 'item') else results['mae_Vmtest1']),
             float(results['mae_Vatest1'].item() if hasattr(results['mae_Vatest1'], 'item') else results['mae_Vatest1'])]
        ]),
        lossvm=np.array(lossvm),
        lossva=np.array(lossva),
        mre_cost=np.array(results['mre_cost']),
        mre_cost1=np.array(results['mre_cost1']),
    )
    print(f'NumPy data saved to: {npz_path}')
    
    # Print timing summary
    if timing_info:
        print(f'\nTiming Summary for {timing_info.get("model_type", getattr(config, "model_type", "model"))}:')
        print(f'  NN inference per sample: {timing_info.get("time_NN_per_sample_ms", 0):.4f} ms')
        print(f'  Total solving per sample: {timing_info.get("time_total_per_sample_ms", 0):.4f} ms')


import math

def get_gci_for_generators(sys_data):
    """
    Assign GCI (Carbon Emission Intensity) values based on marginal generation cost.
    
    Low-cost generators → High carbon (coal)
    High-cost generators → Low carbon (CCGT)
    
    Args:
        config: Configuration object
        sys_data: PowerSystemData object
        
    Returns:
        gci_values: Array of GCI values for each generator [n_gen]
    """
    # GCI Lookup Tables (tCO2/MWh) - from evaluate_multi_objective.py
    FUEL_LOOKUP_CO2 = {
        "ANT": 0.9095,  # Anthracite Coal
        "BIT": 0.8204,  # Bituminous Coal
        "Oil": 0.7001,  # Heavy Oil
        "GAS": 0.5173,  # Natural Gas
        "CCGT": 0.3621,  # Gas Combined Cycle
        "ICE": 0.6030,  # Internal Combustion Engine
        "Thermal": 0.6874,  # Thermal Power (General)
        "NUC": 0.0,     # Nuclear Power
        "RE": 0.0,      # Renewable Energy
        "HYD": 0.0,     # Hydropower
        "N/A": 0.0      # Default case
    }
    n_gen = sys_data.gen.shape[0] if isinstance(sys_data.gen, np.ndarray) else sys_data.gen.numpy().shape[0]
    gencost = sys_data.gencost if isinstance(sys_data.gencost, np.ndarray) else sys_data.gencost.numpy()
    
    gci_values = np.zeros(n_gen)
    
    # Get marginal cost coefficient c1 (column 1 in gencost format [c2, c1, ...])
    c1_values = gencost[:n_gen, 1]
    
    # Compute percentiles for classification
    p25 = np.percentile(c1_values, 25)
    p50 = np.percentile(c1_values, 50)
    p75 = np.percentile(c1_values, 75)
    
    for i in range(n_gen):
        c1 = c1_values[i]
        
        if c1 <= p25:
            # Lowest cost quartile → Coal (highest carbon)
            fuel_type = "BIT" if i % 2 == 0 else "ANT"
        elif c1 <= p50:
            # Second quartile → Heavy Oil
            fuel_type = "Oil"
        elif c1 <= p75:
            # Third quartile → Natural Gas
            fuel_type = "GAS"
        else:
            # Highest cost quartile → CCGT (lowest carbon)
            fuel_type = "CCGT"
        
        gci_values[i] = FUEL_LOOKUP_CO2[fuel_type]
    
    return gci_values 