#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pareto_loss.py

Pareto Consistency Loss for Multi-Objective OPF
===============================================

This module provides losses to enforce Pareto front consistency in the objective space.

The key insight: as lambda (carbon weight) increases:
- Carbon emission should DECREASE (more carbon-focused optimization)
- Economic cost should INCREASE (trade-off for lower carbon)

We enforce this monotonicity along the lambda axis using soft penalties.

Usage:
    from pareto_loss import ParetoLossComputer
    
    pareto_loss_fn = ParetoLossComputer(ctx, n_va, device)
    loss, details = pareto_loss_fn(Y_pred, fine_norm)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass


@dataclass
class ParetoLossConfig:
    """Configuration for Pareto consistency loss."""
    # Weight for Pareto loss
    alpha: float = 0.1
    # Margin for ranking loss (allows small violations)
    margin: float = 0.0
    # Normalization mode: 'none', 'batch', 'running'
    norm_mode: str = 'batch'
    # Separate weights for cost and carbon monotonicity
    w_cost_mono: float = 1.0
    w_carbon_mono: float = 1.0
    # Compute every N batches (1 = every batch, 5~10 recommended for efficiency)
    # Full mode involves power flow computation which is expensive
    compute_freq: int = 5
    # Use soft margin (smooth) vs hard margin (hinge)
    use_soft_margin: bool = True
    # Temperature for soft margin
    temperature: float = 1.0


class ParetoLossComputer:
    """
    Computes Pareto consistency loss by:
    1. Reconstructing full Vm/Va from partial NGT format
    2. Computing power flow to get Pg
    3. Computing cost and carbon objectives
    4. Enforcing monotonicity along lambda axis
    """
    
    def __init__(
        self,
        ctx,                          # EvalContext or similar with system data
        n_va: int,                    # Number of Va dimensions in Y
        device: torch.device,
        config: ParetoLossConfig = None,
    ):
        self.ctx = ctx
        self.n_va = n_va
        self.device = device
        self.config = config or ParetoLossConfig()
        
        # Cache system data as tensors
        self._setup_system_tensors()
        
        # Running statistics for normalization
        self.cost_ema = None
        self.carbon_ema = None
        self.ema_momentum = 0.1
        
        # Counter for compute frequency
        self._call_count = 0
    
    def _setup_system_tensors(self):
        """Cache system data as torch tensors for efficient computation."""
        ctx = self.ctx
        sys_data = ctx.sys_data if hasattr(ctx, 'sys_data') else ctx
        
        # Ybus for power flow
        self.Ybus = torch.from_numpy(ctx.Ybus.toarray()).to(
            device=self.device, dtype=torch.complex128
        )
        
        # Generator cost coefficients
        if hasattr(ctx, 'gencost_Pg') and ctx.gencost_Pg is not None:
            self.gencost_c2 = torch.from_numpy(ctx.gencost_Pg[:, 0]).float().to(self.device)
            self.gencost_c1 = torch.from_numpy(ctx.gencost_Pg[:, 1]).float().to(self.device)
        else:
            # Fallback to extracting from gencost
            gencost = ctx.gencost if hasattr(ctx, 'gencost') else sys_data.gencost
            if hasattr(gencost, 'numpy'):
                gencost = gencost.numpy()
            idxPg = ctx.idxPg if hasattr(ctx, 'idxPg') else sys_data.idxPg
            if gencost.shape[1] > 4:
                self.gencost_c2 = torch.from_numpy(gencost[idxPg, 4]).float().to(self.device)
                self.gencost_c1 = torch.from_numpy(gencost[idxPg, 5]).float().to(self.device)
            else:
                self.gencost_c2 = torch.from_numpy(gencost[idxPg, 0]).float().to(self.device)
                self.gencost_c1 = torch.from_numpy(gencost[idxPg, 1]).float().to(self.device)
        
        # GCI values for carbon emission
        if hasattr(ctx, 'gci_values') and ctx.gci_values is not None:
            self.gci_values = torch.from_numpy(ctx.gci_values).float().to(self.device)
        else:
            # Need to compute GCI
            from utils import get_gci_for_generators
            gci_all = get_gci_for_generators(sys_data)
            idxPg = ctx.idxPg if hasattr(ctx, 'idxPg') else sys_data.idxPg
            self.gci_values = torch.from_numpy(gci_all[idxPg]).float().to(self.device)
        
        # Bus indices for generators
        self.bus_Pg = torch.from_numpy(
            np.asarray(ctx.bus_Pg if hasattr(ctx, 'bus_Pg') else sys_data.bus_Pg)
        ).long().to(self.device)
        
        self.baseMVA = float(ctx.baseMVA if hasattr(ctx, 'baseMVA') else sys_data.baseMVA)
        self.Nbus = int(ctx.Nbus if hasattr(ctx, 'Nbus') else sys_data.Nbus)
        
        # ==================== Load bus indices (CRITICAL for Pg calculation) ====================
        # x_batch format: [Pd_nonzero, Qd_nonzero] in p.u.
        # We need bus_Pd to reconstruct full Pd from x_batch
        if hasattr(ctx, 'bus_Pd'):
            self.bus_Pd = torch.from_numpy(np.asarray(ctx.bus_Pd).astype(int)).long().to(self.device)
        elif hasattr(sys_data, 'bus_Pd'):
            bus_Pd = sys_data.bus_Pd
            if hasattr(bus_Pd, 'numpy'):
                bus_Pd = bus_Pd.numpy()
            self.bus_Pd = torch.from_numpy(np.asarray(bus_Pd).astype(int)).long().to(self.device)
        else:
            # Fallback: try to get from load_idx or similar
            self.bus_Pd = None
            print("[ParetoLoss] WARNING: bus_Pd not found, Pd reconstruction will be disabled!")
        
        self.num_Pd = len(self.bus_Pd) if self.bus_Pd is not None else 0
        
        # Reconstruction indices
        self.bus_slack = int(ctx.bus_slack if hasattr(ctx, 'bus_slack') else ctx.sys_data.bus_slack)
        
        if hasattr(ctx, 'bus_Pnet_all'):
            self.bus_Pnet_all = torch.from_numpy(
                np.asarray(ctx.bus_Pnet_all).astype(int)
            ).long().to(self.device)
        else:
            self.bus_Pnet_all = None
        
        if hasattr(ctx, 'bus_Pnet_noslack_all'):
            self.bus_Pnet_noslack_all = torch.from_numpy(
                np.asarray(ctx.bus_Pnet_noslack_all).astype(int)
            ).long().to(self.device)
        else:
            self.bus_Pnet_noslack_all = None
        
        # Kron reconstruction matrix
        if hasattr(ctx, 'param_ZIMV') and ctx.param_ZIMV is not None:
            self.param_ZIMV = torch.from_numpy(np.asarray(ctx.param_ZIMV)).to(
                device=self.device, dtype=torch.complex128
            )
            self.bus_ZIB_all = torch.from_numpy(
                np.asarray(ctx.bus_ZIB_all).astype(int)
            ).long().to(self.device) if hasattr(ctx, 'bus_ZIB_all') else None
        else:
            self.param_ZIMV = None
            self.bus_ZIB_all = None
    
    def reconstruct_full_voltage(
        self,
        Y: torch.Tensor,           # [B, D] or [B, K, D]
        x_batch: torch.Tensor = None,  # [B, input_dim] load data (for Pd/Qd)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct full Vm and Va from partial NGT format.
        
        NOTE: This function avoids inplace operations to preserve gradient computation.
        
        Args:
            Y: Partial voltage in NGT format [Va_noslack, Vm_nonZIB]
            x_batch: Load data (optional, for accurate Pd/Qd)
        
        Returns:
            Vm_full: [B, Nbus] or [B, K, Nbus]
            Va_full: [B, Nbus] or [B, K, Nbus]
        """
        squeeze_output = False
        if Y.dim() == 2:
            Y = Y.unsqueeze(1)  # [B, 1, D]
            squeeze_output = True
        
        B, K, D = Y.shape
        NPred_Va = self.n_va
        NPred_Vm = D - NPred_Va
        
        # Split Va (no slack) and Vm (non-ZIB)
        Va_noslack = Y[..., :NPred_Va]       # [B, K, NPred_Va]
        Vm_nonZIB = Y[..., NPred_Va:]        # [B, K, NPred_Vm]
        
        # ============ Non-inplace reconstruction for Va ============
        # Use scatter to avoid inplace operations that break autograd
        if self.bus_Pnet_noslack_all is not None:
            # Create index tensor for scatter: [B, K, NPred_Va]
            idx_va = self.bus_Pnet_noslack_all.view(1, 1, -1).expand(B, K, -1)
            Va_full = torch.zeros(B, K, self.Nbus, device=self.device, dtype=Y.dtype)
            Va_full = Va_full.scatter(2, idx_va, Va_noslack)
            # Slack bus stays 0 (already initialized)
        else:
            # Fallback: assume full bus format with slack removed
            # Use cat instead of inplace assignment
            zeros_slack = torch.zeros(B, K, 1, device=self.device, dtype=Y.dtype)
            if self.bus_slack == 0:
                Va_full = torch.cat([zeros_slack, Va_noslack], dim=2)
            elif self.bus_slack == self.Nbus - 1:
                Va_full = torch.cat([Va_noslack, zeros_slack], dim=2)
            else:
                Va_full = torch.cat([
                    Va_noslack[:, :, :self.bus_slack],
                    zeros_slack,
                    Va_noslack[:, :, self.bus_slack:]
                ], dim=2)
        
        # ============ Non-inplace reconstruction for Vm ============
        if self.bus_Pnet_all is not None:
            idx_vm = self.bus_Pnet_all.view(1, 1, -1).expand(B, K, -1)
            Vm_full = torch.zeros(B, K, self.Nbus, device=self.device, dtype=Y.dtype)
            Vm_full = Vm_full.scatter(2, idx_vm, Vm_nonZIB)
        else:
            Vm_full = Vm_nonZIB
        
        # ============ Kron reconstruction for ZIB nodes (non-inplace) ============
        if self.param_ZIMV is not None and self.bus_ZIB_all is not None:
            V_nonZIB = Vm_full * torch.exp(1j * Va_full.to(torch.float64))
            
            # Compute ZIB node voltages
            Vx = V_nonZIB[:, :, self.bus_Pnet_all]  # [B, K, NPred_Vm]
            # Reshape for batch matmul: [B*K, NPred_Vm]
            Vx_flat = Vx.reshape(B * K, -1)
            Vy_flat = torch.matmul(self.param_ZIMV, Vx_flat.T).T  # [B*K, NZIB]
            Vy = Vy_flat.reshape(B, K, -1)  # [B, K, NZIB]
            
            # Use scatter to update ZIB nodes (non-inplace)
            idx_zib = self.bus_ZIB_all.view(1, 1, -1).expand(B, K, -1)
            Vm_zib = torch.abs(Vy).float()
            Va_zib = torch.angle(Vy).float()
            Vm_full = Vm_full.scatter(2, idx_zib, Vm_zib)
            Va_full = Va_full.scatter(2, idx_zib, Va_zib)
        
        if squeeze_output:
            Vm_full = Vm_full.squeeze(1)
            Va_full = Va_full.squeeze(1)
        
        return Vm_full, Va_full
    
    def compute_power_generation(
        self,
        Vm_full: torch.Tensor,     # [B, Nbus] or [B, K, Nbus]
        Va_full: torch.Tensor,     # [B, Nbus] or [B, K, Nbus]
        Pd: torch.Tensor = None,   # [B, Nbus] load power
        Qd: torch.Tensor = None,   # [B, Nbus] load reactive power
    ) -> torch.Tensor:
        """
        Compute generator active power output from voltage.
        
        NOTE: Vectorized and avoids inplace operations for autograd compatibility.
        
        Returns:
            Pg: [B, n_gen] or [B, K, n_gen] in p.u.
        """
        squeeze_K = False
        if Vm_full.dim() == 2:
            Vm_full = Vm_full.unsqueeze(1)
            Va_full = Va_full.unsqueeze(1)
            squeeze_K = True
        
        B, K, Nbus = Vm_full.shape
        
        # Complex voltage: [B, K, Nbus]
        V = (Vm_full * torch.exp(1j * Va_full.double())).to(torch.complex128)
        
        # Vectorized power flow computation
        # Reshape V for batch matrix multiplication: [B*K, Nbus]
        V_flat = V.reshape(B * K, Nbus)
        
        # I = Ybus @ V, then conj  =>  [B*K, Nbus]
        I_flat = torch.matmul(V_flat, self.Ybus.T).conj()  # Note: V @ Y.T = (Y @ V.T).T
        
        # S = V * conj(I) => P = real(S)
        S_flat = V_flat * I_flat
        P_flat = S_flat.real.float()  # [B*K, Nbus]
        
        # Reshape back: [B, K, Nbus]
        P = P_flat.reshape(B, K, Nbus)
        
        # Extract generator buses: [B, K, n_gen]
        Pg = P[:, :, self.bus_Pg]
        
        # Add load at generator buses (Pg = P_injection + Pd)
        if Pd is not None:
            # Pd is [B, Nbus], expand to [B, 1, Nbus] for broadcasting
            Pd_gen = Pd[:, self.bus_Pg].unsqueeze(1)  # [B, 1, n_gen]
            Pg = Pg + Pd_gen  # Broadcasting over K
        
        if squeeze_K:
            Pg = Pg.squeeze(1)
        
        return Pg
    
    def compute_objectives(
        self,
        Pg: torch.Tensor,          # [B, n_gen] or [B, K, n_gen]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cost and carbon objectives from generator power.
        
        IMPORTANT: Both cost and carbon use clamped Pg (>=0) to avoid:
        - Negative Pg leading to "lower cost is better" becoming "more negative is better"
        - Negative carbon values
        
        Returns:
            cost: [B] or [B, K]
            carbon: [B] or [B, K]
        """
        squeeze_K = False
        if Pg.dim() == 2:
            Pg = Pg.unsqueeze(1)
            squeeze_K = True
        
        B, K, n_gen = Pg.shape
        
        # CRITICAL: Clamp Pg to non-negative values for BOTH cost and carbon
        # Otherwise negative Pg makes cost/carbon calculations meaningless
        # (e.g., cost could become "more negative = better" which is wrong)
        Pg_clamped = torch.clamp(Pg, min=0)
        Pg_MVA = Pg_clamped * self.baseMVA
        
        # Cost = sum(c2 * Pg^2 + c1 * Pg) in physical units (baseMVA)
        cost = torch.sum(
            self.gencost_c2 * (Pg_MVA ** 2) + self.gencost_c1 * Pg_MVA,
            dim=-1
        )  # [B, K]
        
        # Carbon = sum(GCI * Pg) in physical units
        carbon = torch.sum(self.gci_values * Pg_MVA, dim=-1)  # [B, K]
        
        if squeeze_K:
            cost = cost.squeeze(1)
            carbon = carbon.squeeze(1)
        
        return cost, carbon
    
    def compute_monotonicity_loss(
        self,
        cost: torch.Tensor,        # [B, K]
        carbon: torch.Tensor,      # [B, K]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Pareto monotonicity loss.
        
        For increasing lambda (k -> k+1):
        - carbon[k+1] <= carbon[k] (carbon should decrease)
        - cost[k+1] >= cost[k] (cost should increase)
        
        Returns:
            loss: scalar tensor
            details: dict with violation statistics
        """
        cfg = self.config
        
        # Differences along lambda axis
        carbon_diff = carbon[:, 1:] - carbon[:, :-1]  # [B, K-1], positive = violation
        cost_diff = cost[:, 1:] - cost[:, :-1]        # [B, K-1], negative = violation
        
        if cfg.norm_mode == 'batch':
            # Normalize by batch statistics
            cost_scale = cost.std() + 1e-8
            carbon_scale = carbon.std() + 1e-8
            carbon_diff = carbon_diff / carbon_scale
            cost_diff = cost_diff / cost_scale
        elif cfg.norm_mode == 'running':
            # Update and use running statistics
            if self.cost_ema is None:
                self.cost_ema = cost.std().item()
                self.carbon_ema = carbon.std().item()
            else:
                self.cost_ema = (1 - self.ema_momentum) * self.cost_ema + self.ema_momentum * cost.std().item()
                self.carbon_ema = (1 - self.ema_momentum) * self.carbon_ema + self.ema_momentum * carbon.std().item()
            carbon_diff = carbon_diff / (self.carbon_ema + 1e-8)
            cost_diff = cost_diff / (self.cost_ema + 1e-8)
        
        if cfg.use_soft_margin:
            # Soft margin (smooth) - shifted softplus so loss ≈ 0 when no violation
            # softplus(x) - softplus(0) makes the loss start from ~0 at x=0
            # This way, only actual violations (x > 0) produce significant loss
            offset = F.softplus(torch.tensor(cfg.margin), beta=1.0/cfg.temperature).item()
            carbon_violation = F.softplus(carbon_diff + cfg.margin, beta=1.0/cfg.temperature) - offset
            cost_violation = F.softplus(-cost_diff + cfg.margin, beta=1.0/cfg.temperature) - offset
            # Clamp to non-negative (in case of numerical issues)
            carbon_violation = torch.clamp(carbon_violation, min=0)
            cost_violation = torch.clamp(cost_violation, min=0)
        else:
            # Hard margin (hinge)
            carbon_violation = torch.relu(carbon_diff + cfg.margin)
            cost_violation = torch.relu(-cost_diff + cfg.margin)
        
        # Weighted sum
        loss = (cfg.w_carbon_mono * carbon_violation.mean() + 
                cfg.w_cost_mono * cost_violation.mean())
        
        # Statistics
        with torch.no_grad():
            n_carbon_vio = (carbon_diff > 0).sum().item()
            n_cost_vio = (cost_diff < 0).sum().item()
            total_pairs = carbon_diff.numel()
        
        details = {
            'carbon_violation_mean': carbon_violation.mean().item(),
            'cost_violation_mean': cost_violation.mean().item(),
            'carbon_violation_ratio': n_carbon_vio / max(1, total_pairs),
            'cost_violation_ratio': n_cost_vio / max(1, total_pairs),
            'cost_mean': cost.mean().item(),
            'carbon_mean': carbon.mean().item(),
        }
        
        return loss, details
    
    def _reconstruct_Pd_from_x_batch(
        self,
        x_batch: torch.Tensor,  # [B, input_dim] load data in SPARSE format
        B: int,
    ) -> torch.Tensor:
        """
        Reconstruct full Pd tensor from sparse x_batch.
        
        x_batch format: [Pd_nonzero, Qd_nonzero] where Pd_nonzero is at bus_Pd indices.
        Returns: [B, Nbus] full Pd in p.u.
        
        NOTE: Uses scatter for autograd compatibility (no inplace ops).
        """
        if self.bus_Pd is None or self.num_Pd == 0:
            return None
        
        # x_batch[:, :num_Pd] contains Pd at bus_Pd indices (in p.u.)
        Pd_sparse = x_batch[:, :self.num_Pd]  # [B, num_Pd]
        
        # Use scatter instead of inplace assignment
        idx = self.bus_Pd.view(1, -1).expand(B, -1)  # [B, num_Pd]
        Pd_full = torch.zeros((B, self.Nbus), device=self.device, dtype=x_batch.dtype)
        Pd_full = Pd_full.scatter(1, idx, Pd_sparse)
        
        return Pd_full
    
    def __call__(
        self,
        Y_pred: torch.Tensor,          # [B, Kf, D] predicted trajectory
        fine_norm: torch.Tensor = None,  # [Kf] normalized lambda grid (unused, for API)
        x_batch: torch.Tensor = None,    # [B, input_dim] load data
        Pd: torch.Tensor = None,         # [B, Nbus] load power
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Pareto consistency loss for predicted trajectory.
        
        Args:
            Y_pred: [B, Kf, D] predicted voltage trajectory in NGT format
            fine_norm: [Kf] normalized lambda grid (for API consistency)
            x_batch: [B, input_dim] load data (SPARSE format: [Pd_nonzero, Qd_nonzero])
            Pd: [B, Nbus] load power (if available, overrides x_batch reconstruction)
        
        IMPORTANT: Either Pd or x_batch must be provided for correct Pg calculation!
        Without load data, we compute P_injection instead of Pg = P + Pd, which is WRONG.
        
        Returns:
            loss: scalar Pareto consistency loss
            details: dict with objective and violation statistics
        """
        self._call_count += 1
        
        # Check compute frequency
        if self.config.compute_freq <= 0:
            return torch.tensor(0.0, device=self.device), {}
        
        if self.config.compute_freq > 1 and (self._call_count % self.config.compute_freq) != 0:
            return torch.tensor(0.0, device=self.device), {'skipped': True}
        
        B, Kf, D = Y_pred.shape
        
        # ==================== CRITICAL: Reconstruct Pd from x_batch ====================
        # Without Pd, we compute P_injection instead of Pg = P + Pd, which is WRONG!
        # This was the key bug that caused "Pareto front looks geometrically similar but
        # cost/carbon values are wrong and may cross".
        if Pd is None and x_batch is not None:
            Pd = self._reconstruct_Pd_from_x_batch(x_batch, B)
            if Pd is None:
                # Fallback warning - this will produce incorrect results
                print("[ParetoLoss] WARNING: Could not reconstruct Pd, cost/carbon may be incorrect!")
        
        # 1. Reconstruct full voltage
        Vm_full, Va_full = self.reconstruct_full_voltage(Y_pred)  # [B, Kf, Nbus]
        
        # 2. Compute power generation (Pg = P_injection + Pd at generator buses)
        Pg = self.compute_power_generation(Vm_full, Va_full, Pd)  # [B, Kf, n_gen]
        
        # 3. Compute objectives (with clamped Pg to avoid negative values)
        cost, carbon = self.compute_objectives(Pg)  # [B, Kf]
        
        # 4. Compute monotonicity loss
        loss, details = self.compute_monotonicity_loss(cost, carbon)
        
        return self.config.alpha * loss, details


def compute_pareto_direction_loss(
    Y_pred: torch.Tensor,      # [B, Kf, D]
    Y_star: torch.Tensor,      # [B, Kf, D]
    n_va: int,
    alpha: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Lightweight Pareto direction consistency loss.
    
    This is a simpler alternative that doesn't require power flow computation.
    It enforces that Y_pred's changes along lambda axis match Y_star's changes.
    
    Intuition: If Y_star satisfies Pareto monotonicity, and Y_pred changes
    in the same direction as Y_star along lambda, then Y_pred should also
    satisfy Pareto monotonicity (approximately).
    
    Args:
        Y_pred: [B, Kf, D] predicted trajectory
        Y_star: [B, Kf, D] target trajectory (assumed to be Pareto-consistent)
        n_va: number of Va dimensions
        alpha: loss weight
    
    Returns:
        loss: scalar direction consistency loss
        details: dict with statistics
    """
    # Compute changes along lambda axis
    dY_pred = Y_pred[:, 1:, :] - Y_pred[:, :-1, :]  # [B, Kf-1, D]
    dY_star = Y_star[:, 1:, :] - Y_star[:, :-1, :]  # [B, Kf-1, D]
    
    # Wrap angle differences for Va dims
    if n_va > 0:
        dY_pred_va = torch.atan2(
            torch.sin(dY_pred[..., :n_va]),
            torch.cos(dY_pred[..., :n_va])
        )
        dY_star_va = torch.atan2(
            torch.sin(dY_star[..., :n_va]),
            torch.cos(dY_star[..., :n_va])
        )
        dY_pred = torch.cat([dY_pred_va, dY_pred[..., n_va:]], dim=-1)
        dY_star = torch.cat([dY_star_va, dY_star[..., n_va:]], dim=-1)
    
    # Direction consistency: penalize when signs differ
    # sign_product < 0 means signs are opposite
    sign_product = dY_pred * dY_star
    sign_violation = torch.relu(-sign_product)  # Penalize opposite signs
    
    # Also penalize magnitude differences (optional, lighter weight)
    magnitude_diff = (dY_pred - dY_star) ** 2
    
    # Combined loss
    sign_loss = sign_violation.mean()
    mag_loss = magnitude_diff.mean()
    
    loss = alpha * (0.5 * sign_loss + 0.5 * mag_loss)
    
    details = {
        'sign_loss': sign_loss.item(),
        'magnitude_loss': mag_loss.item(),
        'sign_violation_ratio': (sign_product < 0).float().mean().item(),
    }
    
    return loss, details


class _SimpleParetoContext:
    """Simple context object for ParetoLossComputer (avoids EvalContext complexity)."""
    pass


def create_pareto_loss_from_multi_pref_data(
    multi_pref_data: dict,
    sys_data,
    device: torch.device,
    config: ParetoLossConfig = None,
) -> ParetoLossComputer:
    """
    Create ParetoLossComputer from multi_pref_data dict.
    
    This is a convenience function for use in distill_traj_student.py.
    
    IMPORTANT: This function extracts bus_Pd from multi_pref_data to enable
    correct Pg calculation (Pg = P_injection + Pd at generator buses).
    """
    # Create a simple context object with required fields
    # (Avoids using EvalContext which has many required fields)
    ctx = _SimpleParetoContext()
    ctx.sys_data = sys_data
    
    # Ybus for power flow
    ctx.Ybus = sys_data.Ybus
    
    # Generator info
    ctx.bus_Pg = np.asarray(sys_data.bus_Pg).astype(int) if hasattr(sys_data.bus_Pg, '__len__') else np.array([sys_data.bus_Pg])
    ctx.idxPg = np.asarray(sys_data.idxPg).astype(int)
    ctx.gencost = sys_data.gencost.numpy() if hasattr(sys_data.gencost, 'numpy') else np.asarray(sys_data.gencost)
    
    # Extract gencost_Pg
    if ctx.gencost.shape[1] > 4:
        ctx.gencost_Pg = ctx.gencost[ctx.idxPg, 4:6]
    else:
        ctx.gencost_Pg = ctx.gencost[ctx.idxPg, :2]
    
    # GCI values (will be computed in _setup_system_tensors if None)
    ctx.gci_values = None
    
    # System parameters
    ctx.baseMVA = float(sys_data.baseMVA.item() if hasattr(sys_data.baseMVA, 'item') else sys_data.baseMVA)
    ctx.Nbus = int(multi_pref_data.get('Nbus', sys_data.Ybus.shape[0]))
    ctx.bus_slack = int(sys_data.bus_slack.item() if hasattr(sys_data.bus_slack, 'item') else sys_data.bus_slack)
    
    # NGT reconstruction info
    ctx.bus_Pnet_all = np.asarray(multi_pref_data['bus_Pnet_all']).astype(int) if 'bus_Pnet_all' in multi_pref_data else None
    ctx.bus_Pnet_noslack_all = np.asarray(multi_pref_data['bus_Pnet_noslack_all']).astype(int) if 'bus_Pnet_noslack_all' in multi_pref_data else None
    ctx.bus_ZIB_all = np.asarray(multi_pref_data['bus_ZIB_all']).astype(int) if 'bus_ZIB_all' in multi_pref_data and multi_pref_data['bus_ZIB_all'] is not None else None
    ctx.param_ZIMV = np.asarray(multi_pref_data['param_ZIMV']) if 'param_ZIMV' in multi_pref_data and multi_pref_data['param_ZIMV'] is not None else None
    
    # ==================== CRITICAL: bus_Pd for Pg calculation ====================
    if 'bus_Pd' in multi_pref_data:
        bus_Pd = multi_pref_data['bus_Pd']
        if hasattr(bus_Pd, 'numpy'):
            bus_Pd = bus_Pd.numpy()
        ctx.bus_Pd = np.asarray(bus_Pd).astype(int)
    elif hasattr(sys_data, 'bus_Pd'):
        bus_Pd = sys_data.bus_Pd
        if hasattr(bus_Pd, 'numpy'):
            bus_Pd = bus_Pd.numpy()
        ctx.bus_Pd = np.asarray(bus_Pd).astype(int)
    else:
        print("[ParetoLoss] WARNING: bus_Pd not found in multi_pref_data or sys_data!")
        ctx.bus_Pd = None
    
    n_va = int(multi_pref_data.get('NPred_Va', multi_pref_data['output_dim'] // 2))
    
    return ParetoLossComputer(ctx, n_va, device, config)
