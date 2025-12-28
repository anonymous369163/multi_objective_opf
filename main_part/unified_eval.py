#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_eval.py

[UNIFY] One unified evaluation entry for:
  - supervised (Vm model + Va model, full-bus output after slack insert)
  - ngt (single model predicts partial non-ZIB; then Kron reconstruct ZIB)
  - ngt_flow (VAE anchor + flow ODE, optional projection; outputs partial; then Kron)

Also provides unified post-processing:
  - supervised: behaves like your evaluate_model (PQg violated samples only, branch correction aligned to PQg)
  - ngt/ngt_flow: same logic but STRICTLY on independent variables (non-ZIB) + re-Kron ZIB

Author: Peng Yue
Date: 2025-12-18 

Key revisions:
  - [FIX] Va MAE after post-processing uses no-slack (consistent with evaluate_model definition)
  - [FIX] Free-subspace post-processing: solve in Jacobian's native column layout (2Nbus or 2Nbus-1), then map back to full 2Nbus dV
  - [FIX] Branch correction indexing robustness (avoid 1D slicing crash)
  - [NEW] Assertions for NGT-Flow anchor scaling dims
"""

from __future__ import annotations
import time
import os 
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable, Tuple, Union
import numpy as np
import torch
import matplotlib.pyplot as plt 

# ===== Project imports (adjust if your paths differ) =====
from config import get_config
from utils import (
    dPQbus_dV, dSlbus_dV,
    get_genload, get_vioPQg, get_viobran2,
    get_hisdV, get_dV,
    get_clamp, get_mae, get_rerr, get_Pgcost,
    get_viobran, get_rerr2,
    get_carbon_emission_vectorized, compute_hypervolume,
    get_gci_for_generators,
)
 


# =========================
# Generic helpers
# =========================

def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _as_torch(x, device=None, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        t = x.to(dtype=dtype)
        return t.to(device) if device is not None else t
    t = torch.from_numpy(np.asarray(x)).to(dtype)
    return t.to(device) if device is not None else t

def _ensure_1d_int(arr) -> np.ndarray:
    return np.asarray(arr).astype(int).ravel()

def _to_float(x, reduce: str = "mean") -> float:
    """
    Convert tensor/ndarray/list/scalar to python float.
    If x has multiple elements, reduce it (default: mean).
    """
    # python scalar
    if isinstance(x, (int, float, np.floating)):
        return float(x)

    # numpy
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return float(x.reshape(-1)[0])
        return float(np.mean(x) if reduce == "mean" else np.median(x))

    # torch
    if torch.is_tensor(x):
        t = x.detach()
        if t.numel() == 1:
            return float(t.cpu().item())
        if reduce == "mean":
            return float(t.mean().cpu().item())
        elif reduce == "sum":
            return float(t.sum().cpu().item())
        elif reduce == "median":
            return float(t.median().cpu().item())
        else:
            # fallback
            return float(t.mean().cpu().item())

    # list/tuple/others -> numpy
    arr = np.asarray(x)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return float(np.mean(arr) if reduce == "mean" else np.median(arr))


def _build_finc(branch: np.ndarray, nbus: int) -> np.ndarray:
    finc = np.zeros((branch.shape[0], nbus), dtype=float)
    for i in range(branch.shape[0]):
        f = int(branch[i, 0]) - 1  # MATPOWER 1-based -> 0-based
        finc[i, f] = 1.0
    return finc

def _insert_slack_va(Va_noslack: np.ndarray, bus_slack: int) -> np.ndarray:
    return np.insert(Va_noslack, bus_slack, values=0.0, axis=1)

def _remove_slack_va(Va_full: np.ndarray, bus_slack: int) -> np.ndarray:
    return np.delete(Va_full, bus_slack, axis=1)

def get_gci_for_generation_nodes(sys_data, idxPg: np.ndarray) -> np.ndarray:
    """
    Get GCI values aligned with generation nodes (bus_Pg), not all generators.
    
    Since Pred_Pg has shape [Ntest, len(bus_Pg)], we need GCI values of same length.
    We use idxPg to map from generators to the correct indices.
    
    Args:
        sys_data: Power system data
        idxPg: Index mapping from generators to bus_Pg locations
        
    Returns:
        gci_values: Array of GCI values aligned with bus_Pg [len(bus_Pg)]
    """
    # Get all generator GCI values
    gci_all = get_gci_for_generators(sys_data)
    
    # Select GCI values for the generators at bus_Pg nodes
    gci_for_nodes = gci_all[idxPg]
    
    return gci_for_nodes


def _kron_reconstruct_zib(
    Pred_Vm_full: np.ndarray,
    Pred_Va_full: np.ndarray,
    *,
    bus_Pnet_all: np.ndarray,
    bus_ZIB_all: np.ndarray,
    param_ZIMV: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given current non-ZIB values in Pred_Vm_full/Pred_Va_full,
    reconstruct ZIB values via Vy = param_ZIMV @ Vx, and write back into full arrays.
    """
    bus_Pnet_all = _ensure_1d_int(bus_Pnet_all)
    bus_ZIB_all = _ensure_1d_int(bus_ZIB_all)
    Vx = Pred_Vm_full[:, bus_Pnet_all] * np.exp(1j * Pred_Va_full[:, bus_Pnet_all])
    Vy = (np.asarray(param_ZIMV) @ Vx.T).T  # [Ntest, NZIB]
    Pred_Va_full[:, bus_ZIB_all] = np.angle(Vy)
    Pred_Vm_full[:, bus_ZIB_all] = np.abs(Vy)
    return Pred_Vm_full, Pred_Va_full


# =========================
# Prediction helper
# =========================

def predict_with_model(model, test_x, model_type, pretrain_model=None, config=None, device='cuda'):
    """
    Helper function to get predictions from different model types (supervised path).
    """
    model.eval()
    with torch.no_grad():
        if model_type == 'simple':
            y_pred = model(test_x)
        elif model_type == 'vae':
            y_pred = model(test_x, use_mean=True)
        elif model_type in ['rectified', 'gaussian', 'conditional', 'interpolation']:
            if pretrain_model is not None:
                z = pretrain_model(test_x, use_mean=True)
            else:
                output_dim = model.output_dim
                z = torch.randn(test_x.shape[0], output_dim).to(device)
            inf_step = getattr(config, 'inf_step', 100) if config else 100
            y_pred, _ = model.flow_backward(test_x, z, step=1/inf_step, method='Euler')
        elif model_type == 'diffusion':
            output_dim = model.output_dim
            inf_step = getattr(config, 'inf_step', 100) if config else 100
            use_vae_anchor = getattr(config, 'use_vae_anchor', False) if config else False
            if use_vae_anchor and pretrain_model is not None:
                vae_anchor = pretrain_model(test_x, use_mean=True)
                z = torch.randn(test_x.shape[0], output_dim).to(device)
                y_pred = model.diffusion_backward_with_anchor(test_x, z, vae_anchor, inf_step=inf_step)
            else:
                z = torch.randn(test_x.shape[0], output_dim).to(device)
                y_pred = model.diffusion_backward(test_x, z, inf_step=inf_step)
        elif model_type in ['gan', 'wgan']:
            latent_dim = model.latent_dim
            z = torch.randn(test_x.shape[0], latent_dim).to(device)
            y_pred = model(test_x, z)
        elif model_type in ['consistency_training', 'consistency_distillation']:
            y_pred = model.sampling(test_x, inf_step=1)
        else:
            raise NotImplementedError(f"Prediction for model type '{model_type}' not implemented")
    return y_pred


# =========================
# Context
# =========================

@dataclass
class EvalContext:
    # core
    config: Any
    sys_data: Any
    BRANFT: np.ndarray
    device: torch.device

    # test tensors / arrays
    x_test: torch.Tensor              # [Ntest, Din]
    yvmtests: torch.Tensor            # [Ntest, Nbus] physical Vm (torch)
    yvatests_noslack: torch.Tensor    # [Ntest, Nbus-1] physical Va (torch)

    # full-ground-truth
    Real_Vm_full: np.ndarray          # [Ntest, Nbus]
    Real_Va_full: np.ndarray          # [Ntest, Nbus] slack already inserted

    # loads
    Pdtest: np.ndarray                # [Ntest, Nbus]
    Qdtest: np.ndarray                # [Ntest, Nbus]

    # system
    Nbus: int
    Ntest: int
    bus_slack: int
    baseMVA: float

    branch: np.ndarray
    Ybus: Any
    Yf: Any
    Yt: Any
    bus_Pg: np.ndarray
    bus_Qg: np.ndarray
    MAXMIN_Pg: np.ndarray
    MAXMIN_Qg: np.ndarray

    # cost (MATPOWER style)
    idxPg: np.ndarray
    gencost: np.ndarray
    gencost_Pg: Optional[np.ndarray]  # Pre-extracted [c2, c1] format (if available)

    # post-process refs/bounds
    his_V: np.ndarray
    hisVm_min: Union[np.ndarray, float]
    hisVm_max: Union[np.ndarray, float]

    # NGT reconstruction info (None for supervised)
    bus_Pnet_all: Optional[np.ndarray] = None
    bus_Pnet_noslack_all: Optional[np.ndarray] = None
    bus_ZIB_all: Optional[np.ndarray] = None
    param_ZIMV: Optional[np.ndarray] = None

    # NGT bounds
    VmLb: Optional[Union[np.ndarray, float]] = None
    VmUb: Optional[Union[np.ndarray, float]] = None

    # knobs
    DELTA: float = 1e-4
    k_dV: float = 1.0    # origin 1.0
    flag_hisv: bool = True

    relax_ngt_post: bool = False # whether to relax the NGT post-processing
    
    # GCI values for carbon emission calculation
    gci_values: Optional[np.ndarray] = None  # GCI values aligned with bus_Pg


def build_ctx_from_supervised(config, sys_data, dataloaders, BRANFT, device) -> EvalContext:
    """
    Build EvalContext that matches evaluate_model's data/normalization exactly.
    
    Also adds NGT/Flow model required fields to support unified evaluation
    with the same test set. Note: supervised and unsupervised models use
    different normalization schemes:
    - Supervised: yvm_test is scaled (multiplied by scale_vm), converted to physical here
    - NGT/Flow: Use physical values directly, with Vscale/Vbias for sigmoid scaling
    """
    # Convert normalized supervised outputs to physical values
    # Supervised normalization: yvm_test = (Vm_phys - VmLb) / (VmUb - VmLb) * scale_vm
    # So: Vm_phys = yvm_test / scale_vm * (VmUb - VmLb) + VmLb
    yvmtests = sys_data.yvm_test / config.scale_vm * (sys_data.VmUb - sys_data.VmLb) + sys_data.VmLb
    yvatests = sys_data.yva_test / config.scale_va  # Va_phys = yva_test / scale_va

    Real_Vm_full = yvmtests.clone().cpu().numpy()
    Real_Va_full = _insert_slack_va(yvatests.clone().cpu().numpy(), int(sys_data.bus_slack))

    # ===== NGT/Flow model required fields =====
    # These are needed to support NGT and Flow models using the same test set
    
    # 1. Bus indices (already computed in load_training_data)
    bus_Pnet_all = _ensure_1d_int(sys_data.bus_Pnet_all) if hasattr(sys_data, 'bus_Pnet_all') and sys_data.bus_Pnet_all is not None else None
    bus_Pnet_noslack_all = _ensure_1d_int(sys_data.bus_Pnet_noslack_all) if hasattr(sys_data, 'bus_Pnet_noslack_all') and sys_data.bus_Pnet_noslack_all is not None else None
    bus_ZIB_all = _ensure_1d_int(sys_data.bus_ZIB_all) if hasattr(sys_data, 'bus_ZIB_all') and sys_data.bus_ZIB_all is not None else None
    
    # 2. Compute param_ZIMV for Kron reduction (ZIB reconstruction)
    # Note: Vscale and Vbias are not stored here because NGT/Flow models
    # have them built-in (as part of NetV/PreferenceConditionedNetV model)
    param_ZIMV = None
    if bus_Pnet_all is not None and bus_ZIB_all is not None and len(bus_ZIB_all) > 0:
        try:
            import scipy.sparse.linalg
            Ybus = sys_data.Ybus
            Ya = Ybus[np.ix_(bus_ZIB_all, bus_ZIB_all)]
            Yb = Ybus[np.ix_(bus_ZIB_all, bus_Pnet_all)]
            
            # Only invert if Ya is square and non-singular
            if Ya.shape[0] == Ya.shape[1] and np.linalg.matrix_rank(Ya.toarray()) == Ya.shape[0]:
                param_ZIMV = (-scipy.sparse.linalg.inv(Ya) @ Yb).toarray()
            else:
                param_ZIMV = None
                print("[Warning] Cannot compute param_ZIMV - Ya is singular")
        except Exception as e:
            param_ZIMV = None
            print(f"[Warning] Failed to compute param_ZIMV: {e}")
    
    # 4. Extract gencost_Pg (pre-extracted cost coefficients)
    gencost = _as_numpy(sys_data.gencost)
    idxPg = _ensure_1d_int(sys_data.idxPg)
    # Handle both MATPOWER format (7 columns) and simplified format (2 columns)
    if gencost.shape[1] > 4:
        # MATPOWER format: [MODEL, STARTUP, SHUTDOWN, NCOST, c2, c1, c0]
        # Extract columns 4 (c2) and 5 (c1)
        gencost_Pg = gencost[idxPg, 4:6]  # [c2, c1] coefficients
    else:
        # Simplified format: [c2, c1] or [c2, c1, ...]
        gencost_Pg = gencost[idxPg, :2]  # [c2, c1] coefficients
    
    # 5. NGT voltage bounds (for post-processing clipping)
    VmLb_ngt = getattr(config, 'ngt_VmLb', None)
    VmUb_ngt = getattr(config, 'ngt_VmUb', None)
    
    # 6. GCI values for carbon emission calculation
    gci_values = get_gci_for_generation_nodes(sys_data, idxPg)

    ctx = EvalContext(
        config=config,
        sys_data=sys_data,
        BRANFT=np.asarray(BRANFT),
        device=device,

        x_test=sys_data.x_test,
        yvmtests=yvmtests,
        yvatests_noslack=yvatests,

        Real_Vm_full=Real_Vm_full,
        Real_Va_full=Real_Va_full,

        Pdtest=_as_numpy(sys_data.Pdtest),
        Qdtest=_as_numpy(sys_data.Qdtest),

        Nbus=int(config.Nbus),
        Ntest=int(config.Ntest),
        bus_slack=int(sys_data.bus_slack),
        baseMVA=float(sys_data.baseMVA),

        branch=_as_numpy(sys_data.branch),
        Ybus=sys_data.Ybus,
        Yf=sys_data.Yf,
        Yt=sys_data.Yt,
        bus_Pg=_ensure_1d_int(sys_data.bus_Pg),
        bus_Qg=_ensure_1d_int(sys_data.bus_Qg),
        MAXMIN_Pg=_as_numpy(sys_data.MAXMIN_Pg),
        MAXMIN_Qg=_as_numpy(sys_data.MAXMIN_Qg),

        idxPg=idxPg,
        gencost=_as_numpy(sys_data.gencost),
        gencost_Pg=_as_numpy(gencost_Pg),  # Now extracted for NGT/Flow compatibility

        his_V=_as_numpy(sys_data.his_V),
        hisVm_min=_as_numpy(sys_data.hisVm_min),
        hisVm_max=_as_numpy(sys_data.hisVm_max),

        # NGT/Flow model fields
        bus_Pnet_all=bus_Pnet_all,
        bus_Pnet_noslack_all=bus_Pnet_noslack_all,
        bus_ZIB_all=bus_ZIB_all,
        param_ZIMV=param_ZIMV,
        
        # NGT voltage bounds (for post-processing)
        VmLb=VmLb_ngt,
        VmUb=VmUb_ngt,

        DELTA=float(config.DELTA),
        k_dV=float(config.k_dV),
        flag_hisv=bool(config.flag_hisv),
        
        # GCI values for carbon emission calculation
        gci_values=gci_values,
    )
    return ctx


def build_ctx_from_ngt(config, sys_data, ngt_data: Dict[str, Any], BRANFT, device) -> EvalContext:
    """
    Build EvalContext for NGT / NGT-Flow.
    """
    # [FIX] ensure x_test is torch tensor
    x_test = _as_torch(ngt_data["x_test"], device=None, dtype=torch.float32)
    Real_Vm_full = _as_numpy(ngt_data["yvm_test"])
    Real_Va_full = _as_numpy(ngt_data["yva_test"])  # full bus with slack included

    Ntest = int(Real_Vm_full.shape[0])
    Nbus = int(config.Nbus)
    bus_slack = int(sys_data.bus_slack)

    yvatests_noslack = _as_torch(_remove_slack_va(Real_Va_full, bus_slack), dtype=torch.float32)
    yvmtests = _as_torch(Real_Vm_full, dtype=torch.float32)

    # loads
    if "Pdtest" in ngt_data and "Qdtest" in ngt_data:
        Pdtest = _as_numpy(ngt_data["Pdtest"])
        Qdtest = _as_numpy(ngt_data["Qdtest"])
    else:
        baseMVA = float(sys_data.baseMVA)
        Pdtest = np.zeros((Ntest, Nbus), dtype=float)
        Qdtest = np.zeros((Ntest, Nbus), dtype=float)
        idx_test = _ensure_1d_int(ngt_data["idx_test"])
        bus_Pd = _ensure_1d_int(ngt_data["bus_Pd"])
        bus_Qd = _ensure_1d_int(ngt_data["bus_Qd"])
        Pdtest[:, bus_Pd] = _as_numpy(sys_data.RPd)[idx_test][:, bus_Pd] / baseMVA
        Qdtest[:, bus_Qd] = _as_numpy(sys_data.RQd)[idx_test][:, bus_Qd] / baseMVA

    # reconstruction info
    bus_Pnet_all = _ensure_1d_int(ngt_data["bus_Pnet_all"])
    bus_Pnet_noslack_all = bus_Pnet_all[bus_Pnet_all != bus_slack]
    bus_ZIB_all = _ensure_1d_int(ngt_data["bus_ZIB_all"]) if "bus_ZIB_all" in ngt_data else None
    param_ZIMV = ngt_data.get("param_ZIMV", None)

    ctx = EvalContext(
        config=config,
        sys_data=sys_data,
        BRANFT=np.asarray(BRANFT),
        device=device,

        x_test=x_test,
        yvmtests=yvmtests,
        yvatests_noslack=yvatests_noslack,

        Real_Vm_full=Real_Vm_full,
        Real_Va_full=Real_Va_full,

        Pdtest=Pdtest,
        Qdtest=Qdtest,

        Nbus=Nbus,
        Ntest=Ntest,
        bus_slack=bus_slack,
        baseMVA=float(sys_data.baseMVA),

        branch=_as_numpy(sys_data.branch),
        Ybus=sys_data.Ybus,
        Yf=sys_data.Yf,
        Yt=sys_data.Yt,
        bus_Pg=_ensure_1d_int(sys_data.bus_Pg),
        bus_Qg=_ensure_1d_int(sys_data.bus_Qg),
        MAXMIN_Pg=_as_numpy(ngt_data["MAXMIN_Pg"]),
        MAXMIN_Qg=_as_numpy(ngt_data["MAXMIN_Qg"]),

        idxPg=_ensure_1d_int(sys_data.idxPg),
        gencost=_as_numpy(sys_data.gencost),
        gencost_Pg=_as_numpy(ngt_data.get("gencost_Pg", None)),  # Use pre-extracted if available

        his_V=_as_numpy(sys_data.his_V),
        hisVm_min=_as_numpy(sys_data.hisVm_min),
        hisVm_max=_as_numpy(sys_data.hisVm_max),

        bus_Pnet_all=bus_Pnet_all,
        bus_Pnet_noslack_all=bus_Pnet_noslack_all,
        bus_ZIB_all=bus_ZIB_all,
        param_ZIMV=param_ZIMV,

        VmLb=getattr(config, "ngt_VmLb", None),
        VmUb=getattr(config, "ngt_VmUb", None),

        DELTA=float(getattr(config, "DELTA", 1e-4)),
        k_dV=float(getattr(config, "k_dV", 1.0)),
        flag_hisv=bool(getattr(config, "flag_hisv", True)),
        
        # GCI values for carbon emission calculation
        gci_values=get_gci_for_generation_nodes(sys_data, _ensure_1d_int(sys_data.idxPg)),
    )
    return ctx


# =========================
# Predictors
# =========================

@dataclass
class PredPack:
    Pred_Vm_full: np.ndarray
    Pred_Va_full: np.ndarray
    time_vm: float = 0.0
    time_va: float = 0.0
    time_nn_total: float = 0.0


class SupervisedPredictor:
    """
    Mimics evaluate_model's prediction procedure (supervised).
    """
    def __init__(
        self,
        model_vm: torch.nn.Module,
        model_va: torch.nn.Module,
        dataloaders: Dict[str, Any],
        *,
        model_type: str = "simple",
        pretrain_model_vm=None,
        pretrain_model_va=None,
        predict_fn: Optional[Callable] = None,
    ):
        self.model_vm = model_vm
        self.model_va = model_va
        self.dataloaders = dataloaders
        self.model_type = model_type
        self.pretrain_model_vm = pretrain_model_vm
        self.pretrain_model_va = pretrain_model_va
        self.predict_fn = predict_fn or predict_with_model

    def predict(self, ctx: EvalContext) -> PredPack:
        device = ctx.device

        # warmup
        if device.type == "cuda":
            with torch.no_grad():
                dummy_x = ctx.x_test[0:1].to(device)
                _ = self.predict_fn(self.model_vm, dummy_x, self.model_type,
                                    self.pretrain_model_vm, ctx.config, device)
                torch.cuda.synchronize()

        # Vm loop
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yvm_hat_list = []
        for test_x, _ in self.dataloaders["test_vm"]:
            test_x = test_x.to(device)
            pred = self.predict_fn(self.model_vm, test_x, self.model_type,
                                   self.pretrain_model_vm, ctx.config, device)
            yvm_hat_list.append(pred.cpu())
        if device.type == "cuda":
            torch.cuda.synchronize()
        time_vm = time.perf_counter() - t0

        yvm_hat = torch.cat(yvm_hat_list, dim=0).cpu()
        yvm_physical = yvm_hat.detach() / ctx.config.scale_vm * (ctx.sys_data.VmUb - ctx.sys_data.VmLb) + ctx.sys_data.VmLb
        yvm_clip = get_clamp(yvm_physical, ctx.sys_data.hisVm_min, ctx.sys_data.hisVm_max)
        Pred_Vm_full = yvm_clip.clone().numpy()

        # Va loop
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        yva_hat_list = []
        for test_x, _ in self.dataloaders["test_va"]:
            test_x = test_x.to(device)
            pred = self.predict_fn(self.model_va, test_x, self.model_type,
                                   self.pretrain_model_va, ctx.config, device)
            yva_hat_list.append(pred.cpu())
        if device.type == "cuda":
            torch.cuda.synchronize()
        time_va = time.perf_counter() - t1

        yva_hat = torch.cat(yva_hat_list, dim=0).cpu()
        yva_physical = yva_hat.detach() / ctx.config.scale_va
        Pred_Va_full = _insert_slack_va(yva_physical.clone().numpy(), ctx.bus_slack)

        return PredPack(
            Pred_Vm_full=Pred_Vm_full,
            Pred_Va_full=Pred_Va_full,
            time_vm=time_vm,
            time_va=time_va,
            time_nn_total=time_vm + time_va,
        )


class NGTPredictor:
    """
    NGT single model predictor (partial -> full).
    Output partial:
      [Va_noslack_nonZIB (len bus_Pnet_noslack_all), Vm_nonZIB (len bus_Pnet_all)]
    """
    def __init__(self, model_ngt: torch.nn.Module):
        self.model = model_ngt

    def predict(self, ctx: EvalContext) -> PredPack:
        assert ctx.bus_Pnet_all is not None and ctx.bus_Pnet_noslack_all is not None
        self.model.eval()
        x = ctx.x_test.to(ctx.device)

        if ctx.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            V_partial = self.model(x)
        if ctx.device.type == "cuda":
            torch.cuda.synchronize()
        time_nn = time.perf_counter() - t0

        V_partial = _as_numpy(V_partial)
        Pred_Vm_full, Pred_Va_full = reconstruct_full_from_partial(ctx, V_partial)
        return PredPack(Pred_Vm_full=Pred_Vm_full, Pred_Va_full=Pred_Va_full, time_nn_total=time_nn)


class NGTFlowPredictor:
    """
    Flow predictor: outputs the same partial vector format as NGT.

    You must pass:
      - flow_forward_ngt
      - optionally flow_forward_ngt_projected and P_tan_t
    """
    def __init__(
        self,
        model_flow: torch.nn.Module,
        vae_vm: torch.nn.Module,
        vae_va: torch.nn.Module,
        ngt_data: Dict[str, Any],
        preference: torch.Tensor,
        *,
        flow_forward_ngt: Callable,
        flow_forward_ngt_projected: Optional[Callable] = None,
        use_projection: Optional[bool] = None,
        P_tan_t: Optional[torch.Tensor] = None,
        flow_inf_steps: Optional[int] = None,
    ):
        self.model_flow = model_flow
        self.vae_vm = vae_vm
        self.vae_va = vae_va
        self.ngt_data = ngt_data
        self.preference = preference
        self.flow_forward_ngt = flow_forward_ngt
        self.flow_forward_ngt_projected = flow_forward_ngt_projected
        self.use_projection = use_projection
        self.P_tan_t = P_tan_t
        self.flow_inf_steps = flow_inf_steps

    def predict(self, ctx: EvalContext) -> PredPack:
        assert ctx.bus_Pnet_all is not None and ctx.bus_Pnet_noslack_all is not None

        self.model_flow.eval()
        self.vae_vm.eval()
        self.vae_va.eval()

        x = ctx.x_test.to(ctx.device)
        Ntest = x.shape[0]

        flow_steps = self.flow_inf_steps if self.flow_inf_steps is not None else getattr(ctx.config, "ngt_flow_inf_steps", 10)
        use_proj = self.use_projection if self.use_projection is not None else getattr(ctx.config, "ngt_use_projection", False)

        bus_slack = int(ctx.bus_slack)
        bus_Pnet_all = _ensure_1d_int(ctx.bus_Pnet_all)
        bus_Pnet_noslack_all = _ensure_1d_int(ctx.bus_Pnet_noslack_all)

        # ===== VAE anchor (physical) -> logit latent =====
        with torch.no_grad():
            Vm_vae = self.vae_vm(x, use_mean=True)          # scaled
            Va_vae_noslack = self.vae_va(x, use_mean=True)  # scaled

            scale_vm = float(ctx.config.scale_vm.item() if hasattr(ctx.config.scale_vm, "item") else ctx.config.scale_vm)
            scale_va = float(ctx.config.scale_va.item() if hasattr(ctx.config.scale_va, "item") else ctx.config.scale_va)

            VmLb = ctx.sys_data.VmLb
            VmUb = ctx.sys_data.VmUb
            if isinstance(VmLb, np.ndarray):
                VmLb_t = torch.from_numpy(VmLb).float().to(ctx.device)
                VmUb_t = torch.from_numpy(VmUb).float().to(ctx.device)
            elif isinstance(VmLb, torch.Tensor):
                VmLb_t = VmLb.to(ctx.device).float()
                VmUb_t = VmUb.to(ctx.device).float()
            else:
                VmLb_t = torch.full((ctx.Nbus,), float(VmLb), device=ctx.device)
                VmUb_t = torch.full((ctx.Nbus,), float(VmUb), device=ctx.device)

            Vm_vae_phys = Vm_vae / scale_vm * (VmUb_t - VmLb_t) + VmLb_t
            Va_vae_phys_noslack = Va_vae_noslack / scale_va

            Va_full = torch.zeros(Ntest, ctx.Nbus, device=ctx.device)
            Va_full[:, :bus_slack] = Va_vae_phys_noslack[:, :bus_slack]
            Va_full[:, bus_slack + 1:] = Va_vae_phys_noslack[:, bus_slack:]

            Vm_nonZIB = Vm_vae_phys[:, bus_Pnet_all]
            Va_nonZIB_noslack = Va_full[:, bus_Pnet_noslack_all]
            V_anchor_phys = torch.cat([Va_nonZIB_noslack, Vm_nonZIB], dim=1)

            eps = 1e-6

            # [FIX] prefer model_flow's Vscale/Vbias (must match partial ordering)
            Vscale = self.model_flow.Vscale.to(ctx.device)
            Vbias = self.model_flow.Vbias.to(ctx.device)

            # [NEW] assert dims match (avoids silent ordering/dim bugs)
            assert Vscale.numel() == V_anchor_phys.shape[1], (Vscale.shape, V_anchor_phys.shape)
            assert Vbias.numel() == V_anchor_phys.shape[1], (Vbias.shape, V_anchor_phys.shape)

            u = (V_anchor_phys - Vbias) / (Vscale + 1e-12)
            u = torch.clamp(u, eps, 1 - eps)
            z_anchor = torch.log(u / (1 - u))

        pref_batch = self.preference.to(ctx.device).expand(Ntest, -1)

        if ctx.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            if use_proj and (self.P_tan_t is not None) and (self.flow_forward_ngt_projected is not None):
                V_partial = self.flow_forward_ngt_projected(
                    self.model_flow, x, z_anchor, self.P_tan_t.to(ctx.device),
                    pref_batch, flow_steps, training=False
                )
            else:
                V_partial = self.flow_forward_ngt(
                    self.model_flow, x, z_anchor,
                    pref_batch, flow_steps, training=False
                )
        if ctx.device.type == "cuda":
            torch.cuda.synchronize()
        time_nn = time.perf_counter() - t0

        V_partial = _as_numpy(V_partial)
        Pred_Vm_full, Pred_Va_full = reconstruct_full_from_partial(ctx, V_partial)
        return PredPack(Pred_Vm_full=Pred_Vm_full, Pred_Va_full=Pred_Va_full, time_nn_total=time_nn)


class MultiPreferencePredictor:
    """
    Predictor for multi-preference models.
    
    Supports multiple model types:
    - 'simple': MLP with preference concatenated to input
    - 'vae': VAE with preference concatenated to input
    - 'flow'/'rectified': Flow model with preference-aware MLP (FiLM conditioning)
    - 'diffusion': Diffusion model with preference concatenated to input
    
    Key features:
    - Accepts a preference parameter (lambda_carbon) for conditioning
    - Outputs NGT format (partial voltage) and uses Kron reconstruction
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        multi_pref_data: Dict[str, Any],
        lambda_carbon: float,
        model_type: str = 'simple',
        *,
        pretrain_model: Optional[torch.nn.Module] = None,
        num_flow_steps: int = 10,
        flow_method: str = 'euler',
        training_mode: str = 'standard',
    ):
        """
        Initialize the multi-preference predictor.
        
        Args:
            model: Trained model
            multi_pref_data: Multi-preference data dictionary
            lambda_carbon: Preference value for prediction
            model_type: Type of model ('simple', 'vae', 'flow', 'rectified', 'diffusion')
            pretrain_model: Optional VAE model for anchor generation (flow models)
            num_flow_steps: Number of ODE integration steps (flow models)
            flow_method: ODE solver method ('euler' or 'heun')
            training_mode: Training mode ('standard' or 'preference_trajectory')
        """
        self.model = model
        self.multi_pref_data = multi_pref_data
        self.lambda_carbon = lambda_carbon
        self.model_type = model_type
        self.pretrain_model = pretrain_model
        self.num_flow_steps = num_flow_steps
        self.flow_method = flow_method
        self.training_mode = training_mode
        
        # Get normalization factor for preference
        lambda_carbon_values = multi_pref_data.get('lambda_carbon_values', [55.0])
        self.lc_max = max(lambda_carbon_values) if max(lambda_carbon_values) > 0 else 1.0
        
        # For preference_trajectory mode: prepare lambda trajectory
        if training_mode == 'preference_trajectory':
            lambda_carbon_sorted = sorted(lambda_carbon_values)
            self.lambda_min = lambda_carbon_sorted[0]
            self.lambda_max = lambda_carbon_sorted[-1]
            # Create normalized lambda trajectory
            self.lambda_trajectory = [
                (lc - self.lambda_min) / (self.lambda_max - self.lambda_min) 
                if self.lambda_max > self.lambda_min else 0.0
                for lc in lambda_carbon_sorted
            ]
            self.lambda_trajectory_raw = lambda_carbon_sorted
        
        # Get Vscale and Vbias for simple model output transformation
        self.Vscale = multi_pref_data.get('Vscale')
        self.Vbias = multi_pref_data.get('Vbias')
    
    def predict(self, ctx: EvalContext) -> PredPack:
        """
        Predict voltage for test samples with the specified preference.
        
        Args:
            ctx: Evaluation context with test data
        
        Returns:
            PredPack with Pred_Vm_full, Pred_Va_full, and timing info
        """
        assert ctx.bus_Pnet_all is not None and ctx.bus_Pnet_noslack_all is not None
        
        self.model.eval()
        if self.pretrain_model is not None:
            self.pretrain_model.eval()
        
        x = ctx.x_test.to(ctx.device)
        Ntest = x.shape[0]
        output_dim = self.multi_pref_data['output_dim']
        
        # Create preference tensor (normalized)
        pref = torch.full((Ntest, 1), self.lambda_carbon / self.lc_max, device=ctx.device)
        
        # Move Vscale/Vbias to device if needed
        if self.Vscale is not None:
            Vscale = self.Vscale.to(ctx.device) if isinstance(self.Vscale, torch.Tensor) else torch.tensor(self.Vscale, device=ctx.device)
            Vbias = self.Vbias.to(ctx.device) if isinstance(self.Vbias, torch.Tensor) else torch.tensor(self.Vbias, device=ctx.device)
        
        # Timing
        if ctx.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        with torch.no_grad():
            if self.model_type == 'simple':
                # Simple MLP: concatenate preference to input
                x_with_pref = torch.cat([x, pref], dim=1)
                V_partial = self.model(x_with_pref)
                # V_partial is already in physical units (model has sigmoid + scale/bias)
                
            elif self.model_type == 'vae':
                # VAE: use preference_aware_mlp if available, otherwise concatenate
                if hasattr(self.model, 'pref_dim') and self.model.pref_dim > 0:
                    # Use preference_aware_mlp with FiLM conditioning
                    V_partial = self.model(x, use_mean=True, pref=pref)
                else:
                    # Fallback: concatenate preference to input
                    x_with_pref = torch.cat([x, pref], dim=1)
                    V_partial = self.model(x_with_pref, use_mean=True)
                
            elif self.model_type in ['flow', 'rectified', 'gaussian', 'conditional', 'interpolation']:
                # Flow model with preference-aware MLP
                if self.training_mode == 'preference_trajectory':
                    # Preference trajectory mode: integrate along lambda trajectory from λ=0 to target λ
                    V_partial = self._sample_preference_trajectory(x, ctx.device)
                else:
                    # Standard mode: Flow Matching from anchor to target
                    # Generate anchor points
                    if self.pretrain_model is not None:
                        # Check if pretrain_model supports preference_aware_mlp
                        if hasattr(self.pretrain_model, 'pref_dim') and self.pretrain_model.pref_dim > 0:
                            # Use preference_aware_mlp with FiLM conditioning
                            # For initial anchor, use lambda=0 (minimum lambda)
                            lambda_min_val = min(self.multi_pref_data.get('lambda_carbon_values', [0.0]))
                            pref_anchor = torch.full((Ntest, 1), lambda_min_val / self.lc_max, device=ctx.device)
                            z = self.pretrain_model(x, use_mean=True, pref=pref_anchor)
                        else:
                            # Fallback: concatenate preference to input
                            lambda_min_val = min(self.multi_pref_data.get('lambda_carbon_values', [0.0]))
                            x_with_pref_anchor = torch.cat([x, torch.full((Ntest, 1), lambda_min_val / self.lc_max, device=ctx.device)], dim=1)
                            z = self.pretrain_model(x_with_pref_anchor, use_mean=True)
                    else:
                        z = torch.randn(Ntest, output_dim, device=ctx.device)
                    
                    # Sample from flow model
                    V_partial = self.model.sampling_with_pref(
                        x, z, pref,
                        num_steps=self.num_flow_steps,
                        method=self.flow_method
                    )
                
            elif self.model_type == 'diffusion':
                # Diffusion model: concatenate preference to input
                x_with_pref = torch.cat([x, pref], dim=1)
                # Use sampling method from DM
                z = torch.randn(Ntest, output_dim, device=ctx.device)
                V_partial = self.model.sample(x_with_pref, z, steps=self.num_flow_steps)
                
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        
        if ctx.device.type == "cuda":
            torch.cuda.synchronize()
        time_nn = time.perf_counter() - t0
        
        # Convert to numpy and reconstruct full voltage
        V_partial = _as_numpy(V_partial)
        Pred_Vm_full, Pred_Va_full = reconstruct_full_from_partial(ctx, V_partial)
        
        return PredPack(
            Pred_Vm_full=Pred_Vm_full,
            Pred_Va_full=Pred_Va_full,
            time_nn_total=time_nn
        )
    
    def _sample_preference_trajectory(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Sample from preference trajectory mode: integrate along lambda trajectory.
        
        This method integrates from λ=0 (initial point from VAE) to target λ using
        the learned velocity field dx/dλ.
        
        Args:
            x: Scene features [B, input_dim]
            device: Device for computation
            
        Returns:
            V_partial: Predicted voltage in partial format [B, output_dim]
        """
        batch_size = x.shape[0]
        output_dim = self.multi_pref_data['output_dim']
        
        # Get initial point at λ=0 (minimum lambda)
        lambda_min_val = self.lambda_min
        lambda_target_norm = (self.lambda_carbon - self.lambda_min) / (self.lambda_max - self.lambda_min) \
            if self.lambda_max > self.lambda_min else 0.0
        
        # Generate initial anchor at λ=0 using VAE
        if self.pretrain_model is not None:
            if hasattr(self.pretrain_model, 'pref_dim') and self.pretrain_model.pref_dim > 0:
                pref_init = torch.full((batch_size, 1), lambda_min_val / self.lc_max, device=device)
                x_current = self.pretrain_model(x, use_mean=True, pref=pref_init)
            else:
                x_with_pref_init = torch.cat([x, torch.full((batch_size, 1), lambda_min_val / self.lc_max, device=device)], dim=1)
                x_current = self.pretrain_model(x_with_pref_init, use_mean=True)
        else:
            # Fallback: use random initialization
            x_current = torch.randn(batch_size, output_dim, device=device)
        
        # Integrate along lambda trajectory from λ_min to λ_target
        # Use the lambda trajectory points that are <= target lambda
        lambda_trajectory_norm = [l for l in self.lambda_trajectory if l <= lambda_target_norm]
        lambda_trajectory_raw = [self.lambda_trajectory_raw[i] for i, l in enumerate(self.lambda_trajectory) if l <= lambda_target_norm]
        
        # Add target lambda if not already in trajectory
        if len(lambda_trajectory_norm) == 0 or lambda_trajectory_norm[-1] < lambda_target_norm:
            lambda_trajectory_norm.append(lambda_target_norm)
            lambda_trajectory_raw.append(self.lambda_carbon)
        
        # Integrate using RK2 (Heun) method with real Δλ for better accuracy
        # RK2 is more stable than Euler and reduces error accumulation
        with torch.no_grad():
            for k in range(len(lambda_trajectory_norm) - 1):
                lambda_current_norm = lambda_trajectory_norm[k]
                lambda_next_norm = lambda_trajectory_norm[k+1]
                dlambda = lambda_next_norm - lambda_current_norm
                
                # RK2 (Heun) method: two-stage predictor-corrector
                # Stage 1: Euler step (predictor)
                lambda_current_tensor = torch.full((batch_size, 1), lambda_current_norm, device=device)
                v0 = self.model.predict_vec(x, x_current, lambda_current_tensor, lambda_current_tensor)
                x_euler = x_current + dlambda * v0
                
                # Stage 2: Use midpoint velocity (corrector)
                lambda_next_tensor = torch.full((batch_size, 1), lambda_next_norm, device=device)
                v1 = self.model.predict_vec(x, x_euler, lambda_next_tensor, lambda_next_tensor)
                
                # Final step: average of v0 and v1
                x_current = x_current + dlambda * 0.5 * (v0 + v1)
        
        return x_current


# Backward compatibility alias
MultiPreferenceFlowPredictor = MultiPreferencePredictor


def build_ctx_from_multi_preference(
    config, sys_data, multi_pref_data, BRANFT, device, lambda_carbon=None
) -> EvalContext:
    """
    Build EvalContext for multi-preference evaluation.
    
    This is similar to build_ctx_from_ngt but uses multi_pref_data
    for NGT-style evaluation with Kron reconstruction.
    
    Args:
        config: Configuration object
        sys_data: Power system data
        multi_pref_data: Multi-preference data dictionary
        BRANFT: Branch from-to indices
        device: Device
        lambda_carbon: Optional preference value. If provided, uses corresponding ground truth labels.
                      If None, uses NGT test set (if available) or training set without labels.
    
    Returns:
        EvalContext configured for NGT-style evaluation
    """
    # Check if we should use test set or training set
    use_test_set = getattr(config, 'multi_pref_use_test_set', False)
    
    if use_test_set:
        # Try to use NGT test set
        try:
            from data_loader import load_ngt_training_data
            ngt_data, _ = load_ngt_training_data(config, sys_data=sys_data)
            if 'x_test' in ngt_data and 'y_test' in ngt_data:
                x_test = _as_torch(ngt_data['x_test'], device=None, dtype=torch.float32)
                y_test = ngt_data['y_test']  # NGT format: [Va_noslack_nonZIB, Vm_nonZIB]
                Ntest = x_test.shape[0]
                print(f"[Eval] Using NGT test set: {Ntest} samples")
            else:
                raise KeyError("NGT test set not available")
        except Exception as e:
            print(f"[Warning] Failed to load NGT test set: {e}")
            print(f"[Eval] Falling back to training set")
            use_test_set = False
    
    if not use_test_set:
        # Use validation set (which has multi-preference labels but was not used for training)
        if 'x_val' in multi_pref_data:
            x_test = _as_torch(multi_pref_data['x_val'], device=None, dtype=torch.float32)
            Ntest = int(multi_pref_data['n_val'])
            print(f"[Eval] Using validation set: {Ntest} samples (not used in training)")
        else:
            # Fallback to training set if validation set not available (backward compatibility)
            x_test = _as_torch(multi_pref_data['x_train'], device=None, dtype=torch.float32)
            Ntest = int(multi_pref_data['n_train'])
            print(f"[Warning] Validation set not found, using training set: {Ntest} samples")
        
        # If lambda_carbon is provided, use corresponding ground truth from validation set
        if lambda_carbon is not None:
            # Try validation set first
            if 'y_val_by_pref' in multi_pref_data:
                y_val_by_pref = multi_pref_data['y_val_by_pref']
                if lambda_carbon in y_val_by_pref:
                    y_test = y_val_by_pref[lambda_carbon]  # NGT format
                    print(f"[Eval] Using validation set ground truth for lambda_carbon={lambda_carbon:.2f}")
                else:
                    # Find closest lambda_carbon
                    lambda_carbon_values = multi_pref_data['lambda_carbon_values']
                    closest_lc = min(lambda_carbon_values, key=lambda x: abs(x - lambda_carbon))
                    y_test = y_val_by_pref[closest_lc]
                    print(f"[Eval] Using closest validation ground truth: lambda_carbon={closest_lc:.2f} (requested {lambda_carbon:.2f})")
            else:
                # Fallback to training set (backward compatibility)
                y_train_by_pref = multi_pref_data['y_train_by_pref']
                if lambda_carbon in y_train_by_pref:
                    y_test = y_train_by_pref[lambda_carbon]
                    print(f"[Eval] Using training set ground truth for lambda_carbon={lambda_carbon:.2f} (validation set not available)")
                else:
                    lambda_carbon_values = multi_pref_data['lambda_carbon_values']
                    closest_lc = min(lambda_carbon_values, key=lambda x: abs(x - lambda_carbon))
                    y_test = y_train_by_pref[closest_lc]
                    print(f"[Eval] Using closest training ground truth: lambda_carbon={closest_lc:.2f} (requested {lambda_carbon:.2f})")
        else:
            # No ground truth available, use placeholder
            output_dim = multi_pref_data['output_dim']
            y_test = torch.zeros((Ntest, output_dim), dtype=torch.float32)
            print(f"[Eval] No ground truth available (lambda_carbon not specified)")
    
    Nbus = int(config.Nbus)
    bus_slack = int(sys_data.bus_slack)
    baseMVA = float(sys_data.baseMVA)
    
    # Convert y_test (NGT format) to full voltage format
    # y_test format: [Va_noslack_nonZIB (NPred_Va), Vm_nonZIB (NPred_Vm)]
    bus_Pnet_all = _ensure_1d_int(multi_pref_data['bus_Pnet_all'])
    bus_Pnet_noslack_all = _ensure_1d_int(multi_pref_data['bus_Pnet_noslack_all'])
    
    NPred_Va = len(bus_Pnet_noslack_all)
    NPred_Vm = len(bus_Pnet_all)
    
    # Extract Va and Vm from NGT format
    y_test_np = _as_numpy(y_test)
    Va_noslack_nonZIB = y_test_np[:, :NPred_Va]
    Vm_nonZIB = y_test_np[:, NPred_Va:]
    
    # Reconstruct full voltage
    Real_Va_full = np.zeros((Ntest, Nbus), dtype=float)
    Real_Vm_full = np.zeros((Ntest, Nbus), dtype=float)
    
    Real_Va_full[:, bus_Pnet_noslack_all] = Va_noslack_nonZIB
    Real_Va_full[:, bus_slack] = 0.0  # Slack bus angle is 0
    Real_Vm_full[:, bus_Pnet_all] = Vm_nonZIB
    
    # Apply Kron reconstruction for ZIB if available
    bus_ZIB_all = multi_pref_data.get('bus_ZIB_all')
    param_ZIMV = multi_pref_data.get('param_ZIMV')
    if bus_ZIB_all is not None and param_ZIMV is not None and len(bus_ZIB_all) > 0:
        Real_Vm_full, Real_Va_full = _kron_reconstruct_zib(
            Real_Vm_full, Real_Va_full,
            bus_Pnet_all=bus_Pnet_all,
            bus_ZIB_all=_ensure_1d_int(bus_ZIB_all),
            param_ZIMV=np.asarray(param_ZIMV),
        )
    
    # Create yvmtests and yvatests_noslack
    yvmtests = _as_torch(Real_Vm_full, dtype=torch.float32)
    yvatests_noslack = _as_torch(_remove_slack_va(Real_Va_full, bus_slack), dtype=torch.float32)
    
    # Bus indices (already defined above, but keep for clarity)
    bus_ZIB_all = _ensure_1d_int(multi_pref_data['bus_ZIB_all']) if multi_pref_data.get('bus_ZIB_all') is not None else None
    
    # Power flow data: extract from x_test
    # x_test format: [Pd_nonzero, Qd_nonzero] / baseMVA
    bus_Pd = _ensure_1d_int(multi_pref_data['bus_Pd'])
    bus_Qd = _ensure_1d_int(multi_pref_data['bus_Qd'])
    
    x_test_np = _as_numpy(x_test)
    n_pd = len(bus_Pd)
    n_qd = len(bus_Qd)
    
    Pdtest = np.zeros((Ntest, Nbus), dtype=float)
    Qdtest = np.zeros((Ntest, Nbus), dtype=float)
    
    if n_pd > 0 and n_qd > 0:
        # x_test contains [Pd, Qd] concatenated and normalized by baseMVA
        Pd_pu = x_test_np[:, :n_pd]  # Active power demand (p.u.)
        Qd_pu = x_test_np[:, n_pd:n_pd + n_qd]  # Reactive power demand (p.u.)
        
        # CRITICAL FIX: get_genload expects Pdtest and Qdtest in p.u. (not MW/MVAr)
        # In single-objective training, Pdtest/Qdtest are already in p.u.
        # So we should NOT multiply by baseMVA here - keep them in p.u.
        Pdtest[:, bus_Pd] = Pd_pu  # Keep in p.u. (not * baseMVA)
        Qdtest[:, bus_Qd] = Qd_pu  # Keep in p.u. (not * baseMVA)
    
    # Extract gencost_Pg
    gencost = _as_numpy(sys_data.gencost)
    idxPg = _ensure_1d_int(sys_data.idxPg)
    if gencost.shape[1] > 4:
        gencost_Pg = gencost[idxPg, 4:6]  # [c2, c1]
    else:
        gencost_Pg = gencost[idxPg, :2]
    
    ctx = EvalContext(
        config=config,
        sys_data=sys_data,
        BRANFT=np.asarray(BRANFT),
        device=device,

        x_test=x_test,
        yvmtests=yvmtests,
        yvatests_noslack=yvatests_noslack,

        Real_Vm_full=Real_Vm_full,
        Real_Va_full=Real_Va_full,

        Pdtest=Pdtest,
        Qdtest=Qdtest,

        Nbus=Nbus,
        Ntest=Ntest,
        bus_slack=bus_slack,
        baseMVA=baseMVA,

        branch=_as_numpy(sys_data.branch),
        Ybus=sys_data.Ybus,
        Yf=sys_data.Yf,
        Yt=sys_data.Yt,
        bus_Pg=_ensure_1d_int(sys_data.bus_Pg),
        bus_Qg=_ensure_1d_int(sys_data.bus_Qg),
        MAXMIN_Pg=_as_numpy(sys_data.MAXMIN_Pg),
        MAXMIN_Qg=_as_numpy(sys_data.MAXMIN_Qg),

        idxPg=idxPg,
        gencost=gencost,
        gencost_Pg=_as_numpy(gencost_Pg),

        his_V=_as_numpy(multi_pref_data.get('his_V')) if multi_pref_data.get('his_V') is not None else _as_numpy(sys_data.his_V),
        hisVm_min=_as_numpy(multi_pref_data.get('hisVm_min')) if multi_pref_data.get('hisVm_min') is not None else _as_numpy(sys_data.hisVm_min),
        hisVm_max=_as_numpy(multi_pref_data.get('hisVm_max')) if multi_pref_data.get('hisVm_max') is not None else _as_numpy(sys_data.hisVm_max),

        bus_Pnet_all=bus_Pnet_all,
        bus_Pnet_noslack_all=bus_Pnet_noslack_all,
        bus_ZIB_all=bus_ZIB_all,
        param_ZIMV=param_ZIMV,

        VmLb=getattr(config, "ngt_VmLb", None),
        VmUb=getattr(config, "ngt_VmUb", None),

        DELTA=float(getattr(config, "DELTA", 1e-4)),
        k_dV=float(getattr(config, "k_dV", 1.0)),
        flag_hisv=bool(getattr(config, "flag_hisv", True)),
        
        # GCI values for carbon emission calculation
        gci_values=get_gci_for_generation_nodes(sys_data, idxPg),
    )
    
    return ctx


# =========================
# Partial -> Full reconstruction (NGT/Flow)
# =========================

def reconstruct_full_from_partial(ctx: EvalContext, V_partial: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    V_partial layout:
      [Va_noslack_nonZIB (ordered by ctx.bus_Pnet_noslack_all),
       Vm_nonZIB (ordered by ctx.bus_Pnet_all)]
    """
    bus_slack = int(ctx.bus_slack)
    Nbus = int(ctx.Nbus)

    bus_Pnet_all = _ensure_1d_int(ctx.bus_Pnet_all)
    bus_Pnet_noslack_all = _ensure_1d_int(ctx.bus_Pnet_noslack_all)

    NPred_Va = len(bus_Pnet_noslack_all)
    NPred_Vm = len(bus_Pnet_all)
    assert V_partial.shape[1] == NPred_Va + NPred_Vm, \
        f"V_partial dim mismatch: got {V_partial.shape[1]}, expect {NPred_Va + NPred_Vm}"

    Va_noslack_nonZIB = V_partial[:, :NPred_Va]
    Vm_nonZIB = V_partial[:, NPred_Va:]

    Pred_Va_full = np.zeros((V_partial.shape[0], Nbus), dtype=float)
    Pred_Vm_full = np.zeros((V_partial.shape[0], Nbus), dtype=float)

    Pred_Va_full[:, bus_Pnet_noslack_all] = Va_noslack_nonZIB
    Pred_Va_full[:, bus_slack] = 0.0
    Pred_Vm_full[:, bus_Pnet_all] = Vm_nonZIB

    # Kron reconstruct ZIB if available
    if ctx.param_ZIMV is not None and ctx.bus_ZIB_all is not None:
        Pred_Vm_full, Pred_Va_full = _kron_reconstruct_zib(
            Pred_Vm_full, Pred_Va_full,
            bus_Pnet_all=bus_Pnet_all,
            bus_ZIB_all=_ensure_1d_int(ctx.bus_ZIB_all),
            param_ZIMV=np.asarray(ctx.param_ZIMV),
        )

    # optional NGT Vm clamp bounds
    if ctx.VmLb is not None and ctx.VmUb is not None:
        Pred_Vm_full = np.clip(Pred_Vm_full, ctx.VmLb, ctx.VmUb)

    return Pred_Vm_full, Pred_Va_full


# =========================
# Jacobian layout adapters (core fix for "free subspace")
# =========================

# [NEW] two common Jacobian layouts
#   - "full":     columns = [Va_full (Nbus), Vm_full (Nbus)] => 2Nbus
#   - "noslack":  columns = [Va_noslack (Nbus-1), Vm_full (Nbus)] => 2Nbus-1

def _infer_jac_layout(nbus: int, jac_cols: int) -> str:
    if jac_cols == 2 * nbus:
        return "full"
    if jac_cols == 2 * nbus - 1:
        return "noslack"
    raise ValueError(f"Unexpected Jacobian cols={jac_cols}, expected 2Nbus or 2Nbus-1 (Nbus={nbus})")

def _fullcol_to_jaccol(full_col: int, *, nbus: int, bus_slack: int, layout: str) -> Optional[int]:
    """
    Map a "full 2Nbus dV column index" to Jacobian column index.
    full_col in [0..Nbus-1] => Va_full[bus]
    full_col in [Nbus..2Nbus-1] => Vm_full[bus]
    """
    if layout == "full":
        return full_col

    # layout == "noslack": Va excludes slack; Vm kept
    if full_col < nbus:
        bus = full_col
        if bus == bus_slack:
            return None  # no slack angle column in Jacobian
        return bus if bus < bus_slack else bus - 1

    # Vm part
    bus = full_col - nbus
    return (nbus - 1) + bus

def _jacvec_to_full(jac_vec: np.ndarray, *, nbus: int, bus_slack: int, layout: str) -> np.ndarray:
    """
    Convert a Jacobian-space dV vector to full 2Nbus vector:
      [Va_full (Nbus), Vm_full (Nbus)]
    """
    jac_vec = np.asarray(jac_vec).ravel()
    full = np.zeros((2 * nbus,), dtype=float)

    if layout == "full":
        assert jac_vec.size == 2 * nbus
        return jac_vec.copy()

    # layout == "noslack"
    assert jac_vec.size == 2 * nbus - 1

    # Va (no-slack) -> Va_full
    # buses < slack: jac col = bus
    # buses > slack: jac col = bus-1
    for bus in range(nbus):
        if bus == bus_slack:
            full[bus] = 0.0
        elif bus < bus_slack:
            full[bus] = jac_vec[bus]
        else:
            full[bus] = jac_vec[bus - 1]

    # Vm: jac col = (Nbus-1) + bus
    for bus in range(nbus):
        full[nbus + bus] = jac_vec[(nbus - 1) + bus]

    return full

def _make_indep_full_cols(ctx: EvalContext) -> np.ndarray:
    """
    Independent variable columns in *full 2Nbus coordinate*:
      [Va_full cols 0..Nbus-1, Vm_full cols Nbus..2Nbus-1]

    - supervised: full 2Nbus
    - ngt/ngt_flow: Va on nonZIB excluding slack, Vm on nonZIB
    """
    nbus = int(ctx.Nbus)
    bus_slack = int(ctx.bus_slack)

    if ctx.bus_Pnet_all is None:
        return np.arange(2 * nbus, dtype=int)

    indep_buses = _ensure_1d_int(ctx.bus_Pnet_all)
    Va_buses = indep_buses[indep_buses != bus_slack]
    Vm_buses = indep_buses

    Va_cols = Va_buses
    Vm_cols = nbus + Vm_buses
    return np.concatenate([Va_cols, Vm_cols], axis=0).astype(int)

def _make_indep_jac_cols(ctx: EvalContext, *, layout: str, jac_dim: int) -> np.ndarray:
    """
    Convert indep_full_cols -> indep_jac_cols (in Jacobian coordinate).
    """
    nbus = int(ctx.Nbus)
    bus_slack = int(ctx.bus_slack)
    indep_full_cols = _make_indep_full_cols(ctx)

    jac_cols = []
    for c in indep_full_cols.tolist():
        jc = _fullcol_to_jaccol(c, nbus=nbus, bus_slack=bus_slack, layout=layout)
        if jc is not None:
            jac_cols.append(jc)
    indep_jac_cols = np.asarray(jac_cols, dtype=int)

    # safety
    if indep_jac_cols.size > 0:
        assert indep_jac_cols.min() >= 0
        assert indep_jac_cols.max() < jac_dim
    return indep_jac_cols

def _lift_sub_to_full(dv_sub: np.ndarray, indep_jac_cols: np.ndarray, *, nbus: int, bus_slack: int, layout: str, jac_dim: int) -> np.ndarray:
    """
    Given dv_sub in indep_jac_cols space, lift to full 2Nbus.
    """
    dv_sub = np.asarray(dv_sub).ravel()
    jac_vec = np.zeros((jac_dim,), dtype=float)
    jac_vec[indep_jac_cols] = dv_sub
    return _jacvec_to_full(jac_vec, nbus=nbus, bus_slack=bus_slack, layout=layout)


# =========================
# Subspace solvers (PQg side)
# =========================

def get_hisdV_subspace_mapped(
    ctx: EvalContext,
    *,
    lsPg, lsQg, lsidxPg, lsidxQg,
    num_viotest: int,
    k_dV: float,
    dPbus_dV: np.ndarray,
    dQbus_dV: np.ndarray,
    indep_jac_cols: np.ndarray,
    layout: str,
) -> np.ndarray:
    """
    [FIX] Strict subspace solve: build A in Jacobian coordinate, solve only indep_jac_cols,
    then lift to full 2Nbus.
    """
    nbus = int(ctx.Nbus)
    bus_slack = int(ctx.bus_slack)
    jac_dim = int(dPbus_dV.shape[1])
    full_dim = 2 * nbus

    dV_full = np.zeros((num_viotest, full_dim), dtype=float)

    j = 0
    for i in range(int(ctx.Ntest)):
        if (lsidxPg[i] + lsidxQg[i]) <= 0:
            continue

        if lsidxPg[i] > 0 and lsidxQg[i] > 0:
            idxPg = lsPg[lsidxPg[i] - 1][:, 0].astype(np.int32)
            idxQg = lsQg[lsidxQg[i] - 1][:, 0].astype(np.int32)
            busPg = ctx.bus_Pg[idxPg]
            busQg = ctx.bus_Qg[idxQg]
            A = np.concatenate((dPbus_dV[busPg, :], dQbus_dV[busQg, :]), axis=0)
            b = np.concatenate((lsPg[lsidxPg[i] - 1][:, 1], lsQg[lsidxQg[i] - 1][:, 1]), axis=0)
        elif lsidxPg[i] > 0:
            idxPg = lsPg[lsidxPg[i] - 1][:, 0].astype(np.int32)
            busPg = ctx.bus_Pg[idxPg]
            A = dPbus_dV[busPg, :]
            b = lsPg[lsidxPg[i] - 1][:, 1]
        else:
            idxQg = lsQg[lsidxQg[i] - 1][:, 0].astype(np.int32)
            busQg = ctx.bus_Qg[idxQg]
            A = dQbus_dV[busQg, :]
            b = lsQg[lsidxQg[i] - 1][:, 1]

        A = np.atleast_2d(A)
        A_sub = A[:, indep_jac_cols]
        dv_sub = np.dot(np.linalg.pinv(A_sub), np.asarray(b).ravel() * k_dV)

        dV_full[j] = _lift_sub_to_full(
            dv_sub, indep_jac_cols,
            nbus=nbus, bus_slack=bus_slack, layout=layout, jac_dim=jac_dim
        )
        j += 1

    return dV_full


def get_dV_subspace_mapped(
    ctx: EvalContext,
    *,
    Pred_V: np.ndarray,
    lsPg, lsQg, lsidxPg, lsidxQg,
    num_viotest: int,
    k_dV: float,
    indep_jac_cols: np.ndarray,
    layout: str,
) -> np.ndarray:
    """
    [FIX] Subspace version for per-sample Jacobian solve (mirrors your get_dV style, but mapped).
    """
    nbus = int(ctx.Nbus)
    bus_slack = int(ctx.bus_slack)
    full_dim = 2 * nbus

    # We'll build per-sample Jacobian in *full* Va+Vm space first, then map to "layout" if needed.
    # NOTE: This is for consistency; if your utils.get_dV is used in supervised, keep it there.
    dV_full = np.zeros((num_viotest, full_dim), dtype=float)

    # helper: map full Jacobian (2Nbus cols) -> Jacobian layout (2Nbus or 2Nbus-1)
    def fullJ_to_layoutJ(fullJ: np.ndarray) -> np.ndarray:
        fullJ = np.asarray(fullJ)
        if layout == "full":
            return fullJ
        # layout noslack: drop slack angle column from Va block
        # full columns: [Va0..Va(N-1), Vm0..Vm(N-1)]
        # drop Va_slack => total cols 2N-1
        keep = [c for c in range(2 * nbus) if c != bus_slack]
        return fullJ[:, keep]

    j = 0
    for i in range(int(ctx.Ntest)):
        if (lsidxPg[i] + lsidxQg[i]) <= 0:
            continue

        V = Pred_V[i].copy()

        # per-sample Jacobian (Va_full + Vm_full) => 2Nbus
        Ibus = ctx.Ybus.dot(V).conj()
        diagV = np.diag(V)
        diagIbus = np.diag(Ibus)
        diagVnorm = np.diag(V / (np.abs(V) + 1e-12))

        dSbus_dVm = np.dot(diagV, ctx.Ybus.dot(diagVnorm).conj()) + np.dot(diagIbus.conj(), diagVnorm)
        dSbus_dVa = 1j * np.dot(diagV, (diagIbus - ctx.Ybus.dot(diagV)).conj())
        dSbus_dV = np.concatenate((dSbus_dVa, dSbus_dVm), axis=1)  # full 2Nbus
        dPbus_dV_full = np.real(dSbus_dV)
        dQbus_dV_full = np.imag(dSbus_dV)

        # map to layout Jacobian
        dPbus_dV = fullJ_to_layoutJ(dPbus_dV_full)
        dQbus_dV = fullJ_to_layoutJ(dQbus_dV_full)
        jac_dim = dPbus_dV.shape[1]

        if lsidxPg[i] > 0 and lsidxQg[i] > 0:
            idxPg = lsPg[lsidxPg[i] - 1][:, 0].astype(np.int32)
            idxQg = lsQg[lsidxQg[i] - 1][:, 0].astype(np.int32)
            busPg = ctx.bus_Pg[idxPg]
            busQg = ctx.bus_Qg[idxQg]
            A = np.concatenate((dPbus_dV[busPg, :], dQbus_dV[busQg, :]), axis=0)
            b = np.concatenate((lsPg[lsidxPg[i] - 1][:, 1], lsQg[lsidxQg[i] - 1][:, 1]), axis=0)
        elif lsidxPg[i] > 0:
            idxPg = lsPg[lsidxPg[i] - 1][:, 0].astype(np.int32)
            busPg = ctx.bus_Pg[idxPg]
            A = dPbus_dV[busPg, :]
            b = lsPg[lsidxPg[i] - 1][:, 1]
        else:
            idxQg = lsQg[lsidxQg[i] - 1][:, 0].astype(np.int32)
            busQg = ctx.bus_Qg[idxQg]
            A = dQbus_dV[busQg, :]
            b = lsQg[lsidxQg[i] - 1][:, 1]

        A = np.atleast_2d(A)
        A_sub = A[:, indep_jac_cols]
        dv_sub = np.dot(np.linalg.pinv(A_sub), np.asarray(b).ravel() * k_dV)

        # lift
        dV_full[j] = _lift_sub_to_full(
            dv_sub, indep_jac_cols,
            nbus=nbus, bus_slack=bus_slack, layout=layout, jac_dim=jac_dim
        )
        j += 1

    return dV_full


# =========================
# Post-processing
# =========================

def post_process_like_evaluate_model(
    ctx: EvalContext,
    Pred_Vm_full: np.ndarray,
    Pred_Va_full: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    """
    Revised post-processing:
    Allows NGT/Flow to use full-space correction to achieve better feasibility.
    """
    t0 = time.perf_counter()

    # ===== [CONFIG] Decide whether to use strict subspace or relax to full space =====
    # 默认建议设为 True：即便是 NGT 模型，也使用全变量修正
    relax_ngt_post = getattr(ctx.config, "relax_ngt_post", True) 
    
    # 只有当模型包含重构信息，且配置要求严格子空间时，才使用 NGT 专用逻辑
    use_strict_subspace = (ctx.bus_Pnet_all is not None) and (not relax_ngt_post)

    # ===== compute PF/violations PRE =====
    Pred_V = Pred_Vm_full * np.exp(1j * Pred_Va_full)
    Pred_Pg, Pred_Qg, _, _ = get_genload(
        Pred_V, ctx.Pdtest, ctx.Qdtest, ctx.bus_Pg, ctx.bus_Qg, ctx.Ybus
    )

    lsPg, lsQg, lsidxPg, lsidxQg, _, vio_PQg, _, _, _, _ = get_vioPQg(
        Pred_Pg, ctx.bus_Pg, ctx.MAXMIN_Pg,
        Pred_Qg, ctx.bus_Qg, ctx.MAXMIN_Qg,
        ctx.DELTA
    )

    lsidxPQg = np.asarray(np.where((lsidxPg + lsidxQg) > 0)[0]).ravel()
    num_viotest = int(lsidxPQg.size)

    vio_branang, vio_branpf, deltapf, vio_branpfidx, lsSf, _, lsSf_sampidx, _ = get_viobran2(
        Pred_V, Pred_Va_full, ctx.branch, ctx.Yf, ctx.Yt, ctx.BRANFT, ctx.baseMVA, ctx.DELTA
    )
    vio_branpf_num = int(np.sum(np.asarray(vio_branpfidx) > 0))
    lsSf_sampidx = np.asarray(lsSf_sampidx, dtype=int)

    if num_viotest == 0:
        return Pred_Vm_full, Pred_Va_full, 0.0, {
            "num_viotest": 0,
            "vio_branpf_num": vio_branpf_num,
            "layout": None
        }

    # ===== Jacobians from current or historical voltage =====
    # [IMPROVEMENT] Use current voltage if available, otherwise fall back to historical
    # This ensures more accurate linearization when voltage deviates significantly
    if hasattr(ctx, 'current_V') and ctx.current_V is not None:
        # Use current voltage for more accurate Jacobian
        current_V_complex = ctx.current_V
        dPbus_dV, dQbus_dV = dPQbus_dV(current_V_complex, ctx.bus_Pg, ctx.bus_Qg, ctx.Ybus)
        finc = _build_finc(ctx.branch, ctx.Nbus)
        bus_Va = np.delete(np.arange(ctx.Nbus), ctx.bus_slack)
        dPfbus_dV, dQfbus_dV = dSlbus_dV(current_V_complex, bus_Va, ctx.branch, ctx.Yf, finc, ctx.BRANFT, ctx.Nbus)
    else:
        # Fall back to historical voltage (original behavior)
        dPbus_dV, dQbus_dV = dPQbus_dV(ctx.his_V, ctx.bus_Pg, ctx.bus_Qg, ctx.Ybus)
        finc = _build_finc(ctx.branch, ctx.Nbus)
        bus_Va = np.delete(np.arange(ctx.Nbus), ctx.bus_slack) 
        dPfbus_dV, dQfbus_dV = dSlbus_dV(ctx.his_V, bus_Va, ctx.branch, ctx.Yf, finc, ctx.BRANFT, ctx.Nbus)

    # Infer Jacobian layout
    jac_dim = int(np.atleast_2d(dPbus_dV).shape[1])
    layout = _infer_jac_layout(int(ctx.Nbus), jac_dim)
    
    # Only calculate subspace columns if strictly needed
    indep_jac_cols = None
    if use_strict_subspace:
        indep_jac_cols = _make_indep_jac_cols(ctx, layout=layout, jac_dim=jac_dim)

    # ===== compute dV1_full =====
    if not use_strict_subspace:
        # [MODIFIED] Logic A: Full Space Correction (Supervised & Relaxed NGT)
        # This gives the solver freedom to move ANY voltage to fix PQ violations.
        if ctx.flag_hisv:
            dV1_full = np.asarray(get_hisdV(
                lsPg, lsQg, lsidxPg, lsidxQg,
                num_viotest, ctx.k_dV,
                ctx.bus_Pg, ctx.bus_Qg, dPbus_dV, dQbus_dV,
                ctx.Nbus, ctx.Ntest
            ))
        else:
            # [IMPROVEMENT] Use current voltage for get_dV if available
            # This ensures Jacobian is computed at the current operating point
            V_for_dV = Pred_V if hasattr(ctx, 'current_V') and ctx.current_V is not None else ctx.his_V
            dV1_full = np.asarray(get_dV(
                Pred_V, lsPg, lsQg, lsidxPg, lsidxQg,
                num_viotest, ctx.k_dV,
                ctx.bus_Pg, ctx.bus_Qg, ctx.Ybus, V_for_dV
            ))
    else:
        # [MODIFIED] Logic B: Strict Subspace Correction (Original NGT)
        # Constrains movement to the manifold. Often yields worse feasibility.
        if ctx.flag_hisv:
            dV1_full = get_hisdV_subspace_mapped(
                ctx,
                lsPg=lsPg, lsQg=lsQg, lsidxPg=lsidxPg, lsidxQg=lsidxQg,
                num_viotest=num_viotest,
                k_dV=ctx.k_dV,
                dPbus_dV=np.asarray(dPbus_dV),
                dQbus_dV=np.asarray(dQbus_dV),
                indep_jac_cols=indep_jac_cols,
                layout=layout
            )
        else:
            dV1_full = get_dV_subspace_mapped(
                ctx,
                Pred_V=Pred_V,
                lsPg=lsPg, lsQg=lsQg, lsidxPg=lsidxPg, lsidxQg=lsidxQg,
                num_viotest=num_viotest,
                k_dV=ctx.k_dV,
                indep_jac_cols=indep_jac_cols,
                layout=layout
            )

    # ===== branch correction =====
    if vio_branpf_num > 0 and lsSf_sampidx.size > 0:
        nbus = int(ctx.Nbus)
        bus_slack = int(ctx.bus_slack)
        full_dim = 2 * nbus

        dV_branch_raw = np.zeros((lsSf_sampidx.shape[0], full_dim), dtype=float)

        for i in range(lsSf_sampidx.shape[0]):
            mp = np.array(lsSf[i][:, 2] / (lsSf[i][:, 1] + 1e-12)).reshape(-1, 1)
            mq = np.array(lsSf[i][:, 3] / (lsSf[i][:, 1] + 1e-12)).reshape(-1, 1)
            br_idx = np.asarray(lsSf[i][:, 0], dtype=int).ravel()

            dPdV = np.atleast_2d(dPfbus_dV[br_idx, :])
            dQdV = np.atleast_2d(dQfbus_dV[br_idx, :])

            # If using relaxed/full mode, use all columns. If strict, use subset.
            if not use_strict_subspace:
                use_cols = np.arange(dPdV.shape[1], dtype=int)
            else:
                use_cols = indep_jac_cols

            dPdV_sub = dPdV[:, use_cols]
            dQdV_sub = dQdV[:, use_cols]

            dmp = mp * dPdV_sub
            dmq = mq * dQdV_sub

            dmpq_inv = np.linalg.pinv(dmp + dmq)
            dv_sub = np.dot(dmpq_inv, np.array(lsSf[i][:, 1])).ravel()

            # Lift back to full
            jac_vec = np.zeros((dPdV.shape[1],), dtype=float)
            jac_vec[use_cols] = dv_sub
            dv_full = _jacvec_to_full(jac_vec, nbus=nbus, bus_slack=bus_slack, layout=layout)

            dV_branch_raw[i] = dv_full

        # Align to dV1_full rows
        dV_branch_aligned = np.zeros_like(dV1_full)
        for j, samp_idx in enumerate(lsSf_sampidx.tolist()):
            pos = np.where(lsidxPQg == samp_idx)[0]
            if pos.size > 0:
                dV_branch_aligned[pos[0], :] = dV_branch_raw[j, :]

        dV1_full = dV1_full + dV_branch_aligned

    # ===== apply corrections =====
    Pred_Va1 = Pred_Va_full.copy()
    Pred_Vm1 = Pred_Vm_full.copy()

    Pred_Va1[lsidxPQg, :] = Pred_Va_full[lsidxPQg, :] - dV1_full[:, 0:ctx.Nbus]
    Pred_Va1[:, ctx.bus_slack] = 0.0
    Pred_Vm1[lsidxPQg, :] = Pred_Vm_full[lsidxPQg, :] - dV1_full[:, ctx.Nbus:2 * ctx.Nbus]

    Pred_Vm1_clip = get_clamp(_as_torch(Pred_Vm1), ctx.hisVm_min, ctx.hisVm_max).detach().cpu().numpy()

    # [CRITICAL CHANGE]
    # Only run Kron reconstruction if we were strictly solving in subspace.
    # If we relaxed to full space, applying Kron reconstruction here would 
    # UNDO our physics corrections and snap back to the (infeasible) manifold.
    if use_strict_subspace and ctx.param_ZIMV is not None:
        Pred_Vm1_clip, Pred_Va1 = _kron_reconstruct_zib(
            Pred_Vm1_clip, Pred_Va1,
            bus_Pnet_all=_ensure_1d_int(ctx.bus_Pnet_all),
            bus_ZIB_all=_ensure_1d_int(ctx.bus_ZIB_all),
            param_ZIMV=np.asarray(ctx.param_ZIMV),
        )

    t = time.perf_counter() - t0
    dbg = {
        "num_viotest": num_viotest,
        "vio_branpf_num": vio_branpf_num,
        "layout": layout,
        "jac_dim": jac_dim,
        "mode": "strict_subspace" if use_strict_subspace else "relaxed_full_space"
    }
    return Pred_Vm1_clip, Pred_Va1, t, dbg


# =========================
# Unified evaluation
# =========================

def _compute_cost(Pg, ctx: EvalContext):
    """
    Compute generation cost using hybrid approach:
    - If ctx.gencost_Pg is available, use it directly (consistent with ORIGINAL method)
    - Otherwise, fall back to get_Pgcost (for supervised scenarios)
    
    Args:
        Pg: Active generation (p.u.) [Ntest, len(bus_Pg)]
        ctx: EvalContext
        
    Returns:
        cost: Total generation cost for each sample [Ntest]
    """
    if ctx.gencost_Pg is not None:
        # Use pre-extracted gencost_Pg (consistent with ORIGINAL method)
        # gencost_Pg shape: [len(bus_Pg), 2] where columns are [c2, c1]
        PgMVA = Pg * ctx.baseMVA
        cost = ctx.gencost_Pg[:, 0] * (PgMVA ** 2) + ctx.gencost_Pg[:, 1] * np.abs(PgMVA)
        return np.sum(cost, axis=1)
    else:
        # Fall back to get_Pgcost (for supervised scenarios or when gencost_Pg not available)
        return get_Pgcost(Pg, ctx.idxPg, ctx.gencost, ctx.baseMVA)


def _compute_carbon(Pg, ctx: EvalContext):
    """
    Compute carbon emission for predictions.
    
    Args:
        Pg: Active generation (p.u.) [Ntest, len(bus_Pg)]
        ctx: EvalContext (must have gci_values set)
        
    Returns:
        carbon: Carbon emission for each sample [Ntest] (tCO2/h)
    """
    if ctx.gci_values is None:
        # Return zeros if GCI values not available
        return np.zeros(Pg.shape[0])
    
    return get_carbon_emission_vectorized(Pg, ctx.gci_values, ctx.baseMVA)


def evaluate_unified(
    ctx: EvalContext,
    predictor: Union[SupervisedPredictor, NGTPredictor, NGTFlowPredictor],
    *,
    apply_post_processing: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Unified evaluation compatible with evaluate_model outputs.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Unified Evaluation")
        print("=" * 60)

    pred_pack: PredPack = predictor.predict(ctx)
    Pred_Vm_full = pred_pack.Pred_Vm_full
    Pred_Va_full = pred_pack.Pred_Va_full

    # -------- PF + constraints (pre) --------
    t_pq0 = time.perf_counter()
    Pred_V = Pred_Vm_full * np.exp(1j * Pred_Va_full)
    Pred_Pg, Pred_Qg, Pred_Pd, Pred_Qd = get_genload(
        Pred_V, ctx.Pdtest, ctx.Qdtest, ctx.bus_Pg, ctx.bus_Qg, ctx.Ybus
    )
    t_pq = time.perf_counter() - t_pq0

    # ground truth PF for load/cost
    Real_V = ctx.Real_Vm_full * np.exp(1j * ctx.Real_Va_full)
    Real_Pg, Real_Qg, Real_Pd, Real_Qd = get_genload(
        Real_V, ctx.Pdtest, ctx.Qdtest, ctx.bus_Pg, ctx.bus_Qg, ctx.Ybus
    )

    # violations pre
    lsPg, lsQg, lsidxPg, lsidxQg, vio_PQgmaxmin, vio_PQg, deltaPgL, deltaPgU, deltaQgL, deltaQgU = get_vioPQg(
        Pred_Pg, ctx.bus_Pg, ctx.MAXMIN_Pg,
        Pred_Qg, ctx.bus_Qg, ctx.MAXMIN_Qg,
        ctx.DELTA
    )
    lsidxPQg = np.asarray(np.where((lsidxPg + lsidxQg) > 0)[0]).ravel()
    num_viotest = int(lsidxPQg.size)

    vio_branang, vio_branpf, deltapf, vio_branpfidx, lsSf, _, lsSf_sampidx, _ = get_viobran2(
        Pred_V, Pred_Va_full, ctx.branch, ctx.Yf, ctx.Yt, ctx.BRANFT, ctx.baseMVA, ctx.DELTA
    )

    # -------- Cost and Carbon calculation --------
    Pred_cost = _compute_cost(Pred_Pg, ctx)
    Real_cost = _compute_cost(Real_Pg, ctx)
    Pred_carbon = _compute_carbon(Pred_Pg, ctx)
    Real_carbon = _compute_carbon(Real_Pg, ctx)
    

    # -------- metrics pre --------
    Pred_Va_noslack = _remove_slack_va(Pred_Va_full, ctx.bus_slack)
    mae_Vmtest = _to_float(get_mae(ctx.yvmtests, _as_torch(Pred_Vm_full)))
    mae_Vatest = _to_float(get_mae(ctx.yvatests_noslack, _as_torch(Pred_Va_noslack)))

    mre_Pd = _to_float(get_rerr(_as_torch(Real_Pd.sum(axis=1)), _as_torch(Pred_Pd.sum(axis=1))))
    mre_Qd = _to_float(get_rerr(_as_torch(Real_Qd.sum(axis=1)), _as_torch(Pred_Qd.sum(axis=1))))

    # Cost already computed above
    mre_cost = _to_float(get_rerr2(_as_torch(Real_cost), _as_torch(Pred_cost)))

    if verbose:
        print("\n[Prediction Accuracy (Before Post-Processing)]")
        print(f"  Vm MAE: {mae_Vmtest:.6f}")
        print(f"  Va MAE: {mae_Vatest:.6f}")
        print(f"  Cost MRE: {mre_cost:.4f}%")
        print(f"  Pd MRE: {mre_Pd:.4f}%")
        print(f"  Qd MRE: {mre_Qd:.4f}%")
        print("\n[Constraint Violations (Before Post-Processing)]")
        print(f"  Violated samples: {num_viotest}/{ctx.Ntest} ({num_viotest/ctx.Ntest*100:.1f}%)")
        print(f"  Pg constraint satisfaction: {float(np.mean(_as_numpy(vio_PQg[:, 0]))):.2f}%")
        print(f"  Qg constraint satisfaction: {float(np.mean(_as_numpy(vio_PQg[:, 1]))):.2f}%")
        print(f"  Branch angle constraint: {float(np.mean(_as_numpy(vio_branang))):.2f}%")
        print(f"  Branch power flow constraint: {float(np.mean(_as_numpy(vio_branpf))):.2f}%")

    # -------- post-processing --------
    time_post = 0.0
    post_dbg = {}
    if apply_post_processing:
        Pred_Vm1, Pred_Va1, time_post, post_dbg = post_process_like_evaluate_model(
            ctx, Pred_Vm_full, Pred_Va_full
        )
        Pred_V1 = Pred_Vm1 * np.exp(1j * Pred_Va1)
        Pred_Pg1, Pred_Qg1, Pred_Pd1, Pred_Qd1 = get_genload(
            Pred_V1, ctx.Pdtest, ctx.Qdtest, ctx.bus_Pg, ctx.bus_Qg, ctx.Ybus
        )

        _, _, lsidxPg1, lsidxQg1, _, vio_PQg1, _, _, _, _ = get_vioPQg(
            Pred_Pg1, ctx.bus_Pg, ctx.MAXMIN_Pg,
            Pred_Qg1, ctx.bus_Qg, ctx.MAXMIN_Qg,
            ctx.DELTA
        )
        lsidxPQg1 = np.asarray(np.where(lsidxPg1 + lsidxQg1 > 0)[0]).ravel()
        num_viotest1 = int(lsidxPQg1.size)

        vio_branang1, vio_branpf1, deltapf1 = get_viobran(
            Pred_V1, Pred_Va1, ctx.branch, ctx.Yf, ctx.Yt, ctx.BRANFT, ctx.baseMVA, ctx.DELTA
        )

        mae_Vmtest1 = _to_float(get_mae(ctx.yvmtests, _as_torch(Pred_Vm1)))

        # [FIX] Va MAE after post-processing must also be no-slack (consistent!)
        Pred_Va1_noslack = _remove_slack_va(Pred_Va1, ctx.bus_slack)
        mae_Vatest1 = _to_float(get_mae(ctx.yvatests_noslack, _as_torch(Pred_Va1_noslack)))

        # Use hybrid approach: prefer gencost_Pg if available, fall back to get_Pgcost
        Pred_cost1 = _compute_cost(Pred_Pg1, ctx)
        Pred_carbon1 = _compute_carbon(Pred_Pg1, ctx)
        mre_cost1 = _to_float(get_rerr2(_as_torch(Real_cost), _as_torch(Pred_cost1)))

        if verbose:
            print("\n[Prediction Accuracy (After Post-Processing)]")
            print(f"  Vm MAE: {mae_Vmtest1:.6f}")
            print(f"  Va MAE: {mae_Vatest1:.6f}")
            print(f"  Cost MRE: {mre_cost1:.4f}%")
            print("\n[Constraint Violations (After Post-Processing)]")
            print(f"  Violated samples: {num_viotest1}/{ctx.Ntest} ({num_viotest1/ctx.Ntest*100:.1f}%)")
            print(f"  Pg constraint satisfaction: {float(np.mean(_as_numpy(vio_PQg1[:, 0]))):.2f}%")
            print(f"  Qg constraint satisfaction: {float(np.mean(_as_numpy(vio_PQg1[:, 1]))):.2f}%")
            print(f"  Branch angle constraint: {float(np.mean(_as_numpy(vio_branang1))):.2f}%")
            print(f"  Branch power flow constraint: {float(np.mean(_as_numpy(vio_branpf1))):.2f}%")

    else:
        Pred_Vm1, Pred_Va1 = Pred_Vm_full, Pred_Va_full
        mae_Vmtest1, mae_Vatest1 = mae_Vmtest, mae_Vatest
        vio_PQg1, vio_branang1, vio_branpf1, mre_cost1, deltapf1 = vio_PQg, vio_branang, vio_branpf, mre_cost, deltapf
        Pred_cost1, Pred_carbon1 = Pred_cost, Pred_carbon
        Pred_Pg1 = Pred_Pg

    # -------- timing_info --------
    time_NN_total = float(pred_pack.time_nn_total)
    time_NN_per_sample = time_NN_total / ctx.Ntest * 1000.0
    time_total_with_post = time_NN_total + float(t_pq) + float(time_post)
    time_total_per_sample = time_total_with_post / ctx.Ntest * 1000.0

    timing_info = {
        "model_type": getattr(predictor, "model_type", predictor.__class__.__name__),
        "num_test_samples": int(ctx.Ntest),
        "time_Vm_prediction": float(pred_pack.time_vm),
        "time_Va_prediction": float(pred_pack.time_va),
        "time_NN_total": float(time_NN_total),
        "time_PQ_calculation": float(t_pq),
        "time_post_processing": float(time_post),
        "time_total_with_post": float(time_total_with_post),
        "time_NN_per_sample_ms": float(time_NN_per_sample),
        "time_total_per_sample_ms": float(time_total_per_sample),
        "post_debug": post_dbg,
    }

    results = {
        "mae_Vmtest": mae_Vmtest,
        "mae_Vatest": mae_Vatest,
        "mae_Vmtest1": mae_Vmtest1,
        "mae_Vatest1": mae_Vatest1,

        "vio_PQg": vio_PQg,
        "vio_PQg1": vio_PQg1,

        "vio_branang": vio_branang,
        "vio_branpf": vio_branpf,
        "vio_branang1": vio_branang1,
        "vio_branpf1": vio_branpf1,

        "mre_cost": mre_cost,
        "mre_cost1": mre_cost1,

        "mre_Pd": mre_Pd,
        "mre_Qd": mre_Qd,

        "deltaPgL": deltaPgL,
        "deltaPgU": deltaPgU,
        "deltaQgL": deltaQgL,
        "deltaQgU": deltaQgU,

        "deltapf": deltapf,
        "deltapf1": deltapf1,

        "timing_info": timing_info,
        
        # Store predictions for batch evaluation
        "Pred_Vm_full": Pred_Vm_full,
        "Pred_Va_full": Pred_Va_full,
        "Pred_Vm1": Pred_Vm1 if apply_post_processing else Pred_Vm_full,
        "Pred_Va1": Pred_Va1 if apply_post_processing else Pred_Va_full,
        "Pred_Pg": Pred_Pg,
        "Pred_Pg1": Pred_Pg1 if apply_post_processing else Pred_Pg,
        "Pred_cost": Pred_cost,
        "Pred_cost1": Pred_cost1 if apply_post_processing else Pred_cost,
        "Real_cost": Real_cost,
        "num_viotest": num_viotest,
        "num_viotest1": num_viotest1 if apply_post_processing else num_viotest,
        
        # Carbon emission metrics
        "Pred_carbon": Pred_carbon,
        "Pred_carbon1": Pred_carbon1 if apply_post_processing else Pred_carbon,
        "Real_carbon": Real_carbon,
        
        # Aggregated metrics for Pareto analysis
        "cost_mean": float(np.mean(_as_numpy(Pred_cost))),
        "cost_mean1": float(np.mean(_as_numpy(Pred_cost1 if apply_post_processing else Pred_cost))),
        "carbon_mean": float(np.mean(_as_numpy(Pred_carbon))),
        "carbon_mean1": float(np.mean(_as_numpy(Pred_carbon1 if apply_post_processing else Pred_carbon))),
        "Real_cost_mean": float(np.mean(_as_numpy(Real_cost))),
        "Real_carbon_mean": float(np.mean(_as_numpy(Real_carbon))),
    }
    return results


# =========================
# Pareto Front Analysis
# =========================

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def plot_pareto_front_extended(results, ref_point, hypervolumes, 
                                save_path='results/pareto_front_comparison.png'):
    """
    Plot Pareto front comparing all model categories with distinct styles.
    
    Args:
        results: List of dicts with 'name', 'cost_mean', 'carbon_mean', 'category'
        ref_point: Reference point [cost_ref, carbon_ref] for hypervolume
        hypervolumes: Dict of hypervolume values for different model groups
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define color scheme and markers by category
    category_styles = {
        'supervised': {'color': '#E74C3C', 'marker': 's', 'label': 'Supervised (MLP/VAE)'},
        'unsupervised': {'color': '#3498DB', 'marker': 'o', 'label': 'Unsupervised (NGT MLP)'},
        'flow': {'color': '#27AE60', 'marker': '^', 'label': 'Rectified Flow'},
    }
    
    # Group results by category
    categories = {}
    for r in results:
        cat = r.get('category', 'unsupervised')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)
    
    # Plot each category
    legend_handles = []
    for cat, style in category_styles.items():
        if cat not in categories:
            continue
        
        cat_results = categories[cat]
        costs = np.array([r['cost_mean'] for r in cat_results])
        carbons = np.array([r['carbon_mean'] for r in cat_results])
        names = [r['name'] for r in cat_results]
        
        scatter = ax.scatter(costs, carbons, c=style['color'], marker=style['marker'], 
                            s=200, label=style['label'], zorder=3, 
                            edgecolors='black', linewidths=1.5, alpha=0.85)
        legend_handles.append(scatter)
        
        # Connect points within category to show Pareto front (sorted by cost)
        if len(costs) > 1:
            sorted_idx = np.argsort(costs)
            ax.plot(costs[sorted_idx], carbons[sorted_idx], 
                   color=style['color'], linestyle='--', alpha=0.4, linewidth=2)
        
        # Add annotations
        for name, cost, carbon in zip(names, costs, carbons):
            # Create shorter name for annotation
            short_name = name.replace('NGT_', '').replace('Flow_', 'F_').replace('_single', '').replace('_prog', '_P')
            ax.annotate(short_name, (cost, carbon), textcoords="offset points", 
                       xytext=(8, 5), fontsize=9, alpha=0.85, fontweight='medium')
    
    # Plot reference point
    ax.scatter(ref_point[0], ref_point[1], c='gray', marker='X', s=250, 
              label='Reference Point', zorder=2, edgecolors='black', linewidths=1.5)
    
    # Labels and formatting
    ax.set_xlabel('Economic Cost ($/h)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Carbon Emission (tCO2/h)', fontsize=14, fontweight='bold')
    ax.set_title('Pareto Front Comparison: Supervised vs Unsupervised vs Flow Models', 
                fontsize=16, fontweight='bold', pad=15)
    
    # Add legend with hypervolume info
    legend_labels = []
    for cat, style in category_styles.items():
        if cat in categories:
            hv = hypervolumes.get(cat, 0)
            legend_labels.append(f"{style['label']} (HV={hv:.2f})")
    
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add hypervolume text box
    hv_text = "Hypervolumes:\n"
    for cat, hv in hypervolumes.items():
        cat_name = category_styles.get(cat, {}).get('label', cat)
        hv_text += f"  {cat_name}: {hv:.2f}\n"
    hv_text += f"  Total: {hypervolumes.get('all', 0):.2f}"
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.02, 0.98, hv_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props, fontfamily='monospace')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPareto front saved to: {save_path}")
    plt.close()


def print_metrics_table(results, title="Evaluation Results"):
    """
    Print complete metrics table for all evaluated models.
    
    Args:
        results: List of result dicts with all metrics
        title: Table title
    """
    print("\n" + "=" * 150)
    print(f" {title}")
    print("=" * 150)
    
    # Define metrics to display
    metrics = [
        ('Cost ($/h)', 'cost_mean', '.2f'),
        ('Carbon (tCO2/h)', 'carbon_mean', '.4f'),
        ('Vm MAE', 'mae_Vm', '.6f'),
        ('Va MAE', 'mae_Va', '.6f'),
        ('Cost Err%', 'cost_error_percent', '.2f'),
        ('Pg Sat%', 'Pg_satisfy', '.2f'),
        ('Qg Sat%', 'Qg_satisfy', '.2f'),
        ('Vm Sat%', 'Vm_satisfy', '.2f'),
        ('BrAng Sat%', 'branch_ang_satisfy', '.2f'),
        ('Violated', 'num_violated', 'd'),
        ('Time(ms)', 'inference_time_ms', '.3f'),
    ]
    
    # Calculate column widths
    name_width = max(20, max(len(r['name']) for r in results) + 2)
    
    # Print header
    header = f"{'Model':<{name_width}} {'Category':<12} {'lambda':<8}"
    for metric_name, _, _ in metrics:
        header += f" {metric_name:>12}"
    print(header)
    print("-" * 150)
    
    # Sort by category then by cost
    sorted_results = sorted(results, key=lambda x: (x.get('category', 'z'), x['cost_mean']))
    
    current_category = None
    for r in sorted_results:
        cat = r.get('category', 'unknown')
        if cat != current_category:
            if current_category is not None:
                print("-" * 150)
            current_category = cat
        
        lc = f"{r['lambda_cost']:.1f}" if r.get('lambda_cost') is not None else "N/A"
        row = f"{r['name']:<{name_width}} {cat:<12} {lc:<8}"
        
        for _, key, fmt in metrics:
            val = r.get(key, 'N/A')
            if val != 'N/A':
                if fmt == 'd':
                    row += f" {int(val):>12}"
                else:
                    row += f" {val:>12{fmt}}"
            else:
                row += f" {'N/A':>12}"
        print(row)
    
    print("-" * 150)


# =========================
# Batch Evaluation Helpers
# =========================

def extract_summary_metrics(
    eval_result: Dict[str, Any],
    model_name: str,
    category: str = "unsupervised",
    lambda_cost: Optional[float] = None,
    use_post_processed: bool = True,
) -> Dict[str, Any]:
    """
    Extract summary metrics from evaluate_unified result for Pareto analysis.
    
    Args:
        eval_result: Result dict from evaluate_unified
        model_name: Name of the model
        category: Category of the model ('supervised', 'unsupervised', 'flow')
        lambda_cost: Cost preference weight (for multi-objective models)
        use_post_processed: Whether to use post-processed metrics
        
    Returns:
        Summary dict suitable for Pareto analysis and metrics table
    """
    suffix = "1" if use_post_processed else ""
    
    # Get VmLb/VmUb from config
    config = get_config()
    VmLb = getattr(config, 'ngt_VmLb', 0.94)
    VmUb = getattr(config, 'ngt_VmUb', 1.06)
    
    # Compute Vm satisfaction
    Pred_Vm = eval_result.get(f"Pred_Vm{suffix}", eval_result.get("Pred_Vm_full"))
    if Pred_Vm is not None:
        Pred_Vm = _as_numpy(Pred_Vm)
        Vm_satisfy = 100 - np.mean(Pred_Vm > VmUb) * 100 - np.mean(Pred_Vm < VmLb) * 100
    else:
        Vm_satisfy = 100.0
    
    # Extract constraint satisfaction
    vio_PQg = eval_result.get(f"vio_PQg{suffix}", eval_result.get("vio_PQg"))
    vio_branang = eval_result.get(f"vio_branang{suffix}", eval_result.get("vio_branang"))
    vio_branpf = eval_result.get(f"vio_branpf{suffix}", eval_result.get("vio_branpf"))
    
    # Get timing info
    timing = eval_result.get("timing_info", {})
    
    return {
        "name": model_name,
        "model_type": category,
        "category": category,
        "lambda_cost": lambda_cost,
        "lambda_carbon": 1.0 - lambda_cost if lambda_cost is not None else None,
        
        # Cost and carbon
        "cost_mean": eval_result.get(f"cost_mean{suffix}", eval_result.get("cost_mean", 0.0)),
        "carbon_mean": eval_result.get(f"carbon_mean{suffix}", eval_result.get("carbon_mean", 0.0)),
        
        # MAE metrics
        "mae_Vm": eval_result.get(f"mae_Vmtest{suffix}", eval_result.get("mae_Vmtest", 0.0)),
        "mae_Va": eval_result.get(f"mae_Vatest{suffix}", eval_result.get("mae_Vatest", 0.0)),
        
        # Cost error
        "cost_error_percent": eval_result.get(f"mre_cost{suffix}", eval_result.get("mre_cost", 0.0)),
        
        # Constraint satisfaction
        "Pg_satisfy": float(np.mean(_as_numpy(vio_PQg)[:, 0])) if vio_PQg is not None else 100.0,
        "Qg_satisfy": float(np.mean(_as_numpy(vio_PQg)[:, 1])) if vio_PQg is not None else 100.0,
        "Vm_satisfy": Vm_satisfy,
        "branch_ang_satisfy": float(np.mean(_as_numpy(vio_branang))) if vio_branang is not None else 100.0,
        "branch_pf_satisfy": float(np.mean(_as_numpy(vio_branpf))) if vio_branpf is not None else 100.0,
        
        # Violation counts
        "num_violated": eval_result.get(f"num_viotest{suffix}", eval_result.get("num_viotest", 0)),
        
        # Timing
        "inference_time_ms": timing.get("time_NN_per_sample_ms", 0.0),
        
        # Store Pred_Pg for further analysis if needed
        "Pred_Pg": eval_result.get(f"Pred_Pg{suffix}", eval_result.get("Pred_Pg")),
    }


def compute_pareto_hypervolumes(
    results: list,
    ref_point: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute hypervolumes for different model categories.
    
    Args:
        results: List of summary dicts from extract_summary_metrics
        ref_point: Reference point [cost_ref, carbon_ref]. If None, auto-computed.
        
    Returns:
        Dict with hypervolumes for each category and 'all'
    """
    costs = np.array([r['cost_mean'] for r in results])
    carbons = np.array([r['carbon_mean'] for r in results])
    
    if ref_point is None:
        ref_point = np.array([
            np.max(costs) * 1.1,
            np.max(carbons) * 1.1
        ])
    
    hypervolumes = {}
    
    for category in ['supervised', 'unsupervised', 'flow']:
        cat_results = [r for r in results if r.get('category') == category]
        if len(cat_results) > 0:
            cat_costs = np.array([r['cost_mean'] for r in cat_results])
            cat_carbons = np.array([r['carbon_mean'] for r in cat_results])
            cat_points = np.column_stack([cat_costs, cat_carbons])
            hypervolumes[category] = compute_hypervolume(cat_points, ref_point)
    
    # Total hypervolume
    all_points = np.column_stack([costs, carbons])
    hypervolumes['all'] = compute_hypervolume(all_points, ref_point)
    
    return hypervolumes

 
def save_evaluation_results(
    results: list,
    hypervolumes: Dict[str, float],
    ref_point: np.ndarray,
    save_path: str,
    config: Optional[Any] = None,
):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: List of summary dicts
        hypervolumes: Hypervolume dict
        ref_point: Reference point array
        save_path: Path to save JSON file
        config: Optional config object for additional metadata
    """
    import json
    
    save_data = {
        'models': [{k: v for k, v in r.items() if k != 'Pred_Pg'} for r in results],
        'hypervolumes': hypervolumes,
        'ref_point': ref_point.tolist() if isinstance(ref_point, np.ndarray) else list(ref_point),
    }
    
    if config is not None:
        save_data['config'] = {
            'Nbus': getattr(config, 'Nbus', None),
            'ngt_Epoch': getattr(config, 'ngt_Epoch', None),
        }
    
    save_data = convert_to_serializable(save_data)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {save_path}")