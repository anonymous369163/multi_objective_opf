#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_eval.py - Unified Evaluation for DeepOPF Models

Provides unified evaluation for different model types:
- Supervised: Vm/Va separate models with full-bus output
- NGT: Single model predicts partial non-ZIB; Kron reconstruct ZIB
- Multi-preference: Flow/VAE models with preference conditioning

Author: Peng Yue
Date: 2025-12-18 
"""

from __future__ import annotations
import time
import os 
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, List
import numpy as np
import torch
import matplotlib.pyplot as plt 

from config import get_config, BaseConfig
from utils import (
    dPQbus_dV, dSlbus_dV,
    get_genload, get_vioPQg, get_viobran2,
    get_hisdV, get_dV,
    get_clamp, get_mae, get_rerr, get_Pgcost,
    get_viobran, get_rerr2,
    get_carbon_emission_vectorized, compute_hypervolume,
    get_gci_for_generators,
)

# Optional CBF-QP imports
try:
    from cbf_qp_projection import cbf_active_set_project
    CBF_QP_AVAILABLE = True
except ImportError:
    cbf_active_set_project = None
    CBF_QP_AVAILABLE = False

try:
    from cbf_qp_train_layer_tube import CBFQPProjectorNGT, CBFQPTrainConfig
    CBF_QP_TRAIN_AVAILABLE = True
except ImportError:
    CBFQPProjectorNGT = None
    CBFQPTrainConfig = None
    CBF_QP_TRAIN_AVAILABLE = False


# ==================== Helper Functions ====================

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
    """Convert tensor/ndarray/scalar to python float."""
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return float(x.reshape(-1)[0])
        return float(np.mean(x) if reduce == "mean" else np.median(x))
    if torch.is_tensor(x):
        t = x.detach()
        if t.numel() == 1:
            return float(t.cpu().item())
            return float(t.mean().cpu().item())
    arr = np.asarray(x)
    return float(np.mean(arr) if reduce == "mean" else np.median(arr))

def _insert_slack_va(Va_noslack: np.ndarray, bus_slack: int) -> np.ndarray:
    return np.insert(Va_noslack, bus_slack, values=0.0, axis=1)

def _remove_slack_va(Va_full: np.ndarray, bus_slack: int) -> np.ndarray:
    return np.delete(Va_full, bus_slack, axis=1)

def _build_finc(branch: np.ndarray, nbus: int) -> np.ndarray:
    finc = np.zeros((branch.shape[0], nbus), dtype=float)
    for i in range(branch.shape[0]):
        f = int(branch[i, 0]) - 1
        finc[i, f] = 1.0
    return finc

def _kron_reconstruct_zib(
    Pred_Vm_full: np.ndarray, Pred_Va_full: np.ndarray, *,
    bus_Pnet_all: np.ndarray, bus_ZIB_all: np.ndarray, param_ZIMV: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct ZIB values via Kron reduction: Vy = param_ZIMV @ Vx."""
    bus_Pnet_all = _ensure_1d_int(bus_Pnet_all)
    bus_ZIB_all = _ensure_1d_int(bus_ZIB_all)
    Vx = Pred_Vm_full[:, bus_Pnet_all] * np.exp(1j * Pred_Va_full[:, bus_Pnet_all])
    Vy = (np.asarray(param_ZIMV) @ Vx.T).T
    Pred_Va_full[:, bus_ZIB_all] = np.angle(Vy)
    Pred_Vm_full[:, bus_ZIB_all] = np.abs(Vy)
    return Pred_Vm_full, Pred_Va_full

def get_gci_for_generation_nodes(sys_data, idxPg: np.ndarray) -> np.ndarray:
    """Get GCI values aligned with generation nodes (bus_Pg)."""
    gci_all = get_gci_for_generators(sys_data)
    return gci_all[idxPg]


# ==================== Evaluation Context ====================

@dataclass
class EvalContext:
    """Holds all data needed for evaluation."""
    config: Any
    sys_data: Any
    BRANFT: np.ndarray
    device: torch.device

    x_test: torch.Tensor
    yvmtests: torch.Tensor
    yvatests_noslack: torch.Tensor
    Real_Vm_full: np.ndarray
    Real_Va_full: np.ndarray
    Pdtest: np.ndarray
    Qdtest: np.ndarray
    
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

    idxPg: np.ndarray
    gencost: np.ndarray
    gencost_Pg: Optional[np.ndarray]

    his_V: np.ndarray
    hisVm_min: Union[np.ndarray, float]
    hisVm_max: Union[np.ndarray, float]

    # NGT reconstruction info
    bus_Pnet_all: Optional[np.ndarray] = None
    bus_Pnet_noslack_all: Optional[np.ndarray] = None
    bus_ZIB_all: Optional[np.ndarray] = None
    param_ZIMV: Optional[np.ndarray] = None
    VmLb: Optional[Union[np.ndarray, float]] = None
    VmUb: Optional[Union[np.ndarray, float]] = None

    DELTA: float = 1e-4
    k_dV: float = 1.0
    flag_hisv: bool = True
    relax_ngt_post: bool = False
    gci_values: Optional[np.ndarray] = None


# ==================== Prediction Pack ====================

@dataclass
class PredPack:
    Pred_Vm_full: np.ndarray
    Pred_Va_full: np.ndarray
    time_vm: float = 0.0
    time_va: float = 0.0
    time_nn_total: float = 0.0


# ==================== Context Builders ====================

def build_ctx_from_supervised(config, sys_data, dataloaders, BRANFT, device) -> EvalContext:
    """Build EvalContext for supervised models."""
    yvmtests = sys_data.yvm_test / config.scale_vm * (sys_data.VmUb - sys_data.VmLb) + sys_data.VmLb
    yvatests = sys_data.yva_test / config.scale_va

    Real_Vm_full = yvmtests.clone().cpu().numpy()
    Real_Va_full = _insert_slack_va(yvatests.clone().cpu().numpy(), int(sys_data.bus_slack))

    # NGT fields for compatibility
    bus_Pnet_all = _ensure_1d_int(sys_data.bus_Pnet_all) if hasattr(sys_data, 'bus_Pnet_all') and sys_data.bus_Pnet_all is not None else None
    bus_Pnet_noslack_all = _ensure_1d_int(sys_data.bus_Pnet_noslack_all) if hasattr(sys_data, 'bus_Pnet_noslack_all') and sys_data.bus_Pnet_noslack_all is not None else None
    bus_ZIB_all = _ensure_1d_int(sys_data.bus_ZIB_all) if hasattr(sys_data, 'bus_ZIB_all') and sys_data.bus_ZIB_all is not None else None
    
    # Extract gencost_Pg
    gencost = _as_numpy(sys_data.gencost)
    idxPg = _ensure_1d_int(sys_data.idxPg)
    gencost_Pg = gencost[idxPg, 4:6] if gencost.shape[1] > 4 else gencost[idxPg, :2]

    return EvalContext(
        config=config, sys_data=sys_data, BRANFT=np.asarray(BRANFT), device=device,
        x_test=sys_data.x_test, yvmtests=yvmtests, yvatests_noslack=yvatests,
        Real_Vm_full=Real_Vm_full, Real_Va_full=Real_Va_full,
        Pdtest=_as_numpy(sys_data.Pdtest), Qdtest=_as_numpy(sys_data.Qdtest),
        Nbus=int(config.Nbus), Ntest=int(config.Ntest), bus_slack=int(sys_data.bus_slack),
        baseMVA=float(sys_data.baseMVA),
        branch=_as_numpy(sys_data.branch), Ybus=sys_data.Ybus, Yf=sys_data.Yf, Yt=sys_data.Yt,
        bus_Pg=_ensure_1d_int(sys_data.bus_Pg), bus_Qg=_ensure_1d_int(sys_data.bus_Qg),
        MAXMIN_Pg=_as_numpy(sys_data.MAXMIN_Pg), MAXMIN_Qg=_as_numpy(sys_data.MAXMIN_Qg),
        idxPg=idxPg, gencost=gencost, gencost_Pg=_as_numpy(gencost_Pg),
        his_V=_as_numpy(sys_data.his_V), hisVm_min=_as_numpy(sys_data.hisVm_min), hisVm_max=_as_numpy(sys_data.hisVm_max),
        bus_Pnet_all=bus_Pnet_all, bus_Pnet_noslack_all=bus_Pnet_noslack_all, bus_ZIB_all=bus_ZIB_all,
        DELTA=float(config.DELTA), k_dV=float(config.k_dV), flag_hisv=bool(config.flag_hisv),
        gci_values=get_gci_for_generation_nodes(sys_data, idxPg),
    )


def build_ctx_from_ngt(config, sys_data, ngt_data: Dict[str, Any], BRANFT, device) -> EvalContext:
    """Build EvalContext for NGT / NGT-Flow models."""
    x_test = _as_torch(ngt_data["x_test"], device=None, dtype=torch.float32)
    Real_Vm_full = _as_numpy(ngt_data["yvm_test"])
    Real_Va_full = _as_numpy(ngt_data["yva_test"])

    Ntest = int(Real_Vm_full.shape[0])
    Nbus = int(config.Nbus)
    bus_slack = int(sys_data.bus_slack)

    yvatests_noslack = _as_torch(_remove_slack_va(Real_Va_full, bus_slack), dtype=torch.float32)
    yvmtests = _as_torch(Real_Vm_full, dtype=torch.float32)

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

    bus_Pnet_all = _ensure_1d_int(ngt_data["bus_Pnet_all"])
    bus_Pnet_noslack_all = bus_Pnet_all[bus_Pnet_all != bus_slack]
    bus_ZIB_all = _ensure_1d_int(ngt_data["bus_ZIB_all"]) if "bus_ZIB_all" in ngt_data else None
    param_ZIMV = ngt_data.get("param_ZIMV", None)

    return EvalContext(
        config=config, sys_data=sys_data, BRANFT=np.asarray(BRANFT), device=device,
        x_test=x_test, yvmtests=yvmtests, yvatests_noslack=yvatests_noslack,
        Real_Vm_full=Real_Vm_full, Real_Va_full=Real_Va_full,
        Pdtest=Pdtest, Qdtest=Qdtest,
        Nbus=Nbus, Ntest=Ntest, bus_slack=bus_slack, baseMVA=float(sys_data.baseMVA),
        branch=_as_numpy(sys_data.branch), Ybus=sys_data.Ybus, Yf=sys_data.Yf, Yt=sys_data.Yt,
        bus_Pg=_ensure_1d_int(sys_data.bus_Pg), bus_Qg=_ensure_1d_int(sys_data.bus_Qg),
        MAXMIN_Pg=_as_numpy(ngt_data["MAXMIN_Pg"]), MAXMIN_Qg=_as_numpy(ngt_data["MAXMIN_Qg"]),
        idxPg=_ensure_1d_int(sys_data.idxPg), gencost=_as_numpy(sys_data.gencost),
        gencost_Pg=_as_numpy(ngt_data.get("gencost_Pg", None)),
        his_V=_as_numpy(sys_data.his_V), hisVm_min=_as_numpy(sys_data.hisVm_min), hisVm_max=_as_numpy(sys_data.hisVm_max),
        bus_Pnet_all=bus_Pnet_all, bus_Pnet_noslack_all=bus_Pnet_noslack_all,
        bus_ZIB_all=bus_ZIB_all, param_ZIMV=param_ZIMV,
        DELTA=float(getattr(config, "DELTA", 1e-4)), k_dV=float(getattr(config, "k_dV", 1.0)),
        flag_hisv=bool(getattr(config, "flag_hisv", True)),
        gci_values=get_gci_for_generation_nodes(sys_data, _ensure_1d_int(sys_data.idxPg)),
    )


def build_ctx_from_multi_preference(
    config, sys_data, multi_pref_data, BRANFT, device, lambda_carbon=None
) -> EvalContext:
    """Build EvalContext for multi-preference evaluation."""
    # Use validation set
    if 'x_val' in multi_pref_data:
        x_test = _as_torch(multi_pref_data['x_val'], device=None, dtype=torch.float32)
        Ntest = int(multi_pref_data['n_val'])
    else:
        x_test = _as_torch(multi_pref_data['x_train'], device=None, dtype=torch.float32)
        Ntest = int(multi_pref_data['n_train'])
    
    # Get ground truth if lambda_carbon specified
    if lambda_carbon is not None:
        y_by_pref = multi_pref_data.get('y_val_by_pref') or multi_pref_data.get('y_train_by_pref', {})
        if lambda_carbon in y_by_pref:
            y_test = y_by_pref[lambda_carbon]
        else:
            lambda_values = multi_pref_data['lambda_carbon_values']
            closest_lc = min(lambda_values, key=lambda x: abs(x - lambda_carbon))
            y_test = y_by_pref.get(closest_lc, torch.zeros((Ntest, multi_pref_data['output_dim'])))
    else:
        y_test = torch.zeros((Ntest, multi_pref_data['output_dim']), dtype=torch.float32)
    
    Nbus = int(config.Nbus)
    bus_slack = int(sys_data.bus_slack)
    
    bus_Pnet_all = _ensure_1d_int(multi_pref_data['bus_Pnet_all'])
    bus_Pnet_noslack_all = _ensure_1d_int(multi_pref_data['bus_Pnet_noslack_all'])
    NPred_Va = len(bus_Pnet_noslack_all)
    
    y_test_np = _as_numpy(y_test)
    Va_noslack_nonZIB = y_test_np[:, :NPred_Va]
    Vm_nonZIB = y_test_np[:, NPred_Va:]
    
    Real_Va_full = np.zeros((Ntest, Nbus), dtype=float)
    Real_Vm_full = np.zeros((Ntest, Nbus), dtype=float)
    Real_Va_full[:, bus_Pnet_noslack_all] = Va_noslack_nonZIB
    Real_Vm_full[:, bus_Pnet_all] = Vm_nonZIB
    
    bus_ZIB_all = multi_pref_data.get('bus_ZIB_all')
    param_ZIMV = multi_pref_data.get('param_ZIMV')
    if bus_ZIB_all is not None and param_ZIMV is not None and len(bus_ZIB_all) > 0:
        Real_Vm_full, Real_Va_full = _kron_reconstruct_zib(
            Real_Vm_full, Real_Va_full,
            bus_Pnet_all=bus_Pnet_all, bus_ZIB_all=_ensure_1d_int(bus_ZIB_all),
            param_ZIMV=np.asarray(param_ZIMV),
        )
    
    yvmtests = _as_torch(Real_Vm_full, dtype=torch.float32)
    yvatests_noslack = _as_torch(_remove_slack_va(Real_Va_full, bus_slack), dtype=torch.float32)
    
    # Power flow data
    bus_Pd = _ensure_1d_int(multi_pref_data['bus_Pd'])
    bus_Qd = _ensure_1d_int(multi_pref_data['bus_Qd'])
    x_test_np = _as_numpy(x_test)
    n_pd, n_qd = len(bus_Pd), len(bus_Qd)
    
    Pdtest = np.zeros((Ntest, Nbus), dtype=float)
    Qdtest = np.zeros((Ntest, Nbus), dtype=float)
    if n_pd > 0 and n_qd > 0:
        Pdtest[:, bus_Pd] = x_test_np[:, :n_pd]
        Qdtest[:, bus_Qd] = x_test_np[:, n_pd:n_pd + n_qd]
    
    gencost = _as_numpy(sys_data.gencost)
    idxPg = _ensure_1d_int(sys_data.idxPg)
    gencost_Pg = gencost[idxPg, 4:6] if gencost.shape[1] > 4 else gencost[idxPg, :2]
    
    return EvalContext(
        config=config, sys_data=sys_data, BRANFT=np.asarray(BRANFT), device=device,
        x_test=x_test, yvmtests=yvmtests, yvatests_noslack=yvatests_noslack,
        Real_Vm_full=Real_Vm_full, Real_Va_full=Real_Va_full,
        Pdtest=Pdtest, Qdtest=Qdtest,
        Nbus=Nbus, Ntest=Ntest, bus_slack=bus_slack, baseMVA=float(sys_data.baseMVA),
        branch=_as_numpy(sys_data.branch), Ybus=sys_data.Ybus, Yf=sys_data.Yf, Yt=sys_data.Yt,
        bus_Pg=_ensure_1d_int(sys_data.bus_Pg), bus_Qg=_ensure_1d_int(sys_data.bus_Qg),
        MAXMIN_Pg=_as_numpy(sys_data.MAXMIN_Pg), MAXMIN_Qg=_as_numpy(sys_data.MAXMIN_Qg),
        idxPg=idxPg, gencost=gencost, gencost_Pg=_as_numpy(gencost_Pg),
        his_V=_as_numpy(multi_pref_data.get('his_V')) if multi_pref_data.get('his_V') is not None else _as_numpy(sys_data.his_V),
        hisVm_min=_as_numpy(multi_pref_data.get('hisVm_min')) if multi_pref_data.get('hisVm_min') is not None else _as_numpy(sys_data.hisVm_min),
        hisVm_max=_as_numpy(multi_pref_data.get('hisVm_max')) if multi_pref_data.get('hisVm_max') is not None else _as_numpy(sys_data.hisVm_max),
        bus_Pnet_all=bus_Pnet_all, bus_Pnet_noslack_all=bus_Pnet_noslack_all,
        bus_ZIB_all=_ensure_1d_int(bus_ZIB_all) if bus_ZIB_all is not None else None, param_ZIMV=param_ZIMV,
        DELTA=float(getattr(config, "DELTA", 1e-4)), k_dV=float(getattr(config, "k_dV", 1.0)),
        flag_hisv=bool(getattr(config, "flag_hisv", True)),
        gci_values=get_gci_for_generation_nodes(sys_data, idxPg),
    )


# ==================== Predictors ====================

class SupervisedPredictor:
    """Predictor for supervised Vm/Va separate models."""
    
    def __init__(self, model_vm, model_va, dataloaders, *, model_type='simple',
                 pretrain_model_vm=None, pretrain_model_va=None, predict_fn=None):
        self.model_vm = model_vm
        self.model_va = model_va
        self.dataloaders = dataloaders
        self.model_type = model_type
        self.pretrain_model_vm = pretrain_model_vm
        self.pretrain_model_va = pretrain_model_va
        self.predict_fn = predict_fn or self._default_predict

    def _default_predict(self, model, test_x, model_type, pretrain_model, config, device):
        with torch.no_grad():
            if model_type in ['simple', 'vae']:
                return model(test_x, use_mean=True) if model_type == 'vae' else model(test_x)
            elif model_type in ['rectified', 'gaussian', 'conditional', 'interpolation']:
                z = pretrain_model(test_x, use_mean=True) if pretrain_model else torch.randn(test_x.shape[0], model.output_dim).to(device)
                inf_step = getattr(config, 'inf_step', 100)
                y_pred, _ = model.flow_backward(test_x, z, step=1/inf_step, method='Euler')
                return y_pred
        raise NotImplementedError(f"Prediction for '{model_type}' not implemented")

    def predict(self, ctx: EvalContext) -> PredPack:
        device = ctx.device

        # Vm prediction
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        yvm_hat_list = []
        for test_x, _ in self.dataloaders["test_vm"]:
            pred = self.predict_fn(self.model_vm, test_x.to(device), self.model_type, self.pretrain_model_vm, ctx.config, device)
            yvm_hat_list.append(pred.cpu())
        if device.type == "cuda": torch.cuda.synchronize()
        time_vm = time.perf_counter() - t0

        yvm_hat = torch.cat(yvm_hat_list, dim=0)
        yvm_physical = yvm_hat.detach() / ctx.config.scale_vm * (ctx.sys_data.VmUb - ctx.sys_data.VmLb) + ctx.sys_data.VmLb
        Pred_Vm_full = get_clamp(yvm_physical, ctx.sys_data.hisVm_min, ctx.sys_data.hisVm_max).numpy()

        # Va prediction
        if device.type == "cuda": torch.cuda.synchronize()
        t1 = time.perf_counter()
        yva_hat_list = []
        for test_x, _ in self.dataloaders["test_va"]:
            pred = self.predict_fn(self.model_va, test_x.to(device), self.model_type, self.pretrain_model_va, ctx.config, device)
            yva_hat_list.append(pred.cpu())
        if device.type == "cuda": torch.cuda.synchronize()
        time_va = time.perf_counter() - t1

        yva_hat = torch.cat(yva_hat_list, dim=0)
        yva_physical = yva_hat.detach() / ctx.config.scale_va
        Pred_Va_full = _insert_slack_va(yva_physical.numpy(), ctx.bus_slack)

        return PredPack(Pred_Vm_full=Pred_Vm_full, Pred_Va_full=Pred_Va_full,
                       time_vm=time_vm, time_va=time_va, time_nn_total=time_vm + time_va)


class NGTPredictor:
    """Predictor for NGT single model (partial -> full)."""
    
    def __init__(self, model_ngt):
        self.model = model_ngt

    def predict(self, ctx: EvalContext) -> PredPack:
        assert ctx.bus_Pnet_all is not None
        self.model.eval()
        x = ctx.x_test.to(ctx.device)

        if ctx.device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            V_partial = self.model(x)
        if ctx.device.type == "cuda": torch.cuda.synchronize()
        time_nn = time.perf_counter() - t0

        V_partial = _as_numpy(V_partial)
        Pred_Vm_full, Pred_Va_full = reconstruct_full_from_partial(ctx, V_partial)
        return PredPack(Pred_Vm_full=Pred_Vm_full, Pred_Va_full=Pred_Va_full, time_nn_total=time_nn)


class MultiPreferencePredictor:
    """Predictor for multi-preference models (VAE, Flow, etc.)."""
    
    def __init__(self, model, multi_pref_data, lambda_carbon, model_type='simple', *,
                 pretrain_model=None, num_flow_steps=10, flow_method='euler',
                 training_mode='standard', ngt_loss_fn=None, use_gt_anchor=False,
                 use_virtual_segment=False, refiner=None):
        """
        Args:
            model: Trained model (Flow, VAE, or MLP)
            multi_pref_data: Multi-preference data dict
            lambda_carbon: Target lambda_carbon value
            model_type: 'simple', 'vae', 'rectified', etc.
            pretrain_model: Pretrained VAE for anchor generation
            num_flow_steps: Number of flow integration steps
            flow_method: Integration method ('euler' or 'rk2')
            training_mode: 'standard' or 'trajectory'/'preference_trajectory'
            ngt_loss_fn: NGT loss function for Best-of-K selection
            use_gt_anchor: Use GT at λ_min as initial anchor (for ablation)
            use_virtual_segment: Use virtual segment mode (start from VAE anchor at λ=-L)
                                 This is for models trained with train_multi_preference_tfm_lmlp.py
            refiner: RefinerMLP model for anchor adjustment (from train_multi_preference_tfm_refiner.py)
                     If provided, uses refiner(scene, anchor) -> (dx, L) to adjust starting point
        """
        self.model = model
        self.multi_pref_data = multi_pref_data
        self.lambda_carbon = lambda_carbon
        self.model_type = model_type
        self.pretrain_model = pretrain_model
        self.num_flow_steps = num_flow_steps
        self.flow_method = flow_method
        self.training_mode = training_mode
        self.ngt_loss_fn = ngt_loss_fn
        self.use_gt_anchor = use_gt_anchor
        self.use_virtual_segment = use_virtual_segment
        # Extract refiner from argument or pretrain_model (attached in test.py)
        self.refiner = refiner or (getattr(pretrain_model, '_refiner', None) if pretrain_model else None)
        
        # Extract L-MLP from pretrain_model if available (attached in test.py)
        self.lmlp = getattr(pretrain_model, '_lmlp', None) if pretrain_model else None
        
        # Extract SimpleRefiner (refiner_v2) from pretrain_model (attached in test.py)
        # SimpleRefiner only predicts dx, no L - different from full Refiner
        self.simple_refiner = getattr(pretrain_model, '_simple_refiner', None) if pretrain_model else None
        
        # One-step distilled student mode (from train_multi_preference_refiner_flow_distill_v1.py)
        # In this mode: x_hat = x_anchor + model.predict_vec(scene, x_anchor, t=0, λ_target)
        self.onestep_student = getattr(pretrain_model, '_onestep_student', False) if pretrain_model else False
        # Multi-step mode: if > 1, use ODE integration with this many steps instead of one-step
        # Set via environment variable or _onestep_num_steps attribute
        import os
        self.onestep_num_steps = int(os.environ.get('ONESTEP_NUM_STEPS', '1'))
        if pretrain_model and hasattr(pretrain_model, '_onestep_num_steps'):
            self.onestep_num_steps = pretrain_model._onestep_num_steps
        
        lambda_values = multi_pref_data.get('lambda_carbon_values', [55.0])
        self.lc_max = max(lambda_values) if max(lambda_values) > 0 else 1.0
        
        # For trajectory mode
        if training_mode in ['trajectory', 'preference_trajectory', 'traj']:
            lambda_sorted = sorted(lambda_values)
            self.lambda_min = lambda_sorted[0]
            self.lambda_max = lambda_sorted[-1]
            self.lambda_trajectory = [(lc - self.lambda_min) / (self.lambda_max - self.lambda_min) 
                                     if self.lambda_max > self.lambda_min else 0.0 for lc in lambda_sorted]
            self.lambda_trajectory_raw = lambda_sorted
    
    def predict(self, ctx: EvalContext) -> PredPack:
        assert ctx.bus_Pnet_all is not None
        
        self.model.eval()
        if self.pretrain_model:
            self.pretrain_model.eval()
        if self.lmlp:
            self.lmlp.eval()
        
        x = ctx.x_test.to(ctx.device)
        Ntest = x.shape[0]
        pref = torch.full((Ntest, 1), self.lambda_carbon / self.lc_max, device=ctx.device)
        
        if ctx.device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        with torch.no_grad():
            if self.model_type == 'simple':
                V_partial = self.model(torch.cat([x, pref], dim=1))
            elif self.model_type == 'vae':
                if hasattr(self.model, 'pref_dim') and self.model.pref_dim > 0:
                    V_partial = self.model(x, use_mean=True, pref=pref)
                else:
                    V_partial = self.model(torch.cat([x, pref], dim=1), use_mean=True)
            elif self.model_type in ['rectified', 'gaussian', 'conditional', 'interpolation']:
                if self.onestep_student:
                    # Distilled student mode
                    if self.onestep_num_steps > 1:
                        # Multi-step mode: ODE integration along bridge time t
                        V_partial = self._sample_multistep(x, ctx.device, num_steps=self.onestep_num_steps)
                    else:
                        # One-step mode: x_hat = x_anchor + model.predict_vec(scene, x_anchor, t=0, λ_target)
                        V_partial = self._sample_onestep(x, ctx.device)
                elif self.simple_refiner is not None:
                    # SimpleRefiner V2 mode: anchor + dx, then integrate from λ=0 to target
                    V_partial = self._sample_refiner_v2_trajectory(x, ctx.device)
                elif self.use_virtual_segment:
                    # Virtual segment mode: start from VAE anchor at λ=-L, integrate to target λ
                    V_partial = self._sample_virtual_segment_trajectory(x, ctx.device)
                elif self.training_mode in ['trajectory', 'preference_trajectory', 'traj']:
                    V_partial = self._sample_preference_trajectory(x, ctx.device)
                else:
                    z = self._get_initial_anchor(x, ctx.device)
                    V_partial = self.model.sampling_with_pref(x, z, pref, num_steps=self.num_flow_steps, method=self.flow_method)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        
        if ctx.device.type == "cuda": torch.cuda.synchronize()
        time_nn = time.perf_counter() - t0
        
        V_partial = _as_numpy(V_partial)
        Pred_Vm_full, Pred_Va_full = reconstruct_full_from_partial(ctx, V_partial)
        return PredPack(Pred_Vm_full=Pred_Vm_full, Pred_Va_full=Pred_Va_full, time_nn_total=time_nn)

    def _get_initial_anchor(self, x, device):
        batch_size = x.shape[0]
        output_dim = self.multi_pref_data['output_dim']
        lambda_min_val = min(self.multi_pref_data.get('lambda_carbon_values', [0.0]))

        # Use ground truth at lambda_min as initial anchor
        if self.use_gt_anchor:
            y_val_by_pref = self.multi_pref_data.get('y_val_by_pref', {})
            if lambda_min_val in y_val_by_pref:
                y_gt = y_val_by_pref[lambda_min_val]
                if isinstance(y_gt, torch.Tensor):
                    return y_gt.to(device)
                return torch.from_numpy(y_gt).float().to(device)
        
        # Fallback: use pretrained VAE
        
        if self.pretrain_model is not None:
            if hasattr(self.pretrain_model, 'pref_dim') and self.pretrain_model.pref_dim > 0:
                pref_init = torch.full((batch_size, 1), lambda_min_val / self.lc_max, device=device)
                return self.pretrain_model(x, use_mean=True, pref=pref_init)
            x_with_pref = torch.cat([x, torch.full((batch_size, 1), lambda_min_val / self.lc_max, device=device)], dim=1)
            return self.pretrain_model(x_with_pref, use_mean=True)
        return torch.randn(batch_size, output_dim, device=device)
    
    def _sample_preference_trajectory(self, x, device):
        batch_size = x.shape[0]
        lambda_target_norm = (self.lambda_carbon - self.lambda_min) / (self.lambda_max - self.lambda_min) \
            if self.lambda_max > self.lambda_min else 0.0
        
        x_current = self._get_initial_anchor(x, device)
        
        lambda_traj = [l for l in self.lambda_trajectory if l <= lambda_target_norm]
        if len(lambda_traj) == 0 or lambda_traj[-1] < lambda_target_norm:
            lambda_traj.append(lambda_target_norm)
        
        with torch.no_grad():
            for k in range(len(lambda_traj) - 1):
                dlambda = lambda_traj[k+1] - lambda_traj[k]
                lambda_curr = torch.full((batch_size, 1), lambda_traj[k], device=device)
                v = self.model.predict_vec(x, x_current, lambda_curr, lambda_curr)
                x_current = x_current + dlambda * v
        
        return x_current
    
    def _sample_virtual_segment_trajectory(self, x, device):
        """
        Virtual segment mode inference for models trained with:
            - train_multi_preference_tfm_lmlp.py (L-MLP version)
            - train_multi_preference_tfm_refiner.py (Refiner version)
        
        Training setup:
            - Virtual segment: anchor -> x0_gt, lambda: -L -> 0
            - Real segments: x_k -> x_{k+1}, lambda: λ_k -> λ_{k+1} (normalized 0 to 1)
        
        Inference:
            1. Get anchor from VAE/MLP at pref=0
            2. If Refiner: predict (dx, L), adjust starting point to anchor + dx
               Else if L-MLP: predict L only, use anchor as starting point
            3. Start at lambda = -L_pred, integrate to target lambda
        
        Note: Lambda normalization matches training:
            - lambda_norm = (lc - lambda_min) / (lambda_max - lambda_min)
            - So lambda_min -> 0, lambda_max -> 1
            - Virtual segment starts at negative lambda (-L) and ends at 0
        """
        batch_size = x.shape[0]
        output_dim = self.multi_pref_data['output_dim']
        
        # Target lambda (normalized)
        lambda_target_norm = (self.lambda_carbon - self.lambda_min) / (self.lambda_max - self.lambda_min) \
            if self.lambda_max > self.lambda_min else 0.0
        
        # Get VAE/MLP anchor at pref=0 (lambda_min in normalized space)
        pref_zero = torch.zeros((batch_size, 1), device=device)
        if self.pretrain_model is not None:
            if hasattr(self.pretrain_model, 'pref_dim') and self.pretrain_model.pref_dim > 0:
                x_anchor = self.pretrain_model(x, use_mean=True, pref=pref_zero)
            else:
                x_with_pref = torch.cat([x, pref_zero], dim=1)
                x_anchor = self.pretrain_model(x_with_pref, use_mean=True)
        else:
            # Fallback: random initialization (not recommended)
            x_anchor = torch.randn(batch_size, output_dim, device=device)
        
        # Determine starting point and L based on available models
        # Get NPred_Va for angle wrapping
        NPred_Va = self.multi_pref_data.get('NPred_Va', output_dim // 2)
        
        if self.refiner is not None:
            # Refiner mode: predict (dx, L) and adjust starting point
            dx_pred, L_pred = self.refiner(x, x_anchor)
            # Wrap angles after adding dx (same as training)
            x_start = x_anchor + dx_pred
            x_start[..., :NPred_Va] = torch.atan2(
                torch.sin(x_start[..., :NPred_Va]),
                torch.cos(x_start[..., :NPred_Va])
            )
            
            if os.environ.get('DEBUG_VIRTUAL_LAMBDA', '0') == '1':
                print(f"  [DEBUG] Refiner mode: dx_pred norm={torch.norm(dx_pred, dim=-1).mean().item():.4f}")
        elif self.lmlp is not None:
            # L-MLP mode: predict L only, use anchor as starting point
            L_pred = self.lmlp(x, x_anchor)  # [B, 1]
            x_start = x_anchor.clone()
        else:
            # Fallback: use default L (middle of typical range)
            L_pred = torch.full((batch_size, 1), 0.5, device=device)
            x_start = x_anchor.clone()
        
        # Starting lambda = -L_pred (negative, before the λ=0 point)
        lambda_start = -L_pred  # [B, 1], negative values
        
        # Debug: verify negative lambda is being used
        if os.environ.get('DEBUG_VIRTUAL_LAMBDA', '0') == '1':
            print(f"  [DEBUG] lambda_start: min={lambda_start.min().item():.4f}, max={lambda_start.max().item():.4f}")
            print(f"  [DEBUG] L_pred: min={L_pred.min().item():.4f}, max={L_pred.max().item():.4f}, mean={L_pred.mean().item():.4f}")
        
        # Build integration trajectory from -L to target lambda
        # We use uniform steps for simplicity
        num_steps = max(self.num_flow_steps, 10)
        
        x_current = x_start.clone()  # Use adjusted starting point
        
        # Total integration range: from -L_pred to lambda_target_norm
        # For each sample, L_pred may be different, so we need per-sample integration
        with torch.no_grad():
            lambda_curr = lambda_start.clone()  # [B, 1]
            lambda_target = torch.full((batch_size, 1), lambda_target_norm, device=device)
            
            # Compute per-sample step size
            total_dlambda = lambda_target - lambda_start  # [B, 1]
            step_dlambda = total_dlambda / num_steps  # [B, 1]
            
            for step in range(num_steps):
                # Debug: print lambda range at first, middle, and last step
                if os.environ.get('DEBUG_VIRTUAL_LAMBDA', '0') == '1' and step in [0, num_steps // 2, num_steps - 1]:
                    print(f"  [DEBUG] Step {step}: lambda_curr min={lambda_curr.min().item():.4f}, max={lambda_curr.max().item():.4f}")
                
                # Predict velocity at current (x, lambda)
                v = self.model.predict_vec(x, x_current, lambda_curr, lambda_curr)
                
                # Euler integration
                x_current = x_current + step_dlambda * v
                lambda_curr = lambda_curr + step_dlambda
        
        return x_current
    
    def _sample_refiner_v2_trajectory(self, x, device):
        """
        Simplified Refiner V2 inference for models trained with:
            - train_multi_preference_tfm_refiner_v2.py
        
        Key differences from virtual segment mode:
            - Refiner only predicts Δx (no L)
            - Start from λ=0 (not λ=-L)
            - Simpler integration: just anchor correction + standard trajectory
        
        Inference:
            1. Get anchor from Standard MLP at pref=0
            2. SimpleRefiner predicts Δx
            3. x̂₀ = wrap_angles(anchor + Δx)  (prediction at λ=0)
            4. Integrate from λ=0 to target λ using standard trajectory
        """
        batch_size = x.shape[0]
        output_dim = self.multi_pref_data['output_dim']
        NPred_Va = self.multi_pref_data.get('NPred_Va', output_dim // 2)
        
        # Target lambda (normalized)
        lambda_target_norm = (self.lambda_carbon - self.lambda_min) / (self.lambda_max - self.lambda_min) \
            if self.lambda_max > self.lambda_min else 0.0
        
        # Get anchor from pretrain model (Standard MLP)
        pref_zero = torch.zeros((batch_size, 1), device=device)
        if self.pretrain_model is not None:
            if hasattr(self.pretrain_model, 'pref_dim') and self.pretrain_model.pref_dim > 0:
                x_anchor = self.pretrain_model(x, use_mean=True, pref=pref_zero)
            else:
                x_with_pref = torch.cat([x, pref_zero], dim=1)
                x_anchor = self.pretrain_model(x_with_pref, use_mean=True)
        else:
            x_anchor = torch.randn(batch_size, output_dim, device=device)
        
        # SimpleRefiner predicts only Δx (no L)
        with torch.no_grad():
            dx_pred = self.simple_refiner(x, x_anchor)
        
        # Compute starting point: x̂₀ = anchor + Δx with angle wrapping
        x_start = x_anchor + dx_pred
        if NPred_Va > 0:
            x_start[..., :NPred_Va] = torch.atan2(
                torch.sin(x_start[..., :NPred_Va]),
                torch.cos(x_start[..., :NPred_Va])
            )
        
        # If target is λ=0, we're done
        if abs(lambda_target_norm) < 1e-6:
            return x_start
        
        # Integrate from λ=0 to target λ
        num_steps = max(self.num_flow_steps, 10)
        x_current = x_start.clone()
        
        with torch.no_grad():
            lambda_curr = torch.zeros((batch_size, 1), device=device)
            lambda_target = torch.full((batch_size, 1), lambda_target_norm, device=device)
            
            step_dlambda = (lambda_target - lambda_curr) / num_steps
            
            for step in range(num_steps):
                v = self.model.predict_vec(x, x_current, lambda_curr, lambda_curr)
                x_current = x_current + step_dlambda * v
                lambda_curr = lambda_curr + step_dlambda
        
        return x_current
    
    def _sample_onestep(self, x, device):
        """
        One-step distilled student inference for models trained with:
            - train_multi_preference_refiner_flow_distill_v1.py
        
        The student model directly predicts the displacement from anchor to target:
            x_hat(λ) = x_anchor + model.predict_vec(scene, x_anchor, t=0, λ_target)
        
        This is a TRUE one-step inference - no ODE integration needed!
        """
        batch_size = x.shape[0]
        output_dim = self.multi_pref_data['output_dim']
        NPred_Va = self.multi_pref_data.get('NPred_Va', output_dim // 2)
        
        # Target lambda (normalized to [0, 1])
        lambda_target_norm = (self.lambda_carbon - self.lambda_min) / (self.lambda_max - self.lambda_min) \
            if self.lambda_max > self.lambda_min else 0.0
        
        # Get anchor from pretrain model (Standard MLP) at pref=0
        pref_zero = torch.zeros((batch_size, 1), device=device)
        if self.pretrain_model is not None:
            if hasattr(self.pretrain_model, 'pref_dim') and self.pretrain_model.pref_dim > 0:
                x_anchor = self.pretrain_model(x, use_mean=True, pref=pref_zero)
            else:
                x_with_pref = torch.cat([x, pref_zero], dim=1)
                x_anchor = self.pretrain_model(x_with_pref, use_mean=True)
        else:
            x_anchor = torch.randn(batch_size, output_dim, device=device)
        
        # Wrap anchor angles
        if NPred_Va > 0:
            x_anchor[..., :NPred_Va] = torch.atan2(
                torch.sin(x_anchor[..., :NPred_Va]),
                torch.cos(x_anchor[..., :NPred_Va])
            )
        
        # One-step prediction: x_hat = x_anchor + v(scene, x_anchor, t=0, λ_target)
        # Note: t=0 means we're at the start of the bridge (anchor), and v predicts the full displacement
        t_zero = torch.zeros((batch_size, 1), device=device)
        lambda_target = torch.full((batch_size, 1), lambda_target_norm, device=device)
        
        with torch.no_grad():
            v_pred = self.model.predict_vec(x, x_anchor, t_zero, lambda_target)
        
        # Compute final prediction
        x_hat = x_anchor + v_pred
        
        # Wrap angles
        if NPred_Va > 0:
            x_hat[..., :NPred_Va] = torch.atan2(
                torch.sin(x_hat[..., :NPred_Va]),
                torch.cos(x_hat[..., :NPred_Va])
            )
        
        return x_hat
    
    def _sample_multistep(self, x, device, num_steps=10):
        """
        Multi-step inference for student models trained with:
            - train_multi_preference_refiner_flow_distill_v1.py
        
        Instead of using t=0 for one-step prediction, we integrate along the bridge time t
        from 0 to 1 using multiple steps:
            x(t+dt) = x(t) + dt * v(scene, x(t), t, λ_target)
        
        This tests whether the student model learned a proper velocity field, not just t=0.
        
        Args:
            x: [B, input_dim] - scene/load features
            device: torch device
            num_steps: number of integration steps (default 10)
        
        Returns:
            x_hat: [B, output_dim] - predicted solution
        """
        batch_size = x.shape[0]
        output_dim = self.multi_pref_data['output_dim']
        NPred_Va = self.multi_pref_data.get('NPred_Va', output_dim // 2)
        
        # Target lambda (normalized to [0, 1])
        lambda_target_norm = (self.lambda_carbon - self.lambda_min) / (self.lambda_max - self.lambda_min) \
            if self.lambda_max > self.lambda_min else 0.0
        
        # Get anchor from pretrain model (Standard MLP) at pref=0
        pref_zero = torch.zeros((batch_size, 1), device=device)
        if self.pretrain_model is not None:
            if hasattr(self.pretrain_model, 'pref_dim') and self.pretrain_model.pref_dim > 0:
                x_anchor = self.pretrain_model(x, use_mean=True, pref=pref_zero)
            else:
                x_with_pref = torch.cat([x, pref_zero], dim=1)
                x_anchor = self.pretrain_model(x_with_pref, use_mean=True)
        else:
            x_anchor = torch.randn(batch_size, output_dim, device=device)
        
        # Wrap anchor angles
        if NPred_Va > 0:
            x_anchor[..., :NPred_Va] = torch.atan2(
                torch.sin(x_anchor[..., :NPred_Va]),
                torch.cos(x_anchor[..., :NPred_Va])
            )
        
        # Lambda stays constant at target throughout integration
        lambda_target = torch.full((batch_size, 1), lambda_target_norm, device=device)
        
        # ODE integration along bridge time t: from 0 to 1
        dt = 1.0 / num_steps
        x_current = x_anchor.clone()
        
        with torch.no_grad():
            for step in range(num_steps):
                t_current = torch.full((batch_size, 1), step * dt, device=device)
                
                # Predict velocity at current position and time
                v_pred = self.model.predict_vec(x, x_current, t_current, lambda_target)
                
                # Euler step
                x_current = x_current + dt * v_pred
                
                # Wrap angles after each step to prevent drift
                if NPred_Va > 0:
                    x_current[..., :NPred_Va] = torch.atan2(
                        torch.sin(x_current[..., :NPred_Va]),
                        torch.cos(x_current[..., :NPred_Va])
                    )
        
        return x_current


# ==================== Partial -> Full Reconstruction ====================

def reconstruct_full_from_partial(ctx: EvalContext, V_partial: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct full Vm/Va from partial NGT format."""
    bus_slack = int(ctx.bus_slack)
    Nbus = int(ctx.Nbus)
    bus_Pnet_all = _ensure_1d_int(ctx.bus_Pnet_all)
    bus_Pnet_noslack_all = _ensure_1d_int(ctx.bus_Pnet_noslack_all)

    NPred_Va = len(bus_Pnet_noslack_all)
    NPred_Vm = len(bus_Pnet_all)

    Va_noslack_nonZIB = V_partial[:, :NPred_Va]
    Vm_nonZIB = V_partial[:, NPred_Va:]

    Pred_Va_full = np.zeros((V_partial.shape[0], Nbus), dtype=float)
    Pred_Vm_full = np.zeros((V_partial.shape[0], Nbus), dtype=float)
    Pred_Va_full[:, bus_Pnet_noslack_all] = Va_noslack_nonZIB
    Pred_Va_full[:, bus_slack] = 0.0
    Pred_Vm_full[:, bus_Pnet_all] = Vm_nonZIB

    if ctx.param_ZIMV is not None and ctx.bus_ZIB_all is not None:
        Pred_Vm_full, Pred_Va_full = _kron_reconstruct_zib(
            Pred_Vm_full, Pred_Va_full,
            bus_Pnet_all=bus_Pnet_all,
            bus_ZIB_all=_ensure_1d_int(ctx.bus_ZIB_all),
            param_ZIMV=np.asarray(ctx.param_ZIMV),
        )

    if ctx.VmLb is not None and ctx.VmUb is not None:
        Pred_Vm_full = np.clip(Pred_Vm_full, ctx.VmLb, ctx.VmUb)

    return Pred_Vm_full, Pred_Va_full


# ==================== Post-Processing ====================

def _infer_jac_layout(nbus: int, jac_cols: int) -> str:
    if jac_cols == 2 * nbus:
        return "full"
    if jac_cols == 2 * nbus - 1:
        return "noslack"
    raise ValueError(f"Unexpected Jacobian cols={jac_cols}")

def _jacvec_to_full(jac_vec: np.ndarray, *, nbus: int, bus_slack: int, layout: str) -> np.ndarray:
    """Convert Jacobian-space dV to full 2Nbus vector."""
    jac_vec = np.asarray(jac_vec).ravel()
    full = np.zeros((2 * nbus,), dtype=float)

    if layout == "full":
        return jac_vec.copy()

    # layout == "noslack"
    for bus in range(nbus):
        if bus == bus_slack:
            full[bus] = 0.0
        elif bus < bus_slack:
            full[bus] = jac_vec[bus]
        else:
            full[bus] = jac_vec[bus - 1]
    for bus in range(nbus):
        full[nbus + bus] = jac_vec[(nbus - 1) + bus]
    return full


def post_process_like_evaluate_model(
    ctx: EvalContext, Pred_Vm_full: np.ndarray, Pred_Va_full: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    """Jacobian-based post-processing for constraint correction."""
    t0 = time.perf_counter()

    relax_ngt_post = getattr(ctx.config, "relax_ngt_post", True) 
    use_strict_subspace = (ctx.bus_Pnet_all is not None) and (not relax_ngt_post)

    Pred_V = Pred_Vm_full * np.exp(1j * Pred_Va_full)
    Pred_Pg, Pred_Qg, _, _ = get_genload(Pred_V, ctx.Pdtest, ctx.Qdtest, ctx.bus_Pg, ctx.bus_Qg, ctx.Ybus)

    lsPg, lsQg, lsidxPg, lsidxQg, _, vio_PQg, _, _, _, _ = get_vioPQg(
        Pred_Pg, ctx.bus_Pg, ctx.MAXMIN_Pg, Pred_Qg, ctx.bus_Qg, ctx.MAXMIN_Qg, ctx.DELTA)

    lsidxPQg = np.asarray(np.where((lsidxPg + lsidxQg) > 0)[0]).ravel()
    num_viotest = int(lsidxPQg.size)

    vio_branang, vio_branpf, deltapf, vio_branpfidx, lsSf, _, lsSf_sampidx, _ = get_viobran2(
        Pred_V, Pred_Va_full, ctx.branch, ctx.Yf, ctx.Yt, ctx.BRANFT, ctx.baseMVA, ctx.DELTA)
    vio_branpf_num = int(np.sum(np.asarray(vio_branpfidx) > 0))
    lsSf_sampidx = np.asarray(lsSf_sampidx, dtype=int)

    if num_viotest == 0:
        return Pred_Vm_full, Pred_Va_full, 0.0, {"num_viotest": 0, "vio_branpf_num": vio_branpf_num}

    # Jacobians
    dPbus_dV, dQbus_dV = dPQbus_dV(ctx.his_V, ctx.bus_Pg, ctx.bus_Qg, ctx.Ybus)
    finc = _build_finc(ctx.branch, ctx.Nbus)
    bus_Va = np.delete(np.arange(ctx.Nbus), ctx.bus_slack) 
    dPfbus_dV, dQfbus_dV = dSlbus_dV(ctx.his_V, bus_Va, ctx.branch, ctx.Yf, finc, ctx.BRANFT, ctx.Nbus)

    jac_dim = int(np.atleast_2d(dPbus_dV).shape[1])
    layout = _infer_jac_layout(int(ctx.Nbus), jac_dim)
    
    # Compute voltage correction
    if ctx.flag_hisv:
        dV1_full = np.asarray(get_hisdV(
            lsPg, lsQg, lsidxPg, lsidxQg, num_viotest, ctx.k_dV,
            ctx.bus_Pg, ctx.bus_Qg, dPbus_dV, dQbus_dV, ctx.Nbus, ctx.Ntest))
    else:
        dV1_full = np.asarray(get_dV(
            Pred_V, lsPg, lsQg, lsidxPg, lsidxQg, num_viotest, ctx.k_dV,
            ctx.bus_Pg, ctx.bus_Qg, ctx.Ybus, ctx.his_V))

    # Branch correction
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
            use_cols = np.arange(dPdV.shape[1], dtype=int)

            dmp = mp * dPdV[:, use_cols]
            dmq = mq * dQdV[:, use_cols]
            dv_sub = np.dot(np.linalg.pinv(dmp + dmq), np.array(lsSf[i][:, 1])).ravel()

            jac_vec = np.zeros((dPdV.shape[1],), dtype=float)
            jac_vec[use_cols] = dv_sub
            dV_branch_raw[i] = _jacvec_to_full(jac_vec, nbus=nbus, bus_slack=bus_slack, layout=layout)

        dV_branch_aligned = np.zeros_like(dV1_full)
        for j, samp_idx in enumerate(lsSf_sampidx.tolist()):
            pos = np.where(lsidxPQg == samp_idx)[0]
            if pos.size > 0:
                dV_branch_aligned[pos[0], :] = dV_branch_raw[j, :]
        dV1_full = dV1_full + dV_branch_aligned

    # Apply corrections
    Pred_Va1 = Pred_Va_full.copy()
    Pred_Vm1 = Pred_Vm_full.copy()
    Pred_Va1[lsidxPQg, :] = Pred_Va_full[lsidxPQg, :] - dV1_full[:, :ctx.Nbus]
    Pred_Va1[:, ctx.bus_slack] = 0.0
    Pred_Vm1[lsidxPQg, :] = Pred_Vm_full[lsidxPQg, :] - dV1_full[:, ctx.Nbus:2*ctx.Nbus]
    Pred_Vm1_clip = get_clamp(_as_torch(Pred_Vm1), ctx.hisVm_min, ctx.hisVm_max).detach().cpu().numpy()

    if use_strict_subspace and ctx.param_ZIMV is not None:
        Pred_Vm1_clip, Pred_Va1 = _kron_reconstruct_zib(
            Pred_Vm1_clip, Pred_Va1,
            bus_Pnet_all=_ensure_1d_int(ctx.bus_Pnet_all),
            bus_ZIB_all=_ensure_1d_int(ctx.bus_ZIB_all),
            param_ZIMV=np.asarray(ctx.param_ZIMV),
        )

    return Pred_Vm1_clip, Pred_Va1, time.perf_counter() - t0, {"num_viotest": num_viotest, "vio_branpf_num": vio_branpf_num}


# ==================== Cost/Carbon Computation ====================

def _compute_cost(Pg, ctx: EvalContext):
    if ctx.gencost_Pg is not None:
        PgMVA = Pg * ctx.baseMVA
        return np.sum(ctx.gencost_Pg[:, 0] * (PgMVA ** 2) + ctx.gencost_Pg[:, 1] * PgMVA, axis=1)
        return get_Pgcost(Pg, ctx.idxPg, ctx.gencost, ctx.baseMVA)

def _compute_carbon(Pg, ctx: EvalContext):
    if ctx.gci_values is None:
        return np.zeros(Pg.shape[0])
    return get_carbon_emission_vectorized(Pg, ctx.gci_values, ctx.baseMVA)


# ==================== Unified Evaluation ====================

def evaluate_unified(
    ctx: EvalContext,
    predictor,
    *,
    apply_post_processing: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Unified evaluation for all model types."""
    if verbose:
        print("\n" + "=" * 60)
        print("Unified Evaluation")
        print("=" * 60)

    pred_pack: PredPack = predictor.predict(ctx)
    Pred_Vm_full = pred_pack.Pred_Vm_full
    Pred_Va_full = pred_pack.Pred_Va_full

    # Power flow calculations
    t_pq0 = time.perf_counter()
    Pred_V = Pred_Vm_full * np.exp(1j * Pred_Va_full)
    Pred_Pg, Pred_Qg, Pred_Pd, Pred_Qd = get_genload(Pred_V, ctx.Pdtest, ctx.Qdtest, ctx.bus_Pg, ctx.bus_Qg, ctx.Ybus)
    t_pq = time.perf_counter() - t_pq0

    Real_V = ctx.Real_Vm_full * np.exp(1j * ctx.Real_Va_full)
    Real_Pg, Real_Qg, Real_Pd, Real_Qd = get_genload(Real_V, ctx.Pdtest, ctx.Qdtest, ctx.bus_Pg, ctx.bus_Qg, ctx.Ybus)

    # Constraint violations
    lsPg, lsQg, lsidxPg, lsidxQg, vio_PQgmaxmin, vio_PQg, deltaPgL, deltaPgU, deltaQgL, deltaQgU = get_vioPQg(
        Pred_Pg, ctx.bus_Pg, ctx.MAXMIN_Pg, Pred_Qg, ctx.bus_Qg, ctx.MAXMIN_Qg, ctx.DELTA)
    num_viotest = int(np.sum((lsidxPg + lsidxQg) > 0))

    vio_branang, vio_branpf, deltapf, vio_branpfidx, lsSf, _, lsSf_sampidx, _ = get_viobran2(
        Pred_V, Pred_Va_full, ctx.branch, ctx.Yf, ctx.Yt, ctx.BRANFT, ctx.baseMVA, ctx.DELTA)

    # Cost and carbon
    Pred_cost = _compute_cost(Pred_Pg, ctx)
    Real_cost = _compute_cost(Real_Pg, ctx)
    Pred_carbon = _compute_carbon(Pred_Pg, ctx)
    Real_carbon = _compute_carbon(Real_Pg, ctx)
    
    # Metrics before post-processing
    Pred_Va_noslack = _remove_slack_va(Pred_Va_full, ctx.bus_slack)
    mae_Vmtest = _to_float(get_mae(ctx.yvmtests, _as_torch(Pred_Vm_full)))
    mae_Vatest = _to_float(get_mae(ctx.yvatests_noslack, _as_torch(Pred_Va_noslack)))
    mre_cost = _to_float(get_rerr2(_as_torch(Real_cost), _as_torch(Pred_cost)))
    mre_Pd = 100.0 - _to_float(get_rerr(_as_torch(Real_Pd.sum(axis=1)), _as_torch(Pred_Pd.sum(axis=1))))
    mre_Qd = 100.0 - _to_float(get_rerr(_as_torch(Real_Qd.sum(axis=1)), _as_torch(Pred_Qd.sum(axis=1))))

    if verbose:
        print(f"\n[Before Post-Processing]")
        print(f"  Vm MAE: {mae_Vmtest:.6f}, Va MAE: {mae_Vatest:.6f}")
        print(f"  Cost MRE: {mre_cost:.4f}%")
        print(f"  Violated samples: {num_viotest}/{ctx.Ntest}")
        print(f"  Pg: {float(np.mean(_as_numpy(vio_PQg[:, 0]))):.2f}%, Qg: {float(np.mean(_as_numpy(vio_PQg[:, 1]))):.2f}%")
        print(f"  Branch angle: {float(np.mean(_as_numpy(vio_branang))):.2f}%, power: {float(np.mean(_as_numpy(vio_branpf))):.2f}%")

    # Post-processing
    time_post = 0.0
    post_dbg = {}
    if apply_post_processing:
        Pred_Vm1, Pred_Va1, time_post, post_dbg = post_process_like_evaluate_model(ctx, Pred_Vm_full, Pred_Va_full)
        
        Pred_V1 = Pred_Vm1 * np.exp(1j * Pred_Va1)
        Pred_Pg1, Pred_Qg1, Pred_Pd1, Pred_Qd1 = get_genload(Pred_V1, ctx.Pdtest, ctx.Qdtest, ctx.bus_Pg, ctx.bus_Qg, ctx.Ybus)

        _, _, lsidxPg1, lsidxQg1, _, vio_PQg1, _, _, _, _ = get_vioPQg(
            Pred_Pg1, ctx.bus_Pg, ctx.MAXMIN_Pg, Pred_Qg1, ctx.bus_Qg, ctx.MAXMIN_Qg, ctx.DELTA)
        num_viotest1 = int(np.sum((lsidxPg1 + lsidxQg1) > 0))

        vio_branang1, vio_branpf1, deltapf1 = get_viobran(
            Pred_V1, Pred_Va1, ctx.branch, ctx.Yf, ctx.Yt, ctx.BRANFT, ctx.baseMVA, ctx.DELTA)

        Pred_Va1_noslack = _remove_slack_va(Pred_Va1, ctx.bus_slack)
        mae_Vmtest1 = _to_float(get_mae(ctx.yvmtests, _as_torch(Pred_Vm1)))
        mae_Vatest1 = _to_float(get_mae(ctx.yvatests_noslack, _as_torch(Pred_Va1_noslack)))

        Pred_cost1 = _compute_cost(Pred_Pg1, ctx)
        Pred_carbon1 = _compute_carbon(Pred_Pg1, ctx)
        mre_cost1 = _to_float(get_rerr2(_as_torch(Real_cost), _as_torch(Pred_cost1)))
        mre_Pd1 = 100.0 - _to_float(get_rerr(_as_torch(Real_Pd.sum(axis=1)), _as_torch(Pred_Pd1.sum(axis=1))))
        mre_Qd1 = 100.0 - _to_float(get_rerr(_as_torch(Real_Qd.sum(axis=1)), _as_torch(Pred_Qd1.sum(axis=1))))

        if verbose:
            print(f"\n[After Post-Processing] ({time_post*1000:.1f}ms)")
            print(f"  Vm MAE: {mae_Vmtest1:.6f}, Va MAE: {mae_Vatest1:.6f}")
            print(f"  Cost MRE: {mre_cost1:.4f}%")
            print(f"  Violated samples: {num_viotest1}/{ctx.Ntest}")
            print(f"  Pg: {float(np.mean(_as_numpy(vio_PQg1[:, 0]))):.2f}%, Qg: {float(np.mean(_as_numpy(vio_PQg1[:, 1]))):.2f}%")
            print(f"  Branch angle: {float(np.mean(_as_numpy(vio_branang1))):.2f}%, power: {float(np.mean(_as_numpy(vio_branpf1))):.2f}%")
    else:
        Pred_Vm1, Pred_Va1 = Pred_Vm_full, Pred_Va_full
        mae_Vmtest1, mae_Vatest1 = mae_Vmtest, mae_Vatest
        vio_PQg1, vio_branang1, vio_branpf1, mre_cost1, deltapf1 = vio_PQg, vio_branang, vio_branpf, mre_cost, deltapf
        Pred_cost1, Pred_carbon1 = Pred_cost, Pred_carbon
        Pred_Pg1 = Pred_Pg
        mre_Pd1, mre_Qd1 = mre_Pd, mre_Qd
        num_viotest1 = num_viotest

    # Timing
    time_NN_total = float(pred_pack.time_nn_total)
    time_NN_per_sample = time_NN_total / ctx.Ntest * 1000.0
    time_total_with_post = time_NN_total + float(t_pq) + float(time_post)

    return {
        "mae_Vmtest": mae_Vmtest, "mae_Vatest": mae_Vatest,
        "mae_Vmtest1": mae_Vmtest1, "mae_Vatest1": mae_Vatest1,
        "vio_PQg": vio_PQg, "vio_PQg1": vio_PQg1,
        "vio_branang": vio_branang, "vio_branpf": vio_branpf,
        "vio_branang1": vio_branang1, "vio_branpf1": vio_branpf1,
        "mre_cost": mre_cost, "mre_cost1": mre_cost1,
        "mre_Pd": mre_Pd, "mre_Qd": mre_Qd, "mre_Pd1": mre_Pd1, "mre_Qd1": mre_Qd1,
        "deltaPgL": deltaPgL, "deltaPgU": deltaPgU, "deltaQgL": deltaQgL, "deltaQgU": deltaQgU,
        "deltapf": deltapf, "deltapf1": deltapf1 if apply_post_processing else deltapf,
        "Pred_Vm_full": Pred_Vm_full, "Pred_Va_full": Pred_Va_full,
        "Pred_Vm1": Pred_Vm1, "Pred_Va1": Pred_Va1,
        "Pred_Pg": Pred_Pg, "Pred_Pg1": Pred_Pg1,
        "Pred_cost": Pred_cost, "Pred_cost1": Pred_cost1, "Real_cost": Real_cost,
        "Pred_carbon": Pred_carbon, "Pred_carbon1": Pred_carbon1, "Real_carbon": Real_carbon,
        "num_viotest": num_viotest, "num_viotest1": num_viotest1,
        "cost_mean": float(np.mean(Pred_cost)), "cost_mean1": float(np.mean(Pred_cost1)),
        "carbon_mean": float(np.mean(Pred_carbon)), "carbon_mean1": float(np.mean(Pred_carbon1)),
        "Real_cost_mean": float(np.mean(Real_cost)), "Real_carbon_mean": float(np.mean(Real_carbon)),
        "timing_info": {
            "time_NN_total": time_NN_total, "time_NN_per_sample_ms": time_NN_per_sample,
            "time_post_processing": time_post, "time_total_with_post": time_total_with_post,
        },
    }


# ==================== Summary Extraction ====================

def extract_summary_metrics(
    eval_result: Dict[str, Any],
    model_name: str,
    category: str = "unsupervised",
    lambda_cost: Optional[float] = None,
    use_post_processed: bool = True,
) -> Dict[str, Any]:
    """Extract summary metrics for Pareto analysis."""
    suffix = "1" if use_post_processed else ""
    
    vio_PQg = eval_result.get(f"vio_PQg{suffix}", eval_result.get("vio_PQg"))
    vio_branang = eval_result.get(f"vio_branang{suffix}", eval_result.get("vio_branang"))
    vio_branpf = eval_result.get(f"vio_branpf{suffix}", eval_result.get("vio_branpf"))
    
    return {
        "name": model_name, "model_type": category, "category": category,
        "lambda_cost": lambda_cost, "lambda_carbon": 1.0 - lambda_cost if lambda_cost is not None else None,
        "cost_mean": eval_result.get(f"cost_mean{suffix}", eval_result.get("cost_mean", 0.0)),
        "carbon_mean": eval_result.get(f"carbon_mean{suffix}", eval_result.get("carbon_mean", 0.0)),
        "mae_Vm": eval_result.get(f"mae_Vmtest{suffix}", eval_result.get("mae_Vmtest", 0.0)),
        "mae_Va": eval_result.get(f"mae_Vatest{suffix}", eval_result.get("mae_Vatest", 0.0)),
        "cost_error_percent": eval_result.get(f"mre_cost{suffix}", eval_result.get("mre_cost", 0.0)),
        "mre_Pd_expected": eval_result.get(f"mre_Pd{suffix}", eval_result.get("mre_Pd", 100.0)),
        "mre_Qd_expected": eval_result.get(f"mre_Qd{suffix}", eval_result.get("mre_Qd", 100.0)),
        "Pg_satisfy": float(np.mean(_as_numpy(vio_PQg)[:, 0])) if vio_PQg is not None else 100.0,
        "Qg_satisfy": float(np.mean(_as_numpy(vio_PQg)[:, 1])) if vio_PQg is not None else 100.0,
        "branch_ang_satisfy": float(np.mean(_as_numpy(vio_branang))) if vio_branang is not None else 100.0,
        "branch_pf_satisfy": float(np.mean(_as_numpy(vio_branpf))) if vio_branpf is not None else 100.0,
        "num_violated": eval_result.get(f"num_viotest{suffix}", eval_result.get("num_viotest", 0)),
        "inference_time_ms": eval_result.get("timing_info", {}).get("time_NN_per_sample_ms", 0.0),
    }


def print_metrics_table(results, title="Evaluation Results"):
    """Print metrics table for all evaluated models."""
    print("\n" + "=" * 120)
    print(f" {title}")
    print("=" * 120)
    
    name_width = max(20, max(len(r['name']) for r in results) + 2)
    
    header = f"{'Model':<{name_width}} {'Cat':<10} {'lambda':<6} {'Cost':>10} {'Carbon':>10} {'Vm MAE':>10} {'Pg%':>8} {'Qg%':>8}"
    print(header)
    print("-" * 120)
    
    for r in sorted(results, key=lambda x: (x.get('category', 'z'), x['cost_mean'])):
        lc = f"{r['lambda_cost']:.1f}" if r.get('lambda_cost') is not None else "N/A"
        print(f"{r['name']:<{name_width}} {r.get('category', 'unk'):<10} {lc:<6} "
              f"{r['cost_mean']:>10.2f} {r['carbon_mean']:>10.4f} {r['mae_Vm']:>10.6f} "
              f"{r['Pg_satisfy']:>8.2f} {r['Qg_satisfy']:>8.2f}")
    print("-" * 120)

 
def save_evaluation_results(results, hypervolumes, ref_point, save_path, config=None):
    """Save evaluation results to JSON."""
    import json
    
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    save_data = convert({
        'models': [{k: v for k, v in r.items() if k != 'Pred_Pg'} for r in results],
        'hypervolumes': hypervolumes,
        'ref_point': ref_point.tolist() if isinstance(ref_point, np.ndarray) else list(ref_point),
    })
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {save_path}")
