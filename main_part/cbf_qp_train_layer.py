#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cbf_qp_train_layer.py

在训练阶段使用的 CBF-QP 投影层（轻量版）：
- 训练数据/状态在 NGT 子空间: x = [Va_nonZIB_noslack, Vm_nonZIB]
- Flow 模型输出速度 v = dx/dλ（同样在 NGT 子空间）
- 通过 CBF-QP 将“建议增量” delta_ref = Δλ * v_ref 投影为 delta_safe
  以尽量保证每一步增量都满足线性化安全约束。

核心特点
- 常数雅可比（在 his_V 或 flat start 处预计算），训练每一步只更新右端 b（margin）
- 约束集使用 top-k / near-bound 选择，保持 QP 小规模
- 对 v_ref -> delta_safe 的映射在固定 active-set 下闭式可微；active-set 选择默认 detach

依赖：
- utils.py: dPQbus_dV, dSlbus_dV
- cbf_qp_projection.py: cbf_active_set_project
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Union

import numpy as np
import torch

from utils import dPQbus_dV, dSlbus_dV
from cbf_qp_projection import cbf_active_set_project


# =========================
# Config
# =========================

@dataclass
class CBFQPTrainConfig:
    enabled: bool = False

    # CBF 强度（beta=1 相当于“一步线性化回到边界内”；beta<1 更保守）
    beta: float = 0.5

    # 投影器内部求解参数
    max_iters: int = 6
    detach_active_set: bool = True
    penalty_rho: float = 1e7

    # 训练期建议：用向量 trust region，Va/Vm 不同尺度
    trust_region_va: float = 0.10   # radians
    trust_region_vm: float = 0.02   # p.u.

    # 约束选择阈值（slack 小 / 已违规）
    slack_eps_vm: float = 0.02
    slack_eps_pqg: float = 0.02
    slack_eps_branch: float = 0.02

    # 每个样本保留的最大约束数量（保持 QP 小）
    k_vm: int = 64
    k_pqg: int = 64
    k_branch: int = 32

    # 训练期“间歇投影”：每个 batch 的投影概率（1.0 = 每个 batch 都投影）
    apply_prob: float = 1.0

    # 可选：distillation 正则（让网络输出更接近投影输出，减少推理时触发）
    distill_weight: float = 0.0


# =========================
# Helper: NGT -> Full V (no grad)
# =========================

def _to_np_int(x) -> np.ndarray:
    if x is None:
        return np.asarray([], dtype=np.int64)
    return np.asarray(x).astype(np.int64).ravel()


def _infer_zero_based(bus_idx: np.ndarray) -> np.ndarray:
    """If bus indices look 1-based, shift to 0-based."""
    if bus_idx.size == 0:
        return bus_idx
    if bus_idx.min() >= 1:
        return bus_idx - 1
    return bus_idx


@torch.no_grad()
def ngt_to_full_voltage_np(
    x_ngt: torch.Tensor,
    *,
    NPred_Va: int,
    Nbus: int,
    bus_slack: int,
    bus_Pnet_all: np.ndarray,
    bus_Pnet_noslack_all: np.ndarray,
    bus_ZIB_all: np.ndarray,
    param_ZIMV: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将 NGT 子空间电压 x=[Va_nonZIB_noslack, Vm_nonZIB] 转为 full-bus Vm/Va/V。
    返回 numpy（不需要梯度）。
    """
    x_np = x_ngt.detach().cpu().numpy()
    B = int(x_np.shape[0])

    Va_sub = x_np[:, :NPred_Va]
    Vm_sub = x_np[:, NPred_Va:]

    Vm_full = np.ones((B, Nbus), dtype=np.float64)
    Va_full = np.zeros((B, Nbus), dtype=np.float64)

    if bus_Pnet_all.size > 0:
        Vm_full[:, bus_Pnet_all] = Vm_sub
    if bus_Pnet_noslack_all.size > 0:
        Va_full[:, bus_Pnet_noslack_all] = Va_sub
    Va_full[:, bus_slack] = 0.0

    # Kron reconstruction for ZIB buses
    if (param_ZIMV is not None) and (bus_ZIB_all.size > 0) and (bus_Pnet_all.size > 0):
        Vx = Vm_full[:, bus_Pnet_all] * np.exp(1j * Va_full[:, bus_Pnet_all])  # [B, NPnet]
        Vy = (param_ZIMV @ Vx.T).T  # [B, NZIB]
        Vm_full[:, bus_ZIB_all] = np.abs(Vy)
        Va_full[:, bus_ZIB_all] = np.angle(Vy)

    V_full = Vm_full * np.exp(1j * Va_full)
    return Vm_full, Va_full, V_full


def _select_small_slack(slack: np.ndarray, eps: float, kmax: int) -> np.ndarray:
    slack = np.asarray(slack).ravel()
    idx = np.where(slack < eps)[0]
    if idx.size == 0:
        return idx
    if idx.size <= kmax:
        return idx
    order = np.argsort(slack[idx])  # smallest slack first
    return idx[order[:kmax]]


# =========================
# Projector
# =========================

class CBFQPProjectorNGT:
    """
    训练期 CBF-QP 投影器：
    - 预计算常数雅可比（his_V 处），并切到 NGT 子空间列
    - 每个 batch 用当前 (x_curr, batch_x) 计算 margin b
    - 调用 cbf_active_set_project 做增量投影
    """
    def __init__(self, sys_data: Any, multi_pref_data: Dict[str, Any], device: torch.device, cfg: CBFQPTrainConfig):
        self.sys = sys_data
        self.mp = multi_pref_data
        self.device = device
        self.cfg = cfg

        self.Nbus = int(getattr(sys_data, "Nbus", sys_data.bus.shape[0]))
        self.bus_slack = int(multi_pref_data.get("bus_slack", getattr(sys_data, "bus_slack", 0)))

        self.bus_Pnet_all = _infer_zero_based(_to_np_int(multi_pref_data.get("bus_Pnet_all")))
        self.bus_Pnet_noslack_all = _infer_zero_based(_to_np_int(multi_pref_data.get("bus_Pnet_noslack_all")))
        self.bus_ZIB_all = _infer_zero_based(_to_np_int(multi_pref_data.get("bus_ZIB_all")))

        self.bus_Pd = _infer_zero_based(_to_np_int(multi_pref_data.get("bus_Pd")))
        self.bus_Qd = _infer_zero_based(_to_np_int(multi_pref_data.get("bus_Qd")))

        self.NPred_Va = int(multi_pref_data.get("NPred_Va"))
        self.NPred_Vm = int(multi_pref_data.get("NPred_Vm"))
        self.nvar = int(self.NPred_Va + self.NPred_Vm)

        self.param_ZIMV = multi_pref_data.get("param_ZIMV", None)
        if self.param_ZIMV is not None:
            self.param_ZIMV = np.asarray(self.param_ZIMV)

        # --- Bounds ---
        VmLb = np.asarray(getattr(sys_data, "VmLb", 0.95)).ravel()
        VmUb = np.asarray(getattr(sys_data, "VmUb", 1.05)).ravel()
        if VmLb.size == 1:
            VmLb = np.full((self.Nbus,), float(VmLb[0]), dtype=np.float64)
        if VmUb.size == 1:
            VmUb = np.full((self.Nbus,), float(VmUb[0]), dtype=np.float64)
        self.VmLb = VmLb
        self.VmUb = VmUb

        # --- Gen info ---
        self.bus_Pg = _infer_zero_based(_to_np_int(getattr(sys_data, "bus_Pg", None)))
        self.bus_Qg = _infer_zero_based(_to_np_int(getattr(sys_data, "bus_Qg", None)))

        self.Pg_maxmin = np.asarray(getattr(sys_data, "MAXMIN_Pg", np.zeros((0, 2)))).astype(np.float64)
        self.Qg_maxmin = np.asarray(getattr(sys_data, "MAXMIN_Qg", np.zeros((0, 2)))).astype(np.float64)

        # --- Branch info & limit ---
        self.branch = np.asarray(getattr(sys_data, "branch")).astype(np.float64)
        baseMVA = float(np.asarray(getattr(sys_data, "baseMVA")).ravel()[0])
        self.baseMVA = baseMVA

        # Try to detect the rateA column location:
        # - In some code paths branch_para is [f,t,rateA,angmin,angmax] => col2
        # - In MATPOWER raw branch, rateA is col5 (0-based)
        rateA = None
        if self.branch.shape[1] >= 3:
            rateA = self.branch[:, 2]
        if self.branch.shape[1] >= 6:
            # if this looks more plausible (many nonzeros), take col5
            cand = self.branch[:, 5]
            if np.count_nonzero(cand) > np.count_nonzero(rateA):
                rateA = cand
        rateA = np.asarray(rateA).ravel()
        self.Smax = np.where(rateA == 0, 1e10, rateA / baseMVA).astype(np.float64)  # p.u.
        self.Smax2 = (self.Smax ** 2).astype(np.float64)

        # --- Build indep columns in full Jacobian (2*Nbus layout: [Va_all, Vm_all]) ---
        Va_cols = self.bus_Pnet_noslack_all.astype(np.int64)         # Va (excluding slack, excluding ZIB)
        Vm_cols = (self.Nbus + self.bus_Pnet_all).astype(np.int64)   # Vm for non-ZIB buses
        self.indep_cols_full = np.concatenate([Va_cols, Vm_cols], axis=0)
        assert self.indep_cols_full.shape[0] == self.nvar

        # --- Precompute constant Jacobians at his_V (or flat start) ---
        his_V = multi_pref_data.get("his_V", getattr(sys_data, "his_V", None))
        if his_V is None:
            his_V = np.ones((self.Nbus,), dtype=np.complex128)
        his_V = np.asarray(his_V).ravel().astype(np.complex128)

        # dPbus_dV, dQbus_dV: [Nbus, 2*Nbus]
        dPbus_dV, dQbus_dV = dPQbus_dV(his_V, self.bus_Pg, self.bus_Qg, sys_data.Ybus)
        dPbus_sub = np.asarray(dPbus_dV)[:, self.indep_cols_full].astype(np.float32)  # [Nbus, nvar]
        dQbus_sub = np.asarray(dQbus_dV)[:, self.indep_cols_full].astype(np.float32)

        # Slice to generator buses directly (smaller)
        if self.bus_Pg.size > 0:
            self.dPg_sub_t = torch.from_numpy(dPbus_sub[self.bus_Pg]).to(device)
        else:
            self.dPg_sub_t = torch.zeros((0, self.nvar), device=device, dtype=torch.float32)

        if self.bus_Qg.size > 0:
            self.dQg_sub_t = torch.from_numpy(dQbus_sub[self.bus_Qg]).to(device)
        else:
            self.dQg_sub_t = torch.zeros((0, self.nvar), device=device, dtype=torch.float32)

        # Branch flow Jacobians (from-end Pf/Qf) at his_V
        # dSlbus_dV returns [nbranch, 2*Nbus] derivatives for Pf and Qf (from end)
        # Need finc (from-bus incidence) for correct mapping in that helper
        fbus = self.branch[:, 0].astype(np.int64).ravel()
        fbus = _infer_zero_based(fbus)
        nbranch = int(self.branch.shape[0])
        finc = np.zeros((nbranch, self.Nbus), dtype=np.float64)
        finc[np.arange(nbranch), fbus] = 1.0

        bus_Va = np.arange(self.Nbus, dtype=np.int64)
        dPf_dV, dQf_dV = dSlbus_dV(his_V, bus_Va, self.branch, sys_data.Yf, finc, None, self.Nbus)
        dPf_sub = np.asarray(dPf_dV)[:, self.indep_cols_full].astype(np.float32)  # [nbranch, nvar]
        dQf_sub = np.asarray(dQf_dV)[:, self.indep_cols_full].astype(np.float32)
        self.dPf_sub_t = torch.from_numpy(dPf_sub).to(device)
        self.dQf_sub_t = torch.from_numpy(dQf_sub).to(device)

        # --- Dense torch matrices for fast margin computation ---
        # Note: for training stability/speed, we compute margins under no_grad, so dtype float32/complex64 is fine.
        Ybus = sys_data.Ybus
        Yf = sys_data.Yf
        # scipy sparse -> dense
        self.Ybus_dense_t = torch.tensor(Ybus.toarray() if hasattr(Ybus, "toarray") else np.asarray(Ybus),
                                         device=device, dtype=torch.complex64)
        self.Yf_dense_t = torch.tensor(Yf.toarray() if hasattr(Yf, "toarray") else np.asarray(Yf),
                                       device=device, dtype=torch.complex64)
        self.fbus_t = torch.tensor(fbus, device=device, dtype=torch.long)

        # trust region vector
        tr = torch.zeros((self.nvar,), device=device, dtype=torch.float32)
        tr[:self.NPred_Va] = float(cfg.trust_region_va)
        tr[self.NPred_Va:] = float(cfg.trust_region_vm)
        self.trust_region_vec = tr

    @torch.no_grad()
    def _loads_from_scene(self, scene: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        scene = [Pd_nonzero, Qd_nonzero] (p.u.) -> Pd_full/Qd_full: [B,Nbus]
        """
        B = int(scene.shape[0])
        Pd = torch.zeros((B, self.Nbus), device=scene.device, dtype=torch.float32)
        Qd = torch.zeros((B, self.Nbus), device=scene.device, dtype=torch.float32)

        nPd = int(self.bus_Pd.size)
        nQd = int(self.bus_Qd.size)
        if nPd > 0:
            Pd_part = scene[:, :nPd].float()
            Pd[:, torch.tensor(self.bus_Pd, device=scene.device)] = Pd_part
        if nQd > 0:
            Qd_part = scene[:, nPd:nPd + nQd].float()
            Qd[:, torch.tensor(self.bus_Qd, device=scene.device)] = Qd_part
        return Pd, Qd

    @torch.no_grad()
    def _fullV_from_ngt(self, x_ngt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        NGT x -> Vm_full/Va_full/V_full (torch). 仅用于计算 margin，不走梯度。
        """
        Vm_full_np, Va_full_np, V_full_np = ngt_to_full_voltage_np(
            x_ngt,
            NPred_Va=self.NPred_Va,
            Nbus=self.Nbus,
            bus_slack=self.bus_slack,
            bus_Pnet_all=self.bus_Pnet_all,
            bus_Pnet_noslack_all=self.bus_Pnet_noslack_all,
            bus_ZIB_all=self.bus_ZIB_all,
            param_ZIMV=self.param_ZIMV,
        )
        Vm_full = torch.from_numpy(Vm_full_np).to(self.device, dtype=torch.float32)
        Va_full = torch.from_numpy(Va_full_np).to(self.device, dtype=torch.float32)
        V_full = torch.from_numpy(V_full_np).to(self.device, dtype=torch.complex64)
        return Vm_full, Va_full, V_full

    @torch.no_grad()
    def build_Ab(self, x_curr_ngt: torch.Tensor, scene: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据当前状态/场景构造每个样本的 A,b（已做 top-k 选择并 padding）。

        返回:
          A: [B, m_max, nvar]
          b: [B, m_max]
        """
        cfg = self.cfg
        B = int(x_curr_ngt.shape[0])
        device = self.device

        # full voltages (for margin computation)
        Vm_full, _, V_full = self._fullV_from_ngt(x_curr_ngt)
        Pd_full, Qd_full = self._loads_from_scene(scene)

        rows_batch: List[torch.Tensor] = []
        rhs_batch: List[torch.Tensor] = []
        m_max = 0

        # Compute Pg/Qg margins (torch, vectorized)
        Pg = Qg = None
        if cfg.k_pqg > 0 and (self.bus_Pg.size > 0 or self.bus_Qg.size > 0):
            # S = V * conj(Ybus * V)
            I = torch.conj(self.Ybus_dense_t @ V_full.transpose(0, 1)).transpose(0, 1)  # [B,Nbus]
            S = V_full * I
            P = torch.real(S).float()
            Q = torch.imag(S).float()
            if self.bus_Pg.size > 0:
                Pg = P[:, torch.tensor(self.bus_Pg, device=device)] + Pd_full[:, torch.tensor(self.bus_Pg, device=device)]
            if self.bus_Qg.size > 0:
                Qg = Q[:, torch.tensor(self.bus_Qg, device=device)] + Qd_full[:, torch.tensor(self.bus_Qg, device=device)]

        # Branch Pf/Qf margins (torch, vectorized)
        Pf = Qf = None
        if cfg.k_branch > 0:
            If = (self.Yf_dense_t @ V_full.transpose(0, 1)).transpose(0, 1)  # [B,nbranch]
            Vf = V_full[:, self.fbus_t]  # [B,nbranch]
            Sf = Vf * torch.conj(If)
            Pf = torch.real(Sf).float()
            Qf = torch.imag(Sf).float()

        # Per-sample selection and row construction
        for i in range(B):
            rows_i: List[torch.Tensor] = []
            rhs_i: List[torch.Tensor] = []

            # ---- Vm bounds on non-ZIB buses (subspace variables) ----
            if cfg.k_vm > 0 and self.bus_Pnet_all.size > 0:
                Vm_sub = x_curr_ngt[i, self.NPred_Va:].detach().cpu().numpy()  # [NPred_Vm]
                ub = self.VmUb[self.bus_Pnet_all]
                lb = self.VmLb[self.bus_Pnet_all]

                slack_up = ub - Vm_sub
                slack_lo = Vm_sub - lb

                idx_up = _select_small_slack(slack_up, cfg.slack_eps_vm, cfg.k_vm)
                idx_lo = _select_small_slack(slack_lo, cfg.slack_eps_vm, cfg.k_vm)

                # upper: +ΔVm_j <= beta*(ub - Vm)
                for j in idx_up.tolist():
                    row = torch.zeros((self.nvar,), device=device, dtype=torch.float32)
                    row[self.NPred_Va + j] = 1.0
                    rows_i.append(row)
                    rhs_i.append(torch.tensor(cfg.beta * float(slack_up[j]), device=device, dtype=torch.float32))

                # lower: -ΔVm_j <= beta*(Vm - lb)
                for j in idx_lo.tolist():
                    row = torch.zeros((self.nvar,), device=device, dtype=torch.float32)
                    row[self.NPred_Va + j] = -1.0
                    rows_i.append(row)
                    rhs_i.append(torch.tensor(cfg.beta * float(slack_lo[j]), device=device, dtype=torch.float32))

            # ---- Pg/Qg bounds ----
            if Pg is not None and self.bus_Pg.size > 0:
                Pg_i = Pg[i].detach().cpu().numpy()  # [nPg]
                # [FIX] MAXMIN format: col0=max, col1=min (consistent with unified_eval.py)
                Pmax = self.Pg_maxmin[:, 0]
                Pmin = self.Pg_maxmin[:, 1]
                slack_up = Pmax - Pg_i
                slack_lo = Pg_i - Pmin
                idx_up = _select_small_slack(slack_up, cfg.slack_eps_pqg, cfg.k_pqg)
                idx_lo = _select_small_slack(slack_lo, cfg.slack_eps_pqg, cfg.k_pqg)

                # upper
                if idx_up.size > 0:
                    J = self.dPg_sub_t[idx_up]  # [k,nvar]
                    for k in range(J.shape[0]):
                        rows_i.append(J[k])
                        rhs_i.append(torch.tensor(cfg.beta * float(slack_up[idx_up[k]]), device=device, dtype=torch.float32))
                # lower
                if idx_lo.size > 0:
                    J = self.dPg_sub_t[idx_lo]
                    for k in range(J.shape[0]):
                        rows_i.append(-J[k])
                        rhs_i.append(torch.tensor(cfg.beta * float(slack_lo[idx_lo[k]]), device=device, dtype=torch.float32))

            if Qg is not None and self.bus_Qg.size > 0:
                Qg_i = Qg[i].detach().cpu().numpy()
                # [FIX] MAXMIN format: col0=max, col1=min (consistent with unified_eval.py)
                Qmax = self.Qg_maxmin[:, 0]
                Qmin = self.Qg_maxmin[:, 1]
                slack_up = Qmax - Qg_i
                slack_lo = Qg_i - Qmin
                idx_up = _select_small_slack(slack_up, cfg.slack_eps_pqg, cfg.k_pqg)
                idx_lo = _select_small_slack(slack_lo, cfg.slack_eps_pqg, cfg.k_pqg)

                if idx_up.size > 0:
                    J = self.dQg_sub_t[idx_up]
                    for k in range(J.shape[0]):
                        rows_i.append(J[k])
                        rhs_i.append(torch.tensor(cfg.beta * float(slack_up[idx_up[k]]), device=device, dtype=torch.float32))

                if idx_lo.size > 0:
                    J = self.dQg_sub_t[idx_lo]
                    for k in range(J.shape[0]):
                        rows_i.append(-J[k])
                        rhs_i.append(torch.tensor(cfg.beta * float(slack_lo[idx_lo[k]]), device=device, dtype=torch.float32))

            # ---- Branch |S|^2 constraints (from end) ----
            if Pf is not None and Qf is not None and cfg.k_branch > 0:
                Pf_i = Pf[i].detach().cpu().numpy()
                Qf_i = Qf[i].detach().cpu().numpy()
                S2 = Pf_i * Pf_i + Qf_i * Qf_i  # [nbranch]
                margin = self.Smax2 - S2
                # select smallest margin
                idx = _select_small_slack(margin, cfg.slack_eps_branch, cfg.k_branch)
                for br_idx in idx.tolist():
                    mp = 2.0 * float(Pf_i[br_idx])
                    mq = 2.0 * float(Qf_i[br_idx])
                    row = mp * self.dPf_sub_t[br_idx] + mq * self.dQf_sub_t[br_idx]
                    rows_i.append(row)
                    rhs_i.append(torch.tensor(cfg.beta * float(margin[br_idx]), device=device, dtype=torch.float32))

            if len(rows_i) == 0:
                # No constraints: add a dummy always-inactive constraint (0*z <= big)
                rows_i.append(torch.zeros((self.nvar,), device=device, dtype=torch.float32))
                rhs_i.append(torch.tensor(1e9, device=device, dtype=torch.float32))

            A_i = torch.stack(rows_i, dim=0)  # [mi,nvar]
            b_i = torch.stack(rhs_i, dim=0)   # [mi]
            m_max = max(m_max, int(A_i.shape[0]))
            rows_batch.append(A_i)
            rhs_batch.append(b_i)

        # pad to [B, m_max, nvar]
        A = torch.zeros((B, m_max, self.nvar), device=device, dtype=torch.float32)
        b = torch.full((B, m_max), 1e9, device=device, dtype=torch.float32)

        for i in range(B):
            mi = rows_batch[i].shape[0]
            A[i, :mi, :] = rows_batch[i]
            b[i, :mi] = rhs_batch[i]

        return A, b

    def project_delta(self, delta_ref: torch.Tensor, A: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """
        对增量 delta_ref 做 CBF-QP 投影，输出 delta_safe.
        """
        cfg = self.cfg
        delta_safe, info = cbf_active_set_project(
            delta_ref,
            A, b,
            trust_region=self.trust_region_vec,
            max_iters=int(cfg.max_iters),
            tol=1e-9,
            active_eps=1e-7,
            max_active=None,  # A/b 已经是筛选后的集合
            penalty_rho=float(cfg.penalty_rho),
            detach_active_set=bool(cfg.detach_active_set),
            use_pinv_fallback=True,
        )
        return delta_safe, info

    def maybe_project_velocity(
        self,
        x_curr_ngt: torch.Tensor,
        scene: torch.Tensor,
        v_ref: torch.Tensor,
        dlambda: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Any]]:
        """
        输入 v_ref=dx/dλ，输出 (v_safe, delta_safe).
        """
        cfg = self.cfg
        if (not cfg.enabled) or (cfg.apply_prob < 1.0 and float(torch.rand(1)) > cfg.apply_prob):
            delta_ref = dlambda * v_ref
            return v_ref, delta_ref, None

        # 约束线性化点取 x_curr（建议 detach）
        A, b = self.build_Ab(x_curr_ngt.detach(), scene.detach())
        delta_ref = dlambda * v_ref
        delta_safe, info = self.project_delta(delta_ref, A, b)
        v_safe = delta_safe / (dlambda + 1e-12)
        return v_safe, delta_safe, info
