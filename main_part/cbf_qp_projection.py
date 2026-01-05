
# -*- coding: utf-8 -*-
"""
cbf_qp_projection.py

一个轻量级、可用于“偏好轨迹/流模型”中的 CBF-QP 投影层：
- 输入：流模型给出的建议速度/增量 z_ref（例如 [ΔVa, ΔVm] 或 dz/dr）
- 输出：满足线性化 CBF 安全不等式约束的修正后速度/增量 z_safe

核心求解器：
- 使用 active-set（活跃约束选择）+ 闭式投影（小系统线性求解）
- 目标：min 1/2 || z - z_ref ||_W^2   s.t.  A z <= b
- 在固定 active-set 下，该投影是闭式的、对 (z_ref, A, b) 分段可微
  （活跃集切换处不可导/不光滑；但在多数深度学习实践中可用）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union, List

import torch


@dataclass
class ProjectionInfo:
    num_iters: int
    max_violation_before: float
    max_violation_after: float
    active_sizes: List[int]


def _ensure_batch(A: torch.Tensor, b: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    统一输入形状：
      z: [B, n]
      A: [m, n] 或 [B, m, n]
      b: [m] 或 [B, m]
    """
    if z.dim() != 2:
        raise ValueError(f"z_ref must be 2D [B, n], got shape={tuple(z.shape)}")

    B, n = z.shape

    if A.dim() == 2:
        m, nA = A.shape
        if nA != n:
            raise ValueError(f"A second dim must equal n={n}, got {nA}")
        A_b = A.unsqueeze(0).expand(B, m, n)
    elif A.dim() == 3:
        if A.shape[0] != B:
            raise ValueError(f"A batch dim must equal B={B}, got {A.shape[0]}")
        m, nA = A.shape[1], A.shape[2]
        if nA != n:
            raise ValueError(f"A last dim must equal n={n}, got {nA}")
        A_b = A
    else:
        raise ValueError(f"A must be 2D or 3D, got dim={A.dim()}")

    if b.dim() == 1:
        if b.shape[0] != A_b.shape[1]:
            raise ValueError(f"b length must equal m={A_b.shape[1]}, got {b.shape[0]}")
        b_b = b.unsqueeze(0).expand(B, -1)
    elif b.dim() == 2:
        if b.shape[0] != B or b.shape[1] != A_b.shape[1]:
            raise ValueError(f"b must be [B, m]={B, A_b.shape[1]}, got {tuple(b.shape)}")
        b_b = b
    else:
        raise ValueError(f"b must be 1D or 2D, got dim={b.dim()}")

    return A_b, b_b


def _batch_Az(A_b: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """A_b: [B,m,n], z: [B,n] -> Az: [B,m]"""
    return torch.einsum("bmn,bn->bm", A_b, z)


def build_cbf_Ab_from_jacobian(
    J: torch.Tensor,
    g_hat: torch.Tensor,
    g_min: Optional[torch.Tensor] = None,
    g_max: Optional[torch.Tensor] = None,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将一组约束 g_min <= g(x) <= g_max 在线性化点处转换为 CBF 线性不等式：

      上界：  J z <= beta (g_max - g_hat)
      下界： -J z <= beta (g_hat - g_min)

    参数
    - J:     [m, n] 或 [B, m, n]，对 z 的雅可比（线性化得到）
    - g_hat: [m] 或 [B, m]，在当前点处的 g 值
    - g_min/g_max: 同形状或 [m]，可为 None（表示无该侧约束）
    - beta:  CBF 系数（越大越“强力向内推”）

    返回
    - A, b 使得 A z <= b
    """
    if beta <= 0:
        raise ValueError("beta must be > 0")

    if g_hat.dim() == 1:
        # [m] -> [1,m]
        g_hat_b = g_hat.unsqueeze(0)
    elif g_hat.dim() == 2:
        g_hat_b = g_hat
    else:
        raise ValueError(f"g_hat must be 1D or 2D, got dim={g_hat.dim()}")

    # 统一 batch
    if J.dim() == 2:
        J_b = J.unsqueeze(0).expand(g_hat_b.shape[0], -1, -1)
    elif J.dim() == 3:
        J_b = J
    else:
        raise ValueError(f"J must be 2D or 3D, got dim={J.dim()}")

    B, m, n = J_b.shape
    if g_hat_b.shape[-1] != m:
        raise ValueError(f"g_hat last dim must equal m={m}, got {g_hat_b.shape[-1]}")

    A_list = []
    b_list = []

    # upper:  J z <= beta (g_max - g_hat)
    if g_max is not None:
        if g_max.dim() == 1:
            g_max_b = g_max.unsqueeze(0).expand(B, -1)
        else:
            g_max_b = g_max
        rhs = beta * (g_max_b - g_hat_b)  # [B,m]
        A_list.append(J_b)               # [B,m,n]
        b_list.append(rhs)               # [B,m]

    # lower: -J z <= beta (g_hat - g_min)
    if g_min is not None:
        if g_min.dim() == 1:
            g_min_b = g_min.unsqueeze(0).expand(B, -1)
        else:
            g_min_b = g_min
        rhs = beta * (g_hat_b - g_min_b)
        A_list.append(-J_b)
        b_list.append(rhs)

    if not A_list:
        raise ValueError("At least one of g_min or g_max must be provided")

    A = torch.cat(A_list, dim=1)  # [B, m', n]
    b = torch.cat(b_list, dim=1)  # [B, m']
    # 若输入本来无 batch，则返回 [m',n] / [m']
    if g_hat.dim() == 1 and J.dim() == 2:
        return A[0], b[0]
    return A, b


def cbf_active_set_project(
    z_ref: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    *,
    W_diag: Optional[torch.Tensor] = None,
    trust_region: Optional[Union[float, torch.Tensor]] = None,
    max_iters: int = 5,
    tol: float = 1e-9,
    active_eps: float = 1e-6,
    max_active: int = 64,
    penalty_rho: float = 1e6,
    detach_active_set: bool = True,
    use_pinv_fallback: bool = True,
) -> Tuple[torch.Tensor, ProjectionInfo]:
    """
    CBF-QP 投影层（硬约束/active-set 版本）：

      min 1/2 || z - z_ref ||_W^2
      s.t. A z <= b

    其中 W = diag(W_diag)（可选），默认为 I。

    求解策略（轻量级）：
    - 迭代地构造活跃集 I（违反或接近违反的约束）
    - 把 I 视为等式 A_I z = b_I（或“强惩罚近似”等式）
    - 闭式投影解（小系统，维度=|I|）：
        z = z_ref - W^{-1} A_I^T (A_I W^{-1} A_I^T + (1/rho)I)^{-1} (A_I z_ref - b_I)

    可微性说明：
    - 在线性求解部分（solve/pinv）对输入是可微的
    - 活跃集的选择是离散的，因此整体是“分段可微”
      若你希望训练时更平滑，可在训练阶段 detach_active_set=True，
      或使用软投影/可微QP层（更复杂，这里先给轻量版本）。

    返回：
    - z_safe: [B,n]
    - info:  统计信息（迭代次数、违规幅度等）
    """
    device = z_ref.device
    dtype = z_ref.dtype

    A_b, b_b = _ensure_batch(A, b, z_ref)
    B, m, n = A_b.shape

    if W_diag is None:
        W_inv = torch.ones((B, n), device=device, dtype=dtype)
    else:
        if W_diag.dim() == 1:
            if W_diag.numel() != n:
                raise ValueError(f"W_diag must have length n={n}, got {W_diag.numel()}")
            W_inv = (1.0 / (W_diag + 1e-12)).unsqueeze(0).expand(B, -1)
        elif W_diag.dim() == 2:
            if W_diag.shape != (B, n):
                raise ValueError(f"W_diag must be [B,n], got {tuple(W_diag.shape)}")
            W_inv = 1.0 / (W_diag + 1e-12)
        else:
            raise ValueError("W_diag must be 1D or 2D")

    # 初始：不修正
    z = z_ref

    viol0 = _batch_Az(A_b, z_ref) - b_b
    max_viol_before = float(viol0.max().detach().cpu())

    active_sizes: List[int] = []

    for it in range(max_iters):
        viol = _batch_Az(A_b, z) - b_b  # [B,m]
        max_viol = viol.max(dim=1).values  # [B]

        if float(max_viol.max().detach().cpu()) <= tol:
            # 全 batch 满足
            break

        # ==================== Vectorized Batch Processing ====================
        # Use fixed max_active for all samples to enable batched operations
        k_fixed = min(max_active, m) if max_active is not None else m
        
        # Get top-k violations per sample (batched)
        _, idx_batch = torch.topk(viol, k=k_fixed, dim=1, largest=True, sorted=False)  # [B, k_fixed]
        
        if detach_active_set:
            idx_batch = idx_batch.detach()
        
        # Gather A_I and b_I for all samples at once
        # A_I[bi] = A_b[bi, idx_batch[bi], :] => use advanced indexing
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, k_fixed)  # [B, k_fixed]
        A_I = A_b[batch_idx, idx_batch, :]  # [B, k_fixed, n]
        b_I = b_b[batch_idx, idx_batch]     # [B, k_fixed]
        
        # Check which constraints are actually active (viol > -active_eps)
        viol_I = torch.gather(viol, 1, idx_batch)  # [B, k_fixed]
        active_mask = viol_I > (-active_eps)  # [B, k_fixed]
        
        # If no constraints active for any sample, skip
        if not active_mask.any():
            active_sizes.extend([0] * B)
            continue
        
        # For simplicity in batched mode, use all k_fixed constraints with masking
        # Inactive constraints get zero contribution via masking
        active_sizes.extend([int(active_mask[bi].sum()) for bi in range(B)])
        
        # rhs = A_I @ z_ref - b_I, batched
        rhs = torch.einsum('bkn,bn->bk', A_I, z_ref) - b_I  # [B, k_fixed]
        
        # M = A_I @ diag(W_inv) @ A_I.T + (1/rho) I
        A_Winv = A_I * W_inv.unsqueeze(1)  # [B, k_fixed, n]
        M = torch.einsum('bkn,bjn->bkj', A_Winv, A_I)  # [B, k_fixed, k_fixed]
        if penalty_rho is not None and penalty_rho > 0:
            eye_k = torch.eye(k_fixed, device=device, dtype=dtype).unsqueeze(0)  # [1, k, k]
            M = M + (1.0 / penalty_rho) * eye_k
        
        # Mask inactive constraints by adding large diagonal (makes them inactive)
        inactive_mask = ~active_mask  # [B, k_fixed]
        # Add large value to diagonal for inactive constraints
        diag_mask = torch.diag_embed(inactive_mask.float() * 1e10)  # [B, k_fixed, k_fixed]
        M = M + diag_mask
        rhs = rhs * active_mask.float()  # Zero out rhs for inactive constraints
        
        # Batched solve: u = M^{-1} @ rhs
        try:
            u = torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)  # [B, k_fixed]
        except RuntimeError:
            if not use_pinv_fallback:
                raise
            # Fallback to per-sample pinv (slower but robust)
            u = torch.zeros_like(rhs)
            for bi in range(B):
                u[bi] = torch.linalg.pinv(M[bi]) @ rhs[bi]
        
        # correction = W_inv * (A_I.T @ u)
        corr = torch.einsum('bkn,bk->bn', A_I, u) * W_inv  # [B, n]
        z = z_ref - corr
        
        # Trust region (batched)
        if trust_region is not None:
            if isinstance(trust_region, (float, int)):
                z = torch.clamp(z, -float(trust_region), float(trust_region))
            else:
                tr = trust_region.to(device=device, dtype=dtype)
                if tr.numel() == 1:
                    z = torch.clamp(z, -float(tr.item()), float(tr.item()))
                else:
                    if tr.shape[-1] != n:
                        raise ValueError("trust_region tensor must have last dim = n")
                    z = torch.max(torch.min(z, tr.unsqueeze(0) if tr.dim() == 1 else tr), 
                                  -tr.unsqueeze(0) if tr.dim() == 1 else -tr)

    viol_after = _batch_Az(A_b, z) - b_b
    max_viol_after = float(viol_after.max().detach().cpu())

    info = ProjectionInfo(
        num_iters=it + 1,
        max_violation_before=max_viol_before,
        max_violation_after=max_viol_after,
        active_sizes=active_sizes,
    )
    return z, info


def demo_cbf_projection(device: str = "cpu") -> None:
    """
    一个最小 demo：
    - 随机生成 z_ref（可看作流模型预测速度）
    - 构造一些线性不等式约束（包含 box + 线性耦合）
    - 做投影并检查违规幅度下降
    - 演示 autograd：对 z_ref 求梯度
    """
    torch.manual_seed(7)
    dev = torch.device(device)

    B = 8
    n = 2

    # “流模型”给出的建议速度
    z_ref = torch.randn(B, n, device=dev, dtype=torch.float32) * 0.6
    z_ref.requires_grad_(True)

    # 约束：box + 耦合
    # -0.10 <= z1 <= 0.10
    # -0.20 <= z2 <= 0.20
    # z1 + z2 <= 0.15
    A = torch.tensor([
        [ 1.0,  0.0],
        [-1.0,  0.0],
        [ 0.0,  1.0],
        [ 0.0, -1.0],
        [ 1.0,  1.0],
    ], device=dev, dtype=torch.float32)
    b = torch.tensor([0.10, 0.10, 0.20, 0.20, 0.15], device=dev, dtype=torch.float32)

    z_safe, info = cbf_active_set_project(
        z_ref, A, b,
        trust_region=0.25,          # 防止过大修正（可选）
        max_iters=5,
        tol=1e-9,
        active_eps=1e-7,
        max_active=5,
        penalty_rho=1e7,
        detach_active_set=True,
    )

    # 打印违规对比
    def max_violation(z):
        return float((z @ A.T - b).max().detach().cpu())

    print("===== Demo: CBF projection effectiveness =====")
    print(f"max violation BEFORE: {max_violation(z_ref):.6e}")
    print(f"max violation AFTER : {max_violation(z_safe):.6e}")
    print(f"iters={info.num_iters}, active_sizes(sampled)={info.active_sizes[:min(10,len(info.active_sizes))]}")

    # 可微性演示：对 z_ref 求梯度
    loss = (z_safe ** 2).sum()
    loss.backward()
    grad_norm = float(z_ref.grad.norm().detach().cpu())
    print(f"autograd grad_norm(d loss / d z_ref) = {grad_norm:.6e}")
    print("NOTE: active-set 切换点处不可导，但大部分区域分段可微；训练中通常可用。")


if __name__ == "__main__":
    demo_cbf_projection("cpu")
