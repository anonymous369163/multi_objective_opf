"""
Generative VAE Utilities for Best-of-K Sampling

This module implements the generative VAE training paradigm that enables 
Best-of-K sampling for improved feasibility in multi-objective OPF problems.

Key features:
1. Softmin/CVaR aggregation to prevent distribution collapse
2. KL annealing + free-bits for stable latent learning
3. Two-stage Best-of-K selection for inference
4. Unified preference tensor construction

Author: Auto-generated from VAE improvement plan v5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, Tuple, List, Optional


# ==================== Preference Utilities ====================

def make_pref_tensors(lc: float, lambda_max: float, B: int, device: torch.device):
    """
    从偏好值 lc 构造网络输入和 NGT objective 用的偏好张量
    
    Args:
        lc: 当前偏好值（lambda_carbon），范围 0~lambda_max
            - 当前数据集：lc ∈ {0, 1, 3, ..., 99}
        lambda_max: 偏好最大值
            - 当前数据集：lambda_max = 99
        B: batch size
        device: 设备
    
    Returns:
        pref_norm: [B, 1] - 网络 conditioning，归一化到 [0, 1]
        pref_raw: [B, 2] - NGT objective，[λ_cost, λ_carbon]，和为 1
    
    权重计算逻辑：
        lc_ratio = lc / lambda_max  # 归一化到 [0, 1]
        λ_cost = 1 - lc_ratio       # lc=0 时 λ_cost=1（纯成本）
        λ_carbon = lc_ratio         # lc=99 时 λ_carbon=1（纯碳排放）
    """
    lc_ratio = lc / lambda_max if lambda_max > 0 else 0.0
    pref_norm = torch.full((B, 1), lc_ratio, device=device, dtype=torch.float32)
    lambda_cost = 1.0 - lc_ratio
    lambda_carbon = lc_ratio
    pref_raw = torch.tensor([[lambda_cost, lambda_carbon]], device=device, dtype=torch.float32)
    pref_raw = pref_raw.expand(B, -1).contiguous()
    return pref_norm, pref_raw


def lambda_to_key(lc: float) -> float:
    """将浮点偏好值转为稳定的 key（round到两位小数）"""
    return round(lc, 2)


def sample_multiple_preferences(lambda_values: List[float], n_prefs: int = 3, 
                                 include_neighbors: bool = True) -> List[float]:
    """
    采样多个偏好，确保包含相邻偏好
    
    Args:
        lambda_values: 所有可用的偏好值列表
        n_prefs: 采样数量
        include_neighbors: 是否确保包含相邻偏好对
    
    Returns:
        选中的偏好值列表
    """
    n = len(lambda_values)
    if n <= n_prefs:
        return list(lambda_values)
    
    if include_neighbors and n >= 2:
        # 随机选一个中心点
        center_idx = random.randint(0, n - 2)
        
        # 取相邻的两个
        pair = [lambda_values[center_idx], lambda_values[center_idx + 1]]
        
        # 再随机选 n_prefs - 2 个
        remaining_indices = [i for i in range(n) if i not in [center_idx, center_idx + 1]]
        if n_prefs > 2 and remaining_indices:
            extra_count = min(n_prefs - 2, len(remaining_indices))
            extra_indices = random.sample(remaining_indices, extra_count)
            extra = [lambda_values[i] for i in extra_indices]
        else:
            extra = []
        
        return pair + extra
    else:
        return random.sample(list(lambda_values), min(n_prefs, n))


# ==================== KL Divergence Utilities ====================

def compute_kl_with_freebits(mean: torch.Tensor, logvar: torch.Tensor, 
                              free_bits: float = 0.1) -> torch.Tensor:
    """
    计算 KL 散度，带 free-bits 保护防止后验坍塌
    
    Args:
        mean: [B, D] - 潜空间均值
        logvar: [B, D] - 潜空间对数方差
        free_bits: 每维度最小 KL 值
    
    Returns:
        KL loss (scalar)
    """
    kl_per_dim = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())  # [B, D]
    kl_clamped = torch.clamp(kl_per_dim, min=free_bits)
    return kl_clamped.sum(dim=1).mean()


# ==================== Training Schedule ====================

def get_loss_weight_schedule(epoch: int, config) -> Tuple[float, float, float]:
    """
    分阶段训练日程：
    - Phase 1 (warmup): 只训 L_rec + L_kl（逐步增加 KL）
    - Phase 2 (ramp): 逐步打开 δ_feas, η_obj
    - Phase 3 (stable): 全部权重达到目标值
    
    Returns:
        alpha_kl, delta_feas, eta_obj
    """
    warmup_epochs = getattr(config, 'generative_vae_warmup_epochs', 50)
    ramp_epochs = getattr(config, 'generative_vae_ramp_epochs', 50)
    
    alpha_kl_target = getattr(config, 'generative_vae_alpha_kl', 0.01)
    delta_feas_target = getattr(config, 'generative_vae_delta_feas', 0.1)
    eta_obj_target = getattr(config, 'generative_vae_eta_obj', 0.01)
    
    if epoch < warmup_epochs:
        alpha_kl = alpha_kl_target * epoch / max(warmup_epochs, 1)
        delta_feas = 0.0
        eta_obj = 0.0
    elif epoch < warmup_epochs + ramp_epochs:
        alpha_kl = alpha_kl_target
        progress = (epoch - warmup_epochs) / max(ramp_epochs, 1)
        delta_feas = delta_feas_target * progress
        eta_obj = eta_obj_target * progress
    else:
        alpha_kl = alpha_kl_target
        delta_feas = delta_feas_target
        eta_obj = eta_obj_target
    
    return alpha_kl, delta_feas, eta_obj


def get_tau_schedule(epoch: int, config) -> float:
    """
    τ 退火：从 tau_start 退火到 tau_end
    
    早期大 τ 更稳定，后期小 τ 更接近 hard best-of-K
    """
    warmup = getattr(config, 'generative_vae_warmup_epochs', 50)
    ramp = getattr(config, 'generative_vae_ramp_epochs', 50)
    tau_start = getattr(config, 'generative_vae_tau_start', 0.5)
    tau_end = getattr(config, 'generative_vae_tau_end', 0.1)
    
    total_anneal = warmup + ramp
    if epoch < total_anneal:
        progress = epoch / max(total_anneal, 1)
        tau = tau_start + (tau_end - tau_start) * progress
    else:
        tau = tau_end
    
    return tau


# ==================== Loss Functions ====================

def compute_feas_loss_by_mode(losses_k: torch.Tensor, tau: float, 
                               mode: str = 'softmin') -> torch.Tensor:
    """
    可配置的可行性损失聚合模式
    
    Args:
        losses_k: [K, B] - K 个采样点，B 个样本的约束违反
        tau: 温度参数
        mode: 聚合模式 ('softmin', 'softmean', 'cvar')
    
    Returns:
        [B] - 每个样本的聚合 loss
    """
    if mode == 'softmin':
        # Softmin: 近似 min，鼓励至少有一个好样本
        return -tau * torch.logsumexp(-losses_k / tau, dim=0)  # [B]
    elif mode == 'softmean':
        # Softmax 加权均值
        weights = torch.softmax(-losses_k / tau, dim=0)  # [K, B]
        return (weights * losses_k).sum(dim=0)  # [B]
    elif mode == 'cvar':
        # CVaR: 最差 20% 样本的均值
        rho = 0.2
        k_worst = max(1, int(losses_k.shape[0] * rho))
        sorted_losses, _ = torch.sort(losses_k, dim=0, descending=True)
        return sorted_losses[:k_worst].mean(dim=0)  # [B]
    else:
        raise ValueError(f"Unknown feas_mode: {mode}")


def compute_ngt_loss_chunked(
    ngt_loss_fn, y_flat: torch.Tensor, PQd_expanded: torch.Tensor, 
    pref_expanded: torch.Tensor, chunk_size: int = 1024
) -> Dict[str, torch.Tensor]:
    """
    分块调用 NGT loss，防止显存爆炸
    
    Args:
        ngt_loss_fn: NGT loss 函数
        y_flat: [N, output_dim] 预测电压
        PQd_expanded: [N, input_dim] 负荷数据
        pref_expanded: [N, 2] 偏好参数
        chunk_size: 每块大小
    
    Returns:
        dict: 包含 constraint_scaled, objective_per_sample 等
    """
    N = y_flat.shape[0]
    
    # 无论 N 大小，始终返回同一结构的 dict
    if N <= chunk_size:
        _, ld = ngt_loss_fn(y_flat, PQd_expanded, pref_expanded)
        return {
            'constraint_per_sample': ld['constraint_per_sample'],
            'constraint_scaled': ld['constraint_scaled'],
            'objective_per_sample': ld['objective_per_sample'],
        }
    
    # 分块处理
    constraint_chunks = []
    constraint_scaled_chunks = []
    objective_chunks = []
    
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        _, ld = ngt_loss_fn(
            y_flat[i:end], 
            PQd_expanded[i:end], 
            pref_expanded[i:end]
        )
        constraint_chunks.append(ld['constraint_per_sample'])
        constraint_scaled_chunks.append(ld['constraint_scaled'])
        objective_chunks.append(ld['objective_per_sample'])
    
    return {
        'constraint_per_sample': torch.cat(constraint_chunks),
        'constraint_scaled': torch.cat(constraint_scaled_chunks),
        'objective_per_sample': torch.cat(objective_chunks),
    }


def wrap_angle_difference(diff: torch.Tensor, NPred_Va: int) -> torch.Tensor:
    """Wrap angle differences to [-π, π] for Va dimensions."""
    if NPred_Va <= 0:
        return diff
    va_diff = diff[:, :NPred_Va]
    vm_diff = diff[:, NPred_Va:]
    wrapped_va = torch.atan2(torch.sin(va_diff), torch.cos(va_diff))
    return torch.cat([wrapped_va, vm_diff], dim=1)


def compute_reconstruction_loss(y_recon: torch.Tensor, y_target: torch.Tensor,
                                 NPred_Va: int = 0) -> torch.Tensor:
    """Compute reconstruction loss with angle wrapping."""
    diff = y_recon - y_target
    wrapped_diff = wrap_angle_difference(diff, NPred_Va)
    return (wrapped_diff ** 2).mean()


def compute_reconstruction_loss_dual(
    vae, scene: torch.Tensor, solution: torch.Tensor,
    mean: torch.Tensor, logvar: torch.Tensor, 
    NPred_Va: int, lambda_sample: float = 0.3
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    双重构损失：L_rec = L_rec_mean + λ * L_rec_sample
    
    Args:
        vae: VAE 模型
        scene: [B, input_dim] 场景特征
        solution: [B, output_dim] 目标解
        mean: [B, D] 潜空间均值
        logvar: [B, D] 潜空间对数方差
        NPred_Va: Va 维度数（用于角度包裹）
        lambda_sample: 采样重构的权重
    
    Returns:
        L_rec: 总重构损失
        info: 详细信息
    """
    # 1. Mean 重构
    y_mean = vae.decode(scene, mean)
    L_rec_mean = compute_reconstruction_loss(y_mean, solution, NPred_Va)
    
    # 2. Sample 重构
    z_sample = vae.reparameterize(mean, logvar)
    y_sample = vae.decode(scene, z_sample)
    L_rec_sample = compute_reconstruction_loss(y_sample, solution, NPred_Va)
    
    L_rec = L_rec_mean + lambda_sample * L_rec_sample
    
    return L_rec, {
        'L_rec_mean': L_rec_mean.item(),
        'L_rec_sample': L_rec_sample.item(),
    }


def compute_feasibility_loss_vectorized(
    vae, scene: torch.Tensor, mean: torch.Tensor, 
    logvar: torch.Tensor, ngt_loss_fn, PQd: torch.Tensor,
    pref_norm: torch.Tensor, pref_raw: torch.Tensor,
    n_samples: int = 5, tau: float = 0.1, feas_mode: str = 'softmin',
    chunk_size: int = 1024
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    向量化计算可行性损失 + 目标损失（使用可微路径）
    
    Args:
        vae: VAE 模型
        scene: [B, input_dim] 场景特征
        mean: [B, D] 潜空间均值
        logvar: [B, D] 潜空间对数方差
        ngt_loss_fn: NGT loss 函数（用于获取 params）
        PQd: [B, input_dim] 负荷数据
        pref_norm: [B, 1] 网络 conditioning 用（归一化后）
        pref_raw: [B, 2] NGT objective 用，[λ_cost, λ_carbon]
        n_samples: 采样数量 K
        tau: softmin 温度
        feas_mode: 聚合模式
        chunk_size: NGT 分块大小
    
    Returns:
        L_feas: 可行性损失 (scalar) - NOW WITH GRADIENTS!
        L_obj: 目标函数损失 (scalar) - NOW WITH GRADIENTS!
        info: 详细信息
    """
    B, D = mean.shape
    K = n_samples
    device = mean.device
    
    # 1. 向量化采样: [K, B, D]
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(K, B, D, device=device)
    z_samples = mean.unsqueeze(0) + std.unsqueeze(0) * eps  # [K, B, D]
    z_flat = z_samples.reshape(K * B, D)
    
    # 2. 向量化 decode
    scene_expanded = scene.unsqueeze(0).expand(K, -1, -1).reshape(K * B, -1)
    y_flat = vae.decode(scene_expanded, z_flat)
    
    # 3. 准备输入
    PQd_expanded = PQd.unsqueeze(0).expand(K, -1, -1).reshape(K * B, -1)
    pref_raw_expanded = pref_raw.unsqueeze(0).expand(K, -1, -1).reshape(K * B, -1)
    
    # 4. 使用可微路径计算 per-sample 指标
    # 这是关键修改：使用 compute_ngt_loss_chunked_differentiable 而非原来的版本
    # 原来的版本通过 torch.autograd.Function 的 params 存储值，没有梯度
    # 新版本使用纯 PyTorch 操作，保持梯度链
    ngt_params = ngt_loss_fn.params
    carbon_scale = getattr(ngt_params, 'carbon_scale', 30.0)
    
    loss_dict = compute_ngt_loss_chunked_differentiable(
        ngt_params, y_flat, PQd_expanded, pref_raw_expanded,
        chunk_size=chunk_size, carbon_scale=carbon_scale
    )
    
    # 获取有梯度的 constraint 和 objective
    constraint_flat = loss_dict['constraint_scaled']
    objective_flat = loss_dict['objective_per_sample']
    
    # 5. Reshape 回 [K, B]
    constraint_kb = constraint_flat.reshape(K, B)
    objective_kb = objective_flat.reshape(K, B)
    
    # 6. 聚合（使用当前 tau）
    L_feas = compute_feas_loss_by_mode(constraint_kb, tau, feas_mode).mean()
    
    # 7. Soft Best-of-K 目标损失
    weights = torch.softmax(-constraint_kb / tau, dim=0)  # [K, B]
    L_obj = (weights * objective_kb).sum(dim=0).mean()
    
    return L_feas, L_obj, {
        'constraint_kb': constraint_kb,
        'objective_kb': objective_kb,
        'weights': weights,
    }


# ==================== Inference Utilities ====================

def select_best_of_k(constraint_kb: torch.Tensor, objective_kb: torch.Tensor, 
                     threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    两阶段选最优：
    1. 选 constraint < threshold 的子集
    2. 在子集中选 objective 最小
    3. 若子集为空，退化为 constraint 最小
    
    Args:
        constraint_kb: [K, B] - 必须是 constraint_scaled（归一化后）
        objective_kb: [K, B] - objective_per_sample
        threshold: 可行性阈值，基于 constraint_scaled（如 0.01）
    
    Returns:
        best_idx: [B] - 每个样本的最佳索引
        is_feasible: [B] - 每个样本是否找到可行解
    """
    K, B = constraint_kb.shape
    device = constraint_kb.device
    
    best_idx = torch.zeros(B, dtype=torch.long, device=device)
    is_feasible = torch.zeros(B, dtype=torch.bool, device=device)
    
    for b in range(B):
        cv_b = constraint_kb[:, b]
        obj_b = objective_kb[:, b]
        
        feasible_mask = cv_b < threshold
        
        if feasible_mask.any():
            feasible_indices = torch.where(feasible_mask)[0]
            feasible_objs = obj_b[feasible_mask]
            best_in_feasible = feasible_objs.argmin()
            best_idx[b] = feasible_indices[best_in_feasible]
            is_feasible[b] = True
        else:
            best_idx[b] = cv_b.argmin()
            is_feasible[b] = False
    
    return best_idx, is_feasible


def compute_diversity_metrics(vae, scene: torch.Tensor, 
                               mean: torch.Tensor, logvar: torch.Tensor,
                               n_samples: int = 10) -> Dict[str, float]:
    """
    计算输出空间多样性指标
    
    Args:
        vae: VAE 模型
        scene: [B, input_dim]
        mean: [B, D]
        logvar: [B, D]
        n_samples: 采样数量
    
    Returns:
        dict: 多样性指标
    """
    B, D = mean.shape
    device = mean.device
    
    # 采样
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(n_samples, B, D, device=device)
    z_samples = mean.unsqueeze(0) + std.unsqueeze(0) * eps  # [K, B, D]
    
    # Decode
    z_flat = z_samples.reshape(n_samples * B, D)
    scene_expanded = scene.unsqueeze(0).expand(n_samples, -1, -1).reshape(n_samples * B, -1)
    y_flat = vae.decode(scene_expanded, z_flat)
    y_samples = y_flat.reshape(n_samples, B, -1)  # [K, B, output_dim]
    
    # 计算多样性指标
    output_std = y_samples.std(dim=0).mean().item()
    latent_std = std.mean().item()
    logvar_mean = logvar.mean().item()
    
    return {
        'output_std': output_std,
        'latent_std': latent_std,
        'logvar_mean': logvar_mean,
    }


# ==================== Main Loss Function ====================

# ==================== Differentiable Per-Sample Loss Computation ====================

def compute_per_sample_metrics_differentiable(
    V_pred: torch.Tensor,
    PQd: torch.Tensor,
    pref_raw: torch.Tensor,
    ngt_params,
    carbon_scale: float = 30.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-sample constraint violation and objective using PURE PyTorch operations.
    
    This function provides a DIFFERENTIABLE path for computing metrics, enabling 
    gradient flow from L_feas and L_obj back to the VAE decoder.
    
    The key difference from Penalty_V.forward is that this uses torch operations
    throughout, rather than numpy, so autograd can track gradients.
    
    Args:
        V_pred: [B, NPred_Va + NPred_Vm] - Predicted voltages (Va without slack, then Vm for non-ZIB)
        PQd: [B, num_Pd + num_Qd] - Load data in p.u.
        pref_raw: [B, 2] - [lambda_cost, lambda_carbon] weights
        ngt_params: DeepOPFNGTParams object containing system parameters
        carbon_scale: Carbon emission scale factor
    
    Returns:
        constraint_per_sample: [B] - Total squared constraint violation per sample
        objective_per_sample: [B] - Weighted objective (cost + carbon) per sample
    """
    device = V_pred.device
    Nsam = V_pred.shape[0]
    Nbus = ngt_params.Nbus
    
    # ============================================================
    # Step 1: Reconstruct full voltage vector (differentiable)
    # ============================================================
    
    # Insert slack bus Va=0 to get full non-ZIB voltage
    NPred_Va = ngt_params.NPred_Va
    NPred_Vm = ngt_params.NPred_Vm
    slack_idx = ngt_params.idx_bus_Pnet_slack[0]
    
    # Va and Vm for non-ZIB buses
    Va_pred = V_pred[:, :NPred_Va]  # [B, NPred_Va]
    Vm_pred = V_pred[:, NPred_Va:]  # [B, NPred_Vm]
    
    # Insert 0 for slack bus Va
    Va_nonZIB = torch.cat([
        Va_pred[:, :slack_idx],
        torch.zeros(Nsam, 1, device=device),
        Va_pred[:, slack_idx:]
    ], dim=1)  # [B, NPred_Vm]
    
    # Complex voltage for non-ZIB: V = Vm * exp(j*Va)
    Ve_nonZIB = Vm_pred * torch.cos(Va_nonZIB)  # Real part
    Vf_nonZIB = Vm_pred * torch.sin(Va_nonZIB)  # Imag part
    
    # Full voltage vectors (initialized)
    Ve = torch.zeros(Nsam, Nbus, device=device)
    Vf = torch.zeros(Nsam, Nbus, device=device)
    
    # Fill non-ZIB positions
    bus_Pnet_all = ngt_params.bus_Pnet_all
    Ve[:, bus_Pnet_all] = Ve_nonZIB
    Vf[:, bus_Pnet_all] = Vf_nonZIB
    
    # ============================================================
    # Step 2: Recover ZIB voltages using Kron Reduction (differentiable)
    # ============================================================
    if ngt_params.NZIB > 0 and ngt_params.param_ZIMV is not None:
        # param_ZIMV: [NZIB, NPred_Vm] complex matrix
        # V_ZIB = param_ZIMV @ V_nonZIB (in complex)
        param_ZIMV = torch.tensor(ngt_params.param_ZIMV, dtype=torch.complex64, device=device)
        
        # Complex voltage for non-ZIB
        V_nonZIB_complex = Ve_nonZIB + 1j * Vf_nonZIB  # [B, NPred_Vm]
        
        # Recover ZIB voltages: [B, NZIB]
        V_ZIB = torch.matmul(V_nonZIB_complex, param_ZIMV.T)  # [B, NZIB]
        
        # Fill ZIB positions
        bus_ZIB_all = ngt_params.bus_ZIB_all
        Ve[:, bus_ZIB_all] = V_ZIB.real
        Vf[:, bus_ZIB_all] = V_ZIB.imag
    
    # ============================================================
    # Step 3: Compute power injection S = V * conj(Ybus @ V) (differentiable)
    # ============================================================
    # Power flow equation: S = V * conj(I), where I = Y @ V = (G + jB) @ (Ve + jVf)
    # 
    # I = Y @ V = (G + jB) @ (Ve + jVf)
    #   = G@Ve - B@Vf + j*(G@Vf + B@Ve)
    # I_real = G@Ve - B@Vf
    # I_imag = G@Vf + B@Ve
    # 
    # conj(I) = I_real - j*I_imag
    # S = V * conj(I) = (Ve + jVf) * (I_real - j*I_imag)
    #   = Ve*I_real + Vf*I_imag + j*(Vf*I_real - Ve*I_imag)
    # 
    # P = Re[S] = Ve * (G@Ve - B@Vf) + Vf * (G@Vf + B@Ve)
    # Q = Im[S] = Vf * (G@Ve - B@Vf) - Ve * (G@Vf + B@Ve)
    # 
    # NOTE: Ybus is NOT symmetric for power systems with transformers/phase shifters!
    # Fixed bug 2024-12: Sign of B terms was incorrect.
    
    # Get G and B matrices from Ybus
    Ybus_dense = ngt_params.Ybus.toarray() if hasattr(ngt_params.Ybus, 'toarray') else ngt_params.Ybus
    G = torch.tensor(Ybus_dense.real, dtype=torch.float32, device=device)
    B = torch.tensor(Ybus_dense.imag, dtype=torch.float32, device=device)
    
    # Compute: for batch V [B, Nbus], G [Nbus, Nbus]
    # V @ G.T computes (G @ V.T).T, which is the correct G @ V for each sample
    GV_e = torch.matmul(Ve, G.T)  # [B, Nbus]: G @ Ve for each sample
    BV_f = torch.matmul(Vf, B.T)  # [B, Nbus]: B @ Vf for each sample
    GV_f = torch.matmul(Vf, G.T)  # [B, Nbus]: G @ Vf for each sample
    BV_e = torch.matmul(Ve, B.T)  # [B, Nbus]: B @ Ve for each sample
    
    # I_real = G@Ve - B@Vf, I_imag = G@Vf + B@Ve
    # P = Ve * I_real + Vf * I_imag
    # Q = Vf * I_real - Ve * I_imag
    Pred_P = Ve * (GV_e - BV_f) + Vf * (GV_f + BV_e)  # [B, Nbus]
    Pred_Q = Vf * (GV_e - BV_f) - Ve * (GV_f + BV_e)  # [B, Nbus]
    
    # ============================================================
    # Step 4: Parse load data and compute generator output
    # ============================================================
    num_Pd = len(ngt_params.bus_Pd)
    
    Pdtest = torch.zeros(Nsam, Nbus, device=device)
    Qdtest = torch.zeros(Nsam, Nbus, device=device)
    Pdtest[:, ngt_params.bus_Pd] = PQd[:, :num_Pd]
    Qdtest[:, ngt_params.bus_Qd] = PQd[:, num_Pd:]
    
    # Generator output: Pg = P + Pd at generator buses
    Pg = Pred_P + Pdtest
    Qg = Pred_Q + Qdtest
    
    # ============================================================
    # Step 5: Compute constraint violations (differentiable)
    # ============================================================
    MAXMIN_Pg = ngt_params.MAXMIN_Pg_tensor.to(device)  # [Npg, 2]
    MAXMIN_Qg = ngt_params.MAXMIN_Qg_tensor.to(device)  # [Nqg, 2]
    
    # Pg limits
    Pg_gen = Pg[:, ngt_params.bus_Pg]  # [B, Npg]
    loss_Pgi_per = (
        torch.clamp(Pg_gen - MAXMIN_Pg[:, 0], min=0).pow(2) +
        torch.clamp(MAXMIN_Pg[:, 1] - Pg_gen, min=0).pow(2)
    ).sum(dim=1)  # [B]
    
    # Qg limits
    Qg_gen = Qg[:, ngt_params.bus_Qg]  # [B, Nqg]
    loss_Qgi_per = (
        torch.clamp(Qg_gen - MAXMIN_Qg[:, 0], min=0).pow(2) +
        torch.clamp(MAXMIN_Qg[:, 1] - Qg_gen, min=0).pow(2)
    ).sum(dim=1)  # [B]
    
    # Load deviation (non-generator buses should have Pg=0, Qg=0)
    loss_Pdi_per = Pg[:, ngt_params.bus_Pnet_nonPg].pow(2).sum(dim=1)  # [B]
    loss_Qdi_per = Qg[:, ngt_params.bus_Pnet_nonQg].pow(2).sum(dim=1)  # [B]
    
    # ZIB voltage violation
    if ngt_params.NZIB > 0:
        VmLb = ngt_params.VmLb.to(device)
        VmUb = ngt_params.VmUb.to(device)
        Vm_ZIB = torch.sqrt(Ve[:, ngt_params.bus_ZIB_all].pow(2) + 
                           Vf[:, ngt_params.bus_ZIB_all].pow(2))  # [B, NZIB]
        loss_Vi_per = (
            torch.clamp(VmLb[0] - Vm_ZIB, min=0).pow(2) +
            torch.clamp(Vm_ZIB - VmUb[0], min=0).pow(2)
        ).sum(dim=1)  # [B]
    else:
        loss_Vi_per = torch.zeros(Nsam, device=device)
    
    # Total constraint violation per sample
    constraint_per_sample = loss_Pgi_per + loss_Qgi_per + loss_Pdi_per + loss_Qdi_per + loss_Vi_per
    
    # Normalized by number of constraint terms (for consistent threshold)
    n_constraint_terms = (
        len(ngt_params.bus_Pg) * 2 +  # Pg upper/lower
        len(ngt_params.bus_Qg) * 2 +  # Qg upper/lower
        len(ngt_params.bus_Pnet_nonPg) +  # Pd deviation
        len(ngt_params.bus_Pnet_nonQg) +  # Qd deviation
        (ngt_params.NZIB * 2 if ngt_params.NZIB > 0 else 0)  # V limits
    )
    n_constraint_terms = max(n_constraint_terms, 1)
    constraint_scaled = constraint_per_sample / n_constraint_terms
    
    # ============================================================
    # Step 6: Compute objective (cost + carbon)
    # ============================================================
    gencost = ngt_params.gencost_tensor.to(device)  # [Npg, 2]
    
    # Cost: c2*Pg^2 + c1*|Pg| (with extra penalty for negative Pg)
    absPg = torch.where(Pg_gen > 0, Pg_gen, -Pg_gen * 2.0)
    cost_per = (gencost[:, 0] * Pg_gen.pow(2) + gencost[:, 1] * absPg).sum(dim=1)  # [B]
    
    # Carbon emission (if multi-objective enabled)
    if ngt_params.use_multi_objective and ngt_params.gci_tensor is not None:
        gci = ngt_params.gci_tensor.to(device)  # [Npg]
        Pg_clamped = torch.clamp(Pg_gen, min=0)
        carbon_per = (Pg_clamped * gci).sum(dim=1) * carbon_scale  # [B]
        
        # Weighted objective
        lam_cost = pref_raw[:, 0]  # [B]
        lam_carbon = pref_raw[:, 1]  # [B]
        objective_per_sample = lam_cost * cost_per + lam_carbon * carbon_per
    else:
        objective_per_sample = cost_per
    
    return constraint_scaled, objective_per_sample


def compute_ngt_loss_chunked_differentiable(
    ngt_params,
    y_flat: torch.Tensor,
    PQd_expanded: torch.Tensor,
    pref_expanded: torch.Tensor,
    chunk_size: int = 1024,
    carbon_scale: float = 30.0,
) -> Dict[str, torch.Tensor]:
    """
    Differentiable version of compute_ngt_loss_chunked.
    
    Uses pure PyTorch operations to enable gradient flow.
    
    Args:
        ngt_params: DeepOPFNGTParams object
        y_flat: [N, output_dim] predicted voltages
        PQd_expanded: [N, input_dim] load data
        pref_expanded: [N, 2] preference parameters
        chunk_size: chunk size for processing
        carbon_scale: carbon emission scale factor
    
    Returns:
        dict with constraint_scaled and objective_per_sample tensors WITH gradients
    """
    N = y_flat.shape[0]
    
    if N <= chunk_size:
        constraint_scaled, objective_per_sample = compute_per_sample_metrics_differentiable(
            y_flat, PQd_expanded, pref_expanded, ngt_params, carbon_scale
        )
        return {
            'constraint_scaled': constraint_scaled,
            'objective_per_sample': objective_per_sample,
        }
    
    # Process in chunks
    constraint_chunks = []
    objective_chunks = []
    
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        c_scaled, obj_per = compute_per_sample_metrics_differentiable(
            y_flat[i:end],
            PQd_expanded[i:end],
            pref_expanded[i:end],
            ngt_params,
            carbon_scale
        )
        constraint_chunks.append(c_scaled)
        objective_chunks.append(obj_per)
    
    return {
        'constraint_scaled': torch.cat(constraint_chunks),
        'objective_per_sample': torch.cat(objective_chunks),
    }


def compute_generative_vae_loss(
    vae,
    scene: torch.Tensor,
    solutions_by_pref: Dict[float, torch.Tensor],
    lambda_values: List[float],
    sample_indices: torch.Tensor,
    ngt_loss_fn,
    PQd_data: torch.Tensor,
    config,
    epoch: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    生成式 VAE 完整训练损失
    
    Args:
        vae: LinearizedVAE 模型
        scene: [B, input_dim] 场景特征
        solutions_by_pref: Dict[lc -> [N, output_dim]] 按偏好组织的解
        lambda_values: 偏好值列表（原始值 0~99）
        sample_indices: [B] 样本索引
        ngt_loss_fn: NGT loss 函数
        PQd_data: [B, input_dim] 负荷数据
        config: 配置对象
        epoch: 当前 epoch
    
    Returns:
        total_loss: 总损失
        loss_dict: 详细损失字典
    """
    device = scene.device
    B = scene.shape[0]
    
    # 配置参数
    n_samples = getattr(config, 'generative_vae_n_samples', 5)
    free_bits = getattr(config, 'generative_vae_free_bits', 0.1)
    n_prefs = getattr(config, 'generative_vae_n_prefs', 3)
    feas_mode = getattr(config, 'generative_vae_feas_mode', 'softmin')
    lambda_sample = getattr(config, 'generative_vae_lambda_sample', 0.3)
    ngt_chunk_size = getattr(config, 'generative_vae_ngt_chunk_size', 1024)
    lambda_max = max(lambda_values)
    
    # 获取当前 epoch 的权重和 τ
    alpha_kl, delta_feas, eta_obj = get_loss_weight_schedule(epoch, config)
    tau = get_tau_schedule(epoch, config)
    
    # 采样多个偏好
    selected_prefs = sample_multiple_preferences(lambda_values, n_prefs, include_neighbors=True)
    
    total_L_rec = torch.zeros((), device=device)
    total_L_kl = torch.zeros((), device=device)
    total_L_feas = torch.zeros((), device=device)
    total_L_obj = torch.zeros((), device=device)
    
    for lc in selected_prefs:
        # 构造偏好张量
        pref_norm, pref_raw = make_pref_tensors(lc, lambda_max, B, device)
        
        # 获取对应解
        lc_key = lambda_to_key(lc)
        solutions = solutions_by_pref[lc_key].to(device)
        batch_solutions = solutions[sample_indices]
        
        # VAE encode
        mean, logvar = vae.encoder(scene, batch_solutions, pref_norm)
        
        # 1. 双重构损失
        L_rec, rec_info = compute_reconstruction_loss_dual(
            vae, scene, batch_solutions, mean, logvar,
            vae.NPred_Va, lambda_sample
        )
        total_L_rec = total_L_rec + L_rec
        
        # 2. KL 损失
        L_kl = compute_kl_with_freebits(mean, logvar, free_bits)
        total_L_kl = total_L_kl + L_kl
        
        # 3. 可行性 + 目标函数损失（仅在 ramp 阶段后启用）
        if delta_feas > 0 or eta_obj > 0:
            L_feas, L_obj, info = compute_feasibility_loss_vectorized(
                vae, scene, mean, logvar, ngt_loss_fn, PQd_data,
                pref_norm, pref_raw,
                n_samples=n_samples, tau=tau, feas_mode=feas_mode,
                chunk_size=ngt_chunk_size
            )
            total_L_feas = total_L_feas + L_feas
            total_L_obj = total_L_obj + L_obj
    
    # 平均多偏好
    n = len(selected_prefs)
    L_rec = total_L_rec / n
    L_kl = total_L_kl / n
    L_feas = total_L_feas / n if delta_feas > 0 else torch.zeros((), device=device)
    L_obj = total_L_obj / n if eta_obj > 0 else torch.zeros((), device=device)
    
    # 总损失
    total_loss = L_rec + alpha_kl * L_kl + delta_feas * L_feas + eta_obj * L_obj
    
    return total_loss, {
        'total': total_loss.item(),
        'L_rec': L_rec.item(),
        'L_kl': L_kl.item(),
        'L_feas': L_feas.item() if torch.is_tensor(L_feas) else L_feas,
        'L_obj': L_obj.item() if torch.is_tensor(L_obj) else L_obj,
        'alpha_kl': alpha_kl,
        'delta_feas': delta_feas,
        'eta_obj': eta_obj,
        'tau': tau,
        'n_prefs': n,
    }
