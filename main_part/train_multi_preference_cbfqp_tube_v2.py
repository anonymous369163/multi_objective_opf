#!/usr/bin/env python
# coding: utf-8
"""
Multi-Preference Training with CBF-QP Safety Projection (Tube Method)

Trains preference-conditioned Flow models using preference trajectory mode
with optional CBF-QP safety projection during training.

For standard training without CBF-QP, use train_multi_preference.py instead.

Author: Peng Yue
Date: December 2025

Usage:
    MODEL_TYPE=rectified MULTI_PREF_USE_CBF_QP_TRAIN=1 python train_multi_preference_cbfqp_tube.py
    DEBUG=1 python train_multi_preference_cbfqp_tube.py  # Evaluation only
"""

import torch
import torch.nn.functional as F  # required for smooth_l1_loss
import time
import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BaseConfig, _SCRIPT_DIR
from data_loader import load_multi_preference_dataset, create_multi_preference_dataloader

# [CBF-QP TRAIN] Projection layer (training-time)
from cbf_qp_train_layer_tube import CBFQPTrainConfig, CBFQPProjectorNGT  # [TUBE]


# ==================== Multi-Preference Configuration ====================

class MultiPreferenceConfig(BaseConfig):
    """Configuration for multi-preference supervised training."""
    
    def __init__(self):
        super().__init__()
        
        # ==================== Multi-Preference Training ==================== 
        self.multi_pref_dataset_path = os.path.join(
            os.path.dirname(_SCRIPT_DIR), 'saved_data', 'multi_preference_solutions', 'fully_covered_dataset_2026-01-02.pt'
        )
        
        # Training parameters
        self.multi_pref_epochs = int(os.environ.get('MULTI_PREF_EPOCHS', '1000'))   # origin: 1000
        self.multi_pref_lr = float(os.environ.get('MULTI_PREF_LR', '1e-4'))
        self.multi_pref_flow_steps = int(os.environ.get('MULTI_PREF_FLOW_STEPS', '10'))
        self.multi_pref_batch_size = int(os.environ.get('MULTI_PREF_BATCH_SIZE', '100'))  # origin: 50
        
        # Validation split
        self.multi_pref_val_ratio = float(os.environ.get('MULTI_PREF_VAL_RATIO', '0.2'))
        self.multi_pref_random_seed = int(os.environ.get('MULTI_PREF_RANDOM_SEED', '42'))
        
        # Model Architecture
        self.hidden_dim = int(os.environ.get('HIDDEN_DIM', '128'))
        self.num_layers = int(os.environ.get('NUM_LAYERS', '2'))
        self.latent_dim = int(os.environ.get('LATENT_DIM', '64'))  # VAE anchor
        self.time_step = 1000
        
        # Batch size (needed by DeepOPFNGTLoss for HV guidance)
        self.batch_size_training = self.multi_pref_batch_size
        
        # Loss weights for preference trajectory training
        # loss = alpha * loss_velocity + beta * loss_endpoint
        # 注意: beta 过大会导致模型只关注终点匹配而忽略整体速度场学习
        self.multi_pref_loss_alpha = float(os.environ.get('MULTI_PREF_LOSS_ALPHA', '1.0'))    # 速度场损失权重
        self.multi_pref_loss_beta = float(os.environ.get('MULTI_PREF_LOSS_BETA', '10.0'))     # 终点误差权重(10-100之间较合理)
        
        # Multi-step rollout method
        self.multi_pref_rollout_use_rk2 = os.environ.get('MULTI_PREF_ROLLOUT_USE_RK2', 'False').lower() == 'true'
        # True: RK2(Heun)二阶精度, 每步2次模型调用, 更稳定
        # False: Euler一阶精度, 每步1次模型调用, 更快

        # ==================== HV Guidance (Pareto Front Optimization) ====================
        # Enable differentiable HV proxy loss to guide model towards better Pareto fronts
        # HV weight ramps up progressively: 0 during warmup, then linear ramp to target
        #
        # 是否启用 HV 引导 (默认开启: '1')
        self.multi_pref_hv_enabled = os.environ.get('MULTI_PREF_HV_ENABLED', '1').lower() in ['1', 'true', 'yes']
        # HV 损失的目标权重 (渐进到达此值)
        self.multi_pref_hv_weight = float(os.environ.get('MULTI_PREF_HV_WEIGHT', '0.1'))
        # HV 引导开始的 epoch 比例 (默认 30% epochs 后开始)
        self.multi_pref_hv_start_ratio = float(os.environ.get('MULTI_PREF_HV_START_RATIO', '0.3'))
        # HV 权重从 0 增长到目标值所需的 epoch 比例 (默认再用 30% epochs)
        self.multi_pref_hv_warmup_ratio = float(os.environ.get('MULTI_PREF_HV_WARMUP_RATIO', '0.3'))
        # HV Proxy 的 softmin 温度参数 (较小值使 softmin 更接近 min)
        self.multi_pref_hv_tau = float(os.environ.get('MULTI_PREF_HV_TAU', '0.05'))
        # HV Proxy 的幂次参数 (影响对 HV 贡献的敏感度)
        self.multi_pref_hv_power = float(os.environ.get('MULTI_PREF_HV_POWER', '2.0'))
        # 参考点 margin: ref = max(obj) * (1 + margin)
        self.multi_pref_hv_ref_margin = float(os.environ.get('MULTI_PREF_HV_REF_MARGIN', '0.05'))
        # 可选: 直接的目标函数损失权重 (加权标量化)
        self.multi_pref_obj_weight = float(os.environ.get('MULTI_PREF_OBJ_WEIGHT', '0.0'))

        # ==================== CBF-QP 超参数说明 (IEEE 118 案例) ====================
        # 
        # CBF-QP 在训练时将模型预测的增量 delta 投影到安全集内:
        #   min ||delta_safe - delta_pred||^2  s.t. A*delta_safe <= b + tube_eps
        #
        # 关键超参数分为4类: (1)核心参数 (2)信赖域 (3)约束选择 (4)QP求解器
        # =========================================================================

        # ---------- (1) 核心参数 ----------
        # 是否启用训练时CBF-QP投影 (默认关闭: '0', 仅在测试时开启)
        self.multi_pref_use_cbf_qp_train = os.environ.get('MULTI_PREF_USE_CBF_QP_TRAIN', '0').lower() in ['1', 'true', 'yes']
        
        # CBF强度参数 beta ∈ (0, 1]
        # - beta=1: 一步完全修正违约(激进), 可能导致过大投影
        # - beta=0.5: 每步修正50%违约(温和), 允许多步逐渐收敛
        # - beta<0.3: 投影太小可能无法有效约束
        # 建议: 0.5 (平衡安全性和训练稳定性)
        self.multi_pref_cbf_beta = float(os.environ.get('MULTI_PREF_CBF_BETA', '0.5'))
        
        # 每个batch应用投影的概率
        # - 1.0: 每个batch都投影(最安全,但计算开销大)
        # - 0.8: 80%batch投影(稍快,略微降低安全保证)
        # - 0.3: 30%batch投影(快速模式,适合初期训练)
        # 建议: 0.5 (平衡速度和安全性)
        self.multi_pref_cbf_apply_prob = float(os.environ.get('MULTI_PREF_CBF_APPLY_PROB', '0.5'))

        # ---------- (2) 信赖域参数 ----------
        # 限制单步投影的最大变化量, 防止线性化误差导致的过大投影
        #
        # IEEE 118案例特点:
        # - Va(相角)典型变化: 相邻preference间 ~0.01-0.05 rad
        # - Vm(幅值)范围: [0.94, 1.06] p.u., 相邻preference间变化 ~0.001-0.01 p.u.
        #
        # trust_va: 相角信赖域半径 (单位: rad)
        # - 0.10 rad ≈ 5.7° (保守,适合训练初期)
        # - 0.15 rad ≈ 8.6° (稍激进)
        # 建议: 0.10 (IEEE 118的Va变化较小,保守设置更稳定)
        self.multi_pref_cbf_trust_va = float(os.environ.get('MULTI_PREF_CBF_TRUST_VA', '0.10'))
        
        # trust_vm: 幅值信赖域半径 (单位: p.u.)
        # - IEEE 118的Vm范围仅0.12 p.u., 需要较小的信赖域
        # - 0.01 p.u. 约为Vm范围的8%
        # - 0.02 p.u. 约为Vm范围的17%
        # 建议: 0.01 (IEEE 118的Vm范围窄,需要精细控制)
        self.multi_pref_cbf_trust_vm = float(os.environ.get('MULTI_PREF_CBF_TRUST_VM', '0.01'))

        # ---------- (3) 约束选择参数 ----------
        # 只选择"接近边界"或"已违反"的约束加入QP, 降低求解复杂度
        # eps_*: 距离边界 < eps 的约束会被选中
        # k_*: 每个样本最多保留的约束数量
        #
        # eps_vm: Vm约束选择阈值
        # - IEEE 118的Vm范围窄(0.12), 需要较小阈值才能精确选择
        # 建议: 0.01 (选择距边界1%范围内的Vm约束)
        self.multi_pref_cbf_eps_vm = float(os.environ.get('MULTI_PREF_CBF_EPS_VM', '0.01'))
        
        # eps_pqg: 发电机Pg/Qg约束选择阈值
        # - 功率约束通常归一化到[0,1], 阈值可稍大
        # 建议: 0.02
        self.multi_pref_cbf_eps_pqg = float(os.environ.get('MULTI_PREF_CBF_EPS_PQG', '0.02'))
        
        # eps_branch: 支路功率流约束选择阈值
        # 建议: 0.02
        self.multi_pref_cbf_eps_branch = float(os.environ.get('MULTI_PREF_CBF_EPS_BRANCH', '0.02'))
        
        # k_vm: 每样本最多选择的Vm约束数
        # - IEEE 118有118*2=236个Vm约束(上下界)
        # - 通常只有少数接近边界
        # 建议: 32 (平衡精度和速度)
        self.multi_pref_cbf_k_vm = int(os.environ.get('MULTI_PREF_CBF_K_VM', '32'))
        
        # k_pqg: 每样本最多选择的Pg/Qg约束数
        # - IEEE 118有54台发电机, 约216个Pg/Qg约束
        # 建议: 32 (平衡精度和速度)
        self.multi_pref_cbf_k_pqg = int(os.environ.get('MULTI_PREF_CBF_K_PQG', '32'))
        
        # k_branch: 每样本最多选择的支路约束数
        # - IEEE 118有186条支路, 约372个支路功率约束
        # - 支路约束通常不太紧
        # 建议: 16 (平衡精度和速度)
        self.multi_pref_cbf_k_branch = int(os.environ.get('MULTI_PREF_CBF_K_BRANCH', '16'))

        # ---------- (4) QP求解器参数 ----------
        # max_iters: QP求解最大迭代次数
        # - 使用的是 differentiable QP solver (如 qpth 或自定义迭代法)
        # - 迭代次数越多越精确但越慢
        # 建议: 3 (平衡精度和速度, 大多数情况2-3次即可收敛)
        self.multi_pref_cbf_max_iters = int(os.environ.get('MULTI_PREF_CBF_MAX_ITERS', '3'))
        
        # detach_active_set: 是否在反向传播时detach活跃集
        # - True: 活跃集不参与梯度计算, 训练更稳定
        # - False: 完整可微, 但可能导致梯度不稳定
        # 建议: True (训练稳定性优先)
        self.multi_pref_cbf_detach_active_set = os.environ.get('MULTI_PREF_CBF_DETACH_ACTIVE_SET', '1').lower() in ['1', 'true', 'yes']
        
        # penalty_rho: 罚函数/增广拉格朗日的惩罚系数
        # - 用于处理不等式约束的内点/罚函数方法
        # - 过小: 约束不严格; 过大: 数值不稳定
        # 建议: 1e7 (通常不需要调整)
        self.multi_pref_cbf_penalty_rho = float(os.environ.get('MULTI_PREF_CBF_PENALTY_RHO', '1e7'))
        
        # distill_weight: 蒸馏损失权重
        # - 鼓励模型预测的速度 v_pred 接近投影后的速度 v_used
        # - 目标: 减少推理时对投影的依赖, 让模型学会"自觉"安全
        # - 0.0: 禁用蒸馏
        # - 0.1: 轻度蒸馏
        # 建议: 0.1 (轻度蒸馏, 不过分干扰主损失)
        self.multi_pref_cbf_distill_weight = float(os.environ.get('MULTI_PREF_CBF_DISTILL_WEIGHT', '0.1'))

        # ==================== Tube Schedule (约束松弛调度) ====================
        # 训练时逐渐收紧约束: A*delta <= b + eps_tube
        # eps_tube 从 start 线性/余弦衰减到 end
        # 优点: 训练初期允许更大探索, 后期逐渐严格
        #
        # tube_eps_vm: Vm约束松弛量
        # - start=0.005: 初始允许Vm超出边界0.005 p.u.(约Vm范围的4%)
        # - end=0.0: 最终严格满足约束
        # 建议: start=0.005, end=0.0
        self.multi_pref_tube_eps_vm_start = float(os.environ.get('MULTI_PREF_TUBE_EPS_VM_START', '0.005'))
        self.multi_pref_tube_eps_vm_end = float(os.environ.get('MULTI_PREF_TUBE_EPS_VM_END', '0.00'))
        
        # tube_eps_pqg: Pg/Qg约束松弛量
        # 建议: start=0.01, end=0.0
        self.multi_pref_tube_eps_pqg_start = float(os.environ.get('MULTI_PREF_TUBE_EPS_PQG_START', '0.01'))
        self.multi_pref_tube_eps_pqg_end = float(os.environ.get('MULTI_PREF_TUBE_EPS_PQG_END', '0.00'))
        
        # tube_eps_branch: 支路约束松弛量
        # 建议: start=0.01, end=0.0
        self.multi_pref_tube_eps_branch_start = float(os.environ.get('MULTI_PREF_TUBE_EPS_BRANCH_START', '0.01'))
        self.multi_pref_tube_eps_branch_end = float(os.environ.get('MULTI_PREF_TUBE_EPS_BRANCH_END', '0.00'))
        
        # tube_schedule: 松弛量调度方式
        # - 'linear': eps(t) = start + (end-start)*t, 线性衰减
        # - 'cosine': eps(t) = end + (start-end)*0.5*(1+cos(π*t)), 余弦衰减(更平滑)
        # - 'exp': eps(t) = end + (start-end)*exp(-k*t), 指数衰减(快速收紧)
        # 建议: 'cosine' (训练中后期衰减更快, 符合收敛特性)
        self.multi_pref_tube_schedule = os.environ.get('MULTI_PREF_TUBE_SCHEDULE', 'cosine')
        
        # tube_exp_k: 指数衰减系数 (仅当schedule='exp'时使用)
        # 建议: 5.0
        self.multi_pref_tube_exp_k = float(os.environ.get('MULTI_PREF_TUBE_EXP_K', '5.0'))

        # ==================== 其他优化选项 ====================
        # gate_before_solve: 如果delta已满足约束, 跳过QP求解
        # - 节省计算, 尤其是训练后期大部分delta已安全
        # 建议: True
        self.multi_pref_cbf_gate_before_solve = os.environ.get('MULTI_PREF_CBF_GATE', '1').lower() in ['1', 'true', 'yes']
        
        # gate_eps: Gate检查的数值容差
        # 建议: 1e-9 (不需要调整)
        self.multi_pref_cbf_gate_eps = float(os.environ.get('MULTI_PREF_CBF_GATE_EPS', '1e-9'))
        
        # rk2_rebuild_ab: RK2第二步是否重建A,b矩阵
        # - True: 在中间点 x_euler 重新线性化, 更精确但更慢
        # - False: 复用第一步的A,b, 更快但略有线性化误差
        # 建议: False (性能优先, 误差通常可接受)
        self.multi_pref_cbf_rk2_rebuild_ab = os.environ.get('MULTI_PREF_CBF_RK2_REBUILD_AB', '0').lower() in ['1', 'true', 'yes']
        
        # bridge_weight: 投影幅度惩罚权重
        # - 惩罚 ||delta_safe - delta_pred||^2, 鼓励模型预测更接近安全值
        # - 0.0: 禁用
        # - 0.1-1.0: 轻度到中度惩罚
        # 建议: 0.0 (初次训练时禁用, 避免干扰学习; 后期微调时可启用)
        self.multi_pref_bridge_weight = float(os.environ.get('MULTI_PREF_BRIDGE_WEIGHT', '0.0'))

        # ==================== Evaluation Config ====================
        self.use_cbf_qp_post = os.environ.get('USE_CBF_QP_POST', '0').lower() in ['1', 'true', 'yes']
        self.post_process_method = os.environ.get('POST_PROCESS_METHOD', '').strip().lower()
        if self.use_cbf_qp_post and not self.post_process_method:
            self.post_process_method = 'cbf_qp'
        self.cbf_beta = float(os.environ.get('CBF_BETA', '1.0'))
        self.cbf_trust_region = float(os.environ.get('CBF_TRUST_REGION', '0.05'))
        
        # Preference conditioning
        self.pref_dim = 1
        
        # VAE config (for loading pretrained anchor model)
        self.vae_use_preference_aware = True
        
        # ==================== Flow Best-of-K Evaluation ====================
        self.flow_best_of_k = int(os.environ.get('FLOW_BEST_OF_K', '32'))
        self.flow_selection_mode = os.environ.get('FLOW_SELECTION_MODE', 'constraint')
        
        # ==================== Training Control ====================
        self.weight_decay = 1e-6
        self.p_epoch = 10
        self.s_epoch = 800
        
    def print_config(self):
        """Print configuration summary."""
        super().print_config()
        print(f"\n[Training Config]")
        print(f"  Epochs: {self.multi_pref_epochs}, LR: {self.multi_pref_lr}, Batch: {self.multi_pref_batch_size}")
        print(f"  Training mode: trajectory")
        print(f"\n[CBF-QP Safety Projection]")
        print(f"  Enabled: {self.multi_pref_use_cbf_qp_train}, Beta: {self.multi_pref_cbf_beta}")
        print(f"  Trust region: Va={self.multi_pref_cbf_trust_va}, Vm={self.multi_pref_cbf_trust_vm}")
        print(f"\n[Tube Schedule]")
        print(f"  Vm: {self.multi_pref_tube_eps_vm_start} -> {self.multi_pref_tube_eps_vm_end}")
        print(f"  Pqg: {self.multi_pref_tube_eps_pqg_start} -> {self.multi_pref_tube_eps_pqg_end}")
        print(f"  Schedule: {self.multi_pref_tube_schedule}")
        if self.multi_pref_hv_enabled:
            print(f"\n[HV Guidance (Pareto Front)]")
            print(f"  Enabled: {self.multi_pref_hv_enabled}, Weight: {self.multi_pref_hv_weight}")
            print(f"  Schedule: start={self.multi_pref_hv_start_ratio}, warmup={self.multi_pref_hv_warmup_ratio}")
            print(f"  Proxy params: tau={self.multi_pref_hv_tau}, power={self.multi_pref_hv_power}, ref_margin={self.multi_pref_hv_ref_margin}")
            if self.multi_pref_obj_weight > 0:
                print(f"  Obj weight: {self.multi_pref_obj_weight}")


def get_multi_preference_config():
    """Get multi-preference training configuration."""
    return MultiPreferenceConfig()


# ==================== Utility Functions ====================

def wrap_angle_difference(dx, NPred_Va):
    """Wrap angle difference to [-pi, pi] for Va dimensions."""
    if torch.is_tensor(dx):
        dx_wrapped = dx.clone()
        if NPred_Va > 0:
            dx_wrapped[..., :NPred_Va] = torch.atan2(
                torch.sin(dx[..., :NPred_Va]), 
                torch.cos(dx[..., :NPred_Va])
            )
        return dx_wrapped
    else:
        dx_np = np.asarray(dx).copy()
        if NPred_Va > 0:
            for i in range(min(NPred_Va, dx_np.shape[-1])):
                dx_np[..., i] = np.arctan2(np.sin(dx_np[..., i]), np.cos(dx_np[..., i]))
        return dx_np


# ==================== HV Guidance Helper Functions ====================

def _extract_cost_carbon_torch(loss_dict, device, dtype):
    """Extract differentiable per-sample cost/carbon tensors from DeepOPFNGTLoss output.
    
    Returns (cost_tensor, carbon_tensor) if available with gradients, else (None, None).
    """
    if loss_dict is None:
        return None, None

    def _pick(keys):
        for k in keys:
            v = loss_dict.get(k, None)
            if torch.is_tensor(v):
                t = v
                if t.dim() > 1:
                    t = t.view(-1)
                return t
        return None

    # Prefer the torch versions that keep gradients
    cost_t = _pick(['cost_per_sample_torch', 'cost_per_sample_tensor', 'cost_per_sample'])
    carbon_t = _pick(['carbon_per_sample_torch', 'carbon_per_sample_tensor', 'carbon_per_sample'])

    # If these are numpy arrays (detached), HV gradient cannot flow — return None to disable HV term.
    if (cost_t is None) or (carbon_t is None):
        return None, None

    # Ensure dtype/device
    cost_t = cost_t.to(device=device, dtype=dtype)
    carbon_t = carbon_t.to(device=device, dtype=dtype)
    return cost_t, carbon_t


def _softmin(x, tau=0.05, dim=-1):
    """Differentiable soft-min with temperature tau."""
    return -tau * torch.logsumexp(-x / max(tau, 1e-12), dim=dim)


def _psl_hv1_proxy_loss(cost, carbon, lam_raw, ref_cost, ref_carbon, tau=0.05, power=2.0):
    """A lightweight 2-objective PSL-HV1-style proxy loss (maximize HV => minimize negative HV proxy).

    - Minimization objectives: cost, carbon
    - Reference point r should be a *dominated* (worse) point: r_i >= f_i.
    - Direction weights w are derived from lambda_carbon (lam_raw) and normalized.

    Returns a scalar loss (to be minimized).
    """
    if lam_raw.dim() == 1:
        lam = lam_raw.view(-1, 1)
    else:
        lam = lam_raw

    lam = torch.clamp(lam, min=0.0)

    # 2D direction weights, normalized (L1)
    # w = [1/(1+lam), lam/(1+lam)] approximately
    w = torch.cat([torch.ones_like(lam), lam + 1e-6], dim=1)
    w = w / (w.sum(dim=1, keepdim=True) + 1e-12)  # [B,2]

    f = torch.stack([cost.view(-1), carbon.view(-1)], dim=1)  # [B,2]
    r = torch.tensor([ref_cost, ref_carbon], device=f.device, dtype=f.dtype).view(1, 2)

    # Ray distance-like quantity: how far from reference along the preference direction
    t = (r - f) / (w + 1e-12)  # [B,2]
    rho = _softmin(t, tau=tau, dim=1)  # [B]
    rho = torch.clamp(rho, min=0.0)
    hv_proxy = rho ** float(power)
    # Maximize HV => minimize negative HV proxy
    return -hv_proxy.mean()


def _get_hv_weight(epoch, num_epochs, config):
    """Progressive HV weight schedule: 0 -> target_weight.
    
    Returns current HV weight based on training progress.
    """
    if not bool(getattr(config, "multi_pref_hv_enabled", False)):
        return 0.0
    
    start_ratio = float(getattr(config, "multi_pref_hv_start_ratio", 0.3))
    warmup_ratio = float(getattr(config, "multi_pref_hv_warmup_ratio", 0.3))
    target_weight = float(getattr(config, "multi_pref_hv_weight", 0.1))
    
    progress = epoch / max(num_epochs - 1, 1)
    if progress < start_ratio:
        return 0.0
    elif progress < start_ratio + warmup_ratio:
        # Linear ramp up
        ramp = (progress - start_ratio) / max(warmup_ratio, 1e-6)
        return target_weight * ramp
    else:
        return target_weight


# ==================== Training Functions ====================

def _generate_model_filename(config, model_type, epoch=None, is_final=False):
    """
    生成模型文件名。
    
    格式: model_multi_pref_{type}_traj_{cbf_tag}[_E{epoch}|_final].pth
    """
    # CBF-QP投影状态
    use_cbf = getattr(config, 'multi_pref_use_cbf_qp_train', False)
    if use_cbf:
        beta = getattr(config, 'multi_pref_cbf_beta', 0.5)
        cbf_tag = f"cbf{beta:.1f}".replace('.', '')
    else:
        cbf_tag = "nocbf"
    
    base = f"model_multi_pref_{model_type}_traj_{cbf_tag}"
    if is_final:
        return f"{base}_final.pth"
    elif epoch is not None:
        return f"{base}_E{epoch}.pth"
    return f"{base}.pth"


def train_multi_preference(config, model, multi_pref_data, sys_data, device,
                           model_type='rectified', pretrain_model=None):
    """
    Train Flow model using preference trajectory mode with optional CBF-QP projection.
    
    Note: Only supports Flow models (rectified, gaussian, etc.) with preference trajectory.
          For standard training of VAE/simple models, use train_multi_preference.py.
    """
    # ==================== CUDA Performance Optimization ====================
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster matmul
        torch.backends.cudnn.allow_tf32 = True
    
    print('=' * 60)
    print(f'CBF-QP Tube Training - Model: {model_type}')
    print('=' * 60)
    
    y_train_by_pref = {lc: y.to(device) for lc, y in multi_pref_data['y_train_by_pref'].items()}

    # [CBF-QP TRAIN] Build training-time projector (optional)
    cbf_cfg = CBFQPTrainConfig(
        enabled=bool(getattr(config, "multi_pref_use_cbf_qp_train", False)),
        beta=float(getattr(config, "multi_pref_cbf_beta", 0.5)),
        max_iters=int(getattr(config, "multi_pref_cbf_max_iters", 6)),
        detach_active_set=bool(getattr(config, "multi_pref_cbf_detach_active_set", True)),
        penalty_rho=float(getattr(config, "multi_pref_cbf_penalty_rho", 1e7)),
        trust_region_va=float(getattr(config, "multi_pref_cbf_trust_va", 0.10)),
        trust_region_vm=float(getattr(config, "multi_pref_cbf_trust_vm", 0.02)),
        slack_eps_vm=float(getattr(config, "multi_pref_cbf_eps_vm", 0.02)),
        slack_eps_pqg=float(getattr(config, "multi_pref_cbf_eps_pqg", 0.02)),
        slack_eps_branch=float(getattr(config, "multi_pref_cbf_eps_branch", 0.02)),
        k_vm=int(getattr(config, "multi_pref_cbf_k_vm", 64)),
        k_pqg=int(getattr(config, "multi_pref_cbf_k_pqg", 64)),
        k_branch=int(getattr(config, "multi_pref_cbf_k_branch", 32)),
        apply_prob=float(getattr(config, "multi_pref_cbf_apply_prob", 1.0)),
        distill_weight=float(getattr(config, "multi_pref_cbf_distill_weight", 0.0)),

        tube_eps_vm_start=float(getattr(config, "multi_pref_tube_eps_vm_start", 0.0)),
        tube_eps_vm_end=float(getattr(config, "multi_pref_tube_eps_vm_end", 0.0)),
        tube_eps_pqg_start=float(getattr(config, "multi_pref_tube_eps_pqg_start", 0.0)),
        tube_eps_pqg_end=float(getattr(config, "multi_pref_tube_eps_pqg_end", 0.0)),
        tube_eps_branch_start=float(getattr(config, "multi_pref_tube_eps_branch_start", 0.0)),
        tube_eps_branch_end=float(getattr(config, "multi_pref_tube_eps_branch_end", 0.0)),
        tube_schedule=str(getattr(config, "multi_pref_tube_schedule", "linear")),
        tube_exp_k=float(getattr(config, "multi_pref_tube_exp_k", 5.0)),
        gate_before_solve=bool(getattr(config, "multi_pref_cbf_gate_before_solve", True)),
        gate_eps=float(getattr(config, "multi_pref_cbf_gate_eps", 1e-9)),
    )
    projector = None
    if cbf_cfg.enabled:
        try:
            projector = CBFQPProjectorNGT(sys_data, multi_pref_data, device, cbf_cfg)
            print(f"[CBF-QP TRAIN] enabled: beta={cbf_cfg.beta}, apply_prob={cbf_cfg.apply_prob}, "
                  f"trust(Va)={cbf_cfg.trust_region_va}, trust(Vm)={cbf_cfg.trust_region_vm}")
        except Exception as e:
            print(f"[CBF-QP TRAIN] WARNING: failed to build projector, fallback to no projection. Error: {e}")
            projector = None
    
    lambda_values = multi_pref_data['lambda_carbon_values']
    n_train = multi_pref_data['n_train']
    
    # ==================== HV Guidance: Create DeepOPFNGTLoss instance ====================
    loss_fn = None
    hv_enabled = bool(getattr(config, "multi_pref_hv_enabled", False))
    hv_w = float(getattr(config, "multi_pref_hv_weight", 0.0))
    obj_w = float(getattr(config, "multi_pref_obj_weight", 0.0))

    if hv_enabled and (hv_w > 0 or obj_w > 0):
        try:
            from deepopf_ngt_loss import DeepOPFNGTLoss
            loss_fn = DeepOPFNGTLoss(sys_data, config)
            loss_fn.cache_to_gpu(device)
            print(f"[HV Guidance] Enabled: hv_w={hv_w}, obj_w={obj_w}")
        except Exception as e:
            print(f"[HV Guidance] Warning: failed to create loss_fn: {e}")
            loss_fn = None
    
    print(f"\nData: {n_train} samples, {len(lambda_values)} preferences")
    print(f"Lambda range: [{lambda_values[0]:.2f}, {lambda_values[-1]:.2f}]")
    
    num_epochs = config.multi_pref_epochs
    lr = config.multi_pref_lr
    
    print(f"\nConfig: epochs={num_epochs}, lr={lr}, mode=trajectory, use_rk2={config.multi_pref_rollout_use_rk2}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    
    # Learning rate scheduler: cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=2, eta_min=lr * 0.01
    )
    print(f"[LR Scheduler] CosineAnnealingWarmRestarts: T_0=100, T_mult=2, eta_min={lr*0.01:.2e}")
    
    dataloader = create_multi_preference_dataloader(multi_pref_data, config, shuffle=True)
    
    lambda_sorted = sorted(lambda_values)
    lambda_min, lambda_max = lambda_sorted[0], lambda_sorted[-1]
    lambda_norm = {lc: (lc - lambda_min) / (lambda_max - lambda_min) if lambda_max > lambda_min else 0.0 
                   for lc in lambda_sorted}
    NPred_Va = multi_pref_data.get('NPred_Va', multi_pref_data.get('output_dim', 0) // 2)
    
    # ==================== Pre-stack y_train_by_pref for Vectorized Training ====================
    # Stack all preference solutions into a single tensor [K, N, D] for fast indexing
    K = len(lambda_sorted)
    y_stacked = torch.stack([y_train_by_pref[lc] for lc in lambda_sorted], dim=0)  # [K, N, D]
    lambda_norm_tensor = torch.tensor([lambda_norm[lc] for lc in lambda_sorted], 
                                       device=device, dtype=torch.float32)  # [K]
    print(f"[Vectorized] Pre-stacked y_train: {y_stacked.shape}, lambda_norm_tensor: {lambda_norm_tensor.shape}")
    
    losses = []
    start_time = time.process_time()
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss, num_batches = 0.0, 0
        
        for batch_x, batch_idx in dataloader:
            batch_x, batch_idx = batch_x.to(device, non_blocking=True), batch_idx.to(device, non_blocking=True) 
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
            loss = _train_trajectory_step(
                model, batch_x, batch_idx, y_train_by_pref, lambda_sorted, lambda_norm,
                NPred_Va, device, config, epoch, num_epochs, projector,
                loss_fn, lambda_min, lambda_max, y_stacked, lambda_norm_tensor
            )
            
            if loss is None: continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        losses.append(epoch_loss / max(num_batches, 1))
        
        # Update learning rate
        scheduler.step()
        
        if (epoch + 1) % config.p_epoch == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch {epoch+1}: Loss = {losses[-1]:.6f}, LR = {current_lr:.2e}')
        
        if (epoch + 1) % 100 == 0 and (epoch + 1) >= config.s_epoch:
            os.makedirs(config.model_save_dir, exist_ok=True)
            checkpoint_filename = _generate_model_filename(config, model_type, epoch=epoch+1, is_final=False)
            checkpoint_path = f'{config.model_save_dir}/{checkpoint_filename}'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'  Checkpoint saved: {checkpoint_filename}')
    
    time_train = time.process_time() - start_time
    print(f'\nCompleted in {time_train:.2f}s ({time_train/60:.2f}min)')
    
    os.makedirs(config.model_save_dir, exist_ok=True)
    final_filename = _generate_model_filename(config, model_type, epoch=None, is_final=True)
    final_path = f'{config.model_save_dir}/{final_filename}'
    torch.save(model.state_dict(), final_path, _use_new_zipfile_serialization=False)
    print(f'Saved: {final_filename}')
    
    return model, losses, time_train


def _train_trajectory_step(
                    model, batch_x, batch_idx, y_train_by_pref, lambda_sorted, lambda_norm,
                    NPred_Va, device, config, epoch, num_epochs,
                    projector, loss_fn=None, lambda_min=0.0, lambda_max=50.0,
                    y_stacked=None, lambda_norm_tensor=None
                ):
    """Training step for preference trajectory mode with optional HV guidance.
    
    Optimized: Uses vectorized operations instead of Python for loops.
    """
    B = batch_x.shape[0]
    K = len(lambda_sorted)
    
    if K < 2:
        return None
    
    # ==================== Vectorized Sampling ====================
    # Use pre-stacked tensor if available, otherwise fall back to dict indexing
    if y_stacked is not None and lambda_norm_tensor is not None:
        # y_stacked: [K, N, D], lambda_norm_tensor: [K]
        # Vectorized: sample random k in [0, K-2] for each sample in batch
        k_indices = torch.randint(0, K - 1, (B,), device=device)  # [B]
        
        # Get sample indices (already on device)
        sample_idx = batch_idx.long()  # [B]
        
        # Gather x_curr and x_next using advanced indexing
        # x_curr[i] = y_stacked[k_indices[i], sample_idx[i], :]
        x_curr_gt = y_stacked[k_indices, sample_idx, :]        # [B, D]
        x_next_gt = y_stacked[k_indices + 1, sample_idx, :]    # [B, D]
        
        # Get lambda values
        lambda_curr_norm = lambda_norm_tensor[k_indices].view(-1, 1)      # [B, 1]
        lambda_next_norm = lambda_norm_tensor[k_indices + 1].view(-1, 1)  # [B, 1]
        
        scene = batch_x  # [B, input_dim]
    else:
        # Fallback: original loop-based sampling (slower but works without pre-stacking)
        x_current_list, x_next_list, lambda_curr_list, lambda_next_list, scene_list = [], [], [], [], []
        
        for i in range(B):
            idx = batch_idx[i].item()
            solutions, lambdas = [], []
            for lc in lambda_sorted:
                if lc in y_train_by_pref:
                    solutions.append(y_train_by_pref[lc][idx])
                    lambdas.append(lc)
            
            if len(solutions) < 2: continue
            
            k = random.randint(0, len(solutions) - 2)
            x_current_list.append(solutions[k])
            x_next_list.append(solutions[k+1])
            lambda_curr_list.append(lambdas[k])
            lambda_next_list.append(lambdas[k+1])
            scene_list.append(batch_x[i])
        
        if not x_current_list: return None
        
        x_curr_gt = torch.stack(x_current_list)
        x_next_gt = torch.stack(x_next_list)
        scene = torch.stack(scene_list)
        
        lambda_curr_norm = torch.tensor([[lambda_norm[lc]] for lc in lambda_curr_list], device=device, dtype=torch.float32)
        lambda_next_norm = torch.tensor([[lambda_norm[lc]] for lc in lambda_next_list], device=device, dtype=torch.float32)
    
    dx = wrap_angle_difference(x_next_gt - x_curr_gt, NPred_Va)
    dlambda = lambda_next_norm - lambda_curr_norm + 1e-8
    v_target = dx / dlambda
    
    v_pred = model.predict_vec(scene, x_curr_gt, lambda_curr_norm, lambda_curr_norm)


    # [CBF-QP TRAIN] Optionally project the incremental update via CBF-QP (tube + gate)
    use_cbf = (projector is not None) and getattr(projector, "cfg", None) is not None and projector.cfg.enabled

    alpha = config.multi_pref_loss_alpha
    beta = config.multi_pref_loss_beta

    # [BRIDGE] projection magnitude penalty (encourage shorter bridges)
    loss_bridge = torch.tensor(0.0, device=device)

    # [TUBE] update tube eps schedule (call once per step; cheap)
    if use_cbf and hasattr(projector, "set_progress"):
        denom = max(int(num_epochs) - 1, 1)
        progress = float(epoch) / float(denom)
        projector.set_progress(progress)

    # Decide whether to apply CBF-QP this batch (honor apply_prob)
    use_cbf_batch = False
    A0 = b0 = None
    if use_cbf:
        ap = float(getattr(projector.cfg, "apply_prob", 1.0))
        if ap >= 1.0:
            use_cbf_batch = True
        else:
            use_cbf_batch = float(torch.rand(1, device=device)) <= ap

        if use_cbf_batch:
            # Build A,b once at x_curr (detached). Both Euler and RK2 can reuse this linearization.
            with torch.no_grad():
                A0, b0 = projector.build_Ab(x_curr_gt.detach(), scene.detach())

    # Use RK2 (Heun) method if enabled, otherwise use Euler method
    if config.multi_pref_rollout_use_rk2:
        # RK2: x_{n+1} = x_n + Δλ * 0.5*(v0 + v1)
        delta1_ref = dlambda * v_pred
        if use_cbf_batch:
            delta1_exec, _info1 = projector.maybe_project_delta_given_Ab(delta1_ref, A0, b0)
        else:
            delta1_exec = delta1_ref
        x_euler = x_curr_gt + delta1_exec

        # Step 2: predict v1 at (possibly safe) intermediate point
        v1 = model.predict_vec(scene, x_euler, lambda_next_norm, lambda_next_norm)
        delta2_ref = dlambda * 0.5 * (v_pred + v1)

        if use_cbf_batch:
            # Optional: rebuild A,b at x_euler (more accurate, slower)
            if bool(getattr(config, "multi_pref_cbf_rk2_rebuild_ab", False)):
                with torch.no_grad():
                    A1, b1 = projector.build_Ab(x_euler.detach(), scene.detach())
                delta2_exec, _info2 = projector.maybe_project_delta_given_Ab(delta2_ref, A1, b1)
            else:
                delta2_exec, _info2 = projector.maybe_project_delta_given_Ab(delta2_ref, A0, b0)

            # [BRIDGE] penalize the final-stage projection magnitude
            bridge_w = float(getattr(config, "multi_pref_bridge_weight", 0.0))
            if bridge_w > 0:
                loss_bridge = torch.mean((delta2_exec - delta2_ref) ** 2)
        else:
            delta2_exec = delta2_ref

        x_pred = x_curr_gt + delta2_exec
        v_used = delta2_exec / (dlambda + 1e-12)  # for loss
        distill = torch.mean((v_pred - v_used) ** 2) if (use_cbf_batch and projector.cfg.distill_weight > 0) else 0.0

    else:
        # Euler: x_{n+1} = x_n + Δλ * v
        delta_ref = dlambda * v_pred
        if use_cbf_batch:
            delta_exec, _info = projector.maybe_project_delta_given_Ab(delta_ref, A0, b0)
            bridge_w = float(getattr(config, "multi_pref_bridge_weight", 0.0))
            if bridge_w > 0:
                loss_bridge = torch.mean((delta_exec - delta_ref) ** 2)
        else:
            delta_exec = delta_ref

        x_pred = x_curr_gt + delta_exec
        v_used = delta_exec / (dlambda + 1e-12)
        distill = torch.mean((v_pred - v_used) ** 2) if (use_cbf_batch and projector.cfg.distill_weight > 0) else 0.0

    # [CBF-QP TRAIN] velocity loss uses the actually executed velocity (v_used)
    loss_v = torch.mean((v_used - v_target) ** 2)
    # Optional distillation regularizer (reduce projection trigger over time)
    if use_cbf_batch and projector is not None and projector.cfg.distill_weight > 0:
        loss_v = loss_v + projector.cfg.distill_weight * distill
    dx_pred = wrap_angle_difference(x_pred - x_next_gt, NPred_Va)
    
    loss_endpoint = torch.nn.functional.smooth_l1_loss(dx_pred, torch.zeros_like(dx_pred))
    # loss_endpoint = torch.mean(dx_pred ** 2)
    bridge_w = float(getattr(config, "multi_pref_bridge_weight", 0.0))
    
    # ==================== HV Guidance Loss ====================
    loss_hv = torch.tensor(0.0, device=device)
    loss_obj = torch.tensor(0.0, device=device)
    
    if loss_fn is not None:
        current_hv_w = _get_hv_weight(epoch, num_epochs, config)
        obj_w = float(getattr(config, "multi_pref_obj_weight", 0.0))
        
        if current_hv_w > 0 or obj_w > 0:
            # Denormalize lambda_next_norm to raw lambda for loss_fn
            lambda_next_raw = lambda_next_norm * (lambda_max - lambda_min) + lambda_min
            lambda_next_raw = lambda_next_raw.view(-1)  # [B]
            
            try:
                # Evaluate OPF objectives at predicted point
                obj_loss_val, obj_dict = loss_fn.forward(x_pred, scene, preference=lambda_next_raw, only_obj=True)
                
                if obj_w > 0:
                    loss_obj = obj_loss_val
                
                if current_hv_w > 0:
                    cost_t, carbon_t = _extract_cost_carbon_torch(obj_dict, device, x_pred.dtype)
                    if cost_t is not None and carbon_t is not None:
                        # Compute reference points (dominated point)
                        ref_margin = float(getattr(config, "multi_pref_hv_ref_margin", 0.05))
                        ref_cost = float(cost_t.max().detach()) * (1.0 + ref_margin)
                        ref_carbon = float(carbon_t.max().detach()) * (1.0 + ref_margin)
                        
                        tau = float(getattr(config, "multi_pref_hv_tau", 0.05))
                        power = float(getattr(config, "multi_pref_hv_power", 2.0))
                        
                        loss_hv = _psl_hv1_proxy_loss(
                            cost_t, carbon_t, lambda_next_raw, 
                            ref_cost, ref_carbon, tau=tau, power=power
                        )
            except Exception as e:
                # Silently ignore HV loss computation failures
                pass
    
    current_hv_w = _get_hv_weight(epoch, num_epochs, config) if loss_fn is not None else 0.0
    obj_w = float(getattr(config, "multi_pref_obj_weight", 0.0)) if loss_fn is not None else 0.0
    
    return alpha * loss_v + beta * loss_endpoint + bridge_w * loss_bridge + current_hv_w * loss_hv + obj_w * loss_obj


# ==================== Main Function ====================

def main(debug=False):
    """Main function for multi-preference supervised training."""
    from unified_eval import MultiPreferencePredictor, build_ctx_from_multi_preference, evaluate_unified

    
    config = get_multi_preference_config()
    
    print("=" * 60)
    print("DeepOPF-V: Multi-Preference Training (Trajectory Mode)")
    print("=" * 60)
    config.print_config()
    
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    
    model_type = config.model_type
    print(f"\nModel type: {model_type}")
    
    # Load data
    multi_pref_data, sys_data = load_multi_preference_dataset(config)
    
    # Compute BRANFT directly from sys_data.branch (branch from-to indices, 0-indexed)
    # BRANFT is used for branch constraint violation checking in evaluation
    BRANFT = torch.from_numpy(sys_data.branch[:, 0:2] - 1).long()
    
    input_dim = multi_pref_data['input_dim']
    output_dim = multi_pref_data['output_dim']  # Now in NGT format (non-ZIB)
    pref_dim = config.pref_dim
    
    # Data is now converted to NGT format in load_multi_preference_dataset
    # Format: [Va_noslack_nonZIB, Vm_nonZIB]
    # Vscale and Vbias from ngt_data should match output_dim
    NPred_Va = multi_pref_data['NPred_Va']
    NPred_Vm = multi_pref_data['NPred_Vm']
    
    # Verify dimensions match
    expected_output_dim = NPred_Va + NPred_Vm
    if output_dim != expected_output_dim:
        raise ValueError(f"output_dim mismatch: got {output_dim}, expected {expected_output_dim} "
                        f"(NPred_Va={NPred_Va} + NPred_Vm={NPred_Vm})")
    
    # Use Vscale and Vbias from ngt_data (dimensions already match NGT format)
    Vscale = multi_pref_data['Vscale']
    Vbias = multi_pref_data['Vbias']
    
    # Verify Vscale/Vbias dimensions
    if len(Vscale) != output_dim:
        raise ValueError(f"Vscale dimension mismatch: got {len(Vscale)}, expected {output_dim}")
    
    print(f"\nDimensions (NGT format): input={input_dim}, output={output_dim}, pref={pref_dim}")
    print(f"NPred_Va={NPred_Va}, NPred_Vm={NPred_Vm}, Vscale.shape={Vscale.shape}, Vbias.shape={Vbias.shape}")
    
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'flow_model'))
    from net_utiles import FM, VAE
    
    # Only support Flow models for trajectory training
    if model_type not in ['rectified', 'gaussian', 'conditional', 'interpolation']:
        raise ValueError(f"CBF-QP tube training only supports Flow models (rectified, etc.), got: {model_type}")
    
    # Create Flow model
    model = FM(network='preference_aware_mlp', input_dim=input_dim, output_dim=output_dim,
               hidden_dim=config.hidden_dim, num_layers=config.num_layers,
               time_step=config.time_step, output_norm=False, pred_type='velocity', pref_dim=pref_dim)
    
    # Load pretrained VAE as anchor generator
    vae_args = dict(output_dim=output_dim, hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers, latent_dim=config.latent_dim,
                    output_act=None, pred_type='node', use_cvae=True)
    pretrain_model = VAE(network='preference_aware_mlp', input_dim=input_dim, pref_dim=pref_dim, **vae_args)
    pretrain_model.load_state_dict(torch.load(PRETRAINED_VAE_MODEL_PATH, map_location=config.device))
    pretrain_model.to(config.device)
    pretrain_model.eval()
    print(f"  Loaded pretrained VAE: {PRETRAINED_VAE_MODEL_PATH}")
    
    model.to(config.device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}") 
            
    if not debug:
        model, _, _ = train_multi_preference(config, model, multi_pref_data, sys_data, config.device,
                                              model_type=model_type, pretrain_model=pretrain_model)
    else:
        print("\n[Debug Mode] Loading model...")
        # 使用文件末尾配置的测试模型路径
        if TEST_MODEL_PATH and os.path.exists(TEST_MODEL_PATH):
            model.load_state_dict(torch.load(TEST_MODEL_PATH, map_location=config.device, weights_only=True))
            model.eval()
            print(f"  Loaded: {TEST_MODEL_PATH}")
        else:
            # 如果配置的路径不存在，尝试使用默认路径（向后兼容）
            final_filename = _generate_model_filename(config, model_type, epoch=None, is_final=True)
            path = f'{config.model_save_dir}/{final_filename}'
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=config.device, weights_only=True))
                model.eval()
                print(f"  Loaded (default path): {final_filename}")
            else:
                old_path = f'{config.model_save_dir}/model_multi_pref_{model_type}_final.pth'
                if os.path.exists(old_path):
                    model.load_state_dict(torch.load(old_path, map_location=config.device, weights_only=True))
                    model.eval()
                    print(f"  Loaded (old format): {old_path}")
                else:
                    print(f"  Warning: Model file not found. Tried:")
                    if TEST_MODEL_PATH:
                        print(f"    - {TEST_MODEL_PATH} (configured)")
                    print(f"    - {path} (default)")
                    print(f"    - {old_path} (old format)")
    
    # Evaluation
    print("\n" + "=" * 80)
    print("Evaluation")
    print("=" * 80)
    
    test_lambdas = [0.0, 25.0, 50.0, 80.0, 90.0]
    results_all = {}
    
    flow_best_of_k = config.flow_best_of_k
    flow_selection_mode = config.flow_selection_mode
    
    # Create NGT loss function for Flow Best-of-K
    ngt_loss_fn = None
    if flow_best_of_k > 1:
        from deepopf_ngt_loss import DeepOPFNGTLoss
        try:
            ngt_loss_fn = DeepOPFNGTLoss(sys_data, config)
            ngt_loss_fn.cache_to_gpu(config.device)
            print(f"[Eval] Flow Best-of-K: K={flow_best_of_k}, mode={flow_selection_mode}")
        except Exception as e:
            print(f"[Warning] Best-of-K disabled: {e}")
            flow_best_of_k = 1
    
    # Post-processing configuration (controlled via config.post_process_method or config.use_cbf_qp_post)
    post_process_method = getattr(config, 'post_process_method', '')
    use_cbf_qp_post = getattr(config, 'use_cbf_qp_post', False)
    cbf_beta = getattr(config, 'cbf_beta', 1.0)
    if post_process_method == 'cbf_qp' or use_cbf_qp_post:
        print(f"[Eval] Post-Processing: method='{post_process_method}', cbf_beta={cbf_beta}")
    
    def eval_on_lambdas(lambdas):
        """Evaluate model on given lambda values."""
        res = {}
        for lc in lambdas:
            print(f"\n--- lambda_carbon = {lc:.2f} ---")
            ctx = build_ctx_from_multi_preference(config, sys_data, multi_pref_data, BRANFT, config.device, lambda_carbon=lc)
            predictor = MultiPreferencePredictor(
                model=model, multi_pref_data=multi_pref_data, lambda_carbon=lc, model_type=model_type,
                num_flow_steps=config.multi_pref_flow_steps, training_mode='trajectory',
                ngt_loss_fn=ngt_loss_fn, flow_n_samples=flow_best_of_k,
                flow_selection_mode=flow_selection_mode, pretrain_model=pretrain_model
            )
            res[lc] = evaluate_unified(
                ctx, predictor, apply_post_processing=True, verbose=True
            )
        return res
    
    # Evaluate on validation set
    print(f"\n{'=' * 40} VALIDATION SET {'=' * 40}")
    results_all['val'] = eval_on_lambdas(test_lambdas)
    
    # Evaluate on training set (to check overfitting)
    print(f"\n{'=' * 40} TRAINING SET {'=' * 40}")
    orig = {k: multi_pref_data.get(k) for k in ['x_val', 'n_val', 'y_val_by_pref']}
    multi_pref_data['x_val'] = multi_pref_data['x_train']
    multi_pref_data['n_val'] = multi_pref_data['n_train']
    multi_pref_data['y_val_by_pref'] = multi_pref_data['y_train_by_pref']
    results_all['train'] = eval_on_lambdas(test_lambdas)
    for k, v in orig.items():  # Restore
        if v is not None: multi_pref_data[k] = v
    
    print("\n" + "=" * 80)
    print("Evaluation Complete")
    print("=" * 80)
    
    return results_all


# ==================== Model Path Configuration ====================
# Configure model paths here for easy modification (before running main)

# Test model path (for debug mode): path to the model you want to evaluate
# Set to None or empty string to use default auto-generated path
# Format: model_multi_pref_{type}_traj_{cbf_tag}_final.pth
TEST_MODEL_PATH = "main_part/saved_models/model_multi_pref_rectified_traj_nocbf_final.pth"

# Pretrained VAE model path: used as anchor generator for flow models (preference_trajectory mode)
PRETRAINED_VAE_MODEL_PATH = "main_part/saved_models/model_multi_pref_vae_final.pth"


if __name__ == "__main__":
    debug = bool(int(os.environ.get('DEBUG', '1')))
    main(debug=debug)
