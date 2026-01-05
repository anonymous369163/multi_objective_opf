#!/usr/bin/env python
# coding: utf-8
"""
Multi-Preference Supervised Training for DeepOPF-V
Trains preference-conditioned models for multi-objective OPF.

Supports: simple, vae, rectified, diffusion

Author: Peng Yue
Date: December 2025

Usage:
    MODEL_TYPE=rectified python train_multi_preference.py
    MODEL_TYPE=vae LOAD_PRETRAINED_MODEL=1 python train_multi_preference.py
"""

import torch
import torch.nn as nn
import time
import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BaseConfig, _SCRIPT_DIR
from models import NetV
from data_loader import load_multi_preference_dataset, create_multi_preference_dataloader

# [CBF-QP TRAIN] Projection layer (training-time)
from cbf_qp_train_layer_tube import CBFQPTrainConfig, CBFQPProjectorNGT  # [TUBE]

# [STAGE2-HV] Optional differentiable OPF objective loss (cost/carbon) for continuous preference training
try:
    from main_part.deepopf_ngt_loss import DeepOPFNGTLoss
except Exception:
    try:
        from deepopf_ngt_loss import DeepOPFNGTLoss
    except Exception:
        DeepOPFNGTLoss = None


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
        self.multi_pref_epochs = int(os.environ.get('MULTI_PREF_EPOCHS', '1000'))
        self.multi_pref_lr = float(os.environ.get('MULTI_PREF_LR', '1e-4'))
        self.multi_pref_flow_steps = int(os.environ.get('MULTI_PREF_FLOW_STEPS', '10'))
        self.multi_pref_batch_size = int(os.environ.get('MULTI_PREF_BATCH_SIZE', '50'))
        
        # Validation split
        self.multi_pref_val_ratio = float(os.environ.get('MULTI_PREF_VAL_RATIO', '0.2'))
        self.multi_pref_random_seed = int(os.environ.get('MULTI_PREF_RANDOM_SEED', '42'))
        
        # ==================== Unified Model Architecture ====================
        # These parameters are shared across VAE, Flow, Diffusion models
        # Chosen to keep model sizes roughly consistent (~100-200K params)
        # Simple: ~95K, Flow: ~183K, Diffusion: ~141K, VAE: ~390K (has Encoder+Decoder)
        self.hidden_dim = int(os.environ.get('HIDDEN_DIM', '128'))
        self.num_layers = int(os.environ.get('NUM_LAYERS', '2'))
        self.latent_dim = int(os.environ.get('LATENT_DIM', '64'))  # For VAE
        self.time_step = 1000  # For Flow/Diffusion ODE solver
        
        # Simple model (NetV) uses a special structure
        self.ngt_hidden_units = 1
        self.ngt_khidden = np.array([64, 224], dtype=int)
        
        # Training mode and flow type
        self.multi_pref_flow_type = self.model_type
        self.multi_pref_training_mode = os.environ.get('MULTI_PREF_TRAINING_MODE', 'preference_trajectory')  # 'standard', 'preference_trajectory'
        
        # Loss weights
        self.multi_pref_loss_alpha = float(os.environ.get('MULTI_PREF_LOSS_ALPHA', '1.0'))
        self.multi_pref_loss_beta = float(os.environ.get('MULTI_PREF_LOSS_BETA', '1000.0'))
        
        # Multi-step rollout
        self.multi_pref_rollout_use_rk2 = os.environ.get('MULTI_PREF_ROLLOUT_USE_RK2', 'True').lower() == 'true'

        # ==================== [STAGE2] Continuous preference training (shooting) ====================
        # Enable a second stage that integrates multiple sub-steps between two discrete lambdas.
        # This makes the learned velocity field smoother w.r.t. *continuous* lambda.
        self.multi_pref_stage2_enabled = os.environ.get('MULTI_PREF_STAGE2_ENABLED', '1').lower() in ['1', 'true', 'yes']
        self.multi_pref_stage2_start_ratio = float(os.environ.get('MULTI_PREF_STAGE2_START_RATIO', '0.5'))  # fraction of epochs
        self.multi_pref_stage2_substeps = int(os.environ.get('MULTI_PREF_STAGE2_SUBSTEPS', str(self.multi_pref_flow_steps)))
        self.multi_pref_stage2_rebuild_ab_every = int(os.environ.get('MULTI_PREF_STAGE2_REBUILD_AB_EVERY', '1'))

        # Keep some coarse velocity supervision in stage-2 (scaled from alpha/beta)
        self.multi_pref_stage2_alpha_scale = float(os.environ.get('MULTI_PREF_STAGE2_ALPHA_SCALE', '0.2'))
        self.multi_pref_stage2_endpoint_beta_scale = float(os.environ.get('MULTI_PREF_STAGE2_BETA_SCALE', '1.0'))

        # Optional differentiable objective/HV guidance (requires DeepOPFNGTLoss)
        self.multi_pref_stage2_obj_weight = float(os.environ.get('MULTI_PREF_STAGE2_OBJ_WEIGHT', '0.0'))
        self.multi_pref_stage2_hv_weight = float(os.environ.get('MULTI_PREF_STAGE2_HV_WEIGHT', '0.0'))
        self.multi_pref_stage2_hv_tau = float(os.environ.get('MULTI_PREF_STAGE2_HV_TAU', '0.05'))
        self.multi_pref_stage2_hv_power = float(os.environ.get('MULTI_PREF_STAGE2_HV_POWER', '2.0'))
        self.multi_pref_stage2_hv_ref_margin = float(os.environ.get('MULTI_PREF_STAGE2_HV_REF_MARGIN', '0.05'))

        # If > 0, use fixed HV reference point; otherwise use batch-adaptive detached maxima
        self.multi_pref_stage2_hv_ref_cost = float(os.environ.get('MULTI_PREF_STAGE2_HV_REF_COST', '-1.0'))
        self.multi_pref_stage2_hv_ref_carbon = float(os.environ.get('MULTI_PREF_STAGE2_HV_REF_CARBON', '-1.0'))

        # ==================== [CBF-QP TRAIN] Training-time safety projection ====================
        # Enable by setting env: MULTI_PREF_USE_CBF_QP_TRAIN=1
        # You can tune these without changing code.

        # 是否启用训练时的CBF-QP安全投影 (默认关闭: '0', 启用: '1')
        self.multi_pref_use_cbf_qp_train = os.environ.get('MULTI_PREF_USE_CBF_QP_TRAIN', '1').lower() in ['1', 'true', 'yes']
        # CBF强度参数: beta=1表示"一步回到边界内", beta<1更保守投影更小 (默认0.5, 建议范围0.3-0.7)
        self.multi_pref_cbf_beta = float(os.environ.get('MULTI_PREF_CBF_BETA', '0.5'))
        # 每个batch应用投影的概率: 1.0=总是投影(最安全但慢), <1.0=间歇投影(更快) (默认1.0, 建议0.8-1.0)
        self.multi_pref_cbf_apply_prob = float(os.environ.get('MULTI_PREF_CBF_APPLY_PROB', '1.0'))

        # Trust region (信赖域): 限制投影时的最大变化幅度, 防止投影过大导致不稳定
        # 电压相角(Va)的信赖域半径, 单位: 弧度 (默认0.10, 约5.7度, 建议范围0.08-0.15)
        self.multi_pref_cbf_trust_va = float(os.environ.get('MULTI_PREF_CBF_TRUST_VA', '0.10'))
        # 电压幅值(Vm)的信赖域半径, 单位: p.u. (默认0.01, IEEE118的Vm范围仅0.04, 建议0.008-0.015)
        self.multi_pref_cbf_trust_vm = float(os.environ.get('MULTI_PREF_CBF_TRUST_VM', '0.01'))

        # Constraint selection (约束选择): 只选择"接近边界"或"已违反"的约束, 降低QP求解复杂度
        # Vm约束选择阈值: 距离边界 < eps_vm 的约束会被包含 (默认0.01 p.u., IEEE118的Vm范围窄需要更小阈值)
        self.multi_pref_cbf_eps_vm = float(os.environ.get('MULTI_PREF_CBF_EPS_VM', '0.01'))
        # 发电机有功/无功(Pg,Qg)约束选择阈值 (默认0.02 p.u.)
        self.multi_pref_cbf_eps_pqg = float(os.environ.get('MULTI_PREF_CBF_EPS_PQG', '0.02'))
        # 支路功率流约束选择阈值 (默认0.02 p.u.)
        self.multi_pref_cbf_eps_branch = float(os.environ.get('MULTI_PREF_CBF_EPS_BRANCH', '0.02'))
        # 每个样本最多保留的Vm约束数量, 保持QP规模可控 (默认64, 建议32-128)
        self.multi_pref_cbf_k_vm = int(os.environ.get('MULTI_PREF_CBF_K_VM', '64'))
        # 每个样本最多保留的Pg/Qg约束数量 (默认64, 建议32-128)
        self.multi_pref_cbf_k_pqg = int(os.environ.get('MULTI_PREF_CBF_K_PQG', '64'))
        # 每个样本最多保留的支路约束数量 (默认32, 建议16-64)
        self.multi_pref_cbf_k_branch = int(os.environ.get('MULTI_PREF_CBF_K_BRANCH', '32'))

        # Solver knobs (QP求解器参数)
        # QP求解器的最大迭代次数 (默认6, 建议4-10, 越大越精确但越慢)
        self.multi_pref_cbf_max_iters = int(os.environ.get('MULTI_PREF_CBF_MAX_ITERS', '6'))
        # 是否在梯度计算时detach活跃集, 避免QP求解器状态影响梯度 (默认启用: '1')
        self.multi_pref_cbf_detach_active_set = os.environ.get('MULTI_PREF_CBF_DETACH_ACTIVE_SET', '1').lower() in ['1', 'true', 'yes']
        # 内点法/罚函数法的惩罚系数, 通常不需要调整 (默认1e7)
        self.multi_pref_cbf_penalty_rho = float(os.environ.get('MULTI_PREF_CBF_PENALTY_RHO', '1e7'))

        # Optional: distillation (蒸馏正则化)
        # 蒸馏损失权重: 鼓励模型预测速度v_pred接近投影后速度v_used, 减少推理时触发投影 (默认0.1, 建议0.1-0.3)
        self.multi_pref_cbf_distill_weight = float(os.environ.get('MULTI_PREF_CBF_DISTILL_WEIGHT', '0.1'))
        

        # ==================== [TUBE] Soft safety tube (软约束管, bridge-friendly) ====================
        # 训练时放宽约束: A*delta <= b + eps_tube, 随训练进行逐渐收紧 (eps_tube从start->end)
        # 有助于训练稳定性和收敛, 特别适合训练初期
        
        # Vm约束的tube松弛量: 训练开始时的值 (默认0.005, IEEE118的Vm范围窄故设置较小)
        self.multi_pref_tube_eps_vm_start = float(os.environ.get('MULTI_PREF_TUBE_EPS_VM_START', '0.005'))
        # Vm约束的tube松弛量: 训练结束时的值 (默认0.00, 表示最终严格满足约束)
        self.multi_pref_tube_eps_vm_end   = float(os.environ.get('MULTI_PREF_TUBE_EPS_VM_END',   '0.00'))
        # Pg/Qg约束的tube松弛量: 训练开始时的值 (默认0.01, 功率约束可适当放宽)
        self.multi_pref_tube_eps_pqg_start = float(os.environ.get('MULTI_PREF_TUBE_EPS_PQG_START', '0.01'))
        # Pg/Qg约束的tube松弛量: 训练结束时的值
        self.multi_pref_tube_eps_pqg_end   = float(os.environ.get('MULTI_PREF_TUBE_EPS_PQG_END',   '0.00'))
        # 支路约束的tube松弛量: 训练开始时的值 (默认0.01)
        self.multi_pref_tube_eps_branch_start = float(os.environ.get('MULTI_PREF_TUBE_EPS_BRANCH_START', '0.01'))
        # 支路约束的tube松弛量: 训练结束时的值
        self.multi_pref_tube_eps_branch_end   = float(os.environ.get('MULTI_PREF_TUBE_EPS_BRANCH_END',   '0.00'))
        # tube松弛量的调度方式: 'linear'(线性), 'cosine'(余弦更平滑), 'exp'(指数) (默认'cosine')
        self.multi_pref_tube_schedule = os.environ.get('MULTI_PREF_TUBE_SCHEDULE', 'cosine')
        # 当schedule='exp'时使用的指数衰减系数 (默认5.0, 建议3.0-10.0)
        self.multi_pref_tube_exp_k = float(os.environ.get('MULTI_PREF_TUBE_EXP_K', '5.0'))

        # ==================== [GATE] Skip QP solve if delta already safe (提前跳过) ====================
        # 如果增量delta已经满足约束, 跳过QP求解以节省计算 (默认启用: '1')
        self.multi_pref_cbf_gate_before_solve = os.environ.get('MULTI_PREF_CBF_GATE', '1').lower() in ['1','true','yes']
        # Gate检查的数值容差, 通常不需要调整 (默认1e-9)
        self.multi_pref_cbf_gate_eps = float(os.environ.get('MULTI_PREF_CBF_GATE_EPS', '1e-9'))

        # Optional: for RK2 (RK2方法相关)
        # 在RK2第二步时, 是否在x_euler处重新构建约束矩阵A,b (更精确但更慢, 默认禁用: '0')
        self.multi_pref_cbf_rk2_rebuild_ab = os.environ.get('MULTI_PREF_CBF_RK2_REBUILD_AB', '0').lower() in ['1','true','yes']

        # ==================== [BRIDGE] Penalize projection magnitude (投影幅度惩罚) ====================
        # 惩罚投影幅度||delta_exec - delta_ref||², 鼓励模型输出更接近可直接使用的值, 减少对投影的依赖
        # 投影幅度惩罚权重 (默认0.0禁用, 建议0.1-1.0, 过大可能影响模型学习)
        self.multi_pref_bridge_weight = float(os.environ.get('MULTI_PREF_BRIDGE_WEIGHT', '0.0'))
        # Preference conditioning
        self.pref_dim = 1
        
        # ==================== VAE Evaluation ====================
        self.vae_best_of_k = int(os.environ.get('VAE_BEST_OF_K', '32'))
        self.vae_use_mean = os.environ.get('VAE_USE_MEAN', '0').lower() in ('1', 'true', 'yes')
        self.vae_selection_mode = os.environ.get('VAE_SELECTION_MODE', 'constraint')
        self.vae_use_preference_aware = True
        self.vae_beta = 1.0
        
        # ==================== Flow Best-of-K Evaluation ====================
        self.flow_best_of_k = int(os.environ.get('FLOW_BEST_OF_K', '32'))  # K for Flow Best-of-K (1=disabled)
        self.flow_selection_mode = os.environ.get('FLOW_SELECTION_MODE', 'constraint')
        
        # ==================== Training Control ====================
        self.weight_decay = 1e-6
        self.p_epoch = 10   # Print every p_epoch epochs
        self.s_epoch = 800  # Start saving checkpoints after s_epoch
        
    def print_config(self):
        """Print configuration summary."""
        super().print_config()
        print(f"\n[Multi-Preference Training Config]")
        print(f"  Epochs: {self.multi_pref_epochs}")
        print(f"  Learning rate: {self.multi_pref_lr}")
        print(f"  Batch size: {self.multi_pref_batch_size}")
        print(f"  Training mode: {self.multi_pref_training_mode}")
        print(f"\n[Unified Model Architecture]")
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Num layers: {self.num_layers}")
        print(f"  Latent dim (VAE): {self.latent_dim}")
        print(f"  Simple khidden: {self.ngt_khidden}")
        print(f"\n[VAE Evaluation]")
        print(f"  Best-of-K: {self.vae_best_of_k} (use_mean={self.vae_use_mean})")
        print(f"\n[Flow Best-of-K Evaluation]")
        print(f"  Best-of-K: {self.flow_best_of_k} (mode={self.flow_selection_mode})")


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


# ==================== [STAGE2-HV] Helper: differentiable HV proxy loss ====================

def _extract_cost_carbon_torch(loss_dict, device, dtype):
    """Try to extract differentiable per-sample cost/carbon tensors from DeepOPFNGTLoss output."""
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
    """A lightweight 2-objective PSL-HV1-style proxy (maximize HV => minimize negative HV proxy).

    - Minimization objectives: cost, carbon
    - Reference point r should be a *dominated* (worse) point: r_i >= f_i.
    - Direction weights w are derived from lambda_carbon (lam_raw) and normalized.

    NOTE:
      This proxy *requires* cost/carbon to be differentiable torch tensors.
      If your DeepOPFNGTLoss currently returns detached numpy arrays in loss_dict,
      add two keys without detach:
        loss_dict['cost_per_sample_torch'] = cost_per_sample
        loss_dict['carbon_per_sample_torch'] = carbon_per_sample
    """
    if lam_raw.dim() == 1:
        lam = lam_raw.view(-1, 1)
    else:
        lam = lam_raw

    lam = torch.clamp(lam, min=0.0)

    # 2D direction weights, normalized (L1)
    w = torch.cat([torch.ones_like(lam), lam + 1e-6], dim=1)
    w = w / (w.sum(dim=1, keepdim=True) + 1e-12)  # [B,2]

    f = torch.stack([cost.view(-1), carbon.view(-1)], dim=1)  # [B,2]
    r = torch.tensor([ref_cost, ref_carbon], device=f.device, dtype=f.dtype).view(1, 2)

    # ray distance-like quantity
    t = (r - f) / (w + 1e-12)  # [B,2]
    rho = _softmin(t, tau=tau, dim=1)  # [B]
    rho = torch.clamp(rho, min=0.0)
    hv_proxy = rho ** float(power)
    return -hv_proxy.mean()

def rk2_step(model, scene, x_current, lambda_current, lambda_next, NPred_Va):
    """RK2 (Heun) integration step for preference trajectory."""
    dlambda = lambda_next - lambda_current
    v0 = model.predict_vec(scene, x_current, lambda_current, lambda_current)
    x_euler = x_current + dlambda * v0
    v1 = model.predict_vec(scene, x_euler, lambda_next, lambda_next)
    return x_current + dlambda * 0.5 * (v0 + v1)


# ==================== Training Functions ====================

def train_multi_preference(config, model, multi_pref_data, sys_data, device,
                           model_type='simple', pretrain_model=None):
    """Train preference-conditioned model for multi-objective OPF."""
    
    print('=' * 60)
    print(f'Training Multi-Preference Model - Type: {model_type}')
    print('=' * 60)
    
    # Note: x_train is loaded through dataloader, which uses multi_pref_data['x_train'] 
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
    
    print(f"\nData: {n_train} samples, {len(lambda_values)} preferences")
    print(f"Lambda range: [{lambda_values[0]:.2f}, {lambda_values[-1]:.2f}]")
    
    num_epochs = config.multi_pref_epochs
    lr = config.multi_pref_lr
    lc_max = max(lambda_values) if max(lambda_values) > 0 else 1.0
    vae_beta = config.vae_beta
    training_mode = config.multi_pref_training_mode
    
    print(f"\nConfig: epochs={num_epochs}, lr={lr}, mode={training_mode}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    dataloader = create_multi_preference_dataloader(multi_pref_data, config, shuffle=True)
    criterion = nn.MSELoss()
    
    lambda_sorted = sorted(lambda_values)
    lambda_min, lambda_max = lambda_sorted[0], lambda_sorted[-1]
    lambda_norm = {lc: (lc - lambda_min) / (lambda_max - lambda_min) if lambda_max > lambda_min else 0.0 
                   for lc in lambda_sorted}
    NPred_Va = multi_pref_data.get('NPred_Va', multi_pref_data.get('output_dim', 0) // 2)
    
    losses = []
    start_time = time.process_time()
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss, num_batches = 0.0, 0
        
        for batch_x, batch_idx in dataloader:
            batch_x, batch_idx = batch_x.to(device), batch_idx.to(device) 
            optimizer.zero_grad()
            
            if training_mode == 'preference_trajectory' and model_type == 'rectified':
                loss = _train_trajectory_step(
                    model, batch_x, batch_idx, y_train_by_pref, lambda_sorted, lambda_norm,
                    NPred_Va, device, config, epoch, num_epochs,
                    projector,
                    loss_fn
                )
            else:
                loss = _train_standard_step(
                    model, batch_x, batch_idx, y_train_by_pref, lambda_values, lc_max,
                    model_type, pretrain_model, criterion, vae_beta, device, config
                )
            
            if loss is None: continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        losses.append(epoch_loss / max(num_batches, 1))
        
        if (epoch + 1) % config.p_epoch == 0:
            print(f'Epoch {epoch+1}: Loss = {losses[-1]:.6f}')
        
        if (epoch + 1) % 100 == 0 and (epoch + 1) >= config.s_epoch:
            os.makedirs(config.model_save_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{config.model_save_dir}/model_multi_pref_{model_type}_E{epoch+1}.pth')
    
    time_train = time.process_time() - start_time
    print(f'\nCompleted in {time_train:.2f}s ({time_train/60:.2f}min)')
    
    os.makedirs(config.model_save_dir, exist_ok=True)
    final_path = f'{config.model_save_dir}/model_multi_pref_{model_type}_final.pth'
    torch.save(model.state_dict(), final_path, _use_new_zipfile_serialization=False)
    print(f'Saved: {final_path}')
    
    return model, losses, time_train


def _train_trajectory_step(
                    model, batch_x, batch_idx, y_train_by_pref, lambda_sorted, lambda_norm,
                    NPred_Va, device, config, epoch, num_epochs,
                    projector,
                    loss_fn=None,
                ):
    """Training step for preference trajectory mode.

    Stage-1 (default): one-step (Euler/RK2) using discrete neighbor supervision.
    Stage-2 (optional): multi-step "shooting" inside the interval to learn a smoother *continuous* velocity field,
                        optionally guided by differentiable OPF objective / HV proxy losses.
    """
    B = batch_x.shape[0]

    x_current_list, x_next_list = [], []
    lambda_curr_list, lambda_next_list = [], []
    scene_list = []

    # -------------------- build supervised pairs (discrete neighbors) --------------------
    for i in range(B):
        idx = batch_idx[i].item()
        solutions, lambdas = [], []
        for lc in lambda_sorted:
            if lc in y_train_by_pref:
                solutions.append(y_train_by_pref[lc][idx])
                lambdas.append(lc)

        if len(solutions) < 2:
            continue

        k = random.randint(0, len(solutions) - 2)
        x_current_list.append(solutions[k])
        x_next_list.append(solutions[k + 1])
        lambda_curr_list.append(lambdas[k])
        lambda_next_list.append(lambdas[k + 1])
        scene_list.append(batch_x[i])

    if not x_current_list:
        return None

    x_curr_gt = torch.stack(x_current_list)   # [B', D]
    x_next_gt = torch.stack(x_next_list)      # [B', D]
    scene = torch.stack(scene_list)           # [B', input_dim]  (this is PQd in NGT pipeline)

    # normalized lambda used by the flow model
    lambda_curr_norm = torch.tensor([[lambda_norm[lc]] for lc in lambda_curr_list], device=device, dtype=torch.float32)
    lambda_next_norm = torch.tensor([[lambda_norm[lc]] for lc in lambda_next_list], device=device, dtype=torch.float32)

    # raw lambda (lambda_carbon) used by DeepOPFNGTLoss / preference direction
    lambda_curr_raw = torch.tensor([[float(lc) ] for lc in lambda_curr_list], device=device, dtype=torch.float32)
    lambda_next_raw = torch.tensor([[float(lc) ] for lc in lambda_next_list], device=device, dtype=torch.float32)

    # coarse discrete velocity supervision
    dx = wrap_angle_difference(x_next_gt - x_curr_gt, NPred_Va)
    dlambda = lambda_next_norm - lambda_curr_norm + 1e-8
    v_target = dx / dlambda

    # predicted velocity at the left endpoint
    v_pred0 = model.predict_vec(scene, x_curr_gt, lambda_curr_norm, lambda_curr_norm)

    # -------------------- CBF-QP projection (tube + gate) --------------------
    use_cbf = (projector is not None) and getattr(projector, "cfg", None) is not None and projector.cfg.enabled

    alpha = float(getattr(config, "multi_pref_loss_alpha", 1.0))
    beta = float(getattr(config, "multi_pref_loss_beta", 1000.0))

    # [BRIDGE] projection magnitude penalty (encourage shorter bridges)
    bridge_w = float(getattr(config, "multi_pref_bridge_weight", 0.0))
    loss_bridge = torch.tensor(0.0, device=device)

    # [TUBE] update tube eps schedule (call once per step; cheap)
    if use_cbf and hasattr(projector, "set_progress"):
        denom = max(int(num_epochs) - 1, 1)
        progress = float(epoch) / float(denom)
        projector.set_progress(progress)

    # Decide whether to apply CBF-QP this batch (honor apply_prob)
    use_cbf_batch = False
    if use_cbf:
        ap = float(getattr(projector.cfg, "apply_prob", 1.0))
        if ap >= 1.0:
            use_cbf_batch = True
        else:
            use_cbf_batch = float(torch.rand(1, device=device)) <= ap

    # -------------------- [STAGE2-HV] multi-step shooting inside the interval --------------------
    stage2_enabled = bool(getattr(config, "multi_pref_stage2_enabled", False))
    start_ratio = float(getattr(config, "multi_pref_stage2_start_ratio", 0.5))
    stage2_start_epoch = int(start_ratio * float(num_epochs))
    substeps = int(getattr(config, "multi_pref_stage2_substeps", getattr(config, "multi_pref_flow_steps", 10)))
    rebuild_every = max(int(getattr(config, "multi_pref_stage2_rebuild_ab_every", 1)), 1)

    obj_w = float(getattr(config, "multi_pref_stage2_obj_weight", 0.0))
    hv_w = float(getattr(config, "multi_pref_stage2_hv_weight", 0.0))
    hv_tau = float(getattr(config, "multi_pref_stage2_hv_tau", 0.05))
    hv_power = float(getattr(config, "multi_pref_stage2_hv_power", 2.0))
    hv_ref_margin = float(getattr(config, "multi_pref_stage2_hv_ref_margin", 0.05))

    # stage2 keeps some supervision but shifts focus to within-interval consistency
    alpha2 = alpha * float(getattr(config, "multi_pref_stage2_alpha_scale", 0.2))
    beta2 = beta * float(getattr(config, "multi_pref_stage2_endpoint_beta_scale", 1.0))

    do_stage2 = stage2_enabled and (epoch >= stage2_start_epoch) and (substeps >= 2)

    if do_stage2:
        # integrate in normalized-lambda coordinate (what the model is trained on)
        dt_norm = (lambda_next_norm - lambda_curr_norm) / float(substeps)   # [B', 1]
        dt_raw  = (lambda_next_raw  - lambda_curr_raw ) / float(substeps)   # [B', 1]

        # evaluate guidance losses on a small set of internal points (keep cost reasonable)
        probe_steps = set([substeps // 2, substeps - 1])

        x = x_curr_gt
        A = b = None

        loss_obj = torch.tensor(0.0, device=device)
        loss_hv = torch.tensor(0.0, device=device)
        obj_cnt = 0
        hv_cnt = 0

        # optional distillation (reduce projection usage at inference)
        loss_distill = torch.tensor(0.0, device=device)

        for s in range(substeps):
            lam_s_norm = lambda_curr_norm + dt_norm * float(s)  # [B', 1]
            v_s = model.predict_vec(scene, x, lam_s_norm, lam_s_norm)
            delta_ref = dt_norm * v_s

            if use_cbf_batch:
                # rebuild linearization periodically to keep projection valid along the trajectory
                if (A is None) or (s % rebuild_every == 0):
                    with torch.no_grad():
                        A, b = projector.build_Ab(x.detach(), scene.detach())
                delta_exec, _info = projector.maybe_project_delta_given_Ab(delta_ref, A, b)

                if bridge_w > 0:
                    loss_bridge = loss_bridge + torch.mean((delta_exec - delta_ref) ** 2)

                if projector is not None and projector.cfg.distill_weight > 0:
                    v_used = delta_exec / (dt_norm + 1e-12)
                    loss_distill = loss_distill + torch.mean((v_s - v_used) ** 2)

            else:
                delta_exec = delta_ref

            x = x + delta_exec

            # ---- guidance losses at selected internal points ----
            if (loss_fn is not None) and ((obj_w > 0) or (hv_w > 0)) and (s in probe_steps):
                lam_s_raw = lambda_curr_raw + dt_raw * float(s + 1)  # preference at the new point
                try:
                    obj_loss_s, obj_dict = loss_fn.forward(x, scene, preference=lam_s_raw, only_obj=True)
                except Exception:
                    obj_loss_s, obj_dict = loss_fn(x, scene, preference=lam_s_raw, only_obj=True)

                if obj_w > 0:
                    loss_obj = loss_obj + obj_loss_s
                    obj_cnt += 1

                if hv_w > 0:
                    cost_t, carbon_t = _extract_cost_carbon_torch(obj_dict, device=device, dtype=x.dtype)
                    # HV proxy needs differentiable per-sample objectives
                    if (cost_t is not None) and (carbon_t is not None) and cost_t.requires_grad and carbon_t.requires_grad:
                        # reference point: either fixed (if >0) or batch-adaptive (detached)
                        ref_cost = float(getattr(config, "multi_pref_stage2_hv_ref_cost", -1.0))
                        ref_carbon = float(getattr(config, "multi_pref_stage2_hv_ref_carbon", -1.0))
                        if ref_cost <= 0:
                            ref_cost_t = cost_t.detach().max() * (1.0 + hv_ref_margin)
                            ref_cost = float(ref_cost_t.item())
                        if ref_carbon <= 0:
                            ref_carbon_t = carbon_t.detach().max() * (1.0 + hv_ref_margin)
                            ref_carbon = float(ref_carbon_t.item())

                        loss_hv = loss_hv + _psl_hv1_proxy_loss(
                            cost_t, carbon_t, lam_s_raw,
                            ref_cost=ref_cost, ref_carbon=ref_carbon,
                            tau=hv_tau, power=hv_power
                        )
                        hv_cnt += 1

        x_pred = x

        # endpoint supervision (still uses discrete neighbor)
        dx_pred = wrap_angle_difference(x_pred - x_next_gt, NPred_Va)
        loss_endpoint = torch.nn.functional.smooth_l1_loss(dx_pred, torch.zeros_like(dx_pred))

        # average velocity regularizer (keeps stage2 aligned with the coarse discrete velocity)
        dx_total = wrap_angle_difference(x_pred - x_curr_gt, NPred_Va)
        v_avg = dx_total / (dlambda + 1e-12)
        loss_v = torch.mean((v_avg - v_target) ** 2)

        # normalize accumulated terms
        if bridge_w > 0:
            loss_bridge = loss_bridge / float(substeps)
        if use_cbf_batch and projector is not None and projector.cfg.distill_weight > 0:
            loss_distill = loss_distill / float(substeps)
        else:
            loss_distill = torch.tensor(0.0, device=device)

        if obj_cnt > 0:
            loss_obj = loss_obj / float(obj_cnt)
        if hv_cnt > 0:
            loss_hv = loss_hv / float(hv_cnt)

        distill_w = float(getattr(projector.cfg, "distill_weight", 0.0)) if (use_cbf_batch and projector is not None) else 0.0

        return (
            alpha2 * loss_v
            + beta2 * loss_endpoint
            + bridge_w * loss_bridge
            + obj_w * loss_obj
            + hv_w * loss_hv
            + distill_w * loss_distill
        )

    # -------------------- Stage-1: original one-step (Euler/RK2) --------------------
    # Build A,b once at x_curr (detached). Both Euler and RK2 can reuse this linearization.
    A0 = b0 = None
    if use_cbf_batch:
        with torch.no_grad():
            A0, b0 = projector.build_Ab(x_curr_gt.detach(), scene.detach())

    # Use RK2 (Heun) method if enabled, otherwise use Euler method
    if bool(getattr(config, "multi_pref_rollout_use_rk2", True)):
        delta1_ref = dlambda * v_pred0
        if use_cbf_batch:
            delta1_exec, _info1 = projector.maybe_project_delta_given_Ab(delta1_ref, A0, b0)
        else:
            delta1_exec = delta1_ref
        x_euler = x_curr_gt + delta1_exec

        v1 = model.predict_vec(scene, x_euler, lambda_next_norm, lambda_next_norm)
        delta2_ref = dlambda * 0.5 * (v_pred0 + v1)

        if use_cbf_batch:
            if bool(getattr(config, "multi_pref_cbf_rk2_rebuild_ab", False)):
                with torch.no_grad():
                    A1, b1 = projector.build_Ab(x_euler.detach(), scene.detach())
                delta2_exec, _info2 = projector.maybe_project_delta_given_Ab(delta2_ref, A1, b1)
            else:
                delta2_exec, _info2 = projector.maybe_project_delta_given_Ab(delta2_ref, A0, b0)

            if bridge_w > 0:
                loss_bridge = torch.mean((delta2_exec - delta2_ref) ** 2)
        else:
            delta2_exec = delta2_ref

        x_pred = x_curr_gt + delta2_exec
        v_used = delta2_exec / (dlambda + 1e-12)

        distill = torch.mean((v_pred0 - v_used) ** 2) if (use_cbf_batch and projector.cfg.distill_weight > 0) else 0.0

    else:
        delta_ref = dlambda * v_pred0
        if use_cbf_batch:
            delta_exec, _info = projector.maybe_project_delta_given_Ab(delta_ref, A0, b0)
            if bridge_w > 0:
                loss_bridge = torch.mean((delta_exec - delta_ref) ** 2)
        else:
            delta_exec = delta_ref

        x_pred = x_curr_gt + delta_exec
        v_used = delta_exec / (dlambda + 1e-12)
        distill = torch.mean((v_pred0 - v_used) ** 2) if (use_cbf_batch and projector.cfg.distill_weight > 0) else 0.0

    # velocity loss uses the actually executed velocity (v_used)
    loss_v = torch.mean((v_used - v_target) ** 2)
    # Optional distillation regularizer
    if use_cbf_batch and projector is not None and projector.cfg.distill_weight > 0:
        loss_v = loss_v + projector.cfg.distill_weight * distill

    dx_pred = wrap_angle_difference(x_pred - x_next_gt, NPred_Va)
    loss_endpoint = torch.nn.functional.smooth_l1_loss(dx_pred, torch.zeros_like(dx_pred))

    return alpha * loss_v + beta * loss_endpoint + bridge_w * loss_bridge

def _train_standard_step(model, batch_x, batch_idx, y_train_by_pref, lambda_values, lc_max,
                         model_type, pretrain_model, criterion, vae_beta, device, config):
    """Training step for standard mode."""
    B = batch_x.shape[0]
    
    lc_batch = [random.choice(lambda_values) for _ in range(B)]
    batch_y = torch.stack([y_train_by_pref[lc][batch_idx[i]] for i, lc in enumerate(lc_batch)])
    pref = torch.tensor([[lc / lc_max] for lc in lc_batch], device=device, dtype=torch.float32)
    
    if model_type == 'simple':
        x_with_pref = torch.cat([batch_x, pref], dim=1)
        return criterion(model(x_with_pref), batch_y)
        
    elif model_type == 'vae':
        use_pref_aware = hasattr(model, 'pref_dim') and model.pref_dim > 0
        if use_pref_aware:
            y_pred, mean, logvar = model.encoder_decode(batch_x, batch_y, pref=pref)
        else:
            y_pred, mean, logvar = model.encoder_decode(torch.cat([batch_x, pref], dim=1), batch_y)
        return model.loss(y_pred, batch_y, mean, logvar, beta=vae_beta)
        
    elif model_type in ['rectified', 'gaussian', 'conditional', 'interpolation']:
        t_batch = torch.rand([B, 1], device=device)
        if pretrain_model:
            with torch.no_grad():
                z_batch = pretrain_model(batch_x, use_mean=True, pref=pref) if hasattr(pretrain_model, 'pref_dim') else pretrain_model(torch.cat([batch_x, pref], dim=1), use_mean=True)
        else:
            z_batch = torch.randn_like(batch_y)
        
        flow_type = config.multi_pref_flow_type
        yt, vec_target = model.flow_forward(batch_y, t_batch, z_batch, flow_type)
        vec_pred = model.predict_vec(batch_x, yt, t_batch, pref)
        return model.loss(batch_y, z_batch, vec_pred, vec_target, flow_type)
        
    elif model_type == 'diffusion':
        t_batch = torch.rand([B, 1], device=device)
        noise = torch.randn_like(batch_y)
        x_with_pref = torch.cat([batch_x, pref], dim=1)
        if pretrain_model:
            with torch.no_grad():
                vae_anchor = pretrain_model(x_with_pref, use_mean=True)
            noise_pred = model.predict_noise_with_anchor(x_with_pref, batch_y, t_batch, noise, vae_anchor)
        else:
            noise_pred = model.predict_noise(x_with_pref, batch_y, t_batch, noise)
        return model.loss(noise_pred, noise)
    
    return None


# ==================== Main Function ====================

def main(debug=False):
    """Main function for multi-preference supervised training."""
    from unified_eval import MultiPreferencePredictor, build_ctx_from_multi_preference, evaluate_unified

    
    config = get_multi_preference_config()
    
    print("=" * 60)
    print("DeepOPF-V: Multi-Preference Training")
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
    from net_utiles import FM, VAE, DM
    
    model, pretrain_model = None, None
    
    if model_type == 'simple':
        model = NetV(input_dim + pref_dim, output_dim, config.ngt_hidden_units, config.ngt_khidden, Vscale, Vbias)
        
    elif model_type == 'vae':
        vae_args = dict(output_dim=output_dim, hidden_dim=config.hidden_dim,
                        num_layers=config.num_layers, latent_dim=config.latent_dim,
                        output_act=None, pred_type='node', use_cvae=True)
        if config.vae_use_preference_aware:
            model = VAE(network='preference_aware_mlp', input_dim=input_dim, pref_dim=pref_dim, **vae_args)
        else:
            model = VAE(network='mlp', input_dim=input_dim + pref_dim, **vae_args)
            
    elif model_type in ['rectified', 'gaussian', 'conditional', 'interpolation']:
        model = FM(network='preference_aware_mlp', input_dim=input_dim, output_dim=output_dim,
                   hidden_dim=config.hidden_dim, num_layers=config.num_layers,
                   time_step=config.time_step, output_norm=False, pred_type='velocity', pref_dim=pref_dim)
        if config.multi_pref_training_mode == 'preference_trajectory':
            pretrain_model_path = "main_part/saved_models/model_multi_pref_vae_final.pth"
            vae_args = dict(output_dim=output_dim, hidden_dim=config.hidden_dim,
                num_layers=config.num_layers, latent_dim=config.latent_dim,
                output_act=None, pred_type='node', use_cvae=True)
            pretrain_model = VAE(network='preference_aware_mlp', input_dim=input_dim, pref_dim=pref_dim, **vae_args)
            pretrain_model.load_state_dict(torch.load(pretrain_model_path, map_location=config.device))
            pretrain_model.to(config.device)  # Move model to device
            pretrain_model.eval()
            print(f"  Loaded pretrained VAE model: {pretrain_model_path}")
                   
    elif model_type == 'diffusion':
        model = DM(network='mlp', input_dim=input_dim + pref_dim, output_dim=output_dim,
                   hidden_dim=config.hidden_dim, num_layers=config.num_layers,
                   time_step=config.time_step, output_norm=False, pred_type='node')
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    

    if model: 
        model.to(config.device)
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}") 
            
    if not debug:
        model, _, _ = train_multi_preference(config, model, multi_pref_data, sys_data, config.device,
                                              model_type=model_type, pretrain_model=pretrain_model)
    else:
        print("\n[Debug Mode] Loading model...")
        path = f'{config.model_save_dir}/model_multi_pref_{model_type}_final.pth'
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=config.device, weights_only=True))
            model.eval()
            print(f"  Loaded: {path}")
    
    # Evaluation
    print("\n" + "=" * 80)
    print("Evaluation")
    print("=" * 80)
    
    test_lambdas = [0.0, 25.0, 50.0, 80.0, 90.0]
    results_all = {}
    
    vae_best_of_k = config.vae_best_of_k
    vae_use_mean = config.vae_use_mean
    vae_selection_mode = config.vae_selection_mode
    flow_best_of_k = config.flow_best_of_k
    flow_selection_mode = config.flow_selection_mode
    
    # Create NGT loss function if Best-of-K is enabled (for VAE or Flow)
    ngt_loss_fn = None
    need_ngt_loss = (model_type == 'vae' and vae_best_of_k > 1 and not vae_use_mean) or \
                   (model_type in ['rectified', 'gaussian', 'conditional', 'interpolation'] and flow_best_of_k > 1)
    
    if need_ngt_loss:
        from deepopf_ngt_loss import DeepOPFNGTLoss
        try:
            ngt_loss_fn = DeepOPFNGTLoss(sys_data, config)
            ngt_loss_fn.cache_to_gpu(config.device)
            if model_type in ['rectified', 'gaussian', 'conditional', 'interpolation']:
                print(f"[Eval] Flow Best-of-K enabled: K={flow_best_of_k}, mode={flow_selection_mode}")
            else:
                print(f"[Eval] VAE Best-of-K enabled: K={vae_best_of_k}, mode={vae_selection_mode}")
        except Exception as e:
            print(f"[Warning] Failed to create ngt_loss_fn: {e}")
            print(f"[Warning] Best-of-K will be disabled.")
            flow_best_of_k = 1
            vae_use_mean = True
    
    def eval_on_lambdas(lambdas):
        """Evaluate model on given lambda values."""
        res = {}
        for lc in lambdas:
            print(f"\n--- lambda_carbon = {lc:.2f} ---") 

            ctx = build_ctx_from_multi_preference(config, sys_data, multi_pref_data, BRANFT, config.device, lambda_carbon=lc)
            predictor = MultiPreferencePredictor(
                model=model, multi_pref_data=multi_pref_data, lambda_carbon=lc, model_type=model_type,
                num_flow_steps=config.multi_pref_flow_steps, training_mode=config.multi_pref_training_mode,
                ngt_loss_fn=ngt_loss_fn, vae_n_samples=vae_best_of_k,
                vae_use_mean=vae_use_mean, vae_selection_mode=vae_selection_mode,
                flow_n_samples=flow_best_of_k, flow_selection_mode=flow_selection_mode,
                pretrain_model=pretrain_model
            )
            res[lc] = evaluate_unified(ctx, predictor, apply_post_processing=True, verbose=True)
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


if __name__ == "__main__":
    debug = bool(int(os.environ.get('DEBUG', '0')))
    main(debug=debug)
