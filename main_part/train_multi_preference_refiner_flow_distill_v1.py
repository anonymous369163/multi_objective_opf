#!/usr/bin/env python
# coding: utf-8
"""
One-Step Refiner-Flow Distillation for Multi-Preference DeepOPF-V

What this script does
---------------------
You currently have:
  (A) a trajectory Flow model trained on adjacent preference segments (teacher candidate)
      - it is accurate, but to reach an arbitrary lambda you typically traverse multiple segments
  (B) an MLP anchor model (fast but has bias)
  (C) a SimpleRefinerMLP in train_multi_preference_tfm_refiner_v2.py (direct regression anchor -> GT(lambda=0))

Goal of this script
-------------------
Train a *one-step* Refiner-Flow Student model that maps:
    (scene, x_anchor, lambda_target)  -->  x*(lambda_target)
in ONE forward pass, using Rectified / displacement-style flow-matching on a bridge
between anchor and the target solution.

Crucially, we also support *continuous* lambda targets by generating pseudo labels
from a pre-trained *trajectory flow teacher* (your existing trajectory Flow model).
This matches your idea:
  - supervised on discrete lambdas (GT)
  - plus continuous lambdas where teacher provides pseudo x_lambda

Key design
----------
Student is an FM model whose predict_vec(scene, x_t, t_bridge, pref=lambda_target)
returns velocity. We train with displacement target:
    (1 - t_bridge) * v_pred  ≈  x_target - x_t

At inference (t_bridge=0), this becomes one-step:
    x_hat = x_anchor + v_pred(scene, x_anchor, 0, lambda_target)

Dependencies
------------
This script assumes the same project structure as your existing scripts:
  - config.py, data_loader.py, mlp_anchor.py
  - flow_model/net_utiles.py provides FM

Author: generated with reference to:
  - main_part/train_multi_preference_tfm_refiner_v2.py (multi-pref + refiner pipeline)
  - main_part/train_standard.py (rectified-flow style training pattern)
Date: Jan 2026
"""

import os
import sys
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BaseConfig, _SCRIPT_DIR
from data_loader import load_multi_preference_dataset, create_multi_preference_dataloader


# ==================== Config ====================

class OneStepRefinerDistillConfig(BaseConfig):
    """Configuration for one-step refiner-flow distillation."""
    def __init__(self):
        super().__init__()

        # Dataset
        self.multi_pref_dataset_path = os.path.join(
            os.path.dirname(_SCRIPT_DIR),
            'saved_data', 'multi_preference_solutions',
            'fully_covered_dataset_2026-01-02.pt'
        )

        # Teacher architecture (MUST match pre-trained teacher checkpoint)
        self.teacher_hidden_dim = int(os.environ.get("TEACHER_HIDDEN_DIM", "128"))  # Fixed: match teacher ckpt
        self.teacher_num_layers = int(os.environ.get("TEACHER_NUM_LAYERS", "2"))    # Fixed: match teacher ckpt
        
        # Student architecture (can be different from teacher)
        self.hidden_dim = int(os.environ.get("HIDDEN_DIM", "256"))  # Student: larger capacity
        self.num_layers = int(os.environ.get("NUM_LAYERS", "3"))    # Student: deeper network
        
        self.time_step = 1000
        self.pref_dim = 1

        # Student (refiner-flow) training
        self.epochs = int(os.environ.get("REFINER_FLOW_EPOCHS", "1000"))  # quick test: 600
        self.lr = float(os.environ.get("REFINER_FLOW_LR", "1e-4"))
        self.weight_decay = float(os.environ.get("WEIGHT_DECAY", "1e-6"))
        self.batch_size_training = int(os.environ.get("BATCH_SIZE", "100"))
        self.multi_pref_batch_size = self.batch_size_training  # for create_multi_preference_dataloader compatibility
        self.multi_pref_val_ratio = float(os.environ.get('MULTI_PREF_VAL_RATIO', '0.2'))
        self.multi_pref_random_seed = int(os.environ.get('MULTI_PREF_RANDOM_SEED', '1'))
        self.p_epoch = int(os.environ.get("P_EPOCH", "10"))
        self.s_epoch = int(os.environ.get("S_EPOCH", "200"))

        # Bridge sampling (rectified / displacement FM)
        self.bridge_alpha_eps = float(os.environ.get("BRIDGE_ALPHA_EPS", "1e-3"))   # for t in (eps, 1-eps)
        self.bridge_sigma = float(os.environ.get("BRIDGE_SIGMA", "0.00"))          # >0 for denoise robustness

        # Loss weights
        self.w_main = float(os.environ.get("W_MAIN", "1.0"))
        self.w_endpoint = float(os.environ.get("W_ENDPOINT", "1.0"))               # increased: 0.5 -> 1.0 (critical for one-step)
        self.w_consistency = float(os.environ.get("W_CONSISTENCY", "0.2"))         # optional

        # Continuous lambda pseudo-labels
        self.pseudo_ratio = float(os.environ.get("PSEUDO_RATIO", "0.2"))           # conservative: rely more on GT
        self.pseudo_lambda_min = float(os.environ.get("PSEUDO_LAMBDA_MIN", "0.0")) # normalized
        self.pseudo_lambda_max = float(os.environ.get("PSEUDO_LAMBDA_MAX", "1.0")) # normalized
        self.teacher_steps = int(os.environ.get("TEACHER_STEPS", "50"))            # for better pseudo labels
        self.teacher_ckpt = os.environ.get(
            "TEACHER_CKPT",
            os.path.join(self.model_save_dir, "model_multi_pref_rectified_traj_tfm_final.pth")
        )
        self.teacher_apply_wrap_each_step = os.environ.get("TEACHER_WRAP_EACH_STEP", "1").lower() in ["1", "true", "yes"]

        # If teacher checkpoint missing: (optional) user can set to 1 and provide training script separately
        self.require_teacher = os.environ.get("REQUIRE_TEACHER", "1").lower() in ["1", "true", "yes"]

        # === Far-lambda emphasis: sample and weight more on high lambda ===
        # GT branch: sample j with prob ∝ lambda_norm_tensor[j]^p (p=0 uniform, p=1~2 favor high lambda)
        self.gt_sampling_power = float(os.environ.get("GT_SAMPLING_POWER", "0.0"))  # gentle bias
        # Pseudo branch: use Beta(alpha, beta) distribution (alpha>beta favors high lambda)
        self.pseudo_beta_alpha = float(os.environ.get("PSEUDO_BETA_ALPHA", "1.0"))  # mild bias toward high λ
        self.pseudo_beta_beta = float(os.environ.get("PSEUDO_BETA_BETA", "1.0"))  # uniform
        # Loss weighting: loss *= (1 + c * lambda^p), higher c means more emphasis on far lambda
        self.loss_lambda_weight_c = float(os.environ.get("LOSS_LAMBDA_WEIGHT_C", "0.0"))  # conservative
        self.loss_lambda_weight_p = float(os.environ.get("LOSS_LAMBDA_WEIGHT_P", "1.0"))
        
        # === Endpoint-specific far-lambda emphasis (deployment is t=0) ===
        # Endpoint loss gets EXTRA lambda weighting: w_end(λ) = w_endpoint * (1 + a*λ^p) * base_lambda_weight
        # Set a > 0 to further emphasize endpoint accuracy at high lambda
        self.endpoint_lambda_extra_a = float(os.environ.get("ENDPOINT_LAMBDA_EXTRA_A", "0.5"))  # conservative
        self.endpoint_lambda_extra_p = float(os.environ.get("ENDPOINT_LAMBDA_EXTRA_P", "1.0"))
        
        # === Bridge t sampling: Beta distribution to favor t near 0 ===
        # Beta(0.5, 1) favors t~0 (mean≈0.33), Beta(1, 1) is uniform (default)
        # Since deployment is t=0, training more on small t can help
        self.bridge_t_beta_alpha = float(os.environ.get("BRIDGE_T_BETA_ALPHA", "1.0"))  # uniform (conservative)
        self.bridge_t_beta_beta = float(os.environ.get("BRIDGE_T_BETA_BETA", "1.0"))

        # === Curriculum Learning: difficulty-aware sampling ===
        # Divide λ into M bins, track endpoint loss EMA per bin, sample harder bins more
        self.curriculum_enabled = os.environ.get("CURRICULUM_ENABLED", "0").lower() in ["1", "true", "yes"]
        self.curriculum_start_epoch = int(os.environ.get("CURRICULUM_START_EPOCH", "100"))  # warmup first
        self.curriculum_num_bins = int(os.environ.get("CURRICULUM_NUM_BINS", "10"))  # M bins
        self.curriculum_ema_beta = float(os.environ.get("CURRICULUM_EMA_BETA", "0.02"))  # EMA update rate
        self.curriculum_gamma = float(os.environ.get("CURRICULUM_GAMMA", "0.5"))  # difficulty -> weight power
        self.curriculum_w_min = float(os.environ.get("CURRICULUM_W_MIN", "0.5"))  # min weight per bin
        self.curriculum_w_max = float(os.environ.get("CURRICULUM_W_MAX", "2.0"))  # max weight per bin
        self.curriculum_alpha_max = float(os.environ.get("CURRICULUM_ALPHA_MAX", "0.5"))  # max hard sampling ratio
        self.curriculum_alpha_warmup_epochs = int(os.environ.get("CURRICULUM_ALPHA_WARMUP", "200"))  # epochs to reach alpha_max

        # === Network configuration ===
        self.student_network = os.environ.get("STUDENT_NETWORK", "preference_aware_mlp")
        self.hyper_rank = int(os.environ.get("HYPER_RANK", "16"))
        self.hyper_use_time = os.environ.get("HYPER_USE_TIME", "False").lower() == "true"  # Default: False for stability
        self.hyper_use_scene = os.environ.get("HYPER_USE_SCENE", "True").lower() == "true"  # Default: True for scene adaptation
        self.hyper_scene_dim = int(os.environ.get("HYPER_SCENE_DIM", "32"))

        # Saving
        self.tag = os.environ.get("TAG", "distill_v1")

    def print_config(self):
        super().print_config()
        print("\n[One-Step Refiner-Flow Distillation]")
        print(f"  epochs={self.epochs}, lr={self.lr:.0e}, batch={self.batch_size_training}")
        print(f"  Teacher: hidden_dim={self.teacher_hidden_dim}, num_layers={self.teacher_num_layers}")
        print(f"  Student: hidden_dim={self.hidden_dim}, num_layers={self.num_layers}")
        print(f"  bridge_sigma={self.bridge_sigma}, alpha_eps={self.bridge_alpha_eps}")
        print(f"  w_main={self.w_main}, w_endpoint={self.w_endpoint}, w_consistency={self.w_consistency}")
        print(f"  pseudo_ratio={self.pseudo_ratio}, teacher_steps={self.teacher_steps}")
        print(f"  teacher_ckpt={self.teacher_ckpt}")
        print(f"  [Far-lambda emphasis]")
        print(f"    GT sampling power={self.gt_sampling_power}")
        print(f"    Pseudo Beta(α={self.pseudo_beta_alpha}, β={self.pseudo_beta_beta})")
        print(f"    Loss weight: (1 + {self.loss_lambda_weight_c}*λ^{self.loss_lambda_weight_p})")
        print(f"    Endpoint extra: (1 + {self.endpoint_lambda_extra_a}*λ^{self.endpoint_lambda_extra_p})")
        print(f"    Bridge t ~ Beta({self.bridge_t_beta_alpha}, {self.bridge_t_beta_beta})")
        print(f"  [Curriculum Learning]")
        print(f"    enabled={self.curriculum_enabled}, start_epoch={self.curriculum_start_epoch}")
        print(f"    bins={self.curriculum_num_bins}, ema_β={self.curriculum_ema_beta}, γ={self.curriculum_gamma}")
        print(f"    w_range=[{self.curriculum_w_min}, {self.curriculum_w_max}], α_max={self.curriculum_alpha_max}")
        print(f"  [HyperNetwork]")
        print(f"    student_network={self.student_network}, rank={self.hyper_rank}")
        print(f"    use_time_in_hyper={self.hyper_use_time}, use_scene_in_hyper={self.hyper_use_scene}")
        print(f"    scene_pool_dim={self.hyper_scene_dim}")
        print(f"  tag={self.tag}")


def get_config() -> OneStepRefinerDistillConfig:
    return OneStepRefinerDistillConfig()


# ==================== Angle utils ====================

def wrap_angle_difference(dx: torch.Tensor, NPred_Va: int) -> torch.Tensor:
    """Wrap angle differences for Va dims to [-pi, pi]."""
    if NPred_Va <= 0:
        return dx
    out = dx.clone()
    out[..., :NPred_Va] = torch.atan2(torch.sin(dx[..., :NPred_Va]), torch.cos(dx[..., :NPred_Va]))
    return out


def wrap_angles(x: torch.Tensor, NPred_Va: int) -> torch.Tensor:
    """Wrap angle values for Va dims to [-pi, pi]."""
    if NPred_Va <= 0:
        return x
    out = x.clone()
    out[..., :NPred_Va] = torch.atan2(torch.sin(x[..., :NPred_Va]), torch.cos(x[..., :NPred_Va]))
    return out


# ==================== Anchor loader (MLP only) ====================

@torch.no_grad()
def get_anchor_from_mlp(pretrain_model, scene: torch.Tensor) -> torch.Tensor:
    """
    Deterministic anchor from Standard MLP.

    pretrain_model(scene, use_mean=True, pref=pref0) is used in your existing multi-pref code.
    """
    pref0 = torch.zeros((scene.shape[0], 1), device=scene.device, dtype=scene.dtype)
    return pretrain_model(scene, use_mean=True, pref=pref0)


# ==================== Teacher: continuous-lambda generation ====================

@torch.no_grad()
def teacher_integrate_to_lambda(
    teacher,
    scene: torch.Tensor,
    x0: torch.Tensor,
    lambda_start: torch.Tensor,    # normalized in [0,1], shape [B,1] - starting lambda
    lambda_target: torch.Tensor,   # normalized in [0,1], shape [B,1] - target lambda
    steps: int,
    NPred_Va: int,
    wrap_each_step: bool = True,
) -> torch.Tensor:
    """
    Use the trajectory-flow teacher as a local vector field v(x, lambda) and integrate from lambda_start to lambda_target.

    Euler integration (offline / for pseudo labels):
      x_{n+1} = x_n + dλ * v(scene, x_n, λ_n, λ_n)

    Note:
      - Teacher in your current implementation uses predict_vec(scene, x_t, lam_t, lam_t).
      - We keep time==pref==current lambda for teacher usage (matches training call convention).
      - Integration starts from lambda_start (not necessarily 0), allowing shorter paths
        when starting from the nearest GT grid point.
    """
    steps = int(max(1, steps))
    
    # Compute delta lambda to integrate
    delta_lambda = torch.clamp(lambda_target - lambda_start, min=0.0)  # [B, 1]
    
    # If delta is very small, just return x0
    # (avoid division issues and unnecessary computation)
    if delta_lambda.max().item() < 1e-6:
        return wrap_angles(x0.clone(), NPred_Va)
    
    lam_t = lambda_start.clone()
    x = x0.clone()
    # per-sample step size; constant across loop
    dlam = delta_lambda / float(steps)

    for _ in range(steps):
        v = teacher.predict_vec(scene, x, lam_t, lam_t)
        x = x + dlam * v
        if wrap_each_step:
            x = wrap_angles(x, NPred_Va)
        lam_t = lam_t + dlam

    return wrap_angles(x, NPred_Va)


# ==================== Student loss: rectified / displacement FM ====================

def compute_student_losses(
    student,
    scene: torch.Tensor,
    x_anchor: torch.Tensor,
    x_target: torch.Tensor,
    lambda_target: torch.Tensor,  # [B,1] normalized
    config: OneStepRefinerDistillConfig,
    NPred_Va: int,
) -> Tuple[torch.Tensor, dict]:
    """
    Rectified / displacement training on the anchor->target bridge.

    Main loss:
      sample t ~ Beta(alpha, beta) or U(eps, 1-eps)
      x_t = (1-t)*x_anchor + t*x_target + sigma*sqrt(t(1-t))*noise
      delta_target = x_target - x_t
      v_pred = student.predict_vec(scene, x_t, t, lambda_target)
      delta_pred = (1-t)*v_pred
      loss_main = MSE(delta_pred, delta_target)

    Endpoint one-step loss:
      v0 = student.predict_vec(scene, x_anchor, 0, lambda_target)
      delta0 = v0
      loss_endpoint = MSE(delta0, x_target - x_anchor)
      
    Note: t sampling uses Beta(alpha, beta) to favor t near 0 (deployment scenario).
          Beta(0.5, 1) has mean ≈ 0.33, Beta(1, 1) is uniform.
    """
    B = scene.shape[0]
    device = scene.device
    dtype = scene.dtype

    eps = float(getattr(config, "bridge_alpha_eps", 1e-3))
    
    # === Bridge t sampling: Beta distribution to favor t near 0 ===
    t_beta_alpha = float(getattr(config, "bridge_t_beta_alpha", 1.0))
    t_beta_beta = float(getattr(config, "bridge_t_beta_beta", 1.0))
    
    if t_beta_alpha != 1.0 or t_beta_beta != 1.0:
        # Use Beta distribution (alpha < 1 favors values near 0)
        t_dist = torch.distributions.Beta(
            torch.tensor(t_beta_alpha, device=device, dtype=dtype),
            torch.tensor(t_beta_beta, device=device, dtype=dtype)
        )
        t = t_dist.sample((B, 1))
    else:
        # Uniform sampling (original behavior)
        t = torch.rand((B, 1), device=device, dtype=dtype)
    
    t = torch.clamp(t, min=eps, max=1.0 - eps)

    mu = (1.0 - t) * x_anchor + t * x_target

    sigma = float(getattr(config, "bridge_sigma", 0.0))
    if sigma > 0:
        noise = torch.randn_like(mu)
        x_t = mu + (sigma * torch.sqrt(torch.clamp(t * (1.0 - t), min=0.0))) * noise
    else:
        x_t = mu

    # targets
    delta_target = wrap_angle_difference(x_target - x_t, NPred_Va)

    # predicted displacement via velocity * remaining time
    v_pred = student.predict_vec(scene, x_t, t, lambda_target)
    delta_pred = (1.0 - t) * v_pred
    delta_err = wrap_angle_difference(delta_pred - delta_target, NPred_Va)
    
    # === Far-lambda loss weighting: weight = (1 + c * lambda^p) ===
    # Higher lambda gets higher weight, emphasizing accuracy on far lambdas
    loss_c = float(getattr(config, "loss_lambda_weight_c", 0.0))
    loss_p = float(getattr(config, "loss_lambda_weight_p", 1.0))
    if loss_c > 0:
        # lambda_target: [B, 1], compute per-sample weight
        lambda_weight = 1.0 + loss_c * (lambda_target ** loss_p)  # [B, 1]
        # Weighted MSE: mean over dims, then weighted mean over batch
        per_sample_loss_main = torch.mean(delta_err ** 2, dim=-1, keepdim=True)  # [B, 1]
        loss_main = torch.mean(per_sample_loss_main * lambda_weight)
    else:
        loss_main = torch.mean(delta_err ** 2)

    # endpoint (one-step) loss at t=0 - THIS IS THE DEPLOYMENT SCENARIO
    t0 = torch.zeros((B, 1), device=device, dtype=dtype)
    v0 = student.predict_vec(scene, x_anchor, t0, lambda_target)
    delta0 = v0
    delta0_target = wrap_angle_difference(x_target - x_anchor, NPred_Va)
    delta0_err = wrap_angle_difference(delta0 - delta0_target, NPred_Va)
    
    # === Endpoint loss gets EXTRA lambda weighting (deployment is t=0) ===
    # w_end(λ) = base_lambda_weight * (1 + a*λ^p)
    # This further emphasizes endpoint accuracy at high lambda
    endpoint_extra_a = float(getattr(config, "endpoint_lambda_extra_a", 0.0))
    endpoint_extra_p = float(getattr(config, "endpoint_lambda_extra_p", 1.0))
    
    per_sample_loss_endpoint = torch.mean(delta0_err ** 2, dim=-1, keepdim=True)  # [B, 1]
    
    # Compute endpoint weight: base_lambda_weight * extra_endpoint_weight
    endpoint_weight = torch.ones_like(lambda_target)
    if loss_c > 0:
        endpoint_weight = endpoint_weight * lambda_weight  # apply base lambda weight
    if endpoint_extra_a > 0:
        endpoint_extra_weight = 1.0 + endpoint_extra_a * (lambda_target ** endpoint_extra_p)
        endpoint_weight = endpoint_weight * endpoint_extra_weight  # apply extra endpoint weight
    
    loss_endpoint = torch.mean(per_sample_loss_endpoint * endpoint_weight)

    # optional consistency (two-step vs one-step), uses the same student twice
    loss_cons = torch.zeros((), device=device, dtype=dtype)
    if float(getattr(config, "w_consistency", 0.0)) > 0:
        t1 = torch.rand((B, 1), device=device, dtype=dtype)
        t2 = torch.rand((B, 1), device=device, dtype=dtype)
        # ensure t1 < t2 and both in (eps,1-eps)
        t1 = torch.clamp(t1, min=eps, max=1.0 - eps)
        t2 = torch.clamp(t2, min=eps, max=1.0 - eps)
        t_low = torch.minimum(t1, t2)
        t_high = torch.maximum(t1, t2)

        x_t1 = (1.0 - t_low) * x_anchor + t_low * x_target
        x_t1 = wrap_angles(x_t1, NPred_Va)

        # Predict velocity at t_low (used for both paths)
        v1 = student.predict_vec(scene, x_t1, t_low, lambda_target)
        
        # Path 1: direct to target from t_low
        x_dir = wrap_angles(x_t1 + (1.0 - t_low) * v1, NPred_Va)

        # Path 2: two-step (t_low -> t_high -> target)
        # step from t_low -> t_high using the same velocity v1
        x_t2_hat = wrap_angles(x_t1 + (t_high - t_low) * v1, NPred_Va)
        # then from t_high to target
        v2 = student.predict_vec(scene, x_t2_hat, t_high, lambda_target)
        x_2 = wrap_angles(x_t2_hat + (1.0 - t_high) * v2, NPred_Va)

        diff = wrap_angle_difference(x_dir - x_2, NPred_Va)
        loss_cons = torch.mean(diff ** 2)

    w_main = float(getattr(config, "w_main", 1.0))
    w_endpoint = float(getattr(config, "w_endpoint", 0.0))
    w_cons = float(getattr(config, "w_consistency", 0.0))

    total = w_main * loss_main + w_endpoint * loss_endpoint + w_cons * loss_cons
    stats = {
        "loss_main": float(loss_main.detach().cpu().item()),
        "loss_endpoint": float(loss_endpoint.detach().cpu().item()),
        "loss_cons": float(loss_cons.detach().cpu().item()),
        "loss_total": float(total.detach().cpu().item()),
    }
    return total, stats


# ==================== Checkpoint ====================

def save_student(config: OneStepRefinerDistillConfig, student, tag: str):
    os.makedirs(config.model_save_dir, exist_ok=True)
    path = os.path.join(config.model_save_dir, f"model_multi_pref_refiner_flow_onestep_{config.tag}_{tag}.pth")
    torch.save(student.state_dict(), path, _use_new_zipfile_serialization=False)
    print(f"  Saved: {os.path.basename(path)}")


# ==================== Main train ====================

def main():
    config = get_config()
    print("=" * 80)
    print("One-Step Refiner-Flow Distillation (MLP anchor + teacher pseudo continuous lambdas)")
    print("=" * 80)
    config.print_config()

    device = config.device

    # Load multi-preference dataset
    multi_pref_data, sys_data = load_multi_preference_dataset(config)
    input_dim = int(multi_pref_data["input_dim"])
    output_dim = int(multi_pref_data["output_dim"])
    NPred_Va = int(multi_pref_data["NPred_Va"])
    NPred_Vm = int(multi_pref_data["NPred_Vm"])

    # Stack GT trajectory
    # Ensure float32 dtype for consistency
    y_train_by_pref = {lc: y.to(device=device, dtype=torch.float32) for lc, y in multi_pref_data["y_train_by_pref"].items()}
    lambda_values = multi_pref_data["lambda_carbon_values"]
    lambda_sorted = sorted(lambda_values)
    lam_min, lam_max = float(lambda_sorted[0]), float(lambda_sorted[-1])
    lam_norm = {lc: (float(lc) - lam_min) / (lam_max - lam_min) if lam_max > lam_min else 0.0 for lc in lambda_sorted}
    y_stacked = torch.stack([y_train_by_pref[lc] for lc in lambda_sorted], dim=0)  # [K, N, D]
    lambda_norm_tensor = torch.tensor([lam_norm[lc] for lc in lambda_sorted], device=device, dtype=torch.float32)  # [K]
    
    print(f"  y_stacked dtype: {y_stacked.dtype}")  # Should be float32

    K = y_stacked.shape[0]
    N = y_stacked.shape[1]

    print(f"\nData: N={N} samples, K={K} preferences")
    print(f"y_stacked: {tuple(y_stacked.shape)}, lambda_norm_tensor: {tuple(lambda_norm_tensor.shape)}")

    # Dataloader over scenes (batch_x, batch_idx)
    dataloader = create_multi_preference_dataloader(multi_pref_data, config, shuffle=True)

    # Load MLP anchor
    from mlp_anchor import load_standard_mlp_anchor
    pretrain_model = load_standard_mlp_anchor(config, sys_data, multi_pref_data, device)
    pretrain_model.eval()
    print("  Using Standard MLP as the ONLY anchor")

    # Load teacher trajectory flow
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "flow_model"))
    from net_utiles import FM

    # Teacher: use teacher-specific architecture (must match pre-trained checkpoint)
    teacher = FM(
        network="preference_aware_mlp",
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=config.teacher_hidden_dim,  # Use teacher architecture
        num_layers=config.teacher_num_layers,  # Use teacher architecture
        time_step=config.time_step,
        output_norm=False,
        pred_type="velocity",
        pref_dim=config.pref_dim,
    ).to(device)
    if os.path.exists(config.teacher_ckpt):
        teacher.load_state_dict(torch.load(config.teacher_ckpt, map_location=device, weights_only=True))
        print(f"  Loaded teacher ckpt: {config.teacher_ckpt}")
    else:
        msg = f"[Teacher missing] teacher_ckpt not found: {config.teacher_ckpt}"
        if config.require_teacher:
            raise FileNotFoundError(msg)
        else:
            print("  " + msg)
            print("  WARNING: pseudo labels disabled because teacher is missing.")
            config.pseudo_ratio = 0.0

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Create student refiner-flow (same FM backbone, but conditioned on lambda_target)
    # 
    # IMPORTANT: Semantic Difference between Teacher and Student
    # ============================================================
    # Teacher (trajectory flow): predict_vec(scene, x, lam_curr, lam_curr)
    #   - time parameter = current lambda position
    #   - pref parameter = current lambda position (same as time)
    #   - Trained to: (lam_next - lam_curr) * v ≈ x_next - x_curr
    #
    # Student (one-step refiner): predict_vec(scene, x_t, t_bridge, lambda_target)
    #   - time parameter = bridge time t ∈ [0, 1] (progress from anchor to target)
    #   - pref parameter = target lambda (constant for each sample)
    #   - Trained to: (1 - t) * v ≈ x_target - x_t (displacement on anchor-target bridge)
    #
    # This reinterpretation of the time embedding is intentional for one-step distillation.
    # ============================================================
    # Network type selection via config (environment variable or default)
    #   - "preference_aware_mlp": Traditional FiLM-based (baseline, recommended)
    #   - "scale_aware_preference_mlp": Per-dimension scale (V2)
    #   - "hyper_last_layer_mlp": HyperNetwork with low-rank last layer
    student_network = config.student_network
    print(f"\n[Student Network] Using: {student_network}")
    
    student = FM(
        network=student_network,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        time_step=config.time_step,
        output_norm=False,
        pred_type="velocity",
        pref_dim=config.pref_dim,
        n_va=NPred_Va,  # Used by scale_aware_preference_mlp, ignored by others
    ).to(device)

    print(f"\nTeacher params: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"Student params: {sum(p.numel() for p in student.parameters()):,}")

    # Optimizer with cosine annealing scheduler (gentler decay)
    optimizer = torch.optim.Adam(student.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # eta_min = 0.1 * lr for gentler decay (was 0.01, too aggressive)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.lr * 0.1)

    # Training loop
    student.train()
    start_time = time.process_time()

    # === Curriculum Learning: Initialize EMA difficulty per λ bin ===
    curriculum_enabled = bool(getattr(config, "curriculum_enabled", False))
    num_bins = int(getattr(config, "curriculum_num_bins", 10))
    ema_beta = float(getattr(config, "curriculum_ema_beta", 0.02))
    curriculum_gamma = float(getattr(config, "curriculum_gamma", 0.5))
    w_min = float(getattr(config, "curriculum_w_min", 0.5))
    w_max = float(getattr(config, "curriculum_w_max", 2.0))
    alpha_max = float(getattr(config, "curriculum_alpha_max", 0.5))
    alpha_warmup_epochs = int(getattr(config, "curriculum_alpha_warmup_epochs", 200))
    curriculum_start_epoch = int(getattr(config, "curriculum_start_epoch", 100))
    
    # EMA difficulty: initialized to 1.0 (equal difficulty)
    bin_ema_difficulty = torch.ones(num_bins, device=device, dtype=torch.float32)
    bin_edges = torch.linspace(0, 1, num_bins + 1, device=device)  # [0, 0.1, 0.2, ..., 1.0]
    
    if curriculum_enabled:
        print(f"\n[Curriculum Learning] Enabled, start at epoch {curriculum_start_epoch}")
        print(f"  {num_bins} bins, EMA β={ema_beta}, γ={curriculum_gamma}")
        print(f"  w_range=[{w_min}, {w_max}], α_max={alpha_max}, warmup={alpha_warmup_epochs} epochs")

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        epoch_loss_main = 0.0      # 追踪 bridge loss
        epoch_loss_endpoint = 0.0  # 追踪 endpoint loss (t=0)
        epoch_loss_cons = 0.0      # 追踪 consistency loss
        nb = 0
        
        # === Curriculum: compute current alpha (hard sampling ratio) ===
        curriculum_active = curriculum_enabled and (epoch >= curriculum_start_epoch)
        if curriculum_active:
            epochs_since_start = epoch - curriculum_start_epoch
            alpha_curriculum = min(alpha_max, alpha_max * epochs_since_start / max(alpha_warmup_epochs, 1))
        else:
            alpha_curriculum = 0.0

        for batch_x, batch_idx in dataloader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_idx = batch_idx.to(device, non_blocking=True)
            sample_idx = batch_idx.long()
            B = batch_x.shape[0]

            optimizer.zero_grad(set_to_none=True)

            # Anchor from MLP (ONLY) - must match inference behavior
            with torch.no_grad():
                x_anchor = get_anchor_from_mlp(pretrain_model, batch_x)
                x_anchor = wrap_angles(x_anchor, NPred_Va)

            # Prepare targets: mix GT (grid) and pseudo (continuous)
            use_pseudo = (torch.rand((B, 1), device=device) < float(config.pseudo_ratio)).view(-1)
            x_target = torch.empty((B, output_dim), device=device, dtype=torch.float32)
            lambda_target = torch.empty((B, 1), device=device, dtype=torch.float32)

            # --- GT branch (discrete grid) with curriculum + far-lambda emphasis ---
            if (~use_pseudo).any():
                bmask = ~use_pseudo
                B_gt = bmask.sum().item()
                
                # === Curriculum Learning: difficulty-aware sampling ===
                if curriculum_active and alpha_curriculum > 0:
                    # Compute hard sampling weights from EMA difficulty
                    # w_i = clip((e_i + eps)^gamma, w_min, w_max)
                    eps_diff = 1e-6
                    hard_weights = torch.clamp(
                        (bin_ema_difficulty + eps_diff) ** curriculum_gamma,
                        min=w_min, max=w_max
                    )  # [num_bins]
                    hard_weights = hard_weights / hard_weights.sum()  # normalize to prob
                    
                    # Uniform weights (1/num_bins each)
                    uniform_weights = torch.ones(num_bins, device=device) / num_bins
                    
                    # Mixed distribution: p = (1-α)*uniform + α*hard
                    mixed_bin_weights = (1 - alpha_curriculum) * uniform_weights + alpha_curriculum * hard_weights
                    mixed_bin_weights = mixed_bin_weights / mixed_bin_weights.sum()
                    
                    # Sample bin indices, then sample λ within each bin
                    bin_indices = torch.multinomial(mixed_bin_weights, B_gt, replacement=True)
                    # Map bin index to λ: uniform within [bin_edges[i], bin_edges[i+1])
                    lam_low = bin_edges[bin_indices]
                    lam_high = bin_edges[bin_indices + 1]
                    lam_sampled = lam_low + (lam_high - lam_low) * torch.rand(B_gt, device=device)
                    lam_sampled = torch.clamp(lam_sampled, 0.0, 1.0)
                    
                    # Map continuous λ to nearest GT grid index
                    j = torch.searchsorted(lambda_norm_tensor, lam_sampled, right=True) - 1
                    j = torch.clamp(j, 0, K - 1)
                else:
                    # === Original: Far-lambda emphasis with power weighting ===
                    gt_power = float(getattr(config, "gt_sampling_power", 0.0))
                    if gt_power > 0:
                        eps_prob = 0.1
                        weights = (eps_prob + lambda_norm_tensor) ** gt_power
                        weights = weights / weights.sum()
                        j = torch.multinomial(weights, B_gt, replacement=True)
                    else:
                        j = torch.randint(0, K, (B_gt,), device=device)
                
                x_target[bmask] = y_stacked[j, sample_idx[bmask], :]
                lambda_target[bmask, 0] = lambda_norm_tensor[j]

            # --- Pseudo branch (continuous) with far-lambda emphasis ---
            if use_pseudo.any() and float(config.pseudo_ratio) > 0:
                bmask = use_pseudo
                B_pseudo = bmask.sum().item()
                
                # === Far-lambda emphasis: sample lambda using Beta(alpha, beta) ===
                # Beta(2,1) favors high lambda, Beta(1,1) is uniform, Beta(1,2) favors low lambda
                beta_alpha = float(getattr(config, "pseudo_beta_alpha", 1.0))
                beta_beta = float(getattr(config, "pseudo_beta_beta", 1.0))
                
                if beta_alpha != 1.0 or beta_beta != 1.0:
                    # Use Beta distribution
                    beta_dist = torch.distributions.Beta(
                        torch.tensor(beta_alpha, device=device),
                        torch.tensor(beta_beta, device=device)
                    )
                    lam_u = beta_dist.sample((B_pseudo, 1))
                else:
                    # Uniform sampling (original behavior)
                    lam_u = torch.rand((B_pseudo, 1), device=device, dtype=torch.float32)
                
                # Scale to [pseudo_lambda_min, pseudo_lambda_max]
                lam_u = float(config.pseudo_lambda_min) + (float(config.pseudo_lambda_max) - float(config.pseudo_lambda_min)) * lam_u
                lam_u = torch.clamp(lam_u, 0.0, 1.0)

                # === KEY IMPROVEMENT: Start from NEAREST GT grid point ===
                # Find the largest k such that lambda_norm_tensor[k] <= lam_u
                # This minimizes integration distance and reduces accumulated error
                lam_u_flat = lam_u.view(-1)  # [B_pseudo]
                
                # searchsorted returns insertion point; subtract 1 to get floor index
                # right=True ensures we get the index where lam_u would be inserted to the right
                k_idx = torch.searchsorted(lambda_norm_tensor, lam_u_flat, right=True) - 1
                k_idx = torch.clamp(k_idx, 0, K - 1)  # ensure valid range [0, K-1]
                
                # Get starting lambda and GT solution at nearest grid point
                lambda_start = lambda_norm_tensor[k_idx].view(-1, 1)  # [B_pseudo, 1]
                x_start = y_stacked[k_idx, sample_idx[bmask], :]  # [B_pseudo, output_dim]
                
                # Integrate only the remaining delta: from lambda_start to lam_u
                x_pseudo = teacher_integrate_to_lambda(
                    teacher=teacher,
                    scene=batch_x[bmask],
                    x0=x_start,
                    lambda_start=lambda_start,
                    lambda_target=lam_u,
                    steps=int(config.teacher_steps),
                    NPred_Va=NPred_Va,
                    wrap_each_step=bool(config.teacher_apply_wrap_each_step),
                )
                x_target[bmask] = x_pseudo
                lambda_target[bmask] = lam_u

            x_target = wrap_angles(x_target, NPred_Va)

            # Compute student losses
            loss, stats = compute_student_losses(
                student=student,
                scene=batch_x,
                x_anchor=x_anchor,
                x_target=x_target,
                lambda_target=lambda_target,
                config=config,
                NPred_Va=NPred_Va,
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            # === Curriculum Learning: Update EMA difficulty per bin ===
            if curriculum_active:
                with torch.no_grad():
                    # Compute per-sample endpoint error (for EMA update)
                    t0 = torch.zeros((B, 1), device=device, dtype=torch.float32)
                    v0 = student.predict_vec(batch_x, x_anchor, t0, lambda_target)
                    delta0_target = wrap_angle_difference(x_target - x_anchor, NPred_Va)
                    delta0_err = wrap_angle_difference(v0 - delta0_target, NPred_Va)
                    per_sample_endpoint_err = torch.mean(delta0_err ** 2, dim=-1)  # [B]
                    
                    # Assign each sample to a bin based on lambda_target
                    lambda_flat = lambda_target.view(-1)  # [B]
                    bin_idx = torch.clamp(
                        (lambda_flat * num_bins).long(),
                        min=0, max=num_bins - 1
                    )  # [B]
                    
                    # Update EMA for each bin that has samples
                    for b_idx in range(num_bins):
                        mask_bin = (bin_idx == b_idx)
                        if mask_bin.any():
                            bin_mean_err = per_sample_endpoint_err[mask_bin].mean()
                            bin_ema_difficulty[b_idx] = (
                                (1 - ema_beta) * bin_ema_difficulty[b_idx] +
                                ema_beta * bin_mean_err
                            )

            epoch_loss += float(loss.detach().cpu().item())
            epoch_loss_main += stats["loss_main"]
            epoch_loss_endpoint += stats["loss_endpoint"]
            epoch_loss_cons += stats["loss_cons"]
            nb += 1

        # Update learning rate
        scheduler.step()
        
        if (epoch + 1) % config.p_epoch == 0:
            current_lr = scheduler.get_last_lr()[0]
            avg_main = epoch_loss_main / max(nb, 1)
            avg_endpoint = epoch_loss_endpoint / max(nb, 1)
            avg_cons = epoch_loss_cons / max(nb, 1)
            avg_total = epoch_loss / max(nb, 1)
            
            # 打印各损失分量（未加权的原始值）用于量纲分析
            print(f"Epoch {epoch+1:4d}/{config.epochs}: total={avg_total:.4e} | "
                  f"main={avg_main:.4e}, endpoint={avg_endpoint:.4e}, cons={avg_cons:.4e} "
                  f"(lr={current_lr:.2e})")
            
            # 打印加权后的贡献（第一个 epoch 打印权重配置）
            if (epoch + 1) == config.p_epoch:
                w_main = float(getattr(config, "w_main", 1.0))
                w_endpoint = float(getattr(config, "w_endpoint", 0.0))
                w_cons = float(getattr(config, "w_consistency", 0.0))
                print(f"  [Loss weights] w_main={w_main}, w_endpoint={w_endpoint}, w_cons={w_cons}")
                print(f"  [Weighted contribution] main*w={avg_main*w_main:.4e}, "
                      f"endpoint*w={avg_endpoint*w_endpoint:.4e}, cons*w={avg_cons*w_cons:.4e}")
            
            # Show curriculum info if active
            if curriculum_active:
                diff_str = ", ".join([f"{d:.2e}" for d in bin_ema_difficulty.tolist()])
                print(f"  [Curriculum] α={alpha_curriculum:.2f}, Bin EMA: [{diff_str}]")

        if (epoch + 1) % 100 == 0 and (epoch + 1) >= config.s_epoch:
            save_student(config, student, f"E{epoch+1}")

    elapsed = time.process_time() - start_time
    print(f"\nTraining done in {elapsed/60:.2f} min")
    save_student(config, student, "final")

    print("\nInference note:")
    print("  After training, one-step prediction is:")
    print("    x_anchor = MLP(scene, pref=0)")
    print("    x_hat(lambda) = x_anchor + student.predict_vec(scene, x_anchor, t=0, pref=lambda)")


if __name__ == "__main__":
    main()
