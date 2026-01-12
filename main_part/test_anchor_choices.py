#!/usr/bin/env python3
"""
Test different anchor choices for trajectory student distillation.

Purpose:
    Find the best anchor for training Student model that will work at inference time.
    
Anchor candidates:
    1. Standard MLP (λ=0): Trained on cost-only OPF, available at inference
    2. VAE (λ=0): Preference-aware VAE at λ=0, available at inference
    3. Random Gaussian: Simple noise, always available
    4. GT[λ=0]: Ground truth at λ=0, ONLY available during training

Key question:
    Which anchor gives strongest V_target signal AND is available at inference?
    
Metrics to compare:
    - V_target magnitude (larger = easier to learn)
    - V_target variation across λ (more variation = better λ dependency)
    - Anchor accuracy at λ=0 (closer to GT = smaller V_target at λ=0)

Author: AI Assistant
Date: January 2026
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'flow_model'))

from distill_traj_student import (DistillTrajStudentConfig, build_y_star_fine, 
                                   build_fine_pref_grid, compute_floor_start_indices, 
                                   wrap_angles, wrap_angle_difference)
from data_loader import load_multi_preference_dataset
from net_utiles import FM, VAE
from mlp_anchor import load_standard_mlp_anchor


def main():
    config = DistillTrajStudentConfig()
    multi_pref_data, sys_data = load_multi_preference_dataset(config)
    device = config.device

    # Setup
    lambda_values = list(multi_pref_data['lambda_carbon_values'])
    lambda_sorted = sorted([float(x) for x in lambda_values])
    lam_min, lam_max = float(lambda_sorted[0]), float(lambda_sorted[-1])

    y_train_by_pref = {float(lc): y.to(device=device, dtype=torch.float32) 
                       for lc, y in multi_pref_data['y_train_by_pref'].items()}
    y_stacked_gt = torch.stack([y_train_by_pref[lc] for lc in lambda_sorted], dim=0)
    
    gt_norm = torch.tensor([(lc - lam_min) / (lam_max - lam_min) for lc in lambda_sorted], 
                           device=device, dtype=torch.float32)
    fine_norm, _, _, _ = build_fine_pref_grid(lambda_sorted, config.fine_k, config.fine_step, device)
    start_idx_for_fine = compute_floor_start_indices(gt_norm, fine_norm)

    n_va = int(multi_pref_data['NPred_Va'])
    input_dim = int(multi_pref_data['input_dim'])
    output_dim = int(multi_pref_data['output_dim'])
    
    K = y_stacked_gt.shape[0]
    Kf = fine_norm.shape[0]

    # Load Teacher
    teacher = FM(network='preference_aware_mlp', input_dim=input_dim, output_dim=output_dim,
        hidden_dim=config.teacher_hidden_dim, num_layers=config.teacher_num_layers,
        time_step=config.time_step, output_norm=False, pred_type='velocity', pref_dim=config.pref_dim).to(device)
    teacher.load_state_dict(torch.load(config.teacher_ckpt, map_location=device, weights_only=True), strict=False)
    teacher.eval()
    print(f'Loaded teacher: {config.teacher_ckpt}')

    # Load Standard MLP anchor
    try:
        mlp_anchor = load_standard_mlp_anchor(config, sys_data, multi_pref_data, device)
        mlp_anchor.eval()
        print(f'Loaded Standard MLP anchor')
        has_mlp = True
    except Exception as e:
        print(f'[WARN] Standard MLP not available: {e}')
        has_mlp = False

    # Load VAE anchor
    vae_args = dict(output_dim=output_dim, hidden_dim=config.vae_hidden_dim, num_layers=config.vae_num_layers,
                    latent_dim=config.vae_latent_dim, output_act=None, pred_type='node', use_cvae=True)
    vae_anchor = VAE(network='preference_aware_mlp', input_dim=input_dim, pref_dim=config.pref_dim, **vae_args).to(device)
    try:
        vae_anchor.load_state_dict(torch.load(config.vae_ckpt, map_location=device, weights_only=True), strict=False)
        vae_anchor.eval()
        print(f'Loaded VAE anchor: {config.vae_ckpt}')
        has_vae = True
    except Exception as e:
        print(f'[WARN] VAE not available: {e}')
        has_vae = False

    x_train = multi_pref_data['x_train'].to(device=device, dtype=torch.float32)
    batch_x = x_train[0:100]  # 100 samples for statistics
    B = batch_x.shape[0]
    sample_idx = torch.arange(B, device=device)

    print()
    print('='*80)
    print('Testing Different Anchor Choices for Trajectory Student')
    print('='*80)
    print()
    print('Anchor candidates:')
    print('  1. GT[lambda=0]: Ground truth at lambda=0 (training only, not available at inference)')
    print('  2. Standard MLP: Trained on cost-only OPF (lambda=0), available at inference')
    print('  3. VAE(lambda=0): Preference-aware VAE at lambda=0, available at inference')
    print('  4. Random Gaussian: N(0, 0.1), always available')
    print()

    # Build Y_star (target trajectory)
    with torch.no_grad():
        Y_star = build_y_star_fine(
            teacher=teacher, batch_x=batch_x, sample_idx=sample_idx,
            y_stacked_gt=y_stacked_gt, gt_norm=gt_norm, fine_norm=fine_norm,
            start_idx_for_fine=start_idx_for_fine, n_va=n_va,
            teacher_steps=config.teacher_steps, wrap_each_step=config.teacher_wrap_each_step,
        )  # [B, Kf, D]

    # Define anchor generators
    def get_anchor_gt0():
        """GT[lambda=0] for all pref points"""
        y_gt0 = y_stacked_gt[0, sample_idx, :]  # [B, D]
        return y_gt0[:, None, :].expand(B, Kf, output_dim)
    
    def get_anchor_mlp():
        """Standard MLP at lambda=0 for all pref points"""
        pref0 = torch.zeros((B, 1), device=device)
        with torch.no_grad():
            y_mlp = mlp_anchor(batch_x, use_mean=True, pref=pref0)  # [B, D]
        return y_mlp[:, None, :].expand(B, Kf, output_dim)
    
    def get_anchor_vae0():
        """VAE at lambda=0 for all pref points"""
        pref0 = torch.zeros((B, 1), device=device)
        with torch.no_grad():
            y_vae = vae_anchor(batch_x, use_mean=True, pref=pref0)  # [B, D]
        return y_vae[:, None, :].expand(B, Kf, output_dim)
    
    def get_anchor_random():
        """Random Gaussian noise"""
        # Use small std to be in reasonable range
        return torch.randn((B, Kf, output_dim), device=device) * 0.1

    # Test each anchor
    anchors = {
        'GT[lambda=0]': (get_anchor_gt0, True),
        'Standard MLP': (get_anchor_mlp, has_mlp),
        'VAE(lambda=0)': (get_anchor_vae0, has_vae),
        'Random Gaussian': (get_anchor_random, True),
    }

    results = {}

    for name, (anchor_fn, available) in anchors.items():
        if not available:
            print(f'\n[SKIP] {name}: not available')
            continue
        
        print(f'\n{"="*60}')
        print(f'Anchor: {name}')
        print(f'{"="*60}')
        
        Y0 = anchor_fn()
        Y0 = wrap_angles(Y0, n_va)
        
        # Compute V_target = Y_star - Y0
        V_target = wrap_angle_difference(Y_star - Y0, n_va)
        
        # Statistics
        v_mean_all = V_target.mean().item()
        v_std_all = V_target.std().item()
        v_min = V_target.min().item()
        v_max = V_target.max().item()
        
        print(f'V_target (Y_star - Y0) statistics:')
        print(f'  Range: [{v_min:.6f}, {v_max:.6f}]')
        print(f'  Mean: {v_mean_all:.6f}')
        print(f'  Std: {v_std_all:.6f}')
        
        # Per-lambda analysis
        V_mean_per_lambda = V_target.mean(dim=0).mean(dim=1)  # [Kf]
        
        v_at_0 = V_mean_per_lambda[0].item()
        v_at_mid = V_mean_per_lambda[Kf//2].item()
        v_at_end = V_mean_per_lambda[-1].item()
        
        variation = abs(v_at_end - v_at_0)
        
        print(f'\nV_target mean at different lambdas:')
        print(f'  lambda=0:   {v_at_0:.6f}')
        print(f'  lambda=mid: {v_at_mid:.6f}')
        print(f'  lambda=1:   {v_at_end:.6f}')
        print(f'  Variation (|end - start|): {variation:.6f}')
        
        # Anchor accuracy at lambda=0
        Y0_at_0 = Y0[:, 0, :]  # Anchor at first lambda point
        Y_star_at_0 = Y_star[:, 0, :]  # Target at first lambda point (should be GT[lambda=0])
        mse_anchor_at_0 = ((Y0_at_0 - Y_star_at_0)**2).mean().item()
        
        print(f'\nAnchor accuracy at lambda=0:')
        print(f'  MSE(anchor, GT[0]): {mse_anchor_at_0:.8f}')
        
        # Sign consistency (do most dims have same sign across lambdas?)
        V_signs = torch.sign(V_mean_per_lambda)  # [Kf]
        sign_consistency = (V_signs == V_signs[0]).float().mean().item()
        
        print(f'  Sign consistency across lambdas: {sign_consistency*100:.1f}%')
        
        # Store results
        results[name] = {
            'v_std': v_std_all,
            'variation': variation,
            'mse_anchor_0': mse_anchor_at_0,
            'sign_consistency': sign_consistency,
            'available_at_inference': name != 'GT[lambda=0]',
        }

    # Summary comparison
    print()
    print('='*80)
    print('Summary Comparison')
    print('='*80)
    print()
    print(f'{"Anchor":<20} | {"V_std":>10} | {"Variation":>10} | {"MSE@0":>12} | {"Sign%":>6} | {"Inference?":>10}')
    print('-'*80)
    
    for name, r in results.items():
        inference_ok = "Yes" if r['available_at_inference'] else "NO"
        print(f'{name:<20} | {r["v_std"]:10.6f} | {r["variation"]:10.6f} | {r["mse_anchor_0"]:12.8f} | {r["sign_consistency"]*100:5.1f}% | {inference_ok:>10}')
    
    print()
    print('='*80)
    print('Recommendation')
    print('='*80)
    
    # Find best anchor that's available at inference
    inference_anchors = {k: v for k, v in results.items() if v['available_at_inference']}
    
    if inference_anchors:
        # Score: high variation + high V_std is better (easier to learn)
        # Lower MSE at 0 means anchor already close to GT at 0
        best_name = max(inference_anchors.keys(), 
                       key=lambda k: inference_anchors[k]['variation'] + inference_anchors[k]['v_std'])
        
        print(f'\nBest anchor for INFERENCE: {best_name}')
        print(f'  - V_target variation: {inference_anchors[best_name]["variation"]:.6f}')
        print(f'  - V_target std: {inference_anchors[best_name]["v_std"]:.6f}')
        print()
        print('Reasoning:')
        print('  - Higher variation = model can distinguish different lambdas')
        print('  - Higher std = stronger signal, easier to learn')
        print('  - Must be available at inference time')
    
    # Training recommendation
    print()
    print('Training Strategy:')
    print('  Option A: Train with GT[lambda=0], test with Standard MLP/VAE')
    print('            (Risk: train-test mismatch)')
    print('  Option B: Train with the SAME anchor used at inference')
    print('            (Recommended: consistency between train and test)')


if __name__ == '__main__':
    main()
