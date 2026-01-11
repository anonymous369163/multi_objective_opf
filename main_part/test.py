#!/usr/bin/env python
# coding: utf-8
"""
Multi-Preference Model Evaluation and Pareto Front Analysis

This script evaluates models trained with train_multi_preference.py:
- Simple (MLP): standard mode, NGT format output
- VAE: standard mode, NGT format output  
- Flow: Rectified Flow with preference_trajectory mode (TFM)
- Flow (Refiner): Flow with SimpleRefiner for anchor correction (3-stage training)

Also evaluates Ground Truth solutions for Pareto front comparison.

Outputs:
- Pareto front visualization with model predictions and ground truth
- Feasibility markers for each solution
- Complete metrics table (MAE, constraint satisfaction, etc.)
- Hypervolume calculation

Usage:
    python test.py                    # Evaluate all models
    python test.py --simple --vae     # Evaluate specific models only
    python test.py --gt --refiner     # Only GT vs Refiner
    python test.py --gt-only          # Evaluate ground truth only
    
Author: Peng Yue
Date: 2025-01
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt 

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'flow_model'))

from train_multi_preference import MultiPreferenceConfig as BaseMultiPrefConfig


# ==================== Configuration ====================

class TestMultiPrefConfig(BaseMultiPrefConfig):
    """Extended config for testing that supports both standard and preference_trajectory modes."""
    
    def __init__(self):
        super().__init__()
        # Post-processing: Use Jacobian method (same as Jan 5 version)
        self.use_cbf_qp_post = False
        self.post_process_method = ''
        
        # Best-of-K sampling (default disabled for deterministic predictions)
        self.flow_best_of_k = int(os.environ.get('FLOW_BEST_OF_K', '1'))
        self.vae_best_of_k = int(os.environ.get('VAE_BEST_OF_K', '1'))
        self.vae_use_mean = True
        self.flow_selection_mode = 'constraint'
        self.vae_selection_mode = 'constraint'


def get_config():
    """Get test configuration."""
    return TestMultiPrefConfig()


# Backward compatibility
MultiPreferenceConfig = TestMultiPrefConfig


# ==================== Imports ====================

from models import NetV, NetVm, NetVa
from data_loader import load_multi_preference_dataset
from unified_eval import (
    build_ctx_from_multi_preference,
    MultiPreferencePredictor,
    evaluate_unified, 
    extract_summary_metrics,
    print_metrics_table,
    save_evaluation_results,
    reconstruct_full_from_partial,
    _as_numpy,
)
from utils import get_carbon_emission_vectorized

# Import StandardMLPAnchor for flow models
from mlp_anchor import load_standard_mlp_anchor

# Import SimpleRefinerMLP for flow_refiner_v2 models
from train_multi_preference_tfm_refiner_v2 import SimpleRefinerMLP

# Import config for one-step distillation model
from train_multi_preference_refiner_flow_distill_v1 import OneStepRefinerDistillConfig


# ==================== Ground Truth Predictor ====================

class GroundTruthPredictor:
    """Predictor that returns ground truth solutions for evaluation."""
    
    def __init__(self, y_gt_ngt: torch.Tensor, multi_pref_data: dict):
        self.y_gt_ngt = y_gt_ngt
        self.multi_pref_data = multi_pref_data
    
    def predict(self, ctx):
        from unified_eval import PredPack
        y_np = _as_numpy(self.y_gt_ngt)
        Pred_Vm_full, Pred_Va_full = reconstruct_full_from_partial(ctx, y_np)
        return PredPack(
            Pred_Vm_full=Pred_Vm_full,
            Pred_Va_full=Pred_Va_full,
            time_vm=0.0, time_va=0.0, time_nn_total=0.0,
        )


# ==================== Standard MLP Predictor ====================

class StandardModelPredictor:
    """
    Predictor for standard MLP model trained with train_standard.py.
    
    This model was trained on single-objective OPF data (e.g., lc=0),
    so it produces a single solution regardless of lambda_carbon.
    """
    
    def __init__(self, model_vm, model_va, config, sys_data, multi_pref_data):
        self.model_vm = model_vm
        self.model_va = model_va
        self.config = config
        self.sys_data = sys_data
        self.multi_pref_data = multi_pref_data
    
    def predict(self, ctx):
        from unified_eval import PredPack, _insert_slack_va
        from utils import get_clamp
        import time
        
        device = ctx.device
        x_val = self.multi_pref_data['x_val'].to(device)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        with torch.no_grad():
            yvm_hat = self.model_vm(x_val)
            yva_hat = self.model_va(x_val)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        time_nn = time.perf_counter() - t0
        
        VmLb = self.sys_data.VmLb
        VmUb = self.sys_data.VmUb
        
        if isinstance(VmLb, torch.Tensor):
            VmLb = VmLb.cpu()
            VmUb = VmUb.cpu()
        else:
            VmLb = torch.from_numpy(VmLb).float()
            VmUb = torch.from_numpy(VmUb).float()
        
        yvm_hat_cpu = yvm_hat.cpu()
        yva_hat_cpu = yva_hat.cpu()
        
        scale_vm = self.config.scale_vm.cpu() if isinstance(self.config.scale_vm, torch.Tensor) else self.config.scale_vm
        scale_va = self.config.scale_va.cpu() if isinstance(self.config.scale_va, torch.Tensor) else self.config.scale_va
        
        yvm_physical = yvm_hat_cpu / scale_vm * (VmUb - VmLb) + VmLb
        yva_physical = yva_hat_cpu / scale_va
        
        hisVm_min = self.sys_data.hisVm_min
        hisVm_max = self.sys_data.hisVm_max
        if isinstance(hisVm_min, np.ndarray):
            hisVm_min = torch.from_numpy(hisVm_min).float()
            hisVm_max = torch.from_numpy(hisVm_max).float()
        
        Pred_Vm_full = get_clamp(yvm_physical, hisVm_min, hisVm_max).numpy()
        Pred_Va_full = _insert_slack_va(yva_physical.numpy(), ctx.bus_slack)
        
        return PredPack(
            Pred_Vm_full=Pred_Vm_full,
            Pred_Va_full=Pred_Va_full,
            time_vm=time_nn / 2, time_va=time_nn / 2, time_nn_total=time_nn,
        )


# ==================== Model Loading ====================

def load_standard_model(config, sys_data, device, multi_pref_data=None):
    """Load standard MLP model trained with train_standard.py."""
    from models import NetVm, NetVa
    
    if multi_pref_data is not None:
        input_dim = multi_pref_data['input_dim']
    elif hasattr(sys_data, 'num_pd') and hasattr(sys_data, 'num_qd'):
        input_dim = sys_data.num_pd + sys_data.num_qd
    elif hasattr(sys_data, 'bus_Pd') and hasattr(sys_data, 'bus_Qd'):
        input_dim = len(sys_data.bus_Pd) + len(sys_data.bus_Qd)
    else:
        raise ValueError("Cannot determine input dimension from sys_data or multi_pref_data")
    
    output_vm = config.Nbus
    output_va = config.Nbus - 1
    
    if config.Nbus == 118:
        khidden_Vm = np.array([8, 4, 2], dtype=int)
        khidden_Va = np.array([8, 4, 2], dtype=int)
    elif config.Nbus == 300:
        khidden_Vm = np.array([8, 6, 4, 2], dtype=int)
        khidden_Va = np.array([8, 6, 4, 2], dtype=int)
    else:
        khidden_Vm = np.array([8, 4, 2], dtype=int)
        khidden_Va = np.array([8, 4, 2], dtype=int)
    
    hidden_units = 128 if config.Nbus >= 100 else (64 if config.Nbus > 30 else 16)
    
    model_vm = NetVm(input_dim, output_vm, hidden_units, khidden_Vm)
    model_va = NetVa(input_dim, output_va, hidden_units, khidden_Va)
    
    nmLm = 'Lm' + ''.join(str(k) for k in khidden_Vm)
    nmLa = 'La' + ''.join(str(k) for k in khidden_Va)
    
    vm_path = os.path.join(config.model_save_dir, f"modelvm{config.Nbus}r{config.sys_R}N{config.model_version}{nmLm}E1000_simple.pth")
    va_path = os.path.join(config.model_save_dir, f"modelva{config.Nbus}r{config.sys_R}N{config.model_version}{nmLa}E1000_simple.pth")
    
    missing = []
    if not os.path.exists(vm_path):
        missing.append(f"Vm: {vm_path}")
    if not os.path.exists(va_path):
        missing.append(f"Va: {va_path}")
    
    if missing:
        raise FileNotFoundError(f"Standard model files not found:\n  " + "\n  ".join(missing) +
                               f"\n\nPlease train with: DEBUG=0 MODEL_TYPE=simple python main_part/train_standard.py")
    
    model_vm.load_state_dict(torch.load(vm_path, map_location=device, weights_only=True))
    model_va.load_state_dict(torch.load(va_path, map_location=device, weights_only=True))
    
    model_vm.to(device).eval()
    model_va.to(device).eval()
    
    print(f"  Model: Standard MLP (NetVm + NetVa)")
    print(f"  Input: {input_dim}, Output Vm: {output_vm}, Output Va: {output_va}")
    print(f"  Loaded Vm: {vm_path}")
    print(f"  Loaded Va: {va_path}")
    print(f"  Parameters: Vm={sum(p.numel() for p in model_vm.parameters()):,}, Va={sum(p.numel() for p in model_va.parameters()):,}")
    
    return model_vm, model_va


def load_model(config, model_type, multi_pref_data, device, use_tfm=False, sys_data=None):
    """
    Load a trained model.
    
    Args:
        config: Configuration object
        model_type: 'simple', 'vae', 'flow', or 'flow_refiner_v2'
        multi_pref_data: Multi-preference data dict
        device: torch device
        use_tfm: For flow models, whether to load TFM-trained variant
        sys_data: Power system data (required for flow models to load Standard MLP anchor)
    
    Returns:
        model: Loaded model
        pretrain_model: Pretrained anchor model (Standard MLP for flow models)
    """
    from net_utiles import FM, VAE
    
    input_dim = multi_pref_data['input_dim']
    output_dim = multi_pref_data['output_dim']
    pref_dim = config.pref_dim
    Vscale = multi_pref_data['Vscale']
    Vbias = multi_pref_data['Vbias']
    
    pretrain_model = None
    
    if model_type == 'simple':
        model = NetV(
            input_dim + pref_dim, output_dim,
            config.ngt_hidden_units, config.ngt_khidden,
            Vscale, Vbias
        )
        model_path = os.path.join(config.model_save_dir, "model_multi_pref_simple_final.pth")
        print(f"  Model: NetV (MLP with sigmoid scaling)")
        print(f"  Input: {input_dim} + {pref_dim} (pref) = {input_dim + pref_dim}, Output: {output_dim}")
        
    elif model_type == 'vae':
        vae_args = dict(
            output_dim=output_dim, hidden_dim=config.hidden_dim,
            num_layers=config.num_layers, latent_dim=config.latent_dim,
            output_act=None, pred_type='node', use_cvae=True
        )
        if config.vae_use_preference_aware:
            model = VAE(network='preference_aware_mlp', input_dim=input_dim, pref_dim=pref_dim, **vae_args)
            print(f"  Model: VAE (preference_aware_mlp with FiLM conditioning)")
        else:
            model = VAE(network='mlp', input_dim=input_dim + pref_dim, **vae_args)
            print(f"  Model: VAE (MLP with concatenated preference)")
        print(f"  Latent dim: {config.latent_dim}")
        model_path = os.path.join(config.model_save_dir, "model_multi_pref_vae_final.pth")
        
    elif model_type == 'flow':
        model = FM(
            network='preference_aware_mlp', input_dim=input_dim, output_dim=output_dim,
            hidden_dim=config.hidden_dim, num_layers=config.num_layers,
            time_step=config.time_step, output_norm=False, pred_type='velocity', pref_dim=pref_dim
        )
        
        use_cbf = getattr(config, 'multi_pref_use_cbf_qp_train', False)
        cbf_tag = f"cbf{getattr(config, 'multi_pref_cbf_beta', 0.5):.1f}".replace('.', '') if use_cbf else "nocbf"
        tfm_tag = "tfm_" if use_tfm else ""
        model_path = os.path.join(config.model_save_dir, f"model_multi_pref_rectified_traj_{tfm_tag}{cbf_tag}_final.pth")
        
        if not os.path.exists(model_path) and use_tfm:
            alt_path = os.path.join(config.model_save_dir, "model_multi_pref_rectified_traj_tfm_final.pth")
            if os.path.exists(alt_path):
                model_path = alt_path
        
        if not os.path.exists(model_path) and not use_tfm:
            alt_path = os.path.join(config.model_save_dir, "model_multi_pref_rectified_traj_cbf05_final.pth")
            if os.path.exists(alt_path):
                model_path = alt_path
        
        variant = "TFM" if use_tfm else "Standard"
        print(f"  Model: Flow Matching ({variant})")
        
        try:
            pretrain_model = load_standard_mlp_anchor(config, sys_data, multi_pref_data, device)
            print(f"  Using Standard MLP as anchor (predicts lc=0 cost-optimal solution)")
        except FileNotFoundError as e:
            print(f"  [WARNING] Standard MLP anchor not found: {e}")
            pretrain_model = None
    
    elif model_type == 'flow_refiner_v2':
        # Flow model trained with SimpleRefiner V2 (train_multi_preference_tfm_refiner_v2.py)
        # SimpleRefiner only predicts dx (no L), starts from λ=0
        model = FM(
            network='preference_aware_mlp', input_dim=input_dim, output_dim=output_dim,
            hidden_dim=config.hidden_dim, num_layers=config.num_layers,
            time_step=config.time_step, output_norm=False, pred_type='velocity', pref_dim=pref_dim
        )
        
        model_path = os.path.join(config.model_save_dir, "model_multi_pref_refiner_v2_flow_final.pth")
        print(f"  Model: Flow Matching (Refiner V2 - Simplified)")
        
        # Load Standard MLP as anchor generator (REQUIRED for Refiner V2)
        # Unlike Refiner V1, V2 requires a valid anchor because SimpleRefiner needs it as input
        pretrain_model = load_standard_mlp_anchor(config, sys_data, multi_pref_data, device)
        print(f"  Using Standard MLP as anchor (predicts lc=0 cost-optimal solution)")
        
        # Load SimpleRefiner MLP (REQUIRED for Refiner V2 mode)
        refiner_path = os.path.join(config.model_save_dir, "model_multi_pref_refiner_v2_mlp_final.pth")
        if os.path.exists(refiner_path):
            refiner_hidden_dim = getattr(config, 'refiner_hidden_dim', 128)
            refiner_num_layers = getattr(config, 'refiner_num_layers', 2)
            
            simple_refiner = SimpleRefinerMLP(
                scene_dim=input_dim,
                anchor_dim=output_dim,
                hidden_dim=refiner_hidden_dim,
                num_layers=refiner_num_layers,
            )
            simple_refiner.load_state_dict(torch.load(refiner_path, map_location=device, weights_only=True))
            simple_refiner.to(device).eval()
            print(f"  Loaded SimpleRefiner: {refiner_path}")
            print(f"    SimpleRefiner params: {sum(p.numel() for p in simple_refiner.parameters()):,}")
            
            # Attach simple_refiner to pretrain_model
            pretrain_model._simple_refiner = simple_refiner
        else:
            raise FileNotFoundError(f"SimpleRefiner model not found: {refiner_path}\n"
                                   f"Please train with: python main_part/train_multi_preference_tfm_refiner_v2.py")
    
    elif model_type == 'flow_onestep':
        # One-step distilled student model (train_multi_preference_refiner_flow_distill_v1.py)
        # Student is an FM model that directly predicts: x_hat = x_anchor + v(scene, x_anchor, t=0, λ)
        # Use distill config to get correct Student architecture (256 hidden, 3 layers)
        distill_config = OneStepRefinerDistillConfig()
        n_va = multi_pref_data['NPred_Va']
        
        # Network type selection via environment variable (must match training)
        # Options: "hyper_last_layer_mlp", "scale_aware_preference_mlp", "preference_aware_mlp"
        student_network = os.environ.get('STUDENT_NETWORK', 'hyper_last_layer_mlp')
        hyper_rank = int(os.environ.get('HYPER_RANK', '16'))
        hyper_use_time = os.environ.get('HYPER_USE_TIME', 'False').lower() == 'true'
        hyper_use_scene = os.environ.get('HYPER_USE_SCENE', 'True').lower() == 'true'
        hyper_scene_dim = int(os.environ.get('HYPER_SCENE_DIM', '32'))
        print(f"  [Student Network] Using: {student_network}")
        if student_network == 'hyper_last_layer_mlp':
            print(f"    HyperNet config: rank={hyper_rank}, use_time={hyper_use_time}, use_scene={hyper_use_scene}, scene_dim={hyper_scene_dim}")
        
        model = FM(
            network=student_network, input_dim=input_dim, output_dim=output_dim,
            hidden_dim=distill_config.hidden_dim, num_layers=distill_config.num_layers,
            time_step=config.time_step, output_norm=False, pred_type='velocity', pref_dim=pref_dim,
            n_va=n_va  # Used by scale_aware_preference_mlp, ignored by others
        )
        
        # Model path: model_multi_pref_refiner_flow_onestep_{tag}_{ckpt_tag}.pth
        # DISTILL_TAG: training tag (default: distill_v1)
        # ONESTEP_CKPT_TAG: checkpoint tag (default: final, can be E200, E500, E700, etc.)
        distill_tag = os.environ.get('DISTILL_TAG', 'distill_v1')
        ckpt_tag = os.environ.get('ONESTEP_CKPT_TAG', 'final')
        model_path = os.path.join(config.model_save_dir, f"model_multi_pref_refiner_flow_onestep_{distill_tag}_{ckpt_tag}.pth")
        print(f"  Model: One-Step Distilled Flow (tag={distill_tag}, ckpt={ckpt_tag})")
        print(f"  Student arch: hidden_dim={distill_config.hidden_dim}, num_layers={distill_config.num_layers}")
        
        # Load Standard MLP as anchor generator (REQUIRED)
        pretrain_model = load_standard_mlp_anchor(config, sys_data, multi_pref_data, device)
        print(f"  Using Standard MLP as anchor")
        
        # Mark this model for one-step inference mode
        pretrain_model._onestep_student = True
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()
    
    print(f"  Loaded: {model_path}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, pretrain_model


# ==================== Evaluation Functions ====================

def evaluate_model(config, model, multi_pref_data, sys_data, BRANFT, device,
                   model_type, lambdas, pretrain_model=None, use_tfm=False,
                   verbose=False, ngt_loss_fn=None, use_gt_anchor=False,
                   compare_pre_post=False):
    """
    Evaluate a model across multiple lambda values.
    
    Args:
        config: Configuration object
        model: Trained model
        multi_pref_data: Multi-preference data dict
        sys_data: Power system data
        BRANFT: Branch from-to indices
        device: torch device
        model_type: 'simple', 'vae', 'flow', or 'flow_refiner_v2'
        lambdas: List of lambda_carbon values to evaluate
        pretrain_model: Pretrained anchor model (for flow), may contain _simple_refiner attribute
        use_tfm: Whether this is a TFM-trained flow model
        verbose: Print detailed evaluation info
        ngt_loss_fn: NGT loss function for Best-of-K selection
        use_gt_anchor: Use ground truth as initial anchor (ablation)
        compare_pre_post: If True, for flow models also evaluate pre-post-processing results
    
    Returns:
        List of result dicts for each lambda
    """
    results = []
    
    # Determine internal model type and category for results
    if model_type in ['flow', 'flow_refiner_v2', 'flow_onestep']:
        internal_type = 'rectified'
    else:
        internal_type = model_type
    
    # Determine category for grouping in results
    if model_type == 'flow' and use_tfm:
        category = 'flow_tfm'
    elif model_type == 'flow_refiner_v2':
        category = 'flow_refiner_v2'
    elif model_type == 'flow_onestep':
        category = 'flow_onestep'
    else:
        category = model_type
    
    # flow_refiner_v2 and flow_onestep use simple refiner mode (no virtual segment)
    use_virtual_segment = False
    
    # For flow_refiner_v2 and flow_onestep, we don't use GT anchor
    if model_type in ['flow_refiner_v2', 'flow_onestep']:
        use_gt_anchor = False
    
    training_mode = 'preference_trajectory' if model_type in ['flow', 'flow_refiner_v2', 'flow_onestep'] else 'standard'
    
    # Get Best-of-K parameters
    flow_best_of_k = getattr(config, 'flow_best_of_k', 1)
    vae_best_of_k = getattr(config, 'vae_best_of_k', 1)
    vae_use_mean = getattr(config, 'vae_use_mean', True)
    flow_selection_mode = getattr(config, 'flow_selection_mode', 'constraint')
    vae_selection_mode = getattr(config, 'vae_selection_mode', 'constraint')
    
    is_flow = model_type in ['flow', 'flow_refiner_v2']
    is_vae = model_type == 'vae'
    use_best_of_k = (is_flow and flow_best_of_k > 1) or (is_vae and vae_best_of_k > 1 and not vae_use_mean)
    
    if use_best_of_k:
        k = flow_best_of_k if is_flow else vae_best_of_k
        mode = flow_selection_mode if is_flow else vae_selection_mode
        print(f"  [Best-of-K] K={k}, selection='{mode}'")
    else:
        if is_vae:
            print(f"  [Deterministic] VAE using mean prediction")
        elif is_flow:
            print(f"  [Deterministic] Flow using single trajectory")
    
    if use_gt_anchor and is_flow:
        print(f"  [GT Anchor] Using ground truth at λ_min as initial anchor")
    
    for lc in lambdas:
        print(f"\n  λ_carbon = {lc:.2f}")
        
        ctx = build_ctx_from_multi_preference(
            config, sys_data, multi_pref_data, BRANFT, device, lambda_carbon=lc
        )
        
        predictor = MultiPreferencePredictor(
            model=model,
            multi_pref_data=multi_pref_data,
            lambda_carbon=lc,
            model_type=internal_type,
            num_flow_steps=config.multi_pref_flow_steps,
            training_mode=training_mode,
            pretrain_model=pretrain_model,
            ngt_loss_fn=ngt_loss_fn,
            use_gt_anchor=use_gt_anchor,
            use_virtual_segment=use_virtual_segment,
        )
        
        eval_result = evaluate_unified(ctx, predictor, apply_post_processing=True, verbose=True)
        
        lc_max = max(multi_pref_data['lambda_carbon_values'])
        lambda_cost = 1.0 - (lc / lc_max) if lc_max > 0 else 1.0
        
        name_suffix = "_TFM" if use_tfm else ""
        name = f"{model_type.upper()}{name_suffix}_lc{lc:.0f}"
        
        # Post-processed results (default)
        summary = extract_summary_metrics(
            eval_result, name,
            category=category,
            lambda_cost=lambda_cost,
            use_post_processed=True
        )
        summary['lambda_carbon'] = lc
        summary['training_mode'] = training_mode
        summary['use_tfm'] = use_tfm
        results.append(summary)
        
        print(f"    [After Post-Processing] Cost: {summary['cost_mean']:.2f}, Carbon: {summary['carbon_mean']:.4f}")
        print(f"    Pg: {summary['Pg_satisfy']:.1f}%, Qg: {summary['Qg_satisfy']:.1f}%")
        
        mre_Pd = summary.get('mre_Pd_expected', 100.0)
        p_mismatch = summary.get('p_mismatch_mean', 0.0)
        if mre_Pd < 99.0 or p_mismatch > 0.01:
            print(f"    [WARN] Load Sat={mre_Pd:.2f}%, P_mismatch={p_mismatch:.6f}")
        
        # Pre-post-processing results (for comparison, especially for flow models)
        if compare_pre_post and model_type == 'flow':
            name_raw = f"{model_type.upper()}{name_suffix}_raw_lc{lc:.0f}"
            category_raw = category + '_raw'
            
            summary_raw = extract_summary_metrics(
                eval_result, name_raw,
                category=category_raw,
                lambda_cost=lambda_cost,
                use_post_processed=False
            )
            summary_raw['lambda_carbon'] = lc
            summary_raw['training_mode'] = training_mode
            summary_raw['use_tfm'] = use_tfm
            summary_raw['is_pre_post'] = True
            results.append(summary_raw)
            
            print(f"    [Before Post-Processing] Cost: {summary_raw['cost_mean']:.2f}, Carbon: {summary_raw['carbon_mean']:.4f}")
            print(f"    Pg: {summary_raw['Pg_satisfy']:.1f}%, Qg: {summary_raw['Qg_satisfy']:.1f}%")
            
            mre_Pd_raw = summary_raw.get('mre_Pd_expected', 100.0)
            if mre_Pd_raw < 99.0:
                print(f"    [WARN] Load Sat={mre_Pd_raw:.2f}%")
    
    return results


def evaluate_ground_truth(config, multi_pref_data, sys_data, BRANFT, device, lambdas, verbose=False):
    """Evaluate ground truth solutions."""
    results = []
    y_val_by_pref = multi_pref_data['y_val_by_pref']
    
    for lc in lambdas:
        if lc not in y_val_by_pref:
            print(f"  [SKIP] λ_carbon={lc:.2f} not in validation set")
            continue
        
        print(f"\n  λ_carbon = {lc:.2f}")
        
        ctx = build_ctx_from_multi_preference(
            config, sys_data, multi_pref_data, BRANFT, device, lambda_carbon=lc
        )
        
        y_gt = y_val_by_pref[lc].to(device)
        predictor = GroundTruthPredictor(y_gt, multi_pref_data)
        
        eval_result = evaluate_unified(ctx, predictor, apply_post_processing=False, verbose=verbose)
        
        lc_max = max(multi_pref_data['lambda_carbon_values'])
        lambda_cost = 1.0 - (lc / lc_max) if lc_max > 0 else 1.0
        
        summary = extract_summary_metrics(
            eval_result, f"GT_lc{lc:.0f}",
            category='ground_truth',
            lambda_cost=lambda_cost,
            use_post_processed=False
        )
        summary['lambda_carbon'] = lc
        results.append(summary)
        
        print(f"    Cost: {summary['cost_mean']:.2f}, Carbon: {summary['carbon_mean']:.4f}")
        print(f"    Pg: {summary['Pg_satisfy']:.1f}%, Qg: {summary['Qg_satisfy']:.1f}%")
        mre_Pd = summary.get('mre_Pd_expected', 100.0)
        print(f"    Load Satisfaction: {mre_Pd:.2f}%")
    
    return results


def evaluate_standard_model(config, model_vm, model_va, multi_pref_data, sys_data, BRANFT, device, verbose=False):
    """Evaluate standard MLP model trained with train_standard.py."""
    print(f"\n  Evaluating standard model (trained on lc=0 data)...")
    
    lc = 0.0
    ctx = build_ctx_from_multi_preference(
        config, sys_data, multi_pref_data, BRANFT, device, lambda_carbon=lc
    )
    
    predictor = StandardModelPredictor(model_vm, model_va, config, sys_data, multi_pref_data)
    eval_result = evaluate_unified(ctx, predictor, apply_post_processing=True, verbose=verbose)
    
    lc_max = max(multi_pref_data['lambda_carbon_values'])
    lambda_cost = 1.0
    
    summary = extract_summary_metrics(
        eval_result, "Standard_MLP",
        category='standard',
        lambda_cost=lambda_cost,
        use_post_processed=True
    )
    summary['lambda_carbon'] = lc
    summary['training_mode'] = 'standard_supervised'
    summary['note'] = 'Trained on single-objective OPF (lc=0)'
    
    print(f"\n  Standard MLP Results:")
    print(f"    Cost: {summary['cost_mean']:.2f}, Carbon: {summary['carbon_mean']:.4f}")
    print(f"    Pg: {summary['Pg_satisfy']:.1f}%, Qg: {summary['Qg_satisfy']:.1f}%")
    mre_Pd = summary.get('mre_Pd_expected', 100.0)
    print(f"    Load Satisfaction: {mre_Pd:.2f}%")
    
    return [summary]


# ==================== Feasibility Check ====================

def compute_feasibility(result, thresholds=None):
    """Compute feasibility for a result."""
    if thresholds is None:
        thresholds = {'Pg': 99.0, 'Qg': 99.0, 'branch': 99.0, 'load': 99.0}
    
    pg_sat = result.get('Pg_satisfy', 100.0)
    qg_sat = result.get('Qg_satisfy', 100.0)
    branch_sat = min(result.get('branch_ang_satisfy', 100.0), result.get('branch_pf_satisfy', 100.0))
    mre_Pd = result.get('mre_Pd_expected', 100.0)
    
    is_feasible = (
        pg_sat >= thresholds['Pg'] and
        qg_sat >= thresholds['Qg'] and
        branch_sat >= thresholds['branch'] and
        mre_Pd >= thresholds['load']
    )
    
    feasibility_score = (pg_sat + qg_sat + branch_sat + mre_Pd) / 4.0
    
    return is_feasible, feasibility_score


# ==================== Visualization ====================

def plot_pareto_front(all_results, ref_point, hypervolumes, save_path):
    """Plot Pareto front with feasibility markers."""
    fig, ax = plt.subplots(figsize=(12, 9))
    
    gt_results = [r for r in all_results if r.get('category') == 'ground_truth']
    feasible_results = [r for r in all_results if compute_feasibility(r)[0]]
    limit_results = gt_results or feasible_results or all_results
    
    limit_costs = np.array([r['cost_mean'] for r in limit_results])
    limit_carbons = np.array([r['carbon_mean'] for r in limit_results])
    
    cost_range = max(limit_costs.max() - limit_costs.min(), limit_costs.mean() * 0.1)
    carbon_range = max(limit_carbons.max() - limit_carbons.min(), limit_carbons.mean() * 0.1)
    
    x_min, x_max = limit_costs.min() - cost_range * 0.1, limit_costs.max() + cost_range * 0.15
    y_min, y_max = limit_carbons.min() - carbon_range * 0.1, limit_carbons.max() + carbon_range * 0.15
    
    # Category styles
    styles = {
        'ground_truth': {'color': '#FFD700', 'marker': '*', 'size': 350, 'label': 'Ground Truth (OPF)'},
        'standard': {'color': '#FF6B35', 'marker': 'P', 'size': 300, 'label': 'Standard MLP (lc=0)'},
        'simple': {'color': '#E74C3C', 'marker': 's', 'size': 180, 'label': 'Simple MLP'},
        'vae': {'color': '#3498DB', 'marker': 'o', 'size': 180, 'label': 'VAE'},
        'flow': {'color': '#27AE60', 'marker': '^', 'size': 180, 'label': 'Flow'},
        'flow_tfm': {'color': '#9B59B6', 'marker': 'D', 'size': 180, 'label': 'Flow (TFM)'},
        'flow_tfm_raw': {'color': '#E74C3C', 'marker': 'X', 'size': 200, 'label': 'Flow (TFM, Raw)'},
        'flow_refiner_v2': {'color': '#00BCD4', 'marker': 'h', 'size': 220, 'label': 'Flow (Refiner)'},
        'flow_onestep': {'color': '#FF1493', 'marker': 'v', 'size': 220, 'label': 'Flow (One-Step)'},
    }
    
    categories = {}
    for r in all_results:
        cat = r.get('category', 'unknown')
        categories.setdefault(cat, []).append(r)
    
    for cat in ['ground_truth', 'standard', 'simple', 'vae', 'flow', 'flow_tfm', 'flow_tfm_raw', 'flow_refiner_v2', 'flow_onestep']:
        if cat not in categories:
            continue
        
        style = styles.get(cat, {'color': 'gray', 'marker': 'x', 'size': 150, 'label': cat})
        cat_results = categories[cat]
        
        costs = np.array([r['cost_mean'] for r in cat_results])
        carbons = np.array([r['carbon_mean'] for r in cat_results])
        feasible_mask = np.array([compute_feasibility(r)[0] for r in cat_results])
        
        # For raw (pre-post-processing) results, use hollow markers
        is_raw = cat.endswith('_raw')
        edge_color = 'black' if not is_raw else style['color']
        edge_width = 1.5 if not is_raw else 2.5
        fill_color = style['color'] if not is_raw else 'white'
        alpha_val = 0.9 if not is_raw else 0.7
        
        # For raw results, always display all points (both feasible and infeasible)
        if is_raw:
            # Display all raw points with the same style (hollow markers)
            # Use higher zorder (5) to ensure they are visible above other points
            ax.scatter(
                costs, carbons,
                c='white', marker=style['marker'], s=style['size'],
                label=f"{style['label']}", 
                zorder=5,
                edgecolors=style['color'], linewidths=3.0, alpha=0.9
            )
        else:
            # For non-raw results, separate feasible and infeasible
            if np.any(feasible_mask):
                ax.scatter(
                    costs[feasible_mask], carbons[feasible_mask],
                    c=fill_color, marker=style['marker'], s=style['size'],
                    label=f"{style['label']} (feasible)", 
                    zorder=4 if cat == 'ground_truth' else 3,
                    edgecolors=edge_color, linewidths=edge_width, alpha=alpha_val
                )
            
            if np.any(~feasible_mask):
                ax.scatter(
                    costs[~feasible_mask], carbons[~feasible_mask],
                    c='white', marker=style['marker'], s=style['size'],
                    label=f"{style['label']} (infeasible)", zorder=3,
                    edgecolors=style['color'], linewidths=2.5, alpha=0.7
                )
        
        if len(costs) > 1:
            sorted_idx = np.argsort(costs)
            ax.plot(costs[sorted_idx], carbons[sorted_idx], 
                   color=style['color'], linestyle='--', alpha=0.3 if is_raw else 0.4, 
                   linewidth=1.5 if is_raw else 2, zorder=1 if is_raw else 2)
        
        if cat == 'ground_truth':
            for r, cost, carbon in zip(cat_results, costs, carbons):
                ax.annotate(f"λ={r.get('lambda_carbon', 0):.0f}", (cost, carbon),
                           textcoords="offset points", xytext=(8, 8), fontsize=8, alpha=0.7)
    
    if ref_point[0] <= x_max and ref_point[1] <= y_max:
        ax.scatter(ref_point[0], ref_point[1], c='gray', marker='X', s=200,
                  label='Reference Point', zorder=2, edgecolors='black', linewidths=1.5)
    
    ax.set_xlabel('Economic Cost ($/h)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Carbon Emission (tCO2/h)', fontsize=13, fontweight='bold')
    ax.set_title('Pareto Front: Multi-Preference Models vs Ground Truth\n(Filled = Feasible, Hollow = Infeasible)',
                fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    hv_text = "Hypervolumes:\n"
    for cat in ['ground_truth', 'standard', 'simple', 'vae', 'flow', 'flow_tfm', 'flow_tfm_raw', 'flow_refiner_v2', 'flow_onestep', 'all']:
        if cat in hypervolumes:
            label = styles.get(cat, {}).get('label', cat)
            hv_text += f"  {label}: {hypervolumes[cat]:.2f}\n"
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, hv_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=props, fontfamily='monospace')
    
    feas_text = "Feasibility:\n  Pg,Qg,Branch ≥ 99%\n  Vm ≥ 95%\n  Load ≥ 99%"
    ax.text(0.02, 0.68, feas_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=props, fontfamily='monospace')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPareto front saved to: {save_path}")
    plt.close()


def print_comparison_table(all_results):
    """Print comparison table."""
    print("\n" + "=" * 120)
    print(" Comparison Table: Cost vs Carbon vs Feasibility")
    print("=" * 120)
    
    header = f"{'Model':<22} {'Category':<12} {'λc':<5} {'Cost':<11} {'Carbon':<9} {'Pg%':<6} {'Qg%':<6} {'Load%':<7} {'Feasible':<8}"
    print(header)
    print("-" * 120)
    
    for r in sorted(all_results, key=lambda x: (x.get('category', 'z'), x.get('lambda_carbon', 0))):
        is_feas, _ = compute_feasibility(r)
        lc = r.get('lambda_carbon', 0)
        mre_Pd = r.get('mre_Pd_expected', 100.0)
        
        print(f"{r['name']:<22} {r.get('category', '?'):<12} {lc:<5.0f} "
              f"{r['cost_mean']:<11.2f} {r['carbon_mean']:<9.4f} "
              f"{r['Pg_satisfy']:<6.1f} {r['Qg_satisfy']:<6.1f} "
              f"{mre_Pd:<7.2f} {'Yes' if is_feas else 'No':<8}")
    
    print("-" * 120)


def compute_hypervolumes(all_results, ref_point):
    """Compute hypervolumes for each category."""
    hypervolumes = {}
    
    for cat in ['ground_truth', 'standard', 'simple', 'vae', 'flow', 'flow_tfm', 'flow_tfm_raw', 'flow_refiner_v2', 'flow_onestep']:
        cat_results = [r for r in all_results if r.get('category') == cat]
        if not cat_results:
            continue
        
        costs = np.array([r['cost_mean'] for r in cat_results])
        carbons = np.array([r['carbon_mean'] for r in cat_results])
        
        points = np.column_stack([costs, carbons])
        points = points[np.argsort(points[:, 0])]
        
        hv = 0.0
        prev_carbon = ref_point[1]
        for cost, carbon in points:
            if carbon < prev_carbon:
                hv += (ref_point[0] - cost) * (prev_carbon - carbon)
                prev_carbon = carbon
        
        hypervolumes[cat] = hv
    
    costs = np.array([r['cost_mean'] for r in all_results])
    carbons = np.array([r['carbon_mean'] for r in all_results])
    points = np.column_stack([costs, carbons])
    points = points[np.argsort(points[:, 0])]
    
    hv = 0.0
    prev_carbon = ref_point[1]
    for cost, carbon in points:
        if carbon < prev_carbon:
            hv += (ref_point[0] - cost) * (prev_carbon - carbon)
            prev_carbon = carbon
    hypervolumes['all'] = hv
    
    return hypervolumes


# ==================== Main ====================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Multi-Preference Model Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test.py                      # Evaluate all models (GT + Standard + Simple + VAE + Flow + Refiner)
    python test.py --gt --flow-refiner-v2  # Only GT vs Flow Refiner
    python test.py --flow --flow-refiner-v2  # Compare Flow vs Flow Refiner
    python test.py --simple --vae       # Evaluate Simple and VAE only
    python test.py --flow               # Evaluate Flow model (TFM default)
    python test.py --flow --compare-pre-post  # Flow model with pre/post-processing comparison
    python test.py --gt-only            # Evaluate ground truth only
    python test.py --lambdas 0,10,20,30 # Custom lambda values
    python test.py --dense              # Dense lambda grid (step=2.5)
        """
    )
    
    # Model selection
    parser.add_argument('--standard', action='store_true', help='Evaluate Standard MLP model')
    parser.add_argument('--simple', action='store_true', help='Evaluate Simple (MLP) model')
    parser.add_argument('--vae', action='store_true', help='Evaluate VAE model')
    parser.add_argument('--flow', action='store_true', help='Evaluate Flow model')
    parser.add_argument('--flow-refiner-v2', '--refiner', action='store_true', help='Evaluate Flow with SimpleRefiner (3-stage)')
    parser.add_argument('--flow-onestep', '--onestep', action='store_true', help='Evaluate One-Step Distilled Flow')
    parser.add_argument('--use-tfm', '--tfm', action='store_true', default=True, help='Use TFM-trained variant')
    parser.add_argument('--compare-pre-post', action='store_true', help='Compare pre/post-processing results for flow models')
    parser.add_argument('--gt', '--ground-truth', action='store_true', dest='gt', help='Evaluate Ground Truth')
    parser.add_argument('--gt-only', action='store_true', help='Evaluate Ground Truth only')
    parser.add_argument('--standard-only', action='store_true', help='Evaluate Standard MLP only')
    parser.add_argument('--all', '-a', action='store_true', help='Evaluate all models')
    
    # Lambda settings
    parser.add_argument('--lambdas', type=str, default=None, help='Comma-separated lambda values')
    parser.add_argument('--lambdas-sparse', action='store_true', help='Use sparse lambda grid')
    parser.add_argument('--lambdas-dense', '--dense', action='store_true', default=True, help='Use dense lambda grid')
    parser.add_argument('--lambda-step', type=float, default=None, help='Custom lambda step size')
    
    # Other settings
    parser.add_argument('--best-of-k', '-k', type=int, default=1, help='Best-of-K sampling')
    parser.add_argument('--flow-steps', type=int, default=100, help='Number of ODE integration steps')
    parser.add_argument('--gt-anchor', action='store_true', default=True, help='Use GT as initial anchor for Flow')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Default: evaluate all models (including flow_refiner_v2 and flow_onestep for comparison)
    if not (args.standard or args.simple or args.vae or args.flow or args.flow_refiner_v2 or args.flow_onestep or args.gt or args.gt_only or args.standard_only):
        args.all = True
    
    if args.all:
        args.standard = args.simple = args.vae = args.flow = args.flow_refiner_v2 = args.flow_onestep = args.gt = True
    
    if args.gt_only:
        args.standard = args.simple = args.vae = args.flow = False
        args.gt = True
    
    if args.standard_only:
        args.simple = args.vae = args.flow = False
        args.standard = args.gt = True
    
    # Parse lambda values
    if args.lambdas:
        args.lambda_values = [float(x) for x in args.lambdas.split(',')]
    elif args.lambdas_sparse:
        args.lambda_values = [0, 10, 20, 30, 40, 50]
    elif args.lambdas_dense or args.lambda_step:
        step = args.lambda_step if args.lambda_step else 2.5
        args.lambda_values = [round(x * step, 2) for x in range(int(50 / step) + 1)]
    else:
        args.lambda_values = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    return args


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("=" * 80)
    print(" Multi-Preference Model Evaluation & Pareto Front Analysis")
    print("=" * 80)
    
    config = get_config()
    device = config.device
    
    if args.best_of_k > 1:
        config.flow_best_of_k = args.best_of_k
        config.vae_best_of_k = args.best_of_k
        config.vae_use_mean = False
    
    config.multi_pref_flow_steps = args.flow_steps
    
    print(f"\nConfiguration:")
    print(f"  Nbus: {config.Nbus}, Device: {device}")
    print(f"  Model directory: {config.model_save_dir}")
    print(f"  Lambda values: {args.lambda_values}")
    print(f"  Flow ODE steps: {config.multi_pref_flow_steps}")
    
    print("\nLoading dataset...")
    multi_pref_data, sys_data = load_multi_preference_dataset(config)
    BRANFT = torch.from_numpy(sys_data.branch[:, 0:2] - 1).long()
    
    ngt_loss_fn = None
    if config.flow_best_of_k > 1 or (config.vae_best_of_k > 1 and not config.vae_use_mean):
        try:
            from deepopf_ngt_loss import DeepOPFNGTLoss
            ngt_loss_fn = DeepOPFNGTLoss(sys_data, config)
            ngt_loss_fn.cache_to_gpu(device)
        except Exception as e:
            print(f"[WARNING] Best-of-K disabled: {e}")
    
    available_gt = multi_pref_data['lambda_carbon_values']
    lambdas = args.lambda_values
    lambdas_gt = [lc for lc in args.lambda_values if lc in available_gt]
    
    lc_max = max(available_gt)
    lambdas = [lc for lc in lambdas if 0 <= lc <= lc_max]
    
    print(f"  Evaluating λ_carbon: {lambdas}")
    print(f"  GT available at: {lambdas_gt}")
    
    all_results = []
    section = 0
    
    # 1. Ground Truth
    if args.gt and lambdas_gt:
        section += 1
        print(f"\n{'='*60}\n {section}. Ground Truth (OPF Solutions)\n{'='*60}")
        gt_results = evaluate_ground_truth(config, multi_pref_data, sys_data, BRANFT, device, lambdas_gt, args.verbose)
        all_results.extend(gt_results)
        print(f"\n  Evaluated {len(gt_results)} GT solutions")
    elif args.gt:
        print(f"\n[SKIP] No GT data available for requested lambdas")
    
    # 2. Standard MLP
    if args.standard:
        section += 1
        print(f"\n{'='*60}\n {section}. Standard MLP Model\n{'='*60}")
        try:
            model_vm, model_va = load_standard_model(config, sys_data, device, multi_pref_data)
            results = evaluate_standard_model(config, model_vm, model_va, multi_pref_data, sys_data, BRANFT, device, args.verbose)
            all_results.extend(results)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
    
    # 3. Simple MLP
    if args.simple:
        section += 1
        print(f"\n{'='*60}\n {section}. Simple (MLP) Model\n{'='*60}")
        try:
            model, _ = load_model(config, 'simple', multi_pref_data, device)
            results = evaluate_model(config, model, multi_pref_data, sys_data, BRANFT, device,
                                    'simple', lambdas, verbose=args.verbose, ngt_loss_fn=ngt_loss_fn)
            all_results.extend(results)
            print(f"\n  Evaluated {len(results)} predictions")
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
    
    # 4. VAE
    if args.vae:
        section += 1
        print(f"\n{'='*60}\n {section}. VAE Model\n{'='*60}")
        try:
            model, _ = load_model(config, 'vae', multi_pref_data, device)
            results = evaluate_model(config, model, multi_pref_data, sys_data, BRANFT, device,
                                    'vae', lambdas, verbose=args.verbose, ngt_loss_fn=ngt_loss_fn)
            all_results.extend(results)
            print(f"\n  Evaluated {len(results)} predictions")
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
    
    # 5. Flow Model (TFM)
    if args.flow:
        variant = "TFM" if args.use_tfm else "Standard"
        section += 1
        print(f"\n{'='*60}\n {section}. Flow Model ({variant})\n{'='*60}")
        try:
            model, pretrain = load_model(config, 'flow', multi_pref_data, device, use_tfm=args.use_tfm, sys_data=sys_data)
            results = evaluate_model(config, model, multi_pref_data, sys_data, BRANFT, device,
                                    'flow', lambdas, pretrain_model=pretrain, use_tfm=args.use_tfm,
                                    verbose=args.verbose, ngt_loss_fn=ngt_loss_fn, use_gt_anchor=args.gt_anchor,
                                    compare_pre_post=args.compare_pre_post)
            all_results.extend(results)
            print(f"\n  Evaluated {len(results)} predictions")
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
    
    # 6. Flow Model with SimpleRefiner
    if args.flow_refiner_v2:
        section += 1
        print(f"\n{'='*60}\n {section}. Flow Model (Refiner)\n{'='*60}")
        try:
            model, pretrain = load_model(config, 'flow_refiner_v2', multi_pref_data, device, sys_data=sys_data)
            results = evaluate_model(config, model, multi_pref_data, sys_data, BRANFT, device,
                                    'flow_refiner_v2', lambdas, pretrain_model=pretrain, use_tfm=True,
                                    verbose=args.verbose, ngt_loss_fn=ngt_loss_fn, use_gt_anchor=False)
            all_results.extend(results)
            print(f"\n  Evaluated {len(results)} predictions")
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
    
    # 7. One-Step Distilled Flow Model
    if args.flow_onestep:
        section += 1
        print(f"\n{'='*60}\n {section}. One-Step Distilled Flow Model\n{'='*60}")
        try:
            model, pretrain = load_model(config, 'flow_onestep', multi_pref_data, device, sys_data=sys_data)
            results = evaluate_model(config, model, multi_pref_data, sys_data, BRANFT, device,
                                    'flow_onestep', lambdas, pretrain_model=pretrain, use_tfm=True,
                                    verbose=args.verbose, ngt_loss_fn=ngt_loss_fn, use_gt_anchor=False)
            all_results.extend(results)
            print(f"\n  Evaluated {len(results)} predictions")
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
    
    # Results
    if not all_results:
        print("\n[ERROR] No models evaluated! Check model paths.")
        return
    
    print_comparison_table(all_results)
    print_metrics_table(all_results, "Complete Metrics")
    
    print(f"\n{'='*60}\n Pareto Front Analysis\n{'='*60}")
    
    gt_results = [r for r in all_results if r.get('category') == 'ground_truth']
    feasible = [r for r in all_results if compute_feasibility(r)[0]]
    ref_results = gt_results or feasible or all_results
    
    ref_costs = np.array([r['cost_mean'] for r in ref_results])
    ref_carbons = np.array([r['carbon_mean'] for r in ref_results])
    ref_point = np.array([ref_costs.max() * 1.05, ref_carbons.max() * 1.05])
    
    print(f"  Reference point: ({ref_point[0]:.2f}, {ref_point[1]:.4f})")
    
    hypervolumes = compute_hypervolumes(all_results, ref_point)
    for cat, hv in hypervolumes.items():
        print(f"  Hypervolume ({cat}): {hv:.2f}")
    
    plot_pareto_front(all_results, ref_point, hypervolumes,
                     f'{config.results_dir}/pareto_front_multi_preference.png')
    
    save_evaluation_results(all_results, hypervolumes, ref_point,
                           f'{config.results_dir}/multi_preference_evaluation_results.json', config=config)
    
    print(f"\n{'='*80}\n Evaluation completed!\n{'='*80}")
    
    return all_results


if __name__ == "__main__":
    main()
