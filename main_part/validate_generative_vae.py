"""
Validation Script for Generative VAE using Standard Evaluation Framework

This script validates the generative VAE using the same evaluation framework
as train_supervised.py, enabling fair comparison with MLP, VAE, and Flow models.

Key features:
1. Uses evaluate_unified() for standardized evaluation
2. Implements Best-of-K sampling strategy with different K values
3. Compares with baseline methods (MLP, standard VAE)
4. Reports constraint satisfaction, cost, carbon, and prediction accuracy

Author: Auto-generated from VAE improvement plan v5
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_part.config import get_config
from main_part.data_loader import load_multi_preference_dataset, load_ngt_training_data, load_all_data
from main_part.deepopf_ngt_loss import DeepOPFNGTLoss
from main_part.unified_eval import (
    EvalContext, PredPack, build_ctx_from_multi_preference, evaluate_unified,
    reconstruct_full_from_partial, _as_numpy, MultiPreferencePredictor
)
from flow_model.linearized_vae import LinearizedVAE
from flow_model.generative_vae_utils import make_pref_tensors, lambda_to_key


class GenerativeVAEPredictor:
    """
    Predictor for Generative VAE with Best-of-K sampling support.
    
    This predictor extends the standard VAE inference to support sampling
    multiple solutions and selecting the best one based on constraint violation.
    
    Key features:
    - K=1: Uses mean prediction (equivalent to standard VAE)
    - K>1: Samples K solutions and selects the best one
    - Selection criterion: minimize constraint violation, then objective
    """
    
    def __init__(
        self,
        model: LinearizedVAE,
        multi_pref_data: Dict[str, Any],
        lambda_carbon: float,
        ngt_loss_fn: DeepOPFNGTLoss,
        *,
        n_samples: int = 1,  # K for Best-of-K
        use_mean: bool = False,  # If True, always use mean (baseline)
        selection_mode: str = 'constraint',  # 'constraint', 'objective', or 'hybrid'
    ):
        """
        Initialize the Generative VAE predictor.
        
        Args:
            model: Trained LinearizedVAE model
            multi_pref_data: Multi-preference data dictionary
            lambda_carbon: Preference value for prediction
            ngt_loss_fn: NGT loss function for computing constraint violations
            n_samples: Number of samples for Best-of-K (K=1 means single sample)
            use_mean: If True, always use mean prediction (ignores n_samples)
            selection_mode: How to select best sample
                - 'constraint': Select lowest constraint violation
                - 'objective': Select lowest objective value (among feasible)
                - 'hybrid': Two-stage: filter feasible, then select best objective
        """
        self.model = model
        self.multi_pref_data = multi_pref_data
        self.lambda_carbon = lambda_carbon
        self.ngt_loss_fn = ngt_loss_fn
        self.n_samples = n_samples
        self.use_mean = use_mean
        self.selection_mode = selection_mode
        
        # Get normalization factor
        lambda_carbon_values = multi_pref_data.get('lambda_carbon_values', [55.0])
        self.lc_max = max(lambda_carbon_values) if max(lambda_carbon_values) > 0 else 1.0
        self.lambda_min = min(lambda_carbon_values)
        self.lambda_max_val = max(lambda_carbon_values)
        
        # Get bus indices for reconstruction
        self.bus_Pnet_all = multi_pref_data.get('bus_Pnet_all')
        self.bus_Pnet_noslack_all = multi_pref_data.get('bus_Pnet_noslack_all')
        
    def predict(self, ctx: EvalContext) -> PredPack:
        """
        Predict voltage for test samples with Best-of-K sampling.
        
        Args:
            ctx: Evaluation context with test data
        
        Returns:
            PredPack with Pred_Vm_full, Pred_Va_full, and timing info
        """
        self.model.eval()
        
        x = ctx.x_test.to(ctx.device)
        Ntest = x.shape[0]
        output_dim = self.multi_pref_data['output_dim']
        
        # Prepare preference tensors
        pref_norm, pref_raw = make_pref_tensors(
            self.lambda_carbon, self.lc_max, Ntest, ctx.device
        )
        
        # Timing
        if ctx.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        with torch.no_grad():
            if self.use_mean or self.n_samples == 1:
                # Single prediction using mean (standard VAE behavior)
                mean, logvar = self.model.encoder.encode_from_condition(x, pref_norm)
                if self.use_mean:
                    z = mean
                else:
                    # Sample once
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    z = mean + std * eps
                V_partial = self.model.decode(x, z)
            else:
                # Best-of-K sampling
                V_partial = self._best_of_k_sampling(
                    x, pref_norm, pref_raw, ctx.device
                )
        
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
    
    def _best_of_k_sampling(
        self, 
        x: torch.Tensor, 
        pref_norm: torch.Tensor,
        pref_raw: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Perform Best-of-K sampling and selection.
        
        For each test sample:
        1. Sample K solutions from the latent distribution
        2. Compute constraint violation for each
        3. Select the best solution based on selection_mode
        
        Args:
            x: [B, input_dim] scene features
            pref_norm: [B, 1] normalized preference for network
            pref_raw: [B, 2] raw preference for NGT
            device: computation device
        
        Returns:
            V_partial: [B, output_dim] best solutions
        """
        B = x.shape[0]
        K = self.n_samples
        
        # Encode to get latent distribution
        mean, logvar = self.model.encoder.encode_from_condition(x, pref_norm)  # [B, D]
        std = torch.exp(0.5 * logvar)
        D = mean.shape[1]
        
        # Sample K times for each sample
        eps = torch.randn(K, B, D, device=device)  # [K, B, D]
        z_samples = mean.unsqueeze(0) + std.unsqueeze(0) * eps  # [K, B, D]
        
        # Decode all samples
        z_flat = z_samples.reshape(K * B, D)  # [K*B, D]
        x_expanded = x.unsqueeze(0).expand(K, -1, -1).reshape(K * B, -1)  # [K*B, input_dim]
        y_flat = self.model.decode(x_expanded, z_flat)  # [K*B, output_dim]
        y_samples = y_flat.reshape(K, B, -1)  # [K, B, output_dim]
        
        # Compute constraint violations for all samples
        pref_raw_expanded = pref_raw.unsqueeze(0).expand(K, -1, -1).reshape(K * B, -1)
        x_pqd_expanded = x_expanded  # Scene is also PQd
        
        _, loss_dict = self.ngt_loss_fn(y_flat, x_pqd_expanded, pref_raw_expanded)
        
        constraint_flat = loss_dict['constraint_scaled']  # [K*B]
        objective_flat = loss_dict['objective_per_sample']  # [K*B]
        
        constraint_kb = constraint_flat.reshape(K, B)  # [K, B]
        objective_kb = objective_flat.reshape(K, B)  # [K, B]
        
        # Select best for each sample
        best_y = torch.zeros(B, y_samples.shape[-1], device=device)
        
        for b in range(B):
            cv_b = constraint_kb[:, b]  # [K]
            obj_b = objective_kb[:, b]  # [K]
            
            if self.selection_mode == 'constraint':
                # Simply select lowest constraint violation
                best_idx = cv_b.argmin()
            elif self.selection_mode == 'objective':
                # Select lowest objective
                best_idx = obj_b.argmin()
            else:  # hybrid
                # Two-stage: among feasible, select best objective
                # If no feasible, select lowest constraint
                threshold = 0.01  # Could be configurable
                feasible_mask = cv_b < threshold
                
                if feasible_mask.any():
                    # Among feasible, select best objective
                    feasible_obj = obj_b.clone()
                    feasible_obj[~feasible_mask] = float('inf')
                    best_idx = feasible_obj.argmin()
                else:
                    # No feasible, select lowest constraint
                    best_idx = cv_b.argmin()
            
            best_y[b] = y_samples[best_idx, b]
        
        return best_y


def run_evaluation_with_k(
    vae: LinearizedVAE,
    ngt_loss_fn: DeepOPFNGTLoss,
    multi_pref_data: Dict,
    sys_data,
    BRANFT,
    config,
    lambda_carbon: float,
    k_value: int,
    use_mean: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run full evaluation for a specific K value.
    
    Args:
        vae: Trained VAE model
        ngt_loss_fn: NGT loss function
        multi_pref_data: Multi-preference data
        sys_data: System data
        BRANFT: Branch data for post-processing
        config: Configuration
        lambda_carbon: Preference value
        k_value: Number of samples for Best-of-K
        use_mean: Whether to use mean-only prediction
        verbose: Whether to print results
    
    Returns:
        Evaluation results dictionary
    """
    device = config.device
    
    # Build evaluation context
    ctx = build_ctx_from_multi_preference(
        config, sys_data, multi_pref_data, BRANFT, device,
        lambda_carbon=lambda_carbon
    )
    
    # Create predictor
    predictor = GenerativeVAEPredictor(
        model=vae,
        multi_pref_data=multi_pref_data,
        lambda_carbon=lambda_carbon,
        ngt_loss_fn=ngt_loss_fn,
        n_samples=k_value,
        use_mean=use_mean,
        selection_mode='hybrid'
    )
    
    # Run evaluation
    mode_str = "mean" if use_mean else f"K={k_value}"
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating Generative VAE ({mode_str}) - lambda_carbon={lambda_carbon}")
        print(f"{'='*60}")
    
    results = evaluate_unified(ctx, predictor, apply_post_processing=True, verbose=verbose)
    results['k_value'] = k_value
    results['use_mean'] = use_mean
    results['lambda_carbon'] = lambda_carbon
    
    return results


def compare_with_baseline(
    vae: LinearizedVAE,
    ngt_loss_fn: DeepOPFNGTLoss,
    multi_pref_data: Dict,
    sys_data,
    BRANFT,
    config,
    lambda_carbon: float = 50.0,
    k_values: List[int] = [1, 4, 8, 16, 32]
) -> Dict[str, Any]:
    """
    Compare Generative VAE (various K) with mean baseline.
    
    Returns a comparison summary across different K values.
    """
    results_all = {}
    
    # Mean baseline (standard VAE)
    print("\n" + "="*80)
    print("Running Mean Baseline (Standard VAE)")
    print("="*80)
    results_mean = run_evaluation_with_k(
        vae, ngt_loss_fn, multi_pref_data, sys_data, BRANFT, config,
        lambda_carbon, k_value=1, use_mean=True, verbose=True
    )
    results_all['mean'] = results_mean
    
    # Best-of-K for each K
    for K in k_values:
        print(f"\n" + "="*80)
        print(f"Running Best-of-K (K={K})")
        print("="*80)
        results_k = run_evaluation_with_k(
            vae, ngt_loss_fn, multi_pref_data, sys_data, BRANFT, config,
            lambda_carbon, k_value=K, use_mean=False, verbose=True
        )
        results_all[f'K={K}'] = results_k
    
    return results_all


def print_comparison_summary(results_all: Dict[str, Any]):
    """Print a comparison summary table."""
    print("\n" + "="*100)
    print("COMPARISON SUMMARY")
    print("="*100)
    
    # Extract key metrics
    metrics = ['num_viotest', 'cost_mean', 'carbon_mean', 'mae_Vmtest', 'mae_Vatest']
    metric_names = ['Violated', 'Cost', 'Carbon', 'Vm MAE', 'Va MAE']
    
    # Print header
    header = f"{'Method':>15}"
    for name in metric_names:
        header += f" | {name:>12}"
    print(header)
    print("-"*100)
    
    # Print results for each method
    keys_order = ['mean'] + [k for k in results_all.keys() if k.startswith('K=')]
    keys_order = [k for k in keys_order if k in results_all]
    
    for key in keys_order:
        r = results_all[key]
        row = f"{key:>15}"
        for m in metrics:
            # Prefer post-processed values (suffix '1') over raw values
            # This ensures we report the best results after power flow correction
            val = r.get(m + '1', r.get(m, 0))  # Try post-processed first, fallback to raw
            if isinstance(val, (list, np.ndarray)):
                val = np.mean(val)
            if m == 'num_viotest':
                row += f" | {int(val):>12}"
            elif 'mae' in m.lower():
                row += f" | {val:>12.6f}"
            else:
                row += f" | {val:>12.4f}"
        print(row)
    
    print("="*100)
    
    # Improvement analysis
    if 'mean' in results_all and len(keys_order) > 1:
        mean_vio = results_all['mean'].get('num_viotest', results_all['mean'].get('num_viotest1', 0))
        best_k = keys_order[-1]  # Last K is usually largest
        best_vio = results_all[best_k].get('num_viotest', results_all[best_k].get('num_viotest1', 0))
        
        print(f"\nViolation reduction: {mean_vio} -> {best_vio} ({mean_vio - best_vio} fewer)")
        
        if mean_vio > 0:
            reduction_pct = (mean_vio - best_vio) / mean_vio * 100
            print(f"Relative reduction: {reduction_pct:.1f}%")


def load_baseline_models(config, input_dim, output_dim, pref_dim=1):
    """
    Load baseline models for comparison (MLP and standard VAE).
    
    Returns dict with loaded models or None if not found.
    """
    baseline_models = {}
    
    # Try to load multi-preference VAE
    vae_path = f'{config.model_save_dir}/model_multi_pref_vae_final.pth'
    if os.path.exists(vae_path):
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'flow_model'))
            from net_utiles import VAE
            
            use_pref_aware = getattr(config, 'vae_use_preference_aware', True)
            if use_pref_aware:
                baseline_vae = VAE(
                    network='preference_aware_mlp',
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    latent_dim=config.latent_dim,
                    output_act=None,
                    pred_type='node',
                    use_cvae=True,
                    pref_dim=pref_dim
                )
            else:
                baseline_vae = VAE(
                    network='mlp',
                    input_dim=input_dim + pref_dim,
                    output_dim=output_dim,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    latent_dim=config.latent_dim,
                    output_act=None,
                    pred_type='node',
                    use_cvae=True
                )
            baseline_vae.load_state_dict(torch.load(vae_path, map_location=config.device, weights_only=True))
            baseline_vae.to(config.device)
            baseline_vae.eval()
            baseline_models['vae'] = baseline_vae
            print(f"[OK] Loaded baseline VAE from {vae_path}")
        except Exception as e:
            print(f"[Warning] Could not load baseline VAE: {e}")
    
    return baseline_models


def main():
    """Main validation function using standard evaluation framework."""
    print("="*80)
    print("Generative VAE Validation (Standard Evaluation Framework)")
    print("="*80)
    
    # Load config
    config = get_config()
    device = config.device
    print(f"Device: {device}")
    
    # Load NGT data (includes sys_data)
    print("\nLoading NGT training data...")
    ngt_data, sys_data = load_ngt_training_data(config)
    
    # Load BRANFT for post-processing
    print("Loading branch data...")
    _, _, BRANFT = load_all_data(config)
    
    # Load multi-preference dataset
    print("\nLoading multi-preference dataset...")
    multi_pref_data, _ = load_multi_preference_dataset(config, sys_data)
    
    lambda_carbon_values = multi_pref_data['lambda_carbon_values']
    input_dim = multi_pref_data['input_dim']
    output_dim = multi_pref_data['output_dim']
    NPred_Va = multi_pref_data.get('NPred_Va', output_dim // 2)
    n_val = multi_pref_data['n_val']
    
    print(f"Validation samples: {n_val}")
    print(f"Preferences: {len(lambda_carbon_values)}")
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")
    
    # Create NGT loss function
    print("\nCreating NGT loss function...")
    ngt_loss_fn = DeepOPFNGTLoss(sys_data, config)
    ngt_loss_fn.cache_to_gpu(device)
    
    # Load trained Generative VAE model
    model_path = os.path.join(
        getattr(config, 'model_save_dir', 'main_part/saved_models'),
        'generative_vae_final.pth'
    )
    
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Generative VAE model not found: {model_path}")
        print("Please train the generative VAE first using train_generative_vae_main.py")
        return None
    
    print(f"\nLoading Generative VAE from {model_path}")
    
    # Create model
    latent_dim = getattr(config, 'linearized_vae_latent_dim', 64)
    hidden_dim = getattr(config, 'linearized_vae_hidden_dim', 512)
    num_layers = getattr(config, 'linearized_vae_num_layers', 4)
    
    vae = LinearizedVAE(
        input_dim=input_dim,
        output_dim=output_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        pref_dim=1,
        NPred_Va=NPred_Va
    ).to(device)
    
    vae.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    vae.eval()
    print(f"Generative VAE loaded successfully")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num layers: {num_layers}")
    
    # Run comparison across K values
    test_lambda_carbon = 50.0  # Middle preference
    k_values = [1, 4, 8, 16, 32]
    
    print("\n" + "="*80)
    print(f"Running Best-of-K Comparison (lambda_carbon={test_lambda_carbon})")
    print("="*80)
    
    results_comparison = compare_with_baseline(
        vae, ngt_loss_fn, multi_pref_data, sys_data, BRANFT, config,
        lambda_carbon=test_lambda_carbon,
        k_values=k_values
    )
    
    # Print comparison summary
    print_comparison_summary(results_comparison)
    
    # Also compare with baseline VAE if available
    print("\n" + "="*80)
    print("Comparing with Baseline Models")
    print("="*80)
    
    baseline_models = load_baseline_models(config, input_dim, output_dim)
    
    if 'vae' in baseline_models:
        print("\nEvaluating baseline VAE (multi-preference trained)...")
        
        ctx = build_ctx_from_multi_preference(
            config, sys_data, multi_pref_data, BRANFT, device,
            lambda_carbon=test_lambda_carbon
        )
        
        baseline_predictor = MultiPreferencePredictor(
            model=baseline_models['vae'],
            multi_pref_data=multi_pref_data,
            lambda_carbon=test_lambda_carbon,
            model_type='vae',
            pretrain_model=None,
            num_flow_steps=1,
            flow_method='euler',
            training_mode='standard'
        )
        
        results_baseline_vae = evaluate_unified(ctx, baseline_predictor, apply_post_processing=True, verbose=True)
        results_comparison['baseline_vae'] = results_baseline_vae
        
        # Final comparison
        print("\n" + "="*80)
        print("FINAL COMPARISON: Generative VAE vs Baseline VAE")
        print("="*80)
        
        gen_vae_best = results_comparison.get('K=32', results_comparison.get('K=16', {}))
        baseline_vae = results_comparison.get('baseline_vae', {})
        
        if gen_vae_best and baseline_vae:
            gen_vio = gen_vae_best.get('num_viotest', gen_vae_best.get('num_viotest1', 0))
            base_vio = baseline_vae.get('num_viotest', baseline_vae.get('num_viotest1', 0))
            
            print(f"\nViolated samples:")
            print(f"  Baseline VAE: {base_vio}")
            print(f"  Generative VAE (Best-of-K): {gen_vio}")
            print(f"  Improvement: {base_vio - gen_vio} fewer violations")
            
            gen_cost = gen_vae_best.get('cost_mean', gen_vae_best.get('cost_mean1', 0))
            base_cost = baseline_vae.get('cost_mean', baseline_vae.get('cost_mean1', 0))
            
            print(f"\nCost:")
            print(f"  Baseline VAE: {base_cost:.4f}")
            print(f"  Generative VAE (Best-of-K): {gen_cost:.4f}")
            
            if base_cost > 0:
                cost_diff = (gen_cost - base_cost) / base_cost * 100
                print(f"  Difference: {cost_diff:+.2f}%")
    
    # Evaluate across multiple preferences
    print("\n" + "="*80)
    print("Evaluating Across Multiple Preferences")
    print("="*80)
    
    test_lambdas = [0.0, 25.0, 50.0, 75.0, 99.0]
    k_for_multi_pref = 16  # Use K=16 for efficiency
    
    results_by_pref = {}
    for lc in test_lambdas:
        print(f"\n--- lambda_carbon = {lc} ---")
        results_k = run_evaluation_with_k(
            vae, ngt_loss_fn, multi_pref_data, sys_data, BRANFT, config,
            lambda_carbon=lc, k_value=k_for_multi_pref, use_mean=False, verbose=False
        )
        results_by_pref[lc] = results_k
        
        vio = results_k.get('num_viotest', results_k.get('num_viotest1', 0))
        cost = results_k.get('cost_mean', results_k.get('cost_mean1', 0))
        carbon = results_k.get('carbon_mean', results_k.get('carbon_mean1', 0))
        print(f"  Violations: {vio}, Cost: {cost:.4f}, Carbon: {carbon:.4f}")
    
    print("\n" + "="*80)
    print("Validation Complete!")
    print("="*80)
    
    return results_comparison


if __name__ == '__main__':
    results = main()
