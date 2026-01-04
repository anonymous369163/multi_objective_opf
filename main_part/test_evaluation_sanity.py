#!/usr/bin/env python
# coding: utf-8
"""
Sanity Check for Evaluation Function

This script tests the evaluation function by using ground truth labels
as if they were model predictions. If the evaluation function is correct:
- Vm MAE and Va MAE should be ~0
- Constraint satisfaction should be high (close to 100%)

If the evaluation returns poor constraint satisfaction even with ground truth,
it indicates potential issues with:
1. Data format conversion (NGT -> full voltage)
2. Kron reconstruction for ZIB nodes
3. The ground truth labels themselves

Author: Sanity Check
Date: 2026-01-03
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BaseConfig, _SCRIPT_DIR
from data_loader import load_multi_preference_dataset
from unified_eval import (
    EvalContext, PredPack, build_ctx_from_multi_preference, evaluate_unified,
    _as_numpy, _as_torch, _remove_slack_va
)


class MultiPreferenceConfig(BaseConfig):
    """Configuration for multi-preference evaluation sanity check."""
    
    def __init__(self):
        super().__init__()
        
        self.multi_pref_dataset_path = os.path.join(
            os.path.dirname(_SCRIPT_DIR), 'saved_data', 'multi_preference_solutions', 'fully_covered_dataset_2026-01-02.pt'
        )
        
        # Default values for evaluation
        self.multi_pref_val_ratio = float(os.environ.get('MULTI_PREF_VAL_RATIO', '0.2'))
        self.multi_pref_random_seed = int(os.environ.get('MULTI_PREF_RANDOM_SEED', '42'))
        self.multi_pref_flow_steps = 10
        self.multi_pref_training_mode = 'standard'
        
        # NGT compatible settings
        self.ngt_hidden_units = 1
        self.ngt_khidden = np.array([64, 224], dtype=int)
        self.pref_dim = 1
        

class GroundTruthPredictor:
    """
    A "fake" predictor that returns ground truth labels as predictions.
    
    This is used to test if the evaluation function itself is correct.
    If we feed ground truth as prediction, we expect:
    - MAE ≈ 0
    - Constraint satisfaction ≈ 100%
    """
    
    def __init__(
        self,
        multi_pref_data: dict,
        lambda_carbon: float,
        use_val_set: bool = True,
    ):
        """
        Initialize the ground truth predictor.
        
        Args:
            multi_pref_data: Multi-preference data dictionary
            lambda_carbon: Preference value to use for ground truth selection
            use_val_set: If True, use validation set. If False, use training set.
        """
        self.multi_pref_data = multi_pref_data
        self.lambda_carbon = lambda_carbon
        self.use_val_set = use_val_set
        self.model_type = "ground_truth"  # For logging
        
        # Get normalization factor for preference
        lambda_carbon_values = multi_pref_data.get('lambda_carbon_values', [55.0])
        self.lc_max = max(lambda_carbon_values) if max(lambda_carbon_values) > 0 else 1.0
        
        # Get bus indices for NGT format conversion
        self.bus_Pnet_all = multi_pref_data.get('bus_Pnet_all')
        self.bus_Pnet_noslack_all = multi_pref_data.get('bus_Pnet_noslack_all')
        self.bus_ZIB_all = multi_pref_data.get('bus_ZIB_all')
        self.param_ZIMV = multi_pref_data.get('param_ZIMV')
        
    def predict(self, ctx: EvalContext) -> PredPack:
        """
        Return ground truth as prediction.
        
        This directly returns ctx.Real_Vm_full and ctx.Real_Va_full
        which are the ground truth labels already stored in the context.
        """
        # The ground truth is already stored in ctx
        # Simply return it as prediction
        Pred_Vm_full = ctx.Real_Vm_full.copy()
        Pred_Va_full = ctx.Real_Va_full.copy()
        
        print(f"\n[GroundTruthPredictor] Returning ground truth as prediction")
        print(f"  Vm shape: {Pred_Vm_full.shape}, range: [{Pred_Vm_full.min():.4f}, {Pred_Vm_full.max():.4f}]")
        print(f"  Va shape: {Pred_Va_full.shape}, range: [{Pred_Va_full.min():.4f}, {Pred_Va_full.max():.4f}]")
        
        return PredPack(
            Pred_Vm_full=Pred_Vm_full,
            Pred_Va_full=Pred_Va_full,
            time_vm=0.0,
            time_va=0.0,
            time_nn_total=0.0,
        )


class GroundTruthFromYPredictor:
    """
    A predictor that reconstructs full voltage from y_val_by_pref (NGT format)
    and returns it as prediction.
    
    This tests the full conversion pipeline:
    y_val_by_pref (NGT) -> reconstruction -> full voltage
    """
    
    def __init__(
        self,
        multi_pref_data: dict,
        lambda_carbon: float,
        use_val_set: bool = True,
    ):
        self.multi_pref_data = multi_pref_data
        self.lambda_carbon = lambda_carbon
        self.use_val_set = use_val_set
        self.model_type = "ground_truth_from_y"
        
        lambda_carbon_values = multi_pref_data.get('lambda_carbon_values', [55.0])
        self.lc_max = max(lambda_carbon_values) if max(lambda_carbon_values) > 0 else 1.0
        
    def predict(self, ctx: EvalContext) -> PredPack:
        """
        Reconstruct full voltage from y labels (NGT format) and return as prediction.
        
        This simulates what the model does:
        1. Model outputs y in NGT format
        2. y is converted to full voltage using Kron reconstruction
        """
        # Get ground truth y in NGT format
        if self.use_val_set and 'y_val_by_pref' in self.multi_pref_data:
            y_by_pref = self.multi_pref_data['y_val_by_pref']
        else:
            y_by_pref = self.multi_pref_data['y_train_by_pref']
        
        # Find the y for the specified lambda_carbon
        lambda_carbon_values = self.multi_pref_data['lambda_carbon_values']
        if self.lambda_carbon in y_by_pref:
            y_test = y_by_pref[self.lambda_carbon]
        else:
            closest_lc = min(lambda_carbon_values, key=lambda x: abs(x - self.lambda_carbon))
            y_test = y_by_pref[closest_lc]
            print(f"[Warning] lambda_carbon={self.lambda_carbon:.2f} not found, using {closest_lc:.2f}")
        
        # Convert to numpy
        y_test_np = _as_numpy(y_test)
        
        # Get bus indices
        bus_Pnet_all = self.multi_pref_data['bus_Pnet_all']
        bus_Pnet_noslack_all = self.multi_pref_data['bus_Pnet_noslack_all']
        bus_ZIB_all = self.multi_pref_data.get('bus_ZIB_all')
        param_ZIMV = self.multi_pref_data.get('param_ZIMV')
        
        NPred_Va = len(bus_Pnet_noslack_all)
        NPred_Vm = len(bus_Pnet_all)
        
        Ntest = y_test_np.shape[0]
        Nbus = ctx.Nbus
        bus_slack = ctx.bus_slack
        
        # Extract Va and Vm from NGT format
        Va_noslack_nonZIB = y_test_np[:, :NPred_Va]
        Vm_nonZIB = y_test_np[:, NPred_Va:]
        
        # Reconstruct full voltage (same as build_ctx_from_multi_preference)
        Pred_Va_full = np.zeros((Ntest, Nbus), dtype=float)
        Pred_Vm_full = np.zeros((Ntest, Nbus), dtype=float)
        
        Pred_Va_full[:, bus_Pnet_noslack_all] = Va_noslack_nonZIB
        Pred_Va_full[:, bus_slack] = 0.0
        Pred_Vm_full[:, bus_Pnet_all] = Vm_nonZIB
        
        # Apply Kron reconstruction for ZIB if available
        if bus_ZIB_all is not None and param_ZIMV is not None and len(bus_ZIB_all) > 0:
            from unified_eval import _kron_reconstruct_zib, _ensure_1d_int
            Pred_Vm_full, Pred_Va_full = _kron_reconstruct_zib(
                Pred_Vm_full, Pred_Va_full,
                bus_Pnet_all=bus_Pnet_all,
                bus_ZIB_all=_ensure_1d_int(bus_ZIB_all),
                param_ZIMV=np.asarray(param_ZIMV),
            )
        
        print(f"\n[GroundTruthFromYPredictor] Reconstructed from y labels (NGT format)")
        print(f"  y shape: {y_test_np.shape}")
        print(f"  NPred_Va: {NPred_Va}, NPred_Vm: {NPred_Vm}")
        print(f"  Vm shape: {Pred_Vm_full.shape}, range: [{Pred_Vm_full.min():.4f}, {Pred_Vm_full.max():.4f}]")
        print(f"  Va shape: {Pred_Va_full.shape}, range: [{Pred_Va_full.min():.4f}, {Pred_Va_full.max():.4f}]")
        
        return PredPack(
            Pred_Vm_full=Pred_Vm_full,
            Pred_Va_full=Pred_Va_full,
            time_vm=0.0,
            time_va=0.0,
            time_nn_total=0.0,
        )


def run_sanity_check():
    """Run sanity check on the evaluation function."""
    
    print("=" * 80)
    print("Evaluation Function Sanity Check")
    print("=" * 80)
    print("\nThis test verifies the evaluation function by using ground truth")
    print("labels as predictions. Expected results:")
    print("  - Vm MAE ≈ 0")
    print("  - Va MAE ≈ 0")
    print("  - Constraint satisfaction ≈ 100%")
    print("=" * 80)
    
    config = MultiPreferenceConfig()
    
    # Load data
    print("\n[1] Loading multi-preference dataset...")
    multi_pref_data, sys_data = load_multi_preference_dataset(config)
    
    # Compute BRANFT
    BRANFT = torch.from_numpy(sys_data.branch[:, 0:2] - 1).long()
    
    # Test lambdas
    test_lambdas = [0.0, 25.0, 50.0]
    
    print("\n[2] Running sanity checks...")
    
    # ========== Test 1: GroundTruthPredictor ==========
    print("\n" + "=" * 60)
    print("TEST 1: GroundTruthPredictor")
    print("(Returns ctx.Real_Vm_full and ctx.Real_Va_full directly)")
    print("=" * 60)
    
    for lc in test_lambdas:
        print(f"\n--- lambda_carbon = {lc:.2f} ---")
        ctx = build_ctx_from_multi_preference(
            config, sys_data, multi_pref_data, BRANFT, config.device, lambda_carbon=lc
        )
        predictor = GroundTruthPredictor(
            multi_pref_data=multi_pref_data,
            lambda_carbon=lc,
            use_val_set=True
        )
        results = evaluate_unified(ctx, predictor, apply_post_processing=False, verbose=True)
        
        print(f"\n[Summary for lambda={lc:.2f}]")
        print(f"  Vm MAE: {results['mae_Vmtest']:.8f}")
        print(f"  Va MAE: {results['mae_Vatest']:.8f}")
        vio_PQg = _as_numpy(results['vio_PQg'])
        print(f"  Pg satisfaction: {float(np.mean(vio_PQg[:, 0])):.2f}%")
        print(f"  Qg satisfaction: {float(np.mean(vio_PQg[:, 1])):.2f}%")
    
    # ========== Test 2: GroundTruthFromYPredictor ==========
    print("\n" + "=" * 60)
    print("TEST 2: GroundTruthFromYPredictor")
    print("(Reconstructs full voltage from y_val_by_pref in NGT format)")
    print("This tests the NGT -> full voltage conversion pipeline.")
    print("=" * 60)
    
    for lc in test_lambdas:
        print(f"\n--- lambda_carbon = {lc:.2f} ---")
        ctx = build_ctx_from_multi_preference(
            config, sys_data, multi_pref_data, BRANFT, config.device, lambda_carbon=lc
        )
        predictor = GroundTruthFromYPredictor(
            multi_pref_data=multi_pref_data,
            lambda_carbon=lc,
            use_val_set=True
        )
        results = evaluate_unified(ctx, predictor, apply_post_processing=False, verbose=True)
        
        print(f"\n[Summary for lambda={lc:.2f}]")
        print(f"  Vm MAE: {results['mae_Vmtest']:.8f}")
        print(f"  Va MAE: {results['mae_Vatest']:.8f}")
        vio_PQg2 = _as_numpy(results['vio_PQg'])
        print(f"  Pg satisfaction: {float(np.mean(vio_PQg2[:, 0])):.2f}%")
        print(f"  Qg satisfaction: {float(np.mean(vio_PQg2[:, 1])):.2f}%")
    
    # ========== Test 3: Compare ground truth with itself ==========
    print("\n" + "=" * 60)
    print("TEST 3: Data Integrity Check")
    print("Comparing y_val_by_pref reconstruction with ctx.Real_Vm/Va_full")
    print("=" * 60)
    
    lc = 0.0
    ctx = build_ctx_from_multi_preference(
        config, sys_data, multi_pref_data, BRANFT, config.device, lambda_carbon=lc
    )
    
    # Get y_val_by_pref
    y_by_pref = multi_pref_data.get('y_val_by_pref', multi_pref_data['y_train_by_pref'])
    y_test = _as_numpy(y_by_pref[lc])
    
    bus_Pnet_all = multi_pref_data['bus_Pnet_all']
    bus_Pnet_noslack_all = multi_pref_data['bus_Pnet_noslack_all']
    
    NPred_Va = len(bus_Pnet_noslack_all)
    NPred_Vm = len(bus_Pnet_all)
    
    # Extract from y
    Va_noslack_nonZIB = y_test[:, :NPred_Va]
    Vm_nonZIB = y_test[:, NPred_Va:]
    
    # Extract from Real_Va/Vm_full (from ctx)
    Real_Va_noslack_nonZIB = ctx.Real_Va_full[:, bus_Pnet_noslack_all]
    Real_Vm_nonZIB = ctx.Real_Vm_full[:, bus_Pnet_all]
    
    # Compare
    diff_Va = np.abs(Va_noslack_nonZIB - Real_Va_noslack_nonZIB)
    diff_Vm = np.abs(Vm_nonZIB - Real_Vm_nonZIB)
    
    print(f"\nComparison (y_val_by_pref vs ctx.Real_*):")
    print(f"  Va difference - max: {diff_Va.max():.8f}, mean: {diff_Va.mean():.8f}")
    print(f"  Vm difference - max: {diff_Vm.max():.8f}, mean: {diff_Vm.mean():.8f}")
    
    if diff_Va.max() < 1e-6 and diff_Vm.max() < 1e-6:
        print("\n  [OK] Data is consistent!")
    else:
        print("\n  [WARNING] Data mismatch detected!")
        print("    This could explain evaluation discrepancies.")
    
    # ========== Test 4: Check the actual ground truth constraint satisfaction ==========
    print("\n" + "=" * 60)
    print("TEST 4: Ground Truth Constraint Satisfaction Analysis")
    print("Are the ground truth labels themselves feasible solutions?")
    print("=" * 60)
    
    for lc in test_lambdas:
        print(f"\n--- lambda_carbon = {lc:.2f} ---")
        ctx = build_ctx_from_multi_preference(
            config, sys_data, multi_pref_data, BRANFT, config.device, lambda_carbon=lc
        )
        
        # Get Vm and Va from ground truth
        Vm_full = ctx.Real_Vm_full
        Va_full = ctx.Real_Va_full
        
        # Check Vm range
        Vm_min = Vm_full.min()
        Vm_max = Vm_full.max()
        
        # Check for zeros (uninitialized nodes)
        zero_count = np.sum(Vm_full == 0)
        
        print(f"  Vm range: [{Vm_min:.4f}, {Vm_max:.4f}]")
        print(f"  Zero values in Vm: {zero_count}")
        
        if Vm_min < 0.8 or Vm_max > 1.2:
            print(f"  [WARNING] Vm values outside typical range [0.9, 1.1]")
        if zero_count > 0:
            print(f"  [WARNING] {zero_count} zero values found (possible reconstruction issue)")
        
        # Print sample statistics
        print(f"  Va range: [{Va_full.min():.4f}, {Va_full.max():.4f}] rad")
    
    print("\n" + "=" * 80)
    print("Sanity Check Complete")
    print("=" * 80)
    print("\nInterpretation:")
    print("  - If TEST 1 shows MAE ≈ 0 and high constraint satisfaction,")
    print("    the ground truth labels are correct and evaluation works.")
    print("  - If TEST 1 fails, check the ground truth label generation process.")
    print("  - If TEST 2 differs from TEST 1, there's an issue in NGT reconstruction.")
    print("  - If TEST 3 shows mismatches, there's a data consistency problem.")
    print("=" * 80)


def run_detailed_constraint_analysis():
    """Analyze constraints in detail to understand why feasibility might be low."""
    
    print("\n" + "=" * 80)
    print("Detailed Constraint Analysis")
    print("=" * 80)
    
    config = MultiPreferenceConfig()
    multi_pref_data, sys_data = load_multi_preference_dataset(config)
    BRANFT = torch.from_numpy(sys_data.branch[:, 0:2] - 1).long()
    
    lc = 0.0
    ctx = build_ctx_from_multi_preference(
        config, sys_data, multi_pref_data, BRANFT, config.device, lambda_carbon=lc
    )
    
    # Import constraint checking functions
    from unified_eval import get_genload, get_vioPQg, get_viobran2
    
    # Use ground truth
    Pred_Vm_full = ctx.Real_Vm_full.copy()
    Pred_Va_full = ctx.Real_Va_full.copy()
    
    print(f"\n[Ground Truth Statistics]")
    print(f"  Ntest: {ctx.Ntest}")
    print(f"  Nbus: {ctx.Nbus}")
    print(f"  bus_slack: {ctx.bus_slack}")
    print(f"  Vm range: [{Pred_Vm_full.min():.4f}, {Pred_Vm_full.max():.4f}]")
    print(f"  Va range: [{Pred_Va_full.min():.4f}, {Pred_Va_full.max():.4f}]")
    
    # Check for zero Vm values
    zero_mask = Pred_Vm_full == 0
    num_zeros = zero_mask.sum()
    if num_zeros > 0:
        print(f"\n  [WARNING] {num_zeros} zero values in Pred_Vm_full!")
        # Find which buses have zeros
        zero_buses = np.where(zero_mask.any(axis=0))[0]
        print(f"  Zero buses: {zero_buses[:10]}... (showing first 10)")
    
    # Compute power flow
    print(f"\n[Power Flow Calculation]")
    Pred_V = Pred_Vm_full * np.exp(1j * Pred_Va_full)
    Pred_Pg, Pred_Qg, Pred_Pd, Pred_Qd = get_genload(
        Pred_V, ctx.Pdtest, ctx.Qdtest, ctx.bus_Pg, ctx.bus_Qg, ctx.Ybus
    )
    
    print(f"  Pg shape: {Pred_Pg.shape}")
    print(f"  Pg range: [{Pred_Pg.min():.4f}, {Pred_Pg.max():.4f}] p.u.")
    print(f"  Qg range: [{Pred_Qg.min():.4f}, {Pred_Qg.max():.4f}] p.u.")
    
    # Check constraints
    print(f"\n[Generator Constraints]")
    print(f"  MAXMIN_Pg shape: {ctx.MAXMIN_Pg.shape}")
    print(f"  MAXMIN_Qg shape: {ctx.MAXMIN_Qg.shape}")
    print(f"  bus_Pg: {ctx.bus_Pg}")
    print(f"  bus_Qg: {ctx.bus_Qg}")
    
    lsPg, lsQg, lsidxPg, lsidxQg, vio_PQgmaxmin, vio_PQg, deltaPgL, deltaPgU, deltaQgL, deltaQgU = get_vioPQg(
        Pred_Pg, ctx.bus_Pg, ctx.MAXMIN_Pg,
        Pred_Qg, ctx.bus_Qg, ctx.MAXMIN_Qg,
        ctx.DELTA
    )
    
    print(f"\n[Constraint Violation Details]")
    vio_PQg_np = _as_numpy(vio_PQg)
    print(f"  Pg constraint satisfaction: {float(np.mean(vio_PQg_np[:, 0])):.2f}%")
    print(f"  Qg constraint satisfaction: {float(np.mean(vio_PQg_np[:, 1])):.2f}%")
    
    # Note: Pred_Pg shape is (Ntest, Ngen), already contains only generator powers
    # bus_Pg contains the bus indices of generators
    # IMPORTANT: MAXMIN_Pg format is [Pmax, Pmin] NOT [Pmin, Pmax]!
    Pg_gen = Pred_Pg  # [Ntest, Ngen] - already generator powers
    Pg_max = ctx.MAXMIN_Pg[:, 0]  # [Ngen] - UPPER bound
    Pg_min = ctx.MAXMIN_Pg[:, 1]  # [Ngen] - LOWER bound
    
    print(f"\n  [Pg Constraint Analysis]")
    print(f"    Pg_gen shape: {Pg_gen.shape}")
    print(f"    Number of generators: {len(ctx.bus_Pg)}")
    print(f"    Pg limits: min={Pg_min.min():.4f} to {Pg_min.max():.4f}, max={Pg_max.min():.4f} to {Pg_max.max():.4f}")
    
    # Check each generator
    violated_gens = []
    for g in range(Pg_gen.shape[1]):
        pg_vals = Pg_gen[:, g]
        below_min = (pg_vals < Pg_min[g] - ctx.DELTA).sum()
        above_max = (pg_vals > Pg_max[g] + ctx.DELTA).sum()
        if below_min > 0 or above_max > 0:
            violated_gens.append(g)
            print(f"    Gen {g} (bus {ctx.bus_Pg[g]}): "
                  f"range=[{pg_vals.min():.4f}, {pg_vals.max():.4f}], "
                  f"limits=[{Pg_min[g]:.4f}, {Pg_max[g]:.4f}], "
                  f"below_min={below_min}, above_max={above_max}")
    
    print(f"    Violated generators: {len(violated_gens)}/{Pg_gen.shape[1]}")
    
    # Note: Pred_Qg shape is (Ntest, Ngen_Q), similar to Pg
    # IMPORTANT: MAXMIN_Qg format is [Qmax, Qmin] NOT [Qmin, Qmax]!
    Qg_gen = Pred_Qg  # [Ntest, Ngen_Q]
    Qg_max = ctx.MAXMIN_Qg[:, 0]  # [Ngen_Q] - UPPER bound
    Qg_min = ctx.MAXMIN_Qg[:, 1]  # [Ngen_Q] - LOWER bound
    
    print(f"\n  [Qg Constraint Analysis]")
    print(f"    Qg_gen shape: {Qg_gen.shape}")
    print(f"    Number of reactive power generators: {len(ctx.bus_Qg)}")
    print(f"    Qg limits: min={Qg_min.min():.4f} to {Qg_min.max():.4f}, max={Qg_max.min():.4f} to {Qg_max.max():.4f}")
    
    violated_gens_q = []
    for g in range(Qg_gen.shape[1]):
        qg_vals = Qg_gen[:, g]
        below_min = (qg_vals < Qg_min[g] - ctx.DELTA).sum()
        above_max = (qg_vals > Qg_max[g] + ctx.DELTA).sum()
        if below_min > 0 or above_max > 0:
            violated_gens_q.append(g)
            print(f"    Gen {g} (bus {ctx.bus_Qg[g]}): "
                  f"range=[{qg_vals.min():.4f}, {qg_vals.max():.4f}], "
                  f"limits=[{Qg_min[g]:.4f}, {Qg_max[g]:.4f}], "
                  f"below_min={below_min}, above_max={above_max}")
    
    print(f"    Violated generators (Q): {len(violated_gens_q)}/{Qg_gen.shape[1]}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_sanity_check()
    
    # Optionally run detailed analysis
    print("\n\n")
    run_detailed_constraint_analysis()

