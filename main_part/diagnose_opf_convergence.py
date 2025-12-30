#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose OPF convergence issues in PyPowerOPFSolver.
Analyzes why certain load scenarios fail to converge.
"""

import numpy as np
import sys
sys.path.insert(0, 'main_part')

from opf_by_pypower import PyPowerOPFSolver, _infer_load_mapping


def diagnose_opf_convergence(n_samples=50, verbose=True):
    """
    Diagnose OPF convergence issues by analyzing failed samples.
    
    Args:
        n_samples: Number of samples to test (default: 50 for speed)
        verbose: Whether to print detailed diagnostics
    """
    from config import get_config
    from data_loader import load_ngt_training_data
    
    print("=" * 70)
    print("OPF Convergence Diagnosis")
    print("=" * 70)
    
    # Load data
    config = get_config()
    ngt_data, sys_data = load_ngt_training_data(config, sys_data=None)
    
    x_train = ngt_data["x_train"].detach().cpu().numpy()
    
    # Initialize solver with default tolerance
    solver = PyPowerOPFSolver(
        case_m_path='main_part/data/case300_ieee_modified.m',
        ngt_data=ngt_data,
        verbose=False,
        use_multi_objective=False
    )
    
    print(f"\n[System Info]")
    print(f"  Buses: {solver.nbus}, Generators: {solver.ngen}, Branches: {solver.nbranch}")
    print(f"  Base MVA: {solver.baseMVA}")
    print(f"  Slack bus row: {solver.slack_row}")
    
    # Get generator limits
    gen = solver.ppc_base["gen"]
    total_Pmax = np.sum(gen[:, 8])  # Column 8 is Pmax
    total_Pmin = np.sum(gen[:, 9])  # Column 9 is Pmin
    total_Qmax = np.sum(gen[:, 3])  # Column 3 is Qmax
    total_Qmin = np.sum(gen[:, 4])  # Column 4 is Qmin
    
    print(f"\n[Generator Capacity]")
    print(f"  Total Pmax: {total_Pmax:.2f} MW")
    print(f"  Total Pmin: {total_Pmin:.2f} MW")
    print(f"  Total Qmax: {total_Qmax:.2f} MVAr")
    print(f"  Total Qmin: {total_Qmin:.2f} MVAr")
    
    # Sample indices (use fixed seed for reproducibility)
    rng = np.random.default_rng(42)
    sample_idxs = rng.choice(x_train.shape[0], size=min(n_samples, x_train.shape[0]), replace=False)
    
    # Collect statistics
    success_samples = []
    failed_samples = []
    load_stats = {
        'success': {'Pd': [], 'Qd': [], 'Pd_max': [], 'Qd_max': [], 'neg_Pd': [], 'neg_Qd': []},
        'failed': {'Pd': [], 'Qd': [], 'Pd_max': [], 'Qd_max': [], 'neg_Pd': [], 'neg_Qd': []}
    }
    
    print(f"\n[Testing {len(sample_idxs)} samples...]")
    
    for i, idx in enumerate(sample_idxs):
        x = x_train[idx]
        
        # Get load mapping
        Pd_pu, Qd_pu, load_mode = _infer_load_mapping(
            x, solver.nbus, solver.slack_row, ngt_data
        )
        
        # Convert to MW/MVAr
        Pd_MW = Pd_pu * solver.baseMVA
        Qd_MVAr = Qd_pu * solver.baseMVA
        
        # Calculate statistics
        total_Pd = np.sum(np.maximum(Pd_MW, 0))
        total_Qd = np.sum(np.maximum(Qd_MVAr, 0))
        max_Pd = np.max(Pd_MW)
        max_Qd = np.max(Qd_MVAr)
        n_neg_Pd = np.sum(Pd_MW < 0)
        n_neg_Qd = np.sum(Qd_MVAr < 0)
        
        # Run OPF
        result = solver.forward(x)
        
        if result["success"]:
            success_samples.append(idx)
            load_stats['success']['Pd'].append(total_Pd)
            load_stats['success']['Qd'].append(total_Qd)
            load_stats['success']['Pd_max'].append(max_Pd)
            load_stats['success']['Qd_max'].append(max_Qd)
            load_stats['success']['neg_Pd'].append(n_neg_Pd)
            load_stats['success']['neg_Qd'].append(n_neg_Qd)
        else:
            failed_samples.append(idx)
            load_stats['failed']['Pd'].append(total_Pd)
            load_stats['failed']['Qd'].append(total_Qd)
            load_stats['failed']['Pd_max'].append(max_Pd)
            load_stats['failed']['Qd_max'].append(max_Qd)
            load_stats['failed']['neg_Pd'].append(n_neg_Pd)
            load_stats['failed']['neg_Qd'].append(n_neg_Qd)
    
    # Print summary
    n_success = len(success_samples)
    n_failed = len(failed_samples)
    print(f"\n[Results Summary]")
    print(f"  Success: {n_success}/{len(sample_idxs)} ({100*n_success/len(sample_idxs):.1f}%)")
    print(f"  Failed: {n_failed}/{len(sample_idxs)} ({100*n_failed/len(sample_idxs):.1f}%)")
    
    # Compare load statistics
    print(f"\n[Load Statistics Comparison]")
    print(f"{'Metric':<25} {'Success Mean':<15} {'Failed Mean':<15} {'Difference':<15}")
    print("-" * 70)
    
    for key in ['Pd', 'Qd', 'Pd_max', 'Qd_max', 'neg_Pd', 'neg_Qd']:
        succ_vals = load_stats['success'][key]
        fail_vals = load_stats['failed'][key]
        
        succ_mean = np.mean(succ_vals) if succ_vals else 0
        fail_mean = np.mean(fail_vals) if fail_vals else 0
        diff = fail_mean - succ_mean
        
        print(f"{key:<25} {succ_mean:<15.2f} {fail_mean:<15.2f} {diff:<+15.2f}")
    
    # Analyze capacity margin
    if load_stats['success']['Pd']:
        succ_Pd_mean = np.mean(load_stats['success']['Pd'])
        succ_margin = (total_Pmax - succ_Pd_mean) / total_Pmax * 100
        print(f"\n[Capacity Analysis]")
        print(f"  Success samples avg Pd: {succ_Pd_mean:.2f} MW")
        print(f"  Capacity margin (success): {succ_margin:.1f}%")
    
    if load_stats['failed']['Pd']:
        fail_Pd_mean = np.mean(load_stats['failed']['Pd'])
        fail_margin = (total_Pmax - fail_Pd_mean) / total_Pmax * 100
        print(f"  Failed samples avg Pd: {fail_Pd_mean:.2f} MW")
        print(f"  Capacity margin (failed): {fail_margin:.1f}%")
    
    # Detailed analysis of failed samples
    if failed_samples and verbose:
        print(f"\n[Detailed Analysis of First 3 Failed Samples]")
        for i, idx in enumerate(failed_samples[:3]):
            x = x_train[idx]
            Pd_pu, Qd_pu, _ = _infer_load_mapping(x, solver.nbus, solver.slack_row, ngt_data)
            Pd_MW = Pd_pu * solver.baseMVA
            Qd_MVAr = Qd_pu * solver.baseMVA
            
            print(f"\n  Sample {idx}:")
            print(f"    Total Pd: {np.sum(np.maximum(Pd_MW, 0)):.2f} MW (limit: {total_Pmax:.2f} MW)")
            print(f"    Total Qd: {np.sum(np.maximum(Qd_MVAr, 0)):.2f} MVAr")
            print(f"    Max bus Pd: {np.max(Pd_MW):.2f} MW")
            print(f"    Negative Pd buses: {np.sum(Pd_MW < 0)}")
            print(f"    Negative Qd buses: {np.sum(Qd_MVAr < 0)}")
            
            # Check if load exceeds capacity
            if np.sum(np.maximum(Pd_MW, 0)) > total_Pmax:
                print(f"    [!] LOAD EXCEEDS GENERATION CAPACITY!")
    
    # Test with relaxed tolerance
    print(f"\n[Testing with Relaxed Tolerance (1e-3)]")
    solver_relaxed = PyPowerOPFSolver(
        case_m_path='main_part/data/case300_ieee_modified.m',
        ngt_data=ngt_data,
        verbose=False,
        use_multi_objective=False,
        opf_violation=1e-3,
        feastol=1e-3
    )
    
    recovered = 0
    for idx in failed_samples[:10]:  # Test first 10 failed samples
        x = x_train[idx]
        result = solver_relaxed.forward(x)
        if result["success"]:
            recovered += 1
    
    if failed_samples:
        print(f"  Recovered with relaxed tolerance: {recovered}/{min(10, len(failed_samples))}")
    
    # Possible causes summary
    print(f"\n{'='*70}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'='*70}")
    
    if n_failed > 0:
        fail_Pd_vals = load_stats['failed']['Pd']
        succ_Pd_vals = load_stats['success']['Pd']
        
        causes = []
        
        # Check 1: Load exceeds capacity
        if fail_Pd_vals and np.max(fail_Pd_vals) > total_Pmax * 0.95:
            causes.append("1. LOAD NEAR/EXCEEDING CAPACITY: Some failed samples have total load close to or exceeding generator capacity.")
        
        # Check 2: High load difference
        if fail_Pd_vals and succ_Pd_vals:
            if np.mean(fail_Pd_vals) > np.mean(succ_Pd_vals) * 1.05:
                causes.append("2. HIGHER LOAD: Failed samples have significantly higher average load than successful ones.")
        
        # Check 3: Negative loads
        if load_stats['failed']['neg_Pd'] and np.mean(load_stats['failed']['neg_Pd']) > 0:
            causes.append("3. NEGATIVE LOAD VALUES: Some buses have negative Pd (net generation), which may cause issues after clamping to 0.")
        
        # Check 4: Numerical issues
        if recovered > 0:
            causes.append("4. NUMERICAL TOLERANCE: Some failures can be recovered with relaxed tolerance, suggesting numerical precision issues.")
        
        if causes:
            print("\nPotential causes of convergence failure:")
            for cause in causes:
                print(f"  - {cause}")
        else:
            print("\nNo obvious cause identified. May need deeper investigation.")
        
        print("\nRecommendations:")
        print("  1. Use relaxed tolerance (opf_violation=1e-3, feastol=1e-3) for edge cases")
        print("  2. Pre-filter samples with load > 95% of capacity")
        print("  3. Consider scaling extreme load scenarios")
        print("  4. Check branch flow limits for congested lines")
    else:
        print("\nAll samples converged successfully!")
    
    return {
        'n_success': n_success,
        'n_failed': n_failed,
        'success_rate': n_success / len(sample_idxs) * 100,
        'failed_samples': failed_samples,
        'load_stats': load_stats
    }


def quick_test_different_tolerances():
    """Quick test with different tolerance levels."""
    from config import get_config
    from data_loader import load_ngt_training_data
    
    print("\n" + "=" * 70)
    print("Testing Different Tolerance Levels")
    print("=" * 70)
    
    config = get_config()
    ngt_data, sys_data = load_ngt_training_data(config, sys_data=None)
    x_train = ngt_data["x_train"].detach().cpu().numpy()
    
    # Test with 30 samples for speed
    rng = np.random.default_rng(42)
    sample_idxs = rng.choice(x_train.shape[0], size=30, replace=False)
    
    tolerances = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    
    results = []
    for tol in tolerances:
        solver = PyPowerOPFSolver(
            case_m_path='main_part/data/case300_ieee_modified.m',
            ngt_data=ngt_data,
            verbose=False,
            opf_violation=tol,
            feastol=tol
        )
        
        n_success = 0
        for idx in sample_idxs:
            result = solver.forward(x_train[idx])
            if result["success"]:
                n_success += 1
        
        rate = n_success / len(sample_idxs) * 100
        results.append((tol, n_success, rate))
        print(f"  Tolerance={tol:.0e}: {n_success}/{len(sample_idxs)} success ({rate:.1f}%)")
    
    return results


if __name__ == "__main__":
    # Run main diagnosis
    results = diagnose_opf_convergence(n_samples=50)
    
    # Test different tolerances
    quick_test_different_tolerances()
