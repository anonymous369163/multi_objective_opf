#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze multi-objective OPF dataset characteristics.

This script loads the multi-preference dataset and analyzes:
- Total number of samples
- Number of load scenarios
- Number of preferences
- Input dimensions and ranges
- Output dimensions and ranges
- Training task difficulty assessment
"""

import os
import sys
import numpy as np
import torch
import math
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_part.config import get_config
from main_part.data_loader import load_multi_preference_dataset, load_ngt_training_data, load_all_data

def analyze_dataset():
    """Analyze the multi-objective OPF dataset."""
    print("=" * 80)
    print("Multi-Objective OPF Dataset Analysis")
    print("=" * 80)
    
    # Load configuration
    config = get_config()
    
    # Try to load multi-preference dataset
    try:
        print("\n[1/3] Loading multi-preference dataset...")
        multi_pref_data, sys_data = load_multi_preference_dataset(config)
        
        # Extract key information
        n_samples = multi_pref_data['n_samples']
        n_train = multi_pref_data['n_train']
        n_val = multi_pref_data['n_val']
        n_preferences = multi_pref_data['n_preferences']
        lambda_carbon_values = multi_pref_data['lambda_carbon_values']
        input_dim = multi_pref_data['input_dim']
        output_dim = multi_pref_data['output_dim']
        
        x_train = multi_pref_data['x_train']
        y_train_by_pref = multi_pref_data['y_train_by_pref']
        
        print("\n" + "=" * 80)
        print("Dataset Statistics")
        print("=" * 80)
        
        print(f"\n1. Data Volume:")
        print(f"   Total samples: {n_samples:,}")
        print(f"   Training samples: {n_train:,} ({n_train/n_samples*100:.1f}%)")
        print(f"   Validation samples: {n_val:,} ({n_val/n_samples*100:.1f}%)")
        print(f"   Number of load scenarios: {n_samples:,}")
        
        print(f"\n2. Preference Settings:")
        print(f"   Number of preferences: {n_preferences}")
        print(f"   Lambda carbon range: [{lambda_carbon_values[0]:.2f}, {lambda_carbon_values[-1]:.2f}]")
        print(f"   Lambda carbon step: {lambda_carbon_values[1] - lambda_carbon_values[0]:.2f}")
        print(f"   Total training pairs: {n_train * n_preferences:,} (samples × preferences)")
        
        print(f"\n3. Input Features:")
        print(f"   Input dimension: {input_dim}")
        if isinstance(x_train, torch.Tensor):
            x_np = x_train.detach().cpu().numpy()
        else:
            x_np = x_train
        print(f"   Input range: [{np.min(x_np):.6f}, {np.max(x_np):.6f}]")
        print(f"   Input mean: {np.mean(x_np):.6f}, std: {np.std(x_np):.6f}")
        print(f"   Input format: [Pd_nonzero, Qd_nonzero] / baseMVA (normalized)")
        
        print(f"\n4. Output Features:")
        print(f"   Output dimension: {output_dim}")
        
        # Analyze output ranges across all preferences
        all_y_min = []
        all_y_max = []
        all_y_mean = []
        all_y_std = []
        
        # Sample a few preferences for detailed analysis
        sample_prefs = [lambda_carbon_values[0], 
                       lambda_carbon_values[len(lambda_carbon_values)//2],
                       lambda_carbon_values[-1]]
        
        print(f"\n   Output statistics (sampling {len(sample_prefs)} preferences):")
        for lc in sample_prefs:
            if lc in y_train_by_pref:
                y = y_train_by_pref[lc]
                if isinstance(y, torch.Tensor):
                    y_np = y.detach().cpu().numpy()
                else:
                    y_np = y
                all_y_min.append(np.min(y_np))
                all_y_max.append(np.max(y_np))
                all_y_mean.append(np.mean(y_np))
                all_y_std.append(np.std(y_np))
                print(f"     λ_c={lc:.2f}: range=[{np.min(y_np):.6f}, {np.max(y_np):.6f}], "
                      f"mean={np.mean(y_np):.6f}, std={np.std(y_np):.6f}")
        
        # Overall output range
        overall_y_min = min(all_y_min)
        overall_y_max = max(all_y_max)
        
        print(f"\n   Overall output range: [{overall_y_min:.6f}, {overall_y_max:.6f}]")
        
        # Parse output dimensions
        NPred_Va = multi_pref_data.get('NPred_Va', output_dim // 2)
        NPred_Vm = multi_pref_data.get('NPred_Vm', output_dim - NPred_Va)
        
        print(f"   Output format: [Va_nonZIB_noslack ({NPred_Va} dims), Vm_nonZIB ({NPred_Vm} dims)]")
        print(f"   Va unit: radians")
        print(f"   Vm unit: per unit (p.u.)")
        
        # Voltage bounds from config
        VmLb = config.ngt_VmLb
        VmUb = config.ngt_VmUb
        VaLb = config.ngt_VaLb
        VaUb = config.ngt_VaUb
        
        print(f"\n5. Voltage Constraints:")
        print(f"   Vm bounds: [{VmLb:.4f}, {VmUb:.4f}] p.u.")
        print(f"   Va bounds: [{VaLb:.4f}, {VaUb:.4f}] rad "
              f"([{VaLb*180/math.pi:.2f}°, {VaUb*180/math.pi:.2f}°])")
        
        # Calculate voltage span
        Vm_span = VmUb - VmLb
        Va_span = VaUb - VaLb
        
        print(f"   Vm span: {Vm_span:.4f} p.u. ({Vm_span/VmLb*100:.2f}% relative to lower bound)")
        print(f"   Va span: {Va_span:.4f} rad ({Va_span*180/math.pi:.2f}°)")
        
        # Analyze output distribution
        print(f"\n6. Output Distribution Analysis:")
        # Sample one preference for detailed analysis
        sample_lc = lambda_carbon_values[0]
        if sample_lc in y_train_by_pref:
            y_sample = y_train_by_pref[sample_lc]
            if isinstance(y_sample, torch.Tensor):
                y_sample_np = y_sample.detach().cpu().numpy()
            else:
                y_sample_np = y_sample
            
            # Split into Va and Vm parts
            y_va_sample = y_sample_np[:, :NPred_Va]
            y_vm_sample = y_sample_np[:, NPred_Va:]
            
            print(f"   Va (first {NPred_Va} dims):")
            print(f"     Range: [{np.min(y_va_sample):.6f}, {np.max(y_va_sample):.6f}] rad")
            print(f"     Mean: {np.mean(y_va_sample):.6f}, Std: {np.std(y_va_sample):.6f}")
            print(f"     Constraint bounds: [{VaLb:.6f}, {VaUb:.6f}] rad")
            
            print(f"   Vm (last {NPred_Vm} dims):")
            print(f"     Range: [{np.min(y_vm_sample):.6f}, {np.max(y_vm_sample):.6f}] p.u.")
            print(f"     Mean: {np.mean(y_vm_sample):.6f}, Std: {np.std(y_vm_sample):.6f}")
            print(f"     Constraint bounds: [{VmLb:.6f}, {VmUb:.6f}] p.u.")
        
        # Task difficulty assessment
        print(f"\n" + "=" * 80)
        print("Training Task Difficulty Assessment")
        print("=" * 80)
        
        # Calculate difficulty factors
        difficulty_factors = []
        
        # 1. Data volume factor
        total_training_pairs = n_train * n_preferences
        if total_training_pairs < 10000:
            data_volume_difficulty = "High"
            difficulty_factors.append(("Data volume", "High", f"Only {total_training_pairs:,} training pairs"))
        elif total_training_pairs < 100000:
            data_volume_difficulty = "Medium"
            difficulty_factors.append(("Data volume", "Medium", f"{total_training_pairs:,} training pairs"))
        else:
            data_volume_difficulty = "Low"
            difficulty_factors.append(("Data volume", "Low", f"{total_training_pairs:,} training pairs"))
        
        # 2. Output dimension factor
        if output_dim < 100:
            output_dim_difficulty = "Low"
        elif output_dim < 500:
            output_dim_difficulty = "Medium"
        else:
            output_dim_difficulty = "High"
        difficulty_factors.append(("Output dimension", output_dim_difficulty, f"{output_dim} dimensions"))
        
        # 3. Preference diversity factor
        pref_range = lambda_carbon_values[-1] - lambda_carbon_values[0]
        if pref_range < 10:
            pref_diversity_difficulty = "Low"
        elif pref_range < 50:
            pref_diversity_difficulty = "Medium"
        else:
            pref_diversity_difficulty = "High"
        difficulty_factors.append(("Preference diversity", pref_diversity_difficulty, 
                                 f"Range: {pref_range:.2f}, {n_preferences} preferences"))
        
        # 4. Output span factor
        # Va span relative to typical range
        va_typical_range = 2 * math.pi  # Full circle
        va_span_ratio = Va_span / va_typical_range
        
        # Vm span relative to typical range
        vm_typical_range = 0.2  # Typical: 0.9-1.1
        vm_span_ratio = Vm_span / vm_typical_range
        
        if va_span_ratio < 0.1 or vm_span_ratio < 0.1:
            span_difficulty = "Low"
        elif va_span_ratio < 0.3 or vm_span_ratio < 0.3:
            span_difficulty = "Medium"
        else:
            span_difficulty = "High"
        difficulty_factors.append(("Output span", span_difficulty, 
                                   f"Va span: {Va_span*180/math.pi:.2f}°, Vm span: {Vm_span:.4f} p.u."))
        
        # 5. Multi-objective complexity
        if n_preferences > 50:
            mo_complexity = "High"
        elif n_preferences > 20:
            mo_complexity = "Medium"
        else:
            mo_complexity = "Low"
        difficulty_factors.append(("Multi-objective complexity", mo_complexity,
                                   f"{n_preferences} different preference settings"))
        
        # Print difficulty factors
        print("\nDifficulty Factors:")
        for factor, level, detail in difficulty_factors:
            level_symbol = {"Low": "[OK]", "Medium": "[WARN]", "High": "[HIGH]"}[level]
            print(f"  {level_symbol} {factor}: {level} ({detail})")
        
        # Overall assessment
        high_count = sum(1 for _, level, _ in difficulty_factors if level == "High")
        medium_count = sum(1 for _, level, _ in difficulty_factors if level == "Medium")
        
        if high_count >= 3:
            overall_difficulty = "Very High"
        elif high_count >= 2:
            overall_difficulty = "High"
        elif high_count >= 1 or medium_count >= 3:
            overall_difficulty = "Medium-High"
        elif medium_count >= 2:
            overall_difficulty = "Medium"
        else:
            overall_difficulty = "Low-Medium"
        
        print(f"\nOverall Difficulty Assessment: {overall_difficulty}")
        
        # Recommendations
        print(f"\nRecommendations:")
        if data_volume_difficulty == "High":
            print("  - Consider data augmentation or increasing training samples")
        if output_dim_difficulty == "High":
            print("  - Use appropriate regularization (dropout, weight decay)")
            print("  - Consider using Flow/Diffusion models for high-dimensional output")
        if pref_diversity_difficulty == "High":
            print("  - Use preference conditioning (FiLM) for better generalization")
            print("  - Consider curriculum learning (start with fewer preferences)")
        if mo_complexity == "High":
            print("  - Use preference-aware models (Flow with FiLM conditioning)")
            print("  - Consider transfer learning from single-objective model")
        
        print("\n" + "=" * 80)
        
    except FileNotFoundError as e:
        print(f"\n[Error] Multi-preference dataset not found: {e}")
        print("\nTrying to analyze base dataset instead...")
        
        # Fall back to base dataset analysis
        sys_data, _, _ = load_all_data(config)
        ngt_data, sys_data = load_ngt_training_data(config, sys_data=sys_data)
        
        print(f"\nBase Dataset Statistics:")
        print(f"  Training samples: {ngt_data['x_train'].shape[0]}")
        print(f"  Test samples: {ngt_data['x_test'].shape[0]}")
        print(f"  Input dimension: {ngt_data['input_dim']}")
        print(f"  Output dimension: {ngt_data['output_dim']}")
        print(f"\nNote: Multi-preference dataset not available. Run expand_training_data_multi_preference.py first.")
        
    except Exception as e:
        print(f"\n[Error] Failed to analyze dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    analyze_dataset()

