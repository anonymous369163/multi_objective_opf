#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expand training dataset with multi-objective OPF solutions for different preferences.

This script:
1. Loads existing training data (load scenarios and economic-only solutions)
2. For each load scenario, solves OPF with different lambda_carbon values (0 to 100)
3. Extracts voltage and phase angle in the same format as y_train
4. Saves solutions for each preference separately (to avoid memory overflow)
5. Maintains index alignment for quick lookup

IMPORTANT NOTES:
- Angle Unit: y_train stores Va in radians (converted from degrees in data_loader.py).
  This script ensures extracted Va matches this format.
- Performance: OPF calculation is slow. For 10,000 samples × 100 preferences = 1M OPF calls,
  serial execution may take days. Consider parallel processing for large datasets.
- Checkpoint: The script saves checkpoints to resume from interruptions.

Usage:
    python expand_training_data_multi_preference.py --case_m main_part/data/case300_ieee_modified.m --lambda_carbon_min 0 --lambda_carbon_max 100 --lambda_carbon_step 1
"""

import os
import sys
import argparse
import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_part.config import get_config
from main_part.data_loader import load_all_data, load_ngt_training_data
from main_part.opf_by_pypower import PyPowerOPFSolver


def _infer_label_angle_unit(y_va_part: np.ndarray) -> str:
    """
    Infer the unit of voltage phase angle from training data.
    
    Heuristic: if max absolute value <= ~3.5, assume radians; otherwise degrees.
    This matches the logic in opf_by_pypower.py
    
    Note: This function is used to detect the unit of the ORIGINAL data before conversion.
    In data_loader.py, degrees are converted to radians, so y_train always contains radians.
    However, we use this to verify consistency.
    
    Args:
        y_va_part: Array of voltage phase angles from training data
    
    Returns:
        "rad" or "deg"
    """
    if len(y_va_part) == 0:
        return "rad"  # Default to radians
    m = float(np.max(np.abs(y_va_part)))
    return "rad" if m <= 3.5 else "deg"


def extract_voltage_from_opf_result(
    result: Dict,
    ngt_data: Dict,
    solver: PyPowerOPFSolver,
    y_train_reference: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[str]]:
    """
    Extract voltage and phase angle from OPF result in the same format as y_train.
    
    IMPORTANT: y_train stores Va in radians (see data_loader.py line 651: RVa * pi/180).
    This function ensures the extracted Va matches this format and dimension.
    
    CRITICAL: This function enforces dimension alignment with y_train_reference to prevent
    shape mismatches. The output dimension must exactly match y_train.shape[1].
    
    Args:
        result: OPF result dictionary from PyPowerOPFSolver.forward()
        ngt_data: Dictionary containing bus indices (bus_Pnet_all, bus_Pnet_noslack_all)
        solver: PyPowerOPFSolver instance
        y_train_reference: REQUIRED reference y_train array for dimension alignment
    
    Returns:
        Tuple of (voltage_array, error_message)
        voltage_array: [Va_nonZIB_noslack, Vm_nonZIB] in same format as y_train (Va in radians)
        error_message: None if success, error string if failed
    """
    # Defensive check: ensure result is a dict with expected structure
    if not isinstance(result, dict) or "success" not in result:
        return None, (f"Unexpected solver output structure: "
                     f"keys={list(result.keys()) if isinstance(result, dict) else type(result)}")
    
    if not result.get("success", False):
        return None, result.get("error", "OPF did not converge")
    
    # y_train_reference is required for dimension alignment
    if y_train_reference is None:
        return None, "y_train_reference is required for dimension alignment"
    
    # Defensive check: ensure bus data exists in result
    if "bus" not in result:
        return None, f"Missing 'bus' key in result. Available keys: {list(result.keys())}"
    
    if not isinstance(result["bus"], dict):
        return None, f"'bus' is not a dict in result. Type: {type(result['bus'])}"
    
    if "Vm" not in result["bus"] or "Va_rad" not in result["bus"]:
        return None, (f"Missing bus keys in result. "
                     f"Available keys: {list(result['bus'].keys())}, "
                     f"Required: ['Vm', 'Va_rad']")
    
    # Extract full bus results
    Vm = result["bus"]["Vm"]  # Voltage magnitude (p.u.)
    Va_rad = result["bus"]["Va_rad"]  # Voltage phase angle (radians)
    
    # Get bus indices for non-ZIB nodes
    nbus = solver.nbus
    slack_row = solver.slack_row  # Use slack_row (matrix row index), not slack_bus_id
    
    # CRITICAL: Infer expected dimensions from y_train_reference
    # This matches the logic in opf_by_pypower.py main() function
    ydim = int(y_train_reference.shape[1])
    n_nonzib = int((ydim + 1) // 2)
    
    # Find bus_Pnet_all (non-ZIB buses, including slack)
    bus_pred0 = None
    for key in ["bus_Pnet_all", "bus_Pnet", "idx_bus_Pnet_all", "idx_bus_Pnet"]:
        if key in ngt_data:
            bus_pred0 = np.array(ngt_data[key]).astype(int).reshape(-1)
            break
    
    if bus_pred0 is None:
        # Fallback: use all buses
        bus_pred0 = np.arange(nbus, dtype=int)
    
    # Convert to 0-based indexing if needed
    if len(bus_pred0) > 0 and bus_pred0.min() >= 1 and bus_pred0.max() <= nbus:
        bus_pred0 = bus_pred0 - 1
    
    # CRITICAL: Crop bus_pred0 to match expected dimension
    # This ensures output dimension matches y_train exactly
    if len(bus_pred0) != n_nonzib:
        n_use = min(len(bus_pred0), n_nonzib)
        bus_pred0 = bus_pred0[:n_use]
        n_nonzib = n_use
    
    # Sanity check: verify slack_row is in bus_pred0 (required for correct Va extraction)
    if slack_row not in set(bus_pred0.tolist()):
        return None, (f"slack_row={slack_row} not in bus_pred0 subset; "
                     f"bus index mapping mismatch. bus_pred0={bus_pred0.tolist()}")
    
    # Remove slack bus from bus_pred0 to get Va indices
    # CRITICAL: Use slack_row (matrix row index), not slack_bus_id - 1
    bus_pred_noslack0 = bus_pred0[bus_pred0 != slack_row]
    
    # Verify angle unit consistency (for warning only, we always use radians)
    angle_unit_detected = "rad"  # Default
    if len(bus_pred_noslack0) > 0 and y_train_reference.shape[0] > 0:
        # Extract Va part from reference (first len(bus_pred_noslack0) elements)
        y_va_ref = y_train_reference[0, :len(bus_pred_noslack0)] if len(y_train_reference.shape) > 1 else y_train_reference[:len(bus_pred_noslack0)]
        angle_unit_detected = _infer_label_angle_unit(y_va_ref)
        # Warn if detected unit doesn't match expected (radians)
        if angle_unit_detected != "rad":
            print(f"  [WARNING] Detected angle unit in y_train: {angle_unit_detected}, "
                  f"but expected radians. Using radians to match data_loader.py conversion.")
    
    # Extract values for non-ZIB nodes using row indices
    Vm_sub = Vm[bus_pred0]  # Vm for non-ZIB buses (including slack)
    
    # Always use radians to match y_train format (data_loader.py converts degrees to radians)
    # Va_rad is already in radians, so we use it directly
    Va_sub_noslack = Va_rad[bus_pred_noslack0]  # Va for non-ZIB buses (excluding slack)
    
    # Combine in same format as y_train: [Va_nonZIB_noslack, Vm_nonZIB]
    voltage_array = np.concatenate([Va_sub_noslack, Vm_sub])
    
    # CRITICAL: Verify output dimension matches y_train
    if voltage_array.shape[0] != ydim:
        return None, (f"Dimension mismatch: extracted voltage has shape {voltage_array.shape[0]}, "
                     f"but y_train expects {ydim}. This indicates a bug in bus index alignment.")
    
    return voltage_array, None


def save_preference_solutions(
    solutions: np.ndarray,
    sample_indices: np.ndarray,
    success_mask: np.ndarray,
    lambda_carbon: float,
    lambda_cost: float,
    output_dir: str,
    ngt_data: Dict
) -> str:
    """
    Save solutions for a specific preference to .npz file.
    
    Args:
        solutions: Array of shape [n_samples, output_dim] containing voltage solutions
        sample_indices: Original sample indices in training data
        success_mask: Boolean array indicating which samples succeeded
        lambda_carbon: Carbon weight used
        lambda_cost: Cost weight used
        output_dir: Directory to save files
        ngt_data: Dictionary containing metadata
    
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with preference info
    # Format: y_train_pref_lc{lambda_carbon:.2f}.npz
    filename = f"y_train_pref_lc{lambda_carbon:.2f}.npz"
    filepath = os.path.join(output_dir, filename)
    
    # Save data
    np.savez_compressed(
        filepath,
        solutions=solutions,  # [n_samples, output_dim]
        sample_indices=sample_indices,  # Original indices in training data
        success_mask=success_mask,  # Boolean array: True if OPF converged
        lambda_carbon=lambda_carbon,
        lambda_cost=lambda_cost,
        output_dim=solutions.shape[1] if solutions.shape[0] > 0 else ngt_data.get('output_dim', 0),
        NPred_Va=ngt_data.get('NPred_Va', 0),
        NPred_Vm=ngt_data.get('NPred_Vm', 0),
    )
    
    return filepath


def load_checkpoint(checkpoint_path: str) -> Dict:
    """Load checkpoint to resume from interruption."""
    if os.path.exists(checkpoint_path):
        checkpoint = np.load(checkpoint_path, allow_pickle=True)
        # Round float values to avoid precision issues
        completed_lambdas = [round(float(x), 6) for x in checkpoint['completed_lambdas'].tolist()] if 'completed_lambdas' in checkpoint else []
        return {
            'completed_lambdas': completed_lambdas,
            'last_lambda_idx': int(checkpoint['last_lambda_idx']) if 'last_lambda_idx' in checkpoint else 0,
            'last_sample_idx': int(checkpoint['last_sample_idx']) if 'last_sample_idx' in checkpoint else 0,
        }
    return {
        'completed_lambdas': [],
        'last_lambda_idx': 0,
        'last_sample_idx': 0,
    }


def save_checkpoint(checkpoint_path: str, completed_lambdas: List[float], 
                   last_lambda_idx: int, last_sample_idx: int):
    """Save checkpoint to resume from interruption."""
    np.savez(
        checkpoint_path,
        completed_lambdas=np.array(completed_lambdas),
        last_lambda_idx=last_lambda_idx,
        last_sample_idx=last_sample_idx,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Expand training dataset with multi-objective OPF solutions"
    )
    parser.add_argument(
        "--case_m",
        type=str,
        default="main_part/data/case300_ieee_modified.m",
        help="Path to MATPOWER case file"
    )
    parser.add_argument(
        "--lambda_carbon_min",
        type=float,
        default=0.0,
        help="Minimum lambda_carbon value (default: 0.0)"
    )
    parser.add_argument(
        "--lambda_carbon_max",
        type=float,
        default=100.0,
        help="Maximum lambda_carbon value (default: 100.0)"
    )
    parser.add_argument(
        "--lambda_carbon_step",
        type=float,
        default=2.0,
        help="Step size for lambda_carbon (default: 1.0)"
    )
    parser.add_argument(
        "--carbon_scale",
        type=float,
        default=1.0,
        help="Carbon scale factor (default: 30.0)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="saved_data/multi_preference_solutions",
        help="Directory to save preference solutions"
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default="saved_data/expand_data_checkpoint.npz",
        help="Path to checkpoint file for resuming (checkpoint saved after each preference completes)"
    )
    parser.add_argument(
        "--start_sample",
        type=int,
        default=0,
        help="Start from this sample index (for resuming, default: 0)"
    )
    parser.add_argument(
        "--end_sample",
        type=int,
        default=None,
        help="End at this sample index (None = all samples, default: None)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Expanding Training Dataset with Multi-Objective OPF Solutions")
    print("=" * 80)
    
    # Load configuration and data
    print("\n[1/4] Loading configuration and training data...")
    config = get_config()
    sys_data, _, _ = load_all_data(config)
    ngt_data, sys_data = load_ngt_training_data(config, sys_data=sys_data)
    
    x_train = ngt_data["x_train"].detach().cpu().numpy()
    y_train = ngt_data["y_train"].detach().cpu().numpy()
    
    n_samples = x_train.shape[0]
    output_dim = y_train.shape[1]
    
    print(f"  Training samples: {n_samples}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Input dimension: {x_train.shape[1]}")
    
    # Determine sample range
    start_idx = args.start_sample
    end_idx = args.end_sample if args.end_sample is not None else n_samples
    n_samples_to_process = end_idx - start_idx
    
    print(f"  Processing samples: {start_idx} to {end_idx-1} ({n_samples_to_process} samples)")
    
    # Generate lambda_carbon values and round to avoid float precision issues
    lambda_carbon_values = np.arange(
        args.lambda_carbon_min,
        args.lambda_carbon_max + args.lambda_carbon_step / 2,  # Include max value
        args.lambda_carbon_step
    )
    lambda_carbon_values = np.round(lambda_carbon_values, 6)  # Round to 6 decimal places
    n_preferences = len(lambda_carbon_values)
    
    print(f"\n[2/4] Preference settings:")
    print(f"  Lambda carbon range: [{args.lambda_carbon_min}, {args.lambda_carbon_max}]")
    print(f"  Step size: {args.lambda_carbon_step}")
    print(f"  Number of preferences: {n_preferences}")
    print(f"  Carbon scale: {args.carbon_scale}")
    
    # Performance warning
    total_opf_calls = n_samples_to_process * n_preferences
    estimated_hours = total_opf_calls * 0.1 / 3600  # Rough estimate: 0.1s per OPF call
    print(f"\n  [PERFORMANCE WARNING]")
    print(f"  Total OPF calls: {total_opf_calls:,}")
    print(f"  Estimated time: ~{estimated_hours:.1f} hours (assuming 0.1s per OPF call)")
    print(f"  Note: This is a serial implementation. Consider parallel processing for large datasets.")
    
    # Initialize OPF solver
    print(f"\n[3/4] Initializing OPF solver...")
    solver = PyPowerOPFSolver(
        case_m_path=args.case_m,
        ngt_data=ngt_data,
        verbose=args.verbose,
        use_multi_objective=True,
        lambda_cost=1.0,  # Will be overridden per preference
        lambda_carbon=0.0,  # Will be overridden per preference
        carbon_scale=args.carbon_scale,
        sys_data=sys_data
    )
    print("  OPF solver initialized")
    
    # Load checkpoint if exists
    checkpoint = load_checkpoint(args.checkpoint_file)
    completed_lambdas = checkpoint['completed_lambdas']
    
    print(f"\n[4/4] Processing samples and preferences...")
    print(f"  Checkpoint: {len(completed_lambdas)} preferences already completed")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each preference
    total_start_time = time.time()
    failed_samples = []
    
    # Filter out already completed preferences (with rounded values for comparison)
    # Convert completed_lambdas to rounded set for fast lookup
    completed_lambdas_rounded = {round(float(x), 6) for x in completed_lambdas}
    remaining_prefs = [(idx, round(float(lc), 6)) for idx, lc in enumerate(lambda_carbon_values) 
                       if round(float(lc), 6) not in completed_lambdas_rounded]
    
    if len(completed_lambdas) > 0:
        print(f"  Skipping {len(completed_lambdas)} already completed preferences")
        print(f"  Remaining: {len(remaining_prefs)} preferences to process")
    
    if len(remaining_prefs) == 0:
        print("\n  All preferences already completed! Nothing to do.")
        return
    
    # Create outer progress bar for preferences
    with tqdm(total=len(remaining_prefs), desc="Preferences", unit="pref", 
              position=0, leave=True, ncols=100, initial=0) as pbar_pref:
        
        # Iterate over remaining preferences only (for consistent progress bar)
        for pref_idx, lambda_carbon in remaining_prefs:
            
            # For multi-objective, lambda_cost and lambda_carbon are independent weights
            # We keep lambda_cost = 1.0 and vary lambda_carbon from 0 to 100
            lambda_cost = 1.0
            
            # Update outer progress bar description (use relative index for remaining prefs)
            remaining_idx = remaining_prefs.index((pref_idx, lambda_carbon)) + 1
            pbar_pref.set_description(f"Preference {remaining_idx}/{len(remaining_prefs)} (λ_c={lambda_carbon:.2f}, orig_idx={pref_idx})")
            
            # Initialize solution array for this preference
            solutions = np.zeros((n_samples_to_process, output_dim), dtype=np.float32)
            success_mask = np.zeros(n_samples_to_process, dtype=bool)
            
            pref_start_time = time.time()
            
            # Process each sample with inner progress bar
            with tqdm(range(start_idx, end_idx), desc=f"  Samples (λ_c={lambda_carbon:.2f})", 
                     unit="sample", position=1, leave=False, ncols=100) as pbar_sample:
                
                for sample_idx in pbar_sample:
                    local_idx = sample_idx - start_idx
                    
                    # Load sample
                    x = x_train[sample_idx]
                    
                    # Solve OPF with current preference
                    try:
                        result = solver.forward(x, preference=[lambda_cost, lambda_carbon])
                        
                        # Extract voltage solution (pass y_train as reference to verify unit consistency)
                        voltage_array, error_msg = extract_voltage_from_opf_result(
                            result, ngt_data, solver, y_train_reference=y_train
                        )
                        
                        if voltage_array is not None:
                            solutions[local_idx] = voltage_array
                            success_mask[local_idx] = True
                        else:
                            if args.verbose:
                                pbar_sample.write(f"    Sample {sample_idx}: Failed - {error_msg}")
                            failed_samples.append((sample_idx, lambda_carbon, error_msg))
                    except Exception as e:
                        if args.verbose:
                            pbar_sample.write(f"    Sample {sample_idx}: Exception - {str(e)}")
                        failed_samples.append((sample_idx, lambda_carbon, str(e)))
                    
                    # Update inner progress bar with success rate
                    n_success_so_far = np.sum(success_mask[:local_idx+1])
                    pbar_sample.set_postfix({
                        'success': f'{n_success_so_far}/{local_idx+1}',
                        'rate': f'{(local_idx+1)/(time.time()-pref_start_time):.1f}/s' if time.time() > pref_start_time else '0.0/s'
                    })
            
            # Save solutions for this preference
            sample_indices = np.arange(start_idx, end_idx)
            filepath = save_preference_solutions(
                solutions,
                sample_indices,
                success_mask,
                lambda_carbon,
                lambda_cost,
                args.output_dir,
                ngt_data
            )
            
            # Update checkpoint (with rounded lambda_carbon to avoid precision issues)
            completed_lambdas.append(round(float(lambda_carbon), 6))
            completed_lambdas = sorted(set(round(float(x), 6) for x in completed_lambdas))
            save_checkpoint(
                args.checkpoint_file,
                completed_lambdas,
                pref_idx,
                end_idx - 1
            )
            
            # Statistics
            n_success = np.sum(success_mask)
            n_failed = n_samples_to_process - n_success
            pref_elapsed = time.time() - pref_start_time
            
            # Update outer progress bar with statistics
            pbar_pref.set_postfix({
                'success': f'{n_success}/{n_samples_to_process}',
                'time': f'{pref_elapsed:.1f}s',
                'file': os.path.basename(filepath)
            })
            
            # Print summary for this preference
            tqdm.write(f"  Completed: {n_success}/{n_samples_to_process} successful "
                      f"({n_success/n_samples_to_process*100:.1f}%), "
                      f"{n_failed} failed")
            tqdm.write(f"  Time: {pref_elapsed:.1f}s ({pref_elapsed/n_samples_to_process:.2f}s per sample)")
            tqdm.write(f"  Saved to: {filepath}")
            
            # Update outer progress bar
            pbar_pref.update(1)
    
    # Final summary
    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total time: {total_elapsed/3600:.2f} hours ({total_elapsed:.1f} seconds)")
    print(f"Preferences completed: {len(completed_lambdas)}/{n_preferences}")
    print(f"Total samples processed: {n_samples_to_process * len(remaining_prefs)}")
    print(f"Failed samples: {len(failed_samples)}")
    
    if failed_samples:
        print(f"\nFailed samples (first 10):")
        for sample_idx, lambda_carbon, error in failed_samples[:10]:
            print(f"  Sample {sample_idx}, lambda_carbon={lambda_carbon:.2f}: {error}")
        if len(failed_samples) > 10:
            print(f"  ... and {len(failed_samples) - 10} more")
    
    # Save metadata
    metadata = {
        'n_samples': n_samples_to_process,
        'start_sample': start_idx,
        'end_sample': end_idx,
        'output_dim': output_dim,
        'lambda_carbon_min': args.lambda_carbon_min,
        'lambda_carbon_max': args.lambda_carbon_max,
        'lambda_carbon_step': args.lambda_carbon_step,
        'lambda_carbon_values': lambda_carbon_values.tolist(),
        'carbon_scale': args.carbon_scale,
        'completed_lambdas': completed_lambdas,
        'n_failed_samples': len(failed_samples),
        'output_dir': args.output_dir,
    }
    
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\nMetadata saved to: {metadata_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

