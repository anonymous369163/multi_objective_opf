#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a dataset containing only samples that have solutions for ALL preferences.
This dataset will include x_train and y_train for each preference.

================================================================================
IMPORTANT: Data Format Specification (verified 2024-12)
================================================================================

OUTPUT DATA FORMAT:
-------------------
The generated dataset (fully_covered_dataset.pt/.npz) contains:

1. x_train: [N, 374] - Input load data
   - Format: [Pd at bus_Pd, Qd at bus_Qd] in P.U. (divided by baseMVA=100)
   - Typical range: [-1.14, 10.21] p.u.
   - This is the same format as NGT x_train

2. y_train_pref_lc_*.npz: [N, 465] - Output voltage solutions
   - Format: [Va_nonZIB_noslack, Vm_nonZIB]
   - Va (voltage angle): in RADIANS (NOT degrees!)
     * Typical range: [-0.76, 0.66] rad (i.e., [-43°, 38°])
     * Extracted from OPF result as Va_rad (already in radians)
   - Vm (voltage magnitude): in P.U.
     * Range: [0.94, 1.06] p.u. (per OPF constraints)
   - These are RAW VALUES, NOT normalized!

3. sample_indices: [N] - Maps to positions in NGT training data
   - Used to ensure x_train and y_train alignment
   - Values are 0-based indices into the original 600-sample NGT training set

DATA ALIGNMENT:
---------------
- Each sample in this dataset corresponds to a specific load scenario
- x_train[i] and y_train_pref_lc_*[i] are guaranteed to be aligned
- sample_indices[i] indicates which NGT training sample this corresponds to
- ALWAYS load via load_multi_preference_dataset() to preserve alignment

NORMALIZATION NOTE:
-------------------
- The supervised data (y_train) is NOT normalized
- Vscale and Vbias in NGT data are for neural network output scaling, NOT for data normalization
- When using NGT loss, pass y_train directly without any denormalization
- Formula in neural network: y_raw = sigmoid(output) * Vscale + Vbias

UNIT SUMMARY:
-------------
| Data Item      | Unit      | Typical Range       |
|----------------|-----------|---------------------|
| x_train (PQd)  | p.u.      | [-1.14, 10.21]      |
| y_train (Va)   | radians   | [-0.76, 0.66]       |
| y_train (Vm)   | p.u.      | [0.94, 1.06]        |
| MAXMIN_Pg      | p.u.      | [0.00, 24.65]       |

================================================================================
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_part.config import get_config
from main_part.data_loader import load_ngt_training_data, load_all_data


def to0_based_index(idx, Nbus):
    """
    Convert bus indices from 1-based to 0-based if needed.
    
    Args:
        idx: Bus indices (may be 1-based or 0-based)
        Nbus: Total number of buses
    
    Returns:
        0-based bus indices
    """
    idx = np.asarray(idx, dtype=int).reshape(-1)
    if len(idx) > 0 and idx.min() >= 1 and idx.max() <= Nbus:
        # Likely 1-based, convert to 0-based
        idx = idx - 1
    return idx


def infer_label_angle_unit(y_va_part: np.ndarray) -> str:
    """
    Infer the unit of voltage angle labels (radians or degrees).
    
    Heuristic: if max absolute value <= 3.5, likely radians; else degrees.
    
    Args:
        y_va_part: Voltage angle values
    
    Returns:
        "rad" or "deg"
    """
    if len(y_va_part) == 0:
        return "rad"
    m = float(np.max(np.abs(y_va_part)))
    return "rad" if m <= 3.5 else "deg"

def build_fully_covered_dataset():
    """
    Build a dataset containing only samples that have solutions for ALL preferences.
    This dataset will include x_train and y_train for each preference, including lambda=0.
    
    Returns:
        dict: Dictionary containing dataset information and file paths
    """
    print("=" * 80)
    print("Building Fully Covered Dataset")
    print("=" * 80)

    # Load original training data
    config = get_config()
    ngt_data, sys_data = load_ngt_training_data(config, sys_data=None)

    x_train_all = ngt_data["x_train"].detach().cpu().numpy()  # All 600 samples
    print(f"\nOriginal training data:")
    print(f"  x_train shape: {x_train_all.shape}")
    print(f"  Total samples: {len(x_train_all)}")

    # Load single-objective training data (lambda=0) for conversion
    print(f"\n{'='*80}")
    print("Loading Single-Objective Training Data (lambda=0)")
    print(f"{'='*80}")
    sys_data_single, _, _ = load_all_data(config)
    print(f"  Loaded single-objective training data")
    print(f"  Vm shape: {sys_data_single.yvm_train.shape}")
    print(f"  Va shape: {sys_data_single.yva_train.shape}")

    # Load preference solution files
    output_dir = "saved_data/multi_preference_solutions"
    pref_files = sorted(Path(output_dir).glob("y_train_pref_lc*.npz"))

    if len(pref_files) == 0:
        print(f"\nError: No preference solution files found in {output_dir}")
        return None

    print(f"\nFound {len(pref_files)} preference solution files")

    # Extract lambda_carbon values and load data
    pref_data = {}
    lambda_carbon_values = []

    for pref_file in pref_files:
        try:
            filename = pref_file.name
            lc_str = filename.replace("y_train_pref_lc", "").replace(".npz", "")
            lambda_carbon = float(lc_str)
            lambda_carbon_values.append(lambda_carbon)
            
            data = np.load(pref_file, allow_pickle=True)
            if 'success_mask' not in data or 'solutions' not in data:
                print(f"  Warning: {filename} missing required keys, skipping")
                continue
                
            pref_data[lambda_carbon] = {
                'solutions': data['solutions'],
                'success_mask': data['success_mask'],
                'sample_indices': data.get('sample_indices', None),
            }
        except Exception as e:
            print(f"  Warning: Error loading {filename}: {str(e)[:100]}")
            continue

    lambda_carbon_values = sorted(lambda_carbon_values)
    print(f"\nLoaded {len(lambda_carbon_values)} preference files (before filtering)")
    print(f"Lambda carbon range: {lambda_carbon_values[0]:.2f} - {lambda_carbon_values[-1]:.2f}")

    # Filter lambda_carbon values: for 0-60 range, keep only values with interval 5 (0, 5, 10, 15, ..., 60)
    # Values after 60 are excluded
    lambda_carbon_filtered = []
    pref_data_filtered = {}

    # Separate values into [0, 60] range and after 60
    lc_range_0_60 = [lc for lc in lambda_carbon_values if lc <= 60.0]
    lc_after_60 = [lc for lc in lambda_carbon_values if lc > 60.0]

    print(f"\nFiltering lambda_carbon values in range [0, 60]:")
    print(f"  Total values in [0, 60]: {len(lc_range_0_60)}")
    print(f"  Original values: {[f'{lc:.0f}' for lc in lc_range_0_60]}")
    if len(lc_after_60) > 0:
        print(f"  Values after 60 (will be excluded): {len(lc_after_60)}")

    # Keep only values with interval 5 in [0, 60] range (0, 5, 10, 15, ..., 60)
    lc_kept_0_60 = []
    for lc in lc_range_0_60:
        # Check if the value is divisible by 5 (0, 5, 10, 15, ..., 60)
        if lc % 5 == 0:  # Values divisible by 5: 0, 5, 10, 15, ..., 60
            lambda_carbon_filtered.append(lc)
            pref_data_filtered[lc] = pref_data[lc]
            lc_kept_0_60.append(lc)

    print(f"  Kept values with interval 5: {[f'{lc:.0f}' for lc in lc_kept_0_60]}")

    # Exclude values after 60
    if len(lc_after_60) > 0:
        print(f"\nValues after 60: {len(lc_after_60)} (excluded)")

    # Sort filtered values to ensure proper order
    lambda_carbon_filtered = sorted(lambda_carbon_filtered)

    # Update variables
    lambda_carbon_values = lambda_carbon_filtered
    pref_data = pref_data_filtered

    print(f"\nAfter filtering: {len(lambda_carbon_values)} preference files")
    print(f"Lambda carbon range: {lambda_carbon_values[0]:.2f} - {lambda_carbon_values[-1]:.2f}")
    print(f"Filtered lambda_carbon values: {[f'{lc:.2f}' for lc in lambda_carbon_values]}")

    # ============================================================
    # Add lambda=0 (single-objective) solutions
    # ============================================================
    print(f"\n{'='*80}")
    print("Adding lambda=0 (Single-Objective) Solutions")
    print(f"{'='*80}")

    # IMPORTANT: Use NGT y_train directly for lambda=0!
    # NGT y_train is already in the correct format: [Va_nonZIB_noslack, Vm_nonZIB]
    # and has exactly 600 samples matching the NGT training set.
    # This ensures lambda=0 has the same sample indexing as other preferences.
    
    y_train_ngt = ngt_data['y_train'].detach().cpu().numpy()  # [600, 465]
    n_ngt_train = y_train_ngt.shape[0]
    
    print(f"  Using NGT y_train directly for lambda=0:")
    print(f"    Shape: {y_train_ngt.shape}")
    print(f"    Number of samples: {n_ngt_train}")

    # Add lambda=0 to preference data
    # Use sequential indices (0, 1, 2, ..., 599) to match other preferences
    lambda_carbon_values.insert(0, 0.0)  # Insert at beginning to keep sorted
    pref_data[0.0] = {
        'solutions': y_train_ngt,
        'success_mask': np.ones(n_ngt_train, dtype=bool),  # All samples are successful
        'sample_indices': np.arange(n_ngt_train),  # Sequential indices (0-599)
    }

    print(f"  Added lambda=0.00 with {n_ngt_train} samples")
    print(f"  Lambda carbon values now: {[f'{lc:.2f}' for lc in lambda_carbon_values]}")

    # Build coverage matrix to find fully covered samples
    # Note: lambda=0 uses NGT training indices, while other preferences may use different indices
    # We need to find the intersection of all sample indices

    # Collect all unique sample indices from all preferences
    all_sample_indices = set()
    for lc in lambda_carbon_values:
        data = pref_data[lc]
        if data['sample_indices'] is not None:
            all_sample_indices.update(data['sample_indices'])
        else:
            # If no sample_indices, assume sequential
            n_samples = len(data['success_mask'])
            all_sample_indices.update(range(n_samples))

    base_sample_indices = np.array(sorted(all_sample_indices))
    n_total_samples = len(base_sample_indices)

    print(f"\n{'='*80}")
    print("Building Coverage Matrix")
    print(f"{'='*80}")
    print(f"  Total unique sample indices: {n_total_samples}")
    print(f"  Number of preferences: {len(lambda_carbon_values)}")

    # Build coverage matrix
    # Map each base_sample_idx to its position in base_sample_indices
    base_idx_to_pos = {idx: pos for pos, idx in enumerate(base_sample_indices)}

    coverage_matrix = np.zeros((n_total_samples, len(lambda_carbon_values)), dtype=bool)

    for pref_idx, lc in enumerate(lambda_carbon_values):
        data = pref_data[lc]
        success_mask = data['success_mask']
        
        if data['sample_indices'] is not None:
            # Create mapping from sample index to position in success_mask
            sample_map = {idx: i for i, idx in enumerate(data['sample_indices'])}
            for pos, sample_idx in enumerate(base_sample_indices):
                if sample_idx in sample_map:
                    mask_pos = sample_map[sample_idx]
                    if mask_pos < len(success_mask):
                        coverage_matrix[pos, pref_idx] = success_mask[mask_pos]
        else:
            # Assume sequential indices: base_sample_indices should match success_mask indices
            for pos, sample_idx in enumerate(base_sample_indices):
                if sample_idx < len(success_mask):
                    coverage_matrix[pos, pref_idx] = success_mask[sample_idx]

    # Find fully covered samples
    n_successful_per_sample = np.sum(coverage_matrix, axis=1)
    fully_covered_mask = n_successful_per_sample == len(lambda_carbon_values)
    fully_covered_positions = np.where(fully_covered_mask)[0]  # Positions in base_sample_indices
    actual_fully_covered_indices = base_sample_indices[fully_covered_positions]  # Actual sample indices

    n_fully_covered = len(actual_fully_covered_indices)
    print(f"\nFully covered samples: {n_fully_covered}/{n_total_samples} ({n_fully_covered/n_total_samples*100:.1f}%)")

    if n_fully_covered == 0:
        print("\nError: No fully covered samples found!")
        return None

    # Extract x_train for fully covered samples
    # IMPORTANT: actual_fully_covered_indices are POSITIONAL indices (0-599) in the NGT training data,
    # NOT the original dataset indices from idx_train_ngt.
    # The multi-preference data was generated using x_train from NGT data directly,
    # so sample_indices in pref_data are positional indices matching x_train_all.
    # Therefore, we directly use these indices to index x_train_all.
    
    x_train_filtered_list = []
    x_train_indices_mapped = []
    for pos_idx in actual_fully_covered_indices:
        pos_idx_int = int(pos_idx)  # Convert from np.int64 to int
        if 0 <= pos_idx_int < len(x_train_all):
            x_train_filtered_list.append(x_train_all[pos_idx_int])
            x_train_indices_mapped.append(pos_idx_int)
        else:
            print(f"  Warning: Position index {pos_idx_int} out of range for x_train_all (len={len(x_train_all)}), skipping")

    if len(x_train_filtered_list) == 0:
        print("\nError: No samples could be mapped to x_train!")
        return None

    x_train_filtered = np.array(x_train_filtered_list)
    n_fully_covered_final = len(x_train_filtered)
    print(f"\nFiltered x_train shape: {x_train_filtered.shape}")
    print(f"  Mapped {n_fully_covered_final} samples from {len(actual_fully_covered_indices)} fully covered samples")

    # Extract solutions for each preference
    y_train_by_pref = {}
    output_dim = None

    for lc in lambda_carbon_values:
        data = pref_data[lc]
        solutions = data['solutions']
        success_mask = data['success_mask']
        
        if data['sample_indices'] is not None:
            sample_map = {idx: i for i, idx in enumerate(data['sample_indices'])}
            # Extract solutions for fully covered samples only
            filtered_solutions = []
            for actual_idx in actual_fully_covered_indices:
                if actual_idx in sample_map:
                    sol_idx = sample_map[actual_idx]
                    if sol_idx < len(success_mask) and success_mask[sol_idx]:
                        if sol_idx < len(solutions):
                            filtered_solutions.append(solutions[sol_idx])
                        else:
                            print(f"  Error: Solution index {sol_idx} out of range for λ_c={lc:.2f}")
                            break
                    else:
                        print(f"  Error: Sample {actual_idx} should be successful for λ_c={lc:.2f} but isn't!")
                        break
            else:
                # All samples extracted successfully
                if len(filtered_solutions) > 0:
                    y_train_by_pref[lc] = np.array(filtered_solutions)
                else:
                    print(f"  Warning: No solutions extracted for λ_c={lc:.2f}")
        else:
            # Assume sequential indices: base_sample_indices should match success_mask indices
            filtered_solutions = []
            for actual_idx in actual_fully_covered_indices:
                if actual_idx < len(success_mask) and success_mask[actual_idx]:
                    if actual_idx < len(solutions):
                        filtered_solutions.append(solutions[actual_idx])
                    else:
                        print(f"  Error: Solution index {actual_idx} out of range for λ_c={lc:.2f}")
                        break
            if len(filtered_solutions) > 0:
                y_train_by_pref[lc] = np.array(filtered_solutions)
        
        if output_dim is None:
            output_dim = y_train_by_pref[lc].shape[1]
        else:
            assert y_train_by_pref[lc].shape[1] == output_dim, \
                f"Output dimension mismatch: {y_train_by_pref[lc].shape[1]} vs {output_dim}"

    print(f"\nExtracted solutions for {len(y_train_by_pref)} preferences")
    print(f"Output dimension: {output_dim}")

    # Verify all solutions have same shape
    print(f"\nVerifying solution shapes:")
    for lc in lambda_carbon_values:
        if lc in y_train_by_pref:
            y_pref = y_train_by_pref[lc]
            print(f"  λ_c={lc:6.2f}: shape={y_pref.shape} (expected: ({n_fully_covered_final}, {output_dim}))")
            assert y_pref.shape[0] == n_fully_covered_final, \
                f"Sample count mismatch for λ_c={lc:.2f}: {y_pref.shape[0]} vs {n_fully_covered_final}"
            assert y_pref.shape[1] == output_dim, \
                f"Output dim mismatch for λ_c={lc:.2f}: {y_pref.shape[1]} vs {output_dim}"
        else:
            print(f"  λ_c={lc:6.2f}: MISSING (no solutions extracted)")

    # Build final dataset dictionary
    dataset = {
        'x_train': x_train_filtered,
        'sample_indices': x_train_indices_mapped,  # Actual sample indices (mapped to NGT training indices)
        'n_samples': n_fully_covered_final,
        'n_preferences': len(lambda_carbon_values),
        'lambda_carbon_values': lambda_carbon_values,
        'output_dim': output_dim,
        'input_dim': x_train_filtered.shape[1],
    }

    # Add y_train for each preference
    for lc in lambda_carbon_values:
        key = f'y_train_pref_lc_{lc:.2f}'
        dataset[key] = y_train_by_pref[lc]

    # Save dataset
    output_dataset_file = Path(output_dir) / "fully_covered_dataset.npz"
    np.savez_compressed(
        output_dataset_file,
        **dataset
    )

    print(f"\n{'='*80}")
    print("Dataset Saved")
    print(f"{'='*80}")
    print(f"\nOutput file: {output_dataset_file}")
    print(f"\nDataset contents:")
    print(f"  x_train: shape {dataset['x_train'].shape}")
    print(f"  sample_indices: {len(dataset['sample_indices'])} indices")
    print(f"  Preferences ({len(lambda_carbon_values)}):")
    for lc in lambda_carbon_values:
        key = f'y_train_pref_lc_{lc:.2f}'
        print(f"    {key}: shape {dataset[key].shape}")

    # Also save a PyTorch version for easy loading
    output_torch_file = Path(output_dir) / "fully_covered_dataset.pt"
    torch_dataset = {
        'x_train': torch.from_numpy(x_train_filtered).float(),
        'sample_indices': torch.from_numpy(np.array(x_train_indices_mapped)).long(),
    }

    # Add all y_train as tensors
    for lc in lambda_carbon_values:
        key = f'y_train_pref_lc_{lc:.2f}'
        torch_dataset[key] = torch.from_numpy(y_train_by_pref[lc]).float()

    # Add metadata
    torch_dataset['metadata'] = {
        'n_samples': n_fully_covered_final,
        'n_preferences': len(lambda_carbon_values),
        'lambda_carbon_values': lambda_carbon_values,
        'output_dim': output_dim,
        'input_dim': x_train_filtered.shape[1],
    }

    torch.save(torch_dataset, output_torch_file)
    print(f"\nPyTorch version saved to: {output_torch_file}")

    # Save metadata as JSON
    # Convert all numpy types to Python native types for JSON serialization
    metadata_dict = {
        'n_samples': int(n_fully_covered_final),
        'n_preferences': len(lambda_carbon_values),
        'lambda_carbon_values': [float(x) for x in lambda_carbon_values],
        'output_dim': int(output_dim),
        'input_dim': int(x_train_filtered.shape[1]),
        'sample_indices': [int(idx) for idx in x_train_indices_mapped],  # Convert numpy int64 to Python int
        'files': {
            'npz': str(output_dataset_file),
            'pt': str(output_torch_file),
        }
    }

    metadata_json_file = Path(output_dir) / "fully_covered_dataset_metadata.json"
    with open(metadata_json_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

    print(f"Metadata saved to: {metadata_json_file}")

    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"Total unique sample indices: {n_total_samples}")
    print(f"Fully covered samples: {n_fully_covered} ({n_fully_covered/n_total_samples*100:.1f}%)")
    print(f"Final dataset samples (mapped to NGT): {n_fully_covered_final}")
    print(f"Number of preferences: {len(lambda_carbon_values)} (including lambda=0)")
    print(f"Output dimension: {output_dim}")
    print(f"\nDataset files:")
    print(f"  NumPy format: {output_dataset_file}")
    print(f"  PyTorch format: {output_torch_file}")
    print(f"  Metadata: {metadata_json_file}")

    print(f"\n{'='*80}")
    print("Dataset Ready for Training!")
    print(f"{'='*80}")
    
    return {
        'dataset_file': output_dataset_file,
        'torch_file': output_torch_file,
        'metadata_file': metadata_json_file,
        'n_samples': n_fully_covered_final,
        'n_preferences': len(lambda_carbon_values),
        'lambda_carbon_values': lambda_carbon_values,
        'output_dim': output_dim,
        'input_dim': x_train_filtered.shape[1],
    }


def load_and_validate_dataset(dataset_file: str = None, ngt_data: dict = None, sys_data=None, 
                              config=None, num_test_samples: int = 5, 
                              exclude_lambda_zero: bool = False,
                              test_stability: bool = False, stability_runs: int = 5,
                              carbon_scale: float = None):
    """
    Load the generated dataset and validate it by comparing with OPF solutions.
    
    Args:
        dataset_file: Path to the dataset file (default: auto-detect)
        ngt_data: NGT data dictionary (for bus indices)
        sys_data: System data object
        config: Configuration object
        num_test_samples: Number of samples to test with OPF
    
    Returns:
        dict: Validation results
    """
    print("\n" + "=" * 80)
    print("Loading and Validating Dataset")
    print("=" * 80)
    
    # Auto-detect dataset file if not provided
    if dataset_file is None:
        output_dir = "saved_data/multi_preference_solutions"
        dataset_file = Path(output_dir) / "fully_covered_dataset.npz"
    
    if not os.path.exists(dataset_file):
        print(f"\nError: Dataset file not found: {dataset_file}")
        return None
    
    # Load dataset
    print(f"\n[1/4] Loading dataset from: {dataset_file}")
    data = np.load(dataset_file, allow_pickle=True)
    
    # Extract metadata
    x_train = data['x_train']
    lambda_carbon_values = data['lambda_carbon_values']
    n_samples = data['n_samples']
    output_dim = data['output_dim']
    input_dim = data['input_dim']
    
    print(f"  Loaded dataset:")
    print(f"    x_train: {x_train.shape}")
    print(f"    n_samples: {n_samples}")
    print(f"    n_preferences: {len(lambda_carbon_values)}")
    print(f"    lambda_carbon_values: {[f'{lc:.2f}' for lc in lambda_carbon_values]}")
    print(f"    output_dim: {output_dim}")
    print(f"    input_dim: {input_dim}")
    
    # Load y_train for each preference
    y_train_by_pref = {}
    for lc in lambda_carbon_values:
        key = f'y_train_pref_lc_{lc:.2f}'
        if key in data:
            y_train_by_pref[lc] = data[key]
            print(f"    {key}: {y_train_by_pref[lc].shape}")
        else:
            print(f"    Warning: {key} not found in dataset")
    
    # Load config and sys_data if not provided
    if config is None:
        config = get_config()
    if ngt_data is None or sys_data is None:
        ngt_data, sys_data = load_ngt_training_data(config, sys_data=None)
    
    # Get bus indices for NGT format
    # Note: bus_Pnet_all is in ngt_data, but bus_Pnet_noslack_all is in sys_data
    # Ensure they are numpy arrays and convert to 0-based if needed
    bus_Pnet_all_raw = np.asarray(ngt_data['bus_Pnet_all'], dtype=int)
    bus_Pnet_noslack_all_raw = np.asarray(sys_data.bus_Pnet_noslack_all, dtype=int)
    bus_slack = int(sys_data.bus_slack)
    
    # Get Nbus for index conversion
    # Nbus can be inferred from config or from OPF solver
    Nbus = config.Nbus if hasattr(config, 'Nbus') else 300  # Default to 300 if not available
    
    # Convert to 0-based indices if needed
    bus_Pnet_all = to0_based_index(bus_Pnet_all_raw, Nbus)
    bus_Pnet_noslack_all = to0_based_index(bus_Pnet_noslack_all_raw, Nbus)
    
    NPred_Va = len(bus_Pnet_noslack_all)
    NPred_Vm = len(bus_Pnet_all)
    
    print(f"\n[2/4] Dataset structure validation:")
    print(f"  NPred_Va: {NPred_Va}, NPred_Vm: {NPred_Vm}, Total: {NPred_Va + NPred_Vm}")
    print(f"  Expected output_dim: {NPred_Va + NPred_Vm}, Got: {output_dim}")
    
    if output_dim != NPred_Va + NPred_Vm:
        print(f"  WARNING: Output dimension mismatch!")
    else:
        print(f"  [OK] Output dimension matches")
    
    # Verify all preferences have same number of samples
    n_samples_per_pref = {lc: y_train_by_pref[lc].shape[0] for lc in y_train_by_pref.keys()}
    if len(set(n_samples_per_pref.values())) > 1:
        print(f"  WARNING: Sample count mismatch across preferences: {n_samples_per_pref}")
    else:
        print(f"  [OK] All preferences have {n_samples} samples")
    
    # [3/4] Validate with OPF solver
    print(f"\n[3/4] Validating with OPF solver ({num_test_samples} samples)")
    
    from generate_data.opf_by_pypower import PyPowerOPFSolver
    
    case_m_path = os.path.join(config.data_path, 'case300_ieee_modified.m')
    if not os.path.exists(case_m_path):
        case_m_path = 'main_part/data/case300_ieee_modified.m'
    
    if not os.path.exists(case_m_path):
        print(f"  Warning: Case file not found: {case_m_path}")
        print(f"  Skipping OPF validation")
        return {
            'dataset_file': dataset_file,
            'n_samples': n_samples,
            'n_preferences': len(lambda_carbon_values),
            'lambda_carbon_values': lambda_carbon_values,
            'validation': 'skipped (case file not found)'
        }
    
    # Initialize OPF solver
    # If carbon_scale is provided, use it; otherwise use default (1.0)
    # This allows testing different carbon_scale values to match dataset generation
    solver_kwargs = {
        'case_m_path': case_m_path,
        'ngt_data': ngt_data,
        'verbose': False,
        'use_multi_objective': True,
        'sys_data': sys_data
    }
    if carbon_scale is not None:
        solver_kwargs['carbon_scale'] = carbon_scale
        print(f"  [INFO] Using carbon_scale={carbon_scale} for validation")
    else:
        print(f"  [INFO] Using default carbon_scale=1.0 for validation")
    
    solver = PyPowerOPFSolver(**solver_kwargs)
    
    # Filter lambda_carbon_values if exclude_lambda_zero is True
    if exclude_lambda_zero:
        lambda_carbon_values_to_test = [lc for lc in lambda_carbon_values if lc > 0]
        print(f"  [Filtered] Excluding λ=0, {len(lambda_carbon_values_to_test)} preferences available")
        
        # Select representative values: 1, 33, 99 (or closest available)
        test_prefs = []
        for target in [1.0, 33.0, 99.0]:
            # Find closest value
            closest = min(lambda_carbon_values_to_test, key=lambda x: abs(x - target))
            if closest not in test_prefs:
                test_prefs.append(closest)
        # Fill remaining slots if needed
        while len(test_prefs) < 3 and len(test_prefs) < len(lambda_carbon_values_to_test):
            for lc in lambda_carbon_values_to_test:
                if lc not in test_prefs:
                    test_prefs.append(lc)
                    break
        lambda_carbon_values_to_test = test_prefs[:3]
        print(f"  [Selected] Testing preferences: {lambda_carbon_values_to_test}")
    else:
        lambda_carbon_values_to_test = lambda_carbon_values[:min(3, len(lambda_carbon_values))]
    
    # Test a few samples with different preferences
    test_indices = np.random.choice(n_samples, size=min(num_test_samples, n_samples), replace=False)
    validation_results = []
    stability_results = []  # For stability testing
    
    for test_idx in test_indices:
        x_sample = x_train[test_idx]
        
        # Test with different preferences
        for lc in lambda_carbon_values_to_test:
            if lc not in y_train_by_pref:
                continue
            
            y_sample = y_train_by_pref[lc][test_idx]
            
            # Use original lambda_carbon value without normalization
            # The solver expects [lambda_cost, lambda_carbon] where:
            # - lambda_cost = 1.0 (fixed)
            # - lambda_carbon = lc (original value, not normalized)
            # This matches the format used in dataset generation (expand_training_data_multi_preference.py)
            preference = [1.0, float(lc)]
            
            # Solve OPF
            result = solver.forward(x_sample, preference=preference)
            
            # Stability test: run OPF multiple times for the same sample and preference
            if test_stability and result["success"]:
                print(f"    [Stability Test] Running {stability_runs} OPF solves for sample {test_idx}, λ_c={lc:.2f}...")
                stability_voltages = []
                
                for run_idx in range(stability_runs):
                    result_stable = solver.forward(x_sample, preference=[1.0, float(lc)])
                    if result_stable["success"]:
                        from generate_data.expand_training_data_multi_preference import extract_voltage_from_opf_result
                        y_opf_stable, _ = extract_voltage_from_opf_result(
                            result_stable, ngt_data, solver, y_train_reference=y_sample.reshape(1, -1)
                        )
                        if y_opf_stable is not None:
                            if len(y_opf_stable.shape) > 1:
                                y_opf_stable = y_opf_stable[0]
                            stability_voltages.append(y_opf_stable)
                
                if len(stability_voltages) >= 2:
                    stability_voltages = np.array(stability_voltages)  # [n_runs, output_dim]
                    va_stable = stability_voltages[:, :NPred_Va]
                    vm_stable = stability_voltages[:, NPred_Va:]
                    
                    # Calculate standard deviation across runs
                    va_std = np.std(va_stable, axis=0)  # [NPred_Va]
                    vm_std = np.std(vm_stable, axis=0)  # [NPred_Vm]
                    
                    va_mean_std = np.mean(va_std)
                    vm_mean_std = np.mean(vm_std)
                    va_max_std = np.max(va_std)
                    vm_max_std = np.max(vm_std)
                    
                    # Calculate range (max - min) across runs
                    va_range = np.max(va_stable, axis=0) - np.min(va_stable, axis=0)  # [NPred_Va]
                    vm_range = np.max(vm_stable, axis=0) - np.min(vm_stable, axis=0)  # [NPred_Vm]
                    
                    va_mean_range = np.mean(va_range)
                    vm_mean_range = np.mean(vm_range)
                    va_max_range = np.max(va_range)
                    vm_max_range = np.max(vm_range)
                    
                    stability_results.append({
                        'sample_idx': int(test_idx),
                        'lambda_carbon': float(lc),
                        'n_runs': len(stability_voltages),
                        'va_mean_std': float(va_mean_std),
                        'va_max_std': float(va_max_std),
                        'vm_mean_std': float(vm_mean_std),
                        'vm_max_std': float(vm_max_std),
                        'va_mean_range': float(va_mean_range),
                        'va_max_range': float(va_max_range),
                        'vm_mean_range': float(vm_mean_range),
                        'vm_max_range': float(vm_max_range),
                    })
                    
                    print(f"      Va: mean_std={va_mean_std:.6f} rad ({va_mean_std*180/np.pi:.4f}°), max_std={va_max_std:.6f} rad, mean_range={va_mean_range:.6f} rad")
                    print(f"      Vm: mean_std={vm_mean_std:.6f} p.u., max_std={vm_max_std:.6f} p.u., mean_range={vm_mean_range:.6f} p.u.")
            
            if result["success"]:
                # Extract voltage from OPF result
                Vm_opf = result["bus"]["Vm"]  # [Nbus]
                Va_opf_rad = result["bus"]["Va_rad"]  # [Nbus] (in radians)
                
                # CRITICAL: Use EXACTLY the same logic as extract_voltage_from_opf_result
                # to ensure consistency with dataset generation
                # This is the key to matching the dataset generation logic
                from generate_data.expand_training_data_multi_preference import extract_voltage_from_opf_result
                
                # Use extract_voltage_from_opf_result to get voltage in the same format as dataset
                # This ensures 100% consistency with dataset generation
                y_opf_extracted, error_msg = extract_voltage_from_opf_result(
                    result=result,
                    ngt_data=ngt_data,
                    solver=solver,
                    y_train_reference=y_sample.reshape(1, -1)  # Add batch dimension
                )
                
                if error_msg is not None:
                    print(f"    [ERROR] Failed to extract voltage using extract_voltage_from_opf_result: {error_msg}")
                    validation_results.append({
                        'sample_idx': int(test_idx),
                        'lambda_carbon': float(lc),
                        'mae': None,
                        'max_error': None,
                        'success': False,
                        'error': f"extract_voltage_from_opf_result failed: {error_msg}"
                    })
                    continue
                
                # y_opf_extracted is already in the correct format: [Va_nonZIB_noslack, Vm_nonZIB]
                # Remove batch dimension if present
                if len(y_opf_extracted.shape) > 1:
                    y_opf = y_opf_extracted[0]
                else:
                    y_opf = y_opf_extracted
                
                # Verify dimensions match
                if y_opf.shape[0] != y_sample.shape[0]:
                    print(f"    [ERROR] Dimension mismatch: y_opf.shape={y_opf.shape}, y_sample.shape={y_sample.shape}")
                    validation_results.append({
                        'sample_idx': int(test_idx),
                        'lambda_carbon': float(lc),
                        'mae': None,
                        'max_error': None,
                        'success': False,
                        'error': f"Dimension mismatch: {y_opf.shape[0]} vs {y_sample.shape[0]}"
                    })
                    continue
                
                # Split into Va and Vm for detailed analysis
                va_dataset = y_sample[:NPred_Va]
                vm_dataset = y_sample[NPred_Va:]
                va_opf = y_opf[:NPred_Va]
                vm_opf = y_opf[NPred_Va:]
                
                # Check Va unit for reporting
                va_unit = infer_label_angle_unit(va_dataset)
                
                # Print Va statistics for debugging
                print(f"    Va dataset stats: min={va_dataset.min():.4f}, max={va_dataset.max():.4f}, unit={va_unit}")
                print(f"    Va OPF stats: min={va_opf.min():.4f}, max={va_opf.max():.4f}, unit={va_unit}")
                
                # Detailed error analysis with per-element breakdown
                va_error = np.abs(va_opf - va_dataset)
                vm_error = np.abs(vm_opf - vm_dataset)
                
                # Find worst errors
                va_worst_idx = np.argmax(va_error)
                vm_worst_idx = np.argmax(vm_error)
                
                print(f"    Worst Va error: idx={va_worst_idx}, "
                      f"dataset={va_dataset[va_worst_idx]:.6f}, opf={va_opf[va_worst_idx]:.6f}, "
                      f"error={va_error[va_worst_idx]:.6f} {va_unit}")
                print(f"    Worst Vm error: idx={vm_worst_idx}, "
                      f"dataset={vm_dataset[vm_worst_idx]:.6f}, opf={vm_opf[vm_worst_idx]:.6f}, "
                      f"error={vm_error[vm_worst_idx]:.6f} p.u.")
                
                # Check if there's a systematic offset
                va_mean_error = np.mean(va_opf - va_dataset)
                vm_mean_error = np.mean(vm_opf - vm_dataset)
                print(f"    Mean error (OPF - Dataset): Va={va_mean_error:.6f} {va_unit}, Vm={vm_mean_error:.6f} p.u.")
                
                # Check if errors are systematic (all positive or all negative)
                va_error_sign = np.sign(va_opf - va_dataset)
                vm_error_sign = np.sign(vm_opf - vm_dataset)
                va_all_same_sign = np.all(va_error_sign == va_error_sign[0]) if len(va_error_sign) > 0 else False
                vm_all_same_sign = np.all(vm_error_sign == vm_error_sign[0]) if len(vm_error_sign) > 0 else False
                
                if va_all_same_sign:
                    print(f"    [WARNING] Va errors are all {('positive' if va_error_sign[0] > 0 else 'negative')} - systematic bias detected!")
                if vm_all_same_sign:
                    print(f"    [WARNING] Vm errors are all {('positive' if vm_error_sign[0] > 0 else 'negative')} - systematic bias detected!")
                
                # Get bus indices for error reporting (reconstruct from extract_voltage_from_opf_result logic)
                # This matches the logic used in extract_voltage_from_opf_result
                slack_row = solver.slack_row
                bus_pred0 = None
                for key in ["bus_Pnet_all", "bus_Pnet", "idx_bus_Pnet_all", "idx_bus_Pnet"]:
                    if key in ngt_data:
                        bus_pred0 = np.array(ngt_data[key]).astype(int).reshape(-1)
                        break
                if bus_pred0 is None:
                    bus_pred0 = np.arange(solver.nbus, dtype=int)
                if len(bus_pred0) > 0 and bus_pred0.min() >= 1 and bus_pred0.max() <= solver.nbus:
                    bus_pred0 = bus_pred0 - 1
                # Crop to match dimension (same as extract_voltage_from_opf_result)
                if len(bus_pred0) != NPred_Vm:
                    bus_pred0 = bus_pred0[:NPred_Vm]
                bus_pred_noslack0 = bus_pred0[bus_pred0 != slack_row]
                
                # Check if errors are correlated with bus index
                if len(va_error) > 1 and len(bus_pred_noslack0) == len(va_error):
                    va_error_by_bus = dict(zip(bus_pred_noslack0, va_error))
                    worst_va_buses = sorted(va_error_by_bus.items(), key=lambda x: x[1], reverse=True)[:3]
                    print(f"    Top 3 worst Va errors by bus: {worst_va_buses}")
                
                if len(vm_error) > 1 and len(bus_pred0) == len(vm_error):
                    vm_error_by_bus = dict(zip(bus_pred0, vm_error))
                    worst_vm_buses = sorted(vm_error_by_bus.items(), key=lambda x: x[1], reverse=True)[:3]
                    print(f"    Top 3 worst Vm errors by bus: {worst_vm_buses}")
                
                va_mae = np.mean(va_error)
                vm_mae = np.mean(vm_error)
                va_max_error = np.max(va_error)
                vm_max_error = np.max(vm_error)
                
                # Overall error (combined)
                error = np.abs(y_opf - y_sample)
                mae = np.mean(error)
                max_error = np.max(error)
                
                # Calculate relative errors (for context)
                va_relative_mae = va_mae / (np.max(np.abs(va_dataset)) + 1e-8) * 100 if len(va_dataset) > 0 else 0
                vm_relative_mae = vm_mae / (np.mean(np.abs(vm_dataset)) + 1e-8) * 100 if len(vm_dataset) > 0 else 0
                
                validation_results.append({
                    'sample_idx': int(test_idx),
                    'lambda_carbon': float(lc),
                    'mae': float(mae),
                    'max_error': float(max_error),
                    'va_mae': float(va_mae),
                    'vm_mae': float(vm_mae),
                    'va_max_error': float(va_max_error),
                    'vm_max_error': float(vm_max_error),
                    'va_relative_mae': float(va_relative_mae),
                    'vm_relative_mae': float(vm_relative_mae),
                    'success': True
                })
                
                # Print detailed error breakdown
                print(f"  Sample {test_idx}, λ_c={lc:.2f}:")
                print(f"    Overall: MAE={mae:.6f}, MaxError={max_error:.6f}")
                print(f"    Va: MAE={va_mae:.6f} {va_unit} ({va_mae*180/np.pi:.4f}° if rad, {va_relative_mae:.2f}% rel), MaxError={va_max_error:.6f}")
                print(f"    Vm: MAE={vm_mae:.6f} p.u. ({vm_relative_mae:.2f}% rel), MaxError={vm_max_error:.6f} p.u.")
            else:
                validation_results.append({
                    'sample_idx': int(test_idx),
                    'lambda_carbon': float(lc),
                    'mae': None,
                    'max_error': None,
                    'success': False,
                    'error': result.get('error', 'OPF failed')
                })
                print(f"  Sample {test_idx}, λ_c={lc:.2f}: OPF FAILED - {result.get('error', 'Unknown error')}")
    
    # [4/4] Summary
    print(f"\n[4/4] Validation Summary")
    print(f"{'='*80}")
    
    successful_validations = [r for r in validation_results if r['success']]
    failed_validations = [r for r in validation_results if not r['success']]
    
    if successful_validations:
        mae_list = [r['mae'] for r in successful_validations]
        max_error_list = [r['max_error'] for r in successful_validations]
        va_mae_list = [r['va_mae'] for r in successful_validations]
        vm_mae_list = [r['vm_mae'] for r in successful_validations]
        va_max_error_list = [r['va_max_error'] for r in successful_validations]
        vm_max_error_list = [r['vm_max_error'] for r in successful_validations]
        va_relative_mae_list = [r['va_relative_mae'] for r in successful_validations]
        vm_relative_mae_list = [r['vm_relative_mae'] for r in successful_validations]
        
        print(f"  Successful validations: {len(successful_validations)}/{len(validation_results)}")
        print(f"\n  Overall Errors:")
        print(f"    MAE: mean={np.mean(mae_list):.6f}, min={np.min(mae_list):.6f}, max={np.max(mae_list):.6f}")
        print(f"    Max Error: mean={np.mean(max_error_list):.6f}, min={np.min(max_error_list):.6f}, max={np.max(max_error_list):.6f}")
        
        print(f"\n  Voltage Angle (Va) Errors:")
        print(f"    MAE: mean={np.mean(va_mae_list):.6f} rad ({np.mean(va_mae_list)*180/np.pi:.4f}°), "
              f"min={np.min(va_mae_list):.6f}, max={np.max(va_mae_list):.6f}")
        print(f"    Max Error: mean={np.mean(va_max_error_list):.6f} rad ({np.mean(va_max_error_list)*180/np.pi:.4f}°), "
              f"min={np.min(va_max_error_list):.6f}, max={np.max(va_max_error_list):.6f}")
        print(f"    Relative MAE: mean={np.mean(va_relative_mae_list):.2f}%, "
              f"min={np.min(va_relative_mae_list):.2f}%, max={np.max(va_relative_mae_list):.2f}%")
        
        print(f"\n  Voltage Magnitude (Vm) Errors:")
        print(f"    MAE: mean={np.mean(vm_mae_list):.6f} p.u., "
              f"min={np.min(vm_mae_list):.6f}, max={np.max(vm_mae_list):.6f}")
        print(f"    Max Error: mean={np.mean(vm_max_error_list):.6f} p.u., "
              f"min={np.min(vm_max_error_list):.6f}, max={np.max(vm_max_error_list):.6f}")
        print(f"    Relative MAE: mean={np.mean(vm_relative_mae_list):.2f}%, "
              f"min={np.min(vm_relative_mae_list):.2f}%, max={np.max(vm_relative_mae_list):.2f}%")
        
        # Assess error acceptability
        # Typical acceptable ranges:
        # - Vm: < 0.01 p.u. (1% of typical range 0.94-1.06) is excellent, < 0.05 p.u. (5%) is acceptable
        # - Va: < 0.01 rad (~0.57°) is excellent, < 0.05 rad (~2.86°) is acceptable
        # - Combined: < 0.01 is excellent, < 0.05 is acceptable
        
        vm_mae_mean = np.mean(vm_mae_list)
        va_mae_mean = np.mean(va_mae_list)
        overall_mae_mean = np.mean(mae_list)
        
        print(f"\n  Error Assessment:")
        print(f"    Vm MAE: {vm_mae_mean:.6f} p.u. - ", end="")
        if vm_mae_mean < 0.01:
            print("[OK] Excellent (< 1% of typical range)")
        elif vm_mae_mean < 0.05:
            print("[OK] Acceptable (< 5% of typical range)")
        else:
            print("[WARNING] Large (> 5% of typical range)")
        
        print(f"    Va MAE: {va_mae_mean:.6f} rad ({va_mae_mean*180/np.pi:.4f} deg) - ", end="")
        if va_mae_mean < 0.01:
            print("[OK] Excellent (< 0.57 deg)")
        elif va_mae_mean < 0.05:
            print("[OK] Acceptable (< 2.86 deg)")
        else:
            print("[WARNING] Large (> 2.86 deg)")
        
        print(f"    Overall MAE: {overall_mae_mean:.6f} - ", end="")
        if overall_mae_mean < 0.01:
            print("[OK] Excellent")
        elif overall_mae_mean < 0.05:
            print("[OK] Acceptable")
        else:
            print("[WARNING] Large (may indicate systematic issues)")
        
        # Final verdict
        if overall_mae_mean < 0.01 and vm_mae_mean < 0.01 and va_mae_mean < 0.01:
            print(f"\n  [PASS] Validation PASSED: All errors are excellent")
        elif overall_mae_mean < 0.05 and vm_mae_mean < 0.05 and va_mae_mean < 0.05:
            print(f"\n  [PASS] Validation PASSED: Errors are within acceptable range")
        else:
            print(f"\n  [WARNING] Validation WARNING: Some errors are larger than expected")
            print(f"     This may indicate:")
            print(f"     - Data format/normalization issues")
            print(f"     - Bus index misalignment")
            print(f"     - Unit conversion problems")
            print(f"     - Numerical precision differences between OPF solver and dataset generation")
    else:
        print(f"  [FAIL] Validation FAILED: No successful OPF runs")
    
    if failed_validations:
        print(f"  Failed validations: {len(failed_validations)}")
        for r in failed_validations[:3]:  # Show first 3 failures
            print(f"    Sample {r['sample_idx']}, λ_c={r['lambda_carbon']:.2f}: {r.get('error', 'Unknown')}")
    
    # Stability test summary
    if test_stability and stability_results:
        print(f"\n[5/5] Stability Test Summary")
        print(f"{'='*80}")
        print(f"  Tested {len(stability_results)} sample-preference combinations")
        print(f"  Each combination ran {stability_runs} OPF solves")
        
        va_mean_std_list = [r['va_mean_std'] for r in stability_results]
        va_max_std_list = [r['va_max_std'] for r in stability_results]
        vm_mean_std_list = [r['vm_mean_std'] for r in stability_results]
        vm_max_std_list = [r['vm_max_std'] for r in stability_results]
        
        va_mean_range_list = [r['va_mean_range'] for r in stability_results]
        va_max_range_list = [r['va_max_range'] for r in stability_results]
        vm_mean_range_list = [r['vm_mean_range'] for r in stability_results]
        vm_max_range_list = [r['vm_max_range'] for r in stability_results]
        
        print(f"\n  Voltage Angle (Va) Stability:")
        print(f"    Mean Std Dev: mean={np.mean(va_mean_std_list):.6f} rad ({np.mean(va_mean_std_list)*180/np.pi:.4f}°), "
              f"max={np.max(va_max_std_list):.6f} rad ({np.max(va_max_std_list)*180/np.pi:.4f}°)")
        print(f"    Mean Range: mean={np.mean(va_mean_range_list):.6f} rad ({np.mean(va_mean_range_list)*180/np.pi:.4f}°), "
              f"max={np.max(va_max_range_list):.6f} rad ({np.max(va_max_range_list)*180/np.pi:.4f}°)")
        
        print(f"\n  Voltage Magnitude (Vm) Stability:")
        print(f"    Mean Std Dev: mean={np.mean(vm_mean_std_list):.6f} p.u., max={np.max(vm_max_std_list):.6f} p.u.")
        print(f"    Mean Range: mean={np.mean(vm_mean_range_list):.6f} p.u., max={np.max(vm_max_range_list):.6f} p.u.")
        
        # Compare stability with validation errors
        if successful_validations:
            va_mae_mean = np.mean([r['va_mae'] for r in successful_validations])
            vm_mae_mean = np.mean([r['vm_mae'] for r in successful_validations])
            
            va_stability_mean = np.mean(va_mean_std_list)
            vm_stability_mean = np.mean(vm_mean_std_list)
            
            print(f"\n  Stability vs Validation Error Comparison:")
            if va_stability_mean > 0:
                print(f"    Va: MAE={va_mae_mean:.6f} rad, Stability_std={va_stability_mean:.6f} rad, "
                      f"Ratio={va_mae_mean/va_stability_mean:.2f}x")
            else:
                print(f"    Va: MAE={va_mae_mean:.6f} rad, Stability_std=0")
            
            if vm_stability_mean > 0:
                print(f"    Vm: MAE={vm_mae_mean:.6f} p.u., Stability_std={vm_stability_mean:.6f} p.u., "
                      f"Ratio={vm_mae_mean/vm_stability_mean:.2f}x")
            else:
                print(f"    Vm: MAE={vm_mae_mean:.6f} p.u., Stability_std=0")
            
            if va_stability_mean > 0:
                if va_mae_mean > va_stability_mean * 5:
                    print(f"    [ANALYSIS] Va validation error is {va_mae_mean/va_stability_mean:.1f}x larger than stability std")
                    print(f"              This suggests systematic differences, not just numerical noise")
                else:
                    print(f"    [ANALYSIS] Va validation error is close to stability std")
                    print(f"              This suggests errors may be due to numerical precision")
            
            if vm_stability_mean > 0:
                if vm_mae_mean > vm_stability_mean * 5:
                    print(f"    [ANALYSIS] Vm validation error is {vm_mae_mean/vm_stability_mean:.1f}x larger than stability std")
                    print(f"              This suggests systematic differences, not just numerical noise")
                else:
                    print(f"    [ANALYSIS] Vm validation error is close to stability std")
                    print(f"              This suggests errors may be due to numerical precision")
    
    return {
        'dataset_file': str(dataset_file),
        'n_samples': int(n_samples),
        'n_preferences': len(lambda_carbon_values),
        'lambda_carbon_values': [float(x) for x in lambda_carbon_values],
        'validation_results': validation_results,
        'successful_count': len(successful_validations),
        'failed_count': len(failed_validations),
        'exclude_lambda_zero': exclude_lambda_zero,
        'stability_results': stability_results if test_stability else None,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build fully covered dataset with lambda=0 solutions")
    parser.add_argument("--build", action="store_true", help="Build the dataset")
    parser.add_argument("--validate", action="store_true", help="Validate the dataset with OPF")
    parser.add_argument("--num_test_samples", type=int, default=5, help="Number of samples to test in validation")
    parser.add_argument("--dataset_file", type=str, default=None, help="Path to dataset file for validation")
    parser.add_argument("--exclude_lambda_zero", action="store_true", help="Exclude λ=0 from validation (test only lc>0)")
    parser.add_argument("--test_stability", action="store_true", help="Test OPF solving stability (run multiple times)")
    parser.add_argument("--stability_runs", type=int, default=5, help="Number of OPF runs for stability test")
    parser.add_argument("--carbon_scale", type=float, default=1.0, help="Carbon scale factor for validation (default: 1.0, try 30.0 to match dataset generation)")
    
    args = parser.parse_args()
    
    # If no flags, do both build and validate
    if not args.build and not args.validate:
        args.build = True
        args.validate = True
    
    if args.build:
        result = build_fully_covered_dataset()
        if result is None:
            print("\nError: Failed to build dataset")
            exit(1)
    
    if args.validate:
        print("\n" + "=" * 80)
        print("Starting Validation")
        print("=" * 80)
        validation_result = load_and_validate_dataset(
            dataset_file=args.dataset_file,
            num_test_samples=args.num_test_samples,
            exclude_lambda_zero=args.exclude_lambda_zero,
            test_stability=args.test_stability,
            stability_runs=args.stability_runs,
            carbon_scale=args.carbon_scale
        )
        
        if validation_result is None:
            print("\nError: Failed to validate dataset")
            exit(1)
        
        print("\n" + "=" * 80)
        print("Validation Complete")
        print("=" * 80)

