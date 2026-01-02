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


def to0_based_index(idx, Nbus):
    """Convert bus indices from 1-based to 0-based if needed."""
    idx = np.asarray(idx, dtype=int).reshape(-1)
    if len(idx) > 0 and idx.min() >= 1 and idx.max() <= Nbus:
        idx = idx - 1
    return idx 

def infer_label_angle_unit(y_va_part: np.ndarray) -> str:
    """Infer the unit of voltage angle labels (radians or degrees)."""
    if y_va_part.size == 0:
        return "rad"
    # 增加阈值到 6.3 (2*pi)，因为弧度制范围通常在 [-pi, pi] 或 [0, 2pi]
    m = float(np.max(np.abs(y_va_part)))
    return "rad" if m <= 6.3 else "deg"

def expand_partial_load_to_full_nodes(x_partial: np.ndarray, case_m_path: str):
    from generate_data.opf_by_pypower import load_case_from_m
    
    ppc = load_case_from_m(case_m_path)
    nbus = ppc["bus"].shape[0]
    
    # 获取基础负荷数据
    Pd_base = ppc["bus"][:, 2]
    Qd_base = ppc["bus"][:, 3]
    
    # 重要：确保这里的逻辑与你保存数据时的逻辑完全锁定
    # 建议：如果可能，直接在数据预处理阶段保存 load_bus_indices 
    has_load = (np.abs(Pd_base) > 1e-6) | (np.abs(Qd_base) > 1e-6)
    load_bus_indices = np.where(has_load)[0]
    n_load_buses = len(load_bus_indices)
    
    x_partial = np.asarray(x_partial)
    original_shape_is_1d = x_partial.ndim == 1
    
    if original_shape_is_1d:
        x_partial = x_partial.reshape(1, -1)
    
    n_samples = x_partial.shape[0]
    
    # 维度校验
    if x_partial.shape[1] != 2 * n_load_buses:
        raise ValueError(f"维度不匹配！输入列数 {x_partial.shape[1]} 并不等于 "
                         f"2 * 识别到的负荷节点数 {n_load_buses}。")

    # 内存优化：直接创建一个完整的大矩阵
    x_full = np.zeros((n_samples, 2 * nbus))
    
    # 填充 Pd 部分 (0 到 nbus-1)
    x_full[:, load_bus_indices] = x_partial[:, :n_load_buses]
    # 填充 Qd 部分 (nbus 到 2*nbus-1)
    x_full[:, nbus + load_bus_indices] = x_partial[:, n_load_buses:]
    
    return x_full[0] if original_shape_is_1d else x_full
def extract_voltage_from_opf_result(result, solver, y_train_reference=None):
    """
    Extract voltage in NGT format from OPF result.
    
    This function ensures consistency with dataset generation format:
    [Va_nonZIB_noslack, Vm_nonZIB]
    
    Args:
        result: OPF result dictionary from PyPowerOPFSolver
        solver: PyPowerOPFSolver instance (must have slack_row attribute)
        y_train_reference: Optional reference y_train for shape validation
    
    Returns:
        y_voltage: [N, output_dim] voltage in NGT format, or None if error
        error_msg: Error message string, or None if successful
    """
    try:
        # Extract voltage from OPF result
        Vm_opf = result["bus"]["Vm"]  # [Nbus]
        Va_opf_rad = result["bus"]["Va_rad"]  # [Nbus] (in radians)
        
        Nbus = len(Vm_opf)
        slack_row = solver.slack_row if hasattr(solver, 'slack_row') else 0
        
        # Extract Va: remove slack bus
        mask = np.ones(Nbus, dtype=bool)
        mask[slack_row] = False
        Va_noslack = Va_opf_rad[mask]
        
        # Extract Vm: all buses (format matches expand_training_data_multi_preference.py)
        Vm_all = Vm_opf
        
        # Combine: [Va_noslack, Vm_all]
        # This matches the format from expand_training_data_multi_preference.py:
        # solutions[i] = np.concatenate([Va_noslack, Vm])
        y_voltage = np.concatenate([Va_noslack, Vm_all])
        
        # Validate shape if reference provided
        if y_train_reference is not None:
            expected_shape = y_train_reference.shape[-1] if y_train_reference.ndim > 0 else len(y_train_reference)
            if len(y_voltage) != expected_shape:
                return None, f"Shape mismatch: got {len(y_voltage)}, expected {expected_shape}"
        
        return y_voltage, None
    
    except Exception as e:
        return None, f"Error extracting voltage: {str(e)}"


def load_preference_files(output_dir: str):
    """
    Load preference solution files from output directory.
    Supports both y_train_lc*.npz and y_train_pref_lc*.npz formats.
    
    Returns:
        pref_data: dict mapping lambda_carbon -> {solutions, success_mask, sample_indices}
        lambda_carbon_values: sorted list of lambda_carbon values
    """
    output_path = Path(output_dir)
    
    # Try both filename patterns
    pref_files = list(output_path.glob("y_train_pref_lc*.npz"))
    if len(pref_files) == 0:
        pref_files = list(output_path.glob("y_train_lc*.npz"))
    
    if len(pref_files) == 0:
        print(f"\nError: No preference solution files found in {output_dir}")
        print(f"  Expected files matching: y_train_pref_lc*.npz or y_train_lc*.npz")
        return None, None
    
    print(f"\nFound {len(pref_files)} preference solution files")
    
    pref_data = {}
    lambda_carbon_values = []
    
    for pref_file in sorted(pref_files):
        try:
            filename = pref_file.name
            # Extract lambda_carbon value from filename
            # Support both: y_train_pref_lc10.00.npz and y_train_lc10.00.npz
            if "y_train_pref_lc" in filename:
                lc_str = filename.replace("y_train_pref_lc", "").replace(".npz", "")
            elif "y_train_lc" in filename:
                lc_str = filename.replace("y_train_lc", "").replace(".npz", "")
            else:
                continue
            
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
    return pref_data, lambda_carbon_values


def filter_lambda_carbon_values(lambda_carbon_values, pref_data, max_lambda=60.0, interval=5.0):
    """
    Filter lambda_carbon values to keep only those in [0, max_lambda] with specified interval.
    
    Args:
        lambda_carbon_values: List of lambda_carbon values
        pref_data: Dictionary mapping lambda_carbon to data
        max_lambda: Maximum lambda_carbon value to keep
        interval: Interval between values (e.g., 5.0 for 0, 5, 10, 15, ...)
    
    Returns:
        filtered_values: Filtered and sorted lambda_carbon values
        filtered_data: Filtered preference data dictionary
    """
    lc_range = [lc for lc in lambda_carbon_values if lc <= max_lambda]
    lc_after = [lc for lc in lambda_carbon_values if lc > max_lambda]
    
    print(f"\nFiltering lambda_carbon values in range [0, {max_lambda}]:")
    print(f"  Total values in [0, {max_lambda}]: {len(lc_range)}")
    print(f"  Original values: {[f'{lc:.0f}' for lc in lc_range]}")
    if len(lc_after) > 0:
        print(f"  Values after {max_lambda} (excluded): {len(lc_after)}")
    
    # Keep only values with specified interval
    filtered_values = []
    filtered_data = {}
    
    for lc in lc_range:
        if lc % interval == 0:  # Values divisible by interval
            filtered_values.append(lc)
            filtered_data[lc] = pref_data[lc]
    
    print(f"  Kept values with interval {interval}: {[f'{lc:.0f}' for lc in filtered_values]}")
    
    if len(lc_after) > 0:
        print(f"\nValues after {max_lambda}: {len(lc_after)} (excluded)")
    
    return sorted(filtered_values), filtered_data


def build_coverage_matrix(pref_data, lambda_carbon_values):
    """
    Build coverage matrix to find samples that have solutions for all preferences.
    
    Returns:
        coverage_matrix: [n_samples, n_preferences] boolean matrix
        base_sample_indices: Array of unique sample indices
    """
    # Collect all unique sample indices from all preferences
    all_sample_indices = set()
    for lc in lambda_carbon_values:
        data = pref_data[lc]
        if data['sample_indices'] is not None:
            all_sample_indices.update(data['sample_indices'])
        else:
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
    base_idx_to_pos = {idx: pos for pos, idx in enumerate(base_sample_indices)}
    coverage_matrix = np.zeros((n_total_samples, len(lambda_carbon_values)), dtype=bool)
    
    for pref_idx, lc in enumerate(lambda_carbon_values):
        data = pref_data[lc]
        success_mask = data['success_mask']
        
        if data['sample_indices'] is not None:
            sample_map = {idx: i for i, idx in enumerate(data['sample_indices'])}
            for pos, sample_idx in enumerate(base_sample_indices):
                if sample_idx in sample_map:
                    mask_pos = sample_map[sample_idx]
                    if mask_pos < len(success_mask):
                        coverage_matrix[pos, pref_idx] = success_mask[mask_pos]
        else:
            # Assume sequential indices
            for pos, sample_idx in enumerate(base_sample_indices):
                if sample_idx < len(success_mask):
                    coverage_matrix[pos, pref_idx] = success_mask[sample_idx]
    
    return coverage_matrix, base_sample_indices


def extract_fully_covered_samples(coverage_matrix, base_sample_indices, x_train_all):
    """
    Extract samples that have solutions for all preferences.
    
    Returns:
        x_train_filtered: Filtered x_train array
        actual_fully_covered_indices: Indices of fully covered samples
    """
    n_successful_per_sample = np.sum(coverage_matrix, axis=1)
    fully_covered_mask = n_successful_per_sample == coverage_matrix.shape[1]
    fully_covered_positions = np.where(fully_covered_mask)[0]
    actual_fully_covered_indices = base_sample_indices[fully_covered_positions]
    
    n_fully_covered = len(actual_fully_covered_indices)
    print(f"\nFully covered samples: {n_fully_covered}/{len(base_sample_indices)} "
          f"({n_fully_covered/len(base_sample_indices)*100:.1f}%)")
    
    if n_fully_covered == 0:
        print("\nError: No fully covered samples found!")
        return None, None
    
    # Extract x_train for fully covered samples
    x_train_filtered_list = []
    x_train_indices_mapped = []
    
    for pos_idx in actual_fully_covered_indices:
        pos_idx_int = int(pos_idx)
        if 0 <= pos_idx_int < len(x_train_all):
            x_train_filtered_list.append(x_train_all[pos_idx_int])
            x_train_indices_mapped.append(pos_idx_int)
        else:
            print(f"  Warning: Position index {pos_idx_int} out of range, skipping")
    
    if len(x_train_filtered_list) == 0:
        print("\nError: No samples could be mapped to x_train!")
        return None, None
    
    x_train_filtered = np.array(x_train_filtered_list)
    print(f"\nFiltered x_train shape: {x_train_filtered.shape}")
    print(f"  Mapped {len(x_train_filtered)} samples from {n_fully_covered} fully covered samples")
    
    return x_train_filtered, actual_fully_covered_indices


def extract_solutions_for_preferences(pref_data, lambda_carbon_values, actual_fully_covered_indices):
    """
    Extract solutions for each preference for fully covered samples.
    
    Returns:
        y_train_by_pref: Dictionary mapping lambda_carbon to solution array
        output_dim: Output dimension
    """
    y_train_by_pref = {}
    output_dim = None
    
    for lc in lambda_carbon_values:
        data = pref_data[lc]
        solutions = data['solutions']
        success_mask = data['success_mask']
        filtered_solutions = []
        
        if data['sample_indices'] is not None:
            sample_map = {idx: i for i, idx in enumerate(data['sample_indices'])}
            for actual_idx in actual_fully_covered_indices:
                if actual_idx in sample_map:
                    sol_idx = sample_map[actual_idx]
                    if sol_idx < len(success_mask) and success_mask[sol_idx] and sol_idx < len(solutions):
                        filtered_solutions.append(solutions[sol_idx])
                    else:
                        print(f"  Error: Sample {actual_idx} failed for λ_c={lc:.2f}")
                        break
        else:
            # Assume sequential indices
            for actual_idx in actual_fully_covered_indices:
                if actual_idx < len(success_mask) and success_mask[actual_idx] and actual_idx < len(solutions):
                    filtered_solutions.append(solutions[actual_idx])
                else:
                    print(f"  Error: Sample {actual_idx} failed for λ_c={lc:.2f}")
                    break
        
        if len(filtered_solutions) > 0:
            y_train_by_pref[lc] = np.array(filtered_solutions)
            if output_dim is None:
                output_dim = y_train_by_pref[lc].shape[1]
            else:
                assert y_train_by_pref[lc].shape[1] == output_dim, \
                    f"Output dimension mismatch: {y_train_by_pref[lc].shape[1]} vs {output_dim}"
        else:
            print(f"  Warning: No solutions extracted for λ_c={lc:.2f}")
    
    return y_train_by_pref, output_dim


def build_fully_covered_dataset():
    """
    Build a dataset containing only samples that have solutions for ALL preferences.
    This dataset will include x_train and y_train for each preference.
    Uses data generated by expand_training_data_multi_preference.py.
    
    Returns:
        dict: Dictionary containing dataset information and file paths
    """
    print("=" * 80)
    print("Building Fully Covered Dataset")
    print("=" * 80)

    output_dir = "saved_data/multi_preference_solutions"
    
    # Load x_train from expand_training_data_multi_preference.py
    x_train_file = Path(output_dir) / "x_train.npz"
    if not x_train_file.exists():
        print(f"\nError: x_train.npz not found in {output_dir}")
        print(f"  Please run expand_training_data_multi_preference.py first to generate the data")
        return None
    
    x_data = np.load(x_train_file, allow_pickle=True)
    if 'x_load_pu' not in x_data:
        print(f"\nError: x_train.npz does not contain 'x_load_pu' key")
        return None
    
    x_train_all = x_data['x_load_pu']
    print(f"\nLoaded x_train from x_train.npz")
    print(f"  x_train shape: {x_train_all.shape}")
    print(f"  Total samples: {len(x_train_all)}")
    
    # Load preference solution files
    pref_data, lambda_carbon_values = load_preference_files(output_dir)
    
    if pref_data is None:
        return None

    print(f"\nLoaded {len(lambda_carbon_values)} preference files")
    if len(lambda_carbon_values) > 0:
        print(f"Lambda carbon range: {lambda_carbon_values[0]:.2f} - {lambda_carbon_values[-1]:.2f}")
        print(f"Lambda carbon values: {[f'{lc:.2f}' for lc in lambda_carbon_values]}")
    
    # Check if lambda=0 exists in preference files
    if 0.0 in pref_data:
        print(f"\nLambda=0 found in preference files")
    else:
        print(f"\nNote: Lambda=0 not found in preference files (will be excluded)")

    # Build coverage matrix to find fully covered samples
    coverage_matrix, base_sample_indices = build_coverage_matrix(pref_data, lambda_carbon_values)
    
    # Extract fully covered samples
    x_train_filtered, actual_fully_covered_indices = extract_fully_covered_samples(
        coverage_matrix, base_sample_indices, x_train_all
    )
    
    if x_train_filtered is None:
        return None
    
    n_fully_covered_final = len(x_train_filtered)

    # Extract solutions for each preference
    y_train_by_pref, output_dim = extract_solutions_for_preferences(
        pref_data, lambda_carbon_values, actual_fully_covered_indices
    )
    
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
        'sample_indices': [int(idx) for idx in actual_fully_covered_indices],
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
    from datetime import datetime
    # 生成训练集的系统时间（仅年月日）
    dataset_generated_date = datetime.now().strftime("%Y-%m-%d")
    output_dataset_file = Path(output_dir) / f"fully_covered_dataset_{dataset_generated_date}.npz"
    np.savez_compressed(output_dataset_file, **dataset)

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

    # Save PyTorch version
    output_torch_file = Path(output_dir) / f"fully_covered_dataset_{dataset_generated_date}.pt"
    torch_dataset = {
        'x_train': torch.from_numpy(x_train_filtered).float(),
        'sample_indices': torch.from_numpy(np.array(dataset['sample_indices'])).long(),
    }
    
    for lc in lambda_carbon_values:
        key = f'y_train_pref_lc_{lc:.2f}'
        torch_dataset[key] = torch.from_numpy(y_train_by_pref[lc]).float()
    
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
    metadata_dict = {
        'n_samples': int(n_fully_covered_final),
        'n_preferences': len(lambda_carbon_values),
        'lambda_carbon_values': [float(x) for x in lambda_carbon_values],
        'output_dim': int(output_dim),
        'input_dim': int(x_train_filtered.shape[1]),
        'sample_indices': dataset['sample_indices'],
        'files': {
            'npz': str(output_dataset_file),
            'pt': str(output_torch_file),
        }
    }

    metadata_json_file = Path(output_dir) / f"fully_covered_dataset_{dataset_generated_date}_metadata.json"
    with open(metadata_json_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

    print(f"Metadata saved to: {metadata_json_file}")

    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"Total unique sample indices: {len(base_sample_indices)}")
    print(f"Fully covered samples: {n_fully_covered_final} "
          f"({n_fully_covered_final/len(base_sample_indices)*100:.1f}%)")
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


def load_and_validate_dataset(dataset_file: str = None, num_test_samples: int = 5, 
                              exclude_lambda_zero: bool = False,
                              test_stability: bool = False, 
                              case_m_path: str = None):
    """
    Load the generated dataset and validate it by comparing with OPF solutions.
    
    Args:
        dataset_file: Path to the dataset file (default: auto-detect)
        num_test_samples: Number of samples to test with OPF
        exclude_lambda_zero: Exclude λ=0 from validation
        test_stability: Test OPF solving stability 
        case_m_path: Path to MATPOWER case file (default: auto-detect)
    
    Returns:
        dict: Validation results
    """
    print("\n" + "=" * 80)
    print("Loading and Validating Dataset")
    print("=" * 80)
    
    # Auto-detect dataset file if not provided
    if dataset_file is None:
         return None 
    
    # Load dataset
    print(f"\n[1/4] Loading dataset from: {dataset_file}")
    data = np.load(dataset_file, allow_pickle=True)
    
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
    
    # Validate dataset structure
    print(f"\n[2/4] Dataset structure validation:")
    print(f"  Output dimension: {output_dim}")
    print(f"  Input dimension: {input_dim}")
    
    # Verify all preferences have same number of samples
    n_samples_per_pref = {lc: y_train_by_pref[lc].shape[0] for lc in y_train_by_pref.keys()}
    if len(set(n_samples_per_pref.values())) > 1:
        print(f"  WARNING: Sample count mismatch across preferences: {n_samples_per_pref}")
    else:
        print(f"  [OK] All preferences have {n_samples} samples")
    
    # [3/4] Validate with OPF solver
    print(f"\n[3/4] Validating with OPF solver ({num_test_samples} samples)")
    
    from generate_data.opf_by_pypower import PyPowerOPFSolver 
    
    print(f"  Using case file: {case_m_path}")
    
    # Initialize OPF solver (no NGT data needed)
    solver_kwargs = {
        'case_m_path': case_m_path,
        'verbose': False,
        'use_multi_objective': True,
    } 
    
    solver = PyPowerOPFSolver(**solver_kwargs)
    
    # Infer output dimensions from solver
    Nbus = solver.nbus
    NPred_Va = Nbus - 1  # Remove slack bus 
    
    # Filter lambda_carbon_values if exclude_lambda_zero is True 
    lambda_carbon_values_to_test = lambda_carbon_values[:min(3, len(lambda_carbon_values))]
    
    # Test samples with different preferences
    test_indices = np.random.choice(n_samples, size=min(num_test_samples, n_samples), replace=False)
    validation_results = [] 
    
    for test_idx in test_indices:
        x_sample = x_train[test_idx] 

        for lc in lambda_carbon_values_to_test:
            if lc not in y_train_by_pref:
                continue
            
            y_sample = y_train_by_pref[lc][test_idx] 
            
            # Solve OPF with full node load data
            result = solver.forward(x_sample, lambda_carbon=lc)
            
            if result["success"]:
                # Extract voltage using consistent function
                y_opf_extracted, error_msg = extract_voltage_from_opf_result(
                    result=result,
                    solver=solver,
                    y_train_reference=y_sample.reshape(1, -1)
                )
                
                if error_msg is not None:
                    print(f"    [ERROR] Failed to extract voltage: {error_msg}")
                    validation_results.append({
                        'sample_idx': int(test_idx),
                        'lambda_carbon': float(lc),
                        'mae': None,
                        'max_error': None,
                        'success': False,
                        'error': error_msg
                    })
                    continue
                
                # Remove batch dimension if present
                if len(y_opf_extracted.shape) > 1:
                    y_opf = y_opf_extracted[0]
                else:
                    y_opf = y_opf_extracted
                
                # Verify dimensions
                if y_opf.shape[0] != y_sample.shape[0]:
                    print(f"    [ERROR] Dimension mismatch: {y_opf.shape[0]} vs {y_sample.shape[0]}")
                    validation_results.append({
                        'sample_idx': int(test_idx),
                        'lambda_carbon': float(lc),
                        'mae': None,
                        'max_error': None,
                        'success': False,
                        'error': f"Dimension mismatch: {y_opf.shape[0]} vs {y_sample.shape[0]}"
                    })
                    continue
                
                # Calculate errors
                error = np.abs(y_opf - y_sample)
                mae = np.mean(error)
                max_error = np.max(error)
                
                # Split into Va and Vm for detailed analysis
                # Infer dimensions: Va is (Nbus-1), Vm is Nbus
                # Total output_dim should be (Nbus-1) + Nbus = 2*Nbus-1
                # So: NPred_Va = (output_dim + 1) // 2 - 1, NPred_Vm = output_dim - NPred_Va
                NPred_Va = (output_dim + 1) // 2 - 1 
                
                va_dataset = y_sample[:NPred_Va]
                vm_dataset = y_sample[NPred_Va:]
                va_opf = y_opf[:NPred_Va]
                vm_opf = y_opf[NPred_Va:]
                
                va_error = np.abs(va_opf - va_dataset)
                vm_error = np.abs(vm_opf - vm_dataset)
                
                va_mae = np.mean(va_error)
                vm_mae = np.mean(vm_error)
                va_max_error = np.max(va_error)
                vm_max_error = np.max(vm_error)
                
                validation_results.append({
                    'sample_idx': int(test_idx),
                    'lambda_carbon': float(lc),
                    'mae': float(mae),
                    'max_error': float(max_error),
                    'va_mae': float(va_mae),
                    'vm_mae': float(vm_mae),
                    'va_max_error': float(va_max_error),
                    'vm_max_error': float(vm_max_error),
                    'success': True
                })
                
                print(f"  Sample {test_idx}, λ_c={lc:.2f}: MAE={mae:.6f}, MaxError={max_error:.6f}")
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
        va_mae_list = [r['va_mae'] for r in successful_validations]
        vm_mae_list = [r['vm_mae'] for r in successful_validations]
        
        print(f"  Successful validations: {len(successful_validations)}/{len(validation_results)}")
        print(f"  Overall MAE: mean={np.mean(mae_list):.6f}, min={np.min(mae_list):.6f}, max={np.max(mae_list):.6f}")
        print(f"  Va MAE: mean={np.mean(va_mae_list):.6f} rad ({np.mean(va_mae_list)*180/np.pi:.4f}°)")
        print(f"  Vm MAE: mean={np.mean(vm_mae_list):.6f} p.u.")
        
        overall_mae_mean = np.mean(mae_list)
        if overall_mae_mean < 0.01:
            print(f"\n  [PASS] Validation PASSED: All errors are excellent")
        elif overall_mae_mean < 0.05:
            print(f"\n  [PASS] Validation PASSED: Errors are within acceptable range")
        else:
            print(f"\n  [WARNING] Validation WARNING: Some errors are larger than expected")
    else:
        print(f"  [FAIL] Validation FAILED: No successful OPF runs")
    
    if failed_validations:
        print(f"  Failed validations: {len(failed_validations)}")
    
    return {
        'dataset_file': str(dataset_file),
        'n_samples': int(n_samples),
        'n_preferences': len(lambda_carbon_values),
        'lambda_carbon_values': [float(x) for x in lambda_carbon_values],
        'validation_results': validation_results,
        'successful_count': len(successful_validations),
        'failed_count': len(failed_validations),
        'exclude_lambda_zero': exclude_lambda_zero, 
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build fully covered dataset with lambda=0 solutions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Why use NGT data?
-----------------
By default, the script uses NGT training data for:
1. x_train: Input load data (600 samples)
2. lambda=0 solutions: Single-objective OPF solutions from NGT y_train
3. Sample index alignment: Ensures all preferences map to the same 600-sample baseline

If --no_ngt_data is set, the script will:
1. Use x_train.npz from expand_training_data_multi_preference.py
2. Use lambda=0 from preference files if available (or skip if not found)
3. Use sample indices from the generated data directly

Note: Using NGT data ensures consistency with the original training set.
        """
    )
    parser.add_argument("--build", action="store_true", help="Build the dataset")
    parser.add_argument("--validate", action="store_true", help="Validate the dataset with OPF")
    parser.add_argument("--num_test_samples", type=int, default=5, help="Number of samples to test in validation")
    parser.add_argument("--dataset_file", type=str, default=None, help="Path to dataset file for validation")
    parser.add_argument("--exclude_lambda_zero", action="store_true", help="Exclude λ=0 from validation")
    parser.add_argument("--test_stability", action="store_true", help="Test OPF solving stability")
    parser.add_argument("--stability_runs", type=int, default=5, help="Number of OPF runs for stability test") 
    parser.add_argument("--case_m", type=str, default="main_part/data/case118_ieee_modified.m", 
                       help="Path to MATPOWER case file (default: auto-detect)")
    
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
            case_m_path=args.case_m
        )
        
        if validation_result is None:
            print("\nError: Failed to validate dataset")
            exit(1)
        
        print("\n" + "=" * 80)
        print("Validation Complete")
        print("=" * 80)
