#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a dataset containing only samples that have solutions for ALL preferences.
This dataset will include x_train and y_train for each preference.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_part.config import get_config
from main_part.data_loader import load_ngt_training_data

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

# Load preference solution files
output_dir = "saved_data/multi_preference_solutions"
pref_files = sorted(Path(output_dir).glob("y_train_pref_lc*.npz"))

if len(pref_files) == 0:
    print(f"\nError: No preference solution files found in {output_dir}")
    exit(1)

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
print(f"\nLoaded {len(lambda_carbon_values)} preference files")
print(f"Lambda carbon range: {lambda_carbon_values[0]:.2f} - {lambda_carbon_values[-1]:.2f}")

# Build coverage matrix to find fully covered samples
first_lc = lambda_carbon_values[0]
first_data = pref_data[first_lc]

# Get sample indices
if first_data['sample_indices'] is not None:
    base_sample_indices = first_data['sample_indices']
else:
    n_total = len(first_data['success_mask'])
    base_sample_indices = np.arange(n_total)

n_total_samples = len(base_sample_indices)

# Build coverage matrix
coverage_matrix = np.zeros((n_total_samples, len(lambda_carbon_values)), dtype=bool)

for pref_idx, lc in enumerate(lambda_carbon_values):
    data = pref_data[lc]
    success_mask = data['success_mask']
    
    if data['sample_indices'] is not None:
        sample_map = {idx: i for i, idx in enumerate(data['sample_indices'])}
        for sample_idx in base_sample_indices:
            if sample_idx in sample_map:
                coverage_matrix[sample_idx, pref_idx] = success_mask[sample_map[sample_idx]]
    else:
        coverage_matrix[:, pref_idx] = success_mask

# Find fully covered samples
n_successful_per_sample = np.sum(coverage_matrix, axis=1)
fully_covered_mask = n_successful_per_sample == len(lambda_carbon_values)
fully_covered_indices = base_sample_indices[fully_covered_mask]

n_fully_covered = len(fully_covered_indices)
print(f"\nFully covered samples: {n_fully_covered}/{n_total_samples} ({n_fully_covered/n_total_samples*100:.1f}%)")

if n_fully_covered == 0:
    print("\nError: No fully covered samples found!")
    exit(1)

# Extract x_train for fully covered samples
x_train_filtered = x_train_all[fully_covered_indices]
print(f"\nFiltered x_train shape: {x_train_filtered.shape}")

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
        for sample_idx in fully_covered_indices:
            if sample_idx in sample_map:
                sol_idx = sample_map[sample_idx]
                if success_mask[sol_idx]:
                    filtered_solutions.append(solutions[sol_idx])
                else:
                    print(f"  Error: Sample {sample_idx} should be successful for λ_c={lc:.2f} but isn't!")
                    break
        else:
            # All samples extracted successfully
            y_train_by_pref[lc] = np.array(filtered_solutions)
    else:
        # Assume sequential indices
        y_train_by_pref[lc] = solutions[fully_covered_mask]
    
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
    y_pref = y_train_by_pref[lc]
    print(f"  λ_c={lc:6.2f}: shape={y_pref.shape} (expected: ({n_fully_covered}, {output_dim}))")
    assert y_pref.shape == (n_fully_covered, output_dim), \
        f"Shape mismatch for λ_c={lc:.2f}: {y_pref.shape} vs ({n_fully_covered}, {output_dim})"

# Build final dataset dictionary
dataset = {
    'x_train': x_train_filtered,
    'sample_indices': fully_covered_indices,  # Original indices from training set
    'n_samples': n_fully_covered,
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
    'sample_indices': torch.from_numpy(fully_covered_indices).long(),
}

# Add all y_train as tensors
for lc in lambda_carbon_values:
    key = f'y_train_pref_lc_{lc:.2f}'
    torch_dataset[key] = torch.from_numpy(y_train_by_pref[lc]).float()

# Add metadata
torch_dataset['metadata'] = {
    'n_samples': n_fully_covered,
    'n_preferences': len(lambda_carbon_values),
    'lambda_carbon_values': lambda_carbon_values,
    'output_dim': output_dim,
    'input_dim': x_train_filtered.shape[1],
}

torch.save(torch_dataset, output_torch_file)
print(f"\nPyTorch version saved to: {output_torch_file}")

# Save metadata as JSON
metadata_dict = {
    'n_samples': int(n_fully_covered),
    'n_preferences': len(lambda_carbon_values),
    'lambda_carbon_values': [float(x) for x in lambda_carbon_values],
    'output_dim': int(output_dim),
    'input_dim': int(x_train_filtered.shape[1]),
    'sample_indices': fully_covered_indices.tolist(),
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
print(f"Total samples in original dataset: {n_total_samples}")
print(f"Fully covered samples: {n_fully_covered} ({n_fully_covered/n_total_samples*100:.1f}%)")
print(f"Number of preferences: {len(lambda_carbon_values)}")
print(f"Output dimension: {output_dim}")
print(f"\nDataset files:")
print(f"  NumPy format: {output_dataset_file}")
print(f"  PyTorch format: {output_torch_file}")
print(f"  Metadata: {metadata_json_file}")

print(f"\n{'='*80}")
print("Dataset Ready for Training!")
print(f"{'='*80}")

