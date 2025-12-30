#!/usr/bin/env python
# coding: utf-8
"""
Quick validation script to check if VAE has linearized the latent space.
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from data_loader import load_multi_preference_dataset
from flow_model.linearized_vae import LinearizedVAE

def main():
    # Load config and data
    config = get_config()
    device = config.device
    print(f'Device: {device}')

    multi_pref_data, sys_data = load_multi_preference_dataset(config)
    input_dim = multi_pref_data['input_dim']
    output_dim = multi_pref_data['output_dim']
    NPred_Va = multi_pref_data.get('NPred_Va', output_dim // 2)

    print(f'Input dim: {input_dim}, Output dim: {output_dim}')

    # Create and load VAE
    vae = LinearizedVAE(
        input_dim=input_dim,
        output_dim=output_dim,
        latent_dim=getattr(config, 'linearized_vae_latent_dim', 32),
        hidden_dim=getattr(config, 'linearized_vae_hidden_dim', 256),
        num_layers=getattr(config, 'linearized_vae_num_layers', 3),
        pref_dim=1,
        NPred_Va=NPred_Va
    ).to(device)

    # Load checkpoint - try different paths
    ckpt_paths = [
        'main_part/saved_models/linearized_vae_epoch1200.pth',
        'saved_models/linearized_vae_epoch1200.pth',
    ]
    
    ckpt_loaded = False
    for ckpt_path in ckpt_paths:
        if os.path.exists(ckpt_path):
            vae.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            print(f'Loaded VAE from: {ckpt_path}')
            ckpt_loaded = True
            break
    
    if not ckpt_loaded:
        print(f"Error: Could not find checkpoint at any of: {ckpt_paths}")
        return
    
    vae.eval()

    # Prepare data
    x_train = multi_pref_data['x_train'].to(device)
    y_by_pref = {lc: y.to(device) for lc, y in multi_pref_data['y_train_by_pref'].items()}
    lambda_values = sorted(multi_pref_data['lambda_carbon_values'])
    lambda_max = max(lambda_values)

    # PCA analysis
    from sklearn.decomposition import PCA
    from scipy.stats import pearsonr

    print('\n' + '='*60)
    print('Validating Latent Space Linearity')
    print('='*60)

    # Analyze multiple samples
    n_train = x_train.shape[0]
    sample_indices = [0, n_train//5, n_train*2//5, n_train*3//5, n_train*4//5]
    sample_indices = [idx for idx in sample_indices if idx < n_train]
    
    pc1_ratios = []
    pc12_ratios = []
    correlations = []

    for sample_idx in sample_indices:
        z_list = []
        with torch.no_grad():
            for lc in lambda_values:
                sol = y_by_pref[lc][sample_idx:sample_idx+1]
                scene = x_train[sample_idx:sample_idx+1]
                pref = torch.tensor([[lc / lambda_max]], device=device, dtype=torch.float32)
                z = vae.encode(scene, sol, pref, use_mean=True)
                z_list.append(z.cpu().numpy().flatten())
        
        z_array = np.array(z_list)
        
        # PCA
        pca = PCA(n_components=min(5, z_array.shape[1]))
        z_pca = pca.fit_transform(z_array)
        explained_var = pca.explained_variance_ratio_
        
        # Correlation between PC1 and lambda
        lambdas_norm = np.array(lambda_values) / lambda_max
        corr, p_val = pearsonr(z_pca[:, 0], lambdas_norm)
        
        pc1_ratios.append(explained_var[0])
        pc12_ratios.append(explained_var[0] + explained_var[1])
        correlations.append(abs(corr))
        
        print(f'Sample {sample_idx:4d}: PC1={explained_var[0]:.1%}, PC1+PC2={explained_var[0]+explained_var[1]:.1%}, |Corr|={abs(corr):.4f}')

    print('\n' + '-'*60)
    print('Summary Statistics:')
    print(f'  PC1 explains: {np.mean(pc1_ratios)*100:.1f}% +/- {np.std(pc1_ratios)*100:.1f}%')
    print(f'  PC1+PC2 explains: {np.mean(pc12_ratios)*100:.1f}% +/- {np.std(pc12_ratios)*100:.1f}%')
    print(f'  |PC1 vs lambda correlation|: {np.mean(correlations):.4f} +/- {np.std(correlations):.4f}')
    print('-'*60)

    # Assessment
    print('\nAssessment:')
    if np.mean(pc1_ratios) > 0.9:
        print('  [PC1] ✅ Excellent! Latent space is approximately 1D (PC1 > 90%)')
    elif np.mean(pc1_ratios) > 0.7:
        print('  [PC1] ⚠️  Good. Latent space is mostly 1D (PC1 > 70%)')
    elif np.mean(pc1_ratios) > 0.5:
        print('  [PC1] ⚠️  Fair. Latent space has some 1D structure (PC1 > 50%)')
    else:
        print('  [PC1] ❌ Poor. Latent space is not well linearized (PC1 < 50%)')

    if np.mean(correlations) > 0.9:
        print('  [Corr] ✅ Excellent! PC1 is highly correlated with lambda (|r| > 0.9)')
    elif np.mean(correlations) > 0.7:
        print('  [Corr] ⚠️  Good. PC1 is moderately correlated with lambda (|r| > 0.7)')
    else:
        print('  [Corr] ❌ Poor. PC1 is not well correlated with lambda (|r| < 0.7)')
    
    print('='*60)


if __name__ == '__main__':
    main()
