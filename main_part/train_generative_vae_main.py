"""
Main script to train Generative VAE for Best-of-K sampling.

Usage:
    conda activate pdp_cp
    python main_part/train_generative_vae_main.py

Environment variables (optional):
    GENERATIVE_VAE_EPOCHS=500
    GENERATIVE_VAE_N_SAMPLES=5
    GENERATIVE_VAE_BATCH_SIZE=32
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from main_part.config import get_config
from main_part.data_loader import load_multi_preference_dataset, load_ngt_training_data
from main_part.deepopf_ngt_loss import DeepOPFNGTLoss
from main_part.train_supervised import train_generative_vae


def main():
    """Main training function."""
    print("=" * 70)
    print("Generative VAE Training for Best-of-K Sampling")
    print("=" * 70)
    
    # Load config
    config = get_config()
    device = config.device
    print(f"Device: {device}")
    
    # Override some settings for faster initial testing
    # Comment out these lines for full training
    # config.generative_vae_epochs = 100  # Shorter for testing
    # config.generative_vae_warmup_epochs = 20
    # config.generative_vae_ramp_epochs = 30
    
    # ============================================================
    # Load NGT data (includes sys_data)
    # ============================================================
    # NGT data contains system parameters needed for constraint calculation
    print("\nLoading NGT training data...")
    ngt_data, sys_data = load_ngt_training_data(config)
    
    # ============================================================
    # Load multi-preference dataset
    # ============================================================
    # IMPORTANT: Data format information (verified 2024-12)
    # 
    # 1. Supervised data is in RAW VALUES (NOT normalized):
    #    - y_train: [Va_nonZIB_noslack, Vm_nonZIB]
    #    - Va (voltage angle): in RADIANS, typical range [-0.76, 0.66] rad
    #    - Vm (voltage magnitude): in P.U., range [0.94, 1.06]
    #
    # 2. Input data (x_train = PQd) is also in P.U.:
    #    - Already divided by baseMVA (100 MVA)
    #    - Format: [Pd at bus_Pd, Qd at bus_Qd]
    #
    # 3. Vscale and Vbias are for neural network output layer:
    #    - Used for sigmoid scaling in NetV model
    #    - NOT for normalizing supervised data
    #    - Formula: y_raw = sigmoid(output) * Vscale + Vbias
    #
    # 4. NGT Loss expects RAW VALUES:
    #    - Do NOT apply denormalization before NGT loss
    #    - NGT loss directly uses y_train as-is
    #
    # 5. Data alignment:
    #    - x_train and y_train are aligned via sample_indices
    #    - sample_indices maps to positions in original NGT training data
    #    - ALWAYS use load_multi_preference_dataset() to ensure correct alignment
    #    - Do NOT manually slice or shuffle x_train/y_train separately
    #
    # 6. MAXMIN_Pg (generator limits) are in P.U.:
    #    - Already divided by baseMVA in DeepOPFNGTLoss
    #    - Consistent with Pg computed from power flow equations
    # ============================================================
    print("\nLoading multi-preference dataset...")
    multi_pref_data, _ = load_multi_preference_dataset(config, sys_data)
    print(f"Training samples: {multi_pref_data['n_train']}")
    print(f"Validation samples: {multi_pref_data['n_val']}")
    print(f"Preferences: {len(multi_pref_data['lambda_carbon_values'])}")
    
    # Create NGT loss function
    print("Creating NGT loss function...")
    ngt_loss_fn = DeepOPFNGTLoss(sys_data, config)
    
    # Train generative VAE
    print("\n" + "=" * 70)
    print("Starting Generative VAE Training")
    print("=" * 70)
    
    vae, losses, loss_components = train_generative_vae(
        config=config,
        multi_pref_data=multi_pref_data,
        sys_data=sys_data,
        device=device,
        ngt_loss_fn=ngt_loss_fn
    )
    
    print("\n" + "=" * 70)
    print("Training Completed!")
    print("=" * 70)
    
    # Run validation
    print("\nRunning validation...")
    os.system(f"{sys.executable} {os.path.join(os.path.dirname(__file__), 'validate_generative_vae.py')}")


if __name__ == '__main__':
    main()
