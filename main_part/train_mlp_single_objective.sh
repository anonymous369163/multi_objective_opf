#!/bin/bash
# ==============================================================================
# Single-Objective MLP Training Script
#
# This script trains a single MLP model for single-objective optimization
# (only economic cost, no multi-objective)
#
# Purpose: Train a single-objective model using unsupervised learning
#
# Usage:
#   chmod +x train_mlp_single_objective.sh
#   ./train_mlp_single_objective.sh
#
# ==============================================================================

set -e  # Exit on error

# Navigate to main_part directory (adjust path as needed)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Activate conda environment (adjust path as needed)
# For Windows with miniconda, use:
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate pdp
# For Linux, use:
# source ~/anaconda3/etc/profile.d/conda.sh && conda activate torch_cuda

# Detect OS and set conda path accordingly
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    CONDA_BASE="${HOME}/miniconda3"
    CONDA_ENV="pdp"
else
    # Linux/Mac
    CONDA_BASE="${HOME}/anaconda3"
    CONDA_ENV="torch_cuda"
fi

if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate ${CONDA_ENV}
else
    echo "Warning: conda.sh not found, assuming conda is already activated"
fi

echo "========================================================================"
echo "Single-Objective MLP unsupervised training"
echo "Training MLP model for economic cost optimization only"
echo "========================================================================"

# ============================================================
# Training Configuration
# ============================================================
EPOCHS=4500
FLOW_STEPS=10
HIDDEN_DIM=144
NUM_LAYERS=2
DEBUG=0

# Preference Sampling Configuration (not used in single-objective, but kept for compatibility)
NGT_PREF_SAMPLING="fixed"
NGT_PREF_LEVEL="batch"
NGT_PREF_METHOD="dirichlet"
NGT_PREF_DIRICHLET_ALPHA="0.7"
NGT_PREF_CORNER_PROB="0.1"

# Other configuration
NGT_OBJ_WEIGHT_MULT=1.0
TB_ENABLED=True                    # Set to True if you want TensorBoard logging
NGT_USE_PROJ=False
NGT_FLOW_ZERO_INIT=False

# Create directories
mkdir -p runs
mkdir -p results
mkdir -p saved_models

# ============================================================
# Single-Objective Training
# ============================================================
echo ""
echo "Starting single-objective training..."
echo ""

# Log file for this training
LOG_FILE="results/logs_mlp_single_obj.txt"

# Run training
set +e
CUDA_VISIBLE_DEVICES=0 \
NGT_USE_FLOW=False \
NGT_MULTI_OBJ=False \
NGT_EPOCH=${EPOCHS} \
NGT_FLOW_STEPS=${FLOW_STEPS} \
NGT_FLOW_HIDDEN_DIM=${HIDDEN_DIM} \
NGT_FLOW_NUM_LAYERS=${NUM_LAYERS} \
NGT_FLOW_ZERO_INIT=${NGT_FLOW_ZERO_INIT} \
TB_ENABLED=${TB_ENABLED} \
NGT_USE_PROJ=${NGT_USE_PROJ} \
NGT_OBJ_WEIGHT_MULT=${NGT_OBJ_WEIGHT_MULT} \
NGT_PREF_SAMPLING=${NGT_PREF_SAMPLING} \
NGT_PREF_LEVEL=${NGT_PREF_LEVEL} \
NGT_PREF_METHOD=${NGT_PREF_METHOD} \
NGT_PREF_DIRICHLET_ALPHA=${NGT_PREF_DIRICHLET_ALPHA} \
NGT_PREF_CORNER_PROB=${NGT_PREF_CORNER_PROB} \
NGT_PREF_CONDITIONING=False \
DEBUG=${DEBUG} \
python train_unsupervised.py > "${LOG_FILE}" 2>&1
EXIT_CODE=$?
set -e

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "  ✓ Training completed successfully"
    echo "  Log saved to: ${LOG_FILE}"
else
    echo "  ✗ Training failed! (exit=${EXIT_CODE}) Check log: ${LOG_FILE}"
    exit ${EXIT_CODE}
fi

echo ""
echo "========================================================================"
echo "Training completed!"
echo "========================================================================"
echo ""
echo "Summary:"
echo "  - Trained single-objective MLP model (economic cost only)"
echo "  - Log saved in: ${LOG_FILE}"
echo "  - Model saved in: saved_models/"
echo ""
echo "To view training log:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "To view TensorBoard (if enabled):"
echo "  tensorboard --logdir runs/"
echo ""

