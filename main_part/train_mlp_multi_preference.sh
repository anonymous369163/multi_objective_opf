#!/bin/bash
# ==============================================================================
# Multi-Preference MLP Training Script
#
# This script trains multiple MLP models with different preference weights
# from lambda_cost=0.1 (lambda_carbon=0.9) to lambda_cost=1.0 (lambda_carbon=0.0)
#
# Purpose: Generate Pareto front by training models with different preferences
#
# Usage:
#   chmod +x train_mlp_multi_preference.sh
#   ./train_mlp_multi_preference.sh
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
echo "Multi-Preference MLP Training"
echo "Training MLP models from λ_cost=0.1 to λ_cost=1.0"
echo "========================================================================"

# ============================================================
# Training Configuration (based on launch.json)
# ============================================================
EPOCHS=4500
FLOW_STEPS=10
HIDDEN_DIM=144
NUM_LAYERS=2
DEBUG=0

# Preference Sampling Configuration
NGT_PREF_SAMPLING="fixed"           # Fixed preference for each training
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
# Training Loop: λ_cost from 0.1 to 1.0 (step 0.1)
# ============================================================
echo ""
echo "Starting training loop..."
echo ""

# Array to store process IDs (if running in parallel)
PIDS=()

# Lambda cost and carbon pairs (0.1 to 1.0, step 0.1)
# Format: "lambda_cost:lambda_carbon"
#错开训练顺序，配合杰西莱机器优先训练两端
LAMBDA_PAIRS=(
    "0.1:0.9"
    "0.9:0.1"
    "0.2:0.8"
    "0.8:0.2"
    "0.3:0.7"
    "0.7:0.3"
    "0.4:0.6"
    "0.6:0.4"
    "0.5:0.5"
)

# Loop through lambda pairs
TOTAL=${#LAMBDA_PAIRS[@]}
for i in $(seq 0 $((TOTAL - 1))); do
    # Parse lambda_cost and lambda_carbon from pair
    PAIR=${LAMBDA_PAIRS[$i]}
    LAMBDA_COST=$(echo "$PAIR" | cut -d':' -f1)
    LAMBDA_CARBON=$(echo "$PAIR" | cut -d':' -f2)
    
    # Format lambda_cost for filename (remove decimal point)
    LAMBDA_COST_STR=$(echo "$LAMBDA_COST" | sed 's/\.//')
    
    TRAIN_NUM=$((i + 1))
    echo "----------------------------------------------------------------------"
    echo "[Training ${TRAIN_NUM}/${TOTAL}] λ_cost=${LAMBDA_COST}, λ_carbon=${LAMBDA_CARBON}"
    echo "----------------------------------------------------------------------"
    
    # Log file for this training
    LOG_FILE="results/logs_mlp_lc${LAMBDA_COST_STR}.txt"
    
    # Run training 
    set +e
    CUDA_VISIBLE_DEVICES=0 \
    NGT_USE_FLOW=False \
    NGT_MULTI_OBJ=True \
    NGT_LAMBDA_COST=${LAMBDA_COST} \
    NGT_LAMBDA_CARBON=${LAMBDA_CARBON} \
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
        echo "  Continuing with next preference..."
    fi

    
    echo ""
done

echo "========================================================================"
echo "All training completed!"
echo "========================================================================"
echo ""
echo "Summary:"
echo "  - Trained ${TOTAL} MLP models with different preferences"
echo "  - λ_cost range: 0.1 to 1.0 (step 0.1)"
echo "  - Logs saved in: results/"
echo "  - Models saved in: saved_models/"
echo ""
echo "To view training logs:"
echo "  tail -f results/logs_mlp_lc*.txt"
echo ""
echo "To view TensorBoard (if enabled):"
echo "  tensorboard --logdir runs/"
echo ""

