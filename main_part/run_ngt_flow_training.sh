#!/bin/bash
# ==============================================================================
# NGT Rectified Flow Model Training Script
# 
# This script trains the NGT Flow model with different preference weights,
# matching the preference combinations used in run_batch_training.sh (MLP training).
#
# Supports two training modes:
#   1. Independent training: Each preference is trained from VAE anchor
#   2. Progressive training: Each preference uses previous Flow model as anchor
#
# Preference combinations (lambda_cost, lambda_carbon) - same as MLP:
#   - (1.0, 0.0): Single-objective (cost only)
#   - (0.9, 0.1), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.1, 0.9): Multi-objective
#
# Usage:
#   chmod +x run_ngt_flow_training.sh
#   ./run_ngt_flow_training.sh               # Independent training (default)
#   ./run_ngt_flow_training.sh --progressive # Progressive/curriculum training
#
# Environment Variables:
#   TRAINING_MODE=unsupervised - Training mode ('supervised' or 'unsupervised')
#   NGT_USE_FLOW=True          - Enable Flow model (vs MLP)
#   NGT_LAMBDA_COST=0.9        - Economic cost preference weight
#   NGT_LAMBDA_CARBON=0.1      - Carbon emission preference weight
#   NGT_FLOW_STEPS=10          - Number of flow integration steps
#   NGT_USE_PROJ=False         - Use tangent-space projection
#   NGT_FLOW_HIDDEN_DIM=144    - Hidden dimension for flow model
#   NGT_FLOW_NUM_LAYERS=2      - Number of hidden layers
#   NGT_ANCHOR_MODEL_PATH      - Path to anchor Flow model (progressive mode)
#   NGT_ANCHOR_LAMBDA_COST     - Anchor model's lambda_cost (progressive mode)
# ==============================================================================

set -e  # Exit on error

# Navigate to main_part directory
cd /home/yuepeng/code/multi_objective_opf/main_part

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh && conda activate torch_cuda

echo "========================================================================"
echo "NGT Rectified Flow Model Training"
echo "========================================================================"

# Training configuration
FLOW_STEPS=10
# Flow model architecture (tuned to match NetV MLP ~360k params)
# hidden_dim=144, num_layers=2 gives 356,769 params vs NetV's 359,875 (ratio=0.99)
HIDDEN_DIM=144
NUM_LAYERS=2
USE_PROJECTION=True

# Epochs for training
EPOCHS=4500
SAVE_DIR="/home/yuepeng/code/multi_objective_opf/main_part/saved_models"

# Check for progressive training mode
PROGRESSIVE_MODE=false
if [[ "$1" == "--progressive" ]]; then
    PROGRESSIVE_MODE=true
    echo "Training Mode: PROGRESSIVE (curriculum learning)"
    echo "  Each model uses previous model's output as anchor"
else
    echo "Training Mode: INDEPENDENT (VAE anchor for all)"
    echo "  Each model is trained from VAE anchor"
fi

# Define preference weights to train (matching MLP training in run_batch_training.sh)
# lambda_cost from high to low (lambda_carbon = 1 - lambda_cost)
# Includes single-objective (1.0, 0.0) and multi-objective combinations
LAMBDA_COSTS=(1.0 0.9 0.7 0.5 0.3 0.1)

echo ""
echo "Training configuration:"
echo "  Flow integration steps: ${FLOW_STEPS}"
echo "  Hidden dimension: ${HIDDEN_DIM}"
echo "  Number of layers: ${NUM_LAYERS}"
echo "  Use projection: ${USE_PROJECTION}"
echo "  Epochs: ${EPOCHS}"
echo "  Preference weights (lambda_cost): ${LAMBDA_COSTS[@]}"
echo ""

# Train models
PREV_MODEL_PATH=""
PREV_LAMBDA_COST=""

for i in "${!LAMBDA_COSTS[@]}"; do
    lc="${LAMBDA_COSTS[$i]}"
    # Calculate lambda_carbon (1.0 - lambda_cost)
    le=$(awk "BEGIN {printf \"%.1f\", 1.0 - $lc}")
    
    echo ""
    echo "========================================================================"
    echo "Training Flow model [$((i+1))/${#LAMBDA_COSTS[@]}]: lambda_cost=${lc}, lambda_carbon=${le}"
    echo "========================================================================"
    
    # Format lambda_cost for filename
    lc_str=$(echo "$lc" | sed 's/\.//g')
    
    # Determine if multi-objective mode should be enabled (same as run_batch_training.sh)
    if [ "$le" = "0.0" ]; then
        multi_obj="False"
        echo "  Mode: Single-objective (cost only)"
    else
        multi_obj="True"
        echo "  Mode: Multi-objective"
    fi
    
    if [[ "$PROGRESSIVE_MODE" == true ]] && [[ -n "$PREV_MODEL_PATH" ]] && [[ -f "$PREV_MODEL_PATH" ]]; then
        # Progressive training: use previous Flow model as anchor
        echo "  Anchor: Previous Flow model (lambda_cost=${PREV_LAMBDA_COST})"
        echo "  Anchor path: ${PREV_MODEL_PATH}"
        
        # Note: NGT_ANCHOR_MODEL_PATH and NGT_ANCHOR_LAMBDA_COST enable progressive training
        # The anchor Flow model's output is DETACHED (no gradient flows back)
        TRAINING_MODE=unsupervised \
        NGT_USE_FLOW=True \
        NGT_MULTI_OBJ=${multi_obj} \
        NGT_LAMBDA_COST=${lc} \
        NGT_LAMBDA_CARBON=${le} \
        NGT_FLOW_STEPS=${FLOW_STEPS} \
        NGT_USE_PROJ=${USE_PROJECTION} \
        NGT_FLOW_HIDDEN_DIM=${HIDDEN_DIM} \
        NGT_FLOW_NUM_LAYERS=${NUM_LAYERS} \
        NGT_ANCHOR_MODEL_PATH="${PREV_MODEL_PATH}" \
        NGT_ANCHOR_LAMBDA_COST="${PREV_LAMBDA_COST}" \
        python train.py
    else
        # Independent training: use VAE as anchor
        echo "  Anchor: VAE (supervised pretrained)"
        
        TRAINING_MODE=unsupervised \
        NGT_USE_FLOW=True \
        NGT_MULTI_OBJ=${multi_obj} \
        NGT_LAMBDA_COST=${lc} \
        NGT_LAMBDA_CARBON=${le} \
        NGT_FLOW_STEPS=${FLOW_STEPS} \
        NGT_USE_PROJ=${USE_PROJECTION} \
        NGT_FLOW_HIDDEN_DIM=${HIDDEN_DIM} \
        NGT_FLOW_NUM_LAYERS=${NUM_LAYERS} \
        python train.py
    fi
    
    # Update previous model path for next iteration (progressive mode)
    PREV_MODEL_PATH="${SAVE_DIR}/NetV_ngt_flow_300bus_lc${lc_str}_E${EPOCHS}_final.pth"
    PREV_LAMBDA_COST="${lc}"
    
    echo ""
    echo "Completed training for lambda_cost=${lc}"
    echo ""
done

echo ""
echo "========================================================================"
echo "All Flow model training completed!"
echo "========================================================================"
echo ""
echo "Trained models saved in: ${SAVE_DIR}/"
echo ""
echo "To compare models:"
echo "  # Direct comparison (single preference)"
echo "  NGT_LAMBDA_COST=0.5 python train.py --compare ngt ngt_flow vae"
echo ""
echo "  # Progressive/chain inference (evaluates preference 0.5 via chain 0.9→0.8→...→0.5)"
echo "  NGT_LAMBDA_COST=0.5 python train.py --compare ngt ngt_flow_progressive vae"
echo ""
echo "Note: Progressive inference uses the chain: VAE → Flow(0.9) → Flow(0.8) → ... → Flow(target)"
echo "      This matches the progressive training mode used during training."
echo ""
