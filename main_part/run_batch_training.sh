#!/bin/bash
# ==============================================================================
# Multi-Preference Batch Training Script
# ==============================================================================
# This script trains multiple DeepOPF-NGT models with different preference weights
# for multi-objective optimization (economic cost vs carbon emission).
#
# Preference combinations (lambda_cost, lambda_carbon):
#   - (1.0, 0.0): Single-objective (cost only)
#   - (0.9, 0.1), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.1, 0.9): Multi-objective
#
# Usage:
#   bash run_batch_training.sh
#
# Models will be saved to: models/NetV_ngt_*bus_E*_final.pth
# ==============================================================================

set -e  # Exit on error

# Change to script directory
cd /home/yuepeng/code/multi_objective_opf/main_part

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch_cuda

echo "============================================================"
echo " Multi-Preference Batch Training for DeepOPF-NGT"
echo "============================================================"
echo ""

# Define preference combinations: lambda_cost values
# lambda_carbon = 1.0 - lambda_cost
LAMBDA_COSTS=(1.0 0.9 0.7 0.5 0.3 0.1)

# Track training progress
TOTAL=${#LAMBDA_COSTS[@]}
CURRENT=0

# Training loop
for lc in "${LAMBDA_COSTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    # Calculate lambda_carbon (use awk for floating point arithmetic)
    le=$(awk "BEGIN {printf \"%.1f\", 1.0 - $lc}")
    
    # Determine if multi-objective mode should be enabled
    if [ "$le" = "0.0" ]; then
        multi_obj="False"
    else
        multi_obj="True"
    fi
    
    echo ""
    echo "============================================================"
    echo " Training Model $CURRENT / $TOTAL"
    echo " lambda_cost = $lc, lambda_carbon = $le"
    echo " multi_objective = $multi_obj"
    echo "============================================================"
    echo ""
    
    # Run training with environment variables
    NGT_MULTI_OBJ=$multi_obj \
    NGT_LAMBDA_COST=$lc \
    NGT_LAMBDA_CARBON=$le \
    python train.py
    
    echo ""
    echo " Model $CURRENT / $TOTAL completed!"
    echo "------------------------------------------------------------"
done

echo ""
echo "============================================================"
echo " All $TOTAL models trained successfully!"
echo "============================================================"
echo ""
echo "Next step: Run evaluation script to compare models and plot Pareto front:"
echo "  python evaluate_multi_preference.py"
echo ""

