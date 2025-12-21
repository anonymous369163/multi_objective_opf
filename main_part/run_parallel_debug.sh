#!/bin/bash
# ==============================================================================
# Parallel Training Script: Flow (GPU0) vs MLP (GPU1) for Fixed Preference
#
# This script trains both models in parallel on separate GPUs with TensorBoard
# logging enabled. This allows direct comparison of training dynamics.
#
# Purpose: Quick validation with fixed preference to compare MLP vs Flow models
#          before testing random preference sampling for Pareto coverage.
#
# Usage:
#   chmod +x run_parallel_debug.sh
#   ./run_parallel_debug.sh
#
# Preference sampling configuration (via environment variables):
#   - NGT_PREF_SAMPLING: "fixed" (for validation) or "random" (for Pareto coverage)
#   - NGT_PREF_LEVEL: "batch" (shared per batch) or "sample" (per sample)
#   - NGT_PREF_METHOD: "dirichlet"|"beta"|"uniform"
#   - NGT_PREF_DIRICHLET_ALPHA: alpha parameter for Dirichlet (<1 favors corners)
#   - NGT_PREF_CORNER_PROB: probability of forcing corner preferences
#
# To test multiple preferences, run this script multiple times with different
# LAMBDA_COST/LAMBDA_CARBON values:
#   - Pure economic: LAMBDA_COST=1.0, LAMBDA_CARBON=0.0
#   - Balanced:      LAMBDA_COST=0.5, LAMBDA_CARBON=0.5
#   - Pure carbon:    LAMBDA_COST=0.0, LAMBDA_CARBON=1.0
#
# TensorBoard:
#   tensorboard --logdir main_part/runs/
#
# ==============================================================================

set -e  # Exit on error

# Navigate to main_part directory
cd /home/yuepeng/code/multi_objective_opf/main_part

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh && conda activate torch_cuda

echo "========================================================================"
echo "Parallel Training: Flow vs MLP (Fixed Preference Validation)"
echo "========================================================================"

# ============================================================
# Training Configuration
# ============================================================
EPOCHS=4500
LAMBDA_COST=0.1
LAMBDA_CARBON=0.9
FLOW_STEPS=10
HIDDEN_DIM=144
NUM_LAYERS=2
DEBUG=0   # 是否直接导入训练好的模型，测试模型的性能，默认不导入

# ============================================================
# Preference Sampling Configuration
# ============================================================
# For fixed preference validation (recommended for initial testing)
NGT_PREF_SAMPLING="fixed"           # "fixed" or "random"
NGT_PREF_LEVEL="batch"               # "batch" or "sample"
NGT_PREF_METHOD="dirichlet"          # "dirichlet"|"beta"|"uniform"
NGT_PREF_DIRICHLET_ALPHA="0.7"       # <1 更偏角点
NGT_PREF_CORNER_PROB="0.1"           # 10% 强制角点

# To enable random sampling for Pareto coverage testing, change:
#   NGT_PREF_SAMPLING="random"

echo ""
echo "Configuration:"
echo "  Epochs: ${EPOCHS}"
echo "  Target preference: λ_cost=${LAMBDA_COST}, λ_carbon=${LAMBDA_CARBON}"
echo "  Flow integration steps: ${FLOW_STEPS}"
echo "  Flow hidden dim: ${HIDDEN_DIM}, layers: ${NUM_LAYERS}"
echo "  Debug mode: ${DEBUG}"
echo ""
echo "Preference Sampling:"
echo "  Sampling mode: ${NGT_PREF_SAMPLING}"
echo "  Level: ${NGT_PREF_LEVEL}"
echo "  Method: ${NGT_PREF_METHOD}"
if [ "${NGT_PREF_METHOD}" = "dirichlet" ]; then
    echo "  Dirichlet alpha: ${NGT_PREF_DIRICHLET_ALPHA}"
fi
echo "  Corner probability: ${NGT_PREF_CORNER_PROB}"
echo ""   

# Create runs directory for TensorBoard logs
mkdir -p runs
# Create results directory for training logs
mkdir -p results

# ============================================================
# Start Flow Training on GPU 0 (Background)
# ============================================================
echo "[GPU 0] Starting Flow model training..."
CUDA_DEVICE=0 \
TRAINING_MODE=unsupervised \
NGT_USE_FLOW=True \
NGT_MULTI_OBJ=True \
NGT_LAMBDA_COST=${LAMBDA_COST} \
NGT_LAMBDA_CARBON=${LAMBDA_CARBON} \
NGT_EPOCH=${EPOCHS} \
NGT_FLOW_STEPS=${FLOW_STEPS} \
NGT_FLOW_HIDDEN_DIM=${HIDDEN_DIM} \
NGT_FLOW_NUM_LAYERS=${NUM_LAYERS} \
NGT_FLOW_ZERO_INIT=False \
TB_ENABLED=True \
NGT_USE_PROJ=False \
NGT_OBJ_WEIGHT_MULT=100.0 \
NGT_PREF_SAMPLING=${NGT_PREF_SAMPLING} \
NGT_PREF_LEVEL=${NGT_PREF_LEVEL} \
NGT_PREF_METHOD=${NGT_PREF_METHOD} \
NGT_PREF_DIRICHLET_ALPHA=${NGT_PREF_DIRICHLET_ALPHA} \
NGT_PREF_CORNER_PROB=${NGT_PREF_CORNER_PROB} \
DEBUG=${DEBUG} \
python train.py > results/logs_flow_lc${LAMBDA_COST}_gpu0.txt 2>&1 &
PID_FLOW=$!

echo "  PID: ${PID_FLOW}"
echo "  Log: results/logs_flow_lc${LAMBDA_COST}_gpu0.txt"

# Small delay to avoid resource conflicts
sleep 2

# ============================================================
# Start MLP Training on GPU 1 (Background)
# ============================================================
echo "[GPU 1] Starting MLP model training..."
CUDA_DEVICE=1 \
TRAINING_MODE=unsupervised \
NGT_USE_FLOW=False \
NGT_MULTI_OBJ=True \
NGT_LAMBDA_COST=${LAMBDA_COST} \
NGT_LAMBDA_CARBON=${LAMBDA_CARBON} \
NGT_EPOCH=${EPOCHS} \
TB_ENABLED=True \
NGT_PREF_SAMPLING=${NGT_PREF_SAMPLING} \
NGT_PREF_LEVEL=${NGT_PREF_LEVEL} \
NGT_PREF_METHOD=${NGT_PREF_METHOD} \
NGT_PREF_DIRICHLET_ALPHA=${NGT_PREF_DIRICHLET_ALPHA} \
NGT_PREF_CORNER_PROB=${NGT_PREF_CORNER_PROB} \
DEBUG=${DEBUG} \
python train.py > results/logs_mlp_lc${LAMBDA_COST}_gpu1.txt 2>&1 &
PID_MLP=$!

echo "  PID: ${PID_MLP}"
echo "  Log: results/logs_mlp_lc${LAMBDA_COST}_gpu1.txt"

echo ""
echo "========================================================================"
echo "Both trainings started in background!"
echo "========================================================================"
echo ""
echo "Monitor progress:"
echo "  tail -f results/logs_flow_lc${LAMBDA_COST}_gpu0.txt    # Flow model (GPU 0)"
echo "  tail -f results/logs_mlp_lc${LAMBDA_COST}_gpu1.txt     # MLP model (GPU 1)"
echo ""
echo "TensorBoard:"
echo "  tensorboard --logdir runs/"
echo ""
echo "Waiting for both trainings to complete..."

# Wait for both processes to finish
wait $PID_FLOW
FLOW_EXIT=$?
echo "[GPU 0] Flow training completed (exit code: ${FLOW_EXIT})"

wait $PID_MLP
MLP_EXIT=$?
echo "[GPU 1] MLP training completed (exit code: ${MLP_EXIT})"

echo ""
echo "========================================================================"
echo "Both Trainings Completed!"
echo "========================================================================"

# Print summary from log files
echo ""
echo "=== Flow Model Summary (GPU 0) ==="
tail -30 results/logs_flow_lc${LAMBDA_COST}_gpu0.txt | grep -E "epoch|Final|cost|carbon|Vm|Va|MAE|Satisfaction|EMA|scale" || true

echo ""
echo "=== MLP Model Summary (GPU 1) ==="
tail -30 results/logs_mlp_lc${LAMBDA_COST}_gpu1.txt | grep -E "epoch|Final|cost|carbon|Vm|Va|MAE|Satisfaction|EMA|scale" || true

echo ""
echo "========================================================================"
echo "Next Steps:"
echo "========================================================================"
echo "1. Compare TensorBoard logs:"
echo "   tensorboard --logdir runs/"
echo ""
echo "2. Check EMA scales in logs (should be reasonable):"
echo "   - Cost scale: ~4000-5000 (typical cost range)"
echo "   - Carbon scale: ~100-200 (typical carbon range after scaling)"
echo ""
echo "3. Verify normalization is working:"
echo "   - Normalized cost/carbon should be in similar magnitude (~0.5-2.0)"
echo ""
echo "4. Test other preferences by modifying LAMBDA_COST/LAMBDA_CARBON:"
echo "   - Pure economic: LAMBDA_COST=1.0, LAMBDA_CARBON=0.0"
echo "   - Pure carbon:   LAMBDA_COST=0.0, LAMBDA_CARBON=1.0"
echo ""
echo "5. After fixed preference validation works, test random sampling:"
echo "   - Change NGT_PREF_SAMPLING='random' in this script"
echo "   - This enables Pareto coverage testing"
echo ""

