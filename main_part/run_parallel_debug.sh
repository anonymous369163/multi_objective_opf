#!/bin/bash
# ==============================================================================
# Parallel Training Script: Flow (GPU0) vs MLP (GPU1) for λ=0.8
#
# This script trains both models in parallel on separate GPUs with TensorBoard
# logging enabled. This allows direct comparison of training dynamics.
#
# Usage:
#   chmod +x run_parallel_debug.sh
#   ./run_parallel_debug.sh
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
echo "Parallel Training: Flow vs MLP (λ=0.5)"
echo "========================================================================"

# ============================================================
# Training Configuration
# ============================================================
EPOCHS=4500
LAMBDA_COST=0.5
LAMBDA_CARBON=0.5
FLOW_STEPS=10
HIDDEN_DIM=144
NUM_LAYERS=2
DEBUG = 0   # 是否直接导入训练好的模型，测试模型的性能，默认不导入

echo ""
echo "Configuration:"
echo "  Epochs: ${EPOCHS}"
echo "  Target preference: λ_cost=${LAMBDA_COST}, λ_carbon=${LAMBDA_CARBON}"
echo "  Flow integration steps: ${FLOW_STEPS}"
echo "  Flow hidden dim: ${HIDDEN_DIM}, layers: ${NUM_LAYERS}"
echo "  Debug mode: ${DEBUG}"
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
NGT_FLOW_ZERO_INIT=True \
TB_ENABLED=True \
NGT_USE_PROJ=True \
DEBUG=${DEBUG} \
python train.py > results/logs_flow_gpu0.txt 2>&1 &
PID_FLOW=$!

echo "  PID: ${PID_FLOW}"
echo "  Log: results/logs_flow_gpu0.txt"

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
python train.py > results/logs_mlp_gpu1.txt 2>&1 &
PID_MLP=$!

echo "  PID: ${PID_MLP}"
echo "  Log: results/logs_mlp_gpu1.txt"

echo ""
echo "========================================================================"
echo "Both trainings started in background!"
echo "========================================================================"
echo ""
echo "Monitor progress:"
echo "  tail -f results/logs_flow_gpu0.txt    # Flow model (GPU 0)"
echo "  tail -f results/logs_mlp_gpu1.txt     # MLP model (GPU 1)"
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
tail -30 results/logs_flow_gpu0.txt | grep -E "epoch|Final|cost|carbon|Vm|Va|MAE|Satisfaction" || true

echo ""
echo "=== MLP Model Summary (GPU 1) ==="
tail -30 results/logs_mlp_gpu1.txt | grep -E "epoch|Final|cost|carbon|Vm|Va|MAE|Satisfaction" || true

echo ""
echo "========================================================================"
echo "Next Steps:"
echo "========================================================================"
echo "1. Compare TensorBoard logs:"
echo "   tensorboard --logdir runs/"
echo ""
echo "2. Evaluate both models:"
echo "   python evaluate_multi_preference.py -s -u --epochs ${EPOCHS} \\"
echo "     --custom-flow NetV_ngt_flow_300bus_lc08_E${EPOCHS}_final.pth 0.8"
echo ""

