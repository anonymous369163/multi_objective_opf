#!/bin/bash
# Run independent flow training: [1,0] -> [0.8, 0.2]
# Usage: bash scripts/run_independent_flow.sh

cd "$(dirname "$0")/../main_part"

python train_pareto_flow.py \
    --model_mode independent \
    --lambda_cost 0.8 \
    --lambda_carbon 0.2 \
    --pref_sampling fixed \
    --epochs 100 \
    --lr 1e-4 \
    --inf_steps 10 \
    --patience 20 \
    --val_freq 10 \
    # --use_projection \
    --adaptive_weights \
    --pareto_validation \
    --start_cost 1.0 \
    --end_cost 0.8 \
    --pref_step 0.2 \
    --gpu 0 \
    --batch_size 1024 \
    --batch_size_test 512