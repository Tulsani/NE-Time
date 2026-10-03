#!/bin/bash
# Multi-seed hyperbolic-vs-Euclidean comparison, with the cross-curvature fusion bug fixed.
#
# Addresses two of the most-repeated reviewer concerns directly:
#   1. "No seed-level uncertainty is reported" (e3ix, major concern #1) — this run gives
#      3 independent seeds per geometry (6 jobs total), paired so seed N's hyperbolic run
#      and seed N's Euclidean run differ ONLY in geometry, everything else identical.
#   2. The cross-curvature fusion bug flagged by both e3ix and q8SY is now fixed in
#      hyperbolic_ops.py/model_no_attn_upd.py (each scale is logmap0'd under its own
#      curvature before fusion, not the fusion curvature) — so this is also the first
#      trustworthy re-measurement of the hyperbolic-vs-Euclidean gap post-fix. The original
#      16/16 zero-shot win for hyperbolic was measured under the buggy fusion; it needs to
#      be re-established (or revised) under the corrected one before it goes in a
#      main-track submission.
#
# Same corpus/size/sampling/LR as the previously-adopted nano_wecm1_sw_lr3e4 throughout —
# only geometry and seed vary.
#
# Usage:
#   bash scripts/train_multiseed_comparison.sh

set -e

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${PROJECT_DIR}"

PRETRAIN_DATASETS="Weather Exchange ECL ETTm1"
ZERO_SHOT_DATASETS="ETTh1 ETTh2 ETTm2 Traffic"
SEEDS=(42 123 2024)

submit() {
    local exp_name="$1"
    local geometry="$2"
    local seed="$3"
    echo "Submitting: exp_name=${exp_name} geometry=${geometry} seed=${seed}"
    GEOMETRY="${geometry}" SEED="${seed}" \
        PRETRAIN_DATASETS="${PRETRAIN_DATASETS}" ZERO_SHOT_DATASETS="${ZERO_SHOT_DATASETS}" \
        MODEL_SIZE=nano PROJ_HIDDEN=32 \
        SIZE_WEIGHTED_SAMPLING=true LR=3e-4 \
        EXP_NAME="${exp_name}" \
        sbatch scripts/train_foundation.sh
}

for seed in "${SEEDS[@]}"; do
    submit "nano_wecm1_sw_lr3e4_fusionfix_seed${seed}"   hyperbolic "${seed}"
    submit "nano_wecm1_sw_lr3e4_euclidean_seed${seed}"   euclidean  "${seed}"
done

echo ""
echo "6 jobs submitted (3 seeds x 2 geometries). Check status with: squeue -u \$USER"
echo "Once all finish, aggregate outputs_foundation/*_fusionfix_seed*_zeroshot_results.json"
echo "and *_euclidean_seed*_zeroshot_results.json to get mean +/- std per (dataset, horizon)"
echo "instead of the single-seed comparison used in the workshop submission."
