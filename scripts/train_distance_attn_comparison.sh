#!/bin/bash
# Option B redesign: HyperbolicDistanceAttention, tested the same way every other change in
# this project has been — 3 seeds x {hyperbolic, Euclidean-control}, same corpus/size/
# sampling/LR as the adopted backbone, parameter-matched within each pair (79,616 params
# either way, vs 79,418 without this component — see model_no_attn_upd.py param_count()).
#
# This is the first test of whether a genuinely non-round-trippable hyperbolic operation
# (patch-to-patch mixing weighted by hyp_distance, not just encode-fuse-decode) produces a
# real, reproducible edge over its Euclidean (squared-distance) twin — the previous
# encode-fuse-decode-only design did not, once the cross-curvature fusion bug was fixed
# (see Documentation/hyperbolic-bug.md and scripts/train_multiseed_comparison.sh's results).
#
# Usage:
#   bash scripts/train_distance_attn_comparison.sh

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
    echo "Submitting: exp_name=${exp_name} geometry=${geometry} seed=${seed} distance_attn=on"
    GEOMETRY="${geometry}" SEED="${seed}" DISTANCE_ATTN=on \
        PRETRAIN_DATASETS="${PRETRAIN_DATASETS}" ZERO_SHOT_DATASETS="${ZERO_SHOT_DATASETS}" \
        MODEL_SIZE=nano PROJ_HIDDEN=32 \
        SIZE_WEIGHTED_SAMPLING=true LR=3e-4 \
        EXP_NAME="${exp_name}" \
        sbatch scripts/train_foundation.sh
}

for seed in "${SEEDS[@]}"; do
    submit "nano_wecm1_sw_lr3e4_distattn_hyp_seed${seed}" hyperbolic "${seed}"
    submit "nano_wecm1_sw_lr3e4_distattn_euc_seed${seed}" euclidean  "${seed}"
done

echo ""
echo "6 jobs submitted (3 seeds x 2 geometries, with HyperbolicDistanceAttention enabled)."
echo "Check status with: squeue -u \$USER"
echo ""
echo "Once done, compare mean +/- std per (dataset, horizon) against:"
echo "  - *_fusionfix_seed*   (fixed fusion, no distance attention -- the current no-effect result)"
echo "  - *_euclidean_seed*   (same, Euclidean control)"
echo "to see whether distance attention recovers a real, reproducible hyperbolic advantage."
