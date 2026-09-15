#!/bin/bash
# Euclidean-geometry ablation control for the adopted forecasting backbone.
#
# Question: is the hyperbolic (Poincare-ball) representation actually responsible for
# nano_wecm1_sw_lr3e4's forecasting performance, or would an equally-sized Euclidean
# model with the same decomposer/patching/horizon-conditioning do just as well?
#
# This submits ONE job that is identical to the adopted backbone in every respect —
# same corpus (Weather+Exchange+ECL+ETTm1 pretrain, ETTh1/ETTh2/ETTm2/Traffic zero-shot),
# same size (nano, proj_hidden=32 -> 79,418 params, confirmed exactly equal param count
# on both geometries), same sampling (size-weighted) and same LR (3e-4) — except
# GEOMETRY=euclidean, which replaces expmap0/logmap0 in HyperbolicEncoder/HyperbolicDecoder
# and the tangent-space fusion with identity / a plain weighted sum (see hyperbolic_ops.py,
# model_no_attn_upd.py). Any performance delta between this run and nano_wecm1_sw_lr3e4 is
# therefore attributable to the geometry itself, not to an unrelated architecture or
# training difference.
#
# Usage:
#   bash scripts/train_euclidean_ablation.sh

set -e

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${PROJECT_DIR}"

PRETRAIN_DATASETS="Weather Exchange ECL ETTm1"       # exact nano_wecm1_sw_lr3e4 corpus
ZERO_SHOT_DATASETS="ETTh1 ETTh2 ETTm2 Traffic"        # exact nano_wecm1_sw_lr3e4 zero-shot set

echo "Submitting: nano_wecm1_sw_lr3e4_euclidean (Euclidean-ablation control)"
GEOMETRY=euclidean \
    PRETRAIN_DATASETS="${PRETRAIN_DATASETS}" ZERO_SHOT_DATASETS="${ZERO_SHOT_DATASETS}" \
    MODEL_SIZE=nano PROJ_HIDDEN=32 \
    SIZE_WEIGHTED_SAMPLING=true LR=3e-4 \
    EXP_NAME="nano_wecm1_sw_lr3e4_euclidean" \
    sbatch scripts/train_foundation.sh

echo ""
echo "Submitted. Check status with: squeue -u \$USER"
echo "Compare against outputs_foundation/nano_wecm1_sw_lr3e4_zeroshot_results.json once done."
