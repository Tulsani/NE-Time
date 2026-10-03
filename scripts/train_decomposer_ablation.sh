#!/bin/bash
# Decomposer ablation: is the fixed multi-scale (global/meso/local) moving-average
# decomposition actually load-bearing, or is most of nano_wecm1_sw_lr3e4's performance
# coming from the three-branch hyperbolic-encoder/fusion structure regardless of what
# feeds it?
#
# Motivation: FixedMADecomposer is a zero-parameter, purely statistical decomposition
# (nested box-car moving averages, structurally close to classical STL/ARIMA-style
# trend-seasonal decomposition) that was adopted specifically because a *learnable*
# decomposer let the model shortcut past the hyperbolic encoders entirely (best val at
# epoch 1 — see FixedMADecomposer's docstring in decomposition_upd.py). That earlier
# finding, plus the hyperbolic-vs-Euclidean ablation's modest (not dominant) effect size,
# raises the question of whether the decomposition itself — not the geometry, not even the
# multi-branch structure — is what's actually carrying most of the parameter-efficient
# forecasting performance.
#
# This submits ONE job identical to the adopted backbone in every respect (same corpus,
# size, sampling, LR, geometry=hyperbolic) except DECOMPOSER=none: all three scale branches
# receive the identical undecomposed patches instead of the global/meso/local split. The
# three encoders remain separately-parameterized and fusion is unchanged, so this isolates
# the decomposition's own contribution specifically (not the three-branch/fusion mechanism,
# which stays intact) — exact same 79,418 params either way (FixedMADecomposer itself has
# zero learnable parameters, so removing it doesn't change capacity).
#
# Usage:
#   bash scripts/train_decomposer_ablation.sh

set -e

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${PROJECT_DIR}"

PRETRAIN_DATASETS="Weather Exchange ECL ETTm1"
ZERO_SHOT_DATASETS="ETTh1 ETTh2 ETTm2 Traffic"

echo "Submitting: nano_wecm1_sw_lr3e4_nodecomp (decomposer-ablation control)"
GEOMETRY=hyperbolic DECOMPOSER=none \
    PRETRAIN_DATASETS="${PRETRAIN_DATASETS}" ZERO_SHOT_DATASETS="${ZERO_SHOT_DATASETS}" \
    MODEL_SIZE=nano PROJ_HIDDEN=32 \
    SIZE_WEIGHTED_SAMPLING=true LR=3e-4 \
    EXP_NAME="nano_wecm1_sw_lr3e4_nodecomp" \
    sbatch scripts/train_foundation.sh

echo ""
echo "Submitted. Check status with: squeue -u \$USER"
echo "Once done, compare outputs_foundation/nano_wecm1_sw_lr3e4_nodecomp_zeroshot_results.json"
echo "against nano_wecm1_sw_lr3e4 (default). Interpretation:"
echo "  - Large drop without the decomposer -> decomposition IS most of the story;"
echo "    hyperbolic geometry's ~1-3% edge (Section 3.2.2) is a secondary refinement on top"
echo "    of a fundamentally classical statistical-decomposition-driven model."
echo "  - Small/no drop -> the three-branch/fusion structure matters more than the specific"
echo "    decomposition feeding it; reframe accordingly."
