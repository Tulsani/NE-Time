#!/bin/bash
# Small hyperparameter sweep for the remapped forecasting backbone.
#
# Pretrain corpus is Weather+Exchange+ECL (not ETTh1/h2/m1/m2) so the full ETT family
# becomes a clean, apples-to-apples zero-shot comparison against Time-MoE/Moirai/
# Chronos/TimesFM's own zero-shot numbers, with Traffic also held out zero-shot.
# ECL confirmed safe to add via scripts/check_wide_channel_memory.py (17.57GB peak on
# an A100 for a single forward+backward at the real batch_size=32/pred_len=720 config,
# vs. Traffic's 47.14GB — Traffic stays zero-shot-only, too thin a margin to risk this
# close to the deadline). See CHANGES_size_proportional_sampling.md for why this remap
# and why `nano`.
#
# Submits one sbatch job per combination below. Each job is independent — safe to run
# in parallel across whatever nodes are free. Exp names are prefixed `nano_wec_` (Weather+
# Exchange+ECL pretrain) so they don't collide with earlier runs in the same
# outputs_foundation/ dir.
#
# Usage:
#   bash scripts/sweep_forecast_backbone.sh
#
# All jobs use --size nano --proj_hidden 32 (the adopted backbone config) and the
# remapped corpus; each combination below is submitted as its own sbatch job.

set -e

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${PROJECT_DIR}"

PRETRAIN_DATASETS="Weather Exchange ECL"
ZERO_SHOT_DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 Traffic"

submit() {
    local exp_name="$1"
    local size_weighted="$2"
    local lr="$3"
    echo "Submitting: exp_name=${exp_name} size_weighted_sampling=${size_weighted} lr=${lr}"
    PRETRAIN_DATASETS="${PRETRAIN_DATASETS}" ZERO_SHOT_DATASETS="${ZERO_SHOT_DATASETS}" \
        MODEL_SIZE=nano PROJ_HIDDEN=32 \
        SIZE_WEIGHTED_SAMPLING="${size_weighted}" LR="${lr}" \
        EXP_NAME="${exp_name}" \
        sbatch scripts/train_foundation.sh
}

# 1. Baseline reference: uniform sampling, default LR — confirms nano still behaves
#    sanely on the new (3-dataset, wide-channel-included) corpus before trusting the rest.
submit "nano_wec_uniform_lr1e3"  false 1e-3

# 2. Best-known config so far (size-weighted sampling) at default LR.
submit "nano_wec_sw_lr1e3"       true  1e-3

# 3-5. Small LR sweep around the size-weighted config, to find the best LR for THIS
#      corpus specifically — Weather+Exchange+ECL is a different scale/composition
#      than ETTh1+ETTh2+ETTm1, so the old LR ablation's conclusion (LR magnitude
#      didn't matter) isn't guaranteed to transfer.
submit "nano_wec_sw_lr5e4"       true  5e-4
submit "nano_wec_sw_lr3e4"       true  3e-4
submit "nano_wec_sw_lr2e3"       true  2e-3

echo ""
echo "5 jobs submitted. Check status with: squeue -u \$USER"
