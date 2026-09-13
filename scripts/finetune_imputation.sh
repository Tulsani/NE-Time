#!/bin/bash
#SBATCH --job-name=netime_imputation
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# ── Usage ─────────────────────────────────────────────────────────────────
# Requires the adopted foundation backbone checkpoint to already exist at
# ${PROJECT_DIR}/outputs_foundation/${CKPT_NAME}_best.pth (produced by
# scripts/train_foundation.sh / scripts/sweep_forecast_backbone.sh).
#
# Default run (all 6 MOMENT-protocol imputation datasets, all 4 mask ratios):
#   sbatch scripts/finetune_imputation.sh
#
# Override checkpoint / datasets / ratios:
#   CKPT_NAME=nano_wecm1_sw_lr3e4 sbatch scripts/finetune_imputation.sh
#   DATASETS="ETTh1 ETTm1" MASK_RATIOS="0.125 0.5" sbatch scripts/finetune_imputation.sh
#
# See finetune_imputation.py for full protocol details (freeze the pretrained
# backbone, train a fresh PatchUnfold reconstruction head per dataset/ratio —
# linear probing, per MOMENT's own imputation eval protocol).

set -e

USERNAME=${USER}
SCRATCH=/gpfs/scratch/${USERNAME}/at6646
ENV_PATH=${SCRATCH}/conda_envs/hypertime
PROJECT_DIR=${SLURM_SUBMIT_DIR}

# ── Config ────────────────────────────────────────────────────────────────
CKPT_NAME=${CKPT_NAME:-nano_wecm1_sw_lr3e4}
DATASETS=${DATASETS:-"ETTh1 ETTh2 ETTm1 ETTm2 Weather ECL"}
MASK_RATIOS=${MASK_RATIOS:-"0.125 0.25 0.375 0.5"}
MASK_CHUNK_LEN=${MASK_CHUNK_LEN:-8}
BATCH_SIZE=${BATCH_SIZE:-32}
EPOCHS=${EPOCHS:-10}
LR=${LR:-1e-3}
BACKBONE_LR=${BACKBONE_LR:-1e-4}
FINETUNE_BACKBONE=${FINETUNE_BACKBONE:-false}
EXP_NAME=${EXP_NAME:-imputation_${CKPT_NAME}}

# ── Environment ───────────────────────────────────────────────────────────
source ~/.bashrc
conda activate ${ENV_PATH}

export HF_HOME=${SCRATCH}/hf_cache
export TORCH_HOME=${SCRATCH}/torch_cache
export TMPDIR=${SCRATCH}/tmp
mkdir -p ${SCRATCH}/tmp

cd ${PROJECT_DIR}
mkdir -p logs outputs_imputation data

CKPT_PATH=${PROJECT_DIR}/outputs_foundation/${CKPT_NAME}_best.pth
if [ ! -f "${CKPT_PATH}" ]; then
    echo "ERROR: backbone checkpoint not found at ${CKPT_PATH}"
    echo "       Run scripts/train_foundation.sh (or the sweep) first, or set CKPT_NAME."
    exit 1
fi

# ── Log job info ──────────────────────────────────────────────────────────
echo "================================================"
echo "Job ID:             ${SLURM_JOB_ID}"
echo "Node:                ${SLURMD_NODENAME}"
echo "Project dir:         ${PROJECT_DIR}"
echo "Backbone checkpoint: ${CKPT_PATH}"
echo "Datasets:            ${DATASETS}"
echo "Mask ratios:         ${MASK_RATIOS}"
echo "Mask chunk len:      ${MASK_CHUNK_LEN}"
echo "Batch size:          ${BATCH_SIZE}"
echo "Epochs (per head):   ${EPOCHS}"
echo "LR:                  ${LR}"
echo "Finetune backbone:   ${FINETUNE_BACKBONE}"
echo "Backbone LR:         ${BACKBONE_LR}"
echo "Exp name:            ${EXP_NAME}"
echo "Start time:          $(date)"
echo "================================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Run ───────────────────────────────────────────────────────────────────
EXTRA_ARGS=()
if [ "${FINETUNE_BACKBONE}" = "true" ]; then
    EXTRA_ARGS+=(--finetune_backbone)
fi

python finetune_imputation.py \
    --ckpt              "${CKPT_PATH}" \
    --datasets          ${DATASETS} \
    --data_path         ${PROJECT_DIR}/data \
    --mask_ratios       ${MASK_RATIOS} \
    --mask_chunk_len    ${MASK_CHUNK_LEN} \
    --batch_size        ${BATCH_SIZE} \
    --epochs            ${EPOCHS} \
    --lr                ${LR} \
    --backbone_lr       ${BACKBONE_LR} \
    --num_workers       4 \
    --exp_name          ${EXP_NAME} \
    --output_dir        ${PROJECT_DIR}/outputs_imputation \
    "${EXTRA_ARGS[@]}"

echo ""
echo "================================================"
echo "Done: $(date)"
echo "Outputs:"
ls -lh ${PROJECT_DIR}/outputs_imputation/ | grep "${EXP_NAME}"
echo "================================================"
