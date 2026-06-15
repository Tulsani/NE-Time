#!/bin/bash
#SBATCH --job-name=netime
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-14:00:00

# ── Usage ─────────────────────────────────────────────────────────────────
# Multi-horizon (joint training):
#   sbatch submit_training.sh
#
# Per-horizon + CI (benchmark comparison — trains 4 separate models):
#   PER_HORIZON=1 CI=1 sbatch submit_training.sh
#   PER_HORIZON=1 CI=1 DATASET=ETTh1 sbatch submit_training.sh
#
# Override any config:
#   DATASET=ETTm1 MODEL_SIZE=large PER_HORIZON=1 CI=1 sbatch submit_training.sh
#
# Tune overfitting-fix knobs:
#   GEO_DROPOUT=0.3 HYP_HIDDEN_SCALE=1.0 CURVATURE_WD=1e-3 sbatch submit_training.sh

set -e

USERNAME=${USER}
SCRATCH=/gpfs/scratch/${USERNAME}/at6646
ENV_PATH=${SCRATCH}/conda_envs/hypertime
PROJECT_DIR=${SLURM_SUBMIT_DIR}

# ── Config ────────────────────────────────────────────────────────────────
DATASET=${DATASET:-ETTm1}
MODEL_SIZE=${MODEL_SIZE:-medium}
EPOCHS=${EPOCHS:-50}
LR=${LR:-1e-3}
PATIENCE=${PATIENCE:-15}
PER_HORIZON=${PER_HORIZON:-0}
CI=${CI:-0}

# Overfitting-fix knobs (see train.py for details)
GEO_DROPOUT=${GEO_DROPOUT:-0.2}         # tangent-space dropout (encoder + decoder)
HYP_HIDDEN_SCALE=${HYP_HIDDEN_SCALE:-1.0}  # hyp encoder hidden = d_model * this (was 2.0)
CURVATURE_WD=${CURVATURE_WD:-1e-3}      # weight decay for CurvatureParam

# Dataset-specific settings
if [ "${DATASET}" = "ETTm1" ] || [ "${DATASET}" = "ETTm2" ]; then
    SEQ_LEN=${SEQ_LEN:-512}
    BATCH_SIZE=${BATCH_SIZE:-64}
    TRAIN_STRIDE=${TRAIN_STRIDE:-2}
else
    SEQ_LEN=${SEQ_LEN:-336}
    BATCH_SIZE=${BATCH_SIZE:-32}
    TRAIN_STRIDE=${TRAIN_STRIDE:-1}
fi

# Experiment name encodes mode
if [ "${PER_HORIZON}" = "1" ] && [ "${CI}" = "1" ]; then
    EXP_NAME=${EXP_NAME:-netime_${DATASET}_${MODEL_SIZE}_perH_CI}
    MODE_FLAG="--per_horizon --ci"
    MODE_LABEL="per-horizon+CI (SOTA comparison)"
elif [ "${PER_HORIZON}" = "1" ]; then
    EXP_NAME=${EXP_NAME:-netime_${DATASET}_${MODEL_SIZE}_perH}
    MODE_FLAG="--per_horizon"
    MODE_LABEL="per-horizon"
else
    EXP_NAME=${EXP_NAME:-netime_${DATASET}_${MODEL_SIZE}_joint}
    MODE_FLAG=""
    MODE_LABEL="multi-horizon (joint training)"
fi

# ── Environment ───────────────────────────────────────────────────────────
source ~/.bashrc
conda activate ${ENV_PATH}

export HF_HOME=${SCRATCH}/hf_cache
export TORCH_HOME=${SCRATCH}/torch_cache
export TMPDIR=${SCRATCH}/tmp
mkdir -p ${SCRATCH}/tmp

cd ${PROJECT_DIR}
mkdir -p logs outputs data

# ── Log job info ──────────────────────────────────────────────────────────
echo "================================================"
echo "Job ID:           ${SLURM_JOB_ID}"
echo "Node:             ${SLURMD_NODENAME}"
echo "Project dir:      ${PROJECT_DIR}"
echo "Dataset:          ${DATASET}"
echo "Model size:       ${MODEL_SIZE}"
echo "Mode:             ${MODE_LABEL}"
echo "Seq len:          ${SEQ_LEN}"
echo "Batch size:       ${BATCH_SIZE}"
echo "LR:               ${LR}"
echo "Patience:         ${PATIENCE}"
echo "Exp name:         ${EXP_NAME}"
echo "--- overfitting fixes ---"
echo "geo_dropout:      ${GEO_DROPOUT}"
echo "hyp_hidden_scale: ${HYP_HIDDEN_SCALE}"
echo "curvature_wd:     ${CURVATURE_WD}"
echo "Start time:       $(date)"
echo "================================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

python -c "
import torch, sys
print(f'Python:     {sys.version.split()[0]}')
print(f'PyTorch:    {torch.__version__}')
print(f'CUDA:       {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:        {torch.cuda.get_device_name(0)}')
    print(f'VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────
echo "Starting training (${MODE_LABEL})..."

python train.py \
    --dataset          ${DATASET} \
    --data_path        ${PROJECT_DIR}/data \
    --seq_len          ${SEQ_LEN} \
    --horizons         96 192 336 720 \
    --size             ${MODEL_SIZE} \
    --patch_size       16 \
    --patch_stride     8 \
    --train_stride     ${TRAIN_STRIDE} \
    --epochs           ${EPOCHS} \
    --warmup_epochs    5 \
    --batch_size       ${BATCH_SIZE} \
    --lr               ${LR} \
    --weight_decay     1e-4 \
    --patience         ${PATIENCE} \
    --num_workers      4 \
    --geo_dropout      ${GEO_DROPOUT} \
    --hyp_hidden_scale ${HYP_HIDDEN_SCALE} \
    --curvature_wd     ${CURVATURE_WD} \
    --exp_name         ${EXP_NAME} \
    --output_dir       ${PROJECT_DIR}/outputs \
    ${MODE_FLAG}

echo ""
echo "================================================"
echo "Done: $(date)"
echo "Outputs:"
ls -lh ${PROJECT_DIR}/outputs/ | grep "${EXP_NAME}"
echo "================================================"