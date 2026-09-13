#!/bin/bash
#SBATCH --job-name=netime_classification
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# ── Usage ─────────────────────────────────────────────────────────────────
# One-time setup (login node, needs internet — same reason Weather/Exchange/ECL/Traffic
# needed download_foundation_data.sh: compute nodes have no internet, and
# aeon.datasets.load_classification() downloads from timeseriesclassification.com):
#   pip install -q aeon
#   python download_classification_data.py
#
# Default run (6 small, commonly-cited univariate UCR datasets):
#   sbatch scripts/finetune_classification.sh
#
# Override checkpoint / datasets:
#   CKPT_NAME=nano_wecm1_sw_lr3e4 sbatch scripts/finetune_classification.sh
#   DATASETS="Chinatown ECG200" sbatch scripts/finetune_classification.sh
#
# See finetune_classification.py for full protocol details (MOMENT's own approach:
# frozen backbone as a feature extractor, mean-pooled embeddings, off-the-shelf SVM —
# no new training loop).

set -e

USERNAME=${USER}
SCRATCH=/gpfs/scratch/${USERNAME}/at6646
ENV_PATH=${SCRATCH}/conda_envs/hypertime
PROJECT_DIR=${SLURM_SUBMIT_DIR}

# ── Config ────────────────────────────────────────────────────────────────
CKPT_NAME=${CKPT_NAME:-nano_wecm1_sw_lr3e4}
DATASETS=${DATASETS:-"Chinatown ECG200 GunPoint ItalyPowerDemand Coffee TwoLeadECG"}
SVM_C=${SVM_C:-1.0}
SVM_KERNEL=${SVM_KERNEL:-rbf}
EXP_NAME=${EXP_NAME:-classification_${CKPT_NAME}}

# ── Environment ───────────────────────────────────────────────────────────
source ~/.bashrc
conda activate ${ENV_PATH}

export HF_HOME=${SCRATCH}/hf_cache
export TORCH_HOME=${SCRATCH}/torch_cache
export TMPDIR=${SCRATCH}/tmp
mkdir -p ${SCRATCH}/tmp

# harmless / near-instant if already installed — only actually needs internet the first time
pip install -q aeon

cd ${PROJECT_DIR}
mkdir -p logs outputs_classification data/aeon_data

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
echo "SVM C / kernel:      ${SVM_C} / ${SVM_KERNEL}"
echo "Exp name:            ${EXP_NAME}"
echo "Start time:          $(date)"
echo "================================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Run ───────────────────────────────────────────────────────────────────
python finetune_classification.py \
    --ckpt        "${CKPT_PATH}" \
    --datasets    ${DATASETS} \
    --data_path   ${PROJECT_DIR}/data/aeon_data \
    --svm_C       ${SVM_C} \
    --svm_kernel  ${SVM_KERNEL} \
    --exp_name    ${EXP_NAME} \
    --output_dir  ${PROJECT_DIR}/outputs_classification

echo ""
echo "================================================"
echo "Done: $(date)"
echo "Outputs:"
ls -lh ${PROJECT_DIR}/outputs_classification/ | grep "${EXP_NAME}"
echo "================================================"
