#!/bin/bash
# Run this ONCE on a login node (with outbound internet) before submitting
# train_foundation.sh — compute nodes on most clusters have no internet, and
# Weather/Exchange/ECL/Traffic can only be fetched via huggingface_hub, not
# urllib (see dataset.py). ETT family is included too so nothing needs to hit
# the network at all once the job is queued.
#
# Usage:
#   bash scripts/download_foundation_data.sh
#   bash scripts/download_foundation_data.sh /path/to/data

set -e

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_PATH=${1:-${PROJECT_DIR}/data}

cd "${PROJECT_DIR}"
mkdir -p "${DATA_PATH}"

echo "Downloading foundation-model datasets to: ${DATA_PATH}"
python download_dataset.py --data_path "${DATA_PATH}" \
    --datasets ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange ECL Traffic

echo ""
echo "Done. Files:"
ls -lh "${DATA_PATH}"/*.csv
