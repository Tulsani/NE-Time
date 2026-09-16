#!/bin/bash
# "Stronger geometry bias" pretraining run.
#
# Motivation: the hyperbolic-vs-Euclidean ablation (nano_wecm1_sw_lr3e4 vs.
# nano_wecm1_sw_lr3e4_euclidean) showed the hyperbolic model wins on all 16 zero-shot/
# near-domain (dataset, horizon) comparisons, but check_hyperbolic_utilization.py found the
# global-scale branch stays close to expmap0's near-linear regime (nonlinearity ratio ~0.98
# at the mean tangent norm) — i.e. curvature is under-exploited, especially globally, yet
# still helps. This run tests whether deliberately forcing more curvature engagement helps
# further, via two knobs (both newly wired, both no-ops on model capacity — same 79,418
# params either way):
#   - Higher curvature init (c_global/meso/local: 0.5/1.0/2.0 -> 1.5/2.5/4.0) — makes
#     expmap0 saturate at smaller tangent-vector norms.
#   - Larger HyperbolicEncoder output-layer init std (0.01 -> 0.05) — starts training with
#     points already further from the ball's origin, in the genuinely nonlinear regime,
#     instead of defaulting there implicitly (and, until a same-session bug fix, not even
#     reliably at 0.01 — HyperTimeV2._init_weights() was silently overwriting this layer's
#     init; now flagged with _custom_init so it isn't).
#
# Same corpus, size, sampling, LR as the adopted backbone (nano_wecm1_sw_lr3e4) and its
# Euclidean control — only these two geometry-bias knobs differ, so all three runs
# (hyperbolic-default, euclidean, hyperbolic-strong-bias) are directly comparable.
#
# Usage:
#   bash scripts/train_strong_geometry_bias.sh

set -e

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${PROJECT_DIR}"

PRETRAIN_DATASETS="Weather Exchange ECL ETTm1"
ZERO_SHOT_DATASETS="ETTh1 ETTh2 ETTm2 Traffic"

echo "Submitting: nano_wecm1_sw_lr3e4_strongcurv (stronger geometry bias)"
GEOMETRY=hyperbolic \
    C_GLOBAL_INIT=1.5 C_MESO_INIT=2.5 C_LOCAL_INIT=4.0 ENC_OUT_INIT_STD=0.05 \
    PRETRAIN_DATASETS="${PRETRAIN_DATASETS}" ZERO_SHOT_DATASETS="${ZERO_SHOT_DATASETS}" \
    MODEL_SIZE=nano PROJ_HIDDEN=32 \
    SIZE_WEIGHTED_SAMPLING=true LR=3e-4 \
    EXP_NAME="nano_wecm1_sw_lr3e4_strongcurv" \
    sbatch scripts/train_foundation.sh

echo ""
echo "Submitted. Check status with: squeue -u \$USER"
echo "Once done, compare outputs_foundation/nano_wecm1_sw_lr3e4_strongcurv_zeroshot_results.json"
echo "against nano_wecm1_sw_lr3e4 (default hyperbolic) and nano_wecm1_sw_lr3e4_euclidean."
echo ""
echo "Also worth re-running the utilization check against the new checkpoint to confirm the"
echo "stronger init/curvature actually pushed the trained model further from the origin:"
echo "  python check_hyperbolic_utilization.py --ckpt outputs_foundation/nano_wecm1_sw_lr3e4_strongcurv_best.pth --dataset ETTh2"
