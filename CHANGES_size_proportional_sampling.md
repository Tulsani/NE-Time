# Size-proportional sampling — changes & next plan

## Why

Prior HPC run OOM-killed during ECL H=720 zero-shot eval, and separately its training log
showed an early-overfit pattern: best val_loss at epoch 4, rising every epoch after through
early-stop at epoch 19. Comparing to published zero-shot foundation-model numbers, ETTh1
(in-domain) scored worse than every zero-shot competitor despite a training advantage they
didn't have, while ETTh2 (same setup) was excellent.

Root cause found for the plateau: `train_foundation.py` sampled ETTh1/ETTh2/ETTm1 with equal
probability per gradient step regardless of size. ETTh1/ETTh2 have 1,174 train batches each;
ETTm1 has 5,094. Equal per-step sampling means each ETTh1/ETTh2 batch got revisited ~4.3x more
often than each ETTm1 batch per unit of training — a real per-sample over-exposure, independent
of any epoch/schedule definition.

## What changed

- **`dataset.py` / `train.py`** (commit `85406c2`, already on `ftr/ne-time-foundational`):
  fixed the eval-path OOM. `evaluate_dataset`/`Trainer.test` were iterating the
  horizon-duplicated test loader once per horizon (a `len(horizons)²` blowup) and
  accumulating every batch's predictions/targets in memory before computing metrics. Added
  `dataset.py::unique_window_loader` (dedupes back to one row per window) and
  `train.py::StreamingMetrics` (running-sum MSE/MAE instead of full concatenation).

- **`train_foundation.py` / `scripts/train_foundation.sh`** (commit `e4d1767`, this branch
  `ftr/size-propotional-sampling`): added `--size_weighted_sampling` (default off, behavior
  unchanged) to sample each pretrain dataset proportional to its own train-batch count instead
  of uniformly, equalizing per-sample revisit rate across datasets. Verified in simulation:
  uniform gives ETTh1/ETTh2 ~1.4-1.5x revisits/epoch vs ETTm1's 0.32x; weighted equalizes all
  three to ~0.68-0.71x. Wired through `scripts/train_foundation.sh` via a
  `SIZE_WEIGHTED_SAMPLING` env var so one HPC code sync serves both the baseline and this
  experiment.

- **Deliberately not changed this round**: `TemporalProjector`'s hardcoded `proj_hidden=64`
  (`model_no_attn_upd.py:179`, ~21% of params, doesn't scale with `--size`) — flagged as a
  plausible secondary overfitting contributor, but tying it to `hyp_dim` would be a no-op for
  the `medium` config these runs use, so it's parked until we're testing other model sizes.
  Also not in scope: broader foundation-model task coverage (imputation / anomaly detection /
  classification) — current eval is forecasting-only, matching the forecasting-only
  competitors (Chronos/TimesFM/Moirai/Time-MoE) already used for comparison; expanding to the
  MOMENT/UniTS-style multi-task protocol would need new heads and new benchmark datasets, a
  separate scope decision.

## Status

- **Job A (baseline rerun)**: ✅ done. Completed end-to-end with no OOM — the eval-path fix
  holds under real Traffic-scale memory pressure. Results reproduce identically to the crashed
  run everywhere they previously overlapped, confirming the fix changed memory behavior only,
  not numbers. Newly obtained: ECL H=720 (MSE 0.580) and full Traffic (MSE 0.947-1.154 across
  horizons) — see `comparison-sota.md` for the complete table. Training itself still shows the
  same epoch-4 plateau (best val_loss=0.36208 at epoch 4, rising through early-stop at epoch
  19) — expected, since Job A intentionally didn't include the sampling fix.
- **Job B (size-weighted sampling)**: ✅ done. **Mixed result — did not resolve the plateau,
  best checkpoint landed even earlier (epoch 1, val_loss=0.35359).** A parallel uniform-sampling
  replicate run (different RNG draw sequence, same distribution) also plateaued early (epoch 2),
  so early plateauing looks like a property of this training setup broadly, not specifically
  caused by the sampling imbalance. Per-dataset effect of the fix itself was as predicted
  directionally — ETTm1 (now 68% of steps, up from 33%) improved, its near-domain sibling ETTm2
  improved, Weather/ECL/Traffic (genuinely out-of-domain) all improved, ECL H=720 in particular
  dropped from 0.580→0.376 (-35%) — but **ETTh1 (now only 15.8% of steps, down from 33%) got
  worse across all horizons** (0.495→0.526 @H96), widening rather than closing the original
  red-flag gap against published zero-shot ETTh1 numbers. **Caveat**: ECL and Traffic MSE are
  suspiciously flat across horizons in this run (Traffic: 0.878→0.867→0.874→0.871 from H=96 to
  H=720) — real forecasting difficulty should increase with horizon; this flatness is
  consistent with an undertrained (epoch-1) checkpoint outputting something close to a
  constant/mean-ish prediction for very out-of-domain wide-channel sets rather than genuinely
  differentiating by horizon. Not confirmed either way — worth spot-checking actual predictions
  before trusting the ECL/Traffic improvement as real forecasting skill.
- **Job C (lower LR, `LR=3e-4`, on top of size-weighted sampling)**: 🔄 running.
- **Job D (smaller hyperbolic capacity, `HYP_HIDDEN_SCALE=0.5`, on top of size-weighted
  sampling)**: 🔄 running.

Job B's best checkpoint landing during LR warmup (epoch 1, avg LR ~1.6e-4 vs. 1e-3 peak)
suggested LR could be the more fundamental issue rather than sampling — Jobs C/D test that and
the capacity/expressiveness hypothesis directly, in parallel, no code changes (both `--lr` and
`--hyp_hidden_scale` were already-wired CLI flags).

## Next plan

1. Wait for Jobs C and D to report back; compare their best-epoch and val trajectory in
   `*_log.csv` against Job B's (best epoch 1, val 0.354):
   - If lower LR pushes the best epoch meaningfully later → LR was the dominant issue.
   - If smaller `hyp_hidden_scale` pushes it later or lowers the val floor → capacity/expressiveness was the bigger factor.
   - If neither moves it much → the plateau may be structural (e.g. `TemporalProjector`'s fixed
     720-width bottleneck, or the joint-pretraining objective simply saturating at this data
     scale) — revisit that deferred item.
2. Sanity-check the ECL/Traffic horizon-flatness in Job B before treating that improvement as
   confirmed (e.g. diff actual H=96 vs H=720 predictions for a few Traffic series).
3. Separately (own branch/experiment, not blocking the above): restructure the pretrain/
   zero-shot split so no zero-shot target shares a dataset family with pretrain (e.g. pretrain
   on Weather(+Exchange), zero-shot on the full ETT family + ECL + Traffic) — needed for an
   honest, fully apples-to-apples SOTA comparison table, since ETTh1/ETTh2 are currently
   in-domain for us but zero-shot for every competitor, and ETTm2 shares family with pretrain-set
   ETTm1.
