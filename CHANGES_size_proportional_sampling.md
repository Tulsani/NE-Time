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
- **Job C (lower LR, `LR=3e-4`, on top of size-weighted sampling)**: ✅ done. Best epoch 1,
  val floor 0.352 — essentially unchanged from Job B. **LR magnitude ruled out** as the driver
  within the range tested.
- **Job D (smaller hyperbolic capacity, `HYP_HIDDEN_SCALE=0.5`, 167K params, on top of
  size-weighted sampling)**: ✅ done. Best epoch 2, val floor 0.355 — essentially unchanged.
  **Hyperbolic-encoder capacity alone ruled out.**
- **Job E (smaller TemporalProjector, `PROJ_HIDDEN=16`, on top of size-weighted sampling)**:
  ✅ done (run on Colab after an HPC node hit a transient `CUDA device busy` failure — unrelated
  infra flake, resubmit-and-it-lands-elsewhere fixed it). Best epoch 1, val floor 0.351 —
  essentially unchanged. **TemporalProjector capacity alone ruled out.**
- **Job F — `tiny` preset (50.7K params, 5x smaller than `medium`, `d_model=48/hyp_dim=24/
  cond_dim=24`, `PROJ_HIDDEN=24`, on top of size-weighted sampling)**: ✅ done, and this is the
  first lever that actually moved things. **Best epoch 8** (latest of all 8 runs to date) and
  **val floor 0.347** (lowest of all 8 runs). Unlike Jobs C/D/E, shrinking the *whole* model
  (not one component in isolation) changed the qualitative training dynamics, not just the
  numbers slightly. Per-dataset: **ETTh1 improved for the first time across every ablation**
  (0.526→0.514 @H96, 0.692→0.675 @H720 vs. Job B) — this is the dataset the original SOTA-gap
  red flag was about, and it's the first lever that's moved it in the right direction. ETTm1
  also improved meaningfully (0.364→0.342 @H96, -6%). ETTh2 ~flat. **But Weather (zero-shot,
  our one fully apples-to-apples comparison) got meaningfully *worse*** (0.208→0.242 @H96, -16%;
  0.362→0.434 @H720, -20%) — a real capacity tradeoff, not a clean win. ETTm2/Exchange roughly
  held or improved slightly. **Traffic's flat-across-horizon MSE pattern persists** even at 5x
  smaller (0.898/0.886/0.896/0.898) — this weakens the earlier "maybe it's TemporalProjector
  specifically" theory (Job E already shrunk that component alone with no effect on the
  flatness), so it looks more like a property of how this model handles Traffic (862 channels,
  the most out-of-domain set) at any capacity tried so far, not a fixable single-component issue.
- **Job G — `nano` preset (79,418 params, `PROJ_HIDDEN=32`, on top of size-weighted
  sampling)**: ✅ done. Confirms the size trend is a real, monotonic pattern, not a tiny-run
  fluke: best epoch **5** (medium=1, nano=5, tiny=8) and val floor **0.351** (medium=0.354,
  nano=0.351, tiny=0.347) — smoothly ordered by size in both directions. More importantly,
  `nano` isn't just an interpolation — it **beats both endpoints outright** on ETTh1 (0.507 @H96,
  best of all three), ETTm1 (0.340, best of all three), ETTm2 (0.159, best of all three), and
  ECL (0.297, best of all three). On Weather, it recovers most of tiny's regression while medium
  stays best (medium 0.208 / nano 0.219 / tiny 0.242 @H96 — nano keeps ~68% of the gap closed
  relative to tiny). Traffic's flat-across-horizon anomaly persists at this size too
  (0.878/0.874/0.883/0.888), now confirmed across all three sizes tested — clearly not a
  capacity artifact of any single size point.

**Recommendation: adopt `nano` (79,418 params, `--proj_hidden 32`) going forward.** It isn't a
compromise between medium and tiny — it wins on 4 of 5 axes (ETTh1, ETTm1, ETTm2, ECL) and loses
least on the 5th (Weather), so there's no real case left for either endpoint over it absent a
reason to specifically prioritize Weather. Further size ablation would have diminishing
returns; time is better spent on the dataset remapping and multi-task work below.

## Where this leaves the four-hypothesis investigation

Sampling imbalance, LR magnitude, and two independent single-component capacity levers
(hyperbolic encoders, TemporalProjector) were all ruled out as the epoch-1 plateau's cause.
**Only shrinking the whole model at once moved it** — consistent with the hypothesis that a
tiny (~250K-1M range) hyperbolic model is simply more parameter-efficient than its Euclidean
equivalent at this task's scale (comparable simple baselines like DLinear already do well on
these benchmarks, implying the underlying task doesn't need much capacity), so the `medium`
config was likely genuinely over-parameterized for this joint 3-dataset objective — not from
any one component, but in aggregate. That's a positive, actionable finding, with one real
caveat: it's not a strictly dominant improvement (Weather zero-shot got worse), so this reads
as a capacity/generalization tradeoff to make deliberately, not a free lunch.

## Next plan

1. **Model size decided: `nano` (79,418 params, `--proj_hidden 32`)** going forward for the
   remaining work below.
2. Sanity-check the Traffic horizon-flatness properly at some point (diff actual H=96 vs H=720
   predictions for a few series) — now evidenced across all three sizes tested, so worth
   resolving before citing Traffic zero-shot numbers in the report either way.
3. Restructure the pretrain/zero-shot split so no zero-shot target shares a dataset family with
   pretrain (e.g. pretrain on Weather(+Exchange), zero-shot on the full ETT family + ECL +
   Traffic) — needed for an honest, fully apples-to-apples SOTA comparison table, since
   ETTh1/ETTh2 are currently in-domain for us but zero-shot for every competitor, and ETTm2
   shares family with pretrain-set ETTm1.
4. Add non-forecasting task capabilities (imputation/anomaly detection/classification) for a
   true multi-task foundation-model comparison, per the MOMENT/UniTS-style protocol — scoped
   separately, needs new heads and new benchmark datasets.
