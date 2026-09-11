# NE-Time Foundation vs. published zero-shot SOTA

**Model**: HyperTimeV2 `medium` (~254K params), jointly pretrained on ETTh1+ETTh2+ETTm1.
**Competitors**: Time-MoE (base/large/ultra, up to several-B params), Moirai-large (311M),
Chronos-large (~700M), TimesFM (~200M) — numbers from Time-MoE's own zero-shot benchmark table
(arXiv:2409.16040, Table 3), the closest available head-to-head comparison across MSE at
H=96/192/336/720.

**Important caveat before reading the tables**: "zero-shot" isn't applied consistently here.
- **ETTh1 / ETTh2** are *in-domain* for us (directly pretrained on) but *zero-shot* for every
  competitor listed. Our numbers should beat theirs by default — when they don't, that's a
  real problem, not just "we're smaller."
- **ETTm2** is zero-shot for us, but ETTm1 (same station family, same instrumentation, 4x
  denser sampling of the same ~2yr span) was in our pretrain corpus. Competitors exclude the
  *entire* ETT family from pretraining. So our ETTm2 result is closer to "near-domain
  transfer" than the literature's fully-blind zero-shot setting — a real result, but an easier
  task than what the competitor numbers represent.
- **Weather** is the one dataset here that's genuinely apples-to-apples zero-shot for us too.

Two of our own runs are in these tables now: **Job A** (uniform per-step dataset sampling,
best checkpoint epoch 4) and **Job B** (size-weighted sampling, best checkpoint epoch 1 —
see `CHANGES_size_proportional_sampling.md` for why that's a mixed result, not a clean fix).

## ETTm2 (near-domain zero-shot for us; fully zero-shot for competitors)

| H | Job A | Job B | Time-MoE-ultra | Moirai-large | Chronos-large | TimesFM |
|---|---|---|---|---|---|---|
| 96  | 0.171 | **0.164** | 0.198 | 0.211 | 0.197 | 0.202 |
| 192 | 0.209 | **0.196** | 0.235 | 0.281 | 0.254 | 0.289 |
| 336 | 0.247 | **0.229** | 0.293 | 0.341 | 0.313 | 0.360 |
| 720 | 0.293 | **0.271** | 0.427 | 0.485 | 0.416 | 0.462 |

Job B widens the lead over every listed competitor at every horizon — but see the near-domain
caveat above, and note ETTm1 (ETTm2's pretrain sibling) went from 33% to 68% of gradient steps
in Job B, so this improvement is expected, not a surprise.

## Weather (genuinely zero-shot for everyone)

| H | Job A | Job B | Time-MoE-ultra | Moirai-large | Chronos-large |
|---|---|---|---|---|---|
| 96  | 0.215 | **0.208** | 0.157 | 0.199 | 0.194 |
| 192 | 0.259 | **0.242** | 0.208 | 0.246 | 0.249 |
| 336 | 0.308 | **0.290** | 0.255 | 0.274 | 0.302 |
| 720 | 0.384 | **0.362** | 0.405 | 0.337 | 0.372 |

Job B improves ~3-6% over Job A across the board — now roughly tied with Moirai/Chronos rather
than trailing them, though still behind Time-MoE at the shorter horizons. Still not a win, but
closer.

## ETTh1 / ETTh2 (in-domain for us — should be an easy win, wasn't for ETTh1)

| Dataset | H | Job A | Job B | Time-MoE-ultra (zero-shot) | Moirai-large (zero-shot) | Chronos-large (zero-shot) |
|---|---|---|---|---|---|---|
| ETTh1 | 96  | 0.495 | **0.526** | 0.349 | 0.381 | 0.441 |
| ETTh1 | 720 | 0.690 | **0.692** | 0.457 | 0.611 | 0.835 |
| ETTh2 | 96  | 0.212 | **0.202** | 0.292 | 0.296 | 0.320 |
| ETTh2 | 720 | 0.291 | **0.282** | 0.439 | 0.423 | 0.603 |

ETTh2: we win decisively at every horizon in both runs, and Job B widens the win further. **ETTh1: we
lose to every zero-shot competitor at every horizon in both runs, and the gap got *worse* in
Job B** (0.526 vs. best-competitor 0.349) — cutting ETTh1's gradient-step share from 33% to
15.8% hurt it more than the reduced over-repetition helped. The original red flag is not
resolved; see `CHANGES_size_proportional_sampling.md` for the LR/capacity ablations (Jobs C/D)
now queued to chase this further.

## Exchange, ECL, Traffic (no published zero-shot MSE table found — not benchmarked, logged for reference)

| Dataset | H | Job A | Job B |
|---|---|---|---|
| Exchange | 96  | 0.124 | **0.136** (worse) |
| Exchange | 720 | 1.090 | **1.116** (worse) |
| ECL      | 96  | 0.351 | **0.320** (better) |
| ECL      | 720 | 0.580 | **0.376** (much better, -35%) |
| Traffic  | 96  | 0.947 | **0.878** (better) |
| Traffic  | 720 | 1.154 | **0.871** (much better, -24%) |

Exchange and ECL are in a plausible range in both runs (Exchange is near-random-walk and
usually excluded from zero-shot benchmarks entirely; supervised *specialist* SOTA on ECL is
roughly 0.15-0.2 MSE, so our zero-shot ~0.32-0.58 is a reasonable, expected zero-shot gap). No
published zero-shot foundation-model comparison table was found for any of these three, so
treat this section as logged numbers, not a benchmark win/loss.

**Caveat on the Job B ECL/Traffic improvement**: both now show MSE that's nearly flat across
horizons (Traffic: 0.878→0.867→0.874→0.871 from H=96 to H=720, vs. Job A's clear
increasing-with-horizon trend). Real forecasting difficulty should increase with horizon — this
flatness is consistent with an undertrained (epoch-1) checkpoint outputting something close to
a constant/mean-ish prediction for these very out-of-domain, wide-channel sets, rather than
genuinely differentiating by horizon. Not confirmed either way yet — the improvement may be
real generalization (less pretrain-specific overfitting bleeding into unrelated domains) or a
degenerate near-constant prediction that happens to score well on MSE. Worth spot-checking
actual predictions before citing this as a benchmark win.

## Bottom line

Job B (size-weighted sampling) is a real but uneven improvement: better on every genuinely
out-of-domain zero-shot set (Weather, ECL, Traffic — modulo the flatness caveat above) and on
the near-domain ETTm2, but **worse on ETTh1**, which is the dataset the original SOTA-gap
red flag was about. The fix didn't resolve *why* training plateaus so early (best checkpoint
moved from epoch 4 to epoch 1, not later) — two more isolated ablations (lower LR, smaller
hyperbolic capacity) are running now to chase that; see `CHANGES_size_proportional_sampling.md`
for status.

**Sources**: [Time-MoE (arXiv:2409.16040)](https://arxiv.org/pdf/2409.16040) · [Moirai blog (Salesforce)](https://www.salesforce.com/blog/moirai/)
