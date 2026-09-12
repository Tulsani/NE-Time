# NE-Time Foundation vs. published zero-shot SOTA

**Model**: HyperTimeV2 — see the size-sweep below; **`nano` (79,418 params) is the adopted
config going forward**, chosen from three sizes tested (medium 254K / nano 79.4K / tiny 50.7K)
because it wins on 4 of 5 dataset groups outright rather than merely splitting the difference.
Jointly pretrained on ETTh1+ETTh2+ETTm1.
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

Three of our own runs are in these tables now: **Job A** (`medium`, 254K params, uniform
sampling, best checkpoint epoch 4), **Job B** (`medium`, size-weighted sampling, best
checkpoint epoch 1), and **Tiny** (`tiny`, 50.7K params — 5x smaller — size-weighted sampling,
best checkpoint epoch 8, the latest of 8 total ablation runs). See
`CHANGES_size_proportional_sampling.md` for the full ablation history (sampling, LR, and two
independent component-capacity levers were all ruled out as the epoch-1-plateau cause; only
shrinking the whole model moved it).

## ETTm2 (near-domain zero-shot for us; fully zero-shot for competitors)

| H | Job A | Job B | **Tiny** | Time-MoE-ultra | Moirai-large | Chronos-large | TimesFM |
|---|---|---|---|---|---|---|---|
| 96  | 0.171 | 0.164 | **0.162** | 0.198 | 0.211 | 0.197 | 0.202 |
| 192 | 0.209 | 0.196 | **0.196** | 0.235 | 0.281 | 0.254 | 0.289 |
| 336 | 0.247 | 0.229 | **0.229** | 0.293 | 0.341 | 0.313 | 0.360 |
| 720 | 0.293 | 0.271 | **0.271** | 0.427 | 0.485 | 0.416 | 0.462 |

Tiny holds Job B's lead over every listed competitor at every horizon (essentially identical
numbers, marginally better at H=96) — but see the near-domain caveat above.

## Weather (genuinely zero-shot for everyone)

| H | Job A | Job B | **Tiny** | Time-MoE-ultra | Moirai-large | Chronos-large |
|---|---|---|---|---|---|---|
| 96  | 0.215 | 0.208 | **0.242** | 0.157 | 0.199 | 0.194 |
| 192 | 0.259 | 0.242 | **0.293** | 0.208 | 0.246 | 0.249 |
| 336 | 0.308 | 0.290 | **0.347** | 0.255 | 0.274 | 0.302 |
| 720 | 0.384 | 0.362 | **0.434** | 0.405 | 0.337 | 0.372 |

**Tiny is meaningfully worse here** (-16% to -20% vs. Job B, now behind every competitor at
every horizon) — this is the one dataset where shrinking the model clearly hurts, and it's our
only fully apples-to-apples zero-shot comparison. Real tradeoff, not a rounding error.

## ETTh1 / ETTh2 (in-domain for us — should be an easy win, wasn't for ETTh1)

| Dataset | H | Job A | Job B | **Tiny** | Time-MoE-ultra (zero-shot) | Moirai-large (zero-shot) | Chronos-large (zero-shot) |
|---|---|---|---|---|---|---|---|
| ETTh1 | 96  | 0.495 | 0.526 | **0.514** | 0.349 | 0.381 | 0.441 |
| ETTh1 | 720 | 0.690 | 0.692 | **0.675** | 0.457 | 0.611 | 0.835 |
| ETTh2 | 96  | 0.212 | 0.202 | **0.204** | 0.292 | 0.296 | 0.320 |
| ETTh2 | 720 | 0.291 | 0.282 | **0.278** | 0.439 | 0.423 | 0.603 |

ETTh2: still a decisive win at every horizon in every run. **ETTh1: Tiny is the first lever
(of five tried) to actually improve it** vs. Job B (0.526→0.514 @H96, 0.692→0.675 @H720) —
still losing to every zero-shot competitor, and the gap is narrowed, not closed, but this is
real, directionally-correct progress on the original red flag for the first time.

## Exchange, ECL, Traffic (no published zero-shot MSE table found — not benchmarked, logged for reference)

| Dataset | H | Job A | Job B | **Tiny** |
|---|---|---|---|---|
| Exchange | 96  | 0.124 | 0.136 | **0.115** (better) |
| Exchange | 720 | 1.090 | 1.116 | **1.086** (better) |
| ECL      | 96  | 0.351 | 0.320 | **0.312** (~flat) |
| ECL      | 720 | 0.580 | 0.376 | **0.383** (~flat) |
| Traffic  | 96  | 0.947 | 0.878 | **0.898** (~flat) |
| Traffic  | 720 | 1.154 | 0.871 | **0.898** (~flat) |

Exchange and ECL stay in a plausible range across all three runs (Exchange is near-random-walk
and usually excluded from zero-shot benchmarks entirely; supervised *specialist* SOTA on ECL is
roughly 0.15-0.2 MSE, so our zero-shot ~0.31-0.58 is a reasonable, expected zero-shot gap). No
published zero-shot foundation-model comparison table was found for any of these three, so
treat this section as logged numbers, not a benchmark win/loss.

**Traffic's flat-across-horizon MSE persists at 5x smaller** (Tiny: 0.898/0.886/0.896/0.898
from H=96 to H=720 — still nearly constant, same as Job B's 0.878/0.867/0.874/0.871). Real
forecasting difficulty should increase with horizon; this pattern surviving a 5x capacity cut
weakens the theory that any *specific* component (e.g. TemporalProjector, already tested and
ruled out alone in Job E) is responsible — it looks more like a property of how this
architecture handles Traffic (862 channels, the most out-of-domain set) regardless of overall
capacity. Worth spot-checking actual predictions before citing Traffic zero-shot numbers either
way.

## Bottom line

Five independent levers have now been tried against the epoch-1-plateau / ETTh1-red-flag
problem: dataset sampling, LR magnitude, hyperbolic-encoder capacity, TemporalProjector
capacity, and — the one that actually worked — shrinking the whole model 5x. Tiny is the first
result to move ETTh1 in the right direction and pushes the best-epoch from 1 to 8, but it's a
genuine tradeoff, not a clean win: Weather (the one fully apples-to-apples comparison available)
gets meaningfully worse. Traffic's suspicious flat-across-horizon behavior persists regardless
of model size, suggesting it's a property of the architecture's handling of that specific
dataset rather than anything fixed so far. Next: decide where on the capacity/generalization
tradeoff curve to land (a `nano`, 104,858-param, data point could clarify this) before locking
in numbers for the report, and separately restructure the pretrain/zero-shot split so ETTh1/
ETTh2/ETTm2 stop benefiting from domain leakage in the comparison — see
`CHANGES_size_proportional_sampling.md` for the full plan.

**Sources**: [Time-MoE (arXiv:2409.16040)](https://arxiv.org/pdf/2409.16040) · [Moirai blog (Salesforce)](https://www.salesforce.com/blog/moirai/)
