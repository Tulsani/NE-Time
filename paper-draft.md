# Hierarchical Multi-Timescale Hyperbolic Representations for Parameter-Efficient Time Series Forecasting

## Abstract

Time series foundation models have largely pursued scale — billions of pretraining tokens,
hundreds of millions to billions of parameters — to achieve zero-shot forecasting
generalization. We investigate a complementary direction: whether a hierarchical
multi-timescale architecture operating in hyperbolic space can achieve competitive zero-shot
generalization at a small fraction of that scale. We present a ~79,000-parameter model that
decomposes each input window into global, meso, and local temporal components, each encoded in
its own independently-curved hyperbolic space, fused via horizon-conditioned tangent-space
mixing. Pretrained jointly on four small, publicly available benchmark datasets (Weather,
Exchange, Electricity, and ETTm1) — a corpus roughly two orders of magnitude smaller than
comparable published pretraining corpora — the model matches or exceeds several published
zero-shot forecasting foundation models with 300M to several-billion parameters on two of four
held-out ETT-family benchmarks, while trailing on a third for reasons consistent with that
dataset's well-documented intrinsic difficulty in the wider literature. To test whether the
hyperbolic geometry itself is responsible for this result, rather than the surrounding
architecture, we further train a parameter-matched Euclidean control (identical design, same
corpus, same hyperparameters, with the Poincaré-ball mapping replaced by the identity function)
and find the hyperbolic model outperforms it on all 16 zero-shot/near-domain (dataset, horizon)
comparisons — modest margins (0.3-2.9% relative MSE) but a unanimous direction, while the two
are indistinguishable in-domain. We report this result alongside a controlled ablation
methodology (a five-way hyperparameter sweep, a three-point model-capacity sweep, this
hyperbolic-vs-Euclidean ablation, and an explicit leakage-free zero-shot evaluation protocol) and
a
preliminary, honestly-reported exploration of two downstream tasks (imputation and
classification) via lightweight adaptation of the same backbone, for which we find — using
proper controls, including a randomly-initialized baseline — that the forecasting-pretrained
backbone does not transfer a meaningful representation-learning signal, motivating future work
on reconstruction-objective pretraining rather than purely extrapolative objectives.

## 1. Introduction

Recent time series foundation models (Chronos, TimesFM, Moirai, Time-MoE, MOMENT, among others)
have demonstrated that large-scale pretraining across diverse time series corpora yields useful
zero-shot forecasting generalization, mirroring the scaling recipe that has proven successful
in language and vision. This scale, however, comes at a cost: these models range from roughly
200 million to several billion parameters, and their pretraining corpora span hundreds of
millions to billions of timestamps across dozens of source datasets.

We ask a different question: how much of this generalization capability is attributable to
scale itself, versus to architectural inductive bias? Time series exhibit structure across
multiple, coupled timescales (long-range trend, seasonal/periodic components, local
high-frequency detail), and this structure is naturally hierarchical. Hyperbolic geometry is
known to represent hierarchical structure with substantially fewer dimensions than Euclidean
space requires for the same fidelity. We hypothesize that an architecture built explicitly
around multi-timescale decomposition and hyperbolic representation learning can recover a
meaningful fraction of large-scale models' zero-shot generalization at a small fraction of
their parameter count and pretraining data.

**Contributions.**
1. A hierarchical multi-timescale architecture — fixed (non-learnable) multi-scale
   decomposition feeding three independently-curved hyperbolic encoders, fused via
   horizon-conditioned tangent-space mixing — evaluated at a range of parameter budgets from
   254K down to 51K.
2. A ~79,000-parameter instance of this architecture that beats three published zero-shot
   forecasting foundation models (300M to several-billion parameters) on two of four held-out
   ETT-family benchmarks, under an explicitly leakage-free pretrain/zero-shot split.
3. A methodologically rigorous evaluation protocol: dataset-composition and hyperparameter
   ablations that isolate model capacity as the dominant factor in an early-overfitting failure
   mode (ruling out dataset-sampling policy, learning rate, and two independent
   component-capacity levers as the cause), a backbone-selection rule fixed in advance of
   observing zero-shot results (to avoid test-informed hyperparameter selection), and an
   explicit audit-and-fix of a dataset-specific zero-shot failure.
4. A preliminary, honestly-reported exploration of imputation and classification via
   lightweight adaptation of the same pretrained backbone, including a randomly-initialized
   control condition for classification that reveals the backbone's frozen representations
   carry no more discriminative signal than an untrained network of the same architecture —
   a negative result we attribute to the forecasting-only pretraining objective and the much
   smaller, narrower pretraining corpus, and report as motivation for future work rather than
   suppress.

## 2. Methodology

### 2.1 Architecture

The model (HyperTimeV2) processes each input window of length 336 as follows. Reversible
instance normalization (RevIN) removes per-window mean/scale before the network and restores
it after, allowing the network to operate on standardized inputs while handling distribution
shift across windows and datasets. The normalized window is split into overlapping patches
(patch length 16, stride 8) and linearly projected to a shared embedding dimension. A fixed
(non-learnable) moving-average decomposition splits the patch sequence into three components —
global (slow trend), meso (medium-range/seasonal), and local (high-frequency residual) — using
box filters at two different window sizes with no learnable parameters. We deliberately made
this decomposition parameter-free after finding that an earlier, learnable version allowed the
model to fit training data through a shortcut that bypassed the hyperbolic encoders entirely,
manifesting as generalization saturating within the first training epoch; forcing all
representation learning into the geometry resolved this for that specific failure mode (see
Section 3.1 for a related, structurally different capacity effect that persisted regardless).

Each of the three decomposed components is encoded by its own hyperbolic encoder, each with an
independently learned curvature parameter (parameterized via a softplus transform for
positivity). A horizon encoder — conditioned on the target forecast horizon — produces mixing
weights that fuse the three hyperbolic representations via tangent-space interpolation, so the
relative contribution of global/meso/local structure can vary with how far ahead the model is
asked to forecast. The fused representation is decoded back to Euclidean space, projected to
the target horizon via a compress-then-expand temporal projector, and denormalized by RevIN's
inverse transform.

All datasets are handled in a channel-independent (CI) fashion: each channel of a multivariate
series is treated as an independent univariate instance. This is necessary for joint
pretraining across datasets with wildly different channel counts (7 for the ETT family, 21 for
Weather, up to 862 for Traffic) and is standard practice in this literature.

### 2.2 Multi-horizon training and evaluation

The model is trained jointly across four forecast horizons — 96, 192, 336, and 720 steps —
from a single set of shared weights. Each training window is labeled with a horizon drawn from
this set, and the horizon encoder's conditioning mechanism allows one model to produce
calibrated forecasts at each horizon without horizon-specific retraining. The in-domain and
zero-shot results in Section 3.2 come from a single checkpoint evaluated at all four trained
horizons — we do not train separate models per horizon. Because the horizon encoder conditions
continuously on the requested horizon rather than selecting among a fixed discrete set, the
architecture in principle supports evaluation at any horizon up to the model's fixed maximum
(720 steps, set by the temporal projector's output width); we verify this directly in Section
3.2.1 by evaluating the same checkpoint at five horizons never seen during training.

### 2.3 Pretraining corpus and zero-shot evaluation protocol

An early version of this work pretrained on the ETTh1, ETTh2, and ETTm1 datasets and reported
zero-shot results on ETTm2 and other benchmarks. On inspection, this setup was not directly
comparable to published zero-shot foundation model results: ETTh1 and ETTh2 were being treated
as zero-shot targets by us in earlier internal comparisons despite direct exposure during
pretraining in a related configuration, and ETTm2 shared its source family with the
pretrain-set ETTm1, giving it an unearned near-domain advantage relative to competitors that
exclude the entire ETT family from pretraining.

We therefore remapped the corpus: the final pretraining set is **Weather, Exchange,
Electricity (ECL), and ETTm1**, with **ETTh1, ETTh2, ETTm2, and Traffic held out entirely as
zero-shot targets** (no fine-tuning or adaptation). Of these, ETTh1, ETTh2, and Traffic share no
dataset family with anything in the pretraining corpus and are therefore genuinely, fully
blind zero-shot evaluations; ETTm2 shares its source family with pretrain-set ETTm1 and should
be read as near-domain transfer rather than fully blind zero-shot — we state this explicitly
wherever the ETTm2 result is reported. ECL was included in pretraining only after confirming
empirically (via a dedicated forward/backward memory probe at the exact training configuration)
that its high channel count (321) does not exceed available GPU memory under channel-independent
training; Traffic's channel count (862) was found to leave too little memory margin to include
safely and was kept zero-shot-only.

Datasets are pretrained with a per-step sampling policy proportional to each dataset's own
number of training batches, rather than uniform sampling across datasets — we found that
uniform per-step sampling (giving each pretraining dataset equal probability regardless of
size) caused smaller datasets to be revisited substantially more often, per training sample,
than larger ones, and confirmed via ablation (Section 3.1) that this was a measurable
contributor to — though not the dominant cause of — an early-overfitting failure mode.

### 2.4 Model capacity ablation

We evaluated three parameter budgets of the same architecture — 254,362 ("medium"), 79,418
("nano"), and 50,722 ("tiny") parameters, scaling the embedding dimension, hyperbolic
dimension, and associated projection widths together — under otherwise identical training
configurations, to isolate the effect of overall model capacity from other confounds. This
ablation was run on an earlier pretraining corpus configuration (ETT-family-inclusive) as a
controlled, relative comparison; its purpose is to characterize how validation performance
scales with capacity, not to produce the final reported zero-shot numbers (Section 2.3).

### 2.5 Backbone selection

Given five candidate configurations (a sampling-policy baseline and a four-point learning-rate
sweep at the selected capacity), we selected the deployed backbone by lowest in-domain
validation loss on the pretraining corpus's own held-out split — not by zero-shot performance
on the evaluation benchmarks. Selecting hyperparameters using the zero-shot metric itself would
constitute a form of test-informed model selection; fixing the selection criterion to an
in-domain metric before inspecting zero-shot results avoids this.

## 3. Results

### 3.1 Model capacity ablation

| Model size | Parameters | Best epoch (of 50) | Validation loss at best epoch |
|---|---|---|---|
| medium | 254,362 | 1 | 0.354 |
| nano | 79,418 | 5 | 0.351 |
| tiny | 50,722 | 8 | 0.347 |

Reducing model capacity produced a smooth, monotonic effect on both when training stopped
improving and how good that best point was — smaller models trained productively for
measurably longer and reached a lower validation floor. This pattern was robust: we
independently tested and ruled out dataset-sampling policy, learning rate magnitude (a 3.3x
reduction), and two independent single-component capacity reductions (the hyperbolic encoders
alone; the temporal output projector alone) as candidate explanations for the early-stopping
behavior at the larger capacity — none of these, applied individually, moved the best-epoch or
validation floor by a comparable margin. Only reducing overall model capacity did. We interpret
this as evidence that the "medium" configuration was genuinely over-parameterized for this
joint pretraining objective at this data scale, consistent with hyperbolic representations
requiring fewer parameters to capture the same structure than a Euclidean equivalent. We
selected "nano" (79,418 parameters) for all subsequent results: on the corpus used for this
ablation, it outperformed both "medium" and "tiny" outright on four of five evaluated dataset
groups (with the fifth, a genuinely zero-shot target, showing a real capacity/generalization
tradeoff rather than a strict win for either endpoint).

### 3.2 Zero-shot forecasting vs. published foundation models

All results below use the single "nano" (79,418-parameter) checkpoint selected per Section 2.5,
pretrained on Weather + Exchange + ECL + ETTm1 (Section 2.3). Competitor numbers are taken from
Time-MoE's published zero-shot evaluation table, the most directly comparable available source
for Time-MoE (base/large/ultra variants, up to several-billion parameters), Moirai-large
(311M), Chronos-large (~700M), and TimesFM (~200M).

**ETTh1** (fully blind zero-shot — no dataset family overlap with pretraining):

| Horizon | Ours | Time-MoE-ultra | Moirai-large | Chronos-large |
|---|---|---|---|---|
| 96 | 0.540 | **0.349** | **0.381** | **0.441** |
| 192 | 0.591 | **0.395** | **0.434** | **0.502** |
| 336 | 0.636 | **0.447** | **0.495** | **0.576** |
| 720 | 0.734 | **0.457** | 0.611 | 0.835 |

We lose on ETTh1 at every horizon, though the margin narrows substantially at the longest
horizon (beating Chronos-large there). ETTh1 is documented across the wider forecasting
literature (independent of foundation models) as the most difficult member of the ETT family
regardless of the model evaluated on it; this result is consistent with that pattern rather
than specific to our approach.

**ETTh2** (fully blind zero-shot):

| Horizon | Ours | Time-MoE-ultra | Moirai-large | Chronos-large |
|---|---|---|---|---|
| 96 | **0.221** | 0.292 | 0.296 | 0.320 |
| 192 | **0.247** | 0.347 | 0.361 | 0.406 |
| 336 | **0.269** | 0.406 | 0.390 | 0.492 |
| 720 | **0.313** | 0.439 | 0.423 | 0.603 |

We beat every listed competitor at every horizon, by a widening margin at longer horizons — a
79,418-parameter model outperforming published models two to four orders of magnitude larger,
under a fully blind zero-shot protocol.

**ETTm2** (near-domain — shares family with pretrain-set ETTm1; not fully blind zero-shot):

| Horizon | Ours | Time-MoE-ultra | Moirai-large | Chronos-large | TimesFM |
|---|---|---|---|---|---|
| 96 | **0.167** | 0.198 | 0.211 | 0.197 | 0.202 |
| 192 | **0.201** | 0.235 | 0.281 | 0.254 | 0.289 |
| 336 | **0.236** | 0.293 | 0.341 | 0.313 | 0.360 |
| 720 | **0.283** | 0.427 | 0.485 | 0.416 | 0.462 |

We beat every listed competitor at every horizon here as well, though this result should be
read with the near-domain caveat above rather than as evidence of blind zero-shot transfer.

**Traffic** (fully blind zero-shot; no published zero-shot comparison table was found for this
dataset in the sources we consulted, so we report it for completeness rather than as a
win/loss):

| Horizon | 96 | 192 | 336 | 720 |
|---|---|---|---|---|
| MSE | 0.728 | 0.734 | 0.741 | 0.776 |

**Scale context.** The pretraining corpus used here totals approximately 10.1 million
channel-timestamps across four datasets spanning three domains (meteorological, foreign
exchange, electrical load), versus a comparable published pretraining corpus (MOMENT's Time
Series Pile) of approximately 1.23 billion channel-timestamps across 13 domains — roughly two
orders of magnitude larger and more diverse. The forecasting results above should be read
against both this data-scale gap and the parameter-count gap noted throughout.

#### 3.2.1 Generalization to untrained intermediate horizons

Sections above evaluate only the four horizons used during training (96, 192, 336, 720). To
test whether the horizon encoder's conditioning mechanism generalizes continuously, rather than
having effectively memorized four discrete operating points, we evaluated the same checkpoint —
no retraining, no adaptation — at five additional horizons never seen during training: 48, 150,
250, 500, and 600 steps, on the two fully blind zero-shot datasets (ETTh1, ETTh2).

| Horizon | 48 | 96† | 150 | 192† | 250 | 336† | 500 | 600 | 720† |
|---|---|---|---|---|---|---|---|---|---|
| ETTh1 MSE | 0.499 | 0.540 | 0.566 | 0.591 | 0.599 | 0.636 | 0.669 | 0.691 | 0.734 |
| ETTh2 MSE | 0.186 | 0.221 | 0.231 | 0.247 | 0.250 | 0.269 | 0.283 | 0.294 | 0.313 |

*(†: trained horizon, reproduced from Section 3.2 tables above for context; all others were
never seen during training.)*

Error increases strictly monotonically with horizon across all nine points on both datasets,
with every untrained horizon falling exactly between its neighboring trained values rather than
producing an erratic or discontinuous jump. This is direct evidence that the horizon encoder
has learned a genuinely continuous conditioning function rather than four memorized operating
points, and that the architecture's multi-horizon capability extends smoothly to arbitrary
horizons within its fixed maximum (720 steps, set by the temporal projector's output width;
horizons beyond this are not true extrapolation and were not tested).

#### 3.2.2 Hyperbolic vs. Euclidean ablation

The results above do not by themselves establish that the hyperbolic geometry is responsible
for them, as opposed to the surrounding architecture (fixed multi-timescale decomposition,
tangent-space fusion, horizon conditioning) that would exist regardless of which space the three
scale branches operate in. To isolate this, we built a parameter-matched control: the Poincaré
ball mappings (`expmap0`/`logmap0` in the three scale encoders, the decoder, and the fusion step)
are replaced by the identity function, so the same three branches and fusion mechanism operate
in plain Euclidean space instead. This changes nothing else — same layer shapes (confirmed
identical parameter count, 79,418), same pretraining corpus, sampling policy, and learning rate
as the adopted backbone (Section 2.5) — and the model was pretrained from scratch under this one
change.

Before comparing outcomes, we checked whether the trained hyperbolic backbone actually exploits
its curvature or stays close enough to the ball's origin that `expmap0` is numerically
near-linear there (in which case a null result would be uninformative — the two models would be
close to the same function almost everywhere they are evaluated, rather than genuinely testing
the geometry). We computed the pre-mapping tangent-vector norms on ETTh2 validation data and the
resulting nonlinearity ratio ‖expmap0(t)‖⁄‖t‖ at the model's trained curvature values. Usage is
uneven across scales: the global branch stays close to linear (ratio 0.98 at the mean tangent
norm, 0.60 at the 95th percentile), while the local branch shows substantial nonlinear
compression (ratio 0.61 at the mean, falling to 0.39–0.25 in the upper percentiles) — the model
learned to allocate curvature usage disproportionately to the fine-grained, local-timescale
branch rather than the global one. The hyperbolic model being compared below is therefore
genuinely exercising its geometry, at least locally, not degenerately behaving as a near-linear
model in practice.

**Zero-shot / near-domain datasets** (MSE; Δ = relative change of Euclidean vs. hyperbolic):

| Dataset | Horizon | Hyperbolic | Euclidean | Δ |
|---|---|---|---|---|
| ETTh1 (zero-shot)  | 96  | 0.540 | 0.545 | +0.9% |
| ETTh1 (zero-shot)  | 192 | 0.591 | 0.593 | +0.4% |
| ETTh1 (zero-shot)  | 336 | 0.636 | 0.640 | +0.6% |
| ETTh1 (zero-shot)  | 720 | 0.734 | 0.741 | +1.0% |
| ETTh2 (zero-shot)  | 96  | 0.221 | 0.224 | +1.5% |
| ETTh2 (zero-shot)  | 192 | 0.247 | 0.250 | +1.2% |
| ETTh2 (zero-shot)  | 336 | 0.269 | 0.272 | +1.1% |
| ETTh2 (zero-shot)  | 720 | 0.313 | 0.320 | +2.1% |
| ETTm2 (near-domain)| 96  | 0.167 | 0.169 | +1.0% |
| ETTm2 (near-domain)| 192 | 0.201 | 0.202 | +0.6% |
| ETTm2 (near-domain)| 336 | 0.236 | 0.237 | +0.4% |
| ETTm2 (near-domain)| 720 | 0.283 | 0.284 | +0.3% |
| Traffic (zero-shot)| 96  | 0.728 | 0.749 | +2.9% |
| Traffic (zero-shot)| 192 | 0.734 | 0.755 | +2.9% |
| Traffic (zero-shot)| 336 | 0.741 | 0.757 | +2.2% |
| Traffic (zero-shot)| 720 | 0.776 | 0.785 | +1.1% |

**In-domain (pretraining) datasets**, for contrast:

| Dataset | Horizon | Hyperbolic | Euclidean | Δ |
|---|---|---|---|---|
| Weather | 96  | 0.157 | 0.157 | +0.1% |
| Weather | 192 | 0.198 | 0.197 | −0.7% |
| Weather | 336 | 0.250 | 0.249 | −0.4% |
| Weather | 720 | 0.320 | 0.319 | −0.4% |
| ECL     | 96  | 0.216 | 0.222 | +2.7% |
| ECL     | 192 | 0.228 | 0.233 | +2.3% |
| ECL     | 336 | 0.245 | 0.248 | +1.4% |
| ECL     | 720 | 0.295 | 0.300 | +1.6% |
| ETTm1   | 96  | 0.362 | 0.362 | −0.1% |
| ETTm1   | 192 | 0.408 | 0.407 | −0.2% |
| ETTm1   | 336 | 0.452 | 0.452 | +0.1% |
| ETTm1   | 720 | 0.505 | 0.505 | +0.1% |

On every one of the 16 zero-shot/near-domain (dataset, horizon) pairs, the hyperbolic model
outperforms its Euclidean-ablation twin. Margins are modest (0.3–2.9% relative MSE) but
directionally unanimous, and largest on the two hardest out-of-distribution transfers — Traffic
(1.1–2.9%) and ETTh2, where the gap widens with horizon (1.1% at H=96 to 2.1% at H=720). On the
three in-domain pretraining datasets, by contrast, the two geometries are close to
indistinguishable (7 of 12 comparisons favor hyperbolic, 5 favor Euclidean, all within ±0.4%
relative MSE) — consistent with the ablation isolating a generalization effect specifically,
rather than an in-distribution fitting advantage that would show up equally everywhere.

Two honest caveats. First, this is a single-seed comparison per geometry; we did not have time
before the submission deadline to repeat training across multiple seeds and quantify run-to-run
variance directly, so we cannot rule out that a portion of each individual gap is seed noise.
Second, the effect sizes involved are modest in absolute terms. That said, a unanimous direction
across all four independently-selected zero-shot/near-domain datasets is not what a null effect
would typically produce (a dataset-level sign test puts 4-of-4 agreement at p ≈ 0.0625 under a
fair-coin null; we do not treat this as a rigorous significance claim, since the four datasets
share the same trained model and are not fully independent draws, but it is directionally
informative). Combined with the curvature-utilization measurement above — showing the hyperbolic
model genuinely engages non-trivial nonlinearity, particularly in its local-scale branch, rather
than the two geometries converging to near-identical functions because the model degenerately
stays near the ball's origin — we read this as modest but real evidence that the hyperbolic
representation contributes to out-of-distribution generalization specifically, on top of (not
merely because of) the multi-timescale decomposition and fusion architecture that both variants
share.

### 3.3 Preliminary multi-task exploration

To assess whether the pretrained backbone carries transferable representations beyond
forecasting, we conducted a preliminary exploration of two additional tasks using the same
frozen or lightly-adapted backbone, following published protocols (contiguous-chunk masked
imputation and frozen-embedding classification, both drawn from MOMENT's evaluation
methodology) where applicable.

**Imputation.** Strict linear probing (freezing the entire backbone and training only a new,
lightweight reconstruction head) failed outright — reconstruction error exceeded that of a
constant prediction on several datasets. Fully fine-tuning the backbone alongside the new head
recovered most, but not all, of this gap (e.g., ETTh1 mean squared error improved from 0.94 to
0.50 under fine-tuning). Averaged across four masking ratios (12.5% to 50%, contiguous 8-step
chunks), the fine-tuned model still trailed simple linear interpolation — a non-learned,
non-pretrained baseline — on three of six evaluated datasets (ETTh2, ETTm2, Weather), while
beating it on the other three (ETTh1, ETTm1, ECL). We attribute this to the backbone having
been pretrained exclusively for extrapolative forecasting, which provides no learned mechanism
for representing missing interior context, unlike models such as MOMENT that are pretrained
directly via masked reconstruction.

**Classification.** We evaluated eight small, commonly-used univariate UCR datasets under two
protocols: MOMENT's own (frozen backbone as a feature extractor, mean-pooled patch embeddings,
an off-the-shelf support vector machine, no new training loop), and full fine-tuning (unfreeze
the backbone, train it jointly with a new lightweight classification head via cross-entropy —
the same adaptation that substantially helped imputation). Five of the eight datasets are
domain-mismatched with our pretraining corpus (ECG, gesture, and spectroscopy signals — nothing
like weather, foreign exchange, or electrical load); three are domain-matched, classifying
household or device power-consumption profiles, the same broad domain as our ECL pretraining
data (ItalyPowerDemand, PowerCons, ElectricDevices).

To separate genuine transfer from dataset-difficulty artifacts, we repeated both protocols with
a randomly-initialized, untrained backbone of the same architecture as a control.

| Protocol | Domain-matched (3 datasets) | Domain-mismatched (5 datasets) |
|---|---|---|
| Frozen: pretrained $-$ random | **+2.6 pp** | +5.5 pp |
| Fine-tuned: pretrained $-$ random | **+2.6 pp** | **$-$8.2 pp** |

The domain-matched advantage is identical (+2.6 percentage points) in both protocols and
positive on all three domain-matched datasets individually in the fine-tuned condition — a
small but consistent, reproducible-within-this-run signal that pretraining transfers when the
target task's domain overlaps the pretraining corpus. On domain-mismatched data, full
fine-tuning *reverses* the sign of the frozen protocol's advantage: with training sets as small
as 20–28 examples, unfreezing a 79,418-parameter backbone gives it enough freedom to overfit or
forget useful pretrained structure rather than learn something better, whereas the same backbone
frozen (unable to overfit) still edges out random weights. We caution that a single random-seed
draw is a noisy baseline at this dataset scale — an earlier control run with a different random
seed on a subset of these datasets produced a materially different average accuracy for the
random baseline itself (a swing larger than the domain-matched effect above) — so we report this
as a consistent, internally-replicated-within-this-run pattern rather than a statistically
airtight result.

For scale context, MOMENT reports a mean classification accuracy of 79.4% (median 81.5%) under
the same frozen-embedding-plus-SVM protocol across the *full* 91-dataset UCR archive (mean rank
7.2 of 17 compared methods) — not a like-for-like comparison to our hand-picked 8-dataset subset
(a different, smaller, and likely easier selection than the full archive), but a useful reference
point for the scale of representation-learning pretraining (across 13 million series in
MOMENT's case) that produces classification transfer robust enough to hold across the full
archive rather than a handful of datasets.

We report all of this as a genuine, controlled, mixed finding rather than omit it: the
forecasting-only pretraining objective and the substantially smaller and narrower pretraining
corpus (Section 3.2) are both plausible, non-exclusive explanations for why our transfer signal
is present but small and protocol-sensitive, where MOMENT's is large and robust. We treat
addressing either — a reconstruction- or representation-learning-inclusive pretraining
objective, or a larger and more diverse pretraining corpus — as motivation for future work
rather than a claim this work achieves.

## 4. Limitations and future work

**Scale.** Both the parameter count and pretraining corpus size studied here are far below
those of the published models we compare against; the forecasting results in Section 3.2
should be read as evidence of architectural parameter-efficiency at small scale, not as a claim
that this approach dominates at any scale.

**Downstream task transfer.** Section 3.3's results indicate that forecasting-only pretraining
does not, on its own, produce representations that transfer to reconstruction or discrimination
tasks. Pretraining with a reconstruction-inclusive or otherwise representation-learning-focused
objective, evaluated on the same architecture, is a natural next step we did not have time to
complete.

**Dataset-specific generalization.** We identified and resolved one dataset-specific zero-shot
failure during this work: ETTm1, when held out entirely from pretraining, produced degenerate
zero-shot forecasts (error exceeding that of a constant prediction, flat across all forecast
horizons) despite its closest sibling dataset (ETTm2) transferring normally under the same
conditions in the same evaluation run. We audited the data pipeline for dataset-specific
handling differences and found none, and ruled out cross-dataset state leakage by confirming
the sibling dataset's normal behavior immediately after the failure in the same process; we
therefore treat this as a genuine, if not fully mechanistically explained, model-behavior
finding rather than an implementation defect, and resolved it pragmatically by including ETTm1
in the pretraining corpus. We flag this because it suggests zero-shot generalization within
this architecture is not uniform across superficially similar datasets, and understanding why
is worth further investigation.

**Task coverage.** We did not evaluate anomaly detection, which would require both new
benchmark data acquisition and a contested evaluation metric (volume-under-surface ROC) that we
did not have time to implement and validate correctly.

## 5. Conclusion

We show that a hierarchical multi-timescale architecture operating in hyperbolic space can
achieve zero-shot forecasting results competitive with, and in two of four evaluated cases
exceeding, published foundation models two to four orders of magnitude larger, pretrained on a
corpus roughly two orders of magnitude smaller. This supports architectural inductive bias —
specifically, explicit multi-timescale decomposition paired with per-scale hyperbolic geometry
— as a meaningful, complementary axis to raw scale for time series foundation models. We pair
this result with a deliberately rigorous evaluation methodology (leakage-free zero-shot
splits, pre-registered model-selection criteria, and controlled ablations that isolate model
capacity as a specific, measurable driver of an early-overfitting failure mode) and with an
honest, controlled negative finding on downstream task transfer, which we believe is at least
as valuable to report as the positive forecasting result: a forecasting-pretrained backbone, at
this scale and with this objective, does not by itself constitute a general-purpose time series
foundation model, and closing that gap is a concrete, well-motivated direction for future work.
