# Virtual Cell Challenge 2026

Arc Institute's [Virtual Cell Challenge](https://virtualcellchallenge.org): given
unperturbed control cells from three held-out cell contexts, predict what 300
gene knockdowns do to them, at single-cell resolution.

This repo is a learning project. The goal is to understand the problem and the
scoring well enough to make a defensible submission, not to win.

**Status (2026-10-22):** phase `final`, contexts D/E/F, deadline **Nov 05
23:59 UTC**. 2 submissions/team/day; the latest one is the prize entry; scores
are hidden until winners are announced.

---

## The task, concretely

Per context we are handed 18,400 real control cells over 18,533 genes, and a
list of 300 target genes. For each (context, target) we return 400 predicted
cells. 360,000 cells in total, as whole integer counts.

The ground truth — 138,400 real perturbed cells per context — stays on Arc's
server. We never see it.

## How scoring actually works

This took a while to understand and it reframes the whole problem.

Six metrics are scored (`pds_cosine`, `expr_mse_unbiased_capped_norm`,
`de_wilcoxon_lfc_nmae`, `de_wilcoxon_direction_fidelity_yield_raw`,
`de_wilcoxon_direction_reach_raw`, `de_wilcoxon_sig_jaccard`) and averaged
unweighted. Each is rescaled:

```
s = (u - b) / (r - b)
```

- `u` your value
- `r` **ceiling**: a split-half replicate of the real data — the noise floor of
  the experiment itself
- `b` **baseline**: and this is the thing. From `cell_eval2/baseline.py`:

> *Generic-response baseline: one average perturbation response, emitted for
> every perturbation… It is an ORACLE comparator… the profile is averaged from
> the evaluated real perturbations, so it is transductive and no real submission
> has access to that data. What it bounds is a metric's TRIVIALITY — a score at
> or below it is not evidence of target-specific skill.*

So the zero point is *"predict the same average response for every target,
computed from the answer key."* Total correlation with the truth is not the
currency. **Only per-target deviation from the generic response scores points.**

A hard constraint found in `vcc/prep.py:472` (`_require_counts`): submissions
must be **non-negative whole numbers**. Scaling counts by `2^log2FC` produces
floats, and plain rounding sends a count of 1 × 1.05 back to 1 — erasing every
small effect. We round **stochastically** (`floor + Bernoulli(frac)`), which is
unbiased in expectation, so a 5% shift survives as a 5% chance of incrementing
across 400 cells.

## Submission 1 — the trivial baseline (validation phase)

Rule: resample control cells, zero out the target gene, change nothing else.
Purpose was to exercise the pipeline and locate the zero point empirically.

**Rank 1501 / Overall −0.302746.** The breakdown is the useful part:

| metric | scaled |
|---|---|
| `de_wilcoxon_direction_fidelity` | **−1.717** |
| `de_wilcoxon_sig_jaccard` | −0.083 |
| `de_wilcoxon_direction_reach` | −0.009 |
| `de_wilcoxon_lfc_nmae` | −0.004 |
| `pds_cosine` | −0.004 |
| `expr_mse_unbiased_capped_norm` | 0.000 |

One metric caused the whole loss. Zeroing a single gene leaves all 18,532
others statistically identical to control, so the Wilcoxon tests call **nothing**
differentially expressed — raw fidelity 0.0024 against a baseline of ~0.51.
The penalty is for *calling nothing*, not for being directionally wrong.

Lesson: a prediction must move enough genes, by enough, for the statistical
tests to have something to chew on.

## The model

```
lfc[target] = scale * ( common + lam[target] * (lookup[target] - common) )
```

Built entirely from the [Replogle K562 genome-wide CRISPRi
screen](https://gwps.wi.mit.edu/) — 9,866 perturbations of a *different* cell
line — and no VCC perturbation data at all.

- **`common`** — mean log2FC across all 9,866 K562 perturbations. Identical for
  every target. Mean |log2FC| only 0.025, because averaging thousands of
  perturbations cancels most of the signal.
- **`lookup[target]`** — that gene's own measured K562 response. Covers
  **277 of 300** final targets.
- **`lam[target]`** — ramps from 0 to `lam` with how far the target's K562
  measurement clears the non-targeting noise band; **0** for the 23 uncovered
  targets. Typical realised value ~0.05.
- **`scale`** — flat amplification, needed because `common` is otherwise too
  small to register.

Then: resample real control cells from the context being predicted, multiply
the readout genes by `2^lfc`, stochastically round.

### Why `lam` is deliberately tiny

Measured in a K562→RPE1 transfer rehearsal (`src/transfer_probe*.py`), median
row-wise Pearson r on held-out pseudobulk:

| predictor | r |
|---|---|
| `common` (same for every target) | **+0.207** |
| `lookup` (target's own response) | +0.083 |
| `residual` (`lookup − common`) | +0.047 |
| same-cell-line ceiling (K562gw vs K562ess) | +0.154 |

A λ sweep peaks at **0.1–0.2**, not 1.0. Individual K562 perturbation
measurements are ~60% noise (typical perturbation mean |log2FC| 0.21 vs
control-vs-control 0.12), so trusting a target's own measurement heavily makes
predictions *worse*.

But `common` is worth ≈0 by construction, since that is exactly Arc's baseline.
So the strategy is: use `common` as a **chassis** so the DE tests have something
to call, and earn points from the small, noisy, genuine per-target residual.

A hypothesis that the data **refuted**: that VCC targets would be unusually
well-measured in K562. They are statistically indistinguishable from average
Replogle perturbations (168 vs 169 cells; 1.27× vs 1.28× effect size). A
VCC-target-only common axis is *worse* than the global one (r 0.051 vs 0.112) —
estimation noise dominates, so use all 9,866.

## Scoring ourselves without the leaderboard

The final phase returns no scores, so we built our own feedback loop:

1. Arc published the **2025** challenge's training data — 221,273 cells, 150
   perturbations, same assay and cell type conventions.
2. Arc ships the scorer as a pip package, `cell-eval2==0.19.0`, including the
   `vcc2026` profile and `build_generic_baseline` / `compute_ceiling`.
3. Our model uses **no** 2025 perturbation data, so all 150 are held out by
   construction. Genuine zero-shot.

Validity checks (`src/rehearsal_overlap.py`): 18,077 of 18,080 genes shared
with the 2026 panel; K562 covers 136/150 of the 2025 perturbations vs 277/300 of
2026's; only **12 perturbations overlap between the two panels**, so no leakage.
Caveats: 2025 targets are stronger (1.30× vs 1.20× null) and sequenced 2.7×
deeper (54k vs 20k UMI), so the rehearsal reads slightly optimistic and is more
DE-sensitive than D/E/F will be.

### Rehearsal result (40 perturbations, 150 cells each, `scale=4.0`, `lam=0.15`)

| metric | scaled | vs submission 1 |
|---|---|---|
| `de_wilcoxon_direction_fidelity` | **+0.386** | −1.717 |
| `pds_cosine` | **+0.346** | −0.004 |
| `de_wilcoxon_sig_jaccard` | +0.049 | −0.083 |
| `expr_mse_unbiased_capped_norm` | 0.000 | 0.000 |
| `de_wilcoxon_lfc_nmae` | −0.071 | −0.004 |
| `de_wilcoxon_direction_reach` | −0.102 | −0.009 |
| **average** | **+0.101** | −0.303 |

Above the triviality line on held-out data. Raw fidelity 0.482 vs 0.0024 —
the failure mode of submission 1 is structurally fixed.

**Read this number with both thumbs on the scale:**

- *Pessimistic:* `compute_ceiling` returned NaN for every metric but
  `pds_cosine` at this size (150 cells ÷ 5 splits = 30/split, too few), so
  `score_metrics` fell back to `r = 1.0`. Confirmed from the arithmetic:
  (0.482−0.156)/(1−0.156) = 0.386. The real leaderboard's ceiling is lower —
  back-solving submission 1 puts it near 0.805 for fidelity — and a lower
  ceiling makes scores *larger*.
- *Optimistic:* the generic baseline's own raw fidelity here is 0.156, against
  ~0.51 on the real validation panel. Thin data weakens the baseline, and a
  weak baseline is easy to beat. This is the bigger bias.

### Fitting the dials

Both references (`build_generic_baseline`, `compute_lfc_nmae_reference`) are
functions of the ground truth alone, so they are cached per rehearsal size and
a sweep point costs one DE pass (~20 s).

**`scale`**, at λ=0.15:

| scale | fidelity | lfc_nmae | reach | jaccard | pds | **avg** |
|---|---|---|---|---|---|---|
| 4 | +0.386 | −0.071 | −0.102 | +0.049 | +0.346 | +0.1013 |
| **6** | +0.451 | −0.102 | −0.105 | +0.062 | +0.343 | **+0.1081** |
| 9 | +0.455 | −0.197 | −0.105 | +0.069 | +0.320 | +0.0904 |
| 13 | +0.447 | −0.364 | −0.067 | +0.073 | +0.290 | +0.0633 |
| 18 | +0.441 | −0.659 | −0.067 | +0.077 | +0.286 | +0.0130 |

A hypothesis this **refuted**: that the negative `reach` meant the predictions
were too timid. Reach is essentially *flat* in `scale` (−0.102 → −0.105 → −0.067).
Reach is about *which* genes move, not how far, so uniform amplification cannot
buy it — only target-specific accuracy can. What amplification does buy is
fidelity and jaccard, and what it costs is `lfc_nmae`, which penalises
over-stated magnitudes hard. The optimum is where those cross, at `scale ≈ 6`.

**λ**, at scale=6:

| λ | avg_score |
|---|---|
| 0.0 — generic response only | +0.0491 |
| 0.15 | +0.1081 |
| 0.30 | +0.1095 |
| 0.50 | +0.0814 |

**The per-target component more than doubles the score** (0.049 → 0.110). This
is the central claim of the whole approach, measured on held-out data: the
noisy K562 residual carries real, transferable, target-specific signal.

Two further notes:

- λ=0 scores **+0.049, not 0**, even though the pure generic response should be
  worth exactly 0 against a generic-response baseline. The difference is that
  Arc's baseline emits an *averaged profile* while we emit *real control cells
  that have been rescaled*, which preserves realistic cell-to-cell variance.
  Modest, but free.
- The λ optimum is flat between 0.15 and 0.30 (Δ = 0.0014 over 40
  perturbations, i.e. noise). **Shipped λ = 0.2**, where the rehearsal and the
  independent RPE1 transfer probe (0.1–0.2) agree, rather than the raw argmax
  of a small rehearsal.

### Denoising the residual by pooling across genes — REFUTED

The standing bottleneck was that the per-target residual is the only thing
that scores and it is also the noisiest thing we have: each Replogle
perturbation was read from a median of **168 cells**, so the residual transfers
to a held-out cell line at r ≈ 0.06–0.09. λ is pinned near 0.2 because of it.
The obvious fix, and the one this README previously proposed: average a
target's residual with its nearest neighbours in residual space, on the theory
that functionally similar genes share true signal but not measurement noise.

**It denoises exactly as predicted, and it scores worse.** Both halves are
measured.

The denoising is real. Built from K562-gwps only, scored against RPE1
(`src/denoise_pool.py`, 2,052 perturbations measured in all three screens):

| neighbour pool | pool size | k | residual r → RPE1 | r (strong) | sign@100 |
|---|---|---|---|---|---|
| none | — | — | +0.064 | +0.086 | 0.540 |
| the 2,052 best-measured | 2,052 | 10 | +0.071 | +0.127 | 0.590 |
| all perturbations | 10,673 | 10 | +0.083 | +0.124 | 0.600 |
| **≥100 cells measured** | **7,816** | **10** | **+0.088** | **+0.129** | **0.600** |
| ≥200 cells measured | 3,921 | 100 | +0.092 | +0.130 | 0.610 |

All 15 configurations beat not pooling, on both metrics. Filtering the pool to
perturbations read from ≥100 cells helps: a neighbour is only worth borrowing
from if it was measured well itself, and the high-quality 2,052-row pool is
*worse* than the filtered 7,816-row one because 2,052 is too few to choose
from.

Then the same artifact, under Arc's own scorer (40 perturbations, scale=6,
`external/k562_axis.npz` `pool{k}` arrays, `rehearsal_score.py --denoises`):

| variant | λ | mean \|log2FC\| | avg_score |
|---|---|---|---|
| **raw (shipped)** | **0.2** | **0.2184** | **+0.1145** |
| pool10 | 0.5 | 0.2689 | +0.0925 |
| pool10 | 0.2 | 0.1836 | +0.0912 |
| pool30 | 0.5 | — | +0.0876 |
| pool100 | 0.5 | — | +0.0870 |
| pool30 | 0.2 | — | +0.0843 |
| raw | 0.5 | 0.3851 | +0.0814 |
| pool100 | 1.0 | — | +0.0754 |
| pool100 | 0.2 | — | +0.0753 |
| pool30 | 1.0 | — | +0.0553 |
| pool10 | 1.0 | 0.4304 | +0.0270 |

Pooling shrinks the residual (to 0.55× at k=10, 0.45× at k=30, 0.34× at
k=100), so λ has to rise to compare fairly — and the magnitude column kills
that defence: pool10 at λ=0.5 carries **23% more** signal magnitude than the
shipped setting and still scores 0.022 lower.

**Why, per metric** — and it is not that the denoising failed:

| metric | raw λ=0.2 | pool10 λ=0.5 | pool30 λ=0.5 |
|---|---|---|---|
| `de_wilcoxon_direction_fidelity` | +0.4562 | **+0.4657** | **+0.4667** |
| `de_wilcoxon_sig_jaccard` | +0.0643 | **+0.0740** | **+0.0722** |
| `de_wilcoxon_lfc_nmae` | −0.1245 | −0.2098 | −0.1766 |
| `de_wilcoxon_direction_reach` | −0.0900 | −0.1232 | −0.1316 |
| **`pds_cosine`** | **+0.3810** | +0.3481 | +0.2949 |
| avg | **+0.1145** | +0.0925 | +0.0876 |

Pooling *improves* both direction metrics, which is the denoising showing up
where it should. It loses more on `pds_cosine` — the **perturbation
discrimination** score, which asks whether the 300 targets can be told apart.
Pooling pulls each target toward its neighbours' average, so the predictions
become mutually more similar and harder to distinguish. The metric named after
target-specificity is the one that punishes blurring targets together.
`lfc_nmae` costs another ~0.05, because restoring magnitude means raising λ and
that metric penalises overstated magnitudes hard.

Keeping λ at 0.2 does not rescue it either (+0.0912 / +0.0843 / +0.0753): that
forfeits the fidelity gain without recovering the discrimination loss.

Two lessons, the second more general than the first:

- **A better estimate of the response is not a better submission.** Agreement
  with held-out truth went up and the score went down. This is the same shape
  as the `cell_set_len` 512 result below, where val loss improved 29% and the
  score fell — and the same shape as the README's own framing that total
  correlation is not the currency. Three independent demonstrations now.
- **Any denoiser that pools across targets is suspect by construction**, since
  pooling trades the exact quantity `pds_cosine` measures. A denoiser worth
  trying would have to cut noise *without* moving targets toward each other —
  e.g. more cells per perturbation, or a pooling that is subtracted back out.

Cost: ~$0.35 of Modal for 16 sweep points in one container, after the local
probes. Cheap for a refutation of the plan of record.

## Submission 2 — the transfer model (final phase)

Shipped `scale=6.0`, `lam=0.2`, `lam_mode=ramp`, `lam_thr=2.0` over contexts
D/E/F. Rehearsal avg_score **+0.11** against −0.30 for submission 1.

Entry `umpZxC06qd3VDMERWNwm`, 360,000 cells / 2.9 GiB, status **published** with
`error_info: null` — so the stochastically-rounded integer counts pass
server-side validation, not just the local `prep` gates.

**Confirmed: no metrics leak in the final phase.** A hypothesis worth recording
as refuted — `vcc status --json` passes the raw entry through, so it seemed
possible Arc nulls only the scaled `score_*` fields and lets the raw
`cell_eval2` values (`pds_cosine`, `de_wilcoxon_*`) through. They are **all**
null, along with `rank` and `score_avg`. Nothing comes back from any submission
until winners are announced, which makes the rehearsal harness the only
instrument available for the rest of the competition.

## Infrastructure notes

Everything heavy runs on Modal (app `vcc-2026`, Volume `vcc-data`) to keep
multi-GB intermediates off the laptop.

- **`pdex` is not optional.** `cell-eval2` picks it up as its CPU
  differential-expression backend and silently falls back to `scanpy` without
  it. One DE pass: **20 s with pdex, ~45 min with scanpy** (~135×). The engines
  do not agree numerically, so DE caches must not be mixed across them.
- **Modal preempts containers.** A long run must checkpoint to the Volume and
  resume, or preemption silently re-does the expensive part — one run paid three
  times for the same ground-truth arm. `rehearsal_score.py` now writes the real
  arm before the prediction work and reloads it if present.
- Use two independent RNG streams when caching part of a pipeline, so reusing a
  cache cannot shift what the rest of it samples.
- Don't pipe Modal output through PowerShell's `Select-Object` — it truncates
  errors into unreadable `RemoteError`s.

Spend to date ≈ **$13.40** (transfer model ~$1.4, STATE route ~$11.6,
pooling refutation ~$0.35).

## Layout

```
src/common.py             Modal app, images, Volume paths, shared lam helper
src/build_axis.py         K562 common axis + per-target lookups -> external/k562_axis.npz
src/transfer_probe*.py    K562->RPE1 transfer experiments (what lam should be)
src/rehearsal.py          fetch Arc's 2025 data to the Volume
src/rehearsal_overlap.py  is 2025 a valid stand-in for D/E/F?
src/rehearsal_score.py    score the model with Arc's own scorer (the feedback loop)
src/predict.py            build + package the real D/E/F submission
src/submit.py             upload a .vcc from the Volume
src/baseline.py           submission 1, the trivial knockout
src/check_final.py        D/E/F vs A/B/C
src/coverage_final.py     K562 coverage of the final target list
src/denoise_probe.py      can pooling across genes denoise the residual?
src/denoise_pool.py       which neighbour pool, and is n=35 just underpowered?
src/denoise_verify.py     do the SHIPPED pooled arrays transfer better?
src/replogle_fetch.py     fetch the Replogle single-cell screens (74.5 GB)
src/replogle_prep.py      shared gene space + log1p(CP10k) + dense -> sparse
src/replogle_bridge.py    project/lift between Replogle's 6,714 genes and a panel
src/state_norm.py         log1p(CP10k) copies of the 2025 splits
src/state_prep.py         2025 data -> cell-load layout + TOML
src/state_perts.py        ESM-2 perturbation vectors (bulk UniProt fetch)
src/state_train.py        STATE training; local-disk staging, contiguous rewrite
src/state_infer.py        inference, counts conversion, ensembling, scoring
src/submit_ensemble.py    the D/E/F ensemble submission end to end
```

---

# Part 2 — STATE with an ESM-2 perturbation encoder

The transfer model is a floor: its signal is a residual that transfers at
r ≈ 0.05 and its dials are fitted, so the next gain has to come from a model
that does something the lookup cannot. Two things, specifically:

- STATE conditions on **the individual cell** (`cell_set_len` cells per set,
  bidirectional attention, `predict_residual`). The transfer model applies one
  identical fold-change to all 400 cells and discards all heterogeneity.
- A learned map from `gene → response` can **pool across similar genes**,
  which is the only way to beat the noise that forces λ ≈ 0.2.

## The blocker, and why it is not the model's fault

STATE identifies a perturbation by a slot in a fixed vocabulary
(`pert_dim: 2024`) with **zero overlap** with the 2026 target list, and
`cell-load` fills an unknown perturbation with a zero vector while
`_infer.py:802-804` falls back to the control vector. Out of the box it would
predict *no change* for all 300 targets and reproduce submission 1's −1.72.
That is an encoder-coverage failure that looks exactly like a model failure,
and only a local harness distinguishes them.

Replacing it needs **no code changes**
(`cell_load/data_modules/perturbation_dataloader.py:567`):

```python
if self.perturbation_features_file:
    featurization_dict = torch.load(self.perturbation_features_file)
    self.pert_onehot_map = featurization_dict   # use the custom featurizations
...
pert_dim = next(iter(self.pert_onehot_map.values())).shape[0]   # auto
```

So `src/state_perts.py` writes `{gene: ESM-2 vector}` and the config points at
it. **572/572 genes resolved from UniProt (reviewed/Swiss-Prot), zero missing**
— coverage of the 2026 panel goes from **0/300 to 300/300**.

### What Arc's released checkpoints actually contain

Worth settling, because the obvious question is "isn't Replogle already in
STATE's training data?" It is. From
`external/state_ckpt_meta/zeroshot__k562__config.yaml`:

```
toml_config_path: .../revisions/replogle_nogwps_v2/k562_zeroshot.toml
tags: [release, replogle]
cell_type_key: cell_line
embed_key: X_state
output_space: gene
pert_rep: onehot
```

So they trained on Replogle, across cell lines, with one checkpoint per
held-out line (k562, rpe1, hepg2, jurkat). But **`nogwps`** — their Replogle
training set *excludes the genome-wide screen*. That is the 65.8 GB file with
all 9,866 perturbations, and it covers **277 of our 300 targets** while the
screens they did train on cover **0**. The most useful part of Replogle for
this challenge is precisely the part no released checkpoint has seen.

And the weights are unusable for us regardless. From the checkpoint's own
`var_dims.pkl` and `pert_onehot_map.pt`:

| | value | consequence |
|---|---|---|
| `pert_dim` | 2024 one-hot slots | **0 of our 300 targets** are in the vocabulary (verified, not inferred; it is a list of essential genes — `AAAS, AAMP, AAR2, AARS…`). Swapping in ESM-2's 1,280-dim vectors replaces the perturbation encoder's input layer, so there is nothing to inherit in the one part that would need to know our genes. |
| `hvg_dim` / `output_dim` | 2000 / 2058 | we are graded on 18,533; the decoder is discarded |
| `hidden_size` / heads | 328 / 12 | 328 is not divisible by 12 and current `transformers` validates it, so the released hyper-parameters do not instantiate at all |
| `embed_key` | `X_state` | consumes SE-model embeddings, not counts — a second model to stand up |

Strip the perturbation encoder and the decoder and what is left to inherit is
the 8-layer middle, which will not build and expects different inputs. So
training from scratch is not a choice to relearn what Arc knows; it is forced.
What we reuse is their *recipe*.

Two claims that turned out to be wrong, both worth recording:

- *"The released checkpoints only predict 2,000 genes, so STATE is
  incompatible."* Wrong. `base.py:300` picks `hvg_dim` for
  `output_space="gene"` and `gene_dim` for `output_space="all"`; their own
  default data config ships `output_space: all`, and there is a dedicated
  `gene_dim > 10000` decoder branch. 2,000 was that artifact's training data,
  not an architectural cap. Only the released *decoder weights* are stuck at
  2,000 — which is a reason not to fine-tune them, not to avoid STATE.
- *"We need the SE embedding model."* No: `embed_key: null` trains straight
  off gene counts.

## Our own training setup

| | |
|---|---|
| data | Arc's 2025 splits — 150 train / 50 val / 100 test, **disjoint** perturbations |
| holdout | `[fewshot]` with explicit gene lists, not three fake cell types |
| perturbation rep | ESM-2 650M, 1280-dim, mean-pooled over residues |
| gene space | 18,080, `output_space: all` |
| size | 240M params (215M trainable); `hidden_dim` 768, 8-layer Llama backbone |

## The bug that cost three scoring runs

`state tx infer` clips its output to **[0, 14]**. That bound is only sensible
in log1p space (`log1p(1.2M) ≈ 14`). Our first run set `is_log1p=false` — a
true statement about the raw 2025 counts, but it puts the model in **count
space**, where the clip saturates every highly expressed gene. And
`normalize_counts=true` does nothing: in cell-load 0.10.4 that option is
stored and round-tripped but **never applied to any data**.

The symptom was diagnostic in hindsight: direction metrics healthy, magnitude
destroyed, `de_wilcoxon_lfc_nmae` pinned at its −6.0 penalty cap.

| attempt | training space | conversion | nmae | avg |
|---|---|---|---|---|
| v2 | raw counts | `expm1` (wrong) | −6.000 (clamp) | −0.892 |
| v2 | raw counts | none (right for v2) | −2.010 | −0.240 |
| **v3** | **log1p(CP10k)** | **`expm1`** | **−0.217** | **+0.0656** |

Note `expm1` is wrong in row 1 and right in row 3 — same call, opposite
correctness, because the training space changed. `src/state_norm.py` does the
normalisation by transforming `X/data` in place on a copy (CP10k + log1p is
per-row and preserves sparsity, so `indices`/`indptr` are untouched and we
never hold 1.9B nonzeros in memory). Verified: max 4.85, p99 2.18, row sums
exactly 10,000, and the model's own output row-sums come back at 9,402.

A re-anchoring "fix" that was rejected: `control_source="pred"` lifted the
average to −0.088 by computing fold-changes against STATE's own simulated
controls. It is not faithful — our submissions contain only perturbed cells,
so the server must anchor on real controls — and it destroyed fidelity
(+0.343 → −0.281) because the anchor absorbed the signal too.

## Head-to-head, same cells, same references

Test split, 20 perturbations, both models scored against the same cached
baseline and lfc_nmae reference:

| metric | STATE v3 | K562 transfer |
|---|---|---|
| `de_wilcoxon_lfc_nmae` | −0.217 | −0.224 |
| `de_wilcoxon_direction_fidelity` | +0.336 | **+0.347** |
| `de_wilcoxon_direction_reach` | −0.019 | **+0.010** |
| `de_wilcoxon_sig_jaccard` | +0.136 | **+0.157** |
| `pds_cosine` | **+0.158** | +0.032 |
| **avg** | **+0.0656** | +0.0535 |

Both clear the triviality line. STATE's entire margin is `pds_cosine`, where it
is 5× better — the per-cell-profile metric, which is exactly what its
cell-conditioning buys. The transfer model is equal or slightly ahead on all
four DE metrics.

**Do not read this as STATE winning.** K562 covers only **4 of these 20**
targets, so λ was ~0 and the transfer model was reduced to its generic-response
floor (independently measured at +0.049). On the 2026 panel it has 92%
coverage, where the λ term previously doubled its score (0.049 → 0.108). The
full-coverage comparison is **unmeasured**, and 0.012 at n=20 is inside noise.

**Scores from different splits are not comparable.** The submitted model's
+0.108 was measured on the train split against that split's own baseline and
ceiling; +0.0656 is against the test split's. This is why the transfer model
had to be re-scored on test rather than reusing +0.108.

## Operational lessons

**Keeping a long run alive.** This cost a day:

| mechanism | survives client disconnect | survives local process kill |
|---|---|---|
| `modal run` | no | no |
| `modal run --detach` | **yes** | **no** |
| `spawn()` on an ephemeral app | no | no |
| `spawn()` on a **deployed** app | yes | yes |

`--detach` is the documented answer and covers only a clean disconnect. When
the local shell was killed for memory pressure, a 12,000-step run died at step
1,349. `spawn()` alone does not help either — `modal run` tears down its
ephemeral app on exit and takes queued calls with it. Only `modal deploy` plus
`spawn()` is independent of this machine. **Always confirm a live task
afterwards**: both failed launches printed success.

Also: `model.checkpoint=` does **not** resume training — it is read at
inference only. A "continuation" run restarted from random init (train loss
78.1 at step 49, versus 77.4 for the original). `model.kwargs.init_from` is the
likely hook; unverified.

**Version drift in Arc's stack.** Every one of these cost a failed run:

- `arc-state 0.11.3` exists only in their git tree; PyPI's latest is 0.11.1,
  which lacks `model.kwargs.log1p_from_raw_counts` (hydra: "not in struct").
- `cell-load` has releases declaring both `>=3.13` and `<3.13`; on Python 3.13
  pip silently resolves to a **1.2 kB stub of the same name**. Pin the
  interpreter and the version together.
- `hidden_dim: 328` from their released checkpoints is **invalid** under
  current `transformers` — 328 is not divisible by 12 attention heads, and the
  Llama config now validates that. Their own released hyper-parameters cannot
  be reconstructed today.
- `state tx infer` dies on its own checkpoints under torch ≥ 2.6
  (`weights_only=True`). Patching `torch.load` is not enough (lightning holds a
  direct reference); needs `add_safe_globals` plus patching
  `torch.serialization.load`, called in-process rather than via their CLI.
- Keys read from `**kwargs` and absent from the hydra schema (`is_log1p`,
  `normalize_counts`) need the `+key=value` append form.

**Smoke runs pay for themselves.** Three config incompatibilities died in
under a minute each for a few cents, any of which would have wasted hours of
GPU.

## Part 3 — ensembling, and a failed architecture change

All numbers below are the **same 20 test perturbations**, same cached baseline
and lfc_nmae reference.

| prediction | avg |
|---|---|
| **ensemble v3 + transfer, w=0.5** | **+0.1041** |
| 3-way (v3 + v4 + transfer) | +0.1033 |
| ensemble w=0.3 / w=0.7 | +0.0975 / +0.0932 |
| ensemble v4 + transfer | +0.0751 |
| STATE v3 (`cell_set_len` 64) | +0.0656 |
| K562 transfer | +0.0535 |
| STATE v4 (`cell_set_len` 512) | +0.0516 |

### The ensemble works, and it is the best thing we have

Averaging STATE v3 and the transfer model 50/50 (then stochastically rounding)
scores **+0.1041 against +0.0535 for the transfer model alone on the same
cells** — close to double, for ~$0.50 of compute and no new model. The blend
sweep is concave with a clean peak at equal weighting, which is also the least
overfit choice.

It works because the two models fail differently: STATE wins `pds_cosine` 5×
(the per-cell-profile metric its cell-conditioning buys), the transfer model is
equal or ahead on all four DE metrics. Complementary, not redundant.

Likely a **floor, not a ceiling**: the transfer arm is running at λ ≈ 0 here
(K562 covers only 4 of these 20 genes), so it contributes just its generic
response. On the 2026 panel it has 92% coverage.

### `cell_set_len` 512 made the model better and the score worse

Held cells-per-step and total data volume identical to v3 (6,000 × 2 × 512 =
6,000 × 16 × 64 = 6.14M cell-presentations), so set size was the only variable.

| | v3 (set 64) | v4 (set 512) |
|---|---|---|
| best val loss | 0.544 | **0.388** |
| `de_wilcoxon_lfc_nmae` | −0.217 | **−0.055** |
| `de_wilcoxon_direction_reach` | −0.019 | **+0.017** |
| `de_wilcoxon_direction_fidelity` | **+0.336** | +0.316 |
| `de_wilcoxon_sig_jaccard` | **+0.136** | +0.110 |
| `pds_cosine` | **+0.158** | **−0.079** |
| avg | **+0.0656** | +0.0516 |

Two findings:

- **Val loss is not the competition metric.** v4 is 29% better on validation
  loss and *worse* on the leaderboard metric. Demonstrated, not asserted.
- **Larger cell sets trade per-cell fidelity for aggregate accuracy.** v4 fixed
  magnitudes (nmae 3.9× better) and reach, but `pds_cosine` collapsed from
  +0.158 to −0.079 — and `pds_cosine` is exactly what makes STATE worth
  ensembling. Prediction, untested: `cell_set_len` *below* 64 may push
  `pds_cosine` higher, which is the lever the ensemble rewards.

A prior hypothesis **refuted**: I expected bigger sets to *strengthen*
`pds_cosine`, since set-level attention is the mechanism behind STATE's edge
there. It did the opposite.

## Part 4 — a measurement bug, and what the corrected numbers say

### The coverage bug

`build_axis.py` built K562 lookups only for the 2025 **Training** perturbations
and the 2026 final targets. The val and test perturbations were never
requested. So on the test split the transfer model ran at λ ≈ 0 — and Part 3
recorded that as a fact about K562 ("covers only 4 of these 20 targets")
instead of as our own omission.

K562 actually measured **97 of the 100** test perturbations and 47 of 50 val.
With the lists added, lookups went 401 → 529 targets and coverage on those 20
went **4/20 → 17/20**.

Consequence: **every Part 3 number compared a model against a handicapped
transfer arm.** The live D/E/F submission was never affected — the 2026 targets
were always in the `want` set at 277/300.

### Corrected head-to-head (same 20 test perturbations, same references)

| model | avg | was |
|---|---|---|
| **ensemble v3 + transfer, w=0.5** | **+0.1134** | +0.1041 |
| STATE v3 | +0.0656 | — |
| K562 transfer | +0.0598 | +0.0535 |
| Replogle v2 (calibrated) | −0.3856 | — |

Fixing the bug *helped the ensemble more than the transfer model* (+0.0093 vs
+0.0063), because the arms stay complementary: `pds_cosine` is +0.158 for v3,
+0.168 for transfer, and **+0.279** for the blend. The ensemble is now the
measured best by 1.9×, under conditions far closer to D/E/F than before.

## Part 5 — STATE trained on Replogle

**Read this with the correction in "Next" #1.** This section concluded that the
model's 37% gene coverage was a structural ceiling. That conclusion is wrong:
the transfer model has the same coverage and scores +0.168 on `pds_cosine`
against this model's -0.079. The binding problem was how the output cells were
assembled, not how many genes were covered.

Built it: 9,866 K562 + 2,393 RPE1 perturbations (12,259 across two cell lines,
versus 150 in one), ESM-2 vectors for 9,873 of 9,912 genes, 25,000 steps.

**Its DE behaviour is the best of anything we have.** Against v3 / transfer:
`sig_jaccard` **+0.222** (vs +0.136 / +0.157), `direction_fidelity` +0.352
(+0.336 / +0.347), `direction_reach` +0.023 (−0.019 / +0.010). The 65×
perturbation diversity carries real target-specific signal.

**We concluded it could not win, and that conclusion does not hold.** The
arithmetic below is correct as arithmetic, but it assumes `pds_cosine` is
pinned at -0.079 by gene coverage. See "Next" #1: coverage is not what pins it.
Replogle measured 8,248
genes, so after intersecting with the panels the model covers **6,714 — 37% of
the 18,080 we are graded on**, against v3's ~100%. `pds_cosine` is therefore
−0.079. Grant a *perfect* fix to the one broken metric and the arithmetic is:

```
(-0.217 + 0 + 0.352 + 0.023 + 0.222 - 0.079) / 6 = +0.0497
```

≈ the transfer model, below v3, far below the ensemble. No inference-time fix
reaches it; the ceiling is set by which genes Replogle measured.

Three things worth keeping:

- **The −6.0 cap was an anchoring problem, and `calibrate` is the cure.** Raw
  score −0.9072 with `de_wilcoxon_lfc_nmae` pinned at its floor while direction
  metrics were healthy. Per-gene calibration against the model's own predicted
  controls took it to −2.83 and the average to −0.3856. The diagnostic: the
  model's predicted controls were **5.3× off per gene** from real H1 controls,
  with 33% of genes beyond the ±10 clip. A model trained on K562/RPE1 predicts
  in its own expression space, and every fold-change against real H1 controls
  inherits that offset.
- **Clipping the output range is not the cure.** Trimming log-space output from
  14 to 5 (the training data's max was 4.85, the model saturated at 14) changed
  the average by −0.0002. Directions were never the problem.
- **Training converged.** Val loss 0.8346 → 0.7810 by step 17,499, then flat
  through 25,000. More compute was not the missing ingredient.

### Operational lessons that cost real hours

- **Random reads of a compressed CSR file dominated everything.** `chunks=(1<<20,)`
  with `lzf` meant 1,024 random cells touched ~927 chunks, decompressing ~7.4 GB
  to use ~22 MB — 340× amplification, 10.4 s/step against 0.046 s of GPU. The
  GPU was idle **99.6%** of the time, so a faster one would have changed
  nothing. Rewriting X contiguous and uncompressed gave **0.28 s/step, a 37×
  speedup**. Check `batch_time_avg` against wall clock before buying hardware.
- **`ckpt_every_n_steps` does not produce checkpoints; validation does.** A run
  with `ckpt_every=2500, val_freq=5000` wrote nothing, then died on a 6 h
  function timeout at step ~3,149. Three settings had to agree and none did.
  Keep `ckpt_every == val_freq`.
- **`modal run --detach` does not survive the local client being killed.** The
  harness reaps background shells under memory pressure, which cancelled a run
  4 minutes in. Only `modal deploy` + `spawn()` is independent of the laptop.
- **Existence is not completeness.** A preempted container left a truncated
  28-part file that the resume logic skipped forever because `os.path.exists`
  was true; the concat failed 29 parts later. Validate the artifact.
- **Read `metrics.csv`, not the progress bar.** Lightning's bar uses carriage
  returns and never reaches `modal app logs`, so step progress looked invisible
  for hours while the CSV had it all along.

## State of play

- **Live entry:** the **ensemble** (STATE v3 + K562 transfer, w=0.5) over
  D/E/F, entry `1S1L62Zw9J9bT9yoVbyq`, status **published** — 360,000 cells,
  18,533 genes, 4.52 GB. Measured **+0.1134** against +0.0598 for the transfer
  model alone on identical cells and references. Replaces
  `umpZxC06qd3VDMERWNwm` (transfer only), since the latest submission is the
  prize entry.
- **Submission-path lessons.** Two failures only appeared at full scale: a
  LIL-based lift that worked at 62,239 rows exhausted 49 GB at 120,000 (fixed
  by chunked COO assembly over explicit column indices), and `vcc prep` died on
  `ValueError: Output dtype not compatible with inputs` because the blend has
  **2.69e9 nonzeros** with int32 `indices` — past int32's 2.147e9. Rewriting
  indices as int64 preserves the values exactly. Blending takes the UNION of
  both arms' nonzero patterns, so a blended submission is ~1.4x the nonzeros of
  either arm; budget for it.
- **Untested idea worth trying first next time:** blend as a per-cell MIXTURE
  (take each output cell wholly from one arm) rather than by averaging counts.
  Averaged cells carry 7,484 nonzero genes against ~5,679 in real data — 1.32x
  too dense, and each is a smear of two models rather than a plausible cell.
  A mixture preserves per-cell sparsity and variance, matches the population
  mean, and keeps nnz under the int32 ceiling.
- **Refuted this round:** pooling the residual across similar genes (~$0.35).
- Spend ≈ **$30**.

## Next — where to pick this up

The entry is submitted, so everything below is for after the deadline. Ordered
by value per hour, cheapest first.

### 1. Stop building chimeric cells (hours, no new data)

**A correction to Part 5.** It says the Replogle model's 37% gene coverage was
a structural ceiling. That is wrong, and the evidence was already in our own
table:

| model | gene coverage | `pds_cosine` |
|---|---|---|
| STATE v3 | ~100% | +0.158 |
| **K562 transfer** | **41%** | **+0.168** |
| **Replogle** | **37%** | **-0.079** |

The transfer model has the same coverage as the Replogle model and scores
+0.168 against -0.079, so coverage cannot be the binding constraint. The
difference is **how the output cell is assembled**:

- transfer: take ONE real control cell, multiply some genes by a fold-change,
  leave the rest alone. All 18,533 values come from the same cell, so it stays
  internally coherent.
- `replogle_bridge.lift`: take the model's 6,714 predicted genes and paste the
  other 11,819 from a RANDOMLY CHOSEN, unrelated control cell. Every output
  cell is a chimera of two cells, and `pds_cosine` grades whole-cell profiles.

The lift's docstring claims it does "exactly what the transfer model already
does". It does not.

**Fix:** convert the model's output to per-gene FOLD-CHANGES and apply them to
a single real control cell. For each predicted cell: take control cell `c`,
compute the model's implied ratio per covered gene, multiply `c`'s covered
genes by it, leave the rest of `c` untouched.

This also removes the "wrong dialect" problem on its own, because a ratio
cancels the per-gene offset that `calibrate` was fighting (predicted controls
were 5.3x off per gene). One change, both problems.

Caveat: this is a hypothesis consistent with the numbers, not a measured
result -- the dialect offset may be hurting `pds_cosine` independently. The
fold-change construction addresses both, so it is worth trying first either
way.

Related: the ~11,800 genes Replogle omits are mostly LOWLY EXPRESSED, and the
DE tests apply a 5 CPM eligibility filter first. So effective coverage for the
four DE metrics was always better than "37%" sounds -- consistent with the
Replogle model having had the best DE metrics of anything we built.

### 2. Train on 200 perturbations instead of 150 (one run)

The 2025 challenge is over, so its validation split (50 perturbations) is
public. v3 only ever saw the 150 training perturbations. Train on train+val
and keep the 100 test perturbations as the clean evaluation set: a third more
data, no new infrastructure.

### 3. Attack `expr_mse_unbiased_capped_norm` (the biggest unclaimed score)

**Exactly 0.000 for every model we have ever built**, including the trivial
baseline. One sixth of the score, never once earned a point, all competition.

It is zero because every prediction we make is ultimately resampled control
cells with some genes rescaled, so our overall expression distribution IS the
control distribution -- which is what Arc's baseline already achieves. Any
model that genuinely predicts perturbed expression scores above zero. +0.3 on
this metric alone is +0.05 on the average, about half our current total.

### 4. Blend as a per-cell mixture, not a count average (hours)

Take each output cell wholly from one arm with probability w, rather than
averaging the two arms' counts. Averaged cells carry 7,484 nonzero genes
against ~5,679 in real data -- 1.32x too dense, and each is a smear of two
models rather than a plausible cell. A mixture preserves per-cell sparsity and
variance, matches the population mean, and keeps nnz under the int32 ceiling
that nearly cost us the deadline.

### 5. `reach`, and why nothing we tried ever moved it

Still the weakest metric and immune to magnitude scaling (scale sweep: -0.102
-> -0.105 -> -0.067). It grades predicted VARIANCE, not mean shift -- and our
predictions apply one identical fold-change to all 400 cells of a target, so
cell-to-cell variability in the response is zero by construction. That is
almost certainly the explanation, and it is untested.

### On more data: what would and would not help

Both of these sit in the same GCS bucket we already pull the challenge data
from (`arc-institute-virtual-cell-atlas`, prefixes `scbasecount/`,
`tahoe100M/`, `virtual-cell-challenge/`), so access is trivial. Temper
expectations:

- **Tahoe-100M** is ~100M cells of DRUG / small-molecule perturbations across
  cancer cell lines. The challenge asks about CRISPR gene knockdowns. A drug
  hitting a pathway is not the same experiment as silencing one gene, so it
  will not teach "what does knocking down gene X do".
- **scBaseCount** is large-scale OBSERVATIONAL data -- unperturbed cells, no
  perturbation signal at all.

What both are good for is pretraining a representation of cell state, which is
what Arc's own SE model does and the component we deliberately skipped. That
would plausibly help `pds_cosine` and #3, but it is a large build.

**What we actually lacked is more full-transcriptome GENETIC perturbation
data**, which is scarcer than cell-state data. Worth checking whether
Replogle's raw deposits retain more than the processed 8,248 genes -- that
filter was theirs, not a physical limit.

### If you only have a day

#1 and #2. If you have a week, #3 is where the biggest unclaimed score is.


### Not an option

**X-Cell**: weights never released.
