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

Spend to date ≈ **$9.00** (transfer model ~$1.4, STATE route ~$7.6).

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

## Next

Ranked by expected payoff. The binding constraint is **150 training
perturbations** — v3's val loss flattened at step 3,199 (0.544) and never
improved, so the model has extracted what is there. More steps and bigger
architectures will not help; more perturbations will.

1. **Ensemble STATE + transfer.** The per-metric table shows they fail
   differently — STATE wins `pds_cosine` 5×, transfer wins all four DE metrics.
   Complementary, not redundant. Cheapest real win, reuses everything.
2. **`cell_set_len` 64 → 512** (Arc's default). 8× more cells per attention
   set, which is the mechanism behind STATE's `pds_cosine` edge.
3. **Train on Replogle single-cell.** 9,866 perturbations across K562 *and*
   RPE1 versus our 150 in one cell type: 65× the perturbation diversity for the
   ESM-2 encoder, and the first genuine cross-cell-line examples — which is
   literally the task. The only change that lifts the data ceiling rather than
   working around it.
4. **Attack `reach`.** The one metric both models fail, and immune to magnitude
   scaling (confirmed by the scale sweep: −0.102 → −0.105 → −0.067). It is
   about predicted *variance*, not mean shift. Untried.

### Not an option

**X-Cell**: weights never released.
