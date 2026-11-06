"""Score the K562-transfer predictor on Arc's 2025 H1 data, under Arc's rules.

The final phase returns no scores, so this is our only feedback loop. Arc's
2025 release is real single-cell perturbation data with ground truth, on the
same assay as D/E/F, and our predictor uses NO H1 perturbation data -- so
every H1 perturbation is held out by construction. That makes this a genuine
zero-shot rehearsal of the real task.

Crucially it computes Arc's own comparators locally:

  baseline b  `build_generic_baseline` -- the average real response emitted for
              every target. An ORACLE: it reads ground truth we never get.
              s = (u-b)/(r-b), so this strategy scores 0 BY CONSTRUCTION.
  ceiling r   `compute_ceiling` -- a split-half replicate of the real data.

So the question this answers is not "does our prediction correlate with the
truth" but "does it beat knowing the average response", which is the only
thing the leaderboard pays for.

The model:   lfc[target] = scale * (common + lam[target] * (lookup - common))
`common` is the K562 mean response over all 9,866 perturbations; `lookup` is
that target's own K562 response. lam is per-target, because a target whose
K562 measurement is noisy is better served by the generic response (worth ~0)
than by its own noise (worth < 0).

Run:
    modal run --detach src/rehearsal_score.py --prepare-only   # cache real DE
    modal run --detach src/rehearsal_score.py --lam 0.15 --scale 4.0
"""

import modal

from common import (
    REHEARSAL, VOL, VOL_PATH, WORK, app, eval_image, lam_per_target,
)

REAL_CELLS_PER_PERT = 400    # mirrors what Arc asks us to predict for D/E/F
N_CONTROL = 18400            # mirrors the control cells each 2026 context ships
AXIS = "k562_axis.npz"


def _run(
    lam: float = 0.15,
    scale: float = 4.0,
    lam_mode: str = "ramp",
    lam_thr: float = 2.0,
    denoise: str = "",
    seed: int = 0,
    prepare_only: bool = False,
    refs: bool = True,
    ceiling: bool = False,
    integer_counts: bool = True,
    n_perts: int = 0,
    cells_per_pert: int = REAL_CELLS_PER_PERT,
    n_control: int = N_CONTROL,
    tag: str = "",
):
    import json
    import os
    import time

    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    import cell_eval2 as ce

    t0 = time.time()
    # A subsampled rehearsal is a different ground truth, so it gets its own
    # cache directory -- mixing its DE with the full one would be meaningless.
    full = not n_perts and cells_per_pert == REAL_CELLS_PER_PERT \
        and n_control == N_CONTROL
    outdir = f"{WORK}/rehearsal" if full else \
        f"{WORK}/rehearsal_p{n_perts}c{cells_per_pert}n{n_control}"
    os.makedirs(outdir, exist_ok=True)
    print(f"{'full' if full else 'SUBSAMPLED'} rehearsal -> {outdir}")
    real_path = f"{outdir}/real.h5ad"
    de_real_path = f"{outdir}/de_real.parquet"

    # ---------------- axis + lookups ----------------
    axis = np.load(f"{REHEARSAL}/{AXIS}", allow_pickle=False)
    k_genes = [str(g) for g in axis["readout"]]
    common_k = axis["common"].astype(np.float64)
    lut_targets = [str(t) for t in axis["targets"]]
    # `lookup` is each target's own K562 response; `pool{k}` replaces its
    # residual with the mean over its k nearest neighbours in residual space,
    # which cancels independent measurement noise (src/denoise_probe.py). The
    # effect-size gate below deliberately still reads the UNPOOLED lookup, so
    # `lam` is driven by how well the target itself was measured either way.
    lut = axis["lookup"].astype(np.float64)
    lut_blend = lut
    if denoise:
        if denoise not in axis.files:
            raise KeyError(f"{denoise!r} not in {AXIS}: {sorted(axis.files)}")
        lut_blend = axis[denoise].astype(np.float64)
        print(f"denoise: blending with {denoise}, "
              f"residual magnitude {np.abs(lut_blend - common_k).mean() / np.abs(lut - common_k).mean():.2f}x unpooled")
    null_med = float(axis["null_med_abs"])
    print(f"axis: {len(k_genes)} K562 readout genes, {len(lut_targets)} lookups, "
          f"null band {null_med:.4f}")

    # ---------------- real H1 data ----------------
    src = f"{REHEARSAL}/adata_Training.h5ad"
    A = ad.read_h5ad(src, backed="r")
    obs = A.obs
    pert = obs["target_gene"].astype(str).to_numpy()
    print(f"source: {A.shape[0]} cells x {A.shape[1]} genes")

    # Two independent streams, so that reusing a cached real arm cannot shift
    # which cells the prediction draws: with one stream, skipping the real
    # build would leave the rng in a different state and silently change the
    # prediction. The control split must stay deterministic either way, since
    # the cached real.h5ad was built against this exact partition.
    rng_split = np.random.default_rng(seed)
    rng = np.random.default_rng(seed + 1)

    ctrl_rows = np.where(pert == "non-targeting")[0]
    rng_split.shuffle(ctrl_rows)
    half = min(n_control, len(ctrl_rows) // 2)
    ctrl_real, ctrl_pred = ctrl_rows[:half], ctrl_rows[half:half * 2]
    print(f"control cells: {len(ctrl_rows)} total -> {half} real / {half} pred")

    targets = sorted(set(pert) - {"non-targeting"})
    if n_perts and n_perts < len(targets):
        # Plain random sample, not a hand-picked one: the point of a rehearsal
        # is that its difficulty resembles the real panel's, so keep the
        # covered/uncovered and strong/weak mix as it falls.
        keep = rng_split.choice(len(targets), n_perts, replace=False)
        targets = [targets[i] for i in sorted(keep)]
        print(f"subsampled to {len(targets)} perturbations")
    var_names = np.array([str(v) for v in A.var_names])

    if os.path.exists(real_path):
        # Modal preempts containers, and rebuilding the real arm from the
        # 15 GB backed file is the slowest startup step. Reuse it -- and it is
        # also what `de_real.parquet` was computed against, so reusing keeps
        # the two consistent.
        A.file.close()
        real = ad.read_h5ad(real_path)
        print(f"reused cached real arm: {real.shape} "
              f"({time.time()-t0:.0f}s)")
    else:
        sel_rows, sel_pert = [], []
        for t in targets:
            r = np.where(pert == t)[0]
            if len(r) > cells_per_pert:
                r = rng_split.choice(r, cells_per_pert, replace=False)
            sel_rows.append(r)
            sel_pert += [t] * len(r)
        real_rows = np.concatenate(sel_rows + [ctrl_real])
        real_pert = np.array(sel_pert + ["non-targeting"] * len(ctrl_real))
        print(f"real arm: {len(targets)} perturbations, {len(real_rows)} cells")

        # Backed CSR row selection wants sorted, unique indices.
        order = np.argsort(real_rows, kind="stable")
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        Xr = A.X[real_rows[order]]
        Xr = sp.csr_matrix(Xr)[inv]
        A.file.close()

        real = ad.AnnData(
            X=Xr,
            obs={"target_gene": real_pert},
            var={"gene": var_names},
        )
        real.var_names = var_names
        real.obs_names = [f"r{i}" for i in range(real.n_obs)]
        print(f"real built in {time.time()-t0:.0f}s, "
              f"{real.X.nnz/1e6:.0f}M nonzeros")
        # Checkpoint before the prediction side, so a preemption during the
        # expensive part does not cost us this again.
        real.write_h5ad(real_path, compression="lzf")
        VOL.commit()
        print(f"wrote {real_path}")

    # ---------------- control pool for the prediction ----------------
    A2 = ad.read_h5ad(src, backed="r")
    cp = np.sort(ctrl_pred)
    Xc = sp.csr_matrix(A2.X[cp])
    A2.file.close()
    print(f"pred control pool: {Xc.shape}")

    # ---------------- the predicted log-fold-changes ----------------
    gi = {g: i for i, g in enumerate(var_names)}
    kcols = np.array([gi[g] for g in k_genes if g in gi])
    kkeep = np.array([i for i, g in enumerate(k_genes) if g in gi])
    print(f"K562 readout genes present in the 2025 space: {len(kcols)}")

    lut_ix = {t: i for i, t in enumerate(lut_targets)}
    covered = np.array([t in lut_ix for t in targets])
    mag = np.ones(len(targets))
    for j, t in enumerate(targets):
        if covered[j]:
            mag[j] = np.abs(lut[lut_ix[t]][kkeep]).mean() / null_med
    lams = lam_per_target(lam_mode, lam, lam_thr, mag, covered)
    print(f"covered {covered.sum()}/{len(targets)}; "
          f"effect size median {np.median(mag):.2f}x; "
          f"lam mode={lam_mode} median {np.median(lams):.3f} "
          f"max {lams.max():.3f} nonzero {int((lams>0).sum())}")

    cm = common_k[kkeep]
    lfc = np.zeros((len(targets), len(kcols)))
    for j, t in enumerate(targets):
        spec = (lut_blend[lut_ix[t]][kkeep] - cm) if covered[j] else 0.0
        lfc[j] = scale * (cm + lams[j] * spec)
    print(f"predicted |log2FC|: mean {np.abs(lfc).mean():.4f}  "
          f"p99 {np.quantile(np.abs(lfc), 0.99):.3f}  "
          f"max {np.abs(lfc).max():.3f}")

    # ---------------- emit cells ----------------
    # Multiply each gene's counts by 2^lfc in the cells we hand back. Only the
    # K562 readout columns move; the rest keep their control values, which is
    # exactly what we would do for D/E/F.
    blocks, pred_pert = [], []
    for j, t in enumerate(targets):
        take = np.sort(rng.choice(Xc.shape[0], cells_per_pert, replace=False))
        # One column rescale per target: genes outside the K562 readout keep
        # their control value (factor 1), which is what we would do for D/E/F.
        fac = np.ones(len(var_names))
        fac[kcols] = np.power(2.0, lfc[j])
        B = sp.csr_matrix(Xc[take], dtype=np.float32) @ sp.diags(fac)
        if integer_counts:
            # Submissions must be whole counts (vcc prep `_require_counts`,
            # mirroring cell_eval2's allow_fractional_counts=False). Round
            # STOCHASTICALLY: plain rounding would send a count of 1 scaled by
            # 1.05 back to 1, erasing every small effect and reproducing the
            # "calls nothing" failure. This is unbiased in expectation, so the
            # mean shift over 400 cells survives.
            d = np.asarray(B.data, dtype=np.float64)
            fl = np.floor(d)
            B.data = (fl + (rng.random(d.shape) < (d - fl))).astype(np.float32)
            B.eliminate_zeros()
        blocks.append(sp.csr_matrix(B, dtype=np.float32))
        pred_pert += [t] * cells_per_pert
        if (j + 1) % 25 == 0:
            print(f"  emitted {j+1}/{len(targets)} ({time.time()-t0:.0f}s)")

    ctrl_block = sp.csr_matrix(Xc[: len(ctrl_real)], dtype=np.float32)
    Xp = sp.vstack(blocks + [ctrl_block], format="csr")
    pred_pert = np.array(pred_pert + ["non-targeting"] * ctrl_block.shape[0])
    pred = ad.AnnData(X=Xp, obs={"target_gene": pred_pert},
                      var={"gene": var_names})
    pred.var_names = var_names
    pred.obs_names = [f"p{i}" for i in range(pred.n_obs)]
    print(f"pred built: {pred.shape}, {Xp.nnz/1e6:.0f}M nonzeros "
          f"({time.time()-t0:.0f}s)")

    # ---------------- score under Arc's rules ----------------
    cfg = ce.EvalConfig(
        metrics="vcc2026",
        pert_col="target_gene",
        control="non-targeting",
        control_source="real",
        allow_fractional_counts=not integer_counts,
        num_threads=8,
        outdir=outdir,
    )

    # Which DE backend we got decides whether this is a 20-minute run or a
    # 90-minute one, and the engines do not agree numerically -- so a cached
    # de_real from a different engine must not be mixed in. Say which one up
    # front rather than inferring it from the wall clock afterwards.
    try:
        import pdex  # noqa: F401
        print("DE backend: pdex available (fast numba CPU path)")
    except ImportError:
        print("DE backend: pdex MISSING -- scanpy fallback, expect ~5x slower")

    de_real = de_real_path if os.path.exists(de_real_path) else None
    if de_real:
        print(f"reusing cached real DE: {de_real}")

    if prepare_only:
        # Compute and cache the expensive real-side DE once, so each later
        # (lam, scale) run only pays for the prediction side.
        if de_real:
            print("nothing to prepare: real arm and real DE are both cached")
            return None
        res = ce.compute_metrics(pred, real, config=cfg, write_de=True)
        print(res)
        VOL.commit()
        print(f"prepare-only done in {time.time()-t0:.0f}s; "
              f"outdir now holds: {sorted(os.listdir(outdir))}")
        return None

    print("\ncomputing user metrics ...")
    results_user = ce.compute_metrics(pred, real, config=cfg, de_real=de_real)
    uw = ce.aggregate_metrics_wide(results_user)
    print("RAW user metrics:")
    print(uw)

    out = {"lam": lam, "scale": scale, "lam_mode": lam_mode,
           "lam_thr": lam_thr, "denoise": denoise, "tag": tag, "full": full,
           "n_perts": len(targets), "cells_per_pert": cells_per_pert,
           "n_control": half, "integer_counts": integer_counts,
           "raw_user": uw.to_dicts()}

    if refs:
        import polars as pl

        # Both references are functions of the GROUND TRUTH alone -- neither
        # looks at our prediction -- so they are identical for every (scale,
        # lam) and get computed once per rehearsal size. This is what makes a
        # sweep point cost one DE pass instead of three.
        base_path = f"{outdir}/base_agg.parquet"
        nmae_path = f"{outdir}/nmae_ref.parquet"
        if os.path.exists(base_path) and os.path.exists(nmae_path):
            bw = pl.read_parquet(base_path)
            nmae_agg = pl.read_parquet(nmae_path)
            print("reusing cached baseline + nmae reference "
                  "(neither depends on scale/lam)")
        else:
            print("\nbuilding Arc's generic-response baseline (the 0 point) ...")
            bw = ce.build_generic_baseline(real, config=cfg, de_real=de_real).agg
            print(bw)

            print("\nestimating the lfc_nmae reference ...")
            _, nmae_agg = ce.compute_lfc_nmae_reference(
                real, config=cfg, de_real=de_real)
            print(nmae_agg)

            bw.write_parquet(base_path)
            nmae_agg.write_parquet(nmae_path)
            VOL.commit()

        if ceiling:
            # Five split-half DE passes, and it does not feed score_metrics --
            # it only tells us how much headroom the experiment's own noise
            # leaves. Off by default.
            print("\nestimating the replicate ceiling (the 1 point) ...")
            _, ceil_agg = ce.compute_ceiling(real, config=cfg)
            print(ceil_agg)
            out["ceiling"] = ceil_agg.to_dicts()

        print("\nSCALED scores (0 = generic response, 1 = replicate):")
        scored = ce.score_metrics(uw, bw, lfc_nmae_ref=nmae_agg)
        print(scored)
        out["scaled"] = scored.to_dicts()
        out["raw_base"] = bw.to_dicts()
        avg = [r for r in scored.to_dicts() if r["metric"] == "avg_score"]
        if avg:
            print(f"\n>>> scale={scale} lam={lam} {lam_mode} "
                  f"avg_score={avg[0]['from_baseline']:+.4f}")

    name = tag or f"{lam_mode}_lam{lam}_scale{scale}"
    with open(f"{outdir}/score_{name}.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    VOL.commit()
    print(f"\ntotal {time.time()-t0:.0f}s -> {outdir}/score_{name}.json")
    return out


# Memory is half the per-second cost on Modal, so the subsampled probe gets a
# container sized for it rather than for the full 74k-cell arm: $0.32/hr
# against $0.76/hr. Same body, two shapes -- `--small` picks the cheap one.
run = app.function(
    image=eval_image, volumes={VOL_PATH: VOL},
    cpu=8.0, memory=49152, timeout=60 * 60 * 6,
)(_run)

run_small = app.function(
    image=eval_image, volumes={VOL_PATH: VOL},
    cpu=4.0, memory=16384, timeout=60 * 60 * 2,
    name="run_small",
)(_run)


def _sweep(points, **kw):
    """Run several (denoise, lam, scale) points inside ONE container.

    Each point reuses the cached ground-truth DE and reference aggregates, so
    the marginal cost of a point is one DE pass -- but a separate container per
    point would pay cold start plus a read of the 15 GB backed source every
    time, which dominated the actual spend. Looping here pays that once.
    """
    out = {}
    for denoise, lam, scale in points:
        tag = f"{denoise or 'raw'}_lam{lam:g}_scale{scale:g}"
        print(f"\n{'='*70}\n{tag}\n{'='*70}", flush=True)
        try:
            r = _run(lam=lam, scale=scale, denoise=denoise, tag=tag, **kw)
        except Exception as e:  # noqa: BLE001 - one bad point must not end the sweep
            print(f"!! {tag} failed: {type(e).__name__}: {e}")
            continue
        if r and r.get("scaled"):
            avg = [x for x in r["scaled"] if x["metric"] == "avg_score"]
            if avg:
                out[tag] = avg[0]["from_baseline"]
                # Keep the per-metric breakdown: the avg can stay flat while
                # fidelity and lfc_nmae move in opposite directions, which is
                # what the scale sweep showed and what we need to see here.
                out[tag + "__metrics"] = {
                    x["metric"]: x["from_baseline"] for x in r["scaled"]
                }
    print(f"\n{'='*70}\nSWEEP RESULTS\n{'='*70}")
    scores = {k: v for k, v in out.items() if not k.endswith("__metrics")}
    for k in sorted(scores, key=lambda k: -scores[k]):
        print(f"  {k:<32s} avg_score {scores[k]:+.4f}")
    return out


sweep = app.function(
    image=eval_image, volumes={VOL_PATH: VOL},
    cpu=4.0, memory=16384, timeout=60 * 60 * 5,
    name="sweep",
)(_sweep)


@app.local_entrypoint()
def main(lam: float = 0.15, scale: float = 4.0, scales: str = "",
         lams: str = "", lam_mode: str = "ramp", lam_thr: float = 2.0,
         denoise: str = "", seed: int = 0,
         prepare_only: bool = False, refs: bool = True, ceiling: bool = False,
         integer_counts: bool = True, n_perts: int = 0,
         cells_per_pert: int = REAL_CELLS_PER_PERT,
         n_control: int = N_CONTROL, small: bool = False, tag: str = "",
         denoises: str = ""):
    # `--denoises raw,pool10,pool30 --lams 0.2,0.5,1.0` sweeps the denoise
    # variant and lam together in ONE container. "raw" means the unpooled
    # lookup, i.e. the model as shipped.
    if denoises:
        pts = [
            ("" if d.strip() in ("raw", "") else d.strip(), lm, s)
            for d in denoises.split(",") if d.strip()
            for lm in ([float(x) for x in lams.split(",") if x.strip()] or [lam])
            for s in ([float(x) for x in scales.split(",") if x.strip()] or [scale])
        ]
        print(f"sweeping {len(pts)} points in one container")
        res = sweep.remote(
            pts, lam_mode=lam_mode, lam_thr=lam_thr, seed=seed, refs=refs,
            ceiling=ceiling, integer_counts=integer_counts, n_perts=n_perts,
            cells_per_pert=cells_per_pert, n_control=n_control,
        )
        scores = {k: v for k, v in (res or {}).items()
                  if not k.endswith("__metrics")}
        if scores:
            best = max(scores, key=scores.get)
            print(f"\nbest: {best} at {scores[best]:+.4f}")
            for m, v in sorted((res.get(best + "__metrics") or {}).items()):
                print(f"    {m:<42s} {v:+.4f}")
        return

    fn = run_small if small else run
    # `--scales 6,9,13,18` or `--lams 0.2,0.5,1.0` sweeps one dial. Each point
    # reuses the cached ground-truth DE and reference aggregates, so it costs
    # one DE pass. Pooling shrinks the residual (pool30 to 0.68x), so a pooled
    # run needs its own lam sweep rather than inheriting the shipped 0.2.
    s_todo = [float(s) for s in scales.split(",") if s.strip()] or [scale]
    l_todo = [float(x) for x in lams.split(",") if x.strip()] or [lam]
    results = {}
    for s in s_todo:
        for lm in l_todo:
            auto = f"{denoise or 'raw'}_lam{lm:g}_scale{s:g}"
            r = fn.remote(lm, s, lam_mode, lam_thr, denoise, seed,
                          prepare_only, refs, ceiling, integer_counts,
                          n_perts, cells_per_pert, n_control,
                          tag or (auto if len(s_todo) * len(l_todo) > 1 else ""))
            if r and r.get("scaled"):
                avg = [x for x in r["scaled"] if x["metric"] == "avg_score"]
                if avg:
                    results[(s, lm)] = avg[0]["from_baseline"]
    if len(results) > 1:
        print(f"\n=== sweep ({denoise or 'raw'}) ===")
        for k in sorted(results):
            print(f"  scale {k[0]:>5g}  lam {k[1]:>5g}  "
                  f"avg_score {results[k]:+.4f}")
        best = max(results, key=results.get)
        print(f"best: scale={best[0]:g} lam={best[1]:g} at {results[best]:+.4f}")
