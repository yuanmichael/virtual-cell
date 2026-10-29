"""Predict with the trained STATE model and score it on the untouched test split.

This is the verdict step. The 100 test-split perturbations were seen by neither
model: STATE trained on the 150 train perturbations (tuned on the 50 val ones),
and the K562 transfer model uses no 2025 data at all. So a score here compares
them fairly, which our earlier +0.108 cannot -- that was measured on the train
split STATE has now learned.

Two conversions are needed between `state tx infer` and `cell_eval2`:

  space   inference clips outputs to [0, 14], i.e. it emits log1p-normalised
          expression, while the real arm is raw counts. We invert with expm1
          and rescale each predicted cell to the median real library size.
  dtype   the scorer (and `vcc prep`) wants whole counts, and plain rounding
          erases sub-unit effects -- the failure that cost our first submission
          -1.72 on direction fidelity. So stochastic rounding again.

Run:
    modal run --detach src/state_infer.py --limit-perts 20   # cheap check
    modal run --detach src/state_infer.py
"""

import modal

from common import (
    REHEARSAL, VOL, VOL_PATH, WORK, app, eval_image, state_image,
)

MODEL_DIR = f"{VOL_PATH}/state/runs/v3/v3"
# Model input must be in the SAME space the model was trained in (log1p CP10k),
# while scoring compares against the raw-count originals.
TEST_IN = f"{VOL_PATH}/state/norm/adata_Test.h5ad"
TEST_H5AD = f"{REHEARSAL}/adata_Test.h5ad"
OUTDIR = f"{WORK}/state_eval"


@app.function(
    image=state_image,
    volumes={VOL_PATH: VOL},
    gpu="A10G",
    cpu=8.0,
    memory=49152,
    timeout=60 * 60 * 4,
)
def predict(limit_perts: int = 0, ckpt: str = "best.ckpt", seed: int = 0,
            calibrate: bool = False, log_space: bool = True):
    """Run STATE on the test split and write a counts-space prediction."""
    import os
    import subprocess
    import time

    import anndata as ad
    import numpy as np
    import scipy.sparse as sp
    import torch

    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()

    ckpt_path = f"{MODEL_DIR}/checkpoints/{ckpt}"
    for p in (ckpt_path, f"{MODEL_DIR}/config.yaml",
              f"{MODEL_DIR}/pert_onehot_map.pt", TEST_H5AD):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    # Confirm the model carries our ESM-2 featurisation and not a regenerated
    # one-hot: at inference, a perturbation absent from this map silently falls
    # back to the control vector (_infer.py:802-804), which would mean "predict
    # no change" and look like a model failure rather than a lookup miss.
    pmap = torch.load(f"{MODEL_DIR}/pert_onehot_map.pt", weights_only=False)
    dims = {tuple(v.shape) for v in pmap.values()}
    n_onehot = sum(1 for v in pmap.values()
                   if float((v != 0).sum()) == 1 and float(v.sum()) == 1.0)
    print(f"model pert map: {len(pmap)} entries, shapes {dims}, "
          f"{n_onehot} look like one-hot rows")
    if n_onehot > len(pmap) // 2:
        raise RuntimeError(
            "pert map looks one-hot -- the ESM-2 features did not reach "
            "training, so inference cannot generalise to unseen genes"
        )

    # Two arms over the SAME cells: the model is fed log1p(CP10k) because that
    # is the space it was trained in, while scoring needs the raw-count
    # original as ground truth. The normalised file is a byte-for-byte copy
    # apart from X, so one boolean mask selects identical cells in both.
    src, real_src = TEST_IN, TEST_H5AD
    if limit_perts:
        # Subset to a few perturbations so the first pass is cheap. Controls
        # must come along: inference samples its basal cells from them.
        Araw = ad.read_h5ad(TEST_H5AD)
        pert = Araw.obs["target_gene"].astype(str)
        keep_perts = sorted(set(pert) - {"non-targeting"})[:limit_perts]
        mask = pert.isin(keep_perts + ["non-targeting"]).to_numpy()
        real_src = f"{OUTDIR}/test_subset.h5ad"
        Araw[mask].copy().write_h5ad(real_src, compression="lzf")
        del Araw

        Anorm = ad.read_h5ad(TEST_IN)
        if len(Anorm) != len(mask):
            raise RuntimeError(
                f"normalised test has {len(Anorm)} cells, raw has {len(mask)} "
                "-- the mask would select different cells on each arm"
            )
        src = f"{OUTDIR}/test_subset_norm.h5ad"
        Anorm[mask].copy().write_h5ad(src, compression="lzf")
        print(f"subset to {len(keep_perts)} perturbations -> "
              f"{int(mask.sum())} cells (raw + normalised)")
        del Anorm
        VOL.commit()

    raw_out = f"{OUTDIR}/state_pred_raw.h5ad"

    # Call run_tx_infer in-process instead of shelling out to `state tx infer`,
    # so we can patch torch.load first. arc-state 0.11.1 predates torch 2.6's
    # switch to `weights_only=True` by default and calls torch.load bare, which
    # now dies with "Unsupported global: numpy._core.multiarray.scalar" on its
    # own checkpoint. These are files we just wrote, so trusting them is fine.
    import argparse

    from state._cli._tx._infer import run_tx_infer

    import torch.serialization as _ts

    # Allowlist the globals weights_only rejects, which is the supported fix
    # and applies no matter who calls torch.load. Patching torch.load alone
    # was not enough: lightning captured a direct reference at import time.
    try:
        import numpy._core.multiarray as _ma
        _ts.add_safe_globals([_ma.scalar, np.dtype, np.ndarray])
    except Exception as e:  # noqa: BLE001
        print(f"add_safe_globals: {type(e).__name__}: {e}")

    _orig_load = _ts.load

    def _load(*a, **kw):
        kw["weights_only"] = False
        return _orig_load(*a, **kw)

    torch.load = _load
    _ts.load = _load

    args = argparse.Namespace(
        checkpoint=ckpt_path, adata=src, output=raw_out, model_dir=MODEL_DIR,
        embed_key=None, pert_col="target_gene", control_pert="non-targeting",
        celltype_col="cell_type", celltypes=None, batch_col="batch",
        seed=seed, max_set_len=None, quiet=False, tsv=False, all_perts=False,
        virtual_cells_per_pert=None, min_cells=None, max_cells=None,
    )
    try:
        run_tx_infer(args)
    finally:
        torch.load = _orig_load
        _ts.load = _orig_load
        VOL.commit()
    print(f"inference done in {time.time()-t0:.0f}s")

    # ---------------- log1p-normalised -> whole counts ----------------
    P = ad.read_h5ad(raw_out)
    R = ad.read_h5ad(real_src)
    print(f"pred {P.shape}  real {R.shape}")

    Xr = R.X.tocsr() if sp.issparse(R.X) else sp.csr_matrix(R.X)
    depth = np.asarray(Xr.sum(axis=1)).ravel()
    target_depth = float(np.median(depth[depth > 0]))
    print(f"real median library size: {target_depth:.0f} UMI")

    Xp = P.X.toarray() if sp.issparse(P.X) else np.asarray(P.X)
    if log_space:
        # The v3 model was trained on log1p(CP10k), so expm1 is the correct
        # inverse. (An earlier run applied expm1 to a COUNT-space model, which
        # exponentiated already-linear values and pinned lfc_nmae at its cap --
        # same call, opposite correctness, because the training space changed.)
        Xp = np.expm1(np.clip(Xp, 0.0, 14.0))
    # Otherwise the model's output is already count-scale, nothing to invert.
    # cell-load applies log1p only when is_log1p=True, and its normalize_counts
    # option is dead config in 0.10.4 (stored and round-tripped, never applied)
    # -- so training targets were raw unnormalised counts, and `state tx infer`
    # adds no transform of its own. An earlier expm1() here, inferred from the
    # [0, 14] clip in their inference code rather than from the dataloader,
    # exponentiated already-linear values: that concentrated the distribution
    # (median per-gene ratio 4.75, 24% of genes clipped) and pinned
    # de_wilcoxon_lfc_nmae at its -6.0 penalty cap.
    Xp = np.clip(Xp, 0.0, None)
    print(f"raw model output: median {np.median(Xp):.3f}  "
          f"p99 {np.quantile(Xp, 0.99):.2f}  max {Xp.max():.2f}  "
          f"median row sum {np.median(Xp.sum(axis=1)):.0f}")
    rs = Xp.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    Xp = Xp / rs * target_depth                     # match real depth

    if calibrate:
        # Per-gene calibration onto real-count space, using the model's own
        # simulated CONTROL cells as the bridge.
        #
        # Submissions contain only perturbed cells, so the scorer derives our
        # fold-changes against REAL controls. Any per-gene offset between the
        # model's output space and real counts therefore lands in every
        # fold-change, which pinned de_wilcoxon_lfc_nmae at its -6.0 penalty
        # cap. Re-anchoring on predicted controls "fixed" that but destroyed
        # fidelity and jaccard, because the anchor absorbed the signal too --
        # and it is not what the server does.
        #
        # So instead: whatever the model does to a CONTROL cell should
        # reproduce a real control. Scale each gene by the ratio of the two
        # control means. That is a per-gene constant, so it cancels the offset
        # while leaving every perturbation-vs-control contrast intact.
        pm = P.obs["target_gene"].astype(str).to_numpy() == "non-targeting"
        rm = R.obs["target_gene"].astype(str).to_numpy() == "non-targeting"
        print(f"calibrating on {pm.sum()} predicted / {rm.sum()} real controls")
        if pm.sum() == 0 or rm.sum() == 0:
            raise RuntimeError("need control cells on both arms to calibrate")

        # Depth-normalise the real controls the same way the predictions were,
        # so the ratio reflects composition rather than library size.
        rd = np.asarray(Xr[rm].sum(axis=1)).ravel()
        real_ctrl = np.asarray(Xr[rm].multiply(
            (target_depth / np.maximum(rd, 1))[:, None]).mean(axis=0)).ravel()
        pred_ctrl = Xp[pm].mean(axis=0)

        ok = (pred_ctrl > 1e-8) & (real_ctrl > 1e-8)
        ratio = np.ones_like(pred_ctrl)
        ratio[ok] = real_ctrl[ok] / pred_ctrl[ok]
        # Clip so a near-zero predicted gene cannot explode into noise.
        ratio = np.clip(ratio, 0.1, 10.0)
        print(f"  gene ratios: median {np.median(ratio[ok]):.3f}  "
              f"p01 {np.quantile(ratio[ok], 0.01):.3f}  "
              f"p99 {np.quantile(ratio[ok], 0.99):.3f}  "
              f"clipped {int((ratio <= 0.1).sum() + (ratio >= 10.0).sum())}")
        Xp = Xp * ratio[None, :]
    rng = np.random.default_rng(seed)
    fl = np.floor(Xp)
    Xp = fl + (rng.random(Xp.shape) < (Xp - fl))    # unbiased rounding
    Xp = sp.csr_matrix(Xp.astype(np.float32))
    Xp.eliminate_zeros()
    assert Xp.data.min() >= 0 and np.all(Xp.data == np.floor(Xp.data))
    print(f"counts-space prediction: {Xp.shape}, {Xp.nnz/1e6:.0f}M nonzeros, "
          f"median depth {np.median(np.asarray(Xp.sum(axis=1)).ravel()):.0f}")

    out = ad.AnnData(
        X=Xp,
        obs=P.obs[["target_gene"]].copy(),
        var=P.var[[]].copy() if P.var.shape[1] else P.var.copy(),
    )
    out.var_names = P.var_names
    out.obs_names = [f"s{i}" for i in range(out.n_obs)]
    pred_path = f"{OUTDIR}/state_pred_counts.h5ad"
    out.write_h5ad(pred_path, compression="lzf")
    VOL.commit()
    print(f"wrote {pred_path} in {time.time()-t0:.0f}s")
    return pred_path


@app.function(
    image=eval_image, volumes={VOL_PATH: VOL},
    cpu=8.0, memory=49152, timeout=60 * 60 * 2,
)
def transfer_predict(real_name: str = "test_subset.h5ad", scale: float = 6.0,
                     lam: float = 0.2, lam_mode: str = "ramp",
                     lam_thr: float = 2.0, seed: int = 0):
    """Build the K562-transfer prediction for the SAME cells STATE predicted.

    Without this, comparing STATE's test-split score to the transfer model's
    +0.108 would be meaningless: that number came from the train split, scored
    against a baseline computed from different perturbations. Same cells, same
    references, same scorer -- or it is not a comparison.

    Uses the shipped submission settings (scale=6.0, lam=0.2 ramp).
    """
    import os

    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    from common import lam_per_target

    real_path = f"{OUTDIR}/{real_name}"
    R = ad.read_h5ad(real_path)
    genes = [str(g) for g in R.var_names]
    gidx = {g: i for i, g in enumerate(genes)}
    pert = R.obs["target_gene"].astype(str).to_numpy()
    targets = sorted(set(pert) - {"non-targeting"})
    print(f"real {R.shape}; {len(targets)} targets to predict")

    axis = np.load(f"{REHEARSAL}/k562_axis.npz", allow_pickle=False)
    k_genes = [str(g) for g in axis["readout"]]
    common_k = axis["common"].astype(np.float64)
    lut_targets = [str(t) for t in axis["targets"]]
    lut = axis["lookup"].astype(np.float64)
    null_med = float(axis["null_med_abs"])

    kkeep = np.array([i for i, g in enumerate(k_genes) if g in gidx])
    kcols = np.array([gidx[k_genes[i]] for i in kkeep])
    lut_ix = {t: i for i, t in enumerate(lut_targets)}
    covered = np.array([t in lut_ix for t in targets])
    mag = np.ones(len(targets))
    for j, t in enumerate(targets):
        if covered[j]:
            mag[j] = np.abs(lut[lut_ix[t]][kkeep]).mean() / null_med
    lams = lam_per_target(lam_mode, lam, lam_thr, mag, covered)
    print(f"readout usable {len(kcols)}; covered {int(covered.sum())}/"
          f"{len(targets)}; lam median {np.median(lams):.3f}")

    cm = common_k[kkeep]
    Xc = R[pert == "non-targeting"].X
    Xc = Xc.tocsr() if sp.issparse(Xc) else sp.csr_matrix(Xc)
    Xc.data = np.rint(Xc.data).astype(np.float32)
    rng = np.random.default_rng(seed)

    blocks, obs_pert = [], []
    for j, t in enumerate(targets):
        n = int((pert == t).sum())
        rows = np.sort(rng.choice(Xc.shape[0], min(n, Xc.shape[0]),
                                  replace=n > Xc.shape[0]))
        spec = (lut[lut_ix[t]][kkeep] - cm) if covered[j] else 0.0
        fac = np.ones(len(genes))
        fac[kcols] = np.power(2.0, scale * (cm + lams[j] * spec))
        B = sp.csr_matrix(Xc[rows], dtype=np.float32) @ sp.diags(fac)
        d = np.asarray(B.data, dtype=np.float64)
        fl = np.floor(d)
        B.data = (fl + (rng.random(d.shape) < (d - fl))).astype(np.float32)
        B.eliminate_zeros()
        blocks.append(sp.csr_matrix(B, dtype=np.float32))
        obs_pert += [t] * B.shape[0]

    # The scorer requires both arms to carry the same perturbation set, so the
    # prediction needs a non-targeting block too. Predicting "no perturbation"
    # means handing back the control cells unchanged.
    n_ctrl = int((pert == "non-targeting").sum())
    blocks.append(sp.csr_matrix(Xc[:n_ctrl], dtype=np.float32))
    obs_pert += ["non-targeting"] * n_ctrl

    X = sp.vstack(blocks, format="csr")
    assert X.data.min() >= 0 and np.all(X.data == np.floor(X.data))
    out = ad.AnnData(X=X, obs={"target_gene": np.array(obs_pert)})
    out.var_names = genes
    out.obs_names = [f"t{i}" for i in range(out.n_obs)]
    path = f"{OUTDIR}/transfer_pred_counts.h5ad"
    out.write_h5ad(path, compression="lzf")
    VOL.commit()
    print(f"wrote {path}: {out.shape}, {X.nnz/1e6:.0f}M nonzeros")
    return path


@app.function(
    image=eval_image,
    volumes={VOL_PATH: VOL},
    cpu=8.0,
    memory=49152,
    timeout=60 * 60 * 4,
)
def score(pred_name: str = "state_pred_counts.h5ad",
          real_name: str = "test_subset.h5ad", tag: str = "state",
          control_source: str = "real"):
    """Score a counts-space prediction against the real test cells.

    The references (generic baseline, lfc_nmae reference) depend only on the
    real arm, so they are cached per real file and shared between models --
    which is also what makes the STATE-vs-transfer comparison apples to apples.
    """
    import json
    import os

    import anndata as ad
    import cell_eval2 as ce

    pred_path = f"{OUTDIR}/{pred_name}"
    real_path = f"{OUTDIR}/{real_name}"
    for p in (pred_path, real_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    pred = ad.read_h5ad(pred_path)
    real = ad.read_h5ad(real_path)
    print(f"pred {pred.shape}  real {real.shape}")

    cfg = ce.EvalConfig(
        metrics="vcc2026", pert_col="target_gene", control="non-targeting",
        # control_source="pred" anchors predicted fold-changes on the
        # prediction's OWN control cells. With "real", any systematic offset
        # between the model's output space and real counts lands in every
        # gene's log-fold-change and pins de_wilcoxon_lfc_nmae at its penalty
        # cap -- which is exactly what happened on the first STATE scoring.
        control_source=control_source, allow_fractional_counts=False,
        num_threads=8, outdir=OUTDIR,
    )

    stem = real_name.replace(".h5ad", "") + ("" if control_source == "real" else f"_cs{control_source}")
    de_real_path = f"{OUTDIR}/de_real_{stem}.parquet"
    base_path = f"{OUTDIR}/base_{stem}.parquet"
    nmae_path = f"{OUTDIR}/nmae_{stem}.parquet"

    de_real = de_real_path if os.path.exists(de_real_path) else None
    if de_real:
        print(f"reusing cached real DE: {de_real}")

    results = ce.compute_metrics(pred, real, config=cfg, de_real=de_real,
                                 write_de=de_real is None)
    uw = ce.aggregate_metrics_wide(results)
    print("RAW metrics:")
    print(uw)
    if de_real is None and os.path.exists(f"{OUTDIR}/de_real.parquet"):
        os.replace(f"{OUTDIR}/de_real.parquet", de_real_path)
        de_real = de_real_path

    import polars as pl
    if os.path.exists(base_path) and os.path.exists(nmae_path):
        bw, nmae = pl.read_parquet(base_path), pl.read_parquet(nmae_path)
        print("reusing cached references")
    else:
        bw = ce.build_generic_baseline(real, config=cfg, de_real=de_real).agg
        _, nmae = ce.compute_lfc_nmae_reference(real, config=cfg,
                                                de_real=de_real)
        bw.write_parquet(base_path)
        nmae.write_parquet(nmae_path)
    print("baseline:")
    print(bw)

    scored = ce.score_metrics(uw, bw, lfc_nmae_ref=nmae)
    print("\nSCALED (0 = generic-response oracle, 1 = replicate):")
    print(scored)
    rows = scored.to_dicts()
    avg = [r for r in rows if r["metric"] == "avg_score"]
    if avg:
        print(f"\n>>> {tag} avg_score = {avg[0]['from_baseline']:+.4f}")

    with open(f"{OUTDIR}/score_{tag}_{stem}.json", "w") as fh:
        json.dump({"tag": tag, "real": real_name, "pred": pred_name,
                   "raw": uw.to_dicts(), "scaled": rows}, fh,
                  indent=2, default=str)
    VOL.commit()
    return rows


@app.local_entrypoint()
def main(limit_perts: int = 0, ckpt: str = "best.ckpt", seed: int = 0,
         calibrate: bool = False, log_space: bool = True,
         block: bool = False):
    if block:
        predict.remote(limit_perts, ckpt, seed, calibrate, log_space)
        return
    call = predict.spawn(limit_perts, ckpt, seed, calibrate, log_space)
    print(f"spawned: {call.object_id}")
