"""Build the real D/E/F submission from the K562 transfer model.

This is the production counterpart of `rehearsal_score.py`: the same model,
the same dials, applied to the final panel's 300 targets across contexts
D, E and F. The rehearsal exists to fit `scale` and `lam`; this script ships
whatever those come out as.

    lfc[target] = scale * (common + lam[target] * (lookup[target] - common))

`common` is the mean response over all 9,866 K562 perturbations; `lookup` is
that target's own K562 response; `lam` ramps with how far the target's K562
measurement clears the non-targeting noise band, and is 0 for the 23 targets
the screen does not cover.

Prediction cells are real control cells from the context being predicted, with
the readout genes rescaled by 2^lfc and rounded stochastically back to whole
counts -- `vcc prep` rejects fractional values, and plain rounding would erase
every sub-unit shift, which is exactly how the first submission scored -1.72
on direction fidelity.

Run:
    modal run --detach src/predict.py --scale 6.0 --lam 0.15
    modal run --detach src/predict.py --skip-build          # repackage only
"""

import modal

from common import (
    CONTROLS_FINAL,
    CPU_MEM_MIB,
    REHEARSAL,
    VOL,
    VOL_PATH,
    WORK,
    app,
    cpu_image,
    lam_per_target,
)

CELLS_PER_PERT = 400
SEED = 0
AXIS = "k562_axis.npz"
OUT_H5AD = f"{WORK}/pred_transfer.h5ad"


@app.function(
    image=cpu_image,
    volumes={VOL_PATH: VOL},
    cpu=8.0,
    memory=CPU_MEM_MIB,
    timeout=60 * 60 * 4,
)
def build(
    scale: float = 4.0,
    lam: float = 0.15,
    lam_mode: str = "ramp",
    lam_thr: float = 2.0,
    contexts: str = "D,E,F",
):
    import json
    import os
    import time

    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    os.makedirs(WORK, exist_ok=True)
    ctxs = [c.strip() for c in contexts.split(",") if c.strip()]

    manifest = json.load(open(f"{CONTROLS_FINAL}/manifest.json"))
    genes = pd.read_csv(f"{CONTROLS_FINAL}/gene_names.csv")["gene_name"].tolist()
    targets = pd.read_csv(f"{CONTROLS_FINAL}/pert_counts.csv")["target_gene"].tolist()
    gidx = {g: i for i, g in enumerate(genes)}
    assert len(genes) == manifest["n_genes"], "gene list disagrees with manifest"
    assert manifest["cells_per_pert"] == CELLS_PER_PERT
    print(f"panel {manifest['panel_id']}: {len(genes)} genes, "
          f"{len(targets)} targets, contexts {ctxs}")

    # ---------------- the K562 axis ----------------
    axis = np.load(f"{REHEARSAL}/{AXIS}", allow_pickle=False)
    k_genes = [str(g) for g in axis["readout"]]
    common_k = axis["common"].astype(np.float64)
    lut_targets = [str(t) for t in axis["targets"]]
    lut = axis["lookup"].astype(np.float64)
    null_med = float(axis["null_med_abs"])

    # Readout genes that this panel actually scores, and their columns in it.
    kkeep = np.array([i for i, g in enumerate(k_genes) if g in gidx])
    kcols = np.array([gidx[k_genes[i]] for i in kkeep])
    print(f"K562 readout usable here: {len(kcols)} of {len(genes)} panel genes")

    lut_ix = {t: i for i, t in enumerate(lut_targets)}
    covered = np.array([t in lut_ix for t in targets])
    mag = np.ones(len(targets))
    for j, t in enumerate(targets):
        if covered[j]:
            mag[j] = np.abs(lut[lut_ix[t]][kkeep]).mean() / null_med
    lams = lam_per_target(lam_mode, lam, lam_thr, mag, covered)
    print(f"covered {int(covered.sum())}/{len(targets)}; "
          f"effect size median {np.median(mag):.2f}x null; "
          f"lam {lam_mode} median {np.median(lams):.3f} max {lams.max():.3f} "
          f"nonzero {int((lams > 0).sum())}")

    cm = common_k[kkeep]
    lfc = np.zeros((len(targets), len(kcols)))
    for j, t in enumerate(targets):
        spec = (lut[lut_ix[t]][kkeep] - cm) if covered[j] else 0.0
        lfc[j] = scale * (cm + lams[j] * spec)
    fac = np.ones((len(targets), len(genes)))
    fac[:, kcols] = np.power(2.0, lfc)
    print(f"predicted |log2FC|: mean {np.abs(lfc).mean():.4f}  "
          f"p99 {np.quantile(np.abs(lfc), 0.99):.3f}  "
          f"max {np.abs(lfc).max():.3f}")

    # ---------------- emit cells, one context at a time ----------------
    rng = np.random.default_rng(SEED)
    paths = []
    for ctx in ctxs:
        t0 = time.time()
        path = f"{WORK}/pred_ctx_{ctx}.h5ad"
        if os.path.exists(path):
            print(f"[{ctx}] reusing checkpoint "
                  f"({os.path.getsize(path) / 1e9:.2f} GB)")
            paths.append(path)
            continue

        src = ad.read_h5ad(f"{CONTROLS_FINAL}/context_{ctx}.h5ad")
        assert list(src.var_names) == genes, f"gene order mismatch in {ctx}"
        Xc = src.X.tocsr()
        Xc.data = np.rint(Xc.data).astype(np.float32)
        n_ctrl = Xc.shape[0]
        print(f"[{ctx}] {n_ctrl} control cells, {Xc.nnz:,} nonzeros")

        blocks, obs_pert = [], []
        for j, tgt in enumerate(targets):
            rows = np.sort(rng.choice(n_ctrl, CELLS_PER_PERT, replace=False))
            # One vectorised column rescale per target. Genes outside the K562
            # readout keep their control value (factor 1.0).
            B = sp.csr_matrix(Xc[rows], dtype=np.float32) @ sp.diags(fac[j])
            # Stochastic rounding: floor + Bernoulli(frac). Unbiased in
            # expectation, so a 5% shift on a count of 1 survives as a 5%
            # chance of a 2 across the 400 cells instead of vanishing.
            d = np.asarray(B.data, dtype=np.float64)
            fl = np.floor(d)
            B.data = (fl + (rng.random(d.shape) < (d - fl))).astype(np.float32)
            B.eliminate_zeros()
            blocks.append(sp.csr_matrix(B, dtype=np.float32))
            obs_pert.extend([tgt] * CELLS_PER_PERT)
            if (j + 1) % 50 == 0:
                print(f"  [{ctx}] {j+1}/{len(targets)} ({time.time()-t0:.0f}s)")

        X = sp.vstack(blocks, format="csr")
        del blocks
        assert X.data.min() >= 0, "negative counts would be rejected by prep"
        assert np.all(X.data == np.floor(X.data)), "fractional counts survived"
        obs = pd.DataFrame(
            {
                "target_gene": pd.Categorical(obs_pert),
                "context": pd.Categorical([ctx] * X.shape[0]),
            },
            index=[f"{ctx}_cell_{i}" for i in range(X.shape[0])],
        )
        out = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
        out.write_h5ad(path, compression="lzf")
        del out, X, Xc, src
        paths.append(path)
        print(f"[{ctx}] wrote {path} in {time.time()-t0:.0f}s "
              f"({os.path.getsize(path) / 1e9:.2f} GB)")
        VOL.commit()

    if len(paths) == 1:
        os.replace(paths[0], OUT_H5AD)
    else:
        from anndata.experimental import concat_on_disk

        t0 = time.time()
        concat_on_disk(paths, OUT_H5AD, label=None)
        print(f"concat_on_disk -> {OUT_H5AD} in {time.time()-t0:.0f}s "
              f"({os.path.getsize(OUT_H5AD) / 1e9:.2f} GB)")
        for p in paths:
            os.remove(p)
    VOL.commit()
    return OUT_H5AD


@app.function(
    image=cpu_image,
    volumes={VOL_PATH: VOL},
    cpu=8.0,
    memory=CPU_MEM_MIB,
    timeout=60 * 60 * 3,
)
def prep(h5ad_path: str = OUT_H5AD, out_name: str = "transfer.vcc",
         dry_run: bool = False):
    """Package with `vcc prep` on the Volume, where the temp files fit."""
    import os
    import subprocess

    out = f"{WORK}/{out_name}"
    cmd = [
        "vcc", "prep", h5ad_path,
        "-g", f"{CONTROLS_FINAL}/gene_names.csv",
        "--perts", f"{CONTROLS_FINAL}/pert_counts.csv",
        "-o", out,
        "--pert-col", "target_gene",
        "--context-col", "context",
    ]
    if dry_run:
        cmd.append("--dry-run")
    env = {**os.environ, "TMPDIR": WORK}
    print(" ".join(cmd))
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(r.stdout[-8000:])
    if r.stderr:
        print("STDERR:", r.stderr[-8000:])
    if r.returncode != 0:
        raise RuntimeError(f"vcc prep failed with code {r.returncode}")
    VOL.commit()
    if not dry_run:
        print(f"{out}: {os.path.getsize(out) / 1e9:.2f} GB")
    return out


@app.local_entrypoint()
def main(scale: float = 4.0, lam: float = 0.15, lam_mode: str = "ramp",
         lam_thr: float = 2.0, contexts: str = "D,E,F",
         skip_build: bool = False, out_name: str = "transfer.vcc",
         dry_run: bool = False):
    path = OUT_H5AD if skip_build else build.remote(
        scale, lam, lam_mode, lam_thr, contexts)
    print("prediction:", path)
    print("packaged:", prep.remote(path, out_name, dry_run))
