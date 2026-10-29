"""Trivial end-to-end baseline for VCC 2026.

Prediction rule: for each (context, target_gene), resample 400 real control cells
and zero out the target gene's counts. Everything else is left untouched.

This is deliberately uncompetitive -- it predicts no downstream response at all.
Its job is to exercise the full submission pipeline (build -> vcc prep -> submit)
and to tell us empirically what the challenge's 0-anchor corresponds to.

Run:  modal run src/baseline.py
"""

import modal

from common import (
    CONTROLS,
    CPU_CORES,
    CPU_MEM_MIB,
    WORK,
    app,
    cpu_image,
    VOL,
    VOL_PATH,
)

CELLS_PER_PERT = 400
SEED = 0


@app.function(
    image=cpu_image,
    volumes={VOL_PATH: VOL},
    cpu=CPU_CORES,
    memory=CPU_MEM_MIB,
    timeout=60 * 60 * 2,
)
def build(contexts: list[str] | None = None, knockdown_frac: float = 0.0):
    """Build one .h5ad holding all predicted cells for every context.

    knockdown_frac: residual fraction of the target gene's counts to keep.
        0.0 = full knockout of the target transcript. CRISPRi in this dataset
        achieves >=80% knockdown, so 0.1-0.2 is more faithful; 0.0 is the
        cleanest signal for the DE-direction metrics.
    """
    import json
    import os
    import time

    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    contexts = contexts or ["A", "B", "C"]
    os.makedirs(WORK, exist_ok=True)

    manifest = json.load(open(f"{CONTROLS}/manifest.json"))
    genes = pd.read_csv(f"{CONTROLS}/gene_names.csv")["gene_name"].tolist()
    targets = pd.read_csv(f"{CONTROLS}/pert_counts.csv")["target_gene"].tolist()
    gidx = {g: i for i, g in enumerate(genes)}
    print(f"manifest: {manifest['n_genes']} genes, {len(targets)} targets, "
          f"{manifest['cells_per_pert']} cells/pert")
    assert len(genes) == manifest["n_genes"]
    missing = [t for t in targets if t not in gidx]
    assert not missing, f"targets absent from gene list: {missing}"

    rng = np.random.default_rng(SEED)
    per_context_paths = []

    for ctx in contexts:
        t0 = time.time()
        path = f"{WORK}/pred_context_{ctx}.h5ad"
        if os.path.exists(path):
            # Checkpoint from an earlier run -- don't redo it.
            print(f"[{ctx}] reusing {path} ({os.path.getsize(path) / 1e9:.2f} GB)")
            per_context_paths.append(path)
            continue
        src = ad.read_h5ad(f"{CONTROLS}/context_{ctx}.h5ad")
        assert list(src.var_names) == genes, f"gene order mismatch in context {ctx}"
        Xc = src.X.tocsr()
        # Raw counts are stored as float32; keep them integral.
        Xc.data = np.rint(Xc.data).astype(np.float32)
        n_ctrl = Xc.shape[0]
        print(f"[{ctx}] {n_ctrl} control cells, {Xc.nnz:,} nonzeros")

        blocks, obs_pert = [], []
        for tgt in targets:
            rows = rng.choice(n_ctrl, size=CELLS_PER_PERT, replace=False)
            blk = Xc[rows].copy()
            hit = blk.indices == gidx[tgt]
            if knockdown_frac <= 0.0:
                blk.data[hit] = 0.0
            else:
                blk.data[hit] = np.rint(blk.data[hit] * knockdown_frac)
            blk.eliminate_zeros()
            blocks.append(blk)
            obs_pert.extend([tgt] * CELLS_PER_PERT)

        X = sp.vstack(blocks, format="csr")
        del blocks
        obs = pd.DataFrame(
            {
                "target_gene": pd.Categorical(obs_pert),
                "context": pd.Categorical([ctx] * X.shape[0]),
            },
            index=[f"{ctx}_cell_{i}" for i in range(X.shape[0])],
        )
        out = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
        # lzf, not gzip: these are throwaway intermediates and gzip on ~700M
        # nonzeros costs more in CPU-seconds than it saves in Volume storage.
        out.write_h5ad(path, compression="lzf")
        del out, X, Xc, src
        per_context_paths.append(path)
        print(f"[{ctx}] wrote {path} in {time.time() - t0:.0f}s "
              f"({os.path.getsize(path) / 1e9:.2f} GB)")
        VOL.commit()

    merged = f"{WORK}/pred_baseline.h5ad"
    if len(per_context_paths) == 1:
        os.replace(per_context_paths[0], merged)
    else:
        from anndata.experimental import concat_on_disk

        t0 = time.time()
        concat_on_disk(per_context_paths, merged, label=None)
        print(f"concat_on_disk -> {merged} in {time.time() - t0:.0f}s "
              f"({os.path.getsize(merged) / 1e9:.2f} GB)")
        for p in per_context_paths:
            os.remove(p)
    VOL.commit()
    return merged


@app.function(
    image=cpu_image,
    volumes={VOL_PATH: VOL},
    cpu=CPU_CORES,
    memory=CPU_MEM_MIB,
    timeout=60 * 60 * 2,
)
def prep(h5ad_path: str, out_name: str = "baseline.vcc", dry_run: bool = False):
    """Run `vcc prep` on the volume, where there is room for its temp files."""
    import os
    import subprocess

    out = f"{WORK}/{out_name}"
    cmd = [
        "vcc", "prep", h5ad_path,
        "-g", f"{CONTROLS}/gene_names.csv",
        "--perts", f"{CONTROLS}/pert_counts.csv",
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
def main(contexts: str = "A,B,C", knockdown_frac: float = 0.0, skip_build: bool = False):
    ctx = [c.strip() for c in contexts.split(",") if c.strip()]
    path = f"{WORK}/pred_baseline.h5ad" if skip_build else build.remote(ctx, knockdown_frac)
    print("prediction:", path)
    print("packaged:", prep.remote(path))
