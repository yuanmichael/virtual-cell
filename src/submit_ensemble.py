"""Build and submit the STATE-v3 + K562-transfer ensemble for contexts D/E/F.

The ensemble is the best model we have measured: +0.1134 against +0.0598 for
the transfer model alone on identical test cells and identical references. It
is also the only improvement that has never been submitted, which makes it
worth nothing until this script runs.

Three things make the D/E/F STATE arm different from the rehearsal arm, and all
three are places a silent error could hide:

1. GENE SPACE. v3 emits the 2025 panel's 18,080 genes; the 2026 panel asks for
   18,533. 18,077 overlap. We feed it controls projected down to its own space
   and lift the output back, leaving the ~456 genes it cannot speak about at
   real control values -- exactly what the transfer model already does with the
   genes outside its 7,683-gene readout.

2. VOCABULARIES. v3's `cell_type` map has ONE entry, "H1", and its `batch` map
   has the 48 Flex batches from the 2025 experiment. D/E/F are different
   contexts with different batches, and an unknown category means a zero vector
   -- i.e. "no cell type" -- so we present the cells under labels the model was
   trained on. This is a deliberate lie to the model about provenance; it is
   safe because STATE conditions on the BASAL CELLS (`predict_residual`), which
   are the real D/E/F controls, and the cell-type token only selects an
   embedding. The alternative is an untrained zero vector.

3. NO PERTURBED CELLS EXIST. The rehearsal fed STATE a file that already had
   perturbed cells to predict for. Here we must manufacture the request: 400
   rows per target whose `target_gene` names the target, with resampled real
   control cells as their basal content.

Run:
    modal run src/submit_ensemble.py --action smoke        # 2 targets, 1 ctx
    modal run src/submit_ensemble.py --action state --contexts D,E,F
    modal run src/submit_ensemble.py --action blend
"""

import modal

from common import (
    CONTROLS_FINAL, CPU_MEM_MIB, REHEARSAL, VOL, VOL_PATH, WORK, app,
    cpu_image, eval_image, state_image,
)

OUTDIR = f"{WORK}/submit"
V3_DIR = f"{VOL_PATH}/state/runs/v3/v3"
TRANSFER = f"{WORK}/pred_transfer.h5ad"
CELLS_PER_PERT = 400
SEED = 0
# The only cell type and a batch v3 was trained on. See note 2 above.
CELL_TYPE = "H1"
BATCH = "Flex_1_01"
DEPTH = 1e4


def _panel():
    """2026 panel gene order and target list."""
    import pandas as pd

    genes = pd.read_csv(f"{CONTROLS_FINAL}/gene_names.csv")["gene_name"].tolist()
    targets = pd.read_csv(f"{CONTROLS_FINAL}/pert_counts.csv")["target_gene"].tolist()
    return genes, targets


def _v3_genes():
    """v3's own 18,080-gene output order."""
    import csv

    with open(f"{REHEARSAL}/gene_names.csv", newline="") as fh:
        rows = [r[0] for r in csv.reader(fh) if r and r[0]]
    if rows[0].lower() in ("gene_name", "gene"):
        rows = rows[1:]
    return rows


@app.function(
    image=eval_image, volumes={VOL_PATH: VOL}, cpu=8.0, memory=CPU_MEM_MIB,
    timeout=60 * 60 * 4,
)
def build_input(ctx: str, n_targets: int = 0, n_control: int = 2000):
    """Write the STATE inference request for one context, in v3's gene space."""
    import os
    import time

    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    panel, targets = _panel()
    if n_targets:
        targets = targets[:n_targets]
    v3g = _v3_genes()
    pos = {g: i for i, g in enumerate(panel)}
    take = np.array([pos[g] for g in v3g if g in pos])
    keep = np.array([i for i, g in enumerate(v3g) if g in pos])
    print(f"[{ctx}] v3 genes present in the 2026 panel: {len(take)} of "
          f"{len(v3g)}")

    A = ad.read_h5ad(f"{CONTROLS_FINAL}/context_{ctx}.h5ad")
    assert list(A.var_names) == panel, "control gene order disagrees with panel"
    Xc = A.X.tocsr()
    Xc.data = np.rint(Xc.data).astype(np.float32)
    n_ctrl = Xc.shape[0]
    print(f"[{ctx}] {n_ctrl} control cells, {Xc.nnz:,} nonzeros")

    # Project to v3's gene order. Genes v3 knows that the panel lacks stay 0;
    # there are none in practice, but zero-filling silently would be a bug, so
    # the count is printed above and the arrays are built by position.
    Pc = sp.lil_matrix((n_ctrl, len(v3g)), dtype=np.float32)
    Pc[:, keep] = Xc[:, take]
    Pc = Pc.tocsr()

    # log1p(CP10k) -- the space v3 was trained in.
    tot = np.asarray(Pc.sum(axis=1)).ravel()
    tot[tot == 0] = 1.0
    Pc = sp.diags(DEPTH / tot) @ Pc
    Pc.data = np.log1p(Pc.data).astype(np.float32)
    print(f"[{ctx}] normalised: max {Pc.data.max():.2f} "
          f"p99 {np.quantile(Pc.data, 0.99):.2f}")

    rng = np.random.default_rng(SEED)
    blocks, obs_pert = [], []
    for t in targets:
        rows = np.sort(rng.choice(n_ctrl, CELLS_PER_PERT, replace=True))
        blocks.append(sp.csr_matrix(Pc[rows]))
        obs_pert += [t] * CELLS_PER_PERT
    cr = np.sort(rng.choice(n_ctrl, min(n_control, n_ctrl), replace=False))
    blocks.append(sp.csr_matrix(Pc[cr]))
    obs_pert += ["non-targeting"] * len(cr)

    X = sp.vstack(blocks, format="csr")
    obs = pd.DataFrame(
        {
            "target_gene": pd.Categorical(obs_pert),
            "cell_type": pd.Categorical([CELL_TYPE] * X.shape[0]),
            "batch": pd.Categorical([BATCH] * X.shape[0]),
        },
        index=[f"{ctx}_q{i}" for i in range(X.shape[0])],
    )
    out = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=v3g))
    path = f"{OUTDIR}/state_in_{ctx}.h5ad"
    out.write_h5ad(path, compression="lzf")
    VOL.commit()
    print(f"[{ctx}] wrote {path}: {out.shape}, {len(targets)} targets "
          f"({time.time()-t0:.0f}s)")

    # The real control cells in PANEL space, for depth-matching and for the
    # lift's non-covered genes. Saved so the GPU step need not re-read the
    # 217 MB original.
    ref = ad.AnnData(
        X=sp.csr_matrix(Xc[cr]),
        obs=pd.DataFrame(
            {"target_gene": pd.Categorical(["non-targeting"] * len(cr))},
            index=[f"{ctx}_c{i}" for i in range(len(cr))],
        ),
        var=pd.DataFrame(index=panel),
    )
    rpath = f"{OUTDIR}/ctrl_{ctx}.h5ad"
    ref.write_h5ad(rpath, compression="lzf")
    VOL.commit()
    print(f"[{ctx}] wrote {rpath}: {ref.shape}")
    return path


@app.function(
    image=state_image, volumes={VOL_PATH: VOL}, gpu="A10G", cpu=8.0,
    memory=49152, timeout=60 * 60 * 6,
)
def state_arm(ctx: str, ckpt: str = "best.ckpt", seed: int = 0):
    """v3 inference for one context, converted to counts and lifted to panel."""
    import argparse
    import os
    import time

    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import torch

    t0 = time.time()
    src = f"{OUTDIR}/state_in_{ctx}.h5ad"
    ctrl_p = f"{OUTDIR}/ctrl_{ctx}.h5ad"
    ckpt_path = f"{V3_DIR}/checkpoints/{ckpt}"
    for p in (src, ctrl_p, ckpt_path, f"{V3_DIR}/config.yaml"):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    # Same guard as src/state_infer.py: a perturbation missing from the model's
    # map silently falls back to the control vector, i.e. "predict no change",
    # which would reproduce submission 1's -1.72 and look like a model failure.
    pmap = torch.load(f"{V3_DIR}/pert_onehot_map.pt", weights_only=False)
    _, targets = _panel()
    missing = [t for t in targets if t not in pmap]
    print(f"v3 pert map: {len(pmap)} entries; "
          f"{len(targets)-len(missing)}/{len(targets)} 2026 targets present")
    if missing:
        raise RuntimeError(
            f"{len(missing)} targets absent from the model's perturbation map "
            f"-- they would silently predict no change: {missing[:10]}"
        )

    from state._cli._tx._infer import run_tx_infer
    import torch.serialization as _ts

    try:
        import numpy._core.multiarray as _ma
        _ts.add_safe_globals([_ma.scalar, np.dtype, np.ndarray])
    except Exception as e:  # noqa: BLE001
        print(f"add_safe_globals: {type(e).__name__}: {e}")
    _orig = _ts.load

    def _load(*a, **kw):
        kw["weights_only"] = False
        return _orig(*a, **kw)

    torch.load = _load
    _ts.load = _load
    raw_out = f"{OUTDIR}/state_raw_{ctx}.h5ad"
    args = argparse.Namespace(
        checkpoint=ckpt_path, adata=src, output=raw_out, model_dir=V3_DIR,
        embed_key=None, pert_col="target_gene", control_pert="non-targeting",
        celltype_col="cell_type", celltypes=None, batch_col="batch",
        seed=seed, max_set_len=None, quiet=False, tsv=False, all_perts=False,
        virtual_cells_per_pert=None, min_cells=None, max_cells=None,
    )
    try:
        run_tx_infer(args)
    finally:
        torch.load = _orig
        _ts.load = _orig
        VOL.commit()
    print(f"[{ctx}] inference done in {time.time()-t0:.0f}s")

    panel, targets = _panel()
    v3g = _v3_genes()
    P = ad.read_h5ad(raw_out)
    C = ad.read_h5ad(ctrl_p)
    print(f"[{ctx}] raw pred {P.shape}, controls {C.shape}")

    # Depth target: the real controls' library size over the SAME genes v3
    # emits. Matching against all 18,533 would rescale by the ratio of two
    # different gene sets.
    pos = {g: i for i, g in enumerate(panel)}
    ov = [g for g in v3g if g in pos]
    cov_panel = np.array([pos[g] for g in ov])
    cov_v3 = np.array([i for i, g in enumerate(v3g) if g in pos])
    Xc = C.X.tocsr() if sp.issparse(C.X) else sp.csr_matrix(C.X)
    d = np.asarray(Xc[:, cov_panel].sum(axis=1)).ravel()
    target_depth = float(np.median(d[d > 0]))
    print(f"[{ctx}] real control depth over v3's genes: {target_depth:.0f} UMI")

    Xp = P.X.toarray() if sp.issparse(P.X) else np.asarray(P.X)
    Xp = np.expm1(np.clip(Xp, 0.0, 14.0))
    Xp = np.clip(Xp, 0.0, None)
    rs = Xp.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    Xp = Xp / rs * target_depth

    rng = np.random.default_rng(seed)
    fl = np.floor(Xp)
    Xp = (fl + (rng.random(Xp.shape) < (Xp - fl))).astype(np.float32)
    print(f"[{ctx}] counts: median depth "
          f"{np.median(Xp.sum(axis=1)):.0f}")

    # Keep only the perturbed rows, in the panel's target order.
    pert = P.obs["target_gene"].astype(str).to_numpy()
    found = {t: np.where(pert == t)[0] for t in targets}
    want = np.concatenate([found[t] for t in targets])
    short = {t: len(v) for t, v in found.items() if len(v) != CELLS_PER_PERT}
    # A target absent from the predictions yields an empty index array and
    # would be dropped in silence, leaving the submission short of cells for
    # that perturbation -- which `vcc prep` would reject much later, or worse
    # accept. Fail here, where the cause is visible.
    if short and not os.environ.get("ALLOW_SHORT"):
        raise RuntimeError(
            f"[{ctx}] {len(short)} targets do not have exactly "
            f"{CELLS_PER_PERT} predicted cells: "
            f"{list(short.items())[:8]}"
        )
    Xp = Xp[want]
    pert = pert[want]
    print(f"[{ctx}] kept {len(want)} perturbed rows "
          f"({len(found)} targets x {CELLS_PER_PERT})")

    # Lift to panel space: start from real control cells so the ~456 genes v3
    # cannot speak about carry real variation rather than zeros, then overwrite
    # the covered columns.
    base = sp.csr_matrix(
        Xc[rng.choice(Xc.shape[0], len(want), replace=True)]).tolil()
    base[:, cov_panel] = Xp[:, cov_v3]
    X = sp.csr_matrix(base, dtype=np.float32)
    X.eliminate_zeros()
    assert X.data.min() >= 0 and np.all(X.data == np.floor(X.data))

    out = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"target_gene": pd.Categorical(pert),
             "context": pd.Categorical([ctx] * X.shape[0])},
            index=[f"{ctx}_s{i}" for i in range(X.shape[0])],
        ),
        var=pd.DataFrame(index=panel),
    )
    path = f"{OUTDIR}/pred_state_{ctx}.h5ad"
    out.write_h5ad(path, compression="lzf")
    VOL.commit()
    print(f"[{ctx}] wrote {path}: {out.shape} "
          f"({os.path.getsize(path)/1e9:.2f} GB) in {time.time()-t0:.0f}s")
    return path


@app.function(
    image=cpu_image, volumes={VOL_PATH: VOL}, cpu=8.0, memory=49152,
    timeout=60 * 60 * 6,
)
def finish_arm(ctx: str, seed: int = 0, chunk: int = 10000):
    """Convert + lift an ALREADY-COMPUTED state_raw_{ctx}.h5ad, in row chunks.

    Split out of state_arm because the GPU inference succeeded and was saved
    while the post-processing died: a 120,000 x 18,533 `.tolil()` is far too
    much Python object overhead for a 49 GB container, even though the same
    code worked on 62,239 rows. Chunking also means a failure costs one chunk,
    not a GPU hour.

    No LIL: each chunk is assembled as a COO over explicit panel column
    indices, which never materialises a dense full-width block.
    """
    import os
    import time

    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    t0 = time.time()
    panel, targets = _panel()
    v3g = _v3_genes()
    raw = f"{OUTDIR}/state_raw_{ctx}.h5ad"
    C = ad.read_h5ad(f"{OUTDIR}/ctrl_{ctx}.h5ad")
    Xc = C.X.tocsr() if sp.issparse(C.X) else sp.csr_matrix(C.X)

    pos = {g: i for i, g in enumerate(panel)}
    cov_v3 = np.array([i for i, g in enumerate(v3g) if g in pos])
    cov_panel = np.array([pos[g] for g in v3g if g in pos])
    non_panel = np.setdiff1d(np.arange(len(panel)), cov_panel)
    print(f"[{ctx}] covered {len(cov_panel)} panel genes, "
          f"{len(non_panel)} take control values")

    d = np.asarray(Xc[:, cov_panel].sum(axis=1)).ravel()
    target_depth = float(np.median(d[d > 0]))
    print(f"[{ctx}] real control depth over v3's genes: {target_depth:.0f} UMI")

    A = ad.read_h5ad(raw, backed="r")
    pert = A.obs["target_gene"].astype(str).to_numpy()
    found = {t: np.where(pert == t)[0] for t in targets}
    short = {t: len(v) for t, v in found.items() if len(v) != CELLS_PER_PERT}
    if short:
        raise RuntimeError(f"[{ctx}] {len(short)} targets short: "
                           f"{list(short.items())[:8]}")
    want = np.concatenate([found[t] for t in targets])
    print(f"[{ctx}] {len(want)} perturbed rows over {len(targets)} targets")

    rng = np.random.default_rng(seed)
    parts_dir = f"{OUTDIR}/_sparts_{ctx}"
    os.makedirs(parts_dir, exist_ok=True)
    nparts = 0
    for s in range(0, len(want), chunk):
        e = min(s + chunk, len(want))
        pf = f"{parts_dir}/part_{s//chunk:04d}.h5ad"
        nparts += 1
        if os.path.exists(pf):
            continue
        rows = want[s:e]
        order = np.argsort(rows, kind="stable")
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        D = np.asarray(A.X[rows[order]], dtype=np.float32)[inv]
        D = np.expm1(np.clip(D, 0.0, 14.0))
        np.clip(D, 0.0, None, out=D)
        rs = D.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        D *= target_depth / rs
        fl = np.floor(D)
        D = fl + (rng.random(D.shape) < (D - fl))

        # Covered columns from the prediction, as COO over PANEL indices.
        Mc = sp.coo_matrix(D[:, cov_v3])
        rr = [Mc.row]
        cc = [cov_panel[Mc.col]]
        dd = [Mc.data.astype(np.float32)]
        # Non-covered columns from real control cells.
        base = Xc[rng.choice(Xc.shape[0], e - s, replace=True)][:, non_panel]
        Bn = base.tocoo()
        rr.append(Bn.row)
        cc.append(non_panel[Bn.col])
        dd.append(Bn.data.astype(np.float32))

        X = sp.coo_matrix(
            (np.concatenate(dd), (np.concatenate(rr), np.concatenate(cc))),
            shape=(e - s, len(panel)), dtype=np.float32).tocsr()
        X.eliminate_zeros()
        assert X.data.min() >= 0 and np.all(X.data == np.floor(X.data))
        obs = pd.DataFrame(
            {"target_gene": pd.Categorical(pert[want[s:e]],
                                           categories=sorted(set(targets))),
             "context": pd.Categorical([ctx] * (e - s), categories=[ctx])},
            index=[f"{ctx}_s{i}" for i in range(s, e)],
        )
        ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=panel)).write_h5ad(
            pf, compression="lzf")
        del D, Mc, X
        print(f"  [{ctx}] {e}/{len(want)} ({time.time()-t0:.0f}s)", flush=True)
        VOL.commit()
    A.file.close()

    from anndata.experimental import concat_on_disk
    pfs = sorted(f"{parts_dir}/{x}" for x in os.listdir(parts_dir)
                 if x.endswith(".h5ad"))
    out = f"{OUTDIR}/pred_state_{ctx}.h5ad"
    concat_on_disk(pfs, out, label=None)
    VOL.commit()
    chk = ad.read_h5ad(out, backed="r")
    print(f"[{ctx}] wrote {out}: {chk.shape} "
          f"({os.path.getsize(out)/1e9:.2f} GB) in {time.time()-t0:.0f}s")
    chk.file.close()
    import shutil
    shutil.rmtree(parts_dir)
    VOL.commit()
    return out


@app.function(
    image=cpu_image, volumes={VOL_PATH: VOL}, cpu=8.0, memory=CPU_MEM_MIB,
    timeout=60 * 60 * 6,
)
def blend(contexts: str = "D,E,F", w: float = 0.5, seed: int = 0,
          out_name: str = "ensemble.vcc"):
    """w*STATE + (1-w)*transfer, per context, then package with `vcc prep`."""
    import os
    import subprocess
    import time

    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    t0 = time.time()
    panel, targets = _panel()
    ctxs = [c.strip() for c in contexts.split(",") if c.strip()]
    T = ad.read_h5ad(TRANSFER)
    tpert = T.obs["target_gene"].astype(str).to_numpy()
    tctx = T.obs["context"].astype(str).to_numpy()
    print(f"transfer arm: {T.shape}")
    assert list(T.var_names) == panel, "transfer gene order disagrees"

    rng = np.random.default_rng(seed)
    paths = []
    for ctx in ctxs:
        S = ad.read_h5ad(f"{OUTDIR}/pred_state_{ctx}.h5ad")
        spert = S.obs["target_gene"].astype(str).to_numpy()
        # Align by (context, target) rather than by position: the two arms were
        # built by different scripts and a positional assumption that silently
        # held for D could break for E.
        rows_t = np.concatenate(
            [np.where((tctx == ctx) & (tpert == t))[0] for t in targets])
        rows_s = np.concatenate([np.where(spert == t)[0] for t in targets])
        assert len(rows_t) == len(rows_s), f"{ctx}: arm sizes differ"
        A = S.X.tocsr()[rows_s]
        B = T.X.tocsr()[rows_t]
        M = (A.multiply(w) + B.multiply(1.0 - w)).tocsr()
        dd = np.asarray(M.data, dtype=np.float64)
        fl = np.floor(dd)
        M.data = (fl + (rng.random(dd.shape) < (dd - fl))).astype(np.float32)
        M.eliminate_zeros()
        assert M.data.min() >= 0 and np.all(M.data == np.floor(M.data))
        obs = pd.DataFrame(
            {"target_gene": pd.Categorical(
                np.repeat(targets, CELLS_PER_PERT)),
             "context": pd.Categorical([ctx] * M.shape[0])},
            index=[f"{ctx}_cell_{i}" for i in range(M.shape[0])],
        )
        out = ad.AnnData(X=M, obs=obs, var=pd.DataFrame(index=panel))
        p = f"{OUTDIR}/ens_{ctx}.h5ad"
        out.write_h5ad(p, compression="lzf")
        paths.append(p)
        VOL.commit()
        print(f"[{ctx}] blended {out.shape} -> {p} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del S, A, B, M, out

    final = f"{OUTDIR}/pred_ensemble.h5ad"
    if len(paths) == 1:
        os.replace(paths[0], final)
    else:
        from anndata.experimental import concat_on_disk
        concat_on_disk(paths, final, label=None)
        for p in paths:
            os.remove(p)
    VOL.commit()
    print(f"concatenated -> {final} "
          f"({os.path.getsize(final)/1e9:.2f} GB)")

    outv = f"{WORK}/{out_name}"
    cmd = ["vcc", "prep", final, "-g", f"{CONTROLS_FINAL}/gene_names.csv",
           "--perts", f"{CONTROLS_FINAL}/pert_counts.csv", "-o", outv,
           "--pert-col", "target_gene", "--context-col", "context"]
    env = {**os.environ, "TMPDIR": WORK}
    print(" ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(r.stdout[-6000:])
    if r.stderr:
        print("STDERR:", r.stderr[-4000:])
    if r.returncode != 0:
        raise RuntimeError(f"vcc prep failed ({r.returncode})")
    VOL.commit()
    print(f"{outv}: {os.path.getsize(outv)/1e9:.2f} GB")
    return outv


@app.local_entrypoint()
def main(action: str = "smoke", contexts: str = "D,E,F", n_targets: int = 0,
         w: float = 0.5):
    ctxs = [c.strip() for c in contexts.split(",") if c.strip()]
    if action == "input":
        for c in ctxs:
            print(build_input.remote(c, n_targets))
    elif action == "smoke":
        # 2 targets, one context, all the way through the STATE arm: proves
        # inference, the counts conversion and the lift for a few cents.
        c = ctxs[0]
        build_input.remote(c, n_targets or 2)
        print(state_arm.remote(c))
    elif action == "state":
        for c in ctxs:
            print(state_arm.remote(c))
    elif action == "blend":
        print(blend.remote(contexts, w))
    else:
        raise SystemExit(f"unknown action {action!r}")


@app.function(
    image=cpu_image, volumes={VOL_PATH: VOL}, cpu=4.0, memory=16384,
    timeout=60 * 30,
)
def inspect_pred(path: str = ""):
    """Dtypes and nnz of the packaged prediction. `vcc prep` failed with
    'Output dtype not compatible with inputs', which is a dtype-promotion
    error, so look at the actual types rather than guessing."""
    import h5py
    import numpy as np

    p = path or f"{OUTDIR}/pred_ensemble.h5ad"
    with h5py.File(p, "r") as h:
        X = h["X"]
        print(f"X encoding: {dict(X.attrs)}")
        for k in ("data", "indices", "indptr"):
            if k in X:
                d = X[k]
                print(f"  X/{k}: {d.shape} {d.dtype}")
        nnz = X["data"].shape[0]
        print(f"nnz = {nnz:,}  (int32 max 2,147,483,647 -> "
              f"{'OVERFLOWS' if nnz > 2**31-1 else 'fits'})")
        obs = h["obs"]
        print(f"obs columns: {list(obs.keys())}")
        for c in obs:
            o = obs[c]
            if isinstance(o, h5py.Group):
                print(f"  obs/{c}: categorical, "
                      f"{o['categories'].shape[0]} cats, "
                      f"codes {o['codes'].dtype}")
            else:
                print(f"  obs/{c}: {o.shape} {o.dtype}")
        print(f"var: {h['var'][h['var'].attrs['_index']].shape[0]} genes")
    return p


@app.function(
    image=cpu_image, volumes={VOL_PATH: VOL}, cpu=8.0, memory=131072,
    timeout=60 * 60 * 5,
)
def prep_only(src: str = "", out_name: str = "ensemble.vcc",
              force_int64: bool = True, dry_run: bool = False):
    """Make the prediction representable, then package it.

    `vcc prep` died with scipy's `ValueError: Output dtype not compatible with
    inputs`, raised from the `_sparsetools` C extension. Cause: the blended
    matrix has **2,694,168,217 nonzeros**, past int32's 2,147,483,647, while
    X/indices was written int32. scipy picks its 32- or 64-bit kernels from the
    index dtype, so an int32-indexed matrix that large cannot be operated on.

    (The transfer arm alone packaged fine at ~1.9e9. Blending pushed it over,
    because averaging two predictions takes the UNION of their nonzero
    patterns -- our cells carry 7,484 nonzero genes against ~5,679 in real
    data, 1.32x too dense.)

    Rewriting indices as int64 keeps the measured model byte-for-byte in value
    and only changes how it is addressed. Streamed in blocks, so this never
    holds 2.7e9 elements in memory.
    """
    import os
    import shutil
    import subprocess
    import time

    import h5py
    import numpy as np

    t0 = time.time()
    p = src or f"{OUTDIR}/pred_ensemble.h5ad"
    if force_int64:
        with h5py.File(p, "r") as h:
            idt = h["X"]["indices"].dtype
            nnz = h["X"]["data"].shape[0]
        print(f"X/indices dtype {idt}, nnz {nnz:,}")
        if idt == np.int32 and nnz > 2**31 - 1:
            dst = p.replace(".h5ad", "_i64.h5ad")
            if not os.path.exists(dst):
                print(f"rewriting indices as int64 -> {dst}", flush=True)
                shutil.copyfile(p, dst)
                with h5py.File(dst, "r+") as h:
                    X = h["X"]
                    old = X["indices"]
                    n = old.shape[0]
                    tmp = X.create_dataset(
                        "indices64", shape=(n,), dtype="int64",
                        chunks=(1 << 22,), compression="lzf")
                    step = 1 << 24
                    for i in range(0, n, step):
                        j = min(i + step, n)
                        tmp[i:j] = old[i:j].astype(np.int64)
                        if (i // step) % 20 == 0:
                            print(f"  {j:,}/{n:,} "
                                  f"({time.time()-t0:.0f}s)", flush=True)
                    del X["indices"]
                    X.move("indices64", "indices")
                VOL.commit()
                print(f"rewrote in {time.time()-t0:.0f}s "
                      f"({os.path.getsize(dst)/1e9:.2f} GB)")
            p = dst
        else:
            print("indices already adequate; no rewrite")

    outv = f"{WORK}/{out_name}"
    cmd = ["vcc", "prep", p, "-g", f"{CONTROLS_FINAL}/gene_names.csv",
           "--perts", f"{CONTROLS_FINAL}/pert_counts.csv", "-o", outv,
           "--pert-col", "target_gene", "--context-col", "context"]
    if dry_run:
        cmd.append("--dry-run")
    env = {**os.environ, "TMPDIR": WORK}
    print(" ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(r.stdout[-8000:])
    if r.stderr:
        print("STDERR:", r.stderr[-8000:])
    if r.returncode != 0:
        raise RuntimeError(f"vcc prep failed ({r.returncode})")
    VOL.commit()
    if not dry_run:
        print(f"{outv}: {os.path.getsize(outv)/1e9:.2f} GB")
    return outv
