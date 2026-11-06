"""Move data between Replogle's ~6,700-gene space and the VCC panels'.

A model trained on Replogle emits Replogle's gene space. Everything we are
graded on is in a VCC panel's space -- 18,080 for the 2025 rehearsal, 18,533
for the 2026 final panel. `state_infer.py` assumes the prediction and the
ground truth share a var index, which is true for the 2025-trained models and
false here, so two conversions are needed and neither is a no-op.

`project` (down)
    A VCC-space h5ad -> Replogle's gene space, for FEEDING the model. cell-load
    matches the model's input dimension by position, so a column order that
    merely contains the right genes is not enough; it has to be exactly the
    order the model was trained on, which is `meta.json["genes"]`.

`lift` (up)
    A Replogle-space prediction -> VCC space, for SCORING. The ~11,400 genes
    Replogle never measured take values from real control cells, which is
    precisely what the transfer model already does with the genes outside its
    7,683-gene readout: "no predicted change" expressed as real control
    variation rather than as a constant, so the DE tests see realistic spread
    instead of an impossible zero-variance column.

A gene Replogle measures but the panel does not is simply dropped -- we are not
graded on it.

Run:
    modal run src/replogle_bridge.py --action project --src rehearsal/adata_Test_norm.h5ad
    modal run src/replogle_bridge.py --action lift --src work/state_eval/replogle_pred.h5ad
"""

import modal

from common import REHEARSAL, VOL, VOL_PATH, WORK, app, eval_image

OUTDIR = f"{WORK}/state_eval"
META = f"{VOL_PATH}/replogle_norm/meta.json"


def _vol(p):
    """Resolve a user-supplied path onto the Volume.

    Absolute /vol/... paths are unsafe to pass from a Git Bash shell: MSYS
    rewrites a leading slash, so `--out /vol/work/x.h5ad` arrives as
    `C:/Program Files/Git/vol/work/x.h5ad` and the file lands inside the
    container instead of on the Volume, vanishing when it exits. Normalising
    here means relative paths are the safe default and the mangled form is
    caught rather than silently written somewhere useless.
    """
    if not p:
        return p
    if "Program Files" in p or p.startswith("C:"):
        raise ValueError(
            f"{p!r} looks shell-mangled; pass a Volume-relative path "
            f"like work/state_eval/x.h5ad"
        )
    return p if p.startswith(VOL_PATH) else f"{VOL_PATH}/{p.lstrip('/')}"


def _genes():
    import json
    import os

    if not os.path.exists(META):
        raise FileNotFoundError(f"{META} -- run src/replogle_prep.py first")
    return json.load(open(META))["genes"]


@app.function(
    image=eval_image, volumes={VOL_PATH: VOL},
    cpu=8.0, memory=49152, timeout=60 * 60 * 3,
)
def project(src: str, out: str = ""):
    """VCC-space h5ad -> Replogle gene order, zero-filling anything absent."""
    import os
    import time

    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    t0 = time.time()
    genes = _genes()
    path = _vol(src)
    A = ad.read_h5ad(path)
    have = {str(g): i for i, g in enumerate(A.var_names)}
    print(f"{path}: {A.shape}")

    missing = [g for g in genes if g not in have]
    if missing:
        # Zero-filling is a real distortion of the input, so it must be small.
        # A large overlap failure usually means a symbol-convention mismatch
        # rather than genuinely absent genes, and silently zeroing 2,000
        # columns would look like a bad model later.
        print(f"WARNING: {len(missing)} of {len(genes)} Replogle genes absent "
              f"from the source, zero-filled: {missing[:10]}")
        if len(missing) > len(genes) * 0.05:
            raise RuntimeError(
                f"{len(missing)}/{len(genes)} genes missing (>5%) -- check "
                f"gene symbol conventions before trusting this"
            )

    X = A.X.tocsc() if sp.issparse(A.X) else sp.csc_matrix(A.X)
    cols = []
    for g in genes:
        i = have.get(g)
        cols.append(X[:, i] if i is not None
                    else sp.csc_matrix((A.n_obs, 1), dtype=np.float32))
    Xp = sp.hstack(cols, format="csr", dtype=np.float32)
    print(f"projected to {Xp.shape} in {time.time()-t0:.0f}s")

    import pandas as pd

    obs = A.obs.copy()
    # Plain str, not the source's pandas Index. anndata 0.13 refuses to write
    # a nullable-string index unless `allow_write_nullable_strings` is set --
    # the same trap src/state_infer.py's ensemble() already documents, and
    # reading obs straight through reintroduces it.
    obs.index = [str(x) for x in obs.index]
    for c in obs.columns:
        if obs[c].dtype.name in ("string", "object"):
            obs[c] = pd.Categorical([str(v) for v in obs[c]])
    B = ad.AnnData(X=Xp, obs=obs, var=pd.DataFrame(index=genes))
    dst = _vol(out) if out else f"{OUTDIR}/{os.path.basename(path).replace('.h5ad', '')}_rspace.h5ad"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    B.write_h5ad(dst, compression="lzf")
    VOL.commit()
    print(f"wrote {dst} ({os.path.getsize(dst)/1e9:.2f} GB)")
    return dst


@app.function(
    image=eval_image, volumes={VOL_PATH: VOL},
    cpu=8.0, memory=49152, timeout=60 * 60 * 3,
)
def lift(src: str, panel_src: str, out: str = "", seed: int = 0,
         control: str = "non-targeting"):
    """Replogle-space prediction -> the panel's gene space.

    `panel_src` supplies both the target gene order and the real control cells
    that fill the genes Replogle never measured.
    """
    import os
    import time

    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    t0 = time.time()
    pp = _vol(src)
    rp = _vol(panel_src)
    P = ad.read_h5ad(pp)
    R = ad.read_h5ad(rp)
    print(f"prediction {P.shape} (Replogle space)")
    print(f"panel      {R.shape}")

    panel = [str(g) for g in R.var_names]
    pidx = {str(g): i for i, g in enumerate(P.var_names)}
    # Columns of the panel that the model actually predicts, and where they sit
    # in the prediction.
    tgt = np.array([i for i, g in enumerate(panel) if g in pidx])
    srcc = np.array([pidx[panel[i]] for i in tgt])
    print(f"model covers {len(tgt)} of {len(panel)} panel genes "
          f"({len(tgt)/len(panel)*100:.0f}%); the rest take control values")

    rpert = R.obs["target_gene"].astype(str).to_numpy()
    ctrl_rows = np.where(rpert == control)[0]
    if len(ctrl_rows) == 0:
        raise RuntimeError(f"no '{control}' cells in {rp} to fill from")
    Xr = R.X.tocsr() if sp.issparse(R.X) else sp.csr_matrix(R.X)

    rng = np.random.default_rng(seed)
    take = rng.choice(ctrl_rows, P.n_obs, replace=True)
    # Start from real control cells across the FULL panel, then overwrite the
    # covered columns with the model's prediction. Building it this way means a
    # gene the model does not cover is never a zero -- it is a real
    # measurement, with real cell-to-cell variance.
    Base = sp.csr_matrix(Xr[np.sort(take)], dtype=np.float32).tolil()
    Xp = P.X.toarray() if sp.issparse(P.X) else np.asarray(P.X)
    Base[:, tgt] = Xp[:, srcc]
    X = sp.csr_matrix(Base, dtype=np.float32)
    X.eliminate_zeros()
    assert X.data.min() >= 0, "negative counts would be rejected by prep"
    assert np.all(X.data == np.floor(X.data)), "fractional counts survived"
    print(f"lifted to {X.shape}, {X.nnz/1e6:.0f}M nonzeros "
          f"in {time.time()-t0:.0f}s")

    B = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"target_gene": pd.Categorical(
                P.obs["target_gene"].astype(str).to_numpy())},
            index=[f"l{i}" for i in range(X.shape[0])],
        ),
        var=pd.DataFrame(index=panel),
    )
    dst = _vol(out) if out else f"{OUTDIR}/{os.path.basename(pp).replace('.h5ad','')}_lifted.h5ad"
    B.write_h5ad(dst, compression="lzf")
    VOL.commit()
    print(f"wrote {dst} ({os.path.getsize(dst)/1e9:.2f} GB)")
    return dst


@app.local_entrypoint()
def main(action: str = "project", src: str = "", panel_src: str = "",
         out: str = "", seed: int = 0):
    if action == "project":
        print(project.remote(src, out))
    elif action == "lift":
        if not panel_src:
            raise SystemExit("--panel-src is required for lift")
        print(lift.remote(src, panel_src, out, seed))
    else:
        raise SystemExit(f"unknown action {action!r}")
