"""Convert Replogle's single-cell screens into what cell-load and STATE want.

Three problems the 2025 pipeline never had, and none of them is optional.

1. DIFFERENT GENE SPACES. K562-gwps measures 8,248 genes and RPE1 8,749. A
   model has one output layer, so the two files cannot be trained on together
   until they share a gene space. We take the intersection, further restricted
   to genes some VCC panel actually scores -- a gene we are not graded on is
   capacity spent for nothing.

2. STORED DENSE. 65.83 GB / (8248 x 4 bytes) = 1.99M rows exactly, so X is a
   dense float32 block. Raw UMI counts are mostly zeros, so rewriting as
   sparse CSR shrinks it several-fold and makes training I/O-cheap rather than
   I/O-bound. This is the difference between a feasible run and a slow one.

3. WRONG SPACE FOR THE MODEL. `state tx infer` clips output to [0, 14], which
   only makes sense in log1p space; training in count space saturates every
   highly expressed gene and pins de_wilcoxon_lfc_nmae at its -6.0 penalty cap
   (the bug that cost three scoring runs -- see the README). CP10k + log1p is
   per-row, so it preserves every zero and composes with (2) in one pass.

So: one streaming pass per file that subsets columns, normalises rows, and
emits sparse CSR. Chunked by rows, resumable, and it never holds more than a
chunk of dense data in RAM.

Arc's own Replogle column names, from their released checkpoint config
(`external/state_ckpt_meta/zeroshot__k562__config.yaml`): `pert_col: gene`,
`batch_col: gem_group`, `cell_type_key: cell_line`,
`control_pert: non-targeting`. `cell_line` is spliced in per file, since each
file is a single line.

Run:
    modal run src/replogle_prep.py --plan-only      # gene space + shapes
    modal run --detach src/replogle_prep.py
"""

import modal

from common import (
    CONTROLS_FINAL, REHEARSAL, VOL, VOL_PATH, app, cpu_image, eval_image,
)

REPLOGLE = f"{VOL_PATH}/replogle"
OUTDIR = f"{VOL_PATH}/replogle_norm"
STATE_ROOT = f"{VOL_PATH}/state"
TOML_PATH = f"{STATE_ROOT}/replogle.toml"

FILES = {
    "K562": "K562_gwps_raw_singlecell_01.h5ad",
    "RPE1": "rpe1_raw_singlecell_01.h5ad",
}
CONTROL_PERT = "non-targeting"
DEPTH = 1e4
ROW_CHUNK = 20000   # ~20k x 8.2k float32 = 660 MB dense per read


def _decode(a):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in a]


def _cat_str(side, key):
    """Read a string column, in either of the two layouts these files use.

    Replogle's single-cell h5ads predate anndata 0.7 and store a categorical as
    int16 codes in `obs/<key>` with the levels in a SIBLING `obs/__categories`
    group -- not as a `{categories, codes}` group under `obs/<key>` the way
    modern anndata does. Confirmed by inspection: `obs/gene` is `int16
    (247914,)` and `var/gene_name` reads back as int16 codes.

    Handling only the modern layout returns the integer codes rendered as
    strings, which is silent and catastrophic: every perturbation label becomes
    a number, the control label never matches, and training proceeds on
    nonsense. `src/build_axis.py` has the same helper for the bulk files.
    """
    import h5py
    import numpy as np

    d = side[key]
    if isinstance(d, h5py.Group) and "categories" in d:        # modern
        cats = np.array(_decode(d["categories"][:]))
        codes = d["codes"][:]
        return np.where(codes >= 0, cats[np.clip(codes, 0, None)], "")
    if (d.dtype.kind in "iu" and "__categories" in side
            and key in side["__categories"]):                   # legacy
        cats = np.array(_decode(side["__categories"][key][:]))
        codes = d[:]
        return np.where(codes >= 0, cats[np.clip(codes, 0, None)], "")
    return np.array(_decode(d[:]))


def _obs_str(h, key):
    return _cat_str(h["obs"], key)


def _var_names(h):
    import numpy as np

    var = h["var"]
    if "gene_name" in var:
        out = _cat_str(var, "gene_name")
        # A still-numeric result means neither layout matched and we are about
        # to build a gene space out of integers.
        if np.issubdtype(np.asarray(out).dtype, np.number):
            raise RuntimeError(
                "var/gene_name decoded to numbers -- categorical layout not "
                "recognised; refusing to build a gene space from codes"
            )
        return np.asarray(out)
    idx = var.attrs.get("_index", "_index")
    return np.array(_decode(var[idx][:]))


def _part_ok(path, want_rows=None):
    """Is this part a COMPLETE h5ad, not just an existing file?

    Modal preempts containers. A preemption mid-write leaves a truncated h5ad
    that still exists, and resuming on `os.path.exists` alone then skips it
    forever -- which is exactly what happened to part_00029 (75.6 MB instead of
    291 MB, holding `X/data` and nothing else, no encoding attrs). The concat
    only noticed 70 parts later, and had the truncation landed somewhere
    readable it would have corrupted the dataset silently.
    """
    import os

    import h5py
    import numpy as np

    if not os.path.exists(path):
        return False
    try:
        with h5py.File(path, "r") as h:
            X = h["X"]
            if not isinstance(X, h5py.Group):
                return False
            for k in ("data", "indices", "indptr"):
                if k not in X:
                    return False
            shape = tuple(np.asarray(X.attrs["shape"]).tolist())
            if want_rows is not None and shape[0] != want_rows:
                return False
            # indptr is written last of the three, so a complete one is a good
            # proxy for a complete file.
            if X["indptr"].shape[0] != shape[0] + 1:
                return False
            if int(X["indptr"][-1]) != X["data"].shape[0]:
                return False
            return "target_gene" in h["obs"]
    except Exception:  # noqa: BLE001 - unreadable means not usable
        return False


def _concat(line, parts, out, t0):
    """Join the row-chunk parts into one h5ad, with h5py rather than anndata.

    `anndata.experimental.concat_on_disk` dies here with
    `AttributeError: 'dict' object has no attribute 'shape'` on anndata 0.13.4
    (eval_image), while the identical call works on 0.11.4 (cpu_image, which is
    what src/predict.py uses). Rather than pin a version for one call -- the
    README already records five failures from Arc-stack version drift -- do the
    concatenation explicitly. CSR row-concatenation is append `data` and
    `indices`, then offset `indptr`; nothing about it needs a library.

    This is only safe because every part was written with IDENTICAL categorical
    levels (see `pcats`/`bcats` above), so the `codes` arrays concatenate
    directly instead of needing remapping.
    """
    import os
    import shutil
    import time

    import h5py
    import numpy as np

    pfs = sorted(f"{parts}/{x}" for x in os.listdir(parts)
                 if x.endswith(".h5ad"))
    print(f"[{line}] concatenating {len(pfs)} parts "
          f"({sum(os.path.getsize(p) for p in pfs)/1e9:.2f} GB)", flush=True)

    # Validate every part BEFORE copying a byte. The first version checked
    # only X/data in this pass, so a part holding data and nothing else got
    # through and blew up 29 parts into the copy.
    bad = [p for p in pfs if not _part_ok(p)]
    if bad:
        raise RuntimeError(
            f"{len(bad)} incomplete parts, rerun to rewrite them: "
            + ", ".join(os.path.basename(b) for b in bad[:10])
        )

    n_rows = nnz = 0
    n_cols = None
    for p in pfs:
        with h5py.File(p, "r") as h:
            sh = tuple(np.asarray(h["X"].attrs["shape"]).tolist())
            n_rows += sh[0]
            n_cols = sh[1] if n_cols is None else n_cols
            if sh[1] != n_cols:
                raise RuntimeError(f"{p}: {sh[1]} columns, expected {n_cols}")
            nnz += h["X"]["data"].shape[0]
    print(f"[{line}] -> {n_rows:,} x {n_cols:,}, {nnz/1e9:.2f}G nonzeros "
          f"({nnz*8/1e9:.1f} GB uncompressed)", flush=True)
    if nnz >= 2**31:
        # indices stay int32 (column ids < n_cols) but indptr must hold the
        # cumulative count, which overflows int32 past 2.1e9 nonzeros.
        print(f"[{line}] nnz exceeds int32; indptr is int64")

    vlen = h5py.special_dtype(vlen=str)
    tmp = out + ".building"
    with h5py.File(tmp, "w") as o:
        o.attrs["encoding-type"] = "anndata"
        o.attrs["encoding-version"] = "0.1.0"

        X = o.create_group("X")
        X.attrs["encoding-type"] = "csr_matrix"
        X.attrs["encoding-version"] = "0.1.0"
        X.attrs["shape"] = np.array([n_rows, n_cols], dtype="int64")
        dd = X.create_dataset("data", (nnz,), dtype="float32",
                              chunks=(1 << 20,), compression="lzf")
        di = X.create_dataset("indices", (nnz,), dtype="int32",
                              chunks=(1 << 20,), compression="lzf")
        dp = X.create_dataset("indptr", (n_rows + 1,), dtype="int64")

        roff = coff = 0
        dp[0] = 0
        obs_codes = {}
        obs_cats = {}
        idx_names = []
        var_names = None
        for k, p in enumerate(pfs):
            with h5py.File(p, "r") as h:
                g = h["X"]
                m = tuple(np.asarray(g.attrs["shape"]).tolist())[0]
                cnt = g["data"].shape[0]
                dd[coff:coff + cnt] = g["data"][:]
                di[coff:coff + cnt] = g["indices"][:]
                # Row k of the output starts where row k of this part started,
                # shifted by every nonzero already written.
                #
                # .astype(int64) is load-bearing: the parts store indptr as
                # int32 (anndata picks the narrowest type that fits ONE part's
                # ~55M nonzeros), and the running offset passes 2^31 about 40%
                # of the way through the 5.5e9 total. Without the cast numpy
                # keeps int32 and raises OverflowError -- or, on other numpy
                # versions, wraps silently to negative offsets.
                dp[roff + 1:roff + m + 1] = \
                    g["indptr"][1:].astype(np.int64) + np.int64(coff)
                roff += m
                coff += cnt

                ob = h["obs"]
                ix = ob.attrs.get("_index", "_index")
                ix = ix.decode() if isinstance(ix, bytes) else str(ix)
                idx_names.append(_decode(ob[ix][:]))
                for col in ob:
                    if col == ix or col == "__categories":
                        continue
                    d = ob[col]
                    if isinstance(d, h5py.Group) and "codes" in d:
                        obs_codes.setdefault(col, []).append(d["codes"][:])
                        cats = _decode(d["categories"][:])
                        if col in obs_cats and obs_cats[col] != cats:
                            raise RuntimeError(
                                f"{p}: categories for '{col}' differ from "
                                f"earlier parts; codes cannot be concatenated"
                            )
                        obs_cats[col] = cats
                    else:
                        obs_codes.setdefault(col, []).append(d[:])
                if var_names is None:
                    vg = h["var"]
                    vix = vg.attrs.get("_index", "_index")
                    vix = vix.decode() if isinstance(vix, bytes) else str(vix)
                    var_names = _decode(vg[vix][:])
            if k % 20 == 0:
                print(f"  [{line}] part {k+1}/{len(pfs)}, "
                      f"{roff:,} rows, {(time.time()-t0)/60:.0f} min",
                      flush=True)
        if roff != n_rows or coff != nnz:
            raise RuntimeError(f"wrote {roff} rows / {coff} nnz, "
                               f"expected {n_rows} / {nnz}")

        obs = o.create_group("obs")
        obs.attrs["encoding-type"] = "dataframe"
        obs.attrs["encoding-version"] = "0.2.0"
        obs.attrs["_index"] = "_index"
        obs.create_dataset(
            "_index", data=np.concatenate(idx_names).astype(object), dtype=vlen)
        order = []
        for col, chunks in obs_codes.items():
            arr = np.concatenate(chunks)
            if col in obs_cats:
                cg = obs.create_group(col)
                cg.attrs["encoding-type"] = "categorical"
                cg.attrs["encoding-version"] = "0.2.0"
                cg.attrs["ordered"] = np.bool_(False)
                cg.create_dataset(
                    "categories",
                    data=np.array(obs_cats[col], dtype=object), dtype=vlen)
                cg.create_dataset("codes", data=arr.astype("int32"))
            else:
                obs.create_dataset(col, data=arr)
            order.append(col)
        # attrs.create with an explicit vlen dtype, not plain assignment:
        # h5py infers the string type from the VALUES, so an empty
        # object-dtype array raises "Object dtype has no native HDF5
        # equivalent". obs survives plain assignment only because it has
        # columns; var's list is empty and does not.
        obs.attrs.create("column-order", order, dtype=vlen)

        var = o.create_group("var")
        var.attrs["encoding-type"] = "dataframe"
        var.attrs["encoding-version"] = "0.2.0"
        var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=np.array(var_names, dtype=object),
                           dtype=vlen)
        var.attrs.create("column-order", [], dtype=vlen)

    os.replace(tmp, out)
    VOL.commit()
    print(f"[{line}] wrote {out} ({os.path.getsize(out)/1e9:.2f} GB) in "
          f"{(time.time()-t0)/60:.0f} min")

    # Read it back through anndata before dropping the parts -- the parts are
    # the only copy, and a file that h5py wrote happily can still be one
    # anndata refuses.
    import anndata as ad

    A = ad.read_h5ad(out, backed="r")
    print(f"[{line}] verified: {A.shape}, obs {list(A.obs.columns)}, "
          f"{A.obs['target_gene'].nunique()} perturbations")
    A.file.close()
    shutil.rmtree(parts)
    VOL.commit()
    print(f"[{line}] removed {parts}")


@app.function(
    image=eval_image, volumes={VOL_PATH: VOL}, cpu=2.0, memory=8192,
    timeout=60 * 15,
)
def peek(path: str = ""):
    """Dump one h5ad's actual structure. Written after two failures caused by
    assuming a layout instead of reading it."""
    import os

    import h5py

    if path == "scan":
        # Which parts do NOT have the layout the concat assumes? Pass 1 of
        # _concat only touches X/data, so a part missing X/indices gets all the
        # way to the copy before failing.
        import numpy as np
        d = f"{OUTDIR}/_parts_K562"
        bad = []
        for n in sorted(os.listdir(d)):
            if not n.endswith(".h5ad"):
                continue
            fp = f"{d}/{n}"
            try:
                with h5py.File(fp, "r") as h:
                    X = h["X"]
                    keys = sorted(X.keys()) if isinstance(X, h5py.Group) \
                        else f"DATASET {X.shape}"
                    enc = X.attrs.get("encoding-type", b"?")
                    enc = enc.decode() if isinstance(enc, bytes) else enc
                    if not isinstance(X, h5py.Group) or \
                            "indices" not in X or "indptr" not in X:
                        bad.append((n, enc, keys,
                                    os.path.getsize(fp)))
            except Exception as e:  # noqa: BLE001
                bad.append((n, f"{type(e).__name__}: {e}", None,
                            os.path.getsize(fp)))
        print(f"scanned; {len(bad)} problem parts")
        for n, enc, keys, sz in bad:
            print(f"  {n}  {sz/1e6:.1f} MB  enc={enc}  keys={keys}")
        return bad

    p = path or f"{OUTDIR}/_parts_K562/part_00000.h5ad"
    if not p.startswith("/"):
        p = f"{VOL_PATH}/{p}"
    print(f"{p} ({os.path.getsize(p)/1e6:.1f} MB)")

    def show(name, obj):
        kind = "G" if isinstance(obj, h5py.Group) else \
            f"D {obj.shape} {obj.dtype}"
        attrs = {k: (v.decode() if isinstance(v, bytes) else
                     (v.tolist() if hasattr(v, "tolist") else v))
                 for k, v in obj.attrs.items()}
        print(f"  {name:<40s} {kind:<28s} {attrs}")

    with h5py.File(p, "r") as h:
        print(f"  ROOT attrs: {dict(h.attrs)}")
        h.visititems(show)
    return p


def _panel_genes():
    """Every gene some VCC panel scores -- 2025's and the 2026 final panel's."""
    import csv

    out = set()
    with open(f"{CONTROLS_FINAL}/gene_names.csv", newline="") as fh:
        rd = csv.reader(fh)
        for row in rd:
            if row and row[0] not in ("gene_name", "gene", ""):
                out.add(row[0])
    with open(f"{REHEARSAL}/gene_names.csv", newline="") as fh:
        rd = csv.reader(fh)
        for row in rd:
            if row and row[0] not in ("gene_name", "gene", ""):
                out.add(row[0])
    return out


@app.function(
    image=eval_image, volumes={VOL_PATH: VOL},
    cpu=8.0, memory=32768, timeout=60 * 60 * 10,
)
def convert(plan_only: bool = False, only: str = ""):
    import json
    import os
    import time

    import anndata as ad
    import h5py
    import numpy as np
    import scipy.sparse as sp

    os.makedirs(OUTDIR, exist_ok=True)

    # ---------------- the shared gene space ----------------
    panel = _panel_genes()
    print(f"VCC panels score {len(panel)} distinct genes")

    per_file = {}
    for line, fn in FILES.items():
        p = f"{REPLOGLE}/{fn}"
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} -- run src/replogle_fetch.py first")
        with h5py.File(p, "r") as h:
            names = _var_names(h)
            X = h["X"]
            shape = X.shape if isinstance(X, h5py.Dataset) else \
                tuple(X.attrs["shape"])
            dense = isinstance(X, h5py.Dataset)
            cols = sorted(h["obs"].keys())
        per_file[line] = {"path": p, "genes": names, "shape": shape,
                          "dense": dense, "obs": cols}
        print(f"{line}: {shape[0]:,} cells x {shape[1]:,} genes, "
              f"{'DENSE' if dense else 'sparse'}")
        print(f"  obs: {cols}")

    shared = set(per_file[next(iter(FILES))]["genes"])
    for v in per_file.values():
        shared &= set(v["genes"])
    graded = sorted(shared & panel)
    print(f"\ngene space: {len(shared)} shared across the screens, "
          f"{len(graded)} of those scored by a VCC panel")
    if len(graded) < 4000:
        raise RuntimeError(f"only {len(graded)} usable genes -- check symbols")

    for line, v in per_file.items():
        perts = None
        with h5py.File(v["path"], "r") as h:
            key = "gene" if "gene" in h["obs"] else (
                "gene_transcript" if "gene_transcript" in h["obs"] else None)
            if key is None:
                raise KeyError(f"{line}: no perturbation column in {v['obs']}")
            v["pert_key"] = key
            g = _obs_str(h, key)
            # Replogle labels a row "ENSG..._SYMBOL" in some releases and bare
            # symbols in others; reduce to the symbol either way.
            g = np.array([s.split("_")[-1] if "_" in s else s for s in g])
            perts = sorted(set(g) - {CONTROL_PERT, ""})
            v["n_control"] = int((g == CONTROL_PERT).sum())
        v["perts"] = perts
        print(f"{line}: {len(perts)} perturbations, "
              f"{v['n_control']:,} control cells, pert col '{v['pert_key']}'")

    if plan_only:
        est = sum(v["shape"][0] for v in per_file.values()) * len(graded)
        print(f"\nwould write {est/1e9:.1f}G values as sparse CSR to {OUTDIR}")
        print(f"gene space sample: {graded[:8]}")
        return None

    # ---------------- stream each file ----------------
    want = [l for l in FILES if not only or l in only.split(",")]
    for line in want:
        v = per_file[line]
        out = f"{OUTDIR}/{line}.h5ad"
        if os.path.exists(out):
            print(f"[{line}] {out} exists "
                  f"({os.path.getsize(out)/1e9:.2f} GB) -- skipping")
            continue
        t0 = time.time()
        gi = {g: i for i, g in enumerate(v["genes"])}
        take = np.array([gi[g] for g in graded])
        n = v["shape"][0]
        print(f"\n[{line}] {n:,} cells -> {len(graded)} genes", flush=True)

        import pandas as pd

        # Accumulating K562 in RAM is not an option: ~2M cells x ~2,000
        # nonzeros is ~4e9 nonzeros, ~32 GB as CSR, and sp.vstack needs a
        # second copy of it. So each row-chunk becomes its own small h5ad and
        # `concat_on_disk` joins them -- the same memory-bounded path
        # src/predict.py already uses for the D/E/F contexts. The parts are
        # also the resume points: a preempted container keeps what it wrote.
        parts = f"{OUTDIR}/_parts_{line}"
        os.makedirs(parts, exist_ok=True)

        with h5py.File(v["path"], "r") as h:
            pert = _obs_str(h, v["pert_key"])
            pert = np.array([s.split("_")[-1] if "_" in s else s
                             for s in pert])
            batch = (_obs_str(h, "gem_group") if "gem_group" in h["obs"]
                     else np.array(["1"] * n))
            # Categories fixed up front, so every part shares them and
            # concat_on_disk does not have to reconcile differing level sets.
            pcats = sorted(set(pert))
            bcats = sorted({str(b) for b in batch})
            X = h["X"]
            nnz = 0
            for ci, s in enumerate(range(0, n, ROW_CHUNK)):
                e = min(s + ROW_CHUNK, n)
                pf = f"{parts}/part_{ci:05d}.h5ad"
                if _part_ok(pf, e - s):
                    continue
                if os.path.exists(pf):
                    print(f"  [{line}] part_{ci:05d} exists but is "
                          f"incomplete ({os.path.getsize(pf)/1e6:.1f} MB) "
                          f"-- rewriting", flush=True)
                    os.remove(pf)
                D = np.asarray(X[s:e, :], dtype=np.float32)[:, take]
                # CP10k then log1p, both per row: zeros stay zeros, so the
                # result is as sparse as the counts were.
                tot = D.sum(axis=1, keepdims=True)
                np.divide(D, np.maximum(tot, 1e-9), out=D)
                D *= DEPTH
                np.log1p(D, out=D)
                Xc = sp.csr_matrix(D)
                del D
                nnz += Xc.nnz
                obs = pd.DataFrame(
                    {
                        "target_gene": pd.Categorical(pert[s:e],
                                                      categories=pcats),
                        "batch": pd.Categorical(
                            [str(b) for b in batch[s:e]], categories=bcats),
                        "cell_line": pd.Categorical([line] * (e - s),
                                                    categories=[line]),
                    },
                    index=[f"{line}_{i}" for i in range(s, e)],
                )
                ad.AnnData(X=Xc, obs=obs,
                           var=pd.DataFrame(index=graded)).write_h5ad(
                    pf, compression="lzf")
                del Xc
                if ci % 10 == 0:
                    el = time.time() - t0
                    VOL.commit()
                    print(f"  [{line}] {e:,}/{n:,} ({e/n*100:.0f}%) "
                          f"{el/60:.0f} min, {nnz/1e9:.2f}G nonzeros so far",
                          flush=True)

        VOL.commit()
        _concat(line, parts, out, t0)

    # ---------------- the cell-load TOML ----------------
    os.makedirs(STATE_ROOT, exist_ok=True)
    meta = {l: {"perts": per_file[l]["perts"], "cells": per_file[l]["shape"][0]}
            for l in FILES}
    with open(f"{OUTDIR}/meta.json", "w") as fh:
        json.dump({"genes": graded, "lines": meta}, fh)

    # Two different holdouts, because they answer different questions and one
    # of them must stay clean.
    #
    #   RPE1 (zeroshot, test) -- a held-out CELL LINE. This is the challenge's
    #     own setup and the only honest read on cross-context generalisation.
    #     It must never influence training, which includes checkpoint
    #     selection: validating on RPE1 would pick the checkpoint that happens
    #     to suit RPE1 and then report RPE1 as evidence.
    #   K562 perturbations (fewshot, val) -- held-out PERTURBATIONS inside the
    #     training line, which is what `best.ckpt` gets chosen on.
    rng = __import__("numpy").random.default_rng(0)
    kperts = per_file["K562"]["perts"]
    n_val = min(300, max(50, len(kperts) // 20))
    val_perts = sorted(rng.choice(kperts, n_val, replace=False).tolist())
    print(f"val split: {len(val_perts)} of {len(kperts)} K562 perturbations "
          f"held out for checkpoint selection")

    def fmt(names):
        return "[\n" + "".join(f'  "{n}",\n' for n in names) + "]"

    toml = f'''# Generated by src/replogle_prep.py -- do not hand-edit.
#
# Zeroshot on RPE1: train on K562 only, and RPE1's {per_file["RPE1"]["shape"][0]:,} cells are
# never seen -- not for training and not for checkpoint selection. That is the
# challenge's own setup: a held-out CONTEXT, not a held-out perturbation.
#
# `best.ckpt` is chosen on {len(val_perts)} held-out K562 perturbations instead, so the
# RPE1 number stays a genuine out-of-sample read.

[datasets]
replogle = "{OUTDIR}"

[training]
replogle = "train"

[zeroshot]
"replogle.RPE1" = "test"

[fewshot]

[fewshot."replogle.K562"]
val = {fmt(val_perts)}
'''
    with open(TOML_PATH, "w") as fh:
        fh.write(toml)
    VOL.commit()
    print(f"\nwrote {TOML_PATH}")
    print(f"{OUTDIR} holds: {sorted(os.listdir(OUTDIR))}")
    return TOML_PATH


@app.local_entrypoint()
def main(plan_only: bool = False, only: str = "", do_peek: bool = False,
         path: str = ""):
    if do_peek:
        peek.remote(path)
        return
    convert.remote(plan_only, only)
