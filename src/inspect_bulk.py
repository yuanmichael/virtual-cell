"""Inspect the Replogle pseudobulk files: structure, control rows, value scale.

Everything the transfer model needs has to come out of these three files, so
establish what they actually contain before writing any modelling code.

Usage:  python -I src/inspect_bulk.py
"""

import os
import re
import sys

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BULK_DIR = os.path.join(REPO, "external", "replogle_bulk")


def _decode(arr):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in arr])


def h5_strings(grp, side=None):
    """Decode an h5ad obs/var field.

    Handles three encodings: plain string dataset, modern categorical
    (a group with categories/codes), and the legacy anndata<0.7 layout where
    codes live in the field and categories in a sibling `__categories` group.
    """
    if isinstance(grp, h5py.Dataset):
        if grp.dtype.kind in "iu" and side is not None and "__categories" in side:
            name = grp.name.rsplit("/", 1)[-1]
            if name in side["__categories"]:
                cats = _decode(side["__categories"][name][:])
                return np.array([cats[c] if c >= 0 else "" for c in grp[:]])
        if grp.dtype.kind in "iuf":
            raise TypeError(f"numeric {grp.name}")
        return _decode(grp[:])
    if isinstance(grp, h5py.Group) and "categories" in grp:
        cats = _decode(grp["categories"][:])
        return np.array([cats[c] if c >= 0 else "" for c in grp["codes"][:]])
    raise TypeError(f"cannot decode {grp}")


def describe(path):
    print("=" * 78)
    print(os.path.basename(path))
    print("=" * 78)
    with h5py.File(path, "r") as h:
        print("top-level keys:", sorted(h.keys()))
        print("attrs:", dict(h.attrs))

        X = h["X"]
        if isinstance(X, h5py.Dataset):
            print(f"X: dense {X.shape} {X.dtype}")
            block = X[: min(50, X.shape[0]), :]
        else:
            print(f"X: sparse {dict(X.attrs)} data={X['data'].shape} {X['data'].dtype}")
            block = None

        for side in ("obs", "var"):
            g = h[side]
            idx_key = g.attrs.get("_index", "_index")
            print(f"\n{side}: index={idx_key!r} columns={sorted(k for k in g.keys())}")
            for k in list(g.keys()):
                if k == "__categories":
                    continue
                try:
                    vals = h5_strings(g[k], side=g)
                except TypeError:
                    d = g[k]
                    print(f"    {k:<24s} numeric {d.shape} {d.dtype} "
                          f"e.g. {np.asarray(d[:3])}")
                    continue
                uniq = len(set(vals))
                print(f"    {k:<24s} str n={len(vals)} uniq={uniq} "
                      f"e.g. {list(vals[:3])}")

        if "layers" in h:
            print("\nlayers:", sorted(h["layers"].keys()))
        if "obsm" in h:
            print("obsm:", sorted(h["obsm"].keys()))

        # Value scale: are these raw counts, CPM, or log?
        if block is not None:
            finite = block[np.isfinite(block)]
            print(f"\nX sample: min={finite.min():.4g} max={finite.max():.4g} "
                  f"mean={finite.mean():.4g} frac_zero={np.mean(finite == 0):.3f}")
            print(f"  row sums (first 5): {block.sum(axis=1)[:5]}")
            print(f"  any negative: {bool((finite < 0).any())}")

        # Which obs rows look like non-targeting controls?
        obs = h["obs"]
        for k in list(obs.keys()):
            if k == "__categories":
                continue
            try:
                vals = h5_strings(obs[k], side=obs)
            except TypeError:
                continue
            ctrl = [v for v in set(vals)
                    if re.search(r"non.?target|ntc|control|^ctrl", v, re.I)]
            if ctrl:
                n = sum(v in set(ctrl) for v in vals)
                print(f"\ncontrol-ish labels in obs[{k!r}]: {ctrl[:5]} "
                      f"({len(ctrl)} distinct, {n} rows)")
    print()


def main():
    for fn in sorted(os.listdir(BULK_DIR)):
        if fn.endswith(".h5ad"):
            describe(os.path.join(BULK_DIR, fn))


if __name__ == "__main__":
    sys.exit(main())
