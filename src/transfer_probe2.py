"""Where does the difficulty live: cell-line biology, or measurement noise?

Three correlations over the same perturbations and the same readout genes:

  ceiling   K562-genomewide vs K562-essential  -- same cell line, independent
            screen. Any cross-line method is capped by this.
  crossline K562-genomewide vs RPE1            -- what transfer actually gets.
  null      non-targeting vs non-targeting     -- zero by construction.

If crossline is close to ceiling, cell line barely matters and the enemy is
noise. If ceiling is far above crossline, the cell line genuinely dominates
and a learned correction is the right tool.

Usage:  python -I src/transfer_probe2.py
"""

import os
import re
import sys

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BULK_DIR = os.path.join(REPO, "external", "replogle_bulk")
CONTROLS = os.path.join(REPO, "data", "vcc_2026_controls")

DEPTH = 1e4
EPS = 1e-3
MIN_EXPR = 0.05          # 5 CPM at DEPTH=1e4, matching the VCC DE filter
FILES = {
    "K562gw": "K562_gwps_raw_bulk_01.h5ad",
    "K562ess": "K562_essential_raw_bulk_01.h5ad",
    "RPE1": "rpe1_raw_bulk_01.h5ad",
}


def _decode(arr):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in arr])


def legacy_str(side, key):
    d = side[key]
    if d.dtype.kind in "iu" and "__categories" in side and key in side["__categories"]:
        cats = _decode(side["__categories"][key][:])
        return np.array([cats[c] if c >= 0 else "" for c in d[:]])
    return _decode(d[:])


class Screen:
    def __init__(self, fn):
        with h5py.File(os.path.join(BULK_DIR, fn), "r") as h:
            X = h["X"][:].astype(np.float64)
            labels = legacy_str(h["obs"], "gene_transcript")
            self.genes = legacy_str(h["var"], "gene_name")
            self.ncells = h["obs"]["num_cells_filtered"][:]
            self.fold = h["obs"]["fold_expr"][:]
        self.sym = np.array([l.split("_")[1] if "_" in l else l for l in labels])
        self.ctrl_mask = np.array([bool(re.search("non-targeting", l))
                                   for l in labels])
        self.X = X / X.sum(axis=1, keepdims=True) * DEPTH
        self.ctrl = self.X[self.ctrl_mask].mean(axis=0)
        self.lfc = np.log2((self.X + EPS) / (self.ctrl + EPS))
        self.gidx = {g: i for i, g in enumerate(self.genes)}
        self.pidx = {s: i for i, s in enumerate(self.sym) if not self.ctrl_mask[i]}

    def cols(self, genes):
        return np.array([self.gidx[g] for g in genes])


def corr_rows(A, B):
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    num = (A * B).sum(axis=1)
    den = np.sqrt((A * A).sum(axis=1) * (B * B).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def null_rows(s, cols, n=80):
    """log2FC of each individual NTC row against the mean of the others."""
    idx = np.where(s.ctrl_mask)[0]
    out = []
    for i in idx[:n]:
        ctrl = s.X[np.setdiff1d(idx, [i])].mean(axis=0)
        out.append(np.log2((s.X[i] + EPS) / (ctrl + EPS))[cols])
    return np.array(out)


def main():
    scr = {}
    for k, fn in FILES.items():
        print(f"loading {k} ...")
        scr[k] = Screen(fn)

    with open(os.path.join(CONTROLS, "gene_names.csv")) as fh:
        vcc_genes = {ln.strip() for ln in fh if ln.strip()}
    vcc_genes.discard("gene_name")

    print()
    for k, s in scr.items():
        n = s.ncells
        print(f"{k:<8s} {len(s.sym):>6d} rows ({s.ctrl_mask.sum():>3d} NTC)  "
              f"{len(s.genes):>5d} genes  cells/pert median "
              f"{np.nanmedian(n):.0f} (q10 {np.nanquantile(n, .1):.0f}, "
              f"q90 {np.nanquantile(n, .9):.0f})  "
              f"on-target fold_expr median {np.nanmedian(s.fold):.3f}")

    # Readout genes: present everywhere, scored by VCC, and expressed enough
    # that a log-ratio is meaningful in every screen.
    genes = set(vcc_genes)
    for s in scr.values():
        genes &= set(s.genes)
    genes = sorted(genes)
    keep = np.ones(len(genes), bool)
    for s in scr.values():
        keep &= s.ctrl[s.cols(genes)] > MIN_EXPR
    genes_all = genes
    genes = [g for g, k in zip(genes, keep) if k]
    print(f"\nreadout genes: shared & VCC-scored {len(genes_all)}, "
          f"of those >{MIN_EXPR*100:.0f} CPM everywhere: {len(genes)}")

    cols = {k: s.cols(genes) for k, s in scr.items()}
    nulls = {k: null_rows(s, cols[k]) for k, s in scr.items()}
    for k, nz in nulls.items():
        print(f"  {k:<8s} null |log2FC| median {np.median(np.abs(nz)):.4f}")

    pairs = [("ceiling   K562gw vs K562ess", "K562gw", "K562ess"),
             ("crossline K562gw vs RPE1   ", "K562gw", "RPE1"),
             ("crossline K562ess vs RPE1  ", "K562ess", "RPE1")]

    print()
    for name, a, b in pairs:
        sa, sb = scr[a], scr[b]
        shared = sorted(set(sa.pidx) & set(sb.pidx))
        A = sa.lfc[np.array([sa.pidx[s] for s in shared])][:, cols[a]]
        B = sb.lfc[np.array([sb.pidx[s] for s in shared])][:, cols[b]]
        r = corr_rows(A, B)

        # Strength relative to that screen's own null band.
        na = np.median(np.abs(nulls[a]))
        nb = np.median(np.abs(nulls[b]))
        ma = np.abs(A).mean(axis=1) / na
        mb = np.abs(B).mean(axis=1) / nb

        print(f"{name}  n={len(shared):>5d}  median r = {np.nanmedian(r):+.3f}")
        for thr in (1.5, 2.0, 3.0, 5.0):
            sel = (ma > thr) & (mb > thr)
            if sel.sum() >= 10:
                print(f"    both >{thr}x null  n={sel.sum():>5d}  "
                      f"r = {np.nanmedian(r[sel]):+.3f}")
        # Cells per perturbation drives the noise; split on it.
        ca = np.nan_to_num(sa.ncells[[sa.pidx[s] for s in shared]])
        cb = np.nan_to_num(sb.ncells[[sb.pidx[s] for s in shared]])
        deep = (ca > np.nanmedian(ca)) & (cb > np.nanmedian(cb))
        print(f"    deep-sequenced half n={deep.sum():>5d}  "
              f"r = {np.nanmedian(r[deep]):+.3f}")
        strong_deep = deep & (ma > 2) & (mb > 2)
        if strong_deep.sum() >= 10:
            print(f"    deep AND strong     n={strong_deep.sum():>5d}  "
                  f"r = {np.nanmedian(r[strong_deep]):+.3f}")
        print()


if __name__ == "__main__":
    sys.exit(main())
