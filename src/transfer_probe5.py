"""Which regime are the 300 VCC targets actually in?

Probe 4's two terciles pointed opposite ways:
  strong effect  -> lam 0.0  (common axis is the whole answer)
  many cells     -> lam 0.3  (perturbation-specific signal is worth using)

Those are not in conflict. In a CRISPRi screen, few surviving cells means the
knockdown arrested growth, i.e. an essential gene, whose response is dominated
by the shared stress axis. Many surviving cells means the knockdown was well
tolerated -- which is exactly how Arc chose the 300 VCC targets.

So the question is: do the 272 covered VCC targets sit in the easy regime or
the hard one, and what lam applies THERE? This matches shared perturbations to
the VCC targets' own cell-count / effect-size profile and sweeps lam on that
matched subset only.

Usage:  python -I src/transfer_probe5.py
"""

import csv
import os
import re
import sys

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BULK_DIR = os.path.join(REPO, "external", "replogle_bulk")
CONTROLS = os.path.join(REPO, "data", "vcc_2026_controls")
DEPTH, EPS, MIN_EXPR = 1e4, 1e-3, 0.05


def _decode(a):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in a])


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
            self.ncells = np.nan_to_num(h["obs"]["num_cells_filtered"][:])
            self.fold = h["obs"]["fold_expr"][:]
        self.sym = np.array([l.split("_")[1] if "_" in l else l for l in labels])
        self.ctrl_mask = np.array([bool(re.search("non-targeting", l))
                                   for l in labels])
        self.X = X / X.sum(axis=1, keepdims=True) * DEPTH
        self.ctrl = self.X[self.ctrl_mask].mean(axis=0)
        self.lfc = np.log2((self.X + EPS) / (self.ctrl + EPS))
        self.gidx = {g: i for i, g in enumerate(self.genes)}
        self.pidx = {s: i for i, s in enumerate(self.sym) if not self.ctrl_mask[i]}

    def cols(self, g):
        return np.array([self.gidx[x] for x in g])


def corr_rows(A, B):
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    num = (A * B).sum(axis=1)
    den = np.sqrt((A * A).sum(axis=1) * (B * B).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def topk_sign(pred, true, k=100):
    out = np.empty(len(pred))
    for i in range(len(pred)):
        idx = np.argsort(-np.abs(pred[i]))[:k]
        out[i] = np.mean(np.sign(pred[i][idx]) == np.sign(true[i][idx]))
    return out


LAMS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


def sweep(A, B, loo, label):
    rows = []
    for lam in LAMS:
        P = loo + lam * (A - loo)
        rows.append((np.nanmedian(corr_rows(P, B)),
                     np.nanmedian(topk_sign(P, B))))
    best = int(np.argmax([x[0] for x in rows]))
    print(f"  {label:<28s} n={len(A):>5d}  "
          f"r: {' '.join(f'{x[0]:+.3f}' for x in rows)}")
    print(f"  {'':<28s}         "
          f"s: {' '.join(f'{x[1]:.3f}' for x in rows)}   "
          f"-> best lam {LAMS[best]:.1f} (r={rows[best][0]:+.3f}, "
          f"sign={rows[best][1]:.3f})")
    return LAMS[best], rows[best]


def main():
    k = Screen("K562_gwps_raw_bulk_01.h5ad")
    r = Screen("rpe1_raw_bulk_01.h5ad")

    with open(os.path.join(CONTROLS, "gene_names.csv")) as fh:
        vcc_genes = {l.strip() for l in fh if l.strip()}
    vcc_genes.discard("gene_name")
    with open(os.path.join(CONTROLS, "pert_counts.csv"), newline="") as fh:
        targets = sorted({row["target_gene"] for row in csv.DictReader(fh)})

    genes = sorted(set(k.genes) & set(r.genes) & vcc_genes)
    keep = (k.ctrl[k.cols(genes)] > MIN_EXPR) & (r.ctrl[r.cols(genes)] > MIN_EXPR)
    genes = [g for g, x in zip(genes, keep) if x]
    kc, rc = k.cols(genes), r.cols(genes)

    nullk = np.array([
        np.log2((k.X[i] + EPS) /
                (k.X[np.setdiff1d(np.where(k.ctrl_mask)[0], [i])].mean(axis=0)
                 + EPS))[kc]
        for i in np.where(k.ctrl_mask)[0][:80]])
    nmed = np.median(np.abs(nullk))

    # ---- profile of the 272 covered VCC targets inside K562 ----
    vcc_in_k = [t for t in targets if t in k.pidx]
    vrows = np.array([k.pidx[t] for t in vcc_in_k])
    allrows = np.array(sorted(k.pidx.values()))
    v_n, a_n = k.ncells[vrows], k.ncells[allrows]
    v_mag = np.abs(k.lfc[vrows][:, kc]).mean(axis=1) / nmed
    a_mag = np.abs(k.lfc[allrows][:, kc]).mean(axis=1) / nmed

    print(f"{len(genes)} readout genes\n")
    print("profile of the 272 covered VCC targets vs all 9,866 K562 perturbations")
    print(f"  cells/pert   VCC median {np.median(v_n):>6.0f}   "
          f"all median {np.median(a_n):>6.0f}")
    q = np.quantile(a_n, [1/3, 2/3])
    print(f"  VCC targets by all-pert cell-count tercile: "
          f"few {np.mean(v_n <= q[0]):.3f}  "
          f"mid {np.mean((v_n > q[0]) & (v_n <= q[1])):.3f}  "
          f"many {np.mean(v_n > q[1]):.3f}")
    print(f"  effect size   VCC median {np.median(v_mag):.2f}x   "
          f"all median {np.median(a_mag):.2f}x")
    qm = np.quantile(a_mag, [1/3, 2/3])
    print(f"  VCC targets by all-pert effect tercile:     "
          f"weak {np.mean(v_mag <= qm[0]):.3f}  "
          f"mid {np.mean((v_mag > qm[0]) & (v_mag <= qm[1])):.3f}  "
          f"strong {np.mean(v_mag > qm[1]):.3f}")

    # ---- lam sweep on a subset matched to that profile ----
    shared = sorted(set(k.pidx) & set(r.pidx))
    ki = np.array([k.pidx[s] for s in shared])
    A = k.lfc[ki][:, kc]
    B = r.lfc[np.array([r.pidx[s] for s in shared])][:, rc]
    n = len(shared)
    loo = (A.sum(axis=0)[None, :] - A) / (n - 1)
    sn, smag = k.ncells[ki], np.abs(A).mean(axis=1) / nmed

    print(f"\nlam sweep, scored on held-out RPE1   "
          f"lams = {' '.join(f'{l:.1f}' for l in LAMS)}")
    sweep(A, B, loo, "all shared")

    # Match on cell count: the VCC-like, well-tolerated regime.
    lo = np.quantile(v_n, 0.25)
    m = sn >= lo
    sweep(A[m], B[m], loo[m], f"cells >= VCC q25 ({lo:.0f})")

    m2 = sn >= np.median(v_n)
    sweep(A[m2], B[m2], loo[m2], f"cells >= VCC median ({np.median(v_n):.0f})")

    # Match on BOTH cell count and effect size (the full VCC-like profile).
    hi_m = np.quantile(v_mag, 0.75)
    m3 = (sn >= lo) & (smag <= hi_m)
    sweep(A[m3], B[m3], loo[m3], "cells high AND effect <= VCC q75")

    # The opposite regime, for contrast: the essential-gene core.
    m4 = sn <= np.quantile(sn, 1/3)
    sweep(A[m4], B[m4], loo[m4], "CONTRAST: few cells (essential)")

    # ---- is a VCC-target-specific common axis better than the global one? ----
    print("\ncommon axis estimated from different donor sets, scored on RPE1")
    A_vcc = k.lfc[vrows][:, kc]
    cands = {
        "global (all 9,866 K562 perts)": k.lfc[allrows][:, kc].mean(axis=0),
        "VCC 272 targets only": A_vcc.mean(axis=0),
        "high-cell-count perts only": k.lfc[allrows][:, kc][
            a_n > np.quantile(a_n, 2/3)].mean(axis=0),
    }
    for name, c in cands.items():
        P = np.repeat(c[None, :], m.sum(), axis=0)
        print(f"  {name:<34s} r={np.nanmedian(corr_rows(P, B[m])):+.3f}  "
              f"sign@100={np.nanmedian(topk_sign(P, B[m])):.3f}  "
              f"(on the VCC-like subset, n={m.sum()})")


if __name__ == "__main__":
    sys.exit(main())
