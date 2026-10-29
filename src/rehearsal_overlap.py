"""How usable is Arc's 2025 H1 data as a stand-in for the 2026 D/E/F task?

Three overlaps decide it:
  genes    2025 readout space vs the 18,533 genes D/E/F are scored on
  targets  2025's 150 perturbations vs what K562 gives us a lookup for
  depth    cells per perturbation, which sets how noisy the ground truth is

Usage:  python -I src/rehearsal_overlap.py
"""

import csv
import os
import re
import sys

import h5py
import numpy as np

REPO = r"C:\Coding\virtual-cell"
V25 = os.path.join(REPO, "external", "vcc2025")
BULK = os.path.join(REPO, "external", "replogle_bulk")
FIN = os.path.join(REPO, "data", "vcc_2026_controls_final")
DEPTH, EPS, MIN_EXPR = 1e4, 1e-3, 0.05


def _decode(a):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in a])


def legacy_str(side, key):
    d = side[key]
    if d.dtype.kind in "iu" and "__categories" in side and key in side["__categories"]:
        cats = _decode(side["__categories"][key][:])
        return np.array([cats[c] if c >= 0 else "" for c in d[:]])
    return _decode(d[:])


def plain_list(path):
    with open(path) as fh:
        rows = [l.strip() for l in fh if l.strip()]
    if rows and rows[0].lower() in ("gene_name", "gene", "genes", "target_gene"):
        rows = rows[1:]
    return rows


def main():
    g25 = plain_list(os.path.join(V25, "gene_names.csv"))
    g26 = plain_list(os.path.join(FIN, "gene_names.csv"))

    with open(os.path.join(V25, "pert_counts_Training.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    p25 = [r["target_gene"] for r in rows]
    ncells = np.array([float(r["n_cells"]) for r in rows])
    umi = np.array([float(r["median_umi_per_cell"]) for r in rows])

    with open(os.path.join(FIN, "pert_counts.csv"), newline="") as fh:
        t26 = sorted({r["target_gene"] for r in csv.DictReader(fh)})

    print(f"2025 genes {len(g25)}   2026 genes {len(g26)}   "
          f"shared {len(set(g25) & set(g26))}")
    print(f"2025 perts {len(p25)}   2026 targets {len(t26)}   "
          f"shared {len(set(p25) & set(t26))}")
    print(f"2025 cells/pert: median {np.median(ncells):.0f}  "
          f"min {ncells.min():.0f}  max {ncells.max():.0f}  "
          f"total {ncells.sum():.0f}")
    print(f"2025 median UMI/cell: {np.median(umi):.0f}  "
          f"(2026 contexts: ~20,000)")

    # K562 lookup coverage of the 2025 perturbation list.
    with h5py.File(os.path.join(BULK, "K562_gwps_raw_bulk_01.h5ad"), "r") as h:
        X = h["X"][:].astype(np.float64)
        labels = legacy_str(h["obs"], "gene_transcript")
        kgenes = legacy_str(h["var"], "gene_name")
    sym = np.array([l.split("_")[1] if "_" in l else l for l in labels])
    ctrl_mask = np.array([bool(re.search("non-targeting", l)) for l in labels])
    X = X / X.sum(axis=1, keepdims=True) * DEPTH
    ctrl = X[ctrl_mask].mean(axis=0)
    pidx = {s: i for i, s in enumerate(sym) if not ctrl_mask[i]}

    cov25 = [p for p in p25 if p in pidx]
    cov26 = [t for t in t26 if t in pidx]
    print(f"\nK562 genome-wide lookup covers:")
    print(f"  2025 perts  {len(cov25)}/{len(p25)} "
          f"({len(cov25)/len(p25)*100:.1f}%)")
    print(f"  2026 targets {len(cov26)}/{len(t26)} "
          f"({len(cov26)/len(t26)*100:.1f}%)")

    # Readout genes we can actually predict, in each gene space.
    gi = {g: i for i, g in enumerate(kgenes)}
    for name, gl in (("2025", g25), ("2026", g26)):
        shared = [g for g in gl if g in gi and ctrl[gi[g]] > MIN_EXPR]
        print(f"  K562 readout usable in {name} space: {len(shared)} "
              f"of {len(gl)} ({len(shared)/len(gl)*100:.1f}%)")

    miss = sorted(set(p25) - set(pidx))
    print(f"\n2025 perts with no K562 lookup ({len(miss)}): "
          f"{', '.join(miss[:20])}{' ...' if len(miss) > 20 else ''}")

    # Is the 2025 panel's difficulty comparable to the 2026 panel's?
    cols = np.array([gi[g] for g in g26 if g in gi and ctrl[gi[g]] > MIN_EXPR])
    lfc = np.log2((X + EPS) / (ctrl + EPS))
    nullr = np.array([
        np.log2((X[i] + EPS) /
                (X[np.setdiff1d(np.where(ctrl_mask)[0], [i])].mean(axis=0)
                 + EPS))[cols]
        for i in np.where(ctrl_mask)[0][:60]])
    nmed = np.median(np.abs(nullr))
    for name, cov in (("2025 perts", cov25), ("2026 targets", cov26)):
        rws = np.array([pidx[p] for p in cov])
        mag = np.abs(lfc[rws][:, cols]).mean(axis=1) / nmed
        print(f"  K562 effect size for {name}: median {np.median(mag):.2f}x "
              f"null, frac >2x {np.mean(mag > 2):.3f}")


if __name__ == "__main__":
    sys.exit(main())
