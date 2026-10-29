"""Precompute the K562 common axis and per-target lookups once, locally.

Both the rehearsal (2025 H1 gene space) and the real submission (2026 D/E/F
gene space) need the same two things out of the Replogle K562 genome-wide
screen: the mean response over all perturbations, and each target's own
response. Computing them here keeps the 375 MB screen off the Volume -- the
output is a few MB.

Writes external/k562_axis.npz with log2FC in K562's own readout space,
restricted to genes expressed well enough for a log-ratio to mean anything.

Usage:  python -I src/build_axis.py
"""

import csv
import os
import re
import sys

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BULK = os.path.join(REPO, "external", "replogle_bulk")
V25 = os.path.join(REPO, "external", "vcc2025")
FIN = os.path.join(REPO, "data", "vcc_2026_controls_final")
OUT = os.path.join(REPO, "external", "k562_axis.npz")

DEPTH = 1e4      # rows are already ~depth-normalised; fix it exactly
EPS = 1e-3       # ~1000x below the median gene, so log-ratios stay finite
MIN_EXPR = 0.05  # 5 CPM at DEPTH, matching the VCC DE-eligibility filter


def _decode(a):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in a])


def legacy_str(side, key):
    """Decode a string column in the anndata<0.7 `__categories` layout."""
    d = side[key]
    if d.dtype.kind in "iu" and "__categories" in side and key in side["__categories"]:
        cats = _decode(side["__categories"][key][:])
        return np.array([cats[c] if c >= 0 else "" for c in d[:]])
    return _decode(d[:])


def plain_list(path):
    """One name per line, tolerating a header row. Not for multi-column CSVs."""
    with open(path) as fh:
        rows = [l.strip() for l in fh if l.strip()]
    if rows and rows[0].lower() in ("gene_name", "gene", "genes", "target_gene"):
        rows = rows[1:]
    return rows


def target_col(path):
    """Target genes from a CSV that has a `target_gene` column among others."""
    with open(path, newline="") as fh:
        return [r["target_gene"] for r in csv.DictReader(fh)]


def main():
    with h5py.File(os.path.join(BULK, "K562_gwps_raw_bulk_01.h5ad"), "r") as h:
        X = h["X"][:].astype(np.float64)
        labels = legacy_str(h["obs"], "gene_transcript")
        genes = legacy_str(h["var"], "gene_name")
        ncells = np.nan_to_num(h["obs"]["num_cells_filtered"][:])
        fold = h["obs"]["fold_expr"][:]

    sym = np.array([l.split("_")[1] if "_" in l else l for l in labels])
    is_ctrl = np.array([bool(re.search("non-targeting", l)) for l in labels])
    X = X / X.sum(axis=1, keepdims=True) * DEPTH
    ctrl = X[is_ctrl].mean(axis=0)
    lfc = np.log2((X + EPS) / (ctrl + EPS))

    # Keep genes where a log-ratio is meaningful, and that some VCC panel scores.
    panel_genes = set(plain_list(os.path.join(V25, "gene_names.csv")))
    panel_genes |= set(plain_list(os.path.join(FIN, "gene_names.csv")))
    keep = np.array([g in panel_genes and c > MIN_EXPR
                     for g, c in zip(genes, ctrl)])
    gcols = np.where(keep)[0]
    readout = genes[gcols]
    print(f"K562 readout genes kept: {len(readout)} of {len(genes)} "
          f"(expressed >{MIN_EXPR*100:.0f} CPM and scored by some VCC panel)")

    pidx = {s: i for i, s in enumerate(sym) if not is_ctrl[i]}
    prows = np.array(sorted(pidx.values()))

    # The common axis: average over EVERY perturbation, which is what makes it
    # a clean estimate of the shared response rather than one noisy row.
    common = lfc[prows][:, gcols].mean(axis=0)
    print(f"common axis over {len(prows)} perturbations, "
          f"mean |log2FC| {np.abs(common).mean():.4f}")

    # Null band from the non-targeting rows, for judging effect sizes later.
    cidx = np.where(is_ctrl)[0]
    null = np.array([
        np.log2((X[i] + EPS) / (X[np.setdiff1d(cidx, [i])].mean(axis=0) + EPS))[gcols]
        for i in cidx[:80]])
    print(f"null band: mean |log2FC| {np.abs(null).mean():.4f}, "
          f"median {np.median(np.abs(null)):.4f}")

    # Lookups for every target either panel asks about.
    want = set(target_col(os.path.join(V25, "pert_counts_Training.csv")))
    want |= set(target_col(os.path.join(FIN, "pert_counts.csv")))
    want.discard("non-targeting")
    have = sorted(w for w in want if w in pidx)
    rows = np.array([pidx[w] for w in have])
    lookup = lfc[rows][:, gcols]
    print(f"lookups: {len(have)} of {len(want)} requested targets found "
          f"in the K562 screen")

    np.savez_compressed(
        OUT,
        readout=readout,
        common=common.astype(np.float32),
        targets=np.array(have),
        lookup=lookup.astype(np.float32),
        lookup_ncells=ncells[rows].astype(np.float32),
        lookup_fold=np.nan_to_num(fold[rows]).astype(np.float32),
        null_sd=null.std(axis=0).astype(np.float32),
        null_med_abs=np.float32(np.median(np.abs(null))),
        ctrl_expr=ctrl[gcols].astype(np.float32),
    )
    print(f"\nwrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
