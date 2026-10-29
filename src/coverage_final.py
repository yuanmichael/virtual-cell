"""Coverage and effect-size profile of the FINAL 300 targets (D/E/F).

The final panel shares only 129 of the validation panel's 300 targets, so the
earlier 272/300 K562 coverage figure does not carry over. Recompute it, and
check the new targets sit in the same effect-size regime the predictor's
blend weight was fitted on.

Usage:  python -I src/coverage_final.py
"""

import csv
import os
import re
import sys

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BULK_DIR = os.path.join(REPO, "external", "replogle_bulk")
FIN = os.path.join(REPO, "data", "vcc_2026_controls_final")
VAL = os.path.join(REPO, "data", "vcc_2026_controls")
DEPTH, EPS, MIN_EXPR = 1e4, 1e-3, 0.05


def _decode(a):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in a])


def legacy_str(side, key):
    d = side[key]
    if d.dtype.kind in "iu" and "__categories" in side and key in side["__categories"]:
        cats = _decode(side["__categories"][key][:])
        return np.array([cats[c] if c >= 0 else "" for c in d[:]])
    return _decode(d[:])


def targets(d):
    with open(os.path.join(d, "pert_counts.csv"), newline="") as fh:
        return sorted({r["target_gene"] for r in csv.DictReader(fh)})


def main():
    tf, tv = targets(FIN), targets(VAL)
    print(f"final targets {len(tf)}, validation targets {len(tv)}, "
          f"shared {len(set(tf) & set(tv))}\n")

    with h5py.File(os.path.join(BULK_DIR, "K562_gwps_raw_bulk_01.h5ad"), "r") as h:
        X = h["X"][:].astype(np.float64)
        labels = legacy_str(h["obs"], "gene_transcript")
        genes = legacy_str(h["var"], "gene_name")
        ncells = np.nan_to_num(h["obs"]["num_cells_filtered"][:])
        fold = h["obs"]["fold_expr"][:]

    sym = np.array([l.split("_")[1] if "_" in l else l for l in labels])
    ctrl_mask = np.array([bool(re.search("non-targeting", l)) for l in labels])
    X = X / X.sum(axis=1, keepdims=True) * DEPTH
    ctrl = X[ctrl_mask].mean(axis=0)
    lfc = np.log2((X + EPS) / (ctrl + EPS))
    pidx = {s: i for i, s in enumerate(sym) if not ctrl_mask[i]}

    with open(os.path.join(FIN, "gene_names.csv")) as fh:
        vcc = [l.strip() for l in fh if l.strip()]
    vcc = set(vcc[1:] if vcc[0] == "gene_name" else vcc)
    gi = {g: i for i, g in enumerate(genes)}
    readout = sorted(set(genes) & vcc)
    readout = [g for g in readout if ctrl[gi[g]] > MIN_EXPR]
    cols = np.array([gi[g] for g in readout])
    print(f"K562 readout genes usable for prediction: {len(readout)} "
          f"of the 18,533 VCC genes ({len(readout)/18533*100:.1f}%)")

    nullr = np.array([
        np.log2((X[i] + EPS) /
                (X[np.setdiff1d(np.where(ctrl_mask)[0], [i])].mean(axis=0)
                 + EPS))[cols]
        for i in np.where(ctrl_mask)[0][:80]])
    nmed = np.median(np.abs(nullr))

    for name, tt in (("FINAL  (D/E/F)", tf), ("validation (A/B/C)", tv)):
        hit = [t for t in tt if t in pidx]
        rows = np.array([pidx[t] for t in hit])
        mag = np.abs(lfc[rows][:, cols]).mean(axis=1) / nmed
        print(f"\n{name}: covered by K562 genome-wide "
              f"{len(hit)}/{len(tt)} ({len(hit)/len(tt)*100:.1f}%)")
        print(f"  cells/pert median {np.median(ncells[rows]):.0f}   "
              f"effect size median {np.median(mag):.2f}x null   "
              f"frac >2x {np.mean(mag > 2):.3f}")
        print(f"  on-target fold_expr median {np.nanmedian(fold[rows]):.3f}")
        miss = sorted(set(tt) - set(hit))
        print(f"  uncovered {len(miss)}: "
              f"{', '.join(miss[:18])}{' ...' if len(miss) > 18 else ''}")

    allrows = np.array(sorted(pidx.values()))
    magall = np.abs(lfc[allrows][:, cols]).mean(axis=1) / nmed
    print(f"\nreference: all {len(allrows)} K562 perturbations  "
          f"cells {np.median(ncells[allrows]):.0f}  "
          f"effect {np.median(magall):.2f}x  frac>2x {np.mean(magall>2):.3f}")

    # Save what the predictor needs: the common axis plus per-target lookups,
    # on the final panel's readout genes.
    hit = [t for t in tf if t in pidx]
    np.savez_compressed(
        os.path.join(REPO, "external", "final_lookup.npz"),
        readout=np.array(readout),
        common=lfc[allrows][:, cols].mean(axis=0).astype(np.float32),
        targets_covered=np.array(hit),
        lookup=lfc[np.array([pidx[t] for t in hit])][:, cols].astype(np.float32),
        targets_all=np.array(tf),
        null_sd=np.std(nullr, axis=0).astype(np.float32),
        ncells=ncells[np.array([pidx[t] for t in hit])],
    )
    print(f"\nwrote external/final_lookup.npz "
          f"(common axis + {len(hit)} lookups over {len(readout)} genes)")


if __name__ == "__main__":
    sys.exit(main())
