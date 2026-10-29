"""Compare the final D/E/F bundle against the A/B/C bundle we measured on.

Every measurement so far was made on A/B/C. Nothing guarantees D/E/F share
their gene list, depth, density or target genes, and all three would change
the predictor's calibration. Check before building on them.

Usage:  python -I src/check_final.py
"""

import csv
import json
import os
import sys

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL = os.path.join(REPO, "data", "vcc_2026_controls")
FIN = os.path.join(REPO, "data", "vcc_2026_controls_final")


def read_genes(d):
    with open(os.path.join(d, "gene_names.csv")) as fh:
        g = [l.strip() for l in fh if l.strip()]
    return g[1:] if g and g[0] == "gene_name" else g


def read_targets(d):
    with open(os.path.join(d, "pert_counts.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [r["target_gene"] for r in rows]


def summarise(path, genes, targets):
    """Depth, density and the target genes' expression, from the raw CSR."""
    with h5py.File(path, "r") as h:
        X = h["X"]
        shape = tuple(h["X"].attrs["shape"]) if "shape" in X.attrs else None
        data = X["data"]
        indices = X["indices"]
        indptr = X["indptr"][:]
        n_cells = len(indptr) - 1
        nnz = data.shape[0]

        # Stream a column-sum so we never hold the matrix in memory.
        n_genes = shape[1] if shape else len(genes)
        gsum = np.zeros(n_genes, np.float64)
        gdet = np.zeros(n_genes, np.int64)
        depth = np.zeros(n_cells, np.float64)
        step = 4_000_000
        for s in range(0, nnz, step):
            e = min(s + step, nnz)
            d = data[s:e].astype(np.float64)
            ix = indices[s:e]
            np.add.at(gsum, ix, d)
            np.add.at(gdet, ix, 1)
        # Second pass for per-cell depth: map each nonzero back to its row.
        for s in range(0, nnz, step):
            e = min(s + step, nnz)
            d = data[s:e].astype(np.float64)
            rows = np.searchsorted(indptr, np.arange(s, e), side="right") - 1
            np.add.at(depth, rows, d)

    cpm = gsum / gsum.sum() * 1e6
    gi = {g: i for i, g in enumerate(genes)}
    tix = np.array([gi[t] for t in targets if t in gi])
    return dict(
        cells=n_cells, genes=n_genes, nnz=nnz,
        density=nnz / (n_cells * n_genes),
        med_depth=float(np.median(depth)),
        med_cpm_all=float(np.median(cpm)),
        med_cpm_targets=float(np.median(cpm[tix])),
        n_targets_found=len(tix),
        frac_genes_rare=float(np.mean(gdet / n_cells < 0.01)),
        n_de_eligible=int(np.sum(cpm > 5)),
    )


def main():
    gv, gf = read_genes(VAL), read_genes(FIN)
    tv, tf = read_targets(VAL), read_targets(FIN)

    print("manifest (final):")
    with open(os.path.join(FIN, "manifest.json")) as fh:
        man = json.load(fh)
    print(json.dumps(man, indent=2)[:1400])

    print(f"\ngene list: val {len(gv)}  final {len(gf)}  "
          f"identical order: {gv == gf}  same set: {set(gv) == set(gf)}")
    print(f"targets:   val {len(tv)}  final {len(tf)}  "
          f"same set: {set(tv) == set(tf)}  "
          f"overlap: {len(set(tv) & set(tf))}")
    if set(tv) != set(tf):
        only_f = sorted(set(tf) - set(tv))
        print(f"  final-only targets ({len(only_f)}): "
              f"{', '.join(only_f[:20])}{' ...' if len(only_f) > 20 else ''}")

    print()
    for ctx in ("D", "E", "F"):
        p = os.path.join(FIN, f"context_{ctx}.h5ad")
        s = summarise(p, gf, tf)
        print(f"context {ctx}: {s['cells']} cells x {s['genes']} genes, "
              f"{s['density']*100:.1f}% nonzero, median depth "
              f"{s['med_depth']:.0f} UMI")
        print(f"            median CPM all genes {s['med_cpm_all']:.1f}, "
              f"the {s['n_targets_found']} targets {s['med_cpm_targets']:.1f}; "
              f"genes >5 CPM (DE-eligible) {s['n_de_eligible']}")
        print(f"            genes detected in <1% of cells: "
              f"{s['frac_genes_rare']*100:.1f}%")


if __name__ == "__main__":
    sys.exit(main())
