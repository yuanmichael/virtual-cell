"""Does a K562 response predict the same knockdown's response in RPE1?

This is the premise Route B rests on. If K562 and RPE1 log-fold-changes for
the *same* perturbation are well correlated, copying K562 is a good starting
point and a learned correction can improve it. If they are uncorrelated once
measurement noise is accounted for, the premise fails and we should stop.

Reports, for the perturbations measured in both screens:
  - K562 vs RPE1 log2FC correlation, overall and for strong perturbations
  - a noise floor from non-targeting controls, so we can tell how much of any
    decorrelation is biology rather than shot noise

Usage:  python -I src/transfer_probe.py
"""

import os
import re
import sys

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BULK_DIR = os.path.join(REPO, "external", "replogle_bulk")
CONTROLS = os.path.join(REPO, "data", "vcc_2026_controls")

DEPTH = 1e4   # rows are already ~depth-normalised; fix it exactly
EPS = 1e-3    # ~1000x below the median gene's normalised expression


def _decode(arr):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in arr])


def legacy_str(side, key):
    """Decode a string column in the anndata<0.7 `__categories` layout."""
    d = side[key]
    if d.dtype.kind in "iu" and "__categories" in side and key in side["__categories"]:
        cats = _decode(side["__categories"][key][:])
        return np.array([cats[c] if c >= 0 else "" for c in d[:]])
    return _decode(d[:])


def load(fn):
    """Return (log2fc matrix, perturbed gene symbols, readout gene symbols, obs dict).

    log2fc is perturbation rows x readout genes, computed against the mean of
    the screen's own non-targeting rows.
    """
    with h5py.File(os.path.join(BULK_DIR, fn), "r") as h:
        X = h["X"][:].astype(np.float64)
        labels = legacy_str(h["obs"], "gene_transcript")
        genes = legacy_str(h["var"], "gene_name")
        obs = {k: h["obs"][k][:] for k in
               ("fold_expr", "control_expr", "energy_test_p_value",
                "num_cells_filtered", "mann_whitney_counts")}

    # Row label looks like '0_A1BG_P1_ENSG00000121410'; field 1 is the symbol.
    sym = np.array([lab.split("_")[1] if "_" in lab else lab for lab in labels])
    is_ctrl = np.array([bool(re.search("non-targeting", lab)) for lab in labels])

    X = X / X.sum(axis=1, keepdims=True) * DEPTH
    ctrl = X[is_ctrl].mean(axis=0)
    lfc = np.log2((X + EPS) / (ctrl + EPS))
    return lfc, sym, is_ctrl, genes, obs, X, ctrl


def corr_rows(A, B):
    """Row-wise Pearson correlation between two equally shaped matrices."""
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    num = (A * B).sum(axis=1)
    den = np.sqrt((A * A).sum(axis=1) * (B * B).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def main():
    print("loading K562 genome-wide ...")
    k_lfc, k_sym, k_ctrl_mask, k_genes, k_obs, k_X, k_ctrlvec = load(
        "K562_gwps_raw_bulk_01.h5ad")
    print("loading RPE1 ...")
    r_lfc, r_sym, r_ctrl_mask, r_genes, r_obs, r_X, r_ctrlvec = load(
        "rpe1_raw_bulk_01.h5ad")

    print(f"\nK562: {k_lfc.shape[0]} rows ({k_ctrl_mask.sum()} NTC), "
          f"{k_lfc.shape[1]} readout genes")
    print(f"RPE1: {r_lfc.shape[0]} rows ({r_ctrl_mask.sum()} NTC), "
          f"{r_lfc.shape[1]} readout genes")

    # --- cells per perturbation: how noisy is each row? ---
    for name, obs in (("K562", k_obs), ("RPE1", r_obs)):
        n = obs["num_cells_filtered"]
        print(f"{name} cells/pert: median {np.median(n):.0f}  "
              f"q10 {np.quantile(n, .1):.0f}  q90 {np.quantile(n, .9):.0f}")

    # --- readout gene space ---
    gshared = sorted(set(k_genes) & set(r_genes))
    with open(os.path.join(CONTROLS, "gene_names.csv")) as fh:
        vcc_genes = {ln.strip() for ln in fh if ln.strip()}
    vcc_genes.discard("gene_name")
    gscored = sorted(set(gshared) & vcc_genes)
    print(f"\nreadout genes: K562&RPE1 {len(gshared)}, "
          f"also in the VCC 18,533 {len(gscored)}")

    ki = {g: i for i, g in enumerate(k_genes)}
    ri = {g: i for i, g in enumerate(r_genes)}
    kcols = [ki[g] for g in gscored]
    rcols = [ri[g] for g in gscored]

    # --- shared perturbations ---
    kp = {s: i for i, s in zip(range(len(k_sym)), k_sym) if not k_ctrl_mask[i]}
    rp = {s: i for i, s in zip(range(len(r_sym)), r_sym) if not r_ctrl_mask[i]}
    shared = sorted(set(kp) & set(rp))
    print(f"perturbations in both screens: {len(shared)}")

    A = k_lfc[np.array([kp[s] for s in shared])][:, kcols]
    B = r_lfc[np.array([rp[s] for s in shared])][:, rcols]

    # --- noise floor: individual NTC rows vs the held-out NTC mean ---
    def null_lfc(X, mask, cols, n=60):
        rows = np.where(mask)[0][:n]
        out = []
        for i in rows:
            others = np.setdiff1d(np.where(mask)[0], [i])
            ctrl = X[others].mean(axis=0)
            out.append(np.log2((X[i] + EPS) / (ctrl + EPS))[cols])
        return np.array(out)

    kn = null_lfc(k_X, k_ctrl_mask, kcols)
    rn = null_lfc(r_X, r_ctrl_mask, rcols)
    nfloor = corr_rows(kn[: min(len(kn), len(rn))], rn[: min(len(kn), len(rn))])
    print(f"\nnoise floor (NTC vs NTC, cross-screen r): "
          f"median {np.nanmedian(nfloor):+.3f}")
    print(f"null |log2FC| median: K562 {np.median(np.abs(kn)):.3f}  "
          f"RPE1 {np.median(np.abs(rn)):.3f}")
    print(f"pert |log2FC| median: K562 {np.median(np.abs(A)):.3f}  "
          f"RPE1 {np.median(np.abs(B)):.3f}")

    # --- the headline number ---
    r_all = corr_rows(A, B)
    print(f"\nK562 vs RPE1 log2FC correlation over {len(shared)} shared perts:")
    print(f"  all:          median {np.nanmedian(r_all):+.3f}  "
          f"mean {np.nanmean(r_all):+.3f}")

    # Strength = how far the response sits above the null band, per screen.
    kmag = np.abs(A).mean(axis=1) / np.median(np.abs(kn))
    rmag = np.abs(B).mean(axis=1) / np.median(np.abs(rn))
    for thr in (1.5, 2.0, 3.0):
        sel = (kmag > thr) & (rmag > thr)
        if sel.sum() > 5:
            print(f"  both >{thr}x null ({sel.sum():4d} perts):  "
                  f"median {np.nanmedian(r_all[sel]):+.3f}")

    # Does K562 alone tell us it will be a strong perturbation? (we only ever
    # see the K562 side at inference time)
    sel = kmag > 2.0
    print(f"  K562 >2x null  ({sel.sum():4d} perts):  "
          f"median {np.nanmedian(r_all[sel]):+.3f}")

    # --- how much is a pure cell-line shift, i.e. learnable? ---
    # If RPE1 = a * K562 + (shared offset), a global rescale already helps.
    num = (A * B).sum()
    den = (A * A).sum()
    print(f"\nbest global scale RPE1 ~ a*K562: a = {num / den:.3f}")
    print(f"  mean |lfc| ratio RPE1/K562 = "
          f"{np.abs(B).mean() / np.abs(A).mean():.3f}")

    # Sign agreement on the genes each screen calls biggest.
    def topk_sign(a, b, k=100):
        hits = []
        for i in range(len(a)):
            idx = np.argsort(-np.abs(a[i]))[:k]
            hits.append(np.mean(np.sign(a[i][idx]) == np.sign(b[i][idx])))
        return np.array(hits)

    ts = topk_sign(A, B)
    print(f"sign agreement on K562's top-100 genes: median "
          f"{np.nanmedian(ts):.3f}  (>2x null: {np.nanmedian(ts[sel]):.3f})")

    np.savez_compressed(
        os.path.join(REPO, "external", "transfer_probe.npz"),
        shared=np.array(shared), genes=np.array(gscored),
        A=A.astype(np.float32), B=B.astype(np.float32),
        kmag=kmag, rmag=rmag, r_all=r_all,
        k_null=kn.astype(np.float32), r_null=rn.astype(np.float32),
    )
    print("\nwrote external/transfer_probe.npz")


if __name__ == "__main__":
    sys.exit(main())
