"""Turn the probe-3 finding into a calibrated predictor.

Probe 3 found that the MEAN K562 response transfers across cell lines better
than a perturbation's own measured response (r 0.207 vs 0.083). That makes the
natural model a shrinkage of the lookup toward the common axis:

    pred = common + lam * (lookup - common)

where lam depends on how reliable this perturbation's measurement is. This
script fits lam on held-out RPE1 and checks two things the design hinges on:

  1. the optimal lam overall, and as a function of effect size / cell count
  2. whether the common axis estimated from ESSENTIAL genes (all the Replogle
     screens contain) still applies to the 300 VCC targets, which Arc chose to
     be NON-essential -- the biggest threat to the whole approach

Usage:  python -I src/transfer_probe4.py
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
    print(f"{len(genes)} readout genes, {len(targets)} VCC targets\n")

    # ---------- 1. is the common axis an essential-gene artefact? ----------
    kt = {s: i for s, i in k.pidx.items()}
    vcc_in_k = [t for t in targets if t in kt]
    print(f"VCC targets present in K562 genome-wide: {len(vcc_in_k)}")

    nullk = np.array([
        np.log2((k.X[i] + EPS) /
                (k.X[np.setdiff1d(np.where(k.ctrl_mask)[0], [i])].mean(axis=0) + EPS))[kc]
        for i in np.where(k.ctrl_mask)[0][:80]])
    nmed = np.median(np.abs(nullk))

    allrows = np.array([i for s, i in k.pidx.items()])
    A_all = k.lfc[allrows][:, kc]
    vccrows = np.array([kt[t] for t in vcc_in_k])
    A_vcc = k.lfc[vccrows][:, kc]

    mag_all = np.abs(A_all).mean(axis=1) / nmed
    mag_vcc = np.abs(A_vcc).mean(axis=1) / nmed
    print(f"effect size vs null band (K562):")
    print(f"  all {len(A_all)} perturbations : median {np.median(mag_all):.2f}x  "
          f"frac >2x {np.mean(mag_all > 2):.3f}")
    print(f"  the {len(A_vcc)} VCC targets    : median {np.median(mag_vcc):.2f}x  "
          f"frac >2x {np.mean(mag_vcc > 2):.3f}")
    print(f"  on-target knockdown fold_expr: all "
          f"{np.nanmedian(k.fold[allrows]):.3f}, VCC "
          f"{np.nanmedian(k.fold[vccrows]):.3f}")

    common_all = A_all.mean(axis=0)
    common_vcc = A_vcc.mean(axis=0)
    print(f"\ncommon response, all perts vs VCC-targets-only: "
          f"r = {np.corrcoef(common_all, common_vcc)[0,1]:+.3f}")
    print(f"  |common_all| mean {np.abs(common_all).mean():.4f}, "
          f"|common_vcc| mean {np.abs(common_vcc).mean():.4f}")
    print(f"  sign agreement on common_all's top 200: "
          f"{topk_sign(common_all[None,:], common_vcc[None,:], k=200)[0]:.3f}")

    # ---------- 2. shrinkage sweep, scored on held-out RPE1 ----------
    shared = sorted(set(k.pidx) & set(r.pidx))
    A = k.lfc[np.array([k.pidx[s] for s in shared])][:, kc]
    B = r.lfc[np.array([r.pidx[s] for s in shared])][:, rc]
    # Estimate the common axis WITHOUT the perturbation being predicted, so the
    # sweep is not scored on something it has already seen.
    n = len(shared)
    tot = A.sum(axis=0)
    loo_common = (tot[None, :] - A) / (n - 1)

    magA = np.abs(A).mean(axis=1) / nmed
    nA = k.ncells[[k.pidx[s] for s in shared]]

    print(f"\nshrinkage sweep on {n} shared perturbations, scored vs RPE1")
    print(f"  pred = loo_common + lam * (own_K562 - loo_common)")
    print(f"  {'lam':>5s} {'median r':>9s} {'sign@100':>9s} "
          f"{'r|strong':>9s} {'sign|strong':>11s}")
    sel = magA > 2.0
    best = None
    for lam in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0):
        P = loo_common + lam * (A - loo_common)
        rr = np.nanmedian(corr_rows(P, B))
        ts = np.nanmedian(topk_sign(P, B))
        rs = np.nanmedian(corr_rows(P, B)[sel])
        tss = np.nanmedian(topk_sign(P, B)[sel])
        print(f"  {lam:>5.2f} {rr:>+9.3f} {ts:>9.3f} {rs:>+9.3f} {tss:>11.3f}")
        if best is None or rr > best[1]:
            best = (lam, rr)
    print(f"  best lam overall: {best[0]:.2f} (r={best[1]:+.3f})")

    # Does the ideal lam grow with measurement reliability?
    print(f"\n  optimal lam by effect-size tercile (reliability proxy):")
    qs = np.quantile(magA, [0, 1/3, 2/3, 1.0])
    for lo, hi, lbl in zip(qs[:-1], qs[1:], ("weak", "mid", "strong")):
        m = (magA >= lo) & (magA <= hi)
        row = []
        for lam in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
            P = loo_common[m] + lam * (A[m] - loo_common[m])
            row.append(np.nanmedian(corr_rows(P, B[m])))
        bl = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)[int(np.argmax(row))]
        print(f"    {lbl:<7s} n={m.sum():>5d}  "
              f"r by lam: {' '.join(f'{v:+.3f}' for v in row)}  -> best {bl:.1f}")

    # Deep-cell-count subset: does more data per perturbation help lam?
    print(f"\n  optimal lam by cells-per-perturbation tercile:")
    qs = np.quantile(nA, [0, 1/3, 2/3, 1.0])
    for lo, hi, lbl in zip(qs[:-1], qs[1:], ("few", "mid", "many")):
        m = (nA >= lo) & (nA <= hi)
        row = []
        for lam in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
            P = loo_common[m] + lam * (A[m] - loo_common[m])
            row.append(np.nanmedian(corr_rows(P, B[m])))
        bl = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)[int(np.argmax(row))]
        print(f"    {lbl:<7s} n={m.sum():>5d}  "
              f"r by lam: {' '.join(f'{v:+.3f}' for v in row)}  -> best {bl:.1f}")

    np.savez_compressed(
        os.path.join(REPO, "external", "common_axis.npz"),
        genes=np.array(genes),
        common_all=common_all.astype(np.float32),
        common_vcc=common_vcc.astype(np.float32),
        vcc_in_k=np.array(vcc_in_k),
        A_vcc=A_vcc.astype(np.float32),
        mag_vcc=mag_vcc.astype(np.float32),
        null_sd=np.std(nullk, axis=0).astype(np.float32),
    )
    print("\nwrote external/common_axis.npz")


if __name__ == "__main__":
    sys.exit(main())
