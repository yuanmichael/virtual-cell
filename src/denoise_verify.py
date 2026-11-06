"""Does the pooled lookup we would actually SHIP transfer better than the raw one?

src/denoise_probe.py showed pooling helps, but it pooled within the 2,052
perturbations the three screens share. The artifact in external/k562_axis.npz
pools over all 9,866 K562 perturbations, which is a different (and larger)
neighbour pool. This checks the shipped arrays themselves, on the subset of
requested targets that RPE1 also measures -- a held-out cell line.

Usage:  python -I src/denoise_verify.py
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AXIS = os.path.join(REPO, "external", "k562_axis.npz")
PROBE = os.path.join(REPO, "external", "transfer_probe3.npz")

LAMS = (0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0)


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
    ax = np.load(AXIS, allow_pickle=False)
    pr = np.load(PROBE, allow_pickle=False)

    a_genes = [str(g) for g in ax["readout"]]
    p_genes = [str(g) for g in pr["genes"]]
    genes = sorted(set(a_genes) & set(p_genes))
    acol = np.array([a_genes.index(g) for g in genes])
    pcol = np.array([p_genes.index(g) for g in genes])

    a_t = [str(t) for t in ax["targets"]]
    p_t = [str(t) for t in pr["shared"]]
    both = sorted(set(a_t) & set(p_t))
    arow = np.array([a_t.index(t) for t in both])
    prow = np.array([p_t.index(t) for t in both])
    print(f"{len(both)} requested targets also measured in RPE1, "
          f"{len(genes)} shared readout genes")

    B = pr["RPE1"].astype(np.float64)[prow][:, pcol]
    common = ax["common"].astype(np.float64)[acol]
    strong = pr["strong_k"][prow]

    variants = {"raw": ax["lookup"]}
    for k in (10, 30, 100):
        if f"pool{k}" in ax.files:
            variants[f"pool{k}"] = ax[f"pool{k}"]

    print("\nresidual-only vs RPE1's own residual:")
    Bres = B - B.mean(axis=0, keepdims=True)
    for name, L in variants.items():
        R = L.astype(np.float64)[arow][:, acol] - common[None, :]
        r = corr_rows(R, Bres)
        print(f"  {name:<8s} median r {np.nanmedian(r):+.4f}  "
              f"|strong {np.nanmedian(r[strong > 2.0]):+.4f}  "
              f"sign@100 {np.nanmedian(topk_sign(R, Bres)):.3f}  "
              f"mean |lfc| {np.abs(R).mean():.4f}")

    print("\nfull prediction  common + lam*residual  vs RPE1 (median row r):")
    print("  " + f"{'variant':<8s}" + "".join(f"{l:>8}" for l in LAMS))
    for name, L in variants.items():
        R = L.astype(np.float64)[arow][:, acol] - common[None, :]
        row = [np.nanmedian(corr_rows(common[None, :] + lam * R, B))
               for lam in LAMS]
        j = int(np.argmax(row))
        print("  " + f"{name:<8s}" + "".join(f"{v:>+8.4f}" for v in row)
              + f"   best lam={LAMS[j]:g} r={row[j]:+.4f}")
    n = len(both)
    r0 = np.nanmedian(corr_rows(np.repeat(common[None, :], n, axis=0), B))
    print(f"  lam=0 (common only): {r0:+.4f}")

    print("\nsign@100 of the full prediction (what fid/reach grade):")
    print("  " + f"{'variant':<8s}" + "".join(f"{l:>8}" for l in LAMS))
    for name, L in variants.items():
        R = L.astype(np.float64)[arow][:, acol] - common[None, :]
        row = [np.nanmedian(topk_sign(common[None, :] + lam * R, B))
               for lam in LAMS]
        j = int(np.argmax(row))
        print("  " + f"{name:<8s}" + "".join(f"{v:>8.3f}" for v in row)
              + f"   best lam={LAMS[j]:g} sign={row[j]:.3f}")
    print(f"  lam=0 (common only): "
          f"{np.nanmedian(topk_sign(np.repeat(common[None, :], n, axis=0), B)):.3f}")


if __name__ == "__main__":
    sys.exit(main())
