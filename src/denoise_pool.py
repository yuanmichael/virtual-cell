"""Why did pooling help at n=2,052 and not in the shipped artifact?

src/denoise_probe.py pooled each target's residual with its neighbours drawn
from the 2,052 perturbations that K562-gwps, K562-essential and RPE1 all
measure, and residual transfer to RPE1 rose from r=+0.088 to +0.132.
src/build_axis.py pools from all 9,866 gwps perturbations instead, and
src/denoise_verify.py found no gain.

Two candidate causes, and they imply opposite next moves:
  (a) n=35 in the verify overlap is simply too few to see +0.04.
  (b) the 9,866-row pool is mostly weakly-measured perturbations, so cosine
      similarity on noisy residuals picks bad neighbours, and averaging them
      injects noise instead of cancelling it.

This separates them: same 2,052 evaluation rows and the same RPE1 held-out
scoring throughout, varying ONLY where neighbours come from, and how the pool
is filtered by measurement quality (`num_cells_filtered`, `fold_expr`).

Usage:  python -I src/denoise_pool.py
"""

import os
import re
import sys

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BULK = os.path.join(REPO, "external", "replogle_bulk", "K562_gwps_raw_bulk_01.h5ad")
PROBE = os.path.join(REPO, "external", "transfer_probe3.npz")

DEPTH, EPS = 1e4, 1e-3
KS = (10, 30, 100)


def _decode(a):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in a])


def legacy_str(side, key):
    d = side[key]
    if d.dtype.kind in "iu" and "__categories" in side and key in side["__categories"]:
        cats = _decode(side["__categories"][key][:])
        return np.array([cats[c] if c >= 0 else "" for c in d[:]])
    return _decode(d[:])


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


def pool(R, rows, k, allow):
    """Mean residual over k nearest neighbours of each row, drawn from `allow`.

    The query row is always permitted, whether or not the filter admits it: at
    inference we have the target's own measurement regardless of its quality.
    """
    U = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    out = np.empty((len(rows), R.shape[1]))
    for i, r in enumerate(rows):
        s = U @ U[r]
        s[~allow] = -np.inf
        s[r] = np.inf
        nn = np.argpartition(-s, kth=k - 1)[:k]
        out[i] = R[nn].mean(axis=0)
    return out


def main():
    with h5py.File(BULK, "r") as h:
        X = h["X"][:].astype(np.float64)
        labels = legacy_str(h["obs"], "gene_transcript")
        genes = legacy_str(h["var"], "gene_name")
        ncells = np.nan_to_num(h["obs"]["num_cells_filtered"][:])
        fold = np.nan_to_num(h["obs"]["fold_expr"][:], nan=1.0)

    sym = np.array([l.split("_")[1] if "_" in l else l for l in labels])
    is_ctrl = np.array([bool(re.search("non-targeting", l)) for l in labels])
    X = X / X.sum(axis=1, keepdims=True) * DEPTH
    ctrl = X[is_ctrl].mean(axis=0)
    lfc = np.log2((X + EPS) / (ctrl + EPS))

    pr = np.load(PROBE, allow_pickle=False)
    p_genes = [str(g) for g in pr["genes"]]
    gi = {g: i for i, g in enumerate(genes)}
    cols = np.array([gi[g] for g in p_genes])

    prows = np.where(~is_ctrl)[0]
    R = lfc[prows][:, cols]
    R -= R.mean(axis=0, keepdims=True)
    nc, fe = ncells[prows], fold[prows]
    psym = sym[prows]
    print(f"full pool: {len(prows)} perturbations x {R.shape[1]} genes")
    print(f"  num_cells_filtered: median {np.median(nc):.0f}  "
          f"q10 {np.quantile(nc, 0.1):.0f}  q90 {np.quantile(nc, 0.9):.0f}")
    print(f"  |1-fold_expr| (knockdown strength): median "
          f"{np.median(np.abs(1-fe)):.2f}")

    # The 2,052 evaluation rows: measured in all three screens.
    shared = [str(s) for s in pr["shared"]]
    pos = {s: i for i, s in enumerate(psym)}
    erows = np.array([pos[s] for s in shared if s in pos])
    keep_eval = np.array([s in pos for s in shared])
    B = pr["RPE1"].astype(np.float64)[keep_eval]
    Bres = B - B.mean(axis=0, keepdims=True)
    strong = pr["strong_k"][keep_eval]
    print(f"evaluation rows: {len(erows)} of {len(shared)} shared perts "
          f"located in the full screen\n")

    # Pool definitions, from "only the high-quality shared subset" to "all".
    in_shared = np.zeros(len(prows), bool)
    in_shared[erows] = True
    pools = {
        "shared-2052": in_shared,
        "all-9866": np.ones(len(prows), bool),
        "ncells>=100": nc >= 100,
        "ncells>=200": nc >= 200,
        "ncells>=100 & kd>=0.3": (nc >= 100) & (np.abs(1 - fe) >= 0.3),
    }

    base_r = np.nanmedian(corr_rows(R[erows], Bres))
    base_s = np.nanmedian(topk_sign(R[erows], Bres))
    print(f"unpooled residual -> RPE1:  median r {base_r:+.4f}  "
          f"|strong {np.nanmedian(corr_rows(R[erows], Bres)[strong > 2]):+.4f}"
          f"  sign@100 {base_s:.3f}\n")

    print(f"  {'pool':<24s} {'size':>6s} {'k':>4s} {'median r':>9s} "
          f"{'r|strong':>9s} {'sign@100':>9s}")
    for name, allow in pools.items():
        for k in KS:
            P = pool(R, erows, k, allow)
            r = corr_rows(P, Bres)
            print(f"  {name:<24s} {int(allow.sum()):>6d} {k:>4d} "
                  f"{np.nanmedian(r):>+9.4f} "
                  f"{np.nanmedian(r[strong > 2]):>+9.4f} "
                  f"{np.nanmedian(topk_sign(P, Bres)):>9.3f}")

    # Is the verify script's disagreement just n=35? Resample the evaluation
    # rows down to 35 many times and report the spread of the gain.
    best_pool = "ncells>=100"
    P = pool(R, erows, 30, pools[best_pool])
    g = corr_rows(P, Bres) - corr_rows(R[erows], Bres)
    rng = np.random.default_rng(0)
    draws = np.array([np.nanmedian(g[rng.choice(len(g), 35, replace=False)])
                      for _ in range(2000)])
    print(f"\ngain of pool30/{best_pool} over unpooled, median over all "
          f"{len(g)} rows: {np.nanmedian(g):+.4f}")
    print(f"  at n=35 the same gain lands in "
          f"[{np.quantile(draws, 0.05):+.4f}, {np.quantile(draws, 0.95):+.4f}] "
          f"90% of the time, P(<=0) = {np.mean(draws <= 0):.2f}")
    print("  -> an n=35 check cannot resolve this gain either way"
          if np.mean(draws <= 0) > 0.1 else
          "  -> n=35 should have shown it; the pool is the problem")


if __name__ == "__main__":
    sys.exit(main())
