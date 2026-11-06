"""Can pooling across similar genes denoise the per-target Replogle residual?

The transfer model's whole score above the triviality line comes from the
per-target residual (`lookup - common`), and that residual transfers at only
r ~= 0.05, which is why lam has to be ~0.2. The README's stated bottleneck:
"A learned map from gene -> response can pool across similar genes, which is
the only way to beat the noise that forces lam ~= 0.2."

This tests that claim with no GPU and no new data. Everything runs off
external/transfer_probe3.npz: 2,052 perturbations measured in K562-gwps,
K562-essential and RPE1 over 6,699 shared readout genes.

Setup: build a predictor from K562-gwps ONLY, score it on RPE1 (held-out cell
line). The variants differ only in how the residual is estimated:

  raw          the target's own K562 residual, as shipped
  shrink       the same, scaled (this is exactly the lam dial)
  pc(k)        residual projected onto the top-k PCs of the K562 residual
               matrix -- pooling via a shared low-dimensional response basis
  knn(k)       residual replaced by the mean residual of its k nearest
               neighbours in K562 residual space (self included)
  knn+pc       both
  eb           per-row empirical-Bayes shrinkage, weight from how far that
               row's effect size clears the non-targeting null band

K562-essential is used only as an independent same-line replicate, to show
what the noise ceiling looks like under the identical scoring.

Usage:  python -I src/denoise_probe.py
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(REPO, "external", "transfer_probe3.npz")
OUT = os.path.join(REPO, "external", "denoise_probe.npz")

LAMS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0)
KS_PC = (3, 10, 30, 100)
KS_NN = (3, 10, 30, 100)


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


def unit_rows(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.maximum(n, 1e-12)


def knn_pool(R, k, exclude_self=False):
    """Mean residual over each row's k nearest neighbours in residual space.

    Cosine similarity on the residuals themselves. Noisy rows pick noisy
    neighbours, but averaging k of them still cancels independent noise, which
    is the only free lunch available here.
    """
    U = unit_rows(R)
    S = U @ U.T
    if exclude_self:
        np.fill_diagonal(S, -np.inf)
    idx = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
    return R[idx].mean(axis=1)


def pc_project(R, k, V):
    return (R @ V[:k].T) @ V[:k]


def main():
    d = np.load(NPZ)
    A, C, B = d["K562gw"].astype(np.float64), d["K562ess"].astype(np.float64), \
        d["RPE1"].astype(np.float64)
    common = d["common"].astype(np.float64)
    strong_k, strong_r = d["strong_k"], d["strong_r"]
    n = len(A)
    print(f"{n} perturbations x {A.shape[1]} genes "
          f"(K562gw / K562ess / RPE1, aligned)")

    sel = (strong_k > 2.0) & (strong_r > 2.0)
    print(f"strong in both K562gw and RPE1: {sel.sum()}\n")

    R = A - common[None, :]                      # the per-target residual
    Vt = np.linalg.svd(R, full_matrices=False)[2]

    variants = {"raw": R}
    for k in KS_PC:
        variants[f"pc({k})"] = pc_project(R, k, Vt)
    for k in KS_NN:
        variants[f"knn({k})"] = knn_pool(R, k)
    variants["knn(10)+pc(30)"] = pc_project(knn_pool(R, 10), 30, Vt)
    variants["knn(30)+pc(30)"] = pc_project(knn_pool(R, 30), 30, Vt)
    variants["knn(30)+pc(10)"] = pc_project(knn_pool(R, 30), 10, Vt)

    # Empirical-Bayes-ish: trust a row in proportion to how far its own effect
    # size clears the null band. Same shape as common.lam_per_target's ramp,
    # but applied to the residual itself rather than to the blend weight.
    w = np.clip((strong_k - 1.0) / 1.0, 0.0, 1.0)[:, None]
    variants["eb(ramp)"] = R * w
    variants["eb x knn(30)"] = R * w + (1 - w) * knn_pool(R, 30)

    # The residual alone, against RPE1's own residual: the cleanest read on
    # whether target-specific signal survives, with the shared axis removed
    # from BOTH sides so neither can borrow credit from it.
    Bres = B - B.mean(axis=0, keepdims=True)
    print("residual-only, scored against RPE1's own residual "
          "(no shared axis on either side):")
    print(f"  {'variant':<18s} {'median r':>9s} {'r|strong':>9s} "
          f"{'sign@100':>9s}")
    res_rows = {}
    for name, P in variants.items():
        r = corr_rows(P, Bres)
        ts = topk_sign(P, Bres)
        res_rows[name] = (np.nanmedian(r), np.nanmedian(r[sel]),
                          np.nanmedian(ts))
        print(f"  {name:<18s} {res_rows[name][0]:>+9.4f} "
              f"{res_rows[name][1]:>+9.4f} {res_rows[name][2]:>9.3f}")

    # Same-line replicate under identical scoring: the noise ceiling.
    Cres = C - C.mean(axis=0, keepdims=True)
    selc = (strong_k > 2.0) & (np.abs(C).mean(axis=1) / np.abs(C).mean() > 1.0)
    r = corr_rows(R, Cres)
    print(f"\n  {'CEILING raw->K562ess':<18s} {np.nanmedian(r):>+9.4f} "
          f"{np.nanmedian(r[selc]):>+9.4f} "
          f"{np.nanmedian(topk_sign(R, Cres)):>9.3f}")
    rk = corr_rows(knn_pool(R, 30), Cres)
    print(f"  {'knn(30)->K562ess':<18s} {np.nanmedian(rk):>+9.4f} "
          f"{np.nanmedian(rk[selc]):>+9.4f} "
          f"{np.nanmedian(topk_sign(knn_pool(R, 30), Cres)):>9.3f}")

    # ---- the shipped prediction form: common + lam * residual, vs RPE1 ----
    print("\nfull prediction  common + lam*residual  vs RPE1 (median row r):")
    hdr = "  " + f"{'variant':<18s}" + "".join(f"{l:>8}" for l in LAMS)
    print(hdr)
    best = {}
    table = {}
    for name, P in variants.items():
        row = []
        for lam in LAMS:
            pred = common[None, :] + lam * P
            row.append(np.nanmedian(corr_rows(pred, B)))
        table[name] = row
        j = int(np.argmax(row))
        best[name] = (LAMS[j], row[j])
        print("  " + f"{name:<18s}" + "".join(f"{v:>+8.4f}" for v in row))

    r0 = np.nanmedian(corr_rows(np.repeat(common[None, :], n, axis=0), B))
    print(f"\n  lam=0 (common only): {r0:+.4f}")
    print("\nbest lam per variant:")
    for name, (lam, v) in sorted(best.items(), key=lambda kv: -kv[1][1]):
        print(f"  {name:<18s} lam={lam:<5g} r={v:+.4f}  "
              f"(+{v - r0:+.4f} over common-only)")

    # sign@100 at each variant's best lam -- closer to what fid/reach grade.
    print("\nsign@100 of the full prediction at each variant's best lam:")
    for name, (lam, _) in sorted(best.items()):
        pred = common[None, :] + lam * variants[name]
        ts = topk_sign(pred, B)
        print(f"  {name:<18s} lam={lam:<5g} "
              f"sign@100={np.nanmedian(ts):.4f} "
              f"strong={np.nanmedian(ts[sel]):.4f}")
    ts0 = topk_sign(np.repeat(common[None, :], n, axis=0), B)
    print(f"  {'lam=0':<18s}            "
          f"sign@100={np.nanmedian(ts0):.4f} "
          f"strong={np.nanmedian(ts0[sel]):.4f}")

    np.savez_compressed(
        OUT,
        names=np.array(list(table.keys())),
        lams=np.array(LAMS),
        r_table=np.array([table[k] for k in table], dtype=np.float32),
        r_common_only=np.float32(r0),
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
