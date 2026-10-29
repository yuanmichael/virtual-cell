"""Is transferable signal perturbation-SPECIFIC, or just a shared response axis?

Perturb-seq responses are known to be dominated by a few common axes (stress,
proliferation, ribosome). If cross-line correlation is carried entirely by
that shared axis, then predicting "the average knockdown response" scores as
well as a per-gene lookup, and per-gene identity is worth nothing -- a very
different strategy from a correction model.

Decisive comparison, scored on RPE1 as the held-out line:
  common   every perturbation predicted as the MEAN K562 response
  lookup   each perturbation predicted as ITS OWN K562 response
  residual its own K562 response after removing the common axes

Also reports top-k sign agreement, which is what the VCC fid/reach metrics
actually depend on, for the same-line ceiling and the cross-line transfer.

Usage:  python -I src/transfer_probe3.py
"""

import os
import re
import sys

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BULK_DIR = os.path.join(REPO, "external", "replogle_bulk")
CONTROLS = os.path.join(REPO, "data", "vcc_2026_controls")

DEPTH, EPS, MIN_EXPR = 1e4, 1e-3, 0.05
FILES = {"K562gw": "K562_gwps_raw_bulk_01.h5ad",
         "K562ess": "K562_essential_raw_bulk_01.h5ad",
         "RPE1": "rpe1_raw_bulk_01.h5ad"}


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
            self.ncells = h["obs"]["num_cells_filtered"][:]
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
    """Direction agreement on the k genes the PREDICTION calls largest.

    This mirrors fid/reach: we are graded on the set we choose to call.
    """
    out = np.empty(len(pred))
    for i in range(len(pred)):
        idx = np.argsort(-np.abs(pred[i]))[:k]
        out[i] = np.mean(np.sign(pred[i][idx]) == np.sign(true[i][idx]))
    return out


def null_rows(s, cols, n=80):
    idx = np.where(s.ctrl_mask)[0]
    return np.array([
        np.log2((s.X[i] + EPS) / (s.X[np.setdiff1d(idx, [i])].mean(axis=0) + EPS))[cols]
        for i in idx[:n]])


def main():
    scr = {k: Screen(v) for k, v in FILES.items()}
    with open(os.path.join(CONTROLS, "gene_names.csv")) as fh:
        vcc = {l.strip() for l in fh if l.strip()}
    vcc.discard("gene_name")

    genes = set(vcc)
    for s in scr.values():
        genes &= set(s.genes)
    genes = sorted(genes)
    keep = np.ones(len(genes), bool)
    for s in scr.values():
        keep &= s.ctrl[s.cols(genes)] > MIN_EXPR
    genes = [g for g, k in zip(genes, keep) if k]
    cols = {k: s.cols(genes) for k, s in scr.items()}
    print(f"{len(genes)} readout genes\n")

    nulls = {k: null_rows(s, cols[k]) for k, s in scr.items()}
    nmed = {k: np.median(np.abs(v)) for k, v in nulls.items()}

    # ---- aligned matrices over perturbations shared by all three screens ----
    shared = sorted(set(scr["K562gw"].pidx) & set(scr["K562ess"].pidx)
                    & set(scr["RPE1"].pidx))
    M = {k: scr[k].lfc[np.array([scr[k].pidx[s] for s in shared])][:, cols[k]]
         for k in scr}
    print(f"{len(shared)} perturbations measured in all three screens")

    strong = {k: np.abs(M[k]).mean(axis=1) / nmed[k] for k in M}
    sel = (strong["K562gw"] > 2.0) & (strong["RPE1"] > 2.0)
    print(f"  of those, strong in both K562gw and RPE1: {sel.sum()}\n")

    # ---- how concentrated is the response space? ----
    A = M["K562gw"]
    Ac = A - A.mean(axis=0, keepdims=True)
    common = A.mean(axis=0)                      # the generic response
    U, S, Vt = np.linalg.svd(Ac, full_matrices=False)
    var = S**2 / (S**2).sum()
    print(f"K562 response matrix: PC variance explained "
          f"{', '.join(f'{v:.3f}' for v in var[:6])}  "
          f"(top-10 total {var[:10].sum():.3f})")
    print(f"|common response| mean {np.abs(common).mean():.4f} vs "
          f"per-pert mean {np.abs(A).mean():.4f}  -> "
          f"common is {np.abs(common).mean()/np.abs(A).mean():.2f}x a typical row\n")

    # ---- the decisive comparison, scored against RPE1 ----
    B = M["RPE1"]
    n = len(shared)
    preds = {
        "common  (mean K562 response, same for every pert)":
            np.repeat(common[None, :], n, axis=0),
        "lookup  (this pert's own K562 response)": A,
        "residual(own K562 minus common axes)": A - common[None, :],
    }
    # Removing the top-5 shared axes entirely.
    V5 = Vt[:5]
    preds["resid-5 (own K562, top-5 PCs projected out)"] = (
        Ac - (Ac @ V5.T) @ V5)

    print("predicting RPE1 (held-out cell line):")
    print(f"  {'predictor':<48s} {'median r':>9s} {'r|strong':>9s} "
          f"{'sign@100':>9s} {'sgn|strong':>10s}")
    for name, P in preds.items():
        r = corr_rows(P, B)
        ts = topk_sign(P, B)
        print(f"  {name:<48s} {np.nanmedian(r):>+9.3f} "
              f"{np.nanmedian(r[sel]):>+9.3f} "
              f"{np.nanmedian(ts):>9.3f} {np.nanmedian(ts[sel]):>10.3f}")

    # Same-line ceiling under the identical scoring, for reference.
    C = M["K562ess"]
    r = corr_rows(A, C)
    ts = topk_sign(A, C)
    selc = (strong["K562gw"] > 2.0) & (strong["K562ess"] > 2.0)
    print(f"\n  {'CEILING lookup -> same line (K562ess)':<48s} "
          f"{np.nanmedian(r):>+9.3f} {np.nanmedian(r[selc]):>+9.3f} "
          f"{np.nanmedian(ts):>9.3f} {np.nanmedian(ts[selc]):>10.3f}")
    rc = corr_rows(np.repeat(common[None, :], n, axis=0), C)
    tc = topk_sign(np.repeat(common[None, :], n, axis=0), C)
    print(f"  {'CEILING common -> same line (K562ess)':<48s} "
          f"{np.nanmedian(rc):>+9.3f} {np.nanmedian(rc[selc]):>+9.3f} "
          f"{np.nanmedian(tc):>9.3f} {np.nanmedian(tc[selc]):>10.3f}")

    # ---- does the common axis itself survive the cell-line change? ----
    cB = B.mean(axis=0)
    rr = np.corrcoef(common, cB)[0, 1]
    print(f"\ncommon K562 response vs common RPE1 response: r = {rr:+.3f}")
    print(f"  sign agreement on K562-common's top 200 genes: "
          f"{topk_sign(common[None, :], cB[None, :], k=200)[0]:.3f}")

    np.savez_compressed(
        os.path.join(REPO, "external", "transfer_probe3.npz"),
        shared=np.array(shared), genes=np.array(genes),
        K562gw=A.astype(np.float32), K562ess=C.astype(np.float32),
        RPE1=B.astype(np.float32), common=common.astype(np.float32),
        strong_k=strong["K562gw"], strong_r=strong["RPE1"],
        pcs=Vt[:50].astype(np.float32), var=var[:50],
    )
    print("\nwrote external/transfer_probe3.npz")


if __name__ == "__main__":
    sys.exit(main())
