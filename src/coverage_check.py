"""How many of the 300 VCC 2026 targets does each public perturbation atlas cover?

Runs locally on pseudobulk ("bulk") h5ad files, which carry one row per
perturbation and are small enough to keep on the laptop.

Usage:
    python src/coverage_check.py
"""

import csv
import os
import re
import sys

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLS = os.path.join(REPO, "data", "vcc_2026_controls")
BULK_DIR = os.path.join(REPO, "external", "replogle_bulk")
PERT_LISTS = os.path.join(REPO, "external", "pert_lists")
STATE_META = os.path.join(REPO, "external", "state_ckpt_meta")


def read_col(path, col):
    with open(path, newline="") as fh:
        return [r[col] for r in csv.DictReader(fh)]


def h5_strings(grp):
    """Decode an h5ad obs/var field, following categorical encodings."""
    if isinstance(grp, h5py.Dataset):
        return [v.decode() if isinstance(v, bytes) else str(v) for v in grp[:]]
    if isinstance(grp, h5py.Group) and "categories" in grp:
        cats = [
            v.decode() if isinstance(v, bytes) else str(v) for v in grp["categories"][:]
        ]
        codes = grp["codes"][:]
        return [cats[c] if c >= 0 else "" for c in codes]
    raise TypeError(f"cannot decode {grp}")


def symbols_from(labels):
    """Pull gene symbols out of labels like 'ENSG00000123|TP53' or 'TP53_+_1234'."""
    out = set()
    for lab in labels:
        if not lab:
            continue
        parts = re.split(r"[|_]", lab)
        for p in parts:
            p = p.strip()
            if p and not p.startswith("ENSG") and not p.isdigit():
                out.add(p)
        out.add(lab.strip())
    return out


def report(name, covered, targets, extra=""):
    pct = 100.0 * len(covered) / len(targets)
    print(f"  {name:<34s} {len(covered):>3d}/{len(targets)}  ({pct:5.1f}%) {extra}")
    return covered


def main():
    targets = read_col(os.path.join(CONTROLS, "pert_counts.csv"), "target_gene")
    tset = set(targets)
    print(f"VCC 2026 targets: {len(targets)}\n")
    print("Coverage of the 300 targets by public perturbation data:")

    all_covered = set()

    # 1. STATE's released perturbation vocabulary (Replogle 4-line subset).
    pmap = os.path.join(STATE_META, "zeroshot__k562__pert_onehot_map.pt")
    if os.path.exists(pmap):
        import torch

        vocab = {str(k) for k in torch.load(pmap, weights_only=False).keys()}
        report("STATE vocab (Replogle 4-line)", tset & vocab, targets,
               f"[vocab={len(vocab)}]")

    # 2. VCC 2025 H1 training targets.
    h1 = os.path.join(PERT_LISTS, "h1_2025_train_perts.csv")
    if os.path.exists(h1):
        s = set(read_col(h1, "target_gene"))
        all_covered |= report("VCC 2025 H1 (train)", tset & s, targets,
                              f"[n={len(s)}]")

    # 3. Replogle pseudobulk screens.
    for fn in sorted(os.listdir(BULK_DIR)) if os.path.isdir(BULK_DIR) else []:
        if not fn.endswith(".h5ad"):
            continue
        path = os.path.join(BULK_DIR, fn)
        with h5py.File(path, "r") as h:
            obs = h["obs"]
            cand, best = None, 0
            for key in obs.keys():
                if key.startswith("_"):
                    continue
                try:
                    labels = h5_strings(obs[key])
                except TypeError:
                    continue
                hit = len(tset & symbols_from(labels))
                if hit > best:
                    cand, best = key, hit
            if cand is None:
                idx = h5_strings(obs[obs.attrs.get("_index", "_index")])
                cand, labels = "_index", idx
            else:
                labels = h5_strings(obs[cand])
            syms = symbols_from(labels)
        all_covered |= report(f"{fn[:30]} [{cand}]", tset & syms, targets,
                              f"[rows={len(labels)}]")

    print()
    report("UNION of all above (excl. STATE)", all_covered, targets)
    missing = sorted(tset - all_covered)
    print(f"\n  uncovered targets: {len(missing)}")
    if missing:
        print("   ", ", ".join(missing[:25]) + (" ..." if len(missing) > 25 else ""))


if __name__ == "__main__":
    sys.exit(main())
