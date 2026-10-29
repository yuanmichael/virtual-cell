"""Stage 1 of the rehearsal: fetch Arc's public 2025 challenge data.

Why: the final phase returns no scores, so the only way left to tell a good
idea from a bad one is to score ourselves. Arc's 2025 release is real
single-cell perturbation data with ground-truth labels, produced on the same
assay as the 2026 contexts, which makes it a scoreable stand-in for D/E/F.

Downloads are resumable (HTTP Range against the existing partial file) and
committed to the Volume, so an interrupted run picks up where it stopped
rather than starting the 15.5 GB over.

Run:
    modal run --detach src/rehearsal.py --dry-run          # sizes only
    modal run --detach src/rehearsal.py                    # fetch + inspect
    modal run --detach src/rehearsal.py --cleanup-stale     # also free 20 GB
"""

import modal

from common import CPU_CORES, REHEARSAL, VOL, VOL_PATH, WORK, app, eval_image

BUCKET = "https://storage.googleapis.com/arc-institute-virtual-cell-atlas"
PREFIX = "virtual-cell-challenge/2025"

# Only the processed matrices and their label files; the FASTQs are raw reads.
#
# All three splits, because the 2025 challenge is over and its held-out data is
# now public. Their perturbation lists are fully DISJOINT (150 / 50 / 100), so
# together they give a real train/val/test setup on the same assay as D/E/F --
# train a model on `train`, tune on `val`, and keep `test` untouched as the
# only honest comparison between models.
WANTED = {
    "adata_Training.h5ad": f"{PREFIX}/train/adata_Training.h5ad",
    "pert_counts_Training.csv": f"{PREFIX}/train/pert_counts_Training.csv",
    "adata_Validation.h5ad": f"{PREFIX}/validation/adata_Validation.h5ad",
    "pert_counts_Validation.csv": f"{PREFIX}/validation/pert_counts_Validation.csv",
    "adata_Test.h5ad": f"{PREFIX}/test/adata_Test.h5ad",
    "pert_counts_Test.csv": f"{PREFIX}/test/pert_counts_Test.csv",
    "gene_names.csv": f"{PREFIX}/gene_names.csv",
}

# Files from the baseline submission that have served their purpose.
STALE = ["pred_baseline.h5ad", "baseline.vcc"]


@app.function(
    image=eval_image,
    volumes={VOL_PATH: VOL},
    cpu=CPU_CORES,
    memory=16384,
    timeout=60 * 60 * 4,
)
def fetch(dry_run: bool = False, cleanup_stale: bool = False):
    import os
    import requests

    os.makedirs(REHEARSAL, exist_ok=True)

    if cleanup_stale:
        for name in STALE:
            p = f"{WORK}/{name}"
            if os.path.exists(p):
                gb = os.path.getsize(p) / 1e9
                if dry_run:
                    print(f"would delete {p} ({gb:.2f} GB)")
                else:
                    os.remove(p)
                    print(f"deleted {p} ({gb:.2f} GB)")
        if not dry_run:
            VOL.commit()

    for name, key in WANTED.items():
        url = f"{BUCKET}/{key}"
        dest = f"{REHEARSAL}/{name}"

        head = requests.head(url, timeout=60)
        head.raise_for_status()
        total = int(head.headers["Content-Length"])
        have = os.path.getsize(dest) if os.path.exists(dest) else 0

        if have == total:
            print(f"OK   {name}: already complete ({total/1e9:.2f} GB)")
            continue
        if dry_run:
            print(f"TODO {name}: {total/1e9:.2f} GB "
                  f"(have {have/1e9:.2f} GB)")
            continue

        # Resume from the byte we stopped at rather than refetching.
        mode, headers = "wb", {}
        if 0 < have < total:
            mode, headers = "ab", {"Range": f"bytes={have}-"}
            print(f"resuming {name} at {have/1e9:.2f}/{total/1e9:.2f} GB")
        else:
            print(f"fetching {name}: {total/1e9:.2f} GB")

        done = have
        next_mark = done + 1_000_000_000
        with requests.get(url, headers=headers, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(dest, mode) as fh:
                for chunk in r.iter_content(chunk_size=8 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if done >= next_mark:
                        print(f"  {done/1e9:.1f}/{total/1e9:.1f} GB")
                        # Commit periodically so a crash keeps the partial file.
                        fh.flush()
                        os.fsync(fh.fileno())
                        VOL.commit()
                        next_mark = done + 1_000_000_000
        VOL.commit()
        got = os.path.getsize(dest)
        print(f"{'OK  ' if got == total else 'WARN'} {name}: "
              f"{got/1e9:.2f} GB (expected {total/1e9:.2f})")

    if dry_run:
        return

    inspect_local()


def inspect_local():
    """Report the 2025 training set's schema: what we can and cannot use."""
    import os
    import h5py
    import numpy as np

    path = f"{REHEARSAL}/adata_Training.h5ad"
    if not os.path.exists(path):
        print("no training file to inspect")
        return

    print("\n" + "=" * 70)
    print("adata_Training.h5ad")
    print("=" * 70)
    with h5py.File(path, "r") as h:
        print("top-level:", sorted(h.keys()))
        X = h["X"]
        if isinstance(X, h5py.Group):
            shape = tuple(X.attrs["shape"])
            print(f"X: sparse {X.attrs.get('encoding-type')} shape={shape} "
                  f"nnz={X['data'].shape[0]} dtype={X['data'].dtype}")
        else:
            print(f"X: dense {X.shape} {X.dtype}")
        if "layers" in h:
            print("layers:", sorted(h["layers"].keys()))
        if "obsm" in h:
            print("obsm:", sorted(h["obsm"].keys()))

        obs = h["obs"]
        print(f"\nobs index: {obs.attrs.get('_index')}")
        for k in sorted(obs.keys()):
            if k.startswith("_"):
                continue
            node = obs[k]
            if isinstance(node, h5py.Group) and "categories" in node:
                cats = node["categories"][:]
                cats = [c.decode() if isinstance(c, bytes) else str(c)
                        for c in cats]
                codes = node["codes"][:]
                uniq, cnt = np.unique(codes, return_counts=True)
                top = sorted(zip(cnt, uniq), reverse=True)[:6]
                print(f"  {k:<22s} categorical, {len(cats)} levels")
                for c, u in top:
                    print(f"      {cats[u]:<34s} {c:>8d} cells")
            elif isinstance(node, h5py.Dataset):
                if node.dtype.kind in "iuf":
                    v = node[: min(len(node), 100000)]
                    print(f"  {k:<22s} numeric {node.shape} {node.dtype} "
                          f"min {np.min(v):.3g} max {np.max(v):.3g}")
                else:
                    v = node[: min(len(node), 20)]
                    v = [x.decode() if isinstance(x, bytes) else str(x)
                         for x in v]
                    print(f"  {k:<22s} str {node.shape} e.g. {v[:3]}")

        var = h["var"]
        print(f"\nvar index: {var.attrs.get('_index')} "
              f"columns: {sorted(k for k in var.keys() if not k.startswith('_'))}")

    # Perturbation list and how many cells each has.
    pc = f"{REHEARSAL}/pert_counts_Training.csv"
    if os.path.exists(pc):
        with open(pc) as fh:
            lines = [l.strip() for l in fh if l.strip()]
        print(f"\npert_counts_Training.csv: {len(lines)-1} rows")
        print("  header:", lines[0])
        for l in lines[1:4]:
            print("   ", l)

    gn = f"{REHEARSAL}/gene_names.csv"
    if os.path.exists(gn):
        with open(gn) as fh:
            g = [l.strip() for l in fh if l.strip()]
        g = g[1:] if g and g[0].lower() in ("gene_name", "gene", "genes") else g
        print(f"\n2025 gene_names.csv: {len(g)} genes, e.g. {g[:5]}")


@app.function(
    image=eval_image, volumes={VOL_PATH: VOL},
    cpu=CPU_CORES, memory=16384, timeout=60 * 30,
)
def inspect():
    """Re-run the schema report without re-downloading."""
    inspect_local()


@app.function(
    image=eval_image, volumes={VOL_PATH: VOL},
    cpu=1.0, memory=2048, timeout=60 * 10,
)
def ls():
    """What the Volume currently holds, and what it costs us."""
    import os

    for root in (WORK, REHEARSAL, f"{VOL_PATH}/controls",
                 f"{VOL_PATH}/controls_final"):
        if not os.path.isdir(root):
            print(f"{root}: (absent)")
            continue
        tot = 0
        print(f"{root}:")
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if os.path.isfile(p):
                sz = os.path.getsize(p)
                tot += sz
                print(f"   {sz/1e9:8.2f} GB  {name}")
        print(f"   {'-'*8}")
        print(f"   {tot/1e9:8.2f} GB  total")


@app.local_entrypoint()
def main(dry_run: bool = False, cleanup_stale: bool = False,
         list_only: bool = False):
    if list_only:
        ls.remote()
        return
    fetch.remote(dry_run, cleanup_stale)
