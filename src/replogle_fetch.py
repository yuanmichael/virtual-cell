"""Fetch the Replogle single-cell screens to the Volume.

Why these two files and not the others. Arc's released STATE checkpoints were
trained on `replogle_nogwps_v2` -- Replogle WITHOUT the genome-wide screen
(confirmed from `external/state_ckpt_meta/zeroshot__k562__config.yaml`). Their
perturbation vocabulary is 2,024 essential genes with ZERO overlap with the
2026 target list, while the genome-wide screen covers 277 of our 300 targets.
So the genome-wide screen is simultaneously the part of Replogle that matters
most to us and the part no released checkpoint has seen.

  K562_gwps_raw_singlecell_01.h5ad  65.8 GB  ~1.99M cells, 8,248 genes
  rpe1_raw_singlecell_01.h5ad        8.7 GB  ~249k cells,  8,749 genes

RPE1 comes along because it is a SECOND cell line. Our 2025 training data is
one cell type, so the model has never seen the same perturbation in two
contexts -- which is literally what the challenge asks it to generalise over.

Both files are stored DENSE (65.8e9 / (8248 x 4) = 1.99M rows exactly), so the
download is also the decompression; expect no shrinkage on disk.

Downloads are resumable (HTTP Range against the partial file) and committed to
the Volume as they go, so a preempted container continues rather than
restarting 74 GB.

Run:
    modal run src/replogle_fetch.py --dry-run     # sizes + reachability only
    modal run --detach src/replogle_fetch.py
"""

import modal

from common import CPU_CORES, VOL, VOL_PATH, app, eval_image

REPLOGLE = f"{VOL_PATH}/replogle"

# Figshare+ article 20029387, "Mapping information-rich genotype-phenotype
# landscapes with genome-scale Perturb-seq" (Replogle et al. 2022), processed
# datasets. File ids resolved from the Figshare API rather than hard-coded
# download URLs, which rotate.
ARTICLE = 20029387
WANTED = [
    "rpe1_raw_singlecell_01.h5ad",          # smaller: fetch first, fail fast
    "K562_gwps_raw_singlecell_01.h5ad",
]

CHUNK = 1 << 24  # 16 MiB; large enough that per-request overhead vanishes
COMMIT_EVERY = 1 << 31  # commit the Volume every 2 GiB of progress


@app.function(
    image=eval_image,
    volumes={VOL_PATH: VOL},
    cpu=CPU_CORES,
    memory=8192,
    timeout=60 * 60 * 10,
)
def fetch(dry_run: bool = False, only: str = ""):
    import os
    import time

    import requests

    os.makedirs(REPLOGLE, exist_ok=True)

    r = requests.get(
        f"https://api.figshare.com/v2/articles/{ARTICLE}/files",
        params={"page_size": 50}, timeout=60,
    )
    r.raise_for_status()
    avail = {f["name"]: f for f in r.json()}
    print(f"article {ARTICLE} lists {len(avail)} files")

    want = [w for w in WANTED if not only or w in only.split(",")]
    plan = []
    for name in want:
        if name not in avail:
            raise KeyError(f"{name} not in article: {sorted(avail)}")
        f = avail[name]
        have = os.path.getsize(f"{REPLOGLE}/{name}") \
            if os.path.exists(f"{REPLOGLE}/{name}") else 0
        plan.append((name, f, have))
        print(f"  {name}: {f['size']/1e9:.2f} GB total, "
              f"{have/1e9:.2f} GB already local")

    todo = sum(f["size"] - have for _, f, have in plan)
    print(f"\nto download: {todo/1e9:.2f} GB")
    if dry_run:
        # A ranged GET, not HEAD: figshare's signed URLs answer HEAD with 403
        # while serving 206 Partial Content for the same URL, so HEAD would
        # have condemned a path that works.
        for name, f, _ in plan:
            h = requests.get(f["download_url"], headers={"Range": "bytes=0-1023"},
                             stream=True, timeout=60)
            print(f"  {name}: GET range {h.status_code}, "
                  f"content-range={h.headers.get('content-range')}")
            h.close()
        return None

    for name, f, _ in plan:
        path = f"{REPLOGLE}/{name}"
        total = f["size"]
        t0 = time.time()
        while True:
            have = os.path.getsize(path) if os.path.exists(path) else 0
            if have >= total:
                print(f"[{name}] complete ({have/1e9:.2f} GB)")
                break
            # Range resume. A Volume commit mid-download leaves a valid
            # prefix, so a preempted container picks up from the byte count
            # rather than from zero.
            headers = {"Range": f"bytes={have}-"} if have else {}
            print(f"[{name}] {have/1e9:.2f}/{total/1e9:.2f} GB, "
                  f"requesting from byte {have}", flush=True)
            try:
                with requests.get(f["download_url"], headers=headers,
                                  stream=True, timeout=(60, 600)) as resp:
                    if have and resp.status_code != 206:
                        raise RuntimeError(
                            f"server ignored Range (got {resp.status_code}); "
                            f"refusing to append and corrupt the file"
                        )
                    resp.raise_for_status()
                    since = 0
                    with open(path, "ab" if have else "wb") as fh:
                        for chunk in resp.iter_content(CHUNK):
                            fh.write(chunk)
                            have += len(chunk)
                            since += len(chunk)
                            if since >= COMMIT_EVERY:
                                fh.flush()
                                os.fsync(fh.fileno())
                                VOL.commit()
                                since = 0
                                el = time.time() - t0
                                print(f"  [{name}] {have/1e9:.2f}/"
                                      f"{total/1e9:.2f} GB  "
                                      f"{have/1e6/max(el,1):.0f} MB/s  "
                                      f"{el/60:.0f} min", flush=True)
            except Exception as e:  # noqa: BLE001 - retry the range, don't restart
                print(f"  [{name}] {type(e).__name__}: {e} -- retrying")
                VOL.commit()
                time.sleep(10)
                continue
        VOL.commit()
        got = os.path.getsize(path)
        print(f"[{name}] {got/1e9:.2f} GB in {(time.time()-t0)/60:.0f} min")
        if got != total:
            raise RuntimeError(f"{name}: got {got} bytes, expected {total}")

    print(f"\n{REPLOGLE} now holds:")
    for n in sorted(os.listdir(REPLOGLE)):
        print(f"  {os.path.getsize(f'{REPLOGLE}/{n}')/1e9:8.2f} GB  {n}")
    return REPLOGLE


@app.function(
    image=eval_image, volumes={VOL_PATH: VOL}, cpu=4.0, memory=16384,
    timeout=60 * 30,
)
def inspect(only: str = ""):
    """Shapes, obs columns and perturbation counts, before anything is built.

    The bulk files told us the gene space (8,248 / 8,749) and the column names
    (`gene_transcript`); this confirms the single-cell files use the same
    conventions, because `state_train.py` has to be told the column names and
    getting them wrong trains on the wrong thing silently.
    """
    import os

    import h5py
    import numpy as np

    for n in sorted(os.listdir(REPLOGLE)):
        if not n.endswith(".h5ad"):
            continue
        # `only` exists so a finished file can be inspected while another is
        # still downloading -- h5py cannot open a truncated file, so without
        # the filter this crashes on the partial one.
        if only and only not in n:
            continue
        p = f"{REPLOGLE}/{n}"
        print(f"\n=== {n} ({os.path.getsize(p)/1e9:.2f} GB) ===")
        with h5py.File(p, "r") as h:
            X = h["X"]
            enc = dict(X.attrs) if hasattr(X, "attrs") else {}
            if isinstance(X, h5py.Dataset):
                print(f"X: DENSE {X.shape} {X.dtype}")
            else:
                print(f"X: sparse {enc.get('encoding-type')} "
                      f"shape {tuple(enc.get('shape', ()))}")
            print(f"obs columns: {sorted(h['obs'].keys())}")
            print(f"var columns: {sorted(h['var'].keys())}")
            # These files use the pre-0.7 anndata layout: int codes in
            # obs/<key> with the levels in a SIBLING obs/__categories group.
            # Decoding only the modern layout yields integers rendered as
            # strings, which would make every perturbation label a number.
            def levels(side, key):
                d = side[key]
                if isinstance(d, h5py.Group) and "categories" in d:
                    return _decode(d["categories"][:]), "modern"
                if (d.dtype.kind in "iu" and "__categories" in side
                        and key in side["__categories"]):
                    return _decode(side["__categories"][key][:]), "legacy"
                return None, f"plain {d.dtype}"

            for key in ("gene", "gene_transcript", "gem_group", "cell_line",
                        "sgID_AB"):
                if key in h["obs"]:
                    cats, how = levels(h["obs"], key)
                    if cats is None:
                        print(f"  obs/{key}: {how} {h['obs'][key].shape}")
                    else:
                        print(f"  obs/{key}: {how} categorical, "
                              f"{len(cats)} levels, e.g. {cats[:5]}")
                        ctrl = [c for c in cats
                                if "non-targeting" in c or "control" in c.lower()]
                        if ctrl:
                            codes = h["obs"][key][:]
                            ci = [i for i, c in enumerate(cats) if c in ctrl]
                            n = int(np.isin(codes, ci).sum())
                            print(f"    control levels {ctrl} -> "
                                  f"{n:,} cells ({n/len(codes)*100:.1f}%)")
            if "gene_name" in h["var"]:
                cats, how = levels(h["var"], "gene_name")
                if cats is None:
                    print(f"  var/gene_name: {how}")
                else:
                    codes = h["var"]["gene_name"][:] if how == "legacy" else None
                    names = ([cats[c] for c in codes[:5]] if codes is not None
                             else cats[:5])
                    print(f"  var/gene_name: {how}, e.g. {names}")


@app.local_entrypoint()
def main(dry_run: bool = False, only: str = "", do_inspect: bool = False):
    if do_inspect:
        inspect.remote(only)
        return
    fetch.remote(dry_run, only)
