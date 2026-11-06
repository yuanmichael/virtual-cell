"""Train STATE on Arc's 2025 data with ESM-2 perturbation embeddings.

The one substantive change from Arc's own recipe is the perturbation
representation. STATE ships a one-hot over a closed vocabulary, which covers
0 of the 300 2026 targets and cannot represent a gene it never saw. We hand
cell-load a `{gene: ESM-2 vector}` file instead, so related genes sit near each
other and an unseen target is an interpolation rather than a blank.

Everything else follows their defaults, which already suit us: `output_space:
all` emits the full gene space (not the 2,000 HVGs the released checkpoints are
stuck with), and `embed_key: null` trains straight off gene counts so we need
no separate SE embedding model.

Two settings that are easy to get wrong and would quietly ruin the run:
  * `log1p_from_raw_counts=True` -- the 2025 h5ads are RAW counts, and the
    default assumes pre-normalised input.
  * the column names: Arc's defaults are Replogle's (`gene`, `gem_group`,
    `DMSO_TF`), not the challenge's (`target_gene`, `batch`, `non-targeting`).

Checkpoints land on the Volume and a committer thread flushes them every few
minutes, so a preempted container resumes instead of restarting.

Run:
    modal run --detach src/state_train.py --max-steps 200 --tag smoke
    modal run --detach src/state_train.py --max-steps 30000 --tag v1
"""

import modal

from common import REHEARSAL, VOL, VOL_PATH, app, state_image

STATE_ROOT = f"{VOL_PATH}/state"
# The log1p(CP10k) copies from src/state_norm.py. Training on the raw-count
# originals put the model in count space, where `state tx infer`'s [0, 14]
# output clip saturates every highly expressed gene -- right genes, wrong
# magnitudes, de_wilcoxon_lfc_nmae pinned at its penalty cap.
TOML_PATH = f"{STATE_ROOT}/vcc2025_norm.toml"
PERT_FEATURES = f"{REHEARSAL}/pert_esm2.pt"
RUNS = f"{STATE_ROOT}/runs"
# Container-local scratch for the training data. NOT on the Volume.
STAGE = "/staging"


def _rewrite_contiguous(src, dst, block=1 << 24):
    """Copy an h5ad, re-laying X out for RANDOM ROW reads instead of bulk scans.

    This is the single biggest lever on training speed, and the measurement
    that found it is worth keeping:

      GPU compute per batch   0.046 s
      wall clock per step    10.4   s      -> the GPU was idle 99.6% of the time

    One step needs 1,024 cells chosen at random from 2M. src/replogle_prep.py
    wrote X with `chunks=(1<<20,)` and `compression="lzf"`, and an HDF5 chunk is
    the atomic unit of compression: touching one value costs the whole chunk.
    A chunk holds ~391 cells, and 1,024 random cells land in ~927 distinct
    chunks, so each batch decompressed ~7.4 GB to use ~22 MB -- about 340x read
    amplification. At ~700 MB/s that is 10.6 s, which is the 10.4 s/step we
    measured, so this accounts for essentially all of it.

    Writing the three CSR arrays CONTIGUOUS and UNCOMPRESSED makes a row read a
    direct byte-range read of ~11 KB, so a batch moves ~22 MB instead of
    7.4 GB. Costs ~50% more disk, which local SSD has.

    Everything that is not X is copied verbatim.
    """
    import h5py
    import numpy as np

    with h5py.File(src, "r") as si, h5py.File(dst, "w") as so:
        for k, v in si.attrs.items():
            so.attrs[k] = v
        for key in si:
            if key != "X":
                si.copy(key, so, name=key)

        X = si["X"]
        if not isinstance(X, h5py.Group):
            si.copy("X", so, name="X")
            return
        go = so.create_group("X")
        for k, v in X.attrs.items():
            go.attrs[k] = v
        for name in ("data", "indices", "indptr"):
            d = X[name]
            n = d.shape[0]
            # No `chunks=` and no `compression=` -> contiguous layout, so HDF5
            # reads exactly the slice asked for.
            out = go.create_dataset(name, shape=(n,), dtype=d.dtype)
            step = max(block // max(d.dtype.itemsize, 1), 1)
            for i in range(0, n, step):
                j = min(i + step, n)
                out[i:j] = d[i:j]
            print(f"    X/{name}: {n:,} {d.dtype} contiguous", flush=True)


def _stage(toml_path, max_val_perts=0):
    """Copy the dataset onto container-local disk and rewrite the TOML at it.

    Why this exists. One training step needs batch_size x cell_set_len = 1,024
    cells drawn at RANDOM from a 2M-row CSR matrix. Served from a Modal Volume
    that is network-backed, so every step pays network latency per read, and
    the first Replogle smoke ran at ~81 s/step wall clock. The same step is
    ~0.03 s of arithmetic (metrics.csv: 6.0e11 FLOPs/step against an A10G's
    ~2e13 FLOPs/s), so the run was ~3 orders of magnitude off compute-bound and
    25,000 steps would have taken ~83 hours instead of ~14.

    One sequential copy to local SSD turns every later random read local.

    `max_val_perts` trims the validation perturbation list: validating on all
    300 held-out perturbations means 133,849 cells per pass, which dominated
    the smoke's hour. Perturbations dropped from val return to training.
    """
    import os
    import shutil
    import time
    import tomllib

    with open(toml_path, "rb") as fh:
        cfg = tomllib.load(fh)

    datasets = cfg.get("datasets", {})
    if len(datasets) != 1:
        raise RuntimeError(f"expected one dataset, got {list(datasets)}")
    name, src_dir = next(iter(datasets.items()))

    os.makedirs(STAGE, exist_ok=True)
    dst_dir = f"{STAGE}/{name}"
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".h5ad"))
    # x1.7, not x1.15: the staged copy is written UNCOMPRESSED, so it is larger
    # than the source rather than the same size.
    need = sum(os.path.getsize(f"{src_dir}/{f}") for f in files) * 1.7
    free = shutil.disk_usage(STAGE).free
    print(f"staging {len(files)} files, ~{need/1e9:.1f} GB uncompressed; "
          f"{free/1e9:.1f} GB free on local disk")
    if free < need * 1.15:
        # Better to run slowly off the Volume than to die out of disk after an
        # hour of copying.
        print("NOT ENOUGH LOCAL DISK -- falling back to the Volume path")
        return toml_path

    for f in files:
        d = f"{dst_dir}/{f}"
        s = f"{src_dir}/{f}"
        if os.path.exists(d) and os.path.getsize(d) > os.path.getsize(s):
            # Bigger than the source is the signature of a completed rewrite:
            # dropping compression makes it grow.
            print(f"  {f}: already staged uncompressed "
                  f"({os.path.getsize(d)/1e9:.2f} GB)")
            continue
        t0 = time.time()
        _rewrite_contiguous(s, d)
        el = time.time() - t0
        print(f"  {f}: {os.path.getsize(s)/1e9:.2f} -> "
              f"{os.path.getsize(d)/1e9:.2f} GB in {el:.0f}s", flush=True)

    if max_val_perts:
        for key, spl in cfg.get("fewshot", {}).items():
            if "val" in spl and len(spl["val"]) > max_val_perts:
                keep = sorted(spl["val"])[:max_val_perts]
                print(f"  val for {key}: {len(spl['val'])} -> {len(keep)} "
                      f"perturbations (the rest return to training)")
                spl["val"] = keep

    def emit(v):
        if isinstance(v, list):
            return "[\n" + "".join(f'  "{x}",\n' for x in v) + "]"
        return f'"{v}"'

    out = [f"# Generated at runtime by state_train._stage from {toml_path}.",
           "", "[datasets]", f'{name} = "{dst_dir}"', ""]
    for sect in ("training", "zeroshot"):
        out.append(f"[{sect}]")
        for k, v in cfg.get(sect, {}).items():
            out.append(f'"{k}" = {emit(v)}' if "." in k else f"{k} = {emit(v)}")
        out.append("")
    out.append("[fewshot]")
    out.append("")
    for key, spl in cfg.get("fewshot", {}).items():
        out.append(f'[fewshot."{key}"]')
        for k, v in spl.items():
            out.append(f"{k} = {emit(v)}")
        out.append("")
    staged = f"{STAGE}/{os.path.basename(toml_path)}"
    with open(staged, "w") as fh:
        fh.write("\n".join(out))
    print(f"staged TOML: {staged}")
    return staged


@app.function(
    image=state_image,
    volumes={VOL_PATH: VOL},
    gpu="A10G",
    cpu=8.0,
    memory=32768,
    # 24 h, not 6. The previous Replogle run was killed by a 6 h timeout at
    # step ~3,149 having written NO checkpoint, because checkpoints are tied to
    # validation and val_freq was 5,000. Three settings had to agree and did
    # not; the timeout is the cheapest of the three to make generous.
    timeout=60 * 60 * 24,
)
def train(
    max_steps: int = 200,
    tag: str = "smoke",
    cell_set_len: int = 64,
    hidden_dim: int = 768,
    batch_size: int = 16,
    lr: float = 1e-4,
    val_freq: int = 100,
    ckpt_every: int = 200,
    num_workers: int = 4,
    resume: bool = True,
    init_ckpt: str = "",
    toml_path: str = "",
    pert_features: str = "",
    cell_type_key: str = "cell_type",
    stage_local: bool = False,
    max_val_perts: int = 0,
):
    import os
    import subprocess
    import threading
    import time

    # The Replogle route supplies its own TOML (src/replogle_prep.py), its own
    # ~10,000-gene ESM-2 file (src/state_perts.py --include-replogle) and
    # `cell_line` rather than `cell_type` as the context column. Everything
    # else about the recipe is unchanged, which is the point: one variable at a
    # time.
    toml = toml_path or TOML_PATH
    perts = pert_features or PERT_FEATURES
    for p in (toml, perts):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} missing -- run src/state_prep.py / src/replogle_prep.py "
                f"/ src/state_perts.py"
            )

    if stage_local:
        toml = _stage(toml, max_val_perts)

    import torch

    feats = torch.load(perts)
    pert_dim = next(iter(feats.values())).shape[0]
    print(f"perturbation features: {len(feats)} genes x {pert_dim} dims")
    nonzero = sum(1 for v in feats.values() if float(v.abs().sum()) > 0)
    print(f"  non-zero: {nonzero}/{len(feats)}  "
          f"(a zero vector means 'no perturbation' to the model)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    outdir = f"{RUNS}/{tag}"
    os.makedirs(outdir, exist_ok=True)

    # Keep the Volume's view of the checkpoints fresh without waiting for the
    # process to exit -- Modal preempts, and an uncommitted checkpoint is lost.
    stop = threading.Event()

    def committer():
        while not stop.wait(180):
            try:
                VOL.commit()
            except Exception as e:  # noqa: BLE001 - never kill training
                print(f"[commit] {type(e).__name__}: {e}")

    t = threading.Thread(target=committer, daemon=True)
    t.start()

    overrides = [
        f"data.kwargs.toml_config_path={toml}",
        f"data.kwargs.perturbation_features_file={perts}",
        # src/replogle_prep.py renames Replogle's columns to these same names
        # on the way in, so one set of overrides serves both datasets. Only the
        # context column differs.
        "data.kwargs.pert_col=target_gene",
        "data.kwargs.batch_col=batch",
        f"data.kwargs.cell_type_key={cell_type_key}",
        "data.kwargs.control_pert=non-targeting",
        # Full 18,080-gene output, straight from counts.
        "data.kwargs.output_space=all",
        "data.kwargs.embed_key=null",
        f"data.kwargs.num_workers={num_workers}",
        f"model.kwargs.cell_set_len={cell_set_len}",
        f"model.kwargs.hidden_dim={hidden_dim}",
        # The 2025 h5ads hold RAW counts. cell-load assumes log1p input by
        # default, so say otherwise and let it normalise. These two keys are
        # read from **kwargs and are absent from the hydra schema, hence the
        # `+` prefix, which appends rather than overrides.
        # (arc-state 0.11.3 moved this into the model as
        # `log1p_from_raw_counts`, but 0.11.1 is the newest on PyPI and has no
        # such key -- overriding it fails with "not in struct".)
        "+data.kwargs.is_log1p=true",
        "model.kwargs.batch_encoder=True",
        f"training.batch_size={batch_size}",
        f"training.lr={lr}",
        f"training.max_steps={max_steps}",
        f"training.val_freq={val_freq}",
        f"training.ckpt_every_n_steps={ckpt_every}",
        "training.wandb_track=false",
        "use_wandb=false",
        f"name={tag}",
        f"output_dir={outdir}",
        # Set below: overwriting the output dir would delete the very
        # checkpoint a resumed run needs to warm-start from.
    ]

    # `overwrite=true` can clear output_dir, so a continuation run gets its own
    # tag and is pointed at the previous run's checkpoint explicitly, rather
    # than resuming in place and risking the weights it needs.
    start = init_ckpt or f"{outdir}/{tag}/checkpoints/last.ckpt"
    if init_ckpt and not os.path.exists(init_ckpt):
        raise FileNotFoundError(f"--init-ckpt {init_ckpt} does not exist")
    if resume and os.path.exists(start):
        # `model.kwargs.init_from`, NOT `model.checkpoint`. The latter is read
        # at inference only: a previous "continuation" run passed it and
        # silently restarted from random init (train loss 78.1 at step 49
        # versus 77.4 for the original). `init_from` is the field Arc's own
        # released configs carry for warm starts.
        #
        # This is VERIFIABLE from the logs: a genuine warm start shows the
        # first logged train_loss near where the previous run left off, not at
        # the random-init value. Check that before trusting a resumed run.
        overrides.append("overwrite=false")
        print(f"warm-starting weights from: {start}")
        print("  (verify: first logged train_loss should resume near the "
              "previous run's, not jump to the random-init value)")
        overrides.append(f"model.kwargs.init_from={start}")
    else:
        overrides.append("overwrite=true")
        print("starting from random init")

    cmd = ["state", "tx", "train"] + overrides
    print("\n" + " \\\n  ".join(cmd) + "\n", flush=True)

    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=outdir, text=True)
    finally:
        stop.set()
        VOL.commit()

    print(f"\nexit {r.returncode} after {time.time()-t0:.0f}s")
    for root, _, files in os.walk(outdir):
        for f in sorted(files):
            if f.endswith((".ckpt", ".yaml", ".csv")):
                p = os.path.join(root, f)
                print(f"  {os.path.getsize(p)/1e6:9.1f} MB  "
                      f"{p[len(outdir)+1:]}")
    if r.returncode != 0:
        raise RuntimeError(f"state tx train failed ({r.returncode})")
    return outdir


@app.local_entrypoint()
def main(max_steps: int = 200, tag: str = "smoke", cell_set_len: int = 64,
         hidden_dim: int = 768, batch_size: int = 16, lr: float = 1e-4,
         val_freq: int = 100, ckpt_every: int = 200, num_workers: int = 4,
         resume: bool = True, init_ckpt: str = "", block: bool = False,
         replogle: bool = False, toml_path: str = "",
         pert_features: str = "", cell_type_key: str = "",
         stage_local: bool = False, max_val_perts: int = 0):
    # `--replogle` is shorthand for the three paths the Replogle route needs,
    # so the long form cannot be half-applied (a run with Replogle's TOML but
    # the 2025 perturbation file would train on 572 vectors and silently zero
    # the other ~9,300 perturbations).
    if replogle:
        toml_path = toml_path or f"{VOL_PATH}/state/replogle.toml"
        pert_features = pert_features or f"{REHEARSAL}/pert_esm2_all.pt"
        cell_type_key = cell_type_key or "cell_line"
        # Replogle is 28 GB of random reads per epoch; off the Volume that is
        # ~81 s/step. Staging is not optional for this dataset.
        stage_local = True
        max_val_perts = max_val_perts or 50
        # val_freq, NOT ckpt_every, is what actually produces a checkpoint:
        # saves are tied to validation, so the previous run set ckpt_every=2500
        # with val_freq=5000 and wrote nothing at all before its timeout.
        # Keep them equal so "checkpoint every N steps" means what it says.
        ckpt_every = val_freq
    args = (max_steps, tag, cell_set_len, hidden_dim, batch_size, lr,
            val_freq, ckpt_every, num_workers, resume, init_ckpt,
            toml_path, pert_features, cell_type_key or "cell_type",
            stage_local, max_val_perts)
    if block:
        train.remote(*args)
        return
    # spawn(), not remote() + --detach. `--detach` survives a client
    # DISCONNECT but not the local process being signalled: when the shell
    # running `modal run --detach` was killed, the remote job was cancelled
    # with it and a 12,000-step run died at step 1,349. spawn() enqueues the
    # call server-side and returns a handle, so no local process is load-bearing.
    call = train.spawn(*args)
    print(f"spawned: {call.object_id}")
    print(f"  poll:  modal call-logs {call.object_id}")
    print(f"  files: modal volume ls vcc-data state/runs/{tag}/{tag}/checkpoints")
