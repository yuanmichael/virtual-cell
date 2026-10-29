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


@app.function(
    image=state_image,
    volumes={VOL_PATH: VOL},
    gpu="A10G",
    cpu=8.0,
    memory=32768,
    timeout=60 * 60 * 6,
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
):
    import os
    import subprocess
    import threading
    import time

    for p in (TOML_PATH, PERT_FEATURES):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} missing -- run src/state_prep.py / src/state_perts.py"
            )

    import torch

    feats = torch.load(PERT_FEATURES)
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
        f"data.kwargs.toml_config_path={TOML_PATH}",
        f"data.kwargs.perturbation_features_file={PERT_FEATURES}",
        # The challenge's column names, not Replogle's.
        "data.kwargs.pert_col=target_gene",
        "data.kwargs.batch_col=batch",
        "data.kwargs.cell_type_key=cell_type",
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
        "overwrite=true",
    ]

    # `overwrite=true` can clear output_dir, so a continuation run gets its own
    # tag and is pointed at the previous run's checkpoint explicitly, rather
    # than resuming in place and risking the weights it needs.
    start = init_ckpt or f"{outdir}/{tag}/checkpoints/last.ckpt"
    if init_ckpt and not os.path.exists(init_ckpt):
        raise FileNotFoundError(f"--init-ckpt {init_ckpt} does not exist")
    if resume and os.path.exists(start):
        print(f"starting from weights: {start}")
        overrides.append(f"model.checkpoint={start}")
    else:
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
         resume: bool = True, init_ckpt: str = "", block: bool = False):
    args = (max_steps, tag, cell_set_len, hidden_dim, batch_size, lr,
            val_freq, ckpt_every, num_workers, resume, init_ckpt)
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
