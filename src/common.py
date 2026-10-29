"""Shared Modal app, image and volume for the Virtual Cell Challenge 2026."""

import modal

VOL = modal.Volume.from_name("vcc-data", create_if_missing=True)
VOL_PATH = "/vol"

# Controls bundle lives at /vol/controls (uploaded with `modal volume put`).
CONTROLS = f"{VOL_PATH}/controls"
CONTROLS_FINAL = f"{VOL_PATH}/controls_final"
WORK = f"{VOL_PATH}/work"
# Arc's public 2025 challenge data, used as a scoreable stand-in for D/E/F.
REHEARSAL = f"{VOL_PATH}/rehearsal"

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.2.6",
        "scipy==1.15.3",
        "pandas==2.2.3",
        "h5py==3.14.0",
        "anndata==0.11.4",
        "vcc-cli==0.3.0",
    )
    # Modal 1.x no longer automounts local Python source.
    .add_local_python_source("common")
)

# Arc's own scorer, for running the vcc2026 metrics on the rehearsal set. Kept
# separate from cpu_image because it drags in scanpy/numba/sklearn and would
# slow every build/prep cold start for no benefit.
#
# `pdex` is the reason the pins here are newer than cpu_image's: cell-eval2
# picks it up automatically as its CPU differential-expression backend, and
# without it falls back to scanpy, which took ~90 min for one pass over the
# rehearsal set. pdex wants numpy>=2.4 and anndata>=0.12, so this image tracks
# the versions the local venv resolved to with pdex present.
eval_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.5.4",
        "scipy==1.18.1",
        "pandas==2.3.3",
        "h5py==3.14.0",
        "anndata==0.13.4",
        "cell-eval2==0.19.0",
        "pdex==0.3.0",
        "requests==2.32.3",
    )
    .add_local_python_source("common")
)

# Arc's STATE model. Pinned to python 3.12 because `cell-load` declares
# `<3.13` -- on 3.13 pip silently resolves to a 1.2 kB stub of the same name
# instead of the real package, which fails much later and confusingly.
#
# `arc-state` brings torch, transformers and peft. `fair-esm` is for the
# perturbation featuriser: STATE's default perturbation representation is a
# one-hot over a closed vocabulary, which cannot represent a gene it never saw
# in training, so we replace it with ESM-2 protein embeddings via cell-load's
# `perturbation_features_file` hook.
state_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        # 0.11.1 is the newest on PyPI; the checked-out source in
        # external/state is 0.11.3, so verify any config field we rely on at
        # runtime rather than trusting the source we read locally.
        "arc-state==0.11.1",
        "cell-load==0.10.4",
        "fair-esm==2.0.0",
        "requests==2.32.3",
    )
    .add_local_python_source("common")
)

# GPU shape for training. STATE at these hyper-parameters is small (hidden 328,
# 8 layers), so an A10G is enough; the cost driver is wall clock, not VRAM.
GPU_KIND = "A10G"

# Container shape for the CPU-only build/prep work. Peak RAM is one context's
# sparse matrix plus a copy (~11 GiB), so 24 GiB is ample; RAM is the dominant
# cost driver, so keep this tight.
CPU_CORES = 4.0
CPU_MEM_MIB = 24576

app = modal.App("vcc-2026")


def lam_per_target(mode, lam, thr, mag, covered):
    """Per-target blend weight, from how far the K562 measurement clears noise.

    `mag` is a target's mean |log2FC| over the readout divided by the
    non-targeting null band, so 1.0 means "indistinguishable from a
    control-vs-control comparison". Uncovered targets get 0: the pure generic
    response is worth ~0 against Arc's baseline, while a target's own noise is
    worth less than that.

    Shared by the rehearsal and the real submission so the dial we fit is
    literally the dial we ship.
    """
    import numpy as np

    out = np.zeros(len(mag), dtype=np.float64)
    if mode == "global":
        out[:] = lam
    elif mode == "gated":
        out[mag >= thr] = lam
    elif mode == "ramp":
        out[:] = lam * np.clip((mag - 1.0) / max(thr - 1.0, 1e-9), 0.0, 1.0)
    else:
        raise ValueError(f"unknown lam mode {mode!r}")
    out[~covered] = 0.0
    return out
