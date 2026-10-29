"""Put Arc's 2025 data into the layout cell-load expects, and write the TOML.

Two small obstacles and how this handles them.

1. cell-load needs a cell-type column (`cell_type_key`). The 2025 release has
   `target_gene`, `batch` and `guide_id` but no cell-type column, because every
   cell is the same line. Rather than rewrite 34 GB of counts to add one
   constant column, we splice the column straight into `/obs` with h5py and
   leave `X` untouched -- seconds instead of an hour.

2. cell-load assigns splits by `dataset.celltype`, which is awkward when there
   is only one cell type. The alternative it offers, `[fewshot]`, takes explicit
   perturbation lists, and that is what we actually have: the three 2025 splits
   use fully DISJOINT perturbations (150 / 50 / 100). So one dataset, one cell
   type, and the holdout is stated as gene lists -- which is honest about what
   is being held out (perturbations, not contexts) instead of inventing three
   fake cell types to trick the zeroshot path.

Run:
    modal run src/state_prep.py --dry-run
    modal run src/state_prep.py
"""

import modal

from common import REHEARSAL, VOL, VOL_PATH, VOL_PATH as _VP, app, cpu_image

STATE_ROOT = f"{_VP}/state"
TOML_PATH = f"{STATE_ROOT}/vcc2025.toml"

# cell-load globs `*.h5`/`*.h5ad` out of the dataset directory and ignores
# everything else, so we point it straight at the download directory. Copying
# the three h5ads into a dedicated dir duplicated 34.4 GB on the Volume for no
# benefit -- Modal Volumes do not support hard links, so os.link fell back to a
# real copy. The .npz/.pt/.csv files that live alongside are simply not matched.
DATA_DIR = REHEARSAL
STALE_COPIES = f"{STATE_ROOT}/data/vcc2025"

CELL_TYPE = "H1"          # the 2025 challenge is a single line; name it once
CONTROL_PERT = "non-targeting"

SPLITS = {
    "train": ("adata_Training.h5ad", "pert_counts_Training.csv"),
    "val": ("adata_Validation.h5ad", "pert_counts_Validation.csv"),
    "test": ("adata_Test.h5ad", "pert_counts_Test.csv"),
}


def _add_cell_type_inplace(path, label=CELL_TYPE):
    """Splice a constant categorical `cell_type` into /obs without touching X.

    anndata stores a categorical column as a group holding `categories` and
    `codes`, listed in the parent dataframe's `column-order`. Writing those
    three things by hand is far cheaper than a full round-trip through
    AnnData for a column that is the same value in every row.
    """
    import h5py
    import numpy as np

    with h5py.File(path, "r+") as h:
        obs = h["obs"]
        if "cell_type" in obs:
            print(f"    cell_type already present in {path.split('/')[-1]}")
            return False

        idx_key = obs.attrs.get("_index", "_index")
        n = obs[idx_key].shape[0]

        g = obs.create_group("cell_type")
        g.attrs["encoding-type"] = "categorical"
        g.attrs["encoding-version"] = "0.2.0"
        g.attrs["ordered"] = np.bool_(False)
        g.create_dataset(
            "categories",
            data=np.array([label], dtype=h5py.special_dtype(vlen=str)),
        )
        g.create_dataset("codes", data=np.zeros(n, dtype=np.int8))

        order = list(obs.attrs.get("column-order", []))
        order = [o.decode() if isinstance(o, bytes) else str(o) for o in order]
        if "cell_type" not in order:
            order.append("cell_type")
        obs.attrs["column-order"] = np.array(
            order, dtype=h5py.special_dtype(vlen=str)
        )
        print(f"    added cell_type='{label}' to {n} cells")
        return True


@app.function(
    image=cpu_image,
    volumes={VOL_PATH: VOL},
    cpu=4.0,
    memory=8192,
    timeout=60 * 60 * 2,
)
def prep(dry_run: bool = False):
    import csv
    import os

    os.makedirs(DATA_DIR, exist_ok=True)

    missing = [f for f, _ in SPLITS.values()
               if not os.path.exists(f"{REHEARSAL}/{f}")]
    if missing:
        raise FileNotFoundError(
            f"not downloaded yet: {missing}. Run src/rehearsal.py first."
        )

    perts = {}
    for split, (h5, csvname) in SPLITS.items():
        with open(f"{REHEARSAL}/{csvname}", newline="") as fh:
            perts[split] = sorted(
                {r["target_gene"] for r in csv.DictReader(fh)}
                - {CONTROL_PERT}
            )
        print(f"{split}: {len(perts[split])} perturbations")

    overlap = (set(perts["train"]) & set(perts["val"])) | \
              (set(perts["train"]) & set(perts["test"])) | \
              (set(perts["val"]) & set(perts["test"]))
    if overlap:
        raise ValueError(f"splits are not disjoint: {sorted(overlap)[:10]}")
    print("splits are disjoint -- a held-out perturbation is truly unseen")

    if dry_run:
        print(f"\nwould write {TOML_PATH} and link 3 h5ads into {DATA_DIR}")
        return None

    for split, (h5, _) in SPLITS.items():
        src = f"{REHEARSAL}/{h5}"
        print(f"  [{split}] {h5} ({os.path.getsize(src)/1e9:.2f} GB)")
        _add_cell_type_inplace(src)

    # Reclaim the duplicate copies an earlier version of this script made.
    if os.path.isdir(STALE_COPIES):
        import shutil
        freed = sum(
            os.path.getsize(os.path.join(STALE_COPIES, f))
            for f in os.listdir(STALE_COPIES)
            if os.path.isfile(os.path.join(STALE_COPIES, f))
        )
        shutil.rmtree(STALE_COPIES)
        print(f"  reclaimed {freed/1e9:.2f} GB of duplicate copies")
    VOL.commit()

    def fmt(names):
        return "[\n" + "".join(f'  "{n}",\n' for n in names) + "]"

    toml = f'''# Generated by src/state_prep.py -- do not hand-edit.
#
# One dataset, one cell type. The holdout is by PERTURBATION, stated
# explicitly, because Arc's three 2025 splits use disjoint perturbation lists
# ({len(perts["train"])} train / {len(perts["val"])} val / {len(perts["test"])} test).

[datasets]
vcc2025 = "{DATA_DIR}"

# Everything not named below trains.
[training]
vcc2025 = "train"

[zeroshot]

[fewshot]

[fewshot."vcc2025.{CELL_TYPE}"]
val = {fmt(perts["val"])}
test = {fmt(perts["test"])}
'''
    with open(TOML_PATH, "w") as fh:
        fh.write(toml)
    VOL.commit()
    print(f"\nwrote {TOML_PATH}")
    print(f"dataset dir holds: {sorted(os.listdir(DATA_DIR))}")
    return TOML_PATH


@app.local_entrypoint()
def main(dry_run: bool = False):
    prep.remote(dry_run)
