"""Build ESM-2 perturbation embeddings for STATE.

STATE's default perturbation representation is a one-hot over a closed
vocabulary (`pert_dim: 2024` in the released checkpoints), with ZERO overlap
with the 2026 target list. A one-hot has no slot for a gene it never saw, and
nothing ties related genes together, so it cannot generalise to a new target at
all -- `cell-load` fills unknown perturbations with a zero vector, which means
"no perturbation" and would reproduce our first submission's failure.

Replacing it is a supported config path, not a code change. From
`cell_load/data_modules/perturbation_dataloader.py`:

    if self.perturbation_features_file:
        featurization_dict = torch.load(self.perturbation_features_file)
        self.pert_onehot_map = featurization_dict
    ...
    pert_dim = next(iter(self.pert_onehot_map.values())).shape[0]

So we save `{gene_symbol: tensor}` and point the data config at it; `pert_dim`
follows automatically.

We embed each gene's PROTEIN with ESM-2, because what a knockdown does depends
on the protein's job in the cell. Genes with similar proteins land near each
other, which is exactly the structure that lets the model say something about a
target it never saw perturbed -- and it also denoises, by letting functionally
related genes inform each other instead of trusting one noisy measurement.

Only PERTURBED genes need embedding (the 300 2025 perturbations plus the 300
2026 targets, ~600 unique), not all 18,533 readout genes.

Run:
    modal run src/state_perts.py --dry-run     # resolve sequences only
    modal run src/state_perts.py
"""

import modal

from common import CONTROLS_FINAL, REHEARSAL, VOL, VOL_PATH, app, state_image

OUT = f"{REHEARSAL}/pert_esm2.pt"
SEQ_CACHE = f"{REHEARSAL}/pert_sequences.json"

# ESM-2 650M: 1280-dim embeddings, the usual accuracy/﻿cost point for this kind
# of featurisation. The 3B model is ~5x the compute for a marginal gain here,
# where we only need a similarity structure over ~600 proteins.
ESM_MODEL = "esm2_t33_650M_UR50D"
ESM_LAYER = 33
EMBED_DIM = 1280
MAX_LEN = 1022  # ESM-2's trained context; longer chains get truncated

UNIPROT = "https://rest.uniprot.org/uniprotkb/search"


def _target_lists():
    """Every gene symbol we will ever need a perturbation vector for."""
    import csv
    import os

    want = set()
    for name in ("pert_counts_Training.csv", "pert_counts_Validation.csv",
                 "pert_counts_Test.csv"):
        p = f"{REHEARSAL}/{name}"
        if os.path.exists(p):
            with open(p, newline="") as fh:
                want |= {r["target_gene"] for r in csv.DictReader(fh)}
    p = f"{CONTROLS_FINAL}/pert_counts.csv"
    if os.path.exists(p):
        with open(p, newline="") as fh:
            want |= {r["target_gene"] for r in csv.DictReader(fh)}
    want.discard("non-targeting")
    return sorted(want)


def _fetch_sequences(symbols, dry_run=False):
    """Canonical human protein sequence per gene symbol, from UniProt.

    Reviewed (Swiss-Prot) entries only, so we get one curated sequence per gene
    rather than an arbitrary isoform from the unreviewed pile.
    """
    import json
    import os
    import time

    import requests

    cache = {}
    if os.path.exists(SEQ_CACHE):
        with open(SEQ_CACHE) as fh:
            cache = json.load(fh)
        print(f"sequence cache: {len(cache)} already resolved")

    todo = [s for s in symbols if s not in cache]
    print(f"need sequences for {len(todo)} of {len(symbols)} genes")
    if dry_run:
        return cache, todo

    for i, sym in enumerate(todo):
        q = (f'(gene_exact:{sym}) AND (organism_id:9606) '
             f'AND (reviewed:true)')
        try:
            r = requests.get(
                UNIPROT,
                params={"query": q, "fields": "accession,sequence",
                        "format": "json", "size": 1},
                timeout=60,
            )
            r.raise_for_status()
            hits = r.json().get("results", [])
            if hits:
                cache[sym] = hits[0]["sequence"]["value"]
            else:
                cache[sym] = None  # remember the miss; don't retry every run
        except Exception as e:  # noqa: BLE001 - one bad symbol must not stop 600
            print(f"  {sym}: {type(e).__name__} {e}")
            continue
        if (i + 1) % 50 == 0:
            print(f"  resolved {i+1}/{len(todo)}")
            with open(SEQ_CACHE, "w") as fh:
                json.dump(cache, fh)
            VOL.commit()
        time.sleep(0.1)  # be polite to a free public API

    with open(SEQ_CACHE, "w") as fh:
        json.dump(cache, fh)
    VOL.commit()
    found = sum(1 for v in cache.values() if v)
    print(f"sequences: {found} found, {len(cache)-found} missing")
    return cache, todo


@app.function(
    image=state_image,
    volumes={VOL_PATH: VOL},
    gpu="A10G",
    cpu=4.0,
    memory=16384,
    timeout=60 * 60 * 2,
)
def build(dry_run: bool = False):
    import os

    import torch

    symbols = _target_lists()
    print(f"{len(symbols)} unique perturbation targets across 2025 + 2026")

    seqs, _ = _fetch_sequences(symbols, dry_run=dry_run)
    if dry_run:
        have = sum(1 for s in symbols if seqs.get(s))
        print(f"dry run: {have}/{len(symbols)} resolvable from cache")
        return None

    usable = [(s, seqs[s][:MAX_LEN]) for s in symbols if seqs.get(s)]
    print(f"embedding {len(usable)} proteins with {ESM_MODEL}")

    import esm

    model, alphabet = esm.pretrained.__dict__[ESM_MODEL]()
    model = model.eval().cuda()
    bc = alphabet.get_batch_converter()

    out = {}
    B = 8
    with torch.no_grad():
        for i in range(0, len(usable), B):
            chunk = usable[i:i + B]
            _, _, toks = bc([(s, q) for s, q in chunk])
            toks = toks.cuda()
            rep = model(toks, repr_layers=[ESM_LAYER])["representations"][ESM_LAYER]
            for j, (sym, q) in enumerate(chunk):
                # Mean over residues, skipping BOS and padding: a fixed-length
                # summary of the whole protein rather than one position.
                out[sym] = rep[j, 1:len(q) + 1].mean(0).float().cpu()
            if (i + B) % 80 == 0:
                print(f"  embedded {min(i+B, len(usable))}/{len(usable)}")

    # Genes UniProt could not resolve get an explicit zero vector, which is what
    # cell-load would substitute anyway -- but recorded here so the count is
    # visible rather than a surprise in the training log.
    missing = [s for s in symbols if s not in out]
    for s in missing:
        out[s] = torch.zeros(EMBED_DIM)
    if missing:
        print(f"zero vectors for {len(missing)} unresolved genes: "
              f"{', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}")

    torch.save(out, OUT)
    VOL.commit()
    dim = next(iter(out.values())).shape[0]
    print(f"\nwrote {OUT}: {len(out)} genes x {dim} dims "
          f"({os.path.getsize(OUT)/1e6:.1f} MB)")
    print(f"non-zero embeddings: {len(out)-len(missing)}")
    return OUT


@app.local_entrypoint()
def main(dry_run: bool = False):
    build.remote(dry_run)
