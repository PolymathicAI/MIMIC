---
pretty_name: LORE
license: other
license_name: mixed-see-notice
license_link: https://github.com/PolymathicAI/MIMIC/blob/main/LORE/NOTICE.md
size_categories:
  - 10M<n<100M
task_categories:
  - feature-extraction
tags:
  - biology
  - genomics
  - proteins
  - rna
  - multimodal
  - central-dogma
configs:
  - config_name: rna_splice
    data_files:
      - split: all
        path: data/rna_seq+splice_jctns_5cls+splice_regions+gene_family_txt+is_coding+feature_type/*.parquet
  - config_name: protein_structure
    data_files:
      - split: all
        path: data/aa_seq+dssp+sasa+prot_struct/*.parquet
  - config_name: splice_prediction
    data_files:
      - split: all
        path: data/splice_prediction/*.parquet
---

# LORE

**LORE** is the multimodal central-dogma dataset that
[MIMIC](https://huggingface.co/polymathic-ai/MIMIC) was trained on: **59.8 million rows —
25.4 million distinct molecules** — pairing DNA/RNA sequence, protein sequence and
structure, and a range of functional assays over 26 modalities and 6,283 species.

> **⚠ Private preview.** This repository is a staging push for internal review. Two
> upstream licences are unresolved and the release is not cleared for publication. See
> *Status* below.

| | |
|:-|:-|
| Size | 445.1 GB, 4,393 parquet shards, 40 subsets |
| Rows | 59,839,026 |
| Distinct molecules | 25,388,998 (2.36 rows each) |
| Index | [`polymathic-ai/LORE-index`](https://huggingface.co/datasets/polymathic-ai/LORE-index) — 0.84 GB, **query this first** |
| Tooling | [`LORE/`](https://github.com/PolymathicAI/MIMIC/tree/main/LORE) |

## Start with the index, not the data

Do not download 445 GB to find out what is in it. The index is one row per data row,
carrying location, identifiers, sequence-cluster labels and taxonomy — no payload.

```python
from lore import LoreIndex

idx = LoreIndex.from_hf(root="/path/to/LORE")   # index only; 0.84 GB
idx.describe()

sel = idx.select(["rna_seq", "splice_jctns_5cls"], species="Homo sapiens")
print(sel.n_rows, sel.n_unique)                  # rows vs distinct molecules
table = sel.head(64).to_table()                  # read payload for 64 rows only
```

## Layout, and the two traps in it

```
data/<subset>/data_0000.parquet
```

Each of the 40 **subsets** holds samples carrying the same group of modalities. Subsets
exist because no sample carries all 26 modalities and training wants dense, null-free
batches. Two consequences:

1. **A modality spans several subsets.** There is no single directory holding
   "everything with `rna_seq`".
2. **Subsets are not disjoint.** `atac`, `cage`, `rasp2` and `prot_abund` are mutually
   exclusive *within* a subset, so a transcript assayed for two of them is emitted once
   per assay into different subsets. In sampled shards, 22 of 276 subset pairs shared
   identifiers — one pair at 59.6% of the smaller set.

So a hand-rolled union over directories **double-counts molecules**. Use `n_unique` and
`deduplicate()`.

## Values are token ids, and binned assays are lossy

Payload columns hold **token ids**, not raw values; the `mimic` tokenizers invert them.
Sequence modalities round-trip exactly. Continuous assays — `phylop_*`, `sasa`, `rasp2`,
`atac`, `cage`, `prot_abund`, `masif_*` — were discretized, so detokenizing returns **bin
centres, not measurements**. Go to the upstream source if you need the real values.
`prot_struct` is ESM3 VQVAE codebook indices and decodes to 3-D coordinates, not a scalar.

## ⚠ There are no train/val/test directories, on purpose

MIMIC 1.0's partition **is contaminated and is not suitable for evaluation.** Splits were
assigned on `COALESCE(protein_cluster_30, rna_cluster_30)`, consulting only one of four
clusterings, so homologues of training sequences appear in validation and test. Measured
against de-contaminated allowlists, **only 53.3% of checkable val/test rows are clean**,
and a further 23.4% cannot be checked at all.

Shipping that as a directory tree would make it the default anyone gets from
`load_dataset`. So the published data carries **no trace of it at all** — the filenames
are a plain counter per subset, deliberately information-free. The partition survives in
exactly one place, the **`split` column on the index**, where selecting it is opt-in and
warns:

```python
sel.select(split="train")     # reproduces MIMIC 1.0's training set; logs a warning
```

The method that produced it is published in full at
`pipeline/02_data_selection/02_make_splits.py`.

### Draw your own splits — but split each anchor on its own key

No single cluster key covers the dataset:

| key | coverage |
|:-|-:|
| `rna_cluster_30` / `_70` | 77.0% |
| `protein_cluster_30` / `_70` | 51.4% |
| `root_id` | 77.4% |

Rows are anchored either on a transcript or on a protein and carry the labels native to
their own anchor: of 45.9M genome-anchored rows 99.9% have `rna_cluster_*`; of 13.9M
protein-anchored rows 100% have `protein_cluster_*` but only 1.5% have `rna_cluster_*`.

`make_splits` treats a null key as a singleton — never pooled, but also **unprotected**.
Splitting everything on `rna_cluster_30` leaves the protein-anchored quarter effectively
randomly split, reproducing the very contamination above. So:

```python
prot = idx.select(["aa_seq", "prot_struct"])
rna  = idx.select(["rna_seq"])
p_tr, p_va, p_te = prot.make_splits(key="protein_cluster_30")
r_tr, r_va, r_te = rna.make_splits(key="rna_cluster_30")
```

## Composition

41.1% of rows are human and 14.4% mouse, with 6,281 further species behind them. The
remaining 23% are protein-anchored and carry no taxonomy label at all.

Historical MIMIC 1.0 partition, for reference only:

| split | rows | molecules |
|:-|-:|-:|
| train | 47,611,887 | 20,311,429 |
| validation | 6,258,261 | 2,538,459 |
| test | 5,968,878 | 2,539,110 |

The molecule counts sum exactly to 25,388,998, so no molecule appears in two splits: the
contamination is **homology**, not identity. Exact duplicates were already handled;
near-homologues were not.

## Status

| item | state |
|:-|:-|
| `corpus` subset (PubMed text) | **omitted** — CC-BY-SA share-alike fraction, no central-dogma content |
| FANTOM5 `cage` licence | **unresolved** — 27.8% of rows; query pending with RIKEN |
| RASP2 `rasp2` licence | **unresolved** — 2.9% of rows; query pending with the authors |
| MIMIC 1.0 partition | index column only; not a directory tree, not in the filenames |

Sources and per-modality attribution are in
[`NOTICE.md`](https://github.com/PolymathicAI/MIMIC/blob/main/LORE/NOTICE.md). LORE is a
derived work; AlphaFold DB, UniProt and PaxDb components are CC-BY-4.0 and require
attribution. Cite MIMIC and the upstream resources for whichever modalities you use.
