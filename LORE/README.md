<div align="center">
  <img src="../assets/MIMIC_logo.png" alt="MIMIC" width="320">
</div>

# LORE

**LORE** is the multimodal central-dogma dataset that [MIMIC](https://huggingface.co/polymathic-ai/MIMIC)
was trained on: 60.3 million rows — **25.4 million distinct molecules** — pairing DNA/RNA
sequence, protein sequence and structure, and a range of functional assays, over 26
modalities, across 6,283 species.

Rows outnumber molecules 2.38:1 because a molecule assayed under several conditions is
emitted once per condition. Quote `n_unique`, not `n_rows`, when you report a size.

This directory holds the tooling and the build pipeline. The data itself is on the
Hugging Face Hub.

| | |
|:-|:-|
| Dataset | [`polymathic-ai/LORE`](https://huggingface.co/datasets/polymathic-ai/LORE) — 447.6 GB, 4,396 parquet shards |
| Index | [`polymathic-ai/LORE-index`](https://huggingface.co/datasets/polymathic-ai/LORE-index) — 1.6 GB, query this first |
| Model | [`polymathic-ai/MIMIC`](https://huggingface.co/polymathic-ai/MIMIC) |
| Small sample | [`polymathic-ai/LORE-examples`](https://huggingface.co/datasets/polymathic-ai/LORE-examples) — 44 rows, no download needed |

## Install

```bash
pip install "git+https://github.com/PolymathicAI/MIMIC.git#subdirectory=LORE"
```

Deliberately light — no torch. Install [`mimic`](https://github.com/PolymathicAI/MIMIC)
alongside if you want to turn token ids back into sequences.

## Quickstart: find your data without learning the layout

```python
from lore import LoreIndex

idx = LoreIndex.from_hf(root="/path/to/LORE")   # index only; 1.6 GB
idx.describe()                                   # every modality, and how many rows have it

# Every human transcript that has both a sequence and splice-site labels.
sel = idx.select(["rna_seq", "splice_jctns_5cls"], species="Homo sapiens")
print(sel.n_rows, sel.n_unique)                  # rows vs distinct molecules

# Splits that no homologue can straddle.
train, val, test = sel.make_splits(key="rna_cluster_30", seed=0)

# Read the payload for a handful of rows.
table = sel.head(64).to_table(columns=["rna_seq", "splice_jctns_5cls"])
```

`select()` figures out which shards to read and reports honest counts. You never need to
know which subset directory anything lives in.

## How the data is laid out, and why it bites

LORE ships as **40 subset directories**, each holding samples that carry the same group of
modalities, flat within the subset:

```
data/<subset>/data_0000.parquet
```

The internal build has a `<subset>/<split>/` tree; the release deliberately flattens it
and renumbers, for the reason in the next section. Filenames carry no information — the
index maps every row to its file.

Subsets exist because **no sample carries all 26 modalities** — coverage is sparse and
ragged, and training wants dense, null-free batches. Two consequences matter for anyone
analysing the data:

1. **A modality group spans several subsets.** There is no single directory containing
   "everything with `rna_seq`".
2. **Subsets are not disjoint.** `atac`, `cage`, `rasp2` and `prot_abund` are mutually
   exclusive *within* a subset, so a transcript assayed for two of them is emitted once
   per assay, into different subsets. In sampled shards, 22 of 276 subset pairs shared
   identifiers — one pair at 59.6% of the smaller set.

So a hand-rolled union over directories **double-counts molecules**, and a naive random
split can place the same molecule in both train and test. The index handles both: use
`n_unique` for counts, `deduplicate()` to collapse repeats, and `make_splits()` to split
by cluster.

## ⚠ MIMIC 1.0's partition is contaminated, so it is not a directory tree

**It is not suitable for evaluation.** Splits were assigned on
`COALESCE(protein_cluster_30, rna_cluster_30)`, which consults only one of the four
available clusterings, so homologues of training sequences appear in validation and test.
Measured against de-contaminated allowlists, **only 53.3% of checkable validation/test
rows are clean**, and a further 23.4% (the protein-anchored subsets) cannot be checked at
all because no protein-keyed allowlist exists.

**Settled:** the partition does not ship as directories, and does not ship in the
filenames either. A `train/val/test` tree is the strongest default there is — it is what
`load_dataset` hands back — and pointing it at a contaminated split while warning about it
in prose is not a real warning. A `train-` filename prefix is a weaker version of the same
problem.

So the published shards are renumbered to a plain per-subset counter
(`data_0000.parquet`), and the partition survives in exactly one place: the **`split`
column on the index**. `select(split="train")` reproduces MIMIC 1.0's training set
exactly, and logs a warning.

Renumbering is not optional, incidentally — `data_0.parquet` exists in all three splits,
so flattening without it would collide. `scripts/_layout.py` defines the mapping as a pure
function of the source listing, and both the tree builder and the index rewriter call it,
so they cannot drift.

Nothing is lost: the published model's training run stays reproducible, and the method is
published in full at `pipeline/02_data_selection/02_make_splits.py`. Reproducing it is an
index join rather than a directory read — a deliberate speed bump on the contaminated
path.

For evaluation, draw your own splits with `make_splits(key=...)`, which groups by a
sequence-cluster label so homologues cannot straddle the boundary.

**Cluster labels do not cover every row, and no single key covers the whole dataset.**
Measured over the index:

| key | rows labelled | coverage |
|:-|-:|-:|
| `rna_cluster_30` / `_70` | 46,096,269 | 76.4% |
| `protein_cluster_30` / `_70` | 30,751,613 | 51.0% |
| `root_id` | 46,339,762 | 76.8% |

The split is clean: rows are anchored either on a transcript or on a protein, and each
carries the labels native to its own anchor. Of 45.9M genome-anchored rows, 99.9% have
`rna_cluster_*`; of 13.9M protein-anchored rows, 100% have `protein_cluster_*` but only
1.5% have `rna_cluster_*`. The 501,381 `corpus` rows have no identifier at all.

This matters, because `make_splits` treats a row with a **null** key as a singleton
group. Singletons are never pooled into one bucket — but they also get **no homology
protection**. Splitting the whole index on `rna_cluster_30` therefore leaves the
protein-anchored quarter effectively randomly split, which is the same failure mode as
the shipped `split` column. `corpus` rows, having no identifier, all land in train.

So: **split each anchor on its own native key and concatenate.**

```python
prot = idx.select(["aa_seq", "prot_struct"])          # protein-anchored
rna  = idx.select(["rna_seq"])                        # transcript-anchored
p_tr, p_va, p_te = prot.make_splits(key="protein_cluster_30")
r_tr, r_va, r_te = rna.make_splits(key="rna_cluster_30")
```

Within an anchor this gives strictly more control than our splits offered, at whatever
stringency you want.

## Values are tokenized, and binned modalities are lossy

Payload columns hold **token ids**, not raw values. Sequence modalities (`rna_seq`,
`aa_seq`, `dssp`, `splice_jctns_5cls`, …) round-trip exactly through the `mimic`
tokenizers. Continuous assays — `phylop_*`, `sasa`, `rasp2`, `atac`, `cage`,
`prot_abund`, `masif_*` — were **discretized into bins**, so detokenizing returns bin
centres, not the upstream measurements. Go back to the source (see `NOTICE.md`) if you
need the original values.

```python
from mimic.modality_info import MODALITY_INFO
MODALITY_INFO["tok_rna_seq"]["tokenizer"].detokenize(row["rna_seq"])   # -> "ACGU..."
```

## What is in this directory

```
LORE/
  src/lore/            index tooling
    index.py             LoreIndex / Selection — query, dedupe, split, materialize
    build_index.py       build the index from a local copy of the data
    paths.py             $LORE_DATA_ROOT layout resolution
  configs/             the exact configs that produced the release
  pipeline/            the build pipeline, stages 00-03
  NOTICE.md            per-source attribution and licensing  <- read before redistributing
```

### Rebuilding the index

```bash
export LORE_DATA_ROOT=/path/to/lore
python -m lore.build_index --out ./lore_index \
    --master-ids $LORE_DATA_ROOT/data/intermediate/master_ids/3.0/parquet/dataset.parquet
```

Phase A scans identifiers per subset and is resumable; phase B attaches clusters and
taxonomy from the master-id table.

### Rebuilding the dataset from public sources

`pipeline/` covers the whole path from upstream downloads to tokenized shards:

| stage | what it does |
|:-|:-|
| `00_download` | fetch each upstream source |
| `01_process` | source → one parquet per modality |
| `02_data_selection` | cluster sequences, build the master-id table, assign splits, audit leakage |
| `03_merge` | find modality subsets, merge, tokenize, write parquet + webdataset shards |

Set `LORE_DATA_ROOT` and work through the stages in order. `configs/master_ids_3.0.yaml`
and `configs/merge_config_4.0.yaml` are the exact configurations behind this release —
note that clustering (mmseqs2 over UniRef90/AFDB and RefSeq) is the expensive step and
dominates the total cost.

Only the parquet is published. The webdataset shards MIMIC trained from are 13.8 TB of
the same content and are regenerated locally with
`pipeline/03_merge/04_write_webdataset.py`.

## Open decisions before publication

Tracked here so they are not silently resolved by whoever builds the release:

| decision | exposure if it goes the wrong way | status |
|:-|-:|:-|
| **FANTOM5 (`cage`) data licence** — not stated in the RIKEN datafiles tree | **16,779,046 rows / 27.8%**, 5 subsets | **unresolved** — needs a query to RIKEN |
| RASP2 (`rasp2`) redistribution terms | 1,723,411 rows / 2.9%, 8 subsets | **unresolved** — needs a query to the RASP authors |
| ~~Does `corpus` ship?~~ The one component with a CC-BY-SA share-alike fraction, and it carries no central-dogma content. | 501,381 rows / 0.8%, 1 subset | **settled** — omitted from the release |
| ~~Does the contaminated MIMIC 1.0 partition ship, ship separately, or get dropped?~~ | one column | **settled** — no directory tree; index column + filename prefix, opt-in (see above) |

FANTOM5 is the critical path: it is an order of magnitude more data than the other three
combined. **The fallback is cheap, though.** `cage` and `rasp2` are single columns, and
dropping a column rather than a subset keeps every row: 6 of the 10 de-`cage`d / de-`rasp2`d
subsets collapse onto subsets that already exist, so the rows merge in (dedupe on the
anchor id) and only the assay column is lost. Withholding whole subsets instead would cost 31.5% of the dataset, which
is not necessary.

Settled: the release is **tokenized only**, with `to_table(detokenize=True)` for
conversion on demand — a detokenized copy measured only ~7% larger, which does not
justify a second 450 GB artifact, and would label quantized assays as though they
were measurements.

## Citing

If you use LORE, cite MIMIC and the upstream resources for the modalities you use.
`NOTICE.md` lists them per modality.
