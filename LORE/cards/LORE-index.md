---
pretty_name: LORE index
license: other
license_name: mixed-see-notice
license_link: https://github.com/PolymathicAI/MIMIC/blob/main/LORE/NOTICE.md
size_categories:
  - 10M<n<100M
tags:
  - biology
  - genomics
  - proteins
  - metadata
  - index
configs:
  - config_name: default
    data_files:
      - split: all
        path: index.parquet
---

# LORE-index

The queryable index over [`polymathic-ai/LORE`](https://huggingface.co/datasets/polymathic-ai/LORE).
One row per data row — location, identifiers, sequence-cluster labels, taxonomy and
lengths, **no payload** — so you can work out what you want before downloading any of
the 445 GB it describes.

> **⚠ Private preview.** Staging push for internal review; the release it indexes is not
> yet cleared for publication.

| | |
|:-|:-|
| Rows | 59,839,026 (one per LORE row) |
| Size | 0.84 GB (zstd parquet) |
| Subsets described | 40 |
| Data | [`polymathic-ai/LORE`](https://huggingface.co/datasets/polymathic-ai/LORE) |

## Use

```python
from lore import LoreIndex

idx = LoreIndex.from_hf(root="/path/to/LORE")
idx.describe()

sel = idx.select(["rna_seq", "splice_jctns_5cls"], species="Homo sapiens")
print(sel.n_rows, sel.n_unique)
train, val, test = sel.make_splits(key="rna_cluster_30", seed=0)
```

Or read it directly — it is a plain parquet:

```python
import duckdb
duckdb.sql("""
  SELECT species, COUNT(*) rows, COUNT(DISTINCT rna_cluster_30) clusters
  FROM 'index.parquet' WHERE subset LIKE 'rna_seq%' GROUP BY 1 ORDER BY 2 DESC LIMIT 10
""")
```

## Files

| file | what |
|:-|:-|
| `index.parquet` | 59,839,026 rows × 23 columns |
| `subsets.json` | subset → modality set, splits, shard list, row counts |

## Columns

`subset` and `shard` locate the row in LORE, at `data/<subset>/<shard>`. `key_kind` says
which identifier the row is anchored on — `genome_feature_id` (transcript) or
`uniprot_id` (protein).

| column | coverage | note |
|:-|-:|:-|
| `genome_feature_id`, `uniprot_id` | 76.7% / 23.3% | anchor identifier; exactly one is set |
| `root_id` | 77.4% | |
| `rna_cluster_30`, `rna_cluster_70` | 77.0% | mmseqs2 over RefSeq transcripts, `-c 0.8` |
| `protein_cluster_30`, `protein_cluster_70` | 51.4% | mmseqs2 over UniRef90 ∪ AFDB70, `-c 0.8` |
| `domain` … `species` | 77.0% | NCBI lineage, 8 ranks |
| `is_coding`, `feature_type` | 77.0% | |
| `aa_seq_length`, `rna_seq_length` | 41.0% / 77.0% | |
| `split` | 100% | **the** record of MIMIC 1.0's partition — see below |

**Coverage is by anchor, not random.** Genome-anchored rows carry rna clusters and
taxonomy; protein-anchored rows carry protein clusters. No single key covers everything,
so **split each anchor on its own native key** — see the main dataset card. A null key in
`make_splits` becomes a singleton: never pooled, but also unprotected against homology.

## ⚠ The `split` column

This column is the **only** record of the partition MIMIC 1.0 trained on. The published
data has no train/val/test directories and information-free filenames — that partition is
contaminated (homologues straddle the boundary; only 53.3% of checkable val/test rows are
clean), and a directory tree would make it the default anyone gets from `load_dataset`.

It is kept here so the published model's training run stays reproducible. Selecting on it
is opt-in and warns. For evaluation use `make_splits`.
