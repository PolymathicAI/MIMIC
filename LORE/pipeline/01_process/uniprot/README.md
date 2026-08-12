# UniProt — protein sequence clustering

Clusters protein sequences from UniRef90 and the AlphaFold Database with MMseqs2. The
outputs become the **`protein_cluster_30` and `protein_cluster_70` columns** of the
master-id table, which is what LORE's leak-free splits are drawn on. If you want to
reproduce or re-draw those splits, this is the step that defines them.

These are long, wide, CPU-bound jobs — clustering the combined database ran on 96 cores
for the better part of a day, and is normally submitted as a batch job. The commands are
given directly below rather than as submission scripts, so they port to any scheduler.

Set `$LORE_DATA_ROOT` and make `mmseqs` available (module, conda, or `PATH`) first.
Paths below follow the layout `lore.paths.get_path()` resolves.

## 1. UniRef90 database and clustering

```bash
UNIREF90_MMSEQS=$LORE_DATA_ROOT/data/intermediate/uniref90/mmseqs_db
UNIREF90_30PCT=$LORE_DATA_ROOT/data/intermediate/uniref90/30pct_clust
UNIREF90_70PCT=$LORE_DATA_ROOT/data/intermediate/uniref90/70pct_clust
mkdir -p "$UNIREF90_MMSEQS" "$UNIREF90_30PCT" "$UNIREF90_70PCT"

# Fetch UniRef90 as an MMseqs2 database, plus a FASTA copy
mmseqs databases UniRef90 "$UNIREF90_MMSEQS/uniref90" /tmp
mmseqs convert2fasta "$UNIREF90_MMSEQS/uniref90" "$UNIREF90_MMSEQS/uniref90.fasta"

# 30% identity, 80% coverage
mmseqs cluster "$UNIREF90_MMSEQS/uniref90" "$UNIREF90_30PCT/30pct_clust" /tmp \
    --min-seq-id 0.3 -c 0.8 --threads 96
mmseqs createtsv "$UNIREF90_MMSEQS/uniref90" "$UNIREF90_MMSEQS/uniref90" \
    "$UNIREF90_30PCT/30pct_clust" "$UNIREF90_30PCT/30pct_clust.tsv" --threads 96

# 70% identity, 80% coverage
mmseqs cluster "$UNIREF90_MMSEQS/uniref90" "$UNIREF90_70PCT/70pct_clust" /tmp \
    --min-seq-id 0.7 -c 0.8 --threads 96
mmseqs createtsv "$UNIREF90_MMSEQS/uniref90" "$UNIREF90_MMSEQS/uniref90" \
    "$UNIREF90_70PCT/70pct_clust" "$UNIREF90_70PCT/70pct_clust.tsv" --threads 96
```

## 2. Combine with AlphaFold DB (pLDDT ≥ 70) and re-cluster

**This produces the clusterings LORE actually uses.** Requires `afdb70.fasta` from
`../afdb/04_sequences_to_fasta.py`.

```bash
AFDB70_FASTA=$LORE_DATA_ROOT/data/modality/aa_seq/v4_plddt_70/fasta/afdb70.fasta
AFDB70_MMSEQS=$LORE_DATA_ROOT/data/intermediate/afdb/v4_70/mmseqs
mkdir -p "$AFDB70_MMSEQS"

mmseqs createdb "$AFDB70_FASTA" "$AFDB70_MMSEQS/afdb70"

# Union of UniRef90 and AFDB70 (sequence and header databases both)
mmseqs concatdbs "$UNIREF90_MMSEQS/uniref90"   "$AFDB70_MMSEQS/afdb70"   "$AFDB70_MMSEQS/afdb70_ur90"
mmseqs concatdbs "$UNIREF90_MMSEQS/uniref90_h" "$AFDB70_MMSEQS/afdb70_h" "$AFDB70_MMSEQS/afdb70_ur90_h"

# Cluster the union at both thresholds
for PCT in 30 70; do
    OUT=$LORE_DATA_ROOT/data/intermediate/uniref90/${PCT}pct_afdb70
    NAME=${PCT}pct_afdb70_ur90
    mkdir -p "$OUT"
    mmseqs cluster "$AFDB70_MMSEQS/afdb70_ur90" "$OUT/$NAME" /tmp \
        --min-seq-id "0.${PCT}" -c 0.8 --threads 96
    mmseqs createtsv "$AFDB70_MMSEQS/afdb70_ur90" "$AFDB70_MMSEQS/afdb70_ur90" \
        "$OUT/$NAME" "$OUT/$NAME.tsv" --threads 96
done
```

Convert the TSVs to parquet with `convert_uniprot_mmseqs_to_parquet.py`. The master-id
config (`../../../configs/master_ids_3.0.yaml`) then reads
`30pct_afdb70/30pct_afdb70.parquet` and `70pct_afdb70/70pct_afdb70.parquet` as
`protein_cluster_30` and `protein_cluster_70`.

## Parameters, in one place

| clustering | tool | min identity | coverage | → master-id column |
|:-|:-|-:|-:|:-|
| UniRef90 ∪ AFDB70 | `mmseqs cluster` | 0.3 | 0.8 | `protein_cluster_30` |
| UniRef90 ∪ AFDB70 | `mmseqs cluster` | 0.7 | 0.8 | `protein_cluster_70` |

The RNA/transcript equivalents (`rna_cluster_30`, `rna_cluster_70`) are in
`../refseq/README.md`.

> **The two thresholds are not nested.** A 70%-identity cluster can straddle two
> 30%-identity clusters, which is exactly why splitting on `protein_cluster_30` alone
> left residual leakage at the stricter threshold. See `LORE/README.md`.

## Other scripts here

`build_refprot_fasta.py` / `build_refprot_id_list.py` build the UniRef90∩RefSeq subset;
`save_lens.py` records sequence lengths. An additional
UniRef90∩RefSeq-overlap clustering step is described in the project history but its
submission script was not part of the pipeline that produced LORE 4.0.
