# The LORE build pipeline

Everything from upstream downloads to the tokenized shards MIMIC trained on. Four
stages, run in order:

| stage | what it does | cost |
|:-|:-|:-|
| `00_download` | fetch each upstream source (AlphaFold DB, RefSeq, UniProt, phyloP, ENCODE, FANTOM5, PaxDb, RASP2, PubMed) | network-bound, days |
| `01_process` | source → one parquet per modality; computes DSSP/SASA/MaSIF from structures, extracts per-transcript assay tracks from BigWigs | the bulk of the compute |
| `02_data_selection` | cluster sequences (mmseqs2), build the master-id table, assign splits, audit leakage | **the expensive step** — clustering dominates |
| `03_merge` | find modality subsets, merge, tokenize, write parquet + webdataset shards | I/O-bound |

Each stage directory has its own README with per-script detail. `02_data_selection`
and `03_merge` are the two that matter most for understanding the published dataset:
together they define the clusters, the subsets, and the tokenization.

## Setup

```bash
pip install "git+https://github.com/PolymathicAI/MIMIC.git#subdirectory=LORE[pipeline]"
export LORE_DATA_ROOT=/path/with/many/terabytes/free
```

`LORE_DATA_ROOT` is the single knob for where data lives; `lore.paths.get_path()`
resolves everything beneath it (`data/downloads/…`, `data/intermediate/…`,
`data/modality/…`, `data/final/lore/4.0/…`).

Four dependencies are not pip-friendly and are each needed by exactly one part of
`01_process`, so install them only if you are re-running that part:

| package | needed by |
|:-|:-|
| `pymesh` | MaSIF surface features |
| `foldcomp` | AlphaFold DB decompression |
| `pyBigWig` | phyloP / ATAC / CAGE / RASP2 track extraction |
| `ete3` | taxonomy tree handling |

## Configuration

The two configs that produced the public release are in `../configs/`:

* `master_ids_3.0.yaml` → `02_data_selection`
* `merge_config_4.0.yaml` → `03_merge`

Copy them over the stage-local `config.yaml` files rather than trusting those, which
track whatever version was built most recently.

## Differences from the internal version

This is the private build pipeline with release changes applied. Specifically:

* An internal support package was replaced by the `lore` shims: `from lore import
  logger`, `from lore import paths`. The utility modules the pipeline used are vendored
  into `lore.utils`.
* Data-root resolution is now the single environment variable `$LORE_DATA_ROOT`; the
  per-site env-file machinery it replaced was dropped.
* Stage 03 imported the internal training package for the modality registry; it now
  imports the public equivalents (`from mimic.modality_info import MODALITY_INFO`,
  `GROUP_INFO`), as does the ESM3 structure tokenization in `01_process/afdb`.
* Code for sources that contribute **no modality to LORE** was removed: GTEx, ARCHS4,
  Cistrome DB, Ensembl, and the RMDB / spot-RNA secondary-structure sets (of the RNA 2-D
  sources, only RASP2 is used). The Cistrome scripts also carried personal access
  credentials.
* Superseded code was removed: the `v0_swissprot` MaSIF version (LORE uses `v4_10m`),
  `Deprecated_*` scripts, and two standalone SASA scripts predating the current flow.
* Scheduler submission scripts were removed — see above.
* Exploratory notebooks, intermediate parquets, run logs, generated task lists and a
  vendored NCBI `datasets` binary were removed. Two `--mail-user=` addresses were
  stripped.

## No submission scripts — run the long steps as batch jobs

The original pipeline drove its heavy stages through SLURM submission scripts. Those are
**not distributed**: they encoded one cluster's partitions, module names, virtualenv
locations and absolute paths, none of which would work anywhere else.

What they actually did is documented as plain commands in each stage's README, so nothing
is lost. Wrap those in whatever your scheduler wants. The steps that need it, and roughly
what they cost:

| step | shape | documented in |
|:-|:-|:-|
| Protein clustering (`protein_cluster_30/70`) | one wide CPU job, ~96 cores, ~a day | `01_process/uniprot/README.md` |
| Transcript clustering (`rna_cluster_30/70`) | one wide CPU job, ~96 cores | `01_process/refseq/README.md` |
| MaSIF surface features | array, 60 shards, 2–3 days | `01_process/masif/v4_10m/README.md` |
| DSSP / SASA | disBatch array over structure shards | `01_process/{dssp,sasa}/README.md` |
| ESM3 structure tokenization | GPU array, one task per chunk | `01_process/afdb/README.md` |
| Leakage audit | one large-memory job (~250 GB) | `02_data_selection/README.md` |

DSSP and SASA build their own task files: `01_build_dssp_computation.py` and
`01_build_sasa_computation.py` emit a shell script plus a disBatch file, then print the
submission line. Set `$LORE_REPO` to your checkout if you are not running from its root.

The only absolute paths remaining in the tree are the MaSIF step's **container mount
points** (`/mnt/masif_source`, `/mnt/afdb_datasets`, …). Those are bind targets inside
the Apptainer image, not host paths — keep them exactly as they are.
