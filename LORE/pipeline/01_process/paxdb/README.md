# Processing Protein Abundance

This folder contains scripts for cleaning and preprocessing protein abundance data from PaxDB. It requires that the following files have been downloaded from PaxDB to `$LORE_DATA_ROOT/data/downloads/paxdb/v5` (see `../../00_download/paxdb/README.md`):

- paxdb-abundance-files-v5.0/*
- paxdb-protein-sequences-v5.0/*
- paxdb-uniprot-links-v5.0/*

### 1. Find unmapped names

Although PaxDB provides a mapping between their internal ID and UniProt IDs, they don't all match. In `01_find_unmapped_names.py`, we identify those that don't map properly and write them out to a file

### 2. Map names to valid UniProt IDs

The `02_uniprot_id_mapping.py` file uses UniProt ID mapping API calls to find valid IDs for those which are unmapped. This helps us recover several million records with abundance data that we can map into our system. Should be called as

```bash
PAXDB=$LORE_DATA_ROOT/data/downloads/paxdb/v5
python LORE/pipeline/01_process/paxdb/01.5_uniprot_id_mapping.py \
    "$PAXDB/uniprot-names-to-map.csv" -o "$PAXDB/up-names-out.txt"
```

### 3. Build PaxDB DataFrame

The original mapping and our new mapping are used together with the PaxDB abundance and sequence files, and a clean dataframe is built in `03_build_paxdb_dataframe.py`, where duplicates are dropped. This creates two files:

- `$LORE_DATA_ROOT/data/modality/prot_abund/v5/parquet/dataset.parquet`, the main PaxDB file
- `$LORE_DATA_ROOT/data/intermediate/paxdb/v5/paxdb_filtered_v5.0.parquet`, which filters to entries matching the master ID list. This is unused for now, because matching is done downstream
