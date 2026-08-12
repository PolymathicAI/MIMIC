# PaxDb — protein abundance

Source for the `prot_abund` modality: integrated protein abundance measurements across
organisms, tissues and cell lines. Version **5.0** was used for LORE 4.0.

PaxDb is distributed under **CC-BY-4.0**, so it may be redistributed with attribution.
See `LORE/NOTICE.md`.

## Download

Three archives are needed — the protein sequences, the UniProt id mapping, and the
abundance measurements themselves:

```bash
DL_DIR=$LORE_DATA_ROOT/data/downloads/paxdb/v5
mkdir -p "$DL_DIR" && cd "$DL_DIR"

wget https://pax-db.org/downloads/5.0/paxdb-protein-sequences-v5.0.zip
wget https://pax-db.org/downloads/5.0/paxdb-uniprot-links-v5.0.zip
wget https://pax-db.org/downloads/5.0/datasets/paxdb-abundance-files-v5.0.zip

unzip 'paxdb-*.zip' && rm paxdb-*.zip
```

## Next

`../../01_process/paxdb/` maps PaxDb identifiers onto UniProt accessions and builds the
modality parquet. `01_find_unmapped_names.py` reports accessions that fail to map;
`01.5_uniprot_id_mapping.py` resolves them via UniProt's id-mapping service, e.g.

```bash
python ../../01_process/paxdb/01.5_uniprot_id_mapping.py \
    "$DL_DIR/uniprot-names-to-map.csv" -o "$DL_DIR/up-names-out.txt"
```

then `03_build_paxdb_dataframe.py` writes the modality.

The abundance value that reaches the model is a per-protein ppm scalar; the tissue or
cell-line it was measured in is carried separately as free text in the `context`
modality, so `prot_abund` is condition-conditional.
