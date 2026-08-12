Data Processing Scripts:

Download script: See the download folder.
Extract metadata (cell type, replicate, sample ID) from raw CAGE data file column names. Outputs structured .csv and .pkl files for downstream use. The script supports command-line execution with input and output directories as arguments. Note that pattern-matching in column names may introduce minor metadata leakage if naming conventions vary. Original files are older build so we use LiftOver to map to the newer builds.
Liftover note:

We map CAGE peak locations between genome builds using a pre-generated LiftOver BED file. The script preserves metadata and outputs updated .csv files with lifted coordinates. Assumes liftOver is installed and ran separately.
liftOver commands:

```bash
./liftOver peaks_mm9.bed mm9ToMm39.over.chain.gz peaks_mm39.bed peaks_unmapped_mm39.bed
./liftOver cage/peaks_hg19.bed hg19ToHg38.over.chain.gz peaks_hg38.bed peaks_unmapped.bed
```

The files produced by this code are in `$LORE_DATA_ROOT/data/intermediate/cage/paired_files/`

s