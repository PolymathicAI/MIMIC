# ATAC-seq Processing Pipeline (ENCODE NarrowPeak → Transcript-Level Scores)

This directory contains a four-step processing workflow that converts downloaded ENCODE ATAC-seq narrowPeak files into final per-transcript accessibility scores stored in Parquet format. The workflow performs:

1. **Genome liftover (mm10 → mm39)**
2. **Conversion of peak BED files into cleaned bigWig signal tracks**
3. **Metadata synchronization with final bigWig locations**
4. **Extraction of per-base ATAC signal over transcripts**

---

## Input Requirements

Before running this workflow, ensure you have run the following scripts in order:
1. `python -m data-processing.00_download.atac.00_download_atac`  
   (Downloads the raw ENCODE ATAC-seq narrowPeak files and initial metadata.)
2. `python -m data-processing.01_process.atac.00_liftover_and_metadata_update`  
   (Performs mm10→mm39 liftover and updates metadata.)

These scripts generate:
- `metadata_atac_bed_narrowpeak.csv`
- The raw `.narrowPeak(.gz)` peak files under:  
  `data/downloads/atac/encode-narrowpeak/atac_peaks/`

---

## Pipeline Overview

| Step | Script | Description | Output |
|------|--------|-------------|--------|
| 1 | `00_liftover_and_metadata_update.py` | Detects mouse (mm10) peak files and lifts them to mm39 using CrossMap. Updates metadata to point to lifted files. | `metadata_atac_bed_narrowpeak_LIFTED.csv` + lifted mouse files (`*.lifted.bed`) |
| 2 | `01_convert_bed_to_bigwig.py` | Converts human (GRCh38) and lifted mouse (mm39) BED peak files into cleaned bigWig signal tracks (fixes overlaps and clips to chromosome bounds). | bigWigs: `data/intermediate/atac/encode-narrowpeak/bigwig/<genome>/<species>/*.bw` |
| 3 | `02_update_metadata_bigwig.py` | Rewrites metadata so rows now reference the generated `.bw` files instead of `.bed` files. Organizes metadata per species. | `metadata_atac_bigwig.csv` + per-species split metadata CSVs |
| 4 | `03_fetch_transcript_scores.py` | Loads transcript coordinates and extracts mean per-base ATAC signal for each transcript from each experiment individually (one row per transcript, experiment pair). Saves processed transcript-level accessibility vectors. | `data/modality/atac/1.1/<species>/dataset.parquet` |
---

Execute scripts in order:

```bash
# Step 1: Perform mm10→mm39 liftover and update metadata
python -m data-processing.01_process.atac.00_liftover_and_metadata_update

# Step 2: Convert BED to bigWig
python -m data-processing.01_process.atac.01_convert_bed_to_bigwig

# Step 3: Update metadata to reference final .bw files
python -m data-processing.01_process.atac.02_update_metadata_bigwig

# Step 4: Extract transcript-level ATAC signal
python -m data-processing.01_process.atac.03_fetch_transcript_scores
```

## Output Structure

After completion, key files will exist in:

```text
data/
├─ downloads/atac/encode-narrowpeak/
│  ├─ metadata_atac_bigwig.csv
│  ├─ score_data/
│  │  └─ <kingdom>/
│  │     └─ metadata_<species>_atac_bigwig.csv
│  └─ liftover/
│     └─ lifted/
│        └─ mm10_to_mm39/
│           └─ *.lifted.bed
│
├─ intermediate/atac/encode-narrowpeak/bigwig/
│  ├─ GRCh38/
│  │  └─ human/
│  │     └─ *.bw
│  └─ mm39/
│     └─ mouse/
│        └─ *.bw
│
└─ modality/atac/1.1/
   ├─ human/
   │  └─ dataset.parquet
   └─ mouse/
      └─ dataset.parquet
```


## Notes:
- You may re-run scripts safely. They check for existing output and only regenerate missing components.
- Expected organisms handled: Homo sapiens (GRCh38) and Mus musculus (mm39).
- Tiny or corrupted .bw files (< 2KB) will be automatically ignored.
