# RASP2 Data Processing

This directory contains scripts for processing RASP2 (RNA structure probing) data from various species and experimental contexts.

## Processing Pipeline

The RASP2 data processing pipeline consists of three main steps:

### 1. Metadata Extraction (`00_prepare_metadata.py`)

This should happen after all bigwig files have finished downloading.

Parses RASP2 BigWig filenames to extract structured metadata including:
- Technology/method (e.g., SHAPE-MaP, DMS-seq, icSHAPE)
- Genome/species
- Publication information
- Chemical reagent used
- Experimental condition
- Strand information
- Scale (transcriptome-wide or targeted)
- Cell line (if applicable)

**Output:** Metadata CSV files saved to `$LORE_DATA_ROOT/data/downloads/rna2d/rasp2/score_data/{kingdom}/metadata_{species}_rasp2.csv`


### 2. Transcript Score Extraction (`01_fetch_transcript_scores.py`)

This should happen after all bigwig files have finished downloading, it can run in parallel with metadata extraction as there is no dependency between the two.

Fetches transcript coordinates from RefSeq annotations and extracts corresponding RASP2 scores from BigWig files. This script:
- Maps transcript coordinates to genomic positions
- Extracts per-nucleotide structure probing scores
- Handles chromosome naming conventions across species
- Filters out transcripts with excessive missing data (>90% NaN by default)
- Creates context identifiers combining metadata fields

**Output:** Processed data saved to `$LORE_DATA_ROOT/data/intermediate/rna2d/rasp2/{version}/{species}_transcript_rasp_scores_cleaned.parquet`

**Key Features:**
- Supports multiple species (animals, plants, bacteria/fungi, viruses)
- Handles both transcriptome-wide and targeted datasets
- Manages missing data and quality filtering
- Creates unique context IDs for grouping related experiments

### 3. Normalization (`02_normalize.py`)

This should happen after the experimental-context strings have been cleaned and
canonicalised, which groups replicate experiments under a single context id. That
cleaning step is not part of this release; `01_fetch_transcript_scores.py` already emits
a `context` column, so re-running it here normalises within whatever context grouping
that produced.

Normalizes RASP2 reactivity values to a [0, 1] range within each experimental context group. This ensures comparability across different datasets and experimental conditions.

**Process:**
- Groups data by context (unique combinations of species, reagent, condition, etc.)
- Applies min-max normalization: `(value - min) / (max - min)` for each context
- Preserves masked/missing values (represented as NaN)
- Validates normalization by checking min/max values per context

**Output:** Normalized data saved to modality-specific paths (version)

**Why normalize?**
- Different experimental conditions produce different reactivity ranges
- Normalization enables comparison across contexts
- Maintains relative reactivity patterns within each context


## Configuration

The `species_genome_dict.py` file contains mappings between:
- Species common names
- Kingdom classifications
- Genome assembly identifiers

Used by all scripts to maintain consistent naming conventions.

