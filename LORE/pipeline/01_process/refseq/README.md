# RefSeq Processing Pipeline

This directory contains the RefSeq processing pipeline for the Central Dogma project. The pipeline processes genomic sequence data and annotations from RefSeq and GENCODE databases to create comprehensive biological datasets for machine learning tasks. Scripts should be run in numerical order (00-06) as each step depends on outputs from previous steps.

## Pipeline Overview

The pipeline follows these main steps:
1. **Sequence Conversion**: Convert FASTA genome sequences to efficient HDF5 format
2. **Annotation Database Creation**: Build queryable databases from GFF annotation files
3. **Feature Extraction**: Extract and compile transcript annotations with hierarchical relationships
4. **GENCODE Integration**: Process high-quality GENCODE annotations for human and mouse
5. **Data Merging**: Combine RefSeq and GENCODE data with taxonomic and protein information
6. **Modality Generation**: Create sequence-based modality files for machine learning

## Configuration

The pipeline uses lore.paths for data organization and expects:
- RefSeq v229 downloads in `downloads/refseq/229/`
- GENCODE downloads in `downloads/gencode/{organism}/`
- UniProt ID mapping file in `intermediate/refseq/`

## Scripts

### 00_convert_sequence_to_hdf5.py
**Purpose**: Converts RefSeq genome FASTA files to HDF5 format for efficient sequence storage and retrieval.

**Inputs**:
- RefSeq FASTA files (.fna) in download directories
- Directory structure: `downloads/refseq/229/{genome_name}/ncbi_dataset/data/{genome_name}/{genome_file}`

**Outputs**:
- `genome_sequences.hdf5` files for each genome in `intermediate/refseq/229/{genome_name}/`
- HDF5 files with chromosome groups containing compressed sequence datasets

**Usage**:
```bash
python 00_convert_sequence_to_hdf5.py [--genome GENOME_ID] [--overwrite]
```

**Key Features**:
- Parallel processing of multiple genomes
- Gzip compression for space efficiency
- Chromosome-based hierarchical storage
- Random access sequence retrieval support

---

### 01_build_annotation_db.py
**Purpose**: Creates gffutils databases from RefSeq GFF annotation files, handling duplicate ID issues that prevent database creation.

**Inputs**:
- RefSeq `genomic.gff` files from download directories
- Path: `downloads/refseq/229/{genome_name}/ncbi_dataset/data/{genome_name}/genomic.gff`

**Outputs**:
- `annotations.db` SQLite databases for each genome
- Located in `intermediate/refseq/229/{genome_name}/annotations.db`

**Key Features**:
- Automatic duplicate ID detection and resolution
- Parallel processing with multiprocessing
- Temporary file handling for memory efficiency
- gffutils database creation with optimal settings

---

### 02_compile_isoform_annotations.py
**Purpose**: Extracts transcript isoform annotations from gffutils databases, compiling hierarchical feature relationships and coordinates.

**Inputs**:
- `annotations.db` files from step 1
- Located in `intermediate/refseq/229/{genome_name}/annotations.db`

**Outputs**:
- `annotated_genome.parquet` files for each genome
- Located in `intermediate/refseq/229/{genome_name}/annotated_genome.parquet`

**Key Features**:
- Isoform identification (features with children but no grandchildren)
- Parent hierarchy traversal for gene information
- Exon and CDS coordinate extraction
- Memory monitoring and progress tracking
- ProcessPoolExecutor for parallel genome processing

---

### 03_build_gencode_db.py
**Purpose**: Creates gffutils databases from GENCODE annotation files for human and mouse genomes.

**Inputs**:
- GENCODE GFF3 files for human and mouse
- Human: `downloads/gencode/human/gencode.v47.chr_patch_hapl_scaff.annotation.gff3`
- Mouse: `downloads/gencode/mouse/gencode.vM36.annotation.gff3`

**Outputs**:
- GENCODE annotation databases
- Human: `intermediate/gencode/human/gencode.v47.chr_patch_hapl_scaff.annotations.db`
- Mouse: `intermediate/gencode/mouse/gencode.vM36.annotations.db`

**Key Features**:
- Parallel processing of human and mouse annotations
- Temporary file management for improved I/O
- Dry-run mode for testing
- Optimized gffutils settings for GENCODE format

---

### 04_compile_gencode_annotations.py
**Purpose**: Processes GENCODE databases to extract high-quality transcript annotations with detailed features not available in RefSeq.

**Inputs**:
- GENCODE `annotations.db` files from step 3
- Located in `intermediate/gencode/{organism}/*.annotations.db`

**Outputs**:
- `annotated_genome.parquet` files for human and mouse
- Located in `intermediate/gencode/{organism}/annotated_genome.parquet`

**Key Features**:
- Focus on transcript features (higher quality than RefSeq)
- Detailed UTR region extraction (5' and 3')
- Start/stop codon coordinate identification
- Selenocysteine stop codon handling
- Gene type and transcript type classifications

---

### 05_merge_annotations.py
**Purpose**: Combines RefSeq and GENCODE annotations into a unified transcript database with taxonomic information and UniProt ID mappings.

**Inputs**:
- RefSeq `annotated_genome.parquet` files from step 2
- GENCODE `annotated_genome.parquet` files from step 4
- UniProt ID mapping: `intermediate/refseq/uniprot_refseq_ensembl_id.parquet`
- NCBI assembly summary metadata

**Outputs**:
- `final_merged_annotations.parquet`: Unified transcript database
- Located in `intermediate/transcripts/1.1/final_merged_annotations.parquet`

**Key Features**:
- Parallel RefSeq genome processing
- UniProt ID mapping for cross-database integration
- NCBI taxonomy hierarchy integration
- Enhanced feature creation (is_coding, splice_region_coordinates, etc.)
- Taxonomic classification and kingdom simplification

**Enhanced Features Created**:
- `is_coding`: Boolean flag for protein-coding transcripts
- `splice_region_coordinates`: Unified exon/CDS coordinates
- `clean_feature_type`: Standardized feature classifications
- `kingdom_simplified`: Major kingdom groupings

---

### 06_build_modality_files.py
**Purpose**: Extracts nucleotide sequences and creates individual modality files for machine learning from the unified transcript database.

**Inputs**:
- `final_merged_annotations.parquet` from step 5
- Optional ID filtering dataset: `modality/id/1.1/parquet/dataset.parquet`
- HDF5 genome sequence files from step 0

**Outputs**:
- Individual modality files: `modality/{modality_name}/1.1/parquet/dataset.parquet`
- Modalities: `rna_seq`, `aa_seq`, `utr`, `cds`, `splice_region`, `signal_codons`, `is_coding`, `feature_type`, taxonomic classifications

**Key Features**:
- Parallel processing with ProcessPoolExecutor
- 200bp padding around transcript regions
- Length filtering (default: exclude >25,000bp transcripts)
- Multiple sequence modalities from single transcript
- Taxonomic hierarchy modalities

**Sequence Modalities Generated**:
- RNA sequences from transcript coordinates
- Amino acid sequences from CDS translation
- UTR regions (5' and 3')
- Splice region coordinates
- Signal codons (start/stop)

**Usage**:
```bash
python 06_build_modality_files.py [--filter-by-ids] [--max-length 25000]
```

### 07_make_seq_coords_df.py
**Purpose**: Extracts genomic coordinates and sequence lengths for all transcripts in the unified annotation file, producing a coordinate summary file (`transcript_sequence_coords.parquet`) for downstream analyses. This file is specifically used as the coordinate reference for extracting phyloP conservation scores in downstream processing (see `01_process/phylop/00_fetch_transcript_scores.py`).

**Inputs**:
- `final_merged_annotations.parquet` from step 5 (`intermediate/transcripts/1.1/final_merged_annotations.parquet`)
- RNA sequence modality file from step 6 (`modality/rna_seq/1.1/parquet/dataset.parquet`)

**Outputs**:
- `transcript_sequence_coords.parquet` in `intermediate/transcripts/1.1/`
  - Columns: `genome_feature_id`, `seqid`, `organism_name`, `sequence_start`, `sequence_length`

**Key Features**:
- Parallel extraction of transcript start coordinates using `process_map`
- Maps sequence lengths from the RNA sequence modality
- Drops transcripts with missing sequence lengths
- Ensures only transcripts with valid sequence lengths are retained
- Used as a coordinate reference for downstream tasks (e.g., phyloP score extraction)

**Usage**:
```bash
python 07_make_seq_coords_df.py
```

## Data Flow

```
RefSeq FASTA files + GFF annotations
    ↓ (00_convert_sequence_to_hdf5.py)
HDF5 genome sequences
    ↓ (01_build_annotation_db.py)
gffutils annotation databases
    ↓ (02_compile_isoform_annotations.py)
RefSeq transcript annotations
    ↓ (03_build_gencode_db.py, 04_compile_gencode_annotations.py)
GENCODE transcript annotations
    ↓ (05_merge_annotations.py)
Unified transcript database with taxonomy & UniProt IDs
    ↓ (06_build_modality_files.py)
Individual sequence modality files
```

## Dependencies

- Python 3.7+
- pandas
- h5py
- gffutils  
- pyfaidx
- tqdm
- ete3 (for NCBI taxonomy)
- lore package (`pip install "…#subdirectory=LORE[pipeline]"`)
- concurrent.futures
- multiprocessing

## Performance Notes

- Step 6 (modality file building) is typically the most time-consuming (~2 hours with newer optimizations)
- Step 5 (merging) requires significant memory for large datasets
- Parallel processing is utilized throughout for efficiency
- Temporary file copying improves database I/O performance
- HDF5 format provides fast random access to genomic sequences

## Output Data Structure

The final modality files contain:
- **genome_feature_id**: Unique identifier (genome_assembly + feature_id)
- **Sequence modalities**: rna_seq, aa_seq, utr, cds, splice_region, signal_codons
- **Categorical modalities**: is_coding, feature_type, taxonomic classifications
- **Coordinate modalities**: Genomic positions for various features

This pipeline provides a comprehensive foundation for biological sequence analysis and machine learning applications.

## Transcript clustering

Clusters the RefSeq transcripts with MMseqs2. The outputs become the
**`rna_cluster_30` and `rna_cluster_70` columns** of the master-id table — the RNA-side
counterpart to the protein clusterings in `../uniprot/README.md`, and half of what
LORE's leak-free splits are drawn on.

A wide CPU-bound job, normally submitted as a batch job (it ran on 96 threads). Set
`$LORE_DATA_ROOT` and make `mmseqs` available first.

```bash
TRANSCRIPTS=$LORE_DATA_ROOT/data/intermediate/transcripts/1.2
FASTA_PATH=$TRANSCRIPTS/refseq.fasta.gz
mkdir -p "$TRANSCRIPTS/mmseqs"

for PCT in 0.3 0.7; do
    mmseqs easy-cluster "$FASTA_PATH" "$TRANSCRIPTS/mmseqs/refseq_${PCT}" tmp \
        -c 0.8 --min-seq-id "${PCT}" --threads 96
done
```

Convert the results with `convert_refseq_mmseqs_to_parquet.py`. The master-id config
(`../../../configs/master_ids_3.0.yaml`) reads
`transcripts/1.2/mmseqs/refseq_0.3_cluster.parquet` and `refseq_0.7_cluster.parquet` as
`rna_cluster_30` and `rna_cluster_70`.

| clustering | tool | min identity | coverage | → master-id column |
|:-|:-|-:|-:|:-|
| RefSeq transcripts | `mmseqs easy-cluster` | 0.3 | 0.8 | `rna_cluster_30` |
| RefSeq transcripts | `mmseqs easy-cluster` | 0.7 | 0.8 | `rna_cluster_70` |
