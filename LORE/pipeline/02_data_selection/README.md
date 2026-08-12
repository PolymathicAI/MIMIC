# Data Selection Pipeline

This directory contains the data selection pipeline for the Central Dogma project. The pipeline processes raw biological data modalities and creates curated datasets for machine learning tasks. Scripts should be run in numerical order (00-04) as each step depends on outputs from previous steps.

## Pipeline Overview

The pipeline follows these main steps:
1. **Format Conversion**: Convert data modalities to efficient Parquet format
2. **Master Table Creation**: Join all modalities with ID mappings 
3. **Split Creation**: Create train/validation/test splits at cluster level
4. **Representative Selection**: Select one representative per cluster
5. **Specialized Subsets**: Create task-specific validation subsets

## Configuration

All scripts use `config.yaml` which specifies:
- Data version and modality configurations
- Transformation rules for each modality
- Merge keys for joining data
- Paths for input and output data

## Scripts

### 00_convert_2_parquet.py
**Purpose**: Converts data modalities from HuggingFace Dataset format to Parquet format for efficient processing.

**Inputs**:
- `config.yaml`: Configuration file with modality specifications
- Data modalities in `hfds` or `hfds_chunked` format

**Outputs**:
- `dataset.parquet` files for each converted modality in their respective directories

**Usage**:
```bash
python 00_convert_2_parquet.py [--test]
```

**Key Features**:
- Skips modalities already in Parquet format
- Handles chunked datasets with parallel processing (8 workers)
- Test mode for processing only first chunk of chunked datasets

---

### 01_build_master_ids.py
**Purpose**: Creates a comprehensive master table by joining ID mappings with all configured modalities using DuckDB.

**Inputs**:
- `config.yaml`: Configuration with modality and transformation specs
- ID match table (uniprot_id ↔ genome_feature_id mappings)
- All modalities in Parquet format (from step 0)
- Precomputed modalities in specified formats

**Outputs**:
- `data/intermediate/master_ids/{version}/parquet/dataset_no_split.parquet`: Master table without splits

**Key Features**:
- FULL OUTER JOINs preserve all data across modalities
- Supports transformations: `len` (length), `exists` (boolean), raw values
- Uses DuckDB for efficient large-scale joins
- Validates output with row counts and column inspection

---

### 02_make_test_train_val_splits.py
**Purpose**: Creates train/validation/test splits at the cluster level to prevent data leakage between similar sequences.

**Inputs**:
- `config.yaml`: Configuration with data version
- `dataset_no_split.parquet`: Master table from step 1

**Outputs**:
- `data/intermediate/master_ids/{version}/parquet/dataset.parquet`: Master table with split assignments and `__key__` column

**Key Features**:
- Cluster-based splitting (80% train, 10% val, 10% test)
- Uses `protein_cluster_30` then `rna_cluster_30` for row clustering
- Fixed random seed (42) ensures reproducible splits
- Adds unique `__key__` identifier for each row

---

### 03_select_dataset.py
**Purpose**: Selects one representative sequence per cluster, prioritizing high-quality sequences based on annotation and species.

**Inputs**:
- `config.yaml`: Configuration with data version
- `dataset.parquet`: Master table with splits from step 2

**Outputs**:
- `data/modality/id/{version}/parquet/train.parquet`: Training set representatives
- `data/modality/id/{version}/parquet/val.parquet`: Validation set representatives
- `data/modality/id/{version}/parquet/test.parquet`: Test set representatives
- `data/modality/id/{version}/parquet/dataset.parquet`: Combined dataset

**Key Features**:
- Priority scoring: SwissProt (+4), Human (+2), Mouse (+1)
- Uses `protein_cluster_70` → `protein_cluster_30` → `rna_cluster_30` hierarchy
- ROW_NUMBER() with priority ranking and random tie-breaking
- Outputs only essential columns: `__key__`, `uniprot_id`, `genome_feature_id`

---

### 04_make_val_subsets.py
**Purpose**: Creates specialized validation subsets optimized for specific biological tasks.

**Inputs**:
- `config.yaml`: Configuration with data version and transcript info
- Master IDs table with splits (accessed from intermediate data)

**Outputs**:
- `data/modality/id/{version}/parquet/val_transcription.parquet`: Transcription task validation set
- `data/modality/id/{version}/parquet/val_coding_class.parquet`: Coding classification validation set

**Key Features**:
- **Transcription subset**: Filters for sequences with amino acid sequences and CDS coordinates
- **Coding classification subset**: Balanced dataset with equal coding/non-coding sequences
- Limits to 20,000 sequences per class when available
- Uses fixed random seed (0.42) for reproducible sampling

**Filtering Criteria**:
- Transcription: `aa_seq_length IS NOT NULL AND CDS_coordinates_length >= 2`
- Coding Classification: `rna_seq_length IS NOT NULL AND splice_region_coordinates_length >= 1`

## Data Flow

```
Raw Modalities (hfds/hfds_chunked)
    ↓ (00_convert_2_parquet.py)
Modalities in Parquet Format
    ↓ (01_build_master_ids.py)
Master IDs Table (no splits)
    ↓ (02_make_test_train_val_splits.py)
Master IDs Table (with splits)
    ↓ (03_select_dataset.py)
Representative Dataset (train/val/test)
    ↓ (04_make_val_subsets.py)
Specialized Validation Subsets
```

## Dependencies

- Python 3.7+
- DuckDB
- pandas
- numpy
- PyYAML
- datasets (HuggingFace)
- lore package (`pip install "…#subdirectory=LORE[pipeline]"`)

## Error Handling

All scripts include safety checks:
- Verify input files exist before processing
- Prevent overwriting existing output files
- Log detailed progress and timing information
- Validate output with row counts and structure checks

## Performance Notes

- Step 1 (master table creation) is typically the most memory-intensive
- DuckDB's columnar processing provides good performance on large datasets
- Parallel processing in step 0 speeds up chunked dataset conversion
- Consider available memory when processing very large modalities 