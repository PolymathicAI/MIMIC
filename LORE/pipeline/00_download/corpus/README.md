# PubMed Corpus Download

This directory contains scripts for downloading and processing the PubMed filtered corpus from HuggingFace.

## Scripts

### `download_corpus.py`
**Description**: Downloads the `common-pile/pubmed_filtered` dataset from HuggingFace Hub and converts it to Parquet format. The script processes compressed JSON shards, extracts the text content, and writes it to a single Parquet file in chunks for memory efficiency.

**Key Features**:
- Downloads all `licensed_pubmed-*.json.gz` shards from the HuggingFace repository
- Streams processing to handle large datasets efficiently
- Chunks data writes to manage memory usage (default: 50,000 rows per chunk)
- Extracts only the text column from JSON records

**Usage**:
```bash
python download_corpus.py \
    --cache_dir /path/to/cache \
    --output_path /path/to/output \
    --chunk_size 50000
```

**Arguments**:
- `--cache_dir`: Directory for HuggingFace cache (default: `/tmp/username/`)
- `--output_path`: Path for output Parquet file (default: `/tmp/username/dataset.parquet`)
- `--chunk_size`: Number of rows to buffer before writing (default: 50,000)

**Outputs**:
- Single Parquet file (`dataset.parquet`, or as specified by `--output_path`) in the output directory containing text from all PubMed filtered corpus shards
