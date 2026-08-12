# PubMed Corpus Tokenization

This directory contains scripts for tokenizing the PubMed corpus dataset.

## Scripts

### `tokenize_dataset.py`
**Description**: Tokenizes the PubMed corpus dataset using BioBERT tokenizer. Processes the raw Parquet file and generates a tokenized version for downstream use in language model training or analysis.

**Key Features**:
- Uses `dmis-lab/biobert-base-cased-v1.2` tokenizer optimized for biomedical text
- Parallel processing with configurable worker count
- Batch tokenization for efficiency
- No padding or truncation applied to preserve full text sequences
- Outputs tokenized dataset in Parquet format

**Usage**:
```bash
python tokenize_dataset.py \
    --num_proc 4 \
    --batch_size 500 \
    --cache_path /path/to/cache \
    --split train
```

**Arguments**:
- `--num_proc`: Number of parallel processes for tokenization (default: 4)
- `--batch_size`: Batch size for tokenization (default: 500)
- `--cache_path`: HuggingFace cache directory (default: `/tmp/username/`)
- `--split`: Dataset split to process (default: `train`)

**Inputs**:
- Raw Parquet dataset: `{cache_path}/dataset.parquet`

**Outputs**:
- Tokenized dataset: `{cache_path}/tokenized_ds.parquet`
- Contains tokenized text with input IDs and attention masks
