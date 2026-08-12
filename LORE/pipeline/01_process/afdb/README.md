# AlphaFold Database (AFDB) Structure Processing Pipeline

## Pre-requisites

There should be a Foldcomp database (or sub-database), where the name of all the files in the directory matches the version-- for example `lore.paths.get_path("data", "intermediate", "afdb", "v4_10m") / v4_10m`. For a sub-database, it is preferable that we have already created train/test protein splits (see `uniprot` database).

## Processing Pipeline

The AFDB processing consists of 5 ordered Python scripts that transform the compressed FoldComp database into various output formats:

### 01_decompress_filter_foldcomp.py

**Purpose**: Decompresses FoldComp database and filters structures by pLDDT score

- Reads compressed protein structures from FoldComp database
- Filters structures based on B-factor threshold (default: pLDDT > 70)
- Converts PDB strings to NumPy arrays using `pdb_str_to_numpy`
- Outputs chunked HuggingFace datasets with structure data
- Supports parallel processing and checkpointing for large databases

### 02_recombine_chunks.py

**Purpose**: Recombines individual processing chunks into larger consolidated chunks

- Takes the many small chunks from step 1 and combines them into fewer, larger chunks
- Improves data locality and reduces file system overhead
- Creates compressed chunks for more efficient downstream processing

### 03_extract_sequences.py

**Purpose**: Extracts amino acid sequences from 3D structure data

- Loads the chunked structure datasets from step 2
- Converts NumPy structure arrays to amino acid sequences using `numpy_to_seq`
- Creates a sequence-only dataset (removes 3D coordinates)
- Saves sequences as HuggingFace dataset for sequence-based analyses

### 04_sequences_to_fasta.py

**Purpose**: Converts sequence dataset to standard FASTA format

- Reads the sequence HuggingFace dataset from step 3
- Writes sequences to FASTA file with UniProt IDs as headers
- Enables compatibility with standard bioinformatics tools

### 05_find_uniref_overlap.py

**Purpose**: Analyzes overlap between AFDB proteins and UniRef90 database
**Pre-requisite**: Requires the UniRef90 MMseqs2 database at `data/intermediate/uniref90/mmseqs_db`; see `../uniprot/README.md` step 1.

- Extracts UniProt IDs from processed AFDB structures
- Compares against UniRef90 database IDs
- Computes intersection and union statistics
- Saves ID lists for dataset splitting and analysis

## Running the Pipeline

Execute the scripts in order:

```bash
# Step 1: Decompress and filter (adjust parameters in script)
python 01_decompress_filter_foldcomp.py

# Step 2: Recombine chunks
python 02_recombine_chunks.py

# Step 3: Extract sequences
python 03_extract_sequences.py

# Step 4: Convert to FASTA
python 04_sequences_to_fasta.py

# Step 5: Find UniRef overlap
python 05_find_uniref_overlap.py
```

**Note**: Parameters like chunk sizes, number of processes, and file paths are configured within each script and should be adjusted based on your system resources and data size.

## Output

This pipeline produces:

- Chunked structure datasets (`chunk_*`) containing 3D coordinates
- Sequence datasets in HuggingFace format
- FASTA files for bioinformatics compatibility  
- UniProt ID overlap analysis with UniRef90

Each structure dataset contains a `"structure"` column with arrays that can be read
