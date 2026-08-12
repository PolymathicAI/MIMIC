#!/usr/bin/env python3
"""
RefSeq Processing Pipeline - Step 7: Generate Transcript Sequence Coordinates

Purpose:
Extracts genomic coordinates and sequence lengths for all transcripts in the unified annotation file, producing a coordinate summary file (transcript_sequence_coords.parquet) for downstream analyses. This file is specifically used as the coordinate reference for extracting phyloP conservation scores in downstream processing (see 01_process/phylop/00_fetch_transcript_scores.py).

Input:
- final_merged_annotations.parquet from step 5, located in intermediate/transcripts/{VERSION}/
- rna_seq modality file from step 6, located in modality/rna_seq/{VERSION}/parquet/dataset.parquet

Output:
- transcript_sequence_coords.parquet in intermediate/transcripts/{VERSION}/
  (Contains: genome_feature_id, seqid, organism_name, sequence_start, sequence_length)
- VERSION defaults to 1.4 but can be specified via --version argument

The script:
1. Loads merged transcript annotations and RNA sequence modality
2. Extracts start coordinates for each transcript using sequence_tools
3. Maps sequence lengths from the RNA sequence modality
4. Drops transcripts with missing sequence lengths
5. Saves the resulting DataFrame as transcript_sequence_coords.parquet

Key Features:
- Parallel coordinate extraction using process_map for speed
- Progress bars for all major steps
- Handles both human and mouse transcriptomes
- Ensures only transcripts with valid sequence lengths are retained

Usage:
    python 07_make_seq_coords_df.py [--version VERSION]

Arguments:
    --version VERSION    Version number for input and output directories (default: 1.4)
"""

#%%

import pandas as pd
import argparse

from lore import paths
from lore import logger
from lore.utils import sequence_tools

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import multiprocessing

#%%

# Parse command line arguments
parser = argparse.ArgumentParser(description="Generate transcript sequence coordinates")
parser.add_argument("--version", type=str, help="Version number for input and output directories (no default - must be specified)")
args = parser.parse_args()

transcripts_path = paths.get_path(data_type="data", stage="intermediate", name="transcripts", version=args.version)

logger.info("Loading transcript annotations")
transcripts_df = pd.read_parquet(transcripts_path / "final_merged_annotations.parquet")

# %%

def get_coords(idx):
    row = transcripts_df.iloc[idx]
    start = sequence_tools.get_sequence_from_annotation(row, padding=200, dry=True)
    return {
        'genome_feature_id': row['genome_feature_id'],
        'seqid': row['seqid'],
        'organism_name': row['organism_name'],
        'sequence_start': start,
        'strand': row['strand'],
        'asm_name': row['asm_name'],
    }

# %%

# Set up multiprocessing to speed up coordinate extraction
max_workers = max(1, multiprocessing.cpu_count() - 2)

logger.info(f"Using {max_workers} workers for parallel processing")

# Use executor.map for cleaner parallel processing
results = []
# Prepare data for parallel processing
# Process using indices
indices = list(range(len(transcripts_df)))
# Use process_map for proper progress bar 
results = process_map(get_coords, indices, max_workers=max_workers, 
                     desc="Extracting coordinates", chunksize=1000, total=len(indices))

# Create the new dataframe
seq_coords_df = pd.DataFrame(results)

seq_path = paths.get_path(data_type="data", stage="modality", name="rna_seq", version=args.version, fmt="parquet") / "dataset.parquet"
seq_df = pd.read_parquet(seq_path, engine='fastparquet')

logger.info("Creating sequence length mapping...")

# Create a mapping of genome_feature_id to sequence length
logger.info("Creating sequence length mapping")
tqdm.pandas(desc="Creating sequence length mapping")
seq_length_mapping = dict(zip(seq_df['genome_feature_id'], 
                             seq_df['rna_seq'].progress_apply(len)))

# Add the sequence_length column to coords_df
logger.info("Mapping sequence lengths to coordinates dataframe...")
tqdm.pandas(desc="Mapping sequence lengths")
seq_coords_df['sequence_length'] = seq_coords_df['genome_feature_id'].progress_map(lambda x: seq_length_mapping.get(x))
# Drop rows with missing sequence_length
logger.info("Dropping rows with missing sequence_length...")
seq_coords_df = seq_coords_df.dropna(subset=['sequence_length'])

# Convert sequence_length to integer
logger.info("Converting sequence_length to integer...")
seq_coords_df['sequence_length'] = seq_coords_df['sequence_length'].astype(int)

# Log the number of transcripts with valid sequence lengths
logger.info(f"Retained {len(seq_coords_df)} transcripts with valid sequence lengths")

# Save the dataframe
output_path = paths.get_path(data_type="data", stage="intermediate", name="transcripts", version=args.version)
output_path.mkdir(exist_ok=True, parents=True)
seq_coords_df.to_parquet(output_path / "transcript_sequence_coords.parquet")

logger.info(f"Saved sequence coordinates for {len(seq_coords_df)} transcripts to {output_path / 'transcript_sequence_coords.parquet'}")