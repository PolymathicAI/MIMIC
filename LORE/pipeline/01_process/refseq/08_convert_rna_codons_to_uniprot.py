#!/usr/bin/env python3
"""
Convert RNA Codons Dataset from genome_feature_id to uniprot_id

This script converts the RNA codons modality dataset to use uniprot_id as the primary
identifier instead of genome_feature_id. It creates a mapping from genome_feature_id
to uniprot_id using the final_merged_annotations.parquet file.

Process:
1. Check if dataset_genome.parquet already exists (warn if so)
2. Load transcript annotations to create genome_feature_id -> uniprot_id mapping
3. Load the RNA codons dataset
4. Add uniprot_id column based on genome_feature_id
5. Move original dataset.parquet to dataset_genome.parquet
6. Remove rows with no uniprot_id and save to dataset.parquet

Input:
- final_merged_annotations.parquet from step 5 (unified transcript database)
- rna_codons modality dataset: modality/rna_codons/{VERSION}/parquet/dataset.parquet

Output:
- dataset_genome.parquet: Original dataset with genome_feature_id as primary key
- dataset.parquet: New dataset with uniprot_id as primary key (rows without uniprot_id removed)

Usage:
    python 07_convert_rna_codons_to_uniprot.py [--version VERSION]
    
Arguments:
    --version: Version of the RNA codons dataset (default: 1.3)
"""

import sys
import argparse
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lore import logger
from lore import paths

# Enable pandas progress_apply
tqdm.pandas()

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

# Parse command line arguments
parser = argparse.ArgumentParser(description='Convert RNA codons dataset from genome_feature_id to uniprot_id')
parser.add_argument('--version', type=str, default='1.3', help='Version of the RNA codons dataset (default: 1.3)')
args = parser.parse_args()

VERSION = args.version

logger.info(f"Converting RNA codons dataset version {VERSION} to uniprot_id indexing")

# Define paths
rna_codons_path = paths.get_path("data", "modality", "rna_codons", VERSION, fmt="parquet")
dataset_path = rna_codons_path / "dataset.parquet"
dataset_genome_path = rna_codons_path / "dataset_genome.parquet"

# Check if dataset_genome.parquet already exists
if dataset_genome_path.exists():
    logger.warning(f"WARNING: {dataset_genome_path} already exists!")
    logger.warning("This script will overwrite the existing file.")
    response = input("Do you want to continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        logger.info("Aborting conversion")
        sys.exit(0)

# Check if original dataset exists
if not dataset_path.exists():
    logger.error(f"ERROR: {dataset_path} does not exist!")
    sys.exit(1)

# Load transcript annotations
logger.info("Loading transcript annotations to create mapping")
transcripts_path = paths.get_path(data_type="data", stage="intermediate", name="transcripts", version=VERSION)

transcripts_df = pd.read_parquet(transcripts_path / "final_merged_annotations.parquet")

logger.info(f"Loaded {len(transcripts_df)} transcripts")

# Create genome_feature_id -> uniprot_id mapping
# Keep only rows with both genome_feature_id and uniprot_id
logger.info("Creating genome_feature_id -> uniprot_id mapping")
with tqdm(total=len(transcripts_df), desc="Filtering transcripts with uniprot_id", unit="rows") as pbar:
    mapping_df = transcripts_df[['genome_feature_id', 'uniprot_id']].dropna()
    pbar.update(len(transcripts_df))

logger.info(f"Created mapping with {len(mapping_df)} entries (genome_feature_id -> uniprot_id)")

# Convert to dictionary for efficient lookup
logger.info("Building mapping dictionary")
gf_to_uniprot = dict(zip(mapping_df['genome_feature_id'], mapping_df['uniprot_id']))

logger.info(f"Mapping dictionary created with {len(gf_to_uniprot)} unique genome_feature_id entries")

# Load RNA codons dataset
logger.info(f"Loading RNA codons dataset from {dataset_path}")
rna_codons_df = pd.read_parquet(dataset_path)

logger.info(f"Loaded {len(rna_codons_df)} rows from RNA codons dataset")
logger.info(f"Columns: {list(rna_codons_df.columns)}")

# Add uniprot_id column
logger.info("Adding uniprot_id column based on genome_feature_id (fast vectorized operation)...")
rna_codons_df['uniprot_id'] = rna_codons_df['genome_feature_id'].map(gf_to_uniprot)
logger.info("Mapping complete")

# Count rows with and without uniprot_id
rows_with_uniprot = rna_codons_df['uniprot_id'].notna().sum()
rows_without_uniprot = rna_codons_df['uniprot_id'].isna().sum()

logger.info(f"Rows with uniprot_id: {rows_with_uniprot}")
logger.info(f"Rows without uniprot_id: {rows_without_uniprot}")

# Move original file to dataset_genome.parquet
logger.info(f"Moving {dataset_path} to {dataset_genome_path}")
shutil.move(str(dataset_path), str(dataset_genome_path))
logger.info("Original file moved successfully")

# Filter rows with uniprot_id and reorder columns to put uniprot_id first
logger.info("Filtering rows with uniprot_id")
rna_codons_uniprot_df = rna_codons_df[rna_codons_df['uniprot_id'].notna()].copy()

# Reorder columns to put uniprot_id first
logger.info("Reordering columns to put uniprot_id first")
cols = list(rna_codons_uniprot_df.columns)
cols.remove('uniprot_id')
cols = ['uniprot_id'] + cols
rna_codons_uniprot_df = rna_codons_uniprot_df[cols]

logger.info(f"Final dataset has {len(rna_codons_uniprot_df)} rows")
logger.info(f"New column order: {list(rna_codons_uniprot_df.columns)}")

# Save to dataset.parquet
logger.info(f"Saving uniprot-indexed dataset to {dataset_path}")
rna_codons_uniprot_df.to_parquet(dataset_path, index=False)

logger.info("Conversion complete!")

# Summary
logger.info("=" * 80)
logger.info("SUMMARY:")
logger.info(f"  Original dataset: {len(rna_codons_df)} rows")
logger.info(f"  Rows with uniprot_id: {rows_with_uniprot} ({100*rows_with_uniprot/len(rna_codons_df):.2f}%)")
logger.info(f"  Rows removed: {rows_without_uniprot} ({100*rows_without_uniprot/len(rna_codons_df):.2f}%)")
logger.info(f"  Final dataset: {len(rna_codons_uniprot_df)} rows")
logger.info(f"  Original file saved as: {dataset_genome_path}")
logger.info(f"  New file saved as: {dataset_path}")
logger.info("=" * 80)
