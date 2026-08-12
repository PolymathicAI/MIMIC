#!/usr/bin/env python3
"""
RefSeq Processing Pipeline - Step 6: Build Modality Files

This script extracts nucleotide sequences and creates individual modality files for 
machine learning tasks. It generates sequence-based features from genomic coordinates 
and saves them as separate modality datasets for training and evaluation.

Input:
- final_merged_annotations.parquet from step 5 (unified transcript database)
- ID dataset for filtering (optional): modality/id/{VERSION}/parquet/dataset.parquet
- HDF5 genome sequence files from step 0

Output:
- Individual modality files in modality/{modality_name}/{VERSION}/parquet/dataset.parquet
- Modalities include: rna_seq, aa_seq, utr, cds, splice_region, signal_codons, 
  is_coding, feature_type, and taxonomic classifications
- VERSION set to 1.4, update manually to preserve record of version history

The script:
1. Loads transcript annotations and optionally filters by ID dataset
2. Groups transcripts by genome and chromosome for efficient processing
3. Extracts nucleotide sequences with adaptive flanking regions around transcripts
4. Generates sequence-based features using coordinate information:
   - RNA sequences from transcript coordinates
   - Amino acid sequences from CDS coordinates
   - UTR regions (5' and 3') 
   - Splice region coordinates
   - Signal codons (start/stop codons)
5. Creates categorical modalities (is_coding, feature_type, taxonomy)
6. Saves each modality as a separate Parquet file with genome_feature_id keys

Key Features:
- Parallel processing using ProcessPoolExecutor for efficiency
- Memory-efficient processing with grouped data handling
- Coordinate-based sequence extraction
- Multiple sequence modality generation from single transcript
- Taxonomic hierarchy modalities for phylogenetic analysis
- Length filtering to exclude very long transcripts (default: 25,000bp)

Usage:
    python 06_build_modality_files.py [--filter-by-ids] [--max-length MAX_LENGTH]
    
Arguments:
    --filter-by-ids: Only process transcripts matching IDs in the id dataset
    --max-length: Maximum transcript length to include (default: 25000)
"""

#%%

import sys
import os
import tempfile
import shutil

import pandas as pd

from lore import logger
from lore import paths
from lore.utils import sequence_tools

from tqdm import tqdm
import h5py

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import argparse
import itertools
from functools import partial

#%%

logger.remove()
logger.add(sys.stderr, level="INFO")
#%%

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process transcripts with optional ID filtering')
parser.add_argument('--filter-by-ids', action='store_true', help='Filter transcripts to only include those with matching IDs in id_df')
parser.add_argument('--max-length', type=int, default=25000, help='Maximum transcript length to include (default: 25000)')
parser.add_argument('--test', action='store_true', help='Run in test mode with limited data for quick validation')
args = parser.parse_args()

VERSION = "1.4"

#%%
# Load the transcript annotations

logger.debug("Initializing paths")
transcripts_path = paths.get_path(data_type="data", stage="intermediate", name="transcripts", version=VERSION)

logger.info("Loading transcript annotations")
transcripts_df = pd.read_parquet(transcripts_path / "final_merged_annotations.parquet")

logger.info(f"Loaded {len(transcripts_df)} transcripts from {transcripts_path / 'final_merged_annotations.parquet'}")

transcripts_df = transcripts_df[transcripts_df['root_end'] - transcripts_df['root_start'] < args.max_length] # Filter out long transcripts

logger.info(f"Filtered transcripts to {len(transcripts_df)} with max length {args.max_length}bp")

if args.test:
    transcripts_df = transcripts_df.sample(n=10000, random_state=42)
    logger.info("Running in test mode: using only first 10000 transcripts")

#%%
# Filter by IDs if specified, otherwise use all transcripts
if args.filter_by_ids:
    logger.debug("Loading ID annotations")
    id_path = paths.get_path(data_type="data", stage="modality", name="id", version=VERSION, fmt="parquet")
    id_df = pd.read_parquet(id_path / "dataset.parquet")

    matched_transcripts = transcripts_df[transcripts_df['genome_feature_id'].isin(id_df['genome_feature_id'])]
    logger.info(f"Found {len(matched_transcripts)} matching transcripts out of {len(transcripts_df)} total")
else:
    matched_transcripts = transcripts_df
    logger.info(f"Using all {len(matched_transcripts)} transcripts without ID filtering")

#%%
# Group by genome and seqid to create a list of dataframes
genome_seqid_groups = matched_transcripts.groupby(['genome', 'seqid'])
transcript_dfs = [group for _, group in genome_seqid_groups]

logger.info(f"Split {len(matched_transcripts)} transcripts into {len(transcript_dfs)} genome/seqid groups")
#%%

raw_taxonomy_modalities = ["domain", "phylum", "class", "order", "family", "genus", "species"]

#%%

def copy_to_temp(genome, temp_dir):
    """
    Copy the genome sequence file to a temporary directory for processing.
    Args:
        genome (str): Genome identifier.
        temp_dir (str): Path to the temporary directory where the genome file will be copied.
    Returns:
        str: Path to the copied genome sequence file.
    """
    hdf5_file = sequence_tools.get_hdf5_file(genome)

    temp_path = os.path.join(temp_dir, f"{genome['genome']}.hdf5")
    shutil.copy(hdf5_file, temp_path)
    return temp_path

#%% 
def get_nucleotide_features(df, padding=0, temp_dir=None):
    """
    Get nucleotide features for a given dataframe of transcripts.
    Args:
        df (DataFrame): DataFrame containing transcript annotations.
        padding (int): Number of bases to pad the start and end positions.
    Returns:
        DataFrame: DataFrame with nucleotide features added.
    """

    output_rows = []

    # Open the hdf5 file for the current genome
    # with h5py.File(sequence_tools.get_hdf5_file(df.iloc[0]), "r") as hdf:
    with h5py.File(os.path.join(temp_dir, f"{df.iloc[0]['genome']}.hdf5"), "r") as hdf:
        # Iterate over each row in the dataframe
        for _, row in df.iterrows():
            # Get the sequence for the current row
            result = sequence_tools.get_nucleotide_annotations(row, padding, hdf)
            if result is not None:
                # Add genome_feature_id, is_coding, feature_type_clean, kingdom, 
                result["genome_feature_id"] = row["genome_feature_id"]
                result["is_coding"] = row["is_coding"]
                result["feature_type"] = row["feature_type_simplified"]
                result["kingdom"] = row["kingdom_simplified"]
                # Add taxonomy modalities
                for modality in raw_taxonomy_modalities:
                    result[modality] = row[modality]
                # Append the result to the output list

                result["rna_codons"] = sequence_tools.get_rna_codons_from_processed_modalities(result)

                output_rows.append(result)
            else:
                logger.warning(f"No sequence or splice regions found for genome {row['genome']}, feature_id {row['feature_id']}.")

    return output_rows

#%%
# Verifying modality paths are available

# Check taxonomy modality paths
for modality in raw_taxonomy_modalities:
    modality_path = paths.get_path(data_type="data", stage="modality", name=str.lower(modality), version=VERSION, fmt="parquet")

# Check other modality paths
other_modalities = ["splice_regions", "utr", "cds", "signal_codons", "is_coding", "feature_type", "splice_junctions", "cds_junctions", "rna_codons", "rna_seq"]
for modality in other_modalities:
    modality_path = paths.get_path(data_type="data", stage="modality", name=str.lower(modality), version=VERSION, fmt="parquet")

logger.info("All modality paths verified successfully")

#%%
logger.info("Copying genome sequences to temporary directory for processing")
num_workers = max(1, multiprocessing.cpu_count() - 2)  # Reserve 2 cores for system tasks
logger.info(f"Using {num_workers} workers for parallel processing")

# Get unique genomes and include organism_name for each
genomes = matched_transcripts[['genome', 'organism_name']].drop_duplicates()

records = genomes.to_dict(orient="records")
#%%


with tempfile.TemporaryDirectory(prefix="rna_seq_") as temp_dir:
    # Copy genome sequences to a temporary directory for processing
    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        copy_to_temp_partial = partial(copy_to_temp, temp_dir=temp_dir)
        _ = list(tqdm(exe.map(copy_to_temp_partial, records),
                  total=len(records),
                  desc="Copying genomes to temp directory"))

    #%%
    logger.info("Getting nucleotide features for each entry in the transcript dataframes")


    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        # Define a function to process each dataframe group
        process_group = partial(get_nucleotide_features, padding=0, temp_dir=temp_dir)

        results = list(tqdm(exe.map(process_group, transcript_dfs), total=len(genome_seqid_groups), desc="Processing groups"))

    # Flatten the results list of lists into a single list of dictionaries
    nucleotide_features = [feature for group_result in results for feature in group_result]

    # Convert the flattened list to a DataFrame
    nucleotide_df = pd.DataFrame(nucleotide_features)

    # Reorder columns to have genome_feature_id first
    other_columns = [col for col in nucleotide_df.columns if col != "genome_feature_id"]
    nucleotide_df = nucleotide_df[["genome_feature_id"] + other_columns]

    # Check the result
    logger.info(f"Created DataFrame with {len(nucleotide_df)} rows and {len(nucleotide_df.columns)} columns")
    

#%%
with tempfile.TemporaryDirectory(prefix="nucleotide_features_") as tmp_output_dir:
    logger.info(f"Using temporary directory {tmp_output_dir}")
    # Save individual dataframes for each other column
    for col in other_columns:
        col_df = nucleotide_df[["genome_feature_id"] + [col]]
        col_df = col_df.dropna()

        col_output_path = os.path.join(tmp_output_dir, f"{col}.parquet")
        col_df.to_parquet(col_output_path, index=False)
        logger.info(f"Saved {col} to {col_output_path}")

    logger.info(f"Saved all nucleotide features to temporary directory {tmp_output_dir}")

    # Copy files in tmp to ceph
    for col in other_columns:
        col_output_path = paths.get_path(data_type="data", stage="modality", name=str.lower(col), version=VERSION, fmt="parquet") / 'dataset.parquet'
        os.makedirs(os.path.dirname(col_output_path), exist_ok=True)
        shutil.copy(os.path.join(tmp_output_dir, f"{col}.parquet"), col_output_path)
        logger.info(f"Copied {col} to {col_output_path}")

logger.info("All nucleotide features processed and saved successfully")
