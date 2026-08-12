#!/usr/bin/env python3
"""
RefSeq Processing Pipeline - Step 8: Optimized FASTA Sequence Writer

This script efficiently extracts DNA sequences from genomic coordinates and writes them 
to a compressed FASTA file. It uses parallelization and temporary file strategies 
for optimal performance on large datasets.

The output of this script is used in for clustering transcripts based on sequence similarity
as in PR #67. 

Input:
- final_merged_annotations.parquet from step 5 (unified transcript database)
- ID dataset for filtering (optional): modality/id/1.2/parquet/dataset.parquet
- HDF5 genome sequence files from step 0

Output:
- Compressed FASTA file with all DNA sequences: refseq.fasta.gz. Used in clustering for sample selection.

The script:
1. Loads transcript annotations and optionally filters by ID dataset
2. Groups transcripts by genome and chromosome for efficient processing
3. Copies genome HDF5 files to temporary directory for parallel access
4. Extracts nucleotide sequences with optional padding around transcript regions
5. Writes sequences to compressed FASTA format with genome_feature_id headers

Key Features:
- Parallel processing using ProcessPoolExecutor for efficiency
- Memory-efficient processing with grouped data handling
- Temporary file copying to avoid file locking issues
- Optional sequence padding and length filtering
- Compressed FASTA output for space efficiency

Usage:
    python 08_write_seqs_optimized.py [--filter-by-ids] [--max-length MAX_LENGTH] [--padding PADDING] [--test]
    
Arguments:
    --filter-by-ids: Only process transcripts matching IDs in the id dataset
    --max-length: Maximum transcript length to include (default: 25000)
    --padding: Number of bases to pad around transcript regions (default: 0)
    --transcript-version: Version of transcript annotations to use (default: 1.2)
    --output: Output FASTA file path (default: auto-generated)
    --test: Test mode - process only ~1000 sequences for validation
"""

import sys
import os
import argparse
import tempfile
import shutil
import gzip
import multiprocessing
import h5py
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from lore import logger
from lore import paths
from lore.utils import sequence_tools

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

TEMP_DIR_NAME = "fasta_seq"

def copy_to_temp(genome):
    """
    Copy the genome sequence file to a temporary directory for processing.
    Args:
        genome (dict): Dictionary with 'genome' and 'organism_name' keys.
    Returns:
        str: Path to the copied genome sequence file.
    """
    hdf5_file = sequence_tools.get_hdf5_file(genome)
    temp_dir = os.path.join(tempfile.gettempdir(), TEMP_DIR_NAME)
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{genome['genome']}.hdf5")
    shutil.copy(hdf5_file, temp_path)
    return temp_path

def get_sequences_for_group(df_group, padding=0):
    """
    Extract sequences for a group of transcripts from the same genome/chromosome.
    Args:
        df_group (DataFrame): DataFrame containing transcript annotations for one genome/seqid.
        padding (int): Number of bases to pad around transcript regions.
    Returns:
        list: List of dictionaries with genome_feature_id and sequence.
    """
    output_sequences = []
    
    # Open the HDF5 file for the current genome
    temp_hdf5_path = os.path.join(tempfile.gettempdir(), TEMP_DIR_NAME, f"{df_group.iloc[0]['genome']}.hdf5")
    try:
        with h5py.File(temp_hdf5_path, "r") as hdf:
            for _, row in df_group.iterrows():
                try:
                    # Get the sequence using the existing function
                    seq = sequence_tools.get_sequence_from_annotation(row, padding=padding, hdf=hdf)
                    
                    if seq is not None:
                        output_sequences.append({
                            'genome_feature_id': row['genome_feature_id'],
                            'genome': row['genome'],
                            'seqid': row['seqid'],
                            'feature_id': row['feature_id'],
                            'source': row['source'],
                            'sequence': seq
                        })
                    else:
                        logger.warning(f"Failed to get sequence for {row['genome_feature_id']}")
                        
                except Exception as e:
                    logger.error(f"Error processing {row['genome_feature_id']}: {e}")
                    
    except Exception as e:
        logger.error(f"Error opening HDF5 file {temp_hdf5_path}: {e}")
        
    return output_sequences

def write_sequences_to_fasta(sequences, output_file):
    """
    Write sequences to a compressed FASTA file.
    Args:
        sequences (list): List of dictionaries with genome_feature_id and sequence.
        output_file (str): Path to output FASTA file.
    """
    logger.info(f"Writing {len(sequences)} sequences to {output_file}")
    
    # Create temporary file in the same directory as the output file
    output_dir = os.path.dirname(output_file)
    temp_file = os.path.join(output_dir, f".tmp_{os.path.basename(output_file)}")
    
    try:
        with gzip.open(temp_file, "wt") as f:
            for seq_data in tqdm(sequences, desc="Writing FASTA"):
                header = f">{seq_data['genome_feature_id']}|{seq_data['genome']}|{seq_data['seqid']}|{seq_data['feature_id']}|{seq_data['source']}"
                f.write(header + "\n")
                f.write(seq_data['sequence'] + "\n")
        
        # Move temporary file to final location
        logger.info(f"Moving temporary file to {output_file}")
        shutil.move(temp_file, output_file)
        logger.info(f"FASTA file saved to {output_file}")
        
    except Exception as e:
        # Clean up temporary file if something went wrong
        if os.path.exists(temp_file):
            os.remove(temp_file)
        logger.error(f"Failed to write FASTA file: {e}")
        raise

def process_group_with_padding(args_tuple):
    """
    Wrapper function for processing groups with padding parameter.
    Args:
        args_tuple (tuple): Tuple containing (df_group, padding)
    Returns:
        list: List of dictionaries with genome_feature_id and sequence.
    """
    df_group, padding = args_tuple
    return get_sequences_for_group(df_group, padding=padding)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract sequences and write to FASTA with parallel processing')
    parser.add_argument('--filter-by-ids', action='store_true', 
                       help='Filter transcripts to only include those with matching IDs in id_df')
    parser.add_argument('--max-length', type=int, default=25000, 
                       help='Maximum transcript length to include (default: 25000)')
    parser.add_argument('--padding', type=int, default=0,
                       help='Number of bases to pad around transcript regions (default: 0)')
    parser.add_argument('--transcript-version', type=str, default='1.2', 
                       help='Version of the transcript annotations to use (default: 1.2)')
    parser.add_argument('--output', type=str, 
                       help='Output FASTA file path (default: auto-generated)')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: process only ~1000 sequences for validation')
    args = parser.parse_args()

    # Initialize paths
    logger.debug("Initializing paths")
    transcripts_path = paths.get_path(data_type="data", stage="intermediate", name="transcripts", version=args.transcript_version)

    # Load transcript annotations
    logger.info("Loading transcript annotations")
    transcripts_df = pd.read_parquet(transcripts_path / "final_merged_annotations.parquet")
    logger.info(f"Loaded {len(transcripts_df)} transcripts from {transcripts_path / 'final_merged_annotations.parquet'}")

    # Filter by length
    transcripts_df = transcripts_df[transcripts_df['root_end'] - transcripts_df['root_start'] < args.max_length]
    logger.info(f"Filtered transcripts to {len(transcripts_df)} with max length {args.max_length}bp")

    # Filter by IDs if specified
    if args.filter_by_ids:
        logger.debug("Loading ID annotations")
        id_path = paths.get_path(data_type="data", stage="modality", name="id", version=args.transcript_version, fmt="parquet")
        id_df = pd.read_parquet(id_path / "dataset.parquet")

        matched_transcripts = transcripts_df[transcripts_df['genome_feature_id'].isin(id_df['genome_feature_id'])]
        logger.info(f"Found {len(matched_transcripts)} matching transcripts out of {len(transcripts_df)} total")
    else:
        matched_transcripts = transcripts_df
        logger.info(f"Using all {len(matched_transcripts)} transcripts without ID filtering")

    # Apply test mode if specified
    if args.test:
        test_size = min(1000, len(matched_transcripts))
        matched_transcripts = matched_transcripts.sample(n=test_size, random_state=42)
        logger.info(f"Test mode: Limited to {len(matched_transcripts)} sequences for validation")

    # Group by genome and seqid for efficient processing
    genome_seqid_groups = matched_transcripts.groupby(['genome', 'seqid'])
    transcript_dfs = [group for _, group in genome_seqid_groups]
    logger.info(f"Split {len(matched_transcripts)} transcripts into {len(transcript_dfs)} genome/seqid groups")

    # Prepare parallel processing
    logger.info("Copying genome sequences to temporary directory for processing")
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    logger.info(f"Using {num_workers} workers for parallel processing")

    # Get unique genomes and include organism_name for each
    genomes = matched_transcripts[['genome', 'organism_name']].drop_duplicates()
    records = genomes.to_dict(orient="records")

    # Copy genome sequences to temporary directory
    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        _ = list(tqdm(exe.map(copy_to_temp, records), total=len(records), 
                     desc="Copying genomes to temp directory"))

    # Process groups in parallel to extract sequences
    logger.info(f"Extracting sequences with {args.padding}bp padding")
    
    # Prepare arguments for parallel processing
    process_args = [(df_group, args.padding) for df_group in transcript_dfs]

    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        results = list(tqdm(exe.map(process_group_with_padding, process_args), 
                          total=len(transcript_dfs), desc="Processing groups"))

    # Flatten results
    all_sequences = [seq for group_result in results for seq in group_result]
    logger.info(f"Extracted {len(all_sequences)} sequences")

    # Clean up temporary files
    temp_dir = os.path.join(tempfile.gettempdir(), TEMP_DIR_NAME)
    if os.path.exists(temp_dir):
        logger.info(f"Removing temporary directory {temp_dir}")
        shutil.rmtree(temp_dir)

    # Determine output file path
    if args.test:
        output_file = transcripts_path / "refseq_test.fasta.gz"
    elif args.output:
        output_file = args.output
    else:
        output_file = transcripts_path / "refseq.fasta.gz"
    
    # Write sequences to FASTA
    write_sequences_to_fasta(all_sequences, output_file)

if __name__ == "__main__":
    main()
