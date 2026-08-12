#!/usr/bin/env python3
"""
RefSeq Processing Pipeline - Step 0: Convert Genome Sequences to HDF5 Format

This script converts raw genome sequence files (.fna format) from RefSeq downloads into 
HDF5 format for efficient storage and retrieval. The HDF5 format provides compressed 
storage and fast random access to genomic sequences.

Input:
- RefSeq genome files in FASTA (.fna) format located in download directories
- Each genome directory structure: downloads/refseq/229/{genome_name}/ncbi_dataset/data/{genome_name}/{genome_file}

Output:
- genome_sequences.hdf5 files for each genome in intermediate/refseq/229/{genome_name}/
- Each HDF5 file contains chromosome groups with compressed sequence datasets

The script:
1. Processes genome directories starting with "GCF" from RefSeq downloads
2. For each genome, finds the FASTA file and loads it using pyfaidx
3. Converts sequences to byte arrays and stores them in HDF5 format with gzip compression
4. Creates hierarchical structure: genome -> chromosome -> sequence dataset
5. Supports both single genome processing and batch processing with multiprocessing

Usage:
    python 00_convert_sequence_to_hdf5.py [--genome GENOME_ID] [--overwrite]
    
Arguments:
    --genome: Process specific genome ID (e.g., GCF_000001405.40)
    --overwrite: Overwrite existing HDF5 files
"""

#%% 

import os
import sys
import multiprocessing

from lore import logger
from lore import paths

import h5py
from pyfaidx import Fasta

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from functools import partial

#%%

logger.remove()
logger.add(sys.stderr, level="INFO")  # DEBUG and TRACE won't show now


# Create intermediate processing dir if it does not exist
intermediate_path = paths.get_path(data_type="data", stage="intermediate", name="refseq", version="229")
download_path = paths.get_path(data_type="data", stage="downloads", name="refseq", version="229")
rasp_path = paths.get_path(data_type="data", stage="downloads", name="rna2d", version="rasp2")
#%%

def convert_sequence_to_hdf5(genome_name, overwrite=False, rasp=False):
    # Create a directory for the genome in the intermediate path
    genome_intermediate_dir = intermediate_path / genome_name
    
    if not os.path.exists(genome_intermediate_dir):
        os.makedirs(genome_intermediate_dir, exist_ok=True)
        logger.debug(f"Created directory for genome: {genome_intermediate_dir}")

    # Define the path for the HDF5 file
    hdf5_path = os.path.join(genome_intermediate_dir, "genome_sequences.hdf5")

    # Check if the HDF5 file already exists
    if os.path.exists(hdf5_path) and not overwrite:
        logger.debug(f"HDF5 file for genome {hdf5_path} already exists and overwrite is False. Skipping processing.")
        return

    logger.debug(f"HDF5 file will be stored at: {hdf5_path}")

    if rasp:
        genome_dir = rasp_path / "genome"
    else:
        genome_dir = os.path.join(download_path, genome_name, "ncbi_dataset", "data", genome_name)
    
    logger.debug(f"Processing genome: {genome_name} in directory: {genome_dir}")

    # Find genome file in download dir, they may have slightly different names so this can't be hard coded
    genome_file = next((f for f in os.listdir(genome_dir) if f.startswith(genome_name) and (f.endswith('.fa') or f.endswith('.fna'))), None)

    if genome_file is None:
        logger.error(f"No file starting with {genome_name} found in {genome_dir}")
        return
    else:
        genome_path = os.path.join(genome_dir, genome_file)
        logger.debug(f"Found genome file: {genome_path}")
        genome = Fasta(genome_path)

    with h5py.File(hdf5_path, "w") as hdf:
        for chrom in genome.keys():
            chrom_group = hdf.create_group(chrom)  # Group for each chromosome
            chrom_seq = np.array(list(str(genome[chrom][:])), dtype="S1")  # Store as byte array

            # Store sequence with chunking enabled
            chrom_group.create_dataset(
                "sequence", 
                data=chrom_seq, 
                dtype="S1", 
                compression="gzip", 
                chunks=True  # Enable chunked access
            )

    logger.debug(f"Finished processing genome sequence: {genome_name}")

#%%

if __name__ == "__main__":

    # Create the base intermediate directory if it does not exist
    if not os.path.exists(intermediate_path):
        os.makedirs(intermediate_path)
        logger.info(f"Directory created at {intermediate_path} for intermediate refseq data")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process genome sequences and store them in HDF5 format.")
    parser.add_argument("--genome", type=str, help="Specify a single genome to process.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing annotated genome files.")
    parser.add_argument("--rasp", action="store_true", help="Process sequences downloaded with RNA 2D Rasp Data.")
    args = parser.parse_args()

    if args.genome:
        # Process a single genome if the genome argument is provided
        convert_sequence_to_hdf5(args.genome, overwrite=args.overwrite)

    else:
        if args.rasp:
            genomes = [f.split('.')[0] for f in os.listdir(rasp_path / "genome") if f.endswith(".fa")]
            logger.info(f"Preparing to process {len(genomes)} genome files in {rasp_path}")

            intermediate_path = paths.get_path(data_type="data", stage="intermediate", name="refseq", version="rna2d_rasp")
        
        else:
            # Process all genomes in the download directory
            genomes = [d for d in os.listdir(download_path) if d.startswith("GCF") and os.path.isdir(os.path.join(download_path, d))]
            logger.info(f"Preparing to process {len(genomes)} genome directories in {download_path}")


        convert_sequence_partial = partial(convert_sequence_to_hdf5, overwrite=args.overwrite, rasp=args.rasp)

        with multiprocessing.Pool() as pool:
            # Process all genomes if no argument is provided
            logger.info("Processing all genomes.")
            list(tqdm(pool.imap_unordered(convert_sequence_partial, genomes), 
                    total=len(genomes), 
                    desc="Processing genomes"))
            logger.info(f"All genome sequences have been processed and stored in HDF5 files in {intermediate_path}")
