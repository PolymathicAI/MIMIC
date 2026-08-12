#!/usr/bin/env python3
"""
RefSeq Processing Pipeline - Step 1: Build Annotation Databases

This script processes RefSeq GFF annotation files and creates gffutils databases for 
efficient querying of genomic features. It handles duplicate ID issues in GFF files 
by renaming them to ensure successful database creation.

Input:
- RefSeq genomic.gff files located in download directories
- Each annotation file path: downloads/refseq/229/{genome_name}/ncbi_dataset/data/{genome_name}/genomic.gff

Output:
- annotations.db files for each genome in intermediate/refseq/229/{genome_name}/
- SQLite databases created by gffutils with indexed genomic features

The script:
1. Reads GFF files and identifies duplicate feature IDs
2. Renames duplicate IDs by appending row numbers to make them unique
3. Creates temporary GFF content and builds gffutils databases
4. Stores databases in intermediate directories for downstream processing
5. Supports both single genome processing and parallel batch processing

Key Features:
- Duplicate ID resolution prevents gffutils database creation failures
- Parallel processing using multiprocessing for efficiency
- Temporary file handling for memory-efficient database creation
- Comprehensive error handling and logging

Usage:
    python 01_build_annotation_db.py [--genome GENOME_ID] [--overwrite]
    
Arguments:
    --genome: Process specific genome ID (e.g., GCF_000001405.40)
    --overwrite: Overwrite existing annotation databases
"""

#%%
import os
import multiprocessing
import sys

from lore import paths
from lore import logger

import pandas as pd
import gffutils
import tempfile
from tqdm import tqdm
import argparse
from functools import partial
#%%

logger.remove()
logger.add(sys.stderr, level="INFO")  # DEBUG and TRACE won't show now

#%%

intermediate_path = paths.get_path(data_type="data", stage="intermediate", name="refseq", version="229")
download_path = paths.get_path(data_type="data", stage="downloads", name="refseq", version="229")
rasp_path = paths.get_path(data_type="data", stage="downloads", name="rna2d", version="rasp2")

#%%

def rename_duplicates(attribute, id, i):
    if pd.notna(id) and id in attribute:
        attribute = attribute.replace(f"ID={id}", f"ID={id}-{i}")
    return attribute

def expand_annotation(genome, rasp=False):
    
    logger.debug(f"Removing duplicate IDs in {genome}")


    if rasp:
        # For RASP2, the annotation files are in a different path
        annotation_path = rasp_path / f"annotation/{genome}.gff3"
    else:
        annotation_path = os.path.join(download_path, genome, "ncbi_dataset", "data", genome, "genomic.gff")
    col_names = ["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]

    try:
        df = pd.read_csv(annotation_path, sep="\t", comment="#", names=col_names)
        logger.debug(f"Successfully read GFF file for genome: {genome} with {len(df)} records")
    except Exception as e:
        logger.error(f"Failed to read GFF file for genome: {genome}. Error: {e}")
        exit(1)
    
    # Extract IDs
    df['attribute'] = df['attribute'].fillna('')
    df["ID"] = df['attribute'].str.extract(r'ID=([^;]+)')

    # Find duplicates
    duplicate_ids = df["ID"][df["ID"].duplicated()].dropna().unique()

    if duplicate_ids.size > 0:
        logger.debug(f"Found {len(duplicate_ids)} duplicate IDs in {genome}")
        # Rename duplicates
        df['attribute'] = df.apply(lambda row: rename_duplicates(row['attribute'], row["ID"], row.name) if row["ID"] in duplicate_ids else row['attribute']
                                   , axis=1)
        df.drop(columns=["ID"], inplace=True)

    logger.debug(f"De-duplicated GFF file for {genome} with {len(df)} records")

    return df

#%%

def create_gffutils_db(gff_df, db_path):
    """
        Create a gffutils database from a GFF DataFrame.
    """

    logger.debug(f"Creating gffutils database at {db_path}")

    gff_str = '\n'.join([
        '\t'.join(map(str, row))
        for row in gff_df.values
    ])

    # Write to a temporary GFF file (it will be deleted automatically after the block)
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.gff', delete=True) as tmp_gff:
        tmp_gff.write(gff_str)
        tmp_gff.flush()  # Ensure it's written to disk

        # Create the permanent gffutils database
        gffutils.create_db(
            tmp_gff.name,
            dbfn=db_path,
            force=True,
            keep_order=True,
            merge_strategy='merge',
            sort_attribute_values=True
        )

        logger.debug(f"Database created successfully at {db_path}")

#%%

def convert_annotation_to_gffutils_db(genome, overwrite=False, rasp=False):
    db_path = os.path.join(intermediate_path, genome, "annotations.db")

    if os.path.exists(db_path):
        if overwrite:
            logger.debug(f"Database already exists at {db_path}. Overwriting as requested.")
            os.remove(db_path)
        else:
            logger.debug(f"Database already exists at {db_path}. Skipping conversion for {genome}")
            return

    gff_df = expand_annotation(genome, rasp=rasp)

    create_gffutils_db(gff_df, db_path)


#%%


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process genome data.")
    parser.add_argument("--genome", type=str, help="Specify a single genome to process.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing annotated genome files.")
    parser.add_argument("--rasp", action="store_true", help="Process data for genomes with rna2d structure.")
    args = parser.parse_args()

    # Create the base intermediate directory if it does not exist
    if not os.path.exists(intermediate_path):
        os.makedirs(intermediate_path)
        logger.debug(f"Directory created at {intermediate_path} for intermediate refseq data")


    if args.genome:
        # Process a specific genome if provided
        convert_annotation_to_gffutils_db(args.genome, overwrite=args.overwrite)

    else:

        if args.rasp:
            genomes = [f.split('.')[0] for f in os.listdir(rasp_path / "annotation") if f.endswith(".gff3")]
            logger.info(f"Preparing to process {len(genomes)} genome files in {rasp_path}")

            intermediate_path = paths.get_path(data_type="data", stage="intermediate", name="refseq", version="rna2d_rasp")

        else:
            # Run for all genomes if no genome is specified
            
            # Get genome directories to process
            genomes = [d for d in os.listdir(intermediate_path) if d.startswith("GCF") and os.path.isdir(os.path.join(intermediate_path, d))]
            logger.info(f"Preparing to process {len(genomes)} GCF directories in {intermediate_path}")

        convert_annotation_partial = partial(convert_annotation_to_gffutils_db, overwrite=args.overwrite, rasp=args.rasp)

        with multiprocessing.Pool() as pool:
            for _ in tqdm(pool.imap_unordered(convert_annotation_partial, genomes), total=len(genomes)):
                pass

            logger.info(f"Finished processing {len(genomes)} GCF directories in {intermediate_path}")
            logger.info(f"All GFF files have been converted to gffutils databases.")
