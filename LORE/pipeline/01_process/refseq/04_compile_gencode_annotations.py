#!/usr/bin/env python3
"""
RefSeq Processing Pipeline - Step 4: Compile GENCODE Annotations

This script processes GENCODE gffutils databases to extract transcript annotations 
with higher quality features than RefSeq. GENCODE provides manually curated annotations 
including detailed UTR regions, start/stop codons, and transcript type classifications.

Input:
- GENCODE annotations.db files from step 3
- Located in intermediate/gencode/{organism}/*.annotations.db

Output:
- annotated_genome.parquet files for human and mouse
- Located in intermediate/gencode/{organism}/annotated_genome.parquet

The script:
1. Loads GENCODE gffutils databases for human and mouse
2. Extracts transcript features and their hierarchical relationships
3. Processes child features including:
   - Exons and CDS regions
   - 5' and 3' UTR regions
   - Start and stop codons
   - Selenocysteine-containing stop codons
4. Compiles comprehensive transcript data with gene type information
5. Saves results as Parquet files for downstream merging

Key Features:
- Focuses on transcript features (higher quality than RefSeq isoforms)
- Extracts detailed UTR and codon information not available in RefSeq
- Handles GENCODE-specific feature types and transcript classifications
- Memory-efficient processing with progress monitoring
- Parallel processing support for multiple organisms
- Temporary database copying for improved I/O performance

GENCODE-Specific Features:
- transcript_type: Detailed transcript classification
- gene_type: Gene biotype classification
- UTR regions: 5' and 3' untranslated regions
- Codon features: Start/stop codons and selenocysteine

Usage:
    python 04_compile_gencode_annotations.py [--genome ORGANISM] [--overwrite] [--verbose]
    
Arguments:
    --genome: Process specific organism (human or mouse)
    --overwrite: Overwrite existing annotated genome files
    --verbose: Enable detailed debug logging
"""

#%%
import os
import sys
import multiprocessing

from lore import paths
from lore import logger

import pandas as pd
import h5py
import gffutils
import numpy as np
from tqdm import tqdm
import argparse
from functools import partial
import tempfile
import shutil

#%%
intermediate_path = paths.get_path(data_type="data", stage="intermediate", name="gencode")

#%%

def load_gffutils_db(genome):

    db_path = next(
        (os.path.join(root, file) for root, _, files in os.walk(os.path.join(intermediate_path, genome)) 
         for file in files if file.endswith("annotations.db")), 
        None
    )

        # Check if the database exists
    if not os.path.exists(db_path):
        logger.error(f"Database file not found at {db_path}")
        return None

    # Copy the database to a temporary location for faster access
    tmp_dir = os.path.join(tempfile.gettempdir(), "gencode_genomes")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_db_path = os.path.join(tmp_dir, f"gffutils_{genome}_db")
    try:
        logger.debug(f"Copying database from {db_path} to {tmp_db_path}")
        shutil.copy(db_path, tmp_db_path)
        db_path = tmp_db_path
    except Exception as e:
        logger.warning(f"Failed to copy database to temp location: {e}. Using original path.")



    try:
        logger.debug(f"Loading existing gffutils database from {db_path}")
        return gffutils.FeatureDB(db_path, keep_order=True)
    except Exception as e:
        logger.error(f"Failed to load existing database: {e}")
        return None
    

#%%

def build_database(genome, overwrite = False):
    """
    Build a database for the given genome.
    """

    output_path = intermediate_path / genome / "annotated_genome.parquet"
    if not overwrite and os.path.exists(output_path):
        logger.debug(f"Annotated genome already exists at {output_path}. Skipping genome: {genome}")
        return

    # Create the gffutils database
    db = load_gffutils_db(genome)
    if db is None:
        logger.debug(f"Skipping genome {genome} due to missing annotation db.")
        return
    logger.debug(f"Processing genome: {genome}")

    # Initialize a list to store genome data
    genome_data = []

    for isoform in db.features_of_type("transcript"):

        # Iterate over parents of isoform until reaching the root feature
        parent = isoform
        parent_info = []
        try:
            while parent:
                parent_info.append({
                    "feature_id": parent.id,
                    "feature_type": parent.featuretype,
                    "gene_type": parent.attributes.get('gene_type', [None])[0],
                    "coordinates": [parent.start, parent.end]
                })
                parent = next(db.parents(parent, level=1), None)

        except Exception as e:
            logger.error(f"Error processing isoform {isoform.id} ({isoform.featuretype}) in genome {genome}: {e}")
            continue

        # Initialize a dictionary to store child information
        child_info = {
            'exon': None,
            'CDS': None,
            'five_prime_UTR': None,
            'three_prime_UTR': None,
            'start_codon': None,
            'stop_codon': None,
            'stop_codon_redefined_as_selenocysteine': None
        }


        for child in db.children(isoform, level=1):
            # Do not process children that are isoforms themselves 
            # (rare case, only seen with 'primary_transcript & miRNA')
            # Most features have 2 or 3 levels only
            if list(db.children(child)):
                continue

            # If child is not CDS or exon, skip and throw warning
            if child.featuretype not in child_info.keys():
                logger.warning(f'Warning: Child feature {child.id} in genome {genome} has type other than [CDS | exon]. Skipping feature.')
                continue

            if not child_info[child.featuretype]:                
                child_info[child.featuretype] = {
                    'id': "-".join(child.id.split('-')[:-1]) if len(child.id.split('-')) > 2 else child.id,
                    # 'mask': np.zeros(len(sequence), dtype=int)
                    'coordinates': []
                    
                }
                
            # # Check if start and end indices are within bounds
            # if start_idx < 0 or end_idx > len(sequence):
            if child.start < parent_info[-1]['coordinates'][0] or child.end > parent_info[-1]['coordinates'][1]:
                logger.warning(f"Start/end out of bounds for {child.id} in genome {genome}: {child.start}/{child.end} vs parent start/end {parent_info[-1]['coordinates'][0]}/{parent_info[-1]['coordinates'][1]}")

            child_info[child.featuretype]["coordinates"].append([child.start, child.end])
    
        # Initialize a dictionary to store the feature's data
        feature_data = {
            "feature_type": isoform.featuretype,
            "transcript_type": isoform.attributes.get('transcript_type', [])[0],
            "feature_id": isoform.id,
            "seqid": isoform.seqid,
            "feature_start": isoform.start,
            "feature_end": isoform.end,
            "strand": isoform.strand,
            "root_id":  parent_info[-1]['feature_id'],
            "root_type": parent_info[-1]['feature_type'],
            "root_gene_type": parent_info[-1]['gene_type'],
            "root_start": parent_info[-1]['coordinates'][0],
            "root_end": parent_info[-1]['coordinates'][1],
            "CDS_id": child_info['CDS']['id'] if child_info['CDS'] else None,
            "CDS_coordinates": child_info['CDS']['coordinates'] if child_info['CDS'] else None,
            "exon_id": child_info['exon']['id'] if child_info['exon'] else None,
            "exon_coordinates": child_info['exon']['coordinates'] if child_info['exon'] else None,
            "five_prime_UTR_coordinates": child_info['five_prime_UTR']['coordinates'] if child_info['five_prime_UTR'] else None,
            "three_prime_UTR_coordinates": child_info['three_prime_UTR']['coordinates'] if child_info['three_prime_UTR'] else None,
            "start_codon_coordinates": child_info['start_codon']['coordinates'] if child_info['start_codon'] else None,
            "stop_codon_coordinates": child_info['stop_codon']['coordinates'] if child_info['stop_codon'] else None,
            "stop_codon_redefined_as_selenocysteine_coordinates": child_info['stop_codon_redefined_as_selenocysteine']['coordinates'] if child_info['stop_codon_redefined_as_selenocysteine'] else None
        }

        # Append the feature's data to the genome data list
        genome_data.append(feature_data)

        # Log memory usage every 100 features
        if len(genome_data) % 1000 == 0:
            memory_usage = os.popen("ps -o rss= -p " + str(os.getpid())).read().strip()
            logger.debug(f"Processed {len(genome_data)} features. Current memory usage: {int(memory_usage) / 1024:.2f} MB")


    # Create a DataFrame from the genome data
    genome_df = pd.DataFrame(genome_data)

    # Save the dataframe to a parquet file
    logger.debug(f"Saving annotated genome to {output_path}")
    genome_df.to_parquet(output_path, index=False)
    logger.debug(f"Saved annotated genome with {len(genome_df)} features to {output_path}")

#%%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process genome data.")
    parser.add_argument("--genome", type=str, help="The genome directory to process.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing annotated genome files.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if not args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    if args.genome:
        # Process a specific genome if provided
        build_database(args.genome, overwrite=args.overwrite)

    else:
        # Run for all genomes if no genome is specified
        genomes = ["human", "mouse"]
        logger.info(f"Preparing to process {len(genomes)} GCF directories in {intermediate_path}")

        build_database_partial = partial(build_database, overwrite=args.overwrite)

        with multiprocessing.Pool() as pool:
            for _ in tqdm(pool.imap(build_database_partial, genomes), total=len(genomes)):
                pass

        logger.info("All genome databases have been built.")

            # Clean up the temporary directory
        tmp_dir = os.path.join(tempfile.gettempdir(), "gencode_genomes")
        try:
            if os.path.exists(tmp_dir):
                logger.info(f"Cleaning up temporary directory: {tmp_dir}")
                shutil.rmtree(tmp_dir)
                logger.info(f"Temporary directory removed successfully")
        except Exception as e:
            logger.warning(f"Failed to remove temporary directory {tmp_dir}: {e}")
