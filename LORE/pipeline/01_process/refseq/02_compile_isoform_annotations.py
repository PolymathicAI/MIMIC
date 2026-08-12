#!/usr/bin/env python3
"""
RefSeq Processing Pipeline - Step 2: Compile Isoform Annotations

This script extracts and compiles isoform annotations from gffutils databases created 
in step 1. It identifies transcript isoforms and their hierarchical relationships, 
extracting detailed feature information including exons, CDS regions, and parent genes.

Input:
- annotations.db files from step 1 (gffutils databases)
- Located in intermediate/refseq/229/{genome_name}/annotations.db

Output:
- annotated_genome.parquet files for each genome
- Located in intermediate/refseq/229/{genome_name}/annotated_genome.parquet

The script:
1. Loads gffutils databases and identifies isoforms (features with children but no grandchildren)
2. For each isoform, traverses parent hierarchy to collect gene information
3. Processes child features (exons, CDS) and extracts coordinates
4. Compiles comprehensive feature data including:
   - Feature IDs, types, and coordinates
   - Root gene information and biotypes
   - Exon and CDS coordinate lists
   - Pseudogene flags and other attributes
5. Saves results as Parquet files for efficient downstream processing

Key Features:
- Hierarchical feature relationship parsing
- Memory-efficient processing with progress monitoring
- Temporary database copying for improved I/O performance
- Parallel processing support with ProcessPoolExecutor
- Comprehensive error handling for malformed annotations

Usage:
    python 02_compile_isoform_annotations.py [--genome GENOME_ID] [--overwrite] [--verbose]
    
Arguments:
    --genome: Process specific genome ID (e.g., GCF_000001405.40)
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
import gffutils
from tqdm import tqdm
import argparse
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import concurrent
import shutil
import tempfile

#%%

intermediate_path = paths.get_path(data_type="data", stage="intermediate", name="refseq", version="229")

#%%

def load_gffutils_db(genome):

    db_path = os.path.join(intermediate_path, genome, "annotations.db")

    # Check if the database exists
    if not os.path.exists(db_path):
        logger.error(f"Database file not found at {db_path}")
        return None

    # Copy the database to a temporary location for faster access
    tmp_dir = os.path.join(tempfile.gettempdir(), "refseq_genomes")
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
    
    # Get the list of all isoforms - features with children but no grandchildren    
    isoforms = list(dict.fromkeys(
        feature for feature in db.all_features()
        if (children := list(db.children(feature, level=1)))
        and any(not list(db.children(child, level=1)) for child in children)
    ))
    
    logger.debug(f"Found {len(isoforms)} isoforms for genome {genome}")

    # Initialize a list to store genome data
    genome_data = []

    for isoform in isoforms:
        
        # Iterate over parents of isoform until reaching the root feature
        parent = isoform
        parent_info = []
        try:
            while parent:
                parent_info.append({
                    "feature_id": parent.id,
                    "feature_type": parent.featuretype,
                    "coordinates": [parent.start, parent.end],
                    "gene_biotype": parent.attributes.get('gene_biotype', [None])[0],
                })
                parent = next(db.parents(parent, level=1), None)

        except Exception as e:
            logger.error(f"Error processing isoform {isoform.id} ({isoform.featuretype}) in genome {genome}: {e}")
            continue

        # Initialize a dictionary to store child information
        child_info = {
            'exon': None,
            'CDS': None,
            'other': None  # For any other feature types that might be present
        }


        for child in db.children(isoform, level=1):
            # Do not process children that are isoforms themselves 
            # (rare case, only seen with 'primary_transcript & miRNA')
            # Most features have 2 or 3 levels only
            if list(db.children(child)):
                continue

            # If child is not CDS or exon, add to 'other' category
            if child.featuretype not in ['CDS', 'exon']:
                if child_info['other'] is None:
                    child_info['other'] = {
                        'feature_type': [],
                        'id': []
                    }
                child_info['other']['feature_type'].append(child.featuretype)
                child_info['other']['id'].append(child.id)
                
            else:
                if not child_info[child.featuretype]:                
                    child_info[child.featuretype] = {
                        'id': "-".join(child.id.split('-')[:-1]) if len(child.id.split('-')) > 2 else child.id,
                        'coordinates': []
                    }

                # Check if start and end indices are within bounds
                if child.start < parent_info[-1]['coordinates'][0] or child.end > parent_info[-1]['coordinates'][-1]:
                    logger.warning(f"Start/end out of bounds for {child.id} in genome {genome}: {child.start}/{child.end} vs parent start/end {parent_info[-1]['coordinates'][0]}/{parent_info[-1]['coordinates'][1]}")


                child_info[child.featuretype]["coordinates"].append([child.start, child.end])
    
        # Initialize a dictionary to store the feature's data
        feature_data = {
            "genome": genome,
            "feature_type": isoform.featuretype,
            "gbkey": isoform.attributes.get('gbkey', [None])[0],
            "feature_id": isoform.id,
            "seqid": isoform.seqid,
            "feature_start": isoform.start,
            "feature_end": isoform.end,
            "strand": isoform.strand,
            "root_id": parent_info[-1]['feature_id'],
            "root_type": parent_info[-1]['feature_type'],
            "root_biotype": parent_info[-1].get('gene_biotype', None),
            "root_start": parent_info[-1]['coordinates'][0],
            "root_end": parent_info[-1]['coordinates'][1],
            "CDS_id": child_info['CDS']['id'] if child_info['CDS'] else None,
            "CDS_coordinates": child_info['CDS']['coordinates'] if child_info['CDS'] else None,
            "exon_id": child_info['exon']['id'] if child_info['exon'] else None,
            "exon_coordinates": child_info['exon']['coordinates'] if child_info['exon'] else None,
            "other_child_type": child_info['other']['feature_type'] if child_info['other'] else None,
            "other_child_id": child_info['other']['id'] if child_info['other'] else None,
            'pseudo': isoform.attributes.get('pseudo', [None])[0] == 'true',
        }

        # Append the feature's data to the genome data list
        genome_data.append(feature_data)

        # Log memory usage every 100 features
        if len(genome_data) % 100 == 0:
            memory_usage = os.popen("ps -o rss= -p " + str(os.getpid())).read().strip()
            logger.debug(f"Processed {len(genome_data)} features for genome {genome}. Current memory usage: {int(memory_usage) / 1024:.2f} MB")

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
    parser.add_argument("--rasp", action="store_true", help="Process data for genomes with rna2d structure.")

    args = parser.parse_args()

    if not args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="INFO")
    else:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")


    if args.genome:
        # Process a specific genome if provided
        build_database(args.genome, overwrite=args.overwrite)

    else:
        if args.rasp:
            intermediate_path = paths.get_path(data_type="data", stage="intermediate", name="refseq", version="rna2d_rasp")
            genomes = [d for d in os.listdir(intermediate_path)]
            logger.info(f"Preparing to process {len(genomes)} GCF directories in {intermediate_path}")

        else:
        # Run for all genomes if no genome is specified
            genomes = [d for d in os.listdir(intermediate_path) if d.startswith("GCF") and os.path.isdir(os.path.join(intermediate_path, d))]
            logger.info(f"Preparing to process {len(genomes)} GCF directories in {intermediate_path}")

        build_database_partial = partial(build_database, overwrite=args.overwrite)

        max_workers = max(1, multiprocessing.cpu_count() - 2)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(build_database_partial, genome): genome for genome in genomes}
            for future in tqdm(concurrent.futures.as_completed(futures), 
                      total=len(genomes), 
                      desc="Building genome databases"):

                future.result()

        logger.info("All genome databases have been built.")

        # Clean up the temporary directory
        tmp_dir = os.path.join(tempfile.gettempdir(), "refseq_genomes")
        try:
            if os.path.exists(tmp_dir):
                logger.info(f"Cleaning up temporary directory: {tmp_dir}")
                shutil.rmtree(tmp_dir)
                logger.info(f"Temporary directory removed successfully")
        except Exception as e:
            logger.warning(f"Failed to remove temporary directory {tmp_dir}: {e}")
