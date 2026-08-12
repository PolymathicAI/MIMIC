#!/usr/bin/env python3
"""
RefSeq Processing Pipeline - Step 3: Build GENCODE Annotation Databases

This script creates gffutils databases from GENCODE annotation files for human and mouse 
genomes. GENCODE provides high-quality manual annotation that complements RefSeq data 
with additional transcript isoforms and detailed feature annotations.

Input:
- GENCODE GFF3 annotation files for human and mouse
- Human: downloads/gencode/human/gencode.v47.chr_patch_hapl_scaff.annotation.gff3
- Mouse: downloads/gencode/mouse/gencode.vM36.annotation.gff3

Output:
- GENCODE annotation databases for human and mouse
- Human: intermediate/gencode/human/gencode.v47.chr_patch_hapl_scaff.annotations.db
- Mouse: intermediate/gencode/mouse/gencode.vM36.annotations.db

The script:
1. Processes both human and mouse GENCODE annotations in parallel
2. Copies GFF files to temporary directory for faster processing
3. Creates gffutils databases with optimized settings for GENCODE format
4. Moves completed databases to final locations in intermediate storage
5. Supports dry-run mode for testing without actual database creation

Key Features:
- Parallel processing of human and mouse genomes
- Temporary file management for improved I/O performance
- Dry-run capability for testing and validation
- Automatic cleanup of temporary files
- Comprehensive error handling and progress logging

Database Settings:
- force=True: Overwrites existing databases
- merge_strategy='merge': Handles overlapping features appropriately
- sort_attribute_values=True: Ensures consistent attribute ordering

Usage:
    python 03_build_gencode_db.py [--dry-run] [--overwrite]
    
Arguments:
    --dry-run: Test mode without creating actual databases
    --overwrite: Overwrite existing databases if they exist
"""

#%%
import os
import shutil
import sys

import gffutils

from lore import logger
from lore import paths
import tempfile
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial

logger.remove()
logger.add(sys.stderr, level="INFO")

# Base paths
mouse_base = 'gencode.vM36'
human_base = 'gencode.v47.chr_patch_hapl_scaff'


def build_gencode_db(organism, dry_run=False, overwrite=False):

    gff_path_ceph = paths.get_path(data_type="data", stage="downloads", name="gencode", version=organism) / f"{mouse_base if organism == 'mouse' else human_base}.annotation.gff3"
    db_path_ceph = paths.get_path(data_type="data", stage="intermediate", name="gencode", version=organism) / f"{mouse_base if organism == 'mouse' else human_base}.annotations.db"

    # Check if database already exists
    if os.path.exists(db_path_ceph):
        if not overwrite and not dry_run:
            logger.info(f"Database already exists at {db_path_ceph} and overwrite=False. Skipping creation.")
            return
        else:
            logger.info(f"Database already exists at {db_path_ceph} but overwrite=True. Proceeding with recreation.")

    # Create temporary files with appropriate extensions
    temp_dir = tempfile.gettempdir()
    gff_path_tmp = os.path.join(temp_dir, f"{mouse_base if organism == 'mouse' else human_base}.annotation.gff3")
    db_path_tmp = os.path.join(temp_dir, f"{mouse_base if organism == 'mouse' else human_base}.annotations.db")
    

    # Copy the GFF file from Ceph to temp directory if it doesn't exist already
    if not os.path.exists(gff_path_tmp):
        # If this is a dry run, create the GFF file in temp directory
        if dry_run:
            logger.info(f"DRY RUN: Creating empty GFF file at {gff_path_tmp}")
            # Create an empty file to simulate the GFF file
            with open(gff_path_tmp, 'w') as f:
                pass
        else:
            # Copy the actual GFF file for real processing
            logger.info(f"Copying GFF file from {gff_path_ceph} to {gff_path_tmp}")
            shutil.copy2(gff_path_ceph, gff_path_tmp)
            logger.info(f"GFF file copied successfully to {gff_path_tmp}")
    else:
        logger.info(f"Using existing GFF file at {gff_path_tmp}")
    
    
    if dry_run:
        logger.info(f"DRY RUN: Creating empty database file at {db_path_tmp}")
        # Create an empty file
        with open(db_path_tmp, 'w') as f:
            pass
    else:
        # Normal processing
        logger.info(f"Preparing to create gffutils database at {db_path_tmp} from {gff_path_tmp}")
        gffutils.create_db(
            gff_path_tmp,
            dbfn=db_path_tmp,
            force=True,
            merge_strategy='merge',
            sort_attribute_values=True
        )
    logger.info(f"Database created successfully at {db_path_tmp}")

    # Move the created database to the Ceph location
    if dry_run:
        logger.info(f"DRY RUN: Skipping database move to {db_path_ceph}")
        # Clean up the temporary database file
        if os.path.exists(db_path_tmp):
            os.remove(db_path_tmp)
            logger.info(f"Removed temporary database file at {db_path_tmp}")
    else:
        logger.info(f"Moving database from {db_path_tmp} to {db_path_ceph}")
        os.makedirs(os.path.dirname(db_path_ceph), exist_ok=True)
        shutil.move(db_path_tmp, db_path_ceph)
        logger.info(f"Database moved successfully to {db_path_ceph}")
    
    # Delete the temporary GFF file
    logger.info(f"Removing temporary GFF file at {gff_path_tmp}")
    if os.path.exists(gff_path_tmp):
        os.remove(gff_path_tmp)
        logger.info(f"Temporary GFF file removed successfully")
    else:
        logger.warning(f"Temporary GFF file at {gff_path_tmp} not found")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build GENCODE database for mouse and human')
    parser.add_argument('--dry-run', action='store_true', help='Run in dry-run mode')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing database if it exists')
    args = parser.parse_args()

    species = ['mouse', 'human']
    
    logger.info(f"Starting GENCODE database build for {species}")
    if args.dry_run:
        logger.info("Running in dry-run mode. No actual database will be created.") 
    
    # Use ProcessPoolExecutor to process both species in parallel
    
    # Create a partial function with dry_run parameter fixed
    build_with_dry_run = partial(build_gencode_db, dry_run=args.dry_run, overwrite=args.overwrite)

    with ProcessPoolExecutor(max_workers=2) as executor:
        # Submit both species for processing with the partial function
        futures = [executor.submit(build_with_dry_run, organism) for organism in species]
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()
    
    logger.info("GENCODE database build completed for all species")
