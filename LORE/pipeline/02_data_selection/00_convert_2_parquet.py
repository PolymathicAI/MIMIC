#!/usr/bin/env python3
"""
Data Selection Pipeline - Step 0: Convert Modalities to Parquet Format

This script converts various data modalities from their original formats (HuggingFace Dataset 
format or chunked HuggingFace Dataset format) to Parquet format for more efficient processing 
in subsequent pipeline steps.

Input:
- config.yaml: Configuration file specifying modalities and their versions
- Data modalities in 'hfds' or 'hfds_chunked' format located in the modality data directories

Output:
- Each modality converted to Parquet format in corresponding modality directories
- dataset.parquet files created for each converted modality

The script:
1. Reads the configuration to identify which modalities need conversion
2. Checks existing format (hfds, hfds_chunked, or already parquet)
3. For chunked datasets, loads multiple chunks in parallel using multiprocessing
4. Converts and saves each modality as a single dataset.parquet file

Usage:
    python 00_convert_2_parquet.py [--test]
    
Arguments:
    --test: Run in test mode (only processes first chunk of chunked datasets)
"""
# %%

import yaml 
from lore import logger
from lore import paths
import sys
from pathlib import Path
import datasets
import multiprocessing.pool as mpp
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Set test to True")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for multiprocessing")
    args = parser.parse_args()
    test = args.test
    num_workers = args.num_workers

    if  test:
        logger.info("Test mode activated. This will only affect the hfds_chunked modalities by " \
        "only loading the first file.")
        logger.info("To run the full script, you need to manually delete the files created in the test mode.")

    with open(Path(__file__).parent / "config.yaml", "r") as file:
        config = yaml.safe_load(file)
    # %%
    # get all the different modalities
    all_mods = [{'name':name, 'version': data['version']} for name, data in config['modalities'].items()]
    # update with the mutually_exclusive_modalities
    all_mods += [{'name':name, 'version': data['version']} for name, data in config['mutually_exclusive_modalities'].items()]
    convert_mods = []

    # iterate through the modalities and check what format they are in
    for mod in all_mods:

        # check to see if the mod is already in convert_mods list
        versions = [m['version'] for m in convert_mods if m['name'] == mod['name']]
        if mod['version'] in versions:
            logger.info(f"Modality {mod['name']} version {mod['version']} already in convert_mods list. Skipping...")
            continue

        root_path = paths.get_path(data_type="data", stage="modality", name=mod['name'], version=str(mod['version']))
        if not root_path.exists():
            raise FileNotFoundError(f"Path does not exist: {root_path}")
        folder_names = [f.name for f in root_path.iterdir() if f.is_dir()]
        if len(folder_names) == 0:
            raise ValueError(f"No subfolders found in {root_path}")
        if "parquet" in folder_names:
            logger.info(f"Parquet folder found for {mod['name']} at {root_path}.")
            continue
        elif "hfds" in folder_names:
            logger.info(f"Using the HFDS folder found for {mod['name']} at {root_path}.")
            source_fmt = "hfds"
        elif "hfds_chunked" in folder_names:
            logger.info(f"Using the HFDS chunked folder found for {mod['name']} at {root_path}.")
            source_fmt = "hfds_chunked"
        else:   
            raise ValueError(f"Unknown format {folder_names} found for {mod['name']} at {root_path}.")
        
        convert_mods.append({"source_fmt": source_fmt, "name": mod['name'], "version": mod['version']})

    conversions = [f"{mod['name']}: {mod['source_fmt']} -> parquet" for mod in convert_mods]
    if len(conversions) == 0:
        logger.info("No conversions to be made. Exiting...")
        return
    
    logger.info(f"Converting the following modalities to parquet:")
    for conversion in conversions:
        logger.info("       " + conversion)

    # %%

    def load_chunked_hfds(path: Path, test=False, num_workers=8) -> datasets.Dataset:
        
        # get the list of files in the folder
        folders = [file for file in path.iterdir() if file.is_dir()]

        if len(folders) == 0:
            raise ValueError(f"No files found in {path}")
        
        logger.info(f"Found {len(folders)} files in {path}")

        if test:
            folders = folders[:1]
            logger.info(f"Test mode: only loading the first file")

        logger.info(f"Starting to load the datasets with multiprocessing with {num_workers} workers")
        
        # load the datasets in parallel using multiprocessing with the specified number of workers
        with mpp.ThreadPool(num_workers) as pool:
            datatest_chunks = pool.map(datasets.load_from_disk, folders)

        # concatenate the datasets
        return datasets.concatenate_datasets(datatest_chunks)

    # %%

    logger.info("Starting conversions to parquet...")

    for mod in convert_mods:
        source_path = paths.get_path(data_type="data", stage="modality", name=mod['name'], version=str(mod['version']), fmt=mod['source_fmt'])
        dest_path = paths.get_path(data_type="data", stage="modality", name=mod['name'], version=str(mod['version']), fmt="parquet")
        pqt_file = dest_path / "dataset.parquet"
        if mod["source_fmt"] == "hfds_chunked":
            hf_dataset = load_chunked_hfds(source_path, test=test, num_workers=num_workers)
        elif mod["source_fmt"] == "hfds":
            hf_dataset = datasets.load_from_disk(source_path)
        logger.info(f"Finished loading dataset with shape {hf_dataset.shape}.")
        logger.info(f"Starting conversion to parquet...")
        dest_path.mkdir(parents=True, exist_ok=False)
        hf_dataset.to_parquet(pqt_file)
        logger.info(f"Finished converting {mod} dataset to parquet.")

if __name__ == "__main__":
    main()