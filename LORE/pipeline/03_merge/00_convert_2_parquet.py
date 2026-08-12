#!/usr/bin/env python3
# %%

import yaml 
from lore import logger
from lore import paths
import sys
from pathlib import Path
import datasets
import multiprocessing.pool as mpp
import argparse

def main(debug=False):

    if debug:
        logger.warning("Running in debug mode: only a subset of data will be processed.")

    with open(Path(__file__).parent / "config.yaml", "r") as file:
        config = yaml.safe_load(file)
    # %%
    # get all the different modalities
    all_mods = [{'name':name, 'version': data['version']} for name, data in config['modalities'].items()]
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

    def load_chunked_hfds(path: Path):
        
        # get the list of files in the folder
        folders = [file for file in path.iterdir() if file.is_dir()]

        if len(folders) == 0:
            raise ValueError(f"No files found in {path}")
        
        logger.info(f"Found {len(folders)} files in {path}")

        if debug:
            folders = folders[:1]
            logger.info(f"Debug mode: only loading the first file")

        logger.info("Starting to load the datasets with multiprocessing with 8 workers")
        
        # load the datasets in parallel using multiprocessing with 8 workers
        with mpp.ThreadPool(8) as pool:
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
            hf_dataset = load_chunked_hfds(source_path, test=test)
        elif mod["source_fmt"] == "hfds":
            hf_dataset = datasets.load_from_disk(source_path)
        logger.info(f"Finished loading dataset with shape {hf_dataset.shape}.")
        logger.info(f"Starting conversion to parquet...")
        dest_path.mkdir(parents=True, exist_ok=False)
        hf_dataset.to_parquet(pqt_file)
        logger.info(f"Finished converting {mod} dataset to parquet.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert modalities to parquet.")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run in debug mode (fewer rows).",
    )
    args = parser.parse_args()
    main(debug=args.debug)