#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path
from lore import logger
from lore import paths 
import yaml
import importlib
import argparse
from functools import partial
import time
import gc

convert_to_parquet = importlib.import_module("00_convert_2_parquet").main
find_subsets = importlib.import_module("01_find_subsets").main
merge_script = importlib.import_module("02_merge_script").main
tokenize_dataset = importlib.import_module("03_tokenize_dataset").main
write_webdataset = importlib.import_module("04_write_webdataset").main
make_gittest_ds = importlib.import_module("05_make_gittest_ds").main
make_val_tar = importlib.import_module("06_make_val_tar").main
validate_merged_parquet = importlib.import_module("02.1_validate_merged_parquet").main
validate_tokenization = importlib.import_module("03.1_validate_tokenization").main

def main(debug=False, serial=False, test_frac=0.15, skip_checks=False):
    """
    Runs all the numbered python scripts in order.
    
    This script should be run from an activated virtual environment.
    """

    start_time = time.time()

    if debug:
        logger.warning(f"Running in debug mode: only {test_frac:.0%} of data will be processed.")
    if serial:
        logger.warning("Serial mode is enabled in debug mode.")

    config_file_path = Path(__file__).parent / "config.yaml"
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_file_path.name}")

    logger.info(f"--- Starting build process for merged dataset version {config['version']} with master_id version {config['master_id_version']} with seed {config['seed']} ---")

    merge_output_path = paths.get_path(data_type="data", stage="merged", version=str(config["version"]), fmt=config["fmt"], name=config["name"])
    tokenize_output_path = paths.get_path(data_type='data', stage="final", version=config["version"], fmt="parquet", name=config["name"])
    webds_output_path = paths.get_path(data_type='data', stage="final", version=config["version"], fmt="wds", name=config["name"])
    webds_gittest_output_path = paths.get_path(data_type='data', stage="final", version=str(config["version"]) + "_gittest", fmt="wds", name=config["name"])
    val_tar_path = paths.get_path(data_type='data', stage="final", version=config["version"], fmt="tar", name=config["name"])
    val_git_test_tar_path = paths.get_path(data_type='data', stage="final", version=str(config["version"]) + "_gittest", fmt="tar", name=config["name"])
    
    asset_folders = {"assets": Path("./assets"),
                     "merge_output": merge_output_path,
                     "tokenize_output": tokenize_output_path,
                     "webds_output": webds_output_path,
                     "webds_gittest_output": webds_gittest_output_path,
                     "val_tar_output": val_tar_path,
                     "val_git_test_tar_output": val_git_test_tar_path}

    # if any of the output paths exist, ask user if they want to delete them
    if any([path.exists() for path in asset_folders.values()]):
        for key, path in asset_folders.items():
            if path.exists():
                logger.error(f"Path for {key} already exists at {path}. Please remove it before running this script.")
        sys.exit(1)

    funcs_to_run = {
        'convert_to_parquet': partial(convert_to_parquet, debug=debug),
        'find_subsets': partial(find_subsets, debug=debug, test_frac=test_frac),
        'merge_script': partial(merge_script, debug=debug, test_frac=test_frac),
        'validate_merged_parquet': partial(validate_merged_parquet),
        'tokenize_dataset': partial(tokenize_dataset, debug=debug, serial=serial),
        'validate_tokenization': partial(validate_tokenization),
        'write_webdataset': partial(write_webdataset, serial=serial),
        'make_gittest_ds': make_gittest_ds,
        'make_val_tar': make_val_tar
    }

    for i, (name, func) in enumerate(funcs_to_run.items()):
        print()
        if skip_checks and "validate" in name:
            logger.info(f"--- [{i+1}/{len(funcs_to_run)}] Skipping {name} as per user request ---")
            continue
        logger.info(f"--- [{i+1}/{len(funcs_to_run)}] Running {name} ---")
        print()
        func()
        gc.collect()

    logger.info(f"--- Finished build process for merged dataset version {config['version']} with master_id version {config['master_id_version']} ---")
    total_time = time.time() - start_time
    minutes, seconds = divmod(total_time, 60)
    logger.info(f"Total time taken: {int(minutes)} minutes and {seconds:.2f} seconds")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Tokenize dataset modalities.")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run in debug mode (fewer rows).",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        default=False,
        help="Run in serial mode (disables parallelization).",
    )
    parser.add_argument(
        "--skip_checks",
        action="store_true",
        default=False,
        help="Skip validation checks during the build process.",
    )
    parser.add_argument("--test_frac", type=float, default=0.15, help="Fraction of data to use in test mode.")
    args = parser.parse_args()

    logger.info(f"Running build_all.py with debug={args.debug} and serial={args.serial} and test_frac={args.test_frac}")

    main(debug=args.debug, serial=args.serial, test_frac=args.test_frac, skip_checks=args.skip_checks)