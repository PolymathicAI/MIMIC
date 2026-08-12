#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path
from lore import logger
from lore import paths 
import yaml
import importlib
import time

convert_to_parquet = importlib.import_module("00_convert_2_parquet").main
build_master_ids = importlib.import_module("01_build_master_ids").main
make_splits = importlib.import_module("02_make_splits").main
select_dataset = importlib.import_module("03_select_dataset").main
make_val_subsets = importlib.import_module("04_make_val_subsets").main

def main():

    start_time = time.time()

    config_file_path = Path(__file__).parent / "config.yaml"
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_file_path.name}")

    logger.info(f"--- Starting build process for master id version {config['master_id_version']} with seed {config['seed']} ---")

    int_output_dir = paths.get_path(
        data_type="data", 
        stage="intermediate", 
        name="master_ids", 
        version=config["master_id_version"], 
        fmt="parquet"
    ) 

    mod_output_dir = Path(
      paths.get_path(
          data_type="data",
          stage="modality",
          name="id",
          version=config["master_id_version"],
          fmt="parquet",
      )
  )

    if int_output_dir.exists() or mod_output_dir.exists():
        # log all the folder that exist and exit.
        if int_output_dir.exists():
            logger.error(f"- Intermediate output directory: {int_output_dir}. Please remove it before running this script.")
        if mod_output_dir.exists():
            logger.error(f"- Modality output directory: {mod_output_dir}. Please remove it before running this script.")
        sys.exit(1)

    funcs_to_run = {
        'convert_to_parquet': convert_to_parquet, 
        'build_master_ids': build_master_ids, 
        'make_splits': make_splits, 
        'select_dataset': select_dataset, 
        'make_val_subsets': make_val_subsets
    }
    for i, (name, func) in enumerate(funcs_to_run.items()):
        print()
        logger.info(f"--- [{i+1}/{len(funcs_to_run)}] Running {name} ---")
        print()
        func()

    logger.info(f"--- Finished build process for master id version {config['master_id_version']} ---")
    total_time = time.time() - start_time
    minutes, seconds = divmod(total_time, 60)
    logger.info(f"Total time taken: {int(minutes)} minutes and {seconds:.2f} seconds")

if __name__ == "__main__":
    main()