#!/usr/bin/env python3
# %%
import pandas as pd
from lore import paths
from lore import logger
from lore.utils.wds_utils import write_dataset
import time
from pathlib import Path
import yaml, argparse
from mimic.modality_info import GROUP_INFO as group_info
import multiprocessing as mp
import shutil
import sys
import os 
from tqdm import tqdm
import duckdb

# Set start method to 'fork' for memory sharing (as in original script)
# This is required for the `write_dataset` function's use of a global
mp.set_start_method("fork", force=True)

def process_split(args):
    """
    Worker function to process a single (subset, split) pair.
    This function will be run in parallel.
    """
    # 1. Unpack arguments
    (
        subset_dir,
        split,
        length_splits,
        min_split_size,
        wds_path,
        config,
        group_info,
        parallel_mode,
    ) = args

    log_prefix_base = f"[{subset_dir.stem} | {split}]:"

    split_path = subset_dir / split

    wds_subset_path = wds_path / subset_dir.stem

    try:
        logger.debug(f"{log_prefix_base} Loading dataset...")
        df = pd.read_parquet(split_path)
        logger.debug(
            f"{log_prefix_base} Loaded dataset with shape {df.shape}. Processing length splits..."
        )

        stats = {}
        acum_rows = 0

        # find a column for each modality group
        group_cols = [
            df.columns.intersection(info["mods"])[0]
            for info in group_info.values()
            if any(df.columns.intersection(info["mods"]))
        ]
        # calculate the sum of the lengths of the columns for each row
        lens = df[group_cols].apply(lambda x: sum([len(el) for el in x]), axis=1)

        loop_iterator = range(len(length_splits) - 1)
        if parallel_mode == "process":
            loop_iterator = tqdm(
            loop_iterator,
            desc=f"{log_prefix_base} Length splits",
            leave=True,
            )

        for i in loop_iterator:
            log_prefix = f"{log_prefix_base} [Bin {i+1}/{len(length_splits) - 1}]:"

            start = length_splits[i]
            end = length_splits[i + 1]

            # write a subset of the dataset for each length
            # Check if this is the last bin in the list
            if i == len(length_splits) - 2:
                # This is the last bin, so we take everything > start
                df_subset = df[lens > start]
            else:
                # This is a normal bin, apply the (start, end] bounds
                df_subset = df[(lens > start) & (lens <= end)]

            if len(df_subset) < min_split_size:
                logger.debug(
                    f"{log_prefix} Number of rows ({len(df_subset):,}) for subset of length {start:,} to {end:,} is less than {min_split_size:,}. Skipping."
                )
                stats[i] = {
                    "start": start,
                    "end": end,
                    "rows": 0,
                }
                continue

            logger.debug(
                f"{log_prefix} Writing subset for length {start:,} to {end:,} with {len(df_subset):,} rows..."
            )
            
            write_dataset(
                dataset=df_subset,
                base_path=wds_subset_path,
                dataset_name=f"{split}_{i:03d}",
                max_samples_per_tar=config["samples_per_tar"],
                parallel=True,
                log_prefix=log_prefix,
                parallel_mode=parallel_mode,
            )
            stats[i] = {
                "start": start,
                "end": end,
                "rows": len(df_subset),
            }

            acum_rows += len(df_subset)

        logger.debug(
            f"{log_prefix_base} Dropped {len(df) - acum_rows:,} of {len(df):,} rows (either size too small or length greater than {length_splits[-1]})."
        )
        
        split_stats = {
            "rows": acum_rows,
            "length_splits": stats,
        }

        logger.debug(f"{log_prefix_base} Finished writing webdataset.")
        return (subset_dir.stem, split, split_stats)

    except Exception as e:
        logger.error(f"{log_prefix_base} FAILED with error: {e}")
        return (subset_dir.stem, split, None)


# --- Main execution block ---
# ALL logic must be inside this guard for multiprocessing to work
def main(serial=False):

    if serial:
        logger.warning("Running in serial mode. This will disable parallelization.")

    # %%
    # Load config
    # Assuming __file__ is defined if not in ipykernel.
    # For interactive/notebook use, this path might need to be set manually.
    try:
        config_path = Path(__file__).parent / "tokenize_config.yaml"
    except NameError:
        logger.warning("`__file__` is not defined. Assuming config is in current dir.")
        config_path = Path.cwd() / "tokenize_config.yaml"
        
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    logger.info(f"Loaded config from {file.name}")

    ds_name = config["name"]
    ds_version = str(config["version"])
    length_splits = config["length_splits"]
    min_split_size = config["min_split_size"]
    val_length_splits = config.get("val_length_splits", length_splits)

    # Setup paths
    tokenized_path = paths.get_path(
        data_type="data", name=ds_name, stage="final", version=ds_version, fmt="parquet"
    )
    copied_config_path = tokenized_path / "tokenize_config.yaml"

    # assert the config file matches the copied config file
    with open(copied_config_path, "r") as file:
        copied_config = yaml.safe_load(file)
    if copied_config != config:
        raise ValueError(
            f"Config file {copied_config_path} does not match the current config. This implies a version conflict."
        )

    subsets = [el for el in tokenized_path.iterdir() if el.is_dir()]

    wds_path = paths.get_path(
        data_type="data", name=ds_name, stage="final", version=ds_version, fmt="wds"
    )

    # assert that the webdataset path does not exist.
    assert (
        not wds_path.exists()
    ), f"{wds_path} already exists. Delete before running this script."

    # %%
    # --- Parallel Execution Setup ---
    start_time = time.time()

    # Create all tasks
    all_tasks_args = []
    all_splits = ["train", "val", "test"]
    parallel_task_args = []
    serial_task_args = []

    for subset_dir in subsets:
        for split in all_splits:

            if not (subset_dir / split).exists():
                logger.warning(f"[{subset_dir.stem} | {split}] The path does not exist. Skipping.")
                continue
            
            # Package all necessary arguments for the worker
            args = (
                subset_dir,
                split,
                length_splits if split == "train" else val_length_splits,
                min_split_size,
                wds_path,
                config,
                group_info,
            )
            
            # get the number of rows in the split using duckdb
            row_count = duckdb.query(f"SELECT COUNT(*) FROM '{subset_dir / split / 'dataset.parquet'}'").fetchone()[0]

            # if rna_seq is in the subset_dir.stem and row_count > 1_000_000 then run serially
            if 'rna_seq' in subset_dir.stem and row_count > 1_000_000:
                args = args + ("process",)
                serial_task_args.append(args)
            else:
                args = args + ("thread",)
                parallel_task_args.append(args)


    logger.info(f"Prepared {len(parallel_task_args)} parallel tasks and {len(serial_task_args)} serial tasks.")
    logger.info(f"Serial tasks are: {[ (a[0].stem, a[1]) for a in serial_task_args ]}")

    subset_stats = {s.stem: {} for s in subsets}
    results = []

    if serial:
        logger.info("Running in serial mode...")
        # Run tasks serially
        all_tasks_args = serial_task_args + parallel_task_args
        for args in all_tasks_args:
            results.append(process_split(args))
    else:
        # Run tasks in parallel
        max_workers = min(len(parallel_task_args), os.cpu_count() or 8) 
        logger.info(f"Spawning a pool of {max_workers} workers for {len(parallel_task_args)} parallel tasks...")
        
        logger.info("Processing serial tasks...")
        for args in serial_task_args:
            results.append(process_split(args))

        logger.info("Processing parallel tasks...")
        with mp.Pool(processes=max_workers) as pool:
            pbar = tqdm(pool.imap_unordered(process_split, parallel_task_args), 
                        total=len(parallel_task_args), 
                        desc="Processing subsets/splits")
            for result in pbar:
                results.append(result)


    # %%
    # Process results from all workers
    logger.info("All workers finished. Assembling statistics...")
    for res in results:
        if res is not None:
            subset_stem, split, stats = res
            if stats is not None:
                subset_stats[subset_stem][split] = stats
            else:
                logger.error(f"Task for {subset_stem} | {split} failed and returned no stats.")

    # %%
    # write the stats to a yaml file
    stats_path = wds_path / "stats.yaml"
    with open(stats_path, "w") as file:
        yaml.dump(subset_stats, file, sort_keys=False)
    logger.info(f"Stats written to {stats_path}")

    # copy the config file to the webdataset path
    copied_config_path = wds_path / "tokenize_config.yaml"
    shutil.copyfile(config_path, copied_config_path)
    logger.info(f"Copied config file to {copied_config_path}")

    elapsed_time = time.time() - start_time
    m, s = divmod(elapsed_time, 60)
    logger.info(f"Total time: {int(m)} minutes and {int(s)} seconds.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Tokenize dataset modalities.")
    parser.add_argument(
        "--serial",
        action="store_true",
        default=False,
        help="Run in serial mode (disables parallelization).",
    )

    serial = parser.parse_args().serial
    main(serial=serial )