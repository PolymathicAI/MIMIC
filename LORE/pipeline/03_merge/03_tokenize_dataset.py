#!/usr/bin/env python3
# %%
import os, sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from pathlib import Path
import yaml
import argparse
import shutil

import pandas as pd
import duckdb
import copy
import uuid
from tqdm import tqdm
tqdm.pandas()

from lore import paths
from lore import logger
from mimic.modality_info import MODALITY_INFO


def tokenize_modality(args):
    subset_dir, mod, task_cpus, pos, serial, debug, tmp_out_dir, tokenizers = args
    if not serial:
        os.environ["OMP_NUM_THREADS"] = str(task_cpus)
    else:
        pos = None
    
    df = pd.read_parquet(subset_dir, columns=[mod, '__key__'])
    if debug: # Limit to 1000 rows in debug mode
        df = df.head(1_000)

    tqdm.pandas(desc=f"Tokenizing {mod:<15}", 
                position=pos, leave=True)
    # if running via slurm use apply but if running locally use progress_apply
    if os.isatty(1):
        # Local terminal, use progress_apply
        mod_tok = df[mod].progress_apply(lambda x: tokenizers[mod].tokenize(x))
    else:
        # Likely running via SLURM or non-interactive, use apply
        mod_tok = df[mod].apply(lambda x: tokenizers[mod].tokenize(x))

    output_path = tmp_out_dir / f"{mod}_{uuid.uuid4()}.parquet"
    
    out_df = pd.DataFrame({
        '__key__': df['__key__'],
        mod: mod_tok
    })
        
    # remove the compression for faster write
    out_df.to_parquet(output_path, compression="none")

    return {mod: output_path}

def main(serial=False, debug=False):
    
    start_time = time.time()

    logger.warning("This script uses a lot of memory. Run it on a node with at least 1024GB of RAM.")
    logger.info(f"This will take about 90 minutes but may take longer.")

    if serial:
        logger.warning("Running in serial mode. This will disable parallelization.")
    if debug:
        logger.warning("Running in debug mode: only a subset of data will be processed.")


    # %%

    # ── LOAD YAML CONFIG AND CONFIRM CONFIGS MATCH ─────────────────────────────────────────────
    with open(Path(__file__).parent / "merge_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    logger.info(f"Loaded config from {file.name}")

    # define modalities as the combination of config["modalities"] and config["mutually_exclusive_modalities"]
    modalities = copy.deepcopy(config["modalities"])
    modalities.update(config.get("mutually_exclusive_modalities", {}))
    # drop the ignore mods from modalities

    ds_name = config["name"]
    ds_version = str(config["version"])

    merged_path = paths.get_path(data_type="data", stage="merged", version=ds_version, fmt="parquet", name=ds_name)
    tokenized_path = paths.get_path(data_type='data', stage="final", version=ds_version, fmt="parquet", name=ds_name)

    copied_config_path = merged_path / "merge_config.yaml"
    # assert the config file matches the copied config file
    with open(copied_config_path, "r") as file:
        copied_config = yaml.safe_load(file)
    if copied_config != config:
        # go through the config and report all differences
        for key in config:
            if key not in copied_config:
                logger.error(f"Key {key} not in copied config.")
            elif config[key] != copied_config[key]:
                logger.error(f"Key {key} differs. Original: {copied_config[key]}, New: {config[key]}")
        raise ValueError(f"Config file {copied_config_path} does not match the current config. This implies a version conflict.")

    # assert that the tokenized path does not exist.
    assert not tokenized_path.exists(), f"{tokenized_path} already exists. Delete before running this script."

    # %%
    # Get all modalities and make sure tokenizers exist. Add the tokenizer version to the config.

    subsets = [path for path in merged_path.iterdir() if path.is_dir()]
    logger.info(f"Found {len(subsets)} subsets in {merged_path}")

    merge_keys = {el["merge_key"] for el in modalities.values()}

    all_mods = {mod for subset in subsets if '+' in subset.name for mod in subset.name.split("+")}
    all_cols = {col for mod in all_mods for col in modalities[mod]["column_mapping"].values()}
    logger.info(f"All modalities found: {all_cols}")

    pre_tok_mods = {mod for mod, mod_data in modalities.items() if str(mod_data["version"]).endswith("tok")}
    pre_tok_cols = {col for mod in pre_tok_mods for col in modalities[mod]["column_mapping"].values()}
    logger.info(f"Pre-tokenized columns (will not be tokenized again): {pre_tok_cols}")

    cols_to_tokenize = all_cols - pre_tok_cols
    logger.info(f"Columns to tokenize: {cols_to_tokenize}")

    tokenizers = {k.split("tok_")[1]: v["tokenizer"] for k, v in MODALITY_INFO.items() if "tok_" in k}

    # assert all modalities have a tokenizer
    for mod in all_cols:
        assert mod in tokenizers, f"{mod} not in tokenizers"
    logger.info(f"All modalities have matched tokenizers")

    # go through all modalities and add the tokenizer version to the config
    for mod in all_mods:
        tokenizer_versions = {}
        for col in modalities[mod]["column_mapping"].values():
            tokenizer_versions[col] = tokenizers[col].version
            # if tokenizer has data_version attribute, assert it matches the dataset version
            if hasattr(tokenizers[col], "data_version"):
                assert tokenizers[col].data_version == str(modalities[mod]['version']), \
                    f"Tokenizer {col} has data version {tokenizers[col].data_version} but the modality version is {config['modalities'][mod]['version']}. Please update the tokenizer or the dataset version."
        if mod in config['modalities']:
            config['modalities'][mod]["tokenizer_versions"] = tokenizer_versions
        else:
            config['mutually_exclusive_modalities'][mod]["tokenizer_versions"] = tokenizer_versions
        # if the modality is pre-tokenized, log a warning to manually verify the versions match     
        if mod in pre_tok_mods and hasattr(tokenizers[mod], "data_version"):
            logger.warning(f"Please verify manually that the version used for tokenizing {mod} matches {tokenizers[mod].data_version}.")
    logger.info(f"Added tokenizer versions to config and verified data versions match.")

    # remove the mods in config that are not in all_mods
    logger.info("Removing modalities that are not in the data subsets...")
    for mod in list(modalities.keys()):
        if mod not in all_mods:
            logger.warning(f"Modality {mod} does not appear in the data subsets. Removing it.")
            del modalities[mod]
            if mod in config["modalities"]:
                del config["modalities"][mod]
            elif mod in config.get("mutually_exclusive_modalities", {}):
                del config["mutually_exclusive_modalities"][mod]
        # also remove the extra validations that have any mods missing
    if "extra_validations" in config:
        for val_name in list(config["extra_validations"].keys()):
            val_mods = config["extra_validations"][val_name]['modalities']
            if any([mod not in all_mods for mod in val_mods]):
                logger.info(f"Removing extra validation {val_name} due to missing modality {mod}.")
                del config["extra_validations"][val_name]
                # also remove it from subsets
                subsets = [s for s in subsets if s.stem != val_name]
    # %%

    # Compute CPU allocation
    avail_cpus = multiprocessing.cpu_count() - 2
    assert avail_cpus > 0, "This script is multithreaded. Please run on a machine with many CPUs."

    # write the output to file and return the path
    tmp_out_dir = Path("/tmp/tokenizer_outputs")
    # delete it and its contents if it exists
    if tmp_out_dir.exists():
        shutil.rmtree(tmp_out_dir)
    tmp_out_dir.mkdir(parents=True, exist_ok=True)

    total_tasks = len([el for subset in subsets for el in subset.iterdir() if el.is_dir() and el.name in ["train", "val", "test"]])
    task_counter = 1

    for subset_dir in subsets:

        if subset_dir.name in config["extra_validations"]:
            mods = config["extra_validations"][subset_dir.name]['modalities']
        else:
            mods = subset_dir.stem.split("+")

        cols = {col for mod in set(mods) for col in modalities[mod]["column_mapping"].values()}

        task_cpus = int(avail_cpus / len(cols))
        splits = [split.name for split in subset_dir.iterdir() if split.is_dir() and split.name in ["train", "val", "test"]]
        if not splits:
            logger.warning(f"No train/val/test splits found in {subset_dir}. Skipping...")
            continue
        for split in splits:

            split_path = subset_dir / split

            log_prefix = f"[{task_counter}/{total_tasks}]  [{subset_dir.stem} | {split}]:"

            all_args = [(split_path, col, task_cpus, pos, serial, debug, tmp_out_dir, tokenizers) 
                        for pos, col in enumerate(cols_to_tokenize.intersection(cols))]

            tokenized_paths = {}
            if serial:
                # In debug mode, run sequentially
                logger.info(f"{log_prefix} Starting tokenization...")
                for args in all_args:
                    tokenized_paths.update(tokenize_modality(args))
            else:
                logger.info(f"{log_prefix} Tokenizing with {task_cpus} cores per task...")
                with ProcessPoolExecutor(max_workers=len(cols)) as executor:

                    futures = [executor.submit(tokenize_modality, args) for args in all_args]
                    for fut in as_completed(futures):
                        tokenized_paths.update(fut.result())

            logger.info(f"{log_prefix} Writing to file with DuckDB...")
            
            # Define save path and create directory
            save_path = tokenized_path / subset_dir.stem / split
            save_path.mkdir(parents=True, exist_ok=True)
            final_save_file = save_path / "dataset.parquet"

            # --- Build DuckDB Query (FIXED) ---
            
            # 1. Columns from the original file (pre-tokenized + __key__)
            subset_pre_tok_cols = list(pre_tok_cols.intersection(cols))
            base_select_cols = ['"__key__"'] + [f'"{mod}" AS "tok_{mod}"' for mod in subset_pre_tok_cols]
            
            # 2. Base CTE (Common Table Expression) - No more ROW_NUMBER()
            ctes = [f"""t_base AS (
                SELECT {', '.join(base_select_cols)}
                FROM read_parquet('{str(split_path)}')
            )"""]
            
            final_selects = ['t_base."__key__"'] + [f't_base."tok_{mod}"' for mod in subset_pre_tok_cols]
            joins = []

            # 3. Add one CTE and JOIN for each new tokenized file
            for i, (mod, path) in enumerate(tokenized_paths.items()):
                cte_alias = f"t{i}"
                col_alias = f"tok_{mod}"
                
                # This CTE now selects __key__ and the tokenized col
                # No more ROW_NUMBER()
                ctes.append(f"""{cte_alias} AS (
                    SELECT "__key__", "{mod}" AS "{col_alias}"
                    FROM read_parquet('{str(path)}')
                )""")
                
                final_selects.append(f'{cte_alias}."{col_alias}"')
                
                joins.append(f'JOIN {cte_alias} ON t_base."__key__" = {cte_alias}."__key__"')

            # 4. Assemble the final query
            concat_query = f"""
            COPY (
                WITH {', '.join(ctes)}
                SELECT {', '.join(final_selects)}
                FROM t_base
                {' '.join(joins)}
                ORDER BY t_base."__key__"
            ) 
            TO '{str(final_save_file)}' (FORMAT PARQUET);
            """
            # --- End Query Build ---

            # Execute query
            duckdb.sql(concat_query)
            logger.info(f"{log_prefix} Tokenized dataset saved to {save_path}")

            # Delete the temporary tokenized files
            for path in tokenized_paths.values():
                path.unlink(missing_ok=True) # Use missing_ok=True for robustness

            task_counter += 1

    # %%

    # write the updated config file to the tokenized path
    class FlowSeqDumper(yaml.SafeDumper):
        pass

    def _flow_seq(dumper, value):
        # emit every list in flow style
        return dumper.represent_sequence('tag:yaml.org,2002:seq',
                                        value,
                                        flow_style=True)
    FlowSeqDumper.add_representer(list, _flow_seq)
    with open("tokenize_config.yaml", "w") as file:
        # write a comment at the top of the file
        file.write("# This config file was created from merge_config.yaml by the 04_tokenize_dataset.py script.\n")
        file.write("# It contains the tokenizer versions for each modality and removes the unused modalities.\n")
        file.write("# This file is automatically generated and should not be manually edited.\n")
        yaml.dump(config, file, Dumper=FlowSeqDumper, sort_keys=False)
    copied_config_path = tokenized_path / "tokenize_config.yaml"
    shutil.copy("tokenize_config.yaml", copied_config_path)
    logger.info(f"Wrote config with tokenizer versions to {copied_config_path}")

    # %%
    elapsed_time = time.time() - start_time
    m, s = divmod(elapsed_time, 60)
    logger.info(f"Elapsed time: {int(m)} minutes and {int(s)} seconds")

    del tokenizers


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tokenize dataset modalities.")
    parser.add_argument("--serial", action="store_true", default=False, help="Run in serial mode (disables parallelization).")
    parser.add_argument("--debug", action="store_true", default=False, help="Run in debug mode (fewer rows).")
    args = parser.parse_args()

    main(serial=args.serial, debug=args.debug)