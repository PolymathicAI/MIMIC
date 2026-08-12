#!/usr/bin/env python3
# %%
"""
Builds the merged datasets from the subsets created in 01_create_subsets.py.

"""

import time
from pathlib import Path
import shutil
import duckdb
import yaml
import copy
from lore import logger
from lore import paths
import argparse
import sys


def main(debug=False, test_frac=0.15, in_memory=False):

    if debug:
        logger.warning(f"Running in debug mode with {test_frac:.0%} of the data!")

    SUBFOLDER = Path("./assets")
    copied_config_path = SUBFOLDER / "config.yaml"

    # ── LOAD YAML CONFIG ───────────────────────────────────────────────────────
    with open(Path(__file__).parent / "config.yaml", "r") as file:
        config = yaml.safe_load(file)
    logger.info(f"Loaded config from {file.name}")

    # assert the config file matches the copied config file
    with open(copied_config_path, "r") as file:
        copied_config = yaml.safe_load(file)
    if copied_config != config:
        raise ValueError(f"Config file {copied_config_path} does not match the current config. This implies a version conflict.")

    # Define working_cfg to avoid modifying the original config
    # The updated original config will be saved so that the column mappings can be stored.
    working_cfg = copy.deepcopy(config)

    # ── FIND THE DIFFERENT SUBSETS ────────────────────────────────────────────────
    subset_path = SUBFOLDER / "subsets"
    subset_ds_paths = list(subset_path.glob("*.parquet"))
    logger.info(f"Found {len(subset_ds_paths)} subsets in {subset_path}")

    # sort the subsets according to the number of modalities (ascending)
    subset_ds_paths.sort(key=lambda x: len(x.stem.split("+")))

    # ── FIND THE TRAIN/TEST/VAL SPLITS ────────────────────────────────────────────────
    id_root = paths.get_path(data_type="data", stage="modality", name="id", version=str(working_cfg["master_id_version"]), fmt="parquet")
    for val_name, val_data in working_cfg["extra_validations"].items():
        val_id_path = id_root / f"{val_data['id_filename']}"
        assert val_id_path.exists(), f"Validation ID path {val_id_path} does not exist."
        working_cfg["extra_validations"][val_name]['path'] = val_id_path

    # %%

    # ── SET THE OUTPUT PATHS ────────────────────────────────────────────────────────
    output_path = paths.get_path(data_type="data", stage="merged", version=str(working_cfg["version"]), fmt=working_cfg["fmt"], name=working_cfg["name"])
    if output_path.exists():
        logger.info(f"The folder {output_path} already exists. Exitting.")
        exit(0)
    output_path.mkdir(parents=True, exist_ok=False)


    # %%

    t0_total = time.perf_counter()

    # ── SETTING UP DUCKDB ────────────────────────────────────────────────────────────

    if in_memory:
        logger.info("Using in-memory DuckDB database.")
        con = duckdb.connect(database=":memory:", read_only=False)
    else:
        logger.info("Using on-disk DuckDB database at /tmp/duckdb_temp.db.")
        con = duckdb.connect(database="/tmp/duckdb_temp.db", read_only=False)

    # set preserve insertion order to false for performance
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET threads = 128")

    # %%

    # combine modalities and mutually exclusive modalities
    working_cfg["modalities"].update(working_cfg.get("mutually_exclusive_modalities", {}))

    # ── CREATING MODALITY VIEWS ───────────────────────────────────────────────────────────
    for mod, mod_dict in working_cfg["modalities"].items():
        mod_path = paths.get_path(
            data_type="data",
            stage="modality",
            name=mod,
            version=str(mod_dict["version"]),
            fmt="parquet",
        ) / "dataset.parquet"

        # get the merge_key from the config
        merge_key = mod_dict["merge_key"]

        # get the name of the columns of the dataset using duckdb
        cols = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{mod_path}')"
        ).fetchall()
        cols = [col[0] for col in cols]

        ignore_cols = mod_dict.get("ignore_cols", [])
        prepend_text = mod_dict.get("prepend_text", "")

        for key in ignore_cols + [merge_key]:
            assert key in cols, f"{key} not found in {mod} columns: {cols}"
            cols.remove(key)

        col_name_map = {col: f"{prepend_text}{col}" for col in cols} if len(cols) > 1 else {cols[0]: mod}

        # create a view with the __key__ col_name but renaming col_name to mod
        # get the size of the dataset
        size = con.execute(f"SELECT COUNT(*) FROM read_parquet('{mod_path}')").fetchone()[0]
        con.execute(f"""CREATE OR REPLACE VIEW {mod} AS
                    SELECT {merge_key}, {', '.join([f'{col} AS {col_name_map[col]}' for col in cols])}
                    FROM read_parquet('{mod_path}')
                    {f"LIMIT {int(size*test_frac)}" if debug else ""};
                    """)
        logger.info(f"Created VIEW {mod} with columns {col_name_map} {f'with {test_frac:.0%} of rows.' if debug else ''}")
        
        if mod in config["modalities"]:
            config["modalities"][mod]["column_mapping"] = col_name_map
        else:
            config["mutually_exclusive_modalities"][mod]["column_mapping"] = col_name_map
    print()


    # %%
    # ── ITERATING OVER SUBSETS ────────────────────────────────────────────────────────

    total_tasks = len(subset_ds_paths) + len(config["extra_validations"])
    task_counter = 1

    logger.info(f"Starting merge on {len(subset_ds_paths)} subsets.")
    logger.info(f"This will take about 40 minutes (3 mins in debug mode).")
    print()

    def log_task_count(path, con):
        """
        Logs the number of rows in the given path.
        """
        count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]
        return count

    for subset in subset_ds_paths:

        subset_mods = subset.stem.split("+")
        subset_keys = [working_cfg["modalities"][m]["merge_key"] for m in subset_mods]

        t0_task = time.perf_counter()

        log_prefix = f"[{task_counter}/{total_tasks}] [{subset.stem}]:"

        # do an inner join of the all_ids table and the subset
        con.execute(
            f"""
            CREATE OR REPLACE VIEW current_ids AS
            SELECT * FROM read_parquet('{subset}') 
            """
        )

        # Do an inner join of all the subset_mods and create a view from this
        logger.info(f"{log_prefix} Performing merge for all splits...")
        join_sql = "".join(f" INNER JOIN {m} USING ({k})" for m, k in zip(subset_mods, subset_keys))
        subset_merge_sql = f"""
            CREATE OR REPLACE TABLE merged_data AS
            SELECT * 
            FROM current_ids {join_sql}
            ;
        """
        con.execute(subset_merge_sql)
        logger.info(f"{log_prefix} Created TABLE merged_data with {con.execute('SELECT COUNT(*) FROM merged_data').fetchone()[0]:,} rows.")

        for split in ["train", "val", "test"]:

            # get the output path
            split_path = output_path / subset.stem / split
            split_path.mkdir(parents=True, exist_ok=True)

            subset_merge_sql = f"""
                COPY (
                WITH filtered_data AS (
                    SELECT * EXCLUDE (split)
                    FROM merged_data
                    WHERE split = '{split}'
                )
                SELECT
                    (ROW_NUMBER() OVER (ORDER BY (SELECT NULL))) - 1 AS __key__,
                    *
                FROM filtered_data
                )
                TO '{split_path}'
                (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE);
            """
            logger.info(f"{log_prefix} Writing {split} split to {split_path}...")
            con.execute(subset_merge_sql)
            count = log_task_count(split_path / "*.parquet", con)
            logger.info(f"{log_prefix} Saved with {count:,} rows.")

        # Remove the current_ids and merged_data tables to free up memory
        con.execute("DROP TABLE IF EXISTS merged_data;")
        con.execute("DROP VIEW IF EXISTS current_ids;")

        elapsed = time.perf_counter() - t0_task
        mins, secs = divmod(elapsed, 60)
        logger.info(f"{log_prefix} Done in {mins}m {secs:.2f}s")
        task_counter += 1

        print()
        print()

    # %%
    # ── EXTRA VALIDATION DATASETS ─────────────────────────────────────────────
    for val_name, val_data in working_cfg["extra_validations"].items():

        t0_task = time.perf_counter()
        log_prefix = f"[{task_counter}/{total_tasks}] [extra_val | {val_name}]:"

        output_path_val = output_path / val_name / 'val'
        output_path_val.mkdir(parents=True, exist_ok=True)

        # get the columns of the validation dataset
        val_cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{val_data['path']}')").fetchall()
        val_cols = [col[0] for col in val_cols]
        
        write_sql = f"""
        COPY (
            SELECT 
            (ROW_NUMBER() OVER (ORDER BY (SELECT NULL))) - 1 AS __key__,
            * FROM read_parquet('{val_data['path']}')
            """
        for mod in val_data["modalities"]:
            merge_key = working_cfg["modalities"][mod]["merge_key"]
            write_sql += f"""
            INNER JOIN {mod} USING ({merge_key})
            """
        write_sql += f"""
        ) TO '{output_path_val}'
        (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE);
        """
        logger.info(f"{log_prefix} Performing merge and writing to {output_path_val}...")
        con.execute(write_sql)
        count = log_task_count(output_path_val / "*.parquet", con)
        logger.info(f"{log_prefix} Saved with {count:,} rows.")
        elapsed = time.perf_counter() - t0_task
        mins, secs = divmod(elapsed, 60)
        logger.info(f"{log_prefix} Done in {mins}m {secs:.2f}s")
        task_counter += 1
        print()
        
    # %%
    # ── FINISH ────────────────────────────────────────────────────────────────
    con.close()

    # %%

    class FlowSeqDumper(yaml.SafeDumper):
        pass

    def _flow_seq(dumper, value):
        # emit every list in flow style
        return dumper.represent_sequence('tag:yaml.org,2002:seq',
                                        value,
                                        flow_style=True)
    FlowSeqDumper.add_representer(list, _flow_seq)
    with open("merge_config.yaml", "w") as file:
        # write a comment at the top of the file
        file.write("# This config file was created from config.yaml by the 02_merge_script.py script.\n")
        file.write("# It contains the column mappings for each modality after merging.\n")
        file.write("# This file is automatically generated and should not be manually edited.\n")
        yaml.dump(config, file, Dumper=FlowSeqDumper, sort_keys=False)

    # copy the config file to the output path
    output_config_path = output_path / "merge_config.yaml"
    shutil.copy("merge_config.yaml", output_config_path)
    logger.info(f"Copied config file to {output_config_path}")

    elapsed = time.perf_counter() - t0_total
    mins = int(elapsed // 60)
    secs = elapsed % 60
    logger.info(f"Finished in {mins}m {secs:.2f}s")
    # %%


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False, help="Run in debug mode on a fraction of the data.")
    parser.add_argument("--test_frac", type=float, default=0.15, help="Fraction of data to use in test mode.")
    # add a boolean argument for in_memory
    parser.add_argument("--in_memory", action="store_true", default=False, help="Use in-memory DuckDB database.")
    args = parser.parse_args()
    debug = args.debug
    test_frac = args.test_frac
    in_memory = args.in_memory

    main(debug=debug, test_frac=test_frac, in_memory=in_memory)