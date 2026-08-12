#!/usr/bin/env python3
"""
Data Selection Pipeline - Step 1: Build Master IDs Table

This script creates a comprehensive master table by joining ID mappings with all configured 
modalities and precomputed modalities. It uses DuckDB to perform efficient joins and 
transformations on large datasets. The transformations are picked such that the final Master Table 
contains only essential information for downstream tasks.

Input:
- config.yaml: Configuration file specifying modalities, versions, and transformations
- IDs table: Base table containing uniprot_id and genome_feature_id mappings
- All configured modalities in Parquet format (from step 0)
- All configured precomputed modalities in specified formats

Output:
- dataset_no_split.parquet: Master table with all joined modalities (without train/val/test splits)

The script:
1. Loads the ID match table as the base for all joins
2. Processes each modality with specified transformations (len, exists, or raw values)
3. Performs FULL OUTER JOINs to combine all modalities using specified merge keys
4. Applies transformations like length calculations or existence flags
5. Exports the final joined table to Parquet format

Transformations supported:
- 'len': Calculate length of column values
- 'exists': Create boolean flag for non-null values
- None: Keep raw column values

Usage:
    python 01_build_master_ids.py
"""
#%%
from pathlib import Path
import time
import shutil

from lore import logger # 
from lore import paths 

import yaml
import duckdb

def main():

    logger.warning("This script overcounts the data when the matching between uniprot_id and genome_feature_id is not one-to-one.")

    #%% Load configuration
    config_file_path = Path(__file__).parent / "config.yaml"
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_file_path.name}")

    output_dir = paths.get_path(
        data_type="data", 
        stage="intermediate", 
        name="master_ids", 
        version=config["master_id_version"], 
        fmt="parquet"
    ) 
    output_path = output_dir / "dataset_no_split.parquet"
    config_copy_path = output_dir / "config.yaml"

    if output_path.parent.exists():
        raise FileExistsError(f"Output path {output_path.parent} already exists. Please remove it before running this script.")

    # Initialize DuckDB connection
    con = duckdb.connect(database=':memory:')
    # set preserve insertion order to false for performance
    con.execute("SET preserve_insertion_order = false")
    logger.info("Connected to DuckDB (in-memory)")

    # Record start time for total execution
    t0_total = time.perf_counter()

    #%% Load IDs table as the base
    IDs_config = config["id_match"]
    IDs_path = paths.get_path(
        data_type="data",
        stage=IDs_config['stage'],
        name=IDs_config['name'],
        version=str(IDs_config['version'])
    ) / IDs_config['path']

    logger.info(f"Loading ID match data from {IDs_path}")
    con.execute(f"CREATE VIEW IDs AS SELECT uniprot_id, genome_feature_id FROM read_parquet('{IDs_path}')")

    #%%  Helper function for transform logic
    def get_select_clause(transform, col_name, mod_name, merge_key, final_name):
        if transform == 'len':
            return f'{merge_key}, CAST(LENGTH("{col_name}") AS INTEGER) as "{final_name}_length"'
        elif transform == 'exists':
            return f'{merge_key}, TRUE as "has_{final_name}"'
        elif transform is None or transform == 'None':
            return f'{merge_key}, "{col_name}" as "{final_name}"'
        else:
            raise ValueError(f"Unknown transform '{transform}' for column '{col_name}' in modality '{mod_name}'")

    # %% Process mutually exclusive modalities first

    individual_merge_queries = []
    merge_keys = []

    # For each modality, generate a FULL OUTER JOIN query.
    for mod_name, mod_config in config["mutually_exclusive_modalities"].items():
        mod_path = paths.get_path(
            data_type="data",
            stage="modality",
            name=mod_name,
            version=str(mod_config['version']),
            fmt="parquet"
        ) / 'dataset.parquet'
        
        transform = mod_config.get('transform', None)
        col_name = mod_config.get('col_name', mod_name)
        merge_key = mod_config['merge_key']
        final_name = mod_name

        assert transform == 'exists', "Mutually exclusive modalities must use 'exists' transform"

        select_clause = get_select_clause(transform, col_name, mod_name, merge_key, final_name)
        modality_subquery = f"SELECT {select_clause} FROM read_parquet('{mod_path}')"

        # Use a Right join with implicit column selection.
        merge_query = f"""
        (SELECT *
        FROM IDs
        RIGHT JOIN ({modality_subquery}) AS T2 USING ({merge_key}))
        """
        individual_merge_queries.append(merge_query)

        merge_keys.append(merge_key)

    # Combine all join results. Duplicates for unmatched IDs rows will be present at this stage.
    full_query = "\nUNION ALL BY NAME\n".join(individual_merge_queries)

    # Create the final table, using SELECT DISTINCT to remove the duplicate rows.
    mut_exc_query = f"""
    CREATE OR REPLACE TABLE MUT_EXC_MODS AS
    SELECT DISTINCT *
    FROM ({full_query})
    """
    con.execute(mut_exc_query)
    logger.info("Created MUT_EXC_MODS table with mutually exclusive modalities")

    # Get the mut_exc_mods columns other than the has_ columns
    mut_exc_columns = con.execute("DESCRIBE MUT_EXC_MODS").fetchdf()['column_name'].tolist()

    # Add the rows in IDs where the uniprot_id or genome_feature_id is not present the mut_exc_mods table
    # Use DISTINCT here to only keep the IDs. (No duplicates for multiple cell types etc)
    con.execute(f"""
    INSERT INTO MUT_EXC_MODS
    SELECT DISTINCT t1.*, {', '.join(['FALSE AS ' + col for col in mut_exc_columns if col.startswith("has_")])}
    FROM IDs AS t1
    LEFT JOIN MUT_EXC_MODS AS t2 ON {" AND ".join([f"t1.{key} IS NOT DISTINCT FROM t2.{key}" for key in set(merge_keys)])}
    WHERE {' AND '.join([f"t2.{col} IS NULL" for col in mut_exc_columns if col.startswith("has_")])}
    """)
    logger.info("Added missing ID match rows to MUT_EXC_MODS table")

    # drop the unused tables to free up memory
    con.execute("DROP VIEW IF EXISTS IDs")
    # %%

    final_query = f"""
        CREATE OR REPLACE TABLE final_view AS
            SELECT * FROM MUT_EXC_MODS
    """

    for mod_name, mod_config in config["modalities"].items():
        # Create VIEW for this modality
        mod_path = paths.get_path(
            data_type="data",
            stage="modality",
            name=mod_name,
            version=str(mod_config['version']),
            fmt="parquet"
        ) / 'dataset.parquet'
        
        transform = mod_config.get('transform', None)
        col_name = mod_config.get('col_name', mod_name)
        merge_key = mod_config['merge_key']
        final_name = mod_name

        select_clause = get_select_clause(transform, col_name, mod_name, merge_key, final_name)
        con.execute(f"CREATE VIEW '{final_name}' AS SELECT {select_clause} FROM read_parquet('{mod_path}')")
        columns = con.execute(f"DESCRIBE '{final_name}'").fetchdf()['column_name'].tolist()
        logger.info(f"Created view for {final_name} with columns: {', '.join(columns)}")
        
        # Add JOIN to final query
        merge_key = mod_config['merge_key']
        final_query += f"""
                FULL OUTER JOIN "{final_name}" USING ({merge_key})
                """

    #%% Process all precomputed modalities and add them to the query
    for mod_name, mod_config in config["precomputed_modalities"].items():
        # Create VIEW for this precomputed modality
        mod_path = paths.get_path(
            data_type="data",
            stage=mod_config['stage'],
            name=mod_config.get('name', mod_name),
            version=str(mod_config['version']),
            fmt=mod_config.get('fmt')
        ) / mod_config.get('path', f"{mod_name}.parquet")

        transform = mod_config.get('transform', None)
        col_name = mod_config.get('col_name', mod_name)
        merge_key = mod_config['merge_key']

        if 'keys' not in mod_config:
            # in this case, the mod name is the final_name
            mod_config['keys'] = [{'key': col_name, 'transform': transform, 'final_name': mod_name}]

        for key_config in mod_config['keys']:

            col_name = key_config['key']
            transform = key_config.get('transform', None)
            # in case this is a key in a precomputed modality, we can use the col_name as the final_name
            final_name = key_config.get('final_name', col_name)

            select_clause = get_select_clause(transform, col_name, mod_name, merge_key, final_name)
            con.execute(f"CREATE VIEW '{final_name}' AS SELECT {select_clause} FROM read_parquet('{mod_path}')")
            columns = con.execute(f"DESCRIBE '{final_name}'").fetchdf()['column_name'].tolist()
            logger.info(f"Created view for {final_name} with columns: {', '.join(columns)}")
            
            # Add FULL OUTER JOIN to final query
            merge_key = mod_config['merge_key']
            final_query += f"""
                    FULL OUTER JOIN "{final_name}" USING ({merge_key})
                    """

    #%% Complete the final query with output path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # copy the config file to the output directory
    shutil.copy(config_file_path, config_copy_path)
    logger.info(f"Copied config file to {config_copy_path}")


    # Execute the final query
    logger.info(f"Executing final query and saving the table.")
    logger.debug(f"Final query: {final_query}")
    con.execute(final_query)

    logger.info("Filling in the NULLs for has_ columns with FALSE and exporting to Parquet.")
    final_view_cols = con.execute("DESCRIBE final_view").fetchdf()['column_name'].tolist()
    save_query = f"""
                COPY (
                SELECT {', '.join([f'"{col}"' for col in final_view_cols if not col.startswith("has_")])},
                {', '.join([f"COALESCE({col}, FALSE) AS {col}" for col in final_view_cols if col.startswith("has_")])}
                FROM final_view
                )
                TO '{output_path}'
                (FORMAT PARQUET);
            """
    con.execute(save_query)
    logger.info(f"Exported master IDs to {output_path}")

    # Close connection and report total time
    con.close()
    logger.info("DuckDB connection closed")

    t_total = time.perf_counter() - t0_total
    m, s = divmod(t_total, 60)
    logger.info(f"Total execution time: {int(m)} minutes, {s:.2f} seconds")

    # %% load the final master IDs to verify
    con = duckdb.connect(database=':memory:')

    # get the number of rows in the master_ids table
    count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output_path}')").fetchone()[0]
    logger.info(f"Master IDs table has {count:,} rows")

    # get the first 5 rows and convert to dataframe
    df = con.execute(f"SELECT * FROM read_parquet('{output_path}') LIMIT 5").fetchdf().head()
    logger.info(f"First 5 rows of master IDs:\n{df}")

    # print the columns
    logger.info(f"Columns in master IDs: {df.columns.tolist()}")
    con.close()

    # %%

if __name__ == "__main__":
    main()
