#!/usr/bin/env python3
"""
Data Selection Pipeline - Step 2: Create Train/Validation/Test Splits (DuckDB Version)

This script creates train/validation/test splits by clustering similar sequences and ensuring
that similar sequences are kept within the same split to prevent data leakage. The splitting
is done at the cluster level, not at the individual sequence level.

This version uses DuckDB for efficient, out-of-core processing.

Input:
- config.yaml: Configuration file specifying the data version
- dataset_no_split.parquet: Master IDs table from step 1 (without splits)

Output:
- dataset.parquet: Master IDs table with train/val/test split assignments

The script:
1. Loads configuration and validates paths.
2. Uses DuckDB to read the input parquet file.
3. Creates a 'row_cluster' column by coalescing protein_cluster_30 and rna_cluster_30.
4. Filters out rows where 'row_cluster' is NULL, logging the count.
5. Finds all unique clusters.
6. Sets a reproducible random seed (SET seed=42).
7. Shuffles the unique clusters and assigns them to splits (80% train, 10% val, 10% test)
   using NTILE(10) over a randomly ordered set.
8. Joins the split assignments back to the main data.
9. Gathers split statistics (row and cluster counts).
10. Writes the final table (with the new 'split' column and without 'row_cluster')
    to the output parquet file.

Usage:
    python 02_make_test_train_val_splits_duckdb.py
"""
#%%

import duckdb
import yaml
from pathlib import Path

def main():

    from lore import paths
    from lore import logger

    # Load configuration
    config_file_path = Path(__file__).parent / "config.yaml"
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    version = config["master_id_version"]
    seed = config["seed"]

    # %%
    master_id_dir = paths.get_path(data_type="data", stage="intermediate", name="master_ids", version=version, fmt="parquet")
    master_id_path = master_id_dir / "dataset_no_split.parquet"
    output_path = master_id_dir / "dataset.parquet"

    assert not output_path.exists(), f"Output path {output_path} already exists. Please remove it before running this script."

    copied_config_path = master_id_dir / "config.yaml"

    # Compare the contents of the two config files and make sure they are the same
    with open(copied_config_path, "r") as f:
        copied_config = yaml.safe_load(f)
    if copied_config != config:
        raise ValueError(f"Config file {copied_config_path} does not match the current config. This implies a version conflict.")

    logger.info(f"Loading the master_ids from {master_id_path} using DuckDB...")
    con = duckdb.connect(database=':memory:')

    # Set seed for reproducible random shuffling
    con.execute(f"SELECT setseed(0.{seed});")

    # %%
    # Check for rows with no cluster assignment
    logger.info("Checking for rows without cluster assignments...")

    # Create a temporary view to check for NULLs
    con.execute(f"""
        CREATE TEMP VIEW all_rows AS
        SELECT COALESCE(protein_cluster_30, rna_cluster_30) AS row_cluster
        FROM read_parquet('{master_id_path}')
    """)

    no_cluster = con.execute("SELECT COUNT(*) FROM all_rows WHERE row_cluster IS NULL").fetchone()[0]

    if no_cluster > 0:
        logger.warning(f"Dropping {no_cluster:,} rows without cluster assignments.")

    # Create the main working view, filtering out NULLs
    con.execute(f"""
        CREATE TEMP VIEW master_data AS
        SELECT *, COALESCE(protein_cluster_30, rna_cluster_30) AS row_cluster
        FROM read_parquet('{master_id_path}')
        WHERE COALESCE(protein_cluster_30, rna_cluster_30) IS NOT NULL
    """)

    # %%
    # Get unique clusters
    unique_clusters_count = con.execute("SELECT COUNT(DISTINCT row_cluster) FROM master_data").fetchone()[0]
    logger.info(f"Found {unique_clusters_count:,} unique clusters")

    # Shuffle clusters and assign splits
    logger.info("Shuffling unique clusters and separating into train/val/test splits")

    # Create a table mapping each cluster to a split
    con.execute("""
        CREATE TABLE cluster_splits AS
        WITH UniqueClusters AS (
            SELECT DISTINCT row_cluster FROM master_data
        )
        SELECT
            row_cluster,
            CASE
                WHEN ntile <= 8 THEN 'train'  -- 80% (tiles 1-8)
                WHEN ntile = 9 THEN 'val'    -- 10% (tile 9)
                ELSE 'test'                   -- 10% (tile 10)
            END AS split
        FROM (
            SELECT
                row_cluster,
                NTILE(10) OVER (ORDER BY random()) AS ntile
            FROM UniqueClusters
        ) AS T
    """)

    logger.info("Creating a mapping from clusters to splits for the master_ids...")

    # Create the final view by joining data with splits
    con.execute("""
        CREATE TEMP VIEW final_data AS
        SELECT
            master.* EXCLUDE(row_cluster), -- Drop the temporary cluster column
            splits.split
        FROM master_data AS master
        JOIN cluster_splits AS splits
        ON master.row_cluster = splits.row_cluster
    """)

    # Log split statistics
    logger.info("Calculating split statistics...")

    # Get row counts
    stats = con.execute("SELECT split, COUNT(*) as row_count FROM final_data GROUP BY split").df()
    stats_dict = stats.set_index('split')['row_count'].to_dict()

    # Get cluster counts
    cluster_stats = con.execute("SELECT split, COUNT(DISTINCT row_cluster) as cluster_count FROM cluster_splits GROUP BY split").df()
    cluster_stats_dict = cluster_stats.set_index('split')['cluster_count'].to_dict()

    # Log stats in the same format as the original script
    logger.info(f"Train: {stats_dict.get('train', 0):,} rows ({cluster_stats_dict.get('train', 0):,} clusters)")
    logger.info(f"Val: {stats_dict.get('val', 0):,} rows ({cluster_stats_dict.get('val', 0):,} clusters)")
    logger.info(f"Test: {stats_dict.get('test', 0):,} rows ({cluster_stats_dict.get('test', 0):,} clusters)")
    logger.info(f"Unknown: {stats_dict.get('unknown', 0):,} rows")

    #%%
    # Save the updated master_ids with splits
    logger.info(f"Writing master_ids with splits information back to {output_path}")

    con.execute(f"COPY final_data TO '{output_path}' (FORMAT 'PARQUET', CODEC 'ZSTD');")

    con.close()
    logger.info("Done!")

if __name__ == "__main__":
    main()