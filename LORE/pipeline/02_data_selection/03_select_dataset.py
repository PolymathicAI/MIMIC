#!/usr/bin/env python3
"""
Data Selection Pipeline - Step 3: Select Representative Dataset

This script selects one representative sequence per cluster from the master IDs table,
prioritizing high-quality sequences based on SwissProt annotation and species preference.
It creates the final dataset with train/val/test splits.

Input:
- config.yaml: Configuration file specifying the data version
- dataset.parquet: Master IDs table with splits from step 2

Output:
- train.parquet: Training set with selected representatives
- val.parquet: Validation set with selected representatives  
- test.parquet: Test set with selected representatives
- dataset.parquet: Combined dataset with all splits

The script:
1. Creates row clusters by coalescing protein_cluster_70, protein_cluster_30, and rna_cluster_30
2. Assigns priority scores based on:
   - SwissProt annotation (+4 points)
   - Human species (+2 points)
   - Mouse species (+1 point)
3. Uses ROW_NUMBER() with priority ranking and random tie-breaking to select one representative per cluster
4. Exports separate files for each split and a combined dataset

Selection Strategy:
- One sequence per cluster to reduce redundancy
- Prioritizes high-quality, well-annotated sequences
- Maintains split assignments from step 2
- Uses deterministic selection with random tie-breaking

Usage:
    python 03_select_dataset.py
"""

#%%
import duckdb
from pathlib import Path
import yaml
from lore import paths
from lore import logger
import time
import shutil

def main():

  start_time = time.time()

  # Load configuration
  config_file_path = Path(__file__).parent / "config.yaml"
  with open(config_file_path, "r") as f:
      config = yaml.safe_load(f)
  logger.info(f"Loaded configuration from {config_file_path.name}")
  version = config["master_id_version"]
  seed = config["seed"]

  # Where the original Parquet lives
  master_id_dir = paths.get_path(data_type="data", stage="intermediate", name="master_ids", version=version, fmt="parquet")
  master_id_path = master_id_dir / "dataset.parquet"
  copied_config_path = master_id_dir / "config.yaml"

  # compare the contents of the two config files and make sure they are the same
  with open(copied_config_path, "r") as f:
      copied_config = yaml.safe_load(f)
  if copied_config != config:
      raise ValueError(f"Config file {copied_config_path} does not match the current config. This implies a version conflict.")

  # Where we'll write the three subset files
  output_dir = Path(
      paths.get_path(
          data_type="data",
          stage="modality",
          name="id",
          version=version,
          fmt="parquet",
      )
  )
  if output_dir.exists():
      raise FileExistsError(f"Output path {output_dir} already exists. Please remove it before running this script.")
  output_dir.mkdir(parents=True, exist_ok=False)

  # Copy the config file to the output directory
  copied_config_path = output_dir / "config.yaml"
  shutil.copy(config_file_path, copied_config_path)
  logger.info(f"Copied config file to {copied_config_path}")

  # Open a DuckDB in-memory connection
  con = duckdb.connect(database=":memory:")

  # Set seed for reproducible random shuffling
  con.execute(f"SELECT setseed(0.{seed});")

  # Build and run one big CTE that:
  #   1) Coalesces the three cluster columns into row_cluster
  #   2) Computes a numeric "priority" based on has_funcprot_caption and species
  #   3) Uses ROW_NUMBER() over (PARTITION BY row_cluster ORDER BY priority DESC, RANDOM())
  #      to pick exactly one representative per cluster
  #
  # Then we keep only the four columns we ultimately need (plus split, which we need to filter on).
  #
  big_cte_sql = f"""
  CREATE TABLE selected AS
  WITH
    raw AS (
      SELECT
        CASE
          -- First we take p70 on the protein/rna overlap (i.e. when both protein_cluster_30 and rna_cluster_30 are both available)
          WHEN protein_cluster_30 IS NOT NULL AND rna_cluster_30 IS NOT NULL AND protein_cluster_70 IS NOT NULL THEN protein_cluster_70
          -- Otherwise, fall back to p30
          WHEN protein_cluster_30 IS NOT NULL THEN protein_cluster_30
          -- Finally, fall back to r30
          ELSE rna_cluster_30
        END AS row_cluster,
        has_funcprot_caption,
        species,
        has_masif,
        has_rasp2,
        has_prot_abund,
        has_phylop_human,
        has_phylop_mouse,
        --- has_rna_codons,
        uniprot_id,
        genome_feature_id,
        split
      FROM read_parquet('{master_id_path}')
      WHERE rna_seq_length IS NOT NULL OR aa_seq_length IS NOT NULL
    ),

    scored AS (
      SELECT
        row_cluster,
        uniprot_id,
        genome_feature_id,
        split,
        (
        CAST(has_rasp2 AS INTEGER) * 7
          + CAST(has_prot_abund AS INTEGER) * 6
          + CAST(has_funcprot_caption AS INTEGER) * 5
          + CAST(has_phylop_human AS INTEGER) * 4
          + CAST(has_phylop_mouse AS INTEGER) * 4
          + CAST(has_masif AS INTEGER) * 3
          --- + CAST(has_rna_codons AS INTEGER) * 2 (removed because no tokenizer)
        ) AS priority
      FROM raw
    ),

    ranked AS (
      SELECT
        row_cluster,
        uniprot_id,
        genome_feature_id,
        split,
        ROW_NUMBER()
          OVER (
            PARTITION BY row_cluster
            ORDER BY priority DESC, RANDOM()
          ) AS rn
      FROM scored
    )

  SELECT
    uniprot_id,
    genome_feature_id,
    split,
  FROM ranked
  WHERE rn = 1;
  """

  # Run the CTE (this builds an internal DuckDB view called "selected")
  logger.info("Executing sample selection query...")
  con.execute(big_cte_sql)
  logger.info("Created TABLE 'selected' with one representative per cluster.")
  #%%

  for split_name in ["train", "val", "test"]:
      out_path = output_dir / f"{split_name}.parquet"
      logger.info(f"Writing {split_name} split to {out_path}")
      con.execute(f"""
        COPY (
          SELECT
            uniprot_id,
            genome_feature_id
          FROM selected
          WHERE split = '{split_name}'
        ) TO '{out_path}' (FORMAT PARQUET);
      """)

      count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
      logger.info(f"Saved {split_name} split with {count:,} rows.")

  #%% combine the three splits into one   

  combined_path = output_dir / "dataset.parquet"
  logger.info(f"Combining splits into {combined_path}")
  con.execute(f"""
      COPY (
          SELECT *, 'train' AS split FROM read_parquet('{output_dir}/train.parquet')
          UNION ALL
          SELECT *, 'val' AS split FROM read_parquet('{output_dir}/val.parquet')
          UNION ALL
          SELECT *, 'test' AS split FROM read_parquet('{output_dir}/test.parquet')
      ) TO '{output_dir}/dataset.parquet' (FORMAT PARQUET);
    """)
  count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{combined_path}')").fetchone()[0]
  logger.info(f"Combined dataset has {count:,} rows.")

  # Close the DuckDB connection
  con.close()

  # Log the total time taken
  elapsed_time = time.time() - start_time
  m, s = divmod(elapsed_time, 60)
  logger.info(f"Total time taken: {int(m)} minutes, {s:.2f} seconds.")

if __name__ == "__main__":
    main()