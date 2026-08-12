#!/usr/bin/env python3
# %%

from lore import paths
from lore import logger
import duckdb

aa_seq_path = paths.get_path(data_type="data", stage="modality", name="aa_seq", version="v4_plddt_70", fmt="parquet") / "dataset.parquet"
assert aa_seq_path.exists()
out_path = paths.get_path(data_type="data", stage="modality", name="aa_seq", version="v4_plddt_70", fmt="parquet") / "seq_lens.parquet"
assert not out_path.exists(), f"{out_path} already exists. Delete before running this script."
# %%

conn = duckdb.connect(database=":memory:")
logger.info("Starting to calculate and save sequence lengths...")
conn.execute(f"COPY (SELECT uniprot_id, LENGTH(sequence) AS seq_len FROM read_parquet('{aa_seq_path}')) TO '{out_path}' (FORMAT 'parquet')")
logger.info(f"Saved seq_lens to {out_path}.")
conn.close()

# %%
# load the first 5 rows and print it out
conn = duckdb.connect(database=":memory:")
df = conn.execute(f"SELECT * FROM read_parquet('{out_path}') LIMIT 5").fetchdf()
print(df.head())
# print the number of total rows
total_rows = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
print(f"Total rows: {total_rows}")
conn.close()