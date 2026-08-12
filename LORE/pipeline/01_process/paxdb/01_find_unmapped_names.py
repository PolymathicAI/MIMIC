# %% Imports
import pandas as pd
import duckdb
from lore import logger
from lore.paths import get_path
import re

# UniProt ID regex pattern
uniprot_pattern = r'^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$'

# More readable version with explanation:
uniprot_pattern = re.compile(r'''
    ^(
        [OPQ][0-9][A-Z0-9]{3}[0-9]          # Format 1: O12345, P12345, Q12345 (6 chars)
        |
        [A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}   # Format 2: A0A123B456 (10 chars) or longer
    )$
''', re.VERBOSE)

data_root = get_path("data", "downloads", "paxdb", "v5")
master_ids = get_path("data", "intermediate", "master_ids", "1.2", fmt="parquet") / "dataset.parquet"

#%% Load only the uniprot_id column using DuckDB
logger.info("Loading UniProt IDs from DuckDB...")
uniprot_name = duckdb.execute(f"""
    SELECT uniprot_id
    FROM '{master_ids}'
""").df()

# %% Get mapping data
uniprot_map = pd.read_csv(data_root / "paxdb-uniprot-links-v5.0" / "paxdb-uniprot-links-v5.0.tsv",
                         sep="\t",
                         header=None,
                         names=["paxdb_id", "uniprot_id"]
                         )
uniprot_map = uniprot_map.drop_duplicates(subset=["paxdb_id", "uniprot_id"])
split_upid = uniprot_map["uniprot_id"].str.split("_", expand=True)
uniprot_map["uniprot_id"] = split_upid[0]
uniprot_map["species"] = split_upid[1]
uniprot_map["species_number"] = uniprot_map["paxdb_id"].str.split(".", expand=True)[0]

#%% Validate uniprot IDs
logger.info("Validating UniProt IDs...")

valid_upids = uniprot_map[uniprot_map["uniprot_id"].str.match(uniprot_pattern)]
invalid_upids = uniprot_map[~uniprot_map["uniprot_id"].str.match(uniprot_pattern)]

with open(data_root / "uniprot-names-to-map.csv", "w+") as f:
    for _, row in invalid_upids.iterrows():
        f.write(f"{row['uniprot_id']}_{row['species']},{row['species_number']}\n")
