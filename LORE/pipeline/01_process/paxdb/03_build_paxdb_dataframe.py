# %% Imports
import re

import duckdb
import pandas as pd

from lore import logger
from lore.paths import get_path

uniprot_pattern = re.compile(
    r"""
    ^(
        [OPQ][0-9][A-Z0-9]{3}[0-9]          # Format 1: O12345, P12345, Q12345 (6 chars)
        |
        [A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}   # Format 2: A0A123B456 (10 chars) or longer
    )$
""",
    re.VERBOSE,
)

PAXDB_VERSION = "v5"

data_root = get_path("data", "downloads", "paxdb", PAXDB_VERSION)

# %% Load only the uniprot_id column using DuckDB
logger.info("Loading UniProt IDs from DuckDB...")

master_id_path = (
    get_path("data", "intermediate", "master_ids", version="1.2", fmt="parquet")
    / "dataset.parquet"
)

uniprot_name = duckdb.execute(f"""
    SELECT uniprot_id
    FROM '{master_id_path}'
""").df()

# %% Get mapping data
uniprot_map = pd.read_csv(
    data_root / "paxdb-uniprot-links-v5.0" / "paxdb-uniprot-links-v5.0.tsv",
    sep="\t",
    header=None,
    names=["paxdb_id", "uniprot_id"],
)
uniprot_map = uniprot_map.drop_duplicates(subset=["paxdb_id", "uniprot_id"])
split_upid = uniprot_map["uniprot_id"].str.split("_", expand=True)
uniprot_map["uniprot_id"] = split_upid[0]
uniprot_map["species"] = split_upid[1]
uniprot_map["species_number"] = uniprot_map["paxdb_id"].str.split(".", expand=True)[0]

# %% Validate uniprot IDs
logger.info("Validating UniProt IDs...")

# %% Load mapped names
names_mapped = pd.read_csv(data_root / "up-names-out.txt", sep=",")
names_mapped = names_mapped[["gene_name", "taxonomy_id", "uniprot_id"]]
names_mapped = names_mapped.drop_duplicates(
    subset=["gene_name", "taxonomy_id"], keep="first"
)
names_mapped["taxonomy_id"] = names_mapped["taxonomy_id"].astype(str)
names_mapped["gene_alone"] = names_mapped["gene_name"].str.split("_", expand=True)[0]

# %% Replace incorrect uniprot_id values with correct ones from names_mapped
# Merge to find where uniprot_id matches gene_name and species_number matches taxonomy_id
merged_corrections = pd.merge(
    uniprot_map,
    names_mapped,
    left_on=["uniprot_id", "species_number"],
    right_on=["gene_alone", "taxonomy_id"],
    how="left",
    suffixes=("_original", "_correct"),
)

# Replace uniprot_id with the correct one where a match was found
# If uniprot_id_correct is not null, use it; otherwise keep the original
uniprot_map["uniprot_id"] = merged_corrections["uniprot_id_correct"].combine_first(
    merged_corrections["uniprot_id_original"]
)

# Log the number of corrections made
corrections_made = merged_corrections["uniprot_id_correct"].notna().sum()
logger.info(
    f"Made {corrections_made} corrections to uniprot_id values using names_mapped."
)

# Clean up the temporary columns - keep only the original uniprot_map structure
uniprot_map = uniprot_map[["paxdb_id", "uniprot_id", "species", "species_number"]]
uniprot_map = uniprot_map.drop_duplicates()

# %% Get abundance data

abundance = {}

species = list((data_root / "paxdb-abundance-files-v5.0").glob("*"))

for spec_file in species:
    species_name = spec_file.stem
    logger.info(f"Processing species {species_name}...")
    abundance_files = spec_file.glob("*.txt")
    for abundance_file in abundance_files:
        condition = "_".join(abundance_file.stem.split("-")[1:])
        logger.info(f"Processing file {abundance_file.name}...")
        df = pd.read_csv(abundance_file, sep="\t", comment="#")
        if len(df.columns) == 2:
            df.columns = ["paxdb_id", "abundance"]
        elif len(df.columns) == 3:
            df.columns = ["paxdb_id", "abundance", "raw_spectral_counts"]
        df["species_number"] = species_name
        df["condition"] = condition
        df = df[["species_number", "condition", "paxdb_id", "abundance"]]
        df = df.drop_duplicates(subset=["paxdb_id", "species_number"])
        if species_name not in abundance:
            abundance[species_name] = df
        else:
            abundance[species_name] = pd.concat(
                [abundance[species_name], df], ignore_index=True
            )

abundance_all = pd.concat(abundance.values(), ignore_index=True)

# %% Merge dataframes

paxdb_df = pd.merge(
    abundance_all, uniprot_map, on=["paxdb_id", "species_number"], how="left"
)
paxdb_df = paxdb_df.dropna(subset=["uniprot_id"])
paxdb_df = paxdb_df[
    ["species", "species_number", "condition", "paxdb_id", "uniprot_id", "abundance"]
]

# %% Save dataframes

full_paxdb_clean_file = (
    get_path("data", "intermediate", "paxdb", version=PAXDB_VERSION, fmt="parquet")
    / "paxdb_raw_conditions.parquet"
)

if not full_paxdb_clean_file.parent.exists():
    full_paxdb_clean_file.parent.mkdir(parents=True, exist_ok=True)

paxdb_df.to_parquet(full_paxdb_clean_file, index=False)
logger.info(f"Saved full PAXDB dataframe to {full_paxdb_clean_file}.")
# %%
