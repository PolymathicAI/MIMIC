#!/usr/bin/env python3
# %%

from lore import logger
from lore import paths
import pandas as pd
import pyBigWig
import numpy as np
from tqdm.auto import tqdm
import duckdb
import matplotlib.pyplot as plt

tqdm.pandas()
DROP_NAN_RATIO = 0.50
VERSION = "1.4"
# %%

# Step 1: Resolve all paths
path_mm39 = paths.get_path("data", "downloads", "phylop", "mm39-35way") / "mm39.phyloP35way.bw"
path_hg38 = paths.get_path("data", "downloads", "phylop", "hg38-100way") / "hg38.phyloP100way.bw"
path_seq_coords = paths.get_path("data", "intermediate", "transcripts", VERSION) / "transcript_sequence_coords.parquet"

for path in [path_mm39, path_hg38, path_seq_coords]:
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    
path_out_mouse = paths.get_path("data", "modality", "phylop_mouse", VERSION, fmt="parquet") / "dataset.parquet"
path_out_human = paths.get_path("data", "modality", "phylop_human", VERSION, fmt="parquet") / "dataset.parquet"

for path in [path_out_mouse, path_out_human]:
    if path.parent.exists():
        raise FileExistsError(f"Path already exists: {path.parent}")

# %%

# Step 2: Open BigWig files and ID files
logger.info("Opening phyloP BigWig files…")
bw_mm39 = pyBigWig.open(path_mm39.as_posix())
bw_hg38 = pyBigWig.open(path_hg38.as_posix())
chrom_lens = {"Homo sapiens": bw_hg38.chroms(), "Mus musculus": bw_mm39.chroms()}
chrom_list = {species:len_dict.keys() for species, len_dict in chrom_lens.items()}

# %%

# Step 3: Load and pre-filter transcript metadata
logger.info("Loading transcripts DataFrame")
df_coords = pd.read_parquet(path_seq_coords)
df_coords = df_coords[df_coords["organism_name"].isin(["Homo sapiens", "Mus musculus"])]
logger.info(f"Loaded {len(df_coords)} human and mouse transcripts")

# %%

# Step 4: Row-wise phyloP fetcher (unchanged)
def fetch_phylop(row):
    species, chrom, start, length, strand = (
            row["organism_name"],
            row["seqid"],
            row["sequence_start"],
            row["sequence_length"],
            row["strand"]
    )
    end = start + length

    if chrom not in chrom_list[species]:
        return [np.nan] * length

    chrom_len = chrom_lens[species][row["seqid"]]
    if start >= chrom_len:
        return [np.nan] * length

    # make sure we are not out of bounds
    trunc_end = min(end, chrom_len)

    bw = bw_hg38 if species == "Homo sapiens" else bw_mm39
    try:
        vals = bw.values(chrom, start, trunc_end) 
        if trunc_end < end:
            # pad vals to be the right length
            vals = np.pad(vals, (0, end - trunc_end), constant_values=np.nan)
        if strand == '-':
            vals = vals[::-1]
        return vals
    except Exception as e:
        logger.debug(f"Error fetching {species} {chrom}:{start}-{end}: {e}")
        return [np.nan] * length
    
# %%

# Step 5–8: Process **Mouse first, then Human** in a simple loop
species_cfg = [
    ("Mus musculus", "Mouse", bw_mm39, path_out_mouse),
    ("Homo sapiens", "Human", bw_hg38, path_out_human),
]

for org_name, short, bw_handle, out_path in species_cfg:
    logger.info(f"=== Processing {short} transcripts ===")

    df_sp = df_coords[df_coords["organism_name"] == org_name].copy()
    logger.info(f"{len(df_sp)} {short} transcripts to process")

    logger.info("Fetching phyloP scores…")
    df_sp["phylop"] = df_sp.progress_apply(fetch_phylop, axis=1)

    nan_rows = df_sp["phylop"].apply(
        lambda x: (np.isnan(x).sum() / len(x)) > DROP_NAN_RATIO
    )
    df_good = df_sp[~nan_rows].reset_index(drop=True)
    logger.info(
        f"{len(df_good)} {short} transcripts retained "
        f"({len(df_sp) - len(df_good)} dropped for >{DROP_NAN_RATIO:.0%} NaNs)"
    )

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=False)
    df_good[["genome_feature_id", "phylop"]].to_parquet(out_path, index=False)
    logger.info(f"Saved cleaned {short} data → {out_path}")

    # Quick NaN-ratio histogram
    nan_ratio = df_good["phylop"].apply(lambda v: np.isnan(v).mean())
    plt.figure(figsize=(5, 3.4))
    plt.hist(nan_ratio, bins=100, log=True)
    plt.xlabel("Proportion of NaN values")
    plt.ylabel("Count")
    plt.title(f"{short} phyloP NaN distribution")
    plt.grid(axis="y", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{short.lower()}_phylop_nan_ratio.png", dpi=300)
    plt.close()

    # free up memory
    del df_good, df_sp, nan_rows, nan_ratio
    bw_handle.close()  

# %%
# Step 9: Lightweight verification with DuckDB
logger.info("Verifying saved Parquet files with DuckDB")
con = duckdb.connect()

for path, short in [(path_out_mouse, "Mouse"), (path_out_human, "Human")]:
    row_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path.as_posix()}')"
    ).fetchone()[0]
    first5 = con.execute(
        f"SELECT * FROM read_parquet('{path.as_posix()}') LIMIT 5"
    ).fetchdf()
    logger.info(f"{short}: {row_count} rows\nFirst 5 rows:\n{first5}")

# %%
