#!/usr/bin/env python3
# 03_fetch_transcript_scores.py
# Fetch per-base ATAC signal over transcripts, per experiment (bigWig), without averaging.
#
# Per base in each transcript and experiment:
#   - 1–10 = rank-binned ATAC signal (from that experiment's bigWig)
#   - 0    = no peak at that base in that experiment (closed chromatin)
#   - NaN  = transcript does not map (chrom missing or out of bounds)
#
# Output:
#   For each species (mouse/human), a parquet with one row per (transcript, experiment):
#     columns: genome_feature_id, cellline, context, atac

from lore import logger
from lore import paths
import pandas as pd
import pyBigWig
import numpy as np
from tqdm.auto import tqdm
import duckdb
import matplotlib.pyplot as plt
import os

import pyarrow as pa
import pyarrow.parquet as pq

tqdm.pandas()

# ===== Options =====
SANITY_CHECK = False              # limit number of experiments + extra plots
MAX_EXPERIMENTS_PER_SPECIES = 2   # only used if SANITY_CHECK=True

# ===== Paths =====
path_seq_coords = (
    paths.get_path("data", "intermediate", "transcripts", "1.1")
    / "transcript_sequence_coords.parquet"
)
if not path_seq_coords.exists():
    raise FileNotFoundError(f"Missing transcript coords: {path_seq_coords}")

bw_root = paths.get_path("data", "intermediate", "atac", "encode-narrowpeak", fmt="bigwig")

ATAC_PATH = paths.get_path("data", "downloads", "atac", "encode-narrowpeak")
path_meta_bw = ATAC_PATH / "metadata_atac_bigwig.csv"
if not path_meta_bw.exists():
    raise FileNotFoundError(f"Missing ATAC bigWig metadata: {path_meta_bw}")

base_out = paths.get_path("data", "modality", "atac", "1.1", fmt="parquet")
path_out_mouse = base_out / "mouse" / "dataset.parquet"
path_out_human = base_out / "human" / "dataset.parquet"

spec_to_genome_species = {
    "Homo sapiens": ("GRCh38", "human"),
    "Mus musculus": ("mm39",   "mouse"),
}

# ===== Helper: fetch ATAC for one transcript in one bigWig =====
def fetch_atac_for_transcript(row, bw, chrom_lens, chrom_set):
    """
    For one transcript and one experiment (bigWig):
      - If chrom not present -> all NaNs.
      - If transcript starts beyond chrom length -> all NaNs.
      - Else, fetch bw.values(chrom, start, trunc_end), where trunc_end is clipped
        to chrom_len, replace internal NaNs with 0 (closed chromatin in this experiment),
        and pad the tail with NaNs if the transcript extends past chrom end.
    """
    chrom = row["seqid"]
    start = int(row["sequence_start"])
    length = int(row["sequence_length"])
    end = start + length

    # Chromosome not present in this bigWig
    if chrom not in chrom_set:
        return np.full(length, np.nan, dtype="float32")

    chrom_len = int(chrom_lens[chrom])

    # Transcript starts beyond the chromosome end → fully unmappable
    if start >= chrom_len:
        return np.full(length, np.nan, dtype="float32")

    # Clip end to chromosome length
    trunc_end = min(end, chrom_len)

    try:
        vals = np.array(bw.values(chrom, start, trunc_end), dtype="float32")

        # Internal NaNs within mapped region → 0 (closed chromatin in this experiment)
        nan_mask = np.isnan(vals)
        vals[nan_mask] = 0.0

        # If transcript extends past chrom end, pad the tail with NaNs
        if trunc_end < end:
            pad_len = end - trunc_end
            pad = np.full(pad_len, np.nan, dtype="float32")
            vals = np.concatenate([vals, pad], axis=0)

        # Safety: ensure correct length
        if vals.shape[0] != length:
            logger.warning(
                f"Length mismatch for {chrom}:{start}-{end} in bigWig; "
                f"expected {length}, got {vals.shape[0]}"
            )
            if vals.shape[0] < length:
                extra = length - vals.shape[0]
                vals = np.concatenate(
                    [vals, np.full(extra, np.nan, dtype="float32")],
                    axis=0,
                )
            else:
                vals = vals[:length]

        return vals

    except Exception as e:
        logger.debug(f"Error fetching {chrom}:{start}-{end} from bigWig: {e}")
        return np.full(length, np.nan, dtype="float32")

# ===== Main processing =====
logger.info("Loading transcript coordinates…")
df_coords = pd.read_parquet(path_seq_coords)
df_coords = df_coords[df_coords["organism_name"].isin(["Homo sapiens", "Mus musculus"])].copy()
logger.info(f"Loaded {len(df_coords)} transcripts (human + mouse)")

logger.info(f"Loading ATAC bigWig metadata from {path_meta_bw}")
df_meta = pd.read_csv(path_meta_bw)

species_cfg = [
    ("Mus musculus", "Mouse", path_out_mouse),
    ("Homo sapiens", "Human", path_out_human),
]

for org_name, short, out_path in species_cfg:
    if out_path.exists():
        logger.info(f"Output for {short} already exists at {out_path}, skipping.")
        continue

    if org_name not in spec_to_genome_species:
        logger.warning(f"No genome/species mapping for {org_name}, skipping.")
        continue

    genome, species_key = spec_to_genome_species[org_name]

    df_meta_sp = df_meta[
        (df_meta["species"] == species_key) &
        (df_meta["genome"] == genome)
    ].copy()

    if df_meta_sp.empty:
        logger.warning(f"No ATAC bigWig metadata for {org_name} ({genome}/{species_key}); skipping.")
        continue

    if SANITY_CHECK and len(df_meta_sp) > MAX_EXPERIMENTS_PER_SPECIES:
        logger.warning(
            f"Sanity check enabled: restricting {short} to first "
            f"{MAX_EXPERIMENTS_PER_SPECIES} experiments out of {len(df_meta_sp)}"
        )
        df_meta_sp = df_meta_sp.iloc[:MAX_EXPERIMENTS_PER_SPECIES].copy()

    logger.info(
        f"{short}: {len(df_meta_sp)} ATAC bigWig experiments found "
        f"({genome}/{species_key})"
    )

    df_sp_coords = df_coords[df_coords["organism_name"] == org_name].copy()
    logger.info(f"{len(df_sp_coords)} {short} transcripts to consider")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    total_rows = 0

    for i, meta_row in df_meta_sp.reset_index(drop=True).iterrows():
        bw_name = str(meta_row["file_name"]).strip()
        cellline = str(meta_row.get("cellline", "N/A")).strip()

        bw_path = bw_root / genome / species_key / bw_name
        if not bw_path.exists():
            logger.warning(f"{short}: bigWig not found for experiment {bw_name} at {bw_path}, skipping.")
            continue

        logger.info(
            f"{short}: Processing experiment {i+1}/{len(df_meta_sp)} "
            f"({bw_name}, cell line={cellline})"
        )

        bw = pyBigWig.open(bw_path.as_posix())
        chrom_lens = bw.chroms()
        chrom_set = set(chrom_lens.keys())
        chrom_list = list(chrom_set)

        df_target = df_sp_coords[df_sp_coords["seqid"].isin(chrom_list)].copy()
        logger.info(
            f"{short}: {len(df_target)} transcripts on chromosomes present in {bw_name}"
        )

        if df_target.empty:
            logger.warning(
                f"{short}: No transcripts matched chromosomes for {bw_name}; skipping."
            )
            bw.close()
            continue

        df_target["atac"] = df_target.progress_apply(
            lambda r: fetch_atac_for_transcript(r, bw, chrom_lens, chrom_set),
            axis=1,
        )

        # per-experiment NaN diagnostics (unmappable transcripts)
        def nan_ratio(values):
            arr = np.asarray(values, dtype="float32")
            return float(np.isnan(arr).mean())

        nan_ratio_exp = df_target["atac"].apply(nan_ratio)
        logger.info(
            f"{short} / {bw_name}: NaN ratio (per transcript) — "
            f"mean={nan_ratio_exp.mean():.3f}, median={nan_ratio_exp.median():.3f}, "
            f"min={nan_ratio_exp.min():.3f}, max={nan_ratio_exp.max():.3f}"
        )

        df_good = df_target[["genome_feature_id", "atac"]].copy()
        df_good["cellline"] = cellline
        df_good["context"] = bw_name

        # Convert to Arrow and stream-append to Parquet
        table = pa.Table.from_pandas(df_good, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path.as_posix(), table.schema)
        writer.write_table(table)
        total_rows += len(df_good)

        if SANITY_CHECK:
            plt.figure(figsize=(5, 3.4))
            plt.hist(nan_ratio_exp, bins=100, log=True)
            plt.xlabel("Proportion of NaN values")
            plt.ylabel("Count")
            plt.title(f"{short} ATAC NaN distribution\nexperiment: {bw_name}")
            plt.grid(axis="y", alpha=0.7)
            plt.tight_layout()
            exp_plot_name = f"{short.lower()}_atac_nan_ratio_{os.path.splitext(bw_name)[0]}.png"
            plt.savefig(exp_plot_name, dpi=200)
            plt.close()

        bw.close()
        del df_target, df_good, nan_ratio_exp, table

    if writer is not None:
        writer.close()
        logger.info(
            f"Saved combined {short} ATAC data → {out_path} "
            f"({total_rows} rows = transcript × experiment)"
        )
    else:
        logger.warning(f"{short}: No experiments successfully processed; no Parquet written.")
        continue

# ===== Step: Lightweight verification with DuckDB =====
logger.info("Verifying saved Parquet files with DuckDB")
con = duckdb.connect()
for path, short in [(path_out_mouse, "Mouse"), (path_out_human, "Human")]:
    if not path.exists():
        logger.warning(f"{short}: Parquet file {path} does not exist, skipping verification.")
        continue
    row_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path.as_posix()}')"
    ).fetchone()[0]
    schema = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{path.as_posix()}')"
    ).fetchdf()
    logger.info(f"{short}: {row_count} rows\nschema:\n{schema}")

    first5 = con.execute(
        f"SELECT genome_feature_id, cellline, context, atac "
        f"FROM read_parquet('{path.as_posix()}') LIMIT 5"
    ).fetchdf()
    logger.info(f"{short}: first 5 rows (including atac):\n{first5}")
