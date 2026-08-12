#!/usr/bin/env python3
from lore import logger
from lore import paths
import pandas as pd
import pyBigWig
import numpy as np
from tqdm.auto import tqdm
import duckdb
import matplotlib.pyplot as plt

import glob, os

_VERSION = "1.4"

tqdm.pandas()
DROP_NAN_RATIO = 0.90 # Allow more NaNs since RASP2 has more missing data

rasp_species_list = ['Homo sapiens',
 'Mus musculus',
 'Arabidopsis thaliana',
 'Oryza sativa',
 'Danio rerio',
 'Chlorocebus sabaeus',
 'Saccharomyces cerevisiae',
 'Bacillus cereus',
 'Pseudomonas putida',
 'Salmonella enterica',
 'Escherichia coli',
 'Bacillus subtilis',
 'Synechococcus sp.',
 'Orthoflavivirus zikaense',
 'Orthoflavivirus denguei',
 'Hepacivirus hominis',
 'Chikungunya virus',
 'Alphainfluenzavirus influenzae',
 'Human immunodeficiency virus 1',
 'Severe acute respiratory syndrome-related coronavirus',
 'Rotavirus A',
 'Cytomegalovirus humanbeta5',
 'Tobacco virtovirus 1']
# A simple config for a unifed processing loop
species_cfg = {
    'Homo sapiens': ['animals', 'human'],
    'Mus musculus': ['animals', 'mouse_mm39'],
    'Danio rerio': ['animals', 'zebrafish'],
    'Oryza sativa': ['plants', 'rice'],
    'Chlorocebus sabaeus': ['animals', 'GreenMonkey'],
    'Arabidopsis thaliana': ['plants', 'A.thaliana'],
    'Saccharomyces cerevisiae': ['bacteria_fungi', 'yeast'],
    'Pseudomonas putida': ['bacteria_fungi', 'P.putida'],
    'Salmonella enterica': ['bacteria_fungi', 'S.Enterica'],
    'Escherichia coli': ['bacteria_fungi', 'E.coli'],
    'Bacillus subtilis': ['bacteria_fungi', 'B.subtilis'],
    'Bacillus cereus': ['bacteria_fungi', 'B.cereus'],
    'Synechococcus sp.': ['bacteria_fungi', 'synechococcus'],
    'Chikungunya virus': ['virus', 'CHIKV'],
    'Cytomegalovirus humanbeta5': ['virus', 'CMV'],
    'Orthoflavivirus denguei': ['virus', 'Dengue'],
    'Hepacivirus hominis': ['virus', 'HCV'],
    'Human immunodeficiency virus 1': ['virus', 'HIV'],
    'Alphainfluenzavirus influenzae': ['virus', 'IAV'],
    'Rotavirus A': ['virus', 'Rotavirus'],
    'Severe acute respiratory syndrome-related coronavirus': ['virus', 'SARS-CoV-2'],
    'Tobacco virtovirus 1': ['virus', 'STMV'],
    'Orthoflavivirus zikaense': ['virus', 'Zika']
}

# map chromosome seqids for some species
chrom_seq_map = {
    'A.thaliana': {
        '1': 'NC_003070.9',
        '2': 'NC_003071.7',
        '3': 'NC_003074.8',
        '4': 'NC_003075.7',
        '5': 'NC_003076.8',
        'Mt': 'NC_037304.1',
        'Pt': 'NC_000932.1'
    },
    'zebrafish': {
        '1': 'NC_007112.7',
        '2': 'NC_007113.7',
        '3': 'NC_007114.7',
        '4': 'NC_007115.7',
        '5': 'NC_007116.7',
        '6': 'NC_007117.7',
        '7': 'NC_007118.7',
        '8': 'NC_007119.7',
        '9': 'NC_007120.7',
        '10': 'NC_007121.7',
        '11': 'NC_007122.7',
        '12': 'NC_007123.7',
        '13': 'NC_007124.7',
        '14': 'NC_007125.7',
        '15': 'NC_007126.7',
        '16': 'NC_007127.7',
        '17': 'NC_007128.7',
        '18': 'NC_007129.7',
        '19': 'NC_007130.7',
        '20': 'NC_007131.7',
        '21': 'NC_007132.7',
        '22': 'NC_007133.7',
        '23': 'NC_007134.7',
        '24': 'NC_007135.7',
        '25': 'NC_007136.7'
    }# Note: mitochondrion not included for zebrafish
}

def fetch_rasp_score(row):
    chrom, start, length = (
            row["seqid"],
            row["sequence_start"],
            row["sequence_length"]
    )
    end = start + length

    chrom_len = chrom_lens[row["seqid"]]
    if start >= chrom_len:
        return np.array([np.nan] * length)

    # make sure we are not out of bounds
    trunc_end = min(end, chrom_len)

    try:
        vals = bw_target.values(chrom, start, trunc_end) 
        if trunc_end < end:
            # pad vals to be the right length
            vals = np.pad(vals, (0, end - trunc_end), constant_values=np.nan)
        return np.array(vals)
    except Exception as e:
        logger.debug(f"Error fetching {chrom}:{start}-{end}: {e}")
        return np.array([np.nan] * length)

# Step 1: Get transcript sequence from existing modality & from RASP2 annotations
path_seq_coords = paths.get_path("data", "intermediate", "transcripts", _VERSION) / "transcript_sequence_coords.parquet"
df_coords = pd.read_parquet(path_seq_coords)

path_out = paths.get_path("data", "intermediate", "rna2d", "rasp2") / _VERSION
logger.info("RASP2 transcript extraction for all species:")
for specie in tqdm(rasp_species_list, desc="Processing species"):
    logger.info("...............................")
    kingdom = species_cfg[specie][0]
    short = species_cfg[specie][1]
    path_out_temp = path_out / f"{short}"
    path_out_temp.mkdir(parents=True, exist_ok=True)
    paths_bw_target = list((paths.get_path("data", "downloads", "rna2d", "rasp2") / "score_data"/ f"{kingdom}" / f"RASP_files_{short}").glob("*.bw"))
    logger.info(f"{short} / {specie}: {len(paths_bw_target)} bw files found....")
    if len(paths_bw_target) == 0:
        logger.warning(f"No bw files found for {short} / {specie}, skipping...")
        continue
    list_of_dfs = []
    for i, path in enumerate(paths_bw_target):
        try:
            logger.info(f"Processing {short} {i+1}/{len(paths_bw_target)}: {path}")
            target_path = path.as_posix()
            bw_target = pyBigWig.open(target_path)
            chrom_lens = bw_target.chroms()
            chrom_list = bw_target.chroms().keys()
            # map chrom seqids if needed
            if short in chrom_seq_map:
                chrom_list = [chrom_seq_map[short][chrom] if chrom in chrom_seq_map[short] else chrom for chrom in chrom_list]
            df_target = df_coords[df_coords["organism_name"].isin([specie]) & df_coords["seqid"].isin(chrom_list)].copy()
            # map chrom seqids in df_target if needed
            if short in chrom_seq_map:
                # change seq_id to be the key in chrom_seq_map, now it is the value
                df_target["seqid"] = df_target["seqid"].apply(lambda x: [k for k, v in chrom_seq_map[short].items() if v == x][0] if x in chrom_seq_map[short].values() else x)
            df_target["rasp_score"] = df_target.progress_apply(fetch_rasp_score, axis=1)
            logger.info("Dropping rows with too many NaNs...")
            nan_rows = df_target["rasp_score"].apply(
                lambda x: (np.isnan(x).sum() / len(x)) > DROP_NAN_RATIO
            )
            df_good = df_target[~nan_rows].reset_index(drop=True).copy()
            df_good["context"] = os.path.basename(target_path)
            df_good.drop(columns=["seqid", "organism_name", "sequence_start", "sequence_length"], inplace=True)
            logger.info(
                f"{len(df_good)} {short} transcripts retained "
                f"({len(df_target) - len(df_good)} dropped for >{DROP_NAN_RATIO:.0%} NaNs)"
            )
            list_of_dfs.append(df_good)
            bw_target.close()
            df_good.to_parquet(path_out_temp / f"{short}_subset_{i}_{os.path.basename(target_path).split('.')[0]}.parquet", index=False)
            # free up memory
            del df_target, df_good, nan_rows
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            continue
    df_combined = pd.concat(list_of_dfs, ignore_index=True)
    out_filename = path_out / f"{short}_transcript_rasp_scores.parquet"
    df_combined.to_parquet(out_filename, index=False)
    logger.info(f"Saved combined {short} data → {out_filename}")
    # Quick NaN-ratio histogram
    nan_ratio = df_combined["rasp_score"].apply(lambda v: np.isnan(v).mean())
    plt.figure(figsize=(5, 3.4))
    plt.hist(nan_ratio, bins=100, log=True)
    plt.xlabel("Proportion of NaN values")
    plt.ylabel("Count")
    plt.title(f"{short} RASP2 NaN distribution")
    plt.grid(axis="y", alpha=0.7)
    plt.tight_layout()
    plt.savefig(path_out / f"{short.lower()}_rasp2_nan_ratio.png", dpi=300)
    plt.close()
    # free up memory
    del df_combined, list_of_dfs