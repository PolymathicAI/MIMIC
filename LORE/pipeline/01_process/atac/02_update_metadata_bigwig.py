#!/usr/bin/env python3
# Update metadata to point at .bw outputs produced by 01_convert_bed_to_bigwig.py

import csv
from pathlib import Path
import pandas as pd
from lore import paths
from lore import logger

# Inputs from prior steps
ATAC_PATH = paths.get_path(data_type="data", stage="downloads",   name="atac", version="encode-narrowpeak")
CSV_LIFTED = ATAC_PATH / "metadata_atac_bed_narrowpeak_LIFTED.csv"

# Where 01_convert_bed_to_bigwig.py wrote bigWigs
BW_ROOT = paths.get_path(data_type="data", stage="intermediate", name="atac", version="encode-narrowpeak", fmt="bigwig")

# Outputs
CSV_OUT = ATAC_PATH / "metadata_atac_bigwig.csv"
SPLIT = True  # also write score_data/<kingdom>/metadata_<species>_atac_bigwig.csv

def bw_name_from_source(src_name:str) -> str:
    # replace common extensions with .bw
    for ext in (".narrowPeak.gz",".bed.gz",".bed"):
        if src_name.endswith(ext):
            return src_name[: -len(ext)] + ".bw"
    # if already .lifted.bed etc.
    if src_name.endswith(".lifted.bed"):
        return src_name[:-len(".lifted.bed")] + ".bw"
    return Path(src_name).with_suffix(".bw").name

def find_bw(genome:str, species:str, bw_basename:str) -> Path:
    # 01_convert wrote: <BW_ROOT>/<genome>/<species>/<file>.bw
    return BW_ROOT / genome / species / bw_basename

def main():
    if not CSV_LIFTED.exists():
        raise SystemExit(f"Missing {CSV_LIFTED}; run liftover step first.")
    df = pd.read_csv(CSV_LIFTED)

    df["signal_encoding"] = "atac_signalValue_rank_bin_1_10"
    
    rows = []
    missing = 0
    for r in df.to_dict(orient="records"):
        genome  = str(r.get("genome","")).strip()
        species = str(r.get("species","")).strip()
        src     = str(r.get("file_name","")).strip()

        bw_base = bw_name_from_source(src)
        bw_path = find_bw(genome, species, bw_base)
        if bw_path.exists() and bw_path.stat().st_size > 0:
            r2 = dict(r)
            r2["file_name"] = bw_base   # keep just basename, consistent with earlier metadata
            rows.append(r2)
        else:
            # keep original row if bw missing (so you can see what to regenerate)
            rows.append(r)
            missing += 1

    cols = list(df.columns)
    with CSV_OUT.open("w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    logger.info(f"Wrote {len(rows)} rows -> {CSV_OUT} (missing bw for {missing} rows)")

    if SPLIT:
        out_root = ATAC_PATH / "score_data"
        out_root.mkdir(parents=True, exist_ok=True)
        d = pd.read_csv(CSV_OUT)
        for (kingdom, species), g in d.groupby(["kingdom","species"], dropna=False):
            out_dir = out_root / str(kingdom)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / f"metadata_{species}_atac_bigwig.csv"
            g.to_csv(out_csv, index=False)
            logger.info(f"Wrote {len(g)} rows -> {out_csv}")

if __name__ == "__main__":
    main()
