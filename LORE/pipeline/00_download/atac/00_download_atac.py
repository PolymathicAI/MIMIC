#!/usr/bin/env python3
# 00_download_atac.py
# - Resolves a pipeline path via lore.paths
# - Prompts to overwrite/resume the folder
# - Fetches ENCODE JSON, writes metadata + urls, downloads peak files into path
# - Logs with lore.logger

# %%
import shutil
import json
import csv
import os
import datetime
import time
from pathlib import Path
from urllib.request import Request, urlopen, urlretrieve

from lore import paths
from lore import logger

# %%

# Destination folder (analogous to phylop's versioned folder)
# You can change version label if you later add a specific snapshot date.
path = paths.get_path(data_type="data", stage="downloads", name="atac", version="encode-narrowpeak")

# ENCODE query: all released ATAC-seq narrowPeak BED files (embedded JSON)
SEARCH_URL = (
    "https://www.encodeproject.org/search/?type=File"
    "&assay_title=ATAC-seq&file_format=bed&file_format_type=narrowPeak"
    "&status=released&format=json&frame=embedded&limit=all"
)

# RASP-style metadata columns (kept identical to your prior outputs)
COLS = [
    "technology","genome","publication","year","reagent","condition","strand","scale",
    "cellline","description","kingdom","species","file_name"
]

# Map ENCODE organism/scientific_name -> (kingdom, species_key) for your conventions
ORG_MAP = {
    "Homo sapiens": ("animals","human"),
    "Mus musculus": ("animals","mouse"),
    "Drosophila melanogaster": ("animals","drosophila"),
    "Rattus norvegicus": ("animals","rat"),
    "Danio rerio": ("animals","zebrafish"),
    "Saccharomyces cerevisiae": ("fungi","yeast"),
}

# Fallback organism inference from assembly (used if organism block missing)
ASSEMBLY_TO_ORG = {
    "hg19":"Homo sapiens","GRCh37":"Homo sapiens","hg38":"Homo sapiens","GRCh38":"Homo sapiens",
    "mm10":"Mus musculus","mm39":"Mus musculus",
    "dm6":"Drosophila melanogaster",
    "rn6":"Rattus norvegicus",
    "danRer11":"Danio rerio",
    "sacCer3":"Saccharomyces cerevisiae",
}

# %%

def _fetch_json(url: str) -> dict:
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req) as r:
        return json.load(r)

def _safe_year(iso_date: str) -> str:
    if not iso_date:
        return "N/A"
    try:
        return str(datetime.date.fromisoformat(iso_date[:10]).year)
    except Exception:
        return "N/A"

def _upper_or_na(s: str) -> str:
    return (str(s).upper() if s and str(s).lower() != "none" else "N/A")

def _make_rows(data: dict):
    """Return (rows, urls) from ENCODE embedded JSON."""
    rows, urls = [], []
    for fobj in data.get("@graph", []):
        href = fobj.get("href")  # "/files/ENCFFxxxx/@@download/ENCFFxxxx.narrowPeak.gz"
        accession = fobj.get("accession") or (os.path.basename(href).split(".")[0] if href else "UNKNOWN")
        assembly = fobj.get("assembly") or "N/A"

        ds = fobj.get("dataset")
        if isinstance(ds, dict):
            lab_title = (ds.get("lab") or {}).get("title") or "ENCODE Project"
            date_rel = ds.get("date_released") or fobj.get("date_created")
        else:
            lab_title = "ENCODE Project"
            date_rel = fobj.get("date_created")
        year = _safe_year(date_rel)

        org = fobj.get("organism")
        if isinstance(org, dict):
            organism = org.get("scientific_name") or "N/A"
        else:
            organism = ASSEMBLY_TO_ORG.get(assembly, "N/A")

        bs = fobj.get("biosample_ontology")
        biosample = (bs.get("term_name") if isinstance(bs, dict) else None) or "N/A"

        kingdom, species_key = ORG_MAP.get(organism, ("unknown", organism.lower().replace(" ","_")))
        file_name = os.path.basename(href) if href else f"{accession}.narrowPeak"

        row = {
            "technology": "ATAC-seq",
            "genome": assembly,
            "publication": lab_title,
            "year": year,
            "reagent": "N/A",
            "condition": "N/A",
            "strand": "both",
            # keep identical to previous RASP-style export
            "scale": "transcriptome-wide",
            "cellline": _upper_or_na(biosample),
            "description": accession,
            "kingdom": kingdom,
            "species": species_key,
            "file_name": file_name,
        }
        rows.append(row)
        if href:
            urls.append("https://www.encodeproject.org" + href)
    return rows, urls

def _write_csv(rows, path_csv: Path):
    with path_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

def _write_urls(urls, path_txt: Path):
    with path_txt.open("w") as f:
        for u in urls:
            f.write(u + "\n")

def _download_all(urls, out_dir: Path, delay: float = 0.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    for u in urls:
        fn = u.rsplit("/", 1)[-1]
        dest = out_dir / fn
        if dest.exists():
            continue
        try:
            urlretrieve(u, dest)
            if delay:
                time.sleep(delay)
        except Exception as e:
            logger.warning(f"Failed to download {u}: {e}")

# %%

# Folder prompt & creation (mirrors phyloP)
if path.exists():
    logger.info(f"Folder {path} already exists.")
    overwrite = input("Do you want to overwrite them (y)? Or resume (r) (y/n/[r]): ")
    if overwrite.lower() == "r" or overwrite == "":
        logger.info("Resuming downloads/metadata build.")
        # no deletion; we will reuse existing files and fill missing ones
    elif overwrite.lower() == "y":
        logger.info(f"Deleting {path}")
        shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=False)
    elif overwrite.lower() == "n":
        logger.info("Exiting.")
        raise SystemExit(0)
    else:
        logger.info("Invalid input. Exiting.")
        raise SystemExit(0)
else:
    path.mkdir(parents=True, exist_ok=False)

# Subpaths under the versioned download directory.
# These are *outputs* produced by this script (not required inputs).
json_out_path = path / "encode_atac_bed_narrowpeak.json"
csv_out_path  = path / "metadata_atac_bed_narrowpeak.csv"
urls_out_path = path / "urls.txt"
peaks_dir     = path / "atac_peaks"

# %%

logger.info(f"Querying ENCODE for ATAC narrowPeak (writing JSON to {json_out_path}) ...")
data = _fetch_json(SEARCH_URL)
with json_out_path.open("w") as f:
    json.dump(data, f)
n = len(data.get("@graph", []))
logger.info(f"ENCODE returned {n} files")

logger.info("Building RASP-style metadata + URL list ...")
rows, urls = _make_rows(data)
_write_csv(rows, csv_out_path)
_write_urls(urls, urls_out_path)
logger.info(f"Wrote metadata CSV: {csv_out_path}")
logger.info(f"Wrote URL list    : {urls_out_path}")

logger.info(f"Downloading peak files to {peaks_dir} (resume-safe) ...")
_download_all(urls, peaks_dir, delay=0.0)
logger.info("Download complete.")

logger.info("ATAC download & metadata step finished successfully.")

