#!/usr/bin/env python3
# 01_convert_bed_to_bigwig.py
# Convert ATAC narrowPeak BED/BED.GZ to bigWig using signalValue as track signal.
# Now fixes overlaps via bedtools sort+merge and clips with bedClip.
# Within each experiment (each BED file), signalValue is binned into 10 quantile bins (1–10).

# %%
import csv
import gzip
import os
import shutil
import sys
import platform
import stat
import subprocess as sp
import urllib.request
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from lore import paths
from lore import logger

# ---------------- Pipeline paths ----------------
ATAC_PATH = paths.get_path(data_type="data", stage="downloads", name="atac", version="encode-narrowpeak")

CSV_LIFTED = ATAC_PATH / "metadata_atac_bed_narrowpeak_LIFTED.csv"
PEAKS_DIR = ATAC_PATH / "atac_peaks"                                   # human GRCh38 (gz)
LIFT_DIR = ATAC_PATH / "liftover" / "lifted" / "mm10_to_mm39"          # mouse mm39 (plain .bed)

OUT_ROOT = paths.get_path(
    data_type="data",
    stage="intermediate",
    name="atac",
    version="encode-narrowpeak",
    fmt="bigwig",
)                                                                      # bigWig target root
TMP_DIR = OUT_ROOT / "_tmp"                                            # temp working area
SIZES_DIR = OUT_ROOT / "chrom.sizes"                                   # cache chrom.sizes

OUT_ROOT.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)
SIZES_DIR.mkdir(parents=True, exist_ok=True)

TOOLS_DIR = Path.home() / ".local" / "bin"

# ---------------- Options ----------------
QVALUE_MIN = 0.0   # optional qValue filter (use >0 to filter)
BIN_COUNT = 0      # kept for backwards compatibility but not used (we always bin into 10 quantiles)

# Global histogram of peak counts per bin (1–10) across all experiments
BIN_HIST = np.zeros(10, dtype=np.int64)

# ---------------- Helpers ----------------
UCSC_SIZES = {
    "GRCh38": "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes",
    "hg38":   "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes",
    "mm39":   "http://hgdownload.soe.ucsc.edu/goldenPath/mm39/bigZips/mm39.chrom.sizes",
}

def have_tool(tool: str) -> bool:
    return sp.run(
        ["bash", "-lc", f"command -v {tool} >/dev/null 2>&1"],
        capture_output=True
    ).returncode == 0

def ensure_ucsc_tool(name: str):
    if have_tool(name):
        return
    sys_plat = sys.platform
    arch = platform.machine().lower()
    if sys_plat.startswith("linux"):
        if arch in ("x86_64", "amd64"):
            url = f"http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/{name}"
        elif arch in ("aarch64", "arm64"):
            url = f"http://hgdownload.soe.ucsc.edu/admin/exe/linux.aarch64/{name}"
        else:
            raise SystemExit(
                f"Unsupported linux arch '{arch}' for {name}. "
                f"Try: mamba/conda install -c bioconda ucsc-{name.lower()}"
            )
    elif sys_plat == "darwin":
        url = f"http://hgdownload.soe.ucsc.edu/admin/exe/macOSX.x86_64/{name}"
    else:
        raise SystemExit(
            f"Unsupported platform '{sys_plat}'. Try bioconda: ucsc-{name.lower()}"
        )

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    dest = TOOLS_DIR / name
    logger.info(f"Downloading {name} → {dest}")
    urllib.request.urlretrieve(url, dest.as_posix())
    os.chmod(dest, os.stat(dest).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    os.environ["PATH"] = f"{TOOLS_DIR}{os.pathsep}" + os.environ.get("PATH", "")
    if not have_tool(name):
        raise SystemExit(
            f"Failed to make {name} available; verify {dest} or install via bioconda."
        )

def ensure_bedtools():
    if have_tool("bedtools"):
        return
    raise SystemExit(
        "ERROR: bedtools not found in PATH. Install with:\n"
        "  mamba install -n central -c bioconda bedtools\n"
        "or add bedtools to your active environment."
    )

def get_chrom_sizes(genome: str) -> Path:
    url = UCSC_SIZES.get(genome)
    if not url:
        raise SystemExit(
            f"No chrom.sizes URL known for genome '{genome}'. Add mapping in UCSC_SIZES."
        )
    dest = SIZES_DIR / f"{genome}.chrom.sizes"
    if not dest.exists():
        logger.info(f"Downloading chrom.sizes for {genome} -> {dest}")
        sp.run(["bash", "-lc", f"curl -L -o {dest} {url}"], check=True)
    return dest

def bed_like_iter(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            if not line or line.startswith(("#", "track", "browser")):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            yield parts

def build_bedgraph(in_bed: Path, out_bg: Path, qmin: float = 0.0, bin_count: int = 0) -> np.ndarray:
    """
    Create a bedGraph (chrom, start, end, value) from narrowPeak using signalValue (col 7).
    Within each BED file (one experiment), signalValue is binned into 10 quantile bins
    with integer labels 1–10. (0 is reserved for "closed chromatin" later at transcript level.)
    Returns a 10-element histogram of peak counts per bin for this BED file.
    """
    rows = []
    for p in bed_like_iter(in_bed):
        try:
            chrom, start, end = p[0], int(p[1]), int(p[2])
        except Exception:
            continue
        # signalValue (index 6)
        signal = 0.0
        if len(p) >= 7:
            try:
                signal = float(p[6])
            except Exception:
                signal = 0.0

        if qmin > 0.0 and len(p) >= 9:
            try:
                qval = float(p[8])
                if qval < qmin:
                    continue
            except Exception:
                pass
        rows.append((chrom, start, end, signal))

    # If no rows, write empty file and return zero histogram
    if not rows:
        out_bg.write_text("")
        return np.zeros(10, dtype=np.int64)

    df = pd.DataFrame(rows, columns=["chrom", "start", "end", "val"])

    # --- Quantile binning into 10 bins labeled 1–10 ---
    vals = df["val"].astype(float).to_numpy()
    local_hist = np.zeros(10, dtype=np.int64)

    if np.all(vals == vals[0]):
        # All signalValues identical → put them all in a single bin (e.g. 5)
        df["val"] = 5
        local_hist[4] += len(vals)  # index 4 → bin label 5
    else:
        # Compute 11 quantile boundaries (0%, 10%, ..., 100%)
        cuts = np.quantile(vals, np.linspace(0.0, 1.0, 11))
        edges = cuts[1:-1]  # 9 edges → 10 bins

        # np.digitize: vals <= edges[0] → 0, ..., vals > edges[-1] → 9
        bin_idx = np.digitize(vals, bins=edges, right=True) + 1
        bin_idx = np.clip(bin_idx, 1, 10)
        df["val"] = bin_idx

        # Count peaks in each bin 1–10
        counts = np.bincount(bin_idx, minlength=11)[1:11]  # ignore bin 0
        local_hist += counts

    # Note: BIN_COUNT is ignored; we always produce 10 bins (1–10).

    # pre-sort; bedtools will sort again, but this helps
    df = df.sort_values(["chrom", "start", "end"], kind="mergesort")
    df.to_csv(out_bg, sep="\t", header=False, index=False)

    logger.debug(f"Finished binning {in_bed.name}: peaks={len(vals)}, local_hist_sum={int(local_hist.sum())}")
    return local_hist

def nonoverlap_and_clip(in_bg: Path, chrom_sizes: Path) -> Path:
    """
    sort+merge (-c 4 -o max) with bedtools, then clip to chrom sizes in Python
    to avoid UCSC bedClip's GLIBC dependency.
    """
    sorted_bg = TMP_DIR / (in_bg.stem + ".sorted.bedGraph")
    merged_bg = TMP_DIR / (in_bg.stem + ".merged.bedGraph")
    clipped_bg = TMP_DIR / (in_bg.stem + ".clipped.bedGraph")

    # bedtools sort
    r = sp.run(["bash", "-lc", f"bedtools sort -i {in_bg} > {sorted_bg}"], text=True)
    if r.returncode != 0:
        raise RuntimeError(f"bedtools sort failed for {in_bg}")

    # bedtools merge (collapse overlaps; keep max value)
    r = sp.run(
        ["bash", "-lc", f"bedtools merge -i {sorted_bg} -c 4 -o max > {merged_bg}"],
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"bedtools merge failed for {sorted_bg}")

    # --- Python clip to chrom sizes ---
    sizes = {}
    with open(chrom_sizes) as f:
        for line in f:
            if not line.strip():
                continue
            c, L = line.split()[:2]
            sizes[c] = int(L)

    # load merged bedGraph (chrom start end val)
    df = pd.read_csv(merged_bg, sep="\t", header=None, names=["chrom", "start", "end", "val"])

    # keep rows with known chrom
    df = df[df["chrom"].isin(sizes.keys())].copy()

    # clip to [0, chrom_len], drop invalid (start>=end)
    df["start"] = df["start"].clip(lower=0)
    df["end"] = df.apply(lambda r: min(r["end"], sizes[r["chrom"]]), axis=1)
    df = df[df["start"] < df["end"]]

    # write clipped
    df.to_csv(clipped_bg, sep="\t", header=False, index=False)

    # cleanup intermediates
    for tmp in (sorted_bg, merged_bg):
        try:
            tmp.unlink(missing_ok=True)
        except OSError as e:
            logger.debug(f"Could not delete temporary file {tmp}: {e}")

    return clipped_bg

def convert_one(in_path: Path, genome: str, out_bw: Path):
    """
    Convert one BED/BED.GZ narrowPeak to bigWig via bedGraph; fix overlaps & clip.
    Returns (produced_successfully, local_histogram_of_bins_1_to_10).
    """
    chrom_sizes = get_chrom_sizes(genome)
    out_bw.parent.mkdir(parents=True, exist_ok=True)

    tmp_bg = TMP_DIR / (out_bw.stem + ".bedGraph")
    local_hist = np.zeros(10, dtype=np.int64)
    try:
        local_hist = build_bedgraph(in_path, tmp_bg, qmin=QVALUE_MIN, bin_count=BIN_COUNT)
        if tmp_bg.stat().st_size == 0:
            logger.warning(f"Empty bedGraph for {in_path.name}; skipping.")
            tmp_bg.unlink(missing_ok=True)
            return False, local_hist

        # sort + merge (max) + clip
        try:
            fixed_bg = nonoverlap_and_clip(tmp_bg, chrom_sizes)
        finally:
            tmp_bg.unlink(missing_ok=True)

        # bedGraphToBigWig
        cmd = ["bedGraphToBigWig", str(fixed_bg), str(chrom_sizes), str(out_bw)]
        r = sp.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            logger.error(f"bedGraphToBigWig failed for {in_path.name}: {r.stderr.strip()}")
            return False, local_hist
        if r.stdout:
            logger.debug(r.stdout.strip())
        ok = out_bw.exists() and out_bw.stat().st_size > 0
        return ok, local_hist
    finally:
        # clean fixed bg if present
        for extra in (".sorted.bedGraph", ".merged.bedGraph", ".clipped.bedGraph"):
            p = TMP_DIR / (out_bw.stem + extra)
            p.unlink(missing_ok=True)

if __name__ == "__main__":
    # Ensure tools
    ensure_ucsc_tool("bedGraphToBigWig")
    ensure_bedtools()

    if not CSV_LIFTED.exists():
        raise SystemExit(f"ERROR: {CSV_LIFTED} not found. Run liftover/metadata update first.")

    # Read lifted metadata
    df = pd.read_csv(CSV_LIFTED)

    supported = {"GRCh38", "hg38", "mm39"}
    df = df[df["genome"].isin(supported)].copy()

    jobs = []
    for _, r in df.iterrows():
        genome = r["genome"]
        species = r["species"]
        fname = str(r["file_name"])

        if genome in {"GRCh38", "hg38"}:
            in_path = PEAKS_DIR / fname
        elif genome == "mm39":
            in_path = LIFT_DIR / fname
        else:
            continue

        if not in_path.exists():
            logger.warning(f"Missing source file for {genome}/{species}: {in_path}")
            continue

        bw_name = Path(fname).name
        for ext in (".narrowPeak.gz", ".bed.gz", ".bed"):
            if bw_name.endswith(ext):
                bw_name = bw_name[: -len(ext)] + ".bw"
                break
        out_bw = OUT_ROOT / genome / species / bw_name
        jobs.append((in_path, genome, out_bw))

    logger.info(f"Preparing to convert {len(jobs)} files to bigWig -> {OUT_ROOT}")

    ok = 0
    fail = 0
    for in_path, genome, out_bw in tqdm(jobs, desc="BED->bigWig"):
        # If bigWig already exists, recompute stats only (no conversion)
        if out_bw.exists() and out_bw.stat().st_size > 0:
            tmp_bg = TMP_DIR / (out_bw.stem + ".stats_only.bedGraph")
            try:
                local_hist = build_bedgraph(in_path, tmp_bg, qmin=QVALUE_MIN, bin_count=BIN_COUNT)
            finally:
                tmp_bg.unlink(missing_ok=True)
            BIN_HIST += local_hist
            ok += 1
            continue

        # Otherwise, do full conversion + stats
        produced, local_hist = convert_one(in_path, genome, out_bw)
        if produced:
            ok += 1
            BIN_HIST += local_hist
        else:
            fail += 1

    logger.info(f"Conversion complete. Success: {ok}  Fail: {fail}")
    if fail == 0:
        shutil.rmtree(TMP_DIR, ignore_errors=True)

    # --- Plot global peak-bin distribution across all experiments ---
    total_peaks = int(BIN_HIST.sum())
    if total_peaks > 0:
        bins = np.arange(1, 11)
        plt.figure(figsize=(5, 3.5))
        plt.bar(bins, BIN_HIST)
        plt.xticks(bins)
        plt.xlabel("Quantile bin (1–10)")
        plt.ylabel("Number of peaks")
        plt.title(f"ATAC peak distribution across all experiments\nTotal peaks = {total_peaks:,}")
        plt.tight_layout()
        out_plot = OUT_ROOT / "peak_bin_distribution_all_experiments.png"
        plt.savefig(out_plot, dpi=300)
        plt.close()
        logger.info(f"Saved global peak bin histogram → {out_plot}")
    else:
        logger.warning("No peaks were binned; skipping peak bin histogram plot.")
