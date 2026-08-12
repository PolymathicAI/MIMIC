#!/usr/bin/env python3
# 00_liftover_and_metadata_update.py
# - Uses lore paths/logger
# - Liftover mm10 -> mm39 with CrossMap
# - Updates metadata CSV to point to lifted files
# - Optional: split into per-species CSVs

# %%
import csv
import os
import shutil
import urllib.request

from pathlib import Path
from subprocess import run, PIPE

from lore import paths
from lore import logger

# --------- Config (adjust only if you change versions/names) ----------
# Root for this data set (mirrors phyloP layout)
ATAC_PATH = paths.get_path(data_type="data", stage="downloads", name="atac", version="encode-narrowpeak")

# Inputs produced by 00_download_atac.py (expected inside ATAC_PATH)
CSV_IN = ATAC_PATH / "metadata_atac_bed_narrowpeak.csv"
PEAKS_DIR = ATAC_PATH / "atac_peaks"

# Liftover working dirs (inside ATAC_PATH)
LIFT_BASE = ATAC_PATH / "liftover"
CHAIN_DIR = LIFT_BASE / "chains"
LOG_DIR = LIFT_BASE / "logs"
LIFT_OUTDIR = LIFT_BASE / "lifted" / "mm10_to_mm39"
WORKLIST = LIFT_BASE / "worklist_mm10_to_mm39.txt"

# Output metadata after liftover
CSV_OUT = ATAC_PATH / "metadata_atac_bed_narrowpeak_LIFTED.csv"

# UCSC chain file (must exist)
MM10_TO_MM39_CHAIN = CHAIN_DIR / "mm10ToMm39.over.chain.gz"

# Optional: write per-species CSVs in RASP layout
DO_SPLIT = True   # set False if you don't want the split CSVs
# ---------------------------------------------------------------------


def ensure_dirs():
    for p in [CHAIN_DIR, LOG_DIR, LIFT_OUTDIR]:
        p.mkdir(parents=True, exist_ok=True)


def find_crossmap_cmd():
    """Return a list command to invoke CrossMap (prefer CLI, fallback to python -m CrossMap)."""
    # Try 'CrossMap' in PATH
    r = run(["which", "CrossMap"], stdout=PIPE, stderr=PIPE, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return ["CrossMap"]
    # Fallback: python -m CrossMap (uses current python env)
    return ["python", "-m", "CrossMap"]


def build_mouse_worklist():
    """Write paths of mm10 peak files to WORKLIST."""
    if not CSV_IN.exists():
        raise FileNotFoundError(f"Missing input CSV: {CSV_IN}")
    if not PEAKS_DIR.exists():
        raise FileNotFoundError(f"Missing peaks dir: {PEAKS_DIR}")

    n = 0
    with CSV_IN.open() as f, WORKLIST.open("w") as o:
        for row in csv.DictReader(f):
            if (row.get("genome") or "").strip() == "mm10":
                p = PEAKS_DIR / (row.get("file_name") or "")
                if p.exists():
                    o.write(str(p) + "\n")
                    n += 1
    logger.info(f"Wrote {n} mouse mm10 files to worklist: {WORKLIST}")
    if n == 0:
        logger.warning("No mm10 files found—nothing to liftover.")


def liftover_all():
    """Run CrossMap bed mm10->mm39 on the worklist, with per-file logs."""
    if not MM10_TO_MM39_CHAIN.exists():
        raise FileNotFoundError(f"Missing chain file: {MM10_TO_MM39_CHAIN}\n"
                                f"Download via:\n  curl -L -o {MM10_TO_MM39_CHAIN} "
                                "http://hgdownload.soe.ucsc.edu/goldenPath/mm10/liftOver/mm10ToMm39.over.chain.gz")

    cmd = find_crossmap_cmd()
    total = sum(1 for _ in WORKLIST.open()) if WORKLIST.exists() else 0
    ok = 0
    fail = 0

    with WORKLIST.open() as fh:
        for i, line in enumerate(fh, 1):
            f = Path(line.strip())
            base = f.name
            stem = base.removesuffix(".gz")
            stem = stem[:-4] if stem.endswith(".bed") else stem  # remove .bed
            out = LIFT_OUTDIR / f"{stem}.lifted.bed"
            logf = LOG_DIR / f"{base}.mm10_to_mm39.log"

            # if gz, decompress to a temp file first
            in_path = f
            tmp = None
            if str(f).endswith(".gz"):
                tmp = LIFT_OUTDIR / (stem + ".__tmp__.bed")
                tmp.parent.mkdir(parents=True, exist_ok=True)
                run(["bash", "-lc", f"gzip -cd {f} > {tmp}"], check=True)
                in_path = tmp

            # Remove any previous output for a clean rerun
            if out.exists():
                out.unlink()

            logger.info(f"[{i}/{total}] CrossMap: {base} -> {out.name}")
            # CrossMap bed <chain> <in> <out>
            with logf.open("w") as lf:
                r = run(cmd + ["bed", str(MM10_TO_MM39_CHAIN), str(in_path), str(out)], stdout=lf, stderr=lf)

            if tmp is not None and tmp.exists():
                tmp.unlink(missing_ok=True)

            if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
                ok += 1
            else:
                logger.warning(f"  -> FAILED or empty output. See log: {logf}")
                # Clean empty file to avoid confusion
                if out.exists() and out.stat().st_size == 0:
                    out.unlink()
                fail += 1

    logger.info(f"Liftover finished. Success: {ok}  Fail: {fail}")


def write_updated_metadata():
    """Create CSV_OUT where mm10 rows are rewritten to mm39 and file_name points to lifted .bed."""
    if not CSV_IN.exists():
        raise FileNotFoundError(f"Missing input CSV: {CSV_IN}")

    rows = []
    with CSV_IN.open() as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        for row in reader:
            if (row.get("genome") or "").strip() == "mm10":
                base = os.path.basename(row["file_name"]).removesuffix(".gz")
                stem = base[:-4] if base.endswith(".bed") else base
                lifted = LIFT_OUTDIR / f"{stem}.lifted.bed"
                if lifted.exists():
                    row = dict(row)
                    row["genome"] = "mm39"
                    row["file_name"] = lifted.name
            rows.append(row)

    with CSV_OUT.open("w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    logger.info(f"Wrote updated metadata -> {CSV_OUT} (human unchanged, mouse now mm39 where lifted)")


def split_atac_layout(csv_path: Path):
    """Split into RASP-style CSVs under score_data/<kingdom>/metadata_<species>_atac.csv."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    # default to animals for human/mouse; keep general if other rows appear
    out_root = ATAC_PATH / "score_data"
    out_root.mkdir(exist_ok=True, parents=True)
    for (kingdom, species), g in df.groupby(["kingdom", "species"], dropna=False):
        out_dir = out_root / str(kingdom)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"metadata_{species}_atac.csv"
        g.to_csv(out_csv, index=False)
        logger.info(f"Wrote {len(g)} rows -> {out_csv}")


if __name__ == "__main__":
    logger.info(f"Working path: {ATAC_PATH}")
    ensure_dirs()

    # Resume/overwrite prompt (mirrors phyloP style, but only for the liftover folder)
    if LIFT_BASE.exists():
        logger.info(f"Folder {LIFT_BASE} already exists.")
        choice = input("Resume (r) / Overwrite liftover outputs (y) / Exit (n)? [r]: ").strip().lower()
        if choice == "y":
            logger.info(f"Deleting {LIFT_BASE}")
            shutil.rmtree(LIFT_BASE)
            ensure_dirs()
        elif choice in ("", "r"):
            logger.info("Resuming liftover/metadata update.")
        else:
            logger.info("Exiting.")
            raise SystemExit(0)
    else:
        ensure_dirs()

    # Pre-flight checks
    if not CSV_IN.exists():
        raise SystemExit(f"ERROR: {CSV_IN} not found. Run the downloader first.")
    if not PEAKS_DIR.exists():
        raise SystemExit(f"ERROR: {PEAKS_DIR} not found. Run the downloader first.")
    if not MM10_TO_MM39_CHAIN.exists():
        logger.info("Chain missing — downloading mm10→mm39 chain …")
        MM10_TO_MM39_CHAIN.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "http://hgdownload.soe.ucsc.edu/goldenPath/mm10/liftOver/mm10ToMm39.over.chain.gz",
            MM10_TO_MM39_CHAIN
        )


    build_mouse_worklist()
    liftover_all()
    write_updated_metadata()

    if DO_SPLIT:
        split_atac_layout(CSV_OUT)

    logger.info("Liftover + metadata update step completed.")
