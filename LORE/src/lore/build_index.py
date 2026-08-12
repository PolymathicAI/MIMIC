"""Build the LORE index: one small table describing every row in the dataset.

The released parquet is physically laid out as one directory per *subset* — a set of
samples that all carry the same group of modalities. That layout exists because no
sample carries all 26 modalities, and training wants dense, null-free batches. It is
a poor interface for analysis: to answer "give me every sample with rna_seq and
splice_jctns_5cls" you would have to work out which of the 41 subsets contain both,
and then deduplicate, because **subsets are not disjoint** (a transcript assayed under
two conditions is emitted once per condition, into different subsets).

This script collapses that into two artifacts:

* ``subsets.json``  — subset -> modality set, splits, shard list, row counts.
* ``index.parquet`` — one row per sample row, carrying its location (subset / split /
  shard), its identifiers, all four sequence-cluster labels, taxonomy, and lengths.
  No sequence payload, so it is ~1 GB against 447 GB of data.

With the index in hand, :mod:`lore.index` answers modality/taxonomy queries and draws
leak-free splits without touching the payload.

Usage::

    python -m lore.build_index --out ./lore_index
    python -m lore.build_index --out ./lore_index --subset rna_codons --subset corpus

Phase A (per-subset id scan) is resumable: existing parts are skipped unless
``--overwrite``. Phase B joins the parts against the master-id table to attach
clusters and taxonomy; pass ``--master-ids`` to point at ``dataset.parquet``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from ._log import logger
from .paths import final_parquet_root

# Columns that identify a row rather than carrying modality payload.
ID_COLUMNS = ("__key__", "uniprot_id", "genome_feature_id")

# Attached from the master-id table in phase B.
MASTER_COLUMNS = (
    "root_id",
    "protein_cluster_30",
    "protein_cluster_70",
    "rna_cluster_30",
    "rna_cluster_70",
    "domain",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
    "is_coding",
    "feature_type",
    "aa_seq_length",
    "rna_seq_length",
)

SPLITS = ("train", "val", "test")


def describe_subsets(root: Path) -> dict:
    """Map each subset directory to its modalities, splits and shards.

    Modalities are read from the parquet *columns*, not the directory name: the
    directory name omits ``context`` (present whenever a condition-conditional assay
    is included) and the eval-only subsets are not named after their modalities.

    Column names are normalized by stripping a leading ``tok_``. Payload columns are
    otherwise unprefixed, but the ``corpus`` subset ships its column as ``tok_corpus``;
    normalizing means callers ask for ``corpus`` like every other modality. Each
    subset records a ``columns`` map from modality name back to the on-disk column so
    reads still use the real name.
    """
    out = {}
    for sub in sorted(d for d in root.iterdir() if d.is_dir()):
        splits = {}
        columns: dict[str, str] = {}
        for split in SPLITS:
            files = sorted((sub / split).glob("*.parquet"))
            if not files:
                continue
            schema = pq.ParquetFile(files[0]).schema_arrow
            for f in schema:
                if f.name not in ID_COLUMNS:
                    columns[f.name.removeprefix("tok_")] = f.name
            splits[split] = {
                "shards": [f.name for f in files],
                "rows": sum(pq.read_metadata(f).num_rows for f in files),
            }
        if not splits:
            logger.warning(f"{sub.name}: no parquet found, skipping")
            continue
        probe = next(iter(splits))
        cols = {f.name for f in pq.ParquetFile(sub / probe / splits[probe]["shards"][0]).schema_arrow}
        out[sub.name] = {
            "modalities": sorted(columns),
            "columns": columns,
            "key": ("genome_feature_id" if "genome_feature_id" in cols
                    else "uniprot_id" if "uniprot_id" in cols else None),
            "splits": splits,
            "rows": sum(v["rows"] for v in splits.values()),
        }
        logger.info(f"{sub.name}: {len(columns)} modalities, "
                    f"{out[sub.name]['rows']:,} rows, key={out[sub.name]['key']}")
    return out


def scan_subset(con: duckdb.DuckDBPyConnection, root: Path, subset: str,
                info: dict, dest: Path) -> None:
    """Phase A: write (subset, split, shard, ids) for one subset."""
    key = info["key"]
    selects = []
    for split in info["splits"]:
        glob = str(root / subset / split / "*.parquet")
        # Only the id column is read; the list<> payload columns are never touched.
        # The cast matters: `corpus` has no id column at all, and an untyped NULL is
        # inferred as INTEGER, which then fails to compare against the VARCHAR ids
        # of every other subset when the parts are unioned in phase B.
        cols = (f"CAST({key} AS VARCHAR) AS row_key" if key
                else "CAST(NULL AS VARCHAR) AS row_key")
        selects.append(
            f"SELECT '{subset}' AS subset, '{split}' AS split, "
            f"regexp_extract(filename, '[^/]+$') AS shard, "
            f"{cols}, '{key}' AS key_kind "
            f"FROM read_parquet('{glob}', filename=true)"
        )
    con.execute(f"COPY ({' UNION ALL '.join(selects)}) TO '{dest}' (FORMAT PARQUET)")
    logger.info(f"  -> {dest.name}")


def join_master(con: duckdb.DuckDBPyConnection, parts: Path, master_ids: Path | None,
                out: Path) -> None:
    """Phase B: attach clusters/taxonomy/lengths from the master-id table."""
    parts_glob = str(parts / "*.parquet")
    if master_ids is None:
        logger.warning("no --master-ids given; index will carry location + ids only")
        con.execute(
            f"COPY (SELECT subset, split, shard, key_kind, "
            f"CASE WHEN key_kind='uniprot_id' THEN row_key END AS uniprot_id, "
            f"CASE WHEN key_kind='genome_feature_id' THEN row_key END AS genome_feature_id "
            f"FROM read_parquet('{parts_glob}')) TO '{out}' (FORMAT PARQUET)"
        )
        return

    # Several master columns (`order`, `class`, `domain`, `family`) are reserved SQL
    # words, so every identifier is quoted.
    q = [f'"{c}"' for c in MASTER_COLUMNS]
    mcols = ", ".join(q)
    # The master table has one row per (uniprot_id, genome_feature_id) pair; the
    # released rows are one representative per cluster, so this is a many-to-one
    # lookup on whichever key the subset is anchored on. DISTINCT ON guards against
    # a representative appearing under several pairings.
    #
    # The master table is ~319M rows while the released data is ~60M, so both
    # lookups are semi-joined down to the ids actually present before aggregating.
    # Without that pre-filter the DISTINCT ON scans the whole table and needs far
    # more memory than --memory-limit typically allows.
    con.execute(f"""
        COPY (
          WITH r AS (
            SELECT subset, split, shard, key_kind,
                   CASE WHEN key_kind = 'uniprot_id' THEN row_key END AS uniprot_id,
                   CASE WHEN key_kind = 'genome_feature_id' THEN row_key END AS genome_feature_id
            FROM read_parquet('{parts_glob}')
          ),
          ids_g AS (SELECT DISTINCT genome_feature_id FROM r WHERE genome_feature_id IS NOT NULL),
          ids_u AS (SELECT DISTINCT uniprot_id FROM r WHERE uniprot_id IS NOT NULL),
          mg AS (
            SELECT DISTINCT ON (genome_feature_id) genome_feature_id, {mcols}
            FROM read_parquet('{master_ids}')
            WHERE genome_feature_id IN (SELECT genome_feature_id FROM ids_g)
          ),
          mu AS (
            SELECT DISTINCT ON (uniprot_id) uniprot_id, {mcols}
            FROM read_parquet('{master_ids}')
            WHERE uniprot_id IN (SELECT uniprot_id FROM ids_u)
          )
          SELECT r.subset, r.split, r.shard, r.key_kind,
                 r.uniprot_id, r.genome_feature_id,
                 {', '.join(f'COALESCE(mg."{c}", mu."{c}") AS "{c}"' for c in MASTER_COLUMNS)}
          FROM r
          LEFT JOIN mg ON r.genome_feature_id = mg.genome_feature_id
          LEFT JOIN mu ON r.uniprot_id = mu.uniprot_id
        ) TO '{out}' (FORMAT PARQUET)
    """)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None,
                    help="parquet root (default: $LORE_DATA_ROOT/data/final/lore/4.0/parquet)")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--master-ids", type=Path, default=None,
                    help="master-id dataset.parquet, for cluster/taxonomy columns")
    ap.add_argument("--subset", action="append", default=None,
                    help="restrict to these subsets (repeatable); default all")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--memory-limit", default="16GB")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-join", action="store_true", help="phase A only")
    args = ap.parse_args()

    root = args.root or final_parquet_root()
    if not root.is_dir():
        raise SystemExit(f"parquet root not found: {root}")
    out = args.out
    parts = out / "index_parts"
    parts.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET threads={args.threads}; SET memory_limit='{args.memory_limit}'")

    logger.info(f"describing subsets under {root}")
    subsets = describe_subsets(root)

    # The manifest always describes the whole dataset, so that `subsets_with()` can
    # answer questions about modality coverage even from a partial build. --subset
    # narrows only which parts get scanned.
    (out / "subsets.json").write_text(json.dumps(subsets, indent=1))
    logger.info(f"wrote {out / 'subsets.json'} ({len(subsets)} subsets)")

    scan = subsets
    if args.subset:
        missing = set(args.subset) - set(subsets)
        if missing:
            raise SystemExit(f"unknown subset(s): {sorted(missing)}")
        scan = {k: v for k, v in subsets.items() if k in args.subset}
        logger.info(f"restricting scan to {len(scan)} subset(s)")

    logger.info("phase A: scanning identifiers")
    for name, info in scan.items():
        dest = parts / f"{name}.parquet"
        if dest.exists() and not args.overwrite:
            logger.info(f"  {name}: part exists, skipping")
            continue
        scan_subset(con, root, name, info, dest)

    if args.skip_join:
        logger.info("phase B skipped (--skip-join)")
        return

    logger.info("phase B: joining master-id metadata")
    join_master(con, parts, args.master_ids, out / "index.parquet")
    n = pq.read_metadata(out / "index.parquet").num_rows
    logger.info(f"wrote {out / 'index.parquet'} ({n:,} rows)")


if __name__ == "__main__":
    main()
