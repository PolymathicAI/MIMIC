"""Stage the LORE-index release: the index, filtered to match what actually ships.

The index is built over the whole dataset. Any subset omitted from the release (by
decision, like ``corpus``, or by an unresolved licence) must be dropped from the index
too, or the index will advertise rows that cannot be read.

Usage::

    python make_index_release.py --dest ~/LORE_index_upload
    python make_index_release.py --dest ~/tmp --exclude corpus --exclude cage
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from _layout import shard_names

DEFAULT_SRC = Path("/mnt/home/jkovalic/MIMIC/LORE/lore_index")
DEFAULT_DATA_ROOT = Path(
    "/mnt/ceph/users/polymathic/bio/central_dogma/data/final/nerv/4.0/parquet")
DEFAULT_EXCLUDE = ("corpus",)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                    help="source parquet root, to derive the published shard names")
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--exclude", action="append", default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--memory-limit", default="24GB")
    args = ap.parse_args()

    exclude = list(args.exclude if args.exclude is not None else DEFAULT_EXCLUDE)
    dest = args.dest.expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    src_index = args.src / "index.parquet"
    out_index = dest / "index.parquet"

    con = duckdb.connect()
    con.execute(f"SET threads={args.threads}; SET memory_limit='{args.memory_limit}'")

    before = pq.read_metadata(src_index).num_rows
    # Two rewrites, not a plain filter:
    #
    # * omitted subsets are dropped entirely -- the subset name is itself part of what
    #   we are withholding, so it must not survive in the index;
    # * `shard` is rewritten to the PUBLISHED filename. The release is flat within each
    #   subset with information-free names (`<subset>/data_0000.parquet`), so the index
    #   must name the file as published or `to_table` looks in the wrong place. The
    #   mapping comes from the same pure function the tree builder uses.
    #
    # `split` is preserved as a column: after this rename it is the ONLY record of the
    # MIMIC 1.0 partition anywhere in the release.
    #
    # The map is built only for subsets that ship, so the INNER JOIN below does the
    # exclusion too -- no WHERE needed, and no ambiguity over which `subset` is meant.
    rows = [(sub, split, old, new)
            for sub in sorted(d.name for d in args.data_root.iterdir() if d.is_dir())
            if sub not in set(exclude)
            for (split, old), new in shard_names(args.data_root / sub).items()]
    con.execute("CREATE TABLE shard_map (subset VARCHAR, split VARCHAR, "
                "old VARCHAR, new VARCHAR)")
    con.executemany("INSERT INTO shard_map VALUES (?, ?, ?, ?)", rows)

    con.execute(f"""
        COPY (
          SELECT i.* REPLACE (m.new AS shard)
          FROM read_parquet('{src_index}') i
          JOIN shard_map m
            ON i.subset = m.subset AND i.split = m.split AND i.shard = m.old
        ) TO '{out_index}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"shard map : {len(rows):,} (subset, split, shard) -> published name")
    after = pq.read_metadata(out_index).num_rows

    subsets = json.loads((args.src / "subsets.json").read_text())
    kept = {k: v for k, v in subsets.items() if k not in set(exclude)}
    (dest / "subsets.json").write_text(json.dumps(kept, indent=1))

    # Card is version-controlled in cards/; these trees are rebuilt from scratch.
    card = Path(__file__).resolve().parent.parent / "cards" / "LORE-index.md"
    link = dest / "README.md"
    if card.exists():
        link.unlink(missing_ok=True)
        link.symlink_to(card)
    else:
        print(f"WARNING: no card at {card}; repo will have no README")

    print(f"excluded: {exclude or 'nothing'}")
    print(f"index rows {before:,} -> {after:,}  (dropped {before - after:,})")
    print(f"subsets    {len(subsets)} -> {len(kept)}")
    print(f"size       {src_index.stat().st_size / 1e9:.2f} GB -> "
          f"{out_index.stat().st_size / 1e9:.2f} GB (zstd)")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
