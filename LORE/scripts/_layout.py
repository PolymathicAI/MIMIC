"""The published shard naming, in one place.

The internal build lays data out as ``<subset>/<split>/data_N.parquet``. The release is
flat within each subset and the filenames carry **no** information: ``data_0000.parquet``
and so on. The split lives only as a column on the index.

That is deliberate. A ``train/val/test`` directory tree, or even a ``train-`` filename
prefix, makes MIMIC 1.0's contaminated partition the thing a user reaches for first.
Renaming is not optional either way: ``data_0.parquet`` exists in all three splits, so
flattening without renumbering would collide.

Both the tree builder and the index rewriter must agree on the mapping exactly, or
``to_table`` will look up a path that does not exist. Rather than pass a mapping file
between them, both call :func:`shard_names`, which is a deterministic pure function of
the source directory listing.
"""

from __future__ import annotations

import re
from pathlib import Path

SPLITS = ("train", "val", "test")


def _numeric_key(name: str) -> tuple:
    """Sort ``data_2`` before ``data_10``, unlike lexical order."""
    m = re.search(r"(\d+)", name)
    return (int(m.group(1)) if m else -1, name)


def shard_names(subset_dir: Path) -> dict[tuple[str, str], str]:
    """Map ``(split, original filename)`` to the published filename for one subset.

    Numbering runs over **all** shards of the subset, in (train, val, test) order and
    then numerically within each split. Because the numbering is global to the subset,
    a partial tree (``--max-shards-per-split``, for a smoke test) gets a *subset* of the
    same names rather than a renumbering of its own -- so one published index describes
    both the full release and any sample of it.
    """
    out: dict[tuple[str, str], str] = {}
    i = 0
    for split in SPLITS:
        d = subset_dir / split
        if not d.is_dir():
            continue
        for f in sorted((p.name for p in d.glob("*.parquet")), key=_numeric_key):
            out[(split, f)] = f"data_{i:04d}.parquet"
            i += 1
    return out
