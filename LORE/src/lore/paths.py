"""Path resolution for the LORE data tree.

Replaces the private `central_dogma.config.paths` module the pipeline was written
against. The layout is unchanged, so scripts that called
`paths.get_path(data_type=..., stage=..., name=..., version=...)` keep working;
only the root now comes from `$LORE_DATA_ROOT` instead of internal env-files.

Layout under the root::

    <root>/data/downloads/<name>/<version>/          raw third-party downloads
    <root>/data/intermediate/<name>/<version>/       per-source working files
    <root>/data/modality/<name>/<version>/           one directory per modality
    <root>/data/final/<name>/<version>/{parquet,wds} merged, tokenized dataset

The released dataset is ``data/final/lore/4.0/parquet``.
"""

from __future__ import annotations

import os
from pathlib import Path

from ._log import logger

# Stage names are part of the on-disk layout; keep them stable.
STAGES = ("downloads", "intermediate", "modality", "merged", "final", "downstream")
DATA_TYPES = ("data", "model", "eval")


def get_data_root_path() -> Path:
    """Root of the LORE data tree, from ``$LORE_DATA_ROOT``.

    Raises:
        RuntimeError: if the variable is unset. There is deliberately no default:
            these trees are terabyte-scale and must be sited explicitly.
    """
    root = os.environ.get("LORE_DATA_ROOT")
    if not root:
        raise RuntimeError(
            "LORE_DATA_ROOT is not set. Point it at the root of your LORE data tree, "
            "e.g. `export LORE_DATA_ROOT=/scratch/lore`. The released dataset then "
            "lives at $LORE_DATA_ROOT/data/final/lore/4.0/parquet."
        )
    return Path(root)


def get_path(
    data_type: str = "data",
    stage: str | None = None,
    name: str | None = None,
    version: str | None = None,
    fmt: str | None = None,
    *,
    mkdir: bool = False,
) -> Path:
    """Resolve a path in the LORE tree.

    Args:
        data_type: one of ``DATA_TYPES``.
        stage: one of ``STAGES``.
        name: source or dataset name (e.g. ``refseq``, ``lore``).
        version: version string (e.g. ``1.4``, ``4.0``).
        fmt: optional trailing format directory (e.g. ``parquet``, ``wds``).
        mkdir: create the directory (and parents) if missing.
    """
    if data_type not in DATA_TYPES:
        raise ValueError(f"data_type must be one of {DATA_TYPES}, got {data_type!r}")
    if stage is not None and stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")

    path = get_data_root_path() / data_type
    for part in (stage, name, version, fmt):
        if part is not None:
            path = path / str(part)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"ensured {path}")
    return path


def final_parquet_root(version: str = "4.0") -> Path:
    """Directory holding the released per-subset parquet shards."""
    return get_path("data", "final", "lore", version, "parquet")
