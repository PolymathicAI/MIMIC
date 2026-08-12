"""LORE — the dataset MIMIC was trained on, plus the pipeline that built it.

Public entry points::

    from lore import LoreIndex
    idx = LoreIndex.from_hf()                    # or .from_parquet("lore_index")
    idx.describe()                               # what modalities exist, and where
    sel = idx.select(["rna_seq", "phylop_human"], species="Homo sapiens")
    train, val, test = sel.make_splits(key="rna_cluster_30", seed=0)

See ``lore.build_index`` to construct the index from a local copy of the data, and
``pipeline/`` for the download → process → select → merge code that produced it.
"""

from ._log import logger
from .index import CLUSTER_KEYS, LoreIndex, Selection

__all__ = ["LoreIndex", "Selection", "CLUSTER_KEYS", "logger"]
__version__ = "0.1.0"
