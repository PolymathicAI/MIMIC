"""Query the LORE dataset by modality, taxonomy or cluster — without knowing its layout.

LORE is stored as 41 *subsets*, each holding samples that share a modality group.
That is a training-time layout, and two properties of it make hand-rolled access
error-prone:

1. To find every sample carrying a modality group you must union several subsets.
2. **Subsets are not disjoint.** ``atac``/``cage``/``rasp2``/``prot_abund`` are mutually
   exclusive within a subset, so a transcript assayed for two of them is emitted once
   per assay, into different subsets. A naive union therefore double-counts samples,
   and a naive random split can put the same molecule in train and test.

This module handles both. :meth:`LoreIndex.select` unions the right subsets and reports
unique-sample counts alongside row counts; :meth:`LoreIndex.make_splits` groups by a
sequence-cluster label so duplicates — and homologues — cannot straddle a split.

    >>> idx = LoreIndex.from_parquet("lore_index")
    >>> sel = idx.select(modalities=["rna_seq", "splice_jctns_5cls"], species="Homo sapiens")
    >>> sel.n_rows, sel.n_unique
    (243155, 238904)
    >>> train, val, test = sel.make_splits(key="rna_cluster_30", seed=0)
    >>> table = sel.head(64).to_table()          # materialize payload for 64 rows

Note on ``split``: it records the partition MIMIC 1.0 was actually trained on, and it
is known to be contaminated — homologues of training sequences appear in val/test. It
lives **only as a column on this index**. The published data is flat within each subset
with information-free filenames (``<subset>/data_0000.parquet``), so it carries no trace
of the partition at all. That is deliberate: a train/val/test directory tree would make
the contaminated partition the default anyone gets from ``load_dataset``. Selecting it
is opt-in and warns. Use :meth:`make_splits` for evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ._log import logger

CLUSTER_KEYS = ("protein_cluster_30", "protein_cluster_70",
                "rna_cluster_30", "rna_cluster_70", "root_id")

TAXONOMY_KEYS = ("domain", "kingdom", "phylum", "class", "order", "family",
                 "genus", "species")


# Modalities stored as bin indices. Detokenizing these returns the CENTRE of each
# bin, not the measurement that was binned: the quantization is not invertible.
BINNED_MODALITIES = ("phylop_human", "phylop_mouse", "sasa", "rasp2", "atac", "cage",
                     "prot_abund", "masif_n_vertices", "masif_si_index",
                     "masif_charge", "masif_hbond", "masif_hydrophobicity")

# Detokenizes to 3-D coordinates rather than a scalar per position, so it does not fit
# a flat column and is skipped by `detokenize_table`. It also needs the ESM3 decoder.
NON_TABULAR_MODALITIES = ("prot_struct",)


def _id_column(table: pa.Table) -> pa.ChunkedArray:
    """Best available per-sample identity: the anchor id the row was keyed on."""
    gf = table.column("genome_feature_id")
    up = table.column("uniprot_id")
    return pc.coalesce(gf, up)


def detokenize_table(table: pa.Table, skip: tuple[str, ...] = NON_TABULAR_MODALITIES
                     ) -> pa.Table:
    """Convert token-id columns back to values, in place of shipping a second copy.

    Requires ``mimic`` (``pip install git+https://github.com/PolymathicAI/MIMIC.git``).
    A detokenized view is only ~7% larger than the tokenized one, so LORE ships
    tokenized only and converts on demand.

    Two caveats, both inherent to how the dataset was built rather than to this
    function:

    * The modalities in :data:`BINNED_MODALITIES` were discretized. They come back as
      bin centres — good enough to plot or to train on, but *not* the upstream
      measurements. Go to the original source (see ``NOTICE.md``) if you need those.
    * ``prot_struct`` decodes to 3-D coordinates via the ESM3 decoder, which does not
      fit a flat column; it is left as token ids. Use the ``mimic`` tokenizer directly.

    Columns that are not recognised modalities (ids, ``subset``, …) pass through.
    """
    try:
        from mimic.modality_info import MODALITY_INFO
    except ImportError as e:  # pragma: no cover - depends on optional install
        raise ImportError(
            "detokenize=True needs the `mimic` package: "
            "pip install git+https://github.com/PolymathicAI/MIMIC.git"
        ) from e

    out, lossy = {}, []
    for name in table.column_names:
        col = table.column(name)
        mod = name.removeprefix("tok_")
        info = MODALITY_INFO.get(f"tok_{mod}")
        if info is None or mod in skip:
            out[name] = col
            continue
        tk = info["tokenizer"]
        try:
            out[mod] = pa.array([tk.detokenize(v) for v in col.to_pylist()])
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{mod}: left as token ids ({type(exc).__name__}: {exc})")
            out[name] = col
            continue
        if mod in BINNED_MODALITIES:
            lossy.append(mod)

    if lossy:
        logger.warning(
            f"detokenized {', '.join(sorted(lossy))} to BIN CENTRES, not the original "
            f"measurements — these modalities were quantized when LORE was built"
        )
    return pa.table(out)


@dataclass
class Selection:
    """A set of index rows, plus the machinery to count, split and materialize them."""

    table: pa.Table
    root: Path | None
    subsets: dict

    # ---- counting -------------------------------------------------------------

    @property
    def n_rows(self) -> int:
        """Number of dataset rows selected (counts duplicated samples once each)."""
        return self.table.num_rows

    @property
    def n_unique(self) -> int:
        """Number of distinct molecules selected, de-duplicated across subsets."""
        return len(pc.unique(_id_column(self.table)))

    def counts_by(self, column: str) -> dict:
        """Row counts grouped by an index column (e.g. ``species``, ``subset``)."""
        agg = self.table.group_by(column).aggregate([(column, "count")])
        return dict(zip(agg.column(column).to_pylist(),
                        agg.column(f"{column}_count").to_pylist()))

    # ---- refinement -----------------------------------------------------------

    def filter(self, mask) -> "Selection":
        return Selection(self.table.filter(mask), self.root, self.subsets)

    def head(self, n: int) -> "Selection":
        return Selection(self.table.slice(0, n), self.root, self.subsets)

    def deduplicate(self) -> "Selection":
        """Keep one row per distinct molecule (first occurrence wins)."""
        ids = _id_column(self.table)
        seen: set = set()
        keep = []
        for i, v in enumerate(ids.to_pylist()):
            if v is not None and v in seen:
                keep.append(False)
            else:
                seen.add(v)
                keep.append(True)
        return Selection(self.table.filter(pa.array(keep)), self.root, self.subsets)

    # ---- splitting ------------------------------------------------------------

    def make_splits(self, key: str = "rna_cluster_30",
                    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
                    seed: int = 0) -> tuple["Selection", "Selection", "Selection"]:
        """Draw train/val/test splits that are leak-free with respect to ``key``.

        Every row sharing a ``key`` value lands in the same split, so neither
        duplicated samples nor sequence homologues can straddle the boundary. Rows
        with a null ``key`` are grouped by their own identifier, i.e. treated as
        singletons — they are never silently pooled together.

        Args:
            key: one of :data:`CLUSTER_KEYS`. ``rna_cluster_30`` / ``protein_cluster_30``
                are the conservative choices; the ``_70`` variants group less
                aggressively and admit more near-homology across splits.
            fractions: train/val/test proportions of *groups* (not rows).
            seed: seed for the group shuffle.
        """
        import random

        if key not in CLUSTER_KEYS:
            raise ValueError(f"key must be one of {CLUSTER_KEYS}, got {key!r}")
        if abs(sum(fractions) - 1.0) > 1e-6:
            raise ValueError(f"fractions must sum to 1, got {fractions}")

        group = pc.coalesce(self.table.column(key), _id_column(self.table)).to_pylist()
        groups = sorted({g for g in group if g is not None})
        rng = random.Random(seed)
        rng.shuffle(groups)

        n_tr = int(len(groups) * fractions[0])
        n_va = int(len(groups) * (fractions[0] + fractions[1]))
        assign = {}
        for i, g in enumerate(groups):
            assign[g] = 0 if i < n_tr else 1 if i < n_va else 2

        masks = ([], [], [])
        for g in group:
            a = assign.get(g, 0)
            for j in range(3):
                masks[j].append(a == j)

        out = tuple(Selection(self.table.filter(pa.array(m)), self.root, self.subsets)
                    for m in masks)
        logger.info(
            f"split on {key}: {len(groups):,} groups -> "
            f"train {out[0].n_rows:,} / val {out[1].n_rows:,} / test {out[2].n_rows:,} rows"
        )
        return out

    # ---- materialization ------------------------------------------------------

    def shard_plan(self) -> list[tuple[str, str, str]]:
        """The distinct ``(subset, split, shard)`` files this selection touches."""
        cols = ["subset", "split", "shard"]
        agg = self.table.select(cols).group_by(cols).aggregate([])
        return sorted(zip(*(agg.column(c).to_pylist() for c in cols)))

    def to_table(self, columns: list[str] | None = None,
                 root: Path | None = None, detokenize: bool = False) -> pa.Table:
        """Read the payload rows for this selection.

        Reads only the shards in :meth:`shard_plan`, then keeps the selected rows.
        ``columns`` restricts which modalities are read; the anchor ids are always
        included so the result can be joined back to the index.

        Args:
            columns: modalities to read. Default reads everything in the shard.
            root: data root, if not set on the index.
            detokenize: convert token ids back to values via the ``mimic``
                tokenizers. See :meth:`detokenized` for the caveats — in particular
                that the binned assay modalities come back as bin centres.

        Beware: LORE rows carry full-length sequences. Slice the selection down
        (:meth:`head`) before materializing anything large.
        """
        root = root or self.root
        if root is None:
            raise ValueError("no data root; pass root= or construct with from_parquet(root=...)")

        wanted = set(_id_column(self.table).to_pylist())
        parts = []
        for subset, _split, shard in self.shard_plan():
            # Published layout is flat within a subset: `<subset>/data_0000.parquet`.
            # The filenames are information-free, so the split cannot become anyone's
            # default; the index's `split` column is the record. See the module docstring.
            path = Path(root) / subset / shard
            cols = None
            if columns is not None:
                have = {f.name for f in pq.ParquetFile(path).schema_arrow}
                # Modality names are normalized (`tok_` stripped); map back to the
                # on-disk column via the manifest, falling back to the name itself.
                lookup = self.subsets.get(subset, {}).get("columns", {})
                cols = [c for c in ([lookup.get(m, m) for m in columns]
                                    + ["uniprot_id", "genome_feature_id", "__key__"])
                        if c in have]
            t = pq.read_table(path, columns=cols)
            keep = pc.is_in(_id_column(t), value_set=pa.array(sorted(wanted)))
            t = t.filter(keep)
            if t.num_rows:
                parts.append(t.append_column(
                    "subset", pa.array([subset] * t.num_rows)))
        if not parts:
            return pa.table({})
        table = pa.concat_tables(parts, promote_options="permissive")
        return detokenize_table(table) if detokenize else table

    def __repr__(self) -> str:
        return (f"<Selection rows={self.n_rows:,} unique={self.n_unique:,} "
                f"subsets={len(self.counts_by('subset'))}>")


class LoreIndex:
    """The LORE index: locations, identifiers, clusters and taxonomy for every row."""

    def __init__(self, table: pa.Table, subsets: dict, root: Path | None = None):
        self.table = table
        self.subsets = subsets
        self.root = Path(root) if root else None

    # ---- construction ---------------------------------------------------------

    @classmethod
    def from_parquet(cls, index_dir: str | Path, root: str | Path | None = None) -> "LoreIndex":
        """Load from a directory produced by ``python -m lore.build_index``."""
        d = Path(index_dir)
        subsets = json.loads((d / "subsets.json").read_text())
        table = pq.read_table(d / "index.parquet")
        logger.info(f"loaded index: {table.num_rows:,} rows, {len(subsets)} subsets")
        return cls(table, subsets, root)

    @classmethod
    def from_hf(cls, repo_id: str = "polymathic-ai/LORE-index",
                revision: str | None = None, root: str | Path | None = None) -> "LoreIndex":
        """Download the published index from the Hugging Face Hub."""
        from huggingface_hub import snapshot_download

        d = snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision)
        return cls.from_parquet(d, root=root)

    # ---- introspection --------------------------------------------------------

    @property
    def modalities(self) -> list[str]:
        """Every modality present anywhere in the dataset."""
        return sorted({m for s in self.subsets.values() for m in s["modalities"]})

    def subsets_with(self, modalities: list[str] | str) -> list[str]:
        """Subsets carrying *all* of ``modalities``."""
        want = {modalities} if isinstance(modalities, str) else set(modalities)
        unknown = want - set(self.modalities)
        if unknown:
            raise ValueError(
                f"unknown modality/ies {sorted(unknown)}. Available: {self.modalities}")
        return sorted(n for n, s in self.subsets.items()
                      if want <= set(s["modalities"]))

    def describe(self) -> None:
        """Print a per-modality summary of where the data lives.

        Row counts come from the manifest and so describe the whole dataset, which may
        exceed what this index covers if it was built for a subset of the data.
        """
        total = sum(s["rows"] for s in self.subsets.values())
        print(f"{len(self.subsets)} subsets, {len(self.modalities)} modalities, "
              f"{total:,} rows in the dataset")
        if self.table.num_rows != total:
            print(f"(this index covers {self.table.num_rows:,} of them)")
        print(f"\n{'modality':<22}{'subsets':>8}{'rows':>14}")
        for m in self.modalities:
            subs = self.subsets_with(m)
            rows = sum(self.subsets[s]["rows"] for s in subs)
            print(f"{m:<22}{len(subs):>8}{rows:>14,}")

    # ---- querying -------------------------------------------------------------

    def select(self, modalities: list[str] | str | None = None,
               split: str | None = None,
               deduplicate: bool = False,
               **filters) -> Selection:
        """Select rows carrying all of ``modalities`` and matching ``filters``.

        Args:
            modalities: modality or modalities every returned row must carry.
            split: restrict to the shipped ``train``/``val``/``test`` split. Omit this
                for evaluation work and use :meth:`Selection.make_splits` instead —
                the shipped splits are contaminated (see module docstring).
            deduplicate: collapse molecules that appear in more than one subset.
            **filters: equality (or membership, for a list) on any index column,
                e.g. ``species="Homo sapiens"``, ``kingdom=["Metazoa", "Fungi"]``,
                ``is_coding=True``.

        Returns:
            A :class:`Selection`.
        """
        mask = None

        def combine(m):
            nonlocal mask
            mask = m if mask is None else pc.and_(mask, m)

        if modalities is not None:
            subs = self.subsets_with(modalities)
            if not subs:
                raise ValueError(
                    f"no subset carries all of {modalities} simultaneously. "
                    f"LORE is sparse: not every modality co-occurs. Query them "
                    f"separately, or check `subsets_with` for narrower groups.")
            logger.info(f"{modalities} -> {len(subs)} subset(s)")
            combine(pc.is_in(self.table.column("subset"), value_set=pa.array(subs)))

        if split is not None:
            if split not in ("train", "val", "test"):
                raise ValueError(f"split must be train/val/test, got {split!r}")
            logger.warning(
                "the shipped split column reproduces MIMIC 1.0's training split and is "
                "known-contaminated; prefer Selection.make_splits() for evaluation")
            combine(pc.equal(self.table.column("split"), split))

        for col, val in filters.items():
            if col not in self.table.column_names:
                raise ValueError(
                    f"unknown index column {col!r}. Available: {self.table.column_names}")
            if isinstance(val, (list, tuple, set)):
                combine(pc.is_in(self.table.column(col), value_set=pa.array(sorted(val))))
            else:
                combine(pc.equal(self.table.column(col), val))

        table = self.table if mask is None else self.table.filter(mask)
        sel = Selection(table, self.root, self.subsets)
        return sel.deduplicate() if deduplicate else sel

    def __repr__(self) -> str:
        return (f"<LoreIndex rows={self.table.num_rows:,} "
                f"subsets={len(self.subsets)} modalities={len(self.modalities)}>")
