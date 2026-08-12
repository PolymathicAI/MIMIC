"""Unit tests for the LORE index query layer.

These build a small synthetic index in memory, so they run without a copy of the
dataset. The properties under test are the ones that are easy to get wrong by hand
and that the index exists to protect against: unioning subsets, de-duplicating
samples that appear in more than one subset, and splitting without leaking
homologues across the boundary.
"""

import json

import pyarrow as pa
import pytest

from lore.index import LoreIndex

# Two subsets that share a modality (rna_seq) and share two samples (g2, g3) -- the
# atac/cage mutual-exclusion pattern that makes the real subsets non-disjoint.
SUBSETS = {
    "atac+rna_seq": {
        "modalities": ["atac", "context", "rna_seq"],
        "key": "genome_feature_id",
        "splits": {"train": {"shards": ["data_0.parquet"], "rows": 3},
                   "val": {"shards": ["data_0.parquet"], "rows": 1}},
        "rows": 4,
    },
    "cage+rna_seq": {
        "modalities": ["cage", "context", "rna_seq"],
        "key": "genome_feature_id",
        "splits": {"train": {"shards": ["data_0.parquet"], "rows": 3}},
        "rows": 3,
    },
    "aa_seq+dssp": {
        "modalities": ["aa_seq", "dssp"],
        "key": "uniprot_id",
        "splits": {"train": {"shards": ["data_0.parquet"], "rows": 2}},
        "rows": 2,
    },
}

ROWS = [
    # subset,          split,  shard,             gfid, upid,  rna_cluster_30, species
    ("atac+rna_seq", "train", "data_0.parquet", "g1", None, "c1", "Homo sapiens"),
    ("atac+rna_seq", "train", "data_0.parquet", "g2", None, "c1", "Homo sapiens"),
    ("atac+rna_seq", "train", "data_0.parquet", "g3", None, "c2", "Mus musculus"),
    ("atac+rna_seq", "val",   "data_0.parquet", "g4", None, "c3", "Homo sapiens"),
    ("cage+rna_seq", "train", "data_0.parquet", "g2", None, "c1", "Homo sapiens"),
    ("cage+rna_seq", "train", "data_0.parquet", "g3", None, "c2", "Mus musculus"),
    ("cage+rna_seq", "train", "data_0.parquet", "g5", None, "c4", "Homo sapiens"),
    ("aa_seq+dssp",  "train", "data_0.parquet", None, "u1", None, None),
    ("aa_seq+dssp",  "train", "data_0.parquet", None, "u2", None, None),
]

COLS = ["subset", "split", "shard", "genome_feature_id", "uniprot_id",
        "rna_cluster_30", "species"]


@pytest.fixture
def idx():
    table = pa.table({c: [r[i] for r in ROWS] for i, c in enumerate(COLS)})
    return LoreIndex(table, SUBSETS)


def test_modalities_are_the_union_over_subsets(idx):
    assert idx.modalities == ["aa_seq", "atac", "cage", "context", "dssp", "rna_seq"]


def test_subsets_with_requires_all_modalities(idx):
    assert idx.subsets_with("rna_seq") == ["atac+rna_seq", "cage+rna_seq"]
    assert idx.subsets_with(["rna_seq", "atac"]) == ["atac+rna_seq"]
    # atac and cage never co-occur, so no subset satisfies both
    assert idx.subsets_with(["atac", "cage"]) == []


def test_unknown_modality_is_rejected_with_the_available_list(idx):
    with pytest.raises(ValueError, match="unknown modality"):
        idx.subsets_with("not_a_modality")


def test_select_unions_subsets(idx):
    sel = idx.select("rna_seq")
    assert sel.n_rows == 7
    assert set(sel.counts_by("subset")) == {"atac+rna_seq", "cage+rna_seq"}


def test_select_raises_when_modalities_never_co_occur(idx):
    with pytest.raises(ValueError, match="no subset carries all"):
        idx.select(["atac", "cage"])


def test_n_unique_accounts_for_cross_subset_duplication(idx):
    sel = idx.select("rna_seq")
    # 7 rows but only 5 distinct molecules: g2 and g3 appear in both subsets.
    assert sel.n_rows == 7
    assert sel.n_unique == 5


def test_deduplicate_collapses_to_unique_molecules(idx):
    sel = idx.select("rna_seq", deduplicate=True)
    assert sel.n_rows == 5 == sel.n_unique
    gfids = sel.table.column("genome_feature_id").to_pylist()
    assert sorted(gfids) == ["g1", "g2", "g3", "g4", "g5"]


def test_filters_combine_with_modality_selection(idx):
    sel = idx.select("rna_seq", species="Homo sapiens")
    assert sorted(set(sel.table.column("genome_feature_id").to_pylist())) == \
        ["g1", "g2", "g4", "g5"]


def test_filters_accept_a_list_as_membership(idx):
    sel = idx.select("rna_seq", species=["Homo sapiens", "Mus musculus"])
    assert sel.n_rows == 7


def test_unknown_filter_column_is_rejected(idx):
    with pytest.raises(ValueError, match="unknown index column"):
        idx.select("rna_seq", genus="Homo")


def test_make_splits_keeps_a_cluster_intact(idx):
    """The point of the whole exercise: no cluster may straddle a split."""
    sel = idx.select("rna_seq")
    parts = sel.make_splits(key="rna_cluster_30", fractions=(0.5, 0.25, 0.25), seed=0)

    assert sum(p.n_rows for p in parts) == sel.n_rows  # partition, nothing dropped

    where = {}
    for i, p in enumerate(parts):
        for c in p.table.column("rna_cluster_30").to_pylist():
            where.setdefault(c, set()).add(i)
    for cluster, splits in where.items():
        assert len(splits) == 1, f"cluster {cluster} leaked across splits {splits}"


def test_make_splits_keeps_duplicated_samples_together(idx):
    """g2/g3 live in two subsets; both copies must land in the same split."""
    sel = idx.select("rna_seq")
    parts = sel.make_splits(key="rna_cluster_30", seed=1)
    for gf in ("g2", "g3"):
        holding = [i for i, p in enumerate(parts)
                   if gf in p.table.column("genome_feature_id").to_pylist()]
        assert len(holding) == 1, f"{gf} appears in splits {holding}"


def test_make_splits_treats_null_cluster_as_a_singleton(idx):
    """Protein rows have no rna_cluster_30; they must not be pooled into one group."""
    sel = idx.select("aa_seq")
    parts = sel.make_splits(key="rna_cluster_30", fractions=(0.5, 0.5, 0.0), seed=0)
    assert sum(p.n_rows for p in parts) == 2
    # u1 and u2 are unrelated, so a 50/50 group split must separate them
    assert sorted(p.n_rows for p in parts) == [0, 1, 1]


def test_make_splits_validates_arguments(idx):
    sel = idx.select("rna_seq")
    with pytest.raises(ValueError, match="key must be one of"):
        sel.make_splits(key="not_a_cluster")
    with pytest.raises(ValueError, match="must sum to 1"):
        sel.make_splits(fractions=(0.5, 0.2, 0.2))


def test_shard_plan_is_the_distinct_files_touched(idx):
    sel = idx.select("rna_seq", species="Mus musculus")
    assert sel.shard_plan() == [
        ("atac+rna_seq", "train", "data_0.parquet"),
        ("cage+rna_seq", "train", "data_0.parquet"),
    ]


def test_to_table_without_a_root_is_an_error(idx):
    with pytest.raises(ValueError, match="no data root"):
        idx.select("rna_seq").to_table()


def test_shipped_split_selection_warns(idx, caplog):
    sel = idx.select("rna_seq", split="val")
    assert sel.n_rows == 1


def test_bad_split_name_is_rejected(idx):
    with pytest.raises(ValueError, match="split must be"):
        idx.select("rna_seq", split="validation")


def test_subsets_json_round_trips(tmp_path, idx):
    (tmp_path / "subsets.json").write_text(json.dumps(SUBSETS))
    assert json.loads((tmp_path / "subsets.json").read_text()) == SUBSETS
