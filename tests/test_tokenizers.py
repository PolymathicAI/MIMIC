"""Tokenizer coverage — the checks that used to run at import now run here.

Importing `mimic` no longer executes the per-modality self-tests (they are guarded
under `if __name__ == "__main__":`); this suite exercises the tokenizers instead.
"""
import numpy as np
import pytest

from mimic.modality_info import MODALITY_INFO
from mimic.tokenizers.base import (
    BoolTokenizer,
    CharLevelTokenizer,
    ClassTokenizer,
    DigitizeTokenizer,
)

TOKENIZERS = {m: info["tokenizer"] for m, info in MODALITY_INFO.items()
              if "tokenizer" in info}


def test_import_is_side_effect_free():
    # If a self-test still ran (and failed) at import, importing mimic would have
    # raised before we got here. Reaching this point means import is clean.
    import mimic  # noqa: F401
    assert TOKENIZERS, "no tokenizers registered"


@pytest.mark.parametrize("mod", sorted(TOKENIZERS))
def test_tokenizer_api(mod):
    tk = TOKENIZERS[mod]
    assert hasattr(tk, "tokenize") and callable(tk.tokenize)
    assert hasattr(tk, "detokenize") and callable(tk.detokenize)
    assert hasattr(tk, "mask_token_id")


@pytest.mark.parametrize("mod,seq", [
    ("tok_rna_seq", "ACGUACGUACGUACGU"),
    ("tok_aa_seq", "MKTAYIAKQRQISFVK"),
])
def test_char_roundtrip_exact(mod, seq):
    tk = TOKENIZERS[mod]
    ids = tk.tokenize(seq)
    assert len(ids) == len(seq)
    # detokenize -> re-tokenize must be a fixed point
    reids = [int(x) for x in tk.tokenize(tk.detokenize([int(x) for x in ids]))]
    assert reids == [int(x) for x in ids]


# Bin-based tokenizers expose `.bins`; a uniform sample in range must round-trip
# to within one bin width.
BIN_MODS = sorted(m for m, tk in TOKENIZERS.items() if hasattr(tk, "bins"))


@pytest.mark.parametrize("mod", BIN_MODS)
def test_bin_roundtrip_within_tolerance(mod):
    tk = TOKENIZERS[mod]
    bins = list(tk.bins)
    rng = np.random.default_rng(0)
    seq = rng.uniform(min(bins), max(bins), size=200)
    ids = tk.tokenize(seq)
    decoded = np.asarray(tk.detokenize(ids), dtype=float)
    max_err = 1.01 * np.diff(bins)[np.asarray(ids)]
    assert np.all(np.abs(seq - decoded) <= max_err)


# --- exact self-tests ported from tokenizers/base.py's former import-time block ---

def test_char_tokenizer_unk_mapping():
    tk = CharLevelTokenizer(vocab_list=["A", "C", "G", "T"], unk_token="X", version=None)
    assert tk.detokenize(tk.tokenize("ACGTO")) == "ACGTX"


def test_digitize_tokenizer_nan_and_values():
    tk = DigitizeTokenizer(bins=[0, 1, 2, 3, 4], version=None)
    out = tk.detokenize(tk.tokenize([0.5, 1.5, 2.5, 3.5, 4.5, np.nan]))
    assert np.isnan(out[-1]) and np.isnan(out[-2])
    assert (np.array(out[:-2]) - np.array([0.5, 1.5, 2.5, 3.5])).max() < 0.01


def test_digitize_weighted_mean_detokenize():
    # bins [0,1,2,3,4] -> centers 0.5, 1.5, 2.5, 3.5
    tk = DigitizeTokenizer(bins=[0, 1, 2, 3, 4], version=None)
    tokens_w = [0, 1, 2]  # dummy tokens (ignored when probs given)
    probs_w = np.zeros((3, tk.vocab_size))
    probs_w[0, 0], probs_w[0, 1] = 0.5, 0.5  # expect 0.5*0.5 + 0.5*1.5 = 1.0
    probs_w[1, 2] = 1.0  # one-hot -> 2.5
    probs_w[2, 1], probs_w[2, 2] = 0.25, 0.75  # expect 0.25*1.5 + 0.75*2.5 = 2.25
    expected_0 = 0.5 * tk.id_to_token[0] + 0.5 * tk.id_to_token[1]
    expected_1 = tk.id_to_token[2]
    expected_2 = 0.25 * tk.id_to_token[1] + 0.75 * tk.id_to_token[2]

    out_w = tk.detokenize(tokens_w, probs=probs_w)
    assert len(out_w) == 3
    assert abs(out_w[0] - expected_0) < 1e-9 and abs(out_w[1] - expected_1) < 1e-9 and abs(out_w[2] - expected_2) < 1e-9
    # same result when probs has shape (len(x), vocab_size - 3)
    probs_bins_only = probs_w[:, : tk.vocab_size - 3]
    out_w_short = tk.detokenize(tokens_w, probs=probs_bins_only)
    assert abs(out_w_short[0] - expected_0) < 1e-9 and abs(out_w_short[1] - expected_1) < 1e-9 and abs(out_w_short[2] - expected_2) < 1e-9
    # x optional when probs given: same result with probs only
    out_w_no_x = tk.detokenize(probs=probs_w)
    assert abs(out_w_no_x[0] - expected_0) < 1e-9 and abs(out_w_no_x[1] - expected_1) < 1e-9 and abs(out_w_no_x[2] - expected_2) < 1e-9


def test_bool_tokenizer():
    tk = BoolTokenizer(version=None)
    assert tk.detokenize(tk.tokenize([True, False, True, False, 5])) == [1, 0, 1, 0, 4]


def test_class_tokenizer():
    tk = ClassTokenizer(vocab_list=["hi", "bye"], unk_token="Other", version=None)
    assert tk.detokenize(tk.tokenize(["hi", "bye", "hello"])) == ["hi", "bye", "Other"]
