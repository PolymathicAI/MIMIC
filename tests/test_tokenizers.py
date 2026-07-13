"""Tokenizer coverage — the checks that used to run at import now run here.

Importing `mimic` no longer executes the per-modality self-tests (they are guarded
under `if __name__ == "__main__":`); this suite exercises the tokenizers instead.
"""
import numpy as np
import pytest

from mimic.modality_info import MODALITY_INFO

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
