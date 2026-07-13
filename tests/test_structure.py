"""ESM3 structure tokenizer: detokenize (tokens -> backbone structure) + round-trip.

Needs the ``[structure]`` extra (biotite) and downloads the ESM3 VQVAE weights on
first use, so the whole module skips cleanly when biotite is missing or the weights
cannot be fetched (e.g. offline CI). Runs on CPU.
"""
import numpy as np
import pytest

bs = pytest.importorskip("biotite.structure")


@pytest.fixture(scope="module")
def tokenizer():
    from mimic.tokenizers.protein.structure.struct_tokenizer import ESM3StructureTokenizer

    tok = ESM3StructureTokenizer(device="cpu")
    try:
        tok.init()  # downloads + loads the ESM3 encoder/decoder
    except Exception as e:  # network / weights unavailable
        pytest.skip(f"ESM3 structure weights unavailable ({type(e).__name__}): {e}")
    return tok


@pytest.fixture(scope="module")
def tokens():
    # Arbitrary valid codebook ids; the decoder maps any code sequence to a structure.
    rng = np.random.default_rng(0)
    return rng.integers(0, 4096, size=10).tolist()


def test_detokenize_numpy_default(tokenizer, tokens):
    out = tokenizer.detokenize(tokens)
    assert isinstance(out, np.ndarray)
    # compact struct_to_numpy format: (n_atoms, [x, y, z, res_id, res_idx, atom_idx, b])
    assert out.ndim == 2 and out.shape[1] == 7
    assert out.shape[0] > 0
    assert np.isfinite(out[:, :3]).all()  # coordinates are finite


def test_detokenize_biotite(tokenizer, tokens):
    out = tokenizer.detokenize(tokens, as_biotite=True)
    assert isinstance(out, bs.AtomArray)
    assert out.array_length() > 0
    assert out.coord.shape[1] == 3
    assert np.isfinite(out.coord).all()
    # every decoded residue is one chain of backbone atoms
    assert set(np.unique(out.chain_id)) == {"A"}


def test_numpy_and_biotite_agree(tokenizer, tokens):
    """The two output formats describe the same atoms."""
    np_out = tokenizer.detokenize(tokens)
    bt_out = tokenizer.detokenize(tokens, as_biotite=True)
    assert bt_out.array_length() == np_out.shape[0]
    assert np.allclose(bt_out.coord, np_out[:, :3])


def test_tokenize_detokenize_roundtrip(tokenizer, tokens):
    """Re-encoding a decoded structure yields the same number of per-residue tokens."""
    struct_np = tokenizer.detokenize(tokens)
    retok = tokenizer.tokenize(struct_np)
    assert isinstance(retok, list)
    assert len(retok) == len(tokens)
    assert all(0 <= t < tokenizer.vocab_size for t in retok)
