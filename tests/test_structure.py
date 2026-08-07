"""ESM3 structure tokenizer: structure input formats, detokenize, and round-trip.

Downloads the ESM3 VQVAE weights on first use, so the whole module skips cleanly when
biotite is missing or the weights cannot be fetched (e.g. offline CI). Runs on CPU.
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


def _atom(chain, res_id, res_name, atom_name, element, coord, hetero=False):
    return bs.Atom(
        coord, chain_id=chain, res_id=res_id, res_name=res_name, atom_name=atom_name,
        element=element, hetero=hetero, b_factor=10.0,
    )


@pytest.fixture(scope="module")
def messy_pdb(tmp_path_factory):
    """A PDB with everything a real file throws at us: two chains, waters, hydrogens."""
    import biotite.structure.io as strucio

    atoms = []
    for chain, n_res in (("A", 6), ("B", 3)):
        for res_id in range(1, n_res + 1):
            base = np.array([3.8 * res_id, 0.0, 0.0 if chain == "A" else 10.0])
            for name, element, offset in (
                ("N", "N", [0.0, 0.0, 0.0]), ("CA", "C", [1.4, 0.5, 0.0]),
                ("C", "C", [2.5, -0.4, 0.0]), ("O", "O", [2.3, -1.6, 0.0]),
            ):
                atoms.append(_atom(chain, res_id, "ALA", name, element, base + offset))
            # a hydrogen, which struct_to_numpy has no column for
            atoms.append(_atom(chain, res_id, "ALA", "HA", "H", base + [1.4, 1.6, 0.0]))
    for i in range(2):  # crystallographic waters
        atoms.append(_atom("A", 100 + i, "HOH", "O", "O", np.array([0.0, 8.0, i * 3.0]), hetero=True))

    path = tmp_path_factory.mktemp("struct") / "messy.pdb"
    strucio.save_structure(path, bs.array(atoms))
    return path


def test_load_structure_drops_waters_hydrogens_and_other_chains(messy_pdb):
    from mimic import load_structure

    arr = load_structure(messy_pdb)  # chain "A" by default
    assert set(np.unique(arr.chain_id)) == {"A"}
    assert set(np.unique(arr.res_name)) == {"ALA"}       # no HOH
    assert "H" not in set(np.unique(arr.element))         # no hydrogens
    assert arr.array_length() == 6 * 4                    # 6 residues x N/CA/C/O


def test_load_structure_selects_chain(messy_pdb):
    from mimic import load_structure

    arr = load_structure(messy_pdb, chain="B")
    assert set(np.unique(arr.chain_id)) == {"B"}
    assert arr.array_length() == 3 * 4


def test_missing_chain_raises_and_names_the_alternatives(messy_pdb):
    """The old behaviour was a silent (0, 7) array; an empty selection must be loud."""
    from mimic import load_structure

    with pytest.raises(ValueError, match=r"Chains in this structure: A, B"):
        load_structure(messy_pdb, chain="Z")


def test_missing_file_points_at_the_string_parser(tmp_path):
    from mimic import load_structure

    with pytest.raises(FileNotFoundError, match="parse_structure_string"):
        load_structure(tmp_path / "nope.pdb")


def test_parse_structure_string_matches_the_file(messy_pdb):
    from mimic import load_structure, parse_structure_string

    from_text = parse_structure_string(messy_pdb.read_text())
    assert np.allclose(from_text.coord, load_structure(messy_pdb).coord)


def test_pdb_to_numpy_gives_the_compact_format(messy_pdb):
    from mimic import pdb_to_numpy

    data = pdb_to_numpy(messy_pdb)
    assert data.shape == (6 * 4, 7) and data.dtype == np.float32
    assert np.isfinite(data).all()


def test_pdb_to_numpy_refuses_an_ambiguous_multimer(messy_pdb):
    from mimic import pdb_to_numpy

    with pytest.raises(ValueError, match="single chain"):
        pdb_to_numpy(messy_pdb, chain=None)


def test_numpy_to_struct_rejects_an_atom_array(messy_pdb):
    """The old failure mode here was a bare IndexError from inside biotite."""
    from mimic import load_structure, numpy_to_struct

    with pytest.raises(ValueError, match="load_structure"):
        numpy_to_struct(load_structure(messy_pdb))


def test_unknown_residues_survive_the_numpy_round_trip():
    """res_idx -1 is the UNK sentinel -- it must not come back as a VAL."""
    from mimic import numpy_to_struct, struct_to_numpy

    arr = bs.array([_atom("A", 1, "UNK", "CA", "C", np.zeros(3))])
    assert numpy_to_struct(struct_to_numpy(arr)).res_name[0] == "UNK"


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


def test_tokenize_accepts_file_atom_array_and_numpy(tokenizer, tokens, tmp_path):
    """A path, an AtomArray and the compact array must all encode identically."""
    import biotite.structure.io as strucio

    struct_np = tokenizer.detokenize(tokens)
    atom_array = tokenizer.detokenize(tokens, as_biotite=True)
    path = tmp_path / "decoded.pdb"
    strucio.save_structure(path, atom_array)

    from_numpy = tokenizer.tokenize(struct_np)
    assert tokenizer.tokenize(atom_array) == from_numpy
    assert tokenizer.tokenize(path) == from_numpy
    assert tokenizer.tokenize(str(path)) == from_numpy


def test_tokenize_rejects_unsupported_input(tokenizer):
    with pytest.raises(TypeError, match="Pass a path"):
        tokenizer.tokenize(42)


def test_tokenize_from_messy_pdb(tokenizer, messy_pdb):
    """Waters and hydrogens must not reach the encoder or change the token count."""
    assert len(tokenizer.tokenize(messy_pdb)) == 6
    assert len(tokenizer.tokenize(messy_pdb, chain="B")) == 3
