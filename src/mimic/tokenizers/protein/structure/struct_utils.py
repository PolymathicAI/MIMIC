"""Structure I/O for the protein-structure tokenizer.

Everything needed to turn a structure a user actually has -- a PDB / mmCIF file, a
biotite ``AtomArray``, or the compact ``(n_atoms, 7)`` array MIMIC stores -- into the
form the ESM3 VQVAE encoder consumes. ``load_structure`` is the usual entry point;
``ESM3StructureTokenizer.tokenize`` accepts any of those forms directly.
"""
import io
import os

import numpy as np
import biotite.structure as bs
import biotite.structure.io as strucio
from loguru import logger

RES_TYPES = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
RES_INDEX = {res_type: i for i, res_type in enumerate(RES_TYPES)}
RES_INDEX["UNK"] = -1

ATOM_TYPES = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD",
    "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3",
    "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2",
    "CZ3", "NZ", "OXT",
]
ATOM_INDEX = {atom_type: i for i, atom_type in enumerate(ATOM_TYPES)}

_NONSTANDARD_RESIDUES = {
    "HEM": "HIS",
    "HSD": "HIS",
    "HSE": "HIS",
    "HSP": "HIS",
    "MSE": "MET",
    "PTR": "TYR",
    "NEC": "CYS",
    "CSO": "CYS",
}


def sanitize_nonstandard_residues(res_name: str, strict: bool = True):
    if res_name in RES_INDEX:  # standard residues + "UNK"
        return res_name
    if res_name in _NONSTANDARD_RESIDUES:
        return _NONSTANDARD_RESIDUES[res_name]
    if not strict:
        return "UNK"
    raise ValueError(
        f"Non-standard residue {res_name} not found in the conversion table. "
        "Please add it with an appropriate standard-residue mapping, or pass "
        "strict=False to record it as UNK."
    )


def struct_to_numpy(atom_array: "bs.AtomArray", chain="A", strict: bool = True):
    """Compress a biotite AtomArray into an (n_atoms, 7) float32 array.

    Columns are ``[x, y, z, res_id, res_idx, atom_idx, b_factor]``. The array must
    already be cleaned (see :func:`clean_structure`): every residue name has to be
    resolvable to one of the standard 20 unless ``strict=False``, and every atom name
    has to be one of :data:`ATOM_TYPES`.
    """
    atom_array = atom_array[atom_array.chain_id == chain]
    n_atoms = atom_array.coord.shape[0]
    if n_atoms == 0:
        raise ValueError(
            f"Chain {chain!r} contains no atoms. Pass chain=<id> to select a chain "
            "that exists in this structure."
        )
    data = np.zeros((n_atoms, 7), dtype=np.float32)
    data[:, :3] = atom_array.coord
    data[:, 3] = atom_array.res_id
    data[:, 4] = [
        RES_INDEX[sanitize_nonstandard_residues(res, strict=strict)]
        for res in atom_array.res_name
    ]
    data[:, 5] = [ATOM_INDEX[atom] for atom in atom_array.atom_name]
    if hasattr(atom_array, "b_factor"):
        data[:, 6] = atom_array.b_factor
    return data


def numpy_to_struct(data: np.ndarray, chain="A"):
    """Uncompress an (n_atoms, 7) array back into a biotite AtomArray."""
    if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] != 7:
        raise ValueError(
            "Expected the compact (n_atoms, 7) structure array from struct_to_numpy, "
            f"got {type(data).__name__}"
            + (f" of shape {data.shape}" if isinstance(data, np.ndarray) else "")
            + ". Load a PDB/mmCIF file with load_structure() first."
        )
    n_atoms = data.shape[0]
    atom_array = bs.AtomArray(n_atoms)
    atom_array.coord = data[:, :3]
    atom_array.set_annotation("chain_id", [chain] * n_atoms)
    atom_array.set_annotation("res_id", data[:, 3].astype(np.int32))
    atom_array.set_annotation(
        # res_idx -1 is the UNK sentinel; plain indexing would silently make it a VAL.
        "res_name",
        [RES_TYPES[res] if res >= 0 else "UNK" for res in data[:, 4].astype(np.int32)],
    )
    atom_array.set_annotation(
        "atom_name", [ATOM_TYPES[atom] for atom in data[:, 5].astype(np.int32)]
    )
    atom_array.set_annotation(
        "element", [ATOM_TYPES[atom][0] for atom in data[:, 5].astype(np.int32)]
    )
    atom_array.set_annotation("b_factor", data[:, 6])
    return atom_array


def clean_structure(atom_array, chain: str | None = "A") -> "bs.AtomArray":
    """Reduce a parsed structure to the amino-acid atoms the tokenizer understands.

    Drops waters, ligands and every other hetero atom, drops hydrogens and any atom
    name outside :data:`ATOM_TYPES`, and keeps a single chain. Pass ``chain=None`` to
    keep all chains (they are then encoded as one continuous chain, which is rarely
    what you want for a multimer).

    Raises ``ValueError`` rather than returning an empty selection, since an empty
    structure otherwise fails much later with an opaque error.
    """
    if isinstance(atom_array, bs.AtomArrayStack):
        # Multi-model file (NMR ensemble, trajectory): the first model is the convention.
        atom_array = atom_array[0]

    mask = bs.filter_amino_acids(atom_array) & ~atom_array.hetero
    mask &= np.isin(atom_array.atom_name, ATOM_TYPES)
    if chain is not None:
        mask &= atom_array.chain_id == chain

    cleaned = atom_array[mask]
    if cleaned.array_length() == 0:
        present = ", ".join(sorted(np.unique(atom_array.chain_id))) or "none"
        raise ValueError(
            f"No amino-acid atoms remain after filtering chain {chain!r}. "
            f"Chains in this structure: {present}. Pass chain=<id> to pick one of "
            "those, or chain=None to keep every chain."
        )
    return cleaned


def load_structure(path, chain: str | None = "A", model: int = 1) -> "bs.AtomArray":
    """Read a structure file and clean it for tokenization.

    Handles every format biotite reads -- ``.pdb``, ``.cif``/``.pdbx``, ``.bcif``,
    ``.gro``, ``.mol`` -- and returns a chain-selected, hetero-free ``AtomArray`` ready
    to hand to ``model.input([{"prot_struct": ...}])``.

    ``model`` is the 1-based model number to take from a multi-model file.
    """
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No structure file at {path!r}. To parse text you already have in memory, "
            "use parse_structure_string() instead."
        )
    structure = strucio.load_structure(path, model=model, extra_fields=["b_factor"])
    return clean_structure(structure, chain=chain)


def parse_structure_string(
    text: str, format: str = "pdb", chain: str | None = "A", model: int = 1
) -> "bs.AtomArray":
    """Same as :func:`load_structure`, for structure text already in memory.

    ``format`` is ``"pdb"`` or ``"cif"`` (``"mmcif"``/``"pdbx"`` are accepted spellings
    of the latter) -- useful for structures fetched over the network or produced by a
    prediction tool without touching disk.
    """
    fmt = format.lower()
    if fmt == "pdb":
        from biotite.structure.io.pdb import PDBFile

        structure = PDBFile.read(io.StringIO(text)).get_structure(
            model=model, extra_fields=["b_factor"]
        )
    elif fmt in ("cif", "mmcif", "pdbx"):
        from biotite.structure.io import pdbx

        structure = pdbx.get_structure(
            pdbx.CIFFile.read(io.StringIO(text)), model=model, extra_fields=["b_factor"]
        )
    else:
        raise ValueError(f"Unsupported format {format!r}; expected 'pdb' or 'cif'.")
    return clean_structure(structure, chain=chain)


def pdb_to_numpy(path, chain: str | None = "A", model: int = 1) -> np.ndarray:
    """Read a structure file straight into the compact ``(n_atoms, 7)`` array.

    This is the storage form (what MIMIC's own datasets hold). Tokenizing does not
    require it -- ``tokenize`` takes a path or an ``AtomArray`` directly -- but it is
    the right thing to persist alongside other modalities.

    Residues outside the standard 20 with no entry in the conversion table are recorded
    as UNK with a warning rather than failing the whole structure: residue identity does
    not affect the structure tokens, which encode backbone geometry only.
    """
    atom_array = load_structure(path, chain=chain, model=model)
    chains = np.unique(atom_array.chain_id)
    if len(chains) > 1:
        # The compact format has no chain column, so it cannot round-trip a multimer.
        raise ValueError(
            f"The compact array format holds a single chain, but {len(chains)} are "
            f"selected ({', '.join(sorted(chains))}). Pass chain=<id>."
        )
    unmapped = sorted(
        {
            res
            for res in np.unique(atom_array.res_name)
            if res not in RES_INDEX and res not in _NONSTANDARD_RESIDUES
        }
    )
    if unmapped:
        logger.warning(
            f"Recording non-standard residues as UNK: {', '.join(unmapped)}. "
            "Structure tokens are unaffected (they encode backbone geometry only)."
        )
    return struct_to_numpy(atom_array, chain=chains[0], strict=False)


def to_atom_array(structure, chain: str | None = "A") -> "bs.AtomArray":
    """Coerce any supported user input into a tokenizer-ready ``AtomArray``.

    Accepts a path to a structure file, a biotite ``AtomArray`` / ``AtomArrayStack``,
    or the compact ``(n_atoms, 7)`` array from :func:`struct_to_numpy` (already clean
    by construction, so it is passed straight through).
    """
    if isinstance(structure, (str, os.PathLike)):
        return load_structure(structure, chain=chain)
    if isinstance(structure, (bs.AtomArray, bs.AtomArrayStack)):
        return clean_structure(structure, chain=chain)
    if isinstance(structure, np.ndarray):
        return numpy_to_struct(structure)
    raise TypeError(
        f"Cannot read a structure from {type(structure).__name__}. Pass a path to a "
        "PDB/mmCIF file, a biotite AtomArray, or the compact (n_atoms, 7) array from "
        "struct_to_numpy()."
    )
