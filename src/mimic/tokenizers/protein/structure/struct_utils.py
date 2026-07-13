"""Biotite <-> numpy conversion for the structure tokenizer.

Vendored from the training codebase so the public package has no dependency on
internal modules. Only the two conversions the structure tokenizer needs are kept
here; `biotite` is an optional dependency (installed via the ``mimic[structure]`` extra).
"""
import numpy as np
import biotite.structure as bs

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


def sanitize_nonstandard_residues(res_name: str):
    if res_name in RES_INDEX:  # standard residues + "UNK"
        return res_name
    if res_name in _NONSTANDARD_RESIDUES:
        return _NONSTANDARD_RESIDUES[res_name]
    raise ValueError(
        f"Non-standard residue {res_name} not found in the conversion table. "
        "Please add it with an appropriate standard-residue mapping."
    )


def struct_to_numpy(atom_array: "bs.AtomArray", chain="A"):
    """Compress a biotite AtomArray into an (n_atoms, 7) float32 array."""
    atom_array = atom_array[atom_array.chain_id == chain]
    n_atoms = atom_array.coord.shape[0]
    data = np.zeros((n_atoms, 7), dtype=np.float32)
    data[:, :3] = atom_array.coord
    data[:, 3] = atom_array.res_id
    data[:, 4] = [RES_INDEX[sanitize_nonstandard_residues(res)] for res in atom_array.res_name]
    data[:, 5] = [ATOM_INDEX[atom] for atom in atom_array.atom_name]
    if hasattr(atom_array, "b_factor"):
        data[:, 6] = atom_array.b_factor
    return data


def numpy_to_struct(data: np.ndarray, chain="A"):
    """Uncompress an (n_atoms, 7) array back into a biotite AtomArray."""
    n_atoms = data.shape[0]
    atom_array = bs.AtomArray(n_atoms)
    atom_array.coord = data[:, :3]
    atom_array.set_annotation("chain_id", [chain] * n_atoms)
    atom_array.set_annotation("res_id", data[:, 3].astype(np.int32))
    atom_array.set_annotation(
        "res_name", [RES_TYPES[res] for res in data[:, 4].astype(np.int32)]
    )
    atom_array.set_annotation(
        "atom_name", [ATOM_TYPES[atom] for atom in data[:, 5].astype(np.int32)]
    )
    atom_array.set_annotation(
        "element", [ATOM_TYPES[atom][0] for atom in data[:, 5].astype(np.int32)]
    )
    atom_array.set_annotation("b_factor", data[:, 6])
    return atom_array
