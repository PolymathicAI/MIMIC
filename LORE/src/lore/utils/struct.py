import io

import biotite.structure as bs
import numpy as np
from biotite.sequence import ProteinSequence
from biotite.structure.io.pdb import PDBFile

RES_TYPES = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]
RES_INDEX = {res_type: i for i, res_type in enumerate(RES_TYPES)}
RES_INDEX["UNK"] = -1

ATOM_TYPES = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
ATOM_INDEX = {atom_type: i for i, atom_type in enumerate(ATOM_TYPES)}


def sanitize_nonstandard_residues(res_name: str):
    if res_name in RES_TYPES + ["UNK"]:
        return res_name

    nonstandard_dict = {
        "HEM": "HIS",
        "HSD": "HIS",
        "HSE": "HIS",
        "HSP": "HIS",
        "MSE": "MET",
        "PTR": "TYR",
        "NEC": "CYS",
        "CSO": "CYS",
    }

    if res_name in nonstandard_dict:
        return nonstandard_dict[res_name]

    raise ValueError(
        f"Non-standard residue {res_name} not found in the dictionary. Please add it to the dictionary with an appropriate conversion."
    )


def struct_to_numpy(atom_array: bs.AtomArray, chain="A"):
    """Compress a biotite AtomArray."""
    atom_array = atom_array[atom_array.chain_id == chain]
    n_atoms = atom_array.coord.shape[0]
    data = np.zeros((n_atoms, 7), dtype=np.float32)
    data[:, :3] = atom_array.coord
    data[:, 3] = atom_array.res_id
    data[:, 4] = [
        RES_INDEX[sanitize_nonstandard_residues(res)] for res in atom_array.res_name
    ]
    data[:, 5] = [ATOM_INDEX[atom] for atom in atom_array.atom_name]
    if hasattr(atom_array, "b_factor"):
        data[:, 6] = atom_array.b_factor
    return data


def numpy_to_struct(data: np.array, chain="A"):
    """Uncompress a biotite AtomArray."""
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


def pdb_to_numpy(pdb_file: str):
    model = bs.io.load_structure(pdb_file, extra_fields=["b_factor"])
    data = struct_to_numpy(model)
    return data


def numpy_to_pdb(data: np.array, filename: str):
    model = numpy_to_struct(data)
    bs.io.save_structure(filename, model)


def pdb_str_to_numpy(pdb_str: str):
    struct = pdb_str_to_struct(pdb_str)
    data = struct_to_numpy(struct)
    return data


def pdb_str_to_struct(pdb_str: str):
    pdb_file = io.StringIO(pdb_str)
    pdb_file_obj = PDBFile.read(pdb_file)

    model = pdb_file_obj.get_structure(extra_fields=["b_factor"])
    return model[0]


def numpy_to_seq(data: np.array) -> str:
    """Convert a numpy array to a sequence string."""
    res_ca = data[np.argwhere(data[:, 5] == ATOM_INDEX["CA"]), 4].flatten()
    seq = "".join(
        [ProteinSequence.convert_letter_3to1(RES_TYPES[int(res)]) for res in res_ca]
    )
    return seq


def struct_to_seq(struct: bs.AtomArray) -> str:
    """Convert a biotite AtomArray to a sequence string."""
    atom_res = "".join(
        [ProteinSequence.convert_letter_3to1(res) for res in struct.res_name]
    )
    seq = bs.apply_residue_wise(struct, atom_res, lambda x: x[0])
    return "".join(seq)
