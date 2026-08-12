"""
A survival technique for converting NumPy arrays to PDB files in the pre-biotite era, use Biopython == 1.76

RES_TYPES and ATOM_TYPES inherited from Sam's codes

Minhuan Li, Mar 2025
"""

import numpy as np
from Bio.PDB import PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom

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


def numpy_to_pdb(atom_array, output_file):
    """
    Convert a NumPy array of atom data to a PDB file.

    Parameters:
    -----------
    atom_array : numpy.ndarray
        Array where each row represents an atom with columns:
        0-2: x, y, z coordinates
        3: residue ID
        4: residue name index (in RES_TYPES)
        5: atom name index (in ATOM_TYPES)
        6: B-factor
    output_file : str
        Path to the output PDB file
    """
    # Create a new structure
    structure = Structure("1")
    model = Model(0)
    structure.add(model)
    chain = Chain("A")
    model.add(chain)

    # Track the current residue to avoid duplicates
    current_res_id = None
    current_residue = None

    # Atom serial number counter
    atom_serial = 1

    # Process each atom in the array
    for atom_data in atom_array:
        x, y, z = atom_data[0:3]
        res_id = int(atom_data[3])
        res_name = RES_TYPES[int(atom_data[4])]
        atom_name = ATOM_TYPES[int(atom_data[5])]
        b_factor = float(atom_data[6])

        # Create a new residue if needed
        if res_id != current_res_id:
            current_res_id = res_id
            current_residue = Residue((" ", res_id, " "), res_name, "")
            chain.add(current_residue)

        # Create and add the atom
        atom = Atom(
            atom_name,
            np.array([x, y, z], dtype=np.float32),
            b_factor,
            1.0,
            " ",
            atom_name,
            atom_serial,
            atom_name[0],
        )
        current_residue.add(atom)
        atom_serial += 1

    # Write the structure to a PDB file
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file)

    return structure
