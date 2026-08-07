from ...base import BaseTokenizer, CharLevelTokenizer
from .vqvae import (
    StructureTokenEncoder,
    StructureTokenDecoder,
    VQVAE_CODEBOOK_SIZE,
    STRUCTURE_BOS_TOKEN,
    STRUCTURE_EOS_TOKEN,
    STRUCTURE_MASK_TOKEN,
    STRUCTURE_PAD_TOKEN,
    STRUCTURE_UNDEFINED_TOKEN,
)

import numpy as np
import torch
import biotite.structure as bs

from pathlib import Path
from huggingface_hub import hf_hub_download

from loguru import logger
from .struct_utils import struct_to_numpy, numpy_to_struct, to_atom_array

from .layers.affine3d import Affine3D
from .layers.constants import atom_order, atom_types, atom_type_num

class ESM3StructureTokenizer(BaseTokenizer):
    def __init__(self,
                 codebook_size: int = VQVAE_CODEBOOK_SIZE,
                 device: torch.device | str = "cpu",
                 ):

        self.device = device

        self.codebook_size = codebook_size
        self.pad_token_id = self.pad_token = codebook_size + 5
        self.mask_token_id = self.mask_token = codebook_size + 6
        self.unk_token_id = self.unk_token = codebook_size + 7

        self.vocab_list = np.arange(self.unk_token_id + 1).tolist()
        self.vocab_size = len(self.vocab_list)

        self.version = "esm3_v0"

        self.initialized = False

        self._warned_tensor = False
        self._warned_ndarray = False

        super().__init__()

    def init(self):

        logger.info("Initializing the ESM3 structure tokenizer...")

        # ESM3 VQVAE weights are pulled (and cached) from the Hugging Face Hub;
        # hf_hub_download returns the resolved local path, handling caching itself.
        self.encoder_path, self.decoder_path = self._download_checkpoints()

        self.encoder = self._load_pretrained_encoder(self.device)
        self.decoder = self._load_pretrained_decoder(self.device)

        self.initialized = True

    def _download_checkpoints(self):

        logger.info("Downloading ESM3 structure tokenizer checkpoints (cached after first use)...")

        encoder_path = Path(hf_hub_download(
            repo_id="EvolutionaryScale/esm3-sm-open-v1",
            filename="data/weights/esm3_structure_encoder_v0.pth",
        ))
        decoder_path = Path(hf_hub_download(
            repo_id="EvolutionaryScale/esm3-sm-open-v1",
            filename="data/weights/esm3_structure_decoder_v0.pth",
        ))
        return encoder_path, decoder_path


    def _load_pretrained_encoder(self, device: torch.device | str = "cpu"):

        if not hasattr(self, "encoder_path"):
            raise ValueError("Encoder path not set. Please download the encoder checkpoint first.")

        with torch.device(device):
            encoder = StructureTokenEncoder(
                d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=self.codebook_size,
            ).eval()
        state_dict = torch.load(
            self.encoder_path, map_location=device, weights_only=True
        )
        encoder.load_state_dict(state_dict)
        return encoder

    def _load_pretrained_decoder(self, device: torch.device | str = "cpu"):

        if not hasattr(self, "decoder_path"):
            raise ValueError("Decoder path not set. Please download the decoder checkpoint first.")

        with torch.device(device):
            decoder = StructureTokenDecoder(
                d_model=1280, n_heads=20, n_layers=30
            ).eval()
        state_dict = torch.load(
            self.decoder_path, map_location=device, weights_only=True
        )
        decoder.load_state_dict(state_dict)
        return decoder

    def _biotite_to_atom37(self, atom_array: bs.AtomArray) -> np.array:
        # Chain selection and hetero/hydrogen filtering happen upstream in
        # struct_utils.clean_structure; this only reshapes into the atom37 layout.
        atom_array = atom_array[bs.filter_amino_acids(atom_array) & ~atom_array.hetero]

        num_res = bs.get_residue_count(atom_array)

        atom_positions = np.full(
            [num_res, atom_type_num, 3],
            np.nan,
            dtype=np.float32,
        )
        residue_index = np.full([num_res], -1, dtype=np.int64)

        for i, res in enumerate(bs.residue_iter(atom_array)):
            res_index = res[0].res_id
            residue_index[i] = res_index

            # Atom level features
            for atom in res:
                atom_name = atom.atom_name
                if atom_name == "SE" and atom.res_name == "MSE":
                    # Put the coords of the selenium atom in the sulphur column
                    atom_name = "SD"

                if atom_name in atom_order:
                    atom_positions[i, atom_order[atom_name]] = atom.coord

        return torch.from_numpy(atom_positions), torch.from_numpy(residue_index)

    def _backbone_to_atom37(self, bb_cords: np.array) -> np.array:

        bb_coords = bb_cords[
            0, 1:-1, ...
        ]  # Remove BOS and EOS tokens
        bb_coords = bb_coords.detach().cpu()

        if isinstance(bb_coords, torch.Tensor):
            bb_coords = bb_coords.cpu().numpy()
            if bb_coords.ndim == 4:
                if bb_coords.shape[0] != 1:
                    raise ValueError(
                        f"Cannot handle batched inputs, bb_coords has "
                        f"shape {bb_coords.shape}"
                    )
                bb_coords = bb_coords[0]

        assert isinstance(bb_coords, np.ndarray)
        assert bb_coords.ndim == 3
        assert bb_coords.shape[-2] == 3
        assert bb_coords.shape[-1] == 3

        atom37_positions = np.full(
            (bb_coords.shape[0], 37, 3),
            np.inf,
            dtype=bb_coords.dtype,
        )
        atom37_positions[:, :3, :] = bb_coords
        return atom37_positions

    def _infer_oxygens(self, init_atom37_positions: np.array):
        O_vector = torch.tensor([0.6240, -1.0613, 0.0103], dtype=torch.float32)
        N, CA, C = torch.from_numpy(init_atom37_positions)[...,:3,:].unbind(dim=-2)
        N = torch.roll(N, -3)
        N[..., -1, :] = torch.nan

        # Get the frame defined by the CA-C-N atom
        frames = Affine3D.from_graham_schmidt(CA, C, N)
        O = frames.apply(O_vector)
        atom37_positions = init_atom37_positions.copy()

        atom37_positions[:, atom_order["O"]] = O.numpy()
        return atom37_positions

    def _atom37_to_biotite(self, coords: np.array) -> bs.AtomArray:
        atoms = []

        seqlen = coords.shape[0]
        residue_index = np.arange(1, seqlen + 1)

        for res_idx, positions, in zip(
            residue_index,
            coords,
        ):
            mask = np.isfinite(positions).all(axis=-1)
            for i, (m, pos) in enumerate(zip(mask, positions)):
                if m:
                    atom = bs.Atom(
                        coord=pos,
                        chain_id="A",
                        res_id=res_idx,
                        res_name="UNK",
                        hetero=False,
                        atom_name=atom_types[i],
                        element=atom_types[i][0],
                    )
                    atoms.append(atom)
        return bs.array(atoms)

    def tokenize(self, structure, chain: str | None = "A") -> list[int]:
        """Encode a protein structure as ESM3 VQVAE token ids, one per residue.

        ``structure`` may be any of:

        * a path to a structure file -- PDB, mmCIF, BinaryCIF, anything biotite reads
        * a biotite ``AtomArray`` (or ``AtomArrayStack``; its first model is used)
        * the compact ``(n_atoms, 7)`` array from ``struct_to_numpy``

        Files and ``AtomArray``\\ s are cleaned first (waters, ligands and hydrogens
        dropped, chain ``chain`` selected); the compact array is already clean by
        construction and is used as given. Only backbone geometry is read, so residue
        identity and side chains do not affect the tokens.
        """
        if not self.initialized:
            self.init()

        device = next(self.encoder.parameters()).device
        bs_structure = to_atom_array(structure, chain=chain)
        coords, residue_index = self._biotite_to_atom37(bs_structure)

        with torch.no_grad():
            z_q, tokens = self.encoder.encode(
                coords.to(device).unsqueeze(0),
                residue_index.to(device).unsqueeze(0),
            )
        return tokens.squeeze().cpu().numpy().tolist()

    def detokenize(self, tokens: list[int], as_biotite: bool = False):
        """Decode structure tokens back to a backbone structure.

        Returns a biotite ``AtomArray`` when ``as_biotite=True``, otherwise the
        compact ``struct_to_numpy`` array (default; this is the inverse of
        ``tokenize``, which consumes that same numpy format).
        """
        if not self.initialized:
            self.init()

        if isinstance(tokens, torch.Tensor):
            if not self._warned_tensor:
                logger.warning("Expected input is list[int], received torch.Tensor: Converting tokens tensor to list")
                self._warned_tensor = True
            tokens = tokens.tolist()
        elif isinstance(tokens, np.ndarray):
            if not self._warned_ndarray:
                logger.warning("Expected input is list[int], received np.ndarray: Converting tokens array to list")
                self._warned_ndarray = True
            tokens = tokens.tolist()
        assert isinstance(tokens, list), f"Expected input is list[int], received {type(tokens)}"

        tokens = [STRUCTURE_BOS_TOKEN] + tokens + [STRUCTURE_EOS_TOKEN]
        decoder_output = self.decoder.decode(
            torch.tensor(tokens, device=self.device).unsqueeze(0)
        )

        atom37_positions = self._backbone_to_atom37(decoder_output["bb_pred"])
        atom37_positions = self._infer_oxygens(atom37_positions)
        bs_structure = self._atom37_to_biotite(atom37_positions)

        if as_biotite:
            return bs_structure
        return struct_to_numpy(bs_structure)

    @staticmethod
    def compute_structure_rmsd_aligned(original_structure: bs.AtomArray, decoded_np: np.array,
                                  chain: str = "A") -> float:
        """
        Compute RMSD with optimal superposition alignment.
        """
        from biotite.structure import superimpose

        # Convert and filter as before
        decoded_structure = numpy_to_struct(decoded_np)
        backbone_atoms = ["N", "CA", "C", "O"]

        orig_filtered = original_structure[
            bs.filter_amino_acids(original_structure) &
            ~original_structure.hetero &
            (original_structure.chain_id == chain) &
            np.isin(original_structure.atom_name, backbone_atoms)
        ]

        decoded_filtered = decoded_structure[
            bs.filter_amino_acids(decoded_structure) &
            ~decoded_structure.hetero &
            (decoded_structure.chain_id == chain) &
            np.isin(decoded_structure.atom_name, backbone_atoms)
        ]

        # Sort both structures
        orig_sorted_indices = np.lexsort((orig_filtered.atom_name, orig_filtered.res_id))
        decoded_sorted_indices = np.lexsort((decoded_filtered.atom_name, decoded_filtered.res_id))

        orig_sorted = orig_filtered[orig_sorted_indices]
        decoded_sorted = decoded_filtered[decoded_sorted_indices]

        # Ensure same length
        min_len = min(len(orig_sorted), len(decoded_sorted))
        orig_sorted = orig_sorted[:min_len]
        decoded_sorted = decoded_sorted[:min_len]

        # Perform superposition and calculate RMSD
        decoded_superimposed, _ = superimpose(orig_sorted, decoded_sorted)
        rmsd = bs.rmsd(orig_sorted, decoded_superimposed)

        return rmsd


esm3_tokenizer = ESM3StructureTokenizer()
