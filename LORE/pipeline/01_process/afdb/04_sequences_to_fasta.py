#%% Imports
import datasets
import numpy as np
from multiprocessing import Pool
from lore.paths import get_path
from lore.utils.struct import numpy_to_seq
from loguru import logger
from tqdm import tqdm

# %% Sequence database

sequence_path = get_path("data", "modality", "aa_seq", "v4_plddt_70", fmt="hfds")
fasta_path = get_path("data", "modality", "aa_seq", "v4_plddt_70", fmt="fasta") / "afdb_70.fasta"
fasta_path = fasta_path.resolve()
sequence_dataset = datasets.load_from_disk(str(sequence_path))

with open(fasta_path, "wb+") as f:
    for i in tqdm(range(len(sequence_dataset)), desc="Writing sequences"):
        seq = sequence_dataset[i]["sequence"]
        uniprot_id = sequence_dataset[i]["uniprot_id"]
        uniprot_id = uniprot_id.split("-")[1]
        f.write(f">{uniprot_id}\n".encode('utf-8'))
        f.write(f"{seq}\n".encode('utf-8'))