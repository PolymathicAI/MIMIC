#%% Imports
import datasets
import numpy as np
from multiprocessing import Pool
from lore.paths import get_path
from lore.utils.struct import numpy_to_seq
from loguru import logger
from tqdm import tqdm



#%% 

structure_chunks = get_path("data", "modality", "structure", "v4_plddt_70", fmt="chunk")
all_chunks = sorted(list(structure_chunks.glob("chunk_*")))
# all_chunks = all_chunks[:10]
# %%

num_procs = 16

def load_str(x):
    """
    Load the structure from the chunk and save it to a new file.
    """

    print(f"Loading chunk {x.name}")
    # Load the chunk
    chunk_dataset = datasets.load_from_disk(str(x))
    # select only uniprot Id column
    return chunk_dataset

logger.info("Loading chunks...")
with Pool(processes=min(num_procs, len(all_chunks))) as p:
    results = list(
        tqdm(
            p.imap(load_str, all_chunks),
            total=len(all_chunks),
            desc="Loading chunks",
        )
    )
# %% Get Uniprot IDs from AFDB
logger.info("Concatenating chunks...")
all_ds = datasets.concatenate_datasets(results)
up_ids = [i.split("-")[1] for i in tqdm(all_ds["uniprot_id"])]

# %% Get Uniref 90 IDs
ur90_fasta = get_path("data", "intermediate", "uniref90", "mmseqs_db") / "uniref90.fasta"
ur90_lookup = get_path("data", "intermediate", "uniref90", "mmseqs_db") / "uniref90.lookup"
ur90_fasta = ur90_fasta.resolve()
ur90_lookup = ur90_lookup.resolve()
ur90_ids = []
with open(ur90_lookup, "rb") as f:
    for l in tqdm(f, desc="Reading Uniref90 IDs",total=199553294):
        ur90_ids.append(l.split()[1].split(b"_")[1].decode("utf-8"))

# %% Find overlap
logger.info(f"AFDB Uniprot IDs: {len(up_ids)}")
logger.info(f"Uniref90 IDs: {len(ur90_ids)}")
logger.info("Computing overlap...")
overlap = set(up_ids).intersection(set(ur90_ids))
logger.info(f"Overlap: {len(overlap)}")
logger.info("Computing union...")
union = set(up_ids).union(set(ur90_ids))
logger.info(f"Union: {len(union)}")

# %% Save to disk
inter_path = get_path("data", "intermediate", "uniref90", "id_lists") / "afdb70_intersection.txt"
union_path = get_path("data", "intermediate", "uniref90", "id_lists") / "afdb70_union.txt"
inter_path = inter_path.resolve()
union_path = union_path.resolve()
inter_path.parent.mkdir(parents=True, exist_ok=True)
union_path.parent.mkdir(parents=True, exist_ok=True)
with open(inter_path, "w") as f:
    for i in tqdm(overlap):
        f.write(f"{i}\n")
with open(union_path, "w") as f:
    for i in tqdm(union):
        f.write(f"{i}\n")
logger.info(f"Intersection saved to {inter_path}")
logger.info(f"Union saved to {union_path}")