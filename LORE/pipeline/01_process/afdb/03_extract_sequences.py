#%% Imports
import datasets
import numpy as np
from multiprocessing import Pool
from lore.paths import get_path
from lore.utils.struct import numpy_to_seq
from loguru import logger
from tqdm import tqdm



#%% 

structure_chunks = get_path("data", "modality", "structure", "v4_plddt_70", fmt="hfds_chunked")
all_chunks = sorted(list(structure_chunks.glob("chunk_*")))

# %%

# Run with pool

num_procs = 90 

def load_str(x):
    """
    Load the structure from the chunk and save it to a new file.
    """

    print(f"Loading chunk {x.name}")
    # Load the chunk
    chunk_dataset = datasets.load_from_disk(str(x))
    # select only 1000 sequences
    # chunk_dataset = chunk_dataset.select(range(1000))
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

logger.info("Concatenating chunks...")
all_ds = datasets.concatenate_datasets(results)


# Extract sequences
logger.info("Extracting sequences...")
sequences = all_ds.map(
        lambda x: {"sequence": numpy_to_seq(np.array(x["structure"]))},
        remove_columns=["structure"],
        desc=f"Extracting sequences",
        num_proc=num_procs,
    )

# Save to disk
logger.info("Saving sequences...")
sequences.save_to_disk(
    str(get_path("data", "modality", "aa_seq", "v4_plddt_70", fmt="hfds"))
)
# %%
