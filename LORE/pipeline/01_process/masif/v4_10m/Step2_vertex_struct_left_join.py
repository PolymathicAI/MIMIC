# %%
from lore import logger
from lore import paths
import datasets
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="Run in test mode with a small subset of data")
# add num_proc argument with default of 16
parser.add_argument("--num_proc", type=int, default=16, help="Number of processes to use for parallel processing")
args = parser.parse_args()
test = args.test
num_proc = args.num_proc

if test:
    logger.info("Running in test mode with a small subset of data")

# Define paths to the data
vertex_path = paths.get_path(data_type="data", stage="intermediate", name="masif", version="v4_10m", fmt="hfds_chunked")
struct_path = paths.get_path(data_type="data", stage="modality", name="prot_struct", version="v4_10m", fmt="hfds_chunked")
joined_path = paths.get_path(data_type="data", stage="intermediate", name="masif", version="v4_10m", fmt="parquet") / "vertex_struct_joined.parquet"

assert joined_path.exists() == False, f"Output path {joined_path} already exists. Please remove it before running."


# Get all the individual dataset folders for vertices
vertex_folders = sorted([el for dir in vertex_path.glob("*/") if dir.is_dir() for el in dir.glob("*/") if el.is_dir()])

# Get all the individual dataset folders for structures
struct_folders = sorted([el for el in struct_path.glob("*/") if el.is_dir()])

logger.info(f"Found {len(vertex_folders)} vertex folders and {len(struct_folders)} struct folders")

# In test mode, only process the first few folders of each type
if test:
    vertex_folders = vertex_folders[2:3]
    struct_folders = struct_folders[:1]
    logger.info(f"Test mode: processing {len(vertex_folders)} vertex and {len(struct_folders)} struct folders")



# in parallel load each folder as a dataset and concatenate them
def load_folder(folder):
    try:
        ds = datasets.load_from_disk(folder)
        return ds
    except Exception as e:
        logger.error(f"Error loading folder {folder}: {e}")
        return None
    
with Pool(processes=num_proc) as pool:
    vertex_datasets_list = list(tqdm(pool.imap(load_folder, vertex_folders), total=len(vertex_folders)))
    struct_datasets_list = list(tqdm(pool.imap(load_folder, struct_folders), total=len(struct_folders)))

# concatenate all the datasets
vertex_ds = datasets.concatenate_datasets([ds for ds in vertex_datasets_list if ds is not None])
struct_ds = datasets.concatenate_datasets([ds for ds in struct_datasets_list if ds is not None])

logger.info("Successfully loaded and concatenated datasets.")
logger.info(f"Vertex DS dimensions: {vertex_ds.shape}")
logger.info(f"Struct DS dimensions: {struct_ds.shape}")

# %%


key_column = 'uniprot_id'
column_to_add = 'prot_struct'

# ==============================================================================
# 1. Create a direct mapping from string ID to row number
# ==============================================================================
logger.info(f"Creating a string-to-row_number map for '{key_column}'...")
# This dictionary will hold the direct mapping: {'uniprot_id': row_index}
# This can consume a lot of RAM for large datasets.
id_to_row_number = {uid: i for i, uid in enumerate(tqdm(struct_ds[key_column]))}
logger.info(f"Map created with {len(id_to_row_number)} unique entries.")


# ==============================================================================
# 2. Define the merging function using the dictionary and apply it
# ==============================================================================
def merge_with_dict(batch):
    retrieved_indices = []
    # For each key in the batch, look up its row number in the map
    # Use a default of -1 for keys that aren't found
    for key in batch[key_column]:
        retrieved_indices.append(id_to_row_number.get(key, -1))
    
    retrieved_structs = []
    # For each found index, retrieve the data. Handle misses (-1).
    for idx in retrieved_indices:
        if idx != -1:
            # Directly access the row in struct_ds to get the desired column
            retrieved_structs.append(struct_ds[idx][column_to_add])
        else:
            # If the key was not found, append None
            retrieved_structs.append(None)
            
    batch[column_to_add] = retrieved_structs
    return batch


# (The final .map() call)
logger.info("Merging datasets using dictionary lookup...")
merged_ds = vertex_ds.map(
    merge_with_dict,
    batched=True,
    batch_size=1024,
    num_proc=num_proc
)

logger.info("Successfully merged datasets.")
logger.info(f"Merged DS dimensions: {merged_ds.shape}")

# Use num_proc for parallel writing, which can significantly speed up the process for large datasets
merged_ds.save_to_disk(
    joined_path,
    num_proc=num_proc
)

logger.info("Successfully saved the final dataset. ✅")