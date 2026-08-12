"""Second part of main script for computing secondary structure (DSSP)"""
# %% Variables

VERSION = "v4_10m"

# %% Imports
import os

import datasets
from loguru import logger
from tqdm import tqdm

from lore.paths import get_path

# %% Read all DSSP chunk csv files

chunk_root = get_path("data", "modality", "dssp", VERSION, fmt="hfds_chunked")
hfds_root = get_path("data", "modality", "dssp", VERSION, fmt="hfds")
os.makedirs(hfds_root, exist_ok=True)
chunks = [x for x in chunk_root.iterdir() if x.name.startswith("chunk")]
logger.info(f"Found {len(chunks)} DSSP chunks")

# %% Load all DSSP chunk csv files

dssp_results = [datasets.load_from_disk(str(x)) for x in tqdm(chunks)]
dssp_results = datasets.concatenate_datasets(dssp_results)
logger.info(f"Loaded {len(dssp_results)} DSSP results")

# %% Save the DSSP results as a Hugging Face dataset

dssp_results.save_to_disk(str(hfds_root))
logger.info(f"Saved DSSP results to {hfds_root}")
