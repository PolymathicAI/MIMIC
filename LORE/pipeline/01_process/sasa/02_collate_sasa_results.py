"""Second part of main script for computing solvent accessible surface area (SASA)"""
# %% Variables

VERSION = "v4_10m"

# %% Imports
import os

import datasets
from loguru import logger
from tqdm import tqdm

from lore.paths import get_path

# %% Read all SASA chunk csv files

chunk_root = get_path("data", "modality", "sasa", VERSION, fmt="hfds_chunked")
hfds_root = get_path("data", "modality", "sasa", VERSION, fmt="hfds")
os.makedirs(hfds_root, exist_ok=True)
chunks = [x for x in chunk_root.iterdir() if x.name.startswith("chunk")]
logger.info(f"Found {len(chunks)} SASA chunks")

# %% Load all SASA chunks

sasa_results = [datasets.load_from_disk(str(x)) for x in tqdm(chunks)]
sasa_results = datasets.concatenate_datasets(sasa_results)
logger.info(f"Loaded {len(sasa_results)} SASA results")

# %% Save the SASA results as a Hugging Face dataset

sasa_results.save_to_disk(str(hfds_root))
logger.info(f"Saved SASA results to {hfds_root}")
