""" Main script for computing solvent accessible surface area (SASA) """
#%% Variables

VERSION = "v2"
NUM_PROC = 32
BATCH_SIZE = 100000

#%% Imports
import datasets
from loguru import logger
from tqdm import tqdm
from lore.paths import get_path
from .sasa_compute import compute_sasa_shard

#%% Get all shards
shard_root = get_path("dataset", "structure", VERSION, "processed")
dataset_paths = [x for x in shard_root.iterdir() if x.is_dir()]
dataset_list = [datasets.load_from_disk(str(x)) for x in dataset_paths]
dataset = datasets.concatenate_datasets(dataset_list)

#%% Compute SASA for all shards
sasa = compute_sasa_shard(
    dataset,
    out_path=get_path("dataset", "sasa", VERSION, "processed"),
    num_proc=NUM_PROC,
    batch_size=BATCH_SIZE
    )
