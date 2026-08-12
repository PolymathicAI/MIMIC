# %% Imports
import os
import time
import datetime
import numpy as np
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from tqdm import tqdm

import datasets
import foldcomp
import typer
import biotite.structure as bs
from loguru import logger
from tqdm import tqdm
from multiprocessing import Pool
from itertools import chain, count

from lore.utils.struct import pdb_str_to_numpy, ATOM_INDEX
from lore.paths import get_data_root_path, get_path

# %% Paths and Parameters

foldcomp_db = get_path("data", "downloads", "afdb", "v4") / "v4"
output_path = get_path("data", "modality", "structure", "v4_plddt_70", fmt="hfds")

# foldcomp_db = get_path("data", "downloads", "afdb", "swissprot_v4") / "afdb_swissprot_v4"
# output_path = get_path("data", "modality", "structure", "swissprot_v4_plddt_70_2", fmt="hfds")

chunk_size = 50000
report_every = 10000
shard_size = "1GB"
num_parallel_chunk = 8
num_proc_per_chunk = 1
bfactor_thresh = 70

# chunk_size = 1000
# report_every = 500
# shard_size = "1GB"
# num_parallel_chunk = 32 
# num_proc_per_chunk = 1
# bfactor_thresh = 70

features = datasets.Features(
        {
            "uniprot_id": datasets.Value("string"),
            "structure": datasets.Sequence(
                datasets.Sequence(datasets.Value("float32"))
            ),
        }
    )

from string import Template

class DeltaTemplate(Template):
    delimiter = "%"

def strfdelta(tdelta, fmt="%H:%M:%S"):
    d = {"D": tdelta.days}
    hours, rem = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    d["H"] = '{:02d}'.format(hours)
    d["M"] = '{:02d}'.format(minutes)
    d["S"] = '{:02d}'.format(seconds)
    t = DeltaTemplate(fmt)
    return t.substitute(**d)

# %% Process Foldcomp
foldcomp_db = Path(foldcomp_db)
assert foldcomp_db.exists()
os.makedirs(output_path, exist_ok=True)

logger.info("Opening FoldComp database...")
with foldcomp.open(str(foldcomp_db)) as fcdb:

    def process_index(idx):
        """Process a single index."""
        # Get the structure
        up_idx, pdb_str = fcdb[idx]
        up_idx = up_idx.split("-")[1]
        # Convert to numpy
        struct_np = pdb_str_to_numpy(pdb_str)
        bfactor_res = struct_np[np.argwhere(struct_np[:, 5] == ATOM_INDEX["CA"]), 6].mean()

        # Filter by b-factor
        if bfactor_res <= bfactor_thresh:
            return (None, None)
        return up_idx, struct_np
    
    def process_batch(b):
        """Process a batch of indices."""
        try:
            res_dict = {
                "uniprot_id": [],
                "structure": [],
            }

            batch_idx, num_batches, batch = b
            time_start = time.time()

            chunk_path = f"{output_path}/chunk_{batch_idx:05d}"
            if os.path.exists(f"{chunk_path}/state.json"):
                print(f"Chunk {chunk_path} already exists, skipping...")
                return 1

            print(f"Processing batch {batch_idx}/{num_batches} from {batch[0]} to {batch[-1]}")

            for i_, idx in enumerate(batch):
                if i_ % report_every == 0:

                    time_now = time.time()
                    time_taken = strfdelta(datetime.timedelta(seconds=time_now - time_start))
                    time_per_sample = (time_now - time_start) / (i_ + 1)
                    est_time_remaining = strfdelta(datetime.timedelta(
                        seconds=time_per_sample * (len(batch) - i_)
                    ))
                    print(f"batch {batch_idx}/{num_batches} ({i_}/{batch[-1] - batch[0]}) [{time_taken}<{est_time_remaining}, {1/time_per_sample:.4f}it/s]")
            
                # Process the index
                up_idx, struct_np = process_index(idx)
                if up_idx is not None:
                    res_dict["uniprot_id"].append(up_idx)
                    res_dict["structure"].append(struct_np)

            print(
                f"Converting batch_{batch_idx} to HuggingFace dataset..."
            )

            # Convert to HuggingFace dataset
            struct_ds = datasets.Dataset.from_dict(
                res_dict, features=features
            )

            print(
                f"Saving batch_{batch_idx} to {chunk_path}..."
            )

            struct_ds.save_to_disk(chunk_path, num_proc=num_proc_per_chunk)

            print(
                f"Wrote batch_{batch_idx} to {chunk_path}"
            )
            
            return 1
        except Exception as e:
            print(f"Error processing batch {b[0]}: {e}")
            return e

    N = len(fcdb)
    total_batches = np.ceil(N / chunk_size)
    batches = np.array_split(np.arange(N), total_batches)
    batch_enum = [(i, len(batches), b) for (i,b) in enumerate(batches)]
    # print(batch_enum)
    # print(batch_enum[0])
    # print(len(batch_enum[0]))
    # import sys
    # sys.exit(1)

    logger.info(f"FoldComp database contains {N} structures, split into {total_batches} batches of size {len(batches[0])}.")

    # Checkpointing/resuming
    completed_chunks = sorted([
        int(x.split("_")[-1]) for x in os.listdir(output_path) if "chunk_" in x
    ])
    for c in completed_chunks:
        if os.path.exists(f"{output_path}/chunk_{c:05d}/state.json"):
            print(f"Chunk {c} already exists, skipping...")
        else:
            completed_chunks.remove(c)
    chunks_to_do = [
        i for i in range(len(batch_enum)) if i not in completed_chunks
    ]
    new_batch_enum = [
        batch_enum[i] for i in chunks_to_do
    ]
    if not len(new_batch_enum):
        print("All chunks already processed, exiting...")
        exit(0)
    else:
        print(f"Still have to do {len(new_batch_enum)} chunks: {chunks_to_do}")

    # import sys
    # sys.exit(1)

    with Pool(processes=num_parallel_chunk) as pool:
        result = list(tqdm(pool.imap(
            process_batch, new_batch_enum
        ), total=len(new_batch_enum), desc="Processing batches", unit="batch"))
        for batch_idx, r in enumerate(result):
            if r != 1:
                logger.warning(f"Batch {batch_idx} failed with result {r}.")