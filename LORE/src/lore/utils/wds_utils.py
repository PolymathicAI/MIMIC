import io
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd
import webdataset as wds
from webdataset.writer import numpy_dumps

from lore import logger

_DATASET: pd.DataFrame | None = None  # Placeholder for parallel processing


# A reshaper for tensor components
def serialize_data(data: np.ndarray, dtype: str) -> bytes:
    # The new codebase does not have augmentations so we just use a list.
    data = data.reshape(-1)
    # Cast with the right dtype to reduce memory footprint
    data = data.astype(dtype)
    return numpy_dumps(data)


# A function that writes each dataset in the appropriate format


def write_dataset(
    dataset: pd.DataFrame,
    base_path,
    dataset_name,
    max_samples_per_tar=5_000,
    id_col="__key__",
    parallel=True,
    log_prefix="",
    parallel_mode="thread",
):
    # set the global dataset variable
    global _DATASET
    _DATASET = dataset

    base_path = os.path.expanduser(base_path)

    # Create the base folder for the dataset
    dataset_path = os.path.join(base_path, dataset_name)
    os.makedirs(dataset_path, exist_ok=True)

    # Identify modality columns (exclude '__key__')
    modality_columns = [col for col in dataset.columns if col != id_col]

    logger.debug(f"{log_prefix} Modality columns: {modality_columns}")

    all_args = [
        (modality, dataset_path, id_col, max_samples_per_tar, log_prefix)
        for modality in modality_columns
    ]

    if not parallel:
        logger.debug("Disabling parallelization.")
        for args in all_args:
            write_modality(args)
    elif parallel_mode == "process":
        logger.debug(f"{log_prefix} Writing dataset with {len(all_args)} processes...")
        with ProcessPoolExecutor(max_workers=len(all_args)) as executor:
            executor.map(write_modality, all_args)
    elif parallel_mode == "thread":
        logger.debug(f"{log_prefix} Writing dataset with {len(all_args)} threads...")
        with ThreadPoolExecutor(max_workers=len(all_args)) as executor:
            executor.map(write_modality, all_args)
    else:
        raise ValueError(f"Unknown parallel_mode: {parallel_mode}")

    logger.debug(f"{log_prefix}  Dataset written to {dataset_path}")

    _DATASET = None  # Clear the global dataset variable to free memory


def write_modality(args):
    modality, dataset_path, id_col, max_samples_per_tar, log_prefix = args
    dataset = _DATASET  # Use the global dataset variable
    # Create a subfolder for each modality
    modality_path = os.path.join(dataset_path, modality)
    os.makedirs(modality_path, exist_ok=True)

    # Write the modality to the webdataset format with multiple tar files
    sample_count = 0
    tar_index = 0
    tar_writer = None

    for _, row in dataset.iterrows():
        # Create a new tar file if needed
        if sample_count % max_samples_per_tar == 0:
            if tar_writer:
                tar_writer.close()
                tar_index += 1
            tar_path = os.path.join(modality_path, f"shard-{tar_index:06d}.tar")
            tar_writer = wds.TarWriter(tar_path)

            # Write the sample
        sample = {
            "__key__": str(row[id_col]),
            "npy": serialize_data(
                np.array(row[modality]), dtype=int
            ),  # Save modality data as npy
        }
        tar_writer.write(sample)
        sample_count += 1

    # Close the last tar writer
    if tar_writer:
        tar_writer.close()
    logger.debug(
        f"{log_prefix} Finished writing modality: {modality} with {sample_count:,} rows."
    )


# Define a function to decode the 'npy' field when reading the webdataset
def decode_npy(sample):
    if "npy" in sample:
        npy_data = sample["npy"]
        sample["npy"] = np.load(io.BytesIO(npy_data))  # Decode NumPy array
    return sample
