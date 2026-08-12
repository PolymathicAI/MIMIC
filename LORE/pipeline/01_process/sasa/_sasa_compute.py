## To submit this script to the cluster using sbatch, run the following command:

# %% Load dependencies
import os
from pathlib import Path

import biotite.structure as bs
import datasets
import numpy as np
import typer
from loguru import logger

from lore.utils.struct import numpy_to_struct

# %%


def compute_sasa_batch(batch):
    upid_list = []
    sasa_list = []

    # Get the number of examples in this batch
    batch_size = len(batch["uniprot_id"])
    # logger.debug(f"Computing SASA for {batch_size} examples")

    for i in range(batch_size):
        # Extract data for a single example
        atom_array = numpy_to_struct(np.array(batch["structure"][i]))
        uniprot_id = batch["uniprot_id"][i]

        sasa_per_atom = bs.sasa(atom_array)
        agg_sasa = bs.apply_residue_wise(atom_array, sasa_per_atom, np.nansum)
        assert len(agg_sasa) == len(set(atom_array.res_id))
        upid_list.append(uniprot_id)
        sasa_list.append(agg_sasa)

    return {
        "uniprot_id": upid_list,
        "sasa": sasa_list,
    }


# %%


def compute_sasa_chunk(
    chunk_path: Path,
    out_path: Path,
    num_proc: int = 32,
    batch_size: int = 1000,
    start_idx: int = 0,
    end_idx: int = -1,
):
    #  load the huggingface dataset
    dataset = datasets.load_from_disk(str(chunk_path))
    logger.debug(dataset)

    if end_idx == -1:
        end_idx = len(dataset)

    logger.info(f"Computing SASA for chunk {chunk_path} from {start_idx} to {end_idx}")
    dataset = dataset.select(range(start_idx, end_idx))
    sasa_dataset = dataset.map(
        compute_sasa_batch, batched=True, num_proc=num_proc, batch_size=batch_size
    )

    logger.info(f"Saving SASA dataset to {out_path}")
    os.makedirs(out_path.parent, exist_ok=True)
    sasa_dataset.select_columns(["uniprot_id", "sasa"]).save_to_disk(str(out_path))


# %%
if __name__ == "__main__":
    typer.run(compute_sasa_chunk)
