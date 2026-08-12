# %%
import shlex
import subprocess as sp

import datasets
import numpy as np
import pandas as pd
import typer
from biotite.application.dssp import DsspApp
from biotite.structure import AtomArray
from loguru import logger

from lore.utils.struct import numpy_to_struct
from lore.paths import get_data_root_path

# %%

BINARIES_ROOT = get_data_root_path() / "binaries"
DSSP_LOCAL_BIN = BINARIES_ROOT / "mkdssp/4.4.10/bin/mkdssp"
DSSP_LOCAL_DICT = BINARIES_ROOT / "mkdssp/4.4.10/mmcif_pdbx_v50.dic"
DSSP_LOCAL_ENV = BINARIES_ROOT / "mkdssp/4.4.10/libcifpp"

def compute_dssp_batch(batch):
    upid_list = []
    sse_list = []

    # Get the number of examples in this batch
    batch_size = len(batch["uniprot_id"])
    # logger.debug(f"Computing DSSP for {batch_size} examples")

    for i in range(batch_size):
        # Extract data for a single example
        atom_array = numpy_to_struct(np.array(batch["structure"][i]))
        uniprot_id = batch["uniprot_id"][i]

        app = DsspApp(
            atom_array, bin_path=DSSP_LOCAL_BIN
        )
        app._new_cli = True
        app.add_additional_options(
            [
                f"--mmcif-dictionary {DSSP_LOCAL_DICT}"
            ]
        )
        app.start()
        app._process = sp.Popen(
            shlex.split(app.get_command()),
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            env={
                "LIBCIFPP_DATA_DIR": DSSP_LOCAL_ENV
            },
        )
        # print(app.get_command())
        app.join()
        sse = app.get_sse()
        upid_list.append(uniprot_id)
        sse_list.append("".join(sse))

    return {
        "uniprot_id": upid_list,
        "dssp": sse_list,
    }


# %%
def compute_dssp_chunk(
    chunk_path: str,
    out_path: str,
    num_proc: int = 32,
    batch_size: int = 1000,
    start_idx: int = 0,
    end_idx: int = -1,
):
    #  load the huggingface dataset
    dataset = datasets.load_from_disk(chunk_path)
    logger.debug(dataset)

    if end_idx == -1:
        end_idx = len(dataset)

    logger.info(f"Computing DSSP for chunk {chunk_path} from {start_idx} to {end_idx}")
    dataset = dataset.select(range(start_idx, end_idx))
    dssp_dataset = dataset.map(
        compute_dssp_batch, batched=True, num_proc=num_proc, batch_size=batch_size
    )

    logger.info(f"Saving DSSP dataset to {out_path}")
    dssp_dataset.select_columns(["uniprot_id", "dssp"]).save_to_disk(str(out_path))


# %%

if __name__ == "__main__":
    typer.run(compute_dssp_chunk)
