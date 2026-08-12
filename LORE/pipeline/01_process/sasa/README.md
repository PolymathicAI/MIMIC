# Computing SASA

## Pre-requisites

To run the scripts in this directory requires that all the scripts in `01_process/afdb` have succesfully completed.

All scripts assume that there are several HuggingFace datasets in `lore.paths.get_path("data","afdb_struct",[version],"raw") / chunk_*`. Each dataset contains a `"structure"` column, and the arrays in this column can be read with `lore.utils.struct.numpy_to_struct`.

## Creating the disBatch file

Set the variables at the top of the file and run:

```bash
python 01_build_sasa_computation.py
```

## Running disBatch

This will spit out a line for you to run, something that looks like

```bash
module load disBatch; sbatch -n 8 disBatch data-processing/01_process/sasa/_sasa_compute_v4_10m.disbatch
```

## Collating results

Update the variables at the top of `02_collate_sasa_results.py` and run the script. This will collate the individual chunk results and build a HuggingFace database in `lore.paths.get_path("data","modality","dssp", [version])`.
