# Computing DSSP

## Pre-requisites

To run the scripts in this directory requires that all the scripts in `01_process/afdb` have succesfully completed which create several HuggingFace datasets in `lore.paths.get_path("data","modality", "structure", [version]) / chunk_*`. Each dataset contains a `"structure"` column, and the arrays in this column can be read with `lore.utils.struct.numpy_to_struct`.

## Build disBatch files and running

Update the variables at the top of `01_build_dssp_computation.py` and run the script. It will output a `disBatch` command for you to run. This has not been automated so that the user can set the relevant job submission parameters.

## Collating results

Update the variables at the top of `02_collate_dssp_results.py` and run the script. This will collate the individual chunk results and build a HuggingFace database in `lore.paths.get_path("data","modality","dssp", [version])`.

All scripts assume that there are several HuggingFace datasets in `lore.paths.get_path("data","afdb_struct",[version],"raw") / shard_*`. Each dataset contains a `"structure"` column, and the arrays in this column can be read with `lore.utils.struct.numpy_to_struct`.