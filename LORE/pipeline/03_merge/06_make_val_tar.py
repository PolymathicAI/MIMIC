#!/usr/bin/env python3
# %%

import shutil
from lore import paths
from lore import logger
from tqdm import tqdm
from pathlib import Path
import time, yaml
import random
import subprocess
import concurrent.futures

# ====================================================================
# START: Worker Function Definition
# ====================================================================

def fast_rm(path: Path):
    """
    Fast removal of a directory using 'rm -rf' via subprocess.
    This is often faster than shutil.rmtree for large directories.
    """
    # validate that what we are deleting is inside the /tmp directory for safety
    if not str(path).startswith("/tmp/"):
        raise ValueError(f"Path {path} is not inside /tmp directory. Aborting.")
    subprocess.run(
            ['rm', '-rf', str(path)],
            check=True,
            capture_output=True # Captures stdout/stderr
        )

def process_source_task(args):
    """
    Worker function to process a single source directory (e.g., 'val-001').
    This function is designed to be run in a process pool.
    """
    # Unpack arguments
    src, dst, desc, task_num, total_tasks, max_samples, base_seed = args
    
    try:
        # 1. Copy source directory
        logger.debug(f"[{desc}] [{task_num}/{total_tasks}] Copying {src.name} to {dst}")
        shutil.copytree(src, dst, dirs_exist_ok=True)
        
        delete_list = [] # This list is specific to this task

        # 2. Process modalities
        mod_dirs = [el for el in dst.iterdir() if el.is_dir()]
        
        # This inner loop over modalities must be sequential to share the delete_list
        for mod_dir in mod_dirs:
            shard_path = mod_dir / "shard"

            # 3. Untar files
            tar_files = list(mod_dir.glob("*.tar"))
            if not tar_files:
                logger.error(f"[{desc}] [{task_num}/{total_tasks}] No tar files found in {mod_dir}. Skipping.")
                continue

            for tar_file in tar_files:
                try:
                    shutil.unpack_archive(tar_file, extract_dir=shard_path, filter='fully_trusted')
                    tar_file.unlink()
                except Exception as e:
                    logger.error(f"[{desc}] [{task_num}/{total_tasks}] Failed to unpack or delete {tar_file}: {e}")

            # 4. Sub-sampling logic
            untarred_files = list(shard_path.glob("*"))
            if len(untarred_files) > max_samples:
                
                local_delete_list = []
                
                # CRITICAL: Use a task-specific random generator for reproducible shuffling
                if not delete_list:
                    # Create a new, task-specific Random instance
                    # Seeding with base_seed + task_num ensures this task
                    # is deterministic, regardless of execution order.
                    task_seed = base_seed + task_num
                    local_random = random.Random(task_seed)
                    
                    local_random.shuffle(untarred_files)
                    delete_list = [el.name for el in untarred_files[max_samples:]]
                    local_delete_list = delete_list
                else:
                    # Reuse the delete_list from the first modality
                    local_delete_list = delete_list

                # 5. Parallel deletion (nested pool is fine)
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    # Check for existence before unlinking for robustness
                    files_to_delete = [shard_path / fname for fname in local_delete_list if (shard_path / fname).exists()]
                    futures = [executor.submit(f.unlink, missing_ok=False) for f in files_to_delete]
                    for future in concurrent.futures.as_completed(futures):
                        future.result() # Raise any exceptions from unlink

        return f"[{desc}] [{task_num}/{total_tasks}] Successfully processed {src.name}"

    except Exception as e:
        logger.error(f"[{desc}] [{task_num}/{total_tasks}] FAILED to process {src.name}: {e}")
        # Re-raise the exception so the main executor loop can catch it
        raise

def untar_in_place_task(tar_file, desc):
    """
    Worker function to untar a single file in place for the 'Git Test' case.
    """
    try:
        extract_dir = tar_file.with_suffix("")
        shutil.unpack_archive(tar_file, extract_dir, "tar", filter='fully_trusted')
        logger.debug(f"[{desc}] Untarred {tar_file} to {extract_dir}")
        return tar_file
    except Exception as e:
        logger.error(f"[{desc}] Failed to untar {tar_file}: {e}")
        return None

# ====================================================================
# END: Worker Function Definition
# ====================================================================

def main():

    """
    Creates tar files for the validation subsets from the webdataset format.
    This is used to efficiently copy the validation data to the /tmp folder
    during model training.
    """

    with open(Path(__file__).parent / "tokenize_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    logger.info(f"Loaded config from {file.name}")

    ds_name = config["name"]
    version = str(config["version"])
    max_samples = config["max_val_samples"]
    # We now use this as a *base* seed for workers
    seed = config["seed"]
    logger.info(f"Using max_samples={max_samples} and random seed={seed} for sample selection.")

    # %%

    start_time = time.time()

    for ds_version, desc in [(version + "_gittest", "Git Test"), (version, "Full")]:
        
        logger.info(f"Processing {desc} dataset version {ds_version}...")

        webdataset_path = paths.get_path(data_type='data', name=ds_name, stage='final', version=ds_version, fmt='wds')
        wds_config_path = webdataset_path / "tokenize_config.yaml"
        # assert the config file matches the copied config file
        with open(wds_config_path, "r") as file:
            copied_config = yaml.safe_load(file)
        if copied_config != config:
            raise ValueError(f"Config file {wds_config_path} does not match the current config. This implies a version conflict.")

        subsets = [el for el in webdataset_path.iterdir() if el.is_dir()]
        val_paths = {subset_dir.stem: sorted(subset_dir.glob('val*')) for subset_dir in subsets}

        val_tar_path = paths.get_path(data_type='data', name=ds_name, stage='final', version=ds_version, fmt='tar') 
        if val_tar_path.exists():
            logger.error(f"[{desc}] {val_tar_path} already exists. Exiting.")
            exit(1)

        temp_path = Path("/tmp/make_val_tar")

        # copy the val folder to local /tmp/make_val_tar/ If the folder already exists, delete it.
        logger.info(f"[{desc}] Copying validation splits of {len(val_paths)} subsets to {temp_path}")
        # check to see if the folder exists and delete it. Logger a warning if it does.
        if temp_path.exists():
            logger.warning(f"[{desc}] {temp_path} already exists. Deleting it.")
            fast_rm(temp_path)
        temp_path.mkdir(parents=True, exist_ok=False)

        # ====================================================================
        # START: Parallel Task Collection and Execution
        # ====================================================================
        
        # 1. Collect all tasks into a list
        logger.info(f"[{desc}] Collecting tasks to process...")
        tasks_to_run = []
        i = 0
        for name, sources in val_paths.items():
            logger.debug(f"[{desc}] Found {len(sources)} val folders for subset: {name}")
            for src in sources:
                i += 1
                dst = temp_path / name / src.name
                # Add all arguments for the worker function as a tuple
                tasks_to_run.append((src, dst, desc, i, 0, max_samples, seed))
        
        # Update total_tasks count in all task tuples
        total_tasks = len(tasks_to_run)
        tasks_to_run = [
            (src, dst, desc, i, total_tasks, max_samples, base_seed) 
            for (src, dst, desc, i, _, max_samples, base_seed) in tasks_to_run
        ]
        logger.info(f"[{desc}] Total number of tasks to process: {total_tasks}")

        # 2. Execute tasks in parallel using ProcessPoolExecutor
        #    max_workers=None will use a sensible default (often 5 * CPU cores)
        with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
            # Submit all tasks to the pool
            future_to_task = {executor.submit(process_source_task, task): task for task in tasks_to_run}
            
            # Use tqdm to create a progress bar
            for future in tqdm(concurrent.futures.as_completed(future_to_task), total=total_tasks, desc=f"[{desc}] Processing sources"):
                try:
                    result = future.result() # Get the result (or re-raise the exception)
                    logger.debug(result) # Log success at debug level to avoid spam
                except Exception as e:
                    task_args = future_to_task[future]
                    src_path = task_args[0]
                    i = task_args[3]
                    logger.critical(f"[{desc}] A critical error occurred while processing {src_path} (task {i}/{total_tasks}). Stopping execution. Error: {e}")

        # ====================================================================
        # END: Parallel Task Collection and Execution
        # ====================================================================

        val_tar_path.mkdir(parents=True, exist_ok=False)

        # Define the full path for the tar file
        output_tar_file = val_tar_path / "val.tar"

        logger.info(f"[{desc}] Tarring files from {temp_path} into {output_tar_file}")
        try:
            subprocess.run(
                ['tar', '-cf', str(output_tar_file), '.'],
                cwd=temp_path,
                check=True,  # This will raise an exception if tar fails
                capture_output=True # Captures stdout/stderr
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"[{desc}] Tarring failed with exit code {e.returncode}")
            logger.error(f"[{desc}] STDERR: {e.stderr.decode()}")
            raise  # Re-raise the exception to stop the script

        # copy the config file to the tar path
        copy_config_path = val_tar_path / "tokenize_config.yaml"
        shutil.copyfile(wds_config_path, copy_config_path)
        logger.info(f"[{desc}] Copied config file to {copy_config_path}")

        logger.info(f"[{desc}] Tar file created, deleting temporary folder {temp_path}")
        # Remove the temporary directory
        fast_rm(temp_path)
        logger.info(f"[{desc}] Deleted temporary folder {temp_path}")

        # =G==================================================================
        # START: Parallel "Git Test" Untarring
        # ====================================================================
        
        # for the gittest dataset, iterate over the subsets and untar in place
        if desc == "Git Test":

            # find all the tar files
            logger.info(f"[{desc}] Finding val tars to untar in place...")
            tar_files = [el for subset in webdataset_path.iterdir() if subset.is_dir()
                        for split in subset.iterdir() if split.is_dir() and 'val' in split.name
                        for mod in split.iterdir() if mod.is_dir()
                        for el in mod.glob("*.tar")]

            logger.info(f"[{desc}] Found {len(tar_files)} tar files. Untarring in parallel...")

            # iterate over the tar files and untar in place
            with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
                futures = [executor.submit(untar_in_place_task, tar_file, desc) for tar_file in tar_files]
                
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(tar_files), desc=f"[{desc}] Untarring in place"):
                    future.result() # Check for exceptions

            logger.info(f"[{desc}] Also untarred the val tars in place.")

            # change the permissions of the webdataset_path to 777 recursively using -R
            logger.info(f"[{desc}] Changing permissions of {webdataset_path} to +rw recursively.")
            subprocess.run(['chmod', '-R', '+rw', str(webdataset_path)], check=True)
        # ====================================================================
        # END: Parallel "Git Test" Untarring
        # ====================================================================

        print("\n" + "="*80 + "\n")
        
    # %%
    elapsed_time = time.time() - start_time
    m, s = divmod(elapsed_time, 60)
    logger.info(f"Finished in {int(m)}m {int(s)}s")
    # %%

if __name__ == "__main__":
    main()