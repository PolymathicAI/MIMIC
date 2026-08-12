#!/usr/bin/env python3

# %%
import shutil
from lore import paths
from lore import logger
from lore.utils.download import download_folder
import pandas as pd
import hashlib
from tqdm import tqdm

# %%

path = paths.get_path(data_type="data", stage="downloads", name="phylop", version="hg38-100way")
url = "rsync://hgdownload.cse.ucsc.edu/goldenPath/hg38/phyloP100way/"

# if the directories exist, ask the user if they want to overwrite them
if path.exists():
    logger.info(f"Folder {path} already exists.")
    overwrite = input("Do you want to overwrite them (y)? Or resume (r) (y/n/[r]): ")
    if overwrite.lower() == "r" or overwrite == "":
        logger.info(f"Resuming downloads.")
        pass
    elif overwrite.lower() == "y":
        # delete the directories via rmtree
        logger.info(f"Deleting {path}")
        shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=False)
    elif overwrite.lower() == "n":
        logger.info(f"Exiting.")
        exit(0)
    else:
        logger.info(f"Invalid input. Exiting.")
        exit(0)
else:
    # create the directories
        path.mkdir(parents=True, exist_ok=False)

# %%
logger.info(f"Downloading PhyloP 100-way hg38 to {path}...")
logger.info("The data is about 15GB and takes about 40 mins to download.")
download_folder("PhyloP 100-way hg38", url, path)
logger.info("Download complete.")

# %%

logger.info("Checking md5sum...")

# find all the subfolders that have md5sum.txt file in them
subfolders = [p  for p in path.rglob('*') if p.is_dir()] + [path]
checksum_folders = [p for p in subfolders if (p / "md5sum.txt").exists()]

no_checksum_folders = [p for p in subfolders if not (p / "md5sum.txt").exists()]
if len(no_checksum_folders) > 0:
    logger.warning(f"Found {len(no_checksum_folders)} folders without md5sum.txt file.")
    logger.warning(f"Folders without md5sum.txt file: {no_checksum_folders}")

bad_files = []
missing_md5_files = []
for subfolder in checksum_folders:

    # load the md5sum.txt file
    md5sum = pd.read_csv(subfolder / "md5sum.txt", sep="  ", names=["md5", "file"], engine="python")

    # look at all the files in the subfolder
    files = [el for el in subfolder.iterdir() if el.is_file() and el.name != "md5sum.txt"]
    with tqdm(total=len(files), desc=f"Checking md5sum for {subfolder.name}") as pbar:

        mismatch = 0
        missing = 0
    
        for file in files:

            # check if the file is in the md5sum file
            if file.name not in md5sum["file"].values:
                missing += 1
                missing_md5_files.append(file)
                continue

            # lookup the md5sum for the file
            target_md5 = md5sum[md5sum["file"] == file.name]["md5"].values[0]

            # calculate the md5sum
            md5 = hashlib.md5()
            with open(file, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    md5.update(chunk)

            # check if the md5sum matches
            if md5.hexdigest() != target_md5:
                mismatch += 1
                bad_files.append(file)
        
            pbar.update(1)
            pbar.set_postfix_str(f"mismatch: {mismatch}, missing: {missing}")

other = ""
if len(bad_files) > 0:
    logger.error(f"Found {len(bad_files)} files with mismatched md5sum.")
    logger.error(f"Files with mismatched md5sum: {bad_files}")
    other = "other "
if len(missing_md5_files) > 0:
    logger.warning(f"Found {len(missing_md5_files)} files missing in md5sum.txt.")
    logger.warning(f"Files missing in md5sum.txt: {missing_md5_files}")
    other = "other "

logger.info(f"All {other}files downloaded and verified successfully.")
