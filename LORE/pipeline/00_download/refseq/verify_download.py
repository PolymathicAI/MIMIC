#%%

import os
import multiprocessing

from lore import logger
from lore import paths

from tqdm import tqdm

#%% Get directories to verify

download_path = paths.get_path(data_type="data", stage="downloads", name="refseq", version="229")

logger.info(f"Finding genome downloads to verify in directory: {download_path}")

gcf_dirs = [d for d in os.listdir(download_path) if d.startswith("GCF") and os.path.isdir(os.path.join(download_path, d))]

logger.info(f"Preparing to verify {len(gcf_dirs)} genome downloads")
#%%

def verify_checksum(genome):

    path = os.path.join(download_path, genome)
    os.chdir(path)

    md5sum_file = os.path.join(download_path, genome, "md5sum.txt")
    if not os.path.exists(md5sum_file):
        logger.error(f"Checksum file not found for {genome}")
        return False

    command = f"md5sum -c {md5sum_file}  > /dev/null 2>&1"
    result = os.system(command)

    if result == 0:
        return True
    else:
        logger.error(f"Checksum verification failed for {genome}")
        return False
#%%

with multiprocessing.Pool() as pool:
    results = list(tqdm(pool.imap_unordered(verify_checksum, gcf_dirs), total=len(gcf_dirs)))

    if all(results):
        logger.info("All checksum verifications passed.")
    else:
        logger.error("Some checksum verifications failed.")

