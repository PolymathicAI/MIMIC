import os
import subprocess

import foldcomp
from lore import logger

from lore import paths

path = paths.get_path(data_type="data", stage="downloads", name="afdb", version="v4")

# create the directory if it does not exist
os.makedirs(path, exist_ok=True)

# set the working directory to path
logger.info(f"Working directory set to {path}")
os.chdir(path)

# download the data using foldcomp
logger.info("The data is about 1.1TB and takes about 3 hours to download.")
logger.info("Downloading data from foldcomp...")
foldcomp.setup("afdb_uniprot_v4")
logger.info("Data download complete")

# log the tree structure of the data directory
logger.info("Directory structure:")
result = subprocess.run(["tree", "-L", "3"], capture_output=True, text=True, check=True)
logger.info(result.stdout)
