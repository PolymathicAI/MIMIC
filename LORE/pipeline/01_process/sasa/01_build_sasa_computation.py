"""Main script for building the SASA (solvent accessible surface area) computation jobs.

Emits `_sasa_compute.sh` plus a disBatch task file (one task per structure chunk), then
prints the submission line. Both are generated at run time and are not committed --
adapt the printed command to your scheduler.
"""
# %% Variables

# VERSION = "v4_10m"
# NUM_JOBS = 8
# NUM_PROC = 64
# BATCH_SIZE = 500
VERSION = "v4_plddt_70"
NUM_JOBS = 16
NUM_PROC = 64
BATCH_SIZE = 1000

# %% Imports
import os
from pathlib import Path

from loguru import logger

from lore.paths import get_path

# Root of your checkout. Set $LORE_REPO, or run this from the repository root.
CODE_DIRECTORY = os.environ.get("LORE_REPO", os.getcwd())
SCRIPT_DIR = "LORE/pipeline/01_process/sasa"

# %% Get all chunks
logger.info("Getting all chunks")
chunk_root = get_path("data", "modality", "structure", VERSION, fmt="hfds_chunked")
dataset_paths = [x for x in chunk_root.iterdir() if x.name.startswith("chunk")]
dataset_paths = sorted(dataset_paths, key=lambda x: int(x.name.split("_")[1]))
out_paths_root = get_path("data", "modality", "sasa", VERSION, fmt="hfds_chunked")
os.makedirs(out_paths_root, exist_ok=True)
out_paths = [out_paths_root / x.name for x in dataset_paths]

# %% Build the SASA compute submission script
logger.info("Building the SASA compute submission script")

bash_script = f"""
# Needs a C++ toolchain (gcc, boost) and Python >=3.10. Load these however your site
# does it, and activate an environment with `lore[pipeline]` installed.
module --force purge
module load python gcc boost

# Run from your checkout, or point $LORE_REPO at it.
cd "${{LORE_REPO:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}}"

echo "Running SASA on chunk $1"
python {SCRIPT_DIR}/_sasa_compute.py $1 $2 --num-proc {NUM_PROC} --batch-size {BATCH_SIZE}
"""

sasa_script_out = Path(CODE_DIRECTORY) / SCRIPT_DIR / "_sasa_compute.sh"
with open(sasa_script_out, "w") as f:
    f.write(bash_script)

# %% Build the disBatch file
logger.info("Building the disBatch file")

disbatch_file_out = (
    Path(CODE_DIRECTORY) / SCRIPT_DIR / f"_sasa_compute_{VERSION}.disbatch"
)
db_prefix = f"( bash {sasa_script_out} "
db_suffix = r") &> task_${DISBATCH_TASKID}.log"

with open(disbatch_file_out, "w") as f:
    f.write("# DisBatch file for computing SASA\n")
    f.write(f"#DISBATCH PREFIX {db_prefix}\n")
    f.write(f"#DISBATCH SUFFIX {db_suffix}\n")
    for chunk_path, out_path in zip(dataset_paths, out_paths):
        f.write(f"{chunk_path} {out_path}\n")
# %% Submit the disBatch job
logger.info("Run the following in a terminal (set -p to a partition on your cluster):")
logger.info(
    f"\tmodule load disBatch; sbatch -p PARTITION -n {NUM_JOBS} -c {NUM_PROC} "
    f"--mem 800GB disBatch {disbatch_file_out}"
)
