import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from biotite.structure import apply_residue_wise, get_residues
from datasets import load_from_disk
from scipy.spatial.distance import cdist
from tqdm import tqdm

from lore import logger
from lore import paths
from lore.utils.struct import numpy_to_struct

ROOT_PATH = paths.get_path(data_type="data", stage="intermediate", name="masif", version="v4_10m", fmt="parquet")
HFDS_PATH = ROOT_PATH / "vertex_struct_joined.parquet"
JOB_ID = int(sys.argv[1])
TOTAL_JOBS = 60
test = False

def process_row(row):
    try:
        vi = np.stack(row["vertices_features"]).astype(np.float32) # [N_vertices, 7]
        st_np_array = np.stack(row["prot_struct"]).astype(np.float32) # [N_atoms, 7]
        atom_array = numpy_to_struct(st_np_array)
        atom_coords = atom_array.coord # [N_atoms, 3]
        distances = cdist(atom_coords, vi[:, 0:3], 'euclidean')
        res_vertice_distance = apply_residue_wise(array=atom_array, data=distances, function=np.min, axis=0).astype(np.float32)
        assert res_vertice_distance.shape[0] == len(get_residues(atom_array)[0])
        assert res_vertice_distance.shape[1] == len(vi)
        res_vertice_bool = np.where(res_vertice_distance < 2.8, 1, 0)
        # number of surface vertices for each residue
        N_vertices = res_vertice_bool.sum(axis=1)
        # Shapre index for each residue, mean
        SI_res_sum = res_vertice_bool @ vi[:, 3]
        SI_res = np.divide(SI_res_sum, N_vertices, where=N_vertices != 0)
        SI_res[N_vertices == 0] = np.nan
        # Charge for each residue, sum
        charge_res = res_vertice_bool @ vi[:, 4]
        charge_res[N_vertices == 0] = np.nan
        # h-bond donor for each residue, sum
        hbonds_res = res_vertice_bool @ vi[:, 5]
        hbonds_res[N_vertices == 0] = np.nan
        # hydrophobicity for each residue, mean
        hydro_res_sum = res_vertice_bool @ vi[:, 6]
        hydro_res = np.divide(hydro_res_sum, N_vertices, where=N_vertices != 0)
        hydro_res[N_vertices == 0] = np.nan
        return [row["uniprot_id"], N_vertices.tolist(), SI_res.tolist(), charge_res.tolist(), hbonds_res.tolist(), hydro_res.tolist()]
    except Exception as e:
        logger.error(f"Processing failed for {row['uniprot_id']}: {e}")
        return [row["uniprot_id"], None, None, None, None, None]


logger.info(f"Now Running Job ID {JOB_ID}...", flush=True)

# Load the full HuggingFace dataset
logger.info(f"Loading dataset from {HFDS_PATH}...", flush=True)
dataset = load_from_disk(HFDS_PATH)

# Calculate the chunk size for this job
total_rows = len(dataset)
num_jobs = TOTAL_JOBS
chunk_size = (total_rows + num_jobs - 1) // num_jobs  # Ceiling division
start_idx = JOB_ID * chunk_size
end_idx = min(start_idx + chunk_size, total_rows)
logger.info(f"Total dataset size: {total_rows} rows")


if test:
    test_num_rows = 1000
    end_idx = min(start_idx + test_num_rows, end_idx)
    logger.info(f"Test mode: limiting to {end_idx - start_idx} rows")

logger.info(f"Job {JOB_ID}: Processing rows {start_idx} to {end_idx} (chunk size: {end_idx - start_idx})")
# Select the chunk for this job and convert to pandas
dataset_chunk = dataset.select(range(start_idx, end_idx))
df = dataset_chunk.to_pandas()

logger.info("Dropping rows where structure is None...")
df_filtered = df[df['prot_struct'].notna() & df['vertices_features'].notna()]

logger.info(f"Dropped {len(df) - len(df_filtered)}/{len(df)} rows where structure or vertices_featuresis None.")

N_DATA = len(df_filtered)
N_CPUS = int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count()))
N_WORKERS = max(1, N_CPUS // 2)

batch_results = []
with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = [executor.submit(process_row, row) for _, row in df_filtered.iterrows()]
    for future in tqdm(futures):
        batch_results.append(future.result())

outputs = {
    "uniprot_id": [result[0] for result in batch_results],
    "n_vertices": [result[1] for result in batch_results],
    "si_index": [result[2] for result in batch_results],
    "charge": [result[3] for result in batch_results],
    "hbond": [result[4] for result in batch_results],
    "hydrophobicity": [result[5] for result in batch_results]
}

df_agg = pd.DataFrame.from_dict(outputs)
# Save to disk (efficient Arrow format)
OUTPUT_FOLDER = ROOT_PATH / "chunks_aggregated"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_PATH = OUTPUT_FOLDER / f"masif10_aggregated_chunk_{JOB_ID}.parquet"
df_agg.to_parquet(OUTPUT_PATH, engine="pyarrow")
logger.info(f"JOB ID {JOB_ID} done! Saved to {OUTPUT_PATH}", flush=True)






