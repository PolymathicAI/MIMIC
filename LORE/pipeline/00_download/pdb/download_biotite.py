import sys
from biotite.database import rcsb
from tqdm import tqdm


INPUT_COLUMN_IDX = 2
DOWNLOAD_FORMAT = "pdbx"

input_file = sys.argv[1]
output_dir = sys.argv[2]

id_list = []
with open(input_file,"r") as f:
    id_list = [line.split()[INPUT_COLUMN_IDX] for line in f]

binned_ids = {}
for idx in id_list:
    prefix = idx[:2]
    if prefix not in binned_ids:
        binned_ids[prefix] = []
    binned_ids[prefix].append(idx)

for prefix, idx_list in tqdm(binned_ids.items()):
    print(f"Fetching prefix {prefix} ({len(idx_list)} items)") 
    rcsb.fetch(idx_list, DOWNLOAD_FORMAT, target_path=f"{output_dir}/{prefix}")
