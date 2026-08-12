# %%
import duckdb
import json
import pandas as pd
import time
import os

from lore.paths import get_path

# 1. Define paths
VERSION = "1.4"
_atac = get_path("data", "modality", "atac", VERSION, fmt="parquet")
mouse_path = str(_atac / "mouse" / "dataset.parquet")
human_path = str(_atac / "human" / "dataset.parquet")
json_mapping_path = "answer_atac.json"
output_path = "/tmp/final_merged_atac.parquet"
temp_dir = "/tmp/merge_mem"

# Ensure temp directory exists
os.makedirs(temp_dir, exist_ok=True)

# 2. Load JSON mapping
with open(json_mapping_path, 'r') as f:
    data = json.load(f)
mapping_df = pd.DataFrame(list(data.items()), columns=['raw_condition', 'context'])

# delete the output path
# if os.path.exists(output_path):
    # os.remove(output_path)

# %%
# 3. Initialize & Configure for Low Memory
con = duckdb.connect(database=':memory:')

# --- MEMORY SAFEGUARDS ---
con.execute(f"SET temp_directory='{temp_dir}'")
# Force spilling to disk before the OS kills the process. Adjust '16GB' to your environment.
con.execute("SET memory_limit='1800GB'") 
# Disable strict ordering to allow writing data as soon as a chunk is ready
con.execute("SET preserve_insertion_order=false")
# set numer of threads to 200
con.execute("SET threads=200")

# -------------------------

con.register('mapping_df', mapping_df)

# 4. Define Logic (Lazy View)
con.execute(f"""
CREATE OR REPLACE VIEW atac_processed AS 
WITH combined_data AS (
    SELECT 
        genome_feature_id, 
        atac, 
        concat(cellline, ', Mouse, Mus musculus') AS raw_condition
    FROM read_parquet('{mouse_path}')
    
    UNION ALL
    
    SELECT 
        genome_feature_id, 
        atac, 
        concat(cellline, ', Human, Homo sapiens') AS raw_condition
    FROM read_parquet('{human_path}')
)
SELECT 
    d.genome_feature_id,
    -- Transformation Logic
    array_to_string(
        list_transform(d.atac, x -> 
            CASE 
                WHEN x IS NULL OR isnan(x) THEN 'U'
                WHEN CAST(x AS INTEGER) = 0 THEN 'N'
                WHEN CAST(x AS INTEGER) = 10 THEN 'X'
                ELSE CAST(CAST(x AS INTEGER) AS VARCHAR)
            END
        ), 
        ''
    ) AS atac,
    m.context
FROM combined_data d
LEFT JOIN mapping_df m ON d.raw_condition = m.raw_condition
""")

# 5. Verify correctness: Fetch a few samples
print("\n--- SAMPLES CHECK ---")
try:
    samples = con.execute("""
        SELECT genome_feature_id, atac, context 
        FROM atac_processed 
        LIMIT 10000
    """).fetch_df()
    print(samples)
    
    # Simple logic check on the sample
    print(f"\nSample 'atac' string length: {len(samples.iloc[0]['atac'])}")
except Exception as e:
    print(f"Verification failed: {e}")
print("---------------------\n")

# get all the unique characters that appear in the atac column in the above samples
unique_chars = set()
for atac_string in samples['atac']:
    unique_chars.update(set(atac_string))
print(f"Unique characters in 'atac' column samples: {sorted(unique_chars)}")

# %%
# 6. Write to Disk
# Only proceed if verification didn't crash
user_input = input("Do samples look correct? (y/n): ")
if user_input.lower() == 'y':
    print("Starting streaming write...")
    start = time.time()
    
    con.execute(f"""
        COPY (SELECT * FROM atac_processed) 
        TO '{output_path}' 
        (
            FORMAT 'parquet', 
            ROW_GROUP_SIZE 100000  -- Small row groups to keep memory low per chunk
        )
    """)
    
    elapsed = time.time() - start
    print(f"Write finished. Time taken: {elapsed:.2f} seconds")
else:
    print("Aborted.")
# %%

# get the number of rows in each of mouse, human and output file
mouse_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{mouse_path}')").fetchone()[0]
human_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{human_path}')").fetchone()[0]
output_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output_path}')").fetchone()[0]
print(f"Mouse rows: {mouse_count}, Human rows: {human_count}, Output rows: {output_count}")
# verify that mouse_count + human_count == output_count
if mouse_count + human_count == output_count:
    print("Row count verification passed.")
# %%
