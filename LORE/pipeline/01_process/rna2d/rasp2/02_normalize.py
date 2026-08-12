"""
Normalize the rasp2 values to be between 0 and 1 in each context group.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import duckdb
from lore import paths
from lore import logger

_data_version = "1.4_unnormalized"
_output_version = "1.4"

# Read the input parquet file
path_in = paths.get_path("data", "modality", "rasp2", _data_version, "parquet") / "dataset.parquet"
logger.info(f"Using DuckDB to read from {path_in}")
con = duckdb.connect()
query = f"SELECT * FROM read_parquet('{path_in}')"
df = con.execute(query).df()

logger.info(f"Loaded {len(df):,} rows with columns: {df.columns.tolist()}")
logger.info(f"Unique contexts: {df['context'].nunique()}")

# Normalize by context
def normalize_group(group_df, context_name):
    """Normalize rasp2 values for a single context group."""
    
    # Collect all unmasked values from this context group (OPTIMIZED)
    unmasked_arrays = [
        rasp_array.compressed() 
        for rasp_array in group_df['rasp2'].values
        if isinstance(rasp_array, np.ma.MaskedArray) and rasp_array.count() > 0
    ]
    
    if not unmasked_arrays:
        logger.warning(f"Context '{context_name}': No unmasked values found, skipping")
        return group_df
    
    all_unmasked_values = np.concatenate(unmasked_arrays)
    min_val = all_unmasked_values.min()
    max_val = all_unmasked_values.max()
    
    logger.info(f"Context '{context_name[-10:]}': min={min_val:.4f}, max={max_val:.4f}, n_values={len(all_unmasked_values):,}")
    
    # Check if already normalized (close to 0.0-1.0)
    # Using a tolerance of 0.05 to determine "close to"
    if np.isclose(min_val, 0.0, atol=0.05) and np.isclose(max_val, 1.0, atol=0.05):
        logger.info(f"Context '{context_name[-10:]}': Already normalized, skipping")
        return group_df
    
    # Need to normalize: scale to [0, 1]
    logger.info(f"Context '{context_name[-10:]}': Normalizing to [0, 1] range")
    
    normalized_rasp2 = []
    for rasp_array in group_df['rasp2'].values:
        if isinstance(rasp_array, np.ma.MaskedArray):
            # Scale only the unmasked values (FIXED)
            data = rasp_array.data.copy()
            mask = rasp_array.mask.copy()
            
            # Apply min-max normalization ONLY to unmasked values
            unmasked_positions = ~mask
            if max_val > min_val:
                data[unmasked_positions] = (data[unmasked_positions] - min_val) / (max_val - min_val)
            else:
                # All values are the same, set unmasked positions to 0.5
                data[unmasked_positions] = 0.5
            
            # Reconstruct masked array with same mask
            normalized_array = np.ma.MaskedArray(data=data, mask=mask, fill_value=rasp_array.fill_value)
            normalized_rasp2.append(normalized_array)
        else:
            normalized_rasp2.append(rasp_array)
    
    # Create a copy of the group with normalized values (FIXED)
    result = group_df.copy()
    result['rasp2'] = normalized_rasp2
    
    return result

# Process each context group
logger.info("Starting normalization by context...")
grouped = df.groupby('context', group_keys=False)
normalized_df = pd.concat([
    normalize_group(group, context) 
    for context, group in tqdm(grouped, desc="Normalizing contexts")
])

# Verify the normalization (OPTIMIZED)
logger.info("Verifying normalization...")
for context in tqdm(normalized_df['context'].unique(), desc="Verifying normalization"):
    context_data = normalized_df[normalized_df['context'] == context]
    unmasked_arrays = [
        arr.compressed() for arr in context_data['rasp2'].values 
        if isinstance(arr, np.ma.MaskedArray) and arr.count() > 0
    ]
    
    if unmasked_arrays:
        all_unmasked = np.concatenate(unmasked_arrays)
        logger.info(f"Context '{context[-10:]}': After normalization - min={all_unmasked.min():.4f}, max={all_unmasked.max():.4f}")

# Convert MaskedArrays to regular arrays with NaN in masked positions for parquet compatibility
logger.info("Converting MaskedArrays to regular arrays with NaN in masked positions...")
rasp2_for_saving = []
for rasp_array in tqdm(normalized_df['rasp2'].values, desc="Converting to NaN format"):
    if isinstance(rasp_array, np.ma.MaskedArray):
        # Convert to regular array with NaN in masked positions
        regular_array = np.ma.filled(rasp_array, np.nan)
        rasp2_for_saving.append(regular_array)
    else:
        rasp2_for_saving.append(rasp_array)

# Create a new DataFrame with converted arrays
save_df = normalized_df.copy()
save_df['rasp2'] = rasp2_for_saving

logger.info("Masked positions are now NaN values")

# Save the normalized data
path_out = paths.get_path("data", "modality", "rasp2", _output_version, "parquet") / "dataset.parquet"
path_out = str(path_out)
logger.info(f"Saving normalized data to {path_out}")
os.makedirs(os.path.dirname(path_out), exist_ok=True)

save_df.to_parquet(path_out, index=False)
logger.info("Normalization complete!")