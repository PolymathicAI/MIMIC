#%%
import pandas as pd

from lore import paths
from lore import logger


# %%

logger.info("Loading transcripts data")
transcripts_file = paths.get_path(data_type="data", stage="intermediate", name="transcripts", version="1.0") / "final_merged_annotations.parquet"
logger.info(f"Reading transcripts from: {transcripts_file}")
transcripts = pd.read_parquet(transcripts_file)
logger.info(f"Loaded {len(transcripts)} transcripts")



#%%

# Create 'coding' column based on the given conditions
transcripts['coding'] = pd.Series(pd.NA, index=transcripts.index, dtype='boolean')  # Initialize with pd.NA and set dtype to boolean

# If 'pseudo' is in feature_type, set coding to False
mask_pseudo = transcripts['feature_type'].str.contains('pseudo', na=False)
transcripts.loc[mask_pseudo, 'coding'] = False

# For all other cases, set coding based on whether CDS_coordinates_length is NaN
mask_other = ~mask_pseudo
transcripts.loc[mask_other, 'coding'] = transcripts.loc[mask_other, 'CDS_coordinates'].notna()
#%%
# Print coding value counts
logger.info("Coding value counts:")
coding_counts = transcripts['coding'].value_counts(dropna=False)
for value, count in coding_counts.items():
    logger.info(f"  {value}: {count}")


output_file = paths.get_path(data_type="data", stage="modality", name="coding", version="1.0", fmt="parquet") / 'dataset.parquet'
logger.info(f"Saving coding modality to {output_file}")

transcripts[['genome_feature_id', 'coding']].to_parquet(output_file)
logger.info(f"Successfully saved coding modality to {output_file}")

# %%
