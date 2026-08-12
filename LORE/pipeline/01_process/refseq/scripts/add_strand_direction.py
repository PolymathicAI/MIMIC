#%%

import os 

import pandas as pd

from lore import paths
from lore import logger

import gffutils
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

#%%

transcripts_path = paths.get_path(data_type="data", stage="intermediate", name="transcripts", version="1.0")
refseq_path = paths.get_path(data_type="data", stage="intermediate", name="refseq", version="229")
gencode_path = paths.get_path(data_type="data", stage="intermediate", name="gencode")
#%%
# Load the isoform annotations

logger.debug("Loading transcript annotations")
transcripts_df = pd.read_parquet(transcripts_path / "final_merged_annotations.parquet")


#%%
genome_groups = transcripts_df.groupby(['genome'])
transcript_dfs = [group for _, group in genome_groups]

logger.debug(f"Split matched transcripts into {len(transcript_dfs)} genome groups")

#%%

def process_genome(genome_group):
    """Process all rows for a single genome"""
    genome = genome_group['genome'].iloc[0]
    
    # Open the gffutils database for the current genome
    db_path = refseq_path / genome / "annotations.db"

    if genome == 'GCF_000001635.27':
        db_path = (gencode_path / "mouse") / "gencode.vM36.annotations.db"
    elif genome == 'GCF_000001405.40':
        db_path = (gencode_path / "human") / "gencode.v47.chr_patch_hapl_scaff.annotations.db"

    # Check if path exists
    if not os.path.exists(db_path):
        print(f"Database not found for {genome} at {db_path}")
        # Return the original group with None for strand
        genome_group['strand'] = None
        return genome_group
    
    # Load the database using gffutils
    db = gffutils.FeatureDB(str(db_path))

    all_features = []
    
    if genome == 'GCF_000001635.27' or  genome == 'GCF_000001405.40':
        all_features.extend(db.features_of_type('transcript'))

    # Collect features for each type
    for feature_type in genome_group['feature_type'].unique():
        all_features.extend(db.features_of_type(feature_type))
    
    feature_strand_map = {
        feature.id: feature.strand
        for feature in all_features
    }
    
    # Use .map() for efficient strand assignment
    genome_group['strand'] = genome_group['feature_id'].map(feature_strand_map)
    
    return genome_group
#%%

with ProcessPoolExecutor(max_workers=32) as exe:
    results = list(tqdm(exe.map(process_genome, transcript_dfs), total=len(genome_groups), desc="Processing Genomes"))


# Combine all results back into a single dataframe
isoform_df = pd.concat(results, ignore_index=True)
logger.debug(f"Combined results into a single dataframe with {len(isoform_df)} rows")

#%%

# Save the updated dataframe to a new parquet file
output_path = transcripts_path / "final_merged_annotations_with_strand_full.parquet"
isoform_df.to_parquet(output_path, index=False)
logger.debug(f"Saved updated annotations with strand direction to {output_path}")

