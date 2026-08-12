#%%

from lore import paths
from lore import logger
from lore.utils import sequence_tools

import pandas as pd
from tqdm import tqdm

from ete3 import NCBITaxa

# %% Load transcript annotations
transcripts_path = paths.get_path(data_type="data", stage="intermediate", name="transcripts", version="1.1")

logger.info("Loading transcript annotations")
transcripts_df = pd.read_parquet(transcripts_path / "final_merged_annotations.parquet")

#%% Add genome_feature_id column

transcripts_df['genome_feature_id'] = transcripts_df['genome'] + "_" + transcripts_df['feature_id']
# Reorder columns to have genome_feature_id as the first column and uniprot_id as the second
cols = ['genome_feature_id', 'uniprot_id'] + [col for col in transcripts_df.columns if col not in ['genome_feature_id', 'uniprot_id']]
transcripts_df = transcripts_df[cols]

#%% Add is_coding column

# Create 'is_coding' column based on the given conditions
transcripts_df.loc[:, 'is_coding'] = pd.Series(pd.NA, index=transcripts_df.index, dtype='boolean')  # Initialize with pd.NA and set dtype to boolean

# If 'pseudo' is in feature_type, set is_coding to False
mask_pseudo = transcripts_df['feature_type'].str.contains('pseudo', na=False)
transcripts_df.loc[mask_pseudo, 'is_coding'] = False

# For all other cases, set is_coding based on whether CDS_coordinates_length is NaN
mask_other = ~mask_pseudo
transcripts_df.loc[mask_other, 'is_coding'] = transcripts_df.loc[mask_other, 'CDS_coordinates'].notna()

#%% Add splice_region modality

# TODO: There are a small number (like 3) transcripts that have CDS outside of exon. 
# I think this can be safely ignored, but we should confirm the cause of these cases.
# I think this will be fixed when we correctly only take first level children

# Add splice_region_coordinates based on exon or CDS coordinates
tqdm.pandas(desc="Adding splice_region_coordinates")
transcripts_df.loc[:, 'splice_region_coordinates'] = transcripts_df.progress_apply(
    lambda row: row['exon_coordinates'] if row['exon_coordinates'] is not None else row['CDS_coordinates'], 
    axis=1
)

#%% Add taxonomy information

# Initialize the NCBI taxonomy database
ncbi = NCBITaxa()

# Get unique taxids from the dataframe
unique_taxids = transcripts_df['species_taxid'].unique()

# Create a dictionary to store taxonomy information
taxonomy_dict = {}

# Get taxonomy information for each taxid
for taxid in unique_taxids:
    try:
        # Get lineage
        lineage = ncbi.get_lineage(taxid)
        # Get names for each taxid in the lineage
        names = ncbi.get_taxid_translator(lineage)
        # Get ranks for each taxid in the lineage
        ranks = ncbi.get_rank(lineage)
        
        # Create a dictionary for each rank
        taxonomy = {ranks[tid]: names[tid] for tid in lineage if tid in ranks}
        
        # Add to our dictionary
        taxonomy_dict[taxid] = taxonomy
    except Exception as e:
        logger.warning(f"Error processing taxid {taxid}: {e}")
        taxonomy_dict[taxid] = {}

# Common ranks we want to extract
common_ranks = ['domain', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

# Create a function to get taxonomy info
def get_taxonomy_info(taxid, rank):
    if taxid in taxonomy_dict and rank in taxonomy_dict[taxid]:
        return taxonomy_dict[taxid][rank]
    return None

# Add taxonomy columns to the dataframe
for rank in common_ranks:
    tqdm.pandas(desc=f"Adding {rank} rank")
    transcripts_df[rank] = transcripts_df['species_taxid'].progress_apply(lambda x: get_taxonomy_info(x, rank))

#%% Get clean feature types

tqdm.pandas(desc="Getting clean feature types")
transcripts_df['clean_feature_type'] = transcripts_df.progress_apply(
    lambda x: sequence_tools.get_clean_feature_type(x), axis=1
)

#%% Make clean_feature_type other if count < 10,000
feature_type_counts = transcripts_df['clean_feature_type'].value_counts()
tqdm.pandas(desc="Filtering feature types")
transcripts_df['clean_feature_type'] = transcripts_df['clean_feature_type'].progress_apply(
    lambda x: x if feature_type_counts.get(x, 0) >= 10000 else 'other'
)

#%% Get kingdom_simplified

major_kingdoms = ['Pseudomonadati', 'Bacillati', 'Metazoa', 'Viridiplantae', 'Methanobacteriati', 'Fungi']

# Create new column with simplified kingdom classification
tqdm.pandas(desc="Simplifying kingdom classification")
transcripts_df['kingdom_simplified'] = transcripts_df['kingdom'].progress_apply(
    lambda x: x if x in major_kingdoms else 'Other'
)

#%% Save the cleaned transcripts DataFrame to transcripts_path

logger.info(f"Saving cleaned transcripts to {transcripts_path / 'final_merged_annotations.parquet'}")
transcripts_df.to_parquet(transcripts_path / 'final_merged_annotations.parquet', index=False)
logger.info(f"Saved {len(transcripts_df)} transcript records")
