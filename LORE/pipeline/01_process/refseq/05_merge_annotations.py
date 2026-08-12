#!/usr/bin/env python3
"""
RefSeq Processing Pipeline - Step 5: Merge RefSeq, GENCODE, and RASP Annotations

This script combines RefSeq, GENCODE, and RASP annotation data into a unified transcript database
with taxonomic information, UniProt ID mappings, and enhanced metadata. It creates a 
comprehensive resource for downstream biological analysis and machine learning tasks.

Input:
- RefSeq annotated_genome.parquet files from step 2
- GENCODE annotated_genome.parquet files from step 4
- RASP annotated_genome.parquet files from step 2 (rna2d_rasp version)
- UniProt ID mapping file: intermediate/refseq/uniprot_refseq_ensembl_id.parquet
- RefSeq assembly summary metadata

Output:
- final_merged_annotations.parquet: Unified transcript database with source tracking
- Located in intermediate/transcripts/{VERSION}/final_merged_annotations.parquet
- VERSION set to 1.4, update manually to preserve record of version history

The script:
1. Loads and processes RefSeq annotations from all genome assemblies
2. Loads GENCODE annotations for human and mouse
3. Loads RASP annotations and merges them into appropriate processing streams:
   - RASP transcript features → processed like GENCODE data
   - RASP non-transcript features → processed like RefSeq data
4. Maps UniProt IDs to all transcript sources
5. Adds species metadata from NCBI assembly summaries
6. Enriches data with taxonomic hierarchy using NCBI taxonomy
7. Creates enhanced features:
   - is_coding: Boolean flag for protein-coding transcripts
   - splice_region_coordinates: Unified exon/CDS coordinates
   - feature_type_clean: Standardized feature type classifications
   - feature_type_simplified: Simplified feature types for major categories
   - kingdom_simplified: Major kingdom groupings
   - source: Origin of the annotation (refseq, gencode, rasp)

Key Features:
- Parallel processing of RefSeq and RASP genomes for efficiency
- UniProt ID mapping for cross-database integration
- NCBI taxonomy integration for phylogenetic analysis
- Feature enhancement and standardization
- Comprehensive metadata enrichment
- Data quality filtering and validation
- Source tracking for all annotations

Data Integration:
- RefSeq: Broad taxonomic coverage with automated annotation
- GENCODE: High-quality manual annotation for human and mouse
- RASP: RNA structure prediction annotations
- UniProt: Protein database cross-references
- NCBI Taxonomy: Phylogenetic classification hierarchy

Usage:
    python 05_merge_annotations.py

Note: This script processes all available data from RefSeq, GENCODE, and RASP sources.
"""

#%%
import sys
import os
import subprocess
import argparse

from lore import paths
from lore import logger
from lore.utils import sequence_tools
from ete3 import NCBITaxa

import pandas as pd
from tqdm import tqdm
import concurrent.futures



#%%
#==================================================================================================
# INITIALIZATION AND SETUP
#==================================================================================================

logger.remove()
logger.add(sys.stderr, level="INFO")

# Parse command-line arguments
OUTPUT_VERSION = "1.4"
logger.info("Initializing paths")
gencode_path = paths.get_path(data_type="data", stage="intermediate", name="gencode")

#%%

logger.info("Getting list of directories in refseq, gencode, and rasp paths")

# Set up paths for all data sources
refseq_path = paths.get_path(data_type="data", stage="intermediate", name="refseq", version="229")
rasp_path = paths.get_path(data_type="data", stage="intermediate", name="refseq", version="rna2d_rasp")

# Get directories for each data source
refseq_dirs = [d for d in os.listdir(refseq_path) if d.startswith("GCF") and os.path.isdir(os.path.join(refseq_path, d))]
rasp_dirs = [d for d in os.listdir(rasp_path)]
gencode_dirs = ["human", "mouse"]

logger.info(f"Found {len(refseq_dirs)} RefSeq directories, {len(rasp_dirs)} RASP directories, and {len(gencode_dirs)} Gencode directories")

#%%

# %%
#==================================================================================================
# GENCODE DATA PROCESSING
#==================================================================================================
logger.info("Loading and processing gencode annotations")
gencode_dfs = []

for genome in gencode_dirs:
    logger.debug(f"Processing genome: {genome}")
    annotation_path = gencode_path / genome / "annotated_genome.parquet"
    
    logger.debug(f"Loading gencode annotations from {annotation_path}")
    gencode_annotations = pd.read_parquet(annotation_path)
    
    logger.debug(f"Adding genome identifier for {genome}")
    gencode_annotations["genome"] = "GCF_000001405.40" if genome == "human" else "GCF_000001635.27"
    
    # Add source column
    gencode_annotations["source"] = "gencode"
    
    gencode_dfs.append(gencode_annotations)

logger.info(f"Loaded {len(gencode_dfs)} gencode DataFrames")
merged_gencode_df = pd.concat(gencode_dfs, ignore_index=True)
logger.info("Merged gencode DataFrames into a single DataFrame")

#%%
# Cleaning up gencode DataFrame columns, removing trivial columns

logger.info("Dropping unnecessary columns from gencode DataFrame")
merged_gencode_df = merged_gencode_df.drop(columns=["feature_type", "root_type"])

logger.info("Renaming column 'transcript_type' to 'feature_type'")
merged_gencode_df = merged_gencode_df.rename(columns={"transcript_type": "feature_type"})
logger.info("Renaming column 'root_gene_type' to 'root_type'")
merged_gencode_df = merged_gencode_df.rename(columns={"root_gene_type": "root_type"})

#%%
#==================================================================================================
# REFSEQ DATA PROCESSING
#==================================================================================================
logger.info("Loading and processing refseq annotations")

def process_genome(genome, base_path, source_name):
    annotated_genome_path = base_path / genome / "annotated_genome.parquet"
    if os.path.exists(annotated_genome_path):
        try:
            df = pd.read_parquet(annotated_genome_path)
            df['genome'] = genome
            df['source'] = source_name
            return df
        except Exception as e:
            logger.error(f"Failed to load {annotated_genome_path}: {e}")
            return None
    else:
        logger.warning(f"File not found: {annotated_genome_path}")
        return None

# Process RefSeq data
logger.info("Processing RefSeq genomes")
refseq_dfs = []
with concurrent.futures.ProcessPoolExecutor() as executor:
    # Submit all tasks and create a dictionary mapping futures to their genomes
    future_to_genome = {executor.submit(process_genome, genome, refseq_path, "refseq"): genome 
                       for genome in refseq_dirs}
    
    # Process results as they complete
    for future in tqdm(concurrent.futures.as_completed(future_to_genome), 
                       total=len(future_to_genome), 
                       desc="Processing RefSeq genomes"):
        genome = future_to_genome[future]
        try:
            result = future.result()
            if result is not None:
                refseq_dfs.append(result)
        except Exception as e:
            logger.error(f"Error processing genome {genome}: {e}")

logger.info(f"Successfully processed {len(refseq_dfs)} RefSeq genomes")
logger.info("Merging refseq DataFrames")
merged_refseq_df = pd.concat(refseq_dfs, ignore_index=True) if refseq_dfs else pd.DataFrame()
logger.info("Merged refseq DataFrames into a single DataFrame")

#%%
#==================================================================================================
# RASP DATA PROCESSING AND MERGING
#==================================================================================================
logger.info("Loading and processing RASP annotations")

# Process RASP data
logger.info("Processing RASP genomes")
rasp_dfs = []
with concurrent.futures.ProcessPoolExecutor() as executor:
    # Submit all tasks and create a dictionary mapping futures to their genomes
    future_to_genome = {executor.submit(process_genome, genome, rasp_path, "rasp"): genome 
                       for genome in rasp_dirs}
    
    # Process results as they complete
    for future in tqdm(concurrent.futures.as_completed(future_to_genome), 
                       total=len(future_to_genome), 
                       desc="Processing RASP genomes"):
        genome = future_to_genome[future]
        try:
            result = future.result()
            if result is not None:
                rasp_dfs.append(result)
        except Exception as e:
            logger.error(f"Error processing genome {genome}: {e}")

logger.info(f"Successfully processed {len(rasp_dfs)} RASP genomes")
logger.info("Merging RASP DataFrames")
merged_rasp_df = pd.concat(rasp_dfs, ignore_index=True) if rasp_dfs else pd.DataFrame()
logger.info("Merged RASP DataFrames into a single DataFrame")

#%%
# Split RASP data and merge into appropriate processing streams
if not merged_rasp_df.empty:
    logger.info("Splitting RASP data and merging into appropriate processing streams")
    
    # Create separate DataFrames for transcript and non-transcript features
    rasp_transcript_df = merged_rasp_df[merged_rasp_df['feature_id'].str.contains('transcript')].copy()
    rasp_non_transcript_df = merged_rasp_df[~merged_rasp_df['feature_id'].str.contains('transcript')].copy()
    
    # Process transcript features (like GENCODE)
    if not rasp_transcript_df.empty:
        logger.info(f"Processing {len(rasp_transcript_df)} RASP transcript features")
        # Handle Gencode-style IDs by splitting on ":" and taking the second part
        rasp_transcript_df['feature_id'] = rasp_transcript_df['feature_id'].str.split(':').str[1]
        # Add to GENCODE DataFrame for unified processing
        merged_gencode_df = pd.concat([merged_gencode_df, rasp_transcript_df], ignore_index=True)
    
    # Process non-transcript features (like RefSeq)
    if not rasp_non_transcript_df.empty:
        logger.info(f"Processing {len(rasp_non_transcript_df)} RASP non-transcript features")
        # Add to RefSeq DataFrame for unified processing
        merged_refseq_df = pd.concat([merged_refseq_df, rasp_non_transcript_df], ignore_index=True)

#%%
#==================================================================================================
# UNIPROT ID MAPPING
#==================================================================================================

logger.info("Loading UniProt ID mapping")
id_mapping_file = paths.get_path(data_type="data", stage="intermediate", name="refseq") / "uniprot_refseq_ensembl_id.parquet"
id_mapping_df = pd.read_parquet(id_mapping_file)

#%%
logger.info("Mapping uniprot IDs to gencode DataFrame")
id_mapping_df_unique = id_mapping_df.drop_duplicates(subset=["Ensembl ID"])
ensembl_id_map = id_mapping_df_unique.set_index("Ensembl ID")["UniProtKB"].to_dict()
tqdm.pandas(desc="Mapping UniProt IDs to Gencode DataFrame")
merged_gencode_df["uniprot_id"] = merged_gencode_df["feature_id"].progress_apply(lambda x: ensembl_id_map.get(x) if pd.notna(x) else None)

logger.info("Mapping uniprot IDs to refseq DataFrame")
refseq_id_mapping_df_unique = id_mapping_df.drop_duplicates(subset=["Refseq ID"])
refseq_id_map = refseq_id_mapping_df_unique.set_index("Refseq ID")["UniProtKB"].to_dict()
# Extract the relevant part of CDS_id for mapping
tqdm.pandas(desc="Mapping UniProt IDs to RefSeq DataFrame")
merged_refseq_df["uniprot_id"] = merged_refseq_df["CDS_id"].progress_apply(lambda x: refseq_id_map.get(x[4:]) if pd.notna(x) else None)

# %%
#==================================================================================================
# MERGING DATASETS
#==================================================================================================

logger.info("Combining refseq and gencode DataFrames")
transcripts_df = pd.concat([merged_refseq_df, merged_gencode_df], ignore_index=True)

# Convert start/end columns to integer type
logger.info("Converting start/end columns to integer types")

# Find all columns with 'start' or 'end' in their name
start_end_columns = [col for col in transcripts_df.columns if 'start' in col.lower() or 'end' in col.lower()]

# Convert each identified column to integer type
for col in start_end_columns:
    # Use pd.to_numeric with errors='coerce' to handle non-numeric values
    # This will convert non-numeric values to NaN
    transcripts_df[col] = pd.to_numeric(transcripts_df[col], errors='coerce')
    
    # Then convert to integer type, preserving NaN values
    transcripts_df[col] = transcripts_df[col].astype('Int64')  # Int64 is pandas' nullable integer type

logger.info(f"Converted {len(start_end_columns)} start/end columns to integer type")

# Convert all pandas NA values to None for downstream processing compatibility
logger.info("Converting pandas NA values to None for downstream compatibility")
for col in transcripts_df.columns:
    if transcripts_df[col].dtype == 'Int64' or transcripts_df[col].dtype == 'boolean':
        # Replace pandas NA with None for nullable integer and boolean types
        transcripts_df[col] = transcripts_df[col].replace({pd.NA: None})
    elif transcripts_df[col].dtype == 'object':
        # For object columns, check if they contain pandas NA values
        if transcripts_df[col].isna().any():
            transcripts_df[col] = transcripts_df[col].replace({pd.NA: None})

logger.info("Successfully converted all pandas NA values to None")

#%%
#==================================================================================================
# ADDING METADATA
#==================================================================================================

logger.info("Reading refseq metadata")
download_path = paths.get_path(data_type="data", stage="downloads", name="refseq", version="229")
metadata_path = download_path / "assembly_summary_refseq.txt"

if not os.path.exists(metadata_path):
    assembly_summary_url = "https://ftp.ncbi.nlm.nih.gov/genomes/ASSEMBLY_REPORTS/assembly_summary_refseq.txt"
    logger.debug(f"Downloading assembly summary file from {assembly_summary_url}")
    subprocess.run(["wget", assembly_summary_url, "-O", metadata_path], check=True)
    logger.debug("Assembly summary file downloaded")
else:
    logger.debug("Assembly summary file already exists")

logger.info(f"Loading assembly summary from {metadata_path}")
df_assembly_summary = pd.read_csv(metadata_path, sep='\t', header=1)

#%%

# Add manual mapping for specific genome assemblies. 
# We use the reference genome for metadata but need the specific build for CAGE
# These mappings come from NCBI refseq, for older builds that are compatible with our cell-specific data
manual_mapping = {
    "GCF_000772875.2": "GCF_003339765.1", # Rhesus Monkey 
    "GCF_000002315.4": "GCF_016699485.2", # Chicken
    "GCF_000002285.3": "GCF_011100685.1", # Dog
    "GCF_000001895.5": "GCF_036323735.1"  # Rat
}

# RASP taxid mapping
# Manual Taxid Mapping from NCBI Taxonomy Database for RASP genomes
rasp_taxid_map = {
    "Bcereus": 1396,
    "Bsubtilis": 1423,
    "CHIKV": 37124,
    "CMV": 10359,
    "Dengue": 12637,
    "ecoli": 562,
    "GRCz11": 7955,
    "HCV": 11103,
    "hg38": 9606,
    "HIV": 11676,
    "IAV": 11320,
    "mm10": 10090,
    "Pputida": 303,
    "rice": 4530,
    "rotavirus": 28875,
    "SARS2": 2697049,
    "Senterica": 28901,
    "STMV": 12881,
    "Synechococcus": 1131,
    "TAIR10": 3702,
    "Vero": 60711,
    "yeast": 4932,
    "Y_pseudotuberculosis": 633,
    "Zika": 64320,
}

# Make sure each 'genome' value can be matched in the assembly summary
transcripts_df['match_genome'] = transcripts_df['genome'].apply(
    lambda x: manual_mapping.get(x, x)
)

logger.info("Adding species metadata to merged DataFrame")
transcripts_df = transcripts_df.merge(
    df_assembly_summary[['#assembly_accession', 'organism_name', 'species_taxid', 'asm_name']],
    left_on='match_genome',
    right_on='#assembly_accession',
    how='left'
)

# For RASP data, use the taxid mapping
rasp_mask = transcripts_df['source'].str.contains('rasp', na=False)
transcripts_df.loc[rasp_mask, 'species_taxid'] = transcripts_df.loc[rasp_mask, 'genome'].map(rasp_taxid_map)

# Restore original genome values for asm_name and assembly_accession for manually mapped genomes
# This ensures we keep the original GCF_XXX values instead of the matched assembly values
for original_genome, matched_genome in manual_mapping.items():
    mask = transcripts_df['genome'] == original_genome
    if mask.any():
        transcripts_df.loc[mask, 'asm_name'] = None

# Drop the temporary match column and assembly accession column
logger.info("Dropping temporary columns after merging")
transcripts_df = transcripts_df.drop(columns=['match_genome', '#assembly_accession'])

logger.info("Handling manually asm_name for manually mapped genomes")
# Manually set asm_name for specific genomes if still missing
asm_name_map = {
    "GCF_000772875.2": "rheMac8", # Rhesus Monkey
    "GCF_000002315.4": "galGal5", # Chicken
    "GCF_000002285.3": "canFam3", # Dog
    "GCF_000001895.5": "rn6"  # Rat
}

# Map the asm_name for the manually mapped genomes
# Only apply the mapping to rows with genomes that are in the asm_name_map
for genome_id, asm_name in asm_name_map.items():
    mask = transcripts_df['genome'] == genome_id
    if mask.any():
        transcripts_df.loc[mask, 'asm_name'] = asm_name

logger.info(f"Applied asm_name mapping to {sum(transcripts_df['genome'].isin(asm_name_map.keys()))} rows")


#%%
#==================================================================================================
# TAXONOMY PROCESSING
#==================================================================================================

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
    # Fill NaN values in taxonomy columns with 'Unknown'
    transcripts_df[rank] = transcripts_df[rank].fillna('Unknown')

#%% 
#==================================================================================================
# FEATURE ENHANCEMENT AND ORGANIZATION
#==================================================================================================

# Reorder columns in the final DataFrame
logger.info("Reordering columns in the final DataFrame")

# Create genome_feature_id by combining genome and feature_id
transcripts_df['genome_feature_id'] = transcripts_df['genome'] + "_" + transcripts_df['feature_id']

# For RASP data, add organism_name from species column
rasp_mask = transcripts_df['source'].str.contains('rasp', na=False)
transcripts_df.loc[rasp_mask, 'organism_name'] = transcripts_df.loc[rasp_mask, 'species']

front_cols = ['genome_feature_id', 'uniprot_id', 'genome', 'organism_name', 'species_taxid', 'source', 'asm_name']

cols = front_cols + [col for col in transcripts_df.columns if col not in front_cols]
transcripts_df = transcripts_df[cols]

#%% Add is_coding column

logger.info("Adding 'is_coding' column based on feature type and CDS coordinates")

# Create 'is_coding' column based on the given conditions
transcripts_df.loc[:, 'is_coding'] = pd.Series(pd.NA, index=transcripts_df.index, dtype='boolean')  # Initialize with pd.NA and set dtype to boolean

# If 'pseudo' is in feature_type, set is_coding to False
mask_pseudo = transcripts_df['feature_type'].str.contains('pseudo', na=False)
transcripts_df.loc[mask_pseudo, 'is_coding'] = False

# For all other cases, set is_coding based on whether CDS_coordinates_length is NaN
mask_other = ~mask_pseudo
transcripts_df.loc[mask_other, 'is_coding'] = transcripts_df.loc[mask_other, 'CDS_coordinates'].notna()

#%% Add splice_region modality

# Add splice_region_coordinates based on exon or CDS coordinates
tqdm.pandas(desc="Adding splice_region_coordinates")
transcripts_df.loc[:, 'splice_region_coordinates'] = transcripts_df.progress_apply(
    lambda row: row['exon_coordinates'] if row['exon_coordinates'] is not None else row['CDS_coordinates'], 
    axis=1
)

#%% Get clean feature types

tqdm.pandas(desc="Getting clean feature types")
transcripts_df['feature_type_clean'] = transcripts_df.progress_apply(
    lambda x: sequence_tools.get_clean_feature_type(x), axis=1
)

#%% Make clean_feature_type other if count < 10,000
feature_type_counts = transcripts_df['feature_type_clean'].value_counts()
tqdm.pandas(desc="Filtering feature types")
transcripts_df['feature_type_simplified'] = transcripts_df['feature_type_clean'].progress_apply(
    lambda x: x if feature_type_counts.get(x, 0) >= 10000 else 'other'
)

#%% Get kingdom_simplified

major_kingdoms = ['Pseudomonadati', 'Bacillati', 'Metazoa', 'Viridiplantae', 'Methanobacteriati', 'Fungi', 'Unknown']

# Create new column with simplified kingdom classification
tqdm.pandas(desc="Simplifying kingdom classification")
transcripts_df['kingdom_simplified'] = transcripts_df['kingdom'].progress_apply(
    lambda x: x if x in major_kingdoms else 'Other'
)
#%%
#==================================================================================================
# FLANKING COORDINATES
#==================================================================================================
# Add flanking coordinates for feature extraction based on neighboring features, strand and organism

logger.info("Adding flanking coordinates for feature extraction")

# Set flanking sizes based on domain
def get_flanking_sizes(domain, strand='+'):
    if domain in ['Bacteria', 'Archaea']:
        upstream, downstream = 200, 200
    else:
        upstream, downstream = 1000, 300
    
    # Flip upstream/downstream for negative strand
    if strand == '-':
        return downstream, upstream
    return upstream, downstream


def calculate_flanking_regions(row, prev_row, next_row, upstream_size, downstream_size):
    MIN_FLANK = 200
    
    # Calculate start of flanking region (upstream direction)
    flank_start = row['root_start'] - upstream_size
    if prev_row is not None:
        # Ensure we don't overlap with previous gene
        flank_start = max(flank_start, prev_row['root_end'] + 1)
    flank_start = max(1, flank_start)  # Don't go below position 1
    
    # Ensure minimum flanking size upstream
    if row['root_start'] - flank_start < MIN_FLANK:
        flank_start = max(1, row['root_start'] - MIN_FLANK)
    
    # Calculate end of flanking region (downstream direction)
    flank_end = row['root_end'] + downstream_size
    if next_row is not None:
        # Ensure we don't overlap with next gene
        flank_end = min(flank_end, next_row['root_start'])
    
    # Ensure minimum flanking size downstream
    if flank_end - row['root_end'] < MIN_FLANK:
        flank_end = row['root_end'] + MIN_FLANK

    return row['genome_feature_id'], flank_start, flank_end

def process_chunk(group_df):
    results = []
    upstream_size, downstream_size = get_flanking_sizes(group_df.iloc[0]['domain'], group_df.iloc[0]['strand'])

    for i in range(len(group_df)):
        row = group_df.iloc[i]
        prev_row = group_df.iloc[i - 1] if i > 0 else None
        next_row = group_df.iloc[i + 1] if i < len(group_df) - 1 else None
        results.append(calculate_flanking_regions(row, prev_row, next_row, upstream_size, downstream_size))
    return results


# Group by genome, seqid, and strand into chunks with proper context
# We need to ensure flanking regions are calculated with respect to neighboring features
# This requires processing each group of features on the same chromosome and strand together
# We only want to consider adjacent genes on the same strand for flanking region calculations
# Genome = genome assembly, seqid = chromosome/scaffold, strand = +/-
grouped = transcripts_df.groupby(['genome', 'seqid', 'strand'])
chunks = [group for _, group in grouped]

# Process in parallel
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_chunk, chunks), total=len(chunks), desc="Calculating context-aware flanking coordinates"))

results_dict = {genome_feature_id: (flank_start, flank_end) 
                for sublist in results for genome_feature_id, flank_start, flank_end in sublist}


# Create a mapping series with progress tracking
flank_data = [results_dict.get(gf_id, (None, None)) for gf_id in tqdm(transcripts_df['genome_feature_id'], desc="Mapping flanking regions")]

# Assign to dataframe
transcripts_df[['flanking_start', 'flanking_end']] = pd.DataFrame(flank_data, index=transcripts_df.index)


#%%
#==================================================================================================
# OUTPUT AND SAVING
#==================================================================================================

logger.info("Ensuring the output directory exists")
output_dir = paths.get_path(data_type="data", stage="intermediate", name="transcripts", version=OUTPUT_VERSION)
os.makedirs(output_dir, exist_ok=True)


logger.info("Creating refprot DataFrame with uniprot_id, genome_feature_id, and feature_id")
df_refprot = transcripts_df[transcripts_df['uniprot_id'].notna()][['uniprot_id','genome_feature_id', 'feature_id']]
refprot_path = output_dir / "refprot.parquet"
logger.info(f"Saving refprot DataFrame to {refprot_path}")
df_refprot.to_parquet(refprot_path, index=False)

logger.info("Saving the final merged DataFrame to a parquet file")
output_path = output_dir / "final_merged_annotations_unfiltered.parquet"
transcripts_df.to_parquet(output_path, index=False)
logger.info(f"Final merged annotations saved to {output_path}")

#%%
#==================================================================================================
# FILTERING
#==================================================================================================
# We only want to include one instance of each organism in the final dataset.
# We have multiple sources for some organisms (e.g. human, mouse, B. Cereus)
#   Homo sapiens: taxid 9606
#   Mus musculus: taxid 10090
#   Danio rerio: taxid 7955
#   Arabidopsis thaliana: taxid 3702
#   Bacillus cereus: taxid 1396
# We will keep:
# - B. Cereus from RASP source (taxid 1396)
# - All other overlapping species from non-rasp source (taxids 9606, 10090, 7955, 3702)
# - All other non-overlapping species from both sources

logger.info("Filtering to include only one instance of each organism")
# Create masks for filtering according to overlap/non-overlap and rasp source
overlapping_taxids = [9606, 10090, 7955, 3702]  # Human, Mouse, Zebrafish, Arabidopsis
bcereus_taxid = 1396

# 1. Keep B. cereus ONLY from rasp source
bcereus_rasp_mask = (transcripts_df['species_taxid'] == bcereus_taxid) & (transcripts_df['source'] == 'rasp')

# 2. For other overlapping species, keep ONLY non-rasp sources
other_overlap_mask = (transcripts_df['species_taxid'].isin(overlapping_taxids)) & (transcripts_df['source'] != 'rasp')

# 3. For all non-overlapping species, include ALL sources (including rasp!)
non_overlap_taxids = ~(transcripts_df['species_taxid'].isin(overlapping_taxids + [bcereus_taxid]))
non_overlap_mask = non_overlap_taxids

# Combine all masks
other_mask = other_overlap_mask | non_overlap_mask

# Combine masks to filter the DataFrame
final_mask = bcereus_rasp_mask | other_mask
filtered_transcripts_df = transcripts_df[final_mask].copy()

logger.info(f"Filtered DataFrame now has {len(filtered_transcripts_df)} rows")

# Save the filtered DataFrame
filtered_output_path = output_dir / "final_merged_annotations.parquet"
filtered_transcripts_df.to_parquet(filtered_output_path, index=False)
logger.info(f"Filtered final merged annotations saved to {filtered_output_path}")
