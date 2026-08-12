#%% 
import os
import multiprocessing
from lore import paths

from loguru import logger

import pandas as pd
import gffutils
#%%

intermediate_path = paths.get_path(data_type="data", stage="intermediate", name="refseq", version="229")

gcf_dirs = [d for d in os.listdir(intermediate_path) if d.startswith("GCF") and os.path.isdir(os.path.join(intermediate_path, d))]

logger.info(f"Found {len(gcf_dirs)} genomes in {intermediate_path}")

#%%


def create_gffutils_db(genome):

    annotation_path = os.path.join(intermediate_path, genome, "annotations.gff")
    db_path = os.path.join(intermediate_path, genome, "annotations.db")

    if os.path.exists(annotation_path):

        logger.info(f"Processing genome: {genome}")

        # Load the GFF file into a gffutils database
        if not os.path.exists(db_path):
            # Create a new gffutils database
            logger.info(f"Creating gffutils database at {db_path}")
            db = gffutils.create_db(annotation_path, dbfn=db_path, force=True, keep_order=True)
            logger.info(f"Database created successfully at {db_path}")
        else:
            logger.info(f"Loading existing gffutils database from {db_path}")
            db = gffutils.FeatureDB(db_path, keep_order=True)

    else:
        logger.warning(f"Annotation file not found for genome: {genome}")
        db = None
    
    return db

def get_gff_info(genome):

    db = create_gffutils_db(genome)

    if db:
        features = list(db.features_of_type("mRNA"))
        if not features:
            logger.info(f"No mRNA features found for genome: {genome}. Looking for gene features instead.")
            features = list(db.features_of_type("gene"))
            if not features:
                logger.warning(f"No gene features found for genome: {genome}.")
                return []
        logger.info(f"Found {len(features)} transcripts for genome: {genome}")
        
    else:
        logger.warning(f"No database available for genome: {genome}")
        return []
    

    data = []
    for transcript in features:
        cds_children = list(db.children(transcript, featuretype="CDS"))
        if not cds_children:
            continue

        cds_ids = [cds.id for cds in cds_children]

        processed_cds_ids = [
            "_".join(cds_id[4:].split("_")[:-1]) if len(cds_id.split("_")) == 3 else cds_id[4:]
            for cds_id in cds_ids
        ]

        if len(set(processed_cds_ids)) != 1:
            logger.warning(f"CDS IDs for transcript {transcript.id} are not identical: {processed_cds_ids}")
            continue

        cds_id = processed_cds_ids[0]    

        transcript_length = len(transcript)

        if transcript.featuretype == "mRNA":
            gene_id = transcript.attributes.get("Parent", [""])[0]
        else:
            gene_id = transcript.id

        data.append({
            "genome": genome,
            "transcript_id": transcript.id,
            "gene_id": gene_id,
            "transcript_length": transcript_length,
            "cds_id": cds_id
        })


    df = pd.DataFrame(data)
    output_path = os.path.join(intermediate_path, genome, "transcript_info.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Transcript information saved to {output_path}")

    
    return df

with multiprocessing.Pool() as pool:
    results = pool.map(get_gff_info, gcf_dirs)

    # Merge all results into a single DataFrame
    merged_df = pd.concat(results, ignore_index=True)

    # Save the merged DataFrame to a CSV file
    merged_output_path = os.path.join(intermediate_path, "merged_transcript_info.csv")
    merged_df.to_csv(merged_output_path, index=False)
    logger.info(f"Merged transcript information saved to {merged_output_path}")