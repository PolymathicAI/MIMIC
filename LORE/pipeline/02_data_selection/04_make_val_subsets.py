#!/usr/bin/env python3
"""
Data Selection Pipeline - Step 4: Create Specialized Validation Subsets

This script creates specialized validation subsets for specific tasks from the main validation
set. It generates balanced datasets optimized for transcription and coding classification tasks.

Input:
- config.yaml: Configuration file specifying data version and transcript info
- dataset.parquet: Master IDs table with splits from step 2 (used to access validation set)

Output:
- val_transcription.parquet: Validation subset for transcription tasks (sequences with amino acid sequences and CDS coordinates)
- val_coding_class.parquet: Balanced validation subset for coding classification (equal numbers of coding/non-coding sequences)

The script:
1. Creates transcription validation subset by filtering for:
   - Validation split sequences only
   - Non-null amino acid sequence length
   - CDS coordinates length >= 2
   
2. Creates coding classification validation subset by:
   - Filtering for sequences with RNA sequences and splice regions
   - Balancing coding vs non-coding sequences (equal representation)
   - Limiting to 20,000 sequences per class (if available)
   - Using random sampling with fixed seed for reproducibility

Filtering Criteria:
- Transcription: aa_seq_length IS NOT NULL AND CDS_coordinates_length >= 2
- Coding Classification: rna_seq_length IS NOT NULL AND splice_region_coordinates_length >= 1

Usage:
    python 04_make_val_subsets.py
"""

#%%
from lore import paths
from lore import logger
import yaml
import duckdb

def main():

    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    version = config["master_id_version"]
    seed = config["seed"]

    # these are the columns that we will keep from the master_ids
    id_cols = ["genome_feature_id", "uniprot_id"]

    # %%

    master_id_dir = paths.get_path(data_type="data", stage="intermediate", name="master_ids", version=version, fmt="parquet")
    master_ids_file = master_id_dir / "dataset.parquet"
    id_config_path = master_id_dir / "config.yaml"
    assert master_ids_file.exists(), f"Master IDs file {master_ids_file} does not exist. Please run the previous steps to create it."

    # compare the contents of the two config files and make sure they are the same
    with open(id_config_path, "r") as f:
        copied_config = yaml.safe_load(f)
    if copied_config != config:
        raise ValueError(f"Config file {id_config_path} does not match the current config. This implies a version conflict.")

    output_dir = paths.get_path(data_type="data", stage="modality", name="id", version=version, fmt="parquet")
    output_config_path = output_dir / "config.yaml"

    # compare the contents of the two config files and make sure they are the same
    with open(output_config_path, "r") as f:
        copied_config = yaml.safe_load(f)
    if copied_config != config:
        raise ValueError(f"Config file {id_config_path} does not match the current config. This implies a version conflict.")

    val_transcript_savefile = output_dir / "val_transcription.parquet"
    assert not val_transcript_savefile.exists(), f"Validation transcription file {val_transcript_savefile} already exists. Please remove it before running this script."
    val_coding_class_savefile = output_dir / "val_coding_class.parquet"
    assert not val_coding_class_savefile.exists(), f"Validation coding classification file {val_coding_class_savefile} already exists. Please remove it before running this script."
    val_funcprot_caption_savefile = output_dir / "val_funcprot_caption.parquet"
    assert not val_funcprot_caption_savefile.exists(), f"Validation funcprot caption file {val_funcprot_caption_savefile} already exists. Please remove it before running this script."

    # %% 
    ##############################################################################################
    ################################ Create transcription Val Set ################################
    ##############################################################################################
    logger.info("Creating transcription validation subset with transcripts where aa_seq_length is not None and CDS_coordinates_length >= 2")

    con = duckdb.connect(database=":memory:")

    # set the duckdb seed for reproducibility
    con.execute(f"SELECT setseed(0.{seed});")
    # add the master_val file as a view where split = 'val'
    con.execute(f"""
                CREATE VIEW master_val AS
                SELECT * EXCLUDE(split)
                FROM read_parquet('{master_ids_file}')
                WHERE split = 'val'
    """)
    con.execute(f"""
                COPY (
                    SELECT {', '.join(id_cols)}
                    FROM master_val
                    WHERE aa_seq_length IS NOT NULL
                    AND CDS_coordinates_length >= 2
                ) TO '{val_transcript_savefile}' (FORMAT 'parquet')
    """)
    logger.info(f"Saved transcription validation subset to {val_transcript_savefile}")

    # get the number of rows in this set
    val_transcript_count = con.execute(f"""
                SELECT COUNT(*) AS count
                FROM read_parquet('{val_transcript_savefile}')
    """).fetchone()[0]
    logger.info(f"Validation transcription subset created with {val_transcript_count:,} rows.")

    # %% 
    ##############################################################################################
    ######################### Create coding classification Val Set ###############################
    ##############################################################################################
    logger.info("Creating coding classification validation subset with even is_coding = True/False transcripts making sure splice_regions exist.")

    # first add the is_coding column to the master_val view
    logger.info("Accessing the is_coding column from the transcripts_info config.")
    con.execute(f"""
                CREATE VIEW transcriptions AS
                SELECT *
                FROM master_val
                WHERE rna_seq_length IS NOT NULL
                    AND splice_region_coordinates_length >= 1
    """)

    logger.info("Counting the number of is_coding = True/False in the validation set to balance the dataset.")
    con.execute(f"""
                SELECT is_coding, COUNT(*) AS count
                FROM transcriptions
                GROUP BY is_coding
    """)
    val_counts = con.fetchall()
    min_count = min(val_counts, key=lambda x: x[1])[1]
    log_string = ", ".join([f"{is_coding}: {count:,}" for is_coding, count in val_counts])
    logger.info(f"Validation set is_coding counts: {log_string}")
    if min_count > 20_000:
        logger.info(f"Clipping to 20_000.")
        min_count = 20_000
    else:
        logger.info(f"Using the minimum count {min_count:,} to balance the dataset.")

    # create two views for each is_coding value
    for is_coding_value in [True, False]:
        view_name = "coding_view" if is_coding_value else "non_coding_view"
        con.execute(f"""
            CREATE VIEW {view_name} AS
            SELECT {', '.join(id_cols)}
            FROM transcriptions
            WHERE is_coding = {is_coding_value}
            ORDER BY random()
            LIMIT {min_count}
        """)
        logger.info(f"Created view for is_coding = {is_coding_value} with {min_count:,} rows.")

    # now union the two views and save to file
    con.execute(f"""
                COPY (
                    SELECT {', '.join(id_cols)}
                    FROM coding_view
                    UNION ALL
                    SELECT {', '.join(id_cols)}
                    FROM non_coding_view
                ) TO '{val_coding_class_savefile}' (FORMAT 'parquet')
    """)
    logger.info(f"Saved coding classification validation subset to {val_coding_class_savefile}")

    # get the number of rows in this set
    val_coding_class_count = con.execute(f"""
                SELECT COUNT(*) AS count
                FROM read_parquet('{val_coding_class_savefile}')
    """).fetchone()[0]
    logger.info(f"Validation coding classification subset created with {val_coding_class_count:,} rows.")
    # %%

    ##############################################################################################
    ######################### Create funcprot_caption Val Set ###############################
    ##############################################################################################
    logger.info("Creating function prediction validation subset with transcripts where aa_seq_length is not None and funcprot_caption is not null")

    # set the duckdb seed for reproducibility
    con.execute("SELECT setseed(0.42);")
    con.execute(f"""
                COPY (
                    SELECT {', '.join(id_cols)}
                    FROM master_val
                    WHERE aa_seq_length >= 350
                    AND has_funcprot_caption = TRUE
                    ORDER BY random()
                    LIMIT 20000
                ) TO '{val_funcprot_caption_savefile}' (FORMAT 'parquet')
    """)
    logger.info(f"Saved function prediction captioning validation subset to {val_funcprot_caption_savefile}")

    # get the number of rows in this set
    val_funcprot_caption_count = con.execute(f"""
                SELECT COUNT(*) AS count
                FROM read_parquet('{val_funcprot_caption_savefile}')
    """).fetchone()[0]
    logger.info(f"Validation function prediction captioning subset created with {val_funcprot_caption_count:,} rows.")

    con.close()

if __name__ == "__main__":
    main()