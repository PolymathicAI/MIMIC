import sys
import pandas as pd
from tqdm import tqdm
from lore import logger

tqdm.pandas()

def trim_uniprot_id(uniprot_id):
    """Trim the uniprot name from a UniProt ID if present."""
    if "_" in uniprot_id:
        return uniprot_id.split('_')[1]
    return uniprot_id

def convert_mmseqs_tsv_to_parquet(input_tsv, output_parquet):
    # Read the TSV file using pandas
    logger.info(f"Reading TSV file from {input_tsv}")
    df = pd.read_csv(input_tsv, sep="\t", header=None, names=["cluster_centroid", "uniprot_id"])
    
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"Trimming UniProt from name")
    df["uniprot_id"] = df["uniprot_id"].progress_apply(trim_uniprot_id)
    df["cluster_centroid"] = df["cluster_centroid"].progress_apply(trim_uniprot_id)

    logger.info(f"Getting rid of duplicate rows")
    df = df.drop_duplicates()

    # Save the DataFrame to a Parquet file
    logger.info(f"Writing Parquet file to {output_parquet}")
    df.to_parquet(output_parquet, index=False)
    logger.info(f"Converted {input_tsv} to {output_parquet}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_mmseqs_tsv_to_parquet.py input.tsv output.parquet")
        sys.exit(1)
    
    input_tsv = sys.argv[1]
    output_parquet = sys.argv[2]
    
    convert_mmseqs_tsv_to_parquet(input_tsv, output_parquet)