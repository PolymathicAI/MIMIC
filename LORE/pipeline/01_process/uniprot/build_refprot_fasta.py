import gzip
import pandas as pd
import biotite.sequence.io.fasta as fasta
from lore.paths import get_path
from lore import logger
from tqdm import tqdm

# Paths
refseq_fasta = get_path(data_type="data", stage="intermediate", name="transcripts", version="1.2") / "refseq.fasta.gz"
allprot_fasta = get_path(data_type="data", stage="intermediate", name="afdb", version="v4_70", fmt="mmseqs") / "afdb70_ur90.fasta.gz"
parquet_path = get_path(data_type="data", stage="intermediate", name="transcripts", version="1.2") / "refprot.parquet"
out_fasta = get_path(data_type="data", stage="intermediate", name="transcripts", version="1.2") / "refprot.fasta.gz"

PROTEIN_COLUMN = "uniprot_id"
DNA_COLUMN = "genome_feature_id"

# Read parquet
ids_df = pd.read_parquet(parquet_path)
logger.info(f"Read {len(ids_df)} records from {parquet_path}.")

gzf_in = gzip.open(allprot_fasta, "rt")
gzf_out = gzip.open(out_fasta, "wt")

def iter_in_fasta(gzf, df, column_name):
    i = 0
    for header, seq in tqdm(fasta.FastaFile.read_iter(gzf)):
        seq_id = header.split(" ")[0].split("_")[1]
        if seq_id in df[column_name].values:
            new_record = (seq_id, seq)
            i += 1
            yield new_record
    logger.info(f"Filtered {i} sequences.")

with gzf_in, gzf_out:
    out_fasta = fasta.FastaFile()
    out_fasta.write_iter(gzf_out, iter_in_fasta(gzf_in, ids_df, PROTEIN_COLUMN))

logger.info(f"Wrote filtered fasta to {out_fasta}.")