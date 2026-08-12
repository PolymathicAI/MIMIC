
import argparse
import sys
from pathlib import Path
from lore.paths import get_path
from lore import logger
import pyarrow.dataset as ds
import pyarrow.parquet as pq

def main():
    parser = argparse.ArgumentParser(description="Combine tokenized chunked parquet files into a single parquet file.")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory containing chunked token parquet files (default: data/modality/prot_struct/v4_plddt_70/parquet/)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to output combined parquet file (default: dataset.parquet in input_dir)")
    args = parser.parse_args()

    if args.input_dir is None:
        input_dir = get_path("data", "modality", "prot_struct", version="v4_plddt_70", fmt="parquet")
    else:
        input_dir = Path(args.input_dir)


    if args.output_file is None:
        out_file = get_path("data", "modality", "prot_struct", version="afdb70_esm3_tok", fmt="parquet") / "dataset.parquet"
    else:
        out_file = Path(args.output_file)

    # Ensure output directory exists
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # List all *_tokens.parquet files except the output file
    logger.info(f"Looking for tokenized parquet files in {input_dir}...")
    parquet_files = [str(f) for f in input_dir.glob("*_tokens.parquet") if f.name != out_file.name]
    if not parquet_files:
        logger.error(f"No tokenized parquet files found in {input_dir}")
        sys.exit(1)

    # Use pyarrow.dataset to read and concatenate efficiently
    logger.info(f"Found {len(parquet_files)} tokenized parquet files. Combining into {out_file}...")
    dataset = ds.dataset(parquet_files, format="parquet")
    table = dataset.to_table()
    pq.write_table(table, out_file)
    logger.info(f"Saved combined tokens to {out_file}")

if __name__ == "__main__":
    main()
