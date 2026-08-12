#!/usr/bin/env python3
"""Download common-pile/pubmed_filtered corpus"""

import argparse
import gzip
import json
import tempfile
from pathlib import Path
from lore import logger
from huggingface_hub import snapshot_download
import pyarrow as pa
import pyarrow.parquet as pq

from tqdm import tqdm
import os

REPO_ID = "common-pile/pubmed_filtered"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and process pubmed_filtered corpus."
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "lore_corpus_cache",
        help="Directory where the HF dataset repo will be cached.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path(tempfile.gettempdir()) / "lore_corpus",
        help="Path to the output directory of final parquet file.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=50_000,
        help="Number of rows to buffer before each Parquet write.",
    )
    return parser.parse_args()


def main() -> None:

    # Args
    args = parse_args()

    # Set paths
    cache_dir = Path(args.cache_dir)
    output_path = Path(args.output_path)

    # Avoid HF saving cache in my HOME
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    os.environ["HF_HOME"] = str(cache_dir)

    # Download the dataset repo by shards
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        cache_dir=str(cache_dir),
        allow_patterns=["licensed_pubmed-*.json.gz"],
    )
    dataset_dir = Path(local_dir)

    # Find all shard paths
    shard_paths = sorted(dataset_dir.glob("licensed_pubmed-*.json.gz"))
    logger.info(f"Found {len(shard_paths)} shards in {dataset_dir}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Writing to: {output_path}")

    # Iterate shards and write to a single dataset.parquet
    buffer: list[str] = []  # Buffer for rows
    writer = None  # Parquet writer (this is standard)

    for shard in tqdm(shard_paths, desc="Processing shards..."):
        # Open each shard
        with gzip.open(shard, "rt", encoding="utf-8") as f:  # utf-8 just in case
            for line in f:
                # Process each line
                line = line.strip()
                if not line:
                    # Skip empty lines
                    continue
                # Examples are JSONL format
                obj = json.loads(line)
                # Keep only text column
                buffer.append(obj["text"])

                # If buffer is full, write it to file
                if len(buffer) >= args.chunk_size:
                    # Convert buffer to arrow format and write parquet
                    arr = pa.array(buffer, type=pa.string())
                    table = pa.Table.from_arrays([arr], names=["text"])
                    if writer is None:
                        # Init writer for first time
                        writer = pq.ParquetWriter(
                            str(output_path / "dataset.parquet"), table.schema
                        )
                    # Write to file
                    writer.write_table(table)
                    # Clean buffer
                    buffer.clear()

    # Flush remaining rows
    if buffer:
        arr = pa.array(buffer, type=pa.string())
        table = pa.Table.from_arrays([arr], names=["text"])
        if writer is None:
            # Init writer for first time
            writer = pq.ParquetWriter(str(output_path / "dataset.parquet"), table.schema)
        writer.write_table(table)
        # Clean buffer
        buffer.clear()

    # Close writer
    if writer is not None:
        writer.close()

    logger.info(f"Finished.")


if __name__ == "__main__":
    main()
