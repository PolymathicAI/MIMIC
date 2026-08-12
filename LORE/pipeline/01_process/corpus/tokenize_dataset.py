#!/usr/bin/env python3
"""Tokenize common-pile/pubmed_filtered corpus"""

from transformers import AutoTokenizer
from datasets import load_dataset
import argparse
import os
import tempfile
from pathlib import Path
import time
from lore.paths import get_path

MODEL_ID = "dmis-lab/biobert-base-cased-v1.2"
TEXT_COLUMN_NAME = "text"  # The name of the column that contains the text in the raw dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenizer parser args.")
    parser.add_argument(
        "--num_proc",
        type=int,
        default=4,
        help="Number of processes to use for tokenization.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500,
        help="Batch size for tokenization.",
    )
    parser.add_argument(
        "--cache_path",
        type=Path,
        default=Path(tempfile.gettempdir()) / "lore_corpus_cache",
        help="HF cache path.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split from the raw dataset.",
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Version identifier for the output dataset.",
    )
    return parser.parse_args()


def main() -> None:

    # Args
    args = parse_args()
    cache_path = args.cache_path
    split = args.split

    # Configure HuggingFace to use specified cache directory
    os.environ["HF_DATASETS_CACHE"] = str(cache_path)
    os.environ["HF_HUB_CACHE"] = str(cache_path)
    os.environ["HF_HOME"] = str(cache_path)

    # Get tokenizer and load dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    dataset = load_dataset(
        "parquet",
        data_files=str(cache_path / "dataset.parquet"),
        split=split,
    )

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        """Helper to tokenize a batch of rows"""
        return tokenizer(
            batch[TEXT_COLUMN_NAME],
            padding=False,
            truncation=False,
        )

    # Track tokenization time
    time_start = time.time()

    # Tokenize dataset
    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=[c for c in dataset.column_names if c != "text"],
        desc="Tokenizing",
    )
    time_end = time.time()
    print(f"Tokenization took {time_end - time_start:.2f} seconds.")

    # Save tokenized dataset
    output_path = get_path("data", "intermediate", "corpus", version=args.version, fmt="parquet")
    output_path.mkdir(parents=True, exist_ok=True)
    tokenized.to_parquet(str(output_path / "tokenized_ds.parquet"))

    # Track saving time
    time_save_end = time.time()
    print(f"Saving tokenized dataset took {time_save_end - time_end:.2f} seconds.")


if __name__ == "__main__":
    main()
