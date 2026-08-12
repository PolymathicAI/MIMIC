"""Tokenize uniprot captions in all parquet files in preprocessing folder."""

import argparse, argcomplete
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from tokenizers import Tokenizer


def tokenize_batch(batch, tokenizer):
    """Tokenize batch of captions."""
    return {"input_ids": [tokenizer.encode(c).ids for c in batch["funcprot_caption"]]}


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize captions in parquet file using HuggingFace datasets."
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        required=True,
        help="Input Parquet file.",
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        required=True,
        help="Folder to save tokenized Parquet files.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=Path,
        required=True,
        help="Path to trained WordPiece tokenizer JSON.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=100, help="Batch size for tokenization."
    )
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    args.output_folder.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))

    parquet_file = args.input_file
    if not parquet_file.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {args.input_file}")

    print(f"Processing {parquet_file.name}")
    dataset = load_dataset("parquet", data_files=str(parquet_file), split=None)

    # Check correct format for .map() method
    if isinstance(dataset, DatasetDict | Dataset):
        ds_tok = dataset.map(
            lambda batch: tokenize_batch(batch, tokenizer),
            batched=True,
            batch_size=args.batch_size,
            num_proc=8,
        )
        output_file = args.output_folder / f"{parquet_file.stem}_tokenized.parquet"
        ds_tok.to_parquet(str(output_file))
        print(f"Saved tokenized file to {output_file}")

    else:
        raise ValueError(f"Unexpected dataset type: {type(dataset)}")

if __name__ == "__main__":
    main()
