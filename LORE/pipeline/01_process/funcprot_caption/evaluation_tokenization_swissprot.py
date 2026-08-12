"""Apply different tokenizers on swissprot captions corpus and calculate stats."""

import json
import re
from collections import defaultdict
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import AutoTokenizer

LIST_TOKENIZERS = [
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "trained_tokenizer/wordpiece.json",
]


class BetterTokenizer(AutoTokenizer):
    def __init__(self, tokenizer_name):
        self.tokenizer_name = tokenizer_name
        if "wordpiece" in tokenizer_name:
            self.tokenizer = Tokenizer.from_file(tokenizer_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_batch(self, batch):
        if "wordpiece" in self.tokenizer_name:
            # For WordPiece tokenizer, we need to handle the tokenization differently
            return {"input_ids": self.tokenizer.encode(batch["funcprot_caption"]).ids}
        else:
            return self.tokenizer(
                batch["funcprot_caption"],
                truncation=False,
                padding=False,
            )


def calculate_stats(tokenized_captions):
    """Calculate statistics for tokenized captions."""

    # Calculate length of tokenized captions
    lengths = [len(caption) for caption in tokenized_captions]
    mean_length = sum(lengths) / len(lengths) if lengths else 0

    all_tokens = [token for caption in tokenized_captions for token in caption]
    num_tokens = len(all_tokens)
    unique_tokens = set(all_tokens)
    num_unique_tokens = len(unique_tokens)

    return {
        "min_length": min(lengths),
        "max_length": max(lengths),
        "quantiles": {
            "0.25": pd.Series(lengths).quantile(0.25),
            "0.5": pd.Series(lengths).quantile(0.5),
            "0.75": pd.Series(lengths).quantile(0.75),
            "0.95": pd.Series(lengths).quantile(0.95),
            "0.99": pd.Series(lengths).quantile(0.99),
        },
        "mean_length": mean_length,
        "num_tokens": num_tokens,
        "num_unique_tokens": num_unique_tokens,
        "lengths": lengths,
    }


def plot_results(results, output_folder: Path):
    """Plot the results of tokenization statistics."""
    plt.figure(figsize=(10, 6))
    for tokenizer_name, stats in results.items():
        plt.hist(
            stats["lengths"], label=tokenizer_name, bins=50, alpha=0.5, density=True
        )

    plt.xlabel("Caption token Length")
    plt.ylabel("Frequency")
    plt.title("Distribution of Caption Token Lengths")
    plt.legend()
    #plt.xscale("log")
    plt.savefig(output_folder / "caption_token_length_distribution.png")
    plt.close()


@click.command()
@click.option("--input_parquet", "-i", type=Path, help="Input parquet.")
@click.option("--output_folder", "-o", type=Path, help="Output folder")
def main(input_parquet: Path, output_folder: Path):
    """Tokenize and save as dataset swissprot."""

    # Load sequences from a parquet file (preprocessed)
    dataset = load_dataset("parquet", data_files=str(input_parquet))

    # Perform experiments with tokenizers
    results = {}
    for tokenizer_name in LIST_TOKENIZERS:
        print(f"Processing tokenizer: {tokenizer_name}")
        tokenizer = BetterTokenizer(tokenizer_name)
        tok_dataset = dataset.map(tokenizer.tokenize_batch, batched=False, num_proc=8)

        tok_all_captions = tok_dataset["train"]["input_ids"]
        tok_stats = calculate_stats(tok_all_captions)
        results[tokenizer_name] = tok_stats

    # Save and plot results
    plot_results(results, output_folder)

    print("Tokenization statistics:")
    for tokenizer_name, stats in results.items():
        print(f"Tokenizer: {tokenizer_name}")
        print(f"Min Length: {stats['min_length']}")
        print(f"Max Length: {stats['max_length']}")
        print(f"Mean Length: {stats['mean_length']}")
        print(f"Num Tokens: {stats['num_tokens']}")
        print(f"Num Unique Tokens: {stats['num_unique_tokens']}")
        print(f"Quantiles: {stats['quantiles']}\n")

    with open(output_folder / "tokenization_stats.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
