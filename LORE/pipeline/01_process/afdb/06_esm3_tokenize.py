# Needs `mimic` installed for the ESM3 structure tokenizer, and a GPU.

import os
import argparse
import numpy as np
import torch
import datasets
from tqdm import tqdm
from pathlib import Path
from lore import logger
from lore.paths import get_path

from datasets import disable_progress_bar
disable_progress_bar()

from mimic.tokenizers.protein.structure.struct_tokenizer import ESM3StructureTokenizer

def token_iterator(chunk_path, tokenizer, debug=False):
    logger.info(f"Loading dataset from {chunk_path}...")
    ds = datasets.load_from_disk(chunk_path)
    logger.info(f"Loaded dataset with {len(ds)} entries")

    # take only first 1000 for debugging
    if debug:
        logger.warning("Debug mode: processing only first 1000 entries")
        ds = ds[:1000]

    for i, (upid, structure) in enumerate(zip(ds["uniprot_id"], ds["structure"])):        
        if i % 1000 == 0 and i > 0:
            logger.info(f"Processed {i} structures")

        # Tokenize the structure
        structure = np.array(structure)
        tokens = tokenizer.tokenize(structure)

        # Write the tokens to the output file
        yield {
            "uniprot_id": upid,
            "tokens": tokens
        }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("chunk_path", type=str, help="Path to the chunk of structure data")
    parser.add_argument("tokens_dir", type=str, help="Path to save the tokenized output (Parquet format)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for tokenization (default: cuda)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (process only a small subset)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    struct_tokenizer = ESM3StructureTokenizer(device=device)
    struct_tokenizer.init()

    # Convert to Dataset, then save as Parquet
    out_dataset = datasets.Dataset.from_generator(token_iterator,
        gen_kwargs={"chunk_path": args.chunk_path, "tokenizer": struct_tokenizer, "debug": args.debug},
        features=datasets.Features({
            "uniprot_id": datasets.Value("string"),
            "tokens": datasets.Sequence(datasets.Value("int32"))
        }))

    tokens_dir = Path(args.tokens_dir)
    tokens_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = tokens_dir / (Path(args.chunk_path).name + "_tokens.parquet")
    args.tokens_path = str(tokens_path)
    logger.info(f"Saving tokens to {args.tokens_path}")
    out_dataset.to_parquet(args.tokens_path)