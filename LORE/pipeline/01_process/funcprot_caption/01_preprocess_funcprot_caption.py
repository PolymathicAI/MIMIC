"""Preprocess swissprot captions about function."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import ijson
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Regex utils
ECO_PAT = re.compile(r"\{ECO:.*?\}\.")
PUBMED_PAT = re.compile(r"\(PubMed:\d+\)|PubMed:\d+\)|\(PubMed:\d+")
SPACE_PUNCT = re.compile(r"\s+([.,])")
DESIRED_TAGS = ["FUNCTION", "SIMILARITY", "INTERPRO"]


def remove_paper_ids(text: str) -> str:
    """Remove ECO/PubMed refs and clean whitespace."""
    text = ECO_PAT.sub("", text)
    text = PUBMED_PAT.sub("", text)
    return SPACE_PUNCT.sub(r"\1", text).strip()


def agg_caption(comments: list[dict] | None, references: list[dict] | None) -> str:
    """Concatenate first text value of every FUNCTION and SIMILARITY comment element then INTERPRO references."""
    if not comments:
        # If there are no comments, leave caption empty
        return ""
    parts = defaultdict(list)
    # Aggregate comments of function
    for c in comments:
        tag = c.get("commentType", "").upper()
        if tag not in DESIRED_TAGS:
            continue
        texts = c.get("texts") or c.get("note", {}).get("texts")
        if texts:
            parts[tag].append(texts[0].get("value", ""))

    # References
    if references:
        for ref in references:
            tag = ref.get("database", "").upper()
            if tag not in DESIRED_TAGS:
                continue
            texts = ref.get("properties")
            if texts:
                parts[tag].append(texts[0].get("value", ""))

    return " ".join(
        f"{tag}: {', '.join(texts)}" for tag, texts in parts.items() if texts
    ).strip()


# Parallel Worker
def process_batch(batch: list[dict]) -> tuple[list[str], list[str]]:
    """Return (ids, cleaned captions) for a batch of entries."""
    ids, caps = [], []
    for entry in batch:
        uid = entry.get("primaryAccession") or entry.get("Entry")
        if not uid:
            continue
        raw = agg_caption(
            comments=entry.get("comments"),
            references=entry.get("uniProtKBCrossReferences"),
        )
        caps.append(remove_paper_ids(raw))
        ids.append(uid)
    return ids, caps


# Stream: don't retrieve pages and then download, do it in chunks
def stream_entries(json_fp):
    """Generator yielding entry dicts under results."""
    for entry in ijson.items(json_fp, "results.item"):
        yield entry


# Main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_file", type=Path, required=True)
    ap.add_argument("-o", "--output_folder", type=Path, required=True)
    ap.add_argument(
        "--batch_size", type=int, default=4_000, help="Entries per worker submission."
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=mp.cpu_count(),
        help="Process pool size (default: CPU count).",
    )
    ap.add_argument(
        "--total_entries",
        type=int,
        default=None,
        help="Optionally, known #entries for a bounded progress bar.",
    )
    args = ap.parse_args()

    # Output paths
    args.output_folder.mkdir(parents=True, exist_ok=True)
    stem = args.input_file.stem
    csv_path = args.output_folder / f"{stem}_preprocessed.csv"
    txt_path = args.output_folder / f"{stem}_preprocessed.txt"
    pq_path = args.output_folder / f"{stem}_preprocessed.parquet"

    schema = pa.schema([("uniprot_id", pa.string()), ("funcprot_caption", pa.string())])
    pq_writer = pq.ParquetWriter(pq_path, schema, compression="zstd")

    with (
        open(args.input_file, "rb") as inp,
        open(csv_path, "w", encoding="utf-8") as csv_f,
        open(txt_path, "w", encoding="utf-8") as txt_f,
        ProcessPoolExecutor(max_workers=args.workers) as pool,
        tqdm(
            total=args.total_entries,
            unit="entry",
            unit_scale=True,
            desc="Processed",
            colour=None,
        ) as bar,
    ):
        csv_f.write("uniprot_id\ffuncprot_caption\n")

        futures, batch = [], []
        for entry in stream_entries(inp):
            batch.append(entry)
            if len(batch) == args.batch_size:
                futures.append(pool.submit(process_batch, batch))
                batch = []
        if batch:
            futures.append(pool.submit(process_batch, batch))

        for fut in as_completed(futures):
            ids, caps = fut.result()
            bar.update(len(ids))

            csv_f.writelines(f"{u}\t{c}\n" for u, c in zip(ids, caps))
            txt_f.writelines(f"{c}\n" for c in caps)

            pq_writer.write_table(
                pa.Table.from_arrays([pa.array(ids), pa.array(caps)], schema=schema)
            )

    pq_writer.close()
    print("Finished!")


if __name__ == "__main__":
    main()
