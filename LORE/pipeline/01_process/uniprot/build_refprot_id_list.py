#!/usr/bin/env python3
import polars as pl
import sys

def extract_uniprot_ids(parquet_file, output_file="ids.txt"):
    # Read parquet file
    df = pl.read_parquet(parquet_file)
    
    # Extract unique uniprot_ids, dropping nulls
    uniprot_ids = df.select("uniprot_id").drop_nulls().unique().to_series()
    
    # Write to file
    with open(output_file, 'w') as f:
        for uid in uniprot_ids:
            f.write(f"{uid}\n")
    
    print(f"Extracted {len(uniprot_ids)} unique uniprot IDs to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_ids.py input.parquet output.txt")
        sys.exit(1)
    
    extract_uniprot_ids(sys.argv[1], output_file=sys.argv[2])