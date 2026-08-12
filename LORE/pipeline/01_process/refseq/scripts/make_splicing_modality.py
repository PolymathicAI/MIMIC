import duckdb
from pathlib import Path
from lore import paths
from lore import logger
import numpy as np
import time
from tqdm import tqdm
import pyarrow as pa
import pandas as pd
import pyarrow.parquet as pq

## TODO: This should probably just be part of 06_build_modalities, I can do this  

def main():
    logger.info("Starting splicing modality processing")
    
    # Define the input file paths
    cds_file = paths.get_path(data_type="data", stage="modality", name="cds", version="1.0", fmt="parquet") / 'dataset.parquet'
    exon_file = paths.get_path(data_type="data", stage="modality", name="exon", version="1.0", fmt="parquet") / 'dataset.parquet'
    logger.info(f"Input CDS file: {cds_file}")
    logger.info(f"Input exon file: {exon_file}")

    # Define the output file path
    output_file = paths.get_path(data_type="data", stage="modality", name="splice_regions", version="1.0", fmt="parquet") / 'dataset.parquet'
    logger.info(f"Output will be written to: {output_file}")

    # Check if input files exist
    if not cds_file.exists():
        logger.error(f"CDS file not found: {cds_file}")
        raise FileNotFoundError(f"CDS file not found: {cds_file}")
    if not exon_file.exists():
        logger.error(f"Exon file not found: {exon_file}")
        raise FileNotFoundError(f"Exon file not found: {exon_file}")
    
    # Create DuckDB connection  
    conn = duckdb.connect()

    conn.execute(f"CREATE VIEW cds_view AS SELECT * FROM read_parquet('{cds_file}')")
    conn.execute(f"CREATE VIEW exon_view AS SELECT * FROM read_parquet('{exon_file}')")

    logger.info("Creating DuckDB connection")


    start = time.time()

    df = conn.execute("""
        SELECT 
            COALESCE(c.genome_feature_id, e.genome_feature_id) AS genome_feature_id,
            c.cds AS cds,
            e.exon AS exon
        FROM cds_view c
        FULL OUTER JOIN exon_view e ON c.genome_feature_id = e.genome_feature_id
    """).fetchdf()
    logger.info(f"SQL executed in {time.time() - start:.2f} seconds")


    def combine_arrays(row):
        cds = row['cds']
        exon = row['exon']

        if isinstance(cds, np.ndarray) and isinstance(exon, np.ndarray):
            result = np.logical_or(cds, exon)
        elif isinstance(cds, np.ndarray):
            result = cds
        elif isinstance(exon, np.ndarray):
            result = exon
        else:
            raise ValueError("Both cds and exon are None")

        # # Validate result: must contain only True or False (no NA, int, str, etc.)
        # if not np.issubdtype(result.dtype, np.bool_):
        #     raise ValueError(f"Non-boolean dtype in result: {result.dtype} \n {result} \n {row}")

        # if not np.all(np.isin(result, [True, False])):
        #     raise ValueError(f"Non-binary values found in result: {result}")

        return result


    tqdm.pandas(desc="Processing rows")

    df['splice_regions'] = df.progress_apply(combine_arrays, axis=1)

    logger.info("Finished processing rows")
    logger.info(f"Number of rows in the final DataFrame: {len(df)}")


    def write_large_df_to_parquet(df, file_path, chunk_size=100_000, compression="snappy"):
        schema = pa.Table.from_pandas(df.iloc[:1]).schema  # infer schema from a small sample
        with pq.ParquetWriter(file_path, schema, compression=compression) as writer:
            for start in tqdm(range(0, len(df), chunk_size), desc="Writing to parquet", unit="chunk"):
                end = min(start + chunk_size, len(df))
                batch = pa.Table.from_pandas(df.iloc[start:end], schema=schema, preserve_index=False)
                writer.write_table(batch)

    logger.info("Saving the final DataFrame to Parquet format")

    # tqdm.pandas(desc="Converting splice regions to string format")
    # # Convert splice_regions to string format, skipping None values
    # df['splice_regions'] = df['splice_regions'].progress_apply(
    #     lambda x: ''.join(map(str, x.astype(int))) if isinstance(x, np.ndarray) else None
    # )

    # write_large_df_to_parquet(df[['genome_feature_id', 'splice_regions']], output_file, chunk_size=100_000)

    # Save the DataFrame to Parquet format
    df[['genome_feature_id', 'splice_regions']].to_parquet(output_file)

    logger.info(f"Saved splicing modality to {output_file}")


if __name__ == "__main__":
    logger.info("Script execution started")
    main()
    logger.info("Script execution completed")
