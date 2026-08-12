"""
Another survival technique for reading hfds arrow with old version of datasets

Use parallel reading to maximize the reading performance

Minhuan Li, Mar 2025
"""

import pyarrow as pa
import numpy as np
import concurrent.futures
from tqdm import tqdm
import os


def process_arrow_chunk(args):
    """
    Process a chunk of rows from an Arrow file.
    
    Parameters:
    -----------
    args : tuple
        A tuple containing (file_path, start_row, end_row)
    
    Returns:
    --------
    tuple
        (uniprot_ids, structures) for the specified chunk
    """
    file_path, start_row, end_row = args
    
    # Read the Arrow file
    with open(file_path, "rb") as f:
        reader = pa.ipc.open_stream(f)
        table = pa.Table.from_batches([batch for batch in reader])
    
    # Initialize lists to store the extracted data
    uniprot_ids = []
    structures = []
    
    # Process only the specified chunk of rows
    for i in range(start_row, min(end_row, len(table))):
        # Extract uniprot_id
        uniprot_id = table['uniprot_id'][i].as_py()
        uniprot_ids.append(uniprot_id)
        
        # Extract structure and convert to the appropriate format
        structure_item = table['structure'][i].as_py()
        
        # Convert to numpy array with appropriate type
        structure_array = np.array(structure_item, dtype=np.float32)
        structures.append(structure_array)
    
    return uniprot_ids, structures


def read_arrow_file_parallel(file_path, num_workers=None):
    """
    Read an Arrow file in parallel using ProcessPoolExecutor.
    
    Parameters:
    -----------
    file_path : str
        Path to the Arrow file
    num_workers : int, optional
        Number of worker processes to use. If None, uses os.cpu_count()
    chunk_size : int, optional
        Number of rows to process in each chunk
    
    Returns:
    --------
    tuple
        (uniprot_ids, structures) for the entire file
    """
    # Determine the number of workers
    if num_workers is None:
        num_workers = os.cpu_count()
    
    # First, get the total number of rows in the file
    with open(file_path, "rb") as f:
        reader = pa.ipc.open_stream(f)
        table = pa.Table.from_batches([batch for batch in reader])
        total_rows = len(table)
    
    # Create chunks of work
    chunks_per_worker = 3
    total_chunks = num_workers * chunks_per_worker
    chunk_size = max(1, total_rows // total_chunks)
    chunks = []
    for start_row in range(0, total_rows, chunk_size):
        end_row = min(start_row + chunk_size, total_rows)
        chunks.append((file_path, start_row, end_row))
    
    # Process chunks in parallel
    all_uniprot_ids = []
    all_structures = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_chunk = {executor.submit(process_arrow_chunk, chunk): chunk for chunk in chunks}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_chunk):
            try:
                uniprot_ids, structures = future.result()
                all_uniprot_ids.extend(uniprot_ids)
                all_structures.extend(structures)
            except Exception as e:
                print(f"Error processing chunk: {e}")
    
    return all_uniprot_ids, all_structures


def read_multiple_arrow_files(file_paths, num_workers=None):
    """
    Read multiple Arrow files and concatenate the results.
    
    Parameters:
    -----------
    file_paths : list
        List of paths to Arrow files
    num_workers : int, optional
        Number of worker processes to use per file. If None, uses os.cpu_count()
    
    Returns:
    --------
    tuple
        (uniprot_ids, structures) concatenated from all files
    """
    all_uniprot_ids = []
    all_structures = []
    
    for file_path in tqdm(file_paths, desc="Reading Arrow Files..."):
        uniprot_ids, structures = read_arrow_file_parallel(file_path, num_workers)
        all_uniprot_ids.extend(uniprot_ids)
        all_structures.extend(structures)
    
    return all_uniprot_ids, all_structures
