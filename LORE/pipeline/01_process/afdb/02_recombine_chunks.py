# %%
import datasets
from multiprocessing import Pool
from lore.paths import get_path
from tqdm import tqdm

# %%

structure_chunks = get_path("data", "modality", "prot_struct", "v4_plddt_70", fmt="hfds")
output_path = get_path("data", "modality", "prot_struct", "v4_plddt_70", fmt="hdfs_chunk")
all_chunks = list(structure_chunks.glob("chunk_*"))

# %% Compress chunk function

features = datasets.Features(
        {
            "uniprot_id": datasets.Value("string"),
            "structure": datasets.Sequence(
                datasets.Sequence(datasets.Value("float32"))
            ),
        }
    )

def compress_chunks(chunk_list):
    """
    Compress the chunks of data into a single file.
    """
    # Create a new dataset
    dataset = datasets.Dataset.from_dict({"uniprot_id": [], "structure": []}, features=features)
    chunk_id = int(chunk_list[0].name.split("_")[1]) // len(chunk_list)

    for chunk in tqdm(chunk_list, desc=f"Compressing chunk {chunk_id}", unit="chunk"):
        # continue
        # Load the chunk
        chunk_dataset = datasets.load_from_disk(str(chunk))
        
        # Append the chunk to the dataset
        dataset = datasets.concatenate_datasets([dataset, chunk_dataset])
        
    # Save the dataset to disk
    # print(chunk.parent.parent / "chunk" / f"chunk_{chunk_id}")
    dataset.save_to_disk(str(output_path / f"chunk_{chunk_id}"))

    return chunk_id

# %% Create batches

def create_batches(all_chunks, batch_size):
    """
    Create batches of chunks to be processed in parallel.
    """
    batches = []
    for i in range(0, len(all_chunks), batch_size):
        batches.append(all_chunks[i:i + batch_size])
    return batches

# %% Main function

chunk_size = 100

batches = create_batches(sorted(all_chunks), chunk_size)
print(batches[0])
print(len(batches[0]))

with Pool(processes=16) as pool:
    
    # Process each batch in parallel
    results = list(tqdm(pool.imap(compress_chunks, batches), total=len(batches), desc="Processing batches", unit="batch"))