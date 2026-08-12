
import os
import datasets
from lore import logger
from tqdm import tqdm
from lore import paths

dir_path = paths.get_path(data_type="data", stage="modality", version = "v4_10m", fmt="hfds_chunked", name = "structure")
out_path = paths.get_path(data_type="data", stage="modality", version = "v4_10m_list", fmt="hfds_chunked", name = "structure")

features = datasets.Features(
        {
            "uniprot_id": datasets.Value("string"),
            "structure": datasets.Sequence(
                datasets.Sequence(datasets.Value("float16"))
            ),
        }
    )

for chunk in tqdm(dir_path.iterdir()):
    logger.info(f"Loading {chunk.name}")
    ds = datasets.load_from_disk(str(chunk))
    ds2 = ds.cast(features)
    ds2.save_to_disk(str(out_path / chunk.name))
    logger.info(f"Saved {chunk.name} to {out_path / chunk.name}")