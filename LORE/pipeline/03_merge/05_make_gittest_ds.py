#!/usr/bin/env python3
# %%

import shutil
from lore import paths
from lore import logger
from tqdm import tqdm
from pathlib import Path
import time, yaml
import tarfile


def main():

    """
    Creates a git test dataset by selecting a subset of the full dataset.
    This is used for both debugging and for running tests in the CI/CD pipeline.
    """

    with open(Path(__file__).parent / "tokenize_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    logger.info(f"Loaded config from {file.name}")

    ds_name = config["name"]
    ds_version = str(config["version"])
    ds_version_gittest = ds_version + "_gittest"

    start_time = time.time()

    webdataset_path = paths.get_path(data_type='data', name=ds_name, stage='final', version=ds_version, fmt='wds')
    wds_config_path = webdataset_path / "tokenize_config.yaml"
    stats_path = webdataset_path / "stats.yaml"

    # assert the config file matches the copied config file
    with open(wds_config_path, "r") as file:
        copied_config = yaml.safe_load(file)
    with open(stats_path, "r") as file:
        full_stats = yaml.safe_load(file)
    num_splits = len(config['length_splits']) - 1
    num_splits_val = len(config.get('val_length_splits', config['length_splits'])) - 1
    if copied_config != config:
        raise ValueError(f"Config file {wds_config_path} does not match the current config. This implies a version conflict.")

    git_test_config = config['git_test_config']
    logger.info(f"Git test config: {git_test_config}")

    output_path = paths.get_path(data_type='data', name=ds_name, stage='final', version=ds_version_gittest, fmt='wds')
    # assert that the output path does not exist.
    if output_path.exists():
        logger.error(f"Output path {output_path} already exists. Exiting.")
        exit(1)

    # create the output path and put the config file in it
    output_path.mkdir(parents=True, exist_ok=False)
    shutil.copy(wds_config_path, output_path / "tokenize_config.yaml")
    # %%

    subsets = [el for el in webdataset_path.iterdir() if el.is_dir()]
    subsets = sorted(subsets, key=lambda x: len(x.stem.split('+')), reverse=True)[:git_test_config['num_subsets']]
    logger.info(f"Using {len(subsets)} subsets for git test dataset: {[subset.stem for subset in subsets]}")

    stats = {subset.stem:{} for subset in subsets}

    # iterate over the subsets
    for subset_dir in subsets:

        # get the content of the subset directory
        subset_folders = [el for el in subset_dir.iterdir() if el.is_dir()]
        split_folders = []
        for split in ['train', 'val']:
            matching_splits = [el for el in subset_folders if el.name.startswith(split) and '_' in el.name]
            matching_splits = sorted(matching_splits, key=lambda el: int(el.name.split('_')[-1]))[:git_test_config['num_splits']]
            split_folders += matching_splits
            split_nums = [int(el.name.split('_')[-1]) for el in matching_splits]
            stats[subset_dir.stem][split] = {
                'rows': git_test_config['samples_per_split'] * git_test_config['num_splits'],
                'length_splits': {i: {'start': full_stats[subset_dir.stem][split]['length_splits'][i]['start'],
                                        'end': full_stats[subset_dir.stem][split]['length_splits'][i]['end'],
                                        'rows': git_test_config['samples_per_split'] if i in split_nums else 0}
                                        for i in range(num_splits if split == 'train' else num_splits_val)}}
        logger.info(f"Processing {len(split_folders)} splits for subset {subset_dir.stem}: {[split.name for split in split_folders]}")

        # iterate over the split folders:
        for split_folder in tqdm(
            split_folders, 
            desc=f"Processing {subset_dir.stem[:20] + ('...' if len(subset_dir.stem) > 20 else '')} splits", 
            unit="split"
        ):
            # get the modalities in the subset and iterate over them
            mod_paths = [el for el in split_folder.iterdir() if el.is_dir()]
            for mod_path in mod_paths:
                # get the files in the modality path
                mod_shards = [el for el in mod_path.iterdir() if el.is_file()]

                # get the shard with the lowest number
                mod_shard = min(mod_shards, key=lambda x: int(x.stem.split('shard-')[-1]))

                # look inside the shard and keep only the first git_test_config['samples_per_split'] samples
                # be very efficient in pulling out these samples so you don't load the entire shard into memory
                samples_per_split = git_test_config['samples_per_split']

                # create the corresponding output directory structure
                new_shard_dir = output_path / subset_dir.stem / split_folder.name / mod_path.name
                new_shard_dir.mkdir(parents=True, exist_ok=True)
                new_shard_file = new_shard_dir / mod_shard.name

                with tarfile.open(mod_shard, "r") as src_tar, tarfile.open(new_shard_file, "w") as dst_tar:
                    for i, member in enumerate(src_tar):
                        if i >= samples_per_split:
                            break
                        fileobj = src_tar.extractfile(member)
                        dst_tar.addfile(member, fileobj)
                        if fileobj:
                            fileobj.close()

    logger.info(f"Git test dataset created at {output_path}") 

    stats_path = output_path / "stats.yaml"
    # write the stats to a yaml file
    with open(stats_path, "w") as f:
        yaml.dump(stats, f)

    logger.info(f"Stats written to {stats_path}")

# %%

if __name__ == "__main__":
    main()