import os
from pathlib import Path

import pandas as pd
import yaml

from lore import logger
from lore import paths

def main():
    """
    Validate that the tokenized data has the expected columns after merging.
    """
    
    with open(Path(__file__).parent / "config.yaml") as file:
        config = yaml.safe_load(file)

    with open(Path(__file__).parent / "merge_config.yaml") as file:
        merge_config = yaml.safe_load(file)

    logger.info(f"Loaded config from {file.name}")
    ds_name = config["name"]
    ds_version = str(config["version"])

    tokenized_directory = paths.get_path(
        data_type="data", name=ds_name, stage="final", version=ds_version, fmt="parquet"
    )

    for i in os.listdir(tokenized_directory):
        if not os.path.isdir(f"{tokenized_directory}/{i}"):
            continue

        if i in merge_config["extra_validations"]:
            df = pd.read_parquet(f"{tokenized_directory}/{i}/val")
            logger.info(f"Checking extra validation {i}")

            expected_columns = [
                f"tok_{i}" for i in merge_config["extra_validations"][i]["modalities"]
            ]
            missing_columns = set(expected_columns).difference(df.columns)
            if missing_columns:
                logger.error(f"Missing {missing_columns}")

            extra_columns = set(df.columns).difference(expected_columns + ["__key__"])
            if extra_columns:
                logger.warning(f"Unexpected extra columns {extra_columns}")

            continue

        for subset in ["train", "val", "test"]:
            try:
                df = pd.read_parquet(f"{tokenized_directory}/{i}/{subset}")
                logger.info(f"Checking {i}/{subset}")
                logger.info(f"Found columns: {df.columns.values}")

                if not len(df):
                    logger.warning("Number of rows: 0")
                else:
                    logger.info(f"Number of rows: {len(df)}")

                if "__key__" not in df.columns:
                    logger.error("Missing __key__ column")

                mods = i.split("+")
                all_expected_columns = []

                for m in mods:
                    if m in merge_config["mutually_exclusive_modalities"]:
                        expected_columns = [
                            f"tok_{i}"
                            for i in merge_config["mutually_exclusive_modalities"][m][
                                "column_mapping"
                            ].values()
                        ]
                    else:
                        expected_columns = [
                            f"tok_{i}"
                            for i in merge_config["modalities"][m][
                                "column_mapping"
                            ].values()
                        ]
                    all_expected_columns.extend(expected_columns)
                    missing_columns = set(expected_columns).difference(df.columns)
                    if missing_columns:
                        logger.error(f"Missing {missing_columns}")

                extra_columns = set(df.columns).difference(
                    all_expected_columns + ["__key__"]
                )
                if extra_columns:
                    logger.warning(f"Unexpected extra columns {extra_columns}")

                logger.info("#" * 80)
            except FileNotFoundError:
                logger.error(f"Continuing past {i}/{subset} -- does not exist")
            except NotADirectoryError:
                logger.error(f"Continuing past {i}/{subset} -- not a directory")
            except Exception as e:
                logger.error(f"Continuing past {i}/{subset} -- unexpected error: {e}")

if __name__ == "__main__":
    main()
