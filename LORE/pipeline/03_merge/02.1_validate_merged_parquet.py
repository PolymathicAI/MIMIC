import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from lore import logger
from lore import merge_checks as checks
from lore import paths


def check_merged_set(set_path, check_list):
    logger.info(f"Checking {set_path.name}")
    logger.info("=" * 50)

    subset_failed = {}

    for split_name in ["train", "val", "test"]:
        subset_dir = set_path / split_name

        if not subset_dir.exists():
            logger.warning(f"No data found for {split_name} in {set_path}, skipping.")
            continue

        logger.info(f"Loading data for {split_name} from {subset_dir}")
        df = pd.read_parquet(subset_dir)

        if not len(df):
            logger.warning(f"No data found for {split_name} in {set_path}, skipping.")
            continue

        failed_results = checks.run_check_suite(df, check_list, split_name)
        subset_failed[split_name] = failed_results

    return subset_failed

def main():
    """
    Validates merged parquet datasets based on a configuration file.
    This function reads a `config.yaml` to determine the dataset name and
    version. It then locates the merged parquet files, which are expected to
    be in directories named by the combination of modalities they contain
    (e.g., 'aa_seq+dssp').
    For each merged set, it dynamically determines which validation checks to
    run based on the available modalities. The checks are performed on the
    'train', 'validation', and 'test' subsets within each merged set.
    The validation checks include:
    - aa_dssp_length: Verifies that amino acid sequence and DSSP sequence
      lengths are equal.
    - aa_sasa_length: Verifies that amino acid sequence and SASA sequence
      lengths are equal.
    - cds_aa_length: Verifies that the coding sequence (CDS) length is three
      times the amino acid sequence length.
    - cds_has_start_codon: Checks if the CDS starts with a valid start codon.
    - cds_has_stop_codon: Checks if the CDS ends with a valid stop codon.
    - cds_multiple_of_three: Ensures the CDS length is a multiple of three.
    - rna_codons_valid: Checks if all codons in the `rna_codons` field are
      valid.
    - rna_codons_translation: Verifies that translating `rna_codons` yields
      the corresponding amino acid sequence.
    - cds_translation: Verifies that translating the CDS (spliced from the
      full RNA sequence) yields the corresponding amino acid sequence.
    - rna_seq_lengths: Checks for length consistency between RNA sequence,
      splice regions, and CDS.
    - rna_seq_length_relations: Checks for more complex length relationships
      between RNA sequence, splice regions, and CDS.
    - phylop_length: Verifies that the PhyloP conservation score sequence
      (for mouse or human) has the same length as the corresponding RNA
      sequence.
    Finally, it logs a summary of the validation results. It reports an ERROR
    if any check has a failure rate greater than 5%, and a WARNING for any
    non-zero failure rate. The total execution time is also logged.
    """
    
    with open(Path(__file__).parent / "config.yaml") as file:
        config = yaml.safe_load(file)
    logger.info(f"Loaded config from {file.name}")
    ds_name = config["name"]
    ds_version = str(config["version"])

    parquet_path = paths.get_path(
        data_type="data", name=ds_name, stage="merged", version=ds_version, fmt="parquet"
    )
    merged_sets = sorted(list(parquet_path.glob("*+*")))

    start_time = time.time()


    def check_in(short_list, full_list):
        """
        Check if all items in short_list are in full_list
        """
        return all(item in full_list for item in short_list)


    set_failed = {}

    for merged_set in merged_sets:
        logger.info(f"Checking merged set: {merged_set.name}")
        if not merged_set.is_dir():
            logger.warning(f"{merged_set} is not a directory, skipping.")
            continue
        included_modalities = merged_set.name.split("+")

        tests_to_run = set()
        if check_in(["aa_seq", "dssp"], included_modalities):
            tests_to_run.add("aa_dssp_length")
        if check_in(["aa_seq", "sasa"], included_modalities):
            tests_to_run.add("aa_sasa_length")
        if check_in(["cds", "aa_seq"], included_modalities):
            tests_to_run.add("cds_aa_length")
        if check_in(["cds", "rna_seq"], included_modalities):
            tests_to_run.add("cds_has_start_codon")
            tests_to_run.add("cds_has_stop_codon")
            tests_to_run.add("cds_multiple_of_three")
        if check_in(["rna_codons"], included_modalities):
            tests_to_run.add("rna_codons_valid")
        if check_in(["rna_codons", "aa_seq"], included_modalities):
            tests_to_run.add("rna_codons_translation")
        if check_in(["cds", "rna_seq", "aa_seq"], included_modalities):
            tests_to_run.add("cds_translation")
        if check_in(["rna_seq", "splice_regions", "cds"], included_modalities):
            tests_to_run.add("rna_seq_lengths")
            tests_to_run.add("rna_seq_length_relations")
        if check_in(["phylop_mouse"], included_modalities):
            tests_to_run.add("phylop_length")
        if check_in(["phylop_human"], included_modalities):
            tests_to_run.add("phylop_length")

        if not tests_to_run:
            logger.warning(f"No checks to run for {merged_set.name}, skipping.")
            continue
        logger.info(f"Running checks: {list(tests_to_run)}")

        subset_failed = check_merged_set(merged_set, list(tests_to_run))
        set_failed[merged_set.name] = subset_failed

    logger.info("=" * 20 + "All Checks Completed" + "=" * 20)
    for set_name, subsets in set_failed.items():
        for subset_name, results in subsets.items():
            for test_name, (n_failed, pct_failed) in results.items():
                if pct_failed > 5.0:
                    logger.error(
                        f"{set_name}/{subset_name}/{test_name}: {n_failed} failed checks ({pct_failed:.2f}%)"
                    )
                elif n_failed > 0:
                    logger.warning(
                        f"{set_name}/{subset_name}/{test_name}: {n_failed} failed checks ({pct_failed:.2f}%)"
                    )

    elapsed_time = time.time() - start_time
    m, s = divmod(elapsed_time, 60)
    logger.info(f"Total time: {int(m)} minutes and {int(s)} seconds.")

if __name__ == "__main__":
    main()