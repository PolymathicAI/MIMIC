#!/usr/bin/env python3
# %%
"""
Build a presence-matrix (Boolean columns) for every modality, report pattern
coverage, and compute cumulative counts for excluded signatures.

"""

# ── IMPORTS ────────────────────────────────────────────────────────────────
#   stdlib
import time
from pathlib import Path
import shutil
import argparse

#   third-party
import duckdb
import numpy as np
import pandas as pd
import yaml

#   project
from lore import logger
from lore import paths
import gc

def main(debug=False, test_frac=0.15):

    if debug:
        logger.warning(f"Running in debug mode with {test_frac:.0%} of the data!")

    # ── VARS ──────────────────────────────────────────────────────────────────
    SUBFOLDER = Path("./assets")  # folder to save the subsets
    if SUBFOLDER.exists():
        # Log an error that the folder exists and exit
        logger.error(f"The subfolder {SUBFOLDER} already exists. Exitting to avoid overwriting data.")
        exit(0)
    SUBFOLDER.mkdir(parents=True, exist_ok=False)

    # delete the subset_details.txt if it exists
    log_path = Path("subset_details.txt")
    if log_path.exists():
        log_path.unlink()
        logger.info(f"Deleted {log_path}")

    log_add_int = logger.add(
        log_path,
        format=(
            "{time:MMM-DD HH:mm:ss} | "
            "{level:<8} | "  # level padded to 8 chars
            "{file.name:<20}:"
            "{line:>4} | "  # line right-aligned in 4-char column
            "{message}"
        ),
        level="INFO",
    )

    subset_path = SUBFOLDER / "subsets"
    if not subset_path.exists():
        subset_path.mkdir(parents=True)
    logger.info(f"Subsets will be saved in {subset_path}")

    # %%
    # ── LOAD YAML CONFIG ───────────────────────────────────────────────────────
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    logger.info(f"Loaded config from {file.name}")

    # copy the config.yaml to the subfolder
    copied_config_path = SUBFOLDER / "config.yaml"
    shutil.copyfile(config_path, copied_config_path)
    logger.info(f"Copied config to {copied_config_path}")

    # %%
    # ── INITIAL SETUP ──────────────────────────────────────────────────────────
    t0_total = time.perf_counter()

    assert (
        "master_id_version" in config
    ), "The config must contain a 'master_id_version' key (saved in the 'modality' stage)."

    id_root = paths.get_path(
        data_type="data",
        stage="modality",
        name="id",
        version=str(config["master_id_version"]),
        fmt="parquet",
    )
    id_path = id_root / "dataset.parquet"
    mod_avail_path = SUBFOLDER / "mod_avail.parquet"

    # %%
    # ── CONNECT TO IN-MEMORY DUCKDB AND LOAD IDs ──────────────────────────────
    con = duckdb.connect(database=":memory:")

    # set preserve insertion order to false for performance
    con.execute("SET preserve_insertion_order = false")

    con.execute(f"CREATE TABLE IDs AS SELECT * FROM read_parquet('{id_path}')")
    logger.info("Loaded IDs table into DuckDB")

    # %%

    # add the mutually exclusive modalities first
    individual_merge_queries = []
    merge_keys = []

    # For each modality, generate a LEFT JOIN query.
    for mod_name, mod_config in config["mutually_exclusive_modalities"].items():
        mod_path = paths.get_path(
            data_type="data",
            stage="modality",
            name=mod_name,
            version=str(mod_config['version']),
            fmt="parquet"
        ) / 'dataset.parquet'
        
        merge_key = mod_config['merge_key']

        size = con.execute(f"SELECT COUNT(*) AS cnt FROM read_parquet('{mod_path}')").fetchone()[0]
        modality_subquery = f"""
        SELECT {merge_key}, 
        TRUE as {mod_name} 
        FROM read_parquet('{mod_path}')
        {f"LIMIT {int(size*test_frac)}" if debug else ""}
        """   

        # Use a RIGHT JOIN with implicit column selection.
        merge_query = f"""
        (SELECT *
        FROM IDs
        INNER JOIN ({modality_subquery}) AS T2 USING ({merge_key}))
        """
        individual_merge_queries.append(merge_query)

        merge_keys.append(merge_key)

    # Combine all join results. Duplicates for unmatched IDs rows will be present at this stage.
    full_query = "\nUNION ALL BY NAME\n".join(individual_merge_queries)

    # Create the final table. Not using Distinct for the same reason as before
    mut_exc_query = f"""
    CREATE OR REPLACE TABLE MUT_EXC_MODS AS
    SELECT *
    FROM ({full_query})
    """
    con.execute(mut_exc_query)
    logger.info(f"Created MUT_EXC_MODS table with {f'{test_frac:.0%} of ' if debug else ''}mutually exclusive modalities {', '.join(config['mutually_exclusive_modalities'].keys())}")

    # %%

    # Get the mut_exc_mods columns other than the has_ columns
    mut_exc_columns = con.execute("DESCRIBE MUT_EXC_MODS").fetchdf()['column_name'].tolist()

    # Add the rows in IDs where the uniprot_id or genome_feature_id is not present the mut_exc_mods table
    # Do not use DISTINCT here to accurately count the different cell types etc.
    con.execute(f"""
    CREATE TABLE IDS_W_MUT_EXC AS
    SELECT * FROM MUT_EXC_MODS
    UNION ALL
    SELECT t1.*, {', '.join(['FALSE AS ' + col for col in mut_exc_columns if col in config["mutually_exclusive_modalities"]])}
    FROM IDs AS t1
    LEFT JOIN MUT_EXC_MODS AS t2 ON {" AND ".join([f"t1.{key} IS NOT DISTINCT FROM t2.{key}" for key in set(merge_keys)])}
    WHERE {' AND '.join([f"t2.{col} IS NULL" for col in mut_exc_columns if col in config["mutually_exclusive_modalities"]])}
    """)
    logger.info("Created IDS_W_MUT_EXC table with all IDs and mutually exclusive modalities")


    # fill the Nones with FALSE for the mutually exclusive modalities and insert back into the IDs table
    con.execute(f"""
                CREATE OR REPLACE TABLE IDS_W_MUT_EXC AS
                SELECT {', '.join([col for col in mut_exc_columns if not col in config["mutually_exclusive_modalities"]])},
                {', '.join([f"COALESCE({col}, FALSE) AS {col}" for col in mut_exc_columns if col in config["mutually_exclusive_modalities"]])}
                FROM IDS_W_MUT_EXC
                """
    )
    # Remove the MUT_EXC_MODS and IDs table to free memory
    con.execute("DROP TABLE MUT_EXC_MODS")
    con.execute("DROP TABLE IDs")

    logger.info("Filled NULLs with FALSE for mutually exclusive modalities and dropped intermediate tables.")
    # %%

    # ── BUILD PRESENCE-FLAG VIEW PER MODALITY ─────────────────────────────────
    for mod, mod_dict in config["modalities"].items():
        mod_root = paths.get_path(
            data_type="data",
            stage="modality",
            name=mod,
            version=str(mod_dict["version"]),
            fmt="parquet",
        )
        merge_key = mod_dict["merge_key"]

        con.execute(
            f"""
            CREATE VIEW {mod} AS
            SELECT
                {merge_key},
                TRUE AS {mod}
            FROM read_parquet('{(mod_root / "dataset.parquet")}') AS m
            {f"LIMIT {int(size*test_frac)}" if debug else ""}
            """
        )
        logger.info(f"View built: {mod} {f'with {test_frac:.0%} of rows.' if debug else ''}")

    # ── FULL OUTER JOIN ALL MODALITIES & EXPORT PARQUET ───────────────────────
    mod_names = list(config["modalities"].keys())
    # Generate a LEFT JOIN clause for each modality, joining on its specific merge key
    join_sql = "".join(f" LEFT JOIN {m} USING ({config['modalities'][m]['merge_key']})" for m in mod_names)
    # Select all columns from the base 'IDS_W_MUT_EXC' table
    # Then, for each modality, select its boolean flag, using COALESCE to turn NULLs (from non-matches) into FALSE
    select_sql = "IDS_W_MUT_EXC.*, " + ", ".join(f"COALESCE({m}, FALSE) AS {m}" for m in mod_names)

    merge_sql = f"""
        COPY (
            SELECT {select_sql}
            FROM IDS_W_MUT_EXC{join_sql}  -- Start FROM the 'IDS_W_MUT_EXC' table and apply all LEFT JOINs
        )
        TO '{mod_avail_path}'
        (FORMAT PARQUET);
    """

    t0_sql = time.perf_counter()
    logger.info(f"Exporting modality availability dataset to {mod_avail_path}...")
    con.execute(merge_sql)
    elapsed = time.perf_counter() - t0_sql
    mins = int(elapsed // 60)
    secs = elapsed % 60
    logger.info(f"Export finished in {mins}m {secs:.2f}s")
    con.close()

    # %%
    # ── LOAD RESULTING PARQUET ────────────────────────────────────────────────
    df = pd.read_parquet(mod_avail_path)
    mods = df.columns.drop(["uniprot_id", "genome_feature_id", "split"])  # all modality columns
    # drop the rows that do not have at least one of the req_mods
    req_mods = config["required_modalities"]
    if req_mods:
        before_count = len(df)
        df = df[df[req_mods].any(axis=1)]
        logger.info(
            f"Dropped {before_count - len(df):,} rows that do not have any of the required modalities: "
            f"{', '.join(req_mods)}"
        )

    # ── BUILD SIGNATURE (“0101…”) STRING PER ROW ──────────────────────────────
    logger.info("Building presence signature strings for each sample.")
    sig = df[mods].astype("uint8").astype(str).agg("".join, axis=1)

    # ---- helper lambdas ------------------------------------------------------
    def present(bits):
        """Return tuple of modality names present in the signature bits string."""
        return tuple(mods[i] for i, b in enumerate(bits) if b == "1")
    def absent(bits):
        """Return tuple of modality names absent from the signature bits string."""
        return tuple(mods[i] for i, b in enumerate(bits) if b == "0")

    # ── FULL PATTERN REPORT ───────────────────────────────────────────────────
    logger.info("Counting the patterns and generating report...")
    pattern_counts = sig.value_counts()
    report_df = (
        pattern_counts.rename_axis("signature")
        .reset_index(name="n_samples")
        .assign(
            present_modalities=lambda d: d["signature"].apply(present),
            absent_modalities=lambda d: d["signature"].apply(absent),
        )
        .sort_values("n_samples", ascending=False)
        .reset_index(drop=True)
    )
    report_df.index.name = "idx"
    report_df = report_df.reset_index()

    log_str = report_df[['idx', 'n_samples', 'present_modalities', 'absent_modalities']].to_string(
        index=False,
        line_width=10_000,
        formatters={'n_samples': '{:,}'.format},
    )

    logger.info(
        "Full pattern report (select k by idx):\n"
        f"{log_str}",
    )

    # %%
    # ── CHOOSE SIGNATURES ───────────────────────────────────────────────
    selected_indices = None
    max_index = len(report_df) - 1

    while selected_indices is None:
        # 1. Get input string from the user
        prompt = f"Enter signature indices to include (range 0 to {max_index}), separated by space or comma: "
        indices_str = input(prompt)

        # 2. Normalize delimiters (replace commas with spaces) and split
        # This handles "1,2 3" -> "1 2 3" -> ['1', '2', '3']
        str_indices = indices_str.replace(",", " ").split()

        if not str_indices:
            logger.error("No indices provided. Please try again.")
            continue

        try:
            # 3. Convert all inputs to integers
            int_indices = [int(s.strip()) for s in str_indices]
            
            # 4. Validate that all indices are within the valid range
            out_of_bounds = [i for i in int_indices if not (0 <= i <= max_index)]
            
            if out_of_bounds:
                logger.error(f"Invalid indices found: {out_of_bounds}. Please only use indices from 0 to {max_index}.")
            else:
                # Success: store the list and break the loop
                selected_indices = int_indices

        except ValueError:
            logger.error("Invalid input. Please enter only space- or comma-separated numbers.")
            # selected_indices remains None, so the loop will repeat

    # Use .iloc to select rows by their integer position (indices)
    picked_sigs = report_df.iloc[selected_indices]["signature"]
    included_mask = sig.isin(picked_sigs)
    n_total = len(df)
    n_included = included_mask.sum()
    n_not_included = n_total - n_included

    logger.info(
        f"Selected signatures ({indices_str}) cover {n_included:,}/{n_total:,} "
        f"samples {n_included / n_total :.1%}.  Not included: {n_not_included:,}"
    )
    logger.info(f"Selected signatures present modalities are:")
    for idx, el in zip(selected_indices, picked_sigs.apply(present).tolist()):
        logger.info(f"  {idx}: {el}")

    # %%
    # %%
    # Ask the user for the priority modalities for cumulative counting
    priority_modalities = [] # This will be the ordered list of modality names
    all_mods_str = ", ".join(mods)

    while True:
        prompt = (
            f"Enter priority modalities (space or comma-separated) from this list:\n"
            f"[{all_mods_str}]\n"
            "Press Enter to skip: "
        )
        mods_str = input(prompt)

        if not mods_str.strip():
            logger.info("No priority modalities set. Using default logic (max modalities).")
            break

        str_mods = mods_str.replace(",", " ").split()
        
        valid_priority_mods = []
        invalid_mods = []
        seen_mods = set()

        for s in str_mods:
            mod_name = s.strip()
            if not mod_name: continue
            
            if mod_name not in mods:
                invalid_mods.append(mod_name)
            elif mod_name not in seen_mods:
                seen_mods.add(mod_name)
                valid_priority_mods.append(mod_name)
        
        if invalid_mods:
            logger.error(
                f"Invalid modalities: {invalid_mods}. "
                f"Please only use modalities from the list."
            )
            continue # Ask again
        
        if len(valid_priority_mods) < len(str_mods):
            logger.warning(
                f"Duplicate or empty modalities removed. Using order: {valid_priority_mods}"
            )
        
        priority_modalities = valid_priority_mods
        logger.info("Priority modalities set:")
        for i, mod_name in enumerate(priority_modalities):
            logger.info(f"  P{i+1}: {mod_name}")
        
        break # Success

    # %%
    # ── CUMULATIVE COVERAGE FOR EXCLUDED SIGNATURES ───────────────────────────
    exc_counts = sig[~included_mask].value_counts()

    exc_masks = exc_counts.index.to_series().apply(lambda s: int(s, 2)).astype(np.uint64)
    mask_arr = exc_masks.values
    count_arr = exc_counts.values

    cumul_dict = {}
    for sig_str, mask in zip(exc_counts.index, mask_arr):
        cumul_dict[sig_str] = int(count_arr[(mask_arr & mask) == mask].sum())

    cumul_df = (
        pd.DataFrame(
            {
                "signature": exc_counts.index,
                "n_samples": exc_counts.values,
                "cumulative_samples": [cumul_dict[s] for s in exc_counts.index],
                "idx": exc_counts.index.map(report_df.set_index("signature")["idx"]),
                "present_modalities": exc_counts.index.to_series().apply(present).values,
                "absent_modalities": exc_counts.index.to_series().apply(absent).values,
            }
        )
        .sort_values("cumulative_samples", ascending=False)
        .loc[
            :,
            [
                "idx",
                "signature",
                "n_samples",
                "cumulative_samples",
                "present_modalities",
                "absent_modalities",
            ],
        ]
    )

    logger.info(
        "Cumulative coverage for excluded signatures:\n"
        f"""{cumul_df.drop(columns='signature').to_string(index=False, line_width=10_000,
                                formatters={'n_samples': '{:,}'.format,
                                            'cumulative_samples': '{:,}'.format},
                                )}""" 
    )

    # %%
    # ── CUMULATIVE PATTERN COUNT FOR picked SIGNATURES ─────────────────────────
    cum_count = dict(
        zip(picked_sigs, sig.isin(picked_sigs).groupby(sig).sum().loc[picked_sigs])
    )
    picked_masks = {s: int(s, 2) for s in picked_sigs}
    picked_sizes = {s: s.count("1") for s in picked_sigs}
    added_sigs = {s: [] for s in picked_sigs}
    unmatched = []

    # Map modality names to their index in the signature string
    mod_indices = {mod_name: i for i, mod_name in enumerate(mods)}

    for s_ex, n_ex in exc_counts.items():
        mask_ex = int(s_ex, 2)
        candidates = [s for s, m in picked_masks.items() if (m & mask_ex) == m]

        if candidates:
            filtered_candidates = list(candidates) # Start with all candidates
            
            for mod_name in priority_modalities:
                mod_idx = mod_indices[mod_name]
                
                # Find candidates that have this priority modality
                priority_subset = [
                    s for s in filtered_candidates if s[mod_idx] == '1'
                ]
                
                # If any candidates have it, filter the list.
                # Otherwise, keep the list as-is and try the next modality.
                if priority_subset:
                    filtered_candidates = priority_subset
            
            # After all filtering, pick the candidate with the most modalities
            best = max(filtered_candidates, key=lambda s: picked_sizes[s])
            
            cum_count[best] += int(n_ex)
            added_sigs[best].append(s_ex)
        else:
            unmatched.append((s_ex, int(n_ex)))

    idx_map = report_df.set_index("signature")["idx"]
    cum_pattern_count = (
        pd.DataFrame(
            {
                "signature": list(cum_count.keys()),
                "idx": [idx_map[s] for s in cum_count],
                "original_samples": [
                    exc_counts.get(s, 0) + sig.eq(s).sum() for s in cum_count
                ],
                "cumulative_samples": list(cum_count.values()),
                "added_signatures": [tuple(added_sigs[s]) for s in cum_count],
                "present_modalities": [present(s) for s in cum_count],
                "absent_modalities": [absent(s) for s in cum_count],
            }
        )
        .sort_values("cumulative_samples", ascending=False)
        .loc[
            :,
            [
                "idx",
                "signature",
                "original_samples",
                "cumulative_samples",
                "added_signatures",
                "present_modalities",
                "absent_modalities",
            ],
        ]
    )

    logger.info(
        "Cumulative pattern count for picked signatures:\n"
        f"""{cum_pattern_count.drop(columns=['signature', 'added_signatures']).to_string(index=False, line_width=10_000,
                                        formatters={'original_samples': '{:,}'.format,
                                                    'cumulative_samples': '{:,}'.format},
                                        )}""" 
    )

    # ---- unmatched exclusions -----------------------------------------------
    if unmatched:
        unmatched_df = (
            pd.DataFrame(unmatched, columns=["signature", "n_samples"])
            .assign(
                idx=lambda d: d["signature"].map(idx_map),
                present_modalities=lambda d: d["signature"].apply(present),
                absent_modalities=lambda d: d["signature"].apply(absent),
            )
            .loc[
                :,
                [
                    "idx",
                    "signature",
                    "n_samples",
                    "present_modalities",
                    "absent_modalities",
                ],
            ]
        )
        logger.info(
            "Excluded signatures not accounted for:\n"
            f"{unmatched_df.drop(columns='signature').to_string(index=False, line_width=10_000, formatters={'n_samples': '{:,}'.format})}"
        )
    else:
        logger.info("All excluded signatures were successfully mapped.")
    # %%
    # ── BUILD cum_subset_dict: picked SIG → sample keys it now covers ──────────
    logger.info("Constructing the subset dataframes.")
    cum_subset_dict = {}

    for picked_sig in picked_sigs:
        mask = sig.eq(picked_sig)
        for extra_sig in added_sigs[picked_sig]:
            mask |= sig.eq(extra_sig)
        cum_subset_dict[present(picked_sig)] = list(df.loc[mask, ["uniprot_id", "genome_feature_id", "split"]].itertuples(index=False, name=None))

    dup_count = sum(len(v) - len(set(v)) for v in cum_subset_dict.values())
    logger.info(
        f"Total samples in cumulative subsets: {sum(len(v) for v in cum_subset_dict.values()):,}"
    )
    logger.info(f"Total samples in original dataset   : {len(df):,}")

    # %%
    # ── Write each subset to a parquet file ─────────────────────────────────
    logger.info("Writing the subsets to parquet files.")
    for name, data in cum_subset_dict.items():
        # validate mutual exclusivity of mut_exc modalities
        assert len(set(config["mutually_exclusive_modalities"]).intersection(name)) <= 1, \
            f"Subset {name} contains multiple mutually exclusive modalities."
        # make a dataframe with the keys
        subset_df = pd.DataFrame(data, columns=["uniprot_id", "genome_feature_id", "split"])
        subset_df_no_dup = subset_df.drop_duplicates()
        # write the dataframe to a parquet file
        file_name = "+".join(name) + ".parquet"
        subset_df_no_dup.to_parquet(subset_path / file_name, index=False)
        logger.info(f"Saved subset {file_name} with {len(subset_df):,} samples,  ({len(subset_df_no_dup):,} after dropping duplicates).")

    logger.info("Validated no subset contains multiple mutually exclusive modalities.")
    # %%
    # ── FINISH ────────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0_sql
    mins = int(elapsed // 60)
    secs = elapsed % 60
    logger.info(f"Finished in {mins}m {secs:.2f}s")

    # remove the log path from the logger
    logger.remove(log_add_int)

    # ── CLEANUP ───────────────────────────────────────────────────────────────
    logger.info("Cleaning up large objects to release memory.")
    del df
    del sig
    del report_df
    del cumul_df
    del cum_pattern_count
    del cum_subset_dict
    if 'unmatched_df' in locals():
        del unmatched_df
    gc.collect()
    logger.info("Cleanup complete.")
    # %%


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False, help="Run in debug mode on a fraction of the data.")
    parser.add_argument("--test_frac", type=float, default=0.15, help="Fraction of data to use in test mode.")
    args = parser.parse_args()

    main(debug=args.debug, test_frac=args.test_frac)