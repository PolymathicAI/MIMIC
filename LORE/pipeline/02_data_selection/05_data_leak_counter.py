# %%
import os
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

# This is working with data/intermediate/master_ids/3.0/parquet/dataset.parquet which was copied for faster loading
master_id_path = os.path.expanduser("~/tmp/dataset.parquet")

targets = [
    ("root_id", "root"),
    ("rna_cluster_30", "rna30"),
    ("rna_cluster_70", "rna70")
]

cohorts = [
    ("All species", "1=1", "all"),
    ("All species (Coding)", "is_coding = TRUE", "all_coding"),
    ("All species (Non-coding)", "is_coding = FALSE", "all_noncoding"),
    ("Homo sapiens", "species = 'Homo sapiens'", "hs"),
    ("Homo sapiens (Coding)", "species = 'Homo sapiens' AND is_coding = TRUE", "hs_coding"),
    ("Homo sapiens (Non-coding)", "species = 'Homo sapiens' AND is_coding = FALSE", "hs_noncoding")
]

con = duckdb.connect()

# Dropped row_number() to allow fully parallelized parquet scanning
con.execute(f"CREATE VIEW dataset AS SELECT * FROM read_parquet('{master_id_path}')")

# =============================================================================
# PART 1: DETAILED LEAKAGE REPORT
# =============================================================================
for col_name, _ in targets:
    print(f"\n{'='*95}\n LEAKAGE REPORT: {col_name}\n{'='*95}")
    
    select_clauses = [
        "count(*) AS total_rows",
        "count(DISTINCT root_id) AS total_roots",
        "sum(CASE WHEN split_count > 1 THEN 1 ELSE 0 END) AS leaked_rows",
        "count(DISTINCT CASE WHEN split_count > 1 THEN root_id ELSE NULL END) AS leaked_roots"
    ]
    
    for desc, sql_cond, cohort_prefix in cohorts:
        for split in ['val', 'test']:
            other_split = 'test' if split == 'val' else 'val'
            p = f"{split}_{cohort_prefix}" 
            
            # Use boolean flags instead of list_contains for drastically faster evaluation
            select_clauses.extend([
                f"sum(CASE WHEN split = '{split}' AND {sql_cond} THEN 1 ELSE 0 END) AS {p}_total",
                f"sum(CASE WHEN split = '{split}' AND {sql_cond} AND has_train AND NOT has_{other_split} THEN 1 ELSE 0 END) AS {p}_mixed_train_only",
                f"sum(CASE WHEN split = '{split}' AND {sql_cond} AND has_{other_split} AND NOT has_train THEN 1 ELSE 0 END) AS {p}_mixed_{other_split}_only",
                f"sum(CASE WHEN split = '{split}' AND {sql_cond} AND has_train AND has_{other_split} THEN 1 ELSE 0 END) AS {p}_mixed_both",
                f"sum(CASE WHEN split = '{split}' AND {sql_cond} AND split_count = 1 THEN 1 ELSE 0 END) AS {p}_pure",
                
                f"count(DISTINCT CASE WHEN split = '{split}' AND {sql_cond} THEN root_id ELSE NULL END) AS {p}_total_roots",
                f"count(DISTINCT CASE WHEN split = '{split}' AND {sql_cond} AND has_train AND NOT has_{other_split} THEN root_id ELSE NULL END) AS {p}_mixed_train_only_roots",
                f"count(DISTINCT CASE WHEN split = '{split}' AND {sql_cond} AND has_{other_split} AND NOT has_train THEN root_id ELSE NULL END) AS {p}_mixed_{other_split}_only_roots",
                f"count(DISTINCT CASE WHEN split = '{split}' AND {sql_cond} AND has_train AND has_{other_split} THEN root_id ELSE NULL END) AS {p}_mixed_both_roots",
                f"count(DISTINCT CASE WHEN split = '{split}' AND {sql_cond} AND split_count = 1 THEN root_id ELSE NULL END) AS {p}_pure_roots"
            ])
            
    select_string = ",\n        ".join(select_clauses)
    
    query = f"""
    WITH base AS (
        SELECT * FROM dataset WHERE {col_name} IS NOT NULL
    ),
    aggs AS (
        SELECT {col_name}, 
               bool_or(split = 'train') AS has_train,
               bool_or(split = 'val') AS has_val,
               bool_or(split = 'test') AS has_test,
               (has_train::INT + has_val::INT + has_test::INT) AS split_count
        FROM base GROUP BY {col_name}
    ),
    joined AS (
        SELECT b.root_id, b.split, b.species, b.is_coding, a.*
        FROM base b JOIN aggs a ON b.{col_name} = a.{col_name}
    )
    SELECT {select_string} FROM joined
    """
    
    df_res = con.execute(query).df()
    
    if df_res.empty or df_res['total_rows'].iloc[0] == 0:
        print(f"No valid data found for {col_name}. Skipping...")
        continue
        
    res = df_res.fillna(0).iloc[0]

    print(f"Total entries clustered: {int(res['total_rows']):,} rows ({int(res['total_roots']):,} unique root_ids)")
    print(f"Entries with leakage:    {int(res['leaked_rows']):,} rows ({int(res['leaked_roots']):,} unique root_ids)")

    for desc, sql_cond, cohort_prefix in cohorts:
        for split in ['val', 'test']:
            other_split = 'test' if split == 'val' else 'val'
            p = f"{split}_{cohort_prefix}"
            total, total_roots = int(res[f"{p}_total"]), int(res[f"{p}_total_roots"])
            
            if total > 0:
                print(f"\n--- Split == '{split:<4}' | {desc} ({col_name} level) ---")
                print(f"  Total:               {total:>8,} rows | {total_roots:>8,} roots")
                
                states = [
                    ("mixed_train_only", "Mixed w/ train only:"),
                    (f"mixed_{other_split}_only", f"Mixed w/ {other_split:<4} only:"),
                    ("mixed_both", "Mixed w/ both:"),
                    ("pure", f"Purely {split:<4}:")
                ]
                
                for state, lbl in states:
                    r_cnt = int(res[f"{p}_{state}"])
                    rt_cnt = int(res[f"{p}_{state}_roots"])
                    print(f"  {lbl:<20} {r_cnt:>8,} rows | {rt_cnt:>8,} roots ({100 * r_cnt / total:>6.2f}%)")

# =============================================================================
# PART 2: VENN DIAGRAMS
# =============================================================================
print("\n" + "#"*95 + "\n VENN DIAGRAMS: PURE NON-TRAIN ROWS OVERLAP\n" + "#"*95)

# Materialize the heavy joins once instead of 6 times.
setup_query = """
CREATE TABLE non_train_purity AS
WITH 
root_aggs AS (SELECT root_id, bool_or(split = 'train') AS has_train FROM dataset WHERE root_id IS NOT NULL GROUP BY root_id),
rna30_aggs AS (SELECT rna_cluster_30, bool_or(split = 'train') AS has_train FROM dataset WHERE rna_cluster_30 IS NOT NULL GROUP BY rna_cluster_30),
rna70_aggs AS (SELECT rna_cluster_70, bool_or(split = 'train') AS has_train FROM dataset WHERE rna_cluster_70 IS NOT NULL GROUP BY rna_cluster_70)
SELECT 
    d.genome_feature_id, d.root_id, d.split, d.species, d.is_coding,
    (d.root_id IS NOT NULL AND NOT COALESCE(r.has_train, FALSE)) AS pure_root,
    (d.rna_cluster_30 IS NOT NULL AND NOT COALESCE(r30.has_train, FALSE)) AS pure_rna30,
    (d.rna_cluster_70 IS NOT NULL AND NOT COALESCE(r70.has_train, FALSE)) AS pure_rna70
FROM dataset d
LEFT JOIN root_aggs r ON d.root_id = r.root_id
LEFT JOIN rna30_aggs r30 ON d.rna_cluster_30 = r30.rna_cluster_30
LEFT JOIN rna70_aggs r70 ON d.rna_cluster_70 = r70.rna_cluster_70
WHERE d.split IN ('val', 'test');
"""
con.execute(setup_query)

for desc, sql_cond, _ in cohorts:
    print(f"\n{'='*70}\n{desc}\n{'='*70}")
    
    summary_query = f"""
    SELECT
        COUNT(*) AS total_rows,
        COUNT(DISTINCT root_id) AS total_roots,
        COUNT(CASE WHEN NOT pure_root AND NOT pure_rna30 AND NOT pure_rna70 THEN 1 END) AS leaked_rows,
        COUNT(DISTINCT CASE WHEN NOT pure_root AND NOT pure_rna30 AND NOT pure_rna70 THEN root_id END) AS leaked_roots,
        COUNT(CASE WHEN pure_root AND pure_rna30 AND pure_rna70 THEN 1 END) AS pure_rows,
        COUNT(DISTINCT CASE WHEN pure_root AND pure_rna30 AND pure_rna70 THEN root_id END) AS pure_roots
    FROM non_train_purity WHERE {sql_cond}
    """
    summary = con.execute(summary_query).df().fillna(0).iloc[0]
    
    venn_query = f"""
    SELECT pure_root::INT AS r, pure_rna30::INT AS r30, pure_rna70::INT AS r70, COUNT(*) AS cnt
    FROM non_train_purity WHERE {sql_cond}
    GROUP BY pure_root, pure_rna30, pure_rna70
    """
    df_counts = con.execute(venn_query).df()
    
    if df_counts.empty or int(summary['total_rows']) == 0:
        print("No validation/test data found for this cohort.")
        continue
    
    venn_subsets = {
        f"{row['r']}{row['r30']}{row['r70']}": row['cnt'] 
        for _, row in df_counts.iterrows() if not (row['r'] == 0 and row['r30'] == 0 and row['r70'] == 0)
    }
    
    print(f"Total non-train evaluated:                   {int(summary['total_rows']):>8,} rows | {int(summary['total_roots']):>8,} roots")
    print(f"Completely leaked (unsafe on all 3 metrics): {int(summary['leaked_rows']):>8,} rows | {int(summary['leaked_roots']):>8,} roots")
    print(f"Perfectly pure (safe on all 3 metrics):      {int(summary['pure_rows']):>8,} rows | {int(summary['pure_roots']):>8,} roots")
    
    plt.figure(figsize=(8, 6))
    v = venn3(subsets=venn_subsets, set_labels=('Root', 'RNA 30', 'RNA 70'))
    plt.title(f"Pure (Unleaked) Non-Train Rows:\n{desc}")
    plt.show()

# =============================================================================
# PART 3: RNA 30 vs RNA 70 COMPARISON
# =============================================================================
print("\n" + "#"*95 + "\n CLUSTER COMPARISON: RNA 30 vs RNA 70 (PURE ROWS)\n" + "#"*95)

comp_query = """
SELECT 
    SUM(CASE WHEN pure_rna30 THEN 1 ELSE 0 END) AS rna30_all,
    SUM(CASE WHEN pure_rna70 THEN 1 ELSE 0 END) AS rna70_all,
    
    SUM(CASE WHEN pure_rna30 AND species = 'Homo sapiens' THEN 1 ELSE 0 END) AS rna30_hs,
    SUM(CASE WHEN pure_rna70 AND species = 'Homo sapiens' THEN 1 ELSE 0 END) AS rna70_hs,
    
    SUM(CASE WHEN pure_rna30 AND species = 'Homo sapiens' AND is_coding = TRUE THEN 1 ELSE 0 END) AS rna30_hs_coding,
    SUM(CASE WHEN pure_rna70 AND species = 'Homo sapiens' AND is_coding = TRUE THEN 1 ELSE 0 END) AS rna70_hs_coding
FROM non_train_purity;
"""

comp_df = con.execute(comp_query).df().fillna(0).iloc[0]

comparison_metrics = [
    ("All Species (Total Pure Rows)", "rna30_all", "rna70_all"),
    ("Homo Sapiens (Total Pure Rows)", "rna30_hs", "rna70_hs"),
    ("Homo Sapiens Coding (Total Pure Rows)", "rna30_hs_coding", "rna70_hs_coding")
]

for label, col30, col70 in comparison_metrics:
    val30 = int(comp_df[col30])
    val70 = int(comp_df[col70])
    diff = abs(val70 - val30)
    
    if val70 > val30:
        winner = "RNA 70"
    elif val30 > val70:
        winner = "RNA 30"
    else:
        winner = "Tie"

    print(f"\n--- {label} ---")
    print(f"  RNA 30 Pure Rows: {val30:>10,}")
    print(f"  RNA 70 Pure Rows: {val70:>10,}")
    print(f"  Winner:           {winner} (by {diff:,} rows)")

# =============================================================================
# PART 4: EXTRACT CLEAN DATAFRAMES
# =============================================================================
print("\n" + "#"*95 + "\n EXTRACTING CLEAN DATAFRAMES (BEST & GOOD)\n" + "#"*95)

# df_non_train_best: Clean under ALL THREE metrics (root, rna30, rna70)
query_best = """
SELECT genome_feature_id, root_id, species, is_coding 
FROM non_train_purity 
WHERE pure_root AND pure_rna30 AND pure_rna70;
"""
df_non_train_best = con.execute(query_best).df()

# df_non_train_good: Clean under RNA 30 alone
query_good = """
SELECT genome_feature_id, root_id, species, is_coding 
FROM non_train_purity 
WHERE pure_rna30;
"""
df_non_train_good = con.execute(query_good).df()

print(f"Created 'df_non_train_best' (perfect purity): {len(df_non_train_best):>10,} rows.")
print(f"Created 'df_non_train_good' (RNA 30 purity):  {len(df_non_train_good):>10,} rows.")

# %%
# save them to ~/tmp/df_non_train_best.parquet and ~/tmp/df_non_train_good.parquet
df_non_train_best.to_parquet(os.path.expanduser("~/tmp/df_non_train_best.parquet"))
df_non_train_good.to_parquet(os.path.expanduser("~/tmp/df_non_train_good.parquet"))

print("Saved to ~/tmp/df_non_train_best.parquet and ~/tmp/df_non_train_good.parquet")
# %%
# this was then copied back to data/intermediate/master_ids/3.0/parquet/ with the same names.