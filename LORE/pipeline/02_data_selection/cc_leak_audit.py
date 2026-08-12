#!/usr/bin/env python3
"""
Transitive (multi-hop) leakage audit of the master_ids splits.

The per-key audit in 05_data_leak_counter.py is ONE-HOP: it asks whether a
val/test row shares a cluster id directly with a train row. That misses paths
of the form

    val row A --rna_cluster_30-- row B --protein_cluster_70-- train row C

where A shares no key with train at all, but is reachable through B. Because
protein_cluster_70 provably crosses protein_cluster_30 boundaries (the two
clusterings are not nested), such bridges exist by construction.

This script builds the bipartite row <-> key-value graph and runs a BFS seeded
from every train row, recording how many val/test rows fall in a train-connected
component AT EACH HOP DISTANCE. The hop profile is the actual deliverable: it
distinguishes "a real, bounded tightening" from "the graph percolates and every
row is nominally contaminated", which would make the multi-hop criterion useless.

Modes
  keys3  root_id, rna_cluster_30, rna_cluster_70
         -- validation: hop-1 survivors MUST equal df_non_train_best (2,589,003)
  keys5  + protein_cluster_30, protein_cluster_70
  keys6  + uniprot_id (near-unique per row; included for completeness)

Usage:  cc_leak_audit.py [keys3|keys5|keys6] [--outdir DIR] [--memory 220GB] [--threads 16]
"""
import argparse
import json
import os
import sys
import time

import duckdb

from lore.paths import get_path

SRC = str(get_path("data", "intermediate", "master_ids", "3.0", fmt="parquet")
          / "dataset.parquet")

KEYSETS = {
    "keys3": ["root_id", "rna_cluster_30", "rna_cluster_70"],
    "keys5": ["root_id", "rna_cluster_30", "rna_cluster_70",
              "protein_cluster_30", "protein_cluster_70"],
    "keys6": ["root_id", "rna_cluster_30", "rna_cluster_70",
              "protein_cluster_30", "protein_cluster_70", "uniprot_id"],
}
COHORTS = [("all", "1=1"),
           ("human", "species='Homo sapiens'"),
           ("human_coding", "species='Homo sapiens' AND is_coding=TRUE")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=sorted(KEYSETS), nargs="?", default="keys5")
    ap.add_argument("--outdir", default="./cc_audit")
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    ap.add_argument("--memory", default="220GB")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-hops", type=int, default=60)
    args = ap.parse_args()

    keys = KEYSETS[args.mode]
    os.makedirs(args.outdir, exist_ok=True)
    scratch = os.path.join(args.scratch, f"cc_{args.mode}")
    os.makedirs(scratch, exist_ok=True)
    t_start = time.time()

    def log(m):
        print(f"[{time.time()-t_start:8.1f}s] {m}", flush=True)

    con = duckdb.connect(os.path.join(scratch, "cc.duckdb"))
    con.execute(f"SET memory_limit='{args.memory}'")
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET temp_directory='{scratch}/spill'")
    con.execute("SET preserve_insertion_order=false")

    log(f"mode={args.mode} keys={keys}")
    con.execute(f"CREATE OR REPLACE VIEW src AS SELECT * FROM read_parquet('{SRC}')")

    # ---- integer-encode every key so the BFS moves 4-byte ints, not strings ----
    log("encoding keys to dense integers ...")
    sel, joins = [], []
    for i, k in enumerate(keys):
        con.execute(f"""
        CREATE OR REPLACE TABLE map_{i} AS
        SELECT v, CAST(row_number() OVER () AS INTEGER) AS id
        FROM (SELECT DISTINCT {k} AS v FROM src WHERE {k} IS NOT NULL)
        """)
        n = con.execute(f"SELECT count(*) FROM map_{i}").fetchone()[0]
        log(f"  {k:<20} {n:>12,} distinct values")
        sel.append(f"m{i}.id AS k{i}")
        joins.append(f"LEFT JOIN map_{i} m{i} ON s.{k} = m{i}.v")

    log("materializing compact row table ...")
    con.execute(f"""
    CREATE OR REPLACE TABLE rows AS
    SELECT CAST(row_number() OVER () AS BIGINT) AS rid,
           (s.split = 'train') AS is_train,
           (s.split IN ('val','test')) AS is_vt,
           s.species, s.is_coding, s.root_id,
           {', '.join(sel)},
           (s.split = 'train') AS reached,
           CAST(CASE WHEN s.split='train' THEN 0 ELSE -1 END AS INTEGER) AS hop
    FROM src s {' '.join(joins)}
    """)
    for i in range(len(keys)):
        con.execute(f"DROP TABLE map_{i}")
    tot, ntr, nvt = con.execute(
        "SELECT count(*), sum(is_train::INT), sum(is_vt::INT) FROM rows").fetchone()
    log(f"  rows {tot:,} | train {ntr:,} | val+test {nvt:,}")

    # ---- BFS over the bipartite row <-> key-value graph ----
    # C_i holds every value of key i that belongs to an already-reached row.
    log("seeding frontier from train rows ...")
    for i in range(len(keys)):
        con.execute(f"CREATE OR REPLACE TABLE C_{i} AS "
                    f"SELECT DISTINCT k{i} AS v FROM rows WHERE reached AND k{i} IS NOT NULL")

    profile = []
    for hop in range(1, args.max_hops + 1):
        cond = " OR ".join(
            f"(r.k{i} IS NOT NULL AND EXISTS (SELECT 1 FROM C_{i} c WHERE c.v = r.k{i}))"
            for i in range(len(keys)))
        con.execute(f"""
        CREATE OR REPLACE TABLE frontier AS
        SELECT r.rid, {', '.join(f'r.k{i}' for i in range(len(keys)))}
        FROM rows r WHERE NOT r.reached AND ({cond})
        """)
        nnew = con.execute("SELECT count(*) FROM frontier").fetchone()[0]
        if nnew == 0:
            log(f"hop {hop}: converged")
            break
        con.execute(f"UPDATE rows SET reached = TRUE, hop = {hop} "
                    "WHERE rid IN (SELECT rid FROM frontier)")
        for i in range(len(keys)):
            con.execute(f"INSERT INTO C_{i} SELECT DISTINCT k{i} FROM frontier "
                        f"WHERE k{i} IS NOT NULL AND k{i} NOT IN (SELECT v FROM C_{i})")
        row = {"hop": hop, "newly_reached": int(nnew)}
        for cname, ccond in COHORTS:
            r = con.execute(f"SELECT count(*) FROM rows WHERE is_vt AND hop={hop} AND {ccond}").fetchone()[0]
            c = con.execute(f"SELECT count(*) FROM rows WHERE is_vt AND reached AND {ccond}").fetchone()[0]
            t = con.execute(f"SELECT count(*) FROM rows WHERE is_vt AND {ccond}").fetchone()[0]
            row[f"{cname}_new"] = int(r)
            row[f"{cname}_cum_pct"] = round(100.0 * c / t, 4) if t else None
        profile.append(row)
        log(f"hop {hop}: +{nnew:,} rows | val/test contaminated: "
            + " ".join(f"{c}={row[f'{c}_cum_pct']}%" for c, _ in COHORTS))

    # ---- survivors ----
    log("\n" + "=" * 78)
    log("SURVIVORS (val/test rows never reached from train)")
    summary = {"mode": args.mode, "keys": keys, "profile": profile, "survivors": {}}
    for cname, ccond in COHORTS:
        t, s = con.execute(
            f"SELECT count(*), sum(CASE WHEN NOT reached THEN 1 ELSE 0 END) "
            f"FROM rows WHERE is_vt AND {ccond}").fetchone()
        s = int(s or 0)
        summary["survivors"][cname] = {"total": int(t), "surviving": s,
                                       "pct": round(100.0 * s / t, 4) if t else None}
        log(f"  {cname:<14} {s:>12,} of {int(t):>12,}  ({100.0*s/t:6.3f}%)")

    # Validation against df_non_train_best, which requires every key to be NON-NULL:
    # a NULL-key row is IMPURE there, but merely unreachable here. Comparing raw
    # survivor counts therefore mismatches by the NULL-key population (~59M rows).
    # The like-for-like quantity is hop-1 survivors RESTRICTED to rows carrying all keys.
    if args.mode == "keys3" and profile:
        keyed = " AND ".join(f"k{i} IS NOT NULL" for i in range(len(keys)))
        h1 = con.execute(f"SELECT count(*) FROM rows WHERE is_vt AND {keyed} "
                         "AND (hop < 0 OR hop > 1)").fetchone()[0]
        raw = con.execute("SELECT count(*) FROM rows WHERE is_vt AND (hop < 0 OR hop > 1)").fetchone()[0]
        log(f"\n  VALIDATION hop-1 survivors carrying all keys = {h1:,} "
            f"(df_non_train_best = 2,589,003; {'MATCH' if h1 == 2589003 else 'MISMATCH -- investigate'})")
        log(f"  (raw hop-1 survivors incl. NULL-key rows = {raw:,}; the difference is the "
            f"NULL-key population, which is not comparable to df_non_train_best)")
        summary["validation_hop1_survivors_keyed"] = int(h1)
        summary["validation_hop1_survivors_raw"] = int(raw)

    con.execute(f"""COPY (SELECT root_id, species, is_coding FROM rows
                          WHERE is_vt AND NOT reached)
                    TO '{args.outdir}/survivors_{args.mode}.parquet' (FORMAT PARQUET, CODEC ZSTD)""")
    with open(f"{args.outdir}/cc_profile_{args.mode}.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {args.outdir}/survivors_{args.mode}.parquet")
    log(f"wrote {args.outdir}/cc_profile_{args.mode}.json")
    log("DONE")


if __name__ == "__main__":
    sys.exit(main())
