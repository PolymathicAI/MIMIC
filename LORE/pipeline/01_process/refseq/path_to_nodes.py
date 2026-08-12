#!/usr/bin/env python3
"""
Parse NCBI Taxonomy nodes.dmp into:
  - gc_by_taxid: {taxid -> genetic_code_id}
  - mgc_by_taxid: {taxid -> mitochondrial_genetic_code_id}

Format notes:
- Columns are separated by "<TAB>|<TAB>" and lines end with "<TAB>|<NEWLINE>".
- Relevant columns (0-based index):
    0: tax_id
    6: genetic code id (nuclear/default)
    8: mitochondrial genetic code id

This matches the Taxonomy 'taxdump' README.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse
import json
import sys

SEP = "\t|\t"

def parse_nodes_dmp(nodes_path: str) -> Tuple[Dict[int, int], Dict[int, int]]:
    gc_by_taxid: Dict[int, int] = {}
    mgc_by_taxid: Dict[int, int] = {}

    with open(nodes_path, "r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line:
                continue
            # Strip newline and the trailing "\t|" that ends every record
            line = line.rstrip("\n")
            if line.endswith("\t|"):
                line = line[:-2]

            parts = [p.strip() for p in line.split(SEP)]

            # Expect at least 9 columns (we only need 0,6,8); nodes.dmp typically has ~13–14
            if len(parts) < 9:
                # Fall back to a more tolerant split if the line is malformed
                parts = [p.strip() for p in line.replace("\t|\t", SEP).split(SEP)]
                if len(parts) < 9:
                    # Skip or raise, depending on your tolerance
                    # print(f"Warning: malformed nodes.dmp line {line_no}", file=sys.stderr)
                    continue

            try:
                taxid = int(parts[0])
            except ValueError:
                # Skip headers/odd lines (shouldn't happen in nodes.dmp)
                continue

            def to_int_or_none(s: str) -> Optional[int]:
                s = s.strip()
                if s == "" or s == r"\N":
                    return None
                try:
                    return int(s)
                except ValueError:
                    return None

            gc_id = to_int_or_none(parts[6])
            mgc_id = to_int_or_none(parts[8])

            if gc_id is not None:
                gc_by_taxid[taxid] = gc_id
            if mgc_id is not None:
                mgc_by_taxid[taxid] = mgc_id

    return gc_by_taxid, mgc_by_taxid


def main():
    ap = argparse.ArgumentParser(description="Build {taxid -> genetic code id, mito genetic code id} maps from nodes.dmp")
    ap.add_argument("--nodes", required=True, help="Path to nodes.dmp")
    ap.add_argument("--out-json", default=None, help="Optional path to write JSON with keys: gc_by_taxid, mgc_by_taxid")
    args = ap.parse_args()

    gc_by_taxid, mgc_by_taxid = parse_nodes_dmp(args.nodes)

    if args.out_json:
        out = {"gc_by_taxid": gc_by_taxid, "mgc_by_taxid": mgc_by_taxid}
        Path(args.out_json).write_text(json.dumps(out), encoding="utf-8")
    else:
        # Print quick stats + tiny sample
        print(f"Loaded taxids: {len(gc_by_taxid)} with GC, {len(mgc_by_taxid)} with Mito GC")
        # show a couple of entries
        for i, (tx, gc) in enumerate(gc_by_taxid.items()):
            mito = mgc_by_taxid.get(tx)
            print(f"{tx}\tgc={gc}\tmito={mito}")
            if i >= 9:
                break

if __name__ == "__main__":
    main()

