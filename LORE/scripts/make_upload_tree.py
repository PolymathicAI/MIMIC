"""Stage a LORE release tree for `hf upload-large-folder`, without copying 445 GB.

Two problems this solves:

1. `upload_large_folder` writes its resume state to ``<local_path>/.cache/huggingface``,
   so the folder you hand it must be **writable**. The released parquet lives in a
   directory owned by another user, so it cannot be uploaded in place.
2. We want to omit subsets (``corpus``, and optionally the unresolved-licence assays)
   without duplicating the data.

So we build a tree of real directories containing **symlinks to individual files**.
Directories are real on purpose: ``upload_large_folder`` enumerates with
``Path.glob("**/*")``, which does not descend into symlinked directories, but does
resolve symlinked files (``is_file()`` follows the link).

Usage::

    python make_upload_tree.py --dest ~/LORE_upload
    python make_upload_tree.py --dest ~/LORE_smoke --max-shards-per-split 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _layout import SPLITS, _numeric_key, shard_names

DEFAULT_SRC = Path("/mnt/ceph/users/polymathic/bio/central_dogma/data/final/nerv/4.0/parquet")

# Subsets omitted from the release. `corpus` is the CC-BY-SA share-alike component and
# carries no central-dogma content; it is dropped by decision, not by licence failure.
DEFAULT_EXCLUDE = ("corpus",)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--exclude", action="append", default=None,
                    help=f"subset to omit (repeatable); default {DEFAULT_EXCLUDE}")
    ap.add_argument("--max-shards-per-split", type=int, default=None,
                    help="cap shards per split, for a small smoke-test tree")
    args = ap.parse_args()

    exclude = set(args.exclude if args.exclude is not None else DEFAULT_EXCLUDE)
    dest = args.dest.expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    subsets = sorted(d for d in args.src.iterdir() if d.is_dir())
    linked = skipped_bytes = 0
    total_bytes = 0
    manifest: dict[str, dict[str, int]] = {}

    for sub in subsets:
        if sub.name in exclude:
            skipped_bytes += sum(f.stat().st_size
                                 for s in SPLITS for f in (sub / s).glob("*.parquet"))
            print(f"  omit {sub.name}")
            continue
        # Flat within the subset, with information-free filenames: the published data
        # carries no trace of the split at all. See scripts/_layout.py.
        names = shard_names(sub)
        counts = {}
        out = dest / "data" / sub.name
        out.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            files = sorted((sub / split).glob("*.parquet"), key=lambda p: _numeric_key(p.name))
            if not files:
                continue
            if args.max_shards_per_split:
                files = files[: args.max_shards_per_split]
            for f in files:
                link = out / names[(split, f.name)]
                if not link.is_symlink():
                    link.symlink_to(f)
                total_bytes += f.stat().st_size
                linked += 1
            counts[split] = len(files)
        manifest[sub.name] = counts

    # Written *beside* the tree, not inside it. It records per-split shard counts, and
    # the published data is meant to carry no split information at all; `subsets.json`
    # in the index repo is the user-facing manifest.
    side = dest.parent / f"{dest.name}.manifest.json"
    side.write_text(json.dumps(manifest, indent=1))

    # The dataset card is version-controlled in cards/ and linked in, not written here:
    # these trees get deleted and rebuilt, and a card that lives only inside one is lost
    # with it.
    card = Path(__file__).resolve().parent.parent / "cards" / "LORE.md"
    link = dest / "README.md"
    if card.exists():
        link.unlink(missing_ok=True)
        link.symlink_to(card)
        print(f"  card <- {card}")
    else:
        print(f"  WARNING: no card at {card}; repo will have no README")
    print(f"\n{linked:,} file symlinks over {len(manifest)} subsets -> {dest}")
    print(f"payload {total_bytes / 1e9:.1f} GB   omitted {skipped_bytes / 1e9:.2f} GB "
          f"({', '.join(sorted(exclude)) or 'nothing'})")


if __name__ == "__main__":
    main()
