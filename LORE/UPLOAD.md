# Uploading LORE to the Hub

Everything here is staged and ready; the commands below are the whole procedure. They
are written to be run **by a human on a workstation** — see *Where to run this*.

## Where to run this

**A Slurm batch job is fine.** Compute nodes *do* have outbound access to the Hub —
verified from `worker7003`: `huggingface.co:443` and `cas-server.xethub.hf.co:443` both
connect. Use `scripts/upload_lore.sbatch`.

A workstation under `tmux` works too, but the batch job is better: it survives your
ssh session, and `upload-large-folder` is resumable, so a timeout or preemption costs
progress rather than correctness.

**You do not need a big-memory node.** `upload-large-folder` streams and hashes in
chunks; it is network- and CPU-bound. The sbatch asks for 16 CPUs / 64 GB, which is
generous. Wall time is the resource that actually matters — the script asks for 2 days
on `gen` (limit 7).

## One-time setup

```bash
printf '%s' 'hf_...' > ~/.hf_token && chmod 600 ~/.hf_token
```

A write-scoped token for the `polymathic-ai` org. The sbatch reads this file into
`HF_TOKEN`; it is not passed as a command-line argument, because process arguments are
readable by every user on the node via `ps`.

### Two cache settings the script sets for you, and why

| variable | value | why |
|:-|:-|:-|
| `HF_HOME` | `/mnt/home/$USER/.cache/huggingface` | defaults into `/dev/shm` here — node-local RAM, so a token written by `hf auth login` neither persists nor is visible to a job on another node |
| `HF_XET_CACHE` | `/tmp/$USER/xet-cache` | node-local NVMe, ~1.8 TB free. **Not** `/dev/shm` (RAM, counts against `--mem`); **not** ceph (enough small files to exhaust the ceph file allocation) |

Note `/home/$USER` does **not** exist on compute nodes — `/tmp` is the local NVMe there.

## What is staged

| tree | contents | size |
|:-|:-|-:|
| `~/LORE_smoke` | 1 shard per split × 40 subsets, + card | 17.7 GB |
| `~/LORE_upload` | all 4,393 shards × 40 subsets, + card | 445.1 GB |
| `~/LORE_index_upload` | `index.parquet`, `subsets.json`, + card | 0.83 GB |

All three hold **symlinks** to the parquet on ceph, not copies. `upload-large-folder`
resolves them. Regenerate any of them with `scripts/make_upload_tree.py` /
`scripts/make_index_release.py`.

`corpus` is omitted from all three (the CC-BY-SA component). The index was rebuilt to
match, so it does not advertise rows that are not there: 60,340,407 → 59,839,026.

## Step 1 — index + smoke test first

Push the index and the 17.7 GB sample before the full 445 GB. This is what tells you
whether the cards render, whether the dataset viewer copes with `list<uint8>` token
columns, and whether `LoreIndex.from_hf` resolves — all of which are cheap to fix now
and expensive to fix after a multi-day upload.

```bash
cd ~/MIMIC/LORE
sbatch scripts/upload_lore.sbatch index     # ~0.83 GB, minutes
sbatch scripts/upload_lore.sbatch smoke     # ~17.7 GB
```

Both go to **private** repos. Watch with `tail -f slurm_logs/lore-upload-<jobid>.out`.

`LORE-smoke` is a throwaway repo, deliberately separate from `polymathic-ai/LORE`: the
`cage` and `rasp2` licences are still unresolved, so the real repo should not accumulate
that data yet. Delete `LORE-smoke` once the full push is done.

> Uploading to a private repo still transfers the data to a third party. For a 17.7 GB
> internal preview that is very likely fine, but it is a conscious call, not a
> technicality. To preview without the unresolved-licence assays, restage with
> `python scripts/make_upload_tree.py --dest ~/LORE_smoke_clean --max-shards-per-split 1
> --exclude corpus --exclude <each cage/rasp2 subset>`.

Then check, on the Hub:

- both cards render, and the YAML front-matter parses (a bad `configs:` block shows as a
  card-metadata error);
- the **dataset viewer** on the three declared configs — `rna_splice`,
  `protein_structure`, `splice_prediction`. Expect trouble here: rows are long token-id
  arrays and the viewer has per-row size limits. If it fails, that is information, not a
  blocker — say so on the card rather than reshaping the data.
- `LoreIndex.from_hf` against the private index repo.

## Step 2 — the full push

Only after step 1 looks right.

```bash
sbatch scripts/upload_lore.sbatch full
```

**Measured throughput: 203 MB/s** (the smoke push did 17.7 GB in 87 s, 8 workers). At
that rate 445 GB is **~36 minutes**; allow a couple of hours for larger shards and commit
overhead. This is far quicker than a bulk transfer of this size suggests, so the 2-day
time limit in the script is slack, not an estimate.

Re-running is safe and skips completed files, so if anything interrupts it, resubmit.

**Quota.** 445 GB is over the ~300 GB per-repo soft recommendation, and private storage
counts against the org allowance. Ask the Hub for a quota bump on `polymathic-ai` before
step 2, or it will fail partway. Largest single shard is 4.13 GB, well under the 50 GB
per-file cap, and 4,393 files is far under the 100k limit — size is the only concern.

## Troubleshooting

**`UnboundLocalError: cannot access local variable 'metadata'`** at
`_upload_large_folder.py:229`. A previous run was **cancelled or killed mid-write**,
leaving a half-written file in the tree's resume state. The next run crashes reading it.

```bash
rm -rf <tree>/.cache/huggingface     # e.g. ~/LORE_smoke/.cache/huggingface
```

Then resubmit. This is HF's own documented reinitialize procedure — the state is pure
scratch, and clearing it only costs re-hashing. Do it **only when no upload is running**
against that tree (`squeue -u $USER -n lore-upload`); deleting it under a live process is
what creates the corruption in the first place.

Note the `.cache/huggingface` directory is *inside* the staging tree, so `find`-based
file counts in the job header will read higher than the real payload after a first
attempt (223 vs 100 for smoke). That is cosmetic — `upload-large-folder` ignores it, as
"Found 100 candidate files" in the same log confirms.

## Before this goes public

Not blockers for a private push; blockers for publication.

1. **FANTOM5 `cage`** licence — unresolved, gates 27.8% of rows.
2. **RASP2 `rasp2`** licence — unresolved, gates 2.9%.
3. Decide whether the contaminated `split` column ships.
4. `LORE-smoke` should be deleted once the full repo is up.

If either licence comes back negative, drop the **column**, not the subsets:
`scripts/make_upload_tree.py --exclude` takes subset names, but the cheaper fix is to
rewrite those shards without the one column, which keeps every row. Six of the ten
affected subset signatures then collapse onto subsets that already exist.
