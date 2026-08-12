#!/bin/bash
#SBATCH --job-name=lore-upload
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/lore-upload-%j.out
#SBATCH --error=slurm_logs/lore-upload-%j.out
#
# Push a staged LORE tree to the Hugging Face Hub.
#
#   sbatch scripts/upload_lore.sbatch index     # ~0.83 GB
#   sbatch scripts/upload_lore.sbatch smoke     # ~17.7 GB
#   sbatch scripts/upload_lore.sbatch full      # ~445.1 GB
#
# Resumable: re-running skips files already uploaded, so a timeout or a preemption
# costs progress, not correctness.
#
# Resources are deliberately modest. `upload-large-folder` streams and hashes in
# chunks -- it is network- and CPU-bound, not memory-bound. A big-memory node buys
# nothing here; wall time is the resource that matters.

set -euo pipefail

# NB: no braces in this message. A '}' inside ${1:?...} terminates the expansion
# early, which silently appends the stray '}' to MODE.
MODE="${1:?usage: sbatch upload_lore.sbatch index|smoke|full}"

VENV=/mnt/home/jkovalic/central-dogma/venvs/central
ORG=polymathic-ai

case "$MODE" in
  index) LOCAL=~/LORE_index_upload ; REPO=$ORG/LORE-index ; WORKERS=4  ;;
  smoke) LOCAL=~/LORE_smoke        ; REPO=$ORG/LORE-smoke ; WORKERS=8  ;;
  full)  LOCAL=~/LORE_upload       ; REPO=$ORG/LORE       ; WORKERS=16 ;;
  *) echo "unknown mode: $MODE (want index|smoke|full)" >&2; exit 2 ;;
esac

# --- credentials -------------------------------------------------------------
# The token is read from a file you create yourself, mode 600. It is deliberately
# NOT passed on the command line: process arguments are visible to every user on
# the node via `ps`, and to Slurm accounting.
#
#   printf '%s' 'hf_...' > ~/.hf_token && chmod 600 ~/.hf_token
#
if [[ ! -f ~/.hf_token ]]; then
  echo "missing ~/.hf_token -- create it (chmod 600) before submitting" >&2
  exit 1
fi
HF_TOKEN="$(< ~/.hf_token)"
export HF_TOKEN

# --- caches ------------------------------------------------------------------
# HF_HOME defaults into /dev/shm here, which is node-local RAM: it does not persist
# between jobs and counts against this job's memory cgroup. Point it at home.
export HF_HOME=/mnt/home/$USER/.cache/huggingface
mkdir -p "$HF_HOME"

# The Xet chunk cache must go on node-local NVMe. Not /dev/shm (RAM, would eat the
# --mem allocation), and NOT ceph (it creates enough small files to blow through the
# ceph file allocation). /tmp on a compute node is ~1.8 TB of local NVMe.
export HF_XET_CACHE=/tmp/$USER/xet-cache
mkdir -p "$HF_XET_CACHE"

mkdir -p slurm_logs

# --- go ----------------------------------------------------------------------
source "$VENV/bin/activate"

echo "host       : $(hostname)"
echo "mode       : $MODE"
echo "local      : $LOCAL"
echo "repo       : $REPO (private)"
echo "workers    : $WORKERS"
echo "HF_HOME    : $HF_HOME"
echo "XET cache  : $HF_XET_CACHE ($(df -h /tmp | awk 'NR==2{print $4}') free on /tmp)"
echo "payload    : $(du -shL "$LOCAL" 2>/dev/null | cut -f1) over \
$(find "$LOCAL" -type f -o -type l | wc -l) entries"
echo "started    : $(date -Is)"
echo

# Resolve-only check: `LORE_DRYRUN=1 bash scripts/upload_lore.sbatch smoke` prints the
# real parsed values and exits before touching the network. Use it after editing this
# file -- `bash -n` accepts plenty of things that are wrong at runtime.
if [[ -n "${LORE_DRYRUN:-}" ]]; then
  echo "DRYRUN: would run: hf upload-large-folder '$REPO' '$LOCAL' \
--repo-type dataset --private --num-workers $WORKERS"
  exit 0
fi

hf upload-large-folder "$REPO" "$LOCAL" \
    --repo-type dataset \
    --private \
    --num-workers "$WORKERS" \
    --no-bars

echo
echo "finished   : $(date -Is)"
