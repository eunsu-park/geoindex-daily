#!/bin/bash
# Run a list of lead-1 image runs on the remote GPU server and leave a self-checking output folder.
# Procedure: geoindex-main docs/workflows/remote-gpu-jobs.md. Run it by its absolute path, inside
# `conda activate geoindex`.
#
#   bash ~/GitHub/geoindex/geoindex-daily/scripts/remote_job.sh <pack> <out> <runlist> <gpu> [cores]
#   e.g. nohup bash ~/GitHub/geoindex/geoindex-daily/scripts/remote_job.sh ~/geoindex-packs/lead1-w41 \
#            ~/l5-full-gpu0 ~/GitHub/geoindex/geoindex-daily/configs/remote/l5_full_gpu0.txt 2 16-23,48-55 \
#            > ~/l5-full-gpu0.log 2>&1 &
#
# <runlist>: one run group per line, "<tag> <split> <L> <lead1_image.py arguments ...>" (e.g. --arm B
# --channels hmi_m45 --lasco12 --lasco-cfg lasco12f --grad-checkpoint), seeds 0 1 2.
# Writes <out>/env.txt (one block per start), job.log, the lead1_image.py outputs and SHA256SUMS, and
# ends job.log with "== done" — judge or copy a job only after that line. Rerunning the same command
# resumes: a seed whose prediction file exists is skipped. GEOINDEX_EXTRA_ARGS (env, recorded in env.txt)
# is appended to every run, e.g. GEOINDEX_EXTRA_ARGS="--limit 16 --epochs 1" for a smoke test.
set -u
PACK=$(realpath "$1"); OUT=$2; LIST=$(realpath "$3"); GPU=$4; CORES=${5:-}
REPO=$(cd "$(dirname "$(realpath "$0")")/.." && pwd)
mkdir -p "$OUT"; OUT=$(realpath "$OUT"); LOG=$OUT/job.log
cp "$LIST" "$OUT/runlist.txt"

{
  echo "== start   $(date -u +%FT%TZ)"
  echo "host      $(hostname)"
  echo "commit    $(git -C "$REPO" rev-parse HEAD)"
  echo "dirty     $(git -C "$REPO" status --porcelain | wc -l) changed files"
  echo "pack      $PACK"
  echo "runlist   $LIST ($(grep -cv '^\s*\(#\|$\)' "$LIST") lines)"
  echo "gpu       $GPU  cores ${CORES:-unpinned}"
  echo "extra     ${GEOINDEX_EXTRA_ARGS:-none}"
  python -c "import sys, torch; print('python   ', sys.version.split()[0], ' torch', torch.__version__, ' cuda', torch.version.cuda)"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
} >> "$OUT/env.txt" 2>&1

echo "$(date -u +%FT%TZ) checking pack checksums" >> "$LOG"
(cd "$PACK" && sha256sum -c --quiet SHA256SUMS) >> "$LOG" 2>&1 || { echo "!!! pack checksum failed: re-run the pack rsync" >> "$LOG"; exit 1; }
echo "$(date -u +%FT%TZ) CPU self-test" >> "$LOG"
# from the repo root: `python -m pytest` puts the current directory on the import path, and the job may
# start anywhere (found on the server 2026-10-25, where it started in ~)
(cd "$REPO" && CUDA_VISIBLE_DEVICES="" python -m pytest -q tests) >> "$LOG" 2>&1 || { echo "!!! self-test failed" >> "$LOG"; exit 1; }

export GEOINDEX_DAILY_DATA=$PACK NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       CUDA_VISIBLE_DEVICES=$GPU
PIN=(); [ -n "$CORES" ] && PIN=(taskset -c "$CORES")
while read -r TAG SPLIT L ARGS; do
  [[ -z "$TAG" || "$TAG" == \#* ]] && continue
  SEEDS=""
  for s in 0 1 2; do
    ls "$OUT/${TAG}_preds_${SPLIT}_"*"_L${L}_s${s}.npz" > /dev/null 2>&1 || SEEDS="$SEEDS $s"
  done
  if [ -z "$SEEDS" ]; then
    echo "$(date -u +%FT%TZ) === $TAG split $SPLIT L=$L: done before, skipped" >> "$LOG"; continue
  fi
  echo "$(date -u +%FT%TZ) === $TAG split $SPLIT L=$L seeds$SEEDS $ARGS" >> "$LOG"
  # $ARGS and $SEEDS unquoted on purpose: one word per argument
  ${PIN[@]+"${PIN[@]}"} python "$REPO/scripts/lead1_image.py" --config "$REPO/configs/ap_lead1.yaml" \
      --split "$SPLIT" --lengths "$L" --seeds $SEEDS $ARGS --tag "$TAG" --out-dir "$OUT" \
      --batch 8 --workers 6 ${GEOINDEX_EXTRA_ARGS:-} < /dev/null 2>&1 | grep -vi warning >> "$LOG"   # stdin stays with the run list
  [ "${PIPESTATUS[0]}" -eq 0 ] || echo "!!! $TAG split $SPLIT L=$L failed" >> "$LOG"
done < "$LIST"

(cd "$OUT" && find . -type f ! -name SHA256SUMS ! -name job.log | sort | xargs sha256sum > SHA256SUMS)
echo "== done $(date -u +%FT%TZ)" >> "$LOG"
