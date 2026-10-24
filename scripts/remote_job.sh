#!/bin/bash
# Run a list of lead-1 image runs on the remote GPU server and leave a self-checking output folder.
# Procedure: geoindex-main docs/workflows/remote-gpu-jobs.md.
#
#   bash scripts/remote_job.sh <pack> <out> <runlist> <gpu> [cores]
#   e.g. bash scripts/remote_job.sh ~/geoindex/packs/lead1-w41 ~/geoindex/out/l5-s1-gpu0 \
#            configs/remote/l5_stage1_gpu0.txt 0 0-7,32-39
#
# <runlist>: one run group per line, "<tag> <split> <L> <channel> [<channel> ...]", seeds 0 1 2,
# arm B (Ap + X-ray). Writes <out>/env.txt, job.log, the lead1_image.py outputs, SHA256SUMS, and
# ends job.log with "== done" — judge a job only after that line. EXTRA_ARGS (env) is appended to every
# run, e.g. EXTRA_ARGS="--limit 16 --epochs 1" for a smoke test.
set -u
PACK=$(realpath "$1"); OUT=$2; LIST=$3; GPU=$4; CORES=${5:-}
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"; OUT=$(realpath "$OUT"); LOG=$OUT/job.log
cp "$LIST" "$OUT/runlist.txt"

{
  echo "date      $(date -u +%FT%TZ)"
  echo "host      $(hostname)"
  echo "commit    $(git -C "$REPO" rev-parse HEAD)"
  echo "dirty     $(git -C "$REPO" status --porcelain | wc -l) changed files"
  echo "pack      $PACK"
  echo "gpu       $GPU  cores ${CORES:-unpinned}"
  python -c "import sys, torch; print('python   ', sys.version.split()[0], ' torch', torch.__version__, ' cuda', torch.version.cuda)"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
} > "$OUT/env.txt" 2>&1

echo "$(date -u +%FT%TZ) checking pack checksums" >> "$LOG"
(cd "$PACK" && sha256sum -c --quiet SHA256SUMS) >> "$LOG" 2>&1 || { echo "!!! pack checksum failed" >> "$LOG"; exit 1; }
echo "$(date -u +%FT%TZ) CPU self-test" >> "$LOG"
CUDA_VISIBLE_DEVICES="" python -m pytest -q "$REPO/tests" >> "$LOG" 2>&1 || { echo "!!! self-test failed" >> "$LOG"; exit 1; }

export GEOINDEX_DAILY_DATA=$PACK NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       CUDA_VISIBLE_DEVICES=$GPU
PIN=(); [ -n "$CORES" ] && PIN=(taskset -c "$CORES")
while read -r TAG SPLIT L CH; do
  [[ -z "$TAG" || "$TAG" == \#* ]] && continue
  echo "$(date -u +%FT%TZ) === $TAG split $SPLIT L=$L channels $CH" >> "$LOG"
  # $CH unquoted on purpose: one argument per channel
  ${PIN[@]+"${PIN[@]}"} python "$REPO/scripts/lead1_image.py" --config "$REPO/configs/ap_lead1.yaml" --arm B \
      --split "$SPLIT" --lengths "$L" --seeds 0 1 2 --channels $CH --tag "$TAG" --out-dir "$OUT" \
      --batch 8 --workers 6 ${EXTRA_ARGS:-} < /dev/null 2>&1 | grep -vi warning >> "$LOG"   # stdin stays with the run list
  [ "${PIPESTATUS[0]}" -eq 0 ] || echo "!!! $TAG split $SPLIT L=$L failed" >> "$LOG"
done < "$LIST"

(cd "$OUT" && find . -type f ! -name SHA256SUMS ! -name job.log | sort | xargs sha256sum > SHA256SUMS)
echo "== done $(date -u +%FT%TZ)" >> "$LOG"
