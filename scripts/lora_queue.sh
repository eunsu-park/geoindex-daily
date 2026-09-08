#!/usr/bin/env bash
# Run the i3 LoRA experiment queue on the GPU server once the token cache has finished.
# Usage (on egghouse-gpu):  setsid nohup bash scripts/lora_queue.sh > ~/Data/GeoIndex/daily/lora/queue.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export GEOINDEX_DAILY_DATA=${GEOINDEX_DAILY_DATA:-$HOME/Data/GeoIndex/daily}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-$HOME/Softwares/miniconda3/envs/geoindex/bin/python}
CKPT=${CKPT:-$HOME/Data/GeoIndex/surya_probe/ckpt}
mkdir -p "$GEOINDEX_DAILY_DATA/lora"
COMMON="--ckpt $CKPT --batch 1 --accum 16 --workers 4 --freeze-through 1 --epochs 10 --patience 3"

echo "$(date -u +%FT%TZ) waiting for cache_surya_tokens to finish"
while pgrep -f cache_surya_tokens >/dev/null; do sleep 120; done
echo "$(date -u +%FT%TZ) cache done: $(ls $GEOINDEX_DAILY_DATA/surya/tokens/13ch/*.pt | wc -l) token files"

run() {  # run <mode> <ts-block> <seed> [extra]
  local tag="lora_$1_$2_r8_f1_s$3"
  if [ -f "$GEOINDEX_DAILY_DATA/lora/${tag}_summary.csv" ]; then echo "skip $tag (done)"; return; fi
  echo "$(date -u +%FT%TZ) start $tag"
  $PY scripts/lora_train.py --mode "$1" --ts-block "$2" --seed "$3" $COMMON ${4:-} > "$GEOINDEX_DAILY_DATA/lora/${tag}.log" 2>&1
  echo "$(date -u +%FT%TZ) end   $tag rc=$? :: $(grep -E '^means' "$GEOINDEX_DAILY_DATA/lora/${tag}.log" | tail -1)"
}

# main question first, then the matched controls, then the phase variants
run lora   ts    0
run ts     ts    0
run frozen ts    0
run lora   ts    1
run lora   ts    2
run ts     ts    1
run ts     ts    2
run frozen ts    1
run frozen ts    2
run lora   phase 0
run ts     phase 0
run frozen phase 0
echo "$(date -u +%FT%TZ) queue finished"
