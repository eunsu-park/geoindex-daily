#!/bin/bash
# Check a remote job's output folder after it is back on the NAS (runs here, CPU only).
#
#   bash scripts/remote_check.sh ~/Projects/GeoIndex/postbox/remote/<out>
#
# Passes when: job.log ends with "== done", no "!!!" line, SHA256SUMS verifies, and every
# run-list line has its three seeds' prediction files.
set -u
OUT=$(realpath "$1"); ok=1
tail -1 "$OUT/job.log" | grep -q "^== done" && echo "done line: yes" || { echo "done line: NO (job still running or stopped)"; ok=0; }
n=$(grep -c "^!!!" "$OUT/job.log"); echo "failure lines: $n"; [ "$n" -eq 0 ] || { grep "^!!!" "$OUT/job.log"; ok=0; }
(cd "$OUT" && sha256sum -c --quiet SHA256SUMS) && echo "checksums: ok" || { echo "checksums: FAILED"; ok=0; }
miss=0
while read -r TAG SPLIT L ARGS; do
  [[ -z "$TAG" || "$TAG" == \#* ]] && continue
  for s in 0 1 2; do
    f=$(ls "$OUT/${TAG}_preds_${SPLIT}_"*"_L${L}_s${s}.npz" 2>/dev/null | head -1)
    [ -n "$f" ] && [ -s "$f" ] || { echo "missing: ${TAG}_preds_${SPLIT}_*_L${L}_s${s}.npz"; miss=$((miss + 1)); }
  done
done < "$OUT/runlist.txt"
echo "missing prediction files: $miss"; [ "$miss" -eq 0 ] || ok=0
grep -E "^(commit|dirty|gpu|python)" "$OUT/env.txt"
[ "$ok" -eq 1 ] && echo "== PASS" || { echo "== FAIL"; exit 1; }
