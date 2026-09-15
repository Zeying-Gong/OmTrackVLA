#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/nfs/share/wam_tracking/OmTrackVLA
SESSION=v2_014b_phase3_at_collection
OUTPUT="$ROOT/outputs/evaluation/v2_014b_phase3_at_training_collection"
LOG="$ROOT/outputs/evaluation/v2_014b_phase3_at_training_collection.launch.log"
EXIT_FILE="$ROOT/outputs/evaluation/v2_014b_phase3_at_training_collection.EXIT_CODE"

test ! -e "$OUTPUT"
test ! -e "$LOG"
test ! -e "$EXIT_FILE"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 2
fi

printf -v command \
  "cd %q && set -o pipefail && bash scripts/run_v2_014b_phase3_at_collection_sequential.sh > %q 2>&1; code=\$?; printf '%%s\\n' \"\$code\" > %q; exit \"\$code\"" \
  "$ROOT" "$LOG" "$EXIT_FILE"
tmux new-session -d -s "$SESSION" "$command"
echo "launched session=$SESSION gpu=3 output=$OUTPUT"
