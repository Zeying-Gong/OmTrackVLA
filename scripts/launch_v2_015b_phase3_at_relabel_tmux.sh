#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/nfs/share/wam_tracking/OmTrackVLA
SESSION=v2_015b_phase3_at_relabel
OUTPUT="$ROOT/outputs/evaluation/v2_015b_phase3_at_training_relabel"
LOG="$ROOT/outputs/evaluation/v2_015b_phase3_at_training_relabel.launch.log"
EXIT_FILE="$ROOT/outputs/evaluation/v2_015b_phase3_at_training_relabel.EXIT_CODE"

test ! -e "$OUTPUT"
test ! -e "$LOG"
test ! -e "$EXIT_FILE"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 2
fi

printf -v command \
  "cd %q && set -o pipefail && bash scripts/run_v2_015b_phase3_at_relabel.sh > %q 2>&1; code=\$?; printf '%%s\\n' \"\$code\" > %q; exit \"\$code\"" \
  "$ROOT" "$LOG" "$EXIT_FILE"
tmux new-session -d -s "$SESSION" "$command"
echo "launched session=$SESSION gpu=3 output=$OUTPUT"
