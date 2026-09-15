#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/nfs/share/wam_tracking/OmTrackVLA
SESSION=v2_011_phase3_gate
OUTPUT="$ROOT/outputs/evaluation/v2_011_phase3_closed_loop_gate"
test ! -e "$OUTPUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 2
fi
tmux new-session -d -s "$SESSION" "bash '$ROOT/scripts/run_v2_011_phase3_closed_loop_gate.sh'"
echo "launched session=$SESSION gpu=3 output=$OUTPUT"
