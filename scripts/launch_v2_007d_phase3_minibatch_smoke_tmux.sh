#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/nfs/share/wam_tracking/OmTrackVLA
SESSION=v2_007d_phase3_smoke
LOG="$ROOT/outputs/evaluation/v2_007d_phase3_minibatch_smoke_launcher.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && bash scripts/run_v2_007d_phase3_minibatch_smoke.sh > '$LOG' 2>&1"
tmux has-session -t "$SESSION"
echo "$SESSION"
