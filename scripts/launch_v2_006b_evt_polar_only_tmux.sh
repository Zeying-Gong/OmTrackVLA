#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/data/nfs/share/wam_tracking/OmTrackVLA
SESSION=v2_006b_evt_polar
OUTPUT="$ROOT/outputs/training/v2_006b_evt_polar_only"
mkdir -p "$OUTPUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 2
fi
tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && bash scripts/run_v2_006b_evt_polar_only.sh > '$OUTPUT/launcher.log' 2>&1"
tmux display-message -p -t "$SESSION" '#S #{session_created_string}'

