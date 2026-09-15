#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/nfs/share/wam_tracking/OmTrackVLA
RUNNER="$ROOT/scripts/run_v2_018_phase3_unseen1300_gate.sh"
SESSION=v2_018_unseen1300
LOG_DIR="$ROOT/outputs/logs"
LOG="$LOG_DIR/v2_018_phase3_unseen1300_gate.log"

cd "$ROOT"
test -s "$RUNNER"
test ! -e "$ROOT/outputs/evaluation/v2_018_phase3_unseen1300_closed_loop_gate"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
mkdir -p "$LOG_DIR"
test ! -e "$LOG"
tmux new-session -d -s "$SESSION" "cd '$ROOT' && bash '$RUNNER' > '$LOG' 2>&1"
tmux display-message -p -t "$SESSION" '#{session_name} #{session_created_string}'
printf '%s\n' "$LOG"
