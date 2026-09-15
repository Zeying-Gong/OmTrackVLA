#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
SESSION=v2_002_action_compare
LOG="$ROOT/outputs/training/v2_002_action_compare/launcher.log"
mkdir -p "$(dirname "$LOG")"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  exit 0
fi
tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && bash scripts/run_v2_002_action_compare.sh"
tmux new-window -d -t "$SESSION" -n monitor \
  "while tmux has-session -t '$SESSION:0' 2>/dev/null; do date '+%F %T'; test -f '$LOG' && tail -n 20 '$LOG'; sleep 600; done"
echo "started tmux session: $SESSION"
