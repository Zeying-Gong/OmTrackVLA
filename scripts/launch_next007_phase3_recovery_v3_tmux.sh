#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="next007_phase3_recovery_policy_only_v3"
OUTPUT_DIR="$ROOT/outputs/training/next007_phase3_recovery_policy_only_v3"
RUN_LOG="$OUTPUT_DIR/tmux.log"
EXIT_FILE="$OUTPUT_DIR/exit_code.txt"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"
if [[ -f "$EXIT_FILE" ]]; then
  echo "refusing to overwrite completed v3 metadata: $EXIT_FILE" >&2
  exit 2
fi
command="cd '$ROOT' && set -o pipefail && bash scripts/run_next007_phase3_recovery_v3.sh > '$RUN_LOG' 2>&1; code=\$?; printf '%s\n' \"\$code\" > '$EXIT_FILE'; exit \"\$code\""
tmux new-session -d -s "$SESSION" "$command"
echo "launched session=$SESSION log=$RUN_LOG"
