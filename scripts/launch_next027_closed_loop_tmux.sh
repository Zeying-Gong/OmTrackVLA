#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${SESSION:-next027_closed_loop_smoke}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/evaluation/next027_closed_loop_smoke}"
RUN_LOG="$OUTPUT_ROOT/run.log"
EXIT_FILE="$OUTPUT_ROOT/exit_code.txt"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"
rm -f "$EXIT_FILE"

COMMAND="cd '$ROOT' && set -o pipefail && env PHYSICAL_GPU='${PHYSICAL_GPU:-7}' TASK='${TASK:-stt}' SPLIT='${SPLIT:-val}' DATASET_INDEX='${DATASET_INDEX:-0}' INITIALIZATION_SCAN_EPISODES='${INITIALIZATION_SCAN_EPISODES:-8}' INITIALIZATION_WAIT_STEPS='${INITIALIZATION_WAIT_STEPS:-32}' MAX_STEPS='${MAX_STEPS:-24}' UWB_MODE='${UWB_MODE:-missing}' OUTPUT_ROOT='$OUTPUT_ROOT' bash scripts/run_next027_closed_loop_smoke.sh > '$RUN_LOG' 2>&1; code=\$?; printf '%s\\n' \"\$code\" > '$EXIT_FILE'; exit \"\$code\""
tmux new-session -d -s "$SESSION" "$COMMAND"
echo "launched session=$SESSION log=$RUN_LOG"
