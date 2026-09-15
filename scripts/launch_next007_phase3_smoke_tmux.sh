#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${SESSION:-next007_phase3_smoke_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/evaluation/next007_phase3_single_sample_smoke_v1}"
RUN_LOG="$OUTPUT_DIR/run.log"
EXIT_FILE="$OUTPUT_DIR/exit_code.txt"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"
if [[ -f "$EXIT_FILE" ]]; then
  echo "refusing to overwrite completed smoke metadata: $EXIT_FILE" >&2
  exit 2
fi

COMMAND="cd '$ROOT' && set -o pipefail && env PHYSICAL_GPU='${PHYSICAL_GPU:-7}' OUTPUT_DIR='$OUTPUT_DIR' bash scripts/run_next007_phase3_smoke.sh > '$RUN_LOG' 2>&1; code=\$?; printf '%s\n' \"\$code\" > '$EXIT_FILE'; exit \"\$code\""
tmux new-session -d -s "$SESSION" "$COMMAND"
echo "launched session=$SESSION log=$RUN_LOG"
