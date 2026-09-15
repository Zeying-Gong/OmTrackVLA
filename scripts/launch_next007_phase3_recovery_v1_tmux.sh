#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${SESSION:-next007_phase3_recovery_pilot_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/training/next007_phase3_recovery_pilot_v1}"
RUN_LOG="${RUN_LOG:-$OUTPUT_DIR/tmux.log}"
EXIT_FILE="${EXIT_FILE:-$OUTPUT_DIR/exit_code.txt}"

# Check before tmux launch or any output/log write. The runner repeats this
# guard under its exclusive lock to protect direct invocations as well.
if [[ -f "$OUTPUT_DIR/TRAINING_COMPLETE.json" ]]; then
  echo "existing Phase 3 completion marker; preserving completed run: $OUTPUT_DIR"
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"
if [[ -f "$EXIT_FILE" ]]; then
  echo "refusing to overwrite completed Phase 3 metadata: $EXIT_FILE" >&2
  exit 2
fi

command="cd '$ROOT' && set -o pipefail && bash scripts/run_next007_phase3_recovery_v1.sh > '$RUN_LOG' 2>&1; code=\$?; printf '%s\n' \"\$code\" > '$EXIT_FILE'; exit \"\$code\""
tmux new-session -d -s "$SESSION" "$command"
echo "launched session=$SESSION log=$RUN_LOG"
