#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${SESSION:-next007_expert_relabel}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/evaluation/next007_expert_relabel_probe_v1}"
RUN_LOG="$OUTPUT_ROOT/run.log"
EXIT_FILE="$OUTPUT_ROOT/exit_code.txt"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"
if [[ -f "$EXIT_FILE" ]]; then
  echo "refusing to overwrite completed probe metadata: $EXIT_FILE" >&2
  exit 2
fi

COMMAND="cd '$ROOT' && set -o pipefail && env PHYSICAL_GPU='${PHYSICAL_GPU:-7}' ROLLOUT_RESULT='${ROLLOUT_RESULT:-$ROOT/outputs/evaluation/next027_full50_v6/result.json}' OUTPUT='$OUTPUT_ROOT/report.json' PHASE3_SAMPLE_DIR='$OUTPUT_ROOT/phase3_sample' CHECKPOINT_STEP='${CHECKPOINT_STEP:-28}' bash scripts/run_next007_expert_relabel_probe.sh > '$RUN_LOG' 2>&1; code=\$?; printf '%s\n' \"\$code\" > '$EXIT_FILE'; exit \"\$code\""
tmux new-session -d -s "$SESSION" "$COMMAND"
echo "launched session=$SESSION log=$RUN_LOG"
