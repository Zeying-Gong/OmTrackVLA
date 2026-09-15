#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${SESSION:-next007_at400_after_cache_v2}"
PIPELINE_ROOT="${PIPELINE_ROOT:-$ROOT/outputs/evaluation/next007_train_at400_pipeline_v2}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 2
fi
mkdir -p "$PIPELINE_ROOT"
if [[ -f "$PIPELINE_ROOT/PIPELINE_COMPLETE.json" ]]; then
  echo "pipeline is already complete: $PIPELINE_ROOT" >&2
  exit 2
fi

COMMAND="cd '$ROOT' && env PHYSICAL_GPU='${PHYSICAL_GPU:-4}' PIPELINE_ROOT='$PIPELINE_ROOT' bash scripts/run_next007_at400_after_cache.sh"
tmux new-session -d -s "$SESSION" "$COMMAND"
echo "launched session=$SESSION log=$PIPELINE_ROOT/orchestrator.log"
