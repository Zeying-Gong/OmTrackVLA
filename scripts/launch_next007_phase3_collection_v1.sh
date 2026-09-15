#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/scripts/launch_next007_phase3_collection_v1.sh"

run_candidate() {
  local gpu="$1"
  local task="$2"
  local index="$3"
  local tag="$4"
  env \
    PHYSICAL_GPU="$gpu" \
    TASK="$task" \
    DATASET_INDEX="$index" \
    TAG="$tag" \
    CHECKPOINT_STEP=9 \
    bash "$ROOT/scripts/run_next007_phase3_candidate.sh" || true
}

if [[ "${1:-}" == "--worker" ]]; then
  gpu="$2"
  shift 2
  while (($#)); do
    if (($# < 3)); then
      echo "incomplete worker tuple" >&2
      exit 2
    fi
    run_candidate "$gpu" "$1" "$2" "$3"
    shift 3
  done
  exit 0
fi

launch_worker() {
  local session="$1"
  shift
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "session already exists: $session" >&2
    return 2
  fi
  local command
  printf -v command '%q ' "$SCRIPT" --worker "$@"
  tmux new-session -d -s "$session" "$command"
  echo "launched session=$session"
}

cd "$ROOT"
launch_worker next007_collect_v1_g1 1 \
  stt 500 formal_stt_0500_20260914 \
  stt 900 formal_stt_0900_20260914 \
  stt 1300 formal_stt_1300_20260914
launch_worker next007_collect_v1_g2 2 \
  stt 1700 formal_stt_1700_20260914 \
  stt 2100 formal_stt_2100_20260914 \
  stt 2500 formal_stt_2500_20260914
launch_worker next007_collect_v1_g3 3 \
  dt 500 formal_dt_0500_20260914 \
  dt 900 formal_dt_0900_20260914 \
  dt 1300 formal_dt_1300_20260914
launch_worker next007_collect_v1_g4 4 \
  dt 1700 formal_dt_1700_20260914 \
  dt 2100 formal_dt_2100_20260914 \
  dt 2500 formal_dt_2500_20260914
launch_worker next007_collect_v1_g5 5 \
  at 500 formal_at_0500_20260914 \
  at 900 formal_at_0900_20260914 \
  at 1300 formal_at_1300_20260914
launch_worker next007_collect_v1_g6 6 \
  at 1700 formal_at_1700_20260914 \
  at 2100 formal_at_2100_20260914 \
  at 2500 formal_at_2500_20260914
launch_worker next007_collect_v1_g7 7 \
  stt 3100 formal_stt_3100_20260914 \
  dt 3100 formal_dt_3100_20260914 \
  at 3100 formal_at_3100_20260914

tmux list-sessions | grep 'next007_collect_v1_' || true
