#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/robot/robot-research-raw-data-0/user/gzy/temp/miniconda3/envs/tracevln_V2/bin/python}"
OUT_ROOT="${OUT_ROOT:-$ROOT/data/official_0420}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/prepare_official_0420}"
WORKERS_PER_TASK="${WORKERS_PER_TASK:-4}"

mkdir -p "$OUT_ROOT" "$LOG_DIR"

pids=()
for task in stt dt at; do
  for ((shard_id=0; shard_id<WORKERS_PER_TASK; shard_id++)); do
    "$PYTHON" "$ROOT/make_tracking_data.py" \
      --input_root "$ROOT/sim_data/${task}_seed1000" \
      --output_root "$OUT_ROOT" \
      --namespace "$task" \
      --history 31 \
      --horizon 8 \
      --dt 0.1 \
      --only_success \
      --num_shards "$WORKERS_PER_TASK" \
      --shard_id "$shard_id" \
      >"$LOG_DIR/${task}_shard${shard_id}.log" 2>&1 &
    pids+=("$!")
    echo "started $task shard=$shard_id/$WORKERS_PER_TASK pid=${pids[-1]}"
  done
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

if [[ "$status" -ne 0 ]]; then
  echo "At least one preprocessing job failed; inspect $LOG_DIR" >&2
  exit "$status"
fi

echo "Prepared frames under $OUT_ROOT/frames"
echo "Prepared JSONL under $OUT_ROOT/jsonl"
