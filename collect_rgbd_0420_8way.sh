#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TASK="${TASK:?Set TASK to stt, dt, or at}"
case "$TASK" in stt|dt|at) ;; *) echo "Invalid TASK=$TASK" >&2; exit 2 ;; esac

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NUM_SHARDS="${NUM_SHARDS:-${#GPUS[@]}}"
if [[ "$NUM_SHARDS" -ne "${#GPUS[@]}" ]]; then
  echo "NUM_SHARDS=$NUM_SHARDS must equal GPU count ${#GPUS[@]}" >&2
  exit 2
fi

RUN_WRAPPER="${RUN_WRAPPER:-./run_xvfb.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/robot/robot-research-raw-data-0/user/gzy/omtrackvla_flux_rgbd_0420}"
SOURCE_ROOT="${SOURCE_ROOT:-sim_data/${TASK}_seed1000}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DISPLAY_BASE="${DISPLAY_BASE:-99}"
LOG_ROOT="${LOG_ROOT:-logs/replay_rgbd/full_${TASK}_${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-/robot/robot-research-raw-data-0/user/gzy/temp/miniconda3/envs/tracevln_V2/bin/python}"
GPU_BURN_DUTY="${GPU_BURN_DUTY:-0}"
GPU_BURN_PERIOD="${GPU_BURN_PERIOD:-10}"
GPU_BURN_SIZE="${GPU_BURN_SIZE:-4096}"
GPU_BURN_DTYPE="${GPU_BURN_DTYPE:-fp16}"
MAX_RESTARTS="${MAX_RESTARTS:-1000}"
RESTART_DELAY="${RESTART_DELAY:-2}"
MAX_NO_PROGRESS_RESTARTS="${MAX_NO_PROGRESS_RESTARTS:-3}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-60}"
mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT/$TASK"

BURN_PIDS=()
MONITOR_PID=""
cleanup_burn() {
  local pid
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  for pid in "${BURN_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${BURN_PIDS[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup_burn EXIT
trap 'exit 130' INT TERM

if [[ "$GPU_BURN_DUTY" != "0" && "$GPU_BURN_DUTY" != "0.0" ]]; then
  for ((slot=0; slot<NUM_SHARDS; slot++)); do
    gpu="${GPUS[$slot]}"
    echo "[collect-rgbd] burn gpu=$gpu duty=$GPU_BURN_DUTY period=${GPU_BURN_PERIOD}s"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u gpu_burn_duty.py \
      --duty "$GPU_BURN_DUTY" \
      --period "$GPU_BURN_PERIOD" \
      --size "$GPU_BURN_SIZE" \
      --dtype "$GPU_BURN_DTYPE" \
      >"$LOG_ROOT/gpu_burn_${slot}.log" 2>&1 &
    BURN_PIDS+=("$!")
  done
  sleep 2
  for ((slot=0; slot<NUM_SHARDS; slot++)); do
    if ! kill -0 "${BURN_PIDS[$slot]}" 2>/dev/null; then
      echo "[collect-rgbd] ERROR gpu burn slot=$slot failed: $LOG_ROOT/gpu_burn_${slot}.log" >&2
      exit 1
    fi
  done
fi

pids=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  gpu="${GPUS[$shard]}"
  display=$((DISPLAY_BASE + shard))
  worker_log="$LOG_ROOT/shard_${shard}.log"
  echo "[collect-rgbd] launch task=$TASK shard=$shard/$NUM_SHARDS gpu=$gpu display=:$display log=$worker_log"
  CUDA_VISIBLE_DEVICES="$gpu" \
  TASK="$TASK" \
  SHARD_ID="$shard" \
  NUM_SHARDS="$NUM_SHARDS" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  RUN_ID="$RUN_ID" \
  LOG_FILE="$LOG_ROOT/replay_${shard}.log" \
  OMTRACKVLA_GLX_DISPLAY_NUM="$display" \
  OMTRACKVLA_XVFB_DISPLAY_NUM="$display" \
  RUN_WRAPPER="$RUN_WRAPPER" \
  MAX_RESTARTS="$MAX_RESTARTS" \
  RESTART_DELAY="$RESTART_DELAY" \
  MAX_NO_PROGRESS_RESTARTS="$MAX_NO_PROGRESS_RESTARTS" \
  bash replay_rgbd_shard_retry.sh >"$worker_log" 2>&1 &
  pids+=("$!")
done

progress_loop() {
  local expected start_count start_time current elapsed produced rate eta pct
  expected="$(find -L "$SOURCE_ROOT" -name '*_info.json' -type f 2>/dev/null | wc -l)"
  start_count="$(find "$OUTPUT_ROOT/$TASK" -name complete.json -type f 2>/dev/null | wc -l)"
  start_time="$(date +%s)"
  while true; do
    current="$(find "$OUTPUT_ROOT/$TASK" -name complete.json -type f 2>/dev/null | wc -l)"
    elapsed=$(( $(date +%s) - start_time ))
    produced=$(( current - start_count ))
    pct="$(awk -v n="$current" -v d="$expected" 'BEGIN {printf(d ? "%.2f" : "0.00", 100*n/d)}')"
    rate="$(awk -v n="$produced" -v s="$elapsed" 'BEGIN {printf(s ? "%.2f" : "0.00", 3600*n/s)}')"
    eta="$(awk -v n="$current" -v d="$expected" -v r="$rate" 'BEGIN {printf(r > 0 ? "%.2f" : "inf", (d-n)/r)}')"
    echo "[collect-progress] task=$TASK episodes=$current/$expected (${pct}%) session_added=$produced rate=${rate}eps/h eta=${eta}h"
    sleep "$PROGRESS_INTERVAL"
  done
}
progress_loop &
MONITOR_PID="$!"

failed=0
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  if wait "${pids[$shard]}"; then
    echo "[collect-rgbd] done shard=$shard/$NUM_SHARDS"
  else
    rc=$?
    echo "[collect-rgbd] ERROR shard=$shard/$NUM_SHARDS rc=$rc log=$LOG_ROOT/shard_${shard}.log" >&2
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi
echo "[collect-rgbd] complete task=$TASK output=$OUTPUT_ROOT logs=$LOG_ROOT"
