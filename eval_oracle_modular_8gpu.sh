#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

detect_gpu_list() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != "-1" ]]; then
    echo "$CUDA_VISIBLE_DEVICES"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local detected
    detected="$(
      nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null \
        | sed 's/[[:space:]]//g' \
        | sed '/^$/d' \
        | paste -sd, -
    )"
    if [[ -n "$detected" ]]; then
      echo "$detected"
      return
    fi
  fi
  echo "0"
}

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
GPU_LIST="${GPU_LIST:-$(detect_gpu_list)}"
NUM_WORKERS="${NUM_WORKERS:-1}"
TASKS="${TASKS:-stt,dt,at}"
SPLITS="${SPLITS:-train,val}"
IFS=',' read -r -a PHYSICAL_GPUS <<< "$GPU_LIST"
IFS=',' read -r -a TASK_ARRAY <<< "$TASKS"
IFS=',' read -r -a SPLIT_ARRAY <<< "$SPLITS"

if [[ ! "$NUM_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[oracle-8gpu] ERROR: NUM_WORKERS must be a positive integer, got: $NUM_WORKERS" >&2
  exit 2
fi

# Expand each physical GPU into NUM_WORKERS independent worker slots. Each
# worker receives a unique shard while CUDA_VISIBLE_DEVICES pins it to its GPU.
GPUS=()
for gpu_id in "${PHYSICAL_GPUS[@]}"; do
  for ((worker_id=0; worker_id<NUM_WORKERS; worker_id++)); do
    GPUS+=("$gpu_id")
  done
done
NUM_SHARDS="${NUM_SHARDS:-${#GPUS[@]}}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/robot/robot-research-raw-data-0/user/gzy/omtrackvla_oracle_modular_dataset_eval}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"
DISPLAY_BASE="${DISPLAY_BASE:-220}"
MAX_EPISODES_PER_SHARD="${MAX_EPISODES_PER_SHARD:-}"
EXCLUDE_DATASET_INDICES="${EXCLUDE_DATASET_INDICES:-}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
VIDEO_FPS="${VIDEO_FPS:-8}"
RENDER_BACKEND="${RENDER_BACKEND:-}"
REQUIRE_100_SUCCESS="${REQUIRE_100_SUCCESS:-0}"
MAX_SUCCESS_ATTEMPTS="${MAX_SUCCESS_ATTEMPTS:-5}"
GPU_BURN_DUTY="${GPU_BURN_DUTY:-0.30}"
GPU_BURN_PERIOD="${GPU_BURN_PERIOD:-10}"
GPU_BURN_SIZE="${GPU_BURN_SIZE:-4096}"
GPU_BURN_DTYPE="${GPU_BURN_DTYPE:-fp16}"
SCENES_PER_PROCESS="${SCENES_PER_PROCESS:-1}"
MAX_CONSECUTIVE_NATIVE_RESTARTS="${MAX_CONSECUTIVE_NATIVE_RESTARTS:-3}"
RESTART_DELAY="${RESTART_DELAY:-1}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-15}"
PERCEPTION="${PERCEPTION:-oracle}"
CONTROLLER="${CONTROLLER:-oracle-navmesh}"
TARGET_MODE="${TARGET_MODE:-auto}"
MAP_MEMORY_FRAMES="${MAP_MEMORY_FRAMES:-0}"
MAP_CAMERA_HEIGHT="${MAP_CAMERA_HEIGHT:-0.24}"
MAP_CAMERA_PITCH_DEG="${MAP_CAMERA_PITCH_DEG:-5.0}"
MAP_ROBOT_RADIUS="${MAP_ROBOT_RADIUS:-0.30}"
MAP_MIN_STATIC_HITS="${MAP_MIN_STATIC_HITS:-2}"
PERSON_DETECTOR_WEIGHTS="${PERSON_DETECTOR_WEIGHTS:-$ROOT/models/torchvision/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth}"
PERSON_REID_WEIGHTS="${PERSON_REID_WEIGHTS:-$ROOT/models/reid/osnet_x0_25_msmt17.pt}"
PERSON_SCORE_THRESHOLD="${PERSON_SCORE_THRESHOLD:-0.30}"
TARGET_INITIALIZATION="${TARGET_INITIALIZATION:-auto}"
LOST_TARGET_POLICY="${LOST_TARGET_POLICY:-auto}"
LOST_BRAKE_STEPS="${LOST_BRAKE_STEPS:-2}"
LOST_SEARCH_YAW="${LOST_SEARCH_YAW:-0.35}"
LOST_SEARCH_PERIOD_STEPS="${LOST_SEARCH_PERIOD_STEPS:-8}"
LOST_COAST_STEPS="${LOST_COAST_STEPS:-3}"
LOST_COAST_MIN_RANGE="${LOST_COAST_MIN_RANGE:-2.0}"
LOST_COAST_MAX_TRANSLATION="${LOST_COAST_MAX_TRANSLATION:-0.35}"

if [[ -z "$RENDER_BACKEND" ]]; then
  RENDER_BACKEND="egl"
  if [[ "$PERCEPTION" == "rgb-person" ]]; then
    RENDER_BACKEND="xvfb"
  fi
fi

case "$RENDER_BACKEND" in
  egl)
    RENDER_RUNNER="$ROOT/run_egl.sh"
    ;;
  xvfb)
    RENDER_RUNNER="$ROOT/run_xvfb.sh"
    ;;
  *)
    echo "[oracle-8gpu] ERROR: RENDER_BACKEND must be egl or xvfb, got: $RENDER_BACKEND" >&2
    exit 2
    ;;
esac

if [[ "${SKIP_GPU_PREFLIGHT:-0}" != "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t AVAILABLE_GPUS < <(
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits 2>/dev/null
  )
  if (( ${#AVAILABLE_GPUS[@]} > 0 )); then
    declare -A AVAILABLE_GPU_SET=()
    for gpu_record in "${AVAILABLE_GPUS[@]}"; do
      IFS=',' read -r gpu_index gpu_uuid <<< "$gpu_record"
      gpu_index="${gpu_index//[[:space:]]/}"
      gpu_uuid="${gpu_uuid//[[:space:]]/}"
      AVAILABLE_GPU_SET["$gpu_index"]=1
      AVAILABLE_GPU_SET["$gpu_uuid"]=1
    done
    for gpu_id in "${PHYSICAL_GPUS[@]}"; do
      if [[ -z "${AVAILABLE_GPU_SET[$gpu_id]:-}" ]]; then
        echo "[oracle-8gpu] ERROR: requested GPU $gpu_id is not visible; nvidia-smi records: ${AVAILABLE_GPUS[*]}" >&2
        echo "[oracle-8gpu] Allocate/mount the requested GPUs, or reduce GPU_LIST and NUM_SHARDS." >&2
        exit 2
      fi
    done
  fi
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
export PYTHONPATH="$ROOT/habitat-lab:$ROOT"

# Invalid or truncated results are incomplete. Remove them before workers build
# their resume sets so the owning shard automatically reruns those episodes.
"$PYTHON_BIN" summarize_oracle_modular.py "$OUTPUT_ROOT" --clean-invalid-only

BACKGROUND_PIDS=()
cleanup_background() {
  local pid
  for pid in "${BACKGROUND_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${BACKGROUND_PIDS[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup_background EXIT
trap 'exit 130' INT TERM HUP

if [[ "$GPU_BURN_DUTY" != "0" && "$GPU_BURN_DUTY" != "0.0" ]]; then
  for gpu in "${PHYSICAL_GPUS[@]}"; do
    echo "[oracle-8gpu] burn gpu=$gpu duty=$GPU_BURN_DUTY period=${GPU_BURN_PERIOD}s"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u gpu_burn_duty.py \
      --duty "$GPU_BURN_DUTY" \
      --period "$GPU_BURN_PERIOD" \
      --size "$GPU_BURN_SIZE" \
      --dtype "$GPU_BURN_DTYPE" \
      >"$LOG_ROOT/gpu_burn_${gpu}.log" 2>&1 &
    BACKGROUND_PIDS+=("$!")
  done
  sleep 2
  for i in "${!BACKGROUND_PIDS[@]}"; do
    if ! kill -0 "${BACKGROUND_PIDS[$i]}" 2>/dev/null; then
      echo "[oracle-8gpu] ERROR: gpu burn failed for gpu=${PHYSICAL_GPUS[$i]}; log=$LOG_ROOT/gpu_burn_${PHYSICAL_GPUS[$i]}.log" >&2
      exit 1
    fi
  done
fi

extra_args=()
if [[ -n "$MAX_EPISODES_PER_SHARD" ]]; then
  extra_args+=(--max-episodes "$MAX_EPISODES_PER_SHARD")
fi
if [[ -n "$EXCLUDE_DATASET_INDICES" ]]; then
  extra_args+=(--exclude-dataset-indices "$EXCLUDE_DATASET_INDICES")
fi
if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
  extra_args+=(--continue-on-error)
fi
if [[ "$SAVE_VIDEO" == "1" ]]; then
  extra_args+=(--save-video --video-fps "$VIDEO_FPS")
else
  extra_args+=(--no-save-video)
fi
if [[ "$REQUIRE_100_SUCCESS" == "1" ]]; then
  extra_args+=(--require-success --max-success-attempts "$MAX_SUCCESS_ATTEMPTS")
fi

echo "[oracle-8gpu] tasks=$TASKS splits=$SPLITS shards=$NUM_SHARDS gpus=$GPU_LIST workers_per_gpu=$NUM_WORKERS worker_slots=${#GPUS[@]} backend=$RENDER_BACKEND"
echo "[oracle-8gpu] controller_version=${ORACLE_CONTROLLER_VERSION:-5}"
echo "[oracle-8gpu] perception=$PERCEPTION"
echo "[oracle-8gpu] controller=$CONTROLLER"
echo "[oracle-8gpu] output=$OUTPUT_ROOT logs=$LOG_ROOT"
echo "[oracle-8gpu] progress_interval=${PROGRESS_INTERVAL}s resume=enabled"
if [[ -n "$EXCLUDE_DATASET_INDICES" ]]; then
  echo "[oracle-8gpu] excluded_dataset_indices=$EXCLUDE_DATASET_INDICES"
fi
echo "[oracle-8gpu] gpu_burn duty=$GPU_BURN_DUTY period=${GPU_BURN_PERIOD}s size=$GPU_BURN_SIZE dtype=$GPU_BURN_DTYPE"
if [[ "$REQUIRE_100_SUCCESS" == "1" ]]; then
  echo "[oracle-8gpu] episode_policy=require_success max_attempts=$MAX_SUCCESS_ATTEMPTS"
else
  echo "[oracle-8gpu] episode_policy=run_once (unsuccessful episodes are kept and do not stop the run)"
fi

run_shard() {
  local task="$1" split="$2" shard="$3" gpu="$4" display="$5"
  local rc attempt=0 native_failures=0
  while true; do
    attempt=$((attempt + 1))
    echo "[oracle-worker] attempt=$attempt task=$task split=$split shard=$shard gpu=$gpu"
    if OMTRACKVLA_XVFB_DISPLAY_NUM="$display" \
      OMTRACKVLA_EGL_VENDOR_JSON="/tmp/omtrackvla_egl_${RUN_ID}_${task}_${split}_${shard}.json" \
      MAGNUM_CUDA_DEVICE=0 \
      CUDA_VISIBLE_DEVICES="$gpu" \
      "$RENDER_RUNNER" "$PYTHON_BIN" -u oracle_modular_batch.py \
        --task "$task" --split "$split" \
        --shard-id "$shard" --num-shards "$NUM_SHARDS" \
        --output-root "$OUTPUT_ROOT" \
        --min-distance 1.2 --max-distance 1.5 \
        --max-forward 1.0 --max-lateral 1.0 --max-yaw 1.0 \
        --max-scenes-per-process "$SCENES_PER_PROCESS" \
        --perception "$PERCEPTION" \
        --controller "$CONTROLLER" \
        --target-mode "$TARGET_MODE" \
        --map-memory-frames "$MAP_MEMORY_FRAMES" \
        --map-camera-height "$MAP_CAMERA_HEIGHT" \
        --map-camera-pitch-deg "$MAP_CAMERA_PITCH_DEG" \
        --map-robot-radius "$MAP_ROBOT_RADIUS" \
        --map-min-static-hits "$MAP_MIN_STATIC_HITS" \
        --person-detector-weights "$PERSON_DETECTOR_WEIGHTS" \
        --person-reid-weights "$PERSON_REID_WEIGHTS" \
        --person-score-threshold "$PERSON_SCORE_THRESHOLD" \
        --target-initialization "$TARGET_INITIALIZATION" \
        --lost-target-policy "$LOST_TARGET_POLICY" \
        --lost-brake-steps "$LOST_BRAKE_STEPS" \
        --lost-search-yaw "$LOST_SEARCH_YAW" \
        --lost-search-period-steps "$LOST_SEARCH_PERIOD_STEPS" \
        --lost-coast-steps "$LOST_COAST_STEPS" \
        --lost-coast-min-range "$LOST_COAST_MIN_RANGE" \
        --lost-coast-max-translation "$LOST_COAST_MAX_TRANSLATION" \
        "${extra_args[@]}"; then
      return 0
    else
      rc=$?
    fi
    if [[ "$rc" -eq 75 ]]; then
      native_failures=0
      echo "[oracle-worker] scene batch complete; restarting for pending scenes"
      sleep "$RESTART_DELAY"
      continue
    fi
    if [[ "$rc" -eq 76 ]]; then
      echo "[oracle-worker] ERROR: success attempts exhausted" >&2
      return "$rc"
    fi
    native_failures=$((native_failures + 1))
    if (( native_failures > MAX_CONSECUTIVE_NATIVE_RESTARTS )); then
      echo "[oracle-worker] ERROR: rc=$rc after $native_failures consecutive native failures" >&2
      return "$rc"
    fi
    echo "[oracle-worker] native failure rc=$rc; completed results are durable, retrying" >&2
    sleep "$RESTART_DELAY"
  done
}

progress_args() {
  PROGRESS_ARGS=(
    --output-root "$OUTPUT_ROOT"
    --repo-root "$ROOT"
    --task "$1"
    --split "$2"
    --num-shards "$NUM_SHARDS"
    --interval "$PROGRESS_INTERVAL"
  )
  if [[ -n "$EXCLUDE_DATASET_INDICES" ]]; then
    PROGRESS_ARGS+=(--exclude-dataset-indices "$EXCLUDE_DATASET_INDICES")
  fi
  if [[ -n "$MAX_EPISODES_PER_SHARD" ]]; then
    PROGRESS_ARGS+=(--max-episodes-per-shard "$MAX_EPISODES_PER_SHARD")
  fi
  if [[ "$SAVE_VIDEO" == "1" ]]; then
    PROGRESS_ARGS+=(--save-video)
  fi
  if [[ "$REQUIRE_100_SUCCESS" == "1" ]]; then
    PROGRESS_ARGS+=(--require-success)
  fi
}

combo=0
for task in "${TASK_ARRAY[@]}"; do
  for split in "${SPLIT_ARRAY[@]}"; do
    progress_args "$task" "$split"
    progress_started_at="$(date +%s)"
    progress_baseline="$(
      "$PYTHON_BIN" monitor_oracle_progress.py "${PROGRESS_ARGS[@]}" --count-only
    )"
    "$PYTHON_BIN" -u monitor_oracle_progress.py \
      "${PROGRESS_ARGS[@]}" \
      --baseline "$progress_baseline" \
      --started-at "$progress_started_at" \
      --watch-pid "$$" &
    progress_pid=$!
    BACKGROUND_PIDS+=("$progress_pid")
    for ((wave=0; wave<NUM_SHARDS; wave+=${#GPUS[@]})); do
      pids=()
      labels=()
      for ((slot=0; slot<${#GPUS[@]}; slot++)); do
        shard=$((wave + slot))
        if (( shard >= NUM_SHARDS )); then
          break
        fi
        gpu="${GPUS[$slot]}"
        display=$((DISPLAY_BASE + combo * NUM_SHARDS + shard))
        log="$LOG_ROOT/${task}_${split}_shard${shard}.log"
        echo "[oracle-8gpu] start task=$task split=$split shard=$shard gpu=$gpu display=$display"
        run_shard "$task" "$split" "$shard" "$gpu" "$display" >"$log" 2>&1 &
        pids+=("$!")
        labels+=("$task/$split/shard$shard")
      done
      failed=0
      for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
          echo "[oracle-8gpu] FAILED ${labels[$i]} log=$LOG_ROOT" >&2
          failed=1
        fi
      done
      if (( failed )); then
        exit 1
      fi
    done
    kill "$progress_pid" 2>/dev/null || true
    wait "$progress_pid" 2>/dev/null || true
    "$PYTHON_BIN" monitor_oracle_progress.py \
      "${PROGRESS_ARGS[@]}" \
      --baseline "$progress_baseline" \
      --started-at "$progress_started_at" \
      --once
    combo=$((combo + 1))
    "$PYTHON_BIN" summarize_oracle_modular.py "$OUTPUT_ROOT"
  done
done

summary_args=()
if [[ "$REQUIRE_100_SUCCESS" == "1" ]]; then
  summary_args+=(--require-100-success)
fi
"$PYTHON_BIN" summarize_oracle_modular.py "$OUTPUT_ROOT" "${summary_args[@]}"
echo "[oracle-8gpu] complete output=$OUTPUT_ROOT summary=$OUTPUT_ROOT/summary.json"
