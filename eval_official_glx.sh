#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-${1:-stt}}"
TASK="$(printf '%s' "$TASK" | tr '[:upper:]' '[:lower:]')"
case "$TASK" in
  stt|dt|at) ;;
  *) echo "Usage: TASK=stt|dt|at bash eval_official_glx.sh" >&2; exit 2 ;;
esac

PYTHON_BIN="${PYTHON_BIN:-/robot/robot-research-raw-data-0/user/gzy/temp/miniconda3/envs/tracevln_V2/bin/python}"
HF_MODEL_DIR="${HF_MODEL_DIR:-/robot/robot-research-raw-data-0/user/gzy/temp/tmp/omtrackvla_0_6b_hf}"
OMTRACKVLA_LLM_NAME="${OMTRACKVLA_LLM_NAME:-/robot/robot-research-raw-data-0/user/gzy/temp/tmp/omtrack_hf_home/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
HF_HOME="${HF_HOME:-/robot/robot-research-raw-data-0/user/gzy/temp/tmp/omtrack_hf_home}"
DINOV3_MODEL_PATH="${DINOV3_MODEL_PATH:-/robot/robot-research-raw-data-0/user/gzy/temp/tmp/modelscope_dinov3_vits16}"

CHUNKS="${CHUNKS:-32}"
GPU_LIST="${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
NUM_PARALLEL="${NUM_PARALLEL:-}"
HABITAT_GPU_ID="${HABITAT_GPU_ID:-0}"
RESUME="${RESUME:-1}"
SAVE_BASE="${SAVE_BASE:-sim_data/eval/official_0_6b_${TASK}}"
LOG_BASE="${LOG_BASE:-logs/eval_official_0_6b_${TASK}}"
SAVE_PATH_WAS_SET=0
LOG_DIR_WAS_SET=0
[ -n "${SAVE_PATH+x}" ] && SAVE_PATH_WAS_SET=1
[ -n "${LOG_DIR+x}" ] && LOG_DIR_WAS_SET=1
SCENE_DATASET="${SCENE_DATASET:-data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json}"
EXP_CONFIG="${EXP_CONFIG:-habitat-lab/habitat/config/benchmark/nav/track/track_infer_${TASK}.yaml}"
MAX_EPISODES="${MAX_EPISODES:-}"
MAX_STEPS="${MAX_STEPS:-}"
CONFIG_OVERRIDES="${CONFIG_OVERRIDES:-}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-1405}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-60}"
GPU_BURN_DUTY="${GPU_BURN_DUTY:-0}"
GPU_BURN_PERIOD="${GPU_BURN_PERIOD:-10}"
GPU_BURN_SIZE="${GPU_BURN_SIZE:-4096}"
GPU_BURN_DTYPE="${GPU_BURN_DTYPE:-fp16}"
RUN_WRAPPER="${RUN_WRAPPER:-./run_glx.sh}"
XVFB_DISPLAY_BASE="${XVFB_DISPLAY_BASE:-99}"
WORKER_CPU_THREADS="${WORKER_CPU_THREADS:-1}"

latest_run_id() {
  local candidate id mtime latest_id="" latest_mtime=-1
  for candidate in "${LOG_BASE}"_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]; do
    [ -d "$candidate" ] || continue
    id="${candidate#${LOG_BASE}_}"
    mtime="$(stat -c %Y "$candidate" 2>/dev/null || printf '0')"
    if [ "$mtime" -gt "$latest_mtime" ]; then
      latest_mtime="$mtime"
      latest_id="$id"
    fi
  done
  printf '%s\n' "$latest_id"
}

RUN_SELECTION="explicit/default paths"
if [ "$SAVE_PATH_WAS_SET" -eq 0 ] && [ "$LOG_DIR_WAS_SET" -eq 0 ]; then
  if [ "$RESUME" = "1" ]; then
    RUN_ID="$(latest_run_id)"
    if [ -n "$RUN_ID" ]; then
      SAVE_PATH="${SAVE_BASE}_${RUN_ID}"
      LOG_DIR="${LOG_BASE}_${RUN_ID}"
      RUN_SELECTION="resume latest timestamped run_id=${RUN_ID}"
    elif [ -d "$LOG_BASE" ] || [ -d "$SAVE_BASE" ]; then
      SAVE_PATH="$SAVE_BASE"
      LOG_DIR="$LOG_BASE"
      RUN_SELECTION="resume legacy paths"
    else
      RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
      SAVE_PATH="${SAVE_BASE}_${RUN_ID}"
      LOG_DIR="${LOG_BASE}_${RUN_ID}"
      RUN_SELECTION="new timestamped run_id=${RUN_ID} (no previous run found)"
    fi
  else
    RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
    SAVE_PATH="${SAVE_BASE}_${RUN_ID}"
    LOG_DIR="${LOG_BASE}_${RUN_ID}"
    RUN_SELECTION="new timestamped run_id=${RUN_ID}"
  fi
else
  SAVE_PATH="${SAVE_PATH:-$SAVE_BASE}"
  LOG_DIR="${LOG_DIR:-$LOG_BASE}"
fi

export HF_MODEL_DIR OMTRACKVLA_LLM_NAME HF_HOME DINOV3_MODEL_PATH
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export OMTRACKVLA_DISABLE_HF_SSL_VERIFY="${OMTRACKVLA_DISABLE_HF_SSL_VERIFY:-1}"
export TRACKVLA_SAVE_VIDEO="${TRACKVLA_SAVE_VIDEO:-0}"
export TRACKVLA_VERBOSE_STEPS="${TRACKVLA_VERBOSE_STEPS:-0}"
export SAVE_VIDEO="${SAVE_VIDEO:-0}"

mkdir -p "$SAVE_PATH" "$LOG_DIR"
LAUNCHER_LOG="$LOG_DIR/launcher.log"
log_msg() {
  printf '%s\n' "$*" | tee -a "$LAUNCHER_LOG"
}
log_msg "[eval:${TASK}] path_selection=${RUN_SELECTION} save=${SAVE_PATH} log=${LOG_DIR}"
format_duration() {
  local total="$1"
  printf '%02d:%02d:%02d' $((total / 3600)) $(((total % 3600) / 60)) $((total % 60))
}
count_done_chunks() {
  find "$LOG_DIR" -maxdepth 1 -type f \( -name "chunks_${CHUNKS}_chunk_*.done" -o -name "chunk_*.done" \) -printf '%f\n' 2>/dev/null \
    | sed -E "s/^chunks_${CHUNKS}_chunk_([0-9]+)\.done$/\1/; s/^chunk_([0-9]+)\.done$/\1/" \
    | sort -n -u \
    | wc -l
}
count_result_jsons() {
  find "$SAVE_PATH" -type f -name '*.json' ! -name '*_info.json' 2>/dev/null | wc -l
}
chunk_log_file_for_id() {
  local chunk="$1"
  local current_log="$LOG_DIR/chunks_${CHUNKS}_chunk_${chunk}.log"
  local legacy_log="$LOG_DIR/chunk_${chunk}.log"
  if [ -f "$current_log" ]; then
    printf '%s\n' "$current_log"
  elif [ -f "$legacy_log" ]; then
    printf '%s\n' "$legacy_log"
  fi
  return 0
}
count_finished_in_log() {
  local log_file="$1"
  grep -a -c 'finished episode id:' "$log_file" 2>/dev/null || true
}
chunk_total_from_log() {
  local log_file="$1"
  grep -aoE '[0-9]+/[0-9]+[[:space:]]+\[' "$log_file" 2>/dev/null \
    | tail -n 1 \
    | sed -E 's#^[0-9]+/([0-9]+).*#\1#' || true
}
count_eval_episodes_from_logs() {
  local chunk log_file finished total
  total=0
  for ((chunk = 0; chunk < CHUNKS; chunk++)); do
    log_file="$(chunk_log_file_for_id "$chunk")"
    [ -n "$log_file" ] || continue
    finished="$(count_finished_in_log "$log_file" | tr -d ' ')"
    total=$((total + finished))
  done
  printf '%s\n' "$total"
}
print_active_chunk_progress() {
  local chunk log_file finished total status done_file legacy_done_file fail_file
  local parts=()
  [ "${#ACTIVE_CHUNKS[@]}" -gt 0 ] || return 0
  for chunk in "${ACTIVE_CHUNKS[@]}"; do
    log_file="$(chunk_log_file_for_id "$chunk")"
    finished=0
    total="?"
    status="running"
    done_file="$LOG_DIR/chunks_${CHUNKS}_chunk_${chunk}.done"
    legacy_done_file="$LOG_DIR/chunk_${chunk}.done"
    fail_file="$LOG_DIR/chunks_${CHUNKS}_chunk_${chunk}.fail"
    if [ -f "$fail_file" ]; then
      status="failed"
    elif [ -f "$done_file" ] || [ -f "$legacy_done_file" ]; then
      status="done"
    fi
    if [ -n "$log_file" ]; then
      finished="$(count_finished_in_log "$log_file" | tr -d ' ')"
      total="$(chunk_total_from_log "$log_file" | tr -d ' ')"
    fi
    if [ -z "$total" ]; then
      if [ "$status" = "done" ]; then
        total="$finished"
      else
        total="?"
      fi
    fi
    parts+=("${chunk}:${finished}/${total}(${status})")
  done
  log_msg "[progress:${TASK}] active_chunks=${parts[*]}"
}
print_progress() {
  local now elapsed results eval_episodes progress_episodes progress_source done_chunks pct eta eta_text rate
  now="$(date +%s)"
  elapsed=$((now - START_TIME))
  results="$(count_result_jsons | tr -d ' ')"
  eval_episodes="$(count_eval_episodes_from_logs | tr -d ' ')"
  progress_episodes="$eval_episodes"
  progress_source="chunk_logs"
  if [ "$progress_episodes" -eq 0 ] && [ "$results" -gt 0 ]; then
    progress_episodes="$results"
    progress_source="result_jsons"
  fi
  done_chunks="$(count_done_chunks | tr -d ' ')"
  pct="$(awk -v n="$progress_episodes" -v d="$EXPECTED_EPISODES" 'BEGIN { if (d > 0) printf "%.1f", 100*n/d; else printf "0.0" }')"
  if [ "$progress_episodes" -gt 0 ] && [ "$progress_episodes" -lt "$EXPECTED_EPISODES" ]; then
    eta="$(awk -v n="$progress_episodes" -v d="$EXPECTED_EPISODES" -v e="$elapsed" 'BEGIN { printf "%d", (d-n)*e/n }')"
    eta_text="$(format_duration "$eta")"
  else
    eta_text="--:--:--"
  fi
  rate="$(awk -v n="$progress_episodes" -v e="$elapsed" 'BEGIN { if (e > 0) printf "%.3f", n/e; else printf "0.000" }')"
  log_msg "[progress:${TASK}] progress_episodes=${progress_episodes}/${EXPECTED_EPISODES} (${pct}%) source=${progress_source} eval_log_episodes=${eval_episodes} result_jsons=${results} chunks_done=${done_chunks}/${CHUNKS} elapsed=$(format_duration "$elapsed") eta=${eta_text} rate=${rate} eps/s"
  print_active_chunk_progress
}
IFS=',' read -ra GPUS <<< "$GPU_LIST"
SLOTS=()
for ((j = 0; j < JOBS_PER_GPU; j++)); do
  for gpu in "${GPUS[@]}"; do
    SLOTS+=("$gpu")
  done
done
if [ -z "$NUM_PARALLEL" ]; then
  NUM_PARALLEL="${#SLOTS[@]}"
fi
if [ "${#SLOTS[@]}" -lt "$NUM_PARALLEL" ]; then
  echo "GPU slots has ${#SLOTS[@]} entries but NUM_PARALLEL=${NUM_PARALLEL}: GPU_LIST=${GPU_LIST} JOBS_PER_GPU=${JOBS_PER_GPU}" >&2
  exit 2
fi
EXTRA_ARGS=()
CONFIG_OVERRIDE_ARGS=()
if [ -n "$MAX_EPISODES" ]; then
  EXTRA_ARGS+=(--max-episodes "$MAX_EPISODES")
  if [[ "$MAX_EPISODES" =~ ^[0-9]+$ ]]; then
    EXPECTED_EPISODES=$((MAX_EPISODES * CHUNKS))
  else
    EXPECTED_EPISODES="$MAX_EPISODES"
  fi
fi
if [ -n "$MAX_STEPS" ]; then
  CONFIG_OVERRIDE_ARGS+=(habitat.environment.max_episode_steps="$MAX_STEPS")
fi
if [ -n "$CONFIG_OVERRIDES" ]; then
  read -r -a _CONFIG_OVERRIDES_ARRAY <<< "$CONFIG_OVERRIDES"
  CONFIG_OVERRIDE_ARGS+=("${_CONFIG_OVERRIDES_ARRAY[@]}")
fi

BURN_PIDS=()
RUNNING_PIDS=()
RUNNING_CHUNKS=()
ACTIVE_CHUNKS=()
cleanup_burn() {
  if [ "${#BURN_PIDS[@]}" -gt 0 ]; then
    local pid
    for pid in "${BURN_PIDS[@]}"; do
      [ -n "${pid:-}" ] || continue
      kill_tree "$pid"
    done
    wait "${BURN_PIDS[@]}" >/dev/null 2>&1 || true
  fi
}
kill_tree() {
  local pid="$1" child
  [ -n "${pid:-}" ] || return 0
  kill -0 "$pid" >/dev/null 2>&1 || return 0
  if command -v pgrep >/dev/null 2>&1; then
    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
      kill_tree "$child"
    done
  fi
  kill "$pid" >/dev/null 2>&1 || true
}
cleanup_eval_jobs() {
  local pid
  for pid in "${RUNNING_PIDS[@]:-}"; do
    if [ -n "${pid:-}" ] && kill -0 "$pid" >/dev/null 2>&1; then
      kill_tree "$pid"
    fi
  done
}
cleanup_all() {
  cleanup_eval_jobs
  cleanup_burn
}
handle_signal() {
  local sig="$1" rc="$2"
  log_msg "[eval:${TASK}] received ${sig}; cleaning up child jobs"
  cleanup_all
  exit "$rc"
}
trap cleanup_all EXIT
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM
trap 'handle_signal HUP 129' HUP
if [ "$GPU_BURN_DUTY" != "0" ] && [ "$GPU_BURN_DUTY" != "0.0" ]; then
  for gpu in "${GPUS[@]}"; do
    log_msg "[burn:${TASK}] cuda=${gpu} duty=${GPU_BURN_DUTY} period=${GPU_BURN_PERIOD}s size=${GPU_BURN_SIZE} dtype=${GPU_BURN_DTYPE}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" gpu_burn_duty.py \
      --duty "$GPU_BURN_DUTY" \
      --period "$GPU_BURN_PERIOD" \
      --size "$GPU_BURN_SIZE" \
      --dtype "$GPU_BURN_DTYPE" \
      > "$LOG_DIR/gpu_burn_${gpu}.log" 2>&1 &
    BURN_PIDS+=("$!")
  done
fi

START_TIME="$(date +%s)"
log_msg "[eval:${TASK}] start chunks=${CHUNKS} num_parallel=${NUM_PARALLEL} jobs_per_gpu=${JOBS_PER_GPU} gpus=${GPU_LIST} save=${SAVE_PATH} log=${LOG_DIR} resume=${RESUME} scheduler=dynamic wrapper=${RUN_WRAPPER} gpu_burn_duty=${GPU_BURN_DUTY} gpu_burn_period=${GPU_BURN_PERIOD}s worker_cpu_threads=${WORKER_CPU_THREADS} max_episodes=${MAX_EPISODES:-none} max_steps=${MAX_STEPS:-none}"
print_progress

NEXT_CHUNK=0
FAILED=0

refresh_active_chunks() {
  ACTIVE_CHUNKS=()
  local slot
  for ((slot = 0; slot < NUM_PARALLEL; slot++)); do
    if [ -n "${RUNNING_PIDS[$slot]:-}" ]; then
      ACTIVE_CHUNKS+=("${RUNNING_CHUNKS[$slot]}")
    fi
  done
}

active_count() {
  local slot count=0
  for ((slot = 0; slot < NUM_PARALLEL; slot++)); do
    [ -n "${RUNNING_PIDS[$slot]:-}" ] && count=$((count + 1))
  done
  printf '%s\n' "$count"
}

launch_next_for_slot() {
  local slot="$1"
  local chunk done_file legacy_done_file fail_file chunk_log gpu_id

  while [ "$NEXT_CHUNK" -lt "$CHUNKS" ]; do
    chunk="$NEXT_CHUNK"
    NEXT_CHUNK=$((NEXT_CHUNK + 1))
    done_file="$LOG_DIR/chunks_${CHUNKS}_chunk_${chunk}.done"
    legacy_done_file="$LOG_DIR/chunk_${chunk}.done"
    fail_file="$LOG_DIR/chunks_${CHUNKS}_chunk_${chunk}.fail"
    chunk_log="$LOG_DIR/chunks_${CHUNKS}_chunk_${chunk}.log"

    if [ "$RESUME" = "1" ] && { [ -f "$done_file" ] || [ -f "$legacy_done_file" ]; }; then
      log_msg "[eval:${TASK}] skip completed chunk=${chunk}/${CHUNKS}"
      continue
    fi

    gpu_id="${SLOTS[$slot]}"
    log_msg "[eval:${TASK}] launch chunk=${chunk}/${CHUNKS} cuda=${gpu_id} slot=${slot} save=${SAVE_PATH}"
    rm -f "$done_file" "$fail_file"
    {
      if CUDA_VISIBLE_DEVICES="$gpu_id" \
        RANK="$slot" \
        OMTRACKVLA_XVFB_DISPLAY_NUM="$((XVFB_DISPLAY_BASE + slot))" \
        OMP_NUM_THREADS="$WORKER_CPU_THREADS" \
        MKL_NUM_THREADS="$WORKER_CPU_THREADS" \
        OPENBLAS_NUM_THREADS="$WORKER_CPU_THREADS" \
        NUMEXPR_NUM_THREADS="$WORKER_CPU_THREADS" \
        PYTHONPATH="habitat-lab:${PYTHONPATH:-}" \
        "$RUN_WRAPPER" "$PYTHON_BIN" run_eval.py \
          --split-num "$CHUNKS" \
          --split-id "$chunk" \
          --exp-config "$EXP_CONFIG" \
          --run-type eval \
          --save-path "$SAVE_PATH" \
          "${EXTRA_ARGS[@]}" \
          habitat.simulator.habitat_sim_v0.gpu_device_id="$HABITAT_GPU_ID" \
          habitat.simulator.scene_dataset="$SCENE_DATASET" \
          "${CONFIG_OVERRIDE_ARGS[@]}"; then
        touch "$done_file"
      else
        rc=$?
        echo "[eval:${TASK}] chunk=${chunk} failed with exit code ${rc}" >&2
        touch "$fail_file"
        exit "$rc"
      fi
    } > "$chunk_log" 2>&1 &
    RUNNING_PIDS[$slot]="$!"
    RUNNING_CHUNKS[$slot]="$chunk"
    return 0
  done

  return 1
}

launch_available_slots() {
  local slot
  for ((slot = 0; slot < NUM_PARALLEL; slot++)); do
    [ -n "${RUNNING_PIDS[$slot]:-}" ] && continue
    launch_next_for_slot "$slot" || true
  done
  refresh_active_chunks
}

reap_finished_slots() {
  local slot pid chunk done_file fail_file rc
  for ((slot = 0; slot < NUM_PARALLEL; slot++)); do
    pid="${RUNNING_PIDS[$slot]:-}"
    [ -n "$pid" ] || continue
    chunk="${RUNNING_CHUNKS[$slot]}"
    done_file="$LOG_DIR/chunks_${CHUNKS}_chunk_${chunk}.done"
    fail_file="$LOG_DIR/chunks_${CHUNKS}_chunk_${chunk}.fail"

    if [ -f "$done_file" ]; then
      if wait "$pid"; then
        log_msg "[eval:${TASK}] done chunk=${chunk}/${CHUNKS} slot=${slot}"
      else
        rc=$?
        log_msg "[eval:${TASK}] ERROR chunk=${chunk}/${CHUNKS} exited after done marker rc=${rc}"
        FAILED=1
      fi
      RUNNING_PIDS[$slot]=""
      RUNNING_CHUNKS[$slot]=""
    elif [ -f "$fail_file" ]; then
      if wait "$pid"; then
        rc=0
      else
        rc=$?
      fi
      log_msg "[eval:${TASK}] ERROR chunk=${chunk}/${CHUNKS} failed rc=${rc} log=$(chunk_log_file_for_id "$chunk")"
      RUNNING_PIDS[$slot]=""
      RUNNING_CHUNKS[$slot]=""
      FAILED=1
    fi
  done
  refresh_active_chunks
}

launch_available_slots
print_progress
while [ "$(active_count)" -gt 0 ] || [ "$NEXT_CHUNK" -lt "$CHUNKS" ]; do
  sleep "$PROGRESS_INTERVAL"
  reap_finished_slots
  if [ "$FAILED" -ne 0 ]; then
    print_progress
    exit "$FAILED"
  fi
  launch_available_slots
  print_progress
done

ACTIVE_CHUNKS=()
print_progress

"$PYTHON_BIN" summarize_eval.py "$SAVE_PATH" | tee "$LOG_DIR/summary.txt"
