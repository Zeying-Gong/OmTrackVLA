#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/robot/robot-research-raw-data-0/user/gzy/temp/miniconda3/envs/tracevln_V2/bin/python}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
DATASET_INDICES="${DATASET_INDICES:?Set DATASET_INDICES to comma-separated dataset indices}"
TASK="${TASK:-stt}"
SPLIT="${SPLIT:-train}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/robot/robot-research-raw-data-0/user/gzy/omtrackvla_oracle_modular_dataset_eval}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/oracle_indices_$RUN_ID}"
DISPLAY_BASE="${DISPLAY_BASE:-300}"
SAVE_STEPS="${SAVE_STEPS:-1}"
RENDER_BACKEND="${RENDER_BACKEND:-}"
PERCEPTION="${PERCEPTION:-oracle}"
PERSON_DETECTOR_WEIGHTS="${PERSON_DETECTOR_WEIGHTS:-$ROOT/models/torchvision/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth}"
PERSON_REID_WEIGHTS="${PERSON_REID_WEIGHTS:-$ROOT/models/reid/osnet_x0_25_msmt17.pt}"
PERSON_SCORE_THRESHOLD="${PERSON_SCORE_THRESHOLD:-0.30}"
CONTROLLER="${CONTROLLER:-oracle-navmesh}"
MAP_MEMORY_FRAMES="${MAP_MEMORY_FRAMES:--1}"
MAP_CAMERA_HEIGHT="${MAP_CAMERA_HEIGHT:-0.24}"
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
    echo "[oracle-indices] ERROR: RENDER_BACKEND must be egl or xvfb, got: $RENDER_BACKEND" >&2
    exit 2
    ;;
esac

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
IFS=',' read -r -a INDICES <<< "$DATASET_INDICES"
export PYTHONPATH="$ROOT/habitat-lab:$ROOT"
mkdir -p "$LOG_ROOT"

for ((wave=0; wave<${#INDICES[@]}; wave+=${#GPUS[@]})); do
  pids=()
  labels=()
  for ((slot=0; slot<${#GPUS[@]}; slot++)); do
    item=$((wave + slot))
    if (( item >= ${#INDICES[@]} )); then
      break
    fi
    index="${INDICES[$item]}"
    gpu="${GPUS[$slot]}"
    display=$((DISPLAY_BASE + item))
    log="$LOG_ROOT/${TASK}_${SPLIT}_${index}.log"
    step_args=()
    if [[ "$SAVE_STEPS" == "1" ]]; then
      step_args+=(--save-steps)
    fi
    if [[ -n "${EVASION_SIDE:-}" ]]; then
      step_args+=(--evasion-side "$EVASION_SIDE")
    fi
echo "[oracle-indices] start index=$index gpu=$gpu display=$display"
    echo "[oracle-indices] controller_version=${ORACLE_CONTROLLER_VERSION:-5}"
    OMTRACKVLA_XVFB_DISPLAY_NUM="$display" \
      OMTRACKVLA_EGL_VENDOR_JSON="/tmp/omtrackvla_egl_indices_${RUN_ID}_${index}.json" \
      MAGNUM_CUDA_DEVICE=0 CUDA_VISIBLE_DEVICES="$gpu" \
      "$RENDER_RUNNER" "$PYTHON_BIN" -u oracle_modular_batch.py \
        --task "$TASK" --split "$SPLIT" \
        --shard-id "$slot" --num-shards "${#GPUS[@]}" --dataset-index "$index" \
        --output-root "$OUTPUT_ROOT" \
        --min-distance 1.2 --max-distance 1.5 \
        --max-forward 1.0 --max-lateral 1.0 --max-yaw 1.0 \
        --save-video --video-fps 8 --no-resume \
        --perception "$PERCEPTION" \
        --controller "$CONTROLLER" \
        --map-memory-frames "$MAP_MEMORY_FRAMES" \
        --map-camera-height "$MAP_CAMERA_HEIGHT" \
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
        "${step_args[@]}" >"$log" 2>&1 &
    pids+=("$!")
    labels+=("$index")
  done
  failed=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "[oracle-indices] FAILED index=${labels[$i]}" >&2
      failed=1
    fi
  done
  if (( failed )); then
    exit 1
  fi
done

"$PYTHON_BIN" summarize_oracle_modular.py "$OUTPUT_ROOT"
echo "[oracle-indices] complete logs=$LOG_ROOT"
