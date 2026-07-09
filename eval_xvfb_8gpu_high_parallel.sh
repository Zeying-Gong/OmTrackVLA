#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-${1:-stt}}"
TASK="$(printf '%s' "$TASK" | tr '[:upper:]' '[:lower:]')"
case "$TASK" in
  stt|dt|at) ;;
  *) echo "Usage: TASK=stt|dt|at bash eval_xvfb_8gpu_high_parallel.sh" >&2; exit 2 ;;
esac

export GPU_LIST="${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_LIST}"
export JOBS_PER_GPU="${JOBS_PER_GPU:-16}"
export CHUNKS="${CHUNKS:-256}"
export RESUME="${RESUME:-0}"
export SAVE_VIDEO="${SAVE_VIDEO:-0}"
export TRACKVLA_SAVE_VIDEO="${TRACKVLA_SAVE_VIDEO:-0}"
export GPU_BURN_DUTY="${GPU_BURN_DUTY:-0}"
export GPU_BURN_PERIOD="${GPU_BURN_PERIOD:-10}"
export GPU_BURN_SIZE="${GPU_BURN_SIZE:-4096}"
export GPU_BURN_DTYPE="${GPU_BURN_DTYPE:-fp16}"
export WORKER_CPU_THREADS="${WORKER_CPU_THREADS:-1}"
export XVFB_DISPLAY_BASE="${XVFB_DISPLAY_BASE:-99}"

IFS=',' read -ra _OMTRACK_GPUS <<< "$GPU_LIST"
GPU_COUNT="${#_OMTRACK_GPUS[@]}"
RUN_TAG="${GPU_COUNT}gpu_jpg${JOBS_PER_GPU}_chunks${CHUNKS}"
if [ "$GPU_BURN_DUTY" != "0" ] && [ "$GPU_BURN_DUTY" != "0.0" ]; then
  BURN_TAG="$(printf '%s' "$GPU_BURN_DUTY" | tr '.' 'p')"
  RUN_TAG="${RUN_TAG}_burn${BURN_TAG}"
fi

export SAVE_BASE="${SAVE_BASE:-sim_data/eval/latest_0_6b_${TASK}_xvfb_${RUN_TAG}}"
export LOG_BASE="${LOG_BASE:-logs/eval_latest_0_6b_${TASK}_xvfb_${RUN_TAG}}"

cat <<EOF
================================================
OmTrackVLA XVFB high-parallel evaluation
  TASK=${TASK}
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
  GPU_LIST=${GPU_LIST}
  JOBS_PER_GPU=${JOBS_PER_GPU}
  CHUNKS=${CHUNKS}
  GPU_BURN_DUTY=${GPU_BURN_DUTY}
  SAVE_BASE=${SAVE_BASE}
  LOG_BASE=${LOG_BASE}
================================================
EOF

exec bash eval_official_xvfb.sh
