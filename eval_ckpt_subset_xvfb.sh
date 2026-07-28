#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TASK="${TASK:-${1:-}}"
TASK="$(printf '%s' "$TASK" | tr '[:upper:]' '[:lower:]')"
MODEL="${MODEL:-${2:-}}"
case "$TASK" in
  stt|dt|at) ;;
  *) echo "Usage: TASK=stt|dt|at MODEL=official|stepNNNNNN|v2_stepNNNNNN bash eval_ckpt_subset_xvfb.sh" >&2; exit 2 ;;
esac

case "$MODEL" in
  official) HF_MODEL_DIR="$ROOT/official_ckpt/ckpt_0401_text_hf" ;;
  step445000|step450000|step455000) HF_MODEL_DIR="$ROOT/hf_exports/repro_0420_${MODEL}" ;;
  v2_step425000|v2_step440000|v2_step450000|v2_step459720|v2_step480000)
    HF_MODEL_DIR="$ROOT/hf_exports/repro_${MODEL}"
    ;;
  *) echo "Unknown MODEL=$MODEL" >&2; exit 2 ;;
esac
if [ ! -f "$HF_MODEL_DIR/config.json" ] || [ ! -f "$HF_MODEL_DIR/pytorch_model.bin" ]; then
  echo "Incomplete HuggingFace checkpoint: $HF_MODEL_DIR" >&2
  exit 2
fi

# Eight deterministic dataset splits, with the first N episodes from each split.
# All checkpoints therefore evaluate exactly the same 8 * N episode IDs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export GPU_LIST="${GPU_LIST:-$CUDA_VISIBLE_DEVICES}"
export JOBS_PER_GPU=1
export CHUNKS=8
export MAX_EPISODES="${SUBSET_PER_CHUNK:-16}"
export EXPECTED_EPISODES="$((CHUNKS * MAX_EPISODES))"
export RESUME="${RESUME:-0}"

export HF_MODEL_DIR
export HF_MODEL_ID=""
export SAVE_VIDEO="${SAVE_VIDEO:-0}"
export TRACKVLA_SAVE_VIDEO="${TRACKVLA_SAVE_VIDEO:-0}"
export TRACKVLA_LIVE_FRAME_INTERVAL="${TRACKVLA_LIVE_FRAME_INTERVAL:-0}"
export GPU_BURN_DUTY="${GPU_BURN_DUTY:-0.4}"
export GPU_BURN_PERIOD="${GPU_BURN_PERIOD:-10}"
export GPU_BURN_SIZE="${GPU_BURN_SIZE:-4096}"
export GPU_BURN_DTYPE="${GPU_BURN_DTYPE:-fp16}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
TAG="subset${EXPECTED_EPISODES}_${TASK}_${MODEL}_${RUN_ID}"
export SAVE_PATH="${SAVE_PATH:-sim_data/eval/$TAG}"
export LOG_DIR="${LOG_DIR:-logs/eval_$TAG}"

echo "[eval-subset] task=$TASK model=$MODEL episodes=$EXPECTED_EPISODES checkpoint=$HF_MODEL_DIR"
exec bash eval_repro_0420_xvfb.sh
