#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TASK="${TASK:-${1:-}}"
TASK="$(printf '%s' "$TASK" | tr '[:upper:]' '[:lower:]')"
case "$TASK" in
  stt|dt|at) ;;
  *) echo "Usage: TASK=stt|dt|at bash eval_repro_0420_xvfb.sh" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export GPU_LIST="${GPU_LIST:-$CUDA_VISIBLE_DEVICES}"
CKPT_STEP="${CKPT_STEP:-455000}"
case "$CKPT_STEP" in
  445000|450000|455000) ;;
  *) echo "Unsupported CKPT_STEP=$CKPT_STEP (expected 445000, 450000, or 455000)" >&2; exit 2 ;;
esac
export HF_MODEL_DIR="${HF_MODEL_DIR:-$ROOT/hf_exports/repro_0420_step${CKPT_STEP}}"
export HF_MODEL_ID="${HF_MODEL_ID:-}"
export OMTRACKVLA_LLM_NAME="${OMTRACKVLA_LLM_NAME:-$ROOT/models/Qwen3-0.6B}"
export DINOV3_MODEL_PATH="${DINOV3_MODEL_PATH:-$ROOT/models/dinov3-vits16-pretrain-lvd1689m}"
export SIGLIP_MODEL_PATH="${SIGLIP_MODEL_PATH:-$ROOT/models/siglip-so400m-patch14-384}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

export JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
export CHUNKS="${CHUNKS:-256}"
export RESUME="${RESUME:-0}"
export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export TRACKVLA_SAVE_VIDEO="${TRACKVLA_SAVE_VIDEO:-1}"
export GPU_BURN_DUTY="${GPU_BURN_DUTY:-0.4}"
export GPU_BURN_PERIOD="${GPU_BURN_PERIOD:-10}"
export GPU_BURN_SIZE="${GPU_BURN_SIZE:-4096}"
export GPU_BURN_DTYPE="${GPU_BURN_DTYPE:-fp16}"
export TRACKVLA_LIVE_FRAME_INTERVAL="${TRACKVLA_LIVE_FRAME_INTERVAL:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export SAVE_PATH="${SAVE_PATH:-sim_data/eval/repro_0420_step${CKPT_STEP}_${TASK}_${RUN_ID}}"
export LOG_DIR="${LOG_DIR:-logs/eval_repro_0420_step${CKPT_STEP}_${TASK}_${RUN_ID}}"

echo "[eval-repro] task=$TASK model=$HF_MODEL_DIR save=$SAVE_PATH log=$LOG_DIR"
exec bash eval_official_xvfb.sh
