#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/robot/robot-research-raw-data-0/user/gzy/temp/miniconda3/envs/tracevln_V2/bin/python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/official_0420}"
OUT_DIR="${OUT_DIR:-/robot/robot-research-raw-data-0/user/gzy/omtrackvla_ckpts/repro_v2_grouped_cosine}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
BATCH_PER_GPU="${BATCH_PER_GPU:-1}"
EPOCHS="${EPOCHS:-2}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
LR="${LR:-2e-5}"
LLM_LR="${LLM_LR:-5e-6}"
WARMUP_RATIO="${WARMUP_RATIO:-0.02}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
MAX_CKPTS="${MAX_CKPTS:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_EVERY="${LOG_EVERY:-100}"
VIS_EVERY="${VIS_EVERY:-0}"
SEED="${SEED:-0}"
RESUME="${RESUME:-0}"
RESUME_CKPT="${RESUME_CKPT:-}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/train_repro_v2}"
RUN_NAME="${RUN_NAME:-$(basename "$OUT_DIR")}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log}"

if [ "$((GPUS_PER_NODE * BATCH_PER_GPU))" -ne 8 ]; then
  echo "V2 requires global batch 8: GPUS_PER_NODE * BATCH_PER_GPU must equal 8." >&2
  exit 2
fi
if [ ! -d "$DATA_ROOT/jsonl" ] || [ ! -d "$DATA_ROOT/vision_cache" ]; then
  echo "Missing JSONL or vision cache under $DATA_ROOT" >&2
  exit 2
fi
VISIBLE_GPU_COUNT="$(CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" "$PYTHON" -c 'import torch; print(torch.cuda.device_count())')"
if [ "$VISIBLE_GPU_COUNT" -lt "$GPUS_PER_NODE" ]; then
  echo "Requested $GPUS_PER_NODE GPUs, but evaluation Python sees only $VISIBLE_GPU_COUNT." >&2
  exit 2
fi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export DINOV3_MODEL_PATH="$ROOT/models/dinov3-vits16-pretrain-lvd1689m"
export SIGLIP_MODEL_PATH="$ROOT/models/siglip-so400m-patch14-384"

mkdir -p "$OUT_DIR" "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Training log: $LOG_FILE"
echo "V2 config: lr_other=$LR lr_llm=$LLM_LR scheduler=cosine warmup=$WARMUP_RATIO min_lr_ratio=$MIN_LR_RATIO"

train_args=(
  --train_json "$DATA_ROOT/jsonl"
  --cache_root "$DATA_ROOT/vision_cache"
  --out_dir "$OUT_DIR"
  --llm_name "$ROOT/models/Qwen3-0.6B"
  --epochs "$EPOCHS"
  --max_train_steps "$MAX_TRAIN_STEPS"
  --batch_size "$BATCH_PER_GPU"
  --n_waypoints 8
  --history 31
  --lr "$LR"
  --llm_lr "$LLM_LR"
  --lr_scheduler cosine
  --warmup_ratio "$WARMUP_RATIO"
  --min_lr_ratio "$MIN_LR_RATIO"
  --weight_decay 0.01
  --grad_clip 1.0
  --alpha_xy 2.0
  --beta_nav 10.0
  --mixed_precision
  --no_tanh_actions
  --num_workers "$NUM_WORKERS"
  --log_every "$LOG_EVERY"
  --vis_every "$VIS_EVERY"
  --csv_logging
  --save_every "$SAVE_EVERY"
  --max_ckpts "$MAX_CKPTS"
)
if [ "$RESUME" = "1" ]; then
  train_args+=(--resume)
fi
if [ -n "$RESUME_CKPT" ]; then
  train_args+=(--resume_ckpt "$RESUME_CKPT")
fi

exec "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$GPUS_PER_NODE" \
  "$ROOT/train.py" \
  --distributed \
  "${train_args[@]}"
