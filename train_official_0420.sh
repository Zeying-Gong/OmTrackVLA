#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/robot/robot-research-raw-data-0/user/gzy/temp/miniconda3/envs/tracevln_V2/bin/python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/official_0420}"
OUT_DIR="${OUT_DIR:-$ROOT/ckpts/repro_0420}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
BATCH_PER_GPU="${BATCH_PER_GPU:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
RESUME="${RESUME:-0}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_EVERY="${LOG_EVERY:-10}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/train_official_0420}"
RUN_NAME="${RUN_NAME:-$(basename "$OUT_DIR")}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${RUN_NAME}_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Training log: $LOG_FILE"

global_batch=$((NNODES * GPUS_PER_NODE * BATCH_PER_GPU))
if [[ "$global_batch" -ne 8 ]]; then
  echo "Official checkpoint used global batch 8; requested global batch is $global_batch." >&2
  echo "Set NNODES * GPUS_PER_NODE * BATCH_PER_GPU = 8 for strict reproduction." >&2
  exit 2
fi
if [[ ! -d "$DATA_ROOT/jsonl" || ! -d "$DATA_ROOT/vision_cache" ]]; then
  echo "Missing processed JSONL or vision cache under $DATA_ROOT" >&2
  exit 2
fi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export DINOV3_MODEL_PATH="$ROOT/models/dinov3-vits16-pretrain-lvd1689m"
export SIGLIP_MODEL_PATH="$ROOT/models/siglip-so400m-patch14-384"

mkdir -p "$OUT_DIR"

train_args=(
  --train_json "$DATA_ROOT/jsonl" \
  --cache_root "$DATA_ROOT/vision_cache" \
  --out_dir "$OUT_DIR" \
  --llm_name "$ROOT/models/Qwen3-0.6B" \
  --epochs 2 \
  --max_train_steps "$MAX_TRAIN_STEPS" \
  --batch_size "$BATCH_PER_GPU" \
  --n_waypoints 8 \
  --history 31 \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --grad_clip 1.0 \
  --alpha_xy 2.0 \
  --beta_nav 10.0 \
  --mixed_precision \
  --no_tanh_actions \
  --num_workers "$NUM_WORKERS" \
  --log_every "$LOG_EVERY" \
  --max_ckpts 3
  --save_every "$SAVE_EVERY"
)
if [[ "$RESUME" == "1" ]]; then
  train_args+=(--resume)
fi

if [[ "$NNODES" -eq 1 && "$GPUS_PER_NODE" -eq 1 ]]; then
  exec "$PYTHON" "$ROOT/train.py" "${train_args[@]}"
fi

dist_args=(
  --nproc_per_node "$GPUS_PER_NODE"
  --nnodes "$NNODES"
)
if [[ "$NNODES" -eq 1 ]]; then
  dist_args+=(--standalone)
else
  dist_args+=(
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
  )
fi

exec "$PYTHON" -m torch.distributed.run "${dist_args[@]}" "$ROOT/train.py" \
  --distributed \
  "${train_args[@]}"
