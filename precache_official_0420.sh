#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/robot/robot-research-raw-data-0/user/gzy/temp/miniconda3/envs/tracevln_V2/bin/python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/official_0420}"
CACHE_ROOT="${CACHE_ROOT:-$DATA_ROOT/vision_cache}"
MANIFEST="${MANIFEST:-$DATA_ROOT/frame_manifest.txt}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/precache_official_0420}"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export DINOV3_MODEL_PATH="$ROOT/models/dinov3-vits16-pretrain-lvd1689m"
export SIGLIP_MODEL_PATH="$ROOT/models/siglip-so400m-patch14-384"

IFS=',' read -r -a gpus <<< "$GPU_LIST"
local_workers="${#gpus[@]}"
num_shards=$((NNODES * local_workers))
mkdir -p "$CACHE_ROOT" "$LOG_DIR"

if [[ ! -s "$MANIFEST" ]]; then
  echo "Missing frame manifest: $MANIFEST" >&2
  echo "Generate it once before launching cache workers." >&2
  exit 2
fi

pids=()
for local_rank in "${!gpus[@]}"; do
  shard_id=$((NODE_RANK * local_workers + local_rank))
  CUDA_VISIBLE_DEVICES="${gpus[$local_rank]}" "$PYTHON" "$ROOT/precache_frames.py" \
    --data_root "$DATA_ROOT" \
    --cache_root "$CACHE_ROOT" \
    --batch_size "$BATCH_SIZE" \
    --image_size 384 \
    --manifest "$MANIFEST" \
    --num_shards "$num_shards" \
    --shard_id "$shard_id" \
    >"$LOG_DIR/node${NODE_RANK}_shard${shard_id}.log" 2>&1 &
  pids+=("$!")
  echo "started cuda=${gpus[$local_rank]} shard=$shard_id/$num_shards pid=${pids[-1]}"
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
