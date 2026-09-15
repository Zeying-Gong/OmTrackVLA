#!/usr/bin/env bash
set -Eeuo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_BIN=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
RUN_DIR="${RUN_DIR:-$REPOSITORY_ROOT/outputs/training/next026_direct_phase2_core_v1}"
CONFIG="${CONFIG:-configs/phases/phase2_end_to_end_v1_formal.yaml}"
MASTER_PORT="${MASTER_PORT:-29628}"

cd "$REPOSITORY_ROOT"
mkdir -p "$RUN_DIR/eval"
exec > >(tee -a "$RUN_DIR/eval/launcher.log") 2>&1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  --master_port="$MASTER_PORT" \
  --module omtrackvla.evaluation.end_to_end_evaluate \
  --config "$CONFIG" \
  --checkpoint "$RUN_DIR/checkpoints/best.ckpt" \
  --output "$RUN_DIR/eval/metrics.json" \
  --split val \
  --samples-per-mode 1024 \
  --batch-size-per-device 4 \
  --num-workers 4
