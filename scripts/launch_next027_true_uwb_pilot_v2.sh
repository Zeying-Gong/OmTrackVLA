#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/training/next027_true_uwb_pilot_v2}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-$ROOT/outputs/ablations/next026_abl08_converged_v1/teacher_on/gru_curriculum_4k/checkpoints/best.ckpt}"
MASTER_PORT="${MASTER_PORT:-29729}"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/launcher.log") 2>&1
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc-per-node=7 \
  --master-port="$MASTER_PORT" \
  --module omtrackvla.training.end_to_end_v1 \
  --config configs/phases/next027_true_uwb_pilot_v2.yaml \
  --output-dir "$OUTPUT_DIR" \
  --init-checkpoint "$SOURCE_CHECKPOINT" \
  --max-steps 1024
