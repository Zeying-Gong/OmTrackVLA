#!/usr/bin/env bash
set -Eeuo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_BIN=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
SOURCE_CHECKPOINT="$REPOSITORY_ROOT/outputs/training/next026_phase2_world_action_long_v1/checkpoints/best.ckpt"
OUTPUT_DIR="${OUTPUT_DIR:-$REPOSITORY_ROOT/outputs/training/next026_phase2_waypoint_length_v1}"
MASTER_PORT="${MASTER_PORT:-29649}"

cd "$REPOSITORY_ROOT"
mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/launcher.log") 2>&1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

EXTRA_ARGS=()
if [[ -n "${MAX_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--max-steps "$MAX_STEPS")
fi

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  --master_port="$MASTER_PORT" \
  --module omtrackvla.training.end_to_end_v1 \
  --config configs/phases/phase2_end_to_end_v1_waypoint_length.yaml \
  --output-dir "$OUTPUT_DIR" \
  --init-checkpoint "$SOURCE_CHECKPOINT" \
  --training-heads-checkpoint "$SOURCE_CHECKPOINT" \
  "${EXTRA_ARGS[@]}"
