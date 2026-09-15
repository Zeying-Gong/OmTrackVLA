#!/usr/bin/env bash
set -Eeuo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_BIN=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
OUTPUT_DIR="$REPOSITORY_ROOT/outputs/training/next026_direct_phase2_core_v1"

cd "$REPOSITORY_ROOT"
mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/launcher.log") 2>&1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  --master_port=29627 \
  --module omtrackvla.training.end_to_end_v1 \
  --config configs/phases/phase2_end_to_end_v1_formal.yaml \
  --output-dir "$OUTPUT_DIR"
