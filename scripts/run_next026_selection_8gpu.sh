#!/usr/bin/env bash
set -Eeuo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_BIN=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
RUN_DIR="${RUN_DIR:-$REPOSITORY_ROOT/outputs/training/next026_direct_phase2_core_v1}"
CONFIG="${CONFIG:-configs/phases/phase2_end_to_end_v1_formal.yaml}"
EVAL_DIR="$RUN_DIR/eval"

cd "$REPOSITORY_ROOT"
mkdir -p "$EVAL_DIR"
exec > >(tee -a "$EVAL_DIR/selection_launcher.log") 2>&1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

declare -a CHECKPOINTS=(
  "$RUN_DIR/checkpoints/step_0001000.ckpt"
  "$RUN_DIR/checkpoints/step_0002000.ckpt"
  "$RUN_DIR/checkpoints/step_0003000.ckpt"
  "$RUN_DIR/checkpoints/last.ckpt"
)
declare -a REPORTS=()
PORT="${MASTER_PORT_BASE:-29630}"
for CHECKPOINT in "${CHECKPOINTS[@]}"; do
  STEP="$("$PYTHON_BIN" -c 'import sys, torch; print(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["global_step"])' "$CHECKPOINT")"
  REPORT="$EVAL_DIR/candidate_step_${STEP}.json"
  REPORTS+=("$REPORT")
  "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=8 \
    --master_port="$PORT" \
    --module omtrackvla.evaluation.end_to_end_evaluate \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output "$REPORT" \
    --split val \
    --samples-per-mode 256 \
    --batch-size-per-device 4 \
    --num-workers 4
  PORT=$((PORT + 1))
done

"$PYTHON_BIN" scripts/select_end_to_end_v1_checkpoint.py \
  --run-dir "$RUN_DIR" \
  --reports "${REPORTS[@]}" \
  --output "$EVAL_DIR/checkpoint_selection.json"
