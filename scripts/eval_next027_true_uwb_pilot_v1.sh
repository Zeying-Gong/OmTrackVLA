#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python}"
TRAIN_DIR="${TRAIN_DIR:-$ROOT/outputs/training/next027_true_uwb_pilot_v1_retry1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/evaluation/next027_true_uwb_pilot_v1}"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export OMP_NUM_THREADS=4

for step in 64 128 192 256 320 384 448 512; do
  label="$(printf 'step_%04d' "$step")"
  checkpoint="$TRAIN_DIR/checkpoints/$(printf 'step_%07d.ckpt' "$step")"
  directory="$OUTPUT_ROOT/$label"
  mkdir -p "$directory"
  if [[ -f "$directory/metrics.json" ]]; then
    continue
  fi
  "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc-per-node=7 \
    --master-port="$((29800 + step))" \
    -m omtrackvla.evaluation.end_to_end_evaluate \
    --config configs/phases/next027_true_uwb_pilot_v1.yaml \
    --checkpoint "$checkpoint" \
    --output "$directory/metrics.json" \
    --samples-per-mode 224 \
    --batch-size-per-device 4 \
    --num-workers 4 \
    > "$directory/eval.log" 2>&1
done
