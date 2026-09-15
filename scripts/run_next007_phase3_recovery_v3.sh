#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/training/next007_phase3_recovery_policy_only_v3}"
CONFIG="$ROOT/configs/phases/next007_phase3_recovery_v3.json"
BASE_CONFIG="$ROOT/configs/phases/phase2_end_to_end_v1_phase1_init_formal.yaml"
BASELINE_CHECKPOINT="$ROOT/outputs/ablations/next026_abl08_converged_v1/teacher_on/gru_curriculum_4k/checkpoints/best.ckpt"
MANIFEST="$ROOT/outputs/training/next007_phase3_recovery_pilot_v1/recovery_manifest_frozen.json"
MASTER_PORT="${MASTER_PORT:-29727}"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/orchestrator.log") 2>&1
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nproc_per_node=7 --master_port="$MASTER_PORT" \
  scripts/train_next007_phase3.py \
  --config "$CONFIG" --output-dir "$OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR/eval_candidate_224"
"$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nproc_per_node=7 --master_port="$((MASTER_PORT + 1))" \
  --module omtrackvla.evaluation.end_to_end_evaluate \
  --config "$BASE_CONFIG" \
  --checkpoint "$OUTPUT_DIR/checkpoints/best.ckpt" \
  --output "$OUTPUT_DIR/eval_candidate_224/metrics.json" \
  --split val --samples-per-mode 224 --batch-size-per-device 4 --num-workers 3

CUDA_VISIBLE_DEVICES=7 "$PYTHON_BIN" scripts/render_next007_phase3_recovery.py \
  --config "$BASE_CONFIG" --manifest "$MANIFEST" \
  --before-checkpoint "$BASELINE_CHECKPOINT" \
  --after-checkpoint "$OUTPUT_DIR/checkpoints/best.ckpt" \
  --output-dir "$OUTPUT_DIR/recovery_visualization" \
  --samples-per-task 2 --device cuda:0

echo "[$(date --iso-8601=seconds)] Phase 3 policy-only v3 train/eval/render complete"
