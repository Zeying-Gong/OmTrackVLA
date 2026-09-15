#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python}"
PHYSICAL_GPU="${PHYSICAL_GPU:-7}"
TASK="${TASK:-stt}"
SPLIT="${SPLIT:-val}"
DATASET_INDEX="${DATASET_INDEX:-0}"
MAX_STEPS="${MAX_STEPS:-24}"
INITIALIZATION_SCAN_EPISODES="${INITIALIZATION_SCAN_EPISODES:-8}"
INITIALIZATION_WAIT_STEPS="${INITIALIZATION_WAIT_STEPS:-32}"
UWB_MODE="${UWB_MODE:-missing}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/evaluation/next027_closed_loop_smoke}"
MODEL_CONFIG="${MODEL_CONFIG:-$ROOT/configs/phases/phase2_end_to_end_v1_phase1_init_formal.yaml}"
CHECKPOINT="${CHECKPOINT:-$ROOT/outputs/ablations/next026_abl08_converged_v1/teacher_on/gru_curriculum_4k/checkpoints/best.ckpt}"

mkdir -p "$OUTPUT_ROOT"
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export MAGNUM_CUDA_DEVICE=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

exec "$ROOT/scripts/run_egl.sh" "$PYTHON_BIN" -u -m \
  omtrackvla.evaluation.end_to_end_closed_loop \
  --task "$TASK" \
  --split "$SPLIT" \
  --dataset-index "$DATASET_INDEX" \
  --initialization-scan-episodes "$INITIALIZATION_SCAN_EPISODES" \
  --initialization-wait-steps "$INITIALIZATION_WAIT_STEPS" \
  --uwb-mode "$UWB_MODE" \
  --max-steps "$MAX_STEPS" \
  --device cuda:0 \
  --model-config "$MODEL_CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output-root "$OUTPUT_ROOT"
