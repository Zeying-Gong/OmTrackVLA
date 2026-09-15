#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python}"
PHYSICAL_GPU="${PHYSICAL_GPU:-7}"
ROLLOUT_RESULT="${ROLLOUT_RESULT:-$ROOT/outputs/evaluation/next027_full50_v6/result.json}"
OUTPUT="${OUTPUT:-$ROOT/outputs/evaluation/next007_expert_relabel_probe_v1/report.json}"
PHASE3_SAMPLE_DIR="${PHASE3_SAMPLE_DIR:-$(dirname "$OUTPUT")/phase3_sample}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-28}"

mkdir -p "$(dirname "$OUTPUT")"
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export MAGNUM_CUDA_DEVICE=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

exec "$ROOT/scripts/run_egl.sh" "$PYTHON_BIN" -u \
  scripts/probe_next007_expert_relabel.py \
  --rollout-result "$ROLLOUT_RESULT" \
  --output "$OUTPUT" \
  --checkpoint-step "$CHECKPOINT_STEP" \
  --expert-horizon-steps 21 \
  --waypoint-stride-steps 3 \
  --expert-target-lookahead-steps "${EXPERT_TARGET_LOOKAHEAD_STEPS:-0}" \
  --maximum-safe-total-steps 50 \
  --phase3-sample-dir "$PHASE3_SAMPLE_DIR"
