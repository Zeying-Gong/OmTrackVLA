#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python}"
PHYSICAL_GPU="${PHYSICAL_GPU:-7}"
SAMPLE="${SAMPLE:-$ROOT/outputs/evaluation/next007_expert_relabel_probe_v4/phase3_sample/sample.json}"
CONFIG="${CONFIG:-$ROOT/configs/phases/phase2_end_to_end_v1_formal.yaml}"
CHECKPOINT="${CHECKPOINT:-$ROOT/outputs/ablations/next026_abl08_converged_v1/teacher_on/gru_curriculum_4k/checkpoints/best.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/evaluation/next007_phase3_single_sample_smoke_v1}"

mkdir -p "$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

exec "$PYTHON_BIN" -u scripts/smoke_next007_phase3_sample.py \
  --sample "$SAMPLE" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0
