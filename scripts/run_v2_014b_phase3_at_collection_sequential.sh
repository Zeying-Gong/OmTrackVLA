#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/nfs/share/wam_tracking/OmTrackVLA
PYTHON=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
CONFIG="$ROOT/configs/phases/v2_006b_evt_polar_only.yaml"
CHECKPOINT="$ROOT/outputs/training/v2_006b_evt_polar_only/checkpoints/best.ckpt"
OUTPUT_ROOT="$ROOT/outputs/evaluation/v2_014b_phase3_at_training_collection"
GPU=3

run_rollout() {
  local dataset_index="$1"
  local output="$OUTPUT_ROOT/at_${dataset_index}"
  test ! -e "$output"
  mkdir -p "$output"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" MAGNUM_CUDA_DEVICE=0 \
    timeout --signal=TERM --kill-after=15s 300s \
    "$PYTHON" -u -m omtrackvla.evaluation.end_to_end_closed_loop \
      --task at \
      --split train \
      --dataset-index "$dataset_index" \
      --max-steps 130 \
      --initialization-scan-episodes 8 \
      --device cuda:0 \
      --model-config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --output-root "$output" \
      --uwb-mode missing \
      --action-mode polar_reactive \
      --visual-stop-threshold 0.005 \
      --no-video \
      > "$output/rollout.log" 2>&1
  local code=$?
  set -e
  printf '%s\n' "$code" > "$output/EXIT_CODE"
  if ((code == 0)) && test -s "$output/result.json"; then
    touch "$output/COMPLETE"
  else
    touch "$output/FAILED"
  fi
}

test "$GPU" != 0
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

run_rollout 100
run_rollout 1100

"$PYTHON" scripts/summarize_v2_phase3_collection.py \
  --root "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/SUMMARY.json"
"$PYTHON" scripts/select_v2_phase3_anchor_candidates.py \
  --rollout "$OUTPUT_ROOT/at_100/result.json" \
  --rollout "$OUTPUT_ROOT/at_1100/result.json" \
  --output "$OUTPUT_ROOT/ANCHOR_SELECTION.json"
touch "$OUTPUT_ROOT/COMPLETE"
