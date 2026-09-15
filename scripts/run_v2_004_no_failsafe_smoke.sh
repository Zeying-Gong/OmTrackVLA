#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/nfs/share/wam_tracking/OmTrackVLA
PYTHON=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
CONFIG="$ROOT/configs/phases/phase2_end_to_end_v2_se2_visibility_balanced.yaml"
CHECKPOINT="$ROOT/outputs/training/v2_004_se2_visibility_balanced/checkpoints/best.ckpt"
OUTPUT="$ROOT/outputs/evaluation/v2_004_se2_visibility_balanced_no_failsafe/stt/index2"

mkdir -p "$OUTPUT"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=4
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

timeout --signal=TERM --kill-after=20s 600s \
  "$PYTHON" -m omtrackvla.evaluation.end_to_end_closed_loop \
  --task stt \
  --split train \
  --dataset-index 2 \
  --max-steps 300 \
  --initialization-scan-episodes 8 \
  --device cuda:0 \
  --model-config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output-root "$OUTPUT" \
  --uwb-mode missing \
  --action-mode se2_waypoint \
  --disable-visual-failsafe \
  --video-fps 8 \
  > "$OUTPUT/rollout.log" 2>&1

test -s "$OUTPUT/result.json"
test -s "$OUTPUT/rollout.mp4"
touch "$OUTPUT/COMPLETE"

