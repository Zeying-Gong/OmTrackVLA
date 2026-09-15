#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/nfs/share/wam_tracking/OmTrackVLA
PYTHON=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
OUTPUT="$ROOT/outputs/evaluation/v2_011_phase3_closed_loop_gate"
PARENT_CONFIG="$ROOT/configs/phases/v2_006b_evt_polar_only.yaml"
PARENT_CHECKPOINT="$ROOT/outputs/training/v2_006b_evt_polar_only/checkpoints/best.ckpt"
SELECTED_CONFIG="$ROOT/configs/phases/v2_010_phase3_model_visited_pilot.yaml"
SELECTED_CHECKPOINT="$ROOT/outputs/training/v2_010_phase3_model_visited_pilot/checkpoints/best.ckpt"
GPU=3

rollout() {
  local model="$1"
  local mode="$2"
  local task="$3"
  local config checkpoint
  if [[ "$model" == parent ]]; then
    config="$PARENT_CONFIG"
    checkpoint="$PARENT_CHECKPOINT"
  else
    config="$SELECTED_CONFIG"
    checkpoint="$SELECTED_CHECKPOINT"
  fi
  local destination="$OUTPUT/$model/$mode/${task}_900"
  test ! -e "$destination"
  mkdir -p "$destination"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" MAGNUM_CUDA_DEVICE=0 \
    timeout --signal=TERM --kill-after=15s 300s \
    "$PYTHON" -u -m omtrackvla.evaluation.end_to_end_closed_loop \
      --task "$task" \
      --split train \
      --dataset-index 900 \
      --max-steps 130 \
      --initialization-scan-episodes 8 \
      --device cuda:0 \
      --model-config "$config" \
      --checkpoint "$checkpoint" \
      --output-root "$destination" \
      --uwb-mode missing \
      --action-mode "$mode" \
      --visual-stop-threshold 0.005 \
      --video-fps 8 \
      > "$destination/rollout.log" 2>&1
  local code=$?
  set -e
  printf '%s\n' "$code" > "$destination/EXIT_CODE"
  if ((code != 0)) || ! test -s "$destination/result.json"; then
    touch "$destination/FAILED"
    return 1
  fi
  touch "$destination/COMPLETE"
}

test "$GPU" != 0
test ! -e "$OUTPUT"
mkdir -p "$OUTPUT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

for model in parent selected; do
  for mode in polar_reactive se2_waypoint; do
    for task in stt dt at; do
      rollout "$model" "$mode" "$task"
    done
  done
done

CUDA_VISIBLE_DEVICES= "$PYTHON" scripts/summarize_v2_phase3_closed_loop_gate.py \
  --root "$OUTPUT" \
  --output "$OUTPUT/SUMMARY.json" \
  --visibility-threshold 0.005
touch "$OUTPUT/COMPLETE"
