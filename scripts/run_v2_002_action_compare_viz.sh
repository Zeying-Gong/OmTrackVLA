#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/nfs/share/wam_tracking/OmTrackVLA
PYTHON=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
CONFIG="$ROOT/configs/phases/phase2_end_to_end_v2_action_compare.yaml"
CHECKPOINT="$ROOT/outputs/training/v2_002_action_compare/checkpoints/best.ckpt"
OUTPUT="$ROOT/outputs/evaluation/v2_002_action_compare_viz/stt/index2"
STATUS="$OUTPUT/STATUS.json"

mkdir -p "$OUTPUT"
cd "$ROOT"
export OMP_NUM_THREADS=4
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

write_status() {
  local stage="$1"
  local state="$2"
  local mode="${3:-}"
  local completed="${4:-0}"
  "$PYTHON" -c 'import json,sys,time; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(json.dumps({"stage":sys.argv[2],"status":sys.argv[3],"current_mode":sys.argv[4],"completed_modes":int(sys.argv[5]),"total_modes":3,"updated_unix":time.time(),"physical_gpu":3,"physical_gpu_zero_used":False,"test_locked_used":False},indent=2,sort_keys=True)+"\n")' "$STATUS" "$stage" "$state" "$mode" "$completed"
}

on_error() {
  local code=$?
  write_status failed failed "${CURRENT_MODE:-unknown}" "${COMPLETED:-0}"
  exit "$code"
}
trap on_error ERR

COMPLETED=0
CURRENT_MODE=preflight
write_status preflight running "$CURRENT_MODE" "$COMPLETED"
test -f "$CONFIG"
test -f "$CHECKPOINT"

for mode in direct waypoint_dt lookahead; do
  CURRENT_MODE="$mode"
  destination="$OUTPUT/$mode"
  mkdir -p "$destination"
  write_status rollout running "$mode" "$COMPLETED"
  if [[ ! -f "$destination/result.json" ]]; then
    CUDA_VISIBLE_DEVICES=3 timeout --signal=TERM --kill-after=20s 600s \
      "$PYTHON" -m omtrackvla.evaluation.end_to_end_closed_loop \
      --task stt \
      --split train \
      --dataset-index 2 \
      --max-steps 300 \
      --initialization-scan-episodes 8 \
      --device cuda:0 \
      --model-config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --output-root "$destination" \
      --uwb-mode missing \
      --action-mode "$mode" \
      --video-fps 8 \
      > "$destination/rollout.log" 2>&1
  fi
  test -s "$destination/result.json"
  test -s "$destination/rollout.mp4"
  COMPLETED=$((COMPLETED + 1))
  write_status rollout running "$mode" "$COMPLETED"
done

CURRENT_MODE=diagnostics
write_status diagnostics running "$CURRENT_MODE" "$COMPLETED"
"$PYTHON" scripts/summarize_v2_action_compare_viz.py \
  --root "$OUTPUT" \
  --output "$OUTPUT/diagnostics.json"

write_status complete complete none "$COMPLETED"
touch "$OUTPUT/COMPLETE"

