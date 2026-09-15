#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/nfs/share/wam_tracking/OmTrackVLA
PYTHON=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
CONFIG="$ROOT/configs/phases/phase2_end_to_end_v2_action_compare.yaml"
CHECKPOINT="$ROOT/outputs/training/v2_002_action_compare/checkpoints/best.ckpt"
OUTPUT="$ROOT/outputs/evaluation/v2_002_waypoint_direct_yaw_smoke/stt/index2"
STATUS="$OUTPUT/STATUS.json"

mkdir -p "$OUTPUT"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=4
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" -c 'import json,time; from pathlib import Path; p=Path("outputs/evaluation/v2_002_waypoint_direct_yaw_smoke/stt/index2/STATUS.json"); p.write_text(json.dumps({"status":"running","completed":0,"total":1,"physical_gpu":3,"physical_gpu_zero_used":False,"test_locked_used":False,"updated_unix":time.time()},indent=2)+"\n")'

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
  --action-mode waypoint_direct_yaw \
  --video-fps 8 \
  > "$OUTPUT/rollout.log" 2>&1

test -s "$OUTPUT/result.json"
test -s "$OUTPUT/rollout.mp4"
"$PYTHON" -c 'import json,time; from pathlib import Path; p=Path("outputs/evaluation/v2_002_waypoint_direct_yaw_smoke/stt/index2/STATUS.json"); p.write_text(json.dumps({"status":"complete","completed":1,"total":1,"physical_gpu":3,"physical_gpu_zero_used":False,"test_locked_used":False,"updated_unix":time.time()},indent=2)+"\n")'
touch "$OUTPUT/COMPLETE"

