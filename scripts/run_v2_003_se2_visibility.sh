#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/nfs/share/wam_tracking/OmTrackVLA
PYTHON=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
CONFIG="$ROOT/configs/phases/phase2_end_to_end_v2_se2_visibility.yaml"
INITIAL="$ROOT/outputs/training/v2_002_action_compare/checkpoints/best.ckpt"
PREFLIGHT="$ROOT/outputs/training/v2_003_se2_visibility_preflight"
OUTPUT="$ROOT/outputs/training/v2_003_se2_visibility"
EVAL="$ROOT/outputs/evaluation/v2_003_se2_visibility/stt/index2"
STATUS="$OUTPUT/STATUS.json"

mkdir -p "$OUTPUT" "$EVAL"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

write_status() {
  local stage="$1"
  local state="$2"
  local completed="${3:-0}"
  "$PYTHON" -c 'import json,sys,time; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(json.dumps({"stage":sys.argv[2],"status":sys.argv[3],"completed_optimizer_steps":int(sys.argv[4]),"target_optimizer_steps":8192,"physical_gpus":[1,2,3,4,5,6,7],"physical_gpu_zero_used":False,"test_locked_used":False,"updated_unix":time.time()},indent=2,sort_keys=True)+"\n")' "$STATUS" "$stage" "$state" "$completed"
}

on_error() {
  local code=$?
  write_status failed failed 0
  exit "$code"
}
trap on_error ERR

write_status preflight running 0
"$PYTHON" -m py_compile \
  omtrackvla/models/end_to_end_v2.py \
  omtrackvla/data/end_to_end_training.py \
  omtrackvla/training/end_to_end_v2.py \
  omtrackvla/evaluation/end_to_end_closed_loop.py
"$PYTHON" tests/test_end_to_end_v2.py
"$PYTHON" tests/test_end_to_end_data.py

if [[ ! -f "$PREFLIGHT/TRAINING_COMPLETE.json" ]]; then
  "$PYTHON" -m torch.distributed.run \
    --standalone --nproc_per_node=7 --master_port=29652 \
    --module omtrackvla.training.end_to_end_v2 \
    --config "$CONFIG" \
    --output-dir "$PREFLIGHT" \
    --initial-checkpoint "$INITIAL" \
    --max-steps 2
fi

write_status training running 0
if [[ ! -f "$OUTPUT/TRAINING_COMPLETE.json" ]]; then
  "$PYTHON" -m torch.distributed.run \
    --standalone --nproc_per_node=7 --master_port=29653 \
    --module omtrackvla.training.end_to_end_v2 \
    --config "$CONFIG" \
    --output-dir "$OUTPUT" \
    --initial-checkpoint "$INITIAL"
fi

write_status closed_loop_smoke running 8192
export CUDA_VISIBLE_DEVICES=3
if [[ ! -f "$EVAL/result.json" ]]; then
  timeout --signal=TERM --kill-after=20s 600s \
    "$PYTHON" -m omtrackvla.evaluation.end_to_end_closed_loop \
    --task stt \
    --split train \
    --dataset-index 2 \
    --max-steps 300 \
    --initialization-scan-episodes 8 \
    --device cuda:0 \
    --model-config "$CONFIG" \
    --checkpoint "$OUTPUT/checkpoints/best.ckpt" \
    --output-root "$EVAL" \
    --uwb-mode missing \
    --action-mode se2_waypoint \
    --video-fps 8 \
    > "$EVAL/rollout.log" 2>&1
fi
test -s "$EVAL/result.json"
test -s "$EVAL/rollout.mp4"

write_status complete complete 8192
touch "$OUTPUT/COMPLETE"

