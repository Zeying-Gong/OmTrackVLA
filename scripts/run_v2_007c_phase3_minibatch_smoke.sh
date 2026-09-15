#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/nfs/share/wam_tracking/OmTrackVLA
PYTHON=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
OUTPUT="$ROOT/outputs/evaluation/v2_007c_phase3_minibatch_smoke"
SAMPLES="$OUTPUT/samples"
CONFIG="$ROOT/configs/phases/v2_006b_evt_polar_only.yaml"
CHECKPOINT="$ROOT/outputs/training/v2_006b_evt_polar_only/checkpoints/best.ckpt"
EXISTING_SAMPLE="$ROOT/outputs/evaluation/v2_007b_phase3_minibatch_smoke/samples/ep5_search_step21/sample.json"
STATUS="$OUTPUT/STATUS.json"

test ! -e "$OUTPUT"
test -s "$EXISTING_SAMPLE"
mkdir -p "$SAMPLES"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=3
export MAGNUM_CUDA_DEVICE=0
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

write_status() {
  "$PYTHON" -c 'import json,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"stage":sys.argv[2],"status":sys.argv[3],"completed_new_samples":int(sys.argv[4]),"reused_samples":1,"target_total_samples":3,"physical_gpu":3,"physical_gpu_zero_used":False,"split":"train","test_locked_used":False,"formal_training_started":False,"updated_unix":time.time()},indent=2,sort_keys=True)+"\n")' "$STATUS" "$1" "$2" "$3"
}

fail() {
  local code=$?
  write_status failed failed 0
  printf '%s\n' "$code" > "$OUTPUT/EXIT_CODE"
  exit "$code"
}
trap fail ERR

relabel() {
  local name="$1"
  local rollout="$2"
  local checkpoint_step="$3"
  local maximum_steps="$4"
  "$ROOT/scripts/run_egl.sh" "$PYTHON" -u \
    scripts/probe_next007_expert_relabel.py \
    --rollout-result "$rollout" \
    --output "$SAMPLES/$name.report.json" \
    --checkpoint-step "$checkpoint_step" \
    --expert-horizon-steps 21 \
    --waypoint-stride-steps 3 \
    --expert-target-lookahead-steps 6 \
    --maximum-safe-total-steps "$maximum_steps" \
    --history-size 8 \
    --history-stride-steps 3 \
    --phase3-sample-dir "$SAMPLES/$name" \
    > "$SAMPLES/$name.log" 2>&1
}

write_status relabel_ep5_far_visible running 0
relabel ep5_search_step40 \
  "$ROOT/outputs/evaluation/v2_006c_polar_safety_pair/polar_reactive/result.json" \
  40 70
write_status relabel_ep17_far_loss running 1
relabel ep17_lost_step61 \
  "$ROOT/outputs/evaluation/v2_006b_polar_closed_loop_pair/se2_waypoint/result.json" \
  61 90

write_status optimizer_smoke running 2
"$PYTHON" -u scripts/smoke_v2_phase3_minibatch.py \
  --sample "$EXISTING_SAMPLE" \
  --sample "$SAMPLES/ep5_search_step40/sample.json" \
  --sample "$SAMPLES/ep17_lost_step61/sample.json" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT/training_smoke" \
  --steps 16 \
  --device cuda:0 \
  > "$OUTPUT/training_smoke.log" 2>&1

write_status complete complete 2
printf '%s\n' 0 > "$OUTPUT/EXIT_CODE"
touch "$OUTPUT/COMPLETE"
