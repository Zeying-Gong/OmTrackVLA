#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/nfs/share/wam_tracking/OmTrackVLA
PYTHON=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
OUTPUT="$ROOT/outputs/evaluation/v2_007d_phase3_minibatch_smoke"
SAMPLE_A="$ROOT/outputs/evaluation/v2_007b_phase3_minibatch_smoke/samples/ep5_search_step21/sample.json"
SAMPLE_B="$ROOT/outputs/evaluation/v2_007c_phase3_minibatch_smoke/samples/ep5_search_step40/sample.json"
SAMPLE_C="$OUTPUT/samples/ep17_lost_step61/sample.json"
CONFIG="$ROOT/configs/phases/v2_006b_evt_polar_only.yaml"
CHECKPOINT="$ROOT/outputs/training/v2_006b_evt_polar_only/checkpoints/best.ckpt"
STATUS="$OUTPUT/STATUS.json"

test ! -e "$OUTPUT"
test -s "$SAMPLE_A"
test -s "$SAMPLE_B"
mkdir -p "$OUTPUT/samples"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=3
export MAGNUM_CUDA_DEVICE=0
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

write_status() {
  "$PYTHON" -c 'import json,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"stage":sys.argv[2],"status":sys.argv[3],"reused_samples":2,"new_samples":int(sys.argv[4]),"target_total_samples":3,"physical_gpu":3,"physical_gpu_zero_used":False,"split":"train","test_locked_used":False,"formal_training_started":False,"updated_unix":time.time()},indent=2,sort_keys=True)+"\n")' "$STATUS" "$1" "$2" "$3"
}

fail() {
  local code=$?
  write_status failed failed 0
  printf '%s\n' "$code" > "$OUTPUT/EXIT_CODE"
  exit "$code"
}
trap fail ERR

write_status relabel_ep17_far_loss running 0
"$ROOT/scripts/run_egl.sh" "$PYTHON" -u \
  scripts/probe_next007_expert_relabel.py \
  --rollout-result "$ROOT/outputs/evaluation/v2_006b_polar_closed_loop_pair/se2_waypoint/result.json" \
  --output "$OUTPUT/samples/ep17_lost_step61.report.json" \
  --checkpoint-step 61 \
  --expert-horizon-steps 21 \
  --waypoint-stride-steps 3 \
  --expert-target-lookahead-steps 6 \
  --maximum-safe-total-steps 90 \
  --history-size 8 \
  --history-stride-steps 3 \
  --phase3-sample-dir "$OUTPUT/samples/ep17_lost_step61" \
  > "$OUTPUT/samples/ep17_lost_step61.log" 2>&1

write_status optimizer_smoke running 1
"$PYTHON" -u scripts/smoke_v2_phase3_minibatch.py \
  --sample "$SAMPLE_A" \
  --sample "$SAMPLE_B" \
  --sample "$SAMPLE_C" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT/training_smoke" \
  --steps 16 \
  --device cuda:0 \
  > "$OUTPUT/training_smoke.log" 2>&1

write_status complete complete 1
printf '%s\n' 0 > "$OUTPUT/EXIT_CODE"
touch "$OUTPUT/COMPLETE"
