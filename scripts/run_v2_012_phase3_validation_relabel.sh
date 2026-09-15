#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/nfs/share/wam_tracking/OmTrackVLA
PYTHON=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
OUTPUT_ROOT="$ROOT/outputs/evaluation/v2_012_phase3_validation_relabel"
GPU=3

relabel() {
  local task="$1"
  local source="$2"
  local anchor="$3"
  local category="$4"
  local tag="${task}_${category}_step${anchor}"
  local sample_dir="$OUTPUT_ROOT/samples/$tag"
  local report="$OUTPUT_ROOT/reports/$tag.json"
  local log="$OUTPUT_ROOT/logs/$tag.log"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" MAGNUM_CUDA_DEVICE=0 \
    "$ROOT/scripts/run_egl.sh" "$PYTHON" -u \
      scripts/probe_next007_expert_relabel.py \
      --rollout-result "$source" \
      --output "$report" \
      --checkpoint-step "$anchor" \
      --expert-horizon-steps 21 \
      --waypoint-stride-steps 3 \
      --expert-target-lookahead-steps 6 \
      --maximum-safe-total-steps 140 \
      --history-size 8 \
      --history-stride-steps 3 \
      --phase3-sample-dir "$sample_dir" \
      > "$log" 2>&1
  local code=$?
  set -e
  printf '%s\n' "$code" > "$OUTPUT_ROOT/logs/$tag.EXIT_CODE"
  if ((code == 0)) && test -s "$sample_dir/sample.json"; then
    touch "$OUTPUT_ROOT/logs/$tag.COMPLETE"
  else
    touch "$OUTPUT_ROOT/logs/$tag.FAILED"
  fi
}

test "$GPU" != 0
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT/samples" "$OUTPUT_ROOT/reports" "$OUTPUT_ROOT/logs"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

BASE="$ROOT/outputs/evaluation/v2_011_phase3_closed_loop_gate/parent/polar_reactive"
relabel stt "$BASE/stt_900/result.json" 40 long_lost
relabel stt "$BASE/stt_900/result.json" 66 reacquisition
relabel dt "$BASE/dt_900/result.json" 50 long_lost
relabel dt "$BASE/dt_900/result.json" 80 reacquisition
relabel at "$BASE/at_900/result.json" 40 long_lost
relabel at "$BASE/at_900/result.json" 70 reacquisition

touch "$OUTPUT_ROOT/COMPLETE"
