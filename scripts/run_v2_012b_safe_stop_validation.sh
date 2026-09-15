#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/nfs/share/wam_tracking/OmTrackVLA
PYTHON=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
OUTPUT="$ROOT/outputs/evaluation/v2_012b_phase3_safe_stop_validation"
GPU=3

extract() {
  local task="$1"
  local anchor="$2"
  local source="$ROOT/outputs/evaluation/v2_011_phase3_closed_loop_gate/parent/polar_reactive/${task}_900/result.json"
  local tag="${task}_long_lost_step${anchor}"
  CUDA_VISIBLE_DEVICES="$GPU" MAGNUM_CUDA_DEVICE=0 \
    "$ROOT/scripts/run_egl.sh" "$PYTHON" -u \
      scripts/extract_v2_safe_stop_relabel.py \
      --rollout-result "$source" \
      --checkpoint-step "$anchor" \
      --output "$OUTPUT/reports/$tag.json" \
      --phase3-sample-dir "$OUTPUT/samples/$tag" \
      > "$OUTPUT/logs/$tag.log" 2>&1
  printf '0\n' > "$OUTPUT/logs/$tag.EXIT_CODE"
}

test "$GPU" != 0
test ! -e "$OUTPUT"
mkdir -p "$OUTPUT/reports" "$OUTPUT/samples" "$OUTPUT/logs"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

extract stt 50
extract dt 50
extract at 50
touch "$OUTPUT/COMPLETE"
