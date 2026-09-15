#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/nfs/share/wam_tracking/OmTrackVLA
PYTHON=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
SELECTION="$ROOT/outputs/evaluation/v2_014_phase3_multiscene_collection/ANCHOR_SELECTION.json"
OUTPUT="$ROOT/outputs/evaluation/v2_015_phase3_multiscene_relabel"
GPU=3

test "$GPU" != 0
test -s "$SELECTION"
test ! -e "$OUTPUT"
mkdir -p "$OUTPUT/reports" "$OUTPUT/samples" "$OUTPUT/logs"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

while IFS=$'\t' read -r task dataset_index anchor kind source; do
  tag="${task}_${dataset_index}_${kind}_step${anchor}"
  report="$OUTPUT/reports/$tag.json"
  sample_dir="$OUTPUT/samples/$tag"
  log="$OUTPUT/logs/$tag.log"
  set +e
  if [[ "$kind" == safe_stop ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" MAGNUM_CUDA_DEVICE=0 \
      "$ROOT/scripts/run_egl.sh" "$PYTHON" -u \
        scripts/extract_v2_safe_stop_relabel.py \
        --rollout-result "$source" \
        --checkpoint-step "$anchor" \
        --output "$report" \
        --phase3-sample-dir "$sample_dir" \
        --history-size 8 \
        --history-stride-steps 3 \
        > "$log" 2>&1
  else
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
  fi
  code=$?
  set -e
  printf '%s\n' "$code" > "$OUTPUT/logs/$tag.EXIT_CODE"
  if ((code == 0)) && test -s "$sample_dir/sample.json"; then
    touch "$OUTPUT/logs/$tag.COMPLETE"
  else
    touch "$OUTPUT/logs/$tag.FAILED"
  fi
done < <(
  "$PYTHON" -c \
    'import json,sys; value=json.load(open(sys.argv[1])); [print(item["task"], item["dataset_index"], item["anchor_environment_step"], item["candidate_kind"], item["rollout_result"], sep="\t") for item in value["selected"]]' \
    "$SELECTION"
)

"$PYTHON" scripts/summarize_v2_phase3_relabel_attempts.py \
  --selection "$SELECTION" \
  --root "$OUTPUT" \
  --output "$OUTPUT/SUMMARY.json"
touch "$OUTPUT/COMPLETE"
