#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/nfs/share/wam_tracking/OmTrackVLA
PYTHON=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
OUTPUT="$ROOT/outputs/evaluation/v2_018_phase3_unseen1300_closed_loop_gate"
GPU=3

declare -A CONFIGS=(
  [v2_006b_parent]="$ROOT/configs/phases/v2_006b_evt_polar_only.yaml"
  [v2_016_waypoint]="$ROOT/configs/phases/v2_016_phase3_multiscene_observable_pilot.yaml"
  [v2_017_calibrated]="$ROOT/configs/phases/v2_017_phase3_safety_head_calibration.yaml"
)
declare -A CHECKPOINTS=(
  [v2_006b_parent]="$ROOT/outputs/training/v2_006b_evt_polar_only/checkpoints/best.ckpt"
  [v2_016_waypoint]="$ROOT/outputs/training/v2_016_phase3_multiscene_observable_pilot/checkpoints/step_0000128.ckpt"
  [v2_017_calibrated]="$ROOT/outputs/training/v2_017_phase3_safety_head_calibration/checkpoints/best.ckpt"
)

rollout() {
  local model="$1"
  local mode="$2"
  local task="$3"
  local destination="$OUTPUT/$model/$mode/${task}_1300"
  test ! -e "$destination"
  mkdir -p "$destination"
  printf '{"event":"start","model":"%s","mode":"%s","task":"%s","time":"%s"}\n' \
    "$model" "$mode" "$task" "$(date --iso-8601=seconds)" >> "$OUTPUT/PROGRESS.jsonl"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" MAGNUM_CUDA_DEVICE=0 \
    timeout --signal=TERM --kill-after=15s 300s \
    "$PYTHON" -u -m omtrackvla.evaluation.end_to_end_closed_loop \
      --task "$task" \
      --split train \
      --dataset-index 1300 \
      --max-steps 130 \
      --initialization-scan-episodes 8 \
      --device cuda:0 \
      --model-config "${CONFIGS[$model]}" \
      --checkpoint "${CHECKPOINTS[$model]}" \
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
    printf '{"event":"failed","model":"%s","mode":"%s","task":"%s","exit_code":%s,"time":"%s"}\n' \
      "$model" "$mode" "$task" "$code" "$(date --iso-8601=seconds)" >> "$OUTPUT/PROGRESS.jsonl"
    return 1
  fi
  touch "$destination/COMPLETE"
  printf '{"event":"complete","model":"%s","mode":"%s","task":"%s","time":"%s"}\n' \
    "$model" "$mode" "$task" "$(date --iso-8601=seconds)" >> "$OUTPUT/PROGRESS.jsonl"
}

test "$GPU" != 0
test ! -e "$OUTPUT"
for model in "${!CONFIGS[@]}"; do
  test -s "${CONFIGS[$model]}"
  test -s "${CHECKPOINTS[$model]}"
done
mkdir -p "$OUTPUT"
touch "$OUTPUT/STARTED"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

failures=0
for model in v2_006b_parent v2_016_waypoint v2_017_calibrated; do
  for mode in polar_reactive se2_waypoint; do
    for task in stt dt at; do
      if ! rollout "$model" "$mode" "$task"; then
        failures=$((failures + 1))
      fi
    done
  done
done

if ((failures != 0)); then
  printf '%s\n' "$failures" > "$OUTPUT/FAILURE_COUNT"
  touch "$OUTPUT/FAILED"
  exit 1
fi

CUDA_VISIBLE_DEVICES= "$PYTHON" scripts/summarize_v2_018_phase3_unseen1300_gate.py \
  --root "$OUTPUT" \
  --output "$OUTPUT/SUMMARY.json" \
  --visibility-threshold 0.005 \
  > "$OUTPUT/SUMMARY.stdout.json"
touch "$OUTPUT/COMPLETE"
