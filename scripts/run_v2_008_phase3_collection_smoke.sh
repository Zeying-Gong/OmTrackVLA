#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/nfs/share/wam_tracking/OmTrackVLA
PYTHON=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
CONFIG="$ROOT/configs/phases/v2_006b_evt_polar_only.yaml"
CHECKPOINT="$ROOT/outputs/training/v2_006b_evt_polar_only/checkpoints/best.ckpt"
OUTPUT_ROOT="$ROOT/outputs/evaluation/v2_008b_phase3_collection_smoke"
VISIBILITY_THRESHOLD=0.005

worker() {
  local gpu="$1"
  local task="$2"
  local dataset_index="$3"
  local tag="${task}_${dataset_index}"
  local output="$OUTPUT_ROOT/$tag"

  test "$gpu" != 0
  test "$task" = stt -o "$task" = dt -o "$task" = at
  test ! -e "$output"
  mkdir -p "$output"
  cd "$ROOT"
  export CUDA_VISIBLE_DEVICES="$gpu"
  export MAGNUM_CUDA_DEVICE=0
  export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
  export OMP_NUM_THREADS=2
  export MKL_NUM_THREADS=2

  set +e
  timeout --signal=TERM --kill-after=15s 300s \
    "$PYTHON" -u -m omtrackvla.evaluation.end_to_end_closed_loop \
      --task "$task" \
      --split train \
      --dataset-index "$dataset_index" \
      --max-steps 130 \
      --initialization-scan-episodes 8 \
      --device cuda:0 \
      --model-config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --output-root "$output" \
      --uwb-mode missing \
      --action-mode polar_reactive \
      --visual-stop-threshold "$VISIBILITY_THRESHOLD" \
      --video-fps 8 \
      > "$output/rollout.log" 2>&1
  local code=$?
  set -e
  printf '%s\n' "$code" > "$output/EXIT_CODE"
  if ((code == 0)) && test -s "$output/result.json"; then
    touch "$output/COMPLETE"
  else
    touch "$output/FAILED"
  fi
  return "$code"
}

if [[ "${1:-}" == --worker ]]; then
  shift
  worker "$@"
  exit
fi

test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"
cd "$ROOT"

declare -a SPECS=(
  "1 stt 500"
  "2 stt 1700"
  "3 dt 500"
  "4 dt 1700"
  "5 at 500"
  "6 at 1700"
)

for spec in "${SPECS[@]}"; do
  read -r gpu task dataset_index <<< "$spec"
  session="v2_008_${task}_${dataset_index}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "session already exists: $session" >&2
    exit 2
  fi
  printf -v command '%q ' bash "$0" --worker "$gpu" "$task" "$dataset_index"
  tmux new-session -d -s "$session" "$command"
  echo "launched session=$session gpu=$gpu task=$task dataset_index=$dataset_index"
done

cat > "$OUTPUT_ROOT/PROTOCOL.txt" <<EOF
stage=v2_008b_phase3_collection_smoke
split=train
tasks=stt,dt,at
episodes_per_task=2
max_steps_per_episode=130
action_mode=polar_reactive
visibility_threshold=$VISIBILITY_THRESHOLD
checkpoint=$CHECKPOINT
gpu_zero_used=false
test_locked_used=false
formal_large_scale_training=false
EOF
