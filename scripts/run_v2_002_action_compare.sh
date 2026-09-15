#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
CONFIG="$ROOT/configs/phases/phase2_end_to_end_v2_action_compare.yaml"
INDEX="$ROOT/outputs/indexes/architecture_v2_sage3d_sequences_v1.json"
SOURCE_INDEX="$ROOT/outputs/indexes/architecture_v1_sage3d_sequences_v1.json"
OUTPUT="$ROOT/outputs/training/v2_002_action_compare"
PREFLIGHT="$ROOT/outputs/training/v2_002_action_compare_preflight"
EVAL_ROOT="$ROOT/outputs/evaluation/v2_002_action_compare"
STATUS="$OUTPUT/STATUS.json"

mkdir -p "$OUTPUT" "$EVAL_ROOT"
exec > >(tee -a "$OUTPUT/launcher.log") 2>&1
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

write_status() {
  local stage="$1"
  local state="$2"
  local detail="${3:-}"
  "$PYTHON" -c 'import json,sys,time; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(json.dumps({"stage":sys.argv[2],"status":sys.argv[3],"detail":sys.argv[4],"updated_unix":time.time(),"physical_gpu_zero_used":False,"test_locked_used":False},indent=2,sort_keys=True)+"\n")' "$STATUS" "$stage" "$state" "$detail"
}

failed() {
  local code=$?
  write_status "failed" "failed" "launcher exit code $code; inspect launcher.log"
  exit "$code"
}
trap failed ERR

write_status "preflight" "running" "syntax, unit tests, and 2-step real-DA3 check"
"$PYTHON" -m py_compile \
  omtrackvla/models/end_to_end_v2.py \
  omtrackvla/data/end_to_end_training.py \
  omtrackvla/training/end_to_end_v2.py \
  omtrackvla/evaluation/end_to_end_closed_loop.py
"$PYTHON" tests/test_end_to_end_v2.py
"$PYTHON" tests/test_end_to_end_data.py

if [[ ! -f "$INDEX" ]]; then
  "$PYTHON" scripts/build_end_to_end_v2_index.py \
    --source-index "$SOURCE_INDEX" \
    --output "$INDEX"
fi

if [[ ! -f "$PREFLIGHT/TRAINING_COMPLETE.json" ]]; then
  "$PYTHON" -m torch.distributed.run \
    --standalone --nproc_per_node=7 --master_port=29642 \
    --module omtrackvla.training.end_to_end_v2 \
    --config "$CONFIG" --output-dir "$PREFLIGHT" --max-steps 2
fi

write_status "training" "running" "4096 optimizer steps; one shared trajectory+action checkpoint"
if [[ ! -f "$OUTPUT/TRAINING_COMPLETE.json" ]]; then
  "$PYTHON" -m torch.distributed.run \
    --standalone --nproc_per_node=7 --master_port=29643 \
    --module omtrackvla.training.end_to_end_v2 \
    --config "$CONFIG" --output-dir "$OUTPUT"
fi

CHECKPOINT="$OUTPUT/checkpoints/best.ckpt"
write_status "evt_evaluation" "running" "36 non-locked train-split rollouts; direct/waypoint_dt/lookahead"

run_rollout() {
  local task="$1"
  local mode="$2"
  local index="$3"
  local gpu="$4"
  local destination="$EVAL_ROOT/$task/$mode/$index"
  mkdir -p "$destination"
  if [[ -f "$destination/result.json" ]]; then
    return 0
  fi
  local exit_code=0
  CUDA_VISIBLE_DEVICES="$gpu" timeout --signal=TERM --kill-after=15s 300s \
    "$PYTHON" -m omtrackvla.evaluation.end_to_end_closed_loop \
    --task "$task" \
    --split train \
    --dataset-index "$index" \
    --max-steps 300 \
    --initialization-scan-episodes 8 \
    --device cuda:0 \
    --model-config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output-root "$destination" \
    --uwb-mode missing \
    --action-mode "$mode" \
    --no-video \
    > "$destination/rollout.log" 2>&1 || exit_code=$?
  # Habitat may return 134 while tearing down a CUDA child after the runner
  # has already atomically published a complete result.  The result file is
  # the durable success contract; a missing result is recorded and the other
  # comparison arms are still allowed to finish.
  if [[ -f "$destination/result.json" ]]; then
    rm -f "$destination/exit_code.txt"
    return 0
  fi
  printf '%s\n' "$exit_code" > "$destination/exit_code.txt"
  echo "rollout did not publish result: task=$task mode=$mode index=$index exit=$exit_code"
  return 0
}

jobs=0
# Habitat/EGL evaluation is intentionally serialized on this host. Multiple
# simultaneous simulator processes were observed to enter a shared driver
# rwsem wait at the same environment step despite using distinct GPUs.
for task in stt dt at; do
  for mode in direct waypoint_dt lookahead; do
    for index in 0 1 2 3; do
      # GPU3 is the only device on this host that completed the real Habitat
      # rollout without entering the low-power 100%-utilization driver spin.
      gpu=3
      run_rollout "$task" "$mode" "$index" "$gpu"
      jobs=$((jobs + 1))
    done
  done
done

write_status "aggregation" "running" "computing EVT SR/TR/collision summary"
"$PYTHON" scripts/summarize_v2_action_compare.py \
  --root "$EVAL_ROOT" \
  --output "$EVAL_ROOT/summary.json"
write_status "complete" "complete" "SAGE3D metrics and 36 EVT rollouts are ready"
touch "$OUTPUT/OVERNIGHT_COMPLETE"
echo "V2-002 action comparison complete"
