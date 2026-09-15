#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/nfs/share/wam_tracking/OmTrackVLA
PYTHON=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
CONFIG="$ROOT/configs/phases/v2_006_evt_perception_polar_headonly.yaml"
INITIAL="$ROOT/outputs/training/v2_005_se2_adapter_balanced/checkpoints/best.ckpt"
OUTPUT="$ROOT/outputs/training/v2_006_evt_perception_polar_headonly"
STATUS="$OUTPUT/STATUS.json"

mkdir -p "$OUTPUT"
cd "$ROOT"
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

write_status() {
  local stage="$1"
  local state="$2"
  local completed="${3:-0}"
  "$PYTHON" -c 'import json,sys,time; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(json.dumps({"stage":sys.argv[2],"status":sys.argv[3],"completed_optimizer_steps":int(sys.argv[4]),"target_optimizer_steps":512,"physical_gpus":[1,2,3,4,5,6,7],"physical_gpu_zero_used":False,"test_locked_used":False,"updated_unix":time.time()},indent=2,sort_keys=True)+"\n")' "$STATUS" "$stage" "$state" "$completed"
}

on_error() {
  local code=$?
  write_status failed failed 0
  printf '%s\n' "$code" > "$OUTPUT/EXIT_CODE"
  exit "$code"
}
trap on_error ERR

test ! -e "$OUTPUT/TRAINING_COMPLETE.json"
write_status preflight running 0
"$PYTHON" -m py_compile \
  omtrackvla/models/end_to_end_v2.py \
  omtrackvla/data/evt_perception.py \
  omtrackvla/training/evt_perception.py
"$PYTHON" tests/test_end_to_end_v2.py
"$PYTHON" tests/test_evt_perception_data.py

write_status zero_update_dev running 0
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m omtrackvla.training.evt_perception \
  --config "$CONFIG" \
  --output-dir "$OUTPUT/baseline_dev" \
  --checkpoint "$INITIAL" \
  --eval-only \
  --render-samples 32 \
  > "$OUTPUT/baseline_dev.log" 2>&1

write_status head_only_training running 0
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 "$PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node=7 --master_port=29706 \
  --module omtrackvla.training.evt_perception \
  --config "$CONFIG" \
  --output-dir "$OUTPUT" \
  --initial-checkpoint "$INITIAL" \
  > "$OUTPUT/train.log" 2>&1

write_status final_dev running 512
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m omtrackvla.training.evt_perception \
  --config "$CONFIG" \
  --output-dir "$OUTPUT/final_dev" \
  --checkpoint "$OUTPUT/checkpoints/best.ckpt" \
  --eval-only \
  --render-samples 32 \
  > "$OUTPUT/final_dev.log" 2>&1

"$PYTHON" - "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
before = json.loads((root / "baseline_dev/EVT_DEV_METRICS.json").read_text())["metrics"]["all"]
after = json.loads((root / "final_dev/EVT_DEV_METRICS.json").read_text())["metrics"]["all"]
comparison = {
    "status": "comparison_complete",
    "optimizer_steps": 512,
    "baseline": before,
    "head_only": after,
    "delta_head_only_minus_baseline": {
        "bbox_iou": after["bbox_iou"] - before["bbox_iou"],
        "visibility_balanced_accuracy": (
            after["visibility"]["balanced_accuracy"]
            - before["visibility"]["balanced_accuracy"]
        ),
        "angle_mae_deg": after["angle_mae_deg"] - before["angle_mae_deg"],
        "distance_mae_m": after["distance_mae_m"] - before["distance_mae_m"],
    },
    "interpretation": (
        "negative deltas are improvements for angle/distance MAE; positive deltas "
        "are improvements for bbox IoU/visibility balanced accuracy"
    ),
    "formal_large_scale_training": False,
    "official_evt_validation": False,
    "test_locked_used": False,
}
(root / "COMPARISON.json").write_text(
    json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(comparison, sort_keys=True))
PY

write_status complete complete 512
printf '%s\n' 0 > "$OUTPUT/EXIT_CODE"
touch "$OUTPUT/COMPLETE"

