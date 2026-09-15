#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/nfs/share/wam_tracking/OmTrackVLA
PYTHON=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
CONFIG="$ROOT/configs/phases/v2_006b_evt_polar_only.yaml"
CHECKPOINT="$ROOT/outputs/training/v2_006b_evt_polar_only/checkpoints/best.ckpt"
OUTPUT="${OUTPUT:-$ROOT/outputs/evaluation/v2_006b_polar_closed_loop_pair}"
THRESHOLD="${THRESHOLD:-0.00005144220995134674}"
DATASET_INDEX="${DATASET_INDEX:-2}"
STATUS="$OUTPUT/STATUS.json"

test ! -e "$OUTPUT"
mkdir -p "$OUTPUT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4

write_status() {
  "$PYTHON" -c 'import json,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"stage":sys.argv[2],"status":sys.argv[3],"physical_gpu":3,"physical_gpu_zero_used":False,"split":"train","test_locked_used":False,"visibility_threshold":float(sys.argv[4]),"updated_unix":time.time()},indent=2,sort_keys=True)+"\n")' "$STATUS" "$1" "$2" "$THRESHOLD"
}

fail() {
  local code=$?
  write_status failed failed
  printf '%s\n' "$code" > "$OUTPUT/EXIT_CODE"
  exit "$code"
}
trap fail ERR

write_status preflight running
"$PYTHON" -m py_compile omtrackvla/evaluation/end_to_end_closed_loop.py
CUDA_VISIBLE_DEVICES= "$PYTHON" -m unittest tests.test_end_to_end_closed_loop

for mode in se2_waypoint polar_reactive; do
  destination="$OUTPUT/$mode"
  mkdir -p "$destination"
  write_status "$mode" running
  CUDA_VISIBLE_DEVICES=3 timeout --signal=TERM --kill-after=15s 300s \
    "$PYTHON" -m omtrackvla.evaluation.end_to_end_closed_loop \
    --task stt \
    --split train \
    --dataset-index "$DATASET_INDEX" \
    --max-steps 300 \
    --initialization-scan-episodes 8 \
    --device cuda:0 \
    --model-config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output-root "$destination" \
    --uwb-mode missing \
    --action-mode "$mode" \
    --visual-stop-threshold "$THRESHOLD" \
    --video-fps 8 \
    > "$destination/rollout.log" 2>&1
  test -s "$destination/result.json"
done

write_status aggregation running
"$PYTHON" - "$OUTPUT" "$THRESHOLD" "$DATASET_INDEX" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = {}
for mode in ("se2_waypoint", "polar_reactive"):
    result = json.loads((root / mode / "result.json").read_text())
    rows[mode] = {
        "episode_id": result["episode_id"],
        "checkpoint_sha256": result["loading"]["checkpoint_sha256"],
        **result["summary"],
    }
if rows["se2_waypoint"]["episode_id"] != rows["polar_reactive"]["episode_id"]:
    raise RuntimeError("paired modes did not use the same episode")
if rows["se2_waypoint"]["checkpoint_sha256"] != rows["polar_reactive"]["checkpoint_sha256"]:
    raise RuntimeError("paired modes did not use the same checkpoint")
summary = {
    "schema_version": 1,
    "status": "complete",
    "protocol": "one paired EVT train episode; visual-only; same Polar checkpoint",
    "task": "stt",
    "dataset_index": int(sys.argv[3]),
    "visibility_threshold": float(sys.argv[2]),
    "modes": rows,
    "formal_benchmark": False,
    "physical_gpu_zero_used": False,
    "test_locked_used": False,
}
(root / "SUMMARY.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

write_status complete complete
printf '%s\n' 0 > "$OUTPUT/EXIT_CODE"
touch "$OUTPUT/COMPLETE"
