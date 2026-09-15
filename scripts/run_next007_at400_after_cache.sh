#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python}"
PHYSICAL_GPU="${PHYSICAL_GPU:-4}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT/outputs/perception/sage3d_phase2_frozen_frontend_v1}"
PIPELINE_ROOT="${PIPELINE_ROOT:-$ROOT/outputs/evaluation/next007_train_at400_pipeline_v2}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-$ROOT/outputs/evaluation/next007_train_rollout_at_400_v2}"
RELABEL_ROOT="${RELABEL_ROOT:-$ROOT/outputs/evaluation/next007_train_relabel_at_400_v1}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-9}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

mkdir -p "$PIPELINE_ROOT"
exec > >(tee -a "$PIPELINE_ROOT/orchestrator.log") 2>&1

status() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

if [[ -f "$PIPELINE_ROOT/PIPELINE_COMPLETE.json" ]]; then
  status "pipeline already complete"
  exit 0
fi
if [[ -e "$ROLLOUT_ROOT" || -e "$RELABEL_ROOT" ]]; then
  status "refusing to overwrite an existing rollout or relabel directory"
  exit 2
fi

status "waiting for ABL-09 train/val/viz_val cache completion"
last_report=0
while [[ ! -f "$CACHE_ROOT/CACHE_COMPLETE.json" ]]; do
  now="$(date +%s)"
  if ((now - last_report >= 600)); then
    cached="$(find "$CACHE_ROOT" -path '*/episodes/*.json' -type f 2>/dev/null | wc -l)"
    workers="$(pgrep -fc 'build_sage3d_perception_cache.py' || true)"
    status "cache files=$cached workers=$workers"
    last_report="$now"
  fi
  sleep "$WAIT_SECONDS"
done

"$PYTHON_BIN" - "$CACHE_ROOT/CACHE_COMPLETE.json" <<'PY'
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if value.get("status") != "complete" or value.get("test_locked_used") is not False:
    raise SystemExit("cache completion marker is invalid")
if set(value.get("splits", ())) != {"train", "val", "viz_val"}:
    raise SystemExit("cache completion marker does not cover train/val/viz_val")
PY
while pgrep -f 'build_sage3d_perception_cache.py' >/dev/null; do
  status "completion marker exists; waiting for cache workers to exit"
  sleep "$WAIT_SECONDS"
done

status "starting isolated AT train index 400 rollout on physical GPU $PHYSICAL_GPU"
mkdir -p "$ROLLOUT_ROOT"
set +e
env \
  PHYSICAL_GPU="$PHYSICAL_GPU" \
  TASK=at \
  SPLIT=train \
  DATASET_INDEX=400 \
  INITIALIZATION_SCAN_EPISODES=1 \
  INITIALIZATION_WAIT_STEPS=32 \
  MAX_STEPS=12 \
  OUTPUT_ROOT="$ROLLOUT_ROOT" \
  bash scripts/run_next027_closed_loop_smoke.sh \
  >"$ROLLOUT_ROOT/run.log" 2>&1
rollout_code=$?
set -e
printf '%s\n' "$rollout_code" >"$ROLLOUT_ROOT/exit_code.txt"

if [[ -f "$ROLLOUT_ROOT/result.json" ]]; then
  rollout_result="$ROLLOUT_ROOT/result.json"
elif [[ -f "$ROLLOUT_ROOT/result.partial.json" ]]; then
  rollout_result="$ROLLOUT_ROOT/result.partial.json"
else
  status "rollout exit=$rollout_code and produced no recoverable result"
  exit "$rollout_code"
fi

"$PYTHON_BIN" - "$rollout_result" "$CHECKPOINT_STEP" <<'PY'
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
steps = value.get("steps")
checkpoint_step = int(sys.argv[2])
if value.get("split") != "train" or value.get("dataset_index") != 400:
    raise SystemExit("rollout provenance is not AT train index 400")
if not isinstance(steps, list) or len(steps) < checkpoint_step:
    raise SystemExit(
        f"rollout has {len(steps) if isinstance(steps, list) else 0} steps, "
        f"fewer than checkpoint {checkpoint_step}"
    )
PY

status "rollout exit=$rollout_code; relabeling the post-step-$CHECKPOINT_STEP state"
mkdir -p "$RELABEL_ROOT"
set +e
env \
  PHYSICAL_GPU="$PHYSICAL_GPU" \
  ROLLOUT_RESULT="$rollout_result" \
  OUTPUT="$RELABEL_ROOT/report.json" \
  PHASE3_SAMPLE_DIR="$RELABEL_ROOT/phase3_sample" \
  CHECKPOINT_STEP="$CHECKPOINT_STEP" \
  EXPERT_TARGET_LOOKAHEAD_STEPS=21 \
  bash scripts/run_next007_expert_relabel_probe.sh \
  >"$RELABEL_ROOT/run.log" 2>&1
relabel_code=$?
set -e
printf '%s\n' "$relabel_code" >"$RELABEL_ROOT/exit_code.txt"
if ((relabel_code != 0)); then
  status "expert relabel failed with exit=$relabel_code"
  exit "$relabel_code"
fi

"$PYTHON_BIN" - "$PIPELINE_ROOT/PIPELINE_COMPLETE.json" "$rollout_result" \
  "$ROLLOUT_ROOT/exit_code.txt" "$RELABEL_ROOT/report.json" <<'PY'
import hashlib
import json
import pathlib
import sys

output, rollout, exit_file, report = map(pathlib.Path, sys.argv[1:])
report_value = json.loads(report.read_text(encoding="utf-8"))
sample = pathlib.Path(report_value["phase3_smoke_sample"]["manifest"])
if report_value.get("status") != "passed":
    raise SystemExit("relabel report did not pass")
if report_value["phase3_smoke_sample"].get("formal_training_eligible") is not True:
    raise SystemExit("train relabel sample is not formal-training eligible")
value = {
    "status": "complete",
    "rollout_result": str(rollout),
    "rollout_exit_code": int(exit_file.read_text().strip()),
    "relabel_report": str(report),
    "relabel_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    "sample": str(sample),
    "sample_sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
    "formal_training_eligible": True,
    "test_locked_used": False,
}
temporary = output.with_name(output.name + ".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(output)
PY
status "AT-400 rollout plus train relabel pipeline complete"
