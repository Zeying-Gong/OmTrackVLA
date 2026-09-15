#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python}"
PHYSICAL_GPU="${PHYSICAL_GPU:?PHYSICAL_GPU is required}"
TASK="${TASK:?TASK is required}"
DATASET_INDEX="${DATASET_INDEX:?DATASET_INDEX is required}"
TAG="${TAG:-${TASK}_${DATASET_INDEX}}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-9}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-$ROOT/outputs/evaluation/next007_train_rollout_${TAG}_v1}"
RELABEL_ROOT="${RELABEL_ROOT:-$ROOT/outputs/evaluation/next007_train_relabel_${TAG}_v1}"
PIPELINE_ROOT="${PIPELINE_ROOT:-$ROOT/outputs/evaluation/next007_train_candidate_${TAG}_v1}"

mkdir -p "$PIPELINE_ROOT"
exec > >(tee -a "$PIPELINE_ROOT/orchestrator.log") 2>&1

status() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

if [[ -e "$ROLLOUT_ROOT" || -e "$RELABEL_ROOT" ]]; then
  status "refusing to overwrite an existing rollout or relabel directory"
  exit 2
fi

status "starting $TASK train candidate from dataset index $DATASET_INDEX on GPU $PHYSICAL_GPU"
mkdir -p "$ROLLOUT_ROOT"
set +e
env \
  PHYSICAL_GPU="$PHYSICAL_GPU" \
  TASK="$TASK" \
  SPLIT=train \
  DATASET_INDEX="$DATASET_INDEX" \
  INITIALIZATION_SCAN_EPISODES=8 \
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

"$PYTHON_BIN" - "$rollout_result" "$TASK" "$CHECKPOINT_STEP" <<'PY'
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
task = sys.argv[2]
checkpoint_step = int(sys.argv[3])
steps = value.get("steps")
if value.get("split") != "train" or value.get("task") != task:
    raise SystemExit("rollout provenance is not the requested train task")
if not isinstance(steps, list) or len(steps) < checkpoint_step:
    raise SystemExit("rollout is shorter than the requested checkpoint")
PY

status "rollout exit=$rollout_code; relabeling post-step-$CHECKPOINT_STEP"
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
  status "expert relabel failed quality gate with exit=$relabel_code"
  exit "$relabel_code"
fi

"$PYTHON_BIN" - "$PIPELINE_ROOT/PIPELINE_COMPLETE.json" \
  "$rollout_result" "$RELABEL_ROOT/report.json" <<'PY'
import hashlib
import json
import pathlib
import sys

output, rollout, report = map(pathlib.Path, sys.argv[1:])
value = json.loads(report.read_text(encoding="utf-8"))
sample = pathlib.Path(value["phase3_smoke_sample"]["manifest"])
if value.get("status") != "passed":
    raise SystemExit("relabel report did not pass")
if value["phase3_smoke_sample"].get("formal_training_eligible") is not True:
    raise SystemExit("candidate is not formal-training eligible")
payload = {
    "status": "complete",
    "rollout_result": str(rollout),
    "relabel_report": str(report),
    "relabel_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    "sample": str(sample),
    "sample_sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
    "formal_training_eligible": True,
    "test_locked_used": False,
}
temporary = output.with_name(output.name + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY
status "candidate pipeline complete"
