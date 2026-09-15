#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/outputs/perception/sage3d_phase2_frozen_frontend_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/ablations/next026_abl09_frozen_frontend_formal_v1}"
CONFIG="$REPO_ROOT/configs/phases/next026_abl09_frozen_frontend_formal.yaml"
BENCHMARK="$REPO_ROOT/configs/benchmarks/next026_abl09_frozen_frontend.yaml"
ARCHITECTURE_SUMMARY="$REPO_ROOT/outputs/ablations/next026_abl08_converged_v1/summary.json"
DATA_ROOT="/data/nfs/share/OmTrackVLA/data/sage3d_extracted"
STATUS_LOG="$OUTPUT_ROOT/orchestrator.log"

mkdir -p "$OUTPUT_ROOT"
exec > >(tee -a "$STATUS_LOG") 2>&1

status() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

write_manifest() {
  local path="$1"
  "$PYTHON_BIN" - "$path" <<'PY'
import json
import pathlib
import socket
import sys

path = pathlib.Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({
    "run_id": "next026_abl09_frozen_frontend_formal_v1",
    "host": socket.gethostname(),
    "test_locked_used": False,
    "purpose": "ABL-V1-09 frozen OSNet/cache plus MLP baseline",
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

on_error() {
  local code=$?
  status "ABL-09 failed with exit code $code"
  "$PYTHON_BIN" - "$OUTPUT_ROOT/FAILED.json" "$code" <<'PY'
import json
import pathlib
import sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "status": "failed", "exit_code": int(sys.argv[2]), "test_locked_used": False,
}, indent=2) + "\n", encoding="utf-8")
PY
  exit "$code"
}
trap on_error ERR

if [[ -f "$OUTPUT_ROOT/ABL09_COMPLETE.json" ]]; then
  status "ABL-09 is already complete"
  exit 0
fi

status "waiting for complete train/val/viz_val frozen-perception cache"
last_cached=-1
stalled_intervals=0
while [[ ! -f "$CACHE_ROOT/CACHE_COMPLETE.json" ]]; do
  cached="$(find "$CACHE_ROOT" -path '*/episodes/*.json' -type f 2>/dev/null | wc -l)"
  workers="$(pgrep -fc 'build_sage3d_perception_cache.py' || true)"
  status "cache files=$cached workers=$workers"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
  if ((workers == 0)); then
    status "cache has no live workers and no completion marker"
    exit 3
  fi
  if ((cached == last_cached)); then
    stalled_intervals=$((stalled_intervals + 1))
  else
    stalled_intervals=0
  fi
  if ((stalled_intervals >= 3)); then
    status "cache file count did not advance for three 10-minute intervals"
    exit 4
  fi
  last_cached="$cached"
  sleep 600
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
status "cache complete; running ABL-09 unit tests"
PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" -m unittest \
  tests.test_phase2_training \
  tests.test_phase2_evaluate \
  tests.test_phase2_model \
  tests.test_phase2_data \
  tests.test_merge_phase2_cache \
  tests.test_summarize_next026_abl09

SMOKE_ROOT="$OUTPUT_ROOT/smoke_2step"
if [[ ! -f "$SMOKE_ROOT/SMOKE_COMPLETE.json" ]]; then
  if [[ -d "$SMOKE_ROOT" && -n "$(find "$SMOKE_ROOT" -mindepth 1 -print -quit)" ]]; then
    status "refusing to overwrite incomplete smoke directory: $SMOKE_ROOT"
    exit 2
  fi
  mkdir -p "$SMOKE_ROOT"
  write_manifest "$SMOKE_ROOT/run_manifest.json"
  status "starting two-step frozen-front-end training smoke on physical GPU1"
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
    -m omtrackvla.training.train \
    --phase 2 \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    --output-dir "$SMOKE_ROOT" \
    --run-manifest "$SMOKE_ROOT/run_manifest.json" \
    --max-steps 2 \
    --num-workers 2
  "$PYTHON_BIN" - "$SMOKE_ROOT" <<'PY'
import json
import pathlib
import sys
import torch
root = pathlib.Path(sys.argv[1])
checkpoint = torch.load(root / "checkpoints/last.ckpt", map_location="cpu", weights_only=False)
if checkpoint.get("global_step") != 2 or "scheduler" not in checkpoint:
    raise SystemExit("two-step smoke checkpoint is incomplete")
(root / "SMOKE_COMPLETE.json").write_text(json.dumps({
    "status": "passed", "global_step": 2, "scheduler_saved": True,
    "test_locked_used": False,
}, indent=2) + "\n")
PY
  status "two-step smoke passed"
fi

FORMAL_ROOT="$OUTPUT_ROOT/formal_36864"
mkdir -p "$FORMAL_ROOT"
write_manifest "$FORMAL_ROOT/run_manifest.json"
if [[ ! -f "$FORMAL_ROOT/TRAINING_COMPLETE.json" ]]; then
  resume_args=()
  if [[ -f "$FORMAL_ROOT/checkpoints/last.ckpt" ]]; then
    resume_args=(--resume-from "$FORMAL_ROOT/checkpoints/last.ckpt")
    status "resuming formal baseline from last epoch checkpoint"
  else
    status "starting formal 36,864-step baseline on physical GPU1"
  fi
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
    -m omtrackvla.training.train \
    --phase 2 \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    --output-dir "$FORMAL_ROOT" \
    --run-manifest "$FORMAL_ROOT/run_manifest.json" \
    "${resume_args[@]}"
  "$PYTHON_BIN" - "$FORMAL_ROOT/checkpoints/last.ckpt" "$FORMAL_ROOT/TRAINING_COMPLETE.json" <<'PY'
import json
import pathlib
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if checkpoint.get("global_step") != 36864 or "scheduler" not in checkpoint:
    raise SystemExit("formal baseline did not reach 36,864 steps")
pathlib.Path(sys.argv[2]).write_text(json.dumps({
    "status": "complete", "global_step": 36864, "test_locked_used": False,
}, indent=2) + "\n")
PY
fi

status "evaluating 1,024 val samples per mode on physical GPUs1-7"
mkdir -p "$FORMAL_ROOT/eval_1024"
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
  -m torch.distributed.run \
  --standalone --nproc-per-node=7 \
  -m omtrackvla.evaluation.evaluate \
  --phase 2 \
  --config "$BENCHMARK" \
  --data-root "$DATA_ROOT" \
  --checkpoint "$FORMAL_ROOT/checkpoints/best.ckpt" \
  --output-dir "$FORMAL_ROOT/eval_1024" \
  --metrics-out "$FORMAL_ROOT/eval_1024/metrics.json" \
  --report-out "$FORMAL_ROOT/eval_1024/report.md" \
  --samples-per-mode 1024

status "rendering fixed viz_val samples on physical GPU1"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
  -m omtrackvla.evaluation.render \
  --phase 2 \
  --config "$BENCHMARK" \
  --data-root "$DATA_ROOT" \
  --checkpoint "$FORMAL_ROOT/checkpoints/best.ckpt" \
  --split viz_val \
  --output-dir "$FORMAL_ROOT/render"

PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" scripts/summarize_next026_abl09.py \
  --baseline-metrics "$FORMAL_ROOT/eval_1024/metrics.json" \
  --architecture-summary "$ARCHITECTURE_SUMMARY" \
  --output "$OUTPUT_ROOT/summary.json"

"$PYTHON_BIN" - "$OUTPUT_ROOT/summary.json" "$OUTPUT_ROOT/ABL09_COMPLETE.json" <<'PY'
import hashlib
import json
import pathlib
import sys
summary = pathlib.Path(sys.argv[1])
payload = json.loads(summary.read_text(encoding="utf-8"))
if payload.get("test_locked_used") is not False:
    raise SystemExit("summary test_locked provenance is invalid")
pathlib.Path(sys.argv[2]).write_text(json.dumps({
    "status": "complete",
    "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
    "test_locked_used": False,
}, indent=2) + "\n")
PY
status "ABL-09 complete"
