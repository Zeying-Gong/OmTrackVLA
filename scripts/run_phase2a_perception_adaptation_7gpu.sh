#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python}"
MANIFEST="configs/manifests/phase1_v1.json"
DATA_ROOT="/data/nfs/share/OmTrackVLA/data/tpt_bench_clean_v2"
DETECTOR_WEIGHTS="models/torchvision/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
BASELINE_FUSION="configs/models/candidate_fusion_resnet50_tpt_train37_dualop_v3.json"
BASELINE_VIZ_METRICS="outputs/evaluation/pretrained_identity_v4_resnet50_fusion_dualop_viz/comparison_metrics.json"
RUN_ID="${PHASE2A_RUN_ID:-phase2a_temporal_fusion_v5}"
TRAIN_ROOT="outputs/training/$RUN_ID"
ROLLOUT_ROOT="${PHASE2A_BASELINE_ROLLOUT_ROOT:-outputs/evaluation/${RUN_ID}_baseline_rollouts}"
EXTRA_TRAIN_FUSION="${PHASE2A_EXTRA_TRAIN_FUSION:-}"
EXTRA_TRAIN_ROOT="${PHASE2A_EXTRA_TRAIN_ROOT:-outputs/evaluation/${RUN_ID}_onpolicy_rollouts}"
VAL_ROOT="outputs/evaluation/${RUN_ID}_val"
VIZ_ROOT="outputs/evaluation/${RUN_ID}_viz"
FUSION_WEIGHTS="$TRAIN_ROOT/fusion.json"
FUSION_REPORT="$TRAIN_ROOT/train_report.json"
GPU_CSV="${PHASE2A_GPUS:-1,2,3,4,5,6,7}"
TRAIN_STRIDE="${PHASE2A_TRAIN_STRIDE:-1}"
TRAIN_EPOCHS="${PHASE2A_EPOCHS:-24}"

IFS=',' read -r -a GPUS <<<"$GPU_CSV"
if ((${#GPUS[@]} == 0)); then
  echo "PHASE2A_GPUS must contain at least one GPU" >&2
  exit 64
fi
for gpu in "${GPUS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]] || [[ "$gpu" == 0 ]]; then
    echo "Phase 2A must not use reserved physical GPU 0: $GPU_CSV" >&2
    exit 64
  fi
done
if [[ ! "$TRAIN_STRIDE" =~ ^[1-9][0-9]*$ ]]; then
  echo "PHASE2A_TRAIN_STRIDE must be a positive integer: $TRAIN_STRIDE" >&2
  exit 64
fi
if [[ ! "$TRAIN_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "PHASE2A_EPOCHS must be a positive integer: $TRAIN_EPOCHS" >&2
  exit 64
fi
if [[ -n "$EXTRA_TRAIN_FUSION" && ! -s "$EXTRA_TRAIN_FUSION" ]]; then
  echo "PHASE2A_EXTRA_TRAIN_FUSION does not exist: $EXTRA_TRAIN_FUSION" >&2
  exit 66
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
mkdir -p "$TRAIN_ROOT" "$ROLLOUT_ROOT" "$VAL_ROOT" "$VIZ_ROOT"

PIDS=()
cleanup() {
  if ((${#PIDS[@]})); then
    kill "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

wait_all() {
  local failures=0 pid
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      failures=$((failures + 1))
    fi
  done
  PIDS=()
  if ((failures)); then
    echo "$failures rollout worker(s) failed" >&2
    return 1
  fi
}

manifest_sequences() {
  local split="$1"
  "$PYTHON_BIN" - "$MANIFEST" "$split" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
if sys.argv[2] == "test_locked":
    raise SystemExit("test_locked is forbidden")
print("\n".join(manifest["datasets"]["tpt_bench_clean_v2"]["splits"][sys.argv[2]]))
PY
}

generate_rollouts() {
  local split="$1" stride="$2" output_root="$3" fusion_weights="$4"
  local -a sequences
  mapfile -t sequences < <(manifest_sequences "$split")
  mkdir -p "$output_root"/{logs,records,runs}
  echo "[$(date -Is)] split=$split sequences=${#sequences[@]} stride=$stride"
  local worker_index gpu
  for worker_index in "${!GPUS[@]}"; do
    gpu="${GPUS[$worker_index]}"
    (
      local index sequence output record
      for ((index=worker_index; index<${#sequences[@]}; index+=${#GPUS[@]})); do
        sequence="${sequences[$index]}"
        output="$output_root/runs/$sequence/metrics.json"
        record="$output_root/records/$sequence.jsonl"
        mkdir -p "$output_root/runs/$sequence"
        if [[ -s "$output" && -s "$record" ]]; then
          echo "[$(date -Is)] split=$split gpu=$gpu sequence=$sequence already complete"
          continue
        fi
        echo "[$(date -Is)] split=$split gpu=$gpu sequence=$sequence start"
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" \
          -m omtrackvla.evaluation.pretrained_identity \
          --manifest "$MANIFEST" \
          --data-root "$DATA_ROOT" \
          --split "$split" \
          --sequence-id "$sequence" \
          --output "$output" \
          --records-dir "$output_root/records" \
          --detector-architecture fasterrcnn_resnet50_fpn_v2 \
          --detector-weights "$DETECTOR_WEIGHTS" \
          --fusion-weights "$fusion_weights" \
          --device cuda \
          --frame-stride "$stride" \
          --progress-every 500 \
          >"$output_root/logs/$sequence.log" 2>&1
        echo "[$(date -Is)] split=$split gpu=$gpu sequence=$sequence complete"
      done
    ) &
    PIDS+=("$!")
  done
  wait_all
}

aggregate_runs() {
  local input_root="$1" split="$2" output="$3"
  "$PYTHON_BIN" - "$input_root" "$MANIFEST" "$split" "$output" <<'PY'
import json
import sys
from pathlib import Path

from omtrackvla.evaluation.pretrained_identity import _aggregate

root = Path(sys.argv[1])
manifest = json.load(open(sys.argv[2], encoding="utf-8"))
split = sys.argv[3]
output = Path(sys.argv[4])
if split == "test_locked":
    raise SystemExit("test_locked is forbidden")
sequence_ids = manifest["datasets"]["tpt_bench_clean_v2"]["splits"][split]
summaries = []
for sequence_id in sequence_ids:
    payload = json.loads((root / "runs" / sequence_id / "metrics.json").read_text())
    if payload.get("split") != split:
        raise ValueError(f"split mismatch for {sequence_id}")
    summaries.extend(payload["sequences"])
value = {
    "schema_version": 1,
    "split": split,
    "test_locked_used": False,
    "sequences": sequence_ids,
    "aggregate": _aggregate(summaries),
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
print(json.dumps(value["aggregate"], sort_keys=True))
PY
}

echo "[$(date -Is)] generating current-policy train and val records"
generate_rollouts train "$TRAIN_STRIDE" "$ROLLOUT_ROOT/train" "$BASELINE_FUSION"
generate_rollouts val 1 "$ROLLOUT_ROOT/val" "$BASELINE_FUSION"
aggregate_runs "$ROLLOUT_ROOT/val" val "$ROLLOUT_ROOT/val/aggregate.json"
TRAIN_RECORDS=("$ROLLOUT_ROOT/train/records")
if [[ -n "$EXTRA_TRAIN_FUSION" ]]; then
  echo "[$(date -Is)] generating on-policy train records for dataset aggregation"
  generate_rollouts \
    train "$TRAIN_STRIDE" "$EXTRA_TRAIN_ROOT/train" "$EXTRA_TRAIN_FUSION"
  TRAIN_RECORDS+=("$EXTRA_TRAIN_ROOT/train/records")
fi

echo "[$(date -Is)] training temporal hard-negative fusion"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON_BIN" \
  -m omtrackvla.training.train_temporal_candidate_fusion \
  --records "${TRAIN_RECORDS[@]}" \
  --validation-records "$ROLLOUT_ROOT/val/records" \
  --initial-weights "$BASELINE_FUSION" \
  --train-new-features-only \
  --operating-point-source inherited \
  --output "$FUSION_WEIGHTS" \
  --report "$FUSION_REPORT" \
  --epochs "$TRAIN_EPOCHS" \
  --hidden-dim 32 \
  --learning-rate 1e-4 \
  --weight-decay 1e-5 \
  --pairwise-weight 0.25 \
  --device cuda \
  >"$TRAIN_ROOT/train.log" 2>&1

echo "[$(date -Is)] evaluating trained fusion on val"
generate_rollouts val 1 "$VAL_ROOT" "$FUSION_WEIGHTS"
aggregate_runs "$VAL_ROOT" val "$VAL_ROOT/aggregate.json"

"$PYTHON_BIN" - "$ROLLOUT_ROOT/val/aggregate.json" "$VAL_ROOT/aggregate.json" <<'PY'
import json
import sys

baseline = json.load(open(sys.argv[1]))["aggregate"]
candidate = json.load(open(sys.argv[2]))["aggregate"]
checks = {
    "e2e_improves": candidate["end_to_end_success_iou_0_5"] > baseline["end_to_end_success_iou_0_5"],
    "precision_not_degraded": candidate["output_precision_iou_0_5"] >= baseline["output_precision_iou_0_5"] - 0.02,
    "absent_fpr_not_degraded": candidate["absent_false_positive_rate"] <= baseline["absent_false_positive_rate"] + 0.01,
}
print(json.dumps({"baseline": baseline, "candidate": candidate, "checks": checks}, sort_keys=True))
if not all(checks.values()):
    raise SystemExit("trained fusion failed val safeguards")
PY

echo "[$(date -Is)] val safeguards passed; evaluating fixed viz_val"
generate_rollouts viz_val 1 "$VIZ_ROOT" "$FUSION_WEIGHTS"
mapfile -t VIZ_METRICS < <(
  find "$VIZ_ROOT/runs" -mindepth 2 -maxdepth 2 -name metrics.json -type f | sort
)
"$PYTHON_BIN" -m omtrackvla.evaluation.phase2a_perception_gate \
  --config configs/gates/phase2a_perception.yaml \
  --baseline "$BASELINE_VIZ_METRICS" \
  --candidate "${VIZ_METRICS[@]}" \
  --output "$VIZ_ROOT/gate.json" \
  >"$VIZ_ROOT/gate.log" 2>&1
aggregate_runs "$VIZ_ROOT" viz_val "$VIZ_ROOT/aggregate.json"
touch "$TRAIN_ROOT/PHASE2A_COMPLETE"
echo "[$(date -Is)] Phase 2A gate passed"
