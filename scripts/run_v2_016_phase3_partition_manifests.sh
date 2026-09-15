#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/nfs/share/wam_tracking/OmTrackVLA
PYTHON=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
OUTPUT="$ROOT/outputs/evaluation/v2_016_phase3_multiscene_partitions"
OLD_TRAIN="$ROOT/outputs/evaluation/v2_009_phase3_relabel_smoke/MANIFEST.json"
OLD_VAL="$ROOT/outputs/evaluation/v2_012c_phase3_validation_manifest.json"
SUMMARY="$ROOT/outputs/evaluation/v2_015_phase3_multiscene_relabel/SUMMARY.json"
AT_SUMMARY="$ROOT/outputs/evaluation/v2_015b_phase3_at_training_relabel/SUMMARY.json"

test ! -e "$OUTPUT"
for path in "$OLD_TRAIN" "$OLD_VAL" "$SUMMARY" "$AT_SUMMARY"; do
  test -s "$path"
done
mkdir -p "$OUTPUT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

mapfile -t train_samples < <(
  "$PYTHON" -c \
    'import json,sys; old=json.load(open(sys.argv[1])); new=json.load(open(sys.argv[2])); at=json.load(open(sys.argv[3])); print(*[x["path"] for x in old["samples"]], sep="\n"); print(*[x["sample"] for x in new["attempts"] if x["passed"] and x["dataset_index"]==300], sep="\n"); print(*[x["sample"] for x in at["attempts"] if x["passed"]], sep="\n")' \
    "$OLD_TRAIN" "$SUMMARY" "$AT_SUMMARY"
)
mapfile -t validation_samples < <(
  "$PYTHON" -c \
    'import json,sys; old=json.load(open(sys.argv[1])); new=json.load(open(sys.argv[2])); print(*[x["path"] for x in old["samples"]], sep="\n"); print(*[x["sample"] for x in new["attempts"] if x["passed"] and x["dataset_index"]==700], sep="\n")' \
    "$OLD_VAL" "$SUMMARY"
)

train_args=()
for sample in "${train_samples[@]}"; do
  [[ -n "$sample" ]] && train_args+=(--sample "$sample")
done
validation_args=()
for sample in "${validation_samples[@]}"; do
  [[ -n "$sample" ]] && validation_args+=(--sample "$sample")
done

"$PYTHON" scripts/build_v2_phase3_manifest.py \
  --output "$OUTPUT/TRAIN_MANIFEST.json" \
  --minimum-samples 18 \
  --minimum-per-task 5 \
  "${train_args[@]}"
"$PYTHON" scripts/build_v2_phase3_manifest.py \
  --output "$OUTPUT/RECOVERY_VAL_MANIFEST.json" \
  --minimum-samples 12 \
  --minimum-per-task 3 \
  "${validation_args[@]}"
"$PYTHON" scripts/audit_v2_phase3_observability.py \
  --manifest "$OUTPUT/TRAIN_MANIFEST.json" \
  --output "$OUTPUT/TRAIN_OBSERVABILITY.json" \
  --partition train
"$PYTHON" scripts/audit_v2_phase3_observability.py \
  --manifest "$OUTPUT/RECOVERY_VAL_MANIFEST.json" \
  --output "$OUTPUT/RECOVERY_VAL_OBSERVABILITY.json" \
  --partition recovery_val
"$PYTHON" scripts/verify_v2_phase3_partition_disjoint.py \
  --train-manifest "$OUTPUT/TRAIN_MANIFEST.json" \
  --validation-manifest "$OUTPUT/RECOVERY_VAL_MANIFEST.json" \
  --output "$OUTPUT/PARTITION_DISJOINT_AUDIT.json"
touch "$OUTPUT/COMPLETE"
