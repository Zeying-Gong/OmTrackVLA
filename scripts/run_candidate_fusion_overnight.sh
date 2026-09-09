#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python}"
MANIFEST="configs/manifests/phase1_v1.json"
DATA_ROOT="/data/nfs/share/OmTrackVLA/data/tpt_bench_clean_v2"
DETECTOR_WEIGHTS="models/torchvision/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
RUN_ID="candidate_fusion_resnet50_tpt_train37_dualop_v3"
TRAIN_RECORD_ROOT="outputs/evaluation/candidate_fusion_resnet50_tpt_train37_v2_records"
TRAIN_ROOT="outputs/training/${RUN_ID}"
EVAL_ROOT="outputs/evaluation/pretrained_identity_v4_resnet50_fusion_dualop_viz"
BASE_ROOT="outputs/evaluation/pretrained_identity_v2_resnet50_no_fusion_viz"
TRAIN8_ROOT="outputs/evaluation/pretrained_identity_v2_resnet50_fusion_viz"
STRICT_ROOT="outputs/evaluation/pretrained_identity_v3_resnet50_fusion_train37_viz"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

mkdir -p "$TRAIN_RECORD_ROOT"/{logs,records,runs} "$TRAIN_ROOT"
mkdir -p "$EVAL_ROOT"/{logs,records,runs/0005,runs/0032,visual_review}

mapfile -t TRAIN_SEQUENCES < <(
  "$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
print("\n".join(manifest["datasets"]["tpt_bench_clean_v2"]["splits"]["train"]))
PY
)

echo "[$(date -Is)] generating frozen detector/ReID candidates for ${#TRAIN_SEQUENCES[@]} train sequences"
worker() {
  local gpu="$1"
  local index sequence output record
  for ((index=gpu; index<${#TRAIN_SEQUENCES[@]}; index+=8)); do
    sequence="${TRAIN_SEQUENCES[$index]}"
    output="$TRAIN_RECORD_ROOT/runs/$sequence/metrics.json"
    record="$TRAIN_RECORD_ROOT/records/$sequence.jsonl"
    mkdir -p "$TRAIN_RECORD_ROOT/runs/$sequence"
    if [[ -s "$output" && -s "$record" ]]; then
      echo "[$(date -Is)] gpu=$gpu sequence=$sequence already complete"
      continue
    fi
    echo "[$(date -Is)] gpu=$gpu sequence=$sequence start"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" \
      -m omtrackvla.evaluation.pretrained_identity \
      --manifest "$MANIFEST" \
      --data-root "$DATA_ROOT" \
      --split train \
      --sequence-id "$sequence" \
      --output "$output" \
      --records-dir "$TRAIN_RECORD_ROOT/records" \
      --detector-architecture fasterrcnn_resnet50_fpn_v2 \
      --detector-weights "$DETECTOR_WEIGHTS" \
      --device cuda \
      --frame-stride 4 \
      --progress-every 500 \
      > "$TRAIN_RECORD_ROOT/logs/$sequence.log" 2>&1
    echo "[$(date -Is)] gpu=$gpu sequence=$sequence complete"
  done
}

worker_pids=()
for gpu in {0..7}; do
  worker "$gpu" &
  worker_pids+=("$!")
done
for pid in "${worker_pids[@]}"; do
  wait "$pid"
done

echo "[$(date -Is)] training candidate fusion head"
"$PYTHON_BIN" -m omtrackvla.training.train_candidate_fusion \
  --records "$TRAIN_RECORD_ROOT/records" \
  --output "$TRAIN_ROOT/fusion.json" \
  --report "$TRAIN_ROOT/report.json" \
  --epochs 80 \
  --hidden-dim 32 \
  --batch-size 2048 \
  --learning-rate 0.001 \
  > "$TRAIN_ROOT/train.log" 2>&1

echo "[$(date -Is)] evaluating complete viz_val sequences"
eval_pids=()
for pair in 0:0005 1:0032; do
  gpu="${pair%%:*}"
  sequence="${pair##*:}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" \
    -m omtrackvla.evaluation.pretrained_identity \
    --manifest "$MANIFEST" \
    --data-root "$DATA_ROOT" \
    --split viz_val \
    --sequence-id "$sequence" \
    --output "$EVAL_ROOT/runs/$sequence/metrics.json" \
    --records-dir "$EVAL_ROOT/records" \
    --detector-architecture fasterrcnn_resnet50_fpn_v2 \
    --detector-weights "$DETECTOR_WEIGHTS" \
    --fusion-weights "$TRAIN_ROOT/fusion.json" \
    --device cuda \
    --progress-every 500 \
    > "$EVAL_ROOT/logs/$sequence.log" 2>&1 &
  eval_pids+=("$!")
done
for pid in "${eval_pids[@]}"; do
  wait "$pid"
done

echo "[$(date -Is)] aggregating metrics and rendering review frames"
"$PYTHON_BIN" - "$EVAL_ROOT" "$BASE_ROOT" "$TRAIN8_ROOT" "$STRICT_ROOT" "$DATA_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import cv2

from omtrackvla.evaluation.pretrained_identity import _aggregate

evaluation_root = Path(sys.argv[1])
baseline_root = Path(sys.argv[2])
train8_root = Path(sys.argv[3])
strict_root = Path(sys.argv[4])
data_root = Path(sys.argv[5])
sequence_ids = ("0005", "0032")


def load_mode(root):
    payloads = [
        json.loads((root / "runs" / sequence / "metrics.json").read_text())
        for sequence in sequence_ids
    ]
    summaries = [payload["sequences"][0] for payload in payloads]
    return {
        "root": str(root),
        "aggregate": _aggregate(summaries),
        "sequences": summaries,
    }


results = {"dualop_fusion": load_mode(evaluation_root)}
if all((baseline_root / "runs" / sequence / "metrics.json").is_file() for sequence in sequence_ids):
    results["resnet50_no_fusion"] = load_mode(baseline_root)
if all((train8_root / "runs" / sequence / "metrics.json").is_file() for sequence in sequence_ids):
    results["train8_fusion"] = load_mode(train8_root)
if all((strict_root / "runs" / sequence / "metrics.json").is_file() for sequence in sequence_ids):
    results["single_strict_fusion"] = load_mode(strict_root)

metrics_path = evaluation_root / "comparison_metrics.json"
metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

keys = (
    "detector_candidate_recall_iou_0_5",
    "target_selection_accuracy_when_candidate_present",
    "end_to_end_success_iou_0_5",
    "visibility_recall",
    "output_precision_iou_0_5",
    "absent_false_positive_rate",
    "wrong_target_frames_iou_below_0_2",
    "reappearance_success_rate",
)
lines = [
    "# Frozen detector/ReID candidate-fusion comparison",
    "",
    "Evaluation split: `viz_val` only (`0005`, `0032`); locked test was not used.",
    "",
    "| metric | " + " | ".join(results) + " |",
    "|---|" + "---:|" * len(results),
]
for key in keys:
    values = []
    for mode in results.values():
        value = mode["aggregate"].get(key)
        values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
    lines.append(f"| {key} | " + " | ".join(values) + " |")
(evaluation_root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

records = []
for sequence in sequence_ids:
    with (evaluation_root / "records" / f"{sequence}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            predicted = record["predicted_bbox_xyxy"] is not None
            if record["gt_visible"] and record["iou"] >= 0.5:
                category = "correct"
            elif record["gt_visible"] and predicted and record["iou"] < 0.2:
                category = "wrong_target"
            elif record["gt_visible"]:
                category = "visible_miss"
            elif predicted:
                category = "absent_false_positive"
            else:
                category = "absent_correct"
            record["category"] = category
            records.append(record)

review_root = evaluation_root / "visual_review"
review_manifest = []
for category in (
    "correct",
    "wrong_target",
    "visible_miss",
    "absent_false_positive",
    "absent_correct",
):
    candidates = [record for record in records if record["category"] == category]
    if not candidates:
        continue
    count = min(16, len(candidates))
    chosen = [candidates[round(index * (len(candidates) - 1) / max(1, count - 1))] for index in range(count)]
    destination = review_root / category
    destination.mkdir(parents=True, exist_ok=True)
    for sample_index, record in enumerate(chosen, start=1):
        sequence = record["sequence_id"]
        video_index = int(record["video_index"])
        image_path = data_root / sequence / "rgb_frames" / f"frame_{video_index:06d}.jpg"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        gt = record["gt_bbox_xyxy"]
        pred = record["predicted_bbox_xyxy"]
        if gt is not None:
            p0 = tuple(round(value) for value in gt[:2])
            p1 = tuple(round(value) for value in gt[2:])
            cv2.rectangle(image, p0, p1, (0, 255, 255), 3)
        if pred is not None:
            p0 = tuple(round(value) for value in pred[:2])
            p1 = tuple(round(value) for value in pred[2:])
            cv2.rectangle(image, p0, p1, (0, 255, 0), 3)
        label = f"{category} seq={sequence} frame={video_index} IoU={record['iou']:.3f}"
        cv2.putText(image, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        filename = f"{sample_index:02d}_{sequence}_frame_{video_index:06d}.jpg"
        output_path = destination / filename
        if not cv2.imwrite(str(output_path), image):
            raise OSError(f"failed to write {output_path}")
        review_manifest.append(
            {
                "path": str(output_path),
                "category": category,
                "sequence_id": sequence,
                "video_index": video_index,
                "iou": record["iou"],
                "gt_visible": record["gt_visible"],
            }
        )
(review_root / "index.json").write_text(
    json.dumps(review_manifest, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(results["dualop_fusion"]["aggregate"], sort_keys=True))
PY

echo "[$(date -Is)] complete"
