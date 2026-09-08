#!/usr/bin/env python3
"""Audit SAGE3D projected person boxes against an independent detector.

The script is intentionally read-only with respect to the source dataset.  It
uses the existing modular Faster R-CNN person-detector weights, compares every
sampled projected box with all detected people, and writes metrics plus review
montages to a separate output directory.  Detector disagreement is evidence
for manual review, not an automatic replacement for the source annotation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


DEFAULT_ROOT = Path("/data/nfs/share/OmTrackVLA/data/sage3d_extracted")
DEFAULT_WEIGHTS = Path(
    "/data/nfs/share/OmTrackVLA/models/torchvision/"
    "fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth"
)
def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detector-weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--samples", type=int, default=384)
    parser.add_argument("--frames-per-episode", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--score-threshold", type=float, default=0.30)
    parser.add_argument("--strong-iou", type=float, default=0.50)
    parser.add_argument("--partial-iou", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--montage-cases", type=int, default=72)
    parser.add_argument("--montage-columns", type=int, default=3)
    parser.add_argument("--montage-rows", type=int, default=4)
    return parser.parse_args()


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def safe_source_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve(strict=False)
    candidate.relative_to(root)
    return candidate


def bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx0, ly0, lx1, ly1 = map(float, left)
    rx0, ry0, rx1, ry1 = map(float, right)
    intersection = max(0.0, min(lx1, rx1) - max(lx0, rx0)) * max(
        0.0, min(ly1, ry1) - max(ly0, ry0)
    )
    left_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    right_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def clip_bbox(bbox: Sequence[float], width: int, height: int) -> tuple[list[float], bool]:
    if len(bbox) != 4:
        raise ValueError("bbox must have four coordinates")
    x0, y0, x1, y1 = map(float, bbox)
    x0, x1 = sorted((min(max(x0, 0.0), width), min(max(x1, 0.0), width)))
    y0, y1 = sorted((min(max(y0, 0.0), height), min(max(y1, 0.0), height)))
    clipped = [x0, y0, x1, y1]
    return clipped, bool(x1 > x0 and y1 > y0)


def evenly_spaced(values: Sequence[int], count: int) -> list[int]:
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return list(values)
    offsets = np.linspace(0, len(values) - 1, count).round().astype(np.int64)
    return [int(values[index]) for index in dict.fromkeys(offsets.tolist())]


def choose_entries(
    entries: Sequence[dict[str, object]], episode_count: int, seed: int
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Round-robin deterministic samples across every mode/camera stratum."""
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for entry in entries:
        if int(entry.get("steps", 0)) <= 0 or not entry.get("path"):
            continue
        groups[(str(entry.get("mode", "unknown")), str(entry.get("cam", "unknown")))].append(entry)
    for key, group in groups.items():
        group.sort(key=lambda item: stable_rank(seed, f"{key}:{item['path']}"))
    selected: list[dict[str, object]] = []
    positions = {key: 0 for key in groups}
    keys = sorted(groups)
    while len(selected) < episode_count:
        progressed = False
        for key in keys:
            position = positions[key]
            if position >= len(groups[key]):
                continue
            selected.append(groups[key][position])
            positions[key] += 1
            progressed = True
            if len(selected) >= episode_count:
                break
        if not progressed:
            break
    return selected, {f"{mode}/{camera}": len(group) for (mode, camera), group in sorted(groups.items())}


def build_samples(
    root: Path,
    entries: Sequence[dict[str, object]],
    sample_count: int,
    frames_per_episode: int,
) -> tuple[list[dict[str, object]], Counter[str]]:
    samples: list[dict[str, object]] = []
    skipped: Counter[str] = Counter()
    for entry in entries:
        if len(samples) >= sample_count:
            break
        try:
            episode = safe_source_path(root, str(entry["path"]))
            if not (episode / "_ACCEPTED").is_file():
                skipped["missing_accepted_marker"] += 1
                continue
            quality = load_json(episode / "quality.json")
            if not isinstance(quality, dict) or quality.get("status") != "accepted":
                skipped["quality_not_accepted"] += 1
                continue
            derived = load_json(episode / "derived.json")
            if not isinstance(derived, dict) or not isinstance(derived.get("steps"), list):
                skipped["invalid_derived"] += 1
                continue
            steps = derived["steps"]
            visible = [
                index
                for index, step in enumerate(steps)
                if isinstance(step, dict) and step.get("visible") and step.get("bbox")
            ]
            if not visible:
                skipped["no_visible_bbox"] += 1
                continue
            selected_visible = evenly_spaced(visible, frames_per_episode)
            visible_rank = {index: rank for rank, index in enumerate(visible)}
            for sample_position, index in enumerate(selected_visible):
                step = steps[index]
                rgb_path = episode / "rgb" / f"{int(step['step']):05d}.jpg"
                if not rgb_path.is_file():
                    skipped["missing_rgb"] += 1
                    continue
                samples.append(
                    {
                        "run": str(entry.get("run", "unknown")),
                        "mode": str(entry.get("mode", "unknown")),
                        "episode": str(entry.get("ep", "unknown")),
                        "camera": str(entry.get("cam", "unknown")),
                        "source_path": str(entry["path"]),
                        "step_index": int(index),
                        "step": int(step["step"]),
                        "sample_position_in_episode": int(sample_position),
                        "sample_position_name": (
                            "initial_visible"
                            if sample_position == 0
                            else "final_visible"
                            if sample_position == len(selected_visible) - 1
                            else "middle_visible"
                        ),
                        "visible_sequence_fraction": (
                            visible_rank[index] / max(1, len(visible) - 1)
                        ),
                        "rgb_path": str(rgb_path),
                        "source_bbox_xyxy": [float(value) for value in step["bbox"]],
                    }
                )
                if len(samples) >= sample_count:
                    break
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            skipped[f"invalid_episode:{type(error).__name__}"] += 1
    return samples, skipped


def load_detector(weights: Path, device: str):
    import torch
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn

    if not weights.is_file():
        raise FileNotFoundError(f"detector weights do not exist: {weights}")
    target_device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval().to(target_device)
    return model, target_device


def chunks(values: Sequence[dict[str, object]], size: int) -> Iterable[Sequence[dict[str, object]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def audit_samples(
    samples: list[dict[str, object]],
    detector,
    device,
    score_threshold: float,
    strong_iou: float,
    partial_iou: float,
    batch_size: int,
) -> list[dict[str, object]]:
    import torch
    from torchvision.transforms.functional import pil_to_tensor
    from PIL import Image

    records: list[dict[str, object]] = []
    completed = 0
    for batch in chunks(samples, batch_size):
        tensors = []
        sizes = []
        for sample in batch:
            with Image.open(str(sample["rgb_path"])) as image:
                rgb = image.convert("RGB")
                sizes.append(rgb.size)
                tensors.append(pil_to_tensor(rgb).to(device=device, dtype=torch.float32).div_(255.0))
        with torch.inference_mode():
            outputs = detector(tensors)
        for sample, size, output in zip(batch, sizes, outputs):
            width, height = map(int, size)
            raw_gt = [float(value) for value in sample["source_bbox_xyxy"]]
            gt, valid_gt = clip_bbox(raw_gt, width, height)
            mirrored_gt = [width - gt[2], gt[1], width - gt[0], gt[3]]
            detections = []
            for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
                confidence = float(score.detach().cpu())
                if confidence < score_threshold:
                    continue
                if int(label.detach().cpu()) != 1:
                    continue
                detections.append(
                    {
                        "bbox_xyxy": [float(value) for value in box.detach().cpu().tolist()],
                        "score": confidence,
                    }
                )
            for detection in detections:
                detection["iou_with_source"] = (
                    bbox_iou(gt, detection["bbox_xyxy"]) if valid_gt else None
                )
            detections.sort(key=lambda item: float(item["score"]), reverse=True)
            best = (
                max(detections, key=lambda item: float(item["iou_with_source"]), default=None)
                if valid_gt
                else (detections[0] if detections else None)
            )
            best_iou = (
                float(best["iou_with_source"])
                if best is not None and best["iou_with_source"] is not None
                else None
            )
            best_mirrored = (
                max(
                    detections,
                    key=lambda item: bbox_iou(mirrored_gt, item["bbox_xyxy"]),
                    default=None,
                )
                if valid_gt
                else None
            )
            best_mirrored_iou = (
                bbox_iou(mirrored_gt, best_mirrored["bbox_xyxy"])
                if best_mirrored is not None
                else None
            )
            if not valid_gt:
                status = "invalid_source_bbox"
            elif best_iou is None:
                status = "no_person_detection"
            elif best_iou >= strong_iou:
                status = "strong_agreement"
            elif best_iou >= partial_iou:
                status = "partial_agreement"
            else:
                status = "detector_disagreement"
            diagonal = math.hypot(width, height)
            center_error = None
            area_ratio = None
            if best is not None and valid_gt:
                bx0, by0, bx1, by1 = map(float, best["bbox_xyxy"])
                gx0, gy0, gx1, gy1 = gt
                center_error = math.hypot((bx0 + bx1 - gx0 - gx1) / 2.0, (by0 + by1 - gy0 - gy1) / 2.0) / diagonal
                gt_area = max(1.0, (gx1 - gx0) * (gy1 - gy0))
                area_ratio = ((bx1 - bx0) * (by1 - by0)) / gt_area
            record = dict(sample)
            record.pop("source_bbox_xyxy", None)
            record.update(
                {
                    "image_width": width,
                    "image_height": height,
                    "raw_source_bbox_xyxy": raw_gt,
                    "gt_bbox_xyxy": gt,
                    "horizontal_mirror_bbox_xyxy": mirrored_gt if valid_gt else None,
                    "source_bbox_valid_after_clipping": valid_gt,
                    "gt_area_fraction": ((gt[2] - gt[0]) * (gt[3] - gt[1])) / (width * height),
                    "gt_touches_border": bool(
                        gt[0] <= 0.5 or gt[1] <= 0.5 or gt[2] >= width - 0.5 or gt[3] >= height - 0.5
                    ),
                    "person_detection_count": len(detections),
                    "detections": detections,
                    "best_detection": best,
                    "best_iou": best_iou,
                    "best_horizontal_mirror_detection": best_mirrored,
                    "best_horizontal_mirror_iou": best_mirrored_iou,
                    "center_error_diagonal": center_error,
                    "detector_to_gt_area_ratio": area_ratio,
                    "status": status,
                }
            )
            records.append(record)
        completed += len(batch)
        print(f"audited {completed}/{len(samples)} frames", flush=True)
    return records


def percentile(values: Sequence[float], quantile: float) -> float | None:
    return float(np.quantile(values, quantile)) if values else None


def summarize(
    records: Sequence[dict[str, object]], strong_iou: float, partial_iou: float
) -> dict[str, object]:
    statuses = Counter(str(record["status"]) for record in records)
    valid_records = [record for record in records if record["source_bbox_valid_after_clipping"]]
    ious = [float(record["best_iou"]) for record in records if record["best_iou"] is not None]
    detected = max(1, len(ious))
    mirror_pairs = [
        (float(record["best_iou"]), float(record["best_horizontal_mirror_iou"]))
        for record in records
        if record["best_iou"] is not None and record["best_horizontal_mirror_iou"] is not None
    ]

    def group_summary(group: Sequence[dict[str, object]]) -> dict[str, object]:
        valid_group = [record for record in group if record["source_bbox_valid_after_clipping"]]
        group_ious = [float(record["best_iou"]) for record in valid_group if record["best_iou"] is not None]
        group_status = Counter(str(record["status"]) for record in group)
        return {
            "frames": len(group),
            "invalid_source_bbox_frames": len(group) - len(valid_group),
            "detector_coverage_among_valid_source_boxes": len(group_ious) / max(1, len(valid_group)),
            "strong_agreement_rate": group_status["strong_agreement"] / max(1, len(group_ious)),
            "disagreement_rate": group_status["detector_disagreement"] / max(1, len(group_ious)),
            "median_best_iou": percentile(group_ious, 0.5),
            "status_counts": dict(sorted(group_status.items())),
        }

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    temporal: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[f"{record['mode']}/{record['camera']}"] .append(record)
        temporal[str(record["sample_position_name"])].append(record)
    return {
        "sampled_frames": len(records),
        "sampled_episodes": len({str(record["source_path"]) for record in records}),
        "valid_source_bbox_frames": len(valid_records),
        "invalid_source_bbox_frames": len(records) - len(valid_records),
        "detector_coverage_among_valid_source_boxes": len(ious) / max(1, len(valid_records)),
        "strong_agreement_rate_among_detected": statuses["strong_agreement"] / detected,
        "partial_agreement_rate_among_detected": statuses["partial_agreement"] / detected,
        "disagreement_rate_among_detected": statuses["detector_disagreement"] / detected,
        "best_iou": {
            "p10": percentile(ious, 0.10),
            "median": percentile(ious, 0.50),
            "p90": percentile(ious, 0.90),
        },
        "status_counts": dict(sorted(statuses.items())),
        "horizontal_mirror_counterfactual": {
            "detected_valid_frames": len(mirror_pairs),
            "original_median_best_iou": percentile([pair[0] for pair in mirror_pairs], 0.5),
            "mirrored_median_best_iou": percentile([pair[1] for pair in mirror_pairs], 0.5),
            "original_strong_agreement_frames": sum(pair[0] >= strong_iou for pair in mirror_pairs),
            "mirrored_strong_agreement_frames": sum(pair[1] >= strong_iou for pair in mirror_pairs),
            "original_disagreement_frames": sum(pair[0] < partial_iou for pair in mirror_pairs),
            "mirrored_disagreement_frames": sum(pair[1] < partial_iou for pair in mirror_pairs),
        },
        "by_mode_camera": {key: group_summary(grouped[key]) for key in sorted(grouped)},
        "by_temporal_position": {key: group_summary(temporal[key]) for key in sorted(temporal)},
    }


def record_rank(record: dict[str, object]) -> tuple[float, float, str]:
    if record["status"] == "invalid_source_bbox":
        priority = 0.0
    elif record["status"] == "detector_disagreement":
        priority = 1.0
    elif record["status"] == "partial_agreement":
        priority = 2.0
    elif record["status"] == "no_person_detection":
        priority = 3.0
    else:
        priority = 4.0
    iou = float(record["best_iou"]) if record["best_iou"] is not None else 1.0
    return priority, iou, f"{record['source_path']}:{record['step']}"


def draw_case(record: dict[str, object], cell_width: int = 480, image_height: int = 360) -> np.ndarray:
    image = cv2.imread(str(record["rgb_path"]), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode {record['rgb_path']}")
    height, width = image.shape[:2]
    scale_x, scale_y = cell_width / width, image_height / height
    resized = cv2.resize(image, (cell_width, image_height), interpolation=cv2.INTER_AREA)

    def scaled(box: Sequence[float]) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = map(float, box)
        return tuple(map(int, (round(x0 * scale_x), round(y0 * scale_y), round(x1 * scale_x), round(y1 * scale_y))))

    for detection in record["detections"]:
        x0, y0, x1, y1 = scaled(detection["bbox_xyxy"])
        cv2.rectangle(resized, (x0, y0), (x1, y1), (255, 255, 0), 1)
    best = record["best_detection"]
    if best is not None:
        x0, y0, x1, y1 = scaled(best["bbox_xyxy"])
        cv2.rectangle(resized, (x0, y0), (x1, y1), (0, 255, 0), 3)
    x0, y0, x1, y1 = scaled(record["gt_bbox_xyxy"])
    cv2.rectangle(resized, (x0, y0), (x1, y1), (0, 220, 255), 3)
    if record["horizontal_mirror_bbox_xyxy"] is not None:
        x0, y0, x1, y1 = scaled(record["horizontal_mirror_bbox_xyxy"])
        cv2.rectangle(resized, (x0, y0), (x1, y1), (255, 0, 255), 2)

    cell = np.zeros((image_height + 86, cell_width, 3), dtype=np.uint8)
    cell[:image_height] = resized
    identity = f"{record['run']}/{record['mode']}/{record['episode']}/{record['camera']} step={record['step']}"
    iou_text = "none" if record["best_iou"] is None else f"{float(record['best_iou']):.3f}"
    details = f"{record['status']}  best IoU={iou_text}  people={record['person_detection_count']}"
    legend = "yellow=SAGE GT  magenta=h-mirror  green=best detector"
    for y, text, color in (
        (image_height + 22, identity, (255, 255, 255)),
        (image_height + 48, details, (255, 255, 255)),
        (image_height + 73, legend, (180, 180, 180)),
    ):
        cv2.putText(cell, text[:78], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 1, cv2.LINE_AA)
    return cell


def write_montages(
    records: Sequence[dict[str, object]], output_dir: Path, limit: int, columns: int, rows: int
) -> list[str]:
    selected = sorted(records, key=record_rank)[:limit]
    page_size = columns * rows
    paths = []
    montage_dir = output_dir / "review_montages"
    montage_dir.mkdir(parents=True, exist_ok=True)
    for page_index, start in enumerate(range(0, len(selected), page_size), start=1):
        page_records = selected[start : start + page_size]
        cells = [draw_case(record) for record in page_records]
        blank = np.zeros_like(cells[0])
        cells.extend([blank] * (page_size - len(cells)))
        bands = [np.concatenate(cells[row * columns : (row + 1) * columns], axis=1) for row in range(rows)]
        page = np.concatenate(bands, axis=0)
        path = montage_dir / f"worst_cases_{page_index:02d}.jpg"
        if not cv2.imwrite(str(path), page, [cv2.IMWRITE_JPEG_QUALITY, 91]):
            raise RuntimeError(f"could not write montage: {path}")
        paths.append(path.relative_to(output_dir).as_posix())
    return paths


def markdown_report(config: dict[str, object], summary: dict[str, object], montages: Sequence[str]) -> str:
    lines = [
        "# SAGE3D bbox audit",
        "",
        "This is an independent-detector audit of projected SAGE3D person boxes. Detector disagreement is a review signal, not replacement ground truth.",
        "",
        "## Configuration",
        "",
        f"- Frames: {summary['sampled_frames']} from {summary['sampled_episodes']} episodes",
        f"- Detector: torchvision Faster R-CNN MobileNet V3 320 FPN",
        f"- Person score threshold: {config['score_threshold']}",
        f"- Strong/partial IoU thresholds: {config['strong_iou']} / {config['partial_iou']}",
        f"- Seed: {config['seed']}",
        "",
        "## Summary",
        "",
        f"- Invalid source boxes after image clipping: {summary['invalid_source_bbox_frames']}",
        f"- Detector coverage among valid source boxes: {float(summary['detector_coverage_among_valid_source_boxes']):.3f}",
        f"- Strong agreement among detected frames: {float(summary['strong_agreement_rate_among_detected']):.3f}",
        f"- Partial agreement among detected frames: {float(summary['partial_agreement_rate_among_detected']):.3f}",
        f"- Detector disagreement among detected frames: {float(summary['disagreement_rate_among_detected']):.3f}",
        f"- Median best IoU: {summary['best_iou']['median']}",
        f"- Status counts: `{json.dumps(summary['status_counts'], sort_keys=True)}`",
        f"- Horizontal-mirror counterfactual: `{json.dumps(summary['horizontal_mirror_counterfactual'], sort_keys=True)}`",
        "",
        "## Mode/camera strata",
        "",
        "| Stratum | Frames | Invalid GT | Coverage | Strong | Disagree | Median IoU |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, value in summary["by_mode_camera"].items():
        lines.append(
            f"| {key} | {value['frames']} | {value['invalid_source_bbox_frames']} | "
            f"{value['detector_coverage_among_valid_source_boxes']:.3f} | "
            f"{value['strong_agreement_rate']:.3f} | {value['disagreement_rate']:.3f} | "
            f"{value['median_best_iou']} |"
        )
    lines.extend(
        [
            "",
            "## Temporal position",
            "",
            "| Position | Frames | Invalid GT | Coverage | Strong | Disagree | Median IoU |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, value in summary["by_temporal_position"].items():
        lines.append(
            f"| {key} | {value['frames']} | {value['invalid_source_bbox_frames']} | "
            f"{value['detector_coverage_among_valid_source_boxes']:.3f} | "
            f"{value['strong_agreement_rate']:.3f} | {value['disagreement_rate']:.3f} | "
            f"{value['median_best_iou']} |"
        )
    lines.extend(["", "## Human review", ""])
    lines.extend(f"- `{path}`" for path in montages)
    lines.extend(
        [
            "",
            "Review yellow boxes against the target person. Magenta is the horizontal-mirror counterfactual used to test the extractor sign hypothesis. Green is the detected person with the highest IoU to yellow; cyan boxes are other detected people. A missing detection cannot establish that the source GT is wrong.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = arguments()
    if args.samples <= 0 or args.frames_per_episode <= 0 or args.batch_size <= 0:
        raise ValueError("samples, frames-per-episode, and batch-size must be positive")
    if not 0.0 <= args.partial_iou <= args.strong_iou <= 1.0:
        raise ValueError("expected 0 <= partial-iou <= strong-iou <= 1")
    root = args.data_root.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve(strict=False)
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("output directory must not be inside the source dataset")
    output.mkdir(parents=True, exist_ok=True)

    index = load_json(root / "index.json")
    if not isinstance(index, dict) or not isinstance(index.get("eps"), list):
        raise ValueError("SAGE3D index.json must contain an eps list")
    requested_episodes = math.ceil(args.samples / args.frames_per_episode)
    # Select extra descriptors so rejected/malformed episodes do not reduce the frame target.
    selected_entries, strata_population = choose_entries(
        index["eps"], min(len(index["eps"]), requested_episodes * 2), args.seed
    )
    samples, skipped = build_samples(root, selected_entries, args.samples, args.frames_per_episode)
    if not samples:
        raise RuntimeError("sampling produced no valid SAGE3D frames")
    detector, device = load_detector(args.detector_weights.expanduser(), args.device)
    records = audit_samples(
        samples,
        detector,
        device,
        args.score_threshold,
        args.strong_iou,
        args.partial_iou,
        args.batch_size,
    )
    summary = summarize(records, args.strong_iou, args.partial_iou)
    montages = write_montages(
        records, output, args.montage_cases, args.montage_columns, args.montage_rows
    )
    config = {
        "data_root": str(root),
        "detector_weights": str(args.detector_weights.expanduser()),
        "samples_requested": args.samples,
        "frames_per_episode": args.frames_per_episode,
        "seed": args.seed,
        "score_threshold": args.score_threshold,
        "strong_iou": args.strong_iou,
        "partial_iou": args.partial_iou,
        "batch_size": args.batch_size,
        "device": str(device),
        "source_writes": False,
    }
    report = {
        "schema_version": 1,
        "config": config,
        "strata_population_from_index": strata_population,
        "skipped_entries": dict(sorted(skipped.items())),
        "summary": summary,
        "review_montages": list(montages),
        "interpretation": {
            "strong_agreement": "best independent person detection IoU is at least the strong threshold",
            "partial_agreement": "best independent person detection IoU is between the partial and strong thresholds",
            "detector_disagreement": "a person was detected, but none overlaps the projected GT by the partial threshold",
            "no_person_detection": "the independent detector found no person; this is inconclusive",
            "invalid_source_bbox": "the projected source box has zero area after clipping to the RGB image",
        },
    }
    write_json(output / "report.json", report)
    with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            serializable = dict(record)
            serializable["rgb_path"] = str(Path(str(record["rgb_path"])).relative_to(root))
            handle.write(json.dumps(serializable, sort_keys=True) + "\n")
    (output / "report.md").write_text(
        markdown_report(config, summary, montages), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote audit to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
