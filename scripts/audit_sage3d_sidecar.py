#!/usr/bin/env python3
"""Audit a repaired SAGE3D sidecar against frozen independent detections."""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from omtrackvla.data.sage3d_sidecar import (
    episode_sidecar_path,
    safe_source_path,
    sha256_file,
    validate_episode_sidecar,
)
from scripts.audit_sage3d_bboxes import bbox_iou


DEFAULT_ROOT = Path("/data/nfs/share/OmTrackVLA/data/sage3d_extracted")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--detector-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strong-iou", type=float, default=0.50)
    parser.add_argument("--partial-iou", type=float, default=0.20)
    parser.add_argument("--minimum-median-iou", type=float, default=0.60)
    parser.add_argument("--minimum-strong-rate", type=float, default=0.75)
    parser.add_argument("--maximum-conflict-rate", type=float, default=0.05)
    parser.add_argument("--maximum-rejected-strong", type=int, default=0)
    parser.add_argument("--montage-cases", type=int, default=48)
    parser.add_argument("--write-admission", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_records(path: Path) -> list[dict[str, object]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("detector record must be an object")
                records.append(value)
    if not records:
        raise ValueError("detector records are empty")
    return records


def percentile(values: Sequence[float], quantile: float) -> float | None:
    return float(np.quantile(values, quantile)) if values else None


def audit_records(
    source_root: Path,
    sidecar_root: Path,
    detector_records: Sequence[dict[str, object]],
    strong_iou: float,
    partial_iou: float,
) -> list[dict[str, object]]:
    episode_cache: dict[str, dict[str, object]] = {}
    output = []
    for detector in detector_records:
        source_path = str(detector["source_path"])
        if source_path not in episode_cache:
            path = episode_sidecar_path(sidecar_root, source_path)
            episode_cache[source_path] = validate_episode_sidecar(
                load_json(path), source_path
            )
        episode = episode_cache[source_path]
        index = int(detector["step_index"])
        if index >= len(episode["steps"]):
            raise ValueError(f"sidecar step index out of range: {source_path}:{index}")
        label = episode["steps"][index]
        if int(label["step"]) != int(detector["step"]):
            raise ValueError(f"source/sidecar step mismatch: {source_path}:{index}")
        bbox = label.get("bbox_xyxy")
        projected_bbox = label.get("projected_bbox_xyxy")
        detections = detector.get("detections", [])
        best = None
        best_iou = None
        comparison_bbox = bbox if bbox is not None else projected_bbox
        if comparison_bbox is not None and detections:
            best = max(
                detections,
                key=lambda item: bbox_iou(comparison_bbox, item["bbox_xyxy"]),
            )
            best_iou = bbox_iou(comparison_bbox, best["bbox_xyxy"])
        if not label["visible"]:
            status = "rejected_by_geometry_or_depth"
        elif best_iou is None:
            status = "visible_no_person_detection"
        elif best_iou >= strong_iou:
            status = "strong_agreement"
        elif best_iou >= partial_iou:
            status = "partial_agreement"
        else:
            status = "detector_conflict"
        output.append(
            {
                "source_path": source_path,
                "step_index": index,
                "step": int(detector["step"]),
                "run": detector.get("run"),
                "mode": detector.get("mode"),
                "camera": detector.get("camera"),
                "sample_position_name": detector.get("sample_position_name"),
                "rgb_path": str(
                    safe_source_path(source_root, str(detector["rgb_path"])).relative_to(
                        source_root
                    )
                ).replace("\\", "/"),
                "visible": bool(label["visible"]),
                "visibility_reason": str(label["visibility_reason"]),
                "bbox_xyxy": bbox,
                "projected_bbox_xyxy": projected_bbox,
                "depth_evidence": label.get("depth_evidence"),
                "detections": detections,
                "best_detection": best,
                "best_iou": best_iou,
                "status": status,
            }
        )
    return output


def summarize(records: Sequence[dict[str, object]]) -> dict[str, object]:
    visible = [record for record in records if record["visible"]]
    visible_detected = [record for record in visible if record["best_iou"] is not None]
    ious = [float(record["best_iou"]) for record in visible_detected]
    rejected = [record for record in records if not record["visible"]]
    rejected_detected = [record for record in rejected if record["best_iou"] is not None]
    status_counts = Counter(str(record["status"]) for record in records)
    reason_counts = Counter(str(record["visibility_reason"]) for record in records)
    denominator = max(1, len(ious))
    return {
        "sampled_frames": len(records),
        "sampled_episodes": len({str(record["source_path"]) for record in records}),
        "visible_frames": len(visible),
        "rejected_frames": len(rejected),
        "visible_detected_frames": len(visible_detected),
        "rejected_detected_frames": len(rejected_detected),
        "rejected_strong_detector_frames": sum(
            float(record["best_iou"]) >= 0.50 for record in rejected_detected
        ),
        "median_best_iou": percentile(ious, 0.50),
        "strong_agreement_rate": sum(value >= 0.50 for value in ious) / denominator,
        "partial_agreement_rate": sum(0.20 <= value < 0.50 for value in ious)
        / denominator,
        "conflict_rate": sum(value < 0.20 for value in ious) / denominator,
        "status_counts": dict(sorted(status_counts.items())),
        "visibility_reason_counts": dict(sorted(reason_counts.items())),
    }


def integrity_checks(manifest: dict[str, object], sidecar_root: Path) -> dict[str, bool]:
    accepted = int(manifest.get("accepted_source_episode_count", -1))
    selected = int(manifest.get("selected_episode_count", -1))
    completed = int(manifest.get("completed_episode_count", -1))
    failed = int(manifest.get("failed_episode_count", -1))
    episodes = manifest.get("episodes")
    indexed = isinstance(episodes, dict) and len(episodes) == completed
    files_match = bool(indexed)
    if isinstance(episodes, dict):
        for source_path, metadata in episodes.items():
            path = episode_sidecar_path(sidecar_root, str(source_path))
            if (
                not isinstance(metadata, dict)
                or not path.is_file()
                or metadata.get("sidecar_sha256") != sha256_file(path)
            ):
                files_match = False
                break
    return {
        "full_selection": manifest.get("selection") == "full",
        "all_accepted_episodes_selected": selected == accepted and accepted > 0,
        "all_selected_episodes_completed": completed == selected and selected > 0,
        "no_generation_failures": failed == 0,
        "nonempty_steps": int(manifest.get("total_step_count", 0)) > 0,
        "episode_index_complete": indexed,
        "episode_files_match_manifest": files_match,
    }


def draw_case(source_root: Path, record: dict[str, object]) -> np.ndarray:
    path = safe_source_path(source_root, str(record["rgb_path"]))
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode {path}")
    height, width = image.shape[:2]
    scale = min(480.0 / width, 360.0 / height)
    canvas = cv2.resize(
        image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA
    )

    def box(value: Sequence[float], color: tuple[int, int, int], thickness: int) -> None:
        x0, y0, x1, y1 = (int(round(float(item) * scale)) for item in value)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, thickness)

    for detection in record["detections"]:
        box(detection["bbox_xyxy"], (255, 255, 0), 1)
    if record["best_detection"] is not None:
        box(record["best_detection"]["bbox_xyxy"], (0, 255, 0), 3)
    projected = record["projected_bbox_xyxy"]
    if projected is not None:
        box(projected, (0, 220, 255), 3)
    evidence = record.get("depth_evidence") or {}
    text = (
        f"{record['status']} | {record['visibility_reason']} | "
        f"IoU={record['best_iou']:.3f}" if record["best_iou"] is not None else
        f"{record['status']} | {record['visibility_reason']} | IoU=NA"
    )
    depth_text = (
        f"support={evidence.get('support_fraction', 0):.2f} "
        f"near={evidence.get('near_fraction', 0):.2f}"
        if evidence
        else "depth=not-evaluated"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 48), (0, 0, 0), -1)
    cv2.putText(canvas, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    cv2.putText(canvas, depth_text, (6, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return canvas


def write_montages(
    source_root: Path, output_root: Path, records: Sequence[dict[str, object]], maximum: int
) -> list[str]:
    ranked = sorted(
        records,
        key=lambda record: (
            0
            if not record["visible"] and record["best_iou"] is not None
            else 1
            if record["status"] == "detector_conflict"
            else 2
            if not record["visible"]
            else 3
            if record["best_iou"] is None
            else 4,
            float(record["best_iou"]) if record["best_iou"] is not None else 1.0,
            f"{record['source_path']}:{record['step']}",
        ),
    )[: max(0, int(maximum))]
    outputs = []
    for start in range(0, len(ranked), 12):
        images = [draw_case(source_root, record) for record in ranked[start : start + 12]]
        cell_height = max(image.shape[0] for image in images)
        cell_width = max(image.shape[1] for image in images)
        grid = np.zeros((cell_height * 4, cell_width * 3, 3), dtype=np.uint8)
        for index, image in enumerate(images):
            row, column = divmod(index, 3)
            grid[
                row * cell_height : row * cell_height + image.shape[0],
                column * cell_width : column * cell_width + image.shape[1],
            ] = image
        name = f"review_{start // 12:02d}.jpg"
        cv2.imwrite(str(output_root / name), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
        outputs.append(name)
    return outputs


def main() -> None:
    args = arguments()
    source_root = args.data_root.expanduser().resolve(strict=True)
    sidecar_root = args.sidecar_dir.expanduser().resolve(strict=True)
    output_root = args.output_dir.expanduser().resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = sidecar_root / "manifest.json"
    manifest = load_json(manifest_path)
    detector_records = load_records(args.detector_records)
    records = audit_records(
        source_root,
        sidecar_root,
        detector_records,
        args.strong_iou,
        args.partial_iou,
    )
    summary = summarize(records)
    integrity = integrity_checks(manifest, sidecar_root)
    thresholds = {
        "minimum_median_iou": float(args.minimum_median_iou),
        "minimum_strong_rate": float(args.minimum_strong_rate),
        "maximum_conflict_rate": float(args.maximum_conflict_rate),
        "maximum_rejected_strong": int(args.maximum_rejected_strong),
    }
    sample_checks = {
        "median_iou": float(summary["median_best_iou"] or 0.0)
        >= thresholds["minimum_median_iou"],
        "strong_rate": float(summary["strong_agreement_rate"])
        >= thresholds["minimum_strong_rate"],
        "conflict_rate": float(summary["conflict_rate"])
        <= thresholds["maximum_conflict_rate"],
        "rejected_strong": int(summary["rejected_strong_detector_frames"])
        <= thresholds["maximum_rejected_strong"],
    }
    passed = all(integrity.values()) and all(sample_checks.values())
    montage_files = write_montages(
        source_root, output_root, records, args.montage_cases
    )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "manifest_sha256": sha256_file(manifest_path),
        "detector_records_sha256": sha256_file(args.detector_records),
        "integrity_checks": integrity,
        "thresholds": thresholds,
        "sample_checks": sample_checks,
        "summary": summary,
        "montage_files": montage_files,
    }
    write_json_atomic(output_root / "report.json", report)
    with (output_root / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    markdown = [
        "# SAGE3D repaired sidecar audit",
        "",
        f"Status: **{report['status']}**",
        "",
        f"- Sampled frames/episodes: {summary['sampled_frames']} / {summary['sampled_episodes']}",
        f"- Visible/rejected frames: {summary['visible_frames']} / {summary['rejected_frames']}",
        f"- Median detector IoU: {summary['median_best_iou']:.4f}",
        f"- Strong/partial/conflict rates: {summary['strong_agreement_rate']:.4f} / "
        f"{summary['partial_agreement_rate']:.4f} / {summary['conflict_rate']:.4f}",
        f"- Rejected strong-detector frames: {summary['rejected_strong_detector_frames']}",
        "",
        "Yellow is the repaired geometric box, green the best independent person detection, "
        "and cyan other person detections. Detector boxes validate the projection but never "
        "replace target identity labels.",
        "",
    ]
    (output_root / "report.md").write_text("\n".join(markdown), encoding="utf-8")
    if args.write_admission:
        if not passed:
            raise ValueError("refusing to write admission.json because the audit failed")
        admission = {
            "schema_version": 1,
            "dataset_id": "sage3d_extracted",
            "status": "passed",
            "created_at_utc": report["created_at_utc"],
            "manifest_sha256": report["manifest_sha256"],
            "audit_report_sha256": sha256_file(output_root / "report.json"),
            "detector_records_sha256": report["detector_records_sha256"],
            "thresholds": thresholds,
            "sample_summary": summary,
        }
        write_json_atomic(sidecar_root / "admission.json", admission)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
