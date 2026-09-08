#!/usr/bin/env python3
"""Calibrate and audit DA3 pseudo ego-motion on disjoint Intern scenes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import cv2
import numpy as np
import pyarrow.parquet as pq

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from omtrackvla.geometry.da3_admission import (
    DEFAULT_QUALITY_LIMITS,
    apply_confidence_gate,
    camera_scale_stratum,
    choose_confidence_threshold,
    clip_meets_quality,
    fit_stratified_translation_scales,
    score_clip,
    summarize_admission,
)
from omtrackvla.geometry.se2 import (
    intern_pair_to_canonical_se2,
    relative_w2c_to_base_se2,
)


PINNED_SOURCE_COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
DEFAULT_ROOT = Path("/h100-2/vln_n1/traj_data")
DEFAULT_POLICY = REPOSITORY_ROOT / "configs" / "gates" / "da3_multiscene_v1.json"


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--source-commit", default=PINNED_SOURCE_COMMIT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/manifests/phase1_v1.json"),
    )
    parser.add_argument("--intern-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--calibration-split", default="val")
    parser.add_argument("--evaluation-split", default="viz_val")
    parser.add_argument("--clips-per-group", type=int, default=3)
    parser.add_argument("--max-units-per-group", type=int, default=16)
    parser.add_argument("--max-episodes-per-unit", type=int, default=12)
    parser.add_argument("--max-starts-per-episode", type=int, default=16)
    parser.add_argument("--offsets", type=int, nargs="+", default=[0, 4, 8, 12])
    parser.add_argument("--selection-minimum-motion-m", type=float, default=0.20)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-selected-clips", type=int, default=20)
    parser.add_argument("--minimum-scale-clips", type=int, default=8)
    parser.add_argument("--maximum-scale-relative-iqr", type=float, default=0.50)
    parser.add_argument("--minimum-confidence-coverage", type=float, default=0.50)
    parser.add_argument("--maximum-calibration-bad-rate", type=float, default=0.20)
    parser.add_argument("--minimum-evaluation-coverage", type=float, default=0.50)
    parser.add_argument("--maximum-evaluation-bad-rate", type=float, default=0.20)
    parser.add_argument("--maximum-translation-median-m", type=float, default=0.12)
    parser.add_argument("--maximum-translation-p90-m", type=float, default=0.25)
    parser.add_argument("--maximum-yaw-median-rad", type=float, default=0.12)
    parser.add_argument("--maximum-yaw-p90-rad", type=float, default=0.25)
    parser.add_argument("--maximum-scale-error-median", type=float, default=0.35)
    parser.add_argument("--maximum-scale-error-p90", type=float, default=0.75)
    parser.add_argument("--write-admission", action="store_true")
    parser.add_argument("--skip-visualizations", action="store_true")
    return parser.parse_args(argv)


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_policy(args: argparse.Namespace, policy: Mapping[str, object]) -> str:
    """Fail before inference when CLI values diverge from the frozen policy."""

    if policy.get("schema_version") != 1:
        raise ValueError("DA3 multi-scene policy schema_version must be 1")
    policy_id = policy.get("policy_id")
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError("DA3 multi-scene policy_id must be a non-empty string")

    def nested(path: str) -> object:
        value: object = policy
        for part in path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                raise ValueError(f"DA3 multi-scene policy is missing {path}")
            value = value[part]
        return value

    expected_evaluation_split = nested(
        "selection.admission_evaluation_split"
        if args.write_admission
        else "selection.development_evaluation_split"
    )
    checks = {
        "source_commit": args.source_commit,
        "model_revision": args.model_revision,
        "selection.calibration_split": args.calibration_split,
        "selection.clips_per_group": args.clips_per_group,
        "selection.maximum_units_per_group": args.max_units_per_group,
        "selection.maximum_episodes_per_unit": args.max_episodes_per_unit,
        "selection.maximum_starts_per_episode": args.max_starts_per_episode,
        "selection.offsets": sorted(set(int(value) for value in args.offsets)),
        "selection.minimum_motion_m": args.selection_minimum_motion_m,
        "selection.seed": args.seed,
        "inference.process_res": args.process_res,
        "scale.minimum_calibration_clips_per_stratum": args.minimum_scale_clips,
        "scale.maximum_relative_iqr": args.maximum_scale_relative_iqr,
        "confidence.minimum_calibration_coverage": args.minimum_confidence_coverage,
        "confidence.maximum_calibration_bad_rate": args.maximum_calibration_bad_rate,
        "admission_gate.minimum_selected_clips": args.minimum_selected_clips,
        "admission_gate.minimum_evaluation_coverage": args.minimum_evaluation_coverage,
        "admission_gate.maximum_evaluation_bad_rate": args.maximum_evaluation_bad_rate,
        "admission_gate.maximum_translation_median_m": args.maximum_translation_median_m,
        "admission_gate.maximum_translation_p90_m": args.maximum_translation_p90_m,
        "admission_gate.maximum_yaw_median_rad": args.maximum_yaw_median_rad,
        "admission_gate.maximum_yaw_p90_rad": args.maximum_yaw_p90_rad,
        "admission_gate.maximum_scale_error_median": args.maximum_scale_error_median,
        "admission_gate.maximum_scale_error_p90": args.maximum_scale_error_p90,
    }
    for path, actual in checks.items():
        expected = nested(path)
        if actual != expected:
            raise ValueError(
                f"CLI value for {path} diverges from policy: "
                f"{actual!r} != {expected!r}"
            )
    if args.evaluation_split != expected_evaluation_split:
        raise ValueError(
            "evaluation split diverges from its frozen policy purpose: "
            f"{args.evaluation_split!r} != {expected_evaluation_split!r}"
        )
    fixed_values = {
        "inference.reference_view_strategy": "middle",
        "scale.mode": "intern_camera_family",
        "scale.strata": ["d435i", "zed"],
        "confidence.field": "median_depth_confidence",
        "quality_limits": DEFAULT_QUALITY_LIMITS,
        "admission_gate.require_no_inference_failures": True,
    }
    for path, actual in fixed_values.items():
        expected = nested(path)
        if actual != expected:
            raise ValueError(
                f"implementation value for {path} diverges from policy: "
                f"{actual!r} != {expected!r}"
            )
    if nested("source_commit") != PINNED_SOURCE_COMMIT:
        raise ValueError("policy does not pin the supported DA3 source commit")
    return policy_id


def safe_relative(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe source-relative path: {relative}")
    candidate = (root / Path(*pure.parts)).resolve(strict=False)
    try:
        candidate.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"path escapes Intern root: {relative}") from error
    return candidate


def matrix4(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.size != 16:
        raise ValueError("camera transform must contain 16 values")
    return matrix.reshape(4, 4)


def stable_order(values: Sequence[object], key_prefix: str) -> list[object]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"{key_prefix}:{value}".encode("utf-8")
        ).hexdigest(),
    )


def clip_ground_truth(
    table: Mapping[str, Sequence[object]], indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    base_from_camera = matrix4(table["observation.camera_extrinsic"][indices[0]])
    anchor = table["action"][indices[0]]
    trajectory = np.stack(
        [
            intern_pair_to_canonical_se2(
                anchor, table["action"][index], base_from_camera
            )
            for index in indices
        ]
    )
    return base_from_camera, trajectory


def candidate_from_parquet(
    root: Path,
    unit: str,
    parquet: Path,
    offsets: Sequence[int],
    seed_key: str,
    minimum_motion_m: float,
    maximum_starts: int,
) -> dict[str, object] | None:
    table = pq.read_table(
        parquet,
        columns=["observation.camera_extrinsic", "action"],
    ).to_pydict()
    count = len(table["action"])
    last_offset = max(offsets)
    if count <= last_offset:
        return None
    starts = stable_order(list(range(count - last_offset)), seed_key)
    scene = safe_relative(root, unit)
    chunk = parquet.parent.name
    stem = parquet.stem
    rgb_dir = scene / "videos" / chunk / "observation.images.rgb"
    for start_value in starts[: max(1, int(maximum_starts))]:
        start = int(start_value)
        indices = [start + int(offset) for offset in offsets]
        base_from_camera, ground_truth = clip_ground_truth(table, indices)
        motion = np.linalg.norm(ground_truth[1:, :2], axis=1)
        if float(np.max(motion)) < float(minimum_motion_m):
            continue
        images = [rgb_dir / f"{stem}_{index:03d}.jpg" for index in indices]
        if not all(path.is_file() for path in images):
            continue
        relative_parquet = parquet.relative_to(root).as_posix()
        clip_id = hashlib.sha256(
            f"{relative_parquet}:{','.join(map(str, indices))}".encode("utf-8")
        ).hexdigest()[:16]
        return {
            "clip_id": clip_id,
            "group": unit.split("/", 1)[0],
            "scale_stratum": camera_scale_stratum(unit.split("/", 1)[0]),
            "unit": unit,
            "parquet_path": relative_parquet,
            "parquet_sha256": sha256_file(parquet),
            "indices": indices,
            "image_paths": [path.relative_to(root).as_posix() for path in images],
            "ground_truth_se2": ground_truth.tolist(),
            "base_from_camera": base_from_camera.tolist(),
        }
    return None


def select_split_clips(
    root: Path,
    units: Sequence[str],
    split: str,
    *,
    clips_per_group: int,
    maximum_units_per_group: int,
    maximum_episodes_per_unit: int,
    maximum_starts_per_episode: int,
    offsets: Sequence[int],
    minimum_motion_m: float,
    seed: int,
) -> list[dict[str, object]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for unit in units:
        grouped[str(unit).split("/", 1)[0]].append(str(unit))
    selected = []
    for group, group_units in sorted(grouped.items()):
        ordered_units = stable_order(group_units, f"{seed}:{split}:{group}:unit")
        for unit_value in ordered_units[: max(1, int(maximum_units_per_group))]:
            unit = str(unit_value)
            scene = safe_relative(root, unit)
            parquets = list((scene / "data").glob("chunk-*/episode_*.parquet"))
            ordered_parquets = stable_order(
                parquets, f"{seed}:{split}:{unit}:episode"
            )
            clip = None
            for parquet_value in ordered_parquets[
                : max(1, int(maximum_episodes_per_unit))
            ]:
                parquet = Path(parquet_value)
                clip = candidate_from_parquet(
                    root,
                    unit,
                    parquet,
                    offsets,
                    f"{seed}:{split}:{parquet.relative_to(root).as_posix()}:start",
                    minimum_motion_m,
                    maximum_starts_per_episode,
                )
                if clip is not None:
                    break
            if clip is not None:
                clip["source_split"] = split
                selected.append(clip)
            if sum(item["group"] == group for item in selected) >= clips_per_group:
                break
    return selected


def distribution_summary(value: object) -> dict[str, float]:
    array = np.asarray(value, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def prediction_extrinsics(prediction: object, expected: int) -> np.ndarray:
    extrinsics = np.asarray(prediction.extrinsics, dtype=np.float64)
    if extrinsics.shape[0] != expected or extrinsics.shape[-2:] not in (
        (3, 4),
        (4, 4),
    ):
        raise ValueError(f"unexpected DA3 extrinsics shape: {extrinsics.shape}")
    if extrinsics.shape[-2:] == (3, 4):
        padded = np.broadcast_to(np.eye(4), (expected, 4, 4)).copy()
        padded[:, :3] = extrinsics
        extrinsics = padded
    return extrinsics


def colorize_map(value: np.ndarray, colormap: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    finite = np.isfinite(array)
    if not np.any(finite):
        return np.zeros((*array.shape, 3), dtype=np.uint8)
    lower, upper = np.quantile(array[finite], [0.05, 0.95])
    if upper <= lower:
        normalized = np.zeros(array.shape, dtype=np.uint8)
    else:
        normalized = np.clip((array - lower) / (upper - lower), 0.0, 1.0)
        normalized = np.nan_to_num(normalized, nan=0.0)
        normalized = np.asarray(np.round(normalized * 255.0), dtype=np.uint8)
    return cv2.applyColorMap(normalized, colormap)


def diagnostic_panel(
    rgb_path: Path,
    depth: np.ndarray,
    confidence: np.ndarray,
    title: str,
) -> np.ndarray:
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise ValueError(f"could not decode {rgb_path}")
    height, width = depth.shape
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    depth_color = colorize_map(depth, cv2.COLORMAP_TURBO)
    confidence_color = colorize_map(confidence, cv2.COLORMAP_VIRIDIS)
    body = np.concatenate([rgb, depth_color, confidence_color], axis=1)
    header = np.zeros((54, body.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        title[:150],
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        "RGB | DA3 relative depth | DA3 depth confidence (each normalized p05-p95)",
        (8, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate([header, body], axis=0)


def run_clip(
    model: object,
    torch_module: object,
    root: Path,
    clip: Mapping[str, object],
    process_res: int,
) -> tuple[dict[str, object], np.ndarray]:
    images = [safe_relative(root, str(path)) for path in clip["image_paths"]]
    start = time.perf_counter()
    with torch_module.inference_mode():
        prediction = model.inference(
            [str(path) for path in images],
            process_res=int(process_res),
            export_feat_layers=[],
            ref_view_strategy="middle",
        )
    runtime = time.perf_counter() - start
    required = ("depth", "conf", "extrinsics", "intrinsics", "processed_images")
    absent = [name for name in required if getattr(prediction, name, None) is None]
    if absent:
        raise ValueError(f"DA3 prediction is missing fields: {absent}")
    extrinsics = prediction_extrinsics(prediction, len(images))
    arrays = {
        "depth": np.asarray(prediction.depth),
        "confidence": np.asarray(prediction.conf),
        "extrinsics": extrinsics,
        "intrinsics": np.asarray(prediction.intrinsics),
        "processed_images": np.asarray(prediction.processed_images),
    }
    finite = {name: bool(np.isfinite(value).all()) for name, value in arrays.items()}
    if not all(finite.values()):
        raise ValueError(f"DA3 prediction contains non-finite output: {finite}")
    base_from_camera = matrix4(clip["base_from_camera"])
    predicted = np.stack(
        [
            relative_w2c_to_base_se2(extrinsics[0], pose, base_from_camera)
            for pose in extrinsics
        ]
    )
    confidence_score = float(np.median(arrays["confidence"]))
    output = dict(clip)
    output.pop("base_from_camera", None)
    output.update(
        {
            "predicted_se2_unscaled": predicted.tolist(),
            "confidence_score": confidence_score,
            "confidence_summary": distribution_summary(arrays["confidence"]),
            "depth_summary": distribution_summary(arrays["depth"]),
            "output_shapes": {
                name: list(value.shape) for name, value in arrays.items()
            },
            "finite": finite,
            "runtime_seconds": runtime,
        }
    )
    panel = diagnostic_panel(
        images[-1],
        arrays["depth"][-1],
        arrays["confidence"][-1],
        f"{clip['source_split']} | {clip['group']} | {clip['unit']} | conf={confidence_score:.3f}",
    )
    return output, panel


def trajectory_canvas(record: Mapping[str, object]) -> np.ndarray:
    canvas = np.full((420, 520, 3), 245, dtype=np.uint8)
    predicted = np.asarray(record["predicted_se2"], dtype=np.float64)
    reference = np.asarray(record["ground_truth_se2"], dtype=np.float64)
    extent = max(0.25, float(np.max(np.abs(np.concatenate([predicted[:, :2], reference[:, :2]])))))
    scale = 155.0 / extent
    origin = np.asarray([260.0, 225.0])

    def pixels(points: np.ndarray) -> np.ndarray:
        # Canonical x-forward is up; y-left is image-left.
        return np.asarray(
            [origin + [-point[1] * scale, -point[0] * scale] for point in points],
            dtype=np.int32,
        )

    predicted_px = pixels(predicted[:, :2])
    reference_px = pixels(reference[:, :2])
    cv2.arrowedLine(canvas, tuple(origin.astype(int)), (260, 48), (80, 80, 80), 1, tipLength=0.08)
    cv2.putText(canvas, "x forward", (272, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 50, 50), 1)
    cv2.arrowedLine(canvas, tuple(origin.astype(int)), (75, 225), (80, 80, 80), 1, tipLength=0.08)
    cv2.putText(canvas, "y left", (82, 216), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 50, 50), 1)
    cv2.polylines(canvas, [reference_px], False, (0, 210, 255), 4, cv2.LINE_AA)
    cv2.polylines(canvas, [predicted_px], False, (0, 180, 0), 3, cv2.LINE_AA)
    for point in reference_px:
        cv2.circle(canvas, tuple(point), 5, (0, 210, 255), -1)
    for point in predicted_px:
        cv2.circle(canvas, tuple(point), 4, (0, 180, 0), -1)
    accepted = bool(record.get("admitted"))
    good = clip_meets_quality(record)
    header_color = (0, 110, 0) if accepted and good else (0, 0, 190)
    cv2.putText(
        canvas,
        f"{record['group']} | {'ADMIT' if accepted else 'REJECT'} | {'quality-ok' if good else 'quality-bad'}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        header_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"conf={float(record['confidence_score']):.3f}  trans={float(record['translation_error_m_mean']):.3f}m  yaw={float(record['yaw_error_rad_mean']):.3f}rad",
        (8, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(canvas, "GT", (8, 394), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 170, 220), 2)
    cv2.putText(canvas, "PRED", (55, 394), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 150, 0), 2)
    return canvas


def write_montage(path: Path, images: Sequence[np.ndarray], columns: int = 3) -> None:
    if not images:
        return
    width = max(image.shape[1] for image in images)
    height = max(image.shape[0] for image in images)
    rows = int(math.ceil(len(images) / columns))
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        canvas[row * height : row * height + image.shape[0], column * width : column * width + image.shape[1]] = image
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])


def write_confidence_scatter(
    path: Path,
    records: Sequence[Mapping[str, object]],
    confidence_threshold: float,
    translation_limit: float,
) -> None:
    canvas = np.full((620, 960, 3), 255, dtype=np.uint8)
    left, top, right, bottom = 85, 55, 925, 535
    scores = [float(record["confidence_score"]) for record in records]
    errors = [float(record["translation_error_m_mean"]) for record in records]
    x_min, x_max = min(scores), max(scores)
    y_min, y_max = 0.0, max(max(errors), translation_limit) * 1.08

    def point(x: float, y: float) -> tuple[int, int]:
        px = left + int((x - x_min) / max(1e-9, x_max - x_min) * (right - left))
        py = bottom - int((y - y_min) / max(1e-9, y_max - y_min) * (bottom - top))
        return px, py

    cv2.rectangle(canvas, (left, top), (right, bottom), (40, 40, 40), 1)
    threshold_x = point(confidence_threshold, 0.0)[0]
    limit_y = point(x_min, translation_limit)[1]
    cv2.line(canvas, (threshold_x, top), (threshold_x, bottom), (180, 0, 180), 2)
    cv2.line(canvas, (left, limit_y), (right, limit_y), (0, 0, 180), 2)
    for record in records:
        color = (0, 155, 0) if record.get("admitted") else (120, 120, 120)
        if record.get("admitted") and not clip_meets_quality(record):
            color = (0, 0, 220)
        cv2.circle(
            canvas,
            point(float(record["confidence_score"]), float(record["translation_error_m_mean"])),
            6,
            color,
            -1,
            cv2.LINE_AA,
        )
    cv2.putText(canvas, "held-out confidence vs translation error", (85, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    cv2.putText(canvas, "DA3 median depth confidence", (330, 585), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    cv2.putText(canvas, "translation error (m)", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
    cv2.imwrite(str(path), canvas)


def write_error_curves(path: Path, records: Sequence[Mapping[str, object]], offsets: Sequence[int]) -> None:
    admitted = [record for record in records if record.get("admitted")]
    canvas = np.full((620, 960, 3), 255, dtype=np.uint8)
    left, top, right, bottom = 85, 55, 925, 535
    translation = np.asarray([record["translation_errors_m"] for record in admitted], dtype=np.float64)
    yaw = np.asarray([record["yaw_errors_rad"] for record in admitted], dtype=np.float64)
    curves = {
        "translation median": (np.median(translation, axis=0), (0, 150, 0)),
        "translation p90": (np.quantile(translation, 0.90, axis=0), (0, 80, 0)),
        "yaw median": (np.median(yaw, axis=0), (200, 120, 0)),
        "yaw p90": (np.quantile(yaw, 0.90, axis=0), (180, 0, 120)),
    }
    maximum = max(float(np.max(values)) for values, _ in curves.values()) * 1.10
    x_values = list(offsets)[1:]

    def points(values: np.ndarray) -> np.ndarray:
        output = []
        for index, value in enumerate(values):
            x = left + int(index / max(1, len(values) - 1) * (right - left))
            y = bottom - int(float(value) / max(1e-9, maximum) * (bottom - top))
            output.append([x, y])
        return np.asarray(output, dtype=np.int32)

    cv2.rectangle(canvas, (left, top), (right, bottom), (40, 40, 40), 1)
    for label, (values, color) in curves.items():
        cv2.polylines(canvas, [points(values)], False, color, 3, cv2.LINE_AA)
    cv2.putText(canvas, "held-out error over clip horizon", (85, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    for index, offset in enumerate(x_values):
        x = left + int(index / max(1, len(x_values) - 1) * (right - left))
        cv2.putText(canvas, str(offset), (x - 6, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
    for index, (label, (_, color)) in enumerate(curves.items()):
        cv2.putText(canvas, label, (105 + (index % 2) * 360, 585 + (index // 2) * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1)
    cv2.imwrite(str(path), canvas)


def value_at(summary: Mapping[str, object], field: str, default: float = float("inf")) -> float:
    value = summary.get(field)
    return float(value) if value is not None else default


def markdown_report(report: Mapping[str, object]) -> str:
    scale = report["scale_calibration"]
    confidence = report["confidence_calibration"]
    heldout = report["evaluation_summary"]
    checks = report["gate_checks"]
    scale_text = ", ".join(
        f"{name}={value['scale']:.6f}"
        for name, value in scale["strata"].items()
    )
    lines = [
        "# DA3-SMALL multi-scene admission",
        "",
        f"Status: **{report['status']}**",
        "",
        f"- Calibration/evaluation clips: {report['calibration_successes']} / {report['evaluation_successes']}",
        f"- Groups: {report['calibration_groups']} calibration / {report['evaluation_groups']} held-out",
        f"- Frozen metric scales: {scale_text}; maximum calibration relative IQR: {scale['maximum_relative_iqr']:.4f}",
        f"- Frozen confidence threshold: {confidence['threshold']:.4f}",
        f"- Held-out coverage/bad rate: {heldout['coverage']:.4f} / {heldout['bad_clip_rate']:.4f}",
        f"- Held-out translation median/p90: {heldout['translation_error_m_mean']['median']:.4f} / {heldout['translation_error_m_mean']['p90']:.4f} m",
        f"- Held-out yaw median/p90: {heldout['yaw_error_rad_mean']['median']:.4f} / {heldout['yaw_error_rad_mean']['p90']:.4f} rad",
        "",
        "## Gate checks",
        "",
    ]
    lines.extend(f"- {name}: {'pass' if passed else 'FAIL'}" for name, passed in checks.items())
    lines.extend(
        [
            "",
            f"The scale and confidence threshold are fitted only on `{report['calibration_split']}`; all reported metrics use disjoint `{report['evaluation_split']}` scenes. DA3 confidence remains an empirical filter, not a calibrated probability.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = arguments()
    if args.calibration_split == args.evaluation_split:
        raise ValueError("calibration and evaluation splits must differ")
    if args.write_admission and args.evaluation_split != "test_locked":
        raise ValueError("admission.json may only be written from the test_locked split")
    offsets = sorted(set(int(value) for value in args.offsets))
    if len(offsets) < 3 or offsets[0] != 0:
        raise ValueError("offsets must contain anchor 0 and at least two later frames")
    if args.source_commit != PINNED_SOURCE_COMMIT:
        raise ValueError("multi-scene admission only supports the pinned DA3 source commit")
    policy_path = args.policy.expanduser().resolve(strict=True)
    policy = load_json(policy_path)
    if not isinstance(policy, Mapping):
        raise ValueError("DA3 multi-scene policy must be a JSON object")
    policy_id = validate_policy(args, policy)
    policy_sha256 = sha256_file(policy_path)
    quality_limits = {
        str(name): float(value)
        for name, value in policy["quality_limits"].items()
    }
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    root = args.intern_root.expanduser().resolve(strict=True)
    output_root = args.output_dir.expanduser().resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = load_json(manifest_path)
    entry = manifest["datasets"]["intern_data_n1"]
    calibration_units = list(entry["splits"][args.calibration_split])
    evaluation_units = list(entry["splits"][args.evaluation_split])
    overlap = set(calibration_units) & set(evaluation_units)
    if overlap:
        raise ValueError(f"manifest splits overlap in {len(overlap)} units")
    calibration_selection = select_split_clips(
        root,
        calibration_units,
        args.calibration_split,
        offsets=offsets,
        clips_per_group=args.clips_per_group,
        maximum_units_per_group=args.max_units_per_group,
        maximum_episodes_per_unit=args.max_episodes_per_unit,
        maximum_starts_per_episode=args.max_starts_per_episode,
        minimum_motion_m=args.selection_minimum_motion_m,
        seed=args.seed,
    )
    evaluation_selection = select_split_clips(
        root,
        evaluation_units,
        args.evaluation_split,
        offsets=offsets,
        clips_per_group=args.clips_per_group,
        maximum_units_per_group=args.max_units_per_group,
        maximum_episodes_per_unit=args.max_episodes_per_unit,
        maximum_starts_per_episode=args.max_starts_per_episode,
        minimum_motion_m=args.selection_minimum_motion_m,
        seed=args.seed,
    )
    selection_manifest = {
        "schema_version": 1,
        "seed": int(args.seed),
        "offsets": offsets,
        "selection_minimum_motion_m": float(args.selection_minimum_motion_m),
        "calibration_split": args.calibration_split,
        "evaluation_split": args.evaluation_split,
        "calibration": calibration_selection,
        "evaluation": evaluation_selection,
    }
    write_json_atomic(output_root / "selection.json", selection_manifest)

    try:
        import torch
        from depth_anything_3.api import DepthAnything3
    except ImportError as error:
        raise RuntimeError("pinned Depth Anything 3 runtime is unavailable") from error
    model = DepthAnything3.from_pretrained(args.model).to(torch.device(args.device)).eval()
    raw_records: dict[str, list[dict[str, object]]] = {"calibration": [], "evaluation": []}
    panels: dict[str, np.ndarray] = {}
    failures = []
    for role, selected in (
        ("calibration", calibration_selection),
        ("evaluation", evaluation_selection),
    ):
        for index, clip in enumerate(selected, start=1):
            try:
                record, panel = run_clip(model, torch, root, clip, args.process_res)
                raw_records[role].append(record)
                panels[str(record["clip_id"])] = panel
            except Exception as error:
                failures.append(
                    {
                        "role": role,
                        "clip_id": str(clip["clip_id"]),
                        "unit": str(clip["unit"]),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            print(
                f"{role} {index}/{len(selected)}; total failures={len(failures)}",
                flush=True,
            )

    scale = fit_stratified_translation_scales(
        raw_records["calibration"],
        minimum_clips_per_stratum=args.minimum_scale_clips,
    )

    def score_with_frozen_stratum(record: Mapping[str, object]) -> dict[str, object]:
        stratum = str(record["scale_stratum"])
        return score_clip(record, float(scale["strata"][stratum]["scale"]))

    scored_calibration = [
        score_with_frozen_stratum(record) for record in raw_records["calibration"]
    ]
    scored_evaluation = [
        score_with_frozen_stratum(record) for record in raw_records["evaluation"]
    ]
    confidence = choose_confidence_threshold(
        scored_calibration,
        quality_limits=quality_limits,
        minimum_coverage=args.minimum_confidence_coverage,
        maximum_bad_rate=args.maximum_calibration_bad_rate,
    )
    gated_calibration = apply_confidence_gate(scored_calibration, float(confidence["threshold"]))
    gated_evaluation = apply_confidence_gate(scored_evaluation, float(confidence["threshold"]))
    calibration_summary = summarize_admission(gated_calibration, quality_limits)
    evaluation_summary = summarize_admission(gated_evaluation, quality_limits)
    translation = evaluation_summary["translation_error_m_mean"]
    yaw = evaluation_summary["yaw_error_rad_mean"]
    scale_error = evaluation_summary["scale_relative_error_median"]
    gate_checks = {
        "disjoint_split_units": not overlap,
        "minimum_calibration_clips": len(raw_records["calibration"]) >= args.minimum_selected_clips,
        "minimum_evaluation_clips": len(raw_records["evaluation"]) >= args.minimum_selected_clips,
        "no_inference_failures": not failures,
        "scale_stability": float(scale["maximum_relative_iqr"])
        <= args.maximum_scale_relative_iqr,
        "confidence_calibration": bool(confidence["criteria_met"]),
        "heldout_coverage": float(evaluation_summary["coverage"]) >= args.minimum_evaluation_coverage,
        "heldout_bad_rate": float(evaluation_summary["bad_clip_rate"]) <= args.maximum_evaluation_bad_rate,
        "heldout_translation_median": value_at(translation, "median") <= args.maximum_translation_median_m,
        "heldout_translation_p90": value_at(translation, "p90") <= args.maximum_translation_p90_m,
        "heldout_yaw_median": value_at(yaw, "median") <= args.maximum_yaw_median_rad,
        "heldout_yaw_p90": value_at(yaw, "p90") <= args.maximum_yaw_p90_rad,
        "heldout_scale_error_median": value_at(scale_error, "median") <= args.maximum_scale_error_median,
        "heldout_scale_error_p90": value_at(scale_error, "p90") <= args.maximum_scale_error_p90,
    }
    status = "passed" if all(gate_checks.values()) else "failed"

    visualizations = []
    if not args.skip_visualizations:
        ranked = sorted(
            gated_evaluation,
            key=lambda record: (
                clip_meets_quality(record),
                -float(record["translation_error_m_mean"]),
            ),
        )[:12]
        visualizations = [
            "heldout_rgb_depth_confidence.jpg",
            "heldout_trajectories.jpg",
            "heldout_confidence_error.png",
        ]
        write_montage(
            output_root / visualizations[0],
            [panels[str(record["clip_id"])] for record in ranked],
            columns=2,
        )
        write_montage(
            output_root / visualizations[1],
            [trajectory_canvas(record) for record in ranked],
            columns=3,
        )
        write_confidence_scatter(
            output_root / visualizations[2],
            gated_evaluation,
            float(confidence["threshold"]),
            quality_limits["translation_error_m_mean"],
        )
        if any(record.get("admitted") for record in gated_evaluation):
            visualizations.append("heldout_error_curves.png")
            write_error_curves(
                output_root / visualizations[-1], gated_evaluation, offsets
            )

    model_path = Path(args.model).expanduser()
    weight_path = model_path / "model.safetensors" if model_path.is_dir() else None
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "policy_id": policy_id,
        "policy_sha256": policy_sha256,
        "source_commit": args.source_commit,
        "model": args.model,
        "model_revision": args.model_revision,
        "model_weight_sha256": sha256_file(weight_path) if weight_path and weight_path.is_file() else None,
        "phase1_manifest_sha256": sha256_file(manifest_path),
        "selection_sha256": sha256_file(output_root / "selection.json"),
        "process_res": int(args.process_res),
        "offsets": offsets,
        "calibration_split": args.calibration_split,
        "evaluation_split": args.evaluation_split,
        "calibration_successes": len(raw_records["calibration"]),
        "evaluation_successes": len(raw_records["evaluation"]),
        "calibration_groups": len({record["group"] for record in raw_records["calibration"]}),
        "evaluation_groups": len({record["group"] for record in raw_records["evaluation"]}),
        "inference_failures": failures,
        "scale_calibration": scale,
        "confidence_calibration": confidence,
        "quality_limits": quality_limits,
        "calibration_summary": calibration_summary,
        "evaluation_summary": evaluation_summary,
        "gate_thresholds": {
            "minimum_selected_clips": args.minimum_selected_clips,
            "maximum_scale_relative_iqr": args.maximum_scale_relative_iqr,
            "minimum_confidence_coverage": args.minimum_confidence_coverage,
            "maximum_calibration_bad_rate": args.maximum_calibration_bad_rate,
            "minimum_evaluation_coverage": args.minimum_evaluation_coverage,
            "maximum_evaluation_bad_rate": args.maximum_evaluation_bad_rate,
            "maximum_translation_median_m": args.maximum_translation_median_m,
            "maximum_translation_p90_m": args.maximum_translation_p90_m,
            "maximum_yaw_median_rad": args.maximum_yaw_median_rad,
            "maximum_yaw_p90_rad": args.maximum_yaw_p90_rad,
            "maximum_scale_error_median": args.maximum_scale_error_median,
            "maximum_scale_error_p90": args.maximum_scale_error_p90,
        },
        "gate_checks": gate_checks,
        "visualizations": visualizations,
        "camera_convention": "DA3 OpenCV w2c -> c2w -> OpenCV/Habitat bridge -> Intern base -> canonical x-forward/y-left",
    }
    write_json_atomic(output_root / "report.json", report)
    (output_root / "report.md").write_text(markdown_report(report), encoding="utf-8")
    with (output_root / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in gated_calibration + gated_evaluation:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    if args.write_admission and status == "passed":
        admission = {
            "schema_version": 1,
            "status": "passed",
            "created_at_utc": report["created_at_utc"],
            "policy_id": policy_id,
            "policy_sha256": policy_sha256,
            "model_revision": args.model_revision,
            "model_weight_sha256": report["model_weight_sha256"],
            "source_commit": args.source_commit,
            "phase1_manifest_sha256": report["phase1_manifest_sha256"],
            "selection_sha256": report["selection_sha256"],
            "report_sha256": sha256_file(output_root / "report.json"),
            "metric_scale_mode": scale["mode"],
            "metric_scales": {
                name: value["scale"] for name, value in scale["strata"].items()
            },
            "confidence_threshold": confidence["threshold"],
            "quality_limits": quality_limits,
        }
        write_json_atomic(output_root / "admission.json", admission)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
