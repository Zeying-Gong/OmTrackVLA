#!/usr/bin/env python3
"""Probe an installed Depth Anything 3 model on an InternData-N1 clip."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

PINNED_SOURCE_COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"

from omtrackvla.geometry.se2 import (
    intern_pair_to_canonical_se2,
    relative_w2c_to_base_se2,
    robust_translation_scale,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face id or local DA3 model directory")
    parser.add_argument(
        "--model-revision",
        help="immutable model repository revision recorded in the report",
    )
    parser.add_argument("--source-commit", default=PINNED_SOURCE_COMMIT)
    parser.add_argument("--intern-parquet", type=Path, required=True)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 4, 8, 12])
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--feature-layers", type=int, nargs="*", default=[5, 11])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _matrix(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def _shape(value) -> list[int] | None:
    return list(value.shape) if value is not None and hasattr(value, "shape") else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_summary(value) -> dict[str, float]:
    array = np.asarray(value, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def main() -> int:
    args = _arguments()
    try:
        import torch
        from depth_anything_3.api import DepthAnything3
    except ImportError as error:
        raise RuntimeError(
            "Depth Anything 3 is not installed; install the pinned official source before running the probe"
        ) from error
    table = pq.read_table(
        args.intern_parquet,
        columns=["observation.camera_intrinsic", "observation.camera_extrinsic", "action"],
    ).to_pydict()
    count = len(table["action"])
    indices = sorted(set(args.indices))
    if len(indices) < 2 or indices[0] < 0 or indices[-1] >= count:
        raise ValueError(f"indices must select at least two of the {count} parquet rows")
    stem = args.intern_parquet.stem
    images = [args.rgb_dir / f"{stem}_{index:03d}.jpg" for index in indices]
    missing = [str(path) for path in images if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing RGB frames: {missing}")

    model = DepthAnything3.from_pretrained(args.model).to(torch.device(args.device)).eval()
    prediction = model.inference(
        [str(path) for path in images],
        process_res=args.process_res,
        export_feat_layers=args.feature_layers,
        ref_view_strategy="middle",
    )
    required = ("depth", "conf", "extrinsics", "intrinsics", "processed_images")
    absent = [name for name in required if getattr(prediction, name, None) is None]
    if absent:
        raise RuntimeError(f"DA3 prediction is missing required fields: {absent}")
    extrinsics = np.asarray(prediction.extrinsics, dtype=np.float64)
    if extrinsics.shape[0] != len(images) or extrinsics.shape[-2:] not in ((3, 4), (4, 4)):
        raise RuntimeError(f"unexpected DA3 extrinsics shape: {extrinsics.shape}")
    if extrinsics.shape[-2:] == (3, 4):
        padded = np.broadcast_to(np.eye(4), (len(images), 4, 4)).copy()
        padded[:, :3] = extrinsics
        extrinsics = padded
    base_from_camera = _matrix(table["observation.camera_extrinsic"][0])
    predicted_motion_unscaled = np.stack(
        [
            relative_w2c_to_base_se2(
                extrinsics[0], extrinsics[index], base_from_camera
            )
            for index in range(len(images))
        ]
    )
    ground_truth_motion = np.stack(
        [
            intern_pair_to_canonical_se2(
                table["action"][indices[0]], table["action"][source_index], base_from_camera
            )
            for source_index in indices
        ]
    )
    metric_scale = robust_translation_scale(
        predicted_motion_unscaled[:, :2], ground_truth_motion[:, :2]
    )
    predicted_motion = predicted_motion_unscaled.copy()
    predicted_motion[:, :2] *= metric_scale
    errors = []
    for predicted, target in zip(predicted_motion, ground_truth_motion):
        translation = math.hypot(predicted[0] - target[0], predicted[1] - target[1])
        yaw = abs(math.atan2(math.sin(predicted[2] - target[2]), math.cos(predicted[2] - target[2])))
        errors.append({"translation": translation, "yaw": yaw})
    auxiliary = getattr(prediction, "aux", None) or {}
    expected_features = [f"feat_layer_{layer}" for layer in args.feature_layers]
    missing_features = [name for name in expected_features if auxiliary.get(name) is None]
    if missing_features:
        raise RuntimeError(f"DA3 prediction is missing requested features: {missing_features}")
    finite = {
        "depth": bool(np.isfinite(prediction.depth).all()),
        "confidence": bool(np.isfinite(prediction.conf).all()),
        "extrinsics": bool(np.isfinite(extrinsics).all()),
        "intrinsics": bool(np.isfinite(prediction.intrinsics).all()),
        "processed_images": bool(np.isfinite(prediction.processed_images).all()),
        "features": all(bool(np.isfinite(auxiliary[name]).all()) for name in expected_features),
    }
    if not all(finite.values()):
        raise RuntimeError(f"DA3 prediction contains non-finite output: {finite}")
    model_path = Path(args.model).expanduser()
    weight_path = model_path / "model.safetensors" if model_path.is_dir() else None
    report = {
        "schema_version": 1,
        "backbone": "depth-anything-3",
        "model": args.model,
        "model_revision": args.model_revision,
        "model_weight_sha256": _sha256(weight_path) if weight_path and weight_path.is_file() else None,
        "source_commit": args.source_commit,
        "input_indices": indices,
        "output_shapes": {name: _shape(getattr(prediction, name)) for name in required},
        "feature_shapes": {name: _shape(auxiliary[name]) for name in expected_features},
        "finite": finite,
        "depth_summary": _distribution_summary(prediction.depth),
        "confidence_summary": _distribution_summary(prediction.conf),
        "metric_scale": metric_scale,
        "metric_scale_method": "median matched displacement norm on moving pairs",
        "metric_scale_minimum_motion_m": 0.05,
        "calibration_evaluation_shared_clip": True,
        "predicted_se2_unscaled": predicted_motion_unscaled.tolist(),
        "predicted_se2": predicted_motion.tolist(),
        "ground_truth_se2": ground_truth_motion.tolist(),
        "translation_error_m_mean": float(np.mean([value["translation"] for value in errors])),
        "yaw_error_rad_mean": float(np.mean([value["yaw"] for value in errors])),
        "camera_convention": (
            "DA3 OpenCV w2c -> c2w -> OpenCV/Habitat camera bridge -> "
            "Intern base -> canonical x-forward/y-left"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
