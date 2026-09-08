#!/usr/bin/env python3
"""Probe an installed Depth Anything 3 model on an InternData-N1 clip."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from omtrackvla.geometry.se2 import intern_pair_to_canonical_se2, relative_w2c_to_base_se2


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face id or local DA3 model directory")
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
    camera_from_base = np.linalg.inv(base_from_camera)
    predicted_motion = [
        relative_w2c_to_base_se2(extrinsics[0], extrinsics[index], camera_from_base).tolist()
        for index in range(len(images))
    ]
    ground_truth_motion = [
        intern_pair_to_canonical_se2(
            table["action"][indices[0]], table["action"][source_index], base_from_camera
        ).tolist()
        for source_index in indices
    ]
    errors = []
    for predicted, target in zip(predicted_motion, ground_truth_motion):
        translation = math.hypot(predicted[0] - target[0], predicted[1] - target[1])
        yaw = abs(math.atan2(math.sin(predicted[2] - target[2]), math.cos(predicted[2] - target[2])))
        errors.append({"translation": translation, "yaw": yaw})
    auxiliary = getattr(prediction, "aux", None) or {}
    report = {
        "schema_version": 1,
        "backbone": "depth-anything-3",
        "model": args.model,
        "source_commit": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
        "input_indices": indices,
        "output_shapes": {name: _shape(getattr(prediction, name)) for name in required},
        "feature_shapes": {key: _shape(value) for key, value in auxiliary.items() if key.startswith("feat_layer_")},
        "finite": {
            "depth": bool(np.isfinite(prediction.depth).all()),
            "confidence": bool(np.isfinite(prediction.conf).all()),
            "extrinsics": bool(np.isfinite(extrinsics).all()),
        },
        "predicted_se2": predicted_motion,
        "ground_truth_se2": ground_truth_motion,
        "translation_error_m_mean": float(np.mean([value["translation"] for value in errors])),
        "yaw_error_rad_mean": float(np.mean([value["yaw"] for value in errors])),
        "camera_convention": "DA3 OpenCV w2c converted through inverse and ^C T_B",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
