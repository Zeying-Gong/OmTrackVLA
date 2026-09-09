"""Run the frozen detector/ReID front end over SAGE3D episodes read-only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from omtrackvla.data.phase1 import _manifest, _safe_relative
from omtrackvla.data.sage3d_policy import POLICY_SPEC_ID, load_policy_admission, sha256_file
from omtrackvla.data.sage3d_sidecar import (
    episode_sidecar_path,
    load_admitted_sidecar,
    validate_episode_sidecar,
)
from omtrackvla.rgb_person_perception import RGBPersonPerception, bbox_iou


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "viz_val"), required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--policy-admission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--detector-architecture", default="fasterrcnn_resnet50_fpn_v2")
    parser.add_argument("--reid-weights", type=Path, required=True)
    parser.add_argument("--reid-code", type=Path, required=True)
    parser.add_argument("--fusion-weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--record-stride", type=int, default=3)
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(path)
    return depth


def _crop(rgb: np.ndarray, bbox) -> np.ndarray:
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = map(float, bbox)
    xa, ya = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    xb, yb = min(width, int(np.ceil(x1))), min(height, int(np.ceil(y1)))
    if xb <= xa or yb <= ya:
        raise ValueError("identity initialization bbox is empty")
    return rgb[ya:yb, xa:xb].copy()


def _episode_output(root: Path, relative: str) -> Path:
    parts = Path(*Path(relative).parts)
    return root / "episodes" / parts.parent / f"{parts.name}.json"


def main() -> int:
    args = _arguments()
    if args.record_stride <= 0:
        raise ValueError("record stride must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must be in [0, num_shards)")
    root = args.data_root.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve(strict=False)
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("perception cache must be outside the read-only source root")
    manifest = _manifest(args.manifest)
    admission_path = args.policy_admission.expanduser().resolve(strict=True)
    load_policy_admission(admission_path, root)
    sidecar_root = args.sidecar_root.expanduser().resolve(strict=True)
    sidecar_manifest, _ = load_admitted_sidecar(sidecar_root)
    if sidecar_manifest["source_index_sha256"] != sha256_file(root / "index.json"):
        raise ValueError("SAGE3D sidecar/source mismatch")
    units = list(manifest["datasets"]["sage3d_extracted"]["splits"][args.split])
    if args.max_units is not None:
        units = units[: args.max_units]
    allowed = set(units)
    entries = [
        entry for entry in _load(root / "index.json")["eps"] if str(entry["run"]) in allowed
    ]
    entries.sort(key=lambda entry: str(entry["path"]))
    entries = entries[args.shard_index :: args.num_shards]
    if args.max_episodes is not None:
        entries = entries[: args.max_episodes]
    if not entries:
        raise ValueError("no SAGE3D episodes selected for perception cache")
    front_end = {
        "detector_architecture": args.detector_architecture,
        "detector_weights_sha256": sha256_file(args.detector_weights),
        "reid_weights_sha256": sha256_file(args.reid_weights),
        "reid_code_sha256": sha256_file(args.reid_code),
        "fusion_weights_sha256": sha256_file(args.fusion_weights),
        "frozen": True,
    }
    tracker = RGBPersonPerception(
        weights_path=args.detector_weights,
        detector_architecture=args.detector_architecture,
        reid_weights_path=args.reid_weights,
        reid_code_path=args.reid_code,
        fusion_weights_path=args.fusion_weights,
        device=args.device,
    )
    episodes = {}
    skipped = []
    for episode_number, entry in enumerate(entries, 1):
        relative = str(entry["path"])
        episode = _safe_relative(root, relative)
        derived_path = episode / "derived.json"
        sidecar_path = episode_sidecar_path(sidecar_root, relative)
        sidecar_metadata = sidecar_manifest["episodes"].get(relative)
        if not isinstance(sidecar_metadata, dict):
            raise ValueError(f"missing admitted sidecar metadata: {relative}")
        if sha256_file(sidecar_path) != sidecar_metadata["sidecar_sha256"]:
            raise ValueError(f"sidecar checksum mismatch: {relative}")
        derived = _load(derived_path)
        labels = validate_episode_sidecar(_load(sidecar_path), relative)
        steps, label_steps = derived["steps"], labels["steps"]
        initial_candidates = [
            index for index, label in enumerate(label_steps)
            if label.get("visible") and label.get("bbox_xyxy") is not None
        ]
        final_anchor = len(steps) - 1 - 21
        if not initial_candidates or initial_candidates[0] >= final_anchor:
            skipped.append({"path": relative, "reason": "no_initialization_or_horizon"})
            continue
        initial = initial_candidates[0]
        anchors = list(range(initial + 1, final_anchor + 1, args.record_stride))
        destination = _episode_output(output, relative)
        if args.resume and destination.is_file():
            payload = _load(destination)
            if (
                payload.get("schema_version") == 1
                and payload.get("policy_spec_id") == POLICY_SPEC_ID
                and payload.get("source_path") == relative
                and payload.get("front_end") == front_end
                and payload.get("source_derived_sha256") == sha256_file(derived_path)
                and payload.get("source_sidecar_sha256") == sha256_file(sidecar_path)
                and payload.get("anchors") == anchors
                and len(payload.get("records", ())) == len(anchors)
            ):
                episodes[relative] = {
                    "cache_path": destination.relative_to(output).as_posix(),
                    "cache_sha256": sha256_file(destination),
                    "anchors": anchors,
                }
                print(f"[{episode_number}/{len(entries)}] resume {relative}", flush=True)
                continue
        reference_rgb = _read_rgb(episode / "rgb" / f"{int(steps[initial]['step']):05d}.jpg")
        tracker.reset(_crop(reference_rgb, label_steps[initial]["bbox_xyxy"]))
        anchor_set = set(anchors)
        records = []
        for index in range(initial + 1, anchors[-1] + 1):
            step_id = int(steps[index]["step"])
            rgb = _read_rgb(episode / "rgb" / f"{step_id:05d}.jpg")
            depth = _read_depth(episode / "depth" / f"{step_id:05d}.png")
            observation = tracker(rgb, depth)
            if index not in anchor_set:
                continue
            gt_box = label_steps[index].get("bbox_xyxy")
            predicted_box = observation.bbox_xyxy
            records.append(
                {
                    "anchor_index": index,
                    "visible": bool(observation.visible),
                    "relative_xy": [float(value) for value in observation.relative_xy],
                    "confidence": float(observation.confidence),
                    "predicted_bbox_xyxy": (
                        [float(value) for value in predicted_box] if predicted_box is not None else None
                    ),
                    "candidate_count": int(tracker.last_candidate_count),
                    "association_score": float(tracker.last_association_score),
                    "identity_margin": float(tracker.last_identity_margin),
                    "gt_visible": bool(label_steps[index].get("visible")),
                    "gt_iou": (
                        float(bbox_iou(predicted_box, gt_box))
                        if predicted_box is not None and gt_box is not None
                        else 0.0
                    ),
                }
            )
        payload = {
            "schema_version": 1,
            "policy_spec_id": POLICY_SPEC_ID,
            "source_path": relative,
            "source_derived_sha256": sha256_file(derived_path),
            "source_sidecar_sha256": sha256_file(sidecar_path),
            "front_end": front_end,
            "initial_index": initial,
            "anchors": anchors,
            "records": records,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(destination)
        episodes[relative] = {
            "cache_path": destination.relative_to(output).as_posix(),
            "cache_sha256": sha256_file(destination),
            "anchors": anchors,
        }
        print(f"[{episode_number}/{len(entries)}] complete {relative} anchors={len(anchors)}", flush=True)
    cache_manifest = {
        "schema_version": 1,
        "status": "complete",
        "dataset_id": "sage3d_extracted",
        "split": args.split,
        "policy_spec_id": POLICY_SPEC_ID,
        "source_index_sha256": sha256_file(root / "index.json"),
        "sidecar_manifest_sha256": sha256_file(sidecar_root / "manifest.json"),
        "policy_admission_sha256": sha256_file(admission_path),
        "split_manifest_sha256": sha256_file(args.manifest),
        "front_end": front_end,
        "selection": {
            "unit_count": len(units),
            "requested_episodes": len(entries),
            "cached_episodes": len(episodes),
            "record_stride": args.record_stride,
            "max_units": args.max_units,
            "max_episodes": args.max_episodes,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "test_locked_used": False,
        },
        "episodes": episodes,
        "skipped": skipped,
    }
    if not episodes:
        raise ValueError("no SAGE3D episodes produced a usable perception cache")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    print(json.dumps(cache_manifest["selection"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
