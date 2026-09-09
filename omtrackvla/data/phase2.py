"""Read-only SAGE3D Phase 2 policy records backed by frozen perception output."""
from __future__ import annotations

import bisect
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from omtrackvla.data.phase1 import _bbox_xyxy_norm, _empty_uwb, _manifest, _safe_relative
from omtrackvla.data.sage3d_policy import (
    CLOCK_SPEC_ID,
    CONTROL_HZ,
    POLICY_SPEC_ID,
    SIMULATION_SPEC_ID,
    TRANSFORM_SPEC_ID,
    canonical_waypoints,
    load_policy_admission,
    sha256_file,
    target_position_base,
    waypoint_time_offsets_s,
)
from omtrackvla.data.sage3d_sidecar import (
    episode_sidecar_path,
    load_admitted_sidecar,
    validate_episode_sidecar,
)


ADAPTER_VERSION = "sage3d-phase2-policy-v1"
CONDITION_MODES = ("visual_uwb", "visual_only", "uwb_only", "safe_stop")


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(value: object, where: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{where} must be a lowercase SHA-256")
    return text


def validate_perception_cache_manifest(
    value: object,
    *,
    split: str,
    source_index_sha256: str,
    sidecar_manifest_sha256: str,
    policy_admission_sha256: str,
    split_manifest_sha256: str,
    allow_partial: bool = False,
) -> dict[str, object]:
    """Fail closed before a frozen-front-end cache can supervise Phase 2."""
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("frozen perception cache schema mismatch")
    if value.get("status") != "complete":
        raise ValueError("frozen perception cache is incomplete")
    expected = {
        "dataset_id": "sage3d_extracted",
        "split": str(split),
        "policy_spec_id": POLICY_SPEC_ID,
        "source_index_sha256": source_index_sha256,
        "sidecar_manifest_sha256": sidecar_manifest_sha256,
        "policy_admission_sha256": policy_admission_sha256,
        "split_manifest_sha256": split_manifest_sha256,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"frozen perception cache {key} mismatch")
    front_end = value.get("front_end")
    if not isinstance(front_end, dict) or front_end.get("frozen") is not True:
        raise ValueError("perception cache front end is not frozen")
    for key in (
        "detector_weights_sha256",
        "reid_weights_sha256",
        "reid_code_sha256",
        "fusion_weights_sha256",
    ):
        _sha256(front_end.get(key), f"front_end.{key}")
    selection = value.get("selection")
    if not isinstance(selection, dict) or selection.get("test_locked_used") is not False:
        raise ValueError("frozen perception cache must not consume test_locked")
    partial = (
        selection.get("max_units") is not None
        or selection.get("max_episodes") is not None
        or int(selection.get("num_shards", 1)) != 1
        or int(selection.get("shard_index", 0)) != 0
    )
    if partial and not allow_partial:
        raise ValueError("partial frozen perception cache requires an explicit development override")
    episodes = value.get("episodes")
    if not isinstance(episodes, dict) or not episodes:
        raise ValueError("frozen perception cache contains no episodes")
    if int(selection.get("cached_episodes", -1)) != len(episodes):
        raise ValueError("frozen perception cache episode count mismatch")
    return value


def validate_perception_cache_payload(
    value: object,
    *,
    source_path: str,
    anchors: Sequence[int],
    front_end: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"invalid perception episode cache: {source_path}")
    if value.get("policy_spec_id") != POLICY_SPEC_ID or value.get("source_path") != source_path:
        raise ValueError(f"perception episode provenance mismatch: {source_path}")
    if value.get("front_end") != dict(front_end):
        raise ValueError(f"perception episode front end mismatch: {source_path}")
    _sha256(value.get("source_derived_sha256"), "source_derived_sha256")
    _sha256(value.get("source_sidecar_sha256"), "source_sidecar_sha256")
    expected_anchors = [int(anchor) for anchor in anchors]
    if value.get("anchors") != expected_anchors:
        raise ValueError(f"perception episode anchors mismatch: {source_path}")
    initial = value.get("initial_index")
    if not isinstance(initial, int) or initial < 0 or any(anchor <= initial for anchor in expected_anchors):
        raise ValueError(f"perception episode initialization is invalid: {source_path}")
    records = value.get("records")
    if not isinstance(records, list) or len(records) != len(expected_anchors):
        raise ValueError(f"perception episode record count mismatch: {source_path}")
    for expected_anchor, record in zip(expected_anchors, records):
        if not isinstance(record, dict) or record.get("anchor_index") != expected_anchor:
            raise ValueError(f"perception episode record order mismatch: {source_path}")
        relative = record.get("relative_xy")
        if (
            not isinstance(relative, list)
            or len(relative) != 2
            or not all(math.isfinite(float(component)) for component in relative)
        ):
            raise ValueError(f"perception episode relative position is invalid: {source_path}")
        confidence = float(record.get("confidence", float("nan")))
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"perception episode confidence is invalid: {source_path}")
        visible = record.get("visible")
        predicted_bbox = record.get("predicted_bbox_xyxy")
        if not isinstance(visible, bool) or visible != (predicted_bbox is not None):
            raise ValueError(f"perception episode visibility/bbox mismatch: {source_path}")
        if predicted_bbox is not None and (
            not isinstance(predicted_bbox, list)
            or len(predicted_bbox) != 4
            or not all(math.isfinite(float(component)) for component in predicted_bbox)
        ):
            raise ValueError(f"perception episode bbox is invalid: {source_path}")
    return value


def _history_indices(initial: int, anchor: int, history_size: int, visual: bool) -> list[int]:
    if anchor <= initial or history_size < 2:
        raise ValueError("Phase 2 anchor/history is invalid")
    if not visual:
        return list(range(max(0, anchor - history_size + 1), anchor + 1))
    tail = list(range(max(initial + 1, anchor - history_size + 2), anchor + 1))
    return [initial, *tail]


def _timestamp_ns(step_id: int) -> int:
    return int(round(int(step_id) * 1_000_000_000.0 / CONTROL_HZ))


def conditioned_visual_features(
    perception: Mapping[str, object], condition_mode: str
) -> tuple[list[float], float, bool]:
    if condition_mode not in CONDITION_MODES:
        raise ValueError(f"unsupported Phase 2 condition mode: {condition_mode}")
    if condition_mode not in {"visual_uwb", "visual_only"}:
        return [0.0, 0.0], 0.0, False
    return (
        [float(value) for value in perception["relative_xy"]],
        float(perception["confidence"]),
        bool(perception["visible"]),
    )


def build_policy_record(
    *,
    root: Path,
    episode_relative: str,
    split_unit_id: str,
    episode_id: str,
    steps: Sequence[Mapping[str, object]],
    labels: Sequence[Mapping[str, object]],
    image_size: tuple[int, int],
    initial_index: int,
    anchor_index: int,
    condition_mode: str,
    history_size: int,
) -> dict[str, object]:
    if condition_mode not in CONDITION_MODES:
        raise ValueError(f"unsupported Phase 2 condition mode: {condition_mode}")
    if len(steps) != len(labels):
        raise ValueError("SAGE3D source/sidecar length mismatch")
    visual_condition = condition_mode in {"visual_uwb", "visual_only"}
    uwb_condition = condition_mode in {"visual_uwb", "uwb_only"}
    indices = _history_indices(initial_index, anchor_index, history_size, visual_condition)
    episode = _safe_relative(root, episode_relative)
    relative_paths = [
        (episode / "rgb" / f"{int(steps[index]['step']):05d}.jpg")
        .relative_to(root)
        .as_posix()
        for index in indices
    ]
    timestamps = [_timestamp_ns(int(steps[index]["step"])) for index in indices]
    initial_label = labels[initial_index]
    if not initial_label.get("visible") or initial_label.get("bbox_xyxy") is None:
        raise ValueError("visual identity initialization label is not visible")
    visual_initialization = (
        {
            "valid": True,
            "rgb_path": relative_paths[0],
            "bbox_xyxy_norm": _bbox_xyxy_norm(initial_label["bbox_xyxy"], image_size),
            "timestamp_ns": timestamps[0],
        }
        if visual_condition
        else {
            "valid": False,
            "rgb_path": None,
            "bbox_xyxy_norm": None,
            "timestamp_ns": None,
        }
    )
    anchor_timestamp = timestamps[-1]
    target_xy = target_position_base(steps[anchor_index])
    if uwb_condition:
        uwb = {
            "valid": True,
            "measurement_kind": "simulated_uwb",
            "relative_position_base_xy_m": target_xy,
            "covariance_base_xy_m2": [[0.0, 0.0], [0.0, 0.0]],
            "quality_01": 1.0,
            "source_timestamp_ns": anchor_timestamp,
            "receive_timestamp_ns": anchor_timestamp,
            "age_s": 0.0,
            "los_state": "unknown",
        }
    else:
        uwb = _empty_uwb()

    anchor_label = labels[anchor_index]
    visible = bool(anchor_label.get("visible") and anchor_label.get("bbox_xyxy"))
    bbox = (
        _bbox_xyxy_norm(anchor_label["bbox_xyxy"], image_size) if visible else None
    )
    reason = str(anchor_label.get("visibility_reason", "unknown"))
    occlusion = (
        "visible"
        if visible
        else "occluded"
        if reason == "occluded_by_nearer_surface"
        else "out_of_view"
    )
    waypoints = (
        [[0.0, 0.0] for _ in range(8)]
        if condition_mode == "safe_stop"
        else canonical_waypoints(steps, anchor_index)
    )
    target_id = f"sage:{episode_id}"
    return {
        "schema_version": 1,
        "sample_id": f"sage3d_extracted/{episode_id}/anchor-{anchor_index}/{condition_mode}",
        "sample_role": "policy",
        "source": {
            "dataset_id": "sage3d_extracted",
            "split_unit_id": split_unit_id,
            "episode_id": episode_id,
            "anchor_index": int(anchor_index),
            "adapter_version": ADAPTER_VERSION,
        },
        "model_inputs": {
            "condition_mode": condition_mode,
            "anchor_timestamp_ns": anchor_timestamp,
            "rgb_sensor_valid": True,
            "rgb_history": [
                {"rgb_path": path, "timestamp_ns": timestamp, "valid": True}
                for path, timestamp in zip(relative_paths, timestamps)
            ],
            "visual_initialization": visual_initialization,
            "uwb_target": uwb,
        },
        "routing_metadata": {
            "target_tag_id": target_id if uwb_condition else None,
            "calibration_id": None,
            "simulation_spec_id": SIMULATION_SPEC_ID if uwb_condition else None,
        },
        "supervision": {
            "expert_trajectory": {
                "valid": True,
                "waypoints_base_xy_m": waypoints,
                "time_offsets_s": waypoint_time_offsets_s(),
                "valid_mask": [True] * 8,
            },
            "auxiliary_labels": {
                "target_track_id": target_id,
                "target_bbox_xyxy_norm": bbox,
                "target_visible": visible,
                "target_position_base_xy_m": target_xy,
                "occlusion_state": occlusion,
                "uwb_error_base_xy_m": [0.0, 0.0] if uwb_condition else None,
            },
            "safety": {
                "stop_required": condition_mode == "safe_stop",
                "reason": "no_reliable_target" if condition_mode == "safe_stop" else "none",
            },
        },
        "provenance": {
            "source_record": (
                episode / "derived.json"
            ).relative_to(root).as_posix(),
            "transform_spec_id": TRANSFORM_SPEC_ID,
            "clock_spec_id": CLOCK_SPEC_ID,
            "generation_spec_id": POLICY_SPEC_ID,
        },
    }


def _rgb_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class Sage3DPolicyDataset(Dataset):
    """Four-mode policy views over cache records from the frozen identity front end."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        root: str | Path,
        sidecar_root: str | Path,
        policy_admission: str | Path,
        perception_cache: str | Path,
        *,
        history_size: int = 8,
        image_size: int = 224,
        max_units: int | None = None,
        include_render_data: bool = False,
        allow_partial_cache: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve(strict=True)
        self.split = str(split)
        self.history_size = int(history_size)
        self.image_size = int(image_size)
        self.include_render_data = bool(include_render_data)
        manifest = _manifest(Path(manifest_path))
        admission_path = Path(policy_admission).expanduser().resolve(strict=True)
        load_policy_admission(admission_path, self.root)
        self.sidecar_root = Path(sidecar_root).expanduser().resolve(strict=True)
        sidecar_manifest, _ = load_admitted_sidecar(self.sidecar_root)
        cache_root = Path(perception_cache).expanduser().resolve(strict=True)
        cache_manifest = validate_perception_cache_manifest(
            _load(cache_root / "manifest.json"),
            split=self.split,
            source_index_sha256=sha256_file(self.root / "index.json"),
            sidecar_manifest_sha256=sha256_file(self.sidecar_root / "manifest.json"),
            policy_admission_sha256=sha256_file(admission_path),
            split_manifest_sha256=sha256_file(Path(manifest_path)),
            allow_partial=allow_partial_cache,
        )
        self.cache_root = cache_root
        entries = {str(entry["path"]): entry for entry in _load(self.root / "index.json")["eps"]}
        allowed_units = list(manifest["datasets"]["sage3d_extracted"]["splits"][self.split])
        if max_units is not None:
            allowed_units = allowed_units[: int(max_units)]
        allowed = set(allowed_units)
        sidecar_episodes = sidecar_manifest["episodes"]
        descriptors = []
        for relative, cache_metadata in sorted(cache_manifest["episodes"].items()):
            entry = entries.get(relative)
            if entry is None or str(entry["run"]) not in allowed:
                continue
            sidecar_metadata = sidecar_episodes.get(relative)
            if not isinstance(sidecar_metadata, dict):
                raise ValueError(f"policy cache episode lacks admitted sidecar: {relative}")
            cache_path = _safe_relative(cache_root, str(cache_metadata["cache_path"]))
            if sha256_file(cache_path) != cache_metadata["cache_sha256"]:
                raise ValueError(f"policy cache checksum mismatch: {relative}")
            anchors = [int(value) for value in cache_metadata["anchors"]]
            if not anchors:
                continue
            descriptors.append(
                {
                    "relative": relative,
                    "entry": entry,
                    "cache_path": cache_path,
                    "cache_sha256": cache_metadata["cache_sha256"],
                    "anchors": anchors,
                    "front_end": cache_manifest["front_end"],
                    "sidecar_path": episode_sidecar_path(self.sidecar_root, relative),
                    "sidecar_sha256": sidecar_metadata["sidecar_sha256"],
                }
            )
        if not descriptors:
            raise ValueError(f"no cached SAGE3D Phase 2 samples for split={split}")
        self.descriptors = descriptors
        self._ends = []
        total = 0
        for descriptor in descriptors:
            total += len(descriptor["anchors"]) * len(CONDITION_MODES)
            self._ends.append(total)

    def __len__(self) -> int:
        return self._ends[-1]

    def _locate(self, index: int) -> tuple[dict[str, object], int, str]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        descriptor_index = bisect.bisect_right(self._ends, index)
        start = self._ends[descriptor_index - 1] if descriptor_index else 0
        local = index - start
        descriptor = self.descriptors[descriptor_index]
        mode = CONDITION_MODES[local % len(CONDITION_MODES)]
        anchor = descriptor["anchors"][local // len(CONDITION_MODES)]
        return descriptor, int(anchor), mode

    @lru_cache(maxsize=16)
    def _source(self, relative: str, sidecar_path: str, sidecar_sha256: str):
        episode = _safe_relative(self.root, relative)
        derived = _load(episode / "derived.json")
        if sha256_file(Path(sidecar_path)) != sidecar_sha256:
            raise ValueError(f"SAGE3D sidecar checksum mismatch: {relative}")
        labels = validate_episode_sidecar(_load(Path(sidecar_path)), relative)
        return derived, labels, _load(episode / "camera_info.json")

    @lru_cache(maxsize=16)
    def _cache(
        self,
        path: str,
        expected_sha256: str,
        source_path: str,
        anchors: tuple[int, ...],
        front_end_json: str,
    ):
        cache_path = Path(path)
        if sha256_file(cache_path) != expected_sha256:
            raise ValueError(f"frozen perception cache changed: {cache_path}")
        payload = validate_perception_cache_payload(
            _load(cache_path),
            source_path=source_path,
            anchors=anchors,
            front_end=json.loads(front_end_json),
        )
        return payload, {int(record["anchor_index"]): record for record in payload["records"]}

    def _cached_episode(self, descriptor: Mapping[str, object]):
        return self._cache(
            str(descriptor["cache_path"]),
            str(descriptor["cache_sha256"]),
            str(descriptor["relative"]),
            tuple(int(value) for value in descriptor["anchors"]),
            json.dumps(descriptor["front_end"], sort_keys=True),
        )

    def get_record(self, index: int) -> dict[str, object]:
        descriptor, anchor, mode = self._locate(index)
        derived, labels, camera = self._source(
            str(descriptor["relative"]),
            str(descriptor["sidecar_path"]),
            str(descriptor["sidecar_sha256"]),
        )
        cache_payload, _ = self._cached_episode(descriptor)
        episode = _safe_relative(self.root, str(descriptor["relative"]))
        if cache_payload["source_derived_sha256"] != sha256_file(episode / "derived.json"):
            raise ValueError(f"frozen perception source changed: {descriptor['relative']}")
        if cache_payload["source_sidecar_sha256"] != str(descriptor["sidecar_sha256"]):
            raise ValueError(f"frozen perception sidecar changed: {descriptor['relative']}")
        entry = descriptor["entry"]
        return build_policy_record(
            root=self.root,
            episode_relative=str(descriptor["relative"]),
            split_unit_id=str(entry["run"]),
            episode_id=f"{entry['run']}/{entry['mode']}/{entry['ep']}/{entry['cam']}",
            steps=derived["steps"],
            labels=labels["steps"],
            image_size=(int(camera["camera"]["width"]), int(camera["camera"]["height"])),
            initial_index=int(cache_payload["initial_index"]),
            anchor_index=anchor,
            condition_mode=mode,
            history_size=self.history_size,
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        descriptor, anchor, mode = self._locate(index)
        record = self.get_record(index)
        _, cache_records = self._cached_episode(descriptor)
        perception = cache_records[anchor]
        visual_xy, visual_confidence, visual_valid = conditioned_visual_features(
            perception, mode
        )
        uwb = record["model_inputs"]["uwb_target"]
        current = record["model_inputs"]["rgb_history"][-1]["rgb_path"]
        labels = record["supervision"]["auxiliary_labels"]
        predicted_bbox = perception.get("predicted_bbox_xyxy")
        episode = _safe_relative(self.root, str(descriptor["relative"]))
        camera = _load(episode / "camera_info.json")["camera"]
        item = {
            "visual_xy": torch.tensor(visual_xy, dtype=torch.float32),
            "visual_confidence": torch.tensor(visual_confidence, dtype=torch.float32),
            "visual_valid": torch.tensor(float(visual_valid), dtype=torch.float32),
            "uwb_xy": torch.tensor(
                uwb["relative_position_base_xy_m"] if uwb["valid"] else [0.0, 0.0],
                dtype=torch.float32,
            ),
            "uwb_quality": torch.tensor(float(uwb["quality_01"] or 0.0), dtype=torch.float32),
            "uwb_valid": torch.tensor(float(uwb["valid"]), dtype=torch.float32),
            "uwb_age_s": torch.tensor(float(uwb["age_s"] or 0.0), dtype=torch.float32),
            "condition_index": torch.tensor(CONDITION_MODES.index(mode), dtype=torch.long),
            "waypoints": torch.tensor(
                record["supervision"]["expert_trajectory"]["waypoints_base_xy_m"],
                dtype=torch.float32,
            ),
            "waypoint_mask": torch.ones(8, dtype=torch.bool),
            "safe_stop": torch.tensor(float(mode == "safe_stop"), dtype=torch.float32),
            "predicted_bbox": torch.tensor(
                _bbox_xyxy_norm(
                    predicted_bbox, (int(camera["width"]), int(camera["height"]))
                )
                if predicted_bbox is not None
                else [0.0] * 4,
                dtype=torch.float32,
            ),
            "predicted_visible": torch.tensor(float(predicted_bbox is not None), dtype=torch.float32),
            "target_bbox": torch.tensor(
                labels["target_bbox_xyxy_norm"] or [0.0] * 4, dtype=torch.float32
            ),
            "target_visible": torch.tensor(float(labels["target_visible"]), dtype=torch.float32),
            "sample_id": record["sample_id"],
            "condition_mode": mode,
        }
        if self.include_render_data:
            item["current_rgb"] = _rgb_tensor(
                _safe_relative(self.root, current), self.image_size
            )
        return item
