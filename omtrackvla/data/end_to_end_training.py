"""Formal raw-RGB SAGE3D sequences for Architecture v1 NEXT-026."""
from __future__ import annotations

import bisect
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from omtrackvla.data.end_to_end import _rgb_tensor, sage_camera_calibration
from omtrackvla.data.phase1 import _manifest
from omtrackvla.data.sage3d_policy import (
    POLICY_SPEC_ID,
    canonical_waypoints,
    load_policy_admission,
    sha256_file,
    target_position_base,
    world_delta_to_base,
)
from omtrackvla.data.sage3d_sidecar import (
    episode_sidecar_path,
    load_admitted_sidecar,
    validate_episode_sidecar,
)
from omtrackvla.models.end_to_end import ArchitectureV1Config


SEQUENCE_INDEX_SPEC_ID = "architecture-v1-sage3d-sequences-v1"
CONDITION_MODES = ("visual_uwb", "visual_only", "uwb_only", "safe_stop")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_episode(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe SAGE3D episode path: {relative}")
    episode = (root / Path(*pure.parts)).resolve(strict=True)
    try:
        episode.relative_to(root)
    except ValueError as error:
        raise ValueError(f"SAGE3D episode escaped its root: {relative}") from error
    return episode


def _bbox_norm(
    bbox_xyxy: Sequence[float] | None, width: float, height: float
) -> list[float]:
    if bbox_xyxy is None or len(bbox_xyxy) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    result = [
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
        max(0.0, min(1.0, x2 / width)),
        max(0.0, min(1.0, y2 / height)),
    ]
    if result[2] <= result[0] or result[3] <= result[1]:
        return [0.0, 0.0, 0.0, 0.0]
    return result


def _relative_motion(
    previous: Mapping[str, object], current: Mapping[str, object]
) -> list[float]:
    previous_position = [float(value) for value in previous["robot_pos"][:2]]
    current_position = [float(value) for value in current["robot_pos"][:2]]
    delta_world = [
        current_position[0] - previous_position[0],
        current_position[1] - previous_position[1],
    ]
    dx, dy = world_delta_to_base(delta_world, float(previous["robot_yaw"]))
    dyaw = float(current["robot_yaw"]) - float(previous["robot_yaw"])
    dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
    return [dx, dy, math.sin(dyaw), math.cos(dyaw)]


def _realized_action(
    previous: Mapping[str, object], current: Mapping[str, object]
) -> list[float]:
    """Return the policy-transition action as explicit ``dx, dy, dyaw``."""

    dx, dy, sin_yaw, cos_yaw = _relative_motion(previous, current)
    return [dx, dy, math.atan2(sin_yaw, cos_yaw)]


def _canonical_waypoint_yaws(
    steps: Sequence[Mapping[str, object]],
    anchor_index: int,
    *,
    horizon: int = 8,
    stride_steps: int = 3,
) -> list[float]:
    """Return future base yaw relative to the anchor for each 10 Hz waypoint."""

    anchor_yaw = float(steps[anchor_index]["robot_yaw"])
    values = []
    for point_index in range(horizon):
        future_yaw = float(
            steps[anchor_index + point_index * stride_steps]["robot_yaw"]
        )
        delta = future_yaw - anchor_yaw
        values.append(math.atan2(math.sin(delta), math.cos(delta)))
    values[0] = 0.0
    return values


def build_sage3d_sequence_index(
    *,
    root: str | Path,
    sidecar_root: str | Path,
    split_manifest: str | Path,
    policy_admission: str | Path,
    output: str | Path,
    config: ArchitectureV1Config | None = None,
    sequence_steps: int = 2,
    policy_stride: int = 3,
    anchor_stride: int = 3,
    history_stride_raw: int = 1,
) -> dict[str, object]:
    """Build one label-derived index without reading or writing source data."""

    config = config or ArchitectureV1Config()
    config.validate()
    if (
        sequence_steps < 1
        or policy_stride < 1
        or anchor_stride < 1
        or history_stride_raw < 1
    ):
        raise ValueError("sequence and stride values must be positive")
    root = Path(root).expanduser().resolve(strict=True)
    sidecar_root = Path(sidecar_root).expanduser().resolve(strict=True)
    split_manifest = Path(split_manifest).expanduser().resolve(strict=True)
    policy_admission = Path(policy_admission).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve(strict=False)
    load_policy_admission(policy_admission, root)
    split_value = _manifest(split_manifest)
    sidecar_manifest, _ = load_admitted_sidecar(sidecar_root)
    source_index = _load_json(root / "index.json")

    sage_split = split_value["datasets"]["sage3d_extracted"]
    if sage_split.get("split_unit") != "run":
        raise ValueError("SAGE3D sequence index requires run-level splits")
    allowed_splits = ("train", "val", "viz_val")
    run_to_split: dict[str, str] = {}
    for split in allowed_splits:
        for run in sage_split["splits"][split]:
            run = str(run)
            if run in run_to_split:
                raise ValueError(f"SAGE3D run appears in multiple splits: {run}")
            run_to_split[run] = split

    descriptors: dict[str, list[dict[str, object]]] = {
        split: [] for split in allowed_splits
    }
    skipped: dict[str, int] = {}
    sidecar_episodes = sidecar_manifest["episodes"]
    future_offset = (config.horizon - 1) * 3
    for entry in source_index["eps"]:
        run = str(entry["run"])
        split = run_to_split.get(run)
        if split is None:
            continue
        relative = str(entry["path"])
        metadata = sidecar_episodes.get(relative)
        if not isinstance(metadata, Mapping):
            skipped["missing_sidecar"] = skipped.get("missing_sidecar", 0) + 1
            continue
        sidecar_path = episode_sidecar_path(sidecar_root, relative)
        if sha256_file(sidecar_path) != metadata["sidecar_sha256"]:
            raise ValueError(f"SAGE3D sidecar checksum mismatch: {relative}")
        sidecar = validate_episode_sidecar(_load_json(sidecar_path), relative)
        labels = sidecar["steps"]
        initial_index = next(
            (
                index
                for index, label in enumerate(labels)
                if label.get("visible") and label.get("bbox_xyxy") is not None
            ),
            None,
        )
        if initial_index is None:
            skipped["no_visible_initialization"] = (
                skipped.get("no_visible_initialization", 0) + 1
            )
            continue
        frames = min(int(entry.get("steps", len(labels))), len(labels))
        anchor_start = max(
            (config.history_size - 1) * history_stride_raw,
            int(initial_index) + 1,
        )
        anchor_end = (
            frames
            - 1
            - future_offset
            - (sequence_steps - 1) * policy_stride
        )
        if anchor_start > anchor_end:
            skipped["insufficient_sequence"] = (
                skipped.get("insufficient_sequence", 0) + 1
            )
            continue
        anchor_count = (anchor_end - anchor_start) // anchor_stride + 1
        episode = _safe_episode(root, relative)
        derived_path = episode / "derived.json"
        camera_info_path = episode / "camera_info.json"
        if sha256_file(derived_path) != sidecar["source_derived_sha256"]:
            raise ValueError(f"SAGE3D derived provenance mismatch: {relative}")
        if sha256_file(camera_info_path) != sidecar["source_camera_info_sha256"]:
            raise ValueError(f"SAGE3D camera provenance mismatch: {relative}")
        descriptors[split].append(
            {
                "relative": relative,
                "run": run,
                "mode": str(entry["mode"]),
                "episode": str(entry["ep"]),
                "camera": str(entry["cam"]),
                "frames": frames,
                "initial_index": int(initial_index),
                "anchor_start": int(anchor_start),
                "anchor_end": int(anchor_end),
                "anchor_count": int(anchor_count),
                "sidecar_sha256": str(metadata["sidecar_sha256"]),
            }
        )

    payload: dict[str, object] = {
        "schema_version": 1,
        "spec_id": SEQUENCE_INDEX_SPEC_ID,
        "policy_spec_id": POLICY_SPEC_ID,
        "source_root": str(root),
        "source_index_sha256": sha256_file(root / "index.json"),
        "sidecar_root": str(sidecar_root),
        "sidecar_manifest_sha256": sha256_file(sidecar_root / "manifest.json"),
        "split_manifest": str(split_manifest),
        "split_manifest_sha256": sha256_file(split_manifest),
        "policy_admission": str(policy_admission),
        "policy_admission_sha256": sha256_file(policy_admission),
        "history_size": config.history_size,
        "horizon": config.horizon,
        "sequence_steps": int(sequence_steps),
        "policy_stride": int(policy_stride),
        "anchor_stride": int(anchor_stride),
        "history_stride_raw": int(history_stride_raw),
        "condition_modes": list(CONDITION_MODES),
        "splits": descriptors,
        "counts": {
            split: {
                "episodes": len(values),
                "sequences": sum(int(value["anchor_count"]) for value in values),
            }
            for split, values in descriptors.items()
        },
        "skipped": skipped,
        "test_locked_used": False,
    }
    unsigned = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["index_sha256"] = hashlib.sha256(unsigned).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    return payload


class Sage3DEndToEndSequenceDataset(Dataset):
    """Read-only raw-RGB policy sequences with four deterministic input modes."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        split: str,
        config: ArchitectureV1Config | None = None,
        modes: Sequence[str] = CONDITION_MODES,
        max_episodes: int | None = None,
        history_stride_raw: int | None = None,
        visibility_balance_index: str | Path | None = None,
        visibility_negative_repeats: int = 0,
        small_bbox_repeats: int = 0,
    ) -> None:
        self.config = config or ArchitectureV1Config()
        self.config.validate()
        self.index_path = Path(index_path).expanduser().resolve(strict=True)
        value = _load_json(self.index_path)
        if not isinstance(value, Mapping) or value.get("schema_version") != 1:
            raise ValueError("Architecture v1 sequence index schema mismatch")
        unsigned = dict(value)
        expected_hash = unsigned.pop("index_sha256", None)
        actual_hash = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if expected_hash != actual_hash:
            raise ValueError("Architecture v1 sequence index checksum mismatch")
        if value.get("spec_id") != SEQUENCE_INDEX_SPEC_ID:
            raise ValueError("Architecture v1 sequence index spec mismatch")
        if value.get("test_locked_used") is not False or split == "test_locked":
            raise ValueError("NEXT-026 must not use test_locked")
        if int(value["history_size"]) != self.config.history_size:
            raise ValueError("sequence index history size mismatch")
        if int(value["horizon"]) != self.config.horizon:
            raise ValueError("sequence index waypoint horizon mismatch")
        indexed_history_stride = int(value.get("history_stride_raw", 1))
        if history_stride_raw is not None and indexed_history_stride != int(
            history_stride_raw
        ):
            raise ValueError("sequence index history stride mismatch")
        self.history_stride_raw = indexed_history_stride
        self.root = Path(str(value["source_root"])).resolve(strict=True)
        self.sidecar_root = Path(str(value["sidecar_root"])).resolve(strict=True)
        checks = (
            (self.root / "index.json", value["source_index_sha256"]),
            (self.sidecar_root / "manifest.json", value["sidecar_manifest_sha256"]),
            (Path(str(value["split_manifest"])), value["split_manifest_sha256"]),
            (Path(str(value["policy_admission"])), value["policy_admission_sha256"]),
        )
        for path, expected in checks:
            if sha256_file(path.resolve(strict=True)) != expected:
                raise ValueError(f"Architecture v1 sequence index source changed: {path}")
        load_policy_admission(
            Path(str(value["policy_admission"])).resolve(strict=True), self.root
        )
        self.sequence_steps = int(value["sequence_steps"])
        self.policy_stride = int(value["policy_stride"])
        self.anchor_stride = int(value["anchor_stride"])
        requested_modes = tuple(str(mode) for mode in modes)
        if not requested_modes or any(mode not in CONDITION_MODES for mode in requested_modes):
            raise ValueError("unsupported or empty Architecture v1 condition modes")
        self.modes = requested_modes
        descriptors = list(value["splits"].get(split, []))
        if max_episodes is not None:
            descriptors = descriptors[: int(max_episodes)]
        if not descriptors:
            raise ValueError(f"no Architecture v1 sequences for split={split}")
        self.descriptors = descriptors
        self._ends: list[int] = []
        total = 0
        for descriptor in self.descriptors:
            total += int(descriptor["anchor_count"])
            self._ends.append(total)
        self.split = str(split)
        self.index_sha256 = str(expected_hash)
        self.base_length = self._ends[-1]
        self.visibility_negative_indices: tuple[int, ...] = ()
        self.small_bbox_indices: tuple[int, ...] = ()
        repeats = int(visibility_negative_repeats)
        bbox_repeats = int(small_bbox_repeats)
        if repeats < 0 or bbox_repeats < 0:
            raise ValueError("balancing repeat counts must be non-negative")
        self.visibility_negative_repeats = repeats
        self.small_bbox_repeats = bbox_repeats
        if repeats or bbox_repeats:
            if visibility_balance_index is None:
                raise ValueError("visibility balancing requires an audited index")
            balance_path = Path(visibility_balance_index).expanduser().resolve(strict=True)
            balance = _load_json(balance_path)
            if balance.get("test_locked_used") is not False:
                raise ValueError("visibility balance index used test_locked")
            if Path(str(balance.get("index", ""))).resolve() != self.index_path:
                raise ValueError("visibility balance index source mismatch")
            negative = tuple(
                int(value)
                for value in balance["splits"][self.split]["negative_global_indices"]
            )
            if not negative or min(negative) < 0 or max(negative) >= self.base_length:
                raise ValueError("visibility balance indices are empty or out of range")
            self.visibility_negative_indices = negative
            small_bbox = tuple(
                int(value)
                for value in balance["splits"][self.split]["small_bbox_global_indices"]
            )
            if bbox_repeats and (
                not small_bbox
                or min(small_bbox) < 0
                or max(small_bbox) >= self.base_length
            ):
                raise ValueError("small-bbox balance indices are empty or out of range")
            self.small_bbox_indices = small_bbox

    def __len__(self) -> int:
        return self.base_length + (
            len(self.visibility_negative_indices) * self.visibility_negative_repeats
        ) + (
            len(self.small_bbox_indices) * self.small_bbox_repeats
        )

    def _locate(self, index: int) -> tuple[Mapping[str, object], int, str]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        forced_mode = None
        if index >= self.base_length:
            extra_index = index - self.base_length
            negative_extra = (
                len(self.visibility_negative_indices)
                * self.visibility_negative_repeats
            )
            if extra_index < negative_extra:
                index = self.visibility_negative_indices[
                    extra_index % len(self.visibility_negative_indices)
                ]
            else:
                bbox_extra_index = extra_index - negative_extra
                index = self.small_bbox_indices[
                    bbox_extra_index % len(self.small_bbox_indices)
                ]
            forced_mode = "visual_only"
        descriptor_index = bisect.bisect_right(self._ends, index)
        start = self._ends[descriptor_index - 1] if descriptor_index else 0
        descriptor = self.descriptors[descriptor_index]
        anchor = int(descriptor["anchor_start"]) + (
            index - start
        ) * self.anchor_stride
        if anchor > int(descriptor["anchor_end"]):
            raise RuntimeError("sequence index anchor mapping overflow")
        mode = forced_mode or self.modes[index % len(self.modes)]
        return descriptor, anchor, mode

    @lru_cache(maxsize=8)
    def _source(
        self, relative: str, sidecar_sha256: str
    ) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]], Mapping[str, object]]:
        episode = _safe_episode(self.root, relative)
        sidecar_path = episode_sidecar_path(self.sidecar_root, relative)
        if sha256_file(sidecar_path) != sidecar_sha256:
            raise ValueError(f"SAGE3D sidecar changed after indexing: {relative}")
        sidecar = validate_episode_sidecar(_load_json(sidecar_path), relative)
        derived_path = episode / "derived.json"
        camera_path = episode / "camera_info.json"
        if sha256_file(derived_path) != sidecar["source_derived_sha256"]:
            raise ValueError(f"SAGE3D derived file changed: {relative}")
        if sha256_file(camera_path) != sidecar["source_camera_info_sha256"]:
            raise ValueError(f"SAGE3D camera file changed: {relative}")
        derived = _load_json(derived_path)
        camera = _load_json(camera_path)
        return derived["steps"], sidecar["steps"], camera

    def __getitem__(self, index: int) -> dict[str, object]:
        descriptor, anchor, mode = self._locate(index)
        relative = str(descriptor["relative"])
        steps, labels, camera_value = self._source(
            relative, str(descriptor["sidecar_sha256"])
        )
        episode = _safe_episode(self.root, relative)
        camera = camera_value["camera"]
        width, height = float(camera["width"]), float(camera["height"])
        initial_index = int(descriptor["initial_index"])
        initial_bbox = _bbox_norm(
            labels[initial_index].get("bbox_xyxy"), width, height
        )
        if initial_bbox == [0.0, 0.0, 0.0, 0.0]:
            raise ValueError(f"indexed initialization bbox became invalid: {relative}")
        initial_rgb = _rgb_tensor(
            episode / "rgb" / f"{int(steps[initial_index]['step']):05d}.jpg",
            self.config,
        )
        intrinsics, camera_from_base = sage_camera_calibration(
            camera_value, self.config
        )

        visual = mode in {"visual_uwb", "visual_only"}
        uwb = mode in {"visual_uwb", "uwb_only"}
        sequence_anchors = [
            anchor + step * self.policy_stride for step in range(self.sequence_steps)
        ]
        history = []
        target_waypoints = []
        target_waypoint_yaws = []
        target_bbox = []
        target_visible = []
        target_xy = []
        ego_motion = []
        for current_anchor in sequence_anchors:
            history_indices = list(
                range(
                    current_anchor
                    - (self.config.history_size - 1) * self.history_stride_raw,
                    current_anchor + 1,
                    self.history_stride_raw,
                )
            )
            history.append(
                torch.stack(
                    [
                        _rgb_tensor(
                            episode
                            / "rgb"
                            / f"{int(steps[history_index]['step']):05d}.jpg",
                            self.config,
                        )
                        for history_index in history_indices
                    ]
                )
            )
            target_waypoints.append(
                [[0.0, 0.0] for _ in range(self.config.horizon)]
                if mode == "safe_stop"
                else canonical_waypoints(steps, current_anchor)
            )
            target_waypoint_yaws.append(
                [0.0 for _ in range(self.config.horizon)]
                if mode == "safe_stop"
                else _canonical_waypoint_yaws(
                    steps, current_anchor, horizon=self.config.horizon
                )
            )
            label = labels[current_anchor]
            visible = bool(label.get("visible") and label.get("bbox_xyxy") is not None)
            target_visible.append(float(visible))
            target_bbox.append(
                _bbox_norm(label.get("bbox_xyxy") if visible else None, width, height)
            )
            target_xy.append(target_position_base(steps[current_anchor]))
            ego_motion.append(
                _relative_motion(steps[current_anchor - 1], steps[current_anchor])
            )

        count = self.sequence_steps
        uwb_xy = torch.tensor(target_xy, dtype=torch.float32)
        if not uwb:
            uwb_xy.zero_()
        identity_label_valid = float(visual or uwb)
        transition_action = [
            _realized_action(steps[previous], steps[current])
            for previous, current in zip(sequence_anchors[:-1], sequence_anchors[1:])
        ]
        binding_target = [
            1.0 if visual else visible if mode == "uwb_only" else 0.0
            for visible in target_visible
        ]
        direct_action_target = []
        for current_anchor in sequence_anchors:
            if mode == "safe_stop":
                direct_action_target.append([0.0, 0.0, 0.0])
                continue
            first_future = canonical_waypoints(steps, current_anchor)[1]
            future_yaw = float(steps[current_anchor + 3]["robot_yaw"])
            current_yaw = float(steps[current_anchor]["robot_yaw"])
            delta_yaw = math.atan2(
                math.sin(future_yaw - current_yaw),
                math.cos(future_yaw - current_yaw),
            )
            direct_action_target.append(
                [first_future[0] / 0.1, first_future[1] / 0.1, delta_yaw / 0.1]
            )
        return {
            "initial_rgb": initial_rgb,
            "initial_bbox": torch.tensor(
                initial_bbox if visual else [0.0, 0.0, 0.0, 0.0],
                dtype=torch.float32,
            ),
            "ego_rgb": torch.stack(history),
            "visual_initialization_valid": torch.tensor(
                float(visual), dtype=torch.float32
            ),
            "rgb_valid": torch.ones(count, dtype=torch.float32),
            "binding_valid": torch.full(
                (count,), float(visual), dtype=torch.float32
            ),
            "uwb_xy": uwb_xy,
            "uwb_covariance_xy": torch.eye(2, dtype=torch.float32)[None].repeat(
                count, 1, 1
            )
            * (0.2**2 if uwb else 0.0),
            "uwb_quality": torch.full(
                (count,), float(uwb), dtype=torch.float32
            ),
            "uwb_age_s": torch.zeros(count, dtype=torch.float32),
            "uwb_valid": torch.full((count,), float(uwb), dtype=torch.float32),
            "camera_intrinsics": intrinsics,
            "camera_from_base": camera_from_base,
            "target_waypoints": torch.tensor(
                target_waypoints, dtype=torch.float32
            ),
            "target_waypoint_yaws": torch.tensor(
                target_waypoint_yaws, dtype=torch.float32
            ),
            "waypoint_mask": torch.ones(
                count, self.config.horizon, dtype=torch.bool
            ),
            "stop_target": torch.full(
                (count,), float(mode == "safe_stop"), dtype=torch.float32
            ),
            "direct_action_target": torch.tensor(
                direct_action_target, dtype=torch.float32
            ),
            "target_bbox": torch.tensor(target_bbox, dtype=torch.float32),
            "target_visible": torch.tensor(target_visible, dtype=torch.float32),
            "identity_label_valid": torch.full(
                (count,), identity_label_valid, dtype=torch.float32
            ),
            "binding_target": torch.tensor(binding_target, dtype=torch.float32),
            "ego_motion_target": torch.tensor(ego_motion, dtype=torch.float32),
            "target_xy_label": torch.tensor(target_xy, dtype=torch.float32),
            "transition_action": torch.tensor(
                transition_action, dtype=torch.float32
            ).reshape(count - 1, 3),
            "transition_valid": torch.ones(count - 1, dtype=torch.float32),
            "future_target_xy": torch.tensor(target_xy[1:], dtype=torch.float32),
            "future_target_xy_valid": torch.full(
                (count - 1,), identity_label_valid, dtype=torch.float32
            ),
            "future_target_visible": torch.tensor(
                target_visible[1:], dtype=torch.float32
            ),
            "future_target_visibility_valid": torch.full(
                (count - 1,), identity_label_valid, dtype=torch.float32
            ),
            "mode_index": torch.tensor(
                CONDITION_MODES.index(mode), dtype=torch.long
            ),
            "condition_mode": mode,
            "sample_id": (
                f"sage3d_extracted/{relative}/anchor-{anchor}/sequence-{count}/{mode}"
            ),
        }
