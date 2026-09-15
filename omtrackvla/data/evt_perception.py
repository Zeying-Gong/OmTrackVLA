"""Read-only EVT-train perception/Polar data for Architecture v2.

The loader deliberately never exposes teacher actions, future trajectories,
panoptic rasters, or GT poses as model inputs.  GT state is read only while
constructing supervision.  Histories are sampled causally by recorded world
time because the collected EVT frames are not uniformly spaced at exactly
40 Hz.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from omtrackvla.data.end_to_end import _rgb_tensor
from omtrackvla.models.end_to_end import ArchitectureV1Config


TRAIN_AUDIT_KIND = "corrected48_combined_independent_instance_label_audit_v3"
DEV_AUDIT_KIND = "permanent_dev4_corrected_instance_independent_audit_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _reference_path(value: Mapping[str, Any], parent: Path) -> Path:
    path = Path(str(value["path"])).expanduser()
    if not path.is_absolute():
        path = parent / path
    return path.resolve(strict=True)


def _verified_reference(value: Mapping[str, Any], parent: Path) -> Path:
    path = _reference_path(value, parent)
    if int(value.get("bytes", path.stat().st_size)) != path.stat().st_size:
        raise ValueError(f"referenced file size changed: {path}")
    if sha256_file(path) != str(value["sha256"]):
        raise ValueError(f"referenced file hash changed: {path}")
    return path


def causal_history_indices(
    times_s: Sequence[float], anchor: int, *, count: int = 8, interval_s: float = 0.1
) -> tuple[int, ...]:
    """Return causal, left-padded indices at 10 Hz ending at ``anchor``."""

    if count < 2 or interval_s <= 0.0 or anchor < 0 or anchor >= len(times_s):
        raise ValueError("invalid causal history request")
    times = np.asarray(times_s, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("history timestamps must be one finite vector")
    if len(times) > 1 and not bool((np.diff(times) > 0.0).all()):
        raise ValueError("history timestamps must be strictly increasing")
    anchor_time = float(times[anchor])
    result = []
    available = times[: anchor + 1].tolist()
    for offset in reversed(range(count)):
        desired = anchor_time - offset * float(interval_s)
        index = bisect.bisect_right(available, desired) - 1
        result.append(max(0, index))
    result[-1] = anchor
    return tuple(result)


def polar_from_habitat_state(state: Mapping[str, Any]) -> tuple[float, float]:
    """Return ``(angle_rad, distance_m)`` in base forward/left coordinates."""

    transform = np.asarray(state["robot_transform_world"], dtype=np.float64)
    robot = np.asarray(state["robot_world_xyz_m"], dtype=np.float64)
    target = np.asarray(state["target_world_xyz_m"], dtype=np.float64)
    if (
        transform.shape != (4, 4)
        or robot.shape != (3,)
        or target.shape != (3,)
        or not np.isfinite(transform).all()
        or not np.isfinite(robot).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("invalid Habitat state for Polar supervision")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-6):
        raise ValueError("robot transform is not homogeneous")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-5):
        raise ValueError("robot transform rotation is not orthonormal")
    forward = rotation[:, 0]
    left = -rotation[:, 2]
    delta = target - robot
    forward_m = float(delta @ forward)
    left_m = float(delta @ left)
    return math.atan2(left_m, forward_m), math.hypot(forward_m, left_m)


def bbox_bearing_sincos(bbox_xyxy_norm: Sequence[float]) -> tuple[float, float]:
    """Approximate base bearing from a bbox centre for the frozen 90-degree HFOV."""

    bbox = np.asarray(bbox_xyxy_norm, dtype=np.float64)
    if (
        bbox.shape != (4,)
        or not np.isfinite(bbox).all()
        or bbox[2] <= bbox[0]
        or bbox[3] <= bbox[1]
        or (bbox < 0.0).any()
        or (bbox > 1.0).any()
    ):
        raise ValueError("bbox must be valid normalized xyxy")
    centre_x = float((bbox[0] + bbox[2]) * 0.5)
    # Image right is canonical base right, while positive policy angle is left.
    angle = math.atan((0.5 - centre_x) * 2.0)
    return math.sin(angle), math.cos(angle)


@dataclass(frozen=True)
class EVTPerceptionRecord:
    image_path: Path
    time_s: float
    bbox_xyxy_norm: tuple[float, float, float, float]
    visible: bool
    angle_sincos: tuple[float, float]
    angle_valid: bool
    distance_m: float
    distance_valid: bool


@dataclass(frozen=True)
class EVTPerceptionCase:
    case_id: str
    task: str
    initial_image_path: Path
    initial_bbox_xyxy_norm: tuple[float, float, float, float]
    visual_initialization_valid: bool
    records: tuple[EVTPerceptionRecord, ...]

    @property
    def times_s(self) -> tuple[float, ...]:
        return tuple(record.time_s for record in self.records)


def _bbox(target: Mapping[str, Any]) -> tuple[float, float, float, float]:
    value = target.get("bbox_xyxy_norm")
    visible = bool(target.get("visible") and target.get("bbox_label_valid"))
    if not visible or value is None:
        return (0.0, 0.0, 0.0, 0.0)
    bbox = tuple(float(item) for item in value)
    bbox_bearing_sincos(bbox)
    return bbox  # type: ignore[return-value]


def _artifact_path(root: Path, relative: str, artifacts: Mapping[str, Any]) -> Path:
    if relative not in artifacts:
        raise ValueError(f"sealed artifact is missing from manifest: {relative}")
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"sealed artifact escaped case root: {relative}") from error
    return path


def _load_train_cases(
    summary_path: Path, *, polar_valid_only_when_visible: bool
) -> tuple[EVTPerceptionCase, ...]:
    summary = _json(summary_path)
    aggregate = summary.get("aggregate", {})
    if (
        summary.get("kind") != TRAIN_AUDIT_KIND
        or summary.get("corrected_perception_label_candidate") is not True
        or summary.get("candidate_observation_count") != 2548
        or aggregate.get("sealed_raw_verified_episode_count") != 42
        or aggregate.get("planned_episode_denominator") != 48
    ):
        raise ValueError("fixed corrected48 perception audit contract mismatch")
    cases: list[EVTPerceptionCase] = []
    observations = 0
    for entry in summary["entries"]:
        if entry["status"] != "pass":
            continue
        report_path = _verified_reference(entry["case_report"], summary_path.parent)
        report = _json(report_path)
        if (
            report.get("status") != "pass"
            or report.get("validation_flags", {}).get(
                "raw_mask_bbox_visibility_verified"
            )
            is not True
            or report.get("semantic_id_rasters_match_declared_labels") is not True
        ):
            raise ValueError(f"case lacks admitted perception labels: {report_path}")
        supervisor_path = _verified_reference(report["supervisor"], report_path.parent)
        supervisor = _json(supervisor_path)
        artifacts = supervisor["artifact_manifest"]
        root = supervisor_path.parent
        supervision_path = _artifact_path(
            root, "supervision/observations.jsonl", artifacts
        )
        model_index_path = _artifact_path(
            root, "model_inputs/observations.jsonl", artifacts
        )
        initialization_path = (
            _artifact_path(root, "model_inputs/initialization.json", artifacts)
            if "model_inputs/initialization.json" in artifacts
            else None
        )
        metadata = [
            (supervision_path, "supervision/observations.jsonl"),
            (model_index_path, "model_inputs/observations.jsonl"),
        ]
        if initialization_path is not None:
            metadata.append(
                (initialization_path, "model_inputs/initialization.json")
            )
        for path, relative in metadata:
            if sha256_file(path) != artifacts[relative]["sha256"]:
                raise ValueError(f"sealed metadata changed: {path}")
        labels = [json.loads(line) for line in supervision_path.open(encoding="utf-8")]
        model_rows = [json.loads(line) for line in model_index_path.open(encoding="utf-8")]
        expected = int(entry["observations_verified"])
        if len(labels) != expected or len(model_rows) != expected:
            raise ValueError(f"case observation count changed: {root}")
        if initialization_path is not None:
            initialization = _json(initialization_path)
            initial_bbox = tuple(
                float(value) for value in initialization["bbox_xyxy_norm"]
            )
            initial_relative = "model_inputs/" + str(
                initialization["reference_rgb"]
            )
            initialization_valid = True
        else:
            # One admitted reset-only case has no redundant initialization file;
            # its sealed frame-0 label and model-input index carry the same data.
            initial_bbox = _bbox(labels[0]["target"])
            initial_relative = "model_inputs/" + str(model_rows[0]["rgb_file"])
            initialization_valid = initial_bbox != (0.0, 0.0, 0.0, 0.0)
        if initialization_valid:
            bbox_bearing_sincos(initial_bbox)
        initial_image = _artifact_path(root, initial_relative, artifacts)
        records: list[EVTPerceptionRecord] = []
        previous_time = -math.inf
        for index, (label, model_row) in enumerate(zip(labels, model_rows)):
            if (
                int(label["environment_step"]) != index
                or int(model_row["environment_step"]) != index
            ):
                raise ValueError(f"non-contiguous EVT train case: {root}")
            time_s = float(label["world_time_s"])
            if not math.isfinite(time_s) or (index != 0 and time_s <= previous_time):
                raise ValueError(f"non-monotonic EVT train time: {root}")
            previous_time = time_s
            target = label["target"]
            visible = bool(target.get("visible") and target.get("bbox_label_valid"))
            bbox = _bbox(target)
            angle, distance = polar_from_habitat_state(label["state"])
            relative = "model_inputs/" + str(model_row["rgb_file"])
            image_path = _artifact_path(root, relative, artifacts)
            if artifacts[relative]["sha256"] != model_row["rgb_sha256"]:
                raise ValueError(f"RGB manifest/index mismatch: {image_path}")
            records.append(
                EVTPerceptionRecord(
                    image_path=image_path,
                    time_s=time_s,
                    bbox_xyxy_norm=bbox,
                    visible=visible,
                    angle_sincos=(math.sin(angle), math.cos(angle)),
                    angle_valid=(visible if polar_valid_only_when_visible else True),
                    distance_m=distance,
                    distance_valid=(visible if polar_valid_only_when_visible else True),
                )
            )
        cases.append(
            EVTPerceptionCase(
                case_id=str(entry["sample_id"]),
                task=str(entry["task"]).lower(),
                initial_image_path=initial_image,
                initial_bbox_xyxy_norm=initial_bbox,  # type: ignore[arg-type]
                visual_initialization_valid=initialization_valid,
                records=tuple(records),
            )
        )
        observations += len(records)
    if len(cases) != 42 or observations != 2548:
        raise ValueError("fixed corrected48 retained set changed")
    return tuple(cases)


def _load_dev_cases(
    summary_path: Path, *, polar_valid_only_when_visible: bool
) -> tuple[EVTPerceptionCase, ...]:
    summary = _json(summary_path)
    if (
        summary.get("kind") != DEV_AUDIT_KIND
        or summary.get("status") != "admitted_for_fixed_dev_perception_metrics"
        or summary.get("perception_metric_input_allowed") is not True
        or summary.get("optimizer_input_allowed") is not False
        or summary.get("official_evt_validation") is not False
        or summary.get("test_locked_used") is not False
        or summary.get("case_denominator") != 4
        or summary.get("observations") != 229
    ):
        raise ValueError("fixed dev4 perception audit contract mismatch")
    cases: list[EVTPerceptionCase] = []
    for value in summary["cases"]:
        labels_path = _verified_reference(value["labels"], summary_path.parent)
        root = labels_path.parent
        labels = [json.loads(line) for line in labels_path.open(encoding="utf-8")]
        if len(labels) != int(value["observations"]):
            raise ValueError(f"dev case observation count changed: {root}")
        first_bbox = _bbox(labels[0]["target"])
        if first_bbox == (0.0, 0.0, 0.0, 0.0):
            raise ValueError(f"dev initialization bbox is invalid: {root}")
        records: list[EVTPerceptionRecord] = []
        previous_time = -math.inf
        for index, label in enumerate(labels):
            if int(label["environment_step"]) != index:
                raise ValueError(f"non-contiguous fixed dev case: {root}")
            time_s = float(label["world_time_s"])
            if index and time_s <= previous_time:
                raise ValueError(f"non-monotonic fixed dev time: {root}")
            previous_time = time_s
            target = label["target"]
            visible = bool(target.get("visible") and target.get("bbox_label_valid"))
            bbox = _bbox(target)
            angle_sincos = bbox_bearing_sincos(bbox) if visible else (0.0, 1.0)
            distance = float(label["source_audit"]["gt_distance_m"])
            if not math.isfinite(distance) or distance < 0.0:
                raise ValueError(f"invalid fixed dev target distance: {root}")
            relative = Path(str(label["rgb_file"]["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe fixed dev RGB path: {relative}")
            image_path = (root / relative).resolve(strict=True)
            records.append(
                EVTPerceptionRecord(
                    image_path=image_path,
                    time_s=time_s,
                    bbox_xyxy_norm=bbox,
                    visible=visible,
                    angle_sincos=angle_sincos,
                    angle_valid=visible,
                    distance_m=distance,
                    distance_valid=(visible if polar_valid_only_when_visible else True),
                )
            )
        cases.append(
            EVTPerceptionCase(
                case_id=str(value["case_id"]),
                task=str(value["case_id"]).split("_", 1)[0],
                initial_image_path=records[0].image_path,
                initial_bbox_xyxy_norm=first_bbox,
                visual_initialization_valid=True,
                records=tuple(records),
            )
        )
    if sum(len(case.records) for case in cases) != 229:
        raise ValueError("fixed dev4 observation count changed")
    return tuple(cases)


class EVTPerceptionPolarDataset(Dataset):
    """Architecture-v2 visual input with perception/Polar supervision only."""

    def __init__(
        self,
        audit_summary: str | Path,
        *,
        partition: str,
        config: ArchitectureV1Config | None = None,
        history_interval_s: float = 0.1,
        user_authorized_perception_training: bool = False,
        polar_valid_only_when_visible: bool = False,
    ) -> None:
        self.config = config or ArchitectureV1Config(history_size=8)
        self.config.validate()
        if self.config.history_size != 8:
            raise ValueError("EVT Architecture-v2 requires eight history frames")
        self.audit_summary = Path(audit_summary).expanduser().resolve(strict=True)
        if "test_locked" in str(self.audit_summary).lower():
            raise ValueError("test_locked is forbidden")
        if partition == "train":
            if not user_authorized_perception_training:
                raise PermissionError(
                    "corrected48 optimizer use requires explicit perception-only authorization"
                )
            self.cases = _load_train_cases(
                self.audit_summary,
                polar_valid_only_when_visible=polar_valid_only_when_visible,
            )
        elif partition == "dev":
            if user_authorized_perception_training:
                raise ValueError("fixed dev4 must never be optimizer-authorized")
            self.cases = _load_dev_cases(
                self.audit_summary,
                polar_valid_only_when_visible=polar_valid_only_when_visible,
            )
        else:
            raise ValueError("partition must be train or dev")
        self.partition = partition
        self.polar_valid_only_when_visible = bool(polar_valid_only_when_visible)
        self.history_interval_s = float(history_interval_s)
        if self.history_interval_s <= 0.0:
            raise ValueError("history interval must be positive")
        self.samples = tuple(
            (case_index, frame_index)
            for case_index, case in enumerate(self.cases)
            for frame_index in range(len(case.records))
        )
        self.visible_indices = tuple(
            index
            for index, (case_index, frame_index) in enumerate(self.samples)
            if self.cases[case_index].records[frame_index].visible
        )
        visible_set = set(self.visible_indices)
        self.invisible_indices = tuple(
            index for index in range(len(self.samples)) if index not in visible_set
        )

        width = float(self.config.image_width)
        height = float(self.config.image_height)
        self.camera_intrinsics = torch.tensor(
            [[width / 2.0, 0.0, width / 2.0], [0.0, height / 2.0, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        self.camera_from_base = torch.eye(4, dtype=torch.float32)
        self.camera_from_base[:3, :3] = torch.tensor(
            [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.samples)

    @lru_cache(maxsize=96)
    def _image(self, path: Path) -> torch.Tensor:
        return _rgb_tensor(path, self.config)

    def __getitem__(self, index: int) -> dict[str, Any]:
        case_index, frame_index = self.samples[index]
        case = self.cases[case_index]
        record = case.records[frame_index]
        history_indices = causal_history_indices(
            case.times_s,
            frame_index,
            count=self.config.history_size,
            interval_s=self.history_interval_s,
        )
        return {
            "initial_rgb": self._image(case.initial_image_path),
            "initial_bbox": torch.tensor(
                case.initial_bbox_xyxy_norm, dtype=torch.float32
            ),
            "ego_rgb": torch.stack(
                [self._image(case.records[value].image_path) for value in history_indices]
            ),
            "visual_initialization_valid": torch.tensor(
                float(case.visual_initialization_valid)
            ),
            "rgb_valid": torch.tensor(1.0),
            "binding_valid": torch.tensor(1.0),
            # This stage measures visual generalization; UWB cannot leak GT Polar labels.
            "uwb_xy": torch.zeros(2, dtype=torch.float32),
            "uwb_covariance_xy": torch.zeros(2, 2, dtype=torch.float32),
            "uwb_quality": torch.tensor(0.0),
            "uwb_age_s": torch.tensor(0.0),
            "uwb_valid": torch.tensor(0.0),
            "camera_intrinsics": self.camera_intrinsics.clone(),
            "camera_from_base": self.camera_from_base.clone(),
            "target_bbox": torch.tensor(record.bbox_xyxy_norm, dtype=torch.float32),
            "target_visible": torch.tensor(float(record.visible), dtype=torch.float32),
            "target_angle_sincos": torch.tensor(
                record.angle_sincos, dtype=torch.float32
            ),
            "target_angle_valid": torch.tensor(record.angle_valid),
            "target_distance_m": torch.tensor(record.distance_m, dtype=torch.float32),
            "target_distance_valid": torch.tensor(record.distance_valid),
            "case_id": case.case_id,
            "task": case.task,
            "frame_index": torch.tensor(frame_index, dtype=torch.long),
            "history_indices": torch.tensor(history_indices, dtype=torch.long),
        }
