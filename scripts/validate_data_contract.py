#!/usr/bin/env python3
"""Validate the frozen WP-1 contract and JSON/JSONL clip records read-only."""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator


TOP_LEVEL_KEYS = {
    "schema_version",
    "sample_id",
    "sample_role",
    "source",
    "model_inputs",
    "routing_metadata",
    "supervision",
    "provenance",
}
SOURCE_KEYS = {
    "dataset_id",
    "split_unit_id",
    "episode_id",
    "anchor_index",
    "adapter_version",
}
MODEL_INPUT_KEYS = {
    "condition_mode",
    "anchor_timestamp_ns",
    "rgb_sensor_valid",
    "rgb_history",
    "visual_initialization",
    "uwb_target",
}
RGB_FRAME_KEYS = {"rgb_path", "timestamp_ns", "valid"}
VISUAL_INITIALIZATION_KEYS = {"valid", "rgb_path", "bbox_xyxy_norm", "timestamp_ns"}
UWB_KEYS = {
    "valid",
    "measurement_kind",
    "relative_position_base_xy_m",
    "covariance_base_xy_m2",
    "quality_01",
    "source_timestamp_ns",
    "receive_timestamp_ns",
    "age_s",
    "los_state",
}
ROUTING_KEYS = {"target_tag_id", "calibration_id", "simulation_spec_id"}
SUPERVISION_KEYS = {"expert_trajectory", "auxiliary_labels", "safety"}
EXPERT_KEYS = {"valid", "waypoints_base_xy_m", "time_offsets_s", "valid_mask"}
AUXILIARY_KEYS = {
    "target_track_id",
    "target_bbox_xyxy_norm",
    "target_visible",
    "target_position_base_xy_m",
    "occlusion_state",
    "uwb_error_base_xy_m",
}
SAFETY_KEYS = {"stop_required", "reason"}
PROVENANCE_KEYS = {
    "source_record",
    "transform_spec_id",
    "clock_spec_id",
    "generation_spec_id",
}


class ContractViolation(ValueError):
    """A deterministic, user-facing contract validation failure."""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractViolation(f"{where} must be an object")
    return value


def _exact_keys(value: Any, expected: set[str], where: str) -> dict[str, Any]:
    obj = _object(value, where)
    actual = set(obj)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ContractViolation(f"{where} keys differ; missing={missing}, extra={extra}")
    return obj


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ContractViolation(f"{where} must be boolean")
    return value


def _integer(value: Any, where: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractViolation(f"{where} must be an integer")
    if minimum is not None and value < minimum:
        raise ContractViolation(f"{where} must be >= {minimum}")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractViolation(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractViolation(f"{where} must be finite")
    return result


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractViolation(f"{where} must be a non-empty string")
    return value


def _optional_string(value: Any, where: str) -> str | None:
    if value is None:
        return None
    return _string(value, where)


def _enum(value: Any, choices: Iterable[str], where: str) -> str:
    result = _string(value, where)
    allowed = set(choices)
    if result not in allowed:
        raise ContractViolation(f"{where} must be one of {sorted(allowed)}, found {result!r}")
    return result


def _optional_enum(value: Any, choices: Iterable[str], where: str) -> str | None:
    if value is None:
        return None
    return _enum(value, choices, where)


def _pair(value: Any, where: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ContractViolation(f"{where} must be a two-element array")
    return [_number(item, f"{where}[{index}]") for index, item in enumerate(value)]


def _bbox(value: Any, where: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ContractViolation(f"{where} must be a four-element normalized xyxy array")
    x0, y0, x1, y1 = [
        _number(item, f"{where}[{index}]") for index, item in enumerate(value)
    ]
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ContractViolation(f"{where} must satisfy 0 <= x0 < x1 <= 1 and y likewise")
    return [x0, y0, x1, y1]


def _relative_path(value: Any, where: str) -> str:
    result = _string(value, where)
    if "\\" in result:
        raise ContractViolation(f"{where} must use POSIX separators")
    path = PurePosixPath(result)
    if (
        path.is_absolute()
        or (path.parts and path.parts[0].endswith(":"))
        or path.as_posix() != result
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ContractViolation(f"{where} must be a normalized relative path without '..'")
    return result


def _walk_forbidden(value: Any, forbidden: set[str], where: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in forbidden:
                raise ContractViolation(f"{where}.{key} is prohibited in model_inputs")
            _walk_forbidden(child, forbidden, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_forbidden(child, forbidden, f"{where}[{index}]")


def validate_contract(contract: Any) -> dict[str, Any]:
    obj = _object(contract, "contract")
    required = {
        "schema_version",
        "contract_id",
        "status",
        "scope",
        "model_input_policy",
        "coordinates",
        "time",
        "sample_schema",
        "missing_values",
        "source_adapters",
        "frozen_gates",
    }
    missing = sorted(required - set(obj))
    if missing:
        raise ContractViolation(f"contract is missing keys: {missing}")
    if _integer(obj["schema_version"], "contract.schema_version") != 1:
        raise ContractViolation("only contract schema_version=1 is supported")
    _string(obj["contract_id"], "contract.contract_id")

    policy = _object(obj["model_input_policy"], "contract.model_input_policy")
    if policy.get("natural_language_allowed") is not False:
        raise ContractViolation("contract must prohibit natural-language model input")
    if policy.get("external_bbox_after_history_index") != 0:
        raise ContractViolation("only history index zero may carry an external bbox")
    if policy.get("target_tag_id_is_routing_only") is not True:
        raise ContractViolation("target_tag_id must remain routing-only")
    if set(policy.get("allowed_top_level_keys", [])) != MODEL_INPUT_KEYS:
        raise ContractViolation("contract model-input key set does not match validator v1")
    forbidden = {str(key).lower() for key in policy.get("forbidden_keys_recursive", [])}
    mandatory_forbidden = {
        "instruction",
        "desc",
        "target_track_id",
        "target_tag_id",
        "per_frame_bbox",
        "target_bbox",
        "target_visible",
    }
    if not mandatory_forbidden.issubset(forbidden):
        raise ContractViolation("contract omits mandatory prohibited model-input keys")

    coordinates = _object(obj["coordinates"], "contract.coordinates")
    base = _object(coordinates.get("base_frame"), "contract.coordinates.base_frame")
    expected_base = {
        "name": "base_at_observation",
        "x_axis": "forward",
        "y_axis": "left",
        "z_axis": "up",
        "linear_unit": "metre",
        "yaw_positive": "counter_clockwise_about_positive_z",
        "yaw_unit": "radian",
    }
    for key, expected in expected_base.items():
        if base.get(key) != expected:
            raise ContractViolation(f"contract base-frame {key} must be {expected!r}")
    bbox = _object(coordinates.get("bbox"), "contract.coordinates.bbox")
    if bbox.get("order") != ["x0", "y0", "x1", "y1"] or bbox.get("range") != [0.0, 1.0]:
        raise ContractViolation("contract bbox must be normalized xyxy")
    trajectory = _object(
        coordinates.get("expert_trajectory"), "contract.coordinates.expert_trajectory"
    )
    if trajectory.get("frame") != "base_at_observation":
        raise ContractViolation("expert trajectory must use the anchor base frame")
    if trajectory.get("representation") != "absolute_positions_not_deltas":
        raise ContractViolation("expert trajectory must contain absolute local positions")
    if trajectory.get("horizon_points") != 8 or trajectory.get("includes_anchor_point") is not True:
        raise ContractViolation("validator v1 requires eight points including the anchor")

    time_contract = _object(obj["time"], "contract.time")
    expected_time = {
        "timestamp_type": "non_negative_integer_nanoseconds",
        "clock_domain": "episode_monotonic_after_adapter_synchronization",
        "history_order": "strictly_increasing",
    }
    for key, expected in expected_time.items():
        if time_contract.get(key) != expected:
            raise ContractViolation(f"contract time field {key} must be {expected!r}")

    schema = _object(obj["sample_schema"], "contract.sample_schema")
    if set(schema.get("required_top_level_keys", [])) != TOP_LEVEL_KEYS:
        raise ContractViolation("contract top-level sample keys do not match validator v1")
    for name in (
        "sample_roles",
        "dataset_ids",
        "condition_modes",
        "uwb_measurement_kinds",
        "los_states",
        "occlusion_states",
        "safety_reasons",
    ):
        values = schema.get(name)
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ContractViolation(f"contract.sample_schema.{name} must be a unique list")

    missing_values = _object(obj["missing_values"], "contract.missing_values")
    if missing_values.get("nan_or_infinity_allowed") is not False:
        raise ContractViolation("NaN and infinity must remain prohibited")
    if missing_values.get("serialization") != "null_plus_explicit_valid_or_valid_mask":
        raise ContractViolation("missing values must use null plus a validity indicator")

    adapters = _object(obj["source_adapters"], "contract.source_adapters")
    dataset_ids = set(schema["dataset_ids"])
    if set(adapters) != dataset_ids:
        raise ContractViolation("source_adapters must exactly cover sample_schema.dataset_ids")
    roles = set(schema["sample_roles"])
    for dataset_id, adapter_value in adapters.items():
        adapter = _object(adapter_value, f"contract.source_adapters.{dataset_id}")
        eligible = adapter.get("eligible_roles")
        if not isinstance(eligible, list) or not set(eligible).issubset(roles):
            raise ContractViolation(f"invalid eligible_roles for {dataset_id}")
    mandatory_admission = {
        "intern_data_n1": [],
        "sage3d_extracted": ["identity_auxiliary", "policy"],
        "tpt_bench_clean_v2": ["identity_auxiliary"],
        "habitat_sim": ["policy", "identity_auxiliary"],
        "real_robot": ["policy", "identity_auxiliary"],
    }
    for dataset_id, expected in mandatory_admission.items():
        if adapters[dataset_id]["eligible_roles"] != expected:
            raise ContractViolation(
                f"contract v1 admission gate changed for {dataset_id}; expected {expected}"
            )
    return obj


def _validate_covariance(value: Any, where: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 2:
        raise ContractViolation(f"{where} must be a 2x2 array")
    rows = [_pair(row, f"{where}[{index}]") for index, row in enumerate(value)]
    a, b = rows[0]
    c, d = rows[1]
    tolerance = 1e-9
    if abs(b - c) > tolerance:
        raise ContractViolation(f"{where} must be symmetric")
    if a < 0.0 or d < 0.0 or a * d - b * c < -tolerance:
        raise ContractViolation(f"{where} must be positive semidefinite")
    return rows


def _validate_history(model: dict[str, Any]) -> list[dict[str, Any]]:
    history_value = model["rgb_history"]
    if not isinstance(history_value, list) or len(history_value) < 2:
        raise ContractViolation("model_inputs.rgb_history must contain at least two frames")
    history = []
    previous_timestamp = None
    for index, frame_value in enumerate(history_value):
        where = f"model_inputs.rgb_history[{index}]"
        frame = _exact_keys(frame_value, RGB_FRAME_KEYS, where)
        valid = _bool(frame["valid"], f"{where}.valid")
        timestamp = _integer(frame["timestamp_ns"], f"{where}.timestamp_ns", 0)
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ContractViolation("rgb_history timestamps must be strictly increasing")
        previous_timestamp = timestamp
        if valid:
            _relative_path(frame["rgb_path"], f"{where}.rgb_path")
        elif frame["rgb_path"] is not None:
            raise ContractViolation(f"{where}.rgb_path must be null when valid=false")
        history.append(frame)

    anchor = _integer(model["anchor_timestamp_ns"], "model_inputs.anchor_timestamp_ns", 0)
    if anchor != history[-1]["timestamp_ns"]:
        raise ContractViolation("anchor_timestamp_ns must equal the final RGB timestamp")
    sensor_valid = _bool(model["rgb_sensor_valid"], "model_inputs.rgb_sensor_valid")
    if sensor_valid and not history[-1]["valid"]:
        raise ContractViolation("a valid RGB sensor requires a valid anchor frame")
    if not sensor_valid and history[-1]["valid"]:
        raise ContractViolation("RGB sensor failure requires an invalid anchor frame")
    return history


def _validate_visual_initialization(
    model: dict[str, Any], history: list[dict[str, Any]]
) -> bool:
    visual = _exact_keys(
        model["visual_initialization"],
        VISUAL_INITIALIZATION_KEYS,
        "model_inputs.visual_initialization",
    )
    valid = _bool(visual["valid"], "model_inputs.visual_initialization.valid")
    payload_keys = ("rgb_path", "bbox_xyxy_norm", "timestamp_ns")
    if not valid:
        if any(visual[key] is not None for key in payload_keys):
            raise ContractViolation("invalid visual initialization must have a null payload")
        return False

    path = _relative_path(
        visual["rgb_path"], "model_inputs.visual_initialization.rgb_path"
    )
    _bbox(visual["bbox_xyxy_norm"], "model_inputs.visual_initialization.bbox_xyxy_norm")
    timestamp = _integer(
        visual["timestamp_ns"], "model_inputs.visual_initialization.timestamp_ns", 0
    )
    first = history[0]
    if not first["valid"] or path != first["rgb_path"] or timestamp != first["timestamp_ns"]:
        raise ContractViolation("visual initialization must bind to RGB history index zero")
    return True


def _validate_uwb(
    model: dict[str, Any], routing: dict[str, Any], schema: dict[str, Any]
) -> bool:
    uwb = _exact_keys(model["uwb_target"], UWB_KEYS, "model_inputs.uwb_target")
    valid = _bool(uwb["valid"], "model_inputs.uwb_target.valid")
    kind = _enum(
        uwb["measurement_kind"],
        schema["uwb_measurement_kinds"],
        "model_inputs.uwb_target.measurement_kind",
    )
    calibration_id = _optional_string(
        routing["calibration_id"], "routing_metadata.calibration_id"
    )
    simulation_spec_id = _optional_string(
        routing["simulation_spec_id"], "routing_metadata.simulation_spec_id"
    )
    target_tag_id = _optional_string(
        routing["target_tag_id"], "routing_metadata.target_tag_id"
    )
    if kind == "real_uwb" and calibration_id is None:
        raise ContractViolation("real_uwb requires routing_metadata.calibration_id")
    if kind == "simulated_uwb" and simulation_spec_id is None:
        raise ContractViolation("simulated_uwb requires routing_metadata.simulation_spec_id")
    if kind == "none" and any(
        value is not None for value in (target_tag_id, calibration_id, simulation_spec_id)
    ):
        raise ContractViolation("measurement_kind=none requires null UWB routing metadata")

    payload_keys = (
        "relative_position_base_xy_m",
        "covariance_base_xy_m2",
        "quality_01",
        "source_timestamp_ns",
        "receive_timestamp_ns",
        "age_s",
        "los_state",
    )
    if not valid:
        if any(uwb[key] is not None for key in payload_keys):
            raise ContractViolation("invalid UWB must have a null measurement payload")
        return False
    if kind == "none" or target_tag_id is None:
        raise ContractViolation("valid UWB requires a stream kind and target_tag_id")

    _pair(
        uwb["relative_position_base_xy_m"],
        "model_inputs.uwb_target.relative_position_base_xy_m",
    )
    covariance = uwb["covariance_base_xy_m2"]
    quality = uwb["quality_01"]
    if covariance is None and quality is None:
        raise ContractViolation("valid UWB requires covariance or quality_01")
    if covariance is not None:
        _validate_covariance(covariance, "model_inputs.uwb_target.covariance_base_xy_m2")
    if quality is not None:
        quality_value = _number(quality, "model_inputs.uwb_target.quality_01")
        if not 0.0 <= quality_value <= 1.0:
            raise ContractViolation("model_inputs.uwb_target.quality_01 must be in [0, 1]")
    source_timestamp = _integer(
        uwb["source_timestamp_ns"], "model_inputs.uwb_target.source_timestamp_ns", 0
    )
    receive_timestamp = _integer(
        uwb["receive_timestamp_ns"], "model_inputs.uwb_target.receive_timestamp_ns", 0
    )
    anchor = model["anchor_timestamp_ns"]
    if not source_timestamp <= receive_timestamp <= anchor:
        raise ContractViolation("UWB timestamps must satisfy source <= receive <= anchor")
    age = _number(uwb["age_s"], "model_inputs.uwb_target.age_s")
    expected_age = (anchor - source_timestamp) / 1_000_000_000.0
    if age < 0.0 or not math.isclose(age, expected_age, rel_tol=1e-6, abs_tol=1e-9):
        raise ContractViolation("UWB age_s is inconsistent with anchor and source timestamps")
    _enum(uwb["los_state"], schema["los_states"], "model_inputs.uwb_target.los_state")
    return True


def _validate_expert(
    expert_value: Any, role: str, horizon: int, provenance: dict[str, Any]
) -> None:
    expert = _exact_keys(expert_value, EXPERT_KEYS, "supervision.expert_trajectory")
    valid = _bool(expert["valid"], "supervision.expert_trajectory.valid")
    waypoints = expert["waypoints_base_xy_m"]
    mask = expert["valid_mask"]
    if not isinstance(waypoints, list) or len(waypoints) != horizon:
        raise ContractViolation(f"expert waypoints must contain exactly {horizon} entries")
    if not isinstance(mask, list) or len(mask) != horizon or not all(
        isinstance(item, bool) for item in mask
    ):
        raise ContractViolation(f"expert valid_mask must contain exactly {horizon} booleans")

    if not valid:
        if expert["time_offsets_s"] is not None or any(mask) or any(
            waypoint is not None for waypoint in waypoints
        ):
            raise ContractViolation("invalid expert trajectory must be wholly null and masked")
        if role == "policy":
            raise ContractViolation("policy samples require a valid expert trajectory")
        return

    offsets = expert["time_offsets_s"]
    if not isinstance(offsets, list) or len(offsets) != horizon:
        raise ContractViolation(f"expert time_offsets_s must contain exactly {horizon} entries")
    offset_values = [
        _number(value, f"supervision.expert_trajectory.time_offsets_s[{index}]")
        for index, value in enumerate(offsets)
    ]
    if not math.isclose(offset_values[0], 0.0, abs_tol=1e-12) or any(
        right <= left for left, right in zip(offset_values, offset_values[1:])
    ):
        raise ContractViolation("expert offsets must start at zero and strictly increase")
    valid_count = 0
    saw_invalid = False
    for index, (waypoint, point_valid) in enumerate(zip(waypoints, mask)):
        where = f"supervision.expert_trajectory.waypoints_base_xy_m[{index}]"
        if point_valid:
            if saw_invalid:
                raise ContractViolation("expert valid_mask must be a contiguous valid prefix")
            _pair(waypoint, where)
            valid_count += 1
        else:
            saw_invalid = True
            if waypoint is not None:
                raise ContractViolation(f"{where} must be null when masked invalid")
    anchor_waypoint = _pair(
        waypoints[0], "supervision.expert_trajectory.waypoints_base_xy_m[0]"
    )
    if not all(math.isclose(value, 0.0, abs_tol=1e-9) for value in anchor_waypoint):
        raise ContractViolation("expert waypoint zero must be the [0, 0] anchor pose")
    if role != "policy":
        raise ContractViolation("identity_auxiliary samples cannot carry expert trajectories")
    if valid_count < 2:
        raise ContractViolation("policy samples need at least two valid expert points")
    if provenance["transform_spec_id"] is None or provenance["clock_spec_id"] is None:
        raise ContractViolation("policy samples require transform_spec_id and clock_spec_id")


def _validate_auxiliary(auxiliary_value: Any, schema: dict[str, Any]) -> None:
    auxiliary = _exact_keys(
        auxiliary_value, AUXILIARY_KEYS, "supervision.auxiliary_labels"
    )
    _optional_string(
        auxiliary["target_track_id"], "supervision.auxiliary_labels.target_track_id"
    )
    visible = auxiliary["target_visible"]
    if visible is not None:
        visible = _bool(visible, "supervision.auxiliary_labels.target_visible")
    bbox = auxiliary["target_bbox_xyxy_norm"]
    if bbox is not None:
        _bbox(bbox, "supervision.auxiliary_labels.target_bbox_xyxy_norm")
    occlusion = _optional_enum(
        auxiliary["occlusion_state"],
        schema["occlusion_states"],
        "supervision.auxiliary_labels.occlusion_state",
    )
    if visible is True and (bbox is None or occlusion != "visible"):
        raise ContractViolation("visible targets require a bbox and occlusion_state=visible")
    if visible is not True and bbox is not None:
        raise ContractViolation("non-visible or unknown targets must serialize bbox as null")
    if visible is not True and occlusion == "visible":
        raise ContractViolation(
            "non-visible or unknown target conflicts with occlusion_state=visible"
        )
    if auxiliary["target_position_base_xy_m"] is not None:
        _pair(
            auxiliary["target_position_base_xy_m"],
            "supervision.auxiliary_labels.target_position_base_xy_m",
        )
    if auxiliary["uwb_error_base_xy_m"] is not None:
        _pair(
            auxiliary["uwb_error_base_xy_m"],
            "supervision.auxiliary_labels.uwb_error_base_xy_m",
        )


def _validate_mode_and_safety(
    model: dict[str, Any],
    visual_valid: bool,
    uwb_valid: bool,
    safety_value: Any,
    schema: dict[str, Any],
) -> str:
    mode = _enum(
        model["condition_mode"], schema["condition_modes"], "model_inputs.condition_mode"
    )
    safety = _exact_keys(safety_value, SAFETY_KEYS, "supervision.safety")
    stop_required = _bool(safety["stop_required"], "supervision.safety.stop_required")
    reason = _enum(safety["reason"], schema["safety_reasons"], "supervision.safety.reason")
    rgb_valid = model["rgb_sensor_valid"]
    requirements = {
        "visual_uwb": (True, True, True),
        "visual_only": (True, False, True),
        "uwb_only": (False, True, True),
    }
    if mode in requirements:
        expected = requirements[mode]
        if (visual_valid, uwb_valid, rgb_valid) != expected:
            raise ContractViolation(
                f"{mode} requires visual/UWB/RGB validity {expected}, found "
                f"{(visual_valid, uwb_valid, rgb_valid)}"
            )
        if stop_required or reason != "none":
            raise ContractViolation(f"{mode} requires stop_required=false and reason=none")
    else:
        if not stop_required or reason == "none":
            raise ContractViolation("safe_stop requires stop_required=true and a non-none reason")
    if not rgb_valid and (mode != "safe_stop" or reason != "rgb_sensor_failure"):
        raise ContractViolation("RGB sensor failure must map to safe_stop/rgb_sensor_failure")
    return mode


def _check_file(root: Path, relative: str, where: str) -> None:
    resolved_root = root.expanduser().resolve(strict=True)
    unresolved = resolved_root / PurePosixPath(relative)
    if unresolved.is_symlink():
        raise ContractViolation(f"{where} is a symbolic link: {unresolved}")
    candidate = unresolved.resolve(strict=False)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ContractViolation(f"{where} escapes dataset root") from error
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        raise ContractViolation(f"{where} is missing, empty, or a symbolic link: {candidate}")


def _check_paths(sample: dict[str, Any], root: Path) -> None:
    model = sample["model_inputs"]
    for index, frame in enumerate(model["rgb_history"]):
        if frame["valid"]:
            _check_file(root, frame["rgb_path"], f"rgb_history[{index}].rgb_path")
    visual = model["visual_initialization"]
    if visual["valid"]:
        _check_file(root, visual["rgb_path"], "visual_initialization.rgb_path")
    _check_file(root, sample["provenance"]["source_record"], "provenance.source_record")


def validate_sample(
    sample: Any,
    contract: dict[str, Any],
    data_roots: dict[str, Path] | None = None,
) -> dict[str, str]:
    obj = _exact_keys(sample, TOP_LEVEL_KEYS, "sample")
    if _integer(obj["schema_version"], "sample.schema_version") != contract["schema_version"]:
        raise ContractViolation("sample schema_version does not match the contract")
    _string(obj["sample_id"], "sample.sample_id")
    schema = contract["sample_schema"]
    role = _enum(obj["sample_role"], schema["sample_roles"], "sample.sample_role")

    source = _exact_keys(obj["source"], SOURCE_KEYS, "sample.source")
    dataset_id = _enum(
        source["dataset_id"], schema["dataset_ids"], "sample.source.dataset_id"
    )
    _string(source["split_unit_id"], "sample.source.split_unit_id")
    _string(source["episode_id"], "sample.source.episode_id")
    _integer(source["anchor_index"], "sample.source.anchor_index", 0)
    _string(source["adapter_version"], "sample.source.adapter_version")
    eligible = contract["source_adapters"][dataset_id]["eligible_roles"]
    if role not in eligible:
        raise ContractViolation(
            f"source {dataset_id} is not admitted for role {role}; eligible_roles={eligible}"
        )

    model = _exact_keys(obj["model_inputs"], MODEL_INPUT_KEYS, "sample.model_inputs")
    forbidden = {
        str(key).lower()
        for key in contract["model_input_policy"]["forbidden_keys_recursive"]
    }
    _walk_forbidden(model, forbidden, "model_inputs")
    history = _validate_history(model)

    routing = _exact_keys(obj["routing_metadata"], ROUTING_KEYS, "sample.routing_metadata")
    visual_valid = _validate_visual_initialization(model, history)
    uwb_valid = _validate_uwb(model, routing, schema)

    provenance = _exact_keys(obj["provenance"], PROVENANCE_KEYS, "sample.provenance")
    _relative_path(provenance["source_record"], "sample.provenance.source_record")
    for key in ("transform_spec_id", "clock_spec_id", "generation_spec_id"):
        _optional_string(provenance[key], f"sample.provenance.{key}")
    if uwb_valid and (
        provenance["transform_spec_id"] is None or provenance["clock_spec_id"] is None
    ):
        raise ContractViolation("valid UWB requires transform_spec_id and clock_spec_id")
    if role == "policy" and dataset_id in {"habitat_sim", "real_robot"}:
        if provenance["generation_spec_id"] is None:
            raise ContractViolation(f"{dataset_id} policy samples require generation_spec_id")

    supervision = _exact_keys(obj["supervision"], SUPERVISION_KEYS, "sample.supervision")
    horizon = contract["coordinates"]["expert_trajectory"]["horizon_points"]
    _validate_expert(supervision["expert_trajectory"], role, horizon, provenance)
    _validate_auxiliary(supervision["auxiliary_labels"], schema)
    mode = _validate_mode_and_safety(
        model, visual_valid, uwb_valid, supervision["safety"], schema
    )

    if data_roots and dataset_id in data_roots:
        _check_paths(obj, data_roots[dataset_id])
    return {"dataset_id": dataset_id, "sample_role": role, "condition_mode": mode}


def iter_sample_records(path: Path) -> Iterator[tuple[Any, str]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    yield json.loads(line), f"{path}:{line_number}"
        return
    value = load_json(path)
    if isinstance(value, list):
        for index, record in enumerate(value):
            yield record, f"{path}[{index}]"
    else:
        yield value, str(path)


def parse_data_roots(values: list[str], contract: dict[str, Any]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    allowed = set(contract["sample_schema"]["dataset_ids"])
    for value in values:
        if "=" not in value:
            raise ContractViolation(f"--data-root must be DATASET_ID=PATH, found {value!r}")
        dataset_id, raw_path = value.split("=", 1)
        if dataset_id not in allowed:
            raise ContractViolation(f"unknown --data-root dataset: {dataset_id}")
        if dataset_id in roots:
            raise ContractViolation(f"duplicate --data-root dataset: {dataset_id}")
        root = Path(raw_path).expanduser()
        if not root.is_dir():
            raise ContractViolation(f"data root is not a directory: {root}")
        roots[dataset_id] = root
    return roots


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract", type=Path, default=Path("configs/data_contract.json")
    )
    parser.add_argument("--samples", type=Path, action="append", default=[])
    parser.add_argument(
        "--data-root",
        action="append",
        default=[],
        metavar="DATASET_ID=PATH",
        help="optionally check only paths referenced by records for this dataset",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        contract = validate_contract(load_json(args.contract))
        data_roots = parse_data_roots(args.data_root, contract)
        counts: dict[str, collections.Counter[str]] = {
            "dataset": collections.Counter(),
            "role": collections.Counter(),
            "condition_mode": collections.Counter(),
        }
        sample_ids: set[str] = set()
        total = 0
        for sample_path in args.samples:
            for record, location in iter_sample_records(sample_path):
                try:
                    result = validate_sample(record, contract, data_roots)
                except (ContractViolation, KeyError, TypeError) as error:
                    raise ContractViolation(f"{location}: {error}") from error
                sample_id = record["sample_id"]
                if sample_id in sample_ids:
                    raise ContractViolation(f"{location}: duplicate sample_id {sample_id!r}")
                sample_ids.add(sample_id)
                total += 1
                counts["dataset"][result["dataset_id"]] += 1
                counts["role"][result["sample_role"]] += 1
                counts["condition_mode"][result["condition_mode"]] += 1
        report = {
            "contract_id": contract["contract_id"],
            "contract_schema_version": contract["schema_version"],
            "sample_files": len(args.samples),
            "samples_valid": total,
            "counts": {name: dict(sorted(counter.items())) for name, counter in counts.items()},
            "checked_data_roots": sorted(data_roots),
            "read_only": True,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (ContractViolation, FileNotFoundError, json.JSONDecodeError) as error:
        print(f"data-contract validation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
