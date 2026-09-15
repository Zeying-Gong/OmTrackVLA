"""Fail-closed schema-2 recovery validation and candidate sequence loading.

Validation does not import torch or run a simulator. It verifies the saved
evidence, recomputes physical-time labels and legacy quality metrics, and
returns content hashes for independent scene admission. A successful check is
NOT formal training admission. Camera/collision evidence remains an audit
measurement, not an independent rerun of physics.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


class RecoveryValidationError(ValueError):
    """The candidate or its provenance does not satisfy the v2 contract."""


def _require(condition, message):
    if not condition:
        raise RecoveryValidationError(message)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path):
    def reject(value):
        raise RecoveryValidationError(f"nonfinite JSON constant: {value}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject, object_pairs_hook=unique)
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _contained(value, *, base, root):
    _require(isinstance(value, str) and bool(value), "nonempty artifact path required")
    # Reject locked partitions before attempting to read their files.
    _require("test_locked" not in value.lower().replace("\\", "/"), "test_locked artifact path prohibited")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve(strict=True)
    _require(path.is_relative_to(root) and path.is_file(), "artifact path escapes permitted root or is not a file")
    return path


def _artifact(path, expected=None):
    actual = _sha256(path)
    if expected is not None:
        _require(isinstance(expected, str) and len(expected) == 64 and actual == expected,
                 f"SHA-256 mismatch: {path.name}")
    return {"path": str(path), "sha256": actual}


def _array(value, shape, name):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise RecoveryValidationError(f"{name} must be numeric") from exc
    _require(array.shape == shape and np.isfinite(array).all(), f"{name} must be finite {shape}")
    return array


def _close(actual, expected, name, tolerance=1e-7):
    target = np.asarray(expected, dtype=np.float64)
    value = _array(actual, target.shape, name)
    _require(np.allclose(value, target, rtol=0., atol=tolerance), f"{name} mismatch")
    return value


def _number(value, name):
    _require(type(value) in (int, float) and np.isfinite(value), f"{name} must be finite numeric")
    return float(value)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, f"{name} must be integer >= {minimum}")
    return value


def _flags(value, true=(), false=()):
    for key in true:
        _require(value.get(key) is True, f"{key} must be true")
    for key in false:
        _require(value.get(key) is False, f"{key} must be false")


def _keys(value, expected, name):
    _require(isinstance(value, dict) and set(value) == set(expected), f"{name} contains missing or unapproved input fields")


def _states(records, first, count, name):
    _require(isinstance(records, list) and len(records) == count, f"{name} state count mismatch")
    _require([r.get("environment_step") for r in records] == list(range(first, first + count)),
             f"{name} state steps must be contiguous")
    times = _array([r["world_time_s"] for r in records], (count,), f"{name} times")
    _require(count >= 2 and np.all(np.diff(times) > 0), f"{name} times must be strictly increasing")
    robot = _array([r["robot_world_xyz_m"] for r in records], (count, 3), f"{name} robot XYZ")
    target = _array([r["target_world_xyz_m"] for r in records], (count, 3), f"{name} target XYZ")
    for record in records:
        transform = _array(record["robot_transform_world"], (4, 4), f"{name} transform")
        _close(transform[3], [0, 0, 0, 1], f"{name} transform homogeneous row")
        _close(transform[:3, :3].T @ transform[:3, :3], np.eye(3), f"{name} rotation", 2e-5)
    return times, robot, target


def _stats(points, target):
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    stationary = float(np.linalg.norm(target))
    endpoint = float(np.linalg.norm(target - points[-1]))
    radius = float(np.linalg.norm(points[-1]))
    alignment = float(np.dot(points[-1], target) / (radius * stationary)) if radius * stationary > 1e-18 else 0.
    return ({"path_length_m": float(lengths.sum()), "terminal_radius_m": radius,
             "maximum_segment_m": float(lengths.max(initial=0.))},
            {"stationary_anchor_final_target_distance_m": stationary,
             "expert_final_target_distance_m": endpoint,
             "distance_improvement_over_stationary_anchor_m": stationary - endpoint,
             "terminal_motion_target_alignment_cosine": alignment})


def _validate_impl(sample_path, artifact_root):
    path = Path(sample_path).expanduser().resolve(strict=True)
    root = Path(artifact_root).expanduser().resolve(strict=True) if artifact_root else path.parent.parent
    _require(path.is_relative_to(root), "sample path escapes artifact_root")
    _require("test_locked" not in str(path).lower(), "test_locked sample prohibited")
    sample = _load_json(path)
    _require(sample.get("schema_version") == 2, "schema_version must be 2")
    _require(sample.get("stage") == "phase3_recovery_sequence_candidate_v2", "candidate stage mismatch")
    _require(sample.get("status") == "candidate_pending_independent_admission", "candidate status mismatch")
    _flags(sample, false=("formal_training_eligible", "test_locked_used"))
    source = sample["source"]
    _require(source.get("dataset") == "EVT-Bench" and source.get("split") == "train", "source must be EVT-Bench train")
    _flags(source, true=("evt_bench_used_for_waypoint_supervision",), false=("test_locked_used",))
    _require(source.get("expert") == "OracleNavmeshFollowerV6", "source expert mismatch")
    for key in ("task", "episode_id", "scene_id"):
        _require(isinstance(source.get(key), str) and bool(source[key]), f"missing source {key}")
    index = _integer(source["dataset_index"], "dataset_index")
    anchor = _integer(source["anchor_environment_step"], "anchor", 1)
    size = anchor + 1
    _require(sample.get("sample_id") == f"evt_train/{source['task']}/{source['episode_id']}/post-action-{anchor:03d}/sequence-v2", "sample_id/source mismatch")
    rollout_path = _contained(source["rollout_result"], base=path.parent, root=root)
    report_path = _contained(source["relabel_report"], base=path.parent, root=root)
    artifacts = {"sample": _artifact(path), "rollout": _artifact(rollout_path, source["rollout_result_sha256"]),
                 "report": _artifact(report_path), "media": []}
    rollout, report = _load_json(rollout_path), _load_json(report_path)
    for document in (rollout, report):
        _require(document.get("split") == "train", "source document split must be train")
        for key in ("task", "scene_id", "episode_id", "dataset_index"):
            _require(document.get(key) == source[key], f"source/report/rollout {key} mismatch")
        _require(document.get("test_locked_used", False) is False, "test_locked document prohibited")
    _require(report.get("schema_version") == 2 and report.get("status") == "passed", "report schema/status did not pass")
    _require(report.get("stage") == "next007_recovery_sequence_probe_v2", "report stage mismatch")
    _flags(report, false=("formal_training_eligible",))
    _require(_contained(report["source_rollout"], base=report_path.parent, root=root) == rollout_path, "report rollout path mismatch")
    _require(report["model_visited_checkpoint_step"] == anchor, "report anchor mismatch")
    linked = report["phase3_recovery_sequence_candidate"]
    _require(_contained(linked["manifest"], base=report_path.parent, root=root) == path, "report linked sample mismatch")
    _require(linked["manifest_sha256"] == artifacts["sample"]["sha256"], "report sample SHA-256 mismatch")
    _require(linked.get("prefix_environment_steps") == list(range(size)), "report prefix steps mismatch")
    _flags(linked, false=("formal_training_eligible",))
    boundary = report["label_boundary"]
    _flags(boundary, true=("gt_pose_used_only_in_audit_and_supervision", "replay_actions_never_use_audit_pose", "navmesh_used_only_by_expert"),
           false=("expert_waypoint_entered_deployment_policy", "test_locked_used"))
    _require(boundary.get("source_policy_input_audit") == rollout.get("input_audit"), "source policy input audit mismatch")
    input_audit = boundary["source_policy_input_audit"]
    _flags(input_audit, true=("passed",), false=("depth_used", "gt_target_point_used", "later_bbox_used", "perception_cache_used"))
    _require(input_audit.get("decision_parameters") == ["rgb", "uwb"], "source decision interface is not RGB/UWB")
    _require(set(input_audit.get("model_input_keys", [])) == {
        "initial_rgb", "initial_bbox", "ego_rgb", "visual_initialization_valid", "rgb_valid", "binding_valid",
        "uwb_xy", "uwb_covariance_xy", "uwb_quality", "uwb_age_s", "uwb_valid", "camera_intrinsics",
        "camera_from_base", "hidden_state", "_target_memory_override"}, "source model input whitelist mismatch")
    reference = source["original_sample_reference"]
    if reference.get("status") == "not_supplied":
        _require(reference.get("path") is None and reference.get("sha256") is None, "invalid absent original reference")
    else:
        _require(reference.get("status") == "linked_only_not_admission", "original reference must not grant admission")
        original_path = _contained(reference["path"], base=path.parent, root=root)
        artifacts["original_sample"] = _artifact(original_path, reference["sha256"])
        original_document = _load_json(original_path)
        original = original_document["source"]
        _flags(original, false=("test_locked_used",))
        for key in ("split", "episode_id", "scene_id", "dataset_index", "anchor_environment_step"):
            _require(original.get(key) == source[key], f"original reference {key} mismatch")
        if original_document.get("schema_version") == 1:
            # Legacy schema 1 encoded the task in sample_id and did not have
            # source.task. Check the COMPLETE ID rather than guessing by path.
            expected_id = f"habitat/{source['task']}/{source['episode_id']}/post-action-{anchor:03d}"
            _require(original_document.get("sample_id") == expected_id, "original reference schema-1 sample_id/task mismatch")
            _require(original.get("task", source["task"]) == source["task"], "original reference task mismatch")
        else:
            _require(original.get("task") == source["task"], "original reference task mismatch")

    contract, inputs, supervision, audit = (sample[k] for k in ("sequence_contract", "model_inputs", "supervision", "audit_raw"))
    history = _integer(contract["history_size"], "history_size", 1)
    _flags(contract, true=("starts_at_episode_reset",), false=("expert_future_rgb_in_model_inputs", "fixed_frequency_assumed"))
    for key, expected in {"policy_call_indices": list(range(size)), "anchor_policy_call_index": anchor,
        "startup_padding": "repeat_reset_rgb_on_left", "state_reconstruction": "replay_every_policy_call_with_current_training_weights",
        "policy_interval_source": "observed_simulator_world_time", "world_time_reader": "env.sim.get_world_time",
        "initial_state": {"hidden": "zero", "target_memory": "initialize_from_initial_rgb_bbox", "binding_valid": True}}.items():
        _require(contract.get(key) == expected, f"sequence contract {key} mismatch")
    _keys(inputs, ("condition_mode", "initial_rgb", "initial_bbox_xyxy_norm", "prefix_rgb", "policy_calls",
          "visual_initialization_valid", "rgb_valid", "binding_valid", "uwb", "camera_intrinsics", "camera_from_base"), "model_inputs")
    _require(inputs["condition_mode"] == "visual_only", "v2 loader accepts visual_only candidates")
    _flags(inputs, true=("visual_initialization_valid", "rgb_valid", "binding_valid"))
    _keys(inputs["uwb"], ("valid", "relative_position_base_xy_m", "covariance_base_xy_m2", "quality_01", "age_s"), "uwb")
    _flags(inputs["uwb"], false=("valid",))
    for key, expected in {"relative_position_base_xy_m": [0., 0.], "covariance_base_xy_m2": [[1., 0.], [0., 1.]], "quality_01": 0., "age_s": 0.}.items():
        _close(inputs["uwb"][key], expected, f"unavailable UWB {key}")
    bbox = _array(inputs["initial_bbox_xyxy_norm"], (4,), "initial_bbox")
    _require((bbox >= 0).all() and (bbox <= 1).all() and bbox[0] < bbox[2] and bbox[1] < bbox[3], "invalid initial bbox")
    intrinsics = _array(inputs["camera_intrinsics"], (3, 3), "camera_intrinsics")
    _require(intrinsics[0, 0] > 0 and intrinsics[1, 1] > 0, "camera focal lengths must be positive")
    _close(intrinsics[2], [0, 0, 1], "camera intrinsics last row")
    # This exporter uses a centered Habitat 90-degree camera; without that
    # contract the stored file does not identify its calibration image size.
    _close(intrinsics, [[intrinsics[0, 2], 0, intrinsics[0, 2]],
                        [0, intrinsics[1, 2], intrinsics[1, 2]], [0, 0, 1]], "centered Habitat calibration")
    camera = _array(inputs["camera_from_base"], (4, 4), "camera_from_base")
    _close(camera[3], [0, 0, 0, 1], "camera homogeneous row")
    _close(camera[:3, :3].T @ camera[:3, :3], np.eye(3), "camera rotation", 2e-5)

    prefix = inputs["prefix_rgb"]
    _require(isinstance(prefix, list) and [r.get("environment_step") for r in prefix] == list(range(size)), "RGB prefix must contain exactly reset..anchor without gaps or future frames")
    times, prefix_robot, prefix_target = _states(audit["prefix_states"], 0, size, "prefix")
    _close([r["world_time_s"] for r in prefix], times, "RGB timestamps", 1e-9)
    media_paths = []
    native_size = None
    for rgb in [inputs["initial_rgb"]] + prefix:
        allowed = ("rgb_path", "sha256") if rgb is inputs["initial_rgb"] else ("rgb_path", "sha256", "environment_step", "world_time_s")
        _keys(rgb, allowed, "RGB metadata")
        media_path = _contained(rgb["rgb_path"], base=path.parent, root=path.parent)
        artifacts["media"].append(_artifact(media_path, rgb["sha256"]))
        with Image.open(media_path) as image:
            _require(image.format == "PNG" and image.mode == "RGB", "RGB evidence must be RGB PNG")
            image.load()
            if native_size is None:
                native_size = image.size
            _require(image.size == native_size, "RGB native dimensions changed within sequence")
        media_paths.append(media_path)
    _require(inputs["initial_rgb"]["sha256"] == prefix[0]["sha256"], "initial RGB must equal reset RGB")
    calls = inputs["policy_calls"]
    _require(isinstance(calls, list) and len(calls) == size, "policy call count mismatch")
    for step, call in enumerate(calls):
        _keys(call, ("policy_call_index", "observation_environment_step", "world_time_s", "rgb_history_observation_indices", "supervision_valid", "action_provenance"), "policy call")
        _require(call["policy_call_index"] == step and call["observation_environment_step"] == step, "policy call index mismatch")
        expected = [max(0, j) for j in range(step-history+1, step+1)]
        _require(call["rgb_history_observation_indices"] == expected, "policy RGB history is missing, reordered or contains future frames")
        _close(call["world_time_s"], times[step], "policy observation time", 1e-9)
        _require(call["supervision_valid"] is (step == anchor), "only anchor may be supervised")
        _require(call["action_provenance"] == ("expert_relabel_anchor" if step == anchor else "saved_policy_replay"), "policy action provenance mismatch")
    _require(supervision["anchor_policy_call_index"] == anchor and supervision["supervision_mask"] == [False]*anchor+[True], "supervision must select anchor only")
    _flags(supervision, true=("gt_used_only_on_label_or_audit_side",),
           false=("later_bbox_used_by_model", "stop_label_available", "visibility_label_available", "binding_label_available", "ego_motion_label_available"))
    _flags(audit, true=("legacy_step_stride_quality_passed", "legacy_passed_is_not_formal_admission"))
    _require(rollout.get("initialization", {}).get("environment_step") == 0, "rollout initialization must be reset")
    original_bbox = _array(rollout["initialization"]["bbox_xyxy"], (4,), "rollout bbox")
    _close(bbox, original_bbox / np.asarray([native_size[0], native_size[1]] * 2), "reset initialization bbox", 1e-6)
    steps = rollout["steps"]
    _require(isinstance(steps, list) and len(steps) >= anchor, "rollout missing replay prefix")
    saved_actions = _array(audit["saved_policy_replay_actions"], (anchor, 3), "saved replay actions")
    for step, record in enumerate(steps[:anchor], 1):
        _require(record["step"] == step, "rollout steps are not contiguous")
        _close(saved_actions[step-1], [record["policy"]["action"][k] for k in ("forward", "lateral", "yaw")], "saved replay action", 1e-9)

    count = _integer(report["expert_branch_steps"], "expert_branch_steps", 1) + 1
    _require(report["total_environment_steps"] == anchor + count - 1 <= report["maximum_safe_total_steps"], "expert branch violates audited episode horizon")
    expert_times, robot, target = _states(audit["expert_states"], anchor, count, "expert")
    _require(audit["expert_states"][0] == audit["prefix_states"][-1], "expert branch does not start at replay anchor")
    actual_time = report["actual_time_audit"]
    _flags(actual_time, true=("evt_bench_train_used_for_waypoint_supervision", "legacy_passed_is_not_formal_admission"), false=("fixed_frequency_assumed",))
    _require(actual_time["world_time_reader"] == "env.sim.get_world_time", "report time source mismatch")
    for key in ("prefix_states", "expert_states"):
        _require(actual_time[key] == audit[key], f"report {key} mismatch")
    elapsed = expert_times - expert_times[0]
    _require(elapsed[0] == 0. and elapsed[-1] >= .7 - 1e-9, "expert timeline cannot cover 0..0.7 s without extrapolation")
    transform = _array(audit["prefix_states"][-1]["robot_transform_world"], (4, 4), "anchor transform")
    forward, left = transform[:3, 0].copy(), -transform[:3, 2].copy()
    forward[1], left[1] = 0., 0.
    _require(np.linalg.norm(forward) > 1e-6 and np.linalg.norm(left) > 1e-6, "degenerate anchor axes")
    forward, left = forward / np.linalg.norm(forward), left / np.linalg.norm(left)
    _close(np.dot(forward, left), 0., "planar axis orthogonality", 2e-5)
    origin = robot[0]
    basis = np.stack([forward, left], axis=1)
    projected_robot, projected_target = (robot-origin) @ basis, (target-origin) @ basis
    grid = np.arange(8, dtype=np.float64)/10.
    resampled_robot = np.stack([np.interp(grid, elapsed, robot[:, k]) for k in range(3)], axis=1)
    resampled_target = np.stack([np.interp(grid, elapsed, target[:, k]) for k in range(3)], axis=1)
    trajectory = supervision["expert_trajectory"]
    _flags(trajectory, false=("fixed_frequency_assumed", "extrapolation_used"))
    _require(trajectory["resampling"] == "piecewise_linear_world_xyz_at_recorded_world_time", "resampling rule mismatch")
    _require(trajectory["valid_mask"] == [True]*8, "all eight physical-time waypoints must be valid")
    for key, expected in {"waypoint_times_s": grid, "raw_observed_elapsed_s": elapsed, "observed_horizon_s": elapsed[-1],
        "resampled_robot_world_xyz_m": resampled_robot, "resampled_target_world_xyz_m": resampled_target,
        "waypoints_base_xy_m": (resampled_robot-origin) @ basis}.items():
        _close(trajectory[key], expected, f"resampled {key}")

    replay = report["replay_audit"]
    _flags(replay, true=("saved_policy_actions_replayed_without_gt", "matches_saved_model_visited_state"))
    tolerance = _number(replay["tolerance"], "replay tolerance")
    _require(0 < tolerance <= 1e-4, "replay tolerance exceeds audited 1e-4 bound")
    for key in ("camera_transform_max_abs_error", "target_distance_abs_error_m"):
        _require(0 <= _number(replay[key], key) <= tolerance, f"replay {key} failed")
    saved_distance = _number(steps[anchor-1]["evaluation_only_after_action"]["gt_distance_m"], "saved target distance")
    distance_error = abs(float(np.linalg.norm(prefix_robot[-1]-prefix_target[-1])) - saved_distance)
    _close(replay["target_distance_abs_error_m"], distance_error, "replay distance error")
    coordinate = report["coordinate_audit"]
    _flags(coordinate, true=("passed",))
    coordinate_tolerance = _number(coordinate["tolerance"], "coordinate tolerance")
    _require(0 < coordinate_tolerance <= 1e-4, "coordinate tolerance exceeds audited bound")
    _close(coordinate["anchor_target_xy_m"], projected_target[0], "anchor target coordinate")
    local = _array(coordinate["habitat_local_target_xy_m"], (2,), "Habitat local target")
    coordinate_error = float(np.max(np.abs(local-projected_target[0])))
    _require(coordinate_error <= coordinate_tolerance, "coordinate check failed")
    _close(coordinate["max_abs_error_m"], coordinate_error, "coordinate error")

    expert = report["expert"]
    _require(expert["controller"] == source["expert"] and expert["waypoint_frame"] == "checkpoint robot base x-forward/y-left", "expert controller/frame mismatch")
    records = audit["expert_actions"]
    _require(records == expert["records"] and len(records) == count-1, "expert actions/report records mismatch")
    limits = expert["actuation_limits"]
    for key in ("max_forward", "max_lateral", "max_yaw", "translation_slew_per_step"):
        _require(_number(limits[key], key) > 0, "invalid expert actuation limits")
    for step, record in enumerate(records):
        _require(record["expert_step"] == step, "expert action indices not contiguous")
        for suffix, index in (("before", step), ("after", step+1)):
            _close(record[f"world_time_s_{suffix}"], expert_times[index], "expert action time", 1e-9)
            _close(record[f"robot_world_xyz_m_{suffix}"], robot[index], "expert action robot pose")
            _close(record[f"target_world_xyz_m_{suffix}"], target[index], "expert action target pose")
        _close(record["robot_anchor_xy_m_before"], projected_robot[step], "expert robot anchor coordinate")
        _close(record["target_anchor_xy_m_before"], projected_target[step], "expert target anchor coordinate")
        for key in ("forward", "lateral", "yaw"):
            _require(abs(_number(record["action"][key], key)) <= float(limits[f"max_{key}"])+1e-6, "expert action exceeds recorded limit")
    forecast = expert["target_forecast"]
    lookahead = _integer(forecast["lookahead_steps"], "lookahead_steps")
    _flags(forecast, true=("deterministic_under_expert_actions",))
    _require(0 <= _number(forecast["same_time_target_max_error_m"], "forecast error") <= tolerance, "target forecast replay failed")
    _require(boundary["future_target_trajectory_used_only_by_label_expert"] is (lookahead > 0), "forecast label boundary mismatch")
    offsets = expert["waypoint_offsets_steps"]
    _require(isinstance(offsets, list) and len(offsets) == 8 and offsets[0] == 0 and offsets[-1] == count-1 and all(type(v) is int for v in offsets) and all(a < b for a, b in zip(offsets, offsets[1:])), "legacy offsets must cover expert branch")
    _flags(expert, true=("legacy_step_stride_waypoints_are_not_v2_training_targets",))
    legacy = projected_robot[offsets]
    _close(expert["waypoints_base_xy_m"], legacy, "legacy waypoint projection")
    _close(expert["legacy_step_stride_offsets_seconds_observed"], elapsed[offsets], "legacy observed offsets")
    _close(expert["initial_target_anchor_xy_m"], projected_target[0], "legacy initial target")
    _close(expert["terminal_target_anchor_xy_m"], projected_target[-1], "legacy terminal target")
    statistics, progress = _stats(legacy, projected_target[-1])
    for key, value in statistics.items():
        _close(expert["trajectory_statistics"][key], value, f"legacy {key}")
    for key, value in progress.items():
        _close(expert["target_progress_statistics"][key], value, f"legacy {key}")
    gate = expert["trajectory_gate"]
    _flags(gate, true=("passed",))
    minimum = _number(gate["minimum_path_m"], "minimum_path_m")
    maximum = _number(gate["maximum_path_m"], "maximum_path_m")
    radius = _number(gate["maximum_terminal_radius_m"], "maximum_terminal_radius_m")
    improvement = _number(gate["minimum_stationary_distance_improvement_m"], "minimum_stationary_distance_improvement_m")
    _require(.05 <= minimum <= maximum <= 2.20 and 0 < radius <= 1.80 and improvement >= .05, "legacy quality thresholds weakened")
    _require(minimum <= statistics["path_length_m"] <= maximum and statistics["terminal_radius_m"] <= radius and progress["distance_improvement_over_stationary_anchor_m"] >= improvement and progress["terminal_motion_target_alignment_cosine"] > 0., "recomputed legacy trajectory quality failed")
    actual_statistics, actual_progress = _stats((resampled_robot-origin) @ basis, (resampled_target[-1]-origin) @ basis)
    _require(minimum <= actual_statistics["path_length_m"] <= maximum
             and actual_statistics["terminal_radius_m"] <= radius
             and actual_progress["distance_improvement_over_stationary_anchor_m"] >= improvement
             and actual_progress["terminal_motion_target_alignment_cosine"] > 0.,
             "recomputed actual-time 0..0.7 s trajectory quality failed")
    _require(_number(expert["collision"], "collision") == 0., "expert collision audit failed")
    return {
        "schema_version": 2, "sample_id": sample["sample_id"], "formal_training_eligible": False,
        "source_split": "train", "test_locked_used": False,
        **{key: source[key] for key in ("scene_id", "task", "episode_id", "dataset_index", "anchor_environment_step")},
        "history_size": history, "sequence_length": size, "native_rgb_size": list(native_size),
        "checks": {key: True for key in ("source_identity", "artifact_hashes", "input_whitelist", "complete_prefix",
            "causal_history", "actual_time_resampling", "anchor_supervision_only", "report_replay", "coordinate_projection", "legacy_quality_recomputed", "actual_time_quality_recomputed")},
        "actual_time_quality": {"trajectory_statistics": actual_statistics, "target_progress_statistics": actual_progress},
        "artifacts": artifacts,
    }


def validate_recovery_sample(sample_path, *, artifact_root=None) -> dict[str, Any]:
    """Return JSON evidence or raise; reads only referenced nonlocked artifacts.

    Media must resolve inside the candidate folder; provenance files must stay
    inside artifact_root (default candidate-folder parent). Supply the project
    root for exports that refer to other output directories in that project.
    """
    try:
        return _validate_impl(sample_path, artifact_root)
    except RecoveryValidationError:
        raise
    except (KeyError, TypeError, IndexError, OSError, ValueError) as exc:
        raise RecoveryValidationError(f"invalid recovery candidate: {exc}") from exc


def _development_admission(manifest_path, role, artifact_root):
    _require(role in ("train", "val"), "development partition_role must be train or val")
    from scripts.build_recovery_scene_manifest import verify_manifest
    manifest, paths = verify_manifest(
        manifest_path, validator=lambda path: validate_recovery_sample(path, artifact_root=artifact_root),
        role=role, for_training=role == "train")
    _require(manifest.get("scope") == "development_pilot"
             and manifest.get("status") == "admitted_for_development_pilot"
             and manifest.get("formal_training_eligible") is False,
             "manifest must explicitly admit only a development pilot")
    return manifest, {Path(path).resolve(strict=True) for path in paths}


def load_recovery_sequence(sample_path, *, image_height=280, image_width=504,
                           artifact_root=None, allow_candidate_for_preflight=False,
                           development_manifest=None, partition_role=None):
    """Build one [S,T,3,H,W] input sequence with only the anchor supervised.

    A verified development manifest admits its declared train/val partition.
    Without it, explicit opt-in allows candidate preflight only. No formal
    training admission exists here. No audit pose, action, future image or later bbox
    crosses the whitelist. Prefix label tensors are zero storage with false
    masks; they are never presented as ground truth. Images match the legacy
    PIL RGB/BILINEAR -> float32 / 255 -> CHW conversion.
    """
    if development_manifest is not None:
        _require(allow_candidate_for_preflight is False, "do not combine development admission with candidate preflight")
        _, paths = _development_admission(development_manifest, partition_role, artifact_root)
        _require(Path(sample_path).resolve(strict=True) in paths, "sample is outside the admitted development partition")
    else:
        _require(allow_candidate_for_preflight is True and partition_role is None,
                 "candidate is not formally admitted; set allow_candidate_for_preflight=True only for preflight")
    _integer(image_height, "image_height", 1)
    _integer(image_width, "image_width", 1)
    evidence = validate_recovery_sample(sample_path, artifact_root=artifact_root)
    return _tensor_sequence(evidence, image_height, image_width)


def _tensor_sequence(evidence, image_height, image_width):
    import torch
    sample = _load_json(Path(evidence["artifacts"]["sample"]["path"]))
    _require(_sha256(Path(evidence["artifacts"]["sample"]["path"])) == evidence["artifacts"]["sample"]["sha256"], "sample changed during loading")
    inputs = sample["model_inputs"]
    def rgb(artifact):
        path = Path(artifact["path"])
        _require(_sha256(path) == artifact["sha256"], "RGB changed during loading")
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB").resize((image_width, image_height), Image.Resampling.BILINEAR), dtype=np.float32).copy()/np.float32(255.)
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()
    initial = rgb(evidence["artifacts"]["media"][0])
    frames = [rgb(value) for value in evidence["artifacts"]["media"][1:]]
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32)
    size, anchor = evidence["sequence_length"], evidence["anchor_environment_step"]
    result = {
        "initial_rgb": initial, "initial_bbox": tensor(inputs["initial_bbox_xyxy_norm"]),
        "ego_rgb": torch.stack([torch.stack([frames[j] for j in call["rgb_history_observation_indices"]]) for call in inputs["policy_calls"]]),
        "visual_initialization_valid": tensor(1.), "rgb_valid": tensor(1.), "binding_valid": tensor(1.),
        "uwb_xy": tensor([0., 0.]), "uwb_covariance_xy": tensor([[1., 0.], [0., 1.]]),
        "uwb_quality": tensor(0.), "uwb_age_s": tensor(0.), "uwb_valid": tensor(0.),
        "camera_intrinsics": tensor(inputs["camera_intrinsics"]), "camera_from_base": tensor(inputs["camera_from_base"]),
        "supervision_mask": torch.zeros(size, dtype=torch.bool),
        "target_waypoints": torch.zeros((size, 8, 2), dtype=torch.float32),
        "waypoint_mask": torch.zeros((size, 8), dtype=torch.bool),
        "stop_label_valid": torch.zeros(size, dtype=torch.bool), "binding_label_valid": torch.zeros(size, dtype=torch.bool),
        "identity_label_valid": torch.zeros(size, dtype=torch.bool), "ego_label_valid": torch.zeros(size, dtype=torch.bool),
    }
    # The exporter stores calibration for its requested model resolution.
    # That resolution is represented by the centered principal point in this
    # Habitat camera contract. Scale K together with any requested test resize.
    calibration_width, calibration_height = 2*float(inputs["camera_intrinsics"][0][2]), 2*float(inputs["camera_intrinsics"][1][2])
    _require(calibration_width > 0 and calibration_height > 0, "centered Habitat calibration size missing")
    result["camera_intrinsics"][0] *= image_width/calibration_width
    result["camera_intrinsics"][1] *= image_height/calibration_height
    result["supervision_mask"][anchor] = True
    result["target_waypoints"][anchor] = tensor(sample["supervision"]["expert_trajectory"]["waypoints_base_xy_m"])
    result["waypoint_mask"][anchor] = True
    return result


def recovery_sequence_collate(samples: Sequence[Mapping[str, Any]]):
    """Stack equal-length real sequences; never invent padding policy calls."""
    import torch
    _require(bool(samples), "cannot collate an empty batch")
    keys = set(samples[0])
    _require(all(set(sample) == keys for sample in samples), "batch keys differ")
    for key in keys:
        _require(all(torch.is_tensor(sample[key]) and sample[key].shape == samples[0][key].shape for sample in samples), f"{key} shapes differ; group equal real sequence lengths instead of padding")
    return {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}


class RecoverySequenceDataset:
    """Map-style, homogeneous real sequences, suitable for torch DataLoader.

    This dataset intentionally has no admission shortcut. The manifest caller
    must independently select its partition and opt into candidate preflight.
    All artifacts are revalidated when read, so later modifications fail closed.
    """

    def __init__(self, sample_paths, *, artifact_root=None, image_height=280,
                 image_width=504, required_anchor=9, allow_candidate_for_preflight=False,
                 development_manifest=None, partition_role=None):
        self.paths = [Path(path).resolve(strict=True) for path in sample_paths]
        _require(bool(self.paths), "recovery dataset must not be empty")
        self.manifest_artifact = None
        if development_manifest is not None:
            _require(allow_candidate_for_preflight is False, "do not combine development admission with candidate preflight")
            _, paths = _development_admission(development_manifest, partition_role, artifact_root)
            _require(set(self.paths).issubset(paths), "dataset samples are outside the admitted development partition")
            self.manifest_artifact = _artifact(Path(development_manifest).resolve(strict=True))
        else:
            _require(allow_candidate_for_preflight is True and partition_role is None,
                     "candidate dataset requires explicit allow_candidate_for_preflight=True")
        _integer(image_height, "image_height", 1)
        _integer(image_width, "image_width", 1)
        self.artifact_root, self.image_height, self.image_width = artifact_root, image_height, image_width
        evidence = [validate_recovery_sample(path, artifact_root=artifact_root) for path in self.paths]
        _require(len({(e["sequence_length"], e["history_size"]) for e in evidence}) == 1,
                 "dataset requires equal real sequence lengths and history sizes")
        if required_anchor is not None:
            _require(all(e["anchor_environment_step"] == required_anchor for e in evidence), "dataset anchor differs from required_anchor")
        _require(len({e["artifacts"]["sample"]["sha256"] for e in evidence}) == len(evidence), "duplicate sample in recovery dataset")
        self.evidence = evidence

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.manifest_artifact is not None:
            _artifact(Path(self.manifest_artifact["path"]), self.manifest_artifact["sha256"])
        expected = self.evidence[index]["artifacts"]["sample"]["sha256"]
        _require(_sha256(self.paths[index]) == expected, "dataset sample changed since construction")
        evidence = validate_recovery_sample(self.paths[index], artifact_root=self.artifact_root)
        _require(evidence["artifacts"] == self.evidence[index]["artifacts"], "dataset evidence changed since admission")
        return _tensor_sequence(evidence, self.image_height, self.image_width)
