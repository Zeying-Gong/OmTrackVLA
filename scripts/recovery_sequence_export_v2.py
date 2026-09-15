"""Candidate-only exporter for actual-time recovery replay sequences.

This companion module has no Habitat/GPU dependency. Install beside the new
probe script. Privileged pose/action audits remain outside model_inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


WAYPOINT_TIMES_S = tuple(index / 10.0 for index in range(8))


def world_time_s(env: Any) -> float:
    value = float(env.sim.get_world_time())
    if not math.isfinite(value):
        raise ValueError("simulator world time is not finite")
    return value


def audit_state(env: Any, environment_step: int) -> dict[str, object]:
    """Read-only label/audit state; never used to choose a replay action."""
    robot = env.sim.agents_mgr[1].articulated_agent
    target = env.sim.agents_mgr[0].articulated_agent
    return {
        "environment_step": int(environment_step),
        "world_time_s": world_time_s(env),
        "robot_world_xyz_m": np.asarray(robot.base_pos, dtype=np.float64).tolist(),
        "target_world_xyz_m": np.asarray(target.base_pos, dtype=np.float64).tolist(),
        "robot_transform_world": np.asarray(robot.sim_obj.transformation, dtype=np.float64).tolist(),
    }


def _time_axis(records: Sequence[Mapping[str, object]]) -> np.ndarray:
    times = np.asarray([record["world_time_s"] for record in records], dtype=np.float64)
    if len(times) < 2 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("observed world-time records must be finite and strictly increasing")
    return times


def replay_policy_calls(
    prefix: Sequence[Mapping[str, object]], *, anchor_step: int, history_size: int,
) -> list[dict[str, object]]:
    """Build deployment histories from reset through the anchor, never future RGB."""
    if anchor_step < 1 or history_size < 1:
        raise ValueError("positive anchor_step and history_size are required")
    indices = [record.get("environment_step") for record in prefix]
    if indices != list(range(anchor_step + 1)):
        raise ValueError("prefix must contain exactly reset..anchor, without gaps or future frames")
    _time_axis(prefix)
    result = []
    for index, record in enumerate(prefix):
        history = [max(0, value) for value in range(index - history_size + 1, index + 1)]
        result.append({
            "policy_call_index": index,
            "observation_environment_step": index,
            "world_time_s": float(record["world_time_s"]),
            "rgb_history_observation_indices": history,
            "supervision_valid": index == anchor_step,
            "action_provenance": "saved_policy_replay" if index < anchor_step else "expert_relabel_anchor",
        })
    return result


def resample_expert_positions(
    states: Sequence[Mapping[str, object]], *, anchor_world_time_s: float,
    anchor_position: Sequence[float], forward_axis: Sequence[float], left_axis: Sequence[float],
) -> dict[str, object]:
    """Interpolate recorded positions at 0,.1,...,.7 seconds; forbid extrapolation."""
    times = _time_axis(states)
    if abs(float(times[0]) - float(anchor_world_time_s)) > 1e-8:
        raise ValueError("expert first state must share the replay anchor world time")
    elapsed = times - float(anchor_world_time_s)
    query = np.asarray(WAYPOINT_TIMES_S, dtype=np.float64)
    if elapsed[0] > query[0] or elapsed[-1] + 1e-9 < query[-1]:
        raise ValueError("actual expert timeline does not cover every required waypoint time")
    origin, forward, left = (np.asarray(value, dtype=np.float64) for value in (anchor_position, forward_axis, left_axis))
    if any(value.shape != (3,) or not np.isfinite(value).all() for value in (origin, forward, left)):
        raise ValueError("anchor origin and axes must be finite three-vectors")

    def interpolate(key):
        xyz = np.asarray([record[key] for record in states], dtype=np.float64)
        if xyz.shape != (len(states), 3) or not np.isfinite(xyz).all():
            raise ValueError(f"{key} must contain finite XYZ at every recorded world time")
        return np.stack([np.interp(query, elapsed, xyz[:, axis]) for axis in range(3)], axis=-1)

    robot, target = interpolate("robot_world_xyz_m"), interpolate("target_world_xyz_m")
    if np.max(np.abs(robot[0] - origin)) > 1e-6:
        raise ValueError("expert initial robot position does not match the anchor")
    delta = robot - origin
    waypoints = np.stack((delta @ forward, delta @ left), axis=-1)
    return {
        "waypoint_times_s": query.tolist(),
        "waypoints_base_xy_m": waypoints.tolist(),
        "valid_mask": [True] * 8,
        "resampled_robot_world_xyz_m": robot.tolist(),
        "resampled_target_world_xyz_m": target.tolist(),
        "raw_observed_elapsed_s": elapsed.tolist(),
        "observed_horizon_s": float(elapsed[-1]),
        "resampling": "piecewise_linear_world_xyz_at_recorded_world_time",
        "fixed_frequency_assumed": False,
        "extrapolation_used": False,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_candidate_sequence(
    *, directory: Path, source_rollout: Path, report_output: Path,
    result: Mapping[str, Any], initial_rgb: np.ndarray,
    prefix_frames: Sequence[tuple[int, np.ndarray]], prefix_states: Sequence[Mapping[str, object]],
    initial_bbox_xyxy_norm: Sequence[float], camera_intrinsics: np.ndarray,
    camera_from_base: np.ndarray, expert_states: Sequence[Mapping[str, object]],
    expert_actions: Sequence[Mapping[str, object]], replay_actions: Sequence[Sequence[float]],
    anchor_position: Sequence[float], forward_axis: Sequence[float], left_axis: Sequence[float],
    checkpoint_step: int, history_size: int, legacy_quality_passed: bool,
    original_sample: Path | None = None,
) -> tuple[Path, str]:
    """Persist schema 2 after structural/time checks; formal eligibility stays false."""
    if result.get("split") != "train":
        raise ValueError("sequence candidate export accepts EVT train only")
    if [step for step, _ in prefix_frames] != list(range(checkpoint_step + 1)):
        raise ValueError("RGB prefix must contain reset..anchor only")
    calls = replay_policy_calls(prefix_states, anchor_step=checkpoint_step, history_size=history_size)
    if len(replay_actions) != checkpoint_step:
        raise ValueError("saved replay actions must end exactly at the anchor")
    if len(expert_actions) != len(expert_states) - 1:
        raise ValueError("every expert state transition must retain its actual action")
    for index, action in enumerate(expert_actions):
        if action.get("expert_step") != index or (
            abs(float(action["world_time_s_before"]) - float(expert_states[index]["world_time_s"])) > 1e-8
            or abs(float(action["world_time_s_after"]) - float(expert_states[index + 1]["world_time_s"])) > 1e-8
        ):
            raise ValueError("expert actions must align with observed before/after world times")
    trajectory = resample_expert_positions(
        expert_states, anchor_world_time_s=float(prefix_states[-1]["world_time_s"]),
        anchor_position=anchor_position, forward_axis=forward_axis, left_axis=left_axis,
    )
    reference = {"path": None, "sha256": None, "status": "not_supplied"}
    if original_sample is not None:
        original_sample = original_sample.expanduser().resolve(strict=True)
        original = json.loads(original_sample.read_text(encoding="utf-8"))
        source = original.get("source", {})
        if source.get("split") != "train" or str(source.get("episode_id")) != str(result["episode_id"]):
            raise ValueError("original sample reference must match this train episode")
        if str(source.get("scene_id")) != str(result["scene_id"]) or int(source.get("dataset_index", -1)) != int(result["dataset_index"]):
            raise ValueError("original sample reference must match this scene and dataset index")
        if int(source.get("anchor_environment_step", -1)) != checkpoint_step:
            raise ValueError("original sample reference must match this anchor")
        reference = {"path": str(original_sample), "sha256": _sha256(original_sample), "status": "linked_only_not_admission"}
    directory = directory.expanduser().resolve(strict=False)
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite sequence candidate: {directory}")
    directory.mkdir(parents=True)
    initial_path = directory / "initial_rgb.png"
    Image.fromarray(np.asarray(initial_rgb, dtype=np.uint8)).save(initial_path)
    images = []
    for index, frame in prefix_frames:
        path = directory / f"prefix_{index:04d}.png"
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(path)
        images.append({
            "environment_step": int(index), "rgb_path": path.name,
            "sha256": _sha256(path), "world_time_s": float(prefix_states[index]["world_time_s"]),
        })
    sample = {
        "schema_version": 2, "stage": "phase3_recovery_sequence_candidate_v2",
        "status": "candidate_pending_independent_admission",
        "sample_id": f"evt_train/{result['task']}/{result['episode_id']}/post-action-{checkpoint_step:03d}/sequence-v2",
        "formal_training_eligible": False,
        "training_restriction": "pending_independent_time_contract_quality_and_dataset_split_admission",
        "test_locked_used": False,
        "source": {
            "dataset": "EVT-Bench", "split": "train", "task": str(result["task"]),
            "dataset_index": int(result["dataset_index"]), "episode_id": str(result["episode_id"]),
            "scene_id": str(result["scene_id"]), "anchor_environment_step": checkpoint_step,
            "rollout_result": str(source_rollout), "rollout_result_sha256": _sha256(source_rollout),
            "relabel_report": str(report_output), "original_sample_reference": reference,
            "source_policy_checkpoint_metadata": {key: value for key, value in result.items() if "checkpoint" in key},
            "evt_bench_used_for_waypoint_supervision": True,
            "expert": "OracleNavmeshFollowerV6", "test_locked_used": False,
        },
        "sequence_contract": {
            "starts_at_episode_reset": True, "policy_call_indices": list(range(checkpoint_step + 1)),
            "anchor_policy_call_index": checkpoint_step, "history_size": history_size,
            "startup_padding": "repeat_reset_rgb_on_left",
            "initial_state": {"hidden": "zero", "target_memory": "initialize_from_initial_rgb_bbox", "binding_valid": True},
            "state_reconstruction": "replay_every_policy_call_with_current_training_weights",
            "expert_future_rgb_in_model_inputs": False,
            "policy_interval_source": "observed_simulator_world_time",
            "world_time_reader": "env.sim.get_world_time", "fixed_frequency_assumed": False,
        },
        "model_inputs": {
            "condition_mode": "visual_only",
            "initial_rgb": {"rgb_path": initial_path.name, "sha256": _sha256(initial_path)},
            "initial_bbox_xyxy_norm": list(initial_bbox_xyxy_norm),
            "prefix_rgb": images, "policy_calls": calls,
            "visual_initialization_valid": True, "rgb_valid": True, "binding_valid": True,
            "uwb": {"valid": False, "relative_position_base_xy_m": [0., 0.], "covariance_base_xy_m2": [[1., 0.], [0., 1.]], "quality_01": 0., "age_s": 0.},
            "camera_intrinsics": np.asarray(camera_intrinsics).tolist(),
            "camera_from_base": np.asarray(camera_from_base).tolist(),
        },
        "supervision": {
            "anchor_policy_call_index": checkpoint_step, "supervision_mask": [False] * checkpoint_step + [True],
            "expert_trajectory": trajectory,
            "stop_label_available": False, "visibility_label_available": False,
            "binding_label_available": False, "ego_motion_label_available": False,
            "gt_used_only_on_label_or_audit_side": True, "later_bbox_used_by_model": False,
        },
        "audit_raw": {
            "prefix_states": list(prefix_states), "saved_policy_replay_actions": [list(action) for action in replay_actions],
            "expert_states": list(expert_states), "expert_actions": list(expert_actions),
            "legacy_step_stride_quality_passed": bool(legacy_quality_passed),
            "legacy_passed_is_not_formal_admission": True,
        },
    }
    path = directory / "sample.json"
    path.write_text(json.dumps(sample, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path, _sha256(path)
