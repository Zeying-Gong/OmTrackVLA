#!/usr/bin/env python3
"""Probe 8-point expert relabeling from a NEXT-027 model-visited state.

The saved policy actions are replayed without consulting privileged state.  Only
after the requested checkpoint is reached does the label-side oracle read the
robot/target poses and navmesh.  This keeps expert geometry out of the deployed
policy boundary while testing whether a Phase-3 DAgger target can be produced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from omtrackvla.evaluation.end_to_end_closed_loop import (
    ACTION_NAMES,
    DEFAULT_SCENE_DATASET,
    PANOPTIC_KEY,
    RGB_KEY,
    _assign_unique_humanoid_semantic_ids,
    _camera_transform,
    configure,
    habitat_camera_calibration,
    normalized_bbox,
    target_mask_to_bbox,
)
def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=28)
    parser.add_argument("--expert-horizon-steps", type=int, default=21)
    parser.add_argument("--waypoint-stride-steps", type=int, default=3)
    parser.add_argument("--maximum-safe-total-steps", type=int, default=50)
    parser.add_argument("--expert-max-forward", type=float, default=0.10)
    parser.add_argument("--expert-max-lateral", type=float, default=0.10)
    parser.add_argument("--expert-max-yaw", type=float, default=0.75)
    parser.add_argument("--expert-translation-slew", type=float, default=0.10)
    parser.add_argument("--expert-target-lookahead-steps", type=int, default=0)
    parser.add_argument("--maximum-trajectory-path-m", type=float, default=2.20)
    parser.add_argument("--maximum-terminal-radius-m", type=float, default=1.80)
    parser.add_argument(
        "--minimum-stationary-distance-improvement-m", type=float, default=0.05
    )
    parser.add_argument("--history-size", type=int, default=4)
    parser.add_argument("--history-stride-steps", type=int, default=1)
    parser.add_argument("--model-image-height", type=int, default=280)
    parser.add_argument("--model-image-width", type=int, default=504)
    parser.add_argument("--phase3-sample-dir", type=Path)
    parser.add_argument("--scene-dataset", default=DEFAULT_SCENE_DATASET)
    parser.add_argument("--replay-tolerance", type=float, default=1e-4)
    return parser.parse_args()


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def replay_actions(value: object) -> list[list[float]]:
    """Extract a finite, contiguous Habitat action prefix from a rollout."""
    if not isinstance(value, Mapping) or not isinstance(value.get("steps"), list):
        raise ValueError("rollout result must contain a steps list")
    actions: list[list[float]] = []
    for expected_step, record in enumerate(value["steps"], 1):
        if not isinstance(record, Mapping) or record.get("step") != expected_step:
            raise ValueError("rollout steps must be contiguous and one-indexed")
        policy = record.get("policy")
        action = policy.get("action") if isinstance(policy, Mapping) else None
        if not isinstance(action, Mapping):
            raise ValueError(f"rollout step {expected_step} has no policy action")
        values = [action.get(key) for key in ("forward", "lateral", "yaw")]
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in values):
            raise ValueError(f"rollout step {expected_step} action is not numeric")
        vector = [float(item) for item in values]
        if not all(math.isfinite(item) for item in vector):
            raise ValueError(f"rollout step {expected_step} action is not finite")
        actions.append(vector)
    return actions


def causal_history_steps(
    checkpoint_step: int, history_size: int, history_stride_steps: int
) -> list[int]:
    """Return a causal fixed-stride history with reset-frame left padding."""

    if checkpoint_step < 0 or history_size <= 0 or history_stride_steps <= 0:
        raise ValueError("history sampling arguments are invalid")
    return [
        max(0, checkpoint_step - history_stride_steps * age)
        for age in reversed(range(history_size))
    ]


def anchor_basis(transform: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical forward/left world axes from a Habitat base transform."""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("anchor transform must be a finite 4x4 matrix")
    # Magnum exposes the mathematical 4x4 transform directly here: the
    # translation is in the last column and the local axes are its columns.
    # Local +x is robot-forward and local -z is robot-left.
    forward = matrix[:3, 0].copy()
    left = -matrix[:3, 2].copy()
    forward[1] = 0.0
    left[1] = 0.0
    forward_norm = float(np.linalg.norm(forward))
    left_norm = float(np.linalg.norm(left))
    if forward_norm < 1e-6 or left_norm < 1e-6:
        raise ValueError("anchor transform has degenerate planar axes")
    return forward / forward_norm, left / left_norm


def position_in_anchor_frame(
    world_position: Sequence[float],
    anchor_position: Sequence[float],
    forward_axis: Sequence[float],
    left_axis: Sequence[float],
) -> list[float]:
    delta = np.asarray(world_position, dtype=np.float64) - np.asarray(
        anchor_position, dtype=np.float64
    )
    forward = np.asarray(forward_axis, dtype=np.float64)
    left = np.asarray(left_axis, dtype=np.float64)
    if delta.shape != (3,) or forward.shape != (3,) or left.shape != (3,):
        raise ValueError("world positions and anchor axes must be 3-vectors")
    result = [float(np.dot(delta, forward)), float(np.dot(delta, left))]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("anchor-frame waypoint is not finite")
    return result


def waypoint_offsets(horizon_steps: int, stride_steps: int) -> list[int]:
    if horizon_steps <= 0 or stride_steps <= 0 or horizon_steps % stride_steps:
        raise ValueError("expert horizon must be positive and divisible by waypoint stride")
    offsets = list(range(0, horizon_steps + 1, stride_steps))
    if len(offsets) != 8:
        raise ValueError("Architecture v1 expert relabel must contain exactly 8 points")
    return offsets


def forecast_goal_index(
    expert_step: int, horizon_steps: int, lookahead_steps: int
) -> int:
    if expert_step < 0 or horizon_steps < 0 or lookahead_steps < 0:
        raise ValueError("forecast indices must be non-negative")
    if expert_step > horizon_steps:
        raise ValueError("expert step is outside the forecast horizon")
    return min(expert_step + lookahead_steps, horizon_steps)


def trajectory_statistics(waypoints: Sequence[Sequence[float]]) -> dict[str, float]:
    points = np.asarray(waypoints, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or not np.isfinite(points).all():
        raise ValueError("expert waypoints must be a finite [H,2] array")
    segments = np.diff(points, axis=0)
    return {
        "path_length_m": float(np.linalg.norm(segments, axis=1).sum()),
        "terminal_radius_m": float(np.linalg.norm(points[-1])),
        "maximum_segment_m": float(
            np.linalg.norm(segments, axis=1).max(initial=0.0)
        ),
    }


def target_progress_statistics(
    terminal_robot_xy: Sequence[float],
    terminal_target_xy: Sequence[float],
) -> dict[str, float]:
    """Compare following with leaving the robot at the relabel anchor.

    The humanoid keeps moving during the expert horizon, so the raw
    robot-target range may grow even when the robot follows correctly.  This
    comparison uses the same final target position and asks how much closer
    the expert endpoint is than the stationary anchor.
    """

    robot = np.asarray(terminal_robot_xy, dtype=np.float64)
    target = np.asarray(terminal_target_xy, dtype=np.float64)
    if robot.shape != (2,) or target.shape != (2,) or not (
        np.isfinite(robot).all() and np.isfinite(target).all()
    ):
        raise ValueError("terminal robot and target positions must be finite 2-vectors")
    stationary = float(np.linalg.norm(target))
    expert = float(np.linalg.norm(target - robot))
    robot_norm = float(np.linalg.norm(robot))
    target_norm = float(np.linalg.norm(target))
    alignment = (
        float(np.dot(robot, target) / (robot_norm * target_norm))
        if robot_norm > 1e-9 and target_norm > 1e-9
        else 0.0
    )
    return {
        "stationary_anchor_final_target_distance_m": stationary,
        "expert_final_target_distance_m": expert,
        "distance_improvement_over_stationary_anchor_m": stationary - expert,
        "terminal_motion_target_alignment_cosine": alignment,
    }


@dataclass(frozen=True)
class CoordinateTarget:
    visible: bool
    bbox_xyxy: None
    footpoint_uv: None
    relative_xy: tuple[float, float]
    range_m: float
    bearing_rad: float
    mask_area: int
    confidence: float


def _coordinate_target(relative_xy: Sequence[float]) -> CoordinateTarget:
    forward, left = (float(value) for value in relative_xy)
    return CoordinateTarget(
        visible=False,
        bbox_xyxy=None,
        footpoint_uv=None,
        relative_xy=(forward, left),
        range_m=float(np.hypot(forward, left)),
        bearing_rad=float(np.arctan2(left, forward)),
        mask_area=0,
        confidence=0.0,
    )


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_phase3_sample(
    *,
    directory: Path,
    source_rollout: Path,
    report_output: Path,
    result: Mapping[str, Any],
    initial_rgb: np.ndarray,
    history_frames: Sequence[tuple[int, np.ndarray]],
    initial_bbox_xyxy: Sequence[float],
    camera_intrinsics: np.ndarray,
    camera_from_base: np.ndarray,
    waypoints: Sequence[Sequence[float]],
    checkpoint_step: int,
    history_stride_steps: int,
    target_state: Mapping[str, Any],
    stop_required: bool = False,
) -> tuple[Path, str]:
    """Persist one auditable Phase-3 sample with split-aware eligibility."""

    directory = directory.expanduser().resolve(strict=False)
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite Phase-3 sample: {directory}")
    directory.mkdir(parents=True, exist_ok=False)
    initial_path = directory / "initial_rgb.png"
    Image.fromarray(np.asarray(initial_rgb, dtype=np.uint8)).save(initial_path)
    image_records: list[dict[str, object]] = []
    for history_index, (environment_step, frame) in enumerate(history_frames):
        path = directory / f"history_{history_index:02d}_step_{environment_step:03d}.png"
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(path)
        image_records.append({
            "environment_step": int(environment_step),
            "rgb_path": path.name,
            "sha256": _sha256(path),
        })
    formal_training_eligible = str(result["split"]) == "train"
    sample = {
        "schema_version": 1,
        "stage": "next007_phase3_model_visited_relabel_smoke",
        "sample_id": (
            f"habitat/{result['task']}/{result['episode_id']}"
            f"/post-action-{checkpoint_step:03d}"
        ),
        "formal_training_eligible": formal_training_eligible,
        "training_restriction": (
            "candidate_train_sample_pending_dataset_level_audit"
            if formal_training_eligible
            else "smoke_only_source_episode_is_val"
        ),
        "source": {
            "rollout_result": str(source_rollout),
            "rollout_result_sha256": _sha256(source_rollout),
            "relabel_report": str(report_output),
            "split": str(result["split"]),
            "dataset_index": int(result["dataset_index"]),
            "episode_id": str(result["episode_id"]),
            "scene_id": str(result["scene_id"]),
            "anchor_environment_step": int(checkpoint_step),
            "test_locked_used": False,
        },
        "model_inputs": {
            "condition_mode": "visual_only",
            "initial_rgb": {
                "rgb_path": initial_path.name,
                "sha256": _sha256(initial_path),
            },
            "initial_bbox_xyxy_norm": list(
                normalized_bbox(initial_bbox_xyxy, initial_rgb.shape)
            ),
            "rgb_history": image_records,
            "history_sampling": {
                "method": "causal_fixed_step_stride_with_left_padding",
                "stride_environment_steps": int(history_stride_steps),
            },
            "visual_initialization_valid": True,
            "rgb_valid": True,
            "binding_valid": True,
            "uwb": {
                "valid": False,
                "relative_position_base_xy_m": [0.0, 0.0],
                "covariance_base_xy_m2": [[1.0, 0.0], [0.0, 1.0]],
                "quality_01": 0.0,
                "age_s": 0.0,
            },
            "camera_intrinsics": np.asarray(camera_intrinsics).tolist(),
            "camera_from_base": np.asarray(camera_from_base).tolist(),
        },
        "supervision": {
            "expert_trajectory": {
                "waypoints_base_xy_m": [list(point) for point in waypoints],
                "valid_mask": [True] * len(waypoints),
                "stop_required": bool(stop_required),
            },
            "target_state": dict(target_state),
            "gt_used_only_on_label_side": True,
            "later_bbox_used_by_model": False,
        },
    }
    manifest_path = directory / "sample.json"
    _atomic_json(manifest_path, sample)
    return manifest_path, _sha256(manifest_path)


def _reference_after_step(result: Mapping[str, Any], step: int) -> Mapping[str, Any]:
    records = result["steps"]
    if step <= 0 or step > len(records):
        raise ValueError("checkpoint step has no saved post-action reference")
    evaluation = records[step - 1].get("evaluation_only_after_action")
    if not isinstance(evaluation, Mapping):
        raise ValueError("rollout checkpoint lacks evaluation reference")
    return evaluation


def main() -> int:
    args = arguments()
    result_path = args.rollout_result.expanduser().resolve(strict=True)
    result = _load(result_path)
    if not isinstance(result, dict):
        raise ValueError("rollout result must be a JSON object")
    actions = replay_actions(result)
    offsets = waypoint_offsets(args.expert_horizon_steps, args.waypoint_stride_steps)
    if args.checkpoint_step <= 0 or args.checkpoint_step > len(actions):
        raise ValueError("checkpoint step is outside the saved rollout")
    total_steps = args.checkpoint_step + args.expert_horizon_steps
    if total_steps > args.maximum_safe_total_steps:
        raise ValueError(
            "replay plus expert branch exceeds the audited Habitat step boundary"
        )
    if (
        args.expert_max_forward <= 0.0
        or args.expert_max_lateral <= 0.0
        or args.expert_max_yaw <= 0.0
        or args.expert_translation_slew <= 0.0
        or args.expert_target_lookahead_steps < 0
        or args.history_size <= 0
        or args.history_stride_steps <= 0
        or args.maximum_trajectory_path_m <= 0.0
        or args.maximum_terminal_radius_m <= 0.0
        or args.minimum_stationary_distance_improvement_m < 0.0
        or args.history_size <= 0
        or args.model_image_height <= 0
        or args.model_image_width <= 0
    ):
        raise ValueError("expert actuation and trajectory limits must be positive")
    initialization = result.get("initialization")
    if not isinstance(initialization, Mapping) or initialization.get("environment_step") != 0:
        raise ValueError("probe currently requires a step-0 visual initialization")

    import habitat
    from habitat.config import read_write
    from habitat.datasets import make_dataset
    import evt_bench  # noqa: F401
    from omtrackvla.oracle_modular_follow import local_target
    from omtrackvla.oracle_modular_follow_v6 import OracleNavmeshFollowerV6

    task = str(result["task"])
    split = str(result["split"])
    config_kind = "train" if split == "train" else "infer"
    config_path = (
        "habitat-lab/habitat/config/benchmark/nav/track/"
        f"track_{config_kind}_{task}.yaml"
    )
    def environment_config() -> Any:
        selected_config = configure(
            habitat.get_config(config_path), args.scene_dataset
        )
        with read_write(selected_config):
            if PANOPTIC_KEY not in selected_config.habitat.gym.obs_keys:
                selected_config.habitat.gym.obs_keys.append(PANOPTIC_KEY)
        return selected_config

    config = environment_config()
    dataset_index = int(result["dataset_index"])

    def selected_dataset(dataset_config: Any) -> tuple[Any, Any]:
        selected = make_dataset(
            dataset_config.habitat.dataset.type,
            config=dataset_config.habitat.dataset,
        )
        if not 0 <= dataset_index < len(selected.episodes):
            raise ValueError("saved dataset index is outside the current dataset")
        selected_episode = selected.episodes[dataset_index]
        if (
            str(selected_episode.episode_id) != str(result["episode_id"])
            or str(selected_episode.scene_id) != str(result["scene_id"])
        ):
            raise ValueError(
                "saved rollout episode identity no longer matches the dataset"
            )
        selected.episodes = [selected_episode]
        return selected, selected_episode

    dataset, episode = selected_dataset(config)

    reference = _reference_after_step(result, args.checkpoint_step)
    expert_records: list[dict[str, object]] = []
    waypoints: list[list[float]] = []
    phase3_sample: dict[str, object] | None = None
    def reset_and_replay(
        replay_env: Any,
    ) -> tuple[Any, np.ndarray, list[tuple[int, np.ndarray]]]:
        replay_env.reset()
        _assign_unique_humanoid_semantic_ids(replay_env)
        replay_observations = replay_env.sim.get_sensor_observations()
        replay_initial_rgb = np.asarray(
            replay_observations[RGB_KEY]
        )[..., :3].copy()
        replay_history: list[tuple[int, np.ndarray]] = [
            (0, replay_initial_rgb.copy())
        ]
        for replay_step, action in enumerate(
            actions[: args.checkpoint_step], 1
        ):
            if replay_env.episode_over:
                raise RuntimeError(
                    f"episode ended during replay at step {replay_step}"
                )
            replay_env.step({
                "action": ACTION_NAMES,
                "action_args": {"agent_1_base_vel": action},
            })
            replay_observations = replay_env.sim.get_sensor_observations()
            replay_history.append((
                replay_step,
                np.asarray(replay_observations[RGB_KEY])[..., :3].copy(),
            ))
        return replay_observations, replay_initial_rgb, replay_history

    target_forecast_world: list[np.ndarray] | None = None
    if args.expert_target_lookahead_steps > 0:
        forecast_config = environment_config()
        forecast_dataset, _ = selected_dataset(forecast_config)
        with habitat.TrackEnv(
            config=forecast_config, dataset=forecast_dataset
        ) as forecast_env:
            reset_and_replay(forecast_env)
            forecast_target_agent = (
                forecast_env.sim.agents_mgr[0].articulated_agent
            )
            target_forecast_world = [np.asarray(
                forecast_target_agent.base_pos, dtype=np.float64
            ).copy()]
            for forecast_step in range(1, args.expert_horizon_steps + 1):
                if forecast_env.episode_over:
                    raise RuntimeError(
                        "episode ended during label-side target forecast at "
                        f"step {forecast_step}"
                    )
                forecast_env.step({
                    "action": ACTION_NAMES,
                    "action_args": {"agent_1_base_vel": [0.0, 0.0, 0.0]},
                })
                target_forecast_world.append(np.asarray(
                    forecast_target_agent.base_pos, dtype=np.float64
                ).copy())

    with habitat.TrackEnv(config=config, dataset=dataset) as env:
        observations, initial_rgb, replay_rgb = reset_and_replay(env)

        robot = env.sim.agents_mgr[1].articulated_agent
        target_agent = env.sim.agents_mgr[0].articulated_agent
        replay_distance = float(np.linalg.norm(
            np.asarray(robot.base_pos, dtype=np.float64)
            - np.asarray(target_agent.base_pos, dtype=np.float64)
        ))
        saved_distance = float(reference["gt_distance_m"])
        robot_position = np.asarray(robot.base_pos, dtype=np.float64)
        target_position = np.asarray(target_agent.base_pos, dtype=np.float64)
        local_vector = robot.sim_obj.transformation.inverted().transform_vector(
            target_position - robot_position
        )
        target_forward_m = float(local_vector.x)
        target_left_m = float(-local_vector.z)
        target_angle_rad = math.atan2(target_left_m, target_forward_m)
        target_distance_m = math.hypot(target_forward_m, target_left_m)
        target_mask = (
            np.asarray(observations[PANOPTIC_KEY]).squeeze()
            == int(episode.info["main_human_semantic_id"])
        )
        target_bbox_xyxy = target_mask_to_bbox(target_mask)
        target_visible = target_bbox_xyxy is not None
        target_state = {
            "bbox_xyxy_norm": (
                None if target_bbox_xyxy is None else
                list(normalized_bbox(target_bbox_xyxy, initial_rgb.shape))
            ),
            "visible": target_visible,
            "angle_sincos": [
                math.sin(target_angle_rad), math.cos(target_angle_rad)
            ],
            "distance_m": target_distance_m,
            "polar_valid": target_visible,
            "gt_used_only_on_label_side": True,
        }
        camera = _camera_transform(env.sim)
        saved_camera = np.asarray(
            reference["render_audit"]["camera_transform_after_forced_render"],
            dtype=np.float64,
        )
        camera_max_abs_error = float(np.max(np.abs(camera - saved_camera)))
        distance_abs_error = abs(replay_distance - saved_distance)
        replay_matches = bool(
            camera_max_abs_error <= args.replay_tolerance
            and distance_abs_error <= args.replay_tolerance
        )
        if not replay_matches:
            raise RuntimeError(
                "saved model action replay did not reproduce the model-visited state: "
                f"camera_error={camera_max_abs_error} distance_error={distance_abs_error}"
            )

        anchor_position = np.asarray(robot.base_pos, dtype=np.float64).copy()
        anchor_transform = np.asarray(robot.sim_obj.transformation, dtype=np.float64).copy()
        forward_axis, left_axis = anchor_basis(anchor_transform)
        controller = OracleNavmeshFollowerV6(
            max_forward=args.expert_max_forward,
            max_lateral=args.expert_max_lateral,
            max_yaw=args.expert_max_yaw,
            translation_slew_per_step=args.expert_translation_slew,
            emergency_translation_slew=args.expert_translation_slew,
        )
        controller.reset()
        anchor_collision_metric = float(
            env.get_metrics().get("human_collision", 0.0) or 0.0
        )
        collisions = 0.0
        expert_minimum_target_distance_m = float("inf")
        target_forecast_same_time_max_error_m = 0.0
        offset_set = set(offsets)
        initial_target_anchor_xy = position_in_anchor_frame(
            target_agent.base_pos, anchor_position, forward_axis, left_axis
        )
        initial_target_local_xy = list(local_target(robot, target_agent))
        coordinate_frame_max_abs_error = float(np.max(np.abs(
            np.asarray(initial_target_anchor_xy, dtype=np.float64)
            - np.asarray(initial_target_local_xy, dtype=np.float64)
        )))
        if coordinate_frame_max_abs_error > args.replay_tolerance:
            raise RuntimeError(
                "anchor-frame projection disagrees with Habitat local_target: "
                f"max_abs_error={coordinate_frame_max_abs_error}"
            )
        for expert_step in range(args.expert_horizon_steps + 1):
            if expert_step in offset_set:
                waypoints.append(position_in_anchor_frame(
                    robot.base_pos, anchor_position, forward_axis, left_axis
                ))
            if expert_step == args.expert_horizon_steps:
                break
            if env.episode_over:
                raise RuntimeError(f"episode ended during expert branch at step {expert_step}")
            relative_xy = local_target(robot, target_agent)
            target = _coordinate_target(relative_xy)
            if target_forecast_world is None:
                expert_target_world = np.asarray(
                    target_agent.base_pos, dtype=np.float64
                ).copy()
                target_goal_index = expert_step
            else:
                same_time_error = float(np.linalg.norm(
                    np.asarray(target_agent.base_pos, dtype=np.float64)
                    - target_forecast_world[expert_step]
                ))
                target_forecast_same_time_max_error_m = max(
                    target_forecast_same_time_max_error_m, same_time_error
                )
                target_goal_index = forecast_goal_index(
                    expert_step,
                    args.expert_horizon_steps,
                    args.expert_target_lookahead_steps,
                )
                expert_target_world = target_forecast_world[target_goal_index]
            decision = controller(
                env.sim,
                robot,
                target_agent,
                target,
                target_world=np.asarray(expert_target_world, dtype=np.float32),
            )
            expert_records.append({
                "expert_step": expert_step,
                "target_relative_xy_m": list(relative_xy),
                "target_distance_m": float(np.hypot(*relative_xy)),
                "target_goal_forecast_step": target_goal_index,
                "target_goal_world": expert_target_world.tolist(),
                "mode": decision.mode,
                "action": {
                    "forward": decision.action.forward,
                    "lateral": decision.action.lateral,
                    "yaw": decision.action.yaw,
                },
                "navmesh_waypoint_world": (
                    list(decision.waypoint_world)
                    if decision.waypoint_world is not None else None
                ),
                "robot_anchor_xy_m_before": position_in_anchor_frame(
                    robot.base_pos, anchor_position, forward_axis, left_axis
                ),
                "target_anchor_xy_m_before": position_in_anchor_frame(
                    target_agent.base_pos, anchor_position, forward_axis, left_axis
                ),
            })
            env.step({
                "action": ACTION_NAMES,
                "action_args": {"agent_1_base_vel": decision.action.as_habitat()},
            })
            metrics = env.get_metrics()
            post_step_target_distance_m = float(np.linalg.norm(
                np.asarray(robot.base_pos, dtype=np.float64)
                - np.asarray(target_agent.base_pos, dtype=np.float64)
            ))
            expert_minimum_target_distance_m = min(
                expert_minimum_target_distance_m, post_step_target_distance_m
            )
            current_collision_metric = float(
                metrics.get("human_collision", 0.0) or 0.0
            )
            # EVT's human_collision measure is sticky for the remainder of an
            # episode. A recovery sample taken after an earlier collision must
            # therefore distinguish that anchor history from a new expert-side
            # unsafe contact. The distance check remains active in both cases.
            new_collision = (
                current_collision_metric > anchor_collision_metric + 1.0e-6
                or post_step_target_distance_m < 0.30
            )
            collisions = max(
                collisions, float(new_collision)
            )

        if target_forecast_world is not None:
            target_forecast_same_time_max_error_m = max(
                target_forecast_same_time_max_error_m,
                float(np.linalg.norm(
                    np.asarray(target_agent.base_pos, dtype=np.float64)
                    - target_forecast_world[args.expert_horizon_steps]
                )),
            )
        terminal_robot_xy = position_in_anchor_frame(
            robot.base_pos, anchor_position, forward_axis, left_axis
        )
        terminal_target_xy = position_in_anchor_frame(
            target_agent.base_pos, anchor_position, forward_axis, left_axis
        )

    waypoint_array = np.asarray(waypoints, dtype=np.float64)
    trajectory = trajectory_statistics(waypoints)
    target_progress = target_progress_statistics(
        terminal_robot_xy, terminal_target_xy
    )
    passed = bool(
        waypoint_array.shape == (8, 2)
        and np.isfinite(waypoint_array).all()
        and np.max(np.abs(waypoint_array[0])) <= 1e-6
        and replay_matches
        and coordinate_frame_max_abs_error <= args.replay_tolerance
        and target_forecast_same_time_max_error_m <= args.replay_tolerance
        and len(expert_records) == args.expert_horizon_steps
        and 0.05 <= trajectory["path_length_m"] <= args.maximum_trajectory_path_m
        and trajectory["terminal_radius_m"] <= args.maximum_terminal_radius_m
        and target_progress["distance_improvement_over_stationary_anchor_m"]
        >= args.minimum_stationary_distance_improvement_m
        and target_progress["terminal_motion_target_alignment_cosine"] > 0.0
        and collisions == 0.0
    )
    payload = {
        "schema_version": 1,
        "stage": "next007_arbitrary_state_expert_relabel_probe",
        "status": "passed" if passed else "failed",
        "source_rollout": str(result_path),
        "source_rollout_stage": result.get("stage"),
        "task": task,
        "split": split,
        "dataset_index": dataset_index,
        "episode_id": str(episode.episode_id),
        "scene_id": str(episode.scene_id),
        "model_visited_checkpoint_step": args.checkpoint_step,
        "expert_branch_steps": args.expert_horizon_steps,
        "total_environment_steps": total_steps,
        "maximum_safe_total_steps": args.maximum_safe_total_steps,
        "replay_audit": {
            "saved_policy_actions_replayed_without_gt": True,
            "camera_transform_max_abs_error": camera_max_abs_error,
            "target_distance_abs_error_m": distance_abs_error,
            "tolerance": args.replay_tolerance,
            "matches_saved_model_visited_state": replay_matches,
            "human_collision_metric_at_anchor": anchor_collision_metric,
        },
        "coordinate_audit": {
            "anchor_target_xy_m": initial_target_anchor_xy,
            "habitat_local_target_xy_m": initial_target_local_xy,
            "max_abs_error_m": coordinate_frame_max_abs_error,
            "tolerance": args.replay_tolerance,
            "passed": coordinate_frame_max_abs_error <= args.replay_tolerance,
        },
        "label_boundary": {
            "gt_target_pose_used_only_after_replay_for_supervision": True,
            "future_target_trajectory_used_only_by_label_expert": bool(
                args.expert_target_lookahead_steps > 0
            ),
            "navmesh_used_only_by_expert": True,
            "expert_waypoint_entered_deployment_policy": False,
            "source_policy_input_audit": result.get("input_audit"),
            "test_locked_used": False,
        },
        "expert": {
            "controller": "OracleNavmeshFollowerV6",
            "actuation_limits": {
                "max_forward": args.expert_max_forward,
                "max_lateral": args.expert_max_lateral,
                "max_yaw": args.expert_max_yaw,
                "translation_slew_per_step": args.expert_translation_slew,
            },
            "target_forecast": {
                "lookahead_steps": args.expert_target_lookahead_steps,
                "branch_robot_action": "fixed_zero_hold",
                "separate_reset_before_expert": bool(
                    args.expert_target_lookahead_steps > 0
                ),
                "same_time_target_max_error_m": (
                    target_forecast_same_time_max_error_m
                ),
                "deterministic_under_expert_actions": (
                    target_forecast_same_time_max_error_m
                    <= args.replay_tolerance
                ),
            },
            "coordinate_source": "Habitat label-side robot/target pose",
            "waypoint_frame": "checkpoint robot base x-forward/y-left",
            "waypoint_offsets_steps": offsets,
            "waypoint_offsets_seconds_at_30hz": [value / 30.0 for value in offsets],
            "waypoints_base_xy_m": waypoints,
            "trajectory_statistics": trajectory,
            "target_progress_statistics": target_progress,
            "initial_target_anchor_xy_m": initial_target_anchor_xy,
            "terminal_target_anchor_xy_m": terminal_target_xy,
            "trajectory_gate": {
                "minimum_path_m": 0.05,
                "maximum_path_m": args.maximum_trajectory_path_m,
                "maximum_terminal_radius_m": args.maximum_terminal_radius_m,
                "minimum_stationary_distance_improvement_m": (
                    args.minimum_stationary_distance_improvement_m
                ),
                "passed": bool(
                    0.05 <= trajectory["path_length_m"] <= args.maximum_trajectory_path_m
                    and trajectory["terminal_radius_m"] <= args.maximum_terminal_radius_m
                    and target_progress[
                        "distance_improvement_over_stationary_anchor_m"
                    ] >= args.minimum_stationary_distance_improvement_m
                    and target_progress[
                        "terminal_motion_target_alignment_cosine"
                    ] > 0.0
                ),
            },
            "collision": collisions,
            "minimum_target_distance_m": expert_minimum_target_distance_m,
            "collision_gate": (
                "new_metric_transition_or_distance_below_0.30m"
            ),
            "records": expert_records,
        },
    }
    if passed and args.phase3_sample_dir is not None:
        replay_by_step = dict(replay_rgb)
        requested_history_steps = causal_history_steps(
            args.checkpoint_step, args.history_size, args.history_stride_steps
        )
        history_frames = [
            (environment_step, replay_by_step[environment_step])
            for environment_step in requested_history_steps
        ]
        if len(history_frames) != args.history_size:
            raise RuntimeError(
                f"Phase-3 history has {len(history_frames)} frames, "
                f"expected {args.history_size}"
            )
        intrinsics, camera_from_base = habitat_camera_calibration(
            initial_rgb.shape,
            output_height=args.model_image_height,
            output_width=args.model_image_width,
            hfov_deg=90.0,
        )
        manifest_path, manifest_sha256 = _write_phase3_sample(
            directory=args.phase3_sample_dir,
            source_rollout=result_path,
            report_output=args.output.expanduser().resolve(strict=False),
            result=result,
            initial_rgb=initial_rgb,
            history_frames=history_frames,
            initial_bbox_xyxy=initialization["bbox_xyxy"],
            camera_intrinsics=intrinsics,
            camera_from_base=camera_from_base,
            waypoints=waypoints,
            checkpoint_step=args.checkpoint_step,
            history_stride_steps=args.history_stride_steps,
            target_state=target_state,
        )
        phase3_sample = {
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "history_environment_steps": [item[0] for item in history_frames],
            "history_stride_environment_steps": args.history_stride_steps,
            "formal_training_eligible": split == "train",
        }
        payload["phase3_smoke_sample"] = phase3_sample
    _atomic_json(args.output, payload)
    print(json.dumps({
        "status": payload["status"],
        "checkpoint_step": args.checkpoint_step,
        "waypoints": waypoints,
        "replay_audit": payload["replay_audit"],
        "trajectory_statistics": trajectory,
        "target_progress_statistics": target_progress,
        "phase3_smoke_sample": phase3_sample,
        "collision": collisions,
        "output": str(args.output),
    }, indent=2))
    if not passed:
        raise RuntimeError("arbitrary-state expert relabel probe failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
