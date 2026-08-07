#!/usr/bin/env python3
"""Sharded dataset evaluation for the modular oracle person follower."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import traceback
from pathlib import Path

import imageio.v2 as imageio
import cv2
import numpy as np
from PIL import Image, ImageDraw

from oracle_modular_follow import (
    ACTION_NAMES,
    DEFAULT_SCENE_DATASET,
    DEPTH_KEY,
    PANOPTIC_KEY,
    RGB_KEY,
    OraclePerception,
    TargetObservation,
    ModularReactiveFollower,
    MapReactiveFollower,
    annotate,
    configure,
    local_target,
    target_mask_to_bbox,
)
from rgb_person_perception import (
    DEFAULT_WEIGHTS,
    DEFAULT_REID_WEIGHTS,
    RGBPersonPerception,
    RGBPersonPerceptionWorker,
    bbox_iou,
    metric_depth,
)

CONTROLLER_VERSION = int(os.environ.get("ORACLE_CONTROLLER_VERSION", "5"))
if CONTROLLER_VERSION == 5:
    from oracle_modular_follow import OracleNavmeshFollower
elif CONTROLLER_VERSION == 6:
    from oracle_modular_follow_v6 import OracleNavmeshFollowerV6 as OracleNavmeshFollower
else:
    raise RuntimeError(f"Unsupported ORACLE_CONTROLLER_VERSION={CONTROLLER_VERSION}")


def parse_dataset_indices(value: str) -> tuple[int, ...]:
    try:
        indices = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "dataset indices must be comma-separated integers"
        ) from exc
    if any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("dataset indices must be non-negative")
    return tuple(sorted(indices))


def scene_key(scene_id: str) -> str:
    return Path(scene_id).name.split(".")[0]


def episode_key(index, episode):
    identity = {
        "index": index,
        "scene": scene_key(episode.scene_id),
        "episode_id": str(episode.episode_id),
        "robot_position": episode.info.get("robot_position"),
        "target": episode.info.get("main_humanoid_name"),
    }
    digest = hashlib.sha1(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    return f"{index:06d}_{identity['scene']}_ep_{identity['episode_id']}_{digest}"


def target_goal_crop(observations, target_semantic_id):
    mask = np.asarray(observations[PANOPTIC_KEY]).squeeze() == int(target_semantic_id)
    bbox = target_mask_to_bbox(mask)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return np.asarray(observations[RGB_KEY])[y1:y2 + 1, x1:x2 + 1, :3].copy()


def assign_unique_humanoid_semantic_ids(env):
    """Give distractor humanoids episode-unique panoptic instance IDs.

    EVT episodes may instantiate two copies of the same avatar, which otherwise
    share one semantic ID and are indistinguishable in the panoptic image. The
    main target keeps the benchmark-provided ID; distractors receive IDs above
    2000 so target metrics remain unchanged.
    """
    target_id = int(env.current_episode.info["main_human_semantic_id"])
    assigned = {0: target_id}
    for agent_index in range(len(env.sim.agents_mgr)):
        if agent_index == 1:
            continue
        semantic_id = target_id if agent_index == 0 else 2000 + agent_index
        articulated_agent = env.sim.agents_mgr[agent_index].articulated_agent
        for node in articulated_agent.sim_obj.visual_scene_nodes:
            node.semantic_id = semantic_id
        assigned[agent_index] = semantic_id
    return assigned


def invisible_target(relative_xy):
    forward, left = (float(value) for value in relative_xy)
    return TargetObservation(
        visible=False,
        bbox_xyxy=None,
        footpoint_uv=None,
        relative_xy=(forward, left),
        range_m=float(np.hypot(forward, left)),
        bearing_rad=float(np.arctan2(left, forward)),
        mask_area=0,
        confidence=0.0,
    )


def compose_rgbd_video_frame(
    rgb_frame: np.ndarray, raw_depth: np.ndarray, max_depth_m: float = 10.0,
) -> np.ndarray:
    rgb = np.asarray(rgb_frame)[..., :3].astype(np.uint8)
    depth = metric_depth(raw_depth, max_depth_m=max_depth_m)
    if depth.ndim != 2 or not depth.size:
        raise ValueError(f"Expected a non-empty depth image, got shape {depth.shape}")

    valid = depth > 0.0
    proximity = np.zeros(depth.shape, dtype=np.uint8)
    proximity[valid] = np.clip(
        255.0 * (1.0 - depth[valid] / max_depth_m), 0.0, 255.0
    ).astype(np.uint8)
    depth_rgb = cv2.applyColorMap(proximity, cv2.COLORMAP_TURBO)[..., ::-1]
    depth_rgb[~valid] = 0
    if depth_rgb.shape[:2] != rgb.shape[:2]:
        height, width = rgb.shape[:2]
        depth_rgb = np.asarray(
            Image.fromarray(depth_rgb).resize((width, height), Image.Resampling.NEAREST)
        )

    depth_image = Image.fromarray(depth_rgb)
    draw = ImageDraw.Draw(depth_image)
    label = f"DEPTH | near=warm far=cool | valid 0.1-{max_depth_m:.1f}m"
    box = draw.textbbox((8, 8), label)
    draw.rectangle(
        (box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2), fill=(0, 0, 0)
    )
    draw.text((8, 8), label, fill=(255, 255, 255))
    return np.concatenate((rgb, np.asarray(depth_image)), axis=1)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def completed_result(
    index, episode, episodes_dir, videos_dir, save_video, require_success=False,
    controller=None,
):
    key = episode_key(index, episode)
    result_path = episodes_dir / f"{key}.json"
    video_path = videos_dir / f"{key}.mp4" if save_video else None
    if not result_path.exists():
        return False
    try:
        existing = json.loads(result_path.read_text())
    except (OSError, ValueError):
        return False
    video_complete = video_path is None or (
        video_path.exists() and video_path.stat().st_size > 0
    )
    complete = bool(
        "summary" in existing
        and existing.get("controller_version") == CONTROLLER_VERSION
        and (
            controller is None
            or existing.get("controller", "oracle-navmesh") == controller
        )
        and video_complete
    )
    if require_success:
        complete = complete and bool(existing["summary"].get("success", 0.0))
    return complete


def prior_success_attempt(result_path: Path) -> int:
    try:
        existing = json.loads(result_path.read_text())
    except (OSError, ValueError):
        return 0
    if existing.get("controller_version") != CONTROLLER_VERSION:
        return 0
    return int(existing.get("success_attempt", 0))


def prior_evasion_side(result_path: Path):
    try:
        existing = json.loads(result_path.read_text())
    except (OSError, ValueError):
        return None
    if existing.get("controller_version") != CONTROLLER_VERSION:
        return None
    side = existing.get("summary", {}).get("evasion_side")
    return float(side) if side in (-1, -1.0, 1, 1.0) else None


def exhausted_success_attempts(result_path: Path, max_attempts: int) -> bool:
    try:
        existing = json.loads(result_path.read_text())
    except (OSError, ValueError):
        return False
    return bool(
        existing.get("controller_version") == CONTROLLER_VERSION
        and "summary" in existing
        and not existing["summary"].get("success", 0.0)
        and int(existing.get("success_attempt", 0)) >= max_attempts
    )


def evaluate_episode(
    env, observations, controller, max_steps, save_steps=False,
    video_path=None, video_fps=8, perception=None, defer_goal_crop=False,
):
    episode = env.current_episode
    robot = env.sim.agents_mgr[1].articulated_agent
    target_agent = env.sim.agents_mgr[0].articulated_agent
    target_semantic_id = int(episode.info["main_human_semantic_id"])
    human_semantic_ids = set()
    for agent_index in range(len(env.sim.agents_mgr)):
        if agent_index == 1:  # Spot robot
            continue
        articulated_agent = env.sim.agents_mgr[agent_index].articulated_agent
        for node in articulated_agent.sim_obj.visual_scene_nodes:
            semantic_id = int(node.semantic_id)
            if semantic_id > 0:
                human_semantic_ids.add(semantic_id)
    human_semantic_ids.add(target_semantic_id)
    human_semantic_ids_array = np.asarray(sorted(human_semantic_ids))
    if perception is None:
        perception = OraclePerception()

    followed = 0.0
    collision = 0.0
    distances = []
    visible_steps = 0
    coordinate_steps = 0
    perception_detected_steps = 0
    perception_correct_steps = 0
    perception_target_ious = []
    perception_gt_visible_steps = 0
    goal_crop_wait_steps = 0
    goal_crop_initialized = not defer_goal_crop
    records = []
    steps = 0
    max_steps = int(max_steps)
    writer = None
    temporary_video = None
    if video_path is not None:
        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_video = video_path.with_name(video_path.stem + ".tmp.mp4")
        writer = imageio.get_writer(
            temporary_video, fps=video_fps, codec="libx264", macro_block_size=1
        )

    try:
        while not env.episode_over and steps < max_steps:
            if isinstance(perception, OraclePerception) and not isinstance(controller, MapReactiveFollower):
                required = (RGB_KEY, PANOPTIC_KEY)
            elif isinstance(perception, OraclePerception):
                required = (RGB_KEY, DEPTH_KEY, PANOPTIC_KEY)
            elif not goal_crop_initialized:
                required = (RGB_KEY, DEPTH_KEY, PANOPTIC_KEY)
            else:
                required = (RGB_KEY, DEPTH_KEY)
            missing = [key for key in required if key not in observations]
            if missing:
                raise KeyError(f"Missing observations {missing}; got {sorted(observations)}")
            if isinstance(perception, OraclePerception):
                target = perception(
                    observations[RGB_KEY], observations[PANOPTIC_KEY], target_semantic_id,
                    local_target(robot, target_agent),
                )
            elif not goal_crop_initialized:
                reference_rgb = target_goal_crop(observations, target_semantic_id)
                if reference_rgb is None:
                    goal_crop_wait_steps += 1
                    target = invisible_target(local_target(robot, target_agent))
                else:
                    perception.reset(reference_rgb=reference_rgb)
                    goal_crop_initialized = True
                    print(json.dumps({
                        "event": "oracle_batch_goal_crop_initialized",
                        "task": episode.info.get("episode_mode"),
                        "dataset_index": episode.info.get("_oracle_batch_index"),
                        "episode_id": str(episode.episode_id),
                        "step": steps,
                    }), flush=True)
                    target = perception(observations[RGB_KEY], observations[DEPTH_KEY])
            else:
                target = perception(observations[RGB_KEY], observations[DEPTH_KEY])
            current_target_mask = (
                np.asarray(observations[PANOPTIC_KEY]).squeeze() == target_semantic_id
            )
            current_gt_bbox = target_mask_to_bbox(current_target_mask)
            perception_gt_visible_steps += int(current_gt_bbox is not None)
            target_iou = 0.0
            if target.bbox_xyxy is not None and current_gt_bbox is not None:
                target_iou = bbox_iou(target.bbox_xyxy, current_gt_bbox)
                perception_target_ious.append(target_iou)
            perception_detected_steps += int(target.visible)
            target_selection_correct = bool(target.visible and target_iou >= 0.3)
            perception_correct_steps += int(target_selection_correct)
            candidate_records = []
            for candidate in (
                getattr(perception, "last_candidate_diagnostics", None) or []
            ):
                candidate_record = dict(candidate)
                candidate_iou = (
                    bbox_iou(candidate_record["bbox_xyxy"], current_gt_bbox)
                    if current_gt_bbox is not None else 0.0
                )
                candidate_record["target_iou"] = candidate_iou
                candidate_record["target_match"] = candidate_iou >= 0.3
                candidate_records.append(candidate_record)
            person_dynamic_mask = None
            if isinstance(perception, OraclePerception):
                # Keep every visible humanoid agent in the dynamic layer; only
                # the target semantic id is used for tracking metrics/control.
                panoptic_frame = np.asarray(observations[PANOPTIC_KEY]).squeeze()
                person_dynamic_mask = np.isin(
                    panoptic_frame, human_semantic_ids_array
                )
            elif candidate_records:
                # Learned perception has no privileged person IDs. Union all
                # detector candidates so non-target people are not fossilized
                # into the persistent static map.
                person_dynamic_mask = np.zeros(
                    np.asarray(observations[DEPTH_KEY]).squeeze().shape, dtype=bool
                )
                mask_h, mask_w = person_dynamic_mask.shape
                for candidate in candidate_records:
                    x1, y1, x2, y2 = candidate["bbox_xyxy"]
                    person_dynamic_mask[
                        max(0, int(y1)):min(mask_h, int(y2) + 1),
                        max(0, int(x1)):min(mask_w, int(x2) + 1),
                    ] = True
            if hasattr(controller, "update_observation"):
                controller.update_observation(
                    observations,
                    target,
                    dynamic_mask=(
                        person_dynamic_mask
                        if person_dynamic_mask is not None
                        else None
                    ),
                    robot=robot,
                )
            decision = controller(env.sim, robot, target_agent, target)
            if (
                not target.visible
                and not getattr(controller, "uses_invisible_pointgoal", False)
            ):
                coordinate_steps += 1
            if writer is not None:
                current_following = env.get_metrics().get("human_following")
                frame = annotate(
                    observations[RGB_KEY], target, decision,
                    f"{episode.info.get('_oracle_batch_index')} | {scene_key(episode.scene_id)} | ep {episode.episode_id} | step {steps}",
                    current_following,
                )
                frame = np.asarray(frame)
                if DEPTH_KEY in observations and (
                    not isinstance(perception, OraclePerception)
                    or isinstance(controller, MapReactiveFollower)
                ):
                    frame = compose_rgbd_video_frame(
                        frame, observations[DEPTH_KEY]
                    )
                map_frame = getattr(controller, "last_map_visualization", None)
                if map_frame is not None:
                    map_frame = cv2.resize(
                        map_frame,
                        (frame.shape[0], frame.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    frame = np.concatenate((frame, map_frame), axis=1)
                writer.append_data(frame)

            observations = env.step({
                "action": ACTION_NAMES,
                "action_args": {"agent_1_base_vel": decision.action.as_habitat()},
            })
            steps += 1
            metrics = env.get_metrics()
            current_distance = float(np.linalg.norm(
                np.asarray(robot.base_pos) - np.asarray(target_agent.base_pos)
            ))
            distances.append(current_distance)
            followed += float(metrics.get("human_following", 0.0) or 0.0)
            collision = max(
                collision, float(metrics.get("human_collision", 0.0) or 0.0)
            )
            current_mask = np.asarray(observations[PANOPTIC_KEY]).squeeze()
            visible = bool(np.any(current_mask == target_semantic_id))
            visible_human_ids = np.unique(
                current_mask[np.isin(
                    current_mask, human_semantic_ids_array
                )]
            )
            visible_steps += int(visible)
            if save_steps:
                records.append({
                    "step": steps,
                    "distance_m": current_distance,
                    "visible": visible,
                    "visible_human_count": int(visible_human_ids.size),
                    "visible_human_ids": [int(v) for v in visible_human_ids.tolist()],
                    "mask_area": target.mask_area,
                    "rgb_mean": float(np.asarray(observations[RGB_KEY])[..., :3].mean()),
                    "perception_confidence": target.confidence,
                    "perception_range_m": target.range_m,
                    "perception_bearing_rad": target.bearing_rad,
                    "perception_candidate_count": getattr(
                        perception, "last_candidate_count", None
                    ),
                    "perception_association_score": getattr(
                        perception, "last_association_score", None
                    ),
                    "perception_goal_similarity": getattr(
                        perception, "last_goal_similarity", None
                    ),
                    "perception_candidates": candidate_records,
                    "target_bbox_iou": target_iou,
                    "target_selection_correct": target_selection_correct,
                    "mode": decision.mode,
                    "map_mode": getattr(controller, "last_map_mode", None),
                    "map_clearance": getattr(controller, "last_map_clearance", None),
                    "target_dynamic_points": getattr(getattr(controller, "obstacle_map", None), "last_target_dynamic_points", None),
                    "map_static_cells": int(getattr(getattr(controller, "obstacle_map", None), "static_map", np.zeros(1)).sum()),
                    "map_dynamic_cells": int(getattr(getattr(controller, "obstacle_map", None), "dynamic_map", np.zeros(1)).sum()),
                    "map_path_length": len(getattr(getattr(controller, "obstacle_map", None), "last_path_px", [])),
                    "map_ground_filtered_points": getattr(getattr(controller, "obstacle_map", None), "last_ground_filtered_points", None),
                    "map_ceiling_filtered_points": getattr(getattr(controller, "obstacle_map", None), "last_ceiling_filtered_points", None),
                    "action": decision.action.as_habitat(),
                    "human_following": float(metrics.get("human_following", 0.0) or 0.0),
                    "human_collision": float(metrics.get("human_collision", 0.0) or 0.0),
                })
            if collision:
                break
    finally:
        if writer is not None:
            writer.close()
    if temporary_video is not None:
        temporary_video.replace(video_path)

    final_metrics = env.get_metrics()
    if steps < max_steps:
        success = float(bool(
            final_metrics.get("human_following_success", 0.0)
            and final_metrics.get("human_following", 0.0)
        ))
    else:
        success = float(bool(final_metrics.get("human_following", 0.0)))
    summary = {
        "finish": bool(env.episode_over),
        "status": "Collision" if collision else ("Normal" if env.episode_over else "MaxSteps"),
        "success": success,
        "following_rate": followed / steps if steps else 0.0,
        "following_step": followed,
        "total_step": steps,
        "collision": collision,
        "visible_rate": visible_steps / steps if steps else 0.0,
        "coordinate_takeover_rate": coordinate_steps / steps if steps else 0.0,
        "perception_detection_rate": (
            perception_detected_steps / steps if steps else 0.0
        ),
        "perception_target_precision": (
            perception_correct_steps / perception_detected_steps
            if perception_detected_steps else 0.0
        ),
        "perception_target_recall": (
            perception_correct_steps / perception_gt_visible_steps
            if perception_gt_visible_steps else 0.0
        ),
        "perception_mean_target_iou": (
            float(np.mean(perception_target_ious)) if perception_target_ious else 0.0
        ),
        "goal_crop_deferred": bool(defer_goal_crop),
        "goal_crop_wait_steps": goal_crop_wait_steps,
        "goal_crop_initialized": goal_crop_initialized,
        "distance_start_m": distances[0] if distances else None,
        "distance_min_m": min(distances) if distances else None,
        "distance_mean_m": float(np.mean(distances)) if distances else None,
        "distance_max_m": max(distances) if distances else None,
        "distance_end_m": distances[-1] if distances else None,
        "final_metrics": {
            key: float(final_metrics[key])
            for key in (
                "human_following", "human_following_success",
                "human_collision", "distance_to_leader",
            )
            if key in final_metrics and np.isscalar(final_metrics[key])
        },
    }
    return summary, records


def main():
    import habitat
    from habitat.datasets import make_dataset

    import evt_bench  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("stt", "dt", "at"), required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene-dataset", default=DEFAULT_SCENE_DATASET)
    parser.add_argument("--min-distance", type=float, default=1.2)
    parser.add_argument("--max-distance", type=float, default=1.5)
    parser.add_argument("--max-forward", type=float, default=1.0)
    parser.add_argument("--max-lateral", type=float, default=1.0)
    parser.add_argument("--max-yaw", type=float, default=1.0)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="debug limit for rollout steps; does not affect full evaluation defaults",
    )
    parser.add_argument("--max-scenes-per-process", type=int)
    parser.add_argument("--dataset-index", type=int)
    parser.add_argument(
        "--exclude-dataset-indices",
        type=parse_dataset_indices,
        default=(),
        help="comma-separated global dataset indices to skip",
    )
    parser.add_argument("--save-steps", action="store_true")
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-fps", type=int, default=8)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--max-success-attempts", type=int, default=5)
    parser.add_argument("--evasion-side", type=float, choices=(-1.0, 1.0))
    parser.add_argument(
        "--perception", choices=("oracle", "rgb-person"), default="oracle"
    )
    parser.add_argument(
        "--controller",
        choices=("oracle-navmesh", "reactive", "map-reactive", "map-reactive-c2"),
        default="oracle-navmesh",
    )
    parser.add_argument(
        "--map-memory-frames",
        type=int,
        default=-1,
        help="C2 static-map memory: -1=controller default, 0=full episode, N=last N frames",
    )
    parser.add_argument("--map-camera-height", type=float, default=0.24)
    parser.add_argument("--person-detector-weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--person-reid-weights", default=str(DEFAULT_REID_WEIGHTS))
    parser.add_argument("--person-score-threshold", type=float, default=0.30)
    parser.add_argument("--perception-device", default="cuda")
    parser.add_argument(
        "--target-initialization",
        choices=("auto", "first-visible", "goal-crop"),
        default="auto",
    )
    parser.add_argument(
        "--lost-target-policy",
        choices=("auto", "coordinate", "stop-search"),
        default="auto",
    )
    parser.add_argument("--lost-brake-steps", type=int, default=2)
    parser.add_argument("--lost-search-yaw", type=float, default=0.35)
    parser.add_argument("--lost-search-period-steps", type=int, default=8)
    parser.add_argument("--lost-coast-steps", type=int, default=3)
    parser.add_argument("--lost-coast-min-range", type=float, default=2.0)
    parser.add_argument("--lost-coast-max-translation", type=float, default=0.35)
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("shard-id must be in [0, num-shards)")
    if args.max_scenes_per_process is not None and args.max_scenes_per_process <= 0:
        parser.error("max-scenes-per-process must be positive")
    if args.max_success_attempts <= 0:
        parser.error("max-success-attempts must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("max-steps must be positive")
    if args.map_memory_frames < -1:
        parser.error("map-memory-frames must be -1, 0, or a positive integer")
    if args.map_camera_height <= 0.0:
        parser.error("map-camera-height must be positive")
    if args.lost_brake_steps < 0:
        parser.error("lost-brake-steps must be non-negative")
    if args.lost_search_period_steps <= 0:
        parser.error("lost-search-period-steps must be positive")
    if args.lost_coast_steps < 0:
        parser.error("lost-coast-steps must be non-negative")
    if args.lost_coast_max_translation < 0.0:
        parser.error("lost-coast-max-translation must be non-negative")
    target_initialization = args.target_initialization
    if target_initialization == "auto":
        target_initialization = "goal-crop" if args.task == "dt" else "first-visible"
    lost_target_policy = args.lost_target_policy
    if lost_target_policy == "auto":
        lost_target_policy = "coordinate"

    config_kind = "train" if args.split == "train" else "infer"
    config_path = (
        "habitat-lab/habitat/config/benchmark/nav/track/"
        f"track_{config_kind}_{args.task}.yaml"
    )
    config = configure(habitat.get_config(config_path), args.scene_dataset)
    if args.perception == "rgb-person" or args.controller in (
        "map-reactive", "map-reactive-c2"
    ):
        from habitat.config import read_write

        agent_sensors = config.habitat.simulator.agents.agent_1.sim_sensors
        if "jaw_depth_sensor" not in agent_sensors:
            depth_template_path = (
                "habitat-lab/habitat/config/benchmark/nav/track/"
                f"track_{config_kind}_stt.yaml"
            )
            depth_template = habitat.get_config(depth_template_path)
            depth_sensor = copy.deepcopy(
                depth_template.habitat.simulator.agents.agent_1
                .sim_sensors.jaw_depth_sensor
            )
        with read_write(config):
            if "jaw_depth_sensor" not in agent_sensors:
                agent_sensors.jaw_depth_sensor = depth_sensor
            obs_keys = config.habitat.gym.obs_keys
            for key in (RGB_KEY, DEPTH_KEY, PANOPTIC_KEY):
                if key not in obs_keys:
                    obs_keys.append(key)
    dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    total_dataset_episodes = len(dataset.episodes)
    invalid_exclusions = [
        index for index in args.exclude_dataset_indices
        if index >= total_dataset_episodes
    ]
    if invalid_exclusions:
        parser.error(
            f"excluded dataset indices are outside the dataset: {invalid_exclusions}"
        )
    if args.dataset_index is not None:
        if not 0 <= args.dataset_index < total_dataset_episodes:
            parser.error("dataset-index is outside the dataset")
        indexed = [(args.dataset_index, dataset.episodes[args.dataset_index])]
    else:
        indexed = [
            (index, episode) for index, episode in enumerate(dataset.episodes)
            if index % args.num_shards == args.shard_id
        ]
    assigned_episodes = len(indexed)
    excluded_indices = set(args.exclude_dataset_indices)
    excluded_episodes = sum(index in excluded_indices for index, _ in indexed)
    indexed = [
        (index, episode) for index, episode in indexed
        if index not in excluded_indices
    ]
    if args.max_episodes is not None:
        indexed = indexed[: args.max_episodes]
    output_root = Path(args.output_root) / args.task / args.split
    episodes_dir = output_root / "episodes"
    videos_dir = output_root / "videos"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    if args.save_video:
        videos_dir.mkdir(parents=True, exist_ok=True)

    if args.require_success:
        exhausted = []
        for index, episode in indexed:
            result_path = episodes_dir / f"{episode_key(index, episode)}.json"
            if exhausted_success_attempts(result_path, args.max_success_attempts):
                exhausted.append(str(result_path))
        if exhausted:
            print(json.dumps({
                "event": "oracle_batch_success_attempts_exhausted",
                "count": len(exhausted),
                "max_success_attempts": args.max_success_attempts,
                "examples": exhausted[:10],
            }, indent=2))
            raise SystemExit(76)
    if not args.no_resume:
        indexed = [
            (index, episode) for index, episode in indexed
            if not completed_result(
                index, episode, episodes_dir, videos_dir, args.save_video,
                require_success=args.require_success,
                controller=args.controller,
            )
        ]
    pending_before = len(indexed)
    if args.max_scenes_per_process is not None:
        selected_scenes = []
        selected_scene_set = set()
        for _, episode in indexed:
            scene = scene_key(episode.scene_id)
            if scene not in selected_scene_set:
                if len(selected_scenes) >= args.max_scenes_per_process:
                    continue
                selected_scenes.append(scene)
                selected_scene_set.add(scene)
        indexed = [
            (index, episode) for index, episode in indexed
            if scene_key(episode.scene_id) in selected_scene_set
        ]
    remaining_episodes = pending_before - len(indexed)
    for index, episode in indexed:
        episode.info["_oracle_batch_index"] = index
    dataset.episodes = [episode for _, episode in indexed]

    controller_kwargs = {}
    if CONTROLLER_VERSION >= 6:
        controller_kwargs["tracking_mask_min_pixels"] = (
            12000 if args.split == "train" else 3600
        )
    controller = OracleNavmeshFollower(
        min_distance_m=args.min_distance,
        max_distance_m=args.max_distance,
        max_forward=args.max_forward,
        max_lateral=args.max_lateral,
        max_yaw=args.max_yaw,
        lost_target_policy=lost_target_policy,
        lost_brake_steps=args.lost_brake_steps,
        lost_search_yaw=args.lost_search_yaw,
        lost_search_period_steps=args.lost_search_period_steps,
        lost_coast_steps=args.lost_coast_steps,
        lost_coast_min_range_m=args.lost_coast_min_range,
        lost_coast_max_translation=args.lost_coast_max_translation,
        **controller_kwargs,
    )
    if args.controller == "reactive":
        controller = ModularReactiveFollower(
            min_distance_m=args.min_distance,
            max_distance_m=args.max_distance,
            max_forward=args.max_forward,
            max_lateral=args.max_lateral,
            max_yaw=args.max_yaw,
            lost_search_yaw=args.lost_search_yaw,
            use_invisible_pointgoal=args.perception == "oracle",
        )
    elif args.controller in ("map-reactive", "map-reactive-c2"):
        if args.controller == "map-reactive-c2":
            map_memory_frames = (
                4 if args.map_memory_frames == -1 else args.map_memory_frames
            )
            map_memory_frames = map_memory_frames or None
            camera_height_m = args.map_camera_height
            min_obstacle_height_m = 0.08
        else:
            map_memory_frames = None
            camera_height_m = 0.85
            min_obstacle_height_m = 0.06
        controller = MapReactiveFollower(
            min_distance_m=args.min_distance,
            max_distance_m=args.max_distance,
            max_forward=args.max_forward,
            max_lateral=args.max_lateral,
            max_yaw=args.max_yaw,
            lost_search_yaw=args.lost_search_yaw,
            use_invisible_pointgoal=args.perception == "oracle",
            camera_height_m=camera_height_m,
            map_memory_frames=map_memory_frames,
            min_obstacle_height_m=min_obstacle_height_m,
        )
    if args.perception == "oracle":
        perception = OraclePerception()
        perception_name = "oracle-panoptic-pose"
    else:
        # Habitat must establish its EGL context before Torch initializes CUDA;
        # doing this in the opposite order produces black jaw-camera frames.
        perception = None
        perception_name = RGBPersonPerceptionWorker.name
    counts = {
        "completed": 0, "skipped": 0, "errors": 0,
        "unsuccessful": 0, "exhausted": 0,
    }
    manifest = {
        "task": args.task,
        "split": args.split,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "dataset_episodes": total_dataset_episodes,
        "assigned_episodes": assigned_episodes,
        "excluded_dataset_indices": list(args.exclude_dataset_indices),
        "excluded_episodes": excluded_episodes,
        "pending_before": pending_before,
        "selected_episodes": len(indexed),
        "remaining_episodes": remaining_episodes,
        "config": config_path,
        "controller": args.controller,
        "map_memory_frames": getattr(controller.obstacle_map, "memory_frames", None) if hasattr(controller, "obstacle_map") else None,
        "map_camera_height_m": getattr(controller.obstacle_map, "camera_height_m", None) if hasattr(controller, "obstacle_map") else None,
        "controller_input": (
            "oracle-pointgoal"
            if getattr(controller, "uses_invisible_pointgoal", False)
            else "target-observation"
        ),
        "perception": perception_name,
        "person_score_threshold": args.person_score_threshold,
        "target_initialization": target_initialization,
        "lost_target_policy": lost_target_policy,
        "lost_brake_steps": args.lost_brake_steps,
        "lost_search_yaw": args.lost_search_yaw,
        "lost_search_period_steps": args.lost_search_period_steps,
        "lost_coast_steps": args.lost_coast_steps,
        "lost_coast_min_range": args.lost_coast_min_range,
        "lost_coast_max_translation": args.lost_coast_max_translation,
        "lost_retreat_steps": controller.lost_retreat_steps,
    }
    atomic_json(output_root / f"shard_{args.shard_id:03d}_manifest.json", manifest)
    import time as _time
    _wall_start = _time.monotonic()
    _total_to_run = len(indexed)
    _skipped_resume = assigned_episodes - excluded_episodes - pending_before
    print(
        f"[shard {args.shard_id}] Starting: {_total_to_run} episodes to run "
        f"({assigned_episodes} assigned, "
        f"{_skipped_resume} already completed (resume), "
        f"{excluded_episodes} excluded, "
        f"{pending_before} pending)",
        flush=True,
    )

    if not indexed:
        worker_summary = {**manifest, **counts}
        atomic_json(output_root / f"shard_{args.shard_id:03d}_summary.json", worker_summary)
        print(json.dumps({"event": "oracle_batch_complete", **worker_summary}, indent=2))
        return

    with habitat.TrackEnv(config=config, dataset=dataset) as env:
        for _ep_idx in range(_total_to_run):
            observations = env.reset()
            assigned_human_ids = assign_unique_humanoid_semantic_ids(env)
            observations = env.sim.get_sensor_observations()
            if perception is None:
                # Force Habitat's first RGB render before the detector worker
                # establishes a CUDA context on the same physical GPU.
                perception = RGBPersonPerceptionWorker(
                weights_path=args.person_detector_weights,
                reid_weights_path=args.person_reid_weights,
                    score_threshold=args.person_score_threshold,
                    device=args.perception_device,
                )
            episode = env.current_episode
            index = int(episode.info["_oracle_batch_index"])
            key = episode_key(index, episode)
            result_path = episodes_dir / f"{key}.json"
            video_path = videos_dir / f"{key}.mp4" if args.save_video else None
            success_attempt = prior_success_attempt(result_path) + 1
            metadata = {
                "schema_version": 1,
                "controller_version": CONTROLLER_VERSION,
                "controller": args.controller,
                "map_memory_frames": getattr(controller.obstacle_map, "memory_frames", None) if hasattr(controller, "obstacle_map") else None,
                "map_camera_height_m": getattr(controller.obstacle_map, "camera_height_m", None) if hasattr(controller, "obstacle_map") else None,
                "controller_input": (
                    "oracle-pointgoal"
                    if getattr(controller, "uses_invisible_pointgoal", False)
                    else "target-observation"
                ),
                "task": args.task,
                "split": args.split,
                "dataset_index": index,
                "episode_key": key,
                "episode_id": str(episode.episode_id),
                "scene_id": episode.scene_id,
                "robot_start_position": episode.info.get("robot_position"),
                "target_name": episode.info.get("main_humanoid_name"),
                "target_semantic_id": episode.info.get("main_human_semantic_id"),
                "human_agent_semantic_ids": assigned_human_ids,
                "instruction": episode.info.get("instruction"),
                "video": str(video_path) if video_path is not None else None,
                "success_attempt": success_attempt,
                "perception": perception_name,
                "person_score_threshold": args.person_score_threshold,
                "target_initialization": target_initialization,
                "lost_target_policy": lost_target_policy,
                "lost_brake_steps": args.lost_brake_steps,
                "lost_search_yaw": args.lost_search_yaw,
                "lost_search_period_steps": args.lost_search_period_steps,
                "lost_coast_steps": args.lost_coast_steps,
                "lost_coast_min_range": args.lost_coast_min_range,
                "lost_coast_max_translation": args.lost_coast_max_translation,
                "lost_retreat_steps": controller.lost_retreat_steps,
            }
            try:
                evasion_side = args.evasion_side
                if evasion_side is None and args.require_success:
                    previous_side = prior_evasion_side(result_path)
                    if previous_side is not None:
                        evasion_side = -previous_side
                controller.reset(evasion_side=evasion_side)
                defer_goal_crop = False
                if hasattr(perception, "reset"):
                    reference_rgb = None
                    if (
                        args.perception == "rgb-person"
                        and target_initialization == "goal-crop"
                    ):
                        reference_rgb = target_goal_crop(
                            observations,
                            episode.info["main_human_semantic_id"],
                        )
                        if reference_rgb is None:
                            defer_goal_crop = True
                            print(json.dumps({
                                "event": "oracle_batch_goal_crop_deferred",
                                "task": args.task,
                                "split": args.split,
                                "dataset_index": index,
                                "episode_key": key,
                                "message": (
                                    "Target is not visible at reset; ReID initialization "
                                    "will occur on its first visible frame"
                                ),
                            }), flush=True)
                    perception.reset(reference_rgb=reference_rgb)
                summary, records = evaluate_episode(
                    env, observations, controller,
                    max_steps=(
                        args.max_steps
                        if args.max_steps is not None
                        else config.habitat.environment.max_episode_steps
                    ),
                    save_steps=args.save_steps,
                    video_path=video_path,
                    video_fps=args.video_fps,
                    perception=perception,
                    defer_goal_crop=defer_goal_crop,
                )
                summary["evasion_side"] = controller._evasion_side
                result = {**metadata, "summary": summary}
                if args.save_steps:
                    result["steps"] = records
                atomic_json(result_path, result)
                counts["completed"] += 1
                elapsed = _time.monotonic() - _wall_start
                done = counts["completed"]
                rate = done / elapsed if elapsed > 0 else 0
                remaining = _total_to_run - done
                eta = remaining / rate if rate > 0 else float("inf")
                print(
                    f"[shard {args.shard_id}] "
                    f"{done}/{_total_to_run} episodes | "
                    f"{elapsed/60:.1f} min elapsed | "
                    f"{rate:.1f} ep/min | "
                    f"ETA {eta/60:.1f} min",
                    flush=True,
                )
                if args.require_success and not summary["success"]:
                    counts["unsuccessful"] += 1
                    if success_attempt >= args.max_success_attempts:
                        counts["exhausted"] += 1
                if not summary["success"]:
                    print(json.dumps({
                        "event": "oracle_batch_unsuccessful",
                        "task": args.task,
                        "split": args.split,
                        "dataset_index": index,
                        "episode_key": key,
                        "result_path": str(result_path),
                        "video_path": str(video_path) if video_path is not None else None,
                        "success_attempt": success_attempt,
                        "max_success_attempts": args.max_success_attempts,
                        "summary": summary,
                    }, indent=2), flush=True)
                print(json.dumps({
                    "event": "oracle_batch_episode",
                    "key": key,
                    "success": summary["success"],
                    "tr": summary["following_rate"],
                    "cr": summary["collision"],
                    "steps": summary["total_step"],
                    "counts": counts,
                }), flush=True)
            except Exception as exc:
                counts["errors"] += 1
                elapsed = _time.monotonic() - _wall_start
                done = counts["completed"] + counts["errors"]
                print(
                    f"[shard {args.shard_id}] "
                    f"ERROR ({counts['errors']} total): {exc} | "
                    f"{done}/{_total_to_run} done | "
                    f"{elapsed/60:.1f} min elapsed",
                    flush=True,
                )
                error = {
                    **metadata,
                    "error": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                atomic_json(result_path, error)
                print(json.dumps({
                    "event": "oracle_batch_error", "key": key,
                    "task": args.task, "split": args.split,
                    "dataset_index": index,
                    "result_path": str(result_path),
                    "video_path": str(video_path) if video_path is not None else None,
                    "success_attempt": success_attempt,
                    "error": error["error"], "message": error["message"],
                    "traceback": error["traceback"],
                }, indent=2), flush=True)
                if not args.continue_on_error:
                    raise

        if hasattr(perception, "close"):
            perception.close()

    _wall_total = _time.monotonic() - _wall_start
    worker_summary = {**manifest, **counts, "wall_time_seconds": round(_wall_total, 1)}
    atomic_json(output_root / f"shard_{args.shard_id:03d}_summary.json", worker_summary)
    print(json.dumps({"event": "oracle_batch_complete", **worker_summary}, indent=2))
    print(
        f"[shard {args.shard_id}] Done: {counts['completed']}/{_total_to_run} completed, "
        f"{counts['errors']} errors, {counts['unsuccessful']} unsuccessful, "
        f"{counts['exhausted']} exhausted | "
        f"Total time: {_wall_total/60:.1f} min",
        flush=True,
    )
    if counts["errors"]:
        raise SystemExit(2)
    if counts["exhausted"]:
        raise SystemExit(76)
    if counts["unsuccessful"]:
        # The durable failed result remains pending when --require-success is
        # enabled, so the launcher will retry it in a fresh simulator process.
        raise SystemExit(75)
    if remaining_episodes:
        raise SystemExit(75)


if __name__ == "__main__":
    main()
