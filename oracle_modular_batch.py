#!/usr/bin/env python3
"""Sharded dataset evaluation for the modular oracle person follower."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from oracle_modular_follow import (
    ACTION_NAMES,
    DEFAULT_SCENE_DATASET,
    DEPTH_KEY,
    PANOPTIC_KEY,
    RGB_KEY,
    OraclePerception,
    annotate,
    configure,
    local_target,
    target_mask_to_bbox,
)
from rgb_person_perception import (
    DEFAULT_WEIGHTS,
    RGBPersonPerception,
    RGBPersonPerceptionWorker,
    bbox_iou,
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


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def completed_result(
    index, episode, episodes_dir, videos_dir, save_video, require_success=False,
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
    video_path=None, video_fps=8, perception=None,
):
    episode = env.current_episode
    robot = env.sim.agents_mgr[1].articulated_agent
    target_agent = env.sim.agents_mgr[0].articulated_agent
    target_semantic_id = int(episode.info["main_human_semantic_id"])
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
            required = (
                (RGB_KEY, PANOPTIC_KEY)
                if isinstance(perception, OraclePerception)
                else (RGB_KEY, DEPTH_KEY)
            )
            missing = [key for key in required if key not in observations]
            if missing:
                raise KeyError(f"Missing observations {missing}; got {sorted(observations)}")
            if isinstance(perception, OraclePerception):
                target = perception(
                    observations[RGB_KEY], observations[PANOPTIC_KEY], target_semantic_id,
                    local_target(robot, target_agent),
                )
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
            decision = controller(env.sim, robot, target_agent, target)
            if not target.visible:
                coordinate_steps += 1
            if writer is not None:
                current_following = env.get_metrics().get("human_following")
                frame = annotate(
                    observations[RGB_KEY], target, decision,
                    f"{episode.info.get('_oracle_batch_index')} | {scene_key(episode.scene_id)} | ep {episode.episode_id} | step {steps}",
                    current_following,
                )
                writer.append_data(np.asarray(frame))

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
            visible_steps += int(visible)
            if save_steps:
                records.append({
                    "step": steps,
                    "distance_m": current_distance,
                    "visible": visible,
                    "mask_area": target.mask_area,
                    "rgb_mean": float(np.asarray(observations[RGB_KEY])[..., :3].mean()),
                    "perception_confidence": target.confidence,
                    "perception_candidate_count": getattr(
                        perception, "last_candidate_count", None
                    ),
                    "perception_association_score": getattr(
                        perception, "last_association_score", None
                    ),
                    "target_bbox_iou": target_iou,
                    "target_selection_correct": target_selection_correct,
                    "mode": decision.mode,
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
    parser.add_argument("--person-detector-weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--person-score-threshold", type=float, default=0.55)
    parser.add_argument("--perception-device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("shard-id must be in [0, num-shards)")
    if args.max_scenes_per_process is not None and args.max_scenes_per_process <= 0:
        parser.error("max-scenes-per-process must be positive")
    if args.max_success_attempts <= 0:
        parser.error("max-success-attempts must be positive")

    config_kind = "train" if args.split == "train" else "infer"
    config_path = (
        "habitat-lab/habitat/config/benchmark/nav/track/"
        f"track_{config_kind}_{args.task}.yaml"
    )
    config = configure(habitat.get_config(config_path), args.scene_dataset)
    if args.perception == "rgb-person":
        from habitat.config import read_write

        with read_write(config):
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
        **controller_kwargs,
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
        "perception": perception_name,
    }
    atomic_json(output_root / f"shard_{args.shard_id:03d}_manifest.json", manifest)

    if not indexed:
        worker_summary = {**manifest, **counts}
        atomic_json(output_root / f"shard_{args.shard_id:03d}_summary.json", worker_summary)
        print(json.dumps({"event": "oracle_batch_complete", **worker_summary}, indent=2))
        return

    with habitat.TrackEnv(config=config, dataset=dataset) as env:
        for _ in range(len(indexed)):
            observations = env.reset()
            if perception is None:
                # Force Habitat's first RGB render before the detector worker
                # establishes a CUDA context on the same physical GPU.
                perception = RGBPersonPerceptionWorker(
                    weights_path=args.person_detector_weights,
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
                "task": args.task,
                "split": args.split,
                "dataset_index": index,
                "episode_key": key,
                "episode_id": str(episode.episode_id),
                "scene_id": episode.scene_id,
                "robot_start_position": episode.info.get("robot_position"),
                "target_name": episode.info.get("main_humanoid_name"),
                "target_semantic_id": episode.info.get("main_human_semantic_id"),
                "instruction": episode.info.get("instruction"),
                "video": str(video_path) if video_path is not None else None,
                "success_attempt": success_attempt,
                "perception": perception_name,
            }
            try:
                evasion_side = args.evasion_side
                if evasion_side is None and args.require_success:
                    previous_side = prior_evasion_side(result_path)
                    if previous_side is not None:
                        evasion_side = -previous_side
                controller.reset(evasion_side=evasion_side)
                if hasattr(perception, "reset"):
                    perception.reset()
                summary, records = evaluate_episode(
                    env, observations, controller,
                    max_steps=config.habitat.environment.max_episode_steps,
                    save_steps=args.save_steps,
                    video_path=video_path,
                    video_fps=args.video_fps,
                    perception=perception,
                )
                summary["evasion_side"] = controller._evasion_side
                result = {**metadata, "summary": summary}
                if args.save_steps:
                    result["steps"] = records
                atomic_json(result_path, result)
                counts["completed"] += 1
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

    worker_summary = {**manifest, **counts}
    atomic_json(output_root / f"shard_{args.shard_id:03d}_summary.json", worker_summary)
    print(json.dumps({"event": "oracle_batch_complete", **worker_summary}, indent=2))
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
