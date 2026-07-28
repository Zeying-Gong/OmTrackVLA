"""Replay recorded Habitat tracking actions and export aligned RGB-D frames."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import habitat
import imageio.v3 as iio
import numpy as np
from habitat.config import read_write
from habitat.datasets import make_dataset

import evt_bench  # noqa: F401 - register tracking actions and sensors


RGB_KEY = "agent_1_articulated_agent_jaw_rgb"
DEPTH_KEY = "agent_1_articulated_agent_jaw_depth"
ACTION_NAMES = (
    "agent_0_humanoid_navigate_action",
    "agent_1_base_velocity",
    "agent_2_oracle_nav_randcoord_action_obstacle",
    "agent_3_oracle_nav_randcoord_action_obstacle",
    "agent_4_oracle_nav_randcoord_action_obstacle",
    "agent_5_oracle_nav_randcoord_action_obstacle",
    "agent_6_oracle_nav_randcoord_action_obstacle",
    "agent_7_oracle_nav_randcoord_action_obstacle",
    "agent_8_oracle_nav_randcoord_action_obstacle",
)
REPLAY_DISABLED_MEASUREMENTS = ("top_down_map_following",)
OUTPUT_SCHEMA_VERSION = 7
MAX_INITIAL_ROBOT_PLANAR_ERROR_M = 0.5


def scene_key(scene_id):
    return Path(scene_id).name.split(".")[0]


def episode_key(episode):
    return scene_key(episode.scene_id), str(episode.episode_id)


def source_index(root):
    index = {}
    for path in Path(root).rglob("*_info.json"):
        episode_id = path.name[:-10]
        # Released layout: .../<scene>/<episode>/<episode>_info.json
        scene = path.parent.parent.name
        index[(scene, episode_id)] = path
    return index


def source_start_position(path):
    with path.open() as f:
        steps = json.load(f)
    if not steps or steps[0].get("robot_pos_pre") is None:
        return None
    return np.asarray(steps[0]["robot_pos_pre"], dtype=np.float32)


def choose_episode(candidates, source_path):
    """Disambiguate duplicate scene/episode IDs using the recorded robot pose."""
    source_pos = source_start_position(source_path)
    if source_pos is None:
        return candidates[0]
    scored = []
    for episode in candidates:
        dataset_pos = episode.info.get("robot_position")
        if dataset_pos is None:
            score = float("inf")
        else:
            # Vertical offsets differ before/after Spot settles; match on floor XY (world XZ).
            dataset_pos = np.asarray(dataset_pos, dtype=np.float32)
            score = float(np.linalg.norm(dataset_pos[[0, 2]] - source_pos[[0, 2]]))
        scored.append((score, episode))
    score, episode = min(scored, key=lambda item: item[0])
    if not np.isfinite(score) or score > 0.25:
        raise RuntimeError(
            f"No duplicate episode matches {source_path}; best initial XZ error={score:.3f}m"
        )
    return episode


def is_complete(output_root, task, episode):
    path = (
        Path(output_root) / task / scene_key(episode.scene_id)
        / str(episode.episode_id) / "complete.json"
    )
    try:
        return json.loads(path.read_text()).get("schema_version", 0) >= 6
    except (OSError, ValueError):
        return False


def matrix_list(transform):
    return np.asarray(transform, dtype=np.float32).tolist()


def local_target(robot, target):
    delta = target.base_pos - robot.base_pos
    local = robot.sim_obj.transformation.inverted().transform_vector(delta)
    # BaseVelAction maps longitudinal/lateral to local x/-z.
    return [float(local.x), float(-local.z), 0.0]


def apply_recorded_human_positions(env, source):
    """Restore recorded human translations before rendering the current frame."""
    positions = [(0, source.get("target_pos"))]
    positions.extend(enumerate(source.get("other_humans_pos", []), start=2))
    for agent_index, position in positions:
        if position is None or agent_index >= len(env.sim.agents_mgr):
            continue
        env.sim.agents_mgr[agent_index].articulated_agent.base_pos = np.asarray(
            position, dtype=np.float32
        )


def clear_replay_termination(env):
    """The recorded step count, rather than randomized oracle wait time, ends replay."""
    env.task.should_end = False
    env.task._is_episode_active = True
    env.task.is_stop_called = False
    env._episode_over = False


def configure_replay(config):
    """Remove metrics that are unrelated to rendering recorded trajectories."""
    disabled = []
    with read_write(config):
        measurements = config.habitat.task.measurements
        for name in REPLAY_DISABLED_MEASUREMENTS:
            if name in measurements:
                del measurements[name]
                disabled.append(name)
    return disabled


def replay_error_context(env, source_path, task):
    lower_bound, upper_bound = env.sim.pathfinder.get_bounds()
    agents = {}
    for agent_index, agent_data in enumerate(env.sim.agents_mgr):
        agents[str(agent_index)] = np.asarray(
            agent_data.articulated_agent.base_pos, dtype=np.float32
        ).tolist()
    return {
        "event": "rgbd_replay_episode_error",
        "task": task,
        "scene_id": scene_key(env.current_episode.scene_id),
        "episode_id": str(env.current_episode.episode_id),
        "source": str(source_path),
        "agent_base_positions": agents,
        "navmesh_lower_bound": np.asarray(lower_bound).tolist(),
        "navmesh_upper_bound": np.asarray(upper_bound).tolist(),
    }


def metric_depth(raw, max_depth):
    depth = np.asarray(raw, dtype=np.float32).squeeze()
    depth[~np.isfinite(depth)] = 0.0
    if depth.size and float(depth.max()) <= 1.01:
        depth *= max_depth
    depth[depth < 0.0] = 0.0
    return depth


def save_frame(ep_dir, sample_idx, obs, max_depth, depth_scale):
    rgb_rel = f"rgb/{sample_idx:06d}.jpg"
    depth_rel = f"depth/{sample_idx:06d}.png"
    rgb = np.asarray(obs[RGB_KEY])[..., :3]
    depth = metric_depth(obs[DEPTH_KEY], max_depth)
    iio.imwrite(ep_dir / rgb_rel, rgb, quality=95)
    iio.imwrite(
        ep_dir / depth_rel,
        np.clip(depth * depth_scale, 0, 65535).astype(np.uint16),
    )
    return rgb_rel, depth_rel, depth


def replay_episode(env, source_path, output_root, task, stride, max_depth, depth_scale):
    episode = env.current_episode
    scene = scene_key(episode.scene_id)
    episode_id = str(episode.episode_id)
    ep_dir = output_root / task / scene / episode_id
    done_path = ep_dir / "complete.json"
    if done_path.exists():
        try:
            if json.loads(done_path.read_text()).get("schema_version", 0) >= OUTPUT_SCHEMA_VERSION:
                return "skipped"
        except (OSError, ValueError):
            pass

    with source_path.open() as f:
        source_steps = json.load(f)
    if not source_steps:
        return "empty"

    (ep_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (ep_dir / "depth").mkdir(parents=True, exist_ok=True)
    manifest_tmp = ep_dir / "frames.jsonl.tmp"
    sample_idx = 0
    replay_planar_errors = []

    robot = env.sim.agents_mgr[1].articulated_agent
    target = env.sim.agents_mgr[0].articulated_agent
    with manifest_tmp.open("w") as manifest:
        for step_idx, source in enumerate(source_steps):
            clear_replay_termination(env)
            apply_recorded_human_positions(env, source)
            obs = env.sim.get_sensor_observations()
            if RGB_KEY not in obs or DEPTH_KEY not in obs:
                raise KeyError(f"RGB-D observation missing; keys={sorted(obs)}")

            if step_idx % stride == 0:
                rgb_rel, depth_rel, depth = save_frame(
                    ep_dir, sample_idx, obs, max_depth, depth_scale
                )
                actual_robot_pos = np.asarray(robot.base_pos, dtype=np.float32)
                source_robot_pos = source.get("robot_pos_pre")
                planar_error = None
                if source_robot_pos is not None:
                    source_robot_pos = np.asarray(source_robot_pos, dtype=np.float32)
                    planar_error = float(np.linalg.norm(
                        actual_robot_pos[[0, 2]] - source_robot_pos[[0, 2]]
                    ))
                    if step_idx == 0 and planar_error > MAX_INITIAL_ROBOT_PLANAR_ERROR_M:
                        raise RuntimeError(
                            "Replay episode does not match its recorded robot pose: "
                            f"initial XZ error={planar_error:.3f}m, "
                            f"actual={actual_robot_pos.tolist()}, "
                            f"source={source_robot_pos.tolist()}"
                        )
                    replay_planar_errors.append(planar_error)
                row = {
                    "task": task,
                    "scene_id": scene,
                    "episode_id": episode_id,
                    "sample_index": sample_idx,
                    "sim_step": step_idx,
                    "source_info": str(source_path),
                    "rgb": rgb_rel,
                    "depth": depth_rel,
                    "depth_valid_fraction": float(((depth >= 0.1) & (depth <= 5.0)).mean()),
                    "point_goal": local_target(robot, target),
                    "robot_transform": matrix_list(robot.sim_obj.transformation),
                    "target_transform": matrix_list(target.sim_obj.transformation),
                    "replay_robot_planar_error_m": planar_error,
                    "action": source.get("base_velocity_cmd", source.get("base_velocity")),
                    "source_step": source,
                }
                manifest.write(json.dumps(row, separators=(",", ":")) + "\n")
                sample_idx += 1

            action = source.get("base_velocity_cmd", source.get("base_velocity"))
            if action is None or len(action) != 3:
                raise ValueError(f"Invalid action at {source_path}:{step_idx}")
            env.step({
                "action": ACTION_NAMES,
                "action_args": {"agent_1_base_vel": action},
            })
            clear_replay_termination(env)

    os.replace(manifest_tmp, ep_dir / "frames.jsonl")
    done_tmp = ep_dir / "complete.json.tmp"
    done_tmp.write_text(json.dumps({
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "rgb_depth_calibration": "spot_jaw_colocated_v1",
        "human_replay": "recorded_translation_v2_agent_indices",
        "task": task,
        "scene_id": scene,
        "episode_id": episode_id,
        "source_steps": len(source_steps),
        "replayed_steps": min(len(source_steps), step_idx + 1),
        "saved_frames": sample_idx,
        "stride": stride,
        "depth_scale": depth_scale,
        "depth_unit": "meter",
        "mean_replay_robot_planar_error_m": (
            float(np.mean(replay_planar_errors)) if replay_planar_errors else None
        ),
        "max_replay_robot_planar_error_m": (
            float(np.max(replay_planar_errors)) if replay_planar_errors else None
        ),
    }, indent=2) + "\n")
    os.replace(done_tmp, done_path)
    return "written"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("stt", "dt", "at"), required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument(
        "--depth-max", type=float, default=10.0,
        help="Sensor max range, used only if Habitat returns normalized [0,1] depth.",
    )
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--one-scene-per-process", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("shard-id must be in [0, num-shards)")

    config_path = args.config or f"habitat-lab/habitat/config/benchmark/nav/track/track_train_{args.task}.yaml"
    config = habitat.get_config(config_path, args.opts)
    disabled_measurements = configure_replay(config)
    random.seed(config.habitat.simulator.seed)
    np.random.seed(config.habitat.simulator.seed)
    dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    sources = source_index(args.source_root)
    candidates = {}
    for episode in dataset.episodes:
        key = episode_key(episode)
        if key not in sources:
            continue
        if args.episode_id is not None and str(episode.episode_id) != args.episode_id:
            continue
        candidates.setdefault(key, []).append(episode)
    matched = [
        (choose_episode(episodes, sources[key]), sources[key])
        for key, episodes in candidates.items()
    ]
    matched = matched[args.shard_id :: args.num_shards]
    if args.max_episodes is not None:
        matched = matched[: args.max_episodes]
    if not matched:
        raise RuntimeError("No dataset episodes matched the recorded source files")
    assigned_count = len(matched)
    matched = [
        item for item in matched
        if not is_complete(args.output_root, args.task, item[0])
    ]
    if not matched:
        print(json.dumps({
            "event": "rgbd_replay_already_complete",
            "task": args.task,
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "assigned": assigned_count,
        }, indent=2), flush=True)
        return
    pending_count = len(matched)
    if args.one_scene_per_process:
        first_scene = scene_key(matched[0][0].scene_id)
        matched = [item for item in matched if scene_key(item[0].scene_id) == first_scene]
    has_more_pending = len(matched) < pending_count
    dataset.episodes = [item[0] for item in matched]
    source_by_episode = {episode_key(episode): source for episode, source in matched}
    if len(source_by_episode) != len(matched):
        raise RuntimeError("Matched replay episodes do not have unique scene/episode IDs")

    output_root = Path(args.output_root)
    counts = {"written": 0, "skipped": 0, "empty": 0}
    print(json.dumps({
        "event": "rgbd_replay_start",
        "task": args.task,
        "config": config_path,
        "source_root": args.source_root,
        "output_root": args.output_root,
        "matched": len(matched),
        "assigned": assigned_count,
        "pending_before_process": pending_count,
        "one_scene_per_process": args.one_scene_per_process,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "stride": args.stride,
        "depth_max": args.depth_max,
        "depth_scale": args.depth_scale,
        "seed": int(config.habitat.simulator.seed),
        "disabled_measurements": disabled_measurements,
    }, indent=2), flush=True)
    with habitat.TrackEnv(config=config, dataset=dataset) as env:
        for _ in range(len(matched)):
            env.reset()
            current_key = episode_key(env.current_episode)
            try:
                source_path = source_by_episode[current_key]
            except KeyError:
                raise RuntimeError(
                    "Environment selected an episode outside the replay shard: "
                    f"scene={current_key[0]} episode={current_key[1]}"
                ) from None
            print(json.dumps({
                "event": "rgbd_replay_episode_start",
                "task": args.task,
                "scene_id": scene_key(env.current_episode.scene_id),
                "episode_id": str(env.current_episode.episode_id),
                "source": str(source_path),
            }), flush=True)
            try:
                status = replay_episode(
                    env, source_path, output_root, args.task, args.stride,
                    args.depth_max, args.depth_scale,
                )
            except Exception:
                print(
                    json.dumps(replay_error_context(env, source_path, args.task)),
                    flush=True,
                )
                raise
            counts[status] += 1
            print(f"[rgbd-replay] {status} {source_path} counts={counts}", flush=True)
    print(json.dumps({"matched": len(matched), **counts}, indent=2))
    if has_more_pending:
        print("[rgbd-replay] scene batch complete; more assigned episodes remain", flush=True)
        raise SystemExit(75)


if __name__ == "__main__":
    main()
