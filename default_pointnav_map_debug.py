#!/usr/bin/env python3
"""RGB-D obstacle-map smoke test with Habitat's default PointNav agent.

This intentionally does not load the multi-human tracking task or Spot.  It
isolates the Ascent/VLFM-style height-filtered map on Habitat's standard
``main_agent`` RGB-D camera and writes a short diagnostic video.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import habitat
import imageio.v2 as imageio
import numpy as np
from habitat.utils.geometry_utils import quaternion_rotate_vector

from modular_obstacle_map import LocalObstacleMap
from oracle_modular_batch import compose_rgbd_video_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=int, default=49)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--no-inflation", action="store_true")
    parser.add_argument("--navmesh-calibration", action="store_true")
    parser.add_argument("--min-obstacle-height", type=float, default=0.20)
    parser.add_argument("--max-obstacle-height", type=float, default=1.20)
    parser.add_argument("--output", default="outputs/default_pointnav_map_index49.mp4")
    parser.add_argument(
        "--dataset-path",
        default="data/datasets/track/AT/val/val.json.gz",
        help="PointNav-compatible episode file; defaults to the AT scene for comparison",
    )
    parser.add_argument(
        "--config",
        default="habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_hm3d.yaml",
    )
    args = parser.parse_args()

    config = habitat.get_config(args.config)
    from habitat.config import read_write

    with read_write(config):
        config.habitat.dataset.data_path = args.dataset_path
        config.habitat.dataset.split = "val"
        config.habitat.simulator.scene_dataset = (
            "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
        )
        # The AT file is PointNav-compatible for scene/start poses but carries
        # tracking metadata instead of a PointNav goal. Remove goal-dependent
        # task sensors/measures; RGB-D rendering is all this diagnostic needs.
        if "pointgoal_with_gps_compass" in config.habitat.task.lab_sensors:
            del config.habitat.task.lab_sensors["pointgoal_with_gps_compass"]
        config.habitat.task.measurements = {}
        config.habitat.task.reward_measure = ""
        config.habitat.task.success_measure = ""
    dataset = habitat.make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    if not 0 <= args.dataset_index < len(dataset.episodes):
        raise IndexError(f"dataset-index {args.dataset_index} outside {len(dataset.episodes)} episodes")
    dataset.episodes = [dataset.episodes[args.dataset_index]]
    # Tracking JSON stores goals as [[x, y, z]], while Nav-v0 expects [x, y, z].
    # Normalize only the copied diagnostic episode for the default agent.
    episode = dataset.episodes[0]
    if episode.goals:
        goal = episode.goals[0]
        goal.position = np.asarray(goal.position, dtype=np.float32).reshape(-1)[:3].tolist()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with habitat.Env(config=config, dataset=dataset) as env:
        observations = env.reset()
        depth = np.asarray(observations["depth"]).squeeze()
        height, width = depth.shape
        obstacle_map = LocalObstacleMap(
            image_width=width,
            image_height=height,
            hfov_deg=90.0,
            camera_height_m=1.25,
            camera_pitch_deg=0.0,
            min_obstacle_height_m=args.min_obstacle_height,
            max_obstacle_height_m=args.max_obstacle_height,
            robot_radius_m=0.18,
            min_static_hits=1,
            memory_frames=None,
        )
        if args.no_inflation:
            # Keep raw occupied cells and disable only the robot-radius dilation.
            obstacle_map.robot_radius_m = 0.0
            obstacle_map._inflation_radius_px = 0
        writer = imageio.get_writer(output, fps=8, codec="libx264", macro_block_size=1)
        initial_state = env.sim.get_agent_state()
        origin = np.asarray(initial_state.position, dtype=np.float32)
        initial_forward = quaternion_rotate_vector(
            initial_state.rotation, np.array([0.0, 0.0, -1.0], dtype=np.float32)
        )
        initial_left = quaternion_rotate_vector(
            initial_state.rotation, np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        )
        local_navmesh = None
        calibration_records = []
        if args.navmesh_calibration:
            size = obstacle_map.grid_size_px
            local_navmesh = np.zeros((size, size), dtype=bool)
            for gy in range(size):
                forward = (
                    obstacle_map.center_px - gy
                ) / obstacle_map.pixels_per_meter
                for gx in range(size):
                    left = (
                        obstacle_map.center_px - gx
                    ) / obstacle_map.pixels_per_meter
                    world = origin + forward * initial_forward + left * initial_left
                    local_navmesh[gy, gx] = bool(
                        env.sim.pathfinder.is_navigable(world)
                    )
            navmesh_settings = getattr(env.sim.pathfinder, "nav_mesh_settings", None)
            navmesh_radius_m = float(getattr(navmesh_settings, "agent_radius", 0.0))
            navmesh_radius_px = max(
                0,
                int(math.ceil(
                    navmesh_radius_m * obstacle_map.pixels_per_meter
                )),
            )
        try:
            for step in range(args.max_steps):
                rgb = np.asarray(observations["rgb"])[..., :3]
                depth = np.asarray(observations["depth"]).squeeze()
                state = env.sim.get_agent_state()
                position = np.asarray(state.position, dtype=np.float32)
                forward_axis = quaternion_rotate_vector(
                    state.rotation, np.array([0.0, 0.0, -1.0], dtype=np.float32)
                )
                delta = position - origin
                robot_pose = (
                    float(np.dot(delta, initial_forward)),
                    float(np.dot(delta, initial_left)),
                    float(np.arctan2(
                        np.dot(forward_axis, initial_left),
                        np.dot(forward_axis, initial_forward),
                    )),
                )
                obstacle_map.update(depth, robot_pose=robot_pose)
                frame = compose_rgbd_video_frame(rgb, depth)
                map_frame = cv2.resize(
                    obstacle_map.visualize(),
                    (frame.shape[0], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                frame = np.concatenate((frame, map_frame), axis=1)
                if local_navmesh is not None:
                    observed = (
                        (obstacle_map.static_hits > 0)
                        | (obstacle_map.free_hits > 0)
                    )
                    predicted_obstacle = obstacle_map.static_map.astype(bool)
                    if navmesh_radius_px > 0:
                        kernel = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE,
                            (2 * navmesh_radius_px + 1,) * 2,
                        )
                        predicted_obstacle = cv2.dilate(
                            predicted_obstacle.astype(np.uint8), kernel
                        ).astype(bool)
                    predicted_free = observed & ~predicted_obstacle
                    false_obstacle = observed & predicted_obstacle & local_navmesh
                    unsafe_free = predicted_free & ~local_navmesh
                    reference_free = observed & local_navmesh
                    intersection = int(np.sum(predicted_free & reference_free))
                    union = int(np.sum(predicted_free | reference_free))
                    obstacle_count = int(np.sum(observed & predicted_obstacle))
                    free_count = int(np.sum(predicted_free))
                    calibration = {
                        "step": step,
                        "observed_cells": int(np.sum(observed)),
                        "navmesh_agent_radius_m": navmesh_radius_m,
                        "false_obstacle_rate": (
                            float(np.sum(false_obstacle)) / obstacle_count
                            if obstacle_count else 0.0
                        ),
                        "unsafe_free_rate": (
                            float(np.sum(unsafe_free)) / free_count
                            if free_count else 0.0
                        ),
                        "navigable_iou": (
                            float(intersection) / union if union else 1.0
                        ),
                    }
                    calibration_records.append(calibration)

                    navmesh_image = np.full(
                        (*local_navmesh.shape, 3), 230, dtype=np.uint8
                    )
                    navmesh_image[~local_navmesh] = (55, 55, 55)
                    navmesh_image[local_navmesh] = (235, 250, 235)
                    cv2.putText(
                        navmesh_image,
                        "LOCAL NAVMESH (same grid)",
                        (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.32,
                        (10, 10, 10),
                        1,
                        cv2.LINE_AA,
                    )
                    error_image = np.full(
                        (*local_navmesh.shape, 3), 205, dtype=np.uint8
                    )
                    error_image[observed & local_navmesh] = (225, 245, 225)
                    error_image[observed & ~local_navmesh] = (70, 70, 70)
                    error_image[false_obstacle] = (230, 70, 70)
                    error_image[unsafe_free] = (70, 120, 235)
                    cv2.putText(
                        error_image,
                        "red=false obstacle blue=unsafe free",
                        (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.30,
                        (10, 10, 10),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        error_image,
                        (
                            f"FO={calibration['false_obstacle_rate']:.3f} "
                            f"UF={calibration['unsafe_free_rate']:.3f} "
                            f"IoU={calibration['navigable_iou']:.3f}"
                        ),
                        (4, 27),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.30,
                        (10, 10, 10),
                        1,
                        cv2.LINE_AA,
                    )
                    robot_gx, robot_gy = obstacle_map._grid(
                        np.array([robot_pose[0]]), np.array([robot_pose[1]])
                    )
                    for image in (navmesh_image, error_image):
                        if (
                            0 <= robot_gx[0] < image.shape[1]
                            and 0 <= robot_gy[0] < image.shape[0]
                        ):
                            cv2.circle(
                                image,
                                (int(robot_gx[0]), int(robot_gy[0])),
                                4,
                                (255, 0, 0),
                                -1,
                            )
                    navmesh_image = cv2.resize(
                        navmesh_image,
                        (frame.shape[0], frame.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    error_image = cv2.resize(
                        error_image,
                        (frame.shape[0], frame.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    frame = np.concatenate(
                        (frame, navmesh_image, error_image), axis=1
                    )
                cv2.putText(
                    frame,
                    f"default PointNav agent | step {step} | height={args.min_obstacle_height:.2f}-{args.max_obstacle_height:.2f}m | inflation={'off' if args.no_inflation else 'on'}",
                    (8, frame.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                writer.append_data(frame)
                if env.episode_over:
                    break
                observations = env.step({"action": "move_forward"})
        finally:
            writer.close()
    if calibration_records:
        metrics_path = output.with_suffix(".navmesh_metrics.json")
        metrics_path.write_text(json.dumps({
            "dataset_index": args.dataset_index,
            "video": str(output),
            "final": calibration_records[-1],
            "steps": calibration_records,
        }, indent=2) + "\n")
        print(metrics_path)
    print(output)


if __name__ == "__main__":
    main()
