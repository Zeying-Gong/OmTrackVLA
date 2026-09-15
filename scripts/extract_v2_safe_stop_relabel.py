#!/usr/bin/env python3
"""Extract a zero-waypoint label when the target is absent from the full RGB history."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

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
from scripts.probe_next007_expert_relabel import (
    _atomic_json,
    _reference_after_step,
    _write_phase3_sample,
    causal_history_steps,
    replay_actions,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-result", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase3-sample-dir", type=Path, required=True)
    parser.add_argument("--history-size", type=int, default=8)
    parser.add_argument("--history-stride-steps", type=int, default=3)
    parser.add_argument("--model-image-height", type=int, default=280)
    parser.add_argument("--model-image-width", type=int, default=504)
    parser.add_argument("--scene-dataset", default=DEFAULT_SCENE_DATASET)
    parser.add_argument("--replay-tolerance", type=float, default=1.0e-4)
    return parser.parse_args()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = arguments()
    result_path = args.rollout_result.expanduser().resolve(strict=True)
    result = load(result_path)
    if (
        not isinstance(result, dict)
        or result.get("split") != "train"
        or result.get("uwb_mode") != "missing"
        or result.get("input_audit", {}).get("passed") is not True
    ):
        raise ValueError("safe-stop relabel requires an admitted visual-only train rollout")
    actions = replay_actions(result)
    if not 21 <= args.checkpoint_step <= len(actions):
        raise ValueError("safe-stop relabel needs a full causal history")
    requested_steps = causal_history_steps(
        args.checkpoint_step, args.history_size, args.history_stride_steps
    )
    if any(second - first != args.history_stride_steps for first, second in zip(requested_steps, requested_steps[1:])):
        raise ValueError("safe-stop history is left padded or not fixed-stride")

    import habitat
    from habitat.config import read_write
    from habitat.datasets import make_dataset
    import evt_bench  # noqa: F401

    task = str(result["task"])
    config_path = f"habitat-lab/habitat/config/benchmark/nav/track/track_train_{task}.yaml"
    config = configure(habitat.get_config(config_path), args.scene_dataset)
    with read_write(config):
        if PANOPTIC_KEY not in config.habitat.gym.obs_keys:
            config.habitat.gym.obs_keys.append(PANOPTIC_KEY)
    dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    dataset_index = int(result["dataset_index"])
    episode = dataset.episodes[dataset_index]
    if str(episode.episode_id) != str(result["episode_id"]) or str(episode.scene_id) != str(result["scene_id"]):
        raise ValueError("safe-stop rollout episode identity changed")
    dataset.episodes = [episode]

    history_rgb: dict[int, np.ndarray] = {}
    history_visible: dict[int, bool] = {}
    with habitat.TrackEnv(config=config, dataset=dataset) as env:
        env.reset()
        _assign_unique_humanoid_semantic_ids(env)

        def capture(step: int) -> tuple[np.ndarray, bool]:
            observations = env.sim.get_sensor_observations()
            rgb = np.asarray(observations[RGB_KEY])[..., :3].copy()
            target_mask = (
                np.asarray(observations[PANOPTIC_KEY]).squeeze()
                == int(episode.info["main_human_semantic_id"])
            )
            visible = target_mask_to_bbox(target_mask) is not None
            history_rgb[step] = rgb
            history_visible[step] = visible
            return rgb, visible

        initial_rgb, initial_visible = capture(0)
        if not initial_visible:
            raise RuntimeError("admitted visual initialization target is not visible")
        for step, action in enumerate(actions[: args.checkpoint_step], 1):
            if env.episode_over:
                raise RuntimeError(f"episode ended during safe-stop replay at step {step}")
            env.step(
                {
                    "action": ACTION_NAMES,
                    "action_args": {"agent_1_base_vel": action},
                }
            )
            capture(step)
        requested_visibility = [history_visible[step] for step in requested_steps]
        if any(requested_visibility):
            raise RuntimeError(
                "safe-stop relabel refused: target is visible in the causal RGB history"
            )
        reference = _reference_after_step(result, args.checkpoint_step)
        robot = env.sim.agents_mgr[1].articulated_agent
        target = env.sim.agents_mgr[0].articulated_agent
        replay_distance = float(
            np.linalg.norm(
                np.asarray(robot.base_pos, dtype=np.float64)
                - np.asarray(target.base_pos, dtype=np.float64)
            )
        )
        camera_error = float(
            np.max(
                np.abs(
                    _camera_transform(env.sim)
                    - np.asarray(
                        reference["render_audit"]["camera_transform_after_forced_render"],
                        dtype=np.float64,
                    )
                )
            )
        )
        distance_error = abs(replay_distance - float(reference["gt_distance_m"]))
        if camera_error > args.replay_tolerance or distance_error > args.replay_tolerance:
            raise RuntimeError("safe-stop replay did not reproduce the saved model state")
        local = robot.sim_obj.transformation.inverted().transform_vector(
            np.asarray(target.base_pos) - np.asarray(robot.base_pos)
        )
        forward = float(local.x)
        left = float(-local.z)
        angle = math.atan2(left, forward)
        target_state = {
            "bbox_xyxy_norm": None,
            "visible": False,
            "angle_sincos": [math.sin(angle), math.cos(angle)],
            "distance_m": math.hypot(forward, left),
            "polar_valid": False,
            "gt_used_only_on_label_side": True,
        }
        intrinsics, camera_from_base = habitat_camera_calibration(
            initial_rgb.shape,
            output_height=args.model_image_height,
            output_width=args.model_image_width,
            hfov_deg=90.0,
        )
        frames = [(step, history_rgb[step]) for step in requested_steps]

    manifest_path, manifest_sha = _write_phase3_sample(
        directory=args.phase3_sample_dir,
        source_rollout=result_path,
        report_output=args.output.expanduser().resolve(strict=False),
        result=result,
        initial_rgb=initial_rgb,
        history_frames=frames,
        initial_bbox_xyxy=result["initialization"]["bbox_xyxy"],
        camera_intrinsics=intrinsics,
        camera_from_base=camera_from_base,
        waypoints=[[0.0, 0.0] for _ in range(8)],
        checkpoint_step=args.checkpoint_step,
        history_stride_steps=args.history_stride_steps,
        target_state=target_state,
        stop_required=True,
    )
    payload = {
        "schema_version": 1,
        "stage": "architecture_v2_unobservable_safe_stop_relabel_v1",
        "status": "passed",
        "source_rollout": str(result_path),
        "task": task,
        "split": "train",
        "dataset_index": dataset_index,
        "episode_id": str(episode.episode_id),
        "checkpoint_step": args.checkpoint_step,
        "history_environment_steps": requested_steps,
        "history_gt_visible": requested_visibility,
        "waypoint_supervision": "safe_stop",
        "expert_target_direction_used": False,
        "replay_camera_max_abs_error": camera_error,
        "replay_target_distance_abs_error_m": distance_error,
        "phase3_sample": str(manifest_path),
        "phase3_sample_sha256": manifest_sha,
        "test_locked_used": False,
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
