import io
import json
import math
import os
from typing import Optional

import cv2
import imageio
import numpy as np
import requests


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


class FluxHTTPAgent:
    """Adapter from EVT-Bench tracking observations to FLUX pointgoal HTTP API."""

    def __init__(self, result_path: str):
        self.result_path = result_path
        os.makedirs(self.result_path, exist_ok=True)

        self.host = os.environ.get("FLUX_HOST", "127.0.0.1")
        self.port = int(os.environ.get("FLUX_PORT", "8892"))
        self.base_url = f"http://{self.host}:{self.port}"
        self.timeout = float(os.environ.get("FLUX_TIMEOUT", "30"))
        self.stop_threshold = float(os.environ.get("FLUX_STOP_THRESHOLD", "-3.0"))
        self.controller_speed = float(os.environ.get("FLUX_CONTROLLER_SPEED", "15.0"))
        self.controller_w_max = float(os.environ.get("FLUX_CONTROLLER_W_MAX", "3.0"))
        self.controller_yaw_kp = float(os.environ.get("FLUX_CONTROLLER_YAW_KP", "1.5"))
        self.controller_lookahead = float(os.environ.get("FLUX_CONTROLLER_LOOKAHEAD", "0.5"))
        self.controller_turn_in_place = math.radians(
            float(os.environ.get("FLUX_CONTROLLER_TURN_IN_PLACE_DEG", "45"))
        )
        self.env_max_vx = float(os.environ.get("FLUX_ENV_MAX_VX", "15.0"))
        self.env_max_vy = float(os.environ.get("FLUX_ENV_MAX_VY", "10.0"))
        self.env_max_wz = float(os.environ.get("FLUX_ENV_MAX_WZ", "6.28"))
        self.goal_distance = float(os.environ.get("FLUX_FOLLOW_DISTANCE", "1.2"))
        self.sensor_depth_max = float(os.environ.get("FLUX_SENSOR_DEPTH_MAX", "10.0"))
        self.pointgoal_source = os.environ.get("FLUX_POINTGOAL_SOURCE", "visual").strip().lower()
        if self.pointgoal_source not in {"visual", "oracle"}:
            raise ValueError("FLUX_POINTGOAL_SOURCE must be 'visual' or 'oracle'")
        self.save_video = _env_flag("TRACKVLA_SAVE_VIDEO", _env_flag("SAVE_VIDEO", True))
        self.verbose_steps = _env_flag("TRACKVLA_VERBOSE_STEPS", True)

        self.rgb_list = []
        self._reset_server()

    def reset(self, episode=None):
        if len(self.rgb_list) != 0:
            if self.save_video and episode is not None:
                scene_key = os.path.splitext(os.path.basename(episode.scene_id))[0].split(".")[0]
                save_dir = os.path.join(self.result_path, scene_key)
                os.makedirs(save_dir, exist_ok=True)
                output_video_path = os.path.join(save_dir, f"{episode.episode_id}.mp4")
                imageio.mimsave(output_video_path, self.rgb_list)
                print(f"Successfully save the episode video with episode id {episode.episode_id}")
            self.rgb_list = []
        self._reset_server()

    def act(self, observations, detector, episode_id, instruction=None, robot_agent=None, humanoid_agent=None):
        if robot_agent is None or humanoid_agent is None:
            raise RuntimeError("FluxHTTPAgent requires robot_agent and humanoid_agent to build pointgoal.")

        rgb = observations["agent_1_articulated_agent_jaw_rgb"][:, :, :3]
        depth = self._get_depth(observations)
        if self.pointgoal_source == "oracle":
            raw_goal = self._oracle_human_pointgoal(robot_agent, humanoid_agent)
        else:
            raw_goal = self._visual_human_pointgoal(detector, depth)
        goal = self._apply_follow_distance(raw_goal)
        trajectory = self._pointgoal_step(goal, rgb, depth)

        action = self._trajectory_to_velocity(trajectory)
        if self.verbose_steps:
            print(
                f"[flux:{self.pointgoal_source}] raw_goal={raw_goal.tolist()} "
                f"goal={goal.tolist()} controller={self._last_control_debug} action={action}"
            )

        if self.save_video:
            self.rgb_list.append(rgb)
        return action

    def _reset_server(self):
        payload = {
            "intrinsic": self._intrinsic().tolist(),
            "stop_threshold": self.stop_threshold,
            "batch_size": 1,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/navigator_reset",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to reset FLUX server at {self.base_url}. "
                "Start it first, for example: "
                "python /robot/robot-research-exp-0/user/gzy/FLUX/baselines/flux/server.py "
                "--port 8892 --checkpoint /path/to/flux.ckpt"
            ) from exc

    def _get_depth(self, observations):
        key = "agent_1_articulated_agent_jaw_depth"
        if key not in observations:
            raise RuntimeError(
                f"Missing {key}. Use the FLUX sensor setup or add jaw_depth_sensor to the eval config."
            )
        depth = np.asarray(observations[key])
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth = depth.astype(np.float32)
        if np.nanmax(depth) <= 1.01:
            depth = depth * self.sensor_depth_max
        depth[~np.isfinite(depth)] = 0.0
        return depth

    def _oracle_human_pointgoal(self, robot_agent, humanoid_agent):
        delta_world = humanoid_agent.base_pos - robot_agent.base_pos
        delta_local = robot_agent.sim_obj.transformation.inverted().transform_vector(delta_world)
        # Habitat BaseVelAction maps [longitudinal, lateral] to local [x, -z].
        return np.array([[float(delta_local.x), float(-delta_local.z)]], dtype=np.float32)

    def _visual_human_pointgoal(self, detector, depth):
        detection = detector.get("agent_1_main_humanoid_detector_sensor", detector)
        bbox = np.asarray(detection.get("box", np.zeros(4)), dtype=np.float32)
        if bbox.shape != (4,) or not np.any(bbox):
            return np.zeros((1, 2), dtype=np.float32)

        height, width = depth.shape
        x0, y0, x1, y1 = bbox
        x0 = int(np.clip(np.floor(x0), 0, width - 1))
        x1 = int(np.clip(np.ceil(x1), x0 + 1, width))
        y0 = int(np.clip(np.floor(y0), 0, height - 1))
        y1 = int(np.clip(np.ceil(y1), y0 + 1, height))
        roi_depth = depth[y0:y1, x0:x1]
        valid_depth = roi_depth[np.isfinite(roi_depth) & (roi_depth > 0.1)]
        if valid_depth.size == 0:
            return np.zeros((1, 2), dtype=np.float32)

        forward = float(np.median(valid_depth))
        center_x = 0.5 * (x0 + x1 - 1)
        intrinsic = self._intrinsic()
        lateral = (center_x - intrinsic[0, 2]) * forward / intrinsic[0, 0]

        return np.array([[forward, lateral]], dtype=np.float32)

    def _apply_follow_distance(self, pointgoal):
        forward, lateral = pointgoal[0]
        distance = float(np.hypot(forward, lateral))
        if distance <= self.goal_distance:
            return np.zeros((1, 2), dtype=np.float32)
        scale = (distance - self.goal_distance) / distance
        return np.array([[forward * scale, lateral * scale]], dtype=np.float32)

    def _pointgoal_step(self, point_goal, rgb, depth):
        _, rgb_image = cv2.imencode(".jpg", rgb)
        depth_u16 = np.clip(depth * 10000.0, 0, 65535.0).astype(np.uint16)
        _, depth_image = cv2.imencode(".png", depth_u16)
        files = {
            "image": ("image.jpg", io.BytesIO(rgb_image.tobytes()), "image/jpeg"),
            "depth": ("depth.png", io.BytesIO(depth_image.tobytes()), "image/png"),
        }
        data = {
            "goal_data": json.dumps(
                {
                    "goal_x": point_goal[:, 0].tolist(),
                    "goal_y": point_goal[:, 1].tolist(),
                }
            )
        }
        resp = requests.post(
            f"{self.base_url}/pointgoal_step",
            files=files,
            data=data,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return np.asarray(payload["trajectory"], dtype=np.float32)[0]

    def _trajectory_to_velocity(self, trajectory):
        self._last_control_debug = {"status": "invalid"}
        if trajectory.ndim != 2 or trajectory.shape[0] == 0:
            return [0.0, 0.0, 0.0]
        path_xy = np.asarray(trajectory[:, :2], dtype=np.float32)
        if not np.isfinite(path_xy).all():
            return [0.0, 0.0, 0.0]

        distances = np.linalg.norm(path_xy, axis=1)
        valid = np.flatnonzero(distances >= self.controller_lookahead)
        if valid.size:
            target_idx = int(valid[0])
        elif distances[-1] > 1e-3:
            target_idx = len(path_xy) - 1
        else:
            self._last_control_debug = {"status": "zero_path"}
            return [0.0, 0.0, 0.0]

        target = path_xy[target_idx]
        heading_error = float(np.arctan2(target[1], target[0]))
        # Match FLUX's official differential-drive MPC semantics: forward-only
        # linear velocity and angular velocity derived from the XY path tangent.
        if abs(heading_error) >= self.controller_turn_in_place:
            vx_mps = 0.0
        else:
            vx_mps = self.controller_speed * max(0.0, float(np.cos(heading_error))) ** 2
        wz_rps = float(np.clip(
            self.controller_yaw_kp * heading_error,
            -self.controller_w_max,
            self.controller_w_max,
        ))
        self._last_control_debug = {
            "status": "tracking",
            "lookahead_index": target_idx,
            "lookahead_xy": target.tolist(),
            "heading_error": heading_error,
            "path_end_xy": path_xy[-1].tolist(),
        }
        # BaseVelAction expects normalized commands and applies its configured
        # physical speed limits internally.
        return [
            float(np.clip(vx_mps / self.env_max_vx, -1.0, 1.0)),
            0.0,
            float(np.clip(wz_rps / self.env_max_wz, -1.0, 1.0)),
        ]

    def _intrinsic(self):
        # Default for the 384x384, 90-degree jaw camera used by OmTrackVLA eval.
        size = float(os.environ.get("FLUX_CAMERA_SIZE", "384"))
        hfov = math.radians(float(os.environ.get("FLUX_CAMERA_HFOV", "90")))
        focal = size / (2.0 * math.tan(hfov / 2.0))
        center = (size - 1.0) / 2.0
        return np.array(
            [
                [focal, 0.0, center],
                [0.0, focal, center],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
