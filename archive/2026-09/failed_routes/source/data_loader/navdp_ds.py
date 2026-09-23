"""InternNav NavDP pointnav/imagegoal dataset (unified lazy-path samples).

Replicates the geometry core of internnav/dataset/navdp_lerobot_dataset.py
(relative_pose, xyz_to_xyt, process_actions trajectory) but emits the unified
OmTrackVLA sample dict and reads parquet via pyarrow (no pandas needed).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pyarrow.parquet as pq

from unified import TASK_IMAGEGOAL, TASK_POINTNAV, make_sample_dict


def _relative_pose_point(ext_start, base_extrinsic, world_point):
    """World -> robot-local point given extrinsics of the start frame.

    Matches InternNav navdp_lerobot_dataset.py: the robot rotation is composed
    with the inverse of the base/camera extrinsic rotation so the resulting
    local frame is the robot body frame (base_extrinsic is generally NOT an
    identity rotation on D435i).
    """
    R_base = np.array(ext_start[:3, :3], dtype=np.float64)
    T_base = np.array(ext_start[:3, 3], dtype=np.float64)
    R_base = R_base @ np.linalg.inv(np.array(base_extrinsic[:3, :3], dtype=np.float64))
    homo_RT = np.eye(4)
    homo_RT[:3, :3] = R_base
    homo_RT[:3, 3] = T_base
    T_frame = np.dot(np.linalg.inv(homo_RT), np.array([*world_point, 1.0])).T[:3]
    # swap to (y, -x, z) convention used by NavDP
    return np.array([T_frame[1], -T_frame[0], T_frame[2]], dtype=np.float64)


def _xyz_to_xyt(xyz, init_vector):
    """Waypoints with heading from consecutive displacement.

    Emits M points (every element of xyz, INCLUDING the final position), with
    heading held from the last displacement. Omitting the last point dropped the
    goal whenever the trajectory is short (target=mem+1 -> only origin).
    """
    n = xyz.shape[0]
    if n < 2:
        return np.empty((0, 3), dtype=np.float64)
    yaw = np.zeros(n, dtype=np.float64)
    for i in range(n - 1):
        cv = xyz[i + 1] - xyz[i]
        dot = np.dot(init_vector[:2], cv[:2])
        cross = np.cross(init_vector[:2], cv[:2])
        yaw[i] = float(np.arctan2(cross, dot))
    yaw[-1] = yaw[-2]
    return np.column_stack([xyz[:, 0], xyz[:, 1], yaw])


class NavDPDataset:
    def __init__(
        self,
        root_dirs,
        memory_size=8,
        predict_size=24,
        pred_digit=4,
        seed=0,
        scene_scale=1.0,
        cap_episodes=None,
        condition="pointnav",
        require_depth=True,
    ):
        self.root_dirs = root_dirs
        self.memory_size = memory_size
        self.predict_size = predict_size
        self.pred_digit = pred_digit
        self.condition = condition
        self.rng = np.random.RandomState(seed)
        self.episodes = []  # dict: parquet, rgb_paths, depth_paths

        group_dirs = sorted(
            d for d in os.listdir(root_dirs) if os.path.isdir(os.path.join(root_dirs, d))
        )
        n_eps = 0
        for group in group_dirs:
            group_root = os.path.join(root_dirs, group)
            scene_dirs = sorted(d for d in os.listdir(group_root) if os.path.isdir(os.path.join(group_root, d)))
            scene_dirs = scene_dirs[: int(len(scene_dirs) * scene_scale)]
            for scene in scene_dirs:
                scene_root = os.path.join(group_root, scene)
                stats_path = os.path.join(scene_root, "meta", "episodes_stats.jsonl")
                if not os.path.exists(stats_path):
                    continue
                chunks = [c for c in os.listdir(os.path.join(scene_root, "data")) if c.startswith("chunk")]
                for chunk in chunks:
                    data_dir = os.path.join(scene_root, "data", chunk)
                    rgb_dir = os.path.join(scene_root, "videos", chunk, "observation.images.rgb")
                    depth_dir = os.path.join(scene_root, "videos", chunk, "observation.images.depth")
                    if not os.path.isdir(rgb_dir):
                        continue
                    if require_depth and not os.path.isdir(depth_dir):
                        continue
                    rgb_paths = [os.path.join(rgb_dir, p) for p in sorted(os.listdir(rgb_dir))]
                    depth_paths = (
                        [os.path.join(depth_dir, p) for p in sorted(os.listdir(depth_dir))]
                        if os.path.isdir(depth_dir)
                        else [None] * len(rgb_paths)
                    )
                    try:
                        stats = [json.loads(l) for l in open(stats_path) if l.strip()]
                    except Exception:
                        continue
                    parquet_paths = sorted(os.path.join(data_dir, p) for p in os.listdir(data_dir) if p.endswith(".parquet"))
                    for ep_i, ep in enumerate(stats):
                        if cap_episodes and n_eps >= cap_episodes:
                            break
                        parquet_path = parquet_paths[ep_i] if ep_i < len(parquet_paths) else None
                        if not parquet_path or not os.path.exists(parquet_path):
                            continue
                        i0 = ep["image_index"]["min"]
                        i1 = ep["image_index"]["max"]
                        if i1 < i0 or i1 >= len(rgb_paths):
                            continue
                        self.episodes.append(
                            {
                                "parquet": parquet_path,
                                "rgb": rgb_paths[i0 : i1 + 1],
                                "depth": depth_paths[i0 : i1 + 1],
                                "ep_id": f"{group}/{scene}/{chunk}/{ep_i:06d}",
                            }
                        )
                        n_eps += 1
                    if cap_episodes and n_eps >= cap_episodes:
                        break
            if cap_episodes and n_eps >= cap_episodes:
                break

    def __len__(self):
        return len(self.episodes)

    def _parquet(self, ep):
        t = pq.read_table(ep["parquet"])
        d = t.to_pydict()
        extrinsics = np.array([np.array(f, dtype=np.float64).reshape(4, 4) for f in d["action"]])
        base_ext = np.vstack(d["observation.camera_extrinsic"][0]).reshape(4, 4)
        intrinsic = np.vstack(d["observation.camera_intrinsic"][0]).reshape(3, 3)
        return d, extrinsics, base_ext, intrinsic

    def _sample_indices(self, traj_len):
        start = int(self.rng.randint(0, max(1, traj_len // 2)))
        target = int(self.rng.randint(start + 1, max(start + 2, traj_len)))
        mem = int(self.rng.randint(start, max(start + 1, target)))
        return start, target, mem

    def _img_shape(self, ep):
        """(H, W) of the rgb frames; cached per episode."""
        if "img_shape" not in ep:
            import cv2

            img = cv2.imread(ep["rgb"][0])
            ep["img_shape"] = img.shape[:2] if img is not None else (480, 640)
        return ep["img_shape"]

    def get(self, index, task_override=None):
        ep = self.episodes[index]
        d, extrinsics, base_ext, intrinsic = self._parquet(ep)
        traj_len = extrinsics.shape[0]
        start, target, mem = self._sample_indices(traj_len)

        task = task_override or self.condition

        n = self.memory_size
        # history strictly BEFORE the current (mem) frame; `current` is appended
        # by the policy, so including mem here would duplicate the current image
        # at two different 3D-mRoPE time coordinates.
        mem_idx = np.arange(mem - (n - 1) * self.pred_digit, mem, self.pred_digit)
        mem_idx = mem_idx[mem_idx >= 0]
        window = [ep["rgb"][int(i)] for i in mem_idx]
        current = ep["rgb"][mem]
        depth = ep["depth"][mem] if mem < len(ep["depth"]) else None
        future_depth = ep["depth"][target] if target < len(ep["depth"]) else None

        # future trajectory in the local frame of the CURRENT (mem) frame.
        # InternNav's process_actions() starts from memory_start_choice = the
        # frame we actually observe; using extrinsics[start] mis-anchors the
        # trajectory (a `start` frame we never see as `current`).
        local = []
        for t in range(mem, target + 1):
            local.append(_relative_pose_point(extrinsics[mem], base_ext, extrinsics[t][:3, 3]))
        local = np.array(local, dtype=np.float64)
        init_vec = local[1] - local[0] if local.shape[0] > 1 else np.array([1.0, 0.0, 0.0])
        xyt = _xyz_to_xyt(local, init_vec)
        if xyt.shape[0] == 0:
            return None
        action_indexes = np.clip(np.arange(self.predict_size + 1) * self.pred_digit, 0, xyt.shape[0] - 1)
        traj = xyt[action_indexes]  # absolute ego waypoints (predict_size+1,3)
        diffs = (traj[1:] - traj[:-1]) * 4.0
        pointgoal = list(xyt[-1][:2]) + [float(xyt[-1][2])]
        goal_dist = float(np.hypot(pointgoal[0], pointgoal[1]))

        # goal visibility in the current view (same projection as NavDP pixel_goal)
        camera_coord = np.matmul(base_ext[:3, :3], np.array([-local[-1][1], local[-1][0], base_ext[2, 3] * 0.8]))
        u = intrinsic[0, 2] + (camera_coord[0] / camera_coord[2]) * intrinsic[0, 0]
        v = intrinsic[1, 2] + (-camera_coord[1] / camera_coord[2]) * intrinsic[1, 1]
        ih, iw = self._img_shape(ep)
        visible = bool(0 <= u < iw and 0 <= v < ih) and camera_coord[2] > 0.05

        # Modal isolation: a task sees exactly ONE goal modality.
        #   pointnav  -> PointGoal only (goal image withheld)
        #   imagegoal -> goal image only (PointGoal withheld)
        # The goal distance is still emitted as a supervision label (stop /
        # target-state), which is legitimate: it is a label, not an input.
        if task == TASK_POINTNAV:
            target_img = None
            pg = pointgoal
            pg_valid = visible
        elif task == TASK_IMAGEGOAL:
            target_img = ep["rgb"][target]
            pg = None
            pg_valid = False
        else:
            target_img = None
            pg = pointgoal
            pg_valid = visible
        future_rgb = ep["rgb"][min(len(ep["rgb"]) - 1, target)]

        return make_sample_dict(
            task=task,
            ep_id=ep["ep_id"],
            instruction="",
            current=current,
            window=window,
            target_img=target_img,
            bbox=None,
            traj=[list(p) for p in traj],
            actions=[list(p) for p in diffs],
            collision=False,
            target_dist=goal_dist,
            valid=visible,
            pointgoal=pg,
            pointgoal_valid=pg_valid,
            future_rgb=future_rgb,
            depth=depth,
            extra={"start": start, "target": target, "mem": mem, "goal_dist": goal_dist,
                   "future_depth": future_depth},
        )

    def sample(self, task_override=None):
        if not self.episodes:
            return None
        return self.get(int(self.rng.randint(0, len(self.episodes))), task_override)