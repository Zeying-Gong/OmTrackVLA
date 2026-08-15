"""Sage3D person_follow dataset (extracted tar runs -> indexed episodes -> steps)."""
from __future__ import annotations

import glob
import json
import os

from unified import TASK_PERSON_FOLLOW, make_sample_dict


def _frame_index(f):
    stem = os.path.splitext(os.path.basename(f))[0]
    try:
        return int(stem)
    except ValueError:
        return -1


def _list_frame_paths(ep_dir, sub):
    d = os.path.join(ep_dir, sub)
    if not os.path.isdir(d):
        return []
    files = [os.path.join(d, p) for p in os.listdir(d)]
    files = [f for f in files if _frame_index(f) >= 0]
    files.sort(key=_frame_index)
    return files


def _crop_target(rgb_path, bbox, max_side=224, pad=8):
    import cv2

    img = cv2.imread(rgb_path)
    if img is None:
        return None
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in bbox]
    x0, y0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
    x1, y1 = min(W, int(x1) + pad), min(H, int(y1) + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = img[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        crop = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))))
    path = os.path.join(os.path.dirname(rgb_path), "_target_crops", os.path.basename(rgb_path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, crop)
    return path


class Sage3DShot:
    """Single person-follow step of an accepted episode."""


class Sage3DDataset:
    def __init__(self, extracted_root, step_stride=1, only_eps_with_derived=True):
        self.extracted_root = extracted_root
        self.step_stride = max(1, step_stride)
        self.index = []  # list of dict: run, mode, ep, cam, instruction, success, ep_dir, derived
        self._derived_cache = {}

        runs = sorted(
            d for d in os.listdir(extracted_root)
            if os.path.isdir(os.path.join(extracted_root, d))
        )
        for run in runs:
            idx_path = os.path.join(extracted_root, run, "index.json")
            if not os.path.exists(idx_path):
                continue
            idx = json.load(open(idx_path))
            for e in idx.get("eps", []):
                mode, ep, cam = e.get("mode"), e.get("ep"), e.get("cam")
                ep_dir = os.path.join(extracted_root, run, str(mode), str(ep), str(cam))
                derived_path = os.path.join(ep_dir, "derived.json")
                if not os.path.exists(derived_path):
                    if only_eps_with_derived:
                        continue
                    derived_path = None
                self.index.append(
                    {
                        "run": run,
                        "mode": mode,
                        "ep": ep,
                        "cam": cam,
                        "instruction": e.get("instruction", ""),
                        "success": e.get("success", 0.0),
                        "following_rate": e.get("following_rate", 0.0),
                        "ep_dir": ep_dir,
                        "derived_path": derived_path,
                    }
                )

        # Next non-consecutive maximal-descent gives ~future frame.
        self._future_offset = self.step_stride * 3

    def __len__(self):
        return len(self.index)

    def episodes(self):
        return [dict(i) for i in self.index]

    def _derived(self, idx):
        p = idx["derived_path"]
        if p not in self._derived_cache:
            self._derived_cache[p] = json.load(open(p))
        return self._derived_cache[p]

    def _ep_frames(self, idx):
        return _list_frame_paths(idx["ep_dir"], "rgb")

    def _resolve(self, i):
        return self.index[i] if isinstance(i, int) else i

    def step_count(self, i):
        idx = self._resolve(i)
        return len(self._derived(idx)["steps"])

    def get(self, i, k):
        """Build sample dict for episode index `i` at derived step k."""
        idx = self._resolve(i)
        ep_dir = idx["ep_dir"]
        derived = self._derived(idx)
        steps = derived["steps"]
        nh = derived.get("waypoint_horizon", 8)
        if k < 0 or k >= len(steps):
            return None
        rgb = _list_frame_paths(ep_dir, "rgb")
        if k >= len(rgb):
            return None
        st = steps[k]
        current = rgb[k]
        window = [rgb[i] for i in range(max(0, k - self.step_stride * 8), k, self.step_stride)]
        bbox = st.get("bbox")
        valid = bool(st.get("visible", False))
        target_img = None
        if valid and bbox:
            try:
                target_img = _crop_target(current, bbox)
            except Exception:
                target_img = None
        wpts = st.get("waypoints_ego") or []
        if wpts and len(wpts) < nh:
            wpts = wpts + [[wpts[-1][0], wpts[-1][1]]] * (nh - len(wpts))
        future_rgb = rgb[min(len(rgb) - 1, k + self._future_offset)] if rgb else None
        extra = {
            "success": idx["success"],
            "following_rate": idx["following_rate"],
            "dis_to_human": st.get("dis_to_human"),
            "target_local": st.get("target_local"),
            "facing": st.get("facing"),
        }
        return make_sample_dict(
            task=TASK_PERSON_FOLLOW,
            ep_id=f'{idx["run"]}/{idx["mode"]}/{idx["ep"]}/{idx["cam"]}',
            instruction=idx["instruction"],
            current=current,
            window=window,
            target_img=target_img,
            bbox=bbox,
            traj=wpts[:nh],
            collision=bool(st.get("collision", False)),
            target_dist=st.get("target_dist"),
            valid=valid,
            future_rgb=future_rgb,
            extra=extra,
        )