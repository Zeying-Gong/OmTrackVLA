"""TpT-bench person_follow dataset (per-video-frame samples from frames.parquet)."""
from __future__ import annotations

import json
import os

import numpy as np
import pyarrow.parquet as pq

from unified import TASK_PERSON_FOLLOW, make_sample_dict


def _yaw_from_quat_xyzw(q):
    qx, qy, qz, qw = q
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _body_disp(pos_a, qa, pos_b):
    """Global displacement pos_b-pos_a rotated into the local frame of pose a (yaw only)."""
    d = np.asarray(pos_b, dtype=np.float64) - np.asarray(pos_a, dtype=np.float64)
    yaw = _yaw_from_quat_xyzw(qa)
    c, s = np.cos(yaw), np.sin(yaw)
    return [float(c * d[0] + s * d[1]), float(-s * d[0] + c * d[1])]


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


class TpTDataset:
    def __init__(self, tpt_root, step_stride=4, history=8, future_offset=12, exclude=("0002",)):
        self.tpt_root = tpt_root
        self.step_stride = max(1, step_stride)
        self.history = history
        self.future_offset = future_offset
        self.exclude = set(exclude)
        self.seqs = []
        for s in sorted(os.listdir(tpt_root)):
            if not s.isdigit() or s in self.exclude:
                continue
            parquet = os.path.join(tpt_root, s, "frames.parquet")
            if not os.path.exists(parquet):
                continue
            desc = ""
            desc_path = os.path.join(tpt_root, s, "desc.txt")
            if os.path.exists(desc_path):
                desc = open(desc_path).read().strip()
            meta = {}
            meta_path = os.path.join(tpt_root, s, "meta.json")
            if os.path.exists(meta_path):
                try:
                    meta = json.load(open(meta_path))
                except Exception:
                    pass
            self.seqs.append({"seq": s, "desc": desc, "meta": meta, "parquet": parquet})
        self._tab = {}

    def __len__(self):
        return len(self.seqs)

    def _table(self, seq):
        if seq not in self._tab:
            self._tab[seq] = pq.read_table(self.seqs[self._seq_idx(seq)]["parquet"])
        return self._tab[seq]

    def _seq_idx(self, seq):
        for i, s in enumerate(self.seqs):
            if s["seq"] == seq:
                return i
        raise KeyError(seq)

    def video_frames(self, seq):
        return os.path.join(self.tpt_root, seq, "rgb_frames")

    def get(self, seq, row_idx):
        t = self._table(seq)
        cols = t.to_pydict()
        if row_idx < 0 or row_idx >= len(cols["video_idx"]):
            return None
        vid = int(cols["video_idx"][row_idx])
        rgb_dir = self.video_frames(seq)
        current = os.path.join(rgb_dir, f"frame_{vid:06d}.jpg")
        if not os.path.isfile(current):
            return None
        history = []
        for i in range(1, self.history + 1):
            p = os.path.join(rgb_dir, f"frame_{vid - i * self.step_stride:06d}.jpg")
            history.append(p if os.path.isfile(p) else None)
        history = [p for p in history if p]
        future = os.path.join(rgb_dir, f"frame_{vid + self.future_offset:06d}.jpg")
        future = future if os.path.isfile(future) else None
        bbox_raw = list(cols["bbox_qv"][row_idx])
        # TpT stores bbox as [x, y, w, h]; convert to [x0, y0, x1, y1]
        bbox = [bbox_raw[0], bbox_raw[1], bbox_raw[0] + bbox_raw[2], bbox_raw[1] + bbox_raw[3]]
        is_exist = bool(cols["is_exist"][row_idx])
        target_img = _crop_target(current, bbox) if is_exist else None
        pos = list(cols["odom_pos"][row_idx])
        quat = list(cols["odom_quat_xyzw"][row_idx])
        motion = None
        if all(x == x for x in pos):
            # displacement vs the sample before the window
            for j in range(max(0, row_idx - self.history * self.step_stride), row_idx):
                pj = list(cols["odom_pos"][j])
                qj = list(cols["odom_quat_xyzw"][j])
                if all(x == x for x in pj) and tuple(pj) != tuple(pos):
                    motion = _body_disp(pj, qj, pos)
                    break
        extra = {
            "is_behind_glass": abs(int(cols["is_behind_glass"][row_idx])) == 1,
            "interpolated": bool(cols["interpolated"][row_idx]),
            "history_motion": motion,
            "vid_pts_ms": float(cols["vid_pts_ms"][row_idx]),
        }
        seq_meta = self.seqs[self._seq_idx(seq)]["meta"]
        instr = self.seqs[self._seq_idx(seq)]["desc"]
        return make_sample_dict(
            task=TASK_PERSON_FOLLOW,
            ep_id=seq,
            instruction=instr,
            current=current,
            window=history,
            target_img=target_img,
            bbox=bbox,
            traj=None,
            collision=False,
            target_dist=None,
            valid=is_exist,
            future_rgb=future,
            extra=extra,
        )