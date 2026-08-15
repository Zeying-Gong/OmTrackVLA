"""Unified sample schema + loader utilities for OmTrackVLA data preparation.

Sample dict keys (canonical). All paths are absolute strings; loaded arrays are
numpy (H,W,C) float32 in [0,1] (rgb) or meters (depth).
"""
from __future__ import annotations

import math
import os

import cv2
import numpy as np

TASK_PERSON_FOLLOW = "person_follow"
TASK_POINTNAV = "pointnav"
TASK_IMAGEGOAL = "imagegoal"
TASKS = (TASK_PERSON_FOLLOW, TASK_POINTNAV, TASK_IMAGEGOAL)


def imread_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_keep_aspect(img, size, normalize=True):
    """Letterbox-resize to a square of `size`, matching NavDP process_image.

    Supports both 3-channel RGB and 2D single-channel (depth) arrays. RGB is
    normalized to [0,1]; depth is returned in native units (never /255).
    """
    H, W = img.shape[:2]
    prop = size / max(H, W)
    img = cv2.resize(img, (-1, -1), fx=prop, fy=prop, interpolation=cv2.INTER_AREA)
    pad_w = max((size - img.shape[1]) // 2, 0)
    pad_h = max((size - img.shape[0]) // 2, 0)
    pad = ((pad_h, pad_h), (pad_w, pad_w)) + ((0, 0),) if img.ndim == 3 else ((pad_h, pad_h), (pad_w, pad_w))
    img = np.pad(img, pad, mode="constant", constant_values=0)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    out = img.astype(np.float32)
    if normalize and img.ndim == 3:
        out = out / 255.0
    return out


def scale_bbox(bbox, from_size, to_size):
    """Map [x0,y0,x1,y1] from from_size=(W,H) into to_size=(W,H)."""
    x0, y0, x1, y1 = bbox
    fx = to_size[0] / float(from_size[0])
    fy = to_size[1] / float(from_size[1])
    return [round(x0 * fx), round(y0 * fy), round(x1 * fx), round(y1 * fy)]


def make_sample_dict(
    task,
    ep_id,
    instruction=None,
    current=None,
    window=None,
    target_img=None,
    bbox=None,
    traj=None,
    actions=None,
    collision=False,
    target_dist=None,
    valid=True,
    pointgoal=None,
    pointgoal_valid=False,
    future_rgb=None,
    depth=None,
    extra=None,
    target_visible=None,
    trajectory_valid=None,
    future_valid=None,
):
    """Split validity flags (note: target_visible != trajectory_valid).

    `valid` is kept as a compatibility alias for the visibility flag; explicit
    per-axis validity fields override the guess. Waypoint supervision must use
    `trajectory_valid`; target conditioning uses `target_visible` /
    `target_img is not None`; forward-dynamics teacher uses `future_valid`.
    """
    vis = bool(valid) if target_visible is None else bool(target_visible)
    traj_ok = bool(traj is not None and len(traj) > 0)
    return {
        "task": task,
        "ep_id": ep_id,
        "instruction": instruction or "",
        "current": current,
        "window": window or [],
        "target_img": target_img,
        "bbox": bbox,
        "traj": traj,
        "actions": actions,
        "collision": bool(collision),
        "target_dist": float(target_dist) if target_dist is not None else 0.0,
        "valid": vis,
        "target_visible": vis,
        "trajectory_valid": traj_ok if trajectory_valid is None else bool(trajectory_valid),
        "future_valid": bool(future_rgb is not None) if future_valid is None else bool(future_valid),
        "pointgoal": pointgoal,
        "pointgoal_valid": bool(pointgoal_valid),
        "future_rgb": future_rgb,
        "depth": depth,
        "extra": extra or {},
    }


def load_sample_arrays(s, img_size=224, memory_size=8):
    """Turn a lazy path-dict into tensors-shaped dict of numpy arrays.

    Returns dict with:
      current: (img_size,img_size,3) f32
      window:  (memory_size,img_size,img_size,3) f32 (zero-padded history)
      target:  (img_size,img_size,3) f32 or None
    """
    out = {"current": resize_keep_aspect(imread_rgb(s["current"]), img_size)}
    n = memory_size
    win = np.zeros((n, img_size, img_size, 3), np.float32)
    for i in range(n):  # oldest..newest just before current
        idx = len(s["window"]) - n + i
        if idx >= 0 and s["window"][idx] and os.path.isfile(s["window"][idx]):
            win[i] = resize_keep_aspect(imread_rgb(s["window"][idx]), img_size)
    out["window"] = win
    if s.get("future_rgb") and os.path.isfile(s["future_rgb"]):
        out["future_rgb"] = resize_keep_aspect(imread_rgb(s["future_rgb"]), img_size)
    out["target"] = None
    if s.get("target_img") and os.path.isfile(s["target_img"]):
        out["target"] = resize_keep_aspect(imread_rgb(s["target_img"]), img_size)
    if s.get("depth") and os.path.isfile(s["depth"]):
        d = cv2.imread(s["depth"], cv2.IMREAD_ANYDEPTH).astype(np.float32) / 10000.0
        d = resize_keep_aspect(d, img_size, normalize=False)
        d[d > 5.0] = 0
        d[d < 0.1] = 0
        out["depth"] = d[..., None] if d.ndim == 2 else d
    return out