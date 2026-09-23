"""Convert unified sample dicts (tools/data_loader) into model training batches.

Model convention (see architecture note 0814):
  a_k = (dx, dy, sin(delta_yaw), cos(delta_yaw))   ego-local incremental action

Absolute ego waypoints w = [(x, y) | (x, y, yaw)] in the robot frame at time t
are converted to incremental actions consistent with the cumulative-SE(2)
reconstruction in losses._cumulative_se2_error: action k's translation is
expressed in the frame of the accumulated heading before step k.

Source specifics handled here:
  sage3d : traj = [x, y] x8, no heading -> heading from displacement
  tpt    : no trajectory -> action_valid = 0
  navdp  : traj = [x, y, yaw] x(predict+1) -> heading available; subsample
"""
from __future__ import annotations

import numpy as np
import torch

from unified import (
    TASK_PERSON_FOLLOW, TASK_POINTNAV, TASK_IMAGEGOAL,
    resize_keep_aspect, imread_rgb,
)

TASK_ID = {TASK_PERSON_FOLLOW: 0, TASK_POINTNAV: 1, TASK_IMAGEGOAL: 2}
SUCCESS_RADIUS = 1.0  # unified 1.0 m protocol (note 16)


def waypoints_to_actions(wpts, horizon=8):
    """Absolute ego waypoints -> (H, 4) actions (dx, dy, sin, cos) + n_valid.

    wpts: list of [x, y] or [x, y, yaw]; yaw recovered from displacement when absent.
    The origin (current pose) is prepended EXPLICITLY when the first waypoint is
    not the origin (e.g. Sage3D stores only FUTURE +1,+4,... frame positions, no
    origin). Positions are padded to horizon+1 (last point repeated) so horizon
    actions are produced; a held final step maps to a zero displacement action.

    Returns (acts, n_valid): acts (horizon, 4); n_valid = number of REAL
    displacement steps = count of non-held (non-zero) consecutive displacements
    (capped at horizon). Held/duplicated trailing steps are NOT supervised.
    """
    if not wpts:
        return None
    pts = [np.asarray(w, dtype=np.float64) for w in wpts if w is not None]
    if len(pts) < 2:
        return None
    pos = np.stack([p[:2] for p in pts])            # (M, 2)

    # Sage3D/TpT store future-only waypoints (no origin) -> prepend [0,0].
    if np.hypot(pos[0, 0], pos[0, 1]) > 1e-6:
        pos = np.concatenate([np.zeros((1, 2)), pos], axis=0)
        if pts[0].size >= 3:
            pts = [np.array([0.0, 0.0, pts[0][2]], dtype=np.float64)] + pts
    M = pos.shape[0]

    if pts[0].size >= 3:
        yaw = np.array([p[2] for p in pts], dtype=np.float64)
    else:
        d = np.diff(pos, axis=0)                    # (M-1, 2)
        yaw = np.arctan2(d[:, 1], d[:, 0])
        yaw = np.concatenate([yaw, yaw[-1:]])

    # count real (non-held) displacement steps BEFORE padding/truncation
    disp = np.diff(pos, axis=0)                     # (M-1, 2)
    n_real = int((np.hypot(disp[:, 0], disp[:, 1]) > 1e-6).sum())
    n_valid = min(n_real, horizon)

    if M < horizon + 1:
        pad = np.repeat(pos[-1:], horizon + 1 - M, axis=0)
        pos = np.concatenate([pos, pad], axis=0)
        yaw = np.concatenate([yaw, np.repeat(yaw[-1:], horizon + 1 - M)])
    else:
        pos = pos[: horizon + 1]
        yaw = yaw[: horizon + 1]

    acts = np.zeros((horizon, 4), dtype=np.float64)
    h0 = yaw[0]
    for k in range(horizon):
        dyaw = yaw[k + 1] - yaw[k]
        psi = yaw[k] - h0                                # accumulated heading before step k
        c, s = np.cos(psi), np.sin(psi)
        d = pos[k + 1] - pos[k]
        acts[k, 0] = c * d[0] + s * d[1]                 # rotate into frame of psi
        acts[k, 1] = -s * d[0] + c * d[1]
        acts[k, 2] = np.sin(dyaw)
        acts[k, 3] = np.cos(dyaw)
    return acts, n_valid


def _task_id(task):
    return TASK_ID.get(task, 0)


def _goal_spec(task):
    """[desired_distance, d_min, d_max, success_radius, terminal_mode]."""
    if task == TASK_PERSON_FOLLOW:
        return np.array([2.0, 1.0, 3.0, 0.5, 0.0], dtype=np.float32)   # maintain
    return np.array([1.0, 0.0, 0.0, SUCCESS_RADIUS, 1.0], dtype=np.float32)  # stop-on-arrival


def _history_frames(sample, max_history=8):
    """Filter zero-padded window frames; return (stack, mask) or None.

    mask (K,) marks REAL frames (1) vs repeat-last padded history (0), so
    padded copies of the oldest real frame do not get treated as independent
    observations in Global Attention.
    """
    win = sample.get("window_arr")
    if win is None:
        return None
    valid = [w for w in win if w is not None and float(np.abs(w).sum()) > 1e-4]
    if not valid:
        return None
    valid = valid[-max_history:]
    K = len(valid)
    mask = np.ones((K,), np.float32)
    if K < max_history:
        n_pad = max_history - K
        valid = [valid[0]] * n_pad + valid
        mask = np.concatenate([np.zeros((n_pad,), np.float32), mask])
    return np.stack(valid), mask                      # ((K, H, W, 3), (K,))


def collate_batch(samples, image_size=224, max_history=8, horizon=8, device="cpu"):
    """samples: list of dicts (with *_arr arrays). Returns (model_kwargs, loss_batch)."""
    B = len(samples)
    cur = np.stack([s["current_arr"] for s in samples])           # (B, H, W, 3)
    cur_t = torch.as_tensor(cur).permute(0, 3, 1, 2).float().to(device)

    mkw = {"current_rgb": cur_t}

    # ---- target image / validity -------------------------------------------
    tgts, tvalid = [], []
    for s in samples:
        t = s.get("target_arr")
        if t is not None:
            tgts.append(t)
            tvalid.append(1.0)
        else:
            tgts.append(np.zeros_like(s["current_arr"]))
            tvalid.append(0.0)
    mkw["target_image"] = torch.as_tensor(np.stack(tgts)).permute(0, 3, 1, 2).float().to(device)
    mkw["target_valid"] = torch.as_tensor(tvalid, dtype=torch.float32, device=device)
    mkw["target_confidence"] = torch.as_tensor(tvalid, dtype=torch.float32, device=device)
    mkw["target_type"] = torch.as_tensor(
        [0 if s["task"] == TASK_PERSON_FOLLOW else (1 if s["task"] == TASK_POINTNAV else 2)
         for s in samples], dtype=torch.long, device=device)

    # ---- history frames ------------------------------------------------------
    hist = [_history_frames(s, max_history) for s in samples]
    if any(h is not None for h in hist):
        K = hist[0][0].shape[0] if hist[0] is not None else max_history
        harr = np.zeros((B, K, image_size, image_size, 3), np.float32)
        hmask = np.zeros((B, K), np.float32)
        for i, h in enumerate(hist):
            if h is not None:
                harr[i] = h[0]
                hmask[i] = h[1]
        mkw["history_rgb"] = torch.as_tensor(harr).permute(0, 1, 4, 2, 3).float().to(device)
        mkw["history_valid"] = torch.as_tensor(hmask, device=device)

    # ---- history_motion (B, K, 2): ego displacement over the history window ----
    hm_arr = np.zeros((B, 1, 2), np.float32)
    hm_valid = np.zeros((B, 1), np.float32)
    for i, s in enumerate(samples):
        m = s.get("extra", {}).get("history_motion")
        if m is not None and len(m) >= 2:
            hm_arr[i, 0, :2] = m[:2]
            hm_valid[i, 0] = 1.0
    if hm_valid.sum() > 0:
        mkw["history_motion"] = torch.as_tensor(hm_arr, device=device)
        mkw["history_motion_valid"] = torch.as_tensor(hm_valid, device=device)

    # ---- pointgoal (B, 7): x, y, range, bearing, uncertainty, age, valid ----
    if any(s.get("pointgoal") is not None for s in samples):
        pg = np.zeros((B, 7), np.float32)
        for i, s in enumerate(samples):
            p = s.get("pointgoal")
            if p is not None and len(p) >= 2:
                x, y = float(p[0]), float(p[1])
                pg[i, 0], pg[i, 1] = x, y
                pg[i, 2] = np.sqrt(x * x + y * y)
                pg[i, 3] = np.arctan2(y, x)
                pg[i, 6] = 1.0 if s.get("pointgoal_valid") else 0.0
        mkw["pointgoal"] = torch.as_tensor(pg, device=device)

    # ---- task / goal spec ----------------------------------------------------
    mkw["task_type"] = torch.as_tensor([_task_id(s["task"]) for s in samples],
                                       dtype=torch.long, device=device)
    mkw["goal_spec"] = torch.as_tensor(np.stack([_goal_spec(s["task"]) for s in samples]),
                                       dtype=torch.float32, device=device)

    # ---- future rgb (training-only teacher path) -----------------------------
    fvalid = np.zeros((B,), np.float32)
    if any(s.get("future_rgb") for s in samples):
        fut = []
        for i, s in enumerate(samples):
            p = s.get("future_rgb")
            if p and isinstance(p, str):
                fut.append(resize_keep_aspect(imread_rgb(p), image_size))
                fvalid[i] = 1.0
            elif isinstance(p, np.ndarray):
                fut.append(p)
                fvalid[i] = 1.0
            else:
                fut.append(np.zeros((image_size, image_size, 3), np.float32))
        mkw["future_rgb"] = torch.as_tensor(np.stack(fut)).permute(0, 3, 1, 2).float().to(device)
    mkw["future_valid"] = torch.as_tensor(fvalid, device=device)

    # ---- forward-dynamics depth/free-space targets (B, gridH*gridW) ------------
    grid_h = grid_w = image_size // 14      # DINOv2 patch grid (16x16 at 224)
    depth_res = np.zeros((B, grid_h * grid_w), np.float32)
    free_tgt = np.zeros((B, grid_h * grid_w), np.float32)
    dvalid = np.zeros((B,), np.float32)
    for i, s in enumerate(samples):
        d_cur = s.get("depth_arr")
        fd = s.get("future_depth")
        if d_cur is not None and fd is not None:
            # 2D -> single channel, downscale to grid
            dc = d_cur[..., 0] if d_cur.ndim == 3 else d_cur
            fc = fd[..., 0] if fd.ndim == 3 else fd
            import cv2
            dc_g = cv2.resize(dc, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
            fc_g = cv2.resize(fc, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
            depth_res[i] = (fc_g - dc_g).reshape(-1)
            free_tgt[i] = ((fc_g > 0.15) & (fc_g < 4.5)).astype(np.float32).reshape(-1)
            dvalid[i] = 1.0

    # ---- loss batch ------------------------------------------------------------
    traj = np.zeros((B, horizon, 4), np.float32)
    a_valid = np.zeros((B, horizon), np.float32)
    a_conf = np.zeros((B, horizon), np.float32)
    tstate = np.zeros((B, 3), np.float32)
    tstate_valid = np.zeros((B, 3), np.float32)
    stop_label = np.zeros((B,), np.float32)
    stop_mask = np.zeros((B,), np.int64)

    for i, s in enumerate(samples):
        r = waypoints_to_actions(s.get("traj"), horizon)
        # Waypoint supervision keyed on TRAJECTORY validity (not target
        # visibility): a hidden target still has a reliable sim trajectory.
        if r is not None and s.get("trajectory_valid", True):
            acts, n_valid = r
            traj[i] = acts
            succ = s.get("extra", {}).get("success")
            conf = float(succ) if succ is not None else 1.0   # success=0 must stay 0, not or-1.0
            a_valid[i, :n_valid] = 1.0                        # only REAL displacement steps
            a_conf[i, :n_valid] = max(conf, 0.1)
        # target relative state (dx, dy, vis)
        tlocal = s.get("extra", {}).get("target_local")
        if tlocal is not None and len(tlocal) >= 2:
            tstate[i, 0], tstate[i, 1] = float(tlocal[0]), float(tlocal[1])
            tstate[i, 2] = 1.0 if s.get("target_visible", True) else 0.0
            tstate_valid[i] = 1.0
        elif s.get("pointgoal") is not None:
            p = s["pointgoal"]
            tstate[i, 0], tstate[i, 1] = float(p[0]), float(p[1])
            tstate[i, 2] = 1.0 if s.get("pointgoal_valid") else 0.0
            tstate_valid[i] = 1.0
        # stop supervision: only terminal tasks (pointnav/imagegoal)
        if s["task"] in (TASK_POINTNAV, TASK_IMAGEGOAL):
            stop_mask[i] = 1
            gd = s.get("extra", {}).get("goal_dist")
            d = float(gd) if gd is not None else (
                float(np.hypot(tstate[i, 0], tstate[i, 1])) if s.get("pointgoal") else float("inf")
            )
            stop_label[i] = 1.0 if d <= SUCCESS_RADIUS else 0.0

    loss_batch = {
        "trajectory": torch.as_tensor(traj, device=device),
        "action_valid": torch.as_tensor(a_valid, device=device),
        "action_confidence": torch.as_tensor(a_conf, device=device),
        "target_state": torch.as_tensor(tstate, device=device),
        "target_state_valid": torch.as_tensor(tstate_valid, device=device),
        "stop_label": torch.as_tensor(stop_label, device=device),
        "stop_task_mask": torch.as_tensor(stop_mask, device=device),
        "future_valid": torch.as_tensor(fvalid, device=device),
        "depth_residual": torch.as_tensor(depth_res, device=device),
        "free_target": torch.as_tensor(free_tgt, device=device),
        "depth_valid": torch.as_tensor(dvalid, device=device),
    }
    return mkw, loss_batch
