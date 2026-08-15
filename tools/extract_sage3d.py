#!/usr/bin/env python3
"""Selectively extract sage3d_track_data tars into OmTrackVLA's person_follow layout.

Per tar (one run):
  pass1: stream through the tar.gz once: read <ep>.json for every (mode,ep) to build the
         accepted set (status=="Normal" and finish==true), buffer tmp_episodes jsons, and
         record which robot*cam dirs exist per ep.
  pass2: stream again; extract ONLY accepted eps of the selected camera (fallback to the
         first available cam if the preferred one is missing):
           <ep>.json, <ep>_info.json, camera_info.json, track_object.jpg, _ACCEPTED,
           rgb frames (png -> jpg q90 to cut ~10x disk), depth frames (16-bit png as-is)
  derive: from <ep>_info.json + camera_info.json compute per-step ego-centric 8-step
         waypoints, target relative state, geometric visibility, projected target pixel
         bbox, collision -> derived.json per ep, index.json per run.

Usage:
  PY=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
  $PY tools/extract_sage3d.py --tars "A.tar.gz,B.tar.gz" \
      --tar-root /data/nfs/share/sage3d_track_data \
      --out-root /data/nfs/share/OmTrackVLA/data/sage3d_extracted \
      [--cam go2_realsense_d435i] [--frame-step 1] [--depth-step 1]
"""
import argparse
import json
import tarfile
from pathlib import Path

import numpy as np

MODES = ("stt", "dt", "at")


def read_member(tf, member):
    f = tf.extractfile(member)
    return f.read() if f else b""


def pass1(tf, run_id):
    """Return (ep_meta, cam_present, tmp_eps_buffer)."""
    ep_meta = {}
    cam_present = {}
    tmp_eps_buffer = []
    for m in tf:
        if not m.isfile():
            continue
        n = m.name
        parts = n.split("/")
        if len(parts) == 5 and parts[0] == run_id and parts[1] in MODES and parts[4] == f"{parts[2]}.json":
            mode, ep, cam = parts[1], parts[2], parts[3]
            cam_present.setdefault((mode, ep), set()).add(cam)
            if (mode, ep) not in ep_meta:
                try:
                    ep_meta[(mode, ep)] = json.loads(read_member(tf, m))
                except Exception as e:
                    print(f"  [warn] bad {n}: {e}", flush=True)
        elif len(parts) == 5 and parts[0] == run_id and parts[1] in MODES:
            cam_present.setdefault((parts[1], parts[2]), set()).add(parts[3])
        elif parts[0] == run_id and parts[1] == "tmp_episodes" and n.endswith(".json"):
            tmp_eps_buffer.append((n, read_member(tf, m)))
    return ep_meta, cam_present, tmp_eps_buffer


def select_cam(cam_present, mode, ep, preferred):
    cams = cam_present.get((mode, ep), set())
    if preferred in cams:
        return preferred
    if cams:
        return sorted(cams)[0]
    return None


def project_target(robot_pos, robot_yaw, target_pos, cam):
    """Project feet/head/left/right points of the target into camera pixels.

    Convention: habitat forward f=(cos yaw, sin yaw,0); camera on robot at +[0,0,0.3]
    with identity rotation (camera_info has only translation). ROS camera axes:
    +x right, +y down, +z forward. u=cx+fx*x/z, v=cy-fy*up/z.
    """
    R = cam["camera"]
    w, h = R["width"], R["height"]
    fx, fy = R["intrinsics"]["fx"], R["intrinsics"]["fy"]
    cx, cy = R["intrinsics"]["cx"], R["intrinsics"]["cy"]
    yaw = float(robot_yaw)
    f = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    right = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    cam_origin = np.array([robot_pos[0], robot_pos[1], robot_pos[2] + 0.3])
    base = np.array(target_pos, float)
    pts = []
    for tag, z, side in [("feet", 0.0, 0.0), ("head", 1.7, 0.0),
                         ("left", 1.0, 0.25), ("right", 1.0, -0.25)]:
        p = base.copy()
        p[2] = z
        if side:
            p = p + right * side
        d = p - cam_origin
        fwd = float(np.dot(d, f))
        if fwd <= 0.05:
            return None
        rc = float(np.dot(d, right))
        uc = float(np.dot(d, up))
        u = cx + fx * rc / fwd
        v = cy - fy * uc / fwd
        pts.append((u, v))
    us = [p[0] for p in pts]
    vs = [p[1] for p in pts]
    bbox = [max(0.0, min(us)), max(0.0, min(vs)), min(w, max(us)), min(h, max(vs))]
    return bbox


def build_waypoints(steps, s, horizon=8, stride=3):
    cur = steps[s]
    yaw = cur["robot_yaw"]
    f = np.array([np.cos(yaw), np.sin(yaw)])
    left = np.array([-np.sin(yaw), np.cos(yaw)])
    pts = []
    for j in range(horizon):
        idx = s + 1 + j * stride
        if idx >= len(steps):
            return None
        nxt = steps[idx]
        d = np.array(nxt["robot_pos"][:2]) - np.array(cur["robot_pos"][:2])
        pts.append([round(float(np.dot(d, f)), 4), round(float(np.dot(d, left)), 4)])
    return pts


def derive_ep(ep_out, ep, cam):
    info_steps = json.loads((ep_out / f"{ep}_info.json").read_text())
    cam_info = json.loads((ep_out / "camera_info.json").read_text())
    out = {"steps": [], "waypoint_horizon": 8, "waypoint_stride": 3,
           "projection_note": "target pixel bbox from world->camera projection "
                              "(cam at +0.3m, identity rotation); validate before training"}
    for s, st in enumerate(info_steps):
        robot_pos = st["robot_pos"]
        yaw = float(st["robot_yaw"])
        target_pos = st["target_pos"]
        entry = {
            "step": st["step"],
            "collision": bool(st.get("collision", False)),
            "facing": st.get("facing", 0.0),
            "dis_to_human": st.get("dis_to_human", None),
            "robot_pos": robot_pos,
            "robot_yaw": yaw,
            "target_pos": target_pos,
        }
        f = np.array([np.cos(yaw), np.sin(yaw)])
        left = np.array([-np.sin(yaw), np.cos(yaw)])
        d = np.array(target_pos[:2]) - np.array(robot_pos[:2])
        entry["target_local"] = [round(float(np.dot(d, f)), 4), round(float(np.dot(d, left)), 4)]
        entry["target_dist"] = float(np.linalg.norm(d))
        bbox = project_target(robot_pos, yaw, target_pos, cam_info)
        if bbox is None:
            entry["visible"] = False
            entry["bbox"] = None
        else:
            entry["visible"] = True
            entry["bbox"] = [float(x) for x in bbox]
        entry["waypoints_ego"] = build_waypoints(info_steps, s)
        out["steps"].append(entry)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tars", required=True)
    ap.add_argument("--tar-root", default="/data/nfs/share/sage3d_track_data")
    ap.add_argument("--out-root", default="/data/nfs/share/OmTrackVLA/data/sage3d_extracted")
    ap.add_argument("--cam", default="go2_realsense_d435i")
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--depth-step", type=int, default=1)
    ap.add_argument("--reencode-rgb", action="store_true", default=True)
    args = ap.parse_args()

    import cv2

    tars = [t.strip() for t in args.tars.split(",") if t.strip()]
    tar_root = Path(args.tar_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    global_index = []

    for tar_name in tars:
        tar_path = tar_root / tar_name
        run_id = tar_name.replace("formal_runs_", "").replace(".tar.gz", "")
        print(f"==== {tar_name} ====", flush=True)

        accepted = []
        with tarfile.open(tar_path, "r:gz") as tf:
            ep_meta, cam_present, tmp_buffer = pass1(tf, run_id)
            for (mode, ep), meta in ep_meta.items():
                if meta.get("status") == "Normal" and meta.get("finish") is True:
                    accepted.append((mode, ep, meta))
            print(f"  eps={len(ep_meta)} accepted={len(accepted)}", flush=True)

        # write tmp_episodes (all defs kept)
        tmp_root = out_root / run_id / "tmp_episodes"
        tmp_root.mkdir(parents=True, exist_ok=True)
        for name, data in tmp_buffer:
            rel = name.split("/", 2)[-1]  # {mode}/{run}/episode_X.json
            p = tmp_root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)

        run_out = out_root / run_id
        run_out.mkdir(parents=True, exist_ok=True)
        run_index = {"run_id": run_id, "tar": tar_name, "eps": []}
        for mode, ep, meta in accepted:
            cam = select_cam(cam_present, mode, ep, args.cam)
            if cam is None:
                print(f"  [warn] {mode}/{ep}: no cam, skip", flush=True)
                continue
            run_index["eps"].append({
                "mode": mode, "ep": ep, "cam": cam,
                "success": meta.get("success"), "following_rate": meta.get("following_rate"),
                "total_step": meta.get("total_step"), "instruction": meta.get("instruction"),
            })
        (run_out / "index.json").write_text(json.dumps(run_index, indent=1, ensure_ascii=False))

        # pass2: extract accepted eps, selected cam
        need_prefix = set()
        for mode, ep, meta in accepted:
            cam = select_cam(cam_present, mode, ep, args.cam)
            if cam is not None:
                need_prefix.add(f"{run_id}/{mode}/{ep}/{cam}/")

        n_rgb = n_depth = n_small = 0
        with tarfile.open(tar_path, "r:gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                n = m.name
                hit = next((p for p in need_prefix if n.startswith(p)), None)
                if hit is None:
                    continue
                ep_out = out_root / hit.rstrip("/")
                ep_out.mkdir(parents=True, exist_ok=True)
                (ep_out / "rgb").mkdir(parents=True, exist_ok=True)
                (ep_out / "depth").mkdir(parents=True, exist_ok=True)
                base = n.rsplit("/", 1)[-1]
                if base.endswith(".png"):
                    stem = int(base[:-4])
                    if "/rgb/" in n:
                        if stem % args.frame_step != 0:
                            continue
                        data = read_member(tf, m)
                        if args.reencode_rgb:
                            arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                            ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                            (ep_out / "rgb" / f"{base[:-4]}.jpg").write_bytes(buf.tobytes())
                        else:
                            (ep_out / "rgb" / base).write_bytes(data)
                        n_rgb += 1
                    elif "/depth/" in n:
                        if stem % args.depth_step != 0:
                            continue
                        (ep_out / "depth" / base).write_bytes(read_member(tf, m))
                        n_depth += 1
                    continue
                # small metadata files (ep.json, ep_info.json, camera_info.json,
                # quality.json, track_object.jpg, _ACCEPTED)
                (ep_out / base).write_bytes(read_member(tf, m))
                n_small += 1

        # derive per accepted ep
        for mode, ep, meta in accepted:
            cam = select_cam(cam_present, mode, ep, args.cam)
            if cam is None:
                continue
            ep_out = out_root / run_id / mode / ep / cam
            info_p = ep_out / f"{ep}_info.json"
            cam_p = ep_out / "camera_info.json"
            if not info_p.exists() or not cam_p.exists():
                print(f"  [warn] missing {mode}/{ep}: info.json", flush=True)
                continue
            derived = derive_ep(ep_out, ep, json.loads(cam_p.read_text()))
            (ep_out / "derived.json").write_text(json.dumps(derived, ensure_ascii=False))
            global_index.append({
                "run": run_id, "mode": mode, "ep": ep, "cam": cam,
                "path": f"{run_id}/{mode}/{ep}/{cam}",
                "steps": len(derived["steps"]),
            })
            print(f"  + {mode}/{ep}/{cam} steps={len(derived['steps'])}", flush=True)

        print(f"  frames rgb={n_rgb} depth={n_depth} small={n_small}", flush=True)

    (out_root / "index.json").write_text(json.dumps({"eps": global_index}, indent=1, ensure_ascii=False))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
