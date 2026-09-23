"""Aggregate statistics over the three OmTrackVLA data sources (lightweight).

Reads index/derived/metadata files only; does not decode images except a tiny
sample for bbox sanity. Outputs JSON to stdout.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq


def stat_tpt(tpt_root, exclude=("0002",)):
    rows = {"seqs": 0, "video_frames": 0, "valid_frames": 0, "behind_glass": 0,
            "interpolated": 0, "bboxes": 0}
    geom = {"in_frame_ratio": [], "median_center_disp_px": []}
    for s in sorted(os.listdir(tpt_root)):
        if not s.isdigit() or s in exclude:
            continue
        p = os.path.join(tpt_root, s, "frames.parquet")
        if not os.path.exists(p):
            continue
        t = pq.read_table(p)
        d = t.to_pydict()
        n = t.num_rows
        rows["seqs"] += 1
        rows["video_frames"] += n
        v = np.array(d["is_exist"], dtype=np.int8)
        rows["valid_frames"] += int((v == 1).sum())
        rows["behind_glass"] += int(np.array(d["is_behind_glass"]).sum())
        rows["interpolated"] += int(np.array(d["interpolated"]).sum())
        b = np.array(d["bbox_qv"])
        rows["bboxes"] += int((v == 1).sum())
        meta_path = os.path.join(tpt_root, s, "meta.json")
        if os.path.exists(meta_path):
            try:
                m = json.load(open(meta_path))
                g = m.get("geometry", {})
                geom["in_frame_ratio"].append(g.get("in_frame_ratio", 0))
                geom["median_center_disp_px"].append(g.get("median_center_disp_px", 0))
            except Exception:
                pass
    return {
        "rows": rows,
        "in_frame_ratio_mean": float(np.mean(geom["in_frame_ratio"])),
        "median_center_disp_px_mean": float(np.mean(geom["median_center_disp_px"])),
    }


def stat_sage(sage_root):
    eps = {"n": 0, "accepted": 0, "steps": 0, "collision": 0, "visible": 0,
           "cams": set(), "modes": set(), "runs": set()}
    dis = []
    wp_pts = []
    for run in sorted(os.listdir(sage_root)):
        if not os.path.isdir(os.path.join(sage_root, run)):
            continue
        ip = os.path.join(sage_root, run, "index.json")
        if not os.path.exists(ip):
            continue
        idx = json.load(open(ip))
        eps["runs"].add(run)
        for e in idx.get("eps", []):
            eps["n"] += 1
            if float(e.get("success", 0)) > 0:
                eps["accepted"] += 1
            eps["modes"].add(str(e.get("mode")))
            eps["cams"].add(str(e.get("cam")))
            dp = os.path.join(sage_root, run, str(e.get("mode")), str(e.get("ep")),
                              str(e.get("cam")), "derived.json")
            if not os.path.exists(dp):
                continue
            dr = json.load(open(dp))
            steps = dr["steps"]
            eps["steps"] += len(steps)
            eps["collision"] += sum(1 for st in steps if st.get("collision"))
            eps["visible"] += sum(1 for st in steps if st.get("visible"))
            dis += [float(st.get("dis_to_human", 0)) for st in steps if st.get("dis_to_human")]
            wp_pts += [len(st.get("waypoints_ego") or []) for st in steps]
    return {
        "episodes": eps["n"],
        "accepted": eps["accepted"],
        "steps_total": eps["steps"],
        "collision_frames": eps["collision"],
        "visible_frames": eps["visible"],
        "visible_ratio": round(eps["visible"] / max(1, eps["steps"]), 4),
        "collision_ratio": round(eps["collision"] / max(1, eps["steps"]), 4),
        "dis_to_human": {
            "min": round(float(np.min(dis)), 2) if dis else None,
            "median": round(float(np.median(dis)), 2) if dis else None,
            "max": round(float(np.max(dis)), 2) if dis else None,
        },
        "waypoint_pts": {"min": min(wp_pts) if wp_pts else 0, "max": max(wp_pts) if wp_pts else 0},
        "modes": sorted(eps["modes"]),
        "cams": sorted(eps["cams"]),
        "runs": sorted(eps["runs"]),
    }


def stat_navdp(navdp_root, max_scenes_per_group=200):
    groups = {}
    eps = 0
    traj_lens = []
    for g in sorted(os.listdir(navdp_root)):
        gr = os.path.join(navdp_root, g)
        if not os.path.isdir(gr):
            continue
        scenes = [s for s in os.listdir(gr) if os.path.isdir(os.path.join(gr, s))]
        groups[g] = {"scenes": len(scenes), "episodes": 0}
        for sc in scenes[:max_scenes_per_group]:
            sp = os.path.join(gr, sc, "meta", "episodes_stats.jsonl")
            if not os.path.exists(sp):
                continue
            n = sum(1 for _ in open(sp))
            groups[g]["episodes"] += n
            eps += n
        data_dir = os.path.join(gr, scenes[0], "data") if scenes else None
        if data_dir and os.path.isdir(data_dir):
            chunks = [c for c in os.listdir(data_dir) if c.startswith("chunk")]
            if chunks:
                for par in os.listdir(os.path.join(data_dir, chunks[0])):
                    if par.endswith(".parquet"):
                        t = pq.read_table(os.path.join(data_dir, chunks[0], par))
                        traj_lens.append(t.num_rows)
                        break
    return {
        "groups": groups,
        "total_episodes": eps,
        "traj_len_sample": {
            "min": int(np.min(traj_lens)) if traj_lens else None,
            "median": int(np.median(traj_lens)) if traj_lens else None,
            "max": int(np.max(traj_lens)) if traj_lens else None,
        },
    }


def main():
    sage_root = "/data/nfs/share/OmTrackVLA/data/sage3d_extracted"
    tpt_root = "/data/nfs/share/OmTrackVLA/data/tpt_bench"
    navdp_root = "/h100-2/vln_n1/traj_data"
    out = {
        "tpt": stat_tpt(tpt_root),
        "sage3d": stat_sage(sage_root),
        "navdp": stat_navdp(navdp_root),
    }
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()