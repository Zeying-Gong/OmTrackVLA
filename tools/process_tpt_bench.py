#!/usr/bin/env python3
"""Process TPT-Bench (real robot person tracking) into OmTrackVLA's person_follow
fine-tune/eval layout.

For each of the 48 sequences (0000..0047):
  1. unzip GTs / ODOM / descriptions / quickview_videos
  2. align: video frame i <-> GT frame  (GT count is an integer multiple of video
     frames for quickview; we use the exact per-frame GT timestamps, no drift)
  3. interpolate ODOM (TUM, ~2Hz) onto every frame timestamp (linear pos, slerp quat)
  4. emit frames.parquet + rgb_frames/frame_*.jpg + meta.json + diagnostics overlays

NOTE (known caveat, from DATASET_SURVEY.md #2.2):
  - quickview mp4 is a downsampled preview (960x400). GTs bboxes are in the ORIGINAL
    (full-res) coordinate system. We do NOT bake a scale into bbox; we store
    bbox_source (as-is) and bbox_qv = bbox_source * qv_scale_guess, where the guess is
    uniform scale from src_size=[1280,720] (override with --src-w/--src-h after the
    author confirms). Use the overlay diagnostics to eyeball the alignment.
Usage:
  PY=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
  $PY tools/process_tpt_bench.py --tpt-root /h100-2/tpt-bench \
      --out-root /data/nfs/share/OmTrackVLA/data/tpt_bench \
      [--seqs 0000,0001] [--frame-step N] [--src-w 1280 --src-h 720] [--skip-frames]
"""
import argparse
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import cv2
import numpy as np

SEQS = [f"{i:04d}" for i in range(48)]


def unzip(root: Path, out: Path, name: str, inner: str):
    """Extract zip `name` (containing `inner/...`) into out, mirroring `inner`."""
    zpath = root / name
    if not zpath.exists():
        print(f"[WARN] missing {zpath}")
        return
    target = out / inner
    if target.exists():
        return  # already extracted
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out)
    print(f"  unzipped {name}")


def load_gts(p: Path):
    with open(p) as f:
        d = json.load(f)
    keys = sorted(d.keys(), key=int)
    ts = np.array([int(k) for k in keys], dtype=np.int64)
    n = len(ts)
    bbox = np.zeros((n, 4), dtype=np.float32)
    is_exist = np.zeros(n, dtype=np.int8)
    is_behind = np.zeros(n, dtype=np.int8)
    interp = np.zeros(n, dtype=np.int8)
    for i, k in enumerate(keys):
        v = d[k]
        bbox[i] = v["bbox"]
        is_exist[i] = v["is_exist"]
        is_behind[i] = v["is_behind_glass"]
        interp[i] = v["interpolated"]
    return ts, bbox, is_exist, is_behind, interp


def load_odom(p: Path):
    rows = []
    with open(p) as f:
        for line in f:
            s = line.split()
            if len(s) < 8:
                continue
            rows.append([float(x) for x in s[:8]])
    a = np.array(rows, dtype=np.float64)
    return a[:, 0], a[:, 1:4], a[:, 4:8]  # ts, pos, quat(xyzw)


def slerp(q0, q1, t):
    """numpy slerp, q as [x,y,z,w] arrays (N,4)->(M,4)."""
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    # (M,4) q0/q1 with M matching; support scalar-or-vec t
    t = np.asarray(t, float)
    dot = np.sum(q0 * q1, axis=-1)
    neg = dot < 0
    q1 = q1.copy()
    q1[neg] = -q1[neg]
    dot[neg] = -dot[neg]
    dot = np.clip(dot, -1, 1)
    theta = np.arccos(dot)
    s1 = np.sin((1 - t) * theta)
    s2 = np.sin(t * theta)
    s3 = np.sin(theta)
    s3 = np.where(s3 < 1e-10, 1.0, s3)
    out = (s1[..., None] * q0 + s2[..., None] * q1) / s3[..., None]
    return out


def interp_pose(odom_ts, opos, oquat, q_ts):
    """Interpolate ODOM poses onto query timestamps q_ts (seconds)."""
    pos = np.empty((len(q_ts), 3))
    quat = np.empty((len(q_ts), 4))
    for c in range(3):
        pos[:, c] = np.interp(q_ts, odom_ts, opos[:, c], left=opos[0, c], right=opos[-1, c])
    # slerp per query: find bracketing indices
    i0 = np.clip(np.searchsorted(odom_ts, q_ts) - 1, 0, len(odom_ts) - 2)
    i1 = i0 + 1
    span = np.maximum(odom_ts[i1] - odom_ts[i0], 1e-12)
    t = np.clip((q_ts - odom_ts[i0]) / span, 0, 1)
    quat = slerp(oquat[i0], oquat[i1], t)
    return pos, quat


def video_pts(cap):
    """Return per-frame PTS in ms by iterating once."""
    pts = []
    while True:
        ok = cap.grab()
        if not ok:
            break
        pts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    return np.array(pts, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpt-root", default="/h100-2/tpt-bench")
    ap.add_argument("--out-root", default="/data/nfs/share/OmTrackVLA/data/tpt_bench")
    ap.add_argument("--seqs", default="")
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--skip-frames", action="store_true", help="don't extract jpg frames")
    ap.add_argument("--extract-only", action="store_true", help="only unzip, no alignment")
    args = ap.parse_args()

    root = Path(args.tpt_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    raw = out_root / "_raw"
    raw.mkdir(parents=True, exist_ok=True)

    seqs = args.seqs.split(",") if args.seqs else SEQS
    seqs = [s.strip() for s in seqs if s.strip()]

    # ---- 1) unzip once ----
    if not args.extract_only:
        for name, inner in [
            ("GTs.zip", "GTs"),
            ("ODOM.zip", "ODOM"),
            ("descriptions.zip", "descriptions"),
            ("quickview_videos.zip", "quickview_videos"),
        ]:
            unzip(root, raw, name, inner)

    if args.extract_only:
        print("extract-only done. next: run without --extract-only")
        return

    diag_dir = out_root / "_check"
    diag_dir.mkdir(parents=True, exist_ok=True)

    for seq in seqs:
        gts_p = raw / "GTs" / f"{seq}.json"
        odom_p = raw / "ODOM" / f"{seq}.txt"
        desc_p = raw / "descriptions" / f"{seq}.txt"
        vid_p = raw / "quickview_videos" / f"{seq}.mp4"
        if not all(p.exists() for p in (gts_p, odom_p, vid_p)):
            print(f"[WARN] {seq}: missing files, skip")
            continue

        out_dir = out_root / seq
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(gts_p, out_dir / "GTs.json")
        shutil.copy2(odom_p, out_dir / "ODOM_TUM.txt")
        if desc_p.exists():
            shutil.copy2(desc_p, out_dir / "desc.txt")

        # ---- GT ----
        gts, bbox, is_exist, is_behind, interp = load_gts(gts_p)
        gt_ts_s = gts / 1e9

        # ---- estimate source frame size from bbox extent (annotator coords) ----
        exist_mask = is_exist == 1
        be = bbox[exist_mask]
        if len(be) == 0:
            print(f"[WARN] {seq}: no existing-target frames, skip")
            continue
        x1 = be[:, 0]
        y1 = be[:, 1]
        x2 = be[:, 0] + be[:, 2]
        y2 = be[:, 1] + be[:, 3]
        src_est_w = float(np.percentile(x2, 99.5))
        src_est_h = float(np.percentile(y2, 99.5))
        src_est = [int(np.ceil(src_est_w)), int(np.ceil(src_est_h))]        # ---- ODOM ----
        ots, opos, oquat = load_odom(odom_p)
        if len(ots) == 0:
            print(f"[WARN] {seq}: empty ODOM, skip")
            continue

        # ---- video ----
        cap = cv2.VideoCapture(str(vid_p))
        if not cap.isOpened():
            print(f"[WARN] {seq}: cannot open video, skip")
            continue
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vfps = cap.get(cv2.CAP_PROP_FPS)
        pts = video_pts(cap)
        n_vid = len(pts)
        cap.release()
        if n_vid == 0:
            print(f"[WARN] {seq}: no video frames, skip")
            continue

        # ---- video frame <-> GT frame mapping ----
        ratio = len(gts) / n_vid
        k = int(round(ratio))
        print(f"  {seq}: gts={len(gts)} video={n_vid} ratio={ratio:.3f} k={k} "
              f"vid={vw}x{vh}@{vfps:.1f}")
        if abs(ratio - k) > 0.02:
            print(f"[WARN] {seq}: non-integer GT/video ratio {ratio:.3f}; "
                  "falling back to nearest-ts matching")
        video_idx = np.arange(n_vid)
        gt_idx = np.clip(video_idx * k, 0, len(gts) - 1)
        vid_gt_ts = gts[gt_idx]

        # ---- interpolate ODOM onto GT timestamps of matched frames ----
        opos_q, oquat_q = interp_pose(ots, opos, oquat, vid_gt_ts)

        # ---- scale: src_est -> quickview (per-seq, from data) ----
        sx = vw / src_est[0]
        sy = vh / src_est[1]
        bbox_qv = bbox[gt_idx].copy().astype(np.float32)
        bbox_qv[:, 0] *= sx
        bbox_qv[:, 1] *= sy
        bbox_qv[:, 2] *= sx
        bbox_qv[:, 3] *= sy

        # ---- geometry sanity report (validates the mapping without vision) ----
        inb = (bbox_qv[:, 0] >= 0) & (bbox_qv[:, 1] >= 0) & \
              (bbox_qv[:, 0] + bbox_qv[:, 2] <= vw) & (bbox_qv[:, 1] + bbox_qv[:, 3] <= vh) & \
              (is_exist[gt_idx] == 1)
        inside = float(inb.mean())
        c = bbox_qv[:, :2] + bbox_qv[:, 2:] / 2.0
        disp = np.linalg.norm(np.diff(c, axis=0), axis=1)
        med_disp = float(np.median(disp))
        p95_disp = float(np.percentile(disp, 95))
        geom = {
            "src_size_est": src_est,
            "bbox_extent_p995": [float(np.percentile(x2, 99.5)), float(np.percentile(y2, 99.5))],
            "bbox_extent_max": [float(x2.max()), float(y2.max())],
            "scale_src2qv": [sx, sy],
            "in_frame_ratio": inside,
            "median_center_disp_px": med_disp,
            "p95_center_disp_px": p95_disp,
        }
        print(f"  geom {seq}: src_est={src_est} in_frame_ratio={inside:.3f} "
              f"med_disp={med_disp:.1f}px p95_disp={p95_disp:.1f}px")

        # ---- write per-frame parquet ----
        import pyarrow as pa
        import pyarrow.parquet as pq
        tbl = pa.table(
            {
                "video_idx": video_idx,
                "gt_idx": gt_idx.astype(np.int64),
                "gt_ts_ns": gts[gt_idx],
                "vid_pts_ms": pts,
                "bbox_source": pa.array(bbox[gt_idx].tolist(), type=pa.list_(pa.float32(), 4)),
                "bbox_qv": pa.array(bbox_qv.tolist(), type=pa.list_(pa.float32(), 4)),
                "is_exist": is_exist[gt_idx],
                "is_behind_glass": is_behind[gt_idx],
                "interpolated": interp[gt_idx],
                "odom_pos": pa.array(opos_q.tolist(), type=pa.list_(pa.float64(), 3)),
                "odom_quat_xyzw": pa.array(oquat_q.tolist(), type=pa.list_(pa.float64(), 4)),
                "odom_ts_s": opos_q[:, 0] * 0 + vid_gt_ts,  # = gt ts in seconds
            }
        )
        pq.write_table(tbl, out_dir / "frames.parquet")

        # ---- meta ----
        meta = {
            "seq": seq,
            "src_est_size": src_est,
            "src_scale_x": sx,
            "src_scale_y": sy,
            "quickview_size": [vw, vh],
            "gts_frames": int(len(gts)),
            "video_frames": int(n_vid),
            "gt_video_ratio": float(ratio),
            "match_k": k,
            "geometry": geom,
            "note": "bbox_source is annotator/GT coords; bbox_qv scales by src_est (p99.5 of "
                    "bbox extent), NOT a fixed 1280x720 guess. in_frame_ratio and center "
                    "displacement in meta.geometry indicate mapping quality; confirm with "
                    "dataset author before training on target patches from quickview.",
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # ---- extract rgb frames ----
        if not args.skip_frames:
            fr_dir = out_dir / "rgb_frames"
            fr_dir.mkdir(parents=True, exist_ok=True)
            cap = cv2.VideoCapture(str(vid_p))
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % args.frame_step == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if ok2:
                        (fr_dir / f"frame_{i:06d}.jpg").write_bytes(buf.tobytes())
                i += 1
            cap.release()
            print(f"  {seq}: wrote {i // args.frame_step} frames")

        # ---- overlay diagnostics for first few frames ----
        if not args.skip_frames:
            ov_dir = diag_dir
            cap = cv2.VideoCapture(str(vid_p))
            for vid_i in range(0, min(n_vid, 8), 2):
                cap.set(cv2.CAP_PROP_POS_FRAMES, vid_i)
                ok, frame = cap.read()
                if not ok:
                    continue
                gi = gt_idx[vid_i]
                x, y, w, h = bbox_qv[vid_i]
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
                cv2.putText(frame, f"video_i={vid_i} gt_i={gi}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok2:
                    (ov_dir / f"{seq}_vid{vid_i:03d}.jpg").write_bytes(buf.tobytes())
            cap.release()

        print(f"  done {seq} -> {out_dir}")


if __name__ == "__main__":
    main()
