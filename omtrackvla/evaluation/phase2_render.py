"""Render Phase 2 open-loop diagnostics as PNG frames and an MP4."""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import distributed as dist

from omtrackvla.data.phase2 import CONDITION_MODES
from omtrackvla.evaluation.phase2_common import (
    build_phase2_dataset,
    load_phase2_model,
)


def _image(tensor: torch.Tensor, size: int = 480) -> np.ndarray:
    rgb = (
        tensor.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0
    ).astype(np.uint8)
    return cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (size, size))


def _box(image: np.ndarray, bbox, color, label: str, label_row: int) -> None:
    height, width = image.shape[:2]
    scales = (width, height, width, height)
    x0, y0, x1, y1 = [
        int(round(float(value) * scale)) for value, scale in zip(bbox, scales)
    ]
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
    cv2.putText(
        image,
        label,
        (8, 20 + 22 * label_row),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA,
    )


def _polyline(panel: np.ndarray, waypoints, color, origin, scale: float) -> None:
    points = []
    for forward, left in waypoints:
        point = origin + np.asarray((-float(left), -float(forward))) * scale
        points.append(np.clip(point, (10, 10), (panel.shape[1] - 10, panel.shape[0] - 10)).astype(int))
    if len(points) > 1:
        cv2.polylines(panel, [np.asarray(points)], False, color, 3, cv2.LINE_AA)
    for point in points:
        cv2.circle(panel, tuple(point), 3, color, -1, cv2.LINE_AA)


def _render_frame(model, item, device: torch.device) -> np.ndarray:
    inputs = {
        key: item[key][None].to(device)
        for key in (
            "visual_xy",
            "visual_confidence",
            "visual_valid",
            "uwb_xy",
            "uwb_quality",
            "uwb_valid",
            "uwb_age_s",
            "condition_index",
        )
    }
    with torch.inference_mode():
        output = model(**inputs)
    rgb = _image(item["current_rgb"])
    if bool(item["predicted_visible"]):
        visual_enabled = item["condition_mode"] in {"visual_uwb", "visual_only"}
        _box(
            rgb,
            item["predicted_bbox"],
            (0, 255, 0) if visual_enabled else (180, 180, 180),
            "PRED frozen perception" if visual_enabled else "PRED diagnostic (masked)",
            0,
        )
    if bool(item["target_visible"]):
        _box(rgb, item["target_bbox"], (0, 220, 255), "GT diagnostic", 1)
    canvas = np.zeros((540, 960, 3), dtype=np.uint8)
    canvas[60:540, :480] = rgb
    panel = canvas[60:540, 480:960]
    origin = np.asarray((240.0, 410.0))
    cv2.line(panel, (240, 410), (240, 30), (80, 80, 80), 1)
    cv2.line(panel, (30, 410), (450, 410), (80, 80, 80), 1)
    _polyline(
        panel,
        output["waypoints"][0].detach().cpu().tolist(),
        (0, 255, 0),
        origin,
        120.0,
    )
    _polyline(
        panel,
        item["waypoints"].tolist(),
        (0, 220, 255),
        origin,
        120.0,
    )
    if bool(item["uwb_valid"]):
        forward, left = item["uwb_xy"].tolist()
        point = origin + np.asarray((-left, -forward)) * 120.0
        point = np.clip(point, (10, 10), (470, 470)).astype(int)
        cv2.circle(panel, tuple(point), 7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(panel, "UWB", tuple(point + (8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    cv2.putText(canvas, "Phase 2: frozen identity/depth front end -> waypoint decoder", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.67, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, "x forward / y left | 1 m = 120 px", (12, 466), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190, 190, 190), 1, cv2.LINE_AA)
    lines = (
        f"mode: {item['condition_mode']}",
        f"visual valid/conf: {bool(item['visual_valid'])} / {float(item['visual_confidence']):.3f}",
        f"UWB valid/age: {bool(item['uwb_valid'])} / {float(item['uwb_age_s']):.3f}s",
        "external bbox input after initialization: false",
        "green=PRED; yellow=GT (diagnostic labels only)",
    )
    for number, line in enumerate(lines):
        cv2.putText(panel, line, (12, 24 + number * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.line(panel, (12, 150), (42, 150), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(panel, "PRED waypoint", (50, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(panel, (12, 177), (42, 177), (0, 220, 255), 3, cv2.LINE_AA)
    cv2.putText(panel, "GT expert", (50, 182), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 220, 255), 1, cv2.LINE_AA)
    return canvas


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def run(args, distributed_device) -> int:
    if args.split != "viz_val":
        raise ValueError("Phase 2 renderer requires --split viz_val")
    world_size, rank, device = distributed_device()
    if rank == 0:
        model, checkpoint = load_phase2_model(args.checkpoint, device)
        dataset, benchmark = build_phase2_dataset(
            args.config,
            "viz_val",
            maximum_units=args.max_units_per_dataset,
            perception_cache=args.perception_cache,
            include_render_data=True,
            allow_partial_cache=args.allow_partial_cache,
        )
        frames_per_mode = int(args.frames_per_mode or benchmark["render_frames_per_mode"])
        if frames_per_mode <= 0:
            raise ValueError("Phase 2 render_frames_per_mode must be positive")
        selections = []
        for mode_index, mode in enumerate(CONDITION_MODES):
            candidates = list(range(mode_index, len(dataset), len(CONDITION_MODES)))
            count = min(frames_per_mode, len(candidates))
            positions = np.linspace(0, len(candidates) - 1, count, dtype=np.int64)
            selections.extend((mode, candidates[int(position)]) for position in positions)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        frame_dir = args.output_dir / "phase2_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        video_path = args.output_dir / "phase2_viz.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(benchmark.get("render_fps", 4)),
            (960, 540),
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer: {video_path}")
        index = []
        try:
            for frame_number, (mode, dataset_index) in enumerate(selections, 1):
                item = dataset[dataset_index]
                frame = _render_frame(model, item, device)
                name = f"frame_{frame_number:03d}_{mode}_{_safe_name(item['sample_id'])}.png"
                path = frame_dir / name
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"could not write Phase 2 frame: {path}")
                writer.write(frame)
                index.append(
                    {
                        "frame": frame_number,
                        "file": name,
                        "sample_id": item["sample_id"],
                        "condition_mode": mode,
                    }
                )
        finally:
            writer.release()
        if not index or not video_path.is_file() or video_path.stat().st_size <= 0:
            raise RuntimeError("Phase 2 renderer produced no valid output")
        (frame_dir / "index.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
                    "frames": index,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"rendered {len(index)} PNG frames and {video_path}")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0
