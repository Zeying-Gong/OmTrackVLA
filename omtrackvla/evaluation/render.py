"""Render a fixed Phase 1 viz_val video with PRED/GT provenance labels."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import distributed as dist

from omtrackvla.evaluation.common import build_datasets, distributed_device, load_model


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="viz_val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-units-per-dataset", type=int)
    parser.add_argument("--identity-frames", type=int)
    parser.add_argument("--geometry-frames", type=int)
    return parser.parse_args()


def _image(tensor: torch.Tensor, size: int = 320) -> np.ndarray:
    rgb = (tensor.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (size, size))


def _box(image: np.ndarray, bbox, color, label: str) -> None:
    height, width = image.shape[:2]
    x0, y0, x1, y1 = [int(round(float(value) * scale)) for value, scale in zip(bbox, (width, height, width, height))]
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
    cv2.putText(image, label, (x0, max(16, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _canvas(left: np.ndarray, middle: np.ndarray, title: str) -> np.ndarray:
    canvas = np.zeros((380, 960, 3), dtype=np.uint8)
    canvas[40:360, :320] = left
    canvas[40:360, 320:640] = middle
    cv2.putText(canvas, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "external bbox input after frame 0: false", (650, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _identity_indices(dataset, count: int, candidate_count: int) -> list[int]:
    """Choose a deterministic visible/absent mix without decoding candidate RGB."""

    candidates = np.linspace(
        0, len(dataset) - 1, min(candidate_count, len(dataset)), dtype=np.int64
    ).tolist()
    visible, absent = [], []
    for index in candidates:
        labels = dataset.get_record(index)["supervision"]["auxiliary_labels"]
        (visible if labels["target_visible"] else absent).append(index)
    absent_count = min(len(absent), max(1, count // 4))
    visible_count = min(len(visible), count - absent_count)
    absent_indices = (
        np.linspace(0, len(absent) - 1, absent_count, dtype=np.int64).tolist()
        if absent_count
        else []
    )
    visible_indices = (
        np.linspace(0, len(visible) - 1, visible_count, dtype=np.int64).tolist()
        if visible_count
        else []
    )
    selected = [visible[index] for index in visible_indices]
    selected.extend(absent[index] for index in absent_indices)
    return selected


def _identity_frames(
    model, dataset, count: int, candidate_count: int, device: torch.device
):
    indices = _identity_indices(dataset, count, candidate_count)
    for index in indices:
        item = dataset[int(index)]
        with torch.inference_mode():
            output = model(
                item["frame0"][None].to(device),
                item["frame1"][None].to(device),
                history=item["history"][None].to(device),
                history_mask=item["history_mask"][None].to(device),
            )
        left = _image(item["frame0"])
        middle = _image(item["frame1"])
        predicted = output["bbox"][0].detach().cpu().tolist()
        confidence = float(torch.sigmoid(output["visibility_logit"])[0])
        predicted_visible = confidence >= model.visibility_probability_threshold
        if predicted_visible:
            _box(middle, predicted, (0, 255, 0), "PRED")
        if float(item["target_visible"]) > 0.5:
            _box(middle, item["target_bbox"].tolist(), (0, 220, 255), "GT")
        canvas = _canvas(left, middle, "B1-ID: initialization crop | current RGB")
        cv2.putText(canvas, f"identity confidence: {confidence:.3f}", (650, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"threshold: {model.visibility_probability_threshold:.3f}", (650, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"PRED visible: {predicted_visible}", (650, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"GT visible: {bool(item['target_visible'])}", (650, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, str(item["dataset_id"]), (650, 221), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        yield canvas


def _draw_motion(panel: np.ndarray, motion, color, label: str) -> None:
    origin = np.asarray([160.0, 270.0])
    scale = 180.0
    endpoint = origin + np.asarray([-float(motion[1]), -float(motion[0])]) * scale
    endpoint = np.clip(endpoint, [10, 10], [310, 310]).astype(int)
    cv2.arrowedLine(panel, tuple(origin.astype(int)), tuple(endpoint), color, 3, tipLength=0.15)
    cv2.putText(panel, label, (12, 25 if label == "PRED" else 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _geometry_frames(model, dataset, count: int, device: torch.device):
    for index in range(min(count, len(dataset))):
        item = dataset[index]
        with torch.inference_mode():
            output = model(item["frame0"][None].to(device), item["frame1"][None].to(device))
        left = _image(item["frame0"])
        middle = _image(item["frame1"])
        canvas = _canvas(left, middle, "B1-GEO: anchor RGB | target RGB | ego motion")
        panel = canvas[40:360, 640:960]
        _draw_motion(panel, output["motion"][0].detach().cpu(), (0, 255, 0), "PRED")
        _draw_motion(panel, item["motion"], (0, 220, 255), "GT")
        cv2.putText(panel, "x forward / y left", (12, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        yield canvas


def main() -> int:
    args = _arguments()
    if args.phase != 1 or args.split != "viz_val":
        raise ValueError("Phase 1 renderer requires --phase 1 --split viz_val")
    world_size, rank, device = distributed_device()
    if rank == 0:
        model, _ = load_model(args.checkpoint, device)
        identity, geometry, benchmark = build_datasets(
            args.config,
            split="viz_val",
            identity_limit_units=args.max_units_per_dataset,
            geometry_limit_units=args.max_units_per_dataset,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / "phase1_viz.mp4"
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(benchmark.get("render_fps", 4)), (960, 380)
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer: {output}")
        frames = 0
        try:
            identity_frames = args.identity_frames or int(benchmark["render_identity_frames"])
            identity_candidates = int(
                benchmark.get("render_identity_candidate_samples", identity_frames)
            )
            geometry_frames = args.geometry_frames or int(benchmark["render_geometry_frames"])
            if identity_frames <= 0 or identity_candidates <= 0 or geometry_frames <= 0:
                raise ValueError("render frame counts must be positive")
            for frame in _identity_frames(
                model, identity, identity_frames, identity_candidates, device
            ):
                writer.write(frame)
                frames += 1
            for frame in _geometry_frames(model, geometry, geometry_frames, device):
                writer.write(frame)
                frames += 1
        finally:
            writer.release()
        if frames == 0 or not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("renderer produced no video frames")
        print(f"rendered {frames} frames to {output}")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
