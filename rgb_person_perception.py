#!/usr/bin/env python3
"""RGB person detection and short-term association for the oracle controller."""

from __future__ import annotations

import math
import multiprocessing as mp
from pathlib import Path
import traceback
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from oracle_modular_follow import TargetObservation, bbox_to_footpoint


DEFAULT_WEIGHTS = Path(__file__).resolve().parent / (
    "models/torchvision/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth"
)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def metric_depth(raw: np.ndarray, max_depth_m: float = 10.0) -> np.ndarray:
    depth = np.asarray(raw, dtype=np.float32).squeeze().copy()
    depth[~np.isfinite(depth)] = 0.0
    if depth.size and float(depth.max()) <= 1.01:
        depth *= max_depth_m
    depth[(depth < 0.1) | (depth > max_depth_m)] = 0.0
    return depth


def bbox_depth_to_relative(
    bbox: Sequence[float],
    raw_depth: np.ndarray,
    hfov_deg: float = 90.0,
    max_depth_m: float = 10.0,
) -> Optional[Tuple[float, float]]:
    depth = metric_depth(raw_depth, max_depth_m=max_depth_m)
    if depth.ndim != 2 or not depth.size:
        return None
    height, width = depth.shape
    x1, y1, x2, y2 = map(float, bbox)
    # Use the torso, where depth is less contaminated by floor/background pixels.
    xa = int(np.clip(round(x1 + 0.25 * (x2 - x1)), 0, width - 1))
    xb = int(np.clip(round(x1 + 0.75 * (x2 - x1)), xa + 1, width))
    ya = int(np.clip(round(y1 + 0.20 * (y2 - y1)), 0, height - 1))
    yb = int(np.clip(round(y1 + 0.75 * (y2 - y1)), ya + 1, height))
    values = depth[ya:yb, xa:xb]
    values = values[values > 0.0]
    if values.size < 8:
        return None
    forward = float(np.median(values))
    focal = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    center_x = 0.5 * (x1 + x2)
    # Image x grows rightward, while TargetObservation uses positive-left.
    left = -((center_x - (width - 1.0) / 2.0) * forward / focal)
    return forward, float(left)


def color_histogram(rgb: np.ndarray, bbox: Sequence[float]) -> Optional[np.ndarray]:
    image = np.asarray(rgb)[..., :3]
    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(float, bbox)
    x1 = int(np.clip(math.floor(x1), 0, width - 1))
    x2 = int(np.clip(math.ceil(x2), x1 + 1, width))
    y1 = int(np.clip(math.floor(y1), 0, height - 1))
    y2 = int(np.clip(math.ceil(y2), y1 + 1, height))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    hist = cv2.normalize(hist, None, norm_type=cv2.NORM_L1).reshape(-1)
    return hist.astype(np.float32)


class RGBPersonPerception:
    """COCO person detector with first-visible initialization and RGB tracking."""

    name = "rgb-person-mobilenet-v1"

    def __init__(
        self,
        weights_path: Path | str = DEFAULT_WEIGHTS,
        score_threshold: float = 0.55,
        association_threshold: float = 0.20,
        device: str = "cuda",
        max_depth_m: float = 10.0,
        hfov_deg: float = 90.0,
    ) -> None:
        import torch
        from torchvision.models.detection import (
            fasterrcnn_mobilenet_v3_large_320_fpn,
        )

        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(f"Person detector weights not found: {weights_path}")
        self.device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None, weights_backbone=None
        )
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)
        self.score_threshold = float(score_threshold)
        self.association_threshold = float(association_threshold)
        self.max_depth_m = float(max_depth_m)
        self.hfov_deg = float(hfov_deg)
        self.reset()

    def reset(self) -> None:
        self._bbox = None
        self._reference_hist = None
        self._last_relative = None
        self.last_candidate_count = 0
        self.last_association_score = 0.0

    def _detect(self, rgb: np.ndarray):
        import torch

        image = np.asarray(rgb)[..., :3]
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        tensor = tensor.to(self.device, dtype=torch.float32).div_(255.0)
        with torch.inference_mode():
            output = self.model([tensor])[0]
        boxes = output["boxes"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        return [
            (box.astype(np.float32), float(score))
            for box, label, score in zip(boxes, labels, scores)
            if int(label) == 1 and float(score) >= self.score_threshold
        ]

    @staticmethod
    def _initial_score(box, detector_score, width, height):
        x1, y1, x2, y2 = box
        area_fraction = max(0.0, (x2 - x1) * (y2 - y1)) / (width * height)
        cx = 0.5 * (x1 + x2) / width
        center_penalty = abs(cx - 0.5)
        return detector_score + 0.35 * math.sqrt(area_fraction) - 0.25 * center_penalty

    def _select(self, rgb: np.ndarray, candidates):
        height, width = np.asarray(rgb).shape[:2]
        features = [(box, score, color_histogram(rgb, box)) for box, score in candidates]
        if self._bbox is None:
            box, score, hist = max(
                features,
                key=lambda item: self._initial_score(item[0], item[1], width, height),
            )
            return box, score, hist, score

        previous_center = np.array(
            [0.5 * (self._bbox[0] + self._bbox[2]), 0.5 * (self._bbox[1] + self._bbox[3])]
        )
        diagonal = math.hypot(width, height)
        ranked = []
        for box, detector_score, hist in features:
            center = np.array([0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])])
            center_score = math.exp(-4.0 * float(np.linalg.norm(center - previous_center)) / diagonal)
            appearance = 0.0
            if self._reference_hist is not None and hist is not None:
                appearance = float(cv2.compareHist(
                    self._reference_hist, hist, cv2.HISTCMP_INTERSECT
                ))
            association = (
                0.45 * bbox_iou(self._bbox, box)
                + 0.35 * appearance
                + 0.15 * center_score
                + 0.05 * detector_score
            )
            ranked.append((association, box, detector_score, hist))
        association, box, score, hist = max(ranked, key=lambda item: item[0])
        if association < self.association_threshold:
            return None
        return box, score, hist, association

    def __call__(self, rgb: np.ndarray, depth: np.ndarray) -> TargetObservation:
        candidates = self._detect(rgb)
        self.last_candidate_count = len(candidates)
        selected = self._select(rgb, candidates) if candidates else None
        if selected is None:
            self.last_association_score = 0.0
            relative = self._last_relative or (self.max_depth_m, 0.0)
            forward, left = relative
            return TargetObservation(
                visible=False,
                bbox_xyxy=None,
                footpoint_uv=None,
                relative_xy=(forward, left),
                range_m=math.hypot(forward, left),
                bearing_rad=math.atan2(left, forward),
                mask_area=0,
                confidence=0.0,
            )

        box, detector_score, hist, association = selected
        relative = bbox_depth_to_relative(
            box, depth, hfov_deg=self.hfov_deg, max_depth_m=self.max_depth_m
        )
        if relative is None:
            relative = self._last_relative
        if relative is None:
            self.last_association_score = 0.0
            return TargetObservation(
                visible=False,
                bbox_xyxy=None,
                footpoint_uv=None,
                relative_xy=(self.max_depth_m, 0.0),
                range_m=self.max_depth_m,
                bearing_rad=0.0,
                mask_area=0,
                confidence=0.0,
            )

        self._bbox = box.copy()
        self._last_relative = relative
        if hist is not None:
            self._reference_hist = (
                hist if self._reference_hist is None
                else 0.9 * self._reference_hist + 0.1 * hist
            )
        self.last_association_score = float(association)
        bbox = tuple(int(round(value)) for value in box)
        forward, left = relative
        return TargetObservation(
            visible=True,
            bbox_xyxy=bbox,
            footpoint_uv=bbox_to_footpoint(bbox),
            relative_xy=(forward, left),
            range_m=math.hypot(forward, left),
            bearing_rad=math.atan2(left, forward),
            mask_area=max(1, int(0.55 * (box[2] - box[0]) * (box[3] - box[1]))),
            confidence=float(detector_score),
        )


def _perception_worker(connection, kwargs) -> None:
    try:
        perception = RGBPersonPerception(**kwargs)
        connection.send({"event": "ready", "device": str(perception.device)})
        while True:
            request = connection.recv()
            command = request[0]
            if command == "close":
                break
            if command == "reset":
                perception.reset()
                connection.send({"event": "reset"})
                continue
            if command != "infer":
                raise ValueError(f"Unknown perception worker command: {command}")
            target = perception(request[1], request[2])
            connection.send({
                "event": "result",
                "target": target,
                "candidate_count": perception.last_candidate_count,
                "association_score": perception.last_association_score,
            })
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send({
                "event": "error",
                "error": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            })
        except (EOFError, BrokenPipeError):
            pass
    finally:
        connection.close()


class RGBPersonPerceptionWorker:
    """Run Torch CUDA perception outside the Habitat EGL process."""

    name = RGBPersonPerception.name + "-worker"

    def __init__(self, **kwargs) -> None:
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(
            target=_perception_worker,
            args=(child, kwargs),
            daemon=True,
            name="rgb-person-perception",
        )
        self._process.start()
        child.close()
        response = self._connection.recv()
        self._check_response(response, "ready")
        self.device = response["device"]
        self.last_candidate_count = 0
        self.last_association_score = 0.0

    @staticmethod
    def _check_response(response, expected_event) -> None:
        if response.get("event") == "error":
            raise RuntimeError(
                f"Perception worker failed: {response['error']}: "
                f"{response['message']}\n{response['traceback']}"
            )
        if response.get("event") != expected_event:
            raise RuntimeError(
                f"Expected perception worker event {expected_event}, got {response}"
            )

    def reset(self) -> None:
        self._connection.send(("reset",))
        response = self._connection.recv()
        self._check_response(response, "reset")
        self.last_candidate_count = 0
        self.last_association_score = 0.0

    def __call__(self, rgb: np.ndarray, depth: np.ndarray) -> TargetObservation:
        self._connection.send(("infer", np.asarray(rgb), np.asarray(depth)))
        response = self._connection.recv()
        self._check_response(response, "result")
        self.last_candidate_count = int(response["candidate_count"])
        self.last_association_score = float(response["association_score"])
        return response["target"]

    def close(self) -> None:
        if getattr(self, "_connection", None) is None:
            return
        try:
            self._connection.send(("close",))
        except (EOFError, BrokenPipeError):
            pass
        self._connection.close()
        self._connection = None
        self._process.join(timeout=5.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)

    def __del__(self):
        self.close()
