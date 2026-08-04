#!/usr/bin/env python3
"""RGB person detection and short-term association for the oracle controller."""

from __future__ import annotations

import math
import multiprocessing as mp
import importlib.util
from pathlib import Path
import traceback
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from oracle_modular_follow import TargetObservation, bbox_to_footpoint


DEFAULT_WEIGHTS = Path(__file__).resolve().parent / (
    "models/torchvision/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth"
)
DEFAULT_REID_WEIGHTS = (
    Path(__file__).resolve().parent / "models/reid/osnet_x0_25_msmt17.pt"
)
DEFAULT_REID_CODE = (
    Path(__file__).resolve().parent
    / "third_party/torchreid/torchreid/reid/models/osnet.py"
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
    raw_depth: Optional[np.ndarray],
    image_width: int = 640,
    hfov_deg: float = 90.0,
    max_depth_m: float = 10.0,
) -> Optional[Tuple[float, float]]:
    if raw_depth is None:
        x1, y1, x2, y2 = map(float, bbox)
        center_x = 0.5 * (x1 + x2)
        focal = image_width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
        left = -((center_x - (image_width - 1.0) / 2.0) * max_depth_m / focal)
        return max_depth_m, float(left)
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

    name = "rgb-person-mobilenet-osnet-v6"

    def __init__(
        self,
        weights_path: Path | str = DEFAULT_WEIGHTS,
        reid_weights_path: Path | str = DEFAULT_REID_WEIGHTS,
        reid_code_path: Path | str = DEFAULT_REID_CODE,
        score_threshold: float = 0.30,
        association_threshold: float = 0.20,
        reid_threshold: float = 0.55,
        ambiguity_margin: float = 0.04,
        spatial_reset_after_misses: int = 4,
        global_identity_threshold: float = 0.67,
        global_single_identity_threshold: float = 0.72,
        global_identity_margin: float = 0.01,
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
        reid_weights_path = Path(reid_weights_path)
        reid_code_path = Path(reid_code_path)
        if not reid_weights_path.is_file():
            raise FileNotFoundError(f"ReID feature weights not found: {reid_weights_path}")
        if not reid_code_path.is_file():
            raise FileNotFoundError(f"OSNet model code not found: {reid_code_path}")
        spec = importlib.util.spec_from_file_location("omtrackvla_osnet", reid_code_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load OSNet module: {reid_code_path}")
        osnet_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(osnet_module)
        reid_state = torch.load(reid_weights_path, map_location="cpu", weights_only=True)
        num_classes = int(reid_state["classifier.weight"].shape[0])
        self.reid_model = osnet_module.osnet_x0_25(
            num_classes=num_classes, pretrained=False
        )
        self.reid_model.load_state_dict(reid_state, strict=True)
        self.reid_model.eval().to(self.device)
        self.score_threshold = float(score_threshold)
        self.association_threshold = float(association_threshold)
        self.reid_threshold = float(reid_threshold)
        self.ambiguity_margin = float(ambiguity_margin)
        self.spatial_reset_after_misses = int(spatial_reset_after_misses)
        self.global_identity_threshold = float(global_identity_threshold)
        self.global_single_identity_threshold = float(
            global_single_identity_threshold
        )
        self.global_identity_margin = float(global_identity_margin)
        self.max_depth_m = float(max_depth_m)
        self.hfov_deg = float(hfov_deg)
        self.reset()

    def reset(self, reference_rgb: Optional[np.ndarray] = None) -> None:
        self._bbox = None
        self._reference_hist = (
            color_histogram(
                reference_rgb,
                (0, 0, reference_rgb.shape[1], reference_rgb.shape[0]),
            )
            if reference_rgb is not None and np.asarray(reference_rgb).size else None
        )
        self._goal_embedding = (
            self._embed_crops([np.asarray(reference_rgb)[..., :3]])[0]
            if reference_rgb is not None and np.asarray(reference_rgb).size else None
        )
        self._track_embedding = self._goal_embedding
        self._track_hist = (
            self._reference_hist.copy() if self._reference_hist is not None else None
        )
        self._bbox_velocity = np.zeros(4, dtype=np.float32)
        self._missed_steps = 0
        self._recent_goal_similarities = []
        self._confirmed_track_steps = 0
        self._last_relative = None
        self.last_candidate_count = 0
        self.last_association_score = 0.0
        self.last_goal_similarity = 0.0
        self.last_candidate_diagnostics = []

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

    def _embed_crops(self, crops):
        if not crops or getattr(self, "reid_model", None) is None:
            return [None] * len(crops)
        import torch

        tensors = []
        for crop in crops:
            resized = cv2.resize(
                np.asarray(crop)[..., :3], (128, 256), interpolation=cv2.INTER_LINEAR
            )
            tensor = torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1)
            tensors.append(tensor.float().div_(255.0))
        inputs = torch.stack(tensors).to(self.device)
        mean = torch.tensor((0.485, 0.456, 0.406), device=self.device).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=self.device).view(1, 3, 1, 1)
        inputs = (inputs - mean) / std
        with torch.inference_mode():
            embedding = self.reid_model(inputs).float()
        embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return [value for value in embedding.cpu().numpy()]

    @staticmethod
    def _crop(rgb, box):
        image = np.asarray(rgb)[..., :3]
        height, width = image.shape[:2]
        x1, y1, x2, y2 = map(float, box)
        x1 = int(np.clip(math.floor(x1), 0, width - 1))
        x2 = int(np.clip(math.ceil(x2), x1 + 1, width))
        y1 = int(np.clip(math.floor(y1), 0, height - 1))
        y2 = int(np.clip(math.ceil(y2), y1 + 1, height))
        return image[y1:y2, x1:x2]

    @staticmethod
    def _cosine(a, b):
        if a is None or b is None:
            return 0.0
        return float(np.clip(np.dot(a, b), -1.0, 1.0))

    @staticmethod
    def _initial_score(box, detector_score, width, height):
        x1, y1, x2, y2 = box
        area_fraction = max(0.0, (x2 - x1) * (y2 - y1)) / (width * height)
        cx = 0.5 * (x1 + x2) / width
        center_penalty = abs(cx - 0.5)
        return detector_score + 0.35 * math.sqrt(area_fraction) - 0.25 * center_penalty

    def _select(self, rgb: np.ndarray, candidates):
        height, width = np.asarray(rgb).shape[:2]
        embeddings = self._embed_crops([
            self._crop(rgb, box) for box, _ in candidates
        ])
        features = [
            (box, score, color_histogram(rgb, box), embedding)
            for (box, score), embedding in zip(candidates, embeddings)
        ]
        self.last_candidate_diagnostics = []
        for box, detector_score, hist, embedding in features:
            goal_histogram = 0.0
            if self._reference_hist is not None and hist is not None:
                goal_histogram = float(cv2.compareHist(
                    self._reference_hist, hist, cv2.HISTCMP_INTERSECT
                ))
            self.last_candidate_diagnostics.append({
                "bbox_xyxy": [float(value) for value in box],
                "detector_score": float(detector_score),
                "goal_reid": 0.5 * (
                    self._cosine(self._goal_embedding, embedding) + 1.0
                ),
                "goal_histogram": goal_histogram,
                "selected": False,
            })
        if self._bbox is None:
            if self._reference_hist is not None or self._goal_embedding is not None:
                def goal_match(item):
                    box, detector_score, hist, embedding = item
                    histogram_score = (
                        float(cv2.compareHist(
                            self._reference_hist, hist, cv2.HISTCMP_INTERSECT
                        ))
                        if self._reference_hist is not None and hist is not None else 0.0
                    )
                    reid_score = 0.5 * (
                        self._cosine(self._goal_embedding, embedding) + 1.0
                    )
                    return (
                        0.70 * reid_score
                        + 0.20 * histogram_score
                        + 0.10 * detector_score
                    )

                missed_steps = int(getattr(self, "_missed_steps", 0))
                ranked = []
                for item in features:
                    box, score, hist, embedding = item
                    goal_reid = 0.5 * (
                        self._cosine(self._goal_embedding, embedding) + 1.0
                    )
                    goal_histogram = (
                        float(cv2.compareHist(
                            self._reference_hist, hist, cv2.HISTCMP_INTERSECT
                        ))
                        if self._reference_hist is not None and hist is not None
                        else 0.0
                    )
                    identity_score = 0.55 * goal_reid + 0.45 * goal_histogram
                    identity_threshold = float(getattr(
                        self,
                        "global_single_identity_threshold"
                        if len(features) == 1 else "global_identity_threshold",
                        0.72 if len(features) == 1 else 0.67,
                    ))
                    if (
                        missed_steps > 0
                        and self._goal_embedding is not None
                        and identity_score < identity_threshold
                    ):
                        continue
                    ranked.append(
                        (
                            goal_match(item), identity_score, goal_reid,
                            box, score, hist, embedding,
                        )
                    )
                if not ranked:
                    return None
                ranked.sort(
                    key=lambda item: item[1] if missed_steps > 0 else item[0],
                    reverse=True,
                )
                (
                    match_score, identity_score, goal_reid,
                    box, score, hist, embedding,
                ) = ranked[0]
                if (
                    missed_steps > 0
                    and len(ranked) > 1
                    and identity_score - ranked[1][1]
                    < float(getattr(self, "global_identity_margin", 0.01))
                ):
                    return None
                return box, score, hist, embedding, match_score, goal_reid
            box, score, hist, embedding = max(
                features,
                key=lambda item: self._initial_score(item[0], item[1], width, height),
            )
            return box, score, hist, embedding, score

        predicted_bbox = self._bbox + self._bbox_velocity
        predicted_center = np.array(
            [
                0.5 * (predicted_bbox[0] + predicted_bbox[2]),
                0.5 * (predicted_bbox[1] + predicted_bbox[3]),
            ]
        )
        diagonal = math.hypot(width, height)
        predicted_area = max(
            1.0,
            float(
                (predicted_bbox[2] - predicted_bbox[0])
                * (predicted_bbox[3] - predicted_bbox[1])
            ),
        )
        ranked = []
        for box, detector_score, hist, embedding in features:
            center = np.array([0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])])
            center_score = math.exp(
                -4.0 * float(np.linalg.norm(center - predicted_center)) / diagonal
            )
            overlap = bbox_iou(predicted_bbox, box)
            candidate_area = max(
                1.0, float((box[2] - box[0]) * (box[3] - box[1]))
            )
            area_ratio = candidate_area / predicted_area
            scale_score = math.exp(-abs(math.log(area_ratio)))
            goal_reid = 0.5 * (
                self._cosine(self._goal_embedding, embedding) + 1.0
            )
            missed_steps = int(getattr(self, "_missed_steps", 0))
            if (
                missed_steps > 0
                and self._goal_embedding is not None
                and goal_reid < 0.81
            ):
                continue

            # A close distractor often produces one very large box while fully
            # occluding the tracked person. Rapid scale growth alone is not
            # sufficient evidence, though: the tracked person produces the same
            # geometry while approaching the camera.
            scale_anomaly = area_ratio > 2.5 or (
                overlap < 0.05 and area_ratio < 0.4
            )
            if scale_anomaly and (
                overlap < 0.15
                or goal_reid < max(self.reid_threshold, 0.85)
            ):
                continue

            goal_histogram_score = 0.0
            if self._reference_hist is not None and hist is not None:
                goal_histogram_score = float(cv2.compareHist(
                    self._reference_hist, hist, cv2.HISTCMP_INTERSECT
                ))
            track_histogram_score = 0.0
            track_hist = getattr(self, "_track_hist", None)
            if track_hist is not None and hist is not None:
                track_histogram_score = float(cv2.compareHist(
                    track_hist, hist, cv2.HISTCMP_INTERSECT
                ))
            track_reid = 0.5 * (
                self._cosine(self._track_embedding, embedding) + 1.0
            )
            appearance = (
                0.60 * goal_reid
                + 0.20 * track_reid
                + 0.15 * goal_histogram_score
                + 0.05 * track_histogram_score
            )
            motion_weight = 0.25 if missed_steps == 0 else max(
                0.05, 0.25 / (missed_steps + 1)
            )
            association = (
                motion_weight
                * (0.55 * overlap + 0.30 * center_score + 0.15 * scale_score)
                + (0.95 - motion_weight) * appearance
                + 0.05 * detector_score
            )
            ranked.append(
                (association, box, detector_score, hist, embedding, goal_reid)
            )
        if not ranked:
            return None
        ranked.sort(key=lambda item: item[0], reverse=True)
        association, box, score, hist, embedding, goal_reid = ranked[0]
        if association < self.association_threshold:
            return None
        if self._goal_embedding is not None and goal_reid < self.reid_threshold:
            return None
        if len(ranked) > 1 and association - ranked[1][0] < self.ambiguity_margin:
            return None
        return box, score, hist, embedding, association, goal_reid

    def __call__(self, rgb: np.ndarray, depth: np.ndarray) -> TargetObservation:
        self.last_candidate_diagnostics = []
        candidates = self._detect(rgb)
        self.last_candidate_count = len(candidates)
        selected = self._select(rgb, candidates) if candidates else None
        if selected is None:
            self.last_association_score = 0.0
            self.last_goal_similarity = 0.0
            if self._bbox is not None:
                self._bbox = self._bbox + self._bbox_velocity
                self._bbox_velocity *= 0.8
                self._missed_steps += 1
                if self._missed_steps >= self.spatial_reset_after_misses:
                    self._bbox = None
                    self._bbox_velocity.fill(0.0)
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

        if len(selected) == 5:
            box, detector_score, hist, embedding, association = selected
            goal_similarity = 1.0
        else:
            box, detector_score, hist, embedding, association, goal_similarity = selected
        for candidate in self.last_candidate_diagnostics:
            candidate["selected"] = bbox_iou(candidate["bbox_xyxy"], box) >= 0.95
        image_width = int(np.asarray(rgb).shape[1])
        relative = bbox_depth_to_relative(
            box,
            depth,
            image_width=image_width,
            hfov_deg=self.hfov_deg,
            max_depth_m=self.max_depth_m,
        )
        if relative is None:
            relative = self._last_relative
        if relative is None:
            relative = (self.max_depth_m, 0.0)

        previous_bbox = self._bbox
        missed_before_selection = self._missed_steps
        if previous_bbox is not None:
            observed_velocity = box - self._bbox
            self._bbox_velocity = (
                0.5 * self._bbox_velocity + 0.5 * observed_velocity
            ).astype(np.float32)
        self._bbox = box.copy()
        self._missed_steps = 0
        self._last_relative = relative
        if hist is not None:
            if self._reference_hist is None:
                self._reference_hist = hist.copy()
            self._track_hist = (
                hist.copy() if self._track_hist is None
                else 0.9 * self._track_hist + 0.1 * hist
            )
        if embedding is not None:
            if self._goal_embedding is None:
                self._goal_embedding = embedding.copy()
            self._track_embedding = (
                embedding.copy() if self._track_embedding is None
                else 0.95 * self._track_embedding + 0.05 * embedding
            )
            norm = float(np.linalg.norm(self._track_embedding))
            if norm > 1e-6:
                self._track_embedding /= norm
        self.last_association_score = float(association)
        self.last_goal_similarity = float(goal_similarity)
        if previous_bbox is None or missed_before_selection > 0:
            self._recent_goal_similarities = [float(goal_similarity)]
            self._confirmed_track_steps = 1
        else:
            self._recent_goal_similarities.append(float(goal_similarity))
            self._recent_goal_similarities = self._recent_goal_similarities[-5:]
            self._confirmed_track_steps += 1
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
                perception.reset(reference_rgb=request[1])
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
                "goal_similarity": perception.last_goal_similarity,
                "candidate_diagnostics": perception.last_candidate_diagnostics,
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
        self.last_goal_similarity = 0.0
        self.last_candidate_diagnostics = []

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

    def reset(self, reference_rgb: Optional[np.ndarray] = None) -> None:
        self._connection.send(("reset", reference_rgb))
        response = self._connection.recv()
        self._check_response(response, "reset")
        self.last_candidate_count = 0
        self.last_association_score = 0.0
        self.last_goal_similarity = 0.0
        self.last_candidate_diagnostics = []

    def __call__(self, rgb: np.ndarray, depth: np.ndarray) -> TargetObservation:
        self._connection.send(("infer", np.asarray(rgb), np.asarray(depth)))
        response = self._connection.recv()
        self._check_response(response, "result")
        self.last_candidate_count = int(response["candidate_count"])
        self.last_association_score = float(response["association_score"])
        self.last_goal_similarity = float(response["goal_similarity"])
        self.last_candidate_diagnostics = response["candidate_diagnostics"]
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
