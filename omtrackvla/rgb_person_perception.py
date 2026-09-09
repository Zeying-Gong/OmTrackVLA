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

from .oracle_modular_follow import TargetObservation, bbox_to_footpoint
from .models.candidate_fusion import CandidateFusionModel


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_WEIGHTS = REPO_ROOT / (
    "models/torchvision/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth"
)
DEFAULT_REID_WEIGHTS = (
    REPO_ROOT / "models/reid/osnet_x0_25_msmt17.pt"
)
DEFAULT_REID_CODE = (
    REPO_ROOT / "third_party/torchreid/torchreid/reid/models/osnet.py"
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


class TargetAppearanceMemory:
    """Immutable target anchor plus cautiously updated appearance galleries."""

    def __init__(
        self,
        anchor_embedding: Optional[np.ndarray] = None,
        max_positive_embeddings: int = 8,
        max_negative_embeddings: int = 16,
        duplicate_cosine: float = 0.995,
    ) -> None:
        if max_positive_embeddings < 1 or max_negative_embeddings < 0:
            raise ValueError("appearance gallery capacities are invalid")
        self.max_positive_embeddings = int(max_positive_embeddings)
        self.max_negative_embeddings = int(max_negative_embeddings)
        self.duplicate_cosine = float(duplicate_cosine)
        self.reset(anchor_embedding)

    @staticmethod
    def _normalized(embedding: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if embedding is None:
            return None
        value = np.asarray(embedding, dtype=np.float32).reshape(-1).copy()
        norm = float(np.linalg.norm(value))
        if not np.isfinite(norm) or norm <= 1e-6:
            return None
        value /= norm
        return value

    @staticmethod
    def _cosine01(left: Optional[np.ndarray], right: Optional[np.ndarray]) -> float:
        if left is None or right is None:
            return 0.0
        return 0.5 * (float(np.clip(np.dot(left, right), -1.0, 1.0)) + 1.0)

    def reset(self, anchor_embedding: Optional[np.ndarray]) -> None:
        self.anchor_embedding = self._normalized(anchor_embedding)
        self.positive_embeddings: list[np.ndarray] = []
        self.negative_embeddings: list[np.ndarray] = []

    def scores(self, embedding: Optional[np.ndarray]) -> dict[str, float]:
        value = self._normalized(embedding)
        anchor = self._cosine01(self.anchor_embedding, value)
        gallery = max(
            (self._cosine01(item, value) for item in self.positive_embeddings),
            default=anchor,
        )
        positive = max(anchor, 0.35 * anchor + 0.65 * gallery)
        negative = max(
            (self._cosine01(item, value) for item in self.negative_embeddings),
            default=0.0,
        )
        # A known distractor may look somewhat like the target.  Penalize it
        # mildly while retaining the immutable anchor as the identity root.
        identity = float(np.clip(positive - 0.20 * max(0.0, negative - 0.5), 0.0, 1.0))
        return {
            "anchor": anchor,
            "gallery": gallery,
            "positive": positive,
            "negative": negative,
            "identity": identity,
        }

    def _append_diverse(self, collection: list[np.ndarray], embedding: np.ndarray, capacity: int) -> bool:
        value = self._normalized(embedding)
        if value is None or capacity <= 0:
            return False
        if any(float(np.dot(item, value)) >= self.duplicate_cosine for item in collection):
            return False
        collection.append(value)
        if len(collection) > capacity:
            del collection[0]
        return True

    def add_positive(self, embedding: Optional[np.ndarray]) -> bool:
        if embedding is None:
            return False
        return self._append_diverse(
            self.positive_embeddings,
            embedding,
            self.max_positive_embeddings,
        )

    def add_negative(self, embedding: Optional[np.ndarray]) -> bool:
        if embedding is None:
            return False
        return self._append_diverse(
            self.negative_embeddings,
            embedding,
            self.max_negative_embeddings,
        )


class RGBPersonPerception:
    """COCO person detector with first-visible initialization and RGB tracking."""

    name = "rgb-person-pretrained-detector-osnet-v7"

    def __init__(
        self,
        weights_path: Path | str = DEFAULT_WEIGHTS,
        detector_architecture: str = "fasterrcnn_mobilenet_v3_large_320_fpn",
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
        memory_update_detector_threshold: float = 0.70,
        memory_update_identity_threshold: float = 0.82,
        memory_update_anchor_threshold: float = 0.72,
        memory_update_association_threshold: float = 0.65,
        memory_update_margin: float = 0.04,
        memory_update_min_confirmed_steps: int = 2,
        memory_max_positive_embeddings: int = 8,
        memory_max_negative_embeddings: int = 16,
        detector_min_size: Optional[int] = None,
        detector_max_size: Optional[int] = None,
        fusion_weights_path: Optional[Path | str] = None,
        device: str = "cuda",
        max_depth_m: float = 10.0,
        hfov_deg: float = 90.0,
    ) -> None:
        import torch
        from torchvision.models.detection import (
            fasterrcnn_mobilenet_v3_large_320_fpn,
            fasterrcnn_resnet50_fpn_v2,
        )

        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(f"Person detector weights not found: {weights_path}")
        self.device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        detector_builders = {
            "fasterrcnn_mobilenet_v3_large_320_fpn": (
                fasterrcnn_mobilenet_v3_large_320_fpn
            ),
            "fasterrcnn_resnet50_fpn_v2": fasterrcnn_resnet50_fpn_v2,
        }
        if detector_architecture not in detector_builders:
            raise ValueError(f"unsupported person detector: {detector_architecture}")
        self.detector_architecture = str(detector_architecture)
        self.model = detector_builders[self.detector_architecture](
            weights=None, weights_backbone=None
        )
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        if detector_min_size is not None:
            self.model.transform.min_size = (int(detector_min_size),)
        if detector_max_size is not None:
            self.model.transform.max_size = int(detector_max_size)
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
        self.memory_update_detector_threshold = float(memory_update_detector_threshold)
        self.memory_update_identity_threshold = float(memory_update_identity_threshold)
        self.memory_update_anchor_threshold = float(memory_update_anchor_threshold)
        self.memory_update_association_threshold = float(memory_update_association_threshold)
        self.memory_update_margin = float(memory_update_margin)
        self.memory_update_min_confirmed_steps = int(memory_update_min_confirmed_steps)
        self.memory_max_positive_embeddings = int(memory_max_positive_embeddings)
        self.memory_max_negative_embeddings = int(memory_max_negative_embeddings)
        self.fusion_model = (
            CandidateFusionModel.load(fusion_weights_path)
            if fusion_weights_path is not None
            else None
        )
        self.detector_min_size = int(self.model.transform.min_size[0])
        self.detector_max_size = int(self.model.transform.max_size)
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
        self._appearance_memory = TargetAppearanceMemory(
            self._goal_embedding,
            max_positive_embeddings=getattr(self, "memory_max_positive_embeddings", 8),
            max_negative_embeddings=getattr(self, "memory_max_negative_embeddings", 16),
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
        self.last_anchor_similarity = 0.0
        self.last_identity_margin = 0.0
        self.last_memory_updated = False
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

    def _appearance_scores(self, embedding) -> dict[str, float]:
        memory = getattr(self, "_appearance_memory", None)
        if memory is not None:
            return memory.scores(embedding)
        anchor = 0.5 * (self._cosine(getattr(self, "_goal_embedding", None), embedding) + 1.0)
        return {
            "anchor": anchor,
            "gallery": anchor,
            "positive": anchor,
            "negative": 0.0,
            "identity": anchor,
        }

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
        self._last_candidate_features = features
        self.last_identity_margin = 0.0
        self.last_candidate_diagnostics = []
        recent_goal = list(getattr(self, "_recent_goal_similarities", ()))
        recent_goal_mean = float(np.mean(recent_goal)) if recent_goal else 0.0
        recent_goal_min = float(np.min(recent_goal)) if recent_goal else 0.0
        temporal_context = {
            "candidate_count": len(features),
            "confirmed_track_steps": int(
                getattr(self, "_confirmed_track_steps", 0)
            ),
            "previous_association_score": float(
                getattr(self, "last_association_score", 0.0)
            ),
            "recent_goal_mean": recent_goal_mean,
            "recent_goal_min": recent_goal_min,
        }
        for candidate_index, (box, detector_score, hist, embedding) in enumerate(features):
            appearance_scores = self._appearance_scores(embedding)
            goal_histogram = 0.0
            if self._reference_hist is not None and hist is not None:
                goal_histogram = float(cv2.compareHist(
                    self._reference_hist, hist, cv2.HISTCMP_INTERSECT
                ))
            self.last_candidate_diagnostics.append({
                "bbox_xyxy": [float(value) for value in box],
                "detector_score": float(detector_score),
                "goal_reid": appearance_scores["identity"],
                "anchor_reid": appearance_scores["anchor"],
                "gallery_reid": appearance_scores["gallery"],
                "negative_reid": appearance_scores["negative"],
                "goal_histogram": goal_histogram,
                "track_reid": 0.0,
                "track_histogram": 0.0,
                "spatial_overlap": 0.0,
                "center_score": 0.0,
                "scale_score": 0.0,
                "association_score": 0.0,
                "identity_score": 0.0,
                "missed_steps": int(getattr(self, "_missed_steps", 0)),
                "global_search": float(self._bbox is None),
                "fusion_score": None,
                "selected": False,
                **temporal_context,
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
                    reid_score = self._appearance_scores(embedding)["identity"]
                    return (
                        0.70 * reid_score
                        + 0.20 * histogram_score
                        + 0.10 * detector_score
                    )

                missed_steps = int(getattr(self, "_missed_steps", 0))
                ranked = []
                for candidate_index, item in enumerate(features):
                    box, score, hist, embedding = item
                    appearance_scores = self._appearance_scores(embedding)
                    goal_reid = appearance_scores["identity"]
                    goal_histogram = (
                        float(cv2.compareHist(
                            self._reference_hist, hist, cv2.HISTCMP_INTERSECT
                        ))
                        if self._reference_hist is not None and hist is not None
                        else 0.0
                    )
                    track_reid = 0.5 * (
                        self._cosine(
                            getattr(self, "_track_embedding", None), embedding
                        )
                        + 1.0
                    )
                    track_histogram = (
                        float(
                            cv2.compareHist(
                                self._track_hist, hist, cv2.HISTCMP_INTERSECT
                            )
                        )
                        if getattr(self, "_track_hist", None) is not None
                        and hist is not None
                        else 0.0
                    )
                    identity_score = 0.55 * goal_reid + 0.45 * goal_histogram
                    match_score = goal_match(item)
                    self.last_candidate_diagnostics[candidate_index].update(
                        association_score=float(match_score),
                        identity_score=float(identity_score),
                        track_reid=float(track_reid),
                        track_histogram=float(track_histogram),
                    )
                    fusion_model = getattr(self, "fusion_model", None)
                    fusion_score = (
                        fusion_model.predict(
                            self.last_candidate_diagnostics[candidate_index]
                        )
                        if fusion_model is not None
                        else None
                    )
                    self.last_candidate_diagnostics[candidate_index][
                        "fusion_score"
                    ] = fusion_score
                    identity_threshold = float(getattr(
                        self,
                        "global_single_identity_threshold"
                        if len(features) == 1 else "global_identity_threshold",
                        0.72 if len(features) == 1 else 0.67,
                    ))
                    if (
                        fusion_model is None
                        and self._goal_embedding is not None
                        and identity_score < identity_threshold
                    ):
                        continue
                    selection_score = (
                        float(fusion_score)
                        if fusion_score is not None
                        else identity_score
                        if missed_steps > 0
                        else match_score
                    )
                    ranked.append(
                        (
                            selection_score, identity_score, goal_reid,
                            appearance_scores["anchor"],
                            box, score, hist, embedding, candidate_index,
                        )
                    )
                if not ranked:
                    return None
                ranked.sort(key=lambda item: item[0], reverse=True)
                (
                    selection_score, identity_score, goal_reid, anchor_reid,
                    box, score, hist, embedding, candidate_index,
                ) = ranked[0]
                self.last_identity_margin = (
                    float(selection_score - ranked[1][0]) if len(ranked) > 1 else 1.0
                )
                if fusion_model is not None and not fusion_model.accepts(
                    selection_score,
                    self.last_identity_margin,
                    global_search=True,
                ):
                    return None
                if (
                    fusion_model is None
                    and missed_steps > 0
                    and len(ranked) > 1
                    and identity_score - ranked[1][1]
                    < float(getattr(self, "global_identity_margin", 0.01))
                ):
                    return None
                return (
                    box,
                    score,
                    hist,
                    embedding,
                    selection_score,
                    goal_reid,
                    anchor_reid,
                )
            box, score, hist, embedding = max(
                features,
                key=lambda item: self._initial_score(item[0], item[1], width, height),
            )
            self.last_identity_margin = 1.0
            return box, score, hist, embedding, score, 1.0, 1.0

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
        for candidate_index, (box, detector_score, hist, embedding) in enumerate(
            features
        ):
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
            appearance_scores = self._appearance_scores(embedding)
            goal_reid = appearance_scores["identity"]
            missed_steps = int(getattr(self, "_missed_steps", 0))
            fusion_model = getattr(self, "fusion_model", None)
            if (
                fusion_model is None
                and missed_steps > 0
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
            if fusion_model is None and scale_anomaly and (
                overlap < 0.15 or goal_reid < max(self.reid_threshold, 0.85)
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
            self.last_candidate_diagnostics[candidate_index].update(
                track_reid=float(track_reid),
                track_histogram=float(track_histogram_score),
                spatial_overlap=float(overlap),
                center_score=float(center_score),
                scale_score=float(scale_score),
                association_score=float(association),
                identity_score=float(goal_reid),
            )
            fusion_score = (
                fusion_model.predict(self.last_candidate_diagnostics[candidate_index])
                if fusion_model is not None
                else None
            )
            self.last_candidate_diagnostics[candidate_index][
                "fusion_score"
            ] = fusion_score
            selection_score = (
                float(fusion_score) if fusion_score is not None else association
            )
            ranked.append(
                (
                    selection_score,
                    box,
                    detector_score,
                    hist,
                    embedding,
                    goal_reid,
                    appearance_scores["anchor"],
                    candidate_index,
                )
            )
        if not ranked:
            return None
        ranked.sort(key=lambda item: item[0], reverse=True)
        (
            selection_score,
            box,
            score,
            hist,
            embedding,
            goal_reid,
            anchor_reid,
            candidate_index,
        ) = ranked[0]
        self.last_identity_margin = (
            float(selection_score - ranked[1][0]) if len(ranked) > 1 else 1.0
        )
        if fusion_model is not None:
            if not fusion_model.accepts(
                selection_score,
                self.last_identity_margin,
                global_search=False,
            ):
                return None
        else:
            if selection_score < self.association_threshold:
                return None
            if self._goal_embedding is not None and goal_reid < self.reid_threshold:
                return None
            if (
                len(ranked) > 1
                and selection_score - ranked[1][0] < self.ambiguity_margin
            ):
                return None
        return (
            box,
            score,
            hist,
            embedding,
            selection_score,
            goal_reid,
            anchor_reid,
        )

    def _update_appearance_memory(
        self,
        selected_box: np.ndarray,
        detector_score: float,
        hist: Optional[np.ndarray],
        embedding: Optional[np.ndarray],
        association: float,
        identity_similarity: float,
        anchor_similarity: float,
        previous_bbox: Optional[np.ndarray],
        missed_before_selection: int,
    ) -> bool:
        """Update adaptive appearance only after a strong continuous match."""

        if embedding is None:
            return False
        memory = getattr(self, "_appearance_memory", None)
        if memory is None:
            memory = TargetAppearanceMemory(
                getattr(self, "_goal_embedding", None),
                max_positive_embeddings=int(
                    getattr(self, "memory_max_positive_embeddings", 8)
                ),
                max_negative_embeddings=int(
                    getattr(self, "memory_max_negative_embeddings", 16)
                ),
            )
            self._appearance_memory = memory
        if memory.anchor_embedding is None:
            normalized = TargetAppearanceMemory._normalized(embedding)
            self._goal_embedding = normalized
            memory.reset(normalized)
            self._track_embedding = normalized
            if hist is not None:
                self._reference_hist = hist.copy()
                self._track_hist = hist.copy()
            return True

        allowed = (
            previous_bbox is not None
            and missed_before_selection == 0
            and detector_score >= float(
                getattr(self, "memory_update_detector_threshold", 0.70)
            )
            and identity_similarity >= float(
                getattr(self, "memory_update_identity_threshold", 0.82)
            )
            and anchor_similarity >= float(
                getattr(self, "memory_update_anchor_threshold", 0.72)
            )
            and association >= float(
                getattr(self, "memory_update_association_threshold", 0.65)
            )
            and float(getattr(self, "last_identity_margin", 0.0))
            >= float(getattr(self, "memory_update_margin", 0.04))
            and int(getattr(self, "_confirmed_track_steps", 0))
            >= int(getattr(self, "memory_update_min_confirmed_steps", 2))
        )
        if not allowed:
            return False

        memory.add_positive(embedding)
        if hist is not None:
            self._track_hist = (
                hist.copy()
                if getattr(self, "_track_hist", None) is None
                else 0.9 * self._track_hist + 0.1 * hist
            )
        self._track_embedding = (
            np.asarray(embedding, dtype=np.float32).copy()
            if getattr(self, "_track_embedding", None) is None
            else 0.95 * self._track_embedding + 0.05 * embedding
        )
        norm = float(np.linalg.norm(self._track_embedding))
        if norm > 1e-6:
            self._track_embedding /= norm

        # Only a trusted winner may turn spatially separate, low-identity
        # detections into persistent distractor evidence.
        for box, _, _, candidate_embedding in getattr(
            self, "_last_candidate_features", []
        ):
            if candidate_embedding is None or bbox_iou(selected_box, box) >= 0.10:
                continue
            if memory.scores(candidate_embedding)["identity"] <= 0.58:
                memory.add_negative(candidate_embedding)
        return True

    def __call__(self, rgb: np.ndarray, depth: np.ndarray) -> TargetObservation:
        self.last_candidate_diagnostics = []
        self.last_memory_updated = False
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

        (
            box,
            detector_score,
            hist,
            embedding,
            association,
            goal_similarity,
            anchor_similarity,
        ) = selected
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
        if previous_bbox is None or missed_before_selection > 0:
            self._recent_goal_similarities = [float(goal_similarity)]
            self._confirmed_track_steps = 1
        else:
            self._recent_goal_similarities.append(float(goal_similarity))
            self._recent_goal_similarities = self._recent_goal_similarities[-5:]
            self._confirmed_track_steps += 1
        self.last_memory_updated = self._update_appearance_memory(
            box,
            float(detector_score),
            hist,
            embedding,
            float(association),
            float(goal_similarity),
            float(anchor_similarity),
            previous_bbox,
            missed_before_selection,
        )
        self.last_association_score = float(association)
        self.last_goal_similarity = float(goal_similarity)
        self.last_anchor_similarity = float(anchor_similarity)
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
                "anchor_similarity": perception.last_anchor_similarity,
                "identity_margin": perception.last_identity_margin,
                "memory_updated": perception.last_memory_updated,
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
        self.last_anchor_similarity = 0.0
        self.last_identity_margin = 0.0
        self.last_memory_updated = False
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
        self.last_anchor_similarity = 0.0
        self.last_identity_margin = 0.0
        self.last_memory_updated = False
        self.last_candidate_diagnostics = []

    def __call__(self, rgb: np.ndarray, depth: np.ndarray) -> TargetObservation:
        self._connection.send(("infer", np.asarray(rgb), np.asarray(depth)))
        response = self._connection.recv()
        self._check_response(response, "result")
        self.last_candidate_count = int(response["candidate_count"])
        self.last_association_score = float(response["association_score"])
        self.last_goal_similarity = float(response["goal_similarity"])
        self.last_anchor_similarity = float(response["anchor_similarity"])
        self.last_identity_margin = float(response["identity_margin"])
        self.last_memory_updated = bool(response["memory_updated"])
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
