"""Small learned fusion head over frozen detector and ReID diagnostics."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


FEATURE_NAMES = (
    "detector_score",
    "anchor_reid",
    "gallery_reid",
    "negative_reid",
    "goal_histogram",
    "track_reid",
    "track_histogram",
    "spatial_overlap",
    "center_score",
    "scale_score",
    "log1p_missed_steps",
    "global_search",
)


def feature_vector(diagnostic: Mapping[str, object]) -> np.ndarray:
    values = dict(diagnostic)
    values["log1p_missed_steps"] = math.log1p(
        max(0.0, float(values.get("missed_steps", 0.0)))
    )
    return np.asarray(
        [float(values.get(name, 0.0)) for name in FEATURE_NAMES],
        dtype=np.float32,
    )


class CandidateFusionModel:
    """Numpy inference for a two-layer candidate classifier saved as JSON."""

    def __init__(self, payload: Mapping[str, object]) -> None:
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("candidate fusion feature contract mismatch")
        self.mean = np.asarray(payload["feature_mean"], dtype=np.float32)
        self.scale = np.asarray(payload["feature_scale"], dtype=np.float32)
        self.weight1 = np.asarray(payload["weight1"], dtype=np.float32)
        self.bias1 = np.asarray(payload["bias1"], dtype=np.float32)
        self.weight2 = np.asarray(payload["weight2"], dtype=np.float32)
        self.bias2 = float(payload["bias2"])
        self.score_threshold = float(payload["score_threshold"])
        self.margin_threshold = float(payload["margin_threshold"])
        raw_operating_points = payload.get("operating_points", {})
        if raw_operating_points is None:
            raw_operating_points = {}
        if not isinstance(raw_operating_points, Mapping):
            raise ValueError("candidate fusion operating points must be a mapping")
        self.operating_points = {}
        for mode in ("tracking", "reacquisition"):
            raw_point = raw_operating_points.get(mode, {})
            if not isinstance(raw_point, Mapping):
                raise ValueError(f"candidate fusion {mode} operating point is invalid")
            score_threshold = float(
                raw_point.get("score_threshold", self.score_threshold)
            )
            margin_threshold = float(
                raw_point.get("margin_threshold", self.margin_threshold)
            )
            if not 0.0 <= score_threshold <= 1.0:
                raise ValueError(f"candidate fusion {mode} score threshold is invalid")
            if not 0.0 <= margin_threshold <= 1.0:
                raise ValueError(f"candidate fusion {mode} margin threshold is invalid")
            self.operating_points[mode] = {
                "score_threshold": score_threshold,
                "margin_threshold": margin_threshold,
            }
        if self.mean.shape != (len(FEATURE_NAMES),) or self.scale.shape != self.mean.shape:
            raise ValueError("candidate fusion normalization shape mismatch")
        if self.weight1.shape[1] != len(FEATURE_NAMES):
            raise ValueError("candidate fusion first-layer shape mismatch")
        if self.bias1.shape != (self.weight1.shape[0],):
            raise ValueError("candidate fusion first-layer bias shape mismatch")
        if self.weight2.shape != (self.weight1.shape[0],):
            raise ValueError("candidate fusion output-layer shape mismatch")
        self.scale = np.where(self.scale > 1e-6, self.scale, 1.0)

    @classmethod
    def load(cls, path: str | Path) -> "CandidateFusionModel":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def predict(self, diagnostic: Mapping[str, object]) -> float:
        features = (feature_vector(diagnostic) - self.mean) / self.scale
        hidden = np.maximum(self.weight1 @ features + self.bias1, 0.0)
        logit = float(np.dot(self.weight2, hidden) + self.bias2)
        if logit >= 0.0:
            return float(1.0 / (1.0 + math.exp(-logit)))
        exponential = math.exp(logit)
        return float(exponential / (1.0 + exponential))

    def predict_many(self, diagnostics: Sequence[Mapping[str, object]]) -> np.ndarray:
        return np.asarray([self.predict(item) for item in diagnostics], dtype=np.float32)

    def thresholds(self, global_search: bool) -> tuple[float, float]:
        mode = "reacquisition" if global_search else "tracking"
        point = self.operating_points[mode]
        return point["score_threshold"], point["margin_threshold"]

    def accepts(self, score: float, margin: float, global_search: bool) -> bool:
        score_threshold, margin_threshold = self.thresholds(global_search)
        return float(score) >= score_threshold and float(margin) >= margin_threshold
