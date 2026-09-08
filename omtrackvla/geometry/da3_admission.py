"""Pure analysis helpers for multi-scene DA3 pseudo-motion admission.

The model runner lives in :mod:`scripts.audit_da3_multiscene`.  This module
keeps scale fitting, clip scoring, and confidence-threshold selection free of
GPU/runtime dependencies so the admission policy can be unit tested.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


DEFAULT_QUALITY_LIMITS = {
    "translation_error_m_mean": 0.12,
    "yaw_error_rad_mean": 0.12,
    "scale_relative_error_median": 0.35,
}


def percentile(values: Sequence[float], quantile: float) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.quantile(finite, quantile)) if finite else None


def distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(finite),
        "min": min(finite) if finite else None,
        "p10": percentile(finite, 0.10),
        "median": percentile(finite, 0.50),
        "mean": float(np.mean(finite)) if finite else None,
        "p90": percentile(finite, 0.90),
        "max": max(finite) if finite else None,
    }


def _poses(record: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(record["predicted_se2_unscaled"], dtype=np.float64)
    reference = np.asarray(record["ground_truth_se2"], dtype=np.float64)
    if (
        predicted.shape != reference.shape
        or predicted.ndim != 2
        or predicted.shape[0] < 2
        or predicted.shape[1] != 3
    ):
        raise ValueError("predicted and ground-truth SE(2) must have matching [N,3] shapes")
    if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        raise ValueError("SE(2) trajectories contain non-finite values")
    return predicted, reference


def fit_global_translation_scale(
    records: Sequence[Mapping[str, object]],
    *,
    minimum_reference_motion_m: float = 0.05,
    minimum_predicted_motion: float = 0.01,
    minimum_clips: int = 8,
) -> dict[str, object]:
    """Fit one scale on calibration clips, balancing each clip equally."""

    per_clip = []
    for record in records:
        try:
            predicted, reference = _poses(record)
        except (KeyError, TypeError, ValueError):
            continue
        predicted_norm = np.linalg.norm(predicted[1:, :2], axis=1)
        reference_norm = np.linalg.norm(reference[1:, :2], axis=1)
        valid = (reference_norm >= minimum_reference_motion_m) & (
            predicted_norm >= minimum_predicted_motion
        )
        if not np.any(valid):
            continue
        ratios = reference_norm[valid] / predicted_norm[valid]
        per_clip.append(
            {
                "clip_id": str(record.get("clip_id", len(per_clip))),
                "scale": float(np.median(ratios)),
                "moving_pairs": int(valid.sum()),
            }
        )
    if len(per_clip) < int(minimum_clips):
        raise ValueError(
            f"scale calibration needs at least {minimum_clips} moving clips; got {len(per_clip)}"
        )
    scales = [float(item["scale"]) for item in per_clip]
    median = float(np.median(scales))
    q25 = float(np.quantile(scales, 0.25))
    q75 = float(np.quantile(scales, 0.75))
    return {
        "scale": median,
        "method": "median of per-clip median GT/predicted displacement ratios",
        "minimum_reference_motion_m": float(minimum_reference_motion_m),
        "minimum_predicted_motion": float(minimum_predicted_motion),
        "calibration_clip_count": len(per_clip),
        "per_clip": per_clip,
        "distribution": distribution(scales),
        "relative_iqr": (q75 - q25) / median,
    }


def camera_scale_stratum(group: str) -> str:
    """Return the known Intern camera family encoded by a split-unit group."""

    if str(group).endswith("_d435i"):
        return "d435i"
    if str(group).endswith("_zed"):
        return "zed"
    raise ValueError(f"unknown Intern camera scale stratum: {group}")


def fit_stratified_translation_scales(
    records: Sequence[Mapping[str, object]],
    *,
    minimum_clips_per_stratum: int = 8,
) -> dict[str, object]:
    """Fit separate frozen scales for the two known Intern camera families."""

    grouped: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        stratum = str(record["scale_stratum"])
        grouped.setdefault(stratum, []).append(record)
    if set(grouped) != {"d435i", "zed"}:
        raise ValueError(f"expected d435i and zed calibration strata, got {sorted(grouped)}")
    strata = {
        name: fit_global_translation_scale(
            values, minimum_clips=minimum_clips_per_stratum
        )
        for name, values in sorted(grouped.items())
    }
    return {
        "mode": "Intern camera-family stratified",
        "stratum_field": "scale_stratum",
        "strata": strata,
        "maximum_relative_iqr": max(
            float(value["relative_iqr"]) for value in strata.values()
        ),
    }


def score_clip(
    record: Mapping[str, object],
    metric_scale: float,
    *,
    minimum_reference_motion_m: float = 0.05,
    minimum_scaled_predicted_motion_m: float = 0.02,
    minimum_motion_pairs: int = 2,
) -> dict[str, object]:
    """Apply a frozen scale and compute per-clip held-out pose errors."""

    predicted_unscaled, reference = _poses(record)
    predicted = predicted_unscaled.copy()
    predicted[:, :2] *= float(metric_scale)
    delta_xy = predicted[1:, :2] - reference[1:, :2]
    translation_errors = np.linalg.norm(delta_xy, axis=1)
    yaw_delta = predicted[1:, 2] - reference[1:, 2]
    yaw_errors = np.abs(np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta)))
    predicted_norm = np.linalg.norm(predicted[1:, :2], axis=1)
    reference_norm = np.linalg.norm(reference[1:, :2], axis=1)
    reference_moving = reference_norm >= float(minimum_reference_motion_m)
    predicted_moving = predicted_norm >= float(minimum_scaled_predicted_motion_m)
    valid_motion = reference_moving & predicted_moving
    scale_errors = np.abs(predicted_norm[reference_moving] / reference_norm[reference_moving] - 1.0)
    direction_cosines = []
    for predicted_xy, reference_xy, valid in zip(
        predicted[1:, :2], reference[1:, :2], valid_motion
    ):
        if valid:
            direction_cosines.append(
                float(
                    np.dot(predicted_xy, reference_xy)
                    / (np.linalg.norm(predicted_xy) * np.linalg.norm(reference_xy))
                )
            )
    confidence_score = float(record["confidence_score"])
    motion_geometry_valid = bool(
        int(valid_motion.sum()) >= int(minimum_motion_pairs)
        and math.isfinite(confidence_score)
    )
    scored = dict(record)
    scored.update(
        {
            "metric_scale": float(metric_scale),
            "predicted_se2": predicted.tolist(),
            "translation_errors_m": translation_errors.tolist(),
            "yaw_errors_rad": yaw_errors.tolist(),
            "translation_error_m_mean": float(np.mean(translation_errors)),
            "translation_error_m_p90": float(np.quantile(translation_errors, 0.90)),
            "yaw_error_rad_mean": float(np.mean(yaw_errors)),
            "yaw_error_rad_p90": float(np.quantile(yaw_errors, 0.90)),
            "scale_relative_errors": scale_errors.tolist(),
            "scale_relative_error_median": (
                float(np.median(scale_errors)) if scale_errors.size else None
            ),
            "direction_cosine_median": (
                float(np.median(direction_cosines)) if direction_cosines else None
            ),
            "reference_moving_pairs": int(reference_moving.sum()),
            "valid_motion_pairs": int(valid_motion.sum()),
            "motion_geometry_valid": motion_geometry_valid,
        }
    )
    return scored


def clip_meets_quality(
    record: Mapping[str, object],
    limits: Mapping[str, float] = DEFAULT_QUALITY_LIMITS,
) -> bool:
    if not bool(record.get("motion_geometry_valid")):
        return False
    for field, maximum in limits.items():
        value = record.get(field)
        if value is None or not math.isfinite(float(value)) or float(value) > float(maximum):
            return False
    return True


def choose_confidence_threshold(
    calibration_records: Sequence[Mapping[str, object]],
    *,
    quality_limits: Mapping[str, float] = DEFAULT_QUALITY_LIMITS,
    minimum_coverage: float = 0.50,
    maximum_bad_rate: float = 0.20,
) -> dict[str, object]:
    """Choose a depth-confidence cutoff using calibration records only.

    The lowest cutoff with the greatest feasible coverage is preferred.  If
    no candidate meets the frozen coverage/bad-rate constraints, the best
    diagnostic candidate is returned with ``criteria_met=false``.
    """

    if not calibration_records:
        raise ValueError("confidence calibration records are empty")
    scores = sorted(
        {
            float(record["confidence_score"])
            for record in calibration_records
            if math.isfinite(float(record["confidence_score"]))
        }
    )
    if not scores:
        raise ValueError("confidence calibration has no finite scores")
    candidates = []
    total = len(calibration_records)
    for threshold in scores:
        accepted = [
            record
            for record in calibration_records
            if bool(record.get("motion_geometry_valid"))
            and float(record["confidence_score"]) >= threshold
        ]
        good = sum(clip_meets_quality(record, quality_limits) for record in accepted)
        bad_rate = 1.0 - good / len(accepted) if accepted else 1.0
        candidates.append(
            {
                "threshold": threshold,
                "accepted_clips": len(accepted),
                "coverage": len(accepted) / total,
                "bad_rate": bad_rate,
            }
        )
    feasible = [
        candidate
        for candidate in candidates
        if candidate["coverage"] >= float(minimum_coverage)
        and candidate["bad_rate"] <= float(maximum_bad_rate)
    ]
    if feasible:
        selected = sorted(
            feasible,
            key=lambda item: (-item["coverage"], item["bad_rate"], item["threshold"]),
        )[0]
        criteria_met = True
    else:
        coverage_candidates = [
            candidate
            for candidate in candidates
            if candidate["coverage"] >= float(minimum_coverage)
        ]
        pool = coverage_candidates or candidates
        selected = sorted(
            pool,
            key=lambda item: (item["bad_rate"], -item["coverage"], item["threshold"]),
        )[0]
        criteria_met = False
    return {
        "confidence_field": "median DA3 dense depth confidence over all selected frames",
        "threshold": float(selected["threshold"]),
        "criteria_met": criteria_met,
        "minimum_coverage": float(minimum_coverage),
        "maximum_bad_rate": float(maximum_bad_rate),
        "quality_limits": dict(quality_limits),
        "selected_candidate": selected,
        "candidates": candidates,
    }


def apply_confidence_gate(
    records: Sequence[Mapping[str, object]], threshold: float
) -> list[dict[str, object]]:
    output = []
    for record in records:
        value = dict(record)
        if not bool(record.get("motion_geometry_valid")):
            accepted, reason = False, "motion_degenerate"
        elif float(record["confidence_score"]) < float(threshold):
            accepted, reason = False, "below_confidence_threshold"
        else:
            accepted, reason = True, "accepted"
        value["admitted"] = accepted
        value["admission_reason"] = reason
        output.append(value)
    return output


def summarize_admission(
    records: Sequence[Mapping[str, object]],
    quality_limits: Mapping[str, float] = DEFAULT_QUALITY_LIMITS,
) -> dict[str, object]:
    admitted = [record for record in records if bool(record.get("admitted"))]
    good = sum(clip_meets_quality(record, quality_limits) for record in admitted)
    groups = sorted({str(record.get("group")) for record in records})
    group_coverage = {}
    for group in groups:
        group_records = [record for record in records if str(record.get("group")) == group]
        group_coverage[group] = {
            "selected": len(group_records),
            "admitted": sum(bool(record.get("admitted")) for record in group_records),
        }
    return {
        "selected_clips": len(records),
        "admitted_clips": len(admitted),
        "coverage": len(admitted) / max(1, len(records)),
        "bad_clip_rate": 1.0 - good / len(admitted) if admitted else 1.0,
        "translation_error_m_mean": distribution(
            [float(record["translation_error_m_mean"]) for record in admitted]
        ),
        "yaw_error_rad_mean": distribution(
            [float(record["yaw_error_rad_mean"]) for record in admitted]
        ),
        "scale_relative_error_median": distribution(
            [
                float(record["scale_relative_error_median"])
                for record in admitted
                if record.get("scale_relative_error_median") is not None
            ]
        ),
        "confidence_score": distribution(
            [float(record["confidence_score"]) for record in admitted]
        ),
        "group_coverage": group_coverage,
    }
