#!/usr/bin/env python3
"""Fit KPR operating points from train-only candidate records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sequences", type=int, default=37)
    return parser.parse_args()


def _quantiles(values):
    return {
        str(q): float(np.quantile(values, q))
        for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    }


def _fit(examples, *, single_candidate=False):
    eligible = [
        row
        for row in examples
        if (row["candidate_count"] == 1) == single_candidate
    ]
    best = None
    for threshold in np.arange(0.30, 0.751, 0.005):
        for margin in np.arange(0.0, 0.201, 0.005):
            accepted = [
                row
                for row in eligible
                if row["score"] >= threshold and row["margin"] >= margin
            ]
            true_positive = sum(row["correct"] for row in accepted)
            false_positive = len(accepted) - true_positive
            precision = (
                true_positive / len(accepted) if accepted else 1.0
            )
            present = sum(row["candidate_present"] for row in eligible)
            recall = true_positive / present if present else 0.0
            absent = sum(not row["gt_visible"] for row in eligible)
            absent_fp = sum(not row["gt_visible"] for row in accepted)
            absent_fpr = absent_fp / absent if absent else 0.0
            if precision < 0.90 or absent_fpr > 0.05:
                continue
            beta2 = 0.25
            f_beta = (
                (1.0 + beta2) * precision * recall
                / (beta2 * precision + recall)
                if precision + recall > 0.0
                else 0.0
            )
            candidate = (f_beta, true_positive, -threshold, -margin)
            if best is None or candidate > best[0]:
                best = (
                    candidate,
                    {
                        "score_threshold": float(threshold),
                        "margin_threshold": float(margin),
                        "precision": float(precision),
                        "candidate_present_recall": float(recall),
                        "absent_fpr": float(absent_fpr),
                        "accepted_frames": len(accepted),
                    },
                )
    if best is None:
        raise RuntimeError("no train operating point meets precision/FPR safeguards")
    return best[1]


def _fit_memory_updates(examples):
    """Choose a zero-contamination gallery-write policy on train records.

    Gallery contamination is persistent and is consequently treated more
    strictly than an emitted box.  The records expose the tracker state before
    selection; ``confirmed_track_steps`` is incremented once before the real
    memory-update check, hence the returned minimum is ``before + 1``.
    """

    best = None
    for detector_threshold in (0.70, 0.80, 0.90, 0.95, 0.98, 0.99):
        for identity_threshold in np.arange(0.45, 0.751, 0.025):
            for association_threshold in np.arange(0.45, 0.851, 0.025):
                for margin_threshold in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20):
                    for confirmed_before in (1, 2, 3, 4, 5, 8):
                        accepted = [
                            row
                            for row in examples
                            if row["detector_score"] >= detector_threshold
                            and row["identity"] >= identity_threshold
                            and row["anchor"] >= identity_threshold
                            and row["association"] >= association_threshold
                            and row["margin"] >= margin_threshold
                            and row["confirmed_before"] >= confirmed_before
                        ]
                        if not accepted or not all(row["correct"] for row in accepted):
                            continue
                        sequence_count = len({row["sequence_id"] for row in accepted})
                        # Prefer broad sequence coverage, then more safe views.
                        # For exact ties retain the less anchor-biased threshold
                        # and the stronger temporal/association evidence.
                        rank = (
                            sequence_count,
                            len(accepted),
                            -float(identity_threshold),
                            float(association_threshold),
                            int(confirmed_before),
                            float(detector_threshold),
                            float(margin_threshold),
                        )
                        if best is None or rank > best[0]:
                            best = (
                                rank,
                                {
                                    "update_detector_threshold": float(detector_threshold),
                                    "update_identity_threshold": float(identity_threshold),
                                    "update_anchor_threshold": float(identity_threshold),
                                    "update_association_threshold": float(association_threshold),
                                    "update_margin": float(margin_threshold),
                                    "update_min_confirmed_steps": int(confirmed_before + 1),
                                    "accepted_updates": len(accepted),
                                    "sequence_coverage": sequence_count,
                                    "precision": 1.0,
                                    "false_updates": 0,
                                },
                            )
    if best is None:
        raise RuntimeError("no zero-contamination KPR memory-update policy found")
    return best[1]


def main() -> int:
    args = _arguments()
    files = sorted(args.records_root.glob("shard_gpu*/records/*.jsonl"))
    sequence_ids = {path.stem for path in files}
    if len(sequence_ids) != args.expected_sequences:
        raise RuntimeError(
            f"expected {args.expected_sequences} train sequences, found {len(sequence_ids)}"
        )
    positive_anchor = []
    negative_anchor = []
    examples = []
    memory_examples = []
    frames = 0
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("split") != "train":
                raise RuntimeError(f"non-train record found in {path}")
            frames += 1
            candidates = row.get("candidates", [])
            for candidate in candidates:
                target = bool(candidate["target_match_iou_0_5"])
                (positive_anchor if target else negative_anchor).append(
                    float(candidate["anchor_reid"])
                )
            if not candidates:
                continue
            selected = [candidate for candidate in candidates if candidate.get("selected")]
            if len(selected) > 1:
                raise RuntimeError(f"multiple selected candidates in {path}")
            if selected:
                candidate = selected[0]
                if (
                    float(candidate.get("global_search", 1.0)) < 0.5
                    and int(candidate.get("missed_steps", 1)) == 0
                ):
                    memory_examples.append(
                        {
                            "sequence_id": str(row["sequence_id"]),
                            "correct": bool(candidate["target_match_iou_0_5"]),
                            "detector_score": float(candidate["detector_score"]),
                            "identity": float(candidate["goal_reid"]),
                            "anchor": float(candidate["anchor_reid"]),
                            "association": float(candidate["association_score"]),
                            "margin": float(row.get("identity_margin") or 0.0),
                            "confirmed_before": int(
                                candidate.get("confirmed_track_steps") or 0
                            ),
                        }
                    )
            ranked = sorted(
                candidates,
                key=lambda candidate: (
                    0.55 * float(candidate["anchor_reid"])
                    + 0.45 * float(candidate["goal_histogram"])
                ),
                reverse=True,
            )
            top = ranked[0]
            score = (
                0.55 * float(top["anchor_reid"])
                + 0.45 * float(top["goal_histogram"])
            )
            second = (
                0.55 * float(ranked[1]["anchor_reid"])
                + 0.45 * float(ranked[1]["goal_histogram"])
                if len(ranked) > 1
                else score - 1.0
            )
            examples.append(
                {
                    "score": score,
                    "margin": score - second,
                    "correct": bool(top["target_match_iou_0_5"]),
                    "candidate_present": any(
                        item["target_match_iou_0_5"] for item in candidates
                    ),
                    "gt_visible": bool(row["gt_visible"]),
                    "candidate_count": len(candidates),
                }
            )
    if not positive_anchor or not negative_anchor:
        raise RuntimeError("train records do not contain both target and distractor candidates")
    multiple = _fit(examples, single_candidate=False)
    single = _fit(examples, single_candidate=True)
    positive_q = _quantiles(positive_anchor)
    result = {
        "schema_version": 1,
        "source_split": "train",
        "record_files": len(files),
        "sequence_count": len(sequence_ids),
        "frames": frames,
        "positive_candidates": len(positive_anchor),
        "negative_candidates": len(negative_anchor),
        "positive_anchor_quantiles": positive_q,
        "negative_anchor_quantiles": _quantiles(negative_anchor),
        "tracking": {
            "identity_floor": float(np.clip(positive_q["0.01"], 0.30, 0.50)),
            "short_reacquisition_identity_threshold": float(
                np.clip(positive_q["0.1"], 0.35, 0.60)
            ),
        },
        "global_multiple": multiple,
        "global_single": single,
        "reacquisition": {
            "confirm_frames": 3,
            "consistency_iou": 0.10,
            "consistency_reid": 0.45,
        },
        "memory_updates": _fit_memory_updates(memory_examples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
