"""Evaluate the frozen detector/ReID target tracker on continuous TpT sequences."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np
import pyarrow.parquet as pq

from omtrackvla.rgb_person_perception import (
    DEFAULT_REID_CODE,
    DEFAULT_REID_WEIGHTS,
    DEFAULT_WEIGHTS,
    RGBPersonPerception,
    bbox_iou,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_fusion_metadata(
    weights_path: Optional[Path],
    fusion_model: object,
) -> Optional[dict[str, object]]:
    """Describe the exact fusion artifact and operating points used for evaluation."""
    if weights_path is None:
        return None
    operating_points = getattr(fusion_model, "operating_points", None)
    if not isinstance(operating_points, Mapping):
        raise ValueError("loaded candidate fusion model has no operating points")
    return {
        "weights_sha256": _sha256(weights_path),
        "trainable_component": True,
        "operating_points": {
            mode: {
                "score_threshold": float(point["score_threshold"]),
                "margin_threshold": float(point["margin_threshold"]),
            }
            for mode, point in operating_points.items()
        },
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "viz_val"), default="viz_val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path)
    parser.add_argument("--detector-weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--detector-architecture",
        choices=(
            "fasterrcnn_mobilenet_v3_large_320_fpn",
            "fasterrcnn_resnet50_fpn_v2",
        ),
        default="fasterrcnn_mobilenet_v3_large_320_fpn",
    )
    parser.add_argument("--reid-weights", type=Path, default=DEFAULT_REID_WEIGHTS)
    parser.add_argument("--reid-code", type=Path, default=DEFAULT_REID_CODE)
    parser.add_argument("--fusion-weights", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector-min-size", type=int)
    parser.add_argument("--detector-max-size", type=int)
    parser.add_argument(
        "--memory-mode",
        choices=("adaptive", "anchor_only"),
        default="adaptive",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--sequence-id",
        action="append",
        dest="sequence_ids",
        help="Evaluate only this manifest-listed sequence; may be repeated.",
    )
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--max-frames-per-sequence", type=int)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not decode TpT RGB frame: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _xywh_to_xyxy(box: Sequence[float], width: int, height: int) -> tuple[float, ...]:
    if box is None or len(box) != 4:
        raise ValueError("visible TpT frame has no xywh target bbox")
    x, y, w, h = map(float, box)
    x0 = float(np.clip(x, 0.0, width - 1.0))
    y0 = float(np.clip(y, 0.0, height - 1.0))
    x1 = float(np.clip(x + w, x0 + 1.0, width))
    y1 = float(np.clip(y + h, y0 + 1.0, height))
    return x0, y0, x1, y1


def _crop(image: np.ndarray, box: Sequence[float]) -> np.ndarray:
    height, width = image.shape[:2]
    x0, y0, x1, y1 = _xywh_to_xyxy(box, width, height)
    return image[int(y0) : int(math.ceil(y1)), int(x0) : int(math.ceil(x1))].copy()


@dataclass
class IdentitySequenceMetrics:
    """Stateful end-to-end metrics for one continuous target sequence."""

    sequence_id: str
    iou_success_threshold: float = 0.5
    wrong_target_iou_threshold: float = 0.2
    evaluated_frames: int = 0
    gt_visible_frames: int = 0
    gt_absent_frames: int = 0
    emitted_frames: int = 0
    emitted_on_visible_frames: int = 0
    emitted_on_absent_frames: int = 0
    successful_frames: int = 0
    wrong_target_frames: int = 0
    wrong_target_track_starts: int = 0
    memory_updates: int = 0
    candidate_count_sum: int = 0
    target_candidate_frames: int = 0
    selected_target_candidate_frames: int = 0
    iou_sum_visible: float = 0.0
    iou_sum_emitted_visible: float = 0.0
    false_follow_streak: int = 0
    max_false_follow_streak: int = 0
    miss_streak: int = 0
    max_miss_streak: int = 0
    reappearance_events: int = 0
    reappearance_failures: int = 0
    reacquisition_latencies: list[int] = field(default_factory=list)
    _previous_gt_visible: Optional[bool] = None
    _wrong_target_active: bool = False
    _pending_reacquisition: Optional[int] = None

    def update(
        self,
        gt_visible: bool,
        gt_bbox: Optional[Sequence[float]],
        predicted_bbox: Optional[Sequence[float]],
        candidate_count: int = 0,
        target_candidate_present: bool = False,
        memory_updated: bool = False,
        frame_step: int = 1,
    ) -> None:
        self.evaluated_frames += 1
        self.candidate_count_sum += int(candidate_count)
        self.memory_updates += int(memory_updated)
        predicted_visible = predicted_bbox is not None
        self.emitted_frames += int(predicted_visible)

        if gt_visible:
            if gt_bbox is None:
                raise ValueError("a visible target requires a bbox")
            self.gt_visible_frames += 1
            self.emitted_on_visible_frames += int(predicted_visible)
            iou = bbox_iou(gt_bbox, predicted_bbox) if predicted_visible else 0.0
            self.iou_sum_visible += iou
            if predicted_visible:
                self.iou_sum_emitted_visible += iou
            success = predicted_visible and iou >= self.iou_success_threshold
            self.target_candidate_frames += int(target_candidate_present)
            self.selected_target_candidate_frames += int(
                target_candidate_present and success
            )
            wrong_target = predicted_visible and iou < self.wrong_target_iou_threshold
            self.successful_frames += int(success)
            self.wrong_target_frames += int(wrong_target)
            if wrong_target and not self._wrong_target_active:
                self.wrong_target_track_starts += 1
            self._wrong_target_active = wrong_target
            self.miss_streak = 0 if success else self.miss_streak + frame_step
            self.max_miss_streak = max(self.max_miss_streak, self.miss_streak)
            self.false_follow_streak = 0

            if self._previous_gt_visible is False:
                self.reappearance_events += 1
                self._pending_reacquisition = 0
            if self._pending_reacquisition is not None:
                if success:
                    self.reacquisition_latencies.append(self._pending_reacquisition)
                    self._pending_reacquisition = None
                else:
                    self._pending_reacquisition += frame_step
        else:
            self.gt_absent_frames += 1
            self.emitted_on_absent_frames += int(predicted_visible)
            self._wrong_target_active = False
            self.miss_streak = 0
            if predicted_visible:
                self.false_follow_streak += frame_step
                self.max_false_follow_streak = max(
                    self.max_false_follow_streak, self.false_follow_streak
                )
            else:
                self.false_follow_streak = 0
            if self._pending_reacquisition is not None:
                self.reappearance_failures += 1
                self._pending_reacquisition = None
        self._previous_gt_visible = bool(gt_visible)

    def finish(self) -> None:
        if self._pending_reacquisition is not None:
            self.reappearance_failures += 1
            self._pending_reacquisition = None

    @staticmethod
    def _ratio(numerator: float, denominator: int) -> Optional[float]:
        return float(numerator / denominator) if denominator else None

    def summary(self) -> dict[str, object]:
        completed_reappearances = len(self.reacquisition_latencies)
        return {
            "sequence_id": self.sequence_id,
            "evaluated_frames": self.evaluated_frames,
            "gt_visible_frames": self.gt_visible_frames,
            "gt_absent_frames": self.gt_absent_frames,
            "emitted_frames": self.emitted_frames,
            "emitted_on_visible_frames": self.emitted_on_visible_frames,
            "emitted_on_absent_frames": self.emitted_on_absent_frames,
            "successful_frames_iou_0_5": self.successful_frames,
            "memory_updates": self.memory_updates,
            "mean_candidate_count": self._ratio(
                self.candidate_count_sum, self.evaluated_frames
            ),
            "detector_candidate_recall_iou_0_5": self._ratio(
                self.target_candidate_frames, self.gt_visible_frames
            ),
            "target_selection_accuracy_when_candidate_present": self._ratio(
                self.selected_target_candidate_frames,
                self.target_candidate_frames,
            ),
            "target_candidate_frames_iou_0_5": self.target_candidate_frames,
            "selected_target_candidate_frames_iou_0_5": (
                self.selected_target_candidate_frames
            ),
            "visibility_recall": self._ratio(
                self.emitted_on_visible_frames, self.gt_visible_frames
            ),
            "absent_false_positive_rate": self._ratio(
                self.emitted_on_absent_frames, self.gt_absent_frames
            ),
            "end_to_end_success_iou_0_5": self._ratio(
                self.successful_frames, self.gt_visible_frames
            ),
            "output_precision_iou_0_5": self._ratio(
                self.successful_frames, self.emitted_frames
            ),
            "mean_iou_visible_missing_as_zero": self._ratio(
                self.iou_sum_visible, self.gt_visible_frames
            ),
            "mean_iou_when_emitted_on_visible": self._ratio(
                self.iou_sum_emitted_visible, self.emitted_on_visible_frames
            ),
            "wrong_target_frames_iou_below_0_2": self.wrong_target_frames,
            "wrong_target_track_starts": self.wrong_target_track_starts,
            "max_consecutive_visible_miss_frames": self.max_miss_streak,
            "max_consecutive_false_follow_frames": self.max_false_follow_streak,
            "reappearance_events": self.reappearance_events,
            "reappearance_success_rate": self._ratio(
                completed_reappearances, self.reappearance_events
            ),
            "reappearance_failures": self.reappearance_failures,
            "mean_reacquisition_latency_frames": (
                float(np.mean(self.reacquisition_latencies))
                if self.reacquisition_latencies
                else None
            ),
            "max_reacquisition_latency_frames": (
                max(self.reacquisition_latencies)
                if self.reacquisition_latencies
                else None
            ),
        }


def _aggregate(summaries: Sequence[Mapping[str, object]]) -> dict[str, object]:
    integer_keys = (
        "evaluated_frames",
        "gt_visible_frames",
        "gt_absent_frames",
        "emitted_frames",
        "emitted_on_visible_frames",
        "emitted_on_absent_frames",
        "successful_frames_iou_0_5",
        "memory_updates",
        "target_candidate_frames_iou_0_5",
        "selected_target_candidate_frames_iou_0_5",
        "wrong_target_frames_iou_below_0_2",
        "wrong_target_track_starts",
        "reappearance_events",
        "reappearance_failures",
    )
    totals = {key: sum(int(item[key]) for item in summaries) for key in integer_keys}
    successful = totals["successful_frames_iou_0_5"]
    target_candidate_frames = totals["target_candidate_frames_iou_0_5"]
    selected_target_candidate_frames = totals[
        "selected_target_candidate_frames_iou_0_5"
    ]
    emitted_visible = totals["emitted_on_visible_frames"]
    false_positive = totals["emitted_on_absent_frames"]
    iou_visible = sum(
        float(item["mean_iou_visible_missing_as_zero"] or 0.0)
        * int(item["gt_visible_frames"])
        for item in summaries
    )
    completed_reappearances = sum(
        round(float(item["reappearance_success_rate"] or 0.0) * int(item["reappearance_events"]))
        for item in summaries
    )

    def ratio(numerator: float, denominator_key: str) -> Optional[float]:
        denominator = totals[denominator_key]
        return float(numerator / denominator) if denominator else None

    totals.update(
        {
            "sequence_count": len(summaries),
            "visibility_recall": ratio(emitted_visible, "gt_visible_frames"),
            "absent_false_positive_rate": ratio(
                false_positive, "gt_absent_frames"
            ),
            "end_to_end_success_iou_0_5": ratio(
                successful, "gt_visible_frames"
            ),
            "detector_candidate_recall_iou_0_5": ratio(
                target_candidate_frames, "gt_visible_frames"
            ),
            "target_selection_accuracy_when_candidate_present": (
                float(selected_target_candidate_frames / target_candidate_frames)
                if target_candidate_frames
                else None
            ),
            "output_precision_iou_0_5": (
                float(successful / totals["emitted_frames"])
                if totals["emitted_frames"]
                else None
            ),
            "mean_iou_visible_missing_as_zero": ratio(
                iou_visible, "gt_visible_frames"
            ),
            "reappearance_success_rate": ratio(
                completed_reappearances, "reappearance_events"
            ),
            "max_consecutive_visible_miss_frames": max(
                (int(item["max_consecutive_visible_miss_frames"]) for item in summaries),
                default=0,
            ),
            "max_consecutive_false_follow_frames": max(
                (int(item["max_consecutive_false_follow_frames"]) for item in summaries),
                default=0,
            ),
        }
    )
    return totals


def evaluate_tpt_sequence(
    tracker: RGBPersonPerception,
    root: Path,
    sequence_id: str,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    progress_every: int = 0,
    records_path: Optional[Path] = None,
    split: Optional[str] = None,
) -> dict[str, object]:
    parquet = root / sequence_id / "frames.parquet"
    table = pq.read_table(
        parquet,
        columns=["video_idx", "bbox_qv", "is_exist"],
    ).to_pydict()
    visible_indices = [
        index for index, visible in enumerate(table["is_exist"]) if bool(visible)
    ]
    if not visible_indices:
        raise ValueError(f"TpT sequence {sequence_id} has no visible initialization")
    initial = visible_indices[0]
    initial_path = root / sequence_id / "rgb_frames" / (
        f"frame_{int(table['video_idx'][initial]):06d}.jpg"
    )
    initial_rgb = _load_rgb(initial_path)
    tracker.reset(_crop(initial_rgb, table["bbox_qv"][initial]))
    metrics = IdentitySequenceMetrics(sequence_id)
    records_handle = None
    if records_path is not None:
        records_path.parent.mkdir(parents=True, exist_ok=True)
        records_handle = records_path.open("w", encoding="utf-8")

    try:
        stop = len(table["video_idx"])
        if max_frames is not None:
            stop = min(stop, initial + 1 + max_frames * frame_stride)
        for offset, index in enumerate(range(initial + frame_stride, stop, frame_stride), start=1):
            video_index = int(table["video_idx"][index])
            path = root / sequence_id / "rgb_frames" / f"frame_{video_index:06d}.jpg"
            rgb = _load_rgb(path)
            observation = tracker(rgb, None)
            visible = bool(table["is_exist"][index])
            gt_bbox = (
                _xywh_to_xyxy(table["bbox_qv"][index], rgb.shape[1], rgb.shape[0])
                if visible
                else None
            )
            predicted_bbox = observation.bbox_xyxy if observation.visible else None
            target_candidate_present = bool(
                visible
                and any(
                    bbox_iou(gt_bbox, candidate["bbox_xyxy"]) >= 0.5
                    for candidate in tracker.last_candidate_diagnostics
                )
            )
            metrics.update(
                visible,
                gt_bbox,
                predicted_bbox,
                candidate_count=tracker.last_candidate_count,
                target_candidate_present=target_candidate_present,
                memory_updated=tracker.last_memory_updated,
                frame_step=frame_stride,
            )
            if records_handle is not None:
                candidate_records = []
                for candidate in tracker.last_candidate_diagnostics:
                    candidate_record = dict(candidate)
                    candidate_record["target_iou"] = (
                        bbox_iou(gt_bbox, candidate["bbox_xyxy"])
                        if gt_bbox is not None
                        else 0.0
                    )
                    candidate_record["target_match_iou_0_5"] = (
                        candidate_record["target_iou"] >= 0.5
                    )
                    candidate_records.append(candidate_record)
                records_handle.write(
                    json.dumps(
                        {
                            "sequence_id": sequence_id,
                            "split": split,
                            "frame_index": index,
                            "video_index": video_index,
                            "gt_visible": visible,
                            "gt_bbox_xyxy": gt_bbox,
                            "predicted_bbox_xyxy": predicted_bbox,
                            "iou": (
                                bbox_iou(gt_bbox, predicted_bbox)
                                if gt_bbox is not None and predicted_bbox is not None
                                else 0.0
                            ),
                            "association_score": tracker.last_association_score,
                            "identity_similarity": tracker.last_goal_similarity,
                            "anchor_similarity": tracker.last_anchor_similarity,
                            "identity_margin": tracker.last_identity_margin,
                            "memory_updated": tracker.last_memory_updated,
                            "candidates": candidate_records,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            if progress_every and offset % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "sequence_id": sequence_id,
                            "processed_frames": offset,
                            "end_to_end_success": metrics.summary()[
                                "end_to_end_success_iou_0_5"
                            ],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        if records_handle is not None:
            records_handle.close()
    metrics.finish()
    return metrics.summary()


def main() -> int:
    args = _arguments()
    if args.frame_stride <= 0:
        raise ValueError("frame stride must be positive")
    if args.max_sequences is not None and args.max_sequences <= 0:
        raise ValueError("max sequences must be positive")
    if args.max_frames_per_sequence is not None and args.max_frames_per_sequence <= 0:
        raise ValueError("max frames per sequence must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sequences = list(
        manifest["datasets"]["tpt_bench_clean_v2"]["splits"][args.split]
    )
    if args.sequence_ids:
        unknown = sorted(set(args.sequence_ids) - set(sequences))
        if unknown:
            raise ValueError(
                f"requested sequences are not in split {args.split}: {unknown}"
            )
        sequences = list(dict.fromkeys(args.sequence_ids))
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]
    tracker_options = {}
    if args.memory_mode == "anchor_only":
        tracker_options.update(
            memory_update_detector_threshold=2.0,
            memory_max_positive_embeddings=1,
            memory_max_negative_embeddings=0,
        )
    tracker = RGBPersonPerception(
        weights_path=args.detector_weights,
        detector_architecture=args.detector_architecture,
        reid_weights_path=args.reid_weights,
        reid_code_path=args.reid_code,
        fusion_weights_path=args.fusion_weights,
        device=args.device,
        detector_min_size=args.detector_min_size,
        detector_max_size=args.detector_max_size,
        **tracker_options,
    )
    summaries = [
        evaluate_tpt_sequence(
            tracker,
            args.data_root,
            sequence_id,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames_per_sequence,
            progress_every=args.progress_every,
            records_path=(
                args.records_dir / f"{sequence_id}.jsonl"
                if args.records_dir is not None
                else None
            ),
            split=args.split,
        )
        for sequence_id in sequences
    ]
    result = {
        "schema_version": 1,
        "benchmark_id": "pretrained-fasterrcnn-osnet-target-memory-v1",
        "split": args.split,
        "source": "tpt_bench_clean_v2",
        "frame_stride": args.frame_stride,
        "memory_mode": args.memory_mode,
        "components": {
            "person_detector": {
                "architecture": tracker.detector_architecture,
                "weights_sha256": _sha256(args.detector_weights),
                "frozen": True,
                "min_size": tracker.detector_min_size,
                "max_size": tracker.detector_max_size,
            },
            "person_reid": {
                "architecture": "osnet_x0_25",
                "pretraining_dataset": "MSMT17",
                "weights_sha256": _sha256(args.reid_weights),
                "frozen": True,
            },
            "candidate_fusion": _candidate_fusion_metadata(
                args.fusion_weights,
                tracker.fusion_model,
            ),
        },
        "adaptive_memory": {
            "immutable_anchor": True,
            "max_positive_embeddings": tracker.memory_max_positive_embeddings,
            "max_negative_embeddings": tracker.memory_max_negative_embeddings,
            "update_detector_threshold": tracker.memory_update_detector_threshold,
            "update_identity_threshold": tracker.memory_update_identity_threshold,
            "update_anchor_threshold": tracker.memory_update_anchor_threshold,
            "update_association_threshold": tracker.memory_update_association_threshold,
            "update_margin": tracker.memory_update_margin,
            "update_min_confirmed_steps": tracker.memory_update_min_confirmed_steps,
        },
        "aggregate": _aggregate(summaries),
        "sequences": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
