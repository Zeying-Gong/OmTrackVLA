"""Train a sequence-aware fusion head over frozen detector/ReID features."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from omtrackvla.models.candidate_fusion import FEATURE_NAMES, feature_vector
from omtrackvla.training.train_candidate_fusion import (
    _calibrate,
    _calibration_partitions,
    _calibration_sequences,
    _record_paths,
    _selection_metrics,
    _sha256,
)


@dataclass(frozen=True)
class CandidateRow:
    sequence_id: str
    frame_key: tuple[str, str, int]
    features: np.ndarray
    label: float
    target_iou: float


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--validation-records", type=Path, nargs="+", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--initial-weights", type=Path)
    parser.add_argument("--train-new-features-only", action="store_true")
    parser.add_argument(
        "--operating-point-source",
        choices=("calibrated", "inherited"),
        default="calibrated",
    )
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--pair-batch-size", type=int, default=4096)
    parser.add_argument("--hard-negatives-per-frame", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_rows(
    inputs: Sequence[Path], required_split: str
) -> tuple[list[CandidateRow], dict[tuple[str, str, int], list[int]], set[str], list[Path]]:
    paths = _record_paths(inputs)
    rows: list[CandidateRow] = []
    frames: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    sequence_ids: set[str] = set()
    for path in paths:
        source_id = hashlib.sha256(str(path).encode()).hexdigest()[:16]
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("split") != required_split:
                    raise ValueError(
                        f"expected {required_split} candidate records: {path}"
                    )
                sequence_id = str(record["sequence_id"])
                sequence_ids.add(sequence_id)
                frame_key = (
                    sequence_id,
                    source_id,
                    int(record["frame_index"]),
                )
                candidates = list(record.get("candidates", ()))
                frames[frame_key]
                for candidate in candidates:
                    enriched = dict(candidate)
                    enriched.setdefault("candidate_count", len(candidates))
                    index = len(rows)
                    rows.append(
                        CandidateRow(
                            sequence_id=sequence_id,
                            frame_key=frame_key,
                            features=feature_vector(enriched),
                            label=float(candidate["target_match_iou_0_5"]),
                            target_iou=float(candidate.get("target_iou", 0.0)),
                        )
                    )
                    frames[frame_key].append(index)
    if not rows or not any(row.label > 0.5 for row in rows):
        raise ValueError(
            f"{required_split} candidate records contain no positive detections"
        )
    return rows, frames, sequence_ids, paths


def _hard_pairs(
    rows: Sequence[CandidateRow],
    frames: dict[tuple[str, str, int], list[int]],
    allowed_sequences: set[str],
    negatives_per_frame: int,
) -> list[tuple[int, int]]:
    association_index = FEATURE_NAMES.index("association_score")
    pairs: list[tuple[int, int]] = []
    for frame_key, indices in frames.items():
        if frame_key[0] not in allowed_sequences:
            continue
        positive = [index for index in indices if rows[index].label > 0.5]
        negative = [index for index in indices if rows[index].label <= 0.5]
        if not positive or not negative:
            continue
        best_positive = max(positive, key=lambda index: rows[index].target_iou)
        hardest = sorted(
            negative,
            key=lambda index: float(rows[index].features[association_index]),
            reverse=True,
        )[:negatives_per_frame]
        pairs.extend((best_positive, index) for index in hardest)
    return pairs


class _FusionNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layer2(torch.relu(self.layer1(value))).squeeze(1)


def _initialize_from_fusion(
    model: _FusionNetwork,
    payload: Mapping[str, object],
    target_mean: np.ndarray,
    target_scale: np.ndarray,
) -> None:
    """Expand a legacy fusion head while preserving its exact logits."""
    source_names = tuple(str(value) for value in payload["feature_names"])
    if not set(source_names).issubset(FEATURE_NAMES):
        raise ValueError("initial fusion contains unsupported features")
    source_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
    source_scale = np.asarray(payload["feature_scale"], dtype=np.float32)
    source_weight1 = np.asarray(payload["weight1"], dtype=np.float32)
    source_bias1 = np.asarray(payload["bias1"], dtype=np.float32)
    source_weight2 = np.asarray(payload["weight2"], dtype=np.float32)
    source_bias2 = float(payload["bias2"])
    if source_mean.shape != (len(source_names),):
        raise ValueError("initial fusion normalization shape mismatch")
    if source_scale.shape != source_mean.shape or np.any(source_scale <= 1e-6):
        raise ValueError("initial fusion scale is invalid")
    if source_weight1.shape != (model.layer1.out_features, len(source_names)):
        raise ValueError("initial fusion hidden dimension mismatch")
    if source_bias1.shape != (model.layer1.out_features,):
        raise ValueError("initial fusion first-layer bias mismatch")
    if source_weight2.shape != (model.layer1.out_features,):
        raise ValueError("initial fusion output-layer shape mismatch")

    expanded_weight1 = np.zeros(
        (model.layer1.out_features, len(FEATURE_NAMES)), dtype=np.float32
    )
    source_offsets = np.zeros(len(source_names), dtype=np.float32)
    for source_index, name in enumerate(source_names):
        target_index = FEATURE_NAMES.index(name)
        expanded_weight1[:, target_index] = (
            source_weight1[:, source_index]
            * target_scale[target_index]
            / source_scale[source_index]
        )
        source_offsets[source_index] = (
            target_mean[target_index] - source_mean[source_index]
        ) / source_scale[source_index]
    expanded_bias1 = source_bias1 + source_weight1 @ source_offsets

    with torch.no_grad():
        model.layer1.weight.copy_(torch.from_numpy(expanded_weight1))
        model.layer1.bias.copy_(torch.from_numpy(expanded_bias1))
        model.layer2.weight.copy_(torch.from_numpy(source_weight2[None, :]))
        model.layer2.bias.fill_(source_bias2)


def _new_feature_only_parameters(
    model: _FusionNetwork, source_feature_names: Sequence[str]
) -> list[nn.Parameter]:
    new_indices = [
        index for index, name in enumerate(FEATURE_NAMES)
        if name not in set(source_feature_names)
    ]
    if not new_indices:
        raise ValueError("initial fusion leaves no new features to train")
    mask = torch.zeros_like(model.layer1.weight)
    mask[:, new_indices] = 1.0
    model.layer1.weight.register_hook(lambda gradient: gradient * mask)
    model.layer1.bias.requires_grad_(False)
    model.layer2.weight.requires_grad_(False)
    model.layer2.bias.requires_grad_(False)
    return [model.layer1.weight]


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _sequence_weights(rows: Sequence[CandidateRow], indices: Sequence[int]) -> np.ndarray:
    counts = Counter(rows[index].sequence_id for index in indices)
    weights = np.asarray(
        [1.0 / counts[rows[index].sequence_id] for index in indices],
        dtype=np.float32,
    )
    return weights / max(float(weights.mean()), 1e-12)


def _normalized_matrix(
    rows: Sequence[CandidateRow],
    indices: Sequence[int],
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return (
        np.stack([rows[index].features for index in indices]) - mean
    ) / scale


def _predict_probabilities(
    model: nn.Module, values: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values)),
        batch_size=batch_size,
        shuffle=False,
    )
    outputs = []
    model.eval()
    with torch.inference_mode():
        for (features,) in loader:
            outputs.append(torch.sigmoid(model(features.to(device))).cpu())
    return torch.cat(outputs).numpy()


def _calibrate_or_raise(
    probabilities: np.ndarray,
    labels: np.ndarray,
    frame_rows: Sequence[Sequence[int]],
    mode: str,
    precision_floor: float,
    beta: float,
) -> tuple[float, float, dict[str, float]]:
    score, margin, metrics = _calibrate(
        probabilities,
        labels,
        frame_rows,
        precision_floor=precision_floor,
        beta=beta,
    )
    if not metrics["precision_floor_met"]:
        raise ValueError(
            f"{mode} calibration cannot satisfy precision floor "
            f"{precision_floor:.2f} on train-only calibration data"
        )
    return score, margin, metrics


def _validation_loss(
    model: nn.Module,
    values: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    row_lookup: dict[int, int],
    device: torch.device,
    batch_size: int,
    positive_weight: float,
    pairwise_weight: float,
    pairwise_margin: float,
) -> float:
    model.eval()
    features = torch.from_numpy(values).to(device)
    labels = torch.from_numpy(targets).to(device)
    sample_weights = torch.from_numpy(weights).to(device)
    with torch.inference_mode():
        logits = []
        for start in range(0, len(features), batch_size):
            logits.append(model(features[start : start + batch_size]))
        logits = torch.cat(logits)
        losses = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=torch.tensor(positive_weight, device=device),
            reduction="none",
        )
        candidate_loss = (losses * sample_weights).sum() / sample_weights.sum()
        if pairs:
            local_pairs = torch.tensor(
                [(row_lookup[a], row_lookup[b]) for a, b in pairs],
                dtype=torch.long,
                device=device,
            )
            pair_loss = F.softplus(
                logits[local_pairs[:, 1]]
                - logits[local_pairs[:, 0]]
                + pairwise_margin
            ).mean()
        else:
            pair_loss = torch.zeros((), device=device)
    return float(candidate_loss + pairwise_weight * pair_loss)


def main() -> int:
    args = _arguments()
    if args.epochs <= 0 or args.hard_negatives_per_frame <= 0:
        raise ValueError("epochs and hard-negatives-per-frame must be positive")
    if (
        args.train_new_features_only
        or args.operating_point_source == "inherited"
    ) and args.initial_weights is None:
        raise ValueError(
            "new-feature-only training and inherited operating points require "
            "--initial-weights"
        )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )

    rows, frames, sequence_ids, paths = _load_rows(args.records, "train")
    validation_rows, validation_frames, validation_sequences, validation_paths = (
        _load_rows(args.validation_records, "val")
    )
    calibration_sequences = _calibration_sequences(sequence_ids, args.seed)
    model_sequences = set(sequence_ids) - calibration_sequences
    train_indices = [
        index
        for index, row in enumerate(rows)
        if row.sequence_id in model_sequences
    ]
    calibration_indices = [
        index
        for index, row in enumerate(rows)
        if row.sequence_id in calibration_sequences
    ]
    validation_indices = list(range(len(validation_rows)))

    raw_train = np.stack([rows[index].features for index in train_indices])
    mean = raw_train.mean(axis=0)
    scale = raw_train.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    train_x = (raw_train - mean) / scale
    train_y = np.asarray(
        [rows[index].label for index in train_indices], dtype=np.float32
    )
    train_w = _sequence_weights(rows, train_indices)
    validation_x = _normalized_matrix(
        validation_rows, validation_indices, mean, scale
    )
    validation_y = np.asarray(
        [row.label for row in validation_rows], dtype=np.float32
    )
    validation_w = _sequence_weights(validation_rows, validation_indices)

    train_pairs = _hard_pairs(
        rows,
        frames,
        model_sequences,
        args.hard_negatives_per_frame,
    )
    validation_pairs = _hard_pairs(
        validation_rows,
        validation_frames,
        validation_sequences,
        args.hard_negatives_per_frame,
    )
    if not train_pairs:
        raise ValueError(
            "train candidate records contain no positive/negative hard pairs"
        )
    train_lookup = {row_index: offset for offset, row_index in enumerate(train_indices)}
    validation_lookup = {
        row_index: offset for offset, row_index in enumerate(validation_indices)
    }
    validation_frame_rows = [
        [validation_lookup[index] for index in indices]
        for indices in validation_frames.values()
        if indices
    ]

    candidate_dataset = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(train_y),
        torch.from_numpy(train_w),
    )
    pair_positive = np.stack(
        [train_x[train_lookup[positive]] for positive, _ in train_pairs]
    )
    pair_negative = np.stack(
        [train_x[train_lookup[negative]] for _, negative in train_pairs]
    )
    pair_dataset = TensorDataset(
        torch.from_numpy(pair_positive), torch.from_numpy(pair_negative)
    )
    generator = torch.Generator().manual_seed(args.seed)
    candidate_loader = DataLoader(
        candidate_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    pair_loader = DataLoader(
        pair_dataset,
        batch_size=args.pair_batch_size,
        shuffle=True,
        generator=generator,
    )

    model = _FusionNetwork(len(FEATURE_NAMES), args.hidden_dim).to(device)
    initial_payload = None
    if args.initial_weights is not None:
        initial_payload = json.loads(args.initial_weights.read_text(encoding="utf-8"))
        _initialize_from_fusion(model, initial_payload, mean, scale)
    optimizer_parameters = list(model.parameters())
    optimizer_weight_decay = args.weight_decay
    if args.train_new_features_only:
        optimizer_parameters = _new_feature_only_parameters(
            model, initial_payload["feature_names"]
        )
        # Decoupled weight decay would also move the frozen columns because
        # they share one parameter tensor with the new columns.
        optimizer_weight_decay = 0.0
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=args.learning_rate,
        weight_decay=optimizer_weight_decay,
    )
    positives = float(train_y.sum())
    positive_weight = min(20.0, (len(train_y) - positives) / max(1.0, positives))
    best_loss = float("inf")
    best_state = None
    best_key = None
    best_epoch = None
    history = []

    def consider_checkpoint(epoch: int, train_loss: Optional[float]) -> None:
        nonlocal best_epoch, best_key, best_loss, best_state
        validation_loss = _validation_loss(
            model,
            validation_x,
            validation_y,
            validation_w,
            validation_pairs,
            validation_lookup,
            device,
            args.batch_size,
            positive_weight,
            args.pairwise_weight,
            args.pairwise_margin,
        )
        probabilities = _predict_probabilities(
            model, validation_x, device, args.batch_size
        )
        ranking = _selection_metrics(
            probabilities,
            validation_y,
            validation_frame_rows,
            score_threshold=0.0,
            margin_threshold=0.0,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_candidate_ranking": ranking,
            }
        )
        key = (
            float(ranking["f0_5"]),
            int(ranking["true_positive"]),
            -validation_loss,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if initial_payload is not None:
        consider_checkpoint(epoch=0, train_loss=None)
    for epoch in range(args.epochs):
        model.train()
        candidate_iterator = iter(candidate_loader)
        pair_iterator = iter(pair_loader)
        steps = max(len(candidate_loader), len(pair_loader))
        total_loss = 0.0
        for _ in range(steps):
            (features, targets, sample_weights), candidate_iterator = _next_batch(
                candidate_iterator, candidate_loader
            )
            (positive, negative), pair_iterator = _next_batch(
                pair_iterator, pair_loader
            )
            features = features.to(device)
            targets = targets.to(device)
            sample_weights = sample_weights.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            losses = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=torch.tensor(positive_weight, device=device),
                reduction="none",
            )
            candidate_loss = (losses * sample_weights).sum() / sample_weights.sum()
            pair_loss = F.softplus(
                model(negative) - model(positive) + args.pairwise_margin
            ).mean()
            loss = candidate_loss + args.pairwise_weight * pair_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())

        consider_checkpoint(
            epoch=epoch + 1,
            train_loss=total_loss / max(1, steps),
        )

    if best_state is None:
        raise RuntimeError("candidate fusion did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()

    calibration_x = _normalized_matrix(
        rows, calibration_indices, mean, scale
    )
    calibration_y = np.asarray(
        [rows[index].label for index in calibration_indices], dtype=np.float32
    )
    calibration_probabilities = _predict_probabilities(
        model, calibration_x, device, args.batch_size
    )
    calibration_lookup = {
        row_index: offset for offset, row_index in enumerate(calibration_indices)
    }
    tuple_rows = [
        (row.sequence_id, row.frame_key, row.features, row.label)
        for row in rows
    ]
    calibration_partitions = _calibration_partitions(
        tuple_rows,
        frames,
        calibration_sequences,
        calibration_lookup,
    )
    policies = {
        "tracking": {"precision_floor": 0.70, "beta": 1.0},
        "reacquisition": {"precision_floor": 0.75, "beta": 0.5},
    }
    operating_points = {}
    for mode, policy in policies.items():
        if args.operating_point_source == "inherited":
            raw_point = initial_payload.get("operating_points", {}).get(mode, {})
            score = float(
                raw_point.get("score_threshold", initial_payload["score_threshold"])
            )
            margin = float(
                raw_point.get("margin_threshold", initial_payload["margin_threshold"])
            )
            metrics = _selection_metrics(
                calibration_probabilities,
                calibration_y,
                calibration_partitions[mode],
                score_threshold=score,
                margin_threshold=margin,
            )
            metrics.update(
                precision_floor=float(policy["precision_floor"]),
                precision_floor_met=bool(
                    metrics["precision"] >= policy["precision_floor"]
                ),
                inherited=True,
            )
        else:
            score, margin, metrics = _calibrate_or_raise(
                calibration_probabilities,
                calibration_y,
                calibration_partitions[mode],
                mode,
                policy["precision_floor"],
                policy["beta"],
            )
        operating_points[mode] = {
            "score_threshold": score,
            "margin_threshold": margin,
            "source": args.operating_point_source,
            "precision_floor": policy["precision_floor"],
            "beta": policy["beta"],
            "calibration_frame_count": len(calibration_partitions[mode]),
            "calibration": metrics,
        }

    validation_probabilities = _predict_probabilities(
        model, validation_x, device, args.batch_size
    )
    validation_ranking = _selection_metrics(
        validation_probabilities,
        validation_y,
        validation_frame_rows,
        score_threshold=0.0,
        margin_threshold=0.0,
    )
    fallback = operating_points["reacquisition"]
    payload = {
        "schema_version": 3,
        "model_id": "frozen-detector-osnet-temporal-candidate-fusion-v3",
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "hidden_dim": args.hidden_dim,
        "weight1": model.layer1.weight.detach().cpu().numpy().tolist(),
        "bias1": model.layer1.bias.detach().cpu().numpy().tolist(),
        "weight2": model.layer2.weight.detach().cpu().numpy()[0].tolist(),
        "bias2": float(model.layer2.bias.detach().cpu().numpy()[0]),
        "score_threshold": fallback["score_threshold"],
        "margin_threshold": fallback["margin_threshold"],
        "operating_points": operating_points,
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "selected_epoch": best_epoch,
            "model_selection_metric": "val_candidate_ranking_f0_5",
            "model_selection_split": "val",
            "threshold_calibration_split": "train",
            "train_new_features_only": args.train_new_features_only,
            "operating_point_source": args.operating_point_source,
            "initial_weights": (
                {
                    "path": str(args.initial_weights.resolve()),
                    "sha256": _sha256(args.initial_weights.resolve()),
                }
                if args.initial_weights is not None
                else None
            ),
            "model_train_sequences": sorted(model_sequences),
            "calibration_sequences": sorted(calibration_sequences),
            "validation_sequences": sorted(validation_sequences),
            "train_candidates": len(train_indices),
            "calibration_candidates": len(calibration_indices),
            "validation_candidates": len(validation_indices),
            "hard_negative_pairs": len(train_pairs),
            "records": [
                {"path": str(path), "sha256": _sha256(path)} for path in paths
            ],
            "validation_records": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in validation_paths
            ],
        },
    }
    report = {
        "schema_version": 3,
        "model_id": payload["model_id"],
        "best_validation_loss": best_loss,
        "minimum_validation_loss": min(
            float(item["validation_loss"]) for item in history
        ),
        "validation_candidate_ranking": validation_ranking,
        "operating_points": operating_points,
        "history": history,
        "training": payload["training"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "best_validation_loss": best_loss,
                "validation_candidate_ranking": validation_ranking,
                "operating_points": operating_points,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
