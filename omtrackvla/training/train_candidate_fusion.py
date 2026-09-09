"""Train a small candidate fusion head while detector and ReID stay frozen."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from omtrackvla.models.candidate_fusion import FEATURE_NAMES, feature_vector


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260908)
    return parser.parse_args()


def _record_paths(inputs: Sequence[Path]) -> list[Path]:
    paths = []
    for value in inputs:
        if value.is_dir():
            paths.extend(value.rglob("*.jsonl"))
        elif value.is_file():
            paths.append(value)
        else:
            raise FileNotFoundError(value)
    paths = sorted(set(path.resolve() for path in paths))
    if not paths:
        raise ValueError("no candidate record files found")
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _calibration_sequences(sequence_ids: Iterable[str], seed: int) -> set[str]:
    ordered = sorted(set(sequence_ids))
    if len(ordered) < 2:
        raise ValueError("candidate fusion requires at least two train sequences")
    count = max(1, round(0.20 * len(ordered)))
    ranked = sorted(
        ordered,
        key=lambda item: hashlib.sha256(f"{seed}:{item}".encode()).digest(),
    )
    return set(ranked[:count])


def _load_rows(paths: Sequence[Path]):
    rows = []
    frames = defaultdict(list)
    sequence_ids = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("split") != "train":
                    raise ValueError(f"fusion training accepts train records only: {path}")
                sequence_id = str(record["sequence_id"])
                sequence_ids.add(sequence_id)
                frame_key = (sequence_id, int(record["frame_index"]))
                for candidate in record["candidates"]:
                    index = len(rows)
                    label = float(candidate["target_match_iou_0_5"])
                    rows.append((sequence_id, frame_key, feature_vector(candidate), label))
                    frames[frame_key].append(index)
    if not rows or not any(row[3] > 0.5 for row in rows):
        raise ValueError("candidate records contain no positive target detections")
    return rows, frames, sequence_ids


class _FusionNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layer2(torch.relu(self.layer1(value))).squeeze(1)


def _selection_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    frame_rows: Sequence[Sequence[int]],
    score_threshold: float,
    margin_threshold: float,
) -> dict[str, float]:
    true_positive = false_positive = false_negative = 0
    for indices in frame_rows:
        indices = np.asarray(indices, dtype=np.int64)
        order = indices[np.argsort(probabilities[indices])[::-1]]
        best = int(order[0])
        margin = (
            float(probabilities[best] - probabilities[int(order[1])])
            if len(order) > 1
            else 1.0
        )
        accepted = probabilities[best] >= score_threshold and margin >= margin_threshold
        has_target = bool(np.any(labels[indices] > 0.5))
        selected_target = accepted and labels[best] > 0.5
        true_positive += int(selected_target)
        false_positive += int(accepted and not selected_target)
        false_negative += int(has_target and not selected_target)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    beta_squared = 0.25
    f_beta = (
        (1.0 + beta_squared) * precision * recall
        / max(1e-12, beta_squared * precision + recall)
    )
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f0_5": f_beta,
    }


def _calibrate(
    probabilities,
    labels,
    frame_rows,
    precision_floor: float = 0.90,
    beta: float = 0.5,
):
    if not frame_rows:
        raise ValueError("candidate fusion calibration partition has no frames")
    best = None
    for score_threshold in np.linspace(0.70, 0.99, 30):
        for margin_threshold in np.linspace(0.0, 0.30, 16):
            metrics = _selection_metrics(
                probabilities,
                labels,
                frame_rows,
                float(score_threshold),
                float(margin_threshold),
            )
            beta_squared = beta * beta
            f_beta = (
                (1.0 + beta_squared) * metrics["precision"] * metrics["recall"]
                / max(
                    1e-12,
                    beta_squared * metrics["precision"] + metrics["recall"],
                )
            )
            metrics = dict(metrics)
            metrics.update(
                beta=float(beta),
                f_beta=float(f_beta),
                precision_floor=float(precision_floor),
                precision_floor_met=bool(
                    metrics["precision"] >= precision_floor
                ),
            )
            key = (
                metrics["precision"] >= precision_floor,
                f_beta,
                metrics["precision"],
                metrics["recall"],
            )
            if best is None or key > best[0]:
                best = (key, float(score_threshold), float(margin_threshold), metrics)
    return best[1], best[2], best[3]


def _calibration_partitions(rows, frames, calibration_sequences, lookup):
    partitions = {"tracking": [], "reacquisition": []}
    for frame_key, indices in frames.items():
        if frame_key[0] not in calibration_sequences:
            continue
        mapped = [lookup[index] for index in indices]
        if not mapped:
            continue
        global_search = bool(rows[indices[0]][2][-1] >= 0.5)
        mode = "reacquisition" if global_search else "tracking"
        partitions[mode].append(mapped)
    return partitions


def main() -> int:
    args = _arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    paths = _record_paths(args.records)
    rows, frames, sequence_ids = _load_rows(paths)
    calibration_sequences = _calibration_sequences(sequence_ids, args.seed)
    train_indices = [
        index for index, row in enumerate(rows) if row[0] not in calibration_sequences
    ]
    calibration_indices = [
        index for index, row in enumerate(rows) if row[0] in calibration_sequences
    ]
    train_x = np.stack([rows[index][2] for index in train_indices])
    train_y = np.asarray([rows[index][3] for index in train_indices], dtype=np.float32)
    calibration_x = np.stack([rows[index][2] for index in calibration_indices])
    calibration_y = np.asarray(
        [rows[index][3] for index in calibration_indices], dtype=np.float32
    )
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    train_x = (train_x - mean) / scale
    calibration_x = (calibration_x - mean) / scale

    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    model = _FusionNetwork(len(FEATURE_NAMES), args.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    positives = float(train_y.sum())
    negative_to_positive = float((len(train_y) - positives) / max(1.0, positives))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(min(20.0, negative_to_positive))
    )
    calibration_tensor = torch.from_numpy(calibration_x)
    calibration_target = torch.from_numpy(calibration_y)
    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        count = 0
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(features)
            count += len(features)
        model.eval()
        with torch.inference_mode():
            calibration_loss = float(
                criterion(model(calibration_tensor), calibration_target)
            )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total / max(1, count),
                "calibration_loss": calibration_loss,
            }
        )
        if calibration_loss < best_loss:
            best_loss = calibration_loss
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        calibration_probabilities = torch.sigmoid(model(calibration_tensor)).numpy()
    calibration_lookup = {row_index: offset for offset, row_index in enumerate(calibration_indices)}
    calibration_partitions = _calibration_partitions(
        rows,
        frames,
        calibration_sequences,
        calibration_lookup,
    )
    operating_point_policies = {
        # Continuity prevents a single conservative rejection from forcing the
        # tracker into the much harder global-search state.
        "tracking": {"precision_floor": 0.67, "beta": 1.0},
        # A false global reacquisition can permanently switch identities, so it
        # remains precision-weighted and more conservative.
        "reacquisition": {"precision_floor": 0.70, "beta": 0.5},
    }
    operating_points = {}
    for mode, policy in operating_point_policies.items():
        score_threshold, margin_threshold, metrics = _calibrate(
            calibration_probabilities,
            calibration_y,
            calibration_partitions[mode],
            precision_floor=policy["precision_floor"],
            beta=policy["beta"],
        )
        operating_points[mode] = {
            "score_threshold": score_threshold,
            "margin_threshold": margin_threshold,
            "precision_floor": policy["precision_floor"],
            "beta": policy["beta"],
            "calibration_frame_count": len(calibration_partitions[mode]),
            "calibration": metrics,
        }
    score_threshold = operating_points["reacquisition"]["score_threshold"]
    margin_threshold = operating_points["reacquisition"]["margin_threshold"]

    payload = {
        "schema_version": 2,
        "model_id": "frozen-detector-osnet-candidate-fusion-v2",
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "hidden_dim": args.hidden_dim,
        "weight1": model.layer1.weight.detach().numpy().tolist(),
        "bias1": model.layer1.bias.detach().numpy().tolist(),
        "weight2": model.layer2.weight.detach().numpy()[0].tolist(),
        "bias2": float(model.layer2.bias.detach().numpy()[0]),
        "score_threshold": score_threshold,
        "margin_threshold": margin_threshold,
        "operating_points": operating_points,
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "train_sequences": sorted(set(sequence_ids) - calibration_sequences),
            "calibration_sequences": sorted(calibration_sequences),
            "train_candidates": len(train_indices),
            "calibration_candidates": len(calibration_indices),
            "train_positive_candidates": int(train_y.sum()),
            "calibration_positive_candidates": int(calibration_y.sum()),
            "records": [
                {"path": str(path), "sha256": _sha256(path)} for path in paths
            ],
        },
    }
    report = {
        "schema_version": 2,
        "model_id": payload["model_id"],
        "best_calibration_loss": best_loss,
        "selection_calibration": {
            mode: point["calibration"] for mode, point in operating_points.items()
        },
        "score_threshold": score_threshold,
        "margin_threshold": margin_threshold,
        "operating_points": operating_points,
        "history": history,
        "training": payload["training"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["operating_points"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
