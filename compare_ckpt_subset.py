#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_MODELS = ("official", "step445000", "step450000", "step455000")
METRICS = ("success", "following_rate", "collision")


def load_results(directory: Path) -> dict[str, dict]:
    results = {}
    for path in directory.glob("*/*.json"):
        if path.name.endswith("_info.json"):
            continue
        try:
            item = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(item, dict) and "success" in item:
            results[path.relative_to(directory).as_posix()] = item
    return results


def aggregate(items: list[dict]) -> tuple[float, float, float]:
    return tuple(
        sum(float(item.get(metric, 0.0)) for item in items) * 100.0 / len(items)
        for metric in METRICS
    )


def paired_success(left: dict[str, dict], right: dict[str, dict], keys: set[str]) -> tuple[int, int, float]:
    right_wins = sum(
        float(right[key].get("success", 0.0)) > float(left[key].get("success", 0.0))
        for key in keys
    )
    left_wins = sum(
        float(left[key].get("success", 0.0)) > float(right[key].get("success", 0.0))
        for key in keys
    )
    discordant = right_wins + left_wins
    if discordant == 0:
        return right_wins, left_wins, 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(right_wins, left_wins) + 1))
    p_value = min(1.0, 2.0 * tail / (2**discordant))
    return right_wins, left_wins, p_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("stt", "dt", "at"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected", type=int, default=128)
    parser.add_argument("--root", type=Path, default=Path("sim_data/eval"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    args = parser.parse_args()

    runs = {}
    models = tuple(args.models)
    for model in models:
        directory = args.root / f"subset{args.expected}_{args.task}_{model}_{args.run_id}"
        if not directory.is_dir():
            raise SystemExit(f"Missing result directory: {directory}")
        runs[model] = load_results(directory)

    common = set.intersection(*(set(run) for run in runs.values()))
    print(f"task={args.task} expected={args.expected} common={len(common)}")
    print("model         N       SR       TR       CR")
    for model in models:
        values = [runs[model][key] for key in sorted(common)]
        if not values:
            print(f"{model:12s} {len(runs[model]):4d}  no common episodes")
            continue
        sr, tr, cr = aggregate(values)
        print(f"{model:12s} {len(values):4d}  {sr:7.2f}  {tr:7.2f}  {cr:7.2f}")

    print("\npaired comparisons on common episodes (right - left)")
    pairs = tuple(zip(models, models[1:]))
    for left, right in pairs:
        left_values = [runs[left][key] for key in sorted(common)]
        right_values = [runs[right][key] for key in sorted(common)]
        left_metrics = aggregate(left_values)
        right_metrics = aggregate(right_values)
        deltas = tuple(r - l for l, r in zip(left_metrics, right_metrics))
        right_wins, left_wins, p_value = paired_success(runs[left], runs[right], common)
        print(
            f"{left:12s} -> {right:12s} "
            f"dSR={deltas[0]:+6.2f} dTR={deltas[1]:+6.2f} dCR={deltas[2]:+6.2f} "
            f"SR_wins={right_wins}:{left_wins} exact_p={p_value:.4f}"
        )

    for model in models:
        if len(runs[model]) != args.expected:
            print(f"WARNING: {model} has {len(runs[model])}/{args.expected} valid result JSONs")


if __name__ == "__main__":
    main()
