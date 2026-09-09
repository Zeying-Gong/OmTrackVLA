"""Gate a trained Phase 2A fusion head against the frozen-front-end baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from omtrackvla.evaluation.pretrained_identity import _aggregate


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be a mapping: {path}")
    return value


def _baseline_aggregate(payload: Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(payload.get("dualop"), Mapping):
        value = payload["dualop"].get("aggregate")
    else:
        value = payload.get("aggregate")
    if not isinstance(value, Mapping):
        raise ValueError("baseline metrics do not contain an aggregate")
    return value


def _candidate_summaries(
    paths: Sequence[Path], expected_split: str
) -> list[dict[str, object]]:
    summaries = []
    for path in paths:
        payload = _load_json(path)
        if payload.get("split") != expected_split:
            raise ValueError(
                f"candidate split must be {expected_split}, got {payload.get('split')}"
            )
        values = payload.get("sequences")
        if not isinstance(values, list) or not values:
            raise ValueError(f"candidate metrics contain no sequences: {path}")
        summaries.extend(values)
    if len({str(item["sequence_id"]) for item in summaries}) != len(summaries):
        raise ValueError("candidate metrics contain duplicate sequences")
    return summaries


def evaluate_gate(
    config: Mapping[str, object],
    baseline: Mapping[str, object],
    candidate_summaries: Sequence[dict[str, object]],
) -> dict[str, object]:
    expected_sequences = [str(value) for value in config["expected_sequences"]]
    observed_sequences = sorted(
        str(value["sequence_id"]) for value in candidate_summaries
    )
    checks = []

    def check(name: str, value: object, operator: str, threshold: object) -> None:
        if operator == "eq":
            passed = value == threshold
        elif operator == "min":
            passed = float(value) >= float(threshold)
        elif operator == "max":
            passed = float(value) <= float(threshold)
        else:
            raise ValueError(operator)
        checks.append(
            {
                "name": name,
                "value": value,
                "operator": operator,
                "threshold": threshold,
                "passed": bool(passed),
            }
        )

    check("sequences", observed_sequences, "eq", sorted(expected_sequences))
    candidate = _aggregate(candidate_summaries)
    for name, expected in config["expected_counts"].items():
        check(name, candidate[name], "eq", expected)
        check(f"baseline_{name}", baseline[name], "eq", expected)

    improvement = float(candidate["end_to_end_success_iou_0_5"]) - float(
        baseline["end_to_end_success_iou_0_5"]
    )
    thresholds = config["thresholds"]
    check(
        "end_to_end_absolute_improvement",
        improvement,
        "min",
        thresholds["end_to_end_absolute_improvement"]["min"],
    )
    for name in (
        "output_precision_iou_0_5",
        "absent_false_positive_rate",
        "wrong_target_frames_iou_below_0_2",
        "reappearance_success_rate",
    ):
        rule = thresholds[name]
        operator = next(iter(rule))
        check(name, candidate[name], operator, rule[operator])

    return {
        "schema_version": 1,
        "gate_id": config["gate_id"],
        "split": config["split"],
        "test_locked_used": False,
        "baseline": dict(baseline),
        "candidate": candidate,
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }


def main() -> int:
    args = _arguments()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    baseline = _baseline_aggregate(_load_json(args.baseline))
    summaries = _candidate_summaries(args.candidate, str(config["split"]))
    result = evaluate_gate(config, baseline, summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
