"""Enforce frozen per-phase thresholds against metrics.json."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _lookup(value: object, dotted: str) -> object:
    current = value
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted)
        current = current[part]
    return current


def evaluate_thresholds(metrics: dict[str, object], config: dict[str, object]) -> dict[str, object]:
    checks = []
    passed = True
    for path, bounds in config["thresholds"].items():
        try:
            value = _lookup(metrics, path)
            numeric = float(value)
            finite = math.isfinite(numeric)
        except (KeyError, TypeError, ValueError):
            value = None
            numeric = float("nan")
            finite = False
        ok = finite
        if finite and "min" in bounds:
            ok = ok and numeric >= float(bounds["min"])
        if finite and "max" in bounds:
            ok = ok and numeric <= float(bounds["max"])
        checks.append({"metric": path, "value": value, "bounds": bounds, "passed": ok})
        passed = passed and ok
    return {
        "schema_version": 1,
        "phase": int(config.get("phase", metrics.get("phase", 1))),
        "gate_id": config["gate_id"],
        "gate_scope": config.get("gate_scope", "phase_exit"),
        "passed": passed,
        "checks": checks,
    }


def main() -> int:
    args = _arguments()
    if args.phase not in (1, 2):
        raise ValueError("this gate currently implements Phase 1 and Phase 2 only")
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with args.metrics.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if config.get("phase") != args.phase or metrics.get("phase") != args.phase:
        raise ValueError("gate, metrics, and --phase do not match")
    result = evaluate_thresholds(metrics, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
