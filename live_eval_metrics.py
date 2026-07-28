#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()

    count = 0
    success = 0.0
    tracking = 0.0
    collision = 0.0
    invalid = 0
    for path in args.result_dir.rglob("*.json"):
        if path.name.endswith("_info.json"):
            continue
        try:
            with path.open("r") as handle:
                result = json.load(handle)
        except (OSError, json.JSONDecodeError):
            invalid += 1
            continue
        count += 1
        success += as_float(result.get("success", 0.0))
        tracking += as_float(result.get("following_rate", 0.0))
        collision += as_float(result.get("collision", 0.0))

    if count:
        sr = 100.0 * success / count
        tr = 100.0 * tracking / count
        cr = 100.0 * collision / count
        metrics = f"SR={sr:.2f} TR={tr:.2f} CR={cr:.2f}"
    else:
        metrics = "SR=-- TR=-- CR=--"
    print(
        f"metrics_episodes={count}/{args.expected} {metrics}"
        f" invalid_jsons={invalid}"
    )


if __name__ == "__main__":
    main()
