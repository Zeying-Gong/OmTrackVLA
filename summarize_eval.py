#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--expected", type=int, default=1405)
    args = parser.parse_args()

    files = sorted(
        p for p in args.result_dir.rglob("*.json")
        if not p.name.endswith("_info.json")
    )
    vals = []
    for path in files:
        with path.open("r") as fh:
            vals.append(json.load(fh))

    n = len(vals)
    if n == 0:
        print(f"{args.result_dir}: no result json files found")
        return

    sr = sum(float(x.get("success", 0.0)) for x in vals) / n * 100.0
    tr = sum(float(x.get("following_rate", 0.0)) for x in vals) / n * 100.0
    cr = sum(float(x.get("collision", 0.0)) for x in vals) / n * 100.0
    done = "complete" if n == args.expected else f"partial {n}/{args.expected}"
    print(f"{args.result_dir} [{done}]")
    print(f"SR / TR / CR = {sr:.2f} / {tr:.2f} / {cr:.2f}")


if __name__ == "__main__":
    main()
