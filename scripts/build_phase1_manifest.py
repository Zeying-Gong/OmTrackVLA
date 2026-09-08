#!/usr/bin/env python3
"""Create fixed Phase 1 train/val/viz_val/test_locked split-unit lists."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from omtrackvla.data.phase1_manifest import DEFAULT_ROOTS, build_phase1_manifest, write_phase1_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intern-root", type=Path, default=DEFAULT_ROOTS["intern_data_n1"])
    parser.add_argument("--sage-root", type=Path, default=DEFAULT_ROOTS["sage3d_extracted"])
    parser.add_argument("--tpt-root", type=Path, default=DEFAULT_ROOTS["tpt_bench_clean_v2"])
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = {
        "intern_data_n1": args.intern_root,
        "sage3d_extracted": args.sage_root,
        "tpt_bench_clean_v2": args.tpt_root,
    }
    manifest = build_phase1_manifest(roots=roots, seed=args.seed)
    write_phase1_manifest(args.output, manifest, roots.values())
    print(json.dumps({"output": str(args.output), "datasets": {
        key: value["split_counts"] for key, value in manifest["datasets"].items()
    }}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
