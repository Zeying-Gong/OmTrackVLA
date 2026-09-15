#!/usr/bin/env python3
"""Build the audited non-locked SAGE3D sequence index for NEXT-026."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from omtrackvla.data.end_to_end_training import build_sage3d_sequence_index


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--policy-admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-steps", type=int, default=2)
    parser.add_argument("--policy-stride", type=int, default=3)
    parser.add_argument("--anchor-stride", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    report = build_sage3d_sequence_index(
        root=args.root,
        sidecar_root=args.sidecar_root,
        split_manifest=args.split_manifest,
        policy_admission=args.policy_admission,
        output=args.output,
        sequence_steps=args.sequence_steps,
        policy_stride=args.policy_stride,
        anchor_stride=args.anchor_stride,
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "output": str(args.output),
                "index_sha256": report["index_sha256"],
                "counts": report["counts"],
                "skipped": report["skipped"],
                "test_locked_used": report["test_locked_used"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
