#!/usr/bin/env python3
"""Build the audited 8-frame/10-Hz Architecture-v2 SAGE3D index."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omtrackvla.data.end_to_end_training import build_sage3d_sequence_index
from omtrackvla.models.end_to_end import ArchitectureV1Config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_index.read_text(encoding="utf-8"))
    result = build_sage3d_sequence_index(
        root=source["source_root"],
        sidecar_root=source["sidecar_root"],
        split_manifest=source["split_manifest"],
        policy_admission=source["policy_admission"],
        output=args.output,
        config=ArchitectureV1Config(history_size=8),
        sequence_steps=1,
        policy_stride=3,
        anchor_stride=3,
        history_stride_raw=3,
    )
    print(json.dumps({
        "status": "passed",
        "output": str(args.output),
        "counts": result["counts"],
        "history_size": result["history_size"],
        "history_stride_raw": result["history_stride_raw"],
        "test_locked_used": result["test_locked_used"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
