"""Read-only hash audit for artifacts referenced by a frozen JSON plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_rows(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if {"path", "sha256"} <= set(value) and isinstance(value["path"], str):
            yield value
        for child in value.values():
            yield from _artifact_rows(child)
    elif isinstance(value, list):
        for child in value:
            yield from _artifact_rows(child)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    checked: dict[str, dict[str, Any]] = {}
    for reference in _artifact_rows(plan):
        path = Path(reference["path"])
        key = str(path)
        if key in checked:
            continue
        if "test_locked" in key.lower():
            raise RuntimeError("refusing to read test_locked artifact: " + key)
        row: dict[str, Any] = {
            "path": key,
            "expected_sha256": reference["sha256"],
            "expected_bytes": reference.get("bytes"),
            "exists": path.is_file(),
        }
        if path.is_file():
            row.update(current_sha256=_sha256(path), current_bytes=path.stat().st_size)
            row["matches"] = (
                row["current_sha256"] == row["expected_sha256"]
                and (
                    row["expected_bytes"] is None
                    or row["current_bytes"] == row["expected_bytes"]
                )
            )
        else:
            row["matches"] = False
        checked[key] = row
    mismatches = [row for row in checked.values() if not row["matches"]]
    print(
        json.dumps(
            {
                "plan": str(args.plan),
                "artifact_count": len(checked),
                "matching_count": len(checked) - len(mismatches),
                "mismatch_count": len(mismatches),
                "mismatches": mismatches,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
