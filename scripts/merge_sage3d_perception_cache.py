#!/usr/bin/env python3
"""Merge complete frozen-perception shard manifests without copying cache files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from omtrackvla.data.sage3d_policy import sha256_file


INVARIANT_KEYS = (
    "schema_version",
    "status",
    "dataset_id",
    "split",
    "policy_spec_id",
    "source_index_sha256",
    "sidecar_manifest_sha256",
    "policy_admission_sha256",
    "split_manifest_sha256",
    "front_end",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_manifests(
    manifests: Sequence[Mapping[str, object]],
    shard_relative_dirs: Sequence[str],
    *,
    allow_partial: bool = False,
) -> dict[str, object]:
    if not manifests or len(manifests) != len(shard_relative_dirs):
        raise ValueError("cache merge requires one relative directory per shard manifest")
    reference = manifests[0]
    for index, manifest in enumerate(manifests):
        for key in INVARIANT_KEYS:
            if manifest.get(key) != reference.get(key):
                raise ValueError(f"perception shard invariant mismatch: {key}")
        selection = manifest.get("selection")
        if not isinstance(selection, Mapping):
            raise ValueError("perception shard selection is missing")
        if selection.get("test_locked_used") is not False:
            raise ValueError("perception shard consumed test_locked")
        if not allow_partial and (
            selection.get("max_units") is not None
            or selection.get("max_episodes") is not None
        ):
            raise ValueError("formal cache merge refuses development-limited shards")
        if int(selection.get("num_shards", -1)) != len(manifests):
            raise ValueError("perception shard count mismatch")
        if int(selection.get("shard_index", -1)) != index:
            raise ValueError("perception shard index mismatch")
        if (
            selection.get("max_units") != reference["selection"].get("max_units")
            or selection.get("max_episodes")
            != reference["selection"].get("max_episodes")
        ):
            raise ValueError("perception shard development limits mismatch")

    episodes = {}
    skipped = []
    requested = 0
    cached = 0
    stride = None
    unit_count = None
    maximum_units = reference["selection"].get("max_units")
    maximum_episodes = reference["selection"].get("max_episodes")
    for manifest, relative_dir in zip(manifests, shard_relative_dirs):
        pure_dir = PurePosixPath(relative_dir)
        if pure_dir.is_absolute() or ".." in pure_dir.parts:
            raise ValueError(f"unsafe shard-relative directory: {relative_dir}")
        selection = manifest["selection"]
        current_stride = int(selection["record_stride"])
        current_units = int(selection["unit_count"])
        stride = current_stride if stride is None else stride
        unit_count = current_units if unit_count is None else unit_count
        if current_stride != stride or current_units != unit_count:
            raise ValueError("perception shard selection policy mismatch")
        requested += int(selection["requested_episodes"])
        cached += int(selection["cached_episodes"])
        for source_path, metadata in manifest["episodes"].items():
            if source_path in episodes:
                raise ValueError(f"duplicate episode across perception shards: {source_path}")
            merged_metadata = dict(metadata)
            merged_metadata["cache_path"] = (
                pure_dir / PurePosixPath(str(metadata["cache_path"]))
            ).as_posix()
            episodes[str(source_path)] = merged_metadata
        for value in manifest.get("skipped", ()):
            skipped.append(dict(value))
    if cached != len(episodes) or not episodes:
        raise ValueError("merged perception episode count mismatch")
    return {
        **{key: reference[key] for key in INVARIANT_KEYS},
        "selection": {
            "unit_count": unit_count,
            "requested_episodes": requested,
            "cached_episodes": cached,
            "record_stride": stride,
            "max_units": maximum_units,
            "max_episodes": maximum_episodes,
            "num_shards": 1,
            "shard_index": 0,
            "merged_from_shards": len(manifests),
            "test_locked_used": False,
        },
        "episodes": episodes,
        "skipped": skipped,
    }


def main() -> int:
    args = _arguments()
    if args.num_shards <= 1:
        raise ValueError("cache merge requires at least two shards")
    shard_root = args.shard_root.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve(strict=False)
    manifests = []
    relative_dirs = []
    for index in range(args.num_shards):
        name = f"shard-{index:03d}-of-{args.num_shards:03d}"
        path = shard_root / name / "manifest.json"
        value = _load(path)
        if not isinstance(value, dict):
            raise ValueError(f"perception shard manifest is invalid: {path}")
        for metadata in value.get("episodes", {}).values():
            cache_path = shard_root / name / PurePosixPath(str(metadata["cache_path"]))
            if not cache_path.is_file() or sha256_file(cache_path) != metadata["cache_sha256"]:
                raise ValueError(f"perception shard cache is missing or changed: {cache_path}")
        manifests.append(value)
        relative_dirs.append((shard_root / name).relative_to(output).as_posix())
    merged = merge_manifests(manifests, relative_dirs, allow_partial=args.allow_partial)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "manifest.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps(merged["selection"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
