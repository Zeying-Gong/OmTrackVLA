"""Read-only admission audit for SAGE3D Phase 2 policy trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Mapping

import numpy as np

from omtrackvla.data.sage3d_policy import (
    CLOCK_SPEC_ID,
    CONTROL_HZ,
    POLICY_SPEC_ID,
    TRANSFORM_SPEC_ID,
    WAYPOINT_HORIZON,
    WAYPOINT_STRIDE_STEPS,
    canonical_waypoints,
    moving_speed_ratio,
    source_waypoints,
    target_position_base,
    waypoint_time_offsets_s,
)


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe SAGE3D path: {relative}")
    path = (root / Path(*pure.parts)).resolve(strict=False)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"SAGE3D path escapes source root: {relative}") from error
    return path


def _manifest(path: Path) -> dict[str, object]:
    value = _load(path)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("manifest must use schema_version=1")
    unsigned = dict(value)
    expected = unsigned.pop("manifest_sha256", None)
    actual = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected != actual:
        raise ValueError("manifest checksum mismatch")
    return value


def _episode_metrics(episode: Path) -> dict[str, object]:
    derived = _load(episode / "derived.json")
    quality = _load(episode / "quality.json")
    if not isinstance(derived, Mapping) or not isinstance(derived.get("steps"), list):
        raise ValueError("invalid derived.json")
    if not isinstance(quality, Mapping) or quality.get("status") != "accepted":
        raise ValueError("episode is not canonically accepted")
    steps = derived["steps"]
    if len(steps) < 2:
        raise ValueError("episode has fewer than two steps")
    step_ids = [int(step["step"]) for step in steps]
    contiguous = all(right == left + 1 for left, right in zip(step_ids, step_ids[1:]))
    target_errors = [
        float(np.max(np.abs(np.asarray(target_position_base(step)) - step["target_local"])))
        for step in steps
    ]
    source_errors = []
    contract_anchor_errors = []
    contract_displacements = []
    source_count = 0
    contract_count = 0
    for anchor, step in enumerate(steps):
        recorded = step.get("waypoints_ego")
        if recorded is not None:
            expected = source_waypoints(steps, anchor)
            source_errors.append(
                float(np.max(np.abs(np.asarray(expected) - np.asarray(recorded))))
            )
            source_count += 1
        try:
            contract = np.asarray(canonical_waypoints(steps, anchor))
        except ValueError:
            continue
        contract_anchor_errors.append(float(np.linalg.norm(contract[0])))
        contract_displacements.append(float(np.max(np.linalg.norm(contract, axis=1))))
        contract_count += 1
    speed = quality.get("filter_config", {}).get("character_speed")
    if speed is None:
        speed = quality.get("metrics", {}).get("character_speed")
    speed_ratio = moving_speed_ratio(steps, float(speed)) if speed is not None else None
    return {
        "step_count": len(steps),
        "contiguous_steps": contiguous,
        "source_waypoint_count": source_count,
        "contract_waypoint_count": contract_count,
        "target_local_max_error_m": max(target_errors, default=float("inf")),
        "source_waypoint_max_error_m": max(source_errors, default=float("inf")),
        "contract_anchor_max_error_m": max(contract_anchor_errors, default=float("inf")),
        "contract_horizon_max_displacement_m": max(
            contract_displacements, default=float("inf")
        ),
        "moving_speed_ratio": speed_ratio,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--split", action="append", dest="splits", required=True)
    parser.add_argument("--episodes-per-split", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    root = args.data_root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("audit output must be outside the read-only source root")
    manifest = _manifest(args.manifest)
    gate = _load(args.gate)
    if not isinstance(gate, Mapping) or gate.get("policy_spec_id") != POLICY_SPEC_ID:
        raise ValueError("policy gate does not match the implemented policy spec")
    allowed_splits = set(gate["allowed_splits"])
    if any(split == "test_locked" or split not in allowed_splits for split in args.splits):
        raise ValueError("only declared non-locked development splits may be audited")
    if args.episodes_per_split <= 0:
        raise ValueError("episodes-per-split must be positive")

    source_index_path = root / "index.json"
    source_index = _load(source_index_path)
    entries = list(source_index["eps"])
    records = []
    errors = []
    strata = Counter()
    rng = random.Random(args.seed)
    split_units = manifest["datasets"]["sage3d_extracted"]["splits"]
    for split in args.splits:
        runs = set(split_units[split])
        candidates = [entry for entry in entries if str(entry["run"]) in runs]
        candidates.sort(key=lambda entry: str(entry["path"]))
        rng.shuffle(candidates)
        selected = candidates[: args.episodes_per_split]
        for entry in selected:
            relative = str(entry["path"])
            episode = _safe_path(root, relative)
            try:
                metrics = _episode_metrics(episode)
            except Exception as error:  # report every malformed source record
                errors.append({"split": split, "path": relative, "error": str(error)})
                continue
            record = {
                "split": split,
                "path": relative,
                "mode": str(entry["mode"]),
                "camera": str(entry["cam"]),
                **metrics,
            }
            records.append(record)
            strata[(str(entry["mode"]), str(entry["cam"]))] += 1

    ratios = [float(record["moving_speed_ratio"]) for record in records if record["moving_speed_ratio"] is not None]
    aggregate = {
        "audited_episodes": len(records),
        "invalid_episodes": len(errors),
        "strata": len(strata),
        "noncontiguous_step_episodes": sum(
            not bool(record["contiguous_steps"]) for record in records
        ),
        "target_local_max_error_m": max(
            (float(record["target_local_max_error_m"]) for record in records),
            default=float("inf"),
        ),
        "source_waypoint_max_error_m": max(
            (float(record["source_waypoint_max_error_m"]) for record in records),
            default=float("inf"),
        ),
        "contract_anchor_max_error_m": max(
            (float(record["contract_anchor_max_error_m"]) for record in records),
            default=float("inf"),
        ),
        "contract_horizon_max_displacement_m": max(
            (float(record["contract_horizon_max_displacement_m"]) for record in records),
            default=float("inf"),
        ),
        "speed_ratio_coverage": len(ratios) / max(1, len(records)),
        "median_moving_speed_ratio": float(np.median(ratios)) if ratios else None,
        "speed_ratio_p10": float(np.quantile(ratios, 0.1)) if ratios else None,
        "speed_ratio_p90": float(np.quantile(ratios, 0.9)) if ratios else None,
    }
    checks = {
        "minimum_audited_episodes": aggregate["audited_episodes"]
        >= int(gate["minimum_audited_episodes"]),
        "minimum_strata": aggregate["strata"] >= int(gate["minimum_strata"]),
        "maximum_invalid_episodes": aggregate["invalid_episodes"]
        <= int(gate["maximum_invalid_episodes"]),
        "contiguous_steps": aggregate["noncontiguous_step_episodes"]
        <= int(gate["maximum_noncontiguous_step_episodes"]),
        "target_local_transform": aggregate["target_local_max_error_m"]
        <= float(gate["maximum_target_local_error_m"]),
        "source_waypoint_reconstruction": aggregate["source_waypoint_max_error_m"]
        <= float(gate["maximum_source_waypoint_error_m"]),
        "contract_anchor": aggregate["contract_anchor_max_error_m"]
        <= float(gate["maximum_contract_anchor_error_m"]),
        "clock_speed_lower": aggregate["median_moving_speed_ratio"] is not None
        and aggregate["median_moving_speed_ratio"]
        >= float(gate["minimum_median_speed_ratio"]),
        "clock_speed_upper": aggregate["median_moving_speed_ratio"] is not None
        and aggregate["median_moving_speed_ratio"]
        <= float(gate["maximum_median_speed_ratio"]),
        "clock_speed_coverage": aggregate["speed_ratio_coverage"]
        >= float(gate["minimum_speed_ratio_coverage"]),
        "horizon_displacement": aggregate["contract_horizon_max_displacement_m"]
        <= float(gate["maximum_contract_horizon_displacement_m"]),
    }
    result = {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "policy_spec_id": POLICY_SPEC_ID,
        "transform_spec_id": TRANSFORM_SPEC_ID,
        "clock_spec_id": CLOCK_SPEC_ID,
        "source_index_sha256": _sha256(source_index_path),
        "manifest_sha256": _sha256(args.manifest),
        "gate_sha256": _sha256(args.gate),
        "selection": {
            "splits": args.splits,
            "episodes_per_split": args.episodes_per_split,
            "seed": args.seed,
            "test_locked_used": False,
        },
        "trajectory_contract": {
            "control_hz": CONTROL_HZ,
            "horizon_points": WAYPOINT_HORIZON,
            "stride_steps": WAYPOINT_STRIDE_STEPS,
            "time_offsets_s": waypoint_time_offsets_s(),
            "source_offsets_steps": [1 + point * WAYPOINT_STRIDE_STEPS for point in range(WAYPOINT_HORIZON)],
            "canonical_offsets_steps": [point * WAYPOINT_STRIDE_STEPS for point in range(WAYPOINT_HORIZON)],
        },
        "aggregate": aggregate,
        "checks": checks,
        "strata": {f"{mode}/{camera}": count for (mode, camera), count in sorted(strata.items())},
        "errors": errors,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], **aggregate}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
