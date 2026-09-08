"""Build deterministic Phase 1 split-unit manifests without scanning media."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


DATASET_ORDER = ("intern_data_n1", "sage3d_extracted", "tpt_bench_clean_v2")
DEFAULT_ROOTS = {
    "intern_data_n1": Path("/h100-2/vln_n1/traj_data"),
    "sage3d_extracted": Path("/data/nfs/share/OmTrackVLA/data/sage3d_extracted"),
    "tpt_bench_clean_v2": Path("/data/nfs/share/OmTrackVLA/data/tpt_bench_clean_v2"),
}
DEFAULT_RATIOS = {"train": 0.80, "val": 0.10, "viz_val": 0.05, "test_locked": 0.05}


def _json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _intern_units(root: Path) -> list[str]:
    units = []
    for group in sorted(path for path in root.iterdir() if path.is_dir()):
        for scene in sorted(path for path in group.iterdir() if path.is_dir()):
            if (scene / "meta" / "episodes_stats.jsonl").is_file():
                units.append(f"{group.name}/{scene.name}")
    return units


def _sage_units(root: Path) -> list[str]:
    index = _json(root / "index.json")
    if not isinstance(index, dict) or not isinstance(index.get("eps"), list):
        raise ValueError("SAGE3D root index.json must contain an eps list")
    runs = {str(item["run"]) for item in index["eps"] if isinstance(item, dict) and item.get("run")}
    return sorted(runs)


def _tpt_units(root: Path) -> list[str]:
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and path.name.isdigit() and (path / "frames.parquet").is_file()
    )


def _partition(
    dataset_id: str,
    units: Iterable[str],
    seed: int,
    ratios: Mapping[str, float],
) -> dict[str, list[str]]:
    ordered_splits = tuple(DEFAULT_RATIOS)
    if tuple(ratios) != ordered_splits:
        raise ValueError(f"ratios must use the ordered keys {ordered_splits}")
    total_ratio = sum(float(ratios[name]) for name in ordered_splits)
    if abs(total_ratio - 1.0) > 1e-9 or any(float(ratios[name]) <= 0.0 for name in ordered_splits):
        raise ValueError("split ratios must be positive and sum to one")
    ranked = sorted(
        set(units),
        key=lambda unit: hashlib.sha256(f"{seed}:{dataset_id}:{unit}".encode()).hexdigest(),
    )
    count = len(ranked)
    boundaries = []
    cumulative = 0.0
    for split in ordered_splits[:-1]:
        cumulative += float(ratios[split])
        boundaries.append(int(count * cumulative))
    chunks = {}
    start = 0
    for split, end in zip(ordered_splits[:-1], boundaries):
        chunks[split] = sorted(ranked[start:end])
        start = end
    chunks[ordered_splits[-1]] = sorted(ranked[start:])
    return chunks


def build_phase1_manifest(
    roots: Mapping[str, Path] | None = None,
    seed: int = 20260907,
    ratios: Mapping[str, float] | None = None,
) -> dict[str, object]:
    roots = {key: Path(value) for key, value in (roots or DEFAULT_ROOTS).items()}
    ratios = dict(ratios or DEFAULT_RATIOS)
    missing = set(DATASET_ORDER) - set(roots)
    if missing:
        raise ValueError(f"missing dataset roots: {sorted(missing)}")
    enumerators = {
        "intern_data_n1": _intern_units,
        "sage3d_extracted": _sage_units,
        "tpt_bench_clean_v2": _tpt_units,
    }
    uses = {
        "intern_data_n1": ["phase1_geometry", "phase1_dynamics"],
        "sage3d_extracted": ["phase1_identity"],
        "tpt_bench_clean_v2": ["phase1_identity"],
    }
    datasets = {}
    for dataset_id in DATASET_ORDER:
        root = roots[dataset_id].expanduser().resolve(strict=True)
        units = enumerators[dataset_id](root)
        if not units:
            raise ValueError(f"no split units found for {dataset_id}: {root}")
        splits = _partition(dataset_id, units, seed, ratios)
        datasets[dataset_id] = {
            "root_hint": root.as_posix(),
            "split_unit": {
                "intern_data_n1": "group/scene",
                "sage3d_extracted": "run",
                "tpt_bench_clean_v2": "sequence",
            }[dataset_id],
            "allowed_uses": uses[dataset_id],
            "unit_count": len(units),
            "split_counts": {name: len(value) for name, value in splits.items()},
            "splits": splits,
        }
    payload: dict[str, object] = {
        "schema_version": 1,
        "manifest_id": "omtrackvla-phase1-splits-v1",
        "seed": int(seed),
        "partition_method": "per-dataset sha256 rank over immutable split units",
        "split_ratios": ratios,
        "media_scanned": False,
        "datasets": datasets,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def write_phase1_manifest(
    output: Path,
    manifest: Mapping[str, object],
    source_roots: Iterable[Path],
) -> None:
    destination = output.expanduser().resolve(strict=False)
    for root in source_roots:
        source = Path(root).expanduser().resolve(strict=True)
        try:
            destination.relative_to(source)
        except ValueError:
            continue
        raise ValueError(f"refusing to write a manifest inside source data: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
