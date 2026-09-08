#!/usr/bin/env python3
"""Build repaired SAGE3D bbox/visibility labels outside the source dataset."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from omtrackvla.data.sage3d_sidecar import (
    GENERATION_SPEC_ID,
    SIDECAR_SCHEMA_VERSION,
    depth_visibility_evidence,
    episode_sidecar_path,
    project_target_bbox,
    safe_source_path,
    sha256_file,
    sha256_json,
    validate_episode_sidecar,
)


DEFAULT_ROOT = Path("/data/nfs/share/OmTrackVLA/data/sage3d_extracted")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-paths",
        type=Path,
        help="optional text file with one episode source_path per line",
    )
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--person-height-m", type=float, default=1.7)
    parser.add_argument("--person-width-m", type=float, default=0.5)
    parser.add_argument("--depth-scale-to-m", type=float, default=0.001)
    parser.add_argument("--absolute-depth-tolerance-m", type=float, default=0.30)
    parser.add_argument("--relative-depth-tolerance", type=float, default=0.15)
    parser.add_argument("--minimum-depth-support", type=float, default=0.20)
    parser.add_argument("--maximum-near-occluder", type=float, default=0.50)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def generation_config(args: argparse.Namespace) -> dict[str, float]:
    return {
        "person_height_m": float(args.person_height_m),
        "person_width_m": float(args.person_width_m),
        "depth_scale_to_m": float(args.depth_scale_to_m),
        "absolute_depth_tolerance_m": float(args.absolute_depth_tolerance_m),
        "relative_depth_tolerance": float(args.relative_depth_tolerance),
        "minimum_depth_support": float(args.minimum_depth_support),
        "maximum_near_occluder": float(args.maximum_near_occluder),
        "depth_inner_margin_x": 0.20,
        "depth_inner_margin_y": 0.15,
    }


def accepted_entries(data_root: Path) -> list[dict[str, object]]:
    index = load_json(data_root / "index.json")
    if not isinstance(index, dict) or not isinstance(index.get("eps"), list):
        raise ValueError("SAGE3D index.json must contain eps")
    entries = []
    for entry in index["eps"]:
        source_path = str(entry["path"])
        episode = safe_source_path(data_root, source_path)
        if not (episode / "_ACCEPTED").is_file():
            continue
        quality = load_json(episode / "quality.json")
        if not isinstance(quality, dict) or quality.get("status") != "accepted":
            continue
        entries.append(dict(entry))
    entries.sort(key=lambda item: str(item["path"]))
    return entries


def select_entries(
    entries: Sequence[dict[str, object]], source_paths: Path | None, maximum: int | None
) -> tuple[list[dict[str, object]], str]:
    selection = "full"
    selected = list(entries)
    if source_paths is not None:
        requested = {
            line.strip()
            for line in source_paths.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        known = {str(entry["path"]) for entry in entries}
        unknown = requested - known
        if unknown:
            raise ValueError(f"source-path list contains {len(unknown)} unknown episodes")
        selected = [entry for entry in entries if str(entry["path"]) in requested]
        selection = "subset"
    if maximum is not None:
        selected = selected[: max(0, int(maximum))]
        selection = "subset"
    if not selected:
        raise ValueError("no accepted SAGE3D episodes selected")
    return selected, selection


def build_episode(
    data_root: Path,
    output_root: Path,
    entry: dict[str, object],
    config: dict[str, float],
    resume: bool,
) -> dict[str, object]:
    source_path = str(entry["path"])
    episode = safe_source_path(data_root, source_path)
    derived_path = episode / "derived.json"
    camera_path = episode / "camera_info.json"
    derived_sha256 = sha256_file(derived_path)
    camera_sha256 = sha256_file(camera_path)
    config_sha256 = sha256_json(config)
    destination = episode_sidecar_path(output_root, source_path)
    if resume and destination.is_file():
        try:
            existing = validate_episode_sidecar(load_json(destination), source_path)
            if (
                existing.get("source_derived_sha256") == derived_sha256
                and existing.get("source_camera_info_sha256") == camera_sha256
                and existing.get("generation_config_sha256") == config_sha256
            ):
                counts = Counter(
                    str(step["visibility_reason"]) for step in existing["steps"]
                )
                return {
                    "source_path": source_path,
                    "steps": len(existing["steps"]),
                    "visible_steps": int(counts["visible"]),
                    "counts": counts,
                    "sidecar_sha256": sha256_file(destination),
                    "resumed": True,
                }
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass

    derived = load_json(derived_path)
    camera_info = load_json(camera_path)
    if not isinstance(derived, dict) or not isinstance(derived.get("steps"), list):
        raise ValueError(f"invalid derived.json: {source_path}")
    if not isinstance(camera_info, dict):
        raise ValueError(f"invalid camera_info.json: {source_path}")
    output_steps = []
    counts: Counter[str] = Counter()
    for source_step in derived["steps"]:
        projection = project_target_bbox(
            source_step,
            camera_info,
            person_height_m=config["person_height_m"],
            person_width_m=config["person_width_m"],
        )
        bbox = projection["bbox_xyxy"]
        evidence = None
        if bbox is None:
            visible = False
            reason = str(projection["projection_reason"])
        else:
            frame_id = int(source_step["step"])
            depth_path = episode / "depth" / f"{frame_id:05d}.png"
            if not depth_path.is_file():
                visible = False
                reason = "missing_depth"
            else:
                with Image.open(depth_path) as image:
                    raw_depth = np.asarray(image)
                evidence = depth_visibility_evidence(
                    raw_depth,
                    bbox,
                    float(projection["expected_depth_m"]),
                    depth_scale_to_m=config["depth_scale_to_m"],
                    inner_margin_x=config["depth_inner_margin_x"],
                    inner_margin_y=config["depth_inner_margin_y"],
                    absolute_tolerance_m=config["absolute_depth_tolerance_m"],
                    relative_tolerance=config["relative_depth_tolerance"],
                    minimum_support_fraction=config["minimum_depth_support"],
                    maximum_near_fraction=config["maximum_near_occluder"],
                )
                visible = bool(evidence["visible"])
                reason = str(evidence["visibility_reason"])
        counts[reason] += 1
        output_steps.append(
            {
                "step": int(source_step["step"]),
                "bbox_xyxy": [float(value) for value in bbox] if visible else None,
                "projected_bbox_xyxy": (
                    [float(value) for value in bbox] if bbox is not None else None
                ),
                "raw_projected_bbox_xyxy": projection["raw_bbox_xyxy"],
                "expected_depth_m": float(projection["expected_depth_m"]),
                "visible": visible,
                "visibility_reason": reason,
                "depth_evidence": evidence,
            }
        )
    value = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "dataset_id": "sage3d_extracted",
        "generation_spec_id": GENERATION_SPEC_ID,
        "generation_config_sha256": config_sha256,
        "source_path": source_path,
        "source_derived_sha256": derived_sha256,
        "source_camera_info_sha256": camera_sha256,
        "steps": output_steps,
        "summary": {"steps": len(output_steps), "visibility_reason_counts": dict(counts)},
    }
    write_json_atomic(destination, value)
    return {
        "source_path": source_path,
        "steps": len(output_steps),
        "visible_steps": int(counts["visible"]),
        "counts": counts,
        "sidecar_sha256": sha256_file(destination),
        "resumed": False,
    }


def main() -> None:
    args = arguments()
    data_root = args.data_root.expanduser().resolve(strict=True)
    output_root = args.output_dir.expanduser().resolve(strict=False)
    try:
        output_root.relative_to(data_root)
    except ValueError:
        pass
    else:
        raise ValueError("sidecar output must be outside the immutable source root")
    output_root.mkdir(parents=True, exist_ok=True)
    entries = accepted_entries(data_root)
    selected, selection = select_entries(entries, args.source_paths, args.max_episodes)
    config = generation_config(args)
    totals: Counter[str] = Counter()
    episode_summaries: dict[str, dict[str, object]] = {}
    failures: list[dict[str, str]] = []
    completed = 0
    resumed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(
                build_episode, data_root, output_root, entry, config, bool(args.resume)
            ): str(entry["path"])
            for entry in selected
        }
        for future in as_completed(futures):
            source_path = futures[future]
            try:
                result = future.result()
                totals.update(result["counts"])
                episode_summaries[str(result["source_path"])] = {
                    "steps": int(result["steps"]),
                    "visible_steps": int(result["visible_steps"]),
                    "sidecar_sha256": str(result["sidecar_sha256"]),
                }
                completed += 1
                resumed += int(bool(result["resumed"]))
            except Exception as error:  # preserve all episode failures in manifest
                failures.append(
                    {"source_path": source_path, "error": f"{type(error).__name__}: {error}"}
                )
            if (completed + len(failures)) % 25 == 0:
                print(
                    f"processed {completed + len(failures)}/{len(selected)} episodes "
                    f"({len(failures)} failures)",
                    flush=True,
                )
    source_paths = [str(entry["path"]) for entry in selected]
    manifest = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "dataset_id": "sage3d_extracted",
        "generation_spec_id": GENERATION_SPEC_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root_hint": str(data_root),
        "source_index_sha256": sha256_file(data_root / "index.json"),
        "generation_config": config,
        "generation_config_sha256": sha256_json(config),
        "selection": selection,
        "accepted_source_episode_count": len(entries),
        "selected_episode_count": len(selected),
        "completed_episode_count": completed,
        "resumed_episode_count": resumed,
        "failed_episode_count": len(failures),
        "source_paths_sha256": sha256_json(source_paths),
        "episodes": {
            key: episode_summaries[key] for key in sorted(episode_summaries)
        },
        "total_step_count": int(sum(totals.values())),
        "visibility_reason_counts": dict(sorted(totals.items())),
        "failures": failures,
    }
    write_json_atomic(output_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
