#!/usr/bin/env python3
"""Read-only, reproducible audit for OmTrackVLA's three external datasets."""

from __future__ import annotations

import argparse
import binascii
import json
import math
import os
import statistics
import struct
import sys
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_VERSION = 1
DATASET_IDS = ("intern_data_n1", "sage3d_extracted", "tpt_bench_clean_v2")
DEFAULT_ROOTS = {
    "intern_data_n1": Path("/h100-2/vln_n1/traj_data"),
    "sage3d_extracted": Path("/data/nfs/share/OmTrackVLA/data/sage3d_extracted"),
    "tpt_bench_clean_v2": Path("/data/nfs/share/OmTrackVLA/data/tpt_bench_clean_v2"),
}
LANGUAGE_KEYS = {
    "desc",
    "instruction",
    "revised_sub_instruction",
    "sub_instruction",
    "sum_instruction",
    "task",
    "tasks",
}


def issue(severity: str, code: str, message: str, path: Path | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"severity": severity, "code": code, "message": message}
    if path is not None:
        value["path"] = str(path)
    return value


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield value


def sample_positions(length: int, count: int) -> list[int]:
    """Return deterministic, approximately even indices including both ends."""
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    return sorted({round(i * (length - 1) / (count - 1)) for i in range(count)})


def stratified_samples(
    values: Iterable[Any], key, count_per_stratum: int
) -> list[Any]:
    strata: dict[Any, list[Any]] = defaultdict(list)
    for value in values:
        strata[key(value)].append(value)
    selected = []
    for stratum in sorted(strata, key=str):
        choices = sorted(strata[stratum], key=str)
        selected.extend(choices[index] for index in sample_positions(len(choices), count_per_stratum))
    return selected


def nested_language_keys(value: Any) -> list[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in LANGUAGE_KEYS:
                found.add(key)
            found.update(nested_language_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(nested_language_keys(child))
    return sorted(found)


def sage_is_canonically_accepted(
    indexed: bool, accepted_marker: bool, quality_status: str | None
) -> bool:
    return indexed and accepted_marker and quality_status == "accepted"


def elapsed_ratio(video_ms: Iterable[float], reference_ms: Iterable[float]) -> float | None:
    video = list(video_ms)
    reference = list(reference_ms)
    if len(video) < 2 or len(reference) < 2:
        return None
    video_elapsed = float(video[-1]) - float(video[0])
    reference_elapsed = float(reference[-1]) - float(reference[0])
    if not math.isfinite(video_elapsed) or video_elapsed <= 0.0:
        return None
    ratio = reference_elapsed / video_elapsed
    return ratio if math.isfinite(ratio) else None


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_output_path(output: Path, roots: Iterable[Path]) -> None:
    resolved_output = output.expanduser().resolve(strict=False)
    for root in roots:
        resolved_root = root.expanduser().resolve(strict=False)
        if is_relative_to(resolved_output, resolved_root):
            raise ValueError(f"refusing to write audit output below source root: {resolved_root}")


def _validate_png_builtin(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("bad PNG signature")
    offset = 8
    width = height = None
    compressed = bytearray()
    saw_end = False
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        crc_offset = offset + 8 + length
        if crc_offset + 4 > len(data):
            raise ValueError("truncated PNG chunk")
        expected_crc = struct.unpack(">I", data[crc_offset : crc_offset + 4])[0]
        actual_crc = binascii.crc32(chunk_type)
        actual_crc = binascii.crc32(chunk_data, actual_crc) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError(f"bad PNG CRC in {chunk_type!r}")
        if chunk_type == b"IHDR":
            width, height = struct.unpack(">II", chunk_data[:8])
        elif chunk_type == b"IDAT":
            compressed.extend(chunk_data)
        elif chunk_type == b"IEND":
            saw_end = True
            break
        offset = crc_offset + 4
    if width is None or height is None or not saw_end:
        raise ValueError("incomplete PNG")
    zlib.decompress(bytes(compressed))
    return {"decoder": "stdlib_png", "width": width, "height": height}


def decode_image(path: Path) -> dict[str, Any]:
    """Fully decode with Pillow/OpenCV, with a dependency-free PNG fallback."""
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as image:
            image.load()
            return {"decoder": "pillow", "width": image.width, "height": image.height}
    except ImportError:
        pass
    except Exception as exc:
        raise ValueError(f"Pillow decode failed: {exc}") from exc

    try:
        import cv2  # type: ignore

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("cv2.imread returned None")
        return {"decoder": "opencv", "width": int(image.shape[1]), "height": int(image.shape[0])}
    except ImportError:
        pass
    except Exception as exc:
        raise ValueError(f"OpenCV decode failed: {exc}") from exc

    if path.suffix.lower() == ".png":
        return _validate_png_builtin(path)
    raise RuntimeError("Pillow or OpenCV is required to decode non-PNG media")


def decode_paths(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decoded = []
    problems = []
    for path in sorted(set(paths), key=str):
        try:
            details = decode_image(path)
            decoded.append({"path": str(path), **details})
        except Exception as exc:
            problems.append(issue("error", "media_decode_failed", str(exc), path))
    return decoded, problems


def require_pyarrow():
    try:
        import pyarrow.parquet as parquet  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for InternData-N1 and TpT; use the documented omtrackvla environment"
        ) from exc
    return parquet


def _episode_number(path: Path) -> int:
    return int(path.stem.removeprefix("episode_"))


def audit_intern(root: Path, media_samples: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": "intern_data_n1",
        "root": str(root),
        "completeness": "user_confirmed_complete",
        "groups": {},
        "issues": [],
    }
    if not root.is_dir():
        result["issues"].append(issue("error", "missing_root", "dataset root does not exist", root))
        return result

    parquet = require_pyarrow()
    all_sample_candidates: list[tuple[str, Path]] = []
    total_scene_directories = usable_scenes = media_only_scenes = total_episodes = 0
    language_keys: set[str] = set()
    required_meta = ("info.json", "episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl")

    for group in sorted(path for path in root.iterdir() if path.is_dir()):
        scenes = sorted(path for path in group.iterdir() if path.is_dir())
        group_episodes = group_frames = group_usable_scenes = group_media_only = 0
        for scene in scenes:
            meta_dir = scene / "meta"
            episode_paths = sorted((scene / "data").glob("chunk-*/episode_*.parquet"))
            if not episode_paths:
                group_media_only += 1
                result["issues"].append(
                    issue(
                        "warning",
                        "intern_unindexed_media_only_scene",
                        "directory has no Parquet episodes and is excluded from manifests",
                        scene,
                    )
                )
                continue
            group_usable_scenes += 1
            for filename in required_meta:
                path = meta_dir / filename
                if not path.is_file():
                    result["issues"].append(issue("error", "intern_missing_metadata", filename, scene))
            episode_numbers = {_episode_number(path) for path in episode_paths}
            group_episodes += len(episode_paths)
            all_sample_candidates.extend((group.name, path) for path in episode_paths)

            try:
                info = load_json(meta_dir / "info.json")
                declared = int(info.get("total_episodes", -1))
                if declared != len(episode_paths):
                    result["issues"].append(
                        issue(
                            "error",
                            "intern_episode_count_mismatch",
                            f"declared={declared}, parquet={len(episode_paths)}",
                            scene,
                        )
                    )
                stats = list(jsonl_records(meta_dir / "episodes_stats.jsonl"))
                stats_numbers = {int(value["episode_index"]) for value in stats}
                if stats_numbers != episode_numbers:
                    result["issues"].append(
                        issue("error", "intern_episode_index_mismatch", "stats and parquet IDs differ", scene)
                    )
                frames = sum(int(value.get("image_index", {}).get("count", 0)) for value in stats)
                group_frames += frames
                if int(info.get("total_frames", -1)) != frames:
                    result["issues"].append(
                        issue(
                            "error",
                            "intern_frame_count_mismatch",
                            f"declared={info.get('total_frames')}, stats={frames}",
                            scene,
                        )
                    )
                first_episode = next(iter(jsonl_records(meta_dir / "episodes.jsonl")), {})
                first_task = next(iter(jsonl_records(meta_dir / "tasks.jsonl")), {})
                language_keys.update(nested_language_keys(first_episode))
                language_keys.update(nested_language_keys(first_task))
            except Exception as exc:
                result["issues"].append(issue("error", "intern_metadata_parse_failed", str(exc), scene))

        result["groups"][group.name] = {
            "scene_directories": len(scenes),
            "scenes": group_usable_scenes,
            "media_only_scene_directories": group_media_only,
            "episodes": group_episodes,
            "frames_from_metadata": group_frames,
        }
        total_scene_directories += len(scenes)
        usable_scenes += group_usable_scenes
        media_only_scenes += group_media_only
        total_episodes += group_episodes

    media_paths: list[Path] = []
    parquet_samples = []
    selected = stratified_samples(all_sample_candidates, lambda value: value[0], media_samples)
    for group_name, path in selected:
        try:
            metadata = parquet.ParquetFile(path).metadata
            rows = int(metadata.num_rows)
            schema_names = list(metadata.schema.names)
            parquet_samples.append(
                {"group": group_name, "path": str(path), "rows": rows, "leaf_schema_names": schema_names}
            )
            scene = path.parents[2]
            chunk = path.parent.name
            episode = path.stem
            for modality, suffix in (("rgb", ".jpg"), ("depth", ".png")):
                image_dir = scene / "videos" / chunk / f"observation.images.{modality}"
                images = sorted(image_dir.glob(f"{episode}_*{suffix}"))
                if len(images) != rows:
                    result["issues"].append(
                        issue(
                            "error",
                            "intern_sample_frame_count_mismatch",
                            f"{modality}: parquet rows={rows}, images={len(images)}",
                            image_dir,
                        )
                    )
                media_paths.extend(images[index] for index in sample_positions(len(images), 2))
        except Exception as exc:
            result["issues"].append(issue("error", "intern_parquet_sample_failed", str(exc), path))

    decoded, decode_issues = decode_paths(media_paths)
    result["issues"].extend(decode_issues)
    result.update(
        {
            "counts": {
                "groups": len(result["groups"]),
                "scene_directories": total_scene_directories,
                "scenes": usable_scenes,
                "media_only_scene_directories": media_only_scenes,
                "episodes": total_episodes,
            },
            "natural_language_metadata_keys": sorted(language_keys),
            "natural_language_model_input": False,
            "parquet_samples": parquet_samples,
            "decoded_media": decoded,
            "disk_size_scanned": False,
        }
    )
    return result


def audit_sage(root: Path, media_samples: int) -> dict[str, Any]:
    result: dict[str, Any] = {"id": "sage3d_extracted", "root": str(root), "issues": []}
    index_path = root / "index.json"
    if not index_path.is_file():
        result["issues"].append(issue("error", "missing_sage_index", "index.json not found", index_path))
        return result

    entries = load_json(index_path).get("eps", [])
    seen_paths: set[str] = set()
    candidates: list[tuple[tuple[str, str], Path]] = []
    runs: set[str] = set()
    modes: Counter[str] = Counter()
    cameras: Counter[str] = Counter()
    total_steps = waypoint_steps = terminal_missing = acceptance_conflicts = 0
    source_success_false = 0
    canonical_accepted = indexed_rejected = 0
    language_keys: set[str] = set()

    for entry in entries:
        relative = str(entry.get("path", ""))
        ep_dir = root / relative
        if relative in seen_paths:
            result["issues"].append(issue("error", "sage_duplicate_index_path", relative, index_path))
        seen_paths.add(relative)
        runs.add(str(entry.get("run")))
        modes[str(entry.get("mode"))] += 1
        cameras[str(entry.get("cam"))] += 1
        if not ep_dir.is_dir():
            result["issues"].append(issue("error", "sage_missing_episode_dir", relative, ep_dir))
            continue
        candidates.append(((str(entry.get("mode")), str(entry.get("cam"))), ep_dir))
        required = ("derived.json", "quality.json", "camera_info.json")
        for filename in required:
            if not (ep_dir / filename).is_file():
                result["issues"].append(issue("error", "sage_missing_metadata", filename, ep_dir))
        try:
            quality = load_json(ep_dir / "quality.json")
            accepted = sage_is_canonically_accepted(
                indexed=True,
                accepted_marker=(ep_dir / "_ACCEPTED").is_file(),
                quality_status=quality.get("status"),
            )
            if accepted:
                canonical_accepted += 1
            else:
                indexed_rejected += 1
            summary_path = ep_dir / f"{entry.get('ep')}.json"
            summary = load_json(summary_path)
            language_keys.update(nested_language_keys(summary))
            source_success = bool(summary.get("success"))
            if not source_success:
                source_success_false += 1
            if accepted != source_success:
                acceptance_conflicts += 1

            derived = load_json(ep_dir / "derived.json")
            steps = derived.get("steps")
            if not isinstance(steps, list):
                raise ValueError("derived.json.steps is not a list")
            expected_steps = int(entry.get("steps", -1))
            if len(steps) != expected_steps:
                result["issues"].append(
                    issue(
                        "error",
                        "sage_step_count_mismatch",
                        f"index={expected_steps}, derived={len(steps)}",
                        ep_dir,
                    )
                )
            ids = [int(value.get("step", -1)) for value in steps]
            expected_ids = list(range(len(steps)))
            if ids != expected_ids:
                result["issues"].append(issue("error", "sage_noncontiguous_steps", "step IDs are not 0..N-1", ep_dir))
            rgb_ids = sorted(int(path.stem) for path in (ep_dir / "rgb").glob("[0-9]*.jpg"))
            depth_ids = sorted(int(path.stem) for path in (ep_dir / "depth").glob("[0-9]*.png"))
            if rgb_ids != expected_ids or depth_ids != expected_ids:
                result["issues"].append(
                    issue("error", "sage_media_path_mismatch", "RGB/depth IDs differ from derived step IDs", ep_dir)
                )
            episode_waypoints = 0
            missing_ids = []
            for value in steps:
                waypoints = value.get("waypoints_ego")
                if isinstance(waypoints, list) and len(waypoints) == 8:
                    episode_waypoints += 1
                else:
                    missing_ids.append(int(value.get("step", -1)))
            if missing_ids and missing_ids != list(range(len(steps) - len(missing_ids), len(steps))):
                result["issues"].append(
                    issue("error", "sage_nonterminal_missing_waypoint", str(missing_ids[:20]), ep_dir)
                )
            total_steps += len(steps)
            waypoint_steps += episode_waypoints
            terminal_missing += len(missing_ids)
        except Exception as exc:
            result["issues"].append(issue("error", "sage_metadata_parse_failed", str(exc), ep_dir))

    media_paths = []
    for _, ep_dir in stratified_samples(candidates, lambda value: value[0], media_samples):
        for modality, suffix in (("rgb", ".jpg"), ("depth", ".png")):
            images = sorted((ep_dir / modality).glob(f"[0-9]*{suffix}"))
            media_paths.extend(images[index] for index in sample_positions(len(images), 3))
    decoded, decode_issues = decode_paths(media_paths)
    result["issues"].extend(decode_issues)
    marker_paths = {
        str(path.parent.relative_to(root))
        for path in root.glob("*/*/*/*/_ACCEPTED")
        if path.is_file()
    }
    markers_not_indexed = sorted(marker_paths - seen_paths)
    if markers_not_indexed:
        result["issues"].append(
            issue(
                "warning",
                "sage_accepted_markers_not_indexed",
                f"{len(markers_not_indexed)} marker directories are excluded by the canonical rule: "
                + ", ".join(markers_not_indexed[:20]),
                root,
            )
        )
    result.update(
        {
            "counts": {
                "runs": len(runs),
                "episodes": len(entries),
                "canonically_accepted_episodes": canonical_accepted,
                "indexed_rejected_episodes": indexed_rejected,
                "accepted_marker_directories": len(marker_paths),
                "accepted_markers_not_indexed": len(markers_not_indexed),
                "indexed_source_success_false": source_success_false,
                "steps": total_steps,
                "steps_with_8_point_waypoints": waypoint_steps,
                "terminal_steps_without_waypoints": terminal_missing,
                "canonical_acceptance_vs_source_success_conflicts": acceptance_conflicts,
            },
            "episodes_by_mode": dict(sorted(modes.items())),
            "episodes_by_camera": dict(sorted(cameras.items())),
            "canonical_acceptance": "root index.json membership AND _ACCEPTED AND quality.status=accepted",
            "natural_language_metadata_keys": sorted(language_keys),
            "natural_language_model_input": False,
            "decoded_media": decoded,
        }
    )
    return result


def audit_tpt(root: Path, media_samples: int) -> dict[str, Any]:
    result: dict[str, Any] = {"id": "tpt_bench_clean_v2", "root": str(root), "issues": []}
    if not root.is_dir():
        result["issues"].append(issue("error", "missing_root", "dataset root does not exist", root))
        return result
    parquet = require_pyarrow()
    required_columns = {
        "video_idx",
        "gt_idx",
        "gt_ts_ns",
        "vid_pts_ms",
        "bbox_source",
        "bbox_qv",
        "is_exist",
        "is_behind_glass",
        "interpolated",
        "odom_pos",
        "odom_quat_xyzw",
        "odom_ts_s",
    }
    sequences = sorted(
        path for path in root.iterdir() if path.is_dir() and (path / "frames.parquet").is_file()
    )
    rows = visible = glass = missing_rgb = zero_rgb = generated_crops = 0
    ratios = []
    sequence_results = {}
    media_paths = []
    language_keys: set[str] = set()

    for sequence in sequences:
        try:
            table = parquet.read_table(
                sequence / "frames.parquet",
                columns=[
                    "video_idx",
                    "gt_ts_ns",
                    "vid_pts_ms",
                    "is_exist",
                    "is_behind_glass",
                    "odom_ts_s",
                ],
            )
            available = set(parquet.ParquetFile(sequence / "frames.parquet").schema_arrow.names)
            missing_columns = sorted(required_columns - available)
            if missing_columns:
                result["issues"].append(
                    issue("error", "tpt_missing_columns", ", ".join(missing_columns), sequence)
                )
            values = table.to_pydict()
            count = int(table.num_rows)
            video_ids = [int(value) for value in values["video_idx"]]
            expected_names = {f"frame_{value:06d}.jpg" for value in video_ids}
            rgb_dir = sequence / "rgb_frames"
            actual_paths = {
                path.name: path
                for path in rgb_dir.glob("frame_*.jpg")
                if path.parent == rgb_dir
            }
            missing_names = expected_names - set(actual_paths)
            extra_names = set(actual_paths) - expected_names
            missing_rgb += len(missing_names)
            zero_names = [name for name in expected_names & set(actual_paths) if actual_paths[name].stat().st_size == 0]
            zero_rgb += len(zero_names)
            if missing_names or extra_names or zero_names:
                result["issues"].append(
                    issue(
                        "error",
                        "tpt_rgb_path_mismatch",
                        f"missing={len(missing_names)}, extra={len(extra_names)}, zero={len(zero_names)}",
                        sequence,
                    )
                )
            reference_ms = [float(value) / 1e6 for value in values["gt_ts_ns"]]
            ratio = elapsed_ratio(values["vid_pts_ms"], reference_ms)
            if ratio is not None:
                ratios.append(ratio)
            seq_visible = sum(int(value) != 0 for value in values["is_exist"])
            seq_glass = sum(abs(int(value)) == 1 for value in values["is_behind_glass"])
            rows += count
            visible += seq_visible
            glass += seq_glass
            crop_dir = rgb_dir / "_target_crops"
            generated_crops += sum(1 for path in crop_dir.glob("*" ) if path.is_file())
            for index in sample_positions(len(video_ids), media_samples):
                path = actual_paths.get(f"frame_{video_ids[index]:06d}.jpg")
                if path is not None:
                    media_paths.append(path)
            desc_path = sequence / "desc.txt"
            if desc_path.is_file() and desc_path.read_text(encoding="utf-8", errors="replace").strip():
                language_keys.add("desc")
            sequence_results[sequence.name] = {
                "rows": count,
                "visible_rows": seq_visible,
                "glass_rows": seq_glass,
                "gt_elapsed_over_video_elapsed": ratio,
            }
        except Exception as exc:
            result["issues"].append(issue("error", "tpt_sequence_audit_failed", str(exc), sequence))

    decoded, decode_issues = decode_paths(media_paths)
    result["issues"].extend(decode_issues)
    clock = {
        "gt_elapsed_over_video_elapsed_min": min(ratios) if ratios else None,
        "gt_elapsed_over_video_elapsed_median": statistics.median(ratios) if ratios else None,
        "gt_elapsed_over_video_elapsed_max": max(ratios) if ratios else None,
        "status": "unresolved_do_not_freeze_horizon",
    }
    result.update(
        {
            "counts": {
                "sequences": len(sequences),
                "rows": rows,
                "rgb_frames": rows - missing_rgb,
                "visible_rows": visible,
                "glass_occlusion_rows": glass,
                "missing_rgb_frames": missing_rgb,
                "zero_byte_rgb_frames": zero_rgb,
                "generated_target_crops_excluded": generated_crops,
            },
            "clock_audit": clock,
            "natural_language_metadata_keys": sorted(language_keys),
            "natural_language_model_input": False,
            "sequences": sequence_results,
            "decoded_media": decoded,
        }
    )
    return result


def compare_to_manifest(audit: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    expected = {value["id"]: value for value in manifest.get("datasets", [])}
    problems = []
    for dataset in audit.get("datasets", []):
        dataset_id = dataset["id"]
        expected_counts = expected.get(dataset_id, {}).get("counts", {})
        actual_counts = dataset.get("counts", {})
        for key, expected_value in expected_counts.items():
            if expected_value is None or key not in actual_counts:
                continue
            if actual_counts[key] != expected_value:
                problems.append(
                    issue(
                        "error",
                        "manifest_count_mismatch",
                        f"{dataset_id}.{key}: expected={expected_value}, actual={actual_counts[key]}",
                    )
                )
    return problems


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=DATASET_IDS)
    parser.add_argument("--intern-root", type=Path, default=DEFAULT_ROOTS["intern_data_n1"])
    parser.add_argument("--sage-root", type=Path, default=DEFAULT_ROOTS["sage3d_extracted"])
    parser.add_argument("--tpt-root", type=Path, default=DEFAULT_ROOTS["tpt_bench_clean_v2"])
    parser.add_argument("--media-samples-per-stratum", type=int, default=1)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected = args.dataset or list(DATASET_IDS)
    roots = {
        "intern_data_n1": args.intern_root,
        "sage3d_extracted": args.sage_root,
        "tpt_bench_clean_v2": args.tpt_root,
    }
    if args.media_samples_per_stratum < 1:
        raise ValueError("--media-samples-per-stratum must be at least 1")
    if args.output is not None:
        validate_output_path(args.output, roots.values())

    functions = {
        "intern_data_n1": audit_intern,
        "sage3d_extracted": audit_sage,
        "tpt_bench_clean_v2": audit_tpt,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "source_access": "read_only",
            "media_validation": "deterministic_stratified_decode",
            "media_samples_per_stratum": args.media_samples_per_stratum,
            "natural_language_model_input": False,
        },
        "datasets": [],
        "issues": [],
    }
    for dataset_id in selected:
        try:
            report["datasets"].append(
                functions[dataset_id](roots[dataset_id], args.media_samples_per_stratum)
            )
        except Exception as exc:
            report["datasets"].append(
                {
                    "id": dataset_id,
                    "root": str(roots[dataset_id]),
                    "issues": [issue("error", "audit_crashed", str(exc), roots[dataset_id])],
                }
            )

    if args.manifest is not None:
        report["manifest"] = str(args.manifest)
        report["issues"].extend(compare_to_manifest(report, load_json(args.manifest)))
    all_issues = list(report["issues"])
    for dataset in report["datasets"]:
        all_issues.extend(dataset.get("issues", []))
    report["status"] = "fail" if any(value.get("severity") == "error" for value in all_issues) else "pass"
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
