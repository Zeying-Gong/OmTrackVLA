#!/usr/bin/env python3
"""Export a tiny, self-contained, read-only sample of the three datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

if __package__:
    from scripts.audit_data_inventory import validate_output_path
else:
    from audit_data_inventory import validate_output_path


INTERN_ROOT = Path("/h100-2/vln_n1/traj_data")
SAGE_ROOT = Path("/data/nfs/share/OmTrackVLA/data/sage3d_extracted")
TPT_ROOT = Path("/data/nfs/share/OmTrackVLA/data/tpt_bench_clean_v2")

INTERN_SCENE = Path("3dfront_d435i/00154c06-2ee2-408a-9664-b8fd74742897")
INTERN_EPISODE = 0
SAGE_EPISODE = Path("0001_83992/stt/0/go2_realsense_d435i")
TPT_SEQUENCE = "0000"
FRAME_COUNT = 16


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def contiguous_indices(total: int, start: int, count: int) -> list[int]:
    if total < 1:
        raise ValueError("total must be positive")
    if start < 0 or start >= total:
        raise ValueError(f"start {start} is outside 0..{total - 1}")
    if count < 1:
        raise ValueError("count must be positive")
    return list(range(start, min(total, start + count)))


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"missing or empty source file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def first_jsonl_record(path: Path, key: str, expected: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if int(value.get(key, -1)) == expected:
                return value
    raise KeyError(f"no {key}={expected} in {path}")


def image_preview(
    sources: list[Path],
    destination: Path,
    boxes: list[list[float] | None] | None = None,
    labels: list[str] | None = None,
) -> None:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    if not sources:
        raise ValueError("preview needs at least one image")
    positions = sorted({round(i * (len(sources) - 1) / 3) for i in range(4)})
    panels = []
    for source_index in positions:
        image = cv2.imread(str(sources[source_index]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode preview source: {sources[source_index]}")
        if boxes and boxes[source_index] is not None:
            x0, y0, x1, y1 = [round(value) for value in boxes[source_index] or []]
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 255), 3)
        label = labels[source_index] if labels else sources[source_index].name
        cv2.putText(
            image,
            label,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        height, width = image.shape[:2]
        scale = min(1.0, 320.0 / width)
        panel = cv2.resize(image, (round(width * scale), round(height * scale)))
        panels.append(panel)
    target_height = min(panel.shape[0] for panel in panels)
    normalized = [
        cv2.resize(panel, (round(panel.shape[1] * target_height / panel.shape[0]), target_height))
        for panel in panels
    ]
    preview = np.concatenate(normalized, axis=1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), preview, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise OSError(f"failed to write preview: {destination}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksums(root: Path) -> int:
    checksum_path = root / "checksums.sha256"
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in paths),
        encoding="utf-8",
    )
    return len(paths)


def export_intern(output: Path, frame_count: int) -> dict[str, Any]:
    import pyarrow.parquet as pq  # type: ignore

    source = INTERN_ROOT / INTERN_SCENE
    destination = output / "intern_data_n1" / INTERN_SCENE
    parquet_source = source / "data/chunk-000" / f"episode_{INTERN_EPISODE:06d}.parquet"
    table = pq.read_table(parquet_source)
    indices = contiguous_indices(table.num_rows, 0, frame_count)
    parquet_destination = destination / "data/chunk-000" / parquet_source.name
    parquet_destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.slice(indices[0], len(indices)), parquet_destination, compression="snappy")

    rgb_sources = []
    for modality, suffix in (("rgb", ".jpg"), ("depth", ".png")):
        for index in indices:
            name = f"episode_{INTERN_EPISODE:06d}_{index:03d}{suffix}"
            src = source / "videos/chunk-000" / f"observation.images.{modality}" / name
            dst = destination / "videos/chunk-000" / f"observation.images.{modality}" / name
            copy_file(src, dst)
            if modality == "rgb":
                rgb_sources.append(src)

    info = load_json(source / "meta/info.json")
    write_json(destination / "meta/info.source.json", info)
    sample_info = dict(info)
    sample_info.update(
        {
            "total_episodes": 1,
            "total_frames": len(indices),
            "total_videos": 0,
            "sample_subset": True,
            "source_episode_index": INTERN_EPISODE,
            "source_frame_indices": indices,
        }
    )
    write_json(destination / "meta/info.json", sample_info)
    episode = first_jsonl_record(source / "meta/episodes.jsonl", "episode_index", INTERN_EPISODE)
    stats = first_jsonl_record(source / "meta/episodes_stats.jsonl", "episode_index", INTERN_EPISODE)
    stats["source_image_index"] = stats.get("image_index")
    stats["image_index"] = {"min": indices[0], "max": indices[-1], "count": len(indices)}
    (destination / "meta/episodes.jsonl").write_text(
        json.dumps(episode, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (destination / "meta/episodes_stats.jsonl").write_text(
        json.dumps(stats, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    image_preview(
        rgb_sources,
        output / "previews/intern_data_n1.jpg",
        labels=[f"Intern frame {index}" for index in indices],
    )
    return {
        "source": str(source),
        "selection": {"episode": INTERN_EPISODE, "frame_indices": indices},
        "parquet_rows": len(indices),
        "rgb_frames": len(indices),
        "depth_frames": len(indices),
        "notes": ["instruction fields are metadata only and prohibited from model inputs"],
    }


def export_sage(output: Path, frame_count: int) -> dict[str, Any]:
    source = SAGE_ROOT / SAGE_EPISODE
    destination = output / "sage3d_extracted" / SAGE_EPISODE
    derived = load_json(source / "derived.json")
    info_path = source / "0_info.json"
    source_info = load_json(info_path)
    indices = contiguous_indices(len(derived["steps"]), 0, frame_count)
    selected_steps = [derived["steps"][index] for index in indices]
    selected_info = [source_info[index] for index in indices]
    subset_derived = dict(derived)
    subset_derived["steps"] = selected_steps
    subset_derived["sample_subset"] = True
    subset_derived["source_step_indices"] = indices
    write_json(destination / "derived.json", subset_derived)
    write_json(destination / "0_info.json", selected_info)
    for filename in ("0.json", "camera_info.json", "quality.json"):
        copy_file(source / filename, destination / filename)
    (destination / "_ACCEPTED").parent.mkdir(parents=True, exist_ok=True)
    (destination / "_ACCEPTED").write_text("1", encoding="ascii")

    rgb_sources = []
    boxes = []
    for index, step in zip(indices, selected_steps):
        for modality, suffix in (("rgb", ".jpg"), ("depth", ".png")):
            name = f"{index:05d}{suffix}"
            src = source / modality / name
            dst = destination / modality / name
            copy_file(src, dst)
            if modality == "rgb":
                rgb_sources.append(src)
                box = step.get("bbox")
                boxes.append([float(value) for value in box] if box else None)
    image_preview(
        rgb_sources,
        output / "previews/sage3d_extracted.jpg",
        boxes=boxes,
        labels=[f"SAGE3D step {index}" for index in indices],
    )
    return {
        "source": str(source),
        "selection": {"step_indices": indices},
        "rgb_frames": len(indices),
        "depth_frames": len(indices),
        "canonical_acceptance": "root index member AND _ACCEPTED AND quality.status=accepted",
        "notes": ["yellow preview boxes are labels, not model inputs"],
    }


def export_tpt(output: Path, frame_count: int) -> dict[str, Any]:
    import pyarrow.parquet as pq  # type: ignore

    source = TPT_ROOT / TPT_SEQUENCE
    destination = output / "tpt_bench_clean_v2" / TPT_SEQUENCE
    table = pq.read_table(source / "frames.parquet")
    columns = table.to_pydict()
    start = next(
        index
        for index, exists in enumerate(columns["is_exist"])
        if bool(exists) and any(float(value) != 0.0 for value in columns["bbox_qv"][index])
    )
    indices = contiguous_indices(table.num_rows, start, frame_count)
    selected = table.slice(indices[0], len(indices))
    parquet_destination = destination / "frames.parquet"
    parquet_destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(selected, parquet_destination, compression="snappy")
    for filename in ("meta.json", "desc.txt"):
        copy_file(source / filename, destination / filename)

    selected_columns = selected.to_pydict()
    rgb_sources = []
    boxes = []
    labels = []
    video_indices = []
    for video_index, bbox in zip(selected_columns["video_idx"], selected_columns["bbox_qv"]):
        video_index = int(video_index)
        video_indices.append(video_index)
        name = f"frame_{video_index:06d}.jpg"
        src = source / "rgb_frames" / name
        copy_file(src, destination / "rgb_frames" / name)
        rgb_sources.append(src)
        x, y, width, height = [float(value) for value in bbox]
        boxes.append([x, y, x + width, y + height])
        labels.append(f"TpT frame {video_index}")
    image_preview(
        rgb_sources,
        output / "previews/tpt_bench_clean_v2.jpg",
        boxes=boxes,
        labels=labels,
    )
    return {
        "source": str(source),
        "selection": {"parquet_row_indices": indices, "video_indices": video_indices},
        "parquet_rows": len(indices),
        "rgb_frames": len(indices),
        "notes": [
            "yellow preview boxes are labels, not model inputs",
            "GT/ODOM and video clock semantics remain unresolved",
        ],
    }


def verify_no_symlinks(root: Path) -> None:
    links = [path for path in root.rglob("*") if path.is_symlink()]
    if links:
        raise ValueError(f"export contains symbolic links: {links}")


def export(output: Path, frame_count: int = FRAME_COUNT) -> dict[str, Any]:
    roots = (INTERN_ROOT, SAGE_ROOT, TPT_ROOT)
    validate_output_path(output, roots)
    if output.exists():
        raise FileExistsError(f"output already exists; refusing to overwrite: {output}")
    output.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "purpose": "small real-data examples for understanding layout and appearance; not training data",
        "frame_count_per_dataset": frame_count,
        "datasets": {
            "intern_data_n1": export_intern(output, frame_count),
            "sage3d_extracted": export_sage(output, frame_count),
            "tpt_bench_clean_v2": export_tpt(output, frame_count),
        },
        "excluded": [
            "absolute symbolic links",
            "target crops and target refs",
            "videos and point clouds",
            "Python caches",
            "all non-selected frames and episodes",
        ],
    }
    manifest["checksummed_files"] = (
        sum(1 for path in output.rglob("*") if path.is_file()) + 1
    )
    write_json(output / "subset_manifest.json", manifest)
    verify_no_symlinks(output)
    checksummed_files = write_checksums(output)
    if checksummed_files != manifest["checksummed_files"]:
        raise RuntimeError(
            f"checksum file count changed: expected {manifest['checksummed_files']}, "
            f"found {checksummed_files}"
        )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=FRAME_COUNT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = export(args.output, args.frame_count)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
