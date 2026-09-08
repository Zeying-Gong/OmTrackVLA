"""Read-only Phase 1 datasets that enforce the WP-1 identity boundary."""
from __future__ import annotations

import bisect
import hashlib
import json
import random
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

from omtrackvla.geometry.se2 import intern_pair_to_canonical_se2


ADAPTER_VERSION = "phase1-readonly-v1"


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_relative(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe source-relative path: {relative}")
    candidate = (root / Path(*pure.parts)).resolve(strict=False)
    source = root.resolve(strict=True)
    try:
        candidate.relative_to(source)
    except ValueError as error:
        raise ValueError(f"path escapes source root: {relative}") from error
    return candidate


def _manifest(path: Path) -> dict[str, object]:
    value = _load_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Phase 1 manifest must use schema_version=1")
    expected = value.get("manifest_sha256")
    unsigned = dict(value)
    unsigned.pop("manifest_sha256", None)
    actual = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected != actual:
        raise ValueError("Phase 1 manifest checksum mismatch")
    return value


@lru_cache(maxsize=2048)
def _image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _bbox_xyxy_norm(bbox: Sequence[float], size: tuple[int, int], xywh: bool = False) -> list[float]:
    if len(bbox) != 4:
        raise ValueError("bbox must have four values")
    x0, y0, c, d = (float(value) for value in bbox)
    x1, y1 = (x0 + c, y0 + d) if xywh else (c, d)
    width, height = size
    x0, x1 = sorted((max(0.0, min(float(width), x0)), max(0.0, min(float(width), x1))))
    y0, y1 = sorted((max(0.0, min(float(height), y0)), max(0.0, min(float(height), y1))))
    if width <= 0 or height <= 0 or x1 <= x0 or y1 <= y0:
        raise ValueError("bbox is empty after clipping")
    return [x0 / width, y0 / height, x1 / width, y1 / height]


def _history_indices(initial: int, anchor: int, history_size: int) -> list[int]:
    if initial < 0 or anchor <= initial:
        raise ValueError("identity anchor must follow its initialization frame")
    tail_start = max(initial + 1, anchor - max(0, history_size - 2))
    indices = [initial, *range(tail_start, anchor + 1)]
    if len(indices) < 2:
        raise ValueError("identity history must contain at least two frames")
    if len(indices) <= history_size:
        return indices
    return indices[:1] + indices[-(history_size - 1) :]


def _empty_uwb() -> dict[str, object]:
    return {
        "valid": False,
        "measurement_kind": "none",
        "relative_position_base_xy_m": None,
        "covariance_base_xy_m2": None,
        "quality_01": None,
        "source_timestamp_ns": None,
        "receive_timestamp_ns": None,
        "age_s": None,
        "los_state": None,
    }


def _identity_record(
    *,
    dataset_id: str,
    split_unit_id: str,
    episode_id: str,
    anchor_index: int,
    root: Path,
    history_paths: Sequence[Path],
    history_timestamps_ns: Sequence[int],
    initialization_bbox: Sequence[float],
    target_bbox: Sequence[float] | None,
    target_visible: bool,
    target_track_id: str,
    source_record: Path,
    occlusion_state: str,
) -> dict[str, object]:
    if len(history_paths) != len(history_timestamps_ns) or len(history_paths) < 2:
        raise ValueError("RGB history path/timestamp lengths are inconsistent")
    relative_paths = [path.relative_to(root).as_posix() for path in history_paths]
    source_relative = source_record.relative_to(root).as_posix()
    return {
        "schema_version": 1,
        "sample_id": f"{dataset_id}/{episode_id}/anchor-{anchor_index}",
        "sample_role": "identity_auxiliary",
        "source": {
            "dataset_id": dataset_id,
            "split_unit_id": split_unit_id,
            "episode_id": episode_id,
            "anchor_index": int(anchor_index),
            "adapter_version": ADAPTER_VERSION,
        },
        "model_inputs": {
            "condition_mode": "visual_only",
            "anchor_timestamp_ns": int(history_timestamps_ns[-1]),
            "rgb_sensor_valid": True,
            "rgb_history": [
                {"rgb_path": relative, "timestamp_ns": int(timestamp), "valid": True}
                for relative, timestamp in zip(relative_paths, history_timestamps_ns)
            ],
            "visual_initialization": {
                "valid": True,
                "rgb_path": relative_paths[0],
                "bbox_xyxy_norm": [float(value) for value in initialization_bbox],
                "timestamp_ns": int(history_timestamps_ns[0]),
            },
            "uwb_target": _empty_uwb(),
        },
        "routing_metadata": {
            "target_tag_id": None,
            "calibration_id": None,
            "simulation_spec_id": None,
        },
        "supervision": {
            "expert_trajectory": {
                "valid": False,
                "waypoints_base_xy_m": [None] * 8,
                "time_offsets_s": None,
                "valid_mask": [False] * 8,
            },
            "auxiliary_labels": {
                "target_track_id": target_track_id,
                "target_bbox_xyxy_norm": (
                    [float(value) for value in target_bbox] if target_bbox is not None else None
                ),
                "target_visible": bool(target_visible),
                "target_position_base_xy_m": None,
                "occlusion_state": occlusion_state,
                "uwb_error_base_xy_m": None,
            },
            "safety": {"stop_required": False, "reason": "none"},
        },
        "provenance": {
            "source_record": source_relative,
            "transform_spec_id": None,
            "clock_spec_id": None,
            "generation_spec_id": None,
        },
    }


def _rgb_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _reference_tensor(path: Path, bbox: Sequence[float], image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        width, height = image.size
        x0, y0, x1, y1 = bbox
        crop = image.convert("RGB").crop(
            (
                int(round(x0 * width)),
                int(round(y0 * height)),
                int(round(x1 * width)),
                int(round(y1 * height)),
            )
        )
        crop = crop.resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(crop, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class ContractIdentityDataset(Dataset):
    """SAGE3D/TpT identity clips with no writes or generated source crops."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        roots: Mapping[str, str | Path] | None = None,
        datasets: Sequence[str] = ("sage3d_extracted", "tpt_bench_clean_v2"),
        image_size: int = 224,
        history_size: int = 8,
        max_units_per_dataset: int | None = None,
    ) -> None:
        self.manifest = _manifest(Path(manifest_path))
        self.split = split
        self.image_size = int(image_size)
        self.history_size = int(history_size)
        if self.history_size < 2:
            raise ValueError("history_size must be at least two")
        manifest_datasets = self.manifest["datasets"]
        self.roots = {}
        self.descriptors: list[dict[str, object]] = []
        roots = dict(roots or {})
        for dataset_id in datasets:
            entry = manifest_datasets[dataset_id]
            root = Path(roots.get(dataset_id, entry["root_hint"])).expanduser().resolve(strict=True)
            self.roots[dataset_id] = root
            units = list(entry["splits"][split])
            if max_units_per_dataset is not None:
                units = units[: int(max_units_per_dataset)]
            if dataset_id == "sage3d_extracted":
                self.descriptors.extend(self._index_sage(root, set(units)))
            elif dataset_id == "tpt_bench_clean_v2":
                self.descriptors.extend(self._index_tpt(root, units))
            else:
                raise ValueError(f"unsupported identity source: {dataset_id}")
        if not self.descriptors:
            raise ValueError(f"no identity clips found for split={split}")
        self._ends = []
        total = 0
        for descriptor in self.descriptors:
            total += max(0, int(descriptor["frames"]) - 1)
            self._ends.append(total)
        if total <= 0:
            raise ValueError("identity sources contain no anchor frames")

    @staticmethod
    def _index_sage(root: Path, runs: set[str]) -> list[dict[str, object]]:
        index = _load_json(root / "index.json")
        descriptors = []
        for entry in index["eps"]:
            run = str(entry["run"])
            if run not in runs:
                continue
            relative = str(entry["path"])
            episode = _safe_relative(root, relative)
            marker = episode / "_ACCEPTED"
            quality_path = episode / "quality.json"
            derived_path = episode / "derived.json"
            if not marker.is_file() or not quality_path.is_file() or not derived_path.is_file():
                continue
            quality = _load_json(quality_path)
            if not isinstance(quality, dict) or quality.get("status") != "accepted":
                continue
            descriptors.append(
                {
                    "dataset_id": "sage3d_extracted",
                    "root": root,
                    "split_unit_id": run,
                    "episode_id": f"{run}/{entry['mode']}/{entry['ep']}/{entry['cam']}",
                    "episode": episode,
                    "derived": derived_path,
                    "frames": int(entry.get("steps", 0)),
                }
            )
        return descriptors

    @staticmethod
    def _index_tpt(root: Path, sequences: Sequence[str]) -> list[dict[str, object]]:
        descriptors = []
        for sequence in sequences:
            parquet = _safe_relative(root, f"{sequence}/frames.parquet")
            frames = pq.read_metadata(parquet).num_rows
            if frames > 1:
                descriptors.append(
                    {
                        "dataset_id": "tpt_bench_clean_v2",
                        "root": root,
                        "split_unit_id": sequence,
                        "episode_id": sequence,
                        "parquet": parquet,
                        "frames": frames,
                    }
                )
        return descriptors

    def __len__(self) -> int:
        return self._ends[-1]

    def _locate(self, index: int) -> tuple[dict[str, object], int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        descriptor_index = bisect.bisect_right(self._ends, index)
        start = self._ends[descriptor_index - 1] if descriptor_index else 0
        return self.descriptors[descriptor_index], index - start + 1

    @lru_cache(maxsize=16)
    def _derived(self, path: str) -> dict[str, object]:
        value = _load_json(Path(path))
        if not isinstance(value, dict) or not isinstance(value.get("steps"), list):
            raise ValueError(f"invalid SAGE3D derived record: {path}")
        return value

    @lru_cache(maxsize=8)
    def _table(self, path: str) -> dict[str, list[object]]:
        return pq.read_table(path).to_pydict()

    def _sage_record(self, descriptor: dict[str, object], requested_anchor: int) -> dict[str, object]:
        derived = self._derived(str(descriptor["derived"]))
        steps = derived["steps"]
        visible_indices = [
            index for index, step in enumerate(steps) if step.get("visible") and step.get("bbox")
        ]
        if not visible_indices:
            raise ValueError(f"SAGE3D episode has no initialization bbox: {descriptor['episode_id']}")
        initial = visible_indices[0]
        if initial >= len(steps) - 1:
            raise ValueError(f"SAGE3D initialization occurs on the final frame: {descriptor['episode_id']}")
        anchor = initial + 1 + ((requested_anchor - 1) % (len(steps) - initial - 1))
        history_indices = _history_indices(initial, anchor, self.history_size)
        episode = Path(descriptor["episode"])
        paths = [episode / "rgb" / f"{int(steps[index]['step']):05d}.jpg" for index in history_indices]
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(f"missing SAGE3D RGB in {descriptor['episode_id']}")
        frame_period = 33_333_333
        timestamps = [int(steps[index]["step"]) * frame_period for index in history_indices]
        init_bbox = _bbox_xyxy_norm(steps[initial]["bbox"], _image_size(str(paths[0])))
        target_step = steps[anchor]
        visible = bool(target_step.get("visible") and target_step.get("bbox"))
        target_bbox = (
            _bbox_xyxy_norm(target_step["bbox"], _image_size(str(paths[-1]))) if visible else None
        )
        return _identity_record(
            dataset_id="sage3d_extracted",
            split_unit_id=str(descriptor["split_unit_id"]),
            episode_id=str(descriptor["episode_id"]),
            anchor_index=anchor,
            root=Path(descriptor["root"]),
            history_paths=paths,
            history_timestamps_ns=timestamps,
            initialization_bbox=init_bbox,
            target_bbox=target_bbox,
            target_visible=visible,
            target_track_id=f"sage:{descriptor['episode_id']}",
            source_record=Path(descriptor["derived"]),
            occlusion_state="visible" if visible else "out_of_view",
        )

    def _tpt_record(self, descriptor: dict[str, object], requested_anchor: int) -> dict[str, object]:
        table = self._table(str(descriptor["parquet"]))
        count = len(table["video_idx"])
        visible_indices = [index for index, value in enumerate(table["is_exist"]) if bool(value)]
        if not visible_indices:
            raise ValueError(f"TpT sequence has no initialization bbox: {descriptor['episode_id']}")
        initial = visible_indices[0]
        if initial >= count - 1:
            raise ValueError(f"TpT initialization occurs on the final frame: {descriptor['episode_id']}")
        anchor = initial + 1 + ((requested_anchor - 1) % (count - initial - 1))
        history_indices = _history_indices(initial, anchor, self.history_size)
        root = Path(descriptor["root"])
        sequence = str(descriptor["episode_id"])
        paths = [
            root / sequence / "rgb_frames" / f"frame_{int(table['video_idx'][index]):06d}.jpg"
            for index in history_indices
        ]
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(f"missing TpT RGB in sequence {sequence}")
        timestamps = [int(round(float(table["vid_pts_ms"][index]) * 1_000_000.0)) for index in history_indices]
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError(f"non-monotonic TpT video timestamps in sequence {sequence}")
        init_bbox = _bbox_xyxy_norm(
            table["bbox_qv"][initial], _image_size(str(paths[0])), xywh=True
        )
        visible = bool(table["is_exist"][anchor])
        target_bbox = (
            _bbox_xyxy_norm(table["bbox_qv"][anchor], _image_size(str(paths[-1])), xywh=True)
            if visible
            else None
        )
        behind_glass = abs(int(table["is_behind_glass"][anchor])) == 1
        return _identity_record(
            dataset_id="tpt_bench_clean_v2",
            split_unit_id=sequence,
            episode_id=sequence,
            anchor_index=anchor,
            root=root,
            history_paths=paths,
            history_timestamps_ns=timestamps,
            initialization_bbox=init_bbox,
            target_bbox=target_bbox,
            target_visible=visible,
            target_track_id=f"tpt:{sequence}",
            source_record=Path(descriptor["parquet"]),
            occlusion_state="visible" if visible else ("occluded" if behind_glass else "out_of_view"),
        )

    def get_record(self, index: int) -> dict[str, object]:
        descriptor, requested_anchor = self._locate(index)
        if descriptor["dataset_id"] == "sage3d_extracted":
            return self._sage_record(descriptor, requested_anchor)
        return self._tpt_record(descriptor, requested_anchor)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.get_record(index)
        dataset_id = record["source"]["dataset_id"]
        root = self.roots[dataset_id]
        visual = record["model_inputs"]["visual_initialization"]
        current_relative = record["model_inputs"]["rgb_history"][-1]["rgb_path"]
        current_path = _safe_relative(root, current_relative)
        initial_path = _safe_relative(root, visual["rgb_path"])
        labels = record["supervision"]["auxiliary_labels"]
        bbox = labels["target_bbox_xyxy_norm"]
        temporal_paths = [
            _safe_relative(root, frame["rgb_path"])
            for frame in record["model_inputs"]["rgb_history"][1:]
        ]
        temporal = torch.zeros(self.history_size - 1, 3, self.image_size, self.image_size)
        temporal_mask = torch.zeros(self.history_size - 1, dtype=torch.bool)
        offset = len(temporal) - len(temporal_paths)
        for temporal_index, path in enumerate(temporal_paths):
            temporal[offset + temporal_index] = _rgb_tensor(path, self.image_size)
            temporal_mask[offset + temporal_index] = True
        return {
            "task_id": 0,
            "frame0": _reference_tensor(initial_path, visual["bbox_xyxy_norm"], self.image_size),
            "frame1": _rgb_tensor(current_path, self.image_size),
            "history": temporal,
            "history_mask": temporal_mask,
            "target_bbox": torch.tensor(bbox if bbox is not None else [0.0] * 4, dtype=torch.float32),
            "target_visible": torch.tensor(float(labels["target_visible"]), dtype=torch.float32),
            "motion": torch.zeros(3, dtype=torch.float32),
            "motion_valid": torch.tensor(0.0, dtype=torch.float32),
            "sample_id": record["sample_id"],
            "dataset_id": dataset_id,
        }


class InternGeometryDataset(Dataset):
    """Deterministic random access to InternData-N1 RGB pose pairs."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        root: str | Path | None = None,
        image_size: int = 224,
        maximum_gap: int = 8,
        history_size: int = 8,
        samples_per_epoch: int = 4096,
        seed: int = 20260907,
        max_units: int | None = None,
    ) -> None:
        manifest = _manifest(Path(manifest_path))
        entry = manifest["datasets"]["intern_data_n1"]
        self.root = Path(root or entry["root_hint"]).expanduser().resolve(strict=True)
        self.units = list(entry["splits"][split])
        if max_units is not None:
            self.units = self.units[: int(max_units)]
        if not self.units:
            raise ValueError(f"no InternData-N1 units for split={split}")
        self.image_size = int(image_size)
        self.maximum_gap = max(1, int(maximum_gap))
        self.history_size = max(2, int(history_size))
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.samples_per_epoch

    @lru_cache(maxsize=64)
    def _episodes(self, unit: str) -> tuple[Path, ...]:
        scene = _safe_relative(self.root, unit)
        paths = tuple(sorted((scene / "data").glob("chunk-*/episode_*.parquet")))
        if not paths:
            raise ValueError(f"no InternData-N1 parquet episodes in {unit}")
        return paths

    def __getitem__(self, index: int) -> dict[str, object]:
        rng = random.Random(f"{self.seed}:{index}")
        for _ in range(12):
            unit = self.units[rng.randrange(len(self.units))]
            episodes = self._episodes(unit)
            parquet = episodes[rng.randrange(len(episodes))]
            table = pq.read_table(
                parquet,
                columns=["observation.camera_extrinsic", "action"],
            ).to_pydict()
            count = len(table["action"])
            if count < 2:
                continue
            gap = rng.randint(1, min(self.maximum_gap, count - 1))
            start = rng.randrange(0, count - gap)
            target = start + gap
            scene = _safe_relative(self.root, unit)
            chunk = parquet.parent.name
            stem = parquet.stem
            rgb_dir = scene / "videos" / chunk / "observation.images.rgb"
            frame0 = rgb_dir / f"{stem}_{start:03d}.jpg"
            frame1 = rgb_dir / f"{stem}_{target:03d}.jpg"
            if not frame0.is_file() or not frame1.is_file():
                continue
            base_from_camera = np.asarray(table["observation.camera_extrinsic"][0]).reshape(4, 4)
            motion = intern_pair_to_canonical_se2(
                table["action"][start], table["action"][target], base_from_camera
            )
            return {
                "task_id": 1,
                "frame0": _rgb_tensor(frame0, self.image_size),
                "frame1": _rgb_tensor(frame1, self.image_size),
                "history": torch.zeros(
                    self.history_size - 1, 3, self.image_size, self.image_size, dtype=torch.float32
                ),
                "history_mask": torch.zeros(self.history_size - 1, dtype=torch.bool),
                "target_bbox": torch.zeros(4, dtype=torch.float32),
                "target_visible": torch.tensor(0.0, dtype=torch.float32),
                "motion": torch.from_numpy(motion),
                "motion_valid": torch.tensor(1.0, dtype=torch.float32),
                "sample_id": f"intern/{unit}/{stem}/{start}-{target}",
                "dataset_id": "intern_data_n1",
            }
        raise RuntimeError(f"could not construct an InternData-N1 pair for index {index}")


class Phase1MultiTaskDataset(Dataset):
    """Mix identity and geometry streams without mixing their supervision."""

    def __init__(
        self,
        identity: ContractIdentityDataset,
        geometry: InternGeometryDataset,
        samples_per_epoch: int,
        identity_fraction: float = 0.5,
        seed: int = 20260907,
    ) -> None:
        if not 0.0 < identity_fraction < 1.0:
            raise ValueError("identity_fraction must be between zero and one")
        self.identity = identity
        self.geometry = geometry
        self.samples_per_epoch = int(samples_per_epoch)
        self.identity_fraction = float(identity_fraction)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, object]:
        digest = hashlib.sha256(f"{self.seed}:{index}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / float(2**64)
        mapped = int.from_bytes(digest[8:16], "big")
        if value < self.identity_fraction:
            return self.identity[mapped % len(self.identity)]
        return self.geometry[mapped % len(self.geometry)]
