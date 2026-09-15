"""Architecture-v1-isomorphic Phase 1 identity and world-action data."""
from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from omtrackvla.data.end_to_end import _rgb_tensor, sage_camera_calibration
from omtrackvla.data.end_to_end_training import _relative_motion
from omtrackvla.data.phase1 import (
    ContractIdentityDataset,
    InternGeometryDataset,
    _safe_relative,
)
from omtrackvla.data.sage3d_policy import target_position_base
from omtrackvla.models.end_to_end import ArchitectureV1Config


def _xi4(motion: torch.Tensor) -> torch.Tensor:
    """Convert canonical ``[dx,dy,dyaw]`` to the supervised bottleneck target."""

    return torch.stack(
        (motion[0], motion[1], torch.sin(motion[2]), torch.cos(motion[2]))
    )


def _neutral_calibration(
    config: ArchitectureV1Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a finite placeholder calibration for samples with UWB disabled."""

    intrinsics = torch.tensor(
        [
            [float(config.image_width), 0.0, config.image_width / 2.0],
            [0.0, float(config.image_width), config.image_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    camera_from_base = torch.eye(4, dtype=torch.float32)
    camera_from_base[:3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    return intrinsics, camera_from_base


def _left_padded_window(
    frames: Sequence[torch.Tensor], history_size: int
) -> torch.Tensor:
    if not frames:
        raise ValueError("an Architecture v1 history window cannot be empty")
    selected = list(frames[-history_size:])
    selected = [selected[0]] * (history_size - len(selected)) + selected
    return torch.stack(selected)


def _common_inputs(
    *,
    initial_rgb: torch.Tensor,
    initial_bbox: torch.Tensor,
    ego_rgb: torch.Tensor,
    visual_valid: float,
    binding_valid: float,
    intrinsics: torch.Tensor,
    camera_from_base: torch.Tensor,
) -> dict[str, torch.Tensor]:
    sequence_steps = int(ego_rgb.shape[0])
    return {
        "initial_rgb": initial_rgb,
        "initial_bbox": initial_bbox,
        "ego_rgb": ego_rgb,
        "visual_initialization_valid": torch.tensor(
            visual_valid, dtype=torch.float32
        ),
        "rgb_valid": torch.ones(sequence_steps, dtype=torch.float32),
        "binding_valid": torch.full(
            (sequence_steps,), binding_valid, dtype=torch.float32
        ),
        "uwb_xy": torch.zeros(sequence_steps, 2, dtype=torch.float32),
        "uwb_covariance_xy": torch.zeros(
            sequence_steps, 2, 2, dtype=torch.float32
        ),
        "uwb_quality": torch.zeros(sequence_steps, dtype=torch.float32),
        "uwb_age_s": torch.zeros(sequence_steps, dtype=torch.float32),
        "uwb_valid": torch.zeros(sequence_steps, dtype=torch.float32),
        "camera_intrinsics": intrinsics,
        "camera_from_base": camera_from_base,
    }


class ArchitectureV1IdentityDataset(Dataset):
    """Adapt admitted SAGE3D/TpT identity records to the deployment input graph."""

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        split: str,
        roots: Mapping[str, str | Path],
        datasets: Sequence[str],
        sage3d_sidecar: str | Path,
        config: ArchitectureV1Config,
        max_units_per_dataset: int | None = None,
    ) -> None:
        self.config = config
        # One initialization frame plus five recent frames is sufficient to form
        # the two overlapping T=4 policy windows used by the Phase 1 transition.
        self.base = ContractIdentityDataset(
            manifest_path=manifest_path,
            split=split,
            roots=roots,
            datasets=datasets,
            sage3d_sidecar=sage3d_sidecar,
            image_size=config.image_width,
            history_size=config.history_size + 2,
            max_units_per_dataset=max_units_per_dataset,
        )

    def __len__(self) -> int:
        return len(self.base)

    def balanced_index(self, value: int) -> int:
        return self.base.balanced_index(value)

    @staticmethod
    @lru_cache(maxsize=64)
    def _derived(path: str) -> dict[str, object]:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("steps"), list):
            raise ValueError(f"invalid SAGE3D derived record: {path}")
        return value

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.base.get_record(index)
        dataset_id = str(record["source"]["dataset_id"])
        root = self.base.roots[dataset_id]
        visual = record["model_inputs"]["visual_initialization"]
        initial_path = _safe_relative(root, visual["rgb_path"])
        history_paths = [
            _safe_relative(root, frame["rgb_path"])
            for frame in record["model_inputs"]["rgb_history"]
        ]
        frames = [_rgb_tensor(path, self.config) for path in history_paths]
        if len(frames) < 2:
            raise ValueError("identity record must contain initialization and current RGB")
        state0 = _left_padded_window(frames[:-1], self.config.history_size)
        state1 = _left_padded_window(frames, self.config.history_size)
        ego_rgb = torch.stack((state0, state1))

        intrinsics, camera_from_base = _neutral_calibration(self.config)
        transition_action = torch.zeros(1, 3, dtype=torch.float32)
        transition_valid = torch.zeros(1, dtype=torch.float32)
        future_target_xy = torch.zeros(1, 2, dtype=torch.float32)
        future_target_xy_valid = torch.zeros(1, dtype=torch.float32)
        future_target_visible = torch.zeros(1, dtype=torch.float32)
        future_target_visibility_valid = torch.zeros(1, dtype=torch.float32)
        ego_motion_target = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        ego_motion_valid = torch.zeros(2, dtype=torch.float32)

        if dataset_id == "sage3d_extracted":
            derived_path = _safe_relative(root, record["provenance"]["source_record"])
            derived = self._derived(str(derived_path))
            steps = derived["steps"]
            anchor = int(record["source"]["anchor_index"])
            if anchor <= 0 or anchor >= len(steps):
                raise ValueError("SAGE3D identity anchor cannot form a transition")
            motion_xi = torch.tensor(
                _relative_motion(steps[anchor - 1], steps[anchor]),
                dtype=torch.float32,
            )
            transition_action[0] = torch.stack(
                (
                    motion_xi[0],
                    motion_xi[1],
                    torch.atan2(motion_xi[2], motion_xi[3]),
                )
            )
            transition_valid[0] = 1.0
            ego_motion_target[1] = motion_xi
            ego_motion_valid[1] = 1.0
            future_target_xy[0] = torch.tensor(
                target_position_base(steps[anchor]), dtype=torch.float32
            )
            future_target_xy_valid[0] = 1.0
            future_target_visible[0] = float(
                record["supervision"]["auxiliary_labels"]["target_visible"]
            )
            future_target_visibility_valid[0] = 1.0
            camera_path = derived_path.parent / "camera_info.json"
            intrinsics, camera_from_base = sage_camera_calibration(
                camera_path, self.config
            )

        labels = record["supervision"]["auxiliary_labels"]
        target_bbox = labels["target_bbox_xyxy_norm"]
        result: dict[str, object] = _common_inputs(
            initial_rgb=_rgb_tensor(initial_path, self.config),
            initial_bbox=torch.tensor(
                visual["bbox_xyxy_norm"], dtype=torch.float32
            ),
            ego_rgb=ego_rgb,
            visual_valid=1.0,
            binding_valid=1.0,
            intrinsics=intrinsics,
            camera_from_base=camera_from_base,
        )
        result.update(
            {
                "identity_label_valid": torch.tensor([0.0, 1.0]),
                "target_bbox": torch.tensor(
                    [[0.0] * 4, target_bbox if target_bbox is not None else [0.0] * 4],
                    dtype=torch.float32,
                ),
                "target_visible": torch.tensor(
                    [0.0, float(labels["target_visible"])], dtype=torch.float32
                ),
                "ego_motion_target": ego_motion_target,
                "ego_motion_valid": ego_motion_valid,
                "transition_action": transition_action,
                "transition_valid": transition_valid,
                "future_target_xy": future_target_xy,
                "future_target_xy_valid": future_target_xy_valid,
                "future_target_visible": future_target_visible,
                "future_target_visibility_valid": future_target_visibility_valid,
                "sample_role": "identity_auxiliary",
                "sample_id": record["sample_id"],
                "dataset_id": dataset_id,
            }
        )
        return result


class ArchitectureV1InternGeometryDataset(Dataset):
    """Adapt Intern pose pairs to two Architecture v1 deployment steps."""

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        split: str,
        root: str | Path,
        config: ArchitectureV1Config,
        maximum_gap: int,
        samples_per_epoch: int,
        seed: int,
        max_units: int | None = None,
    ) -> None:
        self.config = config
        self.base = InternGeometryDataset(
            manifest_path=manifest_path,
            split=split,
            root=root,
            image_size=config.image_width,
            maximum_gap=maximum_gap,
            history_size=2,
            samples_per_epoch=samples_per_epoch,
            seed=seed,
            max_units=max_units,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, object]:
        source = self.base[index]
        frames = F.interpolate(
            torch.stack((source["frame0"], source["frame1"])),
            size=(self.config.image_height, self.config.image_width),
            mode="bilinear",
            align_corners=False,
        )
        frame0, frame1 = frames[0], frames[1]
        state0 = torch.stack([frame0] * self.config.history_size)
        state1_frames = [frame0] * (self.config.history_size - 1) + [frame1]
        ego_rgb = torch.stack((state0, torch.stack(state1_frames)))
        motion = source["motion"].float()
        intrinsics, camera_from_base = _neutral_calibration(self.config)
        result: dict[str, object] = _common_inputs(
            initial_rgb=frame0,
            initial_bbox=torch.zeros(4, dtype=torch.float32),
            ego_rgb=ego_rgb,
            visual_valid=0.0,
            binding_valid=0.0,
            intrinsics=intrinsics,
            camera_from_base=camera_from_base,
        )
        result.update(
            {
                "identity_label_valid": torch.zeros(2, dtype=torch.float32),
                "target_bbox": torch.zeros(2, 4, dtype=torch.float32),
                "target_visible": torch.zeros(2, dtype=torch.float32),
                "ego_motion_target": torch.stack(
                    (torch.tensor([0.0, 0.0, 0.0, 1.0]), _xi4(motion))
                ),
                "ego_motion_valid": torch.tensor([0.0, 1.0]),
                "transition_action": motion[None],
                "transition_valid": torch.ones(1, dtype=torch.float32),
                "future_target_xy": torch.zeros(1, 2, dtype=torch.float32),
                "future_target_xy_valid": torch.zeros(1, dtype=torch.float32),
                "future_target_visible": torch.zeros(1, dtype=torch.float32),
                "future_target_visibility_valid": torch.zeros(
                    1, dtype=torch.float32
                ),
                "sample_role": "geometry_world_action_auxiliary",
                "sample_id": source["sample_id"],
                "dataset_id": "intern_data_n1",
            }
        )
        return result


class ArchitectureV1Phase1Dataset(Dataset):
    """Deterministically mix equal-source identity data with Intern geometry."""

    def __init__(
        self,
        identity: ArchitectureV1IdentityDataset,
        geometry: ArchitectureV1InternGeometryDataset,
        *,
        samples_per_epoch: int,
        identity_fraction: float,
        identity_sampling: str,
        seed: int,
    ) -> None:
        if not 0.0 < identity_fraction < 1.0:
            raise ValueError("identity_fraction must be strictly between zero and one")
        if identity_sampling not in {"proportional", "uniform_by_dataset"}:
            raise ValueError(f"unsupported identity sampling: {identity_sampling}")
        self.identity = identity
        self.geometry = geometry
        self.samples_per_epoch = int(samples_per_epoch)
        self.identity_fraction = float(identity_fraction)
        self.identity_sampling = identity_sampling
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, object]:
        digest = hashlib.sha256(f"{self.seed}:{index}".encode()).digest()
        selector = int.from_bytes(digest[:8], "big") / float(2**64)
        mapped = int.from_bytes(digest[8:16], "big")
        if selector < self.identity_fraction:
            identity_index = (
                self.identity.balanced_index(mapped)
                if self.identity_sampling == "uniform_by_dataset"
                else mapped % len(self.identity)
            )
            return self.identity[identity_index]
        return self.geometry[mapped % len(self.geometry)]
