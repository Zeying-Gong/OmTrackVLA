"""Admitted variable-length recovery prefixes without padded policy calls.

Length groups are a storage/loading detail. Sampling is uniform over the fixed
manifest sample index, never uniform over groups or weighted by success. This
module does not create data, split scenes, add labels, or start training.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from omtrackvla.data.recovery_sequence import (
    RecoverySequenceDataset, recovery_sequence_collate, validate_recovery_sample,
)
from scripts.build_recovery_scene_manifest import sha256, verify_manifest


MODEL_KEYS = (
    "initial_rgb", "initial_bbox", "ego_rgb", "visual_initialization_valid",
    "rgb_valid", "binding_valid", "uwb_xy", "uwb_covariance_xy", "uwb_quality",
    "uwb_age_s", "uwb_valid", "camera_intrinsics", "camera_from_base",
)
LABEL_KEYS = (
    "target_waypoints", "waypoint_mask", "supervision_mask", "stop_label_valid",
    "binding_label_valid", "identity_label_valid", "ego_label_valid",
)
AUXILIARY_VALID_KEYS = LABEL_KEYS[3:]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _snapshot(path, expected=None):
    path = Path(path).resolve(strict=True)
    _require("test_locked" not in str(path).lower(), "test_locked is forbidden")
    digest = sha256(path)
    _require(expected is None or digest == expected, "frozen manifest/scene artifact changed")
    return path, digest


@dataclass(frozen=True)
class RecoverySequenceIdentity:
    manifest_index: int
    sample_path: Path
    sample_sha256: str
    sample_id: str
    task: str
    episode_id: str
    dataset_index: int
    source_scene_id: str
    canonical_scene_id: str
    partition_role: str
    sequence_length: int
    history_size: int
    anchor_policy_call_index: int
    manifest_sha256: str
    scene_plan_sha256: str


@dataclass(frozen=True)
class RecoveryLengthGroup:
    sequence_length: int
    history_size: int
    # Indices refer to the manifest order, not a reordered training dataset.
    manifest_indices: tuple[int, ...]


@dataclass(frozen=True)
class RecoverySequenceBatch:
    # All real policy calls, including the complete burn-in prefix.
    batch: Mapping[str, Any]
    # Only supervision tensors are sliced to align with model outputs.
    labels: Mapping[str, Any]
    burn_in_steps: int
    identity: RecoverySequenceIdentity


@dataclass(frozen=True)
class RecoverySequenceItem:
    sample: Mapping[str, Any]
    burn_in_steps: int
    identity: RecoverySequenceIdentity
    optimizer_input_allowed: bool

    def as_batch(self, *, for_optimizer=False) -> RecoverySequenceBatch:
        """Return batch size one, with every observation and aligned labels.

        ``sequence_model_inputs(result.batch)`` passes the full prefix to the
        unroll. Feed ``result.labels`` to the loss AFTER that unroll uses
        ``result.burn_in_steps``. No metadata is part of model inputs.
        """
        if for_optimizer:
            _require(self.optimizer_input_allowed, "validation/read-only item cannot be optimizer input")
        _validate_tensor_contract(self.sample, self.identity)
        batch = recovery_sequence_collate([self.sample])
        labels = {key: batch[key][:, self.burn_in_steps:] for key in LABEL_KEYS}
        expected_steps = min(4, self.identity.sequence_length)
        _require(labels["supervision_mask"].shape == (1, expected_steps), "burn-in/label suffix mismatch")
        _require(labels["supervision_mask"].sum().item() == 1
                 and labels["supervision_mask"][0, -1].item(), "suffix must supervise only the real anchor")
        return RecoverySequenceBatch(batch, labels, self.burn_in_steps, self.identity)


def _validate_tensor_contract(sample, identity):
    """Check loader output without deriving any additional target labels."""
    import torch
    _require(set(sample) == set(MODEL_KEYS) | set(LABEL_KEYS), "unexpected model input or label fields")
    steps, history = identity.sequence_length, identity.history_size
    rgb = sample["ego_rgb"]
    _require(rgb.ndim == 5 and tuple(rgb.shape[:3]) == (steps, history, 3),
             "real RGB prefix/history shape changed; padding is forbidden")
    _require(identity.anchor_policy_call_index == steps - 1, "anchor must be the last real call")
    supervised = sample["supervision_mask"]
    _require(supervised.dtype == torch.bool and tuple(supervised.shape) == (steps,)
             and supervised.sum().item() == 1 and supervised[-1].item(), "only the real anchor may be supervised")
    mask = sample["waypoint_mask"]
    _require(mask.dtype == torch.bool and tuple(mask.shape) == (steps, 8)
             and not mask[:-1].any().item() and mask[-1].all().item(), "prefix waypoint labels must remain invalid")
    waypoints = sample["target_waypoints"]
    _require(tuple(waypoints.shape) == (steps, 8, 2) and torch.isfinite(waypoints).all().item()
             and not waypoints[:-1].any().item(), "prefix contains invented waypoint labels")
    for key in AUXILIARY_VALID_KEYS:
        value = sample[key]
        _require(value.dtype == torch.bool and tuple(value.shape) == (steps,)
                 and not value.any().item(), "recovery has no stop/visibility/binding/ego labels")


class VariableRecoverySequences:
    """Map-style loader for an explicit frozen manifest partition.

    Each actual (policy-call count, history count) is loaded with the existing
    homogeneous RecoverySequenceDataset. The public index remains exactly the
    partition's manifest order. ``sample_uniform(rng)`` uses a caller-owned RNG
    so the trainer can capture/restore its state without a hidden sampler.

    Train access requires ``partition_role='train', for_training=True``. Val
    permits observation/evaluation only; explicit optimizer APIs reject it.
    As with any tensor API, callers remain responsible for using eval/no_grad
    while evaluating. This helper cannot police arbitrary downstream code.
    """

    def __init__(self, manifest_path, *, partition_role, for_training=False,
                 artifact_root=None, image_height=280, image_width=504):
        _require(partition_role in ("train", "val"), "explicit train or val partition_role is required")
        _require(type(for_training) is bool, "for_training must be an explicit boolean")
        _require(not for_training or partition_role == "train", "validation cannot be optimizer input")
        validator = lambda path: validate_recovery_sample(path, artifact_root=artifact_root)
        manifest, paths = verify_manifest(manifest_path, validator=validator,
                                          role=partition_role, for_training=for_training)
        _require(bool(paths), "selected manifest partition is empty")
        self.partition_role, self.for_training = partition_role, for_training
        self.manifest_path, self.manifest_sha256 = _snapshot(manifest_path)
        _require(json.loads(self.manifest_path.read_text(encoding="utf-8")) == manifest,
                 "manifest changed during verification")
        plan_path, plan_sha256 = _snapshot(manifest["plan"]["path"], manifest["plan"]["sha256"])
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        self._snapshots = [(self.manifest_path, self.manifest_sha256), (plan_path, plan_sha256)]
        self._snapshots.extend(_snapshot(value["path"], value["sha256"]) for value in plan["prior_manifests"])
        records = {Path(record["artifacts"]["sample"]["path"]).resolve(strict=True): record
                   for record in manifest["samples"]}
        identities = []
        grouped_paths, grouped_indices = {}, {}
        locations = []
        for index, path in enumerate(paths):
            path = path.resolve(strict=True)
            record = records[path]
            evidence = record["validation_evidence"]
            _require(record["partition_role"] == partition_role
                     and record["optimizer_input_allowed"] is (partition_role == "train"), "manifest partition drift")
            steps, history = evidence["sequence_length"], evidence["history_size"]
            _require(type(steps) is int and steps >= 2 and type(history) is int and history >= 1,
                     "invalid validated sequence length")
            anchor = evidence["anchor_environment_step"]
            _require(anchor == steps - 1, "validated prefix must contain reset through anchor")
            canonical_scene = record["identity"]["scene_id"]
            _require(plan["scene_partitions"].get(canonical_scene) == partition_role,
                     "permanent scene partition changed")
            identity = RecoverySequenceIdentity(
                index, path, record["artifacts"]["sample"]["sha256"], evidence["sample_id"],
                evidence["task"], evidence["episode_id"], evidence["dataset_index"], evidence["scene_id"],
                canonical_scene, partition_role, steps, history, anchor, self.manifest_sha256, plan_sha256)
            identities.append(identity)
            key = steps, history
            group = grouped_paths.setdefault(key, [])
            locations.append((key, len(group)))
            group.append(path)
            grouped_indices.setdefault(key, []).append(index)
        self.identities = tuple(identities)
        self.groups = tuple(RecoveryLengthGroup(*key, tuple(indices)) for key, indices in grouped_indices.items())
        self._locations = tuple(locations)
        self._datasets = {
            key: RecoverySequenceDataset(group, artifact_root=artifact_root,
                image_height=image_height, image_width=image_width, required_anchor=None,
                development_manifest=self.manifest_path, partition_role=partition_role)
            for key, group in grouped_paths.items()
        }
        self._assert_frozen()

    def _assert_frozen(self):
        for path, digest in self._snapshots:
            _snapshot(path, digest)

    def __len__(self):
        return len(self.identities)

    def __getitem__(self, index) -> RecoverySequenceItem:
        _require(type(index) is int and 0 <= index < len(self), "manifest sample index is outside the partition")
        self._assert_frozen()
        key, local_index = self._locations[index]
        sample = self._datasets[key][local_index]
        identity = self.identities[index]
        _validate_tensor_contract(sample, identity)
        return RecoverySequenceItem(sample, max(0, identity.sequence_length - 4), identity,
                                    self.for_training and self.partition_role == "train")

    def optimizer_item(self, index) -> RecoverySequenceItem:
        _require(self.for_training and self.partition_role == "train", "validation/read-only partition cannot be optimizer input")
        return self[index]

    def sample_uniform(self, rng, *, for_optimizer=False) -> RecoverySequenceItem:
        """One sample uniformly with replacement; RNG state belongs to caller."""
        if for_optimizer:
            _require(self.for_training and self.partition_role == "train", "validation/read-only partition cannot be optimizer input")
        index = rng.randrange(len(self))
        return self.optimizer_item(index) if for_optimizer else self[index]
