"""Pinned train4 perception supervision; no optimizer or product admission.

Only the nineteen original recovery prefixes can receive these label-side
targets. Existing recovery loading/admission remains responsible for waypoint
supervision. This module adds no input channel and never reads raw panoptic.
Torch is imported only when a caller requests CPU tensors or overlays tensors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from PIL import Image


TRAIN4_VERIFICATION_SHA256 = "aec0690b2d2a3727d7e3ba80afce07670876568e49ed2028e48d6545d70cfc5d"
EXPECTED = {
    "at_0401_128steps": ("at", 401, "9hjwm8k7gka", "9", 85),
    "dt_0200_128steps": ("dt", 200, "xxbs57z6pdu", "7", 104),
    "stt_0000_128steps": ("stt", 0, "16tymptm7us", "4", 84),
    "stt_3100_128steps": ("stt", 3100, "qxwfvs8mq67", "2", 93),
}
MODEL_KEYS = frozenset(("initial_rgb", "initial_bbox", "ego_rgb", "visual_initialization_valid",
    "rgb_valid", "binding_valid", "uwb_xy", "uwb_covariance_xy", "uwb_quality", "uwb_age_s",
    "uwb_valid", "camera_intrinsics", "camera_from_base"))
RECOVERY_LABEL_KEYS = frozenset(("supervision_mask", "target_waypoints", "waypoint_mask",
    "stop_label_valid", "binding_label_valid", "identity_label_valid", "ego_label_valid"))
PERCEPTION_LABEL_KEYS = ("target_visible", "visibility_label_valid", "target_bbox", "bbox_label_valid")
UNAVAILABLE = ("stop_label_available", "motion_permission_label_available",
    "visual_identity_prediction_label_available", "binding_label_available", "ego_label_available",
    "occluded_vs_out_of_view_label_available", "later_bbox_used_by_model")


class PerceptionSidecarError(ValueError):
    """Frozen supervision or its matching recovery input is invalid."""


def _require(value, message):
    if not value:
        raise PerceptionSidecarError(message)


def _readonly(for_optimizer):
    _require(type(for_optimizer) is bool and not for_optimizer,
             "perception sidecar has no optimizer admission; for_optimizer=True is forbidden")


def _hash(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "explicit SHA-256 pin required")
    return value


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def _check_file(path, expected):
    path = Path(path).resolve(strict=True)
    _require("test_locked" not in str(path).lower(), "test_locked path is forbidden")
    _require(path.is_file() and _digest(path) == _hash(expected), "frozen artifact SHA-256 mismatch: " + str(path))
    return path


def _decode(text):
    def reject(value):
        raise PerceptionSidecarError("nonfinite JSON: " + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    result = json.loads(text, parse_constant=reject, object_pairs_hook=unique)
    _require(isinstance(result, dict), "JSON object required")
    return result


def _json(path):
    return _decode(Path(path).read_text(encoding="utf-8"))


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _array_hash(value):
    value = np.ascontiguousarray(value)
    result = hashlib.sha256(value.dtype.str.encode("ascii"))
    result.update(repr(value.shape).encode("ascii"))
    result.update(value.tobytes(order="C"))
    return result.hexdigest()


def _finite(value):
    return type(value) in (int, float) and np.isfinite(value)


def _scene(value):
    return re.sub(r"^\d+-", "", str(value).replace("\\", "/").rsplit("/", 1)[-1].split(".")[0]).lower()


def _false(document, keys):
    _require(all(document.get(key) is False for key in keys), "unavailable supervision or eligibility was granted")


def _rgb(path, expected):
    _check_file(path, expected)
    with Image.open(path) as image:
        _require(image.mode == "RGB", "original prefix must be RGB")
        return np.asarray(image).copy()


@dataclass(frozen=True)
class PerceptionSupervision:
    sample_path: Path
    sample_sha256: str
    run_id: str
    sequence_length: int
    learning_start: int
    verification_sha256: str
    plan_file_sha256: str
    labels_sha256: str
    labels: Mapping[str, np.ndarray]
    optimizer_input_allowed: bool = field(default=False, init=False)
    formal_training_eligible: bool = field(default=False, init=False)
    product_acceptance_evidence: bool = field(default=False, init=False)
    _pins: tuple = field(default=(), repr=False)
    _input_assets: tuple = field(default=(), repr=False)
    _model_inputs: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def as_numpy(self, *, for_optimizer=False):
        _readonly(for_optimizer)
        for path, digest in self._pins:
            _check_file(path, digest)
        return {key: value.copy() for key, value in self.labels.items()}

    def as_tensors(self, *, for_optimizer=False):
        """Return unbatched CPU tensors [S], [S,4]; this grants no training use."""
        _readonly(for_optimizer)
        import torch
        return {key: torch.from_numpy(value) for key, value in self.as_numpy(for_optimizer=for_optimizer).items()}


class PerceptionSidecarStore:
    """Explicit externally pinned verification → plan → source/labels → sample.

    ``artifact_root`` is a byte-identical repository mirror. Logical absolute
    paths in the frozen plan are mapped under that root without rewriting any
    document. No fallback search or basename matching is performed.
    """
    optimizer_input_allowed = False
    formal_training_eligible = False
    product_acceptance_evidence = False

    def __init__(self, verification_path, *, verification_sha256, plan_path,
                 plan_file_sha256, artifact_root, for_optimizer=False):
        _readonly(for_optimizer)
        try:
            self.root = Path(artifact_root).resolve(strict=True)
            _require(self.root.is_dir() and "test_locked" not in str(self.root).lower(), "invalid artifact_root")
            self.verification_path = _check_file(verification_path, verification_sha256)
            self.plan_path = _check_file(plan_path, plan_file_sha256)
            self.verification_sha256, self.plan_file_sha256 = verification_sha256, plan_file_sha256
            self._pins = {self.verification_path: verification_sha256, self.plan_path: plan_file_sha256}
            self.verification, self.plan = _json(self.verification_path), _json(self.plan_path)
            self._initialize()
        except PerceptionSidecarError:
            raise
        except (OSError, KeyError, TypeError, IndexError, ValueError) as exc:
            raise PerceptionSidecarError("invalid pinned perception sidecar: " + str(exc)) from exc

    def _resolve(self, logical):
        _require(isinstance(logical, str) and "test_locked" not in logical.lower()
                 and "\\" not in logical, "forbidden logical artifact path")
        path = PurePosixPath(logical)
        _require(path.is_absolute() and ".." not in path.parts and str(path) == logical,
                 "canonical absolute repository path required")
        try:
            relative = path.relative_to(self.repository)
        except ValueError as exc:
            raise PerceptionSidecarError("artifact escapes frozen repository") from exc
        resolved = self.root.joinpath(*relative.parts).resolve(strict=True)
        _require(resolved.is_relative_to(self.root) and resolved.is_file(), "artifact escapes mirror or is missing")
        return resolved

    def _ref(self, ref, *, remember=True):
        _require(self.registry.get(ref["path"]) == ref["sha256"], "consumed artifact missing from plan pins")
        path = _check_file(self._resolve(ref["path"]), ref["sha256"])
        if remember:
            self._pins[path] = ref["sha256"]
        return path

    def _assert_frozen(self):
        for path, digest in self._pins.items():
            _check_file(path, digest)

    def _initialize(self):
        verification, plan = self.verification, self.plan
        _require(verification.get("status") == "verified_development_perception_labels_pending_loader_integration",
                 "independent perception verification did not pass")
        _false(verification, ("optimizer_input_allowed", "formal_training_eligible", "product_acceptance_evidence",
            "partial_admission_allowed", "model_loaded", "gt_training_inputs_generated", "source_files_modified"))
        _require(verification.get("original_inputs_unchanged") is True and verification.get("batch_summary_valid") is True,
                 "original inputs or complete batch were not verified")
        _require(verification.get("source_count") == 4 and verification.get("admitted_observation_count") == 370
                 and verification.get("expected_observation_count") == 370
                 and verification.get("required_distinct_original_samples") == 19, "partial or unexpected admission denominator")
        _require(verification["plan_file_sha256"] == self.plan_file_sha256, "verification refers to a different plan file")
        value = dict(plan)
        seal = value.pop("plan_sha256")
        _require(_canonical(value) == seal == verification["plan_sha256"], "plan canonical seal mismatch")
        _require(plan.get("stage") == "train4_perception_sidecar_collection_v1", "unsupported sidecar stage")
        _false(plan, ("optimizer_input_allowed", "formal_training_eligible", "product_acceptance_evidence", "test_locked_used"))
        self.repository = PurePosixPath(plan["repository"])
        _require(self.repository.is_absolute() and ".." not in self.repository.parts, "invalid logical repository")
        self.registry = {ref["path"]: _hash(ref["sha256"]) for ref in plan["artifacts"]}
        _require(len(self.registry) == len(plan["artifacts"]), "duplicate plan artifact paths")
        manifest = _json(self._ref(plan["current_manifest"]))
        permanent = _json(self._ref(plan["permanent_manifest"]))
        roles = {row["identity"]["scene_id"]: row["partition_role"] for row in permanent["samples"]}
        _require(roles == plan["permanent_scene_roles"], "permanent scene roles differ from plan")
        manifest_rows = {row["artifacts"]["sample"]["path"]: row for row in manifest["samples"]}
        _require(len(manifest_rows) == len(manifest["samples"]), "duplicate manifest sample paths")
        _require([entry["run_id"] for entry in plan["entries"]] == list(EXPECTED), "four original sources required in order")
        verified = {row["run_id"]: row for row in verification["sources"]}
        _require(len(verification["sources"]) == 4 and set(verified) == set(EXPECTED), "incomplete independently verified source set")
        self._samples, self._sources, self._labels = {}, {}, {}
        for entry in plan["entries"]:
            run_id = entry["run_id"]
            task, index, scene, episode, count = EXPECTED[run_id]
            _require((entry["task"], entry["dataset_index"], entry["canonical_scene_id"], entry["episode_id"], entry["action_count"])
                     == EXPECTED[run_id] and entry["observation_count"] == count+1, "source identity or terminal count changed")
            _require(entry["permanent_partition_role"] == "train" and roles.get(scene) == "train", "val/held-out sidecar forbidden")
            source = _json(self._ref(entry["source"]))
            _require(source["assigned_humanoid_semantic_ids"] == entry["assigned_humanoid_semantic_ids"],
                     "source semantic assignment mismatch")
            status = _json(self._ref(entry["source_status"]))
            launch = _json(self._ref(entry["source_launch"]))
            _require(source["split"] == "train" and source["task"] == task and source["dataset_index"] == index
                     and str(source["episode_id"]) == episode and _scene(source["scene_id"]) == scene, "source identity mismatch")
            _require(source["initialization"] == entry["initialization"] == launch["run"]["original_initialization"]
                     and source["initialization"]["environment_step"] == 0, "original target initialization changed")
            _require(status["status"] == "fixed_train_rollout_complete" and type(status["exit_code"]) is int
                     and status["exit_code"] == 0 and status["result_sha256"] == entry["source"]["sha256"], "source execution did not complete")
            _require(source["summary"]["episode_finished"] is True and source["summary"]["termination_reason"] == "episode_over"
                     and source["summary"]["steps"] == count and len(source["steps"]) == count, "source natural terminal missing")
            _require([r["step"] for r in source["steps"]] == list(range(1, count+1)), "source action indices changed")
            actions = [[float(row["policy"]["action"][k]) for k in ("forward", "lateral", "yaw")] for row in source["steps"]]
            _require(np.isfinite(actions).all() and (np.abs(actions) <= 1).all()
                     and _canonical(actions) == entry["saved_actions_sha256"], "saved source action stream changed")
            report = verified[run_id]
            _require(report["status"] == "independently_verified" and report["observations_verified"] == count+1
                     and report["distinct_original_samples"] == len(entry["prefix_samples"]), "partial source verification")
            labels_logical = str(PurePosixPath(entry["output_dir"])/"labels.jsonl")
            _require(PurePosixPath(entry["output_dir"]) == PurePosixPath(plan["output_root"])/run_id,
                     "labels source output identity changed")
            labels_path = _check_file(self._resolve(labels_logical), report["labels_sha256"])
            self._pins[labels_path] = report["labels_sha256"]
            rows = [_decode(line) for line in labels_path.read_text(encoding="utf-8").splitlines()]
            _require(len(rows) == count+1, "labels reset/terminal frame missing")
            self._validate_labels(rows, entry, actions)
            self._sources[run_id], self._labels[run_id] = source, rows
            for reference in entry["prefix_samples"]:
                logical = reference["sample"]["path"]
                _require(logical not in self._samples and logical in manifest_rows, "duplicate or unadmitted prefix")
                original = manifest_rows[logical]
                _require(original["partition_role"] == "train" and original["identity"]["scene_id"] == scene,
                         "sample permanent role/source scene mismatch")
                _require(original["artifacts"]["sample"]["sha256"] == reference["sample"]["sha256"]
                         and original["artifacts"]["rollout"]["path"] == entry["source"]["path"]
                         and original["artifacts"]["rollout"]["sha256"] == entry["source"]["sha256"], "sample/source manifest pins differ")
                self._ref(reference["sample"])
                self._ref(original["artifacts"]["report"])
                self._samples[logical] = (entry, reference, original)
        _require(len(self._samples) == 19 and len(verification["distinct_original_sample_paths"]) == 19
                 and set(self._samples) == set(verification["distinct_original_sample_paths"]), "only all nineteen original prefixes are admitted")

    def _validate_labels(self, rows, entry, actions):
        assigned = entry["assigned_humanoid_semantic_ids"]
        _require("0" in assigned and "1" not in assigned and all(type(v) is int and v > 0 for v in assigned.values())
                 and len(set(assigned.values())) == len(assigned), "ambiguous target assignment")
        previous = -1.
        for step, row in enumerate(rows):
            terminal = step == len(actions)
            _require(type(row["environment_step"]) is int and row["environment_step"] == step
                     and row["policy_call_index"] == step and row["terminal_observation"] is terminal
                     and row["after_source_action_step"] == (None if step == 0 else step)
                     and row["next_source_action_step"] == (None if terminal else step+1), "label action/observation causality changed")
            time = row["world_time_s"]
            _require(_finite(time) and time >= 0 and time > previous, "label world time must increase from reset")
            previous = time
            _false(row, (*UNAVAILABLE, "optimizer_input_allowed", "formal_training_eligible"))
            _require(row["gt_used_only_on_label_or_audit_side"] is True and row["visibility_label_valid"] is True,
                     "visibility label boundary changed")
            audit = row["source_audit"]
            _require(audit["passed"] is True and audit["source_result_sha256"] == entry["source"]["sha256"]
                     and audit["action_just_replayed"] == (None if step == 0 else actions[step-1])
                     and audit["capture_evidence"]["assigned_humanoid_semantic_ids"] == assigned, "label source/action/assignment mismatch")
            target = row["target"]
            _require(target["semantic_id_label_side_only"] == assigned["0"], "label target identity changed")
            height, width = row["image_height"], row["image_width"]
            _require(type(height) is int and type(width) is int and min(height, width) > 0, "invalid label image size")
            visible, area, box = target["visible"], target["mask_area_pixels"], target["bbox_xyxy_inclusive"]
            _require(type(visible) is bool and type(area) is int and 0 <= area <= height*width
                     and visible is (area > 0), "invalid anypixel visibility label")
            if not visible:
                _require(box is None and target["bbox_xyxy_norm"] is None and target["bbox_label_valid"] is False,
                         "invisible target has a fabricated bbox")
            else:
                _require(isinstance(box, list) and len(box) == 4 and all(type(v) is int for v in box)
                         and 0 <= box[0] <= box[2] < width and 0 <= box[1] <= box[3] < height
                         and area <= (box[2]-box[0]+1)*(box[3]-box[1]+1), "invalid inclusive bbox")
                expected = [box[0]/width, box[1]/height, box[2]/width, box[3]/height]
                _require(target["bbox_xyxy_norm"] == expected
                         and target["bbox_label_valid"] is (box[2] > box[0] and box[3] > box[1]), "bbox normalization/independent validity mismatch")

    @property
    def sample_paths(self):
        return tuple(self._resolve(path) for path in self._samples)

    def supervision_for(self, sample_path, *, learning_steps=4, for_optimizer=False):
        _readonly(for_optimizer)
        try:
            _require(type(learning_steps) is int and learning_steps > 0, "learning_steps must be a positive integer")
            self._assert_frozen()
            path = Path(sample_path).resolve(strict=True)
            _require(path.is_relative_to(self.root) and "test_locked" not in str(path).lower(), "sample escapes artifact_root")
            logical = str(self.repository/PurePosixPath(path.relative_to(self.root).as_posix()))
            _require(logical in self._samples, "sample is not one of the nineteen train prefixes; val/legacy/terminal forbidden")
            entry, reference, admitted = self._samples[logical]
            sample = _json(_check_file(path, reference["sample"]["sha256"]))
            source, inputs, contract = sample["source"], sample["model_inputs"], sample["sequence_contract"]
            _require(sample["schema_version"] == 2 and sample["stage"] == "phase3_recovery_sequence_candidate_v2",
                     "recovery sequence schema mismatch")
            _false(sample, ("formal_training_eligible", "test_locked_used"))
            _require(source["split"] == "train" and source["task"] == entry["task"]
                     and source["dataset_index"] == entry["dataset_index"] and str(source["episode_id"]) == entry["episode_id"]
                     and source["scene_id"] == entry["scene_id"]
                     and source["rollout_result"] == entry["source"]["path"]
                     and source["rollout_result_sha256"] == entry["source"]["sha256"], "sample is from another source")
            anchor = source["anchor_environment_step"]
            _require(type(anchor) is int and 0 < anchor < entry["action_count"]
                     and anchor == reference["anchor_environment_step"] == admitted["identity"]["anchor_environment_step"],
                     "terminal or mismatched anchor forbidden")
            size, start = anchor+1, max(0, anchor+1-learning_steps)
            _require(contract["starts_at_episode_reset"] is True and contract["history_size"] == 4
                     and contract["policy_call_indices"] == list(range(size)) and contract["anchor_policy_call_index"] == anchor,
                     "reset-through-anchor sequence required")
            allowed_inputs = {"condition_mode", "initial_rgb", "initial_bbox_xyxy_norm", "prefix_rgb", "policy_calls",
                "visual_initialization_valid", "rgb_valid", "binding_valid", "uwb", "camera_intrinsics", "camera_from_base"}
            _require(set(inputs) == allowed_inputs and inputs["condition_mode"] == "visual_only", "unexpected model input channel")
            _require(inputs["initial_bbox_xyxy_norm"] == entry["initialization"]["bbox_xyxy_norm"], "original target initialization differs")
            frames, calls = inputs["prefix_rgb"], inputs["policy_calls"]
            _require(len(frames) == len(calls) == len(reference["prefix_rgb"]) == size, "incomplete causal prefix")
            labels = {"target_visible": np.zeros(size, dtype=np.float32), "target_bbox": np.zeros((size, 4), dtype=np.float32),
                      "visibility_label_valid": np.zeros(size, dtype=bool), "bbox_label_valid": np.zeros(size, dtype=bool)}
            assets = []
            initial_ref = reference["initial_rgb"]
            self._match_image(inputs["initial_rgb"], initial_ref, logical, path, assets)
            _require(_array_hash(_rgb(*assets[0])) == self._labels[entry["run_id"]][0]["rgb_array_sha256"], "initial RGB is not reset RGB")
            previous_time = -1.
            for step, (frame, call, frozen) in enumerate(zip(frames, calls, reference["prefix_rgb"])):
                row = self._labels[entry["run_id"]][step]
                _require(row["terminal_observation"] is False, "terminal observation cannot supervise a policy call")
                time = frame["world_time_s"]
                _require(frame["environment_step"] == frozen["environment_step"] == step
                         and call["policy_call_index"] == call["observation_environment_step"] == step
                         and call["rgb_history_observation_indices"] == [max(0, j) for j in range(step-3, step+1)]
                         and _finite(time) and time > previous_time
                         and time == frozen["world_time_s"] == call["world_time_s"]
                         and abs(time-row["world_time_s"]) <= 1e-8, "prefix clock/history/observation alignment mismatch")
                previous_time = time
                self._match_image(frame, frozen, logical, path, assets)
                pixels = _rgb(*assets[-1])
                _require(_array_hash(pixels) == frozen["rgb_array_sha256"] == row["rgb_array_sha256"]
                         and pixels.shape[:2] == (row["image_height"], row["image_width"]), "prefix RGB differs from verified same-render labels")
                if step >= start:
                    labels["target_visible"][step] = float(row["target"]["visible"])
                    labels["visibility_label_valid"][step] = True
                    if row["target"]["bbox_label_valid"]:
                        labels["target_bbox"][step] = row["target"]["bbox_xyxy_norm"]
                        labels["bbox_label_valid"][step] = True
            for value in labels.values():
                value.setflags(write=False)
            self._assert_frozen()
            return PerceptionSupervision(path, reference["sample"]["sha256"], entry["run_id"], size, start,
                self.verification_sha256, self.plan_file_sha256,
                next(r["labels_sha256"] for r in self.verification["sources"] if r["run_id"] == entry["run_id"]),
                MappingProxyType(labels), _pins=tuple(self._pins.items()), _input_assets=tuple(assets), _model_inputs=inputs)
        except PerceptionSidecarError:
            raise
        except (OSError, KeyError, TypeError, IndexError, ValueError) as exc:
            raise PerceptionSidecarError("invalid perception prefix: " + str(exc)) from exc

    def _match_image(self, value, reference, sample_logical, sample_path, assets):
        relative = value["rgb_path"]
        _require(isinstance(relative, str) and PurePosixPath(relative).name == relative
                 and "\\" not in relative and relative not in (".", ".."), "RGB path escapes sample directory")
        _require(reference["path"] == str(PurePosixPath(sample_logical).parent/relative)
                 and reference["sha256"] == value["sha256"], "RGB reference does not match original sample")
        path = self._ref(reference, remember=False)
        _require(path.parent == sample_path.parent, "RGB resolved outside sample directory")
        assets.append((path, reference["sha256"]))


def _numpy(value):
    if isinstance(value, np.ndarray):
        return value
    import torch
    _require(torch.is_tensor(value) and value.device.type == "cpu", "loader overlay requires CPU tensors")
    return value.detach().numpy()


def overlay_recovery_sequence(original_batch, supervision: PerceptionSupervision, *, for_optimizer=False):
    """Copy a full [B=1,S,...] batch; preserve inputs and anchor waypoint labels.

    Call this after ``RecoverySequenceItem.as_batch()``. Slice ONLY label keys
    by the same burn-in used by model unroll. Do not feed the augmented sample
    back into the original groups helper, whose contract is anchor-only.
    """
    _readonly(for_optimizer)
    _require(set(original_batch) == MODEL_KEYS | RECOVERY_LABEL_KEYS, "unexpected recovery input/label fields")
    _require(isinstance(supervision, PerceptionSupervision), "verified PerceptionSupervision required")
    values = supervision.as_numpy()
    size = supervision.sequence_length
    mask = _numpy(original_batch["supervision_mask"])
    _require(mask.dtype == np.bool_ and mask.shape == (1, size) and mask.sum() == 1 and mask[0, -1],
             "original recovery batch must supervise only its anchor")
    waypoint_mask = _numpy(original_batch["waypoint_mask"])
    _require(waypoint_mask.dtype == np.bool_ and waypoint_mask.shape == (1, size, 8)
             and not waypoint_mask[:, :-1].any() and waypoint_mask[:, -1].all(), "waypoint mask must stay anchor-only")
    for name in ("stop_label_valid", "binding_label_valid", "identity_label_valid", "ego_label_valid"):
        valid = _numpy(original_batch[name])
        _require(valid.dtype == np.bool_ and valid.shape == (1, size) and not valid.any(), "unknown auxiliary labels must remain invalid")
    waypoint = _numpy(original_batch["target_waypoints"])
    _require(waypoint.shape == (1, size, 8, 2) and np.isfinite(waypoint).all() and not waypoint[:, :-1].any(),
             "prefix contains fabricated waypoint supervision")
    _verify_batch_inputs(original_batch, supervision)
    selected = mask.copy() | values["visibility_label_valid"][None] | values["bbox_label_valid"][None]
    result = dict(original_batch)
    if isinstance(original_batch["supervision_mask"], np.ndarray):
        result.update({key: value[None].copy() for key, value in values.items()})
        result["supervision_mask"] = selected
    else:
        import torch
        result.update({key: torch.from_numpy(value.copy()).unsqueeze(0) for key, value in values.items()})
        result["supervision_mask"] = torch.from_numpy(selected)
    return result


def _verify_batch_inputs(batch, supervision):
    """Bind the supplied tensor batch to the exact original RGB/history inputs."""
    inputs, size = supervision._model_inputs, supervision.sequence_length
    ego, initial = _numpy(batch["ego_rgb"]), _numpy(batch["initial_rgb"])
    _require(ego.ndim == 6 and ego.shape[:4] == (1, size, 4, 3) and ego.dtype == np.float32,
             "full B=1 reset-through-anchor RGB sequence required")
    height, width = ego.shape[-2:]
    _require(min(height, width) > 0 and initial.shape == (1, 3, height, width), "initial RGB shape mismatch")
    def resized(asset):
        pixels = _rgb(*asset)
        array = np.asarray(Image.fromarray(pixels).resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32).copy()/np.float32(255.)
        return array.transpose(2, 0, 1)
    _require(np.array_equal(initial[0], resized(supervision._input_assets[0])), "batch initial RGB belongs to another sample")
    # Cache only recent original frames, not an additional entire history tensor.
    cache = {}
    for step, call in enumerate(inputs["policy_calls"]):
        for history_index, frame_index in enumerate(call["rgb_history_observation_indices"]):
            if frame_index not in cache:
                cache[frame_index] = resized(supervision._input_assets[frame_index+1])
            _require(np.array_equal(ego[0, step, history_index], cache[frame_index]), "batch RGB/history is not the verified prefix")
        cache = {index: value for index, value in cache.items() if index >= step-3}
    expected = {"initial_bbox": inputs["initial_bbox_xyxy_norm"], "visual_initialization_valid": 1., "rgb_valid": 1.,
        "binding_valid": 1., "uwb_xy": [0., 0.], "uwb_covariance_xy": [[1., 0.], [0., 1.]],
        "uwb_quality": 0., "uwb_age_s": 0., "uwb_valid": 0., "camera_from_base": inputs["camera_from_base"]}
    intrinsics = np.array(inputs["camera_intrinsics"], dtype=np.float32)
    calibration_width, calibration_height = 2*float(inputs["camera_intrinsics"][0][2]), 2*float(inputs["camera_intrinsics"][1][2])
    _require(calibration_width > 0 and calibration_height > 0, "original camera calibration missing")
    intrinsics[0] *= width/calibration_width
    intrinsics[1] *= height/calibration_height
    expected["camera_intrinsics"] = intrinsics
    for name, value in expected.items():
        target = np.asarray(value, dtype=np.float32)[None]
        _require(np.array_equal(_numpy(batch[name]), target), "batch model input differs from original sample: " + name)
