"""Label-side semantic observations. Never import into a policy/controller.

Visibility means at least one rendered pixel of the initialized target. It does
not distinguish occlusion from out-of-view, certify identity predictions, or
provide stop/motion-permission labels. Boxes use the existing inclusive pixel
extrema divided by image width/height; degenerate boxes have no bbox label.
"""
from __future__ import annotations

from typing import Mapping
import hashlib
import math

import numpy as np


def array_sha256(value: np.ndarray) -> str:
    """Hash array type, shape and bytes, independently of file encoding."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def instance_label(panoptic: np.ndarray, semantic_id: int) -> dict:
    if type(semantic_id) is not int or semantic_id < 0:
        raise ValueError("semantic id must be a nonnegative integer")
    height, width = panoptic.shape
    yy, xx = np.nonzero(panoptic == semantic_id)
    area = int(xx.size)
    box = None if not area else [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]
    valid_box = box is not None and box[2] > box[0] and box[3] > box[1]
    return {
        "semantic_id_label_side_only": semantic_id,
        "visible": area > 0,
        "mask_area_pixels": area,
        "bbox_xyxy_inclusive": box,
        "bbox_xyxy_norm": None if box is None else
            [box[0] / width, box[1] / height, box[2] / width, box[3] / height],
        "bbox_label_valid": valid_box,
    }


def label_observation(*, rgb: np.ndarray, panoptic: np.ndarray,
                      assigned_humanoid_semantic_ids: Mapping[int, int],
                      environment_step: int, world_time_s: float,
                      terminal_observation: bool) -> dict:
    """Describe one shared RGB/panoptic render at a policy observation index.

    Agent zero is the original initialized target; agent one is the robot.
    Other mapped agents are distractors, without selecting by model confidence.
    Source assignment/provenance must be verified separately by the collector.
    """
    if type(environment_step) is not int or environment_step < 0:
        raise ValueError("environment_step must be a nonnegative integer")
    if isinstance(world_time_s, bool) or not isinstance(world_time_s, (int, float)) or not math.isfinite(world_time_s) or world_time_s < 0:
        raise ValueError("world time must be finite and nonnegative")
    if type(terminal_observation) is not bool:
        raise ValueError("terminal_observation must be boolean")
    rgb, panoptic = np.asarray(rgb), np.asarray(panoptic)
    if panoptic.ndim == 3 and panoptic.shape[-1] == 1:
        panoptic = panoptic[..., 0]
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8 or not min(rgb.shape[:2]):
        raise ValueError("RGB must be nonempty uint8 HxWx3")
    if panoptic.ndim != 2 or panoptic.shape != rgb.shape[:2] or panoptic.dtype.kind not in "iu":
        raise ValueError("panoptic must be a spatially aligned integer HxW array")
    if np.any(panoptic < 0):
        raise ValueError("negative semantic labels are unsupported")
    assigned = dict(assigned_humanoid_semantic_ids)
    if 0 not in assigned or 1 in assigned or any(type(k) is not int or k < 0 for k in assigned):
        raise ValueError("assignment must contain target agent 0 and exclude robot agent 1")
    if any(type(v) is not int or v < 0 for v in assigned.values()) or len(set(assigned.values())) != len(assigned):
        raise ValueError("humanoid semantic IDs must be unique nonnegative integers")
    target = instance_label(panoptic, assigned[0])
    return {
        "environment_step": environment_step,
        "policy_call_index": environment_step,
        "after_source_action_step": None if environment_step == 0 else environment_step,
        "next_source_action_step": None if terminal_observation else environment_step + 1,
        "world_time_s": float(world_time_s),
        "terminal_observation": terminal_observation,
        "rgb_array_sha256": array_sha256(rgb),
        "panoptic_array_sha256": array_sha256(panoptic),
        "image_height": int(rgb.shape[0]), "image_width": int(rgb.shape[1]),
        "target": target,
        "distractors": [dict(agent_index=k, **instance_label(panoptic, v))
                        for k, v in sorted(assigned.items()) if k != 0],
        "visibility_label_valid": True,
        "visibility_definition": "any_initialized_target_semantic_pixel",
        "occluded_vs_out_of_view_label_available": False,
        "stop_label_available": False,
        "motion_permission_label_available": False,
        "visual_identity_prediction_label_available": False,
        "binding_label_available": False,
        "ego_label_available": False,
        "gt_used_only_on_label_or_audit_side": True,
        "later_bbox_used_by_model": False,
    }
