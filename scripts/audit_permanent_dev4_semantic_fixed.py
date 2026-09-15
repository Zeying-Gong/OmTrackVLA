"""Independent CPU admission audit for the corrected permanent dev4 capture."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image


REPO = Path("/data/nfs/share/wam_tracking/OmTrackVLA")
ROOT = REPO / "outputs/takeover/permanent_dev4_semantic_fixed_collection_v1"
BUNDLE = REPO / ".codex_upload/permanent_dev4_semantic_fixed_collection_v1"
OUTPUT = REPO / "outputs/takeover/permanent_dev4_semantic_fixed_independent_audit_v1"
PLAN_INTERNAL_SHA256 = "2f93655f838233eee8c2d1ec62924ab2d9cda400453ad993b38ae1a8f5af98e4"
BATCH_STATUS_SHA256 = "ee373af1802e006e0db2daf8ac53b40fdbede011d836bd130e47577496cac9fc"
BATCH_ARTIFACTS_SHA256 = "ce8a47ba304462eb3c4806efab36cb7cecfcd82d5e1e4cbeb109d1c96b18f62e"
QUALIFICATION_SHA256 = "25c96e6727ad94a5fc547cf2a3be2a9c6a66655f962412d17d0f44afe417eba2"
RGB_KEY = "agent_1_articulated_agent_jaw_rgb"
PANOPTIC_KEY = "agent_1_articulated_agent_jaw_panoptic"

sys.path.insert(0, str(BUNDLE))
import val_runtime_contract as contract  # noqa: E402


def need(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(repr(value.shape).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def reference(path: Path) -> dict:
    path = path.resolve(strict=True)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def verify_manifest(root: Path, manifest: dict) -> int:
    total = 0
    for relative, record in manifest.items():
        need("test_locked" not in relative.lower(), "forbidden artifact path")
        path = (root / relative).resolve(strict=True)
        need(path.is_relative_to(root.resolve()) and path.is_file(),
             "artifact escapes or is missing: " + relative)
        need(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"],
             "artifact changed: " + relative)
        total += record["bytes"]
    return total


def bbox(mask: np.ndarray) -> tuple[int, list[int] | None, list[float] | None, bool]:
    yy, xx = np.nonzero(mask)
    area = int(len(xx))
    box = [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())] if area else None
    normal = [value / 384.0 for value in box] if box is not None else None
    valid = bool(box is not None and box[2] > box[0] and box[3] > box[1])
    return area, box, normal, valid


def check_raster_label(value: dict, panoptic: np.ndarray) -> None:
    area, box, normal, valid = bbox(panoptic == value["semantic_id_label_side_only"])
    need(value["mask_area_pixels"] == area and value["visible"] is bool(area) and
         value["bbox_xyxy_inclusive"] == box and value["bbox_label_valid"] is valid,
         "semantic raster label differs")
    if normal is None:
        need(value["bbox_xyxy_norm"] is None, "missing raster has normalized bbox")
    else:
        need(len(value["bbox_xyxy_norm"]) == 4 and
             max(abs(a - b) for a, b in zip(value["bbox_xyxy_norm"], normal)) < 1e-12,
             "normalized raster bbox differs")


def check_camera_pair(pair: dict) -> None:
    need(set(pair) == {RGB_KEY, PANOPTIC_KEY}, "actual camera keys changed")
    need(pair[RGB_KEY]["sensor_type"] == "SensorType.COLOR" and
         pair[PANOPTIC_KEY]["sensor_type"] == "SensorType.SEMANTIC" and
         pair[PANOPTIC_KEY]["semantic_target"] == "SemanticSensorTarget.SEMANTIC_ID",
         "actual camera sensor target/type changed")
    for field in ("absolute_transformation", "projection_matrix"):
        arrays = [np.asarray(pair[key][field], dtype=np.float64)
                  for key in (RGB_KEY, PANOPTIC_KEY)]
        need(all(value.shape == (4, 4) and np.isfinite(value).all() for value in arrays),
             "invalid actual camera matrix")
        need(np.allclose(arrays[0], arrays[1], rtol=0, atol=1e-6),
             "RGB/panoptic camera matrices differ")


def check_camera_stability(before: dict, after: dict) -> None:
    check_camera_pair(before)
    check_camera_pair(after)
    for key in (RGB_KEY, PANOPTIC_KEY):
        need(before[key]["sensor_type"] == after[key]["sensor_type"] and
             before[key]["semantic_target"] == after[key]["semantic_target"],
             "actual camera specification changed during render")
        for field in ("absolute_transformation", "projection_matrix"):
            need(np.allclose(np.asarray(before[key][field]), np.asarray(after[key][field]),
                             rtol=0, atol=1e-6),
                 "actual camera changed during render")


def check_assignment(readback: dict, assigned: dict, *, require_source_recheck: bool) -> None:
    need(readback["field_readback_verified"] is True and
         readback["method"] == "AO_creation_attributes_root_private_template_and_source_refs",
         "semantic assignment readback is not qualified")
    actors = readback["actors"]
    expected = {int(key): int(value) for key, value in assigned.items()}
    actual = {int(row["actor_index"]): int(row["semantic_id"]) for row in actors}
    need(actual == expected and len(set(actual.values())) == len(actual),
         "semantic actor mapping changed")
    for row in actors:
        sid = int(row["semantic_id"])
        need(row["root_semantic_id"] == sid and
             row["creation_attributes"]["semantic_id"] == sid and
             row["private_template_attributes"]["semantic_id"] == sid,
             "private AO semantic assignment evidence differs")
        need(row["original_source_bytes_rechecked"] is require_source_recheck,
             "AO source recheck schedule differs")


def audit_case(entry: dict, status_row: dict, batch_manifest: dict) -> dict:
    case_id = entry["case_id"]
    directory = ROOT / case_id
    need(status_row["case_id"] == case_id and status_row["worker_reaped"] is True and
         status_row["worker_unreaped"] is False and status_row["artifacts_sealed"] is True and
         status_row["status"] == "val_collected_pending_independent_admission" and
         status_row["exit_code"] == 0, "case execution did not finish cleanly")
    case_manifest_path = directory / "val_case_artifacts.json"
    need(sha(case_manifest_path) == status_row["case_artifacts_sha256"],
         "case seal digest differs")
    case_manifest = load(case_manifest_path)
    verify_manifest(directory, case_manifest)
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
    need(actual == set(case_manifest) | {"val_case_artifacts.json", "val_case_execution.json"},
         "case has unsealed extra or missing files")
    execution = load(directory / "val_case_execution.json")
    need(execution == status_row, "case execution/status row changed")
    worker_manifest_path = directory / "val_worker_artifacts.json"
    worker_manifest = load(worker_manifest_path)
    verify_manifest(directory, worker_manifest)
    worker = load(directory / "val_worker_result.json")
    need(sha(worker_manifest_path) == worker["worker_artifacts_sha256"] and
         sha(directory / "val_worker_result.json") == status_row["worker_result_sha256"] and
         worker["status"] == "val_collected_pending_independent_admission" and
         worker["case_id"] == case_id and worker["plan_sha256"] == PLAN_INTERNAL_SHA256,
         "worker result/seal binding differs")

    labels = jsonl(directory / "labels.jsonl")
    need(len(labels) == entry["observation_count"] == status_row["expected_observations"] and
         status_row["expected_actions"] + 1 == len(labels),
         "case observation/action count differs")
    assigned = entry["assigned_humanoid_semantic_ids"]
    visible = 0
    valid_bbox = 0
    target_pixels = 0
    distractor_visible = 0
    world_times = []
    for index, row in enumerate(labels):
        need(row["environment_step"] == row["policy_call_index"] == index and
             row["image_height"] == row["image_width"] == 384,
             "label step/image geometry changed")
        need(row["optimizer_input_allowed"] is False and
             row["formal_training_eligible"] is False and
             row["gt_used_only_on_label_or_audit_side"] is True and
             row["later_bbox_used_by_model"] is False,
             "collection boundary changed")
        rgb_rel = row["rgb_file"]["path"]
        pan_rel = row["raw_panoptic_file"]["path"]
        need(rgb_rel == f"observations/rgb_{index:04d}.png" and
             pan_rel == f"observations/panoptic_{index:04d}.npy",
             "cross-step media reference")
        rgb_path = directory / rgb_rel
        pan_path = directory / pan_rel
        need(sha(rgb_path) == row["rgb_file"]["sha256"] ==
             batch_manifest[f"{case_id}/{rgb_rel}"]["sha256"] and
             sha(pan_path) == row["raw_panoptic_file"]["sha256"] ==
             batch_manifest[f"{case_id}/{pan_rel}"]["sha256"],
             "label media file digest differs")
        rgb = np.asarray(Image.open(rgb_path))
        raw = np.load(pan_path, allow_pickle=False)
        panoptic = raw[:, :, 0] if raw.ndim == 3 and raw.shape[-1] == 1 else raw
        need(rgb.shape == (384, 384, 3) and rgb.dtype == np.uint8 and
             panoptic.shape == (384, 384) and raw.dtype.kind in "iu",
             "label media shape/dtype changed")
        need(array_sha(rgb) == row["rgb_array_sha256"] and
             array_sha(raw) == row["raw_panoptic_file"]["raw_array_sha256"] and
             array_sha(panoptic) == row["panoptic_array_sha256"],
             "label media array digest differs")
        need(row["target"]["semantic_id_label_side_only"] == int(assigned["0"]),
             "main target semantic ID changed")
        check_raster_label(row["target"], panoptic)
        for distractor in row["distractors"]:
            need(distractor["semantic_id_label_side_only"] ==
                 int(assigned[str(distractor["agent_index"])]),
                 "distractor semantic ID changed")
            check_raster_label(distractor, panoptic)
        audit = row["source_audit"]
        need(audit["passed"] is True, "collector hard alignment check failed")
        capture = audit["capture_evidence"]
        need(capture["renderer_qualification_sha256"] == QUALIFICATION_SHA256 and
             capture["assigned_humanoid_semantic_ids"] == assigned,
             "renderer qualification/semantic mapping lineage changed")
        check_camera_stability(capture["actual_camera_matrices_before"],
                               capture["actual_camera_matrices_after"])
        check_assignment(capture["semantic_assignment_readback"], assigned,
                         require_source_recheck=False)
        if index == 0:
            refresh = capture["reset_metric_refresh_audit"]
            need(refresh is not None and refresh["state_unchanged_verified"] is True and
                 refresh["untouched_measures_unchanged_verified"] is True and
                 refresh["elapsed_steps"] == 0,
                 "frame-zero reset metric refresh evidence differs")
            check_assignment(refresh["assignment_report"], assigned,
                             require_source_recheck=True)
        else:
            need(capture["reset_metric_refresh_audit"] is None,
                 "reset refresh evidence appears after frame zero")
        before = np.asarray(capture["camera_before_render"], dtype=np.float64)
        after = np.asarray(capture["camera_after_render"], dtype=np.float64)
        need(before.shape == after.shape == (4, 4) and
             np.allclose(before, after, rtol=0, atol=1e-6),
             "render camera transform changed")
        before_time = float(capture["world_time_before_render_s"])
        world_time = float(capture["world_time_after_render_s"])
        need(math.isfinite(before_time) and before_time == world_time,
             "render world time changed")
        world_times.append(world_time)
        visible += int(row["target"]["visible"])
        valid_bbox += int(row["target"]["bbox_label_valid"])
        target_pixels += int(row["target"]["mask_area_pixels"])
        distractor_visible += sum(int(value["visible"]) for value in row["distractors"])
    need(all(right > left for left, right in zip(world_times, world_times[1:])),
         "case world time is not strictly increasing")
    first = labels[0]
    actual_initialization = first["source_audit"]["capture_evidence"]["initialization"]
    need(first["target"]["bbox_label_valid"] is True and
         actual_initialization["environment_step"] == 0 and
         actual_initialization["used_frames"] == [0] and
         actual_initialization["bbox_xyxy"] == first["target"]["bbox_xyxy_inclusive"] and
         actual_initialization["bbox_xyxy_norm"] == first["target"]["bbox_xyxy_norm"],
         "corrected frame-zero initialization changed")
    return dict(
        case_id=case_id,
        status="pass",
        observations=len(labels),
        actions=len(labels) - 1,
        target_visible_observations=visible,
        target_valid_bbox_observations=valid_bbox,
        target_visibility_fraction=visible / len(labels),
        target_mask_pixels=target_pixels,
        distractor_visible_instances=distractor_visible,
        world_window_s=world_times[-1] - world_times[0],
        case_artifacts=reference(case_manifest_path),
        labels=reference(directory / "labels.jsonl"),
        corrected_instance_camera_bbox_visibility_verified=True,
    )


def main() -> int:
    import os
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU audit must hide CUDA")
    completion = load(ROOT / "batch_completion.json")
    completion_checks = {
        "status": completion["status"] == "all_val_labels_pending_independent_admission",
        "plan_sha256": completion["plan_sha256"] == PLAN_INTERNAL_SHA256,
        "batch_status_sha256": completion["batch_status_sha256"] == BATCH_STATUS_SHA256,
        "artifact_manifest_sha256": completion["artifact_manifest_sha256"] == BATCH_ARTIFACTS_SHA256,
        "case_count": completion["case_denominator"] == completion["successful_case_count"] == 4,
        "observation_count": completion["expected_observation_count"] == completion["observations_saved"] == 229,
        "action_count": completion["expected_action_count"] == 225,
        "worker_reaped": completion["worker_unreaped"] is False,
        "not_pre_admitted": completion["independent_admission"] is False,
    }
    need(all(completion_checks.values()), "dev4 completion boundary changed: " +
         ",".join(key for key, value in completion_checks.items() if not value))
    need(sha(ROOT / "batch_status.json") == BATCH_STATUS_SHA256 and
         sha(ROOT / "batch_artifacts.json") == BATCH_ARTIFACTS_SHA256,
         "dev4 top-level seal changed")
    batch = load(ROOT / "batch_status.json")
    manifest = load(ROOT / "batch_artifacts.json")
    verified_bytes = verify_manifest(ROOT, manifest)
    actual = {path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_file()}
    need(actual == set(manifest) | {"batch_artifacts.json", "batch_completion.json"},
         "dev4 has unsealed extra or missing files")
    plan = load(ROOT / "frozen_execution_plan.json")
    need(plan["plan_sha256"] == PLAN_INTERNAL_SHA256 and
         plan == load(BUNDLE / "frozen_execution_plan.json"),
         "dev4 frozen plan changed")
    contract_report = contract.validate_plan(plan, verify_files=False)
    need(contract_report["case_denominator"] == 4 and
         contract_report["expected_observation_count"] == 229 and
         contract_report["expected_action_count"] == 225 and
         contract_report["optimizer_input_allowed"] is False,
         "dev4 source/action/partition contract changed")
    known_later_drift = {
        str(REPO / "omtrackvla/data/end_to_end_training.py"),
        str(REPO / "omtrackvla/evaluation/end_to_end_closed_loop.py"),
    }
    drifted = set()
    for artifact in plan["artifacts"]:
        path = Path(artifact["path"])
        if not path.is_file() or path.stat().st_size != artifact["bytes"] or sha(path) != artifact["sha256"]:
            drifted.add(str(path))
    need(drifted == known_later_drift,
         "dev4 frozen source drift differs from the two recorded post-collection files")
    need(batch["test_locked_read"] is False and batch["optimizer_input_allowed"] is False and
         batch["worker_unreaped"] is False and len(batch["entries"]) == 4,
         "dev4 batch safety boundary changed")

    reports = [audit_case(entry, status, manifest)
               for entry, status in zip(plan["entries"], batch["entries"])]
    need(sum(row["observations"] for row in reports) == 229 and
         sum(row["actions"] for row in reports) == 225,
         "dev4 audited denominator changed")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = OUTPUT / ("final_" + stamp)
    directory.mkdir(parents=True, exist_ok=False)
    summary = dict(
        kind="permanent_dev4_corrected_instance_independent_audit_v1",
        schema_version=1,
        status="admitted_for_fixed_dev_perception_metrics",
        permanent_partition_role="val",
        official_evt_validation=False,
        case_denominator=4,
        cases=reports,
        observations=sum(row["observations"] for row in reports),
        actions=sum(row["actions"] for row in reports),
        target_visible_observations=sum(row["target_visible_observations"] for row in reports),
        target_valid_bbox_observations=sum(row["target_valid_bbox_observations"] for row in reports),
        target_mask_pixels=sum(row["target_mask_pixels"] for row in reports),
        distractor_visible_instances=sum(row["distractor_visible_instances"] for row in reports),
        sealed_bytes_verified=verified_bytes,
        batch_completion=reference(ROOT / "batch_completion.json"),
        batch_status=reference(ROOT / "batch_status.json"),
        batch_artifacts=reference(ROOT / "batch_artifacts.json"),
        frozen_plan=reference(ROOT / "frozen_execution_plan.json"),
        validator_source=reference(Path(__file__)),
        post_collection_source_drift=sorted(drifted),
        all_other_frozen_source_references_verified=True,
        independent_admission=True,
        perception_metric_input_allowed=True,
        optimizer_input_allowed=False,
        training_input_allowed=False,
        teacher_action_quality_qualified=False,
        test_locked_used=False,
        simulator_created=False,
        gpu_started=False,
        model_loaded=False,
        interpretation=[
            "This admits the four fixed cases only for held-out perception and polar-state measurement.",
            "It does not admit dev4 pixels to optimization and is not the official EVT validation set.",
            "It does not certify the teacher actions or closed-loop navigation quality.",
        ],
    )
    path = directory / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(dict(
        summary=reference(path),
        status=summary["status"],
        cases=4,
        observations=summary["observations"],
        visible=summary["target_visible_observations"],
        valid_bbox=summary["target_valid_bbox_observations"],
        optimizer_started=False,
        test_locked_used=False,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
