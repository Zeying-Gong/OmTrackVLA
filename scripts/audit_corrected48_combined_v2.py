"""Independently audit the completed corrected fixed-48 collection.

The first 32 STT/DT members live in the original collection root.  The final
16 AT members live in the immutable continuation root.  This auditor keeps the
fixed denominator, verifies both control chains, and replays the existing raw
instance/camera/temporal/action checks without importing Habitat or a model.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import sys


REPO = Path("/data/nfs/share/wam_tracking/OmTrackVLA")
AUDITOR = REPO / ".codex_upload/corrected48_results_audit_v1/attempt_002"
ORIGINAL_ROOT = REPO / "outputs/takeover/clean48_teacher_v5_semantic_fixed_collection_v1"
AT_ROOT = REPO / "outputs/takeover/clean48_teacher_v5_semantic_fixed_at_v2"
AT_STAGE = REPO / ".codex_upload/clean48_teacher_v5_semantic_fixed_at_v2"
OUTPUT_ROOT = REPO / "outputs/takeover/corrected48_combined_audit_v2"

ORIGINAL_CONTROL_SHA256 = "fa1bb2a930fa27a54e25a9685621a6047131904ca4d1d28ef3349c1fd3466ea3"
ORIGINAL_COLLECTION_SHA256 = "17097a3bee5a39d0bdfa9e839565e19f3651f96d4ba8751cbc2043c66665a7a2"
AT_PREPARED_SHA256 = "15e519f4f6f3ac19d8d78dc771f4a30cbeaee8b0a24138960fd7cc34002e9e4a"
AT_BUNDLE_SHA256 = "4d8d2d4b3fc3a1a66f912058b07c80e6fb1d57ba25452b574608fbf8da2437dc"
AT_LEDGER_SHA256 = "bbc83ef6541186b6d6db08f2ff62487b6bc48a767df122a890a348516ca88a69"
AT_OBSERVATIONS = 761

sys.path.insert(0, str(AUDITOR))
import audit_corrected_batch as original_audit  # noqa: E402
from audit_io import load, need, reference, sha, verify_ref  # noqa: E402
from batch_audit_core import aggregate  # noqa: E402
import raw_case_core as raw  # noqa: E402


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("rb") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def verify_auditor_sources() -> list[dict]:
    manifest = load(AUDITOR / "audit_bundle_manifest.json")
    result = []
    for name, pin in manifest["files"].items():
        path = AUDITOR / name
        need(path.stat().st_size == pin["bytes"] and sha(path) == pin["sha256"],
             "auditor source changed: " + name)
        result.append(reference(path))
    return result


def verify_at_controls(entries: list[dict]) -> tuple[list[dict], list[dict]]:
    need(sha(AT_STAGE / "prepared.json") == AT_PREPARED_SHA256,
         "AT prepared record changed")
    need(sha(AT_STAGE / "bundle_manifest.json") == AT_BUNDLE_SHA256,
         "AT bundle manifest changed")
    prepared = load(AT_STAGE / "prepared.json")
    bundle = load(AT_STAGE / "bundle_manifest.json")
    need(prepared["status"] == "all_16_at_frozen_cpu_validated" and
         prepared["case_count"] == 16 and not prepared["gpu_started"] and
         not prepared["optimizer_started"], "AT prepared boundary changed")
    need(len(bundle["cases"]) == 16 and bundle["automatic_case_replacement"] is False and
         bundle["automatic_retry"] is False, "AT fixed-member policy changed")
    expected = entries[32:]
    need([row["pool_index"] for row in bundle["cases"]] == list(range(32, 48)),
         "AT pool indices changed")
    need([row["entry"] for row in bundle["cases"]] == expected,
         "AT fixed member metadata changed")
    need([row["sample_id"] for row in prepared["cases"]] ==
         [row["sample_id"] for row in expected], "AT prepared order changed")
    for prepared_case, bundle_case in zip(prepared["cases"], bundle["cases"]):
        need(prepared_case["pool_index"] == bundle_case["pool_index"] and
             prepared_case["plan"] == bundle_case["plan"],
             "AT prepared/bundle plan binding differs")
        verify_ref(prepared_case["plan"])
        verify_ref(bundle_case["source_plan"])

    completion_path = AT_ROOT / "_runner/CONTINUATION_COMPLETE.json"
    status_path = AT_ROOT / "_runner/status.json"
    ledger_path = AT_ROOT / "_runner/ledger.jsonl"
    actual_ledger_sha256 = sha(ledger_path)
    need(actual_ledger_sha256 == AT_LEDGER_SHA256,
         "AT continuation ledger changed: path=" + str(ledger_path) +
         " actual=" + repr(actual_ledger_sha256) +
         " expected=" + repr(AT_LEDGER_SHA256))
    completion = load(completion_path)
    status = load(status_path)
    for record in (completion, status):
        need(record["status"] == "completed_all_16_attempts" and
             record["planned_at_cases"] == record["cases_attempted"] ==
             record["sealed_case_count"] == 16 and
             record["observations_saved"] == AT_OBSERVATIONS and
             record["optimizer_started"] is False and
             record["test_locked_used"] is False and
             record["physical_gpu"] == 3 and
             record["source_bundle_sha256"] == AT_BUNDLE_SHA256 and
             record["source_collection_sha256"] == ORIGINAL_COLLECTION_SHA256 and
             record["ledger_sha256"] == AT_LEDGER_SHA256,
             "AT completion boundary changed")
    need(completion == status, "AT completion/status final records differ")

    ledger = jsonl(ledger_path)
    need(len(ledger) == 16 and
         [row["pool_index"] for row in ledger] == list(range(32, 48)) and
         [row["sample_id"] for row in ledger] ==
         [row["sample_id"] for row in expected], "AT ledger membership changed")
    need(all(row["task"] == "AT" and row["worker_reaped"] is True and
             row["artifact_sealed"] is True and
             row["continuation_status"] == "sealed_retained"
             for row in ledger), "AT worker sealing/reaping differs")
    need(sum(row["observations_saved"] for row in ledger) == AT_OBSERVATIONS,
         "AT observation sum differs")
    return prepared["cases"], ledger


def normalized_case(value: dict, entry: dict, index: int,
                    expected_observations: int) -> dict:
    need(value.get("timeline", {}).get("observation_count", 0) == expected_observations,
         "saved observation count differs from ledger")
    status = ("pass" if value["audit_status"] == "sealed_raw_and_instance_labels_verified"
              else "no_observation" if
              value["audit_status"] == "sealed_bytes_verified_no_observations"
              else "reject")
    value.update(
        kind="corrected48_case_instance_label_audit_v2",
        schema_version=2,
        pool_index=index,
        task=entry["task"],
        status=status,
        observations_verified=value.get("timeline", {}).get("observation_count", 0),
        supervisor_sha256=value.get("supervisor", {}).get("sha256"),
        validation_flags=dict(
            main_instance_mapping_verified=status == "pass",
            actual_camera_pair_verified=status == "pass",
            raw_mask_bbox_visibility_verified=status == "pass",
            temporal_action_records_verified=status == "pass",
        ),
        teacher_quality_qualified=False,
        action_supervision_allowed=False,
        waypoint_supervision_allowed=False,
        formal_training_eligible=False,
    )
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
         "CPU auditor must hide all CUDA devices")
    need("test_locked" not in str(REPO).lower(), "forbidden repository path")

    source_refs = verify_auditor_sources()
    original_prepared, _bundle, entries = original_audit.bind_inputs()
    original_collection, original_controls, original_ledger = original_audit.verify_controls(
        ORIGINAL_CONTROL_SHA256, ORIGINAL_COLLECTION_SHA256)
    need(len(original_ledger) == 32 and
         [row["pool_index"] for row in original_ledger] == list(range(32)),
         "original first-32 ledger changed")
    at_prepared, at_ledger = verify_at_controls(entries)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(args.output_root) / ("final_" + stamp)
    output.mkdir(parents=True, exist_ok=False)
    values = []
    report_entries = []
    for index, entry in enumerate(entries):
        if index < 32:
            prepared = original_prepared["cases"][index]
            ledger = original_ledger[index]
            expected_supervisor = original_collection["entries"][index]["supervisor"]
            case_root = ORIGINAL_ROOT / entry["sample_id"]
            source_collection = "original_corrected_stt_dt_v1"
        else:
            prepared = at_prepared[index - 32]
            ledger = at_ledger[index - 32]
            case_root = AT_ROOT / entry["sample_id"]
            expected_supervisor = reference(case_root / "supervisor_result.json")
            need(expected_supervisor["sha256"] == ledger["supervisor_sha256"],
                 "AT ledger supervisor digest differs")
            source_collection = "derived_corrected_at_v2"
        try:
            value = raw.audit_case(case_root, prepared, entry, expected_supervisor)
            value = normalized_case(value, entry, index, ledger["observations_saved"])
        except Exception as error:
            value = dict(
                kind="corrected48_case_instance_label_audit_v2",
                schema_version=2,
                pool_index=index,
                sample_id=entry["sample_id"],
                task=entry["task"],
                status="reject",
                audit_status="audit_failed_retained",
                observations_verified=0,
                supervisor_sha256=ledger.get("supervisor_sha256"),
                error_type=type(error).__name__,
                error=str(error),
                teacher_quality_qualified=False,
                action_supervision_allowed=False,
                waypoint_supervision_allowed=False,
                formal_training_eligible=False,
            )
        value["source_collection"] = source_collection
        case_path = output / (entry["sample_id"] + ".json")
        write_new(case_path, value)
        values.append(value)
        report_entries.append(dict(
            pool_index=index,
            sample_id=entry["sample_id"],
            task=entry["task"],
            source_collection=source_collection,
            status=value["status"],
            supervisor_sha256=value.get("supervisor_sha256"),
            case_report=reference(case_path),
        ))
        print(json.dumps(dict(
            reviewed=index + 1,
            sample_id=entry["sample_id"],
            status=value["status"],
            observations=value.get("observations_verified", 0),
            error=value.get("error"),
        ), sort_keys=True), flush=True)

    summary = dict(
        kind="corrected48_combined_independent_instance_label_audit_v2",
        schema_version=2,
        status="complete_all48_audit_retaining_rejections",
        planned_episode_denominator=48,
        entries=report_entries,
        original_collection_manifest_sha256=ORIGINAL_COLLECTION_SHA256,
        original_batch_control_evidence_seal_sha256=ORIGINAL_CONTROL_SHA256,
        original_control_seal=reference(ORIGINAL_ROOT / "batch_control_evidence_seal.json"),
        at_prepared_sha256=AT_PREPARED_SHA256,
        at_bundle_manifest_sha256=AT_BUNDLE_SHA256,
        at_continuation_ledger_sha256=AT_LEDGER_SHA256,
        at_completion=reference(AT_ROOT / "_runner/CONTINUATION_COMPLETE.json"),
        validator_source_references=source_refs + [reference(Path(__file__))],
        aggregate=aggregate(values, 48),
        status_counts=dict(Counter(row["status"] for row in values)),
        observation_count=sum(row.get("observations_verified", 0) for row in values),
        test_locked_used=False,
        simulator_created=False,
        gpu_started=False,
        optimizer_started=False,
        teacher_quality_qualified=False,
        action_supervision_allowed=False,
        waypoint_supervision_allowed=False,
        formal_training_eligible=False,
        interpretation=[
            "Pass certifies sealed RGB/panoptic bytes, corrected instance assignment, camera pairing, recomputed bbox/visibility, time, transition, and guard records.",
            "Failed and zero-observation cases remain in the fixed 48 denominator and are never replaced.",
            "This audit does not certify teacher action quality and does not admit teacher actions or trajectories for waypoint optimization.",
        ],
    )
    summary_path = output / "summary.json"
    write_new(summary_path, summary)
    print("AUDIT_RECEIPT=" + json.dumps(dict(
        summary=reference(summary_path),
        status_counts=summary["status_counts"],
        observations=summary["observation_count"],
        optimizer_started=False,
        test_locked_used=False,
    ), sort_keys=True), flush=True)
    return 0 if not summary["status_counts"].get("reject", 0) else 2


if __name__ == "__main__":
    raise SystemExit(main())
