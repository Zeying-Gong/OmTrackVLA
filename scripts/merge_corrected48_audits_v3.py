"""Merge sealed first-32 and freshly audited AT-16 reports into fixed-48 v3.

This intentionally does not replay historical source references against the
current worktree.  The first-32 reports were completed while their frozen
source closure still matched and are consumed only after their own byte/hash
references and collection bindings are rechecked.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


REPO = Path("/data/nfs/share/wam_tracking/OmTrackVLA")
AUDITOR = REPO / ".codex_upload/corrected48_results_audit_v1/attempt_002"
ORIGINAL_AUDIT_ROOT = REPO / "outputs/takeover/corrected48_results_audit_v1/final_20260914T114102101426Z"
AT_AUDIT_ROOT = REPO / "outputs/takeover/corrected48_combined_audit_v2/final_20260915T084401254718Z"
OUTPUT_ROOT = REPO / "outputs/takeover/corrected48_combined_audit_v3"

ORIGINAL_SUMMARY_SHA256 = "4f7d60ea6a72acdce5cda085e515f4031dfa25d5ea21deb779051142f9966d88"
AT_SUMMARY_SHA256 = "852571d551c1cd73e4d8a6a28864c1c9e6dd360ff718ddd861273649c0015221"
ORIGINAL_CONTROL_SHA256 = "fa1bb2a930fa27a54e25a9685621a6047131904ca4d1d28ef3349c1fd3466ea3"
ORIGINAL_COLLECTION_SHA256 = "17097a3bee5a39d0bdfa9e839565e19f3651f96d4ba8751cbc2043c66665a7a2"
AT_PREPARED_SHA256 = "15e519f4f6f3ac19d8d78dc771f4a30cbeaee8b0a24138960fd7cc34002e9e4a"
AT_BUNDLE_SHA256 = "4d8d2d4b3fc3a1a66f912058b07c80e6fb1d57ba25452b574608fbf8da2437dc"
AT_LEDGER_SHA256 = "bbc83ef6541186b6d6db08f2ff62487b6bc48a767df122a890a348516ca88a69"

sys.path.insert(0, str(AUDITOR))
import audit_corrected_batch as original_audit  # noqa: E402
from audit_io import load, need, reference, sha, verify_ref  # noqa: E402
from batch_audit_core import aggregate  # noqa: E402


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def checked_report(row: dict, expected_entry: dict, expected_observations: int,
                   expected_supervisor: dict | None, allowed_kinds: set[str]) -> dict:
    verify_ref(row["case_report"])
    report = load(row["case_report"]["path"])
    need(report["kind"] in allowed_kinds and report["schema_version"] in (1, 2),
         "case audit schema changed")
    need(row["sample_id"] == expected_entry["sample_id"] == report["sample_id"] and
         row["task"] == expected_entry["task"] == report["task"],
         "case audit identity changed")
    need(row["status"] == report["status"] and row["status"] in ("pass", "no_observation"),
         "case audit is not an admitted sealed disposition")
    need(report["observations_verified"] == expected_observations,
         "case audit observation count differs from sealed ledger")
    expected_sha = expected_supervisor["sha256"] if expected_supervisor is not None else None
    need(row["supervisor_sha256"] == report["supervisor_sha256"] == expected_sha,
         "case audit supervisor binding changed")
    need(report["teacher_quality_qualified"] is False and
         report.get("formal_training_eligible") is False,
         "case audit overclaims training quality")
    if row["status"] == "pass":
        flags = report["validation_flags"]
        need(flags["main_instance_mapping_verified"] is True and
             flags["actual_camera_pair_verified"] is True and
             flags["raw_mask_bbox_visibility_verified"] is True and
             expected_observations > 0,
             "case pass lacks corrected-instance/camera/raw-label evidence")
    else:
        need(expected_observations == 0, "no-observation case has saved observations")
    return report


def main() -> int:
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
         "CPU merger must hide all CUDA devices")
    need(sha(ORIGINAL_AUDIT_ROOT / "summary.json") == ORIGINAL_SUMMARY_SHA256,
         "sealed original audit summary changed")
    need(sha(AT_AUDIT_ROOT / "summary.json") == AT_SUMMARY_SHA256,
         "sealed AT audit summary changed")
    old_summary = load(ORIGINAL_AUDIT_ROOT / "summary.json")
    at_summary = load(AT_AUDIT_ROOT / "summary.json")
    need(old_summary["kind"] == "corrected48_independent_instance_label_audit_v1" and
         old_summary["mode"] == "final" and len(old_summary["entries"]) == 48,
         "original independent audit boundary changed")
    need(at_summary["kind"] == "corrected48_combined_independent_instance_label_audit_v2" and
         len(at_summary["entries"]) == 48 and at_summary["test_locked_used"] is False and
         at_summary["optimizer_started"] is False and
         at_summary["at_prepared_sha256"] == AT_PREPARED_SHA256 and
         at_summary["at_bundle_manifest_sha256"] == AT_BUNDLE_SHA256 and
         at_summary["at_continuation_ledger_sha256"] == AT_LEDGER_SHA256,
         "AT independent audit boundary changed")
    for ref in old_summary["validator_source_references"]:
        verify_ref(ref)

    _prepared, _bundle, entries = original_audit.bind_inputs()
    original_collection, _controls, original_ledger = original_audit.verify_controls(
        ORIGINAL_CONTROL_SHA256, ORIGINAL_COLLECTION_SHA256)
    need(len(original_ledger) == 32, "original attempted denominator changed")
    at_ledger_path = REPO / "outputs/takeover/clean48_teacher_v5_semantic_fixed_at_v2/_runner/ledger.jsonl"
    need(sha(at_ledger_path) == AT_LEDGER_SHA256, "AT ledger changed after audit")
    at_ledger = [json.loads(line) for line in at_ledger_path.read_text().splitlines() if line]
    need(len(at_ledger) == 16, "AT attempted denominator changed")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = OUTPUT_ROOT / ("final_" + stamp)
    output.mkdir(parents=True, exist_ok=False)
    values = []
    result_entries = []
    for index, entry in enumerate(entries):
        if index < 32:
            source_row = old_summary["entries"][index]
            ledger = original_ledger[index]
            supervisor = original_collection["entries"][index]["supervisor"]
            provenance = "sealed_original_independent_audit_v1"
            kinds = {"corrected48_case_instance_label_audit_v1"}
        else:
            source_row = at_summary["entries"][index]
            ledger = at_ledger[index - 32]
            case_root = REPO / "outputs/takeover/clean48_teacher_v5_semantic_fixed_at_v2" / entry["sample_id"]
            supervisor = reference(case_root / "supervisor_result.json")
            need(supervisor["sha256"] == ledger["supervisor_sha256"],
                 "AT supervisor changed after audit")
            provenance = "fresh_at_independent_audit_v2"
            kinds = {"corrected48_case_instance_label_audit_v2"}
        report = checked_report(source_row, entry, ledger["observations_saved"], supervisor, kinds)
        source_ref = source_row["case_report"]
        merged = dict(report)
        merged.update(
            kind="corrected48_case_instance_label_audit_v3",
            schema_version=3,
            pool_index=index,
            source_audit_provenance=provenance,
            source_case_report=source_ref,
            teacher_quality_qualified=False,
            action_supervision_allowed=False,
            waypoint_supervision_allowed=False,
            formal_training_eligible=False,
        )
        path = output / (entry["sample_id"] + ".json")
        write_new(path, merged)
        values.append(merged)
        result_entries.append(dict(
            pool_index=index,
            sample_id=entry["sample_id"],
            task=entry["task"],
            status=merged["status"],
            observations_verified=merged["observations_verified"],
            source_audit_provenance=provenance,
            supervisor_sha256=merged["supervisor_sha256"],
            case_report=reference(path),
        ))

    counts = dict(Counter(row["status"] for row in values))
    need(not counts.get("reject", 0) and counts.get("pass", 0) +
         counts.get("no_observation", 0) == 48,
         "combined fixed48 has rejected/unaccounted members")
    summary = dict(
        kind="corrected48_combined_independent_instance_label_audit_v3",
        schema_version=3,
        status="complete_fixed48_all_members_accounted",
        planned_episode_denominator=48,
        entries=result_entries,
        status_counts=counts,
        observation_count=sum(row["observations_verified"] for row in values),
        aggregate=aggregate(values, 48),
        original_sealed_audit_summary=reference(ORIGINAL_AUDIT_ROOT / "summary.json"),
        fresh_at_audit_summary=reference(AT_AUDIT_ROOT / "summary.json"),
        validator_source_references=[reference(Path(__file__))],
        historical_source_drift_handling=(
            "First-32 reports are consumed from their already sealed audit. "
            "They are not replayed against the later modified worktree."
        ),
        test_locked_used=False,
        simulator_created=False,
        gpu_started=False,
        optimizer_started=False,
        corrected_perception_label_candidate=True,
        candidate_observation_count=sum(row["observations_verified"] for row in values
                                        if row["status"] == "pass"),
        teacher_quality_qualified=False,
        action_supervision_allowed=False,
        waypoint_supervision_allowed=False,
        formal_training_eligible=False,
        interpretation=[
            "All 48 fixed members remain represented: valid failed prefixes pass, while six zero-observation failures remain explicit.",
            "Corrected mask, bbox, visibility, instance and camera labels are candidates for a perception-only smoke after a separate admission record.",
            "Teacher actions and future trajectories remain disallowed because dynamic teacher quality has not passed its own gate.",
        ],
    )
    summary_path = output / "summary.json"
    write_new(summary_path, summary)
    print(json.dumps(dict(
        summary=reference(summary_path),
        status=summary["status"],
        status_counts=counts,
        observations=summary["observation_count"],
        candidate_observations=summary["candidate_observation_count"],
        optimizer_started=False,
        test_locked_used=False,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
