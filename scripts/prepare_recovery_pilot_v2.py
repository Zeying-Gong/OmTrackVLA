"""Audit an explicit completed collection, then freeze a scene-isolated pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omtrackvla.data.recovery_sequence import validate_recovery_sample
from scripts.build_recovery_scene_manifest import freeze_scene_plan, build_manifest, verify_manifest


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-result", type=Path, required=True)
    parser.add_argument("--collection-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result_path = args.collection_result.resolve(strict=True)
    plan_path = args.collection_plan.resolve(strict=True)
    plan, result = read(plan_path), read(result_path)
    if result["status"] != "candidate_collection_complete_pending_admission":
        raise ValueError("collection must account for every planned candidate")
    if plan["test_locked_used"] is not False or result["test_locked_used"] is not False:
        raise ValueError("locked test boundary failed")
    expected = {entry["candidate_id"]: entry for entry in plan["entries"]}
    actual = {entry["candidate_id"]: entry for entry in result["results"]}
    if (len(expected) != len(plan["entries"]) or len(actual) != len(result["results"])
            or set(expected) != set(actual)):
        raise ValueError("missing, duplicate, or unexpected candidate accounting")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    audits, paths, errors = [], [], []
    validator = lambda path: validate_recovery_sample(path, artifact_root=ROOT)
    for identity, item in sorted(actual.items()):
        try:
            source = expected[identity]
            if item["status"] not in {"candidate_collected_pending_admission", "reused_verified_existing"}:
                raise ValueError("candidate collection failed")
            wanted = (source["sample_path"] if source["action"] == "collect_candidate"
                      else source["expected_existing_sample"]["path"])
            path = Path(item["sample_path"]).resolve(strict=True)
            if path != Path(wanted).resolve(strict=True) or file_hash(path) != item["sample_sha256"]:
                raise ValueError("candidate result differs from the frozen collection")
            evidence = validator(path)
            for key in ("task", "dataset_index", "episode_id", "scene_id", "anchor_environment_step"):
                if str(evidence[key]).lower() != str(source[key]).lower():
                    raise ValueError(f"source identity mismatch: {key}")
            audits.append({"candidate_id": identity, "status": "passed", "evidence": evidence})
            paths.append(path)
        except Exception as error:
            failure = {"candidate_id": identity, "status": "failed", "error": f"{type(error).__name__}: {error}"}
            audits.append(failure)
            errors.append(failure)
    write_new(output / "admission_audit.json", {
        "stage": "independent_recovery_candidate_integrity_and_quality_audit",
        "status": "failed" if errors else "passed", "formal_training_eligible": False,
        "test_locked_used": False, "collection_result_sha256": file_hash(result_path),
        "collection_plan_sha256": file_hash(plan_path), "candidates": audits,
    })
    if errors:
        print(json.dumps({"status": "failed", "errors": errors, "output": str(output)}), flush=True)
        return 1
    parents = {entry["source_checkpoint"]["path"] for entry in plan["entries"]}
    if len(parents) != 1:
        raise ValueError("pilot requires one frozen parent checkpoint")
    parent = Path(parents.pop()).resolve(strict=True)
    if file_hash(parent) != plan["source_checkpoint_sha256"]:
        raise ValueError("parent checkpoint changed")
    scene_plan = output / "scene_plan.json"
    freeze_scene_plan(sample_paths=paths, output=scene_plan, parent_checkpoint=parent,
                      collection_protocol=plan_path, validator=validator, seed=42,
                      val_fraction=1.0/3.0, min_train_scenes=2, min_val_scenes=2)
    manifest_path = output / "manifest.json"
    build_manifest(plan_path=scene_plan, output=manifest_path, validator=validator)
    manifest, train = verify_manifest(manifest_path, validator=validator, role="train", for_training=True)
    _, val = verify_manifest(manifest_path, validator=validator, role="val")
    summary = {"status": "admitted_for_development_pilot", "formal_training_eligible": False,
               "test_locked_used": False, "candidate_count": len(paths),
               "train_samples": len(train), "val_samples": len(val),
               "manifest": str(manifest_path), "manifest_sha256": file_hash(manifest_path),
               "parent_checkpoint_sha256": file_hash(parent),
               "limitation": "Early visited-state anchors verify the pipeline; this is not recovery or product acceptance."}
    write_new(output / "PREPARATION_RESULT.json", summary)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
