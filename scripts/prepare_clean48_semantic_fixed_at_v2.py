"""Derive append-only AT collection plans after two unrelated repo files changed.

The original semantic-fixed 48-case plan stopped after STT and DT.  This CPU
preparation step creates a new 16-case AT bundle and output root.  It updates
only the hashes/paths needed for the derived bundle, and proves that every
symbol from ``end_to_end_closed_loop`` used by the teacher backend is byte-for-
byte unchanged.  No simulator, GPU, model, or optimizer is started here.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


REPO = Path("/data/nfs/share/wam_tracking/OmTrackVLA")
SOURCE = REPO / ".codex_upload/clean48_teacher_v5_semantic_fixed_collection_v1"
DERIVED = REPO / ".codex_upload/clean48_teacher_v5_semantic_fixed_at_v2"
OUTPUT = REPO / "outputs/takeover/clean48_teacher_v5_semantic_fixed_at_v2"
OLD_RUNTIME = REPO / ".codex_upload/omtrackvla/evaluation/end_to_end_closed_loop.py"
CURRENT_RUNTIME = REPO / "omtrackvla/evaluation/end_to_end_closed_loop.py"
CURRENT_TRAINING_DATA = REPO / "omtrackvla/data/end_to_end_training.py"
PYTHON = Path("/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python")

SOURCE_PREPARED_SHA256 = "809501641fee9206ecf7d2fc7b4703f5e29c90d937a033be4978a61a0ad83455"
SOURCE_BUNDLE_SHA256 = "6feb88bd00271b83821c12438e181e9c53197e06d2c9def5070693ee33d89d7d"
SOURCE_COLLECTION_SHA256 = "17097a3bee5a39d0bdfa9e839565e19f3651f96d4ba8751cbc2043c66665a7a2"
OLD_RUNTIME_SHA256 = "dc99cf00e719f4613ce0792c7247d6d0bcc1677a7fb6b3cb7296705df9f8c8d5"
CURRENT_RUNTIME_SHA256 = "0dcc0ca0a4967082709dc44741d7e6ac9dac981d52fa736a20ff3abace0a7bfe"
CURRENT_TRAINING_DATA_SHA256 = "86be5ac90df6562dd692bd7b87b46be4e5a1176b19dc292e75c123f9740a62b6"
RUNTIME_SYMBOLS_USED_BY_TEACHER = (
    "RGB_KEY",
    "PANOPTIC_KEY",
    "ACTION_NAMES",
    "configure",
    "_camera_transform",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def reference(path: Path) -> dict[str, Any]:
    if "test_locked" in str(path.resolve()).lower():
        raise RuntimeError("refusing locked-test reference")
    return {
        "path": str(path),
        "resolved_path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def symbol_sources(path: Path) -> dict[str, str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    result: dict[str, str] = {}
    for node in tree.body:
        name = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            identifiers = [target.id for target in targets if isinstance(target, ast.Name)]
            if len(identifiers) == 1:
                name = identifiers[0]
        if name in RUNTIME_SYMBOLS_USED_BY_TEACHER:
            segment = ast.get_source_segment(source, node)
            if segment is None:
                raise RuntimeError("cannot recover source for runtime symbol: " + str(name))
            result[str(name)] = segment
    if set(result) != set(RUNTIME_SYMBOLS_USED_BY_TEACHER):
        raise RuntimeError("runtime symbol audit is incomplete")
    return result


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError("derived source replacement count is not one: " + old)
    return source.replace(old, new)


def update_closure(
    closure: dict[str, Any], original_bundle: Path, derived_bundle: Path
) -> dict[str, Any]:
    updated = dict(closure)
    artifacts = []
    original_prefix = str(original_bundle) + "/"
    for old in closure["artifacts"]:
        old_path = str(old["path"])
        if old_path.startswith(original_prefix):
            artifacts.append(reference(derived_bundle / Path(old_path).name))
        elif old_path == str(CURRENT_RUNTIME):
            artifacts.append(reference(CURRENT_RUNTIME))
        elif old_path == str(CURRENT_TRAINING_DATA):
            artifacts.append(reference(CURRENT_TRAINING_DATA))
        else:
            artifacts.append(old)
    updated["artifacts"] = artifacts
    return updated


def main() -> None:
    if Path.cwd().resolve() != REPO.resolve():
        raise RuntimeError("run from repository root")
    if DERIVED.exists() or OUTPUT.exists():
        raise RuntimeError("derived bundle or output already exists")
    if sha256(SOURCE / "prepared.json") != SOURCE_PREPARED_SHA256:
        raise RuntimeError("source prepared manifest changed")
    if sha256(SOURCE / "bundle_manifest.json") != SOURCE_BUNDLE_SHA256:
        raise RuntimeError("source bundle manifest changed")
    if sha256(REPO / "outputs/takeover/clean48_teacher_v5_semantic_fixed_collection_v1/collection_manifest.json") != SOURCE_COLLECTION_SHA256:
        raise RuntimeError("source stopped collection manifest changed")
    if sha256(OLD_RUNTIME) != OLD_RUNTIME_SHA256:
        raise RuntimeError("frozen runtime backup changed")
    if sha256(CURRENT_RUNTIME) != CURRENT_RUNTIME_SHA256:
        raise RuntimeError("current runtime changed after audit")
    if sha256(CURRENT_TRAINING_DATA) != CURRENT_TRAINING_DATA_SHA256:
        raise RuntimeError("current training-data module changed after audit")

    old_symbols = symbol_sources(OLD_RUNTIME)
    current_symbols = symbol_sources(CURRENT_RUNTIME)
    changed_symbols = [
        name
        for name in RUNTIME_SYMBOLS_USED_BY_TEACHER
        if old_symbols[name] != current_symbols[name]
    ]
    if changed_symbols:
        raise RuntimeError("teacher-used runtime symbols changed: " + str(changed_symbols))

    source_manifest = load(SOURCE / "bundle_manifest.json")
    source_prepared = load(SOURCE / "prepared.json")
    source_collection = load(
        REPO / "outputs/takeover/clean48_teacher_v5_semantic_fixed_collection_v1/collection_manifest.json"
    )
    cases = source_manifest["cases"][32:]
    if len(cases) != 16 or any(case["entry"].get("task") != "AT" for case in cases):
        raise RuntimeError("source AT slice changed")
    if any(
        entry.get("collection_status") != "not_collected"
        for entry in source_collection["entries"][32:]
    ):
        raise RuntimeError("source manifest no longer marks every AT case uncollected")

    DERIVED.mkdir(parents=True, exist_ok=False)
    derived_cases = []
    for source_case in cases:
        sample_id = source_case["entry"]["sample_id"]
        original_bundle = Path(source_case["bundle"])
        derived_bundle = DERIVED / "cases" / sample_id
        derived_output = OUTPUT / sample_id
        derived_bundle.mkdir(parents=True, exist_ok=False)

        for path in sorted(original_bundle.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            if path.name == "pilot_contract.py":
                text = replace_once(text, str(original_bundle), str(derived_bundle))
                text = replace_once(text, str(source_case["output_dir"]), str(derived_output))
                text = replace_once(text, OLD_RUNTIME_SHA256, CURRENT_RUNTIME_SHA256)
            ast.parse(text)
            write_new(derived_bundle / path.name, text.encode("utf-8"))

        closure = update_closure(
            load(original_bundle / "asset_closure.json"),
            original_bundle,
            derived_bundle,
        )
        closure_path = derived_bundle / "asset_closure.json"
        write_new(closure_path, json_bytes(closure))
        plan_path = derived_bundle / "frozen_plan.json"
        completed = subprocess.run(
            [
                str(PYTHON),
                "-B",
                str(derived_bundle / "freeze_pilot_plan.py"),
                "--closure",
                str(closure_path),
                "--output",
                str(plan_path),
            ],
            cwd=str(REPO),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        write_new(derived_bundle / "freeze.log", completed.stdout.encode("utf-8"))
        if completed.returncode != 0:
            raise RuntimeError(sample_id + " derived freeze failed:\n" + completed.stdout[-4000:])
        plan = load(plan_path)
        derived_cases.append(
            {
                "pool_index": source_case["pool_index"],
                "entry": source_case["entry"],
                "bundle": str(derived_bundle),
                "output_dir": str(derived_output),
                "source_plan": reference(original_bundle / "frozen_plan.json"),
                "plan": reference(plan_path),
                "plan_sha256": plan["plan_sha256"],
            }
        )
        print(json.dumps({"prepared": len(derived_cases), "sample_id": sample_id}), flush=True)

    derivation = {
        "schema_version": 1,
        "stage": "clean48_teacher_v5_semantic_fixed_at_v2_derivation",
        "status": "qualified_cpu_derivation",
        "source_prepared_sha256": SOURCE_PREPARED_SHA256,
        "source_bundle_sha256": SOURCE_BUNDLE_SHA256,
        "source_collection_sha256": SOURCE_COLLECTION_SHA256,
        "source_collection_status": "stopped_gpu3_busy_after_32_of_48",
        "source_at_cases_were_uncollected": True,
        "derived_case_count": 16,
        "derived_task": "AT",
        "runtime_diff": {
            "old_sha256": OLD_RUNTIME_SHA256,
            "current_sha256": CURRENT_RUNTIME_SHA256,
            "teacher_used_symbols": list(RUNTIME_SYMBOLS_USED_BY_TEACHER),
            "teacher_used_symbols_byte_identical": True,
            "changed_teacher_used_symbols": changed_symbols,
        },
        "training_data_module": {
            "current_sha256": CURRENT_TRAINING_DATA_SHA256,
            "imported_by_teacher_bundle": False,
        },
        "semantic_teacher_and_habitat_sources_changed": False,
        "optimizer_started": False,
        "gpu_started": False,
        "test_locked_used": False,
    }
    write_new(DERIVED / "DERIVATION.json", json_bytes(derivation))
    manifest = {
        "schema_version": 1,
        "stage": "clean48_teacher_v5_semantic_fixed_at_v2",
        "status": "all_16_at_plans_frozen",
        "output": str(OUTPUT),
        "physical_gpu": 3,
        "parallel_collectors": 1,
        "automatic_retry": False,
        "automatic_case_replacement": False,
        "optimizer_started": False,
        "test_locked_used": False,
        "derivation": reference(DERIVED / "DERIVATION.json"),
        "cases": derived_cases,
    }
    write_new(DERIVED / "bundle_manifest.json", json_bytes(manifest))
    prepared = {
        "schema_version": 1,
        "status": "all_16_at_frozen_cpu_validated",
        "case_count": 16,
        "task": "AT",
        "bundle_manifest": reference(DERIVED / "bundle_manifest.json"),
        "derivation": reference(DERIVED / "DERIVATION.json"),
        "cases": [
            {
                "sample_id": case["entry"]["sample_id"],
                "pool_index": case["pool_index"],
                "plan": case["plan"],
            }
            for case in derived_cases
        ],
        "optimizer_started": False,
        "gpu_started": False,
        "test_locked_used": False,
    }
    write_new(DERIVED / "prepared.json", json_bytes(prepared))
    print(
        json.dumps(
            {
                "status": prepared["status"],
                "case_count": 16,
                "manifest_sha256": sha256(DERIVED / "bundle_manifest.json"),
                "prepared_sha256": sha256(DERIVED / "prepared.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
