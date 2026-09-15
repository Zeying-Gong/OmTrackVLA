"""Run the 16 derived corrected EVT-train AT collection plans.

The original 48-case collector stopped after case 31 because GPU 3 became busy.
This runner preserves every byte of that stopped run, executes only the CPU-
qualified derived AT plans, and writes a new output root and ledger.  It never
starts an optimizer and it refuses to use a different physical GPU.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO = Path("/data/nfs/share/wam_tracking/OmTrackVLA")
BUNDLE = REPO / ".codex_upload/clean48_teacher_v5_semantic_fixed_at_v2"
OUTPUT = REPO / "outputs/takeover/clean48_teacher_v5_semantic_fixed_at_v2"
CONTINUATION = OUTPUT / "_runner"
PYTHON = Path("/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python")

EXPECTED_PREPARED_SHA256 = "15e519f4f6f3ac19d8d78dc771f4a30cbeaee8b0a24138960fd7cc34002e9e4a"
EXPECTED_BUNDLE_SHA256 = "4d8d2d4b3fc3a1a66f912058b07c80e6fb1d57ba25452b574608fbf8da2437dc"
EXPECTED_OLD_COLLECTION_SHA256 = "17097a3bee5a39d0bdfa9e839565e19f3651f96d4ba8751cbc2043c66665a7a2"
PROTECTED_FLUX_PID = "4011047"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_new_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _gpu_snapshot() -> dict[str, Any]:
    devices = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=30,
    )
    gpu3_uuid = next(
        line.split(",")[1].strip()
        for line in devices.splitlines()
        if line.split(",")[0].strip() == "3"
    )
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,process_name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=30,
    )
    rows = [line for line in processes.splitlines() if line.strip()]
    return {
        "gpu3_uuid": gpu3_uuid,
        "gpu3_compute_processes": [row for row in rows if gpu3_uuid in row],
        "protected_flux_present": any(
            row.split(",")[0].strip() == PROTECTED_FLUX_PID for row in rows
        ),
    }


def _existing_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    identifiers = [event["sample_id"] for event in events]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("continuation ledger contains duplicate sample IDs")
    return events


def _validate_frozen_inputs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise RuntimeError("parent CUDA_VISIBLE_DEVICES must be empty")
    if not REPO.is_dir() or Path.cwd().resolve() != REPO.resolve():
        raise RuntimeError("runner must execute from the frozen repository")
    prepared_path = BUNDLE / "prepared.json"
    manifest_path = BUNDLE / "bundle_manifest.json"
    old_output = REPO / "outputs/takeover/clean48_teacher_v5_semantic_fixed_collection_v1"
    old_collection_path = old_output / "collection_manifest.json"
    if _sha256(prepared_path) != EXPECTED_PREPARED_SHA256:
        raise RuntimeError("prepared.json changed")
    if _sha256(manifest_path) != EXPECTED_BUNDLE_SHA256:
        raise RuntimeError("bundle_manifest.json changed")
    if _sha256(old_collection_path) != EXPECTED_OLD_COLLECTION_SHA256:
        raise RuntimeError("stopped collection manifest changed")

    prepared = _load(prepared_path)
    manifest = _load(manifest_path)
    old_result = _load(old_output / "batch_result.json")
    old_collection = _load(old_collection_path)
    if prepared.get("status") != "all_16_at_frozen_cpu_validated":
        raise RuntimeError("derived 16-case AT freeze is not qualified")
    if manifest.get("status") != "all_16_at_plans_frozen":
        raise RuntimeError("derived AT bundle manifest is not qualified")
    if old_result.get("status") != "stopped_gpu3_busy" or old_result.get("cases_attempted") != 32:
        raise RuntimeError("original batch is not the expected stopped 32/48 run")
    if len(manifest.get("cases", [])) != 16 or len(old_collection.get("entries", [])) != 48:
        raise RuntimeError("derived 16-case or original 48-case denominator changed")

    cases = manifest["cases"]
    old_entries = old_collection["entries"][32:]
    if len(cases) != 16 or any(case["entry"].get("task") != "AT" for case in cases):
        raise RuntimeError("indices 32..47 are not the frozen AT slice")
    if any(entry.get("collection_status") != "not_collected" for entry in old_entries):
        raise RuntimeError("an original AT entry was already marked collected")
    for case in cases:
        bundle = Path(case["bundle"])
        plan = bundle / "frozen_plan.json"
        if _sha256(plan) != next(
            row["plan"]["sha256"]
            for row in prepared["cases"]
            if row["sample_id"] == case["entry"]["sample_id"]
        ):
            raise RuntimeError("frozen AT plan changed: " + case["entry"]["sample_id"])
    return manifest, cases


def main() -> int:
    manifest, cases = _validate_frozen_inputs()
    CONTINUATION.mkdir(parents=True, exist_ok=True)
    ledger_path = CONTINUATION / "ledger.jsonl"
    lock_path = CONTINUATION / "continuation.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result_path = CONTINUATION / "CONTINUATION_COMPLETE.json"
        if result_path.is_file():
            result = _load(result_path)
            if result.get("status") != "completed_all_16_attempts":
                raise RuntimeError("existing continuation result is not complete")
            print(json.dumps({"status": "already_complete", "attempted": 16}), flush=True)
            return 0

        events = _existing_events(ledger_path)
        completed = {event["sample_id"] for event in events}
        expected_order = [case["entry"]["sample_id"] for case in cases]
        if [event["sample_id"] for event in events] != expected_order[: len(events)]:
            raise RuntimeError("continuation ledger is not a prefix of the frozen AT order")

        status = {
            "schema_version": 1,
            "stage": "clean48_semantic_fixed_at_continuation_v1",
            "status": "running",
            "physical_gpu": 3,
            "optimizer_started": False,
            "source_bundle_sha256": EXPECTED_BUNDLE_SHA256,
            "source_collection_sha256": EXPECTED_OLD_COLLECTION_SHA256,
            "planned_at_cases": 16,
            "cases_attempted": len(events),
            "observations_saved": sum(event.get("observations_saved", 0) for event in events),
            "start_or_resume_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _atomic_json(CONTINUATION / "status.json", status)

        for case in cases:
            sample_id = case["entry"]["sample_id"]
            episode = Path(case["output_dir"])
            supervisor_file = episode / "supervisor_result.json"
            if sample_id in completed:
                if not supervisor_file.is_file():
                    raise RuntimeError("completed ledger entry lost its supervisor result: " + sample_id)
                continue
            if episode.exists():
                raise RuntimeError("unledgered AT output already exists: " + str(episode))

            before = _gpu_snapshot()
            if before["gpu3_compute_processes"]:
                status.update(status="stopped_gpu3_busy", gpu_snapshot=before)
                _atomic_json(CONTINUATION / "status.json", status)
                print(json.dumps(status, sort_keys=True), flush=True)
                return 3
            if not before["protected_flux_present"]:
                status.update(status="stopped_protected_flux_missing", gpu_snapshot=before)
                _atomic_json(CONTINUATION / "status.json", status)
                print(json.dumps(status, sort_keys=True), flush=True)
                return 4

            status["active_sample_id"] = sample_id
            _atomic_json(CONTINUATION / "status.json", status)
            started = time.monotonic()
            child = subprocess.run(
                [
                    str(PYTHON),
                    "-B",
                    "-u",
                    str(Path(case["bundle"]) / "collect_teacher_pilot.py"),
                    "--plan",
                    str(Path(case["bundle"]) / "frozen_plan.json"),
                    "--execute",
                ],
                cwd=str(REPO),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log_path = CONTINUATION / (sample_id + ".supervisor.log")
            with log_path.open("x", encoding="utf-8") as stream:
                stream.write(child.stdout)

            worker_status = None
            observations = 0
            worker_reaped = False
            artifact_sealed = False
            supervisor_sha256 = None
            if supervisor_file.is_file():
                supervisor = _load(supervisor_file)
                worker = supervisor.get("worker_result") or {}
                worker_status = worker.get("status")
                observations = int(worker.get("observations_saved", 0))
                worker_reaped = (
                    supervisor.get("worker_reaped") is True
                    and supervisor.get("worker_unreaped") is False
                )
                artifact_sealed = isinstance(supervisor.get("artifact_manifest"), dict)
                supervisor_sha256 = _sha256(supervisor_file)
            event = {
                "sample_id": sample_id,
                "pool_index": case["pool_index"],
                "task": "AT",
                "exit_code": child.returncode,
                "worker_status": worker_status,
                "observations_saved": observations,
                "worker_reaped": worker_reaped,
                "artifact_sealed": artifact_sealed,
                "supervisor_sha256": supervisor_sha256,
                "elapsed_wall_s": time.monotonic() - started,
            }
            if not worker_reaped or not artifact_sealed:
                event["continuation_status"] = "failed_unsealed"
            else:
                event["continuation_status"] = "sealed_retained"
            with ledger_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            events.append(event)
            status.update(
                cases_attempted=len(events),
                observations_saved=sum(row.get("observations_saved", 0) for row in events),
            )
            _atomic_json(CONTINUATION / "status.json", status)
            print(json.dumps(event, sort_keys=True), flush=True)
            if not worker_reaped:
                status.update(status="stopped_unreaped_worker")
                _atomic_json(CONTINUATION / "status.json", status)
                return 5

        status.pop("active_sample_id", None)
        status.update(
            status="completed_all_16_attempts",
            end_time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            cases_attempted=len(events),
            sealed_case_count=sum(row.get("artifact_sealed") is True for row in events),
            observations_saved=sum(row.get("observations_saved", 0) for row in events),
            ledger_sha256=_sha256(ledger_path),
            optimizer_started=False,
            test_locked_used=False,
        )
        _atomic_json(CONTINUATION / "status.json", status)
        _write_new_json(result_path, status)
        print(json.dumps(status, sort_keys=True), flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
