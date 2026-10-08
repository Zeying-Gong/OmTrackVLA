"""Exact 125-key continuation of failed 61833; old one-key evidence stays v1.

This is an independent v2 launcher, not a resume-in-place or training release.
The unchanged v1 launcher owns process cleanup, server commands and GPU guards.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from wa.tools import failure_state_launch as old
from wa.wm.failure_state_protocol import load_plan, canonical_sha
from wa.wm.failure_state_protocol_v2 import EXPERIMENT, PROTOCOL_SHA
from wa.wm.failure_state_continuation import (
    load_continuation, BASE_SHA, KEY, SOURCE_ROOT, REMAINING_LANE_COUNTS,
)

CONTINUATION_SHA = "06d098f30ec52d988dc707249e73b17d11642f44846a75015455315891182822"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output")
    p.add_argument("--plan", required=True)
    p.add_argument("--plan-sha", required=True)
    p.add_argument("--continuation", required=True)
    p.add_argument("--continuation-sha", required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--development-key")
    mode.add_argument("--formal", action="store_true")
    p.add_argument("--port-base", type=int, default=19180)
    p.add_argument("--display-base", type=int, default=580)
    return p.parse_args(argv)


def validate_route(args, env, cuda_count=None):
    devices = old.validate_route(args, env, cuda_count)
    if args.plan_sha != BASE_SHA or args.continuation_sha != CONTINUATION_SHA:
        raise ValueError("only the frozen 61833 base and raw-preserved continuation are allowed")
    path = Path(args.continuation)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("absolute continuation path required")
    return devices


def validate_workset(plan, overlay, args):
    if overlay["base_plan"] != dict(path=args.plan, sha256=args.plan_sha):
        raise ValueError("continuation differs from explicitly pinned base plan")
    if (overlay["expected_count"], overlay["new_count"], overlay["reused_count"]) != (126, 125, 1):
        raise ValueError("only one frozen plus125 new keys are permitted")
    if overlay["reused_keys"] != [KEY] or overlay["training_released"] is not False:
        raise ValueError("wrong frozen row or training claim")
    wanted = [e for e in plan["entries"] if e["key"] != KEY]
    lanes = [[e for e in lane if e["key"] != KEY] for lane in plan["lanes"]]
    if (canonical_sha(overlay["remaining_entries"]) != canonical_sha(wanted)
            or canonical_sha(overlay["remaining_lanes"]) != canonical_sha(lanes)
            or list(map(len, lanes)) != REMAINING_LANE_COUNTS):
        raise ValueError("continuation changed original lane ownership/order")
    keys = [e["key"] for e in wanted]
    if len(keys) != len(set(keys)) or len(keys) != 125 or KEY in keys:
        raise ValueError("duplicate or overlapping continuation keys")
    if set(keys) | {KEY} != {e["key"] for e in plan["entries"]}:
        raise ValueError("missing or foreign original keys")


def planned_entries(overlay, args, lane):
    if args.formal:
        return overlay["remaining_lanes"][lane]
    entries = [e for e in overlay["remaining_entries"] if e["key"] == args.development_key]
    if len(entries) != 1:
        raise ValueError("developer key must be one of the125 missing keys")
    return entries


def make_commands(args, plan, lane, device, dest, inherited, ln_libraries):
    specs = old.make_commands(args, plan, lane, device, dest, inherited, ln_libraries)
    worker = specs[2]["argv"]
    i = worker.index("wa.wm.failure_state_collect")
    worker[i] = "wa.wm.failure_state_collect_v2"
    worker += ["--continuation", args.continuation, "--continuation-sha", args.continuation_sha]
    return specs


def check_completion(collection, expected_keys, args, lane):
    collection = Path(collection)
    development = not args.formal
    marker = json.loads((collection / ("DEVELOPMENT_CHECK.json" if development else "COMPLETE.json")).read_text())
    expected = dict(experiment=EXPERIMENT, entries=len(expected_keys), keys=expected_keys,
                    expected=len(expected_keys), shard=lane, development=development,
                    training_released=False, no_success_rate=True,
                    plan_sha256=args.plan_sha, base_protocol_sha256=old.PROTOCOL_SHA,
                    protocol_sha256=PROTOCOL_SHA, continuation_sha256=args.continuation_sha)
    if set(marker) != set(expected) or any(
        type(marker.get(k)) is not type(v) or marker[k] != v for k, v in expected.items()
    ):
        raise ValueError("collector completion differs from exact v2 continuation lane")
    raw = (collection / "records.jsonl").read_text()
    if not raw.endswith("\n"):
        raise ValueError("partial records JSONL")
    rows = [json.loads(line) for line in raw.splitlines() if line]
    if [row.get("key") for row in rows] != expected_keys:
        raise ValueError("records differ from planned keys/order")
    for row in rows:
        pins = dict(experiment=EXPERIMENT, plan_sha256=args.plan_sha,
                    base_protocol_sha256=old.PROTOCOL_SHA, protocol_sha256=PROTOCOL_SHA,
                    continuation_sha256=args.continuation_sha)
        if any(row.get(k) != v for k, v in pins.items()):
            raise ValueError("foreign record protocol/plan/continuation")
        if development:
            checks = dict(status="DEVELOPER_NONZERO_REPLAY_CHECK_ONLY", prefix_actions=24,
                          takeover_step=20, branch_total_actions=24, teacher_owned_actions=4,
                          no_success_rate=True, training_eligible=False)
            if any(type(row.get(k)) is not type(v) or row[k] != v for k, v in checks.items()):
                raise ValueError("wrong developer scope or nonzero interface proof")
        elif (row.get("task") != "stt" or row.get("outcome") not in {
            "rerun_student_success_no_recovery_needed", "student_invalid_initialization",
            "repeated_teacher_recovery_candidate", "no_valid_teacher_recovery"
        }):
            raise ValueError("foreign or incomplete formal record")
    return marker


def coverage(plan, overlay, args, markers):
    keys = [key for marker in markers for key in marker["keys"]]
    wanted = ([e["key"] for e in overlay["remaining_entries"]] if args.formal
              else [args.development_key])
    if (len(markers) != (8 if args.formal else 1) or len(keys) != len(wanted)
            or len(set(keys)) != len(keys) or set(keys) != set(wanted)):
        raise ValueError("incomplete125 continuation or developer1 coverage")
    result = dict(new_keys=wanted, new_count=len(wanted), new_experiment=EXPERIMENT,
                  new_protocol_sha256=PROTOCOL_SHA, base_plan_sha256=args.plan_sha,
                  continuation_sha256=args.continuation_sha, formal=args.formal,
                  training_released=False, no_success_rate=True)
    if args.formal:
        # Reference original provenance; do not rewrite its row/protocol or label it v2.
        result.update(reused_keys=list(overlay["reused_keys"]), reused_count=1,
                      reused_experiment=overlay["experiment"],
                      reused_protocol_sha256=overlay["protocol_sha256"],
                      frozen_records=overlay["frozen_records"], expected_total=126)
        if set(wanted) & set(overlay["reused_keys"]) or (
            set(wanted) | set(overlay["reused_keys"]) != {e["key"] for e in plan["entries"]}
        ):
            raise ValueError("old/new overlap or incomplete original126 coverage")
    return result


def run_lane(args, plan, overlay, lane, device, out, inherited, libs, cancelled):
    dest = out / f"lane{lane}"
    dest.mkdir()
    keys = [e["key"] for e in planned_entries(overlay, args, lane)]
    timeout = len(keys) * old.MAX_RUNS_PER_ENTRY * old.SECONDS_PER_RUN if args.formal else old.DEVELOPER_SECONDS
    specs = make_commands(args, plan, lane, device, dest, inherited, libs)
    old.write_json(dest / "launch.json", dict(experiment=EXPERIMENT, plan_sha256=args.plan_sha,
        continuation_sha256=args.continuation_sha, protocol_sha256=PROTOCOL_SHA,
        device=device, lane=lane, keys=keys, startup_timeout_seconds=old.STARTUP_SECONDS,
        worker_timeout_seconds=timeout, commands=[old.command_record(s) for s in specs],
        training_released=False, no_success_rate=True))
    children = []
    try:
        for spec in specs[:2]:
            children.append(old.OwnedProcess(spec, dest / "processes.jsonl"))
        old.wait_ready(*children, dest, cancelled)
        worker = old.OwnedProcess(specs[2], dest / "processes.jsonl")
        children.append(worker)
        deadline = time.monotonic() + timeout
        while worker.exit_code() is None:
            if cancelled.is_set():
                raise RuntimeError("another lane failed or launcher was interrupted")
            if any(p.exit_code() is not None for p in children[:2]):
                raise RuntimeError("model server exited during collection")
            if time.monotonic() >= deadline:
                raise TimeoutError("bounded collection worker timeout")
            time.sleep(1)
        worker.event("worker_return", pid=worker.process.pid, returncode=worker.exit_code())
        if worker.exit_code():
            raise RuntimeError("collector failed; preserve original logs and partial outputs")
        marker = check_completion(dest / "collection", keys, args, lane)
        old.write_json(dest / "lane_outcome.json", dict(status="PROCESS_AND_COMPLETION_VERIFIED",
                       completion=marker, training_released=False, no_success_rate=True))
        return marker
    except BaseException as error:
        cancelled.set()
        old.write_json(dest / "ERROR.json", dict(error_type=type(error).__name__, message=str(error)[:2000],
                       training_released=False, no_success_rate=True))
        raise
    finally:
        errors = []
        for child in reversed(children):
            try:
                child.stop()
            except BaseException as error:
                errors.append(dict(pid=child.process.pid, error=type(error).__name__ + ": " + str(error)))
        if errors:
            cancelled.set()
            old.write_json(dest / "cleanup_errors.json", errors)
            raise RuntimeError("owned subprocess cleanup incomplete; see cleanup_errors.json")


def main(argv=None):
    args = arguments(argv)
    inherited = dict(os.environ)
    cuda_count = None
    if args.formal:
        if not inherited.get("MD_AK_JOB_ID") or not inherited.get("MD_AK_TASK_ID"):
            raise ValueError("formal launch requires scheduler identity before CUDA import")
        import torch
        cuda_count = torch.cuda.device_count()
    devices = validate_route(args, inherited, cuda_count)
    plan = load_plan(args.plan, args.plan_sha)
    overlay = load_continuation(args.continuation, args.continuation_sha)
    validate_workset(plan, overlay, args)
    for lane in range(len(devices)):
        planned_entries(overlay, args, lane)
    protected = [old.S, old.R / "probe_env", old.R / "dependencies",
                 Path(old.CHECKPOINT_PATH).parent, SOURCE_ROOT,
                 Path(plan["sources"]["summary"]["path"]).parent,
                 Path(args.continuation)]
    protected += [Path(e["artifact_root"]) for e in plan["entries"]]
    out = old.output_path(args.output, protected)
    libs = old.check_paths(plan, inherited)
    for lane in range(len(devices)):
        old.check_address(args.port_base + lane, args.display_base + lane)
    commit = subprocess.run(["git", "-C", str(old.S), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(old.S), "status", "--porcelain"], check=True,
                           capture_output=True, text=True).stdout
    if args.formal and dirty:
        raise ValueError("formal source must be independently frozen and clean")
    out.mkdir(parents=True, exist_ok=False)
    old.write_json(out / "launch.json", dict(experiment=EXPERIMENT, source=str(old.S),
        source_commit=commit, source_git_status=dirty, plan=args.plan, plan_sha256=args.plan_sha,
        continuation=args.continuation, continuation_sha256=args.continuation_sha,
        protocol_sha256=PROTOCOL_SHA, formal=args.formal, devices=devices,
        expected_entries=125 if args.formal else 1, expected_lanes=len(devices),
        frozen_entries=1 if args.formal else 0, total_expected_entries=126 if args.formal else 1,
        port_base=args.port_base, display_base=args.display_base, started_utc=old.now(),
        no_success_rate=True, training_released=False))
    cancelled = threading.Event()
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: cancelled.set())
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(devices))
    markers = []
    try:
        futures = [pool.submit(run_lane, args, plan, overlay, i, device, out, inherited, libs, cancelled)
                   for i, device in enumerate(devices)]
        for future in concurrent.futures.as_completed(futures):
            markers.append(future.result())
        final_overlay = load_continuation(args.continuation, args.continuation_sha)
        validate_workset(plan, final_overlay, args)
        report = coverage(plan, final_overlay, args, markers)
        old.write_json(out / "coverage.json", report)
        old.write_json(out / ("COMPLETE.json" if args.formal else "DEVELOPMENT_CHECK.json"),
            dict(status="COLLECTION_PROCESSES_COMPLETE_NOT_TRAINING_RELEASE", experiment=EXPERIMENT,
                 plan_sha256=args.plan_sha, protocol_sha256=PROTOCOL_SHA,
                 continuation_sha256=args.continuation_sha, entries=report["new_count"],
                 lanes=len(markers), keys=report["new_keys"], formal=args.formal,
                 training_released=False, no_success_rate=True, finished_utc=old.now()))
        print("FAILURE_STATE_V2_LAUNCH_COMPLETE", report["new_count"], flush=True)
    except BaseException as error:
        cancelled.set()
        old.write_json(out / "ERROR.json", dict(error_type=type(error).__name__, message=str(error)[:2000],
                       training_released=False, no_success_rate=True))
        raise
    finally:
        cancelled.set()
        pool.shutdown(wait=True, cancel_futures=True)
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
