"""Execute only 90 still-missing keys after failed 61836.

Reused v1/v2 records are never relabelled or rewritten. Selection and teacher
error evidence remain v2; this v3 execution explicitly opts into the reviewed
missing-final-RVQ-layer boundary, separately pinned in records and metadata.
"""
import argparse
import json
import os
from pathlib import Path

from wa.wm.failure_state_collect import Runtime, write_json
from wa.wm.failure_state_protocol import load_plan, PROTOCOL_SHA as BASE_PROTOCOL_SHA
from wa.wm.failure_state_continuation_61836 import load_continuation
from wa.wm.failure_state_boundary_policy import CONTINUATION_POLICY
from wa.wm.failure_state_protocol_v2 import EXPERIMENT, PROTOCOL_SHA, search_recovery


def load_workload(plan_path, plan_sha, continuation_path, continuation_sha, shard, development_key=None):
    if type(shard) is not int or not 0 <= shard < 8:
        raise ValueError("eight original lanes required")
    plan = load_plan(plan_path, plan_sha)
    overlay = load_continuation(continuation_path, continuation_sha)
    if (overlay["base_plan"] != dict(path=str(Path(plan_path).resolve()), sha256=plan_sha)
            or overlay["expected_count"] != 126 or overlay["new_count"] != 90
            or overlay["reused_count"] != 36):
        raise ValueError("continuation does not bind the unchanged v1 base plan")
    entries = ([e for e in overlay["remaining_entries"] if e["key"] == development_key]
               if development_key is not None else overlay["remaining_lanes"][shard])
    if not entries or (development_key is not None and len(entries) != 1):
        raise ValueError("empty, reused or foreign v3 workload")
    base_entries = {e["key"]: e for e in plan["entries"]}
    if any(e != base_entries.get(e["key"]) for e in entries):
        raise ValueError("continuation entry differs from original definition")
    return plan, overlay, entries


def execute_entry(runtime, entry, base, teacher_url, *, development=False):
    student = runtime.run(entry, base, "student", teacher_url,
                          development_actions=24 if development else None)
    prefix = student.pop("prefix")
    if development:
        if len(prefix) != 24:
            raise ValueError("real developer prefix must contain24 actions")
        expected = runtime.originals[entry["key"]]["initial_pair_evidence"]
        from wa.wm.recovery_replay import check_dynamic_state
        if prefix[0]["rgb_sha256"] != expected["rgb"]:
            raise ValueError("developer original raw initial RGB mismatch")
        check_dynamic_state(expected["state"], prefix[0]["dynamic_state"])
        probes = {}
        for name in ("lightnav", "oracle"):
            probe = runtime.run(entry, base, name+"_takeover0020", teacher_url,
                                teacher_name=name, prefix=prefix, takeover_step=20,
                                development_actions=24)
            probe.pop("prefix")
            probes[name] = probe
        record = dict(status="DEVELOPER_NONZERO_REPLAY_CHECK_ONLY", key=entry["key"],
                      prefix_actions=24, takeover_step=20, branch_total_actions=24,
                      teacher_owned_actions=4, no_success_rate=True,
                      training_eligible=False, student=student, teachers=probes)
        return record
    def run(name, k, repeat):
        label = f"{name}_{k:04d}" + ("_repeat" if repeat else "")
        return runtime.run(entry, base, label, teacher_url, teacher_name=name,
                           prefix=prefix, takeover_step=k, verification_only=repeat)
    return search_recovery(student, prefix, run)


def completion_marker(args, completed, expected, development):
    """Build the production marker shared with launcher contract tests."""
    return dict(
        experiment=EXPERIMENT, entries=len(completed), keys=completed,
        expected=expected, shard=args.shard, development=development,
        training_released=False, no_success_rate=True, plan_sha256=args.plan_sha,
        base_protocol_sha256=BASE_PROTOCOL_SHA, protocol_sha256=PROTOCOL_SHA,
        continuation_sha256=args.continuation_sha,
        teacher_boundary_policy=CONTINUATION_POLICY)


def main():
    parser = argparse.ArgumentParser()
    for name in ("plan", "plan-sha", "continuation", "continuation-sha",
                 "ready", "teacher-url", "output"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--development-key")
    args = parser.parse_args()
    development = args.development_key is not None
    if development and (os.environ.get("WA_DEVELOPMENT") != "1" or os.environ.get("MD_AK_JOB_ID")):
        raise ValueError("developer check cannot be a cluster smoke")
    if not development and os.environ.get("WA_DEVELOPMENT") == "1":
        raise ValueError("explicit development key required")
    plan, overlay, entries = load_workload(
        args.plan, args.plan_sha, args.continuation, args.continuation_sha,
        args.shard, args.development_key)
    out = Path(args.output)
    if not str(out.resolve()).startswith("/data/nas_ray/"):
        raise ValueError("persistent NAS output required")
    out.mkdir(parents=True, exist_ok=False)
    runtime = Runtime(plan, args.plan_sha, json.loads(Path(args.ready).read_text()), out,
                      experiment=EXPERIMENT, protocol_sha=PROTOCOL_SHA,
                      continuation_sha=args.continuation_sha,
                      teacher_boundary_policy=CONTINUATION_POLICY)
    write_json(out/"plan_identity.json", dict(
        path=args.plan, sha256=args.plan_sha, base_protocol_sha256=BASE_PROTOCOL_SHA,
        experiment=EXPERIMENT, protocol_sha256=PROTOCOL_SHA,
        continuation_path=args.continuation, continuation_sha256=args.continuation_sha,
        base_plan_unchanged=True, old_records_relabelled=False,
        teacher_boundary_policy=CONTINUATION_POLICY))
    completed = []
    for entry in entries:
        base = out/entry["task"]/entry["key"]
        base.mkdir(parents=True, exist_ok=False)
        record = execute_entry(runtime, entry, base, args.teacher_url, development=development)
        record.update(
            experiment=EXPERIMENT, task=entry["task"], key=entry["key"],
            plan_sha256=args.plan_sha, base_protocol_sha256=BASE_PROTOCOL_SHA,
            continuation_sha256=args.continuation_sha, protocol_sha256=PROTOCOL_SHA,
            teacher_boundary_policy=CONTINUATION_POLICY,
            baseline_result_unchanged=entry["baseline_result"])
        write_json(base/("developer_check.json" if development else "search.json"), record)
        with (out/"records.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False)+"\n")
        completed.append(entry["key"])
        print("RECOVERY_ENTRY_COMPLETE", entry["key"], record.get("outcome", record.get("status")), flush=True)
    name = "DEVELOPMENT_CHECK.json" if development else "COMPLETE.json"
    write_json(out/name, completion_marker(args, completed, len(entries), development))
    print(name, len(completed), flush=True)


if __name__ == "__main__":
    main()
