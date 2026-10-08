"""CPU-only independent raw-evidence audit of the 90-only job61844 continuation.

Never runs models or simulation, releases training, rewrites source artifacts,
or re-decodes the previously audited36 searches.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

PRINT_LOCK = threading.Lock()

R = Path("/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928")
C = R / "checkout"
SOURCE = R / "source_failure_state_collect_v3"
ROOT = Path("/data/nas_ray/project/md-ak/users/zeying.gong/job_61844/task_73066/wa_failure_state_collect_a800_v3")
PLAN = R / "artifacts/failure_state_plan_61609_v1.json"
PLAN_SHA = "2eff9e83e89008ce1ee73632def406a27131fece6e77b01f6d4ca8c02624292b"
OVERLAY = R / "artifacts/failure_state_continuation_61836_v1.json"
OVERLAY_SHA = "e5fe2986517c171fd93f507ebe0f6d9937c420d3c37efd3fe7ff391899444641"
PREVIOUS = R / "artifacts/failure_state_completed_search_audit_61836_v1.json"
PREVIOUS_SHA = "cbdc709aa4f8df67fb5edf2e06faeb2476f77de365ec9954c8e5b323ed957299"
SOURCE_COMMIT = "726544449a5e7ad10c7d246b2bf302d783210c7e"
POLICY = "missing_rvq_or_final_l2_v1"
EXPERIMENT = "evaluation_adaptation_failure_state_v2"
OUTPUT = R / "artifacts/failure_state_completed_search_audit_61844_v1.json"
SCHEMA = "failure_state_completed_search_audit_61844_v1"
COUNTS = [12, 13, 8, 13, 12, 9, 12, 11]
CORE = ("schema", "base_plan", "continuation", "previous_audit", "searches",
        "training_released", "score_backfill_allowed")
AUDITOR_FILES = (
    "wa/wm/failure_state_search_audit.py", "wa/wm/failure_state_branch_audit.py",
    "wa/wm/failure_state_protocol.py", "wa/wm/failure_state_protocol_v2.py",
    "wa/wm/failure_state_teacher_error.py", "wa/wm/failure_state_teacher_adapter.py",
    "wa/wm/failure_state_boundary_policy.py", "wa/wm/recovery_replay.py",
    "wa/tools/audit_failure_state_completed_61844.py",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    p = Path(path)
    require(p.is_file() and not p.is_symlink(), "missing/symlink file: " + str(p))
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def same(a, b, name):
    require(canonical(a) == canonical(b), "mismatch: " + name)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def announce(**value):
    with PRINT_LOCK:
        print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def strict_loads(text):
    def pairs(items):
        value = {}
        for k, v in items:
            require(k not in value, "duplicate JSON key")
            value[k] = v
        return value
    def invalid(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def validate_rows(rows, expected_entries, frozen_keys):
    keys = [r["key"] for r in rows]
    same(keys, [e["key"] for e in expected_entries], "exact original lane order")
    require(len(keys) == len(set(keys)) and not set(keys).intersection(frozen_keys),
            "duplicate or reused search")
    for row in rows:
        require(row.get("task") == "stt" and row.get("experiment") == EXPERIMENT,
                "wrong task/experiment")
        require(row.get("teacher_boundary_policy") == POLICY, "unbound execution policy")
        require(row.get("training_released") is False and
                row.get("score_backfill_allowed") is False, "premature release")
    return keys


def validate_output_error(branch):
    """Narrow current-runtime decoder error identity, in addition to raw proof."""
    from wa.wm.failure_state_teacher_adapter import _missing_final_level
    require(branch.get("status") == "teacher_output_invalid", "unsupported runtime error")
    require(branch.get("teacher") == "lightnav", "only LightNav decoder boundary allowed")
    raw = branch.get("raw_error", {})
    require(raw.get("source") == "server_response" and type(raw.get("rc")) is int
            and raw["rc"] == 500 and type(raw.get("seq")) is int and raw["seq"] >= 0
            and raw.get("exception_type") is None, "unknown decoder error evidence")
    message = raw.get("message")
    require(type(message) is str, "decoder message required")
    legacy = re.fullmatch(r"Missing rvq act levels \[[0-9]+(?:, [0-9]+)*\] in .+",
                          message, flags=re.DOTALL) is not None
    final = _missing_final_level(message)
    require(legacy or final, "unknown decoder category is fatal")
    require(branch.get("result") is None and branch.get("complete") is False
            and branch.get("replay_verified") is False
            and branch.get("fallback_executed") is False
            and branch.get("training_eligible") is False
            and branch.get("training_released") is False, "invalid teacher cannot supply terminal/release")
    return "missing_rvq" if legacy else "missing_final_l2"


def validate_processes(events):
    result = {}
    for role in ("wa", "lightnav", "worker"):
        starts = [e for e in events if e.get("event") == "spawn" and e.get("role") == role]
        exits = [e for e in events if e.get("event") == "exit" and e.get("role") == role]
        require(len(starts) == len(exits) == 1, "exact owned spawn/exit required")
        a, b = starts[0], exits[0]
        require(type(a.get("pid")) is int and a["pid"] > 0 and
                a.get("process_group") == a["pid"] and b.get("pid") == a["pid"],
                "owned PID mismatch")
        require(type(b.get("returncode")) is int, "missing exit status")
        if role == "worker":
            require(b["returncode"] == 0, "worker failed")
        else:
            require(b["returncode"] in (0, -15, -9), "unexpected server cleanup exit")
        result[role] = dict(pid=a["pid"], returncode=b["returncode"])
    returned = [e for e in events if e.get("event") == "worker_return"]
    require(len(returned) == 1 and returned[0].get("returncode") == 0 and
            returned[0].get("pid") == result["worker"]["pid"], "worker return mismatch")
    require(not any(e.get("event") in ("cleanup_error", "error") for e in events),
            "process journal error")
    return result


def ordered_parallel_map(function, items, workers=4):
    """At most four active CPU searches; retain input order; propagate errors.

    Only bounded work is submitted, and unstarted work is cancelled on error.
    The caller must return per-search state rather than mutating shared evidence.
    """
    require(type(workers) is int and 1 <= workers <= 4, "workers must be 1..4")
    items = list(items)
    if workers == 1:
        return [function(item) for item in items]
    results, pending, next_index = [None] * len(items), {}, 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="raw-audit") as pool:
        try:
            while next_index < min(workers, len(items)):
                pending[pool.submit(function, items[next_index])] = next_index
                next_index += 1
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    results[index] = future.result()
                while len(pending) < workers and next_index < len(items):
                    pending[pool.submit(function, items[next_index])] = next_index
                    next_index += 1
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return results


def audit_completed():
    from wa.wm.failure_state_search_audit import audit_search
    from wa.wm.failure_state_branch_audit import audit_branch
    from wa.wm import failure_state_protocol as v1
    from wa.wm import failure_state_protocol_v2 as v2
    from wa.wm.failure_state_boundary_policy import validate_boundary_policy
    validate_boundary_policy(POLICY, experiment=EXPERIMENT)
    started, timer = stamp(), time.monotonic()
    inputs, source_sha, terminal, items = {}, {}, {}, []

    def read(path, expected=None, jsonl=False, text=False):
        path = Path(path)
        before = digest(path)
        if expected is not None:
            same(before, expected, "pinned SHA " + str(path))
        data = path.read_bytes()
        same(hashlib.sha256(data).hexdigest(), before, "stable read " + str(path))
        inputs[str(path)] = before
        if text:
            return data.decode(errors="replace")
        if jsonl:
            require(data.endswith(b"\n"), "incomplete JSONL")
            return [strict_loads(line) for line in data.splitlines()]
        return strict_loads(data)

    for name in AUDITOR_FILES:
        source_sha[str(C/name)] = digest(C/name)
    def frozen():
        commit = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "-C", str(SOURCE), "status", "--porcelain"], text=True)
        same(commit, SOURCE_COMMIT, "frozen commit")
        require(not status, "frozen source is dirty")
        return dict(path=str(SOURCE), commit=commit, status=status)
    terminal["frozen_source_before"] = frozen()
    p = read(PLAN, PLAN_SHA)
    o = read(OVERLAY, OVERLAY_SHA)
    old = read(PREVIOUS, PREVIOUS_SHA)
    same(o["base_plan"], dict(path=str(PLAN), sha256=PLAN_SHA), "overlay plan")
    same(o["completed_search_audit"], dict(path=str(PREVIOUS), sha256=PREVIOUS_SHA), "old audit pin")
    require(len(p["entries"]) == 126 and len(o["remaining_entries"]) == 90 and
            len(o["frozen_records"]) == len(old["searches"]) == 36, "fixed126 coverage")
    old_by_key = {s["row"]["key"]: s for s in old["searches"]}
    require(len(old_by_key) == 36 and old["training_released"] is False
            and old["score_backfill_allowed"] is False, "old audit coverage/release")
    for fr in o["frozen_records"]:
        s = old_by_key[fr["key"]]
        same(fr["row"], s["row"], "old frozen row")
        same(fr["audit_sha256"], canonical(s["audit"]), "old frozen audit")
    frozen_keys = set(old_by_key)
    entries = {e["key"]: e for e in p["entries"]}
    remaining = [e["key"] for e in o["remaining_entries"]]
    require(len(entries) == 126 and len(set(remaining)) == 90 and
            not frozen_keys.intersection(remaining) and
            frozen_keys.union(remaining) == set(entries), "exact disjoint126")
    same(o["remaining_entries"], [e for e in p["entries"] if e["key"] not in frozen_keys],
         "manifest order")
    same([len(l) for l in o["remaining_lanes"]], COUNTS, "eight fixed lane counts")
    manifest = read(p["manifest"]["path"], p["manifest"]["sha256"])
    originals = {r["key"]:r for r in read(p["sources"]["combined"]["path"],
                   p["sources"]["combined"]["sha256"], jsonl=True) if r["task"] == "stt"}
    require(len(originals) == 1405, "original STT coverage")
    for key, e in entries.items():
        same(canonical(originals[key]), e["baseline_row_sha256"], "baseline row")
    contract = dict(experiment=EXPERIMENT, protocol_sha256=v2.PROTOCOL_SHA,
                    plan_sha256=PLAN_SHA, continuation_sha256=OVERLAY_SHA,
                    teacher_boundary_policy=POLICY, training_released=False, no_success_rate=True)
    launch = read(ROOT/"launch.json")
    for k,v in dict(source=str(SOURCE), source_commit=SOURCE_COMMIT, source_git_status="",
                    formal=True, expected_entries=90, expected_lanes=8,
                    frozen_entries=36, total_expected_entries=126).items():
        same(launch.get(k), v, "launch " + k)
    same(launch.get("devices"), [str(i) for i in range(8)], "eight devices")
    for k in ("teacher_boundary_policy", "plan_sha256", "continuation_sha256"):
        same(launch.get(k), contract[k], "launch pin " + k)
    completion = read(ROOT/"COMPLETE.json")
    for k,v in dict(contract, status="COLLECTION_PROCESSES_COMPLETE_NOT_TRAINING_RELEASE",
                    entries=90, lanes=8, formal=True, keys=remaining).items():
        same(completion.get(k), v, "root completion " + k)
    coverage = read(ROOT/"coverage.json")
    for k,v in dict(new_count=90, reused_count=36, expected_total=126, formal=True,
                    new_keys=remaining, teacher_boundary_policy=POLICY,
                    training_released=False, no_success_rate=True).items():
        same(coverage.get(k), v, "coverage " + k)
    same(coverage.get("frozen_records"), o["frozen_records"], "preserved36 coverage")
    require(not (ROOT/"ERROR.json").exists(), "root ERROR exists")
    terminal.update(launch=launch, completion=completion, lanes=[])
    all_keys = []
    for lane in range(8):
        root = ROOT/f"lane{lane}"
        for name in ("ERROR.json", "cleanup_errors.json", "collection/ERROR.json"):
            require(not (root/name).exists(), "lane failure marker " + str(root/name))
        expected_entries = o["remaining_lanes"][lane]
        same(expected_entries, [e for e in p["lanes"][lane] if e["key"] not in frozen_keys],
             "original lane assignment")
        records = root/"collection/records.jsonl"
        rows = read(records, jsonl=True)
        keys = validate_rows(rows, expected_entries, frozen_keys)
        all_keys.extend(keys)
        cm = read(root/"collection/COMPLETE.json")
        for k,v in dict(contract, base_protocol_sha256=v1.PROTOCOL_SHA,
                        entries=COUNTS[lane], expected=COUNTS[lane], shard=lane,
                        development=False, keys=keys).items():
            same(cm.get(k), v, "lane completion " + k)
        outcome = read(root/"lane_outcome.json")
        require(outcome.get("status") == "PROCESS_AND_COMPLETION_VERIFIED",
                "lane outcome not verified")
        same(outcome.get("completion"), cm, "lane outcome completion")
        ready = read(root/"wa_ready.json")
        for k,v in dict(checkpoint=p["checkpoint"]["path"],
                        checkpoint_sha256=p["checkpoint"]["sha256"],
                        step=p["checkpoint"]["step"], gpu="NVIDIA A800-SXM4-80GB",
                        mode="mixed", noise_mode="zero", sampling_steps=4,
                        seed="7+step", text_used=False, world_predictor_inference=False).items():
            same(ready.get(k), v, "WA ready " + k)
        require(not (root/"teacher.ready").is_symlink(), "teacher ready symlink")
        events = read(root/"processes.jsonl", jsonl=True)
        owned = validate_processes(events)
        workers = [e for e in events if e.get("event") == "spawn" and e.get("role") == "worker"]
        env = workers[0]["command"]["env"]
        for k,v in dict(WA_DEVELOPMENT="0", WA_DIAG_CONTROLLER="learned_yaw_guard_v1",
                        WA_EVAL_MODE="mixed", WA_EVAL_CHECKPOINT_SHA=p["checkpoint"]["sha256"],
                        WA_EVAL_CHECKPOINT_STEP=str(p["checkpoint"]["step"]),
                        WA_INIT_REPAIR_PLAN=p["repair_plan"]["path"],
                        WA_INIT_REPAIR_PLAN_SHA=p["repair_plan"]["sha256"],
                        WA_SEMANTIC_PLY_FIX="mp3d_semantic_ply_v1").items():
            same(env.get(k), v, "worker environment " + k)
        log_counts = {}
        for log in ("worker.log", "teacher.log", "wa.log"):
            value = read(root/log, text=True)
            if log == "teacher.log":
                require("server listening on 127.0.0.1:" + str(19190+lane) in value,
                        "teacher listening evidence missing")
            fatal = {pat:value.count(pat) for pat in
                     ("Traceback (most recent call last):", "CUDA out of memory",
                      "ChildFailedError", "Segmentation fault", "REPLAY_MISMATCH")}
            log_counts[log] = dict(fatal_counts=fatal,
                warning_lines=sum("warning" in line.lower() for line in value.splitlines()))
        terminal["lanes"].append(dict(lane=lane, completed=len(rows), owned_exits=owned,
            logs=log_counts, ready=ready))
        items.extend((records, inputs[str(records)], row) for row in rows)
    require(len(all_keys) == len(set(all_keys)) == 90 and set(all_keys) == set(remaining),
            "all90 exact coverage")
    console = read(ROOT.parent/"console.log", text=True)
    require("FAILURE_STATE_V3_WRAPPER_END launcher_exit=0 dependency_exit=0" in console,
            "missing successful postdependency/wrapper")
    terminal["postdependency_wrapper_line"] = [l for l in console.splitlines()
        if "FAILURE_STATE_V3_WRAPPER_END" in l]
    terminal["scheduler_log_lines"] = [l for l in console.splitlines()
        if "task 73066 status" in l]
    teacher_server = Path("/data/nas_ray/home/zeying.gong/algorithm/repos/LightNav-0/src/lightnav/serving/ws_server.py")
    server_code = read(teacher_server, text=True)
    require("ready_file.unlink(missing_ok=True)" in server_code and "ready_file.touch()" in server_code,
            "teacher ready lifecycle changed")
    launcher_code = read(SOURCE/"wa/tools/failure_state_launch.py", text=True)
    require('(dest / "teacher.ready").exists()' in launcher_code, "teacher startup gate changed")
    terminal["teacher_ready_lifecycle"] = dict(server=str(teacher_server),
        source_sha256=inputs[str(teacher_server)], transient_empty_marker_removed_on_normal_cleanup=True,
        teacher_listening_logs_verified=8, worker_spawn_after_frozen_ready_gate=True)
    terminal["process_scope"] = "24 owned spawn/exit records matched; worker-container PIDs not queried in devpod PID namespace"
    # Hash the original policy checkpoint and repair plan, never load a model.
    same(digest(p["checkpoint"]["path"]), p["checkpoint"]["sha256"], "actual checkpoint")
    inputs[p["checkpoint"]["path"]] = p["checkpoint"]["sha256"]
    read(p["repair_plan"]["path"], p["repair_plan"]["sha256"])
    report = dict(schema=SCHEMA, base_plan=dict(path=str(PLAN),sha256=PLAN_SHA),
        continuation=dict(path=str(OVERLAY),sha256=OVERLAY_SHA),
        previous_audit=dict(path=str(PREVIOUS),sha256=PREVIOUS_SHA),
        searches=[], training_released=False,score_backfill_allowed=False)
    prov = dict(schema="failure_state_completed_search_audit_provenance_v1",
        started_utc=started, terminal=terminal, branch_statistics=[], errors=[],
        source_sha256=source_sha, training_released=False, score_backfill_allowed=False)
    totals, evidence = Counter(), {}
    announce(phase="TERMINAL_PASS_RAW_AUDIT_START", searches=90, lane_counts=COUNTS,
             old_searches_redecoded=0, cpu_search_workers=4, utc=stamp())
    progress_lock, completed = threading.Lock(), [0]
    def audit_one(item):
        seq, records, rsha, row = item
        statistics, errors = [], []
        def raw(root, **kw):
            got = audit_branch(root, **kw)
            ap = got["actualproof"]
            stat = dict(root=str(root), kind=got["kind"], teacher=kw["expected"]["teacher"],
                        k=kw["expected"]["takeover_step"], repeat=kw["expected"]["verification_only"],
                        frames=ap["observed_frames"],actions=ap["recorded_actions"],
                        candidate_windows=got["candidate_windows"],
                        pngs=sum(f.endswith(".png") for f in got["source_files"]),
                        source_inventory_sha256=canonical(got["source_files"]))
            if got["kind"] == "teacher_error":
                category = validate_output_error(got["branch"])
                errors.append(dict(root=str(root),category=category,branch=got["branch"]))
            statistics.append(stat)
            announce(phase="BRANCH_PASS",search_seq=seq,index=len(statistics),**stat)
            return got
        e=entries[row["key"]]
        metadata=dict(experiment=EXPERIMENT,partition="evaluation_adaptation",
            protocol_sha256=v2.PROTOCOL_SHA,plan_sha256=PLAN_SHA,
            checkpoint_sha256=p["checkpoint"]["sha256"],checkpoint_step=p["checkpoint"]["step"],
            source_dataset_sha256=manifest["tasks"]["stt"]["sha256"],seed=7,
            episode_id=e["episode_id"],base_plan_sha256=PLAN_SHA,
            base_protocol_sha256=v1.PROTOCOL_SHA,continuation_sha256=OVERLAY_SHA,
            teacher_boundary_policy=POLICY)
        got=audit_search(row,e,records.parent/"stt"/row["key"],
            original_start=originals[row["key"]]["initial_pair_evidence"],
            expected_metadata=metadata,audit_branch=raw)
        search=dict(job_id=61844,task_id=73066,records_path=str(records),
                    records_sha256=rsha,row=row,audit=got)
        candidate=got["candidate"]
        with progress_lock:
            completed[0] += 1
            announce(phase="SEARCH_PASS",seq=seq,completed=completed[0],key=row["key"],
                     outcome=got["outcome"],candidate_windows=0 if candidate is None else
                     candidate["candidate_windows"],utc=stamp())
        return search, statistics, errors
    # All evidence/provenance merging is on the main thread, in original lane order.
    audited = ordered_parallel_map(audit_one, [(i,*item) for i,item in enumerate(items,1)],4)
    for search, statistics, errors in audited:
        got=search["audit"]
        for name,d in got["source_files"].items():
            require(name not in evidence or evidence[name]==d,"conflicting evidence")
            evidence[name]=d
        report["searches"].append(search)
        prov["branch_statistics"].extend(statistics)
        prov["errors"].extend(errors)
        for stat in statistics:
            totals.update(branches=1,frames=stat["frames"],actions=stat["actions"],
                          pngs=stat["pngs"],teacher_errors=int(stat["kind"]=="teacher_error"))
        candidate=got["candidate"]
        totals.update(searches=1,candidates=int(candidate is not None),
                      candidate_windows=0 if candidate is None else candidate["candidate_windows"])
    prov["cpu_search_workers"] = 4
    prov["report_order"] = "original lane order, not parallel completion order"
    announce(phase="FINAL_SOURCE_REHASH",files=len(evidence),utc=stamp())
    for name,d in {**inputs,**source_sha,**evidence}.items():
        same(digest(name),d,"unchanged final source "+name)
    terminal["frozen_source_after"]=frozen()
    prov.update(ended_utc=stamp(),elapsed_s=time.monotonic()-timer,status="PASS_NONRELEASE",
        source_unchanged=True,source_input_sha256=inputs,totals=dict(totals),
        source_evidence_count=len(evidence),source_evidence_inventory_sha256=canonical(evidence),
        limitations=["90 new completed searches only; old36 referenced by previously audited hash",
            "candidate windows are not numerically converted or released training windows",
            "no new student SR; no score backfill; evaluation-set adaptation, not untouched test",
            "saved PNGs decoded and teacher prefix byte-paired; raw RGBA digest cannot be independently reconstructed from RGB PNG",
            "recorded simulator state proof does not establish hidden RNG/contact state equivalence",
            "allowed sidecar media hashed but not semantically audited",
            "scheduler SUCCEEDED appears in preserved task console; no fresh scheduler query made by this CPU audit"])
    return report,prov


def save_report(report, provenance, output):
    output=Path(output)
    require(output.is_absolute() and output.is_relative_to("/data/nas_ray") and output.parent.is_dir(),
            "persistent NAS output required")
    prov_path=output.with_suffix(".provenance.json")
    require(not output.exists() and not output.is_symlink() and
            not prov_path.exists() and not prov_path.is_symlink(),"never overwrite prior audit")
    require(set(report)==set(CORE) and report["schema"]==SCHEMA,"strict report schema")
    require(provenance.get("status")=="PASS_NONRELEASE" and
            provenance.get("source_unchanged") is True,"complete source-stable audit required")
    searches=report["searches"]
    require(len(searches)==90 and len({s["row"]["key"] for s in searches})==90,"exact90 required")
    for s in searches:
        require(set(s)=={"job_id","task_id","records_path","records_sha256","row","audit"} and
                s["job_id"]==61844 and s["task_id"]==73066,"foreign search/schema")
        require(s["audit"]["training_released"] is False and
                s["audit"]["score_backfill_allowed"] is False,"search release forbidden")
    require(report["training_released"] is False and report["score_backfill_allowed"] is False,
            "release forbidden")
    payload=json.dumps(report,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    sha=hashlib.sha256(payload).hexdigest()
    provenance=dict(provenance,report_path=str(output),report_sha256=sha,report_bytes=len(payload))
    pb=json.dumps(provenance,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    with output.open("xb") as f:f.write(payload)
    with prov_path.open("xb") as f:f.write(pb)
    return dict(path=str(output),sha256=sha,bytes=len(payload),
        provenance_path=str(prov_path),provenance_sha256=hashlib.sha256(pb).hexdigest(),
        provenance_bytes=len(pb),totals=provenance["totals"],training_released=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,default=OUTPUT)
    a=ap.parse_args()
    require(not a.output.exists() and not a.output.with_suffix(".provenance.json").exists(),
            "existing audit: read it, do not rerun or overwrite")
    report,prov=audit_completed()
    announce(phase="SAVED",**save_report(report,prov,a.output))


if __name__=="__main__":
    main()
