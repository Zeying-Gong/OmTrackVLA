"""Read-only 36-frozen/90-new continuation of FAILED 61836/73058.

This is not a training release or a replacement for the completed-search raw
media audit. That independently produced report and every referenced file are
required and rehashed. Original v1/v2 records and baseline numeric JSON types
remain unchanged. No runtime, scheduler query, file write or collection occurs.
"""
import copy
import hashlib
import json
from pathlib import Path
import re

from wa.wm import failure_state_continuation as previous
from wa.wm.failure_state_protocol import canonical_sha, load_plan, binary_flag
from wa.wm.failure_state_protocol_v2 import EXPERIMENT as V2, PROTOCOL_SHA as V2_SHA
from wa.wm.failure_state_search_audit import eligible, expected_candidates

SCHEMA = "wa_failure_state_continuation_61836_v1"
AUDIT_SCHEMA = "failure_state_completed_search_audit_61836_v1"
BASE_SHA = previous.BASE_SHA
PREVIOUS_SHA = "06d098f30ec52d988dc707249e73b17d11642f44846a75015455315891182822"
TERMINAL_SHA = "96dd66c1f3ddbc12a97f6977c238ab482ae66529ed0a0fd48e69784076905d26"
SOURCE_ROOT = Path("/data/nas_ray/project/md-ak/users/zeying.gong/job_61836/task_73058/wa_failure_state_collect_a800_v2")
SOURCE_COMMIT = "d676d9e760d066ef3c01c271bd9315ae2e296895"
COMPLETED_LANE_COUNTS = [3, 3, 8, 3, 4, 7, 3, 4]
REMAINING_LANE_COUNTS = [12, 13, 8, 13, 12, 9, 12, 11]
OUTCOMES = {"repeated_teacher_recovery_candidate": 23, "no_valid_teacher_recovery": 12}
HEX = re.compile(r"[0-9a-f]{64}")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def same(a, b, label):
    need(canonical_sha(a) == canonical_sha(b), "mismatch: " + label)


def sha_value(value):
    need(type(value) is str and HEX.fullmatch(value), "invalid SHA")
    return value


def path_value(path):
    p = Path(path)
    need(p.is_absolute() and ".." not in p.parts and p.resolve() == p,
         "absolute nonsymlink path required")
    return p


def decode(blob):
    def pairs(items):
        result = {}
        for k, v in items:
            need(k not in result, "duplicate JSON key")
            result[k] = v
        return result
    def bad(value):
        raise ValueError("nonfinite JSON: " + value)
    result = json.loads(blob, object_pairs_hook=pairs, parse_constant=bad)
    canonical_sha(result)  # Includes exponent-overflow rejection.
    return result


class Pins:
    def __init__(self):
        self.files, self.stat = {}, {}

    def blob(self, path, expected=None):
        p = path_value(path)
        need(p.is_file() and not p.is_symlink(), "missing/symlink evidence: " + str(p))
        st = p.stat()
        blob = p.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if expected is not None:
            same(digest, sha_value(expected), "evidence SHA: " + str(p))
        need(str(p) not in self.files or self.files[str(p)] == digest, "evidence changed during audit")
        self.files[str(p)] = digest
        self.stat[str(p)] = (st.st_size, st.st_mtime_ns, st.st_ino)
        need(len(blob) == st.st_size, "evidence changed while reading")
        return blob

    def read(self, path, expected=None):
        return decode(self.blob(path, expected))

    def lines(self, path, expected=None):
        blob = self.blob(path, expected)
        need(blob.endswith(b"\n"), "partial JSONL")
        need(all(line.strip() for line in blob.splitlines()), "blank JSONL row")
        return [decode(line) for line in blob.splitlines()]

    def finish(self):
        for name, original in self.stat.items():
            p = path_value(name)
            need(p.is_file() and not p.is_symlink(), "evidence disappeared")
            st = p.stat()
            same(list(original), [st.st_size, st.st_mtime_ns, st.st_ino], "evidence changed during audit")


def _false(doc, *fields):
    for field in fields:
        need(doc.get(field) is False, "explicit false required: " + field)


def _identity(doc, expected, label):
    same({k: doc.get(k) for k in expected}, expected, label)


def _terminal(report, plan, old, base, previous_pin, pins):
    _identity(report, dict(schema="wa_failure_state_terminal_evidence_v1", job=61836, task=73058,
                          status="FAILED", root=str(SOURCE_ROOT), completed=35,
                          lanes=COMPLETED_LANE_COUNTS, outcomes=OUTCOMES), "terminal report")
    _false(report, "training_released", "new_model_sr", "new_job_submitted")
    same(report.get("cleanup"), dict(role_exit_files=24, cleanup_errors=False, complete_markers=0,
                                    worker_lane4=1, worker_other=-15, lightnav=0, wa=-15), "terminal cleanup")
    need(SOURCE_ROOT.is_dir() and not SOURCE_ROOT.is_symlink(), "missing original61836 root")
    launch = pins.read(SOURCE_ROOT/"launch.json")
    expected = dict(experiment=V2, source_commit=SOURCE_COMMIT, source_git_status="",
                    plan=base["path"], plan_sha256=BASE_SHA, continuation=previous_pin["path"],
                    continuation_sha256=PREVIOUS_SHA, protocol_sha256=V2_SHA, formal=True,
                    expected_entries=125, expected_lanes=8, frozen_entries=1,
                    total_expected_entries=126, devices=[str(i) for i in range(8)],
                    training_released=False, no_success_rate=True)
    _identity(launch, expected, "original v2 launch")
    root_error = pins.read(SOURCE_ROOT/"ERROR.json", report["root_error_sha256"])
    _false(root_error, "training_released")
    need(root_error.get("no_success_rate") is True and root_error.get("error_type") == "RuntimeError"
         and root_error.get("message") == "collector failed; preserve original logs and partial outputs",
         "missing original root failure")
    need(not (SOURCE_ROOT/"COMPLETE.json").exists(), "conflicting root COMPLETE")
    need(isinstance(report.get("records"), list) and len(report["records"]) == 8, "eight terminal record pins required")
    records, exits, outcomes = [], [], {}
    for lane, terminal_lane in enumerate(report["records"]):
        same(terminal_lane.get("lane"), lane, "terminal lane order")
        keys = terminal_lane.get("keys")
        need(isinstance(keys, list) and len(keys) == COMPLETED_LANE_COUNTS[lane], "terminal completed count")
        same(keys, [e["key"] for e in old["remaining_lanes"][lane]][:len(keys)], "completed original lane prefix")
        dest = SOURCE_ROOT/f"lane{lane}"
        for marker in ("COMPLETE.json", "collection/COMPLETE.json", "cleanup_errors.json"):
            need(not (dest/marker).exists(), "conflicting lane completion/cleanup error")
        error = pins.read(dest/"ERROR.json")
        _false(error, "training_released")
        need(error.get("no_success_rate") is True and error.get("error_type") == "RuntimeError"
             and type(error.get("message")) is str and error["message"], "lane failure missing")
        header = pins.read(dest/"launch.json")
        _identity(header, dict(experiment=V2, plan_sha256=BASE_SHA, protocol_sha256=V2_SHA,
                               continuation_sha256=PREVIOUS_SHA, lane=lane, device=str(lane),
                               keys=[e["key"] for e in old["remaining_lanes"][lane]],
                               training_released=False, no_success_rate=True), "original lane launch")
        events = pins.lines(dest/"processes.jsonl")
        spawned, ended, returned = {}, {}, {}
        for ev in events:
            role, kind, pid = ev.get("role"), ev.get("event"), ev.get("pid")
            need(role in ("wa", "lightnav", "worker") and type(pid) is int and pid > 0, "invalid owned process")
            need(type(ev.get("utc")) is str and ev["utc"], "missing owned-process time")
            if kind == "spawn":
                need(role not in spawned and ev.get("process_group") == pid
                     and isinstance(ev.get("command"), dict), "duplicate/unowned spawn")
                spawned[role] = pid
            elif kind in ("worker_return", "exit"):
                need(spawned.get(role) == pid and type(ev.get("returncode")) is int, "unmatched process exit")
                if kind == "worker_return":
                    need(role == "worker" and role not in returned and role not in ended, "invalid worker return")
                    returned[role] = ev["returncode"]
                else:
                    need(role not in ended, "duplicate process exit")
                    ended[role] = ev["returncode"]
            else:
                raise ValueError("unknown process event")
        same(sorted(spawned), ["lightnav", "wa", "worker"], "three owned processes")
        same(ended, dict(lightnav=0, wa=-15, worker=1 if lane == 4 else -15), "three terminal exit codes")
        same(returned, dict(worker=1) if lane == 4 else {}, "only failing worker returned")
        exits.append(dict(lane=lane, roles={r:dict(pid=spawned[r], returncode=ended[r]) for r in sorted(ended)}))
        records_path = dest/"collection/records.jsonl"
        rows = pins.lines(records_path, terminal_lane["sha"])
        same([r.get("key") for r in rows], keys, "complete records order")
        for row in rows:
            outcomes[row.get("outcome")] = outcomes.get(row.get("outcome"), 0) + 1
            records.append(dict(job_id=61836, task_id=73058, lane=lane, records_path=str(records_path),
                                records_sha256=terminal_lane["sha"], row=row))
    same(outcomes, OUTCOMES, "terminal outcome counts")
    return records, launch, exits


def _branch_roots(row, base):
    refs = [(base/"student", row["student"])]
    for attempt in row.get("attempts", []):
        k = attempt.get("takeover_step")
        need(type(k) is int and k >= 0, "invalid takeover")
        need(set(attempt.get("branches", {})) == {"lightnav", "oracle"}, "missing same-k teacher branches")
        for teacher in ("lightnav", "oracle"):
            refs.append((base/f"{teacher}_{k:04d}", attempt["branches"][teacher]))
        if "repeat" in attempt:
            teacher = attempt.get("selection", {}).get("selected_teacher")
            need(teacher in ("lightnav", "oracle"), "repeat without teacher")
            refs.append((base/f"{teacher}_{k:04d}_repeat", attempt["repeat"]))
    need(len({str(p) for p, _ in refs}) == len(refs), "duplicate branch root")
    for path, ref in refs:
        if ref.get("status") in ("teacher_output_invalid", "teacher_transport_error"):
            need(ref.get("complete") is False and ref.get("result") is None,
                 "typed-error branch cannot have terminal result")
            evidence = ref.get("observed_partial", {}).get("evidence_files", {})
            for field, filename in (("partial_replay", "partial_replay.json"),
                                    ("partial_actions", "partial_actions.json")):
                same(evidence.get(field, {}).get("path"), str(path/filename),
                     "typed-error partial path differs")
            need("artifact_root" not in ref, "typed-error schema must retain original path evidence")
        else:
            same(ref.get("artifact_root"), str(path), "foreign branch artifact root")
    return refs


def _search(item, reference, entry, pins):
    need(set(item) == {"job_id", "task_id", "records_path", "records_sha256", "row", "audit"},
         "unexpected report search fields")
    for k in ("job_id", "task_id", "records_path", "records_sha256", "row"):
        same(item[k], reference[k], "report row source: " + k)
    row, audit = item["row"], item["audit"]
    job = item["job_id"]
    experiment, protocol = (previous.EXPERIMENT, previous.PROTOCOL_SHA) if job == 61833 else (V2, V2_SHA)
    _identity(row, dict(experiment=experiment, protocol_sha256=protocol, plan_sha256=BASE_SHA,
                        task="stt", key=entry["key"], baseline_result_unchanged=entry["baseline_result"]),
              "original row identity")
    _false(row, "training_released", "score_backfill_allowed")
    if job == 61836:
        _identity(row, dict(base_protocol_sha256=previous.PROTOCOL_SHA, continuation_sha256=PREVIOUS_SHA),
                  "original v2 provenance")
    base = path_value(Path(item["records_path"]).parent/"stt"/entry["key"])
    need(base.is_dir(), "missing complete search root")
    same(pins.read(base/"search.json"), row, "persisted complete search")
    _identity(audit, dict(schema="failure_state_search_audit_v1", task="stt", key=entry["key"],
                         outcome=row["outcome"], experiment=experiment, protocol_sha256=protocol),
              "independent audit identity")
    _false(audit, "training_released", "score_backfill_allowed")
    refs = _branch_roots(row, base)
    need(type(audit.get("branch_count")) is int and audit["branch_count"] == len(refs), "audit branch coverage")
    need(isinstance(audit.get("source_files"), dict) and audit["source_files"], "explicit independent source hashes required")
    expected_files = {str(base/"search.json")}
    branch_names = {p.name for p, _ in refs}
    allowed_sidecars = {name+"_metrics" for name in branch_names}
    allowed_sidecars |= {name+"_client" for name in branch_names if name.startswith("lightnav_")}
    sidecar_paths = set()
    seen_dirs = set()
    for child in base.iterdir():
        need(not child.is_symlink(), "symlink search member")
        if child.is_dir():
            seen_dirs.add(child.name)
            need(child.name in branch_names | allowed_sidecars, "unreviewed or interrupted branch under complete search")
            if child.name in allowed_sidecars:
                for artifact in child.rglob("*"):
                    need(not artifact.is_symlink() and (artifact.is_dir() or artifact.is_file()),
                         "symlink/nonfile sidecar")
                    if artifact.is_file():
                        sidecar_paths.add(str(artifact))
        else:
            need(child.is_file() and child.name in ("search.json", "student.trace.jsonl"),
                 "unknown search artifact")
            if child.name == "student.trace.jsonl":
                sidecar_paths.add(str(child))
    need(branch_names <= seen_dirs, "missing audited branch")
    need(audit.get("sidecar_media_semantics_audited") is False, "sidecar media semantics not audited")
    need(isinstance(audit.get("sidecar_files"), dict), "explicit sidecar pins required")
    same(sorted(audit["sidecar_files"]), sorted(sidecar_paths), "sidecar coverage")
    for name, digest in audit["sidecar_files"].items():
        same(audit["source_files"].get(name), digest, "sidecar pin differs")
    expected_files |= sidecar_paths
    for branch, ref in refs:
        path_value(branch)
        need(branch.is_dir() and not branch.is_symlink(), "missing/symlink branch")
        for p in branch.iterdir():
            need(p.is_file() and not p.is_symlink(), "nonfile/symlink branch artifact")
            expected_files.add(str(p))
        if branch.name != "student":
            same(pins.read(branch/"branch.json"), ref, "branch reference")
    same(sorted(audit["source_files"]), sorted(expected_files), "complete exact independent branch/file coverage")
    for name, digest in sorted(audit["source_files"].items()):
        pins.blob(name, digest)
    result = row["student"]["result"]
    need(type(result.get("policy_init_valid")) is bool, "student initialization flag required")
    success, collision = (binary_flag(result.get(x), x) for x in ("success", "collision"))
    need(not success or result["policy_init_valid"] and not collision, "inconsistent student success")
    attempts = row.get("attempts")
    need(isinstance(attempts, list), "missing attempts")
    chosen = None
    if not result["policy_init_valid"] or success:
        need(not attempts, "diagnostic-only student cannot have teacher attempts")
        outcome = "student_invalid_initialization" if not result["policy_init_valid"] else "rerun_student_success_no_recovery_needed"
    else:
        prefix = pins.read(base/"student/replay.json")
        candidates = expected_candidates(len(prefix))
        need(0 < len(attempts) <= len(candidates), "invalid bounded search")
        for index, attempt in enumerate(attempts):
            k = candidates[index]
            same(attempt["takeover_step"], k, "bounded descending search")
            branches = attempt["branches"]
            flags = {t:eligible(branches[t]) for t in ("lightnav", "oracle")}
            selected = None
            if any(flags.values()):
                selected = ("oracle" if flags["oracle"] and (not flags["lightnav"] or
                            branches["oracle"]["result"]["following_rate"] > branches["lightnav"]["result"]["following_rate"])
                            else "lightnav")
            selection = attempt.get("selection", {})
            same(selection.get("selected_teacher"), selected, "winner success/TR/tie")
            same(selection.get("eligible"), flags, "eligibility")
            _false(selection, "training_released")
            if selected is None:
                need("repeat" not in attempt and "repeat_valid" not in attempt, "repeat without winner")
                continue
            repeat = attempt.get("repeat")
            need(isinstance(repeat, dict) and repeat.get("verification_only") is True, "independent repeat missing")
            same(attempt.get("repeat_valid"), eligible(repeat), "repeat eligibility")
            original = branches[selected]
            need(original.get("verification_only") is False, "repeat cannot supply demonstration")
            wins = pins.read(base/f"{selected}_{k:04d}"/"windows.json")
            count = len(wins)
            same(original.get("candidate_windows"), count, "candidate windows file count")
            same(attempt.get("selected_candidate_windows"), count, "selected candidate count")
            if eligible(repeat) and count:
                need(index == len(attempts)-1, "extra attempts after accepted recovery")
                chosen = dict(teacher=selected, takeover_step=k, branch=str(base/f"{selected}_{k:04d}"),
                              repeat_branch=str(base/f"{selected}_{k:04d}_repeat"),
                              window_indices=[w["current_index"] for w in wins], candidate_windows=count)
                break
        need(chosen is not None or len(attempts) == len(candidates), "premature search stop")
        outcome = "repeated_teacher_recovery_candidate" if chosen else "no_valid_teacher_recovery"
    same(row["outcome"], outcome, "outcome rederived")
    if chosen is None:
        need(row.get("accepted") is None and audit.get("candidate") is None, "unaccepted search has demonstration")
    else:
        candidate = audit.get("candidate")
        need(isinstance(candidate, dict), "missing independent candidate")
        expected = dict(chosen, task="stt", key=entry["key"], experiment=experiment,
                        protocol_sha256=protocol,
                        source_files={p:h for p,h in audit["source_files"].items() if p not in sidecar_paths},
                        training_released=False, conversion_pending=True)
        same(candidate, expected, "independent candidate exact")
        accepted = row.get("accepted", {})
        _identity(accepted, dict(teacher=chosen["teacher"], takeover_step=chosen["takeover_step"],
                                artifact_root=chosen["branch"], repeat_artifact_root=chosen["repeat_branch"],
                                candidate_only=True, training_released=False), "row accepted original winner")
    return dict(task="stt", key=entry["key"], original_lane=reference["lane"],
                source_job_id=job, source_task_id=item["task_id"],
                source_experiment=experiment, source_protocol_sha256=protocol,
                records_path=item["records_path"], records_sha256=item["records_sha256"],
                row_sha256=canonical_sha(row), row=copy.deepcopy(row),
                search_path=str(base/"search.json"), search_sha256=pins.files[str(base/"search.json")],
                audit_sha256=canonical_sha(audit), outcome=outcome,
                candidate=copy.deepcopy(chosen), training_released=False, score_backfill_allowed=False)


def build_continuation(base_plan_path, base_sha, previous_continuation_path, previous_sha,
                       audit_report_path, audit_sha, *, expected_job_status=None):
    """Construct without writing. FAILED is caller-confirmed, not a live query.

    The auditor report is evidence of prior full branch/media validation, not a
    release flag. This constructor rechecks its exact rows, bounded selection,
    full file coverage and every hash, plus terminal provenance and ownership.
    Formal submission must freshly re-query both source jobs' terminal states.
    """
    need(expected_job_status == "FAILED", "explicit scheduler-verified FAILED required")
    need(base_sha == BASE_SHA and previous_sha == PREVIOUS_SHA, "foreign base/previous SHA")
    pins = Pins()
    base_path, prev_path, audit_path = map(path_value, (base_plan_path, previous_continuation_path, audit_report_path))
    base = dict(path=str(base_path), sha256=BASE_SHA)
    prev_pin = dict(path=str(prev_path), sha256=PREVIOUS_SHA)
    pins.blob(base_path, BASE_SHA)
    plan = load_plan(base_path, BASE_SHA)
    before = canonical_sha(plan)
    pins.blob(prev_path, PREVIOUS_SHA)
    old = previous.load_continuation(prev_path, PREVIOUS_SHA)
    _identity(old, dict(base_plan=base, expected_count=126, new_count=125, reused_count=1,
                        reused_keys=[previous.KEY], training_released=False), "original one-key overlay")
    report = pins.read(audit_path, sha_value(audit_sha))
    same(sorted(report), sorted({"schema", "base_plan", "previous_continuation", "terminal_report", "searches",
                       "training_released", "score_backfill_allowed"}), "audit report schema fields")
    _identity(report, dict(schema=AUDIT_SCHEMA, base_plan=base, previous_continuation=prev_pin,
                          training_released=False, score_backfill_allowed=False), "audit report inputs")
    terminal_pin = report["terminal_report"]
    need(set(terminal_pin) == {"path", "sha256"} and terminal_pin["sha256"] == TERMINAL_SHA,
         "foreign terminal evidence report")
    terminal = pins.read(terminal_pin["path"], TERMINAL_SHA)
    new_records, launch, exits = _terminal(terminal, plan, old, base, prev_pin, pins)
    old_frozen = old["frozen_records"]
    need(len(old_frozen) == 1 and old_frozen[0]["key"] == previous.KEY, "old frozen identity")
    frozen = old_frozen[0]
    rows = pins.lines(frozen["records_path"], frozen["records_sha256"])
    need(len(rows) == 1 and rows[0]["key"] == previous.KEY, "old single row changed")
    same(canonical_sha(rows[0]), frozen["row_sha256"], "old canonical row")
    references = [dict(job_id=61833, task_id=73055, lane=0, records_path=frozen["records_path"],
                       records_sha256=frozen["records_sha256"], row=rows[0])] + new_records
    entries = {e["key"]:e for e in plan["entries"]}
    need(len(entries) == 126 and len(plan["lanes"]) == 8, "base126 uniqueness")
    need(len(references) == 36 and len({r["row"]["key"] for r in references}) == 36, "frozen overlap/duplicate")
    searches = report.get("searches")
    need(isinstance(searches, list) and len(searches) == 36, "full36 independent search audits required")
    keyed = {}
    for item in searches:
        key = (item.get("job_id"), item.get("row", {}).get("key"))
        need(key not in keyed, "duplicate audited search")
        keyed[key] = item
    same(sorted(keyed), sorted((r["job_id"], r["row"]["key"]) for r in references), "independent audit coverage")
    frozen_records = []
    for reference in references:
        key = reference["row"]["key"]
        need(key in entries and any(e["key"] == key for e in plan["lanes"][reference["lane"]]), "foreign original lane/key")
        frozen_records.append(_search(keyed[(reference["job_id"], key)], reference, entries[key], pins))
    reused = {r["key"] for r in frozen_records}
    remaining = [copy.deepcopy(e) for e in plan["entries"] if e["key"] not in reused]
    lanes = [[copy.deepcopy(e) for e in lane if e["key"] not in reused] for lane in plan["lanes"]]
    need(len(remaining) == 90 and len({e["key"] for e in remaining}) == 90, "new90 uniqueness")
    same(list(map(len, lanes)), REMAINING_LANE_COUNTS, "original90 lane allocation")
    need(not reused & {e["key"] for e in remaining} and reused | {e["key"] for e in remaining} == set(entries),
         "frozen/new overlap or missing original keys")
    same(plan, decode(pins.blob(base_path, BASE_SHA)), "base plan immutable")
    same(canonical_sha(plan), before, "base plan mutated")
    pins.finish()
    return dict(schema=SCHEMA, base_plan=base, previous_continuation=prev_pin,
                completed_search_audit=dict(path=str(audit_path), sha256=audit_sha),
                terminal_report=copy.deepcopy(terminal_pin), expected_count=126, new_count=90, reused_count=36,
                reused_keys=[e["key"] for e in plan["entries"] if e["key"] in reused],
                frozen_records=frozen_records, remaining_entries=remaining, remaining_lanes=lanes,
                source=dict(root=str(SOURCE_ROOT), job_id=61836, task_id=73058, expected_job_status="FAILED",
                            source=launch["source"], source_commit=SOURCE_COMMIT, source_git_status="",
                            owned_process_exits=exits, scheduler_recheck_required=True,
                            status_authority="caller-confirmed scheduler plus pinned terminal report/errors/owned exits"),
                source_files=dict(sorted(pins.files.items())), base_plan_unchanged=True,
                evaluation_adaptation=True, untouched_test=False, no_success_rate=True,
                training_released=False, score_backfill_allowed=False,
                release_requirements=["fresh scheduler state for61833/61836 and unchanged evidence",
                                      "exact36-original-provenance plus90-new coverage without replacement",
                                      "independent candidate suffix SE2/yaw/history/input audit before training"])


def load_continuation(path, sha):
    """Strictly rebuild, including all source hashes; numeric JSON types matter."""
    pins = Pins()
    overlay = pins.read(path_value(path), sha_value(sha))
    need(overlay.get("schema") == SCHEMA, "foreign continuation schema")
    base, old, audit, source = (overlay[k] for k in
                               ("base_plan", "previous_continuation", "completed_search_audit", "source"))
    expected = build_continuation(base["path"], base["sha256"], old["path"], old["sha256"],
                                  audit["path"], audit["sha256"],
                                  expected_job_status=source.get("expected_job_status"))
    same(overlay, expected, "continuation exactly matches rebuilt frozen sources")
    pins.finish()
    return overlay
