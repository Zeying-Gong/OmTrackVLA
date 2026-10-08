"""Full126 failure-state collection admission; not a training/cache release.

Consumes independent full raw/search audits and numerical audits. This gate
rechecks exact plan/lane coverage, all pinned files, original winner selection
and numeric counts. It never reruns simulation or decodes previously audited
media, and does not turn teacher recovery into a student score.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from wa.wm.failure_state_protocol import canonical_sha
from wa.wm import failure_state_protocol as v1
from wa.wm import failure_state_protocol_v2 as v2
from wa.wm.failure_state_boundary_policy import CONTINUATION_POLICY
from wa.wm.failure_state_numeric import audit_numeric_windows
from wa.wm.failure_state_search_audit import (
    eligible, expected_candidates, require, same,
)

SCHEMA = "failure_state_collection_release_v1"
R = Path("/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928")
ROOT = Path("/data/nas_ray/project/md-ak/users/zeying.gong/job_61844/task_73066/wa_failure_state_collect_a800_v3")
COLLECTION_SOURCE = R/"source_failure_state_collect_v3"
COLLECTION_COMMIT = "726544449a5e7ad10c7d246b2bf302d783210c7e"
PLAN = R/"artifacts/failure_state_plan_61609_v1.json"
PLAN_SHA = "2eff9e83e89008ce1ee73632def406a27131fece6e77b01f6d4ca8c02624292b"
OVERLAY = R/"artifacts/failure_state_continuation_61836_v1.json"
OVERLAY_SHA = "e5fe2986517c171fd93f507ebe0f6d9937c420d3c37efd3fe7ff391899444641"
OLD_AUDIT = R/"artifacts/failure_state_completed_search_audit_61836_v1.json"
OLD_AUDIT_SHA = "cbdc709aa4f8df67fb5edf2e06faeb2476f77de365ec9954c8e5b323ed957299"
OLD_NUMERIC = R/"checkout/wa/results/FAILURE_STATE_36_NUMERIC_20261008.json"
OLD_NUMERIC_SHA = "2965e8bd086169eb3c180dd92db72254c2d420d445d10f7586fc6e4be850340b"


def strict_json(data):
    """Parse the exact hashed bytes, not a second unpinned file read."""
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


class Pins:
    """Stream inventory hashes; retain bytes only for requested small documents."""
    def __init__(self):
        self.files = {}
        self.stats = {}

    def _read(self, path, expected=None, *, retain=False):
        path = Path(path)
        require(path.is_absolute() and path.is_file() and not path.is_symlink()
                and path.resolve() == path, "absolute nonsymlink evidence required")
        st = path.stat()
        state = (st.st_size, st.st_mtime_ns, st.st_ino)
        name = str(path)
        if name in self.stats:
            require(state == self.stats[name], "evidence changed during admission")
        digest, pieces = hashlib.sha256(), []
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                if retain:
                    pieces.append(chunk)
        after = path.stat()
        require(not path.is_symlink() and path.resolve() == path and
                (after.st_size, after.st_mtime_ns, after.st_ino) == state,
                "evidence changed while reading")
        digest = digest.hexdigest()
        require(expected is None or digest == expected, "evidence SHA mismatch: " + name)
        require(name not in self.files or self.files[name] == digest, "conflicting evidence SHA")
        self.files[name], self.stats[name] = digest, state
        return b"".join(pieces) if retain else digest

    def blob(self, path, expected=None):
        return self._read(path, expected, retain=True)

    def doc(self, path, expected=None):
        return strict_json(self.blob(path, expected))

    def verify_all(self, inventory):
        require(isinstance(inventory, dict) and inventory, "nonempty source inventory required")
        for path, digest in inventory.items():
            require(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest),
                    "invalid source SHA")
            if path in self.files:
                require(self.files[path] == digest, "conflicting source pin")
            else:
                self._read(path, digest)

    def finish(self):
        for name, state in self.stats.items():
            path = Path(name)
            st = path.stat()
            require(not path.is_symlink() and path.resolve() == path and
                    (st.st_size, st.st_mtime_ns, st.st_ino) == state,
                    "source changed before admission finished")


def validate_protocol(row, *, new=False):
    experiment = row.get("experiment")
    require(experiment in (v1.EXPERIMENT, v2.EXPERIMENT), "unknown source experiment")
    protocol = v1.PROTOCOL_SHA if experiment == v1.EXPERIMENT else v2.PROTOCOL_SHA
    same(row.get("protocol_sha256"), protocol, "fixed source protocol")
    require(row.get("task") == "stt", "only fixed STT searches admitted")
    if new:
        same(experiment, v2.EXPERIMENT, "continuation experiment")
        same(row.get("teacher_boundary_policy"), CONTINUATION_POLICY, "continuation boundary")
    else:
        require("teacher_boundary_policy" not in row, "old frozen execution policy changed")


def validate_marker(marker, keys, *, lane=None):
    fields = dict(experiment=v2.EXPERIMENT, protocol_sha256=v2.PROTOCOL_SHA,
        plan_sha256=PLAN_SHA, continuation_sha256=OVERLAY_SHA,
        teacher_boundary_policy=CONTINUATION_POLICY, training_released=False,
        no_success_rate=True, keys=keys)
    if lane is None:
        fields.update(status="COLLECTION_PROCESSES_COMPLETE_NOT_TRAINING_RELEASE",
                      entries=90, lanes=8, formal=True)
    else:
        require(type(lane) is int and 0 <= lane < 8, "invalid lane")
        fields.update(base_protocol_sha256=v1.PROTOCOL_SHA, entries=len(keys),
                      expected=len(keys), shard=lane, development=False)
    for field, value in fields.items():
        same(marker.get(field), value, "completion " + field)


def validate_provenance(report, provenance, *, size, path, digest, pins):
    """Old failures are fatal; new typed branch errors are verified nonterminals."""
    require(size in (36, 90), "unsupported audit scope")
    suffix = "61836" if size == 36 else "61844"
    same(report.get("schema"), f"failure_state_completed_search_audit_{suffix}_v1",
         "raw report schema")
    same(provenance.get("schema"), "failure_state_completed_search_audit_provenance_v1",
         "provenance schema")
    same(provenance.get("report_path"), str(path), "provenance report path")
    same(provenance.get("report_sha256"), digest, "provenance report SHA")
    require(provenance.get("status") == "PASS_NONRELEASE"
            and provenance.get("source_unchanged") is True
            and provenance.get("training_released") is False
            and provenance.get("score_backfill_allowed") is False, "raw audit provenance failed")
    if size == 36:
        same(provenance.get("failures"), [], "old raw failures")
    else:
        # These are observed, narrowly classified teacher decoder failures,
        # not auditor/task failures. Unknown/transport errors remain fatal.
        require("failures" not in provenance or provenance["failures"] == [],
                "new raw audit failures")
        errors = provenance.get("errors")
        require(isinstance(errors, list), "new typed error inventory required")
        roots = set()
        from wa.wm.failure_state_teacher_adapter import _missing_final_level
        for error in errors:
            root = error.get("root")
            require(type(root) is str and Path(root).is_relative_to(ROOT)
                    and root not in roots, "duplicate/foreign typed error")
            roots.add(root)
            branch = error.get("branch", {})
            require(branch.get("status") == "teacher_output_invalid"
                    and branch.get("teacher") == "lightnav"
                    and branch.get("result") is None, "typed error is not a terminal")
            for flag in ("complete", "replay_verified", "fallback_executed",
                         "training_eligible", "training_released"):
                require(branch.get(flag) is False, "unsafe typed error flag: " + flag)
            raw = branch.get("raw_error", {})
            require(raw.get("source") == "server_response"
                    and type(raw.get("rc")) is int and raw["rc"] == 500
                    and type(raw.get("seq")) is int and raw["seq"] >= 0
                    and raw.get("exception_type") is None
                    and type(raw.get("message")) is str, "unknown error transport")
            message = raw["message"]
            legacy = re.fullmatch(r"Missing rvq act levels \[[0-9]+(?:, [0-9]+)*\] in .+",
                                  message, flags=re.DOTALL) is not None
            final = _missing_final_level(message)
            require(legacy or final, "unclassified decoder error")
            same(error.get("category"), "missing_rvq" if legacy else "missing_final_l2",
                 "typed error category")
        same(provenance.get("totals", {}).get("teacher_errors"), len(errors),
             "typed error count")
        pins.verify_all(provenance.get("source_input_sha256"))
    pins.verify_all(provenance.get("source_sha256"))


def validate_numeric_report(numbers, size):
    """Check declarations and summaries without trusting their aggregate counts."""
    require(size in (36, 90), "unsupported numeric scope")
    suffix = "61836" if size == 36 else "61844"
    same(numbers.get("schema"), f"failure_state_candidate_numeric_audit_{suffix}_v1",
         "numeric report schema")
    for field in ("training_eligible", "training_released", "cache_generated"):
        require(numbers.get(field) is False, "numeric declaration: " + field)
    require(numbers.get("no_success_rate") is True, "numeric results are not SR")
    require(isinstance(numbers.get("code_sha256_before"), dict)
            and numbers["code_sha256_before"], "numeric code provenance required")
    same(numbers.get("code_sha256_after"), numbers["code_sha256_before"], "numeric code changed")
    episodes, summary = numbers.get("episodes"), numbers.get("summary")
    require(isinstance(episodes, list) and isinstance(summary, dict), "numeric structure")
    require(len({e["key"] for e in episodes}) == len(episodes), "duplicate numeric key")
    for name in ("candidate_windows", "valid_windows", "excluded_windows"):
        require(all(type(e.get(name)) is int and e[name] >= 0 for e in episodes),
                "numeric episode count: " + name)
        same(summary.get(name), sum(e[name] for e in episodes), "numeric total " + name)
    same(summary.get("accepted_original_episodes"), len(episodes), "numeric episode total")
    same(summary.get("episodes_with_valid_windows"),
         sum(e["valid_windows"] > 0 for e in episodes), "numeric nonempty episodes")
    require(summary.get("repeat_windows_counted") == 0
            and type(summary.get("repeat_windows_counted")) is int
            and summary.get("student_prefix_actions_used_as_future_labels") == 0
            and type(summary.get("student_prefix_actions_used_as_future_labels")) is int
            and summary.get("all_candidate_future_transitions_teacher_owned") is True,
            "numeric ownership summary")
    for e in episodes:
        require(e["candidate_windows"] > 0 and
                e["valid_windows"] + e["excluded_windows"] == e["candidate_windows"],
                "numeric count partition")
    reasons = Counter()
    for e in episodes:
        for key, count in e.get("rejection_reason_counts", {}).items():
            require(type(count) is int and count >= 0, "numeric rejection count")
            reasons[key] += count
    same(summary.get("rejection_reason_counts"), dict(reasons), "numeric reason total")
    # Audit all additive fields shared by every episode, while preserving the
    # historic old36 report's explicitly different diagnostic field names.
    for key, value in summary.items():
        if episodes and type(value) is int and all(type(e.get(key)) is int for e in episodes):
            same(value, sum(e[key] for e in episodes), "numeric additive total " + key)
    if size == 36:
        same(numbers.get("input_report"), dict(path=str(OLD_AUDIT), sha256=OLD_AUDIT_SHA),
             "numeric original raw audit")
        same([summary[k] for k in ("accepted_original_episodes", "candidate_windows",
                                  "valid_windows", "excluded_windows")],
             [24, 2039, 1891, 148], "fixed old numeric totals")
    else:
        same(summary.get("completed_searches"), 90, "new numeric scope")
        same((numbers.get("job_id"), numbers.get("task_id")), (61844, 73066), "numeric job")
        same(numbers.get("root"), str(ROOT), "numeric root")
        require(numbers.get("full_search_media_audit") is False, "numeric/media boundary")
        same(numbers.get("source_collection"), dict(experiment=v2.EXPERIMENT,
             protocol_sha256=v2.PROTOCOL_SHA, boundary_policy=CONTINUATION_POLICY,
             commit=COLLECTION_COMMIT, source=str(COLLECTION_SOURCE)), "numeric collection identity")
        no_candidate = numbers.get("no_candidate")
        require(isinstance(no_candidate, list) and all(isinstance(e, dict) for e in no_candidate),
                "numeric no-candidate records")
        empty_keys = [e.get("key") for e in no_candidate]
        require(all(type(k) is str for k in empty_keys) and len(set(empty_keys)) == len(empty_keys)
                and not set(empty_keys).intersection(e["key"] for e in episodes),
                "numeric no-candidate overlap/duplicates")
        same(summary.get("no_candidate_searches"), len(numbers.get("no_candidate", [])),
             "numeric no-candidate total")
        same(summary["no_candidate_searches"] + len(episodes), 90, "numeric total searches")
        same(numbers.get("old36_reference", {}).get("path"), str(OLD_NUMERIC),
             "numeric old reference")
        same(numbers.get("old36_reference", {}).get("sha256"), OLD_NUMERIC_SHA,
             "numeric old reference SHA")
    return episodes


def exact_numeric(candidate, episode, row, pins, *, new):
    """Recompute only numeric tensors; never images, simulation, or model state."""
    numeric_entry(candidate, episode)
    for field in ("experiment", "protocol_sha256"):
        same(episode.get("source_" + field), row[field], "numeric source " + field)
    if new:
        same(episode.get("source_boundary_policy"), CONTINUATION_POLICY, "numeric boundary")
        same(episode.get("repeat_branch"), candidate["repeat_branch"], "numeric repeat")
    branch = Path(candidate["branch"])
    observations = pins.doc(branch/"observations.json")
    actions = pins.doc(branch/"actions.json")
    windows = pins.doc(branch/"windows.json")
    same([w["current_index"] for w in windows], candidate["window_indices"],
         "numeric exact candidate indices")
    derived = audit_numeric_windows(observations, actions, windows,
                                    candidate["takeover_step"], per_window=False)
    summary = derived["summary"]
    same(derived["derived"]["ownership_audit"]["teacher"], candidate["teacher"],
         "numeric executed teacher")
    for key in ("candidate_windows", "valid_windows", "excluded_windows",
                "rejection_reason_counts"):
        same(episode.get(key), summary[key], "numeric recomputation " + key)
    for key in ("early_candidates", "late_candidates", "early_valid", "late_valid",
                "valid_history_uses_student_prefix", "old_mask_disagreement_windows"):
        if key in episode:
            same(episode[key], summary[key], "numeric recomputation " + key)
    valid, excluded = derived["valid_window_indices"], derived["excluded"]
    if new:
        same(episode.get("valid_window_indices"), valid, "numeric exact valid indices")
        same(episode.get("excluded"), excluded, "numeric exact exclusions")
    return dict(valid_window_indices=valid, numeric_exclusions=excluded,
                expected_valid_count=len(valid), expected_excluded_count=len(excluded))

def exact_partition(entries, old_keys, new_lanes):
    """Pure exact-once admission; unit-testable without NAS or model libraries."""
    require(len(entries) == 126 and all(e["task"] == "stt" for e in entries),
            "the fixed126 STT plan is required")
    keys = [e["key"] for e in entries]
    new = [e["key"] for lane in new_lanes for e in lane]
    require(len(keys) == len(set(keys)) == 126, "duplicate plan keys")
    require(len(new_lanes) == 8 and len(new) == len(set(new)) == 90,
            "exact90 new keys/eight lanes required")
    require(len(old_keys) == len(set(old_keys)) == 36, "exact36 frozen keys required")
    require(not set(old_keys) & set(new), "old/new overlap")
    require(set(keys) == set(old_keys) | set(new), "incomplete or foreign coverage")


def numeric_entry(candidate, episode):
    require(episode is not None, "candidate lacks independent numeric audit")
    for key in ("task", "key", "teacher", "branch", "takeover_step", "candidate_windows"):
        same(episode.get(key), candidate[key], "numeric identity: " + key)
    n, v, x = (episode.get(k) for k in ("candidate_windows", "valid_windows", "excluded_windows"))
    require(all(type(i) is int for i in (n, v, x)) and n > 0 and v >= 0 and x >= 0 and v+x == n,
            "invalid numeric counts")
    require(type(episode.get("template_index")) is int and episode["template_index"] == 0,
            "episode-zero template required")
    for name in ("causal_history_verified", "future_teacher_ownership_verified",
                 "jepa_command_available_for_valid", "previous_action_available_for_valid"):
        require(episode.get(name) is True, "numeric invariant missing: " + name)
    return v


def choose_original(row, pins, base):
    """Re-derive bounded success/TR/tie selection from pinned search artifacts."""
    result = row["student"]["result"]
    attempts = row["attempts"]
    if not result["policy_init_valid"] or bool(result["success"]):
        require(not attempts and row.get("accepted") is None, "diagnostic student admitted")
        return None
    prefix = pins.doc(base/"student/replay.json")
    ks = expected_candidates(len(prefix))
    require(0 < len(attempts) <= len(ks), "unbounded or missing attempts")
    for i, attempt in enumerate(attempts):
        k = ks[i]
        same(attempt["takeover_step"], k, "fixed descending takeover order")
        refs = attempt["branches"]
        require(set(refs) == {"oracle", "lightnav"}, "both teachers required")
        flags = {t: eligible(refs[t]) for t in refs}
        selected = None
        if any(flags.values()):
            selected = ("oracle" if flags["oracle"] and
                        (not flags["lightnav"] or refs["oracle"]["result"]["following_rate"] >
                         refs["lightnav"]["result"]["following_rate"]) else "lightnav")
        same(attempt["selection"]["selected_teacher"], selected, "success/TR selection")
        same(attempt["selection"]["eligible"], flags, "teacher eligibility")
        if selected is None:
            require("repeat" not in attempt, "repeat without eligible winner")
            continue
        original, repeat = refs[selected], attempt["repeat"]
        require(original["verification_only"] is False and repeat["verification_only"] is True,
                "original/repeat identity mismatch")
        root = base/f"{selected}_{k:04d}"
        repeat_root = base/f"{selected}_{k:04d}_repeat"
        same(pins.doc(root/"branch.json"), original, "persisted original branch")
        same(pins.doc(repeat_root/"branch.json"), repeat, "persisted repeat branch")
        wins = pins.doc(root/"windows.json")
        same(attempt["repeat_valid"], eligible(repeat), "repeat validity")
        same(attempt["selected_candidate_windows"], len(wins), "window count")
        if eligible(repeat) and wins:
            require(i == len(attempts)-1, "extra attempts after accepted winner")
            return dict(teacher=selected, takeover_step=k, branch=str(root),
                        repeat_branch=str(repeat_root),
                        window_indices=[w["current_index"] for w in wins],
                        candidate_windows=len(wins))
    require(len(attempts) == len(ks), "premature exhaustion")
    return None


def build_release(new_audit, new_audit_sha, new_numeric, new_numeric_sha):
    pins = Pins()
    pins._read(Path(__file__).resolve())
    plan = pins.doc(PLAN, PLAN_SHA)
    overlay = pins.doc(OVERLAY, OVERLAY_SHA)
    same(overlay["base_plan"], dict(path=str(PLAN), sha256=PLAN_SHA), "overlay plan")
    exact_partition(plan["entries"], overlay["reused_keys"], overlay["remaining_lanes"])
    require(plan["evaluation_adaptation"] is True and plan["untouched_test"] is False,
            "explicit in-set adaptation required")
    old = pins.doc(OLD_AUDIT, OLD_AUDIT_SHA)
    new = pins.doc(new_audit, new_audit_sha)
    oldnum = pins.doc(OLD_NUMERIC, OLD_NUMERIC_SHA)
    newnum = pins.doc(new_numeric, new_numeric_sha)
    reports = ((old, OLD_AUDIT, OLD_AUDIT_SHA, 36, oldnum),
               (new, Path(new_audit), new_audit_sha, 90, newnum))
    numeric = {}
    for report, path, digest, size, numbers in reports:
        require(report["training_released"] is False and report["score_backfill_allowed"] is False,
                "raw audit must not itself release training")
        require(len(report["searches"]) == size, "raw audit incomplete")
        same(report["base_plan"], dict(path=str(PLAN), sha256=PLAN_SHA), "raw audit plan")
        provenance = pins.doc(path.with_suffix(".provenance.json"))
        validate_provenance(report, provenance, size=size, path=path, digest=digest, pins=pins)
        require(type(provenance.get("report_bytes")) is int and
                provenance["report_bytes"] == pins.stats[str(path)][0], "raw report byte count")
        episodes = validate_numeric_report(numbers, size)
        pins.verify_all(numbers["source_files"])
        pins.verify_all(numbers["code_sha256_before"])
        for ep in episodes:
            require(ep["key"] not in numeric, "numeric duplicate key")
            numeric[ep["key"]] = ep
    same(newnum["old36_reference"]["summary"], oldnum["summary"], "numeric old summary reference")
    same(sorted({e["key"] for e in newnum["episodes"]} | {e["key"] for e in newnum["no_candidate"]}),
         sorted(e["key"] for e in overlay["remaining_entries"]), "numeric exact new coverage")
    same(new["continuation"], dict(path=str(OVERLAY), sha256=OVERLAY_SHA), "new continuation")
    same(new["previous_audit"], dict(path=str(OLD_AUDIT), sha256=OLD_AUDIT_SHA), "old audit binding")
    complete = pins.doc(ROOT/"COMPLETE.json")
    require(complete["status"] == "COLLECTION_PROCESSES_COMPLETE_NOT_TRAINING_RELEASE"
            and complete["entries"] == 90 and complete["lanes"] == 8
            and complete["training_released"] is False, "terminal90 marker")
    same(complete["plan_sha256"], PLAN_SHA, "terminal plan")
    same(complete["continuation_sha256"], OVERLAY_SHA, "terminal continuation")
    validate_marker(complete, [e["key"] for e in overlay["remaining_entries"]])
    require(not (ROOT/"ERROR.json").exists(), "root error marker")
    bykey = {s["row"]["key"]: s for s in old["searches"]+new["searches"]}
    require(len(bykey) == 126, "duplicate or incomplete raw searches")
    frozen = {r["key"]: r for r in overlay["frozen_records"]}
    require(set(frozen) == set(overlay["reused_keys"]), "frozen scope changed")
    for key, row in frozen.items():
        item = bykey[key]
        same(item["row"], row["row"], "frozen original record")
        for k in ("records_path", "records_sha256"):
            same(item[k], row[k], "frozen original source")
        same(item["job_id"], row["source_job_id"], "frozen source job")
        same(item["task_id"], row["source_task_id"], "frozen source task")
    for lane, entries in enumerate(overlay["remaining_lanes"]):
        same(entries, [e for e in plan["lanes"][lane] if e["key"] not in frozen],
             "original lane assignment")
        for name in ("ERROR.json", "cleanup_errors.json", "collection/ERROR.json"):
            require(not (ROOT/f"lane{lane}"/name).exists(), "lane error marker")
        path = ROOT/f"lane{lane}/collection/records.jsonl"
        blob = pins.blob(path)
        rows = [strict_json(line) for line in blob.splitlines() if line.strip()]
        keys = [e["key"] for e in entries]
        same([r["key"] for r in rows], keys, "original lane order")
        marker = pins.doc(path.parent/"COMPLETE.json")
        same(marker["keys"], keys, "lane completion keys")
        validate_marker(marker, keys, lane=lane)
        require(marker["entries"] == marker["expected"] == len(keys) and
                marker["training_released"] is False, "lane completion size")
        for row in rows:
            validate_protocol(row, new=True)
            item = bykey[row["key"]]
            same(item["row"], row, "new audited record")
            same((item["job_id"], item["task_id"]), (61844, 73066), "new source job/task")
            same(item["records_path"], str(path), "new source path")
            same(item["records_sha256"], pins.files[str(path)], "new records SHA")
    demonstrations, outcomes = [], Counter()
    for entry in plan["entries"]:
        item = bykey[entry["key"]]
        row, audit = item["row"], item["audit"]
        validate_protocol(row, new=entry["key"] not in frozen)
        same((row["task"], row["key"]), (entry["task"], entry["key"]), "plan row identity")
        same(row["baseline_result_unchanged"], entry["baseline_result"], "baseline result")
        base = Path(item["records_path"]).parent/"stt"/entry["key"]
        pins.blob(item["records_path"], item["records_sha256"])
        same(pins.doc(base/"search.json"), row, "persisted search")
        for k in ("task", "key", "experiment", "protocol_sha256", "outcome"):
            same(audit[k], row[k], "search audit identity")
        require(row["training_released"] is False and row["score_backfill_allowed"] is False
                and audit["training_released"] is False, "premature release")
        pins.verify_all(audit["source_files"])
        chosen = choose_original(row, pins, base)
        candidate = audit["candidate"]
        if chosen is None:
            require(candidate is None and row["accepted"] is None, "failed branch admitted")
        else:
            require(isinstance(candidate, dict), "missing audited winner")
            for k, v in chosen.items():
                same(candidate[k], v, "independent original winner")
            same(row["accepted"]["artifact_root"], chosen["branch"], "original branch only")
            require(candidate["training_released"] is False and candidate["conversion_pending"] is True,
                    "candidate unexpectedly released")
            indices = chosen["window_indices"]
            require(indices == sorted(set(indices)) and indices[0] >= chosen["takeover_step"],
                    "duplicate or preteacher windows")
            numeric_admission = exact_numeric(candidate, numeric.get(entry["key"]),
                row, pins, new=entry["key"] not in frozen)
            branch = Path(chosen["branch"])
            hashes = {str(Path(p).relative_to(branch)): h for p, h in audit["source_files"].items()
                      if Path(p).is_relative_to(branch)}
            require({"metadata.json", "observations.json", "actions.json", "windows.json",
                     "result.json", "branch.json"} <= set(hashes), "missing selected evidence")
            demonstrations.append(dict(**chosen, task="stt", key=entry["key"],
                source_experiment=row["experiment"], source_protocol_sha256=row["protocol_sha256"],
                source_job_id=item["job_id"], source_task_id=item["task_id"],
                **({"source_boundary_policy": CONTINUATION_POLICY} if entry["key"] not in frozen else {}),
                **numeric_admission, hashes=hashes))
        outcomes[row["outcome"]] += 1
    require(set(numeric) == {d["key"] for d in demonstrations}, "extra/missing numeric candidates")
    pins.finish()
    return dict(schema=SCHEMA, collection_validated=True, expected=126, completed_searches=126,
        evaluation_adaptation=True, untouched_test=False, no_success_rate=True,
        score_backfill_allowed=False, training_released=False, cache_conversion_required=True,
        base_plan=dict(path=str(PLAN), sha256=PLAN_SHA),
        continuation=dict(path=str(OVERLAY), sha256=OVERLAY_SHA),
        teacher_demonstrations=demonstrations, outcomes=dict(outcomes),
        summary=dict(accepted_original_episodes=len(demonstrations),
                     candidate_windows=sum(d["candidate_windows"] for d in demonstrations),
                     valid_windows=sum(d["expected_valid_count"] for d in demonstrations),
                     excluded_windows=sum(d["expected_excluded_count"] for d in demonstrations),
                     episodes_with_valid_windows=sum(d["expected_valid_count"] > 0 for d in demonstrations)),
        source_files=pins.files,
        limitations=["Teacher collection is not a new student SR",
                     "Full raw audits decode branch PNGs, not hidden RNG/contact state",
                     "Numeric labels/cache/loader must pass before any training",
                     "Original winner only; repeat and failed student future labels excluded"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("new-audit", "new-audit-sha", "new-numeric", "new-numeric-sha", "output"):
        p.add_argument("--"+name, required=True)
    a = p.parse_args()
    output = Path(a.output)
    require(output.is_absolute() and output.is_relative_to(R/"artifacts")
            and not output.exists() and not output.is_symlink(), "new NAS artifact required")
    report = build_release(a.new_audit, a.new_audit_sha, a.new_numeric, a.new_numeric_sha)
    payload = json.dumps(report, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    with output.open("xb") as stream:
        stream.write(payload)
    print(json.dumps(dict(path=str(output), sha256=hashlib.sha256(payload).hexdigest(),
                          summary=report["summary"], training_released=False)), flush=True)


if __name__ == "__main__":
    main()
