"""Independent audit of a persisted failure-state search, never a training release.

Re-derive the fixed candidate sequence, same-k winner and repeat admission from
audited files. Collector summaries alone cannot admit a demonstration.
"""
import hashlib
import json
import re
from pathlib import Path

from wa.wm.failure_state_protocol import (
    EXPERIMENT as V1, PROTOCOL_SHA as V1_SHA, binary_flag, canonical_sha,
)
from wa.wm.failure_state_protocol_v2 import EXPERIMENT as V2, PROTOCOL_SHA as V2_SHA

SCHEMA = "failure_state_search_audit_v1"
PAIR = ("experiment", "task", "key", "takeover_step", "seed",
        "protocol_sha256", "initial_rgb_sha256",
        "takeover_state_sha256", "prefix_sha256")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "missing/symlink file: " + str(path))
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    def invalid(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)


def same(actual, expected, name):
    require(canonical_sha(actual) == canonical_sha(expected), "mismatch: " + name)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def eligible(branch):
    if branch.get("status") in ("teacher_output_invalid", "teacher_transport_error"):
        require(branch.get("result") is None and branch.get("complete") is False,
                "invalid teacher must not have a terminal result")
        return False
    r = branch.get("result")
    require(isinstance(r, dict), "missing terminal result")
    flags = {k: binary_flag(r.get(k), k)
             for k in ("success", "collision", "policy_init_valid")}
    rate = r.get("following_rate")
    require(type(rate) in (int, float) and 0 <= rate <= 1, "invalid following_rate")
    require(not flags["success"] or flags["policy_init_valid"] and not flags["collision"],
            "inconsistent success")
    for k in ("complete", "replay_verified", "transport_fallback"):
        require(type(branch.get(k)) is bool, "missing branch flag: " + k)
    return (branch["complete"] and branch["replay_verified"] and
            not branch["transport_fallback"] and flags["success"] and
            flags["policy_init_valid"] and not flags["collision"])


def expected_candidates(n):
    require(type(n) is int and n > 0, "student action count must be positive")
    return sorted({0} | {max(0, n-b) for b in (5, 15, 30, 60)}, reverse=True)


def audit_search(row, entry, base, *, original_start, expected_metadata, audit_branch=None):
    """Audit all attempted branches; return only the original winning suffix.

    No file writes, no rollout, no collector callback, no copied repeat windows.
    This deliberately does not release a training cache or a partial collection.
    """
    if audit_branch is None:
        from wa.wm.failure_state_branch_audit import audit_branch
    base = Path(base)
    require(base.is_absolute() and base.is_dir() and not base.is_symlink(),
            "absolute independent episode directory required")
    require(row.get("experiment") in (V1, V2), "unknown search experiment")
    protocol = V1_SHA if row["experiment"] == V1 else V2_SHA
    same(row.get("protocol_sha256"), protocol, "search protocol")
    for k in ("task", "key"):
        same(row.get(k), entry.get(k), k)
    require(row["task"] == "stt", "STT failure recovery only")
    same(row.get("baseline_result_unchanged"), entry["baseline_result"], "baseline preserved")
    require(row.get("training_released") is False and row.get("score_backfill_allowed") is False,
            "premature release or score backfill")
    stored = read_json(base/"search.json")
    same(stored, row, "JSONL/search bytes meaning")
    pins = {str(base/"search.json"): sha(base/"search.json")}
    audited = {}
    exp = row["experiment"]
    common = dict(experiment=exp, protocol_sha256=protocol,
                  task=row["task"], key=row["key"])
    required = {"experiment", "partition", "protocol_sha256", "plan_sha256",
                "checkpoint_sha256", "checkpoint_step", "source_dataset_sha256", "seed"}
    if exp == V2:
        required |= {"base_plan_sha256", "base_protocol_sha256", "continuation_sha256"}
    require(isinstance(expected_metadata, dict) and required <= set(expected_metadata),
            "exact source metadata pins required")
    for key in required:
        value = expected_metadata[key]
        if key.endswith("sha256"):
            require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value),
                    "invalid source metadata SHA: " + key)
    same(expected_metadata["experiment"], exp, "metadata experiment")
    same(expected_metadata["protocol_sha256"], protocol, "metadata protocol")
    require(expected_metadata["partition"] == "evaluation_adaptation", "metadata partition")
    require(type(expected_metadata["seed"]) is int and expected_metadata["seed"] == 7,
            "metadata seed")
    require(type(expected_metadata["checkpoint_step"]) is int and
            expected_metadata["checkpoint_step"] > 0, "metadata checkpoint step")
    same(row.get("plan_sha256"), expected_metadata["plan_sha256"], "source plan")
    if exp == V2:
        same(expected_metadata["base_plan_sha256"], expected_metadata["plan_sha256"],
             "metadata base plan")
        same(expected_metadata["base_protocol_sha256"], V1_SHA, "metadata base protocol")
        for key in ("base_protocol_sha256", "continuation_sha256"):
            same(row.get(key), expected_metadata[key], "source " + key)
    student_images = None

    def inspect(label, teacher, k, repeat, ref, prefix=None):
        root = base/label
        expected = dict(common, teacher=teacher, takeover_step=k, verification_only=repeat,
                        metadata=dict(expected_metadata))
        if teacher != "student":
            require(isinstance(student_images, list) and len(student_images) > k,
                    "audited student PNG evidence required")
            expected["prefix_image_sha256"] = student_images
        got = audit_branch(root, expected=expected, student_prefix=prefix,
                           original_start=original_start)
        require(got.get("training_released") is False, "branch audit cannot release training")
        require(isinstance(got.get("source_files"), dict) and got["source_files"],
                "branch must supply actual file pins")
        for path, digest in got["source_files"].items():
            p = Path(path)
            require(p.is_absolute() and p.is_relative_to(root), "foreign branch evidence")
            require(path not in pins or pins[path] == digest, "conflicting evidence pin")
            pins[path] = digest
        branch = got["branch"]
        if teacher == "student":
            same(ref["result"], branch["result"], "student result")
            require(ref.get("artifact_root") == str(root) and
                    ref.get("initial_pair_verified") is True, "student original pairing absent")
            same(ref["first_start"], read_json(root/"first_start.json"), "student first start")
        else:
            same(branch, ref, "persisted branch/row")
        require(got.get("candidate_indices") == sorted(set(got["candidate_indices"])),
                "duplicate or unordered windows")
        require(type(got.get("candidate_windows")) is int and
                got["candidate_windows"] == len(got["candidate_indices"]),
                "candidate count differs from independently audited windows")
        if repeat or teacher == "student" or got["kind"] == "teacher_error":
            require(got["candidate_windows"] == 0, "ineligible branch has candidate labels")
        audited[str(root)] = got
        return got

    student = inspect("student", "student", None, False, row["student"])
    student_images = student.get("image_sha256")
    prefix = read_json(base/"student/replay.json")
    require(isinstance(prefix, list), "student replay must be a list")
    result = row["student"]["result"]
    require(type(result.get("policy_init_valid")) is bool, "explicit student initialization required")
    success = binary_flag(result.get("success"), "student success")
    collision = binary_flag(result.get("collision"), "student collision")
    require(not success or result["policy_init_valid"] and not collision, "invalid successful student")
    attempts = row.get("attempts")
    require(isinstance(attempts, list), "missing attempts")
    accepted = None
    if not result["policy_init_valid"]:
        outcome = "student_invalid_initialization"
        require(not attempts, "initialization failure cannot have teacher attempts")
    elif success and not collision:
        outcome = "rerun_student_success_no_recovery_needed"
        require(not attempts, "successful rerun must not generate recovery labels")
    else:
        candidates = expected_candidates(len(prefix))
        require(0 < len(attempts) <= len(candidates), "wrong fixed search length")
        for idx, attempt in enumerate(attempts):
            k = candidates[idx]
            same(attempt.get("takeover_step"), k, "fixed descending candidate order")
            refs = attempt.get("branches")
            require(isinstance(refs, dict) and set(refs) == {"lightnav", "oracle"},
                    "both independent teachers required at each k")
            evidence = {}
            for t in ("lightnav", "oracle"):
                evidence[t] = inspect(f"{t}_{k:04d}", t, k, False, refs[t], prefix)
            flags = {t: eligible(refs[t]) for t in refs}
            selected = None
            if flags["lightnav"] or flags["oracle"]:
                selected = ("oracle" if flags["oracle"] and
                            (not flags["lightnav"] or refs["oracle"]["result"]["following_rate"]
                             > refs["lightnav"]["result"]["following_rate"]) else "lightnav")
            selection = attempt.get("selection")
            require(isinstance(selection, dict), "missing stored selection")
            same(selection.get("selected_teacher"), selected, "success/TR/tie-LightNav selection")
            same(selection.get("eligible"), flags, "teacher eligibility")
            same(selection.get("results"), {t: refs[t].get("result") for t in refs}, "selection results")
            require(selection.get("training_released") is False, "collector selection is not release")
            same(selection.get("demonstration_candidate"), selected is not None, "candidate flag")
            same(selection.get("verification_required"), selected is not None, "repeat requirement")
            pair = dict(common, takeover_step=k, seed=7,
                        initial_rgb_sha256=prefix[0]["rgb_sha256"],
                        takeover_state_sha256=canonical_sha(prefix[k]["dynamic_state"]),
                        prefix_sha256=canonical_sha(prefix))
            # v1 pairing omits experiment; the v2 contract explicitly includes it.
            expected_pair = pair if exp == V2 else {a:b for a,b in pair.items() if a != "experiment"}
            same(selection.get("pair"), expected_pair, "selection pairing")
            if exp == V2:
                full = all(b.get("complete") is True and b.get("replay_verified") is True
                           and b.get("transport_fallback") is False for b in refs.values())
                same(selection.get("pair_comparability"), "full" if full else "partial",
                     "full versus partial teacher proof")
                same(selection.get("runtime_invalid"), {t:evidence[t]["kind"] == "teacher_error" for t in refs},
                     "invalid teacher classification")
            if selected is None:
                require("repeat" not in attempt and "repeat_valid" not in attempt,
                        "no retry when neither teacher is eligible")
                continue
            require("repeat" in attempt, "selected winner requires independent repeat")
            verify = inspect(f"{selected}_{k:04d}_repeat", selected, k, True, attempt["repeat"], prefix)
            repeat_ok = eligible(attempt["repeat"])
            same(attempt.get("repeat_valid"), repeat_ok, "repeat validity")
            count = evidence[selected]["candidate_windows"]
            same(attempt.get("selected_candidate_windows"), count, "selected raw window count")
            if repeat_ok and count:
                expected_accept = dict(teacher=selected, takeover_step=k,
                    artifact_root=str(base/f"{selected}_{k:04d}"),
                    repeat_artifact_root=str(base/f"{selected}_{k:04d}_repeat"),
                    selection=selection, prefix_sha256=pair["prefix_sha256"],
                    candidate_only=True, training_released=False)
                same(row.get("accepted"), expected_accept, "accepted original winner")
                require(idx == len(attempts)-1, "extra attempts after accepted recovery")
                accepted = dict(task=row["task"], key=row["key"], teacher=selected,
                    takeover_step=k, branch=expected_accept["artifact_root"],
                    repeat_branch=expected_accept["repeat_artifact_root"],
                    window_indices=evidence[selected]["candidate_indices"],
                    candidate_windows=count, experiment=exp, protocol_sha256=protocol,
                    source_files={p:h for p,h in pins.items()},
                    training_released=False, conversion_pending=True)
                break
        if accepted is None:
            require(len(attempts) == len(candidates), "search stopped before exhausting fixed candidates")
        outcome = "repeated_teacher_recovery_candidate" if accepted else "no_valid_teacher_recovery"
    same(row.get("outcome"), outcome, "search outcome")
    if accepted is None:
        require(row.get("accepted") is None, "failed search admitted a teacher")
    expected_dirs = {Path(p).name for p in audited}
    # Runtime creates named evaluator/client sidecars, not extra teacher trials.
    # Permit only names derived from branches already independently audited.
    allowed_sidecars = {name + "_metrics" for name in expected_dirs}
    allowed_sidecars |= {name + "_client" for name in expected_dirs
                         if name.startswith("lightnav_")}
    actual_dirs = set()
    sidecar_files = {}
    for path in base.iterdir():
        require(not path.is_symlink(), "symlink search member")
        if path.is_dir():
            actual_dirs.add(path.name)
            if path.name in allowed_sidecars:
                for child in path.rglob("*"):
                    require(not child.is_symlink(), "symlink sidecar member")
                    require(child.is_dir() or child.is_file(), "nonfile sidecar member")
                    if child.is_file():
                        sidecar_files[str(child)] = sha(child)
        else:
            require(path.is_file() and path.name in ("search.json", "student.trace.jsonl"),
                    "orphan/unreviewed search file")
            if path.name != "search.json":
                sidecar_files[str(path)] = sha(path)
    require(expected_dirs <= actual_dirs and
            actual_dirs <= expected_dirs | allowed_sidecars, "orphan/unreviewed branches")
    pins.update(sidecar_files)
    return dict(schema=SCHEMA, task=row["task"], key=row["key"], outcome=outcome,
                experiment=exp, protocol_sha256=protocol, candidate=accepted,
                source_files=pins, branch_count=len(audited),
                sidecar_files=sidecar_files, sidecar_media_semantics_audited=False,
                training_released=False, score_backfill_allowed=False)
