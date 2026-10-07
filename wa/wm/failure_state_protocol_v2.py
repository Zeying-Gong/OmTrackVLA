"""CPU-only v2 selection/search; does not relabel or load the frozen v1 plan.

run(teacher, k, verification_only) must return either a normal persisted branch
or validated teacher-error evidence. It must not convert unknown, baseline or
replay failures. This module catches no run exceptions.
"""
import copy
from pathlib import Path

from wa.wm import failure_state_protocol as v1
from wa.wm.failure_state_collect import repeat_valid as _v1_repeat_valid
from wa.wm.failure_state_teacher_error import (
    EXPERIMENT, PAIR_FIELDS, STATUSES, canonical_sha, validate_teacher_error,
)

PROTOCOL = copy.deepcopy(v1.PROTOCOL)
PROTOCOL.update(
    experiment=EXPERIMENT,
    base_protocol_sha=v1.PROTOCOL_SHA,
    teacher_runtime_invalid=dict(
        statuses=list(STATUSES),
        evidence_schema="failure_state_teacher_error_evidence_v1",
        terminal_result=None,
        rule="invalid teacher branch is ineligible; independently run the other teacher at the same fixed k",
        no_retry="no additional same-k teacher attempts beyond the one selected-winner verification",
        expected_vs_observed="expected pair identity is separate; actual takeover only from a verified observed frame",
        partial_proof="never claim complete replay proof for a runtime-invalid branch",
        fallback="detected fallback is not executed, not labeled and not admitted",
        unknown="unclassified exceptions, baseline/identity/hash or replay divergence remain fatal",
        repeat_error="validated same-identity classified repeat error continues to the next earlier fixed k"),
    pair_comparability="full only when both branches have complete fallback-free replay proof; otherwise partial",
    selected_zero_windows="continue earlier k after the existing single winner verification; never switch same-k runner-up",
    old_results="v1 failed collection remains separate; no v1 plan identity substitution",
)
PROTOCOL_SHA = canonical_sha(PROTOCOL)
BASE_PROTOCOL_SHA = v1.PROTOCOL_SHA
candidate_steps = v1.candidate_steps


def _expected_pair(value, prefix):
    if not isinstance(value, dict) or set(value) != set(PAIR_FIELDS):
        raise ValueError("missing or additional expected pair fields")
    if value["experiment"] != EXPERIMENT:
        raise ValueError("foreign v2 experiment")
    v1._key(value["task"], value["key"])
    k = v1._integer(value["takeover_step"], "takeover step")
    if type(value["seed"]) is not int or value["seed"] != 7:
        raise ValueError("seed must remain 7")
    for field in ("protocol_sha256", "initial_rgb_sha256", "takeover_state_sha256", "prefix_sha256"):
        v1._sha(value[field], field)
    if value["protocol_sha256"] != PROTOCOL_SHA:
        raise ValueError("foreign v2 protocol SHA")
    if not isinstance(prefix, list) or not prefix or k >= len(prefix):
        raise ValueError("takeover outside actual student prefix")
    if canonical_sha(prefix) != value["prefix_sha256"]:
        raise ValueError("actual prefix SHA differs")
    if prefix[0].get("rgb_sha256") != value["initial_rgb_sha256"]:
        raise ValueError("initial RGB identity differs")
    if canonical_sha(prefix[k].get("dynamic_state")) != value["takeover_state_sha256"]:
        raise ValueError("takeover state identity differs")
    return copy.deepcopy(value)


def _artifact_root(branch, error):
    if error:
        files = branch["observed_partial"]["evidence_files"]
        a, b = (Path(files[n]["path"]).parent for n in ("partial_replay", "partial_actions"))
        if a != b:
            raise ValueError("error partial files must share an independent branch directory")
        return str(a)
    root = branch.get("artifact_root")
    if not isinstance(root, str) or not root or not Path(root).is_absolute():
        raise ValueError("normal branch requires absolute artifact_root")
    return str(Path(root))


def _inspect(branch, name, prefix, *, verification_only):
    if not isinstance(branch, dict) or branch.get("teacher") != name:
        raise ValueError("wrong branch teacher identity")
    if type(branch.get("verification_only")) is not bool or branch["verification_only"] != verification_only:
        raise ValueError("wrong candidate/verification branch role")
    if "status" in branch:
        if branch["status"] not in STATUSES:
            raise ValueError("unknown top-level branch status")
        validate_teacher_error(branch, prefix=prefix)
        pair = _expected_pair(branch["expected_pair"], prefix)
        return dict(error=True, pair=pair, eligible=False,
                    reasons=[branch["status"]], result=None, full=False,
                    artifact_root=_artifact_root(branch, True))
    if branch.get("experiment") != EXPERIMENT:
        raise ValueError("foreign normal branch experiment")
    try:
        pair = _expected_pair({f: branch[f] for f in PAIR_FIELDS}, prefix)
    except KeyError as exc:
        raise ValueError("missing normal pairing evidence") from exc
    for field in ("complete", "replay_verified", "transport_fallback"):
        v1._strict_bool(branch.get(field), field)
    flags, _ = v1._result(branch.get("result"))
    checks = dict(complete=branch["complete"], replay_verified=branch["replay_verified"],
                  success=flags["success"], policy_init_valid=flags["policy_init_valid"],
                  collision_free=not flags["collision"],
                  transport_fallback_free=not branch["transport_fallback"])
    reasons = [f for f, valid in checks.items() if not valid]
    if branch.get("fallback_executed", False) is not False:
        raise ValueError("executed fallback forbidden in v2")
    return dict(error=False, pair=pair, eligible=not reasons, reasons=reasons,
                result=copy.deepcopy(branch["result"]),
                full=branch["complete"] and branch["replay_verified"] and not branch["transport_fallback"],
                artifact_root=_artifact_root(branch, False))


def select_recovery_teacher(lightnav, oracle, *, prefix):
    """Select at one fixed expected k; runtime-invalid result remains None."""
    branches = dict(lightnav=lightnav, oracle=oracle)
    evidence = {n: _inspect(b, n, prefix, verification_only=False)
                for n, b in branches.items()}
    if evidence["lightnav"]["pair"] != evidence["oracle"]["pair"]:
        raise ValueError("branches differ in expected key/k/prefix/seed/protocol")
    valid = [n for n, e in evidence.items() if e["eligible"]]
    selected = None
    if not valid:
        reason = "no_eligible_teacher_at_this_k"
    elif len(valid) == 1:
        selected, reason = valid[0], "only_eligible_teacher"
    else:
        ln, oc = (evidence[n]["result"]["following_rate"] for n in ("lightnav", "oracle"))
        selected = "oracle" if oc > ln else "lightnav"
        reason = "higher_following_rate" if ln != oc else "equal_rate_fixed_lightnav_tie"
    return dict(
        experiment=EXPERIMENT, protocol_sha256=PROTOCOL_SHA,
        base_protocol_sha=BASE_PROTOCOL_SHA,
        pair=evidence["lightnav"]["pair"],
        pair_comparability="full" if all(e["full"] for e in evidence.values()) else "partial",
        selected_teacher=selected, reason=reason,
        eligible={n: e["eligible"] for n, e in evidence.items()},
        ineligible_reasons={n: e["reasons"] for n, e in evidence.items()},
        results={n: e["result"] for n, e in evidence.items()},
        runtime_invalid={n: e["error"] for n, e in evidence.items()},
        demonstration_candidate=selected is not None,
        verification_required=selected is not None,
        training_released=False,
        pending="additional same-k winner replay and teacher-owned suffix admission")


def repeat_valid(original, repeat, *, prefix):
    """Classified repeat errors return False only after independent identity checks."""
    if not isinstance(original, dict):
        raise ValueError("missing selected original branch")
    name = original.get("teacher")
    if name not in ("lightnav", "oracle"):
        raise ValueError("unknown original teacher")
    first = _inspect(original, name, prefix, verification_only=False)
    second = _inspect(repeat, name, prefix, verification_only=True)
    if first["error"] or not first["eligible"]:
        raise ValueError("only an eligible normal candidate may be verified")
    if first["pair"] != second["pair"]:
        raise ValueError("repeat expected identity differs")
    if first["artifact_root"] == second["artifact_root"]:
        raise ValueError("repeat must have independent artifact root")
    if second["error"]:
        return False
    # Reuse unchanged v1 strict repeat flags/TR/identity semantics for normal runs.
    return _v1_repeat_valid(original, repeat)


def search_recovery(student, prefix, run):
    """Finite search over the unchanged backoffs; no exception suppression or retries."""
    if not isinstance(student, dict):
        raise ValueError("student result required")
    result = student.get("result")
    flags, _ = v1._result(result)
    if type(result.get("policy_init_valid")) is not bool:
        raise ValueError("student init validity must remain an explicit bool")
    report = dict(
        experiment=EXPERIMENT, protocol_sha256=PROTOCOL_SHA,
        base_protocol_sha=BASE_PROTOCOL_SHA, student=student,
        attempts=[], accepted=None, training_released=False,
        score_backfill_allowed=False)
    if not flags["policy_init_valid"]:
        report["outcome"] = "student_invalid_initialization"
        return report
    if flags["success"] and not flags["collision"]:
        report["outcome"] = "rerun_student_success_no_recovery_needed"
        return report
    if not isinstance(prefix, list):
        raise ValueError("actual student prefix required")
    identity = None
    for k in candidate_steps(len(prefix)):
        # Intentional independent calls, but unknown run exceptions still propagate.
        branches = {name: run(name, k, False) for name in ("lightnav", "oracle")}
        selection = select_recovery_teacher(branches["lightnav"], branches["oracle"], prefix=prefix)
        pair = selection["pair"]
        if pair["takeover_step"] != k:
            raise ValueError("run returned a branch for an unrequested k")
        current_identity = {f: v for f, v in pair.items()
                            if f not in ("takeover_step", "takeover_state_sha256")}
        if identity is not None and current_identity != identity:
            raise ValueError("episode identity changed across fixed candidates")
        identity = current_identity
        attempt = dict(takeover_step=k, branches=branches, selection=selection)
        report["attempts"].append(attempt)
        selected = selection["selected_teacher"]
        if selected is None:
            continue
        repeat = run(selected, k, True)
        attempt["repeat"] = repeat
        attempt["repeat_valid"] = repeat_valid(branches[selected], repeat, prefix=prefix)
        windows = branches[selected].get("candidate_windows")
        if type(windows) is not int or windows < 0:
            raise ValueError("explicit nonnegative candidate window count required")
        attempt["selected_candidate_windows"] = windows
        if not attempt["repeat_valid"]:
            attempt["admission_reason"] = "selected_teacher_repeat_invalid"
        elif windows == 0:
            attempt["admission_reason"] = "selected_teacher_no_candidate_windows"
        else:
            report["accepted"] = dict(
                teacher=selected, takeover_step=k,
                artifact_root=branches[selected]["artifact_root"],
                repeat_artifact_root=repeat["artifact_root"],
                selection=selection, prefix_sha256=pair["prefix_sha256"],
                candidate_only=True, training_released=False)
            report["outcome"] = "repeated_teacher_recovery_candidate"
            return report
    report["outcome"] = "no_valid_teacher_recovery"
    return report
