"""CPU-only evidence for explicitly classified teacher branch errors.

No exception is automatically classified or caught here. The caller must keep
baseline/replay/identity failures fatal and invoke this module only at a reviewed
teacher-output/transport boundary. Error branches never contain terminal scores.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = "failure_state_teacher_error_evidence_v1"
EXPERIMENT = "evaluation_adaptation_failure_state_v2"
STATUSES = ("teacher_output_invalid", "teacher_transport_error")
PAIR_FIELDS = ("experiment", "task", "key", "takeover_step", "seed",
               "protocol_sha256", "initial_rgb_sha256",
               "takeover_state_sha256", "prefix_sha256")


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing " + label)
    return value


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("invalid " + label)
    return value


def _sha(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("invalid SHA256: " + label)
    return value


class TeacherOutputInvalid(Exception):
    """Explicit decoder/server-response error; never a general RuntimeError."""
    status = STATUSES[0]

    def __init__(self, *, rc, seq, message):
        self.evidence = dict(source="server_response",
                             rc=_integer(rc, "response rc"),
                             seq=_integer(seq, "response seq"),
                             message=_text(message, "response message"),
                             exception_type=None)
        super().__init__(message)


class TeacherTransportError(Exception):
    """Explicitly identified transport error; no inferred client exception types."""
    status = STATUSES[1]

    def __init__(self, *, exception_type, message, rc=None, seq=None):
        self.evidence = dict(
            source="transport_exception",
            rc=None if rc is None else _integer(rc, "transport rc"),
            seq=None if seq is None else _integer(seq, "transport seq"),
            message=_text(message, "transport message"),
            exception_type=_text(exception_type, "transport exception type"))
        super().__init__(message)


def _error(status, value):
    if status not in STATUSES or not isinstance(value, dict):
        raise ValueError("unknown teacher error status or missing evidence")
    if set(value) != {"source", "rc", "seq", "message", "exception_type"}:
        raise ValueError("wrong raw error fields")
    _text(value["message"], "raw error message")
    if status == "teacher_output_invalid":
        if value["source"] != "server_response" or value["exception_type"] is not None:
            raise ValueError("output error requires explicit server-response evidence")
        _integer(value["rc"], "response rc")
        _integer(value["seq"], "response seq")
    else:
        if value["source"] != "transport_exception":
            raise ValueError("transport error provenance differs")
        _text(value["exception_type"], "transport exception type")
        for f in ("rc", "seq"):
            if value[f] is not None:
                _integer(value[f], f)
    return copy.deepcopy(value)


def _load(path):
    path = Path(path)
    if not path.is_absolute() or not path.is_file():
        raise ValueError("evidence must be an existing absolute file")
    blob = path.read_bytes()

    def no_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result

    value = json.loads(blob, parse_constant=no_constant, object_pairs_hook=unique_pairs)
    return value, dict(path=str(path), bytes=len(blob),
                       sha256=hashlib.sha256(blob).hexdigest())


def _finite(value):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("nonfinite or nonnumeric state/action")
    return float(value)


def _state_delta(expected, actual):
    """Check all recorded numeric state fields; mismatches remain fatal ValueError."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise ValueError("replay state structure differs")
        return max((_state_delta(expected[k], actual[k]) for k in expected), default=0.0)
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError("replay state shape differs")
        return max((_state_delta(x, y) for x, y in zip(expected, actual)), default=0.0)
    return abs(_finite(expected) - _finite(actual))


def _state(value):
    if not isinstance(value, dict) or set(value) != {"timestamp", "agents"}:
        raise ValueError("missing complete recorded dynamic state")
    _finite(value["timestamp"])
    if not isinstance(value["agents"], list) or not value["agents"]:
        raise ValueError("missing articulated agents")
    for agent in value["agents"]:
        if not isinstance(agent, dict) or set(agent) != {"transform", "joints"}:
            raise ValueError("missing transform or joints")
        transform = agent["transform"]
        if (not isinstance(transform, list) or len(transform) != 4
                or any(not isinstance(row, list) or len(row) != 4 for row in transform)
                or not isinstance(agent["joints"], list)):
            raise ValueError("invalid recorded state shape")
    _state_delta(value, value)
    return value


def _action(value):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("expected three-component action")
    if any(abs(_finite(x)) > 1 for x in value):
        raise ValueError("action outside normalized limits")
    return value


def _frame(frame, i):
    if not isinstance(frame, dict):
        raise ValueError("missing recorded frame")
    if type(frame.get("step")) is not int or frame["step"] != i:
        raise ValueError("nonconsecutive recorded step")
    if type(frame.get("observation_index")) is not int or frame["observation_index"] != i:
        raise ValueError("nonconsecutive observation index")
    _sha(frame.get("rgb_sha256"), "recorded RGB")
    state = _state(frame.get("dynamic_state"))
    if frame.get("dynamic_state_sha256") != canonical_sha(state):
        raise ValueError("recorded state SHA differs")
    for field in ("timestamp_s", "simulator_world_time_s"):
        if abs(_finite(frame.get(field)) - state["timestamp"]) > 1e-6:
            raise ValueError("recorded time differs from dynamic state")


def _pair(value, prefix):
    if not isinstance(value, dict) or set(value) != set(PAIR_FIELDS):
        raise ValueError("expected_pair fields differ")
    if value["experiment"] != EXPERIMENT or value["task"] != "stt":
        raise ValueError("wrong failure-state experiment/task")
    if (not isinstance(value["key"], str)
            or re.fullmatch(r"[A-Za-z0-9]+/[0-9]+", value["key"]) is None):
        raise ValueError("wrong episode key")
    if type(value["seed"]) is not int or value["seed"] != 7:
        raise ValueError("seed must remain 7")
    k = _integer(value["takeover_step"], "takeover step")
    for f in ("protocol_sha256", "initial_rgb_sha256", "takeover_state_sha256", "prefix_sha256"):
        _sha(value[f], f)
    if not isinstance(prefix, list) or not prefix or k >= len(prefix):
        raise ValueError("expected takeover outside actual student prefix")
    if canonical_sha(prefix) != value["prefix_sha256"]:
        raise ValueError("expected student prefix SHA differs")
    for i, frame in enumerate(prefix):
        _frame(frame, i)
        _action(frame.get("action"))
    if prefix[0]["rgb_sha256"] != value["initial_rgb_sha256"]:
        raise ValueError("expected initial RGB differs from student prefix")
    if canonical_sha(prefix[k]["dynamic_state"]) != value["takeover_state_sha256"]:
        raise ValueError("expected takeover state differs from student prefix")
    return k


def _construct(*, status, raw_error, teacher, expected_pair, prefix,
               partial_replay_path, partial_actions_path, verification_only,
               fallback_detected, fallback_executed):
    if teacher not in ("lightnav", "oracle"):
        raise ValueError("error branch must name an actual teacher")
    if any(type(x) is not bool for x in
           (verification_only, fallback_detected, fallback_executed)):
        raise ValueError("explicit boolean error flags required")
    if fallback_executed:
        raise ValueError("executed fallback cannot use nonexecuted-error evidence")
    raw_error = _error(status, raw_error)
    k = _pair(expected_pair, prefix)
    replay, replay_file = _load(partial_replay_path)
    actions, action_file = _load(partial_actions_path)
    if replay_file["path"] == action_file["path"]:
        raise ValueError("separate replay/action evidence files required")
    if not isinstance(replay, list) or not isinstance(actions, list):
        raise ValueError("partial evidence must be arrays")
    # During act(), observe has happened but failed action cannot be recorded.
    # Empty arrays cover explicit transport failure before first observation.
    if not ((not replay and not actions) or len(replay) == len(actions) + 1):
        raise ValueError("partial evidence must end before the failed action")
    maximum_delta = 0.0
    for i, frame in enumerate(replay):
        _frame(frame, i)
        if i <= k:
            if frame["rgb_sha256"] != prefix[i]["rgb_sha256"]:
                raise ValueError("REPLAY_RGB_DIVERGED")
            delta = _state_delta(prefix[i]["dynamic_state"], frame["dynamic_state"])
            if delta > 1e-6:
                raise ValueError("REPLAY_DYNAMIC_STATE_DIVERGED")
            maximum_delta = max(maximum_delta, delta)
        if i < len(actions):
            a = actions[i]
            if not isinstance(a, dict) or type(a.get("sim_step")) is not int or a["sim_step"] != i:
                raise ValueError("nonconsecutive partial action")
            cmd = _action(a.get("normalized_action"))
            if frame.get("action") != cmd or frame.get("normalized_action") != cmd:
                raise ValueError("replay/action command disagreement")
            owner = "student" if i < k else "teacher"
            if frame.get("owner") != owner or a.get("owner") != owner:
                raise ValueError("action ownership differs")
            if a.get("teacher") != (None if i < k else teacher):
                raise ValueError("teacher action identity differs")
            if frame.get("action_timing") != "before_env_step" or frame.get("post_state_recorded") is not False:
                raise ValueError("action timing contract differs")
            if i < k and cmd != prefix[i]["action"]:
                raise ValueError("REPLAY_PREFIX_ACTION_DIVERGED")
        elif any(f in frame for f in ("action", "normalized_action", "owner")):
            raise ValueError("failed action must not have a recorded command")
    m, n = len(replay), len(actions)
    last = m - 1 if m else None
    phase = ("before_first_observation" if last is None else
             "before_takeover" if last < k else
             "at_takeover" if last == k else "after_takeover")
    # Derived exclusively from an actual hashed partial frame, never expected_pair.
    takeover = None if m <= k else dict(
        step=k, observation_index=k, rgb_sha256=replay[k]["rgb_sha256"],
        dynamic_state_sha256=canonical_sha(replay[k]["dynamic_state"]),
        source_replay_sha256=replay_file["sha256"])
    observed = dict(
        observations=m, recorded_actions=n, failed_action_step=n if m else None,
        last_observation_step=last, failure_phase=phase,
        verified_prefix_frames=min(m, k + 1), required_prefix_frames=k + 1,
        prefix_validation_complete=m >= k + 1,
        maximum_prefix_state_difference=maximum_delta,
        initial_rgb_sha256=None if not m else replay[0]["rgb_sha256"],
        initial_state_sha256=None if not m else canonical_sha(replay[0]["dynamic_state"]),
        actual_takeover=takeover,
        actual_takeover_state_sha256=None if takeover is None else takeover["dynamic_state_sha256"],
        hidden_rng_contact_state_proven=False,
        evidence_files=dict(partial_replay=replay_file, partial_actions=action_file))
    return dict(
        schema=SCHEMA, experiment=EXPERIMENT, partition="evaluation_adaptation",
        status=status, teacher=teacher, verification_only=verification_only,
        expected_pair=copy.deepcopy(expected_pair), observed_partial=observed,
        raw_error=raw_error, fallback_detected=fallback_detected,
        fallback_executed=False, result=None, complete=False, replay_verified=False,
        candidate_windows=0, training_windows=0, windows=[],
        training_eligible=False, training_released=False,
        no_terminal_result=True, no_success_rate=True,
        ineligible_reason=status, pair_comparability="partial",
        proof_scope="observed raw RGB and time/transforms/joints only; no terminal or full branch proof")


def build_teacher_error(*, error, teacher, expected_pair, prefix,
                        partial_replay_path, partial_actions_path,
                        fallback_detected, verification_only=False, fallback_executed=False):
    """Read existing evidence files and return a JSON-ready dict; never writes.

    Caller explicitly supplies one of the two narrow exception instances. No
    automatic conversion of ValueError/RuntimeError or client exception occurs.
    """
    if type(error) not in (TeacherOutputInvalid, TeacherTransportError):
        raise TypeError("explicit classified teacher exception required")
    return _construct(status=error.status, raw_error=error.evidence,
                      teacher=teacher, expected_pair=expected_pair, prefix=prefix,
                      partial_replay_path=partial_replay_path,
                      partial_actions_path=partial_actions_path,
                      verification_only=verification_only,
                      fallback_detected=fallback_detected, fallback_executed=fallback_executed)


def validate_teacher_error(document, *, prefix):
    """Re-read bound partial files and independently derive every evidence field.

    Exact reconstruction rejects unknown fields, fake terminal/TR data, forged
    takeover evidence, incorrect counts and altered file hashes. Returns a copy.
    """
    if not isinstance(document, dict):
        raise ValueError("error evidence must be a document")
    try:
        files = document["observed_partial"]["evidence_files"]
        rebuilt = _construct(
            status=document["status"], raw_error=document["raw_error"],
            teacher=document["teacher"], expected_pair=document["expected_pair"],
            prefix=prefix, partial_replay_path=files["partial_replay"]["path"],
            partial_actions_path=files["partial_actions"]["path"],
            verification_only=document["verification_only"],
            fallback_detected=document["fallback_detected"],
            fallback_executed=document["fallback_executed"])
    except (KeyError, TypeError) as exc:
        raise ValueError("missing or malformed teacher-error evidence") from exc
    if canonical_sha(document) != canonical_sha(rebuilt):
        raise ValueError("teacher-error document differs from observed evidence")
    return rebuilt
