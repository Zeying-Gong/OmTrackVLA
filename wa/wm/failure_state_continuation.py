"""Frozen, one-key continuation of failed 61833/73055; no training release.

The original 126-key protocol is never rewritten. Only the exact previously
audited candidate can be reused. The caller supplies a freshly checked scheduler
status; file-based exit evidence alone is not interpreted as scheduler FAILED.
Every load rebuilds the overlay from the pinned source bytes, without writing.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from wa.wm.failure_state_protocol import (
    EXPERIMENT, PROTOCOL_SHA, binary_flag, canonical_sha, load_plan,
    select_recovery_teacher,
)

SCHEMA = "wa_failure_state_continuation_v1"
SOURCE_ROOT = Path("/data/nas_ray/project/md-ak/users/zeying.gong/job_61833/task_73055/wa_failure_state_collect_a800_v1")
SOURCE_COMMIT = "d44a06c6c4d9b14c9c8662468f2b043fb1fb5267"
BASE_SHA = "2eff9e83e89008ce1ee73632def406a27131fece6e77b01f6d4ca8c02624292b"
INVENTORY_SHA = "5d54cd4b63fa19ae4c4ecc8a804e83b6e54b60196ce90f177e4cae9bb6730a15"
INVENTORY_COUNT = 180
INVENTORY_BYTES = 33664365
KEY = "bzCsHPLDztK/8"
BRANCHES = ("student", "lightnav_0006", "oracle_0006", "oracle_0006_repeat")
REMAINING_LANE_COUNTS = [15, 16, 16, 16, 16, 16, 15, 15]
PAIR_FIELDS = ("task", "key", "takeover_step", "seed", "protocol_sha256",
               "initial_rgb_sha256", "takeover_state_sha256", "prefix_sha256")
JSON_NAMES = ("metadata", "observations", "actions", "replay", "first_start",
              "pair_start", "takeover", "fallback_events", "admission", "complete",
              "result", "windows", "raw_windows")


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _blob(path):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), "missing/symlink source: " + str(path))
    return path.read_bytes()


def _sha(path):
    return hashlib.sha256(_blob(path)).hexdigest()


def _document(blob):
    def bad(value):
        raise ValueError("nonfinite JSON: " + value)
    def pairs(values):
        result = {}
        for key, value in values:
            _require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(blob, parse_constant=bad, object_pairs_hook=pairs)


def _read(path):
    return _document(_blob(path))


def _lines(path):
    blob = _blob(path)
    _require(blob.endswith(b"\n"), "partial JSONL: " + str(path))
    rows = [_document(line) for line in blob.splitlines() if line.strip()]
    return rows


def _same(left, right, name):
    _require(canonical_sha(left) == canonical_sha(right), "mismatch: " + name)


def _false(mapping, key):
    _require(mapping.get(key) is False, "explicit false required: " + key)


def _true(mapping, key):
    _require(mapping.get(key) is True, "explicit true required: " + key)


def _integer(value, label, minimum=0):
    _require(type(value) is int and value >= minimum, "bad integer: " + label)
    return value


def _check_terminal(root, plan, base_path, source_pins):
    def read(relative):
        path = root / relative
        source_pins[relative] = _sha(path)
        return _read(path)
    launch = read("launch.json")
    _same({k: launch.get(k) for k in ("experiment", "source_commit", "source_git_status",
                                    "plan", "plan_sha256", "protocol_sha256", "formal",
                                    "expected_entries", "expected_lanes")},
          dict(experiment=EXPERIMENT, source_commit=SOURCE_COMMIT, source_git_status="",
               plan=str(base_path), plan_sha256=BASE_SHA, protocol_sha256=PROTOCOL_SHA,
               formal=True, expected_entries=126, expected_lanes=8), "original launch")
    _false(launch, "training_released")
    _true(launch, "no_success_rate")
    _same(launch.get("devices"), [str(i) for i in range(8)], "eight old devices")
    error = read("ERROR.json")
    _false(error, "training_released")
    _true(error, "no_success_rate")
    _require(error.get("error_type") == "RuntimeError" and
             error.get("message") == "collector failed; preserve original logs and partial outputs",
             "missing original terminal failure")
    _require(not (root / "COMPLETE.json").exists(), "source has conflicting completion marker")
    exits = []
    for lane in range(8):
        lane_path = f"lane{lane}"
        error = read(lane_path + "/ERROR.json")
        _false(error, "training_released")
        _true(error, "no_success_rate")
        _require(error.get("error_type") == "RuntimeError" and bool(error.get("message")),
                 "missing lane terminal error")
        _require(not (root / lane_path / "cleanup_errors.json").exists(),
                 "owned-process cleanup failed")
        header = read(lane_path + "/launch.json")
        _same({k: header.get(k) for k in ("experiment", "plan_sha256", "protocol_sha256", "lane", "keys")},
              dict(experiment=EXPERIMENT, plan_sha256=BASE_SHA, protocol_sha256=PROTOCOL_SHA,
                   lane=lane, keys=[e["key"] for e in plan["lanes"][lane]]), "original lane")
        relative = lane_path + "/processes.jsonl"
        source_pins[relative] = _sha(root / relative)
        events = _lines(root / relative)
        spawned, ended, returned = {}, {}, {}
        for event in events:
            role, kind = event.get("role"), event.get("event")
            _require(role in ("wa", "lightnav", "worker"), "foreign process role")
            pid = _integer(event.get("pid"), "owned process pid", 1)
            _require(isinstance(event.get("utc"), str) and event["utc"], "missing event time")
            if kind == "spawn":
                _require(role not in spawned and event.get("process_group") == pid,
                         "duplicate/nonowned spawn")
                spawned[role] = pid
                _require(isinstance(event.get("command"), dict), "missing owned command")
            elif kind in ("worker_return", "exit"):
                _require(spawned.get(role) == pid and type(event.get("returncode")) is int,
                         "unmatched owned process exit")
                if kind == "worker_return":
                    _require(role == "worker" and role not in returned and role not in ended,
                             "invalid worker-return event")
                    returned[role] = event["returncode"]
                else:
                    _require(role not in ended, "duplicate owned exit")
                    ended[role] = event["returncode"]
            else:
                raise ValueError("unexpected owned-process event")
        _same(set_to_list(spawned), ["lightnav", "wa", "worker"], "three spawned roles")
        _same(set_to_list(ended), ["lightnav", "wa", "worker"], "three exited roles")
        if lane == 1:
            _require(returned.get("worker") == ended["worker"] == 1, "missing failing worker exit")
        exits.append(dict(lane=lane, roles={r: dict(pid=spawned[r], returncode=ended[r])
                                          for r in sorted(spawned)}))
    return launch, exits


def set_to_list(mapping):
    return sorted(mapping)


def _branch(path, name):
    _require(path.is_dir() and not path.is_symlink(), "missing branch root")
    d = {field: _read(path / (field + ".json")) for field in JSON_NAMES}
    meta, adm, done = d["metadata"], d["admission"], d["complete"]
    teacher = "student" if name == "student" else ("lightnav" if name.startswith("lightnav") else "oracle")
    repeat = name.endswith("_repeat")
    for doc in (meta, adm):
        _same({k: doc.get(k) for k in ("experiment", "partition", "task", "key", "teacher", "verification_only")},
              dict(experiment=EXPERIMENT, partition="evaluation_adaptation", task="stt",
                   key=KEY, teacher=teacher, verification_only=repeat), "branch identity")
    for doc in (meta, adm, done):
        _false(doc, "training_eligible")
    for doc in (adm, done):
        _false(doc, "training_released")
        _true(doc, "complete")
    _false(adm, "transport_fallback")
    _same(d["fallback_events"], [], "fallback events")
    _same(adm.get("issues"), [], "recorder issues")
    _same(d["result"], adm.get("result"), "stored branch result")
    _require(binary_flag(d["result"].get("policy_init_valid"), "initialization"), "invalid initialization")
    success = binary_flag(d["result"].get("success"), "success")
    collision = binary_flag(d["result"].get("collision"), "collision")
    _require(success == (teacher == "oracle") and collision == (teacher != "oracle"),
             "unexpected frozen branch outcome")
    expected_n = 47 if teacher == "oracle" else 11
    obs, acts, replay = d["observations"], d["actions"], d["replay"]
    _require(len(obs) == len(acts) == len(replay) == expected_n == done.get("frames"),
             "observation/action completeness differs")
    _require(d["result"].get("total_step") == expected_n, "terminal count differs")
    expected_names = {f + ".json" for f in JSON_NAMES} | {"initial_panoptic.npy"}
    expected_names |= {f"rgb_{i:04d}.png" for i in range(expected_n)}
    if teacher != "student":
        expected_names.add("branch.json")
    _same(sorted(p.name for p in path.iterdir()), sorted(expected_names), "branch file inventory")
    for p in path.iterdir():
        _require(p.is_file() and not p.is_symlink(), "nonfile branch member")
    _same(d["first_start"], d["pair_start"], "paired first start")
    _same(d["first_start"]["dynamic_state_sha256"], canonical_sha(d["first_start"]["dynamic_state"]),
          "initial state hash")
    times = np.asarray([o["timestamp_s"] for o in obs], dtype=float)
    _require(np.isfinite(times).all() and (np.diff(times) > 0).all(), "invalid observed timebase")
    _same(meta["observed_frame_dt_s"], np.diff(times).tolist(), "actual frame intervals")
    for i, (o, a, r) in enumerate(zip(obs, acts, replay)):
        _require(o["sim_step"] == a["sim_step"] == r["step"] == r["observation_index"] == i,
                 "step/index mismatch")
        _require(o["frame"] == f"rgb_{i:04d}.png", "foreign image path")
        _require(r["timestamp_s"] == times[i] and
                 o["simulator_world_time_s"] == r["simulator_world_time_s"] == r["dynamic_state"]["timestamp"],
                 "observed state time mismatch")
        _same(r["dynamic_state_sha256"], canonical_sha(r["dynamic_state"]), "dynamic state hash")
        owner = "student" if teacher == "student" or i < 6 else "teacher"
        _require(a.get("owner") == r.get("owner") == owner, "action ownership mismatch")
        _same(a["normalized_action"], r["action"], "selected action")
        _same(r["normalized_action"], r["action"], "normalized action")
        values = np.asarray(r["action"], dtype=float)
        _require(values.shape == (3,) and np.isfinite(values).all() and (abs(values) <= 1).all(),
                 "invalid action")
        _false(r, "post_state_recorded")
        _require("post_state" not in r, "fabricated poststate")
    positions = np.asarray([o["robot_position_world"] for o in obs], dtype=float)
    _require(positions.shape == (expected_n, 3) and np.isfinite(positions).all(), "invalid robot poses")
    raw, windows = d["raw_windows"], d["windows"]
    expected_raw = [i for i in range(expected_n) if times[i] + .7 <= times[-1] + 1e-8]
    _same([w["current_index"] for w in raw], expected_raw, "raw window set")
    raw_by_index = {}
    for w in raw:
        i = _integer(w["current_index"], "window index")
        offsets = np.asarray(w["future_times_s"], dtype=float)
        _require(offsets.shape == (7,) and np.allclose(offsets, np.arange(1, 8)/10, rtol=0, atol=1e-12),
                 "incorrect seven time offsets")
        query = times[i] + offsets
        right = np.clip(np.searchsorted(times, query, side="left"), 1, expected_n-1)
        _same(w["future_bracket_indices"], np.stack((right-1, right), axis=-1).tolist(),
              "actual time interpolation bracket")
        future = np.stack([np.interp(query, times, positions[:, axis]) for axis in range(3)], axis=-1)
        rotation = np.asarray(obs[i]["robot_rotation_world_from_body"])
        _require(rotation.shape == (3, 3) and np.isfinite(rotation).all(), "invalid rotation")
        local = (future-positions[i]) @ rotation
        expected = np.stack((local[:, 0], -local[:, 2]), axis=-1)
        actual = np.asarray(w["trajectory_xy_m"])
        _require(actual.shape == (7, 2) and np.isfinite(actual).all()
                 and np.allclose(actual, expected, rtol=0, atol=1e-9), "executed-pose label mismatch")
        raw_by_index[i] = w
    expected_windows = list(range(6, 32)) if name == "oracle_0006" else []
    _same([w["current_index"] for w in windows], expected_windows, "candidate window set")
    for w in windows:
        i = w["current_index"]
        for field, value in raw_by_index[i].items():
            _same(w[field], value, "candidate/raw label")
        end = max(b[1] for b in w["future_bracket_indices"])
        _same(w["teacher_owned_action_indices"], [i, end-1], "complete owned action interval")
        _require(w["label_endpoint_observation_index"] == end and
                 all(acts[j]["owner"] == replay[j]["owner"] == "teacher" for j in range(i, end)),
                 "student action leaked into suffix labels")
        _false(w, "training_eligible")
    _require(adm["candidate_windows"] == done["windows"] == done["candidate_windows"] == len(windows),
             "candidate count mismatch")
    _require(adm["raw_windows"] == done["raw_windows"] == len(raw), "raw count mismatch")
    if teacher != "student":
        _true(adm, "replay_verified")
        proof = adm["agent_validation"]
        _require(proof["verified_prefix_frames"] == proof["required_prefix_frames"] == 7,
                 "missing actual 0..k coverage")
        _require(proof["agent_step_before_reset"] == expected_n, "post-reset proof")
        for field in ("replay_wrapper", "prefix_hash_matches", "environment_bound", "takeover_matches"):
            _true(proof, field)
        _false(proof, "hidden_rng_contact_state_proven")
        _require(meta["takeover_step"] == adm["takeover_step"] == 6, "wrong takeover")
        _require(d["takeover"]["step"] == d["takeover"]["observation_index"] == 6, "missing takeover frame")
        _require(adm["owned_suffix"]["action_count"] == expected_n-6, "wrong suffix count")
        d["branch"] = _read(path/"branch.json")
    return d


def _audit_record(root, plan):
    collection = root/"lane0/collection"
    base = collection/"stt"/KEY
    rows = _lines(collection/"records.jsonl")
    _require(len(rows) == 1, "exactly one complete original row required")
    row = rows[0]
    _same(_read(base/"search.json"), row, "records/search")
    _same({k: row.get(k) for k in ("experiment", "task", "key", "plan_sha256", "protocol_sha256", "outcome")},
          dict(experiment=EXPERIMENT, task="stt", key=KEY, plan_sha256=BASE_SHA, protocol_sha256=PROTOCOL_SHA,
               outcome="repeated_teacher_recovery_candidate"), "frozen row identity")
    _false(row, "training_released")
    _false(row, "score_backfill_allowed")
    originals = [e for e in plan["entries"] if e["key"] == KEY]
    _require(len(originals) == 1 and any(e["key"] == KEY for e in plan["lanes"][0]), "foreign reusable key")
    _same(row["baseline_result_unchanged"], originals[0]["baseline_result"], "original baseline preserved")
    _require(len(row.get("attempts", [])) == 1, "unexpected number of k attempts")
    attempt = row["attempts"][0]
    _require(attempt.get("takeover_step") == 6 and set(attempt["branches"]) == {"lightnav", "oracle"},
             "missing same-k dual teacher")
    data = {name: _branch(base/name, name) for name in BRANCHES}
    refs = {"student": row["student"], "lightnav_0006": attempt["branches"]["lightnav"],
            "oracle_0006": attempt["branches"]["oracle"], "oracle_0006_repeat": attempt["repeat"]}
    for name, ref in refs.items():
        _require(ref["artifact_root"] == str(base/name), "foreign branch root")
        _same(ref["result"], data[name]["result"], "row/stored result")
        if name != "student":
            _same(ref, data[name]["branch"], "row/stored branch")
            for field in PAIR_FIELDS + ("complete", "replay_verified", "transport_fallback",
                                        "actual_takeover_state_sha256", "candidate_windows", "verification_only"):
                _same(ref[field], data[name]["admission"][field], "branch/admission " + field)
    _true(row["student"], "initial_pair_verified")
    _same(row["student"]["first_start"], data["student"]["first_start"], "student start")
    selection = select_recovery_teacher(attempt["branches"]["lightnav"], attempt["branches"]["oracle"])
    _same(attempt["selection"], selection, "same-k winner selection")
    _require(selection["selected_teacher"] == "oracle", "wrong winner")
    original, repeat = attempt["branches"]["oracle"], attempt["repeat"]
    for field in PAIR_FIELDS + ("teacher",):
        _same(original[field], repeat[field], "repeat identity " + field)
    _require(original.get("verification_only") is False and repeat.get("verification_only") is True,
             "repeat must remain verification-only")
    _true(attempt, "repeat_valid")
    _require(attempt["selected_candidate_windows"] == 26, "wrong selected window count")
    expected_accept = dict(
        teacher="oracle", takeover_step=6, artifact_root=str(base/"oracle_0006"),
        repeat_artifact_root=str(base/"oracle_0006_repeat"), selection=selection,
        prefix_sha256=original["prefix_sha256"], candidate_only=True, training_released=False)
    _same(row["accepted"], expected_accept, "accepted candidate scope")
    prefix = data["student"]["replay"]
    prefix_sha = canonical_sha(prefix)
    for name in BRANCHES[1:]:
        d = data[name]
        _same(d["admission"]["prefix_sha256"], prefix_sha, "fresh WA prefix hash")
        for i in range(7):
            _same(d["replay"][i]["rgb_sha256"], prefix[i]["rgb_sha256"], "prefix RGB")
            _same(d["replay"][i]["dynamic_state"], prefix[i]["dynamic_state"], "prefix dynamic state")
            _require(_sha(base/name/f"rgb_{i:04d}.png") ==
                     _sha(base/"student"/f"rgb_{i:04d}.png"), "prefix saved RGB differs")
            if i < 6:
                _same(d["replay"][i]["action"], prefix[i]["action"], "WA prefix action")
        _same(d["takeover"]["dynamic_state"], prefix[6]["dynamic_state"], "takeover state")
        _same(d["admission"]["takeover_state_sha256"], canonical_sha(prefix[6]["dynamic_state"]), "canonical k state")
        _same(d["admission"]["actual_takeover_state_sha256"], canonical_sha(d["takeover"]["dynamic_state"]), "actual k state")
        _same(d["metadata"]["initial_bbox_sensor_xyxy_original"],
              data["student"]["metadata"]["initial_bbox_sensor_xyxy_original"], "original first bbox")
    for field in ("observations", "actions", "replay", "raw_windows", "result"):
        _same(data["oracle_0006"][field], data["oracle_0006_repeat"][field], "repeat actual " + field)
    for i in range(47):
        _require(_sha(base/"oracle_0006"/f"rgb_{i:04d}.png") ==
                 _sha(base/"oracle_0006_repeat"/f"rgb_{i:04d}.png"), "repeat saved RGB differs")
    identity = _read(collection/"plan_identity.json")
    _same(identity["sha256"], BASE_SHA, "collection plan SHA")
    _same(identity["protocol_sha256"], PROTOCOL_SHA, "collection protocol")
    model = _read(collection/"model.json")
    _require(model["checkpoint"] == plan["checkpoint"]["path"] and
             model["checkpoint_sha256"] == plan["checkpoint"]["sha256"] and
             model["step"] == plan["checkpoint"]["step"], "original model identity")
    _false(model, "text_used")
    _false(model, "world_predictor_inference")
    return row, originals[0], identity


def _inventory(root):
    collection = root/"lane0/collection"
    base = collection/"stt"/KEY
    files = [p for name in BRANCHES for p in (base/name).iterdir()]
    files += [base/"search.json"] + [collection/name for name in (
        "records.jsonl", "plan_identity.json", "model.json", "simulator_config.yaml")]
    _require(len(files) == len(set(files)) == INVENTORY_COUNT, "frozen artifact count changed")
    result = {str(p.relative_to(collection)): _sha(p) for p in sorted(files)}
    size = sum(p.stat().st_size for p in files)
    _require(size == INVENTORY_BYTES, "frozen artifact bytes changed")
    _require(canonical_sha(result) == INVENTORY_SHA, "frozen artifact checksum inventory changed")
    return dict(root=str(collection), file_count=len(result), bytes=size,
                canonical_sha256=canonical_sha(result), files=result)


def _remaining(plan):
    # Defend even if a test substitutes the base loader. Never mutate its object.
    entries, lanes = plan["entries"], plan["lanes"]
    _require(len(entries) == 126 and len(lanes) == 8, "invalid full base plan")
    keys = [e["key"] for e in entries]
    _require(len(set(keys)) == 126 and keys.count(KEY) == 1, "duplicate/missing base key")
    _same(lanes, [entries[i::8] for i in range(8)], "original lane partition")
    remaining = [copy.deepcopy(e) for e in entries if e["key"] != KEY]
    remaining_lanes = [[copy.deepcopy(e) for e in lane if e["key"] != KEY] for lane in lanes]
    _same([len(lane) for lane in remaining_lanes], REMAINING_LANE_COUNTS, "original lane sizes")
    flat = [e["key"] for lane in remaining_lanes for e in lane]
    _require(len(flat) == len(set(flat)) == 125 and KEY not in flat, "overlapping continuation keys")
    _require(set(flat) | {KEY} == set(keys), "missing/foreign continuation keys")
    return remaining, remaining_lanes


def build_continuation(base_plan_path, base_sha, old_root, *, expected_job_status=None):
    """Read-only constructor. Explicit FAILED is a caller claim, not a live query.

    The formal release MUST re-query the scheduler before using this overlay.
    """
    _require(expected_job_status == "FAILED", "explicit scheduler-verified FAILED status required")
    base_path, root = Path(base_plan_path), Path(old_root)
    _require(base_path.is_absolute(), "absolute base plan required")
    _require(base_sha == BASE_SHA, "unknown base plan SHA")
    _require(root == SOURCE_ROOT and root.is_dir() and not root.is_symlink(),
             "only original 61833/73055 output is reusable")
    plan = load_plan(base_path, base_sha)  # Original strict126 validation, unchanged.
    original_digest = canonical_sha(plan)
    remaining, lanes = _remaining(plan)
    pins = {}
    launch, exits = _check_terminal(root, plan, base_path, pins)
    row, original, identity = _audit_record(root, plan)
    _require(identity["path"] == str(base_path), "collection base plan path differs")
    inventory = _inventory(root)
    _require(canonical_sha(plan) == original_digest, "base plan mutated")
    collection = root/"lane0/collection"
    base = collection/"stt"/KEY
    frozen = dict(
        task="stt", key=KEY, original_lane=0,
        baseline_row_sha256=original["baseline_row_sha256"],
        records_path=str(collection/"records.jsonl"),
        records_sha256=inventory["files"]["records.jsonl"],
        row_sha256=canonical_sha(row),
        search_path=str(base/"search.json"),
        search_sha256=inventory["files"][f"stt/{KEY}/search.json"],
        accepted_artifact_root=row["accepted"]["artifact_root"],
        repeat_artifact_root=row["accepted"]["repeat_artifact_root"],
        selected_teacher="oracle", takeover_step=6, candidate_windows=26,
        prefix_sha256=row["accepted"]["prefix_sha256"],
        initial_rgb_sha256=row["attempts"][0]["selection"]["pair"]["initial_rgb_sha256"],
        takeover_state_sha256=row["attempts"][0]["selection"]["pair"]["takeover_state_sha256"],
        windows_sha256=inventory["files"][f"stt/{KEY}/oracle_0006/windows.json"],
        training_released=False, candidate_only=True, score_backfill_allowed=False)
    return dict(
        schema=SCHEMA, experiment=EXPERIMENT,
        base_plan=dict(path=str(base_path), sha256=base_sha), protocol_sha256=PROTOCOL_SHA,
        expected_count=126, new_count=125, reused_count=1, reused_keys=[KEY],
        source=dict(root=str(root), job_id=61833, task_id=73055, expected_job_status="FAILED",
                    status_authority="caller-confirmed scheduler status; persisted errors/exits additionally checked",
                    scheduler_recheck_required=True, source=launch["source"],
                    source_commit=launch["source_commit"], source_git_status=launch["source_git_status"],
                    owned_process_exits=exits, source_pins=pins),
        artifact_inventory=inventory, frozen_records=[frozen],
        remaining_entries=remaining, remaining_lanes=lanes,
        base_plan_unchanged=True, evaluation_adaptation=True, untouched_test=False,
        training_released=False, score_backfill_allowed=False, no_success_rate=True,
        release_requirements=[
            "recheck original scheduler terminal status and unchanged frozen evidence",
            "prove valid LightNav action semantics are unchanged by error-handling repair",
            "merge exactly one frozen plus125 new keys with no duplicates or omissions",
            "teacher suffix SE2/yaw/history/student-input audit before any training release"])


def load_continuation(path, sha):
    """Rebuild and compare everything: no arbitrary skip list or stale pin trust."""
    path = Path(path)
    _require(path.is_absolute(), "absolute continuation path required")
    _require(isinstance(sha, str) and len(sha) == 64 and _sha(path) == sha,
             "continuation file SHA mismatch")
    overlay = _read(path)
    _require(overlay.get("schema") == SCHEMA, "foreign continuation schema")
    base, source = overlay.get("base_plan", {}), overlay.get("source", {})
    expected = build_continuation(
        base.get("path", ""), base.get("sha256"), source.get("root", ""),
        expected_job_status=source.get("expected_job_status"))
    _same(overlay, expected, "continuation must exactly match reconstructed frozen evidence")
    return overlay
