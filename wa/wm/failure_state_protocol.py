"""Independent, CPU-only failure-state recovery protocol.

This never changes the legacy 4215 step-zero teacher admission. A selected
teacher is only a candidate: the collector must verify a second same-k replay
before admitting a teacher-owned suffix, and must exclude WA-prefix labels.
"""
import hashlib
import json
import math
from pathlib import Path
import re

from wa.wm.full_mixed_contract import summarize, validate_ready
from wa.wm.student_eval_contract import validate_student_rows

EXPERIMENT = "evaluation_adaptation_failure_state_v1"
MANIFEST_SHA = "a1f515534153ccaa720f787f01ea23b9097c174fe25020f756c7d7947eedf98f"
SUMMARY_SHA = "223b4b8140e79842408bc7104d16f75b7b4ff1d409354ee547b00e4b49f937f7"
COMBINED_SHA = "0ab45e1b35bb0b8809fcc77fcaabca59b35bab0839507d716c371ce2d5a6f358"
REPAIR_SHA = "6535f7b9a53e399ba7f2323c2d7f72d223146ee77f090223eb260f5885f6269a"
CHECKPOINT_SHA = "c510c04d987d141423401a25323ea353314107a16f4dac5d2901370ab199fa52"
CHECKPOINT_PATH = "/data/nas_ray/project/md-ak/users/zeying.gong/job_61609/task_72803/wa_hard_stt_train_a800_v1/checkpoint.pt"
CHECKPOINT = dict(path=CHECKPOINT_PATH, sha256=CHECKPOINT_SHA, step=59716)
PAIR_FIELDS = ("task", "key", "takeover_step", "seed", "protocol_sha256",
               "initial_rgb_sha256", "takeover_state_sha256", "prefix_sha256")
BACKOFFS = (5, 15, 30, 60)
MODEL_CONTRACT = dict(
    checkpoint=CHECKPOINT, mode="mixed", noise_mode="zero", sampling_steps=4,
    controller="learned_yaw_guard_v1", seed=7, inference_seed="7+step",
    semantic_protocol="mp3d_semantic_ply_v1",
    initialization_repair_plan_sha256=REPAIR_SHA,
    text_used=False, world_predictor_inference=False)
PROTOCOL = dict(
    experiment=EXPERIMENT, source_student=61609, model_contract=MODEL_CONTRACT,
    source_summary_sha256=SUMMARY_SHA, source_combined_sha256=COMBINED_SHA,
    manifest_sha256=MANIFEST_SHA, repair_plan_sha256=REPAIR_SHA,
    scope="all 126 success-zero STT episodes in original manifest order",
    expected_count=126, lanes=8, backoffs=list(BACKOFFS),
    candidate_rule="sorted(set(max(0,N-b) for b in backoffs)|{0},reverse=True); N>=1",
    prefix_source="new actual student rollout; N is executed action count",
    same_k_teachers=["lightnav", "oracle"], exact_tie="lightnav",
    eligibility="complete AND replay_verified AND success AND init_valid AND NOT collision AND NOT transport_fallback",
    teacher_choice="higher original result.following_rate at the same k only",
    verification="one additional same-k same-teacher successful collision-free fallback-free verified replay",
    verification_failure="continue to earlier k, never use the failed candidate",
    already_successful_student="record no_recovery_needed; no recovery labels",
    labels="successful teacher-owned suffix only; no WA prefix or repeated verification windows",
    evaluation_adaptation=True, untouched_test=False, legacy_step0_admission_unchanged=True)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


PROTOCOL_SHA = canonical_sha(PROTOCOL)


def _sha(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("invalid SHA256: " + label)
    return value


def binary_flag(value, label):
    """Accept actual JSON bool/int/float binary flags, not strings or NaN."""
    if type(value) is bool:
        return value
    if type(value) not in (int, float) or not math.isfinite(value) or value not in (0, 1):
        raise ValueError("invalid binary flag: " + label)
    return bool(value)


def _strict_bool(value, label):
    if type(value) is not bool:
        raise ValueError("explicit boolean required: " + label)
    return value


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("invalid integer: " + label)
    return value


def candidate_steps(N):
    """Closest bounded predecessor first; never the failed terminal state N."""
    _integer(N, "executed action count N", 1)
    return sorted({max(0, N-b) for b in BACKOFFS} | {0}, reverse=True)


def _key(task, key):
    if task != "stt" or not isinstance(key, str) or re.fullmatch(r"[A-Za-z0-9]+/[0-9]+", key) is None:
        raise ValueError("expected explicit STT scene/episode key")


def _result(result):
    if not isinstance(result, dict):
        raise ValueError("missing result")
    flags = {f: binary_flag(result.get(f), f)
             for f in ("success", "collision", "policy_init_valid")}
    rate = result.get("following_rate")
    if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("following_rate must be finite in [0,1]")
    if flags["success"] and (flags["collision"] or not flags["policy_init_valid"]):
        raise ValueError("inconsistent successful branch")
    return flags, rate


def select_recovery_teacher(lightnav, oracle):
    """Validate one same-k pair; return a candidate, NEVER training admission.

    Malformed/missing pairing evidence raises ValueError. Explicitly incomplete,
    unverified, failed or fallback branches are ineligible rather than selected.
    The additional winner replay is deliberately the collector's responsibility.
    """
    branches = dict(lightnav=lightnav, oracle=oracle)
    eligible, reasons = {}, {}
    for name, b in branches.items():
        if not isinstance(b, dict) or b.get("experiment") != EXPERIMENT or b.get("teacher") != name:
            raise ValueError("wrong experiment or teacher identity")
        if b.get("verification_only", False) is not False:
            raise ValueError("verification replay cannot be a demonstration candidate")
        for f in ("complete", "replay_verified", "transport_fallback"):
            _strict_bool(b.get(f), f)
        _key(b.get("task"), b.get("key"))
        _integer(b.get("takeover_step"), "takeover_step")
        if type(b.get("seed")) is not int or b["seed"] != 7:
            raise ValueError("seed must remain 7")
        for f in PAIR_FIELDS[4:]:
            _sha(b.get(f), f)
        if b["protocol_sha256"] != PROTOCOL_SHA:
            raise ValueError("foreign failure-state protocol")
        flags, _ = _result(b.get("result"))
        checks = dict(complete=b["complete"], replay_verified=b["replay_verified"],
                      success=flags["success"], policy_init_valid=flags["policy_init_valid"],
                      collision_free=not flags["collision"],
                      transport_fallback_free=not b["transport_fallback"])
        reasons[name] = [field for field, valid in checks.items() if not valid]
        eligible[name] = not reasons[name]
    if any(lightnav[f] != oracle[f] for f in PAIR_FIELDS):
        raise ValueError("branches do not share the same k, prefix, start and protocol")
    valid = [name for name in branches if eligible[name]]
    selected = None
    if not valid:
        reason = "no_eligible_teacher_at_this_k"
    elif len(valid) == 1:
        selected, reason = valid[0], "only_eligible_teacher"
    else:
        ln, oc = (branches[t]["result"]["following_rate"] for t in ("lightnav", "oracle"))
        selected = "oracle" if oc > ln else "lightnav"
        reason = "higher_following_rate" if ln != oc else "equal_rate_fixed_lightnav_tie"
    return dict(experiment=EXPERIMENT, pair={f: lightnav[f] for f in PAIR_FIELDS},
                selected_teacher=selected, reason=reason, eligible=eligible,
                ineligible_reasons=reasons,
                results={n: dict(b["result"]) for n, b in branches.items()},
                demonstration_candidate=selected is not None,
                verification_required=selected is not None, training_released=False,
                pending="additional same-k winner replay and teacher-owned suffix admission")


def _read_pinned(path, expected):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("source paths must be absolute")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("source hash mismatch: " + str(path))
    return data


def _document(data):
    def bad_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    def unique_object(pairs):
        obj = {}
        for k, v in pairs:
            if k in obj:
                raise ValueError("duplicate JSON key: " + k)
            obj[k] = v
        return obj
    return json.loads(data, parse_constant=bad_constant, object_pairs_hook=unique_object)


def derive_entries(manifest, summary, rows, repair_plan):
    """Derive the entire failure set; caller supplies hash-verified documents."""
    if manifest.get("seed_each_episode") != 7 or type(manifest.get("seed_each_episode")) is not int or manifest.get("shards") != 8:
        raise ValueError("manifest seed/lane contract changed")
    if repair_plan.get("version") != "initial_rgb_mesh_bbox_v1" or len(repair_plan.get("repairs", [])) != 7:
        raise ValueError("frozen seven-BBox repair contract changed")
    validate_student_rows(rows, dict(checkpoint_sha=CHECKPOINT_SHA, step=59716))
    recomputed = summarize(rows, manifest, checkpoint_sha=CHECKPOINT_SHA, step=59716)
    for field, value in recomputed.items():
        if field != "limits" and canonical_sha(summary.get(field)) != canonical_sha(value):
            raise ValueError("source summary differs: " + field)
    if summary.get("experiment") != "evaluation_set_adaptation_v1" or summary.get("new_episodes") != 4215 or summary.get("reused_baseline_episodes") != 0:
        raise ValueError("fully audited new 61609 evaluation required")
    by_key = {}
    for row in rows:
        _result(row)
        if type(row.get("policy_init_valid")) is not bool or not row["policy_init_valid"]:
            raise ValueError("invalid original policy initialization")
        if row["task"] == "stt":
            by_key[row["key"]] = row
    definitions = manifest["tasks"]["stt"]["episodes"]
    if len(definitions) != 1405 or len(by_key) != 1405 or len({e["key"] for e in definitions}) != 1405:
        raise ValueError("expected 1405 unique original STT definitions")
    entries = []
    for manifest_index, definition in enumerate(definitions):
        key = definition["key"]
        _key("stt", key)
        row = by_key[key]
        if row["success"] == 0:
            fields = ("finish", "status", "success", "following_rate", "following_step",
                      "total_step", "collision", "policy_init_valid")
            entries.append(dict(task="stt", key=key, scene_id=definition["scene_id"],
                episode_id=definition["episode_id"], manifest_index=manifest_index,
                baseline_result={f: row[f] for f in fields},
                baseline_row_sha256=canonical_sha(row), artifact_root=row["artifact_root"]))
    if len(entries) != 126:
        raise ValueError("must include all and only 126 original STT failures")
    return entries


def make_plan(manifest, summary, rows, repair_plan, paths, ready_hashes):
    """Pure constructor, separate from source IO so schema has CPU fixture tests."""
    entries = derive_entries(manifest, summary, rows, repair_plan)
    specs = {
        "manifest": dict(path=paths["manifest"], sha256=MANIFEST_SHA),
        "repair_plan": dict(path=paths["repair_plan"], sha256=REPAIR_SHA),
        "summary": dict(path=paths["summary"], sha256=SUMMARY_SHA),
        "combined": dict(path=paths["combined"], sha256=COMBINED_SHA)}
    hashes = {v["path"]: v["sha256"] for v in specs.values()}
    hashes.update(ready_hashes)
    return dict(experiment=EXPERIMENT, protocol=PROTOCOL, protocol_sha256=PROTOCOL_SHA,
        expected_count=126, seed=7, evaluation_adaptation=True, untouched_test=False,
        legacy_step0_admission_unchanged=True, checkpoint=CHECKPOINT,
        model_contract=MODEL_CONTRACT, manifest=specs["manifest"], repair_plan=specs["repair_plan"],
        sources=dict(summary=specs["summary"], combined=specs["combined"]),
        source_hashes=hashes, entries=entries, lanes=[entries[i::8] for i in range(8)])


def build_plan(manifest_path, student_root, bbox_plan):
    """Read exact pinned sources and eight STT model headers, without writing.

    Checkpoint SHA is pinned identity metadata here; the collector must verify
    actual checkpoint bytes before inference. No model weights are loaded.
    """
    root = Path(student_root).resolve()
    paths = dict(manifest=str(Path(manifest_path).resolve()),
        summary=str(root/"summary.json"), combined=str(root/"combined_episodes.jsonl"),
        repair_plan=str(Path(bbox_plan).resolve()))
    blobs = {name: _read_pinned(paths[name], expected) for name, expected in
             (("manifest", MANIFEST_SHA), ("summary", SUMMARY_SHA),
              ("combined", COMBINED_SHA), ("repair_plan", REPAIR_SHA))}
    manifest, summary, repair = (_document(blobs[name]) for name in ("manifest", "summary", "repair_plan"))
    rows = [_document(line) for line in blobs["combined"].splitlines() if line.strip()]
    ready_hashes = {}
    stt_root = Path(summary["partition_roots"]["stt"])
    for lane in range(8):
        path = str(stt_root/f"shard_{lane:02d}"/"server_ready.json")
        expected = summary["source_hashes"].get(path)
        _sha(expected, "frozen ready header")
        ready = _document(_read_pinned(path, expected))
        validate_ready(ready, checkpoint_sha=CHECKPOINT_SHA, step=59716)
        if ready.get("checkpoint") != CHECKPOINT_PATH or ready.get("seed") != "7+step":
            raise ValueError("original checkpoint path or inference seed differs")
        ready_hashes[path] = expected
    plan = make_plan(manifest, summary, rows, repair, paths, ready_hashes)
    validate_plan_structure(plan)
    return plan


def validate_plan_structure(plan):
    if not isinstance(plan, dict):
        raise ValueError("plan must be an object")
    fixed = dict(experiment=EXPERIMENT, protocol=PROTOCOL, protocol_sha256=PROTOCOL_SHA,
                 expected_count=126, seed=7, evaluation_adaptation=True, untouched_test=False,
                 legacy_step0_admission_unchanged=True, checkpoint=CHECKPOINT,
                 model_contract=MODEL_CONTRACT)
    for key, value in fixed.items():
        if canonical_sha(plan.get(key)) != canonical_sha(value):
            raise ValueError("wrong plan contract: " + key)
    entries, lanes = plan.get("entries"), plan.get("lanes")
    if not isinstance(entries, list) or len(entries) != 126 or not isinstance(lanes, list) or len(lanes) != 8:
        raise ValueError("expected 126 entries and 8 lanes")
    seen, last_index = set(), -1
    for e in entries:
        _key(e.get("task"), e.get("key"))
        if e["key"] in seen:
            raise ValueError("duplicate planned key")
        seen.add(e["key"])
        index = _integer(e.get("manifest_index"), "manifest_index")
        if index <= last_index or index >= 1405:
            raise ValueError("entries must retain original manifest order")
        last_index = index
        _sha(e.get("baseline_row_sha256"), "baseline_row_sha256")
        flags, _ = _result(e.get("baseline_result"))
        if flags["success"] or not flags["policy_init_valid"]:
            raise ValueError("only original valid-initialization failures may be planned")
    if canonical_sha(lanes) != canonical_sha([entries[i::8] for i in range(8)]):
        raise ValueError("lane order/coverage differs; duplicates and cross-lane overlaps forbidden")
    specs = dict(manifest=plan.get("manifest"), repair_plan=plan.get("repair_plan"),
                 **plan.get("sources", {}))
    for name, expected in (("manifest", MANIFEST_SHA), ("summary", SUMMARY_SHA),
                           ("combined", COMBINED_SHA), ("repair_plan", REPAIR_SHA)):
        spec = specs.get(name)
        if not isinstance(spec, dict) or spec.get("sha256") != expected or not isinstance(spec.get("path"), str) or not Path(spec["path"]).is_absolute():
            raise ValueError("wrong frozen source: " + name)
        if plan.get("source_hashes", {}).get(spec["path"]) != expected:
            raise ValueError("source inventory missing " + name)
    return plan


def load_plan(path, sha):
    """Fail closed on plan hash, contract, full fixed set, order and source bytes."""
    _sha(sha, "plan")
    plan = _document(_read_pinned(Path(path).resolve(), sha))
    validate_plan_structure(plan)
    summary_path = Path(plan["sources"]["summary"]["path"])
    if Path(plan["sources"]["combined"]["path"]) != summary_path.parent/"combined_episodes.jsonl" or summary_path.name != "summary.json":
        raise ValueError("summary and combined must share the fixed audit root")
    expected = build_plan(plan["manifest"]["path"], summary_path.parent,
                          plan["repair_plan"]["path"])
    if canonical_sha(plan) != canonical_sha(expected):
        raise ValueError("plan does not exactly match pinned original failures and headers")
    return plan
