"""Fixed best61609 handoff allowlist; no packaging, uploading, or media rehash.

All scientific JSON remains byte-for-byte unchanged.  The streaming packer must
independently recheck regular paths and each existing SHA while reading payloads.
This module verifies small pinned anchors and stats, not the large payload bytes.
"""
import hashlib
import json
import re
import stat
from pathlib import Path, PurePosixPath

SCHEMA = "wa_best61609_asset_manifest_v1"
R = "/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928"
J = "/data/nas_ray/project/md-ak/users/zeying.gong"
AUDIT = R + "/artifacts/student61609_full_audit_20261007_v1/student_teacher_pair_audit.json"
MANIFEST = "/data/nas_ray/home/zeying.gong/algorithm/repos/WLA-EVT-20260925/evt_full_20260926/manifest.json"
PLAN = R + "/artifacts/initial_bbox_repair_v2/plan.json"
SELECTIONS = R + "/artifacts/dual_teacher_complete_audit_20261006_v1/combined_selections.jsonl"
BASELINE = J + "/job_61171/task_72191/wa_semantic_targeted_v1/combined_episodes.jsonl"
CHECKPOINT = J + "/job_61609/task_72803/wa_hard_stt_train_a800_v1/checkpoint.pt"
CHECKPOINT_SHA256 = "c510c04d987d141423401a25323ea353314107a16f4dac5d2901370ab199fa52"
CHECKPOINT_BYTES = 4358277705
EVIDENCE_COUNT = 25305
EVIDENCE_BYTES = 2330389716
ANCHORS = {
    AUDIT: "dc98149bdcfb6baf05d1fae167de572b476679f48384dd91ea92e57851645844",
    MANIFEST: "a1f515534153ccaa720f787f01ea23b9097c174fe25020f756c7d7947eedf98f",
    PLAN: "6535f7b9a53e399ba7f2323c2d7f72d223146ee77f090223eb260f5885f6269a",
    SELECTIONS: "f45f21d71b63d7f2a0898e1e762c393a95b0f3eaf8291bd1affbc827d23354d8",
}
VAL_ROOT = "/data/nas_ray/home/zeying.gong/algorithm/repos/OmTrackVLA/data/datasets/track"
VAL_SHA256 = {
    "stt": "8a96fe3be38ab78b0b8985e71645eef337d396be126bbd78ce1c87ffef5a0c63",
    "dt": "1997a14b2a03fb1319c924a3d38fe5824f1b627130d77b7e3d137633f6b63398",
    "at": "ed4aee0d49c2bd4b0f75e9922c53bd41d7aa50da2aa35fd5f994205c4537b093",
}
TEACHER_ROOTS = (
    J + "/job_61264/task_72341/wa_dual_teacher_inset_v1",
    J + "/job_61269/task_72346/wa_dual_teacher_resume_v1",
)
TEACHERS = ("lightnav", "oracle")
TEACHER_FILES = {"pair_start.json", "observations.json", "rgb_0000.png"}
REPAIR_KEYS = {(t, "pRbA3pwrgk9/" + e) for t, es in
               (("stt", ("27", "38")), ("dt", ("27", "38")),
                ("at", ("27", "38", "71"))) for e in es}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             "invalid SHA256")
    return value


def _path(value):
    _require(isinstance(value, str) and "\x00" not in value, "invalid source path")
    p = PurePosixPath(value)
    _require(p.is_absolute() and p.root == "/" and str(p) == value and ".." not in p.parts,
             "source path is not canonical absolute")
    return p


def _key(value):
    _require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", value)
             is not None, "invalid episode key")
    return value


def _json(data):
    def pairs(items):
        out = {}
        for k, v in items:
            _require(k not in out, "duplicate JSON key")
            out[k] = v
        return out
    def constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def _collect_evidence(audit, manifest, plan, selections):
    """Pure object validation; production callers cannot remap fixed roots/pins."""
    _require(set(audit) == {"status", "episodes", "teacher_artifact_hashes", "comparison", "scope"},
             "unexpected audit schema")
    _require(audit["status"] == "PASS" and type(audit["episodes"]) is int and
             audit["episodes"] == 4215, "audit is not full PASS")
    _require(type(manifest["seed_each_episode"]) is int and manifest["seed_each_episode"] == 7
             and type(manifest["shards"]) is int and manifest["shards"] == 8,
             "manifest seed/shards changed")
    _require(set(manifest["tasks"]) == {"stt", "dt", "at"}, "manifest task coverage")
    pins = dict(ANCHORS)
    def add(path, digest):
        _path(path); _sha(digest)
        _require(path not in pins, "duplicate source path")
        pins[path] = digest
    expected = set()
    for task, spec in manifest["tasks"].items():
        _require(spec["path"] == VAL_ROOT + "/" + task.upper() + "/val/val.json.gz" and
                 spec["sha256"] == VAL_SHA256[task], "val path/hash changed")
        add(spec["path"], spec["sha256"])
        _require(isinstance(spec["episodes"], list) and len(spec["episodes"]) == 1405,
                 "manifest episode count")
        for row in spec["episodes"]:
            identity = task + ":" + _key(row["key"])
            _require(identity not in expected, "duplicate manifest episode")
            expected.add(identity)
    _require(isinstance(selections, list) and len(selections) == 4215, "selection count")
    indexed = {}
    for row in selections:
        pair = row["pair"]
        _require(pair["task"] in manifest["tasks"], "unknown selection task")
        identity = pair["task"] + ":" + _key(pair["key"])
        _require(identity not in indexed, "duplicate selection episode")
        _require(type(pair["seed"]) is int and pair["seed"] == 7 and
                 type(pair["takeover_step"]) is int and pair["takeover_step"] == 0,
                 "selection is not seed7 initial state")
        indexed[identity] = row
    _require(set(indexed) == expected == set(audit["teacher_artifact_hashes"]),
             "manifest/selection/audit coverage differs")
    for identity, teachers in audit["teacher_artifact_hashes"].items():
        _require(set(teachers) == set(TEACHERS), "teacher coverage")
        selected = indexed[identity]
        task, key = identity.split(":", 1)
        _require(set(selected["branches"]) == set(TEACHERS), "selection branch coverage")
        for teacher in TEACHERS:
            branch = selected["branches"][teacher]
            _require(branch["task"] == task and branch["key"] == key and
                     branch["teacher"] == teacher, "branch identity mismatch")
            root = _path(branch["artifact_root"])
            allowed = {PurePosixPath(base) / ("lane" + str(lane)) / "collection" /
                       task / key / teacher for base in TEACHER_ROOTS for lane in range(8)}
            _require(root in allowed, "teacher root outside fixed jobs/lanes/key")
            teacher_pins = teachers[teacher]
            _require(isinstance(teacher_pins, dict) and len(teacher_pins) == 3 and
                     set(teacher_pins) == {str(root / name) for name in TEACHER_FILES},
                     "teacher must have exactly three fixed evidence files")
            for path, digest in teacher_pins.items():
                add(path, digest)
    _require(plan["version"] == "initial_rgb_mesh_bbox_v1" and
             isinstance(plan["repairs"], list) and len(plan["repairs"]) == 7,
             "repair protocol/count changed")
    repairs = {(r["task"], r["key"]): r for r in plan["repairs"]}
    _require(set(repairs) == REPAIR_KEYS, "repair keys changed")
    _require(isinstance(plan["lanes"], list) and len(plan["lanes"]) == 2 and
             all(isinstance(lane, list) for lane in plan["lanes"]), "repair lane schema")
    lanes = [r for lane in plan["lanes"] for r in lane]
    _require(len(lanes) == 7 and {(r["task"], r["key"]): r for r in lanes} == repairs,
             "repair lane entries differ")
    _require(plan["baseline"] == BASELINE, "baseline path changed")
    add(plan["baseline"], plan["baseline_sha256"])
    for (task, key), row in repairs.items():
        path = R + "/artifacts/residual7_no_semantic_mesh_v3_" + task + "_" + key.replace("/", "_") + "/report.json"
        _require(row["probe_report"] == path, "probe path changed")
        add(path, row["probe_sha256"])
    _require(len(pins) == EVIDENCE_COUNT, "evidence allowlist count changed")
    return pins


def _checked_size(source, checked_directories=None):
    p = Path(_path(source))
    checked = checked_directories if checked_directories is not None else set()
    info = p.lstat()
    _require(stat.S_ISREG(info.st_mode), "source is not a regular file: " + source)
    for directory in p.parents:
        if directory in checked:
            break
        _require(stat.S_ISDIR(directory.lstat().st_mode),
                 "source ancestor is not a real directory: " + str(directory))
        checked.add(directory)
    _require(info.st_size > 0, "empty evidence file")
    return info.st_size


def _read_pinned(source, expected):
    _checked_size(source)
    data = Path(source).read_bytes()
    _require(hashlib.sha256(data).hexdigest() == _sha(expected), "anchor hash changed: " + source)
    return data


def _manifest_from_pins(pins, size_of):
    _require(len(pins) == EVIDENCE_COUNT and CHECKPOINT not in pins, "invalid evidence pin count")
    files = []
    for source, digest in sorted(pins.items()):
        _path(source); _sha(digest)
        size = size_of(source)
        _require(type(size) is int and size > 0, "invalid file size")
        files.append(dict(source=source, sha256=digest, size=size, package="evidence"))
    _require(sum(f["size"] for f in files) == EVIDENCE_BYTES, "evidence byte count changed")
    checkpoint_size = size_of(CHECKPOINT)
    _require(type(checkpoint_size) is int and checkpoint_size == CHECKPOINT_BYTES,
             "WA checkpoint size changed")
    files.append(dict(source=CHECKPOINT, sha256=CHECKPOINT_SHA256,
                      size=CHECKPOINT_BYTES, package="weights"))
    return dict(schema=SCHEMA, files=files)


def build_manifest():
    """Read fixed NAS anchors; no override, write, recursion, or large-file hash."""
    blobs = {p: _read_pinned(p, h) for p, h in ANCHORS.items()}
    audit, manifest, plan = (_json(blobs[p]) for p in (AUDIT, MANIFEST, PLAN))
    selections = [_json(line) for line in blobs[SELECTIONS].splitlines()]
    pins = _collect_evidence(audit, manifest, plan, selections)
    # Only the 11 remaining small definitions are read; all 25,290 teacher
    # files and the WA checkpoint are stat-only here, verified by the packer.
    teacher_prefixes = tuple(root + "/" for root in TEACHER_ROOTS)
    for source, digest in pins.items():
        if source not in blobs and not source.startswith(teacher_prefixes):
            _read_pinned(source, digest)
    checked = set()
    return _manifest_from_pins(pins, lambda source: _checked_size(source, checked))
