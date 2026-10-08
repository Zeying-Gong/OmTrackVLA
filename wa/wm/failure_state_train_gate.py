"""Read-only external-SHA gate for a completed real loader-audit report.

This verifies the existing report and its current producer code identities. It
does not rerun getitems, decode images, scan the sixteen-file graph, or release
training. FailureStateData must still admit the actual cache independently.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import stat

SCHEMA = "failure_state_real_loader_audit_v1"
STATUS = "CPU_LOADER_AUDIT_PASS_NOT_TRAINING_OR_MODEL_SR"
EXPECTED = dict(completed_searches=126, accepted_original_episodes=96,
                candidate_windows=7396, valid_windows=6864, excluded_windows=532,
                episodes_with_valid_windows=90)
SHAPES = dict(rgb=[4,3,224,224], template=[3,224,224], polar=[2],
              times=[4], pose=[7,4], geometry=[3], wm_rgb=[4,3,224,224],
              commands=[4,4], proprio=[4,4], future=[3,224,224])
TOOL_SOURCES = (
    "wa/tools/audit_failure_state_loader.py", "wa/wm/failure_state_data.py",
    "wa/wm/robot_data.py", "wa/data.py", "wa/wm/training.py")
NONPOLICY_FIELDS = ["pose","geometry","wm_rgb","commands","proprio","future"]
TOP_FIELDS = set(("schema status started_utc finished_utc cache admission_sha256 "
    "loader_admission tool_sources_sha256 runtime selection shapes dtype "
    "pose_cache_comparison raw_pose_max_absolute_tolerance episodes windows "
    "condition_routing timing stat_gate_probe per_getitem_stat_scope boundaries "
    "training_released no_new_model_success_rate").split())
WINDOW_FIELDS = set(("position cache_row episode now history jepa_indices "
    "proprio_indices reasons prefix_crossing max_absolute_errors pose_tensor_sha256 "
    "template_tensor_sha256 current_polar getitem_seconds independent_verification_seconds").split())
EP_COMMON = {"episode","episode_uid","valid_windows","selected_positions"}
EP_NONZERO = EP_COMMON | set(("root teacher takeover_step first_position early_position "
    "tail_position early_definition early_elapsed_after_takeover_s representative_unique_count").split())
SELECTION_RULE = "all prefix-crossing valid rows plus first/second/tail per nonzero episode"


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _same(actual, expected, label):
    _need(json.dumps(actual, sort_keys=True, allow_nan=False) ==
          json.dumps(expected, sort_keys=True, allow_nan=False), label)


def _integer(value, label, minimum=0):
    _need(type(value) is int and value >= minimum, label + ": integer")
    return value


def _number(value, label, minimum=0.):
    _need(type(value) in (int,float) and math.isfinite(value) and value >= minimum,
          label + ": finite number")
    return value


def _digest(value):
    _need(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
          "canonical external SHA256 required")
    return value


def _canonical_path(value, *, directory=False):
    _need(isinstance(value, (str,Path)) and str(value), "explicit absolute path required")
    p = Path(value)
    _need(p.is_absolute() and not p.is_symlink() and p.resolve() == p,
          "absolute nonsymlink path required")
    _need(p.is_dir() if directory else p.is_file(), "missing directory/file: " + str(p))
    return p


def _state(path):
    s = path.stat()
    _need(stat.S_ISREG(s.st_mode), "regular evidence file required")
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def _read(path, expected, states):
    path = _canonical_path(path)
    before = _state(path)
    blob = path.read_bytes()
    actual = hashlib.sha256(blob).hexdigest()
    _need(_state(path) == before and not path.is_symlink() and path.resolve() == path,
          "evidence changed during read")
    _need(actual == _digest(expected), "evidence SHA mismatch: " + str(path))
    states[path] = before
    return blob


def _json(blob):
    def pairs(items):
        result = {}
        for key,value in items:
            _need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite JSON: " + value)
    doc = json.loads(blob, object_pairs_hook=pairs, parse_constant=invalid)
    def visit(value):
        if type(value) is float:
            _need(math.isfinite(value), "nonfinite JSON number")
        elif isinstance(value, dict):
            for item in value.values(): visit(item)
        elif isinstance(value, list):
            for item in value: visit(item)
    visit(doc)
    return doc


def _source_root():
    return Path(__file__).resolve().parents[2]


def _verify_sources(pins, states):
    _need(type(pins) is dict and len(pins) == 5, "exact five audit tool sources required")
    mapped, roots = {}, set()
    for name,pin in pins.items():
        _digest(pin)
        _need(type(name) is str, "source path string")
        path = Path(name)
        _need(path.is_absolute() and str(path) == name and ".." not in path.parts,
              "canonical absolute recorded source path")
        matches = [relative for relative in TOOL_SOURCES
                   if tuple(path.parts[-len(Path(relative).parts):]) == Path(relative).parts]
        _need(len(matches) == 1 and matches[0] not in mapped, "unique relative tool source required")
        relative = matches[0]
        mapped[relative] = pin
        roots.add(path.parents[len(Path(relative).parts)-1])
    _need(set(mapped) == set(TOOL_SOURCES) and len(roots) == 1,
          "five relative audit sources must share one recorded checkout")
    for relative,pin in mapped.items():
        _read(_source_root()/relative, pin, states)
    return mapped


def _coverage(doc):
    episodes,windows = doc["episodes"],doc["windows"]
    _need(type(episodes) is list and len(episodes) == 96 and
          type(windows) is list and 0 < len(windows) <= 6864, "episode/window coverage")
    position_rows = {}
    cache_rows = []
    for row in windows:
        _need(type(row) is dict and set(row) == WINDOW_FIELDS, "window schema")
        pos = _integer(row["position"], "position")
        _need(pos < 6864 and pos not in position_rows, "unique valid dataset position")
        raw = _integer(row["cache_row"], "cache row")
        _need(raw < 7396, "cache row range")
        _integer(row["episode"], "window episode")
        now = _integer(row["now"], "now", 4)
        for field in ("history","jepa_indices","proprio_indices"):
            seq = row[field]
            _need(type(seq) is list and len(seq) == 4, field + " shape")
            for index in seq:
                _integer(index, field)
                _need(index <= now, "future observation in causal index")
        _need(row["history"] == sorted(row["history"]) and row["history"][-1] == now,
              "causal sparse history")
        _same(row["jepa_indices"], list(range(now-3,now+1)), "continuous JEPA history")
        _same(row["proprio_indices"], list(range(now-4,now)), "preceding command history")
        _need(type(row["reasons"]) is list and type(row["prefix_crossing"]) is list,
              "selection reason lists")
        _need(type(row["max_absolute_errors"]) is dict and
              set(row["max_absolute_errors"]) == set(SHAPES), "ten tensor comparison results")
        for field,error in row["max_absolute_errors"].items():
            _number(error, "tensor error")
            _need(error <= (2e-6 if field == "pose" else 0.), "tensor error exceeds audit contract")
        for field in ("pose_tensor_sha256","template_tensor_sha256"):
            _digest(row[field])
        polar = row["current_polar"]
        _need(type(polar) is list and len(polar) == 2, "current polar shape")
        _number(polar[0], "current range")
        _need(type(polar[1]) in (int,float) and math.isfinite(polar[1])
              and abs(polar[1]) <= math.pi + 1e-6, "current bearing")
        for field in ("getitem_seconds","independent_verification_seconds"):
            _number(row[field], field)
        position_rows[pos] = row
        cache_rows.append(raw)
    _need(list(position_rows) == sorted(position_rows), "selected positions must be ordered")
    _need(cache_rows == sorted(set(cache_rows)), "selected cache rows must be unique ordered")
    offset,nonzero = 0,[]
    all_positions,all_ids,roots = [],set(),set()
    for ep,entry in enumerate(episodes):
        _need(type(entry) is dict, "episode entry")
        _same(entry.get("episode"), ep, "ordered complete episode index")
        uid = entry.get("episode_uid")
        _need(type(uid) is str and uid.startswith("stt:") and uid not in all_ids,
              "unique STT episode UID")
        all_ids.add(uid)
        count = _integer(entry.get("valid_windows"), "valid windows")
        selected = entry.get("selected_positions")
        _need(type(selected) is list, "episode selected positions")
        for pos in selected:
            _integer(pos, "episode selected position")
        _need(selected == sorted(set(selected)), "unique ordered episode positions")
        if not count:
            _need(set(entry) == EP_COMMON and selected == [], "zero-valid episode must remain empty")
            continue
        _need(set(entry) == EP_NONZERO, "nonzero episode schema")
        nonzero.append(ep)
        k = _integer(entry["takeover_step"], "takeover")
        _need(entry["teacher"] in ("lightnav","oracle"), "selected teacher")
        root = entry["root"]
        _need(type(root) is str and Path(root).is_absolute() and root not in roots,
              "unique original winner root")
        roots.add(root)
        first,early,tail = offset,offset+min(1,count-1),offset+count-1
        for name,value in (("first_position",first),("early_position",early),("tail_position",tail)):
            _same(entry[name], value, "chronological representative: " + name)
        _same(entry["representative_unique_count"], len({first,early,tail}), "representative count")
        _same(entry["early_definition"], "second chronological valid window", "early definition")
        _number(entry["early_elapsed_after_takeover_s"], "early elapsed")
        _need({first,early,tail} <= set(selected), "missing first/second/tail coverage")
        _need(all(offset <= pos < offset+count and pos in position_rows for pos in selected),
              "episode selected position range/coverage")
        last_now = -1
        for pos in selected:
            row = position_rows[pos]
            _same(row["episode"], ep, "selected window episode")
            now = row["now"]
            _need(now >= k and now > last_now, "teacher-owned chronological current index")
            last_now = now
            crossing = [name for name,field in
                (("policy","history"),("jepa_command","jepa_indices"),("proprio","proprio_indices"))
                if any(index < k for index in row[field])]
            reasons = [name for name,value in
                       (("first",first),("early_second",early),("tail",tail)) if pos == value]
            if crossing: reasons.append("student_prefix")
            _same(row["prefix_crossing"], crossing, "prefix crossing evidence")
            _same(row["reasons"], reasons, "exact selection reasons")
            _need(bool(reasons), "unexplained selected window")
        all_positions.extend(selected)
        offset += count
    _need(offset == 6864 and len(nonzero) == 90 and len(episodes)-len(nonzero) == 6,
          "fixed valid/nonzero/zero counts")
    _same(all_positions, list(position_rows), "bidirectional selected window coverage")
    expected_selection = dict(rule=SELECTION_RULE, all_valid_rows_inspected=6864,
        nonzero_episodes=90, zero_valid_episodes=6, selected_unique_windows=len(windows),
        all_prefix_crossing_windows=sum(bool(row["prefix_crossing"]) for row in windows),
        prefix_crossing_by_input=dict(Counter(name for row in windows for name in row["prefix_crossing"])))
    _same(doc["selection"], expected_selection, "selection summary disagrees with evidence")
    routing = doc["condition_routing"]
    _need(type(routing) is list and len(routing) == 90, "condition routing coverage")
    for row,ep in zip(routing,nonzero):
        _same(row, dict(episode=ep,
            status="ACTUAL_MAKE_CONDITIONS_ROUTING_PASS_CPU_ENCODER_STANDIN",
            loaded_encoder_weights=False,loaded_policy_weights=False,mode="mixed",
            perturbed_nonpolicy_fields=NONPOLICY_FIELDS), "actual condition routing proof")


def verify_loader_audit(report_path, expected_sha, cache, admission_sha):
    """Verify immutable identities and report evidence; return a nonrelease dict.

    Signature: verify_loader_audit(report_path, expected_sha, cache, admission_sha).
    The returned report is unchanged. Source paths may have a different recorded
    checkout root after freezing, but each exact wa-relative source must match
    the bytes in this calling checkout. No source root is supplied by callers.
    """
    states = {}
    cache = _canonical_path(cache, directory=True)
    report = _json(_read(report_path, expected_sha, states))
    admission = _json(_read(cache/"admission.json", admission_sha, states))
    _need(type(report) is dict and set(report) == TOP_FIELDS, "complete audit report schema")
    for field,value in dict(schema=SCHEMA,status=STATUS,cache=str(cache),
            admission_sha256=admission_sha,dtype="torch.float32",
            pose_cache_comparison="bit-exact",raw_pose_max_absolute_tolerance=2e-6,
            training_released=False,no_new_model_success_rate=True).items():
        _same(report[field], value, "audit field: " + field)
    _same(report["shapes"], SHAPES, "exact ten tensor shapes")
    times = []
    for field in ("started_utc","finished_utc"):
        _need(type(report[field]) is str, "audit timestamp")
        value = datetime.fromisoformat(report[field])
        _need(value.utcoffset() is not None, "timezone-aware audit timestamp")
        times.append(value)
    _need(times[1] >= times[0], "audit completion precedes start")
    for field,value in dict(schema="failure_state_cache_v1_admission_candidate",
            status="CACHE_CONVERTED_NOT_TRAINING_RELEASED",training_released=False,
            expected_admission_sha_required_by_loader=True,independent_audit_required=True,
            summary=EXPECTED).items():
        _same(admission.get(field), value, "actual admission field: " + field)
    nested = report["loader_admission"]
    _need(type(nested) is dict and set(nested) == set(("status summary admission_sha256 "
        "release_sha256 actual_consumed_sources_hashed consumed_source_files "
        "unconsumed_historical_weights_rehashed input_scope training_released_on_disk").split()),
        "loader admission report schema")
    for field,value in dict(status="LOADER_ADMISSION_PASS_NOT_NEW_MODEL_RESULT",
            summary=EXPECTED,admission_sha256=admission_sha,actual_consumed_sources_hashed=True,
            unconsumed_historical_weights_rehashed=False,training_released_on_disk=False,
            input_scope="RGB/episode-zero BBox/current ideal zero-delay simulated polar UWB; no text").items():
        _same(nested[field], value, "loader admission field: " + field)
    _digest(nested["release_sha256"])
    _integer(nested["consumed_source_files"], "consumed sources", 1)
    release = admission.get("collection_release")
    _need(type(release) is dict and release.get("sha256") == nested["release_sha256"],
          "actual admission collection-release binding")
    runtime = report["runtime"]
    _need(type(runtime) is dict and set(runtime) ==
          {"python","numpy","torch","torch_cpu_threads","cuda_used"}, "audit runtime schema")
    _same(runtime["cuda_used"], False, "CPU-only audit runtime")
    _integer(runtime["torch_cpu_threads"], "CPU threads", 1)
    for field in ("python","numpy","torch"):
        _need(type(runtime[field]) is str and bool(runtime[field]), "runtime version string")
    _coverage(report)
    source_pins = _verify_sources(report["tool_sources_sha256"], states)
    for path,before in states.items():
        _need(not path.is_symlink() and path.resolve() == path and _state(path) == before,
              "audit evidence changed before verification finished: " + str(path))
    return dict(report=report,report_sha256=expected_sha,admission_sha256=admission_sha,
                tool_sources_by_relative_path=source_pins,training_released=False)
