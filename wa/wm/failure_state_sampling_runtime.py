"""Pinned three-source sampling adapter; no training or source-cache mutation.

This gate is downstream of the actual FailureStateData admission. It maps full
candidate rows to loader indices explicitly, retains old executed positions,
and never promotes collection or cache files to training_released.
"""
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from wa.wm.robot_data import RobotWorldData
from wa.wm.dual_teacher_data import DualTeacherData
from wa.wm.failure_state_data import FailureStateData
from wa.wm.teacher_window_plan import PlannedTeacherMix, canonical_sha, ExposureTaggedData, strip_exposure_tag
from wa.wm.failure_state_sampling import build_plan, simulate_exposure, SOURCES, ARRAY_KEYS, _array_sha
from wa.tools.audit_failure_state_dedup import tensor_digest

SCHEMA = "failure_state_sampling_runtime_candidate_v1"
PROJECT = Path("/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928")
OLD_AUDIT = PROJECT/"artifacts/hard_stt_training_audit_61609_v1.json"
OLD_AUDIT_SHA = "59d5b468eec4283db1a1d87ad7b9a9d1bc2123739cffc38cebfc9a1e14dcc727"
RELEASE_SHA = "73fe7d6b7cdcdfa144a5bcf540cbac33b0d2c0971b25d6aeca35e928d8af0fbd"
FIXED = dict(base=726631, teacher=436816, old_positions=1184273, old_executed=1184272,
             old_teacher_exposures=457641, candidate=7396, valid=6864, episodes=96,
             duplicate=526, new=6338, recovery_budget=49152, omitted_position=1151263)
TENSORS = {"rgb", "template", "polar", "times", "pose", "geometry",
           "wm_rgb", "commands", "proprio", "future"}
STAT_FIELDS = ("st_size", "st_mtime_ns", "st_ino")
STATUS = {"EXACT_DUPLICATE", "NO_NONIMAGE_MATCH", "CONSUMED_IMAGE_DIFFERENCE",
          "NO_SAME_KEY_OLD_EPISODE"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    require(type(value) is str and re.fullmatch("[0-9a-f]{64}", value), "explicit lowercase SHA256 required")
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def strict_json(blob):
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, "duplicate JSON key")
            out[k] = v
        return out
    def invalid(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(blob, object_pairs_hook=pairs, parse_constant=invalid)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for b in iter(lambda: stream.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def path(value, *, directory=False):
    p = Path(value)
    require(p.is_absolute() and not p.is_symlink() and p.resolve() == p,
            "absolute nonsymlink source path required: " + str(p))
    require(p.is_dir() if directory else p.is_file(), "missing source: " + str(p))
    return p


class Reader:
    def __init__(self):
        self.files, self.states = {}, {}

    def read(self, value, expected, *, retain=False):
        p = path(value); digest(expected); name = str(p)
        before = tuple(getattr(p.stat(), k) for k in STAT_FIELDS)
        data = p.read_bytes() if retain else None
        actual = hashlib.sha256(data).hexdigest() if retain else file_sha(p)
        require(actual == expected, "source SHA mismatch: " + name)
        require(tuple(getattr(p.stat(), k) for k in STAT_FIELDS) == before, "source changed while reading")
        require(name not in self.files or self.files[name] == expected, "conflicting source SHA")
        self.files[name], self.states[name] = expected, before
        return data if retain else expected

    def doc(self, value, expected):
        return strict_json(self.read(value, expected, retain=True))

    def unchanged(self):
        for name, old in self.states.items():
            p = path(name)
            require(tuple(getattr(p.stat(), k) for k in STAT_FIELDS) == old, "source changed after validation")


def _arrays_match_dataset(data, cache, complete, reader):
    for suffix, attribute in (("pose", "pose"), ("history", "history"), ("episode", "episode")):
        name = "train_" + suffix + ".npy"
        reader.read(cache/name, complete["files"][name])
        actual = np.load(cache/name, mmap_mode="r", allow_pickle=False)
        require(np.array_equal(actual, getattr(data, attribute)) and actual.dtype == getattr(data, attribute).dtype,
                "dataset arrays differ from pinned cache: " + attribute)
    episodes = reader.doc(cache/"train_episodes.json", complete["files"]["train_episodes.json"])
    require(canonical(episodes) == canonical(data.episodes), "dataset episode inventory changed")


def _old(old_mix, old_plan_root, old_run_root, pins, reader):
    require(type(old_mix) is PlannedTeacherMix, "original PlannedTeacherMix required; no v2 reweighting")
    require(type(old_mix.base) is RobotWorldData and type(old_mix.teacher) is DualTeacherData,
            "admitted original base and teacher datasets required")
    require(len(old_mix.base) == FIXED["base"] and len(old_mix.teacher) == FIXED["teacher"]
            and len(old_mix) == FIXED["old_positions"], "original old source lengths changed")
    root, run = path(old_plan_root, directory=True), path(old_run_root, directory=True)
    anchor = reader.doc(OLD_AUDIT, OLD_AUDIT_SHA)
    require(anchor.get("status") == "TRAINING_ARTIFACT_AUDIT_PASS_OFFLINE_ONLY", "old trusted training audit")
    trusted = anchor["source_hashes"]
    def doc(p):
        require(str(p) in trusted, "old artifact absent from trusted audit: " + str(p))
        return reader.doc(p, trusted[str(p)])
    require(trusted.get(str(root/"plan.json")) == pins["old_plan"], "old plan pin not anchored")
    require(trusted.get(str(run/"actual_exposure_epoch1.npz")) == pins["old_actual_exposure"],
            "old actual exposure pin not anchored")
    plan_doc, report, config = doc(root/"plan.json"), doc(root/"report.json"), doc(run/"config.json")
    require(canonical_sha(plan_doc) == old_mix.plan_sha256 == report["plan_canonical_sha256"]
            and list(old_mix.extra) == plan_doc["extra_teacher_indices"]
            and old_mix.repeats == plan_doc["extra_repeats"],
            "old runtime and persisted plan differ")
    require(report.get("status") == "CANDIDATE_ONLY_NOT_TRAINING_RELEASE", "old plan status")
    for name, h in report["artifacts"].items():
        require(Path(name).name == name and trusted.get(str(root/name)) == h, "old plan artifact trust")
        reader.read(root/name, h)
    base_cache, teacher_cache = path(config["cache"], directory=True), path(config["dual_teacher_cache"], directory=True)
    require(path(old_mix.base.cache, directory=True) == base_cache
            and path(old_mix.teacher.cache, directory=True) == teacher_cache, "old runtime cache paths changed")
    for dataset, cache, index in ((old_mix.base, base_cache, path(config["index_root"], directory=True)),
                                  (old_mix.teacher, teacher_cache, teacher_cache/"index")):
        complete = doc(cache/"complete.json")
        ih = config["index_audit_sha256"] if dataset is old_mix.base else trusted[str(index/"audit.json")]
        ia = reader.doc(index/"audit.json", ih)
        reader.read(index/"train_valid.npy", ia["splits"]["train"]["index_sha256"])
        rows = np.load(index/"train_valid.npy", allow_pickle=False)
        require(rows.dtype == np.int64 and np.array_equal(rows, dataset.rows), "old dataset row mapping changed")
        _arrays_match_dataset(dataset, cache, complete, reader)
    exposure = doc(run/"actual_exposure_epoch1.json")
    require(exposure.get("diagnostic") is False and
            tuple(exposure.get(k) for k in ("world_size", "batch_size", "seed", "epoch")) == (8, 2, 42, 1),
            "old actual sampler identity")
    require(exposure["plan_sha256"] == old_mix.plan_sha256
            and exposure["pre_ddp_total"] == FIXED["old_positions"]
            and exposure["actual_total"] == FIXED["old_executed"]
            and exposure["actual_base"] == FIXED["base"]
            and exposure["actual_teacher"] == FIXED["old_teacher_exposures"], "old actual exposure totals")
    reader.read(run/"actual_exposure_epoch1.npz", pins["old_actual_exposure"])
    with np.load(run/"actual_exposure_epoch1.npz", allow_pickle=False) as z:
        require(set(z.files) == {"position_counts", "base_counts", "teacher_counts"}, "old actual array schema")
        actual = {k: z[k].copy() for k in z.files}
    require(all(v.dtype == np.int64 for v in actual.values()), "old integer count dtype")
    sources = [old_mix.locate(i) for i in range(len(old_mix))]
    require(np.flatnonzero(actual["position_counts"] == 0).tolist() == [FIXED["omitted_position"]],
            "old omitted position changed")
    return sources, actual, dict(plan_root=str(root), run_root=str(run),
        base_cache=str(base_cache), teacher_cache=str(teacher_cache),
        base_index=str(config["index_root"]), old_audit=dict(path=str(OLD_AUDIT), sha256=OLD_AUDIT_SHA))


def _recovery(recovery, dedup_path, pins, reader):
    require(type(recovery) is FailureStateData, "actual FailureStateData admission required")
    require(recovery.admission_sha256 == pins["recovery_admission"]
            and recovery.admission_report.get("status") == "LOADER_ADMISSION_PASS_NOT_NEW_MODEL_RESULT",
            "recovery loader admission differs from caller pin")
    cache = path(recovery.cache, directory=True)
    admission = reader.doc(cache/"admission.json", pins["recovery_admission"])
    require(admission.get("training_released") is False, "cache must not self-release training")
    ref = admission["collection_release"]
    require(ref["sha256"] == RELEASE_SHA == recovery.admission_report["release_sha256"],
            "fixed collection release changed")
    release = reader.doc(ref["path"], RELEASE_SHA)
    files = admission["files"]
    for name in ("complete.json", "index/audit.json", "index/train_valid.npy", "index/train_episodes_audit.json"):
        reader.read(cache/name, files[name])
    complete = reader.doc(cache/"complete.json", files["complete.json"])
    _arrays_match_dataset(recovery, cache, complete, reader)
    rows = np.load(cache/"index/train_valid.npy", allow_pickle=False)
    require(rows.dtype == np.int64 and rows.shape == (FIXED["valid"],) and np.array_equal(rows, recovery.rows)
            and (np.diff(rows) > 0).all() and len(recovery) == FIXED["valid"], "recovery loader rows changed")
    dedup = reader.doc(dedup_path, pins["dedup_report"])
    require(dedup.get("schema") == "failure_state_exact_dedup_v1"
            and dedup.get("status") == "PASS_SCOPED_EXACT_EQUIVALENCE"
            and all(dedup.get(k) is False for k in ("training_released", "sampling_changed", "cache_modified")),
            "dedup must be an unchanged-data diagnostic")
    require(canonical(dedup["release"]) == canonical(ref), "dedup and cache collection release differ")
    demos = release["teacher_demonstrations"]
    require(len(demos) == len(recovery.episodes) == FIXED["episodes"], "recovery episode order/count")
    evidence = dedup["windows"]
    require(len(evidence) == len(rows) and len({w["new_cache_row"] for w in evidence}) == len(rows),
            "dedup complete distinct valid-row coverage")
    byrow = {w["new_cache_row"]: w for w in evidence}
    require(sorted(byrow) == rows.tolist(), "dedup row IDs must exactly equal actual loader rows")
    duplicate = sorted(w["new_cache_row"] for w in evidence if w["status"] == "EXACT_DUPLICATE")
    unique = sorted(w["new_cache_row"] for w in evidence if w["status"] != "EXACT_DUPLICATE")
    require(sorted(dedup["duplicate_new_cache_rows"]) == duplicate
            and sorted(dedup["nonduplicate_new_cache_rows_within_scope"]) == unique
            and len(duplicate) == FIXED["duplicate"] and len(unique) == FIXED["new"], "dedup partition mismatch")
    dataset_for_row = {int(row): i for i, row in enumerate(rows)}
    result, offset, derived_valid = [], 0, []
    for eid, (demo, entry) in enumerate(zip(demos, recovery.episodes)):
        branch = path(demo["branch"], directory=True)
        require(entry["root"] == str(branch) and entry["episode_uid"] == demo["task"] + ":" + demo["key"]
                and entry["teacher"] == demo["teacher"] and entry["takeover_step"] == demo["takeover_step"]
                and entry["template_index"] == 0 and entry["repeat_frames_referenced"] is False,
                "candidate original identity/template/repeat changed")
        for k in ("source_experiment", "source_protocol_sha256", "source_job_id", "source_task_id"):
            require(entry[k] == demo[k], "original source protocol changed")
        oh = demo["hashes"]["observations.json"]
        require(dedup["source_files"].get(str(branch/"observations.json")) == oh, "raw time not dedup-pinned")
        observations = reader.doc(branch/"observations.json", oh)
        require([o["frame"] for o in observations] == entry["frames"], "full episode0 frame sequence changed")
        times = np.asarray([o["timestamp_s"] for o in observations], dtype=np.float64)
        require(np.isfinite(times).all() and (np.diff(times) > 0).all(), "raw actual timestamps")
        k = demo["takeover_step"]; candidates = demo["window_indices"]; count = demo["candidate_windows"]
        require(type(k) is int and 0 <= k < len(times) and len(candidates) == count
                and len(set(candidates)) == count, "candidate index identity")
        sl = slice(offset, offset+count)
        require(np.array_equal(recovery.episode[sl], np.full(count, eid, dtype=np.int32))
                and recovery.history[sl, -1].tolist() == candidates, "candidate raw-row interval/order")
        validset = set(demo["valid_window_indices"])
        require(len(validset) == demo["expected_valid_count"], "valid mask count")
        current_valid = []
        for j, current in enumerate(candidates):
            row = offset+j
            if current not in validset:
                require(row not in byrow, "excluded numeric row entered dedup/runtime")
                continue
            require(row in byrow, "valid numeric row missing from runtime")
            w = byrow[row]; h = recovery.history[row]
            require(w["status"] in STATUS and w["new_episode"] == eid and w["task"] == demo["task"]
                    and w["key"] == demo["key"] and w["source_branch"] == str(branch)
                    and w["source_teacher"] == demo["teacher"] and w["takeover_step"] == k
                    and w["new_current_index"] == current, "dedup original source identity")
            require(current >= k and np.all((h >= 0) & (h <= current)) and h[-1] == current,
                    "causal original-index history")
            require(w["new_actual_timestamp_s"] == float(times[current]), "dedup current raw timestamp")
            require(tensor_digest(np.asarray(recovery.pose[row])) == w["nonimage_components"]["pose"],
                    "all valid cached SE2 pose tensors must equal dedup consumption")
            age = float(times[current] - times[k])
            result.append(dict(dataset_index=dataset_for_row[row], raw_row=row, episode=entry["episode_uid"],
                task=demo["task"], key=demo["key"], teacher=demo["teacher"], current_index=current,
                takeover_step=k, current_age_s=age, valid=True, exact_duplicate=w["status"] == "EXACT_DUPLICATE"))
            current_valid.append(row)
        require(len(current_valid) == demo["expected_valid_count"], "effective mask interval mismatch")
        derived_valid.extend(current_valid); offset += count
    require(offset == FIXED["candidate"] and derived_valid == rows.tolist(), "full candidate/valid mapping")
    return result, dict(cache=str(cache), admission_sha256=pins["recovery_admission"],
        release=ref, dedup_path=str(path(dedup_path)), duplicate=FIXED["duplicate"], new_within_scope=FIXED["new"],
        scope=dedup["scope"], compared_old_cache=dedup["old_cache"], old_trust_anchor=dedup["old_trust_anchor"],
        zero_valid_episodes=[e["episode_uid"] for e in recovery.episodes if e["expected_valid_count"] == 0])


def prepare_candidate(old_mix, recovery, *, old_plan_root, old_run_root, dedup_path, source_pins):
    require(type(source_pins) is dict and set(source_pins) ==
            {"old_plan", "old_actual_exposure", "recovery_admission", "dedup_report"}, "four external pins required")
    for value in source_pins.values(): digest(value)
    reader = Reader()
    sources, actual, old_info = _old(old_mix, old_plan_root, old_run_root, source_pins, reader)
    windows, new_info = _recovery(recovery, dedup_path, source_pins, reader)
    require(new_info["compared_old_cache"]["path"] == old_info["teacher_cache"]
            and new_info["compared_old_cache"]["complete_sha256"] == reader.files[old_info["teacher_cache"]+"/complete.json"]
            and new_info["old_trust_anchor"]["path"] == str(OLD_AUDIT)
            and new_info["old_trust_anchor"]["sha256"] == OLD_AUDIT_SHA, "dedup comparison source differs from retained old teacher")
    plan = build_plan(sources, actual["position_counts"], windows, source_pins=source_pins,
        recovery_budget=FIXED["recovery_budget"], expected_dropped_positions=(FIXED["omitted_position"],))
    for name in ("base", "teacher"):
        require(np.array_equal(plan["arrays"]["old_"+name+"_counts"], actual[name+"_counts"]),
                "old per-window actual exposure mismatch: " + name)
    modules = ("wa.wm.failure_state_sampling_runtime", "wa.wm.failure_state_sampling",
               "wa.wm.teacher_window_plan", "wa.wm.failure_state_data",
               "wa.wm.robot_data", "wa.wm.dual_teacher_data", "wa.tools.audit_failure_state_dedup")
    code = {name: file_sha(Path(importlib.import_module(name).__file__).resolve()) for name in modules}
    reader.unchanged()
    provenance = dict(schema=SCHEMA, source_pins=dict(source_pins), source_files=reader.files,
        old=old_info, recovery=new_info, code_sha256=code,
        mapping_sha256=hashlib.sha256(canonical(plan["metadata"]["recovery_windows"]).encode()).hexdigest(),
        valid_rows=FIXED["valid"], all_valid_pose_hashes_checked=True, age_recomputed_from_raw_times=True,
        old_executed_positions_preserved=True, old_omitted_position=FIXED["omitted_position"],
        training_released=False, source_cache_modified=False, no_new_student_result=True)
    return plan, provenance


def save_candidate(output, plan, provenance):
    root = Path(output)
    require(root.is_absolute() and not root.exists() and not root.is_symlink()
            and root.resolve() == root and PROJECT/"artifacts" in root.parents,
            "fresh exclusive project NAS artifact directory required")
    path(root.parent, directory=True)
    require(provenance["source_pins"] == plan["metadata"]["source_pins"], "candidate source pins")
    simulation = simulate_exposure(plan)
    root.mkdir()
    # Any later failure intentionally leaves a visibly incomplete, non-loadable directory.
    with (root/"positions.npz").open("xb") as stream:
        np.savez_compressed(stream, **plan["arrays"])
    doc = {k: v for k, v in plan.items() if k != "arrays"}
    with (root/"plan.json").open("x") as f:
        f.write(canonical(doc)+"\n")
    with (root/"exposure.json").open("x") as f:
        f.write(canonical(simulation)+"\n")
    admission = dict(schema=SCHEMA, status="CANDIDATE_ONLY_NOT_TRAINING_RELEASE",
        plan_sha256=plan["plan_sha256"], source_pins=provenance["source_pins"], provenance=provenance,
        files={name: file_sha(root/name) for name in ("positions.npz", "plan.json", "exposure.json")},
        training_released=False)
    with (root/"admission.json").open("x") as f:
        f.write(canonical(admission)+"\n")
    return dict(path=str(root), admission_sha256=file_sha(root/"admission.json"),
                plan_sha256=plan["plan_sha256"], simulation=simulation)


class ThreeSourceMix(Dataset):
    def __init__(self, plan, old_mix, recovery, *, source_guard=None):
        self.plan = plan
        self.base, self.teacher, self.recovery = old_mix.base, old_mix.teacher, recovery
        self._old_mix = old_mix
        self.plan_sha256 = plan["plan_sha256"]
        self._guard = source_guard
        self._datasets = (self.base, self.teacher, self.recovery)

    def __len__(self):
        return self.plan["metadata"]["total_positions"]

    def locate(self, index):
        require(type(index) is int or isinstance(index, np.integer) and not isinstance(index, np.bool_),
                "integer plan position")
        if not 0 <= index < len(self): raise IndexError(index)
        arrays = self.plan["arrays"]
        return int(arrays["source_codes"][index]), int(arrays["dataset_indices"][index])

    def __getitem__(self, index):
        if self._guard is not None: self._guard.unchanged()
        code, row = self.locate(index)
        item = self._datasets[code][row]
        require(type(item) is dict and set(item) == TENSORS
                and all(isinstance(v, torch.Tensor) and v.dtype == torch.float32 for v in item.values()),
                "policy receives exactly ten float32 tensors, no metadata")
        return item


def load_candidate(directory, expected_admission_sha256, old_mix, recovery, *,
                   old_plan_root, old_run_root, dedup_path, source_pins):
    root = path(directory, directory=True); reader = Reader()
    require(sorted(p.name for p in root.iterdir()) ==
            ["admission.json", "exposure.json", "plan.json", "positions.npz"], "exact candidate file inventory")
    admission = reader.doc(root/"admission.json", digest(expected_admission_sha256))
    require(admission.get("schema") == SCHEMA and admission.get("status") == "CANDIDATE_ONLY_NOT_TRAINING_RELEASE"
            and admission.get("training_released") is False and admission["source_pins"] == source_pins,
            "candidate external identity/release gate")
    require(set(admission["files"]) == {"plan.json", "positions.npz", "exposure.json"}, "candidate file graph")
    for name, h in admission["files"].items(): reader.read(root/name, h)
    stored = reader.doc(root/"plan.json", admission["files"]["plan.json"])
    with np.load(root/"positions.npz", allow_pickle=False) as z:
        require(set(z.files) == ARRAY_KEYS, "candidate array schema")
        stored["arrays"] = {k: z[k].copy() for k in z.files}
    rebuilt, provenance = prepare_candidate(old_mix, recovery, old_plan_root=old_plan_root,
        old_run_root=old_run_root, dedup_path=dedup_path, source_pins=source_pins)
    require(stored["plan_sha256"] == admission["plan_sha256"] == rebuilt["plan_sha256"], "plan identity changed")
    require(canonical({k:v for k,v in stored.items() if k!="arrays"}) ==
            canonical({k:v for k,v in rebuilt.items() if k!="arrays"}), "candidate manifest changed")
    for name, value in rebuilt["arrays"].items():
        require(stored["arrays"][name].dtype == value.dtype and np.array_equal(stored["arrays"][name], value),
                "candidate position or count mapping changed: " + name)
    require(canonical(admission["provenance"]) == canonical(provenance), "runtime source provenance changed")
    simulation = reader.doc(root/"exposure.json", admission["files"]["exposure.json"])
    require(simulation.get("status") == "EXACT_INDEX_SIMULATION_PASS_NOT_TRAINING"
            and simulation["plan_sha256"] == rebuilt["plan_sha256"] and simulation["dropped"] == 0
            and simulation["actual_total"] == len(rebuilt["arrays"]["source_codes"])
            and (simulation["seed"],simulation["epoch"],simulation["world_size"],simulation["batch_size"]) == (42,1,8,2),
            "actual sampler/loader simulation missing")
    reader.unchanged()
    return ThreeSourceMix(rebuilt, old_mix, recovery, source_guard=reader), provenance, simulation


class RuntimeExposure:
    """API-compatible rank-local CPU bookkeeping; tags never enter the policy."""
    def __init__(self, mix):
        require(type(mix) is ThreeSourceMix, "three-source runtime required")
        self.mix = mix
        self.context_sha256 = hashlib.sha256(canonical(dict(schema=SCHEMA, plan=mix.plan_sha256)).encode()).hexdigest()
        self.position_counts = torch.zeros(len(mix), dtype=torch.int64)

    def validate_contexts(self, contexts):
        require(isinstance(contexts, (list,tuple)) and bool(contexts)
                and all(value == self.context_sha256 for value in contexts), "rank plan contexts differ")

    def consume(self, indices):
        require(isinstance(indices, torch.Tensor) and indices.dtype == torch.int64
                and indices.ndim == 1 and indices.device.type == "cpu"
                and bool(((indices >= 0) & (indices < len(self.mix))).all()), "CPU exposure positions required")
        require(torch.unique(indices).numel() == indices.numel()
                and bool((self.position_counts[indices] == 0).all()), "repeated rank-local plan position")
        self.position_counts.index_add_(0, indices, torch.ones_like(indices))

    def state(self):
        return {"position_counts": self.position_counts.clone()}

    def _counts(self, state):
        require(type(state) is dict and set(state) == {"position_counts"}, "exposure state schema")
        c = state["position_counts"]
        require(isinstance(c,torch.Tensor) and c.dtype == torch.int64 and c.device.type == "cpu"
                and c.shape == self.position_counts.shape and bool((c>=0).all()), "integer CPU exposure counts")
        return c

    def source_counts(self, state):
        c = self._counts(state)
        a = self.mix.plan["arrays"]
        out = []
        for name, code in SOURCES.items():
            selected = np.flatnonzero(a["source_codes"] == code)
            values = torch.zeros(len(self.mix._datasets[code]),dtype=torch.int64)
            values.index_add_(0,torch.tensor(a["dataset_indices"][selected].copy()),c[selected])
            out.append(values)
        return tuple(out)

    def summarize(self, state):
        c = self._counts(state)
        source_counts = self.source_counts(state)
        sources = {name:dict(exposures=int(values.sum()),unique=int((values>0).sum()),
            counts_sha256=_array_sha(values.numpy())) for name,values in zip(SOURCES,source_counts)}
        ep, layer = Counter(), Counter()
        for row in self.mix.plan["metadata"]["recovery_windows"]:
            count = int(source_counts[2][row["dataset_index"]])
            ep[row["episode"]] += count
            layer["early" if row["current_age_s"] <= 1.+1e-8 else "late"] += count
        return dict(schema=SCHEMA,context_sha256=self.context_sha256,plan_sha256=self.mix.plan_sha256,
            status="PARTIAL_EXPOSURE_NOT_FINAL",pre_ddp_total=len(self.mix),actual_total=int(c.sum()),
            actual_base=sources["base"]["exposures"],actual_teacher=sources["teacher"]["exposures"],
            actual_recovery=sources["recovery"]["exposures"],unconsumed_positions=int((c==0).sum()),
            repeated_positions=int((c>1).sum()),sources=sources,recovery_per_episode=dict(ep),
            recovery_early_late=dict(layer),metadata_entered_policy=False,
            scope="Actual exposure counts only; not offline quality, closed-loop SR or generalization")

    def finalize(self, *, state, world, batch, seed, epoch):
        require((world,batch,seed,epoch)==(8,2,42,1), "formal exposure recipe changed")
        c = self._counts(state)
        require(bool((c==1).all()), "complete actual all-reduced exposure required")
        source_counts = self.source_counts(state)
        a = self.mix.plan["arrays"]
        for name, actual in zip(("base","teacher"),source_counts):
            require(np.array_equal(actual.numpy(),a["old_"+name+"_counts"]),
                    "actual old per-window exposures differ: "+name)
        for local,row in enumerate(self.mix.plan["metadata"]["recovery_windows"]):
            require(int(source_counts[2][row["dataset_index"]]) == int(a["recovery_counts"][local]),
                    "actual recovery exposure differs from capped plan")
        for name,value in a.items():
            require(_array_sha(value)==self.mix.plan["array_sha256"][name], "plan changed during training")
        report = self.summarize(state)
        report.update(status="ACTUAL_THREE_SOURCE_EXPOSURE_MATCHES_PLAN",
            world_size=world,batch_size=batch,seed=seed,epoch=epoch,dropped=0,
            old_per_window_counts_exact=True,closed_loop_success_not_established=True)
        return report


def diagnostic_positions(mix, records, windows, count=16):
    """Map old coverage plus recovery time/prefix boundaries to plan positions."""
    from wa.wm.teacher_plan_runtime import diagnostic_positions as old_diagnostics
    require(type(count) is int and count>=16 and count%2==0 and count<=len(mix),
            "even diagnostic count >=16 within plan length")
    old = mix.plan["arrays"]["old_plan_positions"]
    n = mix.plan["metadata"]["old_retained_positions"]
    positions = []
    # Reconstruct no dataset: old diagnostic contract only uses old source lengths
    # and locate/extra fields, all exposed by a minimal original-mix reference.
    original = mix._old_mix
    for position in old_diagnostics(original,records,windows,8):
        mapped = int(np.searchsorted(old[:n],position))
        if mapped<n and old[mapped]==position and mapped not in positions:
            positions.append(mapped)
    first = {}
    a = mix.plan["arrays"]
    for position in range(n,len(mix)):
        first.setdefault(int(a["dataset_indices"][position]),position)
    candidates = [r for r in mix.plan["metadata"]["recovery_windows"] if r["valid"] and not r["exact_duplicate"]]
    predicates = (
        lambda r:r["current_age_s"]<=1.+1e-8,
        lambda r:r["current_age_s"]>1.+1e-8,
        lambda r:int(mix.recovery.history[r["raw_row"],0])<r["takeover_step"],
        lambda r:int(mix.recovery.history[r["raw_row"],0])>=r["takeover_step"],
        lambda r:r["current_index"]-3<r["takeover_step"])
    for predicate in predicates:
        choices = [r for r in candidates if predicate(r)]
        if choices:
            row = min(choices,key=lambda r:(r["current_age_s"],r["dataset_index"]))
            position = first[row["dataset_index"]]
            if position not in positions: positions.append(position)
    for position in np.linspace(0,len(mix)-1,count*3,dtype=int).tolist():
        if position not in positions: positions.append(position)
        if len(positions)>=count: break
    require(len(positions)>=count, "not enough distinct diagnostic positions")
    return positions[:count]
