"""Convert admitted failure-state original winners; no cache/training release.

Preserves the full episode-zero frame sequence and the original seven-step
SE2/four-sparse-history cache contract. Invalid candidate rows remain auditable
but only effective-mask rows appear in train_valid.npy. A separate admission
auditor and loader with an explicit admission SHA are still required.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from wa.wm.failure_state_numeric import audit_numeric_windows
from wa.wm.robot_data import CONTRACT, transition_records
from wa.wm.failure_state_protocol import EXPERIMENT as V1, PROTOCOL_SHA as P1
from wa.wm.failure_state_protocol_v2 import EXPERIMENT as V2, PROTOCOL_SHA as P2

SCHEMA = "failure_state_cache_v1"
RELEASE_SCHEMA = "failure_state_collection_release_v1"
NAS = Path("/data/nas_ray")
EXPECTED = dict(completed_searches=126, accepted_original_episodes=96,
                candidate_windows=7396, valid_windows=6864,
                excluded_windows=532, episodes_with_valid_windows=90)
SUFFIXES = ("pose.npy", "history.npy", "episode.npy", "episodes.json")
SOURCE_JOBS = {(61833, 73055): (V1, P1), (61836, 73058): (V2, P2),
               (61844, 73066): (V2, P2)}
REQUIRED = {"metadata.json", "observations.json", "actions.json", "windows.json",
            "result.json", "branch.json", "admission.json", "complete.json",
            "replay.json", "first_start_pair.json", "takeover_pair.json"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(obj):
    return json.dumps(obj, sort_keys=True, allow_nan=False, separators=(",", ":"))


def same(a, b, message):
    require(canonical(a) == canonical(b), message)


def strict_json(data):
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, "duplicate JSON key")
            out[k] = v
        return out
    def invalid(x):
        raise ValueError("nonfinite JSON: " + x)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def path_file(path):
    p = Path(path)
    require(p.is_absolute() and p.is_file() and not p.is_symlink()
            and p.resolve() == p, "absolute nonsymlink source required: " + str(p))
    return p


def path_dir(path):
    p = Path(path)
    require(p.is_absolute() and p.is_dir() and not p.is_symlink()
            and p.resolve() == p, "absolute nonsymlink directory required")
    return p


class Pins:
    def __init__(self):
        self.files, self.states = {}, {}

    def read(self, path, expected=None, *, retain=False):
        p = path_file(path)
        if expected is not None:
            require(type(expected) is str and re.fullmatch("[0-9a-f]{64}", expected),
                    "invalid expected SHA")
        before = p.stat()
        state = (before.st_size, before.st_mtime_ns, before.st_ino)
        h, chunks = hashlib.sha256(), []
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
                if retain:
                    chunks.append(chunk)
        after = p.stat()
        require(state == (after.st_size, after.st_mtime_ns, after.st_ino),
                "source changed while reading")
        digest = h.hexdigest()
        require(expected is None or digest == expected, "source SHA mismatch: " + str(p))
        key = str(p)
        require(key not in self.files or self.files[key] == digest, "conflicting source SHA")
        require(key not in self.states or self.states[key] == state, "source changed during conversion")
        self.files[key], self.states[key] = digest, state
        return b"".join(chunks) if retain else digest

    def doc(self, path, expected=None):
        return strict_json(self.read(path, expected, retain=True))

    def inventory(self, files):
        require(isinstance(files, dict) and bool(files), "nonempty SHA inventory required")
        for p, h in files.items():
            self.read(p, h)

    def finish(self):
        for name, state in self.states.items():
            p = path_file(name)
            s = p.stat()
            require(state == (s.st_size, s.st_mtime_ns, s.st_ino),
                    "source changed before finalization")


def flag(v, name):
    require(isinstance(v, (bool, int, float)) and v in (0, 1), "invalid binary " + name)
    return bool(v)


def successful(result):
    require(isinstance(result, dict) and flag(result.get("success"), "success")
            and not flag(result.get("collision"), "collision")
            and flag(result.get("policy_init_valid"), "initialization"),
            "unsuccessful or invalid demonstration")


def prepare_release(path, digest, pins):
    release = pins.doc(path, digest)
    require(release.get("schema") == RELEASE_SCHEMA, "wrong collection release schema")
    for name in ("collection_validated", "evaluation_adaptation", "no_success_rate",
                 "cache_conversion_required"):
        require(release.get(name) is True, "missing release condition: " + name)
    for name in ("training_released", "untouched_test", "score_backfill_allowed"):
        require(release.get(name) is False, "unsafe release condition: " + name)
    require(type(release.get("expected")) is int
            and release["expected"] == EXPECTED["completed_searches"]
            and release.get("completed_searches") == EXPECTED["completed_searches"],
            "full fixed collection required")
    for k, v in EXPECTED.items():
        if k != "completed_searches":
            require(type(release.get("summary", {}).get(k)) is int
                    and release["summary"][k] == v, "unexpected release total: " + k)
    pins.inventory(release.get("source_files"))
    entries = release.get("teacher_demonstrations")
    require(isinstance(entries, list) and len(entries) == EXPECTED["accepted_original_episodes"],
            "fixed original winner count required")
    keys, roots, repeats = set(), set(), set()
    for e in entries:
        require(e.get("task") == "stt" and type(e.get("key")) is str, "invalid STT identity")
        key = (e["task"], e["key"])
        require(key not in keys, "duplicate demonstration key"); keys.add(key)
        root, repeat = path_dir(e["branch"]), path_dir(e["repeat_branch"])
        require(root != repeat and root.parent == repeat.parent, "invalid repeat branch")
        require(str(root) not in roots and str(repeat) not in repeats, "duplicate original/repeat")
        roots.add(str(root)); repeats.add(str(repeat))
        require(e.get("teacher") in ("lightnav", "oracle"), "unknown teacher")
        require(type(e.get("takeover_step")) is int and e["takeover_step"] >= 0, "bad takeover")
        pair = (e.get("source_job_id"), e.get("source_task_id"))
        require(pair in SOURCE_JOBS, "unknown source collection")
        same((e.get("source_experiment"), e.get("source_protocol_sha256")),
             SOURCE_JOBS[pair], "source protocol relabeling")
    require(not roots & repeats, "repeat admitted as original demonstration")
    return release


def branch_data(entry, release, pins):
    root = path_dir(entry["branch"])
    hashes = entry.get("hashes")
    require(isinstance(hashes, dict) and REQUIRED <= set(hashes), "incomplete original evidence")
    for name, digest in hashes.items():
        rel = Path(name)
        require(type(name) is str and not rel.is_absolute() and rel.parts
                and "." not in rel.parts and ".." not in rel.parts, "unsafe artifact relative path")
        p = path_file(root/rel)
        require(p.is_relative_to(root), "artifact escape")
        same(release["source_files"].get(str(p)), digest, "artifact missing from release inventory")
        pins.read(p, digest)
    actual = set()
    for p in root.rglob("*"):
        require(not p.is_symlink() and p.resolve() == p, "symlink inside branch")
        if p.is_file():
            actual.add(str(p.relative_to(root)))
    same(sorted(actual), sorted(hashes), "unmanifested/missing original branch artifacts")

    def doc(name):
        return pins.doc(root/name, hashes[name])
    meta, obs, actions, windows = (doc(n) for n in
                                 ("metadata.json", "observations.json", "actions.json", "windows.json"))
    result, branch, admission, complete = (doc(n) for n in
                                         ("result.json", "branch.json", "admission.json", "complete.json"))
    identity = dict(experiment=entry["source_experiment"],
                    protocol_sha256=entry["source_protocol_sha256"], task=entry["task"],
                    key=entry["key"], teacher=entry["teacher"], takeover_step=entry["takeover_step"])
    for record in (meta, branch, admission):
        for k, v in identity.items():
            same(record.get(k), v, "branch identity " + k)
        require(record.get("verification_only") is False, "repeat is not a demonstration")
    require(meta.get("partition") == "evaluation_adaptation"
            and meta.get("training_eligible") is False, "unmarked/admitted raw branch")
    require(meta.get("camera_alignment_verified") is True and meta.get("initial_bbox_status")
            in ("VERIFIED_CONFIG_AND_SEMANTIC", "VERIFIED_FROZEN_FIRST_RGB_REPAIR"),
            "unverified episode-zero template")
    require(meta.get("timebase_version") == "v3_actual_world_time_interpolated",
            "actual-time source required")
    shape = np.asarray(meta.get("rgb_shape"))
    box = np.asarray(meta.get("initial_bbox_rgb_xyxy"))
    require(shape.shape == (3,) and shape.dtype.kind in "iu" and (shape > 0).all()
            and box.shape == (4,) and box.dtype.kind in "iuf" and np.isfinite(box).all()
            and 0 <= box[0] < box[2] <= shape[1] and 0 <= box[1] < box[3] <= shape[0],
            "invalid template bbox")
    successful(result)
    for record in (branch, admission):
        require(record.get("complete") is True and record.get("replay_verified") is True
                and record.get("transport_fallback") is False, "incomplete/fallback branch")
        same(record.get("result"), result, "terminal result disagreement")
    same(branch.get("artifact_root"), str(root), "original root mismatch")
    for record in (admission, complete):
        require(record.get("training_eligible") is False
                and record.get("training_released") is False, "premature raw release")
    require(complete.get("complete") is True
            and complete.get("status") == "FAILURE_STATE_RAW_COLLECTION_COMPLETE",
            "missing raw completion")
    require(isinstance(obs, list) and len(obs) >= 2 and complete.get("frames") == len(obs),
            "frame count mismatch")
    frames = [o["frame"] for o in obs]
    require(len(frames) == len(set(frames)), "duplicate observation frames")
    inventory = {}
    for frame in frames:
        require(type(frame) is str and Path(frame).name == frame and frame.endswith(".png"),
                "unsafe frame path")
        p = path_file(root/frame)
        require(frame in hashes, "observation image not audited")
        inventory[frame] = pins.read(p, hashes[frame])
    target = np.asarray([o.get("target_position_world_label_only") for o in obs])
    require(target.shape == (len(obs), 3) and target.dtype.kind in "iuf"
            and np.isfinite(target).all(), "current simulated UWB source missing/nonfinite")
    numeric = audit_numeric_windows(obs, actions, windows, entry["takeover_step"], per_window=False)
    same([w["current_index"] for w in windows], entry.get("window_indices"), "candidate index mismatch")
    same(numeric["valid_window_indices"], entry.get("valid_window_indices"), "effective index mismatch")
    same(numeric["excluded"], entry.get("numeric_exclusions"), "effective exclusions mismatch")
    for key, expected_key in (("candidate_windows", "candidate_windows"),
                              ("valid_windows", "expected_valid_count"),
                              ("excluded_windows", "expected_excluded_count")):
        same(numeric["summary"][key], entry.get(expected_key), "numeric count mismatch")
    commands, bad, stats = transition_records(root, obs)
    require(np.array_equal(commands, numeric["derived"]["commands"])
            and np.array_equal(bad, numeric["derived"]["transition_bad"]), "runtime transition mismatch")
    stats.update(numeric["summary"])
    return meta, obs, frames, inventory, numeric, stats


def prepare_base(base, complete_sha, index, audit_sha, pins):
    base, index = path_dir(base), path_dir(index)
    complete = pins.doc(base/"complete.json", complete_sha)
    audit = pins.doc(index/"audit.json", audit_sha)
    require(audit.get("contract") == CONTRACT and audit.get("cache_complete_sha256") == complete_sha,
            "base index/cache contract mismatch")
    files = complete.get("files", {})
    for suffix in SUFFIXES:
        name = "heldout_" + suffix
        require(name in files, "base heldout file not pinned")
        pins.read(base/name, files[name])
    heldout = pins.doc(base/"heldout_episodes.json", files["heldout_episodes.json"])
    require(isinstance(heldout, list) and bool(heldout), "empty original heldout")
    pose = np.load(base/"heldout_pose.npy", mmap_mode="r", allow_pickle=False)
    history = np.load(base/"heldout_history.npy", mmap_mode="r", allow_pickle=False)
    episode = np.load(base/"heldout_episode.npy", mmap_mode="r", allow_pickle=False)
    require(pose.dtype == np.float32 and pose.shape == (len(episode), 7, 4)
            and history.dtype == np.int32 and history.shape == (len(episode), 4)
            and episode.dtype == np.int32 and len(episode) > 0, "base heldout array schema")
    require((episode >= 0).all() and (episode < len(heldout)).all(), "base episode range")
    split = audit.get("splits", {}).get("heldout", {})
    pins.read(index/"heldout_valid.npy", split.get("index_sha256"))
    rows = np.load(index/"heldout_valid.npy", allow_pickle=False)
    require(rows.dtype == np.int64 and rows.ndim == 1 and len(rows) == split.get("retained")
            and split.get("original") == len(episode) and split.get("episode_errors") == []
            and (rows >= 0).all() and (rows < len(episode)).all()
            and (np.diff(rows) > 0).all(), "base heldout index contract")
    held_audit = pins.doc(index/"heldout_episodes_audit.json")
    require(held_audit.get("errors") == [] and isinstance(held_audit.get("stats"), list),
            "base heldout episode audit")
    return heldout, audit


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, doc):
    with Path(path).open("x", encoding="utf-8") as f:
        f.write(canonical(doc) + "\n")


def convert(release, release_sha, base_cache, base_complete_sha,
            base_index, base_index_audit_sha, output):
    """Build a candidate cache in a fresh NAS directory. No training admission."""
    out = Path(output)
    require(out.is_absolute() and out.is_relative_to(NAS) and out.name not in ("", ".", "..")
            and not out.exists() and not out.is_symlink(), "fresh NAS output required")
    path_dir(out.parent)
    pins = Pins()
    m = prepare_release(release, release_sha, pins)
    heldout, original_audit = prepare_base(base_cache, base_complete_sha,
                                          base_index, base_index_audit_sha, pins)
    base, base_index = path_dir(base_cache), path_dir(base_index)
    sources = [path_dir(e["branch"]) for e in m["teacher_demonstrations"]] + [base, base_index]
    require(not any(out.is_relative_to(p) or p.is_relative_to(out) for p in sources),
            "output overlaps source")
    code_paths = [Path(__file__).resolve(), Path(__import__("wa.wm.failure_state_numeric",
                   fromlist=["x"]).__file__).resolve(),
                  Path(__import__("wa.wm.failure_state_labels", fromlist=["x"]).__file__).resolve(),
                  Path(__import__("wa.wm.robot_data", fromlist=["x"]).__file__).resolve()]
    code = {str(p): pins.read(p) for p in code_paths}
    out.mkdir(exist_ok=False); (out/"index").mkdir()
    try:
        poses, histories, episodeids, valid, episodes, stats = [], [], [], [], [], []
        images, overlap, offset = {}, set(), 0
        heldscenes = {Path(e["scene"]).name.split(".")[0] for e in heldout}
        for ep, entry in enumerate(m["teacher_demonstrations"]):
            meta, obs, frames, image_hashes, numeric, s = branch_data(entry, m, pins)
            d = numeric["derived"]
            count = len(entry["window_indices"])
            poses.append(d["pose"]); histories.append(d["history"])
            episodeids.append(np.full(count, ep, dtype=np.int32))
            rows = offset + np.flatnonzero(numeric["effective_mask"])
            valid.extend(rows.tolist())
            s.update(episode=ep, rows=count, retained=len(rows), candidate_row_start=offset,
                     candidate_current_indices=entry["window_indices"],
                     valid_current_indices=numeric["valid_window_indices"],
                     valid_cache_rows=rows.tolist(), numeric_exclusions=numeric["excluded"],
                     source_sha256={n:entry["hashes"][n]
                                    for n in ("metadata.json", "observations.json", "actions.json")})
            stats.append(s)
            offset += count
            scene = Path(meta["scene_id"]).name.split(".")[0]
            if scene in heldscenes:
                overlap.add(scene)
            images[entry["branch"]] = image_hashes
            episodes.append(dict(root=entry["branch"], frames=frames, task=entry["task"],
                episode_uid=entry["task"]+":"+entry["key"], scene=meta["scene_id"],
                takeover_step=entry["takeover_step"], category="failure_state_evaluation_adaptation",
                teacher=entry["teacher"], template_index=0,
                source_experiment=entry["source_experiment"],
                source_protocol_sha256=entry["source_protocol_sha256"],
                source_job_id=entry["source_job_id"], source_task_id=entry["source_task_id"],
                source_repeat_branch=entry["repeat_branch"], repeat_frames_referenced=False,
                expected_valid_count=len(rows)))
            if "source_boundary_policy" in entry:
                episodes[-1]["source_boundary_policy"] = entry["source_boundary_policy"]
        require(offset == EXPECTED["candidate_windows"] and len(valid) == EXPECTED["valid_windows"]
                and sum(s["retained"] > 0 for s in stats) == EXPECTED["episodes_with_valid_windows"],
                "final numeric totals differ")
        for key, arrays in (("pose", poses), ("history", histories), ("episode", episodeids)):
            np.save(out/f"train_{key}.npy", np.concatenate(arrays), allow_pickle=False)
        write_json(out/"train_episodes.json", episodes)
        write_json(out/"source_image_hashes.json", images)
        write_json(out/"source_files.json", pins.files)
        for suffix in SUFFIXES:
            (out/("heldout_"+suffix)).symlink_to(base/("heldout_"+suffix))
        for name in ("heldout_valid.npy", "heldout_episodes_audit.json"):
            (out/"index"/name).symlink_to(base_index/name)
        np.save(out/"index/train_valid.npy", np.asarray(valid, dtype=np.int64), allow_pickle=False)
        write_json(out/"index/train_episodes_audit.json", dict(stats=stats, errors=[]))
        complete = dict(schema=SCHEMA, experiment="evaluation_adaptation_failure_state_cache_v1",
            contract=CONTRACT, source_release=str(path_file(release)), source_sha256=release_sha,
            source_image_inventory_sha256=file_sha(out/"source_image_hashes.json"),
            source_file_inventory_sha256=file_sha(out/"source_files.json"),
            files={f"{part}_{s}":file_sha(out/f"{part}_{s}")
                   for part in ("train", "heldout") for s in SUFFIXES},
            summary=dict(EXPECTED), code_sha256=code, template_index=0,
            original_frame_sequences_preserved=True, future_labels_teacher_owned=True,
            student_prefix_actions_used_as_future_labels=0, repeat_windows_counted=0,
            known_heldout_scene_overlap=sorted(overlap),
            inputs="causal RGB/episode-zero BBox/current simulated polar UWB; no text",
            heldout_interpretation="Original files unchanged; evaluation-set adaptation, not untouched-test generalization",
            training_released=False, independent_cache_admission_required=True)
        pins.finish()
        write_json(out/"complete.json", complete)
        train_index_sha = file_sha(out/"index/train_valid.npy")
        audit = dict(schema=SCHEMA+"_index", contract=CONTRACT,
            experiment="evaluation_adaptation_failure_state_cache_v1",
            cache_complete_sha256=file_sha(out/"complete.json"),
            train_episode_audit_sha256=file_sha(out/"index/train_episodes_audit.json"),
            heldout_episode_audit_sha256=file_sha(out/"index/heldout_episodes_audit.json"),
            training_released=False,
            splits=dict(train=dict(original=offset, retained=len(valid), excluded=offset-len(valid),
                                   index_sha256=train_index_sha, episode_errors=[]),
                        heldout=original_audit["splits"]["heldout"]))
        write_json(out/"index/audit.json", audit)
        names = [f"{p}_{s}" for p in ("train", "heldout") for s in SUFFIXES]
        names += ["source_image_hashes.json", "source_files.json", "complete.json",
                  "index/train_valid.npy", "index/train_episodes_audit.json",
                  "index/heldout_valid.npy", "index/heldout_episodes_audit.json", "index/audit.json"]
        pins.finish()
        for name, digest in code.items():
            require(file_sha(name) == digest, "converter code changed during conversion")
        admission = dict(schema=SCHEMA+"_admission_candidate", status="CACHE_CONVERTED_NOT_TRAINING_RELEASED",
            collection_release=dict(path=str(path_file(release)), sha256=release_sha),
            files={n:file_sha(out/n) for n in names},
            source_files_sha256=file_sha(out/"source_files.json"),
            original_heldout=dict(cache=str(base), cache_complete_sha256=base_complete_sha,
                                  index=str(base_index), index_audit_sha256=base_index_audit_sha),
            expected_admission_sha_required_by_loader=True, independent_audit_required=True,
            training_released=False, summary=dict(EXPECTED), code_sha256=code)
        write_json(out/"admission.json", admission)
        return dict(cache=str(out), admission_sha256=file_sha(out/"admission.json"),
                    summary=dict(EXPECTED), training_released=False)
    except Exception as exc:
        write_json(out/"failed_conversion.json", dict(status="FAILED_UNRELEASED",
            exception_type=type(exc).__name__, error=str(exc), training_released=False,
            source_release=str(release), source_sha256=release_sha))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "release-sha", "base-cache", "base-complete-sha",
                 "base-index", "base-index-audit-sha", "output"):
        p.add_argument("--"+name, required=True)
    a = p.parse_args()
    print(json.dumps(convert(a.release, a.release_sha, a.base_cache, a.base_complete_sha,
                             a.base_index, a.base_index_audit_sha, a.output), indent=2))


if __name__ == "__main__":
    main()
