"""Read-only raw failure-state branch audit; never a training release.

Supports the original v1 record and the explicitly pinned v2 teacher-error union.
The caller owns whole126 coverage, paired winner/repeat selection, source commit
and scheduler provenance. Returned pins include every branch file and RGB PNG.
"""
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image

from wa.wm import failure_state_protocol as v1
from wa.wm.failure_state_protocol_v2 import EXPERIMENT as V2, PROTOCOL_SHA as V2_SHA
from wa.wm.failure_state_teacher_error import validate_teacher_error, _state, _state_delta

REQUIRED = {"experiment", "protocol_sha256", "task", "key", "teacher",
            "takeover_step", "verification_only"}
NORMAL_FILES = ("metadata", "observations", "actions", "replay", "first_start",
                "pair_start", "takeover", "fallback_events", "admission", "complete",
                "result", "windows", "raw_windows")
EVIDENCE = ("step", "observation_index", "timestamp_s", "simulator_world_time_s",
            "rgb_sha256", "dynamic_state", "dynamic_state_sha256")
OFFSETS = np.arange(1, 8, dtype=float) / 10


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _same(a, b, message):
    _need(v1.canonical_sha(a) == v1.canonical_sha(b), message)


def _int(value, message, minimum=0):
    return v1._integer(value, message, minimum)


def _float(value, message):
    _need(type(value) in (int, float) and math.isfinite(value), message)
    return float(value)


def _array(value, shape, message):
    def numeric(x):
        if isinstance(x, list):
            for y in x:
                numeric(y)
        else:
            _float(x, message)
    numeric(value)
    a = np.asarray(value, dtype=float)
    _need(a.shape == shape and np.isfinite(a).all(), message)
    return a


def _close(a, b, message, atol=1e-6):
    a, b = np.asarray(a), np.asarray(b)
    _need(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
          and np.allclose(a, b, atol=atol, rtol=0), message)


def _json(blob):
    def pairs(items):
        result = {}
        for k, v in items:
            _need(k not in result, "duplicate JSON key")
            result[k] = v
        return result
    def bad(x):
        raise ValueError("nonfinite JSON constant: " + x)
    result = json.loads(blob, object_pairs_hook=pairs, parse_constant=bad)
    # Also rejects exponent overflow (e.g. 1e999), not just literal NaN/Infinity.
    v1.canonical_sha(result)
    return result


class Files:
    def __init__(self, root):
        self.root = Path(root)
        _need(self.root.is_absolute() and ".." not in self.root.parts
              and self.root.resolve() == self.root and self.root.is_dir(),
              "absolute nonsymlink branch root required")
        self.blobs, self.stats, self.source_files = {}, {}, {}
        for path in sorted(self.root.iterdir()):
            _need(path.is_file() and not path.is_symlink(), "nonfile/symlink branch member")
            s = path.stat()
            blob = path.read_bytes()
            _need(len(blob) == s.st_size, "branch changed while reading")
            self.blobs[path.name] = blob
            self.stats[path.name] = (s.st_size, s.st_mtime_ns, s.st_ino)
            self.source_files[str(path)] = hashlib.sha256(blob).hexdigest()

    def read(self, name):
        _need(name in self.blobs, "missing branch file: " + name)
        return _json(self.blobs[name])

    def exact(self, names):
        _same(sorted(self.blobs), sorted(names), "unexpected/missing branch files")

    def finish(self, pins):
        for name, prior in self.stats.items():
            path = self.root / name
            _need(not path.is_symlink() and path.is_file(), "branch replaced during audit")
            s = path.stat()
            _same(list(prior), [s.st_size, s.st_mtime_ns, s.st_ino], "branch changed during audit")
        _same(sorted(p.name for p in self.root.iterdir()), sorted(self.blobs), "branch inventory changed")
        _need(isinstance(pins, dict), "source_files pins must be a mapping")
        for path, digest in pins.items():
            _need(path in self.source_files, "pin outside branch or missing file")
            _same(self.source_files[path], v1._sha(digest, "source file pin"), "source SHA mismatch")


def _expected(expected, prefix):
    _need(isinstance(expected, dict) and REQUIRED <= set(expected)
          and set(expected) <= REQUIRED | {"metadata", "source_files", "prefix_image_sha256"},
          "wrong expected identity fields")
    e = {k: expected[k] for k in REQUIRED}
    protocols = {v1.EXPERIMENT: v1.PROTOCOL_SHA, V2: V2_SHA}
    _need(e["experiment"] in protocols
          and e["protocol_sha256"] == protocols[e["experiment"]], "unrecognized experiment/protocol")
    v1._key(e["task"], e["key"])
    _need(type(e["verification_only"]) is bool, "explicit verification role required")
    k = e["takeover_step"]
    if e["teacher"] == "student":
        _need(k is None and prefix is None and not e["verification_only"], "invalid student role")
        _need("prefix_image_sha256" not in expected, "student cannot claim teacher image pins")
    else:
        _need(e["teacher"] in ("lightnav", "oracle"), "invalid teacher")
        _int(k, "takeover step")
        _need(isinstance(prefix, list) and len(prefix) > k, "teacher requires audited student prefix k<N")
        pins = expected.get("prefix_image_sha256")
        _need(isinstance(pins, list) and len(pins) == len(prefix),
              "teacher requires full audited student PNG SHA list")
        for digest in pins:
            v1._sha(digest, "audited student PNG SHA")
    _need(isinstance(expected.get("metadata", {}), dict), "metadata pins must be a mapping")
    return e


def _metadata(meta, e, extra, prefix):
    for k, v in dict(e, partition="evaluation_adaptation", schema=e["experiment"], seed=7,
                     training_eligible=False, camera_alignment_verified=True).items():
        _same(meta.get(k), v, "metadata identity: " + k)
    _same(meta.get("checkpoint_sha256"), v1.CHECKPOINT_SHA, "wrong student checkpoint")
    _same(meta.get("checkpoint_step"), v1.CHECKPOINT["step"], "wrong student checkpoint step")
    for field in ("plan_sha256", "source_dataset_sha256"):
        v1._sha(meta.get(field), field)
    _same(meta.get("episode_id"), e["key"].split("/")[1], "episode identity differs")
    _need(isinstance(meta.get("scene_id"), str)
          and Path(meta["scene_id"]).name.split(".")[0] == e["key"].split("/")[0], "scene identity differs")
    _same(meta.get("target_positions_use"), "offline_label_only_never_policy_input", "target input contract")
    _same(meta.get("action_recording"), "selected command before env.step; poststate not fabricated",
          "action timing contract")
    _same(meta.get("policy_inputs"),
          "WA RGB+firstGTBBox+idealpolarUWB no text; LN RGB+text; Oracle privileged teacher only",
          "policy input contract")
    if prefix is not None:
        _same(meta.get("prefix_sha256"), v1.canonical_sha(prefix), "metadata prefix differs")
    else:
        _need(meta.get("prefix_sha256") is None, "student must not claim teacher prefix")
    if e["experiment"] == V2:
        _same(meta.get("base_protocol_sha256"), v1.PROTOCOL_SHA, "base protocol differs")
        _same(meta.get("base_plan_sha256"), meta["plan_sha256"], "base plan differs")
        v1._sha(meta.get("continuation_sha256"), "continuation SHA")
    for field, value in extra.items():
        _same(meta.get(field), value, "pinned metadata differs: " + field)


def _frames(files, meta, obs, acts, replay, e, prefix, image_pins=None, *, partial=False):
    n, na = len(obs), len(acts)
    _need(isinstance(obs, list) and isinstance(acts, list) and isinstance(replay, list),
          "record arrays required")
    _need(n > 0, "zero-frame schema unsupported; independent review required")
    _need(len(replay) == n and (na == n - 1 if partial else na == n), "frame/action coverage differs")
    times, positions, rotations, images = [], [], [], []
    max_delta = 0.
    k = e["takeover_step"]
    world0 = _float(obs[0]["simulator_world_time_s"], "first world time")
    robot_body_offset = None
    for i, (o, r) in enumerate(zip(obs, replay)):
        for field, value in ((o.get("sim_step"), i), (r.get("step"), i), (r.get("observation_index"), i)):
            _need(type(field) is int and field == value, "step/observation mismatch")
        _same(o.get("frame"), f"rgb_{i:04d}.png", "foreign frame path")
        t = _float(o.get("timestamp_s"), "invalid actual timestamp")
        world = _float(o.get("simulator_world_time_s"), "invalid world time")
        _close(t, world - world0, "elapsed/world time disagree")
        _close(o.get("simulator_elapsed_s"), t, "simulator elapsed differs")
        _close(r.get("timestamp_s"), t, "replay elapsed differs")
        _close(r.get("simulator_world_time_s"), world, "replay world time differs")
        _float(o.get("config_nominal_timestamp_s"), "invalid nominal diagnostic time")
        _need(not times or t > times[-1], "nonmonotonic actual timestamps")
        state = _state(r.get("dynamic_state"))
        _close(state["timestamp"], world, "state timestamp differs")
        _same(r.get("dynamic_state_sha256"), v1.canonical_sha(state), "state SHA differs")
        p = _array(o.get("robot_position_world"), (3,), "invalid robot world position")
        rot = _array(o.get("robot_rotation_world_from_body"), (3, 3), "invalid robot rotation")
        target = _array(o.get("target_position_world_label_only"), (3,), "invalid current target position")
        _close(rot.T @ rot, np.eye(3), "nonorthogonal rotation", atol=1e-5)
        _close(np.linalg.det(rot), 1., "improper rotation", atol=1e-5)
        _need(len(state["agents"]) >= 2, "missing robot/target dynamic-state agents")
        robot_tf = np.asarray(state["agents"][1]["transform"])
        _close(rot, robot_tf[:3, :3], "pose/replay rotation differs")
        if robot_body_offset is None:
            robot_body_offset = rot.T @ (p - robot_tf[:3, 3])
        _close(p, robot_tf[:3, 3] + rot @ robot_body_offset, "pose/replay position differs", atol=1e-5)
        # Humanoid.base_pos uses dynamic inverse_offset_transform and
        # offset_transform_base; neither is persisted by dynamic_state().
        # Finite target base labels cannot be reconstructed from sim_obj alone.
        # They remain offline-only and are never used for seven-step actions.
        blob = files.blobs.get(o["frame"])
        _need(blob is not None, "missing RGB PNG")
        with Image.open(io.BytesIO(blob)) as image:
            _need(image.format == "PNG" and image.mode == "RGB", "expected lossless RGB PNG")
            image.load()
            rgb = np.array(image)
        _same(list(rgb.shape), meta.get("rgb_shape"), "RGB dimensions differ")
        # Sensor raw hashes include shape/dtype and a nonconstant alpha channel.
        # TeacherRecorder persists only RGB PNGs, so raw RGBA cannot be rebuilt.
        # Pin/decode every PNG independently; compare teacher prefix PNGs to the
        # already audited student in addition to the recorded sensor hash/state.
        v1._sha(r.get("rgb_sha256"), "recorded sensor raw RGB SHA")
        if prefix is not None and i <= k:
            _same(r["rgb_sha256"], prefix[i].get("rgb_sha256"), "prefix raw RGB differs")
            _same(files.source_files[str(files.root/o["frame"])], image_pins[i],
                  "prefix RGB PNG SHA differs from audited student")
            delta = _state_delta(prefix[i].get("dynamic_state"), state)
            _need(delta <= 1e-6, "prefix dynamic state differs")
            max_delta = max(max_delta, delta)
            for name in ("timestamp_s", "simulator_world_time_s"):
                _close(r[name], prefix[i].get(name), "prefix time differs")
        if i < na:
            a = acts[i]
            _need(type(a.get("sim_step")) is int and a["sim_step"] == i, "action step differs")
            action = _array(a.get("normalized_action"), (3,), "invalid action")
            _need((abs(action) <= 1).all(), "action outside normalized range")
            _same(r.get("action"), a["normalized_action"], "replay/action differs")
            _same(r.get("normalized_action"), r["action"], "normalized action differs")
            owner = "student" if k is None or i < k else "teacher"
            _same(a.get("owner"), owner, "action owner differs")
            _same(r.get("owner"), owner, "replay owner differs")
            _same(a.get("teacher"), e["teacher"] if owner == "teacher" else None, "action teacher differs")
            _same(r.get("action_timing"), "before_env_step", "execution timing differs")
            _need(r.get("post_state_recorded") is False and "post_state" not in r,
                  "fabricated poststate")
            if prefix is not None and i < k:
                _same(r["action"], prefix[i].get("action"), "prefix executed action differs")
        else:
            _need(not any(x in r for x in ("action", "normalized_action", "owner", "post_state")),
                  "error action was executed/recorded")
        times.append(t); positions.append(p); rotations.append(rot); images.append(rgb)
    return np.asarray(times), np.asarray(positions), np.asarray(rotations), images, dict(
        observed_frames=n, recorded_actions=na, maximum_prefix_state_difference=max_delta,
        verified_prefix_frames=min(n, k + 1) if k is not None else 0,
        raw_alpha_reconstruction=False, raw_sensor_digest_independently_recomputed=False,
        decoded_png_count=n, png_prefix_frames_verified=min(n, k+1) if k is not None else 0,
        original_start_rgb_proof="recorded raw sensor digest paired to original61609; original PNG not supplied",
        robot_pose_dynamic_state_aligned=True,
        target_base_position_reconstruction=False,
        target_position_proof="finite observer human.base_pos only; animation offsets not persisted",
        hidden_rng_contact_state_proven=False,
        proof_scope="recorded raw RGB digests paired independently of decoded/pinned RGB PNGs; "
                    "teacher prefix PNG bytes equal audited student; actual time/transforms/joints/actions only")


def _start(first, takeover, replay, meta, e, prefix, original):
    _need(isinstance(original, dict) and {"rgb", "state"} <= set(original), "original61609 start required")
    for field in EVIDENCE:
        _same(first.get(field), replay[0].get(field), "first-start frame differs: " + field)
    _same(first.get("initial_rgb_sha256"), first["rgb_sha256"], "first raw RGB differs")
    _same(first.get("rgb_sha256"), original["rgb"], "original61609 initial RGB differs")
    _need(_state_delta(_state(original["state"]), first["dynamic_state"]) <= 1e-6,
          "original61609 initial state differs")
    box = meta.get("initial_bbox_sensor_xyxy_original")
    _same(first.get("initial_bbox_sensor_xyxy_original"), box, "first original box differs")
    _same(meta.get("initial_bbox_sensor_xyxy"), box, "original sensor box was replaced")
    for field in ("initial_bbox_rgb_xyxy", "initial_bbox_sensor_xyxy_original", "initial_bbox_repair"):
        if field in original:
            _same(meta.get(field), original[field], "original initial annotation differs")
    k = e["takeover_step"]
    if k is None or len(replay) <= k:
        _need(takeover is None, "unobserved takeover claimed")
    else:
        for field in EVIDENCE:
            _same(takeover.get(field), replay[k].get(field), "takeover observation differs")
        _same(takeover.get("expected_rgb_sha256"), prefix[k]["rgb_sha256"], "expected takeover RGB differs")
        _same(takeover.get("expected_dynamic_state_sha256"), v1.canonical_sha(prefix[k]["dynamic_state"]),
              "expected takeover SHA differs")
        _same(takeover.get("initial_rgb_sha256"), first["rgb_sha256"], "takeover changed template identity")
        _same(takeover.get("initial_bbox_sensor_xyxy_original"), box, "takeover replaced original box")


def _template(files, meta, images, *, required):
    status = meta.get("initial_bbox_status")
    valid = status in ("VERIFIED_CONFIG_AND_SEMANTIC", "VERIFIED_FROZEN_FIRST_RGB_REPAIR")
    if not valid:
        _need(not required and status in ("INVALID_NOT_VISIBLE", "UNVERIFIED_CAMERA_ALIGNMENT"),
              "missing verified template")
        return dict(valid=False, status=status)
    rgb = images[0]
    box = _array(meta.get("initial_bbox_rgb_xyxy"), (4,), "invalid initial bbox")
    x1, y1, x2, y2 = box
    _need(0 <= x1 < x2 <= rgb.shape[1] and 0 <= y1 < y2 <= rgb.shape[0], "initial bbox outside image")
    if status == "VERIFIED_CONFIG_AND_SEMANTIC":
        _need("initial_panoptic.npy" in files.blobs, "missing initial semantic evidence")
        mask = np.load(io.BytesIO(files.blobs["initial_panoptic.npy"]), allow_pickle=False).squeeze()
        _need(mask.shape == rgb.shape[:2], "semantic/RGB alignment differs")
        mask = mask == _int(meta.get("initial_target_semantic_id"), "initial semantic id")
        _need(mask.any(), "initial target invisible")
        yy, xx = np.where(mask)
        _close(box, [xx.min(), yy.min(), xx.max()+1, yy.max()+1], "semantic template bbox differs", atol=0)
    else:
        repair = meta.get("initial_bbox_repair", {})
        _same(repair.get("task"), meta["task"], "repair task differs")
        _same(repair.get("key"), meta["key"], "repair key differs")
        _same(repair.get("bbox"), meta["initial_bbox_rgb_xyxy"], "repair bbox differs")
        _same(repair.get("rgb_raw_sha256"), hashlib.sha256(rgb.tobytes()).hexdigest(), "repair RGB differs")
        _close(meta.get("initial_bbox_sensor_xyxy_original"), [0., 0., 0., 0.], "repair overwrote valid box", atol=0)
    return dict(valid=True, status=status, frame="rgb_0000.png", bbox=box.tolist(),
                image_sha256=files.source_files[str(files.root/"rgb_0000.png")])


def _windows(raw, windows, times, pos, rot, actions, replay, e, *, eligible):
    expected_indices = [i for i, t in enumerate(times) if t + .7 <= times[-1] + 1e-8]
    _need(isinstance(raw, list) and isinstance(windows, list), "window lists required")
    _same([w.get("current_index") for w in raw], expected_indices, "incomplete/duplicate raw windows")
    candidates, rejections = [], {}
    for w, i in zip(raw, expected_indices):
        _need(type(w.get("current_index")) is int, "window index must be int")
        _same(sorted(w), sorted({"current_index", "future_bracket_indices", "future_times_s",
                     "trajectory_xy_m", "interpolation"}), "unknown raw window fields")
        _close(_array(w["future_times_s"], (7,), "bad offsets"), OFFSETS, "wrong seven offsets", atol=1e-12)
        q = times[i] + OFFSETS
        right = np.clip(np.searchsorted(times, q, side="left"), 1, len(times)-1)
        _same(w["future_bracket_indices"], np.stack([right-1, right], axis=-1).tolist(), "wrong interpolation bracket")
        future = np.stack([np.interp(q, times, pos[:, j]) for j in range(3)], axis=-1)
        local = (future - pos[i]) @ rot[i]
        xy = np.stack([local[:, 0], -local[:, 2]], axis=-1)
        _close(_array(w["trajectory_xy_m"], (7, 2), "invalid trajectory"), xy, "executed-pose labels differ")
        _same(w["interpolation"], "linear_world_position_at_actual_simulator_time", "wrong interpolation")
        if e["takeover_step"] is None or i < e["takeover_step"]:
            rejections["before_takeover"] = rejections.get("before_takeover", 0) + 1
            continue
        end = int(right.max())
        _need(end > i and all(actions[j]["owner"] == replay[j]["owner"] == "teacher"
                              for j in range(i, end)), "future label interval not teacher-owned")
        candidate = copy.deepcopy(w)
        candidate.update(teacher_owned_action_indices=[i, end-1],
                         label_endpoint_observation_index=end, training_eligible=False)
        candidates.append(candidate)
    wanted = candidates if eligible else []
    _same(windows, wanted, "candidate windows differ from complete teacher-owned suffix")
    return [w["current_index"] for w in wanted], rejections


def _proof(proof, n, k, *, partial=False):
    _need(isinstance(proof, dict), "missing agent proof")
    if k is None:
        for field in ("replay_wrapper", "prefix_hash_matches", "environment_bound", "takeover_matches", "replay_verified"):
            _need(proof.get(field) is False, "student has teacher proof")
        _same(proof.get("verified_prefix_frames"), 0, "student prefix proof")
        _same(proof.get("required_prefix_frames"), 0, "student required proof")
    else:
        for field in ("replay_wrapper", "prefix_hash_matches", "environment_bound", "takeover_matches"):
            _need(proof.get(field) is True, "missing recorded replay binding: " + field)
        _same(proof.get("agent_step_before_reset"), n, "proof taken after reset or wrong action count")
        _same(proof.get("verified_prefix_frames"), min(n, k+1), "verified executed prefix coverage differs")
        _same(proof.get("required_prefix_frames"), k+1, "required prefix coverage differs")
        _same(proof.get("replay_verified"), n >= k+1, "recorded full-prefix flag differs")
        if not partial:
            _need(n >= k+1, "normal teacher lacks takeover action")
    _need(proof.get("hidden_rng_contact_state_proven") is False, "unsupported hidden-state proof")
    _same(proof.get("verification_source"), "successful ReplayThenTeacher.act calls before final reset", "unknown proof source")
    _same(proof.get("proof_scope"), "t=0..k raw RGB plus world time and articulated transforms/joints, atol=1e-6", "unknown proof scope")


def _normal(files, e, expected, prefix, original):
    d = {name: files.read(name+".json") for name in NORMAL_FILES}
    meta, adm, done, result = (d[k] for k in ("metadata", "admission", "complete", "result"))
    _metadata(meta, e, expected.get("metadata", {}), prefix)
    flags, rate = v1._result(result)
    times, pos, rot, images, actual = _frames(files, meta, d["observations"], d["actions"],
                                           d["replay"], e, prefix, expected.get("prefix_image_sha256"))
    n = len(times); student = e["teacher"] == "student"; k = e["takeover_step"]
    names = {name+".json" for name in NORMAL_FILES} | {f"rgb_{i:04d}.png" for i in range(n)}
    if "initial_panoptic.npy" in files.blobs:
        names.add("initial_panoptic.npy")
    if not student:
        names.add("branch.json")
    files.exact(names)
    _same(d["pair_start"], d["first_start"], "paired first-start differs")
    _start(d["first_start"], d["takeover"], d["replay"], meta, e, prefix, original)
    actual["template"] = _template(files, meta, images, required=flags["policy_init_valid"])
    _same(meta.get("observed_frame_dt_s"), np.diff(times).tolist(), "actual interval record differs")
    _same(meta.get("timebase_version"), "v3_actual_world_time_interpolated", "wrong timebase")
    _same(meta.get("label_interpolation"), "linear_world_position_no_extrapolation", "wrong label interpolation")
    _same(d["fallback_events"], [], "normal branch has fallback/reply errors")
    _same(adm.get("issues"), [], "normal branch has recorder issues")
    for field, value in dict(e, partition="evaluation_adaptation", schema=e["experiment"], seed=7,
                             complete=True, replay_verified=not student, transport_fallback=False,
                             training_eligible=False, training_released=False).items():
        _same(adm.get(field), value, "admission differs: " + field)
    _same(adm.get("result"), result, "admission result differs")
    _int(result.get("total_step"), "terminal step count", 1)
    _same(result["total_step"], n, "result count differs from executed actions")
    _proof(adm.get("agent_validation"), n, k)
    prefix_sha = v1.canonical_sha(prefix) if prefix is not None else None
    takeover_sha = v1.canonical_sha(prefix[k]["dynamic_state"]) if prefix is not None else None
    actual_takeover = v1.canonical_sha(d["replay"][k]["dynamic_state"]) if prefix is not None else None
    for field, value in dict(prefix_sha256=prefix_sha, initial_rgb_sha256=d["replay"][0]["rgb_sha256"],
                             takeover_state_sha256=takeover_sha, actual_takeover_state_sha256=actual_takeover,
                             takeover_observation_index=k).items():
        _same(adm.get(field), value, "admission pair identity differs: " + field)
    owned = dict(action_count=0 if student else n-k, start_observation_index=k,
                 end_observation_index_inclusive=None if student else n-1, start_step=k,
                 end_step_inclusive=None if student else n-1, contiguous=not student,
                 poststate_claim="none; only next preaction frames are observed")
    _same(adm.get("owned_suffix"), owned, "owned suffix proof differs")
    eligible = not student and not e["verification_only"] and flags["success"] and flags["policy_init_valid"] and not flags["collision"]
    indices, rejected = _windows(d["raw_windows"], d["windows"], times, pos, rot,
                                 d["actions"], d["replay"], e, eligible=eligible)
    _same(adm.get("window_rejections"), rejected, "window rejection accounting differs")
    for field, value in dict(candidate_windows=len(indices), raw_windows=len(d["raw_windows"]),
                             no_recovery_needed=student and flags["success"]).items():
        _same(adm.get(field), value, "admission count/status differs: " + field)
    status = ("rerun_student_success" if flags["success"] else "student_failure_recorded") if student else (
        "verification_only" if e["verification_only"] else "teacher_suffix_candidate" if eligible else "teacher_branch_rejected")
    _same(adm.get("status"), status, "wrong branch admission status")
    for field, value in dict(schema=e["experiment"], status="FAILURE_STATE_RAW_COLLECTION_COMPLETE",
                             frames=n, complete=True, windows=len(indices), candidate_windows=len(indices),
                             raw_windows=len(d["raw_windows"]), training_eligible=False, training_released=False,
                             label_source="executed_robot_pose", future_times_s=OFFSETS.tolist()).items():
        _same(done.get(field), value, "completion differs: " + field)
    if student:
        branch = dict(result=result, prefix=d["replay"], first_start=d["first_start"],
                      artifact_root=str(files.root), initial_pair_verified=True)
    else:
        branch = dict(experiment=e["experiment"], teacher=e["teacher"], complete=True,
                      replay_verified=True, transport_fallback=False, task=e["task"], key=e["key"],
                      takeover_step=k, seed=7, protocol_sha256=e["protocol_sha256"], prefix_sha256=prefix_sha,
                      initial_rgb_sha256=d["replay"][0]["rgb_sha256"], takeover_state_sha256=takeover_sha,
                      actual_takeover_state_sha256=actual_takeover, candidate_windows=len(indices),
                      result=result, artifact_root=str(files.root), verification_only=e["verification_only"])
        _same(files.read("branch.json"), branch, "persisted branch summary differs")
    return dict(kind="normal", branch=branch, result=result, candidate_indices=indices,
                candidate_windows=len(indices), actualproof=actual)


def _error_branch(files, e, expected, prefix, original):
    _need(e["experiment"] == V2 and e["teacher"] != "student", "error union requires v2 teacher")
    doc = files.read("teacher_error.json")
    _same(files.read("branch.json"), doc, "branch/error differs")
    _same(doc.get("teacher"), e["teacher"], "error teacher differs")
    _same(doc.get("verification_only"), e["verification_only"], "error verification role differs")
    pair = doc.get("expected_pair", {})
    for field in ("experiment", "task", "key", "takeover_step", "protocol_sha256"):
        _same(pair.get(field), e[field], "error expected identity differs")
    for field, filename in (("partial_replay", "partial_replay.json"), ("partial_actions", "partial_actions.json")):
        _same(doc["observed_partial"]["evidence_files"][field]["path"], str(files.root/filename),
              "error evidence path escapes branch")
    # This re-reads exact persisted files and rederives partial/null-terminal proof.
    validate_teacher_error(doc, prefix=prefix)
    context = files.read("teacher_error_context.json")
    meta = context.get("metadata", {})
    _metadata(meta, e, expected.get("metadata", {}), prefix)
    obs, acts, replay = (files.read(name+".json") for name in
                         ("partial_observations", "partial_actions", "partial_replay"))
    _, _, _, images, actual = _frames(files, meta, obs, acts, replay, e, prefix,
                                      expected.get("prefix_image_sha256"), partial=True)
    names = {"teacher_error.json", "branch.json", "teacher_error_context.json",
             "partial_observations.json", "partial_actions.json", "partial_replay.json"}
    names |= {f"rgb_{i:04d}.png" for i in range(len(obs))}
    if "initial_panoptic.npy" in files.blobs:
        names.add("initial_panoptic.npy")
    files.exact(names)  # no result/admission/complete/windows or generic ERROR file
    _start(context.get("first_start"), context.get("takeover"), replay, meta, e, prefix, original)
    actual["template"] = _template(files, meta, images, required=False)
    for field in ("training_eligible", "training_released"):
        _need(context.get(field) is False, "error branch claims training release")
    _need(context.get("no_terminal_result") is True, "error has terminal claim")
    _proof(context.get("agent_validation"), len(acts), e["takeover_step"], partial=True)
    _need(isinstance(context.get("fallback_events"), list)
          and bool(context["fallback_events"]) == doc["fallback_detected"],
          "missing or inconsistent actual error/fallback detection")
    _need(doc["fallback_executed"] is False, "error fallback boundary differs")
    if doc["status"] == "teacher_output_invalid":
        reply = context.get("raw_reply", {})
        raw = doc["raw_error"]
        _same({k: reply.get(k) for k in ("rc", "seq", "msg")},
              dict(rc=raw["rc"], seq=raw["seq"], msg=raw["message"]), "raw reply/error mismatch")
        _same(context.get("raw_client_sequence"), raw["seq"]+1, "client sequence mismatch")
        _need(_int(context.get("episode_fallback_count"), "detected fallback count", 1) >= 1,
              "missing decoder-error count")
    actual["partial_only"] = True
    return dict(kind="teacher_error", branch=doc, result=None, candidate_indices=[],
                candidate_windows=0, actualproof=actual)


def audit_branch(root, *, expected, student_prefix=None, original_start=None):
    """Return evidence/pins only; parent auditor owns paired selection/release.

    expected requires REQUIRED keys; optional metadata pins exact metadata values,
    and source_files pins any subset of absolute files under this branch. A full
    returned source_files inventory can be supplied later to detect any tampering.
    Teachers require prefix_image_sha256: the full image_sha256 list returned by
    an audited student, without adding fields to its canonical replay prefix.
    Sensor alpha is not saved: raw digests are paired, not independently rebuilt.
    original_start is the pinned61609 {rgb,state}, optionally initial bbox/repair.
    """
    try:
        e = _expected(expected, student_prefix)
        files = Files(root)
        result = (_error_branch(files, e, expected, student_prefix, original_start)
                  if "teacher_error.json" in files.blobs else
                  _normal(files, e, expected, student_prefix, original_start))
        files.finish(expected.get("source_files", {}))
        result.update(source_files=files.source_files, branch_root=str(files.root),
                      image_sha256=[files.source_files[str(files.root/f"rgb_{i:04d}.png")]
                                    for i in range(result["actualproof"]["observed_frames"])],
                      training_released=False, cache_conversion_required=True)
        return result
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
        raise ValueError("missing/malformed branch evidence: " + str(exc)) from exc
