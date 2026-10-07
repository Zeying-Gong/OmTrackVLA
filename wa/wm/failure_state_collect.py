"""Bounded failure-state dual-teacher adaptation; never an untouched-test score.

The benchmark still owns reset, stepping, physics and success. This driver records
a fresh WA prefix, replays its executed actions to both teachers, and only
proposes a teacher-owned successful suffix after an additional successful replay.
It does not release a training cache or modify an existing evaluation result.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path

from wa.wm.failure_state_protocol import (
    EXPERIMENT, binary_flag, candidate_steps, load_plan, select_recovery_teacher,
)

RGB_KEY = "agent_1_articulated_agent_jaw_rgb"
PAIR_FIELDS = ("experiment", "task", "key", "takeover_step", "seed",
               "protocol_sha256", "initial_rgb_sha256",
               "takeover_state_sha256", "prefix_sha256")


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def good_result(result):
    for name in ("success", "collision"):
        value = result.get(name)
        if isinstance(value, str) or value not in (0, 1):
            raise ValueError("nonbinary outcome: " + name)
    return bool(result["success"]) and not bool(result["collision"])


def repeat_valid(original, repeat):
    for field in PAIR_FIELDS + ("teacher",):
        if field not in original or original[field] != repeat.get(field):
            raise ValueError("repeat identity differs: " + field)
    for name in ("complete", "replay_verified", "transport_fallback"):
        if type(repeat.get(name)) is not bool:
            raise ValueError("missing repeat proof: " + name)
    if original.get("verification_only") is not False or repeat.get("verification_only") is not True:
        raise ValueError("separate verification replay required")
    if original.get("artifact_root") == repeat.get("artifact_root"):
        raise ValueError("repeat must have independent artifact root")
    init_valid = binary_flag(repeat.get("result", {}).get("policy_init_valid"), "repeat initialization")
    tr = repeat.get("result", {}).get("following_rate")
    if type(tr) not in (int, float) or not math.isfinite(float(tr)) or not 0 <= tr <= 1:
        raise ValueError("invalid repeat following_rate")
    return (repeat["complete"] and repeat["replay_verified"]
            and init_valid and not repeat["transport_fallback"] and good_result(repeat["result"]))


def search_recovery(student, prefix_length, run, *, choose=select_recovery_teacher):
    """run(teacher, k, verification_only) returns a persisted complete branch.

    Only the original chosen branch is proposed, never the repeat or WA prefix.
    Each k is independent; success at one k does not imply later recoverability.
    """
    result = student["result"]
    report = dict(experiment=EXPERIMENT, student=student, attempts=[], accepted=None,
                  training_released=False, score_backfill_allowed=False)
    if type(result.get("policy_init_valid")) is not bool:
        raise ValueError("student init validity required")
    if not result["policy_init_valid"]:
        report["outcome"] = "student_invalid_initialization"
        return report
    if good_result(result):
        report["outcome"] = "rerun_student_success_no_recovery_needed"
        return report
    for k in candidate_steps(prefix_length):
        branches = {name: run(name, k, False) for name in ("lightnav", "oracle")}
        selection = choose(branches["lightnav"], branches["oracle"])
        attempt = dict(takeover_step=k, branches=branches, selection=selection)
        report["attempts"].append(attempt)
        selected = selection["selected_teacher"]
        if selected is None:
            continue
        if selected not in branches:
            raise ValueError("unknown selected teacher")
        repeat = run(selected, k, True)
        attempt["repeat"] = repeat
        attempt["repeat_valid"] = repeat_valid(branches[selected], repeat)
        windows = branches[selected].get("candidate_windows")
        if type(windows) is not int or windows < 0:
            raise ValueError("explicit nonnegative candidate window count required")
        attempt["selected_candidate_windows"] = windows
        if attempt["repeat_valid"] and windows == 0:
            attempt["admission_reason"] = "selected_teacher_no_candidate_windows"
        if attempt["repeat_valid"] and windows > 0:
            report["accepted"] = dict(
                teacher=selected, takeover_step=k,
                artifact_root=branches[selected]["artifact_root"],
                repeat_artifact_root=repeat["artifact_root"],
                selection=selection, prefix_sha256=branches[selected]["prefix_sha256"],
                candidate_only=True, training_released=False)
            report["outcome"] = "repeated_teacher_recovery_candidate"
            return report
    report["outcome"] = "no_valid_teacher_recovery"
    return report


class DevelopmentLimitReached(RuntimeError):
    pass


class LimitedRecorder:
    """Only a developer interruption, never an alternative success criterion."""
    def __init__(self, inner, actions):
        self.inner, self.actions, self.early_result = inner, actions, None

    def observe(self, obs, detector, robot, human, episode, step, frequency, world_time=None):
        if step >= self.actions:
            raise DevelopmentLimitReached("bounded developer action limit")
        return self.inner.observe(obs, detector, robot, human, episode, step, frequency, world_time)

    def record_action(self, *args):
        return self.inner.record_action(*args)

    def finish(self, result):
        # A natural early terminal must not create formal result/complete/windows.
        self.early_result = result


class Runtime:
    def __init__(self, plan, plan_sha, ready, output):
        import habitat
        import evt_bench  # Register benchmark sensors/actions before Hydra composition.
        from evt_full_20260926.common import BENCH, SCENES, sha, scene
        from wa.wm.initial_bbox_repair import load_plan as load_bbox
        from wa.wm.full_mixed_contract import validate_ready
        self.plan, self.plan_sha, self.ready = plan, plan_sha, ready
        self.output, self.bench = Path(output), BENCH
        self.protocol_sha = plan["protocol_sha256"]
        validate_ready(ready, checkpoint_sha=plan["checkpoint"]["sha256"],
                       step=plan["checkpoint"]["step"], mode="mixed")
        if ready.get("seed") != "7+step":
            raise ValueError("WA server seed differs")
        if any(type(ready.get(k)) is not bool or ready[k]
               for k in ("text_used", "world_predictor_inference")):
            raise ValueError("WA policy input/inference contract differs")
        if os.environ.get("WA_DIAG_CONTROLLER") != "learned_yaw_guard_v1":
            raise ValueError("wrong controller")
        if os.environ.get("WA_SEMANTIC_PLY_FIX") != "mp3d_semantic_ply_v1":
            raise ValueError("wrong semantic protocol")
        if os.environ.get("WA_STUDENT_EVAL") != "evaluation_set_adaptation_v1":
            raise ValueError("explicit WA input/start-evidence contract required")
        self.manifest = json.loads(Path(plan["manifest"]["path"]).read_text())
        spec = self.manifest["tasks"]["stt"]
        if sha(spec["path"]) != spec["sha256"]:
            raise ValueError("STT data changed")
        self.config = habitat.get_config(
            str(BENCH / "habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml"),
            ["habitat.simulator.habitat_sim_v0.gpu_device_id=0",
             "habitat.environment.iterator_options.shuffle=false",
             "habitat.dataset.data_path=" + spec["path"],
             "habitat.simulator.scene_dataset=" + SCENES])
        from omegaconf import OmegaConf
        sensors = OmegaConf.to_container(self.config.habitat.simulator.agents.agent_1.sim_sensors,
                                         resolve=True)
        for field in ("height", "width", "position", "orientation", "hfov", "sensor_subtype"):
            if sensors["jaw_rgb_sensor"][field] != sensors["jaw_panoptic_sensor"][field]:
                raise ValueError("RGB/semantic camera mismatch: " + field)
        self.dataset = habitat.make_dataset(self.config.habitat.dataset.type,
                                           config=self.config.habitat.dataset)
        self.episodes = {scene(dict(scene_id=e.scene_id)) + "/" + str(e.episode_id): e
                         for e in self.dataset.episodes}
        self.entries = {e["key"]: e for e in self.manifest["tasks"]["stt"]["episodes"]}
        if len(self.episodes) != 1405 or set(self.episodes) != set(self.entries):
            raise ValueError("wrong STT definitions")
        self.repairs = {(r["task"], r["key"]): r for r in
                        load_bbox(plan["repair_plan"]["path"], plan["repair_plan"]["sha256"])["repairs"]}
        source = plan["sources"]["combined"]
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("pinned original results changed")
        self.originals = {r["key"]: r for r in
                          map(json.loads, Path(source["path"]).read_text().splitlines())
                          if r["task"] == "stt"}
        with (self.output / "simulator_config.yaml").open("x") as stream:
            stream.write(OmegaConf.to_yaml(self.config))
        write_json(self.output / "model.json", ready)

    def check_entry(self, entry):
        if entry["task"] != "stt" or entry["key"] not in self.episodes:
            raise ValueError("foreign collection key")
        ep = self.episodes[entry["key"]]
        if ep.info["instruction"] != self.entries[entry["key"]]["instruction"]:
            raise ValueError("instruction definition mismatch")
        original = self.originals[entry["key"]]
        if canonical_sha(original) != entry["baseline_row_sha256"]:
            raise ValueError("baseline row changed")
        if original["success"] != 0:
            raise ValueError("collection must be fixed original failures")
        return ep

    def run(self, entry, base, name, teacher_url, *, teacher_name=None,
            prefix=None, takeover_step=None, verification_only=False, development_actions=None):
        import random
        import numpy as np
        import torch
        import trained_agent
        from lightnav_transport_20260927.agent import Agent as LightNav
        from wa.wm.diagnostic_agent import DiagnosticAgent
        from wa.wm.initial_bbox_repair_agent import InitialBBoxRepairAgent
        from wa.wm.dual_teacher_collect import BoundTeacher
        from wa.wm.oracle_teacher import OracleTeacher
        from wa.wm.recovery_replay import ReplayThenTeacher, check_dynamic_state
        from wa.wm.semantic_scene import prepare_episode
        from wa.wm.failure_state_recorder import FailureStateRecorder

        ep = self.check_entry(entry)
        random.seed(7); np.random.seed(7); torch.manual_seed(7); torch.cuda.manual_seed_all(7)
        branch_root = base / name
        repair = self.repairs.get(("stt", entry["key"]))
        teacher = None
        if teacher_name is None:
            args = (self.ready["url"], self.config.habitat.task.actions.agent_1_base_velocity,
                    "mixed", base / (name + ".trace.jsonl"))
            agent = (InitialBBoxRepairAgent(*args, repair=repair) if repair else DiagnosticAgent(*args))
            environment = lambda: agent.diagnostic_env
        else:
            if teacher_name not in ("lightnav", "oracle"):
                raise ValueError("unknown teacher")
            teacher = (LightNav(str(base / (name + "_client")), teacher_url)
                       if teacher_name == "lightnav" else OracleTeacher())
            agent = ReplayThenTeacher(BoundTeacher(teacher, allow_released_fallback=False),
                                      prefix, takeover_step, RGB_KEY)
            environment = lambda: agent.environment
        meta = dict(experiment=EXPERIMENT, partition="evaluation_adaptation", task="stt",
                    key=entry["key"], teacher=teacher_name or "student",
                    verification_only=verification_only, plan_sha256=self.plan_sha,
                    protocol_sha256=self.protocol_sha, seed=7,
                    checkpoint_sha256=self.plan["checkpoint"]["sha256"],
                    checkpoint_step=self.plan["checkpoint"]["step"],
                    source_dataset_sha256=self.manifest["tasks"]["stt"]["sha256"],
                    scene_id=ep.scene_id, episode_id=str(ep.episode_id),
                    camera_alignment_verified=True,
                    policy_inputs="WA RGB+firstGTBBox+idealpolarUWB no text; LN RGB+text; Oracle privileged teacher only")
        if prefix is not None:
            meta["prefix_sha256"] = canonical_sha(prefix)
        rec = FailureStateRecorder(branch_root, meta, environment, takeover_step=takeover_step,
                                   repair=repair, expected_prefix=prefix, agent=agent)
        limited = LimitedRecorder(rec, development_actions) if development_actions is not None else None
        subset = copy.copy(self.dataset)
        subset.episodes = [prepare_episode(ep, self.bench)]
        interrupted = False
        try:
            try:
                trained_agent.evaluate_agent(copy.deepcopy(self.config), subset, str(base/(name+"_metrics")),
                                             agent_factory=lambda _: agent, recorder=limited or rec)
            except DevelopmentLimitReached:
                if limited is None:
                    raise
                interrupted = True
            if limited is not None:
                snapshot = rec.finish_development("action_limit" if interrupted else "unexpected_early_terminal")
                if not interrupted or len(rec.replay) != development_actions:
                    raise RuntimeError("developer prefix terminated before required action limit")
                if teacher_name is not None:
                    proof = snapshot["agent_validation"]
                    if (snapshot["replay_verified"] is not True or snapshot["transport_fallback"] is not False
                            or proof["verified_prefix_frames"] != takeover_step + 1
                            or snapshot["owned_suffix"]["action_count"] != development_actions - takeover_step):
                        raise ValueError("developer nonzero replay proof failed")
                return dict(development=snapshot, prefix=rec.replay, artifact_root=str(branch_root))
            result = json.loads((branch_root/"result.json").read_text())
            admission = json.loads((branch_root/"admission.json").read_text())
            first = json.loads((branch_root/"first_start.json").read_text())
            replay = json.loads((branch_root/"replay.json").read_text())
            if teacher_name is None:
                expected = self.originals[entry["key"]]["initial_pair_evidence"]
                if first["rgb_sha256"] != expected["rgb"]:
                    raise ValueError("student original raw initial RGB mismatch")
                check_dynamic_state(expected["state"], first["dynamic_state"])
                return dict(result=result, prefix=replay, first_start=first,
                            artifact_root=str(branch_root), initial_pair_verified=True)
            if admission["replay_verified"] is not True or admission["complete"] is not True:
                raise ValueError("missing complete teacher prefix proof")
            branch = dict(experiment=EXPERIMENT, teacher=teacher_name, complete=True,
                          replay_verified=True, transport_fallback=admission["transport_fallback"],
                          task="stt", key=entry["key"], takeover_step=takeover_step, seed=7,
                          protocol_sha256=self.protocol_sha, prefix_sha256=canonical_sha(prefix),
                          initial_rgb_sha256=prefix[0]["rgb_sha256"],
                          takeover_state_sha256=canonical_sha(prefix[takeover_step]["dynamic_state"]),
                          actual_takeover_state_sha256=admission["actual_takeover_state_sha256"],
                          candidate_windows=admission["candidate_windows"],
                          result=result, artifact_root=str(branch_root), verification_only=verification_only)
            if (admission["initial_rgb_sha256"] != branch["initial_rgb_sha256"]
                    or admission["takeover_state_sha256"] != branch["takeover_state_sha256"]
                    or admission["prefix_sha256"] != branch["prefix_sha256"]):
                raise ValueError("recorder and branch identity mismatch")
            write_json(branch_root/"branch.json", branch)
            return branch
        except BaseException as exc:
            # Failed runtime cannot become a successful score or training example.
            if not (branch_root/"ERROR.json").exists():
                write_json(branch_root/"ERROR.json", dict(error_type=type(exc).__name__,
                           message=str(exc)[:2000], training_eligible=False))
                write_json(branch_root/"partial_replay.json", rec.replay)
                write_json(branch_root/"partial_actions.json", rec.actions)
            raise
        finally:
            if teacher is not None:
                teacher.close()


def main():
    p = argparse.ArgumentParser()
    for name in ("plan", "plan-sha", "ready", "teacher-url", "output"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--development-key")
    a = p.parse_args()
    development = a.development_key is not None
    if not 0 <= a.shard < 8:
        raise ValueError("eight planned lanes")
    if development and (os.environ.get("WA_DEVELOPMENT") != "1" or os.environ.get("MD_AK_JOB_ID")):
        raise ValueError("developer check cannot be a cluster smoke")
    if not development and os.environ.get("WA_DEVELOPMENT") == "1":
        raise ValueError("explicit development key required")
    plan = load_plan(a.plan, a.plan_sha)
    entries = ([e for e in plan["entries"] if e["key"] == a.development_key] if development
               else plan["lanes"][a.shard])
    if not entries or (development and len(entries) != 1):
        raise ValueError("empty or foreign planned key")
    out = Path(a.output)
    if not str(out.resolve()).startswith("/data/nas_ray/"):
        raise ValueError("persistent NAS output required")
    out.mkdir(parents=True, exist_ok=False)
    ready = json.loads(Path(a.ready).read_text())
    runtime = Runtime(plan, a.plan_sha, ready, out)
    write_json(out/"plan_identity.json", dict(path=a.plan, sha256=a.plan_sha,
                                            protocol_sha256=plan["protocol_sha256"]))
    completed = []
    for entry in entries:
        base = out/entry["task"]/entry["key"]
        base.mkdir(parents=True, exist_ok=False)
        student = runtime.run(entry, base, "student", a.teacher_url,
                              development_actions=24 if development else None)
        prefix = student.pop("prefix")
        if development:
            if len(prefix) != 24:
                raise ValueError("real developer prefix must contain24 actions")
            expected = runtime.originals[entry["key"]]["initial_pair_evidence"]
            from wa.wm.recovery_replay import check_dynamic_state
            if prefix[0]["rgb_sha256"] != expected["rgb"]:
                raise ValueError("developer original raw initial RGB mismatch")
            check_dynamic_state(expected["state"], prefix[0]["dynamic_state"])
            probes = {}
            for name in ("lightnav", "oracle"):
                probe = runtime.run(entry, base, name+"_takeover0020", a.teacher_url,
                                    teacher_name=name, prefix=prefix, takeover_step=20,
                                    development_actions=24)
                probe.pop("prefix")
                probes[name] = probe
            record = dict(status="DEVELOPER_NONZERO_REPLAY_CHECK_ONLY", key=entry["key"],
                          prefix_actions=24, takeover_step=20, branch_total_actions=24, teacher_owned_actions=4,
                          no_success_rate=True, training_eligible=False,
                          student=student, teachers=probes)
            write_json(base/"developer_check.json", record)
        else:
            def run(name, k, repeat):
                label = f"{name}_{k:04d}" + ("_repeat" if repeat else "")
                return runtime.run(entry, base, label, a.teacher_url, teacher_name=name,
                                   prefix=prefix, takeover_step=k, verification_only=repeat)
            record = search_recovery(student, len(prefix), run)
            record.update(task=entry["task"], key=entry["key"], plan_sha256=a.plan_sha,
                          protocol_sha256=plan["protocol_sha256"],
                          baseline_result_unchanged=entry["baseline_result"])
            write_json(base/"search.json", record)
        with (out/"records.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False)+"\n")
        completed.append(entry["key"])
        print("RECOVERY_ENTRY_COMPLETE", entry["key"], record.get("outcome", record.get("status")), flush=True)
    name = "DEVELOPMENT_CHECK.json" if development else "COMPLETE.json"
    write_json(out/name, dict(experiment=EXPERIMENT, entries=len(completed), keys=completed,
                            expected=len(entries), shard=a.shard, development=development,
                            training_released=False, no_success_rate=True,
                            plan_sha256=a.plan_sha, protocol_sha256=plan["protocol_sha256"]))
    print(name, len(completed), flush=True)


if __name__ == "__main__":
    main()
