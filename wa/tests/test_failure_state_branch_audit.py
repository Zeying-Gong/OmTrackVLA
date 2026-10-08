"""Small synthetic CPU fixtures; no Habitat, GPU, job or real-result mutation."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from wa.wm.failure_state_branch_audit import audit_branch
from wa.wm import failure_state_protocol as v1
from wa.wm.failure_state_protocol_v2 import EXPERIMENT as V2, PROTOCOL_SHA as V2_SHA
from wa.wm.failure_state_recorder import FailureStateRecorder, RGB_KEY
from wa.wm.recovery_replay import ReplayThenTeacher
from wa.wm.failure_state_teacher_error import TeacherOutputInvalid, build_teacher_error


class Transform:
    def __init__(self):
        self.matrix = np.eye(4)
    def __array__(self, dtype=None):
        return np.asarray(self.matrix, dtype=dtype)
    def rotation(self):
        return self.matrix[:3, :3]


class Teacher:
    def act(self, *args, **kwargs):
        return [.1, 0., 0.]
    def reset(self, *args, **kwargs):
        pass


class Fixture:
    def __init__(self):
        self.robot_tf, self.human_tf = Transform(), Transform()
        self.robot = SimpleNamespace(base_pos=np.zeros(3), sim_obj=SimpleNamespace(
            transformation=self.robot_tf, joint_positions=np.zeros(2)))
        self.human = SimpleNamespace(base_pos=np.array([2., 0., 0.]), sim_obj=SimpleNamespace(
            transformation=self.human_tf, joint_positions=np.zeros(2)))
        self.time = 0.
        self.env = SimpleNamespace(sim=SimpleNamespace(agents_mgr=[
            SimpleNamespace(articulated_agent=self.human),
            SimpleNamespace(articulated_agent=self.robot)], get_world_time=lambda: self.time))
        self.episode = SimpleNamespace(info=dict(main_human_semantic_id=1, instruction="unused text"))

    def step(self, i):
        self.time = i * .1
        angle = i * .01
        c, s = np.cos(angle), np.sin(angle)
        self.robot_tf.matrix[:3, :3] = [[c, 0., s], [0., 1., 0.], [-s, 0., c]]
        self.robot.base_pos = np.array([i*.02, 0., 0.])
        self.robot_tf.matrix[:3, 3] = self.robot.base_pos + [0., .48, 0.]
        # Animated sim_obj transform can move relative to human ground base.
        self.human_tf.matrix[:3, 3] = self.human.base_pos + [i*.003, .9+i*.001, 0.]
        rgb = np.full((4, 5, 4), i % 255, dtype=np.uint8)
        # Real Habitat alpha is nonconstant and is not persisted in RGB PNG.
        rgb[..., 3] = (np.arange(20).reshape(4, 5) + i) % 255
        return {RGB_KEY: rgb, "agent_1_articulated_agent_jaw_panoptic": np.ones((4, 5), dtype=np.int32)}

    def detector(self):
        return {"agent_1_main_humanoid_detector_sensor": {"box": np.array([0., 0., 5., 4.])}}


class BranchAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.serial = 0
        self.prefix_images = {}

    def write(self, root, name, value):
        (root/name).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")

    def read(self, root, name):
        return json.loads((root/name).read_text())

    def metadata(self, version, teacher, k, repeat, prefix):
        exp, protocol = (v1.EXPERIMENT, v1.PROTOCOL_SHA) if version == 1 else (V2, V2_SHA)
        meta = dict(experiment=exp, protocol_sha256=protocol, partition="evaluation_adaptation",
                    task="stt", key="scene/1", teacher=teacher, takeover_step=k, verification_only=repeat,
                    plan_sha256="a"*64, source_dataset_sha256="b"*64, checkpoint_sha256=v1.CHECKPOINT_SHA,
                    checkpoint_step=v1.CHECKPOINT["step"], seed=7, scene_id="/fixture/scene.basis.glb",
                    episode_id="1", camera_alignment_verified=True,
                    policy_inputs="WA RGB+firstGTBBox+idealpolarUWB no text; LN RGB+text; Oracle privileged teacher only")
        if version == 2:
            meta.update(base_plan_sha256=meta["plan_sha256"], base_protocol_sha256=v1.PROTOCOL_SHA,
                        continuation_sha256="c"*64)
        if prefix is not None:
            meta["prefix_sha256"] = v1.canonical_sha(prefix)
        return meta

    def make(self, *, version=1, teacher="student", k=None, repeat=False, prefix=None,
             success=False, collision=False, valid=True, count=24, error=False):
        self.serial += 1
        root = self.base / str(self.serial)
        f = Fixture()
        agent = SimpleNamespace() if teacher == "student" else ReplayThenTeacher(Teacher(), prefix, k, RGB_KEY)
        if teacher != "student":
            agent.bind_environment(f.env)
        meta = self.metadata(version, teacher, k, repeat, prefix)
        rec = FailureStateRecorder(root, meta, lambda: f.env, takeover_step=k,
                                   expected_prefix=prefix, agent=agent, experiment=meta["experiment"])
        for i in range(count):
            obs = f.step(i)
            rec.observe(obs, f.detector(), f.robot, f.human, f.episode, i, 10, f.time)
            action = [-.1, 0., 0.] if teacher == "student" else agent.act(obs, None, "1")
            rec.record_action(i, action, None)
        expected = {field: meta[field] for field in ("experiment", "protocol_sha256", "task", "key",
                                                    "teacher", "takeover_step", "verification_only")}
        if prefix is not None:
            expected["prefix_image_sha256"] = self.prefix_images[v1.canonical_sha(prefix)]
        if error:
            i = count
            obs = f.step(i)
            rec.observe(obs, f.detector(), f.robot, f.human, f.episode, i, 10, f.time)
            for name, data in (("partial_replay", rec.replay), ("partial_actions", rec.actions),
                               ("partial_observations", rec.frames)):
                self.write(root, name+".json", data)
            err = TeacherOutputInvalid(rc=500, seq=42, message="Missing rvq act levels [0] in reply")
            pair = dict(experiment=V2, task="stt", key="scene/1", takeover_step=k, seed=7,
                        protocol_sha256=V2_SHA, initial_rgb_sha256=prefix[0]["rgb_sha256"],
                        takeover_state_sha256=v1.canonical_sha(prefix[k]["dynamic_state"]),
                        prefix_sha256=v1.canonical_sha(prefix))
            doc = build_teacher_error(error=err, teacher=teacher, expected_pair=pair, prefix=prefix,
                partial_replay_path=root/"partial_replay.json", partial_actions_path=root/"partial_actions.json",
                verification_only=repeat, fallback_detected=True, fallback_executed=False)
            context = dict(metadata=rec.metadata, first_start=rec.first_start, takeover=rec.takeover,
                           agent_validation=rec._agent_validation(),
                           fallback_events=[dict(step=count, indicators=dict(episode_fallback_count=1))],
                           raw_reply=dict(rc=500, seq=42, msg=err.evidence["message"]),
                           raw_client_sequence=43, episode_fallback_count=1, no_terminal_result=True,
                           training_eligible=False, training_released=False)
            for name, data in (("teacher_error", doc), ("branch", doc), ("teacher_error_context", context)):
                self.write(root, name+".json", data)
            return root, expected, rec.replay
        result = dict(success=float(success), collision=float(collision), policy_init_valid=valid,
                      following_rate=.75, total_step=count, following_step=int(count*.75),
                      status="Normal", finish=True)
        rec.finish(result)
        if teacher == "student":
            self.prefix_images[v1.canonical_sha(rec.replay)] = [
                hashlib.sha256((root/f"rgb_{i:04d}.png").read_bytes()).hexdigest()
                for i in range(count)]
        if teacher != "student":
            adm = self.read(root, "admission.json")
            branch = dict(experiment=meta["experiment"], teacher=teacher, complete=True, replay_verified=True,
                          transport_fallback=False, task="stt", key="scene/1", takeover_step=k, seed=7,
                          protocol_sha256=meta["protocol_sha256"], prefix_sha256=v1.canonical_sha(prefix),
                          initial_rgb_sha256=prefix[0]["rgb_sha256"],
                          takeover_state_sha256=v1.canonical_sha(prefix[k]["dynamic_state"]),
                          actual_takeover_state_sha256=adm["actual_takeover_state_sha256"],
                          candidate_windows=adm["candidate_windows"], result=result, artifact_root=str(root),
                          verification_only=repeat)
            self.write(root, "branch.json", branch)
        return root, expected, rec.replay

    def start(self, prefix):
        return dict(rgb=prefix[0]["rgb_sha256"], state=prefix[0]["dynamic_state"])

    def teacher_case(self, **kwargs):
        _, _, prefix = self.make(version=kwargs.get("version", 1))
        root, exp, _ = self.make(teacher="oracle", k=3, prefix=prefix, success=True, count=28, **kwargs)
        return root, exp, prefix

    def audit(self, root, expected, prefix=None, original=None):
        if original is None:
            own = self.read(root, "replay.json") if prefix is None else prefix
            original = self.start(own)
        return audit_branch(root, expected=expected, student_prefix=prefix, original_start=original)

    def test_v1_student_failure_and_rerun_success_never_demonstrations(self):
        for success in (False, True):
            root, e, _ = self.make(success=success)
            result = self.audit(root, e)
            self.assertEqual(result["candidate_indices"], [])
            self.assertFalse(result["training_released"])
            self.assertEqual(result["result"]["success"], float(success))

    def test_v1_v2_teacher_windows_rederived_and_full_inventory(self):
        for version in (1, 2):
            root, e, prefix = self.teacher_case(version=version)
            result = self.audit(root, e, prefix)
            self.assertEqual(result["kind"], "normal")
            self.assertGreater(result["candidate_windows"], 0)
            self.assertGreaterEqual(min(result["candidate_indices"]), 3)
            self.assertEqual(len(result["source_files"]), len(list(root.iterdir())))
            self.assertFalse(result["actualproof"]["raw_alpha_reconstruction"])
            self.assertFalse(result["actualproof"]["target_base_position_reconstruction"])
            self.assertFalse(result["actualproof"]["raw_sensor_digest_independently_recomputed"])
            self.assertEqual(result["actualproof"]["png_prefix_frames_verified"], 4)
            self.assertEqual(result["image_sha256"][:4], e["prefix_image_sha256"][:4])
            self.assertEqual(result["branch"], self.read(root, "branch.json"))

    def test_missing_or_wrong_student_png_pins_fail(self):
        root, e, prefix = self.teacher_case()
        for bad in ({k:v for k,v in e.items() if k != "prefix_image_sha256"},
                    dict(e, prefix_image_sha256=["0"*64]*len(prefix)),
                    dict(e, prefix_image_sha256=e["prefix_image_sha256"][:3])):
            with self.assertRaises(ValueError):
                self.audit(root, bad, prefix)

    def test_mutated_teacher_prefix_png_fails_independently_of_raw_digest(self):
        root, e, prefix = self.teacher_case()
        Image.fromarray(np.full((4, 5, 3), 99, dtype=np.uint8)).save(root/"rgb_0002.png")
        with self.assertRaisesRegex(ValueError, "prefix RGB PNG SHA"):
            self.audit(root, e, prefix)

    def test_student_returns_decoded_png_pins_without_claiming_rgba_reconstruction(self):
        root, e, _ = self.make()
        result = self.audit(root, e)
        self.assertEqual(len(result["image_sha256"]), 24)
        self.assertFalse(result["actualproof"]["raw_alpha_reconstruction"])
        self.assertIn("original PNG not supplied", result["actualproof"]["original_start_rgb_proof"])
        for i, digest in enumerate(result["image_sha256"]):
            self.assertEqual(digest, hashlib.sha256((root/f"rgb_{i:04d}.png").read_bytes()).hexdigest())

    def test_repeat_success_is_proof_only(self):
        root, e, prefix = self.teacher_case(repeat=True)
        self.assertEqual(self.audit(root, e, prefix)["candidate_indices"], [])

    def test_failed_teacher_has_no_candidate(self):
        _, _, prefix = self.make()
        root, e, _ = self.make(teacher="lightnav", k=3, prefix=prefix, collision=True)
        self.assertEqual(self.audit(root, e, prefix)["candidate_windows"], 0)

    def test_student_invalid_initialization_is_diagnostic(self):
        root, e, _ = self.make(valid=False)
        result = self.audit(root, e)
        self.assertFalse(result["result"]["policy_init_valid"])
        self.assertEqual(result["candidate_indices"], [])

    def test_pinned_file_tampering_fails(self):
        root, e, prefix = self.teacher_case()
        pins = self.audit(root, e, prefix)["source_files"]
        with (root/"metadata.json").open("a") as stream:
            stream.write(" ")
        e["source_files"] = pins
        with self.assertRaisesRegex(ValueError, "source SHA"):
            self.audit(root, e, prefix)

    def test_symlink_and_extra_terminal_file_fail(self):
        for kind in ("symlink", "extra"):
            root, e, prefix = self.teacher_case()
            if kind == "symlink":
                target = root/"rgb_0000.png"
                saved = self.base/f"saved{self.serial}.png"
                target.rename(saved); target.symlink_to(saved)
            else:
                self.write(root, "ERROR.json", {})
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.audit(root, e, prefix)

    def test_mutated_action_ownership_and_execution_fail(self):
        for field, value in (("owner", "student"), ("normalized_action", [.9, 0., 0.])):
            root, e, prefix = self.teacher_case()
            acts = self.read(root, "actions.json"); acts[3][field] = value
            self.write(root, "actions.json", acts)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.audit(root, e, prefix)

    def test_prefix_state_action_and_rgb_tampering_fail(self):
        for field in ("state", "action", "rgb"):
            root, e, prefix = self.teacher_case()
            replay = self.read(root, "replay.json")
            if field == "state":
                replay[1]["dynamic_state"]["agents"][0]["joints"][0] += .2
                replay[1]["dynamic_state_sha256"] = v1.canonical_sha(replay[1]["dynamic_state"])
            elif field == "action":
                replay[1]["action"] = [.8, 0., 0.]
            else:
                replay[1]["rgb_sha256"] = "0"*64
            self.write(root, "replay.json", replay)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.audit(root, e, prefix)

    def test_timestamp_pose_rotation_mutations_fail(self):
        for field, value in (("timestamp_s", .0), ("robot_position_world", [9., 0., 0.]),
                              ("robot_rotation_world_from_body", np.zeros((3, 3)).tolist())):
            root, e, prefix = self.teacher_case()
            obs = self.read(root, "observations.json"); obs[2][field] = value
            self.write(root, "observations.json", obs)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.audit(root, e, prefix)

    def test_raw_and_candidate_window_mutations_fail(self):
        for filename, field in (("raw_windows.json", "trajectory_xy_m"),
                                 ("windows.json", "teacher_owned_action_indices"),
                                 ("windows.json", "future_bracket_indices")):
            root, e, prefix = self.teacher_case()
            wins = self.read(root, filename)
            wins[0][field][0] = [999., 999.] if field != "teacher_owned_action_indices" else 0
            self.write(root, filename, wins)
            with self.subTest(filename=filename, field=field), self.assertRaises(ValueError):
                self.audit(root, e, prefix)

    def test_student_or_repeat_fake_windows_fail(self):
        for repeat in (False, True):
            if repeat:
                root, e, prefix = self.teacher_case(repeat=True)
            else:
                root, e, prefix = self.make(); prefix = None
            self.write(root, "windows.json", self.read(root, "raw_windows.json")[:1])
            with self.subTest(repeat=repeat), self.assertRaises(ValueError):
                self.audit(root, e, prefix)

    def test_fallback_repeat_role_and_original_start_fail(self):
        root, e, prefix = self.teacher_case()
        self.write(root, "fallback_events.json", [{"reply_error": "bad"}])
        with self.assertRaises(ValueError):
            self.audit(root, e, prefix)
        root, e, prefix = self.teacher_case()
        with self.assertRaises(ValueError):
            self.audit(root, dict(e, verification_only=True), prefix)
        with self.assertRaises(ValueError):
            self.audit(root, e, prefix, dict(self.start(prefix), rgb="f"*64))

    def test_template_and_partial_normal_fail(self):
        root, e, prefix = self.teacher_case()
        meta = self.read(root, "metadata.json"); meta["initial_bbox_rgb_xyxy"] = [0, 0, 2, 2]
        self.write(root, "metadata.json", meta)
        with self.assertRaises(ValueError):
            self.audit(root, e, prefix)
        root, e, prefix = self.teacher_case()
        acts = self.read(root, "actions.json"); acts.pop()
        self.write(root, "actions.json", acts)
        with self.assertRaises(ValueError):
            self.audit(root, e, prefix)

    def test_nonfinite_duplicate_json_and_bool_step_fail(self):
        for kind in ("nan", "duplicate", "bool"):
            root, e, prefix = self.teacher_case()
            if kind == "nan":
                (root/"result.json").write_text('{"success":NaN}')
            elif kind == "duplicate":
                (root/"result.json").write_text('{"success":0,"success":1}')
            else:
                obs = self.read(root, "observations.json"); obs[0]["sim_step"] = False
                self.write(root, "observations.json", obs)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.audit(root, e, prefix)

    def error_case(self, count=2):
        _, _, prefix = self.make(version=2)
        root, e, _ = self.make(version=2, teacher="lightnav", k=3, prefix=prefix, error=True, count=count)
        return root, e, prefix

    def test_typed_error_before_and_after_takeover_is_partial_no_label(self):
        for count in (2, 5):
            root, e, prefix = self.error_case(count)
            result = self.audit(root, e, prefix)
            self.assertEqual(result["kind"], "teacher_error")
            self.assertIsNone(result["result"])
            self.assertEqual(result["candidate_indices"], [])
            self.assertFalse(result["branch"]["complete"])

    def test_error_external_path_and_fake_terminal_fail(self):
        for change in ("path", "terminal", "context"):
            root, e, prefix = self.error_case()
            if change == "path":
                doc = self.read(root, "teacher_error.json")
                doc["observed_partial"]["evidence_files"]["partial_replay"]["path"] = "/outside/replay.json"
                self.write(root, "teacher_error.json", doc); self.write(root, "branch.json", doc)
            elif change == "terminal":
                self.write(root, "result.json", {"success":1})
            else:
                context = self.read(root, "teacher_error_context.json")
                context["raw_reply"]["msg"] = "different"
                self.write(root, "teacher_error_context.json", context)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.audit(root, e, prefix)

    def test_error_executed_fallback_and_altered_partial_fail(self):
        for change in ("fallback", "action"):
            root, e, prefix = self.error_case()
            if change == "fallback":
                doc = self.read(root, "teacher_error.json"); doc["fallback_executed"] = True
                self.write(root, "teacher_error.json", doc); self.write(root, "branch.json", doc)
            else:
                acts = self.read(root, "partial_actions.json")
                acts[0]["normalized_action"] = [.6, 0., 0.]
                self.write(root, "partial_actions.json", acts)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.audit(root, e, prefix)

    def test_foreign_protocol_and_unrecognized_expected_fields_fail(self):
        root, e, prefix = self.teacher_case()
        for bad in (dict(e, protocol_sha256=V2_SHA), dict(e, unknown_skip=True)):
            with self.assertRaises(ValueError):
                self.audit(root, bad, prefix)

    def test_metadata_pin_and_zero_frame_are_not_silently_admitted(self):
        root, e, prefix = self.teacher_case()
        with self.assertRaises(ValueError):
            self.audit(root, dict(e, metadata={"source_dataset_sha256":"0"*64}), prefix)
        self.write(root, "observations.json", [])
        self.write(root, "actions.json", [])
        self.write(root, "replay.json", [])
        with self.assertRaisesRegex(ValueError, "zero-frame"):
            self.audit(root, e, prefix)


if __name__ == "__main__":
    unittest.main()
