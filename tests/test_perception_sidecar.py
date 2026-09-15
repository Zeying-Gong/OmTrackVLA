"""CPU tests of pinned sidecar wiring, independent validity and input binding."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

# The repository data package imports Torch for unrelated datasets. Load this
# deliberately NumPy-only validation module directly on CPU-only review hosts.
MODULE_PATH = Path(__file__).resolve().parents[1]/"omtrackvla/data/perception_sidecar.py"
spec = importlib.util.spec_from_file_location("perception_sidecar_under_test", MODULE_PATH)
sidecar = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sidecar
spec.loader.exec_module(sidecar)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


class Fixture:
    """Minimal consumed schemas; no fake simulator, model or panoptic decoder."""
    repository = "/pinned/repository"

    def __init__(self, root):
        self.root, self.refs, self.samples, self.label_paths = root, {}, {}, {}
        self.manifest = {"samples": []}
        roles = {row[2]: "train" for row in sidecar.EXPECTED.values()}
        self.plan = dict(stage="train4_perception_sidecar_collection_v1", repository=self.repository,
            output_root=self.repository+"/outputs/labels", permanent_scene_roles=roles, entries=[],
            optimizer_input_allowed=False, formal_training_eligible=False, product_acceptance_evidence=False,
            test_locked_used=False)
        verification_sources = []
        for run_index, (run_id, (task, index, scene, episode, count)) in enumerate(sidecar.EXPECTED.items()):
            assigned = {"0": 100+run_index, "2": 200+run_index}
            init = dict(environment_step=0, bbox_xyxy=[0,0,3,2], bbox_xyxy_norm=[0.,0.,.75,2/3])
            actions = [[.1,0.,0.] for _ in range(count)]
            source = dict(split="train", task=task, dataset_index=index, episode_id=episode,
                scene_id=f"data/scenes/{scene}.glb", assigned_humanoid_semantic_ids=assigned,
                initialization=init, summary=dict(episode_finished=True, termination_reason="episode_over", steps=count),
                steps=[dict(step=i+1, policy={"action": dict(zip(("forward","lateral","yaw"), action))})
                       for i, action in enumerate(actions)])
            source_ref = self.document(f"sources/{run_id}/result.json", source)
            entry = dict(run_id=run_id, task=task, dataset_index=index, episode_id=episode,
                canonical_scene_id=scene, scene_id=source["scene_id"], action_count=count, observation_count=count+1,
                permanent_partition_role="train", source=source_ref, initialization=init,
                assigned_humanoid_semantic_ids=assigned, saved_actions_sha256=sidecar._canonical(actions),
                source_status=self.document(f"sources/{run_id}/status.json", dict(status="fixed_train_rollout_complete",
                    exit_code=0, result_sha256=source_ref["sha256"])),
                source_launch=self.document(f"sources/{run_id}/launch.json", {"run": {"original_initialization": init}}),
                output_dir=self.repository+f"/outputs/labels/{run_id}", prefix_samples=[])
            labels = []
            for step in range(count+1):
                visible = step % 4 != 1
                box = None if not visible else ([1,1,1,1] if step % 4 == 2 else [0,0,3,2])
                target = dict(semantic_id_label_side_only=assigned["0"], visible=visible,
                    mask_area_pixels=0 if box is None else (1 if step % 4 == 2 else 12),
                    bbox_xyxy_inclusive=box, bbox_xyxy_norm=None if box is None else [box[0]/4,box[1]/3,box[2]/4,box[3]/3],
                    bbox_label_valid=visible and step % 4 != 2)
                labels.append(dict(environment_step=step, policy_call_index=step, terminal_observation=step==count,
                    after_source_action_step=None if step==0 else step, next_source_action_step=None if step==count else step+1,
                    world_time_s=step*.05, image_height=3, image_width=4, target=target,
                    rgb_array_sha256=sidecar._array_hash(self.pixels(run_index,step)), visibility_label_valid=True,
                    gt_used_only_on_label_or_audit_side=True, optimizer_input_allowed=False, formal_training_eligible=False,
                    **{key:False for key in sidecar.UNAVAILABLE},
                    source_audit=dict(passed=True, source_result_sha256=source_ref["sha256"],
                        action_just_replayed=None if step==0 else actions[step-1],
                        capture_evidence={"assigned_humanoid_semantic_ids": assigned})))
            labels_path = self.local(entry["output_dir"]+"/labels.jsonl")
            labels_path.parent.mkdir(parents=True)
            self.label_paths[run_id] = labels_path
            self.write_labels(run_id, labels)
            anchors = ((2,3,4), (2,3,4,5,6), (3,4,5,6,7), (2,3,4,5,6,7))[run_index]
            for anchor in anchors:
                folder = f"samples/{run_id}/anchor{anchor}"
                initial = self.picture(folder+"/initial.png", self.pixels(run_index,0))
                frames, frozen = [], []
                for step in range(anchor+1):
                    ref = self.picture(folder+f"/prefix{step}.png", self.pixels(run_index,step))
                    frames.append(dict(rgb_path=f"prefix{step}.png", sha256=ref["sha256"], environment_step=step, world_time_s=step*.05))
                    frozen.append(dict(ref, environment_step=step, world_time_s=step*.05,
                                       rgb_array_sha256=labels[step]["rgb_array_sha256"]))
                inputs = dict(condition_mode="visual_only", initial_rgb={"rgb_path":"initial.png","sha256":initial["sha256"]},
                    initial_bbox_xyxy_norm=init["bbox_xyxy_norm"], prefix_rgb=frames,
                    policy_calls=[dict(policy_call_index=i, observation_environment_step=i, world_time_s=i*.05,
                                       rgb_history_observation_indices=[max(0,j) for j in range(i-3,i+1)]) for i in range(anchor+1)],
                    visual_initialization_valid=True, rgb_valid=True, binding_valid=True,
                    uwb=dict(valid=False, relative_position_base_xy_m=[0.,0.], covariance_base_xy_m2=[[1.,0.],[0.,1.]], quality_01=0., age_s=0.),
                    camera_intrinsics=[[2.,0.,2.],[0.,1.5,1.5],[0.,0.,1.]], camera_from_base=np.eye(4).tolist())
                sample = dict(schema_version=2, stage="phase3_recovery_sequence_candidate_v2", formal_training_eligible=False,
                    test_locked_used=False, source=dict(split="train",task=task,dataset_index=index,episode_id=episode,
                        scene_id=source["scene_id"],rollout_result=source_ref["path"],rollout_result_sha256=source_ref["sha256"],
                        anchor_environment_step=anchor), model_inputs=inputs,
                    sequence_contract=dict(starts_at_episode_reset=True,history_size=4,
                        policy_call_indices=list(range(anchor+1)),anchor_policy_call_index=anchor))
                sample_ref = self.document(folder+"/sample.json",sample)
                self.samples[(run_id,anchor)] = (self.local(sample_ref["path"]),sample)
                reference = dict(sample=sample_ref,anchor_environment_step=anchor,initial_rgb=initial,prefix_rgb=frozen)
                entry["prefix_samples"].append(reference)
                report = self.document(folder+"/report.json",{"status":"passed"})
                self.manifest["samples"].append(dict(partition_role="train",identity={"scene_id":scene,"anchor_environment_step":anchor},
                    artifacts={"sample":sample_ref,"rollout":source_ref,"report":report}))
            self.plan["entries"].append(entry)
            verification_sources.append(dict(run_id=run_id,status="independently_verified",observations_verified=count+1,
                distinct_original_samples=len(anchors),labels_sha256=digest(labels_path)))
        self.plan["permanent_manifest"] = self.document("permanent.json",{"samples":[
            dict(identity={"scene_id":scene},partition_role=role) for scene,role in roles.items()]})
        self.plan["current_manifest"] = self.document("manifest.json",self.manifest)
        self.plan_path = self.root/"plan.json"
        self.verification_path = self.root/"verification.json"
        self.verification = dict(status="verified_development_perception_labels_pending_loader_integration",
            optimizer_input_allowed=False,formal_training_eligible=False,product_acceptance_evidence=False,
            partial_admission_allowed=False,model_loaded=False,gt_training_inputs_generated=False,source_files_modified=False,
            original_inputs_unchanged=True,batch_summary_valid=True,source_count=4,admitted_observation_count=370,
            expected_observation_count=370,required_distinct_original_samples=19,sources=verification_sources,
            distinct_original_sample_paths=[ref["artifacts"]["sample"]["path"] for ref in self.manifest["samples"]])
        self.seal()

    @staticmethod
    def pixels(run_index, step):
        return np.full((3,4,3), run_index*30+step, dtype=np.uint8)

    def local(self, logical):
        return self.root / logical.removeprefix(self.repository+"/")

    def ref(self, relative):
        path = self.root/relative
        ref = {"path":self.repository+"/"+relative,"sha256":digest(path),"bytes":path.stat().st_size}
        self.refs[ref["path"]] = ref
        return ref

    def document(self, relative, value):
        write(self.root/relative,value)
        return self.ref(relative)

    def picture(self, relative, pixels):
        path = self.root/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(pixels).save(path)
        return self.ref(relative)

    def write_labels(self, run_id, rows):
        self.label_paths[run_id].write_text("\n".join(json.dumps(r,allow_nan=False) for r in rows)+"\n",encoding="utf-8")

    def labels(self, run_id):
        return [json.loads(line) for line in self.label_paths[run_id].read_text().splitlines()]

    def seal(self):
        # Test-only trust root refresh after deliberate semantic mutations.
        # In production the caller supplies externally frozen expected hashes.
        for ref in self.refs.values():
            ref["sha256"] = digest(self.local(ref["path"]))
        write(self.root/"manifest.json",self.manifest)
        self.plan["current_manifest"]["sha256"] = digest(self.root/"manifest.json")
        self.plan["artifacts"] = list(self.refs.values())
        self.plan.pop("plan_sha256",None)
        self.plan["plan_sha256"] = sidecar._canonical(self.plan)
        write(self.plan_path,self.plan)
        self.verification["plan_sha256"] = self.plan["plan_sha256"]
        self.verification["plan_file_sha256"] = digest(self.plan_path)
        for row in self.verification["sources"]:
            row["labels_sha256"] = digest(self.label_paths[row["run_id"]])
        write(self.verification_path,self.verification)
        self.verification_sha = digest(self.verification_path)
        self.plan_sha = digest(self.plan_path)

    def store(self):
        return sidecar.PerceptionSidecarStore(self.verification_path,verification_sha256=self.verification_sha,
            plan_path=self.plan_path,plan_file_sha256=self.plan_sha,artifact_root=self.root)

    def selected(self):
        return self.samples[("stt_0000_128steps",7)][0]

    def batch(self, path=None):
        path = path or self.selected()
        sample = json.loads(path.read_text())
        inputs = sample["model_inputs"]
        frames = [np.asarray(Image.open(path.parent/f["rgb_path"]),dtype=np.float32).transpose(2,0,1)/np.float32(255.)
                  for f in inputs["prefix_rgb"]]
        size = len(frames)
        tensor = lambda value: np.asarray(value,dtype=np.float32)[None]
        batch = dict(initial_rgb=frames[0][None].copy(),initial_bbox=tensor(inputs["initial_bbox_xyxy_norm"]),
            ego_rgb=np.stack([np.stack([frames[j] for j in c["rgb_history_observation_indices"]]) for c in inputs["policy_calls"]])[None],
            visual_initialization_valid=tensor(1.),rgb_valid=tensor(1.),binding_valid=tensor(1.),
            uwb_xy=tensor([0.,0.]),uwb_covariance_xy=tensor([[1.,0.],[0.,1.]]),uwb_quality=tensor(0.),uwb_age_s=tensor(0.),uwb_valid=tensor(0.),
            camera_intrinsics=tensor(inputs["camera_intrinsics"]),camera_from_base=tensor(inputs["camera_from_base"]),
            supervision_mask=np.zeros((1,size),bool),target_waypoints=np.zeros((1,size,8,2),np.float32),waypoint_mask=np.zeros((1,size,8),bool),
            **{key:np.zeros((1,size),bool) for key in ("stop_label_valid","binding_label_valid","identity_label_valid","ego_label_valid")})
        batch["supervision_mask"][0,-1] = True
        batch["waypoint_mask"][0,-1] = True
        batch["target_waypoints"][0,-1,:,0] = np.arange(8,dtype=np.float32)*.05
        return batch


class PerceptionSidecarTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="perception-loader-test-")
        self.addCleanup(self.temporary.cleanup)
        self.fixture = Fixture(Path(self.temporary.name))

    def test_nineteen_prefixes_and_visibility_bbox_validity_are_independent(self):
        store = self.fixture.store()
        self.assertEqual(len(store.sample_paths),19)
        supervision = store.supervision_for(self.fixture.selected())
        values = supervision.as_numpy()
        self.assertEqual(supervision.learning_start,4)
        self.assertEqual(values["target_visible"].tolist(),[0,0,0,0,1,0,1,1])
        self.assertEqual(values["visibility_label_valid"].tolist(),[False]*4+[True]*4)
        self.assertEqual(values["bbox_label_valid"].tolist(),[False]*4+[True,False,False,True])
        self.assertFalse(values["target_bbox"][:4].any())
        self.assertFalse(values["target_bbox"][5:7].any())
        self.assertEqual(set(values),set(sidecar.PERCEPTION_LABEL_KEYS))
        self.assertFalse(supervision.optimizer_input_allowed)
        self.assertFalse(supervision.formal_training_eligible)

    def test_overlay_is_new_and_preserves_inputs_waypoints_and_unknown_auxiliary_masks(self):
        original = self.fixture.batch()
        before = {key:value.copy() for key,value in original.items()}
        supervision = self.fixture.store().supervision_for(self.fixture.selected())
        combined = sidecar.overlay_recovery_sequence(original,supervision)
        self.assertIsNot(combined,original)
        for key in original:
            np.testing.assert_array_equal(original[key],before[key])
        for key in sidecar.MODEL_KEYS | (sidecar.RECOVERY_LABEL_KEYS-{"supervision_mask"}):
            self.assertIs(combined[key],original[key])
        self.assertEqual(combined["supervision_mask"].tolist(),[[False]*4+[True]*4])
        self.assertEqual(combined["waypoint_mask"].any(-1).sum(),1)
        self.assertEqual(combined["target_bbox"].shape,(1,8,4))

    def test_optimizer_access_is_explicitly_rejected_at_every_api(self):
        store = self.fixture.store()
        supervision = store.supervision_for(self.fixture.selected())
        calls = [lambda:store.supervision_for(self.fixture.selected(),for_optimizer=True),
                 lambda:supervision.as_numpy(for_optimizer=True),lambda:supervision.as_tensors(for_optimizer=True),
                 lambda:sidecar.overlay_recovery_sequence(self.fixture.batch(),supervision,for_optimizer=True)]
        for call in calls:
            with self.subTest(call=call),self.assertRaisesRegex(ValueError,"optimizer"):
                call()

    def test_wrong_same_length_source_rgb_cannot_receive_labels(self):
        supervision = self.fixture.store().supervision_for(self.fixture.selected())
        wrong = self.fixture.batch(self.fixture.samples[("stt_3100_128steps",7)][0])
        with self.assertRaisesRegex(ValueError,"RGB"):
            sidecar.overlay_recovery_sequence(wrong,supervision)

    def test_gt_input_and_unknown_auxiliary_activation_are_rejected(self):
        supervision = self.fixture.store().supervision_for(self.fixture.selected())
        contaminated = self.fixture.batch()
        contaminated["panoptic"] = np.zeros((1,8,3,4),np.uint32)
        with self.assertRaisesRegex(ValueError,"unexpected"):
            sidecar.overlay_recovery_sequence(contaminated,supervision)
        contaminated = self.fixture.batch()
        contaminated["binding_label_valid"][0,-2] = True
        with self.assertRaisesRegex(ValueError,"unknown auxiliary"):
            sidecar.overlay_recovery_sequence(contaminated,supervision)

    def test_labels_changed_after_loading_fail_even_for_another_source(self):
        store = self.fixture.store()
        supervision = store.supervision_for(self.fixture.selected())
        path = self.fixture.label_paths["at_0401_128steps"]
        path.write_text(path.read_text()+" ",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"SHA-256"):
            store.supervision_for(self.fixture.selected())
        with self.assertRaisesRegex(ValueError,"SHA-256"):
            supervision.as_numpy()

    def test_verification_hash_is_an_external_required_pin(self):
        self.fixture.verification_path.write_text("{}",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"SHA-256"):
            self.fixture.store()

    def test_partial_admission_is_rejected_even_when_repinned(self):
        self.fixture.verification["sources"].pop()
        self.fixture.seal()
        with self.assertRaisesRegex(ValueError,"source set"):
            self.fixture.store()

    def test_real_source_assignment_is_checked_before_label_assignment(self):
        entry = self.fixture.plan["entries"][0]
        path = self.fixture.local(entry["source"]["path"])
        value = json.loads(path.read_text())
        value["assigned_humanoid_semantic_ids"]["0"] += 10
        write(path,value)
        self.fixture.seal()
        with self.assertRaisesRegex(ValueError,"source semantic assignment mismatch"):
            self.fixture.store()

    def test_duplicate_prefix_denominator_is_rejected(self):
        refs = self.fixture.plan["entries"][0]["prefix_samples"]
        refs[1] = copy.deepcopy(refs[0])
        self.fixture.seal()
        with self.assertRaisesRegex(ValueError,"duplicate"):
            self.fixture.store()

    def test_relabelled_val_role_cannot_bypass_permanent_scene_table(self):
        self.fixture.plan["entries"][0]["permanent_partition_role"] = "val"
        self.fixture.seal()
        with self.assertRaisesRegex(ValueError,"val/held-out"):
            self.fixture.store()

    def test_legacy_or_unlisted_sample_is_not_silently_filled_with_labels(self):
        path = self.fixture.root/"legacy.json"
        write(path,{})
        with self.assertRaisesRegex(ValueError,"nineteen"):
            self.fixture.store().supervision_for(path)

    def test_terminal_cannot_be_relabelled_as_a_policy_observation(self):
        rows = self.fixture.labels("stt_0000_128steps")
        rows[7]["terminal_observation"] = True
        self.fixture.write_labels("stt_0000_128steps",rows)
        self.fixture.seal()
        with self.assertRaisesRegex(ValueError,"causality"):
            self.fixture.store()

    def test_prefix_clock_mismatch_is_not_hidden_by_matching_pixels(self):
        rows = self.fixture.labels("stt_0000_128steps")
        rows[7]["world_time_s"] += .001
        self.fixture.write_labels("stt_0000_128steps",rows)
        self.fixture.seal()
        with self.assertRaisesRegex(ValueError,"alignment"):
            self.fixture.store().supervision_for(self.fixture.selected())

    def test_label_pixel_mismatch_is_rejected(self):
        rows = self.fixture.labels("stt_0000_128steps")
        rows[7]["rgb_array_sha256"] = "0"*64
        self.fixture.write_labels("stt_0000_128steps",rows)
        self.fixture.seal()
        with self.assertRaisesRegex(ValueError,"RGB differs"):
            self.fixture.store().supervision_for(self.fixture.selected())

    def test_future_rgb_history_is_rejected_even_after_repinned_sample(self):
        path,sample = self.fixture.samples[("stt_0000_128steps",7)]
        sample["model_inputs"]["policy_calls"][2]["rgb_history_observation_indices"][-1] = 3
        write(path,sample)
        self.fixture.seal()
        with self.assertRaisesRegex(ValueError,"alignment"):
            self.fixture.store().supervision_for(path)

    def test_current_prefix_file_is_rechecked(self):
        store = self.fixture.store()
        path = self.fixture.selected().parent/"prefix7.png"
        Image.fromarray(np.zeros((3,4,3),np.uint8)).save(path)
        with self.assertRaisesRegex(ValueError,"SHA-256"):
            store.supervision_for(self.fixture.selected())

    def test_degenerate_bbox_cannot_be_marked_valid(self):
        rows = self.fixture.labels("stt_0000_128steps")
        rows[6]["target"]["bbox_label_valid"] = True
        self.fixture.write_labels("stt_0000_128steps",rows)
        self.fixture.seal()
        with self.assertRaisesRegex(ValueError,"independent validity"):
            self.fixture.store()

    @unittest.skipUnless(importlib.util.find_spec("torch"),"Torch unavailable on this CPU review host")
    def test_real_torch_overlay_outputs_match_numpy_without_input_changes(self):
        import torch
        numpy_batch = self.fixture.batch()
        original = {key:torch.from_numpy(value.copy()) for key,value in numpy_batch.items()}
        supervision = self.fixture.store().supervision_for(self.fixture.selected())
        actual = sidecar.overlay_recovery_sequence(original,supervision)
        expected = sidecar.overlay_recovery_sequence(numpy_batch,supervision)
        for key in actual:
            np.testing.assert_array_equal(actual[key].numpy(),expected[key])
        self.assertEqual(supervision.as_tensors()["target_bbox"].dtype,torch.float32)
        self.assertEqual(actual["visibility_label_valid"].dtype,torch.bool)
        for key in sidecar.MODEL_KEYS:
            self.assertIs(actual[key],original[key])


if __name__ == "__main__":
    unittest.main()
