"""CPU integration tests using actual tiny PNG/schema-2/frozen manifests."""
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

import torch

from omtrackvla.data.recovery_sequence import recovery_sequence_collate, validate_recovery_sample
from omtrackvla.data.recovery_sequence_groups import VariableRecoverySequences
from omtrackvla.training.sequence_training import phase3_sequence_loss, sequence_model_inputs
from scripts.build_recovery_scene_manifest import build_manifest, freeze_scene_plan, sha256


# Reuse the established complete schema fixture, not a mock validator. This
# test is installed beside the existing test_recovery_sequence.py.
_spec = importlib.util.spec_from_file_location(
    "_recovery_sequence_group_fixture", Path(__file__).with_name("test_recovery_sequence.py"))
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)


def make_scene_sample(root, scene, anchor, episode="4"):
    path, sample, report, save = _fixture.make_fixture(root, anchor=anchor)
    scene_id = f"data/scene_datasets/hm3d/train/{scene}/{scene}.basis.glb"
    dataset_index = int(episode) - 4
    sample["source"].update(scene_id=scene_id, episode_id=episode, dataset_index=dataset_index)
    sample["sample_id"] = f"evt_train/stt/{episode}/post-action-{anchor:03d}/sequence-v2"
    report.update(scene_id=scene_id, episode_id=episode, dataset_index=dataset_index)
    rollout_path = Path(sample["source"]["rollout_result"])
    rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
    rollout.update(scene_id=scene_id, episode_id=episode, dataset_index=dataset_index)
    _fixture.write(rollout_path, rollout)
    sample["source"]["rollout_result_sha256"] = sha256(rollout_path)
    save()
    return path


class VariableRecoverySequenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = []
        for scene in ("scene00001", "scene00002"):
            for index, (anchor, episode) in enumerate(((2, "4"), (5, "4"), (9, "4"), (9, "5"))):
                self.paths.append(make_scene_sample(self.root/f"{scene}_{index}", scene, anchor, episode))
        self.parent, self.protocol = self.root/"parent.ckpt", self.root/"collection.json"
        self.parent.write_bytes(b"fixture parent identity; no torch checkpoint is loaded")
        self.protocol.write_text('{"scope":"tiny_fixture_only"}', encoding="utf-8")
        self.plan_path, self.manifest_path = self.freeze(self.paths, "initial")

    def tearDown(self):
        self.temp.cleanup()

    def freeze(self, paths, name, *, prior_manifests=(), seed=41):
        plan_path, manifest_path = self.root/f"{name}_plan.json", self.root/f"{name}_manifest.json"
        validator = lambda path: validate_recovery_sample(path, artifact_root=self.root)
        freeze_scene_plan(sample_paths=paths, output=plan_path, parent_checkpoint=self.parent,
                          collection_protocol=self.protocol, validator=validator, seed=seed,
                          min_train_scenes=1, min_val_scenes=1, required_tasks=("STT",),
                          val_fraction=.5, prior_manifests=prior_manifests)
        build_manifest(plan_path=plan_path, output=manifest_path, validator=validator)
        return plan_path, manifest_path

    def pool(self, role="train", *, training=True, manifest=None):
        return VariableRecoverySequences(manifest or self.manifest_path, partition_role=role,
                                         for_training=training, artifact_root=self.root,
                                         image_height=3, image_width=4)

    def test_groups_retain_manifest_order_and_unequal_real_lengths(self):
        pool = self.pool()
        self.assertEqual(len(pool), 4)
        self.assertEqual({group.sequence_length: len(group.manifest_indices) for group in pool.groups},
                         {3: 1, 6: 1, 10: 2})
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        expected = [record["artifacts"]["sample"]["path"] for record in manifest["samples"]
                    if record["partition_role"] == "train"]
        self.assertEqual([str(identity.sample_path) for identity in pool.identities], expected)
        self.assertEqual(sorted(i for group in pool.groups for i in group.manifest_indices), list(range(len(pool))))

    def test_each_item_keeps_every_call_and_only_slices_supervision(self):
        pool = self.pool()
        for index, identity in enumerate(pool.identities):
            with self.subTest(steps=identity.sequence_length, sample=identity.sample_id):
                item = pool.optimizer_item(index)
                sample = item.sample
                original = json.loads(identity.sample_path.read_text(encoding="utf-8"))
                self.assertEqual(sample["ego_rgb"].shape[0], len(original["model_inputs"]["policy_calls"]))
                self.assertEqual(item.burn_in_steps, max(0, identity.sequence_length-4))
                # Fixture RGB stores its actual observation index. Repetition
                # is allowed only for the existing reset-left history rule.
                for step in range(identity.sequence_length):
                    wanted = [max(0, j) for j in range(step-3, step+1)]
                    for history_index, observation in enumerate(wanted):
                        torch.testing.assert_close(sample["ego_rgb"][step, history_index],
                                                   torch.full((3, 3, 4), observation/255))
                prepared = item.as_batch(for_optimizer=True)
                model_inputs = sequence_model_inputs(prepared.batch)
                self.assertEqual(model_inputs["ego_rgb"].shape[1], identity.sequence_length)
                self.assertEqual(prepared.labels["supervision_mask"].tolist(),
                                 [[False]*(min(4, identity.sequence_length)-1)+[True]])
                self.assertFalse(prepared.labels["waypoint_mask"][:, :-1].any())
                self.assertTrue(prepared.labels["waypoint_mask"][:, -1].all())
                self.assertNotIn("audit_raw", prepared.batch)
                self.assertNotIn("stop_target", prepared.labels)
                for key in ("stop_label_valid", "binding_label_valid", "identity_label_valid", "ego_label_valid"):
                    self.assertFalse(prepared.labels[key].any())

    def test_actual_loss_gradient_selects_final_suffix_anchor_only(self):
        pool = self.pool()
        for index in range(len(pool)):
            prepared = pool.optimizer_item(index).as_batch(for_optimizer=True)
            count = prepared.labels["supervision_mask"].shape[1]
            predicted = torch.zeros(1, count, 8, 2, requires_grad=True)
            loss, _ = phase3_sequence_loss({"waypoints": predicted}, prepared.labels, {"waypoint": 1.})
            loss.backward()
            self.assertFalse(predicted.grad[:, :-1].any())
            self.assertGreater(float(predicted.grad[:, -1, 1:].norm()), 0.)

    def test_mixed_lengths_cannot_be_collated_by_inventing_padding(self):
        pool = self.pool()
        short = next(pool[index] for index, value in enumerate(pool.identities) if value.sequence_length == 3)
        long = next(pool[index] for index, value in enumerate(pool.identities) if value.sequence_length == 10)
        with self.assertRaisesRegex(ValueError, "padding"):
            recovery_sequence_collate([short.sample, long.sample])

    def test_uniform_sampling_uses_sample_count_not_length_group_count(self):
        pool = self.pool()
        calls = []
        class IndexRng:
            def randrange(self, count):
                calls.append(count)
                return len(calls)-1
        rng = IndexRng()
        observed = [pool.sample_uniform(rng, for_optimizer=True).identity.sample_path for _ in range(len(pool))]
        self.assertEqual(calls, [len(pool)]*len(pool))
        self.assertEqual(observed, [identity.sample_path for identity in pool.identities])
        # In particular both long samples retain separate equally selectable
        # entries; their group is not first selected with probability 1/3.
        self.assertEqual(len(set(observed)), 4)

    def test_caller_rng_restore_repeats_the_same_uniform_sample(self):
        pool = self.pool()
        rng = random.Random(704)
        state = rng.getstate()
        first = pool.sample_uniform(rng, for_optimizer=True).identity
        rng.setstate(state)
        second = pool.sample_uniform(rng, for_optimizer=True).identity
        self.assertEqual(first, second)

    def test_validation_is_rejected_by_every_explicit_optimizer_entry(self):
        with self.assertRaisesRegex(ValueError, "validation cannot"):
            self.pool("val", training=True)
        val = self.pool("val", training=False)
        self.assertEqual({identity.partition_role for identity in val.identities}, {"val"})
        with self.assertRaisesRegex(ValueError, "cannot be optimizer"):
            val.optimizer_item(0)
        with self.assertRaisesRegex(ValueError, "cannot be optimizer"):
            val.sample_uniform(random.Random(1), for_optimizer=True)
        item = val[0]
        self.assertFalse(item.optimizer_input_allowed)
        with self.assertRaisesRegex(ValueError, "cannot be optimizer"):
            item.as_batch(for_optimizer=True)
        self.assertEqual(item.as_batch().identity.partition_role, "val")

    def test_train_read_only_view_cannot_silently_become_optimizer_input(self):
        pool = self.pool(training=False)
        self.assertFalse(pool[0].optimizer_input_allowed)
        with self.assertRaisesRegex(ValueError, "read-only"):
            pool.optimizer_item(0)

    def test_mutating_frozen_scene_plan_is_rejected_after_construction(self):
        val = self.pool("val", training=False)
        plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        plan["scene_partitions"][val.identities[0].canonical_scene_id] = "train"
        _fixture.write(self.plan_path, plan)
        with self.assertRaisesRegex(ValueError, "frozen manifest/scene artifact changed"):
            val[0]

    def test_new_anchors_preserve_prior_validation_scene_identity(self):
        old = self.pool("val", training=False)
        additional = [make_scene_sample(self.root/f"additional_{scene}", scene, 14)
                      for scene in ("scene00001", "scene00002")]
        _, next_manifest = self.freeze(self.paths + additional, "extended",
                                       prior_manifests=[self.manifest_path], seed=940)
        new = self.pool("val", training=False, manifest=next_manifest)
        self.assertEqual({value.canonical_scene_id for value in old.identities},
                         {value.canonical_scene_id for value in new.identities})
        self.assertEqual(len(new), len(old)+1)
        self.assertTrue(all(value.partition_role == "val" for value in new.identities))
        self.assertFalse(any(new[index].optimizer_input_allowed for index in range(len(new))))

    def test_changed_candidate_fails_existing_strict_artifact_validation(self):
        pool = self.pool()
        path = pool.identities[0].sample_path
        sample = json.loads(path.read_text(encoding="utf-8"))
        sample["model_inputs"]["policy_calls"][0]["rgb_history_observation_indices"][-1] = 1
        _fixture.write(path, sample)
        with self.assertRaisesRegex(ValueError, "sample changed"):
            pool[0]

    def test_mutated_item_cannot_add_auxiliary_or_prefix_supervision(self):
        pool = self.pool()
        item = pool[0]
        item.sample["stop_label_valid"][-1] = True
        with self.assertRaisesRegex(ValueError, "no stop/visibility"):
            item.as_batch()
        item = pool[0]
        item.sample["supervision_mask"][0] = True
        with self.assertRaisesRegex(ValueError, "only the real anchor"):
            item.as_batch()


if __name__ == "__main__":
    unittest.main()
