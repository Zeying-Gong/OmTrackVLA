"""CPU alignment tests with real schema-2 media and frozen scene manifests."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

REVIEW = Path(__file__).resolve().parents[1]
REPOSITORY = Path(os.environ.get('OMTRACKVLA_REPOSITORY', REVIEW.parent.parent / 'OmTrackVLA_takeover'))

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

trainer = load_module('trainer_v4_under_review', REVIEW / 'scripts/train_recovery_sequence_v4.py')

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

try:
    import torch
except ImportError:
    torch = None


class ReadinessAndConfigTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text('{"fixture": "bound manifest"}')
        self.ready = self.root / 'training_readiness.json'
        self.record = {'training_ready': True, 'status': 'ready_for_development_pilot',
                       'validation_unchanged': True,
                       'manifest': {'path': str(self.manifest), 'sha256': file_hash(self.manifest)}}
        self.config = json.loads((REVIEW / 'configs/recovery_sequence_variable_prefix_v4.template.json').read_text())
        self.config['recovery_manifest'] = str(self.manifest)
        self.config['training_readiness'] = str(self.ready)
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        self.ready.write_text(json.dumps(self.record))
        self.config['training_readiness_sha256'] = file_hash(self.ready)

    def verify(self):
        return trainer.verify_training_readiness(self.root, self.config, self.manifest, file_hash)

    def test_template_stays_unrunnable_until_both_manifest_and_readiness_are_frozen(self):
        template = json.loads((REVIEW / 'configs/recovery_sequence_variable_prefix_v4.template.json').read_text())
        base = {'method': 'architecture_v1_end_to_end', 'test_locked_used': False}
        with self.assertRaisesRegex(ValueError, 'root must freeze'):
            trainer.effective_configuration(template, base, output_dir=self.root / 'new_run')
        template['recovery_manifest'] = str(self.manifest)
        with self.assertRaisesRegex(ValueError, 'readiness'):
            trainer.effective_configuration(template, base, output_dir=self.root / 'new_run')
        effective = trainer.effective_configuration(self.config, base, output_dir=self.root / 'new_run')
        self.assertEqual(effective['resolved_maximum_steps'], 64)
        self.assertEqual(effective['pilot']['training']['safe_stop_waypoint_multiplier'], 8)
        for section, key, value in [('training', 'burn_in_steps', 6), ('training', 'maximum_steps', 128),
                                    ('training', 'safe_stop_waypoint_multiplier', 1),
                                    ('training', 'recovery_weight', 1),
                                    ('validation', 'clean_relative_tolerance', .06)]:
            changed = json.loads(json.dumps(self.config))
            changed[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                trainer.effective_configuration(changed, base, output_dir=self.root / 'new_run')

    def test_readiness_is_bound_to_exact_bytes_and_manifest(self):
        path, digest, record = self.verify()
        self.assertEqual((path, digest), (self.ready, file_hash(self.ready)))
        self.assertEqual(record, self.record)
        self.ready.write_text(self.ready.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'readiness SHA-256'):
            self.verify()
        self.save()
        self.manifest.write_text('{"fixture":"changed"}')
        with self.assertRaisesRegex(ValueError, 'manifest hash'):
            self.verify()

    def test_unready_status_or_changed_validation_cannot_start(self):
        for field, value in [('training_ready', False), ('training_ready', 1),
                             ('status', 'blocked_missing_predeclared_loss_coverage'),
                             ('validation_unchanged', False), ('validation_unchanged', 1)]:
            original = self.record[field]
            self.record[field] = value
            self.save()
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, 'not satisfied'):
                self.verify()
            self.record[field] = original

    def test_ready_for_another_manifest_path_is_rejected_even_if_bytes_equal(self):
        other = self.root / 'other_manifest.json'
        other.write_bytes(self.manifest.read_bytes())
        self.record['manifest']['path'] = str(other)
        self.save()
        with self.assertRaisesRegex(ValueError, 'another manifest path'):
            self.verify()


@unittest.skipUnless(torch is not None, 'Torch is absent on this local CPU host')
class VariablePrefixTrainerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.fixture = load_module('v4_schema_fixture', REPOSITORY / 'tests/test_recovery_sequence_groups.py')
        from omtrackvla.data.recovery_sequence import validate_recovery_sample
        from omtrackvla.data.recovery_sequence_groups import VariableRecoverySequences
        from scripts.build_recovery_scene_manifest import freeze_scene_plan, build_manifest
        cls.VariableRecoverySequences = VariableRecoverySequences
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        paths = [cls.fixture.make_scene_sample(cls.root / f'{scene}_{anchor}', scene, anchor)
                 for scene in ('scene00001', 'scene00002') for anchor in (2, 5, 9, 14)]
        parent, protocol = cls.root / 'parent.ckpt', cls.root / 'protocol.json'
        parent.write_bytes(b'CPU schema fixture parent only')
        protocol.write_text('{"scope":"CPU fixture only"}')
        plan, cls.manifest = cls.root / 'plan.json', cls.root / 'manifest.json'
        validator = lambda path: validate_recovery_sample(path, artifact_root=cls.root)
        freeze_scene_plan(sample_paths=paths, output=plan, parent_checkpoint=parent, collection_protocol=protocol,
                          validator=validator, seed=41, min_train_scenes=1, min_val_scenes=1,
                          required_tasks=('STT',), val_fraction=.5)
        build_manifest(plan_path=plan, output=cls.manifest, validator=validator)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def pool(self, role='train', training=True):
        return self.VariableRecoverySequences(self.manifest, partition_role=role, for_training=training,
                                              artifact_root=self.root, image_height=3, image_width=4)

    def test_unequal_prefixes_keep_every_observation_and_only_slice_labels(self):
        from omtrackvla.training.sequence_training import sequence_model_inputs
        pool = self.pool()
        self.assertEqual({identity.sequence_length for identity in pool.identities}, {3, 6, 10, 15})
        for index in range(len(pool)):
            item = pool.optimizer_item(index)
            batch, labels, burn, meta = trainer.prepare_recovery(item, torch, 'cpu', for_optimizer=True)
            steps = item.identity.sequence_length
            with self.subTest(steps=steps):
                self.assertEqual(tuple(batch['ego_rgb'].shape[:2]), (1, steps))
                self.assertEqual(burn, max(0, steps - 4))
                self.assertEqual(labels['supervision_mask'].tolist(), [[False]*(min(4, steps)-1)+[True]])
                self.assertEqual(meta['manifest_index'], index)
                self.assertEqual(meta['anchor_policy_call_index'], steps - 1)
                self.assertEqual(meta['sequence_length'], steps)
                self.assertEqual(meta['burn_in_steps'], burn)
                self.assertEqual(meta['partition_role'], 'train')
                # Every policy call keeps its genuine four-frame history.
                for call in range(steps):
                    for history in range(4):
                        observed = max(0, call - 3 + history) / 255
                        torch.testing.assert_close(batch['ego_rgb'][0, call, history], torch.full((3, 3, 4), observed))
                inputs = sequence_model_inputs(batch)
                self.assertEqual(inputs['ego_rgb'].shape[1], steps)
                self.assertFalse(set(inputs) & set(labels))
                self.assertNotIn('stop_target', labels)
                for key in ('stop_label_valid', 'binding_label_valid', 'identity_label_valid', 'ego_label_valid'):
                    self.assertFalse(labels[key].any())

    def test_central_unroll_uses_full_history_with_matching_loss_suffix_and_fresh_state(self):
        from omtrackvla.training.sequence_training import SequenceTrainingPolicy, sequence_model_inputs, phase3_sequence_loss

        class TracePolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(.5))
                self.calls = []
            def forward(self, *, ego_rgb, hidden_state, _target_memory_override, **unused):
                value = ego_rgb[:, -1].mean(dim=(1, 2, 3))[:, None]
                self.calls.append((float(value[0, 0]), torch.is_grad_enabled(), hidden_state is None))
                hidden = torch.zeros_like(value) if hidden_state is None else hidden_state
                hidden = hidden + self.scale * (value + .1)
                return {'waypoints': hidden[:, None].expand(-1, 8, 2), 'hidden_state': hidden,
                        'target_memory_next': hidden, 'binding_logit': torch.zeros_like(hidden)}

        policy = TracePolicy()
        wrapped = SequenceTrainingPolicy(policy)
        for index in range(len(self.pool())):
            item = self.pool().optimizer_item(index)
            batch, labels, burn, meta = trainer.prepare_recovery(item, torch, 'cpu', for_optimizer=True)
            policy.calls.clear()
            policy.zero_grad()
            outputs = wrapped(**sequence_model_inputs(batch), burn_in_steps=burn)
            steps = meta['sequence_length']
            self.assertEqual(len(policy.calls), steps)
            self.assertEqual([enabled for _, enabled, _ in policy.calls], [False]*burn+[True]*min(4, steps))
            self.assertTrue(policy.calls[0][2])  # New independent sequence, no previous item state.
            self.assertFalse(any(first for _, _, first in policy.calls[1:]))
            self.assertEqual(outputs['waypoints'].shape, labels['target_waypoints'].shape)
            outputs['waypoints'].retain_grad()
            loss, _ = phase3_sequence_loss(outputs, labels, {'waypoint': 1.})
            loss.backward()
            self.assertFalse(outputs['waypoints'].grad[:, :-1].any())
            self.assertGreater(float(outputs['waypoints'].grad[:, -1, 1:].norm()), 0.)
            self.assertTrue(torch.isfinite(policy.scale.grad))
            self.assertGreater(float(policy.scale.grad.abs()), 0.)
        self.assertFalse(torch.cuda.is_initialized())

    def test_anchor9_preparation_matches_unchanged_v3_labels(self):
        v3 = load_module('unchanged_v3_alignment_reference', REPOSITORY / 'scripts/train_recovery_sequence_v3.py')
        pool = self.pool()
        item = next(pool.optimizer_item(i) for i, identity in enumerate(pool.identities) if identity.sequence_length == 10)
        batch, labels, burn, _ = trainer.prepare_recovery(item, torch, 'cpu', for_optimizer=True)
        self.assertEqual(burn, 6)
        expected = v3.recovery_labels(batch, 6)
        self.assertEqual(labels.keys(), expected.keys())
        for key in expected:
            self.assertTrue(torch.equal(labels[key], expected[key]), key)

    def test_trainer_optimizer_intent_cannot_admit_validation_or_read_only_train(self):
        for role in ('val', 'train'):
            item = self.pool(role=role, training=False)[0]
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, 'cannot be optimizer'):
                trainer.prepare_recovery(item, torch, 'cpu', for_optimizer=True)
            batch, labels, burn, meta = trainer.prepare_recovery(item, torch, 'cpu', for_optimizer=False)
            self.assertEqual(meta['partition_role'], role)
            self.assertEqual(labels['supervision_mask'].shape[1], batch['ego_rgb'].shape[1] - burn)


if __name__ == '__main__':
    unittest.main()
