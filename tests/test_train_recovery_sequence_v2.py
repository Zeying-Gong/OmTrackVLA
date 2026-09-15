"""CPU helper tests; importing the trainer never imports Torch or initializes CUDA."""
import importlib.util
import json
import math
import pickle
import random
import sys
import unittest
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIRECTORY / 'scripts'))
import train_recovery_sequence_v2 as trainer


class PilotLogicTests(unittest.TestCase):
    def test_top_level_gru_and_nested_adapter_grouping(self):
        self.assertEqual(trainer.parameter_group('gru.weight_ih_l0'), 'gru')
        self.assertEqual(trainer.parameter_group('policy.gru.weight_hh'), 'gru')
        self.assertEqual(trainer.parameter_group('da3.blocks.10.adapter.2.weight'), 'adapter')
        self.assertEqual(trainer.parameter_group('gru.adapter.0.weight'), 'adapter')
        self.assertEqual(trainer.parameter_group('fusion.weight'), 'policy')

    def test_training_sampler_balances_modes_for_nondivisible_size(self):
        rng = random.Random(102)
        for _ in range(40):
            indices = trainer.sample_clean_indices(19, rng)
            self.assertEqual([index % 4 for index in indices], [0, 1, 2, 3])
            self.assertTrue(all(0 <= index < 19 for index in indices))

    def test_sampler_rng_restores_clean_and_recovery_next_step(self):
        rng = random.Random(233)
        for _ in range(8):
            trainer.sample_clean_indices(101, rng)
            rng.randrange(4)
        state = pickle.loads(pickle.dumps(rng.getstate()))
        expected = (trainer.sample_clean_indices(101, rng), rng.randrange(4))
        resumed = random.Random(1)
        resumed.setstate(state)
        self.assertEqual(expected, (trainer.sample_clean_indices(101, resumed), resumed.randrange(4)))

    def test_fixed_validation_is_balanced_unique_and_does_not_consume_training_rng(self):
        random.seed(33)
        before = random.getstate()
        indices = trainer.fixed_validation_indices(101)
        self.assertEqual(before, random.getstate())
        self.assertEqual(indices, trainer.fixed_validation_indices(101))
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual([sum(index % 4 == mode for index in indices) for mode in range(4)], [4] * 4)
        with self.assertRaises(ValueError):
            trainer.fixed_validation_indices(15)

    def test_clean_gate_protects_individual_modes_and_heads(self):
        baseline = {'aggregate': {'ade_m': 1.}, 'per_mode': {'safe_stop': {'loss_stop': .2}, 'visual_only': {'ade_m': .5}}}
        candidate = {'aggregate': {'ade_m': .9}, 'per_mode': {'safe_stop': {'loss_stop': .22}, 'visual_only': {'ade_m': .4}}}
        gate = trainer.clean_retention_gate(candidate, baseline)
        self.assertFalse(gate['passed'])
        self.assertIn('per_mode.safe_stop.loss_stop', gate['failed_metrics'])

    def test_clean_gate_accepts_boundary_and_rejects_nan_or_schema_drift(self):
        self.assertTrue(trainer.clean_retention_gate({'ade': 1.0501}, {'ade': 1.})['passed'])
        self.assertFalse(trainer.clean_retention_gate({'ade': 1.05011}, {'ade': 1.})['passed'])
        for candidate in ({'ade': float('nan')}, {'different': 1.}):
            with self.assertRaises(ValueError):
                trainer.clean_retention_gate(candidate, {'ade': 1.})

    def test_best_requires_val_improvement_over_parent_and_best_and_clean_gate(self):
        cases = [(0.8, 1., .9, True, True), (.95, 1., .9, True, False),
                 (.8, 1., .9, False, False), (1., 1., 1., True, False)]
        for value, parent, best, gate, expected in cases:
            self.assertEqual(trainer.should_select_best(value, parent, best, {'passed': gate}), expected)
        with self.assertRaises(ValueError):
            trainer.should_select_best(float('inf'), 1., 1., {'passed': True})

    def test_clean_backward_precedes_recovery_forward(self):
        events = []

        class Loss:
            def __init__(self, name):
                self.name = name
            def __mul__(self, weight):
                events.append((self.name, 'weight', weight))
                return self
            def backward(self):
                events.append((self.name, 'backward'))

        def term(name):
            def forward():
                events.append((name, 'forward'))
                return Loss(name), {'source': name}
            return forward

        self.assertEqual(trainer.sequential_backward(term('clean'), term('recovery')),
                         ({'source': 'clean'}, {'source': 'recovery'}))
        self.assertLess(events.index(('clean', 'backward')), events.index(('recovery', 'forward')))
        self.assertIn(('recovery', 'weight', .25), events)

    def test_schedule_uses_full_64_step_budget_after_interruption(self):
        sequence = [trainer.learning_rate_multiplier(step) for step in range(65)]
        interrupted_then_resumed = [trainer.learning_rate_multiplier(step) for step in range(8)]
        interrupted_then_resumed.extend(trainer.learning_rate_multiplier(step) for step in range(8, 65))
        self.assertEqual(sequence, interrupted_then_resumed)
        self.assertGreater(sequence[8], 0.)
        self.assertEqual(sequence[64], 0.)
        self.assertEqual(sequence[7], 1.)

    def test_development_budget_and_gate_cannot_be_relaxed(self):
        config = json.loads((DIRECTORY / 'configs/recovery_sequence_pilot_v2.json').read_text())
        base = {'method': 'architecture_v1_end_to_end', 'test_locked_used': False}
        effective = trainer.effective_configuration(config, base, output_dir='/run')
        self.assertEqual(effective['resolved_maximum_steps'], 64)
        for section, key, value in [('training', 'maximum_steps', 8), ('training', 'num_workers', 1),
                                    ('validation', 'clean_relative_tolerance', .2), ('validation', 'automatic_promotion', True)]:
            changed = json.loads(json.dumps(config))
            changed[section][key] = value
            with self.assertRaises(ValueError):
                trainer.effective_configuration(changed, base, output_dir='/run')

    def test_metric_average_rejects_missing_mode_metric(self):
        self.assertEqual(trainer.average_metrics([{'ade': 1.}, {'ade': 3.}]), {'ade': 2.})
        with self.assertRaises(ValueError):
            trainer.average_metrics([{'ade': 1.}, {'loss': 3.}])

    def test_signal_at_final_step_finalizes_and_last64_can_resume_finalization(self):
        self.assertTrue(trainer.stop_at_boundary(8, 8, False))
        self.assertTrue(trainer.stop_at_boundary(63, None, True))
        self.assertFalse(trainer.stop_at_boundary(64, None, True))
        trainer.validate_restored_step(64, 64)
        with self.assertRaises(ValueError):
            trainer.validate_restored_step(65, 65)
        with self.assertRaises(ValueError):
            trainer.validate_restored_step(16, 15)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch not installed on local CPU host')
class TorchPilotTests(unittest.TestCase):
    def test_separate_backward_matches_combined_gradient(self):
        import torch
        torch.set_num_threads(1)
        parameter = torch.tensor([.3, -.2], requires_grad=True)
        clean = lambda: ((parameter - 1.).square().sum(), {})
        recovery = lambda: ((parameter + .4).square().sum(), {})
        trainer.sequential_backward(clean, recovery)
        observed = parameter.grad.clone()
        parameter.grad = None
        (clean()[0] + .25 * recovery()[0]).backward()
        self.assertTrue(torch.equal(observed, parameter.grad))
        self.assertFalse(torch.cuda.is_initialized())

    def test_burn_in_drops_only_unsupervised_prefix(self):
        import torch
        batch = {'supervision_mask': torch.zeros(1, 10, dtype=torch.bool)}
        batch['supervision_mask'][:, 9] = True
        batch['target_waypoints'] = torch.zeros(1, 10, 8, 2)
        batch['waypoint_mask'] = torch.zeros(1, 10, 8, dtype=torch.bool)
        batch['waypoint_mask'][:, 9] = True
        for key in trainer.LABEL_KEYS[3:]:
            batch[key] = torch.zeros(1, 10, dtype=torch.bool)
        labels = trainer.recovery_labels(batch, 6)
        self.assertEqual(labels['supervision_mask'].tolist(), [[False, False, False, True]])
        self.assertEqual(labels['target_waypoints'].shape, (1, 4, 8, 2))
        batch['supervision_mask'][:, 0] = True
        with self.assertRaises(ValueError):
            trainer.recovery_labels(batch, 6)


if __name__ == '__main__':
    unittest.main()
