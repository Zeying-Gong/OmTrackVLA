import copy
import unittest
from collections import Counter
import torch
from torch.utils.data import DataLoader, DistributedSampler
from wa.wm.teacher_window_plan import (SCHEMA, SOURCE_KEYS, SAMPLE_INDEX,
    PlannedTeacherMix, ExposureTaggedData, strip_exposure_tag, simulate_exposure)
from wa.wm.recovery_mix import RecoveryMix

class TeacherWindowPlanTest(unittest.TestCase):
    def make(self, base=101, teacher=17, extras=(0, 3, 12), repeats=2):
        hashes = {k: 'a' * 64 for k in SOURCE_KEYS}
        plan = dict(schema=SCHEMA, base_count=base, teacher_count=teacher,
            extra_repeats=repeats, extra_teacher_indices=list(extras), source_hashes=hashes)
        return plan, hashes

    def mix(self, plan=None, **kwargs):
        default, hashes = self.make(**kwargs)
        plan = default if plan is None else plan
        return PlannedTeacherMix(list(range(default['base_count'])),
            list(range(default['base_count'], default['base_count'] + default['teacher_count'])),
            plan, source_hashes=hashes, eligible_teacher_indices=default['extra_teacher_indices'])

    def test_only_verified_windows_gain_bounded_exposure(self):
        mix = self.mix()
        counts = Counter(mix[i] for i in range(len(mix)))
        self.assertEqual(len(mix), 124)
        for i in range(118):
            self.assertEqual(counts[i], 3 if i in (101, 104, 113) else 1)
        for invalid in (-1, len(mix), True):
            with self.assertRaises(IndexError): mix[invalid]

    def test_zero_extra_is_identical_to_old_repeat_one(self):
        mix = self.mix(extras=(), repeats=0)
        old = RecoveryMix(mix.base, mix.teacher, 1)
        self.assertEqual([mix[i] for i in range(len(mix))], [old[i] for i in range(len(old))])
        for rank in range(8):
            args = dict(num_replicas=8, rank=rank, shuffle=True, seed=42, drop_last=True)
            one, two = DistributedSampler(mix, **args), DistributedSampler(old, **args)
            one.set_epoch(1); two.set_epoch(1)
            self.assertEqual(list(one), list(two))

    def test_invalid_or_foreign_plan_rejected(self):
        plan, _ = self.make()
        mutations = [('extra_repeats', x) for x in (-1, 3, True, 1.5)]
        mutations += [('extra_teacher_indices', x) for x in ([0, 0], [3, 0], [-1], [17], [1], [True], [])]
        mutations += [('base_count', x) for x in (True, 100)]
        mutations += [('schema', 'foreign'), ('source_hashes', {})]
        for key, value in mutations:
            bad = copy.deepcopy(plan); bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): self.mix(bad)
        bad = copy.deepcopy(plan); bad['source_hashes']['student_rows'] = 'b' * 64
        with self.assertRaises(ValueError): self.mix(bad)
        bad = copy.deepcopy(plan); bad['unknown'] = 1
        with self.assertRaises(ValueError): self.mix(bad)

    def test_actual_dataloader_counts_equal_simulation(self):
        for sizes in ((101, 17), (103, 19), (117, 17)):
            mix = self.mix(base=sizes[0], teacher=sizes[1])
            report, counts = simulate_exposure(mix)
            actual = Counter()
            for rank in range(8):
                sampler = DistributedSampler(mix, num_replicas=8, rank=rank, shuffle=True, seed=42, drop_last=True)
                sampler.set_epoch(1)
                for batch in DataLoader(mix, sampler=sampler, batch_size=2, drop_last=True):
                    actual.update(batch.tolist())
            self.assertEqual(sum(actual.values()), report['actual_total'])
            self.assertEqual([actual[i + len(mix.base)] for i in range(len(mix.teacher))], counts.tolist())
            again, again_counts = simulate_exposure(mix)
            self.assertEqual(report, again)
            self.assertEqual(counts.tolist(), again_counts.tolist())

    def test_metadata_stripped_without_changing_model_batch(self):
        raw = [{'pose': torch.ones(2, 4), 'rgb': torch.zeros(3, 2, 2)} for _ in range(3)]
        tagged = ExposureTaggedData(raw)
        batch = next(iter(DataLoader(tagged, batch_size=2)))
        cleaned, indices = strip_exposure_tag(batch)
        expected = next(iter(DataLoader(raw, batch_size=2)))
        self.assertEqual(set(cleaned), set(expected))
        self.assertEqual(indices.tolist(), [0, 1])
        for key in expected: self.assertTrue(torch.equal(cleaned[key], expected[key]))
        self.assertNotIn(SAMPLE_INDEX, raw[0])
        self.assertIn(SAMPLE_INDEX, batch)
        with self.assertRaises(ValueError): strip_exposure_tag(expected)
        with self.assertRaises(ValueError): ExposureTaggedData([{SAMPLE_INDEX: 0}])[0]

    def test_empty_sources_and_invalid_simulation_rejected(self):
        plan, hashes = self.make()
        with self.assertRaises(ValueError):
            PlannedTeacherMix([], [1], plan, source_hashes=hashes, eligible_teacher_indices=[])
        for world, batch in ((0, 2), (8, 0), (True, 2)):
            with self.assertRaises(ValueError): simulate_exposure(self.mix(), world=world, batch=batch)

if __name__ == '__main__':
    unittest.main()
