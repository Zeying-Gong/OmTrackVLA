import copy
import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from wa.wm.teacher_window_plan import (SCHEMA, SOURCE_KEYS, PlannedTeacherMix,
    ExposureTaggedData, strip_exposure_tag)
from wa.wm.runtime_exposure import RuntimeExposure

class RuntimeExposureTest(unittest.TestCase):
    def fixture(self, base=103, teacher=19, extras=(0, 3, 8), repeats=2):
        hashes = {k: 'a'*64 for k in SOURCE_KEYS}
        plan = dict(schema=SCHEMA, base_count=base, teacher_count=teacher,
            extra_repeats=repeats, extra_teacher_indices=list(extras), source_hashes=hashes)
        data = lambda n: [{'rgb': torch.ones(1), 'pose': torch.zeros(1)} for _ in range(n)]
        mix = PlannedTeacherMix(data(base), data(teacher), plan,
            source_hashes=hashes, eligible_teacher_indices=list(extras))
        meta = dict(episode_index=np.arange(teacher)%3, episodes=[
            dict(episode_uid='stt:a/1', task='stt', teacher='lightnav'),
            dict(episode_uid='dt:b/2', task='dt', teacher='oracle'),
            dict(episode_uid='at:c/3', task='at', teacher='lightnav')],
            hard=np.arange(teacher)%3==0, early=np.arange(teacher)%2==0)
        return mix, meta

    def test_real_eight_rank_loader_and_complete_simulation_match(self):
        for base in (101, 103, 117):
            mix, meta = self.fixture(base=base)
            counters = []
            for rank in range(8):
                counter = RuntimeExposure(mix, **meta)
                sampler = DistributedSampler(mix, num_replicas=8, rank=rank,
                    shuffle=True, seed=42, drop_last=True)
                sampler.set_epoch(1)
                for batch in DataLoader(ExposureTaggedData(mix), sampler=sampler, batch_size=2, drop_last=True):
                    cleaned, ids = strip_exposure_tag(batch)
                    self.assertEqual(set(cleaned), {'rgb', 'pose'})
                    counter.consume(ids)
                counters.append(counter)
            combined = counters[0].merge_states([x.state() for x in counters])
            report = counters[0].finalize(state=combined)
            base_counts, teacher_counts = counters[0].source_counts(combined)
            self.assertEqual(report['status'], 'ACTUAL_EXPOSURE_MATCHES_SIMULATION')
            self.assertEqual(sum(report['groups'].values()), int(teacher_counts.sum()))
            self.assertEqual(sum(x['exposures'] for x in report['per_episode']), int(teacher_counts.sum()))
            self.assertEqual(report['hard_teacher_exposures'], int(teacher_counts[meta['hard']].sum()))
            self.assertEqual(report['early_teacher_exposures'], int(teacher_counts[meta['early']].sum()))
            self.assertEqual(report['hard_early_teacher_exposures'], int(teacher_counts[meta['hard'] & meta['early']].sum()))
            self.assertEqual(report['actual_base'], int(base_counts.sum()))
            self.assertEqual(report['dropped'], len(mix)-report['actual_total'])

    def test_distinct_mix_positions_can_repeat_same_teacher_source(self):
        mix, meta = self.fixture()
        c = RuntimeExposure(mix, **meta)
        b, t = len(mix.base), len(mix.teacher)
        c.consume(torch.tensor([b, b+t, b+t+len(mix.extra)]))
        self.assertEqual(c.source_counts()[1][0].item(), 3)
        self.assertNotIn('dropped', c.summarize())

    def test_bad_batches_fail_without_partial_mutation(self):
        mix, meta = self.fixture(); c = RuntimeExposure(mix, **meta)
        for ids in (torch.tensor([-1]), torch.tensor([len(mix)]), torch.tensor([0, 0]),
                    torch.tensor([1.]), torch.tensor([True]), torch.tensor([[1]]),
                    torch.tensor([], dtype=torch.long), [1, 2]):
            with self.subTest(ids=ids), self.assertRaises(ValueError): c.consume(ids)
            self.assertEqual(c.summarize()['actual_total'], 0)
        c.consume(torch.tensor([1, 2]))
        with self.assertRaises(ValueError): c.consume(torch.tensor([3, 1]))
        self.assertEqual(c.summarize()['actual_total'], 2)
        state = c.state(); state['position_counts'].zero_()
        self.assertEqual(c.summarize()['actual_total'], 2)

    def test_cross_rank_overlap_and_foreign_context_rejected(self):
        mix, meta = self.fixture(); c = RuntimeExposure(mix, **meta)
        c.consume(torch.tensor([1, 2])); state = c.state()
        with self.assertRaises(ValueError): c.merge_states([state, state])
        other_meta = copy.deepcopy(meta); other_meta['early'][0] = False
        other = RuntimeExposure(mix, **other_meta)
        with self.assertRaises(ValueError): c.merge_states([state, other.state()])
        with self.assertRaises(ValueError): c.validate_contexts([c.context_sha256, other.context_sha256])
        c.validate_contexts([c.context_sha256]*8)
        for tensor in (torch.zeros(len(mix), dtype=torch.float32), torch.full((len(mix),), -1),
                       torch.zeros(len(mix)-1, dtype=torch.long)):
            invalid = dict(state, position_counts=tensor)
            with self.assertRaises(ValueError): c.summarize(invalid)

    def test_incomplete_or_wrong_epoch_never_labeled_complete(self):
        mix, meta = self.fixture(); c = RuntimeExposure(mix, **meta)
        with self.assertRaises(ValueError): c.finalize()
        c.consume(torch.arange(len(mix)))
        with self.assertRaises(ValueError): c.finalize(world=8, batch=3)
        for seed, epoch in ((True, 1), (42, -1), (42, 1.5)):
            with self.assertRaises(ValueError): c.finalize(seed=seed, epoch=epoch)

    def test_metadata_bounds_types_and_identity_fail_closed(self):
        mix, meta = self.fixture()
        changes = [dict(episode_index=np.full(19, -1)), dict(episode_index=np.full(19, 3)),
                   dict(episode_index=np.arange(19, dtype=float)), dict(hard=np.ones(19)),
                   dict(hard=np.ones(19, dtype=bool)), dict(early=np.ones(18, dtype=bool))]
        for change in changes:
            with self.assertRaises(ValueError): RuntimeExposure(mix, **dict(meta, **change))
        invalid = copy.deepcopy(meta); invalid['episodes'][1]['episode_uid'] = 'stt:a/1'
        with self.assertRaises(ValueError): RuntimeExposure(mix, **invalid)
        c = RuntimeExposure(mix, **meta); meta['hard'].fill(False)
        self.assertTrue(c.hard.any())

if __name__ == '__main__': unittest.main()
