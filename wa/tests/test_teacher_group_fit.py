import unittest
import numpy as np
import torch
from wa.tools.audit_teacher_group_fit import choose_early_index, fit_metrics, select_episodes

class GroupFitTest(unittest.TestCase):
    def test_actual_early_nearest_one(self):
        self.assertEqual(choose_early_index([.2, .9, 1.6, 2.01]), 1)
        self.assertEqual(choose_early_index([2.0]), 0)
        self.assertIsNone(choose_early_index([2.001, 4]))
        self.assertIsNone(choose_early_index([]))
        for bad in ([float('nan')], [-.01], [[1]]):
            with self.assertRaises(ValueError): choose_early_index(bad)
    def test_wrapped_yaw_and_xy(self):
        p = torch.zeros(2, 7, 4); y = p.clone()
        p[..., 3] = -1; y[..., 3] = -1
        p[..., 2] = .0001; y[..., 2] = -.0001
        p[..., 0] = 3; p[..., 1] = 4
        got = fit_metrics(p, y)
        torch.testing.assert_close(got['ADE_m'], torch.tensor([5., 5.]))
        self.assertTrue((got['yaw_MAE_rad'] < .001).all())
        p[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError): fit_metrics(p, y)
    def test_stable_disjoint_groups_no_early_excluded(self):
        entries = [dict(task=t, episode_uid=t+':a/'+str(i))
                   for t in ('stt', 'dt', 'at') for i in range(3)]
        hard = {'a/0': dict(status='Collision', early_valid=1),
                'a/1': dict(status='Lost', early_valid=0)}
        rs = select_episodes(entries, hard)
        mapping = {e['episode_uid']: g for g, _, e in rs}
        self.assertEqual(mapping['stt:a/0'], 'hard_stt_collision')
        self.assertNotIn('stt:a/1', mapping)
        self.assertEqual(mapping['stt:a/2'], 'successful_stt_control')
        self.assertEqual(len(rs), len(mapping))
        self.assertEqual([(g,e['episode_uid']) for g,_,e in rs],
                         [(g,e['episode_uid']) for g,_,e in select_episodes(entries, hard)])

if __name__ == '__main__': unittest.main()
