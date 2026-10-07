import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wa.tools.build_hard_stt_plan import (
    Sources, derive_admissions, derive_windows, make_plan,
    summarize_exposure, sha,
)
from wa.wm.dual_teacher_selection import EXPERIMENT, select_teacher
from wa.wm.teacher_window_plan import SOURCE_KEYS, simulate_exposure


class HardSttPlanTest(unittest.TestCase):
    def fixture(self):
        rows, selections, demos, entries = [], [], [], []
        # Failed STT is eligible; successful STT and failed DT are not.
        for task, key, success in [('stt', 'scene/1', 0), ('stt', 'scene/2', 1),
                                   ('dt', 'scene/3', 0)]:
            pair = dict(task=task, key=key, takeover_step=0, seed=7,
                protocol_sha256='a'*64, initial_rgb_sha256='b'*64,
                takeover_state_sha256='c'*64)
            branches = {}
            for teacher, rate in [('lightnav', .8), ('oracle', .9)]:
                branches[teacher] = dict(pair, experiment=EXPERIMENT, teacher=teacher,
                    complete=True, replay_verified=True, transport_fallback=False,
                    artifact_root='/fixture/'+key+'/'+teacher,
                    result=dict(success=1, collision=0, policy_init_valid=True,
                                following_rate=rate))
            selected = select_teacher(branches['lightnav'], branches['oracle'])
            selections.append(dict(selected, branches=branches))
            demos.append(dict(task=task, key=key, teacher='oracle', takeover_step=0,
                              branch=branches['oracle']['artifact_root']))
            entries.append(dict(task=task, episode_uid=task+':'+key, teacher='oracle',
                                takeover_step=0, root=branches['oracle']['artifact_root']))
            rows.append(dict(task=task, key=key, success=success, status='Normal'))
        return rows, selections, demos, entries

    def test_derives_hard_identity_without_caller_whitelist(self):
        records, _ = derive_admissions(*self.fixture())
        self.assertEqual([r['hard'] for r in records], [True, False, False])
        self.assertEqual([r['teacher'] for r in records], ['oracle']*3)

    def test_rejects_duplicate_missing_or_foreign_sources(self):
        for field in range(4):
            fixture = list(self.fixture())
            fixture[field].append(copy.deepcopy(fixture[field][0]))
            with self.subTest(field=field), self.assertRaises(ValueError):
                derive_admissions(*fixture)
        fixture = list(self.fixture()); fixture[2].pop()
        with self.assertRaises(ValueError): derive_admissions(*fixture)
        fixture = list(self.fixture()); fixture[3][0]['root'] = '/wrong'
        with self.assertRaises(ValueError): derive_admissions(*fixture)
        fixture = list(self.fixture()); fixture[3][0]['teacher'] = 'lightnav'
        with self.assertRaises(ValueError): derive_admissions(*fixture)

    def test_rejects_stale_selection_and_failed_demo(self):
        fixture = list(self.fixture())
        fixture[1][0]['selected_teacher'] = 'lightnav'
        with self.assertRaises(ValueError): derive_admissions(*fixture)
        fixture = list(self.fixture())
        fixture[1][0]['branches']['oracle']['result']['success'] = 0
        with self.assertRaises(ValueError): derive_admissions(*fixture)

    def test_fallback_selected_branch_cannot_enter_release(self):
        fixture = list(self.fixture())
        branches = fixture[1][0]['branches']
        branches['oracle']['result']['success'] = 0
        ln = branches['lightnav']
        ln.update(transport_fallback=True, fallback_policy='released_lightnav_client_v1',
                  fallback_events=[dict(step=1, action=[0., 0., 0.])])
        fixture[1][0] = dict(select_teacher(ln, branches['oracle']), branches=branches)
        with self.assertRaises(ValueError): derive_admissions(*fixture)

    def test_valid_intersection_and_actual_timestamp_early_boundary(self):
        records, _ = derive_admissions(*self.fixture())
        history = np.array([[0, 1], [0, 2], [0, 3], [0, 1], [0, 2], [0, 1]])
        episode = np.array([0, 0, 0, 1, 1, 2])
        valid = np.array([0, 2, 3, 5])
        times = {ep: [5., 5.2, 6., 7.01] for ep in range(3)}
        extras, windows = derive_windows(history, episode, valid, records, times)
        self.assertEqual(extras, [0, 1])
        self.assertEqual(windows['raw_row'].tolist(), [0, 2, 3, 5])
        self.assertEqual(windows['early'].tolist(), [True, False, True, True])
        self.assertEqual(records[0]['early_windows'], 1)
        self.assertAlmostEqual(windows['age_s'][1], 2.01)
        self.assertEqual(windows['hard'].tolist(), [True, True, False, False])

    def test_zero_window_episode_is_not_eligible(self):
        records, _ = derive_admissions(*self.fixture())
        extras, windows = derive_windows(np.array([[0, 1], [0, 1]]),
            np.array([1, 2]), np.array([0, 1]), records,
            {ep: [0., .2] for ep in range(3)})
        self.assertEqual(extras, [])
        self.assertEqual(records[0]['valid_windows'], 0)

    def test_plan_schema_bounded_budget_and_exact_simulation(self):
        hashes = {k: 'a'*64 for k in SOURCE_KEYS}
        plan, mix = make_plan(101, 17, [0, 4], hashes, 2)
        self.assertEqual(len(mix), 122)
        report, counts = simulate_exposure(mix)
        self.assertEqual(report['actual_total'], 112)
        self.assertLessEqual(int(counts.max()), 3)
        for repeats in (0, 3, True):
            with self.assertRaises(ValueError): make_plan(101, 17, [0], hashes, repeats)
        with self.assertRaises(ValueError): make_plan(101, 17, [], hashes, 2)

    def test_metadata_and_exposure_groups_are_external(self):
        records, _ = derive_admissions(*self.fixture())
        extras, windows = derive_windows(np.array([[0, 1]]*3), np.array([0, 1, 2]),
            np.array([0, 1, 2]), records, {ep: [0., .2] for ep in range(3)})
        summary = summarize_exposure(records, windows, np.array([3, 1, 1]))
        self.assertEqual(summary['groups']['hard_stt'], 3)
        self.assertEqual(summary['groups']['hard_stt_early'], 3)
        self.assertEqual(summary['groups']['stt:oracle'], 4)
        plan, mix = make_plan(1, 3, extras, {k: 'a'*64 for k in SOURCE_KEYS}, 2)
        self.assertIsInstance(mix[1], int)  # Metadata has not become a model sample.
        self.assertNotIn('task', plan)

    def test_hash_pin_and_during_build_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'evidence.json'
            path.write_text(json.dumps(dict(status='PASS')))
            sources = Sources()
            sources.read(path, sha(path))
            with self.assertRaises(ValueError): sources.verify(path, '0'*64)
            path.write_text('{}')
            with self.assertRaises(ValueError): sources.recheck()


if __name__ == '__main__':
    unittest.main()
