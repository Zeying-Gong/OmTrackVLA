import contextlib
import copy
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from wa.tools import audit_student_goal as goal
from wa.wm.full_mixed_contract import summarize
from wa.wm.initial_bbox_repair import KEYS, VERSION
from wa.wm.student_eval_contract import REPAIR_SHA
from wa.wm.student_eval_finalize import superiority


class StudentGoalTests(unittest.TestCase):
    def setUp(self):
        self.manifest = dict(tasks={}, reference_steps={})
        self.teachers, rows = [], []
        for task in goal.TASKS:
            repair_keys = sorted(key for t, key in KEYS if t == task)
            keys = repair_keys + ['scene/'+str(i) for i in range(1405-len(repair_keys))]
            self.manifest['tasks'][task] = dict(episodes=[dict(key=k, shard=i % 8) for i, k in enumerate(keys)])
            for i, key in enumerate(keys):
                row = dict(task=task, key=key, success=float(i < goal.BASELINE_COUNTS[task]),
                    collision=float(i % 100 == 0), policy_init_valid=i != 1404,
                    mode='mixed', noise_mode='zero', controller='learned_yaw_guard_v1',
                    checkpoint_sha256=goal.BASELINE_SHA, checkpoint_step=goal.BASELINE_STEP,
                    semantic_protocol='mp3d_semantic_ply_v1', following_rate=.5,
                    following_step=5, total_step=10, finish=True,
                    initial_pair_evidence=dict(rgb='d'*64,
                        state=dict(timestamp=0., agents=[dict(transform=[[1., 0.]], joints=[0.])])))
                if (task, key) in KEYS:
                    row.update(initialization_repair=VERSION, initialization_repair_plan_sha256=REPAIR_SHA)
                rows.append(row)
                self.teachers.append(dict(pair=dict(task=task, key=key),
                    results=dict(lightnav=dict(success=int(i < dict(stt=1273, dt=1128, at=944)[task])))))
        self.teacher_metrics = dict(fixture='pure function, not real teacher performance')
        self.base = dict(rows=rows, summary={}, pair=dict(status='PASS', episodes=4215,
            comparison='fixture paired evidence', scope='fixture',
            teacher_artifact_hashes={r['task']+':'+r['key']:
                dict(lightnav={'/fixture/start': 'e'*64}, oracle={'/fixture/start': 'e'*64}) for r in rows}))
        self.refresh(self.base, dict(checkpoint_sha=goal.BASELINE_SHA, step=goal.BASELINE_STEP))
        self.contract = dict(checkpoint_sha='f'*64, step=59716)  # Synthetic fixture only.
        self.new = copy.deepcopy(self.base)
        for row in self.new['rows']:
            row.update(checkpoint_sha256=self.contract['checkpoint_sha'], checkpoint_step=self.contract['step'])
        self.set_counts(goal.TARGETS)

    def refresh(self, bundle, contract):
        summary = summarize(bundle['rows'], self.manifest, **contract)
        summary['limits'].append(goal.INSET_LIMIT)
        summary.update(experiment='evaluation_set_adaptation_v1', new_episodes=4215,
            reused_baseline_episodes=0, teacher_reference_sha256=goal.TEACHER_SHA,
            teacher_metrics_percent=self.teacher_metrics,
            superiority=superiority(bundle['rows'], self.teachers))
        bundle['summary'] = summary

    def set_counts(self, counts):
        for task in goal.TASKS:
            for i, row in enumerate(r for r in self.new['rows'] if r['task'] == task):
                row['success'] = float(i < counts[task])
        self.refresh(self.new, self.contract)

    def compare(self):
        return goal.compare_goal(self.new, self.base, self.manifest,
                                 self.contract, self.teachers, self.teacher_metrics)

    def test_exact_boundary_met_and_full_denominators(self):
        report = self.compare()
        self.assertEqual(report['goal_status'], 'MET')
        self.assertEqual(report['audit_status'], 'PASS')
        self.assertFalse(report['baseline_self_check'])
        self.assertEqual(report['tasks']['stt']['net_success'], 13)
        for task in goal.TASKS:
            row = report['tasks'][task]
            self.assertEqual(row['episodes'], 1405)
            self.assertEqual(row['candidate_success'], goal.TARGETS[task])
            self.assertEqual(row['candidate_collision_count'], 15)
            self.assertEqual(row['candidate_CR'], 100*15/1405)
            self.assertEqual(row['candidate_invalid_init_count'], 1)
            self.assertEqual(row['candidate_invalid_init_percent'], 100/1405)

    def test_each_task_one_below_fails_even_if_all_exceed_lightnav(self):
        for task in goal.TASKS:
            with self.subTest(task=task):
                counts = dict(goal.TARGETS); counts[task] -= 1
                self.set_counts(counts)
                report = self.compare()
                self.assertEqual(report['goal_status'], 'NOT_MET')
                self.assertEqual(report['tasks'][task]['shortfall'], 1)
                self.assertTrue(report['lightnav_comparison']['all_three_strictly_exceed'])

    def test_gain_regression_net_and_keys(self):
        stt = [r for r in self.new['rows'] if r['task'] == 'stt']
        stt[0]['success'] = 0.
        stt[1300]['success'] = 1.
        self.refresh(self.new, self.contract)
        report = self.compare()['tasks']['stt']
        self.assertEqual((report['gains'], report['regressions'], report['net_success']), (14, 1, 13))
        self.assertEqual(report['regression_keys'], [stt[0]['key']])
        self.assertIn(stt[1300]['key'], report['gain_keys'])

    def test_frozen_baseline_self_check_is_not_met(self):
        report = goal.compare_goal(self.base, self.base, self.manifest,
            dict(checkpoint_sha=goal.BASELINE_SHA, step=goal.BASELINE_STEP), self.teachers, self.teacher_metrics)
        self.assertTrue(report['baseline_self_check'])
        self.assertEqual(report['goal_status'], 'NOT_MET')
        self.assertEqual(report['tasks']['stt']['shortfall'], 13)
        self.assertTrue(all(r['gains'] == r['regressions'] == 0 for r in report['tasks'].values()))

    def test_missing_duplicate_and_foreign_task_rejected(self):
        original = self.new
        for change in ('missing', 'duplicate', 'foreign'):
            self.new = copy.deepcopy(original)
            if change == 'missing': self.new['rows'].pop()
            elif change == 'duplicate': self.new['rows'][-1] = self.new['rows'][0]
            else: self.new['rows'][0]['task'] = 'other'
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.compare()

    def test_mixed_checkpoint_and_old_model_rejected(self):
        for field, value in (('checkpoint_sha256', goal.BASELINE_SHA), ('checkpoint_step', goal.BASELINE_STEP)):
            original = self.new['rows'][0][field]
            self.new['rows'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.compare()
            self.new['rows'][0][field] = original
        self.contract['checkpoint_sha'] = goal.OLD_BASELINE_SHA
        with self.assertRaises(ValueError): self.compare()

    def test_semantic_and_seven_bbox_protocol_required(self):
        for field, value in (('semantic_protocol', 'wrong'), ('initialization_repair', None),
                             ('initialization_repair_plan_sha256', 'a'*64)):
            original = self.new['rows'][0][field]
            self.new['rows'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): self.compare()
            self.new['rows'][0][field] = original

    def test_summary_metrics_counts_identity_and_reference_cannot_be_forged(self):
        original = self.new['summary']
        for field, value in (('checkpoint_step', 1), ('new_episodes', 4214),
                ('reused_baseline_episodes', 1), ('teacher_reference_sha256', 'a'*64),
                ('superiority', {}), ('teacher_metrics_percent', {}), ('metrics_percent', {}),
                ('limits', [])):
            self.new['summary'] = dict(original, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError): self.compare()
        self.new['summary'] = original

    def test_pair_status_count_keys_and_teacher_evidence_rejected(self):
        original = self.new['pair']
        for change in ('status', 'count', 'key', 'hash'):
            self.new['pair'] = copy.deepcopy(original)
            if change == 'status': self.new['pair']['status'] = 'PARTIAL'
            elif change == 'count': self.new['pair']['episodes'] = 4214
            elif change == 'key': self.new['pair']['teacher_artifact_hashes'].pop(next(iter(self.new['pair']['teacher_artifact_hashes'])))
            else:
                key = next(iter(self.new['pair']['teacher_artifact_hashes']))
                self.new['pair']['teacher_artifact_hashes'][key]['lightnav']['/fixture/start'] = 'a'*64
            with self.subTest(change=change), self.assertRaises(ValueError): self.compare()

    def test_wrong_rgb_dynamic_state_and_nan_start_rejected(self):
        original = self.new['rows'][0]['initial_pair_evidence']
        for change in ('rgb', 'state', 'nan', 'missing'):
            evidence = copy.deepcopy(original)
            if change == 'rgb': evidence['rgb'] = 'a'*64
            elif change == 'state': evidence['state']['agents'][0]['joints'][0] = .01
            elif change == 'nan': evidence['state']['timestamp'] = float('nan')
            else: evidence = None
            self.new['rows'][0]['initial_pair_evidence'] = evidence
            with self.subTest(change=change), self.assertRaises(ValueError): self.compare()

    def test_numeric_outcome_bool_and_nonfinite_rejected(self):
        for value in (True, float('nan'), 2):
            self.new['rows'][0]['success'] = value
            with self.subTest(value=value), self.assertRaises(ValueError): self.compare()

    def test_changed_baseline_counts_rejected(self):
        self.base['rows'][0]['success'] = 0.
        self.refresh(self.base, dict(checkpoint_sha=goal.BASELINE_SHA, step=goal.BASELINE_STEP))
        with self.assertRaisesRegex(ValueError, 'baseline success counts'):
            self.compare()

    def test_baseline_sha_requires_identical_self_check(self):
        for row in self.new['rows']:
            row.update(checkpoint_sha256=goal.BASELINE_SHA, checkpoint_step=goal.BASELINE_STEP)
        self.contract = dict(checkpoint_sha=goal.BASELINE_SHA, step=goal.BASELINE_STEP)
        self.refresh(self.new, self.contract)
        with self.assertRaisesRegex(ValueError, 'self-check'):
            self.compare()

    def mocked_load(self, *, changed_pin=False, changed_source=False, changed_rows=False):
        root = Path('/fixture/base')
        source_hashes = {'/fixture/source/'+str(i): 'e'*64 for i in range(78)}
        summary = dict(self.base['summary'], partition_roots={t: '/fixture/'+t for t in goal.TASKS},
                       source_hashes={} if changed_source else source_hashes)
        files = {str(root/'summary.json'): json.dumps(summary),
            str(root/'combined_episodes.jsonl'): '\n'.join(json.dumps(r) for r in self.base['rows']),
            str(root/'student_teacher_pair_audit.json'): json.dumps(self.base['pair'])}
        def fake_sha(path):
            result = goal.BASELINE_PINS[Path(path).name]
            return 'a'*64 if changed_pin else result
        original = self.base['rows'][:-1] if changed_rows else self.base['rows']
        with patch.object(goal, 'file_sha', side_effect=fake_sha), patch.object(Path, 'read_text', autospec=True, side_effect=lambda p: files[str(p)]), patch.object(goal, 'read_partitions', return_value=(original, source_hashes)):
            return goal.load_bundle(root, self.manifest,
                dict(checkpoint_sha=goal.BASELINE_SHA, step=goal.BASELINE_STEP), pins=goal.BASELINE_PINS)

    def test_load_rechecks_frozen_pins_and_all_source_shards(self):
        loaded = self.mocked_load()
        self.assertEqual(len(loaded['hashes']), 81)
        for option in ('changed_pin', 'changed_source', 'changed_rows'):
            with self.subTest(option=option), self.assertRaises(ValueError):
                self.mocked_load(**{option: True})

    def test_cli_requires_explicit_actual_identity(self):
        base = ['--candidate', '/candidate', '--manifest', '/manifest',
            '--teacher-selections', '/teachers', '--checkpoint-sha', 'f'*64, '--output', '/output']
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            goal.arguments(base)
        args = goal.arguments(base + ['--checkpoint-step', '59716'])
        self.assertEqual(args.checkpoint_step, 59716)

    def test_source_hash_inventory_cannot_shadow_prior_evidence(self):
        original = {'/source': 'a'*64}
        self.assertEqual(goal.merge_hashes(original, original, {'/other': 'b'*64}),
                         {'/source': 'a'*64, '/other': 'b'*64})
        with self.assertRaisesRegex(ValueError, 'conflicting source'):
            goal.merge_hashes(original, {'/source': 'b'*64})
        self.assertEqual(original, {'/source': 'a'*64})

    def test_existing_output_rejected_before_reading_any_input(self):
        with patch.object(Path, 'exists', return_value=True), patch.object(goal, 'file_sha') as sha:
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                goal.audit('/candidate', '/baseline', '/manifest', '/teachers', 'f'*64, 59716, '/existing')
            sha.assert_not_called()


if __name__ == '__main__':
    unittest.main()
