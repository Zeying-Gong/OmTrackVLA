import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wa.tools import audit_hard_stt_outcomes as audit
from wa.tools import audit_student_goal as goal
from wa.wm.dual_teacher_selection import EXPERIMENT, select_teacher
from wa.tests import test_audit_student_goal as goal_fixtures


class HardOutcomesTests(unittest.TestCase):
    def setUp(self):
        self.fixture = goal_fixtures.StudentGoalTests()
        self.fixture.setUp()
        self.old = self.fixture.base['rows']
        self.new = self.fixture.new['rows']
        self.teachers, self.plan = [], []
        per_task = {task: 0 for task in goal.TASKS}
        for old, new in zip(self.old, self.new):
            task, key = old['task'], old['key']
            index = per_task[task]; per_task[task] += 1
            old['status'] = 'Normal' if old['success'] else 'Lost'
            new['status'] = 'Normal' if new['success'] else 'Lost'
            hard = task == 'stt' and 1276 <= index < 1369
            pair = dict(task=task, key=key, takeover_step=0, seed=7,
                protocol_sha256='a'*64, initial_rgb_sha256='d'*64, takeover_state_sha256='b'*64)
            branches = {}
            for name in ('lightnav', 'oracle'):
                success = bool(old['success']) if name == 'lightnav' else hard
                branches[name] = dict(experiment=EXPERIMENT, teacher=name, complete=True,
                    replay_verified=True, artifact_root='/fixture/'+task+'/'+key+'/'+name,
                    result=dict(success=success, collision=False, policy_init_valid=True,
                                following_rate=.9), **pair)
            choice = select_teacher(**branches)
            self.teachers.append(dict(choice, branches=branches))
            if choice['demonstration_candidate']:
                teacher = choice['selected_teacher']
                valid = (0 if index >= 1367 else 1) if hard else 3
                if hard and index == 1276: valid = 10413 - 90
                self.plan.append(dict(task=task, key=key, episode_uid=task+':'+key,
                    teacher=teacher, branch=branches[teacher]['artifact_root'], hard=hard,
                    student_success=bool(old['success']), student_status=old['status'],
                    selected_teacher_success=True, selected_teacher_fallback=False,
                    valid_windows=valid, early_windows=min(valid, 1)))
        self.fixture.teachers = self.teachers
        self.fixture.refresh(self.fixture.base, dict(checkpoint_sha=goal.BASELINE_SHA, step=goal.BASELINE_STEP))
        self.fixture.refresh(self.fixture.new, self.fixture.contract)

    def compare(self, new=None):
        bundle = self.fixture.new if new is None else new
        checked = goal.compare_goal(bundle, self.fixture.base, self.fixture.manifest,
            self.fixture.contract, self.teachers, self.fixture.teacher_metrics)
        return audit.classify(bundle['rows'], self.old, self.teachers, self.plan, checked)

    def test_original91_and_two_zero_window_cases_are_disjoint(self):
        groups = self.compare()
        self.assertEqual(groups['original91_recovered']['count'], 13)
        self.assertEqual(groups['original91_still_failed']['count'], 78)
        self.assertEqual(groups['original2_zero_window_still_failed']['count'], 2)
        zero = groups['original2_zero_window_still_failed']['records']
        self.assertTrue(all(r['demonstration_class'] == 'admitted_zero_windows' for r in zero))
        self.assertTrue(all(not r['original_hard_nonzero'] for r in zero))
        self.assertEqual(groups['teacher_solvable_still_failed']['by_task']['stt']['count'], 80)

    def test_gains_regressions_reconcile_and_report_scene_status(self):
        stt = [r for r in self.new if r['task'] == 'stt']
        stt[0].update(success=False, status='Collision', collision=True)
        stt[1300].update(success=True, status='Normal')
        self.fixture.refresh(self.fixture.new, self.fixture.contract)
        groups = self.compare()
        self.assertEqual(groups['gains']['by_task']['stt']['count'], 14)
        self.assertEqual(groups['regressions']['by_task']['stt']['count'], 1)
        self.assertEqual(groups['regressions']['by_task']['stt']['by_status'], {'Collision': 1})
        self.assertEqual(groups['regressions']['records'][0]['key'], stt[0]['key'])

    def test_missing_duplicate_and_rgb_mismatch_fail_closed(self):
        for change in ('missing', 'duplicate', 'rgb'):
            new = copy.deepcopy(self.fixture.new)
            if change == 'missing': new['rows'].pop()
            elif change == 'duplicate': new['rows'][-1] = new['rows'][0]
            else: new['rows'][0]['initial_pair_evidence']['rgb'] = 'e'*64
            with self.subTest(change=change), self.assertRaises(ValueError): self.compare(new)

    def test_duplicate_plan_and_changed_hard_qualification_rejected(self):
        original = self.plan
        for change in ('duplicate', 'hard', 'zero', 'teacher', 'missing'):
            self.plan = copy.deepcopy(original)
            hard = next(r for r in self.plan if r['hard'])
            if change == 'duplicate': self.plan.append(self.plan[0])
            elif change == 'hard': hard['hard'] = False
            elif change == 'zero': hard['valid_windows'] = 0
            elif change == 'teacher': hard['teacher'] = 'lightnav'
            else: self.plan.pop()
            with self.subTest(change=change), self.assertRaises(ValueError): self.compare()
        self.plan = original

    def test_nonfixed_plan_hash_refused_without_loading_json(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'episodes.json'; path.write_text('not JSON')
            with self.assertRaisesRegex(ValueError, 'frozen 61377'): audit.load_plan(path)

    def test_teacher_selection_not_trusted(self):
        self.teachers[0]['selected_teacher'] = 'oracle'
        with self.assertRaises(ValueError): self.compare()

    def test_selected_fallback_success_not_counted_as_admitted(self):
        row = next(r for r in self.teachers if r['pair']['task'] == 'dt')
        row['branches']['lightnav'].update(transport_fallback=True,
            fallback_policy='released_lightnav_client_v1', fallback_events=[dict(step=0, action=[0, 0, 0])])
        choice = select_teacher(**row['branches']); row.update(choice)
        key = row['pair']['key']
        self.plan = [r for r in self.plan if (r['task'], r['key']) != ('dt', key)]
        new = next(r for r in self.new if (r['task'], r['key']) == ('dt', key))
        new.update(success=False, status='Lost')
        self.fixture.refresh(self.fixture.new, self.fixture.contract)
        groups = self.compare()
        record = next(r for r in groups['teacher_solvable_still_failed']['records'] if r['task'] == 'dt' and r['key'] == key)
        self.assertEqual(record['demonstration_class'], 'successful_teacher_not_admitted_selected_fallback')
        self.assertEqual(record['valid_windows'], 0)

    def test_not_met_valid_selfcheck_and_no_recovery(self):
        checked = goal.compare_goal(self.fixture.base, self.fixture.base, self.fixture.manifest,
            dict(checkpoint_sha=goal.BASELINE_SHA, step=goal.BASELINE_STEP), self.teachers, self.fixture.teacher_metrics)
        groups = audit.classify(self.old, self.old, self.teachers, self.plan, checked)
        self.assertEqual(checked['goal_status'], 'NOT_MET')
        self.assertEqual(groups['original91_recovered']['count'], 0)
        self.assertEqual(groups['original91_still_failed']['count'], 91)

    def test_incomplete_bundle_prevents_grouping(self):
        with patch.object(goal, 'file_sha', return_value=goal.MANIFEST_SHA), \
             patch.object(Path, 'read_text', return_value='{}'), \
             patch.object(audit, 'load_plan', return_value=self.plan), \
             patch.object(audit, 'load_teachers', return_value=self.teachers), \
             patch.object(audit, 'summarize_teachers', return_value={'metrics_percent': {}}), \
             patch.object(goal, 'load_bundle', side_effect=FileNotFoundError('PARTITION_COMPLETE.json')), \
             patch.object(audit, 'classify') as classify:
            with self.assertRaises(FileNotFoundError):
                audit.diagnose('/candidate', '/baseline', '/manifest', '/teachers', 'f'*64, 59716, '/plan')
            classify.assert_not_called()

    def test_output_no_overwrite_or_write_into_inputs(self):
        with tempfile.TemporaryDirectory() as root:
            protected = Path(root)/'input'; protected.mkdir()
            with self.assertRaises(ValueError): audit.write_report({}, protected/'new', [protected])
            output = Path(root)/'audit'
            artifact = audit.write_report({'goal_status': 'NOT_MET'}, output, [protected])
            self.assertEqual(json.loads(Path(artifact['path']).read_text())['goal_status'], 'NOT_MET')
            with self.assertRaises(ValueError): audit.write_report({}, output, [protected])

    def test_dangling_output_symlink_rejected_without_target_creation(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root)/'absent-target'
            output = Path(root)/'audit-link'
            output.symlink_to(target, target_is_directory=True)
            with self.assertRaises(ValueError): audit.write_report({}, output, [])
            self.assertTrue(output.is_symlink())
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
