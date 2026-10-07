import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wa.tools import compare_student61609 as comparison
from wa.tools import audit_student_goal as goal
from wa.tests import test_audit_student_goal as fixtures


class Student61609ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.StudentGoalTests()
        self.fixture.setUp()
        self.base_contract = dict(checkpoint_sha=comparison.BASELINE_SHA,
                                 step=comparison.BASELINE_STEP)
        for task in goal.TASKS:
            for index, row in enumerate(r for r in self.fixture.base['rows'] if r['task'] == task):
                row.update(success=bool(index < comparison.BASELINE_COUNTS[task]),
                    checkpoint_sha256=comparison.BASELINE_SHA,
                    checkpoint_step=comparison.BASELINE_STEP)
        self.fixture.refresh(self.fixture.base, self.base_contract)

    def compare(self):
        f = self.fixture
        return comparison.compare(f.new, f.base, f.manifest, f.contract,
                                  f.teachers, f.teacher_metrics)

    def test_complete_counts_and_no_redefined_goal(self):
        report = self.compare()
        self.assertEqual(report['comparison_baseline'], 61609)
        self.assertEqual(report['episodes'], 4215)
        self.assertNotIn('goal_status', report)
        self.assertEqual(goal.BASELINE_COUNTS, dict(stt=1276, dt=1173, at=1203))
        self.assertEqual(goal.TARGETS, dict(stt=1289, dt=1173, at=1203))
        self.assertEqual({t: r['net_success'] for t, r in report['tasks'].items()},
                         dict(stt=10, dt=-5, at=-4))
        self.assertEqual(report['tasks']['stt']['candidate_invalid_init_count'], 1)
        self.assertEqual(report['tasks']['stt']['candidate_SR'], 100*1289/1405)

    def test_recovery_regression_and_common_groups_reconcile(self):
        f = self.fixture
        rows = [r for r in f.new['rows'] if r['task'] == 'stt']
        rows[0]['success'] = False
        rows[1300]['success'] = True
        f.refresh(f.new, f.contract)
        row = self.compare()['tasks']['stt']
        self.assertEqual((row['gains'], row['regressions'], row['net_success']), (11, 1, 10))
        self.assertEqual(row['regression_keys'], [rows[0]['key']])
        self.assertIn(rows[1300]['key'], row['gain_keys'])
        self.assertEqual(row['common_success']+row['common_failure']+row['gains']+row['regressions'], 1405)

    def test_identical_baseline_self_check(self):
        f = self.fixture
        report = comparison.compare(f.base, f.base, f.manifest,
            self.base_contract, f.teachers, f.teacher_metrics)
        self.assertTrue(report['baseline_self_check'])
        self.assertTrue(all(r['gains'] == r['regressions'] == 0 for r in report['tasks'].values()))

    def test_same_sha_foreign_step_or_different_rows_rejected(self):
        f = self.fixture
        for step in (comparison.BASELINE_STEP, comparison.BASELINE_STEP+1):
            contract = dict(checkpoint_sha=comparison.BASELINE_SHA, step=step)
            for row in f.new['rows']:
                row.update(checkpoint_sha256=contract['checkpoint_sha'], checkpoint_step=step)
            f.refresh(f.new, contract)
            with self.subTest(step=step), self.assertRaises(ValueError):
                comparison.compare(f.new, f.base, f.manifest, contract, f.teachers, f.teacher_metrics)

    def test_missing_duplicate_foreign_and_mixed_model_rejected(self):
        f = self.fixture
        original = f.new
        for mutation in ('missing', 'duplicate', 'foreign', 'checkpoint', 'step', 'mode'):
            f.new = copy.deepcopy(original)
            if mutation == 'missing': f.new['rows'].pop()
            elif mutation == 'duplicate': f.new['rows'][-1] = f.new['rows'][0]
            elif mutation == 'foreign': f.new['rows'][0]['task'] = 'other'
            elif mutation == 'checkpoint': f.new['rows'][0]['checkpoint_sha256'] = comparison.BASELINE_SHA
            elif mutation == 'step': f.new['rows'][0]['checkpoint_step'] = 0
            else: f.new['rows'][0]['mode'] = 'image'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.compare()

    def test_rgb_state_nonfinite_and_pair_hash_rejected(self):
        f = self.fixture
        original = f.new
        for mutation in ('rgb', 'state', 'nan', 'pair'):
            f.new = copy.deepcopy(original)
            evidence = f.new['rows'][0]['initial_pair_evidence']
            if mutation == 'rgb': evidence['rgb'] = 'a'*64
            elif mutation == 'state': evidence['state']['agents'][0]['joints'][0] = 1.
            elif mutation == 'nan': evidence['state']['timestamp'] = float('nan')
            else: f.new['pair']['teacher_artifact_hashes'].pop(next(iter(f.new['pair']['teacher_artifact_hashes'])))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.compare()

    def test_baseline_counts_cannot_be_relabelled(self):
        f = self.fixture
        f.base['rows'][0]['success'] = False
        f.refresh(f.base, self.base_contract)
        with self.assertRaisesRegex(ValueError, '61609 success counts'): self.compare()

    def test_semantic_bbox_summary_and_teacher_claims_fail_closed(self):
        f = self.fixture
        original = f.new
        for mutation in ('semantic', 'bbox', 'summary', 'reuse', 'teachers'):
            f.new = copy.deepcopy(original)
            if mutation == 'semantic': f.new['rows'][0]['semantic_protocol'] = 'old'
            elif mutation == 'bbox': f.new['rows'][0]['initialization_repair_plan_sha256'] = 'a'*64
            elif mutation == 'summary': f.new['summary']['metrics_percent'] = {}
            elif mutation == 'reuse': f.new['summary']['reused_baseline_episodes'] = 1
            else: f.new['summary']['teacher_reference_sha256'] = 'a'*64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.compare()

    def test_initialization_failure_is_kept(self):
        f = self.fixture
        f.new['rows'][0].update(success=False, policy_init_valid=False, following_rate=0.,
                              policy_failure_reason='invalid bbox')
        f.refresh(f.new, f.contract)
        row = self.compare()['tasks']['stt']
        self.assertEqual(row['episodes'], 1405)
        self.assertEqual(row['candidate_invalid_init_count'], 2)
        self.assertEqual(row['candidate_success'], 1288)

    def test_61609_pins_are_passed_and_failed_load_never_compares(self):
        with patch.object(goal, 'file_sha', return_value=goal.MANIFEST_SHA), \
             patch.object(Path, 'read_text', return_value='{}'), \
             patch.object(comparison, 'load_teachers', return_value=[]), \
             patch.object(comparison, 'summarize_teachers', return_value={'metrics_percent': {}}), \
             patch.object(goal, 'load_bundle', side_effect=ValueError('pin mismatch')) as loader, \
             patch.object(comparison, 'compare') as compare:
            with self.assertRaisesRegex(ValueError, 'pin mismatch'):
                comparison.diagnose('/candidate', '/baseline', '/manifest', '/teachers', 'f'*64, 59716)
            self.assertEqual(loader.call_args.kwargs['pins'], comparison.BASELINE_PINS)
            self.assertEqual(loader.call_args.args[2], self.base_contract)
            compare.assert_not_called()

    def test_source_mutation_during_diagnosis_rejected(self):
        f = self.fixture
        base = dict(f.base, root='/baseline', hashes={'/immutable/source': 'a'*64})
        new = dict(f.new, root='/candidate', hashes={'/immutable/source': 'a'*64})
        def fake_sha(path):
            if str(path) == '/manifest': return goal.MANIFEST_SHA
            if str(path) == '/teachers': return goal.TEACHER_SHA
            return 'b'*64
        with patch.object(goal, 'file_sha', side_effect=fake_sha), \
             patch.object(Path, 'read_text', return_value=json.dumps(f.manifest)), \
             patch.object(comparison, 'load_teachers', return_value=f.teachers), \
             patch.object(comparison, 'summarize_teachers', return_value={'metrics_percent': f.teacher_metrics}), \
             patch.object(goal, 'load_bundle', side_effect=[base, new]):
            with self.assertRaisesRegex(ValueError, 'source changed during comparison'):
                comparison.diagnose('/candidate', '/baseline', '/manifest', '/teachers', 'f'*64, 59716)

    def test_existing_symlink_protected_and_nonfinite_output_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            protected = root/'source'; protected.mkdir()
            with self.assertRaises(ValueError): comparison.write_report({}, protected/'new', [protected])
            dangling = root/'dangling'; dangling.symlink_to(root/'absent')
            with self.assertRaises(ValueError): comparison.write_report({}, dangling, [])
            self.assertFalse((root/'absent').exists())
            with self.assertRaises(ValueError):
                comparison.write_report({'bad': float('nan')}, root/'nan', [])
            self.assertFalse((root/'nan').exists())
            artifact = comparison.write_report({'audit_status': 'PASS'}, root/'result', [protected])
            self.assertEqual(json.loads(Path(artifact['path']).read_text()), {'audit_status': 'PASS'})
            with self.assertRaises(ValueError): comparison.write_report({}, root/'result', [])

    def test_cli_explicit_step_and_no_expensive_read_for_existing_output(self):
        args = ['--candidate', '/candidate', '--manifest', '/manifest',
            '--teacher-selections', '/teachers', '--checkpoint-sha', 'f'*64, '--output', '/output']
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            comparison.arguments(args)
        with patch.object(Path, 'exists', return_value=True), patch.object(comparison, 'diagnose') as diagnose:
            with self.assertRaises(ValueError): comparison.main(args+['--checkpoint-step', '59716'])
            diagnose.assert_not_called()


    def test_reference_normalized_tr_is_not_macro_average(self):
        f = self.fixture
        f.manifest['reference_steps'] = {r['key']: 20 for r in f.new['rows']}
        f.refresh(f.base, self.base_contract)
        f.refresh(f.new, f.contract)
        report = self.compare()
        for label in ('candidate_metrics_percent', 'baseline_metrics_percent'):
            for task in goal.TASKS:
                self.assertEqual(report[label][task]['TR'], 25.)
                self.assertEqual(report[label][task]['macro_TR'], 50.)
                self.assertEqual(report[label][task]['reference_missing'], 0)

    def test_final_at_row_is_also_paired_and_shapes_are_checked(self):
        f = self.fixture
        original = f.new
        for mutation in ('rgb', 'agents', 'joints', 'transform', 'missing'):
            f.new = copy.deepcopy(original)
            evidence = f.new['rows'][-1]['initial_pair_evidence']
            self.assertEqual(f.new['rows'][-1]['task'], 'at')
            if mutation == 'rgb': evidence['rgb'] = 'a'*64
            elif mutation == 'agents': evidence['state']['agents'].append(copy.deepcopy(evidence['state']['agents'][0]))
            elif mutation == 'joints': evidence['state']['agents'][0]['joints'].append(0.)
            elif mutation == 'transform': evidence['state']['agents'][0]['transform'][0].append(0.)
            else: f.new['rows'][-1].pop('initial_pair_evidence')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.compare()

    def test_each_61609_pin_is_independently_enforced(self):
        for name in comparison.BASELINE_PINS:
            def fake_sha(path):
                return 'a'*64 if Path(path).name == name else comparison.BASELINE_PINS[Path(path).name]
            with self.subTest(name=name), patch.object(goal, 'file_sha', side_effect=fake_sha), \
                 patch.object(Path, 'read_text') as reader, patch.object(goal, 'read_partitions') as source:
                with self.assertRaises(ValueError):
                    goal.load_bundle('/baseline', self.fixture.manifest, self.base_contract,
                                     pins=comparison.BASELINE_PINS)
                reader.assert_not_called()
                source.assert_not_called()

    def test_pair_hash_change_and_nonbinary_metric_rejected(self):
        f = self.fixture
        original = f.new
        for mutation in ('pair_hash', 'nonbinary', 'string', 'nan'):
            f.new = copy.deepcopy(original)
            if mutation == 'pair_hash':
                k = next(iter(f.new['pair']['teacher_artifact_hashes']))
                f.new['pair']['teacher_artifact_hashes'][k]['oracle']['/fixture/start'] = 'b'*64
            else:
                f.new['rows'][-1]['collision'] = {'nonbinary': 2, 'string': '0', 'nan': float('nan')}[mutation]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.compare()



    def test_diagnose_success_keeps_two_identities_roots_and_source_inventory(self):
        f = self.fixture
        roots = {t: '/partitions/61609/'+t for t in goal.TASKS}
        new_roots = {t: '/partitions/candidate/'+t for t in goal.TASKS}
        base = dict(f.base, root='/baseline', hashes={'/baseline/source': 'e'*64},
                    summary=dict(f.base['summary'], partition_roots=roots))
        new = dict(f.new, root='/candidate', hashes={'/candidate/source': 'e'*64},
                   summary=dict(f.new['summary'], partition_roots=new_roots))
        def fake_sha(path):
            if str(path) == '/manifest': return goal.MANIFEST_SHA
            if str(path) == '/teachers': return goal.TEACHER_SHA
            return 'e'*64
        with patch.object(goal, 'file_sha', side_effect=fake_sha), \
             patch.object(Path, 'read_text', return_value=json.dumps(f.manifest)), \
             patch.object(comparison, 'load_teachers', return_value=f.teachers), \
             patch.object(comparison, 'summarize_teachers', return_value={'metrics_percent': f.teacher_metrics}), \
             patch.object(goal, 'load_bundle', side_effect=[base, new]) as loader:
            report = comparison.diagnose('/candidate', '/baseline', '/manifest', '/teachers', 'f'*64, 59716)
        self.assertEqual(report['baseline_root'], '/baseline')
        self.assertEqual(report['candidate_root'], '/candidate')
        self.assertEqual(report['candidate_partition_roots'], new_roots)
        self.assertEqual(report['baseline_partition_roots'], roots)
        self.assertEqual(report['baseline_checkpoint_sha256'], comparison.BASELINE_SHA)
        self.assertEqual(report['candidate_checkpoint_sha256'], 'f'*64)
        self.assertEqual(report['pinned_baseline_artifacts'], comparison.BASELINE_PINS)
        self.assertIn('/baseline/source', report['source_hashes'])
        self.assertIn('/candidate/source', report['source_hashes'])
        self.assertEqual(loader.call_args_list[1].args[2], f.contract)
        self.assertNotIn('goal_status', report)


if __name__ == '__main__':
    unittest.main()
