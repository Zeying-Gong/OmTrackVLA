"""SYNTHETIC_NOT_MODEL_RESULT: full4215 CPU cross-mode comparison fixtures.

Core shard loading, metrics, teacher selection and pairing are real code paths.
Only immutable manifest/teacher SHA constants are rebound to synthetic bytes.
No teacher media, checkpoint, simulator or GPU is used.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wa.tools import compare_uwb_modes as comparison
from wa.wm import student_eval_finalize as finalizer
from wa.wm.dual_teacher_metrics import summarize_teachers
from wa.wm.dual_teacher_selection import select_teacher
from wa.wm.full_mixed_contract import summarize
from wa.wm.initial_bbox_repair import KEYS, VERSION
from wa.wm.student_eval_contract import REPAIR_SHA
from wa.wm.student_eval_partition import write_partition

CONTRACT = dict(checkpoint_sha='f'*64, step=61252)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False))


def build_fixture(root):
    manifest = dict(tasks={}, reference_steps={})
    teachers, rows = [], {'mixed': [], 'image': []}
    pair_hashes = {}
    state = dict(timestamp=0., agents=[dict(transform=[[1., 0.], [0., 1.]], joints=[0.])])
    for task in ('stt', 'dt', 'at'):
        keys = sorted(key for t, key in KEYS if t == task)
        keys += ['synthetic/'+str(i) for i in range(1405-len(keys))]
        manifest['tasks'][task] = dict(episodes=[dict(key=k, shard=i % 8) for i, k in enumerate(keys)])
        for i, key in enumerate(keys):
            manifest['reference_steps'][key] = 20
            branches, saved = {}, {}
            for teacher in ('lightnav', 'oracle'):
                branch_root = Path('/synthetic_teacher')/task/key/teacher
                result = dict(success=True, collision=False, policy_init_valid=True,
                              following_rate=.8, following_step=8, total_step=10)
                branches[teacher] = dict(experiment='evaluation_set_adaptation_v1',
                    teacher=teacher, complete=True, replay_verified=True,
                    task=task, key=key, takeover_step=0, seed=7, protocol_sha256='a'*64,
                    initial_rgb_sha256='d'*64, takeover_state_sha256='b'*64,
                    result=result, artifact_root=str(branch_root))
                saved[teacher] = {str(branch_root/name): 'e'*64
                                  for name in ('pair_start.json', 'observations.json', 'rgb_0000.png')}
            teacher_row = select_teacher(branches['lightnav'], branches['oracle'])
            teacher_row['branches'] = branches
            teachers.append(teacher_row)
            pair_hashes[task+':'+key] = saved
            for mode in ('mixed', 'image'):
                valid = i not in ((7,) if mode == 'mixed' else (3, 7))
                success = i % 4 in ((0, 1) if mode == 'mixed' else (0, 2))
                row = dict(task=task, key=key, mode=mode, noise_mode='zero',
                    controller='learned_yaw_guard_v1', checkpoint_sha256=CONTRACT['checkpoint_sha'],
                    checkpoint_step=CONTRACT['step'], semantic_protocol='mp3d_semantic_ply_v1',
                    success=success, collision=bool(not success and i % 7 == 0),
                    policy_init_valid=valid, finish=True,
                    following_rate=(.5 if mode == 'mixed' else .75) if valid else 0.,
                    following_step=(5 if mode == 'mixed' else 9) if valid else 0,
                    total_step=10 if mode == 'mixed' else 12,
                    initial_pair_evidence=dict(rgb='d'*64, state=copy.deepcopy(state)))
                if (task, key) in KEYS:
                    row.update(initialization_repair=VERSION, initialization_repair_plan_sha256=REPAIR_SHA)
                rows[mode].append(row)
    manifest_path, teacher_path = root/'manifest.json', root/'teachers.jsonl'
    write_json(manifest_path, manifest)
    teacher_path.write_text(''.join(json.dumps(r)+'\n' for r in teachers))
    teacher_sha = comparison.file_sha(teacher_path)
    teacher_metrics = summarize_teachers(teachers, manifest)['metrics_percent']
    pair = dict(status='PASS', episodes=4215, teacher_artifact_hashes=pair_hashes,
                comparison='SYNTHETIC_NOT_MODEL_RESULT', scope='synthetic stored evidence')
    bundles = {}
    for mode in ('mixed', 'image'):
        merged = root/(mode+'_merged'); merged.mkdir()
        partitions = {}
        for task in ('stt', 'dt', 'at'):
            part = root/(mode+'_'+task); part.mkdir(); partitions[task] = str(part)
            combined = []
            for shard in range(8):
                dest = part/f'shard_{shard:02d}'; dest.mkdir()
                keys = {e['key'] for e in manifest['tasks'][task]['episodes'] if e['shard'] == shard}
                local = [r for r in rows[mode] if r['task'] == task and r['key'] in keys]
                (dest/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in local))
                write_json(dest/'COMPLETE.json', dict(episodes=len(local), mode=mode))
                write_json(dest/'server_ready.json', dict(checkpoint_sha256=CONTRACT['checkpoint_sha'],
                    step=CONTRACT['step'], mode=mode, noise_mode='zero', sampling_steps=4,
                    seed='7+step', text_used=False, world_predictor_inference=False))
                combined += [dict(r, artifact_root=str(dest)) for r in local]
            write_partition(part, combined, manifest, CONTRACT, task, mode=mode)
        canonical, hashes = comparison.read_partitions(partitions, manifest, CONTRACT, mode=mode)
        summary = summarize(canonical, manifest, mode=mode, **CONTRACT)
        summary['limits'].append(comparison.INSET_LIMIT)
        summary.update(experiment='evaluation_set_adaptation_v1', new_episodes=4215,
            reused_baseline_episodes=0, partition_roots=partitions, source_hashes=hashes,
            teacher_reference_sha256=teacher_sha, teacher_metrics_percent=teacher_metrics,
            superiority=finalizer.superiority(canonical, teachers))
        write_json(merged/'summary.json', summary)
        write_json(merged/'student_teacher_pair_audit.json', pair)
        (merged/'combined_episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in canonical))
        bundles[mode] = dict(root=str(merged), rows=canonical, summary=summary, pair=copy.deepcopy(pair))
    return dict(manifest=manifest, teachers=teachers, teacher_metrics=teacher_metrics,
                manifest_path=manifest_path, teacher_path=teacher_path, bundles=bundles,
                teacher_sha=teacher_sha, manifest_sha=comparison.file_sha(manifest_path))


class UWBModeComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='wa_synthetic_uwb_')
        cls.root = Path(cls.temp.name)
        cls.fixture = build_fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        f = self.fixture
        self.bundles = copy.deepcopy(f['bundles'])
        self.addCleanup(patch.stopall)
        patch.object(comparison, 'MANIFEST_SHA', f['manifest_sha']).start()
        patch.object(comparison, 'TEACHER_SHA', f['teacher_sha']).start()
        patch.object(finalizer, 'TEACHER_SHA', f['teacher_sha']).start()

    def compare(self):
        f = self.fixture
        return comparison.compare(self.bundles['mixed'], self.bundles['image'],
            f['manifest'], CONTRACT, f['teachers'], f['teacher_metrics'])

    def diagnose(self):
        f = self.fixture
        return comparison.diagnose(f['bundles']['mixed']['root'], f['bundles']['image']['root'],
            f['manifest_path'], f['teacher_path'], CONTRACT['checkpoint_sha'], CONTRACT['step'])

    def refresh(self, mode='image'):
        f = self.fixture
        bundle = self.bundles[mode]
        saved = bundle['summary']
        computed = summarize(bundle['rows'], f['manifest'], mode=mode, **CONTRACT)
        computed['limits'].append(comparison.INSET_LIMIT)
        saved.update(computed, superiority=finalizer.superiority(bundle['rows'], f['teachers']))

    @contextlib.contextmanager
    def changed_file(self, path, data):
        path = Path(path); old = path.read_bytes()
        try:
            path.write_bytes(data)
            yield
        finally:
            path.write_bytes(old)

    def test_full_disk_pipeline_and_four_cells(self):
        report = self.diagnose()
        self.assertEqual(report['audit_status'], 'PASS_STORED_EVIDENCE_COMPARISON_ONLY')
        self.assertFalse(report['ablation_release'])
        self.assertEqual(len(report['protected_teacher_artifact_roots']), 8430)
        self.assertTrue(any(p.endswith('/wa/wm/student_pair_audit.py') for p in report['source_hashes']))
        self.assertNotIn('goal_status', report)
        self.assertEqual(len(report['paired']), 4215)
        for task in ('stt', 'dt', 'at'):
            values = report['tasks'][task]
            self.assertEqual(values['paired_counts'], dict(both_success=352,
                mixed_only_success=351, image_only_success=351, both_failure=351))
            self.assertEqual(sum(map(len, values['paired_keys'].values())), 1405)
            self.assertEqual(values['delta_image_minus_mixed']['success_count'], 0)
        self.assertFalse(report['paired'][0]['image']['media_verified_by_this_tool'])
        self.assertTrue(report['paired'][0]['image']['video_path'].endswith('/review.mp4'))
        for mode in ('mixed', 'image'):
            source = comparison.load_bundle(self.fixture['bundles'][mode]['root'],
                self.fixture['manifest'], CONTRACT, mode)
            self.assertEqual(len(source['hashes']), 81)

    def test_reference_tr_macro_and_invalid_full_denominator(self):
        r = self.compare()['tasks']['stt']
        self.assertEqual(r['mixed']['TR_denominator'], 28100)
        self.assertEqual(r['image']['TR_denominator'], 28100)
        self.assertAlmostEqual(r['mixed']['TR'], 100*1404*5/28100)
        self.assertAlmostEqual(r['image']['TR'], 100*1403*9/28100)
        self.assertNotEqual(r['image']['TR'], r['image']['macro_TR'])
        self.assertEqual(r['mixed']['invalid_init_count'], 1)
        self.assertEqual(r['image']['invalid_init_count'], 2)
        self.assertEqual(r['delta_image_minus_mixed']['invalid_init_count'], 1)
        self.assertAlmostEqual(r['delta_image_minus_mixed']['invalid_init_percent'], 100/1405)
        self.assertAlmostEqual(r['image']['SR'], 100*703/1405)

    def test_foreign_sha_step_mode_task_key_and_coverage(self):
        original = self.bundles['image']
        for mutation in ('sha', 'step', 'mode', 'task', 'key', 'duplicate', 'truncated'):
            self.bundles['image'] = copy.deepcopy(original)
            rows = self.bundles['image']['rows']
            if mutation == 'sha': rows[0]['checkpoint_sha256'] = 'a'*64
            elif mutation == 'step': rows[0]['checkpoint_step'] = 59716
            elif mutation == 'mode': rows[0]['mode'] = 'mixed'
            elif mutation == 'task': rows[0]['task'] = 'other'
            elif mutation == 'key': rows[0]['key'] = 'foreign/1'
            elif mutation == 'duplicate': rows[-1] = rows[0]
            else: rows.pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.compare()

    def test_rgb_state_nonfinite_missing_and_last_key(self):
        original = self.bundles['image']
        for mutation in ('rgb', 'state', 'nan', 'missing', 'agents', 'shape'):
            self.bundles['image'] = copy.deepcopy(original)
            row = self.bundles['image']['rows'][-1]
            e = row['initial_pair_evidence']
            if mutation == 'rgb': e['rgb'] = 'a'*64
            elif mutation == 'state': e['state']['agents'][0]['joints'][0] = .01
            elif mutation == 'nan': e['state']['timestamp'] = float('nan')
            elif mutation == 'missing': row.pop('initial_pair_evidence')
            elif mutation == 'agents': e['state']['agents'] = []
            else: e['state']['agents'][0]['transform'].append([0., 0.])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.compare()

    def test_dynamic_tolerance_and_future_outcomes_may_differ(self):
        row = self.bundles['image']['rows'][0]
        row['initial_pair_evidence']['state']['timestamp'] = 5e-7
        row['success'] = False
        row['total_step'] = 100
        self.refresh()
        report = self.compare()
        self.assertEqual(report['tasks'][row['task']]['delta_image_minus_mixed']['success_count'], -1)

    def test_semantic_repair_and_controller_rejected(self):
        original = self.bundles['image']
        for field, value in (('semantic_protocol', 'old'), ('initialization_repair', None),
                             ('initialization_repair_plan_sha256', 'a'*64), ('controller', 'uwb_heading_v1')):
            self.bundles['image'] = copy.deepcopy(original)
            row = next(r for r in self.bundles['image']['rows'] if (r['task'], r['key']) in KEYS)
            row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.compare()

    def test_summary_metrics_reuse_identity_and_teacher_tamper(self):
        original = self.bundles['image']
        changes = [('metrics_percent', {}), ('new_episodes', 4214), ('reused_baseline_episodes', 1),
                   ('checkpoint_step', 1), ('checkpoint_sha256', 'b'*64), ('mode', 'mixed'),
                   ('teacher_reference_sha256', 'b'*64), ('superiority', {}),
                   ('teacher_metrics_percent', {}), ('limits', [])]
        for field, value in changes:
            self.bundles['image'] = copy.deepcopy(original)
            self.bundles['image']['summary'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.compare()

    def test_teacher_pair_status_coverage_hash_and_path_tamper(self):
        original = self.bundles['image']
        for mutation in ('status', 'count', 'key', 'hash', 'path', 'teacher'):
            self.bundles['image'] = copy.deepcopy(original)
            pair = self.bundles['image']['pair']
            inventory = pair['teacher_artifact_hashes']
            item = inventory[next(iter(inventory))]
            if mutation == 'status': pair['status'] = 'PARTIAL'
            elif mutation == 'count': pair['episodes'] = 4214
            elif mutation == 'key': inventory.pop(next(iter(inventory)))
            elif mutation == 'hash': item['oracle'][next(iter(item['oracle']))] = 'a'*64
            elif mutation == 'path':
                files = item['oracle']; files['/foreign/start'] = files.pop(next(iter(files)))
            else: item.pop('oracle')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.compare()

    def test_bad_metrics_and_unsafe_video_path(self):
        for value in (2, '0', float('nan')):
            self.bundles = copy.deepcopy(self.fixture['bundles'])
            self.bundles['image']['rows'][0]['success'] = value
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                self.compare()
        row = dict(task='stt', key='../escape', artifact_root='/source')
        with self.assertRaises(ValueError): comparison.video_reference(row)

    def test_distinct_bundle_roots_required(self):
        self.bundles['image']['root'] = self.bundles['mixed']['root']
        with self.assertRaisesRegex(ValueError, 'distinct'):
            self.compare()

    def test_real_disk_source_inventory_and_combined_tamper(self):
        root = Path(self.fixture['bundles']['image']['root'])
        for mutation in ('inventory', 'rows'):
            if mutation == 'inventory':
                path = root/'summary.json'; payload = json.loads(path.read_text())
                payload['source_hashes'] = {}; data = json.dumps(payload).encode()
            else:
                path = root/'combined_episodes.jsonl'; data = path.read_bytes().splitlines(keepends=True)[0]
            with self.subTest(mutation=mutation), self.changed_file(path, data):
                with self.assertRaisesRegex(ValueError, 'original shards'):
                    self.diagnose()

    def test_persisted_ready_seed_is_required(self):
        part = self.fixture['bundles']['image']['summary']['partition_roots']['stt']
        path = Path(part)/'shard_00'/'server_ready.json'
        value = json.loads(path.read_text()); value['seed'] = '8+step'
        updated = json.dumps(value).encode()
        summary_path = Path(self.fixture['bundles']['image']['root'])/'summary.json'
        summary = json.loads(summary_path.read_text())
        with self.changed_file(path, updated):
            summary['source_hashes'][str(path)] = comparison.file_sha(path)
            with self.changed_file(summary_path, json.dumps(summary).encode()):
                with self.assertRaisesRegex(ValueError, 'persisted ready seed'):
                    self.diagnose()

    def test_manifest_and_teacher_file_pins_not_mocked_validation(self):
        f = self.fixture
        for path in (f['manifest_path'], f['teacher_path']):
            with self.subTest(path=path.name), self.changed_file(path, path.read_bytes()+b' '):
                with self.assertRaises(ValueError):
                    self.diagnose()

    def test_source_mutation_during_actual_comparison_is_detected(self):
        path = Path(self.fixture['bundles']['image']['root'])/'summary.json'
        original_bytes = path.read_bytes()
        original_compare = comparison.compare
        def compare_then_change(*args, **kwargs):
            result = original_compare(*args, **kwargs)
            path.write_bytes(original_bytes+b' ')
            return result
        try:
            with patch.object(comparison, 'compare', side_effect=compare_then_change):
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    self.diagnose()
        finally:
            path.write_bytes(original_bytes)

    def test_output_existing_symlink_ancestor_and_protected(self):
        with tempfile.TemporaryDirectory(prefix='wa_synthetic_output_') as d:
            root = Path(d)
            source = root/'source'; source.mkdir()
            cases = [source, source/'new']
            dangling = root/'dangling'; dangling.symlink_to(root/'missing')
            ancestor = root/'linked'; ancestor.symlink_to(source, target_is_directory=True)
            cases += [dangling, ancestor/'new']
            for output in cases:
                with self.subTest(output=output), self.assertRaises(ValueError):
                    comparison.output_path(output, [source])
            with self.assertRaises(ValueError):
                comparison.output_path(root/'future', [root/'future'/'protected'])

    def test_write_rechecks_sources_and_refuses_nonfinite_or_overwrite(self):
        report = self.diagnose()
        with tempfile.TemporaryDirectory(prefix='wa_synthetic_output_') as d:
            root = Path(d)
            invalid = dict(report, bad=float('nan'))
            with self.assertRaises(ValueError):
                comparison.write_report(invalid, root/'nan', [])
            self.assertFalse((root/'nan').exists())
            path = Path(self.fixture['bundles']['image']['root'])/'summary.json'
            with self.changed_file(path, path.read_bytes()+b' '):
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    comparison.write_report(report, root/'changed', [])
            self.assertFalse((root/'changed').exists())
            artifact = comparison.write_report(report, root/'pass', [])
            saved = json.loads(Path(artifact['path']).read_text())
            self.assertEqual(saved['schema'], comparison.SCHEMA)
            with self.assertRaises(ValueError):
                comparison.write_report(report, root/'pass', [])

    def test_teacher_source_tree_is_protected_without_reading_media(self):
        report = self.diagnose()
        destination = Path(report['protected_teacher_artifact_roots'][0])/'new_result'
        with self.assertRaisesRegex(ValueError, 'immutable inputs'):
            comparison.write_report(report, destination, [])

    def test_cli_requires_explicit_step_and_early_output_guard(self):
        args = ['--mixed', '/m', '--image', '/i', '--manifest', '/manifest',
                '--teacher-selections', '/teachers', '--checkpoint-sha', 'f'*64,
                '--output', str(self.root)]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            comparison.arguments(args)
        with patch.object(comparison, 'diagnose') as diagnose:
            with self.assertRaises(ValueError):
                comparison.main(args+['--checkpoint-step', '61252'])
            diagnose.assert_not_called()


if __name__ == '__main__':
    unittest.main()
