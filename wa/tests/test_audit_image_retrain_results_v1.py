"""SYNTHETIC_NOT_MODEL_RESULT tests for the two-checkpoint image audit.

Exercises the real 4,215-row validation and pairing functions with in-memory
synthetic rows. No GPU, simulator, model file, or live evaluation is accessed.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wa.tests.test_compare_uwb_modes import build_fixture
from wa.tools import audit_image_retrain_results_v1 as audit
from wa.tools import compare_uwb_modes as old_comparison
from wa.wm import student_eval_finalize as finalizer
from wa.wm.full_mixed_contract import summarize

OLD = 'f'*64
NEW = 'e'*64


class ImageRetrainAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='wa_synthetic_image_retrain_')
        cls.root = Path(cls.temp.name)
        cls.fixture = build_fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        f = self.fixture
        self.old = copy.deepcopy(f['bundles']['image'])
        self.new = copy.deepcopy(self.old)
        self.new['root'] = str(self.root/'new_merged')
        for row in self.new['rows']:
            row['checkpoint_sha256'] = NEW
        self._refresh()
        self.patchers = [
            patch.object(audit, 'OLD_SHA', OLD),
            patch.object(audit, 'NEW_SHA', NEW),
            patch.object(audit, 'MANIFEST_SHA', f['manifest_sha']),
            patch.object(audit, 'TEACHER_SHA', f['teacher_sha']),
            patch.object(old_comparison, 'TEACHER_SHA', f['teacher_sha']),
        ]
        for item in self.patchers:
            item.start()
            self.addCleanup(item.stop)

    def _refresh(self):
        f = self.fixture
        computed = summarize(self.new['rows'], f['manifest'], mode='image',
                             checkpoint_sha=NEW, step=audit.STEP)
        computed['limits'].append(old_comparison.INSET_LIMIT)
        self.new['summary'].update(
            computed, superiority=finalizer.superiority(self.new['rows'], f['teachers']))
        self.new['summary']['teacher_reference_sha256'] = f['teacher_sha']

    def _compare(self):
        f = self.fixture
        return audit.compare(self.old, self.new, f['manifest'], f['teachers'],
                             f['teacher_metrics'], dict(synthetic=True))

    def test_full_paired_different_weights_and_corrected_wording(self):
        row = next(row for row in self.new['rows']
                   if row['task'] == 'stt' and not row['success'])
        row['success'] = True
        self._refresh()
        result = self._compare()
        self.assertEqual(result['episodes'], 4215)
        self.assertEqual(result['input_mode'], 'image')
        self.assertEqual(result['tasks']['stt']['delta_retrained_minus_direct']['success_count'], 1)
        self.assertEqual(result['tasks']['stt']['paired_counts']['retrained_only_success'], 1)
        self.assertEqual(len(result['paired']), 4215)
        self.assertIn('not no-UWB training', self.new['summary']['limits'][0])
        self.assertIn('checkpoint retrained image-only',
                      result['derived_new_summary']['limits'][0])
        self.assertEqual(result['tasks']['stt']['retrained']['episodes'], 1405)
        self.assertEqual(result['audit_status'],
                         'PASS_STORED_EVIDENCE_DIFFERENT_WEIGHTS_ONLY')
        self.assertFalse(result['tasks']['stt']['stt_image_strictly_above_80_goal'])
        self.assertNotIn('stt_image_strictly_above_80_goal', result['tasks']['dt'])
        self.assertNotIn('stt_image_strictly_above_80_goal', result['tasks']['at'])

    def test_initial_state_mismatch_rejected(self):
        self.new['rows'][0]['initial_pair_evidence']['state']['timestamp'] = 9.
        with self.assertRaisesRegex(ValueError, 'initial dynamic state'):
            self._compare()

    def test_missing_completed_row_rejected(self):
        self.new['rows'].pop()
        with self.assertRaises(ValueError):
            self._compare()

    def test_foreign_weight_row_rejected(self):
        self.new['rows'][0]['checkpoint_sha256'] = OLD
        with self.assertRaises(ValueError):
            self._compare()

    def test_same_weight_ablation_cannot_be_mislabelled_retraining(self):
        with patch.object(audit, 'NEW_SHA', OLD):
            with self.assertRaisesRegex(ValueError, 'distinct checkpoint'):
                self._compare()

    def test_training_audit_requires_actual_image_only_exposure(self):
        def record(checkpoint, sha, source, config, schema, status):
            return dict(schema=schema, status=status, checkpoint=str(checkpoint),
                checkpoint_sha256=sha, final_step=audit.STEP, completed_epochs=2,
                closed_loop=False, parent_checkpoint=str(audit.PARENT_CHECKPOINT),
                parent_sha256=audit.PARENT_SHA, source_commit=source,
                config_sha256=config, source_hashes={str(checkpoint): sha},
                hardware=dict(unique_gpu_uuids=[str(i) for i in range(8)]),
                optimizer_continuation=dict(parent_step=audit.PARENT_STEP,
                    final_step=audit.STEP, updates=audit.UPDATES),
                exposure=dict(calculated=dict(audit.EXPOSURES)))
        old = record(audit.OLD_CHECKPOINT, OLD, audit.OLD_SOURCE,
                     audit.OLD_TRAIN_CONFIG, 'failure_state_training_audit_v1',
                     'TRAINING_ARTIFACT_AUDIT_PASS_OFFLINE_ONLY')
        new = record(audit.NEW_CHECKPOINT, NEW, audit.NEW_SOURCE,
                     audit.NEW_TRAIN_CONFIG,
                     'failure_state_image_only_training_audit_v1',
                     'TRAINING_ARTIFACT_AUDIT_PASS_OFFLINE_ONLY_IMAGE_INPUT')
        new.update(train_input_mode='image', mode_exposure=dict(audit.MODE_EXPOSURE),
                   worker_postcheck=dict(train_input_mode='image',
                       mode_exposure=dict(audit.MODE_EXPOSURE),
                       status='TRAINING_WORKER_POSTCHECK_PASS_OFFLINE_ONLY'))
        provenance = audit.validate_training_audits(old, new)
        self.assertEqual(provenance['new']['mode_exposure']['image'], 1233424)
        new['mode_exposure']['mixed'] = 1
        with self.assertRaisesRegex(ValueError, 'image-only retraining'):
            audit.validate_training_audits(old, new)


class BoundJobRootTests(unittest.TestCase):
    """Synthetic on-disk pins reject plausible but foreign merged bundles."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='wa_synthetic_image_pins_')
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.old, self.new = root/'old_merged', root/'new_merged'
        self.old.mkdir()
        self.new.mkdir()
        self.old_parts = {task: root/('old_'+task) for task in ('stt', 'dt', 'at')}
        self.new_parts = {task: root/('new_'+task) for task in ('stt', 'dt', 'at')}
        for part in (*self.old_parts.values(), *self.new_parts.values()):
            part.mkdir()
        self.write_summary(self.old, self.old_parts)
        self.write_summary(self.new, self.new_parts)
        (self.old/'combined_episodes.jsonl').write_text('synthetic\n')
        (self.old/'student_teacher_pair_audit.json').write_text('{}')
        pins = {name: audit.file_sha(self.old/name) for name in audit.OLD_MERGED_PINS}
        for field, value in (
                ('OLD_MERGED', self.old), ('OLD_MERGED_PINS', pins),
                ('OLD_PARTITIONS', self.old_parts), ('NEW_PARTITIONS', self.new_parts)):
            patcher = patch.object(audit, field, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def write_summary(root, parts):
        (root/'summary.json').write_text(json.dumps(
            {'partition_roots': {task: str(path) for task, path in parts.items()}}))

    def test_exact_old_file_hashes_and_job_roots_pass(self):
        hashes = audit.validate_bound_roots(self.old, self.new)
        self.assertEqual(len(hashes), 3)
        self.assertEqual(hashes[str(self.old/'summary.json')],
                         audit.OLD_MERGED_PINS['summary.json'])

    def test_foreign_old_merged_root_rejected(self):
        foreign = self.old.parent/'foreign_old'
        foreign.mkdir()
        self.write_summary(foreign, self.old_parts)
        with self.assertRaisesRegex(ValueError, 'audited 62256'):
            audit.validate_bound_roots(foreign, self.new)

    def test_mutated_pinned_old_combined_rows_rejected(self):
        (self.old/'combined_episodes.jsonl').write_text('different synthetic\n')
        with self.assertRaisesRegex(ValueError, 'audited old merged file changed'):
            audit.validate_bound_roots(self.old, self.new)

    def test_old_summary_self_declared_foreign_partition_rejected(self):
        foreign = self.old.parent/'foreign_task'
        foreign.mkdir()
        swapped = dict(self.old_parts, stt=foreign)
        self.write_summary(self.old, swapped)
        with self.assertRaisesRegex(ValueError, 'pinned Job/Task IDs'):
            audit.validate_bound_roots(self.old, self.new)

    def test_new_summary_self_declared_foreign_partition_rejected(self):
        foreign = self.new.parent/'foreign_task'
        foreign.mkdir()
        swapped = dict(self.new_parts, at=foreign)
        self.write_summary(self.new, swapped)
        with self.assertRaisesRegex(ValueError, 'pinned Job/Task IDs'):
            audit.validate_bound_roots(self.old, self.new)

    def test_new_root_cannot_be_old_root_or_nested_under_old(self):
        with self.assertRaisesRegex(ValueError, 'distinct'):
            audit.validate_bound_roots(self.old, self.old)
        nested = self.old/'nested'
        nested.mkdir()
        self.write_summary(nested, self.new_parts)
        with self.assertRaisesRegex(ValueError, 'distinct'):
            audit.validate_bound_roots(self.old, nested)


if __name__ == '__main__':
    unittest.main()
