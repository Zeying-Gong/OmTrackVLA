"""Independent pure-CPU admission/runtime fixtures; no real dataset or model IO."""
import copy
import json
from contextlib import ExitStack
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from wa.wm import teacher_plan_runtime as runtime
from wa.wm.teacher_window_plan import PlannedTeacherMix, SCHEMA, SOURCE_KEYS, canonical_sha


class TinyTeacher:
    def __init__(self):
        self.cache = '/fixture/cache'
        self.episodes = [dict(task=t, episode_uid=t+':scene/'+str(i)) for i, t in
                         enumerate(('stt', 'stt', 'dt', 'at'))]
        self.episode = np.repeat(np.arange(4, dtype=np.int64), 3)
        self.rows = np.array([0, 2, 3, 4, 6, 7, 9, 10], dtype=np.int64)
        self.history = np.tile(np.array([0, 1, 2, 4], dtype=np.int64), (12, 1))
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        return dict(index=index)


def metadata_fixture():
    teacher = TinyTeacher()
    expected = [dict(episode_uid=e['episode_uid'], task=e['task'], hard=(i == 0),
                     student_success=(i != 0), teacher='oracle',
                     selected_teacher_success=True, selected_teacher_fallback=False)
                for i, e in enumerate(teacher.episodes)]
    records = [dict(e, valid_windows=2, early_windows=1) for e in expected]
    ep = teacher.episode[teacher.rows]
    windows = dict(dataset_index=np.arange(8, dtype=np.int64), raw_row=teacher.rows.copy(),
                   episode_index=ep.copy(), current_observation_index=teacher.history[teacher.rows, -1].copy(),
                   age_s=np.tile([.5, 2.5], 4), hard=ep == 0,
                   early=np.tile([True, False], 4))
    return teacher, records, windows, expected


class FakeNPZ:
    def __init__(self, arrays):
        self.arrays = arrays
        self.files = list(arrays)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def __getitem__(self, key):
        return self.arrays[key]


class TeacherPlanRuntimeTest(unittest.TestCase):
    def test_metadata_correct_returns_local_teacher_indices(self):
        teacher, records, windows, expected = metadata_fixture()
        before = copy.deepcopy((records, windows))
        self.assertEqual(runtime.verify_metadata(records, windows, expected, teacher), [0, 1])
        self.assertEqual(records, before[0])
        for key in windows:
            np.testing.assert_array_equal(windows[key], before[1][key])

    def test_schema_count_shapes_rejected(self):
        for mode in ('missing', 'extra', 'record_count', 'column', 'short'):
            teacher, records, windows, expected = metadata_fixture()
            if mode == 'missing': windows.pop('hard')
            elif mode == 'extra': windows['extra'] = windows['hard']
            elif mode == 'record_count': records.pop()
            elif mode == 'column': windows['age_s'] = windows['age_s'][:, None]
            else: windows['age_s'] = windows['age_s'][:-1]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                runtime.verify_metadata(records, windows, expected, teacher)

    def test_index_dtype_bool_float_and_out_of_bounds_rejected(self):
        for field in ('dataset_index', 'raw_row', 'episode_index', 'current_observation_index'):
            for mode in ('bool', 'float', 'negative', 'large'):
                teacher, records, windows, expected = metadata_fixture()
                if mode in ('bool', 'float'):
                    windows[field] = windows[field].astype(bool if mode == 'bool' else float)
                else:
                    windows[field][0] = -1 if mode == 'negative' else 99999
                with self.subTest(field=field, mode=mode), self.assertRaises(ValueError):
                    runtime.verify_metadata(records, windows, expected, teacher)

    def test_hard_early_must_be_boolean_and_correct(self):
        for field in ('hard', 'early'):
            for mode in ('integer', 'incorrect'):
                teacher, records, windows, expected = metadata_fixture()
                if mode == 'integer': windows[field] = windows[field].astype(np.int64)
                else: windows[field][0] = not windows[field][0]
                with self.subTest(field=field, mode=mode), self.assertRaises(ValueError):
                    runtime.verify_metadata(records, windows, expected, teacher)

    def test_bad_age_and_episode_counts_rejected(self):
        for bad in (-.1, float('nan'), float('inf')):
            teacher, records, windows, expected = metadata_fixture()
            windows['age_s'][0] = bad
            with self.subTest(age=bad), self.assertRaises(ValueError):
                runtime.verify_metadata(records, windows, expected, teacher)
        for field in ('valid_windows', 'early_windows'):
            teacher, records, windows, expected = metadata_fixture()
            records[0][field] += 1
            with self.subTest(count=field), self.assertRaises(ValueError):
                runtime.verify_metadata(records, windows, expected, teacher)

    def test_record_flags_and_counts_reject_bool_int_aliases(self):
        for field, bad in [('hard', 1), ('student_success', 0), ('selected_teacher_success', 1),
                           ('selected_teacher_fallback', 0), ('early_windows', True)]:
            teacher, records, windows, expected = metadata_fixture()
            records[0][field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                runtime.verify_metadata(records, windows, expected, teacher)

    def mix_fixture(self):
        teacher, records, windows, expected = metadata_fixture()
        hashes = {k: 'a'*64 for k in SOURCE_KEYS}
        plan = dict(schema=SCHEMA, base_count=10, teacher_count=8, extra_repeats=2,
                    extra_teacher_indices=[0, 1], source_hashes=hashes)
        mix = PlannedTeacherMix(range(10), teacher, plan, source_hashes=hashes,
                                eligible_teacher_indices=[0, 1])
        return mix, records, windows

    def test_diagnostic_positions_exact_sources_and_extras(self):
        mix, records, windows = self.mix_fixture()
        for count in (8, 16):
            positions = runtime.diagnostic_positions(mix, records, windows, count)
            self.assertEqual(len(positions), count)
            self.assertEqual(len(set(positions)), count)
            self.assertTrue(all(type(p) is int and 0 <= p < len(mix) for p in positions))
            self.assertTrue({0, len(mix.base)-1, len(mix.base)+len(mix.teacher)}.issubset(positions))
            self.assertEqual(positions, runtime.diagnostic_positions(mix, records, windows, count))
            self.assertEqual({records[int(windows['episode_index'][p-len(mix.base)])]['task']
                              for p in positions if len(mix.base) <= p < len(mix.base)+len(mix.teacher)},
                             {'stt', 'dt', 'at'})

    def test_diagnostic_count_rejects_bad_types_and_bounds(self):
        mix, records, windows = self.mix_fixture()
        for count in (True, False, 0, -2, 6, 9, 100, 8.0, 16.0):
            with self.subTest(count=count), self.assertRaises(ValueError):
                runtime.diagnostic_positions(mix, records, windows, count)

    def test_diagnostic_missing_task_or_hard_group_rejected(self):
        mix, records, windows = self.mix_fixture()
        records[-1]['task'] = 'dt'
        with self.assertRaises(ValueError):
            runtime.diagnostic_positions(mix, records, windows)
        mix, records, windows = self.mix_fixture()
        windows['hard'][1] = False
        with self.assertRaises(ValueError):
            runtime.diagnostic_positions(mix, records, windows)

    def candidate_fixture(self):
        teacher, records, windows, expected = metadata_fixture()
        directory, base_index = Path('/fixture/candidate'), Path('/fixture/base')
        student_rows, selections = Path('/fixture/students.jsonl'), Path('/fixture/selections.jsonl')
        pinned = {k: format(i+1, '064x') for i, k in enumerate((*SOURCE_KEYS, 'teacher_release'))}
        source_hashes = {k: pinned[k] for k in SOURCE_KEYS}
        plan = dict(schema=SCHEMA, base_count=10, teacher_count=8, extra_repeats=2,
                    extra_teacher_indices=[0, 1], source_hashes=source_hashes)
        paths = dict(cache_complete=Path(teacher.cache)/'complete.json',
                     teacher_index=Path(teacher.cache)/'index/train_valid.npy',
                     base_index=base_index/'train_valid.npy', student_rows=student_rows,
                     teacher_selections=selections)
        report = dict(status='CANDIDATE_ONLY_NOT_TRAINING_RELEASE', experiment=runtime.EXPERIMENT,
                      checkpoint_sha256=runtime.CHECKPOINT_SHA, checkpoint_step=59065,
                      artifacts={name: 'b'*64 for name in
                                 ('plan.json', 'episodes.json', 'teacher_windows.npz', 'exposure.json')},
                      source_hashes={str(path): pinned[key] for key, path in paths.items()},
                      plan_canonical_sha256=canonical_sha(plan), eligible_teacher_windows=2)
        texts = {str(directory/'report.json'): json.dumps(report),
                 str(directory/'plan.json'): json.dumps(plan),
                 str(directory/'episodes.json'): json.dumps(records),
                 str(student_rows): '{}\n', str(selections): '{}\n',
                 str(Path(teacher.cache)/'complete.json'): json.dumps(dict(source_release='/fixture/release.json')),
                 '/fixture/release.json': json.dumps(dict(teacher_demonstrations=[]))}
        hashes = {str(path): pinned[key] for key, path in paths.items()}
        hashes.update({str(directory/name): value for name, value in report['artifacts'].items()})
        hashes[str(directory/'report.json')] = 'c'*64
        hashes['/fixture/release.json'] = pinned['teacher_release']
        kwargs = dict(base_index=base_index, student_rows=student_rows, teacher_selections=selections)
        return directory, teacher, expected, windows, report, texts, hashes, pinned, kwargs

    def run_candidate(self, fixture):
        directory, teacher, expected, windows, report, texts, hashes, pinned, kwargs = fixture
        texts[str(directory/'report.json')] = json.dumps(report)
        with ExitStack() as stack:
            stack.enter_context(patch.object(runtime, 'PINNED', pinned))
            stack.enter_context(patch.object(runtime, 'sha', side_effect=lambda path: hashes[str(Path(path))]))
            stack.enter_context(patch.object(Path, 'read_text', lambda path, *a, **k: texts[str(path)]))
            stack.enter_context(patch.object(runtime.np, 'load', return_value=FakeNPZ(windows)))
            stack.enter_context(patch.object(runtime, 'derive_admissions', return_value=(expected, {})))
            return runtime.load_candidate(directory, 'c'*64, range(10), teacher, **kwargs)

    def test_candidate_correct_with_mocked_immutable_evidence(self):
        mix, records, windows, report = self.run_candidate(self.candidate_fixture())
        self.assertEqual(len(mix), 22)
        self.assertEqual(mix.extra, (0, 1))
        self.assertEqual(report['eligible_teacher_windows'], 2)

    def test_candidate_report_artifact_source_and_qualification_rejected(self):
        for mode in ('report_hash', 'artifact_hash', 'source_sha', 'source_inventory',
                     'protocol', 'checkpoint', 'canonical', 'extras'):
            fixture = self.candidate_fixture()
            directory, teacher, expected, windows, report, texts, hashes, pinned, kwargs = fixture
            if mode == 'report_hash': hashes[str(directory/'report.json')] = '0'*64
            elif mode == 'artifact_hash': hashes[str(directory/'plan.json')] = '0'*64
            elif mode == 'source_sha': hashes[str(kwargs['student_rows'])] = '0'*64
            elif mode == 'source_inventory': report['source_hashes'][str(kwargs['student_rows'])] = '0'*64
            elif mode == 'protocol': report['status'] = 'PASS'
            elif mode == 'checkpoint': report['checkpoint_sha256'] = '0'*64
            elif mode == 'canonical': report['plan_canonical_sha256'] = '0'*64
            else: report['eligible_teacher_windows'] = 3
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.run_candidate(fixture)


if __name__ == '__main__':
    unittest.main()
