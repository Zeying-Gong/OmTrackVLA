"""Bounded CPU validation of v2 admission, sampling and non-policy bookkeeping."""
from contextlib import ExitStack
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from wa.wm.teacher_window_plan import (SCHEMA as V1, SOURCE_KEYS, PlannedTeacherMix,
    ExposureTaggedData, strip_exposure_tag, simulate_exposure, canonical_sha)
from wa.wm.teacher_window_schedule import (SCHEMA, SELECTION_SOURCE_KEYS,
    ScheduledTeacherMix)
from wa.wm.runtime_exposure import RuntimeExposure
from wa.wm import teacher_plan_runtime_v2 as runtime


def fixture(base=103):
    n = 20
    hashes = {k: 'a'*64 for k in SOURCE_KEYS}
    selections = {k: 'b'*64 for k in SELECTION_SOURCE_KEYS}
    groups = dict(hard_stt_early=[0], hard_stt_late=[1],
        new_regression_stt=[2], successful_stt_anchor=[3])
    extra = [0, 0, 1, 2, 3]
    plan = dict(schema=SCHEMA, base_count=base, teacher_count=n,
        extra_teacher_indices=extra, source_hashes=hashes,
        selection_source_hashes=selections)
    data = lambda size: [dict(rgb=torch.tensor([float(i)]),
        pose=torch.tensor([float(i), 0.]), commands=torch.tensor([i])) for i in range(size)]
    kwargs = dict(source_hashes=hashes, selection_source_hashes=selections,
        expected_extra_teacher_indices=extra, sampling_groups=groups)
    mix = ScheduledTeacherMix(data(base), data(n), plan, **kwargs)
    records = [dict(episode_uid=(('dt' if i == 18 else 'at' if i == 19 else 'stt')+':scene/'+str(i)),
        task='dt' if i == 18 else 'at' if i == 19 else 'stt', teacher='lightnav') for i in range(n)]
    hard = np.arange(n) < 2
    early = np.arange(n) % 2 == 0
    meta = dict(episode_index=np.arange(n), episodes=records, hard=hard, early=early)
    return mix, plan, kwargs, meta


class FakeNPZ:
    def __init__(self, arrays): self.arrays, self.files = arrays, list(arrays)
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def __getitem__(self, key): return self.arrays[key]


class ScheduleV2Test(unittest.TestCase):
    def test_explicit_positions_repeat_source_not_labels(self):
        mix, plan, kwargs, meta = fixture()
        offset = len(mix.base) + len(mix.teacher)
        self.assertEqual([mix.locate(i) for i in range(offset, len(mix))],
            [('teacher', i) for i in [0, 0, 1, 2, 3]])
        for i in range(offset, len(mix)):
            src = mix.teacher[mix.locate(i)[1]]
            self.assertEqual(set(mix[i]), set(src))
            for key in src: torch.testing.assert_close(mix[i][key], src[key], rtol=0, atol=0)
        tagged = next(iter(DataLoader(ExposureTaggedData(mix), batch_size=2)))
        clean, ids = strip_exposure_tag(tagged)
        self.assertEqual(set(clean), {'rgb', 'pose', 'commands'})
        self.assertEqual(ids.tolist(), [0, 1])

    def test_v1_order_and_cross_schema_rejection(self):
        mix, plan, kwargs, _ = fixture()
        v1 = dict(schema=V1, base_count=103, teacher_count=20,
            extra_repeats=2, extra_teacher_indices=[0, 3], source_hashes=plan['source_hashes'])
        old = PlannedTeacherMix(mix.base, mix.teacher, v1, source_hashes=plan['source_hashes'],
            eligible_teacher_indices=[0, 3])
        self.assertEqual(old.extra_positions, (0, 3, 0, 3))
        self.assertEqual([old.locate(i)[1] for i in range(123, len(old))], [0, 3, 0, 3])
        with self.assertRaises(ValueError): ScheduledTeacherMix(mix.base, mix.teacher, v1, **kwargs)
        with self.assertRaises(ValueError):
            PlannedTeacherMix(mix.base, mix.teacher, plan, source_hashes=plan['source_hashes'],
                eligible_teacher_indices=[0, 1, 2, 3])

    def test_type_bounds_and_exact_multiplicity(self):
        mix, plan, kwargs, _ = fixture()
        edits = [
            ('base_count', True), ('teacher_count', 20.), ('schema', V1),
            ('extra_teacher_indices', (0, 0, 1, 2, 3)),
            ('extra_teacher_indices', [False, 0, 1, 2, 3]),
            ('extra_teacher_indices', [0., 0, 1, 2, 3]),
            ('extra_teacher_indices', [-1, 0, 1, 2, 3]),
            ('extra_teacher_indices', [0, 0, 1, 2, 20]),
            ('extra_teacher_indices', [0, 1, 0, 2, 3]),
            ('extra_teacher_indices', [0, 0, 0, 2, 3]),
            ('extra_teacher_indices', [0, 1, 1, 2, 3]),
            ('extra_teacher_indices', [0, 0, 1, 2]),
        ]
        for key, value in edits:
            bad = copy.deepcopy(plan); bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                ScheduledTeacherMix(mix.base, mix.teacher, bad, **kwargs)
        bad = dict(plan, extra_repeats=1)
        with self.assertRaises(ValueError): ScheduledTeacherMix(mix.base, mix.teacher, bad, **kwargs)

    def test_source_pins_and_group_admission(self):
        mix, plan, kwargs, _ = fixture()
        for which in ('source_hashes', 'selection_source_hashes'):
            bad = copy.deepcopy(plan); bad[which][next(iter(bad[which]))] = 'c'*64
            with self.subTest(which=which), self.assertRaises(ValueError):
                ScheduledTeacherMix(mix.base, mix.teacher, bad, **kwargs)
        for groups in (
            dict(kwargs['sampling_groups'], successful_stt_anchor=[2]),
            dict(kwargs['sampling_groups'], hard_stt_early=[0, 0]),
            dict(kwargs['sampling_groups'], hard_stt_early=[True]),
            dict(kwargs['sampling_groups'], successful_stt_anchor=[4]),
        ):
            with self.assertRaises(ValueError):
                ScheduledTeacherMix(mix.base, mix.teacher, plan, **dict(kwargs, sampling_groups=groups))

    def test_actual_eight_rank_loader_matches_simulation_and_groups(self):
        for base in (101, 103, 117):
            mix, plan, kwargs, meta = fixture(base)
            counters = []
            for rank in range(8):
                sampler = DistributedSampler(mix, num_replicas=8, rank=rank,
                    seed=42, shuffle=True, drop_last=True)
                sampler.set_epoch(1)
                counter = RuntimeExposure(mix, **meta)
                for batch in DataLoader(ExposureTaggedData(mix), sampler=sampler, batch_size=2, drop_last=True):
                    clean, ids = strip_exposure_tag(batch)
                    self.assertNotIn('_wa_exposure_index', clean)
                    counter.consume(ids)
                counters.append(counter)
            state = counters[0].merge_states([counter.state() for counter in counters])
            report = counters[0].finalize(state=state)
            _, teacher = counters[0].source_counts(state)
            extra_counts = np.zeros(len(mix.teacher), dtype=np.int64)
            for p, count in enumerate(state['position_counts'].tolist()):
                if count and p >= len(mix.base)+len(mix.teacher):
                    extra_counts[mix.locate(p)[1]] += 1
            self.assertEqual(report['schedule_schema'], SCHEMA)
            for group, indices in kwargs['sampling_groups'].items():
                self.assertEqual(report['sampling_group_exposures'][group], int(teacher[indices].sum()))
                self.assertEqual(report['sampling_group_extra_exposures'][group], int(extra_counts[indices].sum()))
            self.assertEqual(report['hard_teacher_exposures'], int(teacher[:2].sum()))
            self.assertEqual(report['sampling_group_unique_windows']['successful_stt_anchor'], 1)
            self.assertEqual(report['actual_total'], simulate_exposure(mix)[0]['actual_total'])

    def test_rank_tail_and_metadata_do_not_relax_checks(self):
        mix, _, _, meta = fixture()
        counter = RuntimeExposure(mix, **meta)
        with self.assertRaises(ValueError): counter.finalize()
        counter.consume(torch.tensor([123, 124]))  # Two distinct positions, same teacher window.
        self.assertEqual(counter.source_counts()[1][0].item(), 2)
        with self.assertRaises(ValueError): counter.consume(torch.tensor([124]))
        with self.assertRaises(ValueError): counter.merge_states([counter.state(), counter.state()])
        bad = copy.deepcopy(meta); bad['hard'][2] = True
        with self.assertRaises(ValueError): RuntimeExposure(mix, **bad)
        bad = copy.deepcopy(meta); bad['episodes'][2] = dict(episode_uid='dt:scene/2', task='dt', teacher='lightnav')
        with self.assertRaises(ValueError): RuntimeExposure(mix, **bad)

    def test_diagnostic_positions_cover_all_groups_and_both_early_slots(self):
        mix, _, _, meta = fixture()
        windows = dict(episode_index=meta['episode_index'], hard=meta['hard'])
        positions = runtime.diagnostic_positions_v2(mix, meta['episodes'], windows, 16)
        self.assertEqual(len(set(positions)), 16)
        self.assertTrue({123, 124, 125, 126, 127}.issubset(positions))
        self.assertTrue({0, 102}.issubset(positions))
        for count in (True, 8, 13, 16.):
            with self.assertRaises(ValueError):
                runtime.diagnostic_positions_v2(mix, meta['episodes'], windows, count)

    def test_base_source_complete_split_and_identity_fail_closed(self):
        class Base(SimpleNamespace):
            def __len__(self): return self.length
        valid = np.arange(4, dtype=np.int64)
        stream = io.BytesIO(); np.save(stream, valid)
        payload = stream.getvalue()
        audit = dict(cache_complete_sha256='a'*64)
        for change in ('valid', 'audit', 'cache', 'rows', 'dtype', 'length', 'split', 'redirect'):
            base = Base(cache=Path('/fixture/base'), rows=valid.copy(), length=4,
                part='train', data_root=None, source_prefix=None)
            if change == 'rows': base.rows[-1] += 1
            elif change == 'dtype': base.rows = base.rows.astype(float)
            elif change == 'length': base.length = 3
            elif change == 'split': base.part = 'heldout'
            elif change == 'redirect': base.data_root = Path('/foreign')
            def digest(path):
                if Path(path).name == 'audit.json':
                    return 'f'*64 if change == 'audit' else runtime.recipe.old.PINNED['base_index_audit']
                return 'f'*64 if change == 'cache' else 'a'*64
            with self.subTest(change=change), ExitStack() as stack:
                stack.enter_context(patch.object(runtime, 'sha', side_effect=digest))
                stack.enter_context(patch.object(Path, 'read_text', return_value=json.dumps(audit)))
                stack.enter_context(patch.object(Path, 'open', side_effect=lambda *a, **k: io.BytesIO(payload)))
                if change == 'valid': runtime.verify_base_source(base, '/fixture/index')
                else:
                    with self.assertRaises(ValueError): runtime.verify_base_source(base, '/fixture/index')

    def test_parent_selection_and_actual_recipe_are_separate(self):
        args = SimpleNamespace(resume=runtime.PARENT_PATH, resume_sha256=runtime.PARENT_SHA,
            completed_epochs=1, epochs=1, seed=42, batch_size=2, accumulation=2,
            history_repeat_probability=.25, dual_teacher_repeats=1, evaluation_set_adaptation=True,
            kind='jepa', world_weight=.1, recovery_cache=None, recovery_index=None, diagnostic=False)
        report = dict(training_recipe=copy.deepcopy(runtime.TRAINING_RECIPE))
        runtime.validate_training_recipe(report, args, world=8)
        for key, value in [('resume_sha256', runtime.recipe.CHECKPOINT_SHA),
                ('resume', '/candidate61609/checkpoint.pt'), ('epochs', 2), ('completed_epochs', 2),
                ('batch_size', True), ('history_repeat_probability', 0.), ('world_weight', .2)]:
            bad = copy.copy(args); setattr(bad, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                runtime.validate_training_recipe(report, bad, world=8)
        with self.assertRaises(ValueError): runtime.validate_training_recipe(report, args, world=1)
        args.diagnostic = True
        runtime.validate_training_recipe(report, args, world=1)
        report['training_recipe']['model_and_optimizer'] = 1
        with self.assertRaises(ValueError): runtime.validate_training_recipe(report, args, world=1)


class RuntimeV2AdmissionTest(unittest.TestCase):
    def setup_fixture(self):
        mix, plan, kwargs, meta = fixture()
        root = Path('/fixture/project'); directory = root/'artifacts/new_candidate'
        paths = runtime.selection_paths(root)
        plan['source_hashes'] = {k: runtime.recipe.old.PINNED[k] for k in SOURCE_KEYS}
        plan['selection_source_hashes'] = dict(runtime.recipe.SOURCE_SHA)
        kwargs.update(source_hashes=plan['source_hashes'], selection_source_hashes=plan['selection_source_hashes'])
        mix = ScheduledTeacherMix(mix.base, mix.teacher, plan, **kwargs)
        records = meta['episodes']
        windows = dict(dataset_index=np.arange(20), episode_index=meta['episode_index'],
            raw_row=np.arange(20), current_observation_index=np.arange(20)+4,
            age_s=np.where(meta['early'], 1., 3.), hard=meta['hard'], early=meta['early'])
        allocation = dict(anchor_details=[dict(index=3)], regression_episode_indices=[2])
        outcome = dict(audit_status='PASS', episodes=4215,
            candidate_checkpoint_sha256=runtime.recipe.CHECKPOINT_SHA, candidate_checkpoint_step=59716,
            groups=dict(regressions=dict(by_task=dict(stt=dict(keys=['scene/2'])))))
        replay = dict(status='FIXED88_CANDIDATE_LABEL_FIT_ONLY',
            candidate=dict(sha256=runtime.recipe.CHECKPOINT_SHA, step=59716))
        report = dict(status='CANDIDATE_ONLY_NOT_TRAINING_RELEASE', candidate_kind=runtime.recipe.KIND,
            runtime_integration_required=True, experiment=runtime.recipe.old.EXPERIMENT,
            checkpoint_sha256=runtime.recipe.old.CHECKPOINT_SHA, checkpoint_step=59065,
            candidate_checkpoint_sha256=runtime.recipe.CHECKPOINT_SHA, candidate_checkpoint_step=59716,
            budget=copy.deepcopy(runtime.BUDGET), training_recipe=copy.deepcopy(runtime.TRAINING_RECIPE),
            tool_sha256='d'*64, selection_source_hashes=dict(runtime.recipe.SOURCE_SHA),
            source_hashes={str(path): runtime.recipe.SOURCE_SHA[key] for key, path in paths.items()},
            artifacts={name: 'e'*64 for name in ('plan.json', 'episodes.json', 'teacher_windows.npz',
                'exposure.json', 'simulated_counts.npz')}, plan_canonical_sha256=canonical_sha(plan),
            anchor_selection=allocation['anchor_details'], regression_episode_uids=[records[2]['episode_uid']],
            all12_regression_keys=['scene/2'])
        exposure, counts = simulate_exposure(mix)
        extra = np.bincount(plan['extra_teacher_indices'], minlength=20)
        ranks = np.zeros((8, 20), dtype=np.int64)
        base_counts = np.zeros(103, dtype=np.int64)
        for rank in range(8):
            sampler = DistributedSampler(mix, num_replicas=8, rank=rank, seed=42,
                shuffle=True, drop_last=True)
            sampler.set_epoch(1)
            ids = list(sampler); ids = ids[:len(ids)//2*2]
            for p in ids:
                source, index = mix.locate(p)
                if source == 'teacher': ranks[rank,index] += 1
                else: base_counts[index] += 1
        arrays = dict(planned_teacher_counts=1+extra, simulated_teacher_counts=counts,
            extra_counts=extra, simulated_base_counts=base_counts, rank_teacher_counts=ranks)
        teacher = SimpleNamespace(cache=Path('/fixture/teacher'))
        original_paths = dict(cache_complete=teacher.cache/'complete.json',
            teacher_index=teacher.cache/'index/train_valid.npy', base_index=Path('/fixture/index/train_valid.npy'),
            student_rows=Path('/fixture/baseline.jsonl'), teacher_selections=Path('/fixture/teachers.jsonl'))
        report['source_hashes'].update({str(p): runtime.recipe.old.PINNED[k] for k,p in original_paths.items()})
        texts = {str(directory/'plan.json'): json.dumps(plan),
            str(directory/'episodes.json'): json.dumps(records),
            str(directory/'exposure.json'): json.dumps(exposure),
            str(paths['candidate_outcomes']): json.dumps(outcome),
            str(paths['fixed88_replay']): json.dumps(replay)}
        hashes = {str(path): runtime.recipe.SOURCE_SHA[k] for k,path in paths.items()}
        hashes.update({str(directory/name): value for name,value in report['artifacts'].items()})
        hashes[str(Path(runtime.recipe.__file__))] = 'd'*64
        return locals()

    def run_fixture(self, f):
        with ExitStack() as stack:
            stack.enter_context(patch.object(runtime, 'BUDGET', dict(runtime.BUDGET, base_once=103, teacher_once=20)))
            report = copy.deepcopy(f['report']); report['budget'] = dict(runtime.BUDGET)
            stack.enter_context(patch.object(runtime, 'sha', side_effect=lambda p: f['hashes'][str(Path(p))]))
            stack.enter_context(patch.object(Path, 'read_text', lambda p,*a,**k: f['texts'][str(p)]))
            stack.enter_context(patch('wa.wm.teacher_plan_runtime.load_candidate',
                return_value=(f['mix'], f['records'], f['windows'], {})))
            stack.enter_context(patch('wa.wm.teacher_plan_runtime.verify_metadata'))
            stack.enter_context(patch.object(runtime, 'verify_base_source'))
            stack.enter_context(patch.object(runtime.recipe, 'read_rows', return_value=[]))
            stack.enter_context(patch.object(runtime.recipe, 'validate_student_rows'))
            stack.enter_context(patch.object(runtime.recipe, 'unique_rows', return_value={}))
            stack.enter_context(patch.object(runtime, 'derive_schedule', return_value=(
                f['plan']['extra_teacher_indices'], f['kwargs']['sampling_groups'], f['allocation'])))
            stack.enter_context(patch.object(runtime.np, 'load', side_effect=lambda p,**k:
                FakeNPZ(f['windows'] if str(p).endswith('teacher_windows.npz') else f['arrays'])))
            class TeacherList(list): pass
            teacher = TeacherList(f['mix'].teacher); teacher.cache = f['teacher'].cache
            return runtime.load_v2(f['directory'], report, f['mix'].base, teacher,
                base_index='/fixture/index', student_rows='/fixture/baseline.jsonl',
                teacher_selections='/fixture/teachers.jsonl', project_root=f['root'])

    def test_complete_mocked_admission(self):
        result = self.run_fixture(self.setup_fixture())
        self.assertEqual(result[0].extra_positions, (0, 0, 1, 2, 3))

    def test_identity_selection_artifact_and_multiplicity_rejections(self):
        for mode in ('parent', 'selection_identity', 'old_identity', 'tool', 'source', 'inventory',
                     'canonical', 'stored_selection', 'counts', 'multiset',
                     'array_type', 'array_shape', 'rank_counts', 'original_source'):
            f = self.setup_fixture()
            if mode == 'parent': f['report']['training_recipe']['parent_sha256'] = runtime.recipe.CHECKPOINT_SHA
            elif mode == 'selection_identity': f['report']['candidate_checkpoint_step'] = 59065
            elif mode == 'old_identity': f['report']['checkpoint_sha256'] = runtime.recipe.CHECKPOINT_SHA
            elif mode == 'tool': f['report']['tool_sha256'] = 'f'*64
            elif mode == 'source': f['hashes'][str(f['paths']['candidate_rows'])] = 'f'*64
            elif mode == 'inventory': f['report']['artifacts'].pop('simulated_counts.npz')
            elif mode == 'canonical': f['report']['plan_canonical_sha256'] = 'f'*64
            elif mode == 'stored_selection': f['report']['regression_episode_uids'] = []
            elif mode == 'counts': f['arrays']['simulated_teacher_counts'][0] += 1
            elif mode == 'array_type': f['arrays']['extra_counts'] = f['arrays']['extra_counts'].astype(float)
            elif mode == 'array_shape': f['arrays']['rank_teacher_counts'] = f['arrays']['rank_teacher_counts'][:7]
            elif mode == 'rank_counts': f['arrays']['rank_teacher_counts'][0, 0] += 1
            elif mode == 'original_source': f['report']['source_hashes'].pop('/fixture/index/train_valid.npy')
            elif mode == 'multiset':
                changed = copy.deepcopy(f['plan']); changed['extra_teacher_indices'] = [0, 1, 1, 2, 3]
                f['texts'][str(f['directory']/'plan.json')] = json.dumps(changed)
                f['report']['plan_canonical_sha256'] = canonical_sha(changed)
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.run_fixture(f)


if __name__ == '__main__': unittest.main()
