import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from wa.tools import replay_teacher_group_fit as replay


def reference_fixture():
    selected, rows = [], []
    for group, n in replay.original.LIMITS.items():
        task = 'dt' if group.startswith('dt_') else 'at' if group.startswith('at_') else 'stt'
        for _ in range(n):
            i = len(selected)
            selected.append(dict(group=group, index=i, raw_row=i, episode=i,
                                 episode_uid=f'{task}:scene/{i}', teacher='oracle', age_s=1.0))
            for repeat in (False, True):
                rows.append(dict(selection_index=i, group=group, repeat_history=repeat,
                                 **{k: 0.0 for k in replay.METRICS},
                                 label_first=[0., 0., 0., 1.], pred_first=[0., 0., 0., 1.]))
    return dict(status='BOUNDED_PAIRED_LABEL_FIT_ONLY', tool_sha256=replay.ORIGINAL_TOOL_SHA,
                selected=selected, models={name: dict(sha256=d[1], step=d[2], records=copy.deepcopy(rows),
                    summaries=replay.summaries(rows)) for name, d in replay.original.MODELS.items()},
                source_hashes={'/a/source': 'a'*64})


class ReplayFitTest(unittest.TestCase):
    def test_exact_reference_and_pairs(self):
        ref = reference_fixture()
        pairs = replay.validate_reference(ref)
        self.assertEqual(len(pairs), 176)
        self.assertEqual(len({(i, r) for i, r in pairs}), 176)

    def test_reference_rejects_identity_group_and_metric_changes(self):
        changes = [
            lambda r: r.update(status='RUNNING'),
            lambda r: r.update(tool_sha256='0'*64),
            lambda r: r['selected'][1].update(index=0),
            lambda r: r['selected'][0].update(episode_uid='dt:scene/0'),
            lambda r: r['selected'][0].update(age_s=2.01),
            lambda r: r['selected'][0].update(teacher='fallback'),
            lambda r: r['models']['student61377'].update(step=59716),
            lambda r: r['models']['student61377']['records'].pop(),
            lambda r: r['models']['student61377']['records'][1].update(repeat_history=False),
            lambda r: r['models']['student61377']['records'][0].update(ADE_m=float('nan')),
            lambda r: r['models']['student61377']['records'][0].update(label_first=[0]*3),
            lambda r: r['models']['student61377']['summaries']['hard_stt_other']['False'].update(ADE_m=1),
        ]
        for change in changes:
            with self.subTest(change=change):
                ref = reference_fixture()
                change(ref)
                with self.assertRaises(ValueError):
                    replay.validate_reference(ref)

    def test_fixed_reference_hash_enforced_before_parse(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp)/'ref.json'
            p.write_text(json.dumps(reference_fixture()))
            with self.assertRaisesRegex(ValueError, 'changed file'):
                replay.read_reference(p, {})

    def test_only_exact_old_checkpoint_hashes_skipped(self):
        ref = reference_fixture()
        jobroot = Path('/jobs')
        ref['source_hashes'].update({str(jobroot/d[0]): d[1] for d in replay.original.MODELS.values()})
        with patch.object(replay, 'verify') as verify:
            skipped = replay.verify_reference_sources(ref, jobroot, {})
        self.assertEqual(len(skipped), 2)
        self.assertEqual([c.args[0] for c in verify.call_args_list], ['/a/source'])
        with self.assertRaises(ValueError):
            replay.verify_reference_sources(ref, Path('/wrong-jobs'), {})

    def test_output_existing_and_dangling_symlink(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp)/'output.json'
            replay.write_report(p, {'ok': True})
            with self.assertRaises(ValueError):
                replay.write_report(p, {'ok': False})
            link = Path(tmp)/'dangling.json'
            link.symlink_to(Path(tmp)/'missing')
            with self.assertRaises(ValueError):
                replay.write_report(link, {'ok': False})
            self.assertFalse((Path(tmp)/'missing').exists())

    def fake_data(self, tmp):
        observations = Path(tmp)/'observations.json'
        observations.write_text(json.dumps([{'timestamp_s': 0.0}, {'timestamp_s': 1.0}]))
        label = torch.zeros(7, 4)
        label[:, 3] = 1
        sample = dict(rgb=torch.zeros(4, 3, 2, 2), template=torch.zeros(3, 2, 2),
                      polar=torch.ones(2), times=torch.zeros(4), pose=label)
        class Data:
            rows = np.array([0])
            episode = np.array([0])
            episodes = [dict(task='stt', episode_uid='stt:scene/0', teacher='oracle', root=tmp)]
            history = np.array([[0, 0, 0, 1]])
            pose_labels = label.numpy()[None]
            source_records = {0: {'observations.json': replay.sha(observations)}}
            def __getitem__(self, i):
                return sample
        return Data(), sample

    def test_fixed_sample_no_reselection_metadata_excluded(self):
        with TemporaryDirectory() as tmp:
            ref = reference_fixture()
            data, sample = self.fake_data(tmp)
            sample['future'] = torch.ones(3, 2, 2)
            sample['task_id'] = torch.tensor(9)
            baseline = replay.validate_reference(ref)
            with patch.object(replay.original, 'select_episodes', side_effect=AssertionError('no reselection')), \
                 patch.object(replay.original, 'choose_early_index', side_effect=AssertionError('no reselection')):
                got = replay.fixed_sample(data, ref['selected'], baseline, 0, {})
            self.assertEqual(set(got), set(replay.INPUT_KEYS) | {'pose'})

    def test_fixed_sample_refuses_remap_age_and_label(self):
        for mutation in ('index', 'identity', 'teacher', 'age', 'label'):
            with self.subTest(mutation=mutation), TemporaryDirectory() as tmp:
                ref = reference_fixture()
                data, sample = self.fake_data(tmp)
                baseline = replay.validate_reference(ref)
                if mutation == 'index':
                    data.rows = np.array([1])
                elif mutation == 'identity':
                    data.episodes[0]['episode_uid'] = 'stt:other/0'
                elif mutation == 'teacher':
                    data.episodes[0]['teacher'] = 'lightnav'
                elif mutation == 'age':
                    ref['selected'][0]['age_s'] = .9
                else:
                    sample['pose'][0, 0] = 1
                with self.assertRaises(ValueError):
                    replay.fixed_sample(data, ref['selected'], baseline, 0, {})

    def test_candidate_gate_refuses_old_incomplete_and_wrong_identities(self):
        checkpoint = '/jobs/new/checkpoint.pt'
        audit = dict(status='TRAINING_ARTIFACT_AUDIT_PASS_OFFLINE_ONLY',
                     checkpoint=checkpoint, checkpoint_sha256='c'*64, final_step=59716,
                     completed_epochs=2, source_hashes={checkpoint: 'c'*64})
        replay.validate_candidate_gate(audit, checkpoint, 'c'*64, 59716)
        for mutation in ('status', 'step', 'path', 'epochs', 'old'):
            bad = copy.deepcopy(audit)
            sha = 'c'*64
            if mutation == 'status':
                bad['status'] = 'RUNNING'
            elif mutation == 'step':
                bad['final_step'] = 59065
            elif mutation == 'path':
                bad['checkpoint'] = '/wrong/checkpoint.pt'
            elif mutation == 'epochs':
                bad['completed_epochs'] = 3
            else:
                sha = replay.original.MODELS['student61377'][1]
                bad['checkpoint_sha256'] = sha
                bad['source_hashes'][checkpoint] = sha
            with self.assertRaises(ValueError):
                replay.validate_candidate_gate(bad, checkpoint, sha, 59716)

    def test_load_candidate_once(self):
        from unittest.mock import Mock
        audit = dict(parent_checkpoint='/parent', parent_sha256='a'*64)
        ck = dict(step=59716, kind='jepa', contract=replay.CONTRACT, completed_epochs=2,
                  **audit, model={'policy.weight': torch.tensor([1.])})
        model = SimpleNamespace(load_state_dict=lambda state, strict:
            SimpleNamespace(missing_keys=['encoder.weight'], unexpected_keys=[]))
        loader = Mock(return_value=ck)
        replay.load_candidate(model, '/new/checkpoint.pt', 59716, audit, loader=loader)
        loader.assert_called_once_with('/new/checkpoint.pt', map_location='cpu', weights_only=True, mmap=True)
        for bad_ck in (dict(ck, step=59065), dict(ck, model={'encoder.weight': torch.tensor(1.)})):
            with self.assertRaises(ValueError):
                replay.load_candidate(model, '/new', 59716, audit, loader=lambda *a, **kw: bad_ck)
        bad_model = SimpleNamespace(load_state_dict=lambda state, strict:
            SimpleNamespace(missing_keys=['policy.weight'], unexpected_keys=[]))
        with self.assertRaises(ValueError):
            replay.load_candidate(bad_model, '/new', 59716, audit, loader=lambda *a, **kw: ck)

    def test_cpu_prediction_176pairs_full_trajectories_and_delta(self):
        ref = reference_fixture()
        baseline = replay.validate_reference(ref)
        samples = []
        for _ in ref['selected']:
            pose = torch.zeros(7, 4)
            pose[:, 3] = 1
            samples.append(dict(rgb=torch.arange(48).float().reshape(4, 3, 2, 2),
                template=torch.zeros(3, 2, 2), polar=torch.ones(2),
                times=torch.tensor([-1.5, -1., -.5, 0.]), pose=pose))
        test = self
        class Predictor:
            calls = 0
            def predict(self, batch, modes, noise):
                self.calls += 1
                test.assertEqual(set(batch), set(replay.INPUT_KEYS))
                test.assertTrue((modes == 2).all() and (noise == 0).all())
                if self.calls % 2 == 0:
                    test.assertTrue((batch['times'] == 0).all())
                    test.assertTrue(torch.equal(batch['rgb'][:, 0], batch['rgb'][:, -1]))
                p = torch.zeros_like(noise)
                p[..., 3] = 1
                return p, None
        model = Predictor()
        records = replay.predict_records(model, samples, ref['selected'], device='cpu')
        self.assertEqual(model.calls, 88)
        self.assertEqual(len(records), 176)
        self.assertEqual(np.asarray(records[0]['prediction_7x4']).shape, (7, 4))
        self.assertEqual(np.asarray(records[0]['label_7x4']).shape, (7, 4))
        rows, groups = replay.paired_deltas(records, ref['selected'], baseline)
        self.assertEqual(len(rows), 176)
        self.assertEqual(groups['hard_stt_collision']['False']['ADE_m']['tied'], 24)
        self.assertNotEqual(records[0]['input_tensor_sha256']['rgb'],
                            records[2]['input_tensor_sha256']['rgb'])


if __name__ == '__main__':
    unittest.main()
