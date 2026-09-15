"""Targeted tamper/causality tests; synthetic CPU replay only, no Habitat/GPU."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collection_contract as contract
import semantic_labels as labels
import verify_collected_sidecars as verifier
from collect_perception_sidecar import collect_episode

HELPERS = SimpleNamespace(contract=contract, labels=labels)


def save(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True) + '\n', encoding='utf-8')


class FakeReplay:
    def __init__(self, frames): self.frames = frames
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def reset(self): pass
    def capture(self, step): return copy.deepcopy(self.frames[step])
    def step(self, action): pass


class RawSidecarVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root/'output'
        self.output.mkdir()
        init = dict(source='first_frame_panoptic_mask', bbox_xyxy=[0, 0, 1, 1],
                    bbox_xyxy_norm=[0., 0., .25, 1/3], environment_step=0, used_frames=[0])
        self.frames = []
        for step in range(3):
            raw = np.zeros((3, 4, 1), dtype=np.uint32)
            if step != 1: raw[:2, :2, 0] = 100
            raw[2, 3, 0] = 200
            camera = np.eye(4); camera[0, 3] = float(step)
            self.frames.append(dict(rgb=np.arange(36, dtype=np.uint8).reshape(3, 4, 3) + step * 10,
                panoptic=raw, assigned_ids={0: 100, 2: 200}, world_time_s=step*.1,
                world_time_before_render_s=step*.1, camera_before_render=camera, camera_after_render=camera,
                gt_distance_m=2.+step, terminal=step==2, initialization=init if step==0 else None))
        actions = [[.1, .2, -.3], [0., -.1, .4]]
        source = {'steps': []}
        for index, action in enumerate(actions):
            source['steps'].append(dict(step=index+1,
                policy=dict(action=dict(zip(('forward', 'lateral', 'yaw'), action)),
                    rgb_mean_0_255=float(self.frames[index]['rgb'].mean()),
                    rgb_std_0_255=float(self.frames[index]['rgb'].std()),
                    rgb_temporal_absdiff_0_255=None if index==0 else 10.),
                evaluation_only_after_action=dict(gt_distance_m=self.frames[index+1]['gt_distance_m'],
                    gt_visible=index==1, render_audit=dict(
                        camera_transform_before_action=self.frames[index]['camera_before_render'].tolist(),
                        camera_transform_after_forced_render=self.frames[index+1]['camera_after_render'].tolist()))))
        self.source_path = self.root/'source.json'; save(self.source_path, source)
        sample_path = self.root/'sample.json'; save(sample_path, {'synthetic_original_sample': True})
        refs = []
        for step in range(2):
            path = self.root/f'original_{step}.png'; Image.fromarray(self.frames[step]['rgb']).save(path)
            refs.append(dict(contract.artifact(path), environment_step=step, world_time_s=step*.1,
                             rgb_array_sha256=labels.array_sha256(self.frames[step]['rgb'])))
        self.entry = dict(run_id='synthetic', source=contract.artifact(self.source_path),
            saved_actions_sha256=contract.canonical_sha(actions), observation_count=3, action_count=2,
            assigned_humanoid_semantic_ids={'0':100, '2':200}, initialization=init,
            output_dir=str(self.output), prefix_samples=[dict(sample=contract.artifact(sample_path),
                initial_rgb={k:v for k,v in refs[0].items() if k not in ('environment_step', 'world_time_s')}, prefix_rgb=refs)])
        self.plan = dict(plan_sha256='synthetic-plan', replay_tolerance=1e-4,
                         world_time_tolerance_s=1e-8, rgb_statistics_tolerance_0_255=1e-5)
        result = collect_episode(self.entry, self.plan, self.output, lambda *args: FakeReplay(self.frames))
        self.assertEqual(result['status'], 'sidecar_collected_pending_independent_admission')
        save(self.output/'launch_contract.json', dict(entry=self.entry, plan_sha256=self.plan['plan_sha256'],
                                                     formal_training_eligible=False, optimizer_input_allowed=False))
        self.status = dict(run_id='synthetic', output_dir=str(self.output), exit_code=0,
            status='sidecar_collected_pending_independent_admission', worker_result=result,
            worker_result_sha256=verifier.sha(self.output/'worker_result.json'))
        # Fixtures emulate the frozen Linux collector's relative POSIX paths
        # even when these CPU tests run on the local Windows workstation.
        rows = self.rows()
        for row in rows:
            for key in ('rgb_file', 'raw_panoptic_file'):
                row[key]['path'] = row[key]['path'].replace('\\', '/')
        self.rebind_reports(rows)

    def rows(self):
        return [json.loads(line) for line in (self.output/'labels.jsonl').read_text().splitlines()]

    def rebind_reports(self, rows):
        # Forged worker reports still say every check passed; the verifier must
        # inspect raw evidence after these self-consistent summary hashes change.
        (self.output/'labels.jsonl').write_text(''.join(json.dumps(row, sort_keys=True)+'\n' for row in rows))
        worker = self.status['worker_result']
        worker['labels_sha256'] = verifier.sha(self.output/'labels.jsonl')
        save(self.output/'worker_result.json', worker)
        save(self.output/'worker_progress.json', worker)
        self.status['worker_result_sha256'] = verifier.sha(self.output/'worker_result.json')

    def verify(self):
        return verifier.verify_source(self.entry, self.plan, self.status, HELPERS)

    def test_raw_recomputation_passes_and_limits_scope(self):
        report = self.verify()
        self.assertEqual(report['observations_verified'], 3)
        self.assertEqual(report['saved_frame_files_verified'], 6)
        self.assertEqual(report['original_prefix_unique_observations_verified'], 2)
        self.assertEqual(report['terminal_observations_without_source_policy_rgb_statistics'], 1)

    def test_forged_semantic_label_rejected_even_with_rebound_worker_hashes(self):
        rows = self.rows(); rows[1]['target']['visible'] = True
        self.rebind_reports(rows)
        with self.assertRaisesRegex(ValueError, 'raw-derived'): self.verify()

    def test_tampered_rgb_file_fails_file_hash(self):
        path = self.output/'observations/rgb_0001.png'
        with path.open('ab') as stream: stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'file hash mismatch'): self.verify()

    def test_same_statistics_different_prefix_pixels_cannot_pass(self):
        rows = self.rows(); rgb = self.frames[0]['rgb'].copy()
        rgb[0, 0, 0], rgb[0, 0, 1] = rgb[0, 0, 1], rgb[0, 0, 0]
        path = self.output/rows[0]['rgb_file']['path']; Image.fromarray(rgb).save(path)
        rows[0]['rgb_file']['sha256'] = verifier.sha(path)
        rows[0]['rgb_array_sha256'] = labels.array_sha256(rgb)
        self.rebind_reports(rows)
        self.assertEqual(float(rgb.mean()), float(self.frames[0]['rgb'].mean()))
        self.assertEqual(float(rgb.std()), float(self.frames[0]['rgb'].std()))
        with self.assertRaisesRegex(ValueError, 'independent recomputation'): self.verify()

    def test_raw_panoptic_change_not_hidden_by_rebound_file_and_array_hash(self):
        rows = self.rows(); path = self.output/rows[1]['raw_panoptic_file']['path']
        raw = np.load(path); raw[0, 0, 0] = 100; np.save(path, raw, allow_pickle=False)
        rows[1]['raw_panoptic_file']['sha256'] = verifier.sha(path)
        rows[1]['raw_panoptic_file']['raw_array_sha256'] = labels.array_sha256(raw)
        rows[1]['panoptic_array_sha256'] = labels.array_sha256(raw[..., 0])
        self.rebind_reports(rows)
        with self.assertRaisesRegex(ValueError, 'raw-derived'): self.verify()

    def test_raw_shape_identity_and_squeezed_hash_are_distinct(self):
        rows = self.rows(); rows[0]['raw_panoptic_file']['raw_array_sha256'] = rows[0]['panoptic_array_sha256']
        self.rebind_reports(rows)
        with self.assertRaisesRegex(ValueError, 'raw panoptic array identity'): self.verify()

    def test_causal_policy_index_shift_is_rejected(self):
        rows = self.rows(); rows[1]['policy_call_index'] = 2
        self.rebind_reports(rows)
        with self.assertRaisesRegex(ValueError, 'causal index'): self.verify()

    def test_terminal_without_next_policy_is_not_relabelled_nonterminal(self):
        rows = self.rows(); rows[2]['source_audit']['capture_evidence']['terminal_observation'] = False
        self.rebind_reports(rows)
        with self.assertRaisesRegex(ValueError, 'natural terminal'): self.verify()

    def test_nonmonotonic_measured_clock_rejected_even_if_report_says_passed(self):
        rows = self.rows(); capture = rows[1]['source_audit']['capture_evidence']
        capture['world_time_before_render_s'] = capture['world_time_after_render_s'] = 0.
        rows[1]['world_time_s'] = 0.; self.rebind_reports(rows)
        with self.assertRaisesRegex(ValueError, 'monotonic time'): self.verify()

    def test_camera_stability_recomputed_from_original_matrices(self):
        rows = self.rows(); rows[0]['source_audit']['capture_evidence']['camera_before_render'][0][3] = 5.
        self.rebind_reports(rows)
        with self.assertRaisesRegex(ValueError, 'camera moved'): self.verify()

    def test_observation_filename_swap_is_rejected(self):
        rows = self.rows(); rows[0]['rgb_file']['path'] = 'observations/rgb_0001.png'
        self.rebind_reports(rows)
        with self.assertRaisesRegex(ValueError, 'filename or observation index'): self.verify()

    def test_unavailable_motion_supervision_cannot_be_granted(self):
        rows = self.rows(); rows[0]['motion_permission_label_available'] = True
        self.rebind_reports(rows)
        with self.assertRaisesRegex(ValueError, 'raw-derived'): self.verify()

    def test_original_source_mutation_is_rejected(self):
        with self.source_path.open('a') as stream: stream.write(' ')
        with self.assertRaisesRegex(ValueError, 'file hash mismatch'): self.verify()

    def test_cli_rejection_is_saved_without_changing_input(self):
        audit = self.root/'new_audit'
        original = self.source_path.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            code = verifier.main(['--plan', str(self.source_path), '--plan-file-sha256', '0'*64,
                '--batch-status', str(self.output/'batch_status.json'), '--admission-policy', str(self.root/'missing_policy'),
                '--output-dir', str(audit)])
        self.assertEqual(code, 2)
        result = verifier.load(audit/'verification.json')
        self.assertEqual(result['status'], verifier.REJECT)
        self.assertIs(result['optimizer_input_allowed'], False)
        self.assertEqual(original, self.source_path.read_bytes())


class BatchAdmissionTests(unittest.TestCase):
    def fixture(self):
        entries = []
        for index, (run, count) in enumerate(zip(verifier.RUNS, verifier.COUNTS)):
            entries.append(dict(run_id=run, observation_count=count, permanent_partition_role='train',
                prefix_samples=[dict(sample={'path':f'/synthetic/source{index}/sample{j}'}) for j in range(4 if index==0 else 5)]))
        plan = dict(entries=entries, plan_sha256='fixture', stage='fixture')
        batch = dict(plan_sha256='fixture', stage='fixture', entries=[{'run_id': run} for run in verifier.RUNS],
            status='sidecars_pending_independent_admission', expected_source_count=4, expected_observation_count=370,
            successful_source_count=4, observed_frames=370, all_four_sources_retained=True,
            formal_training_eligible=False, optimizer_input_allowed=False, test_locked_used=False)
        return plan, batch

    def test_duplicate_sample_alias_cannot_satisfy_nineteen(self):
        plan, _ = self.fixture()
        plan['entries'][1]['prefix_samples'][0] = plan['entries'][0]['prefix_samples'][0]
        with self.assertRaisesRegex(ValueError, '19 distinct'): verifier.validate_denominators(plan)

    def test_missing_terminal_count_cannot_satisfy_370(self):
        plan, _ = self.fixture(); plan['entries'][0]['observation_count'] -= 1
        with self.assertRaisesRegex(ValueError, '370 observations'): verifier.validate_denominators(plan)

    def test_validation_source_is_rejected(self):
        plan, _ = self.fixture(); plan['entries'][2]['permanent_partition_role'] = 'val'
        with self.assertRaisesRegex(ValueError, 'validation source'): verifier.validate_denominators(plan)

    def test_partial_success_retains_four_results_and_admits_zero(self):
        plan, batch = self.fixture()
        def check(entry, *args):
            if entry['run_id'] == verifier.RUNS[1]: raise ValueError('raw frame failed')
            return dict(run_id=entry['run_id'], status='independently_verified', observations_verified=entry['observation_count'])
        with patch.object(verifier, 'verify_source', side_effect=check):
            result = verifier.audit_outputs(plan, batch, HELPERS)
        self.assertEqual(result['status'], verifier.REJECT)
        self.assertEqual(len(result['sources']), 4)
        self.assertEqual(result['sources'][1]['status'], 'failed_retained')
        self.assertEqual(result['admitted_observation_count'], 0)

    def test_full_success_grants_only_pending_loader_scope(self):
        plan, batch = self.fixture()
        def check(entry, *args):
            return dict(run_id=entry['run_id'], status='independently_verified', observations_verified=entry['observation_count'])
        with patch.object(verifier, 'verify_source', side_effect=check):
            result = verifier.audit_outputs(plan, batch, HELPERS)
        self.assertEqual(result['status'], verifier.GRANT)
        self.assertEqual(result['admitted_observation_count'], 370)
        for key in ('optimizer_input_allowed', 'formal_training_eligible', 'product_acceptance_evidence'):
            self.assertIs(result[key], False)


if __name__ == '__main__':
    unittest.main()
