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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collect_perception_sidecar as collector
from collection_contract import canonical_sha, digest, load
from semantic_labels import array_sha256


class FakeReplay:
    calls = []
    mutation = None

    def __init__(self, entry, plan):
        self.entry = entry
        self.index = 0

    def __enter__(self): return self
    def __exit__(self, *args): pass
    def reset(self): self.index = 0
    def capture(self, step):
        assert step == self.index
        rgb = np.full((5, 6, 3), step, dtype=np.uint8)
        panoptic = np.zeros((5, 6, 1), dtype=np.uint32)
        if step != 1:
            panoptic[1:4, 2:5, 0] = 100
        result = dict(rgb=rgb, panoptic=panoptic, assigned_ids={0:100, 2:2002},
            world_time_before_render_s=step * .048, world_time_s=step * .048,
            camera_before_render=np.eye(4), camera_after_render=np.eye(4),
            gt_distance_m=1.0, terminal=step == 2,
            initialization=self.entry['initialization'] if step == 0 else None)
        if self.mutation is not None:
            self.mutation(result, step)
        return result
    def step(self, action):
        self.calls.append(list(action))
        self.index += 1


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root/'output'
        self.output.mkdir()
        self.actions = [[0.1, 0., 0.2], [0., -0.1, -0.2]]
        source = {'steps': [dict(step=s+1,
            policy={'action':dict(zip(('forward','lateral','yaw'), action)),
                    'rgb_mean_0_255':float(s), 'rgb_std_0_255':0.,
                    'rgb_temporal_absdiff_0_255':None if s == 0 else 1.},
            evaluation_only_after_action={'gt_distance_m':1., 'gt_visible':s == 1,
                'render_audit':{'camera_transform_before_action':np.eye(4).tolist(),
                                'camera_transform_after_forced_render':np.eye(4).tolist()}})
            for s, action in enumerate(self.actions)]}
        self.source_path = self.root/'source.json'
        self.source_path.write_text(json.dumps(source))
        self.entry = dict(run_id='fixture', source={'path':str(self.source_path), 'sha256':digest(self.source_path)},
            saved_actions_sha256=canonical_sha(self.actions), observation_count=3,
            assigned_humanoid_semantic_ids={'0':100, '2':2002},
            initialization={'bbox_xyxy':[2,1,4,3]},
            prefix_samples=[{'sample':{'path':'fixed_sample'}, 'initial_rgb':{'rgb_array_sha256':
                array_sha256(np.zeros((5,6,3), dtype=np.uint8))},
                'prefix_rgb':[{'path':'prefix0', 'rgb_array_sha256':array_sha256(np.zeros((5,6,3), dtype=np.uint8)),
                              'world_time_s':0.},
                              {'path':'prefix1', 'rgb_array_sha256':array_sha256(np.ones((5,6,3), dtype=np.uint8)),
                              'world_time_s':.048}]}])
        self.plan = {'replay_tolerance':1e-4, 'world_time_tolerance_s':1e-8}
        FakeReplay.calls = []
        FakeReplay.mutation = None

    def tearDown(self): self.temp.cleanup()

    def collect(self):
        return collector.collect_episode(self.entry, self.plan, self.output, FakeReplay)

    def labels(self):
        return [json.loads(line) for line in (self.output/'labels.jsonl').read_text().splitlines()]

    def test_all_original_actions_and_reset_terminal_saved(self):
        result = self.collect()
        self.assertEqual(FakeReplay.calls, self.actions)
        self.assertEqual(result['observations_saved'], 3)
        self.assertEqual(result['completed_source_actions'], 2)
        self.assertEqual(result['status'], 'sidecar_collected_pending_independent_admission')
        rows = self.labels()
        self.assertEqual([r['environment_step'] for r in rows], [0,1,2])
        self.assertEqual([r['next_source_action_step'] for r in rows], [1,2,None])
        self.assertEqual([r['target']['visible'] for r in rows], [True,False,True])
        self.assertFalse(result['optimizer_input_allowed'])
        self.assertFalse(result['model_loaded'])

    def test_raw_panoptic_shape_preserved_and_hashes_distinguished(self):
        self.collect()
        row = self.labels()[0]
        raw = np.load(self.output/row['raw_panoptic_file']['path'], allow_pickle=False)
        self.assertEqual(raw.shape, (5,6,1))
        self.assertEqual(array_sha256(raw), row['raw_panoptic_file']['raw_array_sha256'])
        self.assertEqual(array_sha256(raw[...,0]), row['panoptic_array_sha256'])
        self.assertNotEqual(row['panoptic_array_sha256'], row['raw_panoptic_file']['raw_array_sha256'])

    def test_rgb_statistics_use_next_source_action_and_skip_terminal(self):
        self.collect()
        stats = [r['source_audit']['source_policy_input_rgb_statistics'] for r in self.labels()]
        self.assertEqual([r.get('source_action_step') for r in stats], [1,2,None])
        self.assertEqual(stats[0]['actual']['rgb_temporal_absdiff_0_255'], None)
        self.assertEqual(stats[1]['actual']['rgb_temporal_absdiff_0_255'], 1.)
        self.assertFalse(stats[2]['available'])

    def test_pixel_mismatch_retained_and_actions_not_replaced(self):
        FakeReplay.mutation = staticmethod(lambda capture, step: capture['rgb'].__setitem__((0,0,0), 99) if step == 1 else None)
        result = self.collect()
        self.assertEqual(result['status'], 'failed_alignment')
        self.assertEqual(result['observations_saved'], 3)
        self.assertEqual(FakeReplay.calls, self.actions)
        checks = self.labels()[1]['source_audit']['checks']
        self.assertTrue(any(c['kind'] == 'stored_prefix_rgb_and_worldtime' and not c['passed'] for c in checks))

    def test_worldtime_mismatch_not_hidden_by_matching_pixels(self):
        def mutate(capture, step):
            if step == 1:
                capture['world_time_s'] += .001
                capture['world_time_before_render_s'] += .001
        FakeReplay.mutation = staticmethod(mutate)
        result = self.collect()
        self.assertEqual(result['status'], 'failed_alignment')
        self.assertEqual(FakeReplay.calls, self.actions)

    def test_sensor_render_does_not_advance_simulator_clock(self):
        FakeReplay.mutation = staticmethod(lambda capture, step: capture.update(world_time_before_render_s=capture['world_time_s']+.1))
        result = self.collect()
        self.assertEqual(result['status'], 'failed_alignment')
        self.assertTrue(all(any(c['kind']=='single_render_worldtime_stable' and not c['passed']
                            for c in r['source_audit']['checks']) for r in self.labels()))

    def test_camera_mismatch_detected_without_changing_actions(self):
        def mutate(capture, step):
            capture['camera_after_render'][0,3] = .01
        FakeReplay.mutation = staticmethod(mutate)
        self.assertEqual(self.collect()['status'], 'failed_alignment')
        self.assertEqual(FakeReplay.calls, self.actions)

    def test_early_terminal_retains_partial_frames_and_fails(self):
        FakeReplay.mutation = staticmethod(lambda capture, step: capture.update(terminal=True) if step == 1 else None)
        with self.assertRaisesRegex(ValueError, 'early natural termination'):
            self.collect()
        self.assertEqual(FakeReplay.calls, self.actions[:1])
        self.assertEqual(len(self.labels()), 2)
        self.assertFalse((self.output/'worker_result.json').exists())

    def test_no_natural_terminal_is_failed_after_saving_every_frame(self):
        FakeReplay.mutation = staticmethod(lambda capture, step: capture.update(terminal=False))
        result = self.collect()
        self.assertEqual(result['status'], 'failed_alignment')
        self.assertFalse(result['natural_terminal_reproduced'])
        self.assertEqual(len(self.labels()), 3)

    def test_different_original_target_assignment_rejected(self):
        FakeReplay.mutation = staticmethod(lambda capture, step: capture.update(assigned_ids={0:101, 2:2002}))
        with self.assertRaisesRegex(ValueError, 'assignment changed'):
            self.collect()
        self.assertEqual(FakeReplay.calls, [])

    def test_overwrite_is_refused_before_any_action(self):
        self.collect()
        FakeReplay.calls = []
        with self.assertRaises(FileExistsError): self.collect()
        self.assertEqual(FakeReplay.calls, [])

    def test_no_stop_binding_permission_or_identity_prediction_labels(self):
        self.collect()
        for row in self.labels():
            for key in ('stop_label_available','binding_label_available','motion_permission_label_available',
                        'visual_identity_prediction_label_available','ego_label_available'):
                self.assertIs(row[key], False)

    def test_default_cli_only_verifies_without_execute(self):
        plan_path = self.root/'plan.json'
        plan_path.write_text('{}')
        with patch.object(collector, 'verify_plan', return_value={'status':'read_only'}), \
             patch.object(collector, 'execute_plan', side_effect=AssertionError('must not execute')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(collector.main(['--plan', str(plan_path)]), 0)

    def test_native_failure_keeps_all_four_batch_entries_without_retry(self):
        repo = self.root/'repo'; repo.mkdir()
        output = repo/'batch'
        entries = [dict(run_id=str(i), output_dir=str(output/str(i)), observation_count=i+1) for i in range(4)]
        plan = dict(output_root=str(output), stage='fixture', plan_sha256='fixture', entries=entries,
            environment={}, repository=str(repo), bundle=str(repo), python={'path':sys.executable},
            maximum_worker_wall_seconds=10)
        calls = []
        class NativeFailure:
            def __init__(self, command, **kwargs):
                calls.append((command, kwargs['env']['CUDA_VISIBLE_DEVICES']))
            def wait(self, timeout=None): return -6
        with patch.object(collector.subprocess, 'Popen', NativeFailure):
            self.assertEqual(collector.execute_plan(self.root/'fake_plan.json', plan), 2)
        batch = load(output/'batch_status.json')
        self.assertEqual(len(calls), 4)
        self.assertEqual([v[1] for v in calls], ['3']*4)
        self.assertEqual(len(batch['entries']), 4)
        self.assertEqual({r['status'] for r in batch['entries']}, {'failed_retained'})
        self.assertTrue(batch['all_four_sources_retained'])
        self.assertEqual(batch['successful_source_count'], 0)

    def test_habitat_capture_reads_shared_sensor_dictionary_once(self):
        reads = []
        obs = {'rgb':np.zeros((3,4,4), dtype=np.uint8), 'panoptic':np.zeros((3,4,1), dtype=np.uint32)}
        def sensors(): reads.append('render'); return obs
        sim = SimpleNamespace(get_world_time=lambda:0., get_sensor_observations=sensors,
            agents_mgr=[SimpleNamespace(articulated_agent=SimpleNamespace(base_pos=[0,0,0])),
                        SimpleNamespace(articulated_agent=SimpleNamespace(base_pos=[0,0,1]))])
        backend = collector.HabitatReplay({}, {})
        backend.env = SimpleNamespace(sim=sim, episode_over=False)
        backend.runtime = SimpleNamespace(RGB_KEY='rgb', PANOPTIC_KEY='panoptic', _camera_transform=lambda _:np.eye(4))
        backend.assigned = {0:100}
        capture = backend.capture(1)
        self.assertEqual(reads, ['render'])
        self.assertEqual(capture['rgb'].shape, (3,4,3))
        self.assertEqual(capture['panoptic'].shape, (3,4,1))
        self.assertFalse(np.shares_memory(capture['rgb'], obs['rgb']))
        self.assertFalse(np.shares_memory(capture['panoptic'], obs['panoptic']))


if __name__ == '__main__':
    unittest.main()
