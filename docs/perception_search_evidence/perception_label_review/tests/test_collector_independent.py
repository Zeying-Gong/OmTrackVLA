"""CPU-only tests of sidecar causality, persistence and same-render capture."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect_perception_sidecar import collect_episode
from collection_contract import artifact, canonical_sha


class FakeReplay:
    def __init__(self, frames):
        self.frames = frames
        self.actions = []
        self.captures = []
        self.closed = False

    def __enter__(self):
        return self

    def reset(self):
        pass

    def capture(self, step):
        self.captures.append(step)
        return copy.deepcopy(self.frames[step])

    def step(self, action):
        self.actions.append(list(action))

    def __exit__(self, *args):
        self.closed = True


class CollectorPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='omtrack-sidecar-cpu-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / 'output'
        self.output.mkdir()
        self.actions = [[.1, .2, -.3], [0., -.1, .4]]
        self.frames = []
        for step in range(3):
            camera = np.eye(4)
            camera[0, 3] = float(step)
            panoptic = np.zeros((3, 4, 1), dtype=np.uint32)
            if step != 1:
                panoptic[0:2, 0:2, 0] = 100
            panoptic[2, 3, 0] = 101
            self.frames.append(dict(rgb=np.full((3, 4, 3), 10 * step, dtype=np.uint8),
                panoptic=panoptic, assigned_ids={0: 100, 2: 101}, world_time_before_render_s=step*.1,
                world_time_s=step*.1, camera_before_render=camera, camera_after_render=camera,
                gt_distance_m=2.+step, terminal=step==2, initialization={'source': 'synthetic_test'}))
        source = {'steps': []}
        for step, action in enumerate(self.actions):
            source['steps'].append({'step': step+1,
                'policy': dict(action=dict(zip(('forward', 'lateral', 'yaw'), action)),
                               rgb_mean_0_255=float(10*step), rgb_std_0_255=0.,
                               rgb_temporal_absdiff_0_255=None if step == 0 else 10.),
                'evaluation_only_after_action': {'gt_distance_m': self.frames[step+1]['gt_distance_m'],
                    'gt_visible': step+1 != 1, 'render_audit': {
                        'camera_transform_before_action': self.frames[step]['camera_before_render'].tolist(),
                        'camera_transform_after_forced_render': self.frames[step+1]['camera_after_render'].tolist()}}})
        source_path = self.root / 'source.json'
        source_path.write_text(json.dumps(source), encoding='utf-8')
        self.entry = dict(source=artifact(source_path), saved_actions_sha256=canonical_sha(self.actions),
            run_id='synthetic_unit_test', observation_count=3, assigned_humanoid_semantic_ids={'0': 100, '2': 101},
            initialization=self.frames[0]['initialization'], prefix_samples=[])
        self.plan = dict(replay_tolerance=1e-4, world_time_tolerance_s=1e-8,
                         source_rgb_statistics_tolerance=1e-8, rgb_statistics_tolerance=1e-8)

    def collect(self, frames=None):
        self.backend = FakeReplay(self.frames if frames is None else frames)
        return collect_episode(self.entry, self.plan, self.output,
                               backend_factory=lambda entry, plan: self.backend)

    def labels(self):
        return [json.loads(line) for line in (self.output/'labels.jsonl').read_text().splitlines()]



    def test_failed_gt_audit_does_not_select_change_or_truncate_actions(self):
        frames = copy.deepcopy(self.frames)
        frames[1]['gt_distance_m'] = 999.
        frames[1]['panoptic'][0:2, 0:2, 0] = 100
        result = self.collect(frames)
        self.assertEqual(result['status'], 'failed_alignment')
        self.assertEqual(result['observations_saved'], 3)
        self.assertGreaterEqual(result['failed_check_count'], 2)
        self.assertEqual(self.backend.actions, self.actions)
        self.assertFalse(self.labels()[1]['source_audit']['passed'])


    def test_nonincreasing_actual_clock_fails_without_altering_replay(self):
        frames = copy.deepcopy(self.frames)
        frames[1]['world_time_s'] = frames[1]['world_time_before_render_s'] = 0.
        result = self.collect(frames)
        self.assertEqual(result['status'], 'failed_alignment')
        self.assertEqual(self.backend.actions, self.actions)
        checks = self.labels()[1]['source_audit']['checks']
        self.assertFalse(next(c for c in checks if c['kind'] == 'worldtime_strictly_increasing')['passed'])





if __name__ == '__main__':
    unittest.main()
