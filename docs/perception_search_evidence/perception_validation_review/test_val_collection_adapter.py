"""CPU fake simulator checks; these never certify any real validation labels."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import generate_val4_protocol as protocol
import val_collection_adapter as adapter


class SyntheticBaselineBackend:
    cpu_mock_only = True
    early_terminal_at = None
    render_clock_drift = False
    observed_actions = []

    def __init__(self, entry, plan):
        self.entry, self.plan = entry, plan
        self.source = protocol.read(entry['source']['path'])
        self.assigned = {int(k): v for k, v in entry['assigned_humanoid_semantic_ids'].items()}
        self.expected_actions = entry['saved_actions']
        self.current_step = 0

    def __enter__(self): return self
    def __exit__(self, *args): pass
    def reset(self):
        type(self).observed_actions = []
        self.current_step = 0

    def capture(self, step):
        if step != self.current_step: raise AssertionError('incorrect observation/action alignment')
        after = self.source['steps'][max(0, step-1)]['evaluation_only_after_action']
        camera = after['render_audit']['camera_transform_after_forced_render'] if step else (
            self.source['steps'][0]['evaluation_only_after_action']['render_audit']['camera_transform_before_action'])
        visible = bool(after['gt_visible']) if step else True
        rgb = np.full((4, 4, 3), step, dtype=np.uint8)
        panoptic = np.zeros((4, 4, 1), dtype=np.int32)
        if visible: panoptic[1:3, 1:3, 0] = self.assigned[0]
        now = step*.1
        return dict(rgb=rgb, panoptic=panoptic, assigned_ids=self.assigned,
                    world_time_s=now, world_time_before_render_s=now+.01 if self.render_clock_drift else now,
                    camera_before_render=np.array(camera), camera_after_render=np.array(camera),
                    gt_distance_m=float(after['gt_distance_m']),
                    terminal=step == len(self.expected_actions) or step == self.early_terminal_at,
                    initialization=self.entry['initialization'] if step == 0 else None)

    def step(self, action):
        if action != self.expected_actions[self.current_step]: raise AssertionError('action changed')
        type(self).observed_actions.append(list(action))
        self.current_step += 1


class EarlyTerminal(SyntheticBaselineBackend):
    early_terminal_at = 2


class ClockDrift(SyntheticBaselineBackend):
    render_clock_drift = True


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = protocol.generate()
        cls.case = cls.plan['entries'][0]
        cls.source_path = Path(cls.case['source_artifacts']['result.json']['local_review_path'])
        cls.config = protocol.ROOT/'perception_label_review/source_config_inventory.json'
        cls.template = protocol.ROOT/'perception_label_review'

    def collect(self, output, backend=SyntheticBaselineBackend):
        return adapter.collect_cpu_mock_case(self.plan, self.case['case_id'], output=output,
            source_path=self.source_path, configuration_snapshot=self.config,
            template_directory=self.template, backend_factory=backend)

    def test_projection_keeps_val_scope_and_excludes_unrelated_old_prefixes(self):
        entry, runtime = adapter.runtime_projection(self.plan, self.case['case_id'],
            source_path=self.source_path, configuration_snapshot=self.config)
        self.assertEqual(entry['permanent_partition_role'], 'val')
        self.assertEqual(entry['source_dataset_split'], 'train')
        self.assertEqual(entry['prefix_samples'], [])
        self.assertFalse(runtime['source_worldtime_reference_available'])
        self.assertFalse(runtime['source_raw_RGB_reference_available'])
        self.assertEqual(entry['saved_actions'], self.case['saved_actions'])

    def test_resealed_wrong_role_arm_or_counts_still_rejected(self):
        for key, value in [('permanent_partition_role', 'train'), ('source_arm', 'pilot_best16'),
                           ('observation_count', 90), ('optimizer_input_allowed', True)]:
            plan = copy.deepcopy(self.plan); plan['entries'][0][key] = value
            del plan['protocol_sha256']; plan['protocol_sha256'] = protocol.canonical_sha(plan)
            with self.subTest(key=key), self.assertRaises(ValueError): adapter.validate_protocol(plan)

    def test_changed_source_bytes_fail_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory)/'changed.json'; changed.write_bytes(self.source_path.read_bytes()+b' ')
            output = Path(directory)/'output'
            with self.assertRaisesRegex(ValueError, 'source bytes'):
                adapter.collect_cpu_mock_case(self.plan, self.case['case_id'], output=output,
                    source_path=changed, configuration_snapshot=self.config,
                    template_directory=self.template, backend_factory=SyntheticBaselineBackend)
            self.assertFalse(output.exists())

    def test_all_original_actions_labels_and_failures_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'case'; result = self.collect(output)
            self.assertEqual(result['status'], 'cpu_mock_failed_alignment')
            self.assertEqual(SyntheticBaselineBackend.observed_actions, self.case['saved_actions'])
            self.assertEqual(result['observations_saved'], 91)
            self.assertFalse(result['real_val_collection'])
            self.assertFalse(result['independent_admission'])
            rows = [json.loads(row) for row in (output/'labels.jsonl').read_text().splitlines()]
            self.assertEqual([r['environment_step'] for r in rows], list(range(91)))
            self.assertEqual(rows[0]['next_source_action_step'], 1)
            self.assertEqual(rows[-1]['next_source_action_step'], None)
            self.assertTrue(rows[-1]['terminal_observation'])
            self.assertTrue(any(not r['source_audit']['passed'] for r in rows))
            self.assertFalse(any(check['kind'] == 'stored_prefix_rgb_and_worldtime'
                                 for row in rows for check in row['source_audit']['checks']))
            manifest = protocol.read(output/'val_artifact_manifest.json')
            self.assertEqual(manifest['labels.jsonl']['sha256'], protocol.file_sha(output/'labels.jsonl'))
            self.assertEqual(result['artifact_manifest_sha256'], protocol.file_sha(output/'val_artifact_manifest.json'))
            for relative, reference in manifest.items():
                self.assertEqual(reference['sha256'], protocol.file_sha(output/relative))

    def test_early_terminal_retains_partial_evidence_and_fixed_expectation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'case'; result = self.collect(output, EarlyTerminal)
            self.assertEqual(result['status'], 'failed_collection')
            self.assertEqual(result['expected_observations'], 91)
            self.assertIn('early natural termination', result['error'])
            self.assertEqual(len(EarlyTerminal.observed_actions), 2)
            self.assertEqual(len((output/'labels.jsonl').read_text().splitlines()), 3)
            self.assertTrue((output/'val_artifact_manifest.json').exists())

    def test_same_capture_clock_change_is_recorded_as_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'case'; result = self.collect(output, ClockDrift)
            self.assertEqual(result['status'], 'cpu_mock_failed_alignment')
            rows = [json.loads(row) for row in (output/'labels.jsonl').read_text().splitlines()]
            for row in rows:
                check = next(c for c in row['source_audit']['checks'] if c['kind'] == 'single_render_worldtime_stable')
                self.assertFalse(check['passed'])

    def test_outputs_never_overwritten_and_real_backend_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'case'; output.mkdir()
            with self.assertRaisesRegex(ValueError, 'overwrite'): self.collect(output)
            with self.assertRaisesRegex(ValueError, 'CPU mock'):
                self.collect(Path(directory)/'uncreated', backend=object)
            self.assertFalse((Path(directory)/'uncreated').exists())


if __name__ == '__main__':
    unittest.main()
