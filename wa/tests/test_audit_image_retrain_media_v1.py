"""Synthetic, local-file tests; never read live 63933/63934/63935 outputs."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from wa.tools import audit_image_retrain_media_v1 as audit


class MediaTraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='wa_synthetic_media_trace_')
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)/'shard_00'
        root.mkdir()
        key = 'scene/7'
        self.photo = root/'stt/_live'/key/'step_0000.jpg'
        self.video = root/'stt/_review'/key/'review.mp4'
        self.steps = self.video.parent/'steps.jsonl'
        self.complete = self.video.parent/'complete.json'
        self.trace = root/'stt/scene_7.trace.jsonl'
        self.photo.parent.mkdir(parents=True)
        self.video.parent.mkdir(parents=True)
        self.trace.parent.mkdir(parents=True, exist_ok=True)
        self.photo.write_bytes(b'synthetic jpeg')
        self.video.write_bytes(b'synthetic mp4')
        self.row = dict(task='stt', key=key, artifact_root=str(root),
            mode='image', checkpoint_sha256=audit.NEW_SHA,
            checkpoint_step=audit.STEP, total_step=2, success=0,
            collision=0, finish=False, following_rate=0.2,
            following_step=1, policy_init_valid=True, status='failure',
            initial_rgb_sha256=hashlib.sha256(self.photo.read_bytes()).hexdigest())
        self.trace_rows = []
        self.review_rows = []
        for index in range(2):
            action = [0.1, 0.2, 0.3]
            position = dict(robot_position_world=[1., 2., 3.],
                            robot_rotation_world_from_body=[[1., 0., 0.]],
                            target_position_world=[4., 5., 6.])
            observer = dict(position, usage='offline_diagnosis_only_not_policy_input',
                            timing='before_current_action_after_previous_action')
            self.trace_rows.append(dict(step=index+1, mode='image',
                action=copy.deepcopy(action), failure=None, observer_state=copy.deepcopy(observer),
                diagnostics=dict(mode='image', noise_mode='zero',
                    inputs='RGB+first_frame_bbox', uwb=None,
                    world_predictor_inference=False)))
            self.review_rows.append(dict(step=index, action=copy.deepcopy(action),
                usage='observer_only_not_policy_input', **copy.deepcopy(position)))
        self._write()

    def _write(self):
        self.trace.write_text(''.join(json.dumps(item)+'\n' for item in self.trace_rows))
        self.steps.write_text(''.join(json.dumps(item)+'\n' for item in self.review_rows))
        self.complete.write_text(json.dumps(dict(
            frames=2, policy_inputs_unchanged=True,
            video_timebase='20fps frame playback; actual variable simulator timestamps',
            result={name: self.row[name] for name in audit.METRIC_FIELDS})))

    @staticmethod
    def _probe(path, ffprobe):
        return 0.1, [768, 432]

    def _audit(self):
        return audit.audit_episode(self.row, '/synthetic/ffprobe', self._probe)

    def test_complete_episode_and_truthful_caption(self):
        result = self._audit()
        self.assertEqual(result['steps'], 2)
        self.assertEqual(result['prediction_steps'], 2)
        summary = audit.corrected_summary(dict(
            mode='image', metrics_percent={}, limits=[
                'RGB+first GT bbox; no UWB at inference; same model interface, '
                'not no-UWB training; no text']))
        page, _ = audit.render_index(summary, [result], {self.row['artifact_root']})
        self.assertIn('图像输入重训权重', page)
        self.assertNotIn('不是无UWB重训', page)
        self.assertIn('checkpoint retrained image-only', summary['limits'][0])

    def test_missing_trace_rejected(self):
        self.trace.unlink()
        with self.assertRaisesRegex(ValueError, 'missing or empty'):
            self._audit()

    def test_misaligned_trace_step_rejected(self):
        self.trace_rows[1]['step'] = 9
        self._write()
        with self.assertRaisesRegex(ValueError, 'step or mode'):
            self._audit()

    def test_tampered_trace_action_rejected(self):
        self.trace_rows[0]['action'][0] = 0.9
        self._write()
        with self.assertRaisesRegex(ValueError, 'action mismatch'):
            self._audit()

    def test_tampered_no_uwb_declaration_rejected(self):
        self.trace_rows[0]['diagnostics']['uwb'] = {'valid': True}
        self._write()
        with self.assertRaisesRegex(ValueError, 'runtime contract'):
            self._audit()

    def test_tampered_observer_state_rejected(self):
        self.trace_rows[0]['observer_state']['robot_position_world'][0] = 5.
        self._write()
        with self.assertRaisesRegex(ValueError, 'observer-only'):
            self._audit()

    def test_missing_frame_rejected(self):
        self.complete.write_text(json.dumps(dict(
            frames=1, policy_inputs_unchanged=True,
            video_timebase='20fps frame playback',
            result={name: self.row[name] for name in audit.METRIC_FIELDS})))
        with self.assertRaisesRegex(ValueError, 'counts differ'):
            self._audit()


if __name__ == '__main__':
    unittest.main()
