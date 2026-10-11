"""Synthetic, local-file tests; never read live 63933/63934/63935 outputs."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

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
        yy, xx = np.indices((384, 384))
        self.front = np.stack(((xx//2+yy//4) % 256, (yy//2) % 256,
                               (xx//3) % 256), axis=2).astype(np.uint8)
        Image.fromarray(self.front).save(self.photo, quality=90)
        self._encode_video(self.front)
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

    def _encode_video(self, front, destination=None, frames=2):
        destination = destination or self.video
        canvas = np.zeros((432, 768, 3), dtype=np.uint8)
        canvas[48:432, :384] = front
        canvas[48:432, 384:] = (20, 40, 80)
        still = destination.with_suffix('.png')
        Image.fromarray(canvas).save(still)
        subprocess.run([
            '/usr/bin/ffmpeg', '-nostdin', '-v', 'error', '-y',
            '-loop', '1', '-framerate', '20', '-i', str(still),
            '-frames:v', str(frames), '-c:v', 'libx264', '-preset', 'ultrafast',
            '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            str(destination),
        ], check=True, capture_output=True)
        return destination

    def _write(self):
        self.trace.write_text(''.join(json.dumps(item)+'\n' for item in self.trace_rows))
        self.steps.write_text(''.join(json.dumps(item)+'\n' for item in self.review_rows))
        self.complete.write_text(json.dumps(dict(
            frames=len(self.trace_rows), policy_inputs_unchanged=True,
            video_timebase='20fps frame playback; actual variable simulator timestamps',
            result={name: self.row[name] for name in audit.METRIC_FIELDS})))

    def _audit(self):
        return audit.audit_episode(self.row, '/usr/bin/ffprobe')

    def test_complete_episode_and_truthful_caption(self):
        result = self._audit()
        self.assertEqual(result['steps'], 2)
        self.assertEqual(result['prediction_steps'], 2)
        self.assertEqual(result['video_decoded_frames'], 2)
        self.assertEqual(result['video_sha256'], audit.file_sha(self.video))
        self.assertEqual(result['frame_snapshots'][0]['sha256'],
                         self.row['initial_rgb_sha256'])
        summary = audit.corrected_summary(dict(
            mode='image', metrics_percent={}, limits=[
                'RGB+first GT bbox; no UWB at inference; same model interface, '
                'not no-UWB training; no text']))
        page, _ = audit.render_index(summary, [result], {self.row['artifact_root']})
        self.assertIn('图像输入重训权重', page)
        self.assertNotIn('不是无UWB重训', page)
        self.assertIn('每帧视频解码', page)
        self.assertNotIn('没有逐帧解码', page)
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


    def test_swapped_video_with_valid_metadata_rejected(self):
        other = self._encode_video(255-self.front,
                                   self.video.with_name('unrelated.mp4'))
        self.video.write_bytes(other.read_bytes())
        duration, dimensions = audit.probe_video(self.video, '/usr/bin/ffprobe')
        self.assertEqual(dimensions, audit.VIDEO_DIMENSIONS)
        self.assertAlmostEqual(duration, 0.1)
        with self.assertRaisesRegex(ValueError, 'JPEG disagrees'):
            self._audit()

    def test_truncated_after_first_frame_with_metadata_rejected(self):
        # faststart leaves stream/duration metadata at the front of the MP4.
        payload = self.video.read_bytes()
        self.video.write_bytes(payload[:int(len(payload)*0.7)])
        duration, dimensions = audit.probe_video(self.video, '/usr/bin/ffprobe')
        self.assertEqual(dimensions, audit.VIDEO_DIMENSIONS)
        self.assertAlmostEqual(duration, 0.1)
        with self.assertRaisesRegex(ValueError, 'full video decode|decoded video frame count'):
            self._audit()

    def test_wrong_decoded_frame_count_rejected(self):
        trace = copy.deepcopy(self.trace_rows[-1])
        step = copy.deepcopy(self.review_rows[-1])
        trace['step'] = 3
        step['step'] = 2
        self.trace_rows.append(trace)
        self.review_rows.append(step)
        self.row['total_step'] = 3
        self._write()
        # The old 0.1-second duration tolerance accepts a 2-vs-3 mismatch.
        with self.assertRaisesRegex(ValueError, 'decoded video frame count'):
            self._audit()

    def test_success_row_cannot_claim_invalid_bbox_failure(self):
        self.row['success'] = 1
        self.trace_rows[0]['failure'] = 'invalid_initialization_bbox: fake'
        self.trace_rows[0]['diagnostics'] = {}
        self.trace_rows[0]['action'] = [0., 0., 0.]
        self.review_rows[0]['action'] = [0., 0., 0.]
        self._write()
        with self.assertRaisesRegex(ValueError, 'invalid-init failure exemption'):
            self._audit()

    def test_true_invalid_bbox_noop_is_preserved(self):
        self.row['policy_init_valid'] = False
        for trace, step in zip(self.trace_rows, self.review_rows):
            trace['failure'] = 'invalid_initialization_bbox: empty first box'
            trace['diagnostics'] = {}
            trace['action'] = [0., 0., 0.]
            step['action'] = [0., 0., 0.]
        self._write()
        result = self._audit()
        self.assertEqual(result['prediction_steps'], 0)
        self.assertEqual(result['invalid_init_noop_steps'], 2)

    def test_periodic_snapshot_set_and_pixels(self):
        # Small synthetic cadence exercises the same exact set/pixel logic.
        with mock.patch.object(audit, 'SNAPSHOT_INTERVAL', 1):
            second = self.photo.parent/'step_0001.jpg'
            Image.fromarray(self.front).save(second, quality=90)
            result = self._audit()
            self.assertEqual([item['step'] for item in result['frame_snapshots']],
                             [0, 1])
            Image.fromarray(255-self.front).save(second, quality=90)
            with self.assertRaisesRegex(ValueError, 'JPEG disagrees'):
                self._audit()

    def test_missing_periodic_jpeg_rejected(self):
        with mock.patch.object(audit, 'SNAPSHOT_INTERVAL', 1):
            with self.assertRaisesRegex(ValueError, 'periodic JPEG set'):
                self._audit()

    def test_extra_saved_jpeg_rejected(self):
        Image.fromarray(self.front).save(
            self.photo.parent/'step_0100.jpg', quality=90)
        with self.assertRaisesRegex(ValueError, 'periodic JPEG set'):
            self._audit()


if __name__ == '__main__':
    unittest.main()
