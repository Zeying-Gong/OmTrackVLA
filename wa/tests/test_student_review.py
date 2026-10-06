import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from wa.tools.build_student_review import audit_media

class ReviewMediaTests(unittest.TestCase):
    row=dict(artifact_root='/unused',task='stt',key='scene/1',initial_rgb_sha256='expected')
    def run_media(self,duration='1.2',streams=None,digest='expected',size=100):
        if streams is None:streams=[dict(codec_type='video',width=384,height=384)]
        response=SimpleNamespace(stdout=json.dumps(dict(format=dict(duration=duration),streams=streams)))
        with patch('wa.tools.build_student_review.digest',return_value=digest),patch('pathlib.Path.stat',return_value=SimpleNamespace(st_size=size)),patch('wa.tools.build_student_review.subprocess.run',return_value=response):
            return audit_media(self.row,'ffprobe')
    def test_valid(self):
        self.assertEqual(self.run_media(),('stt/_review/scene/1/review.mp4',1.2))
    def test_changed_first_frame(self):
        with self.assertRaises(ValueError):self.run_media(digest='changed')
    def test_empty_video(self):
        with self.assertRaises(ValueError):self.run_media(size=0)
    def test_no_video_stream(self):
        with self.assertRaises(ValueError):self.run_media(streams=[dict(codec_type='audio')])
    def test_invalid_durations(self):
        for duration in ('0','-1','nan','inf','-inf'):
            with self.subTest(duration=duration),self.assertRaises(ValueError):self.run_media(duration=duration)

if __name__=='__main__':unittest.main()
