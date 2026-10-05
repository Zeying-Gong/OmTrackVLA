import json,unittest
from unittest.mock import patch
from wa.wm.dual_teacher_resume import load_resume
from wa.wm.dual_teacher_selection import EXPERIMENT
class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.manifest={'tasks':{t:{'episodes':[{'key':str(i)} for i in range(1405)]} for t in ('stt','dt','at')}}
        remaining=[dict(task=t,key=str(i)) for t in ('stt','dt','at') for i in range(1405) if (t,i)!=('stt',0)]
        self.plan=dict(experiment=EXPERIMENT,expected_total=4215,manifest='manifest',manifest_sha256='ok',
            completed_rows=[dict(pair=dict(task='stt',key='0'))],remaining=remaining,
            lanes=[remaining[i::8] for i in range(8)],completed_count=1,remaining_count=4214)
    def load(self,h='ok'):
        plan,manifest=json.dumps(self.plan),json.dumps(self.manifest)
        with patch('wa.wm.dual_teacher_resume.digest',return_value='ok'),patch('pathlib.Path.read_text',lambda p:manifest if str(p)=='manifest' else plan):
            return load_resume('plan',h)
    def test_exact_remainder(self):
        self.assertEqual(self.load()['remaining_count'],4214)
    def test_hash(self):
        with self.assertRaisesRegex(ValueError,'hash'):self.load('bad')
    def test_repeated_old(self):
        self.plan['completed_rows']*=2
        with self.assertRaisesRegex(ValueError,'duplicate'):self.load()
    def test_old_new_overlap(self):
        self.plan['remaining'][0]=dict(task='stt',key='0')
        with self.assertRaisesRegex(ValueError,'coverage'):self.load()
    def test_lane_duplicate(self):
        self.plan['lanes'][0].append(self.plan['lanes'][1][0])
        with self.assertRaisesRegex(ValueError,'duplicate'):self.load()
    def test_missing_lane(self):
        self.plan['lanes'].pop()
        with self.assertRaisesRegex(ValueError,'lane'):self.load()
    def test_wrong_count(self):
        self.plan['remaining_count']-=1
        with self.assertRaisesRegex(ValueError,'count'):self.load()
if __name__=='__main__':unittest.main()
