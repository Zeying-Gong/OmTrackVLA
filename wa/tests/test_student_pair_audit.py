import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from wa.wm.student_pair_audit import audit_starts

class PairAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        state=dict(timestamp=0.,agents=[dict(transform=[[1.,0.],[0.,1.]],joints=[0.])])
        self.start=dict(rgb='a'*64,state=state)
        self.student=[dict(task='stt',key='scene/1',initial_pair_evidence=copy.deepcopy(self.start))]
        self.teacher=[dict(pair=dict(task='stt',key='scene/1',seed=7,takeover_step=0,
            initial_rgb_sha256='a'*64),branches={})]
        hashes={}
        for name in ('lightnav','oracle'):
            root=Path(self.tmp.name)/name;root.mkdir()
            path=root/'pair_start.json';path.write_text(json.dumps(self.start))
            self.teacher[0]['branches'][name]=dict(artifact_root=str(root))
            hashes[name]={str(path):hashlib.sha256(path.read_bytes()).hexdigest()}
        self.mock=patch('wa.wm.student_pair_audit.verify_pair_evidence',return_value=hashes)
        self.mock.start();self.addCleanup(self.mock.stop)
        self.keys=[('stt','scene/1')]
    def run_audit(self):return audit_starts(self.student,self.teacher,self.keys)
    def test_equal(self):self.assertEqual(self.run_audit()['episodes'],1)
    def test_coverage(self):
        for students in ([],self.student*2,[dict(self.student[0],key='foreign/1')]):
            with self.assertRaises(ValueError):audit_starts(students,self.teacher,self.keys)
        with self.assertRaises(ValueError):audit_starts(self.student,[],self.keys)
    def test_raw_rgb(self):
        self.student[0]['initial_pair_evidence']['rgb']='b'*64
        with self.assertRaises(ValueError):self.run_audit()
    def test_state_divergence(self):
        self.student[0]['initial_pair_evidence']['state']['agents'][0]['joints']=[.1]
        with self.assertRaises(RuntimeError):self.run_audit()
    def test_nonfinite_time(self):
        self.student[0]['initial_pair_evidence']['state']['timestamp']=float('nan')
        with self.assertRaises(ValueError):self.run_audit()
    def test_missing(self):
        self.student[0].pop('initial_pair_evidence')
        with self.assertRaises(ValueError):self.run_audit()
    def test_teacher_changed(self):
        path=Path(self.teacher[0]['branches']['oracle']['artifact_root'])/'pair_start.json'
        path.write_text('{}')
        with self.assertRaises(ValueError):self.run_audit()
    def test_wrong_seed(self):
        self.teacher[0]['pair']['seed']=8
        with self.assertRaises(ValueError):self.run_audit()

if __name__=='__main__':unittest.main()
