import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from wa.wm.dual_teacher_pair_evidence import verify_pair_evidence

class PairEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state=dict(timestamp=0.,agents=[dict(transform=[[1.,0.],[0.,1.]],joints=[0.])])
        self.row=dict(pair=dict(initial_rgb_sha256='runtime_rgba_hash',
            takeover_state_sha256=hashlib.sha256(json.dumps(self.state,sort_keys=True).encode()).hexdigest()),branches={})
        for teacher in ('lightnav','oracle'):
            root=Path(self.tmp.name)/teacher;root.mkdir()
            self.row['branches'][teacher]=dict(artifact_root=str(root))
            (root/'pair_start.json').write_text(json.dumps(dict(rgb='runtime_rgba_hash',state=self.state)))
            (root/'observations.json').write_text(json.dumps([dict(sim_step=0,timestamp_s=0.,frame='rgb_0000.png')]))
            Image.new('RGB',(4,4),(1,2,3)).save(root/'rgb_0000.png')
    def root(self):return Path(self.row['branches']['oracle']['artifact_root'])
    def test_rgb_rgba_hash_not_conflated(self):
        self.assertEqual(set(verify_pair_evidence(self.row)),{'lightnav','oracle'})
    def test_changed_pixels_rejected(self):
        Image.new('RGB',(4,4),(3,2,1)).save(self.root()/'rgb_0000.png')
        with self.assertRaisesRegex(ValueError,'pixels differ'):verify_pair_evidence(self.row)
    def test_changed_runtime_record_rejected(self):
        p=self.root()/'pair_start.json';r=json.loads(p.read_text());r['rgb']='changed';p.write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError,'RGB hash differs'):verify_pair_evidence(self.row)
    def test_changed_dynamic_state_rejected(self):
        p=self.root()/'pair_start.json';r=json.loads(p.read_text());r['state']['agents'][0]['joints']=[1.];p.write_text(json.dumps(r))
        with self.assertRaisesRegex(RuntimeError,'DIVERGED'):verify_pair_evidence(self.row)
    def test_noninitial_frame_rejected(self):
        (self.root()/'observations.json').write_text(json.dumps([dict(sim_step=1,timestamp_s=.1,frame='rgb_0000.png')]))
        with self.assertRaisesRegex(ValueError,'not start state'):verify_pair_evidence(self.row)
    def test_wrong_canonical_hash_rejected(self):
        self.row['pair']['takeover_state_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'canonical'):verify_pair_evidence(self.row)
