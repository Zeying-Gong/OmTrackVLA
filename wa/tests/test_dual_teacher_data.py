"""Fail-closed identity boundaries; synthetic files never enter a real cache."""
import json
from types import SimpleNamespace
import torch
import tempfile
import unittest
from pathlib import Path
from wa.data import TrackingData, sha
from wa.wm.dual_teacher_data import DualTeacherData
from wa.wm.dual_teacher_selection import EXPERIMENT

class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.meta = dict(experiment=EXPERIMENT, partition='evaluation_adaptation',
            camera_alignment_verified=True, initial_bbox_status='VERIFIED_FROZEN_FIRST_RGB_REPAIR',
            task='stt', key='scene/1', teacher='oracle')
        (self.root/'result.json').write_text(json.dumps(dict(success=True, collision=False)))
        (self.root/'frame.png').write_bytes(b'synthetic frame hash fixture')
        self.data = DualTeacherData.__new__(DualTeacherData)
        self.data.admissions = {str(self.root):dict(task='stt',key='scene/1',teacher='oracle',
            hashes={'result.json':sha(self.root/'result.json')})}
        self.data.image_hashes = {str(self.root):{'frame.png':sha(self.root/'frame.png')}}

    def check(self):
        self.data.validate_identity(self.meta, self.root)

    def test_verified_repair(self):
        self.check()

    def test_original_guard_not_relaxed(self):
        original = TrackingData.__new__(TrackingData)
        with self.assertRaisesRegex(ValueError, 'Unverified'):
            original.validate_identity(self.meta, self.root)
        original.validate_identity(dict(camera_alignment_verified=True,
            initial_bbox_status='VERIFIED_CONFIG_AND_SEMANTIC'), self.root)

    def test_changed_image(self):
        (self.root/'frame.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'image changed'):self.check()

    def test_changed_result(self):
        (self.root/'result.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):self.check()

    def test_failed_branch_even_if_hashed(self):
        (self.root/'result.json').write_text(json.dumps(dict(success=False, collision=False)))
        self.data.admissions[str(self.root)]['hashes']['result.json']=sha(self.root/'result.json')
        with self.assertRaisesRegex(ValueError, 'failed branch'):self.check()

    def test_collision_even_if_success(self):
        (self.root/'result.json').write_text(json.dumps(dict(success=True, collision=True)))
        self.data.admissions[str(self.root)]['hashes']['result.json']=sha(self.root/'result.json')
        with self.assertRaisesRegex(ValueError, 'failed branch'):self.check()

    def test_unreleased(self):
        self.data.admissions={}
        with self.assertRaisesRegex(ValueError, 'unreleased'):self.check()

    def test_wrong_partition(self):
        self.meta['partition']='train'
        with self.assertRaisesRegex(ValueError, 'unmarked'):self.check()

    def test_wrong_teacher(self):
        self.meta['teacher']='lightnav'
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):self.check()

    def test_unknown_identity(self):
        self.meta['initial_bbox_status']='ASSUMED'
        with self.assertRaisesRegex(ValueError, 'unverified'):self.check()

    def test_camera_alignment(self):
        self.meta['camera_alignment_verified']=False
        with self.assertRaisesRegex(ValueError, 'unaligned'):self.check()

class PolicyInputBoundaryTests(unittest.TestCase):
    def test_training_labels_cannot_change_policy_conditions(self):
        from wa.wm.training import JointRobotModel
        model=SimpleNamespace(encode=lambda images:images.clone())
        batch=dict(rgb=torch.zeros(3,4,3,2,2),template=torch.zeros(3,3,2,2),
            polar=torch.ones(3,2),times=torch.tensor([[-1.5,-1.,-.5,0.]]*3))
        modes=torch.tensor([0,1,2])
        before=JointRobotModel.make_conditions(model,batch,modes)
        # Exercise the real condition builder with deliberately unusable labels.
        for name in ('future','pose','geometry','wm_rgb','commands','proprio',
                     'oracle_future_path','instruction'):
            batch[name]=object()
        after=JointRobotModel.make_conditions(model,batch,modes)
        self.assertEqual(set(before),set(after))
        for key in before:self.assertTrue(torch.equal(before[key],after[key]),key)
        self.assertEqual(after['uwb_valid'].tolist(),[False,True,True])
        self.assertEqual(after['template_valid'].tolist(),[True,False,True])

    def test_current_uwb_is_an_explicit_policy_input(self):
        from wa.wm.training import JointRobotModel
        model=SimpleNamespace(encode=lambda images:images.clone())
        batch=dict(rgb=torch.zeros(1,4,3,2,2),template=torch.zeros(1,3,2,2),
            polar=torch.tensor([[2.,.3]]),times=torch.zeros(1,4))
        conditions=JointRobotModel.make_conditions(model,batch,torch.tensor([2]))
        self.assertTrue(torch.equal(conditions['polar'],batch['polar']))

if __name__=='__main__':unittest.main()
