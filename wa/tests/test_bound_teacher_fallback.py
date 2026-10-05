"""Run in Habitat runtime; inject faults, not benchmark performance samples."""
import unittest
from wa.wm.dual_teacher_collect import BoundTeacher

class FakeReleasedClient:
    def __init__(self):self.episode_fallback_count=0;self.calls=0
    def act(self,*args):
        self.calls+=1
        self.reply_error={'rc':500,'msg':'Missing rvq act levels [1]'} if self.calls==1 else None
        if self.reply_error:self.episode_fallback_count+=1
        self.last_trajectory=None
        return [0.,0.,-1.]

class FallbackTests(unittest.TestCase):
    def test_record_once_and_preserve_action(self):
        agent=BoundTeacher(FakeReleasedClient(),allow_released_fallback=True)
        self.assertEqual(agent.act({}, {}, '7'),[0.,0.,-1.])
        self.assertEqual(agent.act({}, {}, '7'),[0.,0.,-1.])
        self.assertEqual(agent.fallback_count,1)
        self.assertEqual(len(agent.fallback_events),1)
        self.assertEqual(agent.fallback_events[0]['step'],0)
        self.assertEqual(agent.fallback_events[0]['error']['rc'],500)

    def test_default_still_rejects(self):
        with self.assertRaisesRegex(RuntimeError,'TEACHER_TRANSPORT_FALLBACK'):
            BoundTeacher(FakeReleasedClient()).act({}, {}, '7')

    def test_bad_action_never_executes(self):
        teacher=FakeReleasedClient()
        teacher.act=lambda *args:[float('nan'),0.,0.]
        with self.assertRaisesRegex(ValueError,'invalid teacher action'):
            BoundTeacher(teacher,allow_released_fallback=True).act({}, {}, '7')

if __name__=='__main__':unittest.main()
