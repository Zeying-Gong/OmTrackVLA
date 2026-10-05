import copy
import unittest
from wa.wm.dual_teacher_selection import select_teacher, EXPERIMENT

def branch(name, success=True, rate=.8):
    return dict(experiment=EXPERIMENT, teacher=name, complete=True,
                replay_verified=True, task="stt", key="scene/1",
                takeover_step=12, seed=7, protocol_sha256="protocol",
                initial_rgb_sha256="rgb", takeover_state_sha256="state",
                result=dict(success=success, collision=False,
                            policy_init_valid=True, following_rate=rate))

class SelectionTests(unittest.TestCase):
    def test_single_success(self):
        for teacher in ("lightnav", "oracle"):
            a, b = branch("lightnav", teacher=="lightnav"), branch("oracle", teacher=="oracle")
            self.assertEqual(select_teacher(a,b)["selected_teacher"], teacher)
    def test_rate(self):
        self.assertEqual(select_teacher(branch("lightnav"),branch("oracle",rate=.9))["selected_teacher"],"oracle")
        self.assertEqual(select_teacher(branch("lightnav",rate=.9),branch("oracle"))["selected_teacher"],"lightnav")
    def test_tie(self):
        self.assertEqual(select_teacher(branch("lightnav"),branch("oracle"))["reason"],"equal_rate_fixed_lightnav_tie")
    def test_neither(self):
        r=select_teacher(branch("lightnav",False),branch("oracle",False))
        self.assertIsNone(r["selected_teacher"])
        self.assertFalse(r["demonstration_candidate"])
    def test_pairing(self):
        for key in ("task","key","takeover_step","seed","protocol_sha256","initial_rgb_sha256","takeover_state_sha256"):
            b=branch("oracle");b[key]="different"
            with self.assertRaises(ValueError):select_teacher(branch("lightnav"),b)
    def test_bad_rates(self):
        for rate in (float("nan"),float("inf"),-1,1.1,None,True):
            with self.assertRaises(ValueError):select_teacher(branch("lightnav"),branch("oracle",rate=rate))
    def test_fail_closed(self):
        for key,val in (("complete",False),("replay_verified",False),("transport_fallback",True),("experiment","train")):
            b=branch("oracle");b[key]=val
            with self.assertRaises(ValueError):select_teacher(branch("lightnav"),b)
    def test_inconsistent_success(self):
        for key,val in (("collision",True),("policy_init_valid",False)):
            b=branch("oracle");b["result"][key]=val
            with self.assertRaises(ValueError):select_teacher(branch("lightnav"),b)
    def test_no_mutation_or_release(self):
        a,b=branch("lightnav"),branch("oracle")
        before=copy.deepcopy((a,b));r=select_teacher(a,b)
        self.assertEqual((a,b),before)
        self.assertFalse(r["training_released"])

if __name__ == "__main__":unittest.main()
