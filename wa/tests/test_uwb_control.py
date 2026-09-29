import unittest
import numpy as np
from wa.wm.uwb_control import measured_guard
class ControlTest(unittest.TestCase):
    def test_far_bearing_turn(self):
        a,info=measured_guard([.2,0.,0.],[5.,.4],[15,10,6.28],.048,.025)
        self.assertGreater(a[2],0);self.assertAlmostEqual(a[0],.2);self.assertEqual(info['heading_blend'],.5)
    def test_close_approach_limited(self):
        a,_=measured_guard([.2,0.,0.],[1.,0.],[15,10,6.28],.048,.025)
        self.assertAlmostEqual(a[0],0)
    def test_retreat_preserved(self):
        a,_=measured_guard([-.2,.1,0.],[1.,0.],[15,10,6.28],.048,.025)
        np.testing.assert_allclose(a[:2],[-.2,.1])
    def test_symmetric_turn(self):
        a,_=measured_guard([.2,0.,0.],[5.,.4],[15,10,6.28],.048,.025)
        b,_=measured_guard([.2,0.,0.],[5.,-.4],[15,10,6.28],.048,.025)
        self.assertAlmostEqual(a[2],-b[2])
if __name__=='__main__':unittest.main()
