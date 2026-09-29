import unittest
import numpy as np
from wa.wm.sim_uwb import polar_from_world,SimulatedUWB

class SensorTest(unittest.TestCase):
    def test_axes(self):
        for target,expect in [([2,0,0],[2,0]),([0,0,-2],[2,np.pi/2]),([0,0,2],[2,-np.pi/2])]:
            np.testing.assert_allclose(polar_from_world([0,0,0],np.eye(3),target),expect)
    def test_rotation_translation(self):
        rot=np.array([[0,0,1],[0,1,0],[-1,0,0]])
        np.testing.assert_allclose(polar_from_world([10,2,10],rot,[10,2,8]),[2,0])
    def test_no_world_state_in_packet(self):
        from types import SimpleNamespace as N
        transform=N(rotation=lambda:np.eye(3))
        env=N(sim=N(agents_mgr=[N(articulated_agent=N(base_pos=[2,0,-1])),N(articulated_agent=N(base_pos=[0,0,0],sim_obj=N(transformation=transform)))]))
        result=SimulatedUWB(env).sample(4.)
        self.assertEqual(set(result),{'polar','timestamp_s','valid','source','range_noise_m','bearing_noise_rad','delay_s'})
        self.assertEqual(result['timestamp_s'],4.)
    def test_bad_transform(self):
        with self.assertRaises(ValueError):polar_from_world([0,0,0],np.eye(3),[np.nan,0,0])
if __name__=='__main__':unittest.main()
