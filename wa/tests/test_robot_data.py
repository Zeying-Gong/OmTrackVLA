import json,tempfile,unittest
from pathlib import Path
import numpy as np
from wa.wm.robot_data import transition_records

class RobotTransitionTests(unittest.TestCase):
    def records(self,commands,dt=.048,jump=False):
        obs=[]
        for i in range(len(commands)+1):
            obs.append(dict(sim_step=i*3,timestamp_s=i*dt,robot_position_world=[i*(1. if jump else .1),0,0],robot_rotation_world_from_body=np.eye(3).tolist()))
        return obs,[dict(sim_step=i*3,normalized_action=c) for i,c in enumerate(commands)]
    def test_actual_time_step_matching_and_clipping(self):
        obs,actions=self.records([[2.,-2.,.2],[.1,.2,.3]])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'actions.json').write_text(json.dumps(list(reversed(actions))))
            x,bad,stats=transition_records(path,obs)
            np.testing.assert_allclose(x[0],[1,-1,.2,.48],atol=1e-6)
            self.assertFalse(bad.any());self.assertEqual(stats['clipped_command_steps'],1)
    def test_jump_rejected_and_duplicates_rejected(self):
        obs,actions=self.records([[0,0,0],[0,0,0]],jump=True)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'actions.json').write_text(json.dumps(actions))
            _,bad,_=transition_records(path,obs);self.assertTrue(bad.all())
            (path/'actions.json').write_text(json.dumps(actions+actions))
            with self.assertRaises(ValueError):transition_records(path,obs)

if __name__=='__main__':unittest.main()
