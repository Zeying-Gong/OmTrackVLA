import unittest
from types import SimpleNamespace as NS
from wa.wm.oracle_teacher import OracleTeacher

class OracleBridgeTests(unittest.TestCase):
    def make(self, action=(.2,-.1,.3)):
        class Perception:
            def reset(self):pass
            def __call__(self,*args):return args
        class Controller:
            def __init__(self,**kwargs):pass
            def reset(self):pass
            def __call__(self,*args):return NS(action=NS(as_habitat=lambda:list(action)))
        m=NS(OraclePerception=Perception,OracleNavmeshFollower=Controller,
             RGB_KEY="rgb",PANOPTIC_KEY="semantic",local_target=lambda r,h:(1.,0.))
        a=OracleTeacher(m)
        env=NS(current_episode=NS(episode_id="1",info={"main_human_semantic_id":3}),
               sim=NS(agents_mgr=[NS(articulated_agent="human"),NS(articulated_agent="robot")]))
        return a,env
    def test_binding_and_action(self):
        a,e=self.make()
        with self.assertRaises(RuntimeError):a.act({}, {}, "1")
        a.bind_environment(e);a.reset()
        self.assertEqual(a.act({"rgb":0,"semantic":0},{},1),[.2,-.1,.3])
        self.assertIsNone(a.last_trajectory)
    def test_episode(self):
        a,e=self.make();a.bind_environment(e)
        with self.assertRaises(RuntimeError):a.act({},{},"2")
    def test_invalid_action(self):
        for action in ((2,0,0),(float("nan"),0,0),(0,0)):
            a,e=self.make(action);a.bind_environment(e)
            with self.assertRaises(RuntimeError):a.act({"rgb":0,"semantic":0},{},1)
if __name__=="__main__":unittest.main()
