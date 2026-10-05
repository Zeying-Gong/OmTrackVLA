import copy,hashlib,unittest
import numpy as np
from wa.wm.initial_bbox_repair import repaired_detector
class TestRepair(unittest.TestCase):
    def setUp(self):
        self.rgb=np.zeros((20,30,3),dtype=np.uint8)
        self.d={"agent_1_main_humanoid_detector_sensor":{"box":np.zeros(4),"facing":False}}
        self.r={"bbox":[1,2,10,15],"rgb_raw_sha256":hashlib.sha256(self.rgb.tobytes()).hexdigest()}
    def test_scoped_copy(self):
        x=repaired_detector(self.d,self.rgb,self.r)
        self.assertEqual(x["agent_1_main_humanoid_detector_sensor"]["box"].tolist(),[1,2,10,15])
        self.assertEqual(self.d["agent_1_main_humanoid_detector_sensor"]["box"].tolist(),[0]*4)
        self.assertFalse(x["agent_1_main_humanoid_detector_sensor"]["facing"])
    def test_mismatch(self):
        with self.assertRaises(ValueError):repaired_detector(self.d,self.rgb+1,self.r)
    def test_nonzero_rejected(self):
        self.d["agent_1_main_humanoid_detector_sensor"]["box"]=np.array([1,2,3,4])
        with self.assertRaises(ValueError):repaired_detector(self.d,self.rgb,self.r)
    def test_bounds(self):
        for box in ([1,1,40,15],[1,1,0,15],[1,1,10,float("nan")]):
            with self.subTest(box=box),self.assertRaises(ValueError):
                repaired_detector(self.d,self.rgb,dict(self.r,bbox=box))
class TestAgentBoundary(unittest.TestCase):
    def test_first_only_no_mutation(self):
        import sys,types,importlib
        from unittest.mock import patch
        class Parent:
            def __init__(self,*a,**kw):self.sim_step=0;self.seen=[]
            def act(self,obs,det,*args):
                self.seen.append((obs,det,args));self.sim_step+=1
                return [0.1,0.2,0.3]
        fake=types.ModuleType("wa.wm.diagnostic_agent");fake.DiagnosticAgent=Parent
        with patch.dict(sys.modules,{"wa.wm.diagnostic_agent":fake}):
            sys.modules.pop("wa.wm.initial_bbox_repair_agent",None)
            from wa.wm.initial_bbox_repair_agent import InitialBBoxRepairAgent
            rgb=np.zeros((20,30,3),dtype=np.uint8)
            r={"key":"pRbA3pwrgk9/27","bbox":[1,2,10,15],"rgb_raw_sha256":hashlib.sha256(rgb.tobytes()).hexdigest()}
            a=InitialBBoxRepairAgent(repair=r)
            d={"box":np.zeros(4),"facing":False};obs={"agent_1_articulated_agent_jaw_rgb":rgb}
            self.assertEqual(a.act(obs,d,"27",None,0),[.1,.2,.3])
            self.assertTrue(a.repair_applied);self.assertEqual(d["box"].tolist(),[0]*4)
            later={"box":np.ones(4)};obs2={"agent_1_articulated_agent_jaw_rgb":rgb+1}
            a.act(obs2,later,"27",None,.05)
            self.assertIs(a.seen[1][1],later);self.assertIs(a.seen[1][0],obs2)
            b=InitialBBoxRepairAgent(repair=r)
            with self.assertRaises(ValueError):b.act(obs,d,"38",None,0)
if __name__=="__main__":unittest.main()
