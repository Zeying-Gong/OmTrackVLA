import copy,json,unittest
from unittest.mock import patch
from wa.wm.semantic_targeted import validate_targeted,PLAN_VERSION
from wa.wm.semantic_scene import VERSION,provenance
from wa.wm.full_mixed_contract import MANIFEST_SHA,TASKS
class TargetedTests(unittest.TestCase):
    def setUp(self):
        self.manifest={"tasks":{}}
        rows=[];lanes=[[] for _ in range(8)];prior=[]
        for t in TASKS:
            entries=[]
            for i in range(1405):
                k=str(i);affected=i<721
                entries.append(dict(key=k,scene_id=("mp3d/s/s.glb" if affected else "hm3d/x/x.basis.glb")))
                if affected:
                    e=dict(task=t,key=k,priority=i<55)
                    lanes[min(range(8),key=lambda j:len(lanes[j]))].append(e)
                    if i<55:prior.append((t,k))
                else:
                    rows.append(dict(task=t,key=k,mode="mixed",noise_mode="zero",controller="learned_yaw_guard_v1",
                        policy_init_valid=True,success=1,collision=0,following_rate=1,following_step=10,total_step=10,
                        initial_rgb_sha256="a"*64,artifact_root="/test",reuse_reason="unaffected_hm3d_baseline"))
            self.manifest["tasks"][t]={"episodes":entries}
        for lane in lanes:lane.sort(key=lambda e:not e["priority"])
        self.plan=dict(version=PLAN_VERSION,manifest_sha256=MANIFEST_SHA,semantic_config_sha256=provenance()["config_sha256"],
            prior_invalid_keys=prior,completed_rows=rows,lanes=lanes,source_files=[])
    def test_valid_full_scope(self):
        validate_targeted(self.plan,self.manifest)
        self.assertEqual(sum(map(len,self.plan["lanes"])),2163)
    def test_old_mp3d_reuse_rejected(self):
        e=self.plan["lanes"][0].pop()
        r=copy.deepcopy(self.plan["completed_rows"][0]);r.update(task=e["task"],key=e["key"])
        self.plan["completed_rows"].append(r)
        with self.assertRaises((AssertionError,KeyError)):validate_targeted(self.plan,self.manifest)
    def test_missing_key_rejected(self):
        self.plan["lanes"][0].pop()
        with self.assertRaises(AssertionError):validate_targeted(self.plan,self.manifest)
    def test_duplicate_key_rejected(self):
        self.plan["lanes"][1].append(self.plan["lanes"][0][0])
        with self.assertRaises(AssertionError):validate_targeted(self.plan,self.manifest)
    def test_wrong_priority_rejected(self):
        self.plan["lanes"][0][0]["priority"]=False
        with self.assertRaises(AssertionError):validate_targeted(self.plan,self.manifest)
    def test_priority_order_rejected(self):
        self.plan["lanes"][0].reverse()
        with self.assertRaises(AssertionError):validate_targeted(self.plan,self.manifest)
    def test_config_mismatch_rejected(self):
        self.plan["semantic_config_sha256"]="bad"
        with self.assertRaises(AssertionError):validate_targeted(self.plan,self.manifest)
    def test_repaired_completed_accepted(self):
        e=self.plan["lanes"][0].pop()
        r=copy.deepcopy(self.plan["completed_rows"][0]);r.update(task=e["task"],key=e["key"],
           reuse_reason="already_completed_repaired",semantic_protocol=VERSION,semantic_ply_repaired=True)
        self.plan["completed_rows"].append(r);validate_targeted(self.plan,self.manifest)
if __name__=="__main__":unittest.main()
