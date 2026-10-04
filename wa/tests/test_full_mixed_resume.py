import copy,unittest
from wa.wm.full_mixed_resume import validate_plan,verify_new_rows
from wa.wm.full_mixed_contract import MANIFEST_SHA,TASKS
class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.m={'tasks':{t:{'episodes':[{'key':str(i)} for i in range(1405)]} for t in TASKS}}
        self.p=dict(manifest_sha256=MANIFEST_SHA,completed_rows=[],source_files=[],lanes=[[] for _ in range(8)])
        for j,(t,k) in enumerate((t,str(i)) for t in TASKS for i in range(1405)):
            self.p['lanes'][j%8].append(dict(task=t,key=k))
    def test_all_pending(self):validate_plan(self.p,self.m)
    def test_reuse_and_balance(self):
        e=self.p['lanes'][0].pop(0)
        self.p['completed_rows']=[dict(e,mode='mixed',noise_mode='zero',controller='learned_yaw_guard_v1',policy_init_valid=False,success=0,collision=0,following_rate=0,following_step=0,total_step=0,initial_rgb_sha256='a'*64)]
        validate_plan(self.p,self.m)
    def test_duplicate(self):
        self.p['lanes'][1].append(self.p['lanes'][0][0])
        with self.assertRaises(AssertionError):validate_plan(self.p,self.m)
    def test_missing(self):
        self.p['lanes'][0].pop()
        with self.assertRaises(AssertionError):validate_plan(self.p,self.m)
    def test_wrong_key(self):
        self.p['lanes'][0][0]['key']='wrong'
        with self.assertRaises(AssertionError):validate_plan(self.p,self.m)
    def test_new_rows(self):
        rows=[r for lane in self.p['lanes'] for r in lane];verify_new_rows(rows,self.p)
        with self.assertRaises(AssertionError):verify_new_rows(rows[:-1],self.p)
        with self.assertRaises(AssertionError):verify_new_rows(rows+[rows[0]],self.p)
if __name__=='__main__':unittest.main()
