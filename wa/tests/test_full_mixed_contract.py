"""CPU checks: strict weights/mode and complete4215 metric accounting."""
import copy,unittest
from wa.wm.full_mixed_contract import CHECKPOINT_SHA,validate_ready,summarize,TASKS

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.ready=dict(checkpoint_sha256=CHECKPOINT_SHA,step=45900,mode='mixed',noise_mode='zero',
                        sampling_steps=4,text_used=False,world_predictor_inference=False)
        self.manifest={'tasks':{t:{'episodes':[{'key':str(i)} for i in range(1405)]} for t in TASKS},'reference_steps':{}}
        self.rows=[dict(task=t,key=str(i),mode='mixed',noise_mode='zero',controller='learned_yaw_guard_v1',
            success=1,collision=0,following_rate=1,following_step=10,total_step=10,finish=True,policy_init_valid=True)
            for t in TASKS for i in range(1405)]
    def test_ready(self):validate_ready(self.ready)
    def test_ready_rejects_wrong_fields(self):
        for k,v in [('checkpoint_sha256','bad'),('step',22707),('mode','image'),('noise_mode','random'),('text_used',True),('world_predictor_inference',True)]:
            with self.subTest(k=k),self.assertRaises(ValueError):validate_ready(dict(self.ready,**{k:v}))
    def test_complete(self):
        r=summarize(self.rows,self.manifest)
        self.assertEqual(r['episodes'],4215)
        for t in TASKS:self.assertEqual(r['metrics_percent'][t]['SR'],100)
    def test_duplicate(self):
        self.rows[-1]=self.rows[0]
        with self.assertRaises(ValueError):summarize(self.rows,self.manifest)
    def test_missing(self):
        with self.assertRaises(ValueError):summarize(self.rows[:-1],self.manifest)
    def test_wrong_key(self):
        self.rows[0]['key']='wrong'
        with self.assertRaises(ValueError):summarize(self.rows,self.manifest)
    def test_invalid_retained(self):
        self.rows[0].update(policy_init_valid=False,success=0,following_rate=0,following_step=0)
        r=summarize(self.rows,self.manifest)['metrics_percent']['stt']
        self.assertEqual(r['episodes'],1405);self.assertEqual(r['invalid_init_count'],1)
        self.assertAlmostEqual(r['SR'],1404/1405*100)
    def test_nonfinite(self):
        self.rows[0]['following_rate']=float('nan')
        with self.assertRaises(ValueError):summarize(self.rows,self.manifest)
    def test_wrong_controller(self):
        self.rows[0]['controller']='learned_target_guard_v3'
        with self.assertRaises(ValueError):summarize(self.rows,self.manifest)
    def test_fixed_shards_across_slots(self):
        for gpus in range(1,9):
            assigned=[i for slot in range(gpus) for i in range(slot,8,gpus)]
            self.assertEqual(sorted(assigned),list(range(8)))
if __name__=='__main__':unittest.main()
