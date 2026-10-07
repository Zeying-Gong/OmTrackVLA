import copy,unittest
from wa.wm.student_eval_contract import model_contract,validate_student_rows,REPAIR_SHA
from wa.wm.initial_bbox_repair import KEYS,VERSION
from wa.wm.full_mixed_contract import validate_ready,CHECKPOINT_SHA,summarize
from wa.tests.test_full_mixed_contract import ContractTests

class StudentTests(unittest.TestCase):
    def setUp(self):
        self.env=dict(WA_STUDENT_EVAL='evaluation_set_adaptation_v1',
            WA_EVAL_CHECKPOINT_SHA='a'*64,WA_EVAL_CHECKPOINT_STEP='59065',
            WA_SEMANTIC_PLY_FIX='mp3d_semantic_ply_v1',WA_INIT_REPAIR_PLAN='/frozen/plan.json',
            WA_INIT_REPAIR_PLAN_SHA=REPAIR_SHA)
        self.contract=model_contract(self.env)
        self.rows=[dict(task=t,key=k,mode='mixed',checkpoint_sha256='a'*64,checkpoint_step=59065,
            semantic_protocol='mp3d_semantic_ply_v1',initialization_repair=VERSION,
            initialization_repair_plan_sha256=REPAIR_SHA) for t,k in sorted(KEYS)]

    def test_opt_in(self):
        self.assertEqual(model_contract({}),{})
        self.assertEqual(self.contract,dict(checkpoint_sha='a'*64,step=59065))

    def test_missing_and_wrong_fields(self):
        for key in self.env:
            env=self.env.copy();env.pop(key)
            with self.subTest(key=key),self.assertRaises(ValueError):model_contract(env)
        for key,value in [('WA_EVAL_CHECKPOINT_SHA',CHECKPOINT_SHA),
            ('WA_EVAL_CHECKPOINT_SHA','bad'),('WA_EVAL_CHECKPOINT_STEP','0'),
            ('WA_INIT_REPAIR_PLAN_SHA','b'*64),('WA_RESUME_PLAN','old'),('WA_TARGETED_PLAN','old')]:
            with self.subTest(key=key),self.assertRaises(ValueError):model_contract(dict(self.env,**{key:value}))

    def test_ready_explicit_identity(self):
        r=dict(checkpoint_sha256='a'*64,step=59065,mode='mixed',noise_mode='zero',
            sampling_steps=4,text_used=False,world_predictor_inference=False)
        validate_ready(r,**self.contract)
        with self.assertRaises(ValueError):validate_ready(r)
        with self.assertRaises(ValueError):validate_ready(dict(r,step=45900),**self.contract)

    def test_repaired_rows(self):validate_student_rows(self.rows,self.contract)

    def test_missing_duplicate_foreign(self):
        for rows in (self.rows[:-1],self.rows+[self.rows[0]]):
            with self.assertRaises(ValueError):validate_student_rows(rows,self.contract)
        for key,value in [('checkpoint_sha256',CHECKPOINT_SHA),('checkpoint_step',45900),
            ('semantic_protocol','old'),('initialization_repair',None),
            ('initialization_repair_plan_sha256','b'*64)]:
            rows=copy.deepcopy(self.rows);rows[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_student_rows(rows,self.contract)

    def test_outside_scope(self):
        row=dict(self.rows[0],key='other/1')
        with self.assertRaises(ValueError):validate_student_rows(self.rows+[row],self.contract)

    def test_summary_only_identity_changes(self):
        f=ContractTests();f.setUp()
        old=summarize(f.rows,f.manifest);new=summarize(f.rows,f.manifest,**self.contract)
        self.assertEqual(old['metrics_percent'],new['metrics_percent'])
        self.assertEqual(new['checkpoint_sha256'],'a'*64)
        self.assertEqual(new['checkpoint_step'],59065)

if __name__=='__main__':unittest.main()
