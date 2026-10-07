import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from wa.wm.student_eval_finalize import load_teachers,validate_shard,superiority,finalize

class FinalizeTests(unittest.TestCase):
    def setUp(self):
        self.rows=[];self.teachers=[]
        self.manifest=dict(tasks={t:dict(episodes=[]) for t in ('stt','dt','at')})
        for task in ('stt','dt','at'):
            for i in range(1405):
                key='scene/'+str(i)
                self.rows.append(dict(task=task,key=key,mode='mixed',success=int(i<1100)))
                self.teachers.append(dict(pair=dict(task=task,key=key),
                    results=dict(lightnav=dict(success=int(i<1000)))))
                self.manifest['tasks'][task]['episodes'].append(dict(key=key,shard=i%8))
    def test_strict_each_task(self):
        self.assertTrue(superiority(self.rows,self.teachers)['all_three_strictly_exceed'])
        for r in self.rows:
            if r['task']=='stt':r['success']=int(int(r['key'].split('/')[1])<1000)
        result=superiority(self.rows,self.teachers)
        self.assertFalse(result['all_three_strictly_exceed'])
        self.assertFalse(result['tasks']['stt']['strictly_exceeds'])
    def test_missing_full_task(self):
        with self.assertRaises(ValueError):superiority(self.rows[:-1],self.teachers)
    def test_shard(self):
        rows=[r for r in self.rows if int(r['key'].split('/')[1])%8==0]
        validate_shard(rows,dict(episodes=len(rows)),self.manifest,0)
        for wrong in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError):validate_shard(wrong,dict(episodes=len(wrong)),self.manifest,0)
        with self.assertRaises(ValueError):validate_shard(rows,dict(episodes=0),self.manifest,0)
    def test_reference_hash_reject(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'wrong.jsonl';path.write_text('{}')
            with self.assertRaises(ValueError):load_teachers(path)
    def test_finalize_calls_pair_audit_and_rechecks_reference(self):
        with patch('wa.wm.student_eval_finalize.load_teachers',return_value=self.teachers) as load,patch('wa.wm.student_eval_finalize.summarize_teachers',return_value=dict(metrics_percent={})) as metrics,patch('wa.wm.student_eval_finalize.audit_starts',return_value=dict(status='PASS')) as audit:
            report,evidence=finalize(self.rows,self.manifest,'frozen')
            self.assertEqual(load.call_count,2)
            self.assertEqual(len(audit.call_args.args[2]),4215)
            self.assertTrue(report['superiority']['all_three_strictly_exceed'])
            self.assertEqual(evidence['status'],'PASS')
    def test_pair_failure_propagates(self):
        with patch('wa.wm.student_eval_finalize.load_teachers',return_value=self.teachers),patch('wa.wm.student_eval_finalize.summarize_teachers',return_value=dict(metrics_percent={})),patch('wa.wm.student_eval_finalize.audit_starts',side_effect=ValueError('different RGB')):
            with self.assertRaises(ValueError):finalize(self.rows,self.manifest,'frozen')

if __name__=='__main__':unittest.main()
