"""Small synthetic audit fixtures; never formal dataset admission."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from wa.tools.audit_dual_teacher import audit, digest
from wa.tests.test_dual_teacher_selection import branch
from wa.wm.dual_teacher_selection import select_teacher
from wa.wm.dual_teacher_partitions import workload

class PartitionAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.out=self.base/'merged';self.out.mkdir()
        self.rows={}
        for t in ('stt','dt','at'):
            for i in range(16):
                branches={}
                for name in ('lightnav','oracle'):
                    b=branch(name,False);b.update(task=t,key=str(i))
                    p=self.base/'artifacts'/t/str(i)/name;p.mkdir(parents=True)
                    (p/'result.json').write_text(json.dumps(b['result']))
                    (p/'windows.json').write_text('[]');b['artifact_root']=str(p)
                    branches[name]=b
                row=select_teacher(branches['lightnav'],branches['oracle'])
                row['branches']=branches;self.rows[t,str(i)]=row
        old=self.rows['stt','0']
        remaining=[dict(task=t,key=k) for t,k in self.rows if (t,k)!=('stt','0')]
        self.plan=dict(completed_rows=[old],remaining=remaining,lanes=[remaining[i::8] for i in range(8)],remaining_count=47)
        self.file=self.base/'plan.json';self.file.write_text(json.dumps(self.plan));self.sha=digest(self.file)
        self.roots=[]
        for j in range(2):
            dest=self.base/f'job{j}';self.roots.append(dest)
            for i,lane in enumerate(workload(self.plan,j,2)['lanes']):
                p=dest/f'lane{i}/collection';p.mkdir(parents=True)
                (p/'COMPLETE.json').write_text(json.dumps(dict(pairs=len(lane))))
                (p/'selections.jsonl').write_text(''.join(json.dumps(self.rows[r['task'],r['key']])+'\n' for r in lane))
    def run_audit(self, roots=None, partial=False):
        with patch('wa.wm.dual_teacher_resume.load_resume',return_value=self.plan):
            return audit(self.roots[0] if partial else self.out,48,str(self.file),self.sha,
                         0,2,None if partial else (roots or self.roots))
    def test_partial_not_full_release(self):
        result=self.run_audit(partial=True)
        self.assertEqual(result['expected'],24);self.assertTrue(result['partial_partition'])
        self.assertFalse(result['training_released'])
        self.assertFalse((self.roots[0]/'combined_selections.jsonl').exists())
    def test_complete_merge(self):
        result=self.run_audit()
        self.assertEqual(result['expected'],48);self.assertFalse(result['partial_partition'])
        self.assertEqual(len((self.out/'combined_selections.jsonl').read_text().splitlines()),48)
    def test_reversed_roots(self):
        with self.assertRaisesRegex(ValueError,'assignment|incomplete'):
            self.run_audit(self.roots[::-1])
    def test_duplicate_roots(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):self.run_audit([self.roots[0]]*2)
    def test_missing_row(self):
        p=self.roots[0]/'lane0/collection/selections.jsonl'
        p.write_text('\n'.join(p.read_text().splitlines()[1:])+'\n')
        with self.assertRaisesRegex(ValueError,'assignment'):self.run_audit()
    def test_missing_completion(self):
        (self.roots[0]/'lane0/collection/COMPLETE.json').unlink()
        with self.assertRaises(FileNotFoundError):self.run_audit()
    def test_bad_selection_does_not_write_combined(self):
        p=self.roots[0]/'lane0/collection/selections.jsonl'
        rows=[json.loads(s) for s in p.read_text().splitlines()]
        rows[0]['reason']='tampered';p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError,'selection'):self.run_audit()
        self.assertFalse((self.out/'combined_selections.jsonl').exists())
    def test_changed_source_rejected(self):
        target=str(self.roots[0]/'lane0/collection/selections.jsonl');calls=[]
        def changing(p):
            if str(p)==target:
                calls.append(p)
                if len(calls)>1:return 'changed'
            return digest(p)
        with patch('wa.tools.audit_dual_teacher.digest',side_effect=changing):
            with self.assertRaisesRegex(ValueError,'changed during'):self.run_audit()
        self.assertFalse((self.out/'combined_selections.jsonl').exists())

if __name__=='__main__':unittest.main()
