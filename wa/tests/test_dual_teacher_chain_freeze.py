"""Tests freeze chaining/guards; outcome validation is mocked separately."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from wa.tools.freeze_dual_teacher_resume import freeze
from wa.wm.dual_teacher_resume import digest, load_resume

class ChainFreezeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.root=self.base/'job_123'/'out'
        p=self.root/'lane0/collection';p.mkdir(parents=True)
        self.row=dict(pair=dict(task='stt',key='1'),branches={})
        (p/'selections.jsonl').write_text(json.dumps(self.row)+'\n')
        manifest={'tasks':{}}
        for t in ('stt','dt','at'):
            data=self.base/(t+'.json');data.write_text('{}')
            manifest['tasks'][t]=dict(path=str(data),sha256=digest(data),episodes=[dict(key=str(i)) for i in range(1405)])
        self.manifest=self.base/'manifest.json';self.manifest.write_text(json.dumps(manifest))
        self.prior=dict(manifest_sha256=digest(self.manifest),completed_rows=[dict(pair=dict(task='stt',key='0'),branches={})],
                        remaining=[dict(task='stt',key='1')])
        self.previous=self.base/'previous.json';self.previous.write_text(json.dumps(self.prior))
        self.output=self.base/'frozen.json'
    def run_freeze(self,status='STOPPED'):
        with patch('wa.tools.freeze_dual_teacher_resume.subprocess.check_output',return_value='Status        : '+status), \
             patch('wa.tools.freeze_dual_teacher_resume.audit',return_value={'mock_outcomes_only':True}), \
             patch('wa.wm.dual_teacher_resume.load_resume',return_value=self.prior):
            return freeze(self.root,self.manifest,self.output,123,str(self.previous),digest(self.previous))
    def test_chain_cover_and_hash(self):
        result=self.run_freeze();self.assertEqual(result['completed'],2);self.assertEqual(result['remaining'],4213)
        new=load_resume(self.output,result['sha256'],verify_artifacts=True)
        self.assertEqual(new['new_completed_count'],1)
        self.assertEqual(new['prior_plan_sha256'],digest(self.previous))
    def test_live_job_rejected(self):
        with self.assertRaisesRegex(ValueError,'STOPPED'):self.run_freeze('RUNNING')
        self.assertFalse(self.output.exists())
    def test_completed_duplicate_rejected(self):
        self.prior['completed_rows'][0]['pair']['key']='1'
        with self.assertRaisesRegex(ValueError,'duplicate'):self.run_freeze()
    def test_foreign_remainder_rejected(self):
        self.prior['remaining']=[dict(task='at',key='1')]
        with self.assertRaisesRegex(ValueError,'outside'):self.run_freeze()
    def test_manifest_mismatch(self):
        self.prior['manifest_sha256']='bad'
        with self.assertRaisesRegex(ValueError,'manifest'):self.run_freeze()
    def test_no_overwrite(self):
        self.run_freeze();before=digest(self.output)
        with self.assertRaises(FileExistsError):self.run_freeze()
        self.assertEqual(digest(self.output),before)

if __name__=='__main__':unittest.main()
