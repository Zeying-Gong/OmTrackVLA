"""Formal admission guards; file digests mocked, no cache generated."""
import unittest
from unittest.mock import patch
from wa.tools.build_dual_teacher_cache import validate_persisted_release

class CacheAdmissionTests(unittest.TestCase):
    def setUp(self):
        evidence={t+':'+str(i):{name:dict(start='ok',obs='ok',rgb='ok') for name in ('lightnav','oracle')}
                  for t in ('stt','dt','at') for i in range(1405)}
        self.m=dict(persisted_pair_evidence=evidence,persisted_pair_evidence_count=4215,
                    source_files={'source':'ok'},teacher_demonstrations=[dict(task='stt',key='0')])
    def run_guard(self, changed=None):
        with patch('wa.tools.build_dual_teacher_cache.sha',side_effect=lambda p:'bad' if p==changed else 'ok'):
            validate_persisted_release(self.m)
    def test_complete_evidence(self):self.run_guard()
    def test_missing_evidence(self):
        self.m.pop('persisted_pair_evidence')
        with self.assertRaisesRegex(ValueError,'full persisted'):self.run_guard()
    def test_missing_branch(self):
        self.m['persisted_pair_evidence']['stt:0'].pop('oracle')
        with self.assertRaisesRegex(ValueError,'missing teacher'):self.run_guard()
    def test_missing_file(self):
        self.m['persisted_pair_evidence']['stt:0']['oracle'].pop('rgb')
        with self.assertRaisesRegex(ValueError,'incomplete'):self.run_guard()
    def test_source_changed(self):
        with self.assertRaisesRegex(ValueError,'source changed'):self.run_guard('source')
    def test_start_changed(self):
        with self.assertRaisesRegex(ValueError,'pair evidence changed'):self.run_guard('rgb')
    def test_foreign_demonstration(self):
        self.m['teacher_demonstrations'][0]['key']='foreign'
        with self.assertRaisesRegex(ValueError,'missing paired'):self.run_guard()

if __name__=='__main__':unittest.main()
