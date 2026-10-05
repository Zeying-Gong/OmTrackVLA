import json
import tempfile
import unittest
from pathlib import Path
from wa.tools.report_dual_teacher import build_report
from wa.wm.dual_teacher_selection import EXPERIMENT

class TeacherReportGuards(unittest.TestCase):
    def test_incomplete_audit_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'audit.json'
            p.write_text(json.dumps(dict(experiment=EXPERIMENT,expected=24,paired_outcomes_validated=True)))
            with self.assertRaisesRegex(ValueError,'complete paired audit required'):
                build_report(p,Path(directory)/'not_read.json')
    def test_unvalidated_audit_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'audit.json'
            p.write_text(json.dumps(dict(experiment=EXPERIMENT,expected=4215,paired_outcomes_validated=False)))
            with self.assertRaisesRegex(ValueError,'complete paired audit required'):
                build_report(p,Path(directory)/'not_read.json')
    def test_changed_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'audit.json';m=Path(directory)/'manifest.json'
            p.write_text(json.dumps(dict(experiment=EXPERIMENT,expected=4215,paired_outcomes_validated=True)))
            m.write_text('{}')
            with self.assertRaisesRegex(ValueError,'official manifest changed'):
                build_report(p,m)
