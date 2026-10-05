"""Generate full teacher metrics only from a completed, unchanged paired audit."""
import argparse
import hashlib
import json
from pathlib import Path
from wa.wm.dual_teacher_metrics import summarize_teachers
from wa.wm.dual_teacher_resume import load_resume
from wa.wm.dual_teacher_selection import EXPERIMENT
from wa.wm.full_mixed_contract import MANIFEST_SHA

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build_report(audit_path, manifest_path):
    audit_path,manifest_path=Path(audit_path),Path(manifest_path)
    audit_sha=digest(audit_path)
    audit=json.loads(audit_path.read_text())
    if audit.get('experiment')!=EXPERIMENT or audit.get('expected')!=4215 or audit.get('paired_outcomes_validated') is not True:
        raise ValueError('complete paired audit required')
    if digest(manifest_path)!=MANIFEST_SHA:raise ValueError('official manifest changed')
    manifest=json.loads(manifest_path.read_text())
    rows=[]
    if not audit.get('source_files'):raise ValueError('missing audited sources')
    for filename,expected in audit['source_files'].items():
        path=Path(filename)
        if digest(path)!=expected:raise ValueError('audited source changed: '+filename)
        if path.name=='selections.jsonl':
            rows.extend(json.loads(s) for s in path.read_text().splitlines() if s.strip())
        else:
            plan=load_resume(path,expected,verify_artifacts=True)
            if plan['manifest_sha256']!=MANIFEST_SHA:raise ValueError('resume manifest mismatch')
            rows.extend(plan['completed_rows'])
    if len(rows)!=4215:raise ValueError('not a full collection')
    for row in rows:
        for branch in row['branches'].values():
            path=Path(branch['artifact_root'])/'result.json'
            if json.loads(path.read_text())!=branch['result']:
                raise ValueError('teacher result artifact changed: '+str(path))
    report=summarize_teachers(rows,manifest)
    # Recheck source snapshots after all reads, including the audit itself.
    for filename,expected in audit['source_files'].items():
        if digest(filename)!=expected:raise ValueError('source changed during report')
    if digest(audit_path)!=audit_sha:raise ValueError('audit changed during report')
    report.update(audit_path=str(audit_path),audit_sha256=audit_sha,
                  manifest_sha256=MANIFEST_SHA,source_files=audit['source_files'])
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--audit',required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    report=build_report(args.audit,args.manifest)
    with Path(args.output).open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report['metrics_percent'],indent=2))

if __name__=='__main__':main()
