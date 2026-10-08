"""Reproduce the fixed old1/new35 failure-state raw search audit (CPU-only).

Reads only completed searches pinned by the terminal report. It never runs
models/simulation, changes old artifacts, admits training, or alters job status.
The output is created exclusively only after all36 raw searches pass.
"""
import argparse
import hashlib
import json
from pathlib import Path

TERMINAL_SHA = "96dd66c1f3ddbc12a97f6977c238ab482ae66529ed0a0fd48e69784076905d26"
DEFAULT_OUTPUT = Path("/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/failure_state_completed_search_audit_61836_v1.json")


def audit_completed():
    import json,hashlib,time,gzip,base64,sys
    from pathlib import Path
    from collections import Counter
    from datetime import datetime,timezone
    from wa.wm.failure_state_search_audit import audit_search
    from wa.wm.failure_state_branch_audit import audit_branch
    from wa.wm import failure_state_protocol as v1
    from wa.wm import failure_state_protocol_v2 as v2
    R=Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928');C=R/'checkout'
    def h(b):return hashlib.sha256(b).hexdigest()
    def pin(path,sha):
     b=Path(path).read_bytes();assert h(b)==sha,(str(path),'SHA mismatch');return b
    started=datetime.now(timezone.utc).isoformat();timer=time.monotonic()
    sp={str(C/'wa/wm'/n):h((C/'wa/wm'/n).read_bytes()) for n in ('failure_state_search_audit.py','failure_state_branch_audit.py','failure_state_protocol.py','failure_state_protocol_v2.py','failure_state_teacher_error.py','recovery_replay.py')}
    pp=R/'artifacts/failure_state_plan_61609_v1.json';ps='2eff9e83e89008ce1ee73632def406a27131fece6e77b01f6d4ca8c02624292b';p=json.loads(pin(pp,ps))
    op=R/'artifacts/failure_state_continuation_61833_v2.json';os='06d098f30ec52d988dc707249e73b17d11642f44846a75015455315891182822';o=json.loads(pin(op,os))
    assert o['base_plan']==dict(path=str(pp),sha256=ps)
    manifest=json.loads(pin(p['manifest']['path'],p['manifest']['sha256']))
    original={r['key']:r for r in map(json.loads,pin(p['sources']['combined']['path'],p['sources']['combined']['sha256']).splitlines()) if r['task']=='stt'}
    entries={e['key']:e for e in p['entries']}
    for k,e in entries.items():assert v1.canonical_sha(original[k])==e['baseline_row_sha256']
    tp=C/'wa/results/FAILURE_STATE_61836_TERMINAL_20261008.json';tb=pin(tp, TERMINAL_SHA);t=json.loads(tb);assert t['completed']==35
    items=[];old=o['frozen_records'][0];b=pin(old['records_path'],old['records_sha256']);row=json.loads(b);assert v1.canonical_sha(row)==old['row_sha256'];items.append((61833,73055,Path(old['records_path']),old['records_sha256'],row))
    for lane in t['records']:
     f=Path(t['root'])/f'lane{lane["lane"]}/collection/records.jsonl';rows=list(map(json.loads,pin(f,lane['sha']).splitlines()));assert [r['key'] for r in rows]==lane['keys']
     items.extend((61836,73058,f,lane['sha'],r) for r in rows)
    assert len(items)==36 and len({i[-1]['key'] for i in items})==36
    items.sort(key=lambda i:0 if i[0]==61833 else 1 if i[-1]['key']=='rJhMRvNn4DS/8' else 2)
    report=dict(schema='failure_state_completed_search_audit_61836_v1',base_plan=dict(path=str(pp),sha256=ps),previous_continuation=dict(path=str(op),sha256=os),terminal_report=dict(path=str(tp),sha256=h(tb)),manifest=p['manifest'],original_combined=p['sources']['combined'],started_utc=started,source_sha256=sp,searches=[],branches=[],failures=[],training_released=False,score_backfill_allowed=False,limitations=['36 completed searches only, not126 wholecollection admission','interrupted keys excluded','no new student score; numeric cache conversion and wholecollection admission remain separate','rawRGB retained as decoded PNG plus recorded hash; hidden RNG/contact state not proven'],initial_call_rejection=dict(reason='optional scene_id exact string pin mismatch',plan_form='hm3d/...',Habitat_form='data/scene_datasets/hm3d/...',resolution='omit optional scene_id; preserve exact dataset SHA, episode key and original61609 initial state checks'))
    tot=Counter();allpins={}
    print(json.dumps(dict(phase='START',utc=started,expected=36,source_sha256=sp)),flush=True)
    def raw(root,**kw):
     g=audit_branch(root,**kw);a=g['actualproof'];pins=g['source_files']
     stat=dict(root=str(root),kind=g['kind'],teacher=kw['expected']['teacher'],k=kw['expected']['takeover_step'],repeat=kw['expected']['verification_only'],frames=a['observed_frames'],actions=a['recorded_actions'],candidate_windows=g['candidate_windows'],files=len(pins),pngs=sum(p.endswith('.png') for p in pins),source_inventory_sha256=v1.canonical_sha(pins))
     report['branches'].append(stat);tot.update(dict(branches=1,frames=stat['frames'],actions=stat['actions'],pngs=stat['pngs'],all_branch_candidate_windows=stat['candidate_windows'],teacher_errors=int(g['kind']=='teacher_error')))
     print(json.dumps(dict(phase='BRANCH',index=tot['branches'],root=str(root),frames=stat['frames'],files=stat['files'])),flush=True)
     return g
    for seq,(job,task,f,fh,row) in enumerate(items,1):
     key=row['key'];e=entries[key];exp=row['experiment'];ph=v1.PROTOCOL_SHA if exp==v1.EXPERIMENT else v2.PROTOCOL_SHA
     md=dict(experiment=exp,partition='evaluation_adaptation',protocol_sha256=ph,plan_sha256=ps,checkpoint_sha256=p['checkpoint']['sha256'],checkpoint_step=p['checkpoint']['step'],source_dataset_sha256=manifest['tasks']['stt']['sha256'],seed=7,episode_id=e['episode_id'])
     if exp==v2.EXPERIMENT:md.update(base_plan_sha256=ps,base_protocol_sha256=v1.PROTOCOL_SHA,continuation_sha256=os)
     try:
      got=audit_search(row,e,f.parent/'stt'/key,original_start=original[key]['initial_pair_evidence'],expected_metadata=md,audit_branch=raw)
      for path,digest in got['source_files'].items():assert path not in allpins or allpins[path]==digest;allpins[path]=digest
      report['searches'].append(dict(job_id=job,task_id=task,records_path=str(f),records_sha256=fh,row=row,audit=got))
      candidate=got['candidate'];tot.update(dict(searches_pass=1,accepted_candidates=int(candidate is not None),accepted_candidate_windows=0 if candidate is None else candidate['candidate_windows']))
      print(json.dumps(dict(phase='SEARCH_PASS',seq=seq,key=key,outcome=got['outcome'],branches=got['branch_count'],files=len(got['source_files']),accepted_windows=0 if candidate is None else candidate['candidate_windows'])),flush=True)
     except Exception as exc:
      report['failures'].append(dict(key=key,type=type(exc).__name__,message=str(exc)));print(json.dumps(dict(phase='FAIL',**report['failures'][-1])),flush=True);break
    unchanged=sp=={path:h(Path(path).read_bytes()) for path in sp}
    report.update(execution=dict(kind='repository_generator',path=str(Path(__file__).resolve()),sha256=h(Path(__file__).read_bytes())),ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_s=time.monotonic()-timer,source_unchanged=unchanged,totals=dict(tot),source_evidence_count=len(allpins),source_evidence_inventory_sha256=v1.canonical_sha(allpins),status='PASS_NONRELEASE' if not report['failures'] and len(report['searches'])==36 and unchanged else 'INCOMPLETE_OR_FAIL')

    return report


def save_report(report, output):
    """Exclusive creation of an already-computed, complete nonrelease report."""
    output = Path(output)
    if not output.is_absolute() or not str(output).startswith("/data/nas_ray/"):
        raise ValueError("new persistent NAS output required")
    if not output.parent.is_dir() or output.exists() or output.is_symlink():
        raise ValueError("output parent missing or target already exists")
    if (report.get("schema") != "failure_state_completed_search_audit_61836_v1"
            or report.get("status") != "PASS_NONRELEASE"
            or report.get("source_unchanged") is not True
            or report.get("failures") != []
            or report.get("training_released") is not False
            or report.get("score_backfill_allowed") is not False):
        raise ValueError("complete nonrelease audit required")
    searches = report.get("searches", [])
    if (len(searches) != 36 or
            len({s["row"]["key"] for s in searches}) != 36 or
            sum(s["job_id"] == 61833 for s in searches) != 1 or
            sum(s["job_id"] == 61836 for s in searches) != 35):
        raise ValueError("exact fixed old1 plus new35 required")
    if any(s["audit"]["training_released"] is not False or
           s["audit"]["score_backfill_allowed"] is not False for s in searches):
        raise ValueError("search audit has unsupported release")
    for filename, digest in report["source_sha256"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise ValueError("auditor source changed since computation")
    core_fields = ("schema", "base_plan", "previous_continuation", "terminal_report",
                   "searches", "training_released", "score_backfill_allowed")
    core = {key: report[key] for key in core_fields}
    payload = json.dumps(core, sort_keys=True, allow_nan=False,
                         separators=(",", ":")).encode()
    digest = hashlib.sha256(payload).hexdigest()
    provenance_path = output.with_suffix(".provenance.json")
    if provenance_path.exists() or provenance_path.is_symlink():
        raise ValueError("provenance already exists; never overwrite")
    provenance = {key: value for key, value in report.items() if key not in core_fields}
    provenance.update(schema="failure_state_completed_search_audit_provenance_v1",
                      report_path=str(output), report_sha256=digest,
                      report_bytes=len(payload), preserved_source_job_status="FAILED",
                      training_released=False, score_backfill_allowed=False)
    provenance_payload = json.dumps(provenance, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()
    with output.open("xb") as stream:
        stream.write(payload)
    with provenance_path.open("xb") as stream:
        stream.write(provenance_payload)
    return dict(path=str(output), bytes=len(payload), sha256=digest,
                provenance_path=str(provenance_path), provenance_bytes=len(provenance_payload),
                provenance_sha256=hashlib.sha256(provenance_payload).hexdigest(),
                searches=36, status=report["status"], training_released=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if (args.output.exists() or args.output.is_symlink()
            or args.output.with_suffix(".provenance.json").exists()
            or args.output.with_suffix(".provenance.json").is_symlink()):
        raise ValueError("output already exists; never overwrite a prior audit")
    report = audit_completed()
    print(json.dumps(save_report(report, args.output), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
