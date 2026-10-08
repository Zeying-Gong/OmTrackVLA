"""CPU-only synthetic pinned evidence; no model, Habitat, NAS or scheduler work."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wa.wm import failure_state_continuation_61836 as m
from wa.wm import failure_state_continuation as prev
from wa.wm.failure_state_protocol import canonical_sha


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.run = self.root/"new61836"
        self.run.mkdir()
        self.oldrun = self.root/"old61833"
        self.oldrun.mkdir()
        entries = [dict(task="stt", key=prev.KEY if i == 0 else f"scene{i}/1",
                        baseline_result=dict(success=0.0, collision=1.0, policy_init_valid=True),
                        baseline_row_sha256=hashlib.sha256(str(i).encode()).hexdigest())
                   for i in range(126)]
        self.plan = dict(entries=entries, lanes=[entries[i::8] for i in range(8)])
        self.basepath = self.root/"base.json"
        self.base_sha = self.write(self.basepath, self.plan)
        self.oldpath = self.root/"oldoverlay.json"
        oldremaining = [e for e in entries if e["key"] != prev.KEY]
        self.old = dict(base_plan=dict(path=str(self.basepath), sha256=self.base_sha),
                        expected_count=126, new_count=125, reused_count=1,
                        reused_keys=[prev.KEY], training_released=False, remaining_entries=oldremaining,
                        remaining_lanes=[[e for e in lane if e["key"] != prev.KEY] for lane in self.plan["lanes"]])
        # Avoid circular fixture SHA dependencies: old overlay's mocked loader
        # returns the independently built old-one object; its bytes stay pinned.
        self.prev_sha = self.write(self.oldpath, {"fixture_old_overlay": True})
        self.terminalpath = self.root/"terminal.json"
        patches = dict(BASE_SHA=self.base_sha, PREVIOUS_SHA=self.prev_sha, SOURCE_ROOT=self.run)
        self.addCleanup(patch.stopall)
        patch.multiple(m, **patches).start()
        patch.object(m, "load_plan", side_effect=lambda *_:self.plan).start()
        patch.object(m.previous, "load_continuation", side_effect=lambda *_:self.old).start()
        self.searches, terminal_records = [], []
        oldrow, oldaudit = self.make_search(self.oldrun/"lane0/collection", entries[0], 61833, candidate=True)
        oldrecords = self.oldrun/"lane0/collection/records.jsonl"
        oldsha = self.lines(oldrecords, [oldrow])
        self.old["frozen_records"] = [dict(key=prev.KEY, records_path=str(oldrecords),
                                          records_sha256=oldsha, row_sha256=canonical_sha(oldrow))]
        self.searches.append(dict(job_id=61833, task_id=73055, records_path=str(oldrecords),
                                  records_sha256=oldsha, row=oldrow, audit=oldaudit))
        n = 0
        for lane, count in enumerate(m.COMPLETED_LANE_COUNTS):
            collection = self.run/f"lane{lane}/collection"
            rows, audits = [], []
            for entry in self.old["remaining_lanes"][lane][:count]:
                row, audit = self.make_search(collection, entry, 61836, candidate=n < 23)
                n += 1
                rows.append(row); audits.append(audit)
            recordspath = collection/"records.jsonl"
            digest = self.lines(recordspath, rows)
            terminal_records.append(dict(lane=lane, sha=digest, keys=[r["key"] for r in rows]))
            self.searches.extend(dict(job_id=61836, task_id=73058, records_path=str(recordspath),
                                      records_sha256=digest, row=r, audit=a) for r,a in zip(rows,audits))
            self.make_lane(lane)
        root_error = dict(error_type="RuntimeError",
                          message="collector failed; preserve original logs and partial outputs",
                          training_released=False, no_success_rate=True)
        root_error_sha = self.write(self.run/"ERROR.json", root_error)
        self.write(self.run/"launch.json", dict(experiment=m.V2, source="/frozen/source",
              source_commit=m.SOURCE_COMMIT, source_git_status="", plan=str(self.basepath),
              plan_sha256=self.base_sha, continuation=str(self.oldpath), continuation_sha256=self.prev_sha,
              protocol_sha256=m.V2_SHA, formal=True, devices=[str(i) for i in range(8)],
              expected_entries=125, expected_lanes=8, frozen_entries=1,total_expected_entries=126,
              training_released=False,no_success_rate=True))
        self.terminal = dict(schema="wa_failure_state_terminal_evidence_v1",job=61836,task=73058,
             status="FAILED",root=str(self.run),completed=35,lanes=m.COMPLETED_LANE_COUNTS,
             outcomes=m.OUTCOMES,root_error_sha256=root_error_sha,records=terminal_records,
             cleanup=dict(role_exit_files=24,cleanup_errors=False,complete_markers=0,
                          worker_lane4=1,worker_other=-15,lightnav=0,wa=-15),
             training_released=False,new_model_sr=False,new_job_submitted=False)
        self.terminal_sha = self.write(self.terminalpath,self.terminal)
        patch.object(m,"TERMINAL_SHA",self.terminal_sha).start()
        self.report = dict(schema=m.AUDIT_SCHEMA,base_plan=dict(path=str(self.basepath),sha256=self.base_sha),
             previous_continuation=dict(path=str(self.oldpath),sha256=self.prev_sha),
             terminal_report=dict(path=str(self.terminalpath),sha256=self.terminal_sha),
             searches=self.searches,training_released=False,score_backfill_allowed=False)
        self.reportpath = self.root/"audit.json"
        self.refresh()

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        blob=(json.dumps(value,indent=2,allow_nan=False)+"\n").encode()
        path.write_bytes(blob)
        return hashlib.sha256(blob).hexdigest()

    def lines(self,path,values):
        path.parent.mkdir(parents=True,exist_ok=True)
        blob="".join(json.dumps(v,allow_nan=False)+"\n" for v in values).encode()
        path.write_bytes(blob)
        return hashlib.sha256(blob).hexdigest()

    def refresh(self):
        self.report_sha=self.write(self.reportpath,self.report)

    def make_lane(self,lane):
        dest=self.run/f"lane{lane}"
        self.write(dest/"ERROR.json",dict(error_type="RuntimeError",message="failure",
                                         training_released=False,no_success_rate=True))
        self.write(dest/"launch.json",dict(experiment=m.V2,plan_sha256=self.base_sha,protocol_sha256=m.V2_SHA,
             continuation_sha256=self.prev_sha,lane=lane,device=str(lane),
             keys=[e["key"] for e in self.old["remaining_lanes"][lane]],training_released=False,no_success_rate=True))
        events=[]
        for i,role in enumerate(("wa","lightnav","worker")):
            pid=100+lane*3+i
            events.append(dict(event="spawn",role=role,pid=pid,process_group=pid,command={},utc="fixture"))
        if lane==4:
            events.append(dict(event="worker_return",role="worker",pid=114,returncode=1,utc="fixture"))
        for i,role in enumerate(("wa","lightnav","worker")):
            events.append(dict(event="exit",role=role,pid=100+lane*3+i,
                              returncode=0 if role=="lightnav" else 1 if lane==4 and role=="worker" else -15,utc="fixture"))
        self.lines(dest/"processes.jsonl",events)

    def make_search(self,collection,entry,job,candidate):
        exp,protocol=(prev.EXPERIMENT,prev.PROTOCOL_SHA) if job==61833 else (m.V2,m.V2_SHA)
        base=collection/"stt"/entry["key"]
        prefix=[dict(step=i) for i in range(8)]
        student=base/"student"
        result=dict(success=0.0,collision=1.0,policy_init_valid=True,following_rate=.2)
        self.write(student/"replay.json",prefix)
        self.write(student/"result.json",result)
        row=dict(experiment=exp,protocol_sha256=protocol,plan_sha256=self.base_sha,task="stt",
                 key=entry["key"],baseline_result_unchanged=entry["baseline_result"],
                 student=dict(artifact_root=str(student),result=result),attempts=[],
                 accepted=None,training_released=False,score_backfill_allowed=False)
        if job==61836:
            row.update(base_protocol_sha256=prev.PROTOCOL_SHA,continuation_sha256=self.prev_sha)
        for k in ([3] if candidate else [3,0]):
            branches={}
            for teacher in ("lightnav","oracle"):
                success=candidate and teacher=="oracle"
                root=base/f"{teacher}_{k:04d}"
                windows=[dict(current_index=k)] if success else []
                result=dict(success=float(success),collision=float(not success),policy_init_valid=True,
                            following_rate=.8 if success else .2)
                branch=dict(teacher=teacher,artifact_root=str(root),complete=True,replay_verified=True,
                            transport_fallback=False,verification_only=False,candidate_windows=len(windows),result=result)
                self.write(root/"branch.json",branch);self.write(root/"windows.json",windows)
                branches[teacher]=branch
            flags=dict(lightnav=False,oracle=candidate)
            selection=dict(selected_teacher="oracle" if candidate else None,eligible=flags,training_released=False)
            attempt=dict(takeover_step=k,branches=branches,selection=selection)
            if candidate:
                root=base/f"oracle_{k:04d}_repeat"
                repeat=dict(branches["oracle"],artifact_root=str(root),verification_only=True,candidate_windows=0)
                self.write(root/"branch.json",repeat);self.write(root/"windows.json",[])
                attempt.update(repeat=repeat,repeat_valid=True,selected_candidate_windows=1)
                row["accepted"]=dict(teacher="oracle",takeover_step=k,artifact_root=str(base/f"oracle_{k:04d}"),
                      repeat_artifact_root=str(root),candidate_only=True,training_released=False)
            row["attempts"].append(attempt)
        row["outcome"]="repeated_teacher_recovery_candidate" if candidate else "no_valid_teacher_recovery"
        self.write(base/"search.json",row)
        files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in base.rglob("*") if p.is_file()}
        accepted=None
        if candidate:
            accepted=dict(task="stt",key=entry["key"],teacher="oracle",takeover_step=3,
               branch=str(base/"oracle_0003"),repeat_branch=str(base/"oracle_0003_repeat"),
               window_indices=[3],candidate_windows=1,experiment=exp,protocol_sha256=protocol,
               source_files=files,training_released=False,conversion_pending=True)
        audit=dict(schema="failure_state_search_audit_v1",task="stt",key=entry["key"],outcome=row["outcome"],
              experiment=exp,protocol_sha256=protocol,candidate=accepted,source_files=files,
              branch_count=4 if candidate else 5,sidecar_files={},sidecar_media_semantics_audited=False,
              training_released=False,score_backfill_allowed=False)
        return row,audit

    def build(self,**kwargs):
        return m.build_continuation(self.basepath,self.base_sha,self.oldpath,self.prev_sha,
                                   self.reportpath,self.report_sha,expected_job_status=kwargs.get("status","FAILED"))

    def test_roundtrip_exact36_90_original_lanes_and_numeric_types(self):
        before=copy.deepcopy(self.plan); raw=self.basepath.read_bytes()
        out=self.build()
        self.assertEqual((out["expected_count"],out["new_count"],out["reused_count"]),(126,90,36))
        self.assertEqual(list(map(len,out["remaining_lanes"])),m.REMAINING_LANE_COUNTS)
        reused=set(out["reused_keys"])
        self.assertEqual(out["remaining_entries"],[e for e in self.plan["entries"] if e["key"] not in reused])
        for original,new in zip(self.plan["lanes"],out["remaining_lanes"]):
            self.assertEqual(new,[e for e in original if e["key"] not in reused])
        self.assertEqual(self.plan,before);self.assertEqual(self.basepath.read_bytes(),raw)
        self.assertFalse(out["training_released"])
        self.assertNotIn("reused_protocol_sha256",out)
        self.assertEqual(out["frozen_records"][0]["source_experiment"],prev.EXPERIMENT)
        self.assertTrue(all(r["source_experiment"]==m.V2 for r in out["frozen_records"][1:]))
        path=self.root/"overlay.json"; digest=self.write(path,out)
        self.assertEqual(m.load_continuation(path,digest),out)

    def test_explicit_failed_required(self):
        for status in (None,"RUNNING","SUCCEEDED"):
            with self.assertRaises(ValueError):self.build(status=status)

    def test_base_old_audit_and_terminal_sha_tampering_fail(self):
        for path in (self.basepath,self.oldpath,self.reportpath,self.terminalpath):
            original=path.read_bytes();path.write_bytes(original+b" ")
            with self.assertRaises(ValueError):self.build()
            path.write_bytes(original)

    def test_missing_duplicate_foreign_report_keys_fail(self):
        baseline=copy.deepcopy(self.report)
        for change in ("missing","duplicate","foreign"):
            self.report=copy.deepcopy(baseline)
            if change=="missing":self.report["searches"].pop()
            elif change=="duplicate":self.report["searches"][-1]=copy.deepcopy(self.report["searches"][0])
            else:self.report["searches"][-1]["row"]["key"]="unrecorded/1"
            self.refresh()
            with self.subTest(change=change),self.assertRaises(ValueError):self.build()

    def test_orphan_interrupted_key_is_not_reused(self):
        interrupted=self.run/"lane0/collection/stt/interrupted/1"
        self.write(interrupted/"student/partial_replay.json",[])
        out=self.build()
        self.assertNotIn("interrupted/1",out["reused_keys"])
        self.assertFalse(any(str(interrupted) in p for p in out["source_files"]))

    def test_wrong_records_path_hash_row_and_protocol_fail(self):
        saved=copy.deepcopy(self.report)
        for change in ("path","sha","row","protocol"):
            self.report=copy.deepcopy(saved);item=self.report["searches"][1]
            if change=="path":item["records_path"]=str(self.root/"foreign.jsonl")
            elif change=="sha":item["records_sha256"]="f"*64
            elif change=="row":item["row"]["baseline_result_unchanged"]["collision"]=0.0
            else:item["audit"]["protocol_sha256"]=prev.PROTOCOL_SHA
            self.refresh()
            with self.subTest(change=change),self.assertRaises(ValueError):self.build()

    def test_full_independent_source_inventory_required(self):
        saved=copy.deepcopy(self.report)
        for change in ("missing","extra","wrongsha","empty"):
            self.report=copy.deepcopy(saved);files=self.report["searches"][1]["audit"]["source_files"]
            if change=="missing":files.pop(next(iter(files)))
            elif change=="extra":files[str(self.root/"foreign.json")]="0"*64
            elif change=="wrongsha":files[next(iter(files))]="0"*64
            else:files.clear()
            self.refresh()
            with self.subTest(change=change),self.assertRaises(ValueError):self.build()

    def test_precise_sidecars_are_pinned_but_not_semantically_audited(self):
        item=self.report["searches"][1];base=Path(item["row"]["student"]["artifact_root"]).parent
        path=base/"student_metrics/sub/log.json"
        digest=self.write(path,{"fixture":"not a trajectory label"})
        item["audit"]["source_files"][str(path)]=digest
        item["audit"]["sidecar_files"][str(path)]=digest
        # source_files in candidate is the core branch inventory captured before sidecars.
        item["audit"]["candidate"]["source_files"].pop(str(path),None)
        # Deepcopy prevents fixture's originally shared dict from erasing full pins.
        item["audit"]["source_files"]=dict(item["audit"]["source_files"],**{str(path):digest})
        (base/"lightnav_0003_client").mkdir()
        self.refresh()
        out=self.build()
        self.assertEqual(out["source_files"][str(path)],digest)
        path.write_text("tampered")
        with self.assertRaises(ValueError):self.build()

    def test_file_tampering_symlink_or_extra_branch_fails(self):
        item=self.report["searches"][1];base=Path(item["row"]["student"]["artifact_root"]).parent
        path=base/"student/result.json"; original=path.read_bytes()
        path.write_bytes(original+b" ")
        with self.assertRaises(ValueError):self.build()
        path.write_bytes(original)
        target=self.root/"elsewhere.json";target.write_bytes(original);path.unlink();path.symlink_to(target)
        with self.assertRaises(ValueError):self.build()
        path.unlink();path.write_bytes(original)
        self.write(base/"oracle_9999/branch.json",{})
        with self.assertRaises(ValueError):self.build()

    def test_wrong_winner_repeat_or_candidate_audit_fails(self):
        saved=copy.deepcopy(self.report)
        for change in ("winner","repeat","windows","release","outcome"):
            self.report=copy.deepcopy(saved);audit=self.report["searches"][1]["audit"]
            if change=="winner":audit["candidate"]["teacher"]="lightnav"
            elif change=="repeat":audit["candidate"]["branch"]=audit["candidate"]["repeat_branch"]
            elif change=="windows":audit["candidate"]["window_indices"]=[0]
            elif change=="release":audit["candidate"]["training_released"]=True
            else:audit["outcome"]="no_valid_teacher_recovery"
            self.refresh()
            with self.subTest(change=change),self.assertRaises(ValueError):self.build()

    def test_typed_error_path_union_has_no_fabricated_artifact_root(self):
        item=self.report["searches"][1]
        row=copy.deepcopy(item["row"]);base=Path(row["student"]["artifact_root"]).parent
        root=base/"lightnav_0003"
        invalid=dict(status="teacher_output_invalid",teacher="lightnav",complete=False,result=None,
             observed_partial=dict(evidence_files={
                 "partial_replay":dict(path=str(root/"partial_replay.json"),sha256="a"*64),
                 "partial_actions":dict(path=str(root/"partial_actions.json"),sha256="b"*64)}))
        row["attempts"][0]["branches"]["lightnav"]=invalid
        refs=m._branch_roots(row,base)
        self.assertEqual(len(refs),4)
        self.assertNotIn("artifact_root",invalid)
        for field in ("partial_replay","partial_actions"):
            changed=copy.deepcopy(row)
            changed["attempts"][0]["branches"]["lightnav"]["observed_partial"]["evidence_files"][field]["path"]="/foreign"
            with self.assertRaises(ValueError):m._branch_roots(changed,base)
        changed=copy.deepcopy(row);changed["attempts"][0]["branches"]["lightnav"]["status"]="unknown_error"
        with self.assertRaises(ValueError):m._branch_roots(changed,base)

    def test_missing_auditor_candidate_for_success_fails(self):
        self.report["searches"][1]["audit"]["candidate"]=None
        self.refresh()
        with self.assertRaises(ValueError):self.build()

    def test_terminal_process_and_complete_guards(self):
        path=self.run/"lane0/processes.jsonl";original=path.read_bytes()
        ev=[json.loads(x) for x in original.splitlines()];ev[-1]["pid"]=999
        self.lines(path,ev)
        with self.assertRaises(ValueError):self.build()
        path.write_bytes(original)
        self.write(self.run/"COMPLETE.json",{})
        with self.assertRaises(ValueError):self.build()

    def test_records_changed_after_terminal_are_rejected(self):
        p=Path(self.report["searches"][1]["records_path"])
        p.write_bytes(p.read_bytes()+b"{}\n")
        with self.assertRaises(ValueError):self.build()

    def test_load_rejects_same_numeric_values_with_changed_json_types(self):
        out=self.build()
        out["remaining_entries"][0]["baseline_result"]["success"]=0
        key=out["remaining_entries"][0]["key"]
        for lane in out["remaining_lanes"]:
            for entry in lane:
                if entry["key"]==key:entry["baseline_result"]["success"]=0
        path=self.root/"typedoverlay.json";digest=self.write(path,out)
        with self.assertRaisesRegex(ValueError,"exactly matches"):
            m.load_continuation(path,digest)

    def test_load_rejects_counts_lane_order_or_overlapping_reuse(self):
        original=self.build()
        for change in ("count","lane","reuse"):
            out=copy.deepcopy(original)
            if change=="count":out["new_count"]=89
            elif change=="lane":out["remaining_lanes"][0].reverse()
            else:out["reused_keys"][0]=out["remaining_entries"][0]["key"]
            path=self.root/f"{change}.json";digest=self.write(path,out)
            with self.subTest(change=change),self.assertRaises(ValueError):m.load_continuation(path,digest)

    def test_no_report_release_flag_or_unknown_fields(self):
        for key,value in (("training_released",True),("allow_release",True)):
            saved=copy.deepcopy(self.report);self.report[key]=value;self.refresh()
            with self.assertRaises(ValueError):self.build()
            self.report=saved;self.refresh()


if __name__=="__main__":
    unittest.main()
