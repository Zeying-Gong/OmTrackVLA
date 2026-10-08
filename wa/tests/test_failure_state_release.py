"""CPU-only rejection checks for the full failure-state collection gate."""
import copy
import unittest
from pathlib import Path
from wa.wm.failure_state_release import exact_partition, numeric_entry, choose_original

class GateTests(unittest.TestCase):
    def setUp(self):
        self.entries=[dict(task="stt",key=f"scene/{i}") for i in range(126)]
        self.old=[e["key"] for e in self.entries[:36]]
        self.lanes=[self.entries[36+i::8] for i in range(8)]

    def test_exact_126(self):
        exact_partition(self.entries,self.old,self.lanes)

    def test_missing_new(self):
        self.lanes[0].pop()
        with self.assertRaises(ValueError): exact_partition(self.entries,self.old,self.lanes)

    def test_overlap(self):
        self.lanes[0][0]=self.entries[0]
        with self.assertRaises(ValueError): exact_partition(self.entries,self.old,self.lanes)

    def test_foreign_key(self):
        self.lanes[0][0]=dict(task="stt",key="foreign/1")
        with self.assertRaises(ValueError): exact_partition(self.entries,self.old,self.lanes)

    def test_duplicate_plan(self):
        self.entries[-1]=self.entries[0]
        with self.assertRaises(ValueError): exact_partition(self.entries,self.old,self.lanes)

    def test_wrong_task(self):
        self.entries[3]["task"]="at"
        with self.assertRaises(ValueError): exact_partition(self.entries,self.old,self.lanes)

    @staticmethod
    def numbers():
        c=dict(task="stt",key="s/1",teacher="oracle",branch="/data/nas_ray/original",
               takeover_step=5,candidate_windows=8)
        n=dict(c,valid_windows=6,excluded_windows=2,template_index=0,
               causal_history_verified=True,future_teacher_ownership_verified=True,
               jepa_command_available_for_valid=True,previous_action_available_for_valid=True)
        return c,n

    def test_numeric_identity(self):
        c,n=self.numbers()
        self.assertEqual(numeric_entry(c,n),6)
        n["branch"]+="_repeat"
        with self.assertRaises(ValueError): numeric_entry(c,n)

    def test_numeric_no_valid_not_hidden(self):
        c,n=self.numbers();n.update(valid_windows=0,excluded_windows=8)
        self.assertEqual(numeric_entry(c,n),0)

    def test_numeric_counts(self):
        for changes in (dict(valid_windows=True),dict(excluded_windows=1),dict(valid_windows=-1)):
            c,n=self.numbers();n.update(changes)
            with self.assertRaises(ValueError):numeric_entry(c,n)

    def test_numeric_missing_causal_guarantee(self):
        for field in ("template_index","causal_history_verified","future_teacher_ownership_verified",
                      "jepa_command_available_for_valid","previous_action_available_for_valid"):
            c,n=self.numbers();n[field]=1 if field=="template_index" else False
            with self.assertRaises(ValueError):numeric_entry(c,n)

    @staticmethod
    def branch(rate,success=True,repeat=False):
        return dict(result=dict(success=success,collision=False,policy_init_valid=True,following_rate=rate),
                    complete=True,replay_verified=True,transport_fallback=False,verification_only=repeat)

    def selection(self,lnrate=.9,orate=.95,lnsuccess=True,osuccess=True,repeatok=True):
        base=Path("/data/nas_ray/s/1")
        refs=dict(lightnav=self.branch(lnrate,lnsuccess),oracle=self.branch(orate,osuccess))
        flags={t:refs[t]["result"]["success"] for t in refs}
        selected=("oracle" if osuccess and (not lnsuccess or orate>lnrate) else "lightnav")
        repeat=self.branch(refs[selected]["result"]["following_rate"],repeatok,True)
        attempt=dict(takeover_step=35,branches=refs,repeat=repeat,repeat_valid=repeatok,
                     selected_candidate_windows=1,selection=dict(selected_teacher=selected,eligible=flags))
        row=dict(student=dict(result=dict(policy_init_valid=True,success=False)),attempts=[attempt],
                 accepted=dict(teacher=selected))
        docs={str(base/"student/replay.json"):[{}]*40,
              str(base/f"{selected}_0035/branch.json"):refs[selected],
              str(base/f"{selected}_0035_repeat/branch.json"):repeat,
              str(base/f"{selected}_0035/windows.json"):[dict(current_index=35)]}
        class FakePins:
            def doc(self,path): return docs[str(path)]
        return row,FakePins(),base

    def test_success_then_tr(self):
        row,pins,base=self.selection()
        self.assertEqual(choose_original(row,pins,base)["teacher"],"oracle")
        row,pins,base=self.selection(lnrate=1,lnsuccess=False)
        self.assertEqual(choose_original(row,pins,base)["teacher"],"oracle")

    def test_tie_lightnav(self):
        row,pins,base=self.selection(lnrate=.95,orate=.95)
        got=choose_original(row,pins,base)
        self.assertEqual(got["teacher"],"lightnav")
        self.assertNotIn("_repeat",got["branch"])

    def test_wrong_winner(self):
        row,pins,base=self.selection()
        row["attempts"][0]["selection"]["selected_teacher"]="lightnav"
        with self.assertRaises(ValueError):choose_original(row,pins,base)

    def test_wrong_takeover_order(self):
        row,pins,base=self.selection();row["attempts"][0]["takeover_step"]=0
        with self.assertRaises(ValueError):choose_original(row,pins,base)

    def test_repeat_failure_does_not_release(self):
        row,pins,base=self.selection(repeatok=False)
        with self.assertRaises(ValueError):choose_original(row,pins,base)  # incomplete fixed search

    def test_no_extra_attempt_after_winner(self):
        row,pins,base=self.selection()
        row["attempts"].append(copy.deepcopy(row["attempts"][0]))
        with self.assertRaises(ValueError):choose_original(row,pins,base)

    def test_repeat_cannot_be_original(self):
        row,pins,base=self.selection()
        row["attempts"][0]["branches"]["oracle"]["verification_only"]=True
        with self.assertRaises(ValueError):choose_original(row,pins,base)


class ProvenanceAndPinsTests(unittest.TestCase):
    @staticmethod
    def evidence(size=90):
        import wa.wm.failure_state_release as g
        path=Path("/data/nas_ray/test/raw.json");digest="a"*64
        report=dict(schema=f"failure_state_completed_search_audit_{61836 if size==36 else 61844}_v1")
        prov=dict(schema="failure_state_completed_search_audit_provenance_v1",
                  report_path=str(path),report_sha256=digest,status="PASS_NONRELEASE",
                  source_unchanged=True,training_released=False,score_backfill_allowed=False,
                  source_sha256={"/source.py":"b"*64})
        if size==36:
            prov["failures"]=[]
        else:
            branch=dict(status="teacher_output_invalid",teacher="lightnav",result=None,
                        complete=False,replay_verified=False,fallback_executed=False,
                        training_eligible=False,training_released=False,
                        raw_error=dict(source="server_response",rc=500,seq=66,
                                       exception_type=None,
                                       message="got 2 act levels, expected 3 from '<act_l0_1><act_l1_2>'"))
            prov.update(errors=[dict(root=str(g.ROOT/"lane4/collection/stt/s/1/lightnav_0062"),
                                     category="missing_final_l2",branch=branch)],
                        totals=dict(teacher_errors=1),
                        source_input_sha256={"/input.bin":"c"*64})
        class Seen:
            def __init__(self):self.inventories=[]
            def verify_all(self,inventory):self.inventories.append(inventory)
        return report,prov,path,digest,Seen()

    def check_provenance(self,size=90,change=None):
        from wa.wm.failure_state_release import validate_provenance
        report,prov,path,digest,pins=self.evidence(size)
        if change:change(report,prov)
        validate_provenance(report,prov,size=size,path=path,digest=digest,pins=pins)
        return pins

    def test_actual_old_provenance_schema(self):
        self.assertEqual(len(self.check_provenance(36).inventories),1)

    def test_new_typed_errors_without_failures_field_are_allowed(self):
        pins=self.check_provenance()
        self.assertEqual(pins.inventories,[{"/input.bin":"c"*64},{"/source.py":"b"*64}])

    def test_legacy_missing_rvq_category_allowed(self):
        def change(r,p):
            p["errors"][0]["category"]="missing_rvq"
            p["errors"][0]["branch"]["raw_error"]["message"]="Missing rvq act levels [1] in '<act_l0_1>'"
        self.check_provenance(change=change)

    def test_fatal_auditor_failures_rejected_both_schemas(self):
        for size in (36,90):
            with self.subTest(size=size),self.assertRaises(ValueError):
                self.check_provenance(size,lambda r,p:p.update(failures=[{"error":"fatal"}]))

    def test_unknown_decoder_and_terminal_typed_error_rejected(self):
        mutations=(
            lambda e:e["branch"]["raw_error"].update(message="arbitrary RuntimeError"),
            lambda e:e["branch"].update(complete=True),
            lambda e:e["branch"].update(result={"success":True}),
            lambda e:e["branch"].update(fallback_executed=True),
            lambda e:e["branch"].update(teacher="oracle"),
            lambda e:e["branch"]["raw_error"].update(rc=True),
            lambda e:e["branch"]["raw_error"].update(seq=True),
            lambda e:e.update(category="missing_rvq"))
        for change in mutations:
            with self.subTest(change=change),self.assertRaises(ValueError):
                self.check_provenance(change=lambda r,p:change(p["errors"][0]))

    def test_unknown_schema_bad_path_or_error_count_rejected(self):
        mutations=(lambda r,p:r.update(schema="other"),
                   lambda r,p:p.update(schema="other"),
                   lambda r,p:p.update(report_path="/wrong.json"),
                   lambda r,p:p.update(report_sha256="d"*64),
                   lambda r,p:p["totals"].update(teacher_errors=0),
                   lambda r,p:p["errors"].append(copy.deepcopy(p["errors"][0])))
        for change in mutations:
            with self.subTest(change=change),self.assertRaises(ValueError):
                self.check_provenance(change=change)

    def test_stream_hash_never_uses_read_bytes(self):
        import hashlib,tempfile
        from unittest.mock import patch
        from wa.wm.failure_state_release import Pins
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp).resolve()/"large.bin";payload=b"abc"*700000
            path.write_bytes(payload);pins=Pins()
            with patch.object(Path,"read_bytes",side_effect=AssertionError("not streaming")):
                pins.verify_all({str(path):hashlib.sha256(payload).hexdigest()})
            self.assertEqual(pins.files[str(path)],hashlib.sha256(payload).hexdigest())
            pins.finish()
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):pins.finish()

    def test_json_hashed_once_and_duplicate_nonfinite_rejected(self):
        import tempfile
        from wa.wm.failure_state_release import Pins
        for text in ('{"x":1,"x":2}','{"x":NaN}'):
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp).resolve()/"data.json";path.write_text(text)
                with self.assertRaises(ValueError):Pins().doc(path)

    def test_symlink_evidence_rejected(self):
        import tempfile
        from wa.wm.failure_state_release import Pins
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();(root/"data").write_bytes(b"x")
            (root/"link").symlink_to(root/"data")
            with self.assertRaises(ValueError):Pins().blob(root/"link")

    def test_source_protocol_and_explicit_boundary(self):
        from wa.wm.failure_state_release import validate_protocol,v1,v2,CONTINUATION_POLICY
        for module in (v1,v2):
            row=dict(task="stt",experiment=module.EXPERIMENT,protocol_sha256=module.PROTOCOL_SHA)
            validate_protocol(row)
        row=dict(task="stt",experiment=v2.EXPERIMENT,protocol_sha256=v2.PROTOCOL_SHA,
                 teacher_boundary_policy=CONTINUATION_POLICY)
        validate_protocol(row,new=True)
        for changes in (dict(experiment=v1.EXPERIMENT),dict(protocol_sha256="a"*64),
                        dict(task="at"),dict(teacher_boundary_policy=None)):
            with self.assertRaises(ValueError):validate_protocol(dict(row,**changes),new=True)
        with self.assertRaises(ValueError):validate_protocol(row,new=False)


class ExactNumericTests(unittest.TestCase):
    @staticmethod
    def fixture(new=True):
        import numpy as np
        from wa.wm.failure_state_labels import derive_labels
        from wa.wm.failure_state_numeric import audit_numeric_windows
        from wa.wm.failure_state_release import v2,CONTINUATION_POLICY
        k=0;n=30;indices=list(range(12))
        obs=[dict(sim_step=i,timestamp_s=i*.05,frame=f"rgb_{i:04d}.png",
                  robot_position_world=[i*.01,0.,0.],
                  robot_rotation_world_from_body=np.eye(3).tolist()) for i in range(n)]
        acts=[dict(sim_step=i,normalized_action=[.1,0.,0.],owner="teacher",teacher="oracle") for i in range(n)]
        d=derive_labels(obs,acts,indices,k)
        wins=[dict(current_index=i,trajectory_xy_m=d["trajectory_xy_m"][j].tolist(),
                   future_bracket_indices=d["future_bracket_indices"][j].tolist(),
                   future_times_s=(np.arange(1,8)/10.).tolist(),
                   teacher_owned_action_indices=[i,int(d["end_indices"][j])-1],
                   label_endpoint_observation_index=int(d["end_indices"][j]),training_eligible=False)
              for j,i in enumerate(indices)]
        derived=audit_numeric_windows(obs,acts,wins,k,per_window=False)
        root=Path("/data/nas_ray/selected/oracle_0000")
        candidate=dict(task="stt",key="s/1",teacher="oracle",takeover_step=k,branch=str(root),
                       repeat_branch=str(root)+"_repeat",candidate_windows=len(indices),window_indices=indices)
        row=dict(experiment=v2.EXPERIMENT,protocol_sha256=v2.PROTOCOL_SHA)
        episode={**candidate,**derived["summary"],"template_index":0,
                 "causal_history_verified":True,"future_teacher_ownership_verified":True,
                 "jepa_command_available_for_valid":True,"previous_action_available_for_valid":True,
                 "source_experiment":row["experiment"],"source_protocol_sha256":row["protocol_sha256"]}
        if new:
            episode.update(valid_window_indices=derived["valid_window_indices"],excluded=derived["excluded"],
                           source_boundary_policy=CONTINUATION_POLICY)
        docs={"observations.json":obs,"actions.json":acts,"windows.json":wins}
        class FakePins:
            def doc(self,path):return docs[Path(path).name]
        return candidate,episode,row,FakePins(),docs

    def test_old_count_only_report_gets_recomputed_sets(self):
        from wa.wm.failure_state_release import exact_numeric
        c,e,row,pins,docs=self.fixture(False)
        self.assertNotIn("valid_window_indices",e)
        got=exact_numeric(c,e,row,pins,new=False)
        self.assertEqual(got["valid_window_indices"],list(range(4,12)))
        self.assertEqual([x["current_index"] for x in got["numeric_exclusions"]],list(range(4)))

    def test_new_exact_sets_and_counts(self):
        from wa.wm.failure_state_release import exact_numeric
        c,e,row,pins,docs=self.fixture()
        got=exact_numeric(c,e,row,pins,new=True)
        self.assertEqual(got["expected_valid_count"],8)
        self.assertEqual(got["expected_excluded_count"],4)

    def test_same_count_wrong_valid_or_excluded_set_rejected(self):
        from wa.wm.failure_state_release import exact_numeric
        for change in (lambda e:e["valid_window_indices"].__setitem__(0,3),
                       lambda e:e["excluded"][0].update(current_index=12),
                       lambda e:e["excluded"][0].update(reasons=["OTHER"]),
                       lambda e:e.update(valid_window_indices=[True]*8)):
            c,e,row,pins,docs=self.fixture();change(e)
            with self.assertRaises(ValueError):exact_numeric(c,e,row,pins,new=True)

    def test_numeric_source_and_ownership_rejected(self):
        from wa.wm.failure_state_release import exact_numeric
        c,e,row,pins,docs=self.fixture();e["source_protocol_sha256"]="a"*64
        with self.assertRaises(ValueError):exact_numeric(c,e,row,pins,new=True)
        c,e,row,pins,docs=self.fixture();docs["actions.json"][5]["owner"]="student"
        with self.assertRaises(ValueError):exact_numeric(c,e,row,pins,new=True)

    def test_old_numeric_count_cannot_override_recomputed_mask(self):
        from wa.wm.failure_state_release import exact_numeric
        c,e,row,pins,docs=self.fixture(False);e.update(valid_windows=9,excluded_windows=3)
        with self.assertRaises(ValueError):exact_numeric(c,e,row,pins,new=False)

    def test_bool_template_index_rejected(self):
        from wa.wm.failure_state_release import numeric_entry
        c,e,_,_,_=self.fixture();e["template_index"]=False
        with self.assertRaises(ValueError):numeric_entry(c,e)


class NumericReportTests(unittest.TestCase):
    @staticmethod
    def report(size):
        from wa.wm.failure_state_release import (
            ROOT,OLD_NUMERIC,OLD_NUMERIC_SHA,OLD_AUDIT,OLD_AUDIT_SHA,
            COLLECTION_SOURCE,COLLECTION_COMMIT,v2,CONTINUATION_POLICY)
        if size==36:
            eps=[dict(key="s/0",candidate_windows=2016,valid_windows=1871,excluded_windows=145,
                      rejection_reason_counts={"FILTER":145})]
            eps += [dict(key=f"s/{i}",candidate_windows=1,valid_windows=int(i<=20),
                         excluded_windows=int(i>20),rejection_reason_counts={"FILTER":int(i>20)})
                    for i in range(1,24)]
        else:
            eps=[dict(key="s/0",candidate_windows=8,valid_windows=6,excluded_windows=2,
                      rejection_reason_counts={"FILTER":2})]
        summary={k:sum(e[k] for e in eps) for k in ("candidate_windows","valid_windows","excluded_windows")}
        summary.update(accepted_original_episodes=len(eps),episodes_with_valid_windows=sum(e["valid_windows"]>0 for e in eps),
                       rejection_reason_counts={"FILTER":summary["excluded_windows"]},
                       repeat_windows_counted=0,student_prefix_actions_used_as_future_labels=0,
                       all_candidate_future_transitions_teacher_owned=True)
        result=dict(schema=f"failure_state_candidate_numeric_audit_{61836 if size==36 else 61844}_v1",
                    training_eligible=False,training_released=False,cache_generated=False,
                    no_success_rate=True,episodes=eps,summary=summary,
                    code_sha256_before={"/numeric.py":"a"*64},code_sha256_after={"/numeric.py":"a"*64})
        if size==36:
            result["input_report"]=dict(path=str(OLD_AUDIT),sha256=OLD_AUDIT_SHA)
        if size==90:
            summary.update(completed_searches=90,no_candidate_searches=89)
            result.update(job_id=61844,task_id=73066,root=str(ROOT),full_search_media_audit=False,
                          no_candidate=[dict(key=f"empty/{i}") for i in range(89)],
                          old36_reference=dict(path=str(OLD_NUMERIC),sha256=OLD_NUMERIC_SHA),
                          source_collection=dict(experiment=v2.EXPERIMENT,protocol_sha256=v2.PROTOCOL_SHA,
                              boundary_policy=CONTINUATION_POLICY,commit=COLLECTION_COMMIT,source=str(COLLECTION_SOURCE)))
        return result

    def test_both_actual_numeric_schemas_supported(self):
        from wa.wm.failure_state_release import validate_numeric_report
        for size in (36,90):validate_numeric_report(self.report(size),size)

    def test_declarations_summary_and_schema_rejected(self):
        from wa.wm.failure_state_release import validate_numeric_report
        for size in (36,90):
            for change in (lambda d:d.update(schema="wrong"),lambda d:d.update(training_eligible=True),
                           lambda d:d.update(cache_generated=True),lambda d:d.update(no_success_rate=False),
                           lambda d:d.update(code_sha256_after={"/numeric.py":"b"*64}),
                           lambda d:d["summary"].update(valid_windows=999),
                           lambda d:d["summary"].update(repeat_windows_counted=False),
                           lambda d:d["episodes"][0].update(valid_windows=True),
                           lambda d:d["summary"]["rejection_reason_counts"].update(FILTER=999)):
                d=self.report(size);change(d)
                with self.subTest(size=size,change=change),self.assertRaises(ValueError):
                    validate_numeric_report(d,size)

    def test_new_scope_reference_and_missing_count_rejected(self):
        from wa.wm.failure_state_release import validate_numeric_report
        for change in (lambda d:d.update(job_id=61836),lambda d:d.update(root="/wrong"),
                       lambda d:d["summary"].update(completed_searches=89),
                       lambda d:d["no_candidate"].pop(),
                       lambda d:d["no_candidate"][0].update(key="s/0"),
                       lambda d:d["source_collection"].update(boundary_policy="other"),
                       lambda d:d["old36_reference"].update(sha256="a"*64)):
            d=self.report(90);change(d)
            with self.assertRaises(ValueError):validate_numeric_report(d,90)

class MarkerTests(unittest.TestCase):
    @staticmethod
    def marker(lane=None):
        import wa.wm.failure_state_release as g
        keys=["s/1","s/2"]
        value=dict(experiment=g.v2.EXPERIMENT,protocol_sha256=g.v2.PROTOCOL_SHA,
                   plan_sha256=g.PLAN_SHA,continuation_sha256=g.OVERLAY_SHA,
                   teacher_boundary_policy=g.CONTINUATION_POLICY,training_released=False,
                   no_success_rate=True,keys=keys)
        if lane is None:
            value.update(status="COLLECTION_PROCESSES_COMPLETE_NOT_TRAINING_RELEASE",
                         entries=90,lanes=8,formal=True)
        else:
            value.update(base_protocol_sha256=g.v1.PROTOCOL_SHA,entries=2,expected=2,
                         shard=lane,development=False)
        return value,keys

    def test_root_and_lane_exact_contract(self):
        from wa.wm.failure_state_release import validate_marker
        for lane in (None,0,7):
            marker,keys=self.marker(lane)
            validate_marker(marker,keys,lane=lane)

    def test_wrong_lane_boundary_plan_or_keys_rejected(self):
        from wa.wm.failure_state_release import validate_marker
        for field,value in (("shard",True),("shard",3),("teacher_boundary_policy","other"),
                            ("plan_sha256","a"*64),("base_protocol_sha256","a"*64),
                            ("keys",["s/2","s/1"]),("development",True),("entries",True)):
            marker,keys=self.marker(1);marker[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_marker(marker,keys,lane=1)

    def test_root_must_be_terminal_full_scope(self):
        from wa.wm.failure_state_release import validate_marker
        for field,value in (("entries",89),("lanes",7),("formal",False),
                            ("status","RUNNING"),("training_released",True)):
            marker,keys=self.marker();marker[field]=value
            with self.assertRaises(ValueError):validate_marker(marker,keys)

class ParallelPinsTests(unittest.TestCase):
    @staticmethod
    def files(root, count=12):
        import hashlib
        inventory = {}
        for i in range(count):
            path = root/f"sample_{i:04d}.bin"
            payload = (str(i) + "-payload").encode()
            path.write_bytes(payload)
            inventory[str(path)] = hashlib.sha256(payload).hexdigest()
        return inventory

    def test_only_one_or_four_workers(self):
        from wa.wm.failure_state_release import Pins
        self.assertEqual(Pins().hash_workers, 1)
        self.assertEqual(Pins(4).hash_workers, 4)
        for value in (0, 2, 3, 5, -1, True, False, 1.0, "4", None):
            with self.subTest(value=value), self.assertRaises(ValueError): Pins(value)

    def test_serial_parallel_full_inventory_and_order_identical(self):
        import tempfile
        from wa.wm.failure_state_release import Pins
        with tempfile.TemporaryDirectory() as tmp:
            inventory = self.files(Path(tmp).resolve(), 263)  # Cross queue boundary.
            serial, parallel = Pins(), Pins(4)
            for pins in (serial, parallel):
                first = next(iter(inventory))
                pins.blob(first, inventory[first])
                pins.verify_all(inventory)
                pins.verify_all(inventory)  # Existing pins remain exact.
                pins.finish()
            self.assertEqual(serial.files, parallel.files)
            self.assertEqual(serial.stats, parallel.stats)
            self.assertEqual(list(serial.files), list(parallel.files))
            self.assertEqual(list(serial.stats), list(parallel.stats))
            self.assertEqual(list(serial.files), list(inventory))

    def test_conflict_validated_before_worker_reads(self):
        import tempfile
        from unittest.mock import patch
        from wa.wm.failure_state_release import Pins
        with tempfile.TemporaryDirectory() as tmp:
            inventory = self.files(Path(tmp).resolve(), 2)
            path = next(iter(inventory))
            for workers in (1, 4):
                pins = Pins(workers); pins.blob(path)
                conflict = dict(inventory); conflict[path] = "f"*64
                with patch.object(Pins, "_read", side_effect=AssertionError("worker started")):
                    with self.assertRaisesRegex(ValueError, "conflicting source pin"):
                        pins.verify_all(conflict)

    def test_invalid_inventory_hash_does_not_start_workers(self):
        from unittest.mock import patch
        from wa.wm.failure_state_release import Pins
        for workers in (1, 4):
            with patch.object(Pins, "_read", side_effect=AssertionError("worker started")):
                with self.assertRaisesRegex(ValueError, "invalid source SHA"):
                    Pins(workers).verify_all({"/unread": "a"*64, "/bad": "invalid"})

    def test_changed_file_rejected_by_parallel_finish(self):
        import tempfile
        from wa.wm.failure_state_release import Pins
        for workers in (1, 4):
            with tempfile.TemporaryDirectory() as tmp:
                inventory = self.files(Path(tmp).resolve())
                pins = Pins(workers); pins.verify_all(inventory)
                Path(next(iter(inventory))).write_bytes(b"modified")
                with self.assertRaisesRegex(ValueError, "source changed"):
                    pins.finish()

    def test_replacement_symlink_rejected_by_parallel_finish(self):
        import tempfile
        from wa.wm.failure_state_release import Pins
        for workers in (1, 4):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve(); inventory = self.files(root, 2)
                pins = Pins(workers); pins.verify_all(inventory)
                source, target = map(Path, inventory)
                source.unlink(); source.symlink_to(target)
                with self.assertRaisesRegex(ValueError, "source changed"):
                    pins.finish()

    def test_hash_mismatch_and_worker_exception_propagate(self):
        import tempfile
        from unittest.mock import patch
        from wa.wm.failure_state_release import Pins
        for workers in (1, 4):
            with tempfile.TemporaryDirectory() as tmp:
                inventory = self.files(Path(tmp).resolve())
                bad = dict(inventory); bad[next(iter(bad))] = "f"*64
                with self.assertRaisesRegex(ValueError, "evidence SHA mismatch"):
                    Pins(workers).verify_all(bad)
                original = Pins._read
                failed = list(inventory)[3]
                def read(worker, path, *args, **kwargs):
                    if str(path) == failed: raise OSError("test read error")
                    return original(worker, path, *args, **kwargs)
                with patch.object(Pins, "_read", read):
                    with self.assertRaisesRegex(OSError, "test read error"):
                        Pins(workers).verify_all(inventory)

    def test_finish_worker_exception_propagates(self):
        import tempfile
        from unittest.mock import patch
        from wa.wm.failure_state_release import Pins
        with tempfile.TemporaryDirectory() as tmp:
            inventory = self.files(Path(tmp).resolve())
            pins = Pins(4); pins.verify_all(inventory)
            with patch.object(Pins, "_check_state", side_effect=OSError("test stat error")):
                with self.assertRaisesRegex(OSError, "test stat error"): pins.finish()

    def test_parallel_reads_are_independent_and_capped(self):
        import tempfile, threading, time
        from unittest.mock import patch
        from wa.wm.failure_state_release import Pins
        original = Pins._read
        with tempfile.TemporaryDirectory() as tmp:
            inventory = self.files(Path(tmp).resolve(), 17)
            for workers in (1, 4):
                state = dict(active=0, peak=0); lock = threading.Lock(); owner = Pins(workers)
                def read(worker, path, *args, **kwargs):
                    self.assertIsNot(worker, owner)
                    self.assertEqual(worker.files, {})
                    with lock:
                        state["active"] += 1
                        state["peak"] = max(state["peak"], state["active"])
                    try:
                        time.sleep(.01)
                        return original(worker, path, *args, **kwargs)
                    finally:
                        with lock: state["active"] -= 1
                with patch.object(Pins, "_read", read): owner.verify_all(inventory)
                self.assertEqual(state["active"], 0)
                self.assertLessEqual(state["peak"], workers)
                self.assertEqual(state["peak"], 1) if workers == 1 else self.assertGreater(state["peak"], 1)
                self.assertEqual(list(owner.files), list(inventory))

    def test_read_time_change_is_rejected_in_both_modes(self):
        import os, tempfile
        from unittest.mock import patch
        from wa.wm.failure_state_release import Pins
        original_open = Path.open
        for workers in (1, 4):
            with tempfile.TemporaryDirectory() as tmp:
                inventory = self.files(Path(tmp).resolve(), 1)
                target = Path(next(iter(inventory)))
                class ChangingStream:
                    def __init__(self, stream): self.stream = stream; self.changed = False
                    def __enter__(self): return self
                    def __exit__(self, *args): self.stream.close()
                    def read(self, size):
                        data = self.stream.read(size)
                        if not self.changed:
                            st = target.stat()
                            os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
                            self.changed = True
                        return data
                def opening(path, *args, **kwargs):
                    stream = original_open(path, *args, **kwargs)
                    return ChangingStream(stream) if path == target else stream
                with patch.object(Path, "open", opening):
                    with self.assertRaisesRegex(ValueError, "changed while reading"):
                        Pins(workers).verify_all(inventory)

    def test_build_release_default_and_explicit_constructor_are_compatible(self):
        from unittest.mock import patch
        from wa.wm.failure_state_release import build_release
        class StopBeforeIO(Exception): pass
        for options, expected in (({}, 1), ({"hash_workers": 4}, 4)):
            with patch("wa.wm.failure_state_release.Pins", side_effect=StopBeforeIO) as ctor:
                with self.assertRaises(StopBeforeIO): build_release("a", "b", "c", "d", **options)
                ctor.assert_called_once_with(hash_workers=expected)

    def test_cli_worker_default_and_explicit_choice(self):
        import contextlib, io, tempfile
        from unittest.mock import patch
        from wa.wm import failure_state_release as gate
        for choices, expected in (([], 1), (["--hash-workers", "4"], 4)):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve(); (root/"artifacts").mkdir()
                output = root/"artifacts/report.json"
                args = ["release", "--new-audit", "a", "--new-audit-sha", "b",
                        "--new-numeric", "c", "--new-numeric-sha", "d",
                        "--output", str(output)] + choices
                with patch.object(gate, "R", root), patch("sys.argv", args), \
                        patch.object(gate, "build_release", return_value={"summary": {}}) as build, \
                        contextlib.redirect_stdout(io.StringIO()):
                    gate.main()
                build.assert_called_once_with("a", "b", "c", "d", hash_workers=expected)
                self.assertTrue(output.is_file())
        with patch("sys.argv", args + ["--hash-workers", "2"]), \
                patch.object(gate, "build_release") as build, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): gate.main()
            build.assert_not_called()

    def test_progress_throttles_bulk_at_thousand_boundaries(self):
        import contextlib, io, json, tempfile
        from wa.wm.failure_state_release import Pins
        with tempfile.TemporaryDirectory() as tmp:
            inventory = self.files(Path(tmp).resolve(), 1001)
            output = io.StringIO()
            with contextlib.redirect_stderr(output): Pins(4).verify_all(inventory)
            rows = [json.loads(s) for s in output.getvalue().splitlines()]
            self.assertEqual([r["completed"] for r in rows], [0, 1000, 1001])
            self.assertTrue(all(r["total"] == 1001 for r in rows))

    def test_progress_is_count_only_stderr(self):
        import contextlib, io, json, tempfile
        from wa.wm.failure_state_release import Pins
        with tempfile.TemporaryDirectory() as tmp:
            inventory = self.files(Path(tmp).resolve(), 2)
            output = io.StringIO()
            with contextlib.redirect_stderr(output):
                pins = Pins(4); pins.verify_all(inventory); pins.finish()
            rows = [json.loads(s) for s in output.getvalue().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(set(r) == {"phase", "completed", "total"} for r in rows))
            self.assertEqual([r["completed"] for r in rows], [0, 2, 0, 2])

if __name__=="__main__":unittest.main()
