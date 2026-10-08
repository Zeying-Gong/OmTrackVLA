"""Search-level mutation tests; raw-file validation is tested separately."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from wa.wm.failure_state_protocol import canonical_sha
from wa.wm import failure_state_protocol_v2 as v2
from wa.wm.failure_state_search_audit import audit_search, expected_candidates


class SearchAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)/"Scene"/"8"
        self.base.mkdir(parents=True)
        self.prefix = [dict(rgb_sha256=canonical_sha(["rgb", i]),
                            dynamic_state=dict(timestamp=i*.05), action=[.1,0.,0.])
                       for i in range(40)]
        self.start = {"rgb":self.prefix[0]["rgb_sha256"],"state":self.prefix[0]["dynamic_state"]}
        self.result = dict(success=0., collision=0., policy_init_valid=True, following_rate=.1)
        self.entry = dict(task="stt",key="Scene/8",baseline_result=copy.deepcopy(self.result))
        self.student = dict(result=copy.deepcopy(self.result),
            artifact_root=str(self.base/"student"), initial_pair_verified=True,
            first_start=dict(rgb=self.start["rgb"]))
        (self.base/"student").mkdir()
        (self.base/"student/replay.json").write_text(json.dumps(self.prefix))
        (self.base/"student/first_start.json").write_text(json.dumps(self.student["first_start"]))
        self.metadata = dict(experiment=v2.EXPERIMENT,partition="evaluation_adaptation",
            protocol_sha256=v2.PROTOCOL_SHA,plan_sha256="1"*64,checkpoint_sha256="2"*64,
            checkpoint_step=59716,source_dataset_sha256="3"*64,seed=7,
            base_plan_sha256="1"*64,base_protocol_sha256=v2.BASE_PROTOCOL_SHA,
            continuation_sha256="4"*64)
        self.audits = {}
        self.audit_add(self.base/"student", dict(result=self.student["result"]), [])

    def audit_add(self, root, branch, indices):
        root.mkdir(exist_ok=True)
        (root/"fixture.json").write_text(json.dumps(branch))
        self.audits[str(root)] = dict(kind="normal",branch=copy.deepcopy(branch),
            source_files={str(root/"fixture.json"):"a"*64},candidate_indices=indices,
            candidate_windows=len(indices),training_released=False,
            image_sha256=[canonical_sha(["PNG",i]) for i in range(40)])

    def normal(self, name, k, repeat, *, tr=.7, success=True, count=3):
        root=self.base/(f"{name}_{k:04d}"+("_repeat" if repeat else ""))
        b=dict(experiment=v2.EXPERIMENT, task="stt",key="Scene/8",takeover_step=k,
            seed=7,protocol_sha256=v2.PROTOCOL_SHA,
            initial_rgb_sha256=self.start["rgb"],
            takeover_state_sha256=canonical_sha(self.prefix[k]["dynamic_state"]),
            prefix_sha256=canonical_sha(self.prefix),teacher=name,complete=True,
            replay_verified=True,transport_fallback=False,verification_only=repeat,
            artifact_root=str(root),candidate_windows=0 if repeat or not success else count,
            result=dict(success=float(success),collision=0.,policy_init_valid=True,following_rate=tr))
        self.audit_add(root,b,list(range(k,k+b["candidate_windows"])))
        return b

    def make(self, *, rates=(.7,.7), success=True, student_success=False, first_zero=False):
        if student_success:
            self.student["result"]["success"]=1.
            self.audits[str(self.base/"student")]["branch"]["result"]=copy.deepcopy(self.student["result"])
        calls=[]
        def run(t,k,repeat):
            calls.append((t,k,repeat))
            return self.normal(t,k,repeat,tr=rates[0 if t=="lightnav" else 1],
                               success=success,count=0 if first_zero and k==35 else 3)
        row=v2.search_recovery(copy.deepcopy(self.student),self.prefix,run)
        row.update(task="stt",key="Scene/8",baseline_result_unchanged=self.entry["baseline_result"],
                   plan_sha256=self.metadata["plan_sha256"],
                   base_protocol_sha256=self.metadata["base_protocol_sha256"],
                   continuation_sha256=self.metadata["continuation_sha256"])
        return row,calls

    def audit(self,row):
        (self.base/"search.json").write_text(json.dumps(row))
        def raw_audit(root, **kw):
            expected = kw["expected"]
            self.assertEqual(expected["metadata"],self.metadata)
            if Path(root).name == "student":
                self.assertIsNone(expected["takeover_step"])
                self.assertEqual(expected["teacher"], "student")
                self.assertFalse(expected["verification_only"])
                self.assertIsNone(kw["student_prefix"])
            else:
                self.assertEqual(expected["prefix_image_sha256"],
                                 self.audits[str(self.base/"student")]["image_sha256"])
                branch = self.audits[str(root)]["branch"]
                self.assertEqual(expected["takeover_step"], branch["takeover_step"])
                self.assertEqual(expected["teacher"], branch["teacher"])
                self.assertEqual(expected["verification_only"], branch["verification_only"])
            return copy.deepcopy(self.audits[str(root)])
        return audit_search(row,self.entry,self.base,original_start=self.start,
            expected_metadata=self.metadata,audit_branch=raw_audit)

    def test_correct_tie_and_original_windows_not_repeat(self):
        row,calls=self.make()
        got=self.audit(row)
        self.assertEqual(got["candidate"]["teacher"],"lightnav")
        self.assertEqual(got["candidate"]["window_indices"],[35,36,37])
        self.assertNotIn("_repeat",got["candidate"]["branch"])
        self.assertEqual(calls,[("lightnav",35,False),("oracle",35,False),("lightnav",35,True)])
        self.assertFalse(got["training_released"])

    def test_higher_original_tr_oracle(self):
        row,_=self.make(rates=(.6,.8))
        self.assertEqual(self.audit(row)["candidate"]["teacher"],"oracle")

    def test_no_success_requires_every_fixed_candidate(self):
        row,_=self.make(success=False)
        got=self.audit(row)
        self.assertIsNone(got["candidate"])
        self.assertEqual([a["takeover_step"] for a in row["attempts"]],[35,25,10,0])
        row["attempts"].pop()
        with self.assertRaisesRegex(ValueError,"exhausting"):
            self.audit(row)

    def test_zero_windows_continues_earlier_not_runner_up(self):
        row,_=self.make(first_zero=True)
        got=self.audit(row)
        self.assertEqual(got["candidate"]["takeover_step"],25)
        self.assertEqual(len(row["attempts"]),2)

    def test_successful_student_no_labels(self):
        row,calls=self.make(student_success=True)
        self.assertFalse(calls)
        self.assertEqual(self.audit(row)["outcome"],"rerun_student_success_no_recovery_needed")

    def test_baseline_and_release_mutations(self):
        for change in ("baseline","release","backfill","protocol"):
            with self.subTest(change=change):
                row,_=self.make()
                if change=="baseline":row["baseline_result_unchanged"]={**self.entry["baseline_result"],"success":1.}
                if change=="release":row["training_released"]=True
                if change=="backfill":row["score_backfill_allowed"]=True
                if change=="protocol":row["protocol_sha256"]="b"*64
                with self.assertRaises(ValueError):self.audit(row)

    def test_missing_or_wrong_repeat_or_selected_teacher_rejected(self):
        for change in ("missing","wrongwinner","repeatflag","repeatcount"):
            with self.subTest(change=change):
                row,_=self.make()
                a=row["attempts"][0]
                if change=="missing":a.pop("repeat")
                if change=="wrongwinner":a["selection"]["selected_teacher"]="oracle"
                if change=="repeatflag":a["repeat_valid"]=False
                if change=="repeatcount":a["selected_candidate_windows"]=4
                with self.assertRaises(ValueError):self.audit(row)

    def test_source_branch_result_tamper(self):
        row,_=self.make()
        row["attempts"][0]["branches"]["oracle"]["result"]["success"]=0.
        with self.assertRaisesRegex(ValueError,"persisted branch"):
            self.audit(row)

    def test_outcome_and_pair_and_scope_tamper(self):
        for change in ("outcome","pair","scope","k"):
            with self.subTest(change=change):
                row,_=self.make()
                if change=="outcome":row["outcome"]="no_valid_teacher_recovery"
                if change=="pair":row["attempts"][0]["selection"]["pair"]["seed"]=8
                if change=="scope":row["attempts"][0]["selection"]["pair_comparability"]="partial"
                if change=="k":row["attempts"][0]["takeover_step"]=34
                with self.assertRaises(ValueError):self.audit(row)

    def test_orphan_branch_rejected(self):
        row,_=self.make()
        (self.base/"extra_teacher").mkdir()
        with self.assertRaisesRegex(ValueError,"orphan"):
            self.audit(row)

    def test_duplicate_window_and_foreign_hash_root_rejected(self):
        row,_=self.make()
        root=str(self.base/"lightnav_0035")
        self.audits[root]["candidate_indices"]=[35,35,37]
        with self.assertRaisesRegex(ValueError,"duplicate"):
            self.audit(row)
        self.audits[root]["candidate_indices"]=[35,36,37]
        self.audits[root]["source_files"]={"/other/project/file":"b"*64}
        with self.assertRaisesRegex(ValueError,"foreign branch evidence"):
            self.audit(row)

    def test_missing_or_changed_metadata_pins_rejected(self):
        row,_=self.make()
        for field in ("plan_sha256","checkpoint_sha256","source_dataset_sha256",
                      "checkpoint_step","continuation_sha256","seed"):
            saved=self.metadata.pop(field)
            with self.assertRaises(ValueError):self.audit(row)
            self.metadata[field]=saved
        self.metadata["continuation_sha256"]="5"*64
        with self.assertRaisesRegex(ValueError,"source continuation"):
            self.audit(row)

    def test_teacher_requires_audited_student_images(self):
        row,_=self.make()
        del self.audits[str(self.base/"student")]["image_sha256"]
        with self.assertRaisesRegex(ValueError,"student PNG"):
            self.audit(row)

    def test_exact_runtime_sidecars_are_pinned_not_extra_branches(self):
        row,_=self.make()
        for name in ("student_metrics","lightnav_0035_metrics","lightnav_0035_client",
                     "oracle_0035_metrics","lightnav_0035_repeat_metrics"):
            (self.base/name).mkdir()
        (self.base/"student.trace.jsonl").write_text('{"observed":true}\n')
        got=self.audit(row)
        self.assertIn(str(self.base/"student.trace.jsonl"),got["source_files"])
        self.assertFalse(got["sidecar_media_semantics_audited"])
        (self.base/"oracle_9999_metrics").mkdir()
        with self.assertRaisesRegex(ValueError,"orphan"):
            self.audit(row)

    def test_sidecar_symlink_rejected(self):
        row,_=self.make()
        (self.base/"student_metrics").mkdir()
        (self.base/"student_metrics/foreign").symlink_to(self.base/"search.json")
        with self.assertRaisesRegex(ValueError,"symlink"):
            self.audit(row)

    def test_fixed_candidates_boundary(self):
        self.assertEqual(expected_candidates(1),[0])
        self.assertEqual(expected_candidates(40),[35,25,10,0])
        for n in (0,-1,True,1.5):
            with self.assertRaises(ValueError):expected_candidates(n)


if __name__=="__main__":
    unittest.main()
