"""CPU orchestration tests; not a substitute for real Habitat nonzero replay."""
import copy
import tempfile
import unittest
from pathlib import Path

from wa.wm.failure_state_collect import (
    PAIR_FIELDS, DevelopmentLimitReached, LimitedRecorder, canonical_sha,
    repeat_valid, search_recovery, write_json,
)
from wa.wm.failure_state_protocol import EXPERIMENT, PROTOCOL_SHA, select_recovery_teacher


def branch(name, k, repeat=False, success=1, collision=0, rate=.8):
    return dict(experiment=EXPERIMENT, teacher=name, complete=True,
                replay_verified=True, transport_fallback=False,
                task="stt", key="VLzqgDo317F/238", takeover_step=k, seed=7,
                protocol_sha256=PROTOCOL_SHA, prefix_sha256="a"*64,
                initial_rgb_sha256="b"*64, takeover_state_sha256="c"*64,
                artifact_root=f"/fixture/{name}/{k}/"+("repeat" if repeat else "candidate"),
                verification_only=repeat, candidate_windows=0 if repeat else 4,
                result=dict(success=success, collision=collision, following_rate=rate,
                            policy_init_valid=True))


class FailureStateCollectTests(unittest.TestCase):
    def student(self, success=0):
        return dict(result=dict(success=success, collision=0, policy_init_valid=True),
                    artifact_root="/fixture/student")

    def test_first_same_k_success_uses_higher_rate_and_one_repeat(self):
        calls=[]
        def run(name,k,repeat):
            calls.append((name,k,repeat))
            return branch(name,k,repeat,rate=.9 if name=="oracle" else .8)
        r=search_recovery(self.student(),100,run)
        self.assertEqual(calls,[("lightnav",95,False),("oracle",95,False),("oracle",95,True)])
        self.assertEqual(r["accepted"]["teacher"],"oracle")
        self.assertFalse(r["training_released"])
        self.assertFalse(r["score_backfill_allowed"])

    def test_failed_repeat_moves_earlier_without_relabeling_repeat(self):
        calls=[]
        def run(name,k,repeat):
            calls.append((name,k,repeat))
            return branch(name,k,repeat,success=0 if repeat and k==95 else 1,rate=.9)
        r=search_recovery(self.student(),100,run)
        self.assertEqual(r["accepted"]["takeover_step"],85)
        self.assertFalse(r["attempts"][0]["repeat_valid"])
        self.assertEqual(calls,[(n,k,rep) for k in (95,85)
                               for n,rep in (("lightnav",False),("oracle",False),("lightnav",True))])

    def test_zero_window_winner_continues_earlier_without_switching_teacher(self):
        calls=[]
        def run(n,k,repeat):
            calls.append((n,k,repeat))
            value=branch(n,k,repeat,rate=.9)
            if k==95 and n=="lightnav":value["candidate_windows"]=0
            return value
        result=search_recovery(self.student(),100,run)
        self.assertEqual(result["accepted"]["takeover_step"],85)
        self.assertEqual(result["accepted"]["teacher"],"lightnav")
        self.assertEqual(result["attempts"][0]["selection"]["selected_teacher"],"lightnav")
        self.assertEqual(result["attempts"][0]["admission_reason"],
                         "selected_teacher_no_candidate_windows")
        self.assertEqual(calls[:3],[("lightnav",95,False),("oracle",95,False),("lightnav",95,True)])

    def test_missing_or_invalid_window_count_cannot_admit(self):
        for count in (None, True, -1, 1.0, "4"):
            def run(n,k,repeat):
                value=branch(n,k,repeat)
                value["candidate_windows"]=count
                return value
            with self.assertRaises(ValueError):
                search_recovery(self.student(),100,run)

    def test_failed_both_teachers_are_bounded_and_not_admitted(self):
        calls=[]
        def run(n,k,r):
            calls.append((n,k,r)); return branch(n,k,r,success=0)
        result=search_recovery(self.student(),100,run)
        self.assertEqual([a["takeover_step"] for a in result["attempts"]],[95,85,70,40,0])
        self.assertEqual(len(calls),10)
        self.assertIsNone(result["accepted"])

    def test_successful_student_no_demonstrations_or_score_backfill(self):
        r=search_recovery(self.student(1),100,lambda *args:self.fail("must not call teacher"))
        self.assertEqual(r["outcome"],"rerun_student_success_no_recovery_needed")
        self.assertEqual(r["attempts"],[])
        self.assertIsNone(r["accepted"])

    def test_invalid_student_preserved_without_teacher_labels(self):
        student=self.student();student["result"]["policy_init_valid"]=False
        r=search_recovery(student,10,lambda *args:self.fail("teacher forbidden"))
        self.assertEqual(r["outcome"],"student_invalid_initialization")

    def test_missing_student_init_is_not_defaulted_true(self):
        student=self.student();del student["result"]["policy_init_valid"]
        with self.assertRaises(ValueError):search_recovery(student,10,lambda *args:None)

    def test_transport_fallback_teacher_does_not_suppress_valid_oracle(self):
        def run(n,k,r):
            b=branch(n,k,r,rate=.99 if n=="lightnav" else .5)
            if n=="lightnav":b["transport_fallback"]=True
            return b
        r=search_recovery(self.student(),7,run)
        self.assertEqual(r["accepted"]["teacher"],"oracle")

    def test_repeat_must_have_independent_identity_and_valid_init(self):
        original=branch("oracle",4);repeat=branch("oracle",4,True)
        self.assertTrue(repeat_valid(original,repeat))
        repeat["result"]["policy_init_valid"]=False
        self.assertFalse(repeat_valid(original,repeat))
        for field in PAIR_FIELDS+("teacher",):
            changed=copy.deepcopy(repeat);changed[field]="DIFFERENT"
            with self.assertRaises(ValueError):repeat_valid(original,changed)
        duplicate=copy.deepcopy(original)
        with self.assertRaises(ValueError):repeat_valid(original,duplicate)
        repeat=branch("oracle",4,True);repeat["artifact_root"]=original["artifact_root"]
        with self.assertRaises(ValueError):repeat_valid(original,repeat)

    def test_nonfinite_rate_and_nonboolean_proof_rejected(self):
        original=branch("oracle",4)
        for field,value in (("following_rate",float("nan")),("following_rate",True)):
            repeated=branch("oracle",4,True);repeated["result"][field]=value
            with self.assertRaises(ValueError):repeat_valid(original,repeated)
        repeated=branch("oracle",4,True);repeated["replay_verified"]=1
        with self.assertRaises(ValueError):repeat_valid(original,repeated)

    def test_limited_recorder_does_not_fabricate_terminal_result(self):
        class Inner:
            def __init__(self):self.observed=[];self.actions=[]
            def observe(self,*args):self.observed.append(args[5])
            def record_action(self,*args):self.actions.append(args)
        inner=Inner();limited=LimitedRecorder(inner,2)
        for step in range(2):limited.observe({},None,None,None,None,step,10,.1*step)
        with self.assertRaises(DevelopmentLimitReached):
            limited.observe({},None,None,None,None,2,10,.2)
        limited.finish({"success":1})
        self.assertEqual(limited.early_result,{"success":1})
        self.assertEqual(inner.observed,[0,1])

    def test_exclusive_outputs_and_canonical_hash(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"result.json"
            write_json(path,{"v":1})
            with self.assertRaises(FileExistsError):write_json(path,{"v":2})
        self.assertEqual(canonical_sha({"a":1,"b":2}),canonical_sha({"b":2,"a":1}))
        with self.assertRaises(ValueError):canonical_sha({"v":float("nan")})


if __name__=="__main__":
    unittest.main()
