"""Focused CPU contract tests; real raw evidence is a separate audit gate."""
import copy
import unittest
import threading
import time
from wa.tools.audit_failure_state_completed_61844 import (
    validate_rows, validate_processes, validate_output_error, strict_loads,
    POLICY, EXPERIMENT, ordered_parallel_map,
)

class Audit61844Tests(unittest.TestCase):
    def test_parallel_matches_sequential(self):
        def f(x):
            time.sleep((9-x)*.001)
            return {"index":x,"squared":x*x}
        self.assertEqual(ordered_parallel_map(f,range(10),1),ordered_parallel_map(f,range(10),4))
    def test_parallel_preserves_order(self):
        def f(x):
            time.sleep(.02 if x==0 else 0)
            return x
        self.assertEqual(ordered_parallel_map(f,range(8)),list(range(8)))
    def test_parallel_exception_propagates(self):
        def f(x):
            if x==2:raise RuntimeError("unknown exact error")
            return x
        with self.assertRaisesRegex(RuntimeError,"unknown exact error"):
            ordered_parallel_map(f,range(10))
    def test_parallel_limit(self):
        counts=[0,0];lock=threading.Lock()
        def f(x):
            with lock:
                counts[0]+=1;counts[1]=max(counts[1],counts[0])
            time.sleep(.01)
            with lock:counts[0]-=1
            return x
        ordered_parallel_map(f,range(16),4)
        self.assertLessEqual(counts[1],4)
        self.assertEqual(counts[0],0)
    def test_invalid_workers(self):
        for workers in (0,5,True):
            with self.assertRaises(ValueError):ordered_parallel_map(lambda x:x,[],workers)
    def rows(self):
        return [dict(key=k,task="stt",experiment=EXPERIMENT,
                teacher_boundary_policy=POLICY,training_released=False,
                score_backfill_allowed=False) for k in ("a/1","b/2")]
    def test_lane_order(self):
        rows=self.rows()
        self.assertEqual(validate_rows(rows,[{"key":"a/1"},{"key":"b/2"}],{"old/0"}),["a/1","b/2"])
        with self.assertRaises(ValueError):validate_rows(rows[::-1],[{"key":"a/1"},{"key":"b/2"}],set())
    def test_reused_rejected(self):
        with self.assertRaises(ValueError):validate_rows(self.rows(),[{"key":"a/1"},{"key":"b/2"}],{"a/1"})
    def test_old_policy_rejected(self):
        rows=self.rows();rows[0]["teacher_boundary_policy"]="missing_rvq_v1"
        with self.assertRaises(ValueError):validate_rows(rows,[{"key":"a/1"},{"key":"b/2"}],set())
    def test_old_experiment_rejected(self):
        rows=self.rows();rows[0]["experiment"]="evaluation_adaptation_failure_state_v1"
        with self.assertRaises(ValueError):validate_rows(rows,[{"key":"a/1"},{"key":"b/2"}],set())
    def test_release_rejected(self):
        rows=self.rows();rows[0]["training_released"]=True
        with self.assertRaises(ValueError):validate_rows(rows,[{"key":"a/1"},{"key":"b/2"}],set())
    def test_json_strict(self):
        for text in ('{"x":1,"x":2}','{"x":NaN}'):
            with self.assertRaises(ValueError):strict_loads(text)
    def events(self):
        out=[]
        for i,role in enumerate(("wa","lightnav","worker"),1):
            out.extend([dict(event="spawn",role=role,pid=i,process_group=i),
                        dict(event="exit",role=role,pid=i,returncode=-15 if role=="wa" else 0)])
        out.append(dict(event="worker_return",role="worker",pid=3,returncode=0))
        return out
    def test_exits(self):
        self.assertEqual(len(validate_processes(self.events())),3)
    def test_wrong_pid(self):
        events=self.events();events[1]["pid"]=99
        with self.assertRaises(ValueError):validate_processes(events)
    def test_failed_worker(self):
        events=self.events();events[5]["returncode"]=1
        with self.assertRaises(ValueError):validate_processes(events)
    def error(self,message):
        return dict(status="teacher_output_invalid",teacher="lightnav",
                    raw_error=dict(source="server_response",rc=500,seq=59,message=message,
                                   exception_type=None),result=None,complete=False,
                    replay_verified=False,fallback_executed=False,
                    training_eligible=False,training_released=False)
    def test_legacy_error(self):
        self.assertEqual(validate_output_error(self.error("Missing rvq act levels [0, 1] in 'text'")),"missing_rvq")
    def test_final_error(self):
        self.assertEqual(validate_output_error(self.error(
            "got 2 act levels, expected 3 from '<act_l0_1><act_l1_2>'")),"missing_final_l2")
    def test_unknown_error(self):
        for msg in ("unknown","got 2 act levels, expected 3 from '<act_l0_1><act_l2_2>'"):
            with self.assertRaises(ValueError):validate_output_error(self.error(msg))
    def test_transport_not_current_runtime(self):
        err=self.error("Missing rvq act levels [0] in 'x'");err["status"]="teacher_transport_error"
        with self.assertRaises(ValueError):validate_output_error(err)
    def test_no_fake_result_or_executed_fallback(self):
        for key,value in (("result",{}),("fallback_executed",True),("training_eligible",True)):
            err=self.error("Missing rvq act levels [0] in 'x'");err[key]=value
            with self.assertRaises(ValueError):validate_output_error(err)

if __name__=="__main__":unittest.main()
