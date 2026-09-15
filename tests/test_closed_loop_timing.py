import ast
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "omtrackvla/evaluation/closed_loop_timing.py"
spec = importlib.util.spec_from_file_location("closed_loop_timing", MODULE)
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


class TimingUnitTests(unittest.TestCase):
    def test_old_rollout_outputs_rejected_even_without_existing_timing(self):
        for name in ('result.json','result.partial.json','rollout.mp4','rollout_frames',
                     'timing.json','timing.steps.jsonl'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                (root/'launcher.log').write_text('new launcher log is permitted')
                timing.assert_new_rollout_artifacts(root)
                path=root/name
                if name=='rollout_frames':
                    path.mkdir()
                else:
                    path.write_text('prior artifact')
                with self.assertRaisesRegex(FileExistsError,'existing rollout'):
                    timing.assert_new_rollout_artifacts(root)

    def test_stage_clock_is_additive_and_rejects_duplicates(self):
        ticks = iter((1_000_000, 4_000_000, 14_000_000))
        timer = timing.StageTimer(clock=lambda: next(ticks))
        self.assertEqual(timer.mark("preprocess"), 3.)
        self.assertEqual(timer.mark("model"), 10.)
        self.assertEqual(timer.total_ms, 13.)
        self.assertEqual(sum(timer.durations.values()), timer.total_ms)
        with self.assertRaisesRegex(ValueError, "unique"):
            timer.mark("model")

    def test_monotonic_clock_backwards_rejected(self):
        ticks = iter((10, 9))
        timer = timing.StageTimer(clock=lambda: next(ticks))
        with self.assertRaisesRegex(ValueError, "backwards"):
            timer.mark("invalid")
        with self.assertRaisesRegex(ValueError, "backwards"):
            timing.elapsed_ms(10, clock=lambda: 9)

    def test_percentile_empty_singleton_and_interpolation(self):
        self.assertIsNone(timing.percentile([], .95))
        self.assertEqual(timing.percentile([3.], .95), 3.)
        self.assertAlmostEqual(timing.percentile([0., 100.], .95), 95.)
        for values in ([float('nan')], [float('inf')], [-1.]):
            with self.assertRaises(ValueError):
                timing.describe(values)

    def test_parent_and_worker_are_not_double_counted_and_first_step_is_separate(self):
        def record(step, ipc, env):
            return {'step':step,'timing':{'complete':True,'parent_total_ms':ipc+env,
                    'parent_ms':{'policy_ipc':ipc,'env_step':env},
                    'worker':{'decision_ms':{'model':ipc-1.}}}}
        summary = timing.summarize_steps([record(1,101.,20.),record(2,11.,20.)])
        self.assertEqual(summary['all_steps']['parent_step_total']['sum_ms'],152.)
        self.assertEqual(summary['after_first_step']['parent_step_total']['mean_ms'],31.)
        self.assertEqual(summary['after_first_step']['worker_nested_stages_do_not_add_to_parent']['model']['mean_ms'],10.)

    def test_partial_records_are_not_reported_as_complete_totals(self):
        records=[{'step':1,'timing':{'complete':False,'parent_ms':{'env_step':700.}}}]
        summary=timing.summarize_steps(records)
        self.assertEqual(summary['all_steps']['complete_step_count'],0)
        self.assertIsNone(summary['all_steps']['parent_step_total']['mean_ms'])
        self.assertEqual(summary['all_steps']['parent_stages']['env_step']['sum_ms'],700.)

    def test_complete_record_inconsistent_sum_rejected(self):
        records=[{'timing':{'complete':True,'parent_total_ms':100.,'parent_ms':{'stage':2.}}}]
        with self.assertRaisesRegex(ValueError,"sum differs"):
            timing.summarize_steps(records)

    def test_jsonl_snapshot_does_not_claim_to_time_its_own_write(self):
        record={'step':1,'timing':{'complete':False,'parent_ms':{'report_write':30.,'stdout':2.}}}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'timing.steps.jsonl'
            timing.append_step_timing(path,record)
            value=json.loads(path.read_text())
        self.assertTrue(value['snapshot_complete_through_stdout'])
        self.assertEqual(value['parent_subtotal_ms_before_timing_jsonl_write'],32.)
        self.assertNotIn('timing_jsonl_write',value['timing']['parent_ms'])
        self.assertFalse(record['timing']['complete'])

    def test_diagnostic_contract_does_not_claim_deployment_or_pure_cuda_latency(self):
        contract=timing.timing_contract()
        self.assertFalse(contract['thor_latency_claim'])
        self.assertFalse(contract['deployment_target_latency_benchmark'])
        self.assertIn('no added CUDA synchronization',contract['cuda_timing'])


class RunnerBehaviorPreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base=ROOT/'omtrackvla/evaluation/end_to_end_closed_loop.py'
        if not base.is_file():
            base=Path((ROOT/'SOURCE_SHA256.txt').read_text().split('  ',1)[1].strip())
        cls.original=ast.parse(base.read_text(encoding='utf-8'))
        cls.instrumented=ast.parse((ROOT/'omtrackvla/evaluation/end_to_end_closed_loop_timed.py').read_text())

    def top(self,tree,name):
        return next(node for node in tree.body if getattr(node,'name',None)==name)

    def test_actions_calibration_metrics_helpers_drawing_encoding_and_policy_loading_unchanged(self):
        names=('ContinuousAction','WaypointActionAdapter','deployment_action','normalized_bbox',
               'habitat_camera_calibration','initialization_bbox','_draw_frame','encode_rollout_frames',
               '_load_policy_after_first_render','partial_rollout_result','_mean_absdiff','_camera_transform')
        for name in names:
            self.assertEqual(ast.dump(self.top(self.original,name)),ast.dump(self.top(self.instrumented,name)),name)

    def test_simulator_actions_and_sensor_render_call_counts_and_arguments_unchanged(self):
        def calls(tree,name):
            return [ast.dump(node) for node in ast.walk(tree)
                    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr==name]
        for name in ('step','get_sensor_observations','synchronize','imwrite'):
            self.assertEqual(calls(self.original,name),calls(self.instrumented,name),name)

    def test_model_input_dictionary_and_decision_signatures_unchanged(self):
        def model_inputs(tree):
            return [ast.dump(node.value) for node in ast.walk(tree) if isinstance(node,ast.Assign)
                    and any(isinstance(target,ast.Name) and target.id=='model_inputs' for target in node.targets)]
        self.assertEqual(model_inputs(self.original),model_inputs(self.instrumented))
        for name in ('ArchitectureV1HabitatController','ArchitectureV1PolicyWorker'):
            def decide(tree):
                return next(node for node in self.top(tree,name).body if getattr(node,'name',None)=='decide')
            self.assertEqual(ast.dump(decide(self.original).args),ast.dump(decide(self.instrumented).args))

    def test_official_summary_assignments_unchanged(self):
        def summaries(tree):
            return [ast.dump(node.value) for node in ast.walk(tree) if isinstance(node,ast.Assign)
                    and any(isinstance(target,ast.Name) and target.id=='summary' for target in node.targets)]
        self.assertEqual(summaries(self.original),summaries(self.instrumented))


if __name__=='__main__':
    unittest.main()
