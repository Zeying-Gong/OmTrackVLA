"""CPU tests for fixed val denominators, invalid-output penalties and provenance."""
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import generate_val4_protocol as protocol
import perception_metrics as metrics


def labels(count, visible=True):
    return [{"environment_step": k, "terminal_observation": k == count,
             "visibility_label_valid": True, "world_time_s": k*.1,
             "rgb_array_sha256": format(k, "064x"),
             "target": {"semantic_id_label_side_only": 100,
                        "visible": visible, "mask_area_pixels": 25 if visible else 0,
                        "bbox_label_valid": visible,
                        "bbox_xyxy_norm": [.1, .1, .6, .6] if visible else None}}
            for k in range(count+1)]


def predictions(gt, probability=1.):
    return [{"observation_environment_step": k,
             "source_rgb_array_sha256": row["rgb_array_sha256"],
             "source_world_time_s": row["world_time_s"],
             "visibility_probability": probability,
             "predicted_bbox_xyxy_norm": [.1, .1, .6, .6]}
            for k, row in enumerate(gt[:-1])]


class MetricTests(unittest.TestCase):
    def test_missing_outputs_stay_in_original_denominator(self):
        result = metrics.score_case(labels(3), [], 3, 100)
        self.assertEqual(result["prediction_denominator"], 3)
        self.assertEqual(result["visibility_at_0_9"]["invalid_gt_positive"], 3)
        self.assertEqual(result["bbox"]["visible_valid_gt_denominator"], 3)
        self.assertEqual(result["bbox"]["mean_iou_including_invalid_failures"], 0.)
        self.assertEqual(result["calibration"]["brier_with_invalid_penalty"], 1.)
        self.assertAlmostEqual(result["calibration"]["log_loss_with_invalid_penalty"], -math.log(1e-6))
        self.assertIsNone(result["calibration"]["ece_valid_probability_only"])

    def test_invalid_probability_and_bbox_domains(self):
        gt = labels(7)
        pred = predictions(gt)
        values = [float("nan"), float("inf"), -.1, 1.1, True, "0.9", None]
        boxes = [[0, 0, 0, 1], [-.1, 0, .5, .5], [0, 0, 1.1, 1],
                 [0, 0, .5, float("nan")], [True, 0, 1, 1], [0, 0, 1], None]
        for row, value, box in zip(pred, values, boxes):
            row["visibility_probability"], row["predicted_bbox_xyxy_norm"] = value, box
        result = metrics.score_case(gt, pred, 7, 100)
        self.assertEqual(result["calibration"]["invalid_probability_count"], 7)
        self.assertEqual(result["bbox"]["invalid_prediction_count"], 7)
        self.assertEqual(result["bbox"]["mean_iou_including_invalid_failures"], 0.)

    def test_rgb_and_clock_binding_failure_invalidates_both_heads(self):
        gt = labels(3); pred = predictions(gt)
        pred[0]["source_rgb_array_sha256"] = "f"*64
        pred[1]["source_world_time_s"] += .01
        pred[2]["source_world_time_s"] += 1e-10
        result = metrics.score_case(gt, pred, 3, 100)
        self.assertEqual(result["input_binding_failures"], {"wrong_source_rgb": 1, "wrong_source_time": 1})
        self.assertEqual(result["calibration"]["invalid_probability_count"], 2)
        self.assertAlmostEqual(result["bbox"]["mean_iou_including_invalid_failures"], 1/3)

    def test_duplicates_terminal_future_negative_bool_indices_fail(self):
        gt = labels(2)
        for index in [2, 3, -1, True, .5]:
            pred = predictions(gt); pred[0]["observation_environment_step"] = index
            with self.subTest(index=index), self.assertRaises(ValueError):
                metrics.score_case(gt, pred, 2, 100)
        pred = predictions(gt); pred.append(copy.deepcopy(pred[0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            metrics.score_case(gt, pred, 2, 100)

    def test_threshold_endpoints_calibration_and_coverage(self):
        gt = labels(5); gt[0]["target"].update(visible=False, mask_area_pixels=0, bbox_label_valid=False)
        pred = predictions(gt)
        for row, p in zip(pred, [0., 1., .9, .5, None]): row["visibility_probability"] = p
        result = metrics.score_case(gt, pred, 5, 100)
        self.assertEqual(result["visibility_at_0_9"]["tp"], 2)
        self.assertEqual(result["visibility_at_0_5"]["tp"], 3)
        calibration = result["calibration"]
        self.assertEqual(calibration["bins"][0]["count"], 1)
        self.assertEqual(calibration["bins"][9]["count"], 2)
        self.assertEqual(calibration["valid_probability_coverage"], .8)
        self.assertAlmostEqual(calibration["ece_valid_probability_only"], .15)
        self.assertAlmostEqual(calibration["brier_with_invalid_penalty"], 1.26/5)

    def test_prediction_visibility_never_filters_bbox_population(self):
        gt = labels(2); pred = predictions(gt, probability=0.)
        result = metrics.score_case(gt, pred, 2, 100)
        self.assertEqual(result["bbox"]["visible_valid_gt_denominator"], 2)
        self.assertEqual(result["bbox"]["mean_iou_including_invalid_failures"], 1.)
        self.assertEqual(result["bbox"]["joint_visibility_0_9_and_iou_0_5_fraction"], 0.)

    def test_degenerate_visible_gt_counts_visibility_only(self):
        gt = labels(1); gt[0]["target"].update(mask_area_pixels=1, bbox_label_valid=False,
                                               bbox_xyxy_norm=[.25, .5, .25, .5])
        result = metrics.score_case(gt, predictions(gt), 1, 100)
        self.assertEqual(result["visibility_at_0_9"]["tp"], 1)
        self.assertEqual(result["bbox"]["visible_degenerate_gt_count"], 1)
        self.assertEqual(result["bbox"]["visible_valid_gt_denominator"], 0)
        self.assertIsNone(result["bbox"]["mean_iou_including_invalid_failures"])

    def test_terminal_never_scored_but_must_be_present_and_valid(self):
        gt = labels(1); gt[-1]["target"].update(visible=False, mask_area_pixels=0, bbox_label_valid=False)
        result = metrics.score_case(gt, predictions(gt), 1, 100)
        self.assertEqual(result["visibility_at_0_9"]["total"], 1)
        with self.assertRaises(ValueError): metrics.score_case(gt[:-1], [], 1, 100)
        for key, value in [("terminal_observation", False), ("world_time_s", 0.),
                           ("rgb_array_sha256", "bad"), ("visibility_label_valid", False)]:
            altered = copy.deepcopy(gt); altered[-1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                metrics.score_case(altered, [], 1, 100)

    def test_gt_requires_real_bool_and_stable_target(self):
        for key, value in [("visible", 1), ("bbox_label_valid", 1), ("semantic_id_label_side_only", 101),
                           ("mask_area_pixels", 0)]:
            gt = labels(1); gt[0]["target"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                metrics.score_case(gt, [], 1, 100)

    def test_continuous_iou_no_plus_one_and_no_clipping(self):
        self.assertAlmostEqual(metrics.bbox_iou([0., 0., .5, 1.], [0., 0., 1., 1.]), .5)
        self.assertEqual(metrics.bbox_iou([-.1, 0., 1., 1.], [0., 0., 1., 1.]), 0.)

    def test_fixed_four_micro_macro_and_all_failure_case(self):
        cases = []
        for index, expected in enumerate(protocol.CASES):
            case_id, count = expected[0], expected[-1]
            gt = labels(count)
            cases.append(dict(case_id=case_id, expected_count=count, target_semantic_id=100,
                              labels=gt, predictions=predictions(gt) if index else []))
        result = metrics.score_suite(cases)
        self.assertEqual((result["case_denominator"], result["label_observation_denominator"],
                          result["prediction_denominator"]), (4, 229, 225))
        self.assertAlmostEqual(result["pooled_micro"]["visibility_at_0_9"]["accuracy_including_invalid_failures"], 135/225)
        self.assertEqual(result["equal_weight_case_macro"]["accuracy"], {"value": .75, "defined_case_count": 4})
        with self.assertRaises(ValueError): metrics.score_suite(cases[::-1])
        with self.assertRaises(ValueError): metrics.score_suite(cases[:3])


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = protocol.generate()
        cls.old = protocol.read(protocol.ROOT/"closed_loop_val4_review/phase2_val4_baseline_protocol_v3.json")
        case_dir = protocol.ROOT/"closed_loop_val4_review/evidence/outputs/takeover/permanent_val4_closed_loop_v1/phase2_baseline/dt_1300_episode3"
        cls.source, cls.status, cls.launch = (protocol.read(case_dir/name) for name in
                                              ("result.json", "execution_status.json", "launch_contract.json"))

    def test_real_evidence_fixed_denominators_and_baseline_only(self):
        self.assertEqual(self.plan["label_observation_denominator"], 229)
        self.assertEqual(self.plan["prediction_denominator"], 225)
        self.assertEqual([e["source_arm"] for e in self.plan["entries"]], ["phase2_baseline"]*4)
        self.assertTrue(all(e["permanent_partition_role"] == "val" and e["optimizer_input_allowed"] is False
                            for e in self.plan["entries"]))
        value = dict(self.plan); seal = value.pop("protocol_sha256")
        self.assertEqual(protocol.canonical_sha(value), seal)

    def test_wrong_checkpoint_identity_roles_counts_and_actions_fail(self):
        changes = [lambda s,t,l,c: s["loading"].update(checkpoint_sha256="0"*64),
                   lambda s,t,l,c: s.update(dataset_index=1301),
                   lambda s,t,l,c: c.update(partition_role="train"),
                   lambda s,t,l,c: t.update(actual_steps=89),
                   lambda s,t,l,c: s["steps"][0]["policy"]["action"].update(forward=float("nan")),
                   lambda s,t,l,c: s["steps"][0].update(step=2),
                   lambda s,t,l,c: s["initialization"].update(environment_step=1),
                   lambda s,t,l,c: s["assigned_humanoid_semantic_ids"].update({"1": 42})]
        for change in changes:
            s,t,l,c = copy.deepcopy((self.source, self.status, self.launch, self.old["cases"][0]))
            change(s,t,l,c)
            with self.subTest(change=change), self.assertRaises(ValueError):
                protocol.validate_baseline_source(s,t,l,c,protocol.CASES[0],self.old)

    def test_default_does_not_write_and_explicit_output_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"protocol.json"
            with contextlib.redirect_stdout(io.StringIO()): protocol.main([])
            self.assertFalse(output.exists())
            with contextlib.redirect_stdout(io.StringIO()): protocol.main(["--output", str(output)])
            before = output.read_bytes()
            self.assertEqual(json.loads(before)["case_denominator"], 4)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                protocol.main(["--output", str(output)])
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
