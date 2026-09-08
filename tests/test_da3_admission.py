import unittest

from omtrackvla.geometry.da3_admission import (
    apply_confidence_gate,
    camera_scale_stratum,
    choose_confidence_threshold,
    fit_global_translation_scale,
    fit_stratified_translation_scales,
    score_clip,
    summarize_admission,
)


def trajectory_record(clip_id: str, confidence: float, predicted_scale: float = 1.0):
    return {
        "clip_id": clip_id,
        "group": "group-a",
        "confidence_score": confidence,
        "predicted_se2_unscaled": [
            [0.0, 0.0, 0.0],
            [0.1 * predicted_scale, 0.0, 0.03],
            [0.2 * predicted_scale, 0.02 * predicted_scale, 0.06],
            [0.3 * predicted_scale, 0.04 * predicted_scale, 0.09],
        ],
        "ground_truth_se2": [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.03],
            [0.4, 0.04, 0.06],
            [0.6, 0.08, 0.09],
        ],
    }


class DA3AdmissionTest(unittest.TestCase):
    def test_global_scale_balances_clips(self):
        records = [
            trajectory_record(f"clip-{index}", 5.0, predicted_scale=1.0)
            for index in range(8)
        ]
        result = fit_global_translation_scale(records)
        self.assertAlmostEqual(result["scale"], 2.0)
        self.assertEqual(result["calibration_clip_count"], 8)
        self.assertAlmostEqual(result["relative_iqr"], 0.0)

    def test_global_scale_rejects_too_few_moving_clips(self):
        with self.assertRaisesRegex(ValueError, "at least 8"):
            fit_global_translation_scale([trajectory_record("one", 2.0)])

    def test_camera_stratified_scales_are_fitted_independently(self):
        records = []
        for stratum, predicted_scale in (("d435i", 1.0), ("zed", 2.0)):
            for index in range(8):
                record = trajectory_record(f"{stratum}-{index}", 4.0, predicted_scale)
                record["scale_stratum"] = stratum
                records.append(record)
        result = fit_stratified_translation_scales(records)
        self.assertAlmostEqual(result["strata"]["d435i"]["scale"], 2.0)
        self.assertAlmostEqual(result["strata"]["zed"]["scale"], 1.0)
        self.assertEqual(camera_scale_stratum("hm3d_d435i"), "d435i")
        self.assertEqual(camera_scale_stratum("3dfront_zed"), "zed")

    def test_score_clip_uses_frozen_scale(self):
        result = score_clip(trajectory_record("clip", 4.0), 2.0)
        self.assertTrue(result["motion_geometry_valid"])
        self.assertAlmostEqual(result["translation_error_m_mean"], 0.0)
        self.assertAlmostEqual(result["yaw_error_rad_mean"], 0.0)
        self.assertAlmostEqual(result["scale_relative_error_median"], 0.0)

    def test_confidence_threshold_is_fitted_on_quality(self):
        records = []
        for index, confidence in enumerate((1.0, 2.0, 3.0, 4.0)):
            record = score_clip(trajectory_record(str(index), confidence), 2.0)
            if confidence < 3.0:
                record["translation_error_m_mean"] = 0.5
            records.append(record)
        result = choose_confidence_threshold(
            records, minimum_coverage=0.5, maximum_bad_rate=0.0
        )
        self.assertTrue(result["criteria_met"])
        self.assertEqual(result["threshold"], 3.0)
        gated = apply_confidence_gate(records, result["threshold"])
        self.assertEqual(sum(record["admitted"] for record in gated), 2)

    def test_motion_degeneracy_fails_closed(self):
        record = score_clip(trajectory_record("clip", 4.0), 0.01)
        self.assertFalse(record["motion_geometry_valid"])
        gated = apply_confidence_gate([record], 1.0)
        self.assertFalse(gated[0]["admitted"])
        self.assertEqual(gated[0]["admission_reason"], "motion_degenerate")
        summary = summarize_admission(gated)
        self.assertEqual(summary["coverage"], 0.0)


if __name__ == "__main__":
    unittest.main()
