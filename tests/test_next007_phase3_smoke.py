import unittest
from pathlib import Path

from scripts.smoke_next007_phase3_sample import validate_sample
from scripts.train_next007_phase3 import _optimizer_parameter_category


def sample_value():
    return {
        "schema_version": 1,
        "stage": "next007_phase3_model_visited_relabel_smoke",
        "formal_training_eligible": False,
        "source": {"split": "val", "test_locked_used": False},
        "model_inputs": {
            "condition_mode": "visual_only",
            "rgb_history": [
                {"environment_step": step} for step in range(25, 29)
            ],
            "uwb": {"valid": False},
        },
        "supervision": {
            "expert_trajectory": {
                "waypoints_base_xy_m": [
                    [0.1 * index, 0.0] for index in range(8)
                ],
                "valid_mask": [True] * 8,
            },
            "gt_used_only_on_label_side": True,
        },
    }


class Next007Phase3SmokeTest(unittest.TestCase):
    def test_top_level_gru_uses_gru_optimizer_group(self):
        self.assertEqual(
            _optimizer_parameter_category("gru.weight_ih_l0"), "gru"
        )
        self.assertEqual(
            _optimizer_parameter_category("nested.gru.weight_hh_l0"), "gru"
        )
        self.assertEqual(
            _optimizer_parameter_category("da3.blocks.10.adapter.down.weight"),
            "adapter",
        )
        self.assertEqual(
            _optimizer_parameter_category("fusion.0.weight"), "policy"
        )

    def test_validation_accepts_exact_visual_only_smoke_contract(self):
        value = sample_value()
        self.assertIs(validate_sample(value, Path("sample.json"), 4), value)

    def test_validation_rejects_val_sample_as_formal_training_data(self):
        value = sample_value()
        value["formal_training_eligible"] = True
        with self.assertRaisesRegex(ValueError, "eligibility mismatch"):
            validate_sample(value, Path("sample.json"), 4)

    def test_validation_rejects_noncontiguous_history(self):
        value = sample_value()
        value["model_inputs"]["rgb_history"][-1]["environment_step"] = 30
        with self.assertRaisesRegex(ValueError, "contiguous"):
            validate_sample(value, Path("sample.json"), 4)

    def test_validation_accepts_train_candidate_for_minibatch_only(self):
        value = sample_value()
        value["formal_training_eligible"] = True
        value["source"]["split"] = "train"
        self.assertIs(
            validate_sample(
                value,
                Path("sample.json"),
                4,
                expected_split="train",
                require_formal_eligible=True,
            ),
            value,
        )


if __name__ == "__main__":
    unittest.main()
