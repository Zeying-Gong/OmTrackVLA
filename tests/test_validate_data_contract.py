import copy
import tempfile
import unittest
from pathlib import Path

from scripts.validate_data_contract import (
    ContractViolation,
    load_json,
    validate_contract,
    validate_sample,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = validate_contract(load_json(REPOSITORY_ROOT / "configs/data_contract.json"))


def policy_sample() -> dict:
    return {
        "schema_version": 1,
        "sample_id": "habitat/scene-a/episode-1/anchor-1",
        "sample_role": "policy",
        "source": {
            "dataset_id": "habitat_sim",
            "split_unit_id": "scene-a/seed-7",
            "episode_id": "episode-1",
            "anchor_index": 1,
            "adapter_version": "habitat-v1",
        },
        "model_inputs": {
            "condition_mode": "visual_uwb",
            "anchor_timestamp_ns": 2_000_000_000,
            "rgb_sensor_valid": True,
            "rgb_history": [
                {"rgb_path": "rgb/000000.jpg", "timestamp_ns": 1_000_000_000, "valid": True},
                {"rgb_path": "rgb/000001.jpg", "timestamp_ns": 2_000_000_000, "valid": True},
            ],
            "visual_initialization": {
                "valid": True,
                "rgb_path": "rgb/000000.jpg",
                "bbox_xyxy_norm": [0.2, 0.1, 0.5, 0.9],
                "timestamp_ns": 1_000_000_000,
            },
            "uwb_target": {
                "valid": True,
                "measurement_kind": "simulated_uwb",
                "relative_position_base_xy_m": [2.0, -0.4],
                "covariance_base_xy_m2": [[0.04, 0.0], [0.0, 0.09]],
                "quality_01": 0.8,
                "source_timestamp_ns": 1_900_000_000,
                "receive_timestamp_ns": 1_950_000_000,
                "age_s": 0.1,
                "los_state": "los",
            },
        },
        "routing_metadata": {
            "target_tag_id": "target-1",
            "calibration_id": None,
            "simulation_spec_id": "uwb-noise-v1",
        },
        "supervision": {
            "expert_trajectory": {
                "valid": True,
                "waypoints_base_xy_m": [[0.1 * index, 0.0] for index in range(8)],
                "time_offsets_s": [0.3 * index for index in range(8)],
                "valid_mask": [True] * 8,
            },
            "auxiliary_labels": {
                "target_track_id": "scene-a-person-3",
                "target_bbox_xyxy_norm": [0.22, 0.1, 0.52, 0.9],
                "target_visible": True,
                "target_position_base_xy_m": [2.1, -0.35],
                "occlusion_state": "visible",
                "uwb_error_base_xy_m": [-0.1, -0.05],
            },
            "safety": {"stop_required": False, "reason": "none"},
        },
        "provenance": {
            "source_record": "episodes/episode-1.json",
            "transform_spec_id": "habitat-base-v1",
            "clock_spec_id": "habitat-clock-v1",
            "generation_spec_id": "oracle-v1",
        },
    }


def identity_sample() -> dict:
    sample = policy_sample()
    sample["sample_id"] = "tpt/0000/anchor-1"
    sample["sample_role"] = "identity_auxiliary"
    sample["source"].update(
        {
            "dataset_id": "tpt_bench_clean_v2",
            "split_unit_id": "0000",
            "episode_id": "0000",
            "adapter_version": "tpt-v1",
        }
    )
    sample["model_inputs"]["condition_mode"] = "visual_only"
    sample["model_inputs"]["uwb_target"] = {
        "valid": False,
        "measurement_kind": "none",
        "relative_position_base_xy_m": None,
        "covariance_base_xy_m2": None,
        "quality_01": None,
        "source_timestamp_ns": None,
        "receive_timestamp_ns": None,
        "age_s": None,
        "los_state": None,
    }
    sample["routing_metadata"] = {
        "target_tag_id": None,
        "calibration_id": None,
        "simulation_spec_id": None,
    }
    sample["supervision"]["expert_trajectory"] = {
        "valid": False,
        "waypoints_base_xy_m": [None] * 8,
        "time_offsets_s": None,
        "valid_mask": [False] * 8,
    }
    sample["supervision"]["auxiliary_labels"]["target_position_base_xy_m"] = None
    sample["supervision"]["auxiliary_labels"]["uwb_error_base_xy_m"] = None
    sample["provenance"] = {
        "source_record": "0000/frames.parquet",
        "transform_spec_id": None,
        "clock_spec_id": None,
        "generation_spec_id": None,
    }
    return sample


class DataContractValidationTest(unittest.TestCase):
    def test_contract_and_policy_sample_are_valid(self):
        result = validate_sample(policy_sample(), CONTRACT)
        self.assertEqual(result["sample_role"], "policy")
        self.assertEqual(result["condition_mode"], "visual_uwb")

    def test_identity_auxiliary_sample_is_valid(self):
        result = validate_sample(identity_sample(), CONTRACT)
        self.assertEqual(result["dataset_id"], "tpt_bench_clean_v2")

    def test_source_admission_gate_blocks_tpt_policy(self):
        sample = policy_sample()
        sample["source"]["dataset_id"] = "tpt_bench_clean_v2"
        with self.assertRaisesRegex(ContractViolation, "not admitted"):
            validate_sample(sample, CONTRACT)

    def test_sage3d_policy_is_admitted_with_explicit_simulation_provenance(self):
        sample = policy_sample()
        sample["source"].update(
            {
                "dataset_id": "sage3d_extracted",
                "split_unit_id": "run-1",
                "episode_id": "run-1/stt/0/camera",
                "adapter_version": "sage3d-policy-v1",
            }
        )
        sample["provenance"].update(
            {
                "transform_spec_id": "sage3d-habitat-world-to-base-se2-v1",
                "clock_spec_id": "sage3d-control-step-30hz-v1",
                "generation_spec_id": "sage3d-pose-derived-noiseless-uwb-v1",
            }
        )
        result = validate_sample(sample, CONTRACT)
        self.assertEqual(result["dataset_id"], "sage3d_extracted")
        self.assertEqual(result["sample_role"], "policy")

    def test_later_bbox_cannot_enter_model_inputs(self):
        sample = policy_sample()
        sample["model_inputs"]["rgb_history"][1]["target_bbox"] = [0.1, 0.1, 0.2, 0.2]
        with self.assertRaisesRegex(ContractViolation, "prohibited in model_inputs"):
            validate_sample(sample, CONTRACT)

    def test_visual_initialization_must_bind_history_zero(self):
        sample = policy_sample()
        sample["model_inputs"]["visual_initialization"]["timestamp_ns"] = 2_000_000_000
        with self.assertRaisesRegex(ContractViolation, "history index zero"):
            validate_sample(sample, CONTRACT)

    def test_uwb_age_and_covariance_are_checked(self):
        bad_age = policy_sample()
        bad_age["model_inputs"]["uwb_target"]["age_s"] = 0.2
        with self.assertRaisesRegex(ContractViolation, "age_s"):
            validate_sample(bad_age, CONTRACT)

        bad_covariance = policy_sample()
        bad_covariance["model_inputs"]["uwb_target"]["covariance_base_xy_m2"] = [
            [1.0, 2.0],
            [2.0, 1.0],
        ]
        with self.assertRaisesRegex(ContractViolation, "positive semidefinite"):
            validate_sample(bad_covariance, CONTRACT)

    def test_valid_uwb_requires_transform_and_clock_specs(self):
        sample = policy_sample()
        sample["sample_role"] = "identity_auxiliary"
        sample["source"]["dataset_id"] = "sage3d_extracted"
        sample["supervision"]["expert_trajectory"] = {
            "valid": False,
            "waypoints_base_xy_m": [None] * 8,
            "time_offsets_s": None,
            "valid_mask": [False] * 8,
        }
        sample["provenance"]["transform_spec_id"] = None
        sample["provenance"]["clock_spec_id"] = None
        with self.assertRaisesRegex(ContractViolation, "valid UWB requires"):
            validate_sample(sample, CONTRACT)

    def test_waypoint_mask_is_a_valid_prefix(self):
        sample = policy_sample()
        sample["supervision"]["expert_trajectory"]["valid_mask"][3] = False
        sample["supervision"]["expert_trajectory"]["waypoints_base_xy_m"][3] = None
        with self.assertRaisesRegex(ContractViolation, "contiguous valid prefix"):
            validate_sample(sample, CONTRACT)

        sample = policy_sample()
        sample["supervision"]["expert_trajectory"]["waypoints_base_xy_m"][0] = [0.1, 0.0]
        with self.assertRaisesRegex(ContractViolation, "anchor pose"):
            validate_sample(sample, CONTRACT)

    def test_visible_label_requires_bbox(self):
        sample = identity_sample()
        sample["supervision"]["auxiliary_labels"]["target_bbox_xyxy_norm"] = None
        with self.assertRaisesRegex(ContractViolation, "visible targets require"):
            validate_sample(sample, CONTRACT)

    def test_rgb_failure_requires_safe_stop(self):
        sample = policy_sample()
        sample["model_inputs"]["rgb_sensor_valid"] = False
        sample["model_inputs"]["rgb_history"][-1]["valid"] = False
        sample["model_inputs"]["rgb_history"][-1]["rgb_path"] = None
        with self.assertRaisesRegex(ContractViolation, "visual_uwb requires"):
            validate_sample(sample, CONTRACT)
        sample["model_inputs"]["condition_mode"] = "safe_stop"
        sample["supervision"]["safety"] = {
            "stop_required": True,
            "reason": "rgb_sensor_failure",
        }
        self.assertEqual(validate_sample(sample, CONTRACT)["condition_mode"], "safe_stop")

    def test_paths_are_normalized_and_optionally_checked(self):
        sample = policy_sample()
        sample["provenance"]["source_record"] = "../episode.json"
        with self.assertRaisesRegex(ContractViolation, "normalized relative path"):
            validate_sample(sample, CONTRACT)

        sample["provenance"]["source_record"] = "C:/dataset/episode.json"
        with self.assertRaisesRegex(ContractViolation, "normalized relative path"):
            validate_sample(sample, CONTRACT)

        sample = policy_sample()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "rgb").mkdir()
            (root / "episodes").mkdir()
            (root / "rgb/000000.jpg").write_bytes(b"frame zero")
            (root / "rgb/000001.jpg").write_bytes(b"frame one")
            (root / "episodes/episode-1.json").write_text("{}", encoding="utf-8")
            validate_sample(sample, CONTRACT, {"habitat_sim": root})
            (root / "rgb/000001.jpg").unlink()
            with self.assertRaisesRegex(ContractViolation, "missing, empty"):
                validate_sample(sample, CONTRACT, {"habitat_sim": root})

    def test_contract_cannot_weaken_input_policy(self):
        weakened = copy.deepcopy(CONTRACT)
        weakened["model_input_policy"]["forbidden_keys_recursive"].remove("instruction")
        with self.assertRaisesRegex(ContractViolation, "mandatory prohibited"):
            validate_contract(weakened)

        weakened = copy.deepcopy(CONTRACT)
        weakened["source_adapters"]["tpt_bench_clean_v2"]["eligible_roles"].append(
            "policy"
        )
        with self.assertRaisesRegex(ContractViolation, "admission gate changed"):
            validate_contract(weakened)


if __name__ == "__main__":
    unittest.main()
