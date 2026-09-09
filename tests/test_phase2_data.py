import copy
import tempfile
import unittest
from pathlib import Path

from omtrackvla.data.phase2 import (
    CONDITION_MODES,
    build_policy_record,
    conditioned_visual_features,
    validate_perception_cache_manifest,
    validate_perception_cache_payload,
)
from omtrackvla.data.sage3d_policy import POLICY_SPEC_ID
from scripts.validate_data_contract import load_json, validate_contract, validate_sample


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = validate_contract(load_json(REPOSITORY_ROOT / "configs/data_contract.json"))
HASH = "a" * 64


def _steps(count=32):
    return [
        {
            "step": index,
            "robot_pos": [0.1 * index, 0.0, 0.0],
            "robot_yaw": 0.0,
            "target_pos": [2.0 + 0.05 * index, 0.5, 0.0],
        }
        for index in range(count)
    ]


def _labels(count=32):
    return [
        {
            "step": index,
            "visible": True,
            "bbox_xyxy": [10.0, 5.0, 40.0, 45.0],
            "visibility_reason": "visible",
        }
        for index in range(count)
    ]


def _front_end():
    return {
        "detector_architecture": "fasterrcnn_resnet50_fpn_v2",
        "detector_weights_sha256": HASH,
        "reid_weights_sha256": HASH,
        "reid_code_sha256": HASH,
        "fusion_weights_sha256": HASH,
        "frozen": True,
    }


def _cache_manifest():
    return {
        "schema_version": 1,
        "status": "complete",
        "dataset_id": "sage3d_extracted",
        "split": "train",
        "policy_spec_id": POLICY_SPEC_ID,
        "source_index_sha256": HASH,
        "sidecar_manifest_sha256": HASH,
        "policy_admission_sha256": HASH,
        "split_manifest_sha256": HASH,
        "front_end": _front_end(),
        "selection": {
            "cached_episodes": 1,
            "max_units": None,
            "max_episodes": None,
            "test_locked_used": False,
        },
        "episodes": {
            "run/mode/episode/camera": {
                "cache_path": "episodes/run/mode/episode/camera.json",
                "cache_sha256": HASH,
                "anchors": [1],
            }
        },
    }


class Phase2DataTest(unittest.TestCase):
    def test_nonvisual_modes_mask_all_frozen_visual_features(self):
        perception = {"relative_xy": [2.0, 0.5], "confidence": 0.999, "visible": True}
        self.assertEqual(
            conditioned_visual_features(perception, "uwb_only"),
            ([0.0, 0.0], 0.0, False),
        )
        self.assertEqual(
            conditioned_visual_features(perception, "safe_stop"),
            ([0.0, 0.0], 0.0, False),
        )
        self.assertEqual(
            conditioned_visual_features(perception, "visual_only"),
            ([2.0, 0.5], 0.999, True),
        )

    def test_policy_records_obey_all_four_mode_contracts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = {}
            for mode in CONDITION_MODES:
                record = build_policy_record(
                    root=root,
                    episode_relative="run/mode/episode/camera",
                    split_unit_id="run",
                    episode_id="run/mode/episode/camera",
                    steps=_steps(),
                    labels=_labels(),
                    image_size=(50, 50),
                    initial_index=0,
                    anchor_index=5,
                    condition_mode=mode,
                    history_size=4,
                )
                self.assertEqual(validate_sample(record, CONTRACT)["condition_mode"], mode)
                records[mode] = record

        self.assertTrue(records["visual_uwb"]["model_inputs"]["visual_initialization"]["valid"])
        self.assertTrue(records["visual_uwb"]["model_inputs"]["uwb_target"]["valid"])
        self.assertFalse(records["visual_only"]["model_inputs"]["uwb_target"]["valid"])
        self.assertFalse(records["uwb_only"]["model_inputs"]["visual_initialization"]["valid"])
        self.assertTrue(records["uwb_only"]["model_inputs"]["uwb_target"]["valid"])
        self.assertEqual(
            records["safe_stop"]["supervision"]["expert_trajectory"]["waypoints_base_xy_m"],
            [[0.0, 0.0]] * 8,
        )

    def test_visual_history_consumes_anchor_bbox_only_at_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = build_policy_record(
                root=Path(temporary),
                episode_relative="run/mode/episode/camera",
                split_unit_id="run",
                episode_id="run/mode/episode/camera",
                steps=_steps(),
                labels=_labels(),
                image_size=(50, 50),
                initial_index=0,
                anchor_index=8,
                condition_mode="visual_only",
                history_size=4,
            )
        self.assertEqual(len(record["model_inputs"]["rgb_history"]), 4)
        self.assertEqual(
            record["model_inputs"]["visual_initialization"]["rgb_path"],
            record["model_inputs"]["rgb_history"][0]["rgb_path"],
        )
        self.assertNotIn("bbox_xyxy_norm", record["model_inputs"]["rgb_history"][-1])

    def test_cache_manifest_binds_every_admitted_input(self):
        value = _cache_manifest()
        validated = validate_perception_cache_manifest(
            value,
            split="train",
            source_index_sha256=HASH,
            sidecar_manifest_sha256=HASH,
            policy_admission_sha256=HASH,
            split_manifest_sha256=HASH,
        )
        self.assertEqual(validated["front_end"]["reid_code_sha256"], HASH)

        changed = copy.deepcopy(value)
        changed["policy_admission_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "policy_admission_sha256 mismatch"):
            validate_perception_cache_manifest(
                changed,
                split="train",
                source_index_sha256=HASH,
                sidecar_manifest_sha256=HASH,
                policy_admission_sha256=HASH,
                split_manifest_sha256=HASH,
            )

        partial = copy.deepcopy(value)
        partial["selection"]["max_episodes"] = 1
        with self.assertRaisesRegex(ValueError, "development override"):
            validate_perception_cache_manifest(
                partial,
                split="train",
                source_index_sha256=HASH,
                sidecar_manifest_sha256=HASH,
                policy_admission_sha256=HASH,
                split_manifest_sha256=HASH,
            )

    def test_cache_payload_rejects_missing_or_reordered_records(self):
        payload = {
            "schema_version": 1,
            "policy_spec_id": POLICY_SPEC_ID,
            "source_path": "run/mode/episode/camera",
            "source_derived_sha256": HASH,
            "source_sidecar_sha256": HASH,
            "front_end": _front_end(),
            "initial_index": 0,
            "anchors": [1],
            "records": [
                {
                    "anchor_index": 1,
                    "visible": False,
                    "relative_xy": [2.0, 0.5],
                    "confidence": 0.0,
                    "predicted_bbox_xyxy": None,
                }
            ],
        }
        validate_perception_cache_payload(
            payload,
            source_path="run/mode/episode/camera",
            anchors=[1],
            front_end=_front_end(),
        )
        payload["records"][0]["anchor_index"] = 2
        with self.assertRaisesRegex(ValueError, "record order"):
            validate_perception_cache_payload(
                payload,
                source_path="run/mode/episode/camera",
                anchors=[1],
                front_end=_front_end(),
            )


if __name__ == "__main__":
    unittest.main()
