import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_evt_reid_full_val import summarize_full, write_csv


def _result(method: str, index: int, correct: bool) -> dict:
    return {
        "dataset_index": index,
        "episode_id": str(index),
        "scene_id": f"scene-{index}",
        "controller": "reactive",
        "controller_input": "oracle-pointgoal",
        "perception": method,
        "summary": {"status": "MaxSteps"},
        "steps": [{
            "visible": True,
            "distance_m": 1.0 + index,
            "perception_gt_visible": True,
            "perception_confidence": 0.9,
            "target_selection_correct": correct,
            "target_bbox_iou": 0.8 if correct else 0.0,
            "perception_memory_updated": correct,
            "perception_candidates": [{"target_match": True}],
        }],
    }


def _manifest(method: str) -> dict:
    return {
        "shard_id": 0,
        "num_shards": 1,
        "dataset_episodes": 2,
        "max_steps": 300,
        "save_steps": True,
        "controller": "reactive",
        "target_mode": "point",
        "controller_input": "oracle-pointgoal",
        "person_detector_architecture": "fasterrcnn_resnet50_fpn_v2",
        "person_reid_backend": method,
    }


class SummarizeEvtReidFullValTest(unittest.TestCase):
    def test_complete_paired_report_ranks_by_micro_f1(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for method in ("osnet", "kpr"):
                episode_dir = root / method / "stt" / "val" / "episodes"
                episode_dir.mkdir(parents=True)
                (episode_dir.parent / "shard_000_manifest.json").write_text(
                    json.dumps(_manifest(method)), encoding="utf-8"
                )
                for index in range(2):
                    correct = method == "osnet" or index == 0
                    (episode_dir / f"{index}.json").write_text(
                        json.dumps(_result(method, index, correct)), encoding="utf-8"
                    )
            report = summarize_full(
                root,
                expected={"stt": 2},
                methods=("osnet", "kpr"),
                tasks=("stt",),
            )
            self.assertTrue(report["complete"])
            self.assertTrue(report["paired_protocol"]["consistent"])
            self.assertEqual(report["winner"], "osnet")
            self.assertEqual(report["methods"]["osnet"]["aggregate"]["completed_episodes"], 2)
            csv_path = root / "report.csv"
            write_csv(report, csv_path)
            self.assertTrue(csv_path.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_missing_episode_fails_completeness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode_dir = root / "osnet" / "stt" / "val" / "episodes"
            episode_dir.mkdir(parents=True)
            (episode_dir.parent / "shard_000_manifest.json").write_text(
                json.dumps(_manifest("osnet")), encoding="utf-8"
            )
            (episode_dir / "0.json").write_text(
                json.dumps(_result("osnet", 0, True)), encoding="utf-8"
            )
            report = summarize_full(
                root,
                expected={"stt": 2},
                methods=("osnet",),
                tasks=("stt",),
            )
            self.assertFalse(report["complete"])
            self.assertEqual(report["methods"]["osnet"]["aggregate"]["missing_episodes"], 1)


if __name__ == "__main__":
    unittest.main()
