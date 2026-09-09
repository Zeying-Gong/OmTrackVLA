import copy
import unittest

from scripts.merge_sage3d_perception_cache import merge_manifests


HASH = "a" * 64


def _manifest(index: int):
    return {
        "schema_version": 1,
        "status": "complete",
        "dataset_id": "sage3d_extracted",
        "split": "train",
        "policy_spec_id": "sage3d-policy-se2-30hz-v1",
        "source_index_sha256": HASH,
        "sidecar_manifest_sha256": HASH,
        "policy_admission_sha256": HASH,
        "split_manifest_sha256": HASH,
        "front_end": {"frozen": True},
        "selection": {
            "unit_count": 10,
            "requested_episodes": 1,
            "cached_episodes": 1,
            "record_stride": 3,
            "max_units": None,
            "max_episodes": None,
            "num_shards": 2,
            "shard_index": index,
            "test_locked_used": False,
        },
        "episodes": {
            f"run-{index}/mode/0/camera": {
                "cache_path": f"episodes/run-{index}/mode/0/camera.json",
                "cache_sha256": HASH,
                "anchors": [1],
            }
        },
        "skipped": [],
    }


class MergePhase2CacheTest(unittest.TestCase):
    def test_merge_preserves_shard_paths_and_counts(self):
        merged = merge_manifests(
            [_manifest(0), _manifest(1)],
            ["shards/shard-000-of-002", "shards/shard-001-of-002"],
        )
        self.assertEqual(merged["selection"]["cached_episodes"], 2)
        self.assertEqual(merged["selection"]["merged_from_shards"], 2)
        self.assertEqual(
            merged["episodes"]["run-1/mode/0/camera"]["cache_path"],
            "shards/shard-001-of-002/episodes/run-1/mode/0/camera.json",
        )

    def test_merge_rejects_invariant_drift_and_limited_shards(self):
        second = _manifest(1)
        second["front_end"] = {"frozen": False}
        with self.assertRaisesRegex(ValueError, "front_end"):
            merge_manifests([_manifest(0), second], ["a", "b"])

        second = _manifest(1)
        second["selection"]["max_episodes"] = 1
        with self.assertRaisesRegex(ValueError, "development-limited"):
            merge_manifests([_manifest(0), second], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
