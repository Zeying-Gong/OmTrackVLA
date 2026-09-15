import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_sage3d_perception_cache import _canonical_rejection_record


class BuildPhase2CacheTest(unittest.TestCase):
    def test_only_explicit_double_evidence_rejection_can_be_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            (episode / "quality.json").write_text(
                json.dumps(
                    {
                        "status": "rejected",
                        "rejection_reasons": ["tracking_rate", "end_reason"],
                    }
                ),
                encoding="utf-8",
            )
            record = _canonical_rejection_record(episode, "run/stt/0/camera")
            self.assertEqual(record["reason"], "source_not_canonically_accepted")
            self.assertEqual(record["quality_status"], "rejected")
            self.assertEqual(
                record["source_rejection_reasons"], ["tracking_rate", "end_reason"]
            )

            (episode / "_ACCEPTED").touch()
            self.assertIsNone(_canonical_rejection_record(episode, "run/stt/0/camera"))

    def test_ambiguous_or_accepted_quality_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            self.assertIsNone(_canonical_rejection_record(episode, "run/stt/0/camera"))
            (episode / "quality.json").write_text(
                json.dumps({"status": "accepted"}), encoding="utf-8"
            )
            self.assertIsNone(_canonical_rejection_record(episode, "run/stt/0/camera"))


if __name__ == "__main__":
    unittest.main()
