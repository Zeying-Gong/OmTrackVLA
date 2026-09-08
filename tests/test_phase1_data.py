import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from omtrackvla.data.phase1 import ContractIdentityDataset, InternGeometryDataset
from omtrackvla.data.phase1_manifest import build_phase1_manifest, write_phase1_manifest
from omtrackvla.data.sage3d_sidecar import (
    GENERATION_SPEC_ID,
    episode_sidecar_path,
    sha256_file,
)
from scripts.validate_data_contract import load_json, validate_contract, validate_sample


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = validate_contract(load_json(REPOSITORY_ROOT / "configs/data_contract.json"))


def _image(path: Path, color=(64, 128, 192)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 50), color=color).save(path)


class Phase1DataTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        base = Path(self.temporary.name)
        self.intern = base / "intern"
        self.sage = base / "sage"
        self.sidecar = base / "sage-sidecar"
        self.tpt = base / "tpt"
        self._intern_fixture()
        self._sage_fixture()
        self._sage_sidecar_fixture()
        self._tpt_fixture()
        self.roots = {
            "intern_data_n1": self.intern,
            "sage3d_extracted": self.sage,
            "tpt_bench_clean_v2": self.tpt,
        }
        manifest = build_phase1_manifest(self.roots, seed=11)
        self.manifest = base / "phase1.json"
        write_phase1_manifest(self.manifest, manifest, self.roots.values())

    def tearDown(self):
        self.temporary.cleanup()

    def _intern_fixture(self):
        scene = self.intern / "group" / "scene"
        (scene / "meta").mkdir(parents=True)
        (scene / "meta" / "episodes_stats.jsonl").write_text("{}\n", encoding="utf-8")
        data = scene / "data" / "chunk-000"
        data.mkdir(parents=True)
        identity = np.eye(4, dtype=np.float32)
        target = identity.copy()
        target[1, 3] = 1.0
        table = pa.table(
            {
                "observation.camera_extrinsic": [identity.reshape(-1).tolist()] * 2,
                "action": [identity.reshape(-1).tolist(), target.reshape(-1).tolist()],
            }
        )
        pq.write_table(table, data / "episode_000000.parquet")
        rgb = scene / "videos" / "chunk-000" / "observation.images.rgb"
        _image(rgb / "episode_000000_000.jpg")
        _image(rgb / "episode_000000_001.jpg", (96, 144, 208))

    def _sage_fixture(self):
        episode = self.sage / "run-a" / "stt" / "0" / "camera"
        episode.mkdir(parents=True)
        (episode / "_ACCEPTED").write_text("accepted\n", encoding="utf-8")
        (episode / "quality.json").write_text(json.dumps({"status": "accepted"}), encoding="utf-8")
        steps = [
            {"step": 0, "visible": True, "bbox": [10, 5, 40, 45]},
            {"step": 1, "visible": True, "bbox": [12, 5, 42, 45]},
            {"step": 2, "visible": False, "bbox": None},
        ]
        (episode / "derived.json").write_text(json.dumps({"steps": steps}), encoding="utf-8")
        (episode / "camera_info.json").write_text("{}", encoding="utf-8")
        for index in range(3):
            _image(episode / "rgb" / f"{index:05d}.jpg", (60 + index, 100, 180))
        self.sage.mkdir(exist_ok=True)
        (self.sage / "index.json").write_text(
            json.dumps(
                {
                    "eps": [
                        {
                            "run": "run-a",
                            "mode": "stt",
                            "ep": "0",
                            "cam": "camera",
                            "path": "run-a/stt/0/camera",
                            "steps": 3,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    def _sage_sidecar_fixture(self):
        source_path = "run-a/stt/0/camera"
        episode = self.sage / source_path
        boxes = ([10, 5, 40, 45], [12, 5, 42, 45], None)
        labels = {
            "schema_version": 1,
            "dataset_id": "sage3d_extracted",
            "generation_spec_id": GENERATION_SPEC_ID,
            "source_path": source_path,
            "source_derived_sha256": sha256_file(episode / "derived.json"),
            "source_camera_info_sha256": sha256_file(episode / "camera_info.json"),
            "steps": [
                {
                    "step": index,
                    "visible": box is not None,
                    "bbox_xyxy": box,
                    "visibility_reason": "visible" if box is not None else "out_of_view",
                }
                for index, box in enumerate(boxes)
            ],
        }
        label_path = episode_sidecar_path(self.sidecar, source_path)
        label_path.parent.mkdir(parents=True)
        label_path.write_text(json.dumps(labels), encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "dataset_id": "sage3d_extracted",
            "generation_spec_id": GENERATION_SPEC_ID,
            "selection": "full",
            "source_index_sha256": sha256_file(self.sage / "index.json"),
            "episodes": {
                source_path: {
                    "steps": 3,
                    "visible_steps": 2,
                    "sidecar_sha256": sha256_file(label_path),
                }
            },
        }
        manifest_path = self.sidecar / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        (self.sidecar / "admission.json").write_text(
            json.dumps(
                {"status": "passed", "manifest_sha256": sha256_file(manifest_path)}
            ),
            encoding="utf-8",
        )

    def _tpt_fixture(self):
        sequence = self.tpt / "0000"
        sequence.mkdir(parents=True)
        table = pa.table(
            {
                "video_idx": [0, 1, 2],
                "vid_pts_ms": [0.0, 33.0, 66.0],
                "bbox_qv": [[10.0, 5.0, 30.0, 40.0], [11.0, 5.0, 30.0, 40.0], [0.0] * 4],
                "is_exist": pa.array([1, 1, 0], type=pa.int8()),
                "is_behind_glass": pa.array([0, 0, 1], type=pa.int8()),
            }
        )
        pq.write_table(table, sequence / "frames.parquet")
        for index in range(3):
            _image(sequence / "rgb_frames" / f"frame_{index:06d}.jpg", (80, 120 + index, 200))

    def test_sage_identity_record_validates_and_writes_no_crop(self):
        dataset = ContractIdentityDataset(
            self.manifest,
            "test_locked",
            roots=self.roots,
            datasets=("sage3d_extracted",),
            sage3d_sidecar=self.sidecar,
            image_size=32,
        )
        record = dataset.get_record(0)
        result = validate_sample(record, CONTRACT, {"sage3d_extracted": self.sage})
        self.assertEqual(result["sample_role"], "identity_auxiliary")
        self.assertFalse(any(self.sage.rglob("_target_crops")))
        item = dataset[0]
        self.assertEqual(tuple(item["frame0"].shape), (3, 32, 32))
        self.assertEqual(tuple(item["frame1"].shape), (3, 32, 32))
        self.assertEqual(tuple(item["history"].shape), (7, 3, 32, 32))
        self.assertFalse(item["history_mask"].any())

    def test_sage_identity_refuses_unadmitted_source_labels(self):
        with self.assertRaises(ValueError):
            ContractIdentityDataset(
                self.manifest,
                "test_locked",
                roots=self.roots,
                datasets=("sage3d_extracted",),
                image_size=32,
            )

    def test_tpt_identity_keeps_later_bbox_out_of_model_inputs(self):
        dataset = ContractIdentityDataset(
            self.manifest,
            "test_locked",
            roots=self.roots,
            datasets=("tpt_bench_clean_v2",),
            image_size=32,
        )
        record = dataset.get_record(0)
        validate_sample(record, CONTRACT, {"tpt_bench_clean_v2": self.tpt})
        self.assertNotIn("target_bbox", json.dumps(record["model_inputs"]))
        self.assertIsNotNone(record["supervision"]["auxiliary_labels"]["target_bbox_xyxy_norm"])
        self.assertFalse(any(self.tpt.rglob("_target_refs")))

    def test_balanced_identity_index_alternates_sources(self):
        dataset = ContractIdentityDataset(
            self.manifest,
            "test_locked",
            roots=self.roots,
            sage3d_sidecar=self.sidecar,
            image_size=32,
        )
        sources = [
            dataset.get_record(dataset.balanced_index(index))["source"]["dataset_id"]
            for index in range(4)
        ]
        self.assertEqual(
            sources,
            [
                "sage3d_extracted",
                "tpt_bench_clean_v2",
                "sage3d_extracted",
                "tpt_bench_clean_v2",
            ],
        )

    def test_intern_geometry_pair_is_canonical(self):
        dataset = InternGeometryDataset(
            self.manifest,
            "test_locked",
            root=self.intern,
            image_size=32,
            maximum_gap=1,
            history_size=8,
            samples_per_epoch=1,
            max_units=1,
        )
        item = dataset[0]
        np.testing.assert_allclose(item["motion"].numpy(), [1.0, 0.0, 0.0], atol=1e-6)
        self.assertEqual(item["dataset_id"], "intern_data_n1")


if __name__ == "__main__":
    unittest.main()
