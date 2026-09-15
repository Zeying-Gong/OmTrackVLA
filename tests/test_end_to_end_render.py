import unittest

import numpy as np
import torch

from omtrackvla.evaluation.end_to_end_render import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    render_pretrain_dashboard,
)
from scripts.render_end_to_end_v1_checkpoint import checkpoint_backend_label


class EndToEndRenderTest(unittest.TestCase):
    def values(self):
        generator = torch.Generator().manual_seed(7)
        batch = {
            "initial_rgb": torch.rand(1, 3, 28, 42, generator=generator),
            "initial_bbox": torch.tensor([[0.2, 0.1, 0.7, 0.9]]),
            "ego_rgb": torch.rand(1, 4, 3, 28, 42, generator=generator),
            "target_waypoints": torch.tensor(
                [[[0.0, 0.0], [0.1, 0.0], [0.2, 0.01], [0.3, 0.02], [0.4, 0.03], [0.5, 0.04], [0.6, 0.05], [0.7, 0.06]]]
            ),
            "uwb_xy": torch.tensor([[2.0, 0.1]]),
            "uwb_covariance_xy": torch.tensor([[[0.04, 0.0], [0.0, 0.04]]]),
        }
        outputs = {
            "target_attention": torch.softmax(torch.randn(1, 6, generator=generator), dim=1),
            "scene_attention": torch.softmax(torch.randn(1, 6, generator=generator), dim=1),
            "uwb_patch_bias": torch.randn(1, 6, generator=generator),
            "uwb_mean_uv": torch.tensor([[21.0, 14.0]]),
            "uwb_covariance_uv": torch.tensor([[[16.0, 0.0], [0.0, 9.0]]]),
            "uwb_projectable": torch.tensor([True]),
            "uwb_gamma": torch.tensor([0.8]),
            "uwb_rho_fov": torch.tensor([0.9]),
            "waypoints": torch.tensor(
                [[[0.0, 0.0], [0.03, -0.02], [0.06, -0.01], [0.08, 0.0], [0.11, 0.01], [0.13, 0.02], [0.15, 0.02], [0.17, 0.03]]]
            ),
            "stop_logit": torch.tensor([[0.2]]),
            "xi_hat": torch.tensor([[0.01, -0.02, 0.03, 0.99]]),
        }
        metadata = {
            "backend": "test",
            "episode": "run/stt/0/camera",
            "initial_index": 0,
            "anchor_index": 8,
            "da3_coverage": 1.0,
            "waypoint_loss": 0.1,
        }
        return batch, outputs, metadata

    def test_dashboard_has_fixed_decodable_shape_and_visible_content(self):
        batch, outputs, metadata = self.values()
        image = render_pretrain_dashboard(
            batch, outputs, metadata, grid_shape=(2, 3)
        )
        self.assertEqual(image.shape, (CANVAS_HEIGHT, CANVAS_WIDTH, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(float(image.std()), 15.0)
        self.assertGreater(int(np.count_nonzero(image)), image.size // 3)

    def test_dashboard_rejects_attention_shape_mismatch(self):
        batch, outputs, metadata = self.values()
        outputs["target_attention"] = torch.ones(1, 5)
        with self.assertRaisesRegex(ValueError, "expected 6"):
            render_pretrain_dashboard(
                batch, outputs, metadata, grid_shape=(2, 3)
            )

    def test_trained_checkpoint_metadata_changes_dashboard_copy(self):
        batch, outputs, metadata = self.values()
        pretrain = render_pretrain_dashboard(
            batch, outputs, metadata, grid_shape=(2, 3)
        )
        trained_metadata = dict(metadata)
        trained_metadata.update(policy_state="trained", checkpoint_step=4096)
        trained = render_pretrain_dashboard(
            batch, outputs, trained_metadata, grid_shape=(2, 3)
        )
        single_step_metadata = dict(trained_metadata)
        single_step_metadata["temporal_fusion"] = "single_step"
        single_step = render_pretrain_dashboard(
            batch, outputs, single_step_metadata, grid_shape=(2, 3)
        )
        self.assertEqual(trained.shape, pretrain.shape)
        self.assertFalse(np.array_equal(trained, pretrain))
        self.assertFalse(np.array_equal(single_step, trained))

    def test_checkpoint_backend_label_preserves_initialization_provenance(self):
        direct = checkpoint_backend_label(
            {
                "stage": "next026_direct_phase2_core_v1",
                "initialization": {"kind": "direct_phase2"},
            }
        )
        phase1_init = checkpoint_backend_label(
            {
                "stage": "next026_phase1_init_phase2_core_v1",
                "initialization": {
                    "kind": "architecture_v1_checkpoint",
                    "checkpoint_phase": 1,
                },
            }
        )
        fallback = checkpoint_backend_label({"stage": "future_stage"})
        self.assertIn("direct DA3 initialization", direct)
        self.assertIn("Phase 1 initialization", phase1_init)
        self.assertNotEqual(direct, phase1_init)
        self.assertIn("future_stage", fallback)


if __name__ == "__main__":
    unittest.main()
