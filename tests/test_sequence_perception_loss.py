"""CPU checks for independent perception-label masks and legacy compatibility."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F


candidate = os.environ.get("OMTRACKVLA_SEQUENCE_LOSS_CANDIDATE")
if candidate:
    spec = importlib.util.spec_from_file_location("sequence_loss_candidate", Path(candidate))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    phase3_sequence_loss = module.phase3_sequence_loss
else:
    from omtrackvla.training.sequence_training import phase3_sequence_loss


def outputs(steps=3):
    return {
        "waypoints": torch.zeros(1, steps, 8, 2, requires_grad=True),
        "bbox_pred": torch.full((1, steps, 4), .6, requires_grad=True),
        "visibility_logit": torch.zeros(1, steps, 1, requires_grad=True),
        "binding_logit": torch.zeros(1, steps, 1, requires_grad=True),
        "stop_logit": torch.zeros(1, steps, 1, requires_grad=True),
    }


class SequencePerceptionLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_new_visibility_mask_is_independent_of_identity(self):
        predicted = outputs(2)
        batch = {"visibility_label_valid": torch.tensor([[True, True]]),
                 "identity_label_valid": torch.zeros(1, 2, dtype=torch.bool),
                 "target_visible": torch.tensor([[1., 0.]])}
        loss, _ = phase3_sequence_loss(predicted, batch, {"visibility": 1.})
        loss.backward()
        self.assertLess(float(predicted["visibility_logit"].grad[0, 0]), 0.)
        self.assertGreater(float(predicted["visibility_logit"].grad[0, 1]), 0.)
        self.assertIsNone(predicted["binding_logit"].grad)

    def test_new_bbox_mask_is_independent_of_identity_and_visibility_mask(self):
        predicted = outputs(1)
        batch = {"bbox_label_valid": torch.ones(1, 1, dtype=torch.bool),
                 "visibility_label_valid": torch.zeros(1, 1, dtype=torch.bool),
                 "identity_label_valid": torch.zeros(1, 1, dtype=torch.bool),
                 "target_visible": torch.ones(1, 1), "target_bbox": torch.zeros(1, 1, 4)}
        loss, terms = phase3_sequence_loss(predicted, batch, {"bbox": 1., "visibility": 1.})
        loss.backward()
        self.assertGreater(float(predicted["bbox_pred"].grad.abs().sum()), 0.)
        self.assertEqual(float(terms["visibility"]), 0.)
        self.assertEqual(float(predicted["visibility_logit"].grad.abs().sum()), 0.)

    def test_invalid_bbox_nan_values_have_exact_zero_gradient(self):
        predicted = outputs(2)
        predicted["bbox_pred"].data[:, 1] = float("nan")
        boxes = torch.zeros(1, 2, 4); boxes[:, 1] = float("nan")
        batch = {"bbox_label_valid": torch.tensor([[True, False]]),
                 "target_visible": torch.ones(1, 2), "target_bbox": boxes}
        loss, _ = phase3_sequence_loss(predicted, batch, {"bbox": 1.})
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(predicted["bbox_pred"].grad[:, 0].abs().sum()), 0.)
        torch.testing.assert_close(predicted["bbox_pred"].grad[:, 1], torch.zeros(1, 4), rtol=0, atol=0)

    def test_single_pixel_visible_target_supervises_only_visibility(self):
        predicted = outputs(1)
        predicted["bbox_pred"].data.fill_(float("nan"))
        predicted["waypoints"].data.fill_(float("nan"))
        batch = {"visibility_label_valid": torch.ones(1, 1, dtype=torch.bool),
                 "bbox_label_valid": torch.zeros(1, 1, dtype=torch.bool),
                 "identity_label_valid": torch.zeros(1, 1, dtype=torch.bool),
                 "target_visible": torch.ones(1, 1),
                 "target_bbox": torch.tensor([[[.25, .5, .25, .5]]])}
        loss, terms = phase3_sequence_loss(predicted, batch, {"visibility": 1., "bbox": 1.})
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(terms["bbox"]), 0.)
        loss.backward()
        self.assertLess(float(predicted["visibility_logit"].grad.item()), 0.)
        torch.testing.assert_close(predicted["bbox_pred"].grad, torch.zeros(1, 1, 4), rtol=0, atol=0)
        self.assertIsNone(predicted["binding_logit"].grad)

    def test_explicit_false_masks_do_not_fall_back_to_true_identity(self):
        predicted = outputs(1)
        for key in ("bbox_pred", "visibility_logit", "waypoints"):
            predicted[key].data.fill_(float("nan"))
        batch = {"bbox_label_valid": torch.zeros(1, 1, dtype=torch.bool),
                 "visibility_label_valid": torch.zeros(1, 1, dtype=torch.bool),
                 "identity_label_valid": torch.ones(1, 1, dtype=torch.bool)}
        loss, _ = phase3_sequence_loss(predicted, batch, {"bbox": 1., "visibility": 1.})
        self.assertEqual(float(loss), 0.)
        loss.backward()
        for key in ("bbox_pred", "visibility_logit"):
            torch.testing.assert_close(predicted[key].grad, torch.zeros_like(predicted[key]), rtol=0, atol=0)

    def test_invalid_nan_visibility_labels_are_excluded_before_bce(self):
        predicted = outputs(2)
        predicted["visibility_logit"].data[:, 1] = float("nan")
        batch = {"visibility_label_valid": torch.tensor([[True, False]]),
                 "target_visible": torch.tensor([[1., float("nan")]])}
        loss, _ = phase3_sequence_loss(predicted, batch, {"visibility": 1.})
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertLess(float(predicted["visibility_logit"].grad[0, 0]), 0.)
        self.assertEqual(float(predicted["visibility_logit"].grad[0, 1]), 0.)

    def test_invisible_target_has_no_bbox_supervision(self):
        predicted = outputs(1)
        batch = {"bbox_label_valid": torch.ones(1, 1, dtype=torch.bool), "target_visible": torch.zeros(1, 1)}
        loss, _ = phase3_sequence_loss(predicted, batch, {"bbox": 1.})
        loss.backward()
        self.assertEqual(float(loss), 0.)
        self.assertEqual(float(predicted["bbox_pred"].grad.abs().sum()), 0.)

    def test_supervision_mask_still_excludes_unselected_perception_steps(self):
        predicted = outputs(3)
        batch = {"supervision_mask": torch.tensor([[False, False, True]]),
                 "visibility_label_valid": torch.ones(1, 3, dtype=torch.bool),
                 "target_visible": torch.tensor([[float("nan"), float("nan"), 1.]])}
        loss, _ = phase3_sequence_loss(predicted, batch, {"visibility": 1.})
        loss.backward()
        self.assertEqual(float(predicted["visibility_logit"].grad[:, :2].abs().sum()), 0.)
        self.assertLess(float(predicted["visibility_logit"].grad[0, 2]), 0.)

    def test_unknown_binding_and_stop_do_not_become_negative_labels(self):
        predicted = outputs(1)
        for key in ("binding_logit", "stop_logit"):
            predicted[key].data.fill_(float("nan"))
        batch = {"visibility_label_valid": torch.ones(1, 1, dtype=torch.bool),
                 "target_visible": torch.ones(1, 1),
                 "binding_label_valid": torch.zeros(1, 1, dtype=torch.bool),
                 "stop_label_valid": torch.zeros(1, 1, dtype=torch.bool)}
        loss, terms = phase3_sequence_loss(predicted, batch,
            {"visibility": 1., "identity_or_binding": 1., "stop": 1., "waypoint": 0.})
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, key in (("identity_or_binding", "binding_logit"), ("stop", "stop_logit")):
            self.assertEqual(float(terms[name]), 0.)
            self.assertEqual(float(predicted[key].grad.abs().sum()), 0.)
        self.assertIsNone(predicted["waypoints"].grad)

    def test_missing_waypoint_labels_are_not_invented(self):
        with self.assertRaises(KeyError):
            phase3_sequence_loss(outputs(1), {}, {"waypoint": 1.})

    def test_legacy_clean_loss_and_gradients_match_previous_formula(self):
        predicted = outputs(3)
        reference = {key: value.detach().clone().requires_grad_() for key, value in predicted.items()}
        batch = {"supervision_mask": torch.tensor([[True, True, False]]),
                 "identity_label_valid": torch.tensor([[1., 1., 0.]]),
                 "target_visible": torch.tensor([[1., 0., float("nan")]]),
                 "target_bbox": torch.tensor([[[0., .1, .2, .3], [float("nan")]*4, [float("nan")]*4]]),
                 "binding_label_valid": torch.tensor([[True, False, False]]),
                 "binding_target": torch.tensor([[1., float("nan"), float("nan")]]),
                 "stop_label_valid": torch.tensor([[False, True, False]]),
                 "stop_target": torch.tensor([[float("nan"), 0., float("nan")]])}
        weights = {"bbox": 2., "visibility": .7, "identity_or_binding": .3, "stop": .2}
        loss, terms = phase3_sequence_loss(predicted, batch, weights)
        selected = batch["supervision_mask"]
        identity = selected & batch["identity_label_valid"].bool()
        bbox = identity & batch["target_visible"].bool()
        expected = {"bbox": F.smooth_l1_loss(reference["bbox_pred"][bbox], batch["target_bbox"][bbox]),
                    "visibility": F.binary_cross_entropy_with_logits(reference["visibility_logit"].squeeze(-1)[identity], batch["target_visible"][identity])}
        for name, key, mask, target in (("identity_or_binding", "binding_logit", "binding_label_valid", "binding_target"),
                                         ("stop", "stop_logit", "stop_label_valid", "stop_target")):
            valid = selected & batch[mask]
            expected[name] = F.binary_cross_entropy_with_logits(reference[key].squeeze(-1)[valid], batch[target][valid])
        old_loss = sum(weights[key]*value for key, value in expected.items())
        torch.testing.assert_close(loss, old_loss, rtol=0, atol=0)
        for key in terms: torch.testing.assert_close(terms[key], expected[key], rtol=0, atol=0)
        loss.backward(); old_loss.backward()
        for key in ("bbox_pred", "visibility_logit", "binding_logit", "stop_logit"):
            torch.testing.assert_close(predicted[key].grad, reference[key].grad, rtol=0, atol=0)

    def test_legacy_label_only_default_still_supervises_visibility(self):
        predicted = outputs(1)
        loss, _ = phase3_sequence_loss(predicted, {"target_visible": torch.ones(1, 1)}, {"visibility": 1.})
        torch.testing.assert_close(loss, torch.tensor(0.6931471805599453))

    def test_nan_validity_masks_are_rejected_instead_of_cast_true(self):
        for key, name in (("visibility_label_valid", "visibility"), ("bbox_label_valid", "bbox"),
                          ("identity_label_valid", "visibility"), ("supervision_mask", "visibility")):
            with self.subTest(mask=key):
                with self.assertRaisesRegex(ValueError, "finite 0/1"):
                    phase3_sequence_loss(outputs(1), {key: torch.full((1, 1), float("nan"))}, {name: 1.})

    def test_fractional_mask_and_wrong_shape_are_rejected(self):
        for value, message in ((torch.tensor([[.5]]), "finite 0/1"), (torch.ones(1), "shape")):
            with self.assertRaisesRegex(ValueError, message):
                phase3_sequence_loss(outputs(1), {"bbox_label_valid": value}, {"bbox": 1.})

    def test_valid_bbox_cannot_infer_visibility_from_nan(self):
        with self.assertRaisesRegex(ValueError, "target_visible"):
            phase3_sequence_loss(outputs(1), {"bbox_label_valid": torch.ones(1, 1, dtype=torch.bool),
                "target_visible": torch.full((1, 1), float("nan"))}, {"bbox": 1.})


if __name__ == "__main__":
    unittest.main()
