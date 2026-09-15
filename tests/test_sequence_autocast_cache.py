"""CPU BF16 regression for no-grad burn-in under one outer autocast scope.

Run against the production sequence_training module. With a susceptible torch
autocast cache, the unpatched helper has a valid loss graph through direct_bias
but loses gradients to cached Linear weights used during burn-in. No simulator,
checkpoint, GPU or fabricated privileged model inputs are involved.
"""
import unittest

import torch
from torch import nn

from omtrackvla.training.sequence_training import unroll_policy_sequence


class TinyLinearPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 2)
        # An uncast pointwise parameter keeps loss.backward valid even if the
        # cached BF16 projection weights have lost their autograd connection.
        self.direct_bias = nn.Parameter(torch.full((2,), .01))
        with torch.no_grad():
            self.projection.weight.fill_(.5)
            self.projection.bias.fill_(.1)
        self.grad_modes = []

    def forward(self, *, ego_rgb, hidden_state=None, **unused):
        self.grad_modes.append(torch.is_grad_enabled())
        x = ego_rgb.mean(dim=(1, 2, 3, 4))[:, None]
        projected = self.projection(x)
        previous = torch.zeros_like(projected) if hidden_state is None else hidden_state
        state = projected + .25 * previous + self.direct_bias
        return {
            "hidden_state": state,
            "target_memory_next": state,
            "binding_logit": state.mean(-1, keepdim=True),
            "waypoints": torch.stack([state, 2. * state], dim=1),
        }


def inputs():
    return {
        "initial_rgb": torch.ones(1, 3, 2, 2),
        "initial_bbox": torch.tensor([[0., 0., 1., 1.]]),
        "ego_rgb": torch.ones(1, 10, 4, 3, 2, 2),
        "visual_initialization_valid": torch.ones(1),
        "rgb_valid": torch.ones(1), "binding_valid": torch.ones(1),
        "uwb_xy": torch.zeros(1, 2),
        "uwb_covariance_xy": torch.eye(2)[None],
        "uwb_quality": torch.zeros(1), "uwb_age_s": torch.zeros(1),
        "uwb_valid": torch.zeros(1),
        "camera_intrinsics": torch.eye(3)[None],
        "camera_from_base": torch.eye(4)[None],
    }


def anchor_backward(policy, *, cache_enabled):
    # One enclosing context deliberately reproduces the real preflight's
    # pattern: six no-grad calls followed by four learning calls.
    with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=cache_enabled):
        output, _ = unroll_policy_sequence(policy, inputs(), burn_in_steps=6)
        loss = output["waypoints"][:, -1].float().square().mean()
    loss.backward()
    return output, loss


class BurnInAutocastCacheTest(unittest.TestCase):
    def test_cpu_bf16_burn_in_preserves_linear_parameter_gradients(self):
        policy = TinyLinearPolicy()
        output, loss = anchor_backward(policy, cache_enabled=True)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(policy.grad_modes, [False]*6 + [True]*4)
        self.assertEqual(tuple(output["waypoints"].shape), (1, 4, 2, 2))
        self.assertIsNotNone(policy.direct_bias.grad)
        for name, parameter in policy.projection.named_parameters():
            self.assertIsNotNone(parameter.grad, f"burn-in autocast cache detached projection.{name}")
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.float().norm()), 0.)

    def test_fixed_cache_matches_cache_disabled_forward_and_gradients(self):
        cached, uncached = TinyLinearPolicy(), TinyLinearPolicy()
        output, loss = anchor_backward(cached, cache_enabled=True)
        reference, reference_loss = anchor_backward(uncached, cache_enabled=False)
        torch.testing.assert_close(output["waypoints"], reference["waypoints"], rtol=0., atol=0.)
        torch.testing.assert_close(loss, reference_loss, rtol=0., atol=0.)
        for name, parameter in cached.named_parameters():
            expected = dict(uncached.named_parameters())[name].grad
            self.assertIsNotNone(parameter.grad, f"cached projection lost {name} gradient")
            # Cached casts can sum branch gradients in BF16 before casting
            # back; uncached casts can sum FP32 branch gradients instead.
            torch.testing.assert_close(parameter.grad, expected, rtol=.01, atol=1e-4)

    def test_outer_no_grad_remains_no_grad(self):
        policy = TinyLinearPolicy()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            output, _ = unroll_policy_sequence(policy, inputs(), burn_in_steps=6)
        self.assertEqual(policy.grad_modes, [False]*10)
        self.assertFalse(output["waypoints"].requires_grad)
        self.assertTrue(all(parameter.grad is None for parameter in policy.parameters()))


if __name__ == "__main__":
    unittest.main()
