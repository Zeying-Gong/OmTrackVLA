import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from omtrackvla.models.end_to_end_phase1 import ArchitectureV1Phase1Model
from omtrackvla.training.end_to_end_v1 import _initialize_training_only_heads


class _PolicyStub(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(policy_dim=dim)
        self.scale = nn.Parameter(torch.ones(()))


class EndToEndTrainingTest(unittest.TestCase):
    def test_phase2_checkpoint_can_continue_training_only_heads(self):
        source_model = ArchitectureV1Phase1Model(_PolicyStub())
        source_state = {
            name: tensor.detach().clone()
            for name, tensor in source_model.state_dict().items()
            if not name.startswith("policy.")
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "phase2.ckpt"
            torch.save(
                {
                    "phase": 2,
                    "method": "architecture_v1_end_to_end",
                    "stage": "short_budget_source",
                    "training_only_heads": source_state,
                },
                checkpoint,
            )
            target_model = ArchitectureV1Phase1Model(_PolicyStub())
            for name, parameter in target_model.named_parameters():
                if not name.startswith("policy."):
                    parameter.data.zero_()

            report = _initialize_training_only_heads(target_model, checkpoint)

        self.assertEqual(report["kind"], "architecture_v1_phase2_checkpoint")
        self.assertEqual(report["checkpoint_phase"], 2)
        self.assertEqual(report["parameter_coverage"], 1.0)
        target_state = target_model.state_dict()
        for name, expected in source_state.items():
            self.assertTrue(torch.equal(target_state[name], expected), name)


if __name__ == "__main__":
    unittest.main()
