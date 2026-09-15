"""CPU-only tests; run with python -m unittest -v test_recovery_state.

Optional Torch tests use CUDA_VISIBLE_DEVICES='' and one CPU thread. No model,
dataset, network or GPU is accessed.
"""
import importlib.util
import json
import os
import pickle
import random
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from omtrackvla.training.recovery_state import (
    BestMetric, ResumeError, RunDirectory, assert_contract, atomic_write,
    canonical_hash, capture_rng, code_hash, file_hash, restore_rng,
    resume_payload, run_contract, validate_resume,
)


def contract(world_size=1, steps=20):
    return run_contract(effective_config={"maximum_steps": steps, "base": {"a": 1}},
                        code_sha256="a" * 64, data_hashes={"train": "b" * 64},
                        world_size=world_size, runtime={"python": "test"})


class RecoveryStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_effective_steps_and_world_size_reject_resume(self):
        expected = contract()
        for changed in (contract(steps=10), contract(world_size=2),
                        {**expected, "code_sha256": "d" * 64},
                        {**expected, "data_sha256": "e" * 64},
                        {**expected, "runtime_sha256": "f" * 64}):
            with self.subTest(changed=changed), self.assertRaises(ResumeError):
                assert_contract(changed, expected)
        with self.assertRaises(ResumeError):
            assert_contract({}, expected)

    def test_code_hash_detects_dirty_source_and_rejects_escape(self):
        source = self.root / "train.py"
        source.write_text("x = 1", encoding="utf-8")
        before = code_hash(self.root, ["train.py"])
        source.write_text("x = 2", encoding="utf-8")
        self.assertNotEqual(before, code_hash(self.root, ["train.py"]))
        with self.assertRaises(ValueError):
            code_hash(self.root, ["../outside.py"])

    def test_atomic_failure_preserves_previous_checkpoint_and_cleans_temp(self):
        target = self.root / "last.ckpt"
        target.write_bytes(b"previous good checkpoint")

        def broken_writer(handle):
            handle.write(b"partial corrupt checkpoint")
            raise OSError("simulated disk full")

        with self.assertRaises(OSError):
            atomic_write(target, broken_writer)
        self.assertEqual(target.read_bytes(), b"previous good checkpoint")
        self.assertEqual(list(self.root.iterdir()), [target])
        atomic_write(target, lambda handle: pickle.dump({"step": 3}, handle))
        with target.open("rb") as handle:
            self.assertEqual(pickle.load(handle), {"step": 3})

    def test_completed_run_cannot_be_overwritten_or_resumed(self):
        path = self.root / "run"
        with RunDirectory(path, contract()) as run:
            last = path / "last.ckpt"
            last.write_bytes(b"complete checkpoint")
            run.complete(global_step=20, artifacts={"last.ckpt": file_hash(last)})
        before = (path / "run_state.json").read_bytes()
        with self.assertRaises(FileExistsError):
            with RunDirectory(path, contract()):
                self.fail("fresh directory overwrite was allowed")
        with self.assertRaises(ResumeError):
            with RunDirectory(path, contract(), resume=True):
                self.fail("complete directory resume was allowed")
        self.assertEqual(before, (path / "run_state.json").read_bytes())
        self.assertFalse((path / ".run.lock").exists())

    def test_failure_releases_lock_and_can_resume_matching_contract(self):
        path = self.root / "run"
        with self.assertRaisesRegex(RuntimeError, "training failed"):
            with RunDirectory(path, contract()):
                raise RuntimeError("training failed")
        state = json.loads((path / "run_state.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertIn("training failed", state["error"])
        self.assertFalse((path / ".run.lock").exists())
        before = (path / "run_state.json").read_bytes()
        with self.assertRaises(ResumeError):
            with RunDirectory(path, contract(steps=10), resume=True):
                pass
        self.assertEqual(before, (path / "run_state.json").read_bytes())
        with RunDirectory(path, contract(), resume=True):
            self.assertEqual(json.loads((path / "run_state.json").read_text())["status"], "running")
        self.assertEqual(json.loads((path / "run_state.json").read_text())["status"], "failed")

    def test_competing_writer_rejected_without_removing_original_lock(self):
        path = self.root / "run"
        with RunDirectory(path, contract()) as first:
            before = first.lock_path.read_bytes()
            with self.assertRaises(FileExistsError):
                with RunDirectory(path, contract(), resume=True):
                    pass
            self.assertEqual(first.lock_path.read_bytes(), before)

    def test_stale_lock_is_not_automatically_deleted(self):
        path = self.root / "run"
        with RunDirectory(path, contract()):
            pass
        lock = path / ".run.lock"
        lock.write_text('{"pid":999999999,"host":"another-host","token":"old"}')
        with self.assertRaises(FileExistsError):
            with RunDirectory(path, contract(), resume=True):
                pass
        self.assertTrue(lock.exists())

    def test_completion_checks_artifacts_before_marking_complete(self):
        path = self.root / "run"
        with RunDirectory(path, contract()) as run:
            artifact = path / "last.ckpt"
            artifact.write_bytes(b"checkpoint")
            with self.assertRaises(ValueError):
                run.complete(global_step=20, artifacts={"last.ckpt": "wrong"})
            self.assertFalse(run.completed)
        self.assertEqual(json.loads((path / "run_state.json").read_text())["status"], "failed")

    def test_per_rank_rng_restores_the_next_random_sample(self):
        rank_states, expected = [], []
        for rank in range(2):
            random.seed(22 + rank)
            rank_states.append({"rng": capture_rng(rank=rank),
                                "data_state": {"kind": "step_random_sampling"}})
            expected.append([random.random() for _ in range(5)])
        payload = resume_payload(contract=contract(world_size=2), global_step=7,
                                 rank_states=rank_states)
        for rank in range(2):
            restore_rng(payload["rank_states"][rank]["rng"], rank=rank)
            self.assertEqual(expected[rank], [random.random() for _ in range(5)])
        with self.assertRaises(ResumeError):
            restore_rng(rank_states[0]["rng"], rank=1)
        payload["rank_states"].reverse()
        with self.assertRaises(ResumeError):
            validate_resume(payload, contract(world_size=2))

    def test_rng_restore_rejects_missing_rank_or_generator(self):
        state = capture_rng(rank=0)
        with self.assertRaises(ResumeError):
            restore_rng(state, rank=0, generators={"dataloader": object()})
        with self.assertRaises(ResumeError):
            resume_payload(contract=contract(world_size=2), global_step=0,
                           rank_states=[{"rng": state, "data_state": {}}])

    def test_best_uses_finite_validation_and_preserves_improving_step(self):
        best = BestMetric("recovery_success_rate", mode="max")
        self.assertTrue(best.improves(0.7, split="val", step=10))
        best.commit(0.7, split="val", step=10)
        self.assertFalse(best.improves(0.6, split="val", step=20))
        self.assertEqual(asdict(best)["step"], 10)
        for split, value in (("train", 0.9), ("test_locked", 0.9), ("val", float("nan"))):
            with self.subTest(split=split, value=value), self.assertRaises(ValueError):
                best.improves(value, split=split, step=30)

    def test_hash_is_order_invariant_but_nonfinite_config_rejected(self):
        self.assertEqual(canonical_hash({"a": 1, "b": 2}), canonical_hash({"b": 2, "a": 1}))
        with self.assertRaises(ValueError):
            canonical_hash({"learning_rate": float("nan")})

    def test_restored_best_metric_rejects_partial_or_nonfinite_state(self):
        for value, step in ((0.5, None), (None, 3), (float("nan"), 3), (0.5, -1)):
            with self.subTest(value=value, step=step), self.assertRaises(ValueError):
                BestMetric("ade", value=value, step=step)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is not installed on this CPU host")
class TorchCPUResumeTests(unittest.TestCase):
    def test_interrupted_adam_dropout_matches_uninterrupted(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(42)
        random.seed(42)
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(0.3),
                                    torch.nn.Linear(4, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        def step():
            optimizer.zero_grad(set_to_none=True)
            inputs = torch.randn(2, 3) * random.random()
            loss = model(inputs).square().mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss.detach().clone()

        step()
        step()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "last.ckpt"
            checkpoint = {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "exact_resume": resume_payload(contract=contract(), global_step=2,
                    rank_states=[{"rng": capture_rng(rank=0, torch=torch),
                                  "data_state": {"kind": "step_random_sampling"}}]),
            }
            atomic_write(path, lambda handle: torch.save(checkpoint, handle))
            expected_loss = step()
            expected_model = {key: value.clone() for key, value in model.state_dict().items()}
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            validate_resume(loaded["exact_resume"], contract())
            model.load_state_dict(loaded["model"])
            optimizer.load_state_dict(loaded["optimizer"])
            scheduler.load_state_dict(loaded["scheduler"])
            restore_rng(loaded["exact_resume"]["rank_states"][0]["rng"], rank=0, torch=torch)
            self.assertTrue(torch.equal(expected_loss, step()))
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, expected_model[key]), key)
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
