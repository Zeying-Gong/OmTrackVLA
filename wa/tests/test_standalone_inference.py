"""CPU admission tests only: small mocked modules, no production weights or GPU."""
from contextlib import ExitStack
import copy
from pathlib import Path
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from wa.wm import standalone_inference as direct
from wa.wm import training


class TinyWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.prop = nn.Linear(2, 2)
        self.predictor = nn.Linear(2, 2)
        self.register_buffer("mask", torch.ones(2), persistent=False)


class TinyAdapter(nn.Module):
    def __init__(self, query):
        super().__init__()
        self.metaquery = query
        self.fusion = nn.Linear(2, 2)


class TinyPolicy(nn.Module):
    def __init__(self, adapter, expert, target, world):
        super().__init__()
        self.adapter = adapter
        self.action_expert = expert
        self.target_head = target
        self.world = world
        self.world_bridge = nn.Linear(2, 2)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        with torch.no_grad():
            self.encoder.weight.fill_(7)
            self.encoder.bias.fill_(8)
        self.policy = TinyPolicy(TinyAdapter(nn.Linear(2, 2)), nn.Linear(2, 2),
                                 nn.Linear(2, 2), TinyWorld())
        self.robot_action = nn.Linear(2, 2)
        self.robot_state = nn.Linear(2, 2)
        self.kind = "jepa"


def fake_modules(names, root):
    result = {}
    for name in names:
        module = types.ModuleType(name)
        module.__file__ = str(Path(root) / (name.replace(".", "/") + ".py"))
        module.__path__ = []
        result[name] = module
    return result


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.wla = self.root / "wla"
        src = self.wla / "src/md_wla"
        src.mkdir(parents=True)
        (src / "fixture.py").write_text("# fixture architecture\n")
        self.encoder = self.root / "encoder.pt"
        self.encoder.write_bytes(b"tiny fixed encoder fixture")
        self.checkpoint = self.root / "complete.pt"
        model = TinyModel()
        self.doc = {
            "step": 59716, "kind": "jepa", "contract": direct.CONTRACT,
            "model": {k: v.clone() for k, v in model.state_dict().items()
                      if not k.startswith("encoder.")},
            "optimizer": {"not_used": True},
        }
        self.encoder_hash = direct.sha(self.encoder)
        self.save()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.factory = self.stack.enter_context(
            patch.object(direct, "_StandaloneStructure", side_effect=lambda *a: TinyModel()))
        self.verify = self.stack.enter_context(patch.object(direct, "verify_source"))
        self.stack.enter_context(patch.dict(direct.HASHES, encoder=self.encoder_hash))

    def save(self):
        torch.save(self.doc, self.checkpoint)
        self.digest = direct.sha(self.checkpoint)

    def load(self, **kw):
        values = dict(root=self.root, encoder_weight=self.encoder, wla_source=self.wla,
                      checkpoint=self.checkpoint, checkpoint_sha256=self.digest,
                      checkpoint_step=59716)
        values.update(kw)
        return direct.load_standalone(**values)

    def test_complete_admission_keeps_all_modules_frozen_and_encoder_external(self):
        original_load = torch.load
        with patch.object(torch, "load", wraps=original_load) as read:
            model = self.load()
        self.assertEqual(read.call_count, 1)
        self.assertEqual(read.call_args.args, (self.checkpoint,))
        self.assertEqual(read.call_args.kwargs,
                         dict(map_location="cpu", weights_only=True, mmap=True))
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        for key, value in self.doc["model"].items():
            self.assertTrue(torch.equal(model.state_dict()[key], value), key)
        self.assertTrue(torch.equal(model.encoder.weight, torch.full((2, 2), 7.)))
        self.assertTrue(torch.equal(model.encoder.bias, torch.full((2,), 8.)))
        self.assertIsInstance(model.policy.world, TinyWorld)
        self.assertIn("policy.world.mask", dict(model.named_buffers()))
        p = model.provenance
        self.assertEqual(p["checkpoint_step"], 59716)
        self.assertEqual(p["checkpoint_sha256"], self.digest)
        self.assertEqual(p["state_tensors_loaded"], len(self.doc["model"]))
        self.assertEqual(p["state_bytes_loaded"],
                         sum(x.numel() * x.element_size() for x in self.doc["model"].values()))
        self.assertEqual(p["initialization_weights_read"], [])
        for field in ("wla_initialization_loaded", "jepa_initialization_loaded",
                      "optimizer_used", "text_used", "world_predictor_inference",
                      "output_equivalence_verified", "closed_loop_verified"):
            self.assertIs(p[field], False)
        self.assertEqual(p["status"], "DIRECT_CHECKPOINT_LOADED_NOT_OUTPUT_EQUIVALENCE")
        self.assertEqual(self.verify.call_count, 2)

    def test_expected_identity_strict_and_checked_before_load_or_construction(self):
        for kwargs in (dict(checkpoint_sha256="a" * 63), dict(checkpoint_sha256="A" * 64),
                       dict(checkpoint_step=True), dict(checkpoint_step=0),
                       dict(checkpoint_step=59716.), dict(checkpoint_sha256="0" * 64)):
            with self.subTest(kwargs=kwargs), patch.object(torch, "load") as read:
                with self.assertRaises(ValueError):
                    self.load(**kwargs)
                read.assert_not_called()
        self.factory.assert_not_called()

    def test_recorded_identity_rejected_before_construction(self):
        original = copy.deepcopy(self.doc)
        for key, value in (("step", 59715), ("step", True), ("step", 59716.),
                           ("kind", "dino"), ("contract", "different")):
            self.doc = copy.deepcopy(original)
            self.doc[key] = value
            self.save()
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "step/kind/contract"):
                self.load()
        self.factory.assert_not_called()

    def test_nonmapping_document_rejected(self):
        torch.save([], self.checkpoint)
        self.digest = direct.sha(self.checkpoint)
        with self.assertRaisesRegex(ValueError, "mapping"):
            self.load()
        self.factory.assert_not_called()

    def test_every_nonencoder_tensor_is_required_including_world_and_robot(self):
        original = copy.deepcopy(self.doc["model"])
        for key in original:
            self.doc["model"] = copy.deepcopy(original)
            del self.doc["model"][key]
            self.save()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "exactly"):
                self.load()

    def test_extra_encoder_or_unknown_state_cannot_be_silently_ignored(self):
        original = copy.deepcopy(self.doc["model"])
        for key in ("encoder.weight", "policy.foreign.weight"):
            self.doc["model"] = copy.deepcopy(original)
            self.doc["model"][key] = torch.ones(2, 2)
            self.save()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "exactly"):
                self.load()

    def test_tensor_shape_dtype_type_and_finite_are_strict(self):
        original = copy.deepcopy(self.doc["model"])
        key = "policy.world.predictor.weight"
        for value in (torch.ones(3, 2), torch.ones(2, 2, dtype=torch.float64),
                      [[1, 2], [3, 4]], torch.full((2, 2), float("nan")),
                      torch.full((2, 2), float("inf")), torch.ones(2, 2).to_sparse()):
            self.doc["model"] = copy.deepcopy(original)
            self.doc["model"][key] = value
            self.save()
            with self.subTest(value=type(value)), self.assertRaises(ValueError):
                self.load()

    def test_load_result_missing_or_unexpected_nonencoder_is_rejected(self):
        for missing, unexpected in ((["encoder.weight"], []),
                                   (["encoder.weight", "encoder.bias"], ["foreign"])):
            model = TinyModel()
            with patch.object(model, "load_state_dict",
                              return_value=types.SimpleNamespace(
                                  missing_keys=missing, unexpected_keys=unexpected)):
                self.factory.side_effect = lambda *a: model
                with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "partial"):
                    self.load()

    def test_checkpoint_mutation_during_copy_rejected(self):
        original_sha = direct.sha
        hits = [0]
        def changed(path):
            if Path(path) == self.checkpoint:
                hits[0] += 1
                if hits[0] == 2:
                    return "0" * 64
            return original_sha(path)
        with patch.object(direct, "sha", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                self.load()

    def test_encoder_source_or_upstream_mutation_rejected(self):
        self.encoder.write_bytes(b"changed encoder")
        with self.assertRaisesRegex(ValueError, "encoder changed"):
            self.load()
        self.encoder.write_bytes(b"tiny fixed encoder fixture")
        actual_manifest = direct._wla_manifest(self.wla)
        with patch.object(direct, "_wla_manifest", side_effect=[actual_manifest, {}]):
            with self.assertRaisesRegex(ValueError, "WLA source changed"):
                self.load()
        self.verify.side_effect = ValueError("modified upstream source")
        with self.assertRaisesRegex(ValueError, "modified upstream"):
            self.load()

    def test_missing_architecture_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing fixed"):
            self.load(wla_source=self.root / "absent")
        self.factory.assert_not_called()


class ConstructorTests(unittest.TestCase):
    def test_constructor_order_and_rng_match_original_with_small_modules(self):
        events = []
        def linear(label):
            events.append(label)
            return nn.Linear(2, 2)
        def encoder(*args):
            return linear("encoder")
        def heads(*args):
            expert, query, target = linear("expert"), linear("query"), linear("target")
            return query, expert, target
        def adapter(query):
            events.append("adapter")
            return TinyAdapter(query)
        def world(*args):
            events.append("world")
            return TinyWorld()
        def policy(*args):
            events.append("policy")
            return TinyPolicy(*args)
        def robot(dim):
            return linear("robot" + str(dim))
        with ExitStack() as stack:
            for module in (direct, training):
                for name, value in (("load_encoder", encoder),
                                    ("GoalMetaQueryAdapter", adapter),
                                    ("WorldActionPolicy", policy),
                                    ("RobotConditionAdapter", robot)):
                    stack.enter_context(patch.object(module, name, value))
            stack.enter_context(patch.object(direct, "_wla_structure", heads))
            stack.enter_context(patch.object(direct, "_JEPAWithoutInitialization", world))
            stack.enter_context(patch.object(training, "load_wla_heads",
                                            lambda *a: (*heads(), {"old": True})))
            stack.enter_context(patch.object(training, "OfficialWorld", world))
            torch.manual_seed(7)
            reference = training.JointRobotModel("root", "encoder", "src", "init", "jepa")
            expected_rng = torch.get_rng_state().clone()
            expected_events = list(events)
            events.clear()
            torch.manual_seed(7)
            candidate = direct._StandaloneStructure("root", "encoder", "src")
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        self.assertEqual(events, expected_events)
        self.assertEqual(events, ["encoder", "expert", "query", "target", "adapter",
                                  "world", "policy", "robot10", "robot4"])
        self.assertEqual(set(candidate.state_dict()), set(reference.state_dict()))
        for key in candidate.state_dict():
            self.assertTrue(torch.equal(candidate.state_dict()[key], reference.state_dict()[key]))
        self.assertIs(direct._StandaloneStructure.predict, training.JointRobotModel.predict)
        self.assertIs(direct._StandaloneStructure.make_conditions, training.JointRobotModel.make_conditions)
        self.assertIs(direct._JEPAWithoutInitialization.forward, direct.OfficialWorld.forward)

    def test_wla_shape_flags_and_environment_restored_without_weight_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            names = ["md_wla", "md_wla.models", "md_wla.models.action",
                     "md_wla.models.action.expert", "md_wla.models.queries",
                     "md_wla.deploy", "md_wla.deploy.rx_model"]
            modules = fake_modules(names, source / "src")
            seen = {}
            exact = {"MD_WLA_DIAGNOSTIC_TINY": "exact",
                     "MD_WLA_ACTION_ACTIVATION_CHECKPOINTING": "0"}
            modules[names[-1]].EXACT_MODEL_ENVIRONMENT = exact
            modules["md_wla.models.action.expert"].ActionExpertConfig = lambda **kw: kw
            def expert(config):
                seen["config"] = config
                seen["env"] = {k: os.environ.get(k) for k in exact}
                self.assertNotIn("MD_WLA_DIAGNOSTIC_FOREIGN", os.environ)
                return nn.Linear(2, 2)
            modules["md_wla.models.action.expert"].LayerwiseActionExpert = expert
            def query(**kw):
                seen["query"] = kw
                return nn.Linear(2, 2)
            modules["md_wla.models.queries"].MetaQueryTokens = query
            with patch.dict(sys.modules, modules), patch.object(sys, "path", list(sys.path)), \
                 patch.dict(os.environ, {"MD_WLA_DIAGNOSTIC_TINY": "before",
                                        "MD_WLA_DIAGNOSTIC_FOREIGN": "keep"}), \
                 patch.object(torch, "load", side_effect=AssertionError("old weight access")):
                before = dict(os.environ)
                q, e, target = direct._wla_structure(source)
                self.assertEqual(dict(os.environ), before)
                modules["md_wla.models.action.expert"].LayerwiseActionExpert = lambda c: (_ for _ in ()).throw(RuntimeError("construct"))
                with self.assertRaisesRegex(RuntimeError, "construct"):
                    direct._wla_structure(source)
                self.assertEqual(dict(os.environ), before)
            self.assertEqual(seen["env"], exact)
            self.assertEqual(seen["config"], dict(
                action_dim=4, horizon=7, backbone_dim=2560, model_dim=1024,
                num_blocks=16, num_heads=32, max_state_dim=0, state_history_frames=4,
                tap_indices=tuple(range(12, 28))))
            self.assertEqual(seen["query"], {"hidden_size": 2560})
            self.assertEqual(target[1].weight.shape, (512, 2560))
            self.assertEqual(target[3].weight.shape, (3, 512))

    def test_jepa_structure_and_plain_mask_retained_without_initialization_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "upstream_audit/jepa-wms"
            names = ["app", "app.plan_common", "app.plan_common.models",
                     "app.plan_common.models.AdaLN_vit", "app.plan_common.models.vit",
                     "app.plan_common.models.prop_embedding"]
            modules = fake_modules(names, source)
            seen = {}
            def make(label, **kw):
                seen[label] = kw
                value = nn.Linear(2, 2)
                if label == "predictor":
                    value.attn_mask = torch.zeros(4, 4)
                return value
            modules[names[-1]].ProprioceptiveEmbedding = lambda **kw: make("prop", **kw)
            modules[names[-3]].vit_predictor_AdaLN = lambda **kw: make("predictor", **kw)
            modules[names[-2]].ViTPredictor = object
            with patch.dict(sys.modules, modules), patch.object(sys, "path", list(sys.path)), \
                 patch.object(direct, "verify_source") as verify, \
                 patch.object(direct.OfficialWorld, "__init__", side_effect=AssertionError("old constructor")), \
                 patch.object(torch, "load", side_effect=AssertionError("old weight access")):
                model = direct._JEPAWithoutInitialization(root)
            verify.assert_called_once_with(source, "jepa-wms")
            self.assertEqual(model.kind, "jepa")
            self.assertEqual(set(model.state_dict()), {"prop.weight", "prop.bias",
                                                     "predictor.weight", "predictor.bias"})
            self.assertEqual(model.predictor.attn_mask.shape, (4, 4))
            self.assertEqual(seen["prop"], dict(num_frames=4, tubelet_size=1,
                                              in_chans=4, embed_dim=16, shift_input=False))
            self.assertEqual(seen["predictor"], dict(
                img_size=224, patch_size=14, num_frames=4, tubelet_size=1,
                embed_dim=384, predictor_embed_dim=384, depth=6, num_heads=16,
                use_rope=True, local_window=(3, -1, -1), action_dim=10,
                proprio_dim=4, proprio_emb_dim=16, proprio_encoder_inpred=False))

    def test_foreign_cached_architecture_is_rejected(self):
        for prefix in ("md_wla", "app"):
            module = types.ModuleType(prefix)
            module.__file__ = "/foreign/" + prefix + ".py"
            with patch.dict(sys.modules, {prefix: module}):
                with self.subTest(prefix=prefix), self.assertRaisesRegex(ValueError, "foreign cached"):
                    direct._check_imports(prefix, "/correct")


if __name__ == "__main__":
    unittest.main()
