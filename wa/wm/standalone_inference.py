"""Load a complete WA checkpoint without reading WLA/JEPA initialization weights.

This is an independent inference constructor, not a replacement training loader.
All JointRobotModel modules are retained, including the JEPA training-only world
and robot adapters. The separately pinned frozen DINO encoder remains required.
A successful load proves state admission, NOT output equivalence or closed-loop SR.
"""
from collections.abc import Mapping
import os
from pathlib import Path
import re
import sys

import torch
from torch import nn

from wa.wm.adapters import GoalMetaQueryAdapter, WorldActionPolicy
from wa.wm.loaders import HASHES, OfficialWorld, load_encoder, sha, verify_source
from wa.wm.robot_data import CONTRACT
from wa.wm.training import JointRobotModel, RobotConditionAdapter

SCHEMA = "wa.standalone_inference.v1"


def _identity(checkpoint_sha256, checkpoint_step):
    if not isinstance(checkpoint_sha256, str) or re.fullmatch(
        "[0-9a-f]{64}", checkpoint_sha256
    ) is None:
        raise ValueError("explicit checkpoint SHA256 required")
    if type(checkpoint_step) is not int or checkpoint_step < 1:
        raise ValueError("explicit positive integer checkpoint step required")


def _wla_manifest(source):
    source = Path(source).resolve()
    files = sorted((source / "src/md_wla").rglob("*.py"))
    if not files:
        raise ValueError("missing fixed WLA architecture source")
    return {str(p.relative_to(source)): sha(p) for p in files}


def _check_imports(prefix, source):
    """Reject a cached package silently taking precedence over the requested source."""
    source = Path(source).resolve()
    for name, module in tuple(sys.modules.items()):
        if name == prefix or name.startswith(prefix + "."):
            origin = getattr(module, "__file__", None)
            if origin is not None and not Path(origin).resolve().is_relative_to(source):
                raise ValueError("foreign cached architecture module: " + name)


def _wla_structure(source):
    source = Path(source).resolve()
    _check_imports("md_wla", source / "src")
    sys.path.insert(0, str(source / "src"))
    from md_wla.models.action.expert import ActionExpertConfig, LayerwiseActionExpert
    from md_wla.models.queries import MetaQueryTokens
    from md_wla.deploy.rx_model import EXACT_MODEL_ENVIRONMENT

    _check_imports("md_wla", source / "src")
    # Keep the exact original architecture flags and random-construction order.
    previous = {
        k: v for k, v in os.environ.items()
        if k.startswith("MD_WLA_DIAGNOSTIC_") or k in EXACT_MODEL_ENVIRONMENT
    }
    try:
        for key in previous:
            os.environ.pop(key, None)
        os.environ.update(EXACT_MODEL_ENVIRONMENT)
        expert = LayerwiseActionExpert(ActionExpertConfig(
            action_dim=4, horizon=7, backbone_dim=2560, model_dim=1024,
            num_blocks=16, num_heads=32, max_state_dim=0, state_history_frames=4,
            tap_indices=tuple(range(12, 28)),
        ))
    finally:
        for key in EXACT_MODEL_ENVIRONMENT:
            os.environ.pop(key, None)
        os.environ.update(previous)
    query = MetaQueryTokens(hidden_size=2560)
    target = nn.Sequential(
        nn.LayerNorm(2560), nn.Linear(2560, 512), nn.SiLU(), nn.Linear(512, 3)
    )
    return query, expert, target


class _JEPAWithoutInitialization(OfficialWorld):
    """Original JEPA shape and forward, without accessing mz_jepa-wm.pth.tar."""

    def __init__(self, root):
        nn.Module.__init__(self)
        root = Path(root)
        upstream = root / "upstream_audit/jepa-wms"
        verify_source(upstream, "jepa-wms")
        _check_imports("app", upstream)
        sys.path.insert(0, str(upstream))
        from app.plan_common.models.AdaLN_vit import vit_predictor_AdaLN
        from app.plan_common.models.vit import ViTPredictor  # same original imports
        from app.plan_common.models.prop_embedding import ProprioceptiveEmbedding

        _check_imports("app", upstream)
        self.kind = "jepa"
        self.prop = ProprioceptiveEmbedding(
            num_frames=4, tubelet_size=1, in_chans=4, embed_dim=16, shift_input=False
        )
        self.predictor = vit_predictor_AdaLN(
            img_size=224, patch_size=14, num_frames=4, tubelet_size=1,
            embed_dim=384, predictor_embed_dim=384, depth=6, num_heads=16,
            use_rope=True, local_window=(3, -1, -1), action_dim=10,
            proprio_dim=4, proprio_emb_dim=16, proprio_encoder_inpred=False,
        )


class _StandaloneStructure(JointRobotModel):
    """Internal architecture-only construction; use load_standalone for admission."""

    def __init__(self, root, encoder_weight, wla_source):
        nn.Module.__init__(self)
        self.encoder = load_encoder(root, encoder_weight)
        query, expert, target = _wla_structure(wla_source)
        self.policy = WorldActionPolicy(
            GoalMetaQueryAdapter(query), expert, target, _JEPAWithoutInitialization(root)
        )
        self.robot_action = RobotConditionAdapter(10)
        self.robot_state = RobotConditionAdapter(4)
        self.kind = "jepa"


def _validate_state(model, state):
    expected = {k: v for k, v in model.state_dict().items()
                if not k.startswith("encoder.")}
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise ValueError("checkpoint must cover exactly every nonencoder state")
    if not expected:
        raise ValueError("empty model state")
    for key, target in expected.items():
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError("non-tensor model state: " + key)
        if value.layout != torch.strided or value.device.type != "cpu":
            raise ValueError("model state must be dense CPU tensor: " + key)
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError("checkpoint shape/dtype mismatch: " + key)
        if not torch.isfinite(value).all().item():
            raise ValueError("nonfinite checkpoint state: " + key)
    return expected


def load_standalone(root, encoder_weight, wla_source, checkpoint,
                    checkpoint_sha256, checkpoint_step):
    """Return a complete, frozen CPU model admitted from the original WA file.

    Only checkpoint and encoder_weight are weight-file inputs. WLA and JEPA
    source are still required; no initialization-weight fallback is permitted.
    The optimizer (if present in the original checkpoint) is not used.
    """
    _identity(checkpoint_sha256, checkpoint_step)
    checkpoint = Path(checkpoint)
    encoder_weight = Path(encoder_weight)
    if sha(checkpoint) != checkpoint_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    doc = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(doc, Mapping):
        raise ValueError("checkpoint must be a mapping")
    if (type(doc.get("step")) is not int or doc["step"] != checkpoint_step
            or doc.get("kind") != "jepa" or doc.get("contract") != CONTRACT):
        raise ValueError("checkpoint step/kind/contract mismatch")
    source_files = _wla_manifest(wla_source)
    model = _StandaloneStructure(root, encoder_weight, wla_source)
    expected = _validate_state(model, doc.get("model"))
    incompatible = model.load_state_dict(doc["model"], strict=False)
    encoder_keys = {k for k in model.state_dict() if k.startswith("encoder.")}
    if (incompatible.unexpected_keys
            or set(incompatible.missing_keys) != encoder_keys):
        raise ValueError("partial standalone state load")
    model.eval().requires_grad_(False)
    # Keep final artifact checks after tensor copy; no new file or exported weights.
    if sha(checkpoint) != checkpoint_sha256:
        raise ValueError("checkpoint changed during standalone load")
    if sha(encoder_weight) != HASHES["encoder"]:
        raise ValueError("encoder changed during standalone load")
    if _wla_manifest(wla_source) != source_files:
        raise ValueError("WLA source changed during standalone load")
    for name in ("dinov2", "jepa-wms"):
        verify_source(Path(root) / "upstream_audit" / name, name)
    model.provenance = {
        "schema": SCHEMA,
        "status": "DIRECT_CHECKPOINT_LOADED_NOT_OUTPUT_EQUIVALENCE",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": checkpoint_step,
        "kind": "jepa",
        "contract": CONTRACT,
        "encoder_sha256": HASHES["encoder"],
        "source_files": source_files,
        "state_tensors_loaded": len(expected),
        "state_bytes_loaded": sum(t.numel() * t.element_size()
                                  for t in expected.values()),
        "initialization_weights_read": [],
        "wla_initialization_loaded": False,
        "jepa_initialization_loaded": False,
        "world_modules_retained": True,
        "optimizer_used": False,
        "text_used": False,
        "world_predictor_inference": False,
        "output_equivalence_verified": False,
        "closed_loop_verified": False,
    }
    return model
