"""Minimal offline inference adapter for the official KPR implementation.

The upstream research repository imports its complete training stack at package
import time.  This adapter deliberately loads only the KPR/SOLIDER model files
needed for frozen inference, so an offline evaluation machine does not need the
training-only albumentations, wandb, MONAI, or timm dependencies.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from typing import Optional, Sequence

import cv2
import numpy as np


class _CheckpointCfgNode(dict):
    """Small YACS-compatible type used only to reconstruct trusted config data."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value) -> None:
        if name.startswith("_CfgNode__"):
            object.__setattr__(self, name, value)
        else:
            self[name] = value


def _install_yacs_compat():
    try:
        from yacs.config import CfgNode

        return CfgNode
    except ImportError:
        yacs_module = types.ModuleType("yacs")
        config_module = types.ModuleType("yacs.config")
        _CheckpointCfgNode.__module__ = "yacs.config"
        _CheckpointCfgNode.__name__ = "CfgNode"
        _CheckpointCfgNode.__qualname__ = "CfgNode"
        config_module.CfgNode = _CheckpointCfgNode
        yacs_module.config = config_module
        sys.modules["yacs"] = yacs_module
        sys.modules["yacs.config"] = config_module
        return _CheckpointCfgNode


def _namespace(name: str, path: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module
    return module


def _source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load KPR module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_minimal_kpr_modules(source_root: Path):
    """Load upstream modules without executing ``torchreid/__init__.py``."""

    torchreid_root = source_root / "torchreid"
    required = (
        torchreid_root / "models/kpr.py",
        torchreid_root / "models/promptable_transformer_backbone.py",
        torchreid_root / "models/promptable_solider.py",
        torchreid_root / "models/solider/backbones/swin_transformer.py",
        torchreid_root / "utils/constants.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete KPR source tree; missing: {missing}")

    # The upstream files use absolute ``torchreid`` imports.  Register a small
    # package graph containing only their inference dependencies.
    for name in tuple(sys.modules):
        if name == "torchreid" or name.startswith("torchreid."):
            del sys.modules[name]
    torchreid_pkg = _namespace("torchreid", torchreid_root)
    models_pkg = _namespace("torchreid.models", torchreid_root / "models")
    utils_pkg = _namespace("torchreid.utils", torchreid_root / "utils")
    solider_pkg = _namespace(
        "torchreid.models.solider", torchreid_root / "models/solider"
    )
    backbones_pkg = _namespace(
        "torchreid.models.solider.backbones",
        torchreid_root / "models/solider/backbones",
    )
    torchreid_pkg.models = models_pkg
    torchreid_pkg.utils = utils_pkg
    models_pkg.solider = solider_pkg
    solider_pkg.backbones = backbones_pkg

    constants = _source_module(
        "torchreid.utils.constants", torchreid_root / "utils/constants.py"
    )
    utils_pkg.constants = constants

    # KPR imports the registry before calling it, so install a late-bound
    # dispatcher before executing the module.
    loaded = {}

    def build_model(
        name,
        num_classes=1,
        loss="softmax",
        pretrained=True,
        use_gpu=True,
        **kwargs,
    ):
        if name == "kpr":
            return loaded["kpr"].KPR(
                num_classes=num_classes,
                loss=loss,
                pretrained=pretrained,
                use_gpu=use_gpu,
                **kwargs,
            )
        if name == "solider_swin_base_patch4_window7_224":
            return loaded["promptable_solider"].solider_swin(
                num_classes=num_classes,
                loss=loss,
                pretrained=pretrained,
                use_gpu=use_gpu,
                **kwargs,
            )
        raise KeyError(f"minimal KPR adapter cannot build model {name!r}")

    models_pkg.build_model = build_model
    loaded["kpr"] = _source_module(
        "torchreid.models.kpr", torchreid_root / "models/kpr.py"
    )
    models_pkg.kpr = loaded["kpr"]
    loaded["promptable_transformer_backbone"] = _source_module(
        "torchreid.models.promptable_transformer_backbone",
        torchreid_root / "models/promptable_transformer_backbone.py",
    )
    loaded["swin_transformer"] = _source_module(
        "torchreid.models.solider.backbones.swin_transformer",
        torchreid_root / "models/solider/backbones/swin_transformer.py",
    )
    backbones_pkg.swin_transformer = loaded["swin_transformer"]
    loaded["promptable_solider"] = _source_module(
        "torchreid.models.promptable_solider",
        torchreid_root / "models/promptable_solider.py",
    )
    models_pkg.promptable_solider = loaded["promptable_solider"]
    return loaded, constants


def _safe_load_checkpoint(path: Path):
    import torch

    cfg_node = _install_yacs_compat()
    dtype_names = (
        "bool int8 int16 int32 int64 uint8 uint16 uint32 uint64 "
        "float16 float32 float64 complex64 complex128"
    ).split()
    safe_types = {
        np.dtype,
        np.core.multiarray.scalar,
        cfg_node,
        *(type(np.dtype(name)) for name in dtype_names),
    }
    with torch.serialization.safe_globals(list(safe_types)):
        return torch.load(path, map_location="cpu", weights_only=True)


class KPRReIDBackend:
    """Frozen KPR/SOLIDER feature extractor with visibility-aware matching."""

    name = "kpr-occ-duke-solider"

    def __init__(
        self,
        source_path: Path | str,
        weights_path: Path | str,
        device="cuda",
    ) -> None:
        import torch

        self.source_path = Path(source_path).resolve()
        self.weights_path = Path(weights_path).resolve()
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"KPR weights not found: {self.weights_path}")
        modules, self.constants = _load_minimal_kpr_modules(self.source_path)
        checkpoint = _safe_load_checkpoint(self.weights_path)
        if "state_dict" not in checkpoint or "config" not in checkpoint:
            raise ValueError("KPR checkpoint must contain state_dict and config")
        self.config = checkpoint["config"]
        if self.config.model.kpr.backbone != "solider_swin_base_patch4_window7_224":
            raise ValueError(
                f"unsupported KPR backbone: {self.config.model.kpr.backbone}"
            )
        self.device = torch.device(
            device if str(device) != "cuda" or torch.cuda.is_available() else "cpu"
        )
        model = modules["kpr"].KPR(
            num_classes=1,
            pretrained=False,
            loss=self.config.loss.name,
            config=self.config,
        )
        model_state = model.state_dict()
        matched = {}
        discarded = []
        for key, value in checkpoint["state_dict"].items():
            clean_key = key[7:] if key.startswith("module.") else key
            if clean_key in model_state and model_state[clean_key].shape == value.shape:
                matched[clean_key] = value
            else:
                discarded.append(clean_key)
        if not matched:
            raise ValueError("no KPR checkpoint tensors matched the constructed model")
        missing, unexpected = model.load_state_dict(matched, strict=False)
        non_classifier_missing = [
            key for key in missing if "identity_classifier" not in key
        ]
        if non_classifier_missing or unexpected:
            raise ValueError(
                "incomplete KPR load: "
                f"missing={non_classifier_missing[:10]}, unexpected={unexpected[:10]}"
            )
        self.discarded_checkpoint_keys = tuple(discarded)
        self.model = model.eval().to(self.device)
        self.height = int(self.config.data.height)
        self.width = int(self.config.data.width)
        self.mean = tuple(float(x) for x in self.config.data.norm_mean)
        self.std = tuple(float(x) for x in self.config.data.norm_std)
        self.test_embeddings = tuple(self.config.model.kpr.test_embeddings)
        self.parts = 1 + int(self.config.model.kpr.masks.parts_num)
        self.dimension = int(self.config.model.kpr.dim_reduce_output)

    def _extract_test_embeddings(self, model_output):
        import torch

        embeddings, visibility, _, _, _, _ = model_output
        embedding_items = []
        visibility_items = []
        for name in self.test_embeddings:
            value = embeddings[name]
            embedding_items.append(value if value.ndim == 3 else value.unsqueeze(1))
            visibility_name = self.constants.bn_correspondants.get(name, name)
            score = visibility[visibility_name]
            visibility_items.append(score if score.ndim == 2 else score.unsqueeze(1))
        features = torch.cat(embedding_items, dim=1)
        visible = torch.cat(visibility_items, dim=1)
        return torch.nn.functional.normalize(features, p=2, dim=-1), visible

    def embed_crops(self, crops: Sequence[np.ndarray]):
        if not crops:
            return []
        import torch

        tensors = []
        for crop in crops:
            image = np.asarray(crop)[..., :3]
            resized = cv2.resize(
                image, (self.width, self.height), interpolation=cv2.INTER_LINEAR
            )
            tensor = torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1)
            tensors.append(tensor.float().div_(255.0))
        inputs = torch.stack(tensors).to(self.device)
        mean = torch.tensor(self.mean, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=self.device).view(1, 3, 1, 1)
        inputs = (inputs - mean) / std
        with torch.inference_mode():
            features, visible = self._extract_test_embeddings(self.model(inputs))
        features = features.float().cpu().numpy()
        visible = visible.float().cpu().numpy()
        return [
            np.concatenate((item, item_visible[:, None]), axis=1).astype(np.float32)
            for item, item_visible in zip(features, visible)
        ]

    def normalize(self, embedding: Optional[np.ndarray]):
        if embedding is None:
            return None
        value = np.asarray(embedding, dtype=np.float32)
        if value.shape != (self.parts, self.dimension + 1):
            return None
        result = value.copy()
        norms = np.linalg.norm(result[:, :-1], axis=1, keepdims=True)
        if not np.all(np.isfinite(norms)) or np.any(norms <= 1e-6):
            return None
        result[:, :-1] /= norms
        result[:, -1] = result[:, -1] > 0.5
        return result

    def similarity01(
        self, left: Optional[np.ndarray], right: Optional[np.ndarray]
    ) -> float:
        left = self.normalize(left)
        right = self.normalize(right)
        if left is None or right is None:
            return 0.0
        common = (left[:, -1] > 0.5) & (right[:, -1] > 0.5)
        if not np.any(common):
            return 0.0
        # This is the upstream KPR metric: mean Euclidean distance across
        # mutually visible, independently L2-normalized body-part embeddings.
        distance = float(
            np.linalg.norm(left[common, :-1] - right[common, :-1], axis=1).mean()
        )
        return float(np.clip(1.0 - 0.5 * distance, 0.0, 1.0))

    def blend(
        self,
        previous: Optional[np.ndarray],
        current: Optional[np.ndarray],
        current_weight: float,
    ):
        previous = self.normalize(previous)
        current = self.normalize(current)
        if previous is None:
            return current
        if current is None:
            return previous
        weight = float(np.clip(current_weight, 0.0, 1.0))
        result = previous.copy()
        current_visible = current[:, -1] > 0.5
        both_visible = current_visible & (previous[:, -1] > 0.5)
        result[current_visible] = current[current_visible]
        result[both_visible, :-1] = (
            (1.0 - weight) * previous[both_visible, :-1]
            + weight * current[both_visible, :-1]
        )
        norms = np.linalg.norm(result[:, :-1], axis=1, keepdims=True)
        result[:, :-1] /= np.maximum(norms, 1e-12)
        result[:, -1] = (previous[:, -1] > 0.5) | current_visible
        return result.astype(np.float32)
