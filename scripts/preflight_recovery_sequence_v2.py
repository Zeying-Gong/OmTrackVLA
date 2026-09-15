"""Real admitted recovery sequence backward, with zero optimizer steps."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--disable-autocast-cache", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    import yaml
    from scripts.build_recovery_scene_manifest import verify_manifest
    from omtrackvla.data.recovery_sequence import (
        RecoverySequenceDataset, recovery_sequence_collate, validate_recovery_sample,
    )
    from omtrackvla.training.recovery_state import file_hash
    from omtrackvla.training.sequence_training import (
        SequenceTrainingPolicy, sequence_model_inputs, phase3_sequence_loss,
    )
    torch.set_num_threads(1)
    torch.manual_seed(20260914)
    config_path = args.config.resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("method") != "architecture_v1_end_to_end" or config.get("test_locked_used") is not False:
        raise ValueError("Architecture-v1 nonlocked config required")
    for field in ("da3_source", "da3_runtime"):
        sys.path.insert(0, str(Path(config[field]).resolve(strict=True)))
    from omtrackvla.models.end_to_end import (
        ArchitectureV1Ablation, ArchitectureV1Config, EndToEndFollowPolicy, load_official_da3_small_l11,
    )
    manifest_path = args.manifest.resolve(strict=True)
    manifest, paths = verify_manifest(manifest_path,
        validator=lambda p: validate_recovery_sample(p, artifact_root=ROOT),
        role="train", for_training=True)
    architecture = ArchitectureV1Config(**config.get("architecture", {}))
    ablation = ArchitectureV1Ablation(**config.get("ablation", {}))
    dataset = RecoverySequenceDataset(paths, artifact_root=ROOT,
        development_manifest=manifest_path, partition_role="train", required_anchor=9,
        image_height=architecture.image_height, image_width=architecture.image_width)
    batch = recovery_sequence_collate([dataset[0]])
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    backbone, loading = load_official_da3_small_l11(Path(config["da3_model"]), architecture, ablation)
    policy = EndToEndFollowPolicy(backbone, architecture, ablation).to(device)
    parent = manifest["parent_checkpoint"]
    checkpoint_path = Path(parent["path"])
    if parent["sha256"] != "32c8f2f73277cfab8c3c7c22a53994061f7178498cfa003e1635b9576641bc9c":
        raise ValueError("preflight requires the pre-recovery Phase 2 parent")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("phase") != 2 or checkpoint.get("test_locked_used") is not False:
        raise ValueError("pre-recovery Phase 2 parent required")
    policy.load_state_dict(checkpoint["model"], strict=True)
    recurrent_diagnostics = []
    def inspect_gru(module, values, output):
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            x, h = (value.float() for value in values)
            from torch.nn import functional as F
            ix = F.linear(x, module.weight_ih.float(), module.bias_ih.float()).chunk(3, 1)
            ih = F.linear(h, module.weight_hh.float(), module.bias_hh.float()).chunk(3, 1)
            z = torch.sigmoid(ix[1] + ih[1])
            recurrent_diagnostics.append({"call_index": len(recurrent_diagnostics),
                "grad_enabled_for_output": output.requires_grad, "dtype": str(output.dtype),
                "input_abs_max": float(x.abs().max()), "previous_hidden_abs_max": float(h.abs().max()),
                "output_abs_max": float(output.abs().max()), "hidden_delta_abs_max": float((output.float()-h).abs().max()),
                "update_gate_fp32_min": float(z.min()), "update_gate_fp32_max": float(z.max()),
                "update_gate_fp32_rounded_one_fraction": float((z == 1).float().mean())})
    hook = policy.gru.register_forward_hook(inspect_gru)
    model = SequenceTrainingPolicy(policy).train()
    inputs = sequence_model_inputs(batch, device)
    burn = 6
    labels = {key: value[:, burn:].to(device) for key, value in batch.items()
              if key in {"target_waypoints", "waypoint_mask", "supervision_mask",
                         "stop_label_valid", "binding_label_valid", "identity_label_valid", "ego_label_valid"}}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16",
                        cache_enabled=not args.disable_autocast_cache):
        outputs = model(**inputs, burn_in_steps=burn)
        loss, components = phase3_sequence_loss(outputs, labels, {"waypoint": 1.0})
    loss.backward()
    hook.remove()
    gradients = {name: float(parameter.grad.float().norm()) for name, parameter in policy.named_parameters()
                 if parameter.grad is not None and (name.startswith("gru.") or name.startswith("waypoint_head.") or name.startswith("fusion."))}
    recurrent = policy.gru.weight_hh.grad
    recurrent_norm = 0.0 if recurrent is None else float(recurrent.float().norm())
    norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0))
    supervised = labels["supervision_mask"]
    checks = {
        "complete_ten_decision_prefix": inputs["ego_rgb"].shape[1] == 10,
        "four_history_images_per_decision": inputs["ego_rgb"].shape[2] == 4,
        "four_learning_decisions_after_six_burn_in": outputs["waypoints"].shape[1] == 4,
        "only_anchor_is_supervised": int(supervised.sum()) == 1 and bool(supervised[0, -1]),
        "missing_auxiliary_labels_are_invalid": all(not bool(labels[key].any()) for key in
            ("stop_label_valid", "binding_label_valid", "identity_label_valid", "ego_label_valid")),
        "finite_waypoint_loss": bool(torch.isfinite(loss)),
        "recurrent_weight_receives_anchor_waypoint_gradient": recurrent_norm > 0,
        "finite_gradient_norm": bool(torch.isfinite(torch.tensor(norm))),
    }
    report = {"stage": "real_schema2_recovery_sequence_backward_preflight",
        "status": "passed" if all(checks.values()) else "failed", "checks": checks,
        "optimizer_steps": 0, "formal_training_eligible": False, "test_locked_used": False,
        "manifest_sha256": file_hash(manifest_path), "parent_checkpoint_sha256": file_hash(checkpoint_path),
        "sample_sha256": file_hash(paths[0]), "config_sha256": file_hash(config_path),
        "source_file_sha256": file_hash(Path(__file__)), "sample_path": str(paths[0]),
        "sequence_rgb_shape": list(inputs["ego_rgb"].shape), "waypoint_shape": list(outputs["waypoints"].shape),
        "waypoint_loss": float(loss.detach()), "gru_weight_hh_gradient_norm": recurrent_norm,
        "gradient_norm_before_clip": norm, "da3_loading": loading,
        "precision": args.precision, "layer_gradient_norms": gradients, "recurrent_diagnostics": recurrent_diagnostics,
        "autocast_cache_enabled": not args.disable_autocast_cache,
        "limitation": "Gradient flow and data admission only; no optimizer step or recovery success claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def json_safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return value
    from omtrackvla.training.recovery_state import atomic_json
    atomic_json(args.output, json_safe(report))
    print(json.dumps({"status": report["status"], "checks": checks, "output": str(args.output)}), flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
