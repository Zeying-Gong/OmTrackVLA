#!/usr/bin/env python3
"""Evaluate one Architecture-v2 Phase-3 checkpoint against frozen pilot gates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from train_v2_phase3_model_visited import (
    clean_metrics,
    load_recovery_samples,
    recovery_metrics,
    resolve,
    sha256,
    verify_manifest,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    config_path = args.config.expanduser().resolve(strict=True)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    baseline_path = args.baseline.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().resolve(strict=False)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite candidate evaluation: {output_path}")

    import torch
    import yaml
    from torch.utils.data._utils.collate import default_collate

    from omtrackvla.data.end_to_end_training import Sage3DEndToEndSequenceDataset
    from omtrackvla.models.end_to_end import (
        ArchitectureV1Ablation,
        ArchitectureV1Config,
        load_official_da3_small_l11,
    )
    from omtrackvla.models.end_to_end_v2 import (
        ArchitectureV2DecoderConfig,
        ArchitectureV2FollowPolicy,
    )
    from omtrackvla.training.end_to_end_v2 import _step_inputs, _targets

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        not isinstance(config, Mapping)
        or config.get("method") != "architecture_v2_evt_perception_polar"
        or config.get("test_locked_used") is not False
    ):
        raise ValueError("candidate config provenance mismatch")
    repository = config_path.parents[2]
    for field in ("da3_source", "da3_runtime"):
        extra = resolve(repository, config[field]).resolve(strict=True)
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("candidate evaluation requires CUDA")
    torch.cuda.set_device(device)
    architecture = ArchitectureV1Config(**config["architecture"])
    decoder = ArchitectureV2DecoderConfig(**config["decoder"])
    manifest = verify_manifest(
        resolve(repository, config["recovery_manifest"]).resolve(strict=True)
    )
    recovery_samples, _ = load_recovery_samples(manifest, architecture, torch)
    clean_modes = tuple(config["data"]["clean_condition_modes"])
    clean_val = Sage3DEndToEndSequenceDataset(
        resolve(repository, config["data"]["sequence_index"]),
        split="viz_val",
        config=architecture,
        modes=clean_modes,
        history_stride_raw=decoder.history_stride_raw,
    )
    da3, _ = load_official_da3_small_l11(
        resolve(repository, config["da3_model"]),
        architecture,
        ArchitectureV1Ablation(backbone_tuning="adapter"),
    )
    policy = ArchitectureV2FollowPolicy(da3, architecture, decoder).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("method") != config["method"]
        or checkpoint.get("test_locked_used") is not False
    ):
        raise ValueError("candidate checkpoint provenance mismatch")
    policy.load_state_dict(checkpoint["model"], strict=True)
    recovery = recovery_metrics(policy, recovery_samples, device, torch)
    clean = clean_metrics(
        policy,
        clean_val,
        int(config["data"]["clean_validation_samples"]),
        device,
        torch,
        default_collate,
        _step_inputs,
        _targets,
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    gate = config["selection_gate"]
    recovery_max = baseline["recovery"]["ade_m"] * (
        1.0 - float(gate["minimum_recovery_ade_relative_improvement"])
    )
    clean_max = baseline["clean_sage3d"]["ade_m"] * (
        1.0 + float(gate["maximum_clean_ade_relative_regression"])
    ) + float(gate["maximum_clean_ade_absolute_slack_m"])
    selected = recovery["ade_m"] <= recovery_max and clean["ade_m"] <= clean_max
    payload = {
        "schema_version": 1,
        "stage": "v2_phase3_candidate_evaluation",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "global_step": checkpoint.get("global_step"),
        "metrics": {"recovery": recovery, "clean_sage3d": clean},
        "selection_gate": {
            "recovery_ade_required_max_m": recovery_max,
            "clean_ade_allowed_max_m": clean_max,
        },
        "selected": selected,
        "test_locked_used": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
