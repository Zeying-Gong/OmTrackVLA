#!/usr/bin/env python3
"""Render parent versus selected Phase-3 predictions on representative failures."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from train_v2_phase3_model_visited import (
    load_recovery_samples,
    recovery_batch,
    recovery_inputs,
    resolve,
    sha256,
    verify_manifest,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def render(
    rgb_tensor: object,
    expert: list[list[float]],
    parent: list[list[float]],
    selected: list[list[float]],
    title: str,
    output: Path,
) -> None:
    array = (
        np.asarray(rgb_tensor.permute(1, 2, 0).float().cpu())
        .clip(0.0, 1.0)
        .__mul__(255.0)
        .astype(np.uint8)
    )
    rgb = Image.fromarray(array, mode="RGB").resize((504, 280))
    canvas = Image.new("RGB", (1008, 280), "white")
    canvas.paste(rgb, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 504, 22), fill=(0, 0, 0))
    draw.text((8, 5), title, fill=(255, 255, 255))
    origin = (756, 255)
    scale = 150.0
    draw.line((origin[0], 28, origin[0], origin[1]), fill=(190, 190, 190), width=1)
    draw.line((520, origin[1], 994, origin[1]), fill=(190, 190, 190), width=1)

    def points(values: list[list[float]]) -> list[tuple[int, int]]:
        return [
            (
                int(round(origin[0] - float(left) * scale)),
                int(round(origin[1] - float(forward) * scale)),
            )
            for forward, left in values
        ]

    for values, color, width in (
        (expert, (230, 130, 30), 5),
        (parent, (40, 170, 70), 3),
        (selected, (40, 90, 230), 3),
    ):
        path = points(values)
        draw.line(path, fill=color, width=width, joint="curve")
        for point in path:
            draw.ellipse(
                (point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2),
                fill=color,
            )
    draw.text((520, 6), "orange expert | green parent | blue step64", fill=(20, 20, 20))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> int:
    args = arguments()
    config_path = args.config.expanduser().resolve(strict=True)
    parent_path = args.parent_checkpoint.expanduser().resolve(strict=True)
    selected_path = args.selected_checkpoint.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite visualization: {output_dir}")

    import torch
    import yaml

    from omtrackvla.models.end_to_end import (
        ArchitectureV1Ablation,
        ArchitectureV1Config,
        load_official_da3_small_l11,
    )
    from omtrackvla.models.end_to_end_v2 import (
        ArchitectureV2DecoderConfig,
        ArchitectureV2FollowPolicy,
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repository = config_path.parents[2]
    for field in ("da3_source", "da3_runtime"):
        extra = resolve(repository, config[field]).resolve(strict=True)
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    architecture = ArchitectureV1Config(**config["architecture"])
    decoder = ArchitectureV2DecoderConfig(**config["decoder"])
    manifest = verify_manifest(
        resolve(repository, config["recovery_manifest"]).resolve(strict=True)
    )
    samples, metadata = load_recovery_samples(manifest, architecture, torch)
    wanted = (("at", "reacquisition"), ("dt", "false_visible"), ("stt", "false_invisible"))
    indices = []
    for task, category in wanted:
        match = next(
            (
                index
                for index, item in enumerate(metadata)
                if item["task"] == task and item["category"] == category
            ),
            None,
        )
        if match is None:
            raise ValueError(f"representative sample missing: {task}/{category}")
        indices.append(match)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    da3, _ = load_official_da3_small_l11(
        resolve(repository, config["da3_model"]),
        architecture,
        ArchitectureV1Ablation(backbone_tuning="adapter"),
    )
    policy = ArchitectureV2FollowPolicy(da3, architecture, decoder).to(device)
    batch = recovery_batch(samples, indices, device)
    predictions = {}
    for label, checkpoint_path in (("parent", parent_path), ("selected", selected_path)):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("method") != config["method"] or checkpoint.get("test_locked_used") is not False:
            raise ValueError(f"inadmissible {label} checkpoint")
        policy.load_state_dict(checkpoint["model"], strict=True)
        policy.eval()
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()
        ):
            output = policy(**recovery_inputs(batch))
        predictions[label] = {
            key: value.detach().float().cpu()
            for key, value in output.items()
            if torch.is_tensor(value)
        }
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for batch_index, sample_index in enumerate(indices):
        sample = samples[sample_index]
        item = metadata[sample_index]
        expert = sample["waypoints"].tolist()
        parent_waypoints = predictions["parent"]["waypoints"][batch_index].tolist()
        selected_waypoints = predictions["selected"]["waypoints"][batch_index].tolist()
        output = output_dir / f"{item['task']}_{item['category']}.png"
        render(
            sample["ego_rgb"][-1],
            expert,
            parent_waypoints,
            selected_waypoints,
            f"{item['sample_id']} | GT visible={bool(sample['visible'])}",
            output,
        )

        def metrics(waypoints: list[list[float]]) -> dict[str, float]:
            predicted = np.asarray(waypoints, dtype=np.float64)
            target = np.asarray(expert, dtype=np.float64)
            error = np.linalg.norm(predicted[1:] - target[1:], axis=1)
            path_length = np.linalg.norm(np.diff(predicted, axis=0), axis=1).sum()
            expert_length = np.linalg.norm(np.diff(target, axis=0), axis=1).sum()
            return {
                "ade_m": float(error.mean()),
                "fde_m": float(error[-1]),
                "path_length_ratio": float(path_length / max(1.0e-6, expert_length)),
            }

        rows.append(
            {
                **item,
                "gt_visible": bool(sample["visible"]),
                "parent": {
                    **metrics(parent_waypoints),
                    "visibility_probability": float(
                        torch.sigmoid(predictions["parent"]["visibility_logit"][batch_index]).item()
                    ),
                },
                "selected": {
                    **metrics(selected_waypoints),
                    "visibility_probability": float(
                        torch.sigmoid(predictions["selected"]["visibility_logit"][batch_index]).item()
                    ),
                },
                "visualization": str(output),
                "visualization_sha256": sha256(output),
            }
        )
    report = {
        "schema_version": 1,
        "stage": "v2_010_phase3_selection_visualization",
        "parent_checkpoint_sha256": sha256(parent_path),
        "selected_checkpoint_sha256": sha256(selected_path),
        "samples": rows,
        "test_locked_used": False,
    }
    (output_dir / "REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
