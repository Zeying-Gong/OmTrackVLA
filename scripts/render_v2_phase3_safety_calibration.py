#!/usr/bin/env python3
"""Render the effects and remaining errors of v2_017 safety-head calibration."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from train_v2_phase3_model_visited import (
    load_recovery_samples,
    resolve,
    sha256,
    verify_manifest,
)
from train_v2_phase3_observable import apply_observability, verify_observability_audit


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def choose_examples(
    base: Mapping[str, Mapping[str, Any]],
    calibrated: Mapping[str, Mapping[str, Any]],
    threshold: float,
) -> list[tuple[str, str]]:
    common = sorted(set(base) & set(calibrated))

    def candidates(predicate: Any) -> list[str]:
        return [sample_id for sample_id in common if predicate(base[sample_id], calibrated[sample_id])]

    repaired_safe_stop = candidates(
        lambda before, after: (
            not bool(after["gt_visible"])
            and bool(after["stop_target"])
            and float(before["visibility_probability"]) >= threshold
            and float(after["visibility_probability"]) < threshold
        )
    )
    repaired_false_invisible = candidates(
        lambda before, after: (
            bool(after["gt_visible"])
            and float(before["visibility_probability"]) < threshold
            and float(after["visibility_probability"]) >= threshold
        )
    )
    remaining_false_invisible = candidates(
        lambda _before, after: (
            bool(after["gt_visible"])
            and float(after["visibility_probability"]) < threshold
        )
    )
    remaining_false_visible = candidates(
        lambda _before, after: (
            not bool(after["gt_visible"])
            and float(after["visibility_probability"]) >= threshold
        )
    )
    groups = (
        ("repaired_hard_safe_stop_visibility", repaired_safe_stop),
        ("repaired_false_invisible", repaired_false_invisible),
        ("remaining_false_invisible", remaining_false_invisible),
        ("remaining_false_visible", remaining_false_visible),
    )
    missing = [label for label, values in groups if not values]
    if missing:
        raise ValueError(f"diagnostic has no examples for: {missing}")

    def priority(label: str, values: list[str]) -> str:
        if label == "repaired_hard_safe_stop_visibility":
            return max(values, key=lambda key: float(base[key]["visibility_probability"]))
        if label == "remaining_false_visible":
            return max(values, key=lambda key: float(calibrated[key]["visibility_probability"]))
        return values[0]

    return [(label, priority(label, values)) for label, values in groups]


def tensor_to_rgb(tensor: Any) -> Image.Image:
    array = (
        np.asarray(tensor.permute(1, 2, 0).float().cpu())
        .clip(0.0, 1.0)
        .__mul__(255.0)
        .astype(np.uint8)
    )
    return Image.fromarray(array, mode="RGB")


def render(
    rgb: Image.Image,
    role: str,
    sample_id: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    threshold: float,
) -> tuple[Image.Image, float]:
    canvas = Image.new("RGB", (1200, 470), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    title_font, text_font, small_font = font(23), font(18), font(15)
    role_text = role.replace("_", " ")
    draw.rectangle((0, 0, 1200, 108), fill=(22, 26, 32))
    draw.text((14, 8), f"{role_text} | {sample_id}", fill="white", font=title_font)
    draw.text(
        (14, 40),
        f"GT visible={bool(after['gt_visible'])}  GT stop={bool(after['stop_target'])}  "
        f"visibility threshold={threshold:.3f}",
        fill=(230, 230, 230),
        font=text_font,
    )
    draw.text(
        (14, 67),
        "visibility  "
        f"{float(before['visibility_probability']):.6f} -> {float(after['visibility_probability']):.6f}    "
        "stop  "
        f"{float(before['stop_probability']):.6f} -> {float(after['stop_probability']):.6f}",
        fill=(230, 230, 230),
        font=text_font,
    )

    rgb = rgb.resize((630, 350), Image.Resampling.BILINEAR)
    canvas.paste(rgb, (0, 120))
    draw.rectangle((0, 120, 630, 470), outline=(70, 70, 70), width=2)
    draw.rectangle((0, 120, 630, 149), fill=(0, 0, 0))
    draw.text((10, 126), "current RGB (last of 8 history frames)", fill="white", font=small_font)

    expert = np.asarray(before["predicted_waypoints_xy_m"], dtype=np.float64)
    calibrated = np.asarray(after["predicted_waypoints_xy_m"], dtype=np.float64)
    max_delta = float(np.abs(expert - calibrated).max())
    target = np.asarray(after.get("gt_waypoints_xy_m", []), dtype=np.float64)
    paths = [(expert, (35, 165, 70), 8), (calibrated, (45, 95, 230), 4)]
    if target.shape == expert.shape:
        paths.insert(0, (target, (230, 125, 25), 5))
    all_points = np.concatenate([values for values, _, _ in paths], axis=0)
    forward_max = max(0.5, float(np.max(all_points[:, 0])) + 0.1)
    forward_min = min(-0.1, float(np.min(all_points[:, 0])) - 0.1)
    lateral_abs = max(0.5, float(np.max(np.abs(all_points[:, 1]))) + 0.1)
    plot_left, plot_right = 662, 1180
    plot_top, plot_bottom = 157, 440
    scale = min(
        (plot_bottom - plot_top) / max(1.0e-6, forward_max - forward_min),
        (plot_right - plot_left) / max(1.0e-6, 2.0 * lateral_abs),
    )
    origin_x = (plot_left + plot_right) // 2
    origin_y = int(round(plot_bottom + forward_min * scale))
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), fill="white", outline=(120, 120, 120))
    draw.line((origin_x, plot_top, origin_x, plot_bottom), fill=(190, 190, 190), width=1)
    draw.line((plot_left, origin_y, plot_right, origin_y), fill=(190, 190, 190), width=1)

    def pixels(values: np.ndarray) -> list[tuple[int, int]]:
        return [
            (
                int(round(origin_x - float(left) * scale)),
                int(round(origin_y - float(forward) * scale)),
            )
            for forward, left in values
        ]

    for values, color, width in paths:
        points = pixels(values)
        if len(points) > 1:
            draw.line(points, fill=color, width=width, joint="curve")
        for point in points:
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
    draw.text((662, 122), "trajectory: green base outline | blue calibrated", fill=(25, 25, 25), font=small_font)
    draw.text(
        (662, 142),
        f"max waypoint coordinate delta = {max_delta:.3e} m (waypoint path frozen)",
        fill=(25, 25, 25),
        font=small_font,
    )
    draw.text((1080, 444), "forward up", fill=(70, 70, 70), font=small_font)
    return canvas, max_delta


def main() -> int:
    args = arguments()
    config_path = args.config.expanduser().resolve(strict=True)
    diagnostic_path = args.diagnostic.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite visualization: {output_dir}")

    repository = config_path.parents[2]
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))

    import torch
    import yaml

    from omtrackvla.models.end_to_end import ArchitectureV1Config

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if (
        config.get("method") != "architecture_v2_evt_perception_polar"
        or config.get("test_locked_used") is not False
        or diagnostic.get("test_locked_used") is not False
    ):
        raise ValueError("visualization provenance mismatch")
    threshold = float(diagnostic["deployment_visibility_threshold"])
    checkpoints = {item["label"]: item for item in diagnostic["checkpoints"]}
    if set(checkpoints) != {"base", "calibrated"}:
        raise ValueError("expected exactly base and calibrated diagnostic checkpoints")
    prediction_maps = {
        label: {sample["sample_id"]: sample for sample in item["samples"]}
        for label, item in checkpoints.items()
    }
    selected = choose_examples(prediction_maps["base"], prediction_maps["calibrated"], threshold)

    architecture = ArchitectureV1Config(**config["architecture"])
    manifest_path = resolve(repository, config["recovery_val_manifest"]).resolve(strict=True)
    audit_path = resolve(repository, config["recovery_val_observability_audit"]).resolve(strict=True)
    manifest = verify_manifest(manifest_path)
    audit = verify_observability_audit(audit_path, manifest_path, "recovery_val")
    samples, metadata = load_recovery_samples(manifest, architecture, torch)
    apply_observability(samples, metadata, audit, torch)
    sample_map = {item["sample_id"]: sample for sample, item in zip(samples, metadata)}

    output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    rendered = []
    for role, sample_id in selected:
        before = prediction_maps["base"][sample_id]
        after = prediction_maps["calibrated"][sample_id]
        sample = sample_map[sample_id]
        if bool(sample["visible"]) != bool(after["gt_visible"]):
            raise ValueError(f"GT visibility mismatch: {sample_id}")
        image, max_delta = render(
            tensor_to_rgb(sample["ego_rgb"][-1]), role, sample_id, before, after, threshold
        )
        path = output_dir / f"{role}.png"
        image.save(path)
        rendered.append(image)
        rows.append(
            {
                "role": role,
                "sample_id": sample_id,
                "gt_visible": bool(after["gt_visible"]),
                "gt_stop": bool(after["stop_target"]),
                "base_visibility_probability": float(before["visibility_probability"]),
                "calibrated_visibility_probability": float(after["visibility_probability"]),
                "base_stop_probability": float(before["stop_probability"]),
                "calibrated_stop_probability": float(after["stop_probability"]),
                "max_waypoint_coordinate_delta_m": max_delta,
                "image": str(path),
                "image_sha256": sha256(path),
            }
        )
    sheet = Image.new("RGB", (2400, 940), "white")
    for index, image in enumerate(rendered):
        sheet.paste(image, ((index % 2) * 1200, (index // 2) * 470))
    sheet_path = output_dir / "CONTACT_SHEET.png"
    sheet.save(sheet_path)
    report = {
        "schema_version": 1,
        "stage": "v2_017_phase3_safety_calibration_visualization",
        "diagnostic": str(diagnostic_path),
        "diagnostic_sha256": sha256(diagnostic_path),
        "visibility_threshold": threshold,
        "samples": rows,
        "contact_sheet": str(sheet_path),
        "contact_sheet_sha256": sha256(sheet_path),
        "test_locked_used": False,
    }
    (output_dir / "REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
