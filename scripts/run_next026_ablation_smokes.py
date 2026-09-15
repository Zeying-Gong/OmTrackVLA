#!/usr/bin/env python3
"""Run short, config-recorded Architecture-v1 ablation train/eval/render checks."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


DEFAULT_VARIANTS = {
    "main_control": {},
    "abl01_dinov2": {
        "dinov2_model": "/data/nfs/share/OmTrackVLA/models/dinov2/dinov2_vits14_pretrain.pth",
        "ablation": {"backbone_weights": "dinov2"},
    },
    "abl01_frozen": {"ablation": {"backbone_tuning": "frozen"}},
    "abl02_l5_l11": {"ablation": {"feature_fusion": "l5_l11"}},
    "abl03_bbox_mean": {"ablation": {"target_pooling": "bbox_mean"}},
    "abl04_single_step": {"ablation": {"temporal_fusion": "single_step"}},
    "abl05_raw_camera_diff": {
        "ablation": {"ego_representation": "raw_camera_difference"}
    },
    "abl05_no_ego": {"ablation": {"ego_representation": "none"}},
    "abl06_late_only": {"ablation": {"uwb_early_fusion": "none"}},
    "abl06_learned_projection": {
        "ablation": {"uwb_early_fusion": "learned"}
    },
    "abl07_b_correct": {
        "training": {
            "world_action_variant": "b",
            "action_input_mode": "correct",
            "loss_weights": {"world_action": 0.1, "inverse": 0.1},
        }
    },
    "abl07_a_latent_only": {
        "training": {
            "world_action_variant": "latent_only",
            "action_input_mode": "correct",
            "loss_weights": {"world_action": 0.1, "inverse": 0.1},
        }
    },
    "abl07_b_zero_action": {
        "training": {
            "world_action_variant": "b",
            "action_input_mode": "zero",
            "loss_weights": {"world_action": 0.1, "inverse": 0.1},
        }
    },
    "abl07_b_shuffled_action": {
        "training": {
            "world_action_variant": "b",
            "action_input_mode": "shuffled",
            "loss_weights": {"world_action": 0.1, "inverse": 0.1},
        }
    },
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/phases/phase2_end_to_end_v1_formal.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/ablations/next026_smoke_v1"),
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--samples-per-mode", type=int, default=128)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--run-kind", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--skip-render", action="store_true")
    return parser.parse_args()


def merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def append_event(path: Path, event: dict) -> None:
    event = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def run_logged(command: list[str], log_path: Path, repository: Path) -> float:
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=repository,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return time.monotonic() - started


def metric_summary(metrics: dict) -> dict:
    normal_names = ("visual_uwb", "visual_only", "uwb_only")
    normal = [metrics["modes"][name] for name in normal_names]
    return {
        "checkpoint_global_step": metrics["checkpoint_global_step"],
        "normal_ade_m": sum(value["waypoint_ade_m"] for value in normal) / 3.0,
        "normal_fde_m": sum(value["waypoint_fde_m"] for value in normal) / 3.0,
        "normal_path_length_ratio": sum(
            value["path_length_ratio_of_sums"] for value in normal
        )
        / 3.0,
        "finite_prediction_coverage": min(
            value["finite_prediction_coverage"] for value in metrics["modes"].values()
        ),
        "safe_stop_accuracy": metrics["modes"]["safe_stop"]["stop_accuracy"],
    }


def main() -> int:
    args = arguments()
    repository = args.repository.expanduser().resolve(strict=True)
    base_config = args.base_config
    if not base_config.is_absolute():
        base_config = repository / base_config
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repository / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    # The existing trainer resolves the repository as config.parents[2], so
    # generated configs must sit directly in configs/phases.  Each run also
    # copies the exact config into its output directory for provenance.
    configs_dir = repository / "configs" / "phases"
    events_path = output_root / "events.jsonl"
    python = Path(sys.executable)
    requested = [name for name in args.variants.split(",") if name]
    unknown = sorted(set(requested) - set(DEFAULT_VARIANTS))
    if unknown:
        raise ValueError(f"unknown ablation variants: {unknown}")
    base = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    summary_path = output_root / "summary.json"
    summary: dict[str, dict] = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )

    for name in requested:
        run_dir = output_root / name
        config_path = configs_dir / f".generated_next026_smoke_{name}.yaml"
        config = merge(base, DEFAULT_VARIANTS[name])
        config["stage"] = f"next026_{args.run_kind}_{name}"
        # The trainer intentionally accepts only the audited formal schema.
        # The ten-step CLI cap and stage name identify this run as a smoke.
        config["formal_training"] = True
        config["test_locked_used"] = False
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        complete = run_dir / "TRAINING_COMPLETE.json"
        append_event(
            events_path,
            {"event": "variant_start", "variant": name, "run_kind": args.run_kind},
        )
        if not complete.is_file():
            elapsed_train = run_logged(
                [
                    str(python),
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc-per-node={args.nproc_per_node}",
                    "-m",
                    "omtrackvla.training.end_to_end_v1",
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(run_dir),
                    "--max-steps",
                    str(args.steps),
                ],
                run_dir.with_suffix(".train.log"),
                repository,
            )
        else:
            elapsed_train = 0.0

        metrics_path = run_dir / f"eval_{args.samples_per_mode}" / "metrics.json"
        if not metrics_path.is_file():
            elapsed_eval = run_logged(
                [
                    str(python),
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc-per-node={args.nproc_per_node}",
                    "-m",
                    "omtrackvla.evaluation.end_to_end_evaluate",
                    "--config",
                    str(config_path),
                    "--checkpoint",
                    str(run_dir / "checkpoints" / "best.ckpt"),
                    "--output",
                    str(metrics_path),
                    "--samples-per-mode",
                    str(args.samples_per_mode),
                ],
                run_dir.with_suffix(".eval.log"),
                repository,
            )
        else:
            elapsed_eval = 0.0

        render_path = run_dir / "fixed_sample.png"
        if not args.skip_render and not render_path.is_file():
            elapsed_render = run_logged(
                [
                    str(python),
                    "scripts/render_end_to_end_v1_checkpoint.py",
                    "--config",
                    str(config_path),
                    "--checkpoint",
                    str(run_dir / "checkpoints" / "best.ckpt"),
                    "--output",
                    str(render_path),
                    "--report",
                    str(run_dir / "fixed_sample.json"),
                    "--device",
                    "cuda:7",
                ],
                run_dir.with_suffix(".render.log"),
                repository,
            )
        else:
            elapsed_render = 0.0
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        summary[name] = {
            **metric_summary(metrics),
            "elapsed_train_s": elapsed_train,
            "elapsed_eval_s": elapsed_eval,
            "elapsed_render_s": elapsed_render,
            "config": str(config_path),
            "checkpoint": str(run_dir / "checkpoints" / "best.ckpt"),
            "visualization": str(render_path),
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        append_event(
            events_path,
            {"event": "variant_complete", "variant": name, **summary[name]},
        )
    append_event(events_path, {"event": "suite_complete", "variants": requested})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
