#!/usr/bin/env python3
"""Run converged Architecture-v1 Phase-1/teacher ablations through one recipe."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import yaml


PHASE1_CHECKPOINTS = {
    "teacher_off": Path(
        "outputs/training/next026_architecture_v1_phase1_pretrain/checkpoints/best.ckpt"
    ),
    "teacher_on": Path(
        "outputs/training/next026_architecture_v1_phase1_osnet_teacher/checkpoints/best.ckpt"
    ),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("outputs/ablations/next026_converged_probe_v1/direct_36k_base.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/ablations/next026_abl08_converged_v1"),
    )
    parser.add_argument("--samples-per-mode", type=int, default=1024)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    return parser.parse_args()


def run(command: list[str], *, repository: Path, log_path: Path) -> float:
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=repository,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return time.monotonic() - started


def write_config(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def evaluate_and_render(
    *,
    python: str,
    repository: Path,
    config_path: Path,
    run_dir: Path,
    samples_per_mode: int,
    nproc_per_node: int,
) -> dict:
    metrics_path = run_dir / f"eval_{samples_per_mode}" / "metrics.json"
    if not metrics_path.is_file():
        run(
            [
                python,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={nproc_per_node}",
                "-m",
                "omtrackvla.evaluation.end_to_end_evaluate",
                "--config",
                str(config_path),
                "--checkpoint",
                str(run_dir / "checkpoints" / "best.ckpt"),
                "--output",
                str(metrics_path),
                "--samples-per-mode",
                str(samples_per_mode),
            ],
            repository=repository,
            log_path=run_dir.with_suffix(".eval.log"),
        )
    render_path = run_dir / "fixed_sample.png"
    if not render_path.is_file():
        run(
            [
                python,
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
            repository=repository,
            log_path=run_dir.with_suffix(".render.log"),
        )
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def summary(metrics: dict) -> dict:
    normal = metrics["normal_modes"]
    return {
        "checkpoint_global_step": metrics["checkpoint_global_step"],
        "waypoint_ade_m": normal["waypoint_ade_m"],
        "waypoint_fde_m": normal["waypoint_fde_m"],
        "path_length_ratio": normal["path_length_ratio_of_sums"],
        "finite_prediction_coverage": normal["finite_prediction_coverage"],
        "safe_stop_accuracy": metrics["modes"]["safe_stop"]["stop_accuracy"],
        "test_locked_used": metrics["test_locked_used"],
    }


def main() -> int:
    args = arguments()
    repository = Path(__file__).resolve().parents[1]
    base_path = (
        args.base_config
        if args.base_config.is_absolute()
        else repository / args.base_config
    ).resolve(strict=True)
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else repository / args.output_root
    ).resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    configs_dir = repository / "configs" / "phases"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    python = sys.executable
    report: dict[str, dict] = {}

    for label, relative_checkpoint in PHASE1_CHECKPOINTS.items():
        phase1_checkpoint = (repository / relative_checkpoint).resolve(strict=True)
        phase1_complete = phase1_checkpoint.parents[1] / "TRAINING_COMPLETE.json"
        completion = json.loads(phase1_complete.read_text(encoding="utf-8"))
        if completion.get("global_step") != 4096 or completion.get("test_locked_used") is not False:
            raise ValueError(f"invalid Phase 1 completion for {label}")

        single_config = deepcopy(base)
        single_config["stage"] = f"next026_abl08_{label}_single_step_36k"
        single_config.setdefault("ablation", {})["temporal_fusion"] = "single_step"
        single_path = configs_dir / f".generated_next026_abl08_{label}_single.yaml"
        write_config(single_path, single_config)
        single_dir = output_root / label / "single_step_36k"
        if not (single_dir / "TRAINING_COMPLETE.json").is_file():
            run(
                [
                    python,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc-per-node={args.nproc_per_node}",
                    "-m",
                    "omtrackvla.training.end_to_end_v1",
                    "--config",
                    str(single_path),
                    "--output-dir",
                    str(single_dir),
                    "--init-checkpoint",
                    str(phase1_checkpoint),
                    "--max-steps",
                    "36864",
                ],
                repository=repository,
                log_path=single_dir.with_suffix(".train.log"),
            )
        single_metrics = evaluate_and_render(
            python=python,
            repository=repository,
            config_path=single_path,
            run_dir=single_dir,
            samples_per_mode=args.samples_per_mode,
            nproc_per_node=args.nproc_per_node,
        )

        gru_config = deepcopy(base)
        gru_config["stage"] = f"next026_abl08_{label}_gru_curriculum_4k"
        gru_config.setdefault("ablation", {})["temporal_fusion"] = "gru"
        training = gru_config["training"]
        training.update(
            epochs=1,
            policy_learning_rate=3.0e-5,
            gru_learning_rate=3.0e-4,
            adapter_learning_rate=3.0e-6,
            gru_initialization_after_checkpoint="near_identity",
            checkpoint_every_steps=500,
        )
        gru_path = configs_dir / f".generated_next026_abl08_{label}_gru.yaml"
        write_config(gru_path, gru_config)
        gru_dir = output_root / label / "gru_curriculum_4k"
        if not (gru_dir / "TRAINING_COMPLETE.json").is_file():
            run(
                [
                    python,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc-per-node={args.nproc_per_node}",
                    "-m",
                    "omtrackvla.training.end_to_end_v1",
                    "--config",
                    str(gru_path),
                    "--output-dir",
                    str(gru_dir),
                    "--init-checkpoint",
                    str(single_dir / "checkpoints" / "best.ckpt"),
                    "--max-steps",
                    "4096",
                ],
                repository=repository,
                log_path=gru_dir.with_suffix(".train.log"),
            )
        gru_metrics = evaluate_and_render(
            python=python,
            repository=repository,
            config_path=gru_path,
            run_dir=gru_dir,
            samples_per_mode=args.samples_per_mode,
            nproc_per_node=args.nproc_per_node,
        )
        report[label] = {
            "phase1_checkpoint": str(phase1_checkpoint),
            "single_step": summary(single_metrics),
            "gru_curriculum": summary(gru_metrics),
        }
        (output_root / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
