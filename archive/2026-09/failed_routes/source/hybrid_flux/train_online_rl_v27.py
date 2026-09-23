"""V27: stable IL -> online-PPO runner for end-to-end FLUX person tracking.

This is a new entry point layered on V26.  The older runners remain
reproducible.  V27 fixes the protocol issues found in the unfinished V26
formal run:

* training and validation use the same 3000-pixel main-person threshold;
* reward uses the instantaneous ``did_multi_agents_collide`` event and a
  rising-edge fallback, rather than the sticky ``human_collision`` metric;
* PPO adds a KL-style anchor to the zero-residual IL FLUX action distribution;
* the default probe/formal configuration trains only the small residual and
  critic heads, so an early RL run cannot destroy the IL controller;
* rollout keeps TrackEnv's already post-processed ``step`` observation and
  does not call ``sim.get_sensor_observations`` a second time;
* partial rollouts are no longer falsely counted as completed episodes.

The residual actor is still attached to the FLUX Linear-RF action path.  The
RL-only warm phase is intentional: after validation demonstrates that the
reward and recovery direction are correct, a separately controlled decoder
unfreeze can be tested with the same anchor.  This file never feeds a bbox,
panoptic mask, target pose, detector image, or oracle action to the policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from . import online_rl_utils_v22 as rl_utils_v22
    from . import train_online_rl_v23 as v23
    from . import train_online_rl_v26 as v26
except ImportError:  # pragma: no cover - supports direct script execution
    from hybrid_flux import online_rl_utils_v22 as rl_utils_v22  # type: ignore
    from hybrid_flux import train_online_rl_v23 as v23  # type: ignore
    from hybrid_flux import train_online_rl_v26 as v26  # type: ignore


v22 = v26.v22
v25 = v26.v25


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name, str(default)).strip()
    try:
        result = float(value)
    except ValueError:
        return float(default)
    return result if math.isfinite(result) else float(default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _patch_cli_namespace() -> None:
    """Record V27 knobs in checkpoints without changing the V22 CLI."""

    if getattr(v22, "_v27_cli_patch", False):
        return
    original_parse_args = v22.parse_args

    def parse_args_v27():
        args = original_parse_args()
        args.v27_anchor_coef = _env_float("V27_ANCHOR_COEF", 0.05)
        args.v27_anchor_log_std = _env_float(
            "V27_ANCHOR_LOG_STD", math.log(max(float(args.initial_std), 1e-6))
        )
        args.v27_anchor_log_std_coef = _env_float(
            "V27_ANCHOR_LOG_STD_COEF", 0.10
        )
        args.v27_rl_only = _env_bool("V27_RL_ONLY", True)
        args.v27_following_pixel_threshold = int(
            max(1, round(_env_float("V27_FOLLOWING_PIXEL_THRESHOLD", 3000)))
        )
        return args

    v22.parse_args = parse_args_v27
    v22._v27_cli_patch = True


def _patch_config_threshold() -> None:
    """Make train and val use the same main-person detector threshold."""

    if getattr(v22, "_v27_config_patch", False):
        return
    original_configure = v22._configure_env_config

    def configure_v27(
        config: Any,
        split: str,
        local_rank: int,
        max_steps: int,
        seed: int,
    ) -> Any:
        result = original_configure(
            config, split, local_rank, max_steps, seed
        )
        # Habitat's structured config is read-only outside read_write().
        from habitat.config.read_write import read_write  # type: ignore

        threshold = int(
            max(1, round(_env_float("V27_FOLLOWING_PIXEL_THRESHOLD", 3000)))
        )
        with read_write(result):
            lab_sensors = result.habitat.task.lab_sensors
            for name in (
                "agent_1_main_humanoid_detector_sensor",
                "agent_1_other_humanoid_detector_sensor",
            ):
                if name in lab_sensors:
                    lab_sensors[name].human_pixel_threshold = threshold
        return result

    v22._configure_env_config = configure_v27
    v22._v27_config_patch = True


RGB_KEY = "agent_1_articulated_agent_jaw_rgb"
DEPTH_KEY = "agent_1_articulated_agent_jaw_depth"


class _ProcessedObservationEnv:
    """TrackEnv proxy that preserves its normal processed observation path."""

    def __init__(self, env: Any, task: str, split: str) -> None:
        self._env = env
        self._task = str(task)
        self._split = str(split)
        self._step_index = 0
        self._records = []
        raw_dir = os.environ.get("V27_OBS_DIAG_DIR", "").strip()
        self._diag_dir = Path(raw_dir).expanduser() if raw_dir else None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @staticmethod
    def _hash(value: Any) -> Optional[str]:
        try:
            return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()[:20]
        except Exception:  # pragma: no cover - diagnostic only
            return None

    def _check_and_record(self, observation: Any, source: str) -> Any:
        if not isinstance(observation, dict):
            raise TypeError(
                f"TrackEnv must return a dict observation, got {type(observation)!r}"
            )
        if RGB_KEY not in observation or DEPTH_KEY not in observation:
            raise KeyError(
                "TrackEnv processed observation is missing product RGB/depth: "
                f"rgb={RGB_KEY in observation} depth={DEPTH_KEY in observation}"
            )
        if self._diag_dir is not None:
            rgb = observation.get(RGB_KEY)
            depth = observation.get(DEPTH_KEY)
            self._records.append(
                {
                    "index": int(self._step_index),
                    "source": source,
                    "rgb_shape": list(np.asarray(rgb).shape),
                    "depth_shape": list(np.asarray(depth).shape),
                    "rgb_sha256_prefix": self._hash(rgb),
                    "depth_sha256_prefix": self._hash(depth),
                }
            )
        return observation

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        observation = self._env.reset(*args, **kwargs)
        self._step_index = 0
        return self._check_and_record(observation, "TrackEnv.reset_processed")

    def step(self, *args: Any, **kwargs: Any) -> Any:
        # TrackEnv.step() already calls the task's _get_observations(), which
        # performs sensor post-processing.  Calling sim.get_sensor_observations
        # again here doubles rendering cost and caused DDP stragglers in V24.
        observation = self._env.step(*args, **kwargs)
        self._step_index += 1
        return self._check_and_record(
            observation, "TrackEnv.step_processed"
        )

    def _write_diagnostic(self) -> None:
        if self._diag_dir is None:
            return
        try:
            self._diag_dir.mkdir(parents=True, exist_ok=True)
            rank = os.environ.get("RANK", "0")
            path = self._diag_dir / f"obs_{self._task}_{self._split}_rank{rank}.json"
            post_step = [
                item
                for item in self._records
                if item["source"] == "TrackEnv.step_processed"
            ]
            payload = {
                "task": self._task,
                "split": self._split,
                "rank": int(rank),
                "post_step_observation_source": "TrackEnv.step_processed",
                "raw_sim_observation_used": False,
                "policy_uses_ground_truth": False,
                "records": self._records,
                "post_step_record_count": len(post_step),
                "unique_post_step_rgb_hashes": len(
                    {item["rgb_sha256_prefix"] for item in post_step}
                ),
            }
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"[v27 observation diagnostic skipped] {exc}", flush=True)

    def close(self) -> Any:
        self._write_diagnostic()
        return self._env.close()


def _make_env_v27(
    habitat: Any,
    make_dataset: Any,
    task: str,
    split: str,
    local_rank: int,
    task_rank: int,
    task_world_size: int,
    max_steps: int,
    seed: int,
    max_episodes: int = 0,
) -> Tuple[Any, Any]:
    env, config = v23._stable_balanced_make_env(
        habitat,
        make_dataset,
        task,
        split,
        local_rank,
        task_rank,
        task_world_size,
        max_steps,
        seed,
        max_episodes,
    )
    return _ProcessedObservationEnv(env, task, split), config


def _patch_observation_and_metrics() -> None:
    if not getattr(v22, "_v27_observation_patch", False):
        v22._make_env = _make_env_v27

        original_snapshot = v22._metric_snapshot

        def metric_snapshot_v27(metrics: Dict[str, Any]) -> Dict[str, float]:
            snapshot = original_snapshot(metrics)
            proper_available = "did_multi_agents_collide" in metrics
            snapshot.update(
                {
                    "did_multi_agents_collide": float(
                        rl_utils_v22.metric_float(
                            metrics, "did_multi_agents_collide", 0.0
                        )
                        > 0.5
                    ),
                    "collision_metric_available": float(proper_available),
                    "legacy_human_collision": rl_utils_v22.metric_float(
                        metrics, "human_collision", 0.0
                    ),
                }
            )
            return snapshot

        v22._metric_snapshot = metric_snapshot_v27
        v22._v27_observation_patch = True

    try:
        from .online_rl_utils_v27 import tracking_reward_v27
    except ImportError:  # pragma: no cover - direct script execution
        from hybrid_flux.online_rl_utils_v27 import tracking_reward_v27  # type: ignore

    # collect_rollout imports tracking_reward from this module at call time.
    rl_utils_v22.tracking_reward = tracking_reward_v27


def _patch_status_annotation() -> None:
    """Undo V24's raw-observation annotation for the V27 processed path."""

    if getattr(v23, "_v27_status_patch", False):
        return
    original = v26.v25.v24._annotate_status_from_cli

    def annotate_status_v27() -> None:
        try:
            index = sys.argv.index("--output-dir") + 1
            output_dir = Path(sys.argv[index]).expanduser()
            status_path = output_dir / "online_rl_status.json"
            if not status_path.exists():
                return
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            contract = payload.setdefault("data_contract", {})
            contract.update(
                {
                    "raw_sim_observation_used": False,
                    "fresh_post_step_observation": False,
                    "post_step_observation_source": "TrackEnv.step_processed",
                    "sim_truth_as_policy_input": False,
                }
            )
            status_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except (IndexError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"[v27 status annotation skipped] {exc}", flush=True)

    # V25.main() invokes this exact function in its finally block.
    v26.v25.v24._annotate_status_from_cli = annotate_status_v27
    v23._v27_status_patch = True
    del original


def _patch_rl_configuration() -> None:
    """Optionally keep only the new residual/critic heads trainable."""

    try:
        from .hybrid_policy_ilrl_v22 import HybridLinearRFActorCriticV22
    except ImportError:  # pragma: no cover - direct script execution
        from hybrid_flux.hybrid_policy_ilrl_v22 import HybridLinearRFActorCriticV22  # type: ignore

    if getattr(HybridLinearRFActorCriticV22, "_v27_config_patch", False):
        return
    original_configure = HybridLinearRFActorCriticV22.configure_rl_finetune

    def configure_v27(self: Any, *args: Any, **kwargs: Any) -> None:
        original_configure(self, *args, **kwargs)
        if _env_bool("V27_RL_ONLY", True):
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(name.startswith("rl_"))
            print(
                "[V27 setup] RL-only warm phase: trainable parameters are "
                "rl_context/rl_residual_head/rl_value_head/rl_log_std",
                flush=True,
            )

    HybridLinearRFActorCriticV22.configure_rl_finetune = configure_v27
    HybridLinearRFActorCriticV22._v27_config_patch = True


def _patch_partial_rollout_logging() -> None:
    """Do not report a truncated rollout as a completed episode."""

    if getattr(v22.RolloutBuffer, "_v27_partial_patch", False):
        return

    def finish_partial_episode_v27(self: Any) -> None:
        if getattr(self, "_length", 0):
            self.partial_episode_return = float(self._return)
            self.partial_episode_length = int(self._length)
            self._return = 0.0
            self._length = 0

    v22.RolloutBuffer.finish_partial_episode = finish_partial_episode_v27
    v22.RolloutBuffer._v27_partial_patch = True


def _v27_synchronized_ppo_update(
    train_model: torch.nn.Module,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: Dict[str, np.ndarray],
    device: torch.device,
    args: Any,
    params: Sequence[torch.nn.Parameter],
) -> Dict[str, float]:
    """PPO update with a synchronized early stop and IL behavior anchor."""

    n = int(data["actions"].shape[0])
    if n < 1:
        raise RuntimeError("cannot run PPO update on an empty rollout")
    data["advantages"] = v22._normalise_advantages(data["advantages"], device)
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda" and args.amp_dtype == "fp16")
    )
    sums = Counter()
    count = 0
    early_kl = False
    reference_log_std = float(args.v27_anchor_log_std)
    anchor_coef = max(0.0, float(args.v27_anchor_coef))
    anchor_log_std_coef = max(0.0, float(args.v27_anchor_log_std_coef))

    for epoch in range(int(args.ppo_epochs)):
        generator = np.random.RandomState(
            int(args.seed) + args.update * 1009 + epoch
        )
        indices = generator.permutation(n)
        for start in range(0, n, int(args.minibatch_size)):
            batch_indices = indices[start : start + int(args.minibatch_size)]
            batch = v22._batch_to_device(data, batch_indices, device)
            optimizer.zero_grad(set_to_none=True)
            with v22._autocast(device, bool(args.amp), args.amp_dtype):
                output = v22._actor_critic_forward(
                    train_model,
                    input_images=batch["rgb"],
                    input_depths=batch["depth"],
                    reference_images=batch["reference"],
                    point_state=batch["point_state"],
                    actions=batch["actions"],
                    deterministic=False,
                )
                log_ratio = output["log_prob"] - batch["old_log_prob"]
                ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))
                clipped_ratio = ratio.clamp(
                    1.0 - float(args.clip_ratio),
                    1.0 + float(args.clip_ratio),
                )
                policy_loss = -torch.minimum(
                    ratio * batch["advantages"],
                    clipped_ratio * batch["advantages"],
                ).mean()
                value = output["value"]
                if args.value_clip > 0:
                    value_clipped = batch["old_value"] + (
                        value - batch["old_value"]
                    ).clamp(-float(args.value_clip), float(args.value_clip))
                    value_loss = 0.5 * torch.maximum(
                        (value - batch["returns"]).square(),
                        (value_clipped - batch["returns"]).square(),
                    ).mean()
                else:
                    value_loss = 0.5 * (
                        value - batch["returns"]
                    ).square().mean()
                entropy = output["entropy"].mean()

                # In the RL-only warm phase, ``base_latent`` is exactly the
                # deterministic IL Linear-RF action.  Since
                # mean=base_latent+residual_latent, the expression below is
                # the diagonal-Gaussian KL(current || IL) in latent space.
                # It keeps exploration possible but prevents PPO from
                # immediately replacing a working IL action with noise.
                current_log_std = output["log_std"]
                current_std_sq = torch.exp(2.0 * current_log_std)
                reference_std_sq = math.exp(2.0 * reference_log_std)
                residual = output["residual_latent"]
                il_kl = 0.5 * (
                    (
                        current_std_sq + residual.square()
                    )
                    / reference_std_sq
                    - 1.0
                    + 2.0 * (reference_log_std - current_log_std)
                ).mean()
                il_kl = il_kl.clamp_min(0.0)
                anchor_loss = il_kl + anchor_log_std_coef * (
                    current_log_std - reference_log_std
                ).square().mean()
                loss = (
                    policy_loss
                    + float(args.value_coef) * value_loss
                    - float(args.entropy_coef) * entropy
                    + anchor_coef * anchor_loss
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite V27 PPO loss at update={args.update}: "
                    f"{loss.detach().item()}"
                )
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                params, float(args.max_grad_norm)
            )
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            approx_kl = float(
                (batch["old_log_prob"] - output["log_prob"])
                .mean()
                .detach()
                .float()
                .cpu()
            )
            clip_fraction = float(
                (torch.abs(ratio - 1.0) > float(args.clip_ratio))
                .float()
                .mean()
                .detach()
                .cpu()
            )
            local_logs = {
                "loss": float(loss.detach().float().cpu()),
                "policy_loss": float(policy_loss.detach().float().cpu()),
                "value_loss": float(value_loss.detach().float().cpu()),
                "entropy": float(entropy.detach().float().cpu()),
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
                "grad_norm": float(grad_norm.detach().float().cpu()),
                "il_anchor_kl": float(il_kl.detach().float().cpu()),
                "il_anchor_loss": float(anchor_loss.detach().float().cpu()),
                "residual_rms": float(
                    residual.detach().square().mean().sqrt().float().cpu()
                ),
            }
            for key, value_item in local_logs.items():
                sums[key] += value_item
            count += 1

            stop_tensor = torch.tensor(
                1 if approx_kl > float(args.target_kl) else 0,
                dtype=torch.int64,
                device=device,
            )
            if v22.dist.is_available() and v22.dist.is_initialized():
                v22.dist.all_reduce(stop_tensor, op=v22.dist.ReduceOp.MAX)
            if int(stop_tensor.item()) == 1:
                early_kl = True
                break
        if early_kl:
            break

    keys = (
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "grad_norm",
        "il_anchor_kl",
        "il_anchor_loss",
        "residual_rms",
    )
    reduced = v22._all_reduce_vector(
        [sums[key] for key in keys] + [float(count)], device
    )
    denom = max(float(reduced[-1]), 1.0)
    return {
        **{key: float(reduced[index] / denom) for index, key in enumerate(keys)},
        "ppo_minibatches": float(reduced[-1]),
        "early_kl_stop": float(1.0 if early_kl else 0.0),
    }


def _patch_ppo() -> None:
    # V25's diagnostic wrapper calls v25._original_ppo_update dynamically.
    v25._original_ppo_update = _v27_synchronized_ppo_update


_patch_cli_namespace()
_patch_config_threshold()
_patch_observation_and_metrics()
_patch_status_annotation()
_patch_rl_configuration()
_patch_partial_rollout_logging()
_patch_ppo()


def main() -> None:
    v25.main()


if __name__ == "__main__":
    main()
