"""Standalone capped recovery exposure plan; does not alter or invoke training.

All identifiers and ages are sampling/diagnostic metadata, never policy inputs.
The budget is pre-DDP; the manifest separately audits the real DDP/drop-last view.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def episode_budgets(sizes, budget, max_repeat, rng):
    """Water-fill total episode exposures, respecting unique floors and caps."""
    sizes = np.asarray(sizes)
    if sizes.ndim != 1 or not len(sizes) or sizes.dtype.kind not in "iu" or (sizes <= 0).any():
        raise ValueError("episode sizes must be positive integers")
    budget = _integer(budget, "budget")
    cap = _integer(max_repeat, "max_repeat", 1)
    totals = sizes.astype(np.int64).copy()
    capacities = totals * cap
    if budget < int(totals.sum()) or budget > int(capacities.sum()):
        raise ValueError("budget outside unique-coverage/capacity bounds")
    remaining = budget - int(totals.sum())
    # At each level only minimum-exposure unsaturated episodes receive tokens.
    while remaining:
        eligible = np.flatnonzero(totals < capacities)
        minimum = totals[eligible].min()
        tied = eligible[totals[eligible] == minimum]
        selected = rng.permutation(tied)[:min(remaining, len(tied))]
        totals[selected] += 1
        remaining -= len(selected)
    return totals


def _layer_extras(n_early, n_late, remaining, cap, rng):
    capacities = np.array([n_early, n_late], dtype=np.int64) * (cap - 1)
    desired = np.full(2, remaining // 2, dtype=np.int64)
    if remaining % 2:
        desired[int(rng.integers(0, 2))] += 1
    extras = np.minimum(desired, capacities)
    overflow = int(remaining - extras.sum())
    # At most two layers; exhausted or empty layer tokens explicitly flow back.
    for side in rng.permutation(2):
        amount = min(overflow, int(capacities[side] - extras[side]))
        extras[side] += amount
        overflow -= amount
    if overflow:
        raise ValueError("insufficient stratum capacity")
    return extras, desired


def _cycle_counts(indices, extras, counts, max_repeat, rng):
    """Repeated shuffled without-replacement cycles; counts differ by <=1."""
    remaining = int(extras)
    while remaining:
        eligible = indices[counts[indices] < max_repeat]
        if not len(eligible):
            raise ValueError("window capacity exhausted")
        selected = rng.permutation(eligible)[:min(remaining, len(eligible))]
        counts[selected] += 1
        remaining -= len(selected)


def build_plan(windows, budget=15536, max_repeat=32, seed=42, early_seconds=1.0):
    """Return a deterministic list of RobotWorldData local indices and a report."""
    _integer(seed, "seed")
    _integer(max_repeat, "max_repeat", 1)
    if not np.isfinite(early_seconds) or early_seconds < 0 or not windows:
        raise ValueError("invalid early threshold or empty windows")
    ordered = sorted(windows, key=lambda row: row["dataset_index"])
    if [w["dataset_index"] for w in ordered] != list(range(len(ordered))):
        raise ValueError("dataset_index must be unique contiguous local indices")
    if len({w["raw_row"] for w in ordered}) != len(ordered):
        raise ValueError("duplicate raw cache rows")
    for row in ordered:
        if not np.isfinite(row["takeover_age_s"]) or row["takeover_age_s"] < 0:
            raise ValueError("nonfinite/negative takeover age")
        if row["category"] not in ("mid_episode_recovery", "teacher_from_start"):
            raise ValueError("unapproved recovery category")
    groups = {}
    for row in ordered:
        groups.setdefault(row["episode_uid"], []).append(row)
    uids = sorted(groups)  # Stable identity order, not input/cache traversal order.
    for uid in uids:
        group = groups[uid]
        for key in ("episode", "task", "category"):
            if len({w[key] for w in group}) != 1:
                raise ValueError(f"inconsistent episode field: {key}")
    rng = np.random.default_rng(seed)
    totals = episode_budgets([len(groups[u]) for u in uids], budget, max_repeat, rng)
    counts = np.ones(len(ordered), dtype=np.int64)
    details = []
    for uid, total in zip(uids, totals):
        group = groups[uid]
        indices = np.array([w["dataset_index"] for w in group], dtype=np.int64)
        is_early = np.array([w["takeover_age_s"] <= early_seconds for w in group])
        early, late = indices[is_early], indices[~is_early]
        remaining = int(total) - len(indices)
        extras, desired = _layer_extras(len(early), len(late), remaining, max_repeat, rng)
        _cycle_counts(early, extras[0], counts, max_repeat, rng)
        _cycle_counts(late, extras[1], counts, max_repeat, rng)
        details.append(dict(episode_uid=uid, episode=group[0]["episode"], task=group[0]["task"],
                            category=group[0]["category"], unique_windows=len(indices),
                            capacity=len(indices) * max_repeat, exposures=int(total),
                            saturated=int(total) == len(indices) * max_repeat,
                            early_unique=len(early), late_unique=len(late),
                            desired_extra_early=int(desired[0]), desired_extra_late=int(desired[1]),
                            allocated_extra_early=int(extras[0]), allocated_extra_late=int(extras[1]),
                            early_exposures=int(counts[early].sum()),
                            late_exposures=int(counts[late].sum()),
                            layer_reallocation=int(np.abs(extras - desired).sum() // 2),
                            empty_early=not len(early), empty_late=not len(late),
                            min_repeat=int(counts[indices].min()), max_repeat=int(counts[indices].max())))
    if int(counts.sum()) != budget or int(counts.min()) < 1 or int(counts.max()) > max_repeat:
        raise AssertionError("exposure invariant failed")
    plan = np.repeat(np.arange(len(ordered), dtype=np.int64), counts)
    plan = plan[rng.permutation(len(plan))]
    report = dict(seed=seed, budget_pre_ddp=budget, max_repeat_cap=max_repeat,
                  early_seconds=early_seconds, early_boundary="age <= threshold; late > threshold",
                  episode_order=uids, episodes=details, unique_windows=len(ordered),
                  min_repeat=int(counts.min()), max_repeat=int(counts.max()),
                  pre_ddp=summarize(ordered, counts, early_seconds))
    return plan, report


def summarize(windows, counts, early_seconds):
    if len(windows) != len(counts) or (np.asarray(counts) < 0).any():
        raise ValueError("invalid counts")
    tasks, categories, episodes, layers = Counter(), Counter(), Counter(), Counter()
    for row, count in zip(windows, counts):
        count = int(count)
        tasks[row["task"]] += count
        categories[row["category"]] += count
        episodes[row["episode_uid"]] += count
        layers["early" if row["takeover_age_s"] <= early_seconds else "late"] += count
    counts = np.asarray(counts)
    return dict(exposures=int(counts.sum()), unique_covered=int((counts > 0).sum()),
                missing_local_indices=np.flatnonzero(counts == 0).tolist(),
                min_repeat=int(counts.min()), max_repeat=int(counts.max()),
                per_task=dict(tasks), per_category=dict(categories), per_episode=dict(episodes),
                early_late=dict(layers))


def simulate_ddp(plan, windows, base_windows=726631, seed=42, epoch=1,
                 world_size=8, batch_size=2, early_seconds=1.0):
    """Exact current DistributedSampler + DataLoader drop_last index semantics."""
    import torch
    from torch.utils.data import DistributedSampler
    for name, value in (("base_windows", base_windows), ("seed", seed), ("epoch", epoch)):
        _integer(value, name)
    _integer(world_size, "world_size", 1)
    _integer(batch_size, "batch_size", 1)
    plan = np.asarray(plan, dtype=np.int64)
    if not len(plan) or (plan < 0).any() or (plan >= len(windows)).any():
        raise ValueError("invalid plan")
    counts = np.zeros(len(windows), dtype=np.int64)
    length = base_windows + len(plan)
    rank_reports = []
    actual_base = 0
    kept_positions = []
    for rank in range(world_size):
        sampler = DistributedSampler(range(length), num_replicas=world_size, rank=rank,
                                     shuffle=True, seed=seed, drop_last=True)
        sampler.set_epoch(epoch)
        selected = np.asarray(list(sampler), dtype=np.int64)
        sampler_count = len(selected)
        selected = selected[:len(selected) // batch_size * batch_size]
        kept_positions.append(selected)
        teacher_positions = selected[selected >= base_windows] - base_windows
        local = plan[teacher_positions]
        counts += np.bincount(local, minlength=len(windows))
        n_base = int((selected < base_windows).sum())
        actual_base += n_base
        rank_reports.append(dict(rank=rank, sampler_rows=sampler_count, loader_rows=len(selected),
                                 base_rows=n_base, recovery_rows=len(local),
                                 loader_dropped=sampler_count - len(selected)))
    kept = np.concatenate(kept_positions)
    if len(np.unique(kept)) != len(kept):
        raise AssertionError("DDP sampled duplicate exposure positions")
    report = summarize(windows, counts, early_seconds)
    report.update(dict(seed=seed, epoch=epoch, world_size=world_size, batch_size=batch_size,
                       sampler_drop_last=True, loader_drop_last=True, torch_version=torch.__version__,
                       combined_pre_ddp=length, combined_actual=len(kept), dropped_total=length-len(kept),
                       base_pre_ddp=base_windows, base_actual=actual_base,
                       base_dropped=base_windows-actual_base, recovery_dropped=len(plan)-int(counts.sum()),
                       ranks=rank_reports))
    if report["unique_covered"] != len(windows):
        raise ValueError("DDP dropped all exposures of a recovery window")
    return report


def load_cache_windows(cache):
    """Verify accepted-cache identity and derive causal age without image loading."""
    cache = Path(cache).resolve()
    index = cache / "index"
    audit = json.loads((index / "audit.json").read_text())
    if audit["cache_complete_sha256"] != sha(cache / "complete.json"):
        raise ValueError("cache complete hash mismatch")
    if audit["splits"]["train"]["index_sha256"] != sha(index / "train_valid.npy"):
        raise ValueError("valid index hash mismatch")
    valid = np.load(index / "train_valid.npy")
    history = np.load(cache / "train_history.npy", mmap_mode="r")
    episode = np.load(cache / "train_episode.npy", mmap_mode="r")
    entries = json.loads((cache / "train_episodes.json").read_text())
    complete = json.loads((cache / "complete.json").read_text())
    for name in ("train_history.npy", "train_episode.npy", "train_episodes.json"):
        if complete["files"][name] != sha(cache / name):
            raise ValueError(f"cache source hash mismatch: {name}")
    if valid.ndim != 1 or valid.dtype.kind not in "iu" or len(np.unique(valid)) != len(valid):
        raise ValueError("invalid valid row index")
    if not len(valid) or (valid < 0).any() or (valid >= len(episode)).any() or len(history) != len(episode):
        raise ValueError("valid rows out of bounds")
    sources = {int(s["episode"]): s["source_sha256"]
               for s in json.loads((index / "train_episodes_audit.json").read_text())["stats"]}
    heldout = json.loads((cache / "heldout_episodes.json").read_text())
    held_scenes = {Path(e["scene"]).name.split(".")[0] for e in heldout}
    info, evidence = {}, []
    for ep in sorted(set(episode[valid].tolist())):
        if not 0 <= ep < len(entries):
            raise ValueError("episode outside metadata")
        entry = entries[ep]
        root = Path(entry["root"])
        for name, expected in sources[ep].items():
            if sha(root / name) != expected:
                raise ValueError(f"raw episode changed: {root / name}")
        meta = json.loads((root / "metadata.json").read_text())
        if meta.get("partition") != "train" or Path(entry["scene"]).name.split(".")[0] in held_scenes:
            raise ValueError("non-training scene in recovery cache")
        obs = json.loads((root / "observations.json").read_text())
        times = np.array([o["timestamp_s"] for o in obs], dtype=np.float64)
        steps = [int(o["sim_step"]) for o in obs]
        if len(set(steps)) != len(steps) or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise ValueError("invalid time/step sequence")
        step_index = {step: i for i, step in enumerate(steps)}
        takeover = _integer(entry["takeover_step"], "takeover_step")
        if takeover not in step_index:
            raise ValueError("takeover step has no observed timestamp")
        category = entry["category"]
        if (takeover == 0) != (category == "teacher_from_start"):
            raise ValueError("takeover/category mismatch")
        info[ep] = (entry, times, steps, step_index[takeover])
        evidence.append(dict(episode=ep, episode_uid=entry["episode_uid"], root=str(root),
                             takeover_step=takeover, takeover_observation_index=step_index[takeover],
                             source_sha256=sources[ep]))
    windows = []
    for k, row in enumerate(valid.tolist()):
        ep = int(episode[row])
        entry, times, steps, takeover_index = info[ep]
        h = np.asarray(history[row])
        if h.shape != (4,) or (h < 0).any() or (h > h[-1]).any() or h[-1] >= len(times):
            raise ValueError("invalid causal history")
        now = int(h[-1])
        if steps[now] < entry["takeover_step"]:
            raise ValueError("failed WA prefix used as positive window")
        age = float(times[now] - times[takeover_index])
        if age < 0:
            raise ValueError("negative takeover age")
        windows.append(dict(dataset_index=k, raw_row=int(row), episode=ep,
                            episode_uid=entry["episode_uid"], task=entry["task"],
                            category=entry["category"], current_index=now,
                            current_sim_step=steps[now], takeover_step=entry["takeover_step"],
                            takeover_age_s=age))
    provenance = dict(cache=str(cache), complete_sha256=sha(cache / "complete.json"),
                      valid_index_sha256=sha(index / "train_valid.npy"),
                      index_audit_sha256=sha(index / "audit.json"), raw_episode_evidence=evidence,
                      no_data_mutation=True, policy_input_change=False)
    return windows, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", type=int, default=15536)
    parser.add_argument("--max-repeat", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-windows", type=int, default=726631)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    windows, provenance = load_cache_windows(args.cache)
    plan, report = build_plan(windows, args.budget, args.max_repeat, args.seed)
    ddp = simulate_ddp(plan, windows, args.base_windows, args.seed)
    payload = dict(schema="wa_recovery_sampling_capped_v1", status="PLAN_ONLY_NOT_USED_BY_TRAINING",
                   provenance=provenance, plan=report, ddp=ddp, windows=windows,
                   recovery_local_indices=plan.tolist(), source_sha256=sha(__file__),
                   sampling_only=True, architecture_loss_optimizer_controller_unchanged=True)
    # Exclusive artifact creation; never replace an existing manifest.
    with output.open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(output=str(output), output_sha256=sha(output), plan=report, ddp=ddp), indent=2))


if __name__ == "__main__":
    main()
