"""Pure, independently pinned three-source exposure planning; never training release.

The caller supplies audited identities. No files, models, caches, or datasets are
opened here. Old positions retain their original multiplicity and omission;
new recovery identities are diagnostic metadata and must not enter the policy.
"""
from collections import defaultdict
import hashlib
import heapq
import json
import math
import re

import numpy as np

SCHEMA = "failure_state_sampling_candidate_v1"
SOURCES = {"base": 0, "teacher": 1, "recovery": 2}
EARLY_SECONDS = 1.0
EARLY_TOLERANCE = 1e-8
PIN_KEYS = {"old_plan", "old_actual_exposure", "recovery_admission", "dedup_report"}
WINDOW_FIELDS = {"dataset_index", "raw_row", "episode", "task", "key", "teacher",
                 "current_index", "takeover_step", "current_age_s", "valid",
                 "exact_duplicate"}
ARRAY_KEYS = {"source_codes", "dataset_indices", "old_plan_positions",
              "recovery_raw_rows", "old_base_counts", "old_teacher_counts",
              "recovery_counts"}


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name}: integer required")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name}: must be >= {minimum}")
    return value


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _array_sha(value):
    value = np.ascontiguousarray(value)
    header = json.dumps(dict(dtype=value.dtype.str, shape=list(value.shape)),
                        sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(header + b"\n" + value.tobytes()).hexdigest()


def _old_inputs(position_sources, position_counts, expected_dropped):
    if not isinstance(position_sources, (list, tuple)) or not position_sources:
        raise ValueError("old positions must be a nonempty ordered sequence")
    codes, indices = [], []
    for row in position_sources:
        if not isinstance(row, (tuple, list)) or len(row) != 2 or row[0] not in ("base", "teacher"):
            raise ValueError("old position must be (base|teacher, dataset_index)")
        codes.append(SOURCES[row[0]])
        indices.append(_integer(row[1], "old dataset_index"))
    counts = np.asarray(position_counts)
    if counts.ndim != 1 or len(counts) != len(codes) or counts.dtype.kind not in "iu":
        raise ValueError("one integer actual exposure count per old plan position required")
    if not np.isin(counts, [0, 1]).all():
        raise ValueError("old recorded positions must be exactly consumed once or omitted")
    expected = [_integer(i, "expected dropped position") for i in expected_dropped]
    if expected != sorted(set(expected)) or any(i >= len(codes) for i in expected):
        raise ValueError("invalid expected dropped positions")
    if np.flatnonzero(counts == 0).tolist() != expected:
        raise ValueError("actual old omissions differ from explicit expected omissions")
    keep = np.flatnonzero(counts == 1).astype(np.int64)
    if not len(keep):
        raise ValueError("no executed old position")
    return np.asarray(codes, dtype=np.int8), np.asarray(indices, dtype=np.int64), counts, keep


def _windows(windows):
    if not isinstance(windows, (list, tuple)) or not windows:
        raise ValueError("explicit recovery window inventory required")
    normalized = []
    identities, raw_rows, currents = set(), set(), set()
    episodes = {}
    for value in windows:
        if not isinstance(value, dict) or set(value) != WINDOW_FIELDS:
            raise ValueError("unexpected recovery window schema")
        row = dict(value)
        for key in ("dataset_index", "raw_row", "current_index", "takeover_step"):
            row[key] = _integer(row[key], key)
        if type(row["valid"]) is not bool or type(row["exact_duplicate"]) is not bool:
            raise ValueError("valid and exact_duplicate must be explicit booleans")
        if row["task"] not in ("stt", "dt", "at") or row["teacher"] not in ("lightnav", "oracle"):
            raise ValueError("unknown recovery task or teacher")
        if not isinstance(row["key"], str) or not row["key"] or row["episode"] != row["task"] + ":" + row["key"]:
            raise ValueError("episode identity must equal task:key")
        age = row["current_age_s"]
        if isinstance(age, (bool, np.bool_)) or not isinstance(age, (int, float, np.number)):
            raise ValueError("current_age_s must be a finite real number")
        if not math.isfinite(float(age)) or age < 0 or row["current_index"] < row["takeover_step"]:
            raise ValueError("noncausal or invalid takeover-relative age")
        if (row["current_index"] == row["takeover_step"]) != (float(age) == 0.0):
            raise ValueError("zero age must be exactly the takeover observation")
        row["current_age_s"] = float(age)
        current = (row["episode"], row["current_index"])
        if row["dataset_index"] in identities or row["raw_row"] in raw_rows or current in currents:
            raise ValueError("duplicate recovery dataset/raw/episode-current identity")
        identities.add(row["dataset_index"]); raw_rows.add(row["raw_row"]); currents.add(current)
        identity = (row["task"], row["key"], row["teacher"], row["takeover_step"])
        if row["episode"] in episodes and episodes[row["episode"]] != identity:
            raise ValueError("inconsistent identity within recovery episode")
        episodes[row["episode"]] = identity
        normalized.append(row)
    ordered = sorted(normalized, key=lambda row: row["dataset_index"])
    grouped = defaultdict(list)
    for row in ordered:
        grouped[row["episode"]].append(row)
    for group in grouped.values():
        time_order = sorted(group, key=lambda row: row["current_index"])
        if any(b["current_age_s"] <= a["current_age_s"] for a, b in zip(time_order, time_order[1:])):
            raise ValueError("takeover-relative time must increase with observation index")
    return ordered


def _episode_budgets(sizes, budget, window_cap, episode_cap, rng):
    floors = np.asarray(sizes, dtype=np.int64)
    capacities = np.minimum(floors * window_cap, episode_cap)
    if (floors > capacities).any() or not int(floors.sum()) <= budget <= int(capacities.sum()):
        raise ValueError("budget outside unique-coverage/episode-window capacity bounds")
    totals = floors.copy()
    # Seed controls the order of equal-water-level episodes, not their capacities.
    tie_order = np.empty(len(floors), dtype=np.int64)
    tie_order[rng.permutation(len(floors))] = np.arange(len(floors))
    heap = [(int(totals[i]), int(tie_order[i]), i) for i in range(len(floors)) if totals[i] < capacities[i]]
    heapq.heapify(heap)
    for _ in range(budget - int(floors.sum())):
        level, tie, i = heapq.heappop(heap)
        totals[i] += 1
        if totals[i] < capacities[i]:
            heapq.heappush(heap, (level + 1, tie, i))
    return totals, capacities


def _layer_extras(early_size, late_size, remaining, cap, rng):
    capacities = np.asarray([early_size, late_size], dtype=np.int64) * (cap - 1)
    desired = np.full(2, remaining // 2, dtype=np.int64)
    if remaining % 2:
        desired[int(rng.integers(2))] += 1
    extras = np.minimum(desired, capacities)
    overflow = remaining - int(extras.sum())
    for side in rng.permutation(2):
        amount = min(overflow, int(capacities[side] - extras[side]))
        extras[side] += amount
        overflow -= amount
    if overflow:
        raise AssertionError("stratum capacity allocation failed")
    return desired, extras


def _cycle_counts(counts, indices, extras, rng):
    if not len(indices):
        if extras:
            raise AssertionError("empty layer cannot receive exposure")
        return
    full, tail = divmod(int(extras), len(indices))
    counts[indices] += full  # Complete without-replacement cycles.
    counts[rng.permutation(indices)[:tail]] += 1


def build_plan(old_position_sources, old_position_counts, recovery_windows, *,
               source_pins, recovery_budget=49152, seed=42,
               expected_dropped_positions=(1151263,), window_cap=16,
               episode_cap=1024, world_size=8, batch_size=2):
    """Construct an in-memory candidate; inputs are caller-audited identities.

    Old source tuples are the original plan.locate(position) output, including
    distinct extra positions that happen to refer to the same teacher window.
    Recovery dataset indices may be sparse; raw_row is the full cache row, not a
    loader index. Invalid or old-exact-duplicate rows receive exactly zero.
    """
    for name, value in (("recovery_budget", recovery_budget), ("seed", seed)):
        _integer(value, name)
    for name, value in (("window_cap", window_cap), ("episode_cap", episode_cap),
                        ("world_size", world_size), ("batch_size", batch_size)):
        _integer(value, name, 1)
    if window_cap > 16 or episode_cap > 1024:
        raise ValueError("hard limits may not exceed 16/window or 1024/episode")
    expected_dropped_positions = tuple(
        _integer(i, "expected dropped position") for i in expected_dropped_positions)
    if not isinstance(source_pins, dict) or set(source_pins) != PIN_KEYS or any(
            not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{64}", v) is None for v in source_pins.values()):
        raise ValueError("all four explicit SHA256 source pins required")
    old_codes, old_indices, old_counts, keep = _old_inputs(
        old_position_sources, old_position_counts, expected_dropped_positions)
    ordered = _windows(recovery_windows)
    eligible = np.asarray([row["valid"] and not row["exact_duplicate"] for row in ordered])
    if not eligible.any():
        raise ValueError("no deduplicated valid recovery window")
    if (len(keep) + recovery_budget) % (world_size * batch_size):
        raise ValueError("combined plan would lose positions through sampler/loader drop_last")
    groups = defaultdict(list)
    for i, row in enumerate(ordered):
        if eligible[i]:
            groups[row["episode"]].append(i)
    uids = sorted(groups)
    rng = np.random.Generator(np.random.PCG64(seed))
    totals, capacities = _episode_budgets([len(groups[uid]) for uid in uids],
        recovery_budget, window_cap, episode_cap, rng)
    counts = eligible.astype(np.int64)
    details = []
    for uid, total, capacity in zip(uids, totals, capacities):
        indices = np.asarray(groups[uid], dtype=np.int64)
        early = np.asarray([ordered[i]["current_age_s"] <= EARLY_SECONDS + EARLY_TOLERANCE for i in indices])
        layers = (indices[early], indices[~early])
        desired, extras = _layer_extras(len(layers[0]), len(layers[1]),
                                      int(total) - len(indices), window_cap, rng)
        for selected, extra in zip(layers, extras):
            _cycle_counts(counts, selected, extra, rng)
        details.append(dict(episode=uid, unique_windows=len(indices), exposures=int(total),
            capacity=int(capacity), saturated=int(total) == int(capacity),
            early_unique=len(layers[0]), late_unique=len(layers[1]),
            desired_extra_early=int(desired[0]), desired_extra_late=int(desired[1]),
            allocated_extra_early=int(extras[0]), allocated_extra_late=int(extras[1]),
            early_exposures=int(counts[layers[0]].sum()), late_exposures=int(counts[layers[1]].sum()),
            reallocated_extras=int(np.abs(extras - desired).sum() // 2)))
    if int(counts.sum()) != recovery_budget or (counts[eligible] < 1).any() or (counts > window_cap).any() or counts[~eligible].any():
        raise AssertionError("recovery exposure invariant failed")
    selected = np.repeat(np.arange(len(ordered), dtype=np.int64), counts)
    selected = selected[rng.permutation(len(selected))]
    recovery_indices = np.asarray([row["dataset_index"] for row in ordered], dtype=np.int64)
    recovery_raw = np.asarray([row["raw_row"] for row in ordered], dtype=np.int64)
    arrays = dict(
        source_codes=np.concatenate((old_codes[keep], np.full(recovery_budget, SOURCES["recovery"], dtype=np.int8))),
        dataset_indices=np.concatenate((old_indices[keep], recovery_indices[selected])),
        old_plan_positions=np.concatenate((keep, np.full(recovery_budget, -1, dtype=np.int64))),
        recovery_raw_rows=np.concatenate((np.full(len(keep), -1, dtype=np.int64), recovery_raw[selected])),
        old_base_counts=np.bincount(old_indices[keep][old_codes[keep] == SOURCES["base"]],
            minlength=int(old_indices[old_codes == SOURCES["base"]].max()) + 1
            if (old_codes == SOURCES["base"]).any() else 0),
        old_teacher_counts=np.bincount(old_indices[keep][old_codes[keep] == SOURCES["teacher"]],
            minlength=int(old_indices[old_codes == SOURCES["teacher"]].max()) + 1
            if (old_codes == SOURCES["teacher"]).any() else 0),
        recovery_counts=counts)
    for array in arrays.values():
        array.flags.writeable = False
    metadata = dict(schema=SCHEMA, source_pins=dict(source_pins), sources=SOURCES.copy(),
        seed=int(seed), sampler_epoch=1, rng="numpy.PCG64", window_cap=int(window_cap), episode_cap=int(episode_cap),
        early_seconds=EARLY_SECONDS, early_tolerance_s=EARLY_TOLERANCE,
        age_definition="observation_time[current_index] - observation_time[takeover_step]",
        early_rule="current_age_s <= 1.0 + 1e-8; remaining extra budget balanced, not total exposure",
        budget=int(recovery_budget), old_pre_ddp_positions=len(old_codes), old_retained_positions=len(keep),
        old_dropped_positions=list(expected_dropped_positions), total_positions=len(keep) + int(recovery_budget),
        world_size=int(world_size), batch_size=int(batch_size), episodes=details,
        recovery_windows=ordered, recovery_unique=int(eligible.sum()),
        zero_exposure_episodes=sorted({row["episode"] for row in ordered} - set(uids)),
        old_input_sha256=_canonical(dict(source_codes=_array_sha(old_codes),
            dataset_indices=_array_sha(old_indices), actual_counts=_array_sha(old_counts))),
        training_released=False, source_files_verified_here=False, training_executed=False,
        scope="Exposure-plan candidate only; not cache admission, new model result, or training authorization.",
        limits=["Same old exposure counts, not identical ordering/augmentation/optimizer trajectory.",
                "Caller must prove effective-mask eligibility and exact dedup against pinned artifacts.",
                "Zero-window episodes absent from input cannot be inventoried by this pure function."])
    hashes = {name: _array_sha(value) for name, value in arrays.items()}
    return dict(metadata=metadata, arrays=arrays, array_sha256=hashes,
                plan_sha256=_canonical(dict(metadata=metadata, array_sha256=hashes)))


def simulate_exposure(plan, *, seed=42, epoch=1, world_size=8, batch_size=2):
    """Run real torch sampler AND loader, comparing exact per-source-window counts."""
    import torch
    from torch.utils.data import DataLoader, DistributedSampler
    for name, value in (("seed", seed), ("epoch", epoch)):
        _integer(value, name)
    for name, value in (("world_size", world_size), ("batch_size", batch_size)):
        _integer(value, name, 1)
    if set(plan) != {"metadata", "arrays", "array_sha256", "plan_sha256"} or set(plan["arrays"]) != ARRAY_KEYS:
        raise ValueError("unexpected plan schema")
    metadata, arrays = plan["metadata"], plan["arrays"]
    if metadata["schema"] != SCHEMA or metadata["training_released"] is not False:
        raise ValueError("not an unreleased independent candidate")
    hashes = {name: _array_sha(value) for name, value in arrays.items()}
    if hashes != plan["array_sha256"] or _canonical(dict(metadata=metadata, array_sha256=hashes)) != plan["plan_sha256"]:
        raise ValueError("plan mutation/hash mismatch")
    if (seed, world_size, batch_size) != (metadata["seed"], metadata["world_size"], metadata["batch_size"]):
        raise ValueError("sampler configuration differs from pinned plan")
    if epoch != metadata["sampler_epoch"]:
        raise ValueError("sampler epoch differs from pinned epoch1")
    length = metadata["total_positions"]
    seen = np.zeros(length, dtype=np.int64)
    ranks = []
    for rank in range(world_size):
        sampler = DistributedSampler(range(length), num_replicas=world_size, rank=rank,
                                     shuffle=True, seed=seed, drop_last=True)
        sampler.set_epoch(epoch)
        loader = DataLoader(range(length), batch_size=batch_size, sampler=sampler,
                            drop_last=True, num_workers=0)
        consumed = 0
        source_counts = np.zeros(3, dtype=np.int64)
        for batch in loader:
            positions = batch.numpy()
            np.add.at(seen, positions, 1)
            source_counts += np.bincount(arrays["source_codes"][positions], minlength=3)
            consumed += len(positions)
        ranks.append(dict(rank=rank, sampler_positions=len(sampler), loader_positions=consumed,
                          batches=len(loader), sources={name: int(source_counts[code]) for name, code in SOURCES.items()}))
    if not np.all(seen == 1):
        raise ValueError("sampler/loader omitted or repeated a plan position")
    source, indices = arrays["source_codes"], arrays["dataset_indices"]
    comparisons = {}
    for name in ("base", "teacher"):
        select = source == SOURCES[name]
        actual = np.bincount(indices[select], weights=seen[select],
                             minlength=len(arrays["old_" + name + "_counts"])).astype(np.int64)
        expected = arrays["old_" + name + "_counts"]
        if not np.array_equal(actual, expected):
            raise ValueError("old per-window exposure changed: " + name)
        comparisons[name] = dict(exposures=int(actual.sum()), unique=int((actual > 0).sum()),
                                 per_window_counts_sha256=_array_sha(actual), exact=True)
    expected_recovery = arrays["recovery_counts"]
    local_lookup = {row["dataset_index"]: i for i, row in enumerate(metadata["recovery_windows"])}
    actual_recovery = np.zeros(len(expected_recovery), dtype=np.int64)
    selected = np.flatnonzero(source == SOURCES["recovery"])
    for position in selected:
        actual_recovery[local_lookup[int(indices[position])]] += seen[position]
    if not np.array_equal(actual_recovery, expected_recovery):
        raise ValueError("recovery per-window exposure mismatch")
    return dict(status="EXACT_INDEX_SIMULATION_PASS_NOT_TRAINING", plan_sha256=plan["plan_sha256"],
        seed=seed, epoch=epoch, world_size=world_size, batch_size=batch_size,
        sampler_drop_last=True, loader_drop_last=True, actual_total=int(seen.sum()), dropped=0,
        old_counts=comparisons, recovery_exposures=int(actual_recovery.sum()),
        recovery_unique=int((actual_recovery > 0).sum()), recovery_max_repeat=int(actual_recovery.max()),
        recovery_counts_sha256=_array_sha(actual_recovery), ranks=ranks,
        training_executed=False, training_released=False, torch_version=torch.__version__)
