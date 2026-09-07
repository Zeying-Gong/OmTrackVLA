# OmTrackVLA data contract (WP-1)

This document freezes the canonical record boundary for target-person identity supervision and person-following policy supervision on the `wam` branch. The matching machine-readable contract is [`configs/data_contract.json`](../configs/data_contract.json); [`scripts/validate_data_contract.py`](../scripts/validate_data_contract.py) enforces it without modifying source data.

“Frozen” means that loaders must convert source-specific data into this representation before collation. It does not mean every source is ready for every objective. In particular, SAGE3D policy trajectories, TpT physical timing, and real-UWB coverage remain blocked by the gates listed below.

## Domain boundary

A serialized clip has five distinct domains:

| Domain | Purpose | May reach the model? |
|---|---|---|
| `model_inputs` | RGB history, one initialization event, current UWB measurement and validity masks | yes |
| `routing_metadata` | select the specified UWB stream and its calibration/noise specification | no; strip before tensorization |
| `supervision.expert_trajectory` | policy target in the anchor base frame | loss/metric only |
| `supervision.auxiliary_labels` | identity, visibility, bbox, target geometry and UWB-error labels | loss/metric only |
| `provenance` | source record and transform/clock/generation specifications | no |

Natural language is prohibited from inputs, targets, sampling weights, and split logic. Keys such as `instruction`, `desc`, `prompt`, later-frame bbox/visibility, target track ID, and target ground-truth pose are rejected recursively below `model_inputs`.

`target_tag_id` is routing metadata: it selects the desired device stream but is not a numeric identity feature. A data loader must remove the whole `routing_metadata` object before calling the model.

## Canonical coordinates

The anchor time is the final RGB-history timestamp, written `t`. Define `B_t` as the robot base frame at that instant:

- origin: robot base origin;
- `+x`: forward;
- `+y`: left;
- `+z`: up;
- linear unit: metre;
- positive yaw: counter-clockwise about `+z`, in radians.

The notation `^A T_B` maps coordinates from frame `B` into frame `A`. Adapters must name the versioned transform specification used to reach `B_t`; matrices with an unconfirmed w2c/c2w convention cannot enter policy records.

Target and UWB positions are `[x_forward_m, y_left_m]` in `B_t`. Expert waypoints are eight absolute XY positions in `B_t`, not increments. Point zero represents the anchor pose at offset zero; policy loss and reported displacement errors exclude that trivial point by default. Remaining time offsets must be strictly increasing and expressed in seconds. Invalid terminal points serialize as `null` with `valid_mask=false`; NaN and infinity are forbidden.

Bboxes use normalized `[x0, y0, x1, y1]`, with image `x` right and `y` down. Source boxes are clipped to the image before normalization and must satisfy `0 <= x0 < x1 <= 1` and `0 <= y0 < y1 <= 1`. TpT `bbox_qv` therefore requires pixel `xywh -> xyxy -> clip -> normalize`; SAGE3D requires pixel `xyxy -> clip -> normalize`.

## Canonical time

All serialized timestamps are non-negative integer nanoseconds in one episode-monotonic clock domain after adapter synchronization. Original device timestamps may be retained outside the model record, but the canonical fields must obey:

```text
rgb_history[i].timestamp_ns < rgb_history[i+1].timestamp_ns
anchor_timestamp_ns = rgb_history[-1].timestamp_ns
uwb.source_timestamp_ns <= uwb.receive_timestamp_ns <= anchor_timestamp_ns
uwb.age_s = (anchor_timestamp_ns - uwb.source_timestamp_ns) / 1e9
```

Every policy record names a `clock_spec_id`. A frame index, nominal FPS, or unexplained timestamp ratio is not a clock specification. This intentionally blocks SAGE3D policy use until its simulator step interval is confirmed and blocks physical velocity or ODOM fusion from TpT until its roughly 4.48× clock-domain ratio is resolved.

## Clip record

The top-level shape is:

```text
{
  schema_version: 1,
  sample_id,
  sample_role: policy | identity_auxiliary,
  source: {
    dataset_id, split_unit_id, episode_id, anchor_index, adapter_version
  },
  model_inputs: {
    condition_mode,
    anchor_timestamp_ns,
    rgb_sensor_valid,
    rgb_history: [{rgb_path, timestamp_ns, valid}, ...],
    visual_initialization: {valid, rgb_path, bbox_xyxy_norm, timestamp_ns},
    uwb_target: {
      valid, measurement_kind,
      relative_position_base_xy_m,
      covariance_base_xy_m2, quality_01,
      source_timestamp_ns, receive_timestamp_ns, age_s, los_state
    }
  },
  routing_metadata: {target_tag_id, calibration_id, simulation_spec_id},
  supervision: {
    expert_trajectory: {
      valid, waypoints_base_xy_m[8], time_offsets_s[8] | null,
      valid_mask[8]
    },
    auxiliary_labels: {
      target_track_id, target_bbox_xyxy_norm, target_visible,
      target_position_base_xy_m, occlusion_state, uwb_error_base_xy_m
    },
    safety: {stop_required, reason}
  },
  provenance: {
    source_record, transform_spec_id, clock_spec_id, generation_spec_id
  }
}
```

All objects use the exact keys above. Paths are POSIX-style, relative to the declared dataset root, contain no `..`, and may optionally be checked on disk by the validator.

### RGB and the one-time initialization event

`rgb_history` is chronological and contains at least two slots. A valid slot has a non-empty relative `rgb_path`; an invalid/dropped slot has `rgb_path=null`. `rgb_sensor_valid` describes the sensor at the anchor: when true the anchor slot is valid; when false the anchor slot is invalid, earlier valid history is retained, and the sample must request a safety stop for `rgb_sensor_failure`.

For normal visual initialization, the first history frame is the one externally annotated frame. `visual_initialization.rgb_path` and `timestamp_ns` must equal that first frame and it alone carries `bbox_xyxy_norm`. This is an initialization event, not a per-frame input. A stored reference may be repeated across independently sampled clips for reproducibility, but a sequential rollout consumes it only at the clip's first history index and thereafter uses internal identity memory.

For UWB-only cold start, all visual-initialization payload fields are `null` and `valid=false`. Later target boxes and track IDs remain in `auxiliary_labels`; they supervise automatic binding but never enter `model_inputs`.

### UWB

`measurement_kind` is `real_uwb`, `simulated_uwb`, or `none`. A valid measurement requires a finite base-frame XY position, synchronized source/receive timestamps, consistent non-negative age, LOS state (`los`, `nlos`, or `unknown`), and at least one uncertainty representation:

- a symmetric positive-semidefinite 2×2 covariance in square metres; or
- a scalar `quality_01` in `[0, 1]`.

Invalid measurements carry `null` for the entire measurement payload. The stream kind and routing metadata may remain populated so a missing packet is distinguishable from a dataset with no UWB stream. `real_uwb` requires a versioned `calibration_id`; `simulated_uwb` requires a versioned `simulation_spec_id` and must never be reported as real-UWB coverage.

### Expert and auxiliary supervision

`sample_role=policy` requires a valid eight-point expert trajectory with at least two valid points, plus non-null transform and clock specifications. `identity_auxiliary` requires the trajectory to be wholly invalid and masked; it can contribute only identity/visibility losses.

A visible target requires a valid normalized auxiliary bbox and `occlusion_state=visible`. A non-visible or unknown target has a null bbox. Target track IDs are scoped to the declared split unit; they are not assumed globally stable.

Missing values always use `null` plus an explicit validity flag/mask. Loaders may replace null numeric values with zero only after constructing the mask. NaN, infinity, zero-filled fake boxes, sentinel coordinates, and magic timestamps are invalid serialization.

## Condition modes and safety invariants

| `condition_mode` | Visual init | Current UWB | RGB sensor | Required behavior |
|---|---:|---:|---:|---|
| `visual_uwb` | valid | valid | valid | normal fused following |
| `visual_only` | valid | invalid | valid | follow from RGB history and identity memory |
| `uwb_only` | invalid | valid | valid | conservative approach and automatic visual binding |
| `safe_stop` | any | any | possibly failed | decelerate/stop for no reliable target, RGB failure, or ambiguous binding |

The first three modes require `safety.stop_required=false` and reason `none`. `safe_stop` requires `stop_required=true` and a non-`none` reason. An RGB sensor failure always maps to `safe_stop`, even if UWB is valid, until an independent verified obstacle-safety stack is explicitly added to the contract.

## Source admission gates

| Source | Currently admitted role | Blocked use |
|---|---|---|
| InternData-N1 | none in this person-following contract | Phase 1 geometry remains valid elsewhere; it has no target identity or person-follow expert |
| SAGE3D extracted | `identity_auxiliary` | `policy` until simulator step timing, base/camera transform and projected bbox validation are frozen |
| TpT clean v2 | `identity_auxiliary` | policy/physical velocity; no expert path and clock semantics unresolved |
| Habitat simulation | policy or identity auxiliary | records still require named transform, clock and generation specifications |
| real robot | policy or identity auxiliary | records require named transform, clock and generation specifications; real UWB also requires calibration |

Admission is enforced by `source_adapters.*.eligible_roles`. Clearing a gate requires evidence plus a reviewed contract update; a loader flag must not silently bypass it.

## Validation

Validate the contract alone:

```bash
python scripts/validate_data_contract.py \
  --contract configs/data_contract.json
```

Validate JSON or JSONL clip records, optionally checking only the referenced files rather than scanning a dataset:

```bash
python scripts/validate_data_contract.py \
  --contract configs/data_contract.json \
  --samples path/to/clips.jsonl \
  --data-root sage3d_extracted=/data/nfs/share/OmTrackVLA/data/sage3d_extracted
```

The validator is read-only. It reports the number of accepted records by dataset, role, and condition mode, and fails on the first contract violation with the file/record location.

## Remaining work after WP-1

This contract deliberately leaves evidence-dependent items visible:

- NEXT-015 must freeze disjoint train/val/`viz_val`/`test_locked` manifests by run, sequence, scene and identity as applicable.
- SAGE3D needs a reviewed adapter specification before its waypoints can supervise the policy.
- TpT clock semantics must be resolved before ODOM-derived motion or velocity is used.
- Real UWB logs, anchor geometry, device calibration, latency and LOS/NLOS semantics are still absent.
- Failure-state label semantics and the safe-stop/recovery state machine continue in NEXT-014.
