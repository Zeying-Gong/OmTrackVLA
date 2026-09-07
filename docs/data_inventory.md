# OmTrackVLA data inventory (WP-0)

This document freezes the first auditable inventory of the three external datasets used by the `wam` branch. The source trees are read-only inputs. Counts and schema observations below were checked on `g0014` on 2026-09-07; the matching machine-readable record is `configs/data_inventory.json`.

## Scope and audit policy

| Dataset | Authoritative root | Atomic split unit | Primary use |
|---|---|---|---|
| InternData-N1 | `/h100-2/vln_n1/traj_data` | `group/scene` | Phase 1 geometry and state-transition pretraining |
| SAGE3D extracted | `/data/nfs/share/OmTrackVLA/data/sage3d_extracted` | `run` | Phase 1 identity plus Phase 2/3 person-follow supervision |
| TpT clean v2 | `/data/nfs/share/OmTrackVLA/data/tpt_bench_clean_v2` | `sequence` | Phase 1 identity and Phase 2/3 identity/visibility auxiliary supervision |

The audit checks every metadata record and its expected path. Media integrity uses deterministic, stratified decoding across source groups/modes/cameras/sequences; it deliberately does not decode millions of images blindly. The script never writes below a dataset root and does not generate target crops. InternData-N1 is treated as complete by project-owner confirmation, and its total disk usage is intentionally not recomputed.

Run the reproducible audit with the environment that provides `pyarrow` and OpenCV:

```bash
PY=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
$PY scripts/audit_data_inventory.py \
  --manifest configs/data_inventory.json \
  --output outputs/data_audit/audit.json
```

`--output` must be outside all source roots. Omit it to print JSON only. Use `--dataset sage3d_extracted` (repeatable) for a targeted rerun.

## Summary

| Dataset | Scale | RGB / depth | Identity labels | Robot / target geometry | Trajectory | Real UWB |
|---|---:|---|---|---|---|---|
| InternData-N1 | 12 groups, 3,725 episode-bearing scenes, 196,536 episodes | RGB + depth | none | per-frame camera extrinsic/intrinsic and realized motion | navigation pose/action can form geometric targets; not person-follow expert motion | no |
| SAGE3D | 912 runs, 7,110 indexed / 7,105 accepted episodes, 2,132,276 steps | 2,132,276 RGB + matching depth | projected bbox and visibility | robot pose/yaw, target world position and robot-local target point | 1,975,856 steps with 8-point ego waypoints | no |
| TpT clean v2 | 47 sequences, 141,326 rows/frames | RGB only | bbox, visibility, glass-occlusion state | sparse/interpolated ODOM position and quaternion | none | no |

Storage baselines are 171,095,801,856 allocated bytes (about 160 GiB) for SAGE3D and 17,604,104,192 allocated bytes (about 17 GiB) for TpT clean v2. Both matched the immediately preceding checks. InternData-N1's capacity is intentionally unspecified; its 12-group/196,536-episode completeness is owner-confirmed.

None of the datasets contains real UWB measurements, tag IDs, anchor geometry, LOS/NLOS state, calibrated quality, or device timestamps. Any UWB derived from synchronized robot/target pose must be named `simulated_uwb`, with the generation and noise model recorded separately.

## InternData-N1

### Layout and schema

The layout is `group/scene/{meta,data,videos}`. There are 12 sensor/source groups. The `Directories` column includes five `3dfront_d435i` directories that contain only unindexed depth media; they have no metadata or Parquet episodes and are excluded from all manifests.

| Group | Directories | Episode-bearing scenes | Episodes |
|---|---:|---:|---:|
| `3dfront_d435i` | 522 | 517 | 20,103 |
| `3dfront_zed` | 657 | 657 | 21,697 |
| `gibson_d435i` | 473 | 473 | 42,295 |
| `gibson_zed` | 493 | 493 | 31,575 |
| `hm3d_d435i` | 590 | 590 | 38,853 |
| `hm3d_zed` | 633 | 633 | 24,338 |
| `hssd_d435i` | 121 | 121 | 6,810 |
| `hssd_zed` | 88 | 88 | 3,182 |
| `matterport3d_d435i` | 65 | 65 | 3,370 |
| `matterport3d_zed` | 66 | 66 | 2,841 |
| `replica_d435i` | 11 | 11 | 835 |
| `replica_zed` | 11 | 11 | 637 |

Each episode has a small Parquet table with `index`, `observation.camera_intrinsic` (3×3), `observation.camera_extrinsic` (4×4), and `action` (4×4). RGB/depth PNG sequences live under `videos/chunk-*/observation.images.{rgb,depth}`; optional MP4 encodings and scene point clouds are also present. `meta/info.json` records a nominal 30 fps and per-scene totals.

`episodes.jsonl` and `tasks.jsonl` contain navigation instructions (`sub_instruction`, `revised_sub_instruction`, and summaries). These strings are metadata only and must never enter OmTrackVLA model inputs, features, prompts, targets, sampling weights, or split logic.

### Training and split constraints

- Split by the full `group/scene` key. Never randomly split episode rows or frames; the same scene across sensor variants should be co-located when cross-sensor leakage matters.
- Use RGB/depth, calibrated matrices, and relative pose/action only for geometric or state-transition objectives. These are realized navigation motions, not person-follow expert waypoints.
- There is no target-person bbox, persistent person identity, target pose, or UWB. Do not use it as direct Phase 2 person-follow policy supervision.
- Before metric SE(2) supervision, verify the matrix convention and scale in WP-1; the inventory records fields, not a frozen coordinate conversion.

## SAGE3D extracted

### Layout, acceptance, and schema

The root `index.json` lists the 7,110 episode directories as `run/mode/episode/camera`. Each contains RGB and depth PNGs, source episode/info JSON, `camera_info.json`, `derived.json`, `quality.json`, videos, and diagnostic reference crops.

The root index is a catalog rather than an acceptance list: it contains 7,110 paths, of which 7,105 are accepted and five have `quality.status == "rejected"` with no `_ACCEPTED` marker. The full tree contains exactly 7,105 `_ACCEPTED` markers and every marker path is indexed. For training manifests, an episode is canonically accepted only when all three conditions hold:

1. its relative path is present in the root `index.json`;
2. the episode directory contains `_ACCEPTED`;
3. `quality.json.status == "accepted"`.

The source episode's `success` value describes the simulator/task outcome and is not the extraction acceptance rule. Seven indexed episodes have `success=false`: five are the rejected entries, while two are canonically accepted. Therefore canonical acceptance and source success disagree for two episodes. No loader should silently substitute source `success` for the rule above. This full-tree audit corrects the earlier preliminary claim of eight marker conflicts.

`derived.json.steps` provides `robot_pos`, `robot_yaw`, `target_pos`, `target_local`, `target_dist`, projected `bbox`, `visible`, collision/facing fields, and `waypoints_ego`. Camera metadata declares a pinhole camera, ROS axes, 640×480 resolution for the inspected D435i sample, intrinsics, and a robot-to-camera translation. Projection metadata explicitly warns that bbox values are world-to-camera projections and require validation before training.

Waypoints use horizon 8 and stride 3. There are 1,975,856 steps with a complete 8-point waypoint and 156,420 terminal steps without one (22 terminal steps per episode). Missing terminal waypoints are expected invalid labels, not corruption.

### Training and split constraints

- Keep every episode from the same run in one split. A `scene/run` assignment table must be frozen in NEXT-015 before training; episode-level random splits are forbidden.
- `bbox`, `visible`, target pose, and target-local position are label/diagnostic fields. Only the initialization bbox is a model input; later bboxes must stay out of the input domain.
- Robot and target coordinates are suitable for generating explicitly labelled `simulated_uwb`, never for claiming real-UWB coverage.
- Confirm ROS camera axes, base axes, yaw sign, waypoint interval, and camera-to-base extrinsics in WP-1 before merging with other trajectory sources.
- Ignore `_target_crops`, `_target_refs`, `track_object.jpg`, and rendered videos as generated diagnostics; do not treat them as independent samples.

## TpT clean v2

### Layout and schema

Each sequence contains `frames.parquet`, `rgb_frames/frame_XXXXXX.jpg`, `meta.json`, and a natural-language `desc.txt`. Generated `_target_crops` and `_target_refs` may also exist and are excluded from source-sample counts.

The Parquet schema is:

```text
video_idx, gt_idx, gt_ts_ns, vid_pts_ms,
bbox_source[4], bbox_qv[4], is_exist, is_behind_glass, interpolated,
odom_pos[3], odom_quat_xyzw[4], odom_ts_s
```

`bbox_source` and `bbox_qv` are `[x, y, width, height]`; convert to `[x0, y0, x1, y1]` at the data-contract boundary. `is_behind_glass` uses `-1/0/1`, with both signs treated as the glass condition. The inventory found 117,611 visible rows and 24,170 glass-condition rows. All 141,326 Parquet rows map to a non-empty RGB frame.

### Unresolved clock contract

Across sequences, elapsed `gt_ts_ns`/`odom_ts_s` time divided by elapsed `vid_pts_ms` time ranges from 4.2553 to 4.8668, with median 4.4752. This is too large to treat as timestamp jitter. Until the data owner confirms whether it is a playback, extraction, annotation, or clock-domain mapping, do not infer physical velocity, choose a waypoint horizon, or fuse TpT ODOM with another source using these clocks.

### Training and split constraints

- Split only by sequence. Never randomly split frames from one sequence across train/validation/test.
- TpT can supervise initialization identity, per-frame matching/visibility, occlusion robustness, and ODOM-aware auxiliary tasks. It has no person-follow expert waypoint, target 3D pose, or UWB.
- `desc.txt` is excluded from all model inputs and targets. Per-frame bbox, `is_exist`, and glass state are labels only.
- `meta.json` notes that quick-view bbox scaling was inferred; visually validate any sequence admitted to training and preserve the per-sequence geometry report.

## Quality status and remaining gates

WP-0 establishes what exists and makes the audit reproducible. It does not freeze the WP-1 data contract or train/val/test manifests.

The following remain blocking for formal training:

- resolve TpT clock semantics;
- freeze camera/base/world transforms, yaw convention, metric scale, waypoint interval, and missing-label representation;
- create run/scene/sequence-level train, val, `viz_val`, and `test_locked` manifests with overlap tests;
- validate SAGE3D projected bboxes on the frozen training subset;
- acquire real UWB logs and calibration, or explicitly limit experiments to `simulated_uwb`.
