# SAGE3D projected bbox audit

## Conclusion

The `bbox` and `visible` fields currently stored in SAGE3D `derived.json` are not admissible as Phase 1 identity supervision without correction. The main defect is a deterministic horizontal sign error in the extraction-time world-to-camera projection. A second defect marks targets as visible whenever their sampled 3D points are in front of the robot, without requiring a non-empty image intersection or checking depth occlusion.

The source dataset remains read-only. No source `derived.json`, RGB, or depth file was changed by this audit.

## Audit setup

- Date: 2026-09-08
- Script: `scripts/audit_sage3d_bboxes.py`
- Source: `/data/nfs/share/OmTrackVLA/data/sage3d_extracted`
- Sampling: 384 visible-labelled frames from 128 episodes, three temporal positions per episode, round-robin across all 18 `mode/camera` strata
- Independent detector: torchvision Faster R-CNN MobileNet V3 320 FPN
- Detector weight SHA-256: `907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5`
- Person score threshold: 0.30
- Agreement thresholds: strong when best person IoU is at least 0.50; disagreement when it is below 0.20
- H100 artifacts: `results/wp2_sage_bbox_audit_v2/` in the shared WAM worktree

The detector is an independent review instrument. Its boxes are not treated as automatic truth, and no detector output is fed to the Phase 1 model.

## Results

Five of 384 projected source boxes have zero area after clipping to the RGB image. Among the 379 remaining source boxes, the detector found at least one person in 331 frames (87.34%). Within those 331 detected frames:

| Result | Frames | Rate |
|---|---:|---:|
| Strong agreement, IoU at least 0.50 | 176 | 53.17% |
| Partial agreement, IoU 0.20 to 0.50 | 86 | 25.98% |
| Detector disagreement, IoU below 0.20 | 69 | 20.85% |

The temporal split is diagnostic:

| Position | Detected valid frames | Median best IoU | Strong agreement | Disagreement |
|---|---:|---:|---:|---:|
| Initial visible frame | 107 | 0.635 | 76.64% | 1.87% |
| Middle visible frame | 111 | 0.386 | 36.94% | 32.43% |
| Final visible frame | 113 | 0.468 | 46.90% | 27.43% |

Initial frames tend to place the target near the optical center, which hides a horizontal sign error. Later frames expose the error as the robot and target acquire lateral separation.

## Root cause

The source extractor at `/data/nfs/share/OmTrackVLA/tools/extract_sage3d.py` has SHA-256 `57e87cd09fb0466672e55f70f9d461c5bee3eeda32f746074dd082c18769918c`. Its `project_target` function constructs:

```python
f = [cos(yaw), sin(yaw), 0]
right = [-sin(yaw), cos(yaw), 0]
u = cx + fx * dot(target - camera, right) / forward
```

For the stated robot convention, `[-sin(yaw), cos(yaw)]` is the left vector, not the right vector. Positive left must decrease image `u`; using it with a positive image-x projection mirrors the bbox horizontally. The correction is equivalently either:

```python
right = [sin(yaw), -cos(yaw), 0]
```

or retaining `left = [-sin(yaw), cos(yaw), 0]` and using `u = cx - fx * left / forward`.

As a counterfactual check, horizontally mirroring the stored source boxes changed the detector comparison as follows:

| Metric over 331 detected valid frames | Stored source box | Horizontal-mirror counterfactual |
|---|---:|---:|
| Median best IoU | 0.533 | 0.638 |
| Strong-agreement frames | 176 | 247 |
| Disagreement frames | 69 | 14 |

This large, directional improvement plus the review montages establishes the sign defect. It does not establish that a simple image flip is a complete repair.

The extractor also returns a box whenever all four sampled target points have positive forward depth. It does not reject boxes with no image intersection and does not compare target depth with the RGB-aligned depth map. Examples found in the audit include raw boxes such as `[2032.13, 0, 640, 480]` for a 640×480 image while `visible=true`.

## Admission and repair plan

Until a corrected sidecar passes a new audit, SAGE3D `bbox` and `visible` must be blocked from Phase 1 identity loss and identity metrics. This does not affect InternData-N1 geometry supervision or TpT identity supervision.

Repair should proceed in this order:

1. Reproject from `robot_pos`, `robot_yaw`, `target_pos`, and `camera_info` with the correct horizontal sign and an explicit non-empty image-intersection test.
2. Audit camera-specific rotation/extrinsics rather than assuming every camera has identity rotation relative to the base.
3. Use the aligned depth image to reject clear occlusion and foreground-depth conflicts.
4. Use the existing modular person detector and OSNet/DINO appearance features only as an offline validator or pseudo-label sidecar. Identity selection must be anchored by a trustworthy first-visible target; a generic person detector alone cannot distinguish the target from distractors.
5. Run a second detector such as YOLO or Grounding DINO only on residual ambiguous cases. Agreement between two detectors still does not override target identity evidence.
6. Keep all repaired labels outside the source root, version their generation configuration, and rerun the same deterministic audit before enabling SAGE3D identity training.

The review montage uses yellow for stored SAGE3D projection, magenta for the horizontal-mirror counterfactual, green for the best independent person detection, and cyan for other detected people.

## Reproduction

```bash
PY=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
CUDA_VISIBLE_DEVICES=1 $PY scripts/audit_sage3d_bboxes.py \
  --output-dir results/wp2_sage_bbox_audit_v2 \
  --samples 384 \
  --frames-per-episode 3 \
  --batch-size 16 \
  --device cuda:0 \
  --montage-cases 72
```

Generated `report.json`, `records.jsonl`, and review montages are experiment artifacts and are not committed to Git.
