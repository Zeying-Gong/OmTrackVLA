# SAGE3D repaired bbox/visibility sidecar

## Purpose

The source `derived.json` files are immutable and their original
`bbox`/`visible` fields remain inadmissible. NEXT-021 repairs those labels
into a versioned sidecar outside the source root. Training fails closed unless
the sidecar is complete, independently audited, and tied to the exact source
metadata by SHA-256. The generation spec is `sage3d-bbox-depth-v1`.

## Corrected projection

The camera basis uses robot/base x-forward, y-left, z-up and image x-right:

- forward = `[cos(yaw), sin(yaw), 0]`
- left = `[-sin(yaw), cos(yaw), 0]`
- right = `[sin(yaw), -cos(yaw), 0]`
- camera origin = robot position + rotated camera translation

This fixes the old extractor's horizontal sign error and honors the recorded
camera translation, including the G1 camera's 0.2 m forward and 0.45 m vertical
offset. The projected 1.7 m by 0.5 m target envelope must have a non-empty
intersection with the image.

## Depth visibility rule

Depth PNGs were verified as RGB-aligned `uint16` millimetres for all six
camera variants. Visibility is evaluated in the inner 60% x 70% of the box:

- tolerance is `max(0.30 m, 0.15 * expected_depth)`;
- at least 20% of valid pixels must support the expected target depth;
- no more than 50% may be nearer than the tolerance band.

Only `visible` carries a training bbox. Other reasons are
`behind_camera`, `out_of_view`, `missing_depth`, `empty_depth_region`,
`depth_inconsistent`, and `occluded_by_nearer_surface`.

## Sidecar layout and safety

- `manifest.json` records source/config hashes, per-episode hashes,
  completeness, and aggregate counts.
- `episodes/<run>/<mode>/<episode>/<camera>.json` contains labels aligned
  one-to-one with source steps.
- `admission.json` is created only by a passing full audit.

The builder refuses output inside the source root, writes episode files
atomically, and supports checksum-verified resume. The adapter validates the
manifest, admission result, source metadata, and episode hash before using any
SAGE3D identity label.

## Fixed-sample evidence

The frozen EXP-003 sample covers 384 frames, 128 episodes, all 18 mode/camera
strata, and initial/middle/final temporal positions. The repaired subset has:

- 369 visible and 15 rejected frames;
- 326 visible frames with an independent person detection;
- median best IoU 0.6662;
- strong/partial/conflict rates 82.21% / 15.34% / 2.45%;
- zero rejected frames with strong detector agreement.

All sample quality gates passed. A subset report remains non-admissible because
only a complete 7,105-episode sidecar may be enabled for training.

Detector, OSNet, and DINO outputs are validators only. A generic person
detection never replaces target identity, especially in multi-person frames.

## Full admission evidence

The complete build covers all 7,105 canonical accepted episodes and 2,131,500
steps with zero failures. It produced 2,111,385 visible labels; the remaining
steps were rejected as 6,919 out of view, 7,220 depth inconsistent, 3,797
behind the camera, 2,158 occluded by a nearer surface, and 21 missing depth.

The full manifest passed all seven integrity checks. Re-running the frozen
384-frame independent audit passed all four quality gates with the same sample
metrics above and wrote `admission.json`. The admitted manifest SHA-256 begins
with `f103f034`; audit results and four worst-case review montages are stored in
`results/sage3d_bbox_sidecar_v1_audit/` on the H100 workspace. Phase 1 now
includes SAGE3D identity samples, but continues to reject the original source
`bbox`/`visible` fields.

## Reproduction

Run from the repository root:

```bash
PYTHON=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
$PYTHON -m scripts.build_sage3d_sidecar \
  --output-dir results/sage3d_bbox_sidecar_v1 --workers 32 --resume
$PYTHON -m scripts.audit_sage3d_sidecar \
  --sidecar-dir results/sage3d_bbox_sidecar_v1 \
  --detector-records results/wp2_sage_bbox_audit_v2/records.jsonl \
  --output-dir results/sage3d_bbox_sidecar_v1_audit --write-admission
```

Only a passing full audit may add `sage3d_extracted` to `identity_datasets` in
the Phase 1 configuration. The admitted `v1` sidecar now satisfies this gate.
