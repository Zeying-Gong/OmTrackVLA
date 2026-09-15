# NEXT-025 Architecture v1 smoke

This implementation is intentionally limited to the deployment path needed to
prove a waypoint-only end-to-end gradient:

```text
raw RGB -> DA3-SMALL L11 -> Target/World/UWB fusion -> GRU -> 8x2 waypoint
```

UWB is represented twice and the two routes are not interchangeable:

1. base-frame position/covariance is propagated through the calibrated camera
   model and numerically integrated into a 20x36 Gaussian patch log-prior for
   Target Cross-Attention;
2. raw position/covariance/quality/age/validity is encoded independently as
   `z_uwb` for late fusion.

The official DA3 checkpoint is loaded before its last two transformer blocks are
wrapped with trainable residual adapters. All original DA3 parameters are
frozen. The loading report records checkpoint tensor/parameter coverage, while
the parameter inventory separately records frozen DA3, adapter, and newly
initialized policy parameters.

The first smoke uses only masked waypoint SmoothL1. It fails unless that loss
alone produces non-zero gradient norms in Fusion MLP, GRU, and the DA3 adapters.
No bbox, visibility, identity, ego-motion, World-Action, inverse, or stop loss is
included in this proof.

## Commands

CPU/stub tests:

```bash
/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python -m unittest \
  tests.test_end_to_end_model tests.test_end_to_end_data -v
```

Single-H100 real-DA3 smoke:

```bash
PY=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
DA3_SOURCE=/data/nfs/share/gzy/third_party/depth-anything-3/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/src
DA3_RUNTIME=/data/nfs/share/gzy/third_party/depth-anything-3/runtime-py39-v1
DA3_MODEL=/data/nfs/share/gzy/models/DA3-SMALL/e08cab65ca0ec38e7826075418411ab90cab4da3

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$DA3_RUNTIME:$DA3_SOURCE:." "$PY" \
  scripts/smoke_end_to_end_v1.py --backend da3 --model "$DA3_MODEL" \
  --sage3d-root /data/nfs/share/OmTrackVLA/data/sage3d_extracted \
  --sidecar-root results/sage3d_bbox_sidecar_v1 \
  --episode 0001_83992/stt/0/go2_realsense_d435i \
  --initial-index 0 --anchor-index 8 \
  --output outputs/training/next025_single_gpu_smoke/report.json
```

That command checks the Phase 1 split manifest, the sidecar admission and source
hashes, and the policy admission before loading four raw-RGB history frames. It
uses the recorded canonical trajectory only as the waypoint target and derives
the explicitly labelled `simulated_uwb` measurement from the recorded target
pose; neither value is passed to the model as privileged geometry.

One-step 8-GPU DDP smoke uses the same script under `torchrun`; no optimizer loop
or formal training is present.

## Pre-training visual inspection gate

Before NEXT-026 formal training, render the initial-weight policy on an admitted
SAGE3D `train` sample with a visibly non-trivial expert trajectory:

```bash
PY=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
DA3_SOURCE=/data/nfs/share/gzy/third_party/depth-anything-3/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/src
DA3_RUNTIME=/data/nfs/share/gzy/third_party/depth-anything-3/runtime-py39-v1
DA3_MODEL=/data/nfs/share/gzy/models/DA3-SMALL/e08cab65ca0ec38e7826075418411ab90cab4da3

CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$DA3_RUNTIME:$DA3_SOURCE:." "$PY" \
  scripts/render_end_to_end_v1_pretrain.py --model "$DA3_MODEL" \
  --episode 0001_83992/stt/0/go2_realsense_d435i \
  --initial-index 0 --anchor-index 106 \
  --output outputs/visualization/next026_pretrain_v1/architecture_v1_pretrain_dashboard.png \
  --report outputs/visualization/next026_pretrain_v1/report.json
```

The dashboard displays the initialization bbox and raw RGB history, Target and
Scene Attention, the camera-geometry UWB patch prior, local predicted/expert
waypoints, the base-frame UWB uncertainty, stop probability, and raw SE(2)
bottleneck values. It is explicitly labelled as an untrained-policy wiring
check: DA3 is pretrained, while adapters and policy parameters are at their
deterministic initial weights. Manual approval of this dashboard is required
before formal training begins.

## Explicit exclusions

- frozen perception cache, OSNet, or KPR in deployment;
- later external bbox, GT target point/pose/depth, or GT realized motion input;
- Flow/autoregressive waypoint decoder;
- multi-scale DA3 fusion, part tokens, memory banks, occupancy/depth decoders;
- old ResNet18 Phase 1 checkpoint migration;
- `test_locked` access or formal-scale training.
