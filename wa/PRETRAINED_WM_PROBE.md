# Official pretrained WM probe — 2026-09-29

Status: PARTIAL. User selected JEPA-WM as main candidate and DINO-WM as comparison,
with academic noncommercial usage accepted. This supersedes the earlier candidate
ordering in WORLD_MODEL_AUDIT.md; it does not change the running baseline.

## What actually ran

Unmodified official Meta JEPA-WMs PointMaze predictor classes and released checkpoints,
with strict loading of every predictor/embedding/encoder key. DINO-WM here is the
Meta reproduction, not the original authors' checkpoint. Shared encoder is DINOv2
ViT-S/14. No language model or text decoder. Pixel decoder is omitted.

Source pins:
- JEPA-WMs: 13cf1d9c7e476f53c17714d2e0f1dc239a883ce0
- DINOv2: 7764ea0f912e53c92e82eb78a2a1631e92725fc8

Official checkpoint URLs:
- https://dl.fbaipublicfiles.com/jepa-wms/mz_jepa-wm.pth.tar
- https://dl.fbaipublicfiles.com/jepa-wms/mz_dino-wm.pth.tar

Checkpoint SHA256 values and encoder hash are in the attached JSON artifacts.
JEPA-WMs license remains CC-BY-NC 4.0; academic permission is not a commercial grant.

## Initial A800 developer measurements

Batch 1, three RGB frames resized to 224x224, ImageNet normalization, FP32,
256 patch tokens/frame. Five warmups, 20 synchronized iterations. Shared developer
GPU; timings are diagnostic, not a controlled performance comparison.

| Official variant | Total parameters | Predictor parameters | Encoder + one prediction p50 / p95 |
|---|---:|---:|---:|
| JEPA-WM | 39,683,136 | 17,626,480 | 19.69 / 33.47 ms |
| DINO-WM reproduction | 42,857,670 | 20,800,884 | 11.72 / 24.00 ms |

Both outputs finite, shape [1,3,256,384], and sensitive to action probes.
Future-token perturbation changed earlier JEPA outputs by max 1.66e-5 and DINO
outputs by 0; JEPA's small discrepancy is recorded, not asserted to be exact zero
or definitively diagnosed as numerical error. These are interface checks only.
PyTorch peak allocated memory was about 271/232 MiB, NOT process/device memory
or training memory. Context encoding is recomputed, not cached.

Real EVT RGB was used, but action/proprio inputs were synthetic normalized probes
(10-dimensional PointMaze stacked actions, 4-dimensional state). There is NO valid
EVT-to-PointMaze control mapping yet. Do not interpret these results as prediction
quality, tracking SR, collision rate, or navigation capability.

Runtime: isolated probe venv inheriting read-only torch2.8/Python3.11 packages,
timm1.0.30 installed only inside the probe venv. This is a narrow module diagnostic,
NOT validation of the repository's full Python3.10 simulator/training environment.

## Reproduction / external H100 result exchange

Script: wa/tools/pretrained_wm_probe.py. Create the pinned upstream source directories
under ROOT/upstream_audit/{jepa-wms,dinov2}, put downloaded checkpoints in
ROOT/models/jepa_wms, and provide a local pretrained DINOv2 weight file plus a folder
containing at least three rgb_*.png frames. Use an independent environment with
torch, torchvision, einops, timm, numpy and Pillow. No model is downloaded by script.

```bash
python wa/tools/pretrained_wm_probe.py --root /YOUR/NAS/ROOT \
  --encoder-weight /YOUR/NAS/dinov2_vits14_pretrain.pth \
  --rgb-root /YOUR/NAS/episode --output /YOUR/NAS/results/probe_h100.json
```

This short developer probe uses one GPU, not a formal 8-H100 training recipe.
Return the small JSON via GitHub; do not commit model weights/data. Different RGB,
runtime or hardware changes comparability. A full 8-GPU world-model training launch
is NOT ready and must not be represented by the old baseline launcher.

Next gates: resolve action/timebase anomalies; align official dynamics training to
EVT controls; add goal conditioning without text; evaluate target identity retention,
occlusion recovery and dynamic avoidance. Measure complete planning/control latency
on Thor separately. Job59519/Task70376 remains RUNNING and untouched as of this check.
