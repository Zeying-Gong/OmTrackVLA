# DA3-SMALL geometry probe

## Status

The pinned DA3-SMALL API, local weights, GPU inference, and coordinate conversion were exercised on one real InternData-N1 clip. The interface probe passed, but this result calibrates and evaluates scale on the same clip and was not itself sufficient for pseudo-label admission. The later disjoint-scene evaluation completed admission; see `docs/da3_multiscene_admission.md`.

## Provenance

- Official source: `ByteDance-Seed/Depth-Anything-3` at `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`.
- Model: `depth-anything/DA3-SMALL`, Apache-2.0, repository revision `e08cab65ca0ec38e7826075418411ab90cab4da3`.
- Weight: `model.safetensors`, 137,248,940 bytes, SHA-256 `364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf`.
- Source directory: `/data/nfs/share/gzy/third_party/depth-anything-3/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`.
- Isolated Python runtime: `/data/nfs/share/gzy/third_party/depth-anything-3/runtime-py39-v1`; the existing `omtrackvla` conda environment was not modified.
- Model directory: `/data/nfs/share/gzy/models/DA3-SMALL/e08cab65ca0ec38e7826075418411ab90cab4da3`.
- Runtime additions: addict 2.4.0, decorator 4.4.2, evo 1.31.1, huggingface-hub 0.36.0, moviepy 1.0.3, plyfile 1.1.2, proglog 0.1.12, pycolmap 3.11.1, safetensors 0.5.3, and trimesh 4.7.4. The base environment supplies PyTorch 2.6.0+cu124 and torchvision 0.21.0+cu124.
- Final machine-readable result: `results/wp2_da3_probe/da3_small_e08cab65_probe_v4.json` (not in Git), SHA-256 `d505d4297a74d3cd97836846c7418696a6b7785d4175a130982fa83df77be727`.

## Probe input and outputs

The probe uses frames 0, 4, 8, and 12 from InternData-N1 unit `3dfront_d435i/00154c06-2ee2-408a-9664-b8fd74742897`, episode 0, at process resolution 504. It reads the formal source paths without modifying them.

| Output | Observed value |
|---|---|
| Processed RGB | `[4, 280, 504, 3]` |
| Depth / confidence | `[4, 280, 504]`; all finite |
| Extrinsics / intrinsics | `[4, 3, 4]` / `[4, 3, 3]`; all extrinsics finite |
| Layer 5 / 11 features | `[4, 20, 36, 384]` each |
| Depth | min 0.2672, median 0.7560, mean 1.0577, p95 3.3563, max 4.7812 |
| Confidence | min 1.0000, p05 1.0000, median 4.3767, mean 4.5881, p95 8.8565, max 10.0488 |
| GPU model forward | about 0.68 seconds for four frames on one H100 |

DA3 confidence is not treated as a calibrated probability. These values are descriptive evidence only and do not define an admission threshold.

## Coordinate result

The initial implementation mixed DA3 OpenCV camera axes with the Intern/Habitat camera extrinsic and omitted the final `(right, forward)` to `(forward, left)` basis change. Its first run produced 0.254 m mean translation error and 0.171 rad mean yaw error, with the characteristic near-90-degree translation and yaw-sign mismatch.

The corrected conversion is:

1. invert DA3 OpenCV w2c to c2w;
2. apply `diag(1, -1, -1, 1)` from Habitat camera coordinates to OpenCV camera coordinates;
3. apply the audited Intern camera/base extrinsic;
4. convert Habitat planar `(right, forward)` to canonical `(forward, left)`;
5. estimate one positive metric scale from the median matched moving-pair displacement ratio.

For this clip, the metric scale is `2.071156`. After scale calibration, mean translation error is `0.004960 m` and mean yaw error is `0.005193 rad`. The last predicted canonical pose is `(0.48473, -0.04656, -0.16204)` versus ground truth `(0.48507, -0.04289, -0.17234)`.

## Scope of this probe

- Passed: local model loading; finite depth/confidence/pose; intrinsics and intermediate features; explicit OpenCV/Habitat/base/canonical conversion; one-clip scale recovery.
- Not established by this probe: a scale rule that does not use the evaluated trajectory itself, representative multi-scene error distributions, motion/rotation degeneracy rejection, or a confidence threshold tied to held-out geometric error.
- Therefore this single-clip probe never independently generated or admitted DA3 pseudo labels. Those missing checks were subsequently completed under `da3-small-intern-multiscene-v1`; this historical result remains interface and coordinate evidence only.
