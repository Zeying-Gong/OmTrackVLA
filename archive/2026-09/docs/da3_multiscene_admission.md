# DA3-SMALL multi-scene admission

## Decision

The pinned DA3-SMALL pose output is admitted as a confidence-gated Phase 1 pseudo ego-motion teacher for InternData-N1. Admission is limited to the frozen camera-family scale rule and thresholds in `configs/gates/da3_multiscene_v1.json`; raw DA3 translation is not metric and DA3 confidence is not a calibrated probability.

The final one-time `test_locked` evaluation passed all 14 gates. It used 32 clips from 12 held-out group/scene strata, admitted 18 clips (56.25% coverage), and had 3 bad clips among the admitted clips (16.67%). No locked images were generated or inspected.

## Frozen policy and provenance

- Policy ID: `da3-small-intern-multiscene-v1`.
- Policy SHA-256: `b4bfbe49a99e844594c9519736d5c4eb48d2933dbf5a14224979bf8f6a2334ab`.
- DA3 source commit: `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`.
- DA3-SMALL model revision: `e08cab65ca0ec38e7826075418411ab90cab4da3`.
- Weight SHA-256: `364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf`.
- Phase 1 split manifest: `configs/manifests/phase1_v1.json`; calibration uses `val`, development review uses `viz_val`, and formal admission uses `test_locked` exactly once.
- Selection: three clips per group when available, offsets 0/4/8/12, minimum 0.2 m reference motion, process resolution 504, seed 20260908.

`scripts/audit_da3_multiscene.py` checks every policy-controlled CLI value before loading the model. A development run must use `viz_val`; `test_locked` is rejected unless `--write-admission` is present, and admission output is written only after every gate passes. Reports and new admission files include both the policy ID and policy hash.

The policy file serializes the exact parameters used for the completed evaluation. It and the enforcement code were finalized without rerunning `test_locked`: the already-created locked artifacts retain the hashes below and must not be regenerated merely to add provenance fields.

## Method

Clips are selected deterministically from disjoint InternData-N1 group/scene split units. Ground truth is derived from the dataset pose/action matrices only for calibration and evaluation. Natural-language task fields are not loaded.

For each four-frame clip, DA3 OpenCV world-to-camera poses are converted through camera-to-world, the OpenCV/Habitat axis bridge, the audited camera-to-base transform, and finally canonical `(x_forward, y_left, yaw)`. Translation scale is fitted only on `val` as the median of per-clip median ground-truth/predicted displacement ratios. Separate values are required for the known D435i and ZED camera families:

| Camera family | Frozen scale | Calibration relative IQR |
|---|---:|---:|
| D435i | 2.074511 | 0.3421 |
| ZED | 1.806175 | 0.2648 |

The frozen confidence threshold is `2.947961`, selected on `val` using median dense depth confidence. A clip is admitted only if its motion geometry is non-degenerate and its confidence meets the threshold. The quality definition used to calibrate the threshold is mean translation error no more than 0.12 m, mean yaw error no more than 0.12 rad, and median relative scale error no more than 0.35.

The final gate additionally requires at least 20 clips per side, zero inference failures, calibration scale relative IQR no more than 0.50, evaluation coverage at least 0.50, admitted bad rate no more than 0.20, translation median/P90 no more than 0.12/0.25 m, yaw median/P90 no more than 0.12/0.25 rad, and scale-error median/P90 no more than 0.35/0.75.

## Experiments

### Rejected global-scale trial

The first trial used one global scale across both camera families. It completed 31 `val` and 31 disjoint `viz_val` clips across 11 groups with no inference failures. The scale was 1.879207, calibration relative IQR was 0.462809, and the fitted confidence threshold was 2.024682. Held-out coverage was 80.65%, but admitted bad rate was 24%, above the frozen 20% maximum. Only `heldout_bad_rate` failed, so this rule is rejected and must not be reused.

### Camera-family development review

The camera-family rule was then calibrated on 31 `val` clips and evaluated on 31 `viz_val` clips. It passed every development gate:

| Metric | Result |
|---|---:|
| Coverage | 61.29% |
| Admitted bad rate | 5.26% |
| Translation mean-error median / P90 | 0.03785 / 0.09934 m |
| Yaw mean-error median / P90 | 0.00622 / 0.01022 rad |
| Scale-error median / P90 | 0.11906 / 0.29710 |

The four development visualizations in `results/wp2_da3_multiscene_v2_viz/` were manually reviewed. Depth structure was generally plausible, low confidence mostly aligned with complex texture or degraded viewpoints, and confidence gating removed most large errors. One high-confidence admitted bad clip remained in the worst-case trajectory page; this residual risk is represented by the bad-rate gate rather than hidden.

### One-time locked result

After the development decision was frozen, `test_locked` was run once with visualizations disabled. Calibration had 31 clips; evaluation had 32 clips from 12 groups; inference failures were zero.

| Metric | Median | P90 | Maximum |
|---|---:|---:|---:|
| Translation mean error | 0.044991 m | 0.119449 m | 0.160654 m |
| Yaw mean error | 0.002505 rad | 0.012585 rad | 0.037179 rad |
| Relative scale error | 0.128413 | 0.393865 | 0.511559 |

Coverage was 56.25% and admitted bad rate was 16.67%; all 14 checks passed. `visualizations` is empty by design. Locked artifact hashes are:

| Artifact | SHA-256 |
|---|---|
| `selection.json` | `88a14504bbdd98374c84f377ec7291e94afe296c7ae3c7aca3c20322c1d0fae7` |
| `report.json` | `fce834f991f8f904be7955efdb40bee860c4b63d383f10139b5c548d90bd1003b` |
| `admission.json` | `093892c6788dd5317002b549f5751dad7249d40129b9043c09e939e344c2ddf73` |
| `records.jsonl` | `5f41403121fda7656985b7d15e5b07c2b8d91b95e6af25cdc9020b7e8e227834b` |

## Development command

The following command is for repeatable `viz_val` diagnostics only. It does not write admission:

```bash
PY=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
DA3_SOURCE=/data/nfs/share/gzy/third_party/depth-anything-3/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/src
DA3_RUNTIME=/data/nfs/share/gzy/third_party/depth-anything-3/runtime-py39-v1
DA3_MODEL=/data/nfs/share/gzy/models/DA3-SMALL/e08cab65ca0ec38e7826075418411ab90cab4da3

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$DA3_RUNTIME:$DA3_SOURCE:." $PY \
  scripts/audit_da3_multiscene.py \
  --policy configs/gates/da3_multiscene_v1.json \
  --model "$DA3_MODEL" \
  --model-revision e08cab65ca0ec38e7826075418411ab90cab4da3 \
  --output-dir results/wp2_da3_multiscene_v2_viz
```

Do not rerun or inspect `test_locked`. Future policy changes require a new policy ID and a separately governed locked split; they must not overwrite this result.
