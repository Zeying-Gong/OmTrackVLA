# SAGE3D Phase 2 policy admission

## Decision

SAGE3D is admitted as the first Phase 2 person-follow policy source under
`sage3d-policy-se2-30hz-v1`. The source dataset remains read-only. This
admission covers expert robot waypoints and pose-derived **simulated** UWB;
it does not claim any real-UWB timing, noise, or LOS/NLOS coverage.

## Waypoint correction

The immutable extractor stores `waypoints_ego` at source control-step offsets
`+1,+4,...,+22`. That representation cannot be passed through because the WP-1
contract requires eight absolute positions in the current base frame with the
first point equal to `[0,0]`.

The Phase 2 adapter therefore recomputes waypoints from recorded robot poses at
offsets `0,+3,...,+21`:

- base x is forward: `[cos(yaw), sin(yaw)]`;
- base y is left: `[-sin(yaw), cos(yaw)]`;
- each world displacement is dotted with those two basis vectors;
- time offsets are `0.0,0.1,...,0.7` seconds at 30 Hz.

The source `waypoints_ego` field remains useful only as immutable extraction
evidence and is not a policy label.

## Fixed development audit

The read-only audit used 128 `val` and 128 `viz_val` episodes selected with seed
`20260909`; `test_locked` was neither read nor run. It covered 13 mode/camera
strata and reported:

- 256/256 episodes valid and contiguous;
- maximum target-local transform error: `0.000050 m`;
- maximum source-extractor reconstruction error: `0.000050 m`;
- canonical point-zero error: `0.0 m`;
- maximum 0.7-second horizon displacement: `1.737010 m`;
- per-episode target-speed evidence coverage: `98.4375%`;
- median recorded/commanded moving-speed ratio at 30 Hz: `1.012375`
  (P10/P90 `0.904398/1.070112`).

All 11 frozen checks passed. The report is stored outside Git at
`results/sage3d_policy_v1_audit/admission.json`; its SHA-256 is
`ec90f7579d7f9503d45c0dcf96023faad5f5a79f880ec17e4142ebd7bb2621802`.
The source `index.json` SHA-256 recorded by the report is
`fe84b71ae82ce69faec5045aefa4d891506b73710f76bf65ee3edaf03e0fe47e`.

## Remaining limits

Phase 2 may use the already-admitted bbox/depth sidecar for the one-time visual
identity initialization and for labels. Later bboxes remain forbidden model
inputs. Target pose may create a noiseless `simulated_uwb` observation only
when the record carries the declared simulation, transform, and clock spec
IDs. Real UWB remains blocked pending device logs and calibration.

## Reproduction

```bash
PYTHON=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
$PYTHON -m scripts.audit_sage3d_policy \
  --data-root /data/nfs/share/OmTrackVLA/data/sage3d_extracted \
  --manifest configs/manifests/phase1_v1.json \
  --gate configs/gates/sage3d_policy_v1.json \
  --split val --split viz_val --episodes-per-split 128 \
  --output results/sage3d_policy_v1_audit/admission.json
```
