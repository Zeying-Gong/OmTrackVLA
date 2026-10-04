# MP3D semantic PLY repair — 2026-10-04

Status: IMPLEMENTED_STATIC_REGRESSION_PASS; corrected closed-loop metrics PENDING.
User explicitly requested repair after reviewing oLBMNvg9in8/55.

## Root cause and minimal repair

The actual MP3D stage used up=[0,0,1], front=[0,1,0] for both RGB GLB and semantic PLY.
Habitat's GenericSemanticMeshData PLY loader already transforms PLY coordinates from -Z to -Y gravity.
Applying the GLB stage frame again misaligns the semantic occlusion mesh. This agrees with the observed
all-ceiling mask and controlled orientation ablation. Local loader source inspected:
../habitat-gs/src/esp/assets/GenericSemanticMeshData.cpp (PLY conversion lines66..72);
runtime is habitat_sim-0.3.1 Python3.9 egg, not the habitat-gs fork.
The runtime ablation, not a claim of identical fork source, establishes the effective correction.

Version mp3d_semantic_ply_v1 retains RGB/collision up=[0,0,1], front=[0,1,0],
sets semantic_up=[0,1,0], semantic_front=[0,0,-1].
Config SHA256 1dc43d5488cdcfc0b66d998a63fa87da588a37f115d6970099032caece9776d6.
TrackEnv overwrites outer simulator.scene_dataset with episode.scene_dataset_config.
Therefore prepare_episode deep-copies the episode and sets this field only for verified MP3D
scene GLB + semantic PLY with original default config. Custom MP3D configs fail closed.
HM3D episodes are unchanged. Shared benchmark code, installed runtime, meshes, datasets and old sources remain untouched.

## Paired static validation

| Task / key | Original target pixels | Fixed target pixels | BBox |
|---|---:|---:|---|
| STT oLBMNvg9in8/55 | 0 | 5198 | zero -> [214,85,259,287] |
| STT oLBMNvg9in8/12 valid control | 6730 | 6720 | same [227,92,291,281] |
| DT E9uDoFAP3SH/10 | 49 | 2382 | partial original box -> fuller target mask |
| AT 2n8kARJN3HM/299 | 0 | 3427 | zero -> [125,115,174,255] |
| HM3D STT bzCsHPLDztK/1 | 5484 | 5484 | unchanged |

All five: identical raw RGB hash, camera matrix, all agent transforms, and checked nonsemantic
stage attributes (asset handles, orientation, gravity, friction, restitution, scale).
STT depth hashes identical. DT/AT have no jaw depth sensor; depth comparison is unavailable, not PASS.
Target overlay for55 visually matches the only person; no arbitrary center box, later-frame hint,
or projected through-wall GT box. Original target ID and current-time semantic visibility retained.
DT49px case is not a zero-box reproduction; do not call it a recovered initialization failure.
2382px remains below the unchanged3000 facing threshold.
18 CPU tests PASS including dataset scope, nonmutation, missing assets/custom config rejection,
runtime frame assertion and original full-evaluation accounting.
All4215 real Habitat episode definitions PASS: each task721MP3D repair +684HM3D unchanged;
source hashes checked and info/start pose/scene/key unchanged.
All165 prior invalid rows belong to MP3D; no claim every failure is already recovered.

Evidence: artifacts/semantic_fix_pair_v2_stt_oLBMNvg9in8_{55,12}/report.json;
artifacts/semantic_fix_pair_v3_{dt_E9uDoFAP3SH_10,at_2n8kARJN3HM_299,stt_bzCsHPLDztK_1}/report.json.
Each contains raw RGB, semantic arrays, green-mask/red-box overlays.
Dataset preflight artifacts/semantic_dataset_preflight_v1.json.
Baseline, frustum-off and camera-reparent probes unchanged zero mask; not adopted.
v5 outer-only config ineffective; v6 changed RGB stage frame and was rejected; v7 restored RGB equality.
Paired probev1 unsupported enum serialization and v2 DT/AT missing-depth errors retained with logs.
Diagnostic probes do not constitute a navigation benchmark or new success rate.

## Evaluation integration and boundary

WA_SEMANTIC_PLY_FIX=mp3d_semantic_ply_v1 activates the versioned repair.
eval_full_mixed copies each eligible episode; DiagnosticAgent validates the actual stage before inference.
Unknown versions and old WA_RESUME_PLAN combinations are rejected. Each output row marks protocol and scope.
The default old protocol remains available for reproduction. Do not modify frozen old jobs.
No model/controller/loss/physics or success threshold changes; first-frame BBox only, no later GT inputs.
Same fixed60502 checkpoint SHA20cc84b3f231ad4056e16b91c52c84cb14e18d886a80324b5f27ffd773be5331.
RGB+initialBBox+ideal simulated current-pose polar UWB noise0 delay0, no text; JEPA auxiliary training only.

HumanFollowing uses the semantic detector too. The defect can affect initialized episodes' TR and SR,
so correcting165 rows alone and merging into the old result is invalid.
Fresh full4215 validation is required. Old3420/4215=81.138790% remains historical buggy-protocol result,
not overwritten and not a corrected score. This is validation including old dev/confirmation, not untouchedtest.
No full LightNav comparison, no new training, no success guarantees.
