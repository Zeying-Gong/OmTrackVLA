# Target grounding and memory: development and collection proposal
Status: DEVELOPMENT_PASS / FULL_COLLECTION_AWAITING_EXPLICIT_AUTHORIZATION
Execution: nas-a800, /data/nas_ray/home/zeying.gong/algorithm/repos/WLA-EVT-20260925/target_memory_20260928.
WLA source is a NAS snapshot without .git; base declaration cb78954213f5ca32494f13639b8340797144a664. Source manifest is the exact executable provenance.
Existing WLA58638 and all baseline code/checkpoints remain unchanged.

## Development implemented
- Training-only current RGB-aligned panoptic target point and visibility recorder; unknown/missing panoptic is an error, not invisibility.
- Foreground point nearest mask centroid, normalized pixel centers; invisible point is masked.
- Explicit caller-owned target state, learned attention pooling of MetaQuery taps, visibility-gated GRU memory.
- Predicted point, predicted visibility and memory representation condition existing WLA action expert through a zero-initialized residual.
- memory_enabled=False supports a grounding-only ablation.
- Same original text+RGB inference contract; training labels are only loss targets.
- Ordered-clip loss entry and explicit predict_step API; no implicit memory across random training batches.
- Full production clip trainer, checkpoint migration/serving adapter and efficacy validation remain future work, not claimed complete.
- Existing optional WLA Belief/World implementation is not enabled by this prototype.

## Checks completed
Four contract tests cover label alignment/foreground membership, missing versus invisible labels, reset/batch isolation, absent-coordinate masking, causality and recurrent gradients.
Real baseline checkpoint + one real training frame: two developer optimizer updates on new adapter only; exact baseline prediction at initialization; point/visibility/conditioning and later memory gradients finite and nonzero. Peak10.313GiB.
One real AT training episode through unchanged WLA/LightNav takeover servers on the development machine:54frames,19student actions/35teacher actions; every frame labeled and every action diagnosed. All54frames visible; actual invisible-simulator case not exercised, masking is unit-tested.
These are bounded developer checks, not formal training, benchmark results or cluster smoke tasks.

## Data admission
Existing teacher release sampled16episodes/1496frames: zero per-step grounding labels.
Only initial panoptic is saved; world positions cannot establish image visibility/occlusion.
Frozen collection manifest:2043episodes,1836train/207heldout,621/70disjoint scenes; all source episode indices/text/start positions/hashes verified.
No benchmark-scene overlap. Selected validation videos and user annotations never enter training.
Expected source: original wla_takeover_20260928 manifest, new recording only.
Do not claim the dataset is ready before actual full collection and label audit.

## Proposed full collection (not submitted)
Cluster: baidu_a800;4A80080GB; image x5-builder:cuda12.8-isaac5.0.0-v2.test1.
Four parallel lanes process all eight frozen shards (two sequential shards/GPU);2043complete episodes.
Timeout43200seconds. Original one-way student-to-teacher takeover/seed7/termination/inputs remain unchanged.
Student: Job58346/Task69086, step-0043203.pt, SHA2560b8f036fd282474d8c9efe9e34e1f4dc56b9282c44295bcda1f16edcf48460e1.
Teacher: existing LightNav-0 NAS checkpoint and released transport.
Command: bash /data/nas_ray/home/zeying.gong/algorithm/repos/WLA-EVT-20260925/target_memory_20260928/collection_pipeline.sh
Submit config: collection_a800.yaml.
Output: /data/nas_ray/project/md-ak/users/zeying.gong/job_${MD_AK_JOB_ID}/task_${MD_AK_TASK_ID}/wla_grounding_collection.
No cross-NAS migration. No auto-training after collection.
Acceptance:2043unique complete episode records, per-step label and action counts equal total_step, camera alignment, all source/data hashes, train/heldout/benchmark isolation, retained failures.
Existing teacher-owned future-window eligibility remains; all raw labels, including unsuccessful trajectories, are preserved. Audit must precede a training release.

## Control diagnosis (not an efficacy result)
For the same numerical first waypoint, released LightNav mapping produces1.78–2.08times WLA command at0.056/0.048second control intervals.
Predicted range1.4m can suppress WLA forward motion; range_only leaves learned yaw unchanged.
Waypoint timing contracts and learned outputs differ, so neither a controller bug nor a beneficial speed multiplier is established.
Do a separate closed-loop control ablation after logging predicted/actual range, bearing, yaw and executed commands. Keep controller fixed for the grounding-only versus memory comparison.

## Verified source and proposed launch
426 source files; SHA256 manifest08856dc6ddab0ac591d0d73acd7bb68962a128664849a38c65b795ce1c4a45c5.
Submit command (requires explicit formal-task authorization):

```bash
/data/nas_ray/home/zeying.gong/algorithm/envs/md_ai_kit_submit/bin/md_ai_kit submit /data/nas_ray/home/zeying.gong/algorithm/repos/WLA-EVT-20260925/target_memory_20260928/collection_a800.yaml
```

Pre-submit scheduler check:md_ai_kit2.0.0; authentication valid; baidu_bj_a8007free at check. Availability may change.
Model code is an independent Baidu NAS prototype; this documentation does not claim source integration into the Git repository.
