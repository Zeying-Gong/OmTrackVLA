# Tracking route decision: WLA
Decision date: 2026-09-28. User authorized adopting WLA and retiring all DA3 cluster work.

## Matched diagnostic
The same ten STT validation IDs in scene 2n8kARJN3HM were compared:
119,25,297,136,130,291,16,40,22,273. All ten initial RGB SHA-256 hashes match.

| System | Success | Collision |
|---|---:|---:|
| DA3 final mixed checkpoint, Job58582 | 0/10 | 4/10 |
| WLA completed candidate, Job58346 | 9/10 | 0/10 |
| Official OmTrackVLA, Job58582 | 7/10 | 0/10 |

Three DA3 initialization bboxes were invalid. On the same seven valid-init episodes:
DA30/7, WLA6/7, official5/7. DA3 failed with four collisions and three lost-target outcomes.
The in-progress WLA Job58638 also completed these ten with9/10 and no collisions.

This is a one-scene system comparison, not a full-benchmark DA3 estimate or backbone ablation.
WLA/official receive text+RGB; DA3 receives initial bbox+RGB, UWB missing.
Training, heads and controllers also differ. No claim that DA3 can never work is justified.

## Full and offline evidence
Completed WLA Job58346 contains1405 unique episodes per task:
STT SR80.28%, DT53.31%, AT49.25%.
New WLA Job58638 is a separate ongoing evaluation; do not replace full results with partial DT/AT.

DA3 Job58085 finished679878updates/2epochs. Final checkpoint SHA-256:
b1fb2129e9f38202b43ab1d050c78da9e182d56bfb44023ff6d2dec363e027c6.
Job58633 evaluated1003heldout episodes/78958windows: ADE0.75114m,FDE1.27461m.
Recomputed zero-displacement reference: ADE0.91008m,FDE1.54204m. This reference is offline only.

## Diagnosis and choice
The frozen DA3 decoder uses100linear betas from0.0001 to0.02.
Terminal alpha_bar=0.36356325: training retains0.602962 times clean signal;
sampling starts from pure Gaussian noise. This mismatch is confirmed, its causal effect UNVERIFIED.
Identity conditioning of the final checkpoint and controlled backbone comparisons remain unverified.
We select the stronger demonstrated WLA system instead of investing in further DA3 experiments.

## Operational outcome
All further DA3 submissions/restarts/retraining are prohibited absent renewed user instruction.
The last active DA3 task, Job58613/Task69363, was verified STOPPED on2026-09-28
at19:06:21Asia/Shanghai after user reauthentication. Initial authentication failures remain historical.
All-cluster RUNNING shows WLA58638 only; SUBMITTED/SUBMITTING/SCHEDULED are empty.
WLA Job58638 remains untouched. Completed and partial DA3 artifacts are retained.

## Evidence locations
Baidu NAS root: /data/nas_ray/project/md-ak/users/zeying.gong/
- job_58582/task_69330/mixed_final_paired_ten/{comparison.json,paired_episodes.csv}
- job_58346/task_69086/wla_teacher_train/wla_after/shard_*/episodes.jsonl
- job_58633/task_69383/da3_internal_heldout/evaluation/complete.json
- job_58613/task_69363/evt_train_corrections/
- job_58638/task_69388/wla_takeover/

[DA3 project](https://github.com/ByteDance-Seed/Depth-Anything-3)
and [diffusion schedule reference](https://arxiv.org/abs/2305.08891).
