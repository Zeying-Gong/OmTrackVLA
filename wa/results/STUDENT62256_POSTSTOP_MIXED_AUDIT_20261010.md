# Student 62256 mixed evaluation: STT post-stop recovery and full audit

Updated: 2026-10-10 08:03 Beijing. This is evaluation-set adaptation on the same 1405 episodes per task, not unseen-test generalization.

## Scheduler and provenance

- STT Job62445/Task73719 was still RUNNING after all eight shards finished by 2026-10-09 13:30:40 Beijing. Each shard had its COMPLETE marker, 176/175 parseable episode rows, worker log, server-ready record, nonempty review video, initial JPEG and trace per key. The 1405 STT keys exactly matched the frozen manifest/teacher selection; no duplicate or missing key; 1405 initial RGB pair-evidence values matched the teacher. Fatal/Traceback/OOM searches found zero hits in the eight worker logs. The precise post-shard process hang was not proven.
- After the user's conditional authorization, the standard md_ai_kit stop --job 62445 command was sent. Job and Task both read back STOPPED at 2026-10-10 07:47:55 Beijing. This is not a scheduler SUCCEEDED run. DT Job62446/Task73720 and AT Job62447/Task73721 remain SUCCEEDED; neither was changed or resubmitted.
- STT root combined_episodes.jsonl and PARTITION_COMPLETE.json were absent at stop time. Using clean frozen evaluation source commit 5a23a982c7ef01f7fdec58faf27f6ea623ed9eeb, CPU-only recovery revalidated manifest SHA, teacher reference, eight server-ready/COMPLETE records, shard assignments, checkpoint 40915b366ee5a2ef5967e2ce49a94d2b0f45f149955dd5cccf85ac3d6ab178fc at step61252, mixed mode and all 1405 rows. It ran the frozen write_partition function in a new NAS staging directory and hard-linked only the two missing derived root files with no overwrite. Original shards, logs, videos, checkpoint and frozen code were not changed.
- Recovery staging: R/artifacts/student62256_stt_poststop_partition_20261010_v1. Its RECOVERY_PROVENANCE.json SHA256 is 82c3c148ea2d3ef1af23476221213aa24e0cc6ee4bf6c8fc7e853b87d2a58779. Recovered STT combined SHA256 is 2bdd41fca4ce10386f18202566306d33673cf30f2badf9c166cfa558163ea0ed.

## Independent stored-evidence audits

The existing clean result-audit source at R/source_student61715_result_audit_v1 commit 32fa14084e71eacb6c9527e493580f089a5b7428 ran the prepared merge, fixed-goal, best61609 comparison and media-review tools on fresh outputs. Here R is /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928.

| Audit | Result | NAS artifact and SHA256 |
| --- | --- | --- |
| 24-shard, 4215-row merge and teacher-initial pairing | MERGED_AND_PAIRED; 1405 exact unique keys/task, zero initial invalid | R/artifacts/student62256_full_audit_20261009_v1/summary.json; a37c4e103d94efade975e06b70cd08bbeb113fd2880885b1dae52518d936beea |
| Fixed 61377 goal thresholds | PASS, MET | R/artifacts/student62256_goal_20261009_v1/goal_report.json; ca159ce4ba8b1a3fcf5f9cca1f67fb5dcd4b792f3ebe5ce6830e3c1091e65f5c |
| Comparison with previously audited best61609 | PASS | R/artifacts/student62256_vs61609_20261009_v1/comparison.json; 0ded64779540f97b0d6609813aafd820514c6b3d17caffd574b6ade828eb4d10 |
| 4215 media and initial-state review | PASS, REVIEW_AUDITED; 4215 initial pairs, first-JPEG hashes and video metadata | R/artifacts/student62256_review_20261009_v1/audit.json; 69553429cb91213fa940a2d3601f5dda54157f8bd52a32121e8d751199d5781f |

| Task | Success | SR | Reference-normalized TR | Macro TR | CR |
| --- | ---: | ---: | ---: | ---: | ---: |
| STT | 1300/1405 | 92.53% | 88.56% | 92.27% | 2.35% |
| DT | 1184/1405 | 84.27% | 80.09% | 82.20% | 5.20% |
| AT | 1212/1405 | 86.26% | 85.77% | 88.33% | 4.06% |

The fixed success targets were STT>=1289, DT>=1173, AT>=1203; all three are met by +11/+11/+9. Compared with best61609's 1279/1178/1207, net changes are +21/+6/+5, or +32 successes across 4215. All three initial-invalid counts are zero. The reference-normalized TR is not macro TR. CR means target-person distance ever below 0.5m, not general obstacle collision.

## Scope and next step

The review verified first JPEG hashes and video stream/duration metadata, not every decoded video frame. The launch YAML contains assertions for eight distinct RTX4090 UUIDs and compute capability 8.9 before the evaluator starts, but the raw UUID list was not retained in the accessible logs; do not claim direct per-UUID audit. Most importantly, the STT scheduler provenance remains STOPPED plus independently recovered root files, not SUCCEEDED. No failed or prior result was erased.

The mixed three-task success-count stage is met on independently audited stored outputs. Same-weight explicit image/no-UWB paired evaluation is still a separate stage: best61609 STT image has 1120/1405 versus its mixed 1279/1405, while DT/AT image and any student62256 image evaluation remain unrun. Direct input removal is not no-UWB retraining. Do not mark the overall SR+UWB goal complete or submit a new GPU job from this report alone.
