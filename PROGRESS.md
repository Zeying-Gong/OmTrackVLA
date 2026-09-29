# Project progress
Updated: 2026-09-30

## Current route
WLA is the primary tracking route by explicit user decision. DA3 is retired.
Official OmTrackVLA and modular baselines remain reference implementations.
No further DA3 collection, training, evaluation or automatic DAgger retraining is authorized.

## WLA paired review and architecture
User permits architecture changes while retaining the WLA paradigm.
Authorized32full rollouts on16paired cases: Job58925/Task69692 SUCCEEDED on4A800 at20:03:23.
Initial Job58917/Task69684 failed before model load due to launcher GPU enumeration; failure retained.
Recording source r3 SHA256 8037a3636e4320c8900c10ef0102e798ba97c0e453579efd80850e4eb9adaffa.
Verified32videos/complete rollouts,2612source frames,16initial-hash matches.
WLA case_02 changed fail-to-success;case_12changed termination but remains unsuccessful.
Browser playback/case switch/copyable JSON export PASS; selected videos are diagnostic only.
Review: http://127.0.0.1:18781 ; tunnel: ssh -N -L 18781:127.0.0.1:18781 nas-a800
See [plan and recording provenance](docs/wla_tracking_improvement_plan.md).
Human review6/16cases:2wrong-target,3occlusion recovery failures,1both-success.
User observes slower WLA turning/following; cause remains unmeasured.
Priority:target grounding+identity memory, yaw/range-controller diagnostics, recovery-and-return data.
Independent target-memory/grounding prototype:4contract tests and real WLA checkpoint two-step developer check PASS; baseline initialization exact;peak10.313GiB.
Real training collector check:54frames,19student/35teacher actions;all per-step labels/diagnostics present.
Sampled old data16episodes/1496frames had0per-step grounding labels. Job59097/69868 SUCCEEDED05:38:21;full2043rollouts/176814frames audited,6539invisible.
Job59110/69881 SUCCEEDED23:24:57;3838updates/2epochs;1543heldout point error0.114515→0.029676; action policy unchanged.
Initial labels are all visible:visibility/GRU/action residual frozen in this stage.
Four-step temporal trainer passed real dual-GPU updates/checkpoint and causal serving reset checks.
Job59352/70125:training completed; evaluation STOPPED_USER_SUPERSEDED at user request2026-09-29. Partial results and checkpoint preserved.
See [overnight recipe and provenance](docs/wla_overnight_training_20260928.md).
[Development and collection specification](docs/wla_target_memory_development.md).
Initial-frame localization pretraining completed; temporal adaptation and tracking improvement remain UNVERIFIED.

## Verified evidence
| Result | Evidence | Status |
|---|---|---|
| DA3 mixed training | Job58085, 4 A800, 679878 updates / 2 epochs, final epoch_2 checkpoint verified | TRAINING_COMPLETE |
| DA3 paired diagnostic | Job58582: 0/10 successes, 40% collisions; valid-init subset 0/7 | EFFICACY_FAILED |
| WLA matched diagnostic | Job58346: 9/10 successes, 0 collisions; same 10 initial image hashes; valid-DA3-init subset 6/7 | VERIFIED_SYSTEM_COMPARISON |
| WLA completed full evaluation | Job58346: 1405 unique episodes/task; STT SR80.28%, DT53.31%, AT49.25% | COMPLETE |
| DA3 offline heldout | Job58633: 1003 episodes / 78958 windows; ADE0.75114m, FDE1.27461m | COMPLETE, not closed-loop success |
| WLA new candidate | Job58638, full STT SR81.28%; DT/AT incomplete at last check | EVALUATION_RUNNING |
| DA3 correction collection | Job58613 / Task69363, 8 A800 | STOPPED_USER_RETIRED |

The matched-ten comparison is one scene. WLA uses text; DA3 uses initial bbox with UWB missing.
Backbone, decoder, training and controller differ. Do not describe this as an encoder ablation.
DA3 diffusion terminal noise mismatch is confirmed in code, but its causal contribution remains UNVERIFIED.

## Cancellation
Initial stop attempts were rejected for expired authentication; no credentials were copied.
After user reauthentication, Job58613 and Task69363 were verified STOPPED at
2026-09-28 19:06:21 Asia/Shanghai.
All-cluster RUNNING returned WLA58638 only; SUBMITTED/SUBMITTING/SCHEDULED returned none.
No DA3 jobs remain active or queued. Logs and NAS outputs were retained.

## Storage and publication
All raw data, logs, checkpoints and partial correction rollouts are retained on NAS.
Partial correction data are not an audited training release; no correction retraining was run.
The code baseline is 9ad4bd1554f97e3a43913bdcd3d4f30af797f6b7 plus uncommitted experimental source.
This cleanup publishes documentation and records only, not those unreviewed source changes.
WLA execution snapshot is on Baidu NAS at algorithm/repos/WLA-EVT-20260925;
this document does not claim its source has been integrated into this repository.

## References
- [Decision and limitations](docs/tracking_route_decision.md)
- [Retirement record](archive/2026-09/da3_retirement/README.md)
- [Full pre-retirement progress](archive/2026-09/da3_retirement/pre_retirement_PROGRESS.md)
- [Original experiment ledger](EXPERIMENTS.csv)

08:29 matched568STT partial: new/baseline SR79.049/79.049%,TR76.372/76.043%,CR8.803/8.099%;initial images/text match. DT/AT pending; no demonstrated SR gain.

User2026-09-29 authorized iterative recovery/return A/B development, formal training and evaluation; A/B may run in parallel after checks. No old59352 restart.
See [A/B plan and implementation gates](docs/wla_intervention_ab_20260929.md). Job59545/70402 SUCCEEDED;2043/2043 collected and full release/cache audit PASS. A59730/70592 SUCCEEDED00:33:28,full2epochs/3114updates. B59734/70596 RUNNING on4RTX4090;404/404 recorded inversion windows pass. Diagnostic59741/70603 COMPLETE72paired rollouts:AT SR+1/12,STT/DT SRunchanged,DT TRmacro-9.55pp;mixed results. A2 repair59752/70614 COMPLETE3114updates/frozen tensors verified;A2 diagnostic59767/70629 RUNNING4A800,36new episodes with baseline reuse;confirmation36 unused.

A real dual-GPU updates and A/B serving reset PASS; B inversion24/24PASS (meanADE3.99mm), full B trainer single/dual-GPU PASS; audited data are ready and A formal training has started. No tracking efficacy claim.

Live NAS dashboard2026-09-29: http://127.0.0.1:18784/ ; tunnel: ssh -N -L 18784:127.0.0.1:18784 nas-a800 . Server/artifacts: Baidu NAS /data/nas_ray/project/md-ak/users/zeying.gong/wla_ab_dashboard_20260929 . Collection counts/rolling ETA and round0 A/B artifact adapters implemented; browser verified315/2043 at16:53. Future round1/eval counters not connected yet; estimates exclude those stages.

17:25 China heartbeat: Job59545/70402 RUNNING;458/2043 completed,0collection exceptions,247takeovers,146qualified returns. All4active shards updated within1minute. Full release/cache not present; A/B remain unsubmitted awaiting audited data. Detailed snapshot: Baidu intervention_ab_20260929/MONITOR_LATEST.json.
