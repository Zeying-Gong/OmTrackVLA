# Current task
Updated: 2026-09-30
Status: WLA_INTERVENTION_AB_TRAINING / DA3_RETIRED

## Decision
- Adopt WLA as the primary tracking route.
- DA3 collection, training and evaluation are retired. Earlier conditional retraining authorizations are superseded.
- Do not submit, restart or automatically resume DA3 jobs without a new explicit user instruction.
- Preserve all NAS data, checkpoints, logs and failure evidence.

## Current WLA work
- User permits WLA architecture changes and references to LightNav/USS. WLA remains the research route.
- Authorized16pairs/32complete rollouts: Job58925/Task69692 SUCCEEDED on4A800;32videos verified.
- Job58917 failed before model loading due to missing CUDA_VISIBLE_DEVICES; preserved and corrected.
- PASS:32complete rollouts/videos,2612frames,16matched initial hashes; browser playback/JSON export verified.
- Human review received for6/16cases:2wrong-target,3occlusion recovery failures,1both-success.
- Target grounding/memory prototype:unit and real-checkpoint developer checks PASS; efficacy UNVERIFIED.
- Real collector54frames PASS; full2043manifest/scene/data audit PASS.
- User authorized overnight formal training and full evaluation. Job59097/69868 SUCCEEDED05:38:21;2043rollouts/176814frames audited,6539invisible,0benchmark scene overlap.
- Job59110/69881 SUCCEEDED23:24:57;3838updates/2epochs;1543heldout point error0.114515→0.029676; action policy unchanged.
- Job59352/70125:training completed; evaluation STOPPED_USER_SUPERSEDED at user request2026-09-29. Partial results and checkpoint preserved.
- [Overnight stages, source hashes and limitations](docs/wla_overnight_training_20260928.md).
- [Development evidence and exact collection recipe](docs/wla_target_memory_development.md).
- Review: http://127.0.0.1:18781 ; tunnel: ssh -N -L 18781:127.0.0.1:18781 nas-a800
- Plan: [architecture and review](docs/wla_tracking_improvement_plan.md).

## Outstanding work
- DA3 Job58613 / Task69363 verified STOPPED at 2026-09-28 19:06:21 Asia/Shanghai after user reauthentication.
- The19:06all-cluster check found no active/queued DA3 jobs; WLA58638 was RUNNING then.
- No SUBMITTED, SUBMITTING or SCHEDULED jobs were returned in the all-cluster check.
- WLA Job58638 scheduler SUCCEEDED; full artifact audit remains pending.
- GitHub update is authorized for the documentation cleanup. Existing uncommitted experimental source changes remain on NAS.

## Recording acceptance
- PASS: scheduler SUCCEEDED2026-09-28 20:03:23; VERIFICATION.json on Baidu NAS.
- WLA case_02 changed fail-to-success; case_12 changed termination type but remains unsuccessful.
- Videos are selected diagnostic cases, not benchmark scores or exact historical replays.

## Retirement acceptance
- PASS: decision and comparative evidence recorded in docs/tracking_route_decision.md.
- PASS: current state compacted and prior state preserved under archive/2026-09/da3_retirement/.
- PASS: scheduler-confirmed DA3 cancellation.
- PASS: documentation/CSV validation; this change is the authorized GitHub documentation publication.

## Boundaries
WLA currently uses text+RGB, whereas DA3 used initial bbox+RGB with UWB missing.
This selects the current complete system, not a proven backbone-only winner.
A bbox/UWB WLA interface is separate, unimplemented work.

- 08:29 matched568STT partial: new/baseline SR79.049/79.049%,TR76.372/76.043%,CR8.803/8.099%;initial images/text match. DT/AT pending; no demonstrated SR gain.

User2026-09-29 authorized iterative recovery/return A/B development, formal training and evaluation; A/B may run in parallel after checks. No old59352 restart.
See [A/B plan and implementation gates](docs/wla_intervention_ab_20260929.md). Job59545/70402 SUCCEEDED;2043/2043 collected and full release/cache audit PASS. A59730/70592 SUCCEEDED00:33:28,full2epochs/3114updates. B59734/70596 RUNNING on4RTX4090;404/404 recorded inversion windows pass. Diagnostic59741/70603 COMPLETE72paired rollouts:AT SR+1/12,STT/DT SRunchanged,DT TRmacro-9.55pp;mixed results. A2 repair59752/70614 COMPLETE3114updates/frozen tensors verified;A2 diagnostic59767/70629 COMPLETE mixed; fixed confirmation59777/70639 RUNNING72 rollouts.

A real dual-GPU updates and A/B serving reset PASS; B inversion24/24PASS (meanADE3.99mm), full B trainer single/dual-GPU PASS; audited data are ready and A formal training has started. No tracking efficacy claim.

2026-09-30 02:03: A2 diagnostic59767/70629 SUCCEEDED,36 matched episodes. AT SR5/12→7/12;STT/DT SRunchanged;DT TRmacro-9.100pp and CR2/12→0. Mixed, not stable improvement; frozen target head did not eliminate DT regression. Fixed A2 reserved confirmation59777/70639 RUNNING4A800: fresh baseline36+A236 complete rollouts, seed7 FP32; no repeated selection on confirmation. B59734 continues full inversion.

2026-09-30 02:27: Confirmation59777/70639 SUCCEEDED02:23:08,72rollouts/36strictpairs. STT baseline/A2 SR83.333/83.333,TRmacro93.892/94.262,CR8.333/8.333;DT SR83.333/66.667,TR78.525/70.931,CR0/0;AT SR58.333/66.667,TR75.169/80.837,CR0/0. Mixed with replicated DT regression; no overall improvement claim. Confirmation is not for further tuning. Latest A2 round1_a package prepared; developer collector verification in progress, NOT submitted. B inversion continues.

2026-09-30 02:41: Latest A2 round1 collection59786/70648 RUNNING4A800,full2043 episodes. Developer completed student-only54actions and separate132action handoff episode:2takeovers/2returns,51teacher windows pass release audit,0fallback. Independent round1_a source SHA256 0ac735b8ea8711a1299090131fdaee94e02f8f1c29f3f3bd67bbe8b3b344d4ee; source immutable during run. B3104 inversion windows/3090pass; no B optimizer or efficacy claim.

2026-09-30 04:35: B59734/70596 FAILED04:21:23 CUDA OOM on24GB4090 during frozen backbone encoding (ranks0/3,22.86GiB allocated). Last progress5504windows/5477pass; optimizer not reached. All partial chunks/logs retained. A800 recovery development check: train5600:5602 had1/2inversion pass, no OOM; expanded contiguous5600:5620 retains failing sample and same95% threshold, currently running; not yet resubmitted. Round1 A59786 remains collecting.

2026-09-30 04:45: B full A800 retry59819/70681 SUBMITTED (not yet running),4A800,unchanged training_source b2501ed49ca9c3169467e7aecf9c2d00f1e8453642e035ef94cb57fa2593e168/config intervention_ab_20260929/train_b_a800.yaml. Full28114 inversion then2epochs,48h. Original4090 failure retained; no partial result reuse. A800 development contiguous train5600:5620 passed19/20;heldout2/2;2actual updates;peak allocated10.161GiB. Earlier2sample failed coverage preserved; expanded check includes same failing sample,95% gate unchanged. Full-run memory/coverage still requires monitoring.

2026-09-30 05:31: B59819/70681 scheduler RUNNING, initializing; no inversion progress or optimizer update yet. A2 round1 completed746/2043 without collection errors.

2026-09-30 10:55: A2 round1 collection59786/70648 SUCCEEDED10:40:14;2043/2043 episodes, RECOVERY_RELEASE_AUDIT_PASS,536 eligible corrections;22666 train/2887 heldout windows. All release/pose/target hashes independently verified; train/heldout scenes disjoint and no final benchmark overlap. Output /data/nas_ray/project/md-ak/users/zeying.gong/job_59786/task_70648/wla_intervention_round1_a_collection; evidence WLA/intervention_ab_20260929/round1_a/COMPLETION_AUDIT.json. B59819/70681 remains RUNNING:7604/28114 inversion windows,7558 passing; optimizer not started. Await full B training, fixed development diagnostic and B round1, then audited union with round0 and normal replay before iterative retraining. Collection completion is not tracking efficacy; confirmation is exhausted and final4215 unused.

2026-09-30 18:41: New A2 scope supersedes independent-query-first: EXISTING MetaQuery predicts current ego-relative ground-plane XY and visibility; invisible position supervision retained. Baseline already supervises log-range/cos/sin bearing; XY alone is not new geometry supervision. Four causal history frames span1.5s; no established persistent target query. Coordinate sign/real checkpoint checks remain pending; NOT TRAINED. See Baidu WLA identity_memory_20260930/SHARED_QUERY_AUDIT.json. B59819/70681 RUNNING,19304/28114 inversion,19175 passing; optimizer not started. Historical A2 action-only/59752 remains distinct; round1_a complete.

2026-09-30 19:23: A2 existing-MetaQuery XY/visibility isolated head passed2 real baseline58346 developer updates on visible+invisible TRAIN windows; query delta2.0027e-5, invisible spatial gradients nonzero, saved query tensors and new auxiliary-head reload outputs exact,12.216GiB peak. No formal training/closed-loop result; no recurrent memory added. Failed attempt before model load (missing replay visibility label) preserved; successful attempt02 sources/checkpoint retained. Full2043 source audit:7035invisible frames,614 beyond1.5s history,max4.912s. Shared round0 train24910 windows:12544 missing visibility,11557 visible,809 invisible; preserve all action/XY rows, mask only missing visibility loss. Simulator lateral-sign/full trainer checks pending. B59819/70681 RUNNING20204/28114 inversion,20071pass,optimizer not started.
