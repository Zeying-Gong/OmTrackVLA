# Current task
Updated: 2026-09-29
Status: WLA_OVERNIGHT_TRAINING_AUTHORIZED / DA3_RETIRED

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
- Job59352/70125:TRAINING_COMPLETE3456updates/2epochs;heldout6489windows;FULL_EVALUATION running,568/4215 at08:29. Final checkpoint hash verified.
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
