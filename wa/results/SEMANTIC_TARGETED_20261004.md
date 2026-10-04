# Targeted semantic re-evaluation — 2026-10-04
Status: SUBMITTED61171/72191 at2026-10-04 17:21:29 Beijing; startupverified17:24 with8A800;16newprioritycomplete,14initvalid+success,2invalid; plus2previoussuccess=18/165 reviewed. No full recoveryclaim.
Config wa/jobs/semantic_targeted_v1.yaml SHAfc119a34fef91892a1fba27a20a911ae632ed465df603c7dac0426f37f282ab5.
Frozen source_semantic_targeted_v1 d6d3a8052539ba2ac419bc836f531b63d7017e09.
Output /data/nas_ray/project/md-ak/users/zeying.gong/job_61171/task_72191/wa_semantic_targeted_v1.
User explicitly authorized replacing full rerun with old165 invalid-first priority and affected MP3D only.

## Immutable accounting
61144/72164 deliberately STOPPED at2026-10-04 17:13:41 Beijing;722 complete rows retained:
684HM3D +38repairedMP3D. Interrupted partial episodes are not counted.
All684 HM3D new/old metrics (finish/status/SR/following_rate/steps/collision/init) and initialRGB exactly match.
prepare_episode is a no-op for HM3D, ready model contracts identical, each episode resets seed7.
Reuse684 freshHM3D and1368 oldHM3D; do not evaluate additional HM3D.
Reuse38 repairedMP3D; evaluate remaining2125 MP3D only.
Thus2090 frozen reused +2125 new =4215unique; each task1405; total2163MP3D and2052HM3D.

Old165invalid: two already repaired and successful in61144 (STT oLBMNvg9in8/55 and2n8kARJN3HM/157).
Prioritize remaining163 across allSTT/DT/AT; eight lanes priority[21,21,21,20,20,20,20,20].
All8 PRIORITY_COMPLETE barriers required before any nonpriority episode begins.
Remaining lane totals[266,266,266,266,266,265,265,265]; other1962MP3D follow priority stage.
No old affected MP3D metrics are reused. Every reused row records artifact_root/reuse_reason.
The old baseline remains immutable and all completed61144 records are preserved.
New results are not a fresh4215 run: report affected-scope rerun plus audited unaffected reuse.

## Plan and validation
NAS plan: /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/semantic_targeted_plan_20261004_v1.json
SHA256: 7ad7697df1a29b6cd5589b784390a31c4483e4118a1dbff79a57e944a38be458
Plan embeds rows,165keys,8lanes,source-file hashes; load validates hashes both before and after run.
36CPU tests PASS; eight-way real temporary-file barrier PASS;8lane x2phase x3task real Habitat data loads PASS.
Artifacts semantic_targeted_preflight_v1/lane_0..7/AUDIT_PASS.json.
Existing dependencies/config/checkpoint SHA checks PASS. A80025free atpreflight, no crossNAS.
No cluster smoke. New independent frozen source/output required, no modifications to old frozen source.
Initial patch new-file metadata failure fixed before applying; no partial runtime mutation.

## Scientific boundaries
Same60502/71381 step45900 checkpoint SHA20cc84b3f231ad4056e16b91c52c84cb14e18d886a80324b5f27ffd773be5331.
Same learned_yaw_guard_v1,mixedzero,seed7,RGB+initialGTBBox+idealUWB; no text/laterGTbox/model change.
Only semantic MP3D rotation fixed; no controller/physics/loss/success threshold change.
Invalids remain in denominators. CR means target-person distance ever<0.5m, not wall contacts.
TR reference-step normalized; macro_TR separate.
Full validation includes development/confirmation; no unseen-test or LightNav-superiority claim.
2/2 prior failures recovered is not evidence all165 recover.
