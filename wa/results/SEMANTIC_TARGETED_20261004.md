# Targeted semantic re-evaluation — 2026-10-04
Status: RUNNING61171/72191; 2026-10-04 18:59 Beijing new822/2125 complete. Priority165 complete:158initialized138success7invalid20laterfailures. Full result pending.
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

## Priority165 completion (2026-10-04 17:46 Beijing)

2026-10-04T17:46+08:00 PRIORITY165_COMPLETE:158 initialized,138 closed-loop success,7 stillinvalid,20 initialized-but-failed (8 terminalCollision/3Lost/9NormalBelowCriterion). STT52/57 DT45/57 AT41/51 successes; invalid2/2/3. All7 invalid in pRbA3pwrgk9 (STT27,38;DT27,38;AT27,38,71), cause UNVERIFIED. HumanCollision12 differs from8 terminalCollision failures. All165 initialRGB match baseline and165 newvideo ffprobe metadata/durationPASS, not full-frame human review.61171/72191 RUNNING new197/2125(STT89 DT57 AT51),lanes25/25/25/24/24/25/24/25;8PRIORITY_COMPLETE,now otherMP3D. All197 unique/disjoint fromreuse andRGBmatch; frozenplan sourcehashesPASS;8A800 ready; logs growing ages0..19s nofatal/OOM. PairedHTML18795 verified HTTP200/UI165 andinvalidfilter7; artifacts/semantic_priority_review_61171/report.json. These are priority-subset results, not fullSR or training improvement.

| Task | Old invalid | Now initialized | Closed-loop success | Still invalid | Later failure |
|---|---:|---:|---:|---:|---:|
| STT |57|55|52|2|3|
| DT |57|55|45|2|10|
| AT |51|48|41|3|7|
| Total |165|158|138|7|20|

Paired old/new video page: http://127.0.0.1:18795/ . Forward with `ssh -N -L 18795:127.0.0.1:18795 devpod-a800`.
NAS report: /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/semantic_priority_review_61171/report.json.
The seven residual invalid cases require diagnosis; do not infer occlusion or discard them from SR denominators. Evaluation continues with unchanged frozen source.

## Monitor 18:14 Beijing

2026-10-04T18:14+08:00 MONITOR61171/72191 RUNNING8A800:450/2125 unique newMP3D completed(+253vs17:46),STT342 DT57 AT51,lanes57/54/58/57/57/56/56/55. All8prioritybarriers complete;otherMP3D phase,0finalCOMPLETE. Invalid7 unchanged;priority165 outcome unchanged158initialized138success7invalid20laterfailures. Readycontracts/unique/disjointfromreuse/all450initialRGB/semanticrepaired PASS. Workerbytes147298/138405/149303/153201/152808/153085/145389/142002 allincreased;lastwrite0..47s,noTraceback/FATAL/CUDAoutofmemory/segfault.454mp4files includesactive,not454completed. Frozen2090reuse+450new=2540/4215 records available,NOTfullSR. No job/source/config/model/threshold changes. Continue remaining1675MP3D;prioritypairedHTML18795 remains subset only;residual7causeUNVERIFIED.

## Monitor 18:37 Beijing

2026-10-04T18:37+08:00 MONITOR61171/72191 RUNNING8A800:645/2125 unique newMP3D completed(+195vs18:14),STT537 DT57 AT51,lanes81/79/83/80/83/80/80/79. Invalid7 unchanged;8prioritybarriers,0finalCOMPLETE;otherMP3D phase. Readycontracts/unique/disjointfromreuse/all645initialRGB/semanticrepaired PASS. Workerbytes215986/204639/219519/214482/221584/222028/212433/208565 allincreased;lastwrite4..24s,noTraceback/FATAL/CUDAoutofmemory/segfault.649mp4files includesactive. Frozen2090reuse+645new=2735/4215 available,NOTfullSR. Remaining1480MP3D;priority165 remains158initialized138success7invalid20laterfailures;residual7causeUNVERIFIED. No running source/model/config changes or newjobs.

## Monitor 18:59 Beijing

2026-10-04T18:59+08:00 MONITOR61171/72191 RUNNING8A800:822/2125 unique newMP3D completed(+177vs18:37),STT683 DT88 AT51,lanes103/101/106/103/105/105/100/99. DTnonpriority nowprogressing. Invalid7 unchanged;8prioritybarriers,0finalCOMPLETE. Readycontracts/unique/disjointfromreuse/all822initialRGB/semanticrepaired PASS. Workerbytes277970/264207/281935/277718/278966/291968/270221/262701 allincreased;lastwrite0..56s,noTraceback/FATAL/CUDAoutofmemory/segfault.827mp4files includesactive. Frozen2090reuse+822new=2912/4215 available,NOTfullSR. Remaining1303MP3D;priority165 outcomeunchanged;residual7causeUNVERIFIED. No runtime/model/config changes or newjobs.
