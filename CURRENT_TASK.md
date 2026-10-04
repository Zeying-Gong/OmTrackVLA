# WA current task — targeted MP3D semantic re-evaluation
Updated:2026-10-04 17:46+08. Status:61171/72191 RUNNING_OTHER_MP3D; PRIORITY165_COMPLETE.

## Authorized scope
- User explicitly requested MP3D-only rerun with old165 invalid starts first; reuse unaffected HM3D and preserve all complete results.
- 61144/72164 deliberately STOPPED17:13:41;722complete retained=684HM3D+38repairedMP3D. No original output deleted/overwritten.
- All684 completedHM3D new/old metrics/status/initialRGB exact; HM3D prepare_episode no-op, each episode seed reset, readycontracts unchanged.
- Plan reuses2090=722fresh+1368oldHM3D. New2125MP3D only; final4215unique eachtask1405, MP3D2163 HM3D2052.
- Old165invalid two alreadyrecovered andclosedloopsuccessful: STT oLBMNvg9in8/55 and2n8kARJN3HM/157.
- First stage remaining163 (21/21/21/20/20/20/20/20). Eight PRIORITY_COMPLETE barriers before any ofother1962MP3D.
- Report priority165 immediately after stage completion: initrecovery separately fromclosedloopsuccess/collision/otherfailure; no all165success claim.

## Active run
- 61171/72191 submitted2026-10-04 17:21:29 Beijing; actual8 A800-SXM4-80GB readycontractsPASS. At17:46 new197/2125 (STT89 DT57 AT51),8prioritybarriers complete; otherMP3D underway, logs growing nofatal/OOM.
- Priority165:158initialized,138success,7stillinvalid,20laterfailures; successesSTT52/57 DT45/57 AT41/51. All7invalid pRbA3pwrgk9 STT27/38 DT27/38 AT27/38/71; causeUNVERIFIED. All165initialRGBmatch andnewvideo metadataPASS.
- Source /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/source_semantic_targeted_v1 commit d6d3a8052539ba2ac419bc836f531b63d7017e09 cleanfrozen.
- Config wa/jobs/semantic_targeted_v1.yaml SHAfc119a34fef91892a1fba27a20a911ae632ed465df603c7dac0426f37f282ab5.
- Plan /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/semantic_targeted_plan_20261004_v1.json SHA7ad7697df1a29b6cd5589b784390a31c4483e4118a1dbff79a57e944a38be458 immutable.
- Eightlane totals266/266/266/266/266/265/265/265, full2125task notcluster smoke.
- Output /data/nas_ray/project/md-ak/users/zeying.gong/job_61171/task_72191/wa_semantic_targeted_v1.
- 36CPU tests/eight-way barrier/8lane x2phase x3task realHabitat loads PASS; dependencies/checkpointSHA verified; A80025freebefore submission. No crossNAS.
- Monitorwa ACTIVE20min; each reportactualnew/prioritycounts anderrors, no inference fromRUNNING alone.
- No new training, LightNavfull, model/controller/loss/physics/threshold change orotherjob interference.

## Model and repair contract
- Fixed60502/71381 checkpointstep45900 SHA20cc84b3f231ad4056e16b91c52c84cb14e18d886a80324b5f27ffd773be5331; learned_yaw_guard_v1 mixedzero sampling4 seed7.
- RGB+firstGTBBox+ideal currentpose polarUWB noise0delay0; no text orlaterGTBBox. JEPA/MetaQuery/ActionExpert retained; predictor training-only noMPC.
- mp3d_semantic_ply_v1 semanticYup only; render/collision Zup unchanged. SemanticconfigSHA1dc43d5488cdcfc0b66d998a63fa87da588a37f115d6970099032caece9776d6.
- Five pairedstaticcases exactRGB/camera/poses; STTdepthsame DTATdepthabsent. STT55 pixels0->5198.
- See wa/results/SEMANTIC_PLY_FIX_20261004.md and wa/results/SEMANTIC_TARGETED_20261004.md.

## Acceptance and history
- New8COMPLETE2125 +frozen2090 =4215unique; scopes/protocol/sourcehashes/initialRGB/video audit; no old affectedMP3D reused. Per-row artifact_root/reuse_reason preserved.
- Report as affected-scope rerun plus audited unaffected reuse, not4215allfresh. SR/TR/CR/invalid per task, allinvalid retained.
- CRtargetperson distanceever<0.5m, not wall/doorframe. TRreference-step normalized;macro_TR separate.
- Existing validation includesdevelopment/confirmation, notuntouchedtest; no generalLightNav superiority ortraining improvementclaim.
- Priorfull60989/71912 combined60885+60989 completed4215, oldSR3420/4215=81.138790%; historicalfaultysemanticprotocol, notoverwrite.
- PriorconfirmationWA20/24 vsLightNav17/24 hasdifferentinputs andsmallN.
- Prior currentstate archived archive/2026-10/CURRENT_TASK_before_semantic_targeted_20261004.md.
- Priority pairedHTML18795 verified; artifacts/semantic_priority_review_61171/report.json. ssh -N -L 18795:127.0.0.1:18795 devpod-a800 then http://127.0.0.1:18795/. Old18794 remainsoldprotocol; finalfullHTML pending. Finalaudit/report thenpausemonitor; researchcompletion separate.
