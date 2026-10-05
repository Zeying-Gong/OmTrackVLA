# WA current task — dual-teacher in-set adaptation
Updated:2026-10-05 12:09+08. Status: RESUME_SUBMITTED61269/72346 at12:07:45;8requestedA800 baidu_bj_a800,schedulerRUNNING;workerprogress pending. No training yet.
New source_dual_teacher_resume_v1 commit16feffa435f4f848ee408c4a1783923070c18233 clean;configwa/jobs/dual_teacher_resume_v1.yaml SHAc767438a17287a7a26c2b4bda63e7658720a0880836ecda12540561c2c5dd377. Output job_61269/task_72346/wa_dual_teacher_resume_v1.
User explicitly approved stop and remaining-only resume. Old61264/72341 STOPPED11:55:27;frozen279complete+3936remaining8x492. OldfailureSTT5cdEh9F2hJL/7 andpartialbranches retained/excluded. Totalprogress279 until newuniquecompletions verified;do notduplicate submit.
Plan artifacts/dual_teacher_resume_20261005_v1.json SHAad074042a27e3b7d4470019e1b1090ed12f9f5dd37538a7070c925dc0d3328a9;fullpairedselection/artifact hashes frozen by freeze_dual_teacher_resume.py with STOPPED check. Do notoverwrite.
Developmentfix keeps officialLightNav fallback for benchmarkscore,records events,excludes selectedfallbackbranch from demonstrations without relabeling its success.26selection/data/bridge tests+3Habitat faultinjection testsPASS. Actualfailedkey rerun:LNsuccessTR.4 OracleSuccessTR.666667 ->Oracle;no fallback recurred,notclaiming actualerrorreproduced. artifacts/dual_teacher_fallback_developer_v1/audit.json retained,notreuseformal.
Resume/combinedaudit implemented;33CPU+3Habitatfaultinjection testsPASS,8lane realdefinitionaudits492eachPASS,100externaldependencySHA PASS. Monitor8workerstartup andnewuniquecompletedpairs;fullcombinedaudit/cache/training pending.
Real joint diagnostic v1 PASS4optimizersteps22708..22711 from59866 model+optimizer;8base+8teacher windows,peak7.31GiB,no checkpoint saved.25CPUtestsPASS including actualpolicy condition builder label/text/oraclefuture exclusion. This is development validation,not model performance gain.
Development8rank sampler137teacherunique/exposures137,max1,no rank overlap. Explicit adaptationtraining CLI and realjointmodel shortcheckPASS. Fullcache/exposure still pending;do not infer formalratio from developmentfixture.
Development-only cache149candidate/137valid windows across3episodes;all137 actualstudentloader checks PASS,finite/causal/inputkeys;formal admission rejects developmentcache. Full collection/cache release still pending.
Cachebuilder andexplicit DualTeacherData committed16feffa4;originalTrackingData retains strict defaultidentity guard. GitHubpush announced to user before syncing;old frozen sources and60502 weights unchanged.
Source source_dual_teacher_v1 commit47822500c1492217c9ba272f40c06bfda31f1db8;configSHA0ff0e1fa776b90e88800ebb2effad161abb18044756234e2271271b57d20bea7.
Output /data/nas_ray/project/md-ak/users/zeying.gong/job_61264/task_72341/wa_dual_teacher_inset_v1. Do not duplicate submission.
User explicitly authorizes learning on the existing evaluation split to test in-set attainable performance.
Goal: WA SR individually exceeds same-protocol LightNav in STT/DT/AT; not untouched-test generalization.
Teacher rule: only successful branch eligible; both success choose higher following_rate; exact tie LightNav; both fail neither.
Pairing requires same task/key/seed/protocol/RGB/replayed dynamic takeover state. Select continuous branch, not action-level splicing.
Preserve baseline60502 weights and audited3627/4215 result. Student input remains RGB/initialBBox/currentUWB only.
Implementation: wa/wm/dual_teacher_selection.py; legacy training-only collector guards remain unchanged.
Oracle adapter real8rollouts PASS: HM3D1pair and MP3D3task pairs;all selected LightNav by higherTR or exacttie.
Data definition4215/cameraalignment PASS;12unit tests PASS;100 external model/runtime dependencies hashed.
Fullcollection scope:4215 paired starts (8430teacher branches),8A800,successful branch first then higherfollowing_rate.
No demonstration released to training until fullpair/record/SE2/cache/input-boundary audit. This is in-set adaptation, not generalization.
## Prior completed phase (immutable baseline)
Updated:2026-10-05 10:12+08. Status: SUCCEEDED_AUDITED_7_ONLY; broader research PARTIAL.

## Current result
- User requested only7 initialization replacements, no full rerun. 61259/72336 SUCCEEDED10:07:05 after10:03:45 submission (3m20s),2actual NVIDIA A800-SXM4-80GB.
- All7 initialization valid;3closed-loop success (STT/DT/AT pRbA3pwrgk9/38);3 pRbA3pwrgk9/27 below following criterion;AT/71 HumanCollision. No change to failure/success thresholds.
- Replaced exactly7 rows;4208 rows unchanged,4215unique each1405. Overall3627/4215=86.049822% versus3624/4215=85.978648%.
- SR STT90.818505 DT82.419929 AT84.911032%;TR87.573986/78.697793/85.206358%;HumanCollisionCR4.341637/6.690391/4.697509%;invalid0/0/0.
- Two lanes COMPLETE4+3;modelcontracts/plan/source/initialRGB PASS. All4215 imageSHA andvideo ffprobe metadata/duration PASS (not full-frame decode).
- Output /data/nas_ray/project/md-ak/users/zeying.gong/job_61259/task_72336/wa_initial_bbox_repair_v2.
- New HTML artifacts/initial_bbox_repair_review_job_61259 at18797;HTTP andUI7-filter verified. Old18796 auditedbaseline remains.
- Access: ssh -N -L 18797:127.0.0.1:18797 devpod-a800 then http://127.0.0.1:18797/.
- Full report wa/results/INITIAL_BBOX_REPAIR_20261005.md;no active WA evaluation/training from this request. Existing monitor remains paused;no automatic further runs.

## Repair boundary and immutable provenance
- Cause evidence: semanticPLY occluded RGB-visible human in7firstframes. Isolated RGB-scene-graph semantic rendering recoveredGTboxes;pairedRGB/camera/poses identical.
- Only initialGTbox annotation repaired using exact firstRGB SHA; no laterGTbox oralternate semantic renderer in closedloop. Subsequent HumanFollowing semantic metrics unchanged;not claiming all visibility errors fixed.
- Plan artifacts/initial_bbox_repair_v2/plan.json SHA6535f7b9a53e399ba7f2323c2d7f72d223146ee77f090223eb260f5885f6269a.
- Frozen source_initial_bbox_repair_v2 commit1efe5e6131e16bda712c6d5d5ee6a4a1f633a973;configwa/jobs/initial_bbox_repair_v2.yaml SHA05c74fcbb6c9273b077b3661ec34c4b03373eac6cd65ddad70b51318475c5b15.
- 7staticlight-pairs/7oldJPEG/7actualfirstcall checks PASS;CPU5boundary+4merge cases PASS.
- 61257/72334 FAILED0completed due staticprobe missingformalLightInfo setup;rawRGBguard rejected beforeaction. Failedsource60e0d7df/config2a881ef1/plan2c163cd4/logs retained;no relaxed checks.
- Fixed60502/71381step45900 SHA20cc84b3f231ad4056e16b91c52c84cb14e18d886a80324b5f27ffd773be5331.
- RGB+firstGTBBox+idealpolarUWB noise0delay0;no text/laterGTboxes. JEPA/MetaQuery/ActionExpert;JEPAtrainingauxonly,noMPC.
- mp3d_semantic_ply_v1 configSHA1dc43d5488cdcfc0b66d998a63fa87da588a37f115d6970099032caece9776d6 retained. No physics/model/loss/controller change.
- CR targetperson distanceever<0.5m,notwall/doorframe. TRreference-step normalized with52missingreferences/task usingexistingactual-stepfallback;macroTR separate.
- Existingvalidation includesdev/confirmation,notuntouchedtest;no fullLightNavcomparison orrealUWB claim. Scorechange is annotation repair,nottraininggain.

## Preserved baseline
- 61171/72191 SUCCEEDED2026-10-04 22:17:30,8A800,new2125+reused2090=4215. Source d6d3a805/configfc119a34/plan7ad7697d.
- Baseline3624success/4215;invalid7;MP3D2163repaired+HM3D2052unaffected. Full audit/page18796 preserved.
- Older faultysemantic protocol81.138790%,61144stopped722complete,60883failed/60885stopped/60989complete retained;never mixed as newmodel gains.
