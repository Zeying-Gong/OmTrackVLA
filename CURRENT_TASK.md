# WA current task — seven initialization repairs completed
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
