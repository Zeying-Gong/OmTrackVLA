# WA current task — repair seven residual initializations only
Updated:2026-10-05. Status: STATIC_VERIFIED; seven-only closed-loop pending.

- User authorized only7 replacements; preserve4208 other rows and baseline4215. No full rerun.
- Seven paired static renders restore valid GT firstboxes with identicalRGB/poses/cameras. Scoped first-call annotation repair; later semantic metrics unchanged.
- Plan artifacts/initial_bbox_repair_v1/plan.json SHA2c163cd46a539480c486d9225d477c6e9354f2fd416d9203c7e0e8d5c3f2fcf8; report wa/results/INITIAL_BBOX_REPAIR_20261005.md.
- CPU5 boundary tests and2lane real data audit4+3 PASS; no formal repair job submitted yet.

## Final outcome
- 2026-10-04 FINAL61171/72191 SUCCEEDED22:17:30;8COMPLETE new2125+frozen2090=4215unique each1405. Sourcecommitd6d3a805 clean;configfc119a34 andplan/sourcehashesPASS;new/reused disjoint;MP3D2163repaired HM3D2052unaffectedreuse;combinedrows and recomputedsummary exact. All4215 initialimage fileSHA andoldRGBpairsPASS;4215video ffprobe metadata/durationPASS(not everyframe decoded). SR STT1275/1405=90.747331 DT1157/1405=82.348754 AT1192/1405=84.839858;overall3624/4215=85.978648%. TRreference-normalized87.533505/78.665741/85.166987;macro_TR92.393361/81.353472/88.007583;each52missingreference usesexistingactual-stepfallback. HumanCollisionCR4.412811/6.761566/4.768683%;invalid2/2/3retained. OldSR81.138790 preserved;semanticprotocol repair nottraininggain;no fullLightNav/unseen-test claim. No workerfatal/OOM. HTML artifacts/semantic_full_review_61171 at18796 verifiedHTTP200/UI4215. Priority165158initialized138success7invalid20laterfailures;remaining7pRbA3pwrgk9 causeUNVERIFIED. Phaseevaluation complete,overallresearchPARTIAL;pausemonitor afterreport.
- 61171/72191 submitted17:21:29, ended22:17:30 Beijing; elapsed4h56m01s,8A800.
- Full output /data/nas_ray/project/md-ak/users/zeying.gong/job_61171/task_72191/wa_semantic_targeted_v1.
- Independent HTML/audit /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/semantic_full_review_61171.
- Access: ssh -N -L 18796:127.0.0.1:18796 devpod-a800 then http://127.0.0.1:18796/.
- Old18794 remains oldprotocol;18795 remains priority165 pairedreview. No old result overwritten.

## Immutable protocol and provenance
- Plan artifacts/semantic_targeted_plan_20261004_v1.json SHA7ad7697df1a29b6cd5589b784390a31c4483e4118a1dbff79a57e944a38be458.
- Frozen source_semantic_targeted_v1 commitd6d3a8052539ba2ac419bc836f531b63d7017e09.
- Config wa/jobs/semantic_targeted_v1.yaml SHAfc119a34fef91892a1fba27a20a911ae632ed465df603c7dac0426f37f282ab5.
- 61144/72164 user-authorized STOPPED17:13:41;722retained(38repairedMP3D684HM3D),plus1368oldunaffectedHM3D;not4215fresh rerun.
- Fixed60502/71381step45900 SHA20cc84b3f231ad4056e16b91c52c84cb14e18d886a80324b5f27ffd773be5331.
- RGB+firstGTBBox+idealpolarUWB noise0delay0;no text/laterGTboxes. JEPA/MetaQuery/ActionExpert;JEPAtrainingauxonly,noMPC.
- mp3d_semantic_ply_v1 semanticYup only,RGB/collisionZup unchanged;no controller/physics/model/loss/threshold changes.
- CR targetperson distanceever<0.5m,notwall/doorframe. Existingvalidation includesdev/confirmation,notuntouchedtest.
- Full report wa/results/SEMANTIC_TARGETED_20261004.md;historical60883failed60885stopped60989complete retained.

## Remaining boundary
- Seven residualinvalid pRbA3pwrgk9 STT27/38 DT27/38 AT27/38/71 remain counted;causeUNVERIFIED.
- No further training,LightNavfull,policytuning orrerun initiated. New research direction requires user direction.
- Final phase monitor wa PAUSED via app tool after audit/report completion; no claim whole research or real-world deployment complete.
