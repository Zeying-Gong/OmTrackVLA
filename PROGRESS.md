# Progress
Updated: 2026-10-06 China
Previous complete task history: archive/2026-10/PROGRESS_before_a4_20261006.md.

## Active A4
User requested repeated TEST-scene learning/tuning to raise A-method full4215 SR. A4 authorization/plan/lesson records are in A800 WLA/a4_sr_adaptation_20261006. All research state stays on NAS.
First implementation: pristineA2 warmstart;全4215采集，其中2635原成功作behavior anchors，1580失败作LightNav online recovery/handback correction; only terminal-SR1 complete0.7s teacher-only/no-fallback windows enter correction data. Failed/zero-window intervention gets an independent reset/full LightNav rescue, also terminal-SR1 gated;source4215,actualrollouts<=5795.
Freeze backbone/MetaQuery/geometry/XYvis,train action_expert only with teacher flow labels plus same-condition frozenA2 velocity distillation. Balanced mode/task sampling;SR selects checkpoints.
Real single batch2/single batch4/dual batch4 each2updates PASS:initialanchor0,expert changed,214frozen tensors/checkpointreload exact,dualpeak17.219GiB. Same132realPNG requests old/new serving bitwise exact. Separate simulator repeat STT0/AT3 exact butDT28 historicalactiondifference retained/rootcauseUNCONFIRMED;sameJPEG is notrawRGB proof. Actual newstep2 checkpoint serving/reset full3episodes166frames PASS.3979collectioninputs frozenSHA b87cfadaa2350892ad8eb34f70fa5165bf316cac9c13ddae88f8910a3b44c6a6,allchecksumPASS. Formal8A800/48h submission rejected by auto-review before SSH execution;noJob/no submission_state executed. Explicit resource approval question pending;no retry via heartbeat or alternateinterface.
Real STT304 intervention restored terminalsuccess and31correctionwindows;DT862 intervention failed butresetteacher rescue succeeded56steps/41windows. These are collection evidence,nottrainedA4efficacy. Historical expert uniquely matched all1580A2failures,874successes/873zerofallback:route feasibility only,nooracleupperbound.
Selector tests on actualA3regression retainA2;partial252/wrongsuccess/SR metadata rejected. Cross-campaign NASbest persistence added.
Fixed252 full-horizon selection panel,84/task uniformhash;A2 panel successSTT68/84,DT55/84,AT37/84. Only full4215 can establish final improvement.
A4 targets first>A2 62.514828%, then>B63.677343%;best originalA2 retained throughout. No guarantee from training loss or short checks.
Report TEST_SCENE_ADAPTATION_NOT_HELDOUT. No inference GT,episode lookup,or global-optimum claim.

## Completed A3
A3_61020/71943 SUCCEEDED;8430adapt,886589transitions,221392SACupdates,final frozen4215. Independent source/index/identity/instruction/seed7/initialRGB/full-horizon and431802-step trace/reward/policy audit PASS.
Checkpoint FINAL.pt SHA c7269fa14ba53ec60f4f27fa80aa9b295f83464ca5e0124597c30e0c10fce435.
STT1061/1405 SR75.516014%,TR81.245010%,CR2.633452%;DT583/1405 SR41.494662%,TR59.191678%,CR3.772242%;AT565/1405 SR40.213523%,TR70.669425%,CR4.768683%.
Overall2209/4215 SR52.408066%,TR70.368704%,CR3.724792%;SR-10.106762pp vsA2,-11.269276pp vsB.
A3 real learned behavior was worse:388 success improvements,814 regressions. Q target182 versus finite episode reward<=11,initial latent behavior notA2,nearbound actor44.81%;seeA4 A3_LESSONS.json.
Only finalA3 full evaluation,not checkpoint-optimum search. Failed attempts andA3 checkpoint retained.

## Preserved comparisons
A2_60058/60857 full4215:2635successes SR62.514828%;STT1143,DT796,AT696.
B_partial60994/60996 full4215:2684successes SR63.677343%;STT1169,DT787,AT728. Checkpoint d4987b0aa961d971c30095c1b914224fff4b4dcc4b604655bb6113cd0f6f2883. Btrained63488durableTRAINrows,notfull78769.
B60766 STOPPED;60507 OOMKilled137;60317 SIGKILLunconfirmed. Preserve all oldoutputs,do not resume.
OriginalA60316 fixed36diagnostic mixed;historicalA2 action-only/59752 separate. No unannotated failure cause filledas0.
A3 completed audit and bothCSV registered;A4 state supersedes oldA3-stop-monitor instruction.

Current concrete task is reviewable at A800 a4_sr_adaptation_20261006/{collection_a800.yaml,COLLECTION_DEVELOPMENT_READY.json,TRAIN_DEVELOPMENT_READY.json,TRAINED_SERVING_DEVELOPMENT.json,APPROVAL_BLOCK.json}. Real rescue developer3source/5rollouts gave90correction+40anchorwindows;traininginterface161windows179RGB includesearlier31handbackwindows. These are developer/teacher-assisted evidence,notA4 efficacy.
