# Progress
Updated: 2026-10-06 China
Historical state: archive/2026-10/PROGRESS_before_a4_20261006.md and archive/2026-10/a4_before_hybrid_reuse_20261006/.

## A4 hybrid: real short offline/online development passed
User requested reuse of A3 data plus online RL to raise A2's actual frozen full4215 SR. New package a4_hybrid_replay_20261006 supersedes full expert recollection as a prerequisite. No formal A4 job or efficacy result.
All1732 NPZ hashes and886589 actual latent/reward/component rows matched8430 complete traces. The complete-episode MC/S index preserves lane/action_id/done boundaries and unchanged old rewards.4760 successful attempts/2914 ever-successful source episodes,including568 A2 failures,are experience coverage across mixed policies.
Cached FP16[16,256] features precede the trainable reader. Actual absolute28D latent is available;8426 episodes lack full RGB. No fabricated RGB history or clipped-action reward reuse.
Fresh MC success value/outcome and weighted actual-action regression now implemented,with separate actor/value readers and soft A2 residual penalty. Actor sees exact CUDA z0;zero delta preserves A2. Old A3 actor/Q/optimizers are excluded.
CORE_DEVELOPMENT:two actual updates on12 real rows,finite gradients,all fresh trainable groups changed,exact checkpoint/optimizer/output reload. REPLAY_LOADER_DEVELOPMENT:task/episode-balanced actual sampling and complete online admission passed.
INFERENCE_DEVELOPMENT:132/132 same-real-PNG/history outputs match original A2 at zero residual;final metadata recheck55/55 exact. Reset,GT-key refusal,1207 frozen parameter/buffer versions and trained checkpoint reload passed.
Single real pipeline:2 offline+4 online updates;2 adapt105steps;2 frozen/reset eval114steps. Mixed update rows124 old/20 new. Final100d067f7a943bce475fa395fed890ac8877169aeb7f40a3dbc732f95d88d1e7.
Dual real pipeline:2 offline+4 shared online updates;2 adapt123steps;2 frozen/reset eval105steps. Mixed rows120 old/24 new. Final5d22c9c5e699720f98aa2d9be53fafdef2fbb1f7921a2685d76f224cfcf1ef4c.
Each developer run starts independently. Complete trace/result durability precedes admission and updates;frozen evaluation stays atupdate6;source hashes unchanged and reload exact. AT3/DT28 outcomes do not establish improvement.
Batch256:2 real updates passed,512 sampled rows,exact reload. IO7.362/7.466s vsGPUupdate0.1865/0.01475s with256MiB cache. Core-process peakRAM1.572GiB/GPU37.07MiB does not validate wholeworker memory. Fullx+a6.8566GiB;8GiB cache proposed,not full-prewarmed.

Independent single/dual CPU audits: PASS_WITH_EXPLICIT_EVIDENCE_LIMITS;105/123 real archived features reload exactly,96 checkpoint tensors finite. JPEG equality is the available initial-image evidence;raw RGB and independent full neural per-action replay remain unestablished because eval features/intermediate checkpoints were not retained. Dual noise steps60..67 have runtime CUDA assertions only. Initial controller reconstruction tolerance failure1.0041e-12 was preserved;actual logged actions match exactly,final CPU tolerance1e-10 passes. Formal entry must improve provenance without fabricating prior evidence.

## Precision finding and formal work remaining
Original A2 heads run under BF16 autocast;A3 ran heads FP32. A4 restores originalA2 precision while retainingFP32 Flow. Nine real probes showed two controller-action differences,max0.00437714;not a causal explanation of the fullA3 decline.
Archived A3_FP32_HEAD outcomes are historical supervision across a different execution kernel than new A2_BF16_HEAD online. Keep actual latent/rewards unchanged;do not infer counterfactualA4 returns.
Current online developer storesJPEG firstframes;equalJPEG hash is not historical rawRGB equality. Formal entry must addraw initialRGB and policyfeature hashes without fabricating missing evidence.
FORMAL_PIPELINE_DESIGN.json is NOT_SUBMITTABLE. Proposal4A800/72h,1024offline+8430onlineupdates,4215onlineepisodes,six252screens,maxone4215promotion;maximum9942complete rollouts. Budget is not a completion-time or efficacy guarantee.
Next:separate formal cohort scheduler,state snapshot/drain/restore,standalone frozen serving,model/optimizer/RNG/queue/cursor recovery,full protocol audit/BEST_FULL updates,then shortchecks/sourcefreeze/live resourcepreflight. Developer limits remain intact.
Old8A80048h recollection submission rejected byautomaticreview beforeexecution;noJob. OldAPPROVAL_BLOCK/source3979/failures remain. Revised hybrid resources have not been requested/approved;no bypass or stale reminders.

## Frozen full references
A3_61020/71943 SUCCEEDED:8430adapt/886589transitions/221392updates;frozen4215 independentpair/431802-step trace/reward auditPASS. FINALc7269fa14ba53ec60f4f27fa80aa9b295f83464ca5e0124597c30e0c10fce435.
STT1061/1405 SR75.516014 TR81.245010 CR2.633452;DT583/1405 SR41.494662 TR59.191678 CR3.772242;AT565/1405 SR40.213523 TR70.669425 CR4.768683(percent).
A3 overall2209/4215 SR52.408066 TR70.368704 CR3.724792;SR-10.106762ppvsA2,-11.269276ppvsB.388improvements/814regressions;Lost1331vs802. Onlyfinalcheckpointfulltested;no globaloptimumclaim.
A2_60058:2635/4215 SR62.514828%;STT1143,DT796,AT696. B60994:2684/4215 SR63.677343%;STT1169,DT787,AT728,highestmeasuredA2/B/A3;checkpointd4987b0aa961d971c30095c1b914224fff4b4dcc4b604655bb6113cd0f6f2883.
A4 BEST_FULL stilloriginalA2. Test-scene adaptation/selection is authorized,notheldoutgeneralization. HistoricalA2action-only59752 staysdistinct;old60766STOPPED/60507OOMKilled137/60317SIGKILLunconfirmed preserved. No oldjobrestart.
