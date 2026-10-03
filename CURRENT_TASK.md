# WA current task — independent LightNav confirmation

2026-10-03 11:42+08 60767/71646 SUCCEEDED11:35:08 complete24 learned_yaw_guard_v1:17/24 vs60509 15/24 vsLightNav17/24;STT7 DT6 AT4 all equalLightNav;CR3/24 unchanged;invalid0;24initialRGB3wayPASS;2doorway gains(STT/DT)0regressions.2022steps controlinvariantsPASS. CandidateyawTV.06419 vslegacy.03748 vsLightNav.20873;saturation.445%/.151%/9.496%;no safetyclaim. Video72paired artifacts/closedloop_review_60767 localhost18792HTTP200. No retraining. ConfirmationWA60770/71649 andLightNav60771/71650 RUNNING2A800each complete24 samepredeclaredconfirmation8scenes disjointdevelopment;source0a12e09aa249b6bcfac5e56ea4b5c82152d49b4b same60502weights. Countsnotyetverified. HeartbeatACTIVE20min. Samplingcappedplan971unique15536exposuresmax32/early6720 CPU10testsPASS;notusedbytraining.

## Acceptance and next work
- Near-term developmental comparator met:17/24 with3collisions and0invalid, same asLightNav; this is NOT broad superiority/product90% acceptance. WA RGB+BBox+ideal polar UWB no text; LightNav RGB+text.
- Check independent24 WA60770 vsLightNav60771, same initialRGB and all predeclared keys; report SR/collision/init per task, not just overall. Neither confirmation nor development becomes training data.
- Preserve60502 weights and learned_yaw_guard_v1 while confirmation runs. No more tuning to confirmation. If it fails, report gap; subsequent model development requires fresh independent validation rather than reusing this confirmation as untouched.
- JEPA/MetaQuery/ActionExpert and original losses unchanged; only legacyyaw override removed, originaltranslationguard/limits/physics retained. DoorwayAT stillLost; obstacleXB4GS9ShBRE bothmethods3collisions.
- No blindepoch3/RL. Capped recovery sampler remains plan-only; if training is justified, independent branch from59866,8GPU,max2cumulativeepochs, unchanged LR and auditable exposure.

## Artifacts
- wa/results/learned_yaw_60767.json: strict24paired/invariants/motion/visibility audit.
- wa/results/recovery_fit_audit_v1.json: fixed44teacher+44heldout pairedfit; teacherADE .56266->.42632, not full971.
- artifacts/recovery_sampling_capped_v1.json:971unique15536exposurescap32; sampling-only;sha6cba15c17dbf6a72c092cebd6dcb1e88c3cee5e8a0d729e46ed82b2c660bc146.
- wa/jobs/learned_yaw_confirmation_a800_v1.yaml configSHA4b98c7bf39e3cb0c70fb439f446dab4a8431de36326365f26a0e4ada3e7f1a49;LightNavconfigSHAd8313f0b6c53cef2c716d3cba2e5d29c8a99574bd2cf4d806946b540383d1ce5.
- Both newoutputs under job_60770/task_71649/wa_learned_yaw_confirmation_v1 and job_60771/task_71650/wa_lightnav_confirmation_v1/lightnav.
- Preview:ssh -N -L18792:127.0.0.1:18792 -L16006:127.0.0.1:6006 devpod-a800. Videohttp://127.0.0.1:18792 TBhttp://127.0.0.1:16006. SameNAS no migration;WLA60766 untouched.
