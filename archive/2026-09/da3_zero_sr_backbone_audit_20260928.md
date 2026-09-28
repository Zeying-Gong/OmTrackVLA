> Historical diagnostic/decision snapshot. Latest retirement and verified cancellation: docs/tracking_route_decision.md; earlier pending/running instructions below are superseded.

# DA3 zero-SR and backbone attribution audit — 2026-09-28
Status: DIAGNOSIS_PARTIAL; no training/config changes or new jobs.
Read-only evidence: Job58582/Task69330 mixed_final_paired_ten; Job58633/Task69383 da3_internal_heldout; frozen eval source da3_mixed_eval_20260928_v3/code.
Verified final checkpoint Job58085 epoch_2.pt SHA b1fb2129e9f38202b43ab1d050c78da9e182d56bfb44023ff6d2dec363e027c6.
Paired CSV: DA3 0/10 successes; three invalid initial bboxes. On seven valid-initialization paired IDs DA3 0/7 (4 collision,3 lost), official 5/7. All ten belong to scene 2n8kARJN3HM; not full benchmark or matched backbone ablation.
Full saved-prediction/original-label CPU recomputation: 1003 episodes,78958 windows; ADE0.7511397986846645,FDE1.2746133059178706m. Zero-displacement offline reference ADE0.9100835381876667,FDE1.5420415195680242m; ADE improvement17.46%. Zero reference is NOT a closed-loop policy result.
Prediction endpoint x<-.05m:21095/78958; GT12474/78958. These are descriptive, not proof of coordinate error or forbidden reverse motion.
Frozen decoder:100 linear betas .0001.. .02; terminal alpha_bar .3635632480554922, signal coefficient .6029620618708048, noise .7977698615167834. Training q(x_T|x_0) retains substantial signal; sample() starts N(0,I),10 DDIM-like steps. Confirmed distribution mismatch; causal contribution to poor SR UNVERIFIED. Do not change schedule only at inference and claim a valid fix.
Encoder:DA3-SMALL ViT-S/14,L11,each frame S=1 (no cross-frame DA3 attention),typically train block11+norm;custom identity/four-query compression/causal temporal/DiT head. This is not evaluation of full DA3 geometry or tracking capability.
Current checkpoint identity-swap and backbone-controlled ablations NOT RUN; old tiny-overfit identity evidence must not be attributed to final checkpoint.
Priority: validate denoising-vs-free-sampling behavior and conditioning dependence on same saved data; then controlled DA3 vs DINO/SigLIP encoder ablation with same head/data/controller. Any architecture/loss changes or new formal training require explicit authorization. Existing correction collection unchanged.
References: https://arxiv.org/abs/2305.08891 ; https://github.com/ByteDance-Seed/Depth-Anything-3
