# WA current task — confirmation phase complete
Updated: 2026-10-03 (Beijing). Status: COMPLETE_COMPARATOR_PHASE / OVERALL_RESEARCH_PARTIAL.

## Final independently paired results
- WA60770/71649 SUCCEEDED11:48:40; LightNav60771/71650 SUCCEEDED11:49:56; 2A800 each, all24 completed.
- WA20/24 (83.33%) vs LightNav17/24 (70.83%); benchmark collisions2/24 vs3/24; invalid initializations0 both.
- STT WA6/8 vsLN7/8; DT7/8 vs5/8; AT7/8 vs5/8. STT remains worse; no claim of every-task superiority.
- Paired4 WA-only successes /1 LN-only /16 both-success /3 both-fail. All24 initial RGB rehashed equal; exact predeclared keys.
- Confirmation8scenes disjoint development8, originaltrain625, recovery11 and collection91 scenes. One seed7; these are24episodes, not24independent scenes.
- Confirmation belongs to the existing validation benchmark previously covered by image-only full evaluation; not a never-run final test set.
- Full report: wa/results/CONFIRMATION_60770_60771.md; strict audit: wa/results/confirmation_60770_60771.json.

## Adopted correction and preserved contract
- Same60502 weights (step45900, cumulative2epochs); no new training/epoch3. SHA20cc84b3f231ad4056e16b91c52c84cb14e18d886a80324b5f27ffd773be5331.
- learned_yaw_guard_v1 retains clipped learned yaw instead of legacy target-heading override; translation guard/limits/physics unchanged.
- All2824 logged steps satisfy exact translation/yaw invariants. Diagnostic observer coverage100% both methods; LN fallback0.
- JEPA/MetaQuery/ActionExpert/original losses preserved; JEPA predictor is training auxiliary, not online MPC.
- Frozen confirmation source0a12e09aa249b6bcfac5e56ea4b5c82152d49b4b. Development60767 previously17/24 vsLN17/24, collision3 both.

## Boundaries and remaining work
- Overall same-set comparator reached with no higher collision count; stage monitor wa PAUSED via app tool and local config verified on2026-10-03; final Git sync completes this handoff.
- WA RGB+initialBBox+ideal simulated polar UWB (noise0,delay0), no text; LN RGB+text. Different-input system comparison.
- ProductSR90%, static/dynamic safety rates, real noisy UWB, edge latency and broad generalization remain UNVERIFIED.
- Benchmark HumanCollision is target-person base distance ever<0.5m, not general obstacle contact; static/dynamic product collision acceptance unverified.
- WA-only residual loss: STT GLAQ4DNUx5U/20, visibility loss with movement obstruction; exact geometry cause unproven.
- Both fail SUHsP6z2gcJ/13 in all3tasks (STT/AT collision,DTlost). Failed teacher branches must not become demonstrations.
- Do not tune repeatedly on this confirmation or train on its/development trajectories. Further changes require train-scene diagnostics and a fresh independent validation plan.
- Capped recovery plan971unique/15536exposures remains UNUSED. No auto newtraining/RL/architecture/loss change.
- Common semantic-descriptor and X11 startup warnings retained in logs; no fatal/OOM/fallback detected; no claim of warning-free runs.

## Review and operation
- NAS artifacts/closedloop_confirmation_60770:48videos, all HTTP200; localhost18793 serverPID4037470.
- Development72video comparison remains localhost18792; TensorBoard6006 run recovery_mix_60502 through45900/final offline metrics.
- Forward: ssh -N -L 18793:127.0.0.1:18793 -L 18792:127.0.0.1:18792 -L 16006:127.0.0.1:6006 devpod-a800
- Open http://127.0.0.1:18793/ (confirmation), http://127.0.0.1:18792/ (development), http://127.0.0.1:16006/ (TB).
- No new GPU jobs, no cross-NAS migration, no changes to frozen source/checkpoints or unrelated WLA jobs.
