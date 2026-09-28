> Archived DA3 evidence. All earlier continuation/retraining instructions and live status claims are superseded by the 2026-09-28 retirement decision; Job58613 is STOPPED. See docs/tracking_route_decision.md.

# Native controller timebase audit — 2026-09-26

Status: NATIVE_CPU_INTEGRATOR_PASS; scene/runtime calibration and corrected evaluation adapter still PENDING.

## Authoritative inspection

- Training Job58085/Task68820 still advancing: step140086 at elapsed25521.11884s; frozen training code not modified.
- Canonical and Baidu evaluation trained_agent.py already pass sim.get_world_time() to DA3EVTAgent.act; observation timestamps are not sim_step/40 anymore.
- da3_policy/controller.py timed_lookahead_v1 still assumes replan0.1s and rotates body velocity by half of predicted yaw over0.1s.
- habitat-lab/habitat/tasks/rearrange/actions/actions.py BaseVelNonCylinderAction.step clips normalized actions, applies longitudinal/lateral/angular scales, and invokes update_base once. update_base integrates1/sim.ctrl_freq (configured40Hz ->0.025s).
- RearrangeSim.step calls internal_step(-1) ac_freq_ratio times; each internal_step calls step_world(dt). Therefore actual action duration must be derived from the instantiated physics timestep and ac_freq_ratio, then verified against world clock deltas. Do NOT assume action duration1/ctrl_freq or0.1s.
- Target habitat Python exposes Simulator.get_physics_time_step. Its native VelocityControl.integrate_transform doc explicitly says explicit Euler.

## Native CPU evidence

Executed in /data/nas_ray/home/zeying.gong/algorithm/envs/habitat/bin/python on nas-a800 without scene/GPU.
With identity rigid pose, local linear velocity(1,0,0), local angular velocity(0,1,0), dt0.025:
- translation[0.02500000037252903,0,0]
- rotation angle0.0249932948499918rad
- assertion abs(x-.025)<1e-6 and abs(z)<1e-7 PASS: no simultaneous-yaw midpoint rotation of translation.
With command velocities multiplied by.048/.025=1.92 and the SAME native dt.025:
- translation[0.04800000041723251,0,0]
- rotation angle0.04799698665738106rad
- assertion abs(x-.048)<1e-6 PASS.

0.048s is a previously recorded scene-world cadence, NOT measured by this isolated CPU test. The ratio test demonstrates native units only; it is not a closed-loop validation result.

## Next scoped implementation

- Keep legacy controller selectable for historical comparison; add a separately named simulator-time-aware adapter, without modifying frozen training/model/loss or Habitat world stepping.
- Use real simulator physics timestep, ac_freq_ratio and ctrl_freq; require positive finite values and validate successive observation-world deltas. Do not infer duration from robot displacement (collisions can clamp displacement).
- Map desired robot velocity over actual action duration to native command integration1/ctrl_freq, preserving normalized clipping. For explicit Euler, use current-body-frame velocity without invented half-yaw rotation.
- Keep label future times0.1..0.7s unchanged; integration duration is a separate field.
- Test forward/reverse/lateral/yaw/mirror, equal-duration equivalence, altered timebase, finite/invalid timing, saturation; verify against native integrator. Record intended/executed action timing in evaluation diagnostics.
- Official baseline keeps its original controller. Freeze all common simulator/episode/metric settings and rerun paired10 after Job58085 final checkpoint; report controller-version change explicitly. No claim of benefit until paired rollout evidence.

## Implementation update — 2026-09-26 22:28 Asia/Shanghai

- Implemented independent da3_policy/native_control.py and explicit native_timed_lookahead_v2 mode in canonical trained_agent.py. Legacy mode remains the default, and the running training snapshot is unchanged.
- NativeActionClock receives actual sim.get_physics_time_step(), sim.ac_freq_ratio, sim.ctrl_freq; validates clock deltas each observation and resets per episode. Missing/nonfinite/changed timebases fail closed.
- Label future times remain0.1..0.7s; command scaling uses world_action_duration/native_integration_dt; explicit Euler uses current body velocity without midpoint rotation.
- 23CPUtests PASS12.363s in actual habitat environment: 10 legacy controller regressions, 10 native controller/clock cases, 2 extracted real agent method clock/reset cases, 1 native Habitat integration test covering4motion cases. No skipped native test.
- Evidence: Baidu artifacts/controller_timebase_20260926_v2/{code,cpu_tests.log}; neither actual scene-clock nor closed-loop quality verified yet.
- SHA256 native_control.py a041e98df6e6138dc8605518bc79f6ae1bc402c4f0e29fcc4d1fa7d11d39c08a; trained_agent.py afb6c75b3480a30af436feb4868f0c9a141f8fb3b3665587f1e25991cc044ae4; test_da3_native_control.py 77936d592c328e155b924600d52594ce73ac7132c5032a8c0d123cf1350444ac.
- Initial patch dry-run failed due to incomplete line context; no partial application. Corrected context passed check and syntax compilation. No runtime failure hidden or training job restarted.
- Next freeze an isolated evaluation snapshot with this opt-in mode; preserve old snapshots and official controller; perform strict scene-clock checks during authorized matched-ten evaluation after the final mixed checkpoint exists.
