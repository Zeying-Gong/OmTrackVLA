# Executor audit (2026-09-30)

Read-only inspection of pinned BENCH habitat-lab/habitat/tasks/rearrange/actions/actions.py BaseVelNonCylinderAction, lines539 onward, and track_infer_stt.yaml.

The actuator clips normalized commands to[-1,1], scales longitudinal/lateral/yaw by15/10/6.28, integrates once at1/40s, then applies collision_check to three offset cylinders. step_filter constrains their movement to NavMesh. If any displacement correction exceeds1e-5m, enabled sliding replaces the desired translation with the filtered motion of the most constrained offset. Desired displacement is therefore not guaranteed to execute.
Actual STT config has allow_back=True; allow_dyn_slide=True; enable_rotation_check_for_dyn_slide=False; offsets[0,0],[.25,0],[-.25,0]. A hypothesis that simultaneous rotation categorically disables sliding is NOT supported by this config. Do not disable safety or change evaluation physics to improve SR.

Measured observer59842 motion is consistent with constrained translation in several collision steps, but positions alone do not prove the exact filter/contact branch. Some DT steps reproduce commanded displacement closely, excluding a universal sign/scale error. Other agents and target also move between policy frames.

Next bounded diagnostic option: instrument the existing collision_check return without changing its arguments or output, log requested and filtered transforms outside policy RPC, then rerun full fixed24 once. This can establish NavMesh suppression versus later simulator correction. No additional controller heuristic should be adopted without a paired test.

Confirmed quality gap remains: first-epoch mixed16/24 versus second-epoch14/24. No checkpoint passed the per-task80% development gate. Independent confirmation unused. Training beyond2epochs remains prohibited by current decision rule.

Training-data repair must use training scenes only, with current-state observations and expert action supervision; diagnostic/confirmation trajectories must not become training data. Rare far recovery and constrained close retreat are distinct needs. A safe recovery collector has not yet been implemented or validated.

This audit is not a successful fix, nor a reason to claim deployment safety.
