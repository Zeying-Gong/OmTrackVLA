# WA current task — close the gap to LightNav

ACTIVE user2026-10-03 authorizes repair/training/evaluation. Near-term comparator: fixed24 >=17successes with collisions <=3/24 and invalid0; report per-stratum/regressions. Independent confirmation before generalization. Old80% was operational not benchmark/product acceptance.

## Evidence and current actions
- Recovery60502/71381 (8A800 45900steps cumulative2epochs) and60509/71388 complete:15/24 vsparent14/24 vsLightNav17/24;3collisions each. Oldcheckpoints preserved.
- Legacycontroller fully overwrites modelyaw within predicted1.5m;572/1987full overrides and1551blend>=.5. Doorwayraw.905 can becomeexecuted.215. Hypothesis not cureproof.
- Single-variable learned_yaw_guard_v1 complete24 planned: keeplegacy x/y;useclipped learned yaw withoutgain;60502weights/mixedzero/originalphysics. Recordraw/legacy/final.10000developer invariantsPASS.
- Fixed44recovery/44heldout probe: teacherADE .56266->.42632;heldout .21936->.20639. Teacherpartlylearned;noLRincrease evidence. Notfull971 orclosedloop.
- Data:single step0teacher281/971windows;takeoverfirst.5s110windows. Candidate balancedepisode/onsetsampling keeps15536teacherexposures;restart from59866 notepoch3. Awaitcontrollerresult beforetraining.

## Boundaries
JEPA/MetaQuery/ActionExpert;originallosses;RGB+BBox+polar simulatedUWB no text. NoGT inputs orbenchmarkphysics/metric changes. Noevalscenes in training. Formaltrain8GPU eval2GPU;A800sameNAS no migration. Independentconfirmation/realUWB/edge unverified. Oldheartbeatpaused. No new successclaim.
