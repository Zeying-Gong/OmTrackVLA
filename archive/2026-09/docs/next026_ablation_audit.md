# NEXT-026 Architecture v1 ablation audit

Date: 2026-09-13

This audit covers `ABL-V1-01` through `ABL-V1-09` from `PROGRESS.md` section
3.1.7. None of these runs may use `test_locked`; EVT-Bench is not waypoint
training data. Architecture-changing controls must not silently change the
frozen Architecture-v1 deployment path.

## Coverage matrix

| ID | Required comparison | Current evidence | Status / next action |
|---|---|---|---|
| ABL-V1-01 | DA3 vs DINOv2; frozen vs adapter | Explicit switches, loading reports and 4k diagnostic arms exist. | Diagnostic only; converged equal-budget result remains. |
| ABL-V1-02 | L11 vs L5+L11 | Explicit feature-fusion switch and 4k diagnostic arm exist. | Diagnostic only; converged equal-budget result remains. |
| ABL-V1-03 | RoIAlign 3x3 vs bbox mean pooling | Explicit target-pooling switch, tiny-box fallback and 4k diagnostic arm exist. | Diagnostic only; converged equal-budget result remains. |
| ABL-V1-04 | GRU vs single-step fusion | Direct 36,864-step comparison exposed the random-GRU optimization bottleneck; the frozen single-step-to-GRU curriculum then improved ADE/FDE, and zero-hidden counterfactual confirmed recurrent-state use. | Closed under the declared curriculum protocol; direct 4k arms remain diagnostic failures, not module conclusions. |
| ABL-V1-05 | SE(2) vs raw camera difference vs no ego | Three explicit ego-representation modes and 4k diagnostic arms exist. | Diagnostic only; converged equal-budget result remains. |
| ABL-V1-06 | geometric early+late vs late-only; geometry vs learned UWB-to-720 | Three explicit early-fusion modes exist. Independent late `z_uwb` remains present in every Architecture-v1 arm. | Diagnostic only; converged equal-budget result remains. |
| ABL-V1-07 | World-Action B/A/none; correct/zero/shuffled action | Training-only controls and 4k diagnostic arms exist. | Rerun from one shared initialization and converged budget; do not compare unequal continuation schedules. |
| ABL-V1-08 | Architecture-v1 Phase-1 init/direct Phase 2; OSNet teacher on/off | `outputs/ablations/next026_abl08_converged_v1/summary.json`: all four converged arms finished. Best is teacher-on + GRU curriculum at ADE/FDE `0.084341/0.146654 m`; finite coverage and safe-stop are 100%. Teacher-on improves matched single-step ADE/FDE by 1.33%/1.13%; GRU curriculum adds 0.62%/0.32%. | Complete. Waypoint-only gradients reach Fusion, GRU and DA3 adapter; DA3 pretrained loading coverage is 100%. |
| ABL-V1-09 | complete Architecture v1 vs Frozen Frontend Baseline | Seven-GPU resumable OSNet/cache build is active. The formal config uses seed `20260911`, 9×131,072 samples, global batch 32 and 36,864 optimizer steps. Evaluator uses the same seven nontrivial future points as E2E and reports ADE/FDE, path lengths, path ratio and per-horizon error. | In progress. After train/val/viz_val cache completion: unit tests, 2-step smoke, formal training, 1,024×4 val evaluation and fixed render run automatically in `next026_abl09_formal`. |

## ABL-V1-08 converged result

| Phase-1 OSNet teacher | Phase-2 policy | ADE (m) | FDE (m) | Path ratio |
|---|---|---:|---:|---:|
| off | single-step | 0.086013 | 0.148800 | 0.755717 |
| off | GRU curriculum | 0.085573 | 0.148125 | 0.768817 |
| on | single-step | 0.084867 | 0.147118 | 0.754872 |
| on | GRU curriculum | **0.084341** | **0.146654** | 0.758918 |

The GRU rows report the 4,096-step recurrent curriculum stage after their
converged single-step parent. They must not be described as fresh 4k models.
All evaluation uses SAGE3D `val`; `test_locked_used=false`.

## ABL-V1-09 fairness contract

- Same SAGE3D run-level train/val split, four condition modes and 1,024
  validation samples per mode.
- Same Phase-2 single-step budget: 36,864 optimizer steps, 1,179,648 samples and
  global batch 32. The best teacher-on GRU reference additionally has its
  declared Phase-1 teacher and recurrent curriculum, which the summary reports
  separately from the matched single-step references.
- ADE excludes point 0 because `canonical_waypoints` defines it as a constant
  zero anchor in both implementations; it averages points 1 through 7.
- Frozen cache anchors normally follow `initial+1, +4, +7, ...`; E2E anchors
  start at `max(history-1, initial+1)` and normally follow `+3, +6, +9, ...`.
  Thus the comparison shares episode splits, budget and metrics but not exact
  sample IDs; the 1–2 simulator-step phase offset is reported explicitly.
- GPU0 is reserved for the existing InternNav server. Cache and evaluation use
  physical GPUs1–7; the small MLP baseline trains on GPU1 so global batch stays
  exactly 32.

## Interpretation rule

The 4k matrix is an implementation/optimization diagnostic and cannot support
final module rankings because most direct-GRU arms remained on the short-path
collapse plateau. A formal claim requires the declared converged schedule,
unchanged validation selection, path-length diagnostics, fixed renders and an
explicit `test_locked_used=false` provenance field.
