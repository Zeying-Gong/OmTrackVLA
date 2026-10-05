# Residual seven initialization repair — 2026-10-05
Status: STATIC_VERIFIED; seven-only closed-loop evaluation pending.
User authorized repairing/replacing only seven residual invalid rows; no full rerun.
## Evidence
All seven belong to pRbA3pwrgk9: STT27/38, DT27/38, AT27/38/71.
Existing semantic PLY rendering returned zero target pixels. Isolated no-action paired rendering using the RGB scene graph recovered 3706/4509 (STT),3706/4468 (DT),3706/4489/5449 (AT) target pixels.
All seven paired RGB arrays, cameras, agent poses and nonsemantic stage properties are identical. Available depth arrays also identical; DT/AT depth absent, not claimed verified. Visually reviewed all seven mask overlays: visible target human, not another person or through-wall projection.
Habitat Simulator._sanitize_config forces load_semantic_mesh true when semantic sensors exist; merely setting config false did not disable it. Diagnostic process-only wrapper disabled it after sanitation. Shared simulator never edited.
Failed alternative GLB semantic-asset probe asserted because GLB lacks semantic vertex colors; retained artifacts, not adopted.
## Scoped repair
Freeze the corrected initial GT bbox with raw first-RGB hash per task/key in artifacts/initial_bbox_repair_v1/plan.json.
Only seven selected first calls may substitute a zero bbox after exact image-hash match. Valid/nonzero boxes, image mismatch and wrong episode fail closed.
Policy RGB/current ideal UWB and weights/controller unchanged. No future target, later GT boxes or semantic image enter policy.
Closed-loop simulator STILL uses mp3d_semantic_ply_v1 throughout. HumanFollowing and all success/collision thresholds unchanged; renderer ablation is not enabled in evaluation.
This fixes initial annotation only; it does NOT claim all subsequent semantic visibility errors are corrected. Broad semantic-metric replacement is out of this request.
Merge all seven actual outcomes, not only improvements, with unchanged4208 baseline rows; preserve old results and fixed4215 denominator.
## Artifacts
Plan: /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/initial_bbox_repair_v1/plan.json
SHA256: 2c163cd46a539480c486d9225d477c6e9354f2fd416d9203c7e0e8d5c3f2fcf8
Review: same directory/first_frame_review.png.
Paired reports: artifacts/residual7_no_semantic_mesh_v2_{task}_pRbA3pwrgk9_{episode}/report.json.
Baseline: job61171/task72191/wa_semantic_targeted_v1/combined_episodes.jsonl, immutable SHA in plan.
CPU5 tests pass incl first-only forwarding; two real dataset lane audits select4+3 episodes total7. Dependency hashes pass. Checkpoint hash recheck pending completion.
