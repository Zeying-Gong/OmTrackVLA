# Independent review of clean perception fit v1

The recorded fit is internally consistent and its metrics reproduce from independently audited source labels. All three variants learned this one training scene. The results do **not** establish a uniformly better architecture, unseen-scene performance, or a working navigation policy.

The offline review passed 7,573 checks. It independently reconstructed all 738 before/after prediction records, all 18 summary groups, and all 768 optimizer-log records. The maximum difference between independently computed box IoU and the reported value was `2.0550355772641637e-7`; every IoU threshold count and visibility confusion count matched exactly. Ground truth, initialization, and causal-input records were verified against the original supervisor's sealed file manifest.

## Training-scene results

These numbers exclude reset step 0. There are 122 scored training observations: 41 with valid visible target boxes and 81 invisible observations. Visibility uses the fixed confidence threshold 0.9.

| Variant | Mean IoU before → after | Boxes IoU ≥ 0.5 | Boxes IoU ≥ 0.75 | Visible TP / FN | Invisible FP / TN | Confident correct boxes |
| --- | --- | --- | --- | --- | --- | --- |
| B0 pooled SmoothL1 | 0.17773 → 0.69914 | 38 / 41 | 17 / 41 | 41 / 0 | 0 / 81 | 38 / 41 |
| B1 pooled L1 + GIoU | 0.17773 → 0.69047 | 34 / 41 | 16 / 41 | 41 / 0 | 0 / 81 | 34 / 41 |
| D1 dense | 0.05885 → 0.73191 | 36 / 41 | 23 / 41 | 39 / 2 | 0 / 81 | 35 / 41 |

D1 has the highest mean IoU and most boxes above 0.75. B0 has more boxes above 0.5, more confident correct boxes, and no visible-target false negatives. These outcomes do not justify saying D1 fully outperforms the baseline. The readouts also have different trainable parameter counts: 724,997 for B0/B1 and 296,711 for D1. The experiment matches the data, optimizer schedule and projector initialization; it does not match total trainable parameters or isolate all architectural/loss effects.

Reset step 0 was excluded from optimization and scored separately. Its post-fit IoU is 0 for B0, 0.00444 for B1, and 0.17094 for D1. This result does not demonstrate reliable localization even on the initial template frame. All-123 scores are preserved in the JSON; they must not be described as the 122-observation optimization subset.

## What was checked

- The exact frozen plan SHA256 is `d8a0580d374043c47f8f9f609f80081c93e606b42a620b0a8300b777bf6d50cb`. The separate active perception grant has SHA256 `13b5e7a9e8e861bbc668d9138e19e7f8bd51a03ef5c08eb26d14541a4c0e8c08`.
- All 256 batches of 8 samples were independently regenerated from seed 20260915, matched to the sealed ledger, and compared against every optimizer log row in all three variants. Every sample belongs to steps 1 through 122; reset 0 and terminal 123 never appear in optimization. Bbox and visibility sample counts match source labels.
- The recorded optimizer budget is exactly 256 steps per variant, 768 total. Losses and gradient norms in every logged step are finite. The reported task hashes change during fitting. All variants report the same initial projector hash; B0 and B1 also report identical complete initial task hashes.
- Each prediction's visibility and bbox-validity labels match the audited episode. IoU is recomputed from predicted coordinates and original normalized ground-truth boxes, using float32 ground-truth conversion followed by independent scalar geometry. Aggregation uses these recomputed values rather than trusting saved IoU or summary fields.
- Local critical project sources match the frozen plan: runner, bootstrap, loader, loader tests, model, dense readout, active grant, raw verification, and train-scene pool. The result and plan retain their restrictions on generalization, teacher action supervision, old task checkpoints, and promotion.
- All reviewed inputs retained their original byte hashes through the review. No fit output or model source was modified. No GPU execution or optimizer step was added.

## Evidence limits

This review checks saved outputs and source contracts. It does not recreate initial tensor snapshots or load the three `.pt` checkpoint files. Initialization equality is supported by cross-record hash agreement and the pinned seed/setup code. Checkpoint hashes are recorded as training-run declarations, not presented as independently rehashed checkpoint bytes. External DA3 source and weight references remain pinned in the frozen plan; their bytes were checked by the fit runner before and after execution, and were not reread in this offline review.

The separate loader CPU suite passed 8/8 synthetic sealed-media tests. Its README and results are in `../clean_perception_fit_v1/README_DATA_LOADER.md`. The original raw audit verified 259 artifacts and all 124 RGB/panoptic observations. Any fit-time `data_verification.json` receipt is additional evidence and can be reviewed in a separate supplement without changing this sealed report.

The teacher episode failed the action/following-quality gate. Only independent bbox and visibility labels were admitted under a new narrow grant. The approximately six-second episode has one scene, a single visible segment, and no reappearance after disappearance. Memorizing background, frame progression, or this target's trajectory could explain good fit scores. Identity robustness, active search, three-mode navigation, arbitrary-target following, closed-loop task success, and SOTA remain unverified by this experiment. Cross-scene inference requires a separate report and cannot retroactively turn these fit scores into generalization evidence.

## Artifacts

- `verify_fit.py`: independent offline verification script; imports no project training or model module.
- `verification.json`: complete audit, source and result pins, recomputed summaries, and evidence limitations. SHA256: `4403906a230e9ecbae96f4887146e064393c48a41767dc6c9303782edc2f75b1`.
- `recomputed_rows.csv`: all 738 predictions with sealed ground truth, original reported IoU, and independently computed IoU.

The verifier creates new outputs with exclusive creation; existing evidence is not overwritten on rerun.
