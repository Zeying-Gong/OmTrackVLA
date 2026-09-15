# Clean48 perception data v1

The fixed48 multi-episode validator, bounded RGB loader, draft schema, and inactive-grant builder are implemented. **14 of 14 remote CPU tests passed**, with CUDA uninitialized and zero optimizer steps. The validator also passed a real compatibility check against the existing V3 pilot's complete 262-file, 119,735,031-byte source. The new48 collection is a separate batch; it has not been validated by these earlier checks.

## Current evidence

| Evidence | Result |
| --- | --- |
| Synthetic multi-episode CPU tests | 14 passed, 0 errors/failures; no real48 collection asserted |
| Existing V3 compatibility | Fixed48 ledger with exactly1 sealed/verified member and47 uncollected |
| Existing V3 raw source | 124 observations,123 transitions; every sealed artifact and target RGB/panoptic label checked |
| Existing V3 fit subset | Steps1..122, with82 valid visible target boxes |
| Existing V3 score subset | Steps0..122, with83 valid visible target boxes |
| Natural terminal | Step123 excluded; reset0 excluded from fit |
| Existing teacher qualification | Preserved as false; no action supervision admitted |
| Production grant | Only drafts; no active production grant created |
| Full48 validator deployment | Staged in a separate remote scratch directory; execution awaits final batch manifest and control seal |

The synthetic tests cover multi-episode causal history, independent visibility/bbox masks, single-pixel visible targets, mutable-batch isolation, bounded cache eviction, grant/hash checks, exact fixed pool membership, runtime-cache identity, media tampering, wrong clocks/semantic assignments, initialization errors, worker failures/incomplete transitions, truncated-terminal lies, extra model-input fields, selective member dropping, and post-admission RGB tampering. They do not stand in for actual48 raw-media validation.

## API

```python
from clean48_perception_data import validate_collection, load_perception_collection

report = validate_collection(collection_root, manifest_path, manifest_sha256)
# Validation is evidence only. Root reviews and creates a separate active grant.
data = load_perception_collection(collection_root, grant_path, grant_sha256,
                                 max_cached_frames=128)
inputs, labels = data.batch([(sample_id, step), ...], purpose="fit")
```

`data.fit_keys` and `data.score_keys` are tuples of `(sample_id, integer_step)` in original pool order, then ascending step. A fit observation must satisfy `step>0` and its original terminal flag must be false. The last nonterminal frame of a complete controlled prefix remains eligible; it is never relabeled as a terminal. The original status/exit code is retained even when a known teacher failure leaves valid perception data.

Per-episode audit metadata is available through `data.metadata['validation_report']['entries']`. Use rows with `status='perception_verified'`; they include fit/score steps, raw counts, source pins and original worker outcomes. All48 members remain in the ledger. There is no hidden retry, case replacement, scene selection based on model performance, or heldout split inside this loader. The pool contains official training scenes; heldout inference needs a separate data path and protocol.

| Returned field | CPU shape / dtype |
| --- | --- |
| `inputs.initial_rgb` | `[B,3,280,504]`, float32 |
| `inputs.initial_bbox` | `[B,4]`, float32; always the same episode's frame0 box |
| `inputs.ego_rgb` | `[B,4,3,280,504]`, float32; causal, reset-padded history |
| `labels.target_visible` | `[B]`, float32 |
| `labels.target_bbox` | `[B,4]`, float32; ignored boxes are NaN |
| `labels.visibility_label_valid` | `[B]`, bool |
| `labels.bbox_label_valid` | `[B]`, bool |

Batches accept1..8 keys. RGB uses OpenCV INTER_LINEAR resizing and CHW normalization to [0,1]. The LRU cache holds only resized raw RGB, with a configurable1..256 frame bound (default64). A frame occupies about1.69 MB, so128/256 cached frames occupy about217/434 MB before overhead. Even cache hits recheck current source-file bytes against the sealed pin. Returned tensors do not share writable storage with caches or stored labels. No DA3 feature cache, actions, GT poses, UWB, identity labels, stop/binding/ego supervision, or later GT boxes enter the input dictionary.

## Collector and admission contract

See `SCHEMA_DRAFT.md` and `collection_manifest_draft.json` for the full schema. Each source episode retains the current V3 supervisor/artifact format. The validator checks all sealed sizes/hashes, exact file membership, the canonical collection plan, original pool member, and actual runtime cache. It checks the raw action/observation clocks for sequence integrity; teacher decisions never become supervision. Target visibility and inclusive/normalized boxes are independently recomputed from the raw panoptic pixels.

Only these completed-prefix worker outcomes can proceed to raw validation: natural terminal, action cap, worker time budget, action guard failure, and guard prediction residual failure. Unknown collection failures, unsealed/unreaped workers, incomplete issued/returned/recorded actions, missing initialization, and invalid raw media are rejected and recorded. A guard residual on a natural terminal frame retains that true terminal flag. Episodes with no usable fit observations are recorded separately and excluded from proposed fit membership.

`build_grant_draft.py` includes every and only usable verified member in pool order. It creates `status='draft_for_root_review'`; the loader rejects that status. The active status, if root later approves concrete evidence, is `active_clean48_perception_fit`. Previous optimizer/formal-training flags and grants are not changed. A separate frozen training runner remains responsible for model/weight/source pins, sampling schedule, optimizer budget and benchmark limitations.

## Full48 validation entrypoint for root

`run_full48_validation.py stage` has already completed successfully. Do not rerun staging into the same immutable directory. After the batch finishes and root runs `simulation_sota_review/clean48_collection_v1/seal_batch_controls.py`, call:

```text
python run_full48_validation.py run --manifest-sha256 FINAL_COLLECTION_MANIFEST_SHA256 --batch-control-seal-sha256 FINAL_BATCH_CONTROL_SEAL_SHA256
```

The wrapper requires the actual finalized hashes and refuses an in-progress manifest. It runs only CPU validation, with no new collection or optimizer. It downloads only `summary.json`; complete validation JSON, draft grant and copied control metadata remain remote with hashes in that summary.

The remote stage is `/data/nfs/share/wam_tracking/OmTrackVLA/.codex_upload/clean48_perception_data_validation_v1`. Raw source is `/data/nfs/share/wam_tracking/OmTrackVLA/outputs/takeover/clean48_collection_v1`. Validation output is `/data/nfs/share/wam_tracking/OmTrackVLA/outputs/takeover/clean48_perception_data_validation_v1`.

The full validation runner explicitly checks the batch control seal's nine files, prepared/bundle/orchestrator amendment bindings, and correspondence between each verified raw episode and its originally prepared plan. It copies that seal and its metadata files next to the grant, and includes `evidence.batch_controls` in the draft. The generic loader itself validates only the manifest/report admission chain. The training runner must independently bind and check the additional batch-control evidence; it should not attribute that check to the generic loader.

## Important pins

| Artifact | SHA256 |
| --- | --- |
| `clean48_perception_data.py` | `2e925587f421c06d5785fb61f86175f32235f81a14590d8a7ace500c4240acad` |
| `test_clean48_perception_data.py` | `df91f0611159c159ada7aeb8d857914d628ace372743a10fc452766abb01e46b` |
| `cpu_result.json` | `39c62757288e837ab5e9a3123a16506ea17f311aba3916c824ee2565ccaae221` |
| Existing V3 `validation_report.json` | `6340c1cd24d67d594b90574e6b03aa7a82c32f414745ad9d6025d7e2d86e26a9` |
| Existing V3 compatibility `result.json` | `773d4761bc105c90401c8f862381fa172a43d22a64d9c278b75f115c6ea4ed57` |

`IMPLEMENTATION_VERIFICATION.json` records the wider local evidence and staged bundle pins. No result here establishes complete48 collection, an active fit grant, policy quality, target identity robustness, navigation success, or generalization.
