# H100 asset inventory — 2026-09-29

User requested public downloads first and no unnecessary NAS migration. Data and WLA-checkpoint transfers have been stopped. Partial destination files are retained but MUST NOT be used without completeness/hash verification. Existing 8-RTX4090 JEPA Job59566 is unaffected.

## Download instead of copying

- JEPA-WM PointMaze checkpoint `mz_jepa-wm.pth.tar`: 211639615 bytes. Official URL: https://dl.fbaipublicfiles.com/jepa-wms/mz_jepa-wm.pth.tar . Expected SHA256: a01d99c4592fbedf44af076cf4c339de230c56f9f377c7559f584b97569b59bc.
- DINOv2 `dinov2_vits14_pretrain.pth`: 88283115 bytes. Official repository/download links: https://github.com/facebookresearch/dinov2 . Expected SHA256: b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9.
- Code: clone OmTrackVLA `wa`, facebookresearch/jepa-wms at13cf1d9c7e476f53c17714d2e0f1dc239a883ce0 and facebookresearch/dinov2 at7764ea0f912e53c92e82eb78a2a1631e92725fc8. No local Git bundle transfer needed.
- Python dependencies: install pinned versions at destination. Do not copy an entire environment.

## Custom training data: not an official downloadable dataset

These are the exact files referenced by the current audited robot-domain corpus, not whole source directories. They include RGB frames, metadata.json, observations.json and actions.json. Public PointMaze trajectories do not substitute for this robot tracking task.

Base on Baidu NAS: `/data/nas_ray/home/zeying.gong/`.

| Relative directory | Selected files | Exact bytes |
|---|---:|---:|
| datasets/evt_teacher_at_20260924_v1/lightnav | 197199 | 29981296236 |
| datasets/evt_teacher_corpus_20260924_v1/lightnav | 447992 | 67488809683 |
| datasets/evt_teacher_corpus_20260924_v1/official | 170382 | 26036488576 |
| datasets/evt_teacher_corpus_20260924_v1/oracle | 15470 | 2239901713 |
| datasets/evt_teacher_dt_20260924_v1/lightnav | 219820 | 33815918895 |
| datasets/wla_evt_se2_cache_20260925_v2 | 11 | 134402898 |
| Total | 1050874 | 159696818001 |

The full per-file byte/path listing is on Baidu NAS:
`/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/h100_migration_v1/file_sizes.tsv`.
Group summary: `sizes.json` in the same directory. These counts describe the current complete-episode manifest, not a proved minimum subset of frames.

## Other custom artifacts

- Original trained WLA checkpoint: `/data/nas_ray/project/md-ak/users/zeying.gong/job_58346/task_69086/wla_teacher_train/training/checkpoints/step-0043203.pt`, 4118468737 bytes. No public equivalent verified. Only action_expert/metaquery/target_head are used; a smaller lossless heads-only export could avoid unused content, but its size is not yet measured.
- Audit/index directory: `/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/robot_transition_audit_v2`, five files totaling11876384 bytes. Already copied; no further transfer needed if hashes match.
- WLA custom source snapshot: `/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/dependencies/wla_v1/src`; `du -sb`6214426 bytes includes directory metadata and Python caches. Already copied; do not confuse with public upstream WLA or repeat the copy.

No H100 training submitted. No further custom-data copying until the user chooses to proceed. Existing 4090 training requires none of these transfers.
