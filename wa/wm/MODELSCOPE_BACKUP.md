# Private ModelScope backup and H100 download

User authorized backup/migration alongside the existing 8-RTX4090 JEPA-WM training. Do not stop Job59566 or submit duplicate training for this transfer.

Repository: https://modelscope.cn/datasets/a597836509/wa-evt-jepa-private-backup-20260929

Verified private=true and license=other from both development machines. Original asset rights remain unchanged. The earlier empty `wa-evt-jepa-backup-20260929` repository received a platform default Apache label and is NOT used; API deletion is unsupported, so it remains empty. No data was uploaded there.

Plan:1051170 files,163832726558 source bytes,150 tar archives. Most shards hold up to1GiB payload; the original WLA4.12GB checkpoint occupies its own larger archive. Tar headers add overhead. Public JEPA/DINOv2 weights, environment packages and GitHub sources are excluded. Custom WLA source and audit indices are included. Dataset scene split and original file bytes remain unchanged.

Script: `wa/tools/modelscope_backup.py`. Uses the SDK's existing login; no token arguments or credential copying. Upload validates repository privacy before each archive. SHA256 and byte counts are published after each successful archive commit in `manifest.json`; `complete=true` is uploaded only after all archives succeed. H100 verifies each archive before registering it. Transfer is a NAS I/O task, not a GPU training job.

Baidu root: `/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/modelscope_backup_v1`
- `plan.json`, `manifest.json`, `upload.log`, `shard-*.tar` and input file lists.
- Low-priority uploader started with `nice19`, `ionice2/7`, `flock`; initial wrapper PID4155449.

H100 root: `/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/modelscope_download_v1`
- `download.log`, `verified.json`, `metadata/manifest.json`, `archives/shard-*.tar`.
- Downloader guarded by `flock`; initial wrapper PID94355.

Current evidence: first archive uploaded and H100 downloaded/SHA256 verified:1086750720bytes /7269files, SHA256 `7ab336a6cb7628c1a5793124564ee0e5a60fdeae30a2ec5de6a2e79d74bc10e0`.

Restart only after checking the lock/process and logs. Existing archives plus manifest provide resume. An interrupted `.partial` archive is rebuilt. SHA/plan mismatches fail closed; do not bypass validation. Downloaded archives are not yet extracted into an active training dataset: safe restore and original cache/episode checks are separate required steps. Do not use remnants of the abandoned SSH-stream migration as validated data.

Completion requires both source `manifest.json` and destination `verified.json` with complete=true and150 archives. A successful first shard is not full backup completion.
