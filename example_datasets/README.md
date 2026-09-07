# Example datasets

This directory documents the small debugging view maintained on `g0014` at:

```text
/data/nfs/share/OmTrackVLA/example_datasets
```

Raw samples are intentionally not tracked in this public repository. The data structure and audited fields are documented in [`docs/data_inventory.md`](../docs/data_inventory.md), while `view_inventory.json` records why the old view cannot safely be copied verbatim.

## Why the old directory is not uploaded as-is

The old directory is a cluster-local view made from five absolute symbolic links. A Git commit would preserve those links as machine-specific, broken paths rather than copy their targets. Dereferencing them would instead add about 1.80 GB and 27,750 files, including generated target crops, caches, and a stale TpT source (`tpt_bench` rather than the authoritative `tpt_bench_clean_v2`).

The destination repository is public, no Git LFS client is installed on the source machine, and no redistribution license was found beside the three dataset roots. Therefore this branch publishes schemas, counts, audit code, and path contracts only. It does not republish third-party RGB/depth frames, videos, point clouds, or Parquet payloads.

## Authoritative layouts

### InternData-N1

```text
/h100-2/vln_n1/traj_data/
  <group>/<scene>/
    meta/{info.json,episodes.jsonl,episodes_stats.jsonl,tasks.jsonl,pointcloud.ply}
    data/chunk-000/episode_XXXXXX.parquet
    videos/chunk-000/
      observation.images.rgb/episode_XXXXXX_YYY.jpg
      observation.images.depth/episode_XXXXXX_YYY.png
      observation.video.rgb/episode_XXXXXX.mp4
      observation.video.depth/episode_XXXXXX.mp4
```

The Parquet episode contains camera intrinsics/extrinsics and action matrices. Instruction text in JSONL is metadata and is prohibited from model inputs.

### SAGE3D extracted

```text
/data/nfs/share/OmTrackVLA/data/sage3d_extracted/
  index.json
  <run>/<mode>/<episode>/<camera>/
    {episode}.json
    {episode}_info.json
    camera_info.json
    derived.json
    quality.json
    _ACCEPTED
    rgb/00000.jpg
    depth/00000.png
```

Only paths that are in the root index, carry `_ACCEPTED`, and have `quality.status=accepted` may enter a training manifest. `derived.json.steps` contains the projected bbox, visibility, robot/target pose, and optional 8-point ego waypoint.

### TpT clean v2

```text
/data/nfs/share/OmTrackVLA/data/tpt_bench_clean_v2/
  <sequence>/
    frames.parquet
    meta.json
    desc.txt
    rgb_frames/frame_XXXXXX.jpg
```

`frames.parquet` maps each `video_idx` to bbox/visibility, video and GT timestamps, and ODOM. Generated `_target_crops` and `_target_refs` are diagnostics, not source samples. `desc.txt` and per-frame labels are not model inputs.

## Cluster use

Use the authoritative roots above for reproducible work. To inspect them without changing source data:

```bash
cd /data/nfs/share/wam_tracking/OmTrackVLA
PY=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
$PY scripts/audit_data_inventory.py --manifest configs/data_inventory.json
```

The audit performs deterministic stratified media decoding and metadata/path checks. It never writes into a dataset root or creates target crops. If actual media must later be published, add it only after the data owner records redistribution permission and approves a separately reviewed, self-contained export.
