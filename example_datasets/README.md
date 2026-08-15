# example_datasets（远端调试视图）

> 位置：`/data/nfs/share/OmTrackVLA/example_datasets/`。这是**用符号链接拼出来的子集视图**，零拷贝、与全量数据永远同步，专供在本机（g0014）上快速调试 loader / 训练 pipeline。

## 1. 与全量的关系（重要）

| 本目录条目 | 真实位置 |
|---|---|
| `tpt_bench/0000`、`0001` | `/data/nfs/share/OmTrackVLA/data/tpt_bench/{0000,0001}`（全量 47 序列） |
| `sage3d_extracted/0001_83992` | `/data/nfs/share/OmTrackVLA/data/sage3d_extracted/0001_83992`（全量 10 run） |
| `traj_data/3dfront_d435i/00154c06-…` | `/h100-2/vln_n1/traj_data/3dfront_d435i/00154c06-…`（全量 12 组 85k+ episode） |
| `data_loader/` | 符号链接 → `/data/nfs/share/OmTrackVLA/tools/data_loader/` |

因为都是 `ln -s`，改全量数据、改 loader 代码，本目录自动跟着变；删除/重建本目录不伤数据。**不要在 `example_datasets/` 里直接改文件**（会改到全量）。

## 2. 用途 & 怎么用

- 目的：全量 loader 扫描耗时（尤其 NavDP 85k episode、TpT 47 序列、sage 10 run）；子集视图让一次扫描/建 index 在数秒内完成，适合反复调试。
- 跑自测（等价于本机那套，只是路径换成远端）：

```bash
cd /data/nfs/share/OmTrackVLA/example_datasets
export OMTVL_SAGE_ROOT=$PWD/sage3d_extracted
export OMTVL_TPT_ROOT=$PWD/tpt_bench
export OMTVL_NAVDP_ROOT=$PWD/traj_data
PY=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
$PY data_loader/self_test.py --sage-only
$PY data_loader/self_test.py --tpt-only
$PY data_loader/self_test.py --navdp-only
$PY data_loader/self_test.py          # mixed 2:1:1
$PY data_loader/data_stats.py         # 统计
```

代码里：

```python
from mixed import MixedOmniDataset
ds = MixedOmniDataset(
    sage_root="/data/nfs/share/OmTrackVLA/example_datasets/sage3d_extracted",
    tpt_root="/data/nfs/share/OmTrackVLA/example_datasets/tpt_bench",
    navdp_root="/data/nfs/share/OmTrackVLA/example_datasets/traj_data",
)
s = ds.get_sample(as_arrays=True)
```

想测**全量**：把上面三个 root 换成 `…/data/sage3d_extracted`、`…/data/tpt_bench`、`/h100-2/vln_n1/traj_data` 即可，loader 无需改动。

## 3. 调试重点

与 `data_loader/README.md` 的 8 条注意事项一致，摘要：

1. TpT `bbox_qv`/`bbox_source` 是 `[x,y,w,h]`，loader 已转 `[x0,y0,x1,y1]`（`tpt_ds.py`）；`is_exist=0` 时 bbox 全 0。
2. TpT `is_behind_glass ∈ {-1,0,1}`，-1/1 都算玻璃遮挡，按 `abs==1` 归一。
3. Sage3D bbox 是投影值，近距目标头/脚出画被裁到帧边界（y0=0/y1=480）；`_target_crops/` 用于人工抽查。
4. NavDP `pointgoal_valid` 约半数 False（终点在视角外），训练必须按此 flag 过滤/加权。
5. NavDP 同一 index 多次调用随机出不同子段；训练用 `get_sample()`（权重随机），不要用 `get_sample(index=k)`（那是确定分桶）。
6. TpT 无 waypoint/距离监督（`traj=None`、`target_dist=0`），轨迹监督来自 sage/navdp。

## 4. 环境

- conda env `omtrackvla`（Python 3.9）：`/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python`（有 pyarrow 21 / cv2 / numpy / imageio，**无 pandas**，loader 不依赖 pandas）。
- 若改 loader 代码，改在 `tools/data_loader/`（本目录的 `data_loader` 是它的符号链接）。

## 5. 版本记录

| 日期 | 内容 |
|---|---|
| 2026-08-15 | 建立远端调试视图：4 个数据子集符号链接 + data_loader 链接；与本地 `example_datasets`（`D:\researchai\Paper\TASE2026\example_datasets`）结构一致 |
