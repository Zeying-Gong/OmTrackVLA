# OmTrackVLA 远程数据准备 —— 数据统计报告

> 采集时间：2026-08-14（远程机 172.16.38.52）。工具：`tools/data_loader/data_stats.py`。
> 仓库：`/data/nfs/share/OmTrackVLA`。共享数据根：NFS `/h100-2`、本地条带 `/data`（7.0T，余 2.2T）。

## 1. 总览

| 数据源 | 任务 | 样本单元 | 关键规模 |
|---|---|---|---|
| TpT-bench | person_follow | 每 video 帧 | 47 序列 / 141,326 帧 / 117,611 有效 |
| Sage3D 真机 | person_follow | 每 derived step | 10 tar / 126 ep(125 accepted) / 37,800 step |
| InternNav NavDP | pointnav + imagegoal | 每 episode（内部随机抽子段） | 12 group / 85,124 episode |

统一 loader 生成的可寻址样本总量（`MixedOmniDataset.__len__`）：**564,402**
（person_follow 171,330 + pointnav 196,536 + imagegoal 196,536）。

## 2. TpT-bench（视频级 person_follow）

- 数据：`data/tpt_bench/{0000..0047}`，每序列 `frames.parquet`（GT×video 对齐）+ `rgb_frames/frame_{idx}.jpg`（quickview 960×400 @30fps）+ `desc.txt` + `meta.json`。
- **47/48 序列**：0002 因源 `ODOM.zip` 内无该序列文件而跳过（数据提供方缺失）。
- 帧分布：141,326 视频帧；目标可见（`is_exist=1`）117,611（**83.2%**）；玻璃遮挡（`is_behind_glass=±1`）24,170（**17.1%**）；`interpolated=0`。
- 几何质量（meta.json 汇总）：目标中心 median 位移 3.5px，bbox in-frame 比例均值 **0.824**。

| 目标 | 说明 |
|---|---|
| current / window / target crop `[x,y,w,h]→[x0,y0,x1,y1]` | 已实现 |
| `valid = is_exist` | 已实现 |
| `is_behind_glass`（-1/1 均算） | 已实现（loader `abs==1`） |
| `history_motion`（ODOM→robot 系位移） | 已实现（有 ODOM 跳变时） |
| ego waypoint / 距离监督 | **无**（源只有 2D bbox + ~2Hz ODOM） → 样本 `traj=None` |

## 3. Sage3D 真机跟随（person_follow）

- 数据：`data/sage3d_extracted/{run_id}/{mode}/{ep}/{cam}/`，含 `rgb/`、`depth/`、`derived.json`、`camera_info.json`、`index.json`。
- 抽取规模：**10/11 tar**（全部正式 run，缺 0008 不存在；尚未抽取 0012+ 后续 tar）。输入 3.2G，源 tar 位于 `/data/nfs/share/sage3d_track_data`。
- 覆盖：modes `{at, dt, stt}`，cameras `{dingo_realsense_d435i, dingo_zed, g1_realsense_d435i, go2_realsense_d435i}`。
- Episodes：**126（125 accepted，1 rejected）**；每 ep 300 step（~30s@10Hz）→ **37,800 step**。
- 质量：
  - 目标可见率 99.7%，碰撞 0 帧（sim 未触发碰撞标志）。
  - `dis_to_human`：min 0.54 / median **1.15** / max 3.88 m。
  - 每 step 提供 8 点 ego-centric waypoint（step0 可能为空已补全）。

## 4. InternNav NavDP（pointnav / imagegoal）

- 数据：`/h100-2/vln_n1/traj_data/{group}/{scene}/{videos|data|meta}/`（NFS，不复制）。
- 规模：**12 group × {d435i, zed}，85,124 episode**；每 ep 轨迹长 123~320 步（中位 169）；全部含 rgb + depth + `action`(4×4 extrinsics) + intrinsic/extrinsic + episodes_stats。
  - 分仓：gibson_d435i 18,540｜hm3d_d435i 13,808｜gibson_zed 13,093｜3dfront_d435i 7,894｜hm3d_zed 7,642｜hssd_d435i 6,810｜3dfront_zed 6,472｜hssd_zed 3,182｜matterport3d_d435i 3,370｜matterport3d_zed 2,841｜replica_d435i 835｜replica_zed 637。
- 每 episode 内部随机抽 (start, mem, target) 子段 → 每次 `get` 新样本。
- 输出：`pointgoal [x,y,θ]`（ego 系轨迹终点）、目标可见 flag、`target_img`=目标时刻整帧、`traj`（predict_size+1 点）+ `actions`（增量）、`window`（memory_size）、`depth`。
- 复刻来源：`/h100-2/InternNav/internnav/dataset/navdp_lerobot_dataset.py`（relative_pose / xyz_to_xyt / process_actions），仅产出 unified dict，不含点云 critic。

## 5. 生成样本单位 & 任务平衡

| 任务 | 来源 | 单元 | 数量 |
|---|---|---|---|
| person_follow | TpT | 每 video 帧 | ≈141,326 |
| person_follow | Sage3D | 每 derived step | ≈37,800（175,000 内）|
| pointnav | NavDP | 每 episode（随机子段） | 85,124 |
| imagegoal | NavDP | 每 episode（随机子段） | 85,124 |

混合器 `weights=(2,1,1)`，随机采样路径按 person_follow : pointnav : imagegoal = 2:1:1 抽取源；person_follow 内部 sage/tpt 按各自单元数轮询。

## 6. 已知注意 / 待办

1. TpT：quickview 与 GTs bbox 坐标系对应关系未获作者确认（meta.note 已记录）；0002 源缺失。
2. Sage3D：bbox 为投影值，近距目标出画被裁剪（y0=0/y1=480），建议训练前人工抽查 `_target_crops/`；后续可补 0012+ tar。
3. NavDP：goal 可见性约半数 False，训练需按 `pointgoal_valid` 过滤/加权。
4. 磁盘：sage 抽取 3.2G；TpT 21G；NavDP 复用 NFS 原目录，无额外拷贝。`/data` 余 2.2T。