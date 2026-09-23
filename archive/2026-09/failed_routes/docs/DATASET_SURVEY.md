# 三数据集实际结构调查报告（sage3d / tpt-bench / vln_n1）

> 调查时间：2026-08-14。方式：远程 g0014 (`10.9.2.14`) 实测抽查，
> 替换了 `OmTrackVLA-AI接入说明.md` 中原笔记里与事实不符的地方。
> 目标：为 person_follow / pointnav / imagegoal 三类任务的数据接入（OmTrackVLA 架构 §14）提供依据。

---

## 0. 环境与连接要点（继承自接入说明）

- Python 环境：
  - `omtrackvla` = `/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python`（Python 3.9，habitat_sim；**无 pyarrow/pandas**）。
  - `internnav` = `/data/nfs/share/gzy/miniconda3/envs/internnav/bin/python`（pyarrow 21.0 / pandas 2.3.1 / torch / cv2 / open3d）——**读 vln_n1 parquet/video 必须用它**。
- 相关仓库：
  - `/data/nfs/share/OmTrackVLA`（本代码库，含 `make_tracking_data.py`、`data/datasets/track/`）。
  - `/h100-2/InternNav`（InternNav + `internnav/dataset/navdp_lerobot_dataset.py`，可直接照抄的 NavDP loader）。

---

## 1. sage3d_track_data（仿真行人跟随，STT/DT/AT）— 与笔记一致，+quality.json +debug_occ +logs

路径 `/data/nfs/share/sage3d_track_data`，共 **912** 个 `formal_runs_XXXX.tar.gz`（+`README.md`、`test_tiny/`），合计约 4.4T。
抽查 `formal_runs_0001_83992.tar.gz`（42394 条目）确认内部布局：

```
0001_83992/tmp_episodes/{stt,dt,at}/0001_839920/episode_{0..9}.json   # spawn 定义
0001_83992/{stt,dt,at}/{ep}/{robot}_{cam}/
    rgb/00000.png …                  # RGB 1280x720 PNG，~800KB/帧
    depth/00000.png …                # depth 1280x720 PNG，~50KB/帧
    debug_occ/00000.png …            # 占用网格调试图（数据侧用不到）
    rgb_video.mp4  depth_video.mp4   # 与 png 帧冗余（抽帧后可不留）
    <ep>.json                        # 任务筛选字段（见下）
    <ep>_info.json                   # 每步 GT（数组）
    camera_info.json                 # 内参+外参，可投影 target_pos→像素
    quality.json                     # 采集质量自检（accepted/metrics/filter_config）
    track_object.jpg  _ACCEPTED      # _ACCEPTED 为空文件 mark
0001_83992/logs/{stt,dt,at}/{robot}_{cam}/*.log
```

robot × cam = {dingo, g1, go2} × {zed, realsense_d435i} = **6 组合，但每 ep 不全有**
（本 tar：stt/9 仅 1 组合，dt/4 无 dingo）。抽取脚本必须容错缺组合。

### 1.1 关键 JSON schema

`<ep>.json`（任务筛选字段，方案 B 用）：
```
{ "finish": true, "status": "Normal", "success": 1.0,
  "following_rate": 0.893…, "following_step": 268, "total_step": 300,
  "collision": 0.0, "episode_id": 0, "ori_episode_id": 0,
  "mode": "stt", "instruction": "Follow the woman." }
```
DT 例：`"Pursue the woman wearing teal top, black pants, maroon shoes."`；`ori_episode_id` 与 mode 内 ep id 不同（跨模式同源）。

`<ep>_info.json` 为**数组**，每步含：
`step, source_fps(30), dt(0.0333), character_speed, dis_to_human, facing(1/0),
base_velocity, base_velocity_cmd, slide, navmesh_collision, collision,
robot_pos_pre, robot_yaw_pre, robot_pos, robot_yaw, target_pos, other_humans_pos`
（`other_humans_pos` 固定 7 个，空位填 `[-98,-98,-98]` 哨兵）。

`camera_info.json`：
```
{ model:"pinhole", axes:"ros", width:1280, height:720,
  intrinsics:{fx:521.846…,fy:521.846…,cx:640.0,cy:360.0, k:[[…]]},
  extrinsics_robot_to_camera:{ "translation":[0,0,0.3], "camera_link":"base" } }
```
→ **方案 B 的“target_pos 投影像素 bbox”可行**，无需另跑检测。

`tmp_episodes/episode_X.json`：robot `start_pos/goal_pos/start_orientation`、
characters `spawn_positions/commands(GoTo + path)`（即笔记说的 episode 定义）。

单 ep 规模：300 rgb + 300 depth 帧 @30fps（10s）、64 个组合目录/文件的量级
可根据 tar 内 `6.json`/`quality.json` 抽一抽统计，不必全解压。

注意：`test_tiny/` 是**原分发格式**样例（`10.mp4 / 10_info.json / 10_left|rear|right|third_person|traj.mp4 / track_object.jpg`），
与 tar 内布局不同（无 mode 分目录、无 depth），**别拿它当 tar 内容参照**。

---

## 2. tpt-bench（真机行人跟踪）— 与笔记的**最大出入**：整套还没有解压

`/h100-2/tpt-bench` 下**不是** 48 个 sequence 目录，而是 7 个 zip：

| 文件 | 大小 | 内容 |
|---|---|---|
| GTs.zip | 119M | `GTs/0000.json … 0047.json`（48 个） |
| ODOM.zip | 2.8M | `ODOM/0000.txt … 0047.txt`（TUM） |
| descriptions.zip | 42K | `descriptions/0000.txt … 0047.txt` |
| quickview_videos.zip | 7.4G | `quickview_videos/0000.mp4 … 0047.mp4` |
| panoramic_images.zip | 591B | 仅 2 个 Error 文本（**有效数据为空，符合预期**） |
| evaluation_results.zip | 8.8G | `evaluation_results/0000/<18 个 tracker 方法>.json`（评测输出，**训练用不到**） |
| OneDrive_1_2026-8-12.zip | 16.4G | = LICENSE + descriptions + GTs + ODOM + quickview + evaluation_results + Error 文本 的**冗余打包** |

### 2.1 解压后内容核实

- `GTs/0000.json`：`ts_ns → {is_exist, is_behind_glass, bbox:[x,y,w,h], interpolated}`。
  实测 **~30fps**（中位帧间隔 33.7ms，不是笔记的 ~25fps）。seq0000 共 31104 帧、跨度 ≈1049s。
- `ODOM/0000.txt`：TUM 8 列 `ts tx ty tz qx qy qz qw`，**间隔 ~2Hz**（中位 500ms），
  比 GTs/视频稀疏得多 → **必须按时间戳插值对齐**（方案 C 已写对）。
- `descriptions/0000.txt`：目标衣帽/鞋/体态 + 场景列表 + 消失次数（如 seq0000：长 1036.8s、消失 14 次）→ 可作 instruction。
- `quickview_videos/0000.mp4`：**960x400@30fps，仅 7776 帧（≈259s），为 GTs 的 1/4**。

### 2.2 必须修正的两处坑（影响 target_bbox 与 RGB 对应关系）

1. **分辨率**：seq0000 的 GTs bbox 右缘 881+154=1035 > 960 → bbox 标注在**原始全分辨率**坐标系，
   quickview 是**降采预览**。**不能直接把 GTs bbox 叠到 quickview 帧上**。
2. **帧数**：video 7776 帧 vs GTs 31104 帧（1/4）→ 视频是**降采+抽帧后的 quickview**，
   帧与帧之间的时间对应关系需要额外的映射（或视频只覆盖部分片段）。

结论：tpt-bench 作为 person_follow **微调/评测集**时，若作者不补发全分辨率视频，
就只能用 quickview 预览 + 自跑检测出 bbox，或用 GTs 但不与 quickview 像素一一对齐。
接入前建议先确认 GTs bbox 与 quickview 帧的对应方式。

---

## 3. vln_n1/traj_data + InternNav NavDP — 与笔记一致，全部验证通过

- 14 个组 = {hm3d, gibson, 3dfront, hssd, **matterport3d**, replica} × {zed, d435i}。
  场景数合计 = **3730**（hm3d_zed 633 / hm3d_d435i 590 / gibson_zed 493 / gibson_d435i 473 /
  3dfront_zed 657 / 3dfront_d435i 522 / matterport3d_zed 66 / matterport3d_d435i 65 /
  hssd_zed 88 / hssd_d435i 121 / replica_zed 11 / replica_d435i 11）。
  全部单 chunk（`chunk-000`）。
- 场景布局（样例 `hm3d_zed/00001-UVdNNRcVyV1`）：
  - `data/chunk-000/episode_{000000..}.parquet`（每 ep 一个，9 个/场景）
  - `videos/chunk-000/observation.images.{rgb,depth}/episode_{i}_{frm}.jpg`（每帧一张）
  - `videos/chunk-000/observation.video.{rgb,depth}/episode_{i}.mp4`
  - `meta/{info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl, pointcloud.ply}`
- `meta/info.json`：v2.1，fps 30，`data_path`/`video_path` 模板与 loader 使用的路径结构一致。
- `meta/episodes_stats.jsonl`：每 ep 的 `task_index{min,max,count}`、`image_index{min,max,count}` —— loader 切帧的依据。
- `meta/episodes.jsonl` / `tasks.jsonl`：VLM 子任务指令（sub_instruction / revised_sub_instruction / sub_indexes）。

### 3.1 parquet 真实 schema（用 internnav 环境读，pyarrow 21）

```
num_rows: 192（=该 ep 帧数）
columns: ['index', 'observation.camera_intrinsic', 'observation.camera_extrinsic', 'action']
```
- 三列都是 `list<element: float>`，各 16 元 → reshape 3x3 / 4x4 / 4x4。
- **没有 observation.image 列**——图像全部在磁盘 jpg。笔记所提 ImageGoal 的 target image 来自 rgb 目录。
- `action` = 相机位姿轨迹（世界→相机 4x4），相邻步平移 ~0.029m；`extrinsic` z≈1.296（zed）/ 1.338（d435i）。
- `camera_intrinsic`：zed fx≈168 / d435i fx≈356（cx=240, cy=135）→ **跨组千万别混内参**。

### 3.2 NavDP loader（照抄即用）

`/h100-2/InternNav/internnav/dataset/navdp_lerobot_dataset.py`，`NavDP_Base_Datset.__init__` 流程：
`root_dirs → group_dir/scene_dir/data/<chunk>/*.parquet`、
`meta/episodes_stats.jsonl` 的 `image_index.min/max` 切 `observation.images.rgb|depth` 帧列表、
`meta/pointcloud.ply` 过滤蓝色点（`color_distance < 0.05` against [0,0,0.5]）当障碍。
`__getitem__` 返回语义：`memory_images(8) + depth + point_goal(xyt 终点) +
image_goal(concat(start_img, target_img), 6ch) + pixel_goal(image+mask) + pred_actions(xyt 增量)`。
→ **PointGoal / ImageGoal 两种条件天然覆盖**。

---

## 4. 现有可用资产（对应组织方案 A）

- `OmTrackVLA/data/datasets/track/{STT,DT,AT}/{train,val}/train.json.gz|val.json.gz`
  每个是 habitat episode 配置（episode_id / scene_id / start_pos/rot / info 等），约 4.4MB。
- `habitat_track_data/sim_data0420/{stt,dt,at}_train_oracle/seed_1000/pass1/<scene>/<ep>/`
  每个 ep：`<ep>.mp4 + <ep>_info.json + <ep>.json + track_object.jpg + <ep>_left|rear|right|third_person|traj.mp4`。
  计数：stt 675 / dt 688 / at 692。与 `make_tracking_data.py` 的输入约定一致（可复用它抽帧转 mp4）。
- `make_tracking_data.py` 期望输入={mp4, <id>_info.json, track_object.jpg}，
  输出 `{seed}/{scene}/{id}/mp4 + info.json + track_object.jpg`——即 sim_data0420 已经是目标结构。

---

## 5. 结论与修正清单（相对原笔记）

| # | 项目 | 原笔记 | 实测 | 影响 |
|---|---|---|---|---|
| 1 | tpt-bench 存放形态 | “48 个 sequence 目录” | 7 个 zip（OneDrive 冗余包在内） | 先解压；不问作者要全分辨率视频前，bbox/RGB 不能直接对齐 |
| 2 | tpt GTs 帧率 | ~25fps | **~30fps** | 插值参数按 30/33ms |
| 3 | tpt ODOM 频率 | （未提） | **~2Hz** | 必须插值到 video/GT 时间戳 |
| 4 | tpt quickview 视频 | “RGB 视频” | 960x400、帧数=GT 的 **1/4**、bbox 坐标系不一致 | target_bbox 需另行处理 |
| 5 | sage3d 每 ep 组合 | “6 种 robot×cam” | 6 种**但不全**（部分 ep 缺组合） | 抽取脚本要容错 |
| 6 | sage3d 文件 | 笔记缺 quality.json / debug_occ/ / logs/ | 存在 | quality.json 可作额外筛选参考 |
| 7 | vln parquet | observation 内嵌图像 | **无图像列**，4 列纯姿态/intrinsic | 明确 loader 只走磁盘 jpg |
| 8 | python 环境 | omtrackvla 全功能 | **无 pyarrow/pandas**；读 vln 需 internnav env | loader/预处理用 internnav |

整体结论：方案 A（统一 loader+task-balanced 混合器）、B（选择性解压）、C（tpt 真机）全部可行、无阻塞；
**实施顺序建议**：
1. sage3d 选择性解压 + 样本生成脚本（plan B，产出 8 步 waypoint + target 像素 bbox + visible/collision）；
2. InternNav NavDP 风格三任务统一 loader/混合器（plan A，person_follow:pointnav:imagegoal = 2:1:1）；
3. tpt-bench 解压 + ODOM→video→GT 对齐，作 person_follow 真机微调/评测（plan C，需先和作者确认 quickview 与 GTs 的对应方式）。