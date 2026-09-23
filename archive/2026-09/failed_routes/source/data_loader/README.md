# OmTrackVLA Unified Data Loader

统一三数据源（Sage3D 真机跟随 / TpT-bench 视频跟随 / InternNav NavDP 仿真导航）为 OmTrackVLA 可训练样本的懒加载器 + task-balanced 混合器。

## 文件

| 文件 | 作用 |
|---|---|
| `unified.py` | 统一 sample schema 常量、图像读取/resize/letterbox、数组加载 |
| `sage3d_ds.py` | Sage3D 真机 person_follow（读取抽取产物的 `derived.json` + `rgb/`） |
| `tpt_ds.py` | TpT-bench person_follow（读取 `frames.parquet` + `rgb_frames/`） |
| `navdp_ds.py` | InternNav NavDP pointnav/imagegoal（复刻 `navdp_lerobot_dataset.py` 几何核心，pyarrow 读取） |
| `mixed.py` | `MixedOmniDataset`：2:1:1 task-balanced 采样 |
| `data_stats.py` | 三源聚合统计（输出 JSON） |
| `self_test.py` | 自测：拉样本打印 shape/路径（`--sage-only/--tpt-only/--navdp-only`） |

依赖：`cv2 numpy pyarrow`（omtrackvla 环境即满足，无需 pandas）。运行：`/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python`。

## 统一 sample schema（懒加载 dict）

每个 `__getitem__`/`get()` 返回纯路径/标量的 dict，训练侧再用 `load_sample_arrays` 转张量：

| 字段 | 类型 | 语义 |
|---|---|---|
| `task` | str | `person_follow` / `pointnav` / `imagegoal` |
| `ep_id` | str | 源 episode 唯一标识 |
| `instruction` | str | 自然语言指令（sage=index 指令，tpt=desc.txt 全文） |
| `current` | str | 当前帧 RGB 绝对路径 |
| `window` | list[str] | 历史帧路径（最旧→最新，不含当前帧；数量 ≤ memory_size） |
| `target_img` | str\|None | 目标图像：person_follow=bbox 裁剪（sage/tpt 的 `_target_crops/`）；imagegoal=目标时刻整帧 |
| `bbox` | list[4]\|None | 目标 bbox `[x0,y0,x1,y1]`（当前帧像素坐标） |
| `traj` | list[list[float]]\|None | ego-centric 未来 waypoint `[[x,y], ...]`（sage 8 点、navdp predict_size+1 点；tpt 无监督=null） |
| `actions` | list[list[float]]\|None | 未来速度/位移增量（navdp 提供；sage 用 waypoints_ego 差分） |
| `collision` | bool | 该步是否碰撞 |
| `target_dist` | float | 目标距离 m（sage 有，tpt=0） |
| `valid` | bool | person_follow 目标是否可见/在框（sage=visible，tpt=is_exist） |
| `pointgoal` | list[3]\|None | pointnav 终点 `[x,y,theta]`（ego 系） |
| `pointgoal_valid` | bool | 终点在当前视角是否可见（navdp 投影） |
| `future_rgb` | str\|None | 未来时刻观测帧路径 |
| `depth` | str\|None | 当前深度图路径（navdp） |
| `extra` | dict | 各源补充：`dis_to_human/success/following_rate/facing/target_local`(sage)、`is_behind_glass/interpolated/history_motion/vid_pts_ms`(tpt)、`start/target/mem`(navdp) |

### 数组加载

```python
from unified import load_sample_arrays
arrs = load_sample_arrays(sample, img_size=224, memory_size=8)
# arrs['current']    (224,224,3) f32 [0,1]   letterbox
# arrs['window']     (8,224,224,3) f32       历史窗，不足 8 帧前面补零
# arrs['target']     (224,224,3) f32|None
# arrs['future_rgb'] (224,224,3) f32|None
# arrs['depth']      (224,224,3)|(224,224,1) f32 米（>/5m、<0.1m 置 0）
```

## 各数据集 API

```python
from sage3d_ds import Sage3DDataset
sage = Sage3DDataset("/data/nfs/share/OmTrackVLA/data/sage3d_extracted", step_stride=1)
print(len(sage))                 # episode 数
s = sage.get(i, k)               # 第 i 个 episode 的第 k 个 derived step
```

```python
from tpt_ds import TpTDataset
tpt = TpTDataset("/data/nfs/share/OmTrackVLA/data/tpt_bench", step_stride=4, history=8, future_offset=12)
s = tpt.get("0000", row_idx)     # seq + parquet 行号
```

```python
from navdp_ds import NavDPDataset
nav = NavDPDataset("/h100-2/vln_n1/traj_data", condition="pointnav", memory_size=8, predict_size=24)
s = nav.get(i)                   # 每个调用随机抽 start/target/mem
```

## 混合器

```python
from mixed import MixedOmniDataset
ds = MixedOmniDataset(
    sage_root="/data/nfs/share/OmTrackVLA/data/sage3d_extracted",
    tpt_root="/data/nfs/share/OmTrackVLA/data/tpt_bench",
    navdp_root="/h100-2/vln_n1/traj_data",
    weights=(2.0, 1.0, 1.0),     # person_follow : pointnav : imagegoal
    seed=0, image_size=224, memory_size=8,
)
s = ds.get_sample()              # 按权重随机选源+样本（推荐训练用）
s = ds.get_sample(index=k, as_arrays=True)  # 确定性分桶（person_follow→pointnav→imagegoal）
len(ds)                          # pf_units + navdp_eps*2
```

混合器构建时枚举 `pf_units`（sage 每个 derived step + tpt 每 video frame 行）与 navdp episode 数；navdp 每次 `get` 内部随机抽样（start 在轨迹前半随机、target 更靠后、memory 在两者间），与 InternNav 原 loader 行为一致。

## 已知注意事项（务必阅读）

1. **TpT bbox 格式**：`GTs.json`/parquet 的 `bbox_qv` 是 `[x,y,w,h]`（quickview 960×400 坐标），loader 已转 `[x0,y0,x1,y1]`。`is_exist=0` 时 bbox 全 0，`valid=False`。
2. **TpT is_behind_glass**：源值 ∈ {-1,0,1}，-1 与 1 均为玻璃遮挡（约 17% 帧），loader 以 `abs==1` 标记。
3. **TpT 无 waypoint/距离监督**：只有 ODOM（~2Hz）与 2D bbox，无世界坐标/速度，故 `traj=None`、`target_dist=0`。轨迹监督主要来自 sage3d（8 点 ego waypoint）与 navdp。
4. **Sage3D bbox 为投影值**：`derived.json` 的 bbox 由世界坐标→相机投影得到（相机在 robot 坐标 +0.3m、identity 旋转、go2_d435i 实为 640×480）。目标很近（dis_to_human 中位 1.15m）时头/脚可能出画，bbox 上下界被裁剪到帧边界（y0=0 或 y1=480）。训练前建议抽查 `_target_crops/` 诊断。
5. **Sage3D waypoints_ego**：8 点、包含原点；step 0 可能为空列表（loader 会补全为 8 点）。
6. **NavDP goal 可见性**：`pointgoal_valid` 用与 InternNav `process_pixel_goal` 相同的投影判断；约一半抽样目标在当前视角外 → 训练时用该 flag 过滤/加权。
7. **NavDP 每 ep 随机抽样**：同一 index 多次调用结果不同（复刻 InternNav 行为）；DataLoader 需配合 seed 或仅作无限流使用。
8. **0002 缺失**：TpT seq 0002 源数据 ODOM.zip 本身无该序列，已排除。

## 自测

```bash
PY=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
$PY self_test.py --sage-only
$PY self_test.py --tpt-only
$PY self_test.py --navdp-only
$PY self_test.py            # mixed 全量
$PY data_stats.py > stats.json
```
