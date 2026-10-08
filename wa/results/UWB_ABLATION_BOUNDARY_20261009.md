# WA 同权重无 UWB 配对边界

状态：STATIC_REVIEW_ONLY。2026-10-09在Git HEAD `8d2196bcba0c66c4a04ee6807d4b12c18e188234`读取当前实现；没有改生产代码、运行GPU预测、仿真轨迹或新训练。Job62256仍在训练，本研究须等同一candidate在1405条/任务达到STT≥1289、DT≥1173、AT≥1203后执行。

## 研究问题

用同一checkpoint、同一DINO和同一评测协议，将mixed改为显式image，测量移除当前理想模拟极坐标UWB后的性能变化。保留episode0的GT人物BBox模板、RGB、因果历史、实际观测时间和预测几何控制，不提供后续GT框、文本或未来目标路径。这不是“从未用过UWB训练的纯视觉模型”，也不是完全无GT初始化。

`mixed`加`noise_mode=zero`仍输入理想UWB，zero只把Flow采样的初始噪声置零。把mixed中的极坐标改为零也不等于image：mixed仍标为有效测量，方位cos项及可用性项非零。无UWB配对必须使用`WA_EVAL_MODE=image`，保持zero采样不变。

## 静态数据通路

| 位置 | 已读实现及结论 |
| --- | --- |
| eval_server.py:22–25、41–55 | image请求只允许RGB、时间和首框，额外uwb字段会被拒绝；内部polar零张量、mode_id0；只有mixed读取测量及数据年龄。 |
| training.py:40–43、58–59 | mode0使uwb_valid为False；RGB与模板编码后进入policy.predict，没有额外几何真值或世界模型预测输入。 |
| adapters.py:24–32、84–96 | 无效polar与age先置零，全部五维特征为零；UWB prompt仍构造，但其valid=False。 |
| adapters.py:45–52 | self-attention和visual-write将该prompt作为key遮蔽；read/reread逐query读取visual，返回只取有效query。未见测量值流向有效policy token的路径；固定缺失状态不等于当前测量泄漏。 |
| diagnostic_agent.py:18–22 | image不创建SimulatedUWB，也不采样或附加UWB到RPC。 |
| student_eval_contract.py:6–15 | image必须显式evaluation-set-adaptation，强制learned_yaw_guard_v1，禁止复用旧mixed/targeted/resume结果。 |
| eval_agent.py:14–30、diagnostic_agent.py:52–58 | 仅首步从detector取得模板框；动作由预测xy/yaw及预测target_geometry生成。外部target_control为纯参数函数，不读取env、目标真值或UWB。 |
| diagnostic_agent.py:25–42、review_recorder.py:14–38 | 初态/世界位置/语义可见性用于证据和独立canvas/日志；没有修改送入策略的原RGB。日志或视频仍含GT遥测，不应误认为这些字段是image策略输入。 |
| initial_bbox_repair_agent.py:8–14、eval_full_mixed.py:73–94 | 两种mode保留同一七框修复、seed7、初RGB和动态初态证据；初始化失败不剔除。 |

WA覆盖了外部WLAAgent的act，旧父类发送instruction的act不执行；父类初始化、reset和bind只处理RPC及运动尺度/时间。JEPA参数仍在checkpoint，当前predict不调用world predictor。模型权重携带的训练信息不等于当前UWB测量输入。

## 真实运行前仍需验证

以下是待执行准入检查，不是已通过结论。使用最终已验收candidate、冻结选择及安全开发GPU，不在集群单独提交smoke。

1. 真实image RPC应拒绝额外uwb字段；捕获全部请求字段，确认首框仅首步、无后续目标/文本/世界坐标。reset后历史、模板、step和噪声种子一致；拒绝请求不能悄悄污染会话。
2. 在受控的相同RGB、模板、时间及噪声fixture上，观测make_conditions和adapter：image的mode_id0、uwb_valid=False、polar/age零、prompt mask无效。用只读observer hook或诊断包装，不改冻结算法。
3. 无效分支的极坐标/年龄扰动（含NaN）不应改变条件、预测和动作。该测试需确保修改的不是合法mixed输入。mixed正控须确认测量改变了有效输入条件，但不强求每次扰动都改变动作；某个固定输入预测相同不自动证明未使用UWB。
4. 检查真实agent image路径没有创建/调用传感器，控制器取的是预测几何；observer开关在固定状态下不改变RPC载荷和动作。observer极坐标日志可以存在，必须与policy输入证据分离。
5. 保留同一首框模板规则。底层Session mixed允许缺模板，但实际WAAgent会将非法首框记初始化失败；image不能通过换框、过滤失败或引入额外模板降低分母。

## 全量配对验收

三项mixed SR达标后，绑定其实际checkpoint SHA/step及完整审计结果；不从best61609或中间权重选择性拼接。image使用全新输出，各STT/DT/AT完整1405，共4215；资源允许时各8GPU共24卡，否则按获准平台全任务安排。每任务8个COMPLETE、ready/image身份和4215唯一task/key必须核对。

逐例对齐mixed/image的checkpoint、场景/episode、seed7、initialRGB、动态初态、首框、语义/七框修复、控制/物理及成功标准。闭环路径随后可以不同，不能要求之后每帧RGB一致。输出每任务SR、reference-step归一TR、单列macro_TR、CR及初始化失败；保留全分母，并列共同成功/仅mixed成功/仅image成功/共同失败及对应视频。CR仅目标人距离曾小于0.5m，不是一般障碍接触。

结论限于当前evaluation-set adaptation权重对理想UWB输入的依赖；没有真实UWB噪声/多径泛化或无UWB重训结论。只有取得该配对证据后才能讨论是否需要另立无UWB重训实验，不能把直接移除输入的降幅等同纯视觉训练上限。

## 本轮读取的身份

下列SHA只绑定本次静态检查的实际文件。将来冻结评测源码时必须核对，不能仅凭本表宣称运行路径已验证。路径未写绝对前缀者均位于checkout；外部文件位于`/data/nas_ray/home/zeying.gong/algorithm/repos/WLA-EVT-20260925/evt_text_action_v3/`。

| 文件 | SHA256 |
| --- | --- |
| wa/wm/eval_server.py | 49d7b0b5b60f98a479a870f65821ec0a0b0a69f8b60cf9f5a5a6c9cd9b217fb1 |
| wa/wm/training.py | 43e2705aeb1175ae5e73225587d0d8caa5cdea6deff43ecec4150721e8ea2bc8 |
| wa/wm/adapters.py | 24d239540315dba258759e14953e61a3a3fe7e9676f7a36e8a58dc672063507a |
| wa/wm/diagnostic_agent.py | 6ea51590ad4a187ac8fd89cb2bf06c9095c72994a98f8ad3ad1142fa0408678a |
| wa/wm/eval_agent.py | 2027cbfcaf82a44ab2a914bb1fcf34a845c009870142d7cf7015b95d75a7dd98 |
| wa/wm/learned_yaw_control.py | cf917b6657116bfaa78743b444689c4ee818a0b75189c9c8c8ffb1012deffbc9 |
| wa/wm/student_eval_contract.py | fdc7e9ad6c9aa14fd1c09ecfdc810f7b32a51f41eaeaf9381080a5e503886dfd |
| wa/wm/review_recorder.py | 1a079b5336bc09568fae84b2c4e04328e700536e5c5370b401ed77c2d025afaf |
| wa/wm/eval_full_mixed.py | 6d4f27759b8e14c385d2bd4aa1a502c6e0fd2cf4e0aee77753ec30cb7531b647 |
| wa/wm/initial_bbox_repair_agent.py | 5755a725f11c71fa2f3a89bd918d1c2e360a084a02cd0749d8d0dd1fd2558472 |
| wa/tests/test_uwb_eval_mode.py | 0b5417a6211ff9b6707401b6a973e42e19323e4efc3ba8b42f40147eec302a01 |
| 外部 control.py | 90b8b7492371f95016f3516cb26e035e09b675bbc276bfa27ac3fb3f895c52d5 |
| 外部 eval_agent.py | 41db78b9e56791df1d663eed5148b267fceb8037f300970b1d02c74613a7c397 |

已有test_uwb_eval_mode使用CPU合成fixture，覆盖模式/身份/复用/分母等边界；本轮只读，没有重跑或把它当成模型/真实observer结果。另一个审阅者的SSH在kex被reset、exit255，未重试；主线程同期A800只读成功。该审阅者独立分析主线程读回的片段，未发现需先修复的静态UWB旁路；结论仅为STATIC_REVIEW_ONLY / REAL_INTERFACE_UNVERIFIED，不是独立远程取证或运行验证。没有以网络错误推断训练中断。
