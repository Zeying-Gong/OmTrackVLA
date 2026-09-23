# e2e_policy ä¿®æ”¹è®°å½•ï¼š3D mRoPE æ›¿æ¢ 2D RoPEï¼ˆ2026-08-15ï¼‰

## ä¸€å¥è¯æ€»ç»“

å°† eVGGT ä¸»å¹²çš„ 2D RoPE æ›¿æ¢ä¸º **Cosmos3/Qwen3VL é£æ ¼çš„ 3D mRoPEï¼ˆtemporal / height / widthï¼‰**ï¼Œ
å¹¶æŠŠä¸»å¹²è¾“å…¥ä»"å•å¸§ patch"å‡çº§ä¸º"**æ—¶é—´å¸§å †å  (T, H, W)**"ï¼Œå†å²å¸§ä¸å½“å‰å¸§ä¸€èµ·è¿›å…¥ backboneã€‚

## å‚è€ƒæ¥æº

å‚è€ƒå®ç°æ¥è‡ªè¿œç«¯ Cosmos3 ä»£ç ï¼ˆæœªæ”¹åŠ¨å…¶åŸæ–‡ä»¶ï¼‰ï¼š

- `/h100-2/cosmos-framework/cosmos_framework/data/generator/sequence_packing/mrope.py`
  - `get_3d_mrope_ids_text_tokens`ã€`get_3d_mrope_ids_vae_tokens`
- `/h100-2/cosmos-framework/cosmos_framework/model/generator/reasoner/qwen3_vl/qwen3_vl.py`
  - `Qwen3VLTextRotaryEmbedding.forward` + `apply_interleaved_mrope`

## æ”¹äº†ä»€ä¹ˆ

### 1. æ–°å¢ `e2e_policy/rope3d.py`ï¼ˆæ–°å»ºï¼Œä¸åŠ¨åŸ rope2d.pyï¼‰

å®Œæ•´ç§»æ¤ Cosmos/Qwen3VL çš„ 3D mRoPEï¼š

| å‡½æ•° | ä½œç”¨ |
| --- | --- |
| `get_3d_mrope_ids_vae_tokens(grid_t, grid_h, grid_w, temporal_offset, reset_spatial_indices=True)` | è§†è§‰ token çš„æœ¬åœ° 3D ç½‘æ ¼ä½ç½® idï¼ŒT-major å±•å¹³ï¼›é»˜è®¤ spatial è½´ä» 0 å¼€å§‹ï¼ˆQwen3VL è®¾è®¡ï¼‰ |
| `get_3d_mrope_ids_text_tokens(num_tokens, temporal_offset)` | text å‹ token ä½ç½® idï¼šä¸‰ä¸ªè½´å…±äº«å•è°ƒé€’å¢ id `(n,n,n)` |
| `build_3d_mrope_cos_sin(position_ids, head_dim, theta=10000.0, mrope_section=None)` | æŠŠ `(3, N)` ä½ç½® id è½¬æˆ cos/sinã€‚é¢‘ç‡å¸ƒå±€ä»åˆ†æ®µ `[TT..HH..WW]` é‡æ’æˆäº¤é”™ `[T H W T H W ...]`ï¼›`emb = cat([freqs, freqs])` ç¿»å€åˆ° head_dimã€‚`mrope_section` é»˜è®¤æŒ‰ head_dim//2 ä¸‰ç­‰åˆ†ï¼ˆhead_dim=96 æ—¶ä¸º `[16,16,16]`ï¼‰ï¼Œä¸‰è€…ä¹‹å’Œå¿…é¡»ç­‰äº head_dim//2 |
| `apply_rotary(q, k, cos, sin)` | Qwen3VL äº¤é”™å¼åº”ç”¨ï¼ˆ`cos`/`sin` åš `::2`/`1::2` äº¤é”™åå† `rotate_half`ï¼‰ |

**çº¦å®šè¯´æ˜ï¼ˆä¸ Cosmos3 ä¸€è‡´ï¼‰ï¼š**
- è§†è§‰ token åæ ‡ï¼šå¸§ `t` å– `temporal_offset + t`ï¼Œé«˜/å®½å–å„è‡ªç½‘æ ¼åæ ‡ã€‚
- ä¸€ä¸ª segment ç»“æŸåï¼Œ`temporal_offset = max(all_positions) + 1`ï¼Œä¸‹ä¸€ä¸ª segmentï¼ˆå¦‚ text/contextï¼‰ä»éé‡å ä½ç½®å¼€å§‹ã€‚
- text å‹ token ä¸‰è½´ id ç›¸åŒ â†’ ç­‰æ•ˆ 1D å•è°ƒä½ç½®ï¼Œç”¨äº context/query ç­‰éè§†è§‰ tokenã€‚

### 2. ä¿®æ”¹ `e2e_policy/evggt.py`ï¼šä¸»å¹²æ”¯æŒæ—¶é—´å¸§å †å 

- `FrameGlobalBackbone.forward(frame_tokens, context_tokens)`ï¼š
  - è¾“å…¥æ”¹ä¸º **`frame_tokens: (B, T, P, dim)`**ï¼ˆT å¸§ Ã— P patchï¼‰ã€‚
  - åºåˆ—å¸ƒå±€ï¼š`[å¸§ patch (T*P)] [context tokens (Cctx)] [policy queries (Q)]`ã€‚
  - **Frame Attention**ï¼šåªå…è®¸åŒå¸§å†…çš„ patch äº’ç›¸ attendï¼ˆ+ å„è‡ª selfï¼‰ï¼Œè·¨å¸§/è·¨ç±»åˆ«è¢« mask æ‰ã€‚
  - **Global Attention**ï¼šå…¨ token å…¨é‡ self-attentionã€‚
  - 3D mRoPEï¼špatch ç”¨ `(t, h, w)` ç½‘æ ¼ä½ç½® idï¼ˆt é€’å¢ = å¸§åºå·ï¼‰ï¼Œcontext/query ç”¨ text å‹å•è°ƒ id æ¥ç»­åœ¨è§†è§‰ grid ä¹‹åã€‚
  - `fused_patches` è¿”å›**æœ€åä¸€å¸§ï¼ˆå½“å‰å¸§ï¼‰**çš„ patch ç‰¹å¾ `seq[:, (T-1)*P : T*P]`ã€‚
- æ–°å¢æ„é€ å‚æ•° `rope_theta=10000.0, mrope_section=None`ã€‚

### 3. ä¿®æ”¹ `e2e_policy/policy.py`ï¼šå–‚æ—¶é—´å¸§å †å 

- `cur_patches` ä» `(B, P, C)` å˜ä¸º `(B, T, P, C)`ï¼š
  - æœ‰ `history_rgb`ï¼š`[å†å²å¸§ (B,K,P,C) + å½“å‰å¸§ (B,1,P,C)]`ï¼Œå½“å‰å¸§åœ¨æœ€åä¸€å¸§ï¼ˆt æœ€å¤§ï¼‰ã€‚
  - æ—  `history_rgb`ï¼š`(B, 1, P, C)` å•å¸§å…œåº•ã€‚
- `history_enc.scene_tokens` ä¸å†è¢«è°ƒç”¨ï¼ˆå†å²å¸§ä»¥åŸå§‹ patch å½¢å¼è¿› backboneï¼Œè€Œé mean-pool è®°å¿† tokenï¼‰ï¼›`motion_tokens` ä»ä¿ç•™è¿› contextã€‚
- `cur_patches_teacher` å–æœ€åä¸€å¸§ `cur_patches[:, -1]`ï¼Œä¸ future teacher `(B,P,C)` å½¢çŠ¶å¯¹é½ï¼ˆå‰å‘åŠ¨åŠ›å­¦ loss ç”¨ï¼‰ã€‚

### æœªæ”¹åŠ¨

- `rope2d.py`ï¼ˆä¿ç•™ï¼Œä¾›å¯¹ç…§/å›é€€ï¼‰
- å…¶ä½™ `encoders.py / heads.py / losses.py / dino_backbone.py / dino_vit.py / smoke_test.py`

## éªŒè¯ç»“æœ

åœ¨è¿œç«¯ H100ï¼ˆ`omtrackvla` ç¯å¢ƒï¼Œcudaï¼‰è·‘é€šï¼š

```
python e2e_policy/smoke_test.py --size 224 --iter 3 --device cuda
```

- trainable params: 13.08Mï¼Œtotal: 35.13M
- loss: 1.6110 â†’ 1.6030 â†’ 1.6801ï¼Œgrad_norm æ­£å¸¸ï¼ŒSMOKE OKã€‚
- ä½ç½® id è‡ªæ£€ï¼š`(T=4, H=W=14)` æ—¶ t è½´ = [0,1,2,3]ï¼Œh/w = [0..13]ï¼›text token ä» 14 èµ·ï¼ˆ=max+1ï¼‰ï¼›cos/sin shape `(789, 96)`ã€‚

## è¿è¡Œå‘½ä»¤

```bash
cd /data/nfs/share/OmTrackVLA
PYTHONPATH=/data/nfs/share/OmTrackVLA /data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python \
  e2e_policy/smoke_test.py --size 224 --iter 3 --device cuda
```

## ä¸‹ä¸€æ­¥ï¼ˆæ•°æ®ç®¡çº¿ï¼‰

- æ–°å»º `e2e_policy/data_collate.py`ï¼šæŠŠ `tools/data_loader/unified.py` çš„ sample dict è½¬æˆæ¨¡å‹ batch
  ï¼ˆcurrent/window â†’ DINO patch æ—¶é—´å †å ã€target cropã€pointgoalã€trajectory å¯¹é½ horizon=8ã€future RGBã€valid/conf maskï¼‰ã€‚
- æ–°å»º `train_e2e.py`ï¼ˆä¸åŠ¨åŸ `train.py`ï¼‰ã€‚
- ç¡®è®¤ `tools/data_loader/self_test.py` ä¸‰æºï¼ˆsage3d / tpt / navdpï¼‰æ­£å¸¸ã€‚

---

## è¿½åŠ ï¼šè¯„å®¡ä¿®å¤ï¼ˆ#2-#9ï¼‰+ æ•°æ®ç®¡çº¿ + ç«¯åˆ°ç«¯è®­ç»ƒï¼ˆ2026-08-15ï¼‰

### è¯„å®¡é—®é¢˜ï¼ˆ10 é¡¹å…¨éƒ¨å±å®ï¼‰

1. **#1 æ— è®­ç»ƒå…¥å£** â†’ æ–°å»º `train_e2e.py`ï¼ˆè§ä¸‹ï¼‰ã€‚
2. **#2 forward åœ¨ç¼º future æ—¶ KeyError** â†’ `policy.py` çš„ forward/inverse ä»…åœ¨ `future_rgb and trajectory` åŒæ—¶å­˜åœ¨æ—¶å¯ç”¨ã€‚
3. **#3 inverse æœªæ¥ç‰¹å¾ç”¨äº† DINO mean-pool+detach** â†’ ä¿ç•™ detachï¼ˆteacherï¼‰ï¼Œä½†æ”¹ä¸ºç» `future_pool`ï¼ˆ2 å±‚ MLPï¼‰æŠ•å½±åçš„ `h_future`ï¼Œshape ä¸ `h_t` å¯¹é½ã€‚
4. **#4 lambda_target æ­»ä»£ç ** â†’ `losses.py` æ–°å¢ `target_loss()`ï¼ˆpos smooth-L1 æŒ‰ mask + vis BCEï¼‰ï¼Œ`compute_total_loss` åœ¨ `"target_state"` å­˜åœ¨æ—¶è®¡ç®— `L_target`ã€‚
5. **#5 é›¶ token é learnable** â†’ `policy.py` æ–°å¢ learnable `null_target` å‚æ•°ï¼ˆtrunc_normal åˆå§‹åŒ–ï¼‰ã€‚
6. **#6 åŠ¨ä½œ mean å‹æ—¶åº** â†’ `heads.py` ForwardDynamics æ”¹ä¸º `act_embed(Linear) + act_gru(GRUCell)` æ—¶åºç¼–ç ï¼Œé€ step çš„ GRU é€’å½’ä¿ç•™ chunk å†…é¡ºåºä¸é—´éš”ã€‚
7. **#7 åˆ†æ”¯è¦†ç›–/æ·±åº¦** â†’ æœªåˆ—åœ¨ 10 é¡¹å†…ï¼Œéšæ•°æ®ç®¡çº¿ä¸€å¹¶å¤„ç†ï¼ˆdepth èµ° monkey-patchï¼Œè§ä¸‹ï¼‰ã€‚
8. **#8 å‡å®šæ–¹ patch ç½‘æ ¼** â†’ `evggt.py` `forward(frame_tokens, context_tokens, grid=None)` ç”¨çœŸå® `(Hp,Wp)`ï¼ˆæ¥è‡ª `dino_backbone.grid_size(h,w)`ï¼‰æ›¿ä»£ `round(P**0.5)`ã€‚
9. **#9 yaw æ— å½’ä¸€åŒ–** â†’ `heads.py` WaypointHead çš„ sin/cos åšéåŸä½ L2 å½’ä¸€åŒ–ï¼ˆ`n=sqrt(sÂ²+cÂ²)`ï¼Œé¿å… autograd ç‰ˆæœ¬å†²çªï¼‰ã€‚
10. **#10ï¼ˆåŸåˆ—è¡¨ç¬¬ 10 é¡¹ï¼‰** â†’ æ¶‰åŠç›®æ ‡è¡¨å¾ï¼Œçº³å…¥æ•°æ®ç®¡çº¿è®¾è®¡ï¼ˆtarget å›¾åƒ + target_local åŒç›‘ç£ï¼‰ã€‚

### æ•°æ®ç®¡çº¿

- **`data_collate.py`ï¼ˆæ–°å»ºï¼‰**ï¼š
  - `waypoints_to_actions(wpts, horizon=8)`ï¼šç»å¯¹ ego èˆªç‚¹ â†’ `(H,4)` å¢é‡åŠ¨ä½œ `(dx,dy,sin,cos)`ï¼Œä¸ `losses._cumulative_se2_error` çš„åæ ‡çº¦å®šä¸€è‡´ï¼ˆæ¯æ­¥å¹³ç§»è½¬åˆ°ç´¯è®¡ heading å‰å¸§ï¼‰ï¼›æ—  heading æ—¶ä»ä½ç§»æ¨ yawï¼›ä¸è¶³ pad æœ«ç‚¹ã€‚
  - `collate_batch(samples, ...)`ï¼šè¾“å‡º `mkw`ï¼ˆcurrent_rgb/target_image/target_valid/target_confidence/target_type/history_rgb/pointgoal/task_type/goal_spec/future_rgbï¼‰ä¸ `loss_batch`ï¼ˆtrajectory/action_valid/action_confidence/target_state/target_state_valid/stop_label/stop_task_maskï¼‰ã€‚
  - future_rgb æ”¯æŒ path å­—ç¬¦ä¸²ä¸ np.ndarray ä¸¤åˆ†æ”¯ã€‚
- **`train_e2e.py`ï¼ˆæ–°å»ºï¼Œä¸åŠ¨åŸ `train.py`ï¼‰**ï¼š
  - `MixedOmniDataset(sage, tpt, navdp)` é‡‡æ ·ï¼Œ`get_sample(as_arrays=True)`ã€‚
  - é»˜è®¤ `use_forward_dyn=False use_inverse_dyn=False`ï¼ˆè¾…åŠ©ä»»åŠ¡é»˜è®¤å…³ï¼‰ï¼Œtarget_state é»˜è®¤å¼€ï¼ˆ`--no-target-state` å…³é—­ï¼‰ã€‚
  - `_patch_unified_2d_depth()` monkey-patchï¼šNavDP depth æ˜¯ 2D `(270,480) uint16`ï¼Œ`unified.resize_keep_aspect` åªæ”¯æŒ 3 é€šé“ï¼›patch åœ¨ 2D åˆ†æ”¯è‡ªè¡Œ letterboxï¼ˆä¸é™¤ 255ï¼‰ï¼Œ3D èµ°åŸå‡½æ•°ã€‚

### éªŒè¯ç»“æœï¼ˆH100ï¼ŒçœŸå®æ•°æ®ï¼‰

```
cd /data/nfs/share/OmTrackVLA
PYTHONPATH=/data/nfs/share/OmTrackVLA /data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python \
  e2e_policy/train_e2e.py --steps 30 --batch 4 --device cuda
```

- æ•°æ®é›†ï¼š`len=13991`ï¼ˆpf_units=13927ï¼Œnavdp_eps=32ï¼‰ã€‚
- 30 steps å…¨ç¨‹è·‘é€šï¼Œloss ä¸‹é™ï¼šstep0 loss=0.919 â†’ step25 loss=0.541ï¼Œ`L_wp` 0.278 â†’ 0.011ã€‚
- **300 steps æ”¶æ•›éªŒè¯**ï¼ˆbatch 8ï¼ŒçœŸå®æ•°æ®ï¼‰ï¼šloss 0.887 â†’ 0.075ï¼ˆstep 290ï¼‰ï¼Œ`L_wp` 0.267 â†’ ~0.01-0.03ï¼›
  è®­ç»ƒæœŸé—´ checkpoint ä¿å­˜äº `e2e_policy/ckpt_e2e.pt`ï¼ˆ133.5 MBï¼‰ã€‚

  | step | loss | L_wp |
  | --- | --- | --- |
  | 0 | 0.887 | 0.267 |
  | 100 | 0.141 | 0.032 |
  | 200 | 0.158 | 0.021 |
  | 290 | 0.075 | 0.026 |
- å¼€å…³éªŒè¯ï¼š
  - é»˜è®¤ï¼ˆtrainable 11.29Mï¼‰ï¼š`L_waypoint + L_target + stop`ã€‚
  - `--no-target-state`ï¼ˆ11.14Mï¼‰ï¼šæ—  `L_target`ã€‚
  - `--use-forward-dyn`ï¼ˆ13.51Mï¼‰ï¼š`L_dino` è¿›å…¥ lossã€‚
  - `--use-inverse-dyn`ï¼ˆ11.75Mï¼‰ï¼š`L_inv` è¿›å…¥ lossã€‚

### æ’æŸ¥è®°å½•

- **depth 2D æŠ¥é”™**ï¼š`cv2.resize` ä¼šæŠŠ `(270,480,1)` å‹å› 2Dï¼Œä¸”åŸå‡½æ•°æœ«å°¾ `/255.0` å¯¹ depth ä¸æ­£ç¡®ï¼›`_rk` æ”¹ä¸º 2D åˆ†æ”¯å®Œæ•´è‡ªåš letterboxã€ä¸é™¤ 255ã€‚
- **TpT `traj=None`**ï¼š`waypoints_to_actions` åŠ  `if not wpts: return None`ï¼Œæ— è½¨è¿¹æ ·æœ¬ action_valid=0ã€‚
- **forward_dyn ä¸è§¦å‘**ï¼š`train_e2e.py` è¡¥ `mkw["trajectory"] = batch["trajectory"]`ï¼Œå¦åˆ™ `policy.forward` ç¼º trajectory åˆ†æ”¯ç›´æ¥è·³è¿‡ã€‚

---

## è¿½åŠ ï¼šç¬¬äºŒè½®è¯„å®¡ä¿®å¤ï¼ˆ2026-08-15ï¼‰

### ä¸¥é‡é—®é¢˜

1. **Sage3D ç¬¬ 8 æ­¥è¢«é”™è¯¯ç›‘ç£ä¸ºåœæ­¢**
   - äº‹å®ï¼š`waypoints_ego` æ˜¯ 8 ç‚¹ä¸”**åŒ…å«åŸç‚¹**ï¼ˆ`README.md:100`ï¼‰ï¼Œåªèƒ½äº§ç”Ÿ 7 ä¸ªä½ç§»ï¼›æ—§ collate é‡å¤æœ«ç‚¹åæŠŠç¬¬ 8 æ­¥ action_valid ç½® 1ï¼Œç³»ç»Ÿå­¦ä¹ "æœ€åä¸€æ­¥åœæ­¢"ã€‚
   - ä¿®å¤ï¼š`waypoints_to_actions` è¿”å› `(acts, n_valid)`ï¼Œ`n_valid = min(M-1, horizon)`ï¼ˆçœŸå®ä½ç§»æ•°ï¼‰ï¼›collate ä»… `a_valid[i, :n_valid]=1`ã€‚éªŒè¯ï¼šSage3D æ ·æœ¬ n_valid=7ï¼Œç¬¬ 8 æ­¥é›¶åŠ¨ä½œä¸å†è¢«ç›‘ç£ã€‚

2. **è¡Œäºº target crop ä¸æ˜¯å›ºå®šèº«ä»½å‚è€ƒ**
   - äº‹å®ï¼šSage3D/TpT æ¯å¸§ä»å½“å‰å¸§ bbox é‡è£ targetï¼Œç›®æ ‡ä¸å¯è§æ—¶æ—  crop â†’ æ— æ³•è®­ç»ƒèº«ä»½ä¿æŒ/é®æŒ¡æ¢å¤/å¤šäºº ReIDã€‚
   - ä¿®å¤ï¼š`sage3d_ds._reference_target` / `tpt_ds._reference_target` æ¯ä¸ª episode/è§†é¢‘ç¼“å­˜**é¦–ä¸ªå¯è§å¸§**çš„ bbox crop åˆ° `<ep_dir>/_target_refs/ref.jpg`ï¼Œæ‰€æœ‰ step å…±ç”¨åŒä¸€ referenceï¼›å½“å‰ bbox ä»…é€šè¿‡ `bbox`/`target_local` ä½œç›‘ç£æ ‡ç­¾ã€‚

3. **å¯å­¦ä¹  null token åŸºæœ¬æœªä½¿ç”¨**
   - äº‹å®ï¼šcollate å¯¹ç¼ºç›®æ ‡æ ·æœ¬å¡«é»‘å›¾å¹¶æ•´ä½“ä¼ å…¥ `target_image`ï¼Œpolicy åªåœ¨ `target_image is None`ï¼ˆæ•´æ‰¹ï¼‰æ‰ç”¨ nullï¼Œæ··åˆ batch ä¸­ç¼ºç›®æ ‡æ ·æœ¬å®é™…èµ°é»‘å›¾ DINO+TargetEncoderã€‚
   - ä¿®å¤ï¼špolicy åœ¨ `target_valid` å­˜åœ¨æ—¶é€æ ·æœ¬æ··åˆ `target_tokens = where(target_valid>0.5, encoded, null)`ï¼›`null_target` æ”¹ä¸º `(1, n_identity_tokens, dim)`ã€‚

4. **inverse dynamics æ ¸å¿ƒé—®é¢˜**
   - äº‹å®ï¼šfuture RGB â†’ frozen DINO â†’ mean pool â†’ `future_pool` â†’ detachï¼Œæœªç”¨å…±äº« backboneã€‚
   - ä¿®å¤ï¼š`h_future` ç”±**åŒä¸€ FrameGlobalBackbone** å‰å‘æœªæ¥å¸§å¾—åˆ° `h_{t+H}`ï¼ˆ`fut_patches[:, None]`ï¼‰ï¼Œä»…å–‚ç»™ inverse æ—¶ `h_future.detach()`ï¼›åˆ é™¤ `future_pool`ã€‚

### å…¶ä»–é‡è¦é—®é¢˜

- **`success or 1.0` bug**ï¼š`success=0` è¢« `or` è½¬æˆ 1.0ï¼ˆå¤±è´¥ episode æ»¡ç½®ä¿¡åº¦ï¼‰ï¼›æ”¹ä¸º `conf = float(succ) if succ is not None else 1.0`ã€‚
- **history-valid mask**ï¼š`_history_frames` è¿”å› `(stack, mask)`ï¼Œrepeat-last å¡«å……å¸§ mask=0ï¼›collate è¾“å‡º `history_valid`ï¼›`evggt.FrameGlobalBackbone` æ¥å— `frame_valid`ï¼ŒGlobal Attention å¯¹æ— æ•ˆå¸§æ„é€  per-batch maskï¼ˆä»…å…è®¸è‡ª attendï¼‰ï¼ŒFrame Attention ä¿æŒåŒå¸§è¯­ä¹‰ã€‚
- **future-valid mask**ï¼šcollate è¾“å‡º `future_valid`ï¼›`forward_loss` çš„ dino/depth/free/target_state ä¸ `a_inv` çš„ waypoint loss å‡æŒ‰ `future_valid` æ©ç ã€‚

### éªŒè¯ç»“æœï¼ˆv2ï¼‰

```
python e2e_policy/smoke_test.py --size 224 --history 3 --iter 2 --device cuda   # SMOKE OK
python e2e_policy/train_e2e.py --steps 300 --batch 8 --device cuda             # æ”¶æ•›
```

- smokeï¼štrainable 13.67Mï¼ˆnull_target=4 tokensï¼Œinverse èµ°å…±äº« backboneï¼‰ï¼Œ`h_future (2,384)` æ­£å¸¸è¾“å‡ºã€‚
- çœŸå®è®­ç»ƒï¼šloss 0.89 â†’ 0.05ï¼ˆstep 250ï¼‰ï¼ŒL_wp 0.27 â†’ 0.01ã€‚
- Sage3D åè®®éªŒè¯ï¼š`n_valid=7`ï¼Œreference crop ç”Ÿæˆäº `_target_refs/ref.jpg`ã€‚

---

## ×·¼Ó£ºµÚÈıÂÖÆÀÉóĞŞ¸´£¨Êı¾İÓïÒå£¬2026-08-15£©

Ğ­×÷Õß·¢ÏÖ 9 ¸öÊı¾İÓïÒåÎÊÌâ£¬È«²¿È·ÈÏÊôÊµ²¢ĞŞ¸´¡£

### ÑÏÖØÎÊÌâ

1. **NavDP µ±Ç°Ö¡Óë¹ì¼£²Î¿¼Ïµ´íÎ»**
   - ÊÂÊµ£ºµ±Ç°Í¼ÓÃ gb[mem]£¬µ« local ¹ì¼£ÒÔ extrinsics[start] ÎªÔ­µã£¨start<mem<target£©¡£Êµ²â start=0/mem=67/target=68 Ê± pointgoal¡Ö0.97m£¬Ä¿±ê½öÍí 1 ²½¡£
   - ĞŞ¸´£ºlocal ¹ì¼£¸ÄÒÔ extrinsics[mem]£¨µ±Ç°Ö¡£©ÎªÔ­µã£¬Óë InternNav memory_start_choice Ò»ÖÂ¡£ÑéÖ¤£º	raj[0]¡Ö[0,0,0]£¬goal_dist ÖĞÎ» 0.36m¡£

2. **NavDP ºöÂÔ base_extrinsic Ğı×ª**
   - ÊÂÊµ£ºº¯Êı½ÓÊÕ ase_extrinsic µ«´ÓÎ´Ê¹ÓÃ£»D435i base_ext ·Çµ¥Î»Ğı×ª¡£
   - ĞŞ¸´£º_relative_pose_point ¼Ó R_base = R_base @ inv(base_extrinsic[:3,:3])£¬Í¬ InternNav navdp_lerobot_dataset.py:268¡£

3. **PointNav/ImageGoal ÌØÈ¨Ä£Ì¬Ğ¹Â©**
   - ÊÂÊµ£ºÁ½ÈÎÎñ¶¼·µ»Ø target_img + pointgoal£¬collate Í¬Ê±ËÍÄ£ĞÍ¡£
   - ĞŞ¸´£º°´ÈÎÎñÆÁ±Î¡ª¡ªpointnav Ö»¸ø pointgoal£¨target_img=None£©£¬imagegoal Ö»¸ø goal image£¨pointgoal=None£©¡£goal_dist ÈÔ×÷Îª¼à¶½±êÇ©½ø extra£¨±êÇ©·ÇÊäÈë£©¡£

4. **valid »ìÏı"Ä¿±ê¿É¼û"Óë"¹ì¼£ÓĞĞ§"**
   - ÊÂÊµ£ºSage3D alid=visible ±» collate ÓÃÀ´ gate waypoint£¬Ä¿±êÕÚµ²¼´²»¼ÆËã waypoint loss£¬¶ªµôÕÚµ²»Ö¸´Ñù±¾¡£
   - ĞŞ¸´£ºmake_sample_dict ²ğ 	arget_visible / 	rajectory_valid / uture_valid Èı×Ö¶Î£»collate waypoint ¼à¶½¸Ä°´ 	rajectory_valid£¬target-state °´ 	arget_visible¡£ÑéÖ¤£º	rajectory_valid ÓÉ traj ´æÔÚ¾ö¶¨¡£

### ÖØÒªÎÊÌâ

5. **TpT ÀúÊ·Ö¡Ë³Ğò·´ÁË**£ºÑ­»· i=history..1 ºóÔÙ¹ıÂË£¬×ÔÈ»µÃµ½¾É¡úĞÂ£»ÑéÖ¤ [18,22,...,46] ÉıĞò¡£
6. **NavDP ¶şÎ¬ depth ÎŞ·¨×ßÍ³Ò» loader**£ºesize_keep_aspect Ô­ÉúÖ§³Ö 2D Êı×é¡¢depth ²» /255£»É¾³ı 	rain_e2e.py µÄ monkey patch¡£ÑéÖ¤ (224,224,1) Õı³£¡£
7. **¿É¼ûĞÔ±ß½ç´íÎó**£º <u<1e6 ¸ÄÎªÕæÊµÍ¼Ïñ¿í¸ß  <=u<W and 0<=v<H£¨_img_shape »º´æ per-episode£©¡£
8. **Sage3D future RGB Óë horizon Î´¶ÔÆë**£ºÔ­¹Ì¶¨ k+3£¬¸Ä k + nh*step_stride£¨waypoint_horizon ÖÕµã£©¡£ÑéÖ¤ k=3¡úframe 11¡£
9. **»ìºÏ²ÉÑù waypoint ¼à¶½Õ¼±ÈÏ¡ÊÍ**£ºÔ­°´ĞĞÊı»ìºÏ£¬TpT(~141k) ÑÍÃ» Sage3D(~38k)£¬waypoint Õ¼±È ~10%¡£¸ÄÎª°´ (ÈÎÎñ, ¼à¶½ÀàĞÍ) ËÄ×é·Ö²ã (sage-wp, tpt-nowp, pointnav, imagegoal) = (1, 0.5, 1, 1)¡£ÑéÖ¤ person_follow+wp Õ¼±È 23%¡£

### ÑéÖ¤½á¹û£¨v3£©

`
python verify_fix2.py    # 9 ÏîÈ«²¿ OK
python e2e_policy/train_e2e.py --steps 300 --batch 8 --device cuda
`

- verify_fix2£º2D depth¡¢NavDP ²Î¿¼Ïµ/Ğı×ª/¿É¼ûĞÔ¡¢Ä£Ì¬¸ôÀë¡¢valid ²ğ·Ö¡¢TpT Ë³Ğò¡¢future horizon¡¢·Ö²ã²ÉÑù¡¢load_sample_arrays(depth) È«²¿Í¨¹ı¡£
- ÕæÊµÑµÁ·£º300-step ÊÕÁ²£¬loss ~0.12@step295£¬ÎŞ±ÀÀ£¡£

---

## ×·¼Ó£ºµÚÈıÂÖÆÀÉóĞŞ¸´£¨±êÇ©Óë mask£¬2026-08-15£©

### Ö÷ÒªÎÊÌâ£¨5 Ïî£¬¾ùÑéÖ¤Í¨¹ı£©

1. **Sage3D ¶¯×÷±êÇ©²»º¬Ô­µã**
   - ÊÂÊµ£ºextract_sage3d.build_waypoints Ê×µãÈ¡ s+1£¨Î´À´Ö¡£©£¬8 µãÈ«ÎªÎ´À´Î»ÖÃ£¬²»º¬Ô­µã£»collate È´¼Ù¶¨Ê×µãÊÇÔ­µã ¡ú Ê×¶Î¶¯×÷±»¶ªÆú¡¢Ö»Éú³É 7 ¸öÓĞĞ§¶¯×÷¡£
   - ĞŞ¸´£ºwaypoints_to_actions ¼ì²âÊ×µã hypot>1e-6 Ê±ÏÔÊ½ prepend [0,0]£¬¼à¶½ 8 ¶Î¶¯×÷¡£ÑéÖ¤£º40/40 Ñù±¾ n_valid=8£¬Ê×¶¯×÷·ÇÁã¡£

2. **Sage3D future frame Î´¶ÔÆë¶¯×÷ÖÕµã**
   - ÊÂÊµ£ºwaypoint ÓÃ waypoint_stride=3£¨µÚ 8 µã¡Ök+22£©£¬future_rgb È´È¡ k+8£¨ÔçÔ¼ 14 Ö¡£©¡£
   - ĞŞ¸´£ºuture_rgb = rgb[k + 1 + (nh-1)*waypoint_stride]£¨k+22£©¡£ÑéÖ¤£ºk=3¡úframe 25¡£

3. **future_valid Î´½øÈë loss**
   - ÊÂÊµ£ºcollate Ö»°Ñ future_valid ·ÅÈë model_kwargs£¬loss_batch Ã»ÓĞËü ¡ú losses.py ºã¶Á None£¬È±Î´À´Ö¡Ñù±¾ÈÔÓÃºÚÍ¼Ëã loss¡£
   - ĞŞ¸´£ºuture_valid ·ÅÈë loss_batch£»losses.py µÄ uture_valid[:, :1] Ë÷Òı¸ÄÎª eshape(-1,1)£¨ĞÎ×´ (B,)£©¡£

4. **NavDP ÖÕµã¶ªÊ§ + ´íÎó stop ±êÇ©**
   - ÊÂÊµ£º_xyz_to_xyt Ö»¼ÓÈë xyz[i]£¬Â©µô×îºóÒ»¸öÎ»ÖÃ£»	arget=mem+1 Ê± xyt Ö»Ê£Ô­µã£¬PointGoal=[0,0,0]¡¢stop Îó±ê 1£¨200 Ñù±¾ÖĞ 13 ¸ö£©¡£
   - ĞŞ¸´£º_xyz_to_xyt Êä³ö M µã£¨º¬ÖÕµã£©£¬heading Ä©µã hold£»collate 
_valid ¸ÄÎª°´"·ÇÁãÎ»ÒÆ"¼ÆÊı£¬¶Ì¹ì¼£Î²²¿µÄÁã/ÖØ¸´¶¯×÷²»ÔÙ¼à¶½¡£ÑéÖ¤£º200 Ñù±¾ 10 ¸ö¶Ì¹ì¼££¬0 ¸öÁã pointgoal£»n_valid ·Ö²¼ {1:32,...,8:74}¡£

5. **NavDP ÀúÊ·ÖØ¸´°üº¬µ±Ç°Ö¡**
   - ÊÂÊµ£ºwindow Ä©Î²¼´ mem£¬policy ÓÖ append µ±Ç°Ö¡ ¡ú Í¬Ò»Í¼³öÏÖÔÚÁ½¸ö 3D-mRoPE Ê±¼ä×ø±ê¡£
   - ĞŞ¸´£ºmem_idx ÉÏ½ç¸ÄÎª mem£¨²»º¬£©£¬window ÑÏ¸ñÔÚµ±Ç°Ö¡Ö®Ç°¡£ÑéÖ¤£º100/100 ÎŞÖØ¸´¡£

### ĞÂÔö·¢ÏÖ£¨2 Ïî£¬¾ùĞŞ¸´£©

- **stop_loss È« person-follow Ê±´íÎó»ØÍËÈ«Åú BCE**£º	ask_mask.sum()==0 Ê±·µ»Ø zeros ËğÊ§¡£
- **¸¨Öú·ÖÖ§Î´»ñµÃÕæÊµ¼à¶½**£ºhistory_motion ¾­ collate ´«Èë policy£¨HistoryEncoder.motion_dim=2 ÊÊÅä TpT 2D motion£©£»uture_depth ´Ó NavDP loader ¡ú load_sample_arrays ¡ú collate Éú³É depth_residual£¨Î´À´-µ±Ç° depth£¬16¡Á16 grid£©Óë ree_target£¨free-space mask£©£¬°´ depth_valid ÑÚÂë£¬forward-loss µÄ L_depth/L_free/L_grad ÏÖÔÚÓĞÕæÊµ¼à¶½¡£

### ÑéÖ¤½á¹û£¨v3£©

`
python verify_fix3.py    # P1-P5 + stop_loss + history_motion È«²¿ OK
python e2e_policy/train_e2e.py --steps 300 --batch 8 --use-forward-dyn --use-inverse-dyn --device cuda
`

- verify_fix3£ºSage3D 8 ¶¯×÷¡¢future ¶ÔÆë¡¢future_valid in loss_batch¡¢NavDP ÖÕµã²»¶ª/¶Ì¹ì¼£ mask¡¢window ÎŞÖØ¸´¡¢stop_loss Áã¡¢history_motion È«Í¨¹ı¡£
- ÑµÁ·£¨forward/inverse dyn + depth/free£©£ºL_wp ÊÕÁ²ÖÁ ~0.01-0.03£¬L_depth/L_free/L_grad Õı³£ÏÂ½µ£¬ÎŞ±ÀÀ£¡£
