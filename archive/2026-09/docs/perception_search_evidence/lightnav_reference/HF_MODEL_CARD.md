---
license: apache-2.0
library_name: transformers
pipeline_tag: robotics
base_model: Qwen/Qwen3-VL-4B-Instruct
tags:
  - embodied-navigation
  - vision-language-navigation
  - visual-tracking
  - vision-language-action
  - qwen3-vl
  - robotics
---

# LightNav-0

**LightNav-0: Eliciting VLM Spatial Intelligence for Generalist Embodied Navigation**

LightNav-0 is a compact generalist embodied navigation model that elicits the spatial
intelligence of a pretrained vision-language model (Qwen3-VL) and aligns it with navigation,
without task-specific prediction heads. Diverse tasks share one token interface: dual-channel
pointing expresses task-, scene- and embodiment-agnostic spatial intent, and a residual
vector-quantized (RVQ) action tokenizer maps that intent to precise, embodiment-specific
trajectories — so instruction following, open-vocabulary object navigation and visual
tracking live in a single model.

Inference, serving and evaluation code: **[LightNav-0 repository](https://github.com/lightrobo/LightNav-0)**.

## Files

| Path | Content |
|---|---|
| `config.json`, `model-*.safetensors`, `model.safetensors.index.json` | Qwen3-VL weights (bf16) with the navigation token embeddings |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `processor_config.json` | tokenizer / processor |
| `eval_config.json` | processing parameters that must match training (video size, pooling, SlowFast history tiers, task settings); read automatically by the inference code |
| `action_tokenizer/` | RVQ action-tokenizer bundle shared by navigation and tracking (one 3 × 256 residual codebook, horizon 10) |

The bundle holds `manifest.json`, the per-level codebooks (`codebook_l{0,1,2}.npy`, 3 × 256 codes),
`jacobian_weights.npy` and `alpha_per_source.json`. The model emits one `<act_l{level}_{code}>`
token per level; the bundle decodes them into a `(10, 3)` chunk of robot-local waypoints
`[forward_m, lateral_m (+left), yaw_rad (+ccw)]`. `eval_config.json` references the bundle by a
path relative to this directory, so the inference code finds them without extra flags.

## Usage

```bash
pip install "lightnav[vllm,video] @ git+https://github.com/lightrobo/LightNav-0"
hf download LightOriginsHQ/LightNav-0 --local-dir ./LightNav-0-ckpt

# offline prediction on a clip
lightnav-predict --model_path ./LightNav-0-ckpt --backend vllm_local \
    --video clip.mp4 --fps 4 --instruction "follow the person in the red shirt"

# serve (tracking prompt) and drive with the reference client
lightnav-serve --task tracking --model_path ./LightNav-0-ckpt --backend vllm_local --port 8050
lightnav-ws-client --server ws://localhost:8050 --video clip.mp4 --fps 4 \
    --instruction "follow the person in the red shirt"

# navigation prompt (instruction following / object navigation)
lightnav-serve --task vln --model_path ./LightNav-0-ckpt --backend vllm_local --port 8051
```

Habitat VLN-CE / ObjectNav and EVT-Bench evaluation recipes, the WebSocket protocol and the
real-robot deployment guide are in the code repository's `docs/`.

## Model details

- Backbone: Qwen3-VL (4B), bf16; the trajectory / pointing tokens are ordinary rows of the
  embedding table, so the checkpoint loads with stock `transformers`.
- Input: a SlowFast-compressed history of first-person RGB frames (model input 256×448, 4 fps)
  plus a natural-language instruction.
- Output: dual-channel pointing tokens (`<apos_*>` / `<opos_*>`) followed by three RVQ action
  tokens (`<act_l0_*><act_l1_*><act_l2_*>`) decoded to a 10-step waypoint chunk.

## Community

Questions, deployment notes and release news — join us on
[Discord](https://discord.gg/zwZuD9JG), or scan to join the WeChat group:

<div align="center">
  <img src="wechat_group.png" alt="WeChat QR code for the LightOrigins discussion group" width="280"/>
</div>

## License

Apache License 2.0.
