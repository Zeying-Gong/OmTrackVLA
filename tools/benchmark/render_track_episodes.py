#!/usr/bin/env python3
"""Render three fixed EVT-Bench validation episodes on a GPU Habitat worker."""
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw
REPO = Path(__file__).resolve().parents[2]
HABITAT_LAB = REPO / "habitat-lab"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HABITAT_LAB))
import evt_bench
import habitat
from habitat.config.default import get_config
from habitat.datasets import make_dataset
OUTPUT = REPO / "artifacts" / "episode_visualizations_20260918"
SPECS = [("STT", 0, "hm3d", "track_infer_stt.yaml"), ("DT", 684, "mp3d", "track_infer_dt.yaml"), ("AT", 6, "hm3d", "track_infer_at.yaml")]
def as_rgb(value):
    arr = np.asarray(value)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4) or arr.dtype != np.uint8:
        return None
    return Image.fromarray(arr[..., :3], mode="RGB")
def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for task, index, family, config_name in SPECS:
        os.chdir(HABITAT_LAB)
        scene_file = "hm3d_annotated_basis.scene_dataset_config.json" if family == "hm3d" else "mp3d.scene_dataset_config.json"
        overrides = [f"habitat.simulator.scene_dataset=data/scene_datasets/{family}/{scene_file}"]
        if os.getenv("OMTRACKVLA_CPU_RENDER") == "1":
            overrides.append("habitat.simulator.habitat_sim_v0.gpu_device_id=-1")
        cfg = get_config(f"habitat/config/benchmark/nav/track/{config_name}", overrides=overrides)
        os.chdir(REPO)
        dataset = make_dataset(id_dataset=cfg.habitat.dataset.type, config=cfg.habitat.dataset)
        episode = dataset.episodes[index]
        dataset.episodes = [episode]
        env = habitat.Env(config=cfg, dataset=dataset)
        try:
            obs = env.reset()
            images = []
            for key, value in obs.items():
                image = as_rgb(value)
                if image is None:
                    continue
                image.thumbnail((512, 384))
                labeled = Image.new("RGB", (image.width, image.height + 28), "white")
                labeled.paste(image, (0, 28))
                ImageDraw.Draw(labeled).text((6, 6), key, fill="black")
                images.append(labeled)
            if not images:
                raise RuntimeError(f"No uint8 RGB observations for {task}: {list(obs)}")
            width, height = max(i.width for i in images), sum(i.height for i in images)
            sheet = Image.new("RGB", (width, height), "white")
            offset = 0
            for image in images:
                sheet.paste(image, (0, offset))
                offset += image.height
            image_path = OUTPUT / f"{task.lower()}_{family}_episode_{episode.episode_id}.png"
            sheet.save(image_path)
            record = {"task": task, "episode_index_in_val": index, "episode_id": episode.episode_id, "scene_id": episode.scene_id, "scene_family": family, "human_num": episode.info.get("human_num"), "main_humanoid_name": episode.info.get("main_humanoid_name"), "extra_humanoid_names": episode.info.get("extra_humanoid_names"), "instruction": episode.info.get("instruction"), "rgb_observation_keys": [k for k, v in obs.items() if as_rgb(v) is not None], "image": str(image_path)}
            manifest.append(record)
            print("RENDER_OK", json.dumps(record, sort_keys=True))
        finally:
            env.close()
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("ALL_RENDER_OK", OUTPUT)
if __name__ == "__main__":
    main()
