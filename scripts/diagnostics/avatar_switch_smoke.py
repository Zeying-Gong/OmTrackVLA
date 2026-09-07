#!/usr/bin/env python3

import random

import habitat
import numpy as np
import evt_bench  # noqa: F401 - registers benchmark actions and sensors
from habitat.datasets import make_dataset


def main() -> None:
    config = habitat.get_config(
        "habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml",
        [
            "habitat.simulator.habitat_sim_v0.gpu_device_id=0",
            "habitat.simulator.scene_dataset=data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
        ],
    )
    random.seed(config.habitat.simulator.seed)
    np.random.seed(config.habitat.simulator.seed)
    dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    wanted = ["246", "32"]
    by_id = {
        str(ep.episode_id): ep
        for ep in dataset.episodes
        if ep.scene_id.split("/")[-2] == "2n8kARJN3HM"
        and str(ep.episode_id) in wanted
    }
    dataset.episodes = [by_id[episode_id] for episode_id in wanted]

    with habitat.TrackEnv(config=config, dataset=dataset) as env:
        seen = set()
        for _ in wanted:
            env.reset()
            episode = env.current_episode
            episode_id = str(episode.episode_id)
            assert episode_id in wanted
            seen.add(episode_id)
            expected_semantic_id = int(episode.info["main_human_semantic_id"])
            panoptic = env.sim.get_sensor_observations()[
                "agent_1_articulated_agent_jaw_panoptic"
            ]
            semantic_pixels = int(np.count_nonzero(panoptic == expected_semantic_id))
            humanoid = env.sim.agents_mgr[0].articulated_agent
            rendered_human_ids = np.unique(panoptic[(panoptic >= 1000) & (panoptic <= 1200)])
            node_semantic_ids = sorted(
                {int(node.semantic_id) for node in humanoid.sim_obj.visual_scene_nodes}
            )
            sensor = env.task.sensor_suite.sensors[
                "agent_1_main_humanoid_detector_sensor"
            ]
            assert sensor._human_id == expected_semantic_id
            # The robot camera does not necessarily see the target at reset.
            # When it does, it must use the current episode's semantic ID.
            if rendered_human_ids.size:
                assert expected_semantic_id in rendered_human_ids, (
                    episode_id,
                    episode.info["main_humanoid_name"],
                    expected_semantic_id,
                    humanoid.sim_obj.object_id,
                    node_semantic_ids,
                    rendered_human_ids,
                )
            print(
                "avatar-switch-smoke",
                f"episode={episode_id}",
                f"avatar={episode.info['main_humanoid_name']}",
                f"semantic_id={expected_semantic_id}",
                f"pixels={semantic_pixels}",
                f"object_id={humanoid.sim_obj.object_id}",
                f"node_ids={node_semantic_ids}",
                f"rendered_human_ids={rendered_human_ids.tolist()}",
            )
        assert seen == set(wanted), seen


if __name__ == "__main__":
    main()
